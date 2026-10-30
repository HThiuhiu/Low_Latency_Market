"""DeepSeek as a feature extractor: advice -> scores + reason embedding -> features -> model.

    python -m research.llm_features collect --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 30 --every 60
    python -m research.llm_features evaluate --symbols BTCUSDT,ETHUSDT,SOLUSDT --days 30 --every 60

collect : every ``--every`` minutes per coin, send a causal market snapshot (no dates, no
          future data) to the LLM; it returns numeric scores + a short reason. The reason is
          embedded (text-embedding-3-small via the same OpenAI-compatible endpoint).
          Everything is cached in research/llm_cache/ -> re-runs cost nothing.
evaluate: XGBoost on [base features] vs [base + LLM scores] vs [base + LLM scores + PCA(embedding)],
          chronological split (train first 70% of time, test the rest), all coins pooled.
          Reports IC, hit rate and a simple net-of-cost strategy on the test part.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np

from qforecast.agent.llm import LLMClient, LLMError
from qforecast.config import Settings
from qforecast.core.features import B_CLOSE, B_HIGH, B_LOW, B_OPEN, FEATURE_NAMES, compute_features
from qforecast.exchange.binance import raw_to_bars
from qforecast.model.predictor import ModelMeta, make_session
from qforecast.sim.backtest import forecast_all, renormalise
from qforecast.trainer.data import load_history
from research.improve import build_table, extra_features

log = logging.getLogger("llmfeat")
CACHE = Path("research/llm_cache")
DAY = 86_400_000
COST_BPS = 12.0  # round trip: 2 x (5 fee + 1 slippage)

FEATURE_PROMPT = """You are a quantitative crypto market analyst. You get a causal snapshot of one
coin on 1-minute bars (only information available now) and the output of a weak neural
forecaster. Judge the NEXT 60 MINUTES.

Reply with ONE JSON object only:
{"bias": -1.0..1.0,               // direction view: -1 strongly down, 0 none, +1 strongly up
 "confidence": 0.0..1.0,          // how sure you are (0.5 = coin flip)
 "expected_move_bps": number,     // signed expected return over the next 60 min, in bps
 "reversal_risk": 0.0..1.0,       // chance the recent move reverses
 "regime": "TREND_UP" | "TREND_DOWN" | "RANGE" | "VOLATILE",
 "action": "LONG" | "SHORT" | "FLAT",
 "reason": "max 25 words"}"""

REGIMES = ("TREND_UP", "TREND_DOWN", "RANGE", "VOLATILE")
ACTIONS = ("LONG", "SHORT", "FLAT")
SCORE_NAMES = ["llm_bias", "llm_conf", "llm_bias_x_conf", "llm_exp_bps", "llm_reversal",
               *[f"llm_regime_{r.lower()}" for r in REGIMES], *[f"llm_act_{a.lower()}" for a in ACTIONS]]


# --------------------------------------------------------------------------- data
async def prepare(s: Settings, days: int, every: int, renorm_days: int = 14,
                  times: list[int] | None = None) -> dict:
    """Bars, features, model forecasts and the sample indices for one coin.

    Samples sit on a clock-aligned grid (every ``every`` minutes), or exactly at ``times``
    (bar open times, e.g. from the cache) so collect and evaluate always match."""
    meta = ModelMeta.load(s.meta_path)
    now = int(time.time() * 1000)
    if times:
        days = max(days, int((now - min(times)) / DAY) + 1)
    raw = await load_history(s, days + renorm_days + 2)
    open_time, bars = raw_to_bars(raw)
    feats, valid = compute_features(bars)
    X = build_table(feats, extra_features(open_time, bars))
    if times:
        idx = np.searchsorted(open_time, np.sort(np.asarray(times, np.int64)))
        idx = idx[(idx < len(bars)) & (open_time[np.minimum(idx, len(bars) - 1)] == np.sort(times))]
    else:
        start = open_time[-1] - days * DAY
        idx = np.flatnonzero((open_time >= start) & (open_time % (every * 60_000) == 0))
    idx = idx[(idx < len(bars) - 61) & valid[idx] & ~np.isnan(X[idx]).any(axis=1)]  # room for label
    if json.loads(Path(s.meta_path).read_text()).get("symbol") != s.symbol and len(idx):
        meta = renormalise(meta, feats, valid, int(idx[0]), renorm_days * 1440)
    pred = await asyncio.to_thread(forecast_all, meta, make_session(s.model_path), feats, idx)
    return {"s": s, "meta": meta, "open_time": open_time, "bars": bars, "feats": feats, "X": X,
            "idx": idx, "pred": pred}


def snapshot(d: dict, k: int) -> dict:
    """Causal context for sample k. No timestamps (limits recall of known prices)."""
    i, bars, f, meta = d["idx"][k], d["bars"], d["feats"][d["idx"][k]], d["meta"]
    c = bars[:i + 1, B_CLOSE]
    px = c[-1]

    def ret(m):
        return round(float(np.log(px / c[-1 - m]) * 1e4), 1)

    def vol(m):
        return round(float(np.diff(np.log(c[-m - 1:])).std() * 1e4), 2)

    hi, lo = bars[i - 1439:i + 1, B_HIGH].max(), bars[i - 1439:i + 1, B_LOW].min()
    named = dict(zip(FEATURE_NAMES, f, strict=True))
    return {
        "symbol": d["s"].symbol, "price": float(px),
        "return_bps": {"5m": ret(5), "15m": ret(15), "1h": ret(60), "4h": ret(240), "24h": ret(1440)},
        "bar_volatility_bps": {"1h": vol(60), "24h": vol(1440)},
        "pos_in_24h_range": round(float((px - lo) / (hi - lo)), 2) if hi > lo else 0.5,
        "indicators": {"rsi14_scaled": round(named["rsi_14"], 3), "bollinger_z20": round(named["bb_z_20"], 2),
                       "dist_ma50_bps": round(named["dist_ma_50"] * 1e4, 1),
                       "taker_buy_imbalance": round(named["taker_imbalance"], 3),
                       "rel_volume_log": round(named["rel_volume"], 2)},
        "nn_forecast_bps": {str(h): round(float(e) * 1e4, 1)
                            for h, e in zip(meta.horizons, d["pred"][k], strict=True)},
        "round_trip_cost_bps": COST_BPS,
    }


# --------------------------------------------------------------------------- collect
def cache_file(sym: str) -> Path:
    return CACHE / f"{sym}.jsonl"


def load_cache(sym: str) -> dict[int, dict]:
    p = cache_file(sym)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        out[r["t"]] = r
    return out


async def embed(texts: list[str], model: str) -> list[list[float]]:
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{base}/embeddings", json={"model": model, "input": texts},
                         headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
    r.raise_for_status()
    return [e["embedding"] for e in sorted(r.json()["data"], key=lambda e: e["index"])]


async def collect(args) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    llm = LLMClient(model=args.model, timeout_s=60)
    if not llm.configured:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    sem = asyncio.Semaphore(args.concurrency)
    base = Settings.from_env()
    t0 = time.perf_counter()
    done = 0

    async def one(sym: str) -> None:
        nonlocal done
        d = await prepare(replace(base, symbol=sym), args.days, args.every)
        cached = load_cache(sym)
        todo = [k for k in range(len(d["idx"])) if int(d["open_time"][d["idx"][k]]) not in cached]
        log.info("%s: %d samples, %d cached, %d to ask", sym, len(d["idx"]),
                 len(d["idx"]) - len(todo), len(todo))
        lock = asyncio.Lock()

        async def ask(k: int) -> None:
            nonlocal done
            ctx = snapshot(d, k)
            async with sem:
                try:
                    adv, _ = await llm.ask_json(FEATURE_PROMPT, json.dumps(ctx, separators=(",", ":")), 300)
                except LLMError as e:
                    log.warning("%s: %s", sym, e)
                    return
            rec = {"t": int(d["open_time"][d["idx"][k]]), "advice": adv}
            async with lock:
                with open(cache_file(sym), "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
            done += 1
            if done % 200 == 0:
                log.info("%d answers (%.0fs, %d tokens)", done, time.perf_counter() - t0, llm.tokens)

        await asyncio.gather(*(ask(k) for k in todo))

    try:
        await asyncio.gather(*(one(x.strip().upper()) for x in args.symbols.split(",") if x.strip()))
    finally:
        await llm.aclose()
    log.info("LLM: %d calls, %d tokens", llm.calls, llm.tokens)
    await embed_missing(args)


async def embed_missing(args) -> None:
    """Embed every cached reason that has no vector yet (batched)."""
    for sym in [x.strip().upper() for x in args.symbols.split(",") if x.strip()]:
        recs = load_cache(sym)
        p = CACHE / f"{sym}.emb.npz"
        have = {}
        if p.exists():
            z = np.load(p)
            have = dict(zip(z["t"].tolist(), z["e"], strict=True))
        todo = [t for t in recs if t not in have]
        for i in range(0, len(todo), 256):
            batch = todo[i:i + 256]
            texts = [str(recs[t]["advice"].get("reason", "")) or "none" for t in batch]
            for t, e in zip(batch, await embed(texts, args.embed_model), strict=True):
                have[t] = np.asarray(e, np.float32)
        if have:
            ts = np.array(sorted(have), np.int64)
            np.savez_compressed(p, t=ts, e=np.stack([have[t] for t in ts]))
            log.info("%s: %d embeddings", sym, len(ts))


# --------------------------------------------------------------------------- evaluate
def scores(adv: dict) -> list[float]:
    def num(k):
        try:
            v = float(adv.get(k, 0.0))
            return v if np.isfinite(v) else 0.0
        except (TypeError, ValueError):
            return 0.0

    b, c = float(np.clip(num("bias"), -1, 1)), float(np.clip(num("confidence"), 0, 1))
    reg, act = str(adv.get("regime", "")).upper(), str(adv.get("action", "")).upper()
    return [b, c, b * c, float(np.clip(num("expected_move_bps"), -500, 500)),
            float(np.clip(num("reversal_risk"), 0, 1)),
            *[float(reg == r) for r in REGIMES], *[float(act == a) for a in ACTIONS]]


async def build_dataset(args) -> dict:
    base = Settings.from_env()
    rows = {"X": [], "nn": [], "llm": [], "emb": [], "y60": [], "t": [], "sym": []}
    for sym in [x.strip().upper() for x in args.symbols.split(",") if x.strip()]:
        recs = load_cache(sym)
        if not recs:
            continue
        d = await prepare(replace(base, symbol=sym), args.days, args.every, times=list(recs))
        p = CACHE / f"{sym}.emb.npz"
        emb = {}
        if p.exists():
            z = np.load(p)
            emb = dict(zip(z["t"].tolist(), z["e"], strict=True))
        o, c = np.log(d["bars"][:, B_OPEN]), np.log(d["bars"][:, B_CLOSE])
        for k, i in enumerate(d["idx"]):
            t = int(d["open_time"][i])
            if t not in recs or t not in emb:
                continue
            rows["X"].append(d["X"][i])
            rows["nn"].append(d["pred"][k])
            rows["llm"].append(scores(recs[t]["advice"]))
            rows["emb"].append(emb[t])
            rows["y60"].append(c[i + 60] - o[i + 1])  # enter next open, exit 60 bars later
            rows["t"].append(t)
            rows["sym"].append(sym)
        log.info("%s: %d samples with LLM features", sym, sum(s == sym for s in rows["sym"]))
    return {k: np.asarray(v) for k, v in rows.items()}


def strategy(pred: np.ndarray, y: np.ndarray, thr: float) -> dict:
    pos = np.where(pred > thr, 1.0, np.where(pred < -thr, -1.0, 0.0))
    net = pos * y * 1e4 - np.abs(pos) * COST_BPS
    n = int(np.count_nonzero(pos))
    return {"trades": n, "net_bps_total": round(float(net.sum()), 1),
            "avg_net_bps": round(float(net[pos != 0].mean()), 2) if n else 0.0,
            "hit": round(float(np.mean(np.sign(pred[pos != 0]) == np.sign(y[pos != 0]))), 3) if n else 0.0}


def evaluate(args) -> None:
    import xgboost as xgb
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA

    D = asyncio.run(build_dataset(args))
    n = len(D["t"])
    if n < 200:
        raise SystemExit(f"only {n} samples with LLM features: run `collect` first")
    cut = np.quantile(D["t"], 0.7)
    tr, te = D["t"] < cut, D["t"] >= cut + 60 * 60_000  # 1h embargo
    pca = PCA(args.pca, random_state=0).fit(D["emb"][tr])
    E = pca.transform(D["emb"])
    y = D["y60"]
    sets = {
        "base (features + NN)": np.column_stack([D["X"], D["nn"]]),
        "base + LLM scores": np.column_stack([D["X"], D["nn"], D["llm"]]),
        "base + LLM scores + emb": np.column_stack([D["X"], D["nn"], D["llm"], E]),
        "LLM scores + emb only": np.column_stack([D["llm"], E]),
    }
    print(f"\nsamples: train {tr.sum()}  test {te.sum()}  ({len(set(D['sym']))} coins, 60-min horizon,"
          f" cost {COST_BPS:.0f} bps round trip, PCA {args.pca} dims = "
          f"{pca.explained_variance_ratio_.sum() * 100:.0f}% var)")
    print(f"{'model':<28}{'IC test':>9}{'hit':>7} | strategy on test (thr = train 70th pct |pred|)")
    res = {}
    # raw LLM views, no learning
    for name, col in (("raw LLM bias*conf", D["llm"][:, 2]), ("raw LLM expected_move", D["llm"][:, 3]),
                      ("raw NN 60-bar forecast", D["nn"][:, -1])):
        ic = spearmanr(col[te], y[te]).statistic
        hit = np.mean(np.sign(col[te][col[te] != 0]) == np.sign(y[te][col[te] != 0]))
        thr = np.quantile(np.abs(col[tr]), 0.7)
        st = strategy(col[te], y[te], thr)
        res[name] = {"ic": ic, "hit": hit, **st}
        print(f"{name:<28}{ic:>+9.4f}{hit:>7.3f} | {st}")
    for name, Xs in sets.items():
        preds = np.zeros(n)
        # 3 seeds averaged: small data -> reduce variance of the comparison
        for seed in range(3):
            m = xgb.XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=3, min_child_weight=30,
                                 subsample=0.8, colsample_bytree=0.7, reg_lambda=5.0, random_state=seed,
                                 n_jobs=16)
            m.fit(Xs[tr], y[tr] / y[tr].std())
            preds += m.predict(Xs) / 3
        ic = spearmanr(preds[te], y[te]).statistic
        hit = np.mean(np.sign(preds[te]) == np.sign(y[te]))
        st = strategy(preds[te], y[te], np.quantile(np.abs(preds[tr]), 0.7))
        res[name] = {"ic": ic, "hit": hit, **st}
        print(f"{name:<28}{ic:>+9.4f}{hit:>7.3f} | {st}")
    bh = {s: float(y[te][D["sym"][te] == s].sum() * 1e4) for s in sorted(set(D["sym"]))}
    print(f"\nlong every test sample (gross, bps): {round(sum(bh.values()), 1)}  per coin: "
          + ", ".join(f"{k}:{v:+.0f}" for k, v in bh.items()))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "llm_features_eval.json").write_text(json.dumps(res, indent=2, default=float))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["collect", "embed", "evaluate"])
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT,DOGEUSDT,BNBUSDT")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--every", type=int, default=60, help="minutes between samples")
    p.add_argument("--model", default="", help="LLM (default DEEPSEEK_MODEL)")
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--pca", type=int, default=8)
    p.add_argument("--out", default="research/results")
    a = p.parse_args()
    if a.cmd == "collect":
        asyncio.run(collect(a))
    elif a.cmd == "embed":  # only embed reasons already in the cache (no chat calls)
        asyncio.run(embed_missing(a))
    else:
        evaluate(a)


if __name__ == "__main__":
    main()
