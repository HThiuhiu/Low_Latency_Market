"""Model-improvement experiments, judged by the paper broker (net of costs).

    python -m research.improve --interval 1m --days 365
    python -m research.improve --interval 15m --days 720

Protocol (no peeking at the test period):
    train : everything before val
    val   : the 30 days before the shipped model was trained -> choose horizon + threshold
    test  : bars after the shipped model was trained (same window as `python -m qforecast.sim`)
Every candidate is a (model, horizon, threshold) tuple chosen on val only; test is run once.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from qforecast.config import Settings
from qforecast.core.features import B_CLOSE, B_HIGH, B_LOW, B_OPEN, B_VOLUME, compute_features
from qforecast.exchange.binance import raw_to_bars
from qforecast.model.predictor import ModelMeta
from qforecast.sim.backtest import _ms, forecast_all, simulate
from qforecast.sim.broker import SimConfig
from qforecast.trainer.data import load_history_sync

log = logging.getLogger("improve")
DAY = 86_400_000
COST = SimConfig(fee_bps=5.0, slippage_bps=1.0)


# --------------------------------------------------------------------------- features
def extra_features(open_time: np.ndarray, bars: np.ndarray) -> np.ndarray:
    """Slower context the 64-bar kernel cannot see + calendar. Causal (uses <= t)."""
    c = np.log(bars[:, B_CLOSE])
    n = len(c)

    def lag(x, k):
        out = np.full(n, np.nan)
        out[k:] = x[k:] - x[:-k]
        return out

    r1 = lag(c, 1)
    cols = [lag(c, 60), lag(c, 240), lag(c, 1440)]
    for w in (60, 240):
        k = np.ones(w) / w
        m = np.convolve(r1 ** 2, k)[:n]
        cols.append(np.sqrt(m))
    cols.append(cols[-2] / (cols[-1] + 1e-12))  # vol ratio short/long
    rng = np.log(bars[:, B_HIGH] / bars[:, B_LOW])
    cols.append(np.convolve(rng, np.ones(30) / 30)[:n])
    v = np.log(bars[:, B_VOLUME] + 1)
    cols.append(v - np.convolve(v, np.ones(1440) / 1440)[:n])
    hour = (open_time // 3_600_000) % 24
    dow = (open_time // DAY + 4) % 7
    cols += [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24), (dow >= 5).astype(float)]
    return np.column_stack(cols)


def build_table(feats: np.ndarray, extra: np.ndarray) -> np.ndarray:
    lags = [np.roll(feats[:, :3], k, axis=0) for k in (5, 15)]  # a little recent history
    return np.column_stack([feats, extra, *lags]).astype(np.float32)


def targets(bars: np.ndarray, horizons: list[int]) -> np.ndarray:
    """Return the trade actually gets: entry at next open, exit at close of t+h."""
    n = len(bars)
    y = np.full((n, len(horizons)), np.nan)
    o, c = np.log(bars[:, B_OPEN]), np.log(bars[:, B_CLOSE])
    for j, h in enumerate(horizons):
        y[:n - h, j] = c[h:] - o[1:n - h + 1] if h >= 1 else np.nan
    return y


# --------------------------------------------------------------------------- strategy eval
def run_strategy(ctx, idx: np.ndarray, pred: np.ndarray, h: int, thr_bps: float,
                 long_only: bool = False) -> dict:
    open_time, bars = ctx["open_time"], ctx["bars"]
    meta = ModelMeta("x", [], 1, [h], np.zeros(1), np.ones(1), np.ones(1), np.ones(1, np.float32))
    cfg = replace(COST, allow_short=not long_only)
    b = simulate(open_time, bars, pred.reshape(-1, 1).astype(np.float32), idx, meta, cfg, thr_bps,
                 int(idx[0]), int(idx[-1]) + 1)
    sm = b.summary(ctx["bars_per_year"])
    return {k: sm[k] for k in ("total_return_pct", "trades", "win_rate_pct", "avg_trade_bps",
                               "max_drawdown_pct", "sharpe_ann")}


def tune(ctx, val_idx, val_pred, test_idx, test_pred, horizons, name: str, min_trades: int = 15):
    """Pick (horizon, threshold, long_only) on val; evaluate that single choice on test."""
    best = None
    for j, h in enumerate(horizons):
        a = np.abs(val_pred[:, j]) * 1e4
        for q in (0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995):
            thr = float(np.quantile(a, q))
            for lo in (False, True):
                r = run_strategy(ctx, val_idx, val_pred[:, j], h, thr, lo)
                if r["trades"] < min_trades:
                    continue
                if best is None or r["total_return_pct"] > best[0]["total_return_pct"]:
                    best = (r, h, j, thr, lo)
    if best is None:
        return {"model": name, "note": "no config traded enough on val"}
    r_val, h, j, thr, lo = best
    r_test = run_strategy(ctx, test_idx, test_pred[:, j], h, thr, lo)
    out = {"model": name, "horizon": h, "threshold_bps": round(thr, 2), "long_only": lo,
           "val": r_val, "test": r_test}
    log.info("%-28s h=%-4d thr=%6.1fbps lo=%d | val %+7.2f%% (%3d tr) | TEST %+7.2f%% (%3d tr, "
             "win %4.1f%%, sharpe %5.2f)", name, h, thr, lo, r_val["total_return_pct"], r_val["trades"],
             r_test["total_return_pct"], r_test["trades"], r_test["win_rate_pct"], r_test["sharpe_ann"])
    return out


def signal_quality(pred, y, name):
    from scipy.stats import spearmanr

    ok = ~np.isnan(y)
    return {"model": name, "ic": [round(float(spearmanr(pred[ok[:, j], j], y[ok[:, j], j]).statistic), 4)
                                  for j in range(y.shape[1])]}


# --------------------------------------------------------------------------- models
def fit_xgb(X, y, tr, va, horizons, objective="reg"):
    import xgboost as xgb

    preds = []
    for j, h in enumerate(horizons):
        ok_tr, ok_va = tr[~np.isnan(y[tr, j])], va[~np.isnan(y[va, j])]
        scale = np.nanstd(y[ok_tr, j])
        params = dict(n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=200,
                      subsample=0.7, colsample_bytree=0.7, reg_lambda=10.0, n_jobs=16,
                      early_stopping_rounds=100, tree_method="hist")
        if objective == "cls":  # direction only, weighted by move size
            m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss", **params)
            w = np.abs(y[ok_tr, j]) / scale
            wv = np.abs(y[ok_va, j]) / scale
            m.fit(X[ok_tr], y[ok_tr, j] > 0, sample_weight=w, eval_set=[(X[ok_va], y[ok_va, j] > 0)],
                  sample_weight_eval_set=[wv], verbose=False)
            # map P(up) to an expected move: (2p-1) * E|y|
            mag = np.nanmean(np.abs(y[ok_tr, j]))
            preds.append(lambda Z, m=m, s=mag: (2 * m.predict_proba(Z)[:, 1] - 1) * s)
        else:
            m = xgb.XGBRegressor(objective="reg:pseudohubererror", huber_slope=1.0, **params)
            m.fit(X[ok_tr], y[ok_tr, j] / scale, eval_set=[(X[ok_va], y[ok_va, j] / scale)], verbose=False)
            preds.append(lambda Z, m=m, s=scale: m.predict(Z) * s)
        log.info("  xgb-%s h=%d: %d trees", objective, h, m.best_iteration + 1)
    return lambda Z: np.column_stack([p(Z) for p in preds])


# --------------------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1m")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--val-days", type=int, default=30)
    ap.add_argument("--out", default="research/results")
    ap.add_argument("--walk", type=int, default=0,
                    help="walk-forward: N consecutive 30-day test folds ending now (retrain each)")
    args = ap.parse_args()

    s = replace(Settings.from_env(), symbol=args.symbol, interval=args.interval)
    meta = ModelMeta.load(s.meta_path)
    t_model = _ms(json.loads(Path(s.meta_path).read_text())["created_at"])
    raw = load_history_sync(s, args.days)
    open_time, bars = raw_to_bars(raw)
    feats, valid = compute_features(bars)
    X = build_table(feats, extra_features(open_time, bars))

    bar_min = s.interval_ms // 60_000
    # horizons in bars: roughly 15m .. 1 day of wall time, whatever the bar size
    horizons = sorted({max(1, m // bar_min) for m in (15, 30, 60, 120, 240, 480, 1440)})
    y = targets(bars, horizons)
    ok = valid & ~np.isnan(X).any(axis=1)
    ok[:1500] = False  # long-window extras need warm-up
    allidx = np.flatnonzero(ok)
    now = int(open_time[-1])
    folds = ([(t_model, now + 1)] if not args.walk else
             [(now - (k + 1) * 30 * DAY, now - k * 30 * DAY + 1) for k in reversed(range(args.walk))])
    ctx = {"open_time": open_time, "bars": bars, "bars_per_year": 365 * DAY / s.interval_ms}
    report = []
    for t0, t1 in folds:
        report.append(run_fold(args, s, meta, ctx, feats, X, y, allidx, horizons, t0, t1,
                               use_nn=s.interval == "1m" and t0 >= t_model))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_walk{args.walk}" if args.walk else ""
    (out / f"{s.symbol}_{s.interval}{tag}.json").write_text(json.dumps(
        {"symbol": s.symbol, "interval": s.interval, "days": args.days, "horizons": horizons,
         "folds": report}, indent=2))
    if args.walk:
        print_walk(report)


def print_walk(report):
    """Test return per fold (trades in brackets) + compounded total."""
    names = [r["model"] for r in report[-1]["results"] if not r["model"].startswith("shipped")]
    print("\n" + f"{'test fold':<24}{'B&H':>8}" + "".join(f"{n:>18}" for n in names))
    tot = dict.fromkeys(names, 1.0)
    for f in report:
        row = f"{f['test_period'][0]}..{f['test_period'][1]:<12}"[:24] + f"{f['buy_hold_test_pct']:>+7.1f}%"
        res = {r["model"]: r for r in f["results"]}
        for n in names:
            r = res[n]
            v = r["test"]["total_return_pct"] if "test" in r else 0.0
            tr = r["test"]["trades"] if "test" in r else 0
            tot[n] *= 1 + v / 100
            row += f"{v:>+11.1f}% ({tr:>3})"
        print(row)
    print(f"{'compounded':<32}" + "".join(f"{(tot[n] - 1) * 100:>+11.1f}%      " for n in names))


def run_fold(args, s, meta, ctx, feats, X, y, allidx, horizons, t0, t1, use_nn):
    open_time, bars = ctx["open_time"], ctx["bars"]
    val_start = t0 - args.val_days * DAY
    emb = max(horizons) * s.interval_ms
    tr = allidx[open_time[allidx] < val_start - emb]
    va = allidx[(open_time[allidx] >= val_start) & (open_time[allidx] < t0 - emb)]
    te = allidx[(open_time[allidx] >= t0) & (open_time[allidx] < t1)]
    te = te[te < len(bars) - 1]
    log.info("%s %s: bars=%d  train=%d val=%d test=%d  horizons=%s", s.symbol, s.interval, len(bars),
             len(tr), len(va), len(te), horizons)

    results, quality = [], []
    bh = bars[te[-1], B_CLOSE] / bars[te[0], B_OPEN] - 1
    log.info("buy & hold over test: %+.2f%%", bh * 100)

    # 0) shipped model (1m only), with its rule and with a val-tuned threshold
    if use_nn:
        from qforecast.model.predictor import make_session

        sess = make_session(s.model_path)
        nn = {"va": forecast_all(meta, sess, feats, va), "te": forecast_all(meta, sess, feats, te)}
        hz = list(meta.horizons)
        results.append(tune(ctx, va, nn["va"], te, nn["te"], hz, "shipped NN (tuned thr)"))
        quality.append(signal_quality(nn["te"], targets(bars, hz)[te], "shipped NN"))

    # 1) XGBoost regression / direction classifier on richer features
    for obj in ("reg", "cls"):
        t = time.perf_counter()
        f = fit_xgb(X, y, tr, va, horizons, obj)
        log.info("  fit in %.0fs", time.perf_counter() - t)
        pv, pt = f(X[va]), f(X[te])
        results.append(tune(ctx, va, pv, te, pt, horizons, f"xgb-{obj}"))
        quality.append(signal_quality(pt, y[te], f"xgb-{obj}"))

    # 2) naive baselines with the same tuning: momentum and mean reversion
    for name, sign in (("momentum", 1.0), ("reversal", -1.0)):
        c = np.log(bars[:, B_CLOSE])
        pv = np.column_stack([sign * (c[va] - c[va - h]) for h in horizons])
        pt = np.column_stack([sign * (c[te] - c[te - h]) for h in horizons])
        results.append(tune(ctx, va, pv, te, pt, horizons, name))

    return {"test_period": [time.strftime("%Y-%m-%d", time.gmtime(open_time[te[0]] / 1000)),
                           time.strftime("%Y-%m-%d", time.gmtime(open_time[te[-1]] / 1000))],
            "buy_hold_test_pct": bh * 100, "results": results, "ic_test": quality}


if __name__ == "__main__":
    main()
