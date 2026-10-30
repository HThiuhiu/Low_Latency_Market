"""Replay the trading agent on recent history, many coins at once, vs the model alone.

    python -m qforecast.agent.replay --symbols BTCUSDT,ETHUSDT,SOLUSDT --hours 24 --max-calls 40

Each bar: features -> ONNX forecast -> (when the agent would ask) LLM advice -> risk rules
-> order filled at the NEXT bar's open. Coins run concurrently; bars within a coin run in
order, waiting for each LLM reply (the replay is not real-time).

Caveats: no order-book microstructure offline (the live agent has it); an LLM may
remember prices from before its training cutoff, so only replay recent periods.
Costs real API tokens: bounded by --max-calls per coin.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from qforecast.agent.core import SYSTEM_PROMPT, AgentConfig, AgentCore
from qforecast.agent.llm import LLMClient, LLMError
from qforecast.config import Settings
from qforecast.core.features import B_CLOSE, B_HIGH, B_LOW, B_OPEN, compute_features
from qforecast.exchange.binance import raw_to_bars
from qforecast.model.predictor import ModelMeta, decide_signal, make_session
from qforecast.sim.backtest import forecast_all, renormalise, simulate
from qforecast.sim.broker import PaperBroker, SimConfig
from qforecast.trainer.data import load_history
from qforecast.trainer.dataset import eligible_indices

log = logging.getLogger("replay")


def fview(exp: np.ndarray, meta: ModelMeta, sig: str, hz: int, edge: float) -> dict:
    return {"signal": sig, "best_horizon_bars": hz, "edge_bps": round(edge, 1),
            "by_horizon": {str(h): {"exp_bps": round(float(e) * 1e4, 1),
                                    "resid_sigma_bps": round(float(sg) * 1e4, 1)}
                           for h, e, sg in zip(meta.horizons, exp, meta.sigma, strict=True)}}


async def replay_symbol(s: Settings, args, llm: LLMClient, sem: asyncio.Semaphore) -> dict:
    meta = ModelMeta.load(s.meta_path)
    trained_on = json.loads(Path(s.meta_path).read_text()).get("symbol")
    raw = await load_history(s, args.renorm_days + args.hours // 24 + 2)
    open_time, bars = raw_to_bars(raw)
    feats, valid = compute_features(bars)
    start_i = len(bars) - args.hours * 60 * 60_000 // s.interval_ms
    if trained_on != s.symbol:
        meta = renormalise(meta, feats, valid, start_i, args.renorm_days * 86_400_000 // s.interval_ms)
    idx = eligible_indices(open_time, valid, s.interval_ms, meta.lookback, 0)
    idx = idx[idx >= start_i]
    pred = await asyncio.to_thread(forecast_all, meta, make_session(s.model_path), feats, idx)
    row = np.full(len(bars), -1)
    row[idx] = np.arange(len(idx))

    sim_cfg = SimConfig(cash=10_000.0, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps)
    thr = s.signal_threshold_bps
    base = simulate(open_time, bars, pred, idx, meta, sim_cfg, thr, start_i, len(bars))

    cfg = AgentConfig(mode=args.mode, min_conf=args.min_conf, max_calls_hour=args.max_calls_hour,
                      review_bars=args.review_bars, entry_gap_bars=args.entry_gap_bars,
                      ask_on_flat=args.ask_on_flat)
    core = AgentCore(PaperBroker(replace(sim_cfg, stop_bps=50.0)), cfg, s.interval_ms)
    for i in range(max(0, start_i - 1500), start_i):  # warm price context only
        core.on_bar(int(open_time[i]), *bars[i, [B_OPEN, B_HIGH, B_LOW, B_CLOSE]], trade=False)
    calls, opens, vetoed, errors = 0, 0, 0, 0
    log_path = Path(args.out) / f"{s.symbol}_decisions.jsonl"
    with open(log_path, "w", encoding="utf-8") as fh:
        for i in range(start_i, len(bars)):
            t = int(open_time[i])
            core.on_bar(t, *bars[i, [B_OPEN, B_HIGH, B_LOW, B_CLOSE]])
            k = row[i]
            if k < 0 or i == len(bars) - 1:
                continue
            sig, hz, edge = decide_signal(pred[k], meta.sigma, meta.horizons, thr)
            now = t + s.interval_ms
            reason = core.should_ask(sig, now) if calls < args.max_calls else None
            if not reason:
                continue
            core.mark_asked(now)
            calls += 1
            ctx = core.context(s.symbol, t, float(bars[i, B_CLOSE]), fview(pred[k], meta, sig, hz, edge))
            advice, lat, err = None, 0.0, ""
            async with sem:
                try:
                    advice, lat = await llm.ask_json(SYSTEM_PROMPT, core.prompt(ctx))
                except LLMError as e:
                    err, errors = str(e), errors + 1
            d = core.decide(advice, sig, hz)
            core.execute(d, t, None if d.action != "CLOSE" else float(bars[i, B_CLOSE]))
            opens += d.action.startswith("OPEN")
            vetoed += bool(d.vetoes)
            fh.write(json.dumps({**core.record(t, reason, d, lat, err), "context": ctx}) + "\n")
    core.broker.close_all()

    bpy = 365 * 86_400_000 / s.interval_ms
    bh = bars[-1, B_CLOSE] / bars[start_i, B_OPEN] - 1
    a, b = core.broker.summary(bpy, bh), base.summary(bpy, bh)
    keys = ("total_return_pct", "trades", "win_rate_pct", "avg_trade_bps", "max_drawdown_pct")
    res = {"symbol": s.symbol, "buy_hold_pct": bh * 100, "llm_calls": calls, "llm_errors": errors,
           "agent_opens": opens, "vetoed": vetoed, "agent": {k: a[k] for k in keys},
           "model_only": {k: b[k] for k in keys}}
    log.info("%s done: agent %+.2f%% (%d trades) vs model %+.2f%% (%d) | B&H %+.2f%% | %d calls", s.symbol,
             a["total_return_pct"], a["trades"], b["total_return_pct"], b["trades"], bh * 100, calls)
    return res


async def main_async(args) -> list[dict]:
    base = Settings.from_env()
    llm = LLMClient(model=args.model, timeout_s=60)
    if not llm.configured:
        raise SystemExit("DEEPSEEK_API_KEY is not set")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(args.concurrency)
    syms = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    t0 = time.perf_counter()
    try:
        res = await asyncio.gather(*(replay_symbol(replace(base, symbol=x), args, llm, sem) for x in syms),
                                   return_exceptions=True)
    finally:
        await llm.aclose()
    ok = []
    for x, r in zip(syms, res, strict=True):
        if isinstance(r, BaseException):
            log.error("%s failed: %s", x, r)
        else:
            ok.append(r)
    print(f"\n{'symbol':<10}{'B&H':>8} | {'model':>8}{'tr':>4}{'win%':>6} | {'agent':>8}{'tr':>4}{'win%':>6}"
          f"{'calls':>6}{'veto':>5}")
    for r in ok:
        m, a = r["model_only"], r["agent"]
        print(f"{r['symbol']:<10}{r['buy_hold_pct']:>+7.2f}% | "
              f"{m['total_return_pct']:>+7.2f}%{m['trades']:>4}"
              f"{m['win_rate_pct']:>6.0f} | {a['total_return_pct']:>+7.2f}%{a['trades']:>4}"
              f"{a['win_rate_pct']:>6.0f}{r['llm_calls']:>6}{r['vetoed']:>5}")
    if ok:
        mean = lambda k, w: np.mean([r[w][k] for r in ok])  # noqa: E731
        print(f"{'mean':<10}{np.mean([r['buy_hold_pct'] for r in ok]):>+7.2f}% | "
              f"{mean('total_return_pct', 'model_only'):>+7.2f}%{'':>10} | "
              f"{mean('total_return_pct', 'agent'):>+7.2f}%")
    print(f"\nLLM: {llm.calls} calls, {llm.tokens} tokens, {time.perf_counter() - t0:.0f}s -> {args.out}")
    (Path(args.out) / "replay_summary.json").write_text(json.dumps(ok, indent=2))
    return ok


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--max-calls", type=int, default=40, help="LLM calls per coin")
    p.add_argument("--max-calls-hour", type=int, default=30)
    p.add_argument("--review-bars", type=int, default=15)
    p.add_argument("--entry-gap-bars", type=int, default=10)
    p.add_argument("--ask-on-flat", action="store_true", help="free mode: also ask when the model is FLAT")
    p.add_argument("--mode", default="veto", choices=["veto", "free"])
    p.add_argument("--min-conf", type=float, default=0.6)
    p.add_argument("--model", default="", help="LLM model (default: DEEPSEEK_MODEL)")
    p.add_argument("--concurrency", type=int, default=6, help="parallel LLM requests")
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--renorm-days", type=int, default=14)
    p.add_argument("--out", default="data/agent_replay")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
