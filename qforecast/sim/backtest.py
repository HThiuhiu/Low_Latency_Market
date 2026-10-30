"""Historical trading simulation with the trained ONNX model.

    python -m qforecast.sim                       # out-of-sample: bars after the model was trained
    python -m qforecast.sim --days 30 --stop-bps 40 --take-bps 80
    python -m qforecast.sim --start 2026-10-10 --end 2026-10-20 --threshold-bps 5

Bar by bar, the same way the live service sees the market:
    bar t closes -> features (same numba kernel) -> z-score with the model's stats
    -> ONNX forecast -> decide_signal -> order filled at the OPEN of bar t+1.

Outputs (artifacts/sim/): summary.json, trades.csv, equity.csv (+ equity.png if
matplotlib is installed).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qforecast.config import Settings
from qforecast.core.features import B_CLOSE, B_HIGH, B_LOW, B_OPEN, compute_features
from qforecast.exchange.binance import raw_to_bars
from qforecast.model.predictor import ModelMeta, decide_signal, make_session
from qforecast.sim.broker import PaperBroker, SimConfig
from qforecast.trainer.data import load_history_sync
from qforecast.trainer.dataset import eligible_indices

log = logging.getLogger("sim")

CLIP = 5.0


def _ms(date: str) -> int:
    d = datetime.fromisoformat(date.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=UTC)
    return int(d.timestamp() * 1000)


def _fmt(ms: int) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.gmtime(ms / 1000))


def renormalise(meta: ModelMeta, feats: np.ndarray, valid: np.ndarray, end_i: int,
                n: int) -> ModelMeta:
    """Adapt a model to another coin: z-score stats from that coin's own last ``n`` bars
    before ``end_i`` (causal), and rescale outputs by its volatility relative to the
    training coin (so thresholds in bps keep their meaning)."""
    rows = np.flatnonzero(valid[:end_i])[-n:]
    if len(rows) < 1000:
        raise SystemExit("not enough history before the period to re-normalise")
    sub = feats[rows]
    mean, std = sub.mean(axis=0).astype(np.float32), (sub.std(axis=0) + 1e-12).astype(np.float32)
    k = float(std[0] / meta.feat_std[0])  # ret_1 std ratio
    return replace(meta, feat_mean=mean, feat_std=std, y_scale=meta.y_scale * k, sigma=meta.sigma * k)


def forecast_all(meta: ModelMeta, session, feats: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Expected log-return per horizon for every bar in ``idx`` (batch-1 ONNX, as served)."""
    L = meta.lookback
    z = np.clip((np.nan_to_num(feats) - meta.feat_mean) / meta.feat_std, -CLIP, CLIP).astype(np.float32)
    name = session.get_inputs()[0].name
    out = np.zeros((len(idx), len(meta.horizons)), np.float32)
    t0 = time.perf_counter()
    for k, i in enumerate(idx):
        out[k] = session.run(None, {name: z[None, i - L + 1:i + 1]})[0][0]
    log.info("%d forecasts in %.1fs (%.0f us each)", len(idx), time.perf_counter() - t0,
             (time.perf_counter() - t0) / max(len(idx), 1) * 1e6)
    return out * meta.y_scale


def simulate(open_time: np.ndarray, bars: np.ndarray, pred: np.ndarray, pred_idx: np.ndarray,
             meta: ModelMeta, cfg: SimConfig, threshold_bps: float, start_i: int, end_i: int,
             horizon: int | None = None) -> PaperBroker:
    """Replay bars [start_i, end_i). ``pred`` row k is the forecast made at close of pred_idx[k]."""
    broker = PaperBroker(cfg)
    broker.signal_counts = {}
    row = np.full(len(bars), -1, np.int64)
    row[pred_idx] = np.arange(len(pred_idx))
    horizons = list(meta.horizons)
    if horizon is not None:  # force one horizon instead of the model's best risk-adjusted one
        j = horizons.index(horizon)
        sigma, horizons = meta.sigma[j:j + 1], [horizon]
    for i in range(start_i, end_i):
        o, h, lo, c = bars[i, B_OPEN], bars[i, B_HIGH], bars[i, B_LOW], bars[i, B_CLOSE]
        broker.on_bar(int(open_time[i]), o, h, lo, c)
        k = row[i]
        if k < 0 or i == end_i - 1:
            continue
        exp = pred[k] if horizon is None else pred[k, j:j + 1]
        sig, hz, edge = decide_signal(exp, meta.sigma if horizon is None else sigma, horizons,
                                      threshold_bps)
        broker.signal_counts[sig] = broker.signal_counts.get(sig, 0) + 1
        broker.on_signal(int(open_time[i]), sig, hz, edge)  # filled at next bar's open
    broker.close_all()
    return broker


def write_outputs(out: Path, broker: PaperBroker, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "trades.csv", "w", newline="") as f:
        w = csv.writer(f)
        fields = list(asdict(broker.trades[0]).keys()) if broker.trades else ["side"]
        w.writerow(fields)
        for t in broker.trades:
            d = asdict(t)
            d["entry_time"], d["exit_time"] = _fmt(t.entry_time), _fmt(t.exit_time)
            w.writerow([d[k] for k in fields])
    with open(out / "equity.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "equity"])
        w.writerows((_fmt(t), round(e, 4)) for t, e in broker.equity_curve)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    t = np.array([x for x, _ in broker.equity_curve], dtype="datetime64[ms]")
    e = np.array([x for _, x in broker.equity_curve])
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, e, lw=1)
    ax.axhline(broker.cfg.cash, color="grey", lw=0.6, ls="--")
    ax.set_title(f"Paper equity ({summary['total_return_pct']:+.2f}%, {summary['trades']} trades)")
    fig.tight_layout()
    fig.savefig(out / "equity.png", dpi=110)
    plt.close(fig)


def run(args) -> dict | list[dict]:
    base = Settings.from_env()
    raw = args.symbols or args.symbol or base.symbol
    symbols = [x.strip().upper() for x in raw.split(",") if x.strip()]
    if len(symbols) == 1:
        return run_one(replace(base, symbol=symbols[0]), args)
    out = []
    root = Path(args.out or base.artifacts_dir / "sim")
    for sym in symbols:
        try:
            out.append(run_one(replace(base, symbol=sym), args, root / sym))
        except (SystemExit, Exception) as e:  # one bad symbol must not stop the sweep
            log.error("%s failed: %s", sym, e)
    print_table(out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "multi_summary.json").write_text(json.dumps(out, indent=2))
    return out


def print_table(rows: list[dict]) -> None:
    print()
    print(f"{'symbol':<10}{'trades':>7}{'win%':>7}{'avg bps':>9}{'return':>9}{'B&H':>9}"
          f"{'maxDD':>8}{'sharpe':>8}")
    for r in rows:
        print(f"{r['symbol']:<10}{r['trades']:>7}{r['win_rate_pct']:>7.1f}{r['avg_trade_bps']:>+9.1f}"
              f"{r['total_return_pct']:>+8.2f}%{r['buy_hold_return_pct']:>+8.2f}%"
              f"{r['max_drawdown_pct']:>7.1f}%{r['sharpe_ann']:>8.2f}")
    if rows:
        print(f"{'mean':<10}{np.mean([r['trades'] for r in rows]):>7.0f}"
              f"{np.mean([r['win_rate_pct'] for r in rows]):>7.1f}"
              f"{np.mean([r['avg_trade_bps'] for r in rows]):>+9.1f}"
              f"{np.mean([r['total_return_pct'] for r in rows]):>+8.2f}%"
              f"{np.mean([r['buy_hold_return_pct'] for r in rows]):>+8.2f}%")


def run_one(s: Settings, args, out: Path | None = None) -> dict:
    meta = ModelMeta.load(s.meta_path)
    meta_raw = json.loads(Path(s.meta_path).read_text())
    trained_at = _ms(meta_raw["created_at"]) if "created_at" in meta_raw else None
    foreign = bool(meta_raw.get("symbol")) and meta_raw["symbol"] != s.symbol
    renorm = args.renorm == "yes" or (args.renorm == "auto" and foreign)
    if foreign:
        log.warning("model trained on %s, simulating %s (re-normalise: %s)", meta_raw["symbol"],
                    s.symbol, renorm)

    now = int(time.time() * 1000)
    if args.start:
        start_ms = _ms(args.start)
    elif args.days:
        start_ms = now - args.days * 86_400_000
    elif trained_at:
        start_ms = trained_at  # default: only data the model has never seen
    else:
        start_ms = now - 30 * 86_400_000
    end_ms = _ms(args.end) if args.end else now
    if trained_at and start_ms < trained_at:
        log.warning("start %s is before the model was trained (%s): part of the period is "
                    "IN-SAMPLE, results will look better than reality", _fmt(start_ms), _fmt(trained_at))

    # Extra history before start so features + lookback window are warm at the first bar.
    warm_ms = (meta.lookback + 200 + (args.renorm_days * 1440 if renorm else 0)) * s.interval_ms
    days = math.ceil((now - start_ms + warm_ms) / 86_400_000) + 1
    raw = load_history_sync(s, days)
    open_time, bars = raw_to_bars(raw)
    feats, valid = compute_features(bars)
    idx = eligible_indices(open_time, valid, s.interval_ms, meta.lookback, 0)

    start_i = int(np.searchsorted(open_time, start_ms))
    end_i = int(np.searchsorted(open_time, end_ms, side="right"))
    idx = idx[(idx >= start_i) & (idx < end_i)]
    if end_i - start_i < 2 or len(idx) == 0:
        raise SystemExit("not enough bars in the requested period")
    log.info("simulating %s %s .. %s (%d bars, model %s)", s.symbol, _fmt(open_time[start_i]),
             _fmt(open_time[end_i - 1]), end_i - start_i, meta.version)

    if renorm:
        meta = renormalise(meta, feats, valid, start_i, args.renorm_days * 86_400_000 // s.interval_ms)
    session = make_session(s.model_path, threads=1)
    pred = forecast_all(meta, session, feats, idx)

    cfg = SimConfig(cash=args.cash, fee_bps=args.fee_bps, slippage_bps=args.slippage_bps,
                    size=args.size, stop_bps=args.stop_bps, take_bps=args.take_bps,
                    allow_short=not args.long_only)
    threshold = s.signal_threshold_bps if args.threshold_bps is None else args.threshold_bps
    broker = simulate(open_time, bars, pred, idx, meta, cfg, threshold, start_i, end_i, args.horizon)

    bars_per_year = 365 * 86_400_000 / s.interval_ms
    bench = bars[end_i - 1, B_CLOSE] / bars[start_i, B_OPEN] - 1
    summary = {
        "symbol": s.symbol, "model_version": meta.version,
        "period": [_fmt(open_time[start_i]), _fmt(open_time[end_i - 1])],
        "out_of_sample": bool(trained_at and start_ms >= trained_at),
        "threshold_bps": threshold, "horizon": args.horizon or "auto", "renormalised": renorm,
        "signals": broker.signal_counts,
        **broker.summary(bars_per_year, bench),
    }
    out = out or (Path(args.out) if args.out else s.artifacts_dir / "sim")
    write_outputs(out, broker, summary)
    print_summary(summary, out)
    return summary


def print_summary(sm: dict, out: Path) -> None:
    pf = sm["profit_factor"]
    rows = [
        ("Period", f"{sm['period'][0]} .. {sm['period'][1]} UTC"),
        ("Out-of-sample", "yes" if sm["out_of_sample"] else "NO (overlaps training)"),
        ("Signals", ", ".join(f"{k}={v}" for k, v in sm["signals"].items())),
        ("Trades", f"{sm['trades']}  (long {sm['longs']}, short {sm['shorts']})"),
        ("Win rate", f"{sm['win_rate_pct']:.1f}%"),
        ("Avg trade", f"{sm['avg_trade_bps']:+.1f} bps (net)"),
        ("Profit factor", "n/a" if pf is None else f"{pf:.2f}"),
        ("Equity", f"{sm['start_equity']:,.2f} -> {sm['end_equity']:,.2f}"),
        ("Total return", f"{sm['total_return_pct']:+.2f}%"),
        ("Buy & hold", f"{sm['buy_hold_return_pct']:+.2f}%"),
        ("Max drawdown", f"{sm['max_drawdown_pct']:.2f}%"),
        ("Sharpe (ann.)", f"{sm['sharpe_ann']:.2f}"),
        ("Fees paid", f"{sm['fees_paid']:,.2f}"),
        ("Exposure", f"{sm['exposure_pct']:.1f}% of bars"),
        ("Exit reasons", ", ".join(f"{k}={v}" for k, v in sm["exit_reasons"].items()) or "-"),
    ]
    w = max(len(k) for k, _ in rows)
    print(f"\n== Paper trading simulation: {sm['symbol']} (model {sm['model_version']}) ==")
    for k, v in rows:
        print(f"  {k:<{w}}  {v}")
    print(f"  -> {out}/summary.json, trades.csv, equity.csv\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("period (default: everything after the model was trained)")
    g.add_argument("--days", type=int, help="last N days")
    g.add_argument("--start", help="ISO date/time, UTC (e.g. 2026-10-10)")
    g.add_argument("--end", help="ISO date/time, UTC (default: now)")
    g.add_argument("--symbol", help="default: QF_SYMBOL / ETHUSDT")
    g.add_argument("--symbols", help="comma list, e.g. BTCUSDT,ETHUSDT,SOLUSDT -> one table")
    g.add_argument("--renorm", default="auto", choices=["auto", "yes", "no"],
                   help="z-score stats from the coin's own past (auto: when not the training coin)")
    g.add_argument("--renorm-days", type=int, default=14)
    g = p.add_argument_group("strategy")
    g.add_argument("--threshold-bps", type=float, help="min |edge| to trade (default: settings)")
    g.add_argument("--horizon", type=int, help="force one horizon (1/5/15/30/60); default: best z")
    g.add_argument("--long-only", action="store_true")
    g.add_argument("--stop-bps", type=float, default=0.0)
    g.add_argument("--take-bps", type=float, default=0.0)
    g = p.add_argument_group("account / costs")
    g.add_argument("--cash", type=float, default=10_000.0)
    g.add_argument("--size", type=float, default=1.0, help="notional / equity (leverage)")
    g.add_argument("--fee-bps", type=float, default=5.0, help="per side")
    g.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--out", help="output dir (default artifacts/sim)")
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
