# ruff: noqa: E501  (markdown report rows are long by nature)
"""Before vs after tuning: both pipelines measured in the same run, interleaved.

    QF_BENCH_PCORES=0-11 python -m benchmarks.bench_before_after   # writes BEFORE_AFTER.md

Why interleaved: on a laptop, absolute timings drift by 2-3x between runs (thermal
state, power management, OS scheduling). Running "before" and "after" alternately,
bar by bar, in one process exposes both to the same conditions, so the *ratio* is
trustworthy even when the absolute numbers are not. Speed-ups come with a 95%
bootstrap confidence interval.

BEFORE = the prototype's approach, rebuilt stage by stage:
    orjson -> dict -> struct · pandas indicator recompute over 500 bars · MinMax scaling
    · PyTorch eager, default threads, first architecture (2-layer BiLSTM on 64 steps)
    · forecast serialised as JSON
AFTER  = the tuned system:
    msgspec typed decode · incremental numba features · z-score into a zero-copy ring
    · ONNX Runtime, 1 thread, latency-shaped architecture (1-layer BiLSTM on 32 steps)
    · forecast as msgpack struct · optionally pinned to P-cores
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import orjson
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.bench_features import legacy_add_features  # noqa: E402
from benchmarks.bench_io import legacy_parse  # noqa: E402
from benchmarks.common import machine, parse_cpus  # noqa: E402
from qforecast.core.features import OnlineFeatureEngine  # noqa: E402
from qforecast.exchange.binance import parse_message  # noqa: E402
from qforecast.model.net import build_model  # noqa: E402
from qforecast.model.predictor import Predictor  # noqa: E402
from qforecast.schemas import BookTicker, Forecast, encode  # noqa: E402
from qforecast.storage import columnar  # noqa: E402
from qforecast.storage.ticklog import TickLogWriter, read_dir  # noqa: E402
from tests.conftest import write_random_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
HISTORY, N_BARS, LOOKBACK, HORIZONS = 500, 300, 128, [1, 5, 15, 30, 60]
LEGACY_FEATS = ["close", "volume", "return_1", "return_3", "roc_10", "bb_width", "atr_14_pct",
                "dist_ma_20", "dist_ma_50", "rvol", "rsi_14", "macd_hist", "dist_to_upper",
                "pump_signal"]
STAGES = ["decode", "features", "normalise", "inference", "encode"]
STAGE_EN = {"decode": "Frame decode", "features": "Features", "normalise": "Normalise + model input",
            "inference": "Model inference", "encode": "Serialise result", "total": "**Total per bar**"}


# --------------------------------------------------------------------------- data
def load_raw() -> tuple[np.ndarray, str]:
    p = ROOT / "data" / "ETHUSDT_1m.qcol"
    if p.exists():
        raw = np.ascontiguousarray(columnar.load(p))[-(HISTORY + N_BARS + 200):]
        return raw, "real ETHUSDT 1-minute bars (Binance cache)"
    from tests.conftest import synthetic_bars

    b = synthetic_bars(HISTORY + N_BARS + 200)
    t = np.arange(len(b)) * 60_000.0
    return np.c_[t, b[:, :5], t + 59_999, b[:, 4] * b[:, 3], b[:, 5], b[:, 6], b[:, 6]], "synthetic bars"


def kline_frame(r: np.ndarray) -> bytes:
    """Rebuild the exact Binance combined-stream frame for a closed 1m kline."""
    k = {"t": int(r[0]), "T": int(r[6]), "s": "ETHUSDT", "i": "1m", "f": 1, "L": 2,
         "o": f"{r[1]:.8f}", "c": f"{r[4]:.8f}", "h": f"{r[2]:.8f}", "l": f"{r[3]:.8f}",
         "v": f"{r[5]:.8f}", "n": int(r[8]), "x": True, "q": f"{r[7]:.8f}",
         "V": f"{r[9]:.8f}", "Q": f"{r[10]:.8f}", "B": "0"}
    return orjson.dumps({"stream": "ethusdt@kline_1m",
                         "data": {"e": "kline", "E": int(r[6]) + 12, "s": "ETHUSDT", "k": k}})


def book_frame(b: BookTicker) -> bytes:
    return orjson.dumps({"stream": "ethusdt@bookTicker", "data": {
        "u": b.update_id, "s": b.symbol, "b": f"{b.bid:.8f}", "B": f"{b.bid_qty:.8f}",
        "a": f"{b.ask:.8f}", "A": f"{b.ask_qty:.8f}"}})


# --------------------------------------------------------------------------- pipelines
class Before:
    def __init__(self, raw: np.ndarray):
        self.raw = raw
        torch.manual_seed(0)
        self.model = build_model("lstm", 14, len(HORIZONS), lstm_layers=2, downsample=2).eval()
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "qv",
                "trades", "tbb", "tbq"]
        f = legacy_add_features(pd.DataFrame(raw[:HISTORY], columns=cols))[LEGACY_FEATS]
        self.cols = cols
        self.lo, self.hi = f.min().values, f.max().values  # MinMaxScaler fitted once

    def run(self, i: int, frame: bytes, t: dict) -> None:
        t0 = time.perf_counter_ns()
        k = legacy_parse(frame, 0)
        t1 = time.perf_counter_ns()
        window = pd.DataFrame(self.raw[i - HISTORY + 1:i + 1], columns=self.cols)
        feats = legacy_add_features(window)[LEGACY_FEATS].values[-LOOKBACK:]
        t2 = time.perf_counter_ns()
        x = torch.tensor(((feats - self.lo) / (self.hi - self.lo + 1e-12))[None], dtype=torch.float32)
        t3 = time.perf_counter_ns()
        with torch.no_grad():
            y = self.model(x).numpy()[0]
        t4 = time.perf_counter_ns()
        json.dumps({"symbol": k.symbol, "bar_time": k.open_time, "close": k.close,
                    "horizons": HORIZONS, "exp_logret": y.tolist(),
                    "exp_price": (k.close * np.exp(y)).tolist()})
        t5 = time.perf_counter_ns()
        for name, a, b in zip(STAGES, (t0, t1, t2, t3, t4), (t1, t2, t3, t4, t5), strict=True):
            t[name].append(b - a)


class After:
    def __init__(self, raw: np.ndarray, artifacts: Path):
        self.engine = OnlineFeatureEngine(1_000)
        self.engine.warmup_jit()
        self.pred = Predictor(artifacts / "model.onnx", artifacts / "model_meta.json", 1_000)
        for r in raw[:HISTORY]:  # warm state exactly like the live service does
            f = self.engine.update(r[1], r[2], r[3], r[4], r[5], r[8], r[9])
            if f is not None:
                self.pred.push(f)

    def run(self, i: int, frame: bytes, t: dict) -> None:
        t0 = time.perf_counter_ns()
        k = parse_message(frame, 0)
        t1 = time.perf_counter_ns()
        f = self.engine.update(k.open, k.high, k.low, k.close, k.volume, k.trades, k.taker_buy_volume)
        t2 = time.perf_counter_ns()
        self.pred.push(f)
        t3 = time.perf_counter_ns()
        y = self.pred.predict()
        t4 = time.perf_counter_ns()
        encode(Forecast(k.symbol, k.open_time, k.close, HORIZONS, y.tolist(),
                        (k.close * np.exp(y)).tolist(), [0.0] * 5, "FLAT", 5, 0.0, "v", 0.0, 0.0, 0.0))
        t5 = time.perf_counter_ns()
        for name, a, b in zip(STAGES, (t0, t1, t2, t3, t4), (t1, t2, t3, t4, t5), strict=True):
            t[name].append(b - a)


# --------------------------------------------------------------------------- stats
def ratio_ci(before: np.ndarray, after: np.ndarray, n: int = 2_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    nb, na = len(before), len(after)
    rs = [np.median(before[rng.integers(0, nb, nb)]) / np.median(after[rng.integers(0, na, na)])
          for _ in range(n)]
    return np.median(before) / np.median(after), np.percentile(rs, 2.5), np.percentile(rs, 97.5)


def fmt_us(ns: float) -> str:
    us = ns / 1e3
    return f"{us:,.0f} µs" if us >= 100 else f"{us:,.1f} µs"


def bar_benchmark(raw: np.ndarray, artifacts: Path, torch_threads: int) -> dict:
    torch.set_num_threads(torch_threads)
    frames = [kline_frame(r) for r in raw]
    before, after = Before(raw), After(raw, artifacts)
    tb = {s: [] for s in STAGES}
    ta = {s: [] for s in STAGES}
    for w in range(20):  # warm both paths (JIT, allocator, caches)
        before.run(HISTORY + w, frames[HISTORY + w], {s: [] for s in STAGES})
    for j in range(N_BARS):
        i = HISTORY + j
        # alternate the order every bar so neither version always runs "second"
        first, second = ((before, tb), (after, ta)) if j % 2 == 0 else ((after, ta), (before, tb))
        first[0].run(i, frames[i], first[1])
        second[0].run(i, frames[i], second[1])
    tb = {k: np.asarray(v, float) for k, v in tb.items()}
    ta = {k: np.asarray(v, float) for k, v in ta.items()}
    tb["total"] = sum(tb[s] for s in STAGES)
    ta["total"] = sum(ta[s] for s in STAGES)
    return {"before": tb, "after": ta}


def tick_benchmark(books: list[BookTicker], n_rounds: int = 3) -> dict:
    frames = [book_frame(b) for b in books]

    def before(fr):  # decode + re-serialise for the bus as JSON
        b = legacy_parse(fr, 0)
        return json.dumps({f: getattr(b, f) for f in b.__struct_fields__}).encode()

    def after(fr):
        return encode(parse_message(fr, 0))

    tb, ta = [], []
    for r in range(n_rounds):
        for j, fr in enumerate(frames):
            order = (before, after) if (j + r) % 2 == 0 else (after, before)
            for fn in order:
                t0 = time.perf_counter_ns()
                fn(fr)
                dt = time.perf_counter_ns() - t0
                (tb if fn is before else ta).append(dt)
    return {"before": np.asarray(tb, float), "after": np.asarray(ta, float),
            "json_bytes": float(np.mean([len(before(f)) for f in frames])),
            "msgpack_bytes": float(np.mean([len(out) for out in map(after, frames)])),
            "frame_bytes": float(np.mean([len(f) for f in frames]))}


def storage_benchmark(raw_full: np.ndarray, books: list[BookTicker]) -> dict:
    out = {}
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        np.save(d / "h.npy", raw_full)
        columnar.save(d / "h.qcol", raw_full)
        out["npy_mb"] = (d / "h.npy").stat().st_size / 1e6
        out["qcol_mb"] = (d / "h.qcol").stat().st_size / 1e6

        def t(fn, reps=10):
            fn()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                ts.append(time.perf_counter() - t0)
            return float(np.median(ts) * 1e3)

        out["npy_ms"] = t(lambda: np.load(d / "h.npy"))
        out["qcol_ms"] = t(lambda: columnar.load(d / "h.qcol"))
        jsonl = sum(len(book_frame(b)) + 1 for b in books)  # "before": raw frames, one per line
        w = TickLogWriter(d / "ticks", ["book"])
        for b in books:
            if w.append("book", encode(b)):
                w.write_block(w.take_block())
        w.close()
        out["tick_json_b"] = jsonl / len(books)
        out["tick_log_b"] = w.written_bytes / len(books)
    return out


def recorded_books() -> list[BookTicker]:
    from qforecast.schemas import decode_md

    tick_dir = ROOT / "data" / "ticks" / "ethusdt"
    books = []
    if tick_dir.exists():
        books = [m for t, p in read_dir(tick_dir) if t.startswith("md.book")
                 for m in [decode_md(p)]]
    if len(books) < 1_000:
        rng = np.random.default_rng(0)
        px = 2_430.0
        for i in range(5_000):
            px += float(rng.normal(0, 0.05))
            books.append(BookTicker("ETHUSDT", 80_000_000_000 + i, round(px, 2),
                                    round(float(rng.uniform(0, 20)), 4), round(px + 0.01, 2),
                                    round(float(rng.uniform(0, 20)), 4)))
    return books


# --------------------------------------------------------------------------- report
def main() -> None:
    import psutil

    raw, source = load_raw()
    books = recorded_books()
    full = columnar.load(ROOT / "data" / "ETHUSDT_1m.qcol") if (ROOT / "data" / "ETHUSDT_1m.qcol").exists() else raw
    full = np.ascontiguousarray(full)
    proc = psutil.Process()
    all_cpus = list(range(os.cpu_count()))
    pcores = parse_cpus(os.environ.get("QF_BENCH_PCORES", ""))
    policies = [("all CPUs", all_cpus)] + ([("P-cores only", pcores)] if pcores else [])

    lines = [
        "# Before vs after tuning, measured in the same run",
        "",
        f"Machine: {machine()}  ",
        f"Run: {time.strftime('%Y-%m-%d %H:%M')} · Data: {source}, {len(books):,} order-book updates · "
        f"Reproduce: `{'QF_BENCH_PCORES=' + os.environ['QF_BENCH_PCORES'] + ' ' if pcores else ''}"
        "python -m benchmarks.bench_before_after`",
        "",
        "## What is compared",
        "",
        "| Stage | Before (prototype approach) | After (tuned) |",
        "|---|---|---|",
        "| Frame decode | `orjson` → dict → `float(str)` → struct | `msgspec` decodes straight into a typed struct |",
        "| Features | pandas recomputes 14 indicators over 500 bars | incremental numba kernel, new bar only |",
        "| Normalise + model input | MinMax over the window, new tensor | z-score into a ring buffer, zero-copy view |",
        "| Model | PyTorch eager, default threads, 2-layer BiLSTM / 64 steps | ONNX Runtime, 1 thread, 1-layer BiLSTM / 32 steps |",
        "| Serialise result | `json.dumps` of a dict | msgpack struct |",
        "| Bar history on disk | `.npy` float64 | QCOL (columnar, compressed, lossless) |",
        "| Tick capture | raw JSON frames | block-compressed tick log |",
        "",
        "Method:",
        f"- {N_BARS} consecutive bars; **both versions process every bar**, alternating which one "
        "goes first, so neither always runs second.",
        "- Both versions see the same machine state (thermal, clock frequency), so the **speed-up "
        "ratio is trustworthy** even though absolute laptop timings drift between runs.",
        "- Speed-ups are ratios of medians (p50) with a **95% bootstrap confidence interval** "
        "(2,000 resamples).",
        "- Model weights are random: latency depends on the architecture, not on weight values.",
    ]

    summary = []
    for pol_name, cpus in policies:
        proc.cpu_affinity(cpus)
        with tempfile.TemporaryDirectory() as d:
            write_random_model(Path(d), LOOKBACK, HORIZONS)
            res = bar_benchmark(raw, Path(d), torch_threads=len(cpus))
        tb, ta = res["before"], res["after"]
        lines += ["", f"## Per bar (hot path), {pol_name} ({len(cpus)} logical CPUs)", "",
                  "| Stage | Before p50 | After p50 | Before p99 | After p99 | Speed-up (95% CI) |",
                  "|---|---:|---:|---:|---:|---:|"]
        for s in STAGES + ["total"]:
            r, lo, hi = ratio_ci(tb[s], ta[s])
            lines.append(f"| {STAGE_EN[s]} | {fmt_us(np.median(tb[s]))} | {fmt_us(np.median(ta[s]))} | "
                         f"{fmt_us(np.percentile(tb[s], 99))} | {fmt_us(np.percentile(ta[s], 99))} | "
                         f"**{r:,.1f}×** ({lo:,.1f}–{hi:,.1f}) |")
        share = {s: np.median(ta[s]) / np.median(ta["total"]) * 100 for s in STAGES}
        bshare = {s: np.median(tb[s]) / np.median(tb["total"]) * 100 for s in STAGES}
        lines.append("")
        lines.append("Where the time goes, before: " + " · ".join(
            f"{STAGE_EN[s].lower()} {bshare[s]:.0f}%" for s in STAGES) + "  ")
        lines.append("Where the time goes, after: " + " · ".join(
            f"{STAGE_EN[s].lower()} {share[s]:.0f}%" for s in STAGES))
        summary.append((pol_name, np.median(tb["total"]), np.median(ta["total"])))
    proc.cpu_affinity(all_cpus)

    tk = tick_benchmark(books)
    r, lo, hi = ratio_ci(tk["before"], tk["after"])
    lines += ["", "## Per order-book update (decode + serialise for the bus)", "",
              "| | Before | After | Speed-up (95% CI) |", "|---|---:|---:|---:|",
              f"| Latency p50 | {fmt_us(np.median(tk['before']))} | {fmt_us(np.median(tk['after']))} | "
              f"**{r:.1f}×** ({lo:.1f}–{hi:.1f}) |",
              f"| Latency p99 | {fmt_us(np.percentile(tk['before'], 99))} | "
              f"{fmt_us(np.percentile(tk['after'], 99))} | |",
              f"| Throughput (1 thread) | {1e9 / np.median(tk['before']):,.0f} msg/s | "
              f"{1e9 / np.median(tk['after']):,.0f} msg/s | |",
              f"| Message size on the bus | {tk['json_bytes']:.0f} B (JSON) | "
              f"{tk['msgpack_bytes']:.0f} B (msgpack) | {tk['json_bytes'] / tk['msgpack_bytes']:.1f}× smaller |"]

    st = storage_benchmark(full, books)
    lines += ["", "## Storage", "",
              "| | Before | After | Change |", "|---|---:|---:|---:|",
              f"| Bar history file ({len(full):,} bars) | {st['npy_mb']:.2f} MB (.npy) | {st['qcol_mb']:.2f} MB (QCOL) | "
              f"{st['npy_mb'] / st['qcol_mb']:.1f}× smaller |",
              f"| Read bar history | {st['npy_ms']:.1f} ms | {st['qcol_ms']:.1f} ms | "
              f"{st['npy_ms'] / st['qcol_ms']:.2f}× (deliberate trade: slower read, smaller file) |",
              f"| Tick capture, per order-book update | {st['tick_json_b']:.0f} B (JSON frame) | "
              f"{st['tick_log_b']:.1f} B (tick log) | {st['tick_json_b'] / st['tick_log_b']:.1f}× smaller |"]

    lines += ["", "## Summary", "", "| CPU policy | Before, per bar | After, per bar | Speed-up |",
              "|---|---:|---:|---:|"]
    for pol, b, a in summary:
        lines.append(f"| {pol} | {fmt_us(b)} | {fmt_us(a)} | **{b / a:,.0f}×** |")
    if len(summary) > 1:
        b0 = summary[0][1]
        a1 = summary[1][2]
        lines.append(f"| before on all CPUs → after on P-cores | {fmt_us(b0)} | {fmt_us(a1)} | **{b0 / a1:,.0f}×** |")
    lines += [
        "", "## Reading the results", "",
        "- **Features gain less here than in the micro-benchmark** ([RESULTS.md](RESULTS.md)). "
        "The versions are interleaved, so the tuned path runs right after pandas and PyTorch have "
        "evicted its data from cache; the micro-benchmark measures a hot loop. These numbers are "
        "closer to live operation, where each bar is also interleaved with other work.",
        "- **The speed-up differs between \"all CPUs\" and \"P-cores only\" mostly because the "
        "*before* version changes**: multi-threaded PyTorch eager benefits from fewer threads, "
        "while the single-threaded ONNX Runtime path barely moves.",
        "- **After tuning, inference dominates the remaining time.** That is the next target "
        "(keeping the core warm between bars, a smaller model, or O(1) streaming inference).",
        "- **Slower history reads are a deliberate trade**: a few milliseconds at trainer start-up "
        "for a ~4× smaller file.",
        "- **Not included:** network latency from the exchange and the Redis hop of the distributed "
        "mode (not measurable on this machine).",
    ]
    out = Path(__file__).with_name("BEFORE_AFTER.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
