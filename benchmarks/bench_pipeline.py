"""End-to-end replay through the real service code over the in-memory bus.

A synthetic market (bars + ~100 book updates and ~20 trades per bar) is pushed
through the *same* ``signal.run`` and ``analytics.run`` coroutines used in
production; latencies come from the services' own histograms.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qforecast.bus import MemoryBus  # noqa: E402
from qforecast.config import Settings  # noqa: E402
from qforecast.core.metrics import Registry  # noqa: E402
from qforecast.schemas import BookTicker, Kline, Trade, encode  # noqa: E402
from qforecast.services import analytics, signal  # noqa: E402
from tests.conftest import synthetic_bars, write_random_model  # noqa: E402

N_BARS = 1_500
BOOK_PER_BAR = 100
TRADES_PER_BAR = 20


async def replay(s: Settings) -> dict:
    bus = MemoryBus(retention=10, queue_size=100_000)
    reg_sig, reg_an = Registry("signal"), Registry("analytics")
    tasks = [asyncio.create_task(signal.run(s, bus, reg_sig)),
             asyncio.create_task(analytics.run(s, bus, reg_an))]
    await asyncio.sleep(0.5)  # services start, JIT warm-up, model load

    bars = synthetic_bars(N_BARS, seed=1)
    rng = np.random.default_rng(2)
    t_ms = int(time.time() * 1000) - N_BARS * s.interval_ms
    n_msgs = 0
    t0 = time.perf_counter()
    for i, b in enumerate(bars):
        o, h, lo, c, v, nt, tb = b.tolist()
        for j in range(BOOK_PER_BAR):
            px = c * (1 + rng.normal(0, 1e-4))
            await bus.publish(s.t_book, encode(BookTicker(s.symbol, i * 1000 + j, px - 0.01,
                                                          float(rng.uniform(1, 10)), px + 0.01,
                                                          float(rng.uniform(1, 10)), time.perf_counter_ns())))
            n_msgs += 1
            if j % (BOOK_PER_BAR // TRADES_PER_BAR) == 0:
                await bus.publish(s.t_trade, encode(Trade(s.symbol, px, 0.5, bool(j % 2), t_ms + j * 500,
                                                          t_ms + j * 500, time.perf_counter_ns())))
                n_msgs += 1
            if j % 10 == 0:
                await asyncio.sleep(0)  # let consumers run, as network IO would
        t_ms += s.interval_ms
        # close_time = "now" so the signal service treats each replayed bar as live
        k = Kline(s.symbol, t_ms - s.interval_ms, int(time.time() * 1000), o, h, lo, c, v, int(nt), tb,
                  0, time.perf_counter_ns())
        await bus.publish(s.t_kline, encode(k))
        n_msgs += 1
        await asyncio.sleep(0)
    while reg_an.counters.get("messages", 0) < n_msgs:  # drain
        await asyncio.sleep(0.01)
    elapsed = time.perf_counter() - t0
    for t in tasks:
        t.cancel()
    return {
        "messages": n_msgs, "elapsed_s": elapsed, "throughput_msgs_s": n_msgs / elapsed,
        "forecasts": int(reg_sig.counters.get("forecasts", 0)), "dropped": bus.dropped,
        "signal": {k: h.summary_us() for k, h in reg_sig.histograms.items()},
        "analytics": {k: h.summary_us() for k, h in reg_an.histograms.items()},
    }


NAMES = {"features": "Feature update (per bar)", "inference": "ONNX inference",
         "pipeline": "Bar published → forecast published",
         "analysis_update": "Analytics state update (per msg)",
         "analysis_snapshot": "Analytics snapshot"}


def run() -> str:
    """Run under several CPU-affinity policies (QF_BENCH_PCORES / QF_BENCH_PIN)."""
    import os

    import psutil

    from benchmarks.common import parse_cpus

    configs = [("all CPUs", list(range(os.cpu_count())))]
    if os.environ.get("QF_BENCH_PCORES"):
        configs.append(("P-cores only", parse_cpus(os.environ["QF_BENCH_PCORES"])))
    if os.environ.get("QF_BENCH_PIN"):
        configs.append((f"pinned to CPU {os.environ['QF_BENCH_PIN']}",
                        parse_cpus(os.environ["QF_BENCH_PIN"])))
    results = []
    before = psutil.Process().cpu_affinity()
    with tempfile.TemporaryDirectory() as d:
        base = Settings()
        write_random_model(Path(d), base.lookback, list(base.horizons))
        s = replace(base, artifacts_dir=Path(d), analysis_publish_interval_ms=250)
        try:
            for label, cpus in configs:
                psutil.Process().cpu_affinity(cpus)
                results.append((label, asyncio.run(replay(s))))
        finally:
            psutil.Process().cpu_affinity(before)

    lines = []
    for label, r in results:
        lines.append(f"- **{label}**: {r['messages']:,} messages ({N_BARS} bars, {BOOK_PER_BAR} book "
                     f"updates + {TRADES_PER_BAR} trades per bar) in {r['elapsed_s']:.2f}s → "
                     f"**{r['throughput_msgs_s']:,.0f} msgs/s** on one event loop, "
                     f"{r['forecasts']} forecasts, {r['dropped']} dropped.")
    header = "| Stage | " + " | ".join(f"{lbl} p50 / p99 (µs)" for lbl, _ in results) + " |"
    lines += ["", header, "|---|" + "---:|" * len(results)]
    for svc in ("signal", "analytics"):
        for k in results[0][1][svc]:
            cells = []
            for _, r in results:
                v = r[svc][k]
                cells.append(f"{v['p50_us']:,} / {v['p99_us']:,}" if v["count"] else "—")
            lines.append(f"| {NAMES.get(k, k)} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(run())
