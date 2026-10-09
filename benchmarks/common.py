"""Benchmark helpers: tight timing loops reporting p50/p99 in microseconds."""

from __future__ import annotations

import gc
import platform
import time
from collections.abc import Callable

import numpy as np


def timeit(fn: Callable[[], object], n: int = 2_000, warmup: int = 200) -> dict:
    for _ in range(warmup):
        fn()
    samples = np.empty(n, dtype=np.int64)
    gc.disable()
    try:
        for i in range(n):
            t0 = time.perf_counter_ns()
            fn()
            samples[i] = time.perf_counter_ns() - t0
    finally:
        gc.enable()
    us = samples / 1e3
    return {"p50_us": float(np.percentile(us, 50)), "p99_us": float(np.percentile(us, 99)),
            "mean_us": float(us.mean())}


def table(rows: list[tuple[str, dict]], baseline: str | None = None) -> str:
    base = dict(rows).get(baseline, {}).get("p50_us") if baseline else None
    out = ["| Variant | p50 (µs) | p99 (µs) | Speed-up (p50) |", "|---|---:|---:|---:|"]
    for name, r in rows:
        sp = f"{base / r['p50_us']:.1f}x" if base else "—"
        out.append(f"| {name} | {r['p50_us']:,.2f} | {r['p99_us']:,.2f} | {sp} |")
    return "\n".join(out)


def parse_cpus(spec: str) -> list[int]:
    """'0-3,8' -> [0, 1, 2, 3, 8]"""
    out: list[int] = []
    for part in filter(None, spec.split(",")):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return out


def usable_cpus() -> int:
    try:
        import psutil

        return len(psutil.Process().cpu_affinity())
    except Exception:
        import os

        return os.cpu_count() or 1


def machine() -> str:
    import os

    return f"{platform.processor() or platform.machine()} · {os.cpu_count()} logical CPUs · " \
           f"{platform.system()} {platform.release()} · Python {platform.python_version()}"
