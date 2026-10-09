"""Plain PyTorch training throughput vs intra-op thread count (CPU).

    python -m benchmarks.bench_train_threads

The default (torch picks the number of physical cores) is compared with explicit thread
counts. On a hybrid CPU more threads is not automatically faster: small kernels pay
thread-pool synchronisation, and threads that land on E-cores slow every parallel
region down to the speed of the slowest thread.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qforecast.core.features import N_FEATURES  # noqa: E402
from qforecast.model.net import build_model  # noqa: E402
from qforecast.trainer.dataset import WindowBatcher  # noqa: E402

BATCH, LOOKBACK, STEPS, REPEATS = 512, 128, 8, 3


def step_time(threads: int) -> float:
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    z = rng.normal(size=(20_000, N_FEATURES)).astype(np.float32)
    y = rng.normal(size=(20_000, 5)).astype(np.float32)
    batcher = WindowBatcher(z, y, LOOKBACK)
    model = build_model("lstm", N_FEATURES, 5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.HuberLoss()
    idx = rng.integers(LOOKBACK, len(z), size=(STEPS + 3, BATCH))

    def one(i):
        xb, yb = batcher.batch(idx[i], "cpu")
        opt.zero_grad(set_to_none=True)
        loss_fn(model(xb), yb).backward()
        opt.step()

    for i in range(3):  # warm-up
        one(i)
    t0 = time.perf_counter()
    for i in range(3, STEPS + 3):
        one(i)
    return (time.perf_counter() - t0) / STEPS


def main() -> None:
    default = torch.get_num_threads()
    counts = sorted({1, 2, 4, 6, 8, 12, default, 20})
    rows = []
    for t in counts:
        ms = [step_time(t) * 1e3 for _ in range(REPEATS)]
        rows.append((t, statistics.median(ms), min(ms), max(ms)))
        print(f"threads={t}: {statistics.median(ms):.0f} ms/step", flush=True)
    best = min(r[1] for r in rows)
    d = next(r for r in rows if r[0] == default)
    print("\n| Threads | ms per training step, median (min–max) | Samples/s | vs default |")
    print("|---:|---:|---:|---:|")
    for t, med, lo, hi in rows:
        tag = " (torch default)" if t == default else ""
        print(f"| {t}{tag} | {med:,.0f} ({lo:,.0f}–{hi:,.0f}) | {BATCH / med * 1e3:,.0f} | "
              f"{d[1] / med:.2f}× |")
    print(f"\nBest setting is {d[1] / best:.2f}× faster than the torch default ({default} threads).")


if __name__ == "__main__":
    main()
