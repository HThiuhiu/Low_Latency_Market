"""Selective-scan backends: memory kept for backward + forward/backward time.

    python -m benchmarks.bench_scan                # CPU: parallel vs chunked
    python -m benchmarks.bench_scan --device cuda  # + fused triton kernel, peak GPU memory

"Saved for backward" is measured with torch.autograd.graph.saved_tensors_hooks, so it is
device-independent: it is exactly what autograd keeps alive between forward and
backward, i.e. the activation memory (and the HBM traffic on a GPU) of the scan.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qforecast.model.scan_fused import HAS_TRITON, selective_scan  # noqa: E402

# (batch, length, channels, state): our training shape, then longer contexts.
SHAPES = [(512, 32, 64, 8), (64, 256, 64, 16), (16, 1024, 64, 16)]


def make(shape, device):
    bsz, L, D, N = shape
    g = torch.Generator().manual_seed(0)
    x = torch.randn(bsz, L, D, generator=g)
    dt = torch.rand(bsz, L, D, generator=g) * 0.1 + 1e-3
    A = -torch.rand(D, N, generator=g) - 0.5
    B, C, Dp = torch.randn(bsz, L, N, generator=g), torch.randn(bsz, L, N, generator=g), torch.ones(D)
    return [t.to(device).requires_grad_() for t in (x, dt, A, B, C, Dp)]


def saved_bytes(fn, args) -> int:
    seen: dict[tuple, int] = {}

    def pack(t):
        key = (t.data_ptr(), tuple(t.shape), t.dtype)
        seen[key] = t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn(*args)
    return sum(seen.values())


def timed(fn, args, device, reps: int) -> tuple[float, float]:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    y = fn(*args)
    y.sum().backward()  # warm-up / triton compile
    sync()
    f = b = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        y = fn(*args)
        sync()
        t1 = time.perf_counter()
        y.sum().backward()
        sync()
        f += t1 - t0
        b += time.perf_counter() - t1
    return f / reps * 1e3, b / reps * 1e3


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()
    device = torch.device(args.device)
    backends = ["parallel", "chunked"] + (["triton"] if device.type == "cuda" and HAS_TRITON else [])
    print(f"device={device}  backends={backends}\n")
    print("| Shape (B, L, D, N) | Backend | Saved for backward | Forward (ms) | Backward (ms) |"
          + (" Peak GPU mem |" if device.type == "cuda" else ""))
    print("|---|---|---:|---:|---:|" + ("---:|" if device.type == "cuda" else ""))
    for shape in SHAPES:
        for be in backends:
            inp = make(shape, device)

            def fn(*a, be=be):
                return selective_scan(*a, backend=be)

            mem = saved_bytes(fn, inp)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            try:
                fwd, bwd = timed(fn, inp, device, args.reps)
            except RuntimeError as e:  # e.g. out of memory for the materialising backend
                print(f"| {shape} | {be} | {mem / 2**20:,.1f} MiB | failed: {str(e)[:40]} | |")
                continue
            peak = f" {torch.cuda.max_memory_allocated() / 2**20:,.0f} MiB |" if device.type == "cuda" else ""
            print(f"| {shape} | {be} | {mem / 2**20:,.1f} MiB | {fwd:,.1f} | {bwd:,.1f} |{peak}")


if __name__ == "__main__":
    main()
