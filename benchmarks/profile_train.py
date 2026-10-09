"""Profile one training step with torch.profiler: where does the time go?

    python -m benchmarks.profile_train                       # CPU or CUDA, auto-detected
    python -m benchmarks.profile_train --device cuda --data-on-device --compile

Each step is split into labelled phases (data / forward / backward / optimizer), so the
report answers "which phase" first, then "which kernel". On CUDA it also reports the
GPU-side kernel time and the host-side launch overhead; a Chrome trace is written to
benchmarks/traces/ (open it at chrome://tracing or https://ui.perfetto.dev).

Variants worth comparing on a GPU box (run each, diff the "phase" tables):
    baseline                         batch gathered on CPU, copied over PCIe every step
    --data-on-device                 feature matrix lives on the GPU, no per-step PCIe copy
    --data-on-device --compile       + TorchInductor kernel fusion
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function, schedule

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qforecast.core.features import N_FEATURES  # noqa: E402
from qforecast.model.net import build_model  # noqa: E402
from qforecast.trainer.dataset import WindowBatcher  # noqa: E402

PHASES = ("data", "forward", "backward", "optimizer")


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build(args, device):
    rng = np.random.default_rng(0)
    n_bars, n_out = 129_600, 5  # same size as the 90-day training set
    z = rng.normal(size=(n_bars, N_FEATURES)).astype(np.float32)
    y = rng.normal(size=(n_bars, n_out)).astype(np.float32)
    batcher = WindowBatcher(z, y, args.lookback)
    if args.data_on_device:  # 7 MB: put the whole feature matrix on the device once
        batcher.z, batcher.y = batcher.z.to(device), batcher.y.to(device)
        batcher.offsets = batcher.offsets.to(device)
    kw = {"scan_backend": args.scan_backend} if args.arch == "mamba" else {}
    model = build_model(args.arch, N_FEATURES, n_out, **kw).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    fwd = torch.compile(model) if args.compile else model
    idx_pool = rng.integers(args.lookback, n_bars, size=(1024, args.batch_size))
    return model, fwd, opt, batcher, idx_pool, nn.HuberLoss()


def train_step(fwd, model, opt, loss_fn, batcher, idx, device, use_amp):
    with record_function("data"):
        x, y = batcher.batch(idx, device)
    with record_function("forward"):
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            loss = loss_fn(fwd(x), y)
    with record_function("backward"):
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    with record_function("optimizer"):
        opt.step()
    return loss


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="auto")
    p.add_argument("--arch", default="lstm", choices=["lstm", "mamba"])
    p.add_argument("--scan-backend", default="auto", choices=["auto", "parallel", "chunked", "triton"],
                   help="mamba only (auto = triton on CUDA)")
    p.add_argument("--cpus", default=None, help="pin the process to these logical CPUs, e.g. 0-11")
    p.add_argument("--steps", type=int, default=20, help="profiled steps")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lookback", type=int, default=128)
    p.add_argument("--top", type=int, default=15, help="rows in the per-operator table")
    p.add_argument("--data-on-device", action="store_true")
    p.add_argument("--compile", action="store_true", help="torch.compile the forward pass")
    p.add_argument("--threads", type=int, default=None,
                   help="torch intra-op threads on CPU (trainer uses os.cpu_count())")
    p.add_argument("--amp", action="store_true", help="fp16 autocast (CUDA only)")
    p.add_argument("--trace", default=None, help="Chrome trace path (default: benchmarks/traces/...)")
    args = p.parse_args()

    device = pick_device(args.device)
    if args.cpus:
        import psutil

        from benchmarks.common import parse_cpus

        cpus = parse_cpus(args.cpus)
        psutil.Process().cpu_affinity(cpus)
        if not args.threads:  # torch sized its pool before pinning; match the allowed CPUs
            torch.set_num_threads(len(cpus))
    if args.threads:
        torch.set_num_threads(args.threads)
    use_amp = args.amp and device.type == "cuda"
    cuda = device.type == "cuda"
    torch.manual_seed(0)
    model, fwd, opt, batcher, idx_pool, loss_fn = build(args, device)
    model.train()

    def step(i):
        train_step(fwd, model, opt, loss_fn, batcher, torch.as_tensor(idx_pool[i % len(idx_pool)]),
                   device, use_amp)

    # Warm-up outside the profiler: allocator, cuDNN autotune, torch.compile, lazy init.
    for i in range(10):
        step(i)
    if cuda:
        torch.cuda.synchronize()

    # Wall-clock per step without profiler overhead (the honest number).
    t0 = time.perf_counter()
    for i in range(args.steps):
        step(i)
    if cuda:
        torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - t0) / args.steps * 1e3

    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    tag = f"{args.arch}_{args.scan_backend}_{device.type}" + ("_ondev" if args.data_on_device else "") + (
        "_compile" if args.compile else "")
    trace = Path(args.trace) if args.trace else Path(__file__).with_name("traces") / f"train_{tag}.json"
    trace.parent.mkdir(exist_ok=True)

    with profile(activities=activities, schedule=schedule(wait=1, warmup=2, active=args.steps),
                 record_shapes=True, profile_memory=True) as prof:
        for i in range(args.steps + 3):
            step(i)
            if cuda:
                torch.cuda.synchronize()
            prof.step()
    prof.export_chrome_trace(str(trace))

    key = "cuda_time_total" if cuda else "cpu_time_total"
    scan = args.scan_backend if args.arch == "mamba" else "-"
    print(f"\narch={args.arch}  scan={scan}  cpus={args.cpus or 'all'}  threads={torch.get_num_threads()}  "
          f"device={device}  batch={args.batch_size}  lookback={args.lookback}  "
          f"data_on_device={args.data_on_device}  compile={args.compile}  amp={use_amp}")
    print(f"wall-clock per step (no profiler): {wall_ms:.2f} ms  "
          f"-> {args.batch_size / wall_ms * 1e3:,.0f} samples/s")

    events = {e.key: e for e in prof.key_averages()}
    rows, total = [], 0.0
    for ph in PHASES:
        e = events.get(ph)
        if e is None:
            continue
        host_ms = e.cpu_time_total / e.count / 1e3
        dev_ms = (e.device_time_total / e.count / 1e3) if cuda else 0.0
        rows.append((ph, host_ms, dev_ms))
        total += host_ms
    print("\nPhase breakdown (per step; host = CPU time incl. waiting, device = GPU kernel time)")
    print(f"{'phase':<10}{'host ms':>10}{'%':>7}{'gpu ms':>10}")
    for ph, h, d in rows:
        print(f"{ph:<10}{h:>10.2f}{h / total * 100:>6.0f}%{d:>10.2f}")

    print(f"\nTop {args.top} operators by {'GPU' if cuda else 'CPU'} time")
    print(prof.key_averages().table(sort_by=key, row_limit=args.top, max_name_column_width=48))

    if cuda:
        k = [e for e in prof.key_averages() if e.device_time_total > 0 and e.key not in PHASES]
        n_launch = sum(e.count for e in k)
        dev_total = sum(e.self_device_time_total for e in k) / args.steps / 1e3
        print(f"GPU kernel launches per step: {n_launch / args.steps:.0f}; "
              f"summed kernel time {dev_total:.2f} ms vs wall {wall_ms:.2f} ms "
              f"-> GPU busy {min(dev_total / wall_ms, 1) * 100:.0f}% "
              "(low = launch/PCIe-bound, high = compute-bound)")
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    print(f"\nChrome trace: {trace}")


if __name__ == "__main__":
    main()
