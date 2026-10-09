"""Do the two optimisations compound? Scan backend x CPU-affinity policy.

    QF_BENCH_PCORES=0-11 QF_BENCH_ECORES=14-19 python -m benchmarks.bench_combo

Workloads (CPU):
  * scan       selective scan alone, forward + backward, training shape (512, 32, 64, 8)
  * train step full Mamba model step (forward + backward + AdamW), batch 512
  * inference  batch-1 ONNX Runtime forward (Mamba with the parallel scan, and LSTM)

Policies: all CPUs (OS decides, P+E), P-cores only, E-cores only (for contrast).
For each workload the table reports the speed-up of every cell against the baseline
(parallel scan, all CPUs) and, for the combined cell, the product of the two individual
speed-ups, to show whether the gains multiply.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import parse_cpus  # noqa: E402
from qforecast.core.features import N_FEATURES  # noqa: E402
from qforecast.model.net import build_model  # noqa: E402
from qforecast.model.predictor import make_session  # noqa: E402
from qforecast.model.scan_fused import selective_scan  # noqa: E402
from qforecast.model.ssm import SelectiveSSM  # noqa: E402


def policies() -> dict[str, list[int]]:
    out = {"all CPUs": list(range(os.cpu_count()))}
    if os.environ.get("QF_BENCH_PCORES"):
        out["P-cores only"] = parse_cpus(os.environ["QF_BENCH_PCORES"])
    if os.environ.get("QF_BENCH_ECORES"):
        out["E-cores only"] = parse_cpus(os.environ["QF_BENCH_ECORES"])
    return out


def median_ms(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def scan_workload(backend: str):
    g = torch.Generator().manual_seed(0)
    bsz, L, D, N = 512, 32, 64, 8
    x = torch.randn(bsz, L, D, generator=g, requires_grad=True)
    dt = (torch.rand(bsz, L, D, generator=g) * 0.1 + 1e-3).requires_grad_()
    A = (-torch.rand(D, N, generator=g) - 0.5).requires_grad_()
    B = torch.randn(bsz, L, N, generator=g, requires_grad=True)
    C = torch.randn(bsz, L, N, generator=g, requires_grad=True)
    Dp = torch.ones(D, requires_grad=True)

    def run():
        selective_scan(x, dt, A, B, C, Dp, backend=backend).sum().backward()

    return run


def train_workload(backend: str):
    torch.manual_seed(0)
    model = build_model("mamba", N_FEATURES, 5)
    for m in model.modules():
        if isinstance(m, SelectiveSSM):
            m.scan_backend = backend
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = torch.randn(512, 128, N_FEATURES), torch.randn(512, 5)
    loss_fn = torch.nn.HuberLoss()

    def run():
        opt.zero_grad(set_to_none=True)
        loss_fn(model(x), y).backward()
        opt.step()

    return run


def onnx_session(arch: str, d: Path):
    torch.manual_seed(0)
    net = build_model(arch, N_FEATURES, 5).eval()
    x = torch.zeros(1, 128, N_FEATURES)
    path = d / f"{arch}.onnx"
    torch.onnx.export(net, (x,), str(path), input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    return path


def main() -> None:
    pols = policies()
    proc = psutil.Process()
    original = proc.cpu_affinity()
    results: dict[tuple[str, str, str], float] = {}
    tmp = Path(tempfile.mkdtemp())
    onnx = {a: onnx_session(a, tmp) for a in ("mamba", "lstm")}
    xin = np.random.default_rng(0).normal(size=(1, 128, N_FEATURES)).astype(np.float32)
    try:
        for pol, cpus in pols.items():
            proc.cpu_affinity(cpus)
            torch.set_num_threads(len(cpus))
            print(f"[{pol}] cpus={cpus[0]}-{cpus[-1]} threads={len(cpus)}", flush=True)
            for be in ("parallel", "chunked"):
                results[("scan fwd+bwd (512, 32, 64, 8)", be, pol)] = median_ms(scan_workload(be), 7)
                results[("Mamba train step, batch 512", be, pol)] = median_ms(train_workload(be), 5)
            for arch in ("mamba", "lstm"):
                sess = make_session(onnx[arch], 1)
                fn = lambda s=sess: s.run(None, {"x": xin})  # noqa: E731
                results[(f"ONNX inference batch 1 ({arch})", "parallel" if arch == "mamba" else "—",
                         pol)] = median_ms(fn, 500, 50)
    finally:
        proc.cpu_affinity(original)

    pol_names = list(pols)
    for wl in dict.fromkeys(k[0] for k in results):
        cells = {(be, pol): v for (w, be, pol), v in results.items() if w == wl}
        bes = list(dict.fromkeys(be for be, _ in cells))
        base = cells[(bes[0], "all CPUs")]
        print(f"\n### {wl}  (baseline: {bes[0]}, all CPUs = {base:,.2f} ms)\n")
        print("| Backend | " + " | ".join(pol_names) + " |")
        print("|---|" + "---:|" * len(pol_names))
        for be in bes:
            row = []
            for pol in pol_names:
                v = cells[(be, pol)]
                row.append(f"{v:,.2f} ms ({base / v:.2f}×)")
            print(f"| {be} | " + " | ".join(row) + " |")
        if "chunked" in bes and "P-cores only" in pol_names:
            s_pc = base / cells[("parallel", "P-cores only")]
            s_be = base / cells[("chunked", "all CPUs")]
            s_both = base / cells[("chunked", "P-cores only")]
            print(f"\nP-cores alone {s_pc:.2f}× · chunked scan alone {s_be:.2f}× · "
                  f"combined measured {s_both:.2f}× vs product {s_pc * s_be:.2f}×")


if __name__ == "__main__":
    main()
