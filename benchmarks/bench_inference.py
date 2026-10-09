"""Batch-1 inference latency: PyTorch eager vs TorchScript vs ONNX Runtime."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import table, timeit, usable_cpus  # noqa: E402
from qforecast.core.features import N_FEATURES  # noqa: E402
from qforecast.model.net import CNNBiLSTMAttention  # noqa: E402
from qforecast.model.predictor import make_session  # noqa: E402

L, H = 128, 5


def _onnx(model, x_t, path: Path) -> Path:
    torch.onnx.export(model, (x_t,), str(path), input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)
    return path


def _time_ort(path: Path, threads: int, x: np.ndarray) -> tuple[dict, np.ndarray]:
    sess = make_session(path, threads)
    return timeit(lambda: sess.run(None, {"x": x}), n=2_000), sess.run(None, {"x": x})[0]


def run() -> str:
    torch.manual_seed(0)
    model = CNNBiLSTMAttention(N_FEATURES, H).eval()
    x_np = np.random.default_rng(0).normal(size=(1, L, N_FEATURES)).astype(np.float32)
    x_t = torch.from_numpy(x_np)
    rows = []

    # First-iteration architecture: 2-layer BiLSTM over 64 steps (sequence halved once).
    legacy = CNNBiLSTMAttention(N_FEATURES, H, lstm_layers=2, downsample=2).eval()
    with tempfile.TemporaryDirectory() as d:
        timing, _ = _time_ort(_onnx(legacy, x_t, Path(d) / "legacy.onnx"), 1, x_np)
        rows.append(("ONNX RT, 1 thread — 2-layer BiLSTM / 64 steps", timing))

    ncpu = usable_cpus()
    for threads in (ncpu, 1):
        torch.set_num_threads(threads)
        with torch.inference_mode():
            rows.append((f"PyTorch eager ({threads} threads)", timeit(lambda: model(x_t), n=500)))
    with torch.inference_mode():
        ts = torch.jit.freeze(torch.jit.trace(model, x_t))
        rows.append(("TorchScript frozen (1 thread)", timeit(lambda: ts(x_t), n=500)))

    with tempfile.TemporaryDirectory() as d:
        path = _onnx(model, x_t, Path(d) / "m.onnx")
        for threads in (ncpu, 1):
            timing, out = _time_ort(path, threads, x_np)
            rows.append((f"ONNX Runtime ({threads} threads)", timing))
        diff = float(np.abs(out - model(x_t).detach().numpy()).max())

    n_params = sum(p.numel() for p in model.parameters())
    return (f"Model: {n_params:,} params, input (1, {L}, {N_FEATURES}); "
            f"ONNX vs PyTorch max |diff| = {diff:.1e}\n\n"
            + table(rows, baseline=f"PyTorch eager ({ncpu} threads)"))


if __name__ == "__main__":
    print(run())
