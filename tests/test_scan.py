"""Selective scan: every backend must match the reference, forward and backward."""

import numpy as np
import pytest
import torch

from qforecast.model.scan_core import parallel_scan, scan_sequential
from qforecast.model.scan_fused import (
    HAS_TRITON,
    _ChunkedScan,
    selective_scan,
    selective_scan_chunked,
    selective_scan_parallel,
)
from qforecast.model.ssm import SelectiveSSM


def _inputs(bsz=2, L=11, D=3, N=4, dtype=torch.float64, device="cpu", seed=0):
    g = torch.Generator().manual_seed(seed)

    def r(*s):
        return torch.randn(*s, generator=g, dtype=dtype)

    x = r(bsz, L, D)
    dt = torch.rand(bsz, L, D, generator=g, dtype=dtype) * 0.5 + 0.05
    A = -torch.rand(D, N, generator=g, dtype=dtype) * 2 - 0.1
    B, C, Dp = r(bsz, L, N), r(bsz, L, N), r(D)
    return [t.to(device).requires_grad_() for t in (x, dt, A, B, C, Dp)]


@pytest.mark.parametrize("L", [1, 7, 32, 100])
def test_parallel_scan_matches_sequential(L):
    a = torch.rand(2, L, 4, 3, dtype=torch.float64) * 0.9 + 0.05
    b = torch.randn(2, L, 4, 3, dtype=torch.float64)
    torch.testing.assert_close(parallel_scan(a, b), scan_sequential(a, b))


@pytest.mark.parametrize("chunk", [1, 3, 4, 11, 16])
def test_chunked_matches_reference_forward_and_backward(chunk):
    args = _inputs()
    y_ref = selective_scan_parallel(*args)
    y = selective_scan_chunked(*args, chunk=chunk)
    torch.testing.assert_close(y, y_ref)
    g = torch.randn_like(y_ref)
    for got, ref in zip(torch.autograd.grad(y, args, g), torch.autograd.grad(y_ref, args, g), strict=True):
        torch.testing.assert_close(got, ref)


def test_chunked_gradcheck():
    args = _inputs(1, 9, 2, 2)
    assert torch.autograd.gradcheck(lambda *a: _ChunkedScan.apply(*a, 4), args)


def test_streaming_step_equals_full_forward():
    torch.manual_seed(0)
    m = SelectiveSSM(16, d_state=4).eval()
    u = torch.randn(2, 20, 16)
    conv = torch.zeros(2, m.d_inner, m.d_conv - 1)
    h = torch.zeros(2, m.d_inner, m.d_state)
    outs = []
    for t in range(20):
        o, conv, h = m.step(u[:, t], conv, h)
        outs.append(o)
    torch.testing.assert_close(torch.stack(outs, 1), m(u), atol=1e-6, rtol=1e-5)


def test_mamba_model_exports_to_onnx(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from qforecast.model.net import build_model

    torch.manual_seed(0)
    net = build_model("mamba", 14, 5).eval()
    x = torch.randn(1, 128, 14)
    torch.onnx.export(net, (x,), str(tmp_path / "m.onnx"), input_names=["x"], output_names=["y"],
                      opset_version=17, dynamo=False)  # "auto" resolves to the parallel scan
    sess = ort.InferenceSession(str(tmp_path / "m.onnx"), providers=["CPUExecutionProvider"])
    np.testing.assert_allclose(sess.run(None, {"x": x.numpy()})[0], net(x).detach().numpy(), atol=1e-5)


cuda_triton = pytest.mark.skipif(not (HAS_TRITON and torch.cuda.is_available()),
                                 reason="needs an NVIDIA GPU and triton")


@cuda_triton
@pytest.mark.parametrize("shape", [(2, 32, 16, 8), (3, 50, 20, 8), (2, 100, 64, 16), (1, 1, 5, 4)])
def test_triton_matches_chunked_reference(shape):
    bsz, L, D, N = shape
    args = _inputs(bsz, L, D, N, dtype=torch.float32, device="cuda")
    ref_args = [a.detach().double().cpu().requires_grad_() for a in args]
    y = selective_scan(*args, backend="triton")
    y_ref = selective_scan_chunked(*ref_args)
    torch.testing.assert_close(y.double().cpu(), y_ref, atol=1e-4, rtol=1e-4)
    g = torch.randn_like(y)
    grads = torch.autograd.grad(y, args, g)
    grads_ref = torch.autograd.grad(y_ref, ref_args, g.double().cpu())
    for name, got, ref in zip("x dt A B C D".split(), grads, grads_ref, strict=True):
        torch.testing.assert_close(got.double().cpu(), ref, atol=2e-3, rtol=2e-3, msg=name)


@cuda_triton
def test_triton_under_autocast_runs_in_fp32():
    args = _inputs(4, 64, 32, 8, dtype=torch.float32, device="cuda")
    with torch.autocast("cuda", dtype=torch.float16):
        y = selective_scan(*args, backend="triton")
    assert y.dtype == torch.float32 and torch.isfinite(y).all()
