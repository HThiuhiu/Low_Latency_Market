"""Selective scan  y = SSM(x, dt, A, B, C) + D*x  with three interchangeable backends.

    h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t          (per channel d, state n)
    y_t = sum_n h_t * C_t + D * x_t

Shapes: x, dt: (Bsz, L, D) · A: (D, N) (negative) · B, C: (Bsz, L, N) · D: (D,)

Backends
--------
``parallel``  Materialises a, b and h as (Bsz, L, D, N) tensors and runs the
              Hillis-Steele scan in plain torch ops. Exportable to ONNX (used for CPU
              serving) and the ground truth for tests, but memory-bound: autograd keeps
              ~log2(L) full-size (Bsz, L, D, N) intermediates alive.

``chunked``   The fused kernel's algorithm, written in torch so it can be verified on
              any machine (tests/test_scan.py runs gradcheck against ``parallel``):
              * the sequence is processed in chunks; only the chunk-start states
                (Bsz, n_chunks, D, N) are saved, never the full (Bsz, L, D, N) history;
              * backward recomputes the forward scan of a chunk instead of loading it
                (FlashAttention-style recomputation), then runs the reverse scan
                dh_t = a_{t+1} dh_{t+1} + C_t gy_t  with a carry across chunks.

``triton``    The same algorithm as one fused GPU kernel per direction (scan_triton.py).
              Each program owns (batch, block of channels); a chunk's a, b, h tiles live
              in registers/shared memory, the scan inside a chunk is a parallel
              ``tl.associative_scan`` (depth log2(chunk)), and HBM only sees x, dt, B, C,
              y (+ the small chunk-state buffer). Requires an NVIDIA GPU and Triton.
"""

from __future__ import annotations

import torch

from qforecast.model.scan_core import parallel_scan

try:  # optional: only on machines with an NVIDIA GPU + triton
    from qforecast.model.scan_triton import selective_scan_triton

    HAS_TRITON = True
except Exception:  # ImportError, or triton present without a usable GPU
    selective_scan_triton = None
    HAS_TRITON = False


def selective_scan_parallel(x, dt, A, B, C, D):
    a = torch.exp(dt.unsqueeze(-1) * A)  # (Bsz, L, D, N)
    b = (dt * x).unsqueeze(-1) * B.unsqueeze(2)
    h = parallel_scan(a, b)
    return (h * C.unsqueeze(2)).sum(-1) + D * x


def _pairs(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inclusive scan of the affine maps h -> a*h + b along dim 1: returns (prod a, h|h0=0)."""
    return torch.cumprod(a, dim=1), parallel_scan(a, b)


class _ChunkedScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, A, B, C, D, chunk: int):
        bsz, length, d = x.shape
        h = x.new_zeros(bsz, d, A.shape[1])
        ys, starts = [], []
        for l0 in range(0, length, chunk):
            sl = slice(l0, l0 + chunk)
            starts.append(h)
            a = torch.exp(dt[:, sl, :, None] * A)
            b = (dt[:, sl] * x[:, sl])[..., None] * B[:, sl, None, :]
            P, S = _pairs(a, b)
            hc = S + P * h[:, None]  # chunk states, carried in from the previous chunk
            ys.append((hc * C[:, sl, None, :]).sum(-1) + D * x[:, sl])
            h = hc[:, -1]
        ctx.save_for_backward(x, dt, A, B, C, D, torch.stack(starts, 1))
        ctx.chunk = chunk
        return torch.cat(ys, 1)

    @staticmethod
    def backward(ctx, gy):
        x, dt, A, B, C, D, starts = ctx.saved_tensors
        chunk = ctx.chunk
        gx, gdt = torch.zeros_like(x), torch.zeros_like(dt)
        gB, gC = torch.zeros_like(B), torch.zeros_like(C)
        gA, gD = torch.zeros_like(A), torch.zeros_like(D)
        carry = torch.zeros_like(starts[:, 0])  # a_{t+1} * dh_{t+1} entering the chunk's last row
        n_chunks = starts.shape[1]
        for c in reversed(range(n_chunks)):
            sl = slice(c * chunk, (c + 1) * chunk)
            xs, dts, Bs, Cs, gys = x[:, sl], dt[:, sl], B[:, sl], C[:, sl], gy[:, sl]
            h0 = starts[:, c]
            # 1. recompute the chunk's forward scan (never stored)
            a = torch.exp(dts[..., None] * A)
            b = (dts * xs)[..., None] * Bs[:, :, None, :]
            P, S = _pairs(a, b)
            hc = S + P * h0[:, None]
            h_prev = torch.cat([h0[:, None], hc[:, :-1]], 1)
            # 2. reverse scan: dh_t = a_{t+1} dh_{t+1} + C_t gy_t, carry at the chunk end
            gh = gys[..., None] * Cs[:, :, None, :]
            a_next = torch.cat([a[:, 1:], torch.ones_like(a[:, :1])], 1)
            Pr, Sr = _pairs(a_next.flip(1), gh.flip(1))
            dh = (Sr + Pr * carry[:, None]).flip(1)
            # 3. local gradients
            ga = dh * h_prev * a  # d/d(dt*A) of exp(dt*A)
            dbB = (dh * Bs[:, :, None, :]).sum(-1)  # sum_n dh * B
            gx[:, sl] = dbB * dts + gys * D
            gdt[:, sl] = dbB * xs + (ga * A).sum(-1)
            gB[:, sl] = (dh * (dts * xs)[..., None]).sum(2)
            gC[:, sl] = (gys[..., None] * hc).sum(2)
            gA += (ga * dts[..., None]).sum((0, 1))
            gD += (gys * xs).sum((0, 1))
            carry = a[:, 0] * dh[:, 0]
        return gx, gdt, gA, gB, gC, gD, None


def selective_scan_chunked(x, dt, A, B, C, D, chunk: int = 32):
    return _ChunkedScan.apply(x, dt, A, B, C, D, chunk)


def resolve_backend(backend: str, x: torch.Tensor) -> str:
    if backend != "auto":
        return backend
    if torch.onnx.is_in_onnx_export():
        return "parallel"
    if x.is_cuda and HAS_TRITON:
        return "triton"
    return "parallel"


def selective_scan(x, dt, A, B, C, D, backend: str = "auto"):
    backend = resolve_backend(backend, x)
    if backend == "parallel":
        return selective_scan_parallel(x, dt, A, B, C, D)
    if backend == "chunked":
        return selective_scan_chunked(x, dt, A, B, C, D)
    if backend == "triton":
        if not HAS_TRITON:
            raise RuntimeError("triton backend requested but triton/CUDA is not available")
        return selective_scan_triton(x, dt, A, B, C, D)
    raise ValueError(f"unknown scan backend {backend!r}")
