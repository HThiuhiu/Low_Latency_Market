"""Fused selective-scan kernels (Triton -> CUDA). Requires an NVIDIA GPU + `pip install triton`.

A line-by-line GPU port of ``_ChunkedScan`` in scan_fused.py (which is verified with
gradcheck on CPU); tests/test_scan.py compares the two on a GPU.

Grid: one program per (batch, block of BLOCK_D channels). A program walks the
sequence in chunks of BLOCK_L steps carrying the state h (BLOCK_D x N) in registers.
Per chunk, everything happens on-chip:

    load  x, dt (BLOCK_L x BLOCK_D), B, C (BLOCK_L x N)        <- the only HBM reads
    a  = exp(dt * A), b = dt * x * B                           (BLOCK_L, BLOCK_D, N) tile
    h  = associative_scan(a, b) + cumprod(a) * h_carry         parallel, depth log2(BLOCK_L)
    y  = sum_n h * C + D * x                                   -> store y (HBM write)

so the (Bsz, L, D, N) tensors that the plain-torch version materialises in HBM never
leave the SM. Backward recomputes the chunk's forward scan on-chip (no stored h), runs
the reverse scan for dh with a carry between chunks, and reduces dA / dD in registers;
dB / dC are reduced over the channel block and summed over blocks on the host.

Padding trick: masked rows load dt = 0, so a = exp(0) = 1 and b = 0, i.e. the identity
map, and the scan needs no special-casing for ragged chunk ends.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

if not torch.cuda.is_available():  # let scan_fused fall back cleanly on CPU-only hosts
    raise ImportError("triton selective scan needs a CUDA device")


@triton.jit
def _combine(a_l, b_l, a_r, b_r):
    # (a_l, b_l) is the earlier map, (a_r, b_r) the later one: h -> a_r*(a_l*h + b_l) + b_r
    return a_l * a_r, a_r * b_l + b_r


@triton.jit
def _fwd_kernel(X, DT, A, Bm, Cm, Dv, Y, HS,
                L, DM, n_chunks,
                s_xb, s_xl, s_xd, s_bb, s_bl, s_bn,
                BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, N)
    mask_d = offs_d < DM
    a_mat = tl.load(A + offs_d[:, None] * N + offs_n[None, :], mask=mask_d[:, None], other=0.0)
    d_vec = tl.load(Dv + offs_d, mask=mask_d, other=0.0)
    h = tl.zeros((BLOCK_D, N), dtype=tl.float32)
    for c in range(0, n_chunks):
        hs_ptr = HS + ((pid_b * n_chunks + c) * DM + offs_d[:, None]) * N + offs_n[None, :]
        tl.store(hs_ptr, h, mask=mask_d[:, None])  # chunk-start state, for the backward pass
        offs_l = c * BLOCK_L + rows
        mask_l = offs_l < L
        m2 = mask_l[:, None] & mask_d[None, :]
        xo = pid_b * s_xb + offs_l[:, None] * s_xl + offs_d[None, :] * s_xd
        bo = pid_b * s_bb + offs_l[:, None] * s_bl + offs_n[None, :] * s_bn
        x = tl.load(X + xo, mask=m2, other=0.0).to(tl.float32)
        dt = tl.load(DT + xo, mask=m2, other=0.0).to(tl.float32)
        b_t = tl.load(Bm + bo, mask=mask_l[:, None], other=0.0).to(tl.float32)
        c_t = tl.load(Cm + bo, mask=mask_l[:, None], other=0.0).to(tl.float32)

        a = tl.exp(dt[:, :, None] * a_mat[None, :, :])  # (BLOCK_L, BLOCK_D, N)
        b = (dt * x)[:, :, None] * b_t[:, None, :]
        p, s = tl.associative_scan((a, b), 0, _combine)
        hc = s + p * h[None, :, :]
        y = tl.sum(hc * c_t[:, None, :], axis=2) + d_vec[None, :] * x
        tl.store(Y + xo, y.to(Y.dtype.element_ty), mask=m2)
        # padded rows are identity maps, so the last row holds the chunk's final state
        h = tl.sum(tl.where((rows == BLOCK_L - 1)[:, None, None], hc, 0.0), axis=0)


@triton.jit
def _bwd_kernel(X, DT, A, Bm, Cm, Dv, GY, HS,
                GX, GDT, GA, GB, GC, GD,
                L, DM, n_chunks, n_dblk,
                s_xb, s_xl, s_xd, s_bb, s_bl, s_bn,
                BLOCK_L: tl.constexpr, BLOCK_D: tl.constexpr, N: tl.constexpr):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    rows = tl.arange(0, BLOCK_L)
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    offs_n = tl.arange(0, N)
    mask_d = offs_d < DM
    a_mat = tl.load(A + offs_d[:, None] * N + offs_n[None, :], mask=mask_d[:, None], other=0.0)
    d_vec = tl.load(Dv + offs_d, mask=mask_d, other=0.0)
    carry = tl.zeros((BLOCK_D, N), dtype=tl.float32)  # a_{t+1} * dh_{t+1} entering the chunk end
    ga_acc = tl.zeros((BLOCK_D, N), dtype=tl.float32)
    gd_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for cc in range(0, n_chunks):
        c = n_chunks - 1 - cc
        l0 = c * BLOCK_L
        offs_l = l0 + rows
        mask_l = offs_l < L
        m2 = mask_l[:, None] & mask_d[None, :]
        xo = pid_b * s_xb + offs_l[:, None] * s_xl + offs_d[None, :] * s_xd
        bo = pid_b * s_bb + offs_l[:, None] * s_bl + offs_n[None, :] * s_bn
        x = tl.load(X + xo, mask=m2, other=0.0).to(tl.float32)
        dt = tl.load(DT + xo, mask=m2, other=0.0).to(tl.float32)
        gy = tl.load(GY + xo, mask=m2, other=0.0).to(tl.float32)
        b_t = tl.load(Bm + bo, mask=mask_l[:, None], other=0.0).to(tl.float32)
        c_t = tl.load(Cm + bo, mask=mask_l[:, None], other=0.0).to(tl.float32)
        hs_ptr = HS + ((pid_b * n_chunks + c) * DM + offs_d[:, None]) * N + offs_n[None, :]
        h0 = tl.load(hs_ptr, mask=mask_d[:, None], other=0.0)

        # 1. recompute the forward scan on-chip
        a = tl.exp(dt[:, :, None] * a_mat[None, :, :])
        b = (dt * x)[:, :, None] * b_t[:, None, :]
        p, s = tl.associative_scan((a, b), 0, _combine)
        hc = s + p * h0[None, :, :]

        # h_{t-1}: the same scan over inputs shifted by one step (identity on row 0),
        # which avoids both a register shift and the unstable (h_t - b_t) / a_t.
        mask_p = mask_l & (rows > 0)
        mp2 = mask_p[:, None] & mask_d[None, :]
        xp = tl.load(X + xo - s_xl, mask=mp2, other=0.0).to(tl.float32)
        dtp = tl.load(DT + xo - s_xl, mask=mp2, other=0.0).to(tl.float32)
        bp = tl.load(Bm + bo - s_bl, mask=mask_p[:, None], other=0.0).to(tl.float32)
        ap = tl.exp(dtp[:, :, None] * a_mat[None, :, :])
        bbp = (dtp * xp)[:, :, None] * bp[:, None, :]
        pp, sp = tl.associative_scan((ap, bbp), 0, _combine)
        h_prev = sp + pp * h0[None, :, :]

        # 2. reverse scan  dh_t = a_{t+1} dh_{t+1} + C_t gy_t  (a_{t+1} := 1 at the chunk
        #    end; the cross-chunk term enters through `carry`). Reverse = flip, scan, flip.
        mask_nx = (rows < BLOCK_L - 1) & (offs_l + 1 < L)
        mn2 = mask_nx[:, None] & mask_d[None, :]
        dtn = tl.load(DT + xo + s_xl, mask=mn2, other=0.0).to(tl.float32)
        a_next = tl.exp(dtn[:, :, None] * a_mat[None, :, :])
        gh = gy[:, :, None] * c_t[:, None, :]
        pr, sr = tl.associative_scan((tl.flip(a_next, 0), tl.flip(gh, 0)), 0, _combine)
        dh = tl.flip(sr + pr * carry[None, :, :], 0)

        # 3. local gradients
        ga = dh * h_prev * a
        dbb = tl.sum(dh * b_t[:, None, :], axis=2)
        tl.store(GX + xo, dbb * dt + gy * d_vec[None, :], mask=m2)
        tl.store(GDT + xo, dbb * x + tl.sum(ga * a_mat[None, :, :], axis=2), mask=m2)
        gb = tl.sum(dh * (dt * x)[:, :, None], axis=1)  # (BLOCK_L, N), partial over channels
        gc = tl.sum(gy[:, :, None] * hc, axis=1)
        po = ((pid_b * n_dblk + pid_d) * L + offs_l[:, None]) * N + offs_n[None, :]
        tl.store(GB + po, gb, mask=mask_l[:, None])
        tl.store(GC + po, gc, mask=mask_l[:, None])
        ga_acc += tl.sum(ga * dt[:, :, None], axis=0)
        gd_acc += tl.sum(gy * x, axis=0)
        carry = tl.sum(tl.where((rows == 0)[:, None, None], a * dh, 0.0), axis=0)

    tl.store(GA + (pid_b * DM + offs_d[:, None]) * N + offs_n[None, :], ga_acc, mask=mask_d[:, None])
    tl.store(GD + pid_b * DM + offs_d, gd_acc, mask=mask_d)


def _blocks(length: int, n: int) -> tuple[int, int]:
    block_l = min(32, triton.next_power_of_2(length))
    block_d = 8 if n <= 8 else 4  # keep the (BLOCK_L, BLOCK_D, N) tiles within registers
    return block_l, block_d


class _TritonScan(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, x, dt, A, B, C, D):
        x, dt, B, C = (t.contiguous() for t in (x, dt, B, C))
        A, D = A.contiguous(), D.contiguous()
        bsz, length, dm = x.shape
        n = A.shape[1]
        if n & (n - 1):
            raise ValueError("d_state must be a power of two for the triton kernel")
        block_l, block_d = _blocks(length, n)
        n_chunks = triton.cdiv(length, block_l)
        y = torch.empty_like(x)
        hs = torch.empty(bsz, n_chunks, dm, n, device=x.device, dtype=torch.float32)
        grid = (bsz, triton.cdiv(dm, block_d))
        _fwd_kernel[grid](x, dt, A, B, C, D, y, hs, length, dm, n_chunks,
                          *x.stride(), *B.stride(), BLOCK_L=block_l, BLOCK_D=block_d, N=n)
        ctx.save_for_backward(x, dt, A, B, C, D, hs)
        ctx.blocks = (block_l, block_d, n_chunks)
        return y

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, gy):
        x, dt, A, B, C, D, hs = ctx.saved_tensors
        block_l, block_d, n_chunks = ctx.blocks
        gy = gy.contiguous().float()
        bsz, length, dm = x.shape
        n = A.shape[1]
        n_dblk = triton.cdiv(dm, block_d)
        gx, gdt = torch.empty_like(x), torch.empty_like(dt)
        ga = torch.empty(bsz, dm, n, device=x.device, dtype=torch.float32)
        gd = torch.empty(bsz, dm, device=x.device, dtype=torch.float32)
        gb = torch.empty(bsz, n_dblk, length, n, device=x.device, dtype=torch.float32)
        gc = torch.empty_like(gb)
        _bwd_kernel[(bsz, n_dblk)](x, dt, A, B, C, D, gy, hs, gx, gdt, ga, gb, gc, gd,
                                   length, dm, n_chunks, n_dblk, *x.stride(), *B.stride(),
                                   BLOCK_L=block_l, BLOCK_D=block_d, N=n,
                                   num_warps=8)  # ~16 live tiles: more threads, fewer spills
        return gx, gdt, ga.sum(0), gb.sum(1), gc.sum(1), gd.sum(0)


def selective_scan_triton(x, dt, A, B, C, D):
    return _TritonScan.apply(x, dt, A, B, C, D)
