# ruff: noqa: E501
"""Batched book engine: the staggered C5-style book for a whole batch of signal-weight sets at once (torch, CPU or GPU).

    book = TorchBook(ctx, spec, kA, kS)           # arrays of one book, evaluated on rows [kA, kS)
    pnl = book.evaluate(W)                        # W (B, F): one weight vector per set -> daily x1 P&L (B, D)

The reference evaluator (qforecast.adaptive.search.NumpyBook) runs qforecast.swing.strategy.run_staggered once per
weight set. Here the weight set is a batch dimension instead: one pass over the rows ranks the coins for every set
with one matrix product and one stable sort, builds every tranche with scatter operations, and keeps the state of
the 7 sub-books for all sets in one tensor. Same tranche, beta hedge, thin leg, squeeze guard, costs, funding;
tests/test_adaptive.py checks that both evaluators give the same daily P&L. float32 on a GPU is the fast setting;
float64 on a CPU is the exact one.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from qforecast.adaptive.search import SIGNALS


class TorchBook:
    """Arrays of one book (full cube length; rows [kA, kS) are evaluated) and the batched evaluator."""

    def __init__(self, ctx, spec, kA: int, kS: int, signals=SIGNALS, cps: float = 7e-4, dt=torch.float32, dev: str = "cpu"):
        assert spec.mode == "LS" and spec.hedge == "beta" and spec.H == spec.every and spec.gross == 1.0 and not spec.vol_target
        cube = ctx.cube
        names = list(ctx.names)
        self.kA, self.kS = int(kA), int(kS)
        self.t = cube.t[self.kA:self.kS]
        self.signals = tuple(signals)
        self.Z = np.nan_to_num(ctx.Z[:, :, [names.index(s) for s in signals]]).astype(np.float64)
        self.valid = ctx.mask & np.isfinite(ctx.vol) & (ctx.vol > 0)
        self.vol, self.beta, self.close, self.close_raw = ctx.vol, ctx.beta, np.nan_to_num(cube.c, nan=0.0), cube.c.astype(np.float64)
        self.ret, self.fund = np.nan_to_num(ctx.ret_next), np.nan_to_num(ctx.funding_next)
        self.skip = (ctx.f7 < 0) if (spec.skip_neg_funding and ctx.f7 is not None) else None
        self.n, self.ns = int(spec.n), int(spec.n_short or spec.n)
        self.cap, self.short_cap = float(spec.cap), float(spec.short_cap or spec.cap)
        self.fixed = self.thin = bool(spec.thin_leg_btc)
        self.stop, self.cps, self.H = float(spec.short_stop), float(cps), int(spec.every)
        self.bench = int(cube.bench)
        self.N = cube.c.shape[1]
        self.dt, self.dev = dt, dev

    @property
    def D(self) -> int:
        return self.kS - self.kA

    def evaluate(self, W: np.ndarray, kA: int | None = None, kS: int | None = None) -> torch.Tensor:
        """(B, kS - kA) daily P&L of every weight set (same interface as search.NumpyBook)."""
        p = self
        if (kA is not None and kA != self.kA) or (kS is not None and kS != self.kS):
            p = copy.copy(self)
            p.kA, p.kS = self.kA if kA is None else int(kA), self.kS if kS is None else int(kS)
        return run_batch(p, W, self.dt, self.dev)[0]


def _legw(x, msk, cap, fixed):
    """Masked leg_weights / fixed_cap_weights of qforecast.swing.book, batched over rows of x (B, L)."""
    xm = torch.where(msk, x, torch.zeros_like(x))
    cnt = msk.sum(1, keepdim=True).to(x.dtype)
    s = xm.sum(1, keepdim=True)
    w = torch.where(s > 0, xm / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(xm))
    c = torch.full_like(cnt, cap) if fixed else torch.maximum(torch.full_like(cnt, cap), 1.0 / torch.clamp(cnt, min=1))
    for _ in range(50):
        w = torch.minimum(w, c)
        s = w.sum(1, keepdim=True)
        w = torch.where(s > 0, w / torch.where(s > 0, s, torch.ones_like(s)), w)
    if fixed:
        thin = (cnt * cap <= 1.0 + 1e-12) & (cnt > 0)
        w = torch.where(thin & msk, torch.full_like(w, cap), w)
    return torch.where(msk, w, torch.zeros_like(w))


def tranche(p: TorchBook, k: int, W: torch.Tensor, dt, dev) -> torch.Tensor | None:
    """(B, N) tranche of every weight set at row k (before the hedge)."""
    idx = np.flatnonzero(p.valid[k])
    nv = len(idx)
    nn, ns = p.n, p.ns
    if nv < nn + ns:
        nn = ns = nv // 2
        if nn < 2:
            return None
    B = W.shape[0]
    S = W @ torch.as_tensor(p.Z[k, idx].T, dtype=dt, device=dev)  # (B, nv)
    order = torch.sort(S, dim=1, stable=True).indices  # ascending, ties by coin order (= numpy mergesort)
    ix = torch.as_tensor(idx, device=dev)
    invv = torch.as_tensor(1.0 / p.vol[k, idx], dtype=dt, device=dev)
    out = torch.zeros((B, p.N), dtype=dt, device=dev)
    top = order[:, nv - nn:]
    wt = _legw(invv[top], torch.ones_like(top, dtype=torch.bool), p.cap, p.fixed) * 0.5
    out.scatter_(1, ix[top], wt)
    if p.skip is not None:
        pool = order[:, :min(2 * ns, nv - nn)]
        ok = ~torch.as_tensor(p.skip[k, idx], device=dev)[pool]
        keep = ok & (torch.cumsum(ok.to(torch.int64), 1) <= ns)
    else:
        pool = order[:, :ns]
        keep = torch.ones_like(pool, dtype=torch.bool)
    wb = _legw(invv[pool], keep, p.short_cap, p.fixed) * 0.5
    out.scatter_add_(1, ix[pool], -wb)
    return out


def hedge(p: TorchBook, k: int, w: torch.Tensor) -> torch.Tensor:
    beta = torch.as_tensor(p.beta[k], dtype=w.dtype, device=w.device)
    lb = torch.clamp(w, min=0) @ beta
    sb = -(torch.clamp(w, max=0) @ beta)
    if p.thin:
        thin = ((w > 0).sum(1) < 4) | ((w < 0).sum(1) < 4)
    else:
        thin = torch.zeros_like(lb, dtype=torch.bool)
    both = (~thin) & (lb > 0) & (sb > 0)
    a = torch.where(both, 2 * sb / torch.where(both, lb + sb, torch.ones_like(lb)), torch.ones_like(lb))[:, None]
    out = torch.where(both[:, None], torch.where(w > 0, w * a, w * (2 - a)), w)
    out[:, p.bench] = out[:, p.bench] + torch.where(thin, -(lb - sb), torch.zeros_like(lb))
    return out


def run_batch(p: TorchBook, W: np.ndarray, dt=torch.float64, dev="cpu"):
    """Daily x1 P&L (B, D) and turnover sum (B,) of every weight set on rows [kA, kS).
    W is (B, F) fixed weights, or (B, D, F) one weight vector per row (time-varying: the tranche opened at row j uses W[:, j])."""
    Wt = torch.as_tensor(W, dtype=dt, device=dev)
    H = p.H
    B, N, D = Wt.shape[0], p.N, p.D
    w = torch.zeros((H, B, N), dtype=dt, device=dev)
    entry = torch.full((H, B, N), float("nan"), dtype=dt, device=dev)
    pnl = torch.zeros((B, D), dtype=dt, device=dev)
    turn = torch.zeros(B, dtype=dt, device=dev)
    started = np.zeros(H, bool)
    for j, k in enumerate(range(p.kA, p.kS)):
        ph = j % H
        ck = torch.as_tensor(p.close[k], dtype=dt, device=dev)
        cost = torch.zeros((H, B), dtype=dt, device=dev)
        others = [q for q in range(H) if q != ph and started[q]]  # sub-books that hold something and do not re-pick today
        if others and p.stop > 0:
            o = torch.as_tensor(others, device=dev)
            wo, eo = w[o], entry[o]
            hit = (wo < 0) & torch.isfinite(eo) & (ck >= eo * (1 + p.stop))
            if bool(hit.any()):
                cost[o] = torch.where(hit, wo.abs(), torch.zeros_like(wo)).sum(2) * p.cps
                turn += torch.where(hit, wo.abs(), torch.zeros_like(wo)).sum(2).sum(0) / H
                w[o] = torch.where(hit, torch.zeros_like(wo), wo)
                entry[o] = torch.where(hit, torch.full_like(eo, float("nan")), eo)
        tr = tranche(p, k, Wt if Wt.ndim == 2 else Wt[:, j], dt, dev)
        tgt = hedge(p, k, tr) if tr is not None else torch.zeros((B, N), dtype=dt, device=dev)
        dw = (tgt - w[ph]).abs().sum(1)
        cost[ph] = dw * p.cps
        turn += dw / H
        w[ph] = tgt
        craw = torch.as_tensor(p.close_raw[k], dtype=dt, device=dev)  # entry = raw close (NaN stays NaN)
        entry[ph] = torch.where(tgt < 0, craw.expand_as(tgt), torch.full_like(tgt, float("nan")))
        started[ph] = True
        r = torch.as_tensor(p.ret[k], dtype=dt, device=dev)
        f = torch.as_tensor(p.fund[k], dtype=dt, device=dev)
        pnl[:, j] = ((w @ r) - (w @ f) - cost).mean(0)
        w = w * (1 + r)
    return pnl, turn
