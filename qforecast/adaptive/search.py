# ruff: noqa: E501
"""Search the signal-weight space for the best set over a trailing window, every day.

Stage 1 (covering pool): weight sets spread over the space (sign-kept Dirichlet draws around the base weights,
plus free-sign draws, the base, the equal-sign set and every single signal), run ONCE over the whole period;
trailing L-day window scores come from cumulative sums of the daily P&L.
Stage 2 (local refinement): for each day, the best pool sets by return and by mean / std are perturbed in two
rounds and re-scored on that day's window only.

Two evaluators share one interface (``evaluate(W, kA, kS)`` -> daily P&L of every set):
    NumpyBook                          the reference: one qforecast.swing.strategy.run_staggered call per set (slow, exact)
    qforecast.adaptive.engine.TorchBook  the whole batch of sets at once in torch, CPU or GPU (same P&L, much faster)
With the batched engine the pool can be large enough to cover the whole weight space every day.
"""

from __future__ import annotations

import time
from dataclasses import replace

import numpy as np
import torch

SIGNALS = ("vol_28d", "beta_90d", "max_7d", "funding_7d", "near_high_90d", "flow_res_7d")
NPOOL, NPOOL_REFERENCE, POOL_SEED = 60_000, 250, 20261123  # pool draws per half: batched engine / reference


def nrm(w):
    w = np.atleast_2d(np.asarray(w, float))
    return w / np.abs(w).sum(-1, keepdims=True)


class NumpyBook:
    """Reference evaluator on rows [kA, kS): daily x1 P&L of each weight set, one run_staggered call per set."""

    def __init__(self, ctx, spec, kA: int, kS: int, signals=SIGNALS, cps: float = 7e-4):
        self.ctx, self.spec, self.signals, self.cps = ctx, spec, tuple(signals), float(cps)
        self.kA, self.kS = int(kA), int(kS)
        self.t = ctx.cube.t[self.kA:self.kS]

    @property
    def D(self) -> int:
        return self.kS - self.kA

    def evaluate(self, W: np.ndarray, kA: int | None = None, kS: int | None = None) -> torch.Tensor:
        from qforecast.swing.strategy import run_staggered

        kA = self.kA if kA is None else kA
        kS = self.kS if kS is None else kS
        out = np.zeros((len(W), kS - kA))
        for i, w in enumerate(W):
            spec = replace(self.spec, weights=dict(zip(self.signals, map(float, w), strict=True)))
            out[i] = np.nan_to_num(run_staggered(self.ctx, spec, kA, kS, self.cps).pnl[kA:kS])
        return torch.as_tensor(out)


def pool(base: np.ndarray, n: int = NPOOL, seed: int = POOL_SEED) -> np.ndarray:
    """(2 + 2F + 2n, F) L1-normalised weight sets. ``base``: the strategy's own (normalised) weights."""
    base = nrm(base)[0]
    sgn = np.sign(base)
    rng = np.random.default_rng(seed)
    F = len(base)
    eye = np.eye(F) * sgn
    return np.vstack([base, nrm(sgn / F), eye, -eye, rng.dirichlet(np.full(F, 0.7), n) * sgn,
                      nrm(rng.dirichlet(np.full(F, 0.7), n) * rng.choice([-1, 1], (n, F)))]).astype(np.float64)


def run_pool(book, W: np.ndarray, bs: int = 8192) -> torch.Tensor:
    """(K, D) daily P&L of every pool set over the book's rows."""
    return torch.cat([book.evaluate(W[s:s + bs]) for s in range(0, len(W), bs)])


def window_scores(Pg: torch.Tensor, L: int, obj: str) -> torch.Tensor:
    """(K, D - j0) score of every set on the trailing window of columns t-L .. t-1, for t = j0 .. D-1 (j0 = L + 7).
    obj "ret" = sum of daily P&L, "sr" = mean / std."""
    K, D = Pg.shape
    z = torch.zeros((K, 1), dtype=Pg.dtype, device=Pg.device)
    c1 = torch.cat([z, torch.cumsum(Pg, 1)], 1)
    j0 = L + 7
    s1 = c1[:, j0:D] - c1[:, j0 - L:D - L]
    if obj == "ret":
        return s1
    c2 = torch.cat([z, torch.cumsum(Pg * Pg, 1)], 1)
    s2 = c2[:, L + 7:D] - c2[:, 7:D - L]
    m = s1 / L
    return m / torch.sqrt(torch.clamp(s2 / L - m * m, min=1e-12))


def refine(book, Wp: np.ndarray, S_ret: torch.Tensor, S_sr: torch.Tensor, L: int, rng, base: np.ndarray,
           first: int | None = None, top: int = 16, kids: int = 31) -> tuple[dict, dict]:
    """Local search for the window [t-L, t) of every column t >= ``first``: refined weights (D, F) for "ret" and "sr",
    and how much the refinement gained over the best pool set. ``top`` parents per objective, ``kids`` children each."""
    D, j0 = book.D, L + 7
    base = nrm(base)[0]
    outw = {"ret": np.tile(base, (D, 1)), "sr": np.tile(base, (D, 1))}
    gap = {"ret": [], "sr": []}
    top_r = torch.topk(S_ret, top, dim=0).indices.cpu().numpy()  # (top, D - j0)
    top_s = torch.topk(S_sr, top, dim=0).indices.cpu().numpy()
    F = Wp.shape[1]
    n2 = top * (kids + 1)  # candidates per objective in round 2
    t0 = time.time()
    for t in range(max(j0, j0 if first is None else first), D):
        kA, kS = book.kA + t - j0, book.kA + t
        par = np.vstack([Wp[top_r[:, t - j0]], Wp[top_s[:, t - j0]]])
        cand = np.vstack([par, nrm(np.repeat(par, kids, 0) + rng.normal(0, 0.06, (2 * top * kids, F)))])
        x = book.evaluate(cand, kA, kS)[:, 7:].double()
        sc = {"ret": x.sum(1), "sr": x.mean(1) / torch.clamp(x.std(1, unbiased=False), min=1e-6)}
        cand2 = []
        for o in ("ret", "sr"):
            best = cand[torch.topk(sc[o], top).indices.cpu().numpy()]
            cand2.append(np.vstack([best, nrm(np.repeat(best, kids, 0) + rng.normal(0, 0.02, (top * kids, F)))]))
        c2 = np.vstack(cand2)
        x = book.evaluate(c2, kA, kS)[:, 7:].double()
        sc2 = {"ret": x[:n2].sum(1), "sr": x[n2:].mean(1) / torch.clamp(x[n2:].std(1, unbiased=False), min=1e-6)}
        for o, off in (("ret", 0), ("sr", n2)):
            i = int(torch.argmax(sc2[o]))
            outw[o][t] = c2[off + i]
            pi = 0 if o == "ret" else top  # the covering-pool best for this objective
            gap[o].append((float(sc2[o][i]) - float(sc[o][pi]), float(np.abs(c2[off + i] - par[pi]).sum()), float(sc[o][pi])))
        if (t - j0) % 300 == 0:
            print(f"  refine t={t}/{D} {time.time() - t0:.0f} s", flush=True)
    summary = {o: {"mean_window_score_pool_best": round(float(np.mean([g[2] for g in gap[o]])), 4) if gap[o] else None,
                   "mean_gain_refined_vs_pool_best": round(float(np.mean([g[0] for g in gap[o]])), 4) if gap[o] else None,
                   "mean_l1_refined_vs_pool_best": round(float(np.mean([g[1] for g in gap[o]])), 3) if gap[o] else None} for o in gap}
    return outw, summary
