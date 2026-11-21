# ruff: noqa: E501
"""Search the whole signal-weight space for the best set over a trailing window, every day.

Stage 1 (covering pool): ~120 000 weight sets spread over the space (sign-kept Dirichlet draws around the base
weights, plus free-sign draws, the base, the equal-sign set and every single signal), run ONCE over the whole
period by the batched engine; trailing L-day window scores come from cumulative sums of the daily P&L.
Stage 2 (local refinement): for each day, the 16 best pool sets by return and by mean / std are perturbed
(2 rounds x ~1 000 candidates) and re-scored by the engine run on that window only.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import torch

from qforecast.adaptive import engine as E

NPOOL, POOL_SEED = 60_000, 20261123


def nrm(w):
    w = np.atleast_2d(np.asarray(w, float))
    return w / np.abs(w).sum(-1, keepdims=True)


def pool(base: np.ndarray, n: int = NPOOL, seed: int = POOL_SEED) -> np.ndarray:
    """(2 + 2F + 2n, F) L1-normalised weight sets. ``base``: the strategy's own (normalised) weights."""
    base = nrm(base)[0]
    sgn = np.sign(base)
    rng = np.random.default_rng(seed)
    F = len(base)
    eye = np.eye(F) * sgn
    return np.vstack([base, nrm(sgn / F), eye, -eye, rng.dirichlet(np.full(F, 0.7), n) * sgn,
                      nrm(rng.dirichlet(np.full(F, 0.7), n) * rng.choice([-1, 1], (n, F)))]).astype(np.float64)


def run_pool(p: E.Prep, W: np.ndarray, dev: str = "cpu", bs: int = 8192) -> torch.Tensor:
    """(K, D) daily P&L of every pool set, float32 on ``dev``."""
    out = []
    for s in range(0, len(W), bs):
        P, _ = E.run_batch(p, W[s:s + bs], torch.float32, dev)
        out.append(P)
    return torch.cat(out)


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


def refine(p: E.Prep, Wp: np.ndarray, S_ret: torch.Tensor, S_sr: torch.Tensor, L: int, rng, base: np.ndarray,
           dev: str = "cpu", first: int | None = None) -> tuple[dict, dict]:
    """Local search for the window [t-L, t) of every column t >= ``first``: refined weights (D, F) for "ret" and "sr",
    and how much the refinement gained over the best pool set."""
    D, j0 = p.D, L + 7
    base = nrm(base)[0]
    p2 = copy.copy(p)
    outw = {"ret": np.tile(base, (D, 1)), "sr": np.tile(base, (D, 1))}
    gap = {"ret": [], "sr": []}
    top_r = torch.topk(S_ret, 16, dim=0).indices.cpu().numpy()  # (16, D - j0)
    top_s = torch.topk(S_sr, 16, dim=0).indices.cpu().numpy()
    F = Wp.shape[1]
    t0 = time.time()
    for t in range(max(j0, j0 if first is None else first), D):
        p2.kA, p2.kS = p.kA + t - j0, p.kA + t
        par = np.vstack([Wp[top_r[:, t - j0]], Wp[top_s[:, t - j0]]])  # 32 parents
        cand = np.vstack([par, nrm(np.repeat(par, 31, 0) + rng.normal(0, 0.06, (32 * 31, F)))])
        P, _ = E.run_batch(p2, cand, torch.float32, dev)
        x = P[:, 7:].double()
        sc = {"ret": x.sum(1), "sr": x.mean(1) / torch.clamp(x.std(1, unbiased=False), min=1e-6)}
        cand2 = []
        for o in ("ret", "sr"):
            best = cand[torch.topk(sc[o], 16).indices.cpu().numpy()]
            cand2.append(np.vstack([best, nrm(np.repeat(best, 31, 0) + rng.normal(0, 0.02, (16 * 31, F)))]))
        c2 = np.vstack(cand2)
        P, _ = E.run_batch(p2, c2, torch.float32, dev)
        x = P[:, 7:].double()
        sc2 = {"ret": x[:512].sum(1), "sr": x[512:].mean(1) / torch.clamp(x[512:].std(1, unbiased=False), min=1e-6)}
        for o, off in (("ret", 0), ("sr", 512)):
            i = int(torch.argmax(sc2[o]))
            outw[o][t] = c2[off + i]
            pi = 0 if o == "ret" else 16  # the covering-pool best for this objective
            gap[o].append((float(sc2[o][i]) - float(sc[o][pi]), float(np.abs(c2[off + i] - par[pi]).sum()), float(sc[o][pi])))
        if (t - j0) % 300 == 0:
            print(f"  refine t={t}/{D} {time.time() - t0:.0f} s", flush=True)
    summary = {o: {"mean_window_score_pool_best": round(float(np.mean([g[2] for g in gap[o]])), 4) if gap[o] else None,
                   "mean_gain_refined_vs_pool_best": round(float(np.mean([g[0] for g in gap[o]])), 4) if gap[o] else None,
                   "mean_l1_refined_vs_pool_best": round(float(np.mean([g[1] for g in gap[o]])), 3) if gap[o] else None} for o in gap}
    return outw, summary
