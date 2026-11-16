# ruff: noqa: E501
"""One place that turns a strategy spec into book targets: used by the backtests, the research tools and the paper bot.

    spec = SwingSpec.load("artifacts/models/<strategy>.json")
    ctx = Context.build(cube)                              # signals, ranks, vol, beta (once per cube)
    target = book_targets(ctx, spec, rebal_rows)           # (T, N) book weights to trade to on each rebalance row
    pnl = run_staggered(ctx, spec, k0, k1, cps).pnl        # mean of the 7 weekday sub-books (no rebalance-day luck)
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from qforecast.swing import book as B
from qforecast.swing.cube import alive_member
from qforecast.swing.features import Trailing, cs_rank_gauss, rolling_mean
from qforecast.swing.signals import SPECS, compute


@dataclass
class SwingSpec:
    mode: str = "LS"  # LS | LO | EW
    H: int = 7  # holding days per tranche
    every: int = 7  # rebalance every N days
    n: int = 8  # names per leg
    cap: float = 0.25  # max share of one coin in its leg
    weights: dict = field(default_factory=dict)  # signal -> weight on its rank; empty = every signal with a prior, sign = prior
    weighting: str = "invvol"  # invvol | equal
    hedge: str = "none"  # LS: none | beta (scale legs to zero beta vs BTC) | btc (BTC position = -book beta)
    vol_target: float = 0.0  # annualised book vol target (0 = off), from trailing realised vol of the target book
    vol_lookback: int = 60
    vol_cap: float = 1.5  # max leverage multiplier from vol targeting
    gross: float = 1.0
    short_cap: float = 0.0  # LS: cap of one coin in the short leg (0 = same as cap)
    short_stop: float = 0.0  # LS: close a short after it rose this much since its rebalance (0 = off)

    @classmethod
    def load(cls, path) -> SwingSpec:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        p = d.get("params", d)
        return cls(**{k: v for k, v in p.items() if k in {f.name for f in fields(cls)}})

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Context:
    cube: object
    tr: object
    names: list
    Z: np.ndarray  # (T, N, F) cross-sectional normal scores
    pri: np.ndarray
    vol: np.ndarray  # 28-day daily vol
    beta: np.ndarray  # 90-day beta to BTC
    mask: np.ndarray
    ret_next: np.ndarray
    funding_next: np.ndarray
    f7: np.ndarray | None = None  # raw 7-day mean funding

    @classmethod
    def build(cls, cube, mask=None) -> Context:
        if mask is None:
            mask = alive_member(cube, 1)  # drop dead (zero-volume, frozen-price) members
        tr = Trailing(cube, window=90)
        raw = compute(cube, tr, mask)
        names = [s.name for s in SPECS]
        Z = np.stack([cs_rank_gauss(raw[n], mask) for n in names], axis=2)
        fn = np.zeros_like(cube.funding_paid)
        fn[:-1] = cube.funding_paid[1:]
        return cls(cube, tr, names, Z, np.array([s.prior for s in SPECS], float), np.sqrt(rolling_mean(tr.r ** 2, 28)),
                   np.nan_to_num(tr.beta, nan=1.0), mask, cube.ret_next(), fn, np.nan_to_num(raw["funding_7d"]))


def weight_vector(ctx: Context, weights: dict) -> np.ndarray:
    """(F,) signal weights in ctx.names order, L1-normalised; empty = the literature priors."""
    w = np.array([float(weights.get(n, 0.0)) for n in ctx.names]) if weights else ctx.pri.copy()
    return w / max(np.abs(w).sum(), 1e-12)


def score(ctx: Context, spec: SwingSpec) -> np.ndarray:
    s = np.nan_to_num(ctx.Z) @ weight_vector(ctx, spec.weights)
    return np.where(ctx.mask, s, np.nan)


def book_targets(ctx: Context, spec: SwingSpec, rebal: np.ndarray, s: np.ndarray | None = None) -> np.ndarray:
    cube = ctx.cube
    b = cube.bench
    s = score(ctx, spec) if s is None else s
    vol = ctx.vol if spec.weighting == "invvol" else np.where(np.isfinite(ctx.vol), 1.0, np.nan)
    tr = B.tranche_targets(s, ctx.mask, vol, rebal, spec.mode, n=spec.n, cap=spec.cap, short_cap=spec.short_cap or None)
    if spec.mode == "LS" and spec.hedge == "beta":  # scale each tranche's legs so long beta = short beta, gross kept
        for k in rebal:
            w = tr[k]
            lb, sb = float(np.clip(w, 0, None) @ ctx.beta[k]), float(-np.clip(w, None, 0) @ ctx.beta[k])
            if lb > 0 and sb > 0:
                a = 2 * sb / (lb + sb)
                tr[k] = np.where(w > 0, w * a, w * (2 - a))
    w = B.overlap(tr, rebal, spec.H, spec.every) * spec.gross
    if spec.mode == "LS" and spec.hedge == "btc":
        for k in rebal:
            w[k, b] = -float(np.delete(w[k], b) @ np.delete(ctx.beta[k], b))
    if spec.vol_target > 0:
        w = _vol_target(ctx, w, rebal, spec)
    return w


def _vol_target(ctx: Context, w: np.ndarray, rebal: np.ndarray, spec: SwingSpec) -> np.ndarray:
    """Scale each rebalance target by vol_target / realised vol of that SAME weight vector over the trailing
    ``vol_lookback`` days (returns known at the decision; no look-ahead), capped at ``vol_cap``."""
    r = np.nan_to_num(ctx.tr.r)  # daily log returns, row k known at the end of k
    out = w.copy()
    L = spec.vol_lookback
    for k in rebal:
        if k < L:
            continue
        pr = r[k - L + 1:k + 1] @ w[k]
        sd = float(np.std(pr, ddof=1) * np.sqrt(365))
        if sd > 0:
            out[k] = w[k] * min(spec.vol_target / sd, spec.vol_cap)
    return out


def run_book(ctx: Context, spec: SwingSpec, k0: int, k1: int, cps: float, s: np.ndarray | None = None, phase: int = 0) -> B.Result:
    """One sub-book: rebalanced every ``spec.every`` rows from k0 + phase, simulated on every row of [k0, k1)."""
    rebal = np.arange(k0 + phase, k1, spec.every)
    return B.simulate(book_targets(ctx, spec, rebal, s), rebal, np.arange(k0, k1), ctx.ret_next, ctx.funding_next, cps,
                      short_stop=spec.short_stop, close=ctx.cube.c)


def run_staggered(ctx: Context, spec: SwingSpec, k0: int, k1: int, cps: float, s: np.ndarray | None = None) -> B.Result:
    """Mean of the ``every`` weekday sub-books (each rebalanced weekly on its own day): no rebalance-day luck,
    and the way a book that re-trades 1 / 7 of itself every day behaves."""
    s = score(ctx, spec) if s is None else s
    subs = [run_book(ctx, spec, k0, k1, cps, s, ph) for ph in range(spec.every)]
    with warnings.catch_warnings():  # rows outside [k0, k1) are NaN in every sub-book
        warnings.simplefilter("ignore", RuntimeWarning)
        return B.Result(**{f.name: np.nanmean([getattr(r, f.name) for r in subs], axis=0) for f in fields(B.Result)})


def staggered_target_row(ctx: Context, spec: SwingSpec, k: int, lookback: int = 120, s: np.ndarray | None = None) -> np.ndarray:
    """(N,) staggered book target at row k: mean over the ``every`` weekday sub-books of their latest rebalance target.
    ``s``: a (T, N) score to use instead of score(ctx, spec), e.g. one scored with each row's own weights."""
    s = score(ctx, spec) if s is None else s
    rows = []
    for ph in range(spec.every):
        rb = np.arange(max(0, k - lookback) + ph, k + 1, spec.every)
        if len(rb):
            rows.append(book_targets(ctx, spec, rb, s)[rb[-1]])
    return np.mean(rows, axis=0) if rows else np.zeros(ctx.cube.N)
