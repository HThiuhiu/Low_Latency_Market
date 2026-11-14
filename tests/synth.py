"""Synthetic daily cube for tests: BTC + N coins with a market factor, random funding and volume."""

from __future__ import annotations

import dataclasses

import numpy as np

from qforecast.swing.cube import BENCH, DAY, Cube

T0 = 1_600_000_000_000 // DAY * DAY


def make_cube(T: int = 320, N: int = 30, seed: int = 0) -> Cube:
    rng = np.random.default_rng(seed)
    mkt = rng.normal(0, 0.03, T)
    beta = rng.uniform(0.5, 1.5, N)
    r = np.c_[mkt, mkt[:, None] * beta + rng.normal(0, 0.04, (T, N))]
    c = 100 * np.exp(np.cumsum(r, axis=0))
    o = np.vstack([c[:1], c[:-1]])
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.02, c.shape))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.02, c.shape))
    qv = rng.lognormal(16, 1, c.shape)
    tbq = qv * rng.uniform(0.4, 0.6, c.shape)
    fp = rng.normal(1e-4, 3e-4, c.shape)
    member = np.ones(c.shape, bool)
    member[:, 0] = False  # BTC is the benchmark, never ranked
    member[: T // 3, -3:] = False  # three coins join later
    syms = [BENCH] + [f"C{i:02d}USDT" for i in range(N)]
    return Cube(t=T0 + np.arange(T, dtype=np.int64) * DAY, syms=syms, o=o, h=h, lo=lo, c=c, qv=qv, tbq=tbq,
                funding_paid=fp, funding_last=fp, member=member)


def cut(cube: Cube, k: int) -> Cube:
    """The cube as it was known at the end of row k (rows > k removed)."""
    kw = {}
    for f in dataclasses.fields(cube):
        v = getattr(cube, f.name)
        kw[f.name] = v[: k + 1] if isinstance(v, np.ndarray) and v.shape[:1] == (cube.T,) else v
    return Cube(**kw)
