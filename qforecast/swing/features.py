"""Trailing statistics behind the signals: rolling means, beta to BTC, cross-sectional normal scores.
Everything at row k uses rows <= k only."""

from __future__ import annotations

import numpy as np


def log_returns(c: np.ndarray) -> np.ndarray:
    r = np.full(c.shape, np.nan)
    r[1:] = np.log(c[1:] / c[:-1])
    return r


def rolling_mean(x: np.ndarray, w: int, min_frac: float = 0.8) -> np.ndarray:
    """Trailing mean over w rows along axis 0, NaN-aware; NaN when < min_frac * w values.
    Cumulative-sum difference: O(T) whatever the window."""
    ok = np.isfinite(x)
    v = np.where(ok, x, 0.0)
    cs = np.cumsum(v, axis=0)
    cn = np.cumsum(ok, axis=0)
    s = cs.copy()
    n = cn.astype(float)
    s[w:] -= cs[:-w]
    n[w:] -= cn[:-w]
    return np.where(n >= min_frac * w, s / np.maximum(n, 1), np.nan)


def rolling_beta(r: np.ndarray, rb: np.ndarray, w: int) -> np.ndarray:
    """Trailing OLS beta of every column of r (T, N) on rb (T,), on the rows where both are known."""
    rb2 = np.broadcast_to(rb[:, None], r.shape)
    both = np.isfinite(r) & np.isfinite(rb2)
    x = np.where(both, rb2, np.nan)
    y = np.where(both, r, np.nan)
    mx, my = rolling_mean(x, w), rolling_mean(y, w)
    mxy, mxx = rolling_mean(x * y, w), rolling_mean(x * x, w)
    var_x = mxx - mx * mx
    return (mxy - mx * my) / np.where(var_x > 0, var_x, np.nan)


class Trailing:
    """Daily log returns and trailing beta to BTC, computed once per cube."""

    def __init__(self, cube, window: int = 90):
        self.r = log_returns(cube.c)
        b = cube.bench
        self.beta = rolling_beta(self.r, self.r[:, b], window)
        self.beta[:, b] = 1.0


def cs_rank_gauss(x: np.ndarray, mask: np.ndarray, min_names: int = 5) -> np.ndarray:
    """Per row: rank the masked, finite values and map them to normal scores (van der Waerden).
    Missing values among members become 0 (neutral); non-members are NaN. Ties keep the column order."""
    from scipy.special import ndtri

    out = np.full(x.shape, np.nan)
    for t in range(x.shape[0]):
        m = mask[t]
        if not m.any():
            continue
        v = x[t, m]
        ok = np.isfinite(v)
        row = np.zeros(m.sum())
        if ok.sum() >= min_names:
            order = np.argsort(np.argsort(v[ok], kind="mergesort"), kind="mergesort")
            row[ok] = ndtri((order + 0.5) / ok.sum())
        out[t, m] = row
    return out
