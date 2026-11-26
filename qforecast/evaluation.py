# ruff: noqa: E501
"""Book evaluation helpers shared by every research script: daily PnL, Sharpe, drawdown, CVaR,
summary metrics and a paired stationary-bootstrap test of Sharpe differences (with a White
reality-check p-value over several candidates). PnL arrays are fractions of equity per row."""

from __future__ import annotations

import numpy as np


def daily(pnl: np.ndarray, rows: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(pnl[rows])
    return np.add.reduceat(x, np.arange(0, len(x), 24))


def sharpe(d: np.ndarray) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0


def maxdd(d: np.ndarray) -> float:
    eq = np.cumsum(d)
    return float((eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]).min())


def cvar(d: np.ndarray, q: float = 0.05) -> float:
    k = max(1, int(np.ceil(q * len(d))))
    return float(np.sort(d)[:k].mean())


def metrics(d: np.ndarray) -> dict:
    dd = maxdd(d)
    yrs = len(d) / 365
    return {"usd_per_10k": float(d.sum() * 1e4), "sharpe": sharpe(d), "maxdd_pct": dd * 100,
            "cvar5_bps": cvar(d) * 1e4, "calmar": float(d.sum() / yrs / abs(dd)) if dd < 0 else 0.0,
            "vol_ann_pct": float(d.std(ddof=1) * np.sqrt(365) * 100)}


def stationary_idx(n: int, block: float, draws: int, rng) -> np.ndarray:
    restart = rng.random((draws, n)) < 1.0 / block
    restart[:, 0] = True
    start = rng.integers(0, n, size=(draws, n))
    t = np.arange(n)
    last = np.maximum.accumulate(np.where(restart, t, 0), axis=1)
    return (np.take_along_axis(start, last, 1) + (t - last)) % n


def boot_sharpe_diff(cands: dict[str, np.ndarray], base: np.ndarray, block: float, draws: int, seed: int) -> dict:
    """Paired stationary bootstrap of Sharpe(c) - Sharpe(base); one-sided p for each and a
    White reality-check p for the best of them (bootstrap of the centred maximum)."""
    rng = np.random.default_rng(seed)
    idx = stationary_idx(len(base), block, draws, rng)

    def sr(x):
        m, s = x.mean(1), x.std(1, ddof=1)
        return np.where(s > 0, m / s * np.sqrt(365), 0.0)

    sb = sr(base[idx])
    obs, cen = {}, []
    out = {}
    for k, c in cands.items():
        dob = sharpe(c) - sharpe(base)
        db = sr(c[idx]) - sb
        obs[k] = dob
        cen.append(db - db.mean())
        out[k] = {"d_sharpe": dob, "se": float(db.std()), "p_one_sided": float(np.mean(db - db.mean() >= dob))}
    best = max(obs.values())
    out["reality_check_p"] = float(np.mean(np.max(np.vstack(cen), axis=0) >= best))
    return out
