# ruff: noqa: E501
"""Swing book on the daily cube: tranches formed on rebalance rows, held H days, P&L with costs and funding.

    tr = tranche_targets(score, mask, vol, rebal_rows, mode="LS")      # (T, N) target of the tranche formed at each rebalance row
    w  = overlap(tr, rebal_rows, H)                                     # book target = mean of the H / every live tranches
    res = simulate(w, rebal_rows, rows, ret_next, funding_next, cps)   # daily P&L, trades only on rebalance rows
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def leg_weights(x: np.ndarray, cap: float = 0.25) -> np.ndarray:
    """Positive weights summing to 1, none above ``cap`` (when feasible)."""
    w = x / x.sum()
    c = max(cap, 1.0 / len(w))
    for _ in range(50):
        w = np.minimum(w, c)
        w = w / w.sum()
    return w


def tranche_targets(score: np.ndarray, mask: np.ndarray, vol: np.ndarray, rebal: np.ndarray, mode: str, n: int = 8,
                    cap: float = 0.25, short_cap: float | None = None) -> np.ndarray:
    """Target weights of the tranche formed at each rebalance row (other rows 0).
    LS: long top n / short bottom n, ~1/vol, 0.5 gross per leg. LO: long top n, ~1/vol, gross 1. EW: all members, 1/n.
    ``short_cap``: cap for the short leg (default = ``cap``)."""
    T, N = score.shape
    out = np.zeros((T, N))
    for k in rebal:
        m = mask[k] & np.isfinite(score[k]) & np.isfinite(vol[k]) & (vol[k] > 0)
        idx = np.flatnonzero(m)
        if mode == "EW":
            idx = np.flatnonzero(mask[k])
            if len(idx):
                out[k, idx] = 1.0 / len(idx)
            continue
        nn = ns = n
        if len(idx) < nn + ns:  # small cross-section: split what is there, at least 2 per leg
            nn = ns = len(idx) // 2
            if nn < 2:
                continue
        order = idx[np.argsort(score[k, idx], kind="mergesort")]
        top = order[-nn:]
        out[k, top] = leg_weights(1.0 / vol[k, top], cap) * (0.5 if mode == "LS" else 1.0)
        if mode == "LS":
            bot = order[:ns]
            out[k, bot] = -leg_weights(1.0 / vol[k, bot], cap if short_cap is None else short_cap) * 0.5
    return out


def overlap(tr: np.ndarray, rebal: np.ndarray, H: int, every: int = 7) -> np.ndarray:
    """Book target at each rebalance row = mean of the tranches formed at the last H / every rebalance rows."""
    n_tr = max(1, H // every)
    out = np.zeros_like(tr)
    for i, k in enumerate(rebal):
        live = rebal[max(0, i - n_tr + 1): i + 1]
        out[k] = tr[live].sum(0) / n_tr
    return out


@dataclass
class Result:
    pnl: np.ndarray  # (T,) net return attributed to the decision row
    gross: np.ndarray
    cost: np.ndarray
    funding: np.ndarray
    turnover: np.ndarray
    exposure: np.ndarray  # net long exposure after trading


def simulate(target: np.ndarray, rebal: np.ndarray, rows: np.ndarray, ret_next: np.ndarray, funding_next: np.ndarray, cps: float,
             short_stop: float = 0.0, close: np.ndarray | None = None) -> Result:
    """Trade to ``target`` on rebalance rows, drift in between. ``rows`` sorted; cash earns 0.
    ``short_stop`` > 0 (needs ``close``): a short whose close at row k is >= (1 + short_stop) x its close at the last
    rebalance is bought back at open[k+1] (row k's trade) and stays flat until the next rebalance."""
    T, N = target.shape
    is_reb = np.zeros(T, bool)
    is_reb[rebal] = True
    out = {k: np.full(T, np.nan) for k in ("pnl", "gross", "cost", "funding", "turnover", "exposure")}
    w = np.zeros(N)
    entry = np.full(N, np.nan)
    for k in rows:
        c = 0.0
        tv = 0.0
        if is_reb[k]:
            tv = float(np.abs(target[k] - w).sum())
            c = tv * cps
            w = target[k].copy()
            if short_stop > 0:
                entry = np.where(w < 0, close[k], np.nan)
        elif short_stop > 0:
            hit = (w < 0) & np.isfinite(entry) & (np.nan_to_num(close[k], nan=0.0) >= entry * (1 + short_stop))
            if hit.any():
                tv = float(np.abs(w[hit]).sum())
                c = tv * cps
                w[hit] = 0.0
                entry[hit] = np.nan
        r = np.nan_to_num(ret_next[k])
        f = np.nan_to_num(funding_next[k])
        g = float(w @ r)
        fu = -float(w @ f)
        out["pnl"][k], out["gross"][k], out["cost"][k], out["funding"][k] = g + fu - c, g, c, fu
        out["turnover"][k], out["exposure"][k] = tv, float(w.sum())
        w = w * (1.0 + r)
    return Result(**out)


def stats(pnl: np.ndarray) -> dict:
    d = np.nan_to_num(pnl)
    eq = np.cumprod(1 + d)
    sd = d.std(ddof=1)
    yrs = len(d) / 365
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    return {"ret_pct": float((eq[-1] - 1) * 100), "cagr_pct": float((eq[-1] ** (1 / yrs) - 1) * 100) if yrs > 0 and eq[-1] > 0 else -100.0,
            "sharpe": float(d.mean() / sd * np.sqrt(365)) if sd > 0 else 0.0, "maxdd_pct": dd * 100, "days": int(len(d))}
