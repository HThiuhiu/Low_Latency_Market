# ruff: noqa: E501
"""History screen for a book variant: is its gain over a base book stable across time, not only on average?

    rep = block_report(base_pnl, var_pnl, k0, k1)     # 6 time blocks, 7-day purge at each boundary: sign of the gain per block
    out = cpcv(n_rows, fit_and_run)                    # combinatorial purged CV (6 groups, 2 test), purge + embargo 7 days

A variant passes the screen only if its daily gain is > 0 in >= 5 of 6 blocks AND on average. History can only
reject an idea: a variant that passes still needs data it has never seen (qforecast.vault, a forward window).
"""

from __future__ import annotations

import math

import numpy as np

N_BLOCKS, PURGE, MIN_POS = 6, 7, 5


def nw_t(y, lag=10):
    """t-stat of the mean with a Newey-West (Bartlett) variance: daily book P&L is autocorrelated."""
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < 30:
        return float("nan")
    e = y - y.mean()
    s = e @ e / n
    for L in range(1, lag + 1):
        s += 2 * (1 - L / (lag + 1)) * (e[L:] @ e[:-L]) / n
    return float(y.mean() / math.sqrt(s / n)) if s > 0 else float("nan")


def blocks(k0: int, k1: int, n: int = N_BLOCKS, purge: int = PURGE) -> list[np.ndarray]:
    """n contiguous row blocks of [k0, k1); the first ``purge`` rows of every block but the first are dropped
    (a 7-day position formed in block i - 1 must not score in block i)."""
    e = np.linspace(k0, k1, n + 1).astype(int)
    return [np.arange(e[i] + (purge if i else 0), e[i + 1]) for i in range(n)]


def block_report(base: np.ndarray, var: np.ndarray, k0: int, k1: int, n: int = N_BLOCKS, purge: int = PURGE, min_pos: int = MIN_POS) -> dict:
    d = np.asarray(var, float) - np.asarray(base, float)
    out = []
    for b in blocks(k0, k1, n, purge):
        x = d[b]
        out.append({"rows": [int(b[0]), int(b[-1])], "diff_ann_pct": round(float(np.nanmean(x) * 36500), 2), "t_nw10": round(nw_t(x), 2)})
    pos = sum(o["diff_ann_pct"] > 0 for o in out)
    allrows = np.concatenate(blocks(k0, k1, n, purge))
    return {"blocks": out, "positive_blocks": f"{pos}/{n}", "mean_diff_ann_pct": round(float(np.nanmean(d[allrows]) * 36500), 2),
            "pass": bool(pos >= min_pos and np.nanmean(d[allrows]) > 0)}


def cpcv(n_rows: int, fit_and_run, n_groups: int = N_BLOCKS, k_test: int = 2, purge: int = PURGE, embargo: int = PURGE) -> dict:
    """For a variant with FITTED parameters: fit_and_run(train_idx, test_idx) -> (base_pnl_test, var_pnl_test) arrays.
    Returns the gain per test group (mean over the splits that test it) and per backtest path."""
    from qforecast.validation import cpcv_splits

    splits, paths = cpcv_splits(n_rows, n_groups, k_test, purge, embargo)
    edges = np.linspace(0, n_rows, n_groups + 1).astype(int)
    by = {}
    for s, (tr, te, combo) in enumerate(splits):
        b, v = fit_and_run(tr, te)
        d = np.asarray(v, float) - np.asarray(b, float)
        for g in combo:
            m = (te >= edges[g]) & (te < edges[g + 1])
            by[(s, g)] = d[m]
    group_gain = {g: float(np.nanmean(np.concatenate([by[(s, gg)] for (s, gg) in by if gg == g])) * 36500) for g in range(n_groups)}
    path_gain = [float(np.nanmean(np.concatenate([by[sg] for sg in p])) * 36500) for p in paths]
    return {"group_gain_ann_pct": {g: round(v, 2) for g, v in group_gain.items()}, "positive_groups": f"{sum(v > 0 for v in group_gain.values())}/{n_groups}",
            "path_gain_ann_pct": [round(v, 2) for v in path_gain], "positive_paths": f"{sum(v > 0 for v in path_gain)}/{len(path_gain)}"}
