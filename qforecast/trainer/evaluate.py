"""Out-of-sample evaluation with quant-style metrics and naive baselines."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def horizon_metrics(pred: np.ndarray, y: np.ndarray, past_ret: np.ndarray) -> dict:
    """pred / y / past_ret: 1-D arrays of log-returns for one horizon."""
    nz = y != 0
    mse = float(np.mean((y - pred) ** 2))
    return {
        "ic_spearman": float(spearmanr(pred, y).statistic),
        "hit_rate": float(np.mean(np.sign(pred[nz]) == np.sign(y[nz]))),
        # 1 - MSE / MSE(zero forecast): >0 means better than "no change" (random walk)
        "r2_oos_vs_zero": float(1.0 - mse / np.mean(y ** 2)),
        "hit_rate_momentum_baseline": float(np.mean(np.sign(past_ret[nz]) == np.sign(y[nz]))),
        "n": int(len(y)),
    }


def strategy_backtest(pred: np.ndarray, y: np.ndarray, h: int, threshold_bps: float,
                      cost_bps: float, bars_per_year: float) -> dict:
    """Non-overlapping trades every h bars: go long/short when |pred| > threshold.

    Deliberately simple and conservative (taker costs on every round trip); it is a
    sanity check of the signal, not a production strategy.
    """
    p, r = pred[::h], y[::h]
    pos = np.where(p * 1e4 > threshold_bps, 1.0, np.where(p * 1e4 < -threshold_bps, -1.0, 0.0))
    gross = pos * r
    net = gross - np.abs(pos) * cost_bps / 1e4
    periods = bars_per_year / h

    def sharpe(x):
        return float(np.mean(x) / np.std(x) * np.sqrt(periods)) if np.std(x) > 0 else 0.0

    return {
        "horizon": h, "trades": int(np.count_nonzero(pos)), "periods": int(len(pos)),
        "gross_return": float(np.sum(gross)), "net_return": float(np.sum(net)),
        "gross_sharpe": sharpe(gross), "net_sharpe": sharpe(net),
        "cost_bps_round_trip": cost_bps, "threshold_bps": threshold_bps,
    }
