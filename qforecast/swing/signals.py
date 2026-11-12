# ruff: noqa: E501
"""Daily cross-sectional signals for a book held 1-4 weeks. Row k uses days <= k only.

Each signal comes from the literature with the sign it predicts for the rank correlation with the future
residual return (``prior``; 0 = no published direction). A strategy picks a subset and a weight per signal on
its cross-sectional normal score (qforecast.swing.strategy).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from qforecast.swing.features import rolling_mean


@dataclass(frozen=True)
class Spec:
    name: str
    prior: int
    hypothesis: str
    source: str


SPECS: tuple[Spec, ...] = (
    Spec("mom_7d", +1, "one-week cross-sectional momentum", "Liu-Tsyvinski-Wu 2022 JF (1-4 week momentum)"),
    Spec("mom_14d", +1, "two-week momentum", "Liu-Tsyvinski-Wu 2022 JF"),
    Spec("mom_28d", +1, "one-month momentum", "Liu-Tsyvinski-Wu 2022 JF; Fieberg-Liedtke-Zaremba 2024"),
    Spec("mom_90d_skip7", +1, "quarter momentum skipping the last week", "Jegadeesh-Titman 1993; crypto replications"),
    Spec("rmom_28d", +1, "residual (BTC-hedged) one-month momentum", "Blitz-Huij-Martens 2011"),
    Spec("trend_ma50", +1, "price above its 50-day mean (trend)", "Zarattini-Aziz-Barbon (crypto trend)"),
    Spec("donchian_28d", +1, "position inside the 28-day high-low channel", "Donchian; Zarattini-Aziz-Barbon"),
    Spec("near_high_90d", +1, "close near the 90-day high: anchoring, slow reaction", "George-Hwang 2004 (52-week high)"),
    Spec("funding_7d", -1, "crowded longs pay funding and underperform", "He-Manela-Ross-von Wachter; BIS WP 1087"),
    Spec("vol_28d", -1, "low-volatility anomaly", "Cakici et al. 2024 (crypto)"),
    Spec("max_7d", -1, "lottery demand: coins with a huge up-day underperform", "Bali-Cakici-Whitelaw 2011; Li-Urquhart et al. (crypto MAX)"),
    Spec("size_28d", -1, "smaller (less traded) coins earn more", "Liu-Tsyvinski-Wu 2022 (size factor)"),
    Spec("beta_90d", -1, "low-beta anomaly (betting against beta)", "Frazzini-Pedersen 2014"),
    Spec("turnover_chg", 0, "volume surge vs 90-day mean (attention)", "Liu-Tsyvinski 2021"),
    Spec("rev_1d", 0, "last-day move (reversal in illiquid names)", "-"),
    Spec("flow_7d", 0, "taker buy share over 7 days", "-"),
    Spec("flow_res_7d", +1, "7-day taker flow not explained by the same-window return (cross-sectional rank residual)",
         "Anastasopoulos-Gradojevic 2026 J. Financial Markets (residual order flow predicts weekly returns)"),
)
SPEC = {s.name: s for s in SPECS}


def _lag(x, k):
    out = np.full(x.shape, np.nan)
    out[k:] = x[:-k]
    return out


def _roll(fn, x, w):
    from numpy.lib.stride_tricks import sliding_window_view

    out = np.full(x.shape, np.nan)
    if len(x) >= w:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[w - 1:] = fn(sliding_window_view(x, w, axis=0), axis=2)
    return out


def _rank_resid(y: np.ndarray, x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per row, among members with both values: rank(y) minus its OLS fit on rank(x) (ranks scaled to [-0.5, 0.5])."""
    from scipy.stats import rankdata

    out = np.full(y.shape, np.nan)
    for k in range(len(y)):
        m = mask[k] & np.isfinite(y[k]) & np.isfinite(x[k])
        if m.sum() < 5:
            continue
        ry = rankdata(y[k, m]) / m.sum() - 0.5
        rx = rankdata(x[k, m]) / m.sum() - 0.5
        bta = float(rx @ ry / (rx @ rx)) if rx @ rx > 0 else 0.0
        out[k, m] = ry - bta * rx
    return out


def compute(cube, tr, mask: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """name -> raw (T, N). ``tr`` = qforecast.swing.features.Trailing built on the same cube (window 90 days)."""
    lc = np.log(cube.c)
    r = tr.r
    rb = r[:, [cube.bench]]
    out = {}
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        out["mom_7d"] = lc - _lag(lc, 7)
        out["mom_14d"] = lc - _lag(lc, 14)
        out["mom_28d"] = lc - _lag(lc, 28)
        out["mom_90d_skip7"] = _lag(lc, 7) - _lag(lc, 90)
        resid = r - _lag(tr.beta, 1) * rb
        out["rmom_28d"] = rolling_mean(resid, 28) * 28
        out["trend_ma50"] = lc - np.log(rolling_mean(cube.c, 50))
        hh, ll = _roll(np.nanmax, cube.h, 28), _roll(np.nanmin, cube.lo, 28)
        out["donchian_28d"] = (cube.c - 0.5 * (hh + ll)) / np.where(hh > ll, 0.5 * (hh - ll), np.nan)
        out["near_high_90d"] = cube.c / _roll(np.nanmax, cube.h, 90)
        out["funding_7d"] = rolling_mean(cube.funding_paid, 7, 0.5)
        out["vol_28d"] = np.sqrt(rolling_mean(r * r, 28))
        out["max_7d"] = _roll(np.nanmax, r, 7)
        out["size_28d"] = np.log(rolling_mean(cube.qv, 28))
        out["beta_90d"] = tr.beta
        out["turnover_chg"] = np.log(rolling_mean(cube.qv, 7) / rolling_mean(cube.qv, 90))
        out["rev_1d"] = -r
        qv = np.where(cube.qv > 0, cube.qv, np.nan)
        out["flow_7d"] = rolling_mean(cube.tbq, 7) / rolling_mean(qv, 7) - 0.5
        out["flow_res_7d"] = _rank_resid(out["flow_7d"], out["mom_7d"], cube.member if mask is None else mask)
    for k in out:
        out[k] = out[k].astype(np.float64)
        out[k][~np.isfinite(out[k])] = np.nan
    return out
