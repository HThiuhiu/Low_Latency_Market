"""Incremental technical-feature engine (numba).

Design goals
------------
* **O(window) per bar, zero allocations** — the hot path touches a contiguous
  ring-buffer view (64 rows) plus an 8-float recursive state (EMAs / Wilder averages).
* **One implementation for training and serving.** ``compute_features`` (offline,
  whole history) simply loops the same ``step`` kernel that the live signal service
  calls once per closed bar, so train/serve skew is impossible by construction.
  ``tests/test_features.py`` cross-checks it against an independent pandas reference.
* **Causal** — the feature row for bar *t* only uses bars <= *t*.
* **Stationary inputs** — returns, ratios and normalised distances; no raw prices.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from qforecast.core.ringbuffer import RingBuffer

# Bar layout (columns of the bar matrix / ring buffer)
B_OPEN, B_HIGH, B_LOW, B_CLOSE, B_VOLUME, B_TRADES, B_TAKER = range(7)
BAR_WIDTH = 7

FEATURE_NAMES = (
    "ret_1", "ret_5", "ret_15", "rvol_20", "bb_z_20", "dist_ma_50", "atr_pct_14",
    "rsi_14", "macd_hist_pct", "rel_volume", "range_pct", "body_ratio",
    "taker_imbalance", "rel_trades",
)
N_FEATURES = len(FEATURE_NAMES)

WINDOW = 64  # rows of history the kernel looks at
WARMUP = 64  # first bar index (0-based count) with valid features is WARMUP-1

# Recursive state layout
S_N, S_PREV_C, S_EMA12, S_EMA26, S_SIG, S_GAIN, S_LOSS, S_ATR = range(8)
STATE_SIZE = 8

_A12 = 2.0 / 13.0
_A26 = 2.0 / 27.0
_A9 = 2.0 / 10.0
_W14 = 1.0 / 14.0


def new_state() -> np.ndarray:
    return np.zeros(STATE_SIZE, dtype=np.float64)


@njit(cache=True, inline="always")
def _mean(x, n):
    m = x.shape[0]
    s = 0.0
    for i in range(m - n, m):
        s += x[i]
    return s / n


@njit(cache=True, inline="always")
def _std(x, n, mean):
    m = x.shape[0]
    s = 0.0
    for i in range(m - n, m):
        d = x[i] - mean
        s += d * d
    return np.sqrt(s / (n - 1))


@njit(cache=True)
def step(state, win, out):
    """Advance the state with the newest bar (``win[-1]``) and write features to ``out``.

    ``win`` holds the most recent min(count, WINDOW) bars, oldest first.
    Returns True when ``out`` contains valid (post warm-up) features.
    """
    m = win.shape[0]
    o = win[m - 1, B_OPEN]
    h = win[m - 1, B_HIGH]
    lo = win[m - 1, B_LOW]
    c = win[m - 1, B_CLOSE]
    v = win[m - 1, B_VOLUME]
    nt = win[m - 1, B_TRADES]
    tb = win[m - 1, B_TAKER]

    k = state[S_N]
    if k == 0:
        ema12 = c
        ema26 = c
        sig = 0.0
        gain = 0.0
        loss = 0.0
        atr = h - lo
    else:
        pc = state[S_PREV_C]
        ema12 = state[S_EMA12] + _A12 * (c - state[S_EMA12])
        ema26 = state[S_EMA26] + _A26 * (c - state[S_EMA26])
        sig = state[S_SIG] + _A9 * ((ema12 - ema26) - state[S_SIG])
        d = c - pc
        up = d if d > 0.0 else 0.0
        dn = -d if d < 0.0 else 0.0
        if k == 1:
            gain = up
            loss = dn
        else:
            gain = state[S_GAIN] + _W14 * (up - state[S_GAIN])
            loss = state[S_LOSS] + _W14 * (dn - state[S_LOSS])
        tr = max(h - lo, abs(h - pc), abs(lo - pc))
        atr = state[S_ATR] + _W14 * (tr - state[S_ATR])

    state[S_N] = k + 1
    state[S_PREV_C] = c
    state[S_EMA12] = ema12
    state[S_EMA26] = ema26
    state[S_SIG] = sig
    state[S_GAIN] = gain
    state[S_LOSS] = loss
    state[S_ATR] = atr

    if k + 1 < WARMUP:
        return False

    closes = win[:, B_CLOSE]
    ma20 = _mean(closes, 20)
    sd20 = _std(closes, 20, ma20)
    ma50 = _mean(closes, 50)

    # realised vol of 1-bar log returns over the last 20 bars
    rs = 0.0
    rss = 0.0
    for i in range(m - 20, m):
        r = np.log(closes[i] / closes[i - 1])
        rs += r
        rss += r * r
    rmean = rs / 20.0
    rv = np.sqrt(max(rss - 20.0 * rmean * rmean, 0.0) / 19.0)

    vma = _mean(win[:, B_VOLUME], 20)
    nma = _mean(win[:, B_TRADES], 20)
    gl = gain + loss
    rsi = 100.0 * gain / gl if gl > 0.0 else 50.0
    rng = h - lo

    out[0] = np.log(c / closes[m - 2])
    out[1] = np.log(c / closes[m - 6])
    out[2] = np.log(c / closes[m - 16])
    out[3] = rv
    out[4] = (c - ma20) / sd20 if sd20 > 0.0 else 0.0
    out[5] = c / ma50 - 1.0
    out[6] = atr / c
    out[7] = (rsi - 50.0) / 50.0
    out[8] = (ema12 - ema26 - sig) / c
    out[9] = np.log((v + 1e-9) / (vma + 1e-9))
    out[10] = rng / c
    out[11] = (c - o) / rng if rng > 0.0 else 0.0
    out[12] = tb / v - 0.5 if v > 0.0 else 0.0
    out[13] = np.log((nt + 1.0) / (nma + 1.0))
    return True


@njit(cache=True)
def compute_features(bars):
    """Run ``step`` over a full bar history. Returns (features[N, F], valid[N])."""
    n = bars.shape[0]
    state = np.zeros(STATE_SIZE, dtype=np.float64)
    feats = np.full((n, N_FEATURES), np.nan, dtype=np.float64)
    valid = np.zeros(n, dtype=np.bool_)
    for i in range(n):
        lo = i - WINDOW + 1 if i >= WINDOW - 1 else 0
        valid[i] = step(state, bars[lo:i + 1], feats[i])
    return feats, valid


class OnlineFeatureEngine:
    """Stateful wrapper used by the signal service: one ``update`` per closed bar."""

    def __init__(self, capacity: int = 1_000):
        if capacity < WINDOW:
            raise ValueError("capacity must be >= WINDOW")
        self.bars = RingBuffer(capacity, BAR_WIDTH)
        self.state = new_state()
        self._out = np.zeros(N_FEATURES, dtype=np.float64)
        self._row = np.zeros(BAR_WIDTH, dtype=np.float64)

    def warmup_jit(self) -> None:
        """Trigger numba compilation (or cache load) before the first live bar."""
        dummy = np.ones((WINDOW, BAR_WIDTH), dtype=np.float64)
        dummy.flags.writeable = False  # ring-buffer views are read-only -> same specialisation
        step(new_state(), dummy, np.zeros(N_FEATURES))

    def update(self, o, h, lo, c, v, trades, taker) -> np.ndarray | None:
        r = self._row
        r[0], r[1], r[2], r[3], r[4], r[5], r[6] = o, h, lo, c, v, trades, taker
        self.bars.push(r)
        win = self.bars.last(min(len(self.bars), WINDOW))
        ok = step(self.state, win, self._out)
        return self._out if ok else None
