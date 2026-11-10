"""The daily cube every other swing module works on, and its point-in-time universe.

Universe: on the first day of every month, rank all crypto USDT perps by their mean daily quote
volume over the previous ``lookback_days`` (rows strictly before that day), require
``min_age_days`` of history, keep the top ``top_n``. Delisted contracts are included, so the
universe never knows who survives. BTC is always loaded: it is the hedge and the beta benchmark,
not a member of the ranked cross-section.

Timing convention (used everywhere): row k is the day [t_k, t_k + 1d). A decision made with data
up to the END of row k is executed at open[k+1]; the weights decided at row k earn
ret_next[k] = open[k+2] / open[k+1] - 1 and pay the funding settled inside (t_{k+1}, t_{k+2}].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

BENCH = "BTCUSDT"
DAY = 86_400_000
FIELDS = ("o", "h", "lo", "c", "qv", "tbq", "funding_paid", "funding_last", "member")


@dataclass
class Cube:
    t: np.ndarray  # (T,) day open times, ms
    syms: list[str]
    o: np.ndarray  # (T, N) open / high / low / close, NaN where the coin has no bar
    h: np.ndarray
    lo: np.ndarray
    c: np.ndarray
    qv: np.ndarray  # quote volume
    tbq: np.ndarray  # taker-buy quote volume
    funding_paid: np.ndarray  # sum of funding rates settled in (t_k, t_k + 1d], 0 if none
    funding_last: np.ndarray  # last settled funding rate known at the end of row k (ffill)
    member: np.ndarray  # (T, N) bool: in the ranked universe at row k (BTC: never)
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return len(self.t)

    @property
    def N(self) -> int:
        return len(self.syms)

    @property
    def bench(self) -> int:
        return self.syms.index(BENCH)

    def ret_next(self) -> np.ndarray:
        """(T, N) return earned by weights decided at row k: open[k+2] / open[k+1] - 1."""
        r = np.full(self.o.shape, np.nan)
        r[:-2] = self.o[2:] / self.o[1:-1] - 1.0
        return r

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, t=self.t, syms=np.array(self.syms), **{f: getattr(self, f) for f in FIELDS})

    @classmethod
    def load(cls, path: Path) -> Cube:
        z = np.load(path, allow_pickle=False)
        return cls(t=z["t"], syms=[str(s) for s in z["syms"]], **{f: z[f] for f in FIELDS})


def month_starts_ms(start_ms: int, end_ms: int) -> list[int]:
    d = datetime.fromtimestamp(start_ms / 1000, UTC).replace(day=1, hour=0, minute=0, second=0,
                                                              microsecond=0)
    out = []
    while int(d.timestamp() * 1000) < end_ms:
        out.append(int(d.timestamp() * 1000))
        d = d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)
    return out


def select_universe(d1: dict[str, np.ndarray], start_ms: int, end_ms: int, top_n: int,
                    lookback_days: int = 30, min_age_days: int = 60,
                    exclude: tuple[str, ...] = (BENCH,), skip_top: int = 0) -> dict[int, list[str]]:
    """month start (ms) -> members, ranked only with daily rows [day_ms, quote_volume, ...] stamped
    before that month. ``skip_top``: drop the ``skip_top`` most liquid coins first."""
    out = {}
    for m in month_starts_ms(start_ms, end_ms):
        scores = []
        for sym, a in d1.items():
            if sym in exclude or len(a) == 0:
                continue
            past = a[a[:, 0] < m]
            if len(past) < min_age_days or past[0, 0] > m - min_age_days * DAY:
                continue
            win = past[past[:, 0] >= m - lookback_days * DAY]
            if len(win) < lookback_days * 0.8:
                continue  # stopped trading (delisted) or long gaps
            scores.append((float(np.mean(win[:, 1])), sym))
        scores.sort(reverse=True)
        out[m] = [s for _, s in scores[skip_top:skip_top + top_n]]
    return out


def align(rows: np.ndarray, t: np.ndarray, cols: list[int]) -> np.ndarray:
    """Place ``rows[:, cols]`` on the time axis ``t`` by their first column (open time); NaN elsewhere."""
    out = np.full((len(t), len(cols)), np.nan)
    if len(rows) == 0:
        return out
    idx = np.searchsorted(t, rows[:, 0])
    ok = (idx < len(t)) & (t[np.minimum(idx, len(t) - 1)] == rows[:, 0])
    out[idx[ok]] = rows[ok][:, cols]
    return out


def alive_member(cube, window: int) -> np.ndarray:
    """member AND the coin traded in the last ``window`` rows (past data only). A delisted or halted
    perp can keep a forward-filled price with zero volume for weeks: it must not be ranked, held or filled."""
    qv = np.nan_to_num(np.asarray(cube.qv, float))
    cs = np.cumsum(qv, axis=0)
    roll = cs.copy()
    roll[window:] = cs[window:] - cs[:-window]
    return cube.member & (roll > 0)
