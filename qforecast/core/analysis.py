"""Real-time market analysis: microstructure + regime detection.

All updates are O(1); snapshots are computed at a throttled rate. Time windows use
*event time* (exchange timestamps), not the wall clock, so replaying a recorded
stream reproduces exactly the same analysis.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from qforecast.schemas import BookTicker, Kline, MarketAnalysis, Trade

_BUCKETS = 300  # 1-second buckets -> 5-minute trade-flow window


class TradeFlow:
    """Per-second taker buy/sell volume buckets in a circular array."""

    BUY, SELL, COUNT, NOTIONAL = range(4)

    def __init__(self):
        self.sec = np.full(_BUCKETS, -1, dtype=np.int64)
        self.data = np.zeros((_BUCKETS, 4))  # one row per second: buy, sell, count, notional
        self._cur = -1  # second of the bucket being filled
        self._acc = [0.0, 0.0, 0.0, 0.0]  # pending sums for that bucket (python floats: fast)

    def _flush(self) -> None:
        if self._cur >= 0:
            self.data[self._cur % _BUCKETS] = self._acc

    def add(self, t: Trade) -> None:
        s = t.trade_time // 1000
        if s != self._cur:
            if s < self._cur:  # late (out-of-order) trade for an older second
                i = s % _BUCKETS
                if self.sec[i] > s:  # slot already reused by a newer second: too old
                    return
                if self.sec[i] < s:
                    self.sec[i] = s
                    self.data[i] = 0.0
                self.data[i, self.SELL if t.is_buyer_maker else self.BUY] += t.qty
                self.data[i, self.COUNT] += 1
                self.data[i, self.NOTIONAL] += t.qty * t.price
                return
            self._flush()
            self._cur = s
            self.sec[s % _BUCKETS] = s
            self._acc = [0.0, 0.0, 0.0, 0.0]
        acc = self._acc
        acc[1 if t.is_buyer_maker else 0] += t.qty  # maker buyer => seller was aggressor
        acc[2] += 1.0
        acc[3] += t.qty * t.price

    def window(self, now_s: int, seconds: int) -> tuple[float, float, float, float]:
        self._flush()
        m = (self.sec > now_s - seconds) & (self.sec <= now_s)
        b, s, n, x = self.data[m].sum(axis=0)
        return float(b), float(s), float(n), float(x)


class MarketAnalyzer:
    def __init__(self, symbol: str, interval_ms: int = 60_000, rv_window: int = 60,
                 er_window: int = 30, regime_history: int = 1_440):
        self.symbol = symbol
        self.bars_per_year = 365 * 86_400_000 / interval_ms
        self.rv_window = rv_window
        self.er_window = er_window
        self.closes: deque[float] = deque(maxlen=max(rv_window, er_window) + 1)
        self.rv_hist: deque[float] = deque(maxlen=regime_history)
        self.flow = TradeFlow()
        self.now_ms = 0
        # book state
        self.bid = self.ask = self.bid_qty = self.ask_qty = math.nan
        self.obi_ewma = 0.0
        # RSI (Wilder)
        self._gain = self._loss = 0.0
        self._n = 0
        self.rv = math.nan

    # ---- event handlers ------------------------------------------------------
    def on_book(self, b: BookTicker) -> None:
        self.bid, self.bid_qty, self.ask, self.ask_qty = b.bid, b.bid_qty, b.ask, b.ask_qty
        tot = b.bid_qty + b.ask_qty
        obi = (b.bid_qty - b.ask_qty) / tot if tot > 0 else 0.0
        self.obi_ewma += 0.05 * (obi - self.obi_ewma)

    def on_trade(self, t: Trade) -> None:
        self.flow.add(t)
        if t.trade_time > self.now_ms:
            self.now_ms = t.trade_time

    def on_kline(self, k: Kline) -> None:
        if self.closes:
            d = k.close - self.closes[-1]
            up, dn = max(d, 0.0), max(-d, 0.0)
            if self._n == 1:
                self._gain, self._loss = up, dn
            else:
                self._gain += (up - self._gain) / 14
                self._loss += (dn - self._loss) / 14
        self._n += 1
        self.closes.append(k.close)
        if k.close_time > self.now_ms:
            self.now_ms = k.close_time
        if len(self.closes) > self.rv_window:
            c = np.fromiter(self.closes, float)[-(self.rv_window + 1):]
            r = np.diff(np.log(c))
            self.rv = float(r.std(ddof=1) * math.sqrt(self.bars_per_year))
            self.rv_hist.append(self.rv)
        # Bar-level regimes only change on bar close: compute them here, once per bar,
        # so snapshot() (called many times per second) stays O(1).
        self._regime = self._vol_regime()
        self._er, self._trend_state = self._trend()

    # ---- derived state -----------------------------------------------------
    _regime = "NORMAL"
    _er = math.nan
    _trend_state = "RANGE"

    def _vol_regime(self) -> str:
        if len(self.rv_hist) < 100 or math.isnan(self.rv):
            return "NORMAL"
        pct = float(np.mean(np.fromiter(self.rv_hist, float) <= self.rv))
        return "HIGH" if pct > 0.8 else "LOW" if pct < 0.2 else "NORMAL"

    def _trend(self) -> tuple[float, str]:
        if len(self.closes) <= self.er_window:
            return math.nan, "RANGE"
        c = np.fromiter(self.closes, float)[-(self.er_window + 1):]
        path = float(np.abs(np.diff(c)).sum())
        net = float(c[-1] - c[0])
        er = abs(net) / path if path > 0 else 0.0
        if er > 0.3:
            return er, "TREND_UP" if net > 0 else "TREND_DOWN"
        return er, "RANGE"

    def snapshot(self) -> MarketAnalysis | None:
        if math.isnan(self.bid):
            return None
        mid = 0.5 * (self.bid + self.ask)
        tot = self.bid_qty + self.ask_qty
        obi = (self.bid_qty - self.ask_qty) / tot if tot > 0 else 0.0
        micro = (self.ask * self.bid_qty + self.bid * self.ask_qty) / tot if tot > 0 else mid
        now_s = self.now_ms // 1000
        b1, s1, n1, _ = self.flow.window(now_s, 60)
        b5, s5, _, notional5 = self.flow.window(now_s, 300)
        vol5 = b5 + s5
        gl = self._gain + self._loss
        rsi = 100.0 * self._gain / gl if gl > 0 else 50.0
        er, trend = self._er, self._trend_state
        return MarketAnalysis(
            symbol=self.symbol, ts=self.now_ms, mid=mid,
            spread_bps=(self.ask - self.bid) / mid * 1e4, microprice=micro,
            obi=obi, obi_ewma=self.obi_ewma,
            flow_imbalance_1m=(b1 - s1) / (b1 + s1) if b1 + s1 > 0 else 0.0,
            flow_imbalance_5m=(b5 - s5) / vol5 if vol5 > 0 else 0.0,
            trade_rate_1m=n1 / 60.0,
            vwap_5m=notional5 / vol5 if vol5 > 0 else mid,
            realized_vol_ann=self.rv if not math.isnan(self.rv) else 0.0,
            vol_regime=self._regime,
            efficiency_ratio=er if not math.isnan(er) else 0.0,
            trend_regime=trend, rsi=rsi,
            momentum_state="OVERBOUGHT" if rsi > 70 else "OVERSOLD" if rsi < 30 else "NEUTRAL",
        )
