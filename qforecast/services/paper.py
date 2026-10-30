"""Paper-trading service: live model signals -> simulated orders -> equity, on the bus.

Consumes closed klines (stops / horizon exits / mark-to-market) and forecasts
(entries, filled at the forecast bar's close + slippage). Publishes a JSON state
snapshot on ``sim.paper.<symbol>``; the gateway serves it at ``/v1/paper``.
No real orders are ever sent.
"""

from __future__ import annotations

import logging
import time

import orjson

from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.metrics import Registry
from qforecast.schemas import Forecast, Kline, decode_forecast, decode_md
from qforecast.sim.broker import PaperBroker, SimConfig

log = logging.getLogger("paper")


class PaperEngine:
    """Transport-agnostic core (unit-testable without a bus)."""

    def __init__(self, s: Settings, reg: Registry | None = None):
        self.s = s
        self.reg = reg or Registry("paper")
        self.broker = PaperBroker(SimConfig(
            cash=s.paper_cash, fee_bps=s.paper_fee_bps, slippage_bps=s.paper_slippage_bps,
            size=s.paper_size, stop_bps=s.paper_stop_bps, take_bps=s.paper_take_bps,
            allow_short=not s.paper_long_only))
        self.last_open = -1
        self.started = int(time.time() * 1000)

    def on_kline(self, k: Kline) -> bool:
        """Only live bars count: replayed history must not open or close paper trades."""
        if k.open_time <= self.last_open or k.close_time < self.started - self.s.interval_ms:
            return False
        self.last_open = k.open_time
        n = len(self.broker.trades)
        self.broker.on_bar(k.open_time, k.open, k.high, k.low, k.close)
        self._log_closed(n)
        return True

    def on_forecast(self, f: Forecast) -> bool:
        if f.bar_time < self.started - 2 * self.s.interval_ms:
            return False
        n, before = len(self.broker.trades), self.broker.pos
        self.broker.on_signal(f.bar_time, f.signal, f.signal_horizon, f.edge_bps, price=f.close)
        self.broker.last_price = f.close
        self._log_closed(n)
        p = self.broker.pos
        if p is not None and p is not before:
            self.reg.inc("paper_entries")
            log.info("OPEN %s %.4f @ %.2f (h=%d, edge=%.1fbps) equity=%.2f",
                     "LONG" if p.side > 0 else "SHORT", p.qty, p.entry_price, p.horizon,
                     f.edge_bps, self.broker.equity())
        return True

    def _log_closed(self, n: int) -> None:
        for t in self.broker.trades[n:]:
            self.reg.inc("paper_trades")
            log.info("CLOSE %s @ %.2f (%s) pnl=%+.2f (%+.1fbps) equity=%.2f", t.side, t.exit_price,
                     t.reason, t.pnl, t.ret_bps, self.broker.equity())

    def snapshot(self) -> bytes:
        b = self.broker
        st = b.state()
        st.update(symbol=self.s.symbol, ts=int(time.time() * 1000), started=self.started,
                  start_equity=b.cfg.cash, return_pct=(b.equity() / b.cfg.cash - 1) * 100,
                  wins=sum(t.pnl > 0 for t in b.trades), fees_paid=b.fees_paid)
        return orjson.dumps(st)


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    eng = PaperEngine(s, reg)
    log.info("paper trading %s with %.2f (fee %.1fbps/side, size %.2fx) - simulation only",
             s.symbol, s.paper_cash, s.paper_fee_bps, s.paper_size)
    async for topic, payload in bus.subscribe([s.t_kline, s.t_forecast]):
        changed = (eng.on_kline(decode_md(payload)) if topic == s.t_kline
                   else eng.on_forecast(decode_forecast(payload)))
        if changed:
            await bus.publish(s.t_paper, eng.snapshot())


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("paper", run)


if __name__ == "__main__":
    main()
