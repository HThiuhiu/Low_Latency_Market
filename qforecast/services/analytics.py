"""Analytics service: book + trades + bars -> market microstructure / regime snapshots.

Snapshots are throttled (conflated) to ``analysis_publish_interval_ms`` so a burst of
book updates costs O(1) state updates, not a publish per message.
"""

from __future__ import annotations

import time

from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.analysis import MarketAnalyzer
from qforecast.core.metrics import Registry
from qforecast.schemas import BookTicker, Kline, Trade, decode_md, encode


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    an = MarketAnalyzer(s.symbol, s.interval_ms)
    h_update = reg.hist("analysis_update", "per-message state update")
    h_snap = reg.hist("analysis_snapshot", "snapshot computation")
    interval_ns = s.analysis_publish_interval_ms * 1_000_000
    last_pub = 0
    handlers = {BookTicker: an.on_book, Trade: an.on_trade, Kline: an.on_kline}

    async for _, payload in bus.subscribe([s.t_kline, s.t_book, s.t_trade], replay=True):
        msg = decode_md(payload)
        t0 = time.perf_counter_ns()
        handlers[type(msg)](msg)
        t1 = time.perf_counter_ns()
        h_update.record(t1 - t0)
        reg.inc("messages")
        fresh_bar = type(msg) is Kline and time.time() * 1000 - msg.close_time < 2 * s.interval_ms
        if t1 - last_pub >= interval_ns or fresh_bar:
            snap = an.snapshot()
            if snap is not None:
                h_snap.record(time.perf_counter_ns() - t1)
                await bus.publish(s.t_analysis, encode(snap))
                reg.inc("snapshots")
                last_pub = t1


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("analytics", run)


if __name__ == "__main__":
    main()
