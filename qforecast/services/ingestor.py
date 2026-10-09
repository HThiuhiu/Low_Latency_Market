"""Ingestor: the only service that talks to the exchange.

Binance WebSocket (kline + bookTicker + aggTrade) -> normalised msgpack -> bus.
On every (re)connect it REST-backfills missed bars so downstream consumers see a
gap-free, de-duplicated, ordered kline stream.
"""

from __future__ import annotations

import logging
import time

from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.metrics import Registry
from qforecast.exchange.binance import fetch_klines, raw_to_klines, stream_market_data, stream_url
from qforecast.schemas import BookTicker, Kline, Trade, decode_md, encode

log = logging.getLogger("ingestor")


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    hist = await bus.history(s.t_kline, 1)
    last_open = decode_md(hist[0]).open_time if hist else 0
    lat = reg.hist("ingest", "ws frame received -> published on bus")
    topics = {Kline: s.t_kline, BookTicker: s.t_book, Trade: s.t_trade}
    names = {Kline: "klines", BookTicker: "book_updates", Trade: "trades"}

    async def backfill() -> None:
        nonlocal last_open
        now = int(time.time() * 1000)
        start = last_open + s.interval_ms if last_open else now - s.history_bars * s.interval_ms
        start = max(start, now - s.history_bars * s.interval_ms)
        t0 = time.perf_counter()
        raw = await fetch_klines(s.symbol, s.interval_ms, s.interval, start, now, base=s.binance_rest)
        n = 0
        for k in raw_to_klines(s.symbol, raw):
            if k.open_time > last_open:
                await bus.publish(s.t_kline, encode(k))
                last_open = k.open_time
                n += 1
        reg.inc("backfilled_bars", n)
        log.info("backfilled %d bars in %.2fs", n, time.perf_counter() - t0)

    url = stream_url(s.binance_ws, s.symbol, s.interval)
    async for msg in stream_market_data(url, on_connect=backfill):
        cls = type(msg)
        if cls is Kline:
            if msg.open_time <= last_open:
                reg.inc("duplicate_bars")
                continue
            if msg.open_time != last_open + s.interval_ms:
                reg.inc("gaps_detected")
                await backfill()  # fills [last_open, msg) then we publish msg
                if msg.open_time <= last_open:
                    continue
            last_open = msg.open_time
            reg.set("exchange_lag_ms", time.time() * 1000 - msg.event_time)
        await bus.publish(topics[cls], encode(msg))
        lat.record(time.perf_counter_ns() - msg.recv_ns)
        reg.inc(names[cls])


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("ingestor", run)


if __name__ == "__main__":
    main()
