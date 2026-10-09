"""Recorder: captures every market-data message from the bus into compressed tick logs.

Off the hot path by construction: it is just another consumer of the bus, and its
disk I/O (zstd compression + write) runs on a dedicated thread, so neither the
ingestor nor the signal service ever waits on the disk. Replay with
``qforecast.storage.ticklog.read_dir``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.metrics import Registry
from qforecast.storage.ticklog import TickLogWriter

log = logging.getLogger("recorder")


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    topics = [s.t_kline, s.t_book, s.t_trade]
    writer = TickLogWriter(s.data_dir / "ticks" / s.sym, topics,
                           block_bytes=s.record_block_kb * 1024)
    io = ThreadPoolExecutor(1, thread_name_prefix="recorder-io")  # 1 worker = ordered blocks
    loop = asyncio.get_running_loop()
    h_append = reg.hist("record_append", "buffer one message (event loop)")
    h_block = reg.hist("record_block", "compress + write one block (I/O thread)")
    inflight: set[asyncio.Future] = set()

    def write(raw: bytes) -> None:
        t0 = time.perf_counter_ns()
        writer.write_block(raw)
        h_block.record(time.perf_counter_ns() - t0)

    def flush() -> None:
        raw = writer.take_block()
        if raw is None:
            return
        fut = loop.run_in_executor(io, write, raw)
        inflight.add(fut)
        fut.add_done_callback(inflight.discard)
        reg.set("record_raw_bytes", writer.raw_bytes)
        reg.set("record_disk_bytes", writer.written_bytes)

    async def timer() -> None:  # bound data loss / latency to disk on quiet markets
        while True:
            await asyncio.sleep(s.record_flush_ms / 1000)
            flush()

    tick = asyncio.create_task(timer())
    log.info("recording %s -> %s", topics, writer.dir)
    try:
        async for topic, payload in bus.subscribe(topics):
            t0 = time.perf_counter_ns()
            full = writer.append(topic, payload)
            h_append.record(time.perf_counter_ns() - t0)
            reg.inc("messages")
            if full:
                flush()
    finally:
        tick.cancel()
        flush()
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
        await loop.run_in_executor(io, writer.close)
        io.shutdown(wait=True)
        log.info("closed %s (%d messages, %.2f MB raw -> %.2f MB on disk)", writer.path,
                 writer.messages, writer.raw_bytes / 1e6, writer.written_bytes / 1e6)


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("recorder", run)


if __name__ == "__main__":
    main()
