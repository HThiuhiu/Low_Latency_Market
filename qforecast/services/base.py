"""Shared service bootstrap: logging, bus lifecycle, metrics heartbeat, shutdown."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from collections.abc import Awaitable, Callable

import orjson

from qforecast.bus import Bus, make_bus
from qforecast.config import Settings
from qforecast.core.cpu import disable_power_throttling, pin_cpus
from qforecast.core.metrics import Registry

OPS_TOPIC = "ops.metrics"

ServiceFn = Callable[[Settings, Bus, Registry], Awaitable[None]]


def setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("QF_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def apply_cpu_policy(s: Settings) -> None:
    """CPU placement at start-up: opt out of Windows EcoQoS, then optional affinity pinning."""
    if s.high_qos and disable_power_throttling():
        logging.getLogger("qforecast").info("EcoQoS disabled (high-QoS scheduling)")
    pin_cpus(s.cpu_affinity)


def bus_for(s: Settings) -> Bus:
    return make_bus(s.bus_url, maxlen=s.stream_maxlen, topic_maxlen={s.t_kline: s.kline_stream_maxlen})


async def metrics_heartbeat(bus: Bus, reg: Registry, every_s: float = 5.0) -> None:
    """Every service publishes its metrics to the bus; the gateway aggregates them."""
    while True:
        await asyncio.sleep(every_s)
        await bus.publish(OPS_TOPIC, orjson.dumps(reg.snapshot()))


async def serve(name: str, fn: ServiceFn, s: Settings, bus: Bus) -> None:
    reg = Registry(name)
    hb = asyncio.create_task(metrics_heartbeat(bus, reg), name=f"{name}-heartbeat")
    try:
        await fn(s, bus, reg)
    finally:
        hb.cancel()


def run(name: str, fn: ServiceFn) -> None:
    """Entry point for a standalone service process."""
    setup_logging()
    s = Settings.from_env()
    log = logging.getLogger(name)
    apply_cpu_policy(s)

    async def main() -> None:
        async with bus_for(s) as bus:
            log.info("starting (bus=%s, symbol=%s)", s.bus_url, s.symbol)
            await serve(name, fn, s, bus)

    try:
        import uvloop  # type: ignore[import-not-found]  # Linux/macOS only

        uvloop.install()
    except ImportError:
        pass
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("stopped")
