"""Run every service in one process over the in-memory bus (dev / demo mode).

    python -m qforecast.services.all

Same service code as the distributed deployment. Uses the in-memory bus by default;
set QF_BUS_URL=redis://... to run all services in one process over Redis.
"""

from __future__ import annotations

import asyncio
import logging

from qforecast.config import Settings
from qforecast.services import analytics, gateway, ingestor, recorder, signal
from qforecast.services.base import apply_cpu_policy, bus_for, serve, setup_logging

log = logging.getLogger("all")

SERVICES = {"ingestor": ingestor.run, "signal": signal.run, "analytics": analytics.run,
            "gateway": gateway.run}


async def supervise(name: str, fn, s: Settings, bus) -> None:
    """Restart a crashed service with exponential backoff (like a container restart policy)."""
    backoff = 1.0
    while True:
        try:
            await serve(name, fn, s, bus)
            return
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("service %s crashed; restarting in %.0fs", name, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30.0)


async def main(s: Settings) -> None:
    async with bus_for(s) as bus:
        services = dict(SERVICES, **({"recorder": recorder.run} if s.record else {}))
        await asyncio.gather(*(supervise(name, fn, s, bus) for name, fn in services.items()))


if __name__ == "__main__":
    setup_logging()
    settings = Settings.from_env()
    apply_cpu_policy(settings)
    try:
        asyncio.run(main(settings))
    except KeyboardInterrupt:
        pass
