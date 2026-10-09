"""Gateway: the public edge. REST + WebSocket fan-out + Prometheus metrics + dashboard.

* Each bus message is JSON-encoded **once** and the same bytes are fanned out to
  every WebSocket client.
* Every client has a bounded queue; a slow client gets conflated (oldest dropped)
  instead of back-pressuring the bus consumer or other clients.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec
import orjson
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, PlainTextResponse, Response

from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.metrics import Registry
from qforecast.schemas import decode_analysis, decode_forecast
from qforecast.services.base import OPS_TOPIC

log = logging.getLogger("gateway")
STATIC = Path(__file__).parent / "static"


def _json(obj) -> bytes:
    return orjson.dumps(msgspec.to_builtins(obj))


class Hub:
    def __init__(self, s: Settings, bus: Bus, reg: Registry):
        self.s, self.bus, self.reg = s, bus, reg
        self.forecast: bytes | None = None
        self.analysis: bytes | None = None
        self.forecast_history: deque[bytes] = deque(maxlen=500)
        self.services: dict[str, dict] = {}
        self.clients: set[asyncio.Queue] = set()
        self.started = time.time()

    def broadcast(self, frame: bytes) -> None:
        for q in self.clients:
            if q.full():
                q.get_nowait()
                self.reg.inc("ws_conflated")
            q.put_nowait(frame)

    async def consume(self) -> None:
        s = self.s
        for raw in await self.bus.history(s.t_forecast, 500):
            self.forecast = _json(decode_forecast(raw))
            self.forecast_history.append(self.forecast)
        for raw in await self.bus.history(s.t_analysis, 1):
            self.analysis = _json(decode_analysis(raw))
        h = self.reg.hist("gateway_fanout", "bus message -> queued to all ws clients")
        async for topic, payload in self.bus.subscribe([s.t_forecast, s.t_analysis, OPS_TOPIC]):
            t0 = time.perf_counter_ns()
            if topic == OPS_TOPIC:
                snap = orjson.loads(payload)
                self.services[snap["service"]] = snap
                continue
            if topic == s.t_forecast:
                body = _json(decode_forecast(payload))
                self.forecast = body
                self.forecast_history.append(body)
                kind = b"forecast"
            else:
                body = self.analysis = _json(decode_analysis(payload))
                kind = b"analysis"
            self.broadcast(b'{"type":"' + kind + b'","data":' + body + b"}")
            h.record(time.perf_counter_ns() - t0)


def create_app(s: Settings, bus: Bus, reg: Registry | None = None) -> FastAPI:
    reg = reg or Registry("gateway")
    hub = Hub(s, bus, reg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(hub.consume(), name="gateway-consumer")
        yield
        task.cancel()

    app = FastAPI(title="qforecast gateway", version="0.1.0", lifespan=lifespan)
    app.state.hub = hub

    def raw_json(body: bytes | None) -> Response:
        if body is None:
            return Response(b'{"detail":"not available yet"}', status_code=503,
                            media_type="application/json")
        return Response(body, media_type="application/json")

    @app.get("/health")
    async def health():
        return {"status": "ok", "uptime_s": round(time.time() - hub.started, 1),
                "symbol": s.symbol, "ws_clients": len(hub.clients),
                "has_forecast": hub.forecast is not None}

    @app.get("/v1/forecast")
    async def forecast():
        return raw_json(hub.forecast)

    @app.get("/v1/forecast/history")
    async def forecast_history(limit: int = 100):
        items = list(hub.forecast_history)[-limit:]
        return Response(b"[" + b",".join(items) + b"]", media_type="application/json")

    @app.get("/v1/analysis")
    async def analysis():
        return raw_json(hub.analysis)

    @app.get("/v1/snapshot")
    async def snapshot():
        f = hub.forecast or b"null"
        a = hub.analysis or b"null"
        return Response(b'{"forecast":' + f + b',"analysis":' + a + b"}", media_type="application/json")

    @app.get("/v1/ops")
    async def ops():
        return {"gateway": reg.snapshot(), **hub.services}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics():
        lines = [reg.prometheus()]
        for name, snap in hub.services.items():
            for metric, summ in snap.get("latency", {}).items():
                for q in ("p50_us", "p99_us", "p999_us"):
                    if summ.get(q) is not None:
                        lines.append(f'qf_latency_us{{service="{name}",stage="{metric}",'
                                     f'quantile="{q[1:-3]}"}} {summ[q]}')
            for metric, v in snap.get("counters", {}).items():
                lines.append(f'qf_{metric}_total{{service="{name}"}} {v}')
        return "\n".join(lines) + "\n"

    @app.websocket("/v1/stream")
    async def stream(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=64)
        for body, kind in ((hub.forecast, b"forecast"), (hub.analysis, b"analysis")):
            if body:
                q.put_nowait(b'{"type":"' + kind + b'","data":' + body + b"}")
        hub.clients.add(q)
        reg.set("ws_clients", len(hub.clients))
        try:
            while True:
                frame: bytes = await q.get()
                await ws.send_text(frame.decode())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            hub.clients.discard(q)
            reg.set("ws_clients", len(hub.clients))

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    import uvicorn

    app = create_app(s, bus, reg)
    config = uvicorn.Config(app, host=s.gateway_host, port=s.gateway_port, log_level="warning",
                            ws="websockets")
    log.info("listening on http://%s:%d", s.gateway_host, s.gateway_port)
    await uvicorn.Server(config).serve()


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("gateway", run)


if __name__ == "__main__":
    main()
