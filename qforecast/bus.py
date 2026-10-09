"""Message bus abstraction.

Two transports behind one interface:

* ``MemoryBus``      – in-process fan-out (asyncio queues). Used for the single-process
                        "monolith" mode, tests and benchmarks. Zero serialization hops.
* ``RedisStreamBus`` – Redis Streams. Durable, replayable (a restarted service warms up
                        from the stream instead of hitting the exchange), and every
                        consumer reads at its own pace. Publishes are micro-batched into
                        one pipeline per event-loop tick to amortise the network RTT.

Payloads are opaque bytes (msgpack-encoded ``qforecast.schemas`` structs).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import AsyncIterator

log = logging.getLogger(__name__)


class Bus(ABC):
    async def start(self) -> None:  # noqa: B027 - optional hook, default no-op
        pass

    async def close(self) -> None:  # noqa: B027 - optional hook, default no-op
        pass

    @abstractmethod
    async def publish(self, topic: str, payload: bytes) -> None: ...

    @abstractmethod
    def subscribe(self, topics: list[str], replay: bool = False) -> AsyncIterator[tuple[str, bytes]]:
        """Yield ``(topic, payload)``. ``replay=True`` first delivers retained history."""

    @abstractmethod
    async def history(self, topic: str, count: int) -> list[bytes]:
        """Return up to the last ``count`` payloads of ``topic`` (oldest first)."""

    async def __aenter__(self) -> Bus:
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()


class MemoryBus(Bus):
    def __init__(self, retention: int = 5_000, queue_size: int = 10_000):
        self._retention = retention
        self._queue_size = queue_size
        self._topics: dict[str, deque[bytes]] = {}
        self._subs: dict[str, list[asyncio.Queue]] = {}
        self.dropped = 0

    async def publish(self, topic: str, payload: bytes) -> None:
        self._topics.setdefault(topic, deque(maxlen=self._retention)).append(payload)
        for q in self._subs.get(topic, ()):
            if q.full():  # slow consumer: drop oldest, never block the publisher
                q.get_nowait()
                self.dropped += 1
            q.put_nowait((topic, payload))

    async def subscribe(self, topics: list[str], replay: bool = False) -> AsyncIterator[tuple[str, bytes]]:
        q: asyncio.Queue = asyncio.Queue(self._queue_size)
        if replay:
            for t in topics:
                for p in self._topics.get(t, ()):
                    if q.full():
                        q.get_nowait()
                    q.put_nowait((t, p))
        for t in topics:
            self._subs.setdefault(t, []).append(q)
        try:
            while True:
                yield await q.get()
        finally:
            for t in topics:
                self._subs[t].remove(q)

    async def history(self, topic: str, count: int) -> list[bytes]:
        items = self._topics.get(topic, ())
        return list(items)[-count:]


class RedisStreamBus(Bus):
    FIELD = b"d"

    def __init__(self, url: str, maxlen: int = 20_000, topic_maxlen: dict[str, int] | None = None,
                 client=None, max_pending: int = 100_000):
        self._url = url
        self._maxlen = maxlen
        self._topic_maxlen = topic_maxlen or {}
        self._client = client
        self._pending: deque[tuple[str, bytes]] = deque()
        self._max_pending = max_pending
        self._wakeup = asyncio.Event()
        self._flusher: asyncio.Task | None = None
        self.dropped = 0
        self.batches = 0

    async def start(self) -> None:
        if self._client is None:
            import redis.asyncio as redis

            self._client = redis.from_url(self._url, decode_responses=False)
        await self._client.ping()
        self._flusher = asyncio.create_task(self._flush_loop(), name="redis-bus-flusher")

    async def close(self) -> None:
        if self._flusher:
            await self._drain()
            self._flusher.cancel()
            try:
                await self._flusher
            except asyncio.CancelledError:
                pass
        if self._client is not None:
            await self._client.aclose()

    async def publish(self, topic: str, payload: bytes) -> None:
        if len(self._pending) >= self._max_pending:  # Redis unreachable: bound memory
            self._pending.popleft()
            self.dropped += 1
        self._pending.append((topic, payload))
        self._wakeup.set()

    async def _drain(self) -> None:
        while self._pending:
            pipe = self._client.pipeline(transaction=False)
            n = 0
            while self._pending and n < 1_000:
                topic, payload = self._pending.popleft()
                pipe.xadd(topic, {self.FIELD: payload},
                          maxlen=self._topic_maxlen.get(topic, self._maxlen), approximate=True)
                n += 1
            await pipe.execute()
            self.batches += 1

    async def _flush_loop(self) -> None:
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            try:
                await self._drain()
            except asyncio.CancelledError:
                raise
            except Exception:  # keep running; messages already popped are lost
                log.exception("redis publish failed")
                await asyncio.sleep(0.5)

    async def _last_id(self, topic: str) -> bytes:
        last = await self._client.xrevrange(topic, count=1)
        return last[0][0] if last else b"0-0"

    async def subscribe(self, topics: list[str], replay: bool = False) -> AsyncIterator[tuple[str, bytes]]:
        ids: dict[str, bytes] = {}
        for t in topics:
            # Resolve '$' to a concrete id up-front so nothing published between
            # two XREAD calls can be missed.
            ids[t] = b"0-0" if replay else await self._last_id(t)
        while True:
            resp = await self._client.xread(ids, count=512, block=5_000)
            for stream, entries in resp or ():
                topic = stream.decode() if isinstance(stream, bytes) else stream
                for entry_id, fields in entries:
                    ids[topic] = entry_id
                    yield topic, fields[self.FIELD]

    async def history(self, topic: str, count: int) -> list[bytes]:
        rows = await self._client.xrevrange(topic, count=count)
        return [fields[self.FIELD] for _, fields in reversed(rows)]


def make_bus(url: str, maxlen: int = 20_000, topic_maxlen: dict[str, int] | None = None) -> Bus:
    if url.startswith("memory://"):
        return MemoryBus(retention=maxlen)
    if url.startswith(("redis://", "rediss://", "unix://")):
        return RedisStreamBus(url, maxlen=maxlen, topic_maxlen=topic_maxlen)
    raise ValueError(f"unsupported bus url: {url}")
