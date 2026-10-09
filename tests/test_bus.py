import asyncio

import pytest

from qforecast.bus import MemoryBus, RedisStreamBus


async def _collect(it, n):
    out = []
    async for item in it:
        out.append(item)
        if len(out) == n:
            break
    return out


async def _exercise(bus):
    await bus.publish("a", b"old")
    live = asyncio.create_task(_collect(bus.subscribe(["a", "b"]), 2))
    replay = asyncio.create_task(_collect(bus.subscribe(["a"], replay=True), 2))
    await asyncio.sleep(0.05)
    await bus.publish("a", b"x")
    await bus.publish("b", b"y")
    got_live = await asyncio.wait_for(live, 2)
    got_replay = await asyncio.wait_for(replay, 2)
    assert sorted(got_live) == [("a", b"x"), ("b", b"y")]
    assert got_replay == [("a", b"old"), ("a", b"x")]
    assert await bus.history("a", 5) == [b"old", b"x"]
    assert await bus.history("a", 1) == [b"x"]


async def test_memory_bus():
    async with MemoryBus() as bus:
        await _exercise(bus)


async def test_memory_bus_drops_oldest_for_slow_consumer():
    bus = MemoryBus(queue_size=3)
    it = bus.subscribe(["t"])
    first = asyncio.create_task(it.__anext__())
    await asyncio.sleep(0)
    for i in range(10):
        await bus.publish("t", bytes([i]))
    # The consumer never ran while 10 messages arrived: it sees only the newest 3.
    got = [await first] + [await it.__anext__() for _ in range(2)]
    assert [p for _, p in got] == [bytes([7]), bytes([8]), bytes([9])]
    assert bus.dropped == 7


async def test_redis_stream_bus():
    fakeredis = pytest.importorskip("fakeredis")
    client = fakeredis.FakeAsyncRedis()
    async with RedisStreamBus("redis://fake", client=client) as bus:
        await _exercise(bus)
        assert bus.batches >= 1
