"""The service supervisor restarts a crashed stage with exponential backoff, stops on a clean exit."""

import asyncio

from qforecast.services import all as A


def test_supervisor_restarts_a_crashed_service_with_backoff(monkeypatch):
    calls, sleeps = [], []

    async def flaky_serve(name, fn, s, bus):
        calls.append(name)
        if len(calls) < 4:
            raise RuntimeError("crash")

    async def no_wait(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(A, "serve", flaky_serve)
    monkeypatch.setattr(A.asyncio, "sleep", no_wait)
    asyncio.run(A.supervise("ingestor", None, None, None))
    assert calls == ["ingestor"] * 4
    assert sleeps == [1.0, 2.0, 4.0]  # doubles after each crash


def test_supervisor_does_not_swallow_cancellation(monkeypatch):
    async def cancelled(name, fn, s, bus):
        raise asyncio.CancelledError

    monkeypatch.setattr(A, "serve", cancelled)
    try:
        asyncio.run(A.supervise("gateway", None, None, None))
    except asyncio.CancelledError:
        return
    raise AssertionError("cancellation must propagate so shutdown works")
