"""Binance spot market-data adapters: concurrent REST backfill + resilient WebSocket."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator, Callable

import aiohttp
import msgspec
import numpy as np
import orjson

from qforecast.schemas import BookTicker, Kline, Trade

log = logging.getLogger(__name__)

KLINE_LIMIT = 1000  # max rows per REST request


# --------------------------------------------------------------------------- REST
async def _get_klines(session: aiohttp.ClientSession, base: str, symbol: str, interval: str,
                      start_ms: int, end_ms: int, sem: asyncio.Semaphore) -> list:
    params = {"symbol": symbol, "interval": interval, "startTime": start_ms,
              "endTime": end_ms, "limit": KLINE_LIMIT}
    for attempt in range(5):
        async with sem:
            try:
                async with session.get(f"{base}/api/v3/klines", params=params) as r:
                    if r.status == 429 or r.status == 418:
                        retry = float(r.headers.get("Retry-After", 2 ** attempt))
                        log.warning("rate limited, sleeping %.1fs", retry)
                        await asyncio.sleep(retry)
                        continue
                    r.raise_for_status()
                    return orjson.loads(await r.read())
            except (TimeoutError, aiohttp.ClientError) as e:
                log.warning("kline request failed (%s), retry %d", e, attempt + 1)
                await asyncio.sleep(0.5 * 2 ** attempt)
    raise RuntimeError(f"failed to fetch klines {start_ms}-{end_ms}")


async def fetch_klines(symbol: str, interval_ms: int, interval: str, start_ms: int, end_ms: int,
                       base: str = "https://api.binance.com", concurrency: int = 8,
                       session: aiohttp.ClientSession | None = None) -> np.ndarray:
    """Fetch closed klines in [start_ms, end_ms) with ``concurrency`` parallel requests.

    Returns a raw float64 matrix with Binance's column order
    (open_time, o, h, l, c, v, close_time, quote_vol, trades, taker_buy_base, ...).
    The time range is split into 1000-bar chunks up-front so pages are fetched in
    parallel instead of the sequential ``since = last + 1`` pagination loop.
    """
    span = KLINE_LIMIT * interval_ms
    chunks = [(s, min(s + span, end_ms) - 1) for s in range(start_ms, end_ms, span)]
    sem = asyncio.Semaphore(concurrency)
    own = session is None
    session = session or aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30))
    try:
        pages = await asyncio.gather(*(_get_klines(session, base, symbol, interval, s, e, sem)
                                       for s, e in chunks))
    finally:
        if own:
            await session.close()
    rows = [r[:11] for page in pages for r in page]
    if not rows:
        return np.empty((0, 11))
    raw = np.asarray(rows, dtype=np.float64)
    raw = raw[np.unique(raw[:, 0], return_index=True)[1]]  # sort + dedupe by open_time
    now_ms = time.time() * 1000
    return raw[raw[:, 6] < now_ms]  # closed bars only


def raw_to_bars(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binance raw matrix -> (open_time int64[N], bars float64[N, 7] in features.B_* layout)."""
    open_time = raw[:, 0].astype(np.int64)
    bars = np.ascontiguousarray(raw[:, [1, 2, 3, 4, 5, 8, 9]])
    return open_time, bars


def raw_to_klines(symbol: str, raw: np.ndarray) -> list[Kline]:
    return [Kline(symbol, int(r[0]), int(r[6]), r[1], r[2], r[3], r[4], r[5], int(r[8]), r[9])
            for r in raw.tolist()]  # tolist(): native floats, msgspec can't encode np.float64


# --------------------------------------------------------------------------- WebSocket
def stream_url(base: str, symbol: str, interval: str) -> str:
    s = symbol.lower()
    return f"{base}/stream?streams={s}@kline_{interval}/{s}@bookTicker/{s}@aggTrade"


# Wire schemas of Binance's JSON. Decoding straight into typed structs skips the
# intermediate dict and the per-field float(str) calls of the orjson path (~1.6x
# faster, see benchmarks/bench_io.py). strict=False lets msgspec parse Binance's
# quoted decimals ("2433.88000000") as floats; unknown fields are skipped.
class _Frame(msgspec.Struct):
    stream: str
    data: msgspec.Raw  # left unparsed until we know which schema applies


class _BookWire(msgspec.Struct):
    u: int
    s: str
    b: float
    B: float
    a: float
    A: float


class _TradeWire(msgspec.Struct):
    s: str
    p: float
    q: float
    m: bool
    T: int
    E: int


class _KlineBody(msgspec.Struct):
    t: int
    T: int
    s: str
    o: float
    h: float
    l: float  # noqa: E741 - Binance field name
    c: float
    v: float
    n: int
    V: float
    x: bool


class _KlineWire(msgspec.Struct):
    E: int
    k: _KlineBody


_frame_dec = msgspec.json.Decoder(_Frame)
_book_dec = msgspec.json.Decoder(_BookWire, strict=False)
_trade_dec = msgspec.json.Decoder(_TradeWire, strict=False)
_kline_dec = msgspec.json.Decoder(_KlineWire, strict=False)


def parse_message(raw: bytes | str, recv_ns: int) -> Kline | BookTicker | Trade | None:
    """Parse a combined-stream frame. Returns None for frames we don't forward
    (e.g. in-progress klines)."""
    f = _frame_dec.decode(raw)
    stream = f.stream
    if stream.endswith("@bookTicker"):
        d = _book_dec.decode(f.data)
        return BookTicker(d.s, d.u, d.b, d.B, d.a, d.A, recv_ns)
    if stream.endswith("@aggTrade"):
        d = _trade_dec.decode(f.data)
        return Trade(d.s, d.p, d.q, d.m, d.T, d.E, recv_ns)
    if "@kline_" in stream:
        w = _kline_dec.decode(f.data)
        k = w.k
        if not k.x:
            return None
        return Kline(k.s, k.t, k.T, k.o, k.h, k.l, k.c, k.v, k.n, k.V, w.E, recv_ns)
    return None


async def stream_market_data(url: str, on_connect: Callable[[], object] | None = None,
                             max_backoff: float = 30.0) -> AsyncIterator[Kline | BookTicker | Trade]:
    """Yield parsed messages forever, reconnecting with exponential backoff + jitter.

    ``on_connect`` (sync or async) runs after every (re)connection — the ingestor uses
    it to REST-backfill any klines missed while disconnected.
    """
    import websockets

    backoff = 0.5
    while True:
        try:
            # compression=None: skip permessage-deflate -> less CPU and latency per frame.
            async with websockets.connect(url, compression=None, ping_interval=20,
                                          ping_timeout=20, max_queue=4096,
                                          open_timeout=10) as ws:
                log.info("connected %s", url)
                backoff = 0.5
                if on_connect is not None:
                    res = on_connect()
                    if asyncio.iscoroutine(res):
                        await res
                async for frame in ws:
                    recv_ns = time.perf_counter_ns()
                    msg = parse_message(frame, recv_ns)
                    if msg is not None:
                        yield msg
        except asyncio.CancelledError:
            raise
        except Exception as e:  # network errors, server-side 24h disconnects, ...
            delay = min(max_backoff, backoff) * (0.5 + random.random())
            log.warning("websocket error: %r — reconnecting in %.1fs", e, delay)
            await asyncio.sleep(delay)
            backoff *= 2
