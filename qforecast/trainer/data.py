"""Historical data download with an incremental, compressed on-disk cache (QCOL)."""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path

import numpy as np

from qforecast.config import Settings
from qforecast.exchange.binance import fetch_klines
from qforecast.storage import columnar

log = logging.getLogger(__name__)


KLINE_COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
                 "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]


def cache_path(s: Settings) -> Path:
    return s.data_dir / f"{s.symbol}_{s.interval}.qcol"


def _read_cache(path: Path) -> np.ndarray:
    if path.exists():
        return np.ascontiguousarray(columnar.load(path))
    legacy = path.with_suffix(".npy")  # cache written by older versions
    if legacy.exists():
        return np.load(legacy)
    return np.empty((0, 11))


async def load_history(s: Settings, days: int) -> np.ndarray:
    """Return the raw kline matrix for the last ``days`` days, downloading only what
    is missing from the cache."""
    path = cache_path(s)
    now_ms = int(time.time() * 1000) // s.interval_ms * s.interval_ms
    start_ms = now_ms - days * 86_400_000
    cached = _read_cache(path)

    parts = [cached]
    if len(cached) == 0 or cached[0, 0] > start_ms:
        end = int(cached[0, 0]) if len(cached) else now_ms
        parts.append(await _fetch(s, start_ms, end))
    if len(cached):
        parts.append(await _fetch(s, int(cached[-1, 0]) + s.interval_ms, now_ms))

    raw = np.concatenate([p for p in parts if len(p)])
    raw = raw[np.unique(raw[:, 0], return_index=True)[1]]
    path.parent.mkdir(parents=True, exist_ok=True)
    size = columnar.save(path, raw, KLINE_COLUMNS)
    log.info("cache %s: %.2f MB (%.1fx smaller than float64)", path.name, size / 1e6, raw.nbytes / size)
    raw = raw[raw[:, 0] >= start_ms]
    log.info("history: %d bars (%s .. %s)", len(raw),
             time.strftime("%Y-%m-%d %H:%M", time.gmtime(raw[0, 0] / 1000)),
             time.strftime("%Y-%m-%d %H:%M", time.gmtime(raw[-1, 0] / 1000)))
    return raw


async def _fetch(s: Settings, start_ms: int, end_ms: int) -> np.ndarray:
    if end_ms <= start_ms:
        return np.empty((0, 11))
    t0 = time.perf_counter()
    raw = await fetch_klines(s.symbol, s.interval_ms, s.interval, start_ms, end_ms, base=s.binance_rest)
    log.info("downloaded %d bars in %.2fs", len(raw), time.perf_counter() - t0)
    return raw


def load_history_sync(s: Settings, days: int) -> np.ndarray:
    return asyncio.run(load_history(s, days))
