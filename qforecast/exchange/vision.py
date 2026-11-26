"""USDⓈ-M perpetual history from data.binance.vision archives + fapi REST, cached on disk.

    from qforecast.exchange.vision import PerpStore
    async with PerpStore(Path("data/um")) as store:
        syms = await store.symbols()                                # every crypto USDT perp ever listed
        fr = await store.funding(symbol, start_ms, end_ms)          # (n, 2) [time_ms, rate]

Why the archive and not only REST: it keeps DELISTED contracts, so a universe built from it is
point-in-time (no survivorship bias: the coins that later died are still in the past rankings).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import aiohttp
import numpy as np

log = logging.getLogger(__name__)

VISION = "https://data.binance.vision"
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FAPI = "https://fapi.binance.com"
HOUR = 3_600_000
DAY = 86_400_000

# bases that are not crypto risk (stablecoins, FX, indices, metals) even if the exchange labels them COIN
STABLE_BASES = {"USDC", "FDUSD", "TUSD", "BUSD", "USDP", "DAI", "EUR", "USDE", "USD1", "PYUSD",
                "XUSD", "RLUSD", "BFUSD", "U"}
NON_CRYPTO_HINT = {"BTCDOM", "DEFI", "FOOTBALL", "BLUEBIRD", "ALL", "XAU", "XAG", "XPT", "XPD",
                   "PAXG", "XAUT"}


def _month_starts(start_ms: int, end_ms: int) -> list[datetime]:
    d = datetime.fromtimestamp(start_ms / 1000, UTC).replace(day=1, hour=0, minute=0, second=0,
                                                              microsecond=0)
    end = datetime.fromtimestamp(end_ms / 1000, UTC)
    out = []
    while d <= end:
        out.append(d)
        d = _next_month(d)
    return out


def _ms(d: datetime) -> int:
    return int(d.timestamp() * 1000)


def _next_month(d: datetime) -> datetime:
    return d.replace(year=d.year + 1, month=1) if d.month == 12 else d.replace(month=d.month + 1)


def parse_csv(raw: bytes, ncols: int) -> np.ndarray:
    """CSV bytes (with or without header) -> float64 (n, ncols)."""
    text = raw.decode("utf-8", "replace")
    first = text.split("\n", 1)[0]
    skip = 1 if re.search(r"[a-df-z_]", first.split(",")[0] or "x") else 0
    try:
        import pandas as pd

        df = pd.read_csv(io.StringIO(text), header=None, skiprows=skip, usecols=range(ncols))
        return df.to_numpy(dtype=np.float64)
    except ImportError:  # pragma: no cover
        return np.loadtxt(io.StringIO(text), delimiter=",", skiprows=skip, usecols=range(ncols),
                          ndmin=2)


def is_crypto(row: dict) -> bool:
    sym, base, typ = row["symbol"], row.get("base", ""), row.get("type", "")
    if "_" in sym or not sym.endswith("USDT"):
        return False
    if typ and typ != "COIN":
        return False
    b = re.sub(r"^1000+", "", base)
    return b not in STABLE_BASES and b not in NON_CRYPTO_HINT


class PerpStore:
    """Download + cache USDⓈ-M perp history. All methods are idempotent and incremental."""

    def __init__(self, root: Path, concurrency: int = 16, timeout_s: float = 60.0):
        self.root = Path(root)
        self.sem = asyncio.Semaphore(concurrency)
        self.timeout_s = timeout_s
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> PerpStore:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _s(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_s))
        return self._session

    async def _get(self, url: str, params: dict | None = None) -> bytes | None:
        """GET with retry; 404 / 400 / 403 -> None (file or symbol does not exist)."""
        for attempt in range(5):
            async with self.sem:
                try:
                    async with self._s().get(url, params=params) as r:
                        if r.status in (404, 400, 403):
                            return None
                        if r.status in (418, 429):
                            wait = float(r.headers.get("Retry-After", 2 ** attempt))
                            log.warning("rate limited, sleeping %.0fs", wait)
                            await asyncio.sleep(wait)
                            continue
                        r.raise_for_status()
                        return await r.read()
                except (TimeoutError, aiohttp.ClientError) as e:
                    log.warning("GET %s failed (%s), retry %d", url, e, attempt + 1)
                    await asyncio.sleep(1.0 * 2 ** attempt)
        raise RuntimeError(f"giving up on {url}")

    async def _zip_csv(self, path: str, ncols: int) -> np.ndarray | None:
        raw = await self._get(f"{VISION}/{path}")
        if raw is None:
            return None
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            data = z.read(z.namelist()[0])
        return await asyncio.to_thread(parse_csv, data, ncols)

    # ------------------------------------------------------------------ symbols
    async def symbols(self, max_age_s: float = 86_400) -> list[dict]:
        """Every USDT-margined crypto perpetual ever listed that we can see.

        exchangeInfo (current + recently delisted, with ``underlyingType``) plus symbols that
        only survive in the archive listing (delisted long ago). Cached in symbols.json.
        """
        path = self.root / "symbols.json"
        if path.exists() and time.time() - path.stat().st_mtime < max_age_s:
            return json.loads(path.read_text())
        info = await self._get(f"{FAPI}/fapi/v1/exchangeInfo")
        if not info:
            raise RuntimeError("exchangeInfo unavailable: cannot classify symbols")
        rows: dict[str, dict] = {}
        known: set[str] = set()  # every symbol exchangeInfo describes, whatever its type
        for s in json.loads(info)["symbols"]:
            known.add(s["symbol"])
            # TRADIFI_PERPETUAL = stocks, ETFs, commodities, FX: not crypto even when USDT-margined
            if s.get("contractType") != "PERPETUAL" or s.get("quoteAsset") != "USDT":
                continue
            rows[s["symbol"]] = {"symbol": s["symbol"], "base": s["baseAsset"],
                                 "status": s["status"], "type": s.get("underlyingType", ""),
                                 "onboard_ms": s.get("onboardDate", 0)}
        for sym in await self._archive_symbols():
            if sym.endswith("USDT") and sym not in known:  # delisted long ago: only in the archive
                rows[sym] = {"symbol": sym, "base": sym[:-4], "status": "ARCHIVE", "type": "",
                             "onboard_ms": 0}
        out = [r for r in rows.values() if is_crypto(r)]
        self.root.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        return out

    async def _archive_symbols(self) -> list[str]:
        prefix = "data/futures/um/monthly/klines/"
        marker, out = "", []
        while True:
            raw = await self._get(S3_LIST, {"delimiter": "/", "prefix": prefix, "marker": marker})
            if not raw:
                break
            text = raw.decode()
            out += re.findall(r"<Prefix>" + re.escape(prefix) + r"([^/<]+)/</Prefix>", text)
            nxt = re.search(r"<NextMarker>([^<]+)</NextMarker>", text)
            if "<IsTruncated>true</IsTruncated>" not in text or not nxt:
                break
            marker = nxt.group(1)
        return out

    # ------------------------------------------------------------------ funding
    async def funding(self, sym: str, start_ms: int, end_ms: int) -> np.ndarray:
        """(n, 2) [funding_time_ms, rate] (rate per funding interval, e.g. 8h). Past months from the
        monthly archive (cached for good), the current month from REST (never cached)."""
        now = datetime.now(UTC)
        this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        parts = []
        for d in _month_starts(start_ms, end_ms):
            if d >= this_month:
                continue
            path = self.root / "funding" / sym / f"{d:%Y-%m}.npy"
            if path.exists():
                parts.append(np.load(path))
                continue
            name = f"{sym}/{sym}-fundingRate-{d:%Y-%m}.zip"
            a = await self._zip_csv(f"data/futures/um/monthly/fundingRate/{name}", 3)
            a = a[:, [0, 2]] if a is not None else await self._funding_rest(sym, _ms(d), _ms(_next_month(d)))
            a = a if a is not None else np.empty((0, 2))
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, a)
            parts.append(a)
        if end_ms > _ms(this_month):
            a = await self._funding_rest(sym, max(start_ms, _ms(this_month)), end_ms)
            if a is not None:
                parts.append(a)
        parts = [p.reshape(-1, 2) for p in parts if len(p)]
        if not parts:
            return np.empty((0, 2))
        a = np.concatenate(parts)
        a = a[np.unique(a[:, 0], return_index=True)[1]]
        return a[(a[:, 0] >= start_ms) & (a[:, 0] < end_ms)]

    async def _funding_rest(self, sym: str, start_ms: int, end_ms: int) -> np.ndarray | None:
        import orjson

        out, t = [], start_ms
        while t < end_ms:
            raw = await self._get(f"{FAPI}/fapi/v1/fundingRate",
                                  {"symbol": sym, "startTime": t, "endTime": end_ms, "limit": 1000})
            rows = orjson.loads(raw) if raw else []
            if not isinstance(rows, list) or not rows:
                break
            out += [[r["fundingTime"], float(r["fundingRate"])] for r in rows]
            if len(rows) < 1000:
                break
            t = rows[-1]["fundingTime"] + 1
        return np.asarray(out, dtype=float).reshape(-1, 2) if out else None
