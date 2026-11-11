# ruff: noqa: E501
"""Build the daily point-in-time cube (top-40 USDT perps, re-ranked monthly, delisted coins included).

    python -m qforecast.swing.data --out data/um/cube_daily_top40.npz          # the first run downloads the history

Sources: fapi REST daily klines (live symbols, 1000 days per call) and data.binance.vision monthly 1d archives
(delisted symbols), so the universe keeps the coins that died later. Funding: monthly archives + REST.
Row k = day [t_k, t_k + 1d); see qforecast.swing.cube for the timing convention.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qforecast.exchange.vision import DAY, FAPI, PerpStore, _month_starts
from qforecast.swing.cube import BENCH, Cube, align, select_universe

log = logging.getLogger("swing.data")
COLS = ("open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote")


class IncompleteData(RuntimeError):
    """A symbol that is trading now has no bar for the last full day: REST failed, the cube must not be built."""


async def daily_klines(store: PerpStore, sym: str, start_ms: int, end_ms: int, max_age_s: float = 20 * 3600,
                       listed_since: int | None = None) -> np.ndarray:
    """(n, 11) daily klines [start_ms, end_ms), cached in <root>/d1k/<sym>.npy.

    ``listed_since``: onboard time of a symbol whose status is TRADING. When the cube is built up to now, such a
    symbol must have yesterday's bar; otherwise IncompleteData is raised and nothing is cached. A REST call that
    comes back 400 / 403 must not be read as "delisted": the monthly archives have no current month, and the coin
    would silently leave that day's ranking."""
    last_full = (int(time.time() * 1000) // DAY) * DAY  # today's 00:00 UTC: bars before it are final
    need = listed_since is not None and listed_since < last_full - DAY and end_ms >= last_full

    def complete(a):
        return not need or (len(a) > 0 and int(a[:, 0].max()) >= last_full - DAY)

    path = store.root / "d1k" / f"{sym}.npy"
    # reuse only a file written after the last UTC midnight: an older one cannot hold yesterday's bar
    midnight_s = (time.time() // 86_400) * 86_400
    if path.exists() and time.time() - path.stat().st_mtime < max_age_s and path.stat().st_mtime >= midnight_s:
        a = np.load(path)
        if complete(a):
            return a[(a[:, 0] >= start_ms) & (a[:, 0] < end_ms)] if len(a) else a
    for attempt in range(3):  # a listed symbol answering "no data" is a transient REST failure (WAF 403, 400): retry
        a = await _rest_or_archive(store, sym, start_ms, end_ms, last_full)
        if complete(a):
            break
        await asyncio.sleep(5 * (attempt + 1))
    else:
        raise IncompleteData(f"{sym}: Binance REST gave no daily bar for {datetime.fromtimestamp((last_full - DAY) / 1000, UTC):%Y-%m-%d} "
                             "although the symbol is trading: cube not built")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, a)
    return a[(a[:, 0] >= start_ms) & (a[:, 0] < end_ms)] if len(a) else a


async def _rest_or_archive(store: PerpStore, sym: str, start_ms: int, end_ms: int, last_full: int) -> np.ndarray:
    import orjson

    rows, t = [], start_ms
    while t < end_ms:
        raw = await store._get(f"{FAPI}/fapi/v1/klines", {"symbol": sym, "interval": "1d", "startTime": t, "limit": 1000})
        k = orjson.loads(raw) if raw else []
        if not isinstance(k, list) or not k:
            break
        rows += [r[:11] for r in k]
        if len(k) < 1000:
            break
        t = int(k[-1][0]) + DAY
    if rows:
        a = np.asarray(rows, dtype=float)
    else:  # not on REST any more (delisted): monthly archives
        parts = await asyncio.gather(*(store._zip_csv(f"data/futures/um/monthly/klines/{sym}/1d/{sym}-1d-{d:%Y-%m}.zip", 11)
                                       for d in _month_starts(start_ms, end_ms)))
        ks = [p for p in parts if p is not None and len(p)]
        a = np.concatenate(ks) if ks else np.empty((0, 11))
    if len(a):
        a = a[np.unique(a[:, 0], return_index=True)[1]]
        a = a[a[:, 0] < last_full]  # drop today's unfinished bar
    return a


def funding_rows(fr: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(paid, last) on the daily axis ``t``: paid[k] = sum of rates settled in (t_k, t_k + 1d];
    last[k] = the last rate known at the end of row k (forward filled)."""
    T = len(t)
    paid = np.zeros(T)
    last = np.full(T, np.nan)
    if not len(fr):
        return paid, last
    k = np.searchsorted(t, fr[:, 0], side="left") - 1
    ok = (k >= 0) & (k < T)
    np.add.at(paid, k[ok], fr[ok, 1])
    kk = np.searchsorted(t + DAY, fr[:, 0], side="left")
    ok = kk < T
    last[kk[ok]] = fr[ok, 1]
    idx = np.where(np.isfinite(last), np.arange(T), 0)
    np.maximum.accumulate(idx, out=idx)
    fl = last[idx]
    fl[: np.argmax(np.isfinite(last)) if np.isfinite(last).any() else T] = np.nan
    return paid, fl


async def build_daily_cube(root: Path, start: str = "2020-09-01", end_ms: int | None = None, top_n: int = 40,
                           lookback_days: int = 30, min_age_days: int = 60, skip_top: int = 0) -> Cube:
    start_ms = int(datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)
    end_ms = end_ms or (int(time.time() * 1000) // DAY) * DAY
    async with PerpStore(root) as store:
        syms_info = await store.symbols()
        all_syms = sorted({r["symbol"] for r in syms_info} | {BENCH})
        log.info("%d crypto USDT perps known (incl. delisted): daily klines", len(all_syms))
        t0 = time.perf_counter()
        kl: dict[str, np.ndarray] = {}

        async def one(s, info):
            r = info.get(s, {})
            try:
                return await daily_klines(store, s, start_ms, end_ms, listed_since=int(r.get("onboard_ms") or 0) if r.get("status") == "TRADING" else None)
            except IncompleteData:
                # symbols.json is cached for a day: a coin delisted since then answers 400 too. Re-check its status once
                fresh = {x["symbol"]: x for x in await store.symbols(max_age_s=0)}
                info.update(fresh)
                if fresh.get(s, {}).get("status") == "TRADING":
                    raise
                return await daily_klines(store, s, start_ms, end_ms)

        info = {r["symbol"]: r for r in syms_info}
        for i in range(0, len(all_syms), 40):
            chunk = all_syms[i:i + 40]
            for s, a in zip(chunk, await asyncio.gather(*(one(s, info) for s in chunk)), strict=True):
                kl[s] = a
            log.info("klines %d/%d (%.0fs)", min(i + 40, len(all_syms)), len(all_syms), time.perf_counter() - t0)
            await asyncio.sleep(8)  # stay well under the REST weight limit
        d1 = {s: a[:, [0, 7, 4]] for s, a in kl.items() if len(a)}
        uni = select_universe(d1, start_ms + (lookback_days + min_age_days) * DAY, end_ms, top_n, lookback_days, min_age_days, skip_top=skip_top)
        members = sorted({s for v in uni.values() for s in v} | {BENCH})
        syms = [BENCH] + [s for s in members if s != BENCH]
        t = np.arange(start_ms, end_ms, DAY, dtype=np.int64)
        T, N = len(t), len(syms)
        arr = {k: np.full((T, N), np.nan) for k in ("o", "h", "lo", "c", "qv", "tbq")}
        fpaid = np.zeros((T, N))
        flast = np.full((T, N), np.nan)
        log.info("universe: %d months, %d distinct members: funding", len(uni), len(syms))
        fr_all = list(await asyncio.gather(*(store.funding(s, start_ms - 3 * DAY, end_ms) for s in syms)))
        if end_ms >= (int(time.time() * 1000) // DAY) * DAY:  # built up to now: same guard as the klines, for funding
            for j, s in enumerate(syms):  # a perp pays funding at least every 8 h; none in the last day = a failed REST call
                if info.get(s, {}).get("status") != "TRADING" or not len(kl[s]) or kl[s][:, 0].max() < end_ms - DAY:
                    continue
                for attempt in range(3):
                    if len(fr_all[j]) and fr_all[j][:, 0].max() >= end_ms - DAY:
                        break
                    await asyncio.sleep(5 * (attempt + 1))
                    fr_all[j] = await store.funding(s, start_ms - 3 * DAY, end_ms)
                else:
                    raise IncompleteData(f"{s}: no funding event in the last day although the symbol is trading: cube not built")
        for j, (s, fr) in enumerate(zip(syms, fr_all, strict=True)):
            m = align(kl[s], t, [1, 2, 3, 4, 7, 10])
            for col, key in enumerate(("o", "h", "lo", "c", "qv", "tbq")):
                arr[key][:, j] = m[:, col]
            fpaid[:, j], flast[:, j] = funding_rows(fr, t)
    member = np.zeros((T, N), dtype=bool)
    ms = sorted(uni)
    for i, m0 in enumerate(ms):
        m1 = ms[i + 1] if i + 1 < len(ms) else end_ms
        rows = (t >= m0) & (t < m1)
        for s in uni[m0]:
            member[rows, syms.index(s)] = True
    member &= np.isfinite(arr["c"])
    return Cube(t=t, syms=syms, o=arr["o"], h=arr["h"], lo=arr["lo"], c=arr["c"], qv=arr["qv"], tbq=arr["tbq"],
                funding_paid=fpaid, funding_last=flast, member=member,
                meta={"freq": "1d", "start_ms": start_ms, "end_ms": end_ms, "top_n": top_n, "skip_top": skip_top,
                      "universe": {str(k): v for k, v in uni.items()}})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/um")
    ap.add_argument("--start", default="2020-09-01")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--skip-top", type=int, default=0)
    ap.add_argument("--out", default="data/um/cube_daily_top40.npz")
    a = ap.parse_args()
    cube = asyncio.run(build_daily_cube(Path(a.data), a.start, top_n=a.top, skip_top=a.skip_top))
    cube.save(Path(a.out))
    log.info("saved %s: %d symbols, %d days, members/day max %d", a.out, cube.N, cube.T, int(cube.member.sum(1).max()))


if __name__ == "__main__":
    main()
