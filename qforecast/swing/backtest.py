# ruff: noqa: E501
"""Exchange-style backtest of a swing spec on the daily cube: an account in USDT, real order sizes, fees, funding.

    python -m qforecast.swing.backtest --start 2025-01-01 --capital 1000 [--gross 2] [--exchange-minimum]   # C5 by default
    python -m qforecast.swing.backtest --spec <other spec.json> ...                                       # any other spec

Decision code = qforecast.swing.strategy.book_targets (the same function the research tools score). Every day:
targets at the close of day k, trade at open[k+1] on a PaperBook (flat cost per side, minimum order, optional exchange
minimum notional per symbol), positions drift until they are traded again, funding charged every day.
``stagger`` (default): the book is the mean of the 7 weekday sub-books, each re-picked weekly on its own day, so
1 / 7 of the book changes every day and the result does not depend on the start weekday.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from qforecast.swing.paperbook import BookConfig, PaperBook
from qforecast.swing.strategy import Context, SwingSpec, book_targets

CUBE = "data/um/cube_daily_top40.npz"
WARMUP_D = 120  # 90-day signals + slack before the first trade


def _d(ms) -> str:
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d")


def _ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


def clean(o):
    """Replace NaN / inf by None everywhere (JSON for browsers)."""
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.generic):
        return clean(o.item())
    return o


def exchange_min_notional(cache: Path = Path("data/exchange_min_notional.json")) -> dict[str, float]:
    """symbol -> minimum order notional (USDT) from Binance exchangeInfo, cached 7 days."""
    if cache.exists() and time.time() - cache.stat().st_mtime < 7 * 86400:
        return json.loads(cache.read_text())
    import httpx

    info = httpx.get("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20).json()
    out = {}
    for s in info.get("symbols", []):
        if s.get("quoteAsset") == "USDT":
            out[s["symbol"]] = next((float(f.get("notional", 5)) for f in s.get("filters", []) if f.get("filterType") == "MIN_NOTIONAL"), 5.0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def staggered_targets(ctx: Context, spec: SwingSpec, k0: int, end: int) -> np.ndarray:
    """(T, N) daily target = mean of the ``every`` sub-books' latest re-pick (each sub-book re-picks every ``every`` days)."""
    T = ctx.cube.T
    subs = []
    for ph in range(spec.every):
        rb = np.arange(k0 + ph, end, spec.every)
        Wp = book_targets(ctx, spec, rb)
        last = np.full(T, -1)
        last[rb] = rb
        np.maximum.accumulate(last, out=last)
        subs.append(np.where((last >= 0)[:, None], Wp[np.maximum(last, 0)], 0.0))
    return np.mean(subs, axis=0)


def run(ctx: Context, spec: SwingSpec, capital: float = 1000.0, gross: float = 1.0, start: str | None = None,
        end: str | None = None, cost_bps: float = 7.0, min_order_usdt: float = 5.5, min_notional: dict | None = None,
        stagger: bool = True) -> dict:
    t0 = time.time()
    spec = dataclasses.replace(spec, gross=spec.gross * float(gross))
    cube = ctx.cube
    b = cube.bench
    first = int(np.argmax(ctx.mask.any(1))) + WARMUP_D
    k0 = max(first, int(np.searchsorted(cube.t, _ms(start)))) if start else first
    kend = cube.T - 2
    if end:
        kend = min(kend, int(np.searchsorted(cube.t, _ms(end) + 86_400_000)) - 1)
    while kend > k0 and not np.isfinite(cube.o[kend + 1, b]):
        kend -= 1
    if kend - k0 < 14:
        raise ValueError("period too short (need >= 14 days after the warm-up)")
    if stagger:
        W = staggered_targets(ctx, spec, k0, kend)
        rebal = np.arange(k0, kend)
    else:
        rebal = np.arange(k0, kend, spec.every)
        W = book_targets(ctx, spec, rebal)
    pb = PaperBook(BookConfig(cash=float(capital), cost_bps=float(cost_bps), min_order_usd=float(min_order_usdt)))
    syms = cube.syms
    coin_pnl = {s: 0.0 for s in syms}
    coin_fee = {s: 0.0 for s in syms}
    coin_days = {s: 0 for s in syms}
    dropped: dict[str, int] = {}
    is_reb = {int(x) for x in rebal}
    entry: dict[str, float] = {}  # short_stop: close when a short was opened
    blocked: dict[str, int] = {}  # short_stop: coin -> first row it may be shorted again
    stops = 0
    ruined_at = None
    last_px: dict[str, float] = {}
    for k in range(k0, kend):
        px_open = {s: float(cube.o[k + 1, j]) for j, s in enumerate(syms) if np.isfinite(cube.o[k + 1, j])}
        stop_now = set()
        if spec.short_stop > 0:  # squeeze guard on the daily close: buy back a short up >= short_stop since it was opened
            for s_, q in pb.qty.items():
                j = syms.index(s_)
                if q < 0 and s_ in entry and np.isfinite(cube.c[k, j]) and cube.c[k, j] >= entry[s_] * (1 + spec.short_stop):
                    stop_now.add(s_)
                    blocked[s_] = k + spec.every
                    stops += 1
        if k in is_reb or stop_now:
            eq = pb.equity()
            tg = {} if k in is_reb else dict(pb.weights())
            for s_ in stop_now:
                tg.pop(s_, None)
            for j in (np.flatnonzero(W[k]) if k in is_reb else []):
                s = syms[j]
                if W[k, j] < 0 and blocked.get(s, -1) > k:
                    continue
                if min_notional is not None and abs(W[k, j]) * eq < min_notional.get(s, 5.0):
                    dropped[s] = dropped.get(s, 0) + 1
                    continue
                tg[s] = float(W[k, j])
            before = dict(pb.qty)
            for fl in pb.rebalance(int(cube.t[k + 1]), tg, px_open):
                coin_fee[fl.symbol] += fl.cost
            for s_, q in pb.qty.items():
                if q < 0 and before.get(s_, 0.0) >= 0:
                    entry[s_] = float(cube.c[k, syms.index(s_)])
            for s_ in list(entry):
                if pb.qty.get(s_, 0.0) >= 0:
                    entry.pop(s_)
        px_close = {s: float(cube.c[k + 1, j]) for j, s in enumerate(syms) if np.isfinite(cube.c[k + 1, j])}
        for s, q in pb.qty.items():
            p0 = px_open.get(s, last_px.get(s))
            p1 = px_close.get(s, p0)
            if p0 and p1:
                coin_pnl[s] += q * (p1 - p0)
            coin_days[s] += 1
        pb.pay_funding({s: float(cube.funding_paid[k + 1, j]) for j, s in enumerate(syms) if cube.funding_paid[k + 1, j] != 0})
        pb.mark(int(cube.t[k + 1] + 86_400_000), px_close)
        last_px.update(px_close)
        if pb.equity() <= 0:
            ruined_at = _d(cube.t[k + 1])
            break
    s = pb.summary(periods_per_year=365.0)
    eqs = np.array([m[1] for m in pb.marks])
    ts = [m[0] for m in pb.marks]
    btc0 = cube.o[k0 + 1, b]
    months, mret, prev = {}, {}, float(capital)
    for t, e in zip(ts, eqs, strict=True):
        months[datetime.fromtimestamp((t - 1) / 1000, UTC).strftime("%Y-%m")] = float(e)
    for m in months:
        mret[m] = round((months[m] / prev - 1) * 100, 2)
        prev = months[m]
    notes = []
    if spec.short_stop > 0:
        notes.append(f"short squeeze guard (close >= {1 + spec.short_stop:.2f} x entry): {stops} stops")
    if dropped:
        notes.append("below the exchange minimum at this capital: " + ", ".join(f"{k} ({v})" for k, v in sorted(dropped.items(), key=lambda x: -x[1])[:8]))
    coins = sorted([{"symbol": sy, "pnl_usdt": round(coin_pnl[sy] - coin_fee[sy], 4), "gross_pnl": round(coin_pnl[sy], 4), "fees": round(coin_fee[sy], 4), "days_held": coin_days[sy]}
                    for sy in syms if coin_days[sy] or coin_fee[sy]], key=lambda x: -x["pnl_usdt"])
    res = {"params": {"capital": capital, "gross": gross, "start": start, "end": end, "cost_bps": cost_bps, "min_order_usdt": min_order_usdt,
                      "exchange_minimum": min_notional is not None, "stagger": stagger, "spec": spec.to_dict()},
           "period": {"from": _d(cube.t[k0 + 1]), "to": _d(ts[-1] - 1) if ts else None, "days": len(pb.marks)},
           "summary": {"start_equity": capital, "end_equity": round(float(eqs[-1]), 4) if len(eqs) else capital, "return_pct": round(s.get("total_return_pct", 0.0), 2),
                       "sharpe": round(s.get("sharpe", 0.0), 2), "max_drawdown_pct": round(s.get("max_drawdown_pct", 0.0), 2), "fees_usdt": round(pb.fees, 4),
                       "funding_usdt": round(pb.funding, 4), "turnover_x": round(pb.turnover / capital, 1), "fills": len(pb.fills), "avg_gross": round(s.get("avg_gross_exposure", 0.0), 3),
                       "btc_hold_pct": round(float((cube.c[min(kend, k0 + len(pb.marks)), b] / btc0 - 1) * 100), 2),
                       "months_positive": f"{sum(v > 0 for v in mret.values())}/{len(mret)}", "ruined_at": ruined_at, "secs": round(time.time() - t0, 1), "notes": notes},
           "equity": [{"t": _d(ts[i] - 1), "equity": round(float(eqs[i]), 4)} for i in range(len(ts))],
           "monthly_pct": mret, "coins": coins[:80]}
    return clean(res)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default="artifacts/models/c5.json")
    ap.add_argument("--cube", default=CUBE)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--gross", type=float, default=1.0)
    ap.add_argument("--cost-bps", type=float, default=7.0)
    ap.add_argument("--exchange-minimum", action="store_true", help="skip orders below each symbol's exchange minimum")
    ap.add_argument("--no-stagger", action="store_true")
    ap.add_argument("--out", help="write the full result as JSON")
    a = ap.parse_args()
    from qforecast.swing.cube import Cube

    ctx = Context.build(Cube.load(Path(a.cube)))
    res = run(ctx, SwingSpec.load(a.spec), a.capital, a.gross, a.start, a.end, a.cost_bps,
              min_notional=exchange_min_notional() if a.exchange_minimum else None, stagger=not a.no_stagger)
    print(json.dumps({"period": res["period"], "summary": res["summary"]}, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
