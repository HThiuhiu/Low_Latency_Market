# ruff: noqa: E501
"""Virtual Binance USDT-M account for the paper bot: real public mainnet prices, simulated fills, state on disk.
No API key, no real order: every order is filled against the public order book and booked in a JSON file.

Order rules (from exchangeInfo): quantity rounded DOWN to the market LOT_SIZE step, notional >= MIN_NOTIONAL (except
reduce-only closes). Taker fills walk the real order book (VWAP of the full size) and pay 5 bps; replays of missed
days fill at a given historical price plus 2 bps of slippage (5 + 2 = the backtest's 7 bps per side).
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

log = logging.getLogger("swing.account")
LIVE_URL = "https://fapi.binance.com"
UA = {"User-Agent": "qforecast-paper/1.0"}


@dataclass
class SymbolRules:
    step: float
    market_step: float
    tick: float
    min_qty: float
    min_notional: float
    status: str = "TRADING"

    def qty(self, q: float, market: bool = False) -> float:
        s = self.market_step if market else self.step
        return math.floor(abs(q) / s + 1e-9) * s


def parse_rules(info: dict) -> dict[str, SymbolRules]:
    out = {}
    for s in info.get("symbols", []):
        if s.get("quoteAsset") != "USDT" or s.get("contractType") != "PERPETUAL":
            continue
        f = {x["filterType"]: x for x in s.get("filters", [])}
        lot, mlot, pf = f.get("LOT_SIZE", {}), f.get("MARKET_LOT_SIZE", f.get("LOT_SIZE", {})), f.get("PRICE_FILTER", {})
        mn = f.get("MIN_NOTIONAL", {})
        out[s["symbol"]] = SymbolRules(step=float(lot.get("stepSize", 1)), market_step=float(mlot.get("stepSize", lot.get("stepSize", 1))),
                                       tick=float(pf.get("tickSize", 0.0001)), min_qty=float(lot.get("minQty", 0)),
                                       min_notional=float(mn.get("notional", 5)), status=s.get("status", "TRADING"))
    return out


@dataclass
class Fill:
    symbol: str
    side: int
    qty: float
    price: float
    fee_usdt: float
    kind: str  # taker / skipped
    note: str = ""

    def as_dict(self) -> dict:
        return {"symbol": self.symbol, "side": "BUY" if self.side > 0 else "SELL", "qty": self.qty, "price": self.price,
                "notional": round(self.qty * self.price, 4), "fee_usdt": round(self.fee_usdt, 6), "kind": self.kind, "note": self.note}


@dataclass
class VirtualAccount:
    path: Path
    start_equity: float = 1000.0
    cash: float = 0.0
    positions: dict = field(default_factory=dict)  # sym -> {"qty", "entry"}
    c: httpx.Client | None = None
    _rules: dict | None = None
    walk_book: bool = True  # taker fills at the order-book VWAP of the full size (else at the best bid / ask)
    clock_ms: int | None = None  # replay: timestamp used for the ledger instead of the wall clock
    # P&L ledger, as an exchange account statement: realised price P&L, fees, funding, per coin, every closed round trip
    ledger: dict = field(default_factory=lambda: {"realized": 0.0, "fees": 0.0, "funding": 0.0, "coins": {}, "open": {}, "trades": []})

    def __post_init__(self):
        self.c = httpx.Client(base_url=LIVE_URL, timeout=15, headers=UA)
        if self.path.exists():
            d = json.loads(self.path.read_text())
            self.cash, self.positions = d["cash"], d["positions"]
            if d.get("ledger"):
                self.ledger = d["ledger"]
        else:
            self.cash = self.start_equity

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"cash": self.cash, "positions": self.positions, "ledger": self.ledger}, indent=1))

    # ---- market data
    def rules(self):
        if self._rules is None:
            self._rules = parse_rules(self.c.get("/fapi/v1/exchangeInfo").json())
        return self._rules

    def book_ticker(self):
        return {d["symbol"]: (float(d["bidPrice"]), float(d["askPrice"])) for d in self.c.get("/fapi/v1/ticker/bookTicker").json()}

    def account(self, bt: dict | None = None) -> dict:
        bt = bt or self.book_ticker()
        pos, upnl = {}, 0.0
        for s, p in self.positions.items():
            m = sum(bt.get(s, (p["entry"], p["entry"]))) / 2
            u = p["qty"] * (m - p["entry"])
            upnl += u
            pos[s] = {"qty": p["qty"], "entry": p["entry"], "mark": m, "upnl": u}
        return {"equity": self.cash + upnl, "wallet": self.cash, "positions": pos}

    # ---- ledger
    def _coin(self, sym):
        return self.ledger["coins"].setdefault(sym, {"realized": 0.0, "fees": 0.0, "funding": 0.0, "trades": 0, "wins": 0})

    def _book(self, sym, realized=0.0, fee=0.0, funding=0.0):
        L, c = self.ledger, self._coin(sym)
        L["realized"] += realized
        L["fees"] += fee
        L["funding"] += funding
        c["realized"] += realized
        c["fees"] += fee
        c["funding"] += funding
        o = L["open"].get(sym)
        if o is not None:
            o["realized"] += realized
            o["fees"] += fee
            o["funding"] += funding

    def _ms(self):
        return self.clock_ms if self.clock_ms is not None else int(time.time() * 1000)

    def _open(self, sym, sign, px):
        self.ledger["open"][sym] = {"side": "LONG" if sign > 0 else "SHORT", "t": self._ms(), "entry_first": px,
                                    "realized": 0.0, "fees": 0.0, "funding": 0.0, "max_notional": 0.0}

    def _close_trip(self, sym, entry, px):
        o = self.ledger["open"].pop(sym, None)
        if o is None:
            return
        net = o["realized"] - o["fees"] + o["funding"]
        c = self._coin(sym)
        c["trades"] += 1
        c["wins"] += int(net > 0)
        now = self._ms()
        self.ledger["trades"].append({"symbol": sym, "side": o["side"], "open_t": o["t"], "close_t": now, "days": round((now - o["t"]) / 86_400_000, 2),
                                      "entry": entry, "exit": px, "max_notional": round(o["max_notional"], 4), "price_pnl": round(o["realized"], 6),
                                      "fees": round(o["fees"], 6), "funding": round(o["funding"], 6), "net": round(net, 6)})
        del self.ledger["trades"][:-3000]

    # ---- funding
    def pay_funding(self, rates: dict[str, float], bt: dict):
        for s, f in rates.items():
            if s in self.positions and f:
                p = self.positions[s]
                m = sum(bt.get(s, (p["entry"], p["entry"]))) / 2
                self.cash -= p["qty"] * m * f
                self._book(s, funding=-p["qty"] * m * f)

    def pay_funding_events(self, since_ms: int, until_ms: int) -> float:
        """Charge every real funding event in (since_ms, until_ms] on the positions held now, at that event's mark price
        (exact when positions did not change in between, as for the paper book between two daily steps)."""
        total = 0.0
        for s, p in list(self.positions.items()):
            ev = self.c.get("/fapi/v1/fundingRate", params={"symbol": s, "startTime": since_ms + 1, "endTime": until_ms, "limit": 1000}).json()
            for e in ev if isinstance(ev, list) else []:
                m = float(e.get("markPrice") or 0) or p["entry"]
                amt = -p["qty"] * m * float(e["fundingRate"])  # longs pay a positive rate
                self.cash += amt
                self._book(s, funding=amt)
                total += amt
        return total

    # ---- fills
    def _fill(self, sym, side, q, px, fee_rate, kind):
        p = self.positions.get(sym, {"qty": 0.0, "entry": px})
        old = p["qty"]
        new = old + side * q
        realized = 0.0
        entry_before = p["entry"]
        if old == 0 or math.copysign(1, old) == math.copysign(1, side * q):  # increase
            if old == 0:
                self._open(sym, side, px)
            p["entry"] = (abs(old) * p["entry"] + q * px) / max(abs(new), 1e-12)
        else:  # reduce / flip: realise P&L on the closed part
            closed = min(abs(old), q)
            realized = closed * (px - p["entry"]) * math.copysign(1, old)
            self.cash += realized
            if abs(new) > 1e-12 and math.copysign(1, new) != math.copysign(1, old):
                p["entry"] = px
        p["qty"] = new
        fee = q * px * fee_rate
        self.cash -= fee
        if sym not in self.ledger["open"] and abs(old) > 0:  # position from before the ledger existed
            self._open(sym, old, entry_before)
        self._book(sym, realized=realized, fee=fee)
        if abs(new) < 1e-12 or (old != 0 and math.copysign(1, new) != math.copysign(1, old)):
            self._close_trip(sym, entry_before, px)
            if abs(new) > 1e-12:  # flipped: the rest opens a new round trip
                self._open(sym, new, px)
        if sym in self.ledger["open"]:
            self.ledger["open"][sym]["max_notional"] = max(self.ledger["open"][sym]["max_notional"], abs(new) * px)
        if abs(new) < 1e-12:
            self.positions.pop(sym, None)
        else:
            self.positions[sym] = p
        return Fill(sym, side, q, px, fee, kind)

    def fill_at(self, sym, side, qty, px, reduce_only, extra_bps=2.0):
        """Replay fill at a given (historical) price: lot rounding and the exchange minimum as a market order, taker
        fee, plus ``extra_bps`` of spread / slippage against us."""
        r = self.rules().get(sym)
        if r is None:
            return [Fill(sym, side, 0, 0, 0, "skipped", "symbol not listed")]
        q = r.qty(qty, market=True)
        if q <= 0 or px <= 0:
            return []
        if not reduce_only and q * px < r.min_notional:
            return [Fill(sym, side, 0, 0, 0, "skipped", f"below exchange minimum {r.min_notional:g} USDT")]
        return [self._fill(sym, side, q, px * (1 + side * extra_bps / 1e4), 0.0005, "taker")]

    def execute(self, sym, side, qty, reduce_only, bt):
        """Market order now: the full size walks the real order book (VWAP), taker fee 5 bps."""
        r = self.rules().get(sym)
        if r is None or r.status != "TRADING":
            return [Fill(sym, side, 0, 0, 0, "skipped", "symbol not trading")]
        mid = (bt[0] + bt[1]) / 2 if bt[0] and bt[1] else 0.0
        if not reduce_only and r.qty(qty) * mid < r.min_notional:
            return [Fill(sym, side, 0, 0, 0, "skipped", f"below exchange minimum {r.min_notional:g} USDT")]
        q = r.qty(qty, market=True)
        if q <= 0:
            return []
        px = bt[1] if side > 0 else bt[0]
        if self.walk_book:
            try:
                d = self.c.get("/fapi/v1/depth", params={"symbol": sym, "limit": 1000}).json()
                levels = d["asks"] if side > 0 else d["bids"]
                left, cost = q, 0.0
                for p_, q_ in levels:
                    take = min(float(q_), left)
                    cost += take * float(p_)
                    left -= take
                    if left <= 1e-12:
                        break
                if left <= 1e-12:
                    px = cost / q
            except Exception:  # keep the top-of-book price when the book is unavailable
                log.warning("depth unavailable for %s, filled at top of book", sym)
        return [self._fill(sym, side, q, px, 0.0005, "taker")]
