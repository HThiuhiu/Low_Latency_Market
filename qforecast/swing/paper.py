# ruff: noqa: E501
"""Paper trading of a swing spec (C5 by default): a virtual account on Binance mainnet prices. No real money.

    python -m qforecast.swing.paper --once                 # one daily step now (refreshes the daily cube first)
    python -m qforecast.swing.paper --once --adaptive      # the same with adaptive signal weights (own folder)

Daily step (~00:05 UTC, run by the dashboard scheduler or cron): refresh the cube, charge the funding settled since
the last step, compute the staggered book target (mean of the 7 weekday sub-books, beta hedged) and trade the
virtual account to it with taker fills that walk the real order book (+5 bps fee).
Squeeze guard: a short whose DAILY close is >= (1 + short_stop) x its entry is bought back and kept out of the short
leg for 7 days. If the machine was off, every missed day is replayed at that day's 00:00 open (+7 bps all-in, as the
backtest) with the real funding events in between, so switching the machine off does not change the result.
Files: <root>/{config,state,virtual_account}.json, {equity,orders,steps}.jsonl.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np

log = logging.getLogger("swing.paper")
CUBE = Path("data/um/cube_daily_top40.npz")
BOOKS = {"c5": Path("data/paper/c5"), "c5_adaptive": Path("data/paper/c5_adaptive")}
DAY_MS = 86_400_000
LIVE_WINDOW_MS = 2 * 3_600_000  # a row acted on later than this after 00:00 UTC is replayed at the 00:00 open
WHY = {"open": "open", "close": "close (left the book)", "adjust": "adjust size", "reduce": "reduce size"}
REASONS = {  # signal -> (phrase when it helps the pick's side, phrase when it works against)
    "funding_7d": ("low 7-day funding", "high 7-day funding (crowded longs)"),
    "near_high_90d": ("near its 90-day high", "far from its 90-day high"),
    "flow_res_7d": ("strong taker buying", "strong taker selling"),
    "vol_28d": ("low volatility", "high volatility"),
    "beta_90d": ("low beta", "high beta"),
    "max_7d": ("no recent spike", "recent big up-day"),
}
_DATA_LOCK = threading.Lock()
_CTX: dict = {}


@dataclass
class PaperConfig:
    enabled: bool = True
    spec: str = "artifacts/models/c5.json"
    capital_usdt: float = 1000.0  # start equity of the virtual account (used when it is created or reset)
    gross: float = 1.0
    min_order_usdt: float = 5.5
    adaptive: bool = False  # signal weights re-optimised every day (qforecast.adaptive); each sub-book keeps its day's set

    def validate(self):
        assert Path(self.spec).exists(), f"spec not found: {self.spec}"
        assert 1.0 <= float(self.gross) <= 3.0, "gross 1-3 (virtual account)"
        assert self.capital_usdt >= 100, "virtual capital at least 100 USDT"


def _now():
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")


def _day(ms):
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d")


def _ts(ms):
    return datetime.fromtimestamp(ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")


def mark_day(t: str, started: str | None = None) -> str:
    """UTC day whose result an equity mark shows. A replayed day writes one mark at exactly 00:00:00 (the 00:00 open,
    after that day's orders): it holds the close of the day BEFORE. Never before the paper's start day."""
    d = t[:10]
    if t[11:19] == "00:00:00":
        d = (date.fromisoformat(d) - timedelta(days=1)).isoformat()
    if started and d < str(started)[:10]:
        d = str(started)[:10]
    return d


def context():
    """Context of the current cube, cached until the file changes (shared by both books and the dashboard)."""
    from qforecast.swing.cube import Cube
    from qforecast.swing.strategy import Context

    key = (str(CUBE), CUBE.stat().st_mtime)
    if key not in _CTX:
        _CTX.clear()
        _CTX[key] = Context.build(Cube.load(CUBE))
    return _CTX[key]


def refresh_data() -> str:
    """Rebuild the daily cube when its last full day is older than yesterday (UTC)."""
    from qforecast.swing.cube import Cube
    from qforecast.swing.data import build_daily_cube

    yesterday = (int(time.time() * 1000) // DAY_MS - 1) * DAY_MS
    with _DATA_LOCK:
        if CUBE.exists():
            cu = Cube.load(CUBE)
            ok = np.flatnonzero(np.isfinite(cu.c[:, cu.bench]))  # the last row that really has prices
            if len(ok) and int(cu.t[ok[-1]]) >= yesterday:
                return "daily data up to date"
        c = asyncio.run(build_daily_cube(CUBE.parent))
        c.save(CUBE)
        return f"daily data updated to {_day(c.t[-1])}"


def plan_orders(targets_w: dict, positions: dict, equity: float, bt: dict, min_order_usdt: float) -> list[dict]:
    """Orders that move each position to its target notional (weight x equity)."""
    out = []
    for sym in sorted(set(targets_w) | set(positions)):
        tw = targets_w.get(sym, 0.0)
        p = positions.get(sym)
        px = (sum(bt[sym]) / 2) if sym in bt else (p["mark"] if p else 0.0)
        if px <= 0:
            continue
        cn = p["qty"] * px if p else 0.0
        tn = tw * equity
        if abs(tn) < min_order_usdt and cn != 0:
            tn = 0.0  # too small to hold: close it
        d = tn - cn
        if tn == 0 and cn != 0:
            out.append({"symbol": sym, "side": -1 if cn > 0 else 1, "qty": abs(p["qty"]), "reduce_only": True, "notional": abs(cn), "target_usdt": 0.0, "current_usdt": cn, "why": "close"})
            continue
        if abs(d) < min_order_usdt:
            continue
        reduce = cn != 0 and np.sign(tn) == np.sign(cn) and abs(tn) < abs(cn)
        out.append({"symbol": sym, "side": 1 if d > 0 else -1, "qty": abs(d) / px, "reduce_only": bool(reduce), "notional": abs(d),
                    "target_usdt": round(tn, 3), "current_usdt": round(cn, 3), "why": "reduce" if reduce else ("open" if cn == 0 else "adjust")})
    return out


def adaptive_weights(ctx, days: list[str]) -> dict[str, dict]:
    """{day: weights} for every signal-row day in ``days``; missing days are computed once in a subprocess."""
    from qforecast.adaptive.weights import CACHE, load_cache

    have = load_cache(CACHE)["weights"]
    miss = sorted(d for d in days if d not in have)
    if miss:
        r = subprocess.run([sys.executable, "-m", "qforecast.adaptive.weights", "--from", miss[0], "--to", max(days)],
                           capture_output=True, text=True, timeout=3600)
        if r.returncode:
            raise RuntimeError(f"adaptive weights {miss[0]}..{max(days)} failed: {(r.stderr or r.stdout)[-800:]}")
        have = load_cache(CACHE)["weights"]
    lost = [d for d in days if d not in have]
    if lost:
        raise RuntimeError(f"adaptive weights missing for {lost}")
    return {d: have[d] for d in days}


class Paper:
    def __init__(self, root: Path = BOOKS["c5"], adaptive: bool | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        d = self._read("config.json", {})
        self.cfg = PaperConfig(**{k: v for k, v in d.items() if k in PaperConfig.__dataclass_fields__})
        if adaptive is not None and not d:
            self.cfg = replace(self.cfg, adaptive=adaptive)
            self._write("config.json", asdict(self.cfg))
        self.state = self._read("state.json", {})
        self.last_error = None
        self.job: dict = {"running": False}

    # ---- files
    def _read(self, name, default):
        p = self.root / name
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def _write(self, name, obj):
        (self.root / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding="utf-8")

    def _append(self, name, rows):
        with (self.root / name).open("a", encoding="utf-8") as f:
            for r in rows if isinstance(rows, list) else [rows]:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")

    def tail(self, name, n):
        p = self.root / name
        if not p.exists():
            return []
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()[-n:] if x.strip()]

    def save_cfg(self, new: dict) -> PaperConfig:
        cfg = replace(self.cfg, **{k: v for k, v in new.items() if k in PaperConfig.__dataclass_fields__})
        cfg.gross, cfg.capital_usdt = float(cfg.gross), float(cfg.capital_usdt)
        cfg.validate()
        self.cfg = cfg
        self._write("config.json", asdict(cfg))
        return cfg

    def account(self):
        from qforecast.swing.account import VirtualAccount

        return VirtualAccount(self.root / "virtual_account.json", start_equity=self.cfg.capital_usdt, walk_book=True)

    def spec(self):
        from qforecast.swing.strategy import SwingSpec

        return SwingSpec.load(self.cfg.spec)

    # ---- background job with live progress (the dashboard polls status()["job"])
    def start(self, kind: str, **kw) -> dict:
        if self.job.get("running"):
            return {"started": False, "error": "another job is running"}
        self.job = {"kind": kind, "running": True, "started": _now(), "stage": "start", "result": None}
        fn = self.plan if kind == "plan" else self.step

        def work():
            res = None
            try:
                res = fn(**kw)
            finally:
                self.job.update(result=res, running=False, stage="done", finished=_now())
        threading.Thread(target=work, daemon=True).start()
        return {"started": True}

    # ---- the model's picks at the latest daily close: coins, reasons, weights, planned orders
    def _picks(self, ctx, spec, k, eq, positions, bt, blocked):
        from qforecast.swing.strategy import score, staggered_target_row, weight_vector

        cube = ctx.cube
        S = score(ctx, spec)
        w_today = spec.weights
        if self.cfg.adaptive:  # each sub-book's last re-pick row scored with that day's own weights
            from qforecast.adaptive.weights import dyn_score

            days = [_day(int(cube.t[q])) for q in range(max(0, k - spec.every + 1), k + 1)]
            S, w_today = dyn_score(ctx, spec, k, adaptive_weights(ctx, days))
        w = staggered_target_row(ctx, spec, k, s=S) * float(self.cfg.gross)
        wv = weight_vector(ctx, w_today)
        ok = np.isfinite(S[k])
        order = np.flatnonzero(ok)[np.argsort(-S[k][ok])]
        rank = {int(j): i + 1 for i, j in enumerate(order)}
        targets, picks = {}, []
        for j in np.flatnonzero(w):
            s = cube.syms[j]
            side = "LONG" if w[j] > 0 else "SHORT"
            contrib = {n: float(np.nan_to_num(ctx.Z[k, j, i]) * wv[i]) for i, n in enumerate(ctx.names) if wv[i] != 0}
            top = sorted(contrib.items(), key=lambda x: -x[1] if side == "LONG" else x[1])[:3]
            why = ", ".join(REASONS[n][0 if c > 0 else 1] for n, c in top if n in REASONS)
            skip = side == "SHORT" and s in blocked
            if not skip:
                targets[s] = float(w[j])
            picks.append({"symbol": s, "side": side, "rank": rank.get(int(j)), "of": len(order), "score": round(float(np.nan_to_num(S[k, j])), 3),
                          "weight": round(float(w[j]), 4), "target_usdt": round(float(w[j]) * eq, 2), "price": float(cube.c[k, j]),
                          "why": why, "blocked": skip})
        picks.sort(key=lambda x: (x["side"] != "LONG", -abs(x["weight"])))
        return targets, picks, plan_orders(targets, positions, eq, bt, float(self.cfg.min_order_usdt)), {n: round(float(x), 4) for n, x in zip(ctx.names, wv, strict=True) if x}

    def _load(self, refresh=True):
        self.job["stage"] = "refreshing the daily cube from Binance (first run of the day: 2-5 min)"
        msg = refresh_data() if refresh else "using the cube on disk"
        self.job["stage"] = "scoring"
        ctx = context()
        cube = ctx.cube
        k = cube.T - 1
        while k > 0 and not np.isfinite(cube.c[k, cube.bench]):
            k -= 1
        return ctx, k, msg

    def plan(self, refresh: bool = True) -> dict:
        """What the book would hold now and the orders it would send (nothing is traded)."""
        try:
            ctx, k, msg = self._load(refresh)
            cl = self.account()
            bt = cl.book_ticker()
            acct = cl.account(bt)
            now_ms = int(time.time() * 1000)
            blocked = {s: u for s, u in self.state.get("blocked", {}).items() if u > now_ms}
            targets, picks, orders, wv = self._picks(ctx, self.spec(), k, acct["equity"], acct["positions"], bt, blocked)
            out = {"t": _now(), "data_until": _day(int(ctx.cube.t[k]) + DAY_MS), "data": msg, "equity": round(acct["equity"], 2),
                   "gross": self.cfg.gross, "adaptive": self.cfg.adaptive, "weights": wv, "picks": picks,
                   "orders": [{"symbol": o["symbol"], "side": "BUY" if o["side"] > 0 else "SELL", "plan_usdt": round(o["notional"], 2),
                               "target_usdt": o["target_usdt"], "why": WHY.get(o["why"], o["why"])} for o in orders],
                   "n_long": sum(1 for p in picks if p["side"] == "LONG"), "n_short": sum(1 for p in picks if p["side"] == "SHORT")}
            self._write("plan.json", out)
            return out
        except Exception as e:
            log.exception("plan failed")
            return {"t": _now(), "error": repr(e)[:500]}

    # ---- daily step
    def step(self, force: bool = False) -> dict:
        with self.lock:
            t0 = time.time()
            try:
                rec = self._step(force)
                self.last_error = None
            except Exception as e:  # keep the scheduler alive; the dashboard shows the error
                rec = {"t": _now(), "error": repr(e)[:500], "trace": traceback.format_exc()[-2000:]}
                self.last_error = rec
                log.exception("step failed")
            rec["secs"] = round(time.time() - t0, 1)
            self._append("steps.jsonl", rec)
            return rec

    def _step(self, force: bool) -> dict:  # noqa: C901
        """Process every daily row not processed yet. A row r is acted on at 00:00 UTC of the next day (t[r] + 1 d).
        Rows whose time has passed by more than LIVE_WINDOW (the machine was off) are REPLAYED at that day's 00:00 open;
        the current row inside the window trades live on the order book."""
        ctx, k, msg = self._load()
        cube = ctx.cube
        now_ms = int(time.time() * 1000)
        day_ms = int(cube.t[k])
        st = self.state
        if st.get("last_row_ms") == day_ms and not force:
            return {"t": _now(), "kind": "daily", "skipped": f"already traded on data up to {_day(day_ms)}", "data": msg}
        cl = self.account()
        spec = self.spec()
        if st.get("last_row_ms") and st["last_row_ms"] < day_ms:  # catch up every missed row
            todo = list(range(int(np.searchsorted(cube.t, st["last_row_ms"])) + 1, k + 1))
        else:  # first start, or a forced re-run of the same row
            todo = [k]
        total = {"orders": 0, "fills": 0, "skipped": 0, "funding": 0.0, "stopped": [], "replayed": []}
        fills: list[dict] = []
        targets = st.get("targets", {})
        for r in todo:
            exec_ms = int(cube.t[r]) + DAY_MS
            live = (r == k) and (not st.get("last_row_ms") or force or now_ms - exec_ms <= LIVE_WINDOW_MS)
            if live:
                prices = cl.book_ticker()
                t_ms = now_ms
            else:
                self.job["stage"] = f"replaying {_day(exec_ms)} at the 00:00 UTC open (the machine was off)"
                prices = self._opens(cl, cube, r)
                t_ms = exec_ms
                total["replayed"].append(_day(exec_ms))
            cl.clock_ms = None if live else t_ms
            # funding: every real event since the last charge, on the positions held in between
            if st.get("funding_ms") and cl.positions and t_ms > st["funding_ms"]:
                try:
                    total["funding"] += cl.pay_funding_events(st["funding_ms"], t_ms)
                except Exception:
                    log.exception("funding history unavailable; daily cube sum used")
                    rr = (cube.t > (st["funding_ms"] - DAY_MS)) & (cube.t < t_ms - DAY_MS + 1)
                    cl.pay_funding({s: float(cube.funding_paid[rr, j].sum()) for j, s in enumerate(cube.syms)}, prices)
            st["funding_ms"] = max(st.get("funding_ms") or 0, t_ms)
            acct = cl.account(prices)
            blocked = {s: u for s, u in st.get("blocked", {}).items() if u > t_ms}
            stop = float(spec.short_stop or 0)  # squeeze guard on the DAILY close of row r, as the backtest
            stopped = []
            if stop > 0:
                for s, p in acct["positions"].items():
                    e = st.get("entry", {}).get(s)
                    j = cube.syms.index(s) if s in cube.syms else -1
                    if p["qty"] < 0 and e and j >= 0 and np.isfinite(cube.c[r, j]) and cube.c[r, j] >= e * (1 + stop):
                        blocked[s] = t_ms + spec.every * DAY_MS
                        stopped.append(s)
                        st.get("entry", {}).pop(s, None)
            self.job["stage"] = "picking coins and planning orders" if live else f"replay {_day(exec_ms)}: picking coins"
            targets, picks, orders, wv = self._picks(ctx, spec, r, acct["equity"], acct["positions"], prices, blocked)
            for o in orders:
                if live:
                    got = cl.execute(o["symbol"], o["side"], o["qty"], o["reduce_only"], prices.get(o["symbol"], (0, 0)))
                else:
                    got = cl.fill_at(o["symbol"], o["side"], o["qty"], prices.get(o["symbol"], (0, 0))[0], o["reduce_only"])
                why = "squeeze guard (daily close)" if o["symbol"] in stopped else WHY.get(o["why"], o["why"])
                for f in got:
                    fills.append({**f.as_dict(), "t": _now() if live else _ts(exec_ms), "why": why, "target_usdt": o["target_usdt"],
                                  "current_usdt": o["current_usdt"], "replay": not live})
                done = [f for f in got if f.kind == "taker"]
                total["fills"] += len(done)
                total["skipped"] += 0 if done else 1
            total["orders"] += len(orders)
            total["stopped"] += [f"{s} ({_day(exec_ms)})" for s in stopped]
            acct2 = cl.account(prices)
            entry = dict(st.get("entry", {}))
            for s, p in acct2["positions"].items():
                if p["qty"] < 0 and s not in entry:
                    entry[s] = p["entry"]
            st["entry"] = {s: v for s, v in entry.items() if acct2["positions"].get(s, {}).get("qty", 0) < 0}
            st.update({"last_row_ms": int(cube.t[r]), "blocked": blocked, "targets": {s: round(v, 5) for s, v in targets.items()},
                       "start_equity": st.get("start_equity", self.cfg.capital_usdt), "started": st.get("started", _now()),
                       "picks": picks, "weights": wv, "picked_at": _now() if live else _ts(exec_ms)})
            row = self._mark_row(acct2, prices)
            if not live:
                row["t"] = _ts(exec_ms)
                row["replay"] = True
            self._append("equity.jsonl", row)
        cl.clock_ms = None
        cl.save()
        st["last_step"] = _now()
        self._write("state.json", st)
        self._append("orders.jsonl", fills)
        acct3 = cl.account()
        pos = acct3["positions"].values()
        return {"t": _now(), "kind": "daily", "data_until": _day(day_ms + DAY_MS), "data": msg, "equity": round(acct3["equity"], 4),
                "targets": len(targets), "orders": total["orders"], "fills": total["fills"], "skipped": total["skipped"],
                "positions": len(acct3["positions"]), "long": sum(1 for p in pos if p["qty"] > 0), "short": sum(1 for p in pos if p["qty"] < 0),
                "blocked_shorts": sorted(st.get("blocked", {})), "stopped": total["stopped"],
                "funding_usdt": round(total["funding"], 4), "replayed_days": total["replayed"]}

    def _opens(self, cl, cube, r) -> dict:
        """(price, price) per symbol at 00:00 UTC of day r + 1: the cube's open when that day is in the cube, else
        Binance's daily candle. Covers the members of row r, the open positions and BTC."""
        t1 = int(cube.t[r]) + DAY_MS
        syms = {cube.syms[j] for j in np.flatnonzero(cube.member[r])} | set(cl.positions) | {cube.syms[cube.bench]}
        out = {}
        if r + 1 < cube.T:
            for s in syms:
                if s in cube.syms:
                    v = float(cube.o[r + 1, cube.syms.index(s)])
                    if np.isfinite(v) and v > 0:
                        out[s] = (v, v)
        for s in syms - set(out):
            try:
                kl = cl.c.get("/fapi/v1/klines", params={"symbol": s, "interval": "1d", "startTime": t1, "limit": 1}).json()
                if kl and int(kl[0][0]) == t1:
                    out[s] = (float(kl[0][1]), float(kl[0][1]))
            except Exception:
                log.warning("no 00:00 open for %s on %s", s, _day(t1))
        return out

    def _mark_row(self, acct, bt):
        btc = bt.get("BTCUSDT")
        st = self.state
        if btc and not st.get("btc_start"):
            st["btc_start"] = sum(btc) / 2
            self._write("state.json", st)
        return {"t": _now(), "equity": round(acct["equity"], 4), "positions": len(acct["positions"]),
                "upnl": round(sum(p["upnl"] for p in acct["positions"].values()), 4), "btc": round(sum(btc) / 2, 2) if btc else None}

    def mark(self) -> dict:
        """Hourly equity mark (the squeeze guard runs in the daily step, on the daily close, as the backtest)."""
        with self.lock:
            cl = self.account()
            if not cl.positions:
                return {"t": _now(), "positions": 0}
            bt = cl.book_ticker()
            acct = cl.account(bt)
            self._append("equity.jsonl", self._mark_row(acct, bt))
            return {"t": _now(), "equity": round(acct["equity"], 4)}

    # ---- statistics, as an exchange account report
    def stats(self) -> dict:
        st = self.state
        start = float(st.get("start_equity") or self.cfg.capital_usdt)
        cl = self.account()
        try:
            acct = cl.account()
        except Exception:
            acct = {"equity": start, "positions": {}}
        L = cl.ledger
        rows = [r for r in self.tail("equity.jsonl", 100000) if r.get("equity") is not None]
        upnl = sum(p["upnl"] for p in acct["positions"].values())
        out = {"started": st.get("started"), "start_equity": start, "equity": acct["equity"], "adaptive": self.cfg.adaptive, "gross": self.cfg.gross,
               "pnl": {"total": acct["equity"] - start, "total_pct": (acct["equity"] / start - 1) * 100, "realized_price": L["realized"],
                       "unrealized": upnl, "fees": -L["fees"], "funding": L["funding"]}}
        day_end: dict[str, dict] = {}
        for r in rows:
            day_end[mark_day(r["t"], st.get("started"))] = r
        daily, prev, prev_btc = [], start, st.get("btc_start")
        for d in sorted(day_end):
            e, b = day_end[d]["equity"], day_end[d].get("btc")
            daily.append({"day": d, "equity": e, "pnl": e - prev, "pnl_pct": (e / prev - 1) * 100,
                          "btc_pct": (b / prev_btc - 1) * 100 if b and prev_btc else None})
            prev, prev_btc = e, b or prev_btc
        eq = np.array([start] + [x["equity"] for x in daily], float)
        out["maxdd_pct"] = float((eq / np.maximum.accumulate(eq) - 1).min() * 100) if len(eq) > 1 else 0.0
        r = np.array([x["pnl_pct"] / 100 for x in daily])
        out["days_running"] = len(daily)
        out["sharpe"] = float(r.mean() / r.std(ddof=1) * np.sqrt(365)) if len(r) >= 7 and r.std(ddof=1) > 0 else None
        out["daily"] = daily[::-1]
        tr = L["trades"]
        wins = [t["net"] for t in tr if t["net"] > 0]
        losses = [t["net"] for t in tr if t["net"] <= 0]
        out["trades"] = {"closed": len(tr), "win_pct": len(wins) / len(tr) * 100 if tr else None,
                         "profit_factor": sum(wins) / -sum(losses) if losses and sum(losses) < 0 else None,
                         "long_net": sum(t["net"] for t in tr if t["side"] == "LONG"), "short_net": sum(t["net"] for t in tr if t["side"] == "SHORT"),
                         "list": tr[::-1][:200]}
        coins = []
        for s, c in L["coins"].items():
            p = acct["positions"].get(s)
            u = p["upnl"] if p else 0.0
            coins.append({"symbol": s, "net": c["realized"] + u - c["fees"] + c["funding"], "fees": -c["fees"], "funding": c["funding"],
                          "trades": c["trades"], "now": ("LONG" if p["qty"] > 0 else "SHORT") if p else ""})
        out["coins"] = sorted(coins, key=lambda x: -x["net"])
        out["positions"] = [{"symbol": s, **p, "notional": round(p["qty"] * p["mark"], 2)} for s, p in sorted(acct["positions"].items(), key=lambda x: -abs(x[1]["qty"] * x[1]["mark"]))]
        out["picks"], out["weights"] = st.get("picks", []), st.get("weights", {})
        return out

    # ---- controls
    def reset(self) -> dict:
        """Archive the logs and the virtual account; the next step starts again from capital_usdt."""
        with self.lock:
            arch = self.root / f"archive_{datetime.now(UTC):%Y%m%d_%H%M%S}"
            arch.mkdir()
            for n in ("virtual_account.json", "state.json", "equity.jsonl", "orders.jsonl", "steps.jsonl", "plan.json"):
                if (self.root / n).exists():
                    shutil.move(str(self.root / n), str(arch / n))
            self.state = {}
            return {"archived_to": str(arch)}

    def status(self) -> dict:
        return {"config": asdict(self.cfg), "last_step": self.state.get("last_step"), "last_row": _day(self.state["last_row_ms"]) if self.state.get("last_row_ms") else None,
                "last_error": self.last_error, "job": self.job}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true", help="run one daily step")
    ap.add_argument("--plan", action="store_true", help="show today's picks and orders without trading")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--adaptive", action="store_true", help="the adaptive-weights book (its own folder)")
    a = ap.parse_args()
    p = Paper(BOOKS["c5_adaptive" if a.adaptive else "c5"], adaptive=a.adaptive)
    if a.plan:
        print(json.dumps(p.plan(), indent=1, default=float, ensure_ascii=False))
    if a.once:
        print(json.dumps(p.step(a.force), indent=1, default=float, ensure_ascii=False))


if __name__ == "__main__":
    main()
