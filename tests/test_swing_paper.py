# ruff: noqa: E501
"""Paper bot and dashboard, fully offline: a synthetic cube and a virtual account whose market data come from it."""

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

import numpy as np
import pytest

from qforecast.swing import paper as P
from qforecast.swing.account import SymbolRules, VirtualAccount
from qforecast.swing.strategy import Context
from tests.synth import cut, make_cube

CUBE = make_cube(T=320, seed=7)
STATE = {"k": 250}


def _ctx():
    """The cube as known on the morning after row k: row k + 1 has its 00:00 open but no close yet."""
    c = cut(CUBE, STATE["k"] + 1)
    c.c = c.c.copy()
    c.c[-1, c.bench] = np.nan
    return Context.build(c)


class FakeAccount(VirtualAccount):
    def rules(self):
        return {s: SymbolRules(step=1e-6, market_step=1e-6, tick=1e-6, min_qty=0, min_notional=5.0) for s in CUBE.syms}

    def book_ticker(self):
        k = STATE["k"]
        return {s: (float(CUBE.c[k, j]) * 0.9999, float(CUBE.c[k, j]) * 1.0001) for j, s in enumerate(CUBE.syms)}

    def pay_funding_events(self, since_ms, until_ms):
        raise RuntimeError("offline")  # the bot falls back to the cube's daily funding


@pytest.fixture
def offline(monkeypatch, tmp_path):
    monkeypatch.setattr(P, "refresh_data", lambda: "offline")
    monkeypatch.setattr(P, "context", _ctx)
    monkeypatch.setattr(P.Paper, "account", lambda self: FakeAccount(self.root / "virtual_account.json", start_equity=self.cfg.capital_usdt, walk_book=False))
    STATE["k"] = 250
    return tmp_path


def test_plan_orders_open_adjust_close():
    pos = {"A": {"qty": 1.0, "mark": 100.0}, "B": {"qty": -2.0, "mark": 50.0}}
    bt = {"A": (99.0, 101.0), "B": (49.0, 51.0), "C": (9.0, 11.0)}
    out = {o["symbol"]: o for o in P.plan_orders({"A": 0.05, "C": -0.02}, pos, 1000.0, bt, 5.5)}
    assert out["A"]["why"] == "reduce" and out["A"]["side"] == -1
    assert out["B"]["why"] == "close" and out["B"]["reduce_only"] and out["B"]["side"] == 1
    assert out["C"]["why"] == "open" and out["C"]["side"] == -1


def test_mark_day_puts_replay_marks_on_the_day_before():
    assert P.mark_day("2026-11-20 00:00:00") == "2026-11-19"
    assert P.mark_day("2026-11-20 00:05:10") == "2026-11-20"
    assert P.mark_day("2026-11-14 00:00:00", started="2026-11-14 00:00:00") == "2026-11-14"


def test_first_step_trades_then_missed_days_are_replayed(offline):
    p = P.Paper(offline / "c5")
    first = p.step()
    assert "error" not in first, first
    assert first["positions"] > 8 and first["long"] > 0 and first["short"] > 0 and first["replayed_days"] == []
    acct = p.account().account()
    assert acct["equity"] == pytest.approx(1000.0, rel=0.01)
    STATE["k"] = 254  # the machine was off for three days
    again = p.step()
    assert again["replayed_days"] and len(again["replayed_days"]) == 4
    assert len(p.tail("equity.jsonl", 100)) == 5 and len(p.tail("steps.jsonl", 100)) == 2
    assert p.step()["skipped"]  # same data: nothing to do
    assert sum(bool(r.get("replay")) for r in p.tail("equity.jsonl", 100)) == 4
    s = p.stats()  # synthetic 2020 dates sit before the wall-clock start day, so all marks land on that day
    assert s["days_running"] >= 1 and s["maxdd_pct"] <= 0 and s["picks"] and s["weights"]


def test_adaptive_book_uses_the_day_weights(offline, monkeypatch):
    rng = np.random.default_rng(0)
    seen = {}

    def fake(ctx, days):
        seen["days"] = days
        return {d: dict(zip(("vol_28d", "beta_90d", "max_7d", "funding_7d", "near_high_90d", "flow_res_7d"), rng.normal(size=6), strict=True)) for d in days}

    monkeypatch.setattr(P, "adaptive_weights", fake)
    p = P.Paper(offline / "c5_adaptive", adaptive=True)
    plan = p.plan()
    assert "error" not in plan, plan
    assert len(seen["days"]) == 7 and plan["adaptive"] and len(plan["weights"]) == 6


def test_dashboard_serves_books_and_stats(offline, monkeypatch):
    from qforecast.swing import dashboard as D

    monkeypatch.setattr(D, "PAPERS", {"c5": P.Paper(offline / "c5"), "c5_adaptive": P.Paper(offline / "c5_adaptive", adaptive=True)})
    D.PAPERS["c5"].step()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), D.H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        books = json.loads(urllib.request.urlopen(base + "/api/books").read())
        assert [b["key"] for b in books] == ["c5", "c5_adaptive"] and books[0]["days"] >= 1
        stats = json.loads(urllib.request.urlopen(base + "/api/stats?book=c5").read())
        assert stats["positions"] and stats["picks"]
        assert b"C5 paper books" in urllib.request.urlopen(base + "/").read()
        req = urllib.request.Request(base + "/api/run", data=b'{"book": "c5", "kind": "nope"}', method="POST", headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req)
    finally:
        srv.shutdown()
