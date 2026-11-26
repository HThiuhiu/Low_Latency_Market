# ruff: noqa: E501
"""Small local dashboard for the C5 paper books (standard-library HTTP server) + the daily scheduler.

    python -m qforecast.swing.dashboard                 # http://127.0.0.1:8765 (this computer only)
    python -m qforecast.swing.dashboard --no-schedule   # view only, never trades

GET  /api/books                       both paper books: equity, P&L, drawdown, days
GET  /api/stats?book=c5|c5_adaptive   account report of one book (daily P&L, positions, picks, trades, coins)
GET  /api/job?book=...                progress of a running step / plan
GET  /api/backtest/job                progress / result of the last backtest
POST /api/run   {"book": ..., "kind": "step" | "plan"}
POST /api/backtest {"start": "2025-01-01", "end": null, "capital": 1000, "gross": 1, "exchange_minimum": false}
Schedule: ~00:05 UTC every enabled book runs its daily step (missed days are replayed); every hour at :05 an equity mark.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import traceback
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from qforecast.swing.backtest import clean
from qforecast.swing.paper import BOOKS, Paper

log = logging.getLogger("swing.dashboard")
STATIC = Path(__file__).parent / "static"
PAPERS: dict[str, Paper] = {}  # filled by papers() on first use (importing this module creates no folder)
LABELS = {"c5": "C5 (fixed signal weights)", "c5_adaptive": "C5 + adaptive weights (research)"}
BT: dict = {"running": False}


def papers() -> dict[str, Paper]:
    if not PAPERS:
        PAPERS.update({"c5": Paper(BOOKS["c5"], adaptive=False), "c5_adaptive": Paper(BOOKS["c5_adaptive"], adaptive=True)})
    return PAPERS


def books() -> list[dict]:
    out = []
    for key, p in papers().items():
        row = {"key": key, "label": LABELS[key], "enabled": p.cfg.enabled, "gross": p.cfg.gross, "capital": p.cfg.capital_usdt,
               "started": p.state.get("started"), "last_step": p.state.get("last_step")}
        try:
            s = p.stats()
            row.update(equity=s["equity"], pnl_pct=s["pnl"]["total_pct"], maxdd_pct=s["maxdd_pct"], days=s["days_running"])
        except Exception as e:  # no account yet, or Binance unreachable
            row["error"] = repr(e)[:200]
        out.append(row)
    return out


def run_backtest(params: dict) -> None:
    from qforecast.swing.backtest import exchange_min_notional, run
    from qforecast.swing.paper import context
    from qforecast.swing.strategy import SwingSpec

    BT.update(running=True, started=datetime.now(UTC).strftime("%H:%M:%S"), result=None, error=None)
    try:
        mn = exchange_min_notional() if params.get("exchange_minimum") else None
        BT["result"] = run(context(), SwingSpec.load(papers()["c5"].cfg.spec), float(params.get("capital", 1000)), float(params.get("gross", 1)),
                           params.get("start") or None, params.get("end") or None, min_notional=mn)
    except Exception as e:
        BT["error"] = repr(e)[:500]
        log.exception("backtest failed")
    finally:
        BT["running"] = False


class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quiet
        return

    def _send(self, code: int, body, ctype: str = "application/json; charset=utf-8"):
        data = body if isinstance(body, bytes) else json.dumps(clean(body), ensure_ascii=False, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _book(self, d: dict) -> Paper | None:
        v = d.get("book", "c5")
        return papers().get(v[0] if isinstance(v, list) else v)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/books":
                return self._send(200, books())
            p = self._book(q)
            if p is None:
                return self._send(404, {"error": "unknown book"})
            if u.path == "/api/stats":
                return self._send(200, p.stats())
            if u.path == "/api/job":
                return self._send(200, p.status())
            if u.path == "/api/backtest/job":
                return self._send(200, BT)
            return self._send(404, {"error": "not found"})
        except Exception as e:
            return self._send(500, {"error": repr(e)[:300], "trace": traceback.format_exc()[-800:]})

    def do_POST(self):
        if self.client_address[0] not in ("127.0.0.1", "::1"):
            return self._send(403, {"error": "local only"})
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        u = urlparse(self.path)
        if u.path == "/api/run":
            p = self._book(body)
            if p is None or body.get("kind") not in ("step", "plan"):
                return self._send(400, {"error": "book and kind (step | plan) needed"})
            return self._send(200, p.start(body["kind"]))
        if u.path == "/api/backtest":
            if BT.get("running"):
                return self._send(409, {"error": "a backtest is running"})
            threading.Thread(target=run_backtest, args=(body,), daemon=True).start()
            return self._send(200, {"started": True})
        return self._send(404, {"error": "not found"})


def scheduler(offset_s: int = 300) -> None:
    """Every hour at :05 UTC: an equity mark of each book; at 00:05 UTC the daily step (missed days are replayed)."""
    done: set[str] = set()
    while True:
        now = datetime.now(UTC)
        slot = now.strftime("%Y-%m-%d %H")
        if now.minute * 60 + now.second >= offset_s and slot not in done:
            done.add(slot)
            for key, p in papers().items():
                if not p.cfg.enabled:
                    continue
                try:
                    if now.hour == 0 or not p.state.get("last_row_ms") or p.state["last_row_ms"] < (int(time.time() * 1000) // 86_400_000 - 1) * 86_400_000:
                        p.step()  # daily step (also catches up after the machine was off)
                    else:
                        p.mark()
                except Exception:
                    log.exception("scheduled run of %s failed", key)
        time.sleep(20)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-schedule", action="store_true")
    a = ap.parse_args()
    if not a.no_schedule:
        threading.Thread(target=scheduler, daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    log.info("dashboard on http://127.0.0.1:%d (%s)", a.port, "view only" if a.no_schedule else "scheduler on")
    srv.serve_forever()


if __name__ == "__main__":
    main()
