"""Trading agent: model signal -> LLM advice -> risk manager makes the final call.

    ONNX forecast ──► when to ask? ──► LLM (advisor, JSON) ──► RiskManager (hard rules) ──► PaperBroker

The LLM never trades directly. It returns a proposal; ``RiskManager.decide`` clamps or
vetoes it with fixed rules (confidence floor, max size, mandatory stop, daily loss
limit, cool-down after a loss, call budget). In ``veto`` mode (default) the LLM can only
confirm or skip the model's direction, or close an open position early.

Transport-agnostic: the live service (services/agent.py) and the historical replay
(agent/replay.py) drive the same ``AgentCore``.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np

from qforecast.sim.broker import PaperBroker

ACTIONS = ("OPEN_LONG", "OPEN_SHORT", "SKIP", "HOLD", "CLOSE")

SYSTEM_PROMPT = """You are the risk-aware trading advisor of a crypto paper-trading system.
A small neural network forecasts log-returns of the next bars. Out-of-sample it is weak:
direction hit rate ~51-53%, edge usually smaller than the 12 bps round-trip cost
(fees + slippage). Most of the time the right answer is SKIP or HOLD.

Decide using the data given (model forecast, price context, microstructure, open position,
recent results). Prefer trades where several independent signals agree and the expected
move clearly exceeds costs. Always set a stop. Never chase after a large move without
confirmation. Be calibrated: confidence 0.5 = coin flip.

Reply with ONE JSON object only:
{"action": "OPEN_LONG" | "OPEN_SHORT" | "SKIP" | "HOLD" | "CLOSE",
 "confidence": 0.0-1.0,
 "size": 0.0-1.0,            // fraction of equity, used only for OPEN_*
 "horizon_bars": integer,    // planned holding time, OPEN_* only
 "stop_bps": number,         // stop-loss distance in basis points, OPEN_* only
 "take_bps": number,         // take-profit distance in bps (0 = none), OPEN_* only
 "reason": "max 30 words"}
SKIP = stay flat. HOLD = keep the open position. CLOSE = exit the open position now."""


@dataclass
class AgentConfig:
    mode: str = "veto"  # veto: LLM may only confirm/skip the model's direction; free: any direction
    min_conf: float = 0.6
    min_close_conf: float = 0.55
    max_size: float = 1.0
    min_stop_bps: float = 15.0
    max_stop_bps: float = 150.0
    default_stop_bps: float = 50.0
    max_take_bps: float = 400.0
    max_horizon: int = 240
    daily_loss_pct: float = 3.0  # halt new entries for the rest of the UTC day
    cooldown_bars: int = 5  # no new entry for N bars after a losing trade
    review_bars: int = 15  # re-ask about an open position every N bars
    entry_gap_bars: int = 10  # after asking, wait N bars before asking about a new entry again
    max_calls_hour: int = 30
    ask_on_flat: bool = False  # also ask when the model says FLAT (only in free mode)


@dataclass
class Decision:
    action: str  # what is executed: OPEN_LONG / OPEN_SHORT / CLOSE / NONE
    size: float = 0.0
    horizon: int = 0
    stop_bps: float = 0.0
    take_bps: float = 0.0
    advice: dict = field(default_factory=dict)
    vetoes: list[str] = field(default_factory=list)


class RiskManager:
    """Final say. Pure function of (advice, model signal, account state)."""

    def __init__(self, cfg: AgentConfig):
        self.cfg = cfg

    def decide(self, advice: dict | None, model_signal: str, model_horizon: int,
               has_pos: int, blocked: list[str]) -> Decision:
        """has_pos: +1 long, -1 short, 0 flat. blocked: reasons new entries are not allowed."""
        cfg = self.cfg
        if advice is None:
            return Decision("NONE", vetoes=["no valid advice -> stay as is"])
        a = normalise_advice(advice)
        d = Decision("NONE", advice=a)
        act, conf = a["action"], a["confidence"]

        if act == "CLOSE":
            if not has_pos:
                d.vetoes.append("CLOSE with no position")
            elif conf < cfg.min_close_conf:
                d.vetoes.append(f"close confidence {conf:.2f} < {cfg.min_close_conf}")
            else:
                d.action = "CLOSE"
            return d
        if act not in ("OPEN_LONG", "OPEN_SHORT"):
            return d  # SKIP / HOLD

        side = 1 if act == "OPEN_LONG" else -1
        model_side = {"LONG": 1, "SHORT": -1}.get(model_signal, 0)
        if cfg.mode == "veto" and side != model_side:
            d.vetoes.append(f"{act} disagrees with model signal {model_signal} (veto mode)")
        if conf < cfg.min_conf:
            d.vetoes.append(f"confidence {conf:.2f} < {cfg.min_conf}")
        if has_pos == side:
            d.vetoes.append("already in this direction")
        d.vetoes += blocked
        size = min(a["size"], cfg.max_size)
        if size <= 0:
            d.vetoes.append("size 0")
        if d.vetoes:
            return d
        stop = a["stop_bps"] or cfg.default_stop_bps
        d.action = act
        d.size = size
        d.stop_bps = float(np.clip(stop, cfg.min_stop_bps, cfg.max_stop_bps))
        d.take_bps = float(np.clip(a["take_bps"], 0.0, cfg.max_take_bps))
        d.horizon = int(np.clip(a["horizon_bars"] or model_horizon, 1, cfg.max_horizon))
        return d


def normalise_advice(raw: dict) -> dict:
    def num(k, default=0.0):
        try:
            v = float(raw.get(k, default))
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    act = str(raw.get("action", "SKIP")).upper().strip()
    act = {"LONG": "OPEN_LONG", "BUY": "OPEN_LONG", "SHORT": "OPEN_SHORT", "SELL": "OPEN_SHORT",
           "FLAT": "SKIP", "EXIT": "CLOSE"}.get(act, act)
    if act not in ACTIONS:
        act = "SKIP"
    return {"action": act, "confidence": float(np.clip(num("confidence"), 0, 1)),
            "size": float(np.clip(num("size"), 0, 1)), "horizon_bars": int(max(num("horizon_bars"), 0)),
            "stop_bps": max(num("stop_bps"), 0.0), "take_bps": max(num("take_bps"), 0.0),
            "reason": str(raw.get("reason", ""))[:300]}


class AgentCore:
    """Agent state shared by the live service and the replay."""

    def __init__(self, broker: PaperBroker, cfg: AgentConfig, interval_ms: int):
        self.broker = broker
        self.cfg = cfg
        self.risk = RiskManager(cfg)
        self.interval_ms = interval_ms
        self.closes: deque[float] = deque(maxlen=1500)
        self.highs: deque[float] = deque(maxlen=1500)
        self.lows: deque[float] = deque(maxlen=1500)
        self.call_times: deque[int] = deque()
        self.bar = 0
        self.last_review_bar = -10**9
        self.cooldown_until = -1
        self.day = -1
        self.day_start_equity = broker.equity()
        self.halted_day = -1
        self.n_trades_seen = 0
        self.decisions: deque[dict] = deque(maxlen=50)

    # ---- inputs ------------------------------------------------------------------
    def on_bar(self, t: int, o: float, h: float, lo: float, c: float, trade: bool = True) -> None:
        """trade=False: only warm the price context (replayed history)."""
        self.closes.append(c)
        self.highs.append(h)
        self.lows.append(lo)
        if not trade:
            return
        self.broker.on_bar(t, o, h, lo, c)
        self.bar += 1
        day = t // 86_400_000
        if day != self.day:
            self.day, self.day_start_equity = day, self.broker.equity()
        for tr in self.broker.trades[self.n_trades_seen:]:
            if tr.pnl < 0:
                self.cooldown_until = self.bar + self.cfg.cooldown_bars
        self.n_trades_seen = len(self.broker.trades)

    # ---- when to ask -------------------------------------------------------------
    def should_ask(self, signal: str, now_ms: int) -> str | None:
        """Reason to call the LLM now, or None. Applies the hourly call budget."""
        pos = self.broker.pos
        reason = None
        if pos is None:
            if signal in ("LONG", "SHORT") and self.bar - self.last_review_bar >= self.cfg.entry_gap_bars:
                reason = "entry"
            elif (self.cfg.mode == "free" and self.cfg.ask_on_flat
                  and self.bar - self.last_review_bar >= self.cfg.review_bars):
                reason = "scan"
        elif signal in ("LONG", "SHORT") and {"LONG": 1, "SHORT": -1}[signal] != pos.side:
            reason = "model flipped"
        elif self.bar - self.last_review_bar >= self.cfg.review_bars:
            reason = "review"
        if reason is None:
            return None
        while self.call_times and now_ms - self.call_times[0] > 3_600_000:
            self.call_times.popleft()
        if len(self.call_times) >= self.cfg.max_calls_hour:
            return None
        if reason == "entry" and self.blocked():
            return None  # do not pay for advice we could not act on
        return reason

    def mark_asked(self, now_ms: int) -> None:
        self.call_times.append(now_ms)
        self.last_review_bar = self.bar

    def blocked(self) -> list[str]:
        out = []
        if self.bar < self.cooldown_until:
            out.append(f"cool-down after loss ({self.cooldown_until - self.bar} bars left)")
        if self.halted_day == self.day:
            out.append("daily loss limit hit")
        elif self.day_start_equity > 0 and (
                self.broker.equity() / self.day_start_equity - 1) * 100 <= -self.cfg.daily_loss_pct:
            self.halted_day = self.day
            out.append("daily loss limit hit")
        return out

    # ---- prompt ------------------------------------------------------------------
    def context(self, symbol: str, t: int, price: float, forecast: dict,
                analysis: dict | None = None) -> dict:
        c = np.asarray(self.closes)
        bar_min = self.interval_ms / 60_000

        def ret(minutes):
            k = int(round(minutes / bar_min))
            return round(float(np.log(price / c[-1 - k]) * 1e4), 1) if len(c) > k else None

        def vol(minutes):
            k = int(round(minutes / bar_min))
            if len(c) <= k:
                return None
            r = np.diff(np.log(c[-k - 1:]))
            return round(float(r.std() * 1e4), 2)

        day_bars = int(1440 / bar_min)
        hi = max(list(self.highs)[-day_bars:]) if self.highs else price
        lo = min(list(self.lows)[-day_bars:]) if self.lows else price
        b, p = self.broker, self.broker.pos
        ctx = {
            "symbol": symbol, "bar_minutes": bar_min, "price": price,
            "model_forecast": forecast,
            "price_context": {
                "return_bps": {"15m": ret(15), "1h": ret(60), "4h": ret(240), "24h": ret(1440)},
                "bar_volatility_bps": {"1h": vol(60), "24h": vol(1440)},
                "pos_in_24h_range": round((price - lo) / (hi - lo), 2) if hi > lo else 0.5,
                "high_24h": hi, "low_24h": lo},
            "position": None if p is None else {
                "side": "LONG" if p.side > 0 else "SHORT",
                "entry": round(p.entry_price, 2),
                "unrealized_bps": round(p.side * (price / p.entry_price - 1) * 1e4, 1),
                "bars_held": p.held, "planned_bars": p.exit_after, "stop_bps": p.stop_bps,
                "take_bps": p.take_bps},
            "account": {"equity": round(b.equity(price), 2), "start": b.cfg.cash,
                        "today_pnl_pct": round((b.equity(price) / self.day_start_equity - 1) * 100, 2)
                        if self.day_start_equity else 0.0},
            "recent_trades": [{"side": tr.side, "ret_bps": round(tr.ret_bps, 1), "bars": tr.bars,
                               "exit": tr.reason} for tr in b.trades[-5:]],
            "costs_bps_round_trip": 2 * (b.cfg.fee_bps + b.cfg.slippage_bps),
        }
        if analysis:
            ctx["microstructure"] = analysis
        return ctx

    @staticmethod
    def prompt(ctx: dict) -> str:
        return "Market snapshot (JSON):\n" + json.dumps(ctx, separators=(",", ":"), default=float)

    # ---- act ---------------------------------------------------------------------
    def decide(self, advice: dict | None, model_signal: str, model_horizon: int) -> Decision:
        pos = self.broker.pos
        return self.risk.decide(advice, model_signal, model_horizon,
                                0 if pos is None else pos.side, self.blocked())

    def execute(self, d: Decision, t: int, price: float | None) -> None:
        """price None -> next bar open (replay); else fill now (live)."""
        b = self.broker
        if d.action == "CLOSE":
            b.close(t, price if price is not None else b.last_price, "agent")
        elif d.action in ("OPEN_LONG", "OPEN_SHORT"):
            b.on_signal(t, "LONG" if d.action == "OPEN_LONG" else "SHORT", d.horizon,
                        0.0, price=price, size=d.size, stop_bps=d.stop_bps, take_bps=d.take_bps)

    def record(self, t: int, reason: str, d: Decision, latency_s: float, error: str = "") -> dict:
        rec = {"t": t, "ask": reason, "advice": d.advice, "action": d.action, "vetoes": d.vetoes,
               "size": d.size, "stop_bps": d.stop_bps, "take_bps": d.take_bps, "horizon": d.horizon,
               "latency_s": round(latency_s, 2), "error": error}
        self.decisions.append(rec)
        return rec

    def state(self) -> dict:
        st = self.broker.state()
        st["decisions"] = list(self.decisions)[-20:]
        st["blocked"] = self.blocked()
        st["config"] = asdict(self.cfg)
        return st
