"""Paper broker: turns model signals into simulated fills, positions and equity.

Pure and transport-agnostic, so the historical backtest and the live paper-trading
service run exactly the same trading logic.

Rules (one position at a time, long or short, perp-style):
* LONG / SHORT while flat  -> open; hold for ``signal_horizon`` bars.
* same-direction signal    -> keep the position, extend the exit to now + horizon.
* opposite signal          -> close, then open the other way.
* FLAT                     -> no action; the open position runs to its exit.
* optional stop-loss / take-profit checked on every bar's high/low
  (if both are hit inside one bar the stop is assumed first: conservative).

Costs: ``fee_bps`` per side on notional, plus ``slippage_bps`` adverse price move.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class SimConfig:
    cash: float = 10_000.0
    fee_bps: float = 5.0  # per side (taker); 10 bps round trip, same as the trainer report
    slippage_bps: float = 1.0
    size: float = 1.0  # notional as a fraction of equity (1.0 = no leverage)
    stop_bps: float = 0.0  # 0 = disabled
    take_bps: float = 0.0  # 0 = disabled
    allow_short: bool = True


@dataclass
class Trade:
    side: str
    entry_time: int
    entry_price: float
    exit_time: int
    exit_price: float
    qty: float
    pnl: float  # net of fees
    fees: float
    ret_bps: float  # net return on entry notional
    bars: int
    reason: str  # horizon / reverse / stop / take / end
    horizon: int
    edge_bps: float


@dataclass
class Position:
    side: int  # +1 long, -1 short
    qty: float
    entry_price: float
    entry_time: int
    entry_fee: float
    horizon: int
    edge_bps: float
    held: int = 0  # bars closed since entry
    exit_after: int = 0  # close when held reaches this
    stop_bps: float = 0.0  # per-position overrides of SimConfig (0 = disabled)
    take_bps: float = 0.0


@dataclass
class PaperBroker:
    cfg: SimConfig = field(default_factory=SimConfig)

    def __post_init__(self):
        self.cash = self.cfg.cash
        self.pos: Position | None = None
        self.pending: tuple[str, int, float] | None = None  # filled at the next bar open
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[int, float]] = []
        self.last_price = 0.0
        self.last_time = 0
        self.fees_paid = 0.0
        self.bars_in_market = 0
        self.bars = 0

    # ---- public API ------------------------------------------------------------
    def equity(self, price: float | None = None) -> float:
        p = self.last_price if price is None else price
        if self.pos is None:
            return self.cash
        return self.cash + self.pos.side * self.pos.qty * (p - self.pos.entry_price)

    def on_signal(self, t: int, signal: str, horizon: int, edge_bps: float,
                  price: float | None = None, size: float | None = None,
                  stop_bps: float | None = None, take_bps: float | None = None) -> None:
        """Act on a model signal. ``price`` given -> fill now (live mode);
        otherwise the order waits for the next bar's open (backtest, no look-ahead).
        ``size`` / ``stop_bps`` / ``take_bps`` override the config for this entry."""
        opts = (size, stop_bps, take_bps)
        if price is None:
            self.pending = (signal, horizon, edge_bps, opts)
        else:
            self._act(t, price, signal, horizon, edge_bps, opts)

    def close(self, t: int, price: float, reason: str = "manual") -> None:
        if self.pos is not None:
            self._close(t, price, reason)

    def on_bar(self, t: int, o: float, h: float, lo: float, c: float) -> None:
        """Feed one closed bar: fill pending order at the open, then stops, then expiry."""
        if self.pending is not None:
            sig, hz, edge, opts = self.pending
            self.pending = None
            self._act(t, o, sig, hz, edge, opts)
        p = self.pos
        if p is not None:
            self._check_stops(t, o, h, lo)
        p = self.pos
        if p is not None:
            p.held += 1
            self.bars_in_market += 1
            if p.held >= p.exit_after:
                self._close(t, c, "horizon")
        self.bars += 1
        self.last_price, self.last_time = c, t
        self.equity_curve.append((t, self.equity(c)))

    def close_all(self, reason: str = "end") -> None:
        if self.pos is not None:
            self._close(self.last_time, self.last_price, reason)
            if self.equity_curve:
                self.equity_curve[-1] = (self.last_time, self.cash)

    # ---- internals -------------------------------------------------------------
    def _act(self, t: int, price: float, signal: str, horizon: int, edge_bps: float,
             opts: tuple = (None, None, None)) -> None:
        side = {"LONG": 1, "SHORT": -1}.get(signal, 0)
        if side == 0 or (side < 0 and not self.cfg.allow_short):
            return
        p = self.pos
        if p is not None and p.side == side:
            p.exit_after = p.held + horizon  # same view again: extend
            return
        if p is not None:
            self._close(t, price, "reverse")
        self._open(t, price, side, horizon, edge_bps, *opts)

    def _open(self, t: int, price: float, side: int, horizon: int, edge_bps: float,
              size: float | None = None, stop_bps: float | None = None,
              take_bps: float | None = None) -> None:
        cfg = self.cfg
        fill = price * (1 + side * cfg.slippage_bps / 1e4)
        eq = self.equity(price)
        if eq <= 0:
            return
        qty = eq * (cfg.size if size is None else size) / fill
        if qty <= 0:
            return
        fee = qty * fill * cfg.fee_bps / 1e4
        self.cash -= fee
        self.fees_paid += fee
        self.pos = Position(side, qty, fill, t, fee, horizon, edge_bps, exit_after=horizon,
                            stop_bps=cfg.stop_bps if stop_bps is None else stop_bps,
                            take_bps=cfg.take_bps if take_bps is None else take_bps)

    def _close(self, t: int, price: float, reason: str) -> None:
        p = self.pos
        fill = price * (1 - p.side * self.cfg.slippage_bps / 1e4)
        gross = p.side * p.qty * (fill - p.entry_price)
        fee = p.qty * fill * self.cfg.fee_bps / 1e4
        self.cash += gross - fee
        self.fees_paid += fee
        pnl = gross - fee - p.entry_fee
        self.trades.append(Trade(
            side="LONG" if p.side > 0 else "SHORT", entry_time=p.entry_time,
            entry_price=p.entry_price, exit_time=t, exit_price=fill, qty=p.qty, pnl=pnl,
            fees=fee + p.entry_fee, ret_bps=pnl / (p.qty * p.entry_price) * 1e4, bars=p.held,
            reason=reason, horizon=p.horizon, edge_bps=p.edge_bps))
        self.pos = None

    def _check_stops(self, t: int, o: float, h: float, lo: float) -> None:
        p = self.pos
        e, s = p.entry_price, p.side
        if p.stop_bps > 0:
            stop = e * (1 - s * p.stop_bps / 1e4)
            if (s > 0 and lo <= stop) or (s < 0 and h >= stop):
                # gap through the stop at the open -> filled at the open
                px = min(o, stop) if s > 0 else max(o, stop)
                self._close(t, px, "stop")
                return
        if p.take_bps > 0:
            take = e * (1 + s * p.take_bps / 1e4)
            if (s > 0 and h >= take) or (s < 0 and lo <= take):
                px = max(o, take) if s > 0 else min(o, take)
                self._close(t, px, "take")

    # ---- reporting -------------------------------------------------------------
    def summary(self, bars_per_year: float, bench_return: float | None = None) -> dict:
        eq = np.array([e for _, e in self.equity_curve]) if self.equity_curve else np.array([self.cash])
        start = self.cfg.cash
        r = np.diff(np.concatenate([[start], eq])) / np.maximum(np.concatenate([[start], eq[:-1]]), 1e-12)
        peak = np.maximum.accumulate(np.concatenate([[start], eq]))
        dd = (np.concatenate([[start], eq]) - peak) / peak
        pnl = np.array([t.pnl for t in self.trades])
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        reasons: dict[str, int] = {}
        for t in self.trades:
            reasons[t.reason] = reasons.get(t.reason, 0) + 1
        out = {
            "start_equity": start,
            "end_equity": float(self.equity()),
            "total_return_pct": float((self.equity() / start - 1) * 100),
            "max_drawdown_pct": float(dd.min() * 100),
            "sharpe_ann": float(r.mean() / r.std() * np.sqrt(bars_per_year)) if r.std() > 0 else 0.0,
            "trades": len(self.trades),
            "win_rate_pct": float(len(wins) / len(pnl) * 100) if len(pnl) else 0.0,
            "avg_trade_bps": float(np.mean([t.ret_bps for t in self.trades])) if self.trades else 0.0,
            "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else None,
            "fees_paid": float(self.fees_paid),
            "exposure_pct": float(self.bars_in_market / max(self.bars, 1) * 100),
            "longs": sum(t.side == "LONG" for t in self.trades),
            "shorts": sum(t.side == "SHORT" for t in self.trades),
            "exit_reasons": reasons,
            "bars": self.bars,
            "config": asdict(self.cfg),
        }
        if bench_return is not None:
            out["buy_hold_return_pct"] = float(bench_return * 100)
        return out

    def state(self) -> dict:
        """Compact live snapshot (for the bus / gateway)."""
        p = self.pos
        return {
            "equity": self.equity(), "cash": self.cash, "last_price": self.last_price,
            "position": None if p is None else {
                "side": "LONG" if p.side > 0 else "SHORT", "qty": p.qty,
                "entry_price": p.entry_price, "entry_time": p.entry_time, "held": p.held,
                "exit_after": p.exit_after, "stop_bps": p.stop_bps, "take_bps": p.take_bps,
                "unrealized": self.equity() - self.cash},
            "trades": len(self.trades),
            "recent_trades": [asdict(t) for t in self.trades[-20:]],
        }
