# ruff: noqa: E501
"""Multi-asset paper account for exchange-style backtests: many perps at once, partial rebalancing, funding.

    cash + sum(qty_i * price_i) = equity          (perp-style: shorts are negative qty)

* ``rebalance(t, targets, prices)`` trades every symbol to target_weight * equity / price at the
  given prices (the next bar's OPEN in a replay), skipping orders below ``min_order_usd``;
* cost per fill = notional * cost_bps (fee + slippage), plus, when ``impact_y > 0``, square-root
  market impact Y * sigma_daily * sqrt(notional / daily volume): the same weights cost more as
  capital grows, which is the capacity of a strategy;
* ``pay_funding`` charges qty * price * rate (longs pay positive funding, shorts receive);
* every fill and every equity mark is kept for the report. No real orders are ever sent.
"""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BookConfig:
    cash: float = 10_000.0
    cost_bps: float = 7.0  # per side, all-in (taker fee + spread / slippage)
    min_order_usd: float = 5.0
    impact_y: float = 0.0  # square-root impact coefficient (0 = off; ~0.5-1 in the literature)


@dataclass
class BookFill:
    t: int
    symbol: str
    qty: float  # signed
    price: float
    notional: float
    cost: float
    impact_bps: float = 0.0


@dataclass
class PaperBook:
    cfg: BookConfig = field(default_factory=BookConfig)

    def __post_init__(self):
        self.cash = self.cfg.cash
        self.qty: dict[str, float] = {}
        self.px: dict[str, float] = {}
        self.fills: list[BookFill] = []
        self.marks: list[tuple[int, float, float, float]] = []  # (t, equity, gross exposure, net exposure)
        self.fees = 0.0
        self.funding = 0.0
        self.turnover = 0.0
        self.impact = 0.0

    # ---- valuation ------------------------------------------------------------------
    def mark(self, t: int, prices: dict[str, float]) -> float:
        for s, p in prices.items():
            if p is not None and math.isfinite(p) and p > 0:
                self.px[s] = p
        eq = self.equity()
        g = sum(abs(q) * self.px.get(s, 0.0) for s, q in self.qty.items())
        n = sum(q * self.px.get(s, 0.0) for s, q in self.qty.items())
        self.marks.append((t, eq, g / eq if eq else 0.0, n / eq if eq else 0.0))
        return eq

    def equity(self) -> float:
        return self.cash + sum(q * self.px.get(s, 0.0) for s, q in self.qty.items())

    def weights(self) -> dict[str, float]:
        eq = self.equity()
        return {s: q * self.px[s] / eq for s, q in self.qty.items() if q and s in self.px and eq}

    # ---- trading --------------------------------------------------------------------
    def rebalance(self, t: int, targets: dict[str, float], prices: dict[str, float],
                  liquidity: dict[str, tuple[float, float]] | None = None) -> list[BookFill]:
        """Trade to ``targets`` (symbol -> weight of equity). Symbols held but missing from
        ``targets`` are closed. Fills at ``prices`` (no price -> that symbol is not traded).
        ``liquidity``: symbol -> (daily volatility, daily quote volume USD) for the impact cost."""
        for s, p in prices.items():
            if p is not None and math.isfinite(p) and p > 0:
                self.px[s] = p
        eq = self.equity()
        out = []
        for s in sorted(set(targets) | {k for k, q in self.qty.items() if q}):
            p = prices.get(s)
            if p is None or not math.isfinite(p) or p <= 0:
                continue
            want = targets.get(s, 0.0) * eq / p
            dq = want - self.qty.get(s, 0.0)
            notional = abs(dq) * p
            if notional < self.cfg.min_order_usd and want != 0.0:
                continue
            if notional == 0:
                continue
            imp_bps = 0.0
            if self.cfg.impact_y > 0 and liquidity and s in liquidity:
                vol_d, adv = liquidity[s]
                if adv > 0 and math.isfinite(vol_d):
                    imp_bps = self.cfg.impact_y * vol_d * math.sqrt(notional / adv) * 1e4
            cost = notional * (self.cfg.cost_bps + imp_bps) / 1e4
            self.impact += notional * imp_bps / 1e4
            self.cash -= dq * p + cost
            self.qty[s] = self.qty.get(s, 0.0) + dq
            if abs(self.qty[s]) * p < 1e-9:
                self.qty.pop(s)
            self.fees += cost
            self.turnover += notional
            f = BookFill(t, s, dq, p, notional, cost, imp_bps)
            self.fills.append(f)
            out.append(f)
        return out

    def pay_funding(self, rates: dict[str, float]) -> float:
        paid = 0.0
        for s, r in rates.items():
            q = self.qty.get(s)
            if q and r and s in self.px:
                paid += q * self.px[s] * r
        self.cash -= paid
        self.funding -= paid
        return -paid

    # ---- report ---------------------------------------------------------------------
    def summary(self, periods_per_year: float = 365.0) -> dict:
        import numpy as np

        eq = np.array([m[1] for m in self.marks])
        if len(eq) < 3:
            return {"marks": len(eq)}
        r = np.diff(eq) / eq[:-1]
        dd = eq / np.maximum.accumulate(eq) - 1
        sd = r.std(ddof=1)
        return {"start_equity": self.cfg.cash, "end_equity": float(eq[-1]),
                "total_return_pct": float((eq[-1] / self.cfg.cash - 1) * 100),
                "sharpe": float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else 0.0,
                "max_drawdown_pct": float(dd.min() * 100), "fees": self.fees, "impact": self.impact,
                "funding": self.funding,
                "turnover_x_equity": self.turnover / self.cfg.cash, "fills": len(self.fills),
                "avg_gross_exposure": float(np.mean([m[2] for m in self.marks])),
                "avg_net_exposure": float(np.mean([m[3] for m in self.marks])),
                "cost_bps_per_side": self.cfg.cost_bps}

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "fills.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "symbol", "qty", "price", "notional", "cost"])
            for x in self.fills:
                w.writerow([x.t, x.symbol, f"{x.qty:.8g}", f"{x.price:.8g}", f"{x.notional:.2f}", f"{x.cost:.4f}"])
        with (out_dir / "equity.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "equity", "gross_exposure", "net_exposure"])
            for m in self.marks:
                w.writerow([m[0], f"{m[1]:.4f}", f"{m[2]:.4f}", f"{m[3]:.4f}"])

    def state(self) -> dict:
        return {"cash": self.cash, "equity": self.equity(), "positions": dict(self.qty), "prices": dict(self.px),
                "fees": self.fees, "funding": self.funding, "turnover": self.turnover, "config": asdict(self.cfg)}

    def restore(self, st: dict) -> None:
        self.cash = st["cash"]
        self.qty = {k: float(v) for k, v in st["positions"].items()}
        self.px = {k: float(v) for k, v in st["prices"].items()}
        self.fees, self.funding, self.turnover = st["fees"], st["funding"], st["turnover"]
