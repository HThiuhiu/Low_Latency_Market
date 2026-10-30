"""Paper broker + live paper engine: fills, costs, exits, no look-ahead."""

import time

import numpy as np
import pytest

from qforecast.config import Settings
from qforecast.schemas import Forecast, Kline
from qforecast.sim.broker import PaperBroker, SimConfig

NOCOST = dict(fee_bps=0.0, slippage_bps=0.0)


def test_long_held_for_horizon_filled_next_open():
    b = PaperBroker(SimConfig(cash=1000, **NOCOST))
    b.on_bar(0, 100, 100, 100, 100)
    b.on_signal(0, "LONG", 2, 10.0)  # no price -> waits for next open
    assert b.pos is None
    b.on_bar(1, 101, 102, 100, 102)  # filled @101, held=1
    assert b.pos.entry_price == 101
    b.on_bar(2, 102, 104, 102, 103)  # held=2 -> exit @ close 103
    assert b.pos is None
    t = b.trades[0]
    assert (t.reason, t.bars, t.exit_price) == ("horizon", 2, 103)
    assert b.cash == pytest.approx(1000 * 103 / 101)


def test_costs_charged_both_sides():
    b = PaperBroker(SimConfig(cash=1000, fee_bps=10, slippage_bps=0))
    b.on_signal(0, "LONG", 1, 0, price=100)
    b.on_bar(1, 100, 100, 100, 100)  # flat price: only fees lost
    assert b.trades[0].ret_bps == pytest.approx(-20, rel=1e-3)
    assert b.cash == pytest.approx(1000 - 1 - 1, rel=1e-6)


def test_short_reverse_and_extend():
    b = PaperBroker(SimConfig(cash=1000, **NOCOST))
    b.on_signal(0, "SHORT", 3, 0, price=100)
    b.on_bar(1, 100, 100, 95, 95)
    b.on_signal(1, "SHORT", 3, 0, price=95)  # same side: extend, no trade
    assert b.pos.exit_after == 4 and not b.trades
    b.on_signal(1, "LONG", 1, 0, price=90)  # reverse
    assert b.trades[0].reason == "reverse" and b.trades[0].pnl == pytest.approx(1000 * 0.1)
    assert b.pos.side == 1


def test_stop_before_take_and_gap():
    b = PaperBroker(SimConfig(cash=1000, stop_bps=100, take_bps=100, **NOCOST))
    b.on_signal(0, "LONG", 10, 0, price=100)
    b.on_bar(1, 100, 102, 98, 100)  # both hit inside the bar -> stop assumed
    assert b.trades[0].reason == "stop" and b.trades[0].exit_price == pytest.approx(99)
    b.on_signal(1, "LONG", 10, 0, price=100)
    b.on_bar(2, 95, 96, 94, 95)  # gap through the stop -> filled at the open
    assert b.trades[1].exit_price == 95


def test_long_only_ignores_shorts_and_flat():
    b = PaperBroker(SimConfig(allow_short=False, **NOCOST))
    b.on_signal(0, "SHORT", 1, 0, price=100)
    b.on_signal(0, "FLAT", 1, 0, price=100)
    assert b.pos is None


def test_simulate_has_no_lookahead():
    """A forecast made at bar t must not be filled before bar t+1's open."""
    from qforecast.model.predictor import ModelMeta
    from qforecast.sim.backtest import simulate

    n = 20
    close = 100 + np.arange(n, dtype=float)
    bars = np.stack([close - 0.5, close + 1, close - 1, close, close, close, close], axis=1)
    ot = np.arange(n, dtype=np.int64) * 60_000
    meta = ModelMeta("v", [], 1, [1], np.zeros(1), np.ones(1), np.ones(1), np.ones(1, np.float32))
    pred = np.full((1, 1), 0.01, np.float32)  # one LONG forecast at bar 5
    b = simulate(ot, bars, pred, np.array([5]), meta, SimConfig(**NOCOST), 8.0, 0, n)
    t = b.trades[0]
    assert t.entry_time == ot[6] and t.entry_price == bars[6, 0]
    assert t.exit_time == ot[6] and t.exit_price == close[6]


def test_paper_engine_ignores_replayed_history():
    from qforecast.services.paper import PaperEngine

    s = Settings()
    eng = PaperEngine(s)
    now = int(time.time() * 1000)
    old = Kline(s.symbol, now - 3_600_000, now - 3_540_001, 1, 1, 1, 1, 1, 1, 1)
    assert not eng.on_kline(old)
    live = Kline(s.symbol, now - 60_000, now - 1, 100, 101, 99, 100, 1, 1, 1)
    assert eng.on_kline(live)
    fc = Forecast(s.symbol, now - 60_000, 100.0, [1], [0.002], [100.2], [0.001], "LONG", 1, 20.0,
                  "v", 0, 0, 0)
    assert eng.on_forecast(fc) and eng.broker.pos is not None
    assert b'"position"' in eng.snapshot()
