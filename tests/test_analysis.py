import math

from qforecast.core.analysis import MarketAnalyzer
from qforecast.schemas import BookTicker, Kline, Trade


def test_microstructure_metrics():
    an = MarketAnalyzer("ETHUSDT")
    assert an.snapshot() is None
    an.on_book(BookTicker("ETHUSDT", 1, 100.0, 3.0, 100.1, 1.0))
    an.on_trade(Trade("ETHUSDT", 100.0, 3.0, False, 1_000_000))  # aggressive buy
    an.on_trade(Trade("ETHUSDT", 100.0, 1.0, True, 1_000_500))  # aggressive sell
    an.on_trade(Trade("ETHUSDT", 100.0, 5.0, False, 1_000_000 - 120_000))  # outside 1m window
    s = an.snapshot()
    assert math.isclose(s.mid, 100.05)
    assert math.isclose(s.obi, 0.5)
    assert math.isclose(s.microprice, (100.1 * 3 + 100.0 * 1) / 4)
    assert math.isclose(s.flow_imbalance_1m, (3 - 1) / 4)
    assert math.isclose(s.flow_imbalance_5m, (8 - 1) / 9)
    assert math.isclose(s.spread_bps, 0.1 / 100.05 * 1e4)


def test_trend_and_momentum_regimes():
    an = MarketAnalyzer("ETHUSDT")
    an.on_book(BookTicker("ETHUSDT", 1, 100.0, 1.0, 100.1, 1.0))
    for i in range(80):
        c = 100.0 + i
        an.on_kline(Kline("ETHUSDT", i * 60_000, i * 60_000 + 59_999, c, c, c, c, 1.0, 1, 0.5))
    s = an.snapshot()
    assert s.trend_regime == "TREND_UP" and s.efficiency_ratio == 1.0
    assert s.momentum_state == "OVERBOUGHT"
    assert s.realized_vol_ann > 0
