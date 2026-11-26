import pytest

from qforecast.swing.paperbook import BookConfig, PaperBook


def test_rebalance_costs_and_equity():
    pb = PaperBook(BookConfig(cash=10_000, cost_bps=5))
    fills = pb.rebalance(0, {"A": 0.5, "B": -0.5}, {"A": 100.0, "B": 50.0})
    assert len(fills) == 2
    assert pb.qty["A"] == pytest.approx(50) and pb.qty["B"] == pytest.approx(-100)
    assert pb.fees == pytest.approx(10_000 * 5e-4)
    assert pb.equity() == pytest.approx(10_000 - 5)
    pb.mark(1, {"A": 110.0, "B": 45.0})  # long up 10%, short down 10%: +500 +500
    assert pb.equity() == pytest.approx(10_995)


def test_funding_and_close_missing_targets():
    pb = PaperBook(BookConfig(cash=1_000, cost_bps=0))
    pb.rebalance(0, {"A": 1.0, "B": -1.0}, {"A": 10.0, "B": 10.0})
    got = pb.pay_funding({"A": 0.001, "B": 0.001})  # long pays 1.0, short receives 1.0
    assert got == pytest.approx(0.0) and pb.funding == pytest.approx(0.0)
    pb.pay_funding({"A": 0.001})
    assert pb.funding == pytest.approx(-1.0)
    pb.rebalance(1, {"B": -1.0}, {"A": 10.0, "B": 10.0})  # A not in targets -> closed
    assert "A" not in pb.qty


def test_min_order_skips_dust_but_still_closes():
    pb = PaperBook(BookConfig(cash=1_000, cost_bps=0, min_order_usd=5))
    pb.rebalance(0, {"A": 0.5}, {"A": 10.0})
    pb.rebalance(1, {"A": 0.502}, {"A": 10.0})  # 2 USD change: skipped
    assert len(pb.fills) == 1
    pb.rebalance(2, {"A": 0.0}, {"A": 10.0})
    assert "A" not in pb.qty


def test_state_roundtrip():
    pb = PaperBook()
    pb.rebalance(0, {"A": 0.3}, {"A": 20.0})
    st = pb.state()
    pb2 = PaperBook()
    pb2.restore(st)
    assert pb2.equity() == pytest.approx(pb.equity())


def test_square_root_impact_grows_with_size():
    small = PaperBook(BookConfig(cash=1_000, cost_bps=0, impact_y=1.0))
    big = PaperBook(BookConfig(cash=1_000_000, cost_bps=0, impact_y=1.0))
    liq = {"A": (0.05, 10_000_000.0)}  # 5% daily vol, 10M USD daily volume
    small.rebalance(0, {"A": 0.5}, {"A": 1.0}, liq)
    big.rebalance(0, {"A": 0.5}, {"A": 1.0}, liq)
    assert small.fills[0].impact_bps == pytest.approx(0.05 * (500 / 1e7) ** 0.5 * 1e4)
    assert big.fills[0].impact_bps == pytest.approx(0.05 * (5e5 / 1e7) ** 0.5 * 1e4)
    assert big.impact / 1_000_000 > 30 * small.impact / 1_000
