# ruff: noqa: E501
"""Virtual account: a market order walks the order book for its full size (VWAP), pays the taker fee,
and keeps a ledger of realised P&L. Offline: the order book is a fake snapshot."""

import pytest

from qforecast.swing.account import SymbolRules, VirtualAccount


class FakeBook:
    def __init__(self, asks, bids):
        self.book = {"asks": asks, "bids": bids}

    def get(self, path, params=None):
        assert path == "/fapi/v1/depth"
        book = self.book

        class R:
            def json(self):
                return book
        return R()


def _account(tmp_path, asks, bids):
    acct = VirtualAccount(tmp_path / "acct.json", start_equity=10_000.0, walk_book=True)
    acct.c = FakeBook(asks, bids)
    acct._rules = {"XUSDT": SymbolRules(step=0.001, market_step=0.001, tick=0.01, min_qty=0.0, min_notional=5.0)}
    return acct


def test_market_buy_walks_the_book_at_vwap(tmp_path):
    acct = _account(tmp_path, asks=[["100", "1"], ["101", "1"], ["103", "10"]], bids=[["99", "5"]])
    fill = acct.execute("XUSDT", +1, 3.0, False, (99.0, 100.0))[0]
    assert fill.kind == "taker" and fill.qty == pytest.approx(3.0)
    assert fill.price == pytest.approx((100 + 101 + 103) / 3)  # not the best ask: the full size eats three levels
    assert fill.fee_usdt == pytest.approx(3.0 * fill.price * 0.0005)


def test_round_trip_books_realised_pnl_and_fees(tmp_path):
    acct = _account(tmp_path, asks=[["100", "10"]], bids=[["110", "10"]])
    acct.execute("XUSDT", +1, 2.0, False, (99.0, 100.0))
    acct.execute("XUSDT", -1, 2.0, True, (110.0, 111.0))
    assert "XUSDT" not in acct.positions
    assert acct.ledger["realized"] == pytest.approx(20.0)
    assert acct.ledger["fees"] == pytest.approx(2 * 100 * 0.0005 + 2 * 110 * 0.0005)
    assert acct.ledger["trades"][-1]["side"] == "LONG" and acct.ledger["trades"][-1]["net"] > 0


def test_order_below_exchange_minimum_is_skipped(tmp_path):
    acct = _account(tmp_path, asks=[["100", "10"]], bids=[["99", "10"]])
    fill = acct.execute("XUSDT", +1, 0.01, False, (99.0, 100.0))[0]
    assert fill.kind == "skipped" and not acct.positions
