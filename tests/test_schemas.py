import orjson

from qforecast.exchange.binance import parse_message
from qforecast.schemas import BookTicker, Kline, Trade, decode_md, encode


def test_market_data_roundtrip():
    msgs = [
        Kline("ETHUSDT", 1, 2, 1.0, 2.0, 0.5, 1.5, 10.0, 7, 4.0, 3, 99),
        BookTicker("ETHUSDT", 5, 1.0, 2.0, 1.1, 3.0, 42),
        Trade("ETHUSDT", 1.0, 0.1, True, 10, 11, 12),
    ]
    for m in msgs:
        assert decode_md(encode(m)) == m


def _frame(stream, data):
    return orjson.dumps({"stream": stream, "data": data})


def test_parse_binance_frames():
    k = {"t": 1000, "T": 60999, "s": "ETHUSDT", "o": "1", "c": "2", "h": "3", "l": "0.5",
         "v": "10", "n": 5, "x": True, "V": "6"}
    msg = parse_message(_frame("ethusdt@kline_1m", {"E": 61000, "k": k}), 7)
    assert msg == Kline("ETHUSDT", 1000, 60999, 1.0, 3.0, 0.5, 2.0, 10.0, 5, 6.0, 61000, 7)
    assert parse_message(_frame("ethusdt@kline_1m", {"E": 1, "k": {**k, "x": False}}), 0) is None

    b = parse_message(_frame("ethusdt@bookTicker",
                             {"u": 9, "s": "ETHUSDT", "b": "1.5", "B": "2", "a": "1.6", "A": "3"}), 0)
    assert b == BookTicker("ETHUSDT", 9, 1.5, 2.0, 1.6, 3.0, 0)

    t = parse_message(_frame("ethusdt@aggTrade",
                             {"s": "ETHUSDT", "p": "1.5", "q": "0.2", "m": False, "T": 5, "E": 6}), 1)
    assert t == Trade("ETHUSDT", 1.5, 0.2, False, 5, 6, 1)
