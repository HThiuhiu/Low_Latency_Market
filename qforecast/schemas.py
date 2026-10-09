"""Wire schemas shared by every service.

msgspec Structs + MessagePack: typed, schema-validated and ~5-10x faster to
encode/decode than json + dicts (see benchmarks/bench_serialization.py).
"""

from __future__ import annotations

import msgspec


class Kline(msgspec.Struct, tag="kline", array_like=True, frozen=True):
    symbol: str
    open_time: int  # ms epoch, bar start
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    trades: int
    taker_buy_volume: float
    event_time: int = 0  # exchange event time (ms), 0 for REST backfill
    recv_ns: int = 0  # local monotonic receive timestamp (perf_counter_ns)


class BookTicker(msgspec.Struct, tag="book", array_like=True, frozen=True):
    symbol: str
    update_id: int
    bid: float
    bid_qty: float
    ask: float
    ask_qty: float
    recv_ns: int = 0


class Trade(msgspec.Struct, tag="trade", array_like=True, frozen=True):
    symbol: str
    price: float
    qty: float
    is_buyer_maker: bool  # True => aggressor was the seller
    trade_time: int
    event_time: int = 0
    recv_ns: int = 0


class Forecast(msgspec.Struct, tag="forecast", frozen=True):
    symbol: str
    bar_time: int  # open_time of the bar the forecast was made on
    close: float
    horizons: list[int]
    exp_logret: list[float]
    exp_price: list[float]
    sigma: list[float]  # out-of-sample residual std per horizon (log-ret)
    signal: str  # LONG / SHORT / FLAT
    signal_horizon: int
    edge_bps: float
    model_version: str
    # Latency breakdown (microseconds), measured with perf_counter_ns.
    feature_us: float
    infer_us: float
    pipeline_us: float  # recv by ingestor -> forecast published (same host only)


class MarketAnalysis(msgspec.Struct, tag="analysis", frozen=True):
    symbol: str
    ts: int  # ms epoch
    mid: float
    spread_bps: float
    microprice: float
    obi: float  # top-of-book imbalance in [-1, 1]
    obi_ewma: float
    flow_imbalance_1m: float  # (buy - sell) / (buy + sell) taker volume
    flow_imbalance_5m: float
    trade_rate_1m: float  # trades / second
    vwap_5m: float
    realized_vol_ann: float
    vol_regime: str  # LOW / NORMAL / HIGH
    efficiency_ratio: float
    trend_regime: str  # TREND_UP / TREND_DOWN / RANGE
    rsi: float
    momentum_state: str  # OVERBOUGHT / OVERSOLD / NEUTRAL


# Market data is array-encoded (no field names on the wire: smaller + faster) and
# decoded as a tagged union; derived messages keep named fields for readability.
MarketData = Kline | BookTicker | Trade
Message = MarketData | Forecast | MarketAnalysis

_encoder = msgspec.msgpack.Encoder()
_md_decoder = msgspec.msgpack.Decoder(MarketData)
_forecast_decoder = msgspec.msgpack.Decoder(Forecast)
_analysis_decoder = msgspec.msgpack.Decoder(MarketAnalysis)


def encode(msg: Message) -> bytes:
    return _encoder.encode(msg)


def decode_md(data: bytes) -> MarketData:
    return _md_decoder.decode(data)


def decode_forecast(data: bytes) -> Forecast:
    return _forecast_decoder.decode(data)


def decode_analysis(data: bytes) -> MarketAnalysis:
    return _analysis_decoder.decode(data)
