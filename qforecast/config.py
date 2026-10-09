"""Runtime configuration. Every field can be overridden with a QF_<NAME> env var."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

_INTERVAL_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000}


def _cast(raw: str, default):
    if isinstance(default, bool):
        return raw.lower() in ("1", "true", "yes")
    if isinstance(default, tuple):
        return tuple(int(x) for x in raw.split(",") if x)
    if isinstance(default, Path):
        return Path(raw)
    return type(default)(raw) if default is not None else raw


@dataclass(frozen=True)
class Settings:
    symbol: str = "ETHUSDT"
    interval: str = "1m"
    # memory:// runs everything in one process; redis://host:6379/0 for the distributed deployment.
    bus_url: str = "memory://"
    artifacts_dir: Path = Path("artifacts")
    data_dir: Path = Path("data")

    # Model / features
    lookback: int = 128
    horizons: tuple = (1, 5, 15, 30, 60)
    # Bars kept in memory by the signal service (must exceed lookback + feature warm-up).
    history_bars: int = 1000
    # Minimum |expected move| in bps before the signal leaves FLAT (≈ round-trip taker fee).
    signal_threshold_bps: float = 8.0

    # Bus
    stream_maxlen: int = 20_000
    kline_stream_maxlen: int = 5_000

    # Analytics
    analysis_publish_interval_ms: int = 250

    # Recorder (tick capture to data_dir/ticks/<symbol>/*.qtlg)
    record: bool = False  # also run the recorder in single-process mode (QF_RECORD=1)
    record_block_kb: int = 256
    record_flush_ms: int = 1000

    # Pin the process to these logical CPUs, e.g. QF_CPU_AFFINITY=2,3 (empty = OS decides).
    # On hybrid Intel CPUs an E-core runs inference ~2.3x slower than a P-core.
    cpu_affinity: tuple = ()
    # Windows 11: opt out of EcoQoS, which otherwise parks background services on E-cores
    # (see benchmarks/HETERO.md). No effect on other OSes.
    high_qos: bool = True

    # Gateway
    gateway_host: str = "0.0.0.0"
    gateway_port: int = 8000

    # Exchange endpoints
    binance_rest: str = "https://api.binance.com"
    binance_ws: str = "wss://stream.binance.com:9443"

    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> Settings:
        kwargs = {}
        for f in fields(cls):
            raw = os.environ.get(f"QF_{f.name.upper()}")
            if raw is None or f.name == "extra":
                continue
            kwargs[f.name] = _cast(raw, getattr(cls, f.name, None))
        return cls(**kwargs)

    @property
    def interval_ms(self) -> int:
        return _INTERVAL_MS[self.interval]

    @property
    def sym(self) -> str:
        return self.symbol.lower()

    # ---- bus topics ------------------------------------------------------
    @property
    def t_kline(self) -> str:
        return f"md.kline.{self.sym}"

    @property
    def t_book(self) -> str:
        return f"md.book.{self.sym}"

    @property
    def t_trade(self) -> str:
        return f"md.trade.{self.sym}"

    @property
    def t_forecast(self) -> str:
        return f"sig.forecast.{self.sym}"

    @property
    def t_analysis(self) -> str:
        return f"ana.market.{self.sym}"

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / "model.onnx"

    @property
    def meta_path(self) -> Path:
        return self.artifacts_dir / "model_meta.json"
