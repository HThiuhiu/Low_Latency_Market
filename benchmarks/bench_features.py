"""Per-bar feature latency: legacy pandas recompute vs incremental numba kernel.

The original project recomputed every indicator with pandas over the whole history
each time it needed a prediction. The live engine instead advances an O(window)
numba kernel by one bar.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmarks.common import table, timeit  # noqa: E402
from qforecast.core.features import OnlineFeatureEngine, compute_features  # noqa: E402
from tests.conftest import synthetic_bars  # noqa: E402
from tests.test_features import pandas_reference  # noqa: E402

HISTORY = 500


def legacy_add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Faithful port of ModelTrade/Model_CNN+LSTM/Process_data_add_features.add_features."""
    df = df.copy()
    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["roc_10"] = df["close"].pct_change(10)
    m20, s20 = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    up, lo = m20 + 2 * s20, m20 - 2 * s20
    df["bb_width"] = (up - lo) / m20
    tr = pd.concat([df["high"] - df["low"], (df["high"] - df["close"].shift()).abs(),
                    (df["low"] - df["close"].shift()).abs()], axis=1).max(axis=1)
    df["atr_14_pct"] = tr.rolling(14).mean() / df["close"]
    df["dist_ma_20"] = (df["close"] - m20) / m20
    m50 = df["close"].rolling(50).mean()
    df["dist_ma_50"] = (df["close"] - m50) / m50
    df["rvol"] = df["volume"] / (df["volume"].rolling(20).mean() + 1e-6)
    d = df["close"].diff()
    rs = d.clip(lower=0).ewm(com=13).mean() / (-d.clip(upper=0)).ewm(com=13).mean()
    df["rsi_14"] = (100 - 100 / (1 + rs)) / 100
    e12, e26 = df["close"].ewm(span=12, adjust=False).mean(), df["close"].ewm(span=26, adjust=False).mean()
    macd = e12 - e26
    df["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    m, s = df["close"].rolling(20).mean(), df["close"].rolling(20).std()
    ub = m + 2 * s
    df["dist_to_upper"] = (df["close"] - ub) / ub
    df["pump_signal"] = (df["close"] > ub).astype(float) * df["rvol"]
    return df.dropna().replace([np.inf, -np.inf], 0)


def run() -> str:
    bars = synthetic_bars(HISTORY + 2_000)
    hist = bars[:HISTORY]
    df = pd.DataFrame(hist, columns=["open", "high", "low", "close", "volume", "trades", "taker"])
    compute_features(hist)  # JIT

    eng = OnlineFeatureEngine(capacity=1_000)
    eng.warmup_jit()
    stream = iter(np.tile(bars, (40, 1)))

    rows = [
        (f"Legacy: pandas add_features over {HISTORY} bars",
         timeit(lambda: legacy_add_features(df), n=300, warmup=20)),
        (f"pandas, same 14 features over {HISTORY} bars",
         timeit(lambda: pandas_reference(hist), n=300, warmup=20)),
        (f"numba batch recompute over {HISTORY} bars", timeit(lambda: compute_features(hist), n=2_000)),
        ("numba incremental update (1 bar, live path)",
         timeit(lambda: eng.update(*next(stream)), n=50_000, warmup=5_000)),
    ]
    return table(rows, baseline=rows[0][0])


if __name__ == "__main__":
    print(run())
