"""The numba engine must match an independent pandas implementation, and the
online (per-bar) path must be bit-identical to the offline (training) path."""

import numpy as np
import pandas as pd

from qforecast.core.features import FEATURE_NAMES, WARMUP, OnlineFeatureEngine, compute_features


def pandas_reference(bars: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(bars, columns=["o", "h", "l", "c", "v", "n", "tb"])
    c = df.c
    out = pd.DataFrame(index=df.index)
    lr = np.log(c / c.shift(1))
    out["ret_1"] = lr
    out["ret_5"] = np.log(c / c.shift(5))
    out["ret_15"] = np.log(c / c.shift(15))
    out["rvol_20"] = lr.rolling(20).std()
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std()
    out["bb_z_20"] = (c - ma20) / sd20
    out["dist_ma_50"] = c / c.rolling(50).mean() - 1
    tr = pd.concat([df.h - df.l, (df.h - c.shift()).abs(), (df.l - c.shift()).abs()], axis=1).max(axis=1)
    out["atr_pct_14"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    out["rsi_14"] = (100 * gain / (gain + loss) - 50) / 50
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    out["macd_hist_pct"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    out["rel_volume"] = np.log((df.v + 1e-9) / (df.v.rolling(20).mean() + 1e-9))
    rng = df.h - df.l
    out["range_pct"] = rng / c
    out["body_ratio"] = np.where(rng > 0, (c - df.o) / rng.where(rng > 0, 1), 0.0)
    out["taker_imbalance"] = np.where(df.v > 0, df.tb / df.v.where(df.v > 0, 1) - 0.5, 0.0)
    out["rel_trades"] = np.log((df.n + 1) / (df.n.rolling(20).mean() + 1))
    return out


def test_matches_pandas_reference(bars):
    feats, valid = compute_features(bars)
    ref = pandas_reference(bars)
    assert list(ref.columns) == list(FEATURE_NAMES)
    assert not valid[:WARMUP - 1].any() and valid[WARMUP - 1:].all()
    np.testing.assert_allclose(feats[valid], ref.values[valid], rtol=1e-7, atol=1e-10)


def test_online_equals_offline(bars):
    feats, valid = compute_features(bars)
    eng = OnlineFeatureEngine(capacity=200)  # smaller than history: exercises wrap-around
    eng.warmup_jit()
    for i, row in enumerate(bars):
        f = eng.update(*row)
        assert (f is not None) == valid[i]
        if f is not None:
            assert np.array_equal(f, feats[i])


def test_causal(bars):
    """Changing future bars must not change past features."""
    feats, _ = compute_features(bars)
    tampered = bars.copy()
    tampered[2_000:, :4] *= 1.5
    feats2, _ = compute_features(tampered)
    np.testing.assert_array_equal(feats[:2_000], feats2[:2_000])
