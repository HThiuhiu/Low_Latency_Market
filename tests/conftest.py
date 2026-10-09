import numpy as np
import pytest


def synthetic_bars(n: int = 3_000, seed: int = 0) -> np.ndarray:
    """Random-walk OHLCV bars in the features.B_* column layout."""
    rng = np.random.default_rng(seed)
    c = 2_000 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + rng.uniform(0, 1e-3, n))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 1e-3, n))
    v = rng.uniform(10, 100, n)
    v[::97] = 0.0  # exercise zero-volume branches
    trades = rng.integers(100, 1_000, n).astype(float)
    taker = v * rng.uniform(0.3, 0.7, n)
    return np.ascontiguousarray(np.c_[o, h, lo, c, v, trades, taker])


def write_random_model(d, lookback: int, horizons: list[int], seed: int = 0, version: str = "test"):
    """Export an untrained model + metadata (identity normalisation) to directory ``d``."""
    import json
    from pathlib import Path

    import torch

    from qforecast.core.features import FEATURE_NAMES, N_FEATURES
    from qforecast.model.net import CNNBiLSTMAttention

    d = Path(d)
    torch.manual_seed(seed)
    m = CNNBiLSTMAttention(N_FEATURES, len(horizons)).eval()
    torch.onnx.export(m, (torch.zeros(1, lookback, N_FEATURES),), str(d / "model.onnx"),
                      input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    meta = {"version": version, "feature_names": list(FEATURE_NAMES), "lookback": lookback,
            "horizons": horizons, "feat_mean": [0.0] * N_FEATURES, "feat_std": [1.0] * N_FEATURES,
            "y_scale": [1e-3] * len(horizons), "sigma": [1e-3] * len(horizons)}
    (d / "model_meta.json").write_text(json.dumps(meta))
    return m


@pytest.fixture
def bars() -> np.ndarray:
    return synthetic_bars()
