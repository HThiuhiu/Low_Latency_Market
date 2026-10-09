"""Signal engine integration: features -> ONNX predictor -> Forecast, hot reload, parity."""

import os
import time
from dataclasses import replace

import numpy as np
import pytest

from qforecast.config import Settings
from qforecast.core.metrics import Registry
from qforecast.schemas import Kline
from tests.conftest import synthetic_bars, write_random_model

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")


def _klines(s: Settings, n: int):
    bars = synthetic_bars(n, seed=3)
    now = int(time.time() * 1000)
    for i, (o, h, lo, c, v, nt, tb) in enumerate(bars.tolist()):
        # last bars carry a fresh close_time so they count as live
        close_time = now if i >= n - 5 else now - 10 * s.interval_ms
        yield Kline(s.symbol, i * s.interval_ms, close_time, o, h, lo, c, v, int(nt), tb, 0,
                    time.perf_counter_ns())


@pytest.fixture
def settings(tmp_path):
    return replace(Settings(), artifacts_dir=tmp_path, lookback=128, history_bars=400)


def test_forecasts_only_live_bars_after_warmup(settings):
    from qforecast.services.signal import SignalEngine

    write_random_model(settings.artifacts_dir, settings.lookback, list(settings.horizons))
    eng = SignalEngine(settings, Registry("signal"))
    out = [eng.on_kline(k) for k in _klines(settings, 300)]
    forecasts = [f for f in out if f is not None]
    assert len(forecasts) == 5  # 300 bars: warm-up done, only the 5 fresh ones forecast
    f = forecasts[-1]
    assert f.horizons == list(settings.horizons)
    assert f.signal in ("LONG", "SHORT", "FLAT")
    np.testing.assert_allclose(f.exp_price, f.close * np.exp(f.exp_logret), rtol=1e-6)
    assert f.infer_us > 0 and f.feature_us > 0


def test_duplicates_are_ignored(settings):
    from qforecast.services.signal import SignalEngine

    eng = SignalEngine(settings, Registry("signal"))
    ks = list(_klines(settings, 3))
    for k in ks + ks:
        eng.on_kline(k)
    assert eng.reg.counters["duplicate_bars"] == 3
    assert eng.reg.counters["bars"] == 3


def test_predictor_matches_torch(settings):
    from qforecast.model.predictor import Predictor

    model = write_random_model(settings.artifacts_dir, 128, [1, 5])
    p = Predictor(settings.model_path, settings.meta_path, capacity=200)
    x = np.random.default_rng(0).normal(size=(150, p.window.width)).astype(np.float32)
    for row in x:
        p.push(row)
    ref = model(torch.from_numpy(np.clip(x[-128:], -5, 5)[None])).detach().numpy()[0] * 1e-3
    np.testing.assert_allclose(p.predict(), ref, atol=1e-6)


def test_hot_reload_keeps_window(settings):
    from qforecast.model.predictor import Predictor

    write_random_model(settings.artifacts_dir, 128, [1, 5], seed=0, version="v1")
    p = Predictor(settings.model_path, settings.meta_path, capacity=200)
    for row in np.ones((130, p.window.width)):
        p.push(row)
    assert not p.maybe_reload()
    write_random_model(settings.artifacts_dir, 128, [1, 5], seed=1, version="v2")
    future = time.time() + 5
    os.utime(settings.model_path, (future, future))
    assert p.maybe_reload()
    assert p.meta.version == "v2" and p.ready
    np.testing.assert_allclose(p.window.last(1), 1.0)
