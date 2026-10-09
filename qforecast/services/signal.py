"""Signal service: closed bar -> incremental features -> ONNX forecast -> bus.

Feature computation and inference are co-located on purpose: splitting them into
two services would add a serialization + network hop to the critical path for no
scaling benefit (both are microseconds per bar).
"""

from __future__ import annotations

import asyncio
import logging
import time

import numpy as np

from qforecast.bus import Bus
from qforecast.config import Settings
from qforecast.core.features import N_FEATURES, OnlineFeatureEngine
from qforecast.core.metrics import Registry
from qforecast.core.ringbuffer import RingBuffer
from qforecast.model.predictor import Predictor, decide_signal
from qforecast.schemas import Forecast, Kline, decode_md, encode

log = logging.getLogger("signal")


class SignalEngine:
    """Pure, transport-agnostic core (unit-testable without a bus)."""

    def __init__(self, s: Settings, reg: Registry):
        self.s = s
        self.reg = reg
        self.features = OnlineFeatureEngine(s.history_bars)
        self.features.warmup_jit()
        # Raw feature history so a model that appears later can be warmed instantly.
        self.feat_hist = RingBuffer(s.history_bars, N_FEATURES)
        self.predictor: Predictor | None = None
        self.last_open = -1
        self.h_feat = reg.hist("features", "feature update per bar")
        self.h_infer = reg.hist("inference", "ONNX forward pass")
        self.h_pipe = reg.hist("pipeline", "ingestor receive -> forecast published")
        self.try_load_model()

    def try_load_model(self) -> None:
        if self.predictor is not None:
            if self.predictor.maybe_reload():
                self.reg.inc("model_reloads")
            return
        if not (self.s.model_path.exists() and self.s.meta_path.exists()):
            return
        self.predictor = Predictor(self.s.model_path, self.s.meta_path, self.s.history_bars)
        for row in self.feat_hist.last(len(self.feat_hist)):
            self.predictor.push(row)
        log.info("loaded model %s (lookback=%d, horizons=%s)", self.predictor.meta.version,
                 self.predictor.lookback, self.predictor.meta.horizons)

    def on_kline(self, k: Kline) -> Forecast | None:
        if k.open_time <= self.last_open:
            self.reg.inc("duplicate_bars")
            return None
        if self.last_open >= 0 and k.open_time != self.last_open + self.s.interval_ms:
            self.reg.inc("gaps")
        self.last_open = k.open_time

        t0 = time.perf_counter_ns()
        f = self.features.update(k.open, k.high, k.low, k.close, k.volume, k.trades,
                                 k.taker_buy_volume)
        t1 = time.perf_counter_ns()
        self.reg.inc("bars")
        if f is None:
            return None
        self.h_feat.record(t1 - t0)
        self.feat_hist.push(f)
        p = self.predictor
        if p is None:
            return None
        p.push(f)
        # Only forecast fresh bars: replayed history just warms the state.
        stale = time.time() * 1000 - k.close_time > 2 * self.s.interval_ms
        if not p.ready or stale:
            return None

        t2 = time.perf_counter_ns()
        exp = p.predict()
        t3 = time.perf_counter_ns()
        self.h_infer.record(t3 - t2)
        meta = p.meta
        sig, hz, edge = decide_signal(exp, meta.sigma, meta.horizons, self.s.signal_threshold_bps)
        pipeline_us = (t3 - k.recv_ns) / 1e3 if k.recv_ns else 0.0
        return Forecast(
            symbol=k.symbol, bar_time=k.open_time, close=k.close, horizons=list(meta.horizons),
            exp_logret=[float(x) for x in exp],
            exp_price=[float(k.close * np.exp(x)) for x in exp],
            sigma=[float(x) for x in meta.sigma], signal=sig, signal_horizon=hz,
            edge_bps=edge, model_version=meta.version,
            feature_us=(t1 - t0) / 1e3, infer_us=(t3 - t2) / 1e3, pipeline_us=pipeline_us,
        )


async def run(s: Settings, bus: Bus, reg: Registry) -> None:
    eng = SignalEngine(s, reg)

    async def watch_model() -> None:
        while True:
            await asyncio.sleep(10)
            eng.try_load_model()

    watcher = asyncio.create_task(watch_model())
    try:
        # replay=True: warm the feature state from the retained kline stream.
        async for _, payload in bus.subscribe([s.t_kline], replay=True):
            k = decode_md(payload)
            fc = eng.on_kline(k)
            if fc is not None:
                await bus.publish(s.t_forecast, encode(fc))
                if k.recv_ns:
                    eng.h_pipe.record(time.perf_counter_ns() - k.recv_ns)
                reg.inc("forecasts")
                log.info("bar %s close=%.2f signal=%s h=%d edge=%.1fbps infer=%.0fus",
                         time.strftime("%H:%M", time.gmtime(k.open_time / 1000)), k.close,
                         fc.signal, fc.signal_horizon, fc.edge_bps, fc.infer_us)
    finally:
        watcher.cancel()


def main() -> None:
    from qforecast.services.base import run as run_service

    run_service("signal", run)


if __name__ == "__main__":
    main()
