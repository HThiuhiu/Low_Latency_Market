"""Serving-side predictor: ONNX Runtime only (no torch in the service images)."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from qforecast.core.ringbuffer import RingBuffer

log = logging.getLogger(__name__)


@dataclass
class ModelMeta:
    version: str
    feature_names: list[str]
    lookback: int
    horizons: list[int]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    y_scale: np.ndarray
    sigma: np.ndarray

    @classmethod
    def load(cls, path: Path) -> ModelMeta:
        d = json.loads(Path(path).read_text())
        return cls(
            version=d["version"], feature_names=d["feature_names"], lookback=d["lookback"],
            horizons=d["horizons"],
            feat_mean=np.asarray(d["feat_mean"], np.float32),
            feat_std=np.asarray(d["feat_std"], np.float32),
            y_scale=np.asarray(d["y_scale"], np.float32),
            sigma=np.asarray(d["sigma"], np.float32),
        )


def make_session(model_path: Path, threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    # Batch-1, small model: a single intra-op thread avoids thread-pool wake-up
    # jitter, which dominates p99 latency at this size (see bench_inference.py).
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])


class Predictor:
    """Keeps a rolling window of *normalised* features and runs the model on it.

    The window lives in a double-write ring buffer, so the model input is a
    zero-copy, C-contiguous view: no per-inference allocation or concatenation.
    """

    CLIP = 5.0

    def __init__(self, model_path: Path, meta_path: Path, capacity: int = 1_000, threads: int = 1):
        self.model_path = Path(model_path)
        self.meta_path = Path(meta_path)
        self.meta = ModelMeta.load(meta_path)
        self.session = make_session(self.model_path, threads)
        self._input = self.session.get_inputs()[0].name
        self.window = RingBuffer(max(capacity, self.meta.lookback), len(self.meta.feature_names),
                                 dtype=np.float32)
        self._z = np.zeros(len(self.meta.feature_names), np.float32)
        self._mtime = self._model_mtime()
        self.warmup()

    @property
    def lookback(self) -> int:
        return self.meta.lookback

    @property
    def ready(self) -> bool:
        return len(self.window) >= self.meta.lookback

    def push(self, features: np.ndarray) -> None:
        z = self._z
        np.subtract(features, self.meta.feat_mean, out=z, casting="unsafe")
        np.divide(z, self.meta.feat_std, out=z)
        np.clip(z, -self.CLIP, self.CLIP, out=z)
        self.window.push(z)

    def predict(self) -> np.ndarray:
        """Expected log-return per horizon (float32[H])."""
        x = self.window.last(self.meta.lookback)[None]
        y = self.session.run(None, {self._input: x})[0][0]
        return y * self.meta.y_scale

    def warmup(self, n: int = 20) -> None:
        x = np.zeros((1, self.meta.lookback, len(self.meta.feature_names)), np.float32)
        for _ in range(n):
            self.session.run(None, {self._input: x})

    # ---- hot reload --------------------------------------------------------
    def _model_mtime(self) -> float:
        try:
            return max(os.stat(self.model_path).st_mtime, os.stat(self.meta_path).st_mtime)
        except FileNotFoundError:
            return 0.0

    def maybe_reload(self) -> bool:
        """Swap in a newly exported model (trainer writes files atomically)."""
        mtime = self._model_mtime()
        if mtime <= self._mtime:
            return False
        meta = ModelMeta.load(self.meta_path)
        session = make_session(self.model_path)
        if (meta.lookback != self.meta.lookback
                or meta.feature_names != self.meta.feature_names):
            log.warning("new model has incompatible inputs; ignoring reload")
            self._mtime = mtime
            return False
        # Re-normalise the stored window with the new statistics.
        raw = self.window.last(len(self.window)) * self.meta.feat_std + self.meta.feat_mean
        self.meta, self.session, self._mtime = meta, session, mtime
        self._input = session.get_inputs()[0].name
        self.window = RingBuffer(self.window.capacity, self.window.width, dtype=np.float32)
        for row in raw:
            self.push(row)
        self.warmup()
        log.info("model hot-reloaded -> version %s", meta.version)
        return True


def decide_signal(exp_logret: np.ndarray, sigma: np.ndarray, horizons: list[int],
                  threshold_bps: float) -> tuple[str, int, float]:
    """Pick the horizon with the best risk-adjusted forecast; trade only if the
    expected move clears the cost threshold."""
    z = np.abs(exp_logret) / np.maximum(sigma, 1e-12)
    j = int(np.argmax(z))
    edge_bps = float(exp_logret[j] * 1e4)
    if edge_bps > threshold_bps:
        sig = "LONG"
    elif edge_bps < -threshold_bps:
        sig = "SHORT"
    else:
        sig = "FLAT"
    return sig, int(horizons[j]), edge_bps
