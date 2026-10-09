"""Allocation-free latency histogram + Prometheus text exposition.

Log-linear buckets (HDR-histogram style): 2**k ranges split into 16 sub-buckets,
giving <= ~6% relative error from 100 ns up to ~70 s with O(1) record().
"""

from __future__ import annotations

import math
import threading

import numpy as np

_SUB = 16
_MIN_EXP = 6  # 64 ns
_MAX_EXP = 36  # ~68 s
_NBUCKETS = (_MAX_EXP - _MIN_EXP) * _SUB


class LatencyHistogram:
    def __init__(self, name: str, help_text: str = ""):
        self.name = name
        self.help = help_text
        self._counts = np.zeros(_NBUCKETS, dtype=np.int64)
        self.n = 0
        self.total_ns = 0
        self.max_ns = 0

    @staticmethod
    def _index(ns: int) -> int:
        if ns < (1 << _MIN_EXP):
            return 0
        e = ns.bit_length() - 1
        if e >= _MAX_EXP:
            return _NBUCKETS - 1
        sub = (ns >> (e - 4)) & (_SUB - 1)  # next 4 bits after the leading one
        return (e - _MIN_EXP) * _SUB + sub

    @staticmethod
    def _upper(idx: int) -> float:
        e, sub = divmod(idx, _SUB)
        e += _MIN_EXP
        return (1 << e) * (1 + (sub + 1) / _SUB)

    def record(self, ns: int) -> None:
        ns = int(ns)
        self._counts[self._index(ns)] += 1
        self.n += 1
        self.total_ns += ns
        if ns > self.max_ns:
            self.max_ns = ns

    def percentile(self, q: float) -> float:
        """Approximate q-th percentile in nanoseconds (upper bucket bound)."""
        if self.n == 0:
            return math.nan
        target = q / 100.0 * self.n
        idx = int(np.searchsorted(np.cumsum(self._counts), target, side="left"))
        return min(self._upper(idx), float(self.max_ns))

    def summary_us(self) -> dict:
        p = lambda q: round(self.percentile(q) / 1e3, 2)  # noqa: E731
        return {
            "count": self.n,
            "mean_us": round(self.total_ns / self.n / 1e3, 2) if self.n else None,
            "p50_us": p(50), "p90_us": p(90), "p99_us": p(99), "p999_us": p(99.9),
            "max_us": round(self.max_ns / 1e3, 2),
        }


class Registry:
    """Tiny metrics registry (counters, gauges, histograms) with Prometheus output."""

    def __init__(self, service: str):
        self.service = service
        self._lock = threading.Lock()
        self.counters: dict[str, float] = {}
        self.gauges: dict[str, float] = {}
        self.histograms: dict[str, LatencyHistogram] = {}

    def inc(self, name: str, value: float = 1.0) -> None:
        self.counters[name] = self.counters.get(name, 0.0) + value

    def set(self, name: str, value: float) -> None:
        self.gauges[name] = value

    def hist(self, name: str, help_text: str = "") -> LatencyHistogram:
        h = self.histograms.get(name)
        if h is None:
            with self._lock:
                h = self.histograms.setdefault(name, LatencyHistogram(name, help_text))
        return h

    def snapshot(self) -> dict:
        return {
            "service": self.service,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "latency": {k: h.summary_us() for k, h in self.histograms.items()},
        }

    def prometheus(self) -> str:
        lbl = f'service="{self.service}"'
        out: list[str] = []
        for k, v in self.counters.items():
            out += [f"# TYPE qf_{k} counter", f"qf_{k}{{{lbl}}} {v}"]
        for k, v in self.gauges.items():
            out += [f"# TYPE qf_{k} gauge", f"qf_{k}{{{lbl}}} {v}"]
        for k, h in self.histograms.items():
            out.append(f"# TYPE qf_{k}_seconds summary")
            for q in (0.5, 0.9, 0.99, 0.999):
                val = h.percentile(q * 100) / 1e9 if h.n else 0.0
                out.append(f'qf_{k}_seconds{{{lbl},quantile="{q}"}} {val:.9f}')
            out.append(f"qf_{k}_seconds_sum{{{lbl}}} {h.total_ns / 1e9:.9f}")
            out.append(f"qf_{k}_seconds_count{{{lbl}}} {h.n}")
        return "\n".join(out) + "\n"
