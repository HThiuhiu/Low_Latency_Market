"""Fixed-capacity ring buffer whose most recent ``n`` rows are always a contiguous view.

Every row is written twice (at ``i`` and ``i + capacity``), so ``last(n)`` is a plain
slice — no modulo arithmetic, no concatenation, no copy. The view can be passed
straight to a numba kernel or to ONNX Runtime as a C-contiguous input tensor.
"""

from __future__ import annotations

import numpy as np


class RingBuffer:
    __slots__ = ("capacity", "width", "_buf", "_head", "count")

    def __init__(self, capacity: int, width: int, dtype=np.float64):
        self.capacity = capacity
        self.width = width
        self._buf = np.zeros((2 * capacity, width), dtype=dtype)
        self._head = 0  # next write position in [0, capacity)
        self.count = 0  # total rows ever written

    def push(self, row) -> None:
        h = self._head
        self._buf[h] = row
        self._buf[h + self.capacity] = row
        self._head = h + 1 if h + 1 < self.capacity else 0
        self.count += 1

    def overwrite_last(self, row) -> None:
        """Replace the most recent row (e.g. a bar that was re-published)."""
        h = self._head - 1 if self._head > 0 else self.capacity - 1
        self._buf[h] = row
        self._buf[h + self.capacity] = row

    def __len__(self) -> int:
        return min(self.count, self.capacity)

    def last(self, n: int) -> np.ndarray:
        """Contiguous read-only view of the latest ``n`` rows (oldest first)."""
        if n > len(self):
            raise ValueError(f"requested {n} rows, only {len(self)} available")
        end = self._head + self.capacity
        view = self._buf[end - n:end]
        view.flags.writeable = False
        return view

    def latest(self) -> np.ndarray:
        return self.last(1)[0]
