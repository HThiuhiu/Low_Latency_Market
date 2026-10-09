"""Leak-free supervised dataset construction.

Sample at bar t (bar t is closed):
    X_t = z-scored features of bars [t-L+1, t]
    y_t = log(close[t+h] / close[t])  for every horizon h

Guarantees (asserted in tests/test_dataset.py):
* inputs only use information available at the close of bar t;
* windows/labels never straddle a gap in the exchange data;
* train / val / test are chronological with an embargo of max(h) bars so that no
  training label overlaps the validation/test period;
* normalisation statistics are fitted on the training split only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def make_targets(close: np.ndarray, horizons: list[int]) -> np.ndarray:
    n = len(close)
    logc = np.log(close)
    y = np.full((n, len(horizons)), np.nan)
    for j, h in enumerate(horizons):
        y[:n - h, j] = logc[h:] - logc[:-h]
    return y


def eligible_indices(open_time: np.ndarray, valid: np.ndarray, interval_ms: int,
                     lookback: int, max_h: int) -> np.ndarray:
    """Bars t whose input window and label span are complete and gap-free."""
    n = len(open_time)
    bad = ~valid.copy()
    gap = np.zeros(n, dtype=bool)
    gap[1:] = np.diff(open_time) != interval_ms  # bar i does not follow bar i-1
    # A gap at i breaks every span that contains both i-1 and i.
    bad_c = np.concatenate([[0], np.cumsum(bad)])
    gap_c = np.concatenate([[0], np.cumsum(gap)])
    t = np.arange(lookback - 1, n - max_h)
    lo, hi = t - lookback + 1, t + max_h
    ok = (bad_c[t + 1] - bad_c[lo] == 0) & (gap_c[hi + 1] - gap_c[lo + 1] == 0)
    return t[ok]


@dataclass
class Splits:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def chronological_split(idx: np.ndarray, embargo: int, val_frac: float = 0.15,
                        test_frac: float = 0.15) -> Splits:
    n = len(idx)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    test_start = idx[n - n_test]
    val_start = idx[n - n_test - n_val]
    train = idx[idx < val_start - embargo]
    val = idx[(idx >= val_start) & (idx < test_start - embargo)]
    test = idx[idx >= test_start]
    return Splits(train, val, test)


def fit_normaliser(feats: np.ndarray, train_idx: np.ndarray, lookback: int):
    """Mean/std over every bar that appears in a training window."""
    rows = np.unique(np.clip(train_idx[:, None] - np.arange(lookback), 0, None))
    sub = feats[rows]
    mean = sub.mean(axis=0)
    std = sub.std(axis=0) + 1e-12
    return mean.astype(np.float32), std.astype(np.float32)


class WindowBatcher:
    """Materialises (B, L, F) windows on the fly by index arithmetic, instead of
    storing an N x L x F tensor (128x less memory) or per-sample __getitem__ calls."""

    def __init__(self, z, y, lookback: int):
        import torch

        self.z = torch.as_tensor(z, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.offsets = torch.arange(-lookback + 1, 1)

    def batch(self, idx, device):
        import torch

        idx = torch.as_tensor(idx)
        x = self.z[idx[:, None] + self.offsets]
        return x.to(device, non_blocking=True), self.y[idx].to(device, non_blocking=True)
