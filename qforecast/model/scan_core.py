"""Scan primitives for the linear recurrence h_t = a_t * h_{t-1} + b_t (h_{-1} = 0)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def parallel_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """h_t for h_t = a_t * h_{t-1} + b_t, h_{-1} = 0.  a, b: (B, L, D, N)."""
    length = a.shape[1]
    d = 1
    while d < length:
        a_prev = F.pad(a[:, : length - d], (0, 0, 0, 0, d, 0), value=1.0)  # identity element
        b_prev = F.pad(b[:, : length - d], (0, 0, 0, 0, d, 0), value=0.0)
        b = a * b_prev + b  # uses the old a: do not reorder these two lines
        a = a * a_prev
        d *= 2
    return b


def scan_sequential(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    h = torch.zeros_like(b[:, 0])
    out = []
    for t in range(a.shape[1]):
        h = a[:, t] * h + b[:, t]
        out.append(h)
    return torch.stack(out, dim=1)
