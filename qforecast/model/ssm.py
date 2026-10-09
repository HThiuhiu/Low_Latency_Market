"""Mamba-style selective state-space block with a parallel (associative) scan.

The recurrence of a selective SSM,

    h_t = a_t * h_{t-1} + b_t           (a_t = exp(dt_t * A),  b_t = dt_t * B_t * x_t)

is linear, so composing two steps is associative:

    (a2, b2) o (a1, b1) = (a2 * a1,  a2 * b1 + b2)

so the scan needs only ceil(log2 L) vectorised steps instead of L sequential ones.
The scan itself is delegated to ``scan_fused.selective_scan``, which picks a backend:
plain-torch parallel scan (CPU, ONNX export), a memory-efficient chunked version, or
the fused Triton/CUDA kernel on GPUs.

``SelectiveSSM.step`` is the O(1)-per-bar recurrent form for streaming inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from qforecast.model.scan_core import parallel_scan, scan_sequential  # noqa: F401 (re-export)
from qforecast.model.scan_fused import selective_scan


class SelectiveSSM(nn.Module):
    """Mamba (S6) mixer: in-proj -> causal depthwise conv -> selective scan -> gate -> out-proj."""

    def __init__(self, d_model: int, d_state: int = 8, expand: int = 1, d_conv: int = 4,
                 dt_rank: int | None = None, scan_backend: str = "auto"):
        super().__init__()
        self.scan_backend = scan_backend  # auto | parallel | chunked | triton
        self.d_inner = d_inner = expand * d_model
        self.d_state, self.d_conv = d_state, d_conv
        self.dt_rank = dt_rank or max(4, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.conv = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # S4D-real initialisation; dt initialised so softplus(dt) lies in [1e-3, 1e-1].
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus

    def _ssm_params(self, x: torch.Tensor):
        dt, B, C = torch.split(self.x_proj(x), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, D)
        A = -torch.exp(self.A_log)  # (D, N)
        return dt, A, B, C

    def forward(self, u: torch.Tensor) -> torch.Tensor:  # u: (B, L, d_model)
        length = u.shape[1]
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2)  # causal
        x = F.silu(x)
        dt, A, B, C = self._ssm_params(x)
        y = selective_scan(x, dt, A, B, C, self.D, backend=self.scan_backend)
        return self.out_proj(y * F.silu(z))

    @torch.no_grad()
    def step(self, u_t: torch.Tensor, conv_state: torch.Tensor, h: torch.Tensor):
        """One bar in O(1): u_t (B, d_model), conv_state (B, d_inner, d_conv-1), h (B, D, N)."""
        x, z = self.in_proj(u_t).chunk(2, dim=-1)
        window = torch.cat([conv_state, x.unsqueeze(-1)], dim=-1)  # (B, D, d_conv)
        x = F.silu((window * self.conv.weight.squeeze(1)).sum(-1) + self.conv.bias)
        dt, A, B, C = self._ssm_params(x)  # dt: (B, D), B/C: (B, N)
        h = torch.exp(dt.unsqueeze(-1) * A) * h + (dt * x).unsqueeze(-1) * B.unsqueeze(1)
        y = (h * C.unsqueeze(1)).sum(-1) + self.D * x
        return self.out_proj(y * F.silu(z)), window[..., 1:], h


class MambaBlock(nn.Module):
    """Pre-norm residual block."""

    def __init__(self, d_model: int, **kw):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = SelectiveSSM(d_model, **kw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixer(self.norm(x))
