"""CNN -> BiLSTM -> multi-head attention forecaster (training side, requires torch).

Sized for latency. The recurrent part is the serial bottleneck of a batch-1 forward
pass, so two stride-2 convolutions shrink the sequence 4x (128 -> 32 steps) before
the BiLSTM, and the head predicts every horizon in one pass. Measured with ONNX
Runtime (benchmarks/bench_inference.py): 2-layer BiLSTM on 64 steps ~560 µs ->
1-layer on 32 steps ~175 µs, at the same forecast quality.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, n_features: int, n_outputs: int, conv_channels: int = 64,
                 hidden: int = 64, lstm_layers: int = 1, heads: int = 4, dropout: float = 0.1,
                 downsample: int = 4):
        super().__init__()
        assert downsample in (2, 4)
        c1 = conv_channels // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=5, padding=2, stride=downsample // 2),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Conv1d(c1, conv_channels, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden, num_layers=lstm_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0)
        d = 2 * hidden
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(
            nn.Linear(2 * d, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, F)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L/4, C)
        out, (hn, _) = self.lstm(z)  # out: (B, L/4, 2H)
        summary = torch.cat((hn[-2], hn[-1]), dim=-1)  # final fwd + bwd states
        ctx, _ = self.attn(summary.unsqueeze(1), out, out, need_weights=False)
        h = torch.cat((self.norm(ctx.squeeze(1)), summary), dim=-1)
        return self.head(h)  # (B, n_outputs): normalised log-returns per horizon


class CNNMambaAttention(nn.Module):
    """Same CNN front-end and head as the BiLSTM model; the recurrent block is replaced
    by Mamba layers evaluated with a parallel scan (depth log2(L), not L).

    Mamba is causal, so the "summary" is the last position's output (which has seen the
    whole window) instead of the BiLSTM's forward+backward final states.
    """

    def __init__(self, n_features: int, n_outputs: int, conv_channels: int = 64,
                 d_model: int = 64, n_layers: int = 2, d_state: int = 8, expand: int = 1,
                 heads: int = 4, dropout: float = 0.1, downsample: int = 4,
                 scan_backend: str = "auto"):
        super().__init__()
        from qforecast.model.ssm import MambaBlock

        assert downsample in (2, 4)
        c1 = conv_channels // 2
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, c1, kernel_size=5, padding=2, stride=downsample // 2),
            nn.BatchNorm1d(c1),
            nn.GELU(),
            nn.Conv1d(c1, d_model, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[MambaBlock(d_model, d_state=d_state, expand=expand,
                                                 scan_backend=scan_backend)
                                      for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, F)
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)  # (B, L/4, d_model)
        out = self.norm_f(self.blocks(z))
        last = out[:, -1]  # causal: position L-1 has seen the whole window
        ctx, _ = self.attn(last.unsqueeze(1), out, out, need_weights=False)
        return self.head(torch.cat((self.norm(ctx.squeeze(1)), last), dim=-1))


ARCHS = {"lstm": CNNBiLSTMAttention, "mamba": CNNMambaAttention}


def build_model(arch: str, n_features: int, n_outputs: int, **kw) -> nn.Module:
    return ARCHS[arch](n_features, n_outputs, **kw)
