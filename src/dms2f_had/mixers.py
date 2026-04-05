from __future__ import annotations

import torch
from torch import nn


class FallbackMixer(nn.Module):
    """Simple sequence mixer used when mamba_ssm is unavailable."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        h = self.norm(x)
        h = h.transpose(1, 2)  # (B, D, L)
        h = self.dw(h)
        h = self.act(self.pw(h))
        h = h.transpose(1, 2)
        return h


def _make_mamba(d_model: int) -> nn.Module:
    try:
        from mamba_ssm import Mamba  # type: ignore

        return Mamba(
            d_model=d_model,
            d_state=64,
            d_conv=4,
            expand=2,
            use_fast_path=False,
        )
    except Exception:
        return FallbackMixer(d_model=d_model)


class SequenceMixerStack(nn.Module):
    def __init__(self, d_model: int, depth: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_make_mamba(d_model) for _ in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(depth)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, D)
        for norm, layer in zip(self.norms, self.layers):
            h = norm(x)
            y = layer(h)
            x = x + y if y.shape == x.shape else y
        return x
