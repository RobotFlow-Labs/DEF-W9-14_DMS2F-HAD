from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig
from .masking import random_spatial_mask
from .mixers import SequenceMixerStack


def _group_starts(length: int, group_size: int, stride: int) -> list[int]:
    if group_size <= 0 or stride <= 0:
        raise ValueError("group_size and stride must be positive")
    if length <= group_size:
        return [0]
    starts = list(range(0, length - group_size + 1, stride))
    last = length - group_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def split_spectral_groups(
    x: torch.Tensor, group_size: int, stride: int
) -> tuple[torch.Tensor, list[int]]:
    """Split channels into overlapping spectral groups.

    Args:
      x: tensor (B, C, H, W)
    Returns:
      groups: (B, G, S, H, W), starts
    """
    b, c, h, w = x.shape
    starts = _group_starts(c, group_size, stride)
    groups = []
    for s in starts:
        e = s + group_size
        if e <= c:
            g = x[:, s:e, :, :]
        else:
            pad = e - c
            tail = x[:, s:c, :, :]
            g = torch.cat([tail, torch.zeros((b, pad, h, w), device=x.device, dtype=x.dtype)], dim=1)
        groups.append(g)
    return torch.stack(groups, dim=1), starts


class SSDecoderBlock(nn.Module):
    def __init__(self, channels: int, depth: int = 1) -> None:
        super().__init__()
        self.global_mixer = SequenceMixerStack(d_model=channels, depth=depth)
        self.local3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.local5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2)
        self.fuse = nn.Conv2d(channels * 3, channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        token = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        g = self.global_mixer(token).reshape(b, h, w, c).permute(0, 3, 1, 2)
        l3 = self.local3(x)
        l5 = self.local5(x)
        out = self.act(self.fuse(torch.cat([g, l3, l5], dim=1)))
        return out + x


class DMS2FHAD(nn.Module):
    """Dual-branch Mamba-inspired spatial-spectral fusion network."""

    def __init__(
        self,
        in_channels: int,
        cfg: ModelConfig | None = None,
        mode: str = "full",
    ) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.mode = mode
        d = self.cfg.embed_dim

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, d, kernel_size=1),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )

        # Spatial branch
        self.spa3 = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=3, padding=1),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )
        self.spa5 = nn.Sequential(
            nn.Conv2d(d, d, kernel_size=5, padding=2),
            nn.BatchNorm2d(d),
            nn.GELU(),
        )
        self.spa_mixer = SequenceMixerStack(d_model=d, depth=self.cfg.depth)

        # Spectral branch
        self.group_size = self.cfg.spectral_group_size
        self.group_stride = self.cfg.spectral_group_stride
        n_groups = len(_group_starts(d, self.group_size, self.group_stride))
        self.spe_mixer = SequenceMixerStack(d_model=self.group_size, depth=self.cfg.depth)
        self.spe_proj = nn.Sequential(
            nn.Linear(n_groups * self.group_size, d),
            nn.LayerNorm(d),
            nn.GELU(),
        )

        self.gate_conv = nn.Sequential(nn.Conv2d(2 * d, 1, kernel_size=1), nn.Sigmoid())
        self.fusion_proj = nn.Sequential(nn.Conv2d(d, d, kernel_size=1), nn.GELU())

        # SS decoder
        self.dec1 = SSDecoderBlock(d, depth=2)
        self.dec2 = SSDecoderBlock(d, depth=2)
        self.out = nn.Conv2d(d, in_channels, kernel_size=1)

    def _spatial_branch(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        ms = self.spa3(feat) + self.spa5(feat)
        token = ms.permute(0, 2, 3, 1).reshape(b, h * w, c)
        token = self.spa_mixer(token)
        return token.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def _spectral_branch(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        groups, _ = split_spectral_groups(feat, self.group_size, self.group_stride)
        # (B, G, S, H, W) -> (B*H*W, G, S)
        token = groups.permute(0, 3, 4, 1, 2).reshape(b * h * w, groups.shape[1], groups.shape[2])
        token = self.spe_mixer(token)
        token = token.reshape(b * h * w, -1)
        token = self.spe_proj(token)
        return token.reshape(b, h, w, c).permute(0, 3, 1, 2)

    def forward(
        self, x: torch.Tensor, apply_mask: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.training and apply_mask:
            x = random_spatial_mask(
                x, prob=self.cfg.random_mask_prob, size_ratio=self.cfg.random_mask_size
            )
        feat = self.pre(x)

        spa = self._spatial_branch(feat)
        spe = self._spectral_branch(feat)

        gate: torch.Tensor | None = None
        if self.mode == "spatial":
            fused = self.fusion_proj(spa)
        elif self.mode == "spectral":
            fused = self.fusion_proj(spe)
        else:
            gate = self.gate_conv(torch.cat([spa, spe], dim=1))
            fused = self.fusion_proj(gate * spa + (1.0 - gate) * spe)

        z = self.dec1(fused)
        z = self.dec2(z)
        recon = self.out(z)
        return recon, fused, gate

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
