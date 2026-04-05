from __future__ import annotations

import torch


def random_spatial_mask(
    x: torch.Tensor, prob: float = 0.5, size_ratio: float = 0.2
) -> torch.Tensor:
    """Mask a random spatial rectangle across all channels for each sample."""
    if prob <= 0:
        return x
    if x.ndim != 4:
        raise ValueError(f"Expected shape (B, C, H, W), got {x.shape}")
    if torch.rand(1, device=x.device).item() > prob:
        return x

    b, _, h, w = x.shape
    mask = torch.ones_like(x)
    ph = max(1, int(h * size_ratio))
    pw = max(1, int(w * size_ratio))
    for i in range(b):
        top = torch.randint(0, h - ph + 1, (1,), device=x.device).item()
        left = torch.randint(0, w - pw + 1, (1,), device=x.device).item()
        mask[i, :, top : top + ph, left : left + pw] = 0.0
    return x * mask
