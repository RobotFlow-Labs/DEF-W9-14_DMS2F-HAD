from __future__ import annotations

import numpy as np
import torch


def compute_patch_positions(
    image_shape: tuple[int, int], patch_size: int, stride: int
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    h, w = image_shape
    pad_h = 0
    pad_w = 0
    if (h - patch_size) % stride != 0:
        pad_h = stride - ((h - patch_size) % stride)
    if (w - patch_size) % stride != 0:
        pad_w = stride - ((w - patch_size) % stride)

    h_pad = h + pad_h
    w_pad = w + pad_w
    positions: list[tuple[int, int]] = []
    for i in range(0, h_pad - patch_size + 1, stride):
        for j in range(0, w_pad - patch_size + 1, stride):
            positions.append((i, j))
    return positions, (h_pad, w_pad)


def extract_patches(
    image: np.ndarray, patch_size: int, stride: int
) -> tuple[torch.Tensor, list[tuple[int, int]], tuple[int, int]]:
    if image.ndim != 3:
        raise ValueError(f"Expected image shape (H, W, C), got {image.shape}")

    h, w, _ = image.shape
    positions, (h_pad, w_pad) = compute_patch_positions((h, w), patch_size, stride)
    pad_h = h_pad - h
    pad_w = w_pad - w
    padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant")

    blocks = []
    for i, j in positions:
        blocks.append(padded[i : i + patch_size, j : j + patch_size, :])
    arr = np.stack(blocks, axis=0).astype(np.float32)
    # (N, H, W, C) -> (N, C, H, W)
    return torch.from_numpy(arr).permute(0, 3, 1, 2), positions, (h_pad, w_pad)


def fold_patches(
    patches: torch.Tensor,
    image_shape: tuple[int, int],
    patch_size: int,
    positions: list[tuple[int, int]],
) -> torch.Tensor:
    """Reconstruct image by overlap averaging.

    Args:
      patches: (N, C, patch_size, patch_size)
      image_shape: original (H, W)
      positions: top-left position for each patch
    Returns:
      Tensor (C, H, W)
    """
    if patches.ndim != 4:
        raise ValueError(f"Expected patches shape (N,C,H,W), got {patches.shape}")
    if len(positions) != patches.shape[0]:
        raise ValueError("positions length and number of patches mismatch")

    h_orig, w_orig = image_shape
    c = patches.shape[1]
    h_pad = max(i for i, _ in positions) + patch_size
    w_pad = max(j for _, j in positions) + patch_size

    out = torch.zeros((c, h_pad, w_pad), dtype=patches.dtype, device=patches.device)
    cnt = torch.zeros((h_pad, w_pad), dtype=patches.dtype, device=patches.device)

    for patch, (i, j) in zip(patches, positions):
        out[:, i : i + patch_size, j : j + patch_size] += patch
        cnt[i : i + patch_size, j : j + patch_size] += 1

    cnt = torch.clamp(cnt, min=1)
    out = out / cnt.unsqueeze(0)
    return out[:, :h_orig, :w_orig]
