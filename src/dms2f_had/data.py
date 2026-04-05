from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset

from .patches import extract_patches


IMAGE_KEYS = ("data", "hsi", "image", "img")
MASK_KEYS = ("map", "hsi_gt", "gt", "mask")


def _read_mat(path: Path) -> dict:
    try:
        return loadmat(str(path))
    except NotImplementedError:
        # MATLAB v7.3/HDF5
        out: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                out[k] = np.array(f[k])
        return out


def _pick_key(d: dict, preferred: str | None, candidates: tuple[str, ...]) -> str:
    if preferred and preferred in d:
        return preferred
    for key in candidates:
        if key in d:
            return key
    raise KeyError(f"None of keys {candidates} found in mat file")


def load_hsi_mat(
    mat_path: str | Path, image_key: str | None = None, mask_key: str | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(mat_path)
    if not path.exists():
        raise FileNotFoundError(path)
    mat = _read_mat(path)

    ikey = _pick_key(mat, image_key, IMAGE_KEYS)
    image = np.array(mat[ikey])

    mkey: str | None = None
    if mask_key and mask_key in mat:
        mkey = mask_key
    else:
        for k in MASK_KEYS:
            if k in mat:
                mkey = k
                break
    mask = np.array(mat[mkey]) if mkey else None

    if image.ndim != 3:
        raise ValueError(f"Expected 3D HSI cube, got shape {image.shape}")

    image = image.astype(np.float32)
    mn, mx = float(image.min()), float(image.max())
    if mx > mn:
        image = (image - mn) / (mx - mn)
    else:
        image = np.zeros_like(image, dtype=np.float32)

    if mask is not None:
        mask = np.squeeze(mask).astype(np.int64)
        if mask.ndim != 2:
            raise ValueError(f"Expected 2D mask, got {mask.shape}")
        # Resolve layout to HWC based on mask spatial shape.
        if image.shape[:2] == mask.shape:
            pass  # already HWC
        elif image.shape[1:] == mask.shape:
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
        elif (image.shape[0], image.shape[2]) == mask.shape:
            image = np.transpose(image, (0, 2, 1))  # HCW -> HWC
        else:
            # Keep as-is when no unambiguous mapping exists.
            pass

    return image, mask


@dataclass(slots=True)
class HSIReconstructionData:
    image: np.ndarray
    mask: np.ndarray | None
    patches: torch.Tensor
    positions: list[tuple[int, int]]
    padded_shape: tuple[int, int]


class HSIPatchDataset(Dataset):
    def __init__(
        self,
        mat_path: str | Path,
        patch_size: int = 16,
        stride: int = 8,
        image_key: str | None = None,
        mask_key: str | None = None,
    ) -> None:
        image, mask = load_hsi_mat(mat_path, image_key=image_key, mask_key=mask_key)
        patches, positions, padded = extract_patches(image, patch_size, stride)
        self.data = HSIReconstructionData(
            image=image, mask=mask, patches=patches, positions=positions, padded_shape=padded
        )
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self) -> int:
        return self.data.patches.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data.patches[idx]
