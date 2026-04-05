import numpy as np
import torch

from dms2f_had.config import ModelConfig
from dms2f_had.model import DMS2FHAD
from dms2f_had.patches import extract_patches, fold_patches


def test_patch_extract_fold_roundtrip():
    image = np.random.rand(31, 29, 10).astype(np.float32)
    patches, positions, _ = extract_patches(image, patch_size=8, stride=4)
    recon = fold_patches(
        patches,
        image_shape=(image.shape[0], image.shape[1]),
        patch_size=8,
        positions=positions,
    )
    original = torch.from_numpy(image).permute(2, 0, 1)
    assert recon.shape == original.shape
    assert torch.allclose(recon, original, atol=1e-5)


def test_model_forward_shapes():
    cfg = ModelConfig(
        embed_dim=32,
        depth=1,
        spectral_group_size=8,
        spectral_group_stride=4,
        random_mask_prob=0.0,
    )
    model = DMS2FHAD(in_channels=16, cfg=cfg, mode="full")
    x = torch.randn(2, 16, 16, 16)
    y, fused, gate = model(x, apply_mask=False)
    assert y.shape == x.shape
    assert fused.shape == (2, 32, 16, 16)
    assert gate is not None
    assert gate.shape == (2, 1, 16, 16)
