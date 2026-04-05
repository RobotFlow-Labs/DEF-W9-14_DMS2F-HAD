from dataclasses import dataclass


@dataclass(slots=True)
class DataConfig:
    patch_size: int = 16
    stride: int = 8
    image_key: str | None = None
    mask_key: str | None = None


@dataclass(slots=True)
class ModelConfig:
    embed_dim: int = 64
    depth: int = 1
    spectral_group_size: int = 16
    spectral_group_stride: int = 8
    random_mask_prob: float = 0.5
    random_mask_size: float = 0.2


@dataclass(slots=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    l1_weight: float = 0.1
    seed: int = 42
