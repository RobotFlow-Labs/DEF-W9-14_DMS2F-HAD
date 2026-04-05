from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .config import DataConfig, ModelConfig, TrainConfig
from .data import HSIPatchDataset
from .model import DMS2FHAD
from .trainer import infer_model, train_model


def _build_dataset(args: argparse.Namespace, data_cfg: DataConfig) -> HSIPatchDataset:
    return HSIPatchDataset(
        mat_path=args.data,
        patch_size=data_cfg.patch_size,
        stride=data_cfg.stride,
        image_key=data_cfg.image_key,
        mask_key=data_cfg.mask_key,
    )


def _build_model(args: argparse.Namespace, in_channels: int, model_cfg: ModelConfig) -> DMS2FHAD:
    return DMS2FHAD(
        in_channels=in_channels,
        cfg=model_cfg,
        mode=args.mode,
    )


def _device(cpu: bool) -> torch.device:
    if cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DMS2F-HAD baseline CLI")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common_data_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--data", type=str, required=True, help="Path to .mat HSI dataset")
        sp.add_argument("--dataset-name", type=str, default=None, help="Dataset name override")
        sp.add_argument("--patch-size", type=int, default=16)
        sp.add_argument("--stride", type=int, default=8)
        sp.add_argument("--image-key", type=str, default=None)
        sp.add_argument("--mask-key", type=str, default=None)
        sp.add_argument("--output-dir", type=str, default="artifacts")
        sp.add_argument("--cpu", action="store_true")

    def add_model_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--mode", type=str, default="full", choices=["full", "spatial", "spectral"])
        sp.add_argument("--embed-dim", type=int, default=64)
        sp.add_argument("--depth", type=int, default=1)
        sp.add_argument("--spectral-group-size", type=int, default=16)
        sp.add_argument("--spectral-group-stride", type=int, default=8)
        sp.add_argument("--mask-prob", type=float, default=0.5)
        sp.add_argument("--mask-size", type=float, default=0.2)

    train = sub.add_parser("train", help="Train reconstruction model")
    add_common_data_args(train)
    add_model_args(train)
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--l1-weight", type=float, default=0.1)
    train.add_argument("--seed", type=int, default=42)

    infer = sub.add_parser("infer", help="Run inference from checkpoint")
    add_common_data_args(infer)
    add_model_args(infer)
    infer.add_argument("--checkpoint", type=str, required=True)
    infer.add_argument("--batch-size", type=int, default=32)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_cfg = DataConfig(
        patch_size=args.patch_size,
        stride=args.stride,
        image_key=args.image_key,
        mask_key=args.mask_key,
    )
    model_cfg = ModelConfig(
        embed_dim=args.embed_dim,
        depth=args.depth,
        spectral_group_size=args.spectral_group_size,
        spectral_group_stride=args.spectral_group_stride,
        random_mask_prob=args.mask_prob,
        random_mask_size=args.mask_size,
    )

    dataset = _build_dataset(args, data_cfg)
    dataset_name = args.dataset_name or Path(args.data).stem
    model = _build_model(args, in_channels=dataset.data.patches.shape[1], model_cfg=model_cfg)
    dev = _device(args.cpu)
    out_dir = Path(args.output_dir)

    print(
        f"Running {args.command} on {dataset_name} | device={dev} "
        f"| params={model.num_parameters:,} trainable={model.trainable_parameters:,}"
    )

    if args.command == "train":
        train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            l1_weight=args.l1_weight,
            seed=args.seed,
        )
        ckpt, result = train_model(
            model=model,
            dataset=dataset,
            dataset_name=dataset_name,
            cfg=train_cfg,
            output_dir=out_dir,
            device=dev,
        )
        print(f"Best checkpoint: {ckpt}")
        print(f"Best AUC: {result.auc if result.auc is not None else 'N/A'}")
    else:
        result = infer_model(
            model=model,
            checkpoint=Path(args.checkpoint),
            dataset=dataset,
            batch_size=args.batch_size,
            output_dir=out_dir,
            dataset_name=dataset_name,
            device=dev,
        )
        print(f"Inference AUC: {result.auc if result.auc is not None else 'N/A'}")


if __name__ == "__main__":
    main()
