from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import TrainConfig
from .data import HSIPatchDataset
from .patches import fold_patches


@dataclass(slots=True)
class EvalResult:
    residual_map: np.ndarray
    reconstructed: np.ndarray
    original: np.ndarray
    auc: float | None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_full_image(
    model: torch.nn.Module,
    dataset: HSIPatchDataset,
    batch_size: int,
    device: torch.device,
) -> EvalResult:
    model.eval()
    patches = dataset.data.patches
    loader = DataLoader(patches, batch_size=batch_size, shuffle=False)
    rec_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for block in loader:
            block = block.to(device)
            pred, _, _ = model(block, apply_mask=False)
            rec_parts.append(pred.detach().cpu())

    rec_patches = torch.cat(rec_parts, dim=0)
    rec_full = fold_patches(
        rec_patches,
        image_shape=(dataset.data.image.shape[0], dataset.data.image.shape[1]),
        patch_size=dataset.patch_size,
        positions=dataset.data.positions,
    )

    orig = torch.from_numpy(dataset.data.image.astype(np.float32)).permute(2, 0, 1)
    diff = rec_full - orig
    residual = torch.linalg.vector_norm(diff, ord=2, dim=0).cpu().numpy()

    auc: float | None = None
    if dataset.data.mask is not None:
        gt = dataset.data.mask.astype(np.int64).ravel()
        score = residual.ravel()
        if np.unique(gt).size >= 2:
            auc = float(roc_auc_score(gt, score))

    return EvalResult(
        residual_map=residual,
        reconstructed=rec_full.cpu().numpy(),
        original=orig.cpu().numpy(),
        auc=auc,
    )


def save_eval_result(result: EvalResult, path: Path, mask: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "residual_map": result.residual_map,
        "reconstructed": result.reconstructed,
        "original": result.original,
    }
    if mask is not None:
        payload["gt_mask"] = mask
    savemat(str(path), payload)


def train_model(
    model: torch.nn.Module,
    dataset: HSIPatchDataset,
    dataset_name: str,
    cfg: TrainConfig,
    output_dir: Path,
    device: torch.device,
) -> tuple[Path, EvalResult]:
    set_seed(cfg.seed)
    model.to(device)
    model.train()

    ckpt_dir = output_dir / "checkpoints" / dataset_name
    res_dir = output_dir / "results" / dataset_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    train_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_metric = -float("inf")
    best_path = ckpt_dir / "best.pt"
    best_eval: EvalResult | None = None

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        n = 0
        for patch in train_loader:
            patch = patch.to(device)
            pred, _, _ = model(patch, apply_mask=True)
            mse = F.mse_loss(pred, patch)
            l1 = F.l1_loss(pred, patch)
            loss = mse + cfg.l1_weight * l1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += loss.item() * patch.shape[0]
            n += patch.shape[0]

        train_loss = running / max(1, n)
        eval_result = evaluate_full_image(model, dataset, cfg.batch_size, device)
        val_mse = float(np.mean((eval_result.reconstructed - eval_result.original) ** 2))
        metric = eval_result.auc if eval_result.auc is not None else -val_mse

        print(
            f"Epoch {epoch + 1}/{cfg.epochs} "
            f"train_loss={train_loss:.6f} val_mse={val_mse:.6f} "
            f"auc={eval_result.auc if eval_result.auc is not None else 'N/A'}"
        )

        if metric > best_metric:
            best_metric = metric
            best_eval = eval_result
            torch.save(model.state_dict(), best_path)
            save_eval_result(
                eval_result,
                res_dir / "residual_best.mat",
                mask=dataset.data.mask,
            )

    if best_eval is None:
        best_eval = evaluate_full_image(model, dataset, cfg.batch_size, device)
        save_eval_result(best_eval, res_dir / "residual_best.mat", mask=dataset.data.mask)

    return best_path, best_eval


def infer_model(
    model: torch.nn.Module,
    checkpoint: Path,
    dataset: HSIPatchDataset,
    batch_size: int,
    output_dir: Path,
    dataset_name: str,
    device: torch.device,
) -> EvalResult:
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    result = evaluate_full_image(model, dataset, batch_size=batch_size, device=device)
    save_eval_result(
        result,
        output_dir / "results" / dataset_name / "residual_infer.mat",
        mask=dataset.data.mask,
    )
    return result
