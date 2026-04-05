# DMS2F-HAD ANIMA Module (Baseline)

This repository contains a clean baseline implementation of **DMS2F-HAD** (dual-branch Mamba-inspired spatial-spectral fusion for hyperspectral anomaly detection), aligned with:
- Paper: `papers/2602.04102.pdf`
- Upstream reference: `references_upstream/DMS2F-HAD`

## What is included
- `.mat` HSI dataset loader (`data`/`hsi` + optional `map`/`hsi_gt`)
- Patch extract/fold reconstruction pipeline
- Dual-branch encoder:
  - spatial branch (multi-scale conv + sequence mixer)
  - spectral branch (grouped spectral sequence mixer)
- Adaptive gated fusion
- SS decoder + residual anomaly map generation
- Train and infer CLI
- Synthetic smoke test

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Train
```bash
dms2f-had train \
  --data /path/to/dataset.mat \
  --dataset-name my_dataset \
  --epochs 10 \
  --output-dir artifacts
```

## Infer
```bash
dms2f-had infer \
  --data /path/to/dataset.mat \
  --checkpoint artifacts/checkpoints/my_dataset/best.pt \
  --output-dir artifacts
```

## Outputs
- Checkpoints: `artifacts/checkpoints/<dataset_name>/`
- Results: `artifacts/results/<dataset_name>/`
  - `residual_best.mat` (during train when best AUC is updated)
  - `residual_infer.mat` (during infer)

## Notes
- If `mamba_ssm` is available, it is used for sequence mixing.
- Otherwise, a lightweight fallback mixer is used so the module remains runnable locally.
- This baseline is intentionally structured for later CUDA-server optimization.
