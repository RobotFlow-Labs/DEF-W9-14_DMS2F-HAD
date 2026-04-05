# ANIMA Defense Module PRD: DMS2F-HAD

## 1. Document Control
- Module: `DEF-DMS2F-HAD`
- Version: `v0.1`
- Date: `2026-04-04`
- Status: Draft (implementation baseline)

## 2. Problem Statement
Hyperspectral anomaly detection (HAD) requires accurate identification of rare targets in noisy, high-dimensional imagery without labels. Existing approaches often trade accuracy for speed:
- CNN-heavy models miss long-range spectral dependencies.
- Transformer-based models are expensive for long spectral sequences.

We need a practical ANIMA defense module that is:
- faithful to DMS2F-HAD paper design,
- efficient enough for later CUDA hardening,
- runnable now in local CPU/GPU environments.

## 3. Goals
- Implement an essential, clean DMS2F-HAD baseline with train/eval/infer support.
- Support unsupervised reconstruction-based HAD on `.mat` datasets.
- Provide anomaly-map output and AUC-based evaluation.
- Keep architecture modular for CUDA-server optimization and kernel replacement.

## 4. Non-Goals (for this phase)
- Reproducing full 14-dataset benchmark numbers.
- Shipping custom CUDA kernels in this repository.
- Full experiment orchestration and distributed training.

## 5. Users and Stakeholders
- ML engineers integrating ANIMA defense modules.
- Research engineers validating HAD models on benchmark datasets.
- Infrastructure team optimizing CUDA execution later.

## 6. Functional Requirements
### FR-1 Data Ingestion
- Load HAD datasets from `.mat` files.
- Support common keys: `data`/`hsi` for image and `map`/`hsi_gt` for mask.
- Normalize image cubes to `[0,1]`.

### FR-2 Patch Pipeline
- Extract overlapping 3D patches using configurable `patch_size` and `stride`.
- Reconstruct full image from patch reconstructions by overlap averaging.

### FR-3 Random Spatial Masking
- During training, apply rectangular spatial masking across all bands with configurable probability and size.

### FR-4 Dual-Branch Encoder
- Spatial branch:
  - multi-scale feature extraction (3x3 + 5x5 conv),
  - sequence modeling over flattened spatial tokens using Mamba-like block.
- Spectral branch:
  - spectral grouping with overlap (`group_size`, `group_stride`),
  - sequence modeling per grouped spectral segment.

### FR-5 Adaptive Gated Fusion
- Fuse spatial and spectral branch outputs with a learnable gate:
  - `G = sigmoid(conv([F_spa, F_spe]))`
  - `F_fused = Proj(G * F_spa + (1-G) * F_spe)`

### FR-6 SS Decoder
- Decode fused features using:
  - global sequence path (Mamba-like),
  - local conv paths (3x3 and 5x5),
  - fusion + projection to original spectral channels.

### FR-7 Anomaly Detection Output
- Compute residual map via pixel-wise spectral L2:
  - `R(i,j) = ||X(i,j,:) - X_hat(i,j,:)||_2`
- Export residual map and optional ROC/AUC metrics when GT exists.

### FR-8 Train/Eval CLI
- Provide command-line entry points for:
  - training,
  - evaluation/inference,
  - saving best checkpoint and residual outputs.

## 7. Non-Functional Requirements
- Deterministic seed support.
- CPU-safe fallback when Mamba dependency is unavailable.
- Clear module boundaries for future CUDA optimization.
- Reasonable memory use for patch-based processing.

## 8. Default Training Configuration (Paper-aligned)
- Optimizer: Adam
- Learning rate: `5e-4`
- Weight decay: `1e-4`
- Epochs: `100` (paper), configurable
- Patch size: `16`
- Stride: `8`
- Embedding dim (`c1`): `64`
- Spectral group size (`c2`): `16`
- Spectral group stride (`k`): `8`

## 9. Success Criteria
- End-to-end train+eval runs on at least one dataset path.
- Residual map is produced at full image resolution.
- AUC computes successfully when GT mask is present and non-uniform.
- Code structure is ready for migration of critical paths to CUDA server.

## 10. Risks and Mitigations
- Risk: `.mat` schema differences across datasets.
  - Mitigation: robust key resolution and helpful load errors.
- Risk: Mamba package/runtime mismatch.
  - Mitigation: local fallback mixer block interface-compatible with Mamba.
- Risk: Large-memory usage on full scenes.
  - Mitigation: patch batching and fold reconstruction.

## 11. Deliverables
- `PRD.md` (this file)
- `TASKS.md` (execution plan)
- Core implementation under `src/`
- CLI scripts for train/eval
- Minimal smoke validation script or test
