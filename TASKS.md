# DMS2F-HAD Build Tasks

## Phase 0: Planning
- [x] T0.1 Read paper and extract architecture/training/eval requirements.
- [x] T0.2 Inspect upstream reference repository.
- [x] T0.3 Write module PRD.

## Phase 1: Project Skeleton
- [x] T1.1 Create package layout under `src/dms2f_had/`.
- [x] T1.2 Add `pyproject.toml` with runtime/dev dependencies.
- [x] T1.3 Add baseline `README.md` usage docs.

## Phase 2: Data + Patch Pipeline
- [x] T2.1 Implement `.mat` loader with key auto-resolution.
- [x] T2.2 Implement normalization and dataset abstraction.
- [x] T2.3 Implement patch extraction (overlap) and fold reconstruction.
- [x] T2.4 Implement training-time random spatial masking.

## Phase 3: Model Implementation
- [x] T3.1 Implement mixer abstraction (`Mamba` if installed, fallback otherwise).
- [x] T3.2 Implement spatial branch (multi-scale conv + sequence mixer).
- [x] T3.3 Implement spectral grouping utility and spectral branch.
- [x] T3.4 Implement adaptive gated fusion.
- [x] T3.5 Implement SS decoder and full reconstruction model.

## Phase 4: Training and Inference
- [x] T4.1 Implement train loop (MSE + L1 objective).
- [x] T4.2 Implement checkpointing on best AUC.
- [x] T4.3 Implement inference residual-map generation.
- [x] T4.4 Implement ROC/AUC evaluation helper with GT checks.

## Phase 5: CLI + Config
- [x] T5.1 Add config dataclasses/defaults aligned with paper.
- [x] T5.2 Add `train` CLI command.
- [x] T5.3 Add `infer` CLI command.
- [x] T5.4 Add output artifact paths (`checkpoints`, `results`).

## Phase 6: Validation
- [x] T6.1 Add synthetic smoke test for shape/path correctness.
- [ ] T6.2 Run local smoke train/eval.
- [x] T6.3 Document what is validated vs deferred for CUDA server.

## Dependency Notes
- `T2.*` blocks `T4.*`.
- `T3.*` blocks `T4.*`.
- `T4.*` blocks `T6.2`.
- `T5.*` can proceed in parallel after `T3.5`.
