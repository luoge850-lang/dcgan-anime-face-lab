# Source map and freeze status

Freeze date: 2026-08-19

## Included

- Original prototype architecture and dataset helpers.
- Standardized baseline framework.
- Representative Phase 2 discriminator/loss experiments.
- Representative Phase 3 Generator, data-scale, Laplacian, and DiffAugment+EMA experiments.
- Formal CLIP evaluation/control and lambda sweep scripts.
- Completed Phase 7 SDXL controlled-study source scripts and selected evidence.
- Deployment source archive covering ONNX, engine benchmark, PTQ, mixed precision, QAT, service preflight, staged stress, monitoring, and local report generation.
- JSON metrics, CSV logs, selected samples, comparison charts, and key Exp09/Exp11 model artifacts.

## Intentionally excluded

- Raw image datasets and generated FID image dumps.
- Optimizer-only checkpoints and redundant binary artifacts.
- Internal work logs, Word reports, document-build folders, and unrelated root-level files.
- Raw SDXL candidate/production image pools, archives, and large checkpoints.
- TensorRT engines, ONNX binaries, and QAT checkpoints from the deployment workspace.

## Current status interpretation

The active project may continue to produce new experiments. This folder is a snapshot, not a replacement for the active workspace. The v0.5 deployment evidence includes one staged stress run but no long soak run. New work should be recorded in a later snapshot or a clearly dated release so the frozen results remain auditable.
