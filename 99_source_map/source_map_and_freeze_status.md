# Source map and freeze status

Freeze date: 2026-08-19

Freeze release: v0.6-stage-freeze

## Included

- Original prototype architecture and dataset helpers.
- Standardized baseline framework.
- Representative Phase 2 discriminator/loss experiments.
- Representative Phase 3 Generator, data-scale, Laplacian, and DiffAugment+EMA experiments.
- Formal CLIP evaluation/control and lambda sweep scripts.
- Completed Phase 7 SDXL controlled-study source scripts and selected evidence.
- Deployment source archive covering ONNX, engine benchmark, PTQ, mixed precision, QAT, service preflight, staged stress, monitoring, and local report generation.
- JSON metrics, CSV logs, selected samples, comparison charts, and key Exp09/Exp11 model artifacts.
- Stage-organized Chinese single-metric figures rebuilt from the scanned `dcgan_lab/results` inputs: 27 result-backed charts across six stages.
- A stage map linking each regenerated chart to its source result scope.

## Intentionally excluded

- Raw image datasets and generated FID image dumps.
- Optimizer-only checkpoints and redundant binary artifacts.
- Internal work logs, Word reports, document-build folders, and unrelated root-level files.
- Raw SDXL candidate/production image pools, archives, and large checkpoints.
- TensorRT engines, ONNX binaries, and QAT checkpoints from the deployment workspace.
- Representative loss-trajectory figures 09–15, excluded per owner note; no deleted loss chart is recreated.

## Current status interpretation

The active project may continue to produce new experiments. This folder is a snapshot, not a replacement for the active workspace. The v0.6 freeze uses 27 result-backed single-metric charts; the current staged service evidence covers P99/RPS, while the source-folder soak/resource charts remain provenance-only because corresponding current raw tables were not available. New work should be recorded in a later snapshot or a clearly dated release so the frozen results remain auditable.
