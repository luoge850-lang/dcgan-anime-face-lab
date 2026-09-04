# Source map and freeze status

Freeze date: 2026-09-04

Freeze release: v0.10.1-handoff-2026-09-04

Previous evidence freeze: v0.9-current-state

## Included

- Original prototype architecture and dataset helpers.
- Standardized baseline framework.
- Representative Phase 2 discriminator/loss experiments.
- Representative Phase 3 Generator, data-scale, Laplacian, and DiffAugment+EMA experiments.
- Formal CLIP evaluation/control and lambda sweep scripts.
- Completed Phase 7 SDXL controlled-study source scripts and selected evidence.
- Deployment source archive covering ONNX, engine benchmark, PTQ, mixed precision, QAT, service preflight, staged stress, monitoring, and local report generation.
- JSON metrics, CSV logs, selected samples, comparison charts, and key Exp09/Exp11 model artifacts.
- Stage-organized Chinese single-metric figures rebuilt from the scanned `dcgan_lab/results` inputs: 37 SVG charts plus one preserved dashboard screenshot across eight stages.
- Latest source-backed Stage 7/8 standalone SVGs and their sanitized provenance manifest; redundant PNG renders are excluded from the public package.
- Latest full result-folder figure catalog copied under `03_metrics_and_logs/figure_catalog/`.
- Newly verified 06E fixed-batch/soak audit, 06F dynamic-batching report, Stage 7 observability evidence, Stage 8 hot-update/A-B evidence, figures, summaries, and manifests.
- A stage map linking each regenerated chart to its source result scope.
- A research-and-engineering showcase README with the problem definition, route map, baseline architecture, stage-by-stage evidence, frozen showcase recipe, evaluation protocol, deployment trade-offs, limitations, and interview summary.
- A complete project engineering report and machine-readable inventory covering the active source boundary, public package, naming aliases, stage status, metric protocols, and update policy.
- A dedicated `00_HANDOFF/` entry point with reading order, source-to-evidence mapping, current status, and handoff manifest.

## Intentionally excluded

- Raw image datasets and generated FID image dumps.
- Optimizer-only checkpoints and redundant binary artifacts.
- Internal work logs, Word reports, document-build folders, and unrelated root-level files.
- Raw SDXL candidate/production image pools, archives, and large checkpoints.
- TensorRT engines, ONNX binaries, and QAT checkpoints from the deployment workspace.
- Representative loss-trajectory figures 09–15, excluded per owner note; no deleted loss chart is recreated.
- Redundant PNG renders of the latest Stage 7/8 SVGs; the source-backed SVGs are the canonical public figures.

## Current status interpretation

The active project may continue to produce new experiments. This folder is a dated handoff, not a replacement for the active workspace. The v0.10 handoff preserves v0.6–v0.9 evidence, updates the active-workspace inventory, and adds the latest source-backed Stage 7/8 figure outputs. Fixed batch 1–512, the 60-minute soak, dynamic batching, observability, and hot update A/B are separate operational scopes; no hard crash or physical GPU-saturation boundary was observed. Dynamic batching remains marked `complete_with_packaging_gaps`. The latest G-strengthening script `12_G_AA_ADA_EMA_21K.py` remains source-only without a matching result directory and is not presented as a measured winner. New work should be recorded in a later snapshot or dated release so the frozen results remain auditable.
