# DCGAN Anime-Face Lab — handoff entry point

This folder is the first place to look when handing the project to a mentor, interviewer, teammate, or your future self. It explains what is in the public snapshot, where the evidence lives, what has actually been measured, and how to continue the project without damaging the audit trail.

## Project identity

| Field | Value |
|---|---|
| Project | DCGAN Anime-Face Lab |
| Description | Unconditional 64 × 64 anime-face generation and inference engineering study |
| Runtime | Kaggle GPU; deployment evidence mainly on Tesla T4 |
| Public release | `v0.9-current-state` |
| Current source of truth | Active Kaggle workspace `dcgan_lab` |
| Public package role | Curated evidence archive, interview showcase, and engineering handoff |
| Reproduction claim | Inspectable and auditable; not a local one-command retraining release |

## Read in this order

1. [Root README](../README.md) — the short public story, visual route, and headline results.
2. [Project Engineering Report](../docs/PROJECT_ENGINEERING_REPORT_2026-09-03.md) — the complete internship narrative.
3. [Experiment Coverage Audit](../docs/experiment_coverage_audit_2026-09-03.md) — measured versus partial/source-only claims.
4. [File and Evidence Index](file_and_evidence_index.md) — where each script family and result family belongs.
5. [Deployment Script Audit](../02_selected_experiments/full_process/deployment_optimization/SCRIPT_AUDIT.md) — deployment execution order and outputs.
6. [Project Inventory](../03_metrics_and_logs/project_inventory_2026-09-03.json) — machine-readable scan of the active workspace and public package.

## What the project contains

### Research track: `R01–R05`

- `R01_budget_preprocessing`: epoch budget and image preprocessing search;
- `R02_gd_module_ablation`: plain no-added-module control and G/D stabilization study;
- `R03_generator_recipe`: generator width, data scale, DiffAugment, and EMA;
- `R04_clip_continuation`: no-CLIP control and frozen CLIP MMD² weight sweep;
- `R05_sdxl_data_side_study`: SDXL candidate filtering and replacement-ratio ablation.

### Deployment track: `D01–D08`

- `D01_onnx_fusion`: export, checker, graph probes, and BN-fold equivalence;
- `D02_engine_benchmark`: ORT, TensorRT, OpenVINO, profiler, and bottleneck mapping;
- `D03_ptq_baseline`: FP32/FP16/INT8 PTQ quality-speed baseline;
- `D04_mixed_precision`: layer sensitivity and selected FP16 retention;
- `D05_qat`: fake-quant fine-tuning and QAT evaluation;
- `D06_serving`: HTTP smoke, stress, soak, and dynamic batching;
- `D07_observability`: Prometheus, Grafana, Alertmanager, and resource monitoring;
- `D08_hot_update_ab`: same-PID hot loading, A/B routing, quality gate, and rollback.

## Current status at handoff

| Area | Status | What can be claimed |
|---|---|---|
| Core DCGAN experiments | Measured, mostly single seed | A staged ablation and decision record with positive and negative results |
| Historical best candidate | Measured | Legacy FID 38.88 for a multi-factor Width ×3 + 20K + DiffAugment + EMA recipe |
| CLIP branch | Measured continuation | C1 is the FID winner; C4 is the CLIP MMD² winner |
| SDXL branch | Measured negative study | Tested replacement ratios degraded FID/Coverage; branch was not promoted |
| ONNX/fusion | Partial | Export/checker and numerical probes passed; whole-graph speedup not proven |
| Precision study | Measured | FP16 near-lossless; INT8 faster with quality cost; mixed precision recovered quality |
| Service and operations | Operational evidence | Stress/soak, monitoring, hot update, A/B, and rollback were validated on one node |

## Non-negotiable interpretation rules

- Do not rank Legacy FID and deployment Standard FID in one leaderboard.
- Do not call the Phase 2 control “no preprocessing”; it is a plain architectural control with the selected input policy.
- Do not attribute Exp11 FID 38.88 to DiffAugment or EMA alone.
- Do not present source-only `12_G_AA_ADA_EMA_21K.py` as a completed result.
- Do not call concurrency 512 the physical GPU limit; it is the highest tested successful point.
- Do not call the Stage 7 queue alert a real incident; it is a controlled alert-path simulation.
- Disclose that Stage 8 candidate B had lower sampled FID but approximately 1.9× the A P99 tail.
- Use “contributed to”, “implemented”, “evaluated”, “audited”, or “documented” unless individual ownership is explicitly confirmed.

## Continue the project safely

1. Keep new Kaggle notebooks, checkpoints, datasets, and raw outputs in the active `dcgan_lab` workspace.
2. When a run is complete, copy only the entry script, a small manifest, normalized CSV/JSON evidence, and readable figures.
3. Give every new evidence package a date and a new comparison scope if the dataset, checkpoint, hardware, or metric protocol changes.
4. Add a new commit and tag; do not silently overwrite an earlier metric or freeze.
5. Update the [coverage audit](../docs/experiment_coverage_audit_2026-09-03.md) and [project inventory](../03_metrics_and_logs/project_inventory_2026-09-03.json) together.

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -v
```

This validates JSON evidence, core metric tables, stage figures, Stage 7/8 packages, the engineering report, and the inventory.

