# Deployment optimization and serving experiments

This directory contains the selected source entry points for the deployment track. The experiments were executed in Kaggle, not as a local one-command package. Every stage is tied to a result manifest or is explicitly marked as source-only.

## Current status

| Stage | Scope | Public status |
|---|---|---|
| 01 | ONNX export, graph inspection, BN folding and fusion probes | Partial: checker, parity, and block-level evidence; no stable whole-generator speedup |
| 02 | ONNX Runtime, TensorRT, OpenVINO and PyTorch profiling | Complete |
| 03 | FP32/FP16/INT8 PTQ quality-speed baseline | Complete |
| 04 | Per-layer sensitivity and mixed precision | Complete: `net.0 + net.12` retained in FP16 |
| 05 | Fake-quant fine-tuning and QAT engine evaluation | Complete under revised global quality gate |
| 06 | Service smoke, staged stress, soak and dynamic batching | Operational evidence; dynamic-batching archive has packaging gaps |
| 07 | Prometheus/Grafana/Alertmanager and controlled alert lifecycle | Complete for one Kaggle single-node run |
| 08 | Same-PID hot loading, A/B routing, quality check and rollback | Complete for one Kaggle single-node run |

Canonical status and evidence index:

- [`SCRIPT_AUDIT.md`](SCRIPT_AUDIT.md) — script-to-output coverage and recommended execution order.
- [`deployment_optimization_current_manifest.json`](../../../03_metrics_and_logs/deployment_optimization/deployment_optimization_current_manifest.json) — current project-level status.
- [`EXPERIMENT_INDEX.md`](../../../03_metrics_and_logs/deployment_optimization/EXPERIMENT_INDEX.md) — normalized deployment table.
- [`experiment_coverage_audit_2026-09-03.md`](../../../docs/experiment_coverage_audit_2026-09-03.md) — measured/source-only/partial boundary.

## Locked deployment contract

Before running any entry point, lock the checkpoint, matching Generator class, latent shape `[batch, 128, 1, 1]`, output shape `[batch, 3, 64, 64]`, TensorRT/GPU compatibility, and the exact quality protocol. Checkpoints, engines, datasets, and Kaggle mounts are intentionally not included in this public snapshot.

## Recommended order

```text
01A → 01B/01C/01D (optional) → 02A/02B/02C/02D/02F → 02E
→ 03A → 03B/03C → 03D → 03E
→ 04A → 04B → 04C → 05A → 05B
→ 06A → 06BC → 06D → 06E → 06F
→ 07 → 08
```

The `02E`, `03E`, `06E`, and `06F_dynamic_batch_report.py` files merge existing evidence; they do not represent additional model-training runs. Stages 07 and 08 are single-cell Kaggle entry points and generate their monitoring/rollout configuration at runtime.

## Stage 07 entry point

`07_MLOps_Observability/07_NOTEBOOK_ALL_IN_ONE.py` starts the service-side monitoring stack, validates `/health`, `/generate`, and `/metrics`, samples GPU/RSS resources, checks two alert rules, and preserves firing/resolved events plus a Grafana screenshot. Queue backlog is deliberately simulated to validate the alert route.

## Stage 08 entry point

`08_Model_Hot_Update_AB/08_NOTEBOOK_ALL_IN_ONE.py` loads a PTQ INT8 engine as A, hot-loads a QAT INT8 engine as B without changing the PID, exercises 10%/50%/100% B traffic, measures version-level latency and sampled quality, and rolls back to A. The recorded candidate B had better sampled FID but a materially higher P99, so the result is a rollout-control validation rather than an unconditional replacement recommendation.

## Public packaging rule

Keep new Kaggle code and raw outputs in the active `dcgan_lab` workspace. After a stage is genuinely complete, copy only the entry script, a small manifest, normalized CSV/JSON evidence, and readable figures into a new dated public snapshot commit. Do not copy `.engine`, `.onnx`, checkpoints, datasets, or uncontrolled logs.

