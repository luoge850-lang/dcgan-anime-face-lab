# Deployment optimization evidence index

This directory contains the public-safe result tables and manifests for the deployment track. The source experiments ran on a Kaggle Tesla T4. Engine, ONNX, checkpoint, dataset, and large service-log binaries are excluded.

## Current evidence map

| Stage | Status | Evidence-backed conclusion |
|---|---|---|
| 01 ONNX/fusion | Partial | ONNX export/checker, numerical parity, and block-level folding were validated; whole-generator speed gain was not established. |
| 02 engine benchmark | Complete | ORT CPU, TensorRT GPU, OpenVINO CPU, FP32/FP16, and dynamic-batch profiles were recorded. |
| 03 PTQ | Complete | FP16 was near-lossless; INT8 improved speed but degraded Standard FID and blur rate. |
| 04 mixed precision | Complete | Keeping `net.0 + net.12` in FP16 recovered quality while retaining high throughput. |
| 05 QAT | Revised complete | QAT improved over full INT8 PTQ, but did not beat the selected mixed-precision PTQ trade-off. |
| 06 service | Operational evidence | Fixed batch passed through tested concurrency 512; a 60-minute soak had 0 failures; no physical crash point was observed. |
| 06F dynamic batch | Complete with packaging gaps | Batch 2/4/8 was observed and RPS increased 14.01% at concurrency 32; some runtime logs are absent from the downloaded archive. |
| 07 observability | Complete, single node | Monitoring endpoints, Prometheus target, two rules, controlled firing/resolution, resource samples, and Grafana screenshot are present. |
| 08 hot update/A-B | Complete, single node | Same-PID candidate load, traffic split, sampled quality comparison, and rollback are present; B P99 was 186.7 ms versus A 98.6 ms. |

## Quality-speed reference

| Precision / strategy | Standard FID | Blur rate | Throughput |
|---|---:|---:|---:|
| FP32 | 29.9911 | 12.0% | 4,354.7 images/s |
| FP16 | 29.9941 | 12.0% | 15,495.0 images/s |
| INT8 PTQ | 35.3198 | 12.5% | 21,106.1 images/s |
| Mixed PTQ (`net.0 + net.12` FP16) | 31.1776 | 12.1% | 23,971.7 images/s |
| QAT INT8 | 31.6456 | 11.3% | 19,092.7 images/s |

These rows come from the canonical Stage 3–5 deployment protocol. Stage 8 A/B quality values are a separate sampled rollout instance and are intentionally kept in its own folder.

## Stage 7 evidence

The curated folder [`07/07_MLOps_Observability/evidence/`](07/07_MLOps_Observability/evidence/) contains the validation summary, load/metric snapshots, 5-second resource monitor, alert webhook events, monitoring configuration, Grafana screenshot, and evidence ZIP. The queue alert is a controlled simulation; the evidence does not claim external email/paging delivery.

## Stage 8 evidence

The curated folder [`08_Model_Hot_Update_AB/evidence/`](08_Model_Hot_Update_AB/evidence/) contains:

- `08_validation_summary.json` — status, health/PID checks, split accuracy, quality gate, and rollback result;
- `08_traffic_split.csv` — target versus observed B ratios;
- `08_latency_by_version.csv` — version-level p50/p95/p99 and success rate;
- `08_fid_sample.csv` — sampled FID, blur rate, Laplacian diagnostic, and sample count;
- `08_update_events.jsonl` and `08_report.md` — update lifecycle and interpretation.

The sampled result is A FID 35.5710 / B FID 32.0422 and A P99 98.6 ms / B P99 186.7 ms. It supports controlled rollout with an explicit tail-latency trade-off, not a claim that B is universally better.

## Canonical references

- [`deployment_optimization_current_manifest.json`](deployment_optimization_current_manifest.json)
- [`deployment_task_status.csv`](deployment_task_status.csv)
- [`deployment_quantization_summary.csv`](deployment_quantization_summary.csv)
- [`deployment_engine_summary.csv`](deployment_engine_summary.csv)
- [`service_operational_summary_v08.csv`](service_operational_summary_v08.csv)
- [`EXPERIMENT_INDEX.md`](EXPERIMENT_INDEX.md)
- [`docs/experiment_coverage_audit_2026-09-03.md`](../../../docs/experiment_coverage_audit_2026-09-03.md)

