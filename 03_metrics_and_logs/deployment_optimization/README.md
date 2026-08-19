# Deployment optimization evidence

This directory contains a curated, public-safe index of the deployment experiments found in the active Kaggle workspace on 2026-08-19. It does not include `.engine`, `.onnx`, or checkpoint binaries. The copied JSON/CSV files retain the measured evidence and runtime metadata needed to audit the claims.

## Current evidence status

| Task | Status | Defensible conclusion |
|---|---|---|
| ONNX and fusion | Partial | Export, checker, numerical equivalence, and block-level fusion were validated; whole-generator speedup was not passed. |
| Engine benchmark | Complete | ORT CPU, TensorRT GPU, and OpenVINO CPU were benchmarked across FP32/FP16 and dynamic batches. |
| PTQ baseline | Complete | FP16 was near-lossless; INT8 was faster but degraded Standard FID and blur rate. |
| Sensitivity and mixed precision | Complete | Retaining `net.0` and `net.12` in FP16 recovered quality while preserving high throughput. |
| QAT | Revised acceptance | QAT improved over all-INT8 PTQ, but did not beat the selected mixed-precision PTQ baseline on the archived quality-speed comparison. |
| Service stress | Staged run complete | Locust reached concurrency 128 with zero failures and 5-second GPU/RSS monitoring; no hard crash was observed. A 30-minute soak result is not included. |

## Quantization headline

The Task 3 Standard FID comparison is:

| Precision | Standard FID | Blur rate | Mean latency | Throughput |
|---|---:|---:|---:|---:|
| FP32 | 29.9911 | 12.0% | 14.6969 ms/batch | 4,354.7 images/s |
| FP16 | 29.9941 | 12.0% | 4.1304 ms/batch | 15,495.0 images/s |
| INT8 PTQ | 35.3198 | 12.5% | 3.0323 ms/batch | 21,106.1 images/s |

Task 4 final confirmation selected `net.0 + net.12` FP16 retention: Standard FID 31.1776, blur rate 12.1%, and 23,971.7 images/s in its recorded benchmark scope.

The full normalized tables are [`deployment_quantization_summary.csv`](deployment_quantization_summary.csv), [`deployment_engine_summary.csv`](deployment_engine_summary.csv), and [`deployment_task_status.csv`](deployment_task_status.csv).

## Service stress headline

The archived staged run used a QAT hybrid TensorRT engine on a Tesla T4, with concurrency stages 1, 2, 4, 8, 16, 32, 48, 64, 80, 96, and 128. Every stage returned zero failed requests. P99 increased from 5 ms at concurrency 1 to 490 ms at concurrency 128, while RPS stayed near 335–342. Peak recorded GPU memory was 677.2 MB and service RSS was 1,077.8 MB across 100 samples at a 5-second interval.

This identifies a latency-pressure region, not a hardware crash point. The run was staged, not a long soak test; therefore the snapshot does not claim long-term memory-leak absence.

See [`06_Service_Stress/service_stress_summary.csv`](06_Service_Stress/service_stress_summary.csv), [`06_Service_Stress/service_monitor_summary.json`](06_Service_Stress/service_monitor_summary.json), and the extracted raw run under [`06_Service_Stress/raw_run_20260819_014504`](06_Service_Stress/raw_run_20260819_014504).
