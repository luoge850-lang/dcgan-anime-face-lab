# Deployment optimization evidence

This directory contains a curated, public-safe index of the deployment experiments found in the active Kaggle workspace through 2026-08-27. It does not include `.engine`, `.onnx`, or checkpoint binaries. The copied JSON/CSV files retain the measured evidence and runtime metadata needed to audit the claims.

## Current evidence status

| Task | Status | Defensible conclusion |
|---|---|---|
| ONNX and fusion | Partial | Export, checker, numerical equivalence, and block-level fusion were validated; whole-generator speedup was not passed. |
| Engine benchmark | Complete | ORT CPU, TensorRT GPU, and OpenVINO CPU were benchmarked across FP32/FP16 and dynamic batches. |
| PTQ baseline | Complete | FP16 was near-lossless; INT8 was faster but degraded Standard FID and blur rate. |
| Sensitivity and mixed precision | Complete | Retaining `net.0` and `net.12` in FP16 recovered quality while preserving high throughput. |
| QAT | Revised acceptance | QAT improved over all-INT8 PTQ, but did not beat the selected mixed-precision PTQ baseline on the archived quality-speed comparison. |
| Service stress | Operational and soak pass; strict physical-limit status incomplete | Fixed batch 1 reached concurrency 512 with zero failures; a 60-minute steady soak served 1,226,890 requests with zero failures; no physical crash/saturation boundary was observed. |
| Dynamic batching | Complete with packaging gaps | Batch 2/4/8 observed; all stages through concurrency 128 passed; +14.01% RPS at concurrency 32 versus fixed batch 1; downloaded archive lacks some runtime summary/log files. |

## Quantization headline

The Task 3 Standard FID comparison is:

| Precision | Standard FID | Blur rate | Mean latency | Throughput |
|---|---:|---:|---:|---:|
| FP32 | 29.9911 | 12.0% | 14.6969 ms/batch | 4,354.7 images/s |
| FP16 | 29.9941 | 12.0% | 4.1304 ms/batch | 15,495.0 images/s |
| INT8 PTQ | 35.3198 | 12.5% | 3.0323 ms/batch | 21,106.1 images/s |

Task 4 final confirmation selected `net.0 + net.12` FP16 retention: Standard FID 31.1776, blur rate 12.1%, and 23,971.7 images/s in its recorded benchmark scope.

The full normalized tables are [`deployment_quantization_summary.csv`](deployment_quantization_summary.csv), [`deployment_engine_summary.csv`](deployment_engine_summary.csv), [`deployment_task_status.csv`](deployment_task_status.csv), and [`service_operational_summary_v08.csv`](service_operational_summary_v08.csv).

## Service stress headline

The fixed-batch control used a QAT hybrid TensorRT engine on a Tesla T4. Every stage from concurrency 1 through 512 returned zero failed requests; P99 increased from 5 ms to 1,600 ms and the soft latency knee was around 32. Peak GPU memory was 677.2 MB / 19% SM, and service RSS increased from approximately 1,070.9 to 1,114.8 MB.

The 60-minute steady soak at concurrency 16 ran for 3,601.6 seconds, served 1,226,890 requests with zero failures, and reached P99 63 ms / 340.83 RPS. Head-tail resource checks were RSS +3.32% and GPU memory 0%. These are workload-scoped operational results, not a physical crash point or a universal leak proof.

The dynamic-batching comparison used a 5 ms wait window and maximum service batch 8. At concurrency 32, P99 was 90 ms versus 110 ms and RPS was 401.73 versus 352.35 (+14.01%) relative to fixed batch 1. Low concurrency pays the queue-window cost; the public status is `complete_with_packaging_gaps` because the downloaded archive does not contain every runtime summary and service log.

See the current [`06E audit`](06_Service_Stress/06E/), [`06F report`](06_Service_Stress/06F/report/), [`06E/06F figures`](../../04_visual_assets/stage_figures/06_服务压测/), and the prior raw staged archive under [`raw_run_20260819_014504`](06_Service_Stress/raw_run_20260819_014504).
