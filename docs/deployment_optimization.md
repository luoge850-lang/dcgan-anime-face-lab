# Deployment optimization: measured evidence and boundaries

The deployment phase was added after the historical DCGAN training experiments. Its target is the current unconditional Generator, not the discriminator and not every experimental feature tested during training.

## Actual inference graph

The exported Generator is a standard operator graph:

```text
z [B, 128, 1, 1]
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + Tanh
  -> image [B, 3, 64, 64]
```

The independent Haar-wavelet and dynamic-SN probes are not nodes in this current main graph. They should be described as ONNX-compatible replacement probes, not as deployed custom operators.

## Quality and speed

Task 3 provides the cleanest FP32/FP16/INT8 quality baseline. FP16 is effectively lossless under the recorded Standard FID protocol. INT8 improves throughput but increases Standard FID from 29.9911 to 35.3198 and blur rate from 12.0% to 12.5%.

Task 4 restores selected sensitive layers to FP16. The final `net.0 + net.12` strategy reaches Standard FID 31.1776 at approximately 23.97K images/s in the recorded final-confirmation benchmark. This is the strongest archived quality-speed trade-off for the actual graph.

Task 5 QAT uses FakeQuantize and a perceptual/distillation-style objective. It improves over all-INT8 PTQ, but the revised acceptance does not establish superiority over mixed-precision PTQ or prove better hair/eyeliner high-frequency detail.

## Service stress, soak, and dynamic batching

Task 6 has three distinct evidence scopes. The fixed-batch staged run uses a single process, single worker, single TensorRT context, HTTP request concurrency, and engine batch 1. It reaches concurrency 512 with zero failures and no hard crash; P99 rises from 5 ms at concurrency 1 to 1,600 ms at 512, with a soft latency knee around 32. This is a measured capacity range, not a physical crash boundary.

The 60-minute steady soak uses concurrency 16 after a 120-second warmup. It ran for 3,601.6 seconds, served 1,226,890 requests with zero failures, reached P99 63 ms and 340.83 RPS, and showed RSS head-tail change of +3.32% with GPU-memory head-tail change of 0%. This is operational leak-screening evidence under the declared workload, not a proof of zero long-term leaks.

The dynamic-batching study uses a 5 ms queue window and service batch up to 8. Actual batches 2/4/8 were observed; all stages through concurrency 128 passed with zero failures. At concurrency 32, P99 improved from 110 to 90 ms and RPS from 352.35 to 401.73 (+14.01%) versus fixed batch 1. The queue window adds latency at low concurrency, and the study is not a hardware-saturation experiment. The downloaded archive has packaging gaps: not every runtime summary and service log is present, so the snapshot marks this evidence `complete_with_packaging_gaps`.

## Observability and rollout

Stage 7 starts Prometheus, Alertmanager, and Grafana around the service. The recorded validation returned HTTP 200 for health, generation, and metrics; loaded two alert rules; observed the Prometheus target as up; recorded both firing and resolved events; and preserved 37 resource-monitor samples plus a Grafana screenshot. The queue-backlog alert is a controlled simulation used to validate the route, not a real incident or external paging proof.

Stage 8 hot-loads a QAT INT8 candidate as version B while version A (PTQ INT8) continues serving. The PID remained unchanged, target B traffic ratios of 10%, 50%, and 100% stayed within 2.5 percentage points, and rollback returned traffic to A. In the separate 5,000-sample rollout evaluation, B recorded FID 32.0422 versus A 35.5710 and blur 11.62% versus 12.50%, but B P99 was 186.7 ms versus A 98.6 ms. This supports a controlled single-node rollout with an explicit tail-latency trade-off; it is not a multi-replica production guarantee and its sampled FID is not merged into the canonical precision table.

## Evidence paths

- Fixed-batch and soak audit: [`06E`](../03_metrics_and_logs/deployment_optimization/06_Service_Stress/06E/)
- Dynamic batching report: [`06F report`](../03_metrics_and_logs/deployment_optimization/06_Service_Stress/06F/report/)
- Stage 7 observability evidence: [`07 evidence`](../03_metrics_and_logs/deployment_optimization/07/07_MLOps_Observability/evidence/)
- Stage 8 hot-update/A-B evidence: [`08 evidence`](../03_metrics_and_logs/deployment_optimization/08_Model_Hot_Update_AB/evidence/)
- Current project manifest: [`deployment_optimization_current_manifest.json`](../03_metrics_and_logs/deployment_optimization/deployment_optimization_current_manifest.json)
- Script coverage audit: [`SCRIPT_AUDIT.md`](../02_selected_experiments/full_process/deployment_optimization/SCRIPT_AUDIT.md)
- Normalized operational table: [`service_operational_summary_v08.csv`](../03_metrics_and_logs/deployment_optimization/service_operational_summary_v08.csv)
- Canonical service and soak figures: [`Stage 6 figures`](../04_visual_assets/stage_figures/06_服务压测/)
- Latest result-folder figure catalog: [`figure_catalog`](../03_metrics_and_logs/figure_catalog/)

## Reproduction boundary

The measurements were produced in Kaggle with the runtime and engine metadata recorded in the manifests. The public snapshot keeps the scripts, metrics, summaries, and a small raw stress-run archive, but excludes large engine/ONNX/checkpoint binaries. Host RSS and whole-device CUDA memory are different measurement types and must not be compared as one unified memory metric.
