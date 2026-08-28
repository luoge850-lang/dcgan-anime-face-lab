# Next phase: generator export and inference evaluation

This document was originally a future-work plan. The active Kaggle workspace now contains measured results for most of the planned deployment chain. It is retained for traceability; the current public status is summarized in [`docs/deployment_optimization.md`](deployment_optimization.md) and [`deployment_task_status.csv`](../03_metrics_and_logs/deployment_optimization/deployment_task_status.csv).

## Current status

| Area | Status |
|---|---|
| ONNX export/checker | Measured and passed |
| Numerical equivalence and block-level fusion | Measured; whole-generator speed gate not passed |
| ORT/TensorRT/OpenVINO benchmark | Measured |
| FP32/FP16/INT8 PTQ | Measured |
| Layer sensitivity and mixed precision | Measured; final `net.0 + net.12` selected |
| QAT | Measured under revised acceptance; strict high-frequency claim not passed |
| HTTP service preflight | Measured |
| Locust staged stress | Fixed batch 1 measured through concurrency 512 with zero failures; dynamic batching measured through 128 |
| 60-minute soak and long-run memory leak conclusion | 60-minute declared soak measured; no universal leak proof or physical saturation boundary |

## Goal

Export the Exp11 EMA Generator for inference, validate numerical equivalence, and compare deployment cost across PyTorch and ONNX Runtime. Image quality and runtime performance are separate evaluation tracks.

## Current generator boundary

The inference path is a 64x64 unconditional generator:

- latent input: 128 x 1 x 1;
- five ConvTranspose2d upsampling stages;
- BatchNorm and ReLU in the first four stages;
- Tanh output in [-1, 1];
- no training-only attention, wavelet, FFT, or discriminator path in the final generator.

The generator is therefore a reasonable export target. The discriminator is not part of the first deployment milestone.

## Proposed milestones

### 1. Baseline export

Export the EMA Generator to ONNX with a fixed batch-1 input, then validate:

- ONNX checker succeeds;
- ONNX Runtime loads the graph;
- output shape is batch x 3 x 64 x 64;
- PyTorch and ONNX outputs use the same Tanh range;
- maximum absolute error and MSE are reported on a fixed set of noise vectors.

### 2. Batch and latency benchmark

Benchmark PyTorch and ONNX Runtime after warm-up at batch sizes 1, 4, 8, 16, 32, and 64. Report median and P95 latency, throughput, peak memory, device, software versions, and whether CUDA is enabled.

Do not report a speedup without recording the exact runtime, provider, precision, warm-up count, timed iterations, and synchronization method.

### 3. ConvTranspose plus BatchNorm folding

Create a separate fused export by folding BatchNorm into ConvTranspose weights and biases. Validate the fused model against the unfused PyTorch model on many fixed noise vectors before measuring speed. Keep the unfused export as the reference.

For ConvTranspose, the channel axis differs from Conv2d. A folding implementation must verify the weight shape and output numerically; a smaller file is not evidence of a correct fusion.

### 4. Precision study

Treat FP16 as the first practical comparison. Treat INT8 as a quality-risk experiment requiring a calibration set and a full image-quality report. Protect the Tanh output and compare FID, coverage, blur rate, edge density, and qualitative samples in addition to latency.

## Service stress boundary

The fixed-batch run used HTTP request concurrency with TensorRT engine batch 1 and collected P99, RPS, failures, GPU memory, GPU SM, and service RSS at a 5-second interval through concurrency 512. A separate 60-minute steady soak at concurrency 16 served 1,226,890 requests with zero failures and stable head-tail resource checks. A dynamic-batching study used a 5 ms queue window and observed service batches up to 8, improving RPS at medium/high concurrency. These are workload-scoped operational results; future work is limited to closing the dynamic-batching packaging gaps, testing queue drain behavior, and measuring a genuine hardware saturation boundary rather than rerunning all historical training experiments.

## Required artifacts

    deployment/
      01_export/
        export_generator.py
        validate_onnx.py
      02_benchmark/
        benchmark_runtime.py
      03_fusion/
        fold_convtranspose_bn.py
      04_precision/
        calibration_manifest.csv
      results/
        numerical_equivalence.json
        latency_benchmark.csv
        quality_comparison.csv

The deployment phase should be added to the public repository only after its outputs have a code version, model checksum, runtime metadata, and a clear distinction between planned and measured values.
