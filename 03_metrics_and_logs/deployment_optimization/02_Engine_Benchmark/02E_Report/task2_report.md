# Task 2 Engine Benchmark Report

## Status

- Required evidence missing: none.
- 30/30 benchmark rows have `status=ok`.
- Each engine covers FP32/FP16 and Batch 1/4/8/16/32.
- ORT raw profile JSON was not retained locally; `ort_operator_summary.csv` is retained as the ORT operator evidence.

## Output map

- `task2_engine_comparison.csv`: complete 30-row engine comparison.
- `task2_operator_evidence.csv`: 331 normalized operator/layer evidence rows.
- `task2_top3_operators.csv`: Top-3 ranking for each retained profile source.
- `task2_manifest.json`: consolidated metadata and coverage audit.

## End-to-end benchmark

| Engine | Precision | Batch | Mean ms | P50 ms | P95 ms | Throughput img/s | Memory |
|---|---|---:|---:|---:|---:|---:|---|
| ONNX Runtime CPU | FP32 | 1 | 3.8029 | 3.7653 | 4.2683 | 262.96 | 228.875 MB RSS |
| ONNX Runtime CPU | FP32 | 4 | 14.3215 | 14.2228 | 15.6067 | 279.30 | 287.961 MB RSS |
| ONNX Runtime CPU | FP32 | 8 | 28.3896 | 28.0181 | 31.2555 | 281.79 | 292.566 MB RSS |
| ONNX Runtime CPU | FP32 | 16 | 55.4039 | 54.2687 | 63.2983 | 288.79 | 300.742 MB RSS |
| ONNX Runtime CPU | FP32 | 32 | 109.5254 | 109.0000 | 120.9659 | 292.17 | 329.043 MB RSS |
| ONNX Runtime CPU | FP16 | 1 | 3.8687 | 3.7726 | 4.6778 | 258.49 | 290.590 MB RSS |
| ONNX Runtime CPU | FP16 | 4 | 14.5073 | 13.9043 | 18.9889 | 275.72 | 282.563 MB RSS |
| ONNX Runtime CPU | FP16 | 8 | 29.0402 | 28.2835 | 33.3488 | 275.48 | 313.836 MB RSS |
| ONNX Runtime CPU | FP16 | 16 | 55.2286 | 54.8281 | 60.1473 | 289.70 | 320.969 MB RSS |
| ONNX Runtime CPU | FP16 | 32 | 109.2069 | 107.5286 | 119.8241 | 293.02 | 347.219 MB RSS |
| TensorRT CUDA | FP32 | 1 | 0.6977 | 0.6786 | 0.7044 | 1433.22 | 236.875 MB device snapshot |
| TensorRT CUDA | FP32 | 4 | 1.0929 | 1.0921 | 1.1159 | 3660.01 | 308.875 MB device snapshot |
| TensorRT CUDA | FP32 | 8 | 1.7575 | 1.7517 | 1.8712 | 4551.86 | 308.875 MB device snapshot |
| TensorRT CUDA | FP32 | 16 | 3.2351 | 3.2258 | 3.3102 | 4945.68 | 308.875 MB device snapshot |
| TensorRT CUDA | FP32 | 32 | 6.0439 | 6.0562 | 6.2086 | 5294.59 | 328.875 MB device snapshot |
| TensorRT CUDA | FP16 | 1 | 0.2771 | 0.2743 | 0.2950 | 3608.41 | 204.875 MB device snapshot |
| TensorRT CUDA | FP16 | 4 | 0.3338 | 0.3259 | 0.3919 | 11982.76 | 220.875 MB device snapshot |
| TensorRT CUDA | FP16 | 8 | 0.4377 | 0.4375 | 0.4625 | 18275.69 | 220.875 MB device snapshot |
| TensorRT CUDA | FP16 | 16 | 0.7432 | 0.7482 | 0.7680 | 21528.25 | 220.875 MB device snapshot |
| TensorRT CUDA | FP16 | 32 | 1.4004 | 1.4073 | 1.4284 | 22851.41 | 220.875 MB device snapshot |
| OpenVINO CPU | FP32 | 1 | 4.1654 | 4.1254 | 4.5211 | 240.07 | 1831.230 MB RSS |
| OpenVINO CPU | FP32 | 4 | 14.5298 | 14.5147 | 15.4542 | 275.30 | 1832.586 MB RSS |
| OpenVINO CPU | FP32 | 8 | 28.3816 | 28.0554 | 32.9674 | 281.87 | 1833.500 MB RSS |
| OpenVINO CPU | FP32 | 16 | 50.8390 | 49.7813 | 57.3120 | 314.72 | 1846.414 MB RSS |
| OpenVINO CPU | FP32 | 32 | 101.9492 | 101.6529 | 112.0650 | 313.88 | 1871.332 MB RSS |
| OpenVINO CPU | FP16 | 1 | 4.1187 | 4.1117 | 4.4483 | 242.80 | 1675.277 MB RSS |
| OpenVINO CPU | FP16 | 4 | 13.5403 | 13.5291 | 14.4880 | 295.42 | 1678.852 MB RSS |
| OpenVINO CPU | FP16 | 8 | 25.4194 | 25.2737 | 27.5104 | 314.72 | 1685.285 MB RSS |
| OpenVINO CPU | FP16 | 16 | 50.4288 | 49.3361 | 55.8632 | 317.28 | 1699.609 MB RSS |
| OpenVINO CPU | FP16 | 32 | 102.3778 | 103.5808 | 110.5097 | 312.57 | 1723.773 MB RSS |

## Observations

- TensorRT FP16 is the fastest recorded configuration: Batch 1 end-to-end latency is 0.2771 ms and Batch 32 throughput is 22,851.41 images/s.
- ONNX Runtime FP16 on CPU does not improve mean latency over FP32; the two are effectively similar within this run.
- OpenVINO FP16 improves latency at Batch 4/8, but not at Batch 16/32; CPU FP16 acceleration is workload- and backend-dependent.
- TensorRT memory values are whole-device CUDA snapshots, not isolated TensorRT engine allocations. ORT and OpenVINO values are host RSS.

## Top-3 operator evidence

| Source | Rank | Operator/layer | Total ms | Evidence |
|---|---:|---|---:|---|
| ORT engine profile | 1 | ConvTranspose | 48931.811000 | `ort_operator_summary.csv` |
| ORT engine profile | 2 | BatchNormalization | 1554.198000 | `ort_operator_summary.csv` |
| ORT engine profile | 3 | Relu | 500.272000 | `ort_operator_summary.csv` |
| PyTorch profiler | 1 | aten::cudnn_convolution_transpose | 76.069563 | `torch_operator_summary.csv` |
| PyTorch profiler | 2 | aten::cudnn_batch_norm | 1.813389 | `torch_operator_summary.csv` |
| PyTorch profiler | 3 | aten::add_ | 1.584881 | `torch_operator_summary.csv` |
| PyTorch layer profiler | 1 | Generator.net.3.ConvTranspose2d | 29.571780 | `layer_operator_summary.csv` |
| PyTorch layer profiler | 2 | Generator.net.9.ConvTranspose2d | 18.066941 | `layer_operator_summary.csv` |
| PyTorch layer profiler | 3 | Generator.net.6.ConvTranspose2d | 17.201705 | `layer_operator_summary.csv` |
| TensorRT IProfiler | 1 | net.3 ConvTranspose + BatchNorm + ReLU | 3.832352 | `tensorrt_layer_profile.csv` |
| TensorRT IProfiler | 2 | net.0 ConvTranspose + BatchNorm + ReLU | 3.504736 | `tensorrt_layer_profile.csv` |
| TensorRT IProfiler | 3 | net.9 ConvTranspose + BatchNorm + ReLU | 3.368928 | `tensorrt_layer_profile.csv` |
| OpenVINO profiling | 1 | Multiply_3796 / ConvolutionBackpropData | 37.769000 | `openvino_operator_profile.csv` |
| OpenVINO profiling | 2 | Multiply_9823 / ConvolutionBackpropData | 35.069000 | `openvino_operator_profile.csv` |
| OpenVINO profiling | 3 | Multiply_9842 / ConvolutionBackpropData | 18.052000 | `openvino_operator_profile.csv` |

The cross-engine common bottleneck is the transposed-convolution/up-sampling path. The PyTorch layer profile identifies `Generator.net.3`, `net.9` and `net.6` as the three slowest concrete layers. TensorRT fuses each ConvTranspose with the following BatchNorm and ReLU, while OpenVINO exposes the corresponding convolution-backpropagation and multiply/reorder work under generated node names.

## Graph-level optimization actions

- Prioritize ConvTranspose layers: compare TensorRT tactics and evaluate a controlled upsample-plus-convolution ablation.
- Fold BatchNorm at inference using the existing 01C output, then verify numerical equivalence.
- Keep TensorRT ConvTranspose+BatchNorm+ReLU fusion and avoid host/device copies.
- Inspect OpenVINO ConvolutionBackpropData and Multiply/Reorder nodes; reduce layout conversions and memory-format changes.
- Do not count profiler wrappers, CUDA synchronization, or low-level duplicate CUDA kernels as independent model operators.

## Memory and evidence boundary

- ORT/OpenVINO memory fields are host RSS.
- TensorRT `cuda_peak_used_snapshot_mb` is a whole-device CUDA snapshot; `peak_allocated_mb` is only a PyTorch allocator diagnostic.
- `torch_trace.json` and `layer_profiler_trace.json` are retained Chrome Trace evidence.
- The final completion decision is recorded in `task2_manifest.json`.
