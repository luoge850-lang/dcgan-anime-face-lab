# Task 2：三引擎部署、动态 Batch 与瓶颈定位

这个目录是一条实验流水线：

```text
01A generator_fp32_raw.onnx
        |
        +--> 02A ONNX Runtime  --------+
        +--> 02B TensorRT GPU ---------+--> 02E 汇总、Top-3、报告
        +--> 02C OpenVINO CPU ---------+
        |
        +--> 02D torch.profiler Chrome Trace
        +--> 02F Generator.net.* 层级 Trace
```

主比较必须使用同一个 01A `generator_fp32_raw.onnx`。各脚本默认测试
Batch Size `1,4,8,16,32`，并使用 `42 + batch` 生成固定 latent noise；
每个引擎的 warmup、iterations、FP32/FP16 状态都写入对应 CSV。

## 脚本职责

| 脚本 | 实验内容 | 主要输出 | 对任务二的作用 |
|---|---|---|---|
| `02A_ORT_Benchmark.py` | ONNX Runtime CPU/CUDA，FP32/FP16，动态 Batch | `ort_benchmark.csv`、`ort_operator_summary.csv`、`profiles/ort_profile_*.json` | 端到端延迟、吞吐量、主机/可选 CUDA 内存，以及 ORT 节点 profile |
| `02B_TensorRT_Benchmark.py` | TensorRT GPU 构建 FP32/FP16 engine 并测试动态 Batch | `tensorrt_benchmark.csv`、`tensorrt_layer_profile.csv`、engine 文件 | engine-only 与端到端延迟、吞吐量、CUDA 内存快照、TensorRT 层耗时 |
| `02C_OpenVINO_Benchmark.py` | OpenVINO CPU，FP32/FP16，启用 `PERF_COUNT` | `openvino_benchmark.csv`、`openvino_operator_profile.csv` | CPU 端到端延迟、吞吐量、RSS 内存、OpenVINO 节点耗时 |
| `02D_TorchProfiler.py` | 对原始 PyTorch Generator 做 `torch.profiler` | `torch_trace.json`、`torch_operator_summary.csv` | 生成 Chrome Trace，并给出 PyTorch 算子级 Top-3 候选 |
| `02F_Profile_Generator_Layers.py` | 给 `Generator.net.*` 每层包裹 profiling scope | `layer_profiler_trace.json`、`layer_operator_summary.csv` | 把 `ConvTranspose/BatchNorm/ReLU` 映射到具体层，支撑图层面优化 |
| `02E_Merge_Task2_Report.py` | 只读取结果，不重新推理 | 见下方“最终结果” | 合并三引擎表、算子证据、Top-3 和结论报告 |

## 最终结果目录

运行 02E 后，`02E_Report` 只保留一套汇总结果：

- `task2_engine_comparison.csv`：FP32/FP16 × 引擎 × Batch 的延迟、吞吐量、显存/内存字段。
- `task2_operator_evidence.csv`：ORT、TensorRT、OpenVINO、PyTorch 和层级 profiler 的统一算子证据。
- `task2_top3_operators.csv`：每个 profile 来源的耗时 Top-3。
- `task2_report.md`：任务二结论、缺失证据和图层面优化建议。
- `task2_manifest.json`：唯一的汇总元数据 JSON，记录协议、覆盖范围、证据文件、缺失项和内存口径。
- `Task2_02E_Report.zip`：上述结果的打包文件。

各引擎脚本不再分别生成 `*_manifest.json`。但 `torch_trace.json`、
`layer_profiler_trace.json` 和 ORT 的 `profiles/*.json` 会保留：它们是
Chrome Trace/原始算子 profile 证据，不能和运行元数据合并而不丢失审计信息。

## 任务二完成判定

02A/02B/02C 完成三引擎部署与动态 Batch 基准；02D 完成 Chrome Trace；
02F 完成从算子到具体网络层的定位；02E 只有在所有必需 CSV、Trace 和
profile 文件找到后才会在 `task2_manifest.json` 标记 `status: complete`。

TensorRT 的 `cuda_*` 是整个 CUDA device 的显存快照，不等同于隔离的
TensorRT engine 显存；ORT/OpenVINO 的内存字段也必须按 CSV 中的来源解释。
