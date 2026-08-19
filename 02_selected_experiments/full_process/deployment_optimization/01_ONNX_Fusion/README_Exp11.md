# Exp11 Deployment Candidate

## 当前定位

本目录暂时使用 Exp11 的 EMA Generator 作为部署候选。它是当前已有完整权重中 FID 最好的参考模型，但还不是经过 ONNX、TensorRT、OpenVINO 和量化验证后的最终部署模型。

## 权重

```text
原始权重：
dcgan_lab/results/G强化实验结果/11_G_DiffAug_EMA_20K/generator_ema_final.pth
```

部署时只需要 Generator EMA 权重，不需要 Discriminator 或优化器权重。

## 模型定义

必须使用与 Exp11 完全一致的 Generator 定义：

- Width3x Generator；
- 五层 ConvTranspose2d；
- BatchNorm + ReLU；
- 最后 Tanh；
- 输出尺寸 64×64；
- 输入尺寸 `[batch, 128, 1, 1]`。

不能使用 Improve Generator 的类定义加载该权重。

## 实验顺序

### Step 1A：导出 ONNX（当前只做这一小步）

脚本：

```text
01_Export_Exp11_ONNX.py
```

在 Kaggle 中上传 Exp11 权重 Dataset，复制脚本运行。推荐显式填写权重路径：

```bash
python 01_Export_Exp11_ONNX.py \
  --checkpoint /kaggle/input/<exp11-weights>/generator_ema_final.pth \
  --output-dir /kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01A_Export
```

如果只找到一个路径包含 `11_G_DiffAug_EMA_20K` 的 `generator_ema_final.pth`，脚本也可以自动发现；找到多个候选时会要求显式填写路径。

本步成功标准：

- 日志出现 `state_dict` 加载成功；
- 输出 `generator_fp32_raw.onnx`；
- `onnx_check.json` 中 `checker` 和 `shape_inference_checker` 都为 `passed`；
- `operator_inventory_raw.csv` 已生成；
- `fx_graph.txt`、`fx_graph_nodes.csv` 和 `custom_operator_probe.csv` 已生成；
- 01C 完成后，`task1_manifest.json` 会汇总 01A/01B/01C 三个阶段；
- `export_sample_grid.png` 可以打开且图像与 Exp11 权重一致。

本步尚未做算子融合和三种引擎性能测试。确认 ONNX 导出成功后，再进入 Step 1B。

### Step 1B：ONNX Runtime 图优化与融合对比

脚本：

```text
02_ORT_Fusion_Exp11.py
```

它读取 `generator_fp32_raw.onnx`，不再读取 `.pth` 权重。默认使用 CPUExecutionProvider，先验证 CPU 路径；TensorRT 和 OpenVINO 在后续步骤单独测试。

脚本会自动检查并安装缺少的 `onnx` / `onnxruntime` 包。

同一 Notebook 中可以直接运行：

```bash
python 02_ORT_Fusion_Exp11.py \
  --raw-onnx /kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01A_Export/generator_fp32_raw.onnx \
  --output-dir /kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01B_ORT_Optimize \
  --provider CPUExecutionProvider
```

本步输出：

- `generator_fp32_fused.onnx`：ONNX Runtime 图优化后的模型；
- `fusion_check.json`：节点数量、输出误差、延迟和 Top 算子；
- `fusion_latency_summary.csv`：融合前后端到端延迟和吞吐量；
- `operator_latency_raw.csv` / `operator_latency_fused.csv`：ORT Node 级耗时；
- `ort_profile_raw.json` / `ort_profile_fused.json`：Chrome Trace/ORT profile；
- `ort_sample_raw.png` / `ort_sample_fused.png`：融合前后的样图。
- `Task1_01B_ORT_Optimize.zip`：脚本自动打包的全部结果。

本步成功标准：

- raw 和 fused 都能被 ORT 加载；
- 输出形状都是 `[batch, 3, 64, 64]`；
- `max_abs_output_difference` 很小（通常应接近 `1e-4` 量级）；
- fused 图节点数不增加；
- 融合后的 p50 延迟不明显变差。

### 后续步骤

1. 用 Exp11 Generator 定义加载并验证 `state_dict`；
2. 导出 `generator.onnx` 和 `generator_fp32_raw.onnx`；
3. 使用 `onnx.checker` 验证；
4. 检查 ONNX 节点和算子融合；
5. 测试 ONNX Runtime、TensorRT 和 OpenVINO；
6. 测试 FP32、FP16 和 INT8；
7. 对每种精度重新计算 FID、模糊率、LPIPS 和延迟；
8. 结果保存到：

```text
dcgan_lab/results/Deployment_Optimization_Results/01_ONNX_Fusion
```

### Step 1D：ConvTranspose 到 Conv+PixelShuffle 等价重参数化

脚本：

```text
04_Subpixel_Reparameterization_Exp11.py
```

01B 的 ORT 自动优化和 01C 的 ConvTranspose+BatchNorm folding 主要减少图节点，
但没有改变最重的 ConvTranspose kernel。01D 针对 Exp11 的四个
`kernel=4, stride=2, padding=1, group=1` 层，将每层重参数化为：

```text
Conv2d(Cin, 4*Cout, kernel=3, padding=1)
    -> Reshape -> Transpose -> Reshape（等价 PixelShuffle×2）
```

转换使用相位重排后的权重，不需要重新训练。脚本会输出：

- `generator_fp32_subpixel_fused.onnx`；
- `subpixel_fusion_check.json`：转换层、节点数、最大误差、各 Batch 加速比例和 speed gate；
- `subpixel_latency_summary.csv`：raw/fused 的 mean、p50、p95、吞吐量；
- `operator_latency_comparison.csv`：ConvTranspose 与 Conv/Reshape/Transpose 链路的 ORT Node profile；
- `ort_profile_raw.json` / `ort_profile_subpixel_fused.json`；
- `Task1_01D_Subpixel_Reparam.zip`。

01D 的通过标准仍是：最大绝对误差不超过 `1e-5`，并且至少两个 Batch 的平均延迟提升不低于 5%。
如果 CPU ORT 未加速，不应直接否定方案；应继续以任务二的 TensorRT GPU engine 重新构建并比较，因为该重参数化的主要目标是让 GPU backend 更容易选择高效 Conv2d/Shuffle kernel。

## 选择规则

Exp11 暂时作为部署基线。任何 FP16 或 INT8 版本都必须同时满足：

- 引擎可以稳定运行；
- FID 没有明显恶化；
- 模糊率没有明显上升；
- LPIPS 多样性没有严重下降；
- 延迟和吞吐量确实改善。

如果量化版本速度更快但生成质量明显变差，则保留 FP32 或 FP16，而不是强行使用 INT8。
