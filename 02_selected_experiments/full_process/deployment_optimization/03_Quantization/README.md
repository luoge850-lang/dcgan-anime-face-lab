# 任务三：FP32 / FP16 / INT8 量化破坏程度验证

## 1. 任务目标

本任务不重新训练 GAN，而是使用任务二已经导出的 Exp11 Generator ONNX，比较三种推理精度：

1. FP32：原始精度，作为质量和数值基准；
2. FP16：半精度，观察速度收益和轻微数值误差；
3. INT8 PTQ：训练后静态量化，观察压缩后的速度收益以及生成质量损失。

最终为后续 QAT（量化感知训练）提供基线：如果 INT8 的 FID、模糊率或 LPIPS 明显恶化，就说明不能直接部署纯 INT8，需要混合精度或 QAT。

## 2. 统一基线

- 模型：任务一的 `01A_Exp11_onnx/generator_fp32_raw.onnx`；
- 不使用 01B/01C 的融合图作为主比较图，避免把“图结构变化”和“精度变化”混在一起；
- 三种精度使用同一组 latent noise、同一批次、同一后处理；
- 推理引擎优先使用 TensorRT，因为它能够直接构建 FP16 和 INT8 engine，并输出校准缓存；
- 记录 TensorRT、CUDA、显卡型号和 TensorRT builder 配置。

## 3. 校准集与评估集的边界

Generator 的输入不是图片，而是 latent `z`（形状 `[B, 128, 1, 1]`）。因此“100 张真实动漫头像”不能直接作为 Generator 的 INT8 输入校准样本。

本实验必须区分两个集合：

- `real_calibration_100`：按导师要求保存的 100 张真实动漫头像，用于数据清单、图像质量阈值和真实分布参考；
- `latent_calibration`：由固定随机种子生成的 latent 向量，用于校准 Generator 的中间激活范围；
- `latent_eval`：另一组固定 latent，用于三种精度的逐样本公平比较；
- `real_eval`：固定的真实图像评估集，建议至少 1,000 张，最好沿用已有 FID 评估集。100 张真实图像只作为校准/审计集时，不应单独承担稳定 FID 参考。

报告中必须明确：真实头像是评估参考集，latent 才是 Generator 的量化校准输入。

## 4. 子阶段

### 03A：校准集与固定评估协议

生成并保存：

- 100 张真实头像及 `real_calibration_100_manifest.csv`；
- 固定 latent 校准集（建议 512 个）；
- 固定 latent 评估集（建议 5,000 个，资源不足时至少 1,000 个）；
- 真实评估集清单、预处理规则和 SHA-256；
- 随机种子、图像尺寸、归一化方式和评估样本数。

### 03B：FP16 PTQ 基准

从同一个 raw ONNX 构建 TensorRT FP16 engine。FP16 不是重新训练，而是把推理计算转换为半精度。保存 engine、构建日志、版本信息和基准延迟。

### 03C：INT8 PTQ

当前 Kaggle 使用 TensorRT 11。该版本已经移除旧版 `IInt8Calibrator` 和
`config.int8_calibrator`，因此不能再使用 TensorRT 10 时代的 Python calibrator。

从同一个 raw ONNX 使用 NVIDIA TensorRT ModelOpt 离线校准：

- 使用固定 latent 校准输入；
- 使用 `modelopt.onnx.quantization` 的 INT8 entropy 方法；
- 输出带 `QuantizeLinear/DequantizeLinear`（Q/DQ）节点的 INT8 ONNX；
- 再将 Q/DQ ONNX 编译为 TensorRT strongly-typed INT8 engine；
- 保存 Q/DQ ONNX、engine 和构建 manifest；
- 首版先测试纯 INT8，不要一开始混入 FP16 层，否则无法判断纯 INT8 的破坏程度。

### 03D：质量、失真与性能评估

对 FP32、FP16、INT8 使用完全相同的 `latent_eval` 生成图片，并计算：

- FID（报告中注明是何种 Inception/FID 实现）；
- 模糊率（沿用真实图像 Laplacian p10 阈值）；
- LPIPS 多样性；
- Laplacian 均值、高频能量和边缘密度；
- FP32 对 FP16/INT8 的逐样本误差：MAE、RMSE、最大绝对误差；
- 延迟、吞吐量和内存增量。

### 03E：报告与结论

输出 FP32 / FP16 / INT8 对比表、样本 contact sheet、量化误差表、INT8 失真原因和是否需要 QAT 的判断。03E 会自动汇总 03A--03D 的 manifest，只保留一个最终 `task3_manifest.json`。

## 5. 判定规则

FP16 可接受的初步标准：

- FID 相对 FP32 没有明显恶化；
- 模糊率变化不超过预设容差；
- LPIPS 多样性不出现明显坍缩；
- 输出没有系统性色块、断边或五官结构破坏。

INT8 需要重点观察：

- FID 是否明显升高；
- 模糊率是否显著上升；
- 高频能量和边缘密度是否下降；
- LPIPS 是否下降，说明多样性减少；
- 最后几层 ConvTranspose、Tanh 前后是否出现激活截断或饱和。

如果 INT8 失败，下一步不是直接宣称量化不可用，而是测试：

1. 最后一层或最后两层保留 FP16；
2. 对 ConvTranspose 使用更合适的 per-channel scale；
3. 增加 latent 校准数量；
4. 最后再进入 QAT。

## 6. 预期结论

本任务的目标不是强行让 INT8 质量变好，而是量化“速度收益—生成质量损失”的关系。可能出现以下情况：

- FP16：质量基本保持，速度和显存有所改善；
- INT8：速度进一步提高，但 FID、模糊率或高频指标恶化；
- INT8 破坏主要来自 ConvTranspose 的激活范围压缩、校准样本不足、per-tensor scale 过粗以及 Tanh 输出附近的饱和。

只有当结果表和失真分析完成后，才能决定是否需要 QAT。
