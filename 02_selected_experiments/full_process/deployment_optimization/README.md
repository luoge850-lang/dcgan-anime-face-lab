# Deployment Optimization Study

本目录用于完成 Generator 的 ONNX、推理引擎、性能分析和量化实验。

## 当前状态

当前暂时使用 Exp11 EMA Generator 作为部署候选，但尚未完成部署验证，因此仍不将它称为最终部署权重。
候选记录位于 `Exp11_Deployment/README.md`。

部署实验必须同时锁定：

1. 权重文件；
2. 与权重完全匹配的 Generator 类定义；
3. 输入形状 `[batch, 128, 1, 1]`；
4. 输出形状 `[batch, 3, 64, 64]`；
5. 固定的 FID、模糊率和 LPIPS 评估协议。

## 目前可见的候选权重

- `results/G强化实验结果/11_G_DiffAug_EMA_20K/generator_ema_final.pth`
  - Exp11 EMA 权重，当前完整的历史参考模型。
- `results/SDXL_Controlled_Study_Results/00_Baseline/B1_Formal_CleanUnique_17K/generator_ema_final.pth`
  - 17K 数据版本的权重，必须使用其对应的 Generator 定义加载。
- `results/G强化实验结果/09_G_Width3x_20K/generator_final.pth`
  - Exp09 原始 Generator 权重，不是 EMA 权重。
- `DCGAN_Improve_Standalone_21K` 的最终 `generator_ema_deploy.pth`
  - 只有 Improve 训练完成并检查指标、样图后，才可以加入候选。

这些文件不能直接互换。不同脚本的层结构、state_dict key 和 BatchNorm 状态可能不同。

## 执行顺序

### Step 0：锁定候选模型

分别整理每个候选的：

- `metrics.json`；
- FID；
- 模糊率；
- LPIPS 多样性；
- 最终样图；
- Generator 类定义。

只有在指标和样图都通过检查后，才能指定一个 `MODEL_TYPE` 和 `CHECKPOINT`。

### Step 1：导出并检查 ONNX

后续脚本必须显式指定模型类型和权重，不允许自动猜测：

```text
MODEL_TYPE=improve_standalone 或 exp11
CHECKPOINT=/path/to/generator_ema_deploy.pth
```

输出 `generator.onnx` 和 `generator_fp32_raw.onnx`，并使用 `onnx.checker.check_model` 验证。01A 同时捕获 FX 计算图，并对第三方 Haar 小波和动态 SN 逻辑执行标准 ONNX 替换探针；当前 Exp11 Generator 主图本身不包含这两类推理算子，因此探针结果与主图结果分开记录。

### Step 2：算子图和算子融合

记录导出前后节点数量、节点类型，并测试 ConvTranspose/BatchNorm 是否可做推理期融合。01B 是 ORT 自动优化负对照，01C 是手动 BN Folding；二者都输出节点级 profile 和融合前后耗时，不能只看端到端均值。

### Step 3：性能测试

在 ONNX Runtime、TensorRT 和 OpenVINO 中测试 FP32/FP16、Batch 1/4/8/16/32，记录 p50、p95、吞吐量和内存。

### Step 4：量化质量测试

FP16 直接转换；INT8 使用与 Generator 输入域一致的 latent noise 做校准。真实动漫头像用于校准集审计、FID/模糊率的真实参考和 LPIPS 评估，而不是直接替代 Generator 的 latent 校准输入。03D 还输出 Haar 高频子带误差，03E 只有在高频误差与质量恶化同时出现时才会判定“高频截断证据成立”。

### Step 5：形成部署报告

最终报告必须同时说明：

- 哪个权重被部署；
- ONNX 是否通过检查；
- Top 3 算子和优化方案；
- 三种引擎的速度/内存对比；
- FP32、FP16、INT8 的 FID、模糊率和 LPIPS 变化；
- 是否值得继续做 QAT。

最终结果按 `results/Deployment_Optimization_Results/01_ONNX_Fusion`、
`02_Engine_Benchmark`、`03_Quantization` 三个阶段保存；每个阶段由最后一个汇总脚本生成一个主 manifest，避免每个子脚本产生重复 JSON。
