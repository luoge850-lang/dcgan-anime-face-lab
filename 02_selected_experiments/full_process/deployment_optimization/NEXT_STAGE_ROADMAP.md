# Deployment Optimization：从已有实验到下一阶段的总路线

## 1. 当前审计结论

本路线以已经保存的 `01A/01B/01C`、`02E` 和 `03E` 结果为准，不把缺失文件或未运行脚本当作已完成证据。

| 部分 | 当前结论 | 证据与边界 |
|---|---|---|
| 01A ONNX 导出 | 已完成 | `generator_fp32_raw.onnx` 通过 ONNX checker 和 shape inference；14 个节点，主图为 5 个 ConvTranspose、4 个 BatchNormalization、4 个 Relu、1 个 Tanh |
| 01A 自定义算子 | 仅完成独立探针 | Haar 小波和动态 SN 的替换探针可以导出标准 ONNX，但 `custom_nodes=[]`，它们没有进入当前 Generator 主推理图 |
| 01B ORT 自动优化 | 已完成验证，但未提速 | B1 raw 4.2141 ms，optimized 4.2753 ms；节点数量和算子类型没有实际变化 |
| 01C 手工 BN folding | 数值等价完成，性能门槛未通过 | 节点 14→10，最大绝对输出差约 `3.07e-6`；B1 反而变慢，B4/B8/B16 只有小幅改善，未达到“至少两个 batch 提升 5%” |
| 01D Subpixel 重参数化 | 待实测 | 脚本已存在，必须在有 ONNX Runtime 的环境中实际运行后才能决定是否成功 |
| 02 Engine Benchmark | 已完成 | ONNX Runtime CPU、TensorRT GPU、OpenVINO CPU；FP32/FP16；Batch 1/4/8/16/32；动态 batch、Trace、层级 profile、Top-3 证据齐全 |
| 03 Quantization Baseline | 已完成 | 100 张真实动漫头像审计集、latent 校准、FP32/FP16/INT8 engine、Legacy/Standard FID、模糊率、LPIPS、高频误差均已保存 |

## 2. 已获得的核心结果

03E 的标准 FID、模糊率和 TensorRT 推理结果为后续实验的固定基线：

| 精度 | Standard FID | 模糊率 | 平均延迟 ms/batch | 吞吐 images/s |
|---|---:|---:|---:|---:|
| FP32 | 29.9911 | 12.0% | 14.6969 | 4354.7 |
| FP16 | 29.9941 | 12.0% | 4.1304 | 15495.0 |
| PTQ INT8 | 35.3198 | 12.5% | 3.0323 | 21106.1 |

当前 INT8 相比 FP32 的主要证据是：FID 增加约 5.3287，模糊率增加 0.5 个百分点，输出 RMSE 为 0.026279；小波分析显示 LL 的绝对误差较大，而 LH/HL/HH 的相对误差分别约为 12.89%、17.33% 和 19.69%。因此报告应写成“全网络累积量化误差与高频相对尺度失真共同导致退化”，不能写成“HH 子带是唯一原因”。

## 3. 小波和动态 SN 是否必须重做

### 推荐路线：不重做 01–03，继续当前 Exp11 主模型

当前主 Generator 推理图没有真实小波分支和动态 Spectral Normalization 节点。这并不妨碍完成后续的量化敏感度、混合精度、QAT 和服务压测，因为当前实际量化对象是：

```text
ConvTranspose → BatchNorm → ReLU → ... → ConvTranspose → Tanh
```

后续敏感度分析应直接测量这些真实存在的层，并继续用 Haar LH/HL/HH 误差作为输出端的高频诊断指标。这样可以保持 01–03 的结果可比，也不会伪造不存在的“wavelet layer”或“SN layer”。

### 只有在作业要求“主 Generator 必须包含小波/SN”时，才走重做路线

如果任务的硬性要求是：第三方小波库或动态 SN 必须作为 Generator 的真实推理逻辑，那么需要另建模型变体，而不是把一个小波节点事后塞进现有 ONNX。原因是加入新分支会改变：

- PyTorch Generator 的结构和参数；
- 训练权重和输出分布；
- ONNX 节点、校准激活范围和 TensorRT engine；
- FID、模糊率、LPIPS 和速度基线。

此时应从新的 PyTorch Generator 开始重新执行：

```text
模型结构 → 训练/微调 → ONNX 导出 → checker → 融合
→ 三引擎 benchmark → FP16/INT8 PTQ → FID/模糊率/LPIPS
```

旧 Exp11 结果只能作为“无小波/SN 的对照组”，不能与新模型结果直接合并成同一条性能曲线。

## 4. 从头到尾的实验顺序

```text
01A 计算图捕获与 ONNX checker
  ↓
01B ORT 自动优化、01C BN folding、01D 新融合候选
  ↓
02A–02E 三引擎 FP32/FP16 benchmark 与瓶颈定位
  ↓
03A–03E FP32/FP16/INT8 PTQ 质量损失基线
  ↓
04A 逐层恢复 FP16，建立敏感度排名
  ↓
04B 选择并验证混合精度策略
  ↓
05A FakeQuantize QAT 微调
  ↓
05B/05C 导出 QAT ONNX、构建真实 TensorRT engine、质量与速度复测
  ↓
06A 服务化部署、06B Locust 阶梯压测、06C GPU/RSS 监控
  ↓
06E 确定计算饱和点、稳定性拐点和硬崩溃拐点
  ↓
最终审计与实验报告
```

## 5. 任务清单

### 已有任务

- [x] 01A：捕获计算图、导出无报错 ONNX、执行 checker。
- [x] 01A：独立验证第三方 Haar 小波和动态 SN 的标准 ONNX 替换路径。
- [x] 01B：ORT 自动图优化并保存算子耗时。
- [x] 01C：BN folding、数值等价和融合前后耗时。
- [ ] 01D：Subpixel/Conv2d+PixelShuffle 重参数化的实际多 batch 性能门槛。
- [x] 02：ORT、TensorRT、OpenVINO 的 FP32/FP16 动态 batch benchmark。
- [x] 02：torch.profiler Chrome Trace、层级 profile、Top-3 瓶颈和图优化建议。
- [x] 03：100 张真实动漫头像审计/校准集及固定 latent 协议。
- [x] 03：FP32、FP16、INT8 engine 和质量指标对比。
- [x] 03：标准 FID、Legacy FID、模糊率、LPIPS、高频误差和 INT8 原因分析。

### 下一阶段任务

- [ ] 04A：逐层或逐块恢复 FP16，计算 FID、模糊率、LPIPS、高频误差和性能变化。
- [ ] 04A：绘制“层索引—精度损失/质量恢复”曲线。
- [ ] 04B：建立主干 INT8、敏感层 FP16 的混合精度候选矩阵。
- [ ] 04B：选择同时满足质量和速度门槛的最终混合精度策略。
- [ ] 05A：在最佳 PyTorch Baseline 中插入 FakeQuantize 并进行低学习率 QAT 微调。
- [ ] 05B：导出 QAT ONNX，执行 checker，并构建实际 TensorRT QAT engine。
- [ ] 05C：比较 PTQ-INT8 与 QAT-INT8 的 FID、模糊率、LPIPS、发丝/眼线高频指标和速度。
- [ ] 06A：部署单进程单 GPU 的本地服务并提供 health/generate 接口。
- [ ] 06B：编写 Locust 阶梯式加压脚本。
- [ ] 06C：每 5 秒记录 GPU 显存、GPU 利用率和服务 RSS。
- [ ] 06E：输出并发数—P99—显存三维图，识别三个拐点。
- [ ] Final：完成总审计，区分“已完成、性能未达标、未测量、可选模型变体”。

## 6. 统一实验纪律

- 所有质量比较使用同一组固定 latent、同一真实评估集和同一后处理。
- 所有速度比较记录 warm-up、iterations、batch、GPU、TensorRT/CUDA 版本和 engine hash。
- 不用“脚本存在”代替“实验已运行”。
- 不把 ORT/OpenVINO 的 RSS 与 TensorRT 的整卡显存快照当成同一种显存口径。
- 不将当前主图之外的 wavelet/SN 探针结果写成 Generator 已部署算子。
- 每个阶段只保留一个汇总 CSV、一个 manifest 和必要的原始证据，避免生成大量重复 JSON。
