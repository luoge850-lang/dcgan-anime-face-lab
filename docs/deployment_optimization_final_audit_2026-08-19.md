# DCGAN Deployment Optimization 全实验最终审计

审计对象：`01_ONNX_Fusion`、`02_Engine_Benchmark`、`03_Quantization`、`04_Quantization_Sensitivity`、`05_QAT`、`06_Service_Stress`。本文件是当前总判断；阶段细节和文件入口见 `EXPERIMENT_INDEX.md`。

## 总体结论

| 阶段 | 状态 | 证据边界 |
|---|---|---|
| 01 ONNX / 融合 | ONNX 与数值验证完成；稳定提速部分未完全闭环 | checker、shape inference、BN folding 数值等价有记录；01D 的 operator block 有通过记录，但 whole-generator speed gate 未形成通过结论；当前主图没有小波或动态 SN |
| 02 三引擎基准 | 完成 | ORT、TensorRT、OpenVINO、Torch 参考、动态 Batch、operator evidence、Top-3 和报告均落盘；不同引擎内存字段口径不同 |
| 03 PTQ 质量基线 | 完成 | FP32/FP16/INT8 的 FID、模糊率、LPIPS/频带误差和报告已落盘 |
| 04 敏感度与混合精度 | 完成（当前图范围） | `net.0 + net.12` 保留 FP16、主干 INT8；不是小波/SN 层结论 |
| 05 QAT | 按修订后的全局质量口径通过 | FakeQuant、QAT 微调、混合策略导出和真实 TensorRT 评估完成；不宣称发丝/眼线 ROI 显著优于 PTQ |
| 06 服务压测 | 服务运行与 60 min soak 通过；严格物理极限未闭环 | `/health`、`/generate` 通过；1–512 并发无失败；32 为软延迟拐点；60 min steady 无失败；硬崩溃边界和 GPU 物理饱和未观察到 |

## 关键量化证据

| 配置 | Standard FID ↓ | 模糊率 | P99 batch=32（ms） | 结论 |
|---|---:|---:|---:|---|
| FP32 | 29.9910 | 12.0% | 7.4696 | 质量参考 |
| FP16 | 29.9941 | 12.0% | 03E 基线 | 基本无损 |
| PTQ INT8 | 35.3198 | 12.5% | 1.4117 | 速度高但质量退化 |
| MIXED PTQ | 31.1776 | 12.1% | 1.6182 | `net.0 + net.12` FP16 |
| QAT INT8 | 31.6456 | 11.3% | 1.7963 | 相对 PTQ FID 改善 3.6741 |

QAT 的 batch=32 P99 是全局 PTQ 的约 1.272 倍、MIXED PTQ 的约 1.110 倍，仍明显快于 FP32（约 4.16 倍）。因此可以写“同一数量级、保留 INT8 速度收益”，不应写“与全局 PTQ 完全相同”。

## 01–03 原始部署链

1. 01A 导出了合法 ONNX：opset 17，输入 `[B,128,1,1]`，输出 `[B,3,64,64]`，checker 和 shape inference 通过，主图包含 5 个 ConvTranspose、4 个 BN、4 个 ReLU、1 个 Tanh。
2. 01B ORT 自动优化和 01C BN folding 已完成数值检查；01C 节点从 14 减少到 10，最大绝对误差约 `3.07e-6`，但端到端收益依赖 Batch，不构成稳定提速。
3. 01D manifest 将 operator block 与 whole-generator 状态分开记录；不能把 operator block 通过改写成整个 Generator 已稳定提速。
4. 02E manifest 为 `complete`，三引擎和性能证据齐全；TensorRT FP16 是已验证的速度基线，ConvTranspose/上采样链是共同瓶颈。
5. 03E 表明 FP16 质量接近 FP32，而全局 INT8 FID 和模糊率恶化；LH/HL/HH 的相对频带误差高于 LL，但不能把 FID 变化只归因于 HH。

## 04–05 量化优化链

- 04A 逐层候选是 `net.0`、`net.3`、`net.6`、`net.9`、`net.12` 这些 ConvTranspose 节点。
- 04C 最终确认 `net.0 + net.12` FP16、其余路径 INT8；FID 31.1776、模糊率 12.1%，相对全 INT8 P99 比值约 1.105。
- 05A 在 PyTorch 中对被量化主干插入 FakeQuant，使用最佳 baseline 微调；net.0/net.12 保留 FP16，训练目标含输出误差、Haar detail 和梯度项。
- 05B 以相同 latent、统一 TensorRT engine 评估 FP32、PTQ、MIXED、PRE-QAT、QAT。修订后的通过条件是全局 FID 改善、模糊率不劣化且速度保持 INT8 数量级；原始严格 Haar/速度门槛仍作为诊断记录。

## 06 服务压测链

- 06A：HTTP `/health` 和 `/generate` 验证通过，生成输出为合法 64×64 PNG。
- 06BC：并发 1、2、4、8、16、32、48、64、80、96、128、160、192、256、320、384、512 全部通过；约 32 并发出现软延迟拐点，P99=110 ms；512 并发 P99=1600 ms，但失败数仍为 0。
- 06BC 资源：GPU 显存峰值 677.19 MB，占约 4.41%；SM 峰值 19%；RSS 约 1070.88→1114.80 MB。GPU 未达到饱和。
- 06D：warmup 120 s + steady 3600 s；steady 实际 3601.60 s，1,226,890 请求，0 失败，P99=63 ms，RPS≈340.83；RSS 头尾增长约 3.32%，GPU 显存头尾变化 0%。
- 06E：运行性结论为 `pass`；严格状态为 `incomplete`，因为没有硬崩溃样本，且 GPU 饱和代理未达到。

严格表述应为：

> 在 Kaggle Tesla T4 环境、当前 QAT INT8 engine、单进程单 worker、batch=1 服务配置下，1–512 并发均未出现硬失败；约 32 并发为软延迟拐点，硬崩溃边界大于 512。60 分钟 steady soak 通过，未发现明显运行期 RSS/GPU 显存增长。当前证据不能声称已测出单卡理论物理极限。

## 目前不能声称的内容

1. 不能声称当前 Generator 已包含或部署小波、动态 SN；现有主图没有这些节点。
2. 不能声称 QAT 在发丝、眼线 ROI 上显著优于 PTQ；现有证据只支持全局 FID、模糊率和无明显画质崩塌。
3. 不能声称已经测出单卡理论显存/算力极限或硬崩溃点；只能报告 128 并发上界和 32 并发软拐点。
4. 不能将 TensorRT 整卡 CUDA 快照、ORT/OpenVINO RSS 直接当作同一“模型显存”指标。
5. 06E 的 `incomplete` 是严格物理极限判定，不表示 HTTP 服务验证或 60 分钟 soak 失败。

## 当前报告入口

- 全实验索引：`EXPERIMENT_INDEX.md`
- 量化图表：`../figures/`
- 任务三汇总：`06_Service_Stress/06E/06E_report.md`、`06_Service_Stress/06E/06E_manifest.json`
- 量化原始证据：`03_Quantization/03E_Report/`、`04_Quantization_Sensitivity/04C/`、`05_QAT/05B/`
