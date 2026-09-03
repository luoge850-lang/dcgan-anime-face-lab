# DCGAN Deployment Optimization 总审计

审计日期：2026-09-03  
审计对象：01 ONNX、02 三引擎、03 PTQ、04 敏感度/混合精度、05 QAT、06 服务压测、07 可观测性、08 热更新/A-B。

## 一、总判断

当前 Deployment 阶段已经形成一条完整的实验和证据链：模型导出、引擎 benchmark、PTQ 质量基线、敏感层筛选、QAT、服务压测、监控告警和模型灰度发布均有落盘结果。需要保留的主要限制只有两项：01D 没有形成稳定的全 Generator 融合提速结论；06 阶段没有观察到真实硬崩溃或 GPU 理论饱和，因此只能报告测试范围上界。

| 阶段 | 状态 | 主要证据 | 可写入报告的结论 |
|---|---|---|---|
| 01 ONNX/融合 | 部分完成 | 01A checker、01C 数值等价、01D profile | ONNX 合法、BN folding 等价；全图稳定融合提速未闭环 |
| 02 三引擎 benchmark | 完成 | 02E comparison、Top-3、Trace、manifest | ORT/TensorRT/OpenVINO FP32/FP16 和动态 Batch 对比完成 |
| 03 PTQ 基线 | 完成 | 03E metrics、frequency error、report | FP16 接近 FP32；全 INT8 更快但质量退化 |
| 04 敏感度/混合精度 | 完成 | 04A CSV/曲线、04C Engine/manifest | `net.0+net.12` 保留 FP16，其余 INT8 |
| 05 QAT | 修订口径通过 | 05A log、05B Engine/CSV/manifest | FakeQuant QAT 改善全局质量；不宣称 ROI 显著优势 |
| 06 服务压测 | 运行性完成 | 06A、06BC、06D、06E | 服务、阶梯并发和 60 分钟稳定性通过；硬崩溃未观测 |
| 06F Dynamic Batching | 核心完成，归档有缺口 | batch probe、stage CSV、report | 实际 batch 2/4/8，软拐点约由 32 推迟至 64 |
| 07 MLOps 监控 | 完成 | Grafana 截图、告警 firing/resolved、summary | 白盒监控和自动告警链路可用 |
| 08 热更新/A-B | 完成 | 08 summary、split、latency、FID、events | 零停机加载、灰度、质量抽样和回滚通过 |

## 二、任务一：ONNX 导出与算子融合

### 已完成

01A 已导出 raw ONNX，主图结构为 5 个 ConvTranspose、4 个 BatchNormalization、4 个 ReLU 和 1 个 Tanh；输入为 `[B,128,1,1]`，输出为 `[B,3,64,64]`。ONNX checker、shape inference 和标准算子替换探针均有结果文件。

01C 的 ConvTranspose+BatchNorm Folding 数值等价通过，节点数量由 14 减少到 10，最大输出误差约为 `3.07e-6`。

### 限制

当前 Exp11 Generator 主推理图没有真实小波分支或动态 SN 节点。Haar/SN 结果属于独立替换探针，不能写成当前模型已经部署了小波或 SN。

01B/01C 的融合速度受 Batch 和后端影响，未形成稳定的全 Generator 加速门槛。01D 目标融合脚本仍保留，但现有结果不足以宣称“融合后一定更快”。

## 三、任务二：三引擎性能

02A–02C 分别完成 ONNX Runtime、TensorRT GPU 和 OpenVINO CPU 的 FP32/FP16 端到端 benchmark；02D 生成 PyTorch Chrome Trace；02F 将 profile 映射到 `net.*` 层；02E 生成总表和 Top-3 算子证据。

共同瓶颈主要集中在 ConvTranspose/上采样链。显存字段来自不同后端的不同统计口径，报告中不能把 TensorRT 设备快照、ORT RSS 和 OpenVINO RSS 直接当成同一种“模型显存”。

## 四、任务三：PTQ、敏感度与 QAT

### PTQ 基线

03A 固定了 100 张真实 calibration 审计集、5000 张 `real_eval` 参考集和 latent 校准/评估数组。03B 生成 FP32/FP16 Engine，03C 生成全 INT8 PTQ Engine，03D 统一评估，03E 汇总报告。

| 配置 | Standard FID ↓ | 模糊率 | Batch=32 P99（ms） | 解释 |
|---|---:|---:|---:|---|
| FP32 | 29.9910 | 12.0% | 7.4696 | 质量参考 |
| PTQ INT8 | 35.3198 | 12.5% | 1.4117 | 速度收益明显，但质量退化 |
| MIXED PTQ | 31.1776 | 12.1% | 1.6182 | `net.0+net.12` 保留 FP16 |
| QAT INT8 | 31.6456 | 11.3% | 1.7963 | 质量恢复，速度仍为 INT8 数量级 |

### 敏感度与混合精度

04A 对 ConvTranspose 层逐层恢复 FP16，04B 做组合筛选，04C 独立复核最终方案。当前最稳妥的表述是：`net.0` 和 `net.12` 的 FP16 保留对质量恢复最有价值，最终采用“主干 INT8、`net.0+net.12` FP16”的混合策略。

### QAT

05A 在 PyTorch 训练中插入 FakeQuantize，并使用最佳 baseline 做微调；05B 独立完成 ONNX 导出、QAT 权重物化、TensorRT Engine 构建和质量/速度复测。

QAT 相对 PTQ 的 FID 降低约 3.6741，模糊率下降 0.7 个百分点。QAT Batch=32 P99 约为全 INT8 PTQ 的 1.272 倍、混合 PTQ 的 1.110 倍，因此可以写“保留 INT8 速度收益”，不能写成“与 PTQ 完全相同”。现有图像证据不支持发丝/眼线 ROI 的显著性声明。

## 五、任务四：服务化、监控和热更新

### 06 服务压测

06A 的 `/health`、`/generate` 和 PNG 输出验证通过。06BC 覆盖 1–512 并发，所有阶段失败数为 0，约 32 并发出现 P99 软延迟拐点；由于没有硬失败，只能写“硬崩溃边界大于 512”。

06D 使用 120 秒预热和 3600 秒 steady soak，steady 实际约 3601.6 秒、1,226,890 请求、0 失败，P99 约 63 ms，RPS 约 340.83；RSS 和 GPU 显存没有显示持续异常增长。

06F 实际观察到服务端 batch=2、4、8。并发 32 时吞吐量相比固定 batch=1 提升约 14.01%，软拐点从约 32 推迟到约 64。06F 归档缺少部分原始日志和运行时摘要，因此报告状态保留为 `complete_with_packaging_gaps`。

### 07 可观测性

07 的 summary 显示：`/health`、`/generate`、`/metrics` 均为 200；Prometheus target up；两条规则成功加载；告警 firing 和 resolved 均有 JSONL 证据；Grafana 截图存在。该阶段证明监控和告警链路可用，不等于已经接入真实邮件系统。

### 08 热更新与 A/B

08 使用 PTQ INT8 作为 A、QAT INT8 作为 B。候选 Engine 加载耗时约 26.71 ms，加载前后 PID 都为 58，加载期间请求全部成功。

| 目标 B 流量 | 实际 B 流量 | 误差 |
|---:|---:|---:|
| 10% | 12.0% | 2.0 个百分点 |
| 50% | 52.5% | 2.5 个百分点 |
| 100% | 100% | 0 个百分点 |

A 的 P99 为 98.57 ms，B 的 P99 为 186.74 ms，均满足当前 200 ms 门槛，但 B 的尾延迟明显更高。

在同一 5000 样本评估中，A FID=35.5710、模糊率=12.50%；B FID=32.0422、模糊率=11.62%。B 的 FID 约下降 9.92%，模糊率下降 0.88 个百分点。该结果支持“B 候选在本次灰度抽样中质量更好”，但仍是单次 sampled FID，不能替代多随机种子统计检验。

回滚后流量比例回到 0%，回滚请求全部命中 A。B 仍显示为 loaded 是有意保留候选上下文，不代表回滚失败。

## 六、最终可提交结论

可以写：

> 本项目构建了从 PyTorch/ONNX 图导出、TensorRT/ORT/OpenVINO 性能评估、PTQ 与 QAT 量化、敏感层混合精度、HTTP 压测、Prometheus/Grafana 可观测性，到模型零停机热更新和 A/B 灰度回滚的完整部署验证链。实验在 Kaggle Tesla T4 单卡上完成，结果均由 CSV、JSON、Engine、日志和截图支撑。

必须保留的限定语：

1. 当前 Exp11 主图不含真实 wavelet/SN；只完成了独立替换探针。
2. QAT 的优势是全局质量改善，不宣称发丝/眼线 ROI 显著改善。
3. 06 阶段未观察到硬崩溃；报告边界为“在最高已测并发内未崩溃，硬失败边界大于 512”。
4. 60 分钟 Soak 支持“未发现明显运行期内存泄漏”，不是绝对无泄漏证明。
5. 07/08 是单进程单 GPU 验证，不等同于 Kubernetes 多副本生产集群。

## 七、唯一结果入口

- 脚本逐项说明：`../../experiments/Deployment_Optimization/SCRIPT_AUDIT.md`
- 结果索引：`EXPERIMENT_INDEX.md`
- 任务一：`01_ONNX_Fusion/`
- 任务二：`02_Engine_Benchmark/02E_Report/`
- 量化和 QAT：`03_Quantization/03E_Report/`、`04_Quantization_Sensitivity/04C/`、`05_QAT/05B/`
- 服务压测：`06_Service_Stress/06E/`、`06_Service_Stress/06F_Dynamic_Batching/report/`
- 可观测性：`07/07_MLOps_Observability/evidence/`
- 热更新：`08_Model_Hot_Update_AB/evidence/`
