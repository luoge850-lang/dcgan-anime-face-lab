# DCGAN Deployment Optimization 结果索引

审计日期：2026-09-03  
规则：只将已保存的 CSV、JSON、日志、Trace、Engine、图片或报告作为证据；脚本存在不代表实验已经完成。

## 目录结构

```text
Deployment_Optimization_Results/
├─ 00_Audit/
├─ 01_ONNX_Fusion/
├─ 02_Engine_Benchmark/
├─ 03_Quantization/
├─ 04_Quantization_Sensitivity/
├─ 05_QAT/
├─ 06_Service_Stress/
├─ 07/07_MLOps_Observability/
└─ 08_Model_Hot_Update_AB/
```

## 阶段完成情况

| 阶段 | 主要问题 | 输入 | 主要输出 | 当前结论 |
|---|---|---|---|---|
| 01A–01D | ONNX 合法性、计算图捕获、融合候选 | Exp11 EMA checkpoint、Generator 定义 | raw ONNX、checker、节点清单、融合 profile | ONNX 和数值等价完成；稳定全图融合提速未闭环；主图没有真实 wavelet/SN |
| 02A–02F | ORT/TensorRT/OpenVINO 性能和瓶颈 | raw ONNX、FP32/FP16 配置 | 三引擎 benchmark、动态 Batch、Chrome Trace、Top-3 | 完成 |
| 03A–03E | FP32/FP16/INT8 PTQ 质量损失 | raw ONNX、固定 latent、真实参考集 | FP32/FP16/INT8 Engine、FID、模糊率、LPIPS、频带误差 | 完成 |
| 04A–04C | 量化敏感层和混合精度 | raw ONNX、PTQ protocol、INT8 baseline | 逐层 CSV/曲线、混合 Engine、确认 manifest | 完成；选择 `net.0+net.12` FP16 |
| 05A–05B | FakeQuant QAT 适应截断误差 | 最佳 baseline、latent protocol、基线 Engine | QAT checkpoint、QAT Engine、质量/速度对比 | 按修订全局口径通过；不宣称特定 ROI 显著优势 |
| 06A–06E | 服务正确性、阶梯并发、Soak | QAT Engine、HTTP 服务 | health/generate、Locust CSV、5 秒资源监控、60 分钟报告 | 运行性完成；1–512 无硬失败，边界为 `>512`；未测出理论物理极限 |
| 06F | Dynamic Batching 是否实际发生 | 支持动态 Batch 的 QAT Engine、固定 batch 对照 | batch histogram、动态/固定 P99/RPS、资源图 | 核心实验完成；归档缺少部分原始日志，状态为 `complete_with_packaging_gaps` |
| 07 | 全链路监控和自动告警 | 06A service、一个 Engine | Grafana 截图、Prometheus/Alertmanager、firing/resolved 事件 | 完成 |
| 08 | 热更新、A/B、回滚 | 06A service、PTQ A、QAT B、03A protocol | 更新事件、流量比例、延迟、FID、回滚证据 | 完成；严格状态 `complete` |

## 关键结果

### 量化与 QAT

| 配置 | Standard FID ↓ | 模糊率 | Batch=32 P99（ms） | 角色 |
|---|---:|---:|---:|---|
| FP32 | 29.9910 | 12.0% | 7.4696 | 质量参考 |
| PTQ INT8 | 35.3198 | 12.5% | 1.4117 | 全 INT8 基线 |
| MIXED PTQ | 31.1776 | 12.1% | 1.6182 | `net.0+net.12` 保留 FP16 |
| QAT INT8 | 31.6456 | 11.3% | 1.7963 | QAT 部署候选 |

QAT 相对 PTQ 的 FID 降低约 3.6741，模糊率下降 0.7 个百分点；Batch=32 P99 约为 PTQ 的 1.272 倍，但仍保持 INT8 数量级的速度收益。由于视觉样本没有独立发丝/眼线 ROI 证据，不能宣称特定 ROI 显著改善。

### 服务、Dynamic Batching 和 MLOps

- 06BC：并发 1、2、4、8、16、32、48、64、80、96、128、160、192、256、320、384、512 均无失败；约 32 并发为 P99 软拐点；硬崩溃未观察到。
- 06D：预热 120 秒，steady 3600 秒以上，1,226,890 请求，0 失败，P99 63 ms，RPS 约 340.83；未发现明显 RSS/GPU 显存持续增长。
- 06F：动态 Batch 实际出现 2、4、8；并发 32 时 RPS 约提升 14.01%，软拐点从约 32 推迟到约 64，但这不是硬崩溃边界。
- 07：两个告警规则加载成功，Prometheus target up，告警 firing/resolved 均有证据，Grafana 截图存在。
- 08：A/B 灰度 10%、50%、100% 的比例误差分别为 2.0、2.5、0 个百分点；A/B 请求成功率均为 100%；回滚后请求全部回到 A。

## 报告入口

- 总审计：`Deployment_Optimization_Final_Audit.md`
- 当前机器可读总状态：`deployment_optimization_current_manifest.json`
- 根目录的 `deployment_optimization_manifest.json` 和 `deployment_optimization_audit.json` 是 2026-08-12 的历史快照，不作为当前总状态。
- 脚本审计：`../../experiments/Deployment_Optimization/SCRIPT_AUDIT.md`
- 任务一：`01_ONNX_Fusion/01A_Export/`、`01B_ORT_Optimize/`、`01C_BN_Fold/`、`01D_TRT/`
- 任务二：`02_Engine_Benchmark/02E_Report/`
- 量化：`03_Quantization/03E_Report/`、`04_Quantization_Sensitivity/04C/`、`05_QAT/05B/`
- 服务：`06_Service_Stress/06E/`、`06_Service_Stress/06F_Dynamic_Batching/report/`
- 监控：`07/07_MLOps_Observability/evidence/`
- 热更新：`08_Model_Hot_Update_AB/evidence/`

## 证据边界

1. 最高成功并发不等于硬崩溃点；本实验只能报告硬失败边界大于 512。
2. GPU 显存和 SM 均未饱和，HTTP 并发拐点不能写成单卡理论物理极限。
3. 60 分钟 Soak 是运行期泄漏筛查，不是绝对无泄漏证明。
4. 07/08 是 Kaggle 单进程单 GPU 验证，不等同于 Kubernetes 多副本生产集群。
5. `legacy_20260819` 只用于追溯，不得与活动 run 混合汇总。
