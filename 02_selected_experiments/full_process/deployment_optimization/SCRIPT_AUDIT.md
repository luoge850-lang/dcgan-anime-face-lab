# Deployment Optimization 脚本审计与唯一执行入口

审计日期：2026-09-03  
审计原则：脚本存在不等于实验完成；只有对应的 CSV、JSON、日志、Engine 或 Trace 实际落盘，才作为执行证据。

## 一、当前保留的正式脚本

当前保留 30 个 Python 实验脚本和 1 个本地报告工具。它们不是 30 个都要连续运行：每个阶段只运行一次执行脚本，阶段末尾的 Merge/Report 脚本只读取前面已经生成的证据。

### 01 ONNX 导出与融合

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `01_ONNX_Fusion/01_Export_Exp11_ONNX.py` | Exp11 EMA checkpoint、Generator 定义 | raw ONNX、checker、算子清单、FX/替换探针 | 导出并检查主计算图 | 核心 ONNX 导出与 checker 有证据；主图没有真实 wavelet/SN 节点 |
| `01_ONNX_Fusion/02_ORT_Fusion_Exp11.py` | raw ONNX | ORT 优化图、节点清单、算子级 profile、延迟表 | 验证 ORT 自动图优化 | 已运行；节点和端到端速度没有形成稳定收益 |
| `01_ONNX_Fusion/03_Manual_BN_Fusion_Exp11.py` | raw ONNX | BN folding 图、数值等价、延迟表 | 折叠 ConvTranspose+BN | 数值等价通过；速度门槛未通过 |
| `01_ONNX_Fusion/04_Subpixel_Reparameterization_Exp11.py` | raw ONNX | Conv+PixelShuffle 候选图和 benchmark | 验证上采样重参数化 | 脚本保留；当前结果不足以把 01D 写成稳定全图加速 |
| `01_ONNX_Fusion/05_TRT_Targeted_Fusion_Exp11.py` | raw ONNX、TensorRT | TRT 目标块/全图对比表和 manifest | 区分局部 block 融合与全图收益 | 局部 block 证据存在；不能外推为全 Generator 稳定加速 |

### 02 三引擎性能与瓶颈

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `02_Engine_Benchmark/02A_ORT_Benchmark.py` | 01A raw ONNX | ORT benchmark、operator summary | 测试 ONNX Runtime CPU/GPU 后端 | 已完成 |
| `02_Engine_Benchmark/02B_TensorRT_Benchmark.py` | 01A raw ONNX、TensorRT | FP32/FP16 Engine、延迟/吞吐/显存表 | 测试 TensorRT GPU | 已完成 |
| `02_Engine_Benchmark/02C_OpenVINO_Benchmark.py` | 01A raw ONNX | OpenVINO benchmark、算子 profile | 测试 OpenVINO CPU | 已完成 |
| `02_Engine_Benchmark/02D_TorchProfiler.py` | PyTorch Generator、checkpoint | Chrome Trace、operator summary | 获取 PyTorch 参考火焰图 | 已完成 |
| `02_Engine_Benchmark/02E_Merge_Task2_Report.py` | 02A–02D 已有 CSV/JSON | 三引擎总表、Top-3、报告、manifest | 只合并，不重新推理 | 已完成 |
| `02_Engine_Benchmark/02F_Profile_Generator_Layers.py` | PyTorch Generator、profiler 配置 | 层级 Trace、`net.*` 汇总 | 将 kernel/operator 映射到具体层 | 已完成 |

### 03 FP32/FP16/INT8 PTQ 基线

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `03_Quantization/03A_Prepare_Quantization_Protocol.py` | 真实动漫头像目录 | 100 张 calibration、5000 张 `real_eval`、latent 数组、协议 ZIP | 固定校准和质量评估协议 | 已完成 |
| `03_Quantization/03B_Build_FP32_FP16_Engines.py` | raw ONNX | FP32/FP16 Engine、build manifest | 建立无损精度参考 | 已完成 |
| `03_Quantization/03C_Build_INT8_PTQ_Engine.py` | raw ONNX、latent calibration | INT8 Q/DQ ONNX、PTQ Engine、manifest | 建立全局 INT8 PTQ 基线 | 已完成 |
| `03_Quantization/03D_Evaluate_FP32_FP16_INT8.py` | 三类 Engine、固定 latent、`real_eval` | FID、模糊率、LPIPS/频带误差、样图 | 统一质量评估 | 已完成 |
| `03_Quantization/03E_Merge_Quantization_Report.py` | 03A–03D 证据 | 量化总表、频域图、报告、`task3_manifest`、量化快照 | 只汇总和解释；不覆盖 04–08 的项目级总审计 | 已完成 |

### 04 量化敏感度与混合精度

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `04_Quantization_Sensitivity/04A_Layer_Sensitivity.py` | raw ONNX、PTQ protocol、INT8 baseline | 每层恢复 FP16 的 Engine、敏感度 CSV、曲线 | 定位 `ConvTranspose` 敏感层 | 已完成；`net.12` 的恢复收益最明显 |
| `04_Quantization_Sensitivity/04B_Mixed_Precision.py` | raw ONNX、候选层组合 | 组合策略 CSV、图、Engine | 筛选混合精度组合 | 已完成；候选为 `net.0+net.12` |
| `04_Quantization_Sensitivity/04C_Final_Confirmation.py` | raw ONNX、FP32/INT8 Engine、固定 protocol | 最终 mixed Engine、确认表、manifest | 独立复核最终策略 | 已完成；`net.0+net.12` 保留 FP16，其余 INT8 |

### 05 QAT

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `05_QAT/05A_QAT_Finetune.py` | 最佳 baseline checkpoint、latent calibration | FakeQuant QAT checkpoint、训练日志和曲线 | 训练时模拟量化误差 | 已完成；不负责 TensorRT 构建 |
| `05_QAT/05B_Evaluate_QAT.py` | 05A checkpoint、FP32/PTQ/MIXED Engine、protocol | QAT Engine、FID/模糊率/速度、对比图、manifest | 导出、构建并评估 QAT | 按修订后的全局质量口径通过；不宣称 ROI 显著优于 PTQ |

### 06 服务压测、Soak 与 Dynamic Batching

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `06_Service_Stress/06A_service.py` | 一个最终 TensorRT Engine | HTTP `/health`、`/generate`、service manifest | 启动单进程服务并做 smoke test | 已完成；实际 06 使用 QAT Engine |
| `06_Service_Stress/06BC_stress_runner.py` | 正在运行的 06A 服务 | 阶梯 Locust CSV、5 秒资源监控、日志、run manifest | 测试并发上界和软拐点 | 已完成；1–512 无硬失败，硬边界为 `>512` |
| `06_Service_Stress/06D_soak_test.py` | 正在运行的 06A 服务 | 60 分钟 steady CSV、5 秒监控和日志 | 筛查运行期内存泄漏 | 已完成；3600 秒、0 失败，未发现明显泄漏 |
| `06_Service_Stress/06E_report.py` | 06A、06BC、06D run | 汇总 CSV、3D 图、报告、manifest | 生成任务三报告 | 已完成运行性报告；硬崩溃和理论物理极限未观测到 |
| `06_Service_Stress/06F_Dynamic_Batching/06F_dynamic_batch_service.py` | 支持动态 Batch 的 Engine | 支持实际微批处理的 HTTP 服务 | 将多个请求合并为一次 TensorRT 执行 | 已完成；实际观测到 batch 2/4/8 |
| `06_Service_Stress/06F_Dynamic_Batching/06F_dynamic_batch_stress_runner.py` | 06F 服务 | 动态 batch 阶梯压测、5 秒监控、histogram | 测试动态 batching 的容量收益 | 已完成核心压测；归档缺少部分原始日志 |
| `06_Service_Stress/06F_Dynamic_Batching/06F_dynamic_batch_report.py` | 06F 原始 CSV、06E 固定 batch 对照 | Dynamic/Fixed P99、RPS、实际 Batch、图表和报告 | 分析动态 batching 是否推迟软拐点 | 已完成；软拐点约由 32 推迟至 64 |

### 07 可观测性与告警

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `07_MLOps_Observability/07_NOTEBOOK_ALL_IN_ONE.py` | 06A service、一个 Engine | Prometheus、Alertmanager、Grafana、告警事件、资源 CSV、Grafana 截图、ZIP | 验证白盒监控、两条告警和模拟触发 | 已完成；Grafana 截图和 firing/resolved 证据存在 |

### 08 模型热更新与 A/B 灰度

| 脚本 | 输入 | 输出 | 职责 | 审计结论 |
|---|---|---|---|---|
| `08_Model_Hot_Update_AB/08_NOTEBOOK_ALL_IN_ONE.py` | 06A service、PTQ Engine A、QAT Engine B、03A protocol | 热加载、A/B 路由、FID/模糊率、延迟、回滚日志、ZIP | 验证不停机模型切换、灰度和回滚 | 已完成；严格状态 `complete` |

## 二、本地报告工具

| 工具 | 用途 | 处理原则 |
|---|---|---|
| `build_deployment_report.ps1` | 根据本地证据生成 Word 报告 | 只用于报告排版，不属于 Kaggle 实验执行链；使用前应核对其读取的 manifest 是否为当前版本 |

## 三、唯一推荐执行顺序

```text
01A → 01B/01C/01D(可选) → 02A/02B/02C/02D/02F → 02E
→ 03A → 03B/03C → 03D → 03E
→ 04A → 04B → 04C
→ 05A → 05B
→ 06A → 06BC → 06D → 06E
→ 06F（Dynamic Batching 补充）
→ 07 → 08
```

其中：

- `02E`、`03E`、`06E`、`06F_dynamic_batch_report.py` 是报告/汇总脚本，不应被误认为重新执行模型推理。
- 07 和 08 都是 Kaggle 单元格脚本，不需要额外上传 YAML/JSON 配置。
- 05B、06A、06F、07、08 使用的 TensorRT Engine 必须在相同 TensorRT 版本和兼容 GPU 上反序列化。
- 01A 的 Haar/SN 是独立替换探针；当前 Exp11 主图没有真实小波分支和动态 SN，报告中不得写成已部署算子。

## 四、已经删除的冗余文件

已删除：

1. `experiments/Deployment_Optimization/_deprecated_20260819/06D_local_report.py`：旧的本地报告器，功能已被 `06E_report.py` 和现有 06E 结果替代。
2. `results/.../06F_Dynamic_Batching/report/06F_dynamic_batch_report.md`：与 `dynamic_batch_report.md` SHA256 完全相同的重复报告。

06_Service_Stress 下的 `legacy_20260819` 结果没有删除，因为它们是历史证据，不是活动脚本；正式报告不得把 legacy 与当前 run 混合。
