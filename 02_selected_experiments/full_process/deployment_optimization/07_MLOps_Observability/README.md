# 07：Kaggle 单元格版可观测性实验

## 1. 当前唯一执行入口

本目录只保留一个执行文件：

`07_NOTEBOOK_ALL_IN_ONE.py`

将该文件的全部内容复制到一个 Kaggle Notebook 单元格中运行即可。旧版
07A、07B、07C、07D、07E 以及独立 yml/json 配置已经删除，不再上传和执行。

## 2. 必须挂载的输入

只需要两个已有输入：

1. 包含 `06A_service.py` 的 Kaggle Dataset；
2. 包含 QAT-INT8 TensorRT Engine 的 Kaggle Dataset。

当前 Engine 候选路径：

`/kaggle/input/datasets/louisharrington/qat-int8-test/generator_trt_qat_int8 (1).engine`

不需要上传：

- 07 阶段的 Python 文件；
- Prometheus、Alertmanager、Grafana 的 yml/json 文件；
- 旧的 06BC、06D、06F 结果；
- FP32、FP16 或 PTQ Engine。

脚本会在 Kaggle 运行时自动安装 Python 依赖、下载监控二进制文件，并在
`/kaggle/working/07_MLOps_Observability/monitoring_stack/` 中生成配置。

## 3. Kaggle 执行步骤

### 第一个单元格：设置路径

在导入 TensorRT 之前运行：

```python
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TENSORRT_VERSION"] = "11.2.1.2"
os.environ["OBSERVABILITY_OUTPUT_DIR"] = "/kaggle/working/07_MLOps_Observability"

services = list(Path("/kaggle/input").rglob("06A_service.py"))
print("06A candidates:", services)
if len(services) != 1:
    raise RuntimeError("请保证 Kaggle 输入中只有一个 06A_service.py，或手动设置 BASE_SERVICE_PATH")

engine = Path("/kaggle/input/datasets/louisharrington/qat-int8-test/generator_trt_qat_int8 (1).engine")
if not engine.is_file():
    raise RuntimeError(f"找不到 QAT Engine: {engine}")

os.environ["BASE_SERVICE_PATH"] = str(services[0])
os.environ["ENGINE_PATH"] = str(engine)

print("BASE_SERVICE_PATH:", os.environ["BASE_SERVICE_PATH"])
print("ENGINE_PATH:", os.environ["ENGINE_PATH"])
```

### 第二个单元格：复制主脚本

复制 `07_NOTEBOOK_ALL_IN_ONE.py` 的全部内容到第二个单元格，直接运行。

脚本自动完成：

- 复用 06A TensorRT 推理逻辑；
- 启动 `/health`、`/generate`、`/metrics`；
- 记录 GPU 显存、GPU SM 利用率、RSS；
- 启动 Webhook；
- 生成 Prometheus 和 Alertmanager 配置；
- 启动 Prometheus、Alertmanager、Grafana；
- 发送并发请求；
- 验证两个告警规则；
- 受控模拟一次队列告警；
- 验证 Firing 和 Resolved；
- 在 Kaggle 容器内部使用无头 Chromium 登录真实 Grafana 页面并保存截图；
- 生成 CSV、JSON、Markdown 报告和证据压缩包。

主脚本运行通常需要几分钟。首次下载 Grafana、Prometheus 和 Alertmanager
时需要 Kaggle 打开 Internet。

## 4. 自动生成的告警规则

脚本在 Kaggle 工作目录中生成两条规则：

1. `DCGANQueueBacklog`：队列深度大于 50，持续 30 秒；
2. `DCGANHighP99Latency`：P99 延迟大于 200 ms，持续 30 秒。

队列值 60 是用于验证告警链路的受控模拟值，不是真实生产故障，也不能写成
真实崩溃拐点。

## 5. 主要输出

```text
/kaggle/working/07_MLOps_Observability/
├── 07_validation_summary.json
├── 07_load_results.csv
├── 07_metric_snapshots.csv
├── resource_monitor_5s.csv
├── alert_webhook_events.jsonl
├── 07_report.md
└── monitoring_stack/
    ├── prometheus.yml
    ├── rules.yml
    ├── alertmanager.yml
    ├── grafana_dashboards/dcgan_07.json
    └── logs/
```

Grafana 截图位于输出目录根部：

`/kaggle/working/07_MLOps_Observability/grafana_dashboard_screenshot.png`

证据压缩包：

`/kaggle/working/07_MLOps_Observability_evidence.zip`

## 6. Grafana 截图

主脚本会在 Kaggle 容器内部启动无头 Chromium，访问真实 Grafana 页面，登录
`admin/admin`，打开 `DCGAN Inference Service - 07`，并保存：

`/kaggle/working/07_MLOps_Observability/grafana_dashboard_screenshot.png`

截图应包含请求速率、P99、队列深度、GPU 显存、GPU SM 和 RSS。该截图来自
真实 Grafana UI，不是普通 Python 图表。Kaggle 浏览器是否能直接访问 3000
端口不再影响截图生成；若无头浏览器安装失败，Manifest 会明确记录失败原因。

## 7. 严格完成标准

`07_validation_summary.json` 中至少应满足：

- `/health` 为 200；
- `/generate` 为 200；
- `/metrics` 为 200；
- Prometheus 能抓取服务；
- 两条告警规则成功加载；
- 至少一条告警进入 Firing；
- Webhook 记录 Firing；
- 清除模拟条件后记录 Resolved；
- 有真实 Grafana 截图；
- `resource_monitor_5s.csv` 至少有 12 个 5 秒采样点。

只有全部满足，才可以在报告中写“07 可观测性与自动化告警实验完成”。

本阶段不验证模型热更新和 A/B 灰度发布；这两个内容应作为后续独立阶段，
避免和服务监控证据混淆。
