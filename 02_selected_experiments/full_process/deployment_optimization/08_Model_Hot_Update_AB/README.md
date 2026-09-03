# 08：模型热更新与 A/B 灰度发布

## 目标

本阶段验证两个能力：

1. 在服务继续接收请求时，加载候选 Engine B，不重启 HTTP 进程；
2. 将流量按 0%、10%、50%、100% 逐级分配到 B，并记录两个版本的延迟、成功率和抽样 FID/模糊率，最后回滚到 A。

这是 Kaggle 单进程、单 GPU 的生产模式验证，不等同于 Kubernetes 多副本部署。

## 唯一执行脚本

将 `08_NOTEBOOK_ALL_IN_ONE.py` 全部复制到一个 Kaggle Notebook 单元格运行。脚本会自己安装 Python 依赖并生成所有 CSV、JSON、Markdown 和 ZIP，不需要上传 yml/json。

## Kaggle 输入

必须挂载：

1. 一个包含 `06A_service.py` 的数据集；
2. PTQ INT8 Engine，作为稳定版本 A；
3. QAT INT8 Engine，作为候选版本 B；
4. 与 04C/05B 使用同一协议的 `real_eval/` 图片目录，或包含该目录的 `Task3_03A_Quantization_Protocol.zip`。

建议：A 使用 `generator_trt_int8 (1).engine`，B 使用 `generator_trt_qat_int8 (1).engine`。两个 Engine 必须由同一 TensorRT 版本构建，并能在同一 GPU 上反序列化。

## 路径设置单元格

```python
import os
from pathlib import Path

root = Path("/kaggle/input")
services = sorted(root.rglob("06A_service.py"))
engines = sorted(root.rglob("*.engine"))
real_dirs = sorted(p for p in root.rglob("real_eval") if p.is_dir())
protocol_zips = sorted(root.rglob("Task3_03A_Quantization_Protocol*.zip"))

print("06A:", services)
print("Engines:", engines)
print("real_eval:", real_dirs)
print("03A protocol ZIP:", protocol_zips)

assert len(services) == 1
ptq = [p for p in engines if "int8" in p.name.lower() and "qat" not in p.name.lower()]
qat = [p for p in engines if "qat" in p.name.lower() and "int8" in p.name.lower()]
assert len(ptq) == 1, ptq
assert len(qat) == 1, qat
assert len(real_dirs) == 1 or len(protocol_zips) == 1

os.environ["BASE_SERVICE_PATH"] = str(services[0])
os.environ["ENGINE_A_PATH"] = str(ptq[0])
os.environ["ENGINE_B_PATH"] = str(qat[0])
os.environ["FID_REAL_DIR"] = str(real_dirs[0] if real_dirs else protocol_zips[0])
os.environ["TENSORRT_VERSION"] = "11.2.1.2"
os.environ["AB_OUTPUT_DIR"] = "/kaggle/working/08_Model_Hot_Update_AB"
os.environ["AB_FID_SAMPLES"] = "5000"
```

如果候选路径数量不为 1，不要让脚本猜，直接在该单元格中打印出的列表里选择唯一文件后设置对应环境变量。

## 实验过程

脚本自动执行：

1. 只加载 A 并预热；
2. 发送 A 基线请求；
3. 并发请求期间加载 B；
4. 检查加载前、加载后和回滚后的健康状态及 PID；
5. 按 B=10%、50%、100% 灰度；
6. 记录每个请求实际命中的模型版本和延迟；
7. 使用相同随机种子对 A/B 生成图片并计算 sampled FID、模糊率；
8. 回滚到 A；
9. 生成证据 ZIP。

## 通过标准

`08_validation_summary.json` 的 `status` 必须为 `complete`，并且：

- 更新前、中、后 `/health` 均为 200；
- 更新期间请求全部成功；
- 服务 PID 不变；
- 实际 B 流量比例与目标误差不超过 5 个百分点；
- 回滚后请求全部回到 A；
- B 的 P99 不超过 200 ms 且不应明显劣于 A；
- B 的 sampled FID 不比 A 恶化超过 5%；
- B 的模糊率相对 A 增量不超过 0.5 个百分点。

## 输出

```text
/kaggle/working/08_Model_Hot_Update_AB/
├── 08_validation_summary.json
├── 08_request_log.csv
├── 08_traffic_split.csv
├── 08_fid_sample.csv
├── 08_latency_by_version.csv
├── 08_update_events.jsonl
└── 08_report.md
```

下载：

```text
/kaggle/working/08_Model_Hot_Update_AB_evidence.zip
```

## 与 07 的关系

07 证明“监控和告警系统可用”；08 证明“模型版本可以在不停机条件下加载、灰度、评估和回滚”。两者证据不能互相替代。最终报告应同时引用 07 的 Grafana 截图、06D 的 60 分钟 Soak Test 和 08 的更新/A-B 文件。
