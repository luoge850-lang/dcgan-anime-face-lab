# 06BC/06D：上限测试、Soak Test 与本地报告

## 重要边界

06BC 必须在 Kaggle GPU 上运行，因为它需要真实的 TensorRT 服务、Tesla T4、NVML 显存和 SM 数据。06D 不需要 GPU，可以在本机运行，但本机只能读取已经下载的结果，不能替代 Kaggle 的崩溃点测试。

## 需要上传到 Kaggle

如果 06A 已经在当前 Notebook 运行，不需要上传 engine、pth 或 ONNX。只需把 `06BC_stress_runner.py` 的全部内容复制到一个新的 Kaggle 单元格中执行；脚本会在 `/kaggle/working` 自动生成临时 Locust 文件。

## 运行前提

1. Kaggle 使用 GPU，Internet 开启；
2. 06A 已经启动，`GET http://127.0.0.1:8000/health` 返回 200 和 `status=ok`；
3. 不要在 06A 运行期间重启 Session；
4. 每次运行使用新的输出目录，避免覆盖上一次结果。

## 第一轮：扩大并发寻找硬崩溃边界

在新的 Kaggle 单元格中粘贴并运行 `06BC_stress_runner.py` 全部内容，默认命令行参数不会自动增加并发。推荐显式设置：

```text
--mode staged
--stages 1,2,4,8,16,32,48,64,80,96,128
--stage-seconds 30
--cooldown-seconds 15
--stop-failure-rate 0.05
```

如果直接粘贴到 Notebook，建议在脚本顶部将默认 `STRESS_STAGES` 改为上述阶梯，或者在执行前设置环境变量：

```python
import os
os.environ["STRESS_MODE"] = "staged"
os.environ["STRESS_STAGES"] = "1,2,4,8,16,32,48,64,80,96,128"
os.environ["STAGE_SECONDS"] = "30"
os.environ["COOLDOWN_SECONDS"] = "15"
```

硬崩溃判定：

- Locust 子进程非零退出；
- 请求失败率超过 5%；
- 阶段结束后 `/health` 失败；
- TensorRT 服务进程消失。

P99 很高但服务仍返回 200，不叫硬崩溃，脚本会记录为延迟风险或 `latency_abort_not_hard_crash`。

## 第二轮：Soak Test

第一轮确定稳定运行点后，建议选择 16 并发进行长期测试：

```text
--mode soak
--soak-concurrency 16
--soak-warmup-seconds 120
--soak-seconds 1800
--monitor-interval-seconds 5
```

其中：

- 120 秒用于预热；
- 1800 秒等于 30 分钟正式运行；
- 如果条件允许，将 `--soak-seconds` 改为 3600，进行 60 分钟测试；
- 06C 会在同一个脚本中每 5 秒保存 GPU 显存、SM、RSS 和健康状态。

Soak Test 通过标准：

- 正式运行阶段失败率为 0；
- `/health` 始终正常；
- 服务进程不退出；
- 预热后 RSS 进入稳定平台；
- 结束时 RSS 相对预热后基线增长小于 5%；
- GPU 显存没有持续单调增长。

## Kaggle 下载文件

每次 06BC 运行会创建一个独立目录，并自动生成 ZIP。至少下载：

```text
run_manifest.json
stage_results.csv              # staged 模式
soak_results.csv               # soak 模式
system_monitor_5s.csv
system_monitor_summary.json
stage_*_stats.csv
stage_*_locust.log
soak_*_stats.csv
soak_*_locust.log
```

最稳妥的方式是直接下载脚本最后打印的整个 ZIP，不要只下载单个 CSV。

## 本机运行 06D

将 Kaggle 下载并解压后的一个完整 run 目录放到本机，例如：

```text
C:\Users\32875\OneDrive\Desktop\image_generator\dcgan_lab\results\Deployment_Optimization_Results\06_Service_Stress\Run_20260819_01
```

然后运行：

```bash
python 06D_local_report.py \
  --run-root C:\Users\32875\OneDrive\Desktop\image_generator\dcgan_lab\results\Deployment_Optimization_Results\06_Service_Stress\Run_20260819_01
```

06D 会生成：

```text
06D_Report/stage_resource_summary.csv
06D_Report/service_stress_report.json
06D_Report/service_stress_report.md
06D_Report/concurrency_p99_memory_3d.png
06D_Report/latency_throughput.png
06D_Report/resource_by_concurrency.png
06D_Report/monitor_time_series.png
```

报告中的硬崩溃结论必须服从 06BC 的真实结果：如果最高测试并发仍然通过，只能写“在该上限以内未观察到硬崩溃”，不能虚构一个崩溃点。
