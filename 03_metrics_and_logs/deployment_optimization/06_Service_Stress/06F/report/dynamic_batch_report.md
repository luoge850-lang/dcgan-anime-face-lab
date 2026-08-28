# 06F Dynamic Batching 补充实验报告

## 1. 审计结论

本报告基于 Kaggle 下载的原始压测包 `06F_Dynamic_Batching.zip`、其中的
`dynamic_stage_results.csv`、`dynamic_monitor_5s.csv`、engine 探针和 06E
固定 batch=1 对照 CSV 生成。没有重新运行实验，也没有补造缺失日志。

06F 的核心执行证据完整：B=1/2/4/8 engine 探针通过，正式压力测试覆盖 1、2、
4、8、16、32、64、128 并发，所有阶段 HTTP 失败率为 0，并且服务端记录到真实
的 batch=2、4、8。因此可以确认 Dynamic Batching 在本次压力测试中实际发生。

## 2. 实验配置

| 项目 | 配置/结果 |
|---|---|
| TensorRT | 11.2.1.2 |
| GPU | Tesla T4 |
| GPU 总显存 | 14911.69 MB |
| engine 输入 | `[B,128,1,1]` |
| engine 输出 | `[B,3,64,64]` |
| 最大服务 batch | 8 |
| 合批等待窗口 | 5 ms |
| 并发阶段 | 1、2、4、8、16、32、64、128 |
| 单阶段时长 | 60 秒 |
| 冷却时间 | 30 秒 |
| 监控周期 | 5 秒 |

## 3. 动态 batch 能力与实际合批

`dynamic_batch_engine_probe.csv` 显示 B=1、2、4、8 全部通过。正式压测的
`stage_batch_histogram` 显示：并发 2 出现 batch=2，并发 4 出现 batch=4，
并发 8 及以上出现 batch=8。因而本实验不是只验证动态 shape，而是验证了服务
端请求聚合和 TensorRT 批量执行。

## 4. 固定 batch=1 与 Dynamic Batching 对比

详细数据见 `dynamic_batch_summary.csv`。

| 并发 | 固定 P99 (ms) | Dynamic P99 (ms) | 固定 RPS | Dynamic RPS | RPS 变化 | 实际平均 Batch | 最大 Batch |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 10 | 260.77 | 108.19 | -58.51% | 1.00 | 1 |
| 2 | 8 | 13 | 329.02 | 173.81 | -47.17% | 2.00 | 2 |
| 4 | 17 | 17 | 336.86 | 263.58 | -21.75% | 4.00 | 4 |
| 8 | 29 | 24 | 346.66 | 401.38 | +15.79% | 7.98 | 8 |
| 16 | 55 | 48 | 350.17 | 396.61 | +13.26% | 7.89 | 8 |
| 32 | 110 | 90 | 352.35 | 401.73 | +14.01% | 8.00 | 8 |
| 64 | 280 | 180 | 357.95 | 400.28 | +11.82% | 8.00 | 8 |
| 128 | 470 | 430 | 350.73 | 398.11 | +13.51% | 8.00 | 8 |

### 结果解释

低并发时请求数量不足以填满 batch，5 ms 合批等待窗口反而增加了额外开销，
因此并发 1～4 时 Dynamic Batching 的 RPS 低于固定 batch=1。并发达到 8 后，
实际平均 batch 接近 8，GPU 执行摊薄了单请求开销，Dynamic Batching 的 RPS
稳定在约 396～402，且 P99 低于固定 batch=1。

在并发 32 时，Dynamic Batching 的 RPS 为 401.73，相比固定 batch=1 的 352.35
提升 14.01%；P99 从 110 ms 降至 90 ms。

## 5. 软拐点分析

本实验预先定义软拐点为：相邻阶段 P99 至少增加 2 倍，同时 RPS 增长低于 5%。

- 固定 batch=1：16→32 并发时，P99 从 55 ms 增至 110 ms，RPS 增长约 0.62%，
  因此软拐点为 32；
- Dynamic Batching：16→32 并发时，P99 从 48 ms 增至 90 ms，比例约 1.875，
  未达到 2 倍；32→64 并发时，P99 从 90 ms 增至 180 ms，RPS 不再增长，
  因此软拐点约为 64。

在本实验的操作性定义下，Dynamic Batching 将软拐点从约 32 并发推迟到约 64
并发。但这不是 CUDA OOM 意义上的硬崩溃拐点。

## 6. 资源监控与稳定性

本次监控共记录 104 个样本：

| 指标 | 观察结果 |
|---|---:|
| GPU 显存最小/最大 | 677.19 / 677.19 MB |
| GPU 显存占总显存 | 4.41% |
| SM 利用率最大 | 4% |
| RSS 最小/最大 | 1084.06 / 1089.45 MB |
| 压测失败请求 | 0 |
| 最后一次队列深度 | 111 |

GPU 显存在测试期间没有增长，RSS 仅小幅变化，未发现明显的显存持续增长或
系统内存泄漏证据。但最后一个监控样本仍有 `queue_depth=111`，说明压测结束时
服务队列可能尚未完全排空。因此本次结果可以说明压力阶段没有失败，不能声称
服务已在最后采样点完成完全排空。

## 7. 图表

- `dynamic_vs_fixed_p99.svg`：固定 batch=1 与 Dynamic Batching 的 P99 对比；
- `dynamic_vs_fixed_rps.svg`：固定 batch=1 与 Dynamic Batching 的吞吐量对比；
- `dynamic_actual_batch_size.svg`：并发数与实际 batch 大小；
- `dynamic_resource_monitor.svg`：GPU 显存、SM 和 RSS 监控摘要。

## 8. 缺失证据与边界

下载的原始 ZIP 缺少：

- `dynamic_batch_smoke.csv`；
- `dynamic_batch_runtime_summary.json`；
- `dynamic_service.log`；
- 原始报告和图表。

这几个文件不影响利用已有阶段 CSV 判断 Dynamic Batching 是否发生，但会降低归档
完整性。`dynamic_batch_runtime_summary.json` 只有在服务正常 shutdown 后才会生成，
不能由本地报告事后伪造。

此外，06F 最高测试到 128 并发，未出现 CUDA OOM、服务进程退出或 HTTP 失败。
因此任务三关于真实硬崩溃点的结论仍应写为：

> 在 128 并发以内未观察到真实硬崩溃；真实硬崩溃边界高于本次测试上限，不能将 128 认定为物理极限。

