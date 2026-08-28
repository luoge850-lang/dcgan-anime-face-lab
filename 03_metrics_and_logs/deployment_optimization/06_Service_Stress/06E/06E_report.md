# 06E 任务三服务化部署最终报告

生成时间：2026-08-19T08:51:38.304Z

## 总体判定：`incomplete`

运行性结论：通过

## 服务器与部署环境

- GPU：NVIDIA Tesla T4
- GPU显存：约 14.9 GB
- TensorRT：11.2.1.2
- CUDA：12.8
- PyTorch：2.10.0+cu128
- 服务模式：单进程、单 worker、batch=1、单 TensorRT context
- Engine：generator_trt_qat_int8 (1).engine

## 06BC 阶梯压测

- 并发范围：1–512
- 阶段数量：17
- 总失败数：0
- 硬崩溃：未观察到
- 硬崩溃边界：大于 512
- 软延迟拐点：32 并发，P99 110 ms

## 06D 严格60分钟 soak

- 并发：16
- steady实际时长：3601.60 秒
- steady请求数：1226890
- 失败数：0
- P99：63.0 ms
- RPS：340.83491714884207
- RSS头尾变化：3.32%
- GPU显存头尾变化：0.00%
- soak判定：pass

## 资源极限判定

- GPU SM峰值：19.00%
- GPU显存占用峰值：4.41%
- GPU饱和代理：not_observed
- 结论：当前HTTP压测未达到GPU算力或显存物理极限。

## 审计结论

任务三的服务正确性和长时间稳定性证据充分；但严格意义上的‘真实硬崩溃拐点’和GPU物理极限尚未测得。因此不能把任务三写成完全完成，只能写为‘服务化部署与60分钟soak通过，容量上界已测至512，硬崩溃边界和物理极限未观察到’。

## 生成文件

- 06BC_stage_resource_summary.csv
- 06D_soak_summary.csv
- concurrency_p99_gpu_memory_3d.svg
- concurrency_p99_gpu_memory_3d.png
- concurrency_p99.svg
- concurrency_p99.png
- concurrency_gpu_memory.svg
- concurrency_gpu_memory.png
- concurrency_rps.svg
- concurrency_rps.png
- 06E_manifest.json
- 06E_report.md
