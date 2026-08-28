# 按阶段重绘的单指标图表

本目录中的图表由 `tools/build_stage_figures.js` 根据 `dcgan_lab/results` 的 CSV/JSON 重新生成。每张图只表达一个指标，使用单系列折线图或水平柱状图，不使用组合图。

## 阶段目录

1. `01_前期训练与增强/`：训练轮数、数据增强与 FID；
2. `02_G_D模块调优/`：G 端与 D 端模块/目标函数对比；
3. `03_G强化与训练策略/`：G 容量、数据规模、DiffAugment、EMA 与 LPIPS；
4. `04_CLIP调优/`：CLIP lambda 与 FID/MMD2；
5. `05_部署与量化/`：BN、三引擎、PTQ、逐层敏感度、混合精度和 QAT；
6. `06_服务压测/`：固定 batch=1 的并发 P99/RPS/GPU/RSS/SM，以及 60 分钟 soak 的 P99/RPS/资源头尾对比。

## 证据边界

- 重新生成了 34 张有结果数据支撑的图表；
- 09–15 的代表性 loss 图按项目所有者说明不重新生成；
- 16 的 CLIP 训练轨迹也没有重新加入主展示区，避免把已删除的训练过程图重新包装成当前证据；
- 34–40 的 GPU/RSS/SM 与 soak 图已经从当前 `results/figures` 纳入本次冻结，并由 06E 的汇总表与报告支撑；它们仍然只表示固定 workload 下的观测，不表示已找到物理崩溃点；
- 历史 DCGAN FID 使用 Legacy Inception-v3 项目协议，部署阶段 Standard FID 单独解释。

生成脚本支持显式传入源项目和输出目录，例如：

```text
node tools/build_stage_figures.js <dcgan_lab> <output_directory>
```
