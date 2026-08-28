# DCGAN Lab 全实验图表（单指标版）

本目录已重做为“单图单论点”：每张 SVG 只展示一个指标，主体使用单系列折线图或单系列水平柱状图。旧版组合图没有删除，已移入带日期的备份目录。

当前版本的上一版组合图备份：`../figures_v1_backup_20260819/`；更早的旧图备份：`../figures_legacy_backup_20260819/`。

## 图表范围

- 前期调优：训练轮数 FID、增强实验 FID、代表实验 D/G loss；
- G 强化：FID、多样性 LPIPS、代表实验 D/G loss；
- 深度调优：G 端方法 FID、D 端方法 FID、代表实验 G adversarial / Wavelet loss；
- CLIP：λ_clip-FID、λ_clip-CLIP MMD²、代表实验 D loss；
- 部署：BN 融合延迟、三引擎 FP32/FP16 Batch=32 延迟与吞吐、TensorRT Top-3 算子；
- 量化：PTQ FID/模糊率/吞吐、逐层恢复 FID/延迟、混合策略 FID/延迟、QAT FID/模糊率/P99；
- 服务：并发-P99、并发-RPS、并发-GPU 显存、并发-RSS、并发-SM、soak 阶段 P99/RPS、soak 头尾 GPU/RSS。

## 文件

- 全实验图表索引.html：按实验顺序逐张查看；
- 全实验指标汇总.csv：训练调优和 CLIP 指标索引；
- 全实验数据清单.csv：results 下原始数据文件清单；
- figure_manifest.json：每张图的来源、指标、单位、行数和限制；
- ../../tools/build_simple_figures.js：Node.js 核心库重生成脚本。

## 口径限制

1. 前期调优、G 强化、深度调优、CLIP 的 FID 是历史 legacy 项目协议；部署阶段 Standard FID 不与它们直接混合排名。
2. 历史训练结果中的 Wavelet/SN 是训练实验方法；当前部署图的算子证据仍以 ConvTranspose、BatchNorm、ReLU、Tanh 为准。
3. 05B 的全局模糊率与全局高频指标不支持 literal hair/eyeliner ROI 显著优于 PTQ 的结论；相关图表只呈现可核实的全局指标。
4. 06D 的 128 并发是已测上界，不是理论物理崩溃点；无失败或 OOM 时不能绘制“已证实崩溃点”。
5. 训练曲线最多等间隔抽样 240 个点，仅为可读性处理，不改变原始 CSV。

生成命令：

node tools/build_simple_figures.js
