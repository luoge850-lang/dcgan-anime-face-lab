# G 强化实验：加深 Generator 层数 + 残差连接

## 实验背景

深度调优阶段 20 组实验的核心发现：
- **所有有效改进都是"限制 D"**：SN (FID -10.21), SN+Hinge (-13.15), R1 (-19.48)
- **所有"增强 G 架构"的尝试仅停留在浅层注意力模块**（SE/CBAM），G 本身仍为 5 层裸 ConvTranspose
- **所有 BCE 实验出现 G→D 对抗失衡**：D_real→1.0, D_fake→0.0, G_loss 持续上升
- **根本瓶颈：G 表征能力不足，无法与 D 形成健康对抗**

## 实验目标

在不改动 D3 (SN+Hinge, 已验证为最优 D 配置) 的前提下，通过加深 G 层数和引入残差连接，探究 G 表征能力的提升能否解决对抗失衡问题。

## 实验设计

| 实验 | G 架构 | D 配置 | 核心探究 |
|------|--------|--------|----------|
| 00 (Baseline) | 标准 5 层 ConvTranspose | D3 (SN+Hinge) | D3 基线 (FID=96.25) |
| 01 | ResG (残差上采样块) | D3 | 残差连接能否改善深层 G 的梯度传播？ |
| 02 | ResG + SE@L3 | D3 | 残差 G + 已验证最优的通道注意力 |
| 03 | DeepResG (3 Conv/阶段) | D3 | G 深度的进一步增加是否有边际收益？ |
| 04 | ResG + G×2 步更新 | D3 | 不改架构，仅增加 G 更新频率的效果 |

## 关键约束

- D 侧完全锁定为 D3（与 13_D_SN_Hinge 逐 bit 一致），仅 G 侧改动
- NOISE_DIM=128，正交初始化（与实验 11-19 一致）
- Hinge Loss，200 Epochs，所有超参数与深度调优阶段保持一致
- 评估指标：FID, LPIPS, Diversity, Laplacian Variance, Edge Density

## 技术参考

- EIGGAN (CMC, 2024): ResBlock + Spatial Attention + R1/R2 混合正则化
- SNGAN (Miyato et al., ICLR 2018): SN + Hinge 标准配方
- StyleGAN2 (Karras et al., CVPR 2020): R1 正则化 + Equalized Conv
- P2D (WACV 2024): 特征空间 R1 正则化
