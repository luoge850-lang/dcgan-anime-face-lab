# 4K SDXL 控制变量实验协议 — Fine-Tuning 版

## 1. 研究目的

在同一个 4,000 张训练预算内，从 Exp11 预训练权重 fine-tune，用清洗后的 SDXL 新图替换部分原图后，生成器的 FID/Coverage/Diversity 是否改善。

不追求绝对最低 FID — 追求"同样 4K 数据 + 同样 fine-tune 100 epoch 下，加 SDXL 是否更好"的可归因结论。

## 2. 五组实验设计

| 组别 | 原图 | SDXL 图 | 总量 | 运行平台 |
|---|---:|---:|---:|---|
| **A0** | 4,000 | 0 | 4,000 | Kaggle |
| **A10** | 3,600 | 400 | 4,000 | Kaggle |
| **A20** | 3,200 | 800 | 4,000 | Kaggle |
| **A30** | 2,800 | 1,200 | 4,000 | Kaggle |
| **A50** | 2,000 | 2,000 | 4,000 | Kaggle |

### 为什么是这五个比例

- **A0**：纯基线。所有其他组与此比较。
- **A10** (10% SDXL)：保守替换。如果 SDXL 有效，应看到信号。
- **A20** (20% SDXL)：中等替换。验证效果是否随 SDXL 比例递增。
- **A30** (30% SDXL)：确认递增趋势，填补 A20→A50 之间的空白。
- **A50** (50% SDXL)：激进上限。直接测试 SDXL 能推多远。

### SDXL 数据需求

当前清洗后有 2,037 张 accepted。A50 需 2,000 张 → 可行。所有五组均可运行。

## 3. 为什么用 Fine-Tuning 而非从头训练

### 优势
- **归因清晰**：所有组从同一权重出发，差异完全归因于数据混合比，不受初始化随机性干扰
- **FID 更低**：Exp11 已学到良好特征，fine-tune 收敛更快、FID 更低
- **时间更短**：100 epoch vs 200 epoch 从头训练
- **消融敏感性更高**：数据变化的信号不会淹没在训练噪声中

### 潜在风险
- **灾难性遗忘**：如果 SDXL 分布与原图差异太大，D 可能过度适应新图，G 产生伪影。A10 作为最保守组可检测此问题。
- **先验偏见**：Exp11 在 20K 原图上训练，fine-tune 到 4K 可能过拟合。A0 baseline 直接揭示此风险。

## 4. 固定条件 (不允许变动)

```
架构:        Exp11 (Width3x G 768ch, SN+Hinge D, 7.77M/1.44M)
损失:        Hinge (L_D = relu(1-D_real) + relu(1+D_fake), L_G = -D(fake))
增强:        DiffAugment (color + translation + cutout)
EMA:         decay=0.9999 (G 权重, 评估和最终采样使用 EMA)
优化器:      Adam (β₁=0.5, β₂=0.99), lr=1e-4
训练:        100 epochs, batch=32, seed=42
数据:        总量严格 4,000, 64×64, Flip(0.5) + EdgeSharpen(0.2)
```

## 5. 混合数据规程

1. **原图池**：从 anime-faces 全量中 SHA-256 去重 → seed=42 采样 N_ORIG 张 → 固定 manifest
2. **SDXL 池**：清洗后的 accepted 图片，seed=42+1000 采样 N_SDXL 张 → 固定 manifest
3. **混合**：合并 → seed=42+2000 shuffle → 固定 manifest
4. **数据集 manifest** 保存为 `dataset_manifest.txt` 供复现

每个脚本的 N_ORIG/N_SDXL 不同，但采样逻辑完全一致（嵌套 seed 设计）。

## 6. 指标与判定

### 主要指标
| 指标 | 含义 | 期望方向 |
|------|------|--------|
| Legacy FID | InceptionV3 特征距离（最多 4,000 张 real / 5,000 张 fake；五组相同代码路径） | ↓ 越低越好 |
| D_real / D_fake | Hinge logit 健康度 | D_real>0, D_fake<0 |

### 辅助指标
| 指标 | 含义 |
|------|------|
| LPIPS AlexFeat | 生成图 vs 真实图的感知相似度 |
| LPIPS Diversity | 生成图之间的多样性 |
| Laplacian Variance | 锐度（越高越锐利） |
| Blur Rate | 模糊图比例（相对于真实图 P10 阈值） |
| Edge Density | 边缘密度及与真实图的比值 |

### 判定逻辑
1. **先看 D 健康**：D_real_aug > 0 且 D_fake_aug < 0 → G-D 均衡
2. **再看 FID**：越低越好，但必须在 D 均衡的前提下
3. **检查锐度-多样性权衡**：FID 改善不应以严重模糊或多样性崩溃为代价
4. **A0 是唯一参照**：所有组与 A0 比较

### 通过条件
```
✅ 有效组:     FID < A0 FID  且  Blur Rate < A0×2  且  Diversity > A0×0.8
⚠ 存疑组:     FID < A0 FID  但 辅助指标有退化 → 需要人工检查 epoch 图
❌ 失败组:     FID ≥ A0 FID  或  D 崩溃 (D_real < -1 持续)
```

## 7. SDXL 生成参数 (锁定)

```
模型:       Animagine XL 4.0 (cagliostrolab/animagine-xl-4.0)
分辨率:     1024×1024
Scheduler:  DPM++ 2M Karras, 20 steps, CFG 7.0
Batch:      2 (T4 16GB VRAM)
Targets:    16 种组合 (4 profile + 4 three-quarter + 3 front + 2 glasses
             + 1 closed eyes + 1 open mouth + 1 looking up)
调度:       Round-robin (global_idx % 16)
负面 prompt: 与 Kaggle 测试版 v3 完全一致
```

## 8. 抗混叠降采样

SDXL 先生成 1024×1024，再 Lanczos 缩放到 64×64。256×256 预览用于人工审核结构，64×64 检查目标特征是否可辨。

## 9. 运行顺序

```
1. Colab: 生成 SDXL 候选 (01_Production/01_SDXL_Generate.py)
2. Colab: 清洗候选图 (02_Cleaning/02_Clean_SDXL.py)
3. 上传: cleaned_accepted.zip → Kaggle Input
4. Kaggle: FT_A0 → FT_A10 → FT_A20 → FT_A30 → FT_A50 (任意顺序, 五个独立脚本)
5. Kaggle: 统一评估 (04_Evaluation/04_Evaluate_All.py)
```

## 10. 报告要求

完成所有实验后，报告必须包含：

| 项目 | 内容 |
|------|------|
| 各组 FID | A0/A10/A20/A30/A50 的最终 EMA FID |
| D 健康度 | D_real_aug / D_fake_aug 表 |
| 锐度 | Laplacian Variance 比较 |
| 多样性 | LPIPS Diversity 比较 |
| 边缘密度 | Edge Density 及比值 |
| 模糊率 | Blur Rate (< 真实图 P10) |
| 最优 epoch | 每组最优 checkpoint epoch |
| 结论 | 哪个混合比最优？是否建议 A50？ |
