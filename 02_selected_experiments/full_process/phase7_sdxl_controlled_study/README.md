# SDXL 4K Controlled Study — 完整工作流

## 研究问题

在固定 DiffAugment + EMA + Width3x DCGAN 架构下，**从 Exp11 预训练权重 fine-tune**，用清洗后的 Animagine XL 4.0 新图替换部分原图（总量保持 4,000），观察 FID/Coverage/Diversity/BlurRate 是否改善。

## 五组消融实验

| 组别 | 原图 | SDXL 图 | 总量 | SDXL 占比 | 运行平台 |
|---|---|---|---:|---:|---|---|
| **A0** baseline | 4,000 | 0 | 4,000 | 0% | Kaggle |
| **A10** | 3,600 | 400 | 4,000 | 10% | Kaggle |
| **A20** | 3,200 | 800 | 4,000 | 20% | Kaggle |
| **A30** | 2,800 | 1,200 | 4,000 | 30% | Kaggle |
| **A50** | 2,000 | 2,000 | 4,000 | 50% | Kaggle |

> ⚠ A50 需要 ~2,000 张清洗后的 SDXL 图片。当前有 2,037 张 → A50 可行。

## 核心设计

- **Fine-tuning（非从头训练）**：所有五组都从 Exp11 预训练权重加载 G+D，在新数据集上 fine-tune 100 epoch
- **A0 先跑** → 输出 `baseline_reference.json`（FID + Coverage + Diversity + BlurRate 全量基准）
- **A10/A20/A30/A50 读取 baseline_reference.json** → 每 25 epoch 评估 FID+Coverage → Coverage 熔断 + FID 早停
- **Ablation 在 Kaggle**：Exp11 权重 + SDXL 图片 + A0 baseline 已上传至 Kaggle Input，脚本自动检测
- **SDXL 生成在 Colab**：Colab 方便挂机生成 + 下载 zip

---

## 流程概览

```
Step 1 (Colab): 生成 SDXL 候选图 → 下载 candidates.zip
Step 2 (Colab): 清洗候选图 → 下载 cleaned_accepted.zip
Step 3 (本地): 将 cleaned_accepted.zip 上传到 Kaggle Input
Step 4 (Kaggle): 依次运行 4 个 fine-tuning 脚本 → 下载各组结果
Step 5 (Kaggle): 运行统一评估
```

---

## Step 1: SDXL 生成 (Colab)

```
┌─────────────────────────────────────────────────────────┐
│ 1. colab.research.google.com → T4 GPU                  │
│ 2. 粘贴 01_Production/01_SDXL_Generate.py              │
│ 3. 修改 TARGET_ACCEPTED = 1000 (或更多)                 │
│ 4. 运行 → 下载 sdxl_candidates.zip                      │
└─────────────────────────────────────────────────────────┘
```

**16 类目标均衡覆盖**：round-robin 调度保证每类均匀采样。
**中断安全**：不依赖上传 zip，每次从头开始 → 下载当前批次即可。多次运行的结果手动合并。

## Step 2: SDXL 清洗 (Colab)

```
┌─────────────────────────────────────────────────────────┐
│ 1. colab.research.google.com → T4 GPU (CPU 也可)       │
│ 2. 拖入所有 SDXL 候选 zip 到 /content/                  │
│ 3. 粘贴 02_Cleaning/02_Clean_SDXL.py → 运行            │
│ 4. 第一轮: 自动筛选 → 下载 review_template.csv          │
│ 5. Excel 打开 → 标记 manual_keep=1/0 → 保存为           │
│    review_checked.csv → 拖回 Colab                      │
│ 6. 重新运行 → 下载 cleaned_accepted.zip                 │
└─────────────────────────────────────────────────────────┘
```

**自动筛选门禁**：SHA-256 去重 → aHash 近重复 → Laplacian 模糊 → 纯色/近空白

## Step 3: 上传到 Kaggle

1. 在 Kaggle 创建新 Dataset → 上传 `cleaned_accepted.zip`
2. 上传 Exp11 权重 (`generator_ema_final.pth` + `discriminator_final.pth`)
3. 在所有消融实验 notebook 中 attach 这两个 dataset + 原版 anime-faces

## Step 4: 消融训练 (Kaggle)

五个独立脚本放在 `03_Ablation/`：

| 脚本 | N_ORIG | N_SDXL | 预计时间 |
|------|-------:|-------:|--------:|
| `FT_A0_Baseline_4K_100E.py` | 4000 | 0 | ~1.5h |
| `FT_A10_90ori_10sdxl_4K_100E.py` | 3600 | 400 | ~1.5h |
| `FT_A20_80ori_20sdxl_4K_100E.py` | 3200 | 800 | ~1.5h |
| `FT_A30_70ori_30sdxl_4K_100E.py` | 2800 | 1200 | ~1.5h |
| `FT_A50_50ori_50sdxl_4K_100E.py` | 2000 | 2000 | ~1.5h |

```
┌─────────────────────────────────────────────────────────┐
│ 1. kaggle.com → T4×2 GPU Notebook, Internet ON         │
│ 2. Attach: anime-faces + Exp11 weights + SDXL cleaned  │
│ 3. 粘贴对应脚本 → 运行                                  │
│ 4. 训练完成后下载输出目录                                │
│ 5. 切换到下一个脚本，重复                                │
└─────────────────────────────────────────────────────────┘
```

**每个脚本自动做的事**：
1. `/kaggle/input/` 中自动检测 Exp11 权重 + SDXL 图片 + 原版数据集
2. 加载 Exp11 的 G+D → 在新的 4K 混合数据上 fine-tune 100 epoch
3. DiffAugment + EMA 全程开启（与 Exp11 一致）
4. 每 25 epoch 保存 checkpoint，支持中断恢复
5. 训练结束后自动计算 FID + Coverage + LPIPS + Diversity + Laplacian + Edge 指标
6. 输出 `metrics.json`, `loss.csv`, `loss_curves.png`, epoch 采样图

**配置 (锁定)**：
- 架构: Exp11 (Width3x G 768ch, SN+Hinge D, 7.77M/1.44M)
- 训练: Fine-tune from Exp11, 100 epochs, batch=32, lr=1e-4, Adam(β₁=0.5, β₂=0.99)
- 增强: DiffAugment (color+translation+cutout), EMA (decay=0.9999)
- 评估: Legacy FID + Coverage（同一代码路径；最多 4,000 张 real 特征和 5,000 张 fake 特征）, LPIPS AlexFeat, Laplacian Variance, Edge Density, Blur Rate

## Step 5: 统一评估 (Kaggle)

```
┌─────────────────────────────────────────────────────────┐
│ 1. 粘贴 04_Evaluation/04_Evaluate_All.py → 运行         │
│ 2. 查看 comparison.csv → 哪组最好?                      │
└─────────────────────────────────────────────────────────┘
```

**判定标准**：
- ✅ 有效: FID < A0 FID **且** Coverage ≥ A0 × 95%
- ❌ 淘汰: Coverage < A0 × 90%
- 🏆 最终推荐: 有效组中 FID 最低者；若无组通过则报告无有效改进
- 同时检查模糊率、边缘密度、LPIPS Diversity

---

## 目录结构

```
SDXL_Controlled_Study/
├── README.md                              ← 你在这里
├── 00_Baseline/
│   └── 00_A0_Baseline_5K.py              ← [备用] Colab 从头训练 A0 (200ep)
├── 01_Production/
│   └── 01_SDXL_Generate.py               ← Step 1: Colab SDXL 生成
├── 02_Cleaning/
│   └── 02_Clean_SDXL.py                  ← Step 2: Colab 清洗
├── 03_Ablation/
│   ├── FT_A0_Baseline_4K_100E.py         ← Step 4: Kaggle A0  (FT, 4K orig)
│   ├── FT_A10_90ori_10sdxl_4K_100E.py    ← Step 4: Kaggle A10 (FT, 3.6K+0.4K)
│   ├── FT_A20_80ori_20sdxl_4K_100E.py    ← Step 4: Kaggle A20 (FT, 3.2K+0.8K)
│   ├── FT_A30_70ori_30sdxl_4K_100E.py    ← Step 4: Kaggle A30 (FT, 2.8K+1.2K)
│   └── FT_A50_50ori_50sdxl_4K_100E.py    ← Step 4: Kaggle A50 (FT, 2K+2K)
├── 04_Evaluation/
│   └── 04_Evaluate_All.py                ← Step 5: Kaggle 统一评估
└── 99_Docs/
    └── EXPERIMENT_PROTOCOL.md             ← 实验设计详细说明
```

## 固定条件 (锁定)

- DCGAN: Exp11 (Width3x G 768ch, SN+Hinge D, DiffAugment+EMA)
- Fine-tune config: 100 epochs, batch=32, lr=1e-4, Adam(β₁=0.5, β₂=0.99)
- IMAGE_SIZE=64, NOISE_DIM=128, SEED=42
- SDXL: Animagine XL 4.0, 1024×1024, DPM++2M Karras 20 steps, CFG=7.0
- DiffAugment: color (brightness+saturation+contrast), translation (0.125), cutout (0.35)
- EMA decay: 0.9999

## 文件流转

```
Exp11 权重 ─────────────────────────────┐
                                        │
Kaggle 原数据集 ──→ A0 FT (4K orig) ────┤ baseline_reference.json
                                        │       │
SDXL 候选 ──→ 清洗 ──→ accepted ────────┤       │ (早停参照)
                    │                   │       │
                    ├──→ A10 FT (3.6K+0.4K) ←──┘
                    ├──→ A20 FT (3.2K+0.8K) ←──┘
                    ├──→ A30 FT (2.8K+1.2K) ←──┘
                    └──→ A50 FT (2K+2K)     ←──┘
                                        │
                                        ├──→ 统一评估 → comparison.csv
```
