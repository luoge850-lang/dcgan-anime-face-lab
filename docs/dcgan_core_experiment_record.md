# DCGAN Core Experiment Record

This document isolates the training and modeling line of the project. It intentionally separates the DCGAN core from the later SDXL data study and the deployment/quantization work.

## Scope

- Task: unconditional anime-face generation at 64 x 64 resolution.
- Runtime: Kaggle GPU experiments.
- Main historical metric: the project's Legacy Inception-v3 FID protocol.
- Evidence source: [`dcgan_core_metrics.csv`](../03_metrics_and_logs/dcgan_core/dcgan_core_metrics.csv), copied from `dcgan_lab/results/figures/`.
- Visual source: the stage-organized SVGs regenerated from `dcgan_lab/results/`; source-folder availability is audited separately.

The numbers below are an experiment record, not a universal leaderboard. Most historical runs use one seed and a real-image distribution drawn from the training pool. The record therefore supports within-scope engineering conclusions, not statistical claims about generalization or SOTA performance.

## Experiment map

```mermaid
flowchart LR
    A[Plain DCGAN baseline] --> B[Epoch budget]
    B --> C[Input preprocessing and augmentation]
    C --> D[Generator / discriminator module tuning]
    D --> E[Generator capacity and training-scale study]
    E --> F[CLIP continuation and lambda sweep]
    F --> G[Candidate checkpoint and data-quality branch]
```

## Phase 0: define the baseline correctly

The Phase 2 `00_baseline` is the clean architectural control for the module study: it has no newly added deep module, while retaining the selected Phase 1 input policy. Its recorded Legacy FID is **109.40**.

This should be described as **"plain baseline / no added deep module"**, not as "the model used no preprocessing at all." That distinction matters because the input policy was already selected in Phase 1.

Archived source: [`phase2_module_tuning/00_baseline.py`](../02_selected_experiments/full_process/phase2_module_tuning/00_baseline.py).

## Phase 1: training budget and input policy

### Epoch budget

| Run | Epochs | Legacy FID | Reading |
|---|---:|---:|---|
| `exp1_epoch_050` | 50 | 184.11 | early checkpoint |
| `exp1_epoch_100` | 100 | 136.78 | large early improvement |
| `exp1_epoch_200` | 200 | 113.97 | continued improvement |
| `exp1_epoch_300` | 300 | 105.34 | lower marginal gain than earlier intervals |

Within this recipe, increasing the budget from 50 to 300 epochs lowered the recorded FID by about 42.8%. This is evidence for a training-budget effect under the tested setup; it does not prove that 300 epochs is globally optimal.

Figure: [`01_训练轮数_FID.svg`](../04_visual_assets/stage_figures/01_前期训练与增强/01_训练轮数_FID.svg).

### Augmentation candidates

| Candidate | Legacy FID | Interpretation |
|---|---:|---|
| Sharpen | 121.11 | lowest recorded candidate in this group |
| Flip + color + sharpen | 121.54 | close to sharpen; not a decisive separation |
| Flip + sharpen | 124.43 | competitive but weaker than the two above |
| Flip | 127.89 | weaker than the selected candidates |
| Color | 136.12 | weaker under this run |
| Blur | 141.15 | associated with a large FID penalty |
| Denoise | 155.81 | weakest listed candidate |

The full nine-row table, including the remaining combinations, is retained in the copied metric catalog. The defensible conclusion is that sharpening was the best recorded candidate in this small search, not that it is a generally optimal augmentation policy.

Figure: [`02_数据增强_FID.svg`](../04_visual_assets/stage_figures/01_前期训练与增强/02_数据增强_FID.svg).

Archived sources: [`phase1_exp1_epoch_study.py`](../02_selected_experiments/full_process/phase1_early_tuning/phase1_exp1_epoch_study.py), [`phase1_exp2_augmentation_ablation.py`](../02_selected_experiments/full_process/phase1_early_tuning/phase1_exp2_augmentation_ablation.py), and [`phase1_exp3_combination.py`](../02_selected_experiments/full_process/phase1_early_tuning/phase1_exp3_combination.py).

## Phase 2: deep module and adversarial-objective tuning

This phase contains two different questions and should not be reduced to one flat ranking.

### Generator-side feature modules

| Variant | Legacy FID | Evidence reading |
|---|---:|---|
| No-added-module baseline | 109.40 | architectural control |
| G + SENet, 1 layer | 96.76 | best listed G-side single-module candidate |
| G + Laplacian | 98.67 | improved FID, but not a universal feature-module result |
| G + Wavelet | 106.14 | close to the control under this run |
| G + FFT | 108.40 | little FID change in this run |
| G + Canny | 109.02 | little FID change in this run |

### Discriminator-side stabilization

| Variant | Legacy FID | Evidence reading |
|---|---:|---|
| D + SN | 99.19 | stabilization candidate |
| D + Hinge | 129.16 | weaker than the control in this run |
| D + SN + Hinge | 96.25 | stronger candidate than either isolated change here |
| D + SN + Hinge + R1 | 89.92 | best recorded Phase 2 candidate |

The result supports a practical engineering observation: the discriminator-side stabilization path was more promising than simply adding every proposed feature module. It does not isolate the causal contribution of every component because the full set of runs is not a perfectly balanced factorial design.

Figures: [`05_G端模块_FID.svg`](../04_visual_assets/stage_figures/02_G_D模块调优/05_G端模块_FID.svg) and [`06_D端模块_FID.svg`](../04_visual_assets/stage_figures/02_G_D模块调优/06_D端模块_FID.svg).

Archived sources: [`phase2_module_tuning/`](../02_selected_experiments/full_process/phase2_module_tuning/) and the selected public entry points [`phase2_08_g_laplacian.py`](../02_selected_experiments/phase2_08_g_laplacian.py), [`phase2_13_d_sn_hinge.py`](../02_selected_experiments/phase2_13_d_sn_hinge.py), and [`phase2_19_d_sn_hinge_r1.py`](../02_selected_experiments/phase2_19_d_sn_hinge_r1.py).

## Phase 3: generator capacity, data scale, and regularization

| Variant | Legacy FID | What changed | Causal scope |
|---|---:|---|---|
| G Width x2 | 78.75 | generator capacity | relatively focused |
| G Width x3 | 59.00 | generator capacity | relatively focused |
| G Width x4 | 63.66 | generator capacity | relatively focused |
| Width x3 + 20K data | 49.17 | data scale plus the Width x3 recipe | combined change |
| Width x3 + 20K + Laplacian | 53.74 | adds a Laplacian term | combined change |
| Width x3 + DiffAugment + EMA + 20K | 38.88 | capacity, data scale, augmentation, EMA | multi-factor candidate |

The important interpretation is not "one module produced FID 38.88." The 38.88 result is a later combined candidate whose recipe changes several factors at once. The figure and raw metrics preserve the path from Width x3 to 20K data and then to DiffAugment + EMA.

Figure: [`03_G结构强化_FID.svg`](../04_visual_assets/stage_figures/03_G强化与训练策略/03_G结构强化_FID.svg) and the supporting diversity chart [`04_G结构强化_LPIPS.svg`](../04_visual_assets/stage_figures/03_G强化与训练策略/04_G结构强化_LPIPS.svg).

The source archive also contains later scripts [`12_G_AA_ADA_EMA_21K.py`](../02_selected_experiments/full_process/phase3_generator_strengthening/12_G_AA_ADA_EMA_21K.py) and [`DCGAN_Improve_Standalone_21K.py`](../02_selected_experiments/full_process/phase3_generator_strengthening/DCGAN_Improve_Standalone_21K.py). At the time of this snapshot, they are **source-only evidence**: no matching final row is present in the DCGAN metric catalog, so they are not used to claim a new measured best result.

## Phase 4: CLIP continuation and lambda sweep

`C0` is the no-CLIP continuation control. It is essential because the CLIP runs start from a prior checkpoint and may improve simply through additional training.

| Run | CLIP lambda | Legacy FID | CLIP MMD2 | Reading |
|---|---:|---:|---:|---|
| C0 continuation | 0 | 33.7846 | 0.042832 | no-CLIP control |
| C1 | 0.0100 | 33.4114 | 0.042657 | lowest FID in this sweep |
| C2 | 0.0250 | 33.6687 | 0.042298 | lower MMD2, not lower FID |
| C3 | 0.0500 | 33.4718 | 0.041856 | lower MMD2, near-best FID |
| C4 | 0.1000 | 33.6231 | 0.040990 | lowest MMD2 in this sweep |

The metrics disagree in a useful way: C1 has the lowest FID, while C4 has the lowest CLIP MMD2. Therefore the result should be described as a perceptual trade-off study, not as proof that a larger CLIP weight is better.

Figures: [`07_CLIP_FID.svg`](../04_visual_assets/stage_figures/04_CLIP调优/07_CLIP_FID.svg) and [`08_CLIP_MMD2.svg`](../04_visual_assets/stage_figures/04_CLIP调优/08_CLIP_MMD2.svg). The MMD2 chart was regenerated from the existing C0-C4 metric files even though its old SVG was absent from the source figure directory.

Archived sources: [`phase5_clip_tuning/`](../02_selected_experiments/full_process/phase5_clip_tuning/) and [`clip_C0_no_clip_control.py`](../02_selected_experiments/clip_C0_no_clip_control.py).

## Objective audit

### Claims supported by the current evidence

- Longer training improved FID within the epoch study recipe.
- Sharpening was the best recorded candidate in the listed Phase 1 augmentation search.
- The D SN + Hinge + R1 path was the strongest recorded Phase 2 candidate.
- Generator capacity, data scale, DiffAugment, and EMA produced the strongest later combined candidate in the archived DCGAN metrics.
- CLIP lambda changes trade off Legacy FID and CLIP MMD2; C0 is a necessary control.

### Claims that require careful wording

- The historical FID values are not directly comparable to the deployment Standard FID values.
- The 38.88 candidate is not a single-factor ablation.
- Loss curves describe optimization behavior, but loss magnitudes are not comparable across BCE, Hinge, R1, CLIP, or auxiliary-loss objectives.
- Diversity/LPIPS, Laplacian variance, and edge ratios are supporting diagnostics, not substitutes for a standardized held-out evaluation.
- Single-seed runs and the absence of a strict held-out real-image set limit the strength of generalization claims.

### Claims this snapshot does not support

- A universal best architecture across all rows.
- SOTA performance or publication-level statistical significance.
- Deleted 09–16 training-trajectory figures as if they were current evidence; 08 is now separately regenerated from the C0-C4 metric files.
- A final production checkpoint for the still-evolving internship project.

## Figure provenance audit

The source directory's `figure_manifest.json` lists 42 expected SVGs, but the directory currently contains 33 SVG files. Seven DCGAN loss-curve entries are marked as intentionally removed according to the project owner; the CLIP MMD2 chart and the final CLIP representative MMD2 entry are absent from the source figure directory. The public stage rebuild regenerates the non-loss C0-C4 MMD2 chart from raw metric files, while the missing loss/trajectory charts remain excluded. All 16 expected DCGAN entries are recorded in [`figure_audit.csv`](../03_metrics_and_logs/dcgan_core/figure_audit.csv).

The public snapshot therefore embeds only the seven verified DCGAN charts copied from the requested source directory. This is a limitation of the current evidence package, not a claim that the corresponding underlying CSV files never existed.

## Reproduction boundary

The record is designed for interview and application review. Re-running the original Kaggle experiments requires the original dataset mounts, checkpoints, GPU environment, and evaluation assets. The snapshot preserves source paths, metrics, figure provenance, and claim boundaries, but it is not a one-command local reproduction package.
