# Metric Protocol and Interpretation

## Evidence level

The frozen values are useful for longitudinal comparison inside this project. They are not published benchmark values. The repository preserves the historical protocol so that the internship experiments remain auditable, while explicitly separating future standardized evaluation from legacy numbers.

FID was introduced as a distribution-level comparison using features from a pretrained image classifier ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)). Later work shows that finite sample size can bias FID estimates ([Binkowski et al., 2018](https://arxiv.org/abs/1911.07023)) and that low-level resizing choices can materially affect GAN evaluation ([Parmar et al., 2021](https://arxiv.org/abs/2104.11222)). Those limitations are directly relevant to this snapshot.

## Frozen protocols

| Field | Frozen definition | Safe interpretation |
|---|---|---|
| `fid_legacy_project` | torchvision Inception-v3 pool3 features; project preprocessing; 10K real + 10K fake | Compare experiments only when dataset size, model choice, preprocessing, and evaluation path are held constant |
| `clip_mmd2_unbiased` | Frozen OpenCLIP ViT-B/32 image features; multi-scale RBF MMD²; 2K evaluation features | Compare the matched Phase 5 continuation sweep; do not treat it as FID |
| `LPIPS_legacy_AlexNet_feature_distance` | Historical AlexNet feature MSE proxy | Do not call this calibrated LPIPS |
| `Diversity`, `Laplacian_Variance`, `Edge_Density` | Project-defined auxiliary diagnostics | Use as supporting evidence, not as a replacement for distributional evaluation |

## Data-quality protocol added after the original snapshot

The Kaggle input audit found 21,551 image paths, 17,029 unique SHA-256 contents, 3,626 exact-duplicate groups, 4,522 redundant copies, and zero bad files. `B1_Formal_CleanUnique_17K` trains the Exp11 recipe on one path per exact content and reports legacy project FID `45.07`.

This B1 value must not be plotted as a direct architecture improvement over the historical Exp11 `38.88`: the training pool changed. It is a formal data-cleaning baseline for the planned SDXL mixture study. The audit record and manifest are under `03_metrics_and_logs/phase6_data_audit/`.

## Known limitations

1. Historical real samples are primarily drawn from the training distribution rather than a strict unseen holdout.
2. Dataset size and training protocol change across phases, so the result table is organized by comparison scope rather than sorted as one universal leaderboard.
3. Fake samples are stochastic; a single 10K draw does not provide an uncertainty interval.
4. The legacy pipeline is not guaranteed to match clean-fid, torch-fidelity, or another implementation.
5. Most historical results use one seed, so small changes should not be described as causal without matched controls and replication.

## Required next protocol

Future experiments should add these fields without overwriting the legacy values:

- `fid_standardized` with a documented implementation and preprocessing;
- a fixed train/holdout manifest and SHA-256 digest;
- real and fake sample counts;
- at least three seeds with mean and standard deviation;
- bootstrap or repeated-draw uncertainty for evaluation metrics;
- nearest-neighbor image grids and a memorization check;
- model, dataset-manifest, and code-commit identifiers.
