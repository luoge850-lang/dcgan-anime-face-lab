# Metric Protocol and Interpretation

## Evidence level

The frozen values are useful for longitudinal comparison inside this project. They are not published benchmark values. The repository preserves the historical protocol so that the internship experiments remain auditable, while explicitly separating future standardized evaluation from legacy numbers.

FID was introduced as a distribution-level comparison using features from a pretrained image classifier ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)). Later work shows that finite sample size can bias FID estimates ([Binkowski et al., 2018](https://arxiv.org/abs/1911.07023)) and that low-level resizing choices can materially affect GAN evaluation ([Parmar et al., 2021](https://arxiv.org/abs/2104.11222)). Those limitations are directly relevant to this snapshot.

## Frozen protocols

| Field | Frozen definition | Safe interpretation |
|---|---|---|
| fid_legacy_project | torchvision Inception-v3 pool3 features; project preprocessing; sample counts vary by phase | Compare only within a declared scope. Phase 2/3/5 use the historical 10K-real/10K-fake path; Phase 7 uses up to 4K real and 5K fake from the fixed 4K fine-tuning pool |
| `clip_mmd2_unbiased` | Frozen OpenCLIP ViT-B/32 image features; multi-scale RBF MMD; 2K evaluation features | Compare the matched Phase 5 continuation sweep; do not treat it as FID |
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
6. The Phase 7 A0-A50 script collects fewer real than fake evaluation features because the real pool contains 4K images while N_FID=5000; the asymmetry is constant across the five groups but is not a standardized FID protocol.
7. Phase 7 D_real_eval/D_fake_eval fields are not used for the headline comparison because the A0 record contains an implausible magnitude and the evaluation path is not a clean cross-group discriminator-health protocol.

## Deployment protocol added on 2026-08-19

The deployment archive contains a second, explicitly labeled quality protocol: Standard Inception-v3 FID from the Task 3/4/5 evaluation pipeline. It must not be merged numerically with the historical Legacy FID leaderboard.

| Deployment scope | Main quality metric | Runtime metric | Boundary |
|---|---|---|---|
| Task 3 FP32/FP16/INT8 | Standard FID, blur rate, LPIPS-like diagnostics, Haar-band error | Mean latency and throughput | Fixed deployment calibration/evaluation protocol |
| Task 4 mixed precision | Standard FID, blur rate, numerical error | Mean latency and throughput | Final confirmation used n_fid=5000; screening used n_fid=1000 |
| Task 5 QAT | Standard FID, blur rate, Haar MAE | Batch-32 dynamic benchmark P99 and throughput | Revised acceptance passed; strict high-frequency superiority did not |
| Task 6 service and soak | Not a generation-quality comparison | HTTP P99, RPS, error rate, GPU/RSS/SM samples; fixed batch 1, 60-minute soak, and dynamic batching are separate scopes | Fixed-batch 1–512 and 60-minute soak passed declared checks; no physical crash/saturation boundary; dynamic batching has packaging gaps |

Memory fields also have different meanings: TensorRT reports whole-device CUDA snapshots in Task 2, while ORT/OpenVINO report host RSS. They are retained as separate fields and should not be compared as one common memory unit.

## Required next protocol

Future experiments should add these fields without overwriting the legacy values:

- `fid_standardized` with a documented implementation and preprocessing;
- a fixed train/holdout manifest and SHA-256 digest;
- real and fake sample counts;
- at least three seeds with mean and standard deviation;
- bootstrap or repeated-draw uncertainty for evaluation metrics;
- nearest-neighbor image grids and a memorization check;
- model, dataset-manifest, and code-commit identifiers.
