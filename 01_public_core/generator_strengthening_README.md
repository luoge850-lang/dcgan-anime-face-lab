# Generator-strengthening experiments

This phase tests whether generator-side capacity and feature-processing changes help after discriminator stabilization. The public source folder preserves the following selected generator-strengthening entry points:

- `03_G_Width3x.py`: generator width milestone;
- `04_G_Width4x.py`: capacity-ceiling negative result;
- `06_G_Width3x_ResG_pre.py`: residual-generator negative result;
- `09_G_Width3x_20K.py`: data-scale experiment;
- `10_G_Width3x_20K_Laplacian.py`: auxiliary-loss comparison;
- `11_G_DiffAug_EMA_20K.py`: multi-factor DiffAugment + EMA candidate;
- `12_G_AA_ADA_EMA_21K.py`: source-only follow-up with no matching result directory in this freeze.

The folder also contains the `00_baseline.py` control and other exploratory scripts. The current result catalog contains entries `00`–`11`; `00` has no metric rows and `12` has no result directory, so neither is presented as a measured winner.

The experiment selection keeps both improvements and negative results. The evidence is not a universal architecture ranking because dataset size, training phase, and metric comparison scope vary. Use [`dcgan_core/全实验指标汇总.csv`](../03_metrics_and_logs/dcgan_core/全实验指标汇总.csv), [`results_summary.csv`](../results_summary.csv), and [`metric_protocol.md`](../docs/metric_protocol.md) as the source of truth. The complete coverage and claim boundary are recorded in [`experiment_coverage_audit_2026-09-03.md`](../docs/experiment_coverage_audit_2026-09-03.md).

