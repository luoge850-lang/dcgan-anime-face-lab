# Generator-strengthening experiments

This phase tests whether generator-side capacity and feature-processing changes help after discriminator stabilization. The selected scripts are:

- `phase3_03_g_width3x.py`: generator width milestone;
- `phase3_04_g_width4x.py`: capacity-ceiling negative result;
- `phase3_06_g_width3x_resg_pre.py`: residual-generator negative result;
- `phase3_09_g_width3x_20k.py`: data-scale experiment;
- `phase3_10_g_width3x_20k_laplacian.py`: controlled auxiliary-loss comparison;
- `final_exp11_diffaug_ema.py`: frozen DiffAugment + EMA baseline.

The experiment selection keeps both improvements and negative results. The evidence is not a universal architecture ranking because dataset size, training phase, and metric comparison scope vary. Use `results_summary.csv` and `metric_protocol.md` as the source of truth.
