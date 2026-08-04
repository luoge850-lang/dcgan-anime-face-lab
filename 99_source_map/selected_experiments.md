# Why these experiments were selected

The snapshot keeps representative experiments that make the project story understandable:

- `phase2_00_baseline`: starting point.
- `phase2_13_d_sn_hinge`: discriminator stabilization milestone.
- `phase2_19_d_sn_hinge_r1`: representative regularization comparison.
- `phase2_08_g_laplacian`: representative auxiliary-loss result.
- `phase3_03_g_width3x`: generator capacity milestone.
- `phase3_04_g_width4x`: useful negative result showing a capacity ceiling.
- `phase3_06_g_width3x_resg_pre`: useful negative result for residual design under the resource constraint.
- `phase3_09_g_width3x_20k`: data-scale milestone.
- `phase3_10_g_width3x_20k_laplacian`: controlled negative result against the best data-scale baseline.
- `01_public_core/final_exp11_diffaug_ema.py`: current frozen DiffAugment + EMA baseline.
- `clip_E0` through `clip_C4`: formal control and CLIP lambda sweep.

The remaining historical metrics are retained under `03_metrics_and_logs` for traceability, but their full training scripts are not part of the public-facing core.
