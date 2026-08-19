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
- `phase6_audit_original_dataset.py` and `phase6_b1_formal_clean_unique_17k.py`: data-quality audit and formal exact-unique B1 baseline from the active Kaggle workspace.
- `full_process/phase7_sdxl_controlled_study/`: completed fixed-budget SDXL replacement study, including A0/A10/A20/A30/A50 fine-tuning scripts and the unified evaluation script.

The `02_selected_experiments/full_process/` archive adds the completed source families that were previously omitted from the interview-facing selection: Phase 1 early tuning, Phase 2 module tuning, Phase 3 generator strengthening, and the complete Phase 5 CLIP sweep including C2.

The remaining historical metrics are retained under `03_metrics_and_logs` for traceability, but their full training scripts are not part of the public-facing core.

The repository contains several baselines with different purposes. Use `docs/baseline_map.md` when presenting them; `00_baseline`, Exp11/B0, and B1 are not interchangeable checkpoints or data protocols.

Phase 7 is a separate fine-tuning comparison scope. Its A0/A10/A20/A30/A50 results should be read together with `docs/sdxl_controlled_study.md`, not appended to the older 20K FID leaderboard.

## Deployment source added in v0.5

`02_selected_experiments/full_process/deployment_optimization/` mirrors the current deployment source families found in the active workspace:

- ONNX export, ORT optimization, manual BN folding, subpixel probe, and targeted TensorRT fusion;
- ORT/TensorRT/OpenVINO benchmark and profiler entry points;
- FP32/FP16/INT8 PTQ and evaluation;
- layer sensitivity, mixed precision, and final confirmation;
- FakeQuantize QAT training/evaluation;
- HTTP service preflight, staged Locust stress, 5-second monitoring, soak-test entry point, and local report builder.

The source archive includes scripts even when the corresponding measured result is partial or absent. The public claim status is controlled by `03_metrics_and_logs/deployment_optimization/deployment_task_status.csv`.

The additional Phase 3 sources `12_G_AA_ADA_EMA_21K.py` and `DCGAN_Improve_Standalone_21K.py` are included as source evidence. The former does not have a matching final result row in the active results tree and is therefore not presented as a completed quantitative experiment.
