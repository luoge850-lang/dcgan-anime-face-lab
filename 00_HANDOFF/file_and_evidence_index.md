# File and evidence index

This index maps the active Kaggle workspace to the curated public snapshot. Historical directory names are preserved for provenance; `R` and `D` aliases are stable conceptual names for discussion and handoff.

## Research families

| Alias | Active source | Public selected scripts | Public evidence |
|---|---|---|---|
| R01 | `experiments/前期调优实验/` | `02_selected_experiments/full_process/phase1_early_tuning/` | `03_metrics_and_logs/dcgan_core/`, `04_visual_assets/stage_figures/01_前期训练与增强/` |
| R02 | `experiments/深度调优实验/` | `02_selected_experiments/full_process/phase2_module_tuning/` | `03_metrics_and_logs/dcgan_core/`, `04_visual_assets/stage_figures/02_G_D模块调优/` |
| R03 | `experiments/G强化实验/` | `02_selected_experiments/full_process/phase3_generator_strengthening/` | `03_metrics_and_logs/dcgan_core/`, `04_visual_assets/stage_figures/03_G强化与训练策略/` |
| R04 | `experiments/CLIP实验/` | `02_selected_experiments/full_process/phase5_clip_tuning/` | `03_metrics_and_logs/dcgan_core/`, `04_visual_assets/stage_figures/04_CLIP调优/` |
| R05 | `experiments/SDXL_Controlled_Study/` | `02_selected_experiments/full_process/phase7_sdxl_controlled_study/` | `03_metrics_and_logs/phase7_sdxl_controlled_study/`, `docs/sdxl_controlled_study.md` |

## Deployment families

| Alias | Active source | Public selected scripts | Public evidence |
|---|---|---|---|
| D01 | `experiments/Deployment_Optimization/01_ONNX_Fusion/` | `02_selected_experiments/full_process/deployment_optimization/01_ONNX_Fusion/` | `03_metrics_and_logs/deployment_optimization/01_ONNX_Fusion/` |
| D02 | `experiments/Deployment_Optimization/02_Engine_Benchmark/` | `.../02_Engine_Benchmark/` | `.../02_Engine_Benchmark/` |
| D03 | `experiments/Deployment_Optimization/03_Quantization/` | `.../03_Quantization/` | `.../03_Quantization/` |
| D04 | `experiments/Deployment_Optimization/04_Quantization_Sensitivity/` | `.../04_Quantization_Sensitivity/` | `.../04_Quantization_Sensitivity/` |
| D05 | `experiments/Deployment_Optimization/05_QAT/` | `.../05_QAT/` | `.../05_QAT/` |
| D06 | `experiments/Deployment_Optimization/06_Service_Stress/` | `.../06_Service_Stress/` | `.../06_Service_Stress/` |
| D07 | `experiments/Deployment_Optimization/07_MLOps_Observability/` | `.../07_MLOps_Observability/` | `03_metrics_and_logs/deployment_optimization/07/07_MLOps_Observability/evidence/` |
| D08 | `experiments/Deployment_Optimization/08_Model_Hot_Update_AB/` | `.../08_Model_Hot_Update_AB/` | `03_metrics_and_logs/deployment_optimization/08_Model_Hot_Update_AB/evidence/` |

The latest public Stage 7/8 figure outputs are collected in `04_visual_assets/deployment_delivery/`. Their source hashes and metric scope are recorded in `03_metrics_and_logs/deployment_optimization/figure_generation_manifest_2026-09-04.json`; the portable rebuild entry point is `tools/generate_deployment_report_figures.py`.

## Evidence reading rules

1. Read the stage README before reading a CSV; it defines the scope and acceptance criteria.
2. Read the current project-level manifest before using deployment status.
3. Use `03_metrics_and_logs/stage_figures_map.csv` to identify the source scope for a chart.
4. Use `docs/experiment_coverage_audit_2026-09-03.md` to determine whether a value is measured, partial, source-only, or non-comparable.
5. Treat old manifests as historical evidence if a revised manifest exists; do not delete them merely because the status changed.

## Excluded artifact classes

The public snapshot intentionally omits raw images, generated evaluation pools, large checkpoints, ONNX binaries, TensorRT engines, optimizer states, and uncontrolled service logs. Their absence is part of the handoff boundary, not missing documentation.
