# Figure guide

## Canonical interview figures

- `04_visual_assets/interview_results_roadmap.svg` is the headline milestone view. It only includes the staged results used in the README narrative.
- `04_visual_assets/clip_control_sweep.svg` shows the Phase 5 matched continuation control and the distinction between legacy FID and CLIP MMD².
- `04_visual_assets/qualitative_samples_compact.png` is the README-facing nine-panel process contact sheet. Its first Phase 2 panel is the plain no-module baseline, followed by module, generator, CLIP, and data-quality milestones.
- `tools/build_interview_figures.py` regenerates both files from `results_summary.csv` using the Python standard library.

## Latest deployment delivery figures

- `04_visual_assets/deployment_delivery/D07_queue_alert_lifecycle.svg` is the controlled Stage 7 alert firing-to-resolved lifecycle.
- `04_visual_assets/deployment_delivery/D08_ab_traffic_split.svg` is the Stage 8 target-versus-observed B traffic split.
- `04_visual_assets/deployment_delivery/D08_ptq_vs_qat_latency.svg` is the Stage 8 PTQ-INT8 versus QAT-INT8 latency comparison.
- `04_visual_assets/deployment_delivery/D08_ptq_vs_qat_fid.svg` is the same-run sampled FID comparison; it is not merged into the canonical training/deployment leaderboard.
- `tools/generate_deployment_report_figures.py` rebuilds these outputs from the curated Stage 7/8 evidence and accepts `--project-root`, `--results-root`, and `--output-dir` overrides.
- Provenance and hashes are recorded in `03_metrics_and_logs/deployment_optimization/figure_generation_manifest_2026-09-04.json`.

## Archival figures

The older PNG dashboards remain in `04_visual_assets/` as provenance artifacts. They summarize earlier experiment groups and should not be read as a complete leaderboard for the full snapshot. The README intentionally does not use them as headline figures.
