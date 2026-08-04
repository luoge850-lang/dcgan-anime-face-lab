# Figure guide

## Canonical interview figures

- `04_visual_assets/interview_results_roadmap.svg` is the headline milestone view. It only includes the staged results used in the README narrative.
- `04_visual_assets/clip_control_sweep.svg` shows the Phase 5 matched continuation control and the distinction between legacy FID and CLIP MMD².
- `04_visual_assets/qualitative_samples_compact.png` is the README-facing nine-panel process contact sheet. Its first Phase 2 panel is the plain no-module baseline, followed by module, generator, CLIP, and data-quality milestones.
- `tools/build_interview_figures.py` regenerates both files from `results_summary.csv` using the Python standard library.

## Archival figures

The older PNG dashboards remain in `04_visual_assets/` as provenance artifacts. They summarize earlier experiment groups and should not be read as a complete leaderboard for the full snapshot. The README intentionally does not use them as headline figures.
