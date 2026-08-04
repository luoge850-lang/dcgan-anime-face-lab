# Update workflow for ongoing experiments

The active internship workspace and this public snapshot have different roles.

## Active workspace

Keep all exploratory scripts, raw logs, datasets, checkpoints, and unfinished hypotheses in the active workspace. Do not copy the entire directory into the public repository.

## Public snapshot

When an experiment is complete and approved for publication:

1. Copy only the final script/config into `02_selected_experiments/`.
2. Copy its metrics JSON and small CSV logs into `03_metrics_and_logs/`.
3. Add one representative sample grid or comparison plot to `04_visual_assets/`.
4. Add one row to `results_summary.csv`.
5. Update the limitations or status section in `README.md`.
6. Add a dated entry to `CHANGELOG.md`.
7. Review data, IP, secrets, and binary-file rules.
8. Commit the curated change and create a tag when it represents a stable milestone.

Suggested commit messages:

```text
experiment: add standardized holdout evaluation
results: add Exp12 data-mixture ablation
docs: clarify FID protocol and limitations
release: freeze v0.2 public snapshot
```
