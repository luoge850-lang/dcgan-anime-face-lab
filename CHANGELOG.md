# Changelog

## v0.2 - 2026-08-04

- Expanded the snapshot from a final-results selection into a full-process archive covering early tuning, augmentation, module tuning, generator strengthening, and CLIP tuning.
- Added 43 normalized source scripts from the completed historical experiment families.
- Corrected the qualitative sample narrative so the first baseline is the plain no-module DCGAN.
- Added a nine-panel process sample sheet covering each major stage and the B1 data-quality baseline.
- Added the Kaggle data-quality audit: 21,551 paths, 17,029 unique contents, 3,626 exact-duplicate groups, 4,522 redundant copies, and zero bad files.
- Added the B1 formal clean-unique baseline with sanitized entry point, manifest, metrics, loss log, and sample grid.
- Added the SDXL extension boundary and explicitly marked the pilot, cleaning, M20, M50, and unified evaluation as unfinished.
- Replaced the long README sample stack with one compact nine-panel qualitative process figure.
- Prepared the repository for public visibility while keeping the dataset, weights, and open-source license decision separate.

## v0.1 - 2026-08-04

- Frozen the selected DCGAN, Generator-strengthening, and CLIP experiment evidence.
- Added a public-facing README with sample grids, metric caveats, and project status.
- Added representative source scripts, JSON/CSV evidence, and selected model artifacts.
- Added SHA-256 checksums for the copied model files.
- Kept the active internship research workspace separate from this snapshot.
- Replaced the stale headline charts with data-driven SVG figures and added a reproducible figure builder.
- Added explicit comparison scopes and entry points to `results_summary.csv`.
- Reframed the repository as a Kaggle research snapshot rather than a local reproduction claim.
- Added English project-story, runtime, and reproduction-boundary documentation.

## Future work

- Add a unified local `train`, `evaluate`, and `sample` CLI.
- Add a fixed holdout manifest and standardized FID beside the legacy metric.
- Add a small smoke-test suite for model shapes, checkpoint loading, and evaluation output.
- Add a permitted dataset/license statement.
