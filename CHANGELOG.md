# Changelog

## v0.6 - 2026-08-19

- Rebuilt the public README around six chronological stages instead of a single visual-results section.
- Regenerated 27 Chinese, single-metric bar/line figures from the scanned `dcgan_lab/results` CSV/JSON evidence.
- Organized regenerated figures into stage folders for early tuning, G/D tuning, G strengthening, CLIP, deployment/quantization, and service stress.
- Excluded deleted loss-trajectory charts from the new public display and recorded the decision in the figure boundary documentation.
- Regenerated the CLIP MMD2 chart from the existing metric files; kept unavailable trajectory charts out of the current evidence.
- Added a stage-to-source map and a v0.6 freeze record.

## v0.5 - 2026-08-19

- Re-scanned the active `dcgan_lab` workspace and refreshed the public snapshot with the current deployment source archive.
- Added ONNX export/fusion, three-backend benchmark, PTQ, layer-sensitivity, mixed-precision, QAT, and service-stress evidence from the latest workspace.
- Added normalized deployment status, quantization, engine, and staged-service summary tables.
- Added a staged Locust stress-run archive covering concurrency 1–128 with zero failures and 5-second GPU/RSS monitoring.
- Added deployment quality-speed and service-stress SVG figures generated from copied evidence tables.
- Updated the README to distinguish complete, partial, revised-acceptance, and staged-only claims.
- Added explicit boundaries for whole-graph fusion, QAT high-frequency claims, and long-run memory-leak testing.

## v0.4 - 2026-08-10

- Rebuilt the public README around the complete experiment progression instead of a final-checkpoint narrative.
- Added an objective one-month audit with safe resume bullets, interview framing, claim boundaries, and compute-efficient next steps.
- Replaced corrupted question-mark characters in user-facing documentation and refreshed the Phase 7 source/protocol files from the latest local workspace.
- Added the Phase 7 FID/Coverage trade-off SVG and verified that the public sample and pilot assets are linked from the README.
- Documented the Phase 7 4K-real/5K-fake evaluation asymmetry and excluded the anomalous A0 discriminator logits from headline conclusions.
- Added a planned generator export/deployment document, clearly marked as future work rather than measured evidence.

## v0.3 - 2026-08-08

- Added the completed Phase 7 controlled SDXL replacement study at a fixed 4K fine-tuning budget.
- Added A0/A10/A20/A30/A50 metrics, loss logs, source scripts, pilot review provenance, and representative sample grids.
- Recorded the negative result: higher SDXL replacement ratios reduced FID and Coverage in the tested setup; the repository does not claim SDXL augmentation as an improvement.
- Added a separate comparison scope so the 4K fine-tuning values are not mixed with the historical 20K Exp11 or 17K B1 results.
- Kept the generated SDXL image pool, archives, checkpoints, and incomplete row-level manual annotations outside Git history.

## v0.2 - 2026-08-04

- Expanded the snapshot from a final-results selection into a full-process archive covering early tuning, augmentation, module tuning, generator strengthening, and CLIP tuning.
- Added 43 normalized source scripts from the completed historical experiment families.
- Corrected the qualitative sample narrative so the first baseline is the plain no-module DCGAN.
- Added a nine-panel process sample sheet covering each major stage and the B1 data-quality baseline.
- Added the Kaggle data-quality audit: 21,551 paths, 17,029 unique contents, 3,626 exact-duplicate groups, 4,522 redundant copies, and zero bad files.
- Added the B1 formal clean-unique baseline with sanitized entry point, manifest, metrics, loss log, and sample grid.
- Added the SDXL extension boundary and explicitly separated the completed 4K replacement study from future synthetic-data work.
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
