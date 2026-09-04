# Changelog

## v0.10.2 - 2026-09-04

- Promoted the later 14:18 source-backed deployment figure refresh to the current immutable handoff tag.
- Preserved v0.10.1 as historical instead of moving its existing tag.

## v0.10.1 - 2026-09-04

- Added `tests/__init__.py` and an explicit `-t .` discovery root so the documented unittest command works on the bundled Python runtime as well as older Python versions.
- Kept v0.10 as an immutable predecessor tag; this patch release is the current handoff freeze.
- Refreshed the Stage 7/8 SVG package from the later 14:18 source generation, including the expanded alert-chain diagram and updated generator hash.

## v0.10 - 2026-09-04

- Rebased the public handoff on the latest `dcgan_lab` scan: 2,874 source files, 94 Python scripts, 130 JSON files, 49 SVG figures, 8 checkpoints, 7 TensorRT engines, and 2 ONNX files.
- Added the current source-backed Stage 7/8 deployment figures as four standalone SVGs under `04_visual_assets/deployment_delivery/`.
- Added a sanitized figure-generation manifest and a portable `tools/generate_deployment_report_figures.py` entry point with CLI path overrides.
- Replaced the dated 2026-09-03 handoff manifest with the 2026-09-04 manifest and preserved the previous v0.9 state as historical evidence.
- Excluded redundant PNG renders and the superseded 2026-08-19 deployment audit from the new handoff package; the active Kaggle workspace remains untouched.

## v0.9 - 2026-09-03

- Added the current Stage 7 observability evidence: monitoring stack validation, controlled alert firing/resolution, 5-second resource samples, alert events, and a Grafana screenshot.
- Added the current Stage 8 hot-update/A/B evidence: same-PID candidate loading, traffic-split accuracy, version-level P99, sampled quality metrics, and rollback events.
- Added three readable single-metric Stage 8 SVGs generated from the curated evidence CSVs.
- Added an experiment coverage audit that separates measured results, source-only scripts, historical manifests, and non-comparable metric protocols.
- Added the complete internship Project Engineering Report and a machine-readable source/public inventory for handoff, interviews, and future updates.
- Added a dedicated `00_HANDOFF/` entry point with a file/evidence index, current status, claim boundaries, and handoff manifest.
- Updated deployment source/result indexes and the README freeze language to v0.9-current-state.
- Kept the active Kaggle workspace, dataset, checkpoints, ONNX/TensorRT engines, and large service artifacts outside the public Git history.

## v0.8 - 2026-08-28

- Re-scanned the active `dcgan_lab` evidence boundary and refreshed the interview/GitHub freeze without modifying the active Kaggle source workspace.
- Added current 06E fixed-batch service evidence through concurrency 512, the 60-minute steady soak summary, and the explicit physical-limit boundary.
- Added the 06F dynamic-batching report, manifests, raw summaries, service/soak figures, and a `complete_with_packaging_gaps` status.
- Updated the README, deployment docs, metric protocol, source map, task status, figure notes, and interview description to match the latest evidence.
- Copied the latest full result-folder figure catalog and expanded the public service figure set from the previous 27-chart freeze to 34 result-backed charts.
- Preserved v0.6 and v0.7 tags as historical freezes; v0.8 is the current-state snapshot for the ongoing internship.

## v0.7 - 2026-08-20

- Upgraded the README from an experiment/audit log into an AI/ML research and engineering showcase.
- Added a 30-second project definition, technical route map, dataset/preprocessing summary, Baseline DCGAN architecture, and headline results.
- Reorganized the complete Stage 1–6 narrative around goal, modification, control, result, conclusion, and failure interpretation.
- Added a frozen showcase recipe, visual milestone comparison, deployment system path, quality-speed table, evaluation protocol, reproducibility boundary, limitations, future work, repository map, and interview summary.
- Preserved the v0.6 experiment data, metrics, and stage figures without presenting documentation changes as new training evidence.

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
