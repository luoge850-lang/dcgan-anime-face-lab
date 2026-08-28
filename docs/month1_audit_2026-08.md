# One-month project audit and application packaging

## Snapshot update: 2026-08-19

The active workspace was re-scanned after the initial month-one packaging. It now contains 2,504 files, 87 Python scripts, 99 JSON records, and 100 CSV files. All 87 Python scripts passed an AST parse; the only JSON parse exception was a UTF-8 BOM in a deployment manifest, which is valid JSON after BOM-aware decoding.

The deployment phase now has measured evidence beyond preflight: ONNX/engine/quantization artifacts, layer sensitivity and QAT results, fixed-batch service tests through concurrency 512 with zero failures, a 60-minute steady soak with zero failures, and a dynamic-batching comparison through concurrency 128. The snapshot reports no observed physical crash/saturation boundary; the soak is a workload-scoped leak screen, and the dynamic-batching archive is marked complete with packaging gaps.

## Audit scope

The local internship snapshot was scanned on 2026-08-10:

| Evidence type | Count |
|---|---:|
| Total files | 2,353 |
| Python scripts | 60 |
| JSON records | 71 |
| CSV logs | 56 |
| PNG visual outputs | 2,137 |
| PTH checkpoints | 8 |
| Source size | approximately 454 MiB |

All 60 Python files passed an AST syntax parse, and all 71 JSON plus 56 CSV files parsed successfully. The public repository intentionally publishes only a curated subset: source entry points, small metrics/logs, figures, manifests, and sample grids. Raw image pools, checkpoints, caches, and platform dumps remain outside Git history.

## Objective assessment

### What is genuinely strong

1. **The project has a real experimental arc.** It is not just a DCGAN implementation. It contains epoch-budget selection, augmentation ablations, module/discriminator tests, generator capacity scaling, data scaling, DiffAugment/EMA stabilization, a no-CLIP control, CLIP-MMD tuning, data deduplication, and a fixed-budget SDXL mixture study.
2. **The controls are getting better over time.** The Phase 2 no-added-module baseline is separated from the later Exp11 historical reference. The SDXL study fixes the checkpoint, 4K budget, epoch count, optimizer, seed, and augmentation policy while varying the replacement ratio.
3. **Negative results are preserved.** Wavelet, FFT, edge, several G variants, and the SDXL mixture study are not removed just because they did not win.
4. **The data audit is valuable.** 21,551 image paths became 17,029 unique SHA-256 contents, with 4,522 redundant copies and zero bad files. That is a concrete data-quality discovery with a follow-up control.
5. **The work is resource-aware.** Kaggle-only execution, fixed budgets, checkpointing, EMA, and one-cell/one-script entry points show practical engineering judgment.

### What the project does not yet prove

- It does not establish a new GAN architecture or state-of-the-art result.
- It does not establish strict generalization because the real evaluation pool is primarily drawn from the training distribution.
- It does not provide seed variance or confidence intervals for most comparisons.
- Its legacy FID is not directly comparable with published clean-fid or torch-fidelity numbers.
- DiffAugment and EMA were introduced together in Exp11, so their separate causal effects are not identified.
- Phase 7 FID uses up to 4,000 real and 5,000 fake features because the 4K pool is smaller than N_FID=5000. The same asymmetry applies to A0–A50, which supports within-study ranking but weakens external interpretation.
- The Phase 7 A0 discriminator evaluation logits contain an implausible magnitude. They should not be used as a headline result; the public README excludes them from the main comparison.

These limitations reduce the research claim, not the value of the internship. For an application, the strongest story is experimental design, evidence hygiene, and decision-making under compute constraints.

## Recommended framing by target role

### Machine learning / research intern

> Designed and executed a staged PyTorch DCGAN ablation study in Kaggle for unconditional 64x64 anime-face generation, covering discriminator stabilization, generator capacity, data scale, DiffAugment, EMA, CLIP-MMD, and synthetic-data mixing. Built matched controls, preserved negative results, and audited 21,551 image paths to 17,029 unique SHA-256 contents.

### Computer vision / generative modeling intern

> Investigated resource-constrained GAN training through controlled changes to the G/D balance and data pipeline. The selected Width x3 + 20K + DiffAugment + EMA recipe reached project-legacy FID 38.88 within its fixed protocol; a subsequent fixed-budget SDXL replacement study found that higher synthetic-data ratios degraded FID and feature coverage.

### ML systems / applied engineering intern

> Built Kaggle-oriented one-shot training/evaluation entry points with automatic dataset and checkpoint discovery, EMA checkpoints, training logs, JSON metrics, sample grids, and snapshot integrity tests. Kept large artifacts outside Git and documented the reproduction boundary.

Only use the bullets that match what you personally designed, ran, debugged, and can explain. Do not claim authorship of every line if the implementation was AI-assisted; claim the experiment design, review, testing, interpretation, and integration work you actually performed.

## How to explain the project in an interview

### 90-second structure

1. **Problem:** generate 64x64 anime faces on a constrained Kaggle GPU budget.
2. **Baseline:** use a plain no-added-module DCGAN as the Phase 2 architectural control; keep its selected Phase 1 input augmentation explicit.
3. **Investigation:** test D stabilization, G capacity, data scale, DiffAugment, EMA, and CLIP-MMD with controls.
4. **Evidence:** report FID only within a comparison scope and pair it with coverage, sharpness, diversity, and training-health diagnostics.
5. **Data quality:** discover exact duplicates and establish a clean-unique B1 baseline.
6. **Decision:** run a 0–50% SDXL replacement ablation; stop synthetic-data mixing after the controlled study showed lower FID and coverage at higher ratios.
7. **Limitation:** one seed, Kaggle-only, legacy FID, no strict holdout.

### Questions you must be able to answer

- Why is the Phase 2 baseline called no-added-module rather than no-augmentation?
- Why is FID 38.88 not directly comparable with FID 45.07?
- Why is C1 not automatically the final model when it has the lowest local legacy FID?
- What did the duplicate audit change in the interpretation of the project?
- Why is A0 the baseline for the SDXL study?
- What does Coverage measure, and why can a lower FID still be unacceptable?
- What would you fix first if more compute became available?

## Highest-return next steps without restarting old runs

1. Keep all existing metrics immutable and add a unified standardized evaluation as a new protocol.
2. Add a row-level, permission-safe accepted-image manifest for any future synthetic-data run.
3. Add nearest-neighbor grids and a memorization check before making generalization claims.
4. If compute permits, repeat only the key final comparison with three seeds; do not rerun every historical experiment.
5. Separate DiffAugment and EMA in a small follow-up ablation.
6. Add a model/code/manifest identifier to every future result record.
7. For deployment work, export only the generator, validate ONNX/PyTorch numerical agreement, and benchmark latency separately from image quality.

## Public packaging rule

The public repository should read like an evidence-backed lab notebook:

- README: one-minute story, canonical numbers, visual figures, limitations.
- results_summary.csv: one row per selected experiment with a comparison scope.
- docs: protocol, baseline relationships, audit findings, and interview story.
- metrics/logs: enough data to check claims without publishing raw datasets or weights.
- visual assets: one compact qualitative figure plus a few decision-relevant charts.
- source map and changelog: what was selected, when, and why.

The project is already above the level of a toy tutorial. Its ceiling for applications now depends less on adding more modules and more on whether you can defend the controls, metric boundaries, negative findings, and engineering trade-offs clearly.
