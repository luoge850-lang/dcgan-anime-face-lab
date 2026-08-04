# Resource-Constrained DCGAN for Anime-Face Generation

![Status: ongoing research](https://img.shields.io/badge/status-ongoing%20research-orange)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)
![Kaggle](https://img.shields.io/badge/runtime-Kaggle%20GPU-20beff)
![Resolution](https://img.shields.io/badge/output-64x64-blue)
![Snapshot](https://img.shields.io/badge/snapshot-v0.1-lightgrey)

> This repository is a curated, frozen snapshot of an ongoing DCGAN internship project. The experiments were run in Kaggle, while the active research workspace remains separate. The snapshot is designed for technical review and interview discussion; it is not yet a claim of a locally reproducible or state-of-the-art generator.

## Reviewer summary

The project studies unconditional 64x64 anime-face generation with PyTorch under limited GPU and batch-size constraints. The main research question was which changes improve the adversarial game most reliably:

- discriminator stabilization with spectral normalization and Hinge loss;
- generator capacity scaling;
- data scaling from 10K to 20K images;
- DiffAugment and exponential moving average (EMA);
- CLIP image-feature distribution matching with an MMD objective.

The strongest project-level lesson is that data scale and adversarial stabilization were more reliable than repeatedly adding generator modules. The CLIP sweep is preserved as a controlled continuation study, not as proof that CLIP caused the entire downstream FID reduction.

## Results at a glance

| Stage | Change | Data | Legacy project FID |
|---|---|---:|---:|
| Phase 2 | Initial baseline | 8K | 109.40 |
| Phase 2 | Discriminator SN + Hinge | 8K | 96.25 |
| Phase 3 | Generator Width x3 | 10K | 59.00 |
| Phase 3 | Width x3 with 20K images | 20K | 49.17 |
| Phase 3 | DiffAugment + EMA | 20K | 38.88 |
| Phase 5 | No-CLIP continuation control | 20K | 33.78 |
| Phase 5 | CLIP MMD, lambda=0.01 | 20K | 33.41 |

The complete curated table, entry points, and comparison scopes are in [`results_summary.csv`](results_summary.csv). The two headline figures below are generated from that table by [`tools/build_interview_figures.py`](tools/build_interview_figures.py).

![DCGAN experiment roadmap](04_visual_assets/interview_results_roadmap.svg)

![Matched CLIP continuation sweep](04_visual_assets/clip_control_sweep.svg)

### Qualitative samples

| Milestone | Generated sample grid |
|---|---|
| Width x3 | ![Width x3 samples](04_visual_assets/milestone_03_width3x_epoch200.png) |
| 20K images | ![20K samples](04_visual_assets/milestone_09_20k_epoch200.png) |
| DiffAugment + EMA | ![DiffAugment and EMA samples](04_visual_assets/milestone_11_diffaug_ema_epoch200.png) |

## How to interpret the metrics

The headline field is named `fid_legacy_project` to prevent it from being confused with a standardized benchmark. It uses the historical torchvision Inception-v3 pool3 pipeline with 10K real and 10K fake samples. Historical real samples are primarily drawn from the training distribution, not a strict unseen holdout, and fake sampling introduces evaluation noise.

FID is useful for longitudinal comparison when the protocol is held constant, but it is sensitive to feature-network, resizing, preprocessing, and finite-sample choices. The project therefore keeps old values unchanged and plans to add a separately named standardized FID beside them. See [`metric_protocol.md`](metric_protocol.md) for the exact boundary and the next evaluation plan.

The historical `LPIPS` fields in older logs are AlexNet feature-distance proxies, not calibrated LPIPS. The CLIP sweep reports both FID and CLIP MMD because the two objectives do not select exactly the same checkpoint: C1 has the lowest local legacy FID, while C4 has the lowest CLIP MMD².

## Reproduction boundary

This is a Kaggle experiment archive, not a one-command local package yet.

- Training was performed in Kaggle GPU notebooks/scripts.
- The dataset is intentionally not included.
- Model weights are kept outside Git history and remain local until redistribution rights are confirmed.
- Several historical scripts retain Kaggle-oriented defaults and expect an attached dataset and, for CLIP experiments, an attached weights Dataset.
- `requirements.txt` is a compatibility floor, not a lockfile.
- Exact environment evidence captured during the CLIP runs is recorded in [`docs/runtime_and_dependencies.md`](docs/runtime_and_dependencies.md).

Representative Kaggle entry points:

```text
02_selected_experiments/phase2_00_baseline.py
01_public_core/final_exp11_diffaug_ema.py
02_selected_experiments/clip_E0_formal_eval.py
02_selected_experiments/clip_C1_lambda_001.py
```

To reproduce a historical result responsibly, attach a permitted dataset, use the matching entry point and configuration, record the dataset manifest hash, and preserve the runtime metadata. The detailed checklist is in [`docs/reproduction_boundary.md`](docs/reproduction_boundary.md).

## Snapshot integrity checks

These checks validate the curated evidence without requiring a GPU or the private dataset:

```powershell
python -m unittest discover -s tests -v
python tools/build_interview_figures.py
```

The first command is also run by GitHub Actions on pushes and pull requests.

## Repository map

| Path | Purpose |
|---|---|
| `01_public_core/` | Baseline, final Exp11 code, and English experiment guides |
| `02_selected_experiments/` | Representative ablations and the CLIP control/sweep scripts |
| `03_metrics_and_logs/` | Curated JSON metrics and CSV logs |
| `04_visual_assets/` | Canonical interview figures, sample grids, and archival charts |
| `06_model_artifacts/` | Local-only model files and checksums; excluded by `.gitignore` |
| `docs/` | Reproduction boundary, runtime evidence, and technical project story |
| `tests/` | CPU-only snapshot integrity checks for JSON, CSV, SVG, and README references |
| `tools/` | Standard-library figure builder for the canonical interview charts |
| `99_source_map/` | Freeze scope and source-to-snapshot mapping |

## What is intentionally not claimed

- This is not a state-of-the-art image generator.
- Legacy FID values are not directly comparable with published clean-fid or torch-fidelity numbers.
- The current snapshot does not establish strict generalization without a holdout set.
- A single seed and one evaluation draw are not enough for a causal claim about a small metric difference.
- The public release license is pending internship ownership and dataset/weight redistribution review.

## Next research and engineering milestones

1. Add a fixed train/holdout manifest and standardized FID beside `fid_legacy_project`.
2. Run at least three seeds and report mean, standard deviation, and evaluation sample counts.
3. Add nearest-neighbor grids to check memorization against the training distribution.
4. Extract shared model/data/metric modules and add a CPU shape smoke test.
5. Publish only after employer, dataset, sample, and model-weight permissions are confirmed.

## Publication status

This repository is currently Private. Do not make it public until the checklist in [`PUBLISH_CHECKLIST.md`](PUBLISH_CHECKLIST.md) and the ownership decision in [`LICENSE_DECISION.md`](LICENSE_DECISION.md) are complete. Large binary artifacts should remain outside Git history; GitHub documents Releases and Git LFS as the appropriate mechanisms for large files.

## Technical project story

For an interview, frame the work as an experiment-design and evaluation project: stabilize the adversarial game, test capacity and data scale under a fixed resource budget, preserve negative results, and treat metric disagreement as a research finding. The concise technical version is in [`docs/project_story.md`](docs/project_story.md).
