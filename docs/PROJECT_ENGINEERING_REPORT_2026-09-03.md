# DCGAN Anime-Face Lab

## Internship project engineering report

**Release:** `v0.9-current-state`  
**Project type:** Kaggle-based generative-model research and inference-engineering study  
**Primary model:** PyTorch DCGAN for unconditional 64 × 64 RGB anime-face generation  
**Execution environment:** Kaggle GPU, mainly Tesla T4 for deployment experiments  
**Public repository:** `luoge850-lang/dcgan-anime-face-lab`

This report is the authoritative narrative for the public snapshot. It separates the active experiment workspace from the curated GitHub package and separates measured evidence from source-only code. It is intentionally not written as a paper, SOTA claim, or proof of production readiness.

## 1. Executive summary

### Problem definition

The project asks:

> Under a constrained Kaggle single-GPU workflow, how can a conventional DCGAN for 64 × 64 anime-face generation be improved, evaluated, and moved toward a measurable inference service?

The work covers two connected tracks:

1. **Model and data track:** training budget, preprocessing, G/D stabilization, generator capacity, data scale, DiffAugment, EMA, CLIP continuation, and a controlled SDXL data-side study.
2. **Deployment and operations track:** ONNX export, graph/fusion probes, ORT/TensorRT/OpenVINO benchmarks, PTQ, layer sensitivity, mixed precision, QAT, HTTP stress, dynamic batching, observability, and hot-update/A/B rollback.

### Project outcome

The strongest defensible outcome is not “one magic module improved the GAN.” It is a documented decision process:

- discriminator-side stabilization was more promising than several isolated feature modules;
- the best historical training candidate reached project Legacy FID **38.88**, but it combines capacity, data scale, DiffAugment, and EMA and is not a single-factor causal result;
- a matched CLIP continuation exposed a metric trade-off: C1 had the lowest Legacy FID, while C4 had the lowest CLIP MMD²;
- the tested SDXL replacement ratios degraded FID/Coverage, so the synthetic-data branch was stopped rather than promoted;
- FP16 was near-lossless under the deployment protocol, full INT8 improved speed at a quality cost, and selective FP16 retention recovered quality;
- service tests reached concurrency 512 with zero failures, but did not observe a physical crash limit;
- monitoring, controlled alert lifecycle, same-PID model loading, A/B routing, and rollback were verified on one Kaggle node;
- the evidence is strong enough for an engineering case study and interview discussion, but not for a publication or general production guarantee.

## 2. Scope, source of truth, and freeze boundary

| Layer | Location / identity | Meaning |
|---|---|---|
| Active experiment source | `dcgan_lab` | Ongoing Kaggle-oriented workspace; later experiments may still change. |
| Public package | `DCGAN_Interview_GitHub_Snapshot_2026-08-04` | Curated, public-safe evidence snapshot. |
| Git release | `v0.9-current-state` | Frozen public state for the currently documented evidence. |
| Current deployment manifest | `03_metrics_and_logs/deployment_optimization/deployment_optimization_current_manifest.json` | Canonical status for deployment stages. |
| Claim boundary | `docs/experiment_coverage_audit_2026-09-03.md` | Distinguishes measured, partial, source-only, and non-comparable results. |

The active workspace is much larger than the public package. It contains raw images, generated samples, checkpoints, engines, ONNX files, archives, logs, and intermediate outputs. These are intentionally not copied into GitHub.

## 3. Complete workspace inventory

The 2026-09-03 scan found:

| Item | Active workspace |
|---|---:|
| Total files | 2,864 |
| Total size | 718,513,136 bytes, approximately 685 MiB |
| Python scripts | 93 |
| JSON files | 129 |
| CSV files | 297 |
| PNG/JPG samples and figures | 2,163 |
| SVG figures | 45 |
| Checkpoints (`.pth`) | 8 |
| TensorRT engines | 7 |
| ONNX files | 2 |
| ZIP archives | 11 |
| Python AST parse errors | 0 |
| JSON parse errors | 0 |
| CSV shape errors | 0 |
| CSV data rows | 47,522 |

### Active experiment families

| Internal family | Scripts | Result family | Public role |
|---|---:|---|---|
| Early tuning | 5 | 前期调优结果 | Epoch budget and preprocessing search |
| Deep G/D tuning | 20 | 深度调优结果 | Baseline and module/objective ablations |
| Generator strengthening | 13 | G强化实验结果 | Capacity, data scale, DiffAugment and EMA |
| CLIP study | 6 | CLIP实验结果 | Frozen CLIP continuation and weight sweep |
| SDXL controlled study | 11 | SDXL_Controlled_Study_Results | Synthetic-data candidate and mixing-ratio study |
| Deployment optimization | 30 | Deployment_Optimization_Results | Export, engines, quantization, service and operations |

The public snapshot contains **657 curated files** (about 36 MiB), including selected scripts, normalized tables, reports, samples, stage charts, tests, and small evidence packages. No file larger than 50 MiB is included in the public package.

## 4. Canonical naming and organization

Historical source names are retained because they are part of the provenance. The public report gives them stable conceptual aliases without mass-renaming the original files.

### Research track

| Public alias | Meaning | Main public source folder |
|---|---|---|
| `R01_budget_preprocessing` | Epoch budget and image transformations | `02_selected_experiments/full_process/phase1_early_tuning/` |
| `R02_gd_module_ablation` | Plain control, edge/frequency/attention/SN/Hinge/R1 comparisons | `02_selected_experiments/full_process/phase2_module_tuning/` |
| `R03_generator_recipe` | Width, data scale, Laplacian, DiffAugment, EMA | `02_selected_experiments/full_process/phase3_generator_strengthening/` |
| `R04_clip_continuation` | No-CLIP control and frozen CLIP MMD² weights | `02_selected_experiments/full_process/phase5_clip_tuning/` |
| `R05_sdxl_data_side_study` | SDXL candidate filtering and replacement ratios | `02_selected_experiments/full_process/phase7_sdxl_controlled_study/` |

### Deployment track

| Public alias | Meaning | Main public source folder |
|---|---|---|
| `D01_onnx_fusion` | Export, checker, graph probes and BN folding | `.../deployment_optimization/01_ONNX_Fusion/` |
| `D02_engine_benchmark` | ORT, TensorRT, OpenVINO, profiler and bottleneck map | `.../02_Engine_Benchmark/` |
| `D03_ptq_baseline` | FP32/FP16/INT8 calibration and quality baseline | `.../03_Quantization/` |
| `D04_mixed_precision` | Layer sensitivity and selected FP16 retention | `.../04_Quantization_Sensitivity/` |
| `D05_qat` | Fake-quant fine-tuning and QAT evaluation | `.../05_QAT/` |
| `D06_serving` | HTTP smoke, stress, soak and dynamic batching | `.../06_Service_Stress/` |
| `D07_observability` | Prometheus, Grafana, Alertmanager and resource monitoring | `.../07_MLOps_Observability/` |
| `D08_hot_update_ab` | Same-PID update, A/B routing, quality gate and rollback | `.../08_Model_Hot_Update_AB/` |

### Public repository layers

| Layer | Purpose |
|---|---|
| `01_public_core/` | Baseline model definitions and public entry-point notes |
| `02_selected_experiments/` | Selected source scripts, grouped by research/deployment stage |
| `03_metrics_and_logs/` | Normalized metrics, manifests, audit tables and small evidence packages |
| `04_visual_assets/` | Stage-organized charts, sample grids and preserved screenshots |
| `06_model_artifacts/` | Artifact inventory/checksums; binaries remain excluded |
| `docs/` | Engineering report, protocols, interview framing and audit boundaries |
| `tools/` | Result-driven figure builders and lightweight packaging utilities |
| `tests/` | Snapshot integrity and evidence-presence checks |
| `99_source_map/` | Freeze record and source/public boundary |

## 5. Research track: full experiment evolution

### R01 — Training budget and preprocessing

**Question.** How much training is useful before changing the architecture, and which low-cost transformations are worth keeping?

**Control and changes.** The epoch study compared 50, 100, 200, and 300 epochs. The preprocessing search tested horizontal flip, color adjustment, blur, denoise, sharpening, and combinations.

| Run | Legacy FID | Interpretation |
|---|---:|---|
| 50 epochs | 184.11 | Early checkpoint |
| 100 epochs | 136.78 | Large early gain |
| 200 epochs | 113.97 | Continued improvement |
| 300 epochs | 105.34 | Smaller marginal gain |
| Sharpen | 121.11 | Best listed single preprocessing candidate |
| Flip + Sharpen | 124.43 | Competitive combination |
| Flip + Color + Sharpen | 121.54 | Close to sharpening |

**Conclusion.** Longer training improved the recorded recipe and sharpening was the best listed preprocessing candidate in this small search. This is a one-seed, recipe-scoped observation, not a universal augmentation law.

**Negative evidence.** Blur and denoise were weaker at 141.15 and 155.81; color alone was 136.12. The project therefore keeps the negative comparisons rather than claiming every augmentation helps.

### R02 — Plain baseline and G/D module ablation

**Control.** `00_baseline` is the plain architectural control for this phase: no newly added attention, frequency, edge, spectral-normalization, Hinge, or R1 module. It retains the selected Phase-1 input policy. This distinction matters: “plain baseline” means no added architecture module, not “no preprocessing.”

**Results.** The explicit baseline recorded Legacy FID **109.40**. The best recorded Phase-2 candidate was D + SN + Hinge + R1 at **89.92**.

| Variant | Legacy FID | Reading |
|---|---:|---|
| G + SENet | 96.76 | Best listed G-side single-module candidate |
| G + Laplacian | 98.67 | Improvement in this run |
| G + Wavelet | 106.14 | Close to control |
| G + FFT | 108.40 | Little change |
| G + Canny | 109.02 | Little change |
| D + SN | 99.19 | Stabilization candidate |
| D + Hinge | 129.16 | Negative result alone |
| D + SN + Hinge | 96.25 | Stronger combined candidate |
| D + SN + Hinge + R1 | 89.92 | Best recorded Phase-2 value |

**Conclusion.** The evidence favors discriminator-side stabilization in this scope. Because the search was not a perfectly balanced factorial design, the table cannot isolate every component's independent causal effect.

### R03 — Generator capacity, data scale, DiffAugment and EMA

**Question.** Is the later gain better explained by generator capacity, more data, training stabilization, or their combination?

| Recipe | Legacy FID | Changed factors |
|---|---:|---|
| Width ×2 | 78.75 | Capacity |
| Width ×3 | 59.00 | Capacity |
| Width ×4 | 63.66 | Capacity; negative versus ×3 |
| Width ×3 + 20K data | 49.17 | Capacity + data scale |
| Width ×3 + 20K + Laplacian | 53.74 | Added Laplacian; regression |
| Width ×3 + 20K + DiffAugment + EMA | 38.88 | Multi-factor candidate |

The `12_G_AA_ADA_EMA_21K.py` script is preserved as a follow-up source file but has no matching result directory in the scanned workspace; it is not a measured winner. The old generator README was corrected in the public snapshot to reflect this fact.

**Conclusion.** The 38.88 candidate is the strongest historical result, but it must be described as a combined recipe. The cleanest within-stage transition is Width ×3 to Width ×3 + 20K; DiffAugment and EMA are not separately identified by the archived result.

### R04 — CLIP continuation

**Question.** After the Exp11 checkpoint, does a frozen OpenCLIP image encoder and distribution-level MMD² objective improve perceptual alignment?

**Control.** C0 continues training without CLIP loss. C1–C4 add different CLIP weights.

| Run | CLIP lambda | Legacy FID | CLIP MMD² | Winner |
|---|---:|---:|---:|---|
| C0 | 0 | 33.7846 | 0.042832 | No-CLIP control |
| C1 | 0.0100 | 33.4114 | 0.042657 | Lowest FID |
| C2 | 0.0250 | 33.6687 | 0.042298 | Lower MMD² than C1 |
| C3 | 0.0500 | 33.4718 | 0.041856 | Near-best FID |
| C4 | 0.1000 | 33.6231 | 0.040990 | Lowest MMD² |

**Conclusion.** The correct claim is a metric-dependent trade-off. CLIP is an optional continuation branch, not part of the recorded 38.88 recipe and not text-conditioned generation.

### R05 — SDXL data-side study

This is a separate data-scope experiment, not an architecture ablation. The tested pool mixed real images with SDXL-generated candidates at fixed ratios.

| Group | Real + SDXL | Legacy FID | Coverage | Decision |
|---|---:|---:|---:|---|
| A0 | 4,000 + 0 | 37.91 | 0.6687 | Control |
| A10 | 3,600 + 400 | 37.99 | 0.6525 | Approximately neutral FID; lower coverage |
| A20 | 3,200 + 800 | 41.58 | 0.6108 | Negative |
| A30 | 2,800 + 1,200 | 44.92 | 0.5423 | Warning region |
| A50 | 2,000 + 2,000 | 49.94 | 0.4397 | Worst tested mixture |

**Conclusion.** The tested SDXL pool did not improve the target distribution. The public report keeps this negative result because stopping a weak branch is part of the engineering decision record. A0 discriminator-logit fields are anomalous and excluded from headline comparisons.

## 6. Deployment and systems track

The deployment track takes a selected Generator candidate and evaluates it as an inference artifact. It does not prove that every training-time module is present in the exported graph.

### D01 — ONNX export and fusion

**Status: partial.** ONNX export, checker validation, FX capture, standard replacement probes, and BN-fold numerical equivalence have evidence. A stable whole-generator fusion speedup was not established. The deployed Exp11 graph is a standard ConvTranspose + BatchNorm + ReLU + Tanh graph; the independent Haar-wavelet and dynamic-SN probes are not deployed nodes.

### D02 — Multi-engine benchmark

**Status: complete.** The same raw ONNX graph was benchmarked across ONNX Runtime CPU, TensorRT GPU, and OpenVINO CPU with FP32/FP16 and dynamic batches. The correct interpretation is an engine–hardware combination, not a pure software multiplier.

### D03–D05 — Precision and quantization

| Precision / strategy | Standard FID | Blur rate | Throughput |
|---|---:|---:|---:|
| FP32 | 29.9911 | 12.0% | 4,354.7 images/s |
| FP16 | 29.9941 | 12.0% | 15,495.0 images/s |
| INT8 PTQ | 35.3198 | 12.5% | 21,106.1 images/s |
| Mixed PTQ: `net.0 + net.12` FP16 | 31.1776 | 12.1% | 23,971.7 images/s |
| QAT INT8 | 31.6456 | 11.3% | 19,092.7 images/s |

**Decision.** Full INT8 was faster but degraded quality. Per-layer sensitivity identified `net.0` and `net.12` as sensitive enough to retain in FP16. QAT improved over full INT8 PTQ but did not beat the selected mixed-precision trade-off. The evidence does not establish a specific hair/eyeliner ROI advantage.

### D06 — Service, stress, soak and dynamic batching

| Test | Recorded result | Boundary |
|---|---|---|
| Fixed-batch concurrency | 1–512, zero failures | Highest tested successful point, not physical limit |
| Fixed-batch P99 | 5 → 1,600 ms | Soft latency knee around 32 |
| 60-minute soak | 3,601.6 s; 1,226,890 requests; 0 failures; P99 63 ms; 340.83 RPS | Leak screen under one workload |
| Dynamic batching | Actual batches 2/4/8; zero failures through concurrency 128 | Downloaded archive has packaging gaps |
| Dynamic batching at concurrency 32 | 401.73 vs 352.35 RPS; +14.01%; P99 90 vs 110 ms | Queue window creates low-load overhead |

### D07 — Observability

**Status: complete for one Kaggle single-node run.** Health, generation, and metrics endpoints returned 200; two alert rules loaded; the Prometheus target was up; firing and resolved events were captured; 37 resource-monitor samples were recorded; and a Grafana screenshot was preserved. Queue backlog was intentionally simulated to validate the alert route. This is not proof of an external paging integration or a multi-replica SLO.

### D08 — Hot update, A/B and rollback

**Status: complete for one Kaggle single-node run.** Version A was PTQ INT8 and version B was QAT INT8. The service kept the same PID while loading B, exercised 10%/50%/100% B traffic, measured quality and latency, and rolled back to A.

| Measure | A: PTQ INT8 | B: QAT INT8 |
|---|---:|---:|
| Sampled FID | 35.5710 | 32.0422 |
| Blur rate | 12.50% | 11.62% |
| P99 latency | 98.6 ms | 186.7 ms |
| Success rate | 100% | 100% |
| PID | 58 | 58 |

The candidate B had better sampled quality in this run but approximately 1.9× the A P99 tail. The sampled A/B FID is a separate evaluation instance and is not merged into the canonical D03–D05 table.

## 7. Evaluation protocol and comparability

| Metric | Protocol | Valid comparison scope |
|---|---|---|
| Legacy FID | Project torchvision Inception-v3 pipeline; historical real samples often from training pool | Within the declared core experiment scope only |
| Standard FID | Deployment `pytorch-fid` Inception-v3 pool3 pipeline | D03–D05 precision/deployment comparison |
| CLIP MMD² | Frozen OpenCLIP ViT-B/32 image features, multi-scale RBF MMD² | Matched C0–C4 continuation |
| `LPIPS` field | Historical AlexNet feature-distance proxy in some files | Supporting diagnostic; not calibrated LPIPS |
| Blur/Laplacian/edge | Project-defined image diagnostics | Supporting evidence, not replacement for held-out FID |
| Coverage | Project-defined Inception-feature diagnostic | A0–A50 SDXL study only |

There is no valid universal ranking such as “38.88 is better than 29.99.” The numbers come from different implementations, pools, checkpoints, sample counts, and stages.

## 8. Reproducibility and engineering quality

### What the public snapshot supports

- inspection of selected source scripts and their stage order;
- review of normalized JSON/CSV metrics and manifests;
- viewing of stage-organized sample grids and single-metric charts;
- checking the public evidence with the integrity test suite;
- rebuilding the public Stage 8 charts from the curated CSV files;
- tracing the source-to-public boundary through the freeze map and coverage audit.

### What it does not support

- local one-command retraining of the entire historical study;
- exact rerun without the original Kaggle dataset mounts, checkpoints, feature extractors, GPU and experiment-specific paths;
- downloading the original dataset, TensorRT engines, ONNX files, or large checkpoints;
- treating the snapshot as a production service distribution.

### Engineering improvements applied in this release

1. Reorganized the narrative around stable research/deployment aliases while preserving historical provenance paths.
2. Added a script-to-evidence coverage matrix and a canonical deployment manifest.
3. Added Stage 7/8 source entry points and small evidence packages.
4. Added separate Stage 8 P99, FID, and traffic-ratio charts rather than a long composite image.
5. Replaced personal absolute paths in public text, manifests, and deployment scripts with explicit arguments, environment variables, relative paths, or placeholders.
6. Added JSON/CSV/figure/link integrity checks and preserved negative results.
7. Kept large artifacts outside Git history and documented the omission rather than pretending local reproducibility.

## 9. Limitations and risk register

| Risk | Impact | Correct wording |
|---|---|---|
| Mostly single-seed | Unknown variance and significance | “Recorded single-seed comparison” |
| Non-held-out historical evaluation | Possible optimistic FID | “Training-pool evaluation under the declared protocol” |
| Legacy vs Standard FID | Cross-stage misinterpretation | “Do not merge into one leaderboard” |
| Multi-factor Exp11 | No isolated DiffAugment/EMA causality | “Combined candidate” |
| D01 fusion | Whole-graph acceleration not proven | “Partial export/fusion evidence” |
| D06 capacity | No hard crash observed | “Stable through tested range; boundary above 512” |
| D07 alert | Queue alarm is simulated | “Controlled alert-path validation” |
| D08 A/B | B tail latency is higher | “Quality improvement with latency trade-off in one run” |
| Kaggle-only execution | Low local reproducibility | “Kaggle evidence package, not local release” |

## 10. Interview-safe project description

> I contributed to a Kaggle-based PyTorch DCGAN study for unconditional 64 × 64 anime-face generation under GPU constraints. I organized the work into controlled stages: training-budget and preprocessing selection, G/D stabilization, generator capacity and data-scale comparisons, a CLIP continuation sweep, and a negative SDXL mixing-ratio study. The strongest historical candidate reached project Legacy FID 38.88, but I present it as a multi-factor recipe rather than attributing the gain to one module. I then helped carry the selected generator into an ONNX/TensorRT deployment study, quantified the quality cost of INT8, used layer sensitivity to select a mixed-precision strategy, and validated stress, monitoring, hot update, A/B routing, and rollback on a single Kaggle node. I keep the claims bounded by single-seed runs, non-held-out historical evaluation, separate FID protocols, and the lack of a measured physical crash boundary.

Use contribution-accurate verbs such as **contributed to**, **implemented**, **evaluated**, **audited**, or **documented**. Do not describe the project as a publication, SOTA result, or multi-node production system.

## 11. Recommended final artifact set

For an interviewer or application reviewer, open the files in this order:

1. [`README.md`](../README.md) — 30-second project story and visual route;
2. this report — complete engineering narrative;
3. [`experiment_coverage_audit_2026-09-03.md`](experiment_coverage_audit_2026-09-03.md) — what is measured and what is not;
4. [`SCRIPT_AUDIT.md`](../02_selected_experiments/full_process/deployment_optimization/SCRIPT_AUDIT.md) — execution and output mapping;
5. [`deployment_optimization_current_manifest.json`](../03_metrics_and_logs/deployment_optimization/deployment_optimization_current_manifest.json) — deployment status;
6. stage figures and sample grids — visual quality and system trade-offs;
7. [`tests/test_snapshot_integrity.py`](../tests/test_snapshot_integrity.py) — evidence integrity checks.

## 12. Future updates

Continue writing new work in the active Kaggle workspace. When a later experiment is complete, add a dated evidence folder, a manifest, a short result interpretation, and a new Git commit/tag. Do not rewrite this freeze or silently replace an old metric. If a future run changes the dataset, checkpoint, metric protocol, or serving hardware, create a new comparison scope instead of appending the value to an existing leaderboard.
