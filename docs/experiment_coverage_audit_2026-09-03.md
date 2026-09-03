# Experiment coverage audit — 2026-09-03

This document is the public snapshot's claim boundary. It distinguishes source code from measured evidence so that a reviewer can see what was actually run, what was only prototyped, and what must not be presented as a completed result.

## Coverage matrix

| Stage | Public source | Public evidence | Audit status | Safe statement |
|---|---|---|---|---|
| 1. Training budget and preprocessing | Selected phase-1 scripts | Legacy FID tables, samples, stage charts | Measured | The tested recipe improved across the recorded epoch and preprocessing comparisons. |
| 2. G/D modules and objectives | 20 module-tuning entries | Baseline and module metrics, samples, charts | Measured, non-factorial | SN + Hinge + R1 was the best recorded Phase-2 candidate; every module is not causally isolated. |
| 3. Generator strengthening | 13 source scripts; 12 has no matching result directory | 00–11 result evidence; Exp11 metrics and samples | Partially measured | Exp11 is a multi-factor candidate; `12_G_AA_ADA_EMA_21K.py` is source-only in this snapshot. |
| 4. CLIP continuation | E0 and C0–C4 scripts | Matched FID/MMD² tables and samples | Measured, single seed | C1 wins the recorded Legacy FID sweep; C4 wins CLIP MMD². |
| 5. SDXL data branch | Pilot, production, A0/A10/A20/A30/A50 scripts | Mixing-ratio metrics and samples | Measured side study | Increasing the tested SDXL ratio degraded FID/Coverage; it is not a claimed improvement. |
| 6. Export, engines, precision, QAT | Deployment scripts 01–05 | ONNX checks, benchmark tables, PTQ/mixed/QAT metrics | Measured with explicit partials | FP16 was near-lossless; full INT8 degraded quality; selective FP16 retention recovered quality. Whole-graph fusion was not proven. |
| 7. Service stress and dynamic batching | 06A–06F scripts | Fixed-batch, soak, dynamic-batch summaries and figures | Operational evidence | Stable behavior was observed through the tested range; no physical crash limit was identified. |
| 8. Observability | One Kaggle notebook | HTTP checks, monitoring CSV, alert events, Grafana screenshot | Complete in single node | Prometheus/Grafana/Alertmanager instrumentation and a controlled firing/resolved path were verified. |
| 9. Hot update and A/B | One Kaggle notebook | Health/PID, traffic split, latency, quality, rollback evidence | Complete in single node | Candidate loading, deterministic split, quality check, and rollback were verified without a PID change. |

## Important interpretation rules

1. Historical Legacy FID uses the project's torchvision Inception-v3 pipeline. Deployment Standard FID uses a separate `pytorch-fid` Inception-v3 pool3 pipeline. They are not one leaderboard.
2. The main historical evaluation reuses a training-pool real-image distribution and is not a strict held-out test set.
3. Most comparisons are single-seed. No confidence interval or multi-seed significance claim is made.
4. The historical `LPIPS` field is an AlexNet feature-distance proxy in some result files; the README calls it a diagnostic rather than calibrated LPIPS.
5. A0 discriminator-logit fields in the SDXL study are numerically anomalous relative to the other groups and are excluded from headline comparisons.
6. Stage 8 sampled A/B FID is a separate evaluation instance. It supports the rollout decision in that run but must not be merged with the canonical Stage 3/5 tables.
7. Stage 7 queue alert firing is a controlled simulation. It validates the alert route, not a real incident or a measured crash boundary.
8. Old manifests and raw archives remain as historical evidence. The current deployment-level manifest is the canonical status file for Stages 1–8.

## Engineering packaging policy

- The active Kaggle directory remains the experiment source of truth.
- This repository contains selected scripts, result tables, checksums, charts, and small evidence packages only.
- Dataset images, checkpoints, ONNX files, TensorRT engines, service logs, and large archives remain excluded.
- A new completed Kaggle run should add a dated evidence folder and an explicit manifest; it should not overwrite earlier claims.

