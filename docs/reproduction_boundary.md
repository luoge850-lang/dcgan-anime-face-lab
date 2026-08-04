# Reproduction boundary

## What this repository is

This repository is a curated record of a Kaggle-based internship experiment. It preserves selected scripts, metrics, plots, sample grids, and checksums so a reviewer can inspect the reasoning and compare milestones.

The latest curated evidence adds a read-only dataset audit and the completed `B1_Formal_CleanUnique_17K` baseline. The audit found 21,551 paths, 17,029 unique SHA-256 contents, 3,626 exact-duplicate groups, 4,522 redundant copies, and zero bad files. The B1 result is legacy project FID 45.07 on the exact-unique pool.

## What it is not yet

It is not a local one-command training package. The historical runs depend on:

- a Kaggle GPU runtime and internet access for pretrained feature extractors;
- an attached, permitted image dataset;
- Kaggle-mounted paths for data, model weights, and outputs;
- the exact experiment configuration and training protocol;
- the environment metadata captured in the Phase 5 JSON files.

The active internship workspace contains additional unfinished experiments and remains the source of truth for new work. This snapshot should not be treated as a copy of the full active workspace.

The SDXL pilot, candidate cleaning, M20/M50 mixtures, and unified Clean-FID/Coverage evaluation are not complete in this release and are not claimed as results.

## Responsible historical reproduction

1. Confirm that the dataset, generated samples, and model weights can be used for the intended audience.
2. Attach the permitted dataset in Kaggle and record its source, version, image count, and SHA-256 manifest.
3. Attach the matching model artifacts for CLIP continuation experiments.
4. Run the exact entry point named in `results_summary.csv`.
5. Record seed, dataset-limit, image count, batch size, epoch count, runtime, GPU, and feature-extractor versions.
6. Recompute the legacy metric without overwriting the frozen value.
7. Add a standardized holdout evaluation before making generalization claims.

For the new B1 baseline, use `ANIME_FACES_DATASET` to point the sanitized Kaggle entry point at the mounted dataset. Use the included unique-content manifest when available, and record the manifest digest `5161f8bf8e388de20beb2f6837dd954abc425711894715128507f2201d63b01b`.

## Planned local reproduction release

The next engineering release should provide:

- a `src/dcgan_lab` package;
- `train`, `evaluate`, and `sample` entry points;
- a fixed manifest rather than directory auto-discovery;
- a CPU shape smoke test;
- a pinned environment file;
- a standardized FID implementation beside the legacy protocol.
