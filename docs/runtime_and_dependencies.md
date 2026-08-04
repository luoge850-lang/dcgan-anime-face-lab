# Runtime and dependency evidence

## Observed Kaggle runtime

The following values were captured in the Phase 5 experiment configuration JSON files. They describe the recorded runs; they are not a guarantee that a future Kaggle image will resolve the same packages.

| Component | Recorded value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| OpenCLIP | 3.3.0 |
| CUDA | 12.8 |
| GPU | Tesla T4 |
| Image resolution | 64x64 |
| Batch size in Phase 5 | 32 |
| Seed in Phase 5 | 42 |
| Legacy FID sample count | 10,000 real + 10,000 fake |
| CLIP evaluation sample count | 2,000 |

## Dependency policy

`requirements.txt` is intentionally a compatibility floor because the active project was run in Kaggle. It is not a lockfile. A future public release should add a tested lockfile or environment specification, including the exact pretrained feature-extractor weights and their licenses.

Do not infer that installing the requirements locally is sufficient to reproduce the historical numbers. The dataset manifest, Kaggle-mounted model artifacts, runtime, and evaluation protocol are equally important.
