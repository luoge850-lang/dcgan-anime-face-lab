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

## Deployment runtime evidence

The latest service preflight and staged stress manifest records the following Kaggle runtime:

| Component | Recorded value |
|---|---|
| Python | 3.12.13 |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| TensorRT | 11.2.1.2 |
| GPU | Tesla T4, approximately 14.9 GiB total memory |
| Service | FastAPI/Uvicorn, one worker, batch 1 |
| Stress monitor | 5-second interval; GPU memory, SM, RSS, health |

The optional utility dependencies are listed in [`requirements-deployment.txt`](../requirements-deployment.txt). TensorRT remains platform-specific and is intentionally not treated as a portable local requirement.
