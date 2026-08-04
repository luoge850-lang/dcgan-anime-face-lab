# Resume and Interview Framing

## One-line project description

Built a reproducible PyTorch DCGAN experimentation pipeline for 64x64 anime-face synthesis under constrained Kaggle GPU resources.

## Resume bullet draft

Conducted controlled ablations across generator capacity, spectral normalization, Hinge loss, dataset scale, DiffAugment, EMA, and CLIP feature-distribution regularization; reduced project-protocol FID from 49.17 to 38.88 with DiffAugment + EMA and reached 33.41 after CLIP-MMD fine-tuning.

## Interview story

The main conclusion was not that a more complicated architecture always wins. Under batch-size and GPU constraints, increasing data diversity and stabilizing the adversarial game were more effective than repeatedly adding residual or attention blocks. Failed experiments were kept because they narrowed the design space and exposed the interaction between BatchNorm statistics, gradient flow, and discriminator strength.

## Claims to avoid

- Do not call the result state of the art.
- Do not present legacy FID as a clean-fid benchmark.
- Do not claim strict generalization without a holdout set.
- Do not claim causal laws from a single seed; describe them as observations or hypotheses supported by the training curves.
