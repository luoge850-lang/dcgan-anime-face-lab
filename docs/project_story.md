# Technical project story

## One-sentence version

Built and audited a resource-constrained PyTorch DCGAN experiment suite for 64x64 anime-face synthesis, using controlled ablations to separate the effects of discriminator stabilization, generator capacity, data scale, DiffAugment, EMA, and CLIP feature matching.

## Research narrative

The project began with a conventional DCGAN baseline. Rather than adding increasingly complex generator blocks without controls, the experiments separated three questions:

1. Can the adversarial game be stabilized with spectral normalization and Hinge loss?
2. Is the generator capacity or the amount of data the stronger bottleneck under a fixed GPU budget?
3. Do DiffAugment, EMA, and CLIP feature-distribution matching improve the generator under matched controls?

The strongest project-level improvement came from combining a stronger generator with more data and then adding DiffAugment plus EMA. The CLIP sweep was intentionally kept with a no-CLIP continuation control because a lower CLIP MMD² did not guarantee the lowest legacy FID.

## Honest resume wording

> Conducted controlled PyTorch DCGAN ablations under Kaggle GPU constraints across discriminator stabilization, generator capacity, dataset scale, DiffAugment, EMA, and CLIP feature-distribution matching; reduced the project’s legacy FID from 49.17 to 38.88 with DiffAugment + EMA and used matched continuation controls to quantify the smaller, metric-dependent effect of CLIP-MMD regularization.

## Questions an interviewer may ask

### Why keep the negative results?

They show that the project tested hypotheses rather than selecting only successful runs. Several attention, residual, and auxiliary-loss variants were not improvements under the same resource constraints.

### Is the FID a standard benchmark?

No. It is a legacy torchvision Inception-v3 protocol used for longitudinal comparison inside the project. A standardized holdout evaluation and multiple seeds are planned before making generalization claims.

### What would you do next?

Freeze a train/holdout manifest, add standardized FID beside the legacy metric, run three seeds, report uncertainty, add nearest-neighbor checks, and refactor the repeated Kaggle scripts into shared modules with smoke tests.
