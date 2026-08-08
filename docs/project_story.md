# Technical project story

## One-sentence version

Built and audited a resource-constrained PyTorch DCGAN experiment suite for 64x64 anime-face synthesis, using controlled ablations to separate the effects of discriminator stabilization, generator capacity, data scale, DiffAugment, EMA, and CLIP feature matching.

The follow-up data audit found 21,551 image paths but only 17,029 unique SHA-256 contents. A formal B1 baseline now uses the exact-unique pool, making data quality a documented experimental variable rather than an invisible assumption.

The subsequent controlled SDXL study fine-tuned the Exp11 checkpoint on a fixed 4,000-image budget with 0%, 10%, 20%, 30%, and 50% SDXL replacement. The A0 control reached FID 37.91 and Coverage 0.6687; the 50% mixture reached FID 49.94 and Coverage 0.4397. This negative result is useful: the tested synthetic pool did not improve the target distribution, and higher replacement ratios caused a quality/coverage trade-off in the wrong direction.

## Research narrative

The project began with a conventional DCGAN baseline. Rather than adding increasingly complex generator blocks without controls, the experiments separated three questions:

1. Can the adversarial game be stabilized with spectral normalization and Hinge loss?
2. Is the generator capacity or the amount of data the stronger bottleneck under a fixed GPU budget?
3. Do DiffAugment, EMA, and CLIP feature-distribution matching improve the generator under matched controls?

The strongest project-level improvement came from combining a stronger generator with more data and then adding DiffAugment plus EMA. The CLIP sweep was intentionally kept with a no-CLIP continuation control because a lower CLIP MMD? did not guarantee the lowest legacy FID.

## Honest resume wording

> Conducted controlled PyTorch DCGAN ablations under Kaggle GPU constraints across discriminator stabilization, generator capacity, dataset scale, DiffAugment, EMA, and CLIP feature-distribution matching; reduced the project?s legacy FID from 49.17 to 38.88 with DiffAugment + EMA and used matched continuation controls to quantify the smaller, metric-dependent effect of CLIP-MMD regularization.

> Audited 21,551 Kaggle image paths with SHA-256 exact deduplication, identified 17,029 unique contents and 4,522 redundant copies, and established a clean-unique B1 baseline (legacy project FID 45.07) before the planned SDXL-mixture study.

> Designed a fixed-budget SDXL replacement ablation (0?50% synthetic images) from the same Exp11 checkpoint; found that the tested synthetic pool did not improve FID or Coverage, and preserved the negative result as a stopping criterion for further data mixing.

## Questions an interviewer may ask

### Why keep the negative results?

They show that the project tested hypotheses rather than selecting only successful runs. Several attention, residual, and auxiliary-loss variants were not improvements under the same resource constraints.

### Is the FID a standard benchmark?

No. It is a legacy torchvision Inception-v3 protocol used for longitudinal comparison inside the project. A standardized holdout evaluation and multiple seeds are planned before making generalization claims.

### What would you do next?

Freeze a train/holdout manifest, add standardized FID beside the legacy metric, run three seeds, report uncertainty, add nearest-neighbor checks, and refactor the repeated Kaggle scripts into shared modules with smoke tests.
