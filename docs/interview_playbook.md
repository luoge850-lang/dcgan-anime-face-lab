# Interview and learning playbook

This document turns the Kaggle archive into a project you can explain in your own words. The goal is not to memorize every historical script. For every experiment, be able to answer four questions:

1. What hypothesis was being tested?
2. What changed relative to the control?
3. Which metric or sample evidence supported the decision?
4. What limitation prevents a stronger causal claim?

## One-sentence project identity

> I designed and audited a staged PyTorch DCGAN study for 64x64 anime-face generation under Kaggle GPU constraints, using controlled ablations to separate architecture, data scale, training stability, CLIP regularization, and data-quality effects.

## The experiment story

### 1. Choose a fair training budget

The early epoch study compared 50, 100, 200, and 300 epochs. FID improved as training continued, but the purpose of selecting 200 epochs was a practical comparison budget, not proof that 200 is globally optimal.

### 2. Establish the architectural control

Phase 2 `00_baseline` is the no-added-module architectural baseline. It uses the Phase 1 `RandomHorizontalFlip + EdgeSharpen` policy, but no attention, frequency, edge, spectral-normalization, Hinge, or R1 module. This baseline lets the module sweep answer whether each added idea helps.

### 3. Test stabilization and capacity

Spectral Normalization and Hinge loss improved the discriminator-side comparison. Generator Width?3 was stronger than the smaller and larger capacity variants in the tested budget. Negative results were kept because they define the tested resource-constrained design space.

### 4. Separate data scale from architecture

Width?3 was held fixed while the data scale changed from 10K to 20K. This is one of the cleanest comparisons in the project because the script records the architecture, optimizer, seed, epoch count, and batch size as unchanged.

### 5. Stabilize the training trajectory

Exp11 added DiffAugment and EMA together. The result improved the historical project FID, but the correct claim is a combined intervention result. The project does not identify the separate causal contribution of DiffAugment and EMA.

### 6. Treat CLIP as a continuation study

CLIP is used as a frozen image encoder and a distribution-level MMD objective. There is no one-to-one real/fake image pairing. The no-CLIP continuation control is essential because it shows how much of the change could come from simply training longer.

### 7. Audit the data before adding SDXL images

The SHA-256 audit found 21,551 paths but only 17,029 unique contents. B1 fixes the exact-duplicate issue before the planned M20/M50 mixture study. B1 is a new data-quality control, not a direct replacement for the older path-based Exp11 score.

## A 90-second answer

> My internship project is a resource-constrained PyTorch DCGAN study for unconditional 64x64 anime-face generation. I structured it as staged experiments rather than only selecting a final checkpoint. I first selected a practical epoch budget and augmentation policy, then used a no-added-module DCGAN as the architectural control for attention, frequency, edge, and discriminator-stabilization ablations. I next tested Generator capacity and held the Width?3 architecture fixed while scaling the data from 10K to 20K images. The strongest historical result came from combining the stronger generator, more data, DiffAugment, and EMA. I then ran a matched CLIP-MMD continuation sweep with a no-CLIP control. The best legacy FID and best CLIP MMD were not the same checkpoint, so I treated the metrics as complementary. Finally, I audited the source data with SHA-256, found 4,522 redundant copies, and established B1 on 17,029 unique contents before the planned SDXL mixture experiments. The main limitations are Kaggle-only execution, legacy FID, a non-holdout real set, and mostly single-seed results.

## Questions you should be ready for

### Why did you choose 200 epochs?

Because the early study showed a useful improvement up to 200 epochs and the project needed a fixed compute budget for later ablations. It is a practical comparison budget, not a global optimum claim.

### Was the baseline completely vanilla?

It was vanilla with respect to added architecture and discriminator modules, but it retained the Phase 1 Flip+Sharpen augmentation. The precise name is ?no-added-module architectural baseline.?

### Which change helped the most?

Within the tested scopes, the largest reliable milestone was the combined progression to Width?3, 20K data, DiffAugment, and EMA. DiffAugment and EMA were introduced together, so their separate effects were not identified.

### Why is B1 FID 45.07 higher than Exp11 FID 38.88?

They use different data pools. Exp11 uses the historical path-based pool, while B1 uses one path per exact SHA-256 content. That difference makes B1 a data-quality baseline, not a direct architecture regression.

### Why use CLIP if the project is unconditional?

The CLIP encoder is used to compare real and generated feature distributions with MMD. It is not used as a text condition and does not require paired real/fake images.

### What would you do next?

Freeze a train/holdout manifest, add standardized FID and Coverage beside the legacy metric, run multiple seeds, add nearest-neighbor and memorization checks, complete the SDXL pilot and accepted manifest, and compare B1/M20/M50 under one evaluation script.

## What you personally need to master

- DCGAN tensor shapes and the Generator/Discriminator training loop.
- BCE versus Hinge loss and why Spectral Normalization affects stability.
- DiffAugment on real/fake inputs and EMA checkpoint semantics.
- FID computation and why implementation, sample count, and holdout choice matter.
- CLIP feature extraction, MMD, and why this is distribution matching rather than image pairing.
- Dataset manifests, SHA-256 deduplication, data leakage, and comparison scope.
- How to distinguish an observed improvement from a causal claim.

## Honest description of AI collaboration

If asked about implementation, say:

> I used AI-assisted coding to accelerate repetitive Kaggle implementation and refactoring, but I owned the experiment questions, comparison boundaries, test execution, result verification, and interpretation. I reviewed the generated code and rejected or preserved experiments based on the recorded evidence.

This is stronger and more honest than claiming that every line was written manually or pretending that the implementation process did not use assistance.
