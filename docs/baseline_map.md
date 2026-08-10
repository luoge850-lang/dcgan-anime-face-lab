# Baseline map

The project uses several baselines because each stage answers a different question. They must not be collapsed into one universal leaderboard.

| Name | Purpose | Data / protocol | Main configuration | Result | Safe wording |
|---|---|---:|---|---:|---|
| `standardized_baseline.py` | Early reusable Kaggle framework | Up to 15K sampled images, 100-epoch configuration | Original DCGAN-style G/D, seed 42 | Framework record, not the headline Phase 2 row | Preliminary baseline framework |
| Phase 2 `00_baseline` | Architecture control for module ablations | 8K, 200 epochs, legacy FID | Standard DCGAN G/D; no added modules; Phase 1 `Flip + EdgeSharpen` retained | 109.40 | No-added-module architectural baseline; not a no-augmentation control |
| Phase 2 `13_D_SN_Hinge` | Discriminator stabilization milestone | 8K, 200 epochs, same Phase 2 scope | Spectral Normalization + Hinge loss in D | 96.25 | Selected discriminator-stabilization result |
| Exp11 / B0 | Frozen historical best reference before the data audit | Approximately 20K path-based images, 200 epochs | Width ×3 G + SN-Hinge D + DiffAugment + EMA | 38.88 | Historical path-based frozen baseline |
| B1 | Formal data-quality control for the SDXL extension | 17,029 SHA-256 unique contents, 200 epochs | Exp11 recipe, same seed and training settings | 45.07 | Clean-unique data baseline; not a direct architecture comparison with Exp11 |

## The key distinction

“No added modules” describes the Phase 2 architecture. It does not mean that the input pipeline has no augmentation. The Phase 2 baseline intentionally inherits the Phase 1 augmentation policy so that the later module sweep starts from the best preliminary data pipeline.

“B1 is worse than Exp11” is also not a valid direct conclusion. B1 changes the data pool from path-based sampling to one path per exact SHA-256 content. The result is a new data-quality boundary, not an isolated model regression.

## Recommended interview sentence

> I used the Phase 2 no-added-module DCGAN as the architectural control for module ablations, then froze the stronger Exp11 recipe as a historical path-based reference. After discovering exact duplicates in the source data, I established B1 on 17,029 unique contents as the formal baseline for future SDXL mixture experiments.
