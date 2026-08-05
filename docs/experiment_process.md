# Complete experiment process

The repository is not only a final-model comparison. It records a staged Kaggle workflow in which each stage answered a different question. The full-process source scripts are under `02_selected_experiments/full_process/`; the corresponding logs and metric JSON files are under `03_metrics_and_logs/`.

## Stage 1 ? Early tuning

### Epoch budget

The preliminary baseline was evaluated at 50, 100, 200, and 300 epochs on the same 8K-scale setup:

| Epochs | Legacy FID |
|---:|---:|
| 50 | 184.11 |
| 100 | 136.78 |
| 200 | 113.97 |
| 300 | 105.34 |

The result is a compute-budget trade-off, not a claim that 200 epochs is the global optimum. Later comparative experiments used a fixed 200-epoch budget so architecture changes were easier to compare.

### Input augmentation

Flip, color, blur, denoise, sharpen, and combinations were tested before the deeper architecture sweep. The best early single augmentation was sharpen at FID 121.11, while the combined flip/color/sharpen run reached 121.54. These runs are exploratory and use a different epoch budget from the later Phase 2 comparisons.

## Stage 2 ? Module and discriminator tuning

The true Phase 2 comparison baseline is `00_baseline`: a no-added-module DCGAN architecture with the Phase 1 augmentation policy (`RandomHorizontalFlip(p=0.5)` + `EdgeSharpen(p=0.2)`). It has no added attention, wavelet, FFT, Canny, Laplacian, spectral-normalization, Hinge, or R1 module. Its Phase 2 legacy FID is 109.40. This should not be described as a no-augmentation control.

The module sweep then tested SENet, CBAM, Wavelet, FFT, Canny, Laplacian, discriminator SN, Hinge, self-attention, and R1 variants. The best 8K row was `D_SN_Hinge_R1` at 89.92, but R1 was not automatically carried into later stages because its result did not improve the Width x3 family under the later comparison.

## Stage 3 ? Generator strengthening and data scale

Generator capacity and structural variants were tested at a fixed 10K-scale setting. Width x3 was the strongest capacity result at FID 59.00; Width x4, residual, PixelShuffle, and attention variants were retained as negative or ceiling results.

The Width x3 architecture was then held fixed while the data scale moved to 20K (FID 49.17). DiffAugment plus EMA produced the frozen historical Exp11 result (FID 38.88). Because DiffAugment and EMA were introduced together, this is a combined intervention rather than two separately identified causal effects.

## Stage 4 ? CLIP continuation study

The CLIP stage starts from the Exp11 family and includes a formal evaluation control, a no-CLIP continuation control, and four CLIP-MMD strengths. C1 has the lowest local legacy FID (33.41), while C4 has the lowest CLIP MMD. The sweep is a metric-trade-off study, not evidence that CLIP universally improves image quality.

## Stage 5 ? Data-quality baseline and planned SDXL extension

The later audit found 21,551 paths but only 17,029 unique SHA-256 contents. B1 is the formal exact-unique baseline with FID 45.07. It is a new data-quality boundary and is not directly comparable to the historical path-based 20K Exp11 result.

The SDXL pilot, candidate cleaning, M20/M50 mixtures, and unified evaluation are planned but are not completed results in this snapshot.

## How to read the repository

- `results_summary.csv` is the short interview-facing table.
- `docs/baseline_map.md` explains the preliminary baseline, Phase 2 no-added-module baseline, Exp11/B0 frozen baseline, and B1 clean-unique data baseline; do not collapse them into one number.
- `03_metrics_and_logs/` preserves the wider historical metric record.
- `02_selected_experiments/full_process/` now contains the earlier source scripts, not only the final selected scripts.
- `04_visual_assets/qualitative_samples_compact.png` shows one labeled sample grid per major milestone, with the no-module baseline explicitly labeled.
