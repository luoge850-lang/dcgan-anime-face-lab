# Metric Protocol and Interpretation

## Is the existing FID calculation wrong?

Not necessarily. The implementation is useful as a consistent internal metric: it extracts Inception-v3 features, estimates real/fake means and covariances, and applies the FID formula. The key issue is protocol comparability, not that every historical number is numerically invalid.

## Limitations of the frozen values

1. The pipeline is the project's legacy torchvision Inception-v3 protocol.
2. Preprocessing and feature extraction are not guaranteed to match clean-fid or torch-fidelity defaults.
3. Historical real images are largely drawn from the training distribution; they are not a strict unseen holdout.
4. Fake images are sampled stochastically, so a single evaluation has sampling noise.
5. Dataset size and training protocol change across phases, so cross-phase comparisons must follow the experiment table rather than only the final number.
6. The historical LPIPS field in some files is an AlexNet feature-distance proxy, not calibrated LPIPS.

## How to use the results

- Keep the existing values unchanged for project continuity.
- Label them `fid_legacy_project` in public reports.
- Do not compare them directly with published clean-fid numbers.
- For future work, add `fid_standardized` beside the legacy metric instead of overwriting old results.
- Add a fixed holdout split, multiple seeds, and nearest-neighbor checks before making generalization claims.
