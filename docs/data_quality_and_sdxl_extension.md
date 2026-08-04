# Data quality audit and SDXL extension

## Completed evidence in this snapshot

The active Kaggle workspace added a data-quality audit before the next round of training. The audit found:

| Audit field | Value |
|---|---:|
| Image paths scanned | 21,551 |
| Unique SHA-256 contents | 17,029 |
| Exact duplicate groups | 3,626 |
| Redundant copies beyond one per group | 4,522 |
| Bad files | 0 |
| Image size | 64x64 |

The public snapshot keeps the audit summary, duplicate-group record, bad-file report, and the unique-content manifest. It does not include the private dataset or model weights.

## B1 formal clean-unique baseline

`B1_Formal_CleanUnique_17K` keeps the Exp11 recipe fixed—Width x3 generator, SN-Hinge discriminator, DiffAugment, EMA, seed 42, batch size 32, and 200 epochs—but trains on one path per exact SHA-256 content.

The recorded legacy project FID is `45.07`. This is a data-quality baseline, not a direct replacement for the historical Exp11 `38.88`: the data pool changed from the earlier approximately-20K path-based run to 17,029 exact-unique contents. The comparison is therefore labeled separately in `results_summary.csv`.

The B1 record includes the sanitized training entry point, configuration, metrics, loss log, manifest digest, and a fixed-noise sample grid. The generated checkpoint remains outside Git history.

## What is not yet complete

The following SDXL extension stages are planned but are not represented as completed results in this release:

- SDXL pilot generation;
- candidate cleaning and accepted/rejected manifests;
- M20 and M50 mixture runs;
- unified Legacy FID, Clean-FID, coverage, blur, and diversity evaluation.

The B1 baseline is the formal M0 control for those future mixtures. Do not create another duplicate baseline with the old 21,551-path protocol.

## Historical Kaggle commands

The scripts remain Kaggle-oriented, but personal dataset-owner paths have been removed. Set the mounted dataset path explicitly when it differs from the default:

```bash
export ANIME_FACES_DATASET=/kaggle/input/your-anime-faces/data
python 01_public_core/phase6_audit_original_dataset.py
python 01_public_core/phase6_b1_formal_clean_unique_17k.py
```

These commands document the historical workflow; they do not make the repository a local CPU training package.
