# CLIP experiment guide

The CLIP continuation scripts are self-contained Kaggle-oriented entry points. The checked-in files are:

- `clip_E0_formal_eval.py`: evaluation-only control for the frozen Exp11 EMA generator;
- `clip_C0_no_clip_control.py`: 50-epoch continuation with lambda=0;
- `clip_C1_lambda_001.py`: CLIP-MMD continuation with lambda=0.01;
- `clip_C3_lambda_005.py`: CLIP-MMD continuation with lambda=0.05;
- `clip_C4_lambda_010.py`: CLIP-MMD continuation with lambda=0.10.

The matching metrics and configurations are under `03_metrics_and_logs/phase5_clip/`. The scripts expect a Kaggle image Dataset and, for continuation runs, attached Exp11 generator/discriminator weights. They auto-detect Kaggle-mounted inputs when invoked with the `auto` settings.

The sweep is deliberately interpreted as a matched-control study. C0 is the no-CLIP control. C1 has the lowest legacy FID in the local sweep, while C4 has the lowest CLIP MMD². These are different objectives and should not be collapsed into one claim.

Older one-click notebook names and internal reports are not part of this snapshot and are not referenced as reproduction targets.
