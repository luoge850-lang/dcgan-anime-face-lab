# Metrics and logs

This directory contains copied JSON metrics and CSV training logs from the selected historical result directories. They are evidence artifacts, not a guarantee that every script is directly runnable outside Kaggle.

Use `results_summary.csv` as the curated index. Each row now records a comparison scope and an entry-point path when that script is present in the snapshot. The headline FID field is `fid_legacy_project`; read it together with the root [`metric_protocol.md`](../metric_protocol.md).

The logs can contain historical field names such as `LPIPS`. In older runs that field is an AlexNet feature-distance proxy rather than calibrated LPIPS. Do not compare metrics across phases without checking dataset size, evaluation protocol, and the model used for the metric.
