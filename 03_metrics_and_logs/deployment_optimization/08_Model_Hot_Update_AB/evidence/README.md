# Stage 8 evidence package

This is a curated copy of the Kaggle output for a single process on one GPU. It preserves the validation summary, traffic split, per-version latency, sampled quality metrics, update events, and report.

Important boundary: the A/B FID values are from this rollout's 5,000-sample evaluation and are not the same evaluation instance as the canonical Stage 3–5 deployment table. The package does not include TensorRT engine binaries or claim a multi-replica Kubernetes rollout.

