# Source-derived deployment and service figure gallery

These 33 SVGs are direct copies of the physically present files in `dcgan_lab/results/figures/`, renamed with readable ASCII filenames.

The group contains:

- ONNX/BN fusion and multi-engine latency/throughput figures;
- PTQ, layer sensitivity, mixed precision, and QAT figures;
- staged service concurrency/resource figures;
- fixed-batch service resource figures and four 60-minute soak figures, all backed by the current 06E summaries.

The current public service claim is based on the structured 06E evidence archived under `03_metrics_and_logs/deployment_optimization/06_Service_Stress/06E/`, plus the separate 06F dynamic-batching report. The charts show workload-scoped latency, throughput, resource, and head-tail observations; they do not establish a physical crash or saturation boundary.
