# Source Figure Gallery

This gallery uses every SVG that is physically present in the scanned `dcgan_lab/results/figures` directory: seven DCGAN-core figures and 26 deployment/quantization/service figures. The images are copied without replotting; only their public filenames are normalized for GitHub readability.

## DCGAN core

![Epoch budget FID](../04_visual_assets/dcgan_core/01_epoch_fid.svg)

![Augmentation FID](../04_visual_assets/dcgan_core/02_augmentation_fid.svg)

![Generator strengthening FID](../04_visual_assets/dcgan_core/03_generator_strengthening_fid.svg)

![Generator strengthening LPIPS diversity](../04_visual_assets/dcgan_core/04_generator_strengthening_lpips.svg)

![Generator-side deep tuning FID](../04_visual_assets/dcgan_core/05_deep_tuning_generator_fid.svg)

![Discriminator-side deep tuning FID](../04_visual_assets/dcgan_core/06_deep_tuning_discriminator_fid.svg)

![CLIP lambda FID](../04_visual_assets/dcgan_core/07_clip_lambda_fid.svg)

## Deployment and quantization

![BN unfused latency](../04_visual_assets/source_figures/deployment_quantization_service/17_bn_unfused_latency.svg)

![BN fused latency](../04_visual_assets/source_figures/deployment_quantization_service/18_bn_fused_latency.svg)

![FP32 engine latency](../04_visual_assets/source_figures/deployment_quantization_service/19_engine_fp32_latency.svg)

![FP32 engine throughput](../04_visual_assets/source_figures/deployment_quantization_service/20_engine_fp32_throughput.svg)

![FP16 engine latency](../04_visual_assets/source_figures/deployment_quantization_service/21_engine_fp16_latency.svg)

![FP16 engine throughput](../04_visual_assets/source_figures/deployment_quantization_service/22_engine_fp16_throughput.svg)

![PTQ FID](../04_visual_assets/source_figures/deployment_quantization_service/22_ptq_fid.svg)

![PTQ blur rate](../04_visual_assets/source_figures/deployment_quantization_service/23_ptq_blur_rate.svg)

![TensorRT top three operators](../04_visual_assets/source_figures/deployment_quantization_service/23_tensorrt_top3_operators.svg)

![PTQ throughput](../04_visual_assets/source_figures/deployment_quantization_service/24_ptq_throughput.svg)

![Layer sensitivity FID](../04_visual_assets/source_figures/deployment_quantization_service/25_layer_sensitivity_fid.svg)

![Layer sensitivity latency](../04_visual_assets/source_figures/deployment_quantization_service/26_layer_sensitivity_latency.svg)

![Mixed precision FID](../04_visual_assets/source_figures/deployment_quantization_service/27_mixed_precision_fid.svg)

![Mixed precision latency](../04_visual_assets/source_figures/deployment_quantization_service/28_mixed_precision_latency.svg)

![QAT FID](../04_visual_assets/source_figures/deployment_quantization_service/29_qat_fid.svg)

![QAT blur rate](../04_visual_assets/source_figures/deployment_quantization_service/30_qat_blur_rate.svg)

![QAT P99](../04_visual_assets/source_figures/deployment_quantization_service/31_qat_p99.svg)

## Service and resource behavior

![Service concurrency P99](../04_visual_assets/source_figures/deployment_quantization_service/32_service_concurrency_p99.svg)

![Service concurrency RPS](../04_visual_assets/source_figures/deployment_quantization_service/33_service_concurrency_rps.svg)

![Service concurrency GPU memory](../04_visual_assets/source_figures/deployment_quantization_service/34_service_concurrency_gpu_memory.svg)

![Service concurrency RSS](../04_visual_assets/source_figures/deployment_quantization_service/35_service_concurrency_rss.svg)

![Service concurrency SM](../04_visual_assets/source_figures/deployment_quantization_service/36_service_concurrency_sm.svg)

The following four files are retained because they exist in the requested source folder, but they are not used as evidence for the latest staged service claim. The scanned current `06D` result directory contains no corresponding raw result files.

![Soak P99 source figure](../04_visual_assets/source_figures/deployment_quantization_service/37_soak_p99.svg)

![Soak RPS source figure](../04_visual_assets/source_figures/deployment_quantization_service/38_soak_rps.svg)

![Soak GPU memory source figure](../04_visual_assets/source_figures/deployment_quantization_service/39_soak_gpu_memory_head_tail.svg)

![Soak RSS source figure](../04_visual_assets/source_figures/deployment_quantization_service/40_soak_rss_head_tail.svg)
