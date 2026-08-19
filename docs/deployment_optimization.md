# Deployment optimization: measured evidence and boundaries

The deployment phase was added after the historical DCGAN training experiments. Its target is the current unconditional Generator, not the discriminator and not every experimental feature tested during training.

## Actual inference graph

The exported Generator is a standard operator graph:

```text
z [B, 128, 1, 1]
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + BatchNorm + ReLU
  -> ConvTranspose + Tanh
  -> image [B, 3, 64, 64]
```

The independent Haar-wavelet and dynamic-SN probes are not nodes in this current main graph. They should be described as ONNX-compatible replacement probes, not as deployed custom operators.

## Quality and speed

Task 3 provides the cleanest FP32/FP16/INT8 quality baseline. FP16 is effectively lossless under the recorded Standard FID protocol. INT8 improves throughput but increases Standard FID from 29.9911 to 35.3198 and blur rate from 12.0% to 12.5%.

Task 4 restores selected sensitive layers to FP16. The final `net.0 + net.12` strategy reaches Standard FID 31.1776 at approximately 23.97K images/s in the recorded final-confirmation benchmark. This is the strongest archived quality-speed trade-off for the actual graph.

Task 5 QAT uses FakeQuantize and a perceptual/distillation-style objective. It improves over all-INT8 PTQ, but the revised acceptance does not establish superiority over mixed-precision PTQ or prove better hair/eyeliner high-frequency detail.

## Service stress

Task 6 now has measured preflight and staged-load evidence. The staged run uses a single process, single worker, single TensorRT context, HTTP request concurrency, and fixed engine batch 1. It reaches concurrency 128 with zero failures and no hard crash. P99 increases from 5 ms at concurrency 1 to 490 ms at 128, which is evidence of latency pressure rather than a crash threshold.

The 5-second monitor recorded 100 samples. Peak GPU memory was 677.2 MB and peak service RSS was 1,077.8 MB. Because the archived run is staged rather than a 30-minute soak, it is not sufficient to claim that long-running memory leaks have been ruled out.

## Reproduction boundary

The measurements were produced in Kaggle with the runtime and engine metadata recorded in the manifests. The public snapshot keeps the scripts, metrics, summaries, and a small raw stress-run archive, but excludes large engine/ONNX/checkpoint binaries. Host RSS and whole-device CUDA memory are different measurement types and must not be compared as one unified memory metric.
