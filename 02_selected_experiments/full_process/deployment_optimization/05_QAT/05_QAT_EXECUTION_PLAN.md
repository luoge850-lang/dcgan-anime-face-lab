# 05A/05B QAT execution contract

This document is the run contract for the rewritten Task 5 scripts. It keeps
training, ONNX graph rewriting, TensorRT building, and metric evaluation as
separate stages.

## Fixed deployment policy

The Exp11 generator has five ConvTranspose layers:

| Layer | Training-time fake quantization | Deployment precision |
|---|---|---|
| `net.0` | none | FP16 |
| `net.3` | input and weight, signed per-tensor INT8 | INT8 |
| `net.6` | input and weight, signed per-tensor INT8 | INT8 |
| `net.9` | input and weight, signed per-tensor INT8 | INT8 |
| `net.12` | none | FP16 |

Weight scales are deliberately scalar. The previous per-channel ConvTranspose
Q/DQ layout triggered unsupported TensorRT tactics on the Kaggle runtime.
There is no output-side FakeQuantize after a ConvTranspose, so high-frequency
detail is not quantized a second time.

## Runtime rule

The baseline engine artifacts in this repository were built with TensorRT
`11.2.1.2`. TensorRT serialized engines are not portable between runtime
versions. 05B now checks the installed distribution and, if TensorRT has not
yet been imported, installs the requested version before the first import.
If another TensorRT version is already in `sys.modules`, a fresh Kaggle session
is still mandatory.

For CUDA 12, 05B performs this installation automatically. The equivalent
manual command is:

```python
!pip install -q --upgrade tensorrt-cu12==11.2.1.2 \
    --extra-index-url https://pypi.nvidia.com
import tensorrt as trt
print(trt.__version__)  # 11.2.1.2
```

If TensorRT 10.16 has already been imported, do not install 11.2 in that
kernel. Restart the Kaggle session first.

The automatic path uses `tensorrt-cu12==11.2.1.2` by default and records the
requested package in `qat_evaluation_manifest.json`. For CUDA 13, pass
`--tensorrt-pip-package tensorrt-cu13` before the first TensorRT import.

## 05A input and output

Inputs:

1. `generator_ema_final.pth`: the best Exp11 EMA baseline.
2. `Task3_03A_Quantization_Protocol.zip` or its extracted folder containing
   `latent_calibration.npy`, `latent_eval.npy`, and `real_eval/`.
3. Optional 04C `final_confirmation_manifest.json`, whose selected policy must
   be `net.0+net.12`. The 04C engine itself is consumed by 05B, not by 05A.

05A performs observer calibration and QAT fine-tuning only. It does not import
TensorRT, export ONNX, or build an engine.

Outputs under:

```text
/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05A_QAT_Training/
```

```text
qat_pre.pth                 calibrated FakeQuant model before fine-tuning
qat_best.pth                best post-QAT checkpoint selected by held-out latent_eval
qat_training_log.csv        step-0 and QAT validation metrics
qat_training_curves.png     training and high-frequency loss curves
qat_manifest.json            layout, hashes, counts, and selection evidence
Task5_05A_QAT_Training.zip  optional package
```

The training pool and checkpoint-selection pool are disjoint slices of
`latent_calibration.npy`. `latent_eval.npy` is only shape-checked by 05A and is
reserved exclusively for the final 05B test. This prevents both training and
checkpoint-selection leakage.

The primary training objective is output L1/MSE distillation plus Haar detail
and image-gradient loss. `net.9`/`net.12` feature loss, Laplacian loss, and
weight anchor are disabled by default and are available only as explicitly
recorded ablations.

05A success conditions:

```text
qat_manifest.json.status == complete
qat_manifest.json.qat_layout_version == 4
qat_manifest.json.training_revision == 3
fake_quant.quantized_layers == [net.3, net.6, net.9]
fake_quant.protected_layers == [net.0, net.12]
qat_pre.pth and qat_best.pth exist
```

## 05B input and output

05B reads the two checkpoints from the same Notebook's 05A output. It also
requires these three baseline engines, all built with TensorRT 11.2.1.2:

```text
generator_trt_fp32.engine                  03B reference
generator_trt_int8.engine                  03C all-INT8 PTQ
generator_trt_mixed_precision_final.engine 04C net.0+net.12 FP16 mixed PTQ
```

The same 03A protocol is required. 05B first runs
`qat_runtime_preflight.json`, deserializing the three baseline engines before
any FID setup or QAT engine build. A serialization mismatch is an environment
failure, not a QAT quality result.

04C supplies an engine only: `generator_trt_mixed_precision_final.engine`.
05B deserializes it as the `MIXED_PTQ` baseline and does not try to edit or
reuse the serialized engine as an ONNX graph. The QAT ONNX graphs are exported
and rewritten independently inside 05B, so no 04C ONNX upload is required.

For each of `qat_pre.pth` and `qat_best.pth`, 05B then performs:

```text
PyTorch FakeQuant checkpoint
    -> raw ONNX export
    -> scalar QAT weight materialization
    -> INT8 DQ for net.3/net.6/net.9
    -> explicit FP16 boundaries for net.0/net.12
    -> ONNX checker + shape inference + layer Q/DQ audit
    -> TensorRT engine built with the same loaded runtime
```

The default benchmark is 50 warmup iterations, 200 timed iterations, and 3
repeats for every batch. The reported P99 is computed over all repeated timed
samples, so a single unusually short 50-sample run cannot decide the speed
gate.

Outputs under:

```text
/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05B_Evaluation/
```

```text
qat_runtime_preflight.json      baseline engine/runtime compatibility
generator_*_qat_raw.onnx        raw ONNX for PRE_QAT and QAT_INT8
generator_*_qat_mixed.onnx      deployment graph after policy rewrite
generator_trt_pre_qat.engine    calibrated pre-QAT TensorRT engine
generator_trt_qat_int8.engine    trained QAT TensorRT engine
qat_before_after_summary.csv    PRE_QAT vs QAT_INT8 vs PTQ_INT8
qat_vs_ptq_summary.csv          FP32/PTQ/MIXED/PRE_QAT/QAT comparison
qat_dynamic_benchmark.csv       P50/P99/throughput/memory by batch
qat_quality_speed_curves.png    FID, blur, and P99 curves
qat_visual_comparison.png       identical-latent contact sheet
qat_evaluation_manifest.json    final status and all hashes/metrics
Task5_05B_QAT_Evaluation.zip    optional package
```

## Acceptance criteria

Under the revised report claim, the experiment is complete only when the final
manifest says `complete` and:

```text
QAT_INT8 FID < PTQ_INT8 FID
QAT_INT8 blur rate is not more than 0.5 percentage point worse than PTQ_INT8
QAT_INT8 P99 / PTQ_INT8 P99 <= 1.30
QAT_INT8 P99 / MIXED_PTQ P99 <= 1.15
```

Haar high-frequency MAE remains in the CSV and manifest as a diagnostic, not a
pass/fail gate. The repository has no hair/eyeliner ROI annotations; therefore
the valid visual claim is no obvious quality collapse in the identical-latent
contact sheet. A literal statistically significant hair/eyeliner improvement
claim is not made.
