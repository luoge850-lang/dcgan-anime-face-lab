# 05 QAT 结果目录

当前目录保留 05A/05B 的运行产物。代码完成不等于实验完成；只有目录中出现真实 `qat_engine_manifest.json`、`qat_evaluation_manifest.json` 和对应 CSV/PNG 后，才可以报告 QAT 画质结论。

## 05A 产出

默认子目录：`05ABC_QAT_Pipeline/`

- `qat_best.pth`、`qat_training_log.csv`、`qat_training_curves.png`
- `generator_qat_hybrid.onnx`、`generator_qat_hybrid_mixed.onnx`
- `generator_trt_qat_int8.engine`
- `qat_export_manifest.json`、`qat_engine_manifest.json`、`qat_manifest.json`
- `qat_engine_smoke_test.csv`、`qat_engine_layer_precision.csv/json`
- `Task5_05ABC_QAT_Pipeline.zip`

05A 负责 FakeQuantize 微调、ONNX/ORT parity、TensorRT 构建、smoke test 和实际层精度审计。

## 05B 产出

默认子目录：`05B_Evaluation/`

- `qat_before_after_summary.csv`：PTQ_INT8（QAT 前）与 QAT_INT8（QAT 后）
- `qat_vs_ptq_summary.csv`：FP32/PTQ/mixed/QAT 完整质量表
- `qat_dynamic_benchmark.csv`：动态 batch 延迟、吞吐和显存
- `qat_vs_ptq_summary.png`、`qat_quality_speed_curves.png`、`qat_visual_comparison.png`
- `qat_evaluation_manifest.json`
- `Task5_05B_QAT_Evaluation.zip`

05B 只做固定协议下的质量和速度复测，不训练、不校准、不重建 engine。它使用 05A 的 QAT engine，并与 03E 的 FP32/PTQ-INT8、04C 的混合精度 engine 对比。

在实际运行前，本目录不写入“QAT 已改善”的结论。
