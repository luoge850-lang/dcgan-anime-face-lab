# 01D v2：TensorRT 目标 GPU 融合审计

旧的 `04_Subpixel_Reparameterization_Exp11.py` 是等价的 ConvTranspose→Conv+Shuffle 重参数化。它可能增加 Reshape/Transpose 开销，而且语义等价变换本来就不应提升 FID；因此本实验不再把“视觉质量提升”作为融合目标。

`05_TRT_Targeted_Fusion_Exp11.py` 只做安全的推理期 BN 折叠：

```text
ConvTranspose -> BatchNormalization -> ReLU
        ↓
ConvTranspose(with folded weight/bias) -> ReLU
```

它在目标 Kaggle GPU 上为 raw/fused 图构建 TensorRT engine，默认只测 FP32，使用同一批次、同一 warmup/iterations 和同一 TensorRT layer profiler，输出端到端延迟、吞吐量、显存、融合前后 layer time 和固定 latent 的输出误差。需要部署验证时再加 `--precisions FP32,FP16`。FP32 使用 `1e-5` 最大绝对误差门限；FP16 使用 `2e-3` 门限，避免将半精度舍入误差误判为结构错误。

脚本兼容 Kaggle Notebook 单元执行：Notebook 中没有 `__file__` 时，会自动搜索 `/kaggle/working`、`/kaggle/input` 和当前目录中的 helper；也可以通过环境变量 `DEPLOYMENT_OPTIMIZATION_ROOT` 指定实验根目录。

报告型运行默认只将汇总 CSV、算子 profile、数值等价 CSV、融合 ONNX 和 manifest 放进 ZIP；TensorRT engine 和显式 FP16 ONNX 只在加 `--package-engines` 后打包。

脚本现在是自包含的，和 01B/01C 一样只需要 raw `generator.onnx`；不再依赖同目录的 01C/02B helper，也不要求用户额外输入 engine 路径。
