# 01E：含真实小波/SN 的 Generator 模型变体（可选）

## 状态

`optional_not_started`。本目录不是当前 Exp11 主路线的必做步骤。

## 为什么单独建目录

当前 `01A_Export` 的主 ONNX 图包含 5 个 `ConvTranspose`、4 个 `BatchNormalization`、4 个 `Relu` 和 1 个 `Tanh`，`custom_nodes=[]`。现有 Haar 小波和动态 SN 结果是独立替换探针，证明了它们可以被改写为标准 ONNX 算子，但不证明它们存在于 Generator 主推理图。

如果将小波或动态 SN 真正加入 Generator，模型结构、权重、激活范围和最终输出分布都会改变。因此必须使用独立目录，避免把新模型和现有 Exp11 的结果混在一起。

## 何时才需要执行

只有在实验要求明确规定“Generator 主推理图必须包含真实小波/SN 分支”时才执行。若目标只是完成量化敏感度分析，推荐继续使用当前 Exp11 主模型，并把 Haar 高频子带分析作为诊断指标。

## 必须重新完成的阶段

```text
PyTorch 模型变体
→ 训练或微调
→ ONNX 导出与 checker
→ 自定义逻辑标准算子化
→ 融合与速度比较
→ ORT/TensorRT/OpenVINO benchmark
→ FP32/FP16/INT8 PTQ
→ FID/模糊率/LPIPS/高频指标
```

旧 Exp11 结果只能作为无小波/SN 的对照组，不能直接宣称新模型继承旧模型的 FID、吞吐量或量化误差。
