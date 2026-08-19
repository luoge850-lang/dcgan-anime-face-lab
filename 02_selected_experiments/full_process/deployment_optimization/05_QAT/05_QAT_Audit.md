# 05 系列重构审计

## 根因判断

旧流程不是 QAT FakeQuantize 本身必然失败，而是把训练图、Q/DQ 图、FP16 Cast、TensorRT builder precision hint 和 engine 审计放在同一个脚本里。已保存的错误证据分别指向：

- TensorRT 11.2：`net.3.ConvTranspose` 的 per-channel INT8 权重 Q/DQ tactic 出现 `expandScalarTensor / vol == 1`，最终没有可用实现；
- TensorRT 10.12：强类型/显式 Cast 图仍被旧的 `layer.precision`、`set_output_type` 修改，触发 `castLayer.cpp::validateTypes` 类型断言。

因此本版的修复不是放宽 parity 阈值，而是拆分阶段、固定图布局，并在 TensorRT 10.12+ 使用强类型网络，不再调用旧 layer precision API。

本次 05B 日志再次出现同一错误，说明当前 `qat_pre.pth/qat_best.pth` 仍来自旧的逐通道权重布局；修改 05B 不能改变 checkpoint 内部已经保存的 FakeQuant scale，必须先用布局版本 4 的 05A 重新生成两个 checkpoint。

## 新脚本职责

| 文件 | 只读取 | 产出 |
|---|---|---|
| `05A_QAT_Finetune.py` | `generator_ema_final.pth`、03A latent、可选 04C policy manifest | `qat_pre.pth`、`qat_best.pth`、训练 CSV/PNG、`qat_manifest.json`、05A ZIP |
| `05B_Evaluate_QAT.py` | 当前 Notebook 的 05A 两个 checkpoint、03B FP32 engine、03C INT8 engine、04C Mixed engine、03A protocol | 两个 QAT ONNX/engine、FID/模糊率/高频/速度 CSV、图、evaluation manifest、05B ZIP |

05A 不导出 ONNX、不调用 ORT、不构建 TensorRT；05B 不导入 05A。04C 只提供已编译的 Mixed PTQ engine，05B 将其作为基线反序列化和测速；QAT 的 ONNX/Q-DQ 图由 05B 独立生成。05B 默认优先读取当前 Notebook 的 `/kaggle/working/.../05A_QAT_Training/qat_pre.pth` 和 `qat_best.pth`。

## 精度布局审计

- `net.0`、`net.12`：无 INT8 FakeQuantize；05B 输出显式 FP16 Cast 边界；
- `net.3`、`net.6`、`net.9`：输入激活 Q/DQ + 权重 Q/DQ；权重使用 scalar scale；
- 由于实际 Kaggle TensorRT 在 ConvTranspose 的逐通道权重 Q/DQ 上出现 `expandScalarTensor`，重构版改为单一 scalar weight scale；层级混合精度策略不变；
- 没有 ConvTranspose 输出侧 FakeQuantize；
- 05B 在 TensorRT 构建前运行 ONNX checker、shape inference 和逐层 Q/DQ 结构检查；
- 任何 QAT weight 未成功 materialize、层数不为 5 或 Q/DQ 布局不符合预期，直接失败并不生成“完成”报告。

## 结果验收

必须读取 `qat_evaluation_manifest.json`，而不是只看 notebook 最后一行：

```text
status == complete
QAT FID < PTQ FID
QAT blur rate is not more than 0.5 percentage point worse than PTQ blur
QAT P99 / all-INT8 PTQ P99 <= 1.30
QAT P99 / Mixed PTQ P99 <= 1.15
```

当前仓库没有发丝/眼线标注 ROI，因此 `roi_status` 必须保持 `not_available`；可以报告 Haar 高频代理和同 latent contact sheet，不能把它写成严格的发丝/眼线 ROI 显著性检验。
