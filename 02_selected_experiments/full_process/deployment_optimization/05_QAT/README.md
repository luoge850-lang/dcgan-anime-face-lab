# 05：混合精度 QAT（重新设计版）

## 先说结论

之前的 05A 把 QAT 训练、ONNX 导出、ORT parity、TensorRT 构建和 engine 审计串成一条链。任何一个环节失败，都会看起来像“QAT 训练失败”。你遇到的两类错误其实分别来自：

- TensorRT 11.2 的 ConvTranspose per-channel Q/DQ tactic 失败；
- TensorRT 10.12 下旧脚本仍调用 `layer.precision` / `set_output_type`，和强类型 Cast 图冲突。

本版彻底拆开：

```text
05A：checkpoint + 03A latent
  -> PyTorch FakeQuantize QAT
  -> qat_pre.pth / qat_best.pth

05B：qat_pre / qat_best + 03B/03C/04C engines + 03A protocol
  -> 独立 ONNX 导出
  -> 每个 checkpoint 独立构建 TensorRT engine
  -> FID / 模糊率 / Haar 高频误差 / P99 / 吞吐量
```

05A 不再构建 TensorRT，所以 TensorRT 构建错误不会污染 QAT 训练结果。04C 的最终产物是 engine，05B 只将它作为 MIXED_PTQ 基线反序列化和测速；05B 自己导出并重写 Q/DQ 图，不需要 04C ONNX。

当前 Exp11 Generator 没有真实的小波或动态 SN 节点，因此本任务的实际敏感层是 5 个 `ConvTranspose`：`net.0/net.3/net.6/net.9/net.12`。沿用 04C 已确认的策略：`net.0 + net.12` FP16，`net.3/net.6/net.9` INT8。

## 需要上传到 Kaggle 的输入

上传或挂载以下 5 类文件：

| 输入 | 来源 | 用途 |
|---|---|---|
| `generator_ema_final.pth` | Exp11 最佳 Baseline/EMA checkpoint | 05A teacher 和 QAT student 初始权重 |
| `Task3_03A_Quantization_Protocol.zip` 或解压目录 | 03A | 固定 `latent_calibration.npy`、`latent_eval.npy`、`real_eval/` |
| `generator_trt_fp32.engine` | 03B | FP32 质量参考 |
| `generator_trt_int8.engine` | 03C | 全 INT8 PTQ 基线 |
| `generator_trt_mixed_precision_final.engine` | 04C | `net.0+net.12` FP16 的 Mixed PTQ 基线 |

不要把旧的 `05ABC_QAT_Pipeline.zip` 当作本版输入；本版 05A 会重新产生两个 checkpoint。

如果 05B 报出 `net.3.weight__qat_int8 + net.3.ConvTranspose`、`expandScalarTensor` 或 `vol == 1`，说明使用的是旧版逐通道权重 QAT checkpoint。必须先重新运行本版 05A；不能只重复运行 05B。

05B 还需要 Kaggle Internet 或预装依赖来安装/使用 `onnx`、`pytorch-fid` 和与 CUDA major 匹配的 TensorRT Python binding；05B 会自动补装 `onnx` 和 `pytorch-fid`，TensorRT 建议按下方版本步骤在首次 import 前安装。

## 05A：QAT 训练

脚本：`05A_QAT_Finetune.py`

```bash
python 05A_QAT_Finetune.py \
  --checkpoint /kaggle/input/<exp11>/generator_ema_final.pth \
  --protocol-path /kaggle/input/<task3-protocol>/ \
  --policy-manifest /kaggle/input/<04c-output>/final_confirmation_manifest.json \
  --output-dir /kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05A_QAT_Training \
  --steps 2500 --batch-size 32 --eval-every 100 \
  --lr 5e-6 --highfreq-weight 0.5 --gradient-weight 0.1
```

FakeQuantize 的具体位置：

- `net.3/net.6/net.9`：输入激活 per-tensor INT8 + 权重 per-tensor INT8（单一 scalar scale）；这是为 TensorRT ConvTranspose 的显式 Q/DQ 部署兼容性固定的布局；
- `net.0/net.12`：不插 INT8 FakeQuantize，作为部署 FP16 保护层；
- 不插 ConvTranspose 输出侧 FakeQuantize，避免高频细节被再次截断，也避免 TensorRT 把 Q/DQ 放在错误位置。

05A 训练目标是 Baseline teacher distillation + `net.9/net.12` 中间特征蒸馏 + Haar `LH/HL/HH` 高频损失 + 图像梯度损失 + Laplacian 损失 + 权重 anchor。BatchNorm 统计量固定；小学习率微调全部 Generator 权重，使模型可以主动补偿量化误差，而不是冻结高精度层后只让中间层被动承受误差。

05A 产出：

- `qat_pre.pth`：observer 校准后的 step-0 checkpoint，代表 QAT 训练前；
- `qat_best.pth`：验证目标最优的训练后 checkpoint；如果训练没有改善，会安全回滚到 step-0；
- `qat_training_log.csv`：step-0 和每次验证的 pixel/high-frequency/gradient/net.9/net.12/Laplacian/objective；
- `qat_training_curves.png`：训练曲线；
- `qat_manifest.json`：FakeQuant 布局、checkpoint hash、协议和最佳 step；
- `Task5_05A_QAT_Training.zip`：上传给 05B 的完整包。

## 05B：导出、构建和评估

脚本：`05B_Evaluate_QAT.py`

同一个 Notebook 中刚运行完 05A 时，不需要重新上传两个 `.pth`，05B 默认会读取：

```text
/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05A_QAT_Training/qat_pre.pth
/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05A_QAT_Training/qat_best.pth
```

推荐直接运行：

```bash
python 05B_Evaluate_QAT.py \
  --fp32-engine /kaggle/input/<task3>/generator_trt_fp32.engine \
  --ptq-int8-engine /kaggle/input/<task3>/generator_trt_int8.engine \
  --mixed-ptq-engine /kaggle/input/<04c-output>/generator_trt_mixed_precision_final.engine \
  --protocol-path /kaggle/input/<task3-protocol>/ \
  --tensorrt-version 11.2.1.2 \
  --output-dir /kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05B_Evaluation \
  --n-fid 5000 --n-image-eval 1000 --batch-size 32
```

如果 05A 使用了自定义输出目录，补充：

```bash
--qat-output-dir /kaggle/working/<你的05A输出目录>
```

如果 05A 输出目录不在当前 Notebook 的 `/kaggle/working`，也可以继续使用 `--pre-qat-checkpoint` 和 `--qat-checkpoint` 显式指定两个文件。05B 会优先读取当前 05A 输出，不会误选旧的 `/kaggle/input` checkpoint。

如果当前 Kaggle session 没有 TensorRT 11.2，先新建 session，再安装并运行：

```bash
pip install -q --upgrade tensorrt-cu12==11.2.1.2 --extra-index-url https://pypi.nvidia.com
```

CUDA 不是 12 时，把 `tensorrt-cu12` 换成当前 CUDA major 对应的 `tensorrt-cu11` 或 `tensorrt-cu13`。不能在已经 import 其他 TensorRT 版本的同一 Python kernel 中直接替换版本。

05B 对 `qat_pre.pth` 和 `qat_best.pth` 各自独立执行：PyTorch ONNX export → QAT 权重 materialize 为 INT8 initializer + DQ → `net.0/net.12` 插入显式 FP16 Cast 边界 → TensorRT 强类型构建。TensorRT 10.12+ 不再调用旧的 `layer.precision` 和 `set_output_type`。

主要产出：

- `qat_before_after_summary.csv`：`PRE_QAT`、`QAT_INT8` 和 `PTQ_INT8` 的 FID、模糊率、Haar 高频误差；
- `qat_vs_ptq_summary.csv`：FP32、全 INT8 PTQ、04C Mixed PTQ、PRE_QAT、QAT_INT8 完整对比；
- `qat_dynamic_benchmark.csv`：batch 1/8/32/64 的 mean、P50、P99、吞吐量和 CUDA 显存；
- `qat_quality_speed_curves.png`：FID、模糊率、P99 延迟曲线；
- `qat_visual_comparison.png`：同一批 latent 的可视化对照，用于观察发丝、眼线等高频纹理；
- `qat_evaluation_manifest.json`：输入 hash、Q/DQ 审计、TensorRT 版本、门槛和最终 status；
- `generator_trt_pre_qat.engine`、`generator_trt_qat_int8.engine`；
- `Task5_05B_QAT_Evaluation.zip`。

## 如何判断任务完成

不能只看脚本退出码，必须同时满足：

1. `qat_manifest.json` 中 `fake_quant.quantized_layers` 为 `net.3/net.6/net.9`，`protected_layers` 为 `net.0/net.12`，且 `trailing_output_fake_quant=false`。
2. `qat_pre.pth` 和 `qat_best.pth` 都存在，05A 日志中有 step-0 和训练后验证记录。
3. 05B 两个 QAT engine 都成功生成并能通过动态 batch smoke/evaluation；不能接受 `build_serialized_network returned None` 后继续写“完成”。
4. `qat_evaluation_manifest.json` 的 `status` 为 `complete`，且：
   - `QAT_INT8` FID 小于 `PTQ_INT8`；
   - QAT 模糊率不比 `PTQ_INT8` 高出超过 0.5 个百分点；
   - QAT P99 / 全 INT8 PTQ P99 ≤ 1.30，QAT P99 / Mixed PTQ P99 ≤ 1.15；
   - Haar 高频误差只作为诊断项，不宣称发丝/眼线显著改善。
5. `qat_visual_comparison.png` 中所有模型使用同一 latent；只有在有标注 ROI 时，才能把全局 Haar proxy 写成严格的“发丝/眼线 ROI 显著改善”。本仓库当前没有发丝/眼线标注，因此报告应写“高频代理指标 + 视觉 QC”，不能虚构 ROI 统计。

如果最终 `status=not_passed`，这不是脚本完成，而是实验结果没有满足门槛；应保留 manifest 和 diagnostics，先看是质量门槛还是 TensorRT 构建门槛失败。
