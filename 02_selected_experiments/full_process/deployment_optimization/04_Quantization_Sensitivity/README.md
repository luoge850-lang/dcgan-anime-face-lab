# Task 4：量化敏感度与混合精度

当前 Generator 主图已经审计为 `ConvTranspose + BatchNormalization + Relu + Tanh`，没有真实的小波分支，也没有动态 Spectral Normalization 节点。因此本阶段只测试图中真实存在的五个 `ConvTranspose` 层，不虚构 wavelet/SN 敏感度结论。

## 04A：逐层恢复 FP16 的敏感度筛选

`04A_Layer_Sensitivity.py` 以 03C 的 PTQ 思路为基础，从同一个 raw ONNX 重新生成多个 Q/DQ 图。每个变体只用 `nodes_to_exclude` 排除一个 `ConvTranspose` 节点；该节点恢复为 FP16，其余可量化层继续保持 INT8。随后用与 03C 相同的 TensorRT Q/DQ builder 构建 engine，使用 03A 的固定 latent 和 `real_eval/` 计算 Standard FID、模糊率、TensorRT 延迟、吞吐量和显存。

输入接口沿用 03 的打包方式：通常只需要 raw `generator.onnx`，脚本会自动识别并解压 `Task3_03A_Quantization_Protocol.zip`、`Task3_03B_FP32_FP16_Engines.zip` 和 `Task3_03C_INT8_PTQ_Engine.zip`。只有 Kaggle 中存在多个不同版本时，才需要显式传 `--task3-bundle`、`--fp32-engine` 或 `--int8-engine`。

脚本支持直接作为 Kaggle Notebook 单元运行；它不会依赖 `__file__`，会自动定位 `/kaggle/working` 或 `/kaggle/input` 中的 Task-3 helper。也可以设置 `DEPLOYMENT_OPTIMIZATION_ROOT` 指向 `Deployment_Optimization` 根目录。

输出采用一个主 CSV 加一个 manifest，避免为每一层堆积 JSON：

- `layer_sensitivity_summary.csv`：FP32、全 INT8、逐层 FP16 恢复的 FID/模糊率/延迟/吞吐量/显存，以及相对于 FP32 的损失和相对于全 INT8 的 FID/模糊率恢复量；
- `layer_index_precision_loss.png`：层索引—FID 损失和层索引—模糊率损失曲线；
- `sensitivity_source_named.onnx`：给候选 ConvTranspose 节点写入稳定名字，保证 ModelOpt 的排除规则不会因导出器命名变化而错层；
- `layer_sensitivity_manifest.json`：输入哈希、候选层、评估样本数、校准方法和状态；
- `Task4_04A_Layer_Sensitivity.zip`：上传后续分析所需的完整结果包。

第一次可以用 `--n-fid 1000` 做筛选；最终报告中对入选层或混合策略再用 `--n-fid 5000` 复核。04A 只负责找敏感层，04B 才负责组合 Pareto 最优策略，04C 负责汇总报告。

## 01D v2：目标 GPU 融合审计

新的 `01_ONNX_Fusion/05_TRT_Targeted_Fusion_Exp11.py` 是原 01D 的替代实验，不覆盖旧的 Subpixel 结果。它把推理期 BN 参数安全折叠进前面的 ConvTranspose，然后分别用 TensorRT FP32/FP16 构建 raw/fused engine，在 GPU 上测端到端延迟和 TensorRT layer profile，并用同一 latent 检查最大绝对误差。

该实验的成功标准是“数值等价且更快”，不是画质提升。语义保持的融合不会凭空增加 FID 质量；如果它改变了 FID 或视觉结果，反而说明融合实现或测试协议有问题。若目标是改善 INT8 画质，应使用本目录的混合精度和后续 QAT。
