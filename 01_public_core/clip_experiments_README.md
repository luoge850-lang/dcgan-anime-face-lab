# CLIP 实验目录

## 一键正式入口

- `00_E0_Exp11_只评估_一键运行.py`
- `01_C0_无CLIP继续训练50epoch_一键运行.py`
- `02_C1_CLIP_L001_50epoch_一键运行.py`
- `03_C2_CLIP_L0025_50epoch_一键运行.py`
- `04_C3_CLIP_L005_50epoch_一键运行.py`
- `05_C4_CLIP_L010_50epoch_一键运行.py`

每个文件都是完整、相互独立的正式脚本。把任意一个文件的全部内容直接粘贴到
Kaggle Notebook 的空白代码单元并运行即可，不需要 `%%writefile`，不需要安装
命令，也不需要再输入训练参数。

脚本会自动：

1. 安装缺少的 `open_clip_torch` 和 `lpips`；
2. 在 `/kaggle/input` 中寻找唯一的 Exp11 G/D 权重；
3. 寻找唯一一个至少有20K张图片的原始数据集；
4. 下载所需的 CLIP、Inception 和 LPIPS 预训练权重；
5. 完成指定实验、评估和全部文件保存；
6. 在 `/kaggle/working/` 生成同名 zip，供直接下载。

Kaggle Notebook 仍需人工选择 GPU 并打开 Internet；平台开关无法由 Python
脚本代替点击。

`CLIP_cmmd_finetune.py` 是用于维护和命令行复现的统一母版，不是当前最简操作
入口。

- `CLIP实验总报告与Kaggle完整执行指南.md`：合并后的唯一正式报告，包含实验
  原理、对照设计、Kaggle 全流程、保存清单、指标和结果表。

操作顺序：

```text
上传 Exp11 权重 Dataset
→ 上传/添加原始训练图像 Dataset
→ 新建GPU Notebook并打开Internet
→ 复制并运行 E0 一键脚本，下载E0 zip
→ 分别复制并运行 C0–C4 一键脚本，下载各自zip
→ 将下载结果交回本项目比较
→ 汇总 E0 与 C0–C4 的50 epoch结果
→ 选择并归档综合最优权重，结束本轮CLIP实验
```

C4 是本轮最终的 λ 强度边界组（λ=0.10）。本轮不进行长程续训，因此C4不保存
约142MB的完整训练checkpoint，只保存可直接采样的EMA Generator权重、配置、
训练日志、最终样图、指标和运行摘要。C0–C3已经下载的checkpoint无需删除，
可作为既有实验的可恢复证据保留。

## 历史材料

旧报告已移动到 `历史文档_已合并/`，仅供追溯。以下旧脚本不得用于正式结果：

- `CLIP_ft_00_baseline.py`
- `CLIP_ft_01_lambda001.py`
- `CLIP_ft_02_lambda005.py`
- `CLIP_ft_03_lambda010.py`

它们使用无条件 GAN 中不成立的随机真假图逐图 cosine 配对，并包含与 Exp11
不一致的 R1 和旧 EMA 实现。
