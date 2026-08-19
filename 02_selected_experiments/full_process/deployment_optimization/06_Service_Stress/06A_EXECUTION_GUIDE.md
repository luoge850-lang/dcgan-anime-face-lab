# 06A：Kaggle 单进程 QAT TensorRT 服务

## 上传到 Kaggle

1. `06A_service.py`
2. `generator_trt_qat_int8.engine`
3. 可选：`qat_evaluation_manifest_revised_acceptance.json`，用于记录 engine SHA256 和前置指标

不需要上传 `.pth`、ONNX、训练数据或 Locust 脚本。06A 只验证已编译 engine 的服务化加载和 smoke test。

## Kaggle 依赖

在 TensorRT 第一次 import 之前安装与 05B 构建一致的 TensorRT。当前 QAT engine 的实验记录使用 TensorRT 11.2 系列；如果 Kaggle 已经 import 了其他 TensorRT，必须重启 session。

依赖由 `06A_service.py` 在脚本内部自动检查和安装，默认包括：

```text
numpy
torch
Pillow
fastapi
uvicorn
requests
tensorrt-cu12==11.2.1.2
```

Kaggle 需要打开 Internet。脚本会在首次 `import tensorrt` 前安装 TensorRT；
如果当前 Notebook 进程已经提前 import 了其他 TensorRT，必须重启 session，
并以新的 Python 进程运行 06A。

TensorRT 不要盲目安装最新版，应与 engine 构建版本一致。运行前检查：

```python
import tensorrt as trt
print(trt.__version__)
```

## 启动

```bash
python 06A_service.py \
  --engine /kaggle/input/<dataset>/generator_trt_qat_int8.engine \
  --output-dir /kaggle/working/dcgan_output/Deployment_Optimization_Results/06_Service_Stress/06A_Service \
  --host 0.0.0.0 \
  --port 8000 \
  --gpu 0 \
  --warmup 20 \
  --smoke-requests 10 \
  --tensorrt-version 11.2.1.2
```

只做依赖、engine 反序列化和 smoke test，不启动服务：

```bash
python 06A_service.py --smoke-only
```

真正供 06B Locust 调用时，使用正常启动模式，并保持单 worker。

调试时可以加 `--skip-dependency-install`，正式实验不要使用该参数。

如果把脚本直接粘贴到 Kaggle Notebook 单元运行，06A 会忽略 Notebook 自动注入的
`-f kernel.json` 等参数；如果没有提供 `--engine`，脚本会在 `/kaggle/input` 中
自动寻找唯一的 `*qat*int8*.engine`。如果找到多个 engine，应显式传入 `--engine`。

06A 使用单进程、单 worker、单 execution context、batch=1。并发数表示 HTTP 请求并发，不表示 TensorRT 动态 batch。

## 接口

```text
GET  http://127.0.0.1:8000/health
POST http://127.0.0.1:8000/generate
Content-Type: application/json
{"seed": 1234}
```

`/generate` 返回 `image/png`，响应头包含 `X-Inference-Ms`。

## 06A 输出

- `service_manifest.json`：GPU、CUDA、TensorRT、engine hash、binding、warm-up 和服务配置
- `service_smoke_test.csv`：逐请求 seed、延迟、输出 shape、输出范围、PNG 大小和错误
- `service.log`：由 Uvicorn 输出的服务日志

## 通过标准

- engine 反序列化成功；
- `/health` 返回 200；
- smoke requests 全部成功；
- 输出 shape 为 `[1,3,64,64]`；
- 输出不是 NaN、全黑或全白；
- warm-up 后进程和 GPU 显存没有异常增长；
- 服务以一个 worker 启动，不能使用 `--workers > 1`。

## 下载回本地

下载整个 06A 输出目录，以及 QAT engine 本身：

```text
service_manifest.json
service_smoke_test.csv
service.log
generator_trt_qat_int8.engine
```

如果 engine 文件 SHA256 与 05B manifest 中的
`b24e799fc5e32a7392bb587e23c314f290c6f314934d00f1cd4cde9e92bf5282`
不一致，应保留实际 hash，并在报告中说明这是重新上传或重新构建的 engine。

## HTTP 独立验证单元（报告证据）

06A 的直接 smoke test 验证的是 `service.infer()`；下面两个 Notebook 单元验证完整的 HTTP 链路。必须在完整启动 06A 后执行，不能使用 `--smoke-only`。

### 单元 1：验证 `/health`

```python
import requests
import time

BASE_URL = "http://127.0.0.1:8000"
health = None
for _ in range(60):
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            break
    except Exception:
        pass
    time.sleep(1)

if health is None or health.get("status") != "ok":
    raise RuntimeError("/health 验证失败")
print("/health PASS", response.status_code, health["status"])
```

### 单元 2：验证 `/generate`

```python
import csv
import io
import time
from pathlib import Path
import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"
rows = []
for index in range(10):
    seed = 10000 + index
    started = time.perf_counter()
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"seed": seed},
        timeout=30,
    )
    http_ms = (time.perf_counter() - started) * 1000.0
    if response.status_code != 200:
        raise RuntimeError(f"/generate 失败: {response.status_code} {response.text}")
    if not response.headers.get("content-type", "").startswith("image/png"):
        raise RuntimeError("/generate 返回类型不是 image/png")
    image = Image.open(io.BytesIO(response.content))
    image.load()
    if image.size != (64, 64):
        raise RuntimeError(f"/generate 输出尺寸错误: {image.size}")
    rows.append({
        "request_id": index,
        "seed": seed,
        "status": response.status_code,
        "http_latency_ms": http_ms,
        "trt_inference_ms": response.headers.get("X-Inference-Ms", ""),
        "content_type": response.headers.get("content-type", ""),
        "image_width": image.width,
        "image_height": image.height,
        "bytes": len(response.content),
    })

output_dir = Path("/kaggle/working/dcgan_output/Deployment_Optimization_Results/06_Service_Stress/06A")
output_dir.mkdir(parents=True, exist_ok=True)
with (output_dir / "http_smoke_test.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print("/generate PASS", len(rows), "/ 10")
print(output_dir / "http_smoke_test.csv")
```

报告判定：`/health` 返回 200 且状态为 `ok`；`/generate` 连续 10 次均返回 200、`image/png` 和 64x64 图像；`http_smoke_test.csv` 作为 HTTP 服务正确性的证据。
