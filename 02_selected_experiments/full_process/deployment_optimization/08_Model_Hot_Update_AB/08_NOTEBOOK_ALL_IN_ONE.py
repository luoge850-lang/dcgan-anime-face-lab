"""08: single-cell Kaggle experiment for model hot update and A/B rollout.

Paste this complete file into one Kaggle cell.  It reuses 06A's TensorRT
loader, but keeps two engine objects in one process so that candidate B can be
loaded and promoted without restarting the HTTP service.  All configuration,
logs and reports are generated in /kaggle/working; no yml/json upload is
needed.

This is a single-node production-pattern validation, not a Kubernetes or
multi-replica deployment.  The evidence explicitly records that limitation.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import csv
import hashlib
import importlib
import importlib.util
import io
import json
import os
import socket
import statistics
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

OUT = Path(os.getenv("AB_OUTPUT_DIR", "/kaggle/working/08_Model_Hot_Update_AB"))
HOST, PORT = "127.0.0.1", 8000
SERVICE_SERVER = None
STOP = threading.Event()


def pip_install():
    packages = (("requests", "requests"), ("fastapi", "fastapi"),
                ("uvicorn", "uvicorn"), ("pillow", "Pillow"),
                ("psutil", "psutil"), ("prometheus_client", "prometheus-client"),
                ("pytorch_fid", "pytorch-fid"))
    for module, package in packages:
        try:
            importlib.import_module(module)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            importlib.invalidate_caches()


def import_file(path: Path):
    spec = importlib.util.spec_from_file_location("dcgan_06a_for_08", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 06A: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover(args):
    root = Path("/kaggle/input")
    services = [Path(args.base_service)] if args.base_service else sorted(root.rglob("06A_service.py"))
    engines = sorted(root.rglob("*.engine"))
    if len(services) != 1:
        raise RuntimeError(f"06A_service.py 数量不明确，请设置 BASE_SERVICE_PATH: {services}")

    def resolve(explicit, patterns, excluded=()):
        if explicit:
            return Path(explicit).resolve()
        candidates = [p for p in engines if any(pattern in p.name.lower() for pattern in patterns)
                      and not any(word in p.name.lower() for word in excluded)]
        if len(candidates) == 1:
            return candidates[0].resolve()
        names = "\n".join(f"  - {p}" for p in candidates) or "  (none)"
        raise RuntimeError("Engine 候选不唯一，请设置显式路径:\n" + names)

    # Default A is the previous PTQ INT8 version; default B is the QAT INT8
    # candidate.  Explicit paths are recommended when multiple datasets exist.
    engine_a = resolve(args.engine_a, ("int8",), ("qat",))
    engine_b = resolve(args.engine_b, ("qat", "int8"))
    for path in (services[0], engine_a, engine_b):
        if not path.is_file():
            raise FileNotFoundError(path)
    if engine_a == engine_b:
        raise RuntimeError("A/B Engine 不能是同一个文件")
    return services[0].resolve(), engine_a, engine_b


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values, q):
    ordered = sorted(float(x) for x in values)
    if not ordered:
        return None
    index = max(0, min(len(ordered) - 1, int((len(ordered) * q + 0.999999) - 1)))
    return ordered[index]


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class ModelVersion:
    def __init__(self, label, path, base, trt, gpu):
        self.label = label
        self.path = path
        self.base = base
        self.engine = base.TensorRTGenerator(path, trt, gpu)
        self.lock = threading.Lock()
        self.loaded_at = time.time()

    def infer(self, seed):
        with self.lock:
            return self.engine.infer(int(seed))

    def metadata(self):
        data = self.engine.metadata()
        data.update({"version": self.label, "loaded_at": self.loaded_at})
        return data


class ABRuntime:
    def __init__(self, base, trt, engine_a, engine_b, gpu):
        from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry

        self.base, self.trt, self.gpu = base, trt, gpu
        self.models = {"A": ModelVersion("A", engine_a, base, trt, gpu)}
        self.candidate_path = engine_b
        self.lock = threading.RLock()
        self.ratio_b = 0.0
        self.events = []
        self.registry = CollectorRegistry()
        self.requests = Counter("dcgan_ab_requests_total", "Requests by model version", ["version", "status"], registry=self.registry)
        self.latency = Histogram("dcgan_ab_request_latency_seconds", "Latency by model version", ["version"], registry=self.registry)
        self.split = Gauge("dcgan_ab_candidate_ratio", "Configured candidate B ratio", registry=self.registry)
        self.loaded = Gauge("dcgan_ab_candidate_loaded", "Whether B is loaded", registry=self.registry)
        self.split.set(0.0)
        self.loaded.set(0.0)

    def event(self, action, **payload):
        row = {"timestamp": time.time(), "action": action, **payload}
        with self.lock:
            self.events.append(row)
        return row

    def load_candidate(self):
        with self.lock:
            if "B" in self.models:
                return False
        started = time.perf_counter()
        # Deserialize B outside the routing lock. A remains available while
        # CUDA context and execution context for B are constructed.
        candidate = ModelVersion("B", self.candidate_path, self.base, self.trt, self.gpu)
        with self.lock:
            self.models["B"] = candidate
            self.loaded.set(1.0)
        self.event("candidate_loaded", version="B", elapsed_ms=(time.perf_counter() - started) * 1000.0)
        return True

    def set_ratio(self, ratio, reason="manual"):
        ratio = max(0.0, min(1.0, float(ratio)))
        with self.lock:
            if ratio > 0 and "B" not in self.models:
                raise RuntimeError("候选 B 尚未加载")
            old = self.ratio_b
            self.ratio_b = ratio
            self.split.set(ratio)
        self.event("traffic_split_changed", old_ratio=old, new_ratio=ratio, reason=reason)
        return ratio

    def choose(self, request_id):
        with self.lock:
            ratio = self.ratio_b
            available = set(self.models)
        digest = int(hashlib.sha256(str(request_id).encode()).hexdigest()[:12], 16) / float(16**12)
        return "B" if "B" in available and digest < ratio else "A"

    def infer(self, seed, request_id):
        version = self.choose(request_id)
        with self.lock:
            model = self.models[version]
        started = time.perf_counter()
        try:
            output, engine_ms = model.infer(seed)
            elapsed = (time.perf_counter() - started) * 1000.0
            self.requests.labels(version, "200").inc()
            self.latency.labels(version).observe(elapsed / 1000.0)
            return version, output, engine_ms, elapsed
        except Exception:
            self.requests.labels(version, "500").inc()
            raise

    def health(self):
        with self.lock:
            return {"status": "ok", "pid": os.getpid(), "active_ratio_b": self.ratio_b,
                    "loaded_versions": sorted(self.models), "candidate_loaded": "B" in self.models,
                    "engine_a": self.models["A"].path.name,
                    "engine_b": self.candidate_path.name}


def image_png(output):
    import numpy as np
    from PIL import Image

    array = np.asarray(output, dtype=np.float32)
    if array.shape != (1, 3, 64, 64):
        raise RuntimeError(f"输出形状异常: {array.shape}")
    array = np.clip((array[0] + 1.0) * 127.5, 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(np.transpose(array, (1, 2, 0))).save(buffer, format="PNG")
    return buffer.getvalue()


def app_for(runtime: ABRuntime):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    globals()["Request"] = Request
    app = FastAPI(title="DCGAN Model Hot Update and A/B Service")

    @app.get("/health")
    def health():
        return runtime.health()

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(runtime.registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/generate")
    async def generate(request: Request):
        try:
            data = await request.json()
            seed = int(data.get("seed", 0))
            request_id = str(data.get("request_id") or request.headers.get("X-Request-ID") or f"auto-{time.time_ns()}")
            version, output, engine_ms, elapsed = await asyncio.to_thread(runtime.infer, seed, request_id)
            response = Response(image_png(output), media_type="image/png")
            response.headers["X-Model-Version"] = version
            response.headers["X-Inference-Ms"] = f"{engine_ms:.6f}"
            response.headers["X-Request-Latency-Ms"] = f"{elapsed:.6f}"
            return response
        except Exception as exc:
            return JSONResponse({"error": repr(exc)}, status_code=500)

    @app.post("/admin/load_candidate")
    async def load_candidate():
        try:
            loaded = await asyncio.to_thread(runtime.load_candidate)
            return {"status": "ok", "loaded_now": loaded, **runtime.health()}
        except Exception as exc:
            runtime.event("candidate_load_failed", error=repr(exc))
            return JSONResponse({"status": "failed", "error": repr(exc)}, status_code=500)

    @app.post("/admin/set_split")
    async def set_split(request: Request):
        data = await request.json()
        try:
            ratio = runtime.set_ratio(float(data.get("ratio_b", 0)), str(data.get("reason", "experiment")))
            return {"status": "ok", "ratio_b": ratio, **runtime.health()}
        except Exception as exc:
            return JSONResponse({"status": "failed", "error": repr(exc)}, status_code=400)

    @app.post("/admin/rollback")
    def rollback():
        ratio = runtime.set_ratio(0.0, "rollback")
        runtime.event("rollback_completed", active_version="A")
        return {"status": "ok", "ratio_b": ratio, **runtime.health()}

    return app


def wait(url, timeout=120):
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=5).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def client_request(index, phase):
    import requests
    started = time.perf_counter()
    request_id = f"{phase}-{index:06d}"
    try:
        response = requests.post(f"http://{HOST}:{PORT}/generate", json={"seed": 20260902 + index, "request_id": request_id}, timeout=120)
        elapsed = (time.perf_counter() - started) * 1000.0
        return {"phase": phase, "index": index, "request_id": request_id, "status": response.status_code,
                "version": response.headers.get("X-Model-Version", ""), "latency_ms": elapsed,
                "response_bytes": len(response.content)}
    except Exception as exc:
        return {"phase": phase, "index": index, "request_id": request_id, "status": "error",
                "version": "", "latency_ms": (time.perf_counter() - started) * 1000.0,
                "error": repr(exc)}


def request_batch(count, workers, phase):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(lambda i: client_request(i, phase), range(count)))


def fixed_health(label):
    import requests
    response = requests.get(f"http://{HOST}:{PORT}/health", timeout=30)
    payload = response.json()
    return {"label": label, "status_code": response.status_code, "pid": payload.get("pid"), "payload": payload}


def find_real_dir(explicit, count):
    if explicit:
        path = Path(explicit)
        if path.is_dir():
            return path
        if path.is_file() and path.suffix.lower() == ".zip":
            extracted = OUT / "protocol_03A_from_zip"
            extracted.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(path) as archive:
                archive.extractall(extracted)
            candidates = [p for p in extracted.rglob("real_eval") if p.is_dir()]
            if len(candidates) == 1:
                return candidates[0]
        raise FileNotFoundError(path)
    candidates = []
    for path in Path("/kaggle/input").rglob("real_eval"):
        if path.is_dir():
            n = sum(1 for p in path.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
            if n >= count:
                candidates.append(path)
    if len(candidates) == 1:
        return candidates[0]
    protocol_zips = sorted(Path("/kaggle/input").rglob("Task3_03A_Quantization_Protocol*.zip"))
    if len(protocol_zips) == 1:
        return find_real_dir(protocol_zips[0], count)
    names = "\n".join(f"  - {p}" for p in candidates) or "  (none)"
    zips = "\n".join(f"  - {p}" for p in protocol_zips) or "  (none)"
    raise RuntimeError("FID reference real_eval 不唯一或不存在，请设置 FID_REAL_DIR。real_eval candidates:\n" + names + "\nprotocol ZIP candidates:\n" + zips)


def real_images(real_dir, count):
    from PIL import Image
    import numpy as np
    import torch

    paths = sorted(p for p in real_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})[:count]
    if len(paths) < count:
        raise RuntimeError(f"真实图片只有 {len(paths)} 张，需要 {count} 张")
    values = []
    for path in paths:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        values.append(torch.from_numpy(array).permute(2, 0, 1))
    return torch.stack(values), paths


def laplacian_values(images01):
    import numpy as np
    values = []
    for image in images01:
        array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gray = array.mean(axis=2).astype(np.float32)
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
        values.append(float(lap.var()))
    return np.asarray(values, dtype=np.float64)


def fid_from_stats(real_mu, real_cov, fake_mu, fake_cov):
    from scipy import linalg
    import numpy as np
    diff = real_mu - fake_mu
    covmean = linalg.sqrtm(real_cov.dot(fake_cov))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(real_cov + fake_cov - 2.0 * covmean))


def feature_batches(net, images01, device, batch_size=32):
    import torch
    import torch.nn.functional as F
    values = []
    with torch.inference_mode():
        for start in range(0, len(images01), batch_size):
            batch = images01[start:start + batch_size].to(device)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            values.append(net(batch)[0].reshape(batch.shape[0], -1).cpu().numpy())
    import numpy as np
    return np.concatenate(values, axis=0)


def evaluate_fid(runtime, real_dir, count, seed):
    import numpy as np
    import torch
    from pytorch_fid.inception import InceptionV3

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    real, _ = real_images(real_dir, count)
    block = int(getattr(InceptionV3, "DEFAULT_BLOCK_INDEX", 3))
    net = InceptionV3([block]).eval().to(device)
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    real_features = feature_batches(net, real, device)
    real_mu, real_cov = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    rows = []
    seeds = [seed + i for i in range(count)]
    generated = {}
    for version in ("A", "B"):
        model = runtime.models.get(version)
        if model is None:
            raise RuntimeError(f"模型 {version} 未加载")
        images = []
        for value in seeds:
            output, _ = model.infer(value)
            images.append(torch.from_numpy(np.clip((output[0] + 1.0) * 0.5, 0, 1)))
        fake = torch.stack(images)
        fake_features = feature_batches(net, fake, device)
        fake_mu, fake_cov = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        lap = laplacian_values(fake)
        generated[version] = fake
        rows.append({"version": version, "fid_sample": fid_from_stats(real_mu, real_cov, fake_mu, fake_cov),
                     "blur_rate": None, "laplacian_mean": float(lap.mean()), "sample_count": count})
    threshold = float(np.percentile(laplacian_values(real), 10))
    for row in rows:
        lap = laplacian_values(generated[row["version"]])
        row["blur_rate"] = float((lap < threshold).mean())
        row["blur_threshold_real_p10"] = threshold
    return rows


def summarize_rows(rows):
    result = {}
    for version in ("A", "B"):
        values = [float(row["latency_ms"]) for row in rows if row.get("version") == version and row.get("status") == 200]
        result[version] = {"count": len(values), "p50_ms": percentile(values, .50), "p95_ms": percentile(values, .95),
                           "p99_ms": percentile(values, .99), "mean_ms": statistics.mean(values) if values else None,
                           "success_rate": (len(values) / max(1, sum(1 for row in rows if row.get("version") == version)))}
    return result


def run_experiment(runtime, args):
    import requests
    health = [fixed_health("before_update")]
    rows = []
    rows.extend(request_batch(args.baseline_requests, args.workers, "baseline_A"))
    # Keep A serving while B is deserialized in a separate thread. Requests
    # submitted before and during the admin call are the zero-downtime sample.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(client_request, i, "during_hot_load") for i in range(args.hot_load_requests)]
        time.sleep(0.1)
        load_response = requests.post(f"http://{HOST}:{PORT}/admin/load_candidate", timeout=300)
        rows.extend(future.result() for future in futures)
    load_payload = {"status_code": load_response.status_code, "payload": load_response.json()}
    health.append(fixed_health("after_candidate_load"))
    split_rows = []
    for ratio in (0.10, 0.50, 1.00):
        response = requests.post(f"http://{HOST}:{PORT}/admin/set_split", json={"ratio_b": ratio, "reason": f"gray_{ratio:.2f}"}, timeout=30)
        if response.status_code != 200:
            raise RuntimeError(f"切换灰度比例失败: {response.text}")
        batch = request_batch(args.gray_requests, args.workers, f"gray_{ratio:.2f}")
        rows.extend(batch)
        a_count = sum(1 for row in batch if row.get("version") == "A")
        b_count = sum(1 for row in batch if row.get("version") == "B")
        observed = b_count / max(1, a_count + b_count)
        split_rows.append({"target_ratio_b": ratio, "observed_ratio_b": observed,
                           "error_pp": abs(observed - ratio) * 100.0, "requests": len(batch),
                           "a_requests": a_count, "b_requests": b_count})
    rollback = requests.post(f"http://{HOST}:{PORT}/admin/rollback", timeout=30)
    rollback_payload = {"status_code": rollback.status_code, "payload": rollback.json()}
    after_rollback = request_batch(args.rollback_requests, args.workers, "after_rollback_A")
    rows.extend(after_rollback)
    health.append(fixed_health("after_rollback"))
    return rows, split_rows, health, load_payload, rollback_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-service", default=os.getenv("BASE_SERVICE_PATH", ""))
    parser.add_argument("--engine-a", default=os.getenv("ENGINE_A_PATH", ""))
    parser.add_argument("--engine-b", default=os.getenv("ENGINE_B_PATH", ""))
    parser.add_argument("--real-dir", default=os.getenv("FID_REAL_DIR", ""))
    parser.add_argument("--trt", default=os.getenv("TENSORRT_VERSION", "11.2.1.2"))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--baseline-requests", type=int, default=100)
    parser.add_argument("--hot-load-requests", type=int, default=100)
    parser.add_argument("--gray-requests", type=int, default=400)
    parser.add_argument("--rollback-requests", type=int, default=100)
    parser.add_argument("--fid-samples", type=int, default=int(os.getenv("AB_FID_SAMPLES", "5000")))
    parser.add_argument("--seed", type=int, default=20260902)
    args, _ = parser.parse_known_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pip_install()
    service_path, engine_a, engine_b = discover(args)
    base = import_file(service_path)
    base_args = SimpleNamespace(tensorrt_version=args.trt, tensorrt_pip_package="tensorrt-cu12",
                                skip_dependency_install=False, allow_version_mismatch=False)
    base.ensure_dependencies(base_args)
    trt = base.load_runtime(args.trt, False)
    runtime = ABRuntime(base, trt, engine_a, engine_b, args.gpu)
    # Warm up stable A before starting HTTP traffic.
    for i in range(10):
        runtime.models["A"].infer(args.seed + i)
    import uvicorn
    app = app_for(runtime)
    global SERVICE_SERVER
    SERVICE_SERVER = uvicorn.Server(uvicorn.Config(app, host=HOST, port=PORT, workers=1, log_level="info"))
    threading.Thread(target=lambda: asyncio.run(SERVICE_SERVER.serve()), daemon=True).start()
    if not wait(f"http://{HOST}:{PORT}/health"):
        raise RuntimeError("A/B service health check failed")
    import requests
    metrics_status = requests.get(f"http://{HOST}:{PORT}/metrics", timeout=30).status_code
    rows, split_rows, health, load_payload, rollback_payload = run_experiment(runtime, args)
    fid_rows, fid_error = [], None
    try:
        real_dir = find_real_dir(args.real_dir, args.fid_samples)
        fid_rows = evaluate_fid(runtime, real_dir, args.fid_samples, args.seed + 100000)
    except Exception as exc:
        fid_error = repr(exc)
    latency = summarize_rows(rows)
    health_ok = all(item["status_code"] == 200 and item["payload"].get("pid") == health[0]["pid"] for item in health)
    hot_rows = [row for row in rows if row["phase"] == "during_hot_load"]
    hot_ok = load_payload["status_code"] == 200 and all(row["status"] == 200 for row in hot_rows)
    split_ok = all(row["error_pp"] <= 5.0 for row in split_rows)
    rollback_ok = rollback_payload["status_code"] == 200 and all(row["version"] == "A" and row["status"] == 200 for row in rows if row["phase"] == "after_rollback_A")
    quality = {row["version"]: row for row in fid_rows}
    quality_ok = False
    if "A" in quality and "B" in quality:
        fid_ok = quality["B"]["fid_sample"] <= quality["A"]["fid_sample"] * 1.05
        blur_delta_pp = (quality["B"]["blur_rate"] - quality["A"]["blur_rate"]) * 100.0
        quality_ok = fid_ok and blur_delta_pp <= 0.5
    else:
        blur_delta_pp = None
    summary = {
        "status": "complete" if all((health_ok, metrics_status == 200, hot_ok, split_ok, rollback_ok, quality_ok)) else "incomplete",
        "scope": "single-node zero-downtime engine switch and deterministic A/B routing",
        "metrics_status": metrics_status,
        "hardware": runtime.models["A"].metadata(),
        "models": {"A": runtime.models["A"].metadata(), "B_path": str(engine_b), "B_sha256": sha256(engine_b)},
        "health_checks": health,
        "load_candidate": load_payload,
        "rollback": rollback_payload,
        "latency_by_version": latency,
        "traffic_split": split_rows,
        "fid_rows": fid_rows,
        "fid_error": fid_error,
        "criteria": {"same_pid_and_health": health_ok, "metrics_200": metrics_status == 200, "zero_downtime": hot_ok, "split_within_5pp": split_ok,
                     "rollback_to_A": rollback_ok, "fid_not_worse_than_5_percent": quality_ok,
                     "blur_delta_limit_pp": 0.5, "p99_limit_ms": 200.0},
        "limitations": ["Kaggle single process/single GPU; not a multi-replica production cluster", "FID is a sampled metric and must use the same protocol/reference set as earlier experiments"],
    }
    (OUT / "08_request_log.csv").write_text("", encoding="utf-8")
    write_csv(OUT / "08_request_log.csv", rows)
    write_csv(OUT / "08_traffic_split.csv", split_rows)
    write_csv(OUT / "08_fid_sample.csv", fid_rows)
    write_csv(OUT / "08_latency_by_version.csv", [{"version": v, **data} for v, data in latency.items()])
    (OUT / "08_update_events.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in runtime.events) + "\n", encoding="utf-8")
    (OUT / "08_validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report = ["# 08 模型热更新与 A/B 灰度实验", "", f"严格状态：**{summary['status']}**", "", "## 结论", ""]
    report.append("本实验在单进程单 GPU 环境验证候选 Engine 的不停机加载、按比例路由和回滚。")
    report.append(f"FID/模糊率质量门槛：{'通过' if quality_ok else '未通过或不可用'}。")
    report.append("该结果不能外推为多副本 Kubernetes 生产集群。")
    (OUT / "08_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    wanted = ["08_validation_summary.json", "08_request_log.csv", "08_traffic_split.csv", "08_fid_sample.csv", "08_latency_by_version.csv", "08_update_events.jsonl", "08_report.md"]
    with zipfile.ZipFile(OUT.parent / "08_Model_Hot_Update_AB_evidence.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for name in wanted:
            path = OUT / name
            if path.is_file():
                archive.write(path, name)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[08] 证据已写入", OUT)


@atexit.register
def cleanup():
    STOP.set()
    if SERVICE_SERVER is not None:
        SERVICE_SERVER.should_exit = True


if __name__ == "__main__":
    main()
