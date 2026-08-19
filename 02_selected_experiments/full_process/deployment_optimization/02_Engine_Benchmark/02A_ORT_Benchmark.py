"""
Task 2A — ONNX Runtime FP32/FP16 benchmark.

Use the validated 01A generator_fp32_raw.onnx as the common engine baseline.
The script tests CPU and, when available, CUDAExecutionProvider at batch
sizes 1/4/8/16. It records end-to-end latency, throughput, host memory and
an ORT profile for each successful configuration.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


NOISE_DIM = 128
IMAGE_SIZE = 64


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 2A ONNX Runtime benchmark")
    p.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark/02A_ORT"))
    p.add_argument("--providers", default=os.getenv("ORT_PROVIDERS", "CPUExecutionProvider"), help="comma-separated ORT providers")
    p.add_argument("--batches", default="1,4,8,16,32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--skip-fp16", action="store_true")
    args, _ = p.parse_known_args(argv)
    args.batches = sorted({int(x) for x in args.batches.split(",") if x.strip()})
    args.providers = [x.strip() for x in args.providers.split(",") if x.strip()]
    if not args.batches or min(args.batches) < 1:
        raise ValueError("batches must contain positive integers")
    return args


def install_if_missing(modules):
    missing = []
    for module_name, package in modules:
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package)
    if missing:
        print("[setup] installing:", " ".join(sorted(set(missing))))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))])


def locate_raw(path_text):
    if path_text and Path(path_text).exists():
        return Path(path_text)
    found = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            found.extend(root.rglob("generator_fp32_raw.onnx"))
    unique = sorted({p.resolve() for p in found})
    if len(unique) == 1:
        print("[raw-onnx] auto-detected", unique[0])
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError("Multiple raw ONNX files found; set RAW_ONNX_PATH explicitly:\n" + "\n".join(map(str, unique)))
    raise FileNotFoundError("generator_fp32_raw.onnx not found. Attach the 01A output Dataset.")


def ensure_fp16_model(raw_path, output_dir):
    fp16_path = Path(output_dir) / "generator_fp16_internal.onnx"
    if fp16_path.exists():
        return fp16_path
    install_if_missing([("onnx", "onnx"), ("onnxconverter_common", "onnxconverter-common")])
    import onnx
    from onnxconverter_common import float16
    model = onnx.load(str(raw_path))
    converted = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(converted, str(fp16_path))
    return fp16_path


def sync_cuda(provider):
    if provider != "CUDAExecutionProvider":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def memory_mb(provider):
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        rss = float("nan")
    if provider != "CUDAExecutionProvider":
        return rss, float("nan")
    gpu = float("nan")
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return rss, gpu


def benchmark(session, input_name, noise, provider, warmup, iters):
    for _ in range(warmup):
        session.run(None, {input_name: noise})
    sync_cuda(provider)
    values = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, {input_name: noise})
        sync_cuda(provider)
        values.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(values, dtype=np.float64)
    mean_ms = float(values.mean())
    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_images_per_s": float(noise.shape[0] / (mean_ms / 1000.0)),
    }


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def save_profile(session, path):
    profile_path = session.end_profiling()
    source = Path(profile_path)
    if source.exists() and source.resolve() != Path(path).resolve():
        source.replace(path)
    return Path(path)


def aggregate_profile(profile_path, precision, provider, batch):
    """Convert one ORT Chrome-style profile into auditable operator rows."""
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    grouped = {}
    for event in events:
        if event.get("cat") != "Node" or "dur" not in event:
            continue
        args = event.get("args", {})
        name = str(args.get("op_name") or event.get("name", "unknown"))
        grouped.setdefault(name, []).append(float(event["dur"]) / 1000.0)
    rows = []
    for operator, durations in grouped.items():
        values = np.asarray(durations, dtype=np.float64)
        rows.append({
            "precision": precision,
            "provider": provider,
            "batch": batch,
            "operator": operator,
            "calls": len(values),
            "total_ms": float(values.sum()),
            "mean_ms": float(values.mean()),
            "p95_ms": float(np.percentile(values, 95)),
            "profile": Path(profile_path).as_posix(),
        })
    return rows


def write_dict_csv(path, rows):
    fields = ["precision", "provider", "batch", "operator", "calls", "total_ms", "mean_ms", "p95_ms", "profile"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_if_missing([("onnxruntime", "onnxruntime"), ("psutil", "psutil")])
    import onnxruntime as ort

    raw_path = locate_raw(args.raw_onnx)
    fp16_path = None if args.skip_fp16 else ensure_fp16_model(raw_path, output_dir)
    providers = []
    available = ort.get_available_providers()
    for provider in args.providers:
        if provider in available:
            providers.append(provider)
        else:
            print(f"[skip] provider unavailable: {provider}; available={available}")
    if not providers:
        raise RuntimeError(f"None of the requested providers are available: {args.providers}")

    rows = []
    operator_rows = []
    profile_dir = output_dir / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    for precision, model_path in (("FP32", raw_path), ("FP16", fp16_path)):
        if model_path is None:
            continue
        for provider in providers:
            for batch in args.batches:
                try:
                    options = ort.SessionOptions()
                    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    options.enable_profiling = True
                    session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider])
                    input_name = session.get_inputs()[0].name
                    # The same precision/batch seed is reused for every
                    # provider, so provider comparisons use identical z.
                    # Reuse identical latent noise for FP32/FP16 and every
                    # requested provider at the same batch size.
                    seed = 42 + batch
                    noise = np.random.default_rng(seed).standard_normal((batch, NOISE_DIM, 1, 1), dtype=np.float32)
                    output = session.run(None, {input_name: noise})[0]
                    expected = (batch, 3, IMAGE_SIZE, IMAGE_SIZE)
                    if tuple(output.shape) != expected:
                        raise RuntimeError(f"unexpected output shape {output.shape}; expected {expected}")
                    timing = benchmark(session, input_name, noise, provider, args.warmup, args.iters)
                    host_mb, gpu_mb = memory_mb(provider)
                    profile_name = f"ort_profile_{provider.replace('ExecutionProvider','').lower()}_{precision.lower()}_b{batch}.json"
                    save_profile(session, profile_dir / profile_name)
                    operator_rows.extend(aggregate_profile(profile_dir / profile_name, precision, provider, batch))
                    row = [precision, provider, batch, args.warmup, args.iters, *timing.values(), host_mb, gpu_mb, "ok"]
                    rows.append(row)
                    print("[ok]", precision, provider, "batch", batch, timing)
                except Exception as exc:
                    print("[skip]", precision, provider, "batch", batch, repr(exc))
                    rows.append([precision, provider, batch, args.warmup, args.iters, *([""] * 8), "failed"])

    write_csv(output_dir / "ort_benchmark.csv", ["precision", "provider", "batch", "warmup", "iterations", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_images_per_s", "host_rss_mb", "gpu_allocated_mb", "status"], rows)
    write_dict_csv(output_dir / "ort_operator_summary.csv", operator_rows)
    archive = output_dir.parent / "Task2_02A_ORT.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                z.write(file_path, arcname=f"02A_ORT/{file_path.relative_to(output_dir).as_posix()}")
    print("[zip]", archive)


if __name__ == "__main__":
    main()
