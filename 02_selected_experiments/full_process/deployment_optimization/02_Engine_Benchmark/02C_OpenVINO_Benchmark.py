"""
Task 2C — OpenVINO CPU FP32/FP16 benchmark.

The input is the validated 01A raw ONNX. OpenVINO FP16 support is attempted
and recorded as unavailable when the installed CPU plugin cannot compile the
converted model; the script never substitutes another engine silently.
"""

import argparse
import csv
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
    p = argparse.ArgumentParser(description="Task 2C OpenVINO benchmark")
    p.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark/02C_OpenVINO"))
    p.add_argument("--batches", default="1,4,8,16,32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--skip-fp16", action="store_true")
    args, _ = p.parse_known_args(argv)
    args.batches = sorted({int(x) for x in args.batches.split(",") if x.strip()})
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
    converted = float16.convert_float_to_float16(onnx.load(str(raw_path)), keep_io_types=True)
    onnx.save(converted, str(fp16_path))
    return fp16_path


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def memory_mb():
    try:
        import psutil
        return float(psutil.Process().memory_info().rss / (1024 ** 2))
    except Exception:
        return float("nan")


def _duration_ms(value):
    """Convert OpenVINO profiling durations across API versions to ms."""
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds() * 1000.0)
    try:
        # Newer bindings expose nanoseconds as an integer-like duration.
        return float(value) / 1_000_000.0
    except Exception:
        return float("nan")


def collect_profiling_info(request, precision, batch):
    rows = []
    try:
        infos = request.profiling_info
    except Exception:
        return rows
    for info in infos:
        rows.append([
            precision,
            batch,
            getattr(info, "node_name", ""),
            getattr(info, "node_type", ""),
            _duration_ms(getattr(info, "real_time", 0)),
            _duration_ms(getattr(info, "cpu_time", 0)),
            str(getattr(info, "status", "")),
        ])
    return rows


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    install_if_missing([("openvino", "openvino"), ("psutil", "psutil")])
    import openvino as ov

    raw_path = locate_raw(args.raw_onnx)
    fp16_path = None if args.skip_fp16 else ensure_fp16_model(raw_path, output_dir)
    core = ov.Core()
    rows = []
    operator_rows = []
    for precision, model_path in (("FP32", raw_path), ("FP16", fp16_path)):
        if model_path is None:
            continue
        try:
            model = core.read_model(str(model_path))
            try:
                compiled = core.compile_model(model, "CPU", {"PERF_COUNT": True})
                profiling_enabled = True
            except Exception:
                compiled = core.compile_model(model, "CPU")
                profiling_enabled = False
        except Exception as exc:
            print(f"[skip] OpenVINO {precision}: {exc}")
            continue
        request = compiled.create_infer_request()
        input_port = compiled.inputs[0]
        input_name = input_port.any_name
        for batch in args.batches:
            noise = np.random.default_rng(42 + batch).standard_normal(
                (batch, NOISE_DIM, 1, 1), dtype=np.float32
            )
            output = request.infer({input_name: noise})[compiled.outputs[0]]
            expected = (batch, 3, IMAGE_SIZE, IMAGE_SIZE)
            if tuple(output.shape) != expected:
                raise RuntimeError(f"unexpected output shape {output.shape}; expected {expected}")
            for _ in range(args.warmup):
                request.infer({input_name: noise})
            values = []
            for _ in range(args.iters):
                start = time.perf_counter()
                request.infer({input_name: noise})
                values.append((time.perf_counter() - start) * 1000.0)
            if profiling_enabled:
                operator_rows.extend(collect_profiling_info(request, precision, batch))
            values = np.asarray(values, dtype=np.float64)
            mean_ms = float(values.mean())
            row = [precision, batch, args.warmup, args.iters, mean_ms, float(np.percentile(values, 50)), float(np.percentile(values, 95)), float(values.min()), float(values.max()), float(batch / (mean_ms / 1000.0)), memory_mb(), "ok"]
            rows.append(row)
            print(f"[ok] OpenVINO {precision} batch={batch} mean_ms={mean_ms:.4f}")
    write_csv(output_dir / "openvino_benchmark.csv", ["precision", "batch", "warmup", "iterations", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_images_per_s", "host_rss_mb", "status"], rows)
    write_csv(output_dir / "openvino_operator_profile.csv", ["precision", "batch", "node_name", "node_type", "real_time_ms", "cpu_time_ms", "status"], operator_rows)
    archive = output_dir.parent / "Task2_02C_OpenVINO.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                z.write(file_path, arcname=f"02C_OpenVINO/{file_path.relative_to(output_dir).as_posix()}")
    print("[zip]", archive)


if __name__ == "__main__":
    main()
