"""
Step 1B — ONNX Runtime execution, graph optimization and fusion comparison.

Inputs
------
generator_fp32_raw.onnx produced by 01_Export_Exp11_ONNX.py.

Outputs
-------
- generator_fp32_fused.onnx: ORT graph-optimized model;
- fusion_check.json: checker result, node counts and numerical difference;
- fusion_latency_summary.csv: raw vs optimized end-to-end latency;
- operator_latency_raw.csv / operator_latency_fused.csv: ORT node profiling;
- ort_sample_raw.png / ort_sample_fused.png: visual sanity checks.

This script is intentionally limited to ONNX Runtime. TensorRT, OpenVINO,
dynamic batch benchmarking and INT8 are subsequent steps.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = 64
NOISE_DIM = 128


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Exp11 ONNX Runtime fusion test")
    parser.add_argument(
        "--raw-onnx",
        default=os.getenv(
            "RAW_ONNX_PATH",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01A_Export/generator_fp32_raw.onnx",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "DEPLOY_OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01B_ORT_Optimize",
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        default=os.getenv("ORT_PROVIDER", "CPUExecutionProvider"),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args, _ = parser.parse_known_args(argv)
    if args.batch_size < 1 or args.warmup < 0 or args.iters < 2:
        raise ValueError("batch-size must be positive, warmup nonnegative and iters >= 2")
    return args


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_rows(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def ensure_dependencies():
    """Install the two deployment packages only when the runtime lacks them."""
    missing = []
    for module_name, package_name in (("onnx", "onnx"), ("onnxruntime", "onnxruntime")):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        print(f"[setup] installing: {' '.join(missing)}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))
        ])


def node_inventory(path):
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    counts = Counter(node.op_type for node in model.graph.node)
    rows = sorted(counts.items())
    return model, counts, rows


def make_session(ort, path, provider, optimization, optimized_path=None, profile=False):
    options = ort.SessionOptions()
    if optimization == "disable":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    elif optimization == "all":
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimized_path is not None:
            options.optimized_model_filepath = str(optimized_path)
    else:
        raise ValueError(f"Unknown optimization mode: {optimization}")
    options.enable_profiling = bool(profile)
    try:
        return ort.InferenceSession(
            str(path), sess_options=options, providers=[provider]
        )
    except Exception as exc:
        if provider != "CPUExecutionProvider":
            raise RuntimeError(
                f"Could not create ORT session with {provider}. "
                "Use CPUExecutionProvider for the first test or install the CUDA provider."
            ) from exc
        raise


def get_input_name(session):
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"Expected one input, got {len(inputs)}")
    return inputs[0].name


def run_once(session, input_name, noise):
    return session.run(None, {input_name: noise})[0]


def benchmark(session, input_name, noise, warmup, iters, provider):
    for _ in range(warmup):
        run_once(session, input_name, noise)
    if provider == "CUDAExecutionProvider":
        try:
            import cupy
            cupy.cuda.Stream.null.synchronize()
        except ImportError:
            pass
    values = []
    for _ in range(iters):
        start = time.perf_counter()
        run_once(session, input_name, noise)
        if provider == "CUDAExecutionProvider":
            try:
                import cupy
                cupy.cuda.Stream.null.synchronize()
            except ImportError:
                pass
        values.append((time.perf_counter() - start) * 1000.0)
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "throughput_images_per_s": float(noise.shape[0] / (values.mean() / 1000.0)),
    }


def finish_profile(session, profile_path):
    profile_file = session.end_profiling()
    source = Path(profile_file)
    target = Path(profile_path)
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    if not target.exists():
        raise FileNotFoundError(f"ORT profile was not produced: {profile_file}")
    return target


def aggregate_profile(profile_path, csv_path):
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for event in events:
        if event.get("cat") != "Node" or "dur" not in event:
            continue
        args = event.get("args", {})
        name = args.get("op_name") or event.get("name", "unknown")
        grouped[str(name)].append(float(event["dur"]))
    rows = []
    for name, durations in grouped.items():
        values = np.asarray(durations, dtype=np.float64)
        rows.append([
            name,
            len(values),
            float(values.sum() / 1000.0),
            float(values.mean() / 1000.0),
            float(np.percentile(values, 95) / 1000.0),
        ])
    rows.sort(key=lambda row: row[2], reverse=True)
    write_rows(
        csv_path,
        ["operator", "calls", "total_ms", "mean_ms", "p95_ms"],
        rows,
    )
    return rows


def save_sample_grid(array, path):
    images = (array + 1.0) / 2.0
    images = np.clip(images, 0, 1)
    n = images.shape[0]
    rows = []
    for start in range(0, n, 8):
        row = images[start:start + 8]
        if row.shape[0] < 8:
            row = np.concatenate([row, np.zeros((8 - row.shape[0], *row.shape[1:]))], axis=0)
        rows.append(np.concatenate([x.transpose(1, 2, 0) for x in row], axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def locate_raw_onnx(explicit):
    if explicit:
        path = Path(explicit)
        if path.exists():
            return path
    candidates = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input"), Path.cwd()):
        if root.exists():
            candidates.extend(root.rglob("generator_fp32_raw.onnx"))
    unique = sorted({p.resolve() for p in candidates})
    if len(unique) == 1:
        print(f"[raw-onnx] auto-detected {unique[0]}")
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(
            "Multiple generator_fp32_raw.onnx files found. Pass --raw-onnx explicitly:\n"
            + "\n".join(str(p) for p in unique)
        )
    raise FileNotFoundError(
        "generator_fp32_raw.onnx was not found. Re-run 01_Export_Exp11_ONNX.py "
        "in this active Kaggle session, or attach the ONNX output as a Dataset."
    )


def main(argv=None):
    args = parse_args(argv)
    raw_path = locate_raw_onnx(args.raw_onnx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ensure_dependencies()
    import onnx
    import onnxruntime as ort

    raw_model, raw_counts, raw_rows = node_inventory(raw_path)
    write_rows(output_dir / "operator_inventory_raw_ort.csv", ["op_type", "count"], raw_rows)
    fused_path = output_dir / "generator_fp32_fused.onnx"

    # Build an optimized graph once. The optimized model is then loaded with
    # graph optimization disabled for a fair raw-vs-fused comparison.
    optimized_session = make_session(
        ort, raw_path, args.provider, "all", optimized_path=fused_path, profile=False
    )
    if not fused_path.exists():
        raise RuntimeError("ONNX Runtime did not write generator_fp32_fused.onnx")
    fused_model, fused_counts, fused_rows = node_inventory(fused_path)
    write_rows(output_dir / "operator_inventory_fused_ort.csv", ["op_type", "count"], fused_rows)

    raw_session = make_session(ort, raw_path, args.provider, "disable", profile=False)
    fused_runtime_session = make_session(ort, fused_path, args.provider, "disable", profile=False)
    raw_name = get_input_name(raw_session)
    fused_name = get_input_name(fused_runtime_session)
    rng = np.random.default_rng(42)
    noise = rng.standard_normal((args.batch_size, NOISE_DIM, 1, 1), dtype=np.float32)
    raw_output = run_once(raw_session, raw_name, noise)
    fused_output = run_once(fused_runtime_session, fused_name, noise)
    if raw_output.shape != (args.batch_size, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Unexpected raw output shape: {raw_output.shape}")
    if fused_output.shape != raw_output.shape:
        raise RuntimeError(f"Fused output shape differs: {fused_output.shape}")
    save_sample_grid(raw_output, output_dir / "ort_sample_raw.png")
    save_sample_grid(fused_output, output_dir / "ort_sample_fused.png")

    raw_timing = benchmark(
        raw_session, raw_name, noise, args.warmup, args.iters, args.provider
    )
    fused_timing = benchmark(
        fused_runtime_session, fused_name, noise, args.warmup, args.iters, args.provider
    )
    write_rows(
        output_dir / "fusion_latency_summary.csv",
        ["graph", "provider", "batch", "warmup", "iterations", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_images_per_s"],
        [
            ["raw", args.provider, args.batch_size, args.warmup, args.iters, *raw_timing.values()],
            ["fused", args.provider, args.batch_size, args.warmup, args.iters, *fused_timing.values()],
        ],
    )

    # Profile one warmup + measured pass for each graph. ORT's Node events are
    # the requested per-operator timing evidence for this step.
    raw_profile_session = make_session(ort, raw_path, args.provider, "disable", profile=True)
    fused_profile_session = make_session(ort, fused_path, args.provider, "disable", profile=True)
    raw_profile_name = get_input_name(raw_profile_session)
    fused_profile_name = get_input_name(fused_profile_session)
    for _ in range(args.warmup):
        run_once(raw_profile_session, raw_profile_name, noise)
        run_once(fused_profile_session, fused_profile_name, noise)
    for _ in range(args.iters):
        run_once(raw_profile_session, raw_profile_name, noise)
        run_once(fused_profile_session, fused_profile_name, noise)
    raw_profile = finish_profile(raw_profile_session, output_dir / "ort_profile_raw.json")
    fused_profile = finish_profile(fused_profile_session, output_dir / "ort_profile_fused.json")
    raw_ops = aggregate_profile(raw_profile, output_dir / "operator_latency_raw.csv")
    fused_ops = aggregate_profile(fused_profile, output_dir / "operator_latency_fused.csv")

    metadata = {
        "raw_onnx": str(raw_path),
        "fused_onnx": str(fused_path),
        "provider": args.provider,
        "batch_size": args.batch_size,
        "raw_node_count": len(raw_model.graph.node),
        "fused_node_count": len(fused_model.graph.node),
        "raw_operator_counts": dict(sorted(raw_counts.items())),
        "fused_operator_counts": dict(sorted(fused_counts.items())),
        "raw_output_shape": list(raw_output.shape),
        "fused_output_shape": list(fused_output.shape),
        "max_abs_output_difference": float(np.max(np.abs(raw_output - fused_output))),
        "mean_abs_output_difference": float(np.mean(np.abs(raw_output - fused_output))),
        "raw_top_operators": raw_ops[:3],
        "fused_top_operators": fused_ops[:3],
        "raw_timing": raw_timing,
        "fused_timing": fused_timing,
        "status": "passed",
    }
    save_json(output_dir / "fusion_check.json", metadata)
    archive_path = output_dir.parent / "Task1_01B_ORT_Optimize.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=f"01B_ORT_Optimize/{file_path.name}")
    print(f"[raw] nodes={len(raw_model.graph.node)} timing={raw_timing}")
    print(f"[fused] nodes={len(fused_model.graph.node)} timing={fused_timing}")
    print(f"[diff] max_abs={metadata['max_abs_output_difference']:.8g}")
    print(f"[zip] {archive_path}")
    print(f"[done] fusion outputs: {output_dir}")


if __name__ == "__main__":
    main()
