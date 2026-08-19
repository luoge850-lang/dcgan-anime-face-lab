"""
Step 1C — Manual ConvTranspose + BatchNorm folding for Exp11.

The previous ORT automatic optimization preserved the output but did not
reduce the graph or improve CPU latency. This script is a separate,
auditable attempt: it folds inference-time BatchNorm parameters into the
preceding ConvTranspose layers, then validates numerical equivalence and
benchmarks raw vs manually folded ONNX at several batch sizes.

This script does not retrain the GAN and does not change the checkpoint.
The raw ONNX remains the immutable reference.
"""

import argparse
import csv
import copy
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
    parser = argparse.ArgumentParser(description="Exp11 manual BN folding audit")
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
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01C_BN_Fold",
        ),
    )
    parser.add_argument(
        "--provider",
        choices=["CPUExecutionProvider", "CUDAExecutionProvider"],
        default=os.getenv("ORT_PROVIDER", "CPUExecutionProvider"),
    )
    parser.add_argument("--batches", default=os.getenv("BENCHMARK_BATCHES", "1,4,8,16,32"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args, _ = parser.parse_known_args(argv)
    args.batches = parse_batches(args.batches)
    if args.warmup < 0 or args.iters < 2:
        raise ValueError("warmup must be nonnegative and iters must be at least 2")
    return args


def parse_batches(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        values = [int(x.strip()) for x in str(value).split(",") if x.strip()]
    values = sorted(set(int(x) for x in values))
    if not values or any(x < 1 for x in values):
        raise ValueError("batches must contain positive integers, for example 1,4,8,16")
    return values


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def build_task1_manifest(output_dir, manual_metadata):
    """Consolidate the three Task 1 stage records into one root manifest."""
    root = output_dir.parent
    stage_files = {
        "01A_export": root / "01A_Export" / "onnx_check.json",
        "01B_ort_optimize": root / "01B_ORT_Optimize" / "fusion_check.json",
        "01C_bn_fold": output_dir / "manual_bn_fusion_check.json",
    }
    stages = {}
    for name, path in stage_files.items():
        if path.exists():
            try:
                stages[name] = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                stages[name] = {"status": "unreadable", "error": repr(exc)}
        else:
            stages[name] = {"status": "missing", "file": path.name}
    required_present = all(item.get("status") not in ("missing", "unreadable") for item in stages.values())
    fusion_evidence_present = all(
        (root / relative).exists()
        for relative in (
            "01B_ORT_Optimize/operator_latency_raw.csv",
            "01B_ORT_Optimize/operator_latency_fused.csv",
            "01C_BN_Fold/operator_latency_raw.csv",
            "01C_BN_Fold/operator_latency_manual_bn_fused.csv",
        )
    )
    manifest = {
        "task": "Task1_ONNX_Fusion",
        "status": "complete" if required_present and fusion_evidence_present else "incomplete",
        "stage_files": {name: str(path.relative_to(root)) for name, path in stage_files.items()},
        "stages": stages,
        "main_model": "01A_Export/generator.onnx",
        "raw_model_alias": "01A_Export/generator_fp32_raw.onnx",
        "fusion_evidence": [
            "01B_ORT_Optimize/operator_latency_raw.csv",
            "01B_ORT_Optimize/operator_latency_fused.csv",
            "01C_BN_Fold/operator_latency_raw.csv",
            "01C_BN_Fold/operator_latency_manual_bn_fused.csv",
        ],
        "manual_bn_decision": manual_metadata.get("optimization_decision"),
        "note": "A not_passed speed decision means the fusion was measured but did not meet the 5% acceleration gate; it is not an execution failure.",
    }
    save_json(root / "task1_manifest.json", manifest)


def write_rows(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def ensure_dependencies():
    missing = []
    for module_name, package_name in (("onnx", "onnx"), ("onnxruntime", "onnxruntime")):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        print(f"[setup] installing: {' '.join(sorted(set(missing)))}")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))
        ])


def locate_raw_onnx(explicit):
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
        "generator_fp32_raw.onnx was not found. Attach the 1A ONNX output or set RAW_ONNX_PATH."
    )


def initializer_map(model):
    return {item.name: item for item in model.graph.initializer}


def attr_float(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return float(default)


def fold_batch_norm(model):
    """Fold only the simple ConvTranspose -> BatchNormalization patterns.

    Exp11 uses four such patterns. If a pattern is not safe or not supported,
    it is left untouched and recorded as skipped rather than guessed.
    """
    import onnx
    from onnx import numpy_helper

    result = copy.deepcopy(model)
    initializers = initializer_map(result)
    nodes = list(result.graph.node)
    consumers = defaultdict(list)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(index)

    folded = []
    remove_indices = set()
    for index, conv in enumerate(nodes):
        if conv.op_type != "ConvTranspose" or len(conv.output) != 1:
            continue
        conv_output = conv.output[0]
        next_indices = consumers.get(conv_output, [])
        if len(next_indices) != 1:
            continue
        bn_index = next_indices[0]
        bn = nodes[bn_index]
        if bn.op_type != "BatchNormalization" or len(bn.input) < 5:
            continue
        if bn.input[0] != conv_output:
            continue
        if any(name not in initializers for name in bn.input[1:5]):
            continue

        weight_name = conv.input[1]
        if weight_name not in initializers:
            continue
        weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
        if weight.ndim != 4:
            continue
        groups = 1
        for attr in conv.attribute:
            if attr.name == "group":
                groups = int(attr.i)
        if groups != 1:
            continue

        scale = numpy_helper.to_array(initializers[bn.input[1]]).astype(np.float32)
        bias_bn = numpy_helper.to_array(initializers[bn.input[2]]).astype(np.float32)
        mean = numpy_helper.to_array(initializers[bn.input[3]]).astype(np.float32)
        variance = numpy_helper.to_array(initializers[bn.input[4]]).astype(np.float32)
        out_channels = weight.shape[1]
        if any(arr.size != out_channels for arr in (scale, bias_bn, mean, variance)):
            continue

        epsilon = attr_float(bn, "epsilon", 1e-5)
        inv_std = scale / np.sqrt(variance + epsilon)
        conv_bias_name = conv.input[2] if len(conv.input) >= 3 and conv.input[2] else None
        if conv_bias_name and conv_bias_name in initializers:
            conv_bias = numpy_helper.to_array(initializers[conv_bias_name]).astype(np.float32)
        else:
            conv_bias = np.zeros(out_channels, dtype=np.float32)
        if conv_bias.size != out_channels:
            continue

        # ONNX ConvTranspose weights are [C_in, C_out, kH, kW] for group=1.
        folded_weight = weight * inv_std.reshape(1, -1, 1, 1)
        folded_bias = bias_bn + inv_std * (conv_bias - mean)
        initializers[weight_name] = numpy_helper.from_array(folded_weight, weight_name)

        if conv_bias_name:
            initializers[conv_bias_name] = numpy_helper.from_array(
                folded_bias, conv_bias_name
            )
        else:
            conv_bias_name = f"{conv.name or weight_name}_bn_fold_bias"
            initializers[conv_bias_name] = numpy_helper.from_array(
                folded_bias, conv_bias_name
            )
            if len(conv.input) >= 3:
                # ONNX permits an empty optional bias input. Replace it rather
                # than appending a fourth input, which would make the graph invalid.
                conv.input[2] = conv_bias_name
            else:
                conv.input.append(conv_bias_name)

        # Keep the BN output tensor name so all downstream consumers remain valid.
        conv.output[0] = bn.output[0]
        remove_indices.add(bn_index)
        folded.append({
            "conv_node": conv.name or f"ConvTranspose_{index}",
            "bn_node": bn.name or f"BatchNormalization_{bn_index}",
            "weight": weight_name,
            "bias": conv_bias_name,
            "channels": int(out_channels),
            "epsilon": epsilon,
        })

    if not folded:
        raise RuntimeError(
            "No safe ConvTranspose -> BatchNormalization pairs were found; refusing to alter the graph."
        )
    result.graph.ClearField("node")
    result.graph.node.extend([node for index, node in enumerate(nodes) if index not in remove_indices])
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(initializers.values())
    onnx.checker.check_model(result)
    return result, folded


def node_inventory(path):
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    counts = Counter(node.op_type for node in model.graph.node)
    return model, counts, sorted(counts.items())


def make_session(ort, path, provider, profile=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = bool(profile)
    return ort.InferenceSession(str(path), sess_options=options, providers=[provider])


def sync_device(provider):
    if provider != "CUDAExecutionProvider":
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def run_once(session, input_name, noise, provider):
    output = session.run(None, {input_name: noise})[0]
    sync_device(provider)
    return output


def benchmark(session, input_name, noise, warmup, iters, provider):
    for _ in range(warmup):
        run_once(session, input_name, noise, provider)
    values = []
    for _ in range(iters):
        start = time.perf_counter()
        run_once(session, input_name, noise, provider)
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


def save_sample_grid(array, path):
    images = np.clip((array + 1.0) / 2.0, 0, 1)
    n = images.shape[0]
    rows = []
    for start in range(0, n, 8):
        row = images[start:start + 8]
        if row.shape[0] < 8:
            row = np.concatenate(
                [row, np.zeros((8 - row.shape[0], *row.shape[1:]))], axis=0
            )
        rows.append(np.concatenate([x.transpose(1, 2, 0) for x in row], axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray((grid * 255).astype(np.uint8)).save(path)


def finish_profile(session, profile_path):
    source = Path(session.end_profiling())
    target = Path(profile_path)
    if source.exists() and source.resolve() != target.resolve():
        shutil.copy2(source, target)
    if not target.exists():
        raise FileNotFoundError(f"ORT profile was not produced: {source}")


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
    write_rows(csv_path, ["operator", "calls", "total_ms", "mean_ms", "p95_ms"], rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    raw_path = locate_raw_onnx(args.raw_onnx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dependencies()
    import onnx
    import onnxruntime as ort

    raw_model, raw_counts, raw_rows = node_inventory(raw_path)
    folded_model, folded_pairs = fold_batch_norm(raw_model)
    folded_path = output_dir / "generator_fp32_manual_bn_fused.onnx"
    onnx.save(folded_model, str(folded_path))
    onnx.checker.check_model(onnx.load(str(folded_path)))

    folded_model_check, folded_counts, folded_rows = node_inventory(folded_path)
    write_rows(output_dir / "operator_inventory_raw.csv", ["op_type", "count"], raw_rows)
    write_rows(output_dir / "operator_inventory_manual_fused.csv", ["op_type", "count"], folded_rows)
    raw_session = make_session(ort, raw_path, args.provider)
    folded_session = make_session(ort, folded_path, args.provider)
    raw_name = raw_session.get_inputs()[0].name
    folded_name = folded_session.get_inputs()[0].name
    rng = np.random.default_rng(42)
    benchmark_rows = []
    diff_rows = []
    all_results = {}
    for batch in args.batches:
        noise = rng.standard_normal((batch, NOISE_DIM, 1, 1), dtype=np.float32)
        raw_output = run_once(raw_session, raw_name, noise, args.provider)
        folded_output = run_once(folded_session, folded_name, noise, args.provider)
        if raw_output.shape != (batch, 3, IMAGE_SIZE, IMAGE_SIZE):
            raise RuntimeError(f"Unexpected raw output shape for batch {batch}: {raw_output.shape}")
        diff_rows.append([
            batch,
            float(np.max(np.abs(raw_output - folded_output))),
            float(np.mean(np.abs(raw_output - folded_output))),
        ])
        raw_timing = benchmark(raw_session, raw_name, noise, args.warmup, args.iters, args.provider)
        folded_timing = benchmark(folded_session, folded_name, noise, args.warmup, args.iters, args.provider)
        for graph, timing in (("raw", raw_timing), ("manual_bn_fused", folded_timing)):
            benchmark_rows.append([
                graph, args.provider, batch, args.warmup, args.iters,
                *timing.values(),
            ])
        all_results[str(batch)] = {"raw": raw_timing, "manual_bn_fused": folded_timing}
        if batch == args.batches[0]:
            save_sample_grid(raw_output, output_dir / "sample_raw.png")
            save_sample_grid(folded_output, output_dir / "sample_manual_bn_fused.png")

    write_rows(
        output_dir / "manual_bn_latency_summary.csv",
        ["graph", "provider", "batch", "warmup", "iterations", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_images_per_s"],
        benchmark_rows,
    )
    write_rows(
        output_dir / "numerical_equivalence.csv",
        ["batch", "max_abs_difference", "mean_abs_difference"],
        diff_rows,
    )

    profile_batch = args.batches[0]
    profile_noise = np.random.default_rng(4242).standard_normal(
        (profile_batch, NOISE_DIM, 1, 1), dtype=np.float32
    )
    raw_profile_session = make_session(ort, raw_path, args.provider, profile=True)
    folded_profile_session = make_session(ort, folded_path, args.provider, profile=True)
    raw_profile_name = raw_profile_session.get_inputs()[0].name
    folded_profile_name = folded_profile_session.get_inputs()[0].name
    for _ in range(args.warmup):
        run_once(raw_profile_session, raw_profile_name, profile_noise, args.provider)
        run_once(folded_profile_session, folded_profile_name, profile_noise, args.provider)
    for _ in range(args.iters):
        run_once(raw_profile_session, raw_profile_name, profile_noise, args.provider)
        run_once(folded_profile_session, folded_profile_name, profile_noise, args.provider)
    finish_profile(raw_profile_session, output_dir / "ort_profile_raw.json")
    finish_profile(folded_profile_session, output_dir / "ort_profile_manual_bn_fused.json")
    raw_ops = aggregate_profile(output_dir / "ort_profile_raw.json", output_dir / "operator_latency_raw.csv")
    fused_ops = aggregate_profile(
        output_dir / "ort_profile_manual_bn_fused.json",
        output_dir / "operator_latency_manual_bn_fused.csv",
    )

    max_diff = max(row[1] for row in diff_rows)
    mean_diff = max(row[2] for row in diff_rows)
    improvements = []
    for batch in args.batches:
        raw_mean = all_results[str(batch)]["raw"]["mean_ms"]
        fused_mean = all_results[str(batch)]["manual_bn_fused"]["mean_ms"]
        improvements.append({
            "batch": batch,
            "mean_speedup_ratio": float(raw_mean / fused_mean),
            "mean_improvement_percent": float((raw_mean / fused_mean - 1.0) * 100.0),
        })
    improved_batches = sum(item["mean_improvement_percent"] >= 5.0 for item in improvements)
    optimization_pass = bool(max_diff <= 1e-5 and improved_batches >= 2)
    metadata = {
        "raw_onnx": str(raw_path),
        "manual_fused_onnx": str(folded_path),
        "provider": args.provider,
        "batches": args.batches,
        "warmup": args.warmup,
        "iterations": args.iters,
        "raw_node_count": len(raw_model.graph.node),
        "manual_fused_node_count": len(folded_model_check.graph.node),
        "raw_operator_counts": dict(sorted(raw_counts.items())),
        "manual_fused_operator_counts": dict(sorted(folded_counts.items())),
        "folded_pairs": folded_pairs,
        "max_abs_output_difference": max_diff,
        "max_mean_abs_output_difference": mean_diff,
        "top_raw_operators": raw_ops[:3],
        "top_manual_fused_operators": fused_ops[:3],
        "improvements": improvements,
        "pass_criteria": {
            "max_abs_difference_le_1e-5": bool(max_diff <= 1e-5),
            "at_least_two_batches_improve_mean_latency_by_5_percent": bool(improved_batches >= 2),
        },
        "optimization_decision": "pass" if optimization_pass else "not_passed",
    }
    save_json(output_dir / "manual_bn_fusion_check.json", metadata)
    build_task1_manifest(output_dir, metadata)
    archive_path = output_dir.parent / "Task1_01C_BN_Fold.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=f"01C_BN_Fold/{file_path.name}")

    print(f"[fold] pairs={len(folded_pairs)} raw_nodes={len(raw_model.graph.node)} manual_nodes={len(folded_model_check.graph.node)}")
    print(f"[equivalence] max_abs={max_diff:.8g} mean_abs={mean_diff:.8g}")
    print(f"[decision] {'PASS' if optimization_pass else 'NOT_PASSED'}")
    print(f"[zip] {archive_path}")
    print(f"[done] outputs: {output_dir}")


if __name__ == "__main__":
    main()
