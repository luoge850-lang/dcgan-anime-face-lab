"""
Step 1D — exact ConvTranspose-to-Conv+PixelShuffle reparameterization.

The Exp11 Generator has four stride-2 ConvTranspose2d layers.  For the
specific configuration used here (kernel=4, stride=2, padding=1, group=1),
each layer can be represented exactly as:

    ConvTranspose2d(Cin, Cout, k=4, s=2, p=1)
        == Conv2d(Cin, 4*Cout, k=3, s=1, p=1) + PixelShuffle(2)

The converted Conv2d kernel contains the phase-rearranged transposed-conv
weights; zero entries are retained explicitly so the transformation is easy
to audit.  BatchNorm and following activations are kept in graph order.  The
script is an experiment, not a claim that the rewrite is always faster: it
exports the graph, checks numerical equivalence, profiles individual nodes,
and applies a speed gate over multiple batch sizes.
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


IMAGE_SIZE = 64
NOISE_DIM = 128


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Task 1D subpixel reparameterization")
    parser.add_argument(
        "--raw-onnx",
        default=os.getenv(
            "RAW_ONNX_PATH",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01A_Export/generator.onnx",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "DEPLOY_OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01D_Subpixel_Reparam",
        ),
    )
    parser.add_argument("--batches", default=os.getenv("BENCHMARK_BATCHES", "1,4,8,16,32"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    args, _ = parser.parse_known_args(argv)
    args.batches = sorted({int(item.strip()) for item in str(args.batches).split(",") if item.strip()})
    if not args.batches or any(batch < 1 for batch in args.batches):
        raise ValueError("batches must contain positive integers")
    if args.warmup < 0 or args.iters < 2:
        raise ValueError("warmup must be nonnegative and iters must be at least 2")
    return args


def ensure_dependencies():
    missing = []
    for module_name, package_name in (("onnx", "onnx"), ("onnxruntime", "onnxruntime")):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        print(f"[setup] installing: {' '.join(sorted(set(missing)))}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))])


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def locate_raw(path_value):
    path = Path(path_value)
    if path.exists():
        return path
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            candidates.extend(root.rglob("generator.onnx"))
            candidates.extend(root.rglob("generator_fp32_raw.onnx"))
    unique = sorted({candidate.resolve() for candidate in candidates})
    if len(unique) == 1:
        print(f"[raw-onnx] auto-detected {unique[0]}")
        return unique[0]
    if not unique:
        raise FileNotFoundError("generator.onnx or generator_fp32_raw.onnx was not found")
    raise RuntimeError("Multiple ONNX files found; pass --raw-onnx explicitly:\n" + "\n".join(map(str, unique)))


def attr_ints(node, name, default):
    import onnx

    for attr in node.attribute:
        if attr.name == name:
            value = onnx.helper.get_attribute_value(attr)
            if isinstance(value, (list, tuple)):
                return list(value)
            return [int(value)]
    return list(default)


def shape_map(model):
    result = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor = value.type.tensor_type
        if not tensor.HasField("shape"):
            continue
        dims = []
        valid = True
        for dim in tensor.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        if len(dims) >= 4:
            result[value.name] = dims
    return result


def initializer_map(model):
    return {item.name: item for item in model.graph.initializer}


def _phase_index(phase, kernel_index):
    """Map a ConvTranspose kernel index to Conv2d k=3 index for one phase."""
    numerator = phase + 1 - kernel_index
    if numerator % 2 != 0:
        return None
    return 1 + numerator // 2


def convert_stride2_convtranspose(model):
    """Rewrite all safe stride-2 ConvTranspose nodes exactly."""
    import onnx
    from onnx import helper, numpy_helper

    result = copy.deepcopy(model)
    inferred = onnx.shape_inference.infer_shapes(result)
    shapes = shape_map(inferred)
    initializers = initializer_map(result)
    nodes = list(result.graph.node)
    rewritten = []
    converted = []

    for index, node in enumerate(nodes):
        if node.op_type != "ConvTranspose":
            rewritten.append(node)
            continue
        strides = attr_ints(node, "strides", [1, 1])
        pads = attr_ints(node, "pads", [0, 0, 0, 0])
        kernels = attr_ints(node, "kernel_shape", [4, 4])
        groups = attr_ints(node, "group", [1])[0]
        input_shape = shapes.get(node.input[0])
        if strides != [2, 2] or kernels != [4, 4] or pads != [1, 1, 1, 1] or groups != 1:
            rewritten.append(node)
            continue
        if input_shape is None or input_shape[2] is None or input_shape[3] is None:
            raise RuntimeError(f"Cannot determine static spatial shape for {node.name or index}")
        weight_name = node.input[1]
        if weight_name not in initializers:
            raise RuntimeError(f"Missing ConvTranspose weight initializer: {weight_name}")
        from onnx import numpy_helper

        old_weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
        if old_weight.ndim != 4:
            raise RuntimeError(f"Unexpected ConvTranspose weight shape: {old_weight.shape}")
        cin, cout, kh, kw = old_weight.shape
        new_weight = np.zeros((cout * 4, cin, 3, 3), dtype=np.float32)
        for out_channel in range(cout):
            for phase_h in range(2):
                for phase_w in range(2):
                    phase_channel = out_channel * 4 + phase_h * 2 + phase_w
                    for old_h in range(4):
                        new_h = _phase_index(phase_h, old_h)
                        if new_h is None or not 0 <= new_h < 3:
                            continue
                        for old_w in range(4):
                            new_w = _phase_index(phase_w, old_w)
                            if new_w is None or not 0 <= new_w < 3:
                                continue
                            new_weight[phase_channel, :, new_h, new_w] = old_weight[:, out_channel, old_h, old_w]

        old_bias = np.zeros(cout, dtype=np.float32)
        if len(node.input) >= 3 and node.input[2] and node.input[2] in initializers:
            old_bias = numpy_helper.to_array(initializers[node.input[2]]).astype(np.float32)
        if old_bias.size != cout:
            raise RuntimeError(f"Unexpected ConvTranspose bias shape: {old_bias.shape}")
        new_bias = np.repeat(old_bias, 4).astype(np.float32)

        base = node.name or f"ConvTranspose_{index}"
        new_weight_name = base + "__subpixel_weight"
        new_bias_name = base + "__subpixel_bias"
        conv_output = base + "__subpixel_conv_output"
        reshape1_output = base + "__subpixel_reshape_output"
        transpose_output = base + "__subpixel_transpose_output"
        shape1_name = base + "__subpixel_shape1"
        shape2_name = base + "__subpixel_shape2"
        initializers[new_weight_name] = numpy_helper.from_array(new_weight, new_weight_name)
        initializers[new_bias_name] = numpy_helper.from_array(new_bias, new_bias_name)
        initializers[shape1_name] = numpy_helper.from_array(
            np.asarray([-1, cout, 2, 2, input_shape[2], input_shape[3]], dtype=np.int64), shape1_name
        )
        initializers[shape2_name] = numpy_helper.from_array(
            np.asarray([-1, cout, input_shape[2] * 2, input_shape[3] * 2], dtype=np.int64), shape2_name
        )

        conv = helper.make_node(
            "Conv",
            [node.input[0], new_weight_name, new_bias_name],
            [conv_output],
            name=base + "__subpixel_conv",
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            dilations=[1, 1],
            group=1,
        )
        reshape1 = helper.make_node("Reshape", [conv_output, shape1_name], [reshape1_output], name=base + "__subpixel_reshape")
        transpose = helper.make_node(
            "Transpose", [reshape1_output], [transpose_output], name=base + "__subpixel_transpose", perm=[0, 1, 4, 2, 5, 3]
        )
        reshape2 = helper.make_node("Reshape", [transpose_output, shape2_name], [node.output[0]], name=base + "__subpixel_pixelshuffle")
        rewritten.extend([conv, reshape1, transpose, reshape2])
        converted.append({
            "original_node": base,
            "replacement": "Conv(k=3,p=1)+Reshape+Transpose+Reshape(PixelShuffle2)",
            "input_channels": int(cin),
            "output_channels": int(cout),
            "input_spatial": [int(input_shape[2]), int(input_shape[3])],
            "output_spatial": [int(input_shape[2] * 2), int(input_shape[3] * 2)],
        })

    if not converted:
        raise RuntimeError("No safe stride-2 ConvTranspose nodes were converted")
    result.graph.ClearField("node")
    result.graph.node.extend(rewritten)
    used_initializers = {name for node in result.graph.node for name in node.input if name}
    result.graph.ClearField("initializer")
    result.graph.initializer.extend([item for name, item in initializers.items() if name in used_initializers])
    result = onnx.shape_inference.infer_shapes(result)
    onnx.checker.check_model(result)
    return result, converted


def make_session(ort, path, profile=False):
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_profiling = bool(profile)
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def benchmark(session, input_name, noise, warmup, iters):
    for _ in range(warmup):
        session.run(None, {input_name: noise})
    values = []
    for _ in range(iters):
        start = time.perf_counter()
        session.run(None, {input_name: noise})
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


def finish_profile(session, target):
    source = Path(session.end_profiling())
    shutil.copy2(source, target)


def aggregate_profile(profile_path, output_path, stage):
    events = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    grouped = defaultdict(list)
    for event in events:
        if event.get("cat") != "Node" or "dur" not in event:
            continue
        args = event.get("args", {})
        name = args.get("op_name") or event.get("name", "unknown")
        grouped[str(name)].append(float(event["dur"]))
    rows = []
    for operator, durations in grouped.items():
        values = np.asarray(durations, dtype=np.float64)
        rows.append([stage, operator, len(values), float(values.sum() / 1000.0), float(values.mean() / 1000.0), float(np.percentile(values, 95) / 1000.0)])
    rows.sort(key=lambda row: row[3], reverse=True)
    write_csv(output_path, ["stage", "operator", "calls", "total_ms", "mean_ms", "p95_ms"], rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    ensure_dependencies()
    import onnx
    import onnxruntime as ort

    raw_path = locate_raw(args.raw_onnx)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_model = onnx.load(str(raw_path))
    onnx.checker.check_model(raw_model)
    subpixel_model, converted = convert_stride2_convtranspose(raw_model)
    fused_path = output_dir / "generator_fp32_subpixel_fused.onnx"
    onnx.save(subpixel_model, str(fused_path))
    onnx.checker.check_model(onnx.load(str(fused_path)))

    raw_session = make_session(ort, raw_path)
    fused_session = make_session(ort, fused_path)
    raw_name = raw_session.get_inputs()[0].name
    fused_name = fused_session.get_inputs()[0].name
    latency_rows = []
    difference_rows = []
    all_results = {}
    for batch in args.batches:
        noise = np.random.default_rng(1000 + batch).standard_normal((batch, NOISE_DIM, 1, 1), dtype=np.float32)
        raw_output = raw_session.run(None, {raw_name: noise})[0]
        fused_output = fused_session.run(None, {fused_name: noise})[0]
        difference_rows.append([batch, float(np.max(np.abs(raw_output - fused_output))), float(np.mean(np.abs(raw_output - fused_output)))])
        raw_timing = benchmark(raw_session, raw_name, noise, args.warmup, args.iters)
        fused_timing = benchmark(fused_session, fused_name, noise, args.warmup, args.iters)
        all_results[str(batch)] = {"raw": raw_timing, "subpixel_fused": fused_timing}
        for stage, timing in (("raw", raw_timing), ("subpixel_fused", fused_timing)):
            latency_rows.append([stage, "CPUExecutionProvider", batch, args.warmup, args.iters, *timing.values()])

    write_csv(output_dir / "subpixel_latency_summary.csv", ["graph", "provider", "batch", "warmup", "iterations", "mean_ms", "p50_ms", "p95_ms", "min_ms", "max_ms", "throughput_images_per_s"], latency_rows)
    write_csv(output_dir / "numerical_equivalence.csv", ["batch", "max_abs_difference", "mean_abs_difference"], difference_rows)

    profile_batch = args.batches[0]
    profile_noise = np.random.default_rng(4242).standard_normal((profile_batch, NOISE_DIM, 1, 1), dtype=np.float32)
    raw_profile_session = make_session(ort, raw_path, profile=True)
    fused_profile_session = make_session(ort, fused_path, profile=True)
    raw_profile_name = raw_profile_session.get_inputs()[0].name
    fused_profile_name = fused_profile_session.get_inputs()[0].name
    for _ in range(args.warmup):
        raw_profile_session.run(None, {raw_profile_name: profile_noise})
        fused_profile_session.run(None, {fused_profile_name: profile_noise})
    for _ in range(args.iters):
        raw_profile_session.run(None, {raw_profile_name: profile_noise})
        fused_profile_session.run(None, {fused_profile_name: profile_noise})
    raw_profile_path = output_dir / "ort_profile_raw.json"
    fused_profile_path = output_dir / "ort_profile_subpixel_fused.json"
    finish_profile(raw_profile_session, raw_profile_path)
    finish_profile(fused_profile_session, fused_profile_path)
    raw_ops = aggregate_profile(raw_profile_path, output_dir / "operator_latency_raw.csv", "raw")
    fused_ops = aggregate_profile(fused_profile_path, output_dir / "operator_latency_subpixel_fused.csv", "subpixel_fused")
    combined_rows = []
    for path, stage in ((output_dir / "operator_latency_raw.csv", "raw"), (output_dir / "operator_latency_subpixel_fused.csv", "subpixel_fused")):
        with open(path, newline="", encoding="utf-8") as handle:
            combined_rows.extend(list(csv.DictReader(handle)))
    write_csv(output_dir / "operator_latency_comparison.csv", ["stage", "operator", "calls", "total_ms", "mean_ms", "p95_ms"], [[row[key] for key in ("stage", "operator", "calls", "total_ms", "mean_ms", "p95_ms")] for row in combined_rows])

    max_diff = max(row[1] for row in difference_rows)
    improvements = []
    for batch in args.batches:
        raw_mean = all_results[str(batch)]["raw"]["mean_ms"]
        fused_mean = all_results[str(batch)]["subpixel_fused"]["mean_ms"]
        improvements.append({"batch": batch, "mean_speedup_ratio": float(raw_mean / fused_mean), "mean_improvement_percent": float((raw_mean / fused_mean - 1.0) * 100.0)})
    improved_batches = sum(item["mean_improvement_percent"] >= 5.0 for item in improvements)
    passed = bool(max_diff <= 1e-5 and improved_batches >= 2)
    metadata = {
        "task": "Task1_01D_Subpixel_Reparameterization",
        "raw_onnx": str(raw_path),
        "fused_onnx": str(fused_path),
        "raw_node_count": len(raw_model.graph.node),
        "fused_node_count": len(subpixel_model.graph.node),
        "converted_layers": converted,
        "max_abs_output_difference": max_diff,
        "improvements": improvements,
        "pass_criteria": {"max_abs_difference_le_1e-5": bool(max_diff <= 1e-5), "at_least_two_batches_improve_mean_latency_by_5_percent": bool(improved_batches >= 2)},
        "optimization_decision": "pass" if passed else "not_passed",
        "top_raw_operators": raw_ops[:3],
        "top_subpixel_operators": fused_ops[:3],
    }
    save_json(output_dir / "subpixel_fusion_check.json", metadata)
    archive_path = output_dir.parent / "Task1_01D_Subpixel_Reparam.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in sorted(output_dir.iterdir()):
            if file_path.is_file():
                archive.write(file_path, arcname=f"01D_Subpixel_Reparam/{file_path.name}")
    print(f"[01D] converted_layers={len(converted)} raw_nodes={len(raw_model.graph.node)} fused_nodes={len(subpixel_model.graph.node)}")
    print(f"[01D] max_abs_difference={max_diff:.8g} decision={'PASS' if passed else 'NOT_PASSED'}")
    print(f"[01D] outputs={output_dir}")


if __name__ == "__main__":
    main()
