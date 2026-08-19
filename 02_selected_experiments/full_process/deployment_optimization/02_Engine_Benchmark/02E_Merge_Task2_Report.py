"""
Task 2E — Merge ORT, TensorRT, OpenVINO and profiler outputs.

Run this after 02A/02B/02C/02D. It does not execute inference; it only
combines saved CSV/JSON evidence and explicitly reports missing engines.
"""

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Merge Task 2 benchmark results")
    p.add_argument("--input-root", default=os.getenv("TASK2_RESULTS_ROOT", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark"))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark/02E_Report"))
    args, _ = p.parse_known_args(argv)
    return args


def find_file(root, name):
    matches = sorted(root.rglob(name)) if root.exists() else []
    return matches[0] if matches else None


def read_csv(path):
    if not path:
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def relative_or_name(path, root):
    if not path:
        return None
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def observed_batches(rows):
    values = set()
    for row in rows:
        try:
            values.add(int(float(row.get("batch", ""))))
        except (TypeError, ValueError):
            continue
    return sorted(values)


def observed_precisions(rows):
    return sorted({row.get("precision", "") for row in rows if row.get("precision", "")})


def evidence_present(path, filename):
    """Treat an empty CSV/trace as missing evidence, not as a completed run."""
    if not path or not path.exists() or path.stat().st_size == 0:
        return False
    if filename.endswith(".csv"):
        return bool(read_csv(path))
    return True


def successful_rows(path):
    """Return benchmark rows that completed inference, excluding skip rows."""
    if not path:
        return []
    rows = read_csv(path)
    return [row for row in rows if row.get("status", "ok").strip().lower() == "ok"]


def finite_value(row, *keys):
    """Return the first finite numeric field, otherwise an empty string."""
    for key in keys:
        value = row.get(key, "")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number == number and abs(number) != float("inf"):
            return value
    return ""


def write_csv(path, rows):
    if not rows:
        path.write_text("status\nmissing\n", encoding="utf-8")
        return
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def make_top3(root, output):
    """Normalize available operator evidence and emit auditable Top-3 rows."""
    specs = [
        ("operator_latency_raw.csv", "ORT raw", "operator", "total_ms"),
        ("operator_latency_fused.csv", "ORT graph optimized", "operator", "total_ms"),
        ("operator_latency_manual_bn_fused.csv", "ORT manual BN", "operator", "total_ms"),
        ("ort_operator_summary.csv", "ORT engine profile", "operator", "total_ms"),
        ("torch_operator_summary.csv", "PyTorch profiler", "operator", "self_device_ms"),
        ("layer_operator_summary.csv", "PyTorch layer profiler", "layer", "self_device_ms"),
        ("tensorrt_layer_profile.csv", "TensorRT IProfiler", "layer", "time_ms"),
        ("openvino_operator_profile.csv", "OpenVINO profiling", "node_name", "real_time_ms"),
    ]
    rows = []
    for filename, source, name_key, time_key in specs:
        path = find_file(root, filename)
        if not path:
            continue
        grouped = {}
        for row in read_csv(path):
            name = row.get(name_key, "")
            if source == "PyTorch profiler":
                lower_name = name.lower()
                if (
                    name in {"Exp11_Generator_Inference", "cudaDeviceSynchronize", "cudaLaunchKernel", "Activity Buffer Request"}
                    or lower_name.startswith("void ")
                    or "dgrad_engine" in lower_name
                    or "cuda" in lower_name
                    or "cudnn" in lower_name and "aten::" not in lower_name
                ):
                    continue
            if source in {"TensorRT IProfiler", "OpenVINO profiling"}:
                lower_name = name.lower()
                if "reformatting copynode" in lower_name or "casttanhcast" in lower_name:
                    continue
            try:
                value = float(row.get(time_key, ""))
            except (TypeError, ValueError):
                continue
            if not name or not value == value:
                continue
            grouped[name] = grouped.get(name, 0.0) + value
        ranked = sorted(grouped.items(), key=lambda item: item[1], reverse=True)[:3]
        for rank, (name, total_ms) in enumerate(ranked, 1):
            rows.append({"source": source, "rank": rank, "operator_or_layer": name, "total_ms": total_ms, "evidence_file": filename})
    write_csv(output / "task2_top3_operators.csv", rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    ort_rows = read_csv(find_file(root, "ort_benchmark.csv"))
    trt_rows = read_csv(find_file(root, "tensorrt_benchmark.csv"))
    ov_rows = read_csv(find_file(root, "openvino_benchmark.csv"))
    merged = []
    for row in ort_rows:
        row = dict(row)
        row["engine"] = "ONNX Runtime"
        row["device"] = row.get("provider", "")
        row["end_to_end_mean_ms"] = row.get("mean_ms", "")
        row["end_to_end_p50_ms"] = row.get("p50_ms", "")
        row["end_to_end_p95_ms"] = row.get("p95_ms", "")
        row["memory_mb"] = finite_value(row, "gpu_allocated_mb", "host_rss_mb")
        row["memory_source"] = "CUDA allocator if finite; otherwise host RSS"
        merged.append(row)
    for row in trt_rows:
        row = dict(row)
        row["engine"] = "TensorRT"
        row["device"] = "CUDA"
        row["end_to_end_mean_ms"] = row.get("end_to_end_mean_ms", "")
        row["end_to_end_p50_ms"] = row.get("end_to_end_p50_ms", "")
        row["end_to_end_p95_ms"] = row.get("end_to_end_p95_ms", "")
        row["memory_mb"] = finite_value(row, "cuda_peak_used_snapshot_mb", "peak_allocated_mb")
        row["memory_source"] = "whole-device CUDA mem_get_info snapshot"
        merged.append(row)
    for row in ov_rows:
        row = dict(row)
        row["engine"] = "OpenVINO"
        row["device"] = "CPU"
        row["end_to_end_mean_ms"] = row.get("mean_ms", "")
        row["end_to_end_p50_ms"] = row.get("p50_ms", "")
        row["end_to_end_p95_ms"] = row.get("p95_ms", "")
        row["memory_mb"] = row.get("host_rss_mb", "")
        merged.append(row)
    write_csv(output / "task2_engine_comparison.csv", merged)

    operator_rows = []
    for name, engine in (("operator_latency_raw.csv", "ORT raw"), ("operator_latency_fused.csv", "ORT graph optimized"), ("operator_latency_manual_bn_fused.csv", "ORT manual BN"), ("ort_operator_summary.csv", "ORT engine profile"), ("torch_operator_summary.csv", "PyTorch profiler"), ("layer_operator_summary.csv", "PyTorch layer profiler"), ("tensorrt_layer_profile.csv", "TensorRT IProfiler"), ("openvino_operator_profile.csv", "OpenVINO profiling")):
        path = find_file(root, name)
        if not path:
            continue
        for row in read_csv(path):
            row = dict(row)
            row["engine"] = engine
            operator_rows.append(row)
    write_csv(output / "task2_operator_evidence.csv", operator_rows)
    top3_rows = make_top3(root, output)

    required_files = {
        "ort_benchmark": "ort_benchmark.csv",
        "ort_operator_summary": "ort_operator_summary.csv",
        "tensorrt_benchmark": "tensorrt_benchmark.csv",
        "openvino_benchmark": "openvino_benchmark.csv",
        "torch_trace": "torch_trace.json",
        "torch_operator_summary": "torch_operator_summary.csv",
        "layer_trace": "layer_profiler_trace.json",
        "layer_operator_summary": "layer_operator_summary.csv",
        "tensorrt_layer_profile": "tensorrt_layer_profile.csv",
        "openvino_operator_profile": "openvino_operator_profile.csv",
    }
    located = {key: find_file(root, filename) for key, filename in required_files.items()}
    missing = [filename for key, filename in required_files.items() if not evidence_present(located[key], filename)]
    for key in ("ort_benchmark", "tensorrt_benchmark", "openvino_benchmark"):
        if located[key] and not successful_rows(located[key]):
            missing.append(f"{key}:no_successful_rows")
    expected_precisions = {"FP32", "FP16"}
    expected_batches = {1, 4, 8, 16, 32}
    coverage_missing = []
    for engine_name, rows in (("ONNX Runtime", successful_rows(located["ort_benchmark"])), ("TensorRT", successful_rows(located["tensorrt_benchmark"])), ("OpenVINO", successful_rows(located["openvino_benchmark"]))):
        precisions = {row.get("precision", "") for row in rows}
        batches = {int(float(row["batch"])) for row in rows if row.get("batch", "").strip().isdigit()}
        if not expected_precisions.issubset(precisions):
            coverage_missing.append(f"{engine_name}:FP32+FP16")
        if not expected_batches.issubset(batches):
            coverage_missing.append(f"{engine_name}:batch_1_4_8_16_32")
    missing.extend(coverage_missing)
    profile_files = sorted(root.rglob("ort_profile_*.json")) if root.exists() else []
    evidence_files = {
        "ort_benchmark": relative_or_name(find_file(root, "ort_benchmark.csv"), root),
        "ort_operator_summary": relative_or_name(find_file(root, "ort_operator_summary.csv"), root),
        "ort_profiles": [relative_or_name(path, root) for path in profile_files],
        "ort_profile_status": "retained" if profile_files else "not_retained_summary_available",
        "tensorrt_benchmark": relative_or_name(find_file(root, "tensorrt_benchmark.csv"), root),
        "tensorrt_layer_profile": relative_or_name(find_file(root, "tensorrt_layer_profile.csv"), root),
        "openvino_benchmark": relative_or_name(find_file(root, "openvino_benchmark.csv"), root),
        "openvino_operator_profile": relative_or_name(find_file(root, "openvino_operator_profile.csv"), root),
        "torch_trace": relative_or_name(find_file(root, "torch_trace.json"), root),
        "torch_operator_summary": relative_or_name(find_file(root, "torch_operator_summary.csv"), root),
        "layer_trace": relative_or_name(find_file(root, "layer_profiler_trace.json"), root),
        "layer_operator_summary": relative_or_name(find_file(root, "layer_operator_summary.csv"), root),
    }
    report = ["# Task 2 Engine Benchmark Report", "", "## Status", ""]
    report.append("- Required evidence missing: " + ", ".join(missing) if missing else "- All mandatory benchmark and trace evidence files were found.")
    report.append("- ORT raw profile JSON is optional when ort_operator_summary.csv is present; the summary is the retained ORT operator evidence.")
    report.append("- TensorRT/OpenVINO backend operator profiles are mandatory for a complete Task 2 audit.")
    report.extend(["", "## Output map", "", "- `task2_engine_comparison.csv`: FP32/FP16 end-to-end latency, throughput and memory comparison.", "- `task2_operator_evidence.csv`: normalized operator/layer evidence from all available profilers.", "- `task2_top3_operators.csv`: per-source Top-3 ranking, aggregated over the recorded precision/batch rows.", "- `task2_report.md`: human-readable interpretation and graph-level actions.", "- `task2_manifest.json`: the only consolidated run metadata file.", "- Raw ORT profiles and Chrome Trace JSON files are retained because they are evidence, not duplicate metadata."])
    report.extend(["", "## Top-3 operator evidence", "", "| Source | Rank | Operator/layer | Total ms | Evidence |", "|---|---:|---|---:|---|"])
    if top3_rows:
        for row in top3_rows:
            report.append(f"| {row['source']} | {row['rank']} | {row['operator_or_layer']} | {float(row['total_ms']):.6f} | {row['evidence_file']} |")
    else:
        report.append("| unavailable | | No usable operator profile was found | | |")
    report.extend(["", "### Graph-level optimization actions", "", "- ConvTranspose: benchmark TensorRT tactics, consider upsample-plus-convolution only as a controlled architecture ablation, and avoid unnecessary host/device copies.", "- BatchNormalization: use the existing 01C inference-time BN folding result and keep the numerical-equivalence CSV.", "- ReLU/Tanh/elementwise chains: rely on backend fusion and inspect the optimized graph before adding manual rewrites."])
    report.extend(["", "## Interpretation rules", "", "- Main comparison should use the same 01A raw ONNX.", "- TensorRT FP16 is an engine/precision result, not a training result.", "- ORT profiles are not substitutes for torch.profiler Chrome Trace.", "- Top-3 operators must be supported by the saved profile/CSV, not assumed in advance.", "- Exclude profiler warm-up, runtime module loading and synchronization-only rows when naming model bottlenecks.", "- TensorRT peak_allocated_mb is only the PyTorch allocator, not isolated total TensorRT memory.", "- The normalized Top-3 evidence is saved in task2_top3_operators.csv; missing backend profiles remain explicit rather than inferred."])
    (output / "task2_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    manifest = {
        "task": "Task2_Engine_Benchmark",
        "status": "complete" if not missing else "incomplete",
        "input_root": str(root),
        "common_model": "01A generator_fp32_raw.onnx",
        "benchmark_protocol": {
            "default_batches": [1, 4, 8, 16, 32],
            "latency": "mean/p50/p95 in milliseconds",
            "throughput": "batch divided by mean latency",
            "warmup_and_iterations": "recorded in each engine CSV",
            "fixed_noise_seed": "42 + batch",
        },
        "coverage": {
            "onnxruntime": {"precisions": observed_precisions(ort_rows), "batches": observed_batches(ort_rows)},
            "tensorrt": {"precisions": observed_precisions(trt_rows), "batches": observed_batches(trt_rows)},
            "openvino": {"precisions": observed_precisions(ov_rows), "batches": observed_batches(ov_rows)},
        },
        "memory_notes": {
            "ort": "host_rss_mb and optional PyTorch CUDA allocator value",
            "tensorrt": "cuda_* fields are whole-device mem_get_info snapshots; peak_allocated_mb is PyTorch-only diagnostic",
            "openvino": "host_rss_mb",
        },
        "evidence_files": evidence_files,
        "outputs": {
            "comparison_csv": "task2_engine_comparison.csv",
            "operator_evidence_csv": "task2_operator_evidence.csv",
            "top3_csv": "task2_top3_operators.csv",
            "report": "task2_report.md",
            "manifest": "task2_manifest.json",
        },
        "merged_rows": len(merged),
        "operator_evidence_rows": len(operator_rows),
        "top3_rows": len(top3_rows),
        "missing": missing,
    }
    (output / "task2_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = output.parent / "Task2_02E_Report.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in output.rglob("*"):
            if file_path.is_file():
                z.write(file_path, arcname=f"02E_Report/{file_path.relative_to(output).as_posix()}")
    print("[done] merged rows:", len(merged))
    print("[missing]", missing)
    print("[zip]", archive)


if __name__ == "__main__":
    main()
