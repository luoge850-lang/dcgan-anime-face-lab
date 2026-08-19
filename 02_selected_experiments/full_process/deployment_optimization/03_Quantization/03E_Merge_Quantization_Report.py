"""
Task 3E - create the final local quantization report and visualizations.

Input is the downloaded 03D_Quality_Evaluation directory. This script does
not rerun TensorRT; it audits the already measured FP32/FP16/INT8 rows and
creates a reproducible Markdown report, JSON decision record and PNG charts.
"""

from __future__ import annotations

import argparse
import csv
import json
import html
import shutil
import zipfile
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Merge Task 3 quantization results")
    parser.add_argument(
        "--input-dir",
        default="C:/Users/32875/OneDrive/Desktop/image_generator/dcgan_lab/results/Deployment_Optimization_Results/03_Quantization/03D_Evaluation",
    )
    parser.add_argument(
        "--output-dir",
        default="C:/Users/32875/OneDrive/Desktop/image_generator/dcgan_lab/results/Deployment_Optimization_Results/03_Quantization/03E_Report",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json_if_exists(path):
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_downloaded_file(directory, name):
    """Resolve an exact result name or a browser-renamed ``name (1).ext``."""
    exact = directory / name
    if exact.exists():
        return exact
    path = Path(name)
    matches = sorted(directory.glob(f"{path.stem}*.{path.suffix.lstrip('.')}"))
    return matches[0] if matches else exact


def read_json_from_zip(path, member_name):
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open(member_name) as handle:
                return json.loads(handle.read().decode("utf-8"))
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None


def locate_upstream_manifests(input_dir):
    root = input_dir.parent
    names = {
        "03A_protocol": "calibration_manifest.json",
        "03B_fp32_fp16_build": "fp32_fp16_build_manifest.json",
        "03C_int8_build": "int8_build_manifest.json",
        "03D_evaluation": "evaluation_manifest.json",
    }
    result = {}
    for key, name in names.items():
        matches = sorted(root.rglob(name)) if root.exists() else []
        if not matches and root.exists():
            # Browser downloads may rename a duplicate to "name (1).json".
            # Accept that transport-level suffix, but still validate its
            # contents below instead of treating any JSON as evidence.
            stem = Path(name).stem
            matches = sorted(root.rglob(f"{stem}*.json"))
        payload = read_json_if_exists(matches[0]) if matches else None
        if payload is None and key == "03A_protocol" and root.exists():
            protocol_zips = sorted(root.rglob("Task3_03A_Quantization_Protocol*.zip"))
            if protocol_zips:
                payload = read_json_from_zip(protocol_zips[0], name)
        result[key] = payload
    return result


def f(row, key):
    return float(row[key])


def pct(new, old):
    return (new / old - 1.0) * 100.0 if old else 0.0


def validate_upstream_manifests(manifests):
    protocol = manifests.get("03A_protocol") or {}
    build = manifests.get("03B_fp32_fp16_build") or {}
    int8 = manifests.get("03C_int8_build") or {}
    evaluation = manifests.get("03D_evaluation") or {}
    checks = {
        "03A_has_100_real_calibration_images": protocol.get("calibration_real_count") == 100,
        "03A_has_fixed_eval_latents": int(protocol.get("latent_eval_count", 0)) >= 1000,
        "03B_has_fp32_and_fp16_engines": set((build.get("engines") or {})) >= {"FP32", "FP16"},
        "03C_has_100_real_calibration_images": int8.get("real_calibration_count") == 100,
        "03C_has_int8_engine": bool((int8.get("engine") or {}).get("file")),
        "03D_has_three_engine_records": set((evaluation.get("engines") or {})) >= {"FP32", "FP16", "INT8"},
        "03D_has_standard_fid": evaluation.get("standard_fid_computed") is True,
    }
    return checks


def frequency_diagnosis(rows, deltas):
    int8 = [row for row in rows if row.get("comparison") == "INT8_vs_FP32"]
    by_band = {row.get("subband"): row for row in int8}
    high_bands = [by_band[name] for name in ("LH", "HL", "HH") if name in by_band]
    ll = float(by_band["LL"]["mae_01"]) if "LL" in by_band else None
    high_mean = sum(float(row["mae_01"]) for row in high_bands) / len(high_bands) if high_bands else None
    ratio = (high_mean / ll) if ll not in (None, 0.0) and high_mean is not None else None
    relative_ratios = {
        band: float(row["error_to_baseline_ratio"])
        for band, row in by_band.items()
        if row.get("error_to_baseline_ratio") not in (None, "")
    }
    high_relative_bands = [relative_ratios[band] for band in ("LH", "HL", "HH") if band in relative_ratios]
    high_relative_mean = sum(high_relative_bands) / len(high_relative_bands) if high_relative_bands else None
    ll_relative = relative_ratios.get("LL")
    relative_ratio = high_relative_mean / ll_relative if ll_relative not in (None, 0.0) and high_relative_mean is not None else None
    dominant = bool(ratio is not None and ratio >= 1.25)
    quality_move = bool(
        deltas["INT8"]["legacy_fid_delta"] > 0.0
        or deltas["INT8"]["standard_fid_delta"] > 0.0
        or deltas["INT8"]["blur_rate_delta_pp"] > 0.0
        or deltas["INT8"]["lpips_delta_percent"] < 0.0
    )
    if relative_ratio is not None and relative_ratio > 1.0 and quality_move:
        conclusion = "high_frequency_error_dominant_and_quality_degraded"
        explanation = "INT8 absolute MAE is LL-dominated, but LH/HL/HH error relative to each band's baseline is larger while quality worsens; high-frequency scale truncation is supported as a contributing mechanism."
    elif high_bands and quality_move:
        conclusion = "quality_degraded_but_high_frequency_dominance_not_proven"
        explanation = "INT8 quality worsened, but the Haar measurements do not show a clear high-frequency-dominant error; inspect activation ranges, calibration coverage and Tanh saturation before attributing causality."
    else:
        conclusion = "no_measured_int8_high_frequency_failure"
        explanation = "The current measurements do not establish a high-frequency truncation failure."
    return {
        "int8_subband_mae": {name: float(row["mae_01"]) for name, row in by_band.items()},
        "ll_mae_01": ll,
        "high_frequency_mean_mae_01": high_mean,
        "high_to_ll_mae_ratio": ratio,
        "high_frequency_relative_error_mean": high_relative_mean,
        "ll_relative_error": ll_relative,
        "high_to_ll_relative_error_ratio": relative_ratio,
        "high_frequency_dominant_threshold": 1.25,
        "quality_metrics_move_worse": quality_move,
        "conclusion": conclusion,
        "explanation": explanation,
    }


def write_svg_bar_chart(path, title, labels, series, y_label):
    """Dependency-free fallback visualization for offline/local execution."""
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 90, 110
    plot_w, plot_h = width - left - right, height - top - bottom
    all_values = [value for _name, _color, values in series for value in values]
    max_value = max(all_values) if all_values else 1.0
    max_value = max(max_value * 1.15, 1e-9)
    colors = [color for _name, color, _values in series]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="42" font-family="Arial,Microsoft YaHei,sans-serif" font-size="26" font-weight="700">{html.escape(title)}</text>',
        f'<text x="{left}" y="68" font-family="Arial,Microsoft YaHei,sans-serif" font-size="14" fill="#64748b">{html.escape(y_label)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>',
    ]
    for tick in range(5):
        value = max_value * tick / 4.0
        y = top + plot_h - plot_h * tick / 4.0
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#64748b">{value:.3g}</text>')
    group_w = plot_w / max(len(labels), 1)
    bar_w = min(60, group_w / max(len(series), 1) * 0.7)
    for group_index, label in enumerate(labels):
        center = left + group_w * (group_index + 0.5)
        parts.append(f'<text x="{center:.1f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial" font-size="14">{html.escape(label)}</text>')
        for series_index, (_name, color, values) in enumerate(series):
            value = values[group_index]
            x = center + (series_index - (len(series) - 1) / 2) * bar_w
            bar_h = plot_h * value / max_value
            y = top + plot_h - bar_h
            parts.append(f'<rect x="{x - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="4"/>')
            parts.append(f'<text x="{x:.1f}" y="{max(y - 6, top + 14):.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.4g}</text>')
    legend_x = left
    for name, color, _values in series:
        parts.append(f'<rect x="{legend_x}" y="{height - 50}" width="14" height="14" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{height - 38}" font-family="Arial" font-size="13">{html.escape(name)}</text>')
        legend_x += 170
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def make_charts(rows, errors, out):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    labels = [r["precision"] for r in rows]
    chart_values = [
        ("Legacy FID", "#2563eb", [f(r, "fid_legacy_inception_v3") for r in rows]),
        ("Standard FID", "#0f766e", [f(r, "fid_standard_inception_v3") for r in rows]),
        ("Blur rate", "#dc2626", [f(r, "fake_blur_rate") for r in rows]),
        ("LPIPS diversity", "#16a34a", [f(r, "lpips_alex_diversity") for r in rows]),
        ("Throughput", "#9333ea", [f(r, "inference_images_per_s") for r in rows]),
    ]
    write_svg_bar_chart(out / "task3_quality_performance.svg", "Task 3 FP32 / FP16 / INT8 comparison", labels, chart_values, "higher/lower depends on metric")
    error_values = [
        ("MAE", "#2563eb", [f(r, "mae_01") for r in errors]),
        ("RMSE", "#16a34a", [f(r, "rmse_01") for r in errors]),
        ("Max abs", "#dc2626", [f(r, "max_abs_01") for r in errors]),
    ]
    write_svg_bar_chart(out / "task3_quantization_error.svg", "Quantization error relative to FP32", [r["comparison"] for r in errors], error_values, "absolute image error")
    if plt is None:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    charts = [
        ("fid_legacy_inception_v3", "Legacy FID", "lower is better"),
        ("fid_standard_inception_v3", "Standard FID", "lower is better"),
        ("fake_blur_rate", "Blur rate", "lower is better"),
        ("lpips_alex_diversity", "LPIPS diversity", "higher means more pairwise diversity"),
        ("inference_images_per_s", "Throughput (images/s)", "higher is better"),
    ]
    for ax, (key, title, subtitle) in zip(axes.flat, charts):
        values = [f(r, key) for r in rows]
        colors = ["#2563eb", "#16a34a", "#dc2626"]
        ax.bar(labels, values, color=colors[: len(values)])
        ax.set_title(title)
        ax.set_ylabel(subtitle)
        ax.grid(axis="y", alpha=0.25)
        for i, value in enumerate(values):
            ax.text(i, value, f"{value:.4g}", ha="center", va="bottom", fontsize=9)
    axes.flat[-1].axis("off")
    fig.suptitle("Task 3 FP32 / FP16 / INT8 comparison")
    fig.tight_layout()
    fig.savefig(out / "task3_quality_performance.png", dpi=180)
    plt.close(fig)

    err_labels = [r["comparison"] for r in errors]
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(len(err_labels)))
    width = 0.25
    for offset, key, title in [(-width, "mae_01", "MAE"), (0, "rmse_01", "RMSE"), (width, "max_abs_01", "Max abs")]:
        vals = [f(r, key) for r in errors]
        ax.bar([i + offset for i in x], vals, width=width, label=title)
    ax.set_xticks(x, err_labels)
    ax.set_title("Quantization error relative to FP32")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "task3_quantization_error.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = resolve_downloaded_file(input_dir, "fp32_fp16_int8_metrics.csv")
    errors_path = resolve_downloaded_file(input_dir, "quantization_error.csv")
    frequency_path = resolve_downloaded_file(input_dir, "frequency_band_error.csv")
    manifest_path = resolve_downloaded_file(input_dir, "evaluation_manifest.json")
    # The NPZ files are intermediate feature caches, not report inputs. 03D
    # already materializes the standard-FID values into the metrics CSV; the
    # report must therefore remain reproducible when only the tabular outputs
    # were downloaded from Kaggle.
    standard_stats_path = input_dir / "real_fid_standard_stats.npz"
    for required in (metrics_path, errors_path, manifest_path, frequency_path):
        if not required.exists():
            raise FileNotFoundError(f"Missing 03D output: {required}")
    evaluation_manifest = read_json_if_exists(manifest_path)
    if not evaluation_manifest or not evaluation_manifest.get("standard_fid_computed"):
        raise ValueError("03D evaluation_manifest does not confirm standard FID; rerun 03D before creating the final report")
    rows = read_csv(metrics_path)
    errors = read_csv(errors_path)
    frequency_rows = read_csv(frequency_path)
    if not frequency_rows:
        raise ValueError("frequency_band_error.csv is empty; rerun 03D before creating the final report")
    by_precision = {r["precision"]: r for r in rows}
    required_precisions = {"FP32", "FP16", "INT8"}
    if set(by_precision) != required_precisions:
        raise ValueError(f"Expected FP32/FP16/INT8 rows, found {sorted(by_precision)}")
    required_metric_columns = {
        "fid_legacy_inception_v3",
        "fid_standard_inception_v3",
        "fake_blur_rate",
        "lpips_alex_diversity",
        "inference_images_per_s",
    }
    missing_metric_columns = sorted(required_metric_columns - set(rows[0])) if rows else sorted(required_metric_columns)
    if missing_metric_columns:
        raise ValueError(
            "03D metrics are incomplete; rerun 03D after the standard-FID update. "
            f"Missing columns: {missing_metric_columns}"
        )
    fp32, fp16, int8 = by_precision["FP32"], by_precision["FP16"], by_precision["INT8"]
    deltas = {
        "FP16": {
            "legacy_fid_delta": f(fp16, "fid_legacy_inception_v3") - f(fp32, "fid_legacy_inception_v3"),
            "standard_fid_delta": f(fp16, "fid_standard_inception_v3") - f(fp32, "fid_standard_inception_v3"),
            "blur_rate_delta_pp": (f(fp16, "fake_blur_rate") - f(fp32, "fake_blur_rate")) * 100,
            "laplacian_delta_percent": pct(f(fp16, "fake_laplacian_mean"), f(fp32, "fake_laplacian_mean")),
            "lpips_delta_percent": pct(f(fp16, "lpips_alex_diversity"), f(fp32, "lpips_alex_diversity")),
            "throughput_speedup_percent": pct(f(fp16, "inference_images_per_s"), f(fp32, "inference_images_per_s")),
        },
        "INT8": {
            "legacy_fid_delta": f(int8, "fid_legacy_inception_v3") - f(fp32, "fid_legacy_inception_v3"),
            "standard_fid_delta": f(int8, "fid_standard_inception_v3") - f(fp32, "fid_standard_inception_v3"),
            "blur_rate_delta_pp": (f(int8, "fake_blur_rate") - f(fp32, "fake_blur_rate")) * 100,
            "laplacian_delta_percent": pct(f(int8, "fake_laplacian_mean"), f(fp32, "fake_laplacian_mean")),
            "lpips_delta_percent": pct(f(int8, "lpips_alex_diversity"), f(fp32, "lpips_alex_diversity")),
            "throughput_speedup_percent": pct(f(int8, "inference_images_per_s"), f(fp32, "inference_images_per_s")),
        },
    }
    fp16_quality_warning = bool(
        deltas["FP16"]["legacy_fid_delta"] > 2.0
        or deltas["FP16"]["standard_fid_delta"] > 2.0
        or deltas["FP16"]["blur_rate_delta_pp"] > 1.0
        or deltas["FP16"]["lpips_delta_percent"] < -2.0
    )
    int8_quality_warning = bool(
        deltas["INT8"]["legacy_fid_delta"] > 2.0
        or deltas["INT8"]["standard_fid_delta"] > 2.0
        or deltas["INT8"]["blur_rate_delta_pp"] > 1.0
        or deltas["INT8"]["lpips_delta_percent"] < -2.0
    )
    frequency_analysis = frequency_diagnosis(frequency_rows, deltas)
    upstream_manifests = locate_upstream_manifests(input_dir)
    upstream_checks = validate_upstream_manifests(upstream_manifests)
    upstream_evidence_complete = all(upstream_checks.values())
    decision = {
        "task": "Task3_03E_Quantization_Report",
        "status": "complete" if upstream_evidence_complete else "incomplete",
        "input_03d": str(input_dir),
        "fid_protocol": {
            "legacy": "project historical torchvision Inception-v3 protocol",
            "standard": "public-style Inception-v3 pool3, 2048 dimensions, ImageNet normalization",
        },
        "rows": rows,
        "quantization_error": errors,
        "frequency_band_error": frequency_rows,
        "int8_frequency_diagnosis": frequency_analysis,
        "deltas_vs_fp32": deltas,
        "fp16_quality_decision": "quality_warning_requires_review" if fp16_quality_warning else "preserved_within_observed_metrics",
        "int8_quality_decision": "quality_warning_requires_mixed_precision_or_QAT" if int8_quality_warning else "no_obvious_quality_warning",
        "int8_speed_decision": "faster_than_fp32" if f(int8, "inference_images_per_s") > f(fp32, "inference_images_per_s") else "not_faster_than_fp32",
        "standard_fid_computed": True,
        "upstream_manifests": upstream_manifests,
        "upstream_checks": upstream_checks,
    }
    (output_dir / "task3_manifest.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    # Close the whole Deployment_Optimization chain with one top-level audit
    # record. Missing Task 1/2 manifests remain explicit until those stages
    # have actually been run; this script must not infer completion from code.
    # input_dir is .../Deployment_Optimization_Results/03_Quantization/03D_Evaluation;
    # parents[1] is the sibling-task root .../Deployment_Optimization_Results.
    result_root = input_dir.parents[1]
    task_manifests = {
        "task1": result_root / "01_ONNX_Fusion" / "task1_manifest.json",
        "task2": result_root / "02_Engine_Benchmark" / "02E_Report" / "task2_manifest.json",
        "task3": output_dir / "task3_manifest.json",
    }
    task_status = {}
    for name, path in task_manifests.items():
        payload = read_json_if_exists(path)
        task_status[name] = {
            "path": str(path),
            "status": payload.get("status", "missing") if payload else "missing",
        }
    whole_task_complete = all(item["status"] == "complete" for item in task_status.values())
    (result_root / "deployment_optimization_manifest.json").write_text(
        json.dumps(
            {
                "study": "Deployment_Optimization",
                "status": "complete" if whole_task_complete else "incomplete",
                "tasks": task_status,
                "note": "This top-level status is evidence-driven; it becomes complete only after Task 1, Task 2 and Task 3 manifests all report complete.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    audit_status = "complete" if whole_task_complete else "incomplete"
    audit_lines = [
        "# Deployment Optimization 全任务最终审计",
        "",
        f"总体状态：**{audit_status}**。该结论只读取已经落盘的任务 manifest，不根据脚本是否存在推断完成。",
        "",
        "| 任务 | 对应实验 | 证据入口 | 状态 |",
        "|---|---|---|---|",
        f"| 任务一：ONNX 导出、检查、算子融合 | 01A → 01B → 01C | `01_ONNX_Fusion/task1_manifest.json` | {task_status['task1']['status']} |",
        f"| 任务二：ORT/TensorRT/OpenVINO 基准与瓶颈 | 02A → 02B → 02C → 02D/02F → 02E | `02_Engine_Benchmark/02E_Report/task2_manifest.json` | {task_status['task2']['status']} |",
        f"| 任务三：100 张校准集、FP16/INT8 PTQ、质量评估 | 03A → 03B → 03C → 03D → 03E | `03_Quantization/03E_Report/task3_manifest.json` | {task_status['task3']['status']} |",
        "",
        "## 写作顺序",
        "",
        "1. 先写统一协议：模型、输入 latent、batch、硬件、软件版本、随机种子和文件哈希。",
        "2. 按任务一、任务二、任务三分别引用对应 manifest 和 CSV；每个任务先给结果表，再给证据解释。",
        "3. 任务一重点写 generator.onnx 的 checker、原始/融合节点数、数值等价和单算子耗时；`not_passed` 速度门槛是实验结论，不等于脚本失败。",
        "4. 任务二重点写三引擎 FP32/FP16 的端到端延迟、吞吐量、内存口径、Chrome Trace/层级 Top-3 及图层优化建议。",
        "5. 任务三同时报告 Legacy FID 和 Standard FID；用模糊率、Laplacian、LPIPS、逐像素误差以及 Haar LL/LH/HL/HH 证据解释 INT8 破坏，最后给出是否进入混合精度或 QAT。",
        "",
        "## 审计限制",
        "",
        "- 任务二 ORT 若只有 operator summary 而没有原始 Chrome Trace，应在报告中明确记录，不得写成 Trace 已保留。",
        "- 任务三的 100 张真实动漫头像是校准/审计参考集；Generator 的激活校准输入仍是 `latent_calibration.npy`，这是由模型输入边界决定的。",
        "- Standard FID 必须在 03D 实际运行并写入 `fid_standard_inception_v3`；旧版只含 Legacy FID 的结果不能通过 03E。",
    ]
    (result_root / "Deployment_Optimization_Final_Audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    make_charts(rows, errors, output_dir)
    contact = resolve_downloaded_file(input_dir, "quality_contact_sheet.png")
    if contact.exists():
        shutil.copy2(contact, output_dir / "quality_contact_sheet.png")
    # Preserve the machine-readable source tables in the final report package.
    shutil.copy2(metrics_path, output_dir / "fp32_fp16_int8_metrics.csv")
    shutil.copy2(errors_path, output_dir / "quantization_error.csv")
    shutil.copy2(frequency_path, output_dir / "frequency_band_error.csv")
    feature_manifest_path = resolve_downloaded_file(input_dir, "feature_manifest.csv")
    if feature_manifest_path.exists():
        shutil.copy2(feature_manifest_path, output_dir / "feature_manifest.csv")
    if standard_stats_path.exists():
        shutil.copy2(standard_stats_path, output_dir / "real_fid_standard_stats.npz")
    legacy_stats_path = resolve_downloaded_file(input_dir, "real_fid_legacy_stats.npz")
    if legacy_stats_path.exists():
        shutil.copy2(legacy_stats_path, output_dir / "real_fid_legacy_stats.npz")
    shutil.copy2(manifest_path, output_dir / "evaluation_manifest.json")
    lines = [
        "# Task 3 — FP32 / FP16 / INT8 量化评估报告",
        "",
        "> 本报告基于 03D 的实际输出，不把 engine 文件大小当作生成质量；同时报告项目历史 Legacy FID 与标准公开口径 FID，二者不混用。",
        "",
        "## 结果摘要",
        "",
        "| 精度 | Legacy FID | Standard FID | 模糊率 | Laplacian 均值 | LPIPS 多样性 | LPIPS fake-real | 吞吐量 img/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(f"| {r['precision']} | {f(r, 'fid_legacy_inception_v3'):.4f} | {f(r, 'fid_standard_inception_v3'):.4f} | {f(r, 'fake_blur_rate'):.4f} | {f(r, 'fake_laplacian_mean'):.2f} | {f(r, 'lpips_alex_diversity'):.4f} | {f(r, 'lpips_alex_fake_real'):.4f} | {f(r, 'inference_images_per_s'):.2f} |")
    lines.extend([
        "",
        "## 判定",
        "",
        f"- FP16：{decision['fp16_quality_decision']}。",
        f"- INT8：{decision['int8_quality_decision']}。",
        f"- INT8 吞吐结论：{decision['int8_speed_decision']}。",
        "- INT8 质量损失若持续存在，后续优先考虑混合精度或 QAT，而不是直接部署全 INT8。",
        "",
        "## 可视化",
        "",
        "![quality and performance](task3_quality_performance.svg)",
        "",
        "![quantization error](task3_quantization_error.svg)",
        "",
        "## INT8 frequency-band evidence",
        "",
        "- The machine-readable Haar LL/LH/HL/HH comparison is `frequency_band_error.csv`.",
        f"- INT8 LL MAE: {frequency_analysis['ll_mae_01'] if frequency_analysis['ll_mae_01'] is not None else 'n/a'}; high-frequency (LH/HL/HH) mean MAE: {frequency_analysis['high_frequency_mean_mae_01'] if frequency_analysis['high_frequency_mean_mae_01'] is not None else 'n/a'}; high/LL ratio: {frequency_analysis['high_to_ll_mae_ratio'] if frequency_analysis['high_to_ll_mae_ratio'] is not None else 'n/a'}.",
        f"- Diagnosis: {frequency_analysis['conclusion']}. {frequency_analysis['explanation']}",
        "- If the diagnosis is not proven, do not claim that INT8 degradation is caused by high-frequency truncation; inspect calibration range coverage, per-tensor versus per-channel scales, ConvTranspose activations and Tanh saturation.",
        "",
        "## 复现边界",
        "",
        "- 三种 engine 使用同一 latent_eval；",
        "- 真实图像用于 FID 和质量参考；",
        "- ORT/OpenVINO 与 TensorRT 的内存口径不能直接混称为单模型显存；",
        "- Standard FID 使用 pytorch-fid 的 Inception-v3 pool3/2048 维协议；Legacy FID 仅用于兼容项目历史结果。",
    ])
    (output_dir / "task3_quantization_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    archive = output_dir.parent / "Task3_Quantization_Report.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=f"03E_Report/{path.relative_to(output_dir).as_posix()}")
    print(f"[done] report={output_dir}")
    print(f"[zip] {archive}")


if __name__ == "__main__":
    main()
