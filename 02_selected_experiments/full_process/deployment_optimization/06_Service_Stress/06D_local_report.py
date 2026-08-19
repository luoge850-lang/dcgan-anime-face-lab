"""06D local-only report builder.

This script does not require TensorRT, CUDA, Locust, or a GPU. It reads a
downloaded 06BC run directory and writes tables, plots, a Markdown report and
an archive. The GPU stress test itself must still be executed on Kaggle.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local Task 3 report from downloaded 06BC data")
    parser.add_argument("--run-root", required=True, help="Downloaded 06BC run directory")
    parser.add_argument("--output-dir", default="", help="Report directory; defaults to <run-root>/06D_Report")
    parser.add_argument(
        "--result-kind",
        choices=("auto", "staged", "soak"),
        default="auto",
        help="Select staged or soak CSV; auto detects the file that exists",
    )
    parsed, ignored = parser.parse_known_args(argv)
    if ignored:
        print(f"[06D] ignored notebook/kernel arguments: {ignored}", flush=True)
    return parsed


def ensure_matplotlib() -> None:
    if importlib.util.find_spec("matplotlib") is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "matplotlib"])
        importlib.invalidate_caches()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def i(value: Any) -> int | None:
    number = f(value)
    return int(number) if number is not None else None


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def monitor_rows_for(rows: list[dict[str, str]], concurrency: int, phase_status: str = "running") -> list[dict[str, str]]:
    result = []
    for row in rows:
        if i(row.get("concurrency")) == concurrency and row.get("phase_status") == phase_status:
            result.append(row)
    return result


def merged_rows(stage_rows: list[dict[str, str]], monitor_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in stage_rows:
        concurrency = i(row.get("concurrency"))
        if concurrency is None:
            continue
        resource = monitor_rows_for(monitor_rows, concurrency)
        memory = [f(x.get("gpu_memory_used_mb")) for x in resource]
        memory = [x for x in memory if x is not None]
        rss = [f(x.get("service_rss_mb")) for x in resource]
        rss = [x for x in rss if x is not None]
        sm = [f(x.get("gpu_sm_percent")) for x in resource]
        sm = [x for x in sm if x is not None]
        output.append({
            "phase": row.get("phase", ""),
            "concurrency": concurrency,
            "duration_seconds": f(row.get("duration_seconds")),
            "requests": i(row.get("requests")),
            "failures": i(row.get("failures")),
            "failure_rate": f(row.get("failure_rate")),
            "p50_ms": f(row.get("p50_ms")),
            "p95_ms": f(row.get("p95_ms")),
            "p99_ms": f(row.get("p99_ms")),
            "rps": f(row.get("rps")),
            "status": row.get("status", ""),
            "health_ok": row.get("health_ok", ""),
            "monitor_samples": len(resource),
            "gpu_memory_peak_mb": max(memory) if memory else None,
            "gpu_sm_peak_percent": max(sm) if sm else None,
            "gpu_sm_mean_percent": mean(sm),
            "rss_start_mb": rss[0] if rss else None,
            "rss_end_mb": rss[-1] if rss else None,
            "rss_peak_mb": max(rss) if rss else None,
            "rss_delta_mb": rss[-1] - rss[0] if len(rss) >= 2 else None,
        })
    return sorted(output, key=lambda x: x["concurrency"])


def classify(rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    hard = next(
        (row for row in rows if (row["failures"] or 0) > 0 or row["status"] != "passed" or str(row["health_ok"]).lower() != "true"),
        None,
    )
    soft = None
    if mode == "soak":
        return {
            "hard_crash": {"status": "observed", "stage": hard} if hard else {"status": "not_observed"},
            "soft_latency_knee": {"status": "not_applicable", "reason": "soak uses fixed concurrency"},
        }
    previous = None
    for row in rows:
        if previous and previous.get("p99_ms") and row.get("p99_ms"):
            ratio = row["p99_ms"] / previous["p99_ms"]
            prev_rps = previous.get("rps") or 0.0
            gain = ((row.get("rps") or 0.0) - prev_rps) / prev_rps if prev_rps else 0.0
            if row["p99_ms"] >= 100 and ratio >= 1.5 and gain <= 0.10:
                soft = {"concurrency": row["concurrency"], "p99_ms": row["p99_ms"], "criterion": "p99>=100ms, ratio>=1.5, throughput gain<=10%"}
                break
        previous = row
    return {
        "hard_crash": {"status": "observed", "stage": hard} if hard else {"status": "not_observed", "max_tested": rows[-1]["concurrency"] if rows else None},
        "soft_latency_knee": soft or {"status": "not_observed"},
    }


def soak_memory_check(monitor_rows: list[dict[str, str]]) -> dict[str, Any]:
    """Compare the beginning and end of the steady soak period.

    This is an operational leak signal, not a proof that no leak exists.
    The runner labels the steady phase as ``soak_steady`` in the monitor CSV.
    """
    steady = [row for row in monitor_rows if str(row.get("phase", "")).startswith("soak_steady")]
    if len(steady) < 4:
        steady = monitor_rows
    if len(steady) < 4:
        return {"status": "insufficient_samples", "samples": len(steady)}
    split = max(2, len(steady) // 5)
    head = steady[:split]
    tail = steady[-split:]

    def average(rows: list[dict[str, str]], key: str) -> float | None:
        values = [f(row.get(key)) for row in rows]
        values = [value for value in values if value is not None]
        return mean(values)

    rss_head = average(head, "service_rss_mb")
    rss_tail = average(tail, "service_rss_mb")
    gpu_head = average(head, "gpu_memory_used_mb")
    gpu_tail = average(tail, "gpu_memory_used_mb")

    def delta_percent(first: float | None, last: float | None) -> float | None:
        if first is None or last is None or first == 0:
            return None
        return (last - first) / first * 100.0

    rss_delta = delta_percent(rss_head, rss_tail)
    gpu_delta = delta_percent(gpu_head, gpu_tail)
    within_limit = (
        rss_delta is not None
        and gpu_delta is not None
        and rss_delta <= 5.0
        and gpu_delta <= 5.0
    )
    return {
        "status": "pass" if within_limit else "warning",
        "samples": len(steady),
        "rss_head_mean_mb": rss_head,
        "rss_tail_mean_mb": rss_tail,
        "rss_delta_percent": rss_delta,
        "gpu_head_mean_mb": gpu_head,
        "gpu_tail_mean_mb": gpu_tail,
        "gpu_delta_percent": gpu_delta,
        "criterion": "head-to-tail mean RSS and GPU memory increase <= 5%; operational signal only",
    }


def resolve_result_path(run_root: Path, result_kind: str) -> tuple[Path, str]:
    candidates: list[tuple[str, Path]] = []
    if result_kind in ("auto", "staged"):
        candidates.append(("staged", run_root / "stage_results.csv"))
        candidates.append(("staged", run_root / "06B" / "stage_results.csv"))
        candidates.extend(("staged", path) for path in sorted(run_root.glob("stage_results*.csv")))
        candidates.extend(("staged", path) for path in sorted((run_root / "06B").glob("stage_results*.csv")))
    if result_kind in ("auto", "soak"):
        candidates.append(("soak", run_root / "soak_results.csv"))
        candidates.append(("soak", run_root / "06B" / "soak_results.csv"))
        candidates.extend(("soak", path) for path in sorted(run_root.glob("soak_results*.csv")))
        candidates.extend(("soak", path) for path in sorted((run_root / "06B").glob("soak_results*.csv")))
    for mode, path in candidates:
        if path.is_file():
            return path, mode
    expected = " or ".join(str(path) for _, path in candidates)
    raise FileNotFoundError(f"No 06BC result CSV found. Expected: {expected}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plots(rows: list[dict[str, Any]], monitor_rows: list[dict[str, str]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    x = [row["concurrency"] for row in rows]
    p99 = [row.get("p99_ms") or 0.0 for row in rows]
    rps = [row.get("rps") or 0.0 for row in rows]
    memory = [row.get("gpu_memory_peak_mb") or 0.0 for row in rows]
    rss = [row.get("rss_peak_mb") or 0.0 for row in rows]
    sm = [row.get("gpu_sm_peak_percent") or 0.0 for row in rows]

    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(x, p99, memory, marker="o")
    axis.set_xlabel("Concurrency")
    axis.set_ylabel("P99 latency (ms)")
    axis.set_zlabel("Peak GPU memory (MB)")
    axis.set_title("Concurrency - P99 - GPU Memory")
    figure.tight_layout()
    figure.savefig(output / "concurrency_p99_memory_3d.png", dpi=180)
    plt.close(figure)

    figure, panels = plt.subplots(1, 2, figsize=(12, 4))
    panels[0].plot(x, p99, marker="o")
    panels[0].set(xlabel="Concurrency", ylabel="P99 (ms)")
    panels[0].grid(alpha=0.3)
    panels[1].plot(x, rps, marker="o", color="tab:green")
    panels[1].set(xlabel="Concurrency", ylabel="RPS")
    panels[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "latency_throughput.png", dpi=180)
    plt.close(figure)

    figure, panels = plt.subplots(1, 2, figsize=(12, 4))
    panels[0].plot(x, memory, marker="o", color="tab:blue")
    panels[0].set(xlabel="Concurrency", ylabel="Peak GPU memory (MB)")
    panels[0].grid(alpha=0.3)
    panels[1].plot(x, rss, marker="o", label="RSS", color="tab:orange")
    panels[1].plot(x, sm, marker="s", label="SM peak (%)", color="tab:red")
    panels[1].set(xlabel="Concurrency", ylabel="RSS / SM")
    panels[1].legend()
    panels[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "resource_by_concurrency.png", dpi=180)
    plt.close(figure)

    if monitor_rows:
        t = [f(row.get("elapsed_seconds")) or 0.0 for row in monitor_rows]
        m = [f(row.get("gpu_memory_used_mb")) or 0.0 for row in monitor_rows]
        r = [f(row.get("service_rss_mb")) or 0.0 for row in monitor_rows]
        s = [f(row.get("gpu_sm_percent")) or 0.0 for row in monitor_rows]
        system_memory = [f(row.get("system_memory_used_mb")) for row in monitor_rows]
        has_system_memory = any(value is not None for value in system_memory)
        system_memory = [value or 0.0 for value in system_memory]
        panel_count = 4 if has_system_memory else 3
        figure, panels = plt.subplots(panel_count, 1, figsize=(11, 10 if has_system_memory else 8), sharex=True)
        if panel_count == 1:
            panels = [panels]
        panels[0].plot(t, m)
        panels[0].set_ylabel("GPU MB")
        panels[1].plot(t, r, color="tab:orange")
        panels[1].set_ylabel("RSS MB")
        panels[2].plot(t, s, color="tab:red")
        panels[2].set_ylabel("SM %")
        if has_system_memory:
            panels[3].plot(t, system_memory, color="tab:purple")
            panels[3].set_ylabel("System used MB")
            panels[3].set_xlabel("Elapsed seconds")
        else:
            panels[2].set_xlabel("Elapsed seconds")
        for panel in panels:
            panel.grid(alpha=0.3)
        figure.tight_layout()
        figure.savefig(output / "monitor_time_series.png", dpi=180)
        plt.close(figure)


def main(argv: list[str] | None = None) -> None:
    parsed = args(argv)
    ensure_matplotlib()
    run_root = Path(parsed.run_root).expanduser().resolve()
    output = Path(parsed.output_dir).expanduser().resolve() if parsed.output_dir else run_root / "06D_Report"
    output.mkdir(parents=True, exist_ok=True)
    result_path, mode = resolve_result_path(run_root, parsed.result_kind)
    monitor_path = run_root / "system_monitor_5s.csv"
    if not monitor_path.is_file():
        monitor_path = run_root / "06C" / "system_monitor_5s.csv"
    if not monitor_path.is_file():
        monitor_candidates = sorted(run_root.glob("system_monitor_5s*.csv"))
        monitor_candidates.extend(sorted((run_root / "06C").glob("system_monitor_5s*.csv")))
        if monitor_candidates:
            monitor_path = monitor_candidates[0]
    result_rows = read_csv(result_path)
    monitor_rows = read_csv(monitor_path)
    merged = merged_rows(result_rows, monitor_rows)
    summary_name = "stage_resource_summary.csv" if mode == "staged" else "soak_resource_summary.csv"
    write_csv(output / summary_name, merged)
    classification = classify(merged, mode)
    soak_check = soak_memory_check(monitor_rows) if mode == "soak" else {"status": "not_applicable"}
    plots(merged, monitor_rows, output)
    report = {
        "task": "Task6_06D_Local_Report",
        "status": "completed",
        "run_root": str(run_root),
        "mode": mode,
        "inputs": {"result_csv": str(result_path), "monitor_csv": str(monitor_path)},
        "classification": classification,
        "soak_memory_check": soak_check,
        "outputs": [path.name for path in sorted(output.iterdir()) if path.is_file()],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "This report does not establish a GPU crash point; it summarizes the crash/upper-bound evidence produced by 06BC. Soak memory check is an operational signal, not a proof of no leak.",
    }
    (output / "service_stress_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# Task 3 Service Stress Report",
        "",
        f"Input: `{run_root}`",
        "",
        f"Mode: `{mode}`",
        f"Hard crash classification: `{classification['hard_crash']}`",
        f"Soft latency knee: `{classification['soft_latency_knee']}`",
        f"Soak memory check: `{soak_check}`",
        "",
        "| Concurrency | Requests | Failures | Failure rate | P99 ms | RPS | Peak GPU MB | Peak RSS MB | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in merged:
        markdown.append(
            f"| {row['concurrency']} | {row['requests'] or 0} | {row['failures'] or 0} | {(row['failure_rate'] or 0):.4f} | {(row['p99_ms'] or 0):.2f} | {(row['rps'] or 0):.2f} | {(row['gpu_memory_peak_mb'] or 0):.2f} | {(row['rss_peak_mb'] or 0):.2f} | {row['status']} |"
        )
    markdown.extend([
        "",
        "Generated figures:",
        "",
        "- `concurrency_p99_memory_3d.png`",
        "- `latency_throughput.png`",
        "- `resource_by_concurrency.png`",
        "- `monitor_time_series.png`",
    ])
    (output / "service_stress_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
