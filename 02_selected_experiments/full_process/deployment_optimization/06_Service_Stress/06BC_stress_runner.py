"""Unified 06B/06C runner for Kaggle.

This is the only stress-test script that needs to be pasted into a Kaggle
cell. It internally creates the Locust user file, runs either:

  --mode staged : increase concurrency until a hard failure or the configured
                  upper bound; record P99, RPS, failures, and health.
  --mode soak   : run a fixed safe concurrency for a warm-up and a long steady
                  period while sampling GPU memory, SM, RSS, and health.

06A must already be serving http://127.0.0.1:8000. This script cannot run a
Tesla T4 stress test on a CPU-only local machine.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


BASE_OUTPUT = "/kaggle/working/dcgan_output/Deployment_Optimization_Results/06_Service_Stress"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified 06B/06C stress and soak runner")
    parser.add_argument("--mode", choices=("staged", "soak"), default=os.getenv("STRESS_MODE", "staged"))
    parser.add_argument("--base-url", default=os.getenv("SERVICE_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--output-dir", default=os.getenv("STRESS_OUTPUT_DIR", ""))
    parser.add_argument("--stages", default=os.getenv("STRESS_STAGES", "1,2,4,8,16,32,48,64,80,96,128"))
    parser.add_argument("--stage-seconds", type=int, default=int(os.getenv("STAGE_SECONDS", "30")))
    parser.add_argument("--cooldown-seconds", type=int, default=int(os.getenv("COOLDOWN_SECONDS", "15")))
    parser.add_argument("--soak-concurrency", type=int, default=int(os.getenv("SOAK_CONCURRENCY", "16")))
    parser.add_argument("--soak-warmup-seconds", type=int, default=int(os.getenv("SOAK_WARMUP_SECONDS", "120")))
    parser.add_argument("--soak-seconds", type=int, default=int(os.getenv("SOAK_SECONDS", "1800")))
    parser.add_argument("--monitor-interval-seconds", type=float, default=5.0)
    parser.add_argument("--spawn-rate", type=int, default=8)
    parser.add_argument("--stop-failure-rate", type=float, default=0.05)
    parser.add_argument(
        "--abort-p99-ms",
        type=float,
        default=5000.0,
        help="Abort an overloaded stage, but do not label it a hardware crash",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-dependency-install", action="store_true")
    args, ignored = parser.parse_known_args(argv)
    if ignored:
        print(f"[06BC] ignored notebook/kernel arguments: {ignored}", flush=True)
    if args.stage_seconds <= 0 or args.cooldown_seconds < 0:
        raise ValueError("stage-seconds must be positive and cooldown-seconds cannot be negative")
    if args.soak_warmup_seconds <= 0 or args.soak_seconds <= 0:
        raise ValueError("soak durations must be positive")
    if args.monitor_interval_seconds <= 0:
        raise ValueError("monitor interval must be positive")
    try:
        args.stage_values = [int(item.strip()) for item in args.stages.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"Invalid stages: {args.stages!r}") from exc
    if not args.stage_values or any(value <= 0 for value in args.stage_values):
        raise ValueError("stages must contain positive integers")
    return args


def install_missing(skip: bool) -> None:
    requirements = [("locust", "locust"), ("psutil", "psutil"), ("pynvml", "nvidia-ml-py"), ("requests", "requests")]
    missing = [package for module, package in requirements if importlib.util.find_spec(module) is None]
    if missing:
        if skip:
            raise RuntimeError(f"Missing dependencies: {missing}")
        print(f"[06BC] installing: {missing}", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        importlib.invalidate_caches()


def health(base_url: str) -> tuple[bool, dict[str, Any]]:
    import requests

    try:
        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=10)
        payload = response.json()
        return response.status_code == 200 and payload.get("status") == "ok", {
            "status_code": response.status_code,
            "status": payload.get("status"),
            "engine": payload.get("engine", {}),
        }
    except Exception as exc:
        return False, {"error": repr(exc)}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_stage_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def make_locust_file(path: Path) -> None:
    source = '''
import itertools
import time
from locust import HttpUser, task, between

SEEDS = itertools.count(int(time.time()) % 1000000)

class GeneratorUser(HttpUser):
    wait_time = between(0.0, 0.0)

    @task
    def generate(self):
        seed = next(SEEDS)
        with self.client.post(
            "/generate",
            json={"seed": seed},
            name="POST /generate",
            timeout=30,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
            elif not response.headers.get("content-type", "").startswith("image/png"):
                response.failure("response is not image/png")
            elif not response.content:
                response.failure("empty response")
            else:
                response.success()
'''
    path.write_text(source, encoding="utf-8")


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def value(row: dict[str, str], key: str, default: str = "") -> str:
    wanted = normalize_key(key)
    for actual, raw in row.items():
        if normalize_key(actual) == wanted:
            return raw
    return default


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(value(row, key, str(default)))
    except (TypeError, ValueError):
        return default


def read_locust_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Locust stats missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Locust stats empty: {path}")
    aggregate = next((row for row in rows if value(row, "Name").lower() == "aggregated"), rows[-1])
    requests_count = int(number(aggregate, "Request Count"))
    failures = int(number(aggregate, "Failure Count"))
    return {
        "requests": requests_count,
        "failures": failures,
        "failure_rate": failures / requests_count if requests_count else 1.0,
        "p50_ms": number(aggregate, "50%"),
        "p95_ms": number(aggregate, "95%"),
        "p99_ms": number(aggregate, "99%"),
        "mean_ms": number(aggregate, "Average Response Time"),
        "min_ms": number(aggregate, "Min Response Time"),
        "max_ms": number(aggregate, "Max Response Time"),
        "rps": number(aggregate, "Requests/s"),
    }


def run_locust_phase(
    *,
    phase_label: str,
    users: int,
    duration_seconds: int,
    args: argparse.Namespace,
    locust_file: Path,
    output_dir: Path,
) -> dict[str, Any]:
    prefix = output_dir / phase_label
    log_path = prefix.with_suffix(".locust.log")
    stats_path = Path(f"{prefix}_stats.csv")
    spawn_rate = max(1, min(users, args.spawn_rate))
    command = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(locust_file),
        "--headless",
        "--host",
        args.base_url.rstrip("/"),
        "-u",
        str(users),
        "-r",
        str(spawn_rate),
        "--run-time",
        f"{duration_seconds}s",
        "--csv",
        str(prefix),
        "--csv-full-history",
        "--only-summary",
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    stats = read_locust_stats(stats_path)
    service_ok, health_info = health(args.base_url)
    if completed.returncode != 0:
        status = "locust_process_failed"
    elif stats["requests"] == 0:
        status = "no_requests"
    elif not service_ok:
        status = "service_unhealthy"
    elif stats["failure_rate"] > args.stop_failure_rate:
        status = "failure_threshold_exceeded"
    elif stats["p99_ms"] > args.abort_p99_ms:
        status = "latency_abort_not_hard_crash"
    else:
        status = "passed"
    return {
        "phase": phase_label,
        "concurrency": users,
        "duration_seconds": duration_seconds,
        "elapsed_seconds": elapsed,
        "spawn_rate": spawn_rate,
        "status": status,
        "locust_returncode": completed.returncode,
        "health_ok": service_ok,
        "health_detail": health_info,
        "stats_file": str(stats_path),
        "log_file": str(log_path),
        **stats,
    }


def monitor_loop(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    stage_state_path: Path,
    service_pid: int,
    stop_event: threading.Event,
) -> None:
    import psutil
    import pynvml

    output_path = run_dir / "system_monitor_5s.csv"
    summary_path = run_dir / "system_monitor_summary.json"
    fields = [
        "timestamp_utc", "elapsed_seconds", "phase", "concurrency", "phase_status",
        "service_pid", "service_alive", "health_ok", "health_code", "health_detail",
        "gpu_index", "gpu_name", "gpu_memory_used_mb", "gpu_memory_total_mb",
        "gpu_memory_percent", "gpu_sm_percent", "service_rss_mb", "service_vms_mb",
        "service_cpu_percent", "system_memory_used_mb", "system_memory_percent",
    ]
    process = psutil.Process(service_pid)
    process.cpu_percent(None)
    rows: list[dict[str, Any]] = []
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")
        started = time.perf_counter()
        with output_path.open("w", newline="", encoding="utf-8") as csv_handle:
            writer = csv.DictWriter(csv_handle, fieldnames=fields)
            writer.writeheader()
            csv_handle.flush()
            while not stop_event.is_set():
                elapsed = time.perf_counter() - started
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                alive = process.is_running()
                if alive:
                    try:
                        info = process.memory_info()
                        rss_mb = info.rss / 1024**2
                        vms_mb = info.vms / 1024**2
                        cpu_percent = process.cpu_percent(None)
                    except Exception:
                        alive = False
                        rss_mb = vms_mb = cpu_percent = 0.0
                else:
                    rss_mb = vms_mb = cpu_percent = 0.0
                health_ok, health_info = health(args.base_url)
                system_memory = psutil.virtual_memory()
                state: dict[str, Any] = {}
                if stage_state_path.is_file():
                    try:
                        state = json.loads(stage_state_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        state = {"phase_status": f"state_error:{exc!r}"}
                row = {
                    "timestamp_utc": utc_now(),
                    "elapsed_seconds": elapsed,
                    "phase": state.get("phase", ""),
                    "concurrency": state.get("concurrency", ""),
                    "phase_status": state.get("status", "not_started"),
                    "service_pid": service_pid,
                    "service_alive": alive,
                    "health_ok": health_ok,
                    "health_code": health_info.get("status_code", ""),
                    "health_detail": health_info.get("status", health_info.get("error", "")),
                    "gpu_index": args.gpu,
                    "gpu_name": gpu_name,
                    "gpu_memory_used_mb": memory.used / 1024**2,
                    "gpu_memory_total_mb": memory.total / 1024**2,
                    "gpu_memory_percent": memory.used / memory.total * 100.0,
                    "gpu_sm_percent": utilization.gpu,
                    "service_rss_mb": rss_mb,
                    "service_vms_mb": vms_mb,
                    "service_cpu_percent": cpu_percent,
                    "system_memory_used_mb": system_memory.used / 1024**2,
                    "system_memory_percent": system_memory.percent,
                }
                writer.writerow(row)
                csv_handle.flush()
                rows.append(row)
                print(
                    f"[06C] phase={row['phase']} conc={row['concurrency']} "
                    f"gpu={row['gpu_memory_used_mb']:.1f}MB "
                    f"sm={row['gpu_sm_percent']:.1f}% rss={row['service_rss_mb']:.1f}MB "
                    f"health={row['health_ok']}",
                    flush=True,
                )
                stop_event.wait(args.monitor_interval_seconds)
    finally:
        pynvml.nvmlShutdown()
    memory_values = [float(row["gpu_memory_used_mb"]) for row in rows]
    rss_values = [float(row["service_rss_mb"]) for row in rows]
    summary = {
        "task": "Task6C_System_Monitor",
        "samples": len(rows),
        "service_pid": service_pid,
        "gpu_name": gpu_name,
        "interval_seconds": args.monitor_interval_seconds,
        "gpu_memory_peak_mb": max(memory_values) if memory_values else None,
        "rss_peak_mb": max(rss_values) if rss_values else None,
        "monitor_csv": str(output_path),
    }
    write_json(summary_path, summary)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    install_missing(args.skip_dependency_install)
    service_ok, service_info = health(args.base_url)
    if not service_ok:
        raise RuntimeError(f"06A service is not healthy: {service_info}")
    service_pid = int(service_info["engine"]["pid"])
    run_id = datetime.now().strftime("Run_%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(BASE_OUTPUT) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stage_state_path = run_dir / "stage_state.json"
    locust_file = Path("/kaggle/working/06BC_locustfile_runtime.py")
    make_locust_file(locust_file)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_loop,
        kwargs={
            "args": args,
            "run_dir": run_dir,
            "stage_state_path": stage_state_path,
            "service_pid": service_pid,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    monitor.start()
    time.sleep(max(2.0, args.monitor_interval_seconds))

    results: list[dict[str, Any]] = []
    try:
        if args.mode == "staged":
            for index, users in enumerate(args.stage_values, start=1):
                write_stage_state(stage_state_path, {
                    "phase": f"stage_{index:02d}",
                    "status": "running",
                    "stage_index": index,
                    "concurrency": users,
                    "started_at": utc_now(),
                })
                result = run_locust_phase(
                    phase_label=f"stage_{index:02d}_u{users:04d}",
                    users=users,
                    duration_seconds=args.stage_seconds,
                    args=args,
                    locust_file=locust_file,
                    output_dir=run_dir,
                )
                results.append(result)
                write_stage_state(stage_state_path, {
                    "phase": result["phase"],
                    "status": result["status"],
                    "stage_index": index,
                    "concurrency": users,
                    "finished_at": utc_now(),
                    "result": result,
                })
                print(
                    f"[06B] {result['phase']} status={result['status']} "
                    f"requests={result['requests']} failures={result['failures']} "
                    f"p99={result['p99_ms']:.2f}ms rps={result['rps']:.2f}",
                    flush=True,
                )
                if result["status"] != "passed":
                    print("[06BC] stopping staged run at the first failed/aborted stage", flush=True)
                    break
                if index != len(args.stage_values) and args.cooldown_seconds:
                    write_stage_state(stage_state_path, {
                        "phase": f"cooldown_after_stage_{index:02d}",
                        "status": "cooldown",
                        "stage_index": index,
                        "concurrency": users,
                    })
                    time.sleep(args.cooldown_seconds)
        else:
            users = args.soak_concurrency
            write_stage_state(stage_state_path, {
                "phase": "soak_warmup",
                "status": "running",
                "concurrency": users,
                "started_at": utc_now(),
            })
            warmup = run_locust_phase(
                phase_label=f"soak_warmup_u{users:04d}",
                users=users,
                duration_seconds=args.soak_warmup_seconds,
                args=args,
                locust_file=locust_file,
                output_dir=run_dir,
            )
            results.append(warmup)
            if warmup["status"] == "passed":
                write_stage_state(stage_state_path, {
                    "phase": "soak_steady",
                    "status": "running",
                    "concurrency": users,
                    "started_at": utc_now(),
                })
                steady = run_locust_phase(
                    phase_label=f"soak_steady_u{users:04d}",
                    users=users,
                    duration_seconds=args.soak_seconds,
                    args=args,
                    locust_file=locust_file,
                    output_dir=run_dir,
                )
                results.append(steady)
            write_stage_state(stage_state_path, {
                "phase": "soak_finished",
                "status": results[-1]["status"],
                "concurrency": users,
                "finished_at": utc_now(),
            })
    finally:
        stop_event.set()
        monitor.join(timeout=max(15.0, args.monitor_interval_seconds * 3))

    if args.mode == "staged":
        with (run_dir / "stage_results.csv").open("w", newline="", encoding="utf-8") as handle:
            if results:
                writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
    else:
        with (run_dir / "soak_results.csv").open("w", newline="", encoding="utf-8") as handle:
            if results:
                writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()), extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)

    overall_status = "passed" if results and all(item["status"] == "passed" for item in results) else "stopped_or_failed"
    manifest = {
        "task": "Task6_06BC_Stress_and_Soak",
        "status": overall_status,
        "run_id": run_id,
        "mode": args.mode,
        "base_url": args.base_url,
        "service_before": service_info,
        "config": {
            "stages": args.stage_values,
            "stage_seconds": args.stage_seconds,
            "cooldown_seconds": args.cooldown_seconds,
            "soak_concurrency": args.soak_concurrency,
            "soak_warmup_seconds": args.soak_warmup_seconds,
            "soak_seconds": args.soak_seconds,
            "monitor_interval_seconds": args.monitor_interval_seconds,
            "stop_failure_rate": args.stop_failure_rate,
            "abort_p99_ms": args.abort_p99_ms,
        },
        "results": results,
        "outputs": [path.name for path in sorted(run_dir.iterdir()) if path.is_file()],
        "interpretation": {
            "hard_crash": "first stage with process failure, service unhealthy, or request failure; latency-only abort is not a hardware crash",
            "soak_leak": "must be judged from post-warmup RSS/GPU time series, not from one endpoint",
        },
    }
    write_json(run_dir / "run_manifest.json", manifest)
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"[06BC] output_dir={run_dir}", flush=True)
    print(f"[06BC] archive={archive}", flush=True)


if __name__ == "__main__":
    main()
