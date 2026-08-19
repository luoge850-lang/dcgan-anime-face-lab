"""Standalone soak test for the already-running 06A HTTP service.

Paste this entire file into one Kaggle cell and run it. It does not require
the staged-load runner. The service from 06A must already be alive at
http://127.0.0.1:8000.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT = "/kaggle/working/dcgan_output/Deployment_Optimization_Results/06_Service_Stress"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone 06C soak test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup-seconds", type=int, default=120)
    parser.add_argument("--soak-seconds", type=int, default=1800)
    parser.add_argument("--monitor-interval-seconds", type=float, default=5.0)
    parser.add_argument("--spawn-rate", type=int, default=8)
    parser.add_argument("--failure-threshold", type=float, default=0.05)
    parser.add_argument("--abort-p99-ms", type=float, default=5000.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--skip-dependency-install", action="store_true")
    parsed, ignored = parser.parse_known_args(argv)
    if ignored:
        print(f"[06C-soak] ignored notebook/kernel arguments: {ignored}", flush=True)
    if parsed.concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if parsed.warmup_seconds <= 0 or parsed.soak_seconds <= 0:
        raise ValueError("warmup and soak durations must be positive")
    if parsed.monitor_interval_seconds <= 0:
        raise ValueError("monitor interval must be positive")
    return parsed


def install_dependencies(skip: bool) -> None:
    requirements = [
        ("locust", "locust"),
        ("psutil", "psutil"),
        ("pynvml", "nvidia-ml-py"),
        ("requests", "requests"),
    ]
    missing = [package for module, package in requirements if importlib.util.find_spec(module) is None]
    if missing:
        if skip:
            raise RuntimeError(f"Missing dependencies: {missing}")
        print(f"[06C-soak] installing: {missing}", flush=True)
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_state(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def make_locust_file(path: Path) -> None:
    path.write_text(
        '''
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
''',
        encoding="utf-8",
    )


def normalize_key(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def cell(row: dict[str, str], key: str, default: str = "") -> str:
    wanted = normalize_key(key)
    for actual, raw in row.items():
        if normalize_key(actual) == wanted:
            return raw
    return default


def number(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(cell(row, key, str(default)))
    except (TypeError, ValueError):
        return default


def read_locust_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Locust stats missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Locust stats empty: {path}")
    aggregate = next((row for row in rows if cell(row, "Name").lower() == "aggregated"), rows[-1])
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


def run_phase(
    *,
    phase: str,
    duration: int,
    args: argparse.Namespace,
    locust_file: Path,
    run_dir: Path,
) -> dict[str, Any]:
    prefix = run_dir / phase
    stats_path = Path(f"{prefix}_stats.csv")
    log_path = prefix.with_suffix(".locust.log")
    spawn_rate = max(1, min(args.concurrency, args.spawn_rate))
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
        str(args.concurrency),
        "-r",
        str(spawn_rate),
        "--run-time",
        f"{duration}s",
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
    elif stats["failure_rate"] > args.failure_threshold:
        status = "failure_threshold_exceeded"
    elif stats["p99_ms"] > args.abort_p99_ms:
        status = "latency_abort_not_hard_crash"
    else:
        status = "passed"
    result = {
        "phase": phase,
        "concurrency": args.concurrency,
        "duration_seconds": duration,
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
    print(
        f"[06C-soak] {phase} status={status} requests={stats['requests']} "
        f"failures={stats['failures']} p99={stats['p99_ms']:.2f}ms "
        f"rps={stats['rps']:.2f}",
        flush=True,
    )
    return result


def monitor_loop(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    state_path: Path,
    service_pid: int,
    stop_event: threading.Event,
) -> None:
    import psutil
    import pynvml

    fields = [
        "timestamp_utc", "elapsed_seconds", "phase", "concurrency", "phase_status",
        "service_pid", "service_alive", "health_ok", "health_code", "health_detail",
        "gpu_index", "gpu_name", "gpu_memory_used_mb", "gpu_memory_total_mb",
        "gpu_memory_percent", "gpu_sm_percent", "service_rss_mb", "service_vms_mb",
        "service_cpu_percent", "system_memory_used_mb", "system_memory_percent",
    ]
    process = psutil.Process(service_pid)
    process.cpu_percent(None)
    monitor_path = run_dir / "system_monitor_5s.csv"
    rows: list[dict[str, Any]] = []
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8", errors="replace")
        started = time.perf_counter()
        with monitor_path.open("w", newline="", encoding="utf-8") as csv_handle:
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
                if state_path.is_file():
                    try:
                        state = json.loads(state_path.read_text(encoding="utf-8"))
                    except Exception as exc:
                        state = {"status": f"state_error:{exc!r}"}
                row = {
                    "timestamp_utc": now_utc(),
                    "elapsed_seconds": elapsed,
                    "phase": state.get("phase", ""),
                    "concurrency": state.get("concurrency", args.concurrency),
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
                    f"[06C-monitor] phase={row['phase']} gpu={row['gpu_memory_used_mb']:.1f}MB "
                    f"sm={row['gpu_sm_percent']:.1f}% rss={row['service_rss_mb']:.1f}MB "
                    f"system={row['system_memory_percent']:.1f}% health={row['health_ok']}",
                    flush=True,
                )
                stop_event.wait(args.monitor_interval_seconds)
    finally:
        pynvml.nvmlShutdown()
    memory_values = [float(row["gpu_memory_used_mb"]) for row in rows]
    rss_values = [float(row["service_rss_mb"]) for row in rows]
    write_json(
        run_dir / "system_monitor_summary.json",
        {
            "task": "Task6C_Standalone_Soak_Monitor",
            "samples": len(rows),
            "service_pid": service_pid,
            "gpu_name": gpu_name,
            "interval_seconds": args.monitor_interval_seconds,
            "gpu_memory_peak_mb": max(memory_values) if memory_values else None,
            "rss_peak_mb": max(rss_values) if rss_values else None,
            "monitor_csv": str(monitor_path),
        },
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    install_dependencies(args.skip_dependency_install)
    service_ok, service_info = health(args.base_url)
    if not service_ok:
        raise RuntimeError(f"06A service is not healthy: {service_info}")
    try:
        service_pid = int(service_info["engine"]["pid"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("/health response does not expose engine.pid; cannot monitor service RSS") from exc

    run_id = datetime.now().strftime("Run_Soak_%Y%m%d_%H%M%S")
    root = Path(args.output_dir).expanduser().resolve() if args.output_dir else Path(DEFAULT_OUTPUT)
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    state_path = run_dir / "stage_state.json"
    runtime_root = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else run_dir
    locust_file = runtime_root / "06C_soak_locustfile_runtime.py"
    make_locust_file(locust_file)
    stop_event = threading.Event()
    monitor = threading.Thread(
        target=monitor_loop,
        kwargs={
            "args": args,
            "run_dir": run_dir,
            "state_path": state_path,
            "service_pid": service_pid,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    monitor.start()
    results: list[dict[str, Any]] = []
    try:
        write_state(state_path, {"phase": "soak_warmup", "status": "running", "concurrency": args.concurrency})
        warmup = run_phase(
            phase=f"soak_warmup_u{args.concurrency:04d}",
            duration=args.warmup_seconds,
            args=args,
            locust_file=locust_file,
            run_dir=run_dir,
        )
        results.append(warmup)
        if warmup["status"] == "passed":
            write_state(state_path, {"phase": "soak_steady", "status": "running", "concurrency": args.concurrency})
            steady = run_phase(
                phase=f"soak_steady_u{args.concurrency:04d}",
                duration=args.soak_seconds,
                args=args,
                locust_file=locust_file,
                run_dir=run_dir,
            )
            results.append(steady)
        else:
            write_state(state_path, {"phase": "soak_stopped_after_warmup", "status": warmup["status"], "concurrency": args.concurrency})
    finally:
        write_state(state_path, {"phase": "soak_finished", "status": results[-1]["status"] if results else "no_result", "concurrency": args.concurrency})
        stop_event.set()
        monitor.join(timeout=max(15.0, args.monitor_interval_seconds * 3))

    with (run_dir / "soak_results.csv").open("w", newline="", encoding="utf-8") as handle:
        if results:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)
    overall_status = "passed" if results and all(row["status"] == "passed" for row in results) else "stopped_or_failed"
    manifest = {
        "task": "Task6C_Standalone_Soak",
        "status": overall_status,
        "run_id": run_id,
        "base_url": args.base_url,
        "service_before": service_info,
        "config": {
            "concurrency": args.concurrency,
            "warmup_seconds": args.warmup_seconds,
            "soak_seconds": args.soak_seconds,
            "monitor_interval_seconds": args.monitor_interval_seconds,
            "failure_threshold": args.failure_threshold,
            "abort_p99_ms": args.abort_p99_ms,
        },
        "results": results,
        "outputs": [path.name for path in sorted(run_dir.iterdir()) if path.is_file()],
        "interpretation": {
            "rss": "service process RSS; compare post-warmup head and tail means",
            "hard_failure": "service process failure, unhealthy endpoint, Locust failure, or request failure",
            "latency_abort": "P99-only abort is overload evidence, not by itself a GPU hardware crash",
        },
        "generated_at": now_utc(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    archive = shutil.make_archive(str(run_dir), "zip", root_dir=run_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print(f"[06C-soak] output_dir={run_dir}", flush=True)
    print(f"[06C-soak] archive={archive}", flush=True)


if __name__ == "__main__":
    main()
