"""Kaggle one-cell runner for phase 07.

Paste this entire file into one Kaggle cell.  It needs only the existing
06A_service.py and one TensorRT engine in Kaggle Inputs.  Prometheus,
Alertmanager, Grafana and all YAML/JSON configuration are generated in the
Kaggle working directory; no 07 script or configuration upload is required.
"""
from __future__ import annotations

import argparse
import asyncio
import atexit
import concurrent.futures
import csv
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

OUT = Path(os.getenv("OBSERVABILITY_OUTPUT_DIR", "/kaggle/working/07_MLOps_Observability"))
HOST = "127.0.0.1"
SERVICE_PORT, PROM_PORT, AM_PORT, GRAFANA_PORT = 8000, 9090, 9093, 3000
SERVICE_SERVER = None
WEBHOOK_SERVER = None
PROCESSES = []
STOP = threading.Event()


def pip_install():
    for module, package in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("requests", "requests"),
                            ("psutil", "psutil"), ("prometheus_client", "prometheus-client"),
                            ("pynvml", "nvidia-ml-py"), ("playwright", "playwright")):
        try:
            importlib.import_module(module)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            importlib.invalidate_caches()


def import_file(path: Path):
    spec = importlib.util.spec_from_file_location("dcgan_06a_for_07", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inputs(args):
    root = Path("/kaggle/input")
    services = [Path(args.base_service)] if args.base_service else sorted(root.rglob("06A_service.py"))
    engines = [Path(args.engine)] if args.engine else sorted(root.rglob("generator_trt_qat_int8*.engine"))
    if len(services) != 1:
        raise RuntimeError(f"06A_service.py 数量不明确，请设置 BASE_SERVICE_PATH。候选：{services}")
    preferred = [p for p in engines if p.name == "generator_trt_qat_int8 (1).engine"]
    if len(preferred) == 1:
        engine = preferred[0]
    elif len(engines) == 1:
        engine = engines[0]
    else:
        raise RuntimeError(f"QAT Engine 数量不明确，请设置 ENGINE_PATH。候选：{engines}")
    if not services[0].is_file() or not engine.is_file():
        raise FileNotFoundError(f"输入不存在：service={services[0]}, engine={engine}")
    return services[0].resolve(), engine.resolve()


class Runtime:
    def __init__(self, base, engine_path, args):
        import torch
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，请在 Kaggle 打开 GPU")
        base_args = SimpleNamespace(tensorrt_version=args.trt, tensorrt_pip_package="tensorrt-cu12",
                                    skip_dependency_install=False, allow_version_mismatch=False)
        base.ensure_dependencies(base_args)
        trt = base.load_runtime(args.trt, False)
        self.base, self.engine_path = base, engine_path
        self.service = base.TensorRTGenerator(engine_path, trt, 0)
        self.lock, self.inflight, self.debug_queue = threading.Lock(), 0, 0
        self.registry = CollectorRegistry()
        self.http_count = Counter("dcgan_http_requests_total", "HTTP requests", ["path", "status"], registry=self.registry)
        self.http_latency = Histogram("dcgan_http_request_latency_seconds", "HTTP latency", registry=self.registry)
        self.trt_latency = Histogram("dcgan_trt_inference_latency_seconds", "TRT latency", registry=self.registry)
        self.queue = Gauge("dcgan_inference_queue_depth", "Queue depth", registry=self.registry)
        self.batch = Gauge("dcgan_actual_batch_size", "Actual batch size", registry=self.registry)
        self.gpu_mem = Gauge("dcgan_gpu_memory_used_bytes", "GPU memory", registry=self.registry)
        self.gpu_sm = Gauge("dcgan_gpu_sm_utilization_ratio", "GPU SM utilization", registry=self.registry)
        self.rss = Gauge("dcgan_process_resident_memory_bytes", "RSS", registry=self.registry)
        self.inflight_gauge = Gauge("dcgan_inflight_requests", "Inflight requests", registry=self.registry)
        self.batch.set(1)

    def metrics(self):
        import psutil
        memory, sm = None, None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2
            sm = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        except Exception:
            pass
        with self.lock:
            inflight, forced = self.inflight, self.debug_queue
        queue_depth = max(forced, max(0, inflight - 1))
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        # Keep Prometheus synchronized with the JSON/CSV resource snapshot.
        # Otherwise Grafana may show empty GPU/RSS panels while the CSV is valid.
        self.queue.set(queue_depth)
        self.batch.set(1)
        self.gpu_mem.set(0 if memory is None else memory * 1024**2)
        self.gpu_sm.set(0 if sm is None else sm / 100.0)
        self.rss.set(rss_mb * 1024**2)
        self.inflight_gauge.set(inflight)
        return {"queue_depth": queue_depth, "actual_batch_size": 1,
                "gpu_memory_used_mb": memory, "gpu_sm_utilization_percent": sm,
                "rss_mb": rss_mb, "inflight": inflight,
                "model_version": "qat-int8-v1"}

    def infer(self, seed):
        started = time.perf_counter()
        with self.lock:
            self.inflight += 1
            current, forced = self.inflight, self.debug_queue
        self.inflight_gauge.set(current)
        self.queue.set(max(forced, max(0, current - 1)))
        try:
            output, trt_ms = self.service.infer(int(seed))
            self.trt_latency.observe(trt_ms / 1000.0)
            return self.base.image_png(output), (time.perf_counter() - started) * 1000
        finally:
            with self.lock:
                self.inflight -= 1
                current, forced = self.inflight, self.debug_queue
            self.inflight_gauge.set(current)
            self.queue.set(max(forced, max(0, current - 1)))


def app_for(rt: Runtime):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    # The module uses postponed annotations.  Expose Request globally so
    # FastAPI resolves it as a request object rather than a query parameter.
    globals()["Request"] = Request
    app = FastAPI(title="DCGAN 07 Observable Service")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_version": "qat-int8-v1", "batcher": False,
                "actual_batch_size": 1, "engine": rt.engine_path.name}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(rt.registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/metrics.json")
    def metrics_json():
        return rt.metrics()

    @app.post("/generate")
    async def generate(request: Request):
        started = time.perf_counter()
        try:
            data = await request.json()
            image, _ = rt.infer(int(data.get("seed", 0)))
            rt.http_count.labels("/generate", "200").inc()
            rt.http_latency.observe(time.perf_counter() - started)
            return Response(image, media_type="image/png")
        except Exception as exc:
            rt.http_count.labels("/generate", "500").inc()
            rt.http_latency.observe(time.perf_counter() - started)
            return JSONResponse({"error": repr(exc)}, status_code=500)

    @app.post("/debug/queue")
    async def debug_queue(request: Request):
        data = await request.json()
        value = max(0, int(data.get("value", 0)))
        with rt.lock:
            rt.debug_queue = value
        rt.queue.set(value)
        return {"queue_depth": value}

    @app.post("/debug/clear")
    def debug_clear():
        with rt.lock:
            rt.debug_queue = 0
        rt.queue.set(0)
        return {"queue_depth": 0}

    return app


def start_webhook(events: Path):
    global WEBHOOK_SERVER
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
            if self.path in ("/alerts", "/webhook"):
                with events.open("a", encoding="utf-8") as handle:
                    handle.write(body.decode("utf-8", errors="replace") + "\n")
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"ok")
            else:
                self.send_response(404)
                self.end_headers()
        def log_message(self, *_):
            return
    WEBHOOK_SERVER = ThreadingHTTPServer((HOST, 9094), Handler)
    threading.Thread(target=WEBHOOK_SERVER.serve_forever, daemon=True).start()


def write_configs(stack):
    stack.mkdir(parents=True, exist_ok=True)
    data = stack / "data"; data.mkdir(exist_ok=True)
    prom = stack / "prometheus.yml"
    prom.write_text(f"""global:\n  scrape_interval: 5s\n  evaluation_interval: 5s\nrule_files:\n- {stack / 'rules.yml'}\nalerting:\n  alertmanagers:\n  - static_configs:\n    - targets: ['127.0.0.1:9093']\nscrape_configs:\n- job_name: dcgan\n  static_configs:\n  - targets: ['127.0.0.1:8000']\n""", encoding="utf-8")
    (stack / "rules.yml").write_text("""groups:\n- name: dcgan\n  rules:\n  - alert: DCGANQueueBacklog\n    expr: dcgan_inference_queue_depth > 50\n    for: 30s\n    labels:\n      severity: warning\n  - alert: DCGANHighP99Latency\n    expr: histogram_quantile(0.99, sum(rate(dcgan_http_request_latency_seconds_bucket[1m])) by (le)) > 0.2\n    for: 30s\n    labels:\n      severity: critical\n""", encoding="utf-8")
    (stack / "alertmanager.yml").write_text("""route:\n  receiver: dcgan-webhook\n  group_wait: 1s\n  group_interval: 5s\n  repeat_interval: 1h\nreceivers:\n- name: dcgan-webhook\n  webhook_configs:\n  - url: http://127.0.0.1:9094/alerts\n    send_resolved: true\n""", encoding="utf-8")
    dashboard_dir = stack / "grafana_dashboards"
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    grafana = dashboard_dir / "dcgan_07.json"
    panels = []
    panel_specs = (("请求速率", "sum(rate(dcgan_http_requests_total[1m]))"), ("P99延迟", "histogram_quantile(0.99, sum(rate(dcgan_http_request_latency_seconds_bucket[1m])) by (le))"), ("队列", "dcgan_inference_queue_depth"), ("GPU显存", "dcgan_gpu_memory_used_bytes"), ("GPU SM", "dcgan_gpu_sm_utilization_ratio * 100"), ("RSS", "dcgan_process_resident_memory_bytes"))
    for i, (title, expr) in enumerate(panel_specs):
        panels.append({"id": i + 1, "title": title, "type": "timeseries",
                       "datasource": {"type": "prometheus", "uid": "dcgan-prometheus"},
                       "gridPos": {"h": 8, "w": 12, "x": (i % 2) * 12, "y": (i // 2) * 8},
                       "targets": [{"expr": expr, "refId": "A"}]})
    grafana.write_text(json.dumps({"id": None, "uid": "dcgan-07", "title": "DCGAN Inference Service - 07", "schemaVersion": 39, "version": 1, "refresh": "5s", "time": {"from": "now-15m", "to": "now"}, "panels": panels}, ensure_ascii=False, indent=2), encoding="utf-8")
    return prom


def capture_grafana_screenshot(out: Path) -> tuple[bool, str]:
    """Capture the real Grafana UI inside the Kaggle container.

    Kaggle normally does not expose container localhost ports to the user's
    browser.  A headless Chromium instance inside the same container does, so
    this produces a genuine Grafana page screenshot rather than a fabricated
    plot or a screenshot of the JSON dashboard definition.
    """
    screenshot = out / "grafana_dashboard_screenshot.png"
    try:
        # Kaggle images may contain Playwright but not Chromium's shared
        # libraries (for example libatk-1.0.so.0). Install browser plus OS
        # dependencies first; retain a fallback for images where apt is
        # restricted but the browser is already usable.
        with_deps = subprocess.run(
            [sys.executable, "-m", "playwright", "install", "--with-deps", "chromium"],
            check=False,
        )
        if with_deps.returncode != 0:
            apt = shutil.which("apt-get")
            if apt:
                subprocess.run([apt, "update", "-qq"], check=False)
                subprocess.run([
                    apt, "install", "-y", "-qq",
                    "libatk1.0-0", "libatk-bridge2.0-0", "libatspi2.0-0",
                    "libcups2", "libdrm2", "libgbm1", "libgtk-3-0", "libnspr4",
                    "libnss3", "libx11-xcb1", "libxcomposite1", "libxdamage1",
                    "libxfixes3", "libxrandr2", "libxshmfence1", "libasound2",
                    "libxkbcommon0",
                ], check=False)
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        code = r'''
import asyncio
from playwright.async_api import async_playwright

async def capture():
    base = "http://127.0.0.1:3000"
    target = r"__SCREENSHOT__"
    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True, timeout=120000, args=["--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(viewport={"width": 1920, "height": 1080}, device_scale_factor=1)
        page = await context.new_page()
        await page.goto(base + "/login", wait_until="domcontentloaded", timeout=120000)
        user = page.locator('input[name="user"]')
        password = page.locator('input[name="password"]')
        await user.fill("admin")
        await password.fill("admin")
        await page.locator('button[type="submit"]').click()
        await page.wait_for_timeout(5000)
        dashboard_url = base + "/d/dcgan-07/dcgan-inference-service-07?orgId=1&refresh=5s"
        await page.goto(dashboard_url, wait_until="domcontentloaded", timeout=120000)
        await page.wait_for_timeout(10000)
        if "login" in page.url:
            raise RuntimeError("Grafana login did not complete")
        body_text = await page.locator("body").inner_text()
        if "DCGAN" not in body_text and "P99" not in body_text and "队列" not in body_text:
            raise RuntimeError("DCGAN Grafana dashboard was not visible")
        await page.screenshot(path=target, full_page=True)
        await browser.close()

asyncio.run(capture())
'''.replace("__SCREENSHOT__", str(screenshot))
        capture_script = out / "_capture_grafana.py"
        capture_script.write_text(code, encoding="utf-8")
        subprocess.check_call([sys.executable, str(capture_script)])
        capture_script.unlink(missing_ok=True)
        if not screenshot.is_file() or screenshot.stat().st_size < 10000:
            return False, "Grafana screenshot file was not created or is too small"
        return True, str(screenshot)
    except Exception as exc:
        return False, repr(exc)


def download_stack(stack: Path):
    versions = (
        ("prometheus", "https://github.com/prometheus/prometheus/releases/download/v2.53.0/prometheus-2.53.0.linux-amd64.tar.gz"),
        ("alertmanager", "https://github.com/prometheus/alertmanager/releases/download/v0.27.0/alertmanager-0.27.0.linux-amd64.tar.gz"),
        ("grafana", "https://dl.grafana.com/enterprise/release/grafana-enterprise-11.1.0.linux-amd64.tar.gz"),
    )
    paths = {}
    errors = {}
    bin_dir = stack / "bin"; bin_dir.mkdir(exist_ok=True)
    for name, url in versions:
        try:
            marker = bin_dir / name
            if not marker.exists():
                archive = bin_dir / Path(url).name
                urllib.request.urlretrieve(url, archive)
                with tarfile.open(archive, "r:gz") as tar:
                    try:
                        tar.extractall(bin_dir, filter="data")
                    except TypeError:
                        tar.extractall(bin_dir)
                found = [p for p in bin_dir.rglob(name) if p.is_file()]
                if not found:
                    raise RuntimeError(f"找不到下载后的可执行文件 {name}")
                shutil.copy2(found[0], marker)
                if name == "grafana":
                    home = found[0].parent.parent
                    (stack / "grafana_home.txt").write_text(str(home), encoding="utf-8")
            paths[name] = marker
        except Exception as exc:
            errors[name] = repr(exc)
            print(f"[07] download failed: {name}: {exc}", flush=True)
    paths["errors"] = errors
    return paths


def start_stack(stack: Path, config: Path, events: Path):
    global PROCESSES
    paths = download_stack(stack)
    logs = stack / "logs"; logs.mkdir(exist_ok=True)
    errors = dict(paths.get("errors", {}))
    for required in ("prometheus", "alertmanager"):
        if required not in paths:
            raise RuntimeError(f"无法启动必需的 {required}：{errors.get(required, 'unknown error')}")

    commands = [
        ("prometheus", [str(paths["prometheus"]), f"--config.file={config}", f"--storage.tsdb.path={stack / 'data' / 'prometheus'}", "--web.listen-address=127.0.0.1:9090"]),
        ("alertmanager", [str(paths["alertmanager"]), f"--config.file={stack / 'alertmanager.yml'}", f"--storage.path={stack / 'data' / 'alertmanager'}", "--web.listen-address=127.0.0.1:9093", "--cluster.listen-address=127.0.0.1:19094"]),
    ]
    if "grafana" in paths and (stack / "grafana_home.txt").is_file():
        grafana_home = Path((stack / "grafana_home.txt").read_text(encoding="utf-8"))
        grafana_data = stack / "data" / "grafana"
        grafana_data.mkdir(parents=True, exist_ok=True)
        provisioning = stack / "grafana_provisioning"
        (provisioning / "datasources").mkdir(parents=True, exist_ok=True)
        (provisioning / "dashboards").mkdir(parents=True, exist_ok=True)
        grafana_ini = stack / "grafana.ini"
        grafana_ini.write_text(
            f"[server]\nhttp_addr = 127.0.0.1\nhttp_port = {GRAFANA_PORT}\n"
            f"[paths]\ndata = {grafana_data}\nprovisioning = {provisioning}\n",
            encoding="utf-8",
        )
        (provisioning / "datasources" / "prometheus.yml").write_text(
            "apiVersion: 1\ndatasources:\n- name: Prometheus\n  uid: dcgan-prometheus\n  type: prometheus\n  access: proxy\n  url: http://127.0.0.1:9090\n  isDefault: true\n",
            encoding="utf-8",
        )
        (provisioning / "dashboards" / "provider.yml").write_text(
            f"apiVersion: 1\nproviders:\n- name: dcgan-07\n  type: file\n  options:\n    path: {stack / 'grafana_dashboards'}\n",
            encoding="utf-8",
        )
        commands.append(("grafana", [str(paths["grafana"]), "server", f"--homepath={grafana_home}", f"--config={grafana_ini}"]))
    else:
        errors.setdefault("grafana", "Grafana binary unavailable; dashboard screenshot gate cannot pass")

    for name, cmd in commands:
        log = (logs / f"{name}.log").open("w", encoding="utf-8")
        PROCESSES.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT))
    (stack / "stack_start_status.json").write_text(
        json.dumps({"errors": errors, "started": [name for name, _ in commands]}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[07] started monitoring components: {[name for name, _ in commands]}", flush=True)
    return {"started": [name for name, _ in commands], "errors": errors}


def wait(url, timeout=90):
    import requests
    end = time.time() + timeout
    while time.time() < end:
        try:
            if requests.get(url, timeout=3).status_code < 500:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def audit(rt: Runtime, events: Path, stack: Path, args):
    import requests
    base = f"http://{HOST}:{SERVICE_PORT}"
    summary = {"health": requests.get(f"{base}/health", timeout=30).status_code, "generate": requests.post(f"{base}/generate", json={"seed": 20260902}, timeout=60).status_code, "metrics": requests.get(f"{base}/metrics", timeout=30).status_code}
    rows = []
    def one(i):
        start = time.perf_counter()
        try:
            r = requests.post(f"{base}/generate", json={"seed": i + 1000}, timeout=120)
            return {"id": i, "status": r.status_code, "latency_ms": (time.perf_counter() - start) * 1000, "bytes": len(r.content)}
        except Exception as exc:
            return {"id": i, "status": "error", "latency_ms": (time.perf_counter() - start) * 1000, "error": repr(exc)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows.extend(ex.map(one, range(args.requests)))
    with (OUT / "07_load_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r})); writer.writeheader(); writer.writerows(rows)
    metrics = []
    for i in range(args.samples):
        metrics.append({"sample": i, "timestamp": time.time(), **requests.get(f"{base}/metrics.json", timeout=30).json()})
        if i + 1 < args.samples: time.sleep(args.interval)
    with (OUT / "07_metric_snapshots.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in metrics for k in r})); writer.writeheader(); writer.writerows(metrics)
    rules_loaded = 0
    target_up = False
    if stack:
        query_deadline = time.time() + 45
        while time.time() < query_deadline and (rules_loaded < 2 or not target_up):
            try:
                rule_data = requests.get("http://127.0.0.1:9090/api/v1/rules", timeout=15).json().get("data", {})
                rules_loaded = sum(len(group.get("rules", [])) for group in rule_data.get("groups", []))
                target_data = requests.get("http://127.0.0.1:9090/api/v1/targets", timeout=15).json().get("data", {})
                target_up = any(target.get("health") == "up" for target in target_data.get("activeTargets", []))
            except Exception as exc:
                summary["prometheus_query_error"] = repr(exc)
            if rules_loaded >= 2 and target_up:
                break
            time.sleep(2)
    firing = False
    if stack:
        requests.post(f"{base}/debug/queue", json={"value": 60}, timeout=15)
        end = time.time() + args.alert_wait
        while time.time() < end:
            try:
                alerts = requests.get("http://127.0.0.1:9090/api/v1/alerts", timeout=10).json().get("data", {}).get("alerts", [])
                if any(a.get("state") == "firing" for a in alerts): firing = True; break
            except Exception: pass
            time.sleep(2)
        requests.post(f"{base}/debug/clear", timeout=15)
    resolved = False
    resolve_deadline = time.time() + 60
    while time.time() < resolve_deadline:
        statuses = set()
        if events.exists():
            for line in events.read_text(encoding="utf-8").splitlines():
                try:
                    payload = json.loads(line)
                    if payload.get("status"):
                        statuses.add(str(payload["status"]).lower())
                except json.JSONDecodeError:
                    continue
        if "resolved" in statuses:
            resolved = True
            break
        time.sleep(2)
    summary.update({"rules_loaded": rules_loaded, "prometheus_target_up": target_up, "firing_seen": firing, "resolved_seen": resolved, "queue_alert_is_controlled_simulation": True, "grafana_screenshot_exists": (OUT / "grafana_dashboard_screenshot.png").is_file()})
    summary["status"] = "complete" if all((summary["health"] == 200, summary["generate"] == 200, summary["metrics"] == 200, rules_loaded >= 2, target_up, firing, resolved, summary["grafana_screenshot_exists"])) else "incomplete"
    (OUT / "07_validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--base-service", default=os.getenv("BASE_SERVICE_PATH", "")); parser.add_argument("--engine", default=os.getenv("ENGINE_PATH", "")); parser.add_argument("--trt", default=os.getenv("TENSORRT_VERSION", "11.2.1.2")); parser.add_argument("--requests", type=int, default=40); parser.add_argument("--workers", type=int, default=16); parser.add_argument("--samples", type=int, default=12); parser.add_argument("--interval", type=float, default=5); parser.add_argument("--alert-wait", type=float, default=75); args, _ = parser.parse_known_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pip_install()
    service_path, engine_path = inputs(args)
    base = import_file(service_path)
    rt = Runtime(base, engine_path, args)
    for i in range(20): rt.service.infer(20260902 + i)
    import uvicorn
    app = app_for(rt)
    global SERVICE_SERVER
    SERVICE_SERVER = uvicorn.Server(uvicorn.Config(app, host=HOST, port=SERVICE_PORT, workers=1, log_level="info"))
    threading.Thread(target=lambda: asyncio.run(SERVICE_SERVER.serve()), daemon=True).start()
    events = OUT / "alert_webhook_events.jsonl"; start_webhook(events)
    threading.Thread(target=lambda: sampler(rt), daemon=True).start()
    if not wait(f"http://{HOST}:{SERVICE_PORT}/health"): raise RuntimeError("服务健康检查失败")
    stack = OUT / "monitoring_stack"; config = write_configs(stack); stack_status = start_stack(stack, config, events)
    wait("http://127.0.0.1:9090/-/ready")
    wait("http://127.0.0.1:9093/-/ready")
    if "grafana" in stack_status.get("started", []):
        wait("http://127.0.0.1:3000/api/health")
    summary = audit(rt, events, stack, args)
    screenshot_ok, screenshot_detail = capture_grafana_screenshot(OUT)
    summary["grafana_screenshot_exists"] = screenshot_ok
    summary["grafana_screenshot_detail"] = screenshot_detail
    resource_path = OUT / "resource_monitor_5s.csv"
    resource_samples = 0
    if resource_path.is_file():
        with resource_path.open("r", encoding="utf-8") as handle:
            resource_samples = max(0, sum(1 for _ in handle) - 1)
    summary["resource_monitor_samples"] = resource_samples
    summary["resource_monitor_5s_pass"] = resource_samples >= args.samples
    summary["status"] = "complete" if all((
        summary["health"] == 200,
        summary["generate"] == 200,
        summary["metrics"] == 200,
        summary.get("rules_loaded", 0) >= 2,
        summary.get("prometheus_target_up") is True,
        summary.get("firing_seen") is True,
        summary.get("resolved_seen") is True,
        screenshot_ok,
        summary["resource_monitor_5s_pass"],
    )) else "incomplete"
    summary["monitoring_components"] = stack_status
    (OUT / "07_validation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "07_report.md").write_text("# 07 可观测性实验\n\n严格状态：**" + summary["status"] + "**\n\n队列值 60 为受控告警链路模拟，不是真实故障。\n", encoding="utf-8")
    wanted = [
        "07_validation_summary.json", "07_load_results.csv", "07_metric_snapshots.csv",
        "resource_monitor_5s.csv", "alert_webhook_events.jsonl", "07_report.md",
        "grafana_dashboard_screenshot.png", "monitoring_stack/stack_start_status.json",
        "monitoring_stack/prometheus.yml", "monitoring_stack/rules.yml",
        "monitoring_stack/alertmanager.yml", "monitoring_stack/grafana_dashboards/dcgan_07.json",
    ]
    with zipfile.ZipFile(OUT.parent / "07_MLOps_Observability_evidence.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name in wanted:
            p = OUT / name
            if p.is_file(): z.write(p, name)
        log_dir = OUT / "monitoring_stack" / "logs"
        if log_dir.is_dir():
            for log_file in log_dir.glob("*.log"):
                z.write(log_file, f"logs/{log_file.name}")
    print(json.dumps(summary, indent=2, ensure_ascii=False)); print("[07] 证据已写入", OUT)


def sampler(rt):
    path = OUT / "resource_monitor_5s.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["timestamp", "queue_depth", "actual_batch_size", "gpu_memory_used_mb", "gpu_sm_utilization_percent", "rss_mb", "inflight", "model_version"]
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        while not STOP.wait(5):
            writer.writerow({"timestamp": time.time(), **rt.metrics()}); f.flush()


@atexit.register
def cleanup():
    STOP.set()
    if SERVICE_SERVER is not None: SERVICE_SERVER.should_exit = True
    if WEBHOOK_SERVER is not None: WEBHOOK_SERVER.shutdown()
    for p in PROCESSES:
        if p.poll() is None: p.terminate()


if __name__ == "__main__":
    main()
