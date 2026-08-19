"""06A: single-process HTTP service for the final TensorRT generator engine.

This script intentionally owns only service startup, engine preflight, warm-up,
and a small smoke test. Locust and resource monitoring are separate processes
so that service latency and GPU/RSS measurements remain auditable.

The service uses one TensorRT execution context and one CUDA stream protected by
a lock. Concurrent HTTP requests therefore queue at the service boundary rather
than creating extra TensorRT contexts or accidentally loading the engine once
per worker.

Kaggle example:
    python 06A_service.py \
      --engine /kaggle/input/.../generator_trt_qat_int8.engine \
      --output-dir /kaggle/working/service_stress \
      --host 0.0.0.0 --port 8000 --gpu 0

The engine is expected to accept z=[batch,128,1,1] and return image=[batch,3,64,64].
Only batch=1 is exposed by this service; concurrency is measured as concurrent
HTTP requests, not dynamic TensorRT batch size.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SERVICE_SERVER = None
SERVICE_THREAD = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="06A single-process TensorRT HTTP service")
    parser.add_argument(
        "--engine",
        default=os.getenv("ENGINE_PATH", ""),
        help="Serialized TensorRT .engine file; if omitted in Kaggle, 06A auto-detects a QAT engine",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv("SERVICE_OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/06_Service_Stress/06A_Service"),
    )
    parser.add_argument("--host", default=os.getenv("SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("SERVICE_PORT", "8000")))
    parser.add_argument("--gpu", type=int, default=int(os.getenv("CUDA_DEVICE", "0")))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--smoke-requests", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--tensorrt-version", default=os.getenv("TENSORRT_VERSION", "11.2.1.2"))
    parser.add_argument(
        "--tensorrt-pip-package",
        default=os.getenv("TENSORRT_PIP_PACKAGE", "tensorrt-cu12"),
        help="TensorRT CUDA distribution installed automatically before first import",
    )
    parser.add_argument(
        "--skip-dependency-install",
        action="store_true",
        help="Do not install missing packages; useful only when the Kaggle image is already prepared",
    )
    parser.add_argument(
        "--smoke-only",
        action="store_true",
        help="Run dependency/engine preflight and smoke test, write artifacts, then exit without starting Uvicorn",
    )
    parser.add_argument("--allow-version-mismatch", action="store_true")
    args, ignored = parser.parse_known_args(argv)
    if ignored:
        print(f"[06A] ignored notebook/kernel arguments: {ignored}", flush=True)
    return args


def discover_engine(explicit: str) -> Path:
    """Resolve an engine path for CLI and direct-notebook execution.

    A pasted notebook cell has no command-line ``--engine`` argument and
    usually contains a Jupyter ``-f kernel.json`` argument. Prefer a QAT INT8
    engine under /kaggle/input, then search the working directory. We refuse to
    guess between multiple non-QAT engines.
    """
    if explicit:
        return Path(explicit).expanduser().resolve()

    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    qat_candidates: list[Path] = []
    all_candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        try:
            files = root.rglob("*.engine")
        except OSError:
            continue
        for candidate in files:
            key = str(candidate.resolve())
            if key in seen:
                continue
            seen.add(key)
            all_candidates.append(candidate)
            lowered = candidate.name.lower()
            if "qat" in lowered and "int8" in lowered:
                qat_candidates.append(candidate)

    if len(qat_candidates) == 1:
        print(f"[06A] auto-detected QAT engine: {qat_candidates[0]}", flush=True)
        return qat_candidates[0].resolve()
    if len(qat_candidates) > 1:
        choices = "\n".join(f"  - {path}" for path in sorted(qat_candidates))
        raise RuntimeError(
            "Multiple QAT INT8 engines were found. Pass --engine explicitly:\n" + choices
        )
    if len(all_candidates) == 1:
        print(f"[06A] auto-detected the only engine: {all_candidates[0]}", flush=True)
        return all_candidates[0].resolve()
    choices = "\n".join(f"  - {path}" for path in sorted(all_candidates)) or "  (none)"
    raise FileNotFoundError(
        "No unique QAT TensorRT engine was found. Upload generator_trt_qat_int8.engine "
        "to a Kaggle Dataset or pass --engine explicitly. Found:\n" + choices
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def pip_install(packages: list[str], extra_index_url: str | None = None) -> None:
    command = [sys.executable, "-m", "pip", "install", "-q"] + packages
    if extra_index_url:
        command.extend(["--extra-index-url", extra_index_url])
    print(f"[06A] installing dependencies: {' '.join(packages)}", flush=True)
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Dependency installation failed with exit code {exc.returncode}: {' '.join(packages)}. "
            "Check Kaggle Internet access and restart the session if a CUDA package was partially installed."
        ) from exc
    importlib.invalidate_caches()


def distribution_version_candidates(names: tuple[str, ...]) -> list[str]:
    versions: list[str] = []
    for name in names:
        try:
            versions.append(f"{name}=={importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def ensure_dependencies(args: argparse.Namespace) -> dict[str, Any]:
    """Install only missing runtime packages before importing TensorRT/PyTorch.

    The service is normally launched as a fresh subprocess in Kaggle. If this
    file is executed inside a Python process that already imported TensorRT,
    replacing its wheel cannot change the loaded serialization runtime, so the
    script fails with an explicit restart instruction.
    """
    if "tensorrt" in sys.modules:
        loaded = getattr(sys.modules["tensorrt"], "__version__", "unknown")
        if not str(loaded).startswith(args.tensorrt_version) and not args.allow_version_mismatch:
            raise RuntimeError(
                f"TensorRT {loaded} is already imported before 06A dependency setup. "
                f"Expected {args.tensorrt_version}; restart the Kaggle session and run 06A as a fresh process."
            )

    installed: dict[str, str] = {}
    missing = []
    for import_name, package_name in (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("PIL", "Pillow"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("requests", "requests"),
    ):
        try:
            module = importlib.import_module(import_name)
            installed[package_name] = str(getattr(module, "__version__", "present"))
        except ImportError:
            missing.append(package_name)
    if missing and not args.skip_dependency_install:
        pip_install(missing)
        for import_name, package_name in (
            ("numpy", "numpy"),
            ("torch", "torch"),
            ("PIL", "Pillow"),
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("requests", "requests"),
        ):
            try:
                module = importlib.import_module(import_name)
                installed[package_name] = str(getattr(module, "__version__", "present"))
            except ImportError as exc:
                raise RuntimeError(f"Package {package_name} is still unavailable after installation") from exc
    elif missing:
        raise RuntimeError(
            f"Missing packages: {', '.join(missing)}. Remove --skip-dependency-install or install them first."
        )

    trt_versions = distribution_version_candidates(("tensorrt", "tensorrt-cu11", "tensorrt-cu12", "tensorrt-cu13"))
    trt_ok = any(version.split("==", 1)[-1].startswith(args.tensorrt_version) for version in trt_versions)
    if not trt_ok:
        if args.skip_dependency_install:
            raise RuntimeError(
                f"TensorRT {args.tensorrt_version} is not installed. Remove --skip-dependency-install "
                f"or install {args.tensorrt_pip_package}=={args.tensorrt_version}."
            )
        pip_install(
            [f"{args.tensorrt_pip_package}=={args.tensorrt_version}"],
            extra_index_url="https://pypi.nvidia.com",
        )
        trt_versions = distribution_version_candidates(("tensorrt", "tensorrt-cu11", "tensorrt-cu12", "tensorrt-cu13"))
        trt_ok = any(version.split("==", 1)[-1].startswith(args.tensorrt_version) for version in trt_versions)
        if not trt_ok:
            raise RuntimeError(
                f"TensorRT installation completed but version {args.tensorrt_version} was not found. "
                "Restart the Kaggle session before retrying 06A."
            )
    installed["TensorRT distributions"] = "; ".join(trt_versions)
    return installed


def load_runtime(expected_prefix: str, allow_mismatch: bool):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "TensorRT is not importable. Install the TensorRT version used to build the engine "
            "before importing this service, then restart the Kaggle session."
        ) from exc
    loaded = str(trt.__version__)
    if expected_prefix and not loaded.startswith(expected_prefix) and not allow_mismatch:
        raise RuntimeError(
            f"TensorRT runtime mismatch: loaded={loaded}, expected prefix={expected_prefix}. "
            "Use the same runtime as 05B or pass --allow-version-mismatch only after an explicit smoke test."
        )
    return trt


def logger_for(trt):
    return trt.Logger(getattr(trt.Logger, "WARNING", 2))


def io_names(engine, trt) -> tuple[str, str]:
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        output_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        return input_name, output_name
    names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
    input_name = next(name for i, name in enumerate(names) if engine.binding_is_input(i))
    output_name = next(name for i, name in enumerate(names) if not engine.binding_is_input(i))
    return input_name, output_name


def tensor_dtype(engine, trt, name: str):
    if hasattr(engine, "get_tensor_dtype"):
        return trt.nptype(engine.get_tensor_dtype(name))
    return trt.nptype(engine.get_binding_dtype(engine.get_binding_index(name)))


def shape_for_context(context, engine, name: str):
    if hasattr(context, "get_tensor_shape"):
        return tuple(int(value) for value in context.get_tensor_shape(name))
    return tuple(int(value) for value in engine.get_binding_shape(engine.get_binding_index(name)))


@dataclass
class InferenceRecord:
    request_id: str
    seed: int
    status: str
    latency_ms: float
    output_shape: list[int] | None = None
    output_min: float | None = None
    output_max: float | None = None
    output_bytes: int | None = None
    error: str = ""


class TensorRTGenerator:
    def __init__(self, path: Path, trt, gpu: int):
        import numpy as np
        import torch

        self.np = np
        self.torch = torch
        self.path = path
        self.trt = trt
        self.gpu = int(gpu)
        torch.cuda.set_device(self.gpu)
        self.runtime = trt.Runtime(logger_for(trt))
        try:
            self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        except Exception as exc:
            raise RuntimeError(
                f"Could not deserialize {path.name} with TensorRT {trt.__version__}: {exc}"
            ) from exc
        if self.engine is None:
            raise RuntimeError(f"TensorRT returned None while deserializing {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT could not create an execution context")
        self.input_name, self.output_name = io_names(self.engine, trt)
        self.input_dtype = tensor_dtype(self.engine, trt, self.input_name)
        self.output_dtype = tensor_dtype(self.engine, trt, self.output_name)
        self.stream = torch.cuda.Stream(device=self.gpu)
        self.lock = threading.Lock()
        self.request_count = 0

        if self.input_dtype not in (np.float16, np.float32):
            raise RuntimeError(f"Unsupported engine input dtype: {self.input_dtype}")
        if self.output_dtype not in (np.float16, np.float32):
            raise RuntimeError(f"Unsupported engine output dtype: {self.output_dtype}")

    def _make_latent(self, seed: int):
        generator = self.torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return self.torch.randn((1, 128, 1, 1), generator=generator, dtype=self.torch.float32).numpy()

    def infer(self, seed: int):
        import numpy as np
        import torch

        latent = self._make_latent(seed)
        with self.lock:
            started = time.perf_counter()
            host = np.ascontiguousarray(latent.astype(self.input_dtype, copy=False))
            with torch.inference_mode():
                input_tensor = torch.from_numpy(host).to(
                    device=f"cuda:{self.gpu}",
                    dtype=torch.float16 if self.input_dtype == np.float16 else torch.float32,
                )
                if hasattr(self.context, "set_input_shape"):
                    if self.context.set_input_shape(self.input_name, tuple(input_tensor.shape)) is False:
                        raise RuntimeError(f"TensorRT rejected input shape {tuple(input_tensor.shape)}")
                    output_shape = shape_for_context(self.context, self.engine, self.output_name)
                else:
                    input_index = self.engine.get_binding_index(self.input_name)
                    self.context.set_binding_shape(input_index, tuple(input_tensor.shape))
                    output_shape = shape_for_context(self.context, self.engine, self.output_name)
                if any(value <= 0 for value in output_shape):
                    raise RuntimeError(f"Invalid resolved output shape: {output_shape}")
                output = torch.empty(
                    output_shape,
                    device=f"cuda:{self.gpu}",
                    dtype=torch.float16 if self.output_dtype == np.float16 else torch.float32,
                )
                stream = self.stream.cuda_stream
                if hasattr(self.context, "set_tensor_address"):
                    self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
                    self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
                    ok = self.context.execute_async_v3(stream)
                else:
                    input_index = self.engine.get_binding_index(self.input_name)
                    output_index = self.engine.get_binding_index(self.output_name)
                    bindings = [0] * self.engine.num_bindings
                    bindings[input_index] = int(input_tensor.data_ptr())
                    bindings[output_index] = int(output.data_ptr())
                    ok = self.context.execute_async_v2(bindings, stream)
                if ok is False:
                    raise RuntimeError("TensorRT execution returned False")
                self.stream.synchronize()
                result = output.float().cpu().numpy()
                del input_tensor, output
            self.request_count += 1
            elapsed_ms = (time.perf_counter() - started) * 1000.0
        return result, elapsed_ms

    def metadata(self) -> dict[str, Any]:
        import torch

        properties = torch.cuda.get_device_properties(self.gpu)
        return {
            "engine_file": self.path.name,
            "engine_sha256": sha256(self.path),
            "engine_bytes": self.path.stat().st_size,
            "tensorrt_version": str(self.trt.__version__),
            "gpu_index": self.gpu,
            "gpu_name": properties.name,
            "gpu_total_memory_mb": round(properties.total_memory / 1024**2, 2),
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "input_dtype": str(self.input_dtype),
            "output_dtype": str(self.output_dtype),
            "expected_input_shape": [1, 128, 1, 1],
            "expected_output_shape": [1, 3, 64, 64],
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "request_count": self.request_count,
        }


def image_png(output) -> bytes:
    import numpy as np
    from PIL import Image

    array = np.asarray(output, dtype=np.float32)
    if array.ndim != 4 or array.shape[0] != 1 or array.shape[1] != 3:
        raise RuntimeError(f"Expected output [1,3,H,W], got {array.shape}")
    array = np.clip((array[0] + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    image = Image.fromarray(np.transpose(array, (1, 2, 0)))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def write_smoke_csv(path: Path, rows: list[InferenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(asdict(InferenceRecord("", 0, "", 0.0)).keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def create_app(service: TensorRTGenerator, manifest: dict[str, Any]):
    try:
        from fastapi import Body, FastAPI, HTTPException, Response
    except ImportError as exc:
        raise RuntimeError(
            "FastAPI/Pydantic are missing. Install fastapi and uvicorn before starting 06A."
        ) from exc

    app = FastAPI(title="DCGAN TensorRT Service", version="06A")
    app.state.service = service
    app.state.manifest = manifest

    @app.get("/health")
    async def health():
        payload = dict(manifest)
        payload.update({"status": "ok", "request_count": service.request_count})
        return payload

    # HTTP validation evidence is collected in two separate Kaggle cells:
    # 1) GET /health must return HTTP 200 and status="ok".
    # 2) POST /generate with JSON {"seed": 1234} must return HTTP 200,
    #    Content-Type image/png, and a 64x64 image.
    #
    # Body(embed=True) is intentional. The previous nested Pydantic model was
    # resolved as a query parameter under postponed annotations, producing
    # 422: missing query parameter "request" for a valid JSON body.
    @app.post("/generate")
    async def generate(seed: int = Body(default=0, embed=True, ge=0, le=2**31 - 1)):
        request_id = f"req-{time.time_ns()}"
        try:
            output, inference_ms = await asyncio.to_thread(service.infer, int(seed))
            content = image_png(output)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"inference_failed: {exc!r}") from exc
        return Response(
            content=content,
            media_type="image/png",
            headers={
                "X-Request-ID": request_id,
                "X-Inference-Ms": f"{inference_ms:.6f}",
                "X-Engine": service.path.name,
            },
        )

    return app


def start_uvicorn(app, host: str, port: int) -> None:
    """Start Uvicorn in CLI mode or in a daemon thread inside Jupyter.

    ``uvicorn.run`` calls ``asyncio.run``. That is correct in a normal Python
    process but raises ``asyncio.run() cannot be called from a running event
    loop`` when the script is pasted into a Kaggle/Jupyter cell. In that case
    keep the existing notebook loop untouched and run the server in a daemon
    thread with its own loop.
    """
    global SERVICE_SERVER, SERVICE_THREAD
    try:
        asyncio.get_running_loop()
        notebook_loop_is_running = True
    except RuntimeError:
        notebook_loop_is_running = False

    if not notebook_loop_is_running:
        import uvicorn

        uvicorn.run(app, host=host, port=port, workers=1, log_level="info")
        return

    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, workers=1, log_level="info")
    server = uvicorn.Server(config)
    SERVICE_SERVER = server
    app.state.uvicorn_server = server

    def serve_in_background() -> None:
        asyncio.run(server.serve())

    thread = threading.Thread(
        target=serve_in_background,
        name="06A-uvicorn",
        daemon=True,
    )
    SERVICE_THREAD = thread
    thread.start()
    print(
        "[06A] detected a running Notebook event loop; Uvicorn started in daemon thread. "
        "Continue in the next cell and call /health.",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    engine_path = discover_engine(args.engine)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not engine_path.is_file():
        raise FileNotFoundError(f"Engine not found: {engine_path}")
    if engine_path.suffix.lower() != ".engine":
        raise ValueError(f"06A expects a serialized TensorRT .engine file: {engine_path.name}")

    dependency_versions = ensure_dependencies(args)
    import numpy as np  # noqa: F401
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; 06A requires a GPU TensorRT runtime")
    trt = load_runtime(args.tensorrt_version, args.allow_version_mismatch)
    service = TensorRTGenerator(engine_path, trt, args.gpu)

    smoke_rows: list[InferenceRecord] = []
    warmup_started = time.perf_counter()
    for index in range(max(0, args.warmup)):
        service.infer(args.seed + index)
    warmup_ms = (time.perf_counter() - warmup_started) * 1000.0

    for index in range(max(0, args.smoke_requests)):
        seed = args.seed + args.warmup + index
        request_id = f"smoke-{index:04d}"
        try:
            output, latency_ms = service.infer(seed)
            encoded = image_png(output)
            smoke_rows.append(
                InferenceRecord(
                    request_id=request_id,
                    seed=seed,
                    status="passed",
                    latency_ms=latency_ms,
                    output_shape=list(output.shape),
                    output_min=float(output.min()),
                    output_max=float(output.max()),
                    output_bytes=len(encoded),
                )
            )
        except Exception as exc:
            smoke_rows.append(
                InferenceRecord(
                    request_id=request_id,
                    seed=seed,
                    status="failed",
                    latency_ms=0.0,
                    error=repr(exc),
                )
            )
            raise

    output_dir.mkdir(parents=True, exist_ok=True)
    write_smoke_csv(output_dir / "service_smoke_test.csv", smoke_rows)
    manifest = {
        "task": "Task6_06A_Service_Preflight",
        "status": "ready",
        "service_mode": "single_process_single_worker_single_context_batch1",
        "engine_role": "QAT_Hybrid_INT8_FP16",
        "engine": service.metadata(),
        "runtime": {
            "python": sys.version,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "selected_gpu_index": args.gpu,
            "host": args.host,
            "port": args.port,
            "dependency_installation": {
                "performed_inside_script": not args.skip_dependency_install,
                "versions": dependency_versions,
                "tensorrt_pip_package": args.tensorrt_pip_package,
                "tensorrt_requested": args.tensorrt_version,
            },
        },
        "warmup": {
            "requests": max(0, args.warmup),
            "total_ms": warmup_ms,
        },
        "smoke": {
            "requests": len(smoke_rows),
            "passed": sum(row.status == "passed" for row in smoke_rows),
            "failed": sum(row.status != "passed" for row in smoke_rows),
            "csv": "service_smoke_test.csv",
        },
        "api": {
            "health": "GET /health",
            "generate": "POST /generate with JSON {\"seed\": integer}",
            "response": "image/png",
            "batch": 1,
        },
        "notes": [
            "Concurrency is HTTP request concurrency; TensorRT dynamic batch is intentionally fixed at 1.",
            "Use one uvicorn worker. Multiple workers would load duplicate engine contexts and invalidate the memory test.",
            "06B Locust and 06C monitoring must run as separate processes.",
        ],
    }
    json_dump(output_dir / "service_manifest.json", manifest)

    if any(row.status != "passed" for row in smoke_rows):
        raise RuntimeError("06A smoke test failed; service will not start")

    if args.smoke_only:
        print(f"[06A] smoke-only completed; artifacts written to {output_dir}")
        return

    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is missing. Install uvicorn before starting 06A.") from exc
    app = create_app(service, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"[06A] serving on http://{args.host}:{args.port}; pid={os.getpid()}")
    start_uvicorn(app, args.host, args.port)


if __name__ == "__main__":
    main()
