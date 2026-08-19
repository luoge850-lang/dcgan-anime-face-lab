"""
Task 4 / 04B - mixed-precision combination screening.

04A answers: "which single ConvTranspose layer is sensitive when restored to
FP16?" 04B answers: "which combination of FP16-restored layers gives the best
quality/latency trade-off?"

The script deliberately starts every combination from the same named raw ONNX
graph and calls ModelOpt INT8 PTQ with a list of exact node-exclusion regexes.
An excluded node is kept in the requested high-precision dtype (FP16); every
other eligible node remains INT8.  This keeps the comparison scientifically
fair and avoids treating an already-built all-INT8 TensorRT engine as if it
could be edited layer by layer.

It is written for a Kaggle Notebook cell or a normal Python process and has no
dependency on the 04A script or 04A result files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = 64
NOISE_DIM = 128
DEFAULT_LAYERS = ("net.0", "net.3", "net.6", "net.9", "net.12")
DEFAULT_COMBINATIONS = (
    "fp32",
    "all_int8",
    # Primary policy: combine the two useful 04A candidates.  net.0 has the
    # smallest speed cost and net.12 has the strongest FID recovery.
    "net.0+net.12",
    # net.9 improves blur rate but damages FID in 04A: test whether that
    # behavior complements or conflicts with the useful pair.
    "net.9+net.12",
    "net.0+net.9",
    "net.0+net.9+net.12",
    # Add one low-value layer as an overhead control; this tests whether a
    # larger FP16 budget is justified instead of assuming it helps.
    "net.0+net.3+net.12",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="04B mixed-precision combination screening")
    parser.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    parser.add_argument("--fp32-engine", default=os.getenv("FP32_ENGINE_PATH", ""))
    parser.add_argument("--int8-engine", default=os.getenv("INT8_ENGINE_PATH", ""))
    parser.add_argument(
        "--protocol-path",
        "--protocol-zip",
        dest="protocol_path",
        default=os.getenv("PROTOCOL_PATH", os.getenv("PROTOCOL_ZIP_PATH", "")),
        help="03A ZIP or extracted directory containing latent_eval.npy and real_eval/",
    )
    parser.add_argument(
        "--task3-bundle",
        default=os.getenv("TASK3_BUNDLE_PATH", ""),
        help="Optional directory or one Task-3 ZIP; otherwise the three Task-3 ZIPs are auto-detected.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/04_Quantization_Sensitivity/04B_Mixed_Precision",
        ),
    )
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    parser.add_argument(
        "--combinations",
        default=";".join(DEFAULT_COMBINATIONS),
        help="Semicolon-separated labels; use 'exhaustive' only for an optional full appendix.",
    )
    parser.add_argument("--calibration-method", choices=("entropy", "max"), default="entropy")
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--n-fid", type=int, default=1000, help="Use 5000 for final report confirmation.")
    parser.add_argument("--n-image-eval", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-fid-delta", type=float, default=1.5)
    parser.add_argument("--max-blur-delta-pp", type=float, default=0.5)
    parser.add_argument("--max-latency-ratio", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=20260817)
    args, _unknown = parser.parse_known_args(argv)
    args.layers = [item.strip() for item in str(args.layers).split(",") if item.strip()]
    args.combinations = [item.strip() for item in str(args.combinations).split(";") if item.strip()]
    if not args.layers:
        raise ValueError("--layers must contain at least one layer label")
    if not args.combinations:
        raise ValueError("--combinations must contain at least one strategy")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


TRT_LOGGER = None


def get_logger(trt):
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return TRT_LOGGER


def ensure_tensorrt():
    try:
        import tensorrt as trt
        print(f"[TensorRT] {trt.__version__}")
        return trt
    except (ImportError, ModuleNotFoundError):
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator.")
    major = str(torch.version.cuda or "").split(".", 1)[0]
    package = {"11": "tensorrt-cu11", "12": "tensorrt-cu12", "13": "tensorrt-cu13"}.get(major)
    if not package:
        raise RuntimeError(f"Unsupported CUDA version: {torch.version.cuda}")
    command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", package]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        subprocess.check_call(command + ["--extra-index-url", "https://pypi.nvidia.com"])
    importlib.invalidate_caches()
    import tensorrt as trt
    print(f"[TensorRT] installed {trt.__version__}")
    return trt


def _engine_candidates(name: str):
    found = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            try:
                found.extend(root.rglob(name))
            except PermissionError:
                pass
    return sorted({path.resolve() for path in found if path.is_file()})


def locate_engine(name: str, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit engine path does not exist: {path}")
        return path
    candidates = _engine_candidates(name)
    if not candidates:
        raise FileNotFoundError(f"{name} was not found. Attach the Task-3 engine package or set the explicit path.")
    by_digest = {}
    for path in candidates:
        by_digest.setdefault(sha256(path), []).append(path)
    if len(by_digest) > 1:
        raise RuntimeError("Multiple different engine files found; set the explicit path:\n" + "\n".join(map(str, candidates)))
    copies = next(iter(by_digest.values()))
    chosen = sorted(copies, key=lambda path: (0 if str(path).startswith("/kaggle/input") else 1, str(path)))[0]
    print(f"[engine] {name}: {chosen} (identical copies collapsed)")
    return chosen


def _valid_protocol_dir(path: Path) -> bool:
    return path.is_dir() and (path / "latent_eval.npy").is_file() and (path / "real_eval").is_dir()


def locate_protocol(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".zip":
            return path
        if _valid_protocol_dir(path):
            return path
        candidates = [candidate.parent for candidate in path.rglob("latent_eval.npy") if _valid_protocol_dir(candidate.parent)] if path.is_dir() else []
        if len(candidates) == 1:
            return candidates[0].resolve()
        raise FileNotFoundError(f"Explicit protocol path is not a valid 03A package: {path}")
    zips, dirs = [], []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.exists():
            continue
        try:
            zips.extend(root.rglob("Task3_03A_Quantization_Protocol.zip"))
            dirs.extend(candidate.parent for candidate in root.rglob("latent_eval.npy") if _valid_protocol_dir(candidate.parent))
        except PermissionError:
            pass
    zips = sorted({path.resolve() for path in zips if path.is_file()})
    dirs = sorted({path.resolve() for path in dirs})
    if len(zips) == 1:
        return zips[0]
    if len(dirs) == 1:
        return dirs[0]
    if len(zips) > 1 or len(dirs) > 1:
        raise RuntimeError("Multiple 03A protocol packages found; set --protocol-path explicitly")
    raise FileNotFoundError("Task3_03A_Quantization_Protocol.zip or its extracted directory was not found")


def extract_protocol(source: Path, output_dir: Path) -> Path:
    if source.is_dir():
        if _valid_protocol_dir(source):
            return source.resolve()
        candidates = [candidate.parent for candidate in source.rglob("latent_eval.npy") if _valid_protocol_dir(candidate.parent)]
        if len(candidates) == 1:
            return candidates[0].resolve()
        raise FileNotFoundError(f"Could not resolve a valid 03A protocol directory under {source}")
    extract_dir = output_dir / "protocol_03a"
    extract_dir.mkdir(parents=True, exist_ok=True)
    marker = extract_dir / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(source) as handle:
            handle.extractall(extract_dir)
        marker.write_text("ok", encoding="utf-8")
    return extract_protocol(extract_dir, output_dir)


def get_io_names(engine, trt):
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        input_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        output_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        return input_name, output_name
    names = [engine.get_binding_name(index) for index in range(engine.num_bindings)]
    input_name = next(name for index, name in enumerate(names) if engine.binding_is_input(index))
    output_name = next(name for index, name in enumerate(names) if not engine.binding_is_input(index))
    return input_name, output_name


class TensorRTEngine:
    def __init__(self, path: Path, trt, device):
        self.path, self.trt, self.device = path, trt, device
        runtime = trt.Runtime(get_logger(trt))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = get_io_names(self.engine, trt)
        self.input_np_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name)) if hasattr(self.engine, "get_tensor_dtype") else np.float32
        self.output_np_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name)) if hasattr(self.engine, "get_tensor_dtype") else np.float32
        self.input_torch_dtype = torch.float16 if self.input_np_dtype == np.float16 else torch.float32
        self.output_torch_dtype = torch.float16 if self.output_np_dtype == np.float16 else torch.float32
        self.stream = torch.cuda.Stream()

    def infer(self, latent: np.ndarray):
        host = np.ascontiguousarray(latent.astype(self.input_np_dtype, copy=False))
        input_tensor = torch.from_numpy(host).to(self.device, dtype=self.input_torch_dtype)
        output_tensor = torch.empty((latent.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device, dtype=self.output_torch_dtype)
        stream = self.stream.cuda_stream
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
            self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))
            ok = self.context.execute_async_v3(stream)
        else:
            input_index = self.engine.get_binding_index(self.input_name)
            output_index = self.engine.get_binding_index(self.output_name)
            self.context.set_binding_shape(input_index, tuple(input_tensor.shape))
            bindings = [0] * self.engine.num_bindings
            bindings[input_index] = int(input_tensor.data_ptr())
            bindings[output_index] = int(output_tensor.data_ptr())
            ok = self.context.execute_async_v2(bindings, stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.path.name}")
        self.stream.synchronize()
        return output_tensor.float().detach().cpu()


class StandardFID:
    def __init__(self, device):
        try:
            from pytorch_fid.inception import InceptionV3
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-fid"])
            importlib.invalidate_caches()
            from pytorch_fid.inception import InceptionV3
        pool3_block = int(getattr(InceptionV3, "DEFAULT_BLOCK_INDEX", 3))
        self.device = device
        self.net = InceptionV3([pool3_block]).eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images01):
        values = self.net(images01.to(self.device))[0]
        return values.reshape(values.shape[0], -1).detach().cpu().numpy()


def load_real_images(real_dir: Path, count: int, batch_size: int):
    paths = sorted(real_dir.glob("*.png"))[:count]
    if len(paths) < count:
        raise RuntimeError(f"real_eval contains {len(paths)} PNG files, but {count} are required")
    for start in range(0, count, batch_size):
        images = []
        batch_paths = paths[start : start + batch_size]
        for path in batch_paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        yield torch.stack(images, dim=0), batch_paths


def stats(features):
    return features.mean(axis=0), np.cov(features, rowvar=False)


def fid_from_stats(real_mu, real_cov, fake_mu, fake_cov):
    from scipy import linalg
    diff = real_mu - fake_mu
    covmean = linalg.sqrtm(real_cov.dot(fake_cov))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(real_cov + fake_cov - 2.0 * covmean))


def laplacian_values(images01):
    values = []
    for image in images01:
        array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gray = array.mean(axis=2).astype(np.float32)
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
        values.append(float(lap.var()))
    return np.asarray(values, dtype=np.float64)


def build_qdq_engine(trt, quant_path: Path, engine_path: Path, opt_batch: int, max_batch: int, workspace_gb: float):
    builder = trt.Builder(get_logger(trt))
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, get_logger(trt))
    if not parser.parse(quant_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT Q/DQ parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    workspace = int(workspace_gb * (1024 ** 3))
    memory_pool = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and memory_pool is not None:
        config.set_memory_pool_limit(memory_pool.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, NOISE_DIM, 1, 1), (opt_batch, NOISE_DIM, 1, 1), (max_batch, NOISE_DIM, 1, 1))
    config.add_optimization_profile(profile)
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
        engine_path.write_bytes(bytes(serialized))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
        engine_path.write_bytes(bytes(engine.serialize()))


def evaluate_engine(
    engine_path: Path,
    label: str,
    trt,
    evaluator_module,
    device,
    latent_eval: np.ndarray,
    real_mu: np.ndarray,
    real_cov: np.ndarray,
    real_metric,
    blur_threshold: float,
    standard_fid_net,
    batch_size: int,
    n_fid: int,
    n_image_eval: int,
    warmup: int,
    iters: int,
):
    engine = evaluator_module.TensorRTEngine(engine_path, trt, device)
    for _ in range(warmup):
        engine.infer(latent_eval[: min(batch_size, n_fid)])
    all_features = []
    metric_parts = []
    for start in range(0, n_fid, batch_size):
        batch = latent_eval[start : start + batch_size]
        fake = engine.infer(batch)
        fake01 = ((fake + 1.0) * 0.5).clamp(0.0, 1.0)
        all_features.append(standard_fid_net.features(fake01))
        if start < n_image_eval:
            metric_parts.append(fake01[: min(len(fake01), n_image_eval - start)])
    fake_features = np.concatenate(all_features, axis=0)[:n_fid]
    fake_mu, fake_cov = evaluator_module.stats(fake_features)
    fake_metric = torch.cat(metric_parts, dim=0)[:n_image_eval]
    fake_lap = evaluator_module.laplacian_values(fake_metric)
    torch.cuda.synchronize()
    free_mb, total_mb = torch.cuda.mem_get_info()
    repeat_noise = latent_eval[:batch_size]
    repeat = []
    for _ in range(iters):
        t0 = time.perf_counter()
        engine.infer(repeat_noise)
        repeat.append((time.perf_counter() - t0) * 1000.0)
    mean_ms = float(np.mean(repeat))
    return {
        "label": label,
        "engine": engine_path.name,
        "fid_standard": evaluator_module.fid_from_stats(real_mu, real_cov, fake_mu, fake_cov),
        "blur_rate": float((fake_lap < blur_threshold).mean()),
        "laplacian_mean": float(fake_lap.mean()),
        "latency_mean_ms_batch": mean_ms,
        "throughput_images_per_s": float(batch_size / (mean_ms / 1000.0)),
        "cuda_used_after_engine_mb": float(total_mb - free_mb),
        "fake_metric": fake_metric,
        "fake_features": fake_features,
    }


def _find_zip(search_root: Path, filename: str):
    if not search_root.exists():
        return []
    try:
        return sorted({path.resolve() for path in search_root.rglob(filename) if path.is_file()})
    except PermissionError:
        return []


def prepare_task3_archives(explicit: str, output_dir: Path):
    names = ("Task3_03A_Quantization_Protocol.zip", "Task3_03B_FP32_FP16_Engines.zip", "Task3_03C_INT8_PTQ_Engine.zip")
    archive_paths = []
    if explicit:
        supplied = Path(explicit).expanduser().resolve()
        if supplied.is_file() and supplied.suffix.lower() == ".zip":
            archive_paths = [supplied]
        elif supplied.is_dir():
            archive_paths = [path for name in names for path in _find_zip(supplied, name)]
    else:
        for name in names:
            found = []
            for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
                found.extend(_find_zip(root, name))
            if found:
                digests = {sha256(path): path for path in found}
                if len(digests) > 1:
                    raise RuntimeError(f"Multiple different {name} files found; set --task3-bundle explicitly")
                archive_paths.append(next(iter(digests.values())))
    unique_archives = []
    seen = set()
    for path in archive_paths:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique_archives.append(path)
    if not unique_archives:
        return None
    staging = output_dir / "_task3_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    protocol_dir = None
    for archive in unique_archives:
        target = staging / archive.stem
        marker = target / ".extracted"
        target.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
            marker.write_text("ok", encoding="utf-8")
        protocol_candidates = [latent.parent for latent in target.rglob("latent_eval.npy") if (latent.parent / "real_eval").is_dir()]
        if len(protocol_candidates) == 1:
            protocol_dir = protocol_candidates[0].resolve()
        print(f"[task3-package] extracted {archive.name} -> {target}")
    return protocol_dir


def ensure_modelopt():
    try:
        import modelopt  # noqa: F401
        print("[ModelOpt] found")
        return
    except (ImportError, ModuleNotFoundError):
        pass
    print("[ModelOpt] installing nvidia-modelopt[onnx]")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "nvidia-modelopt[onnx]", "--extra-index-url", "https://pypi.nvidia.com"])
    importlib.invalidate_caches()
    import modelopt  # noqa: F401


def locate_unique(name: str, explicit: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit path does not exist: {path}")
        print(f"[input] {name}: {path}")
        return path
    if name == "generator_fp32_raw.onnx":
        candidates = []
        for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
            if root.exists():
                try:
                    candidates.extend(root.rglob("generator_fp32_raw.onnx"))
                    candidates.extend(root.rglob("generator.onnx"))
                except PermissionError:
                    pass
        unique = sorted({path.resolve() for path in candidates if path.is_file()})
        if len(unique) == 1:
            return unique[0]
        if len(unique) > 1:
            by_digest = {}
            for path in unique:
                by_digest.setdefault(sha256(path), []).append(path)
            if len(by_digest) == 1:
                return sorted(next(iter(by_digest.values())), key=lambda path: (0 if str(path).startswith("/kaggle/input") else 1, str(path)))[0]
            raise RuntimeError("Multiple different raw ONNX files found; set --raw-onnx explicitly:\n" + "\n".join(map(str, unique)))
        raise FileNotFoundError("generator_fp32_raw.onnx or generator.onnx was not found")
    return locate_engine(name, explicit)


def make_named_source(raw_path: Path, output_path: Path, requested_labels: list[str]):
    import onnx
    model = onnx.load(str(raw_path))
    onnx.checker.check_model(model)
    nodes = [node for node in model.graph.node if node.op_type == "ConvTranspose"]
    if not nodes:
        raise RuntimeError("No ConvTranspose nodes found in raw ONNX")
    labels = list(requested_labels)
    while len(labels) < len(nodes):
        labels.append(f"net.{len(labels) * 3}")
    labels = labels[: len(nodes)]
    candidates = {}
    for index, node in enumerate(nodes):
        node.name = f"{labels[index]}.ConvTranspose"
        candidates[labels[index]] = node.name
    onnx.save(model, str(output_path))
    onnx.checker.check_model(onnx.load(str(output_path)))
    return candidates, len(model.graph.node)


def parse_strategy(strategy: str, layers: list[str]) -> list[str]:
    if strategy in {"fp32", "all_int8"}:
        return []
    parts = [item.strip() for item in strategy.split("+") if item.strip()]
    if not parts or len(parts) != len(set(parts)):
        raise ValueError(f"Invalid duplicate/empty strategy: {strategy}")
    unknown = [item for item in parts if item not in layers]
    if unknown:
        raise ValueError(f"Strategy {strategy} contains unknown layers: {unknown}; layers={layers}")
    return parts


def expand_combinations(requested: list[str], layers: list[str]) -> list[str]:
    """Expand the optional exhaustive mode without making it the default run.

    The default matrix is the efficient, report-oriented design space.  When
    a complete interaction appendix is required, ``--combinations
    exhaustive`` evaluates every multi-layer INT8/FP16 assignment (all
    combinations with at least two FP16-restored layers) plus the separate
    FP32 and all-INT8 references.  With the current five-layer graph this is
    28 rows including the references; the five single-layer rows remain the
    04A responsibility.
    """
    if requested not in (["exhaustive"], ["all"]):
        return requested
    expanded = ["fp32", "all_int8"]
    for mask in range(1, 1 << len(layers)):
        restored = [label for index, label in enumerate(layers) if mask & (1 << index)]
        if len(restored) >= 2:
            expanded.append("+".join(restored))
    return expanded


def slug(strategy: str) -> str:
    if strategy == "fp32":
        return "FP32"
    if strategy == "all_int8":
        return "INT8"
    return "FP16_" + "_".join(strategy.split("+"))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantize_combination(
    named_source: Path,
    latent_calibration: Path,
    output_path: Path,
    excluded_nodes: list[str],
    method: str,
):
    import onnx
    from modelopt.onnx.quantization import quantize

    model = onnx.load(str(named_source))
    input_name = model.graph.input[0].name
    calibration = np.load(latent_calibration).astype(np.float32, copy=False)
    patterns = [f"^{re.escape(node_name)}$" for node_name in excluded_nodes]
    quantize(
        onnx_path=str(named_source),
        quantize_mode="int8",
        calibration_data={input_name: calibration},
        calibration_method=method,
        nodes_to_exclude=patterns,
        high_precision_dtype="fp16",
        output_path=str(output_path),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"ModelOpt did not create {output_path}")
    quantized = onnx.load(str(output_path))
    onnx.checker.check_model(quantized)
    counts = {}
    for node in quantized.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if counts.get("QuantizeLinear", 0) == 0 or counts.get("DequantizeLinear", 0) == 0:
        raise RuntimeError(f"{output_path.name} contains no Q/DQ nodes: {counts}")
    return {
        "excluded_nodes": excluded_nodes,
        "excluded_patterns": patterns,
        "node_count": len(quantized.graph.node),
        "quantize_linear": counts.get("QuantizeLinear", 0),
        "dequantize_linear": counts.get("DequantizeLinear", 0),
    }


def make_plot(rows: list[dict], output_path: Path):
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return False
    baseline = {row["strategy"]: row for row in valid if row["strategy"] in {"fp32", "all_int8"}}
    fp32 = baseline.get("fp32")
    int8 = baseline.get("all_int8")
    variants = [row for row in valid if row["strategy"] not in {"fp32", "all_int8"}]
    variants.sort(key=lambda row: (int(row.get("fp16_layer_count", 0)), row["strategy"]))
    labels = [row["strategy"] for row in variants]
    if not labels:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(15, 9), dpi=180)
    x = np.arange(len(labels))
    colors = ["#2ca02c" if row.get("quality_gate") == "pass" else "#4c78a8" for row in variants]
    axes[0, 0].bar(x, [row["fid_standard"] for row in variants], color=colors)
    axes[0, 0].axhline(fp32["fid_standard"], color="#111111", linestyle="--", label="FP32") if fp32 else None
    axes[0, 0].axhline(int8["fid_standard"], color="#d62728", linestyle=":", label="INT8") if int8 else None
    axes[0, 0].set_title("Standard FID (lower is better)")
    axes[0, 0].set_ylabel("FID")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].bar(x, [row["blur_rate"] * 100.0 for row in variants], color=colors)
    axes[0, 1].axhline(fp32["blur_rate"] * 100.0, color="#111111", linestyle="--", label="FP32") if fp32 else None
    axes[0, 1].axhline(int8["blur_rate"] * 100.0, color="#d62728", linestyle=":", label="INT8") if int8 else None
    axes[0, 1].set_title("Blur rate (lower is better)")
    axes[0, 1].set_ylabel("Blur rate (%)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].bar(x, [row["latency_mean_ms_batch"] for row in variants], color=colors)
    axes[1, 0].axhline(int8["latency_mean_ms_batch"], color="#d62728", linestyle=":", label="INT8") if int8 else None
    axes[1, 0].set_title("TensorRT latency at fixed batch")
    axes[1, 0].set_ylabel("Latency (ms/batch)")
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].scatter(
        [row["latency_mean_ms_batch"] for row in variants],
        [row["fid_standard"] for row in variants],
        c=[row["fp16_layer_count"] for row in variants],
        cmap="viridis",
        s=75,
        edgecolors="black",
        linewidths=0.4,
    )
    for row in variants:
        axes[1, 1].annotate(row["strategy"], (row["latency_mean_ms_batch"], row["fid_standard"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if int8:
        axes[1, 1].scatter([int8["latency_mean_ms_batch"]], [int8["fid_standard"]], marker="X", s=100, color="#d62728", label="INT8")
    if fp32:
        axes[1, 1].scatter([fp32["latency_mean_ms_batch"]], [fp32["fid_standard"]], marker="P", s=100, color="#111111", label="FP32")
    axes[1, 1].set_title("Quality-speed trade-off")
    axes[1, 1].set_xlabel("Latency (ms/batch, lower is better)")
    axes[1, 1].set_ylabel("FID (lower is better)")
    axes[1, 1].legend(fontsize=8)

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.25)
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    fig.suptitle("04B Mixed-Precision Combination Screening", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def choose_recommendation(rows: list[dict], args):
    valid = [row for row in rows if row.get("status") == "ok"]
    fp32 = next((row for row in valid if row["strategy"] == "fp32"), None)
    int8 = next((row for row in valid if row["strategy"] == "all_int8"), None)
    if not fp32 or not int8:
        return None, "missing FP32 or INT8 reference"

    for row in valid:
        if row["strategy"] in {"fp32", "all_int8"}:
            row["quality_gate"] = "reference"
            continue
        fid_ok = row["fid_standard"] <= fp32["fid_standard"] + args.max_fid_delta
        blur_ok = row["blur_delta_pp_vs_fp32"] <= args.max_blur_delta_pp
        speed_ok = row["latency_mean_ms_batch"] <= int8["latency_mean_ms_batch"] * args.max_latency_ratio
        row["quality_gate"] = "pass" if fid_ok and blur_ok and speed_ok else "screen_only"

    candidates = [row for row in valid if row["quality_gate"] == "pass"]
    if candidates:
        chosen = min(candidates, key=lambda row: (row["latency_mean_ms_batch"], row["fid_standard"], row["fp16_layer_count"]))
        return chosen, "quality and speed gates passed"

    variants = [row for row in valid if row["strategy"] not in {"fp32", "all_int8"}]
    if not variants:
        return None, "no mixed-precision variant succeeded"
    def fallback_score(row):
        fid_penalty = max(0.0, row["fid_standard"] - fp32["fid_standard"]) / max(args.max_fid_delta, 1e-6)
        blur_penalty = max(0.0, row["blur_delta_pp_vs_fp32"]) / max(args.max_blur_delta_pp, 1e-6)
        speed_penalty = max(0.0, row["latency_mean_ms_batch"] / int8["latency_mean_ms_batch"] - args.max_latency_ratio)
        return fid_penalty + blur_penalty + speed_penalty
    chosen = min(variants, key=lambda row: (fallback_score(row), row["latency_mean_ms_batch"]))
    return chosen, "no row passed all gates; selected minimum normalized penalty for follow-up"


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    packaged_protocol = prepare_task3_archives(args.task3_bundle, output_dir)
    raw_path = locate_unique("generator_fp32_raw.onnx", args.raw_onnx)
    fp32_engine = locate_unique("generator_trt_fp32.engine", args.fp32_engine)
    int8_engine = locate_unique("generator_trt_int8.engine", args.int8_engine)
    protocol_source = packaged_protocol if packaged_protocol is not None else locate_protocol(args.protocol_path)
    protocol_dir = extract_protocol(protocol_source, output_dir)

    latent_calibration = protocol_dir / "latent_calibration.npy"
    latent_eval_path = protocol_dir / "latent_eval.npy"
    if not latent_calibration.is_file() or not latent_eval_path.is_file():
        raise FileNotFoundError("03A protocol must contain latent_calibration.npy and latent_eval.npy")
    latent_eval = np.load(latent_eval_path).astype(np.float32, copy=False)
    if latent_eval.ndim != 4 or tuple(latent_eval.shape[1:]) != (128, 1, 1):
        raise ValueError(f"Unexpected latent_eval shape: {latent_eval.shape}")
    n_fid = min(args.n_fid, len(latent_eval))
    n_image_eval = min(args.n_image_eval, n_fid)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run 04B on a Kaggle GPU accelerator.")
    device = torch.device("cuda")
    trt = ensure_tensorrt()
    ensure_modelopt()
    runtime = SimpleNamespace(
        TensorRTEngine=TensorRTEngine,
        StandardFID=StandardFID,
        load_real_images=load_real_images,
        stats=stats,
        fid_from_stats=fid_from_stats,
        laplacian_values=laplacian_values,
    )

    named_source = output_dir / "sensitivity_source_named.onnx"
    candidates, source_node_count = make_named_source(raw_path, named_source, args.layers)
    requested = [label for label in args.layers if label in candidates]
    if not requested:
        raise RuntimeError(f"None of the requested layers matched the graph. Candidates: {sorted(candidates)}")
    strategies = []
    for strategy in expand_combinations(args.combinations, requested):
        parse_strategy(strategy, requested)
        if strategy not in strategies:
            strategies.append(strategy)

    real_batches = list(runtime.load_real_images(protocol_dir / "real_eval", n_fid, args.batch_size))
    standard_fid_net = runtime.StandardFID(device)
    real_features = np.concatenate(
        [standard_fid_net.features(images) for images, _paths in real_batches], axis=0
    )[:n_fid]
    real_mu, real_cov = runtime.stats(real_features)
    real_metric = torch.cat([images for images, _paths in real_batches], dim=0)[:n_image_eval]
    blur_threshold = float(np.percentile(runtime.laplacian_values(real_metric), 10))

    rows = []
    evaluated_payload = {}
    build_records = []
    for strategy in strategies:
        restored = parse_strategy(strategy, requested)
        record = {
            "strategy": strategy,
            "restored_layers": "+".join(restored),
            "fp16_layer_count": len(restored),
            "excluded_nodes": ";".join(candidates[label] for label in restored),
            "status": "build_failed",
            "error": "",
            "engine": "",
            "quantized_onnx": "",
            "quality_gate": "pending",
        }
        try:
            quant_path = None
            if strategy == "fp32":
                engine_path = fp32_engine
                qdq_info = {}
            elif strategy == "all_int8":
                engine_path = int8_engine
                qdq_info = {}
            else:
                variant_dir = output_dir / "variants" / slug(strategy)
                variant_dir.mkdir(parents=True, exist_ok=True)
                quant_path = variant_dir / f"generator_{slug(strategy)}.onnx"
                engine_path = variant_dir / f"generator_trt_{slug(strategy)}.engine"
                qdq_info = quantize_combination(
                    named_source,
                    latent_calibration,
                    quant_path,
                    [candidates[label] for label in restored],
                    args.calibration_method,
                )
                build_qdq_engine(
                    trt,
                    quant_path,
                    engine_path,
                    args.opt_batch,
                    args.max_batch,
                    args.workspace_gb,
                )
                record["quantized_onnx"] = quant_path.name
            evaluated = evaluate_engine(
                engine_path,
                strategy,
                trt,
                runtime,
                device,
                latent_eval,
                real_mu,
                real_cov,
                real_metric,
                blur_threshold,
                standard_fid_net,
                args.batch_size,
                n_fid,
                n_image_eval,
                args.warmup,
                args.iters,
            )
            evaluated_payload[strategy] = evaluated
            record.update(
                {
                    "status": "ok",
                    "engine": engine_path.name,
                    "fid_standard": float(evaluated["fid_standard"]),
                    "blur_rate": float(evaluated["blur_rate"]),
                    "laplacian_mean": float(evaluated["laplacian_mean"]),
                    "latency_mean_ms_batch": float(evaluated["latency_mean_ms_batch"]),
                    "throughput_images_per_s": float(evaluated["throughput_images_per_s"]),
                    "cuda_used_after_engine_mb": float(evaluated["cuda_used_after_engine_mb"]),
                    "qdq_quantize_linear": qdq_info.get("quantize_linear", ""),
                    "qdq_dequantize_linear": qdq_info.get("dequantize_linear", ""),
                }
            )
            build_records.append(
                {
                    "strategy": strategy,
                    "status": "ok",
                    "engine_path": str(engine_path),
                    "quantized_onnx_path": str(quant_path) if quant_path is not None else "",
                    "qdq": qdq_info,
                }
            )
            print(f"[04B] {strategy}: ok")
        except Exception as exc:
            record["error"] = repr(exc)
            print(f"[04B] {strategy}: failed: {exc}")
        rows.append(record)

    fp32_payload = evaluated_payload.get("fp32")
    int8_payload = evaluated_payload.get("all_int8")
    if fp32_payload is None or int8_payload is None:
        raise RuntimeError("04B must successfully evaluate both FP32 and all_int8 references")
    fp32_fid = float(fp32_payload["fid_standard"])
    fp32_blur = float(fp32_payload["blur_rate"])
    int8_latency = float(int8_payload["latency_mean_ms_batch"])
    fp32_fake = fp32_payload["fake_metric"].numpy()
    int8_fake = int8_payload["fake_metric"].numpy()
    for row in rows:
        if row.get("status") != "ok":
            continue
        strategy = row["strategy"]
        fake = evaluated_payload[strategy]["fake_metric"].numpy()
        row["fid_delta_vs_fp32"] = row["fid_standard"] - fp32_fid
        row["blur_delta_pp_vs_fp32"] = (row["blur_rate"] - fp32_blur) * 100.0
        row["fid_recovery_vs_int8"] = float(int8_payload["fid_standard"]) - row["fid_standard"]
        row["blur_recovery_pp_vs_int8"] = (float(int8_payload["blur_rate"]) - row["blur_rate"]) * 100.0
        row["speed_ratio_vs_int8"] = row["latency_mean_ms_batch"] / int8_latency
        row["mae_vs_fp32"] = float(np.mean(np.abs(fake - fp32_fake)))
        row["rmse_vs_fp32"] = float(np.sqrt(np.mean((fake - fp32_fake) ** 2)))
        row["mae_vs_int8"] = float(np.mean(np.abs(fake - int8_fake)))

    chosen, selection_reason = choose_recommendation(rows, args)
    if chosen:
        chosen["recommended"] = "yes"
        selected_strategy = chosen["strategy"]
    else:
        selected_strategy = ""
    for row in rows:
        row.setdefault("recommended", "no")

    fieldnames = [
        "strategy", "restored_layers", "fp16_layer_count", "excluded_nodes", "status", "error",
        "quality_gate", "recommended", "engine", "quantized_onnx",
        "fid_standard", "fid_delta_vs_fp32", "blur_rate", "blur_delta_pp_vs_fp32",
        "fid_recovery_vs_int8", "blur_recovery_pp_vs_int8", "laplacian_mean",
        "latency_mean_ms_batch", "throughput_images_per_s", "speed_ratio_vs_int8",
        "cuda_used_after_engine_mb", "mae_vs_fp32", "rmse_vs_fp32", "mae_vs_int8",
        "qdq_quantize_linear", "qdq_dequantize_linear",
    ]
    csv_rows = [{key: row.get(key, "") for key in fieldnames} for row in rows]
    write_csv(output_dir / "mixed_precision_summary.csv", fieldnames, csv_rows)
    plot_created = make_plot(csv_rows, output_dir / "mixed_precision_summary.png")

    recommended_dir = output_dir / "recommended"
    recommended_dir.mkdir(parents=True, exist_ok=True)
    selected_build = next((item for item in build_records if item["strategy"] == selected_strategy), None)
    copied_artifacts = []
    if selected_build:
        source_engine = Path(selected_build["engine_path"])
        target_engine = recommended_dir / "generator_trt_mixed_precision_best.engine"
        shutil.copy2(source_engine, target_engine)
        copied_artifacts.append(target_engine.name)
        source_onnx = selected_build.get("quantized_onnx_path", "")
        if source_onnx:
            source_onnx_path = Path(source_onnx)
            target_onnx = recommended_dir / "generator_mixed_precision_best.onnx"
            shutil.copy2(source_onnx_path, target_onnx)
            copied_artifacts.append(target_onnx.name)
    recommendation_text = [
        "04B mixed-precision recommendation",
        f"selected_strategy={selected_strategy or 'none'}",
        f"selection_reason={selection_reason}",
        f"screen_n_fid={n_fid}",
        "Quality gate: FID <= FP32 + %.4f; blur delta <= %.4f percentage points; latency <= INT8 * %.4f."
        % (args.max_fid_delta, args.max_blur_delta_pp, args.max_latency_ratio),
        "This is a screening recommendation. Run the same script with --n-fid 5000 before using it as the final report result.",
    ]
    (output_dir / "recommended_strategy.txt").write_text("\n".join(recommendation_text) + "\n", encoding="utf-8")

    manifest = {
        "task": "Task4_04B_Mixed_Precision_Combinations",
        "status": "complete" if selected_strategy else "not_passed",
        "method": "Independent mixed-precision experiment: same named raw ONNX + ModelOpt INT8 PTQ; selected ConvTranspose nodes excluded together and retained as FP16; each Q/DQ graph compiled by TensorRT.",
        "graph_scope": "Current Generator graph contains ConvTranspose/BatchNorm/ReLU/Tanh. No wavelet or dynamic-SN node is assumed.",
        "inputs": {
            "raw_onnx": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "fp32_engine": fp32_engine.name,
            "int8_engine": int8_engine.name,
            "protocol": str(protocol_source),
        },
        "candidates": candidates,
        "combinations_requested": strategies,
        "design_rationale": {
            "net.0+net.12": "Primary combination: net.0 had the lowest single-layer speed cost and net.12 had the strongest single-layer FID recovery in the preceding screening.",
            "net.9+net.12": "Interaction test for net.9's blur improvement versus its FID degradation.",
            "net.0+net.9": "No-final-layer control for whether net.9's blur behavior is independent of the output block.",
            "net.0+net.9+net.12": "Three-layer complementary strategy candidate.",
            "net.0+net.3+net.12": "Overhead control: tests whether adding a weak single-layer candidate is justified.",
        },
        "evaluation": {
            "n_fid": n_fid,
            "n_image_eval": n_image_eval,
            "batch_size": args.batch_size,
            "real_blur_threshold_p10": blur_threshold,
            "calibration_method": args.calibration_method,
        },
        "selection_policy": {
            "max_fid_delta": args.max_fid_delta,
            "max_blur_delta_pp": args.max_blur_delta_pp,
            "max_latency_ratio_vs_int8": args.max_latency_ratio,
            "selected_strategy": selected_strategy,
            "reason": selection_reason,
        },
        "successful_count": sum(row.get("status") == "ok" for row in rows),
        "failed_count": sum(row.get("status") != "ok" for row in rows),
        "artifacts": [
            "mixed_precision_summary.csv",
            "mixed_precision_summary.png",
            "recommended_strategy.txt",
            "sensitivity_source_named.onnx",
            "recommended/",
        ],
        "recommended_copied_artifacts": copied_artifacts,
        "final_note": "For the report, rerun with --n-fid 5000 and retain this CSV, plot, manifest, recommendation, and selected engine as the final evidence.",
        "plot_created": plot_created,
    }
    (output_dir / "mixed_precision_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    archive = output_dir.parent / "Task4_04B_Mixed_Precision.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for file_path in output_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(output_dir)
            if relative.parts and relative.parts[0] in {"_task3_inputs", "protocol_03a"}:
                continue
            handle.write(file_path, arcname=f"04B_Mixed_Precision/{relative.as_posix()}")

    print(f"[04B] successful={manifest['successful_count']} failed={manifest['failed_count']}")
    print(f"[04B] selected_strategy={selected_strategy or 'none'}")
    print(f"[zip] {archive}")
    if not selected_strategy:
        raise RuntimeError("04B produced no usable mixed-precision recommendation")


if __name__ == "__main__":
    main()
