"""
Task 4 / 04A - layer-wise INT8 sensitivity screening.

For every ConvTranspose block, quantize the same raw ONNX model with ModelOpt
while excluding exactly one node from quantization.  With
high_precision_dtype=fp16, the excluded node is restored to FP16 and all
other eligible layers remain INT8.  Each variant is compiled by TensorRT and
evaluated with the same latent/evaluation protocol as Task 3.

This is deliberately a screening experiment, not the final mixed-precision
policy.  04B will use the resulting CSV to test combinations such as
"backbone INT8 + last block FP16".  If ModelOpt/TensorRT rejects a variant,
the script records the failure instead of silently claiming a restoration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
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
DEFAULT_LAYER_LABELS = ("net.0", "net.3", "net.6", "net.9", "net.12")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="04A layer-wise INT8 sensitivity screening")
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
        help="Optional 03A/03B/03C ZIP or directory. If omitted, the three Task-3 ZIPs are auto-detected under Kaggle input.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/04_Quantization_Sensitivity/04A_Layer_Sensitivity",
        ),
    )
    parser.add_argument("--layers", default=','.join(DEFAULT_LAYER_LABELS))
    parser.add_argument("--calibration-method", choices=("entropy", "max"), default="entropy")
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--n-fid", type=int, default=1000, help="Use 5000 for the final report confirmation")
    parser.add_argument("--n-image-eval", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260814)
    args, _unknown = parser.parse_known_args(argv)
    args.layers = [item.strip() for item in str(args.layers).split(",") if item.strip()]
    if not args.layers:
        raise ValueError("--layers must contain at least one layer label")
    if min(args.n_fid, args.n_image_eval, args.batch_size, args.opt_batch, args.max_batch) <= 0:
        raise ValueError("counts and batch sizes must be positive")
    if args.n_image_eval > args.n_fid:
        raise ValueError("n-image-eval cannot exceed n-fid")
    if args.max_batch < max(args.opt_batch, args.batch_size):
        raise ValueError("max-batch must be >= both opt-batch and batch-size")
    if args.warmup < 0 or args.iters < 2:
        raise ValueError("warmup must be nonnegative and iters must be at least 2")
    return args


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
            found.extend(root.rglob(name))
    return sorted({path.resolve() for path in found if path.is_file()})


def locate_engine(name: str, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit engine path does not exist: {path}")
        return path
    candidates = _engine_candidates(name)
    if not candidates:
        raise FileNotFoundError(f"{name} was not found. Attach the Task-3 03B/03C package or set the explicit path.")
    by_digest = {}
    for path in candidates:
        by_digest.setdefault(sha256(path), []).append(path)
    if len(by_digest) > 1:
        details = "\n".join(str(path) for path in candidates)
        raise RuntimeError(f"Multiple different {name} files found; set the explicit path:\n{details}")
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
    zips = []
    dirs = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.exists():
            continue
        zips.extend(root.rglob("Task3_03A_Quantization_Protocol.zip"))
        dirs.extend(candidate.parent for candidate in root.rglob("latent_eval.npy") if _valid_protocol_dir(candidate.parent))
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
        self.path = path
        self.trt = trt
        self.device = device
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


def _find_zip(search_root: Path, filename: str):
    if not search_root.exists():
        return []
    try:
        return sorted({path.resolve() for path in search_root.rglob(filename) if path.is_file()})
    except PermissionError:
        return []


def prepare_task3_archives(explicit: str, output_dir: Path):
    """Extract the portable 03A/03B/03C packages when the user uploaded ZIPs.

    The original 03 scripts deliberately package their outputs as ZIP files.
    04A should consume those packages directly instead of asking the user to
    type three engine paths again.  Existing extracted engine files still work
    through the evaluator's normal auto-detection.
    """
    archive_paths = []
    if explicit:
        supplied = Path(explicit).expanduser().resolve()
        if supplied.is_file() and supplied.suffix.lower() == ".zip":
            archive_paths = [supplied]
        elif supplied.is_dir():
            archive_paths = [
                path for name in (
                    "Task3_03A_Quantization_Protocol.zip",
                    "Task3_03B_FP32_FP16_Engines.zip",
                    "Task3_03C_INT8_PTQ_Engine.zip",
                ) for path in _find_zip(supplied, name)
            ]
    else:
        for name in (
            "Task3_03A_Quantization_Protocol.zip",
            "Task3_03B_FP32_FP16_Engines.zip",
            "Task3_03C_INT8_PTQ_Engine.zip",
        ):
            found = []
            for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
                found.extend(_find_zip(root, name))
            if found:
                # Identical copies are the same logical package. If different
                # packages exist, leave the conflict to the explicit-path
                # error rather than silently choosing a version.
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
        protocol_candidates = [
            latent.parent for latent in target.rglob("latent_eval.npy")
            if (latent.parent / "real_eval").is_dir()
        ]
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
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "nvidia-modelopt[onnx]",
            "--extra-index-url",
            "https://pypi.nvidia.com",
        ]
    )
    importlib.invalidate_caches()
    import modelopt  # noqa: F401


def locate_unique(name: str, explicit: str) -> Path:
    """Accept identical duplicate uploads, reject conflicting engines."""
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
                candidates.extend(root.rglob("generator_fp32_raw.onnx"))
                candidates.extend(root.rglob("generator.onnx"))
        unique = sorted({path.resolve() for path in candidates if path.is_file()})
        if len(unique) == 1:
            print(f"[input] {name}: {unique[0]}")
            return unique[0]
        if len(unique) > 1:
            by_digest = {}
            for path in unique:
                by_digest.setdefault(sha256(path), []).append(path)
            if len(by_digest) == 1:
                chosen = sorted(next(iter(by_digest.values())), key=lambda path: (0 if str(path).startswith("/kaggle/input") else 1, str(path)))[0]
                print(f"[input] {name}: {chosen} (identical copies collapsed)")
                return chosen
            raise RuntimeError("Multiple different raw ONNX files found; set --raw-onnx explicitly:\n" + "\n".join(map(str, unique)))
        raise FileNotFoundError("generator_fp32_raw.onnx or generator.onnx was not found")
    return locate_engine(name)


def protocol_root(source: str, output_dir: Path) -> Path:
    return extract_protocol(locate_protocol(source), output_dir)


def make_named_source(raw_path: Path, output_path: Path, requested_labels: list[str]):
    """Give ConvTranspose nodes stable names so ModelOpt can exclude one node.

    The original exporter usually names nodes as ``net.3/ConvTranspose``.
    Naming them explicitly also makes the experiment reproducible if a Kaggle
    upload was produced by a different ONNX exporter version.
    """
    import onnx

    model = onnx.load(str(raw_path))
    onnx.checker.check_model(model)
    candidates = [node for node in model.graph.node if node.op_type == "ConvTranspose"]
    if not candidates:
        raise RuntimeError("No ConvTranspose nodes found in raw ONNX; 04A has no candidate layers to test.")
    labels = list(requested_labels)
    while len(labels) < len(candidates):
        labels.append(f"net.{len(labels) * 3}")
    if len(labels) > len(candidates):
        print(f"[warn] requested {len(labels)} labels but graph has {len(candidates)} ConvTranspose nodes; extra labels are ignored")
        labels = labels[: len(candidates)]
    candidates_by_label = {}
    for index, node in enumerate(candidates):
        label = labels[index]
        node.name = f"{label}.ConvTranspose"
        candidates_by_label[label] = node.name
    onnx.save(model, str(output_path))
    onnx.checker.check_model(onnx.load(str(output_path)))
    return candidates_by_label, len(model.graph.node)


def quantize_variant(raw_path: Path, latent_path: Path, output_path: Path, node_name: str, method: str):
    """Generate INT8 Q/DQ graph with exactly one ConvTranspose excluded.

    ModelOpt documents ``nodes_to_exclude`` as a regex list.  The exact-match
    regex prevents a block name such as net.3 from accidentally excluding
    net.30 in a larger model.
    """
    import onnx

    model = onnx.load(str(raw_path))
    input_name = model.graph.input[0].name
    calibration = np.load(latent_path).astype(np.float32, copy=False)
    from modelopt.onnx.quantization import quantize

    pattern = f"^{re.escape(node_name)}$"
    quantize(
        onnx_path=str(raw_path),
        quantize_mode="int8",
        calibration_data={input_name: calibration},
        calibration_method=method,
        nodes_to_exclude=[pattern],
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
        "node_to_restore": node_name,
        "excluded_pattern": pattern,
        "node_count": len(quantized.graph.node),
        "quantize_linear": counts.get("QuantizeLinear", 0),
        "dequantize_linear": counts.get("DequantizeLinear", 0),
    }


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
    import torch

    engine = evaluator_module.TensorRTEngine(engine_path, trt, device)
    for _ in range(warmup):
        engine.infer(latent_eval[: min(batch_size, n_fid)])
    all_features = []
    metric_parts = []
    timing_ms = []
    for start in range(0, n_fid, batch_size):
        batch = latent_eval[start : start + batch_size]
        t0 = time.perf_counter()
        fake = engine.infer(batch)
        elapsed = (time.perf_counter() - t0) * 1000.0
        timing_ms.append(elapsed)
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
    # The benchmark is intentionally repeated with the same shape so this
    # timing column is not a single noisy first-call observation.
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


def save_sensitivity_plot(rows: list[dict], path: Path):
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row.get("status") == "ok" and row.get("restored_layer") not in ("none", "all_int8")]
    if not valid:
        print("[plot] no successful layer variants; skipping curve")
        return False
    valid.sort(key=lambda row: int(row["layer_index"]))
    x = [int(row["layer_index"]) for row in valid]
    fid_loss = [float(row["fid_delta_vs_fp32"]) for row in valid]
    blur_loss = [float(row["blur_delta_pp_vs_fp32"]) for row in valid]
    fig, ax1 = plt.subplots(figsize=(8, 4.8), dpi=160)
    line1 = ax1.plot(x, fid_loss, marker="o", color="#1f77b4", label="FID delta vs FP32")
    ax1.set_xlabel("ConvTranspose layer index")
    ax1.set_ylabel("FID delta (positive = worse)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xticks(x)
    ax2 = ax1.twinx()
    line2 = ax2.plot(x, blur_loss, marker="s", color="#d62728", label="Blur-rate delta (percentage points)")
    ax2.set_ylabel("Blur-rate delta (percentage points)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax1.grid(alpha=0.25)
    ax1.legend(line1 + line2, [item.get_label() for item in line1 + line2], loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # 04A is self-contained. The portable Task-3 ZIPs provide the engines and
    # the 03A protocol; the original 03C/03D source files are not required.
    packaged_protocol = prepare_task3_archives(args.task3_bundle, output_dir)
    raw_path = locate_unique("generator_fp32_raw.onnx", args.raw_onnx)
    fp32_engine = locate_unique("generator_trt_fp32.engine", args.fp32_engine)
    int8_engine = locate_unique("generator_trt_int8.engine", args.int8_engine)
    if packaged_protocol is not None:
        protocol_source = packaged_protocol
    else:
        protocol_source = locate_protocol(args.protocol_path)
    protocol_dir = extract_protocol(protocol_source, output_dir)
    runtime = SimpleNamespace(
        TensorRTEngine=TensorRTEngine,
        StandardFID=StandardFID,
        load_real_images=load_real_images,
        stats=stats,
        fid_from_stats=fid_from_stats,
        laplacian_values=laplacian_values,
    )
    latent_calibration = protocol_dir / "latent_calibration.npy"
    latent_eval_path = protocol_dir / "latent_eval.npy"
    if not latent_calibration.is_file() or not latent_eval_path.is_file():
        raise FileNotFoundError("03A protocol must contain latent_calibration.npy and latent_eval.npy")
    latent_eval = np.load(latent_eval_path).astype(np.float32, copy=False)
    if latent_eval.ndim != 4 or tuple(latent_eval.shape[1:]) != (NOISE_DIM, 1, 1):
        raise ValueError(f"Unexpected latent_eval shape: {latent_eval.shape}")
    n_fid = min(args.n_fid, len(latent_eval))
    n_image_eval = min(args.n_image_eval, n_fid)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run 04A on a Kaggle GPU accelerator.")
    device = torch.device("cuda")
    trt = ensure_tensorrt()
    ensure_modelopt()

    named_source = output_dir / "sensitivity_source_named.onnx"
    candidates, source_node_count = make_named_source(raw_path, named_source, args.layers)
    requested = [label for label in args.layers if label in candidates]
    if not requested:
        raise RuntimeError(f"None of the requested layers matched the graph. Candidates: {sorted(candidates)}")

    real_batches = list(runtime.load_real_images(protocol_dir / "real_eval", n_fid, args.batch_size))
    standard_fid_net = runtime.StandardFID(device)
    real_features = np.concatenate(
        [standard_fid_net.features(images) for images, _paths in real_batches], axis=0
    )[:n_fid]
    real_mu, real_cov = runtime.stats(real_features)
    real_metric = torch.cat([images for images, _paths in real_batches], dim=0)[:n_image_eval]
    real_lap = runtime.laplacian_values(real_metric)
    blur_threshold = float(np.percentile(real_lap, 10))

    baseline_results = {}
    for label, engine_path in (("FP32", fp32_engine), ("all_int8", int8_engine)):
        baseline_results[label] = evaluate_engine(
            engine_path,
            label,
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

    results = [
        {
            "restored_layer": "none",
            "layer_index": -1,
            "excluded_node": "",
            "status": "ok",
            **{key: value for key, value in baseline_results["FP32"].items() if key not in ("fake_metric", "fake_features")},
        },
        {
            "restored_layer": "all_int8",
            "layer_index": -2,
            "excluded_node": "",
            "status": "ok",
            **{key: value for key, value in baseline_results["all_int8"].items() if key not in ("fake_metric", "fake_features")},
        },
    ]
    variant_records = []
    for layer_index, label in enumerate(requested):
        node_name = candidates[label]
        record = {
            "restored_layer": label,
            "layer_index": layer_index,
            "excluded_node": node_name,
            "status": "build_failed",
            "error": "",
        }
        variant_dir = output_dir / f"variants_{label.replace('.', '_')}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        quant_path = variant_dir / f"generator_int8_restore_{label.replace('.', '_')}.onnx"
        engine_path = variant_dir / f"generator_trt_int8_restore_{label.replace('.', '_')}.engine"
        try:
            qdq_info = quantize_variant(named_source, latent_calibration, quant_path, node_name, args.calibration_method)
            # Keep the Q/DQ network construction local. This is the same
            # explicit-batch TensorRT path as 03C, without importing 03C.py.
            build_qdq_engine(
                trt,
                quant_path,
                engine_path,
                args.opt_batch,
                args.max_batch,
                args.workspace_gb,
            )
            evaluated = evaluate_engine(
                engine_path,
                label,
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
            record.update({key: value for key, value in evaluated.items() if key not in ("fake_metric", "fake_features")})
            record["quantized_onnx"] = quant_path.name
            record["engine"] = engine_path.name
            record["qdq"] = qdq_info
            record["fake_metric"] = evaluated["fake_metric"]
            record["status"] = "ok"
        except Exception as exc:
            record["error"] = repr(exc)
            print(f"[warn] {label} failed: {exc}")
        variant_records.append(record)
        results.append(record)

    fp32_ref = baseline_results["FP32"]["fake_metric"].numpy()
    int8_ref = baseline_results["all_int8"]["fake_metric"].numpy()
    fp32_row = results[0]
    int8_row = results[1]
    for row in results:
        if row.get("status") != "ok":
            continue
        metric = fp32_ref if row["restored_layer"] == "none" else (
            int8_ref if row["restored_layer"] == "all_int8" else next(
                item["fake_metric"].numpy() for item in variant_records if item["restored_layer"] == row["restored_layer"] and item.get("status") == "ok"
            )
        )
        row["mae_vs_fp32"] = float(np.mean(np.abs(metric - fp32_ref)))
        row["rmse_vs_fp32"] = float(np.sqrt(np.mean((metric - fp32_ref) ** 2)))
        row["mae_vs_int8"] = float(np.mean(np.abs(metric - int8_ref)))
        row["fid_delta_vs_fp32"] = float(row["fid_standard"] - fp32_row["fid_standard"])
        row["blur_delta_pp_vs_fp32"] = float((row["blur_rate"] - fp32_row["blur_rate"]) * 100.0)
        row["fid_recovery_vs_int8"] = float(int8_row["fid_standard"] - row["fid_standard"])
        row["blur_recovery_pp_vs_int8"] = float((int8_row["blur_rate"] - row["blur_rate"]) * 100.0)
        row.pop("fake_metric", None)
        row.pop("fake_features", None)

    fieldnames = [
        "restored_layer", "layer_index", "excluded_node", "status", "error", "engine",
        "fid_standard", "fid_delta_vs_fp32", "blur_rate", "blur_delta_pp_vs_fp32",
        "fid_recovery_vs_int8", "blur_recovery_pp_vs_int8",
        "laplacian_mean", "latency_mean_ms_batch", "throughput_images_per_s",
        "cuda_used_after_engine_mb", "mae_vs_fp32", "rmse_vs_fp32", "mae_vs_int8",
        "quantized_onnx",
    ]
    csv_rows = [{key: row.get(key, "") for key in fieldnames} for row in results]
    write_csv(output_dir / "layer_sensitivity_summary.csv", fieldnames, csv_rows)
    plot_created = save_sensitivity_plot(csv_rows, output_dir / "layer_index_precision_loss.png")

    successful_variants = [row for row in csv_rows if row["status"] == "ok" and row["restored_layer"] not in ("none", "all_int8")]
    manifest = {
        "task": "Task4_04A_Layer_Sensitivity",
        "status": "complete" if successful_variants else "blocked",
        "graph_scope": "Current Generator graph has ConvTranspose/BatchNorm/ReLU/Tanh; no wavelet or dynamic-SN node is claimed or tested.",
        "method": "ModelOpt INT8 PTQ on the named raw graph, one exact ConvTranspose node excluded per variant, excluded node restored to FP16.",
        "raw_onnx": {"path": str(raw_path), "sha256": sha256(raw_path)},
        "named_source": {"file": named_source.name, "sha256": sha256(named_source), "node_count": source_node_count},
        "candidates": candidates,
        "requested_layers": requested,
        "calibration": {"file": latent_calibration.name, "count": int(np.load(latent_calibration, mmap_mode="r").shape[0]), "method": args.calibration_method},
        "evaluation": {"n_fid": n_fid, "n_image_eval": n_image_eval, "batch_size": args.batch_size, "real_blur_threshold_p10": blur_threshold},
        "baselines": {"FP32": fp32_engine.name, "INT8": int8_engine.name},
        "artifacts": ["layer_sensitivity_summary.csv", "layer_index_precision_loss.png", "sensitivity_source_named.onnx"],
        "next_step": "Use the lowest-loss and fastest rows to define 04B mixed-precision combinations, then confirm with n_fid=5000.",
        "plot_created": plot_created,
    }
    (output_dir / "layer_sensitivity_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    archive = output_dir.parent / "Task4_04A_Layer_Sensitivity.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(output_dir)
                if relative.parts and relative.parts[0] in {"_task3_inputs", "protocol_03a"}:
                    continue
                handle.write(file_path, arcname=f"04A_Layer_Sensitivity/{relative.as_posix()}")
    print(f"[04A] successful_variants={len(successful_variants)}")
    print(f"[zip] {archive}")
    if not successful_variants:
        raise RuntimeError("04A produced no successful FP16-restored variants; inspect layer_sensitivity_manifest.json")


if __name__ == "__main__":
    main()
