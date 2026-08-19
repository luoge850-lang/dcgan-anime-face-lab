"""
Task 4 / 04C - independent final confirmation.

This file is intentionally self-contained.  It does not import 04A, 04B, any
04A/04B result, or any earlier experiment script.  It accepts only the raw
ONNX, the FP32/INT8 TensorRT references, and the original 03A protocol (or the
portable Task-3 ZIPs).  It then rebuilds and evaluates exactly three rows:

    FP32, all INT8, net.0+net.12 mixed precision

The mixed row is regenerated from the raw ONNX with ModelOpt PTQ and exact
ConvTranspose exclusions, so the final confirmation is a fresh experiment.
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

import numpy as np
import torch
from PIL import Image


IMAGE_SIZE = 64
NOISE_DIM = 128
FP16_LAYERS = ("net.0", "net.12")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="04C independent final confirmation")
    parser.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    parser.add_argument("--fp32-engine", default=os.getenv("FP32_ENGINE_PATH", ""))
    parser.add_argument("--int8-engine", default=os.getenv("INT8_ENGINE_PATH", ""))
    parser.add_argument("--protocol-path", default=os.getenv("PROTOCOL_PATH", ""))
    parser.add_argument("--task3-bundle", default=os.getenv("TASK3_BUNDLE_PATH", ""))
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/04_Quantization_Sensitivity/04C_Final_Confirmation",
        ),
    )
    parser.add_argument("--n-fid", type=int, default=5000)
    parser.add_argument("--n-image-eval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--calibration-method", choices=("entropy", "max"), default="entropy")
    parser.add_argument("--max-fid-delta", type=float, default=1.5)
    parser.add_argument("--max-blur-delta-pp", type=float, default=0.5)
    parser.add_argument("--max-latency-ratio", type=float, default=1.75)
    parser.add_argument("--seed", type=int, default=20260817)
    args, _unknown = parser.parse_known_args(argv)
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
        raise RuntimeError("CUDA is unavailable. Run 04C on a Kaggle GPU accelerator.")
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


def ensure_modelopt():
    try:
        import modelopt  # noqa: F401
        print("[ModelOpt] found")
        return
    except (ImportError, ModuleNotFoundError):
        pass
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


def locate_engine(name: str, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit engine path does not exist: {path}")
        return path
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.exists():
            continue
        try:
            candidates.extend(root.rglob(name))
        except PermissionError:
            pass
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        raise FileNotFoundError(f"{name} was not found")
    digests = {}
    for path in candidates:
        digests.setdefault(sha256(path), []).append(path)
    if len(digests) > 1:
        raise RuntimeError("Multiple different engines found; set an explicit path:\n" + "\n".join(map(str, candidates)))
    chosen = sorted(next(iter(digests.values())), key=lambda path: (0 if str(path).startswith("/kaggle/input") else 1, str(path)))[0]
    print(f"[engine] {name}: {chosen}")
    return chosen


def locate_raw_onnx(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Raw ONNX does not exist: {path}")
        return path
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.exists():
            continue
        try:
            candidates.extend(root.rglob("generator_fp32_raw.onnx"))
            candidates.extend(root.rglob("generator.onnx"))
        except PermissionError:
            pass
    candidates = sorted({path.resolve() for path in candidates if path.is_file()})
    if not candidates:
        raise FileNotFoundError("generator_fp32_raw.onnx or generator.onnx was not found")
    digests = {}
    for path in candidates:
        digests.setdefault(sha256(path), []).append(path)
    if len(digests) > 1:
        raise RuntimeError("Multiple different raw ONNX files found; set --raw-onnx explicitly")
    return sorted(next(iter(digests.values())), key=lambda path: (0 if str(path).startswith("/kaggle/input") else 1, str(path)))[0]


def valid_protocol_dir(path: Path) -> bool:
    return path.is_dir() and (path / "latent_eval.npy").is_file() and (path / "real_eval").is_dir()


def locate_protocol(explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.is_file() and path.suffix.lower() == ".zip":
            return path
        if valid_protocol_dir(path):
            return path
        children = [item.parent for item in path.rglob("latent_eval.npy") if valid_protocol_dir(item.parent)] if path.is_dir() else []
        if len(children) == 1:
            return children[0].resolve()
        raise FileNotFoundError(f"Invalid 03A protocol path: {path}")
    zips, dirs = [], []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if not root.exists():
            continue
        try:
            zips.extend(root.rglob("Task3_03A_Quantization_Protocol.zip"))
            dirs.extend(item.parent for item in root.rglob("latent_eval.npy") if valid_protocol_dir(item.parent))
        except PermissionError:
            pass
    zips = sorted({path.resolve() for path in zips if path.is_file()})
    dirs = sorted({path.resolve() for path in dirs})
    if len(zips) == 1:
        return zips[0]
    if len(dirs) == 1:
        return dirs[0]
    if len(zips) > 1 or len(dirs) > 1:
        raise RuntimeError("Multiple 03A protocols found; set --protocol-path explicitly")
    raise FileNotFoundError("Task3_03A_Quantization_Protocol.zip or extracted protocol was not found")


def extract_protocol(source: Path, output_dir: Path) -> Path:
    if source.is_dir():
        if valid_protocol_dir(source):
            return source.resolve()
        children = [item.parent for item in source.rglob("latent_eval.npy") if valid_protocol_dir(item.parent)]
        if len(children) == 1:
            return children[0].resolve()
        raise FileNotFoundError(f"Could not resolve protocol under {source}")
    target = output_dir / "protocol_03a"
    target.mkdir(parents=True, exist_ok=True)
    marker = target / ".extracted"
    if not marker.exists():
        with zipfile.ZipFile(source) as handle:
            handle.extractall(target)
        marker.write_text("ok", encoding="utf-8")
    return extract_protocol(target, output_dir)


def find_zips(root: Path, name: str):
    if not root.exists():
        return []
    try:
        return sorted({path.resolve() for path in root.rglob(name) if path.is_file()})
    except PermissionError:
        return []


def stage_task3_bundle(explicit: str, output_dir: Path):
    names = (
        "Task3_03A_Quantization_Protocol.zip",
        "Task3_03B_FP32_FP16_Engines.zip",
        "Task3_03C_INT8_PTQ_Engine.zip",
    )
    archives = []
    if explicit:
        supplied = Path(explicit).expanduser().resolve()
        if supplied.is_file() and supplied.suffix.lower() == ".zip":
            archives = [supplied]
        elif supplied.is_dir():
            archives = [path for name in names for path in find_zips(supplied, name)]
    else:
        for name in names:
            found = []
            for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
                found.extend(find_zips(root, name))
            if found:
                by_digest = {sha256(path): path for path in found}
                if len(by_digest) > 1:
                    raise RuntimeError(f"Multiple different {name} files found; set --task3-bundle")
                archives.append(next(iter(by_digest.values())))
    unique = []
    seen = set()
    for path in archives:
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        return None
    staging = output_dir / "_task3_inputs"
    staging.mkdir(parents=True, exist_ok=True)
    protocol = None
    for archive in unique:
        target = staging / archive.stem
        marker = target / ".extracted"
        target.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(target)
            marker.write_text("ok", encoding="utf-8")
        options = [item.parent for item in target.rglob("latent_eval.npy") if valid_protocol_dir(item.parent)]
        if len(options) == 1:
            protocol = options[0].resolve()
        print(f"[task3-package] {archive.name} -> {target}")
    return protocol


def io_names(engine, trt):
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        inp = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        out = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        return inp, out
    names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
    inp = next(name for i, name in enumerate(names) if engine.binding_is_input(i))
    out = next(name for i, name in enumerate(names) if not engine.binding_is_input(i))
    return inp, out


class TensorRTEngine:
    def __init__(self, path: Path, trt, device):
        self.path, self.trt, self.device = path, trt, device
        runtime = trt.Runtime(get_logger(trt))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = io_names(self.engine, trt)
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
        block = int(getattr(InceptionV3, "DEFAULT_BLOCK_INDEX", 3))
        self.device = device
        self.net = InceptionV3([block]).eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images01):
        values = self.net(images01.to(self.device))[0]
        return values.reshape(values.shape[0], -1).detach().cpu().numpy()


def load_real_images(real_dir: Path, count: int, batch_size: int):
    paths = sorted(real_dir.glob("*.png"))[:count]
    if len(paths) < count:
        raise RuntimeError(f"real_eval has {len(paths)} PNG files, need {count}")
    for start in range(0, count, batch_size):
        images, batch_paths = [], paths[start : start + batch_size]
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


def make_named_source(raw_path: Path, output_path: Path):
    import onnx
    model = onnx.load(str(raw_path))
    onnx.checker.check_model(model)
    nodes = [node for node in model.graph.node if node.op_type == "ConvTranspose"]
    if len(nodes) < 5:
        raise RuntimeError(f"Expected five ConvTranspose nodes, found {len(nodes)}")
    candidates = {}
    labels = ("net.0", "net.3", "net.6", "net.9", "net.12")
    for label, node in zip(labels, nodes):
        node.name = f"{label}.ConvTranspose"
        candidates[label] = node.name
    onnx.save(model, str(output_path))
    onnx.checker.check_model(onnx.load(str(output_path)))
    return candidates, len(model.graph.node)


def quantize_mixed(named_source: Path, calibration_path: Path, output_path: Path, excluded_nodes: list[str], method: str):
    import onnx
    from modelopt.onnx.quantization import quantize

    model = onnx.load(str(named_source))
    input_name = model.graph.input[0].name
    calibration = np.load(calibration_path).astype(np.float32, copy=False)
    patterns = [f"^{re.escape(node)}$" for node in excluded_nodes]
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
    result = onnx.load(str(output_path))
    onnx.checker.check_model(result)
    counts = {}
    for node in result.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    if counts.get("QuantizeLinear", 0) == 0 or counts.get("DequantizeLinear", 0) == 0:
        raise RuntimeError(f"No Q/DQ nodes found in {output_path}")
    return {"excluded_nodes": excluded_nodes, "quantize_linear": counts.get("QuantizeLinear", 0), "dequantize_linear": counts.get("DequantizeLinear", 0)}


def build_engine(trt, quant_path: Path, engine_path: Path, opt_batch: int, max_batch: int, workspace_gb: float):
    builder = trt.Builder(get_logger(trt))
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, get_logger(trt))
    if not parser.parse(quant_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
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


def evaluate_engine(engine_path: Path, label: str, trt, device, latent_eval, real_mu, real_cov, fid_net, real_metric, blur_threshold, args):
    engine = TensorRTEngine(engine_path, trt, device)
    for _ in range(args.warmup):
        engine.infer(latent_eval[: min(args.batch_size, args.n_fid)])
    features, metric_parts = [], []
    for start in range(0, args.n_fid, args.batch_size):
        batch = latent_eval[start : start + args.batch_size]
        fake = engine.infer(batch)
        fake01 = ((fake + 1.0) * 0.5).clamp(0.0, 1.0)
        features.append(fid_net.features(fake01))
        if start < args.n_image_eval:
            metric_parts.append(fake01[: min(len(fake01), args.n_image_eval - start)])
    fake_features = np.concatenate(features, axis=0)[: args.n_fid]
    fake_mu, fake_cov = stats(fake_features)
    fake_metric = torch.cat(metric_parts, dim=0)[: args.n_image_eval]
    fake_lap = laplacian_values(fake_metric)
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    repeat = []
    repeat_noise = latent_eval[: args.batch_size]
    for _ in range(args.iters):
        start = time.perf_counter()
        engine.infer(repeat_noise)
        repeat.append((time.perf_counter() - start) * 1000.0)
    latency = float(np.mean(repeat))
    return {
        "strategy": label,
        "engine": engine_path.name,
        "fid_standard": fid_from_stats(real_mu, real_cov, fake_mu, fake_cov),
        "blur_rate": float((fake_lap < blur_threshold).mean()),
        "laplacian_mean": float(fake_lap.mean()),
        "latency_mean_ms_batch": latency,
        "throughput_images_per_s": float(args.batch_size / (latency / 1000.0)),
        "cuda_used_bytes": float(total_bytes - free_bytes),
        "fake_metric": fake_metric,
    }


def write_csv(path: Path, rows: list[dict]):
    fields = [
        "strategy", "status", "error", "recommended", "engine", "quantized_onnx",
        "fid_standard", "fid_delta_vs_fp32", "blur_rate", "blur_delta_pp_vs_fp32",
        "fid_recovery_vs_int8", "blur_recovery_pp_vs_int8", "latency_mean_ms_batch",
        "throughput_images_per_s", "speed_ratio_vs_int8", "cuda_used_bytes",
        "mae_vs_fp32", "rmse_vs_fp32", "mae_vs_int8", "qdq_quantize_linear", "qdq_dequantize_linear",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def plot_results(rows: list[dict], path: Path):
    import matplotlib.pyplot as plt

    valid = [row for row in rows if row.get("status") == "ok"]
    labels = [row["strategy"] for row in valid]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=180)
    colors = ["#222222", "#d62728", "#2ca02c"]
    axes[0].bar(x, [row["fid_standard"] for row in valid], color=colors)
    axes[0].set_title("Standard FID")
    axes[0].set_ylabel("lower is better")
    axes[1].bar(x, [row["blur_rate"] * 100.0 for row in valid], color=colors)
    axes[1].set_title("Blur rate")
    axes[1].set_ylabel("percent")
    axes[2].bar(x, [row["latency_mean_ms_batch"] for row in valid], color=colors)
    axes[2].set_title("TensorRT latency")
    axes[2].set_ylabel("ms / batch")
    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("04C Final Confirmation", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run 04C on Kaggle GPU.")
    if args.n_fid < 5000:
        print(f"[warn] final confirmation requested with n_fid={args.n_fid}; recommended value is 5000")

    staged_protocol = stage_task3_bundle(args.task3_bundle, output_dir)
    raw_path = locate_raw_onnx(args.raw_onnx)
    fp32_engine = locate_engine("generator_trt_fp32.engine", args.fp32_engine)
    int8_engine = locate_engine("generator_trt_int8.engine", args.int8_engine)
    protocol_source = staged_protocol if staged_protocol is not None else locate_protocol(args.protocol_path)
    protocol_dir = extract_protocol(protocol_source, output_dir)
    latent_calibration = protocol_dir / "latent_calibration.npy"
    latent_eval_path = protocol_dir / "latent_eval.npy"
    if not latent_calibration.is_file() or not latent_eval_path.is_file():
        raise FileNotFoundError("03A protocol must contain latent_calibration.npy and latent_eval.npy")
    latent_eval = np.load(latent_eval_path).astype(np.float32, copy=False)
    if latent_eval.ndim != 4 or tuple(latent_eval.shape[1:]) != (NOISE_DIM, 1, 1):
        raise ValueError(f"Unexpected latent_eval shape: {latent_eval.shape}")
    args.n_fid = min(args.n_fid, len(latent_eval))
    args.n_image_eval = min(args.n_image_eval, args.n_fid)

    trt = ensure_tensorrt()
    ensure_modelopt()
    device = torch.device("cuda")
    named_source = output_dir / "final_confirmation_named.onnx"
    candidates, node_count = make_named_source(raw_path, named_source)
    real_batches = list(load_real_images(protocol_dir / "real_eval", args.n_fid, args.batch_size))
    fid_net = StandardFID(device)
    real_features = np.concatenate([fid_net.features(images) for images, _paths in real_batches], axis=0)[: args.n_fid]
    real_mu, real_cov = stats(real_features)
    real_metric = torch.cat([images for images, _paths in real_batches], dim=0)[: args.n_image_eval]
    blur_threshold = float(np.percentile(laplacian_values(real_metric), 10))

    payloads = {}
    rows = []
    for strategy, engine_path in (("fp32", fp32_engine), ("all_int8", int8_engine)):
        try:
            payload = evaluate_engine(engine_path, strategy, trt, device, latent_eval, real_mu, real_cov, fid_net, real_metric, blur_threshold, args)
            payloads[strategy] = payload
            rows.append({"strategy": strategy, "status": "ok", "error": "", "recommended": "no", **payload})
        except Exception as exc:
            rows.append({"strategy": strategy, "status": "failed", "error": repr(exc), "recommended": "no"})

    mixed_dir = output_dir / "mixed_variant"
    mixed_dir.mkdir(parents=True, exist_ok=True)
    mixed_onnx = mixed_dir / "generator_mixed_precision_final.onnx"
    mixed_engine = mixed_dir / "generator_trt_mixed_precision_final.engine"
    try:
        qdq = quantize_mixed(
            named_source,
            latent_calibration,
            mixed_onnx,
            [candidates[label] for label in FP16_LAYERS],
            args.calibration_method,
        )
        build_engine(trt, mixed_onnx, mixed_engine, args.opt_batch, args.max_batch, args.workspace_gb)
        payload = evaluate_engine(mixed_engine, "net.0+net.12", trt, device, latent_eval, real_mu, real_cov, fid_net, real_metric, blur_threshold, args)
        payloads["net.0+net.12"] = payload
        rows.append({
            "strategy": "net.0+net.12",
            "status": "ok",
            "error": "",
            "recommended": "no",
            "quantized_onnx": mixed_onnx.name,
            "qdq_quantize_linear": qdq["quantize_linear"],
            "qdq_dequantize_linear": qdq["dequantize_linear"],
            **payload,
        })
    except Exception as exc:
        rows.append({"strategy": "net.0+net.12", "status": "failed", "error": repr(exc), "recommended": "no"})

    fp32 = payloads.get("fp32")
    int8 = payloads.get("all_int8")
    mixed = payloads.get("net.0+net.12")
    if fp32 and int8 and mixed:
        fp32_fake = fp32["fake_metric"].numpy()
        int8_fake = int8["fake_metric"].numpy()
        for row in rows:
            if row.get("status") != "ok":
                continue
            payload = payloads[row["strategy"]]
            row["fid_delta_vs_fp32"] = payload["fid_standard"] - fp32["fid_standard"]
            row["blur_delta_pp_vs_fp32"] = (payload["blur_rate"] - fp32["blur_rate"]) * 100.0
            row["fid_recovery_vs_int8"] = int8["fid_standard"] - payload["fid_standard"]
            row["blur_recovery_pp_vs_int8"] = (int8["blur_rate"] - payload["blur_rate"]) * 100.0
            row["speed_ratio_vs_int8"] = payload["latency_mean_ms_batch"] / int8["latency_mean_ms_batch"]
            fake = payload["fake_metric"].numpy()
            row["mae_vs_fp32"] = float(np.mean(np.abs(fake - fp32_fake)))
            row["rmse_vs_fp32"] = float(np.sqrt(np.mean((fake - fp32_fake) ** 2)))
            row["mae_vs_int8"] = float(np.mean(np.abs(fake - int8_fake)))
        selected = next(row for row in rows if row["strategy"] == "net.0+net.12")
        pass_quality = selected["fid_delta_vs_fp32"] <= args.max_fid_delta and selected["blur_delta_pp_vs_fp32"] <= args.max_blur_delta_pp
        pass_speed = selected["speed_ratio_vs_int8"] <= args.max_latency_ratio
        selected["recommended"] = "yes" if pass_quality and pass_speed else "no"
    else:
        pass_quality = pass_speed = False

    write_csv(output_dir / "final_confirmation_summary.csv", rows)
    plot_results(rows, output_dir / "final_confirmation_summary.png")
    recommendation = next((row for row in rows if row.get("strategy") == "net.0+net.12"), {})
    recommended_dir = output_dir / "recommended"
    recommended_dir.mkdir(parents=True, exist_ok=True)
    if mixed_onnx.is_file():
        shutil.copy2(mixed_onnx, recommended_dir / "generator_mixed_precision_best.onnx")
    if mixed_engine.is_file():
        shutil.copy2(mixed_engine, recommended_dir / "generator_trt_mixed_precision_best.engine")
    final_manifest = {
        "task": "Task4_04C_Final_Confirmation",
        "status": "complete" if recommendation.get("recommended") == "yes" else "not_passed",
        "independence": "This file does not import or consume 04A/04B scripts or result files.",
        "inputs": {
            "raw_onnx": {"path": str(raw_path), "sha256": sha256(raw_path)},
            "fp32_engine": fp32_engine.name,
            "int8_engine": int8_engine.name,
            "protocol": str(protocol_source),
        },
        "graph": {"named_source": named_source.name, "node_count": node_count, "fp16_layers": list(FP16_LAYERS)},
        "evaluation": {"n_fid": args.n_fid, "n_image_eval": args.n_image_eval, "batch_size": args.batch_size, "blur_threshold_p10": blur_threshold},
        "selection_policy": {
            "selected_strategy": "net.0+net.12",
            "fid_delta_limit": args.max_fid_delta,
            "blur_delta_pp_limit": args.max_blur_delta_pp,
            "latency_ratio_limit_vs_int8": args.max_latency_ratio,
            "quality_pass": pass_quality,
            "speed_pass": pass_speed,
        },
        "artifacts": ["final_confirmation_summary.csv", "final_confirmation_summary.png", "final_confirmation_manifest.json", "recommended/", "mixed_variant/"],
    }
    (output_dir / "final_confirmation_manifest.json").write_text(json.dumps(final_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = output_dir.parent / "Task4_04C_Final_Confirmation.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for file_path in output_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(output_dir)
            if relative.parts and relative.parts[0] in {"_task3_inputs", "protocol_03a"}:
                continue
            handle.write(file_path, arcname=f"04C_Final_Confirmation/{relative.as_posix()}")
    print(f"[04C] status={final_manifest['status']}")
    print(f"[04C] recommended={recommendation.get('recommended', 'no')}")
    print(f"[zip] {archive}")
    if final_manifest["status"] != "complete":
        raise RuntimeError("04C final confirmation did not pass for net.0+net.12")


if __name__ == "__main__":
    main()
