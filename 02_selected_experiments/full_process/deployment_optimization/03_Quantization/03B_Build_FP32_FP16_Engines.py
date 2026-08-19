"""
Task 3 / 03B
Build the FP32 and FP16 TensorRT reference engines from the same 01A raw ONNX.

This script does not quantize INT8 and does not evaluate image quality.  It
creates the two reference engines required by 03D.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path


NOISE_DIM = 128
TRT_LOGGER = None


def parse_args():
    p = argparse.ArgumentParser(description="Build Task 3 FP32 and FP16 TensorRT engines")
    p.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/03_Quantization/03B_FP32_FP16"))
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--workspace-gb", type=float, default=2.0)
    args, _unknown = p.parse_known_args()
    if args.opt_batch < 1 or args.max_batch < args.opt_batch:
        raise ValueError("Require 1 <= opt-batch <= max-batch")
    return args


def ensure_trt(torch):
    try:
        import tensorrt as trt
        print(f"[TensorRT] found {trt.__version__}")
        return trt
    except (ImportError, ModuleNotFoundError):
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator.")
    major = str(torch.version.cuda or "").split(".", 1)[0]
    package = {"11": "tensorrt-cu11", "12": "tensorrt-cu12", "13": "tensorrt-cu13"}.get(major)
    if not package:
        raise RuntimeError(f"Unsupported CUDA version: {torch.version.cuda}")
    print(f"[TensorRT] installing {package}")
    command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", package]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        subprocess.check_call(command + ["--extra-index-url", "https://pypi.nvidia.com"])
    importlib.invalidate_caches()
    import tensorrt as trt
    print(f"[TensorRT] installed {trt.__version__}")
    return trt


def locate_file(name: str, explicit: str) -> Path:
    if explicit and Path(explicit).exists():
        return Path(explicit).resolve()
    found = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            found.extend(root.rglob(name))
    unique = sorted({p.resolve() for p in found})
    if len(unique) == 1:
        print(f"[auto-detect] {name}: {unique[0]}")
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(f"Multiple {name} files found; set the explicit argument:\n" + "\n".join(map(str, unique)))
    raise FileNotFoundError(f"{name} not found. Attach the 01A ONNX output Dataset.")


def ensure_fp16_onnx(raw_path: Path, output_dir: Path) -> Path:
    target = output_dir / "generator_fp16_explicit.onnx"
    if target.exists():
        return target
    missing = []
    for module_name, package in (("onnx", "onnx"), ("onnxconverter_common", "onnxconverter-common")):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *sorted(set(missing))])
    import onnx
    from onnxconverter_common import float16
    converted = float16.convert_float_to_float16(onnx.load(str(raw_path)), keep_io_types=False)
    onnx.save(converted, str(target))
    return target


def get_logger(trt):
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return TRT_LOGGER


def build_engine(trt, model_path: Path, engine_path: Path, opt_batch: int, max_batch: int, workspace_gb: float):
    logger = get_logger(trt)
    builder = trt.Builder(logger)
    creation_flags = getattr(trt, "NetworkDefinitionCreationFlag", None)
    if creation_flags is not None and hasattr(creation_flags, "EXPLICIT_BATCH") and not hasattr(creation_flags, "STRONGLY_TYPED"):
        flags = 1 << int(creation_flags.EXPLICIT_BATCH)
    else:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    input_tensor = network.get_input(0)
    input_name = input_tensor.name
    builder_config = builder.create_builder_config()
    workspace = int(workspace_gb * (1024 ** 3))
    memory_pool_type = getattr(trt, "MemoryPoolType", None)
    if hasattr(builder_config, "set_memory_pool_limit") and memory_pool_type is not None:
        builder_config.set_memory_pool_limit(memory_pool_type.WORKSPACE, workspace)
    else:
        builder_config.max_workspace_size = workspace
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, NOISE_DIM, 1, 1), (opt_batch, NOISE_DIM, 1, 1), (max_batch, NOISE_DIM, 1, 1))
    builder_config.add_optimization_profile(profile)
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
        engine_path.write_bytes(bytes(serialized))
    else:
        engine = builder.build_engine(network, builder_config)
        if engine is None:
            raise RuntimeError(f"TensorRT failed to build {engine_path.name}")
        engine_path.write_bytes(bytes(engine.serialize()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    args = parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator.")
    trt = ensure_trt(torch)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = locate_file("generator_fp32_raw.onnx", args.raw_onnx)
    fp16_onnx = ensure_fp16_onnx(raw_path, output_dir)
    fp32_engine = output_dir / "generator_trt_fp32.engine"
    fp16_engine = output_dir / "generator_trt_fp16.engine"
    build_engine(trt, raw_path, fp32_engine, args.opt_batch, args.max_batch, args.workspace_gb)
    print(f"[ok] {fp32_engine}")
    build_engine(trt, fp16_onnx, fp16_engine, args.opt_batch, args.max_batch, args.workspace_gb)
    print(f"[ok] {fp16_engine}")
    manifest = {
        "task": "Task3_03B_FP32_FP16",
        "raw_onnx": str(raw_path),
        "raw_onnx_sha256": sha256(raw_path),
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": str(trt.__version__),
        "opt_batch": args.opt_batch,
        "max_batch": args.max_batch,
        "workspace_gb": args.workspace_gb,
        "engines": {
            "FP32": {"file": fp32_engine.name, "sha256": sha256(fp32_engine), "bytes": fp32_engine.stat().st_size},
            "FP16": {"file": fp16_engine.name, "sha256": sha256(fp16_engine), "bytes": fp16_engine.stat().st_size},
        },
        "comparison_rule": "Both engines are built from the same 01A raw ONNX; no 01B/01C fusion graph is used.",
    }
    (output_dir / "fp32_fp16_build_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output_dir.parent / "Task3_03B_FP32_FP16_Engines.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("generator_fp16_explicit.onnx", "generator_trt_fp32.engine", "generator_trt_fp16.engine", "fp32_fp16_build_manifest.json"):
            z.write(output_dir / name, arcname=name)
    print(f"[zip] {archive}")


if __name__ == "__main__":
    main()
