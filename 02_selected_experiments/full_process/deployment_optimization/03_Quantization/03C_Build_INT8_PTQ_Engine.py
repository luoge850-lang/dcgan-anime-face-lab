"""
Task 3 / 03C - TensorRT 11 compatible INT8 PTQ.

TensorRT 11 removed IInt8Calibrator and config.int8_calibrator.  Therefore
this script uses NVIDIA TensorRT ModelOpt to calibrate offline and insert
QuantizeLinear/DequantizeLinear (Q/DQ) nodes into an INT8 ONNX graph.  The
Q/DQ ONNX graph is then compiled into a strongly typed TensorRT engine.

The calibration input is latent_calibration.npy because the Generator input is
latent z.  The 100 real images in the 03A package remain evaluation/reference
data and are not passed to the Generator.
"""

from __future__ import annotations

import argparse
import csv
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
    parser = argparse.ArgumentParser(description="Build TensorRT 11 INT8 PTQ engine with ModelOpt Q/DQ")
    parser.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    parser.add_argument("--protocol-zip", default=os.getenv("PROTOCOL_ZIP_PATH", ""))
    parser.add_argument("--latent-calibration", default=os.getenv("LATENT_CALIBRATION_PATH", ""))
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/03_Quantization/03C_INT8_PTQ"))
    parser.add_argument("--calibration-method", choices=("entropy", "max"), default="entropy")
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=64)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    args, _unknown = parser.parse_known_args()
    if args.opt_batch < 1 or args.max_batch < args.opt_batch:
        raise ValueError("Require 1 <= opt-batch <= max-batch")
    return args


def get_logger(trt):
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return TRT_LOGGER


def ensure_tensorrt(torch):
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
    command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", package]
    print(f"[TensorRT] installing {package}")
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        subprocess.check_call(command + ["--extra-index-url", "https://pypi.nvidia.com"])
    importlib.invalidate_caches()
    import tensorrt as trt
    print(f"[TensorRT] installed {trt.__version__}")
    return trt


def locate_file(name: str, explicit: str) -> Path | None:
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
        raise RuntimeError(f"Multiple {name} files found; set the explicit path:\n" + "\n".join(map(str, unique)))
    return None


def extract_protocol_if_needed(explicit_zip: str, work_dir: Path) -> Path | None:
    latent = locate_file("latent_calibration.npy", "")
    if latent is not None:
        return latent
    zip_path = locate_file("Task3_03A_Quantization_Protocol.zip", explicit_zip)
    if zip_path is None:
        return None
    extract_dir = work_dir / "protocol_03a"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    latent = extract_dir / "latent_calibration.npy"
    if not latent.exists():
        raise FileNotFoundError(f"latent_calibration.npy not found inside {zip_path}")
    print(f"[protocol] extracted {zip_path}")
    return latent


def ensure_modelopt():
    try:
        import modelopt  # noqa: F401
        print("[ModelOpt] found")
        return
    except (ImportError, ModuleNotFoundError):
        pass
    print("[ModelOpt] installing nvidia-modelopt[onnx]")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "nvidia-modelopt[onnx]",
        "--extra-index-url",
        "https://pypi.nvidia.com",
    ])
    importlib.invalidate_caches()
    import modelopt  # noqa: F401
    print("[ModelOpt] installed")


def validate_latent(path: Path) -> tuple[list[int], int]:
    import numpy as np
    latent = np.load(path, mmap_mode="r")
    expected = (128, 1, 1)
    if latent.ndim != 4 or tuple(latent.shape[1:]) != expected:
        raise ValueError(f"Expected latent shape [N,128,1,1], got {latent.shape}")
    if latent.dtype not in (np.float16, np.float32):
        raise ValueError(f"Expected float16/float32 latent data, got {latent.dtype}")
    return list(latent.shape), int(latent.shape[0])


def validate_real_calibration_set(latent_path: Path) -> tuple[Path, int]:
    """Require the 100-image audit set while keeping latent z as PTQ input."""
    protocol_dir = latent_path.parent
    manifest = protocol_dir / "real_calibration_100_manifest.csv"
    image_dir = protocol_dir / "real_calibration_100"
    if not manifest.exists() or not image_dir.is_dir():
        raise FileNotFoundError(
            "03A real_calibration_100 is missing; attach the complete 03A protocol output."
        )
    with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    image_count = len(list(image_dir.glob("*.png")))
    if len(rows) != 100 or image_count != 100:
        raise RuntimeError(
            f"Expected exactly 100 real calibration images, manifest={len(rows)}, png={image_count}"
        )
    return manifest, image_count


def run_modelopt_quantization(raw_path: Path, latent_path: Path, quant_path: Path, method: str):
    command = [
        sys.executable,
        "-m",
        "modelopt.onnx.quantization",
        "--onnx_path",
        str(raw_path),
        "--quantize_mode",
        "int8",
        "--calibration_data",
        str(latent_path),
        "--calibration_method",
        method,
        "--output_path",
        str(quant_path),
    ]
    print("[ModelOpt]", " ".join(command))
    subprocess.check_call(command)
    if not quant_path.exists() or quant_path.stat().st_size == 0:
        raise RuntimeError("ModelOpt did not create a quantized ONNX file")


def validate_qdq(quant_path: Path) -> dict:
    import onnx
    model = onnx.load(str(quant_path))
    onnx.checker.check_model(model)
    counts = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    q_count = counts.get("QuantizeLinear", 0)
    dq_count = counts.get("DequantizeLinear", 0)
    if q_count == 0 or dq_count == 0:
        raise RuntimeError(f"Quantized ONNX has no Q/DQ nodes: QuantizeLinear={q_count}, DequantizeLinear={dq_count}")
    return {"opset": max((int(opset.version) for opset in model.opset_import), default=0), "node_count": len(model.graph.node), "quantize_linear": q_count, "dequantize_linear": dq_count, "operators": counts}


def build_engine(trt, quant_path: Path, engine_path: Path, opt_batch: int, max_batch: int, workspace_gb: float):
    logger = get_logger(trt)
    builder = trt.Builder(logger)
    # TensorRT 11 networks are strongly typed by default.  TensorRT <=10
    # also accepts a Q/DQ ONNX graph without the old INT8 calibrator API.
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(quant_path.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT Q/DQ ONNX parse failed:\n" + "\n".join(errors))
    input_name = network.get_input(0).name
    config = builder.create_builder_config()
    workspace = int(workspace_gb * (1024 ** 3))
    memory_pool_type = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and memory_pool_type is not None:
        config.set_memory_pool_limit(memory_pool_type.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, NOISE_DIM, 1, 1), (opt_batch, NOISE_DIM, 1, 1), (max_batch, NOISE_DIM, 1, 1))
    config.add_optimization_profile(profile)
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT Q/DQ build returned None")
        engine_path.write_bytes(bytes(serialized))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT Q/DQ build returned None")
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
    trt = ensure_tensorrt(torch)
    ensure_modelopt()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = locate_file("generator_fp32_raw.onnx", args.raw_onnx)
    if raw_path is None:
        raise FileNotFoundError("generator_fp32_raw.onnx not found. Attach the 01A ONNX output Dataset.")
    latent_path = extract_protocol_if_needed(args.protocol_zip, output_dir.parent)
    if args.latent_calibration and Path(args.latent_calibration).exists():
        latent_path = Path(args.latent_calibration).resolve()
    if latent_path is None:
        raise FileNotFoundError("latent_calibration.npy not found. Attach the 03A protocol ZIP or set --latent-calibration.")
    latent_shape, latent_count = validate_latent(latent_path)
    real_calibration_manifest, real_calibration_count = validate_real_calibration_set(latent_path)
    quant_path = output_dir / "generator_int8_qdq.onnx"
    run_modelopt_quantization(raw_path, latent_path, quant_path, args.calibration_method)
    qdq_info = validate_qdq(quant_path)
    engine_path = output_dir / "generator_trt_int8.engine"
    build_engine(trt, quant_path, engine_path, args.opt_batch, args.max_batch, args.workspace_gb)
    if not engine_path.exists() or engine_path.stat().st_size == 0:
        raise RuntimeError("INT8 engine file was not created")
    print(f"[ok] {quant_path}")
    print(f"[ok] {engine_path}")
    manifest = {
        "task": "Task3_03C_INT8_PTQ_TensorRT11_ModelOpt",
        "raw_onnx": str(raw_path),
        "raw_onnx_sha256": sha256(raw_path),
        "latent_calibration": str(latent_path),
        "latent_shape": latent_shape,
        "latent_count": latent_count,
        "real_calibration_manifest": str(real_calibration_manifest),
        "real_calibration_count": real_calibration_count,
        "calibration_input_boundary": "The 100 real anime images are retained as the requested calibration/reference audit set; Generator activation PTQ itself uses latent_calibration.npy because Generator input is z, not RGB images.",
        "calibration_method": args.calibration_method,
        "quantization_method": "TensorRT ModelOpt offline explicit INT8 Q/DQ",
        "quantized_onnx": {"file": quant_path.name, "sha256": sha256(quant_path), "bytes": quant_path.stat().st_size, **qdq_info},
        "opt_batch": args.opt_batch,
        "max_batch": args.max_batch,
        "workspace_gb": args.workspace_gb,
        "cuda": str(torch.version.cuda),
        "gpu": torch.cuda.get_device_name(0),
        "tensorrt": str(trt.__version__),
        "engine": {"file": engine_path.name, "sha256": sha256(engine_path), "bytes": engine_path.stat().st_size},
        "important_boundary": "100 real images are evaluation/reference data; latent arrays calibrate Generator activations.",
        "migration_note": "TensorRT 11 removed IInt8Calibrator and int8_calibrator; Q/DQ explicit quantization is used instead.",
    }
    manifest_path = output_dir / "int8_build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output_dir.parent / "Task3_03C_INT8_PTQ_Engine.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name in ("generator_trt_int8.engine", "generator_int8_qdq.onnx", "int8_build_manifest.json"):
            path = output_dir / name
            if path.exists():
                z.write(path, arcname=name)
    print(f"[zip] {archive}")


if __name__ == "__main__":
    main()
