"""
Task 2B — TensorRT GPU FP32/FP16 benchmark.

Builds TensorRT engines from the validated 01A raw ONNX and tests dynamic
Batch Size 1/4/8/16. If the Kaggle GPU image does not include the TensorRT
Python bindings, the script installs the matching full package automatically.
"""

import argparse
import csv
import importlib
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np


NOISE_DIM = 128
IMAGE_SIZE = 64
TRT_LOGGER = None


def get_trt_logger(trt):
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return TRT_LOGGER


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 2B TensorRT benchmark")
    p.add_argument("--raw-onnx", default=os.getenv("RAW_ONNX_PATH", ""))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark/02B_TensorRT"))
    p.add_argument("--batches", default="1,4,8,16,32")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--workspace-gb", type=float, default=2.0)
    args, _ = p.parse_known_args(argv)
    args.batches = sorted({int(x) for x in args.batches.split(",") if x.strip()})
    if not args.batches or min(args.batches) < 1:
        raise ValueError("batches must contain positive integers")
    return args


def locate_raw(path_text):
    if path_text and Path(path_text).exists():
        return Path(path_text)
    found = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            found.extend(root.rglob("generator_fp32_raw.onnx"))
    unique = sorted({p.resolve() for p in found})
    if len(unique) == 1:
        print("[raw-onnx] auto-detected", unique[0])
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError("Multiple raw ONNX files found; set RAW_ONNX_PATH explicitly:\n" + "\n".join(map(str, unique)))
    raise FileNotFoundError("generator_fp32_raw.onnx not found. Attach the 01A output Dataset.")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def ensure_tensorrt(torch):
    """Install and import TensorRT so 02B remains a single Kaggle script."""
    try:
        import tensorrt as trt
        print(f"[TensorRT] found {trt.__version__}")
        return trt
    except (ImportError, ModuleNotFoundError):
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator before running 02B.")
    cuda_text = str(torch.version.cuda or "")
    cuda_major = cuda_text.split(".", 1)[0]
    if cuda_major == "11":
        package = "tensorrt-cu11"
    elif cuda_major == "12":
        package = "tensorrt-cu12"
    elif cuda_major == "13":
        package = "tensorrt-cu13"
    else:
        raise RuntimeError(f"Unsupported/unknown PyTorch CUDA version: {cuda_text!r}")
    print(f"[TensorRT] missing; installing {package}")
    command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", package]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        print("[TensorRT] retrying NVIDIA package index")
        subprocess.check_call(command + ["--extra-index-url", "https://pypi.nvidia.com"])
    importlib.invalidate_caches()
    try:
        import tensorrt as trt
    except Exception as exc:
        raise RuntimeError(
            "TensorRT was installed but cannot be imported in this kernel. "
            "Restart the Kaggle session once and rerun this same script. "
            f"Import error: {exc}"
        ) from exc
    print(f"[TensorRT] installed {trt.__version__}")
    return trt


def ensure_fp16_onnx(raw_path, output_dir):
    """Create an explicitly FP16 ONNX for TensorRT 11 strong typing."""
    fp16_path = Path(output_dir) / "generator_fp16_explicit.onnx"
    if fp16_path.exists():
        return fp16_path
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
    # keep_io_types=False makes input/output types explicit as FP16, which is
    # required for TensorRT 11 strong typing instead of the removed FP16 flag.
    converted = float16.convert_float_to_float16(onnx.load(str(raw_path)), keep_io_types=False)
    onnx.save(converted, str(fp16_path))
    return fp16_path


def build_engine(model_path, engine_path, precision, batches, workspace_gb):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError("TensorRT is not installed in this runtime. Select a TensorRT-capable GPU image.") from exc
    logger = get_trt_logger(trt)
    builder = trt.Builder(logger)
    # TensorRT <= 8 requires EXPLICIT_BATCH. TensorRT 10/11 removed the
    # flag because networks are always explicit-batch; passing 0 is correct.
    creation_flags = getattr(trt, "NetworkDefinitionCreationFlag", None)
    # TensorRT 10/11 networks are explicit-batch by default.  STRONGLY_TYPED
    # is deliberately not forced: some Kaggle TRT wheels expose the enum but
    # reject otherwise valid ONNX graphs when it is enabled.  TRT 8 still
    # needs EXPLICIT_BATCH.
    if creation_flags is not None and hasattr(creation_flags, "EXPLICIT_BATCH") and not hasattr(creation_flags, "STRONGLY_TYPED"):
        flags = 1 << int(creation_flags.EXPLICIT_BATCH)
    else:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    raw_bytes = Path(model_path).read_bytes()
    if not parser.parse(raw_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    workspace = int(workspace_gb * (1024 ** 3))
    memory_pool_type = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and memory_pool_type is not None:
        config.set_memory_pool_limit(memory_pool_type.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    if precision == "FP16":
        # TensorRT 11 removed BuilderFlag.FP16. The explicit FP16 ONNX types
        # are used instead. TensorRT <= 10 can still use the legacy flag when
        # it exists, but it is unnecessary for an explicitly typed graph.
        builder_flags = getattr(trt, "BuilderFlag", None)
        fp16_flag = getattr(builder_flags, "FP16", None) if builder_flags is not None else None
        if fp16_flag is not None and "generator_fp16_explicit" not in str(model_path):
            config.set_flag(fp16_flag)
    profiling_verbosity = getattr(trt, "ProfilingVerbosity", None)
    if profiling_verbosity is not None and hasattr(config, "profiling_verbosity"):
        config.profiling_verbosity = getattr(profiling_verbosity, "DETAILED", profiling_verbosity.LAYER_NAMES_ONLY)
    profile = builder.create_optimization_profile()
    profile.set_shape("z", (1, NOISE_DIM, 1, 1), (max(1, min(8, max(batches))), NOISE_DIM, 1, 1), (max(batches), NOISE_DIM, 1, 1))
    config.add_optimization_profile(profile)
    if hasattr(builder, "build_serialized_network"):
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build_serialized_network returned None")
        Path(engine_path).write_bytes(bytes(serialized))
    else:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("TensorRT build_engine returned None")
        Path(engine_path).write_bytes(bytes(engine.serialize()))
    return engine_path


def get_io_names(engine):
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_name = next(name for name in names if engine.get_tensor_mode(name).name == "INPUT")
        output_name = next(name for name in names if engine.get_tensor_mode(name).name == "OUTPUT")
        return input_name, output_name
    names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
    input_name = next(name for i, name in enumerate(names) if engine.binding_is_input(i))
    output_name = next(name for i, name in enumerate(names) if not engine.binding_is_input(i))
    return input_name, output_name


def infer(context, engine, input_name, output_name, input_tensor, output_tensor, stream_obj):
    stream = stream_obj.cuda_stream
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, tuple(input_tensor.shape))
        context.set_tensor_address(input_name, int(input_tensor.data_ptr()))
        context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
        ok = context.execute_async_v3(stream)
    else:
        input_index = engine.get_binding_index(input_name)
        output_index = engine.get_binding_index(output_name)
        context.set_binding_shape(input_index, tuple(input_tensor.shape))
        bindings = [0] * engine.num_bindings
        bindings[input_index] = int(input_tensor.data_ptr())
        bindings[output_index] = int(output_tensor.data_ptr())
        ok = context.execute_async_v2(bindings, stream)
    if not ok:
        raise RuntimeError("TensorRT execution returned False")


def cuda_memory_snapshot(torch):
    """Return whole-device CUDA memory, including allocations outside PyTorch."""
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    total_mb = float(total_bytes / (1024 ** 2))
    free_mb = float(free_bytes / (1024 ** 2))
    return {"total_mb": total_mb, "free_mb": free_mb, "used_mb": total_mb - free_mb}


def benchmark_engine(engine_path, batches, warmup, iters):
    import tensorrt as trt
    global torch
    import torch
    device_before = cuda_memory_snapshot(torch)
    runtime = trt.Runtime(get_trt_logger(trt))
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {engine_path}")
    device_after_engine = cuda_memory_snapshot(torch)
    input_name, output_name = get_io_names(engine)
    rows = []
    layer_rows = []
    profiler_supported = False
    stream_obj = torch.cuda.Stream()
    output_np_dtype = trt.nptype(engine.get_tensor_dtype(output_name)) if hasattr(engine, "get_tensor_dtype") else np.float32
    input_np_dtype = trt.nptype(engine.get_tensor_dtype(input_name)) if hasattr(engine, "get_tensor_dtype") else np.float32
    if input_np_dtype not in (np.float16, np.float32):
        raise RuntimeError(f"Unsupported TensorRT input dtype: {input_np_dtype}")
    output_torch_dtype = torch.float16 if output_np_dtype == np.float16 else torch.float32
    for batch in batches:
        noise = np.random.default_rng(42 + batch).standard_normal(
            (batch, NOISE_DIM, 1, 1), dtype=np.float32
        ).astype(input_np_dtype)
        input_tensor = torch.from_numpy(noise).cuda(non_blocking=False)
        output_tensor = torch.empty((batch, 3, IMAGE_SIZE, IMAGE_SIZE), device="cuda", dtype=output_torch_dtype)
        context = engine.create_execution_context()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            infer(context, engine, input_name, output_name, input_tensor, output_tensor, stream_obj)
        stream_obj.synchronize()
        device_after_warmup = cuda_memory_snapshot(torch)
        values = []
        end_to_end = []
        for _ in range(iters):
            start = time.perf_counter()
            infer(context, engine, input_name, output_name, input_tensor, output_tensor, stream_obj)
            stream_obj.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
            start_e2e = time.perf_counter()
            host_noise = torch.from_numpy(noise).cuda()
            infer(context, engine, input_name, output_name, host_noise, output_tensor, stream_obj)
            stream_obj.synchronize()
            _ = output_tensor.cpu()
            end_to_end.append((time.perf_counter() - start_e2e) * 1000.0)
        values = np.asarray(values, dtype=np.float64)
        end_to_end = np.asarray(end_to_end, dtype=np.float64)
        device_after_iters = cuda_memory_snapshot(torch)
        device_peak_used_mb = max(
            device_before["used_mb"],
            device_after_engine["used_mb"],
            device_after_warmup["used_mb"],
            device_after_iters["used_mb"],
        )
        mean_ms = float(end_to_end.mean())
        rows.append({
            "batch": batch,
            "engine_only_mean_ms": float(values.mean()),
            "engine_only_p50_ms": float(np.percentile(values, 50)),
            "engine_only_p95_ms": float(np.percentile(values, 95)),
            "end_to_end_mean_ms": mean_ms,
            "end_to_end_p50_ms": float(np.percentile(end_to_end, 50)),
            "end_to_end_p95_ms": float(np.percentile(end_to_end, 95)),
            "throughput_images_per_s": float(batch / (mean_ms / 1000.0)),
            "peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024 ** 2)),
            "torch_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024 ** 2)),
            "cuda_total_mb": device_after_iters["total_mb"],
            "cuda_used_before_engine_mb": device_before["used_mb"],
            "cuda_used_after_engine_mb": device_after_engine["used_mb"],
            "cuda_used_after_warmup_mb": device_after_warmup["used_mb"],
            "cuda_used_after_iters_mb": device_after_iters["used_mb"],
            "cuda_peak_used_snapshot_mb": device_peak_used_mb,
            "cuda_engine_delta_mb": device_after_engine["used_mb"] - device_before["used_mb"],
        })
        # Profile one separate context after timing so IProfiler overhead does
        # not contaminate the end-to-end benchmark rows.
        profiler_base = getattr(trt, "IProfiler", None)
        profiler_supported_for_context = False
        profiler = None
        if profiler_base is not None:
            class TRTLayerProfiler(profiler_base):
                def __init__(self):
                    profiler_base.__init__(self)
                    self.records = []

                def report_layer_time(self, layer_name, ms):
                    self.records.append([str(layer_name), float(ms)])

            profiler = TRTLayerProfiler()
            profile_context = engine.create_execution_context()
            try:
                profile_context.profiler = profiler
                infer(profile_context, engine, input_name, output_name, input_tensor, output_tensor, stream_obj)
                stream_obj.synchronize()
                if hasattr(profile_context, "report_to_profiler"):
                    profile_context.report_to_profiler()
                    stream_obj.synchronize()
                profiler_supported_for_context = bool(profiler.records)
            except Exception:
                profiler_supported_for_context = False
        profiler_supported = profiler_supported or profiler_supported_for_context
        if profiler_supported_for_context:
            for layer_name, ms in profiler.records:
                layer_rows.append({
                    "engine": Path(engine_path).name,
                    "batch": batch,
                    "layer": layer_name,
                    "time_ms": ms,
                })
            profiler.records.clear()
    return rows, layer_rows, profiler_supported


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
        trt = ensure_tensorrt(torch)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator.")
    except ImportError as exc:
        raise RuntimeError("PyTorch is unavailable in this Kaggle runtime.") from exc
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = locate_raw(args.raw_onnx)
    fp16_model_path = ensure_fp16_onnx(raw_path, output_dir)
    all_rows = []
    all_layer_rows = []
    for precision in ("FP32", "FP16"):
        engine_path = output_dir / f"generator_trt_{precision.lower()}.engine"
        model_path = raw_path if precision == "FP32" else fp16_model_path
        build_engine(model_path, engine_path, precision, args.batches, args.workspace_gb)
        rows, layer_rows, profiler_supported = benchmark_engine(engine_path, args.batches, args.warmup, args.iters)
        all_layer_rows.extend(layer_rows)
        for row in rows:
            row.update({"precision": precision, "status": "ok"})
            all_rows.append(row)
        print(f"[ok] TensorRT {precision}: {len(rows)} batch results")
    header = ["precision", "batch", "engine_only_mean_ms", "engine_only_p50_ms", "engine_only_p95_ms", "end_to_end_mean_ms", "end_to_end_p50_ms", "end_to_end_p95_ms", "throughput_images_per_s", "peak_allocated_mb", "torch_peak_reserved_mb", "cuda_total_mb", "cuda_used_before_engine_mb", "cuda_used_after_engine_mb", "cuda_used_after_warmup_mb", "cuda_used_after_iters_mb", "cuda_peak_used_snapshot_mb", "cuda_engine_delta_mb", "status"]
    write_csv(output_dir / "tensorrt_benchmark.csv", header, [[row.get(k, "") for k in header] for row in all_rows])
    write_csv(output_dir / "tensorrt_layer_profile.csv", ["engine", "batch", "layer", "time_ms"], [[row.get(k, "") for k in ("engine", "batch", "layer", "time_ms")] for row in all_layer_rows])
    archive = output_dir.parent / "Task2_02B_TensorRT.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                z.write(file_path, arcname=f"02B_TensorRT/{file_path.relative_to(output_dir).as_posix()}")
    print("[zip]", archive)


if __name__ == "__main__":
    main()
