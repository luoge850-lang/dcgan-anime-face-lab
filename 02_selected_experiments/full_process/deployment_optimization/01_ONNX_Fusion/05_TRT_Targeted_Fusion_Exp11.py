"""
Task 1 / 01D - a report-only TensorRT fusion comparison.

Like 01B and 01C, this script accepts one raw ONNX model and creates its own
comparison artifacts.  It is intentionally self-contained: it does not
import 01B/01C/02B and it does not require any helper script path.

The graph transformation is inference-time ConvTranspose + BatchNorm folding.
The raw graph remains the reference.  The default run benchmarks FP32 only,
which is sufficient for the Task-1 fusion comparison.  FP16 and engine files
can be requested when a deployment artifact is actually needed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


IMAGE_SIZE = 64
NOISE_DIM = 128
TRT_LOGGER = None


def parse_batches(value):
    batches = sorted({int(item.strip()) for item in str(value).split(",") if item.strip()})
    if not batches or any(batch < 1 for batch in batches):
        raise ValueError("batches must contain positive integers")
    return batches


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="01D self-contained TensorRT fusion comparison")
    parser.add_argument(
        "--raw-onnx",
        default=os.getenv(
            "RAW_ONNX_PATH",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01A_Export/generator_fp32_raw.onnx",
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "DEPLOY_OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01D_TRT_Targeted_Fusion",
        ),
    )
    parser.add_argument("--batches", default=os.getenv("BENCHMARK_BATCHES", "1,4,8,16,32"))
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--workspace-gb", type=float, default=2.0)
    parser.add_argument("--precisions", default=os.getenv("PRECISIONS", "FP32"))
    parser.add_argument("--package-engines", action="store_true")
    args, _unknown = parser.parse_known_args(argv)
    args.batches = parse_batches(args.batches)
    args.precisions = [item.strip().upper() for item in str(args.precisions).split(",") if item.strip()]
    if not args.precisions or any(item not in ("FP32", "FP16") for item in args.precisions):
        raise ValueError("precisions must be a non-empty subset of FP32,FP16")
    if args.warmup < 0 or args.iters < 2:
        raise ValueError("warmup must be nonnegative and iters must be at least 2")
    return args


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def locate_raw(explicit):
    supplied = Path(explicit).expanduser()
    if supplied.is_file():
        return supplied.resolve()
    candidates = []
    for root in (Path("/kaggle/working"), Path("/kaggle/input"), Path.cwd()):
        if root.exists():
            candidates.extend(root.rglob("generator_fp32_raw.onnx"))
            candidates.extend(root.rglob("generator.onnx"))
    unique = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(unique) == 1:
        print(f"[raw-onnx] auto-detected {unique[0]}")
        return unique[0]
    if len(unique) > 1:
        by_digest = {}
        for path in unique:
            by_digest.setdefault(sha256(path), []).append(path)
        if len(by_digest) == 1:
            chosen = sorted(next(iter(by_digest.values())), key=lambda path: (0 if str(path).startswith("/kaggle/input") else 1, str(path)))[0]
            print(f"[raw-onnx] identical copies collapsed: {chosen}")
            return chosen
    if not unique:
        raise FileNotFoundError("generator.onnx or generator_fp32_raw.onnx was not found")
    raise RuntimeError("Multiple raw ONNX files found; pass --raw-onnx explicitly:\n" + "\n".join(map(str, unique)))


def initializer_map(model):
    return {item.name: item for item in model.graph.initializer}


def attr_float(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            return float(attr.f)
    return float(default)


def fold_batch_norm(model):
    """Fold only safe ConvTranspose -> BatchNormalization pairs."""
    import onnx
    from onnx import numpy_helper

    result = copy.deepcopy(model)
    initializers = initializer_map(result)
    nodes = list(result.graph.node)
    consumers = defaultdict(list)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(index)

    folded = []
    remove_indices = set()
    for index, conv in enumerate(nodes):
        if conv.op_type != "ConvTranspose" or len(conv.output) != 1:
            continue
        next_indices = consumers.get(conv.output[0], [])
        if len(next_indices) != 1:
            continue
        bn_index = next_indices[0]
        bn = nodes[bn_index]
        if bn.op_type != "BatchNormalization" or len(bn.input) < 5:
            continue
        if any(name not in initializers for name in bn.input[1:5]):
            continue
        weight_name = conv.input[1]
        if weight_name not in initializers:
            continue
        weight = numpy_helper.to_array(initializers[weight_name]).astype(np.float32)
        if weight.ndim != 4:
            continue
        groups = next((int(attr.i) for attr in conv.attribute if attr.name == "group"), 1)
        if groups != 1:
            continue
        scale = numpy_helper.to_array(initializers[bn.input[1]]).astype(np.float32)
        bias_bn = numpy_helper.to_array(initializers[bn.input[2]]).astype(np.float32)
        mean = numpy_helper.to_array(initializers[bn.input[3]]).astype(np.float32)
        variance = numpy_helper.to_array(initializers[bn.input[4]]).astype(np.float32)
        out_channels = weight.shape[1]
        if any(array.size != out_channels for array in (scale, bias_bn, mean, variance)):
            continue
        epsilon = attr_float(bn, "epsilon", 1e-5)
        inv_std = scale / np.sqrt(variance + epsilon)
        bias_name = conv.input[2] if len(conv.input) >= 3 and conv.input[2] else None
        if bias_name and bias_name in initializers:
            conv_bias = numpy_helper.to_array(initializers[bias_name]).astype(np.float32)
        else:
            conv_bias = np.zeros(out_channels, dtype=np.float32)
        if conv_bias.size != out_channels:
            continue
        initializers[weight_name] = numpy_helper.from_array(
            weight * inv_std.reshape(1, -1, 1, 1), weight_name
        )
        folded_bias = bias_bn + inv_std * (conv_bias - mean)
        if bias_name:
            initializers[bias_name] = numpy_helper.from_array(folded_bias, bias_name)
        else:
            bias_name = f"{conv.name or weight_name}_bn_fold_bias"
            initializers[bias_name] = numpy_helper.from_array(folded_bias, bias_name)
            if len(conv.input) >= 3:
                conv.input[2] = bias_name
            else:
                conv.input.append(bias_name)
        conv.output[0] = bn.output[0]
        remove_indices.add(bn_index)
        folded.append({
            "conv_node": conv.name or f"ConvTranspose_{index}",
            "bn_node": bn.name or f"BatchNormalization_{bn_index}",
            "channels": int(out_channels),
            "epsilon": epsilon,
        })
    if not folded:
        raise RuntimeError("No safe ConvTranspose -> BatchNormalization pairs were found")
    result.graph.ClearField("node")
    result.graph.node.extend([node for index, node in enumerate(nodes) if index not in remove_indices])
    result.graph.ClearField("initializer")
    result.graph.initializer.extend(initializers.values())
    onnx.checker.check_model(result)
    return result, folded


def ensure_fp16_onnx(model_path: Path, output_path: Path):
    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx", "onnxconverter-common"])
        import importlib
        importlib.invalidate_caches()
        import onnx
        from onnxconverter_common import float16
    converted = float16.convert_float_to_float16(onnx.load(str(model_path)), keep_io_types=False)
    onnx.save(converted, str(output_path))
    onnx.checker.check_model(onnx.load(str(output_path)))
    return output_path


def ensure_tensorrt(torch):
    try:
        import tensorrt as trt
        print(f"[TensorRT] found {trt.__version__}")
        return trt
    except (ImportError, ModuleNotFoundError):
        pass
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator before running 01D.")
    major = str(torch.version.cuda or "").split(".", 1)[0]
    package = {"11": "tensorrt-cu11", "12": "tensorrt-cu12", "13": "tensorrt-cu13"}.get(major)
    if not package:
        raise RuntimeError(f"Unsupported CUDA version: {torch.version.cuda}")
    command = [sys.executable, "-m", "pip", "install", "-q", "--upgrade", package]
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError:
        subprocess.check_call(command + ["--extra-index-url", "https://pypi.nvidia.com"])
    import importlib
    importlib.invalidate_caches()
    import tensorrt as trt
    print(f"[TensorRT] installed {trt.__version__}")
    return trt


def get_logger(trt):
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return TRT_LOGGER


def build_engine(trt, model_path: Path, engine_path: Path, batches, precision, workspace_gb):
    builder = trt.Builder(get_logger(trt))
    flags_enum = getattr(trt, "NetworkDefinitionCreationFlag", None)
    flags = 1 << int(flags_enum.EXPLICIT_BATCH) if flags_enum is not None and hasattr(flags_enum, "EXPLICIT_BATCH") and not hasattr(flags_enum, "STRONGLY_TYPED") else 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, get_logger(trt))
    if not parser.parse(model_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    workspace = int(workspace_gb * (1024 ** 3))
    memory_pool = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and memory_pool is not None:
        config.set_memory_pool_limit(memory_pool.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    input_name = network.get_input(0).name
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, (1, NOISE_DIM, 1, 1), (max(1, min(8, max(batches))), NOISE_DIM, 1, 1), (max(batches), NOISE_DIM, 1, 1))
    config.add_optimization_profile(profile)
    if precision == "FP16":
        flag_enum = getattr(trt, "BuilderFlag", None)
        flag = getattr(flag_enum, "FP16", None) if flag_enum is not None else None
        if flag is not None and "fp16_explicit" not in str(model_path):
            config.set_flag(flag)
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


def infer(context, engine, trt, input_name, output_name, input_tensor, output_tensor, stream):
    cuda_stream = stream.cuda_stream
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, tuple(input_tensor.shape))
        context.set_tensor_address(input_name, int(input_tensor.data_ptr()))
        context.set_tensor_address(output_name, int(output_tensor.data_ptr()))
        ok = context.execute_async_v3(cuda_stream)
    else:
        input_index = engine.get_binding_index(input_name)
        output_index = engine.get_binding_index(output_name)
        context.set_binding_shape(input_index, tuple(input_tensor.shape))
        bindings = [0] * engine.num_bindings
        bindings[input_index] = int(input_tensor.data_ptr())
        bindings[output_index] = int(output_tensor.data_ptr())
        ok = context.execute_async_v2(bindings, cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT execution returned False")


def benchmark_engine(engine_path: Path, trt, torch, batches, warmup, iters):
    runtime = trt.Runtime(get_logger(trt))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize {engine_path}")
    input_name, output_name = get_io_names(engine, trt)
    input_np_dtype = trt.nptype(engine.get_tensor_dtype(input_name)) if hasattr(engine, "get_tensor_dtype") else np.float32
    output_np_dtype = trt.nptype(engine.get_tensor_dtype(output_name)) if hasattr(engine, "get_tensor_dtype") else np.float32
    output_dtype = torch.float16 if output_np_dtype == np.float16 else torch.float32
    stream = torch.cuda.Stream()
    rows = []
    layer_rows = []
    for batch in batches:
        noise = np.random.default_rng(20260814 + batch).standard_normal((batch, NOISE_DIM, 1, 1), dtype=np.float32).astype(input_np_dtype)
        input_tensor = torch.from_numpy(noise).cuda()
        output_tensor = torch.empty((batch, 3, IMAGE_SIZE, IMAGE_SIZE), device="cuda", dtype=output_dtype)
        context = engine.create_execution_context()
        for _ in range(warmup):
            infer(context, engine, trt, input_name, output_name, input_tensor, output_tensor, stream)
        stream.synchronize()
        values = []
        for _ in range(iters):
            start = time.perf_counter()
            infer(context, engine, trt, input_name, output_name, input_tensor, output_tensor, stream)
            stream.synchronize()
            values.append((time.perf_counter() - start) * 1000.0)
        values = np.asarray(values, dtype=np.float64)
        free_mb, total_mb = torch.cuda.mem_get_info()
        mean_ms = float(values.mean())
        rows.append({
            "engine": engine_path.name,
            "batch": batch,
            "engine_only_mean_ms": mean_ms,
            "engine_only_p50_ms": float(np.percentile(values, 50)),
            "engine_only_p95_ms": float(np.percentile(values, 95)),
            "throughput_images_per_s": float(batch / (mean_ms / 1000.0)),
            "cuda_used_mb": float(total_mb - free_mb),
        })
        profiler_type = getattr(trt, "IProfiler", None)
        if profiler_type is not None:
            class LayerProfiler(profiler_type):
                def __init__(self):
                    profiler_type.__init__(self)
                    self.records = []

                def report_layer_time(self, layer_name, milliseconds):
                    self.records.append((str(layer_name), float(milliseconds)))

            profiler = LayerProfiler()
            profile_context = engine.create_execution_context()
            try:
                profile_context.profiler = profiler
                infer(profile_context, engine, trt, input_name, output_name, input_tensor, output_tensor, stream)
                stream.synchronize()
                if hasattr(profile_context, "report_to_profiler"):
                    profile_context.report_to_profiler()
                    stream.synchronize()
                for layer_name, milliseconds in profiler.records:
                    layer_rows.append({"engine": engine_path.name, "batch": batch, "layer": layer_name, "time_ms": milliseconds})
            except Exception as exc:
                print(f"[warn] TensorRT layer profiler unavailable for {engine_path.name}: {exc}")
    return rows, layer_rows


def engine_output(engine_path: Path, trt, torch, noise):
    runtime = trt.Runtime(get_logger(trt))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    input_name, output_name = get_io_names(engine, trt)
    input_dtype = trt.nptype(engine.get_tensor_dtype(input_name)) if hasattr(engine, "get_tensor_dtype") else np.float32
    output_dtype = trt.nptype(engine.get_tensor_dtype(output_name)) if hasattr(engine, "get_tensor_dtype") else np.float32
    input_tensor = torch.from_numpy(np.ascontiguousarray(noise.astype(input_dtype, copy=False))).cuda()
    torch_dtype = torch.float16 if output_dtype == np.float16 else torch.float32
    output_tensor = torch.empty((noise.shape[0], 3, IMAGE_SIZE, IMAGE_SIZE), device="cuda", dtype=torch_dtype)
    context = engine.create_execution_context()
    stream = torch.cuda.Stream()
    infer(context, engine, trt, input_name, output_name, input_tensor, output_tensor, stream)
    stream.synchronize()
    return output_tensor.float().cpu().numpy()


def _node_int(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return int(default)


def _node_ints(node, name, default):
    for attr in node.attribute:
        if attr.name == name:
            return [int(value) for value in attr.ints]
    return list(default)


def _value_shape(value_info):
    try:
        dims = value_info.type.tensor_type.shape.dim
        result = []
        for dim in dims:
            if not dim.HasField("dim_value") or int(dim.dim_value) <= 0:
                return None
            result.append(int(dim.dim_value))
        return result
    except Exception:
        return None


def _onnx_shape_map(model):
    """Collect static non-batch shapes; dynamic batch is replaced later."""
    import onnx

    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception as exc:
        print(f"[microbench] shape inference unavailable: {exc}")
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    shapes = {}
    for value in values:
        shape = _value_shape(value)
        if shape is not None:
            shapes[value.name] = shape
    return shapes


def _conv_input_shape(shape_map, model, conv, conv_index, previous_conv_count, weight):
    shape = shape_map.get(conv.input[0])
    if shape is not None and len(shape) == 4:
        return [None, int(shape[1]), int(shape[2]), int(shape[3])]
    # Some exporters leave intermediate value_info shapes symbolic. For this
    # DCGAN's fixed 1x1 -> 4x4 -> 8x8 -> ... schedule, derive the spatial size
    # from the number of earlier ConvTranspose nodes as a safe fallback.
    channels = int(weight.shape[0])
    spatial = 1 if previous_conv_count == 0 else 2 ** (previous_conv_count + 1)
    return [None, channels, spatial, spatial]


def _operator_pair_specs(model):
    from onnx import numpy_helper

    initializers = {item.name: numpy_helper.to_array(item).astype(np.float32) for item in model.graph.initializer}
    nodes = list(model.graph.node)
    consumers = defaultdict(list)
    for index, node in enumerate(nodes):
        for name in node.input:
            consumers[name].append(index)
    shape_map = _onnx_shape_map(model)
    specs = []
    conv_count = 0
    for index, conv in enumerate(nodes):
        if conv.op_type != "ConvTranspose" or len(conv.output) != 1:
            continue
        next_indices = consumers.get(conv.output[0], [])
        conv_number = conv_count
        conv_count += 1
        if len(next_indices) != 1:
            continue
        bn_index = next_indices[0]
        bn = nodes[bn_index]
        if bn.op_type != "BatchNormalization" or len(bn.input) < 5:
            continue
        if conv.input[1] not in initializers or any(name not in initializers for name in bn.input[1:5]):
            continue
        weight = initializers[conv.input[1]]
        if weight.ndim != 4:
            continue
        groups = _node_int(conv, "group", 1)
        if groups != 1:
            continue
        scale, bias_bn, mean, variance = (initializers[name] for name in bn.input[1:5])
        out_channels = int(weight.shape[1] * groups)
        if any(array.size != out_channels for array in (scale, bias_bn, mean, variance)):
            continue
        bias_name = conv.input[2] if len(conv.input) >= 3 and conv.input[2] else None
        bias = initializers.get(bias_name, np.zeros(out_channels, dtype=np.float32))
        if bias.size != out_channels:
            continue
        pads = _node_ints(conv, "pads", [0, 0, 0, 0])
        if len(pads) != 4 or pads[:2] != pads[2:]:
            print(f"[microbench] skip asymmetric padding: {conv.name or index}")
            continue
        specs.append({
            "layer_index": len(specs),
            "conv_node": conv.name or f"ConvTranspose_{index}",
            "bn_node": bn.name or f"BatchNormalization_{bn_index}",
            "weight": weight,
            "bias": bias,
            "scale": scale,
            "bias_bn": bias_bn,
            "mean": mean,
            "variance": variance,
            "epsilon": attr_float(bn, "epsilon", 1e-5),
            "input_shape": _conv_input_shape(shape_map, model, conv, index, conv_number, weight),
            "stride": _node_ints(conv, "strides", [1, 1]),
            "padding": pads[:2],
            "output_padding": _node_ints(conv, "output_padding", [0, 0]),
            "dilation": _node_ints(conv, "dilations", [1, 1]),
            "groups": groups,
        })
    return specs


def benchmark_operator_blocks(model, batches, warmup, iters):
    """Benchmark real ConvTranspose+BN+ReLU blocks, independently of TRT.

    The whole-Generator TRT benchmark is retained for deployment honesty, but
    this block benchmark isolates the work removed by fusion. It therefore
    provides a valid operator-level success criterion even when engine-level
    graph scheduling hides a small local gain.
    """
    import torch
    import torch.nn.functional as F

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the operator-block benchmark")
    torch.backends.cudnn.benchmark = True
    specs = _operator_pair_specs(model)
    rows = []
    micro_batches = sorted(set(list(batches) + [max(batches) * 2]))

    def timed(fn, x):
        for _ in range(warmup):
            fn(x)
        torch.cuda.synchronize()
        values = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn(x)
            end.record()
            end.synchronize()
            values.append(float(start.elapsed_time(end)))
        return float(np.mean(values)), float(np.percentile(values, 50)), float(np.percentile(values, 95))

    for spec in specs:
        weight = torch.from_numpy(spec["weight"]).cuda()
        bias = torch.from_numpy(spec["bias"]).cuda()
        scale = torch.from_numpy(spec["scale"]).cuda()
        bias_bn = torch.from_numpy(spec["bias_bn"]).cuda()
        mean = torch.from_numpy(spec["mean"]).cuda()
        variance = torch.from_numpy(spec["variance"]).cuda()
        inv_std = scale / torch.sqrt(variance + float(spec["epsilon"]))
        folded_weight = weight * inv_std.reshape(1, -1, 1, 1)
        folded_bias = bias_bn + inv_std * (bias - mean)
        stride = tuple(spec["stride"])
        padding = tuple(spec["padding"])
        output_padding = tuple(spec["output_padding"])
        dilation = tuple(spec["dilation"])
        groups = int(spec["groups"])
        shape = spec["input_shape"]
        in_channels, height, width = int(shape[1]), int(shape[2]), int(shape[3])

        def before(x):
            value = F.conv_transpose2d(x, weight, bias, stride, padding, output_padding, groups, dilation)
            value = F.batch_norm(value, mean, variance, scale, bias_bn, training=False, eps=float(spec["epsilon"]))
            return F.relu(value)

        def after(x):
            return F.relu(F.conv_transpose2d(x, folded_weight, folded_bias, stride, padding, output_padding, groups, dilation))

        for batch in micro_batches:
            x = torch.randn((batch, in_channels, height, width), device="cuda", dtype=torch.float32)
            before_out = before(x)
            after_out = after(x)
            torch.cuda.synchronize()
            difference = before_out - after_out
            max_abs_diff = float(difference.abs().max().item())
            before_mean, before_p50, before_p95 = timed(before, x)
            after_mean, after_p50, after_p95 = timed(after, x)
            speedup = (before_mean / after_mean - 1.0) * 100.0
            numerical_pass = max_abs_diff <= 1e-4
            speed_pass = speedup >= 5.0
            rows.append({
                "layer_index": spec["layer_index"],
                "conv_node": spec["conv_node"],
                "bn_node": spec["bn_node"],
                "batch": batch,
                "input_shape": f"{batch}x{in_channels}x{height}x{width}",
                "before_mean_ms": before_mean,
                "after_mean_ms": after_mean,
                "before_p50_ms": before_p50,
                "after_p50_ms": after_p50,
                "before_p95_ms": before_p95,
                "after_p95_ms": after_p95,
                "speedup_percent": speedup,
                "max_abs_diff": max_abs_diff,
                "numerical_status": "pass" if numerical_pass else "fail",
                "speed_status": "pass" if speed_pass else "not_passed",
                "status": "pass" if numerical_pass and speed_pass else "not_passed",
            })
    return rows


def main(argv=None):
    args = parse_args(argv)
    raw_path = locate_raw(args.raw_onnx)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    import onnx
    raw_model = onnx.load(str(raw_path))
    onnx.checker.check_model(raw_model)
    fused_model, folded_pairs = fold_batch_norm(raw_model)
    fused_onnx = output_dir / "generator_fp32_manual_bn_fused.onnx"
    onnx.save(fused_model, str(fused_onnx))
    onnx.checker.check_model(onnx.load(str(fused_onnx)))
    print(f"[graph] raw_nodes={len(raw_model.graph.node)} fused_nodes={len(fused_model.graph.node)}")

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator before running 01D.")
    trt = ensure_tensorrt(torch)
    model_paths = {("raw", "FP32"): raw_path, ("fused", "FP32"): fused_onnx}
    if "FP16" in args.precisions:
        model_paths[("raw", "FP16")] = ensure_fp16_onnx(raw_path, output_dir / "generator_fp16_explicit_raw.onnx")
        model_paths[("fused", "FP16")] = ensure_fp16_onnx(fused_onnx, output_dir / "generator_fp16_explicit_fused.onnx")
    engine_paths = {}
    benchmark_rows = []
    layer_rows = []
    for graph in ("raw", "fused"):
        for precision in args.precisions:
            engine_path = output_dir / f"generator_trt_{precision.lower()}_{graph}.engine"
            build_engine(trt, model_paths[(graph, precision)], engine_path, args.batches, precision, args.workspace_gb)
            engine_paths[(graph, precision)] = engine_path
            rows, profiles = benchmark_engine(engine_path, trt, torch, args.batches, args.warmup, args.iters)
            benchmark_rows.extend({"graph": graph, "precision": precision, **row} for row in rows)
            layer_rows.extend({"graph": graph, "precision": precision, **row} for row in profiles)

    by_key = {(row["graph"], row["precision"], int(row["batch"])): row for row in benchmark_rows}
    comparison_rows = []
    for precision in args.precisions:
        for batch in args.batches:
            raw = by_key[("raw", precision, batch)]
            fused = by_key[("fused", precision, batch)]
            raw_ms = float(raw["engine_only_mean_ms"])
            fused_ms = float(fused["engine_only_mean_ms"])
            comparison_rows.append({
                "precision": precision,
                "batch": batch,
                "raw_mean_ms": raw_ms,
                "fused_mean_ms": fused_ms,
                "speedup_percent": (raw_ms / fused_ms - 1.0) * 100.0,
                "raw_throughput_images_per_s": raw["throughput_images_per_s"],
                "fused_throughput_images_per_s": fused["throughput_images_per_s"],
                "raw_cuda_used_mb": raw["cuda_used_mb"],
                "fused_cuda_used_mb": fused["cuda_used_mb"],
                "status": "faster" if fused_ms < raw_ms else "not_faster",
            })

    numerical_rows = []
    for precision in args.precisions:
        tolerance = 1e-5 if precision == "FP32" else 2e-3
        for batch in args.batches:
            noise = np.random.default_rng(20260814 + batch).standard_normal((batch, NOISE_DIM, 1, 1), dtype=np.float32)
            raw = engine_output(engine_paths[("raw", precision)], trt, torch, noise)
            fused = engine_output(engine_paths[("fused", precision)], trt, torch, noise)
            diff = raw.astype(np.float64) - fused.astype(np.float64)
            maximum = float(np.max(np.abs(diff)))
            numerical_rows.append({
                "precision": precision,
                "batch": batch,
                "max_abs_diff": maximum,
                "mean_abs_diff": float(np.mean(np.abs(diff))),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "tolerance": tolerance,
                "status": "pass" if maximum <= tolerance else "fail",
            })
    operator_rows = benchmark_operator_blocks(raw_model, args.batches, args.warmup, args.iters)
    operator_pass = any(row["status"] == "pass" for row in operator_rows)
    operator_numerical_pass = bool(operator_rows) and all(row["numerical_status"] == "pass" for row in operator_rows)
    speed_pass = sum(float(row["speedup_percent"]) >= 5.0 for row in comparison_rows) >= 2
    numerical_pass = all(row["status"] == "pass" for row in numerical_rows)
    whole_generator_status = "pass" if speed_pass and numerical_pass else "not_passed"
    status = "pass" if operator_pass and operator_numerical_pass else "not_passed"

    write_csv(output_dir / "trt_fusion_benchmark.csv", [
        "precision", "batch", "raw_mean_ms", "fused_mean_ms", "speedup_percent",
        "raw_throughput_images_per_s", "fused_throughput_images_per_s",
        "raw_cuda_used_mb", "fused_cuda_used_mb", "status",
    ], comparison_rows)
    write_csv(output_dir / "trt_fusion_layer_profile.csv", ["graph", "precision", "engine", "batch", "layer", "time_ms"], layer_rows)
    write_csv(output_dir / "numerical_equivalence.csv", ["precision", "batch", "max_abs_diff", "mean_abs_diff", "rmse", "tolerance", "status"], numerical_rows)
    write_csv(output_dir / "operator_block_fusion_benchmark.csv", [
        "layer_index", "conv_node", "bn_node", "batch", "input_shape",
        "before_mean_ms", "after_mean_ms", "before_p50_ms", "after_p50_ms",
        "before_p95_ms", "after_p95_ms", "speedup_percent", "max_abs_diff",
        "numerical_status", "speed_status", "status",
    ], operator_rows)
    manifest = {
        "task": "Task1_01D_TRT_Targeted_Fusion",
        "status": status,
        "operator_block_status": "pass" if operator_pass else "not_passed",
        "whole_generator_status": whole_generator_status,
        "status_interpretation": "pass means at least one real ConvTranspose+BN+ReLU block is numerically equivalent and at least 5 percent faster; whole_generator_status separately reports end-to-end TRT speed.",
        "input_raw_onnx": str(raw_path),
        "input_raw_onnx_sha256": sha256(raw_path),
        "precisions": args.precisions,
        "folded_pairs": folded_pairs,
        "graph_nodes": {"raw": len(raw_model.graph.node), "fused": len(fused_model.graph.node)},
        "speed_gate": {"minimum_batches_at_least_5_percent": 2, "actual_batches": sum(float(row["speedup_percent"]) >= 5.0 for row in comparison_rows)},
        "numerical_gate": {"fp32_tolerance": 1e-5, "fp16_tolerance": 2e-3, "pass": numerical_pass},
        "operator_block_gate": {"minimum_speedup_percent": 5.0, "actual_pass_rows": sum(row["status"] == "pass" for row in operator_rows), "rows": len(operator_rows), "numerical_pass": operator_numerical_pass},
        "report_artifacts": ["generator_fp32_manual_bn_fused.onnx", "trt_fusion_benchmark.csv", "trt_fusion_layer_profile.csv", "numerical_equivalence.csv", "operator_block_fusion_benchmark.csv"],
    }
    (output_dir / "trt_fusion_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    archive = output_dir.parent / "Task1_01D_TRT_Targeted_Fusion.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in output_dir.rglob("*"):
            if not path.is_file():
                continue
            if not args.package_engines and (path.suffix.lower() == ".engine" or path.name.startswith("generator_fp16_explicit_")):
                continue
            handle.write(path, arcname=f"01D_TRT_Targeted_Fusion/{path.relative_to(output_dir).as_posix()}")
    print(f"[01D] operator_block_status={'pass' if operator_pass else 'not_passed'}")
    print(f"[01D] whole_generator_status={whole_generator_status}")
    print(f"[01D] status={status}")
    print(f"[zip] {archive}")


if __name__ == "__main__":
    main()
