"""Task 5B: independent QAT export, TensorRT build, and quality audit.

05B consumes the two checkpoints produced by 05A and the already validated
FP32/PTQ/mixed-PTQ engines.  The 04C mixed-PTQ artifact is an engine-only
baseline; it is evaluated by deserialization and is not used as an ONNX
template.  05B does not import 05A and does not depend on a merged 05ABC
artifact.  The two QAT checkpoints are exported independently:

    qat_pre.pth  -> calibrated pre-QAT engine
    qat_best.pth -> trained QAT engine

The final report uses the same 03A latent_eval.npy and real_eval/ as Tasks 3
and 4.  It reports standard 2048-d Inception-v3 FID, the fixed real-image
Laplacian blur rate, Haar high-frequency error against FP32, CUDA latency
percentiles, throughput, and a visual contact sheet.  A strict hair/eyeliner
ROI claim is not made unless annotated ROI data are supplied; the global Haar
detail score is an auditable high-frequency proxy and the contact sheet is the
visual QC artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


IMAGE_SIZE = 64
NOISE_DIM = 128
FP16_LAYERS = ("net.0", "net.12")
INT8_LAYERS = ("net.3", "net.6", "net.9")
CONV_LABELS = ("net.0", "net.3", "net.6", "net.9", "net.12")
# 03B/03C/04C engines in this project were built with the Kaggle TensorRT
# 11.2 runtime.  Keep 05B on the same serialization runtime.  The old 11.2
# ConvTranspose failure came from per-channel QAT weight scales; layout v4
# uses one scalar scale per INT8 ConvTranspose weight.
DEFAULT_TRT_VERSION = "11.2.1.2"
DEFAULT_QAT_OUTPUT_DIR = "/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05A_QAT_Training"
QAT_LAYOUT_VERSION = 4
QAT_TRAINING_REVISION = 3


class Exp11Generator(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 768, 4),
            nn.BatchNorm2d(768), nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1),
            nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def make_activation_fake_quant():
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver

    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128, quant_max=127, dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric, reduce_range=False,
    )


def make_weight_fake_quant():
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver

    return FakeQuantize(
        observer=MovingAverageMinMaxObserver,
        quant_min=-128, quant_max=127, dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric, reduce_range=False,
    )


class QATConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self, *args, quant_enabled: bool, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_enabled = bool(quant_enabled)
        self.input_fake_quant = make_activation_fake_quant() if self.quant_enabled else nn.Identity()
        self.weight_fake_quant = make_weight_fake_quant() if self.quant_enabled else nn.Identity()

    def forward(self, x):
        if self.quant_enabled:
            x = self.input_fake_quant(x)
            weight = self.weight_fake_quant(self.weight)
        else:
            weight = self.weight
        return F.conv_transpose2d(
            x, weight, self.bias, self.stride, self.padding,
            self.output_padding, self.groups, self.dilation,
        )


class HybridQATGenerator(nn.Module):
    def __init__(self, noise_dim: int = NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            QATConvTranspose2d(noise_dim, 768, 4, quant_enabled=False),
            nn.BatchNorm2d(768), nn.ReLU(),
            QATConvTranspose2d(768, 384, 4, 2, 1, quant_enabled=True),
            nn.BatchNorm2d(384), nn.ReLU(),
            QATConvTranspose2d(384, 192, 4, 2, 1, quant_enabled=True),
            nn.BatchNorm2d(192), nn.ReLU(),
            QATConvTranspose2d(192, 96, 4, 2, 1, quant_enabled=True),
            nn.BatchNorm2d(96), nn.ReLU(),
            QATConvTranspose2d(96, 3, 4, 2, 1, quant_enabled=False),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="05B independent QAT deployment and evaluation")
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "--mixed-ptq-onnx" in raw_argv:
        raise ValueError(
            "04C provides generator_trt_mixed_precision_final.engine only; "
            "do not pass --mixed-ptq-onnx. 05B creates its own QAT ONNX graph."
        )
    p.add_argument("--qat-bundle", default=os.getenv("QAT_BUNDLE_PATH", ""), help="Optional 05A ZIP or extracted output folder")
    p.add_argument(
        "--qat-output-dir",
        default=os.getenv("QAT_OUTPUT_DIR", DEFAULT_QAT_OUTPUT_DIR),
        help="05A output directory in the same Kaggle Notebook; contains qat_pre.pth and qat_best.pth",
    )
    p.add_argument("--pre-qat-checkpoint", default=os.getenv("PRE_QAT_CHECKPOINT", ""))
    p.add_argument("--qat-checkpoint", default=os.getenv("QAT_CHECKPOINT", ""))
    p.add_argument("--fp32-engine", default=os.getenv("FP32_ENGINE_PATH", ""))
    p.add_argument("--ptq-int8-engine", default=os.getenv("PTQ_INT8_ENGINE_PATH", ""))
    p.add_argument("--mixed-ptq-engine", default=os.getenv("MIXED_PTQ_ENGINE_PATH", ""))
    p.add_argument("--protocol-path", default=os.getenv("PROTOCOL_PATH", ""), help="03A extracted folder or ZIP")
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05B_Evaluation"))
    p.add_argument("--tensorrt-version", default=os.getenv("TENSORRT_VERSION", DEFAULT_TRT_VERSION))
    p.add_argument(
        "--tensorrt-pip-package",
        default=os.getenv("TENSORRT_PIP_PACKAGE", "tensorrt-cu12"),
        help="TensorRT pip distribution used before the first import; use tensorrt-cu13 for CUDA 13.x",
    )
    p.add_argument("--skip-tensorrt-install", action="store_true", help="Only check the already installed TensorRT; useful on offline Kaggle sessions")
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=64)
    p.add_argument("--n-fid", type=int, default=5000)
    p.add_argument("--n-image-eval", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dynamic-batches", default="1,8,32,64")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--benchmark-repeats", type=int, default=3)
    p.add_argument("--max-speed-ratio-all-int8", type=float, default=1.30, help="Accepted QAT/all-INT8 P99 ratio for the revised no-collapse claim")
    p.add_argument("--max-speed-ratio-mixed", type=float, default=1.15)
    p.add_argument("--max-blur-degradation-vs-ptq-pp", type=float, default=0.5, help="Maximum QAT blur-rate increase over PTQ in percentage points")
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--no-zip", action="store_true")
    args, _ = p.parse_known_args(argv)
    batches = {int(item.strip()) for item in args.dynamic_batches.split(",") if item.strip()}
    batches.add(int(args.batch_size))
    args.dynamic_batches = tuple(sorted(batches))
    if not args.dynamic_batches or min(args.dynamic_batches) < 1:
        raise ValueError("dynamic-batches must contain positive integers")
    if args.opt_batch < 1 or args.max_batch < args.opt_batch:
        raise ValueError("Require 1 <= opt-batch <= max-batch")
    if args.n_fid < 1 or args.n_image_eval < 1 or args.n_image_eval > args.n_fid or args.batch_size < 1:
        raise ValueError("Require 1 <= n-image-eval <= n-fid and positive batch-size")
    if max(args.dynamic_batches) > args.max_batch:
        raise ValueError("Every dynamic batch must be <= max-batch")
    if args.warmup < 0 or args.iters < 10 or args.benchmark_repeats < 1:
        raise ValueError("Require warmup >= 0, iters >= 10 and benchmark-repeats >= 1")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_one(name: str, explicit: str = "") -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return path
    candidates = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            try:
                candidates.extend(root.rglob(name))
            except PermissionError:
                pass
    unique = sorted({item.resolve() for item in candidates if item.is_file()})
    if not unique:
        raise FileNotFoundError(f"{name} not found; pass an explicit path")
    by_hash = {}
    for item in unique:
        by_hash.setdefault(sha256(item), []).append(item)
    if len(by_hash) != 1:
        raise RuntimeError("Multiple different files found; pass an explicit path:\n" + "\n".join(map(str, unique)))
    return sorted(next(iter(by_hash.values())), key=lambda item: (0 if str(item).startswith("/kaggle/input") else 1, str(item)))[0]


def locate_qat_checkpoint(name: str, explicit: str, qat_root: Path | None, qat_output_dir: str) -> Path:
    """Prefer the previous 05A output in the same notebook.

    This prevents a rerun from accidentally selecting an older checkpoint
    uploaded under /kaggle/input when the current 05A result is already in
    /kaggle/working.
    """
    if explicit:
        return find_one(name, explicit)
    roots = []
    if qat_root is not None:
        roots.append(qat_root)
    if qat_output_dir:
        roots.append(Path(qat_output_dir).expanduser().resolve())
    roots.append(Path(DEFAULT_QAT_OUTPUT_DIR))
    for root in roots:
        if not root.exists():
            continue
        direct = root / name
        if direct.is_file():
            return direct.resolve()
        nested = sorted({item.resolve() for item in root.rglob(name) if item.is_file()})
        if len(nested) == 1:
            return nested[0]
        if len(nested) > 1:
            raise RuntimeError(f"Multiple {name} files found under {root}; pass --pre-qat-checkpoint or --qat-checkpoint explicitly")
    return find_one(name)


def extract_bundle(bundle: str, staging: Path) -> Path:
    source = Path(bundle).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"05A bundle not found: {source}")
    if source.is_dir():
        return source
    staging.mkdir(parents=True, exist_ok=True)
    marker = staging / ".zip_sha256"
    digest = sha256(source)
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != digest:
        for child in staging.iterdir():
            if child != marker:
                shutil.rmtree(child) if child.is_dir() else child.unlink()
        with zipfile.ZipFile(source) as archive:
            archive.extractall(staging)
        marker.write_text(digest, encoding="utf-8")
    return staging


def locate_protocol(explicit: str, staging: Path) -> Path:
    def valid(path):
        return (path / "latent_eval.npy").is_file() and (path / "real_eval").is_dir()

    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Protocol path not found: {source}")
        if source.is_file():
            source = extract_bundle(str(source), staging)
        if valid(source):
            return source
        options = [item.parent for item in source.rglob("latent_eval.npy") if valid(item.parent)]
        if len(options) != 1:
            raise RuntimeError(f"Expected one 03A protocol folder under {source}, found {len(options)}")
        return options[0]
    options = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            try:
                options.extend(item.parent for item in root.rglob("latent_eval.npy") if valid(item.parent))
            except PermissionError:
                pass
    unique = sorted({item.resolve() for item in options})
    if len(unique) != 1:
        raise FileNotFoundError("03A protocol was not uniquely found; pass --protocol-path")
    return unique[0]


def load_state(path: Path):
    payload = torch.load(path, map_location="cpu")
    state = payload
    if isinstance(payload, dict):
        layout_version = payload.get("qat_layout_version")
        if layout_version is not None and int(layout_version) != QAT_LAYOUT_VERSION:
            raise RuntimeError(
                f"{path.name} uses qat_layout_version={layout_version}; "
                f"05B requires version {QAT_LAYOUT_VERSION} with scalar ConvTranspose weight scales. "
                "Rerun the rewritten 05A before running 05B."
            )
        training_revision = payload.get("qat_training_revision")
        if training_revision is None or int(training_revision) != QAT_TRAINING_REVISION:
            raise RuntimeError(
                f"{path.name} uses qat_training_revision={training_revision}; "
                f"05B requires revision {QAT_TRAINING_REVISION} with disjoint checkpoint-selection data and the audited core QAT objective. "
                "Rerun the optimized 05A before running 05B."
            )
        candidate = payload.get("model_state_dict")
        if isinstance(candidate, dict):
            state = candidate
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported QAT checkpoint: {type(state)}")
    expected = {"net.0.weight", "net.3.weight", "net.6.weight", "net.9.weight", "net.12.weight"}
    if not expected.issubset(set(state)):
        raise RuntimeError(f"QAT checkpoint does not contain the Exp11 Generator: missing={sorted(expected - set(state))}")
    return state


def load_qat_model(checkpoint: Path):
    model = HybridQATGenerator().cpu().eval()
    state = load_state(checkpoint)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [
        key for key in missing
        if ".input_fake_quant." not in key and ".weight_fake_quant." not in key
    ]
    if bad or unexpected:
        raise RuntimeError(f"QAT checkpoint/model mismatch: missing={bad}, unexpected={unexpected}")
    return model


def export_qat_onnx(checkpoint: Path, target: Path):
    model = load_qat_model(checkpoint)
    example = torch.randn(2, NOISE_DIM, 1, 1, dtype=torch.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "input_names": ["z"],
        "output_names": ["image"],
        "dynamic_axes": {"z": {0: "batch"}, "image": {0: "batch"}},
        "opset_version": 17,
        # Keep FakeQuant's Q/DQ structure visible.  With constant folding on,
        # the legacy exporter may fold a scalar FakeQuant into a direct DQ (or
        # into an initializer), while the deployment rewrite still needs the
        # scale to materialize an INT8 ConvTranspose weight.
        "do_constant_folding": False,
        "training": torch.onnx.TrainingMode.EVAL,
    }
    try:
        torch.onnx.export(model, example, str(target), dynamo=False, **kwargs)
    except TypeError as exc:
        if "dynamo" not in str(exc):
            raise
        torch.onnx.export(model, example, str(target), **kwargs)
    import onnx
    checked = onnx.load(str(target))
    onnx.checker.check_model(checked)
    weight_dq_candidates = [
        node.output[0]
        for node in checked.graph.node
        if node.op_type == "DequantizeLinear" and node.output
        and any(
            conv.op_type == "ConvTranspose" and len(conv.input) > 1 and conv.input[1] == node.output[0]
            for conv in checked.graph.node
        )
    ]
    return {
        "file": target.name,
        "sha256": sha256(target),
        "node_count": len(checked.graph.node),
        "operator_counts": dict(Counter(node.op_type for node in checked.graph.node)),
        "conv_weight_dq_candidates": len(weight_dq_candidates),
    }


def set_axis(node, axis: int):
    from onnx import helper
    for attribute in node.attribute:
        if attribute.name == "axis":
            previous = int(attribute.i)
            attribute.i = int(axis)
            return previous
    node.attribute.append(helper.make_attribute("axis", int(axis)))
    return None


def get_axis(node, default=1):
    for attribute in node.attribute:
        if attribute.name == "axis":
            return int(attribute.i)
    return default


def build_mixed_deployment_onnx(source: Path, target: Path):
    """Default path: materialize QAT weights and insert two FP16 islands.

    This path intentionally does not consume the 04C graph.  Weight Q nodes
    are replaced by an INT8 initializer followed by DQ;
    direct-DQ and exporter-inserted Identity/Cast variants are accepted too.
    Activation Q/DQ stays before the INT8 ConvTranspose input.  This avoids
    the TensorRT ConvTranspose per-channel weight tactic failure observed in
    the old all-in-one script.  New 05A checkpoints use scalar weight scales.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(source))
    onnx.checker.check_model(model)
    initializers = {item.name: item for item in model.graph.initializer}
    producer = {output: node for node in model.graph.node for output in node.output if output}
    extra_initializers = []
    remove_nodes = set()
    materialized = []
    conv_weight_dq = {
        node.input[1]: node
        for node in model.graph.node
        if node.op_type == "ConvTranspose" and len(node.input) > 1
    }
    candidate_dq = 0
    candidate_details = []

    def unwrap_passthrough(name):
        """Follow exporter-inserted Identity/Cast nodes to an initializer/Q."""
        chain = []
        current = name
        seen = set()
        while current and current not in initializers and current not in seen:
            seen.add(current)
            node = producer.get(current)
            if node is None or node.op_type not in {"Identity", "Cast"} or not node.input:
                break
            chain.append(node)
            current = node.input[0]
        return current, chain

    for dq in list(model.graph.node):
        if dq.op_type != "DequantizeLinear" or not dq.input or dq.output[0] not in conv_weight_dq:
            continue
        candidate_dq += 1
        quantized_name, input_bridge = unwrap_passthrough(dq.input[0])
        q = producer.get(quantized_name)
        if q is not None and q.op_type == "QuantizeLinear":
            # Normal exporter form: float initializer -> Q -> DQ.
            source_name, _ = unwrap_passthrough(q.input[0] if q.input else "")
            scale_name = q.input[1] if len(q.input) > 1 else ""
            zero_name = q.input[2] if len(q.input) > 2 else ""
            q_node = q
        else:
            # Constant-folded form: float/int8 initializer -> DQ.  The
            # exporter can legally omit zero_point for a symmetric QAT node.
            source_name = quantized_name
            scale_name = dq.input[1] if len(dq.input) > 1 else ""
            zero_name = dq.input[2] if len(dq.input) > 2 else ""
            q_node = None

        source_name, _ = unwrap_passthrough(source_name)
        scale_name, _ = unwrap_passthrough(scale_name)
        zero_name, _ = unwrap_passthrough(zero_name)

        source_weight = initializers.get(source_name)
        scale_init = initializers.get(scale_name)
        zero_init = initializers.get(zero_name) if zero_name else None
        candidate_details.append({
            "conv": conv_weight_dq[dq.output[0]].name or conv_weight_dq[dq.output[0]].output[0],
            "dq": dq.name or dq.output[0],
            "source": source_name,
            "scale": scale_name,
            "has_quantize_node": q_node is not None,
        })
        if source_weight is None or scale_init is None:
            continue

        source_array = numpy_helper.to_array(source_weight)
        scale = numpy_helper.to_array(scale_init).astype(np.float32, copy=False)
        if zero_init is None:
            zero = np.asarray(0, dtype=np.float32)
        else:
            zero = numpy_helper.to_array(zero_init).astype(np.float32, copy=False)
        if scale.size != 1 or zero.size != 1:
            raise RuntimeError(
                f"{source_weight.name} contains a per-channel ConvTranspose weight scale "
                f"scale_shape={list(scale.shape)}, zero_shape={list(zero.shape)}. "
                "This QAT deployment path requires scalar weight Q/DQ scales; "
                "rerun the rewritten 05A with layout version 4."
            )

        scale_b = float(scale.reshape(-1)[0])
        zero_b = float(zero.reshape(-1)[0])
        if not np.isfinite(scale_b) or scale_b == 0.0:
            raise RuntimeError(f"Invalid QAT weight scale {scale_b!r} in {source_weight.name}")

        if source_array.dtype == np.dtype(np.int8):
            # Some exporter versions already materialize the quantized
            # initializer.  Preserve it and only audit the associated DQ.
            q_name = source_weight.name
            weight_shape = list(source_array.shape)
            scale_mode = "existing_int8"
            dq.input[0] = q_name
        elif source_array.dtype == np.dtype(np.uint8):
            raise RuntimeError(
                f"{source_weight.name} was exported as uint8, but 05A uses signed INT8 QAT; "
                "rerun 05A/05B in a fresh Kaggle session."
            )
        else:
            weight = source_array.astype(np.float32, copy=False)
            quantized = np.clip(np.rint(weight / scale_b + zero_b), -128, 127).astype(np.int8)
            q_name = f"{source_weight.name}__qat_int8"
            q_init = numpy_helper.from_array(quantized, name=q_name)
            extra_initializers.append(q_init)
            initializers[q_name] = q_init
            dq.input[0] = q_name
            weight_shape = list(weight.shape)
            scale_mode = "per_tensor"

        if q_node is not None:
            # Remove Q only when it has no remaining consumers after this
            # rewrite.  This keeps the graph valid if an exporter reuses Q.
            q_output = q_node.output[0] if q_node.output else ""
            other_consumers = [
                node for node in model.graph.node
                if node is not q_node and q_output in node.input
            ]
            if not other_consumers:
                remove_nodes.add(q_node.name or q_output)
        for bridge in input_bridge:
            bridge_output = bridge.output[0] if bridge.output else ""
            other_consumers = [
                node for node in model.graph.node
                if node is not dq and node is not q_node and bridge_output in node.input
            ]
            if not other_consumers:
                remove_nodes.add(bridge.name or bridge_output)
        materialized.append({
            "layer": conv_weight_dq[dq.output[0]].name or conv_weight_dq[dq.output[0]].output[0],
            "weight": source_weight.name,
            "initializer": q_name,
            "shape": weight_shape,
            "scale_count": int(scale.size),
            "scale_mode": scale_mode,
        })

    if len(materialized) != len(INT8_LAYERS):
        counts = Counter(node.op_type for node in model.graph.node)
        raise RuntimeError(
            f"Expected {len(INT8_LAYERS)} INT8 ConvTranspose weight DQ nodes, "
            f"materialized {len(materialized)}; weight_dq_candidates={candidate_dq}; "
            f"operator_counts={dict(counts)}; candidates={json.dumps(candidate_details, ensure_ascii=False)}. "
            "The ONNX exporter did not preserve all QAT weight scales. "
            "05B now disables constant folding; if this persists, rerun 05A and 05B "
            "from a fresh Kaggle session with layout version 4."
        )

    original_nodes = list(model.graph.node)
    kept_nodes = [
        node for node in original_nodes
        if (node.name or (node.output[0] if node.output else "")) not in remove_nodes
    ]
    conv_nodes = [node for node in kept_nodes if node.op_type == "ConvTranspose"]
    if len(conv_nodes) != 5:
        raise RuntimeError(f"Expected five ConvTranspose nodes after materialization, got {len(conv_nodes)}")

    node_output_producer = {output: node for node in kept_nodes for output in node.output if output}
    new_nodes = []
    protected_rows = []
    for node in kept_nodes:
        if node.op_type != "ConvTranspose":
            new_nodes.append(node)
            continue
        label = CONV_LABELS[len([item for item in new_nodes if item.op_type == "ConvTranspose"])]
        node.name = f"{label}.ConvTranspose"
        if label not in FP16_LAYERS:
            new_nodes.append(node)
            continue
        casts = []
        for index, input_name in enumerate(list(node.input)):
            initializer = initializers.get(input_name)
            half_name = f"{input_name}__{label.replace('.', '_')}_fp16"
            if initializer is not None:
                array = numpy_helper.to_array(initializer)
                if array.dtype.kind not in "fc":
                    raise RuntimeError(f"Protected layer {label} initializer {input_name} is not float")
                half_init = numpy_helper.from_array(array.astype(np.float16), name=half_name)
                extra_initializers.append(half_init)
                initializers[half_name] = half_init
            else:
                casts.append(helper.make_node("Cast", [input_name], [half_name], name=f"{label}.InputCast.{index}", to=TensorProto.FLOAT16))
            node.input[index] = half_name
        old_output = node.output[0]
        half_output = f"{old_output}__{label.replace('.', '_')}_fp16"
        node.output[0] = half_output
        casts.append(helper.make_node("Cast", [half_output], [old_output], name=f"{label}.OutputCast", to=TensorProto.FLOAT))
        new_nodes.extend(casts[: len(casts) - 1])
        new_nodes.append(node)
        new_nodes.append(casts[-1])
        protected_rows.append({"layer": label, "input_dtype": "FLOAT16", "weight_dtype": "FLOAT16", "compute_output_dtype": "FLOAT16", "boundary_dtype": "FLOAT32"})

    model.graph.ClearField("node")
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(extra_initializers)
    target.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(target))
    checked = onnx.load(str(target))
    onnx.checker.check_model(checked)
    onnx.checker.check_model(onnx.shape_inference.infer_shapes(checked))
    return {"file": target.name, "sha256": sha256(target), "materialized_weights": materialized, "protected_layers": protected_rows, "operator_counts": dict(Counter(node.op_type for node in checked.graph.node))}


def audit_qdq(path: Path):
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(path))
    onnx.checker.check_model(model)
    conv_nodes = [node for node in model.graph.node if node.op_type == "ConvTranspose"]
    if len(conv_nodes) != 5:
        raise RuntimeError(f"Expected five ConvTranspose nodes, got {len(conv_nodes)}")
    producer = {output: node for node in model.graph.node for output in node.output if output}
    initializers = {item.name: item for item in model.graph.initializer}
    rows = []
    for label, node in zip(CONV_LABELS, conv_nodes):
        input_dq = bool(node.input) and producer.get(node.input[0]) is not None and producer[node.input[0]].op_type == "DequantizeLinear"
        weight_dq = len(node.input) > 1 and producer.get(node.input[1]) is not None and producer[node.input[1]].op_type == "DequantizeLinear"
        has_cast_input = bool(node.input) and producer.get(node.input[0]) is not None and producer[node.input[0]].op_type == "Cast"
        weight_scale_count = None
        if label in INT8_LAYERS and weight_dq:
            weight_dq_node = producer[node.input[1]]
            if len(weight_dq_node.input) > 1 and weight_dq_node.input[1] in initializers:
                weight_scale_count = int(numpy_helper.to_array(initializers[weight_dq_node.input[1]]).size)
        passed = (input_dq and weight_dq and weight_scale_count == 1) if label in INT8_LAYERS else (has_cast_input and not input_dq and not weight_dq)
        rows.append({"layer": label, "input_from_dq": input_dq, "weight_from_dq": weight_dq, "weight_scale_count": weight_scale_count, "fp16_cast_boundary": has_cast_input, "expected": "INT8" if label in INT8_LAYERS else "FP16", "passed": bool(passed)})
    counts = Counter(node.op_type for node in model.graph.node)
    if not all(row["passed"] for row in rows):
        raise RuntimeError("QAT deployment graph audit failed: " + json.dumps(rows, ensure_ascii=False))
    return {"checker": "passed", "rows": rows, "quantize_linear": counts.get("QuantizeLinear", 0), "dequantize_linear": counts.get("DequantizeLinear", 0), "cast": counts.get("Cast", 0), "operator_counts": dict(counts)}


def _installed_tensorrt_versions():
    versions = {}
    for distribution in ("tensorrt", "tensorrt-cu11", "tensorrt-cu12", "tensorrt-cu13"):
        try:
            versions[distribution] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            pass
    return versions


def ensure_tensorrt(requested: str, pip_package: str, skip_install: bool = False):
    """Install/check TensorRT before importing it in this Python process.

    A pip install cannot replace a module already present in ``sys.modules``.
    Therefore a previously imported wrong version is a hard stop that requires
    a fresh Kaggle session; an installed-but-not-imported wrong version can be
    replaced safely before the first TensorRT import.
    """
    requested = str(requested).strip()
    target = DEFAULT_TRT_VERSION if requested.lower() in {"", "auto", "any"} else requested
    if "tensorrt" in sys.modules:
        loaded = str(getattr(sys.modules["tensorrt"], "__version__", "unknown"))
        if not loaded.startswith(target):
            raise RuntimeError(
                f"TensorRT {loaded} is already imported; 05B requires {target}. "
                "Restart the Kaggle session, then run 05B before importing TensorRT."
            )
        import tensorrt as trt
        print(f"[TensorRT] already imported: {loaded}")
        return trt

    installed = _installed_tensorrt_versions()
    if not any(str(version).startswith(target) for version in installed.values()):
        if skip_install:
            raise RuntimeError(
                f"TensorRT {target} is not installed. Remove --skip-tensorrt-install "
                f"or install {pip_package}=={target} before running 05B."
            )
        package_spec = f"{pip_package}=={target}"
        command = [
            sys.executable, "-m", "pip", "install", "--upgrade", "-q",
            package_spec, "--extra-index-url", "https://pypi.nvidia.com",
        ]
        print(f"[TensorRT] installing {package_spec} before first import")
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to install {package_spec}. Check Kaggle Internet and CUDA major; "
                "use --tensorrt-pip-package tensorrt-cu13 for CUDA 13.x."
            ) from exc
        importlib.invalidate_caches()

    try:
        import tensorrt as trt
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"TensorRT {target} was installed but cannot be imported. "
            "Restart the Kaggle session and run 05B again before any TensorRT import."
        ) from exc
    loaded = str(trt.__version__)
    if not loaded.startswith(target):
        raise RuntimeError(
            f"TensorRT runtime mismatch after installation: current={loaded}, required={target}. "
            "Restart the Kaggle session and do not import another TensorRT version first."
        )
    print(f"[TensorRT] {loaded}")
    return trt


def ensure_onnx():
    try:
        import onnx  # noqa: F401
        return
    except (ImportError, ModuleNotFoundError):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "onnx"])
        importlib.invalidate_caches()
        import onnx  # noqa: F401


def make_logger(trt):
    severity = getattr(trt.Logger, "WARNING", 2)
    return trt.Logger(severity)


def build_engine(trt, onnx_path: Path, engine_path: Path, args, tag: str):
    version = str(trt.__version__)
    logger = make_logger(trt)
    builder = trt.Builder(logger)
    parts = version.split(".")
    major = int(parts[0])
    minor = int(parts[1]) if len(parts) > 1 else 0
    creation = getattr(trt, "NetworkDefinitionCreationFlag", None)
    strongly_typed = bool(creation is not None and hasattr(creation, "STRONGLY_TYPED") and (major > 10 or (major == 10 and minor >= 12)))
    if strongly_typed:
        flags = 1 << int(creation.STRONGLY_TYPED)
    elif creation is not None and hasattr(creation, "EXPLICIT_BATCH"):
        flags = 1 << int(creation.EXPLICIT_BATCH)
    else:
        flags = 0
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        diagnostic = {"status": "parse_failed", "tag": tag, "tensorrt": version, "onnx": str(onnx_path), "errors": errors}
        (engine_path.parent / f"{tag}_build_diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError("TensorRT ONNX parse failed: " + " | ".join(errors))
    config = builder.create_builder_config()
    workspace = int(args.workspace_gb * 1024 ** 3)
    memory_pool = getattr(trt, "MemoryPoolType", None)
    if hasattr(config, "set_memory_pool_limit") and memory_pool is not None:
        config.set_memory_pool_limit(memory_pool.WORKSPACE, workspace)
    else:
        config.max_workspace_size = workspace
    if not strongly_typed and hasattr(config, "set_flag") and hasattr(trt, "BuilderFlag") and hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    input_name = network.get_input(0).name
    if profile.set_shape(input_name, (1, NOISE_DIM, 1, 1), (args.opt_batch, NOISE_DIM, 1, 1), (args.max_batch, NOISE_DIM, 1, 1)) is False:
        raise RuntimeError(f"TensorRT rejected the dynamic input profile for {tag}")
    config.add_optimization_profile(profile)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if hasattr(builder, "build_serialized_network"):
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise RuntimeError("build_serialized_network returned None")
            engine_path.write_bytes(bytes(serialized))
        else:
            engine = builder.build_engine(network, config)
            if engine is None:
                raise RuntimeError("build_engine returned None")
            engine_path.write_bytes(bytes(engine.serialize()))
    except Exception as exc:
        diagnostic = {"status": "build_failed", "tag": tag, "tensorrt": version, "onnx": str(onnx_path), "workspace_gb": args.workspace_gb, "strongly_typed": strongly_typed, "exception": repr(exc)}
        (engine_path.parent / f"{tag}_build_diagnostics.json").write_text(json.dumps(diagnostic, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(f"TensorRT {tag} build failed; see {tag}_build_diagnostics.json: {exc}") from exc
    if not engine_path.is_file() or engine_path.stat().st_size == 0:
        raise RuntimeError(f"TensorRT produced an empty {tag} engine")
    return {"file": engine_path.name, "sha256": sha256(engine_path), "bytes": engine_path.stat().st_size, "tensorrt": version, "strongly_typed": strongly_typed}


def io_names(engine, trt):
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        return next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT), next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
    names = [engine.get_binding_name(index) for index in range(engine.num_bindings)]
    return next(name for index, name in enumerate(names) if engine.binding_is_input(index)), next(name for index, name in enumerate(names) if not engine.binding_is_input(index))


def preflight_baseline_engines(trt, engine_paths, output_dir: Path):
    """Check serialized baseline engines before any expensive 05B work.

    TensorRT engines cannot be converted between serialization runtimes.  This
    preflight deliberately deserializes FP32/PTQ/MIXED first, so a mismatch is
    reported before QAT ONNX export, TensorRT builds, Inception/FID setup, or
    GPU benchmarking.
    """
    runtime_version = str(trt.__version__)
    runtime = trt.Runtime(make_logger(trt))
    rows = []
    for label, path in engine_paths.items():
        row = {
            "label": label,
            "path": str(path),
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
            "runtime": runtime_version,
        }
        try:
            serialized = path.read_bytes()
            engine = runtime.deserialize_cuda_engine(serialized)
            if engine is None:
                raise RuntimeError("deserialize_cuda_engine returned None")
            row["io_names"] = list(io_names(engine, trt))
            row["status"] = "passed"
            del engine
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = repr(exc)
            rows.append(row)
            report = {
                "status": "failed",
                "runtime": runtime_version,
                "reason": "serialized_engine_runtime_mismatch_or_invalid_engine",
                "engines": rows,
                "action": "start a fresh Kaggle session with the same TensorRT version used to build all baseline engines; do not replace TensorRT after import",
            }
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "qat_runtime_preflight.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            raise RuntimeError(
                f"05B baseline engine preflight failed for {label}: {path.name}. "
                f"Current TensorRT={runtime_version}. See qat_runtime_preflight.json; "
                "all serialized engines must be built and loaded with the same TensorRT runtime."
            ) from exc
        rows.append(row)
    report = {"status": "passed", "runtime": runtime_version, "engines": rows}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "qat_runtime_preflight.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


class TensorRTEngine:
    def __init__(self, path: Path, trt):
        self.path = path
        self.trt = trt
        self.runtime = trt.Runtime(make_logger(trt))
        try:
            self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        except Exception as exc:
            if "Version tag does not match" in str(exc) or "SERIALIZATION" in str(exc):
                raise RuntimeError(
                    f"TensorRT engine serialization mismatch for {path.name}: "
                    f"current runtime={trt.__version__}. Rebuild this engine with "
                    "the same TensorRT version before running 05B."
                ) from exc
            raise
        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize {path}. The engine was likely built with a "
                f"different TensorRT serialization version; current runtime={trt.__version__}."
            )
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = io_names(self.engine, trt)
        if hasattr(self.engine, "get_tensor_dtype"):
            self.input_np_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))
            self.output_np_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))
        else:
            self.input_np_dtype = np.float32
            self.output_np_dtype = np.float32
        self.input_torch_dtype = torch.float16 if self.input_np_dtype == np.float16 else torch.float32
        self.output_torch_dtype = torch.float16 if self.output_np_dtype == np.float16 else torch.float32
        self.stream = torch.cuda.Stream()

    def infer(self, latent: np.ndarray):
        host = np.ascontiguousarray(latent.astype(self.input_np_dtype, copy=False))
        input_tensor = torch.from_numpy(host).to("cuda", dtype=self.input_torch_dtype)
        output = torch.empty((len(latent), 3, IMAGE_SIZE, IMAGE_SIZE), device="cuda", dtype=self.output_torch_dtype)
        stream = self.stream.cuda_stream
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
            self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(output.data_ptr()))
            ok = self.context.execute_async_v3(stream)
        else:
            input_index = self.engine.get_binding_index(self.input_name)
            output_index = self.engine.get_binding_index(self.output_name)
            self.context.set_binding_shape(input_index, tuple(input_tensor.shape))
            bindings = [0] * self.engine.num_bindings
            bindings[input_index] = int(input_tensor.data_ptr())
            bindings[output_index] = int(output.data_ptr())
            ok = self.context.execute_async_v2(bindings, stream)
        if ok is False:
            raise RuntimeError(f"TensorRT execution failed for {self.path.name}")
        self.stream.synchronize()
        return output.float().cpu()


class StandardFID:
    def __init__(self):
        try:
            from pytorch_fid.inception import InceptionV3
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-fid"])
            importlib.invalidate_caches()
            from pytorch_fid.inception import InceptionV3
        block = int(getattr(InceptionV3, "DEFAULT_BLOCK_INDEX", 3))
        self.net = InceptionV3([block]).eval().cuda()
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images01):
        values = self.net(images01.cuda())[0]
        return values.reshape(values.shape[0], -1).detach().cpu().numpy()


def load_real_images(real_dir: Path, count: int, batch_size: int):
    paths = sorted(real_dir.glob("*.png"))[:count]
    if len(paths) < count:
        raise RuntimeError(f"real_eval contains {len(paths)} PNG files; need {count}")
    for start in range(0, count, batch_size):
        current = paths[start:start + batch_size]
        images = []
        for path in current:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        yield torch.stack(images), current


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
        array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.float32)
        gray = array.mean(axis=2)
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
        values.append(float(lap.var()))
    return np.asarray(values, dtype=np.float64)


def haar_detail(x):
    e, o = x[..., 0::2, :], x[..., 1::2, :]
    lh = (e[..., :, 0::2] - e[..., :, 1::2] + o[..., :, 0::2] - o[..., :, 1::2]) * 0.5
    hl = (e[..., :, 0::2] + e[..., :, 1::2] - o[..., :, 0::2] - o[..., :, 1::2]) * 0.5
    hh = (e[..., :, 0::2] - e[..., :, 1::2] - o[..., :, 0::2] + o[..., :, 1::2]) * 0.5
    return torch.cat((lh, hl, hh), dim=1)


def evaluate_engine(engine, label, latent_eval, n_fid, n_image, batch_size, fid_net, real_mu, real_cov, blur_threshold):
    engine.infer(latent_eval[: min(batch_size, n_fid)])
    feature_batches = []
    image_batches = []
    for start in range(0, n_fid, batch_size):
        fake = (engine.infer(latent_eval[start:start + batch_size]) + 1.0).clamp(0, 2) / 2.0
        feature_batches.append(fid_net.features(fake))
        if start < n_image:
            image_batches.append(fake[: min(len(fake), n_image - start)].cpu())
    features = np.concatenate(feature_batches, axis=0)[:n_fid]
    images = torch.cat(image_batches, dim=0)[:n_image]
    fake_mu, fake_cov = stats(features)
    blur = float((laplacian_values(images) < blur_threshold).mean())
    return {"label": label, "fid_standard": fid_from_stats(real_mu, real_cov, fake_mu, fake_cov), "blur_rate": blur, "images": images, "feature_count": int(len(features))}


def benchmark(engine, label, latent_eval, batches, warmup, iters, repeats):
    rows = []
    for batch in batches:
        sample = latent_eval[:batch]
        times = []
        for repeat in range(repeats):
            for _ in range(warmup):
                engine.infer(sample)
            torch.cuda.synchronize()
            for _ in range(iters):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                engine.infer(sample)
                end.record()
                end.synchronize()
                times.append(float(start.elapsed_time(end)))
        rows.append({
            "label": label,
            "batch": batch,
            "benchmark_repeats": repeats,
            "benchmark_samples": len(times),
            "latency_mean_ms": float(np.mean(times)),
            "latency_p50_ms": float(np.percentile(times, 50)),
            "latency_p99_ms": float(np.percentile(times, 99)),
            "latency_std_ms": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "throughput_images_per_s": float(batch / (np.mean(times) / 1000.0)),
            "cuda_allocated_mb": float(torch.cuda.memory_allocated() / 1024 ** 2),
            "cuda_reserved_mb": float(torch.cuda.memory_reserved() / 1024 ** 2),
        })
    return rows


def save_contact_sheet(results, path: Path):
    labels = list(results)
    count = min(8, len(next(iter(results.values()))["images"]))
    canvas = Image.new("RGB", (count * IMAGE_SIZE, len(labels) * (IMAGE_SIZE + 20)), "white")
    draw = ImageDraw.Draw(canvas)
    for row, label in enumerate(labels):
        draw.text((2, row * (IMAGE_SIZE + 20) + 2), label, fill="black")
        images = results[label]["images"][:count]
        for column, tensor in enumerate(images):
            array = (tensor.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(array), (column * IMAGE_SIZE, row * (IMAGE_SIZE + 20) + 20))
    canvas.save(path)


def plot_summary(rows, benchmark_rows, path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "not_created", "reason": "matplotlib_missing"}
    valid = [row for row in rows if row.get("status") == "ok"]
    labels = [row["label"] for row in valid]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), dpi=160)
    axes[0].bar(labels, [float(row["fid_standard"]) for row in valid])
    axes[0].set_title("Standard FID")
    axes[1].bar(labels, [100 * float(row["blur_rate"]) for row in valid])
    axes[1].set_title("Blur rate (%)")
    for label in labels:
        points = [item for item in benchmark_rows if item["label"] == label]
        axes[2].plot([item["batch"] for item in points], [item["latency_p99_ms"] for item in points], marker="o", label=label)
    axes[2].set_title("P99 latency by batch")
    axes[2].set_xlabel("batch")
    axes[2].set_ylabel("ms")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"status": "created", "file": path.name}


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_zip(output_dir: Path):
    archive = output_dir.parent / "Task5_05B_QAT_Evaluation.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output_dir.iterdir():
            if path.is_file():
                bundle.write(path, arcname=path.name)
    return archive


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("05B requires a Kaggle GPU accelerator")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / "_staging"
    qat_root = extract_bundle(args.qat_bundle, staging / "qat") if args.qat_bundle else None
    pre_path = locate_qat_checkpoint("qat_pre.pth", args.pre_qat_checkpoint, qat_root, args.qat_output_dir)
    best_path = locate_qat_checkpoint("qat_best.pth", args.qat_checkpoint, qat_root, args.qat_output_dir)
    fp32_path = find_one("generator_trt_fp32.engine", args.fp32_engine)
    ptq_path = find_one("generator_trt_int8.engine", args.ptq_int8_engine)
    mixed_path = find_one("generator_trt_mixed_precision_final.engine", args.mixed_ptq_engine)
    baseline_engine_paths = {"FP32": fp32_path, "PTQ_INT8": ptq_path, "MIXED_PTQ": mixed_path}
    ensure_onnx()
    trt = ensure_tensorrt(args.tensorrt_version, args.tensorrt_pip_package, args.skip_tensorrt_install)
    runtime_preflight = preflight_baseline_engines(trt, baseline_engine_paths, output_dir)

    protocol = locate_protocol(args.protocol_path, staging / "protocol")
    latent_eval = np.load(protocol / "latent_eval.npy").astype(np.float32, copy=False)
    if latent_eval.ndim != 4 or tuple(latent_eval.shape[1:]) != (NOISE_DIM, 1, 1):
        raise ValueError(f"Unexpected latent_eval shape: {latent_eval.shape}")
    n_fid = min(args.n_fid, len(latent_eval))
    n_image = min(args.n_image_eval, n_fid)
    real_batches = list(load_real_images(protocol / "real_eval", n_fid, args.batch_size))
    real_images = torch.cat([batch for batch, _paths in real_batches], dim=0)[:n_fid]
    blur_threshold = float(np.percentile(laplacian_values(real_images[:n_image]), 10))
    fid_net = StandardFID()
    real_features = np.concatenate([fid_net.features(batch) for batch, _paths in real_batches], axis=0)[:n_fid]
    real_mu, real_cov = stats(real_features)

    build_rows = []
    engine_paths = dict(baseline_engine_paths)
    for label, checkpoint in (("PRE_QAT", pre_path), ("QAT_INT8", best_path)):
        raw = output_dir / f"generator_{label.lower()}_qat_raw.onnx"
        mixed = output_dir / f"generator_{label.lower()}_qat_mixed.onnx"
        engine = output_dir / f"generator_trt_{label.lower()}.engine"
        raw_info = export_qat_onnx(checkpoint, raw)
        graph_info = build_mixed_deployment_onnx(raw, mixed)
        graph_audit = audit_qdq(mixed)
        engine_info = build_engine(trt, mixed, engine, args, label.lower())
        build_rows.append({"label": label, "checkpoint": checkpoint.name, "checkpoint_sha256": sha256(checkpoint), "raw_onnx": raw_info, "mixed_onnx": graph_info, "graph_audit": graph_audit, "engine": engine_info})
        engine_paths[label] = engine

    engines = {label: TensorRTEngine(path, trt) for label, path in engine_paths.items()}
    eval_results = {}
    for label, engine in engines.items():
        eval_results[label] = evaluate_engine(engine, label, latent_eval, n_fid, n_image, args.batch_size, fid_net, real_mu, real_cov, blur_threshold)
    fp32_images = eval_results["FP32"]["images"]
    metric_rows = []
    for label, result in eval_results.items():
        images = result["images"]
        diff = images - fp32_images
        hf = haar_detail(images)
        hf_ref = haar_detail(fp32_images)
        metric_rows.append({
            "label": label,
            "status": "ok",
            "role": {"FP32": "reference", "PTQ_INT8": "all_int8_ptq_baseline", "MIXED_PTQ": "04C_mixed_ptq_baseline", "PRE_QAT": "calibrated_before_qat", "QAT_INT8": "trained_after_qat"}[label],
            "fid_standard": result["fid_standard"],
            "blur_rate": result["blur_rate"],
            "blur_delta_pp_vs_fp32": (result["blur_rate"] - eval_results["FP32"]["blur_rate"]) * 100.0,
            "highfreq_haar_mae_vs_fp32": float(hf.sub(hf_ref).abs().mean().item()),
            "image_mae_vs_fp32": float(diff.abs().mean().item()),
            "image_rmse_vs_fp32": float(torch.sqrt((diff * diff).mean()).item()),
            "feature_count": result["feature_count"],
            "engine": engine_paths[label].name,
            "engine_sha256": sha256(engine_paths[label]),
        })

    benchmark_rows = []
    for label, engine in engines.items():
        benchmark_rows.extend(benchmark(engine, label, latent_eval, args.dynamic_batches, args.warmup, args.iters, args.benchmark_repeats))
    primary = {row["label"]: row for row in benchmark_rows if row["batch"] == args.batch_size}
    for row in metric_rows:
        row.update({
            "latency_mean_ms_batch": primary[row["label"]]["latency_mean_ms"],
            "latency_p99_ms_batch": primary[row["label"]]["latency_p99_ms"],
            "throughput_images_per_s": primary[row["label"]]["throughput_images_per_s"],
        })

    by_label = {row["label"]: row for row in metric_rows}
    fp32 = by_label["FP32"]
    ptq = by_label["PTQ_INT8"]
    mixed = by_label["MIXED_PTQ"]
    pre = by_label["PRE_QAT"]
    qat = by_label["QAT_INT8"]
    comparison = {
        "pre_to_post": {
            "fid_pre": pre["fid_standard"], "fid_post": qat["fid_standard"], "fid_improvement": pre["fid_standard"] - qat["fid_standard"],
            "blur_pre": pre["blur_rate"], "blur_post": qat["blur_rate"], "blur_distance_change_pp": abs(pre["blur_delta_pp_vs_fp32"]) - abs(qat["blur_delta_pp_vs_fp32"]),
            "highfreq_mae_pre": pre["highfreq_haar_mae_vs_fp32"], "highfreq_mae_post": qat["highfreq_haar_mae_vs_fp32"],
        },
        "qat_vs_ptq": {
            "fid_improvement": ptq["fid_standard"] - qat["fid_standard"],
            "blur_change_vs_ptq_pp": (qat["blur_rate"] - ptq["blur_rate"]) * 100.0,
            "blur_distance_improvement_pp": abs(ptq["blur_delta_pp_vs_fp32"]) - abs(qat["blur_delta_pp_vs_fp32"]),
            "highfreq_mae_improvement": ptq["highfreq_haar_mae_vs_fp32"] - qat["highfreq_haar_mae_vs_fp32"],
            "speed_ratio_p99_vs_all_int8_ptq": qat["latency_p99_ms_batch"] / ptq["latency_p99_ms_batch"],
            "speed_ratio_p99_vs_mixed_ptq": qat["latency_p99_ms_batch"] / mixed["latency_p99_ms_batch"],
        },
        "qat_vs_fp32": {
            "speed_ratio_p99": qat["latency_p99_ms_batch"] / fp32["latency_p99_ms_batch"],
            "speedup": fp32["latency_p99_ms_batch"] / qat["latency_p99_ms_batch"],
            "latency_reduction_percent": (1.0 - qat["latency_p99_ms_batch"] / fp32["latency_p99_ms_batch"]) * 100.0,
        },
    }
    quality_pass = bool(
        comparison["qat_vs_ptq"]["fid_improvement"] > 0
        and comparison["qat_vs_ptq"]["blur_change_vs_ptq_pp"] <= args.max_blur_degradation_vs_ptq_pp
    )
    speed_pass = bool(
        comparison["qat_vs_ptq"]["speed_ratio_p99_vs_all_int8_ptq"] <= args.max_speed_ratio_all_int8
        and comparison["qat_vs_ptq"]["speed_ratio_p99_vs_mixed_ptq"] <= args.max_speed_ratio_mixed
    )
    status = "complete" if quality_pass and speed_pass else "not_passed"
    write_csv(output_dir / "qat_vs_ptq_summary.csv", metric_rows)
    write_csv(output_dir / "qat_dynamic_benchmark.csv", benchmark_rows)
    before_after = [row for row in metric_rows if row["label"] in ("PRE_QAT", "QAT_INT8", "PTQ_INT8")]
    write_csv(output_dir / "qat_before_after_summary.csv", before_after)
    save_contact_sheet(eval_results, output_dir / "qat_visual_comparison.png")
    plot = plot_summary(metric_rows, benchmark_rows, output_dir / "qat_quality_speed_curves.png")
    manifest = {
        "task": "Task5_05B_QAT_Evaluation",
        "status": status,
        "inputs": {
            "pre_qat_checkpoint": {"path": str(pre_path), "sha256": sha256(pre_path)},
            "qat_checkpoint": {"path": str(best_path), "sha256": sha256(best_path)},
            "fp32_engine": {"path": str(fp32_path), "sha256": sha256(fp32_path)},
            "ptq_int8_engine": {"path": str(ptq_path), "sha256": sha256(ptq_path)},
            "mixed_ptq_engine": {"path": str(mixed_path), "sha256": sha256(mixed_path)},
            "protocol": str(protocol),
        },
        "runtime_preflight": runtime_preflight,
        "runtime_install": {
            "requested_version": args.tensorrt_version,
            "pip_package": args.tensorrt_pip_package,
            "skip_install": bool(args.skip_tensorrt_install),
            "install_happens_before_first_tensorrt_import": True,
        },
        "protocol": {"n_fid": n_fid, "n_image_eval": n_image, "batch_size": args.batch_size, "latent_shape": list(latent_eval.shape), "fid": "pytorch-fid Inception-v3 pool3 2048-d", "blur": "real_eval Laplacian p10 threshold", "high_frequency": "one-level Haar LH/HL/HH MAE against FP32 on identical latent_eval", "benchmark_warmup": args.warmup, "benchmark_iters": args.iters, "benchmark_repeats": args.benchmark_repeats},
        "deployment_policy": {"fp16_layers": list(FP16_LAYERS), "int8_layers": list(INT8_LAYERS), "weight_scale_mode": "per_tensor_scalar", "trailing_output_fake_quant": False, "tensorrt_version": str(trt.__version__), "strongly_typed_builder_for_10_12_plus": True, "topology_mode": "05B_explicit_qdq_rewrite", "mixed_ptq_baseline_is_engine_only": True},
        "builds": build_rows,
        "metrics": metric_rows,
        "benchmark": benchmark_rows,
        "comparison": comparison,
        "gates": {"quality_pass": quality_pass, "speed_pass": speed_pass, "quality_policy": "FID improves and QAT blur rate is not materially worse than PTQ; Haar is diagnostic only; contact sheet supports no-obvious-collapse visual claim", "max_blur_degradation_vs_ptq_pp": args.max_blur_degradation_vs_ptq_pp, "max_speed_ratio_all_int8_ptq": args.max_speed_ratio_all_int8, "max_speed_ratio_mixed_ptq": args.max_speed_ratio_mixed, "roi_status": "not_available; global Haar metric plus contact sheet used; no literal hair/eyeliner ROI claim"},
        "artifacts": ["qat_runtime_preflight.json", "qat_vs_ptq_summary.csv", "qat_before_after_summary.csv", "qat_dynamic_benchmark.csv", "qat_quality_speed_curves.png", "qat_visual_comparison.png", "qat_evaluation_manifest.json", "generator_trt_pre_qat.engine", "generator_trt_qat_int8.engine"],
    }
    (output_dir / "qat_evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    archive = None if args.no_zip else write_zip(output_dir)
    print(f"[05B] status={status} quality_pass={quality_pass} speed_pass={speed_pass}")
    print(f"[05B] FID PTQ={ptq['fid_standard']:.6f} QAT={qat['fid_standard']:.6f}; blur PTQ={ptq['blur_rate']:.4%} QAT={qat['blur_rate']:.4%}")
    print(f"[05B] p99 speed ratio QAT/PTQ={comparison['qat_vs_ptq']['speed_ratio_p99_vs_all_int8_ptq']:.4f}; QAT/MIXED={comparison['qat_vs_ptq']['speed_ratio_p99_vs_mixed_ptq']:.4f}")
    print(f"[05B] QAT vs FP32 P99 speedup={comparison['qat_vs_fp32']['speedup']:.4f}x; latency reduction={comparison['qat_vs_fp32']['latency_reduction_percent']:.2f}%")
    print(f"[05B] output={output_dir}")
    if archive:
        print(f"[05B] download this ZIP: {archive}")


if __name__ == "__main__":
    main()
