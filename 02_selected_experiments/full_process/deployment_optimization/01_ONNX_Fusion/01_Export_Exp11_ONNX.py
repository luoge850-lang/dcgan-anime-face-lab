"""
Step 1A — load the Exp11 EMA Generator and export a checked ONNX graph.

This script deliberately supports only the Exp11 Generator architecture. It
does not load the Discriminator, optimizer, or Improve Generator. The first
run only performs:

1. explicit/unique discovery of generator_ema_final.pth;
2. strict state_dict loading;
3. FP32 ONNX export with dynamic batch size;
4. onnx.checker and shape-inference validation;
5. node/operator inventory and a fixed-noise sample grid.

Operator fusion and engine benchmarking are intentionally separate next steps.
Do not call this script a successful deployment until the checker and a real
ONNX Runtime inference test also pass.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import make_grid


IMAGE_SIZE = 64
NOISE_DIM = 128
EXPERIMENT_NAME = "Exp11_Deployment"


class Exp11Generator(nn.Module):
    """Exact Generator structure used by 11_G_DiffAug_EMA_20K.py."""

    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 768, 4),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1),
            nn.BatchNorm2d(384),
            nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export Exp11 Generator to ONNX")
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH", ""))
    parser.add_argument(
        "--output-dir",
        default=os.getenv(
            "DEPLOY_OUTPUT_DIR",
            "/kaggle/working/dcgan_output/Deployment_Optimization_Results/01_ONNX_Fusion/01A_Export",
        ),
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--custom-op-probe",
        choices=("none", "wavelet", "dynamic_sn", "both"),
        default="both",
        help="Export standard-ONNX replacements for non-deployable wavelet/SN logic.",
    )
    args, _ = parser.parse_known_args(argv)
    return args


def find_unique_exp11_checkpoint(explicit):
    if explicit:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path

    candidates = []
    for root in (Path("/kaggle/input"), Path.cwd()):
        if root.exists():
            candidates.extend(root.rglob("generator_ema_final.pth"))
    preferred = [
        p for p in candidates
        if "11_G_DiffAug_EMA_20K" in str(p) or "exp11" in str(p).lower()
    ]
    if len(preferred) == 1:
        return preferred[0]
    if len(preferred) > 1:
        raise RuntimeError(
            "More than one Exp11 EMA checkpoint was found. Pass --checkpoint explicitly."
        )
    raise FileNotFoundError(
        "No unique Exp11 generator_ema_final.pth found. Attach the Exp11 weights "
        "dataset or pass --checkpoint explicitly."
    )


def extract_state_dict(path):
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "generator_ema" in state:
        state = state["generator_ema"]
    elif isinstance(state, dict) and "generator" in state:
        keys = list(state.keys())
        if not any(str(k).startswith("net.") for k in keys):
            state = state["generator"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(state)}")
    return state


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_grid(images, path):
    grid = make_grid(images, nrow=8, normalize=True, value_range=(-1, 1))
    array = (
        grid.mul(255)
        .clamp(0, 255)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
        .astype("uint8")
    )
    Image.fromarray(array).save(path)


def inventory_onnx(path, csv_path):
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    inferred = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred)
    counts = Counter(node.op_type for node in model.graph.node)
    custom_nodes = [
        {
            "name": node.name,
            "op_type": node.op_type,
            "domain": node.domain or "ai.onnx",
        }
        for node in model.graph.node
        if (node.domain or "ai.onnx") not in ("", "ai.onnx")
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["op_type", "count"])
        writer.writerows(sorted(counts.items()))
    return model, counts, custom_nodes


def capture_fx_graph(model, output_dir):
    """Capture the PyTorch graph before ONNX lowering for audit evidence."""
    try:
        from torch.fx import symbolic_trace

        traced = symbolic_trace(model)
        (output_dir / "fx_graph.txt").write_text(
            str(traced.graph) + "\n", encoding="utf-8"
        )
        rows = []
        for node in traced.graph.nodes:
            rows.append(
                {
                    "name": node.name,
                    "op": node.op,
                    "target": str(node.target),
                    "args": repr(node.args),
                    "kwargs": repr(node.kwargs),
                }
            )
        with open(output_dir / "fx_graph_nodes.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["name"])
            writer.writeheader()
            writer.writerows(rows)
        return {"status": "passed", "node_count": len(rows), "graph": "fx_graph.txt", "nodes": "fx_graph_nodes.csv"}
    except Exception as exc:
        # FX capture is supplementary evidence; ONNX export/checker remains the
        # hard gate. Keep the failure explicit instead of silently omitting it.
        (output_dir / "fx_graph_error.txt").write_text(repr(exc), encoding="utf-8")
        return {"status": "failed", "error": repr(exc)}


class HaarWaveletONNXProbe(nn.Module):
    """Pure-PyTorch replacement for a third-party one-level Haar DWT.

    The training repository used wavelet-style frequency supervision, but the
    current Exp11 Generator does not call it during inference.  This probe
    proves that the same LL/LH/HL/HH decomposition can be lowered to standard
    ONNX Slice/Add/Sub/Mul/Concat operators without a custom runtime kernel.
    """

    def forward(self, x):
        a = x[:, :, 0::2, 0::2]
        b = x[:, :, 0::2, 1::2]
        c = x[:, :, 1::2, 0::2]
        d = x[:, :, 1::2, 1::2]
        ll = (a + b + c + d) * 0.5
        lh = (a - b + c - d) * 0.5
        hl = (a + b - c - d) * 0.5
        hh = (a - b - c + d) * 0.5
        return torch.cat((ll, lh, hl, hh), dim=1)


class DynamicSpectralNormONNXProbe(nn.Module):
    """Fixed-step dynamic spectral normalization using standard tensor ops.

    A Python spectral-normalization hook or a third-party dynamic SN module
    is not directly deployable.  Two power-iteration steps are unrolled here
    into MatMul/ReduceSum/Sqrt/Div and the normalized weight is used by Conv2d.
    The probe is intentionally separate from the Exp11 Generator because its
    Generator checkpoint has no SN layer in the inference path.
    """

    def __init__(self, in_channels=3, out_channels=8, kernel_size=3, iterations=2):
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size) * 0.02
        )
        self.iterations = int(iterations)

    @staticmethod
    def _normalize(vector):
        denominator = torch.sqrt(torch.sum(vector * vector, dim=1, keepdim=True) + 1e-12)
        return vector / denominator

    def forward(self, x):
        weight = self.weight.reshape(self.weight.shape[0], -1)
        u = torch.ones((1, weight.shape[0]), dtype=x.dtype, device=x.device)
        u = self._normalize(u)
        for _ in range(self.iterations):
            v = self._normalize(torch.matmul(u, weight))
            u = self._normalize(torch.matmul(v, weight.transpose(0, 1)))
        sigma = torch.matmul(torch.matmul(u, weight), v.transpose(0, 1))
        normalized = self.weight / torch.clamp(sigma.reshape(1, 1, 1, 1), min=1e-6)
        return F.conv2d(x, normalized, padding=1)


def export_standard_operator_probes(output_dir, opset, mode, seed):
    """Export and checker-validate deployable replacements for custom logic."""
    if mode == "none":
        return {"status": "skipped", "probes": []}
    probe_dir = output_dir / "custom_operator_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rows = []
    probe_specs = []
    if mode in ("wavelet", "both"):
        probe_specs.append((
            "wavelet_replacement",
            HaarWaveletONNXProbe().eval(),
            torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "third-party Haar DWT -> standard Slice/Add/Sub/Mul/Concat",
        ))
    if mode in ("dynamic_sn", "both"):
        probe_specs.append((
            "dynamic_spectral_norm_replacement",
            DynamicSpectralNormONNXProbe().eval(),
            torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
            "dynamic power iteration -> standard MatMul/ReduceSum/Sqrt/Div/Conv",
        ))
    for name, probe, example, replacement in probe_specs:
        target = probe_dir / f"{name}.onnx"
        row = {
            "probe": name,
            "replacement": replacement,
            "onnx_file": str(target.name),
            "export": "failed",
            "checker": "not_run",
            "node_count": "",
            "custom_domain_count": "",
            "error": "",
        }
        try:
            torch.onnx.export(
                probe,
                example,
                target,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
                opset_version=opset,
                do_constant_folding=True,
                dynamo=False,
            )
            import onnx
            model = onnx.load(str(target))
            onnx.checker.check_model(model)
            custom_nodes = [node for node in model.graph.node if node.domain not in ("", "ai.onnx")]
            row.update({
                "export": "passed",
                "checker": "passed",
                "node_count": len(model.graph.node),
                "custom_domain_count": len(custom_nodes),
            })
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)
    with open(output_dir / "custom_operator_probe.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["status"])
        writer.writeheader()
        writer.writerows(rows)
    failed = [row for row in rows if row["export"] != "passed" or row["checker"] != "passed"]
    return {
        "status": "passed" if not failed else "failed",
        "mode": mode,
        "probe_count": len(rows),
        "passed_count": len(rows) - len(failed),
        "evidence_csv": "custom_operator_probe.csv",
        "probes": rows,
    }


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_unique_exp11_checkpoint(args.checkpoint)
    state = extract_state_dict(checkpoint)

    model = Exp11Generator().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Exp11 state_dict mismatch. "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )

    torch.manual_seed(args.sample_seed)
    example = torch.randn(1, NOISE_DIM, 1, 1)
    with torch.no_grad():
        sample = model(example)
    if tuple(sample.shape) != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
        raise RuntimeError(f"Unexpected Generator output shape: {tuple(sample.shape)}")
    with torch.no_grad():
        export_samples = model(torch.randn(64, NOISE_DIM, 1, 1))
    save_grid(export_samples, output_dir / "export_sample_grid.png")

    fx_capture = capture_fx_graph(model, output_dir)
    if fx_capture["status"] != "passed":
        raise RuntimeError(
            "FX graph capture failed; inspect fx_graph_error.txt before treating deployment as valid."
        )
    probe_audit = export_standard_operator_probes(
        output_dir, args.opset, args.custom_op_probe, args.sample_seed + 1
    )

    if probe_audit["status"] not in ("passed", "skipped"):
        raise RuntimeError(
            "At least one standard-ONNX custom-operator replacement probe failed; "
            "inspect custom_operator_probe.csv before treating deployment as valid."
        )

    # Keep the requested deployment filename and a stable downstream alias.
    onnx_path = output_dir / "generator.onnx"
    raw_onnx_path = output_dir / "generator_fp32_raw.onnx"
    torch.onnx.export(
        model,
        example,
        onnx_path,
        input_names=["z"],
        output_names=["image"],
        dynamic_axes={"z": {0: "batch"}, "image": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    shutil.copy2(onnx_path, raw_onnx_path)

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "onnx is required for checker validation. Install it in the runtime first."
        ) from exc

    onnx_model, counts, custom_nodes = inventory_onnx(
        onnx_path, output_dir / "operator_inventory_raw.csv"
    )
    custom_op_audit = {
        "fx_graph_capture": fx_capture,
        "standard_onnx_replacement_probes": probe_audit,
        "onnx_custom_nodes": custom_nodes,
        "custom_domain_count": len(custom_nodes),
        "implementation_status": (
            "requires_runtime_implementation"
            if custom_nodes
            else "standard_onnx_graph;_probe_replacements_audited"
        ),
        "replacement_probe_status": probe_audit["status"],
        "wavelet_inference_operator_present": False,
        "dynamic_spectral_norm_inference_operator_present": False,
        "boundary": (
            "Exp11 Generator inference graph contains ConvTranspose, BatchNorm, "
            "Relu and Tanh only. The existing Haar wavelet code is a training-loss "
            "path and spectral normalization is used in discriminator/other models, "
            "not in this Generator inference graph."
        ),
    }
    metadata = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx": str(onnx_path),
        "raw_onnx_alias": str(raw_onnx_path),
        "opset": args.opset,
        "input": {"name": "z", "shape": ["batch", NOISE_DIM, 1, 1]},
        "output": {"name": "image", "shape": ["batch", 3, IMAGE_SIZE, IMAGE_SIZE]},
        "node_count": len(onnx_model.graph.node),
        "operator_counts": dict(sorted(counts.items())),
        "custom_nodes": custom_nodes,
        "checker": "passed",
        "shape_inference_checker": "passed",
        "fx_graph_capture": fx_capture,
        "custom_op_audit": custom_op_audit,
    }
    (output_dir / "onnx_check.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[checkpoint] {checkpoint}")
    print(f"[export] {onnx_path}")
    print(f"[checker] passed; nodes={len(onnx_model.graph.node)}")
    if custom_nodes:
        print(f"[custom-op] found {len(custom_nodes)} non-standard nodes")
    else:
        print("[custom-op] none; graph uses standard ONNX domains")
    print(f"[done] outputs: {output_dir}")


if __name__ == "__main__":
    main()
