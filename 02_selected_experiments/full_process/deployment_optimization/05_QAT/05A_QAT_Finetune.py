"""Task 5A: standalone FakeQuantize QAT fine-tuning for the Exp11 Generator.

05A deliberately stops after PyTorch training.  It does not merge ONNX graphs,
does not call ONNX Runtime, and does not build TensorRT.  This makes a failed
deployment build independent from the QAT result.

The confirmed 04C policy is used exactly:
    net.0 and net.12: FP16 at deployment, no INT8 fake quantizer;
    net.3, net.6 and net.9: INT8 input/weight FakeQuantize.

The teacher is the best Exp11 EMA Generator checkpoint.  The default student
is fine-tuned on fixed latent vectors with output distillation, Haar detail loss,
and a small image-gradient loss. Intermediate-feature, Laplacian, and weight-
anchor losses remain explicit ablation options. Both the calibrated pre-QAT
checkpoint and the selected post-QAT checkpoint are saved.  05B builds and
evaluates both, so the report can show QAT training before/after metrics rather
than confusing PTQ with the pre-QAT checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IMAGE_SIZE = 64
NOISE_DIM = 128
FP16_LAYERS = ("net.0", "net.12")
INT8_LAYERS = ("net.3", "net.6", "net.9")
LAYERS = ("net.0", "net.3", "net.6", "net.9", "net.12")
QAT_LAYOUT_VERSION = 4
QAT_TRAINING_REVISION = 3


class Exp11Generator(nn.Module):
    """The exact 768->384->192->96->3 Exp11 Generator."""

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
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )


def make_weight_fake_quant():
    from torch.ao.quantization import FakeQuantize, MovingAverageMinMaxObserver

    return FakeQuantize(
        # TensorRT ConvTranspose explicit Q/DQ is not reliable with the
        # vector weight scales emitted by the per-channel observer on the
        # tested Kaggle runtimes.  A scalar scale keeps QAT and deployment
        # numerically aligned while the layer-level mixed-precision policy is
        # unchanged.
        observer=MovingAverageMinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )


class QATConvTranspose2d(nn.ConvTranspose2d):
    """ConvTranspose with deployment-aligned STE fake quantization.

    The activation fake quantizer is on the input of the weighted operation,
    and the weight fake quantizer uses one scalar scale.  There is intentionally no
    trailing output fake quantizer: that trailing Q/DQ is the layout that
    previously made the TensorRT deconvolution path fragile and it needlessly
    destroys high-frequency detail after the operation.
    """

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
    """QAT student: INT8 simulation only on the three approved middle layers."""

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
    p = argparse.ArgumentParser(description="05A standalone hybrid FakeQuantize QAT")
    p.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH", ""), help="Best Exp11 EMA Generator checkpoint")
    p.add_argument("--protocol-path", default=os.getenv("PROTOCOL_PATH", ""), help="03A extracted folder or ZIP")
    p.add_argument("--policy-manifest", default=os.getenv("POLICY_MANIFEST_PATH", ""), help="04C final_confirmation_manifest.json")
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/05_QAT/05A_QAT_Training"))
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--distill-l1-weight", type=float, default=1.0)
    p.add_argument("--distill-mse-weight", type=float, default=0.25)
    p.add_argument("--highfreq-weight", type=float, default=0.5)
    p.add_argument("--laplacian-weight", type=float, default=0.0)
    p.add_argument("--gradient-weight", type=float, default=0.1)
    p.add_argument("--feature9-weight", type=float, default=0.0)
    p.add_argument("--feature12-weight", type=float, default=0.0)
    p.add_argument("--weight-anchor-weight", type=float, default=0.0)
    p.add_argument("--train-pool", type=int, default=384)
    p.add_argument("--selection-count", type=int, default=128, help="Held-out subset of latent_calibration used only for checkpoint selection")
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--observer-calibration-batches", type=int, default=16)
    p.add_argument("--early-stop-patience", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--no-zip", action="store_true")
    args, _ = p.parse_known_args(argv)
    if args.steps < 0 or args.batch_size < 1 or args.lr <= 0:
        raise ValueError("steps must be non-negative, batch-size positive and lr positive")
    if args.train_pool < 1 or args.eval_every < 1:
        raise ValueError("train-pool and eval-every must be positive")
    if args.early_stop_patience < 0:
        raise ValueError("early-stop-patience must be non-negative")
    weights = (
        args.distill_l1_weight, args.distill_mse_weight, args.highfreq_weight,
        args.laplacian_weight, args.gradient_weight, args.feature9_weight,
        args.feature12_weight, args.weight_anchor_weight,
    )
    if any(float(weight) < 0 for weight in weights):
        raise ValueError("All loss weights must be non-negative")
    if args.selection_count < 1:
        raise ValueError("selection-count must be positive")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        raise FileNotFoundError(f"{name} was not found; pass the explicit path")
    by_hash = {}
    for item in unique:
        by_hash.setdefault(sha256(item), []).append(item)
    if len(by_hash) != 1:
        raise RuntimeError("Multiple different files found; pass an explicit path:\n" + "\n".join(map(str, unique)))
    return sorted(next(iter(by_hash.values())), key=lambda item: (0 if str(item).startswith("/kaggle/input") else 1, str(item)))[0]


def locate_protocol(explicit: str, staging: Path) -> Path:
    def valid(path: Path):
        return (path / "latent_calibration.npy").is_file() and (path / "latent_eval.npy").is_file()

    if explicit:
        source = Path(explicit).expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Protocol path not found: {source}")
        if source.is_dir():
            if valid(source):
                return source
            options = [item.parent for item in source.rglob("latent_calibration.npy") if valid(item.parent)]
            if len(options) == 1:
                return options[0]
            raise RuntimeError(f"Could not identify one 03A protocol folder under {source}")
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
        options = [item.parent for item in staging.rglob("latent_calibration.npy") if valid(item.parent)]
        if len(options) != 1:
            raise RuntimeError(f"Expected one protocol folder in {source}, found {len(options)}")
        return options[0]

    options = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            try:
                options.extend(item.parent for item in root.rglob("latent_calibration.npy") if valid(item.parent))
            except PermissionError:
                pass
    unique = sorted({item.resolve() for item in options})
    if len(unique) == 1:
        return unique[0]
    raise FileNotFoundError("03A protocol was not uniquely found; pass --protocol-path")


def load_generator_state(path: Path):
    payload = torch.load(path, map_location="cpu")
    state = payload
    if isinstance(payload, dict):
        for key in ("generator_ema", "generator", "model_state_dict", "state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, dict) and any(str(k).endswith("weight") for k in candidate):
                state = candidate
                break
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload: {type(state)}")
    cleaned = {}
    for key, value in state.items():
        key = str(key)
        while key.startswith("module."):
            key = key[7:]
        cleaned[key] = value
    expected = {"net.0.weight", "net.3.weight", "net.6.weight", "net.9.weight", "net.12.weight"}
    missing = sorted(expected - set(cleaned))
    if missing:
        raise RuntimeError(f"Checkpoint is not the Exp11 Generator; missing keys: {missing}")
    return cleaned


def load_into(model: nn.Module, state: dict, allow_fake_quant_missing=True):
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = []
    if allow_fake_quant_missing:
        bad_missing = [
            item for item in missing
            if ".input_fake_quant." not in item and ".weight_fake_quant." not in item
        ]
    else:
        bad_missing = list(missing)
    if bad_missing or unexpected:
        raise RuntimeError(f"Checkpoint/model mismatch: missing={bad_missing}, unexpected={unexpected}")


def fake_quant_modules(model):
    from torch.ao.quantization import FakeQuantize
    return [module for module in model.modules() if isinstance(module, FakeQuantize)]


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def calibrate_observers(model, latents, device, batch_size):
    model.eval()
    for module in fake_quant_modules(model):
        module.enable_observer()
        module.enable_fake_quant()
    with torch.no_grad():
        for start in range(0, len(latents), batch_size):
            model(latents[start:start + batch_size].to(device))
    for module in fake_quant_modules(model):
        module.disable_observer()
    freeze_batch_norm(model)


def clone_state(model):
    return {key: value.detach().cpu().clone() if torch.is_tensor(value) else value for key, value in model.state_dict().items()}


def restore_state(model, state):
    model.load_state_dict(state, strict=True)


def haar_detail(x):
    even_h, odd_h = x[..., 0::2, :], x[..., 1::2, :]
    ll = (even_h[..., :, 0::2] + even_h[..., :, 1::2] + odd_h[..., :, 0::2] + odd_h[..., :, 1::2]) * 0.5
    lh = (even_h[..., :, 0::2] - even_h[..., :, 1::2] + odd_h[..., :, 0::2] - odd_h[..., :, 1::2]) * 0.5
    hl = (even_h[..., :, 0::2] + even_h[..., :, 1::2] - odd_h[..., :, 0::2] - odd_h[..., :, 1::2]) * 0.5
    hh = (even_h[..., :, 0::2] - even_h[..., :, 1::2] - odd_h[..., :, 0::2] + odd_h[..., :, 1::2]) * 0.5
    return ll, lh, hl, hh


def laplacian(x):
    gray = x.mean(dim=1, keepdim=True)
    kernel = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], device=x.device, dtype=x.dtype).view(1, 1, 3, 3)
    return F.conv2d(gray, kernel, padding=1)


def gradient_l1(prediction, reference):
    """Compare first-order image gradients without introducing a new op."""
    prediction_horizontal = prediction[..., :, 1:] - prediction[..., :, :-1]
    reference_horizontal = reference[..., :, 1:] - reference[..., :, :-1]
    prediction_vertical = prediction[..., 1:, :] - prediction[..., :-1, :]
    reference_vertical = reference[..., 1:, :] - reference[..., :-1, :]
    horizontal = F.l1_loss(prediction_horizontal, reference_horizontal)
    vertical = F.l1_loss(prediction_vertical, reference_vertical)
    return 0.5 * (horizontal + vertical)


def forward_with_taps(model, z, capture_taps=True):
    """Run the Sequential generator and retain the two detail-sensitive taps."""
    taps = {}
    value = z
    for index, module in enumerate(model.net):
        value = module(value)
        if capture_taps and index in (9, 12):
            taps[f"net.{index}"] = value
    return value, taps


@torch.no_grad()
def validation(student, teacher, latents, device, batch_size, weights):
    student.eval()
    teacher.eval()
    totals = {
        "l1": 0.0,
        "mse": 0.0,
        "highfreq": 0.0,
        "laplacian": 0.0,
        "gradient": 0.0,
        "feature9": 0.0,
        "feature12": 0.0,
    }
    batches = 0
    for start in range(0, len(latents), batch_size):
        z = latents[start:start + batch_size].to(device)
        capture_taps = weights["feature9"] > 0 or weights["feature12"] > 0
        reference, teacher_taps = forward_with_taps(teacher, z, capture_taps=capture_taps)
        prediction, student_taps = forward_with_taps(student, z, capture_taps=capture_taps)
        totals["l1"] += float(F.l1_loss(prediction, reference).item())
        totals["mse"] += float(F.mse_loss(prediction, reference).item())
        ref_detail = torch.cat(haar_detail(reference)[1:], dim=1)
        pred_detail = torch.cat(haar_detail(prediction)[1:], dim=1)
        totals["highfreq"] += float(F.l1_loss(pred_detail, ref_detail).item())
        if weights["laplacian"] > 0:
            totals["laplacian"] += float(F.l1_loss(laplacian(prediction), laplacian(reference)).item())
        if weights["gradient"] > 0:
            totals["gradient"] += float(gradient_l1(prediction, reference).item())
        if weights["feature9"] > 0:
            totals["feature9"] += float(F.l1_loss(student_taps["net.9"], teacher_taps["net.9"]).item())
        if weights["feature12"] > 0:
            totals["feature12"] += float(F.l1_loss(student_taps["net.12"], teacher_taps["net.12"]).item())
        batches += 1
    freeze_batch_norm(student)
    values = {key: value / max(batches, 1) for key, value in totals.items()}
    values["objective"] = (
        weights["l1"] * values["l1"]
        + weights["mse"] * values["mse"]
        + weights["highfreq"] * values["highfreq"]
        + weights["laplacian"] * values["laplacian"]
        + weights["gradient"] * values["gradient"]
        + weights["feature9"] * values["feature9"]
        + weights["feature12"] * values["feature12"]
    )
    return values


def anchor_loss(student, baseline, device):
    total = torch.zeros((), device=device)
    count = 0
    for name, parameter in student.named_parameters():
        reference = baseline.get(name)
        if reference is None or not parameter.requires_grad:
            continue
        total = total + F.mse_loss(parameter, reference.to(device=device, dtype=parameter.dtype))
        count += 1
    return total / max(count, 1)


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_log(rows, path: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "not_created", "reason": "matplotlib_missing"}
    steps = [int(row["step"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=160)
    for key, label in (
        ("objective", "objective"),
        ("l1", "pixel L1"),
        ("highfreq", "Haar detail L1"),
        ("gradient", "gradient L1"),
    ):
        axes[0].plot(steps, [float(row[key]) for row in rows], marker=".", label=label)
    axes[0].set_title("QAT validation")
    axes[0].legend(fontsize=8)
    axes[1].plot(steps, [float(row["laplacian"]) for row in rows], marker=".", label="Laplacian L1")
    axes[1].plot(steps, [float(row["feature9"]) for row in rows], marker=".", label="net.9 feature L1")
    axes[1].plot(steps, [float(row["feature12"]) for row in rows], marker=".", label="net.12 feature L1")
    axes[1].set_title("High-frequency preservation")
    axes[1].legend(fontsize=8)
    axes[2].plot(steps, [float(row["train_loss"]) for row in rows], marker=".", label="train loss")
    axes[2].plot(steps, [float(row["anchor"]) for row in rows], marker=".", label="anchor")
    axes[2].set_title("Training stability")
    axes[2].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("step (0 = pre-QAT)")
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {"status": "created", "file": path.name}


def write_zip(output_dir: Path):
    archive = output_dir.parent / "Task5_05A_QAT_Training.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in output_dir.iterdir():
            if path.is_file():
                bundle.write(path, arcname=path.name)
    return archive


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = find_one("generator_ema_final.pth", args.checkpoint)
    protocol = locate_protocol(args.protocol_path, output_dir / "_protocol")
    policy = {"selected_strategy": "net.0+net.12", "fp16_layers": list(FP16_LAYERS), "source": "04C confirmed policy"}
    if args.policy_manifest:
        manifest_path = Path(args.policy_manifest).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Policy manifest not found: {manifest_path}")
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = payload.get("selection_policy", {}).get("selected_strategy") or payload.get("selected_strategy")
        if selected and selected != "net.0+net.12":
            raise RuntimeError(f"The uploaded 04C policy is {selected}, expected net.0+net.12")
        policy.update({"source": str(manifest_path), "manifest_sha256": sha256(manifest_path)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline_state = load_generator_state(checkpoint)
    teacher = Exp11Generator().to(device).eval()
    teacher.load_state_dict(baseline_state, strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    student = HybridQATGenerator().to(device)
    load_into(student, baseline_state)
    freeze_batch_norm(student)
    calibration = np.load(protocol / "latent_calibration.npy").astype(np.float32, copy=False)
    # latent_eval is the final 05B test set.  Only inspect its shape/count in
    # 05A; never use it for checkpoint selection.
    final_eval = np.load(protocol / "latent_eval.npy", mmap_mode="r")
    expected_shape = (NOISE_DIM, 1, 1)
    if calibration.ndim != 4 or tuple(calibration.shape[1:]) != expected_shape:
        raise ValueError(f"Unexpected latent_calibration shape: {calibration.shape}")
    if final_eval.ndim != 4 or tuple(final_eval.shape[1:]) != expected_shape:
        raise ValueError(f"Unexpected latent_eval shape: {final_eval.shape}")
    final_eval_count = int(len(final_eval))
    del final_eval
    calibration = torch.from_numpy(calibration)
    if len(calibration) <= 1:
        raise RuntimeError("Need at least two calibration latents for train/selection split")
    selection_count = min(args.selection_count, max(1, len(calibration) // 2))
    train_count = min(args.train_pool, len(calibration) - selection_count)
    if train_count < 1:
        raise RuntimeError("selection-count leaves no calibration latents for QAT training")
    train_pool = calibration[:train_count]
    selection_pool = calibration[train_count:train_count + selection_count]
    if len(train_pool) < 1:
        raise RuntimeError("No training latent vectors available")

    calibrate_observers(student, calibration[: args.observer_calibration_batches * args.batch_size], device, args.batch_size)
    pre_state = clone_state(student)
    validation_weights = {
        "l1": args.distill_l1_weight,
        "mse": args.distill_mse_weight,
        "highfreq": args.highfreq_weight,
        "laplacian": args.laplacian_weight,
        "gradient": args.gradient_weight,
        "feature9": args.feature9_weight,
        "feature12": args.feature12_weight,
    }
    pre_metrics = validation(student, teacher, selection_pool, device, args.batch_size, validation_weights)
    torch.save({
        "model_state_dict": pre_state,
        "checkpoint_type": "calibrated_pre_qat",
        "qat_layout_version": QAT_LAYOUT_VERSION,
        "qat_training_revision": QAT_TRAINING_REVISION,
        "policy": policy,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "protocol": str(protocol),
    }, output_dir / "qat_pre.pth")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = pre_state
    best_metrics = pre_metrics
    best_step = 0
    rows = [{"phase": "pre_qat_calibrated", "step": 0, "train_loss": 0.0, "anchor": 0.0, **pre_metrics}]
    no_improve = 0
    generator = torch.Generator().manual_seed(args.seed + 101)
    student.train()
    freeze_batch_norm(student)
    for step in range(1, args.steps + 1):
        indices = torch.randint(len(train_pool), (args.batch_size,), generator=generator)
        z = train_pool[indices].to(device)
        with torch.no_grad():
            capture_taps = args.feature9_weight > 0 or args.feature12_weight > 0
            target, teacher_taps = forward_with_taps(teacher, z, capture_taps=capture_taps)
        prediction, student_taps = forward_with_taps(student, z, capture_taps=capture_taps)
        l1 = F.l1_loss(prediction, target)
        mse = F.mse_loss(prediction, target)
        highfreq = F.l1_loss(torch.cat(haar_detail(prediction)[1:], dim=1), torch.cat(haar_detail(target)[1:], dim=1))
        lap = F.l1_loss(laplacian(prediction), laplacian(target)) if args.laplacian_weight > 0 else prediction.new_zeros(())
        gradient = gradient_l1(prediction, target) if args.gradient_weight > 0 else prediction.new_zeros(())
        feature9 = F.l1_loss(student_taps["net.9"], teacher_taps["net.9"]) if args.feature9_weight > 0 else prediction.new_zeros(())
        feature12 = F.l1_loss(student_taps["net.12"], teacher_taps["net.12"]) if args.feature12_weight > 0 else prediction.new_zeros(())
        anchor = anchor_loss(student, baseline_state, device) if args.weight_anchor_weight > 0 else prediction.new_zeros(())
        loss = (
            args.distill_l1_weight * l1
            + args.distill_mse_weight * mse
            + args.highfreq_weight * highfreq
            + args.laplacian_weight * lap
            + args.gradient_weight * gradient
            + args.feature9_weight * feature9
            + args.feature12_weight * feature12
            + args.weight_anchor_weight * anchor
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()
        freeze_batch_norm(student)
        if step % args.eval_every == 0 or step == args.steps:
            metrics = validation(student, teacher, selection_pool, device, args.batch_size, validation_weights)
            row = {"phase": "qat", "step": step, "train_loss": float(loss.item()), "anchor": float(anchor.item()), **metrics}
            rows.append(row)
            if metrics["objective"] < best_metrics["objective"]:
                best_state = clone_state(student)
                best_metrics = metrics
                best_step = step
                no_improve = 0
            else:
                no_improve += args.eval_every
            if args.early_stop_patience and no_improve >= args.early_stop_patience:
                print(f"[early-stop] no validation improvement for {no_improve} steps")
                break

    student.load_state_dict(best_state, strict=True)
    torch.save({
        "model_state_dict": best_state,
        "checkpoint_type": "qat_best",
        "qat_layout_version": QAT_LAYOUT_VERSION,
        "qat_training_revision": QAT_TRAINING_REVISION,
        "policy": policy,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256(checkpoint),
        "protocol": str(protocol),
        "best_step": best_step,
        "pre_qat_metrics": pre_metrics,
        "best_validation_metrics": best_metrics,
    }, output_dir / "qat_best.pth")
    write_csv(output_dir / "qat_training_log.csv", rows)
    plot = plot_log(rows, output_dir / "qat_training_curves.png")
    manifest = {
        "task": "Task5_05A_QAT_Training",
        "status": "complete",
        "device": str(device),
        "architecture": "Exp11 Generator ConvTranspose 128->768->384->192->96->3",
        "policy": policy,
        "qat_layout_version": QAT_LAYOUT_VERSION,
        "fake_quant": {
            "activation": "per_tensor_symmetric int8 [-128,127] on ConvTranspose input",
            "weight": "per_tensor_symmetric int8 [-128,127], scalar scale",
            "quantized_layers": list(INT8_LAYERS),
            "protected_layers": list(FP16_LAYERS),
            "trailing_output_fake_quant": False,
        },
        "training_objective": {
            "default_profile": "output_l1_mse + Haar detail + image_gradient",
            "pixel_l1": "output distillation",
            "pixel_mse": "output distillation",
            "highfreq": "Haar LH/HL/HH detail L1",
            "gradient": "horizontal/vertical first-order gradient L1",
            "feature9": "pre-final 32x32 ConvTranspose feature L1",
            "feature12": "pre-Tanh final ConvTranspose feature L1",
            "optional_terms": "feature9, feature12, Laplacian, and weight anchor are disabled unless their CLI weights are positive",
            "selection_rule": "minimum weighted objective on a held-out latent_calibration split; latent_eval is reserved for final 05B testing",
        },
        "training_revision": QAT_TRAINING_REVISION,
        "inputs": {
            "baseline_checkpoint": str(checkpoint),
            "baseline_sha256": sha256(checkpoint),
            "protocol": str(protocol),
            "latent_calibration_count": int(len(calibration)),
            "latent_eval_count_available": final_eval_count,
            "train_latent_count": int(len(train_pool)),
            "selection_latent_count": int(len(selection_pool)),
            "train_latent_source": "latent_calibration prefix; selection split excluded",
            "selection_latent_source": "latent_calibration holdout; latent_eval excluded",
        },
        "training": vars(args),
        "selection": {"best_step": best_step, "pre_qat_objective": pre_metrics["objective"], "best_objective": best_metrics["objective"], "rollback_to_step_0_if_not_improved": True, "selection_set_is_disjoint_from_final_latent_eval": True},
        "metrics": {"pre_qat": pre_metrics, "qat_best": best_metrics},
        "outputs": {
            "qat_pre.pth": {"sha256": sha256(output_dir / "qat_pre.pth"), "bytes": (output_dir / "qat_pre.pth").stat().st_size},
            "qat_best.pth": {"sha256": sha256(output_dir / "qat_best.pth"), "bytes": (output_dir / "qat_best.pth").stat().st_size},
        },
        "plot": plot,
        "artifacts": ["qat_pre.pth", "qat_best.pth", "qat_training_log.csv", "qat_training_curves.png", "qat_manifest.json"],
        "deployment_note": "05B must export/build both qat_pre.pth and qat_best.pth; 05A intentionally does not call TensorRT.",
    }
    (output_dir / "qat_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    archive = None if args.no_zip else write_zip(output_dir)
    print(f"[05A] checkpoint={checkpoint}")
    print(f"[05A] pre_qat_objective={pre_metrics['objective']:.8g} best_objective={best_metrics['objective']:.8g} best_step={best_step}")
    print(f"[05A] output={output_dir}")
    if archive:
        print(f"[05A] upload this ZIP to Kaggle for 05B: {archive}")


if __name__ == "__main__":
    main()
