"""
DCGAN Improve — standalone quality-first training script
=========================================================

This is an independent DCGAN-family model. It does not load Exp11 weights,
does not reuse Exp11's network state, and starts from random initialization.
Exp11 can be used only as an external historical reference after training.

Quality choices
---------------
1. Generator: nearest-neighbor upsample + 3x3 convolution residual blocks.
   This avoids the checkerboard artifacts commonly introduced by stacked
   ConvTranspose2d layers while keeping the output at 64x64.
2. Discriminator: spectral-normalized residual downsampling blocks, average
   pooling, and a minibatch-standard-deviation channel for coverage stability.
3. Objective: hinge GAN loss, conservative differentiable augmentation with an
   adaptive probability controller, and EMA for the generator. R1 is omitted
   because it was already a known regression in this project family.
4. Training length: not fixed at 200 epochs. A safety ceiling is combined with
   warm-up + patience + trend-based quick-FID early stopping. At the end, the
   best quick-FID candidate and final EMA are compared using full FID.

Important limitations
---------------------
- This is a 64x64 research-quality candidate, not a guarantee of commercial
  quality. FID alone cannot certify facial correctness, licensing, or safety.
- FID follows the project's legacy torchvision Inception-v3 protocol so that
  the result remains comparable with existing project measurements.
- The script is intentionally from scratch. It has only one optional resume
  path for this script's own checkpoint_latest.pth.

Kaggle usage
------------
Attach only the deduplicated Anime Faces image dataset, paste this complete
file into one Kaggle code cell, enable a GPU and Internet for the first
Inception-v3 weight download, and run it. No Exp11 weight dataset is needed.
"""

import argparse
import csv
import gc
import json
import os
import random
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import numpy as np
from PIL import Image
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import make_grid


EXPERIMENT_NAME = "DCGAN_Improve_Standalone_21K"
IMAGE_SIZE = 64
DATASET_LIMIT = 21000
LATENT_DIM = 128
BATCH_SIZE = 32
G_BASE = 32
LR_G = 1.5e-4
LR_D = 1.5e-4
BETAS = (0.0, 0.99)
SEED = 42

# A ceiling only. FID early stopping normally finishes earlier.
MAX_EPOCHS = 300
MIN_EPOCHS = 50
FID_INTERVAL = 10
FID_PATIENCE = 4
FID_MIN_DELTA = 0.20
FID_RISE_MARGIN = 0.50
QUICK_FID_N = 2000
FINAL_FID_N = 10000
EMA_DECAY = 0.999

# Conservative adaptive augmentation for a 21K image pool.
ADA_TARGET = 0.60
ADA_STEP = 0.01
ADA_INTERVAL_STEPS = 256
ADA_MAX_P = 0.40
TRANSLATION_RATIO = 0.08
CUTOUT_RATIO = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def default_output_root():
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return str(kaggle / "dcgan_output")
    return str(Path.cwd() / "outputs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset-path", default=os.getenv("DATASET_PATH", ""))
    parser.add_argument("--dataset-limit", type=int, default=DATASET_LIMIT)
    parser.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT", default_output_root()))
    parser.add_argument("--resume", default=os.getenv("RESUME_PATH", ""))
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--fid-interval", type=int, default=FID_INTERVAL)
    parser.add_argument("--fid-patience", type=int, default=FID_PATIENCE)
    parser.add_argument("--fid-min-delta", type=float, default=FID_MIN_DELTA)
    parser.add_argument("--fid-rise-margin", type=float, default=FID_RISE_MARGIN)
    parser.add_argument("--quick-fid", type=int, default=QUICK_FID_N)
    parser.add_argument("--final-fid", type=int, default=FINAL_FID_N)
    # Kaggle injects -f kernel.json; parse_known_args ignores it.
    args, _ = parser.parse_known_args(argv)
    if args.dataset_limit < 1:
        raise ValueError("--dataset-limit must be positive")
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.min_epochs < 1 or args.min_epochs > args.max_epochs:
        raise ValueError("--min-epochs must be in [1, --max-epochs]")
    if args.fid_interval < 1 or args.fid_patience < 1:
        raise ValueError("FID interval and patience must be positive")
    if args.quick_fid < 2 or args.final_fid < 2:
        raise ValueError("FID sample counts must be at least 2")
    return args


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic convolutions make the fixed-noise FID trend easier to read.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def find_images(root):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        str(p) for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def discover_dataset(hint=""):
    candidates = []
    if hint:
        candidates.append(Path(hint))
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        candidates.extend(sorted(p for p in kaggle.iterdir() if p.is_dir()))
    candidates.extend([Path("./data"), Path("./dataset")])
    for candidate in candidates:
        paths = find_images(candidate)
        if paths:
            print(f"[dataset] {candidate} ({len(paths)} images)")
            return str(candidate), paths
    raise FileNotFoundError(
        "No image dataset found. Attach the deduplicated Anime Faces dataset "
        "or pass --dataset-path."
    )


class AnimeDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for _ in range(10):
            try:
                image = Image.open(self.paths[index]).convert("RGB")
                return self.transform(image)
            except (OSError, IOError):
                index = random.randrange(len(self.paths))
        raise RuntimeError("Unable to read image after 10 retries")


def train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.LANCZOS),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class GResBlock(nn.Module):
    """Upsample + convolution residual block; no transposed convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        skip = self.skip(x)
        y = self.conv1(F.relu(self.bn1(x), inplace=True))
        y = self.conv2(F.relu(self.bn2(y), inplace=True))
        return y + skip


class ImproveGenerator(nn.Module):
    """Independent 4x4 -> 64x64 residual upsampling generator."""

    def __init__(self, latent_dim=LATENT_DIM, base=G_BASE):
        super().__init__()
        c4, c8, c16, c32, c64 = base * 16, base * 8, base * 4, base * 2, base
        self.latent_dim = latent_dim
        self.fc = nn.Linear(latent_dim, c4 * 4 * 4)
        self.block8 = GResBlock(c4, c8)
        self.block16 = GResBlock(c8, c16)
        self.block32 = GResBlock(c16, c32)
        self.block64 = GResBlock(c32, c64)
        self.out_bn = nn.BatchNorm2d(c64)
        self.out_conv = nn.Conv2d(c64, 3, 3, 1, 1)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, z):
        x = self.fc(z.view(z.size(0), -1)).view(z.size(0), -1, 4, 4)
        x = self.block8(x)
        x = self.block16(x)
        x = self.block32(x)
        x = self.block64(x)
        return torch.tanh(self.out_conv(F.relu(self.out_bn(x), inplace=True)))


class DResBlock(nn.Module):
    """Spectral-normalized residual downsampling block with average pooling."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        sn = nn.utils.spectral_norm
        self.conv1 = sn(nn.Conv2d(in_channels, out_channels, 3, 1, 1))
        self.conv2 = sn(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
        self.skip = sn(nn.Conv2d(in_channels, out_channels, 1, 1, 0))

    def forward(self, x):
        y = F.leaky_relu(self.conv1(x), 0.2, inplace=True)
        y = F.leaky_relu(self.conv2(y), 0.2, inplace=True)
        y = F.avg_pool2d(y, 2)
        return y + F.avg_pool2d(self.skip(x), 2)


class MinibatchStd(nn.Module):
    def forward(self, x):
        if x.size(0) <= 1:
            value = x.new_zeros(1, 1, 1, 1)
        else:
            value = x.var(dim=0, unbiased=False).mean().sqrt().view(1, 1, 1, 1)
        return torch.cat([x, value.expand(x.size(0), 1, x.size(2), x.size(3))], dim=1)


class ImproveDiscriminator(nn.Module):
    """Independent SN residual discriminator for hinge loss."""

    def __init__(self, base=G_BASE):
        super().__init__()
        sn = nn.utils.spectral_norm
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.from_rgb = sn(nn.Conv2d(3, c1, 3, 1, 1))
        self.block32 = DResBlock(c1, c2)
        self.block16 = DResBlock(c2, c3)
        self.block8 = DResBlock(c3, c4)
        self.block4 = DResBlock(c4, c4 * 2)
        self.mbstd = MinibatchStd()
        self.final = sn(nn.Linear((c4 * 2 + 1) * 4 * 4, 1))

    def forward(self, x):
        x = F.leaky_relu(self.from_rgb(x), 0.2, inplace=True)
        x = self.block32(x)
        x = self.block16(x)
        x = self.block8(x)
        x = self.block4(x)
        x = self.mbstd(x)
        return self.final(x.flatten(1)).view(-1)


class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name].mul_(self.decay).add_(value.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def model_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.shadow.items()}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.model_state_dict()}

    def load_state_dict(self, state, model=None):
        self.decay = float(state["decay"])
        self.shadow = {k: v.clone() for k, v in state["shadow"].items()}
        if model is not None:
            for name, value in model.state_dict().items():
                if name not in self.shadow:
                    self.shadow[name] = value.detach().clone()


class ADAController:
    def __init__(self):
        self.p = 0.0
        self.sign_ema = ADA_TARGET

    def update(self, sign_rate):
        self.sign_ema = 0.95 * self.sign_ema + 0.05 * float(sign_rate)
        if self.sign_ema > ADA_TARGET:
            self.p += ADA_STEP
        else:
            self.p -= ADA_STEP
        self.p = float(np.clip(self.p, 0.0, ADA_MAX_P))

    def state_dict(self):
        return {"p": self.p, "sign_ema": self.sign_ema}

    def load_state_dict(self, state):
        self.p = float(state.get("p", 0.0))
        self.sign_ema = float(state.get("sign_ema", ADA_TARGET))


def _brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)


def _saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2
    return (x - mean) * scale + mean


def _contrast(x):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5
    return (x - mean) * scale + mean


def _translation(x):
    sx = int(x.size(2) * TRANSLATION_RATIO + 0.5)
    sy = int(x.size(3) * TRANSLATION_RATIO + 0.5)
    tx = torch.randint(-sx, sx + 1, (x.size(0),), device=x.device)
    ty = torch.randint(-sy, sy + 1, (x.size(0),), device=x.device)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    grid_x = (grid_x[None] - tx[:, None, None]).clamp(0, x.size(3) - 1)
    grid_y = (grid_y[None] - ty[:, None, None]).clamp(0, x.size(2) - 1)
    batch = torch.arange(x.size(0), device=x.device)[:, None, None]
    return x[batch, :, grid_y, grid_x].permute(0, 3, 1, 2).contiguous()


def _cutout(x):
    h, w = x.size(2), x.size(3)
    cut_h, cut_w = max(1, int(h * CUTOUT_RATIO)), max(1, int(w * CUTOUT_RATIO))
    cy = torch.randint(0, h, (x.size(0),), device=x.device)
    cx = torch.randint(0, w, (x.size(0),), device=x.device)
    yy = torch.arange(h, device=x.device)[None, :, None]
    xx = torch.arange(w, device=x.device)[None, None, :]
    mask = ((yy - cy[:, None, None]).abs() < cut_h // 2) & ((xx - cx[:, None, None]).abs() < cut_w // 2)
    return x.masked_fill(mask[:, None], 0.0)


def adaptive_diffaugment(x, p):
    if p <= 0:
        return x
    out = x
    for fn in (_brightness, _saturation, _contrast, _translation, _cutout):
        apply = (torch.rand(x.size(0), 1, 1, 1, device=x.device) < p).to(x.dtype)
        candidate = fn(out)
        out = candidate * apply + out * (1 - apply)
    return out.contiguous()


class FIDCalculator:
    def __init__(self, device):
        self.device = device
        inc = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        inc.fc = nn.Identity()
        self.inc = inc.eval().to(device)
        for param in self.inc.parameters():
            param.requires_grad_(False)

    @torch.no_grad()
    def features(self, images):
        x = (images + 1.0) / 2.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        result = self.inc(x)
        if hasattr(result, "logits"):
            result = result.logits
        if isinstance(result, tuple):
            result = result[0]
        return result.detach().cpu().numpy()

    @torch.no_grad()
    def real_stats(self, loader, count):
        values = []
        seen = 0
        for images in loader:
            values.append(self.features(images.to(self.device)))
            seen += images.size(0)
            if seen >= count:
                break
        features = np.concatenate(values, axis=0)[:count]
        return features.mean(axis=0), np.cov(features, rowvar=False)

    @torch.no_grad()
    def fake_stats(self, model, noise, count):
        model.eval()
        values = []
        for start in range(0, count, 64):
            batch = noise[start:start + min(64, count - start)].to(self.device)
            values.append(self.features(model(batch)))
        features = np.concatenate(values, axis=0)[:count]
        return features.mean(axis=0), np.cov(features, rowvar=False)

    @staticmethod
    def from_stats(real_mean, real_cov, fake_mean, fake_cov):
        diff = real_mean - fake_mean
        covmean = linalg.sqrtm(real_cov.dot(fake_cov))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(real_cov + fake_cov - 2 * covmean))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def save_grid(images, path, nrow=8):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    array = grid.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(array).save(path)


@torch.no_grad()
def sample_images(model, noise, count):
    model.eval()
    result = []
    for start in range(0, count, 64):
        batch = noise[start:start + min(64, count - start)].to(DEVICE)
        result.append(model(batch).cpu())
    return torch.cat(result, dim=0)[:count]


def laplacian_values(images):
    values = []
    for image in images:
        array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.asarray(values, dtype=np.float64)


def edge_density(images):
    values = []
    for image in images:
        array = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
        values.append(float((cv2.Canny(gray, 50, 150) > 0).mean()))
    return float(np.mean(values))


def main(argv=None):
    args = parse_args(argv)
    seed_everything(SEED)
    device = DEVICE
    output_dir = Path(args.output_root) / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, paths = discover_dataset(args.dataset_path)
    if len(paths) > args.dataset_limit:
        rng = random.Random(SEED)
        paths = sorted(rng.sample(paths, args.dataset_limit))
    if len(paths) < 2:
        raise RuntimeError("At least two valid images are required")
    print(f"[init] standalone random initialization; external weights are not used")
    print(f"[dataset] training pool: {len(paths)} images")
    print(f"[device] {device}")

    train_ds = AnimeDataset(paths, train_transform())
    eval_ds = AnimeDataset(paths, eval_transform())
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=False,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=64, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=(device.type == "cuda"), persistent_workers=False,
    )

    generator = ImproveGenerator().to(device)
    discriminator = ImproveDiscriminator().to(device)
    ema = EMA(generator)
    g_opt = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=BETAS)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=BETAS)
    ada = ADAController()
    start_epoch = 1
    global_step = 0

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        state = torch.load(resume_path, map_location="cpu")
        generator.load_state_dict(state["generator"], strict=True)
        discriminator.load_state_dict(state["discriminator"], strict=True)
        ema.load_state_dict(state["ema"], model=generator)
        g_opt.load_state_dict(state["optimizer_G"])
        d_opt.load_state_dict(state["optimizer_D"])
        ada.load_state_dict(state.get("ada", {}))
        start_epoch = int(state.get("epoch", 0)) + 1
        global_step = int(state.get("global_step", 0))
        print(f"[resume] continuing this standalone run from epoch {start_epoch}")

    fixed_noise = torch.randn(64, LATENT_DIM, 1, 1, device=device)
    fid_noise = torch.randn(max(args.quick_fid, args.final_fid), LATENT_DIM, 1, 1)
    config = {
        "experiment": EXPERIMENT_NAME,
        "standalone_from_scratch": True,
        "architecture": "upsample_conv_residual_G + SN_residual_avgpool_D + minibatch_std",
        "dataset_root": dataset_root,
        "dataset_size": len(paths),
        "image_size": IMAGE_SIZE,
        "latent_dim": LATENT_DIM,
        "batch_size": BATCH_SIZE,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "fid_early_stopping": {
            "interval": args.fid_interval,
            "patience": args.fid_patience,
            "min_delta": args.fid_min_delta,
            "rise_margin": args.fid_rise_margin,
        },
        "optimizer": {"lr_g": LR_G, "lr_d": LR_D, "betas": BETAS},
        "ema_decay": EMA_DECAY,
        "augmentation": "adaptive differentiable color+translation+cutout",
        "fid_protocol": "project legacy torchvision Inception-v3",
        "commercial_note": "64x64 FID is not a commercial-quality or licensing guarantee.",
    }
    save_json(output_dir / "training_config.json", config)
    (output_dir / "dataset_manifest.txt").write_text("\n".join(paths), encoding="utf-8")

    fid_calc = None
    fid_ready = False
    fid_error = None
    real_stats_quick = None
    real_stats_final = None
    try:
        print("[fid] loading Inception-v3 and caching real statistics")
        fid_calc = FIDCalculator(device)
        real_stats_quick = fid_calc.real_stats(eval_loader, args.quick_fid)
        real_stats_final = fid_calc.real_stats(eval_loader, args.final_fid)
        fid_ready = True
    except Exception as exc:
        fid_error = str(exc)
        print(f"[fid] unavailable; early stopping will not be active: {exc}")

    loss_path = output_dir / "loss.csv"
    loss_mode = "a" if args.resume and loss_path.exists() else "w"
    loss_file = open(loss_path, loss_mode, newline="", encoding="utf-8")
    loss_writer = csv.writer(loss_file)
    if loss_mode == "w":
        loss_writer.writerow(["epoch", "D_loss", "G_loss", "D_real", "D_fake", "real_sign_rate", "ada_p"])

    best_quick_fid = float("inf")
    best_quick_epoch = None
    best_ema_state = None
    fid_history = []
    bad_fid_evals = 0
    completed_epoch = start_epoch - 1
    stop_reason = None
    if args.resume:
        history_path = output_dir / "fid_history.json"
        best_path = output_dir / "generator_ema_best_quick.pth"
        if history_path.exists():
            try:
                loaded_history = json.loads(history_path.read_text(encoding="utf-8"))
                if isinstance(loaded_history, list):
                    fid_history = loaded_history
                    valid = [x for x in fid_history if float(x.get("quick_fid", -1)) >= 0]
                    if valid:
                        best_item = min(valid, key=lambda x: float(x["quick_fid"]))
                        best_quick_fid = float(best_item["quick_fid"])
                        best_quick_epoch = int(best_item["epoch"])
                        bad_fid_evals = int(valid[-1].get("bad_fid_evals", 0))
                        if best_path.exists():
                            best_ema_state = torch.load(best_path, map_location="cpu")
                        print(
                            f"[resume] restored FID history: {len(fid_history)} evaluations, "
                            f"best={best_quick_fid:.4f} at epoch {best_quick_epoch}"
                        )
            except Exception as exc:
                print(f"[resume] could not restore FID history: {exc}")

    print(
        f"[run] {EXPERIMENT_NAME}; max_epochs={args.max_epochs}; "
        f"min_epochs={args.min_epochs}; steps/epoch={len(train_loader)}"
    )
    for epoch in range(start_epoch, args.max_epochs + 1):
        completed_epoch = epoch
        generator.train()
        discriminator.train()
        last_d = last_g = last_dr = last_df = 0.0
        sign_values = []

        for real in train_loader:
            real = real.to(device, non_blocking=True)
            batch_size = real.size(0)

            d_opt.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            with torch.no_grad():
                fake = generator(z)
            d_real = discriminator(adaptive_diffaugment(real, ada.p))
            d_fake = discriminator(adaptive_diffaugment(fake, ada.p))
            d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            d_loss.backward()
            d_opt.step()

            discriminator.requires_grad_(False)
            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=device)
            fake_for_g = generator(z)
            g_loss = -discriminator(adaptive_diffaugment(fake_for_g, ada.p)).mean()
            g_loss.backward()
            g_opt.step()
            discriminator.requires_grad_(True)
            ema.update(generator)

            last_d = float(d_loss.item())
            last_g = float(g_loss.item())
            last_dr = float(d_real.mean().item())
            last_df = float(d_fake.mean().item())
            sign_values.append(float((d_real > 0).float().mean().item()))
            global_step += 1
            if global_step % ADA_INTERVAL_STEPS == 0 and sign_values:
                ada.update(np.mean(sign_values))
                sign_values = []

        if sign_values:
            ada.update(np.mean(sign_values))
        sign_rate = float(np.mean(sign_values)) if sign_values else float(ada.sign_ema)
        loss_writer.writerow([epoch, last_d, last_g, last_dr, last_df, sign_rate, ada.p])
        loss_file.flush()
        print(
            f"Epoch [{epoch:03d}/{args.max_epochs}] D={last_d:.4f} G={last_g:.4f} "
            f"D(real/fake)={last_dr:+.3f}/{last_df:+.3f} ADA_p={ada.p:.3f}"
        )

        if epoch % 10 == 0 or epoch == args.max_epochs:
            train_state = deepcopy(generator.state_dict())
            ema.apply_to(generator)
            generator.eval()
            save_grid(generator(fixed_noise), output_dir / f"epoch_{epoch:03d}.png")
            generator.load_state_dict(train_state, strict=True)
            # One overwriteable checkpoint is enough for interruption recovery.
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "ema": ema.state_dict(),
                "ada": ada.state_dict(),
                "optimizer_G": g_opt.state_dict(),
                "optimizer_D": d_opt.state_dict(),
                "config": config,
            }, output_dir / "checkpoint_latest.pth")

        should_stop = False
        if fid_ready and (epoch % args.fid_interval == 0 or epoch == args.max_epochs):
            train_state = deepcopy(generator.state_dict())
            ema.apply_to(generator)
            fake_mean, fake_cov = fid_calc.fake_stats(generator, fid_noise, args.quick_fid)
            quick_fid = FIDCalculator.from_stats(
                real_stats_quick[0], real_stats_quick[1], fake_mean, fake_cov
            )
            improved = quick_fid < best_quick_fid - args.fid_min_delta
            if improved:
                best_quick_fid = quick_fid
                best_quick_epoch = epoch
                best_ema_state = ema.model_state_dict()
                bad_fid_evals = 0
                torch.save(best_ema_state, output_dir / "generator_ema_best_quick.pth")
                save_json(output_dir / "best_checkpoint.json", {
                    "epoch": epoch,
                    "quick_fid": quick_fid,
                    "reason": "best quick FID so far",
                })
            else:
                bad_fid_evals += 1
            fid_history.append({
                "epoch": epoch,
                "quick_fid": quick_fid,
                "best_quick_fid": best_quick_fid,
                "bad_fid_evals": bad_fid_evals,
                "ada_p": ada.p,
                "improved": improved,
            })
            save_json(output_dir / "fid_history.json", fid_history)
            print(
                f"[fid] epoch={epoch} quick={quick_fid:.4f} "
                f"best={best_quick_fid:.4f} bad={bad_fid_evals}/{args.fid_patience}"
            )
            recent = [x["quick_fid"] for x in fid_history[-3:]]
            rising = len(recent) == 3 and recent[-1] > recent[-2] > recent[-3]
            should_stop = (
                epoch >= args.min_epochs
                and bad_fid_evals >= args.fid_patience
                and quick_fid > best_quick_fid + args.fid_rise_margin
                and rising
            )
            generator.load_state_dict(train_state, strict=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if should_stop:
                stop_reason = (
                    f"quick FID rose for {args.fid_patience} evaluations after "
                    f"warm-up epoch {args.min_epochs}"
                )
                print(f"[early-stop] {stop_reason}")
                break

    if stop_reason is None:
        stop_reason = (
            "reached max_epochs or completed requested training"
            if fid_ready else "FID unavailable; reached safety max_epochs"
        )
    save_json(output_dir / "stop_reason.json", {
        "reason": stop_reason,
        "completed_epoch": completed_epoch,
        "max_epochs": args.max_epochs,
        "best_quick_epoch": best_quick_epoch,
        "best_quick_fid": best_quick_fid if np.isfinite(best_quick_fid) else None,
        "fid_error": fid_error,
    })
    loss_file.close()

    torch.save(generator.state_dict(), output_dir / "generator_raw_final.pth")
    torch.save(discriminator.state_dict(), output_dir / "discriminator_final.pth")
    final_ema_state = ema.model_state_dict()
    torch.save(final_ema_state, output_dir / "generator_ema_final.pth")
    torch.save(ema.state_dict(), output_dir / "ema_state_final.pth")

    torch.save({
        "epoch": completed_epoch,
        "global_step": global_step,
        "generator": generator.state_dict(),
        "discriminator": discriminator.state_dict(),
        "ema": ema.state_dict(),
        "ada": ada.state_dict(),
        "optimizer_G": g_opt.state_dict(),
        "optimizer_D": d_opt.state_dict(),
        "config": config,
    }, output_dir / "checkpoint_final.pth")

    selected_state = final_ema_state
    selected_label = "final_ema"
    full_fid_best = None
    full_fid_final = None
    if fid_ready and real_stats_final is not None:
        for label, state in (("final_ema", final_ema_state), ("best_quick", best_ema_state)):
            if state is None:
                continue
            generator.load_state_dict(state, strict=True)
            fake_mean, fake_cov = fid_calc.fake_stats(generator, fid_noise, args.final_fid)
            full_fid = FIDCalculator.from_stats(
                real_stats_final[0], real_stats_final[1], fake_mean, fake_cov
            )
            print(f"[fid] {label} full={full_fid:.4f}")
            if label == "final_ema":
                full_fid_final = full_fid
            else:
                full_fid_best = full_fid
        if full_fid_best is not None and full_fid_best < full_fid_final:
            selected_state = best_ema_state
            selected_label = "best_quick_full_eval"
        else:
            selected_label = "final_ema_full_eval"
    torch.save(selected_state, output_dir / "generator_ema_deploy.pth")

    generator.load_state_dict(selected_state, strict=True)
    fake_images = sample_images(generator, fixed_noise, 64)
    save_grid(fake_images, output_dir / "deploy_sample_grid.png")
    real_batch = next(iter(DataLoader(AnimeDataset(paths[:64], eval_transform()), batch_size=64)))
    fake01 = (fake_images + 1) / 2
    real01 = (real_batch + 1) / 2
    fake_lap = laplacian_values(fake01)
    real_lap = laplacian_values(real01)
    threshold = float(np.percentile(real_lap, 10))
    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "standalone_from_scratch": True,
        "dataset_size": len(paths),
        "completed_epoch": completed_epoch,
        "stop_reason": stop_reason,
        "best_quick_fid": best_quick_fid if np.isfinite(best_quick_fid) else None,
        "best_quick_epoch": best_quick_epoch,
        "full_fid_best_quick": full_fid_best,
        "full_fid_final_ema": full_fid_final,
        "selected_deploy_state": selected_label,
        "selected_full_fid": min(
            [x for x in (full_fid_best, full_fid_final) if x is not None], default=None
        ),
        "laplacian_mean_fake_64": float(fake_lap.mean()),
        "laplacian_mean_real_64": float(real_lap.mean()),
        "blur_threshold_real_p10": threshold,
        "blur_rate_fake_below_real_p10": float((fake_lap < threshold).mean()),
        "edge_density_fake_64": edge_density(fake01),
        "fid_protocol": "Legacy project Inception-v3; compare only with same protocol.",
        "fid_error": fid_error,
        "fid_history": fid_history,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_dir / "metrics.json", metrics)
    print(f"[done] {EXPERIMENT_NAME}")
    print(f"[done] deploy weight: {output_dir / 'generator_ema_deploy.pth'}")
    print(f"[done] metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
