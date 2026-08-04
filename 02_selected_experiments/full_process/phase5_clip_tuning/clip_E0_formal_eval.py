"""
Formal CLIP fine-tuning experiment for the Exp11 DCGAN baseline.

The frozen CLIP image encoder supplies a distribution-level MMD loss.  There is
no random one-to-one real/fake pairing: an unconditional GAN has no such target.
The fake branch keeps autograd through the frozen encoder so gradients reach G;
real features are cached once.

This is a one-cell Kaggle program: paste the whole file into a code cell and
run it.  Dependencies, input discovery, evaluation, saving and ZIP packaging
are automatic.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path


def bootstrap_dependencies():
    """Install the two non-default Kaggle packages before importing them."""
    missing = [
        package
        for module, package in (
            ("open_clip", "open_clip_torch"),
            ("lpips", "lpips"),
        )
        if importlib.util.find_spec(module) is None
    ]
    if not missing:
        return
    print(f"[setup] installing: {', '.join(missing)}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *missing]
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Dependency installation failed. Enable Internet in Kaggle Notebook "
            "settings and run the same cell again."
        ) from exc


bootstrap_dependencies()

import cv2
import numpy as np
from PIL import Image, ImageFilter
from scipy import linalg

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.utils import make_grid

try:
    import open_clip
except ImportError as exc:
    raise ImportError("Install once with: pip install open_clip_torch lpips -q") from exc

try:
    import lpips
except ImportError as exc:
    raise ImportError("Install once with: pip install open_clip_torch lpips -q") from exc


IMAGE_SIZE = 64
NOISE_DIM = 128
DIFFAUG_POLICY = "color,translation,cutout"
DIFFAUG_TRANSLATION_RATIO = 0.125
DIFFAUG_CUTOUT_RATIO = 0.35
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
MMD_SIGMAS = (0.1, 0.2, 0.5, 1.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset-path", default="/kaggle/input/gananime-lite")
    p.add_argument("--output-root", default="/kaggle/working/dcgan_output")
    p.add_argument("--g-path", default="/kaggle/input/exp11-weights/generator_ema_final.pth")
    p.add_argument("--d-path", default="/kaggle/input/exp11-weights/discriminator_final.pth")
    p.add_argument("--resume", default="", help="Full checkpoint made by this script.")
    p.add_argument("--experiment-name", default="")
    p.add_argument("--clip-model", choices=["ViT-B-32", "RN50"], default="ViT-B-32")
    p.add_argument("--lambda-clip", type=float, default=0.025)
    p.add_argument("--epochs", type=int, default=50, help="Total target epoch, not extra epochs.")
    p.add_argument("--clip-warmup-epochs", type=int, default=5)
    p.add_argument("--clip-interval", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--dataset-limit", type=int, default=20000)
    p.add_argument("--lr-g", type=float, default=2e-5)
    p.add_argument("--lr-d", type=float, default=2e-5)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.99)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--real-feature-cache", default="")
    p.add_argument("--n-fid", type=int, default=10000)
    p.add_argument("--n-clip-eval", type=int, default=2000)
    p.add_argument("--n-image-eval", type=int, default=500)
    p.add_argument(
        "--eval-only", action="store_true",
        help="Load direct weights or --resume checkpoint and evaluate without optimizer steps.",
    )
    p.add_argument("--skip-final-eval", action="store_true")
    p.add_argument(
        "--zip-output", action="store_true",
        help="Create /kaggle/working/<experiment>.zip after a successful run.",
    )
    if argv is None and (
        "ipykernel_launcher" in Path(sys.argv[0]).name
        or "colab_kernel_launcher" in Path(sys.argv[0]).name
        or any(arg == "-f" for arg in sys.argv[1:])
    ):
        argv = []
    return p.parse_args(argv)


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_images(root):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(
        str(p) for p in Path(root).rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def resolve_dataset(requested, minimum_images=20000):
    paths = (
        find_images(requested)
        if requested and requested.lower() != "auto" and os.path.isdir(requested)
        else []
    )
    if paths:
        return requested, paths
    kaggle_input = Path("/kaggle/input")
    candidates = []
    if kaggle_input.exists():
        roots = []
        for child in sorted(kaggle_input.iterdir()):
            if not child.is_dir():
                continue
            if child.name == "datasets":
                for owner in sorted(child.iterdir()):
                    if owner.is_dir():
                        roots.extend(
                            dataset for dataset in sorted(owner.iterdir())
                            if dataset.is_dir()
                        )
            else:
                roots.append(child)
        for child in roots:
            if any(child.rglob("generator_ema_final.pth")):
                continue
            child_paths = find_images(child)
            if len(child_paths) >= minimum_images:
                candidates.append((child, child_paths))
    if len(candidates) == 1:
        child, child_paths = candidates[0]
        print(f"[dataset] auto-detected {child} ({len(child_paths)} images)")
        return str(child), child_paths
    candidate_text = ", ".join(
        f"{path} ({len(child_paths)} images)" for path, child_paths in candidates
    ) or "none"
    raise FileNotFoundError(
        f"Could not safely auto-detect one training dataset with at least "
        f"{minimum_images} images. Candidates: {candidate_text}. "
        "Attach exactly one original image Dataset or pass --dataset-path."
    )


def resolve_weight(requested, filename):
    if requested and requested.lower() != "auto" and os.path.isfile(requested):
        return requested
    matches = (
        sorted(Path("/kaggle/input").rglob(filename))
        if Path("/kaggle/input").exists()
        else []
    )
    if len(matches) == 1:
        print(f"[weights] auto-detected {matches[0]}")
        return str(matches[0])
    match_text = ", ".join(str(path) for path in matches) or "none"
    raise FileNotFoundError(
        f"Expected exactly one {filename!r} under /kaggle/input; found "
        f"{len(matches)}: {match_text}"
    )


class AnimeDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.transform(Image.open(self.paths[index]).convert("RGB"))
            except (OSError, IOError):
                index = random.randrange(len(self.paths))
        raise RuntimeError("Could not load an image after 10 attempts.")


class EdgeSharpen:
    def __init__(self, prob=0.2, alpha=0.3):
        self.prob, self.alpha = prob, alpha

    def __call__(self, image):
        if random.random() >= self.prob:
            return image
        arr = np.asarray(image, dtype=np.float32) / 255.0
        blur = np.asarray(
            image.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32
        ) / 255.0
        return Image.fromarray(np.clip((arr + self.alpha * (arr - blur)) * 255, 0, 255).astype(np.uint8))


def train_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        EdgeSharpen(prob=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])


class Generator(nn.Module):
    """Bit-for-bit Exp09/Exp11 Generator."""
    def __init__(self, nd=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nd, 768, 4), nn.BatchNorm2d(768), nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1), nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """Bit-for-bit Exp09/Exp11 Discriminator."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 32, 3, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(32, 64, 3, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 3, 2, 1)), nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 3, 2, 1)), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(4 * 4 * 256, 256)), nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class EMA:
    """EMA over trainable parameters and every BN buffer."""
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        for name, tensor in model.state_dict().items():
            self.shadow[name] = tensor.detach().clone()

    @torch.no_grad()
    def update(self, model):
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor):
                self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(tensor.detach())

    @torch.no_grad()
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.shadow = state["shadow"]

    def export(self):
        return {name: value.detach().cpu().clone() for name, value in self.shadow.items()}


def _rand_brightness(x):
    return x + torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5


def _rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2) + mean


def _rand_contrast(x):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5) + mean


def _rand_translation(x):
    sx = int(x.size(2) * DIFFAUG_TRANSLATION_RATIO + 0.5)
    sy = int(x.size(3) * DIFFAUG_TRANSLATION_RATIO + 0.5)
    tx = torch.randint(-sx, sx + 1, (x.size(0), 1, 1), device=x.device)
    ty = torch.randint(-sy, sy + 1, (x.size(0), 1, 1), device=x.device)
    bx = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    gx, gy = torch.meshgrid(
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    gx, gy = gx.unsqueeze(0) + tx + sx, gy.unsqueeze(0) + ty + sy
    padded = F.pad(x, (sy, sy, sx, sx), mode="replicate")
    return padded.permute(0, 2, 3, 1)[bx, gx, gy].permute(0, 3, 1, 2)


def _rand_cutout(x):
    h = int(x.size(2) * DIFFAUG_CUTOUT_RATIO + 0.5)
    w = int(x.size(3) * DIFFAUG_CUTOUT_RATIO + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - h % 2), (x.size(0), 1, 1), device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - w % 2), (x.size(0), 1, 1), device=x.device)
    gx, gy = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    gx = (gx.unsqueeze(0) + ox - h // 2).clamp(0, x.size(2) - 1)
    gy = (gy.unsqueeze(0) + oy - w // 2).clamp(0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), device=x.device, dtype=x.dtype)
    batch = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    mask[batch, gx, gy] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x):
    for item in DIFFAUG_POLICY.split(","):
        if item == "color":
            x = _rand_contrast(_rand_saturation(_rand_brightness(x)))
        elif item == "translation":
            x = _rand_translation(x)
        elif item == "cutout":
            x = _rand_cutout(x)
    return x.contiguous()


class FrozenCLIP(nn.Module):
    def __init__(self, model_name, device):
        super().__init__()
        self.model_name = model_name
        self.device = device
        load_name = "ViT-B-32-quickgelu" if model_name == "ViT-B-32" else model_name
        self.model, _, _ = open_clip.create_model_and_transforms(
            load_name, pretrained="openai"
        )
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
        size = self.model.visual.image_size
        self.image_size = int(size[0] if isinstance(size, (tuple, list)) else size)
        self.register_buffer(
            "mean", torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        )

    def forward(self, images):
        # Input is GAN range [-1,1]. Frozen parameters do not block dL/dimage.
        x = (images + 1.0) * 0.5
        x = F.interpolate(x, (self.image_size, self.image_size), mode="bicubic", align_corners=False)
        x = (x.clamp(0, 1) - self.mean) / self.std
        features = self.model.encode_image(x)
        return F.normalize(features.float(), dim=-1)


def pairwise_sq_distance(x, y):
    return (x.square().sum(1, keepdim=True) + y.square().sum(1).unsqueeze(0) - 2 * x @ y.T).clamp_min(0)


def clip_mmd(x, y, sigmas=MMD_SIGMAS, unbiased=True):
    """Multi-scale RBF MMD². The real and fake batches need not be paired."""
    xx, yy, xy = pairwise_sq_distance(x, x), pairwise_sq_distance(y, y), pairwise_sq_distance(x, y)
    total = x.new_zeros(())
    for sigma in sigmas:
        scale = 2 * sigma * sigma
        kxx, kyy, kxy = torch.exp(-xx / scale), torch.exp(-yy / scale), torch.exp(-xy / scale)
        if unbiased and x.size(0) > 1 and y.size(0) > 1:
            term_x = (kxx.sum() - kxx.diag().sum()) / (x.size(0) * (x.size(0) - 1))
            term_y = (kyy.sum() - kyy.diag().sum()) / (y.size(0) * (y.size(0) - 1))
        else:
            term_x, term_y = kxx.mean(), kyy.mean()
        total = total + term_x + term_y - 2 * kxy.mean()
    return total / len(sigmas)


def gradient_l2(loss, parameters):
    grads = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
    )
    squares = [grad.detach().float().square().sum() for grad in grads if grad is not None]
    return torch.sqrt(torch.stack(squares).sum()) if squares else loss.new_zeros(())


def path_digest(paths):
    h = hashlib.sha256()
    for path in paths:
        h.update(os.path.normpath(path).encode("utf-8", errors="replace"))
    return h.hexdigest()


@torch.no_grad()
def build_or_load_real_cache(clip_encoder, loader, paths, cache_path, device):
    expected = {"clip_model": clip_encoder.model_name, "paths_sha256": path_digest(paths), "count": len(paths)}
    if os.path.isfile(cache_path):
        state = torch.load(cache_path, map_location="cpu")
        if all(state.get(k) == v for k, v in expected.items()):
            print(f"[cache] loaded {cache_path}: {tuple(state['features'].shape)}")
            return state["features"].float()
        print("[cache] metadata mismatch; rebuilding.")
    features = []
    for step, images in enumerate(loader, 1):
        features.append(clip_encoder(images.to(device)).half().cpu())
        if step % 100 == 0:
            print(f"[cache] {min(step * loader.batch_size, len(paths))}/{len(paths)}")
    features = torch.cat(features)
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({**expected, "features": features}, cache_path)
    print(f"[cache] saved {cache_path}")
    return features.float()


def strip_wrapped_state(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError("Weight file does not contain a state_dict.")
    return {k.removeprefix("module."): v for k, v in state.items()}


def load_trusted_full_checkpoint(path, map_location):
    """Full checkpoints contain RNG metadata, so PyTorch 2.6 needs weights_only=False."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=map_location)


def save_grid(images, path, nrow=8):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    arr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).byte().cpu().numpy()
    Image.fromarray(arr).save(path)


def append_csv(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path, epoch, global_step, G, D, ema, opt_g, opt_d, args,
    train_loader_generator, clip_index_generator,
):
    torch.save({
        "format": "clip_cmmd_v1",
        "epoch": epoch,
        "global_step": global_step,
        "G": G.state_dict(),
        "D": D.state_dict(),
        "ema": ema.state_dict(),
        "optimizer_G": opt_g.state_dict(),
        "optimizer_D": opt_d.state_dict(),
        "args": vars(args),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "train_loader_generator_state": train_loader_generator.get_state(),
        "clip_index_generator_state": clip_index_generator.get_state(),
    }, path)


def train(
    args, device, exp_dir, train_loader, real_features, clip_encoder,
    train_loader_generator,
):
    training_started = time.perf_counter()
    G, D = Generator().to(device), Discriminator().to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))
    start_epoch, global_step = 1, 0
    # This independent CPU stream prevents CLIP-only sampling from changing the
    # GAN noise, DiffAugment or DataLoader random sequence in the lambda=0 arm.
    clip_index_generator = torch.Generator(device="cpu")
    clip_index_generator.manual_seed(args.seed + 100_003)

    if args.resume:
        state = load_trusted_full_checkpoint(args.resume, map_location=device)
        if state.get("format") != "clip_cmmd_v1":
            raise ValueError("--resume must be a full checkpoint produced by this script.")
        saved_args = state.get("args", {})
        for key in ("clip_model", "lambda_clip", "clip_interval", "clip_warmup_epochs"):
            if key in saved_args and saved_args[key] != getattr(args, key):
                raise ValueError(
                    f"Resume mismatch for {key}: checkpoint={saved_args[key]!r}, "
                    f"command={getattr(args, key)!r}. Keep the controlled experiment unchanged."
                )
        G.load_state_dict(state["G"], strict=True)
        D.load_state_dict(state["D"], strict=True)
        opt_g.load_state_dict(state["optimizer_G"])
        opt_d.load_state_dict(state["optimizer_D"])
        ema = EMA(G, args.ema_decay)
        ema.load_state_dict(state["ema"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state.get("global_step", 0))
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
                torch.cuda.set_rng_state_all(
                    [rng_state.cpu() for rng_state in state["cuda_rng_state_all"]]
                )
            train_loader_generator.set_state(state["train_loader_generator_state"].cpu())
            clip_index_generator.set_state(state["clip_index_generator_state"].cpu())
        else:
            print("[resume] warning: legacy checkpoint has no RNG states.")
        print(f"[resume] epoch {start_epoch}, global step {global_step}")
    else:
        if not os.path.isfile(args.g_path) or not os.path.isfile(args.d_path):
            raise FileNotFoundError(
                "Exp11 weights missing. Attach the Kaggle Dataset and pass --g-path/--d-path."
            )
        G.load_state_dict(strip_wrapped_state(torch.load(args.g_path, map_location=device)), strict=True)
        D.load_state_dict(strip_wrapped_state(torch.load(args.d_path, map_location=device)), strict=True)
        ema = EMA(G, args.ema_decay)
        print(f"[weights] G={args.g_path}\n[weights] D={args.d_path}")

    preview_generator = torch.Generator(device="cpu")
    preview_generator.manual_seed(args.seed + 300_007)
    fixed_noise = torch.randn(
        64, NOISE_DIM, 1, 1, generator=preview_generator
    ).to(device)
    log_path = str(Path(exp_dir) / "training_log.csv")
    epoch_sequence = [] if args.eval_only else range(start_epoch, args.epochs + 1)
    for epoch in epoch_sequence:
        epoch_started = time.perf_counter()
        G.train()
        D.train()
        totals = {k: 0.0 for k in ("d_loss", "g_loss", "g_adv", "clip_mmd", "d_real_aug", "d_fake_aug")}
        clip_steps = 0
        grad_ratio = float("nan")
        adv_grad_norm = float("nan")
        clip_grad_norm = float("nan")
        warm = 1.0 if args.clip_warmup_epochs <= 0 else min(1.0, epoch / args.clip_warmup_epochs)
        effective_lambda = args.lambda_clip * warm

        for real in train_loader:
            real = real.to(device, non_blocking=True)
            bs = real.size(0)

            opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                fake_d = G(torch.randn(bs, NOISE_DIM, 1, 1, device=device))
            d_real = D(diff_augment(real))
            d_fake = D(diff_augment(fake_d))
            d_loss = F.relu(1 - d_real).mean() + F.relu(1 + d_fake).mean()
            d_loss.backward()
            opt_d.step()

            D.requires_grad_(False)
            opt_g.zero_grad(set_to_none=True)
            fake = G(torch.randn(bs, NOISE_DIM, 1, 1, device=device))
            g_adv = -D(diff_augment(fake)).mean()
            use_clip = effective_lambda > 0 and global_step % args.clip_interval == 0
            if use_clip:
                indices = torch.randint(
                    0, real_features.size(0), (bs,), generator=clip_index_generator
                )
                real_clip = real_features[indices].to(device, non_blocking=True)
                fake_clip = clip_encoder(fake)
                mmd = clip_mmd(fake_clip, real_clip)
                g_loss = g_adv + effective_lambda * mmd
                # One measurement per epoch calibrates lambda by actual gradient
                # influence instead of comparing incomparable scalar loss values.
                if clip_steps == 0:
                    params = [p for p in G.parameters() if p.requires_grad]
                    adv_norm_t = gradient_l2(g_adv, params)
                    clip_norm_t = gradient_l2(mmd, params)
                    adv_grad_norm = float(adv_norm_t)
                    clip_grad_norm = float(clip_norm_t)
                    grad_ratio = effective_lambda * clip_grad_norm / max(adv_grad_norm, 1e-12)
                totals["clip_mmd"] += float(mmd.detach())
                clip_steps += 1
            else:
                g_loss = g_adv
            g_loss.backward()
            opt_g.step()
            D.requires_grad_(True)
            ema.update(G)
            global_step += 1

            totals["d_loss"] += float(d_loss.detach())
            totals["g_loss"] += float(g_loss.detach())
            totals["g_adv"] += float(g_adv.detach())
            totals["d_real_aug"] += float(d_real.detach().mean())
            totals["d_fake_aug"] += float(d_fake.detach().mean())

        batches = len(train_loader)
        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lambda_target": args.lambda_clip,
            "lambda_effective": effective_lambda,
            "clip_interval": args.clip_interval,
            **{k: totals[k] / batches for k in ("d_loss", "g_loss", "g_adv", "d_real_aug", "d_fake_aug")},
            "clip_mmd": totals["clip_mmd"] / max(1, clip_steps),
            "clip_steps": clip_steps,
            "adv_grad_norm": adv_grad_norm,
            "clip_grad_norm_unweighted": clip_grad_norm,
            "weighted_clip_to_adv_grad_ratio": grad_ratio,
            "epoch_seconds": epoch_seconds,
            "training_elapsed_hours": (time.perf_counter() - training_started) / 3600,
        }
        append_csv(log_path, row)
        print(
            f"E{epoch:03d}/{args.epochs} D={row['d_loss']:.4f} G={row['g_loss']:.4f} "
            f"adv={row['g_adv']:.4f} MMD={row['clip_mmd']:.5f} "
            f"lambda={effective_lambda:.5f} grad_ratio={grad_ratio:.3f} "
            f"time={epoch_seconds:.1f}s clip_steps={clip_steps}"
        )

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            preview = Generator().to(device)
            ema.copy_to(preview)
            preview.eval()
            with torch.no_grad():
                save_grid(preview(fixed_noise), str(Path(exp_dir) / f"samples_epoch_{epoch:03d}.png"))
            del preview
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs:
            save_checkpoint(
                str(Path(exp_dir) / f"checkpoint_epoch_{epoch:03d}.pth"),
                epoch, global_step, G, D, ema, opt_g, opt_d, args,
                train_loader_generator, clip_index_generator,
            )

    torch.save(G.state_dict(), str(Path(exp_dir) / "generator_raw_final.pth"))
    torch.save(D.state_dict(), str(Path(exp_dir) / "discriminator_final.pth"))
    torch.save(ema.export(), str(Path(exp_dir) / "generator_ema_final.pth"))
    torch.save(ema.state_dict(), str(Path(exp_dir) / "ema_state_final.pth"))
    eval_g = Generator().to(device)
    ema.copy_to(eval_g)
    eval_g.eval()
    with torch.no_grad():
        final_samples = eval_g(fixed_noise)
        save_grid(final_samples, str(Path(exp_dir) / "samples_final.png"))
        sample_snapshot = (
            "samples_eval_only.png"
            if args.eval_only
            else f"samples_epoch_{args.epochs:03d}_final.png"
        )
        save_grid(final_samples, str(Path(exp_dir) / sample_snapshot))
    return eval_g


class FIDCalculator:
    def __init__(self, device):
        net = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        net.fc = nn.Identity()
        self.net = net.eval().to(device).requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def features(self, images):
        x = F.interpolate((images + 1) * 0.5, (299, 299), mode="bilinear", align_corners=False)
        return self.net((x - 0.5) / 0.5).cpu().numpy()

    @torch.no_grad()
    def compute(self, G, real_loader, n, noise_generator):
        real, fake, count = [], [], 0
        for images in real_loader:
            real.append(self.features(images.to(self.device)))
            count += images.size(0)
            if count >= n:
                break
        count = 0
        while count < n:
            bs = min(64, n - count)
            noise = torch.randn(
                bs, NOISE_DIM, 1, 1, generator=noise_generator
            ).to(self.device)
            fake.append(self.features(G(noise)))
            count += bs
        r, f = np.concatenate(real)[:n], np.concatenate(fake)[:n]
        mr, mf, sr, sf = r.mean(0), f.mean(0), np.cov(r, rowvar=False), np.cov(f, rowvar=False)
        covmean = linalg.sqrtm(sr @ sf)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float((mr - mf) @ (mr - mf) + np.trace(sr + sf - 2 * covmean))


def laplacian_values(images):
    values = []
    for image in images:
        rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.asarray(values)


def edge_density(images):
    values = []
    for image in images:
        rgb = (image.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        values.append((cv2.Canny(gray, 50, 150) > 0).mean())
    return float(np.mean(values))


def effective_rank(features):
    x = features.double().cpu().numpy()
    covariance = np.cov(x, rowvar=False)
    eig = np.linalg.eigvalsh(covariance)
    eig = np.maximum(eig, 0)
    p = eig / max(eig.sum(), 1e-12)
    return float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))


def nearest_real_cosine(fake_features, real_features, device):
    real = real_features.to(device)
    values = []
    for chunk in fake_features.split(256):
        values.append((chunk.to(device) @ real.T).max(dim=1).values.cpu())
    return float(torch.cat(values).mean())


@torch.no_grad()
def collect_fake(G, clip_encoder, n, device, noise_generator):
    images, features, count = [], [], 0
    while count < n:
        bs = min(64, n - count)
        noise = torch.randn(
            bs, NOISE_DIM, 1, 1, generator=noise_generator
        ).to(device)
        generated = G(noise)
        images.append(((generated + 1) * 0.5).cpu())
        features.append(clip_encoder(generated).cpu())
        count += bs
    return torch.cat(images), torch.cat(features)


@torch.no_grad()
def lpips_diversity(images, device, pair_generator, pairs=1000):
    metric = lpips.LPIPS(net="alex").to(device).eval()
    i = torch.randint(0, len(images), (pairs,), generator=pair_generator)
    j = torch.randint(0, len(images), (pairs,), generator=pair_generator)
    j[j == i] = (j[j == i] + 1) % len(images)
    scores = []
    for start in range(0, pairs, 64):
        a = images[i[start:start + 64]].to(device) * 2 - 1
        b = images[j[start:start + 64]].to(device) * 2 - 1
        scores.append(metric(a, b).flatten().cpu())
    return float(torch.cat(scores).mean())


def evaluate(args, G, clip_encoder, real_features, eval_loader, device, exp_dir):
    metrics = {}
    fid_noise_generator = torch.Generator(device="cpu")
    fid_noise_generator.manual_seed(args.seed + 400_009)
    clip_noise_generator = torch.Generator(device="cpu")
    clip_noise_generator.manual_seed(args.seed + 500_009)
    real_eval_generator = torch.Generator(device="cpu")
    real_eval_generator.manual_seed(args.seed + 600_011)
    pair_generator = torch.Generator(device="cpu")
    pair_generator.manual_seed(args.seed + 700_001)

    metrics["fid_legacy_inception_v3"] = FIDCalculator(device).compute(
        G, eval_loader, args.n_fid, fid_noise_generator
    )

    fake_images, fake_features = collect_fake(
        G, clip_encoder, args.n_clip_eval, device, clip_noise_generator
    )
    real_count = min(args.n_clip_eval, real_features.size(0))
    real_indices = torch.randperm(
        real_features.size(0), generator=real_eval_generator
    )[:real_count]
    real_subset = real_features[real_indices]
    fake_eval, real_eval = fake_features.to(device), real_subset.to(device)
    # Report both: the unbiased estimator follows the CMMD statistical idea but
    # can be slightly negative at finite sample size; the biased estimate is
    # non-negative and convenient for dashboard comparisons.
    metrics["clip_mmd2_unbiased"] = float(clip_mmd(fake_eval, real_eval, unbiased=True).cpu())
    metrics["clip_mmd2_biased"] = float(clip_mmd(fake_eval, real_eval, unbiased=False).cpu())
    metrics["nearest_real_clip_cosine"] = nearest_real_cosine(fake_features, real_subset, device)
    metrics["fake_clip_feature_variance"] = float(fake_features.var(dim=0, unbiased=False).sum())
    metrics["real_clip_feature_variance"] = float(real_subset.var(dim=0, unbiased=False).sum())
    metrics["fake_clip_effective_rank"] = effective_rank(fake_features)
    metrics["real_clip_effective_rank"] = effective_rank(real_subset)

    n_img = min(args.n_image_eval, len(fake_images))
    fake_metric_images = fake_images[:n_img]
    real_metric_images = []
    for images in eval_loader:
        real_metric_images.append((images + 1) * 0.5)
        if sum(x.size(0) for x in real_metric_images) >= n_img:
            break
    real_metric_images = torch.cat(real_metric_images)[:n_img]
    fake_lap, real_lap = laplacian_values(fake_metric_images), laplacian_values(real_metric_images)
    threshold = float(np.percentile(real_lap, 10))
    metrics["blur_threshold_real_p10"] = threshold
    metrics["fake_blur_rate"] = float((fake_lap < threshold).mean())
    metrics["real_blur_rate_by_definition"] = float((real_lap < threshold).mean())
    metrics["fake_laplacian_mean"] = float(fake_lap.mean())
    metrics["real_laplacian_mean"] = float(real_lap.mean())
    metrics["fake_edge_density"] = edge_density(fake_metric_images)
    metrics["real_edge_density"] = edge_density(real_metric_images)
    metrics["edge_density_ratio"] = metrics["fake_edge_density"] / max(metrics["real_edge_density"], 1e-12)
    metrics["lpips_alex_diversity"] = lpips_diversity(
        fake_metric_images, device, pair_generator
    )

    with open(Path(exp_dir) / "final_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    metric_snapshot = (
        "metrics_eval_only.json"
        if args.eval_only
        else f"metrics_epoch_{args.epochs:03d}.json"
    )
    with open(Path(exp_dir) / metric_snapshot, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    run_started = time.perf_counter()
    args = parse_args(argv)
    if args.lambda_clip < 0 or args.clip_interval < 1:
        raise ValueError("lambda-clip must be >=0 and clip-interval must be >=1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seeds(args.seed)
    args.g_path = resolve_weight(args.g_path, "generator_ema_final.pth")
    args.d_path = resolve_weight(args.d_path, "discriminator_final.pth")
    resolved_dataset_path, all_paths = resolve_dataset(
        args.dataset_path, minimum_images=args.dataset_limit
    )
    if len(all_paths) < args.dataset_limit:
        raise ValueError(
            f"Dataset contains only {len(all_paths)} readable image files, but "
            f"--dataset-limit={args.dataset_limit}. Use the same original dataset "
            "as Exp11 or explicitly change the limit for every controlled arm."
        )
    rng = random.Random(args.seed)
    selected = rng.sample(all_paths, min(args.dataset_limit, len(all_paths)))
    selected.sort()

    model_tag = args.clip_model.replace("-", "_")
    lambda_tag = f"{args.lambda_clip:.4f}".replace(".", "p")
    exp_name = args.experiment_name or f"CLIP_MMD_{model_tag}_L{lambda_tag}"
    exp_dir = str(Path(args.output_root) / exp_name)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    # Shared across lambda runs. For later Kaggle sessions, upload this file as
    # a small Dataset and pass its read-only path via --real-feature-cache.
    cache_path = args.real_feature_cache or str(
        Path(args.output_root) / "clip_feature_cache" / f"real_features_{model_tag}.pth"
    )

    config = {
        **vars(args),
        "resolved_experiment_name": exp_name,
        "resolved_dataset_path": resolved_dataset_path,
        "dataset_image_count_before_sampling": len(all_paths),
        "device": str(device),
        "selected_images": len(selected),
        "dataset_paths_sha256": path_digest(selected),
        "started_at": datetime.now().isoformat(),
        "objective": "hinge adversarial loss + frozen CLIP image-feature MMD",
        "mmd_sigmas": MMD_SIGMAS,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torchvision": getattr(sys.modules.get("torchvision"), "__version__", "unknown"),
            "open_clip": getattr(open_clip, "__version__", "unknown"),
            "lpips": getattr(lpips, "__version__", "unknown"),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "notes": [
            "No random one-to-one real/fake pairing.",
            "CLIP parameters are frozen; fake image gradient remains enabled.",
            "No R1. Exp11 DiffAugment is preserved.",
            "lambda applies on every clip_interval-th generator step after warmup.",
            "eval_only performs no optimizer step.",
        ],
    }
    with open(Path(exp_dir) / "config.json", "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    config_snapshot = (
        "config_eval_only.json"
        if args.eval_only
        else f"config_target_epoch_{args.epochs:03d}.json"
    )
    with open(Path(exp_dir) / config_snapshot, "w", encoding="utf-8") as handle:
        json.dump(config, handle, ensure_ascii=False, indent=2)
    with open(Path(exp_dir) / "selected_images.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(selected))

    train_ds = AnimeDataset(selected, train_transform())
    eval_ds = AnimeDataset(selected, eval_transform())
    train_loader_generator = torch.Generator(device="cpu")
    train_loader_generator.manual_seed(args.seed + 20_011)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
        pin_memory=device.type == "cuda", drop_last=True, generator=train_loader_generator,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=64, shuffle=False, num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    clip_encoder = FrozenCLIP(args.clip_model, device)
    real_features = build_or_load_real_cache(
        clip_encoder, eval_loader, selected, cache_path, device
    )
    G = train(
        args, device, exp_dir, train_loader, real_features, clip_encoder,
        train_loader_generator,
    )
    if not args.skip_final_eval:
        evaluate(args, G, clip_encoder, real_features, eval_loader, device, exp_dir)
    run_summary = {
        "experiment_name": exp_name,
        "finished_at": datetime.now().isoformat(),
        "total_wall_hours": (time.perf_counter() - run_started) / 3600,
        "target_epochs": args.epochs,
        "resume": args.resume or None,
    }
    with open(Path(exp_dir) / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    summary_snapshot = (
        "run_summary_eval_only.json"
        if args.eval_only
        else f"run_summary_epoch_{args.epochs:03d}.json"
    )
    with open(Path(exp_dir) / summary_snapshot, "w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)
    print(f"[done] {exp_dir}")
    if args.zip_output:
        essential_names = [
            "config_eval_only.json",
            "selected_images.txt",
            "samples_eval_only.png",
            "metrics_eval_only.json",
            "run_summary_eval_only.json",
        ]
        archive_path = str(
            Path("/kaggle/working") / f"{exp_name}_ESSENTIAL.zip"
        )
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
        ) as archive:
            for name in essential_names:
                source = Path(exp_dir) / name
                if not source.is_file():
                    raise FileNotFoundError(
                        f"Essential result missing before packaging: {source}"
                    )
                archive.write(source, arcname=f"{exp_name}/{name}")
        print(f"[download] {archive_path}")


if __name__ == "__main__":
    print("[standalone] E0: evaluate the untouched Exp11 baseline; no training.")
    main([
        "--eval-only",
        "--lambda-clip", "0",
        "--clip-model", "ViT-B-32",
        "--dataset-path", "auto",
        "--g-path", "auto",
        "--d-path", "auto",
        "--experiment-name", "E0_exp11_formal_eval",
        "--zip-output",
    ])
