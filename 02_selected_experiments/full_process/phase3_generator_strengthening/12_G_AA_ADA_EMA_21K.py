"""
Exp12 — final-quality DCGAN candidate: mild anti-aliasing + adaptive DiffAugment + EMA
===========================================================================

目的
----
在 Exp11 (Width3x + SN/Hinge + DiffAugment + EMA) 的 DCGAN 框架内，针对当前
仍然存在的棋盘格、局部色块和训练震荡问题，建立一个可复现实验候选：

1. Generator 仍使用 Exp11 的五层 ConvTranspose 权重结构；在 8/16/32 像素
   的中间特征图上加入轻量固定 3x3 低通混合，减少上采样 aliasing。该模块无
   可学习参数，因此可以从 Exp11 的 Generator EMA 权重继续 fine-tune。
2. Discriminator 仍为 Exp11 的 SN + Hinge，不加入已验证会退化的 R1。
3. 将 Exp11 的固定 DiffAugment 改为保守的自适应概率 p：当 D 对 real 的正号
   比例高于目标值时逐步增加增强，避免在 21K 数据上过度增强。
4. 保留 EMA，并在固定噪声集上周期性计算 quick FID，保存 best EMA 权重；FID
   在 warm-up 后连续上升才提前停止，最后用同一批固定噪声重新计算 full FID，
   选择真正用于部署的权重，而不是假设“最后一个 epoch 就是最好模型”。

这不是消融实验：脚本只训练这一条质量优先路线；max epoch 仅为安全上限，实际
训练长度由 FID 早停决定。

重要边界
--------
- 这是“DCGAN 框架内的高质量候选”，不是 StyleGAN2-ADA 的复刻，也不保证一次
  运行就低于 Exp11。必须与 Exp11 使用相同 FID 协议比较。
- 64x64 仍然限制商业级五官细节；FID 下降不等于每一张脸都无畸变。
- 需要检查数据集/SDXL/Anime Faces 的使用许可，不能仅凭指标宣称商业可用。
- 默认从头训练，不会因为 Kaggle 中存在权重数据集而误加载；只有显式传入
  `--init-g/--init-d`，或添加 `--use-exp11-init`，才会进行 Exp11 初始化。

建议名称：12_G_AA_ADA_EMA_21K
输出：/kaggle/working/dcgan_output/12_G_AA_ADA_EMA_21K
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
import matplotlib.pyplot as plt
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


# This is the single final-quality candidate, not an ablation matrix. The
# epoch count is a safety ceiling; deployment is selected by full FID after
# patience-based early stopping.
EXPERIMENT_NAME = "12_G_AA_ADA_EMA_21K"
DEFAULT_DATASET_LIMIT = 21000
IMAGE_SIZE = 64
BATCH_SIZE = 32
NOISE_DIM = 128
LR = 1e-4
BETAS = (0.5, 0.99)
SEED = 42
# This is a safety ceiling, not the target training length. Training stops
# earlier when fixed-noise FID rises consistently after the warm-up period.
MAX_EPOCHS = 300
MIN_EPOCHS = 50
SAMPLE_INTERVAL = 10
FID_INTERVAL = 10
FID_PATIENCE = 4
FID_MIN_DELTA = 0.20
FID_RISE_MARGIN = 0.50
QUICK_N_FID = 2000
FINAL_N_FID = 10000

# Conservative anti-aliasing. 0.0 disables the candidate without changing the
# rest of the script; 0.10 is intentionally mild for 64x64 anime line art.
AA_BLEND = 0.10

# Adaptive augmentation. The policy is only applied to D inputs and remains
# differentiable on the G path. Starting at p=0 prevents unnecessary noise
# when the 21K discriminator is not overfitting.
ADA_TARGET = 0.60
ADA_STEP = 0.01
ADA_INTERVAL_STEPS = 256
ADA_MAX_P = 0.40
DIFFAUG_TRANSLATION_RATIO = 0.08
DIFFAUG_CUTOUT_RATIO = 0.20
DIFFAUG_POLICY = "color,translation,cutout"

EMA_DECAY = 0.9999


def default_output_dir():
    kaggle = Path("/kaggle/working")
    if kaggle.exists():
        return str(kaggle / "dcgan_output")
    return str(Path.cwd() / "outputs")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--dataset-path", default=os.getenv("DATASET_PATH", ""))
    parser.add_argument("--dataset-limit", type=int, default=DEFAULT_DATASET_LIMIT)
    parser.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT", default_output_dir()))
    parser.add_argument("--init-g", default=os.getenv("INIT_G_PATH", ""))
    parser.add_argument("--init-d", default=os.getenv("INIT_D_PATH", ""))
    parser.add_argument(
        "--use-exp11-init",
        action="store_true",
        help="Only if desired: auto-detect Exp11 EMA/Discriminator weights for fine-tuning. "
             "Default is from-scratch training.",
    )
    parser.add_argument("--resume", default=os.getenv("RESUME_PATH", ""))
    parser.add_argument("--max-epochs", "--epochs", dest="max_epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--min-epochs", type=int, default=MIN_EPOCHS)
    parser.add_argument("--fid-interval", type=int, default=FID_INTERVAL)
    parser.add_argument("--fid-patience", type=int, default=FID_PATIENCE)
    parser.add_argument("--fid-min-delta", type=float, default=FID_MIN_DELTA)
    parser.add_argument("--fid-rise-margin", type=float, default=FID_RISE_MARGIN)
    parser.add_argument("--quick-fid", type=int, default=QUICK_N_FID)
    parser.add_argument("--final-fid", type=int, default=FINAL_N_FID)
    # Kaggle/IPython injects -f kernel.json; parse_known_args intentionally
    # ignores unrelated notebook arguments.
    args, _ = parser.parse_known_args(argv)
    if args.max_epochs < 1:
        raise ValueError("--max-epochs must be positive")
    if args.min_epochs < 1 or args.min_epochs > args.max_epochs:
        raise ValueError("--min-epochs must be in [1, --max-epochs]")
    if args.fid_interval < 1 or args.fid_patience < 1:
        raise ValueError("--fid-interval and --fid-patience must be positive")
    if args.quick_fid < 2 or args.final_fid < 2:
        raise ValueError("FID sample counts must be at least 2")
    return args


def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_images(root_dir):
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


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
    # LANCZOS is used only when the attached images are not already 64x64.
    # Do not add an aggressive sharpen operation: it can make D learn halos.
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


class FeatureBlur(nn.Module):
    """Parameter-free mild low-pass filtering for intermediate feature maps."""

    def __init__(self, channels, blend=AA_BLEND):
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0]) / 4.0
        kernel = torch.outer(kernel_1d, kernel_1d).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel.repeat(channels, 1, 1, 1))
        self.blend = float(blend)

    def forward(self, x):
        if self.blend <= 0:
            return x
        y = F.conv2d(x, self.kernel.to(dtype=x.dtype), padding=1, groups=x.shape[1])
        return x.lerp(y, self.blend)


class Generator(nn.Module):
    """Exp11-compatible state keys plus non-parametric intermediate AA filters."""

    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        # Keep this Sequential unchanged so Exp11 state_dict keys still match.
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 768, 4),
            nn.BatchNorm2d(768), nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1),
            nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1),
            nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1),
            nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh(),
        )
        self.blur_8 = FeatureBlur(384)
        self.blur_16 = FeatureBlur(192)
        self.blur_32 = FeatureBlur(96)
        self.apply(self._init)

    @staticmethod
    def _init(module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.net[0](x)
        x = self.net[1](x)
        x = self.net[2](x)
        x = self.net[3](x)
        x = self.net[4](x)
        x = self.net[5](x)
        x = self.blur_8(x)
        x = self.net[6](x)
        x = self.net[7](x)
        x = self.net[8](x)
        x = self.blur_16(x)
        x = self.net[9](x)
        x = self.net[10](x)
        x = self.net[11](x)
        x = self.blur_32(x)
        x = self.net[12](x)
        return self.net[13](x)


class Discriminator(nn.Module):
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
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay
        self.shadow = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for name, tensor in model.state_dict().items():
            if torch.is_floating_point(tensor):
                self.shadow[name].mul_(self.decay).add_(tensor.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[name].copy_(tensor.detach())

    @torch.no_grad()
    def apply_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

    def model_state_dict(self):
        return {name: tensor.detach().cpu().clone() for name, tensor in self.shadow.items()}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.model_state_dict()}

    def load_state_dict(self, state, model=None):
        self.decay = float(state["decay"])
        self.shadow = {name: tensor.clone() for name, tensor in state["shadow"].items()}
        # Allow resuming from Exp11 checkpoints, whose EMA has no AA buffers.
        if model is not None:
            for name, tensor in model.state_dict().items():
                if name not in self.shadow:
                    self.shadow[name] = tensor.detach().clone()


class ADAController:
    def __init__(self):
        self.p = 0.0
        self.sign_ema = ADA_TARGET

    def update(self, real_sign_rate):
        self.sign_ema = 0.95 * self.sign_ema + 0.05 * float(real_sign_rate)
        if self.sign_ema > ADA_TARGET:
            self.p += ADA_STEP
        else:
            self.p -= ADA_STEP
        self.p = float(np.clip(self.p, 0.0, ADA_MAX_P))
        return self.p

    def state_dict(self):
        return {"p": self.p, "sign_ema": self.sign_ema}

    def load_state_dict(self, state):
        self.p = float(state.get("p", 0.0))
        self.sign_ema = float(state.get("sign_ema", ADA_TARGET))


def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)


def _rand_saturation(x):
    mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0
    return (x - mean) * scale + mean


def _rand_contrast(x):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5
    return (x - mean) * scale + mean


def _rand_translation(x, ratio=DIFFAUG_TRANSLATION_RATIO):
    sx = int(x.size(2) * ratio + 0.5)
    sy = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-sx, sx + 1, (x.size(0), 1, 1), device=x.device)
    ty = torch.randint(-sy, sy + 1, (x.size(0), 1, 1), device=x.device)
    bx = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    gx, gy = torch.meshgrid(
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device), indexing="ij"
    )
    gx = gx.unsqueeze(0) + tx + sx
    gy = gy.unsqueeze(0) + ty + sy
    padded = F.pad(x, (sy, sy, sx, sx), mode="replicate")
    return padded.permute(0, 2, 3, 1)[bx, gx, gy].permute(0, 3, 1, 2)


def _rand_cutout(x, ratio=DIFFAUG_CUTOUT_RATIO):
    h = max(1, int(x.size(2) * ratio + 0.5))
    w = max(1, int(x.size(3) * ratio + 0.5))
    ox = torch.randint(0, x.size(2), (x.size(0), 1, 1), device=x.device)
    oy = torch.randint(0, x.size(3), (x.size(0), 1, 1), device=x.device)
    gx, gy = torch.meshgrid(
        torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij"
    )
    gx = torch.clamp(gx.unsqueeze(0) + ox - h // 2, 0, x.size(2) - 1)
    gy = torch.clamp(gy.unsqueeze(0) + oy - w // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), device=x.device, dtype=x.dtype)
    batch = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    mask[batch, gx, gy] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x):
    y = _rand_brightness(x)
    y = _rand_saturation(y)
    y = _rand_contrast(y)
    y = _rand_translation(y)
    y = _rand_cutout(y)
    return y.contiguous()


def adaptive_diff_augment(x, probability):
    if probability <= 0:
        return x
    mask = (torch.rand(x.size(0), 1, 1, 1, device=x.device) < probability)
    augmented = diff_augment(x)
    return torch.where(mask, augmented, x).contiguous()


class FIDCalculator:
    def __init__(self, device):
        self.device = device
        inc = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        inc.fc = nn.Identity()
        self.inc = inc.eval().to(device)
        for p in self.inc.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def features(self, images):
        x = (images + 1.0) / 2.0
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return self.inc(x).detach().cpu().numpy()

    @torch.no_grad()
    def real_stats(self, loader, n):
        features = []
        count = 0
        for images in loader:
            features.append(self.features(images.to(self.device)))
            count += images.size(0)
            if count >= n:
                break
        feat = np.concatenate(features, axis=0)[:n]
        return feat.mean(axis=0), np.cov(feat, rowvar=False)

    @torch.no_grad()
    def fake_stats(self, model, noise_bank, n):
        model.eval()
        features = []
        for start in range(0, n, 64):
            noise = noise_bank[start:start + min(64, n - start)].to(self.device)
            features.append(self.features(model(noise)))
        feat = np.concatenate(features, axis=0)[:n]
        return feat.mean(axis=0), np.cov(feat, rowvar=False)

    @staticmethod
    def from_stats(real_mean, real_cov, fake_mean, fake_cov):
        diff = real_mean - fake_mean
        covmean = linalg.sqrtm(real_cov.dot(fake_cov))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(real_cov + fake_cov - 2.0 * covmean))


def save_grid(images, path, nrow=8):
    grid = make_grid(images, nrow=nrow, normalize=True, value_range=(-1, 1))
    array = grid.mul(255).clamp(0, 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    Image.fromarray(array).save(path)


@torch.no_grad()
def sample_images(model, noise_bank, n):
    model.eval()
    output = []
    for start in range(0, n, 64):
        output.append(model(noise_bank[start:start + min(64, n - start)].to(DEVICE)).cpu())
    return torch.cat(output, dim=0)[:n]


def laplacian_values(images):
    values = []
    for image in images:
        arr = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        values.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.asarray(values, dtype=np.float64)


def edge_density(images):
    values = []
    for image in images:
        arr = (image.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        values.append(float((cv2.Canny(gray, 50, 150) > 0).mean()))
    return float(np.mean(values))


def load_state(model, path, label):
    if not path:
        return False
    path = Path(path)
    if not path.exists():
        print(f"[init] {label} not found: {path}")
        return False
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        if label == "G" and "generator_ema" in state:
            state = state["generator_ema"]
        elif label == "G" and "generator" in state and not any(k.startswith("net.") for k in state):
            state = state["generator"]
        elif label == "D" and "discriminator" in state:
            state = state["discriminator"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[init] loaded {label}: {path}")
    if missing:
        print(f"[init] missing keys ({label}): {len(missing)}")
    if unexpected:
        print(f"[init] unexpected keys ({label}): {len(unexpected)}")
    return True


def locate_init_paths(args):
    g_path, d_path = args.init_g, args.init_d
    # This script is intentionally from-scratch by default. Explicit paths
    # still take precedence; automatic Exp11 discovery requires the opt-in
    # --use-exp11-init flag so an attached weights dataset cannot be used by
    # accident.
    if not args.use_exp11_init:
        return g_path, d_path
    roots = [Path("/kaggle/input"), Path.cwd()]
    if not g_path:
        for root in roots:
            if root.exists():
                hits = sorted(root.rglob("generator_ema_final.pth"))
                preferred = [
                    p for p in hits
                    if "11_G_DiffAug_EMA_20K" in str(p) or "exp11" in str(p).lower()
                ]
                hits = preferred + [p for p in hits if p not in preferred]
                if hits:
                    g_path = str(hits[0])
                    break
    if not d_path:
        for root in roots:
            if root.exists():
                hits = sorted(root.rglob("discriminator_final.pth"))
                preferred = [
                    p for p in hits
                    if "11_G_DiffAug_EMA_20K" in str(p) or "exp11" in str(p).lower()
                ]
                hits = preferred + [p for p in hits if p not in preferred]
                if hits:
                    d_path = str(hits[0])
                    break
    return g_path, d_path


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def main(argv=None):
    global DEVICE
    args = parse_args(argv)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_all_seeds(SEED)

    output_dir = Path(args.output_root) / EXPERIMENT_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root, all_paths = discover_dataset(args.dataset_path)
    if len(all_paths) > args.dataset_limit:
        rng = random.Random(SEED)
        all_paths = sorted(rng.sample(all_paths, args.dataset_limit))
    print(f"[dataset] training pool: {len(all_paths)} images (limit={args.dataset_limit})")

    train_ds = AnimeDataset(all_paths, train_transform())
    eval_ds = AnimeDataset(all_paths, eval_transform())
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=2, pin_memory=(DEVICE.type == "cuda"), persistent_workers=False,
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=64, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=(DEVICE.type == "cuda"), persistent_workers=False,
    )

    g_path, d_path = locate_init_paths(args)
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    loaded_g = load_state(G, g_path, "G")
    loaded_d = load_state(D, d_path, "D")
    if not loaded_g:
        print("[init] no Exp11 Generator supplied; training from scratch")
    if not loaded_d:
        print("[init] no Exp11 Discriminator supplied; training from scratch")

    ema = EMA(G)
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)
    ada = ADAController()
    start_epoch = 1

    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            state = torch.load(resume_path, map_location="cpu")
            G.load_state_dict(state["generator"], strict=False)
            D.load_state_dict(state["discriminator"], strict=False)
            if "ema" in state:
                ema.load_state_dict(state["ema"], model=G)
            if "optimizer_G" in state:
                g_opt.load_state_dict(state["optimizer_G"])
            if "optimizer_D" in state:
                d_opt.load_state_dict(state["optimizer_D"])
            if "ada" in state:
                ada.load_state_dict(state["ada"])
            start_epoch = int(state.get("epoch", 0)) + 1
            print(f"[resume] continuing from epoch {start_epoch}")

    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)
    fid_noise = torch.randn(max(args.final_fid, args.quick_fid), NOISE_DIM, 1, 1)

    config = {
        "experiment": EXPERIMENT_NAME,
        "base": "Exp11 Width3x + SN/Hinge + EMA",
        "dataset_root": dataset_root,
        "dataset_size": len(all_paths),
        "epochs": args.max_epochs,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "training_mode": "final_quality_candidate_with_fid_early_stopping",
        "early_stopping": {
            "fid_interval": args.fid_interval,
            "patience": args.fid_patience,
            "min_delta": args.fid_min_delta,
            "rise_margin": args.fid_rise_margin,
        },
        "batch_size": BATCH_SIZE,
        "noise_dim": NOISE_DIM,
        "seed": SEED,
        "lr": LR,
        "betas": BETAS,
        "anti_alias": {"type": "fixed_binomial_3x3", "blend": AA_BLEND, "feature_maps": [8, 16, 32]},
        "adaptive_diffaugment": {
            "policy": DIFFAUG_POLICY,
            "target": ADA_TARGET,
            "step": ADA_STEP,
            "max_probability": ADA_MAX_P,
            "translation_ratio": DIFFAUG_TRANSLATION_RATIO,
            "cutout_ratio": DIFFAUG_CUTOUT_RATIO,
        },
        "ema_decay": EMA_DECAY,
        "initial_generator": g_path or None,
        "initial_discriminator": d_path or None,
        "fid_protocol": "project legacy torchvision Inception-v3; same protocol as Exp11",
        "commercial_note": "FID alone is not a commercial quality or licensing guarantee.",
    }
    save_json(output_dir / "training_config.json", config)
    (output_dir / "dataset_manifest.txt").write_text("\n".join(all_paths), encoding="utf-8")

    fid_calc = None
    fid_ready = False
    fid_unavailable_reason = None
    real_stats_quick = None
    real_stats_final = None
    try:
        print("[fid] loading Inception-v3 and caching real feature statistics")
        fid_calc = FIDCalculator(DEVICE)
        real_stats_quick = fid_calc.real_stats(eval_loader, args.quick_fid)
        real_stats_final = fid_calc.real_stats(eval_loader, args.final_fid)
        fid_ready = True
    except Exception as exc:
        fid_unavailable_reason = str(exc)
        print(f"[fid] unavailable during training: {exc}")

    loss_path = output_dir / "loss.csv"
    loss_file = open(loss_path, "w", newline="", encoding="utf-8")
    writer = csv.writer(loss_file)
    writer.writerow(["epoch", "D_loss", "G_loss", "D_real", "D_fake", "real_sign_rate", "ada_p"])

    best_quick_fid = float("inf")
    best_quick_epoch = None
    best_ema_state = None
    global_step = 0
    bad_fid_evals = 0
    fid_history = []
    stop_reason = None
    completed_epoch = start_epoch - 1
    if args.resume:
        history_path = output_dir / "fid_history.json"
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
                        best_path = output_dir / "generator_ema_best_quick.pth"
                        if best_path.exists():
                            best_ema_state = torch.load(best_path, map_location="cpu")
                        bad_fid_evals = int(valid[-1].get("bad_fid_evals", 0))
                        print(
                            f"[resume] restored FID history: {len(fid_history)} evaluations, "
                            f"best={best_quick_fid:.4f} at epoch {best_quick_epoch}"
                        )
            except Exception as exc:
                print(f"[resume] could not restore FID history: {exc}")

    print(
        f"[run] {EXPERIMENT_NAME} on {DEVICE}; max_epochs={args.max_epochs}; "
        f"min_epochs={args.min_epochs}; steps/epoch={len(train_loader)}"
    )
    for epoch in range(start_epoch, args.max_epochs + 1):
        completed_epoch = epoch
        G.train()
        D.train()
        d_last = g_last = dr_last = df_last = 0.0
        sign_values = []
        for real in train_loader:
            real = real.to(DEVICE, non_blocking=True)
            bs = real.size(0)

            d_opt.zero_grad(set_to_none=True)
            z = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            with torch.no_grad():
                fake = G(z)
            d_real = D(adaptive_diff_augment(real, ada.p))
            d_fake = D(adaptive_diff_augment(fake, ada.p))
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_loss.backward()
            d_opt.step()

            D.requires_grad_(False)
            g_opt.zero_grad(set_to_none=True)
            z = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake_for_g = G(z)
            g_loss = -D(adaptive_diff_augment(fake_for_g, ada.p)).mean()
            g_loss.backward()
            g_opt.step()
            D.requires_grad_(True)
            ema.update(G)

            d_last = float(d_loss.item())
            g_last = float(g_loss.item())
            dr_last = float(d_real.mean().item())
            df_last = float(d_fake.mean().item())
            sign_values.append(float((d_real > 0).float().mean().item()))
            global_step += 1
            if global_step % ADA_INTERVAL_STEPS == 0 and sign_values:
                ada.update(float(np.mean(sign_values)))
                sign_values = []

        if sign_values:
            ada.update(float(np.mean(sign_values)))
        sign_rate = float(np.mean(sign_values)) if sign_values else float(ada.sign_ema)
        writer.writerow([epoch, d_last, g_last, dr_last, df_last, sign_rate, ada.p])
        loss_file.flush()

        print(
            f"Epoch [{epoch:03d}/{args.max_epochs}] D={d_last:.4f} G={g_last:.4f} "
            f"D(real/fake)={dr_last:+.3f}/{df_last:+.3f} ADA_p={ada.p:.3f}"
        )

        if epoch % SAMPLE_INTERVAL == 0 or epoch == args.max_epochs:
            train_state = deepcopy(G.state_dict())
            ema.apply_to(G)
            G.eval()
            save_grid(G(fixed_noise), output_dir / f"epoch_{epoch:03d}.png")
            G.load_state_dict(train_state, strict=True)
            # Keep one resumable checkpoint instead of producing many large
            # per-epoch weight files. Best and deploy weights are saved below.
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "generator": G.state_dict(),
                "discriminator": D.state_dict(),
                "generator_ema": ema.model_state_dict(),
                "ema": ema.state_dict(),
                "ada": ada.state_dict(),
                "optimizer_G": g_opt.state_dict(),
                "optimizer_D": d_opt.state_dict(),
                "config": config,
            }, output_dir / "checkpoint_latest.pth")

        should_stop = False
        if fid_ready and (epoch % args.fid_interval == 0 or epoch == args.max_epochs):
            train_state = deepcopy(G.state_dict())
            ema.apply_to(G)
            quick_fid = -1.0
            if real_stats_quick is not None:
                fake_mean, fake_cov = fid_calc.fake_stats(G, fid_noise, args.quick_fid)
                quick_fid = FIDCalculator.from_stats(
                    real_stats_quick[0], real_stats_quick[1], fake_mean, fake_cov
                )
            improved = (
                quick_fid >= 0
                and quick_fid < best_quick_fid - args.fid_min_delta
            )
            if improved:
                best_quick_fid = quick_fid
                best_quick_epoch = epoch
                best_ema_state = ema.model_state_dict()
                bad_fid_evals = 0
                torch.save(best_ema_state, output_dir / "generator_ema_best_quick.pth")
                save_json(output_dir / "best_checkpoint.json", {
                    "epoch": epoch,
                    "quick_fid": quick_fid,
                    "ada_p": ada.p,
                    "reason": "best quick FID so far",
                })
            elif quick_fid >= 0:
                bad_fid_evals += 1

            fid_history.append({
                "epoch": epoch,
                "quick_fid": quick_fid,
                "best_quick_fid": (
                    best_quick_fid if np.isfinite(best_quick_fid) else None
                ),
                "bad_fid_evals": bad_fid_evals,
                "ada_p": ada.p,
                "improved": improved,
            })
            save_json(output_dir / "fid_history.json", fid_history)
            print(
                f"[fid] epoch={epoch} quick_fid={quick_fid:.4f} "
                f"best={best_quick_fid:.4f} bad={bad_fid_evals}/{args.fid_patience}"
            )

            recent = [x["quick_fid"] for x in fid_history[-3:] if x["quick_fid"] >= 0]
            rising = (
                len(recent) == 3
                and recent[-1] > recent[-2]
                and recent[-2] > recent[-3]
            )
            should_stop = (
                epoch >= args.min_epochs
                and quick_fid >= 0
                and bad_fid_evals >= args.fid_patience
                and quick_fid > best_quick_fid + args.fid_rise_margin
                and rising
            )
            G.load_state_dict(train_state, strict=True)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            if should_stop:
                stop_reason = (
                    f"FID rose for {args.fid_patience} evaluations after "
                    f"epoch {args.min_epochs}; safety ceiling not reached"
                )
                print(f"[early-stop] {stop_reason}")
                break

    if stop_reason is None:
        stop_reason = (
            "reached max_epochs or completed requested training"
            if fid_ready
            else "FID unavailable; reached safety max_epochs without early stopping"
        )
    save_json(output_dir / "stop_reason.json", {
        "reason": stop_reason,
        "completed_epoch": completed_epoch,
        "max_epochs": args.max_epochs,
        "best_quick_epoch": best_quick_epoch,
        "best_quick_fid": (
            best_quick_fid if np.isfinite(best_quick_fid) else None
        ),
        "fid_unavailable_reason": fid_unavailable_reason,
    })

    loss_file.close()

    # Save final training artifacts.
    torch.save(G.state_dict(), output_dir / "generator_raw_final.pth")
    torch.save(D.state_dict(), output_dir / "discriminator_final.pth")
    final_ema_state = ema.model_state_dict()
    torch.save(final_ema_state, output_dir / "generator_ema_final.pth")
    torch.save(ema.state_dict(), output_dir / "ema_state_final.pth")
    torch.save(g_opt.state_dict(), output_dir / "optimizer_G.pth")
    torch.save(d_opt.state_dict(), output_dir / "optimizer_D.pth")
    torch.save({
        "epoch": completed_epoch,
        "global_step": global_step,
        "generator": G.state_dict(),
        "discriminator": D.state_dict(),
        "generator_ema": final_ema_state,
        "ema": ema.state_dict(),
        "ada": ada.state_dict(),
        "optimizer_G": g_opt.state_dict(),
        "optimizer_D": d_opt.state_dict(),
        "config": config,
    }, output_dir / "checkpoint_final.pth")

    # Full-FID comparison between the best quick candidate and the final EMA.
    selected_state = final_ema_state
    selected_label = "final_ema"
    full_fid_best = None
    full_fid_final = None
    if fid_ready and real_stats_final is not None:
        for label, state in [("final_ema", final_ema_state), ("best_quick", best_ema_state)]:
            if state is None:
                continue
            G.load_state_dict(state, strict=True)
            fake_mean, fake_cov = fid_calc.fake_stats(G, fid_noise, args.final_fid)
            full_fid = FIDCalculator.from_stats(
                real_stats_final[0], real_stats_final[1], fake_mean, fake_cov
            )
            print(f"[fid] {label} full_fid={full_fid:.4f}")
            if label == "final_ema":
                full_fid_final = full_fid
            else:
                full_fid_best = full_fid
            if full_fid_best is not None and full_fid_final is not None:
                if full_fid_best < full_fid_final:
                    selected_state = best_ema_state
                    selected_label = "best_quick_full_eval"
                else:
                    selected_state = final_ema_state
                    selected_label = "final_ema_full_eval"
    if selected_state is not None:
        torch.save(selected_state, output_dir / "generator_ema_deploy.pth")

    # Simple fixed-sample image diagnostics for sharpness/coverage monitoring.
    G.load_state_dict(selected_state, strict=True)
    fake_images = sample_images(G, fixed_noise, 64).cpu()
    save_grid(fake_images, output_dir / "deploy_sample_grid.png")
    real_images = []
    for batch in DataLoader(AnimeDataset(all_paths[:64], eval_transform()), batch_size=64, shuffle=False):
        real_images = batch
        break
    fake01 = (fake_images + 1.0) / 2.0
    real01 = (real_images + 1.0) / 2.0 if isinstance(real_images, torch.Tensor) else None
    fake_lap = laplacian_values(fake01)
    real_lap = laplacian_values(real01) if real01 is not None else np.asarray([])
    blur_threshold = float(np.percentile(real_lap, 10)) if len(real_lap) else None
    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "dataset_size": len(all_paths),
        "completed_epoch": completed_epoch,
        "max_epochs": args.max_epochs,
        "early_stop_reason": stop_reason,
        "fid_unavailable_reason": fid_unavailable_reason,
        "fid_history": fid_history,
        "best_quick_fid": best_quick_fid if np.isfinite(best_quick_fid) else None,
        "best_quick_epoch": best_quick_epoch,
        "full_fid_best_quick": full_fid_best,
        "full_fid_final_ema": full_fid_final,
        "selected_deploy_state": selected_label,
        "selected_full_fid": min(
            [x for x in [full_fid_best, full_fid_final] if x is not None],
            default=None,
        ),
        "laplacian_mean_fake_64": float(fake_lap.mean()),
        "laplacian_mean_real_64": float(real_lap.mean()) if len(real_lap) else None,
        "blur_threshold_real_p10": blur_threshold,
        "blur_rate_fake_below_real_p10": (
            float((fake_lap < blur_threshold).mean()) if blur_threshold is not None else None
        ),
        "edge_density_fake_64": edge_density(fake01),
        "ada_final_p": ada.p,
        "fid_protocol": "Legacy project Inception-v3; compare only with the same protocol.",
        "quality_note": "This is a 64x64 research candidate; FID is not a commercial guarantee.",
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(output_dir / "metrics.json", metrics)
    print(f"[done] {EXPERIMENT_NAME}")
    print(f"[done] deploy weight: {output_dir / 'generator_ema_deploy.pth'}")
    print(f"[done] metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
