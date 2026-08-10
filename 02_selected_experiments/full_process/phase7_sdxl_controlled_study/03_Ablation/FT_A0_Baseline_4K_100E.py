"""
FT-A0: Exp11 Fine-Tuning Baseline — 100% 原图 (4K), 100 Epoch
================================================================================
Kaggle GPU (T4×2), Internet ON (InceptionV3 下载用)。

前提:
  - Kaggle Input 已上传 Exp11 权重 (generator_ema_final.pth, discriminator_final.pth)
  - 原版 Anime Faces 数据集已 attached

输入 (自动检测 /kaggle/input/):
  - Exp11 generator_ema_final.pth + discriminator_final.pth
  - 原版 anime-faces 数据集

输出 (/kaggle/working/dcgan_output/FT_A0_4K_100E/):
  - generator_ema_final.pth     ← 部署 + 后续评估用
  - discriminator_final.pth
  - baseline_reference.json     ← A20/A50 早停参照 (FID + Coverage + 全部指标)
  - metrics.json, loss.csv, loss_curves.png, epoch_*.png
  - checkpoint_latest.pth       ← 中断续训

设计:
  加载 Exp11 的 G+D 权重 → 在固定 4K 原图上 fine-tune 100 epoch
  DiffAugment + EMA 全程开启。此为 4K/100E 协议下的 A0 对照基线。
"""

import os, csv, json, random, gc, hashlib, sys
from pathlib import Path; from datetime import datetime
import numpy as np; import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageFilter; import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models; from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader; from scipy import linalg

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
EXPERIMENT_NAME = "FT_A0_4K_100E"
OUTPUT_DIR      = "/kaggle/working/dcgan_output"
EXP_DIR         = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

N_ORIG  = 4000    # 原图数量 (4K baseline)
N_SDXL  = 0       # SDXL 新图数量

IMAGE_SIZE = 64; BATCH_SIZE = 32; NOISE_DIM = 128
LR = 1e-4; BETAS = (0.5, 0.99); SEED = 42
EPOCHS = 100; SAMPLE_INTERVAL = 25
N_FID = 5000

DIFFAUG_POLICY = "color,translation,cutout"
DIFFAUG_TRANSLATION_RATIO = 0.125; DIFFAUG_CUTOUT_RATIO = 0.35
EMA_DECAY = 0.9999

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} | Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def set_all_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def find_all_images(root_dir):
    exts = {".png", ".jpg", ".jpeg", ".webp"}; files = []
    if not os.path.exists(root_dir): return files
    for dp, dirs, fn in os.walk(root_dir):
        if os.path.abspath(dp) == os.path.abspath(root_dir):
            dirs[:] = [d for d in dirs if d.lower() != "data"]
        for f in fn:
            if Path(f).suffix.lower() in exts: files.append(os.path.join(dp, f))
    return sorted(files)

def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as h:
        for block in iter(lambda: h.read(1024*1024), b""): digest.update(block)
    return digest.hexdigest()

def save_image_grid(tensor, fp, nrow=8):
    grid = make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).detach().numpy()
    Image.fromarray(ndarr).save(fp)

# ═══════════════════════════════════════════════════════════════
# 数据增强 + DiffAugment
# ═══════════════════════════════════════════════════════════════
class EdgeSharpen:
    def __init__(self, prob=0.2, alpha=0.3): self.prob, self.alpha = prob, alpha
    def __call__(self, img):
        if random.random() < self.prob:
            arr = np.array(img, dtype=np.float32) / 255.0
            blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0
            sharp = arr + self.alpha * (arr - blurred)
            return Image.fromarray(np.clip(sharp * 255, 0, 255).astype(np.uint8))
        return img

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(p=0.5),
        EdgeSharpen(prob=0.2), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# DiffAugment
def _rand_brightness(x): return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)
def _rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0) + x_mean
def _rand_contrast(x):
    x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5) + x_mean
def _rand_translation(x, ratio=DIFFAUG_TRANSLATION_RATIO):
    shift_x = int(x.size(2) * ratio + 0.5); shift_y = int(x.size(3) * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, (x.size(0), 1, 1), device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, (x.size(0), 1, 1), device=x.device)
    bi = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    gx, gy = torch.meshgrid(torch.arange(x.size(2), device=x.device), torch.arange(x.size(3), device=x.device), indexing="ij")
    gx = gx.unsqueeze(0) + tx + shift_x; gy = gy.unsqueeze(0) + ty + shift_y
    xp = F.pad(x, (shift_y, shift_y, shift_x, shift_x), mode="replicate")
    return xp.permute(0, 2, 3, 1)[bi, gx, gy].permute(0, 3, 1, 2)
def _rand_cutout(x, ratio=DIFFAUG_CUTOUT_RATIO):
    h = int(x.size(2) * ratio + 0.5); w = int(x.size(3) * ratio + 0.5)
    ox = torch.randint(0, x.size(2) + (1 - h % 2), (x.size(0), 1, 1), device=x.device)
    oy = torch.randint(0, x.size(3) + (1 - w % 2), (x.size(0), 1, 1), device=x.device)
    gx, gy = torch.meshgrid(torch.arange(h, device=x.device), torch.arange(w, device=x.device), indexing="ij")
    gx = torch.clamp(gx.unsqueeze(0) + ox - h // 2, 0, x.size(2) - 1)
    gy = torch.clamp(gy.unsqueeze(0) + oy - w // 2, 0, x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), device=x.device, dtype=x.dtype)
    bi = torch.arange(x.size(0), device=x.device).view(-1, 1, 1); mask[bi, gx, gy] = 0
    return x * mask.unsqueeze(1)
def diff_augment(x, policy=DIFFAUG_POLICY):
    for item in [p.strip() for p in policy.split(",") if p.strip()]:
        if item == "color": x = _rand_brightness(x); x = _rand_saturation(x); x = _rand_contrast(x)
        elif item == "translation": x = _rand_translation(x)
        elif item == "cutout": x = _rand_cutout(x)
    return x.contiguous()

# ═══════════════════════════════════════════════════════════════
# 模型 (= Exp11, 不变)
# ═══════════════════════════════════════════════════════════════
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nd, 768, 4), nn.BatchNorm2d(768), nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1), nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh()
        ); self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)

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
            nn.Linear(256, 1)
        )
    def forward(self, x): return self.net(x).view(-1)

# ═══════════════════════════════════════════════════════════════
# EMA
# ═══════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay = decay; self.shadow = {}
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n] = p.data.clone().detach()
        for n, b in model.named_buffers(): self.shadow[n] = b.data.clone().detach()
    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
        for n, b in model.named_buffers():
            if torch.is_floating_point(b): self.shadow[n].mul_(self.decay).add_(b.data, alpha=1-self.decay)
            else: self.shadow[n].copy_(b.data)
    @torch.no_grad()
    def apply_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.shadow[n])
        for n, b in model.named_buffers(): b.data.copy_(self.shadow[n])
    def model_state_dict(self, model):
        return {n: self.shadow.get(n, t).detach().cpu().clone() for n, t in model.state_dict().items()}
    def state_dict(self): return {"decay": self.decay, "shadow": self.shadow}
    def load_state_dict(self, sd): self.decay = sd["decay"]; self.shadow = sd["shadow"]

# ═══════════════════════════════════════════════════════════════
# FID Calculator
# ═══════════════════════════════════════════════════════════════
class FIDCalculator:
    def __init__(self, device):
        self.device = device
        inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        inc.fc = nn.Identity(); inc.eval(); self.inc = inc.to(device)
        for p in self.inc.parameters(): p.requires_grad = False
    @torch.no_grad()
    def _feat(self, imgs):
        imgs = (imgs + 1) / 2.0; imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        imgs = (imgs - 0.5) / 0.5; return self.inc(imgs).cpu().numpy()
    @torch.no_grad()
    def compute_fid(self, G, real_loader, n=N_FID):
        G.eval(); rf, ff = [], []; c = 0
        for imgs in real_loader:
            rf.append(self._feat(imgs.to(self.device))); c += imgs.size(0)
            if c >= n: break
        rf = np.concatenate(rf, axis=0)[:n]; g = 0
        while g < n:
            bs = min(64, n - g); noise = torch.randn(bs, NOISE_DIM, 1, 1, device=self.device)
            ff.append(self._feat(G(noise))); g += bs
        ff = np.concatenate(ff, axis=0)[:n]
        mr = np.mean(rf, axis=0); sr = np.cov(rf, rowvar=False)
        mf = np.mean(ff, axis=0); sf = np.cov(ff, rowvar=False)
        d = mr - mf; cm = linalg.sqrtm(sr.dot(sf))
        if np.iscomplexobj(cm): cm = cm.real
        G.train(); return float(d.dot(d) + np.trace(sr + sf - 2 * cm))

# ═══════════════════════════════════════════════════════════════
# Coverage (k-NN Recall — Kynkäänniemi et al. NeurIPS 2019)
# ═══════════════════════════════════════════════════════════════
from scipy.spatial import cKDTree

def compute_coverage(rf: np.ndarray, ff: np.ndarray, k: int = 5) -> dict:
    """
    k-NN Recall (Coverage): 多少比例的真实样本被生成样本"覆盖"。

    算法:
      1. 对每个真实样本, 找到它到第 k 近真实邻居的距离作为半径 r_i
      2. 检查它到最近生成样本的距离是否 ≤ r_i
      3. Coverage = 被覆盖的真实样本数 / 总真实样本数

    返回:
      coverage:    被覆盖的比例 (0~1)
      mean_radius: 平均 k-NN 半径
      mean_dist:   平均最近假样本距离
    """
    tree_real = cKDTree(rf)
    dist_real, _ = tree_real.query(rf, k=k + 1)  # +1 因为自己
    radii = dist_real[:, -1]  # 第 k 近邻居的距离

    tree_fake = cKDTree(ff)
    dist_fake, _ = tree_fake.query(rf, k=1)  # 最近的假样本

    covered = (dist_fake.ravel() <= radii)
    return {
        "coverage": float(covered.mean()),
        "mean_radius": float(radii.mean()),
        "mean_fake_dist": float(dist_fake.mean()),
        "k": k, "n_real": len(rf), "n_fake": len(ff),
    }

# ═══════════════════════════════════════════════════════════════
# 图像质量指标
# ═══════════════════════════════════════════════════════════════
class LPIPSCalculator:
    def __init__(self, device):
        self.device = device
        anet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT); anet.eval()
        self.layers = nn.ModuleList([anet.features[:3], anet.features[:6], anet.features[:9],
                                      anet.features[:12], anet.features]).to(device)
        for p in self.layers.parameters(): p.requires_grad = False
    def _norm(self, x):
        m = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - m) / s
    @torch.no_grad()
    def compute_lpips(self, a, b):
        a, b = self._norm(a), self._norm(b); total = 0.0
        for L in self.layers: f1, f2 = L(a), L(b); total += (f1 - f2).pow(2).mean(dim=[1, 2, 3])
        return (total / len(self.layers)).cpu().numpy()

@torch.no_grad()
def compute_diversity(G, lpips_calc, ns=500):
    G.eval(); imgs = []; g = 0
    while g < ns:
        bs = min(32, ns - g); noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
        imgs.append((G(noise)+1)/2.0); g += bs
    imgs = torch.cat(imgs, dim=0)[:ns]; npairs = 2000
    i1 = torch.randint(0, ns, (npairs,)); i2 = torch.randint(0, ns, (npairs,))
    scores = []
    for i in range(0, npairs, 50):
        e = min(i+50, npairs)
        scores.extend(lpips_calc.compute_lpips(imgs[i1[i:e]].to(DEVICE), imgs[i2[i:e]].to(DEVICE)))
    G.train(); return float(np.mean(scores))

def laplacian_variances(imgs):
    vars_ = []
    for img in imgs:
        arr = (img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); vars_.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.asarray(vars_, dtype=np.float64)

def compute_edge_density(imgs, real_imgs=None):
    densities = []
    for img in imgs:
        arr = (img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); edges = cv2.Canny(gray, 50, 150)
        densities.append((edges > 0).mean())
    fd = float(np.mean(densities))
    if real_imgs is not None:
        rd = [];
        for img in real_imgs:
            arr = (img.detach().permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); edges = cv2.Canny(gray, 50, 150)
            rd.append((edges > 0).mean())
        rdf = float(np.mean(rd)); return fd, rdf, fd/max(rdf, 1e-8)
    return fd, None, None

# ═══════════════════════════════════════════════════════════════
# 数据集
# ═══════════════════════════════════════════════════════════════
class AnimeDataset(Dataset):
    def __init__(self, paths, transform=None): self.paths, self.tf = paths, transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for _ in range(10):
            try: img = Image.open(self.paths[i]).convert("RGB"); return self.tf(img) if self.tf else img
            except (OSError, IOError): i = random.randint(0, len(self.paths)-1)
        raise RuntimeError("Failed to load any image after 10 retries")

def locate_dataset() -> str:
    """自动检测原版 Anime Faces 数据集路径"""
    candidates = [
        "/kaggle/input/datasets/soumikrakshit/anime-faces/data",
        "/kaggle/input/anime-faces/data",
        "/kaggle/input/anime_faces/data",
    ]
    for c in candidates:
        if os.path.isdir(c): return c
    # 搜索 /kaggle/input/
    for dp, dirs, _ in os.walk("/kaggle/input"):
        for d in dirs:
            if "anime" in d.lower() and "face" in d.lower():
                sub = os.path.join(dp, d)
                for sdp, sdirs, _ in os.walk(sub):
                    if "data" in sdirs:
                        return os.path.join(sdp, "data")
                    imgs = find_all_images(sdp)
                    if len(imgs) > 1000: return sdp
    raise FileNotFoundError("Cannot locate anime-faces dataset in /kaggle/input/")

def locate_exp11_weights() -> tuple[str, str]:
    """自动检测 Exp11 权重文件"""
    g_path, d_path = None, None
    for dp, _, fn in os.walk("/kaggle/input"):
        for f in fn:
            if "generator_ema_final" in f and f.endswith(".pth"):
                g_path = os.path.join(dp, f)
            if "discriminator_final" in f and f.endswith(".pth"):
                d_path = os.path.join(dp, f)
    if not g_path: raise FileNotFoundError("Exp11 generator_ema_final.pth not found in /kaggle/input/")
    if not d_path: raise FileNotFoundError("Exp11 discriminator_final.pth not found in /kaggle/input/")
    return g_path, d_path

def prepare_original_paths(dataset_dir: str, n: int) -> list:
    """SHA-256 去重 → 固定 seed=42 选 N 张原图"""
    imgs = find_all_images(dataset_dir)
    if not imgs: raise RuntimeError(f"No images found in {dataset_dir}")
    print(f"[data] Found {len(imgs)} images in dataset")
    seen, unique = set(), []
    for p in imgs:
        d = sha256_file(p)
        if d not in seen: seen.add(d); unique.append(p)
    print(f"[data] SHA-256 dedup: {len(imgs)} → {len(unique)} unique")
    set_all_seeds(SEED)
    if len(unique) > n: unique = random.sample(unique, n)
    print(f"[data] Sampled {len(unique)} original images")
    return sorted(unique)

# ═══════════════════════════════════════════════════════════════
# 主训练
# ═══════════════════════════════════════════════════════════════
def main():
    set_all_seeds(SEED); os.makedirs(EXP_DIR, exist_ok=True)

    # —— 定位数据 ——
    dataset_dir = locate_dataset()
    orig_paths = prepare_original_paths(dataset_dir, N_ORIG)
    all_paths = orig_paths  # A0: 100% 原图, 无 SDXL
    print(f"[data] Total training images: {len(all_paths)} ({N_ORIG} orig + {N_SDXL} SDXL)")

    # —— 定位 Exp11 权重 ——
    g_weight_path, d_weight_path = locate_exp11_weights()
    print(f"[weights] G: {g_weight_path}")
    print(f"[weights] D: {d_weight_path}")

    # —— 数据加载器 ——
    set_all_seeds(SEED)
    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)
    ds = AnimeDataset(all_paths, transform=get_transform())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # —— 模型: 加载 Exp11 权重 ——
    G = Generator().to(DEVICE); D = Discriminator().to(DEVICE)
    print("[model] Loading Exp11 weights ...")
    G.load_state_dict(torch.load(g_weight_path, map_location=DEVICE))
    D.load_state_dict(torch.load(d_weight_path, map_location=DEVICE))
    print("[model] Exp11 weights loaded successfully")

    # 重置 EMA (从加载的权重开始累计)
    ema = EMA(G, decay=EMA_DECAY)
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    gp = sum(p.numel() for p in G.parameters())
    dp = sum(p.numel() for p in D.parameters())
    bn = sum(1 for m in G.modules() if isinstance(m, nn.BatchNorm2d))

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT_NAME} — Fine-Tuning from Exp11")
    print(f"  {'─'*50}")
    print(f"  G: {gp:,} | D: {dp:,} | BN: {bn}")
    print(f"  Data: {len(all_paths):,} ({N_ORIG} orig + {N_SDXL} SDXL)")
    print(f"  Epochs: {EPOCHS} | Batch: {BATCH_SIZE} | LR: {LR}")
    print(f"  DiffAugment: {DIFFAUG_POLICY} | EMA: {EMA_DECAY}")
    print(f"{'='*60}\n")

    # —— Training config ——
    config = {
        "experiment": EXPERIMENT_NAME, "mode": "fine-tuning from Exp11",
        "pretrained_G": g_weight_path, "pretrained_D": d_weight_path,
        "n_original": N_ORIG, "n_sdxl": N_SDXL, "total_images": len(all_paths),
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "lr": LR, "betas": list(BETAS), "noise_dim": NOISE_DIM,
        "diffaugment": DIFFAUG_POLICY, "ema_decay": EMA_DECAY,
        "exp11_base_fid": 38.88
    }
    with open(os.path.join(EXP_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # —— 保存数据集 manifest ——
    with open(os.path.join(EXP_DIR, "dataset_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(all_paths))

    # —— Checkpoint 恢复 ——
    ckpt_path = os.path.join(EXP_DIR, "checkpoint_latest.pth")
    start_epoch = 1
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        G.load_state_dict(ckpt["generator"]); D.load_state_dict(ckpt["discriminator"])
        g_opt.load_state_dict(ckpt["optimizer_G"]); d_opt.load_state_dict(ckpt["optimizer_D"])
        ema.load_state_dict(ckpt["ema"]); start_epoch = ckpt["epoch"] + 1
        print(f"[resume] From epoch {start_epoch}")

    # —— CSV 日志 ——
    csv_mode = "a" if start_epoch > 1 else "w"
    csv_f = open(os.path.join(EXP_DIR, "loss.csv"), csv_mode, newline="")
    w = csv.writer(csv_f)
    if start_epoch == 1:
        w.writerow(["epoch", "D_loss", "G_loss", "D_real_aug", "D_fake_aug", "D_real_raw", "D_fake_raw"])

    print("Training ...\n")
    for ep in range(start_epoch, EPOCHS + 1):
        for img in dl:
            real = img.to(DEVICE); bs = real.size(0)

            d_opt.zero_grad()
            noise_z = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            with torch.no_grad(): fake = G(noise_z)
            d_real = D(diff_augment(real)); d_fake = D(diff_augment(fake))
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_loss.backward(); d_opt.step()

            D.requires_grad_(False)
            noise_z = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            g_loss = -D(diff_augment(G(noise_z))).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
            D.requires_grad_(True)
            ema.update(G)

        dl_v, gl_v = d_loss.item(), g_loss.item()
        dr_v, df_v = d_real.mean().item(), d_fake.mean().item()
        D.eval()
        with torch.no_grad(): dr_raw = D(real).mean().item(); df_raw = D(fake).mean().item()
        D.train()
        w.writerow([ep, dl_v, gl_v, dr_v, df_v, dr_raw, df_raw])
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  "
              f"DR/DF_aug:{dr_v:+.2f}/{df_v:+.2f}  raw:{dr_raw:+.2f}/{df_raw:+.2f}")

        if ep % SAMPLE_INTERVAL == 0:
            train_sd = {k: v.clone() for k, v in G.state_dict().items()}
            ema.apply_to(G); G.eval()
            with torch.no_grad():
                save_image_grid(G(fixed_noise), os.path.join(EXP_DIR, f"epoch_{ep:03d}.png"))
            G.load_state_dict(train_sd); del train_sd; G.train()
            torch.save(G.state_dict(), os.path.join(EXP_DIR, f"generator_epoch_{ep:03d}.pth"))
            torch.save(ema.model_state_dict(G), os.path.join(EXP_DIR, f"generator_ema_epoch_{ep:03d}.pth"))
            torch.save({"epoch": ep, "generator": G.state_dict(), "discriminator": D.state_dict(),
                        "optimizer_G": g_opt.state_dict(), "optimizer_D": d_opt.state_dict(),
                        "ema": ema.state_dict()}, ckpt_path)
            print(f"  [checkpoint] epoch {ep}")

    csv_f.close()

    # —— 最终保存 ——
    torch.save(G.state_dict(), os.path.join(EXP_DIR, "generator_raw_final.pth"))
    torch.save(D.state_dict(), os.path.join(EXP_DIR, "discriminator_final.pth"))
    torch.save(ema.model_state_dict(G), os.path.join(EXP_DIR, "generator_ema_final.pth"))
    save_image_grid(next(iter(dl))[:64], os.path.join(EXP_DIR, "real_images.png"))

    # —— 指标 ——
    print(f"\n{'='*60}\n  Computing Metrics\n{'='*60}")
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_ds = AnimeDataset(all_paths, transform=eval_tf)
    eval_dl = DataLoader(eval_ds, batch_size=64, shuffle=True, drop_last=False, num_workers=2)

    # —— 收集 InceptionV3 特征 (FID + Coverage 共用) ——
    print("Extracting InceptionV3 features (EMA) ...")
    fid_calc = FIDCalculator(DEVICE)
    train_sd = {k: v.clone() for k, v in G.state_dict().items()}
    ema.apply_to(G)

    # 收集真实特征 (遍历 eval_dl)
    rf, rc = [], 0
    for imgs in eval_dl:
        rf.append(fid_calc._feat(imgs.to(DEVICE)))
        rc += imgs.size(0)
        if rc >= N_FID: break
    rf = np.concatenate(rf, axis=0)[:N_FID]

    # 收集假特征
    ff, gc_ = [], 0
    G.eval()
    with torch.no_grad():
        while gc_ < N_FID:
            bs = min(64, N_FID - gc_)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            ff.append(fid_calc._feat(G(noise)))
            gc_ += bs
    ff = np.concatenate(ff, axis=0)[:N_FID]

    # FID (Legacy)
    mr = np.mean(rf, axis=0); sr = np.cov(rf, rowvar=False)
    mf = np.mean(ff, axis=0); sf = np.cov(ff, rowvar=False)
    d = mr - mf
    try:
        cm = linalg.sqrtm(sr.dot(sf))
        if np.iscomplexobj(cm): cm = cm.real
        fid = float(d.dot(d) + np.trace(sr + sf - 2 * cm))
    except Exception: fid = -1
    print(f"  FID: {fid:.2f}" if fid > 0 else "  FID: FAILED")

    # Coverage (k-NN Recall)
    print("Coverage (k-NN) ...")
    try:
        cov = compute_coverage(rf, ff, k=5)
        print(f"  Coverage: {cov['coverage']:.4f} | radius={cov['mean_radius']:.2f} | fake_dist={cov['mean_fake_dist']:.2f}")
    except Exception as e:
        cov = {"coverage": -1, "error": str(e)}
        print(f"  Coverage FAILED: {e}")

    G.load_state_dict(train_sd); del train_sd; del rf; del ff

    print("LPIPS + Diversity + Laplacian + Edge (EMA) ...")
    train_sd2 = {k: v.clone() for k, v in G.state_dict().items()}; ema.apply_to(G)
    lpips_calc = LPIPSCalculator(DEVICE)
    rb, fb = [], []; rc, n_gen = 0, 0
    for imgs in eval_dl:
        rb.append((imgs.to(DEVICE)+1)/2.0); rc += imgs.size(0)
        if rc >= 500: break
    rb = torch.cat(rb, dim=0)[:500]
    with torch.no_grad():
        while n_gen < 500:
            bs = min(32, 500-n_gen); noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fb.append((G(noise)+1)/2.0); n_gen += bs
    fb = torch.cat(fb, dim=0)[:500]
    div = compute_diversity(G, lpips_calc, ns=300)
    G.load_state_dict(train_sd2); del train_sd2

    lpips_scores = []
    for i in range(0, 500, 50):
        lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i+50,500)], rb[i:min(i+50,500)]))
    lpips_mean = float(np.mean(lpips_scores))
    flv = laplacian_variances(fb[:200]); rlv = laplacian_variances(rb[:200])
    lap = float(flv.mean()); rlap = float(rlv.mean())
    bt = float(np.percentile(rlv, 10)); br = float((flv < bt).mean())
    fe, re, er = compute_edge_density(fb[:200], rb[:200])
    with torch.no_grad():
        edr = float(D(rb[:200]*2.0-1.0).mean().item())
        edf = float(D(fb[:200]*2.0-1.0).mean().item())
    del lpips_calc; gc.collect()

    metrics = {
        "experiment": EXPERIMENT_NAME, "mode": "fine-tuning from Exp11",
        "n_original": N_ORIG, "n_sdxl": N_SDXL,
        "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "FID": round(fid, 2), "LPIPS_AlexFeat": round(lpips_mean, 4),
        "Diversity": round(div, 4),
        "Laplacian_Variance": round(lap, 2), "Laplacian_Variance_Real": round(rlap, 2),
        "Blur_Threshold_Real_P10": round(bt, 2), "Blur_Rate": round(br, 4),
        "Edge_Density_Fake": round(fe, 4), "Edge_Density_Real": round(re, 4),
        "Edge_Density_Ratio": round(er, 4),
        "Coverage": round(cov.get("coverage", -1), 4),
        "Coverage_K": cov.get("k", 5),
        "Coverage_MeanRadius": round(cov.get("mean_radius", 0), 2),
        "Coverage_MeanFakeDist": round(cov.get("mean_fake_dist", 0), 2),
        "D_real_eval": round(edr, 4), "D_fake_eval": round(edf, 4),
        "final_G_loss": round(gl_v, 4), "final_D_loss": round(dl_v, 4),
        "G_params": gp, "D_params": dp,
        "exp11_base_fid": 38.88,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)

    # —— baseline_reference.json (供 A20/A50 消融组做早停参照) ——
    baseline_reference = {
        "baseline_experiment": EXPERIMENT_NAME,
        "baseline_n_original": N_ORIG,
        "baseline_epochs": EPOCHS,
        "reference_FID": round(fid, 2),
        "reference_Coverage": round(cov.get("coverage", -1), 4),
        "reference_Diversity": round(div, 4),
        "reference_BlurRate": round(br, 4),
        "reference_EdgeDensity": round(fe, 4),
        "reference_LaplacianVar": round(lap, 2),
        "reference_LPIPS_AlexFeat": round(lpips_mean, 4),
        # 早停阈值 (A20/A50 脚本读取)
        "coverage_abort_threshold": round(cov.get("coverage", 0) * 0.90, 4),   # < 90% → 熔断
        "coverage_warn_threshold": round(cov.get("coverage", 0) * 0.95, 4),    # < 95% → 警告
        "fid_improvement_min": 0.5,          # FID 改善 ≥ 0.5 才考虑更新 best
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(EXP_DIR, "baseline_reference.json"), "w") as f:
        json.dump(baseline_reference, f, indent=2)
    print(f"\n  baseline_reference.json saved — Coverage abort @ <{baseline_reference['coverage_abort_threshold']:.4f}")

    # Loss curves
    df_loss = pd.read_csv(os.path.join(EXP_DIR, "loss.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(df_loss["epoch"], df_loss["G_loss"], color="#e74c3c", lw=1.5)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="G Loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df_loss["epoch"], df_loss["D_loss"], color="#3498db", lw=1.5)
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="D Loss"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(df_loss["epoch"], df_loss["D_real_aug"], color="#2ecc71", lw=1.5, label="D(Real aug)")
    axes[2].plot(df_loss["epoch"], df_loss["D_fake_aug"], color="#e67e22", lw=1.5, label="D(Fake aug)")
    axes[2].plot(df_loss["epoch"], df_loss["D_real_raw"], color="#27ae60", lw=1.0, ls="--", label="D(Real raw)")
    axes[2].plot(df_loss["epoch"], df_loss["D_fake_raw"], color="#d35400", lw=1.0, ls="--", label="D(Fake raw)")
    axes[2].legend(fontsize=7); axes[2].set(xlabel="Epoch", ylabel="Logit", title="D logits"); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, "loss_curves.png"), dpi=150); plt.close()

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT_NAME} DONE")
    print(f"  FID={fid:.2f} | Coverage={cov.get('coverage', -1):.4f} | LapVar={lap:.1f} | Diversity={div:.4f} | BlurRate={br:.4f}")
    print(f"  Output: {EXP_DIR}")
    print(f"  Download: {EXPERIMENT_NAME}/ (Kaggle output tab)")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
