"""
================================================================================
 G强化实验 10_G_Width3x_20K_Laplacian: 09 + Laplacian Pyramid Loss
================================================================================
 SCIENTIFIC QUESTION:
   在09(03 Widthx3 + 20K, FID=49.17)的最优基座上, 给G的损失函数增加
   Laplacian Pyramid多尺度高频监督, 能否在不破坏对抗平衡的前提下
   进一步解决"五官模糊、边界不清晰"的剩余缺陷?

 WHY LAPLACIAN (NOT SA / NOT CLIP / NOT WAVELET):
   Phase 1深度调优证明Laplacian Loss是所有辅助损失中最有效的
   (FID 109.4→98.67, -10.7). 机制: 4层金字塔逐层提取高频残差,
   粗层(8×8/16×16)罚结构 → 解决五官扭曲, 细层(32×32/64×64)罚纹理
   → 解决边缘模糊. LFSRGAN (Chen et al., IET 2025)在超分辨率上
   验证了Laplacian Pyramid + Frequency Loss的组合最优.

   CLIP/Wavelet/FFT 保留给后续实验. 本次仅加Laplacian — 单变量.

 WHY λ=0.05 (NOT 0.1):
   Phase 1弱基线(FID=109)用λ=0.1有效. 09的G更强(FID=49),
   对抗平衡更精细 → 更小的λ避免过度干扰. 若FID下降+DR稳定,
   λ是正确的. 若DR波动剧烈, 下一轮λ=0.01.

 BASELINE: 09 (FID=49.17, 20K, NO SA)
   09 = 03 Widthx3架构 + 20K数据. 无Self-Attention.
   09a (SA+20K)仍在CPU运行, 结果未知.

 SINGLE VARIABLE vs 09:
   仅改: G的损失函数 — Hinge + Laplacian Pyramid (λ=0.05)
   不改: G架构(768→384→192→96, BNx4, ConvTx5), D(SN+Hinge),
         batch=32, lr=1e-4, 优化器, 数据=20K, epoch=200, seed=42

 LAPLACIAN PYRAMID (4-level, RGB pixel space):
   Level 0 (64×64): I_0 = original image
   Level 1 (32×32): I_1 = downsample(I_0)
   Level 2 (16×16): I_2 = downsample(I_1)
   Level 3 (8×8):   I_3 = downsample(I_2)

   Laplacian_k = I_k - upsample(I_{k+1})  ← 高频残差 (该尺度被下采样抹掉的信息)
   L_lap = Σ w_k × L1(Lap_k(fake), Lap_k(real))
     w_0 = 0.25  纹理细节 (发丝、边缘像素)
     w_1 = 0.5   中间结构 (眼睛轮廓、嘴部线条)
     w_2 = 1.0   五官结构 (鼻子位置、眼距)
     w_3 = 1.0   脸型轮廓 (脸型椭圆、发型边界)
     粗层权重大 → 优先确保结构 → 五官扭曲
     细层权重小 → 细节灵活 → 不产生锐化噪声

 LOSS FUNCTION:
   L_G = -D(fake).mean() + 0.05 × L_lap

 DIAGNOSTIC SIGNALS:
   健康: FID↓ + LapVar↑ + DR 0.3~0.7 + DF -0.3~-0.6
   警告: FID↓ + DR>1.0 + DF<-1.0 → λ太大 → 下一轮λ=0.01
   失败: FID↑/不动 + 锐化噪声 → 关闭方向或λ=0.005
   无关: FID≈49 + 指标不变 → 20K数据已足够 → 转CLIP

 REFERENCE:
   Phase 1 08_G_Laplacian: FID 98.67 (baseline 109.4, -10.7)
   LFSRGAN (Chen et al., IET 2025): Laplacian + Frequency Loss
   LMCGAN (IEEE Access 2024): Perceptual Loss on Band-Pass Components
   Burt & Adelson (1983): Laplacian Pyramid as Compact Image Code
================================================================================
"""
import os, csv, json, random, gc
from pathlib import Path; from datetime import datetime
import numpy as np; import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageFilter; import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models; from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader; from scipy import linalg

EXPERIMENT_NAME = "10_G_Width3x_20K_Laplacian"
OUTPUT_DIR = "/kaggle/working/dcgan_output"
DATASET_PATH = "/kaggle/input/gananime-lite"
DATASET_LIMIT = 20000
IMAGE_SIZE = 64
BATCH_SIZE = 32
NOISE_DIM = 128
LR = 1e-4
BETAS = (0.5, 0.99)
SEED = 42
EPOCHS = 200
SAMPLE_INTERVAL = 50
N_FID = 10000
LAP_LAMBDA = 0.05          # [NEW] Laplacian loss weight
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

import time as _time
def _download_with_retry(model_fn, name):
    for attempt in range(5):
        try: return model_fn()
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt * 5
                print(f"  {name} retry in {wait}s...")
                _time.sleep(wait)
            else: raise e

print("Pre-loading models...")
try:
    _inc = _download_with_retry(
        lambda: models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False),
        "InceptionV3")
    del _inc; gc.collect()
    print("  InceptionV3: OK")
except: print("  InceptionV3: FAILED")
try:
    _anet = _download_with_retry(
        lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT),
        "AlexNet")
    del _anet; gc.collect()
    print("  AlexNet: OK")
except: print("  AlexNet: FAILED")
print()

def set_all_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_all_images(root_dir):
    exts = {".png", ".jpg", ".jpeg"}
    files = []
    if not os.path.exists(root_dir): return files
    for dp, _, fn in os.walk(root_dir):
        for f in fn:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(dp, f))
    return sorted(files)

def load_dataset():
    if os.path.exists(DATASET_PATH):
        imgs = find_all_images(DATASET_PATH)
        if imgs: print(f"Dataset: {DATASET_PATH} ({len(imgs)} images)"); return DATASET_PATH, imgs
    print("Scanning /kaggle/input/ ...")
    for sub in sorted(os.listdir("/kaggle/input")):
        sp = os.path.join("/kaggle/input", sub)
        if os.path.isdir(sp):
            imgs = find_all_images(sp)
            if imgs: print(f"Dataset: {sp} ({len(imgs)} images)"); return sp, imgs
    raise FileNotFoundError("No dataset found.")

class AnimeDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths; self.tf = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        for _ in range(10):
            try: img = Image.open(self.paths[i]).convert("RGB"); return self.tf(img) if self.tf else img
            except (OSError, IOError): i = random.randint(0, len(self.paths) - 1)
        raise RuntimeError("Failed to load any image after 10 retries")

def save_image_grid(tensor, fp, nrow=8):
    grid = make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    Image.fromarray(ndarr).save(fp)

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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        EdgeSharpen(prob=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

# =============================================================================
# [10 NEW] Laplacian Pyramid Loss — multi-scale high-frequency supervision
#
#   Pyramid construction (Burt & Adelson, 1983):
#     Level 0: original 64×64 image
#     Level k+1: avg_pool 2×2 of Level k → half resolution
#     Laplacian_k = I_k - upsample(I_{k+1})  ← high-frequency residual
#
#   Weighted L1 across 4 levels:
#     Coarse (8×8, 16×16): high weight → face structure and proportions
#     Fine (32×32, 64×64): low weight → texture and edge details
#
#   Input range: [-1, 1] (DCGAN output, normalized real)
#   Works directly on RGB pixels — no feature extraction dependency.
# =============================================================================

def laplacian_pyramid_loss(fake, real):
    """
    Compute weighted Laplacian pyramid L1 loss between fake and real images.

    Args:
        fake: [B, 3, 64, 64] generated images, range [-1, 1]
        real: [B, 3, 64, 64] real images, range [-1, 1]
    Returns:
        scalar loss
    """
    # Level weights: coarse → structure, fine → texture
    weights = [0.25, 0.5, 1.0, 1.0]  # 64×64, 32×32, 16×16, 8×8

    def build_gaussian_pyramid(img, levels=4):
        pyramid = [img]
        for _ in range(levels - 1):
            # Average pooling 2×2 = smooth downsample
            img = F.avg_pool2d(img, kernel_size=2, stride=2)
            pyramid.append(img)
        return pyramid

    def build_laplacian_pyramid(gaussian_pyr):
        laplacian_pyr = []
        for k in range(len(gaussian_pyr) - 1):
            # Upsample coarser level back to current resolution
            up = F.interpolate(gaussian_pyr[k + 1], size=gaussian_pyr[k].shape[-2:],
                               mode='bilinear', align_corners=False)
            # Laplacian = current - upsampled coarser = high-frequency residual
            lap = gaussian_pyr[k] - up
            laplacian_pyr.append(lap)
        # The coarsest level is kept as-is (no residual to compute)
        laplacian_pyr.append(gaussian_pyr[-1])
        return laplacian_pyr

    real_gauss = build_gaussian_pyramid(real, levels=4)
    fake_gauss = build_gaussian_pyramid(fake, levels=4)

    real_lap = build_laplacian_pyramid(real_gauss)
    fake_lap = build_laplacian_pyramid(fake_gauss)

    loss = 0.0
    for k in range(4):
        loss += weights[k] * F.l1_loss(fake_lap[k], real_lap[k])

    return loss / sum(weights)  # normalize by total weight

# =============================================================================
# Generator — bit-for-bit =03=09 Widthx3
#   ConvTranspose + BN + ReLU x5. NO changes.
# =============================================================================
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nd, 768, 4), nn.BatchNorm2d(768), nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1), nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh()
        )
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)

# =============================================================================
# Discriminator — SN+Hinge, =03=09
# =============================================================================
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


class FIDCalculator:
    def __init__(self, device):
        self.device = device
        inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        inc.fc = nn.Identity(); inc.eval(); self.inc = inc.to(device)
        for p in self.inc.parameters(): p.requires_grad = False

    @torch.no_grad()
    def _feat(self, imgs):
        imgs = (imgs + 1) / 2.0
        imgs = F.interpolate(imgs, size=(299, 299), mode="bilinear", align_corners=False)
        imgs = (imgs - 0.5) / 0.5
        return self.inc(imgs).cpu().numpy()

    @torch.no_grad()
    def compute_fid(self, G, real_loader, n=10000):
        G.eval(); rf, ff = [], []; c = 0
        for imgs in real_loader:
            rf.append(self._feat(imgs.to(self.device))); c += imgs.size(0)
            if c >= n: break
        rf = np.concatenate(rf, axis=0)[:n]; g = 0
        while g < n:
            bs = min(64, n - g)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=self.device)
            ff.append(self._feat(G(noise))); g += bs
        ff = np.concatenate(ff, axis=0)[:n]
        mr = np.mean(rf, axis=0); sr = np.cov(rf, rowvar=False)
        mf = np.mean(ff, axis=0); sf = np.cov(ff, rowvar=False)
        d = mr - mf; cm = linalg.sqrtm(sr.dot(sf))
        if np.iscomplexobj(cm): cm = cm.real
        G.train(); return float(d.dot(d) + np.trace(sr + sf - 2 * cm))


class LPIPSCalculator:
    def __init__(self, device):
        self.device = device
        anet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT); anet.eval()
        self.layers = nn.ModuleList([
            anet.features[:3], anet.features[:6], anet.features[:9],
            anet.features[:12], anet.features
        ]).to(device)
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
        bs = min(32, ns - g)
        noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
        imgs.append((G(noise) + 1) / 2.0); g += bs
    imgs = torch.cat(imgs, dim=0)[:ns]; npairs = 2000
    i1 = torch.randint(0, ns, (npairs,)); i2 = torch.randint(0, ns, (npairs,))
    scores = []
    for i in range(0, npairs, 50):
        e = min(i + 50, npairs)
        scores.extend(lpips_calc.compute_lpips(imgs[i1[i:e]].to(DEVICE), imgs[i2[i:e]].to(DEVICE)))
    G.train(); return float(np.mean(scores))


def compute_laplacian_variance(imgs):
    vars_ = []
    for img in imgs:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        vars_.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.mean(vars_))


def compute_edge_density(imgs, real_imgs=None):
    densities = []
    for img in imgs:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150); densities.append((edges > 0).mean())
    fake_density = float(np.mean(densities))
    if real_imgs is not None:
        real_densities = []
        for img in real_imgs:
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150); real_densities.append((edges > 0).mean())
        real_density = float(np.mean(real_densities))
        ratio = fake_density / max(real_density, 1e-8)
        return fake_density, real_density, ratio
    return fake_density, None, None


def main():
    set_all_seeds(SEED); os.makedirs(EXP_DIR, exist_ok=True)

    dataset_path, image_paths = load_dataset()
    if DATASET_LIMIT and len(image_paths) > DATASET_LIMIT:
        set_all_seeds(SEED); image_paths = random.sample(image_paths, DATASET_LIMIT)
        print(f"Subsampled to {DATASET_LIMIT} images (seed={SEED})")
    actual_dataset_size = len(image_paths)

    set_all_seeds(SEED)
    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)

    ds = AnimeDataset(image_paths, transform=get_transform())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # Verify architecture
    g_tmp = Generator(); d_tmp = Discriminator()
    gp = sum(p.numel() for p in g_tmp.parameters())
    dp = sum(p.numel() for p in d_tmp.parameters())
    bn_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.BatchNorm2d))
    convt_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.ConvTranspose2d))

    # Verify Laplacian loss function
    with torch.no_grad():
        test_real = torch.randn(4, 3, 64, 64)
        test_fake = torch.randn(4, 3, 64, 64)
        test_lap = laplacian_pyramid_loss(test_fake, test_real)
        test_out = g_tmp(torch.randn(4, NOISE_DIM, 1, 1))
    del g_tmp, d_tmp

    steps_per_epoch = len(dl)

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT_NAME}")
    print(f"  10: 09 + Laplacian Pyramid Loss (lambda={LAP_LAMBDA})")
    print(f"  {'─'*50}")
    print(f"  G: {gp:,} params (=09, =03)")
    print(f"  D: {dp:,} params (=09, SN+Hinge)")
    print(f"  BN: {bn_count} | ConvTranspose: {convt_count}")
    print(f"  Architecture = 09 = 03 (768→384→192→96→3), NO SA")
    print(f"  Data: {actual_dataset_size:,} images")
    print(f"  Laplacian Pyramid: 4 levels (64→32→16→8), RGB space")
    print(f"  Loss: L_G = Hinge + {LAP_LAMBDA} × L_lap")
    print(f"  Batch: {BATCH_SIZE} | lr: {LR} | Epochs: {EPOCHS} | Seed: {SEED}")
    print(f"  Baseline 09 FID: 49.17")
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    print(f"  Test L_lap: {test_lap:.4f} (sanity check)")
    print(f"  Output: {list(test_out.shape)}  Range: [{test_out.min():.3f}, {test_out.max():.3f}]")
    print(f"{'='*60}\n")

    G = Generator().to(DEVICE); D = Discriminator().to(DEVICE)
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    csv_f = open(os.path.join(EXP_DIR, "loss.csv"), "w", newline="")
    w = csv.writer(csv_f)
    w.writerow(["epoch", "D_loss", "G_loss", "G_adv", "L_lap", "D_real", "D_fake"])
    fdl, fgl, fdr, fdf = 0.0, 0.0, 0.0, 0.0

    epoch1_d_loss = None

    print("Training ...\n")
    for ep in range(1, EPOCHS + 1):
        for img in dl:
            real = img.to(DEVICE); bs = real.size(0)

            # === Discriminator update (unchanged from 09) ===
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            with torch.no_grad(): fake = G(noise)
            d_real = D(real); d_fake = D(fake)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # === Generator update (NEW: Laplacian term added) ===
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake = G(noise)
            g_adv = -D(fake).mean()
            g_lap = laplacian_pyramid_loss(fake, real)
            g_loss = g_adv + LAP_LAMBDA * g_lap
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v = d_loss.item(); gl_v = g_loss.item()
        dr_v = d_real.mean().item(); df_v = d_fake.mean().item()
        g_adv_v = g_adv.item(); g_lap_v = g_lap.item()
        w.writerow([ep, dl_v, gl_v, g_adv_v, g_lap_v, dr_v, df_v])
        fdl, fgl, fdr, fdf = dl_v, gl_v, dr_v, df_v

        if ep == 1: epoch1_d_loss = dl_v

        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  "
              f"G_adv:{g_adv_v:+.2f}  L_lap:{g_lap_v:.4f}  "
              f"DR:{dr_v:+.2f}  DF:{df_v:+.2f}")

        if ep % SAMPLE_INTERVAL == 0:
            G.eval(); samples = G(fixed_noise); G.train()
            save_image_grid(samples, os.path.join(EXP_DIR, f"epoch_{ep:03d}.png"))
            torch.save(G.state_dict(), os.path.join(EXP_DIR, f"generator_epoch_{ep:03d}.pth"))

    csv_f.close()
    torch.save(G.state_dict(), os.path.join(EXP_DIR, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(EXP_DIR, "discriminator_final.pth"))
    save_image_grid(next(iter(dl))[:64], os.path.join(EXP_DIR, "real_images.png"))

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS: {EXPERIMENT_NAME}")
    print(f"{'='*60}")

    df_loss = pd.read_csv(os.path.join(EXP_DIR, "loss.csv"))

    dr_max = df_loss["D_real"].max()
    dr_min = df_loss["D_real"].min()
    dr_std = df_loss["D_real"].std()
    d_loss_max = df_loss["D_loss"].max()
    near_collapse = (df_loss["D_loss"] > 1.95).sum()
    lap_mean = df_loss["L_lap"].mean()
    lap_final = df_loss["L_lap"].iloc[-1]
    lap_early = df_loss["L_lap"].iloc[:10].mean()
    lap_late  = df_loss["L_lap"].iloc[-10:].mean()

    final_dr = df_loss["D_real"].iloc[-1]
    final_df = df_loss["D_fake"].iloc[-1]

    print(f"  Epoch 1 D_loss: {epoch1_d_loss:.4f}")
    print(f"  D_real range: [{dr_min:+.2f}, {dr_max:+.2f}], std={dr_std:.2f}")
    print(f"  Near-collapse epochs: {near_collapse}/{EPOCHS}")
    print(f"  L_lap trajectory: early={lap_early:.4f} -> late={lap_late:.4f}")
    print(f"    L_lap decreasing → G learning to match Laplacian structure")
    print(f"    L_lap flat/rising → G ignoring or unable to optimize this loss")
    print()
    print(f"  vs 09 baseline (FID=49.17):")
    print(f"    09 DR: +0.48, DF: -0.39")
    print(f"    10 DR: {final_dr:+.2f}, DF: {final_df:+.2f}")
    if final_dr > 0.8:
        print(f"    WARN: DR elevated → λ may be too large, G distribution shifting")
    if final_df < -0.8:
        print(f"    WARN: DF dropped → D detecting Laplacian-induced patterns")
    print()

    # Verdict
    if epoch1_d_loss is not None and epoch1_d_loss > 1.95:
        print(f"  [FAIL] Epoch 1 collapse — unexpected with 09 baseline.")
    elif near_collapse > 20:
        print(f"  [FAIL] Frequent collapse — Laplacian destabilized training.")
    elif final_dr > 1.2 or final_df < -1.0:
        print(f"  [WARN] D significantly shifted — λ too large.")
        print(f"         Next: reduce LAP_LAMBDA to 0.01.")
    elif lap_final > lap_early * 0.95:
        print(f"  [WARN] L_lap not decreasing — G ignoring Laplacian term.")
        print(f"         λ may be too small or Laplacian signal too weak.")
    else:
        print(f"  [PASS] Laplacian integrated without destabilization.")
        print(f"         Check FID vs 09 (49.17).")
    print(f"{'='*60}\n")

    # Loss curves (4-panel: G, D, DR/DF, L_lap)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].plot(df_loss["epoch"], df_loss["G_loss"], color="#e74c3c", lw=1.5, label="G_total")
    axes[0].plot(df_loss["epoch"], df_loss["G_adv"], color="#e67e22", lw=1.0, alpha=0.6, label="G_adv")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Generator Loss")
    axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_loss["epoch"], df_loss["D_loss"], color="#3498db", lw=1.5)
    axes[1].axhline(2.0, color="red", ls="--", alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Discriminator Loss (Hinge)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_loss["epoch"], df_loss["D_real"], color="#2ecc71", lw=1.5, label="D(Real)")
    axes[2].plot(df_loss["epoch"], df_loss["D_fake"], color="#e67e22", lw=1.5, label="D(Fake)")
    axes[2].axhline(0.0, color="gray", ls="--", alpha=0.4)
    axes[2].axhline(1.0, color="orange", ls="--", alpha=0.3, label="DR>1.0 warning")
    axes[2].set(xlabel="Epoch", ylabel="Mean Logit", title="D(Real) vs D(Fake)")
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    axes[3].plot(df_loss["epoch"], df_loss["L_lap"], color="#9b59b6", lw=1.5)
    axes[3].set(xlabel="Epoch", ylabel="L_lap", title="Laplacian Pyramid Loss")
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, "loss_curves.png"), dpi=150); plt.close()

    # Metrics
    print(f"{'='*60}\n  Computing Metrics for {EXPERIMENT_NAME}\n{'='*60}")
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_ds = AnimeDataset(image_paths, transform=eval_tf)
    eval_dl = DataLoader(eval_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2)

    print("FID ...")
    try:
        fid = FIDCalculator(DEVICE).compute_fid(G, eval_dl, n=N_FID)
        print(f"  FID: {fid:.2f}")
    except Exception as e:
        fid = -1; print(f"  FID FAILED: {e}"); gc.collect()

    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("LPIPS + Diversity + Laplacian Variance + Edge Density ...")
    lpips_calc = LPIPSCalculator(DEVICE); rb, fb = [], []; generated = 0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE) + 1) / 2.0)
    rb = torch.cat(rb, dim=0)[:500]

    with torch.no_grad():
        while generated < 500:
            bs = min(32, 500 - generated)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fb.append((G(noise) + 1) / 2.0); generated += bs
    fb = torch.cat(fb, dim=0)[:500]

    lpips_scores = []
    for i in range(0, 500, 50):
        lpips_scores.extend(lpips_calc.compute_lpips(
            fb[i:min(i + 50, 500)], rb[i:min(i + 50, 500)]
        ))
    lpips_mean = float(np.mean(lpips_scores))
    div = compute_diversity(G, lpips_calc, ns=300)
    lap = compute_laplacian_variance(fb[:200])
    fake_edge, real_edge, edge_ratio = compute_edge_density(fb[:200], rb[:200])
    print(f"  LPIPS: {lpips_mean:.4f}  Diversity: {div:.4f}  "
          f"LapVar: {lap:.2f}  EdgeRatio: {edge_ratio:.4f}")

    del lpips_calc; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    fid_delta_vs_09 = round(fid - 49.17, 2) if fid > 0 else None

    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "type": "single_variable",
        "base": "09 Widthx3 + 20K (FID=49.17)",
        "change": f"G loss: Hinge + {LAP_LAMBDA} x Laplacian Pyramid L1",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_size": actual_dataset_size,
        "LAP_LAMBDA": LAP_LAMBDA,
        "technique": (
            "10: 09 baseline + Laplacian Pyramid Loss. "
            "Single variable: G loss function augmented with 4-level "
            "Laplacian Pyramid L1 term (weights: 64=0.25, 32=0.5, "
            f"16=1.0, 8=1.0). Lambda={LAP_LAMBDA}. "
            "G architecture (=09=03), D (=09), hyperparams (=09) unchanged. "
            "Laplacian built on RGB pixels (range [-1,1]). "
            "Coarse levels (8x8, 16x16) constrain face structure; "
            "fine levels (32x32, 64x64) constrain edge/texture detail. "
            "L_lap trajectory: early={lap_early:.4f} -> late={lap_late:.4f}."
        ),
        "FID": round(fid, 2),
        "FID_delta_vs_09_49.17": fid_delta_vs_09,
        "LPIPS": round(lpips_mean, 4),
        "Diversity": round(div, 4),
        "Laplacian_Variance": round(lap, 2),
        "Edge_Density_Fake": round(fake_edge, 4),
        "Edge_Density_Real": round(real_edge, 4),
        "Edge_Density_Ratio": round(edge_ratio, 4),
        "final_G_loss": round(fgl, 4),
        "final_G_adv": round(g_adv_v, 4),
        "final_L_lap": round(g_lap_v, 4),
        "L_lap_early_avg": round(lap_early, 4),
        "L_lap_late_avg": round(lap_late, 4),
        "final_D_loss": round(fdl, 4),
        "D_real": round(fdr, 4),
        "D_fake": round(fdf, 4),
        "epoch1_D_loss": round(epoch1_d_loss, 4) if epoch1_d_loss else None,
        "D_real_max": round(dr_max, 2),
        "D_real_std": round(dr_std, 4),
        "epochs_near_collapse": int(near_collapse),
        "G_params": gp,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Complete: {EXPERIMENT_NAME}  FID={fid:.2f}  vs 09: {fid_delta_vs_09}")


if __name__ == "__main__":
    main()
