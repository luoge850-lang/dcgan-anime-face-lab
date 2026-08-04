"""
================================================================================
 G强化实验 07_G_PixelShuffle_NoBN: PixelShuffle + BN-Free 基底验证
================================================================================
 SCIENTIFIC QUESTION:
   将03的ConvTranspose+BN上采样范式替换为PixelShuffle+无归一化范式,
   在batch=32+10K数据下能否稳定收敛？这是BN-free Generator基底的
   独立验证——不涉及Self-Attention, 纯测试上采样+归一化两个变量.

 EXPERIMENTAL ROADMAP (两步验证策略):
   Step 1 (本实验, 07): PixelShuffle + BN-free 基底
     变量: ConvTranspose→PixelShuffle + BN→无归一化
     若成功: 基底确立, 08在此基座上单变量加SA
     若失败: 根据per-stage feature stats + D_real/D_fake轨迹定位根因

   Step 2 (未来08): 07基底 + SAGAN Attention
     变量: 仅加SA (SN on f/g/h, gamma=0 init), 其余=07
     禁止同时引入多个结构变化

 WHY THIS ORDER:
   之前三合一设计(PS+NoBN+SA同时改)若失败无法归因.
   拆分后07最多两个变量构成一个"范式单元", 失败时可通过
   per-stage feature mean/std精确定位是哪一层的信号衰减失控.

 SUCCESS CRITERIA (NOTE: FID≈60 = SUCCESS):
   导师明确指出: "如果最终FID在60左右, 它依然是成功实验,
   因为它证明你打开了一条新的结构路线, 为08 Attention和
   后续扩展提供了基础." 07不要求和03(FID=59)竞争——它的
   价值在于验证BN-free backbone的生存能力(FID<~68不崩溃).
   生成质量的增益来自后续08(SAGAN Attention)和更远的加宽/加深.

   稳定收敛的前提下:
   FID < 55:  BN-free范式超越BN范式, 历史性突破
   FID 55-68: BN-free基底验证通过, 打开新路线 (成功)
   FID 68-80: 训练稳定但容量不足, 需后续加SA或加宽
   D_loss=2.0: 信号饿死/噪声级联, 需LayerNorm或调整激活

 FAILURE MODE DIAGNOSTICS (per-stage feature stats定位):
   若07失败, 通过feature_stats.csv中每层mean/std的epoch变化:
   - Stage1-2 std快速衰减→0: 早期层信号饿死, 需LayerNorm
   - Stage3-5 std持续膨胀: 后期层方差失控, 需residual scaling
   - 所有stage std稳定: BN-free架构本身可行, 问题在其他地方
   - 某特定stage std突变: 该层Conv初始化或PS重排异常

 DIAGNOSTIC DATA COLLECTED:
   - loss.csv: epoch, D_loss, G_loss, D_real, D_fake (per epoch)
   - feature_stats.csv: per-stage mean/std at each epoch (用fixed_noise)
   - fid_progress.csv: FID at SAMPLE_INTERVAL epochs (每50 epoch)
   - 生成样本: epoch_050/100/150/200.png

 DESIGN:
   ┌────────────┬─────────────────────┬──────────────────────┐
   │            │ 范式A (03, BN-based)│ 范式B (07, BN-free)   │
   ├────────────┼─────────────────────┼──────────────────────┤
   │ 上采样     │ ConvTranspose2d x4  │ Conv2d+PixelShuffle x4│
   │ 归一化     │ BatchNorm2d x4      │ 无                    │
   │ 激活       │ ReLU                │ LeakyReLU(0.2)        │
   │ Stage1投影 │ ConvTranspose2d     │ Linear+Reshape        │
   │ G参数      │ 7.77M               │ ~7.26M                │
   │ D架构      │ SN+Hinge            │ SN+Hinge (不变)       │
   └────────────┴─────────────────────┴──────────────────────┘

 ARCHITECTURE (纯PixelShuffle + 零归一化, 无SA):
   Stage 1 [128,1,1 -> 512,4,4]:
     Linear(128, 512x16) -> Reshape -> LeakyReLU(0.2)

   Stage 2 [512,4,4 -> 256,8,8]:
     Conv2d(512, 256x4=1024, k=3, p=1) -> PixelShuffle(2) -> LeakyReLU(0.2)

   Stage 3 [256,8,8 -> 128,16,16]:
     Conv2d(256, 128x4=512, k=3, p=1) -> PixelShuffle(2) -> LeakyReLU(0.2)

   Stage 4 [128,16,16 -> 64,32,32]:
     Conv2d(128, 64x4=256, k=3, p=1) -> PixelShuffle(2) -> LeakyReLU(0.2)

   Stage 5 [64,32,32 -> 3,64,64]:
     Conv2d(64, 3x4=12, k=3, p=1) -> PixelShuffle(2) -> Tanh

   总计: 1 Linear + 4 Conv2d + 4 PixelShuffle + 5 LeakyReLU
         0 BatchNorm, 0 ConvTranspose, 0 Self-Attention

 TRAINING CONFIG (bit-for-bit =03):
   D: SN+Hinge (不变), Batch=32, Data=10K, lr=1e-4,
   Adam(beta1=0.5, beta2=0.99), Epoch=200, Seed=42

 VARIABLES CHANGED vs 03:
   1. ConvTranspose2d -> Conv2d+PixelShuffle (上采样机制)
   2. BatchNorm2d -> 无 (归一化策略)
   3. ReLU -> LeakyReLU(0.2) (激活函数, 因去BN而必要)
   4. Stage1投影: ConvTranspose2d -> Linear+Reshape
   5. 通道: 768->384->192->96 -> 512->256->128->64 (因PS参数膨胀)
   D架构/优化器/超参数: 完全不动 (=03)

 REFERENCE:
   Odena et al.(2016) — Deconvolution and Checkerboard Artifacts (Distill)
   Shi et al.(CVPR 2016) — Real-Time Single Image SR (PixelShuffle)
   Wang et al.(ECCV 2018) — ESRGAN: Enhanced SRGAN (BN-free generator)
   Zhang et al.(ICML 2019) — SAGAN (attention gamma=0 init, SN on G)
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

EXPERIMENT_NAME = "07_G_PixelShuffle_NoBN"
OUTPUT_DIR = "/kaggle/working/dcgan_output"
DATASET_PATH = "/kaggle/input/gananime-lite"
DATASET_LIMIT = 10000
IMAGE_SIZE = 64
BATCH_SIZE = 32
NOISE_DIM = 128
LR = 1e-4
BETAS = (0.5, 0.99)
SEED = 42
EPOCHS = 200
SAMPLE_INTERVAL = 50
N_FID = 10000
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
# [07] Generator: PixelShuffle + BN-Free 基底
#
#   TWO CHANGES from 03:
#     1) Upsampling: ConvTranspose2d(k=4,s=2) -> Conv2d(k=3)+PixelShuffle(2)
#        - 消除checkerboard伪影 (Odena et al. 2016)
#        - 梯度完全均匀: 每个输出像素恰好一个来源
#     2) Normalization: BatchNorm2d -> 无 (零归一化)
#        - LeakyReLU(0.2)替代ReLU, 负半轴保留20%信号缓解衰减
#        - 移除batch=32下BN的统计噪声和样本间特征污染
#
#   NO Self-Attention — 纯基底验证. SA将在08作为单变量加入.
#
#   get_feature_stats(noise): 返回forward过程中每个stage输出(含激活前/后)的mean/std,
#     用于训练中诊断信号衰减/方差漂移的具体发生层.
#
#   CHANNELS: 512->256->128->64->3 (窄于03的768->384->192->96->3)
#     因PixelShuffle要求Conv输出C*r^2通道, 保持G参数~7.26M接近03的7.77M.
# =============================================================================

class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()

        # Stage 1: noise[128] -> 4x4[512]
        self.s1_fc = nn.Linear(nd, 512 * 4 * 4)

        # Stage 2: 4x4[512] -> 8x8[256]
        self.s2_conv = nn.Conv2d(512, 256 * 4, 3, 1, 1)

        # Stage 3: 8x8[256] -> 16x16[128]
        self.s3_conv = nn.Conv2d(256, 128 * 4, 3, 1, 1)

        # Stage 4: 16x16[128] -> 32x32[64]
        self.s4_conv = nn.Conv2d(128, 64 * 4, 3, 1, 1)

        # Stage 5: 32x32[64] -> 64x64[3]
        self.s5_conv = nn.Conv2d(64, 3 * 4, 3, 1, 1)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage 1: [B,128] -> [B,512,4,4]
        B = x.size(0)
        x = self.s1_fc(x.view(B, -1))
        x = x.view(B, 512, 4, 4)
        x = F.leaky_relu(x, 0.2)

        # Stage 2: [B,512,4,4] -> [B,256,8,8]
        x = self.s2_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = F.leaky_relu(x, 0.2)

        # Stage 3: [B,256,8,8] -> [B,128,16,16]
        x = self.s3_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = F.leaky_relu(x, 0.2)

        # Stage 4: [B,128,16,16] -> [B,64,32,32]
        x = self.s4_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = F.leaky_relu(x, 0.2)

        # Stage 5: [B,64,32,32] -> [B,3,64,64]
        x = self.s5_conv(x)
        x = F.pixel_shuffle(x, 2)
        return torch.tanh(x)

    @torch.no_grad()
    def get_feature_stats(self, noise):
        """
        Run forward pass and collect per-stage feature statistics.
        Used for diagnostic tracking during training — reveals which
        layer's signal decays or variance drifts, enabling precise
        failure attribution (normalization vs. upsampling).

        Returns dict with keys:
          s1_mean, s1_std  — after Linear+Reshape+LeakyReLU  [512,4,4]
          s2_mean, s2_std  — after Conv+PS+LeakyReLU          [256,8,8]
          s3_mean, s3_std  — after Conv+PS+LeakyReLU          [128,16,16]
          s4_mean, s4_std  — after Conv+PS+LeakyReLU          [64,32,32]
          s5_mean, s5_std  — after Conv+PS (pre-Tanh)         [3,64,64]
        """
        B = noise.size(0)
        stats = {}

        # Stage 1
        x = self.s1_fc(noise.view(B, -1))
        x = x.view(B, 512, 4, 4)
        x = F.leaky_relu(x, 0.2)
        stats['s1_mean'] = x.mean().item()
        stats['s1_std']  = x.std().item()
        stats['s1_act']  = (x > 0).float().mean().item()  # fraction of positive activations

        # Stage 2
        x = self.s2_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = F.leaky_relu(x, 0.2)
        stats['s2_mean'] = x.mean().item()
        stats['s2_std']  = x.std().item()
        stats['s2_act']  = (x > 0).float().mean().item()

        # Stage 3
        x = self.s3_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = F.leaky_relu(x, 0.2)
        stats['s3_mean'] = x.mean().item()
        stats['s3_std']  = x.std().item()
        stats['s3_act']  = (x > 0).float().mean().item()

        # Stage 4
        x = self.s4_conv(x)
        x = F.pixel_shuffle(x, 2)
        x = F.leaky_relu(x, 0.2)
        stats['s4_mean'] = x.mean().item()
        stats['s4_std']  = x.std().item()
        stats['s4_act']  = (x > 0).float().mean().item()

        # Stage 5 (pre-Tanh, raw RGB logits)
        x = self.s5_conv(x)
        x = F.pixel_shuffle(x, 2)
        stats['s5_mean'] = x.mean().item()
        stats['s5_std']  = x.std().item()
        stats['s5_act']  = (x > 0).float().mean().item()

        return stats

    @torch.no_grad()
    def get_weight_norms(self):
        """
        Return L2 norm of each trainable weight for tracking.
        Useful for detecting exploding/vanishing weights in BN-free training.
        """
        norms = {}
        for name, param in self.named_parameters():
            if 'weight' in name:
                norms[name] = torch.norm(param).item()
        return norms


# =============================================================================
# Discriminator — SN+Hinge, UNCHANGED from 03
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
            noise = torch.randn(bs, NOISE_DIM, device=self.device)
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
        noise = torch.randn(bs, NOISE_DIM, device=DEVICE)
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
    fixed_noise = torch.randn(64, NOISE_DIM, device=DEVICE)

    ds = AnimeDataset(image_paths, transform=get_transform())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    # Verify architecture
    g_tmp = Generator(); d_tmp = Discriminator()
    gp = sum(p.numel() for p in g_tmp.parameters())
    dp = sum(p.numel() for p in d_tmp.parameters())
    bn_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.BatchNorm2d))
    convt_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.ConvTranspose2d))
    conv2d_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.Conv2d))
    linear_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.Linear))

    with torch.no_grad():
        test_out = g_tmp(torch.randn(4, NOISE_DIM))
        test_stats = g_tmp.get_feature_stats(torch.randn(4, NOISE_DIM))
        test_wnorms = g_tmp.get_weight_norms()
    del g_tmp, d_tmp

    steps_per_epoch = len(dl)

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT_NAME}")
    print(f"  07: PixelShuffle + BN-Free Base Verification")
    print(f"  {'─'*50}")
    print(f"  G: {gp:,} params (03: 7,774,947, delta={gp-7774947:+,})")
    print(f"  D: {dp:,} params (=03, SN+Hinge, unchanged)")
    print(f"  ConvTranspose: {convt_count} (03=5)  |  BatchNorm: {bn_count} (03=4)")
    print(f"  Conv2d: {conv2d_count}  |  Linear: {linear_count}")
    print(f"  PixelShuffle: 4 (x4 = upscale 16x)  |  LeakyReLU: 5 (slope=0.2)")
    print(f"  Self-Attention: NONE (reserved for 08)")
    print(f"  Channels: 512->256->128->64->3 (03: 768->384->192->96->3)")
    print(f"  Output: {list(test_out.shape)}  Range: [{test_out.min():.3f}, {test_out.max():.3f}]")
    print(f"  Batch: {BATCH_SIZE} | Data: {actual_dataset_size:,} | Epochs: {EPOCHS}")
    print(f"  Feature stats keys: {list(test_stats.keys())}")
    print(f"{'='*60}\n")

    G = Generator().to(DEVICE); D = Discriminator().to(DEVICE)
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    # === loss.csv: per-epoch training dynamics ===
    loss_f = open(os.path.join(EXP_DIR, "loss.csv"), "w", newline="")
    loss_w = csv.writer(loss_f)
    loss_w.writerow(["epoch", "D_loss", "G_loss", "D_real", "D_fake"])

    # === feature_stats.csv: per-stage activation statistics (diagnostic) ===
    feat_f = open(os.path.join(EXP_DIR, "feature_stats.csv"), "w", newline="")
    feat_w = csv.writer(feat_f)
    feat_cols = ["epoch"]
    for s in range(1, 6):
        feat_cols.extend([f"s{s}_mean", f"s{s}_std", f"s{s}_act"])
    feat_w.writerow(feat_cols)

    # === fid_progress.csv: FID tracked during training ===
    fid_f = open(os.path.join(EXP_DIR, "fid_progress.csv"), "w", newline="")
    fid_w = csv.writer(fid_f)
    fid_w.writerow(["epoch", "FID"])

    # === weight_norms.csv: L2 norm of each trainable weight per epoch ===
    wnorm_f = open(os.path.join(EXP_DIR, "weight_norms.csv"), "w", newline="")
    wnorm_w = csv.writer(wnorm_f)
    # Write header after first epoch to know column names

    fdl, fgl, fdr, fdf = 0.0, 0.0, 0.0, 0.0
    epoch1_d_loss = None

    print("Training ...\n")
    for ep in range(1, EPOCHS + 1):
        for img in dl:
            real = img.to(DEVICE); bs = real.size(0)
            noise = torch.randn(bs, NOISE_DIM, device=DEVICE)
            with torch.no_grad(): fake = G(noise)
            d_real = D(real); d_fake = D(fake)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            noise = torch.randn(bs, NOISE_DIM, device=DEVICE)
            g_loss = -D(G(noise)).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v = d_loss.item(); gl_v = g_loss.item()
        dr_v = d_real.mean().item(); df_v = d_fake.mean().item()
        loss_w.writerow([ep, dl_v, gl_v, dr_v, df_v])
        fdl, fgl, fdr, fdf = dl_v, gl_v, dr_v, df_v

        if ep == 1:
            epoch1_d_loss = dl_v

        # === Per-stage feature statistics (using fixed_noise for comparability) ===
        G.eval()
        feat_stats = G.get_feature_stats(fixed_noise)
        G.train()
        feat_row = [ep]
        for s in range(1, 6):
            feat_row.extend([round(feat_stats[f"s{s}_mean"], 6),
                             round(feat_stats[f"s{s}_std"], 6),
                             round(feat_stats[f"s{s}_act"], 6)])
        feat_w.writerow(feat_row)

        # === Weight norms (every epoch for diagnostic) ===
        wnorms = G.get_weight_norms()
        if ep == 1:
            wnorm_w.writerow(["epoch"] + sorted(wnorms.keys()))
        wnorm_w.writerow([ep] + [round(wnorms[k], 6) for k in sorted(wnorms.keys())])

        # === Periodic FID computation ===
        fid_current = -1
        if ep % SAMPLE_INTERVAL == 0 or ep == 1:
            G.eval()
            samples = G(fixed_noise)
            save_image_grid(samples, os.path.join(EXP_DIR, f"epoch_{ep:03d}.png"))
            torch.save(G.state_dict(), os.path.join(EXP_DIR, f"generator_epoch_{ep:03d}.pth"))

            # FID at this checkpoint
            eval_tf = transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            eval_ds = AnimeDataset(image_paths, transform=eval_tf)
            eval_dl = DataLoader(eval_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2)
            try:
                fid_current = FIDCalculator(DEVICE).compute_fid(G, eval_dl, n=N_FID)
                print(f"  >>> FID@{ep}: {fid_current:.2f}")
            except Exception as e:
                print(f"  >>> FID@{ep}: FAILED ({e})")
                fid_current = -1
            fid_w.writerow([ep, round(fid_current, 2) if fid_current > 0 else -1])
            fid_f.flush()
            G.train()

        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  "
              f"DR:{dr_v:+.2f}  DF:{df_v:+.2f}  "
              f"s2a:{feat_stats['s2_act']:.2f}  s4a:{feat_stats['s4_act']:.2f}")

    loss_f.close()
    feat_f.close()
    wnorm_f.close()

    torch.save(G.state_dict(), os.path.join(EXP_DIR, "generator_final.pth"))
    torch.save(D.state_dict(), os.path.join(EXP_DIR, "discriminator_final.pth"))
    save_image_grid(next(iter(dl))[:64], os.path.join(EXP_DIR, "real_images.png"))

    # =========================================================================
    # FAILURE MODE DIAGNOSTICS
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTICS: {EXPERIMENT_NAME}")
    print(f"{'='*60}")

    df_loss = pd.read_csv(os.path.join(EXP_DIR, "loss.csv"))
    df_feat = pd.read_csv(os.path.join(EXP_DIR, "feature_stats.csv"))
    df_wnorm = pd.read_csv(os.path.join(EXP_DIR, "weight_norms.csv"))

    dr_max = df_loss["D_real"].max()
    dr_min = df_loss["D_real"].min()
    dr_std = df_loss["D_real"].std()

    d_loss_max = df_loss["D_loss"].max()
    d_loss_min = df_loss["D_loss"].min()
    near_collapse = (df_loss["D_loss"] > 1.95).sum()

    final_d_loss = df_loss["D_loss"].iloc[-1]
    final_dr = df_loss["D_real"].iloc[-1]

    # Feature stats trajectory (early vs late)
    s1_std_early = df_feat["s1_std"].iloc[:10].mean()
    s1_std_late  = df_feat["s1_std"].iloc[-10:].mean()
    s3_std_early = df_feat["s3_std"].iloc[:10].mean()
    s3_std_late  = df_feat["s3_std"].iloc[-10:].mean()
    s5_std_early = df_feat["s5_std"].iloc[:10].mean()
    s5_std_late  = df_feat["s5_std"].iloc[-10:].mean()

    s1_std_trend = (s1_std_late - s1_std_early) / max(abs(s1_std_early), 1e-8)
    s3_std_trend = (s3_std_late - s3_std_early) / max(abs(s3_std_early), 1e-8)
    s5_std_trend = (s5_std_late - s5_std_early) / max(abs(s5_std_early), 1e-8)

    # Activation ratio (fraction of positive activations after LeakyReLU)
    s1_act_early = df_feat["s1_act"].iloc[:10].mean()
    s1_act_late  = df_feat["s1_act"].iloc[-10:].mean()
    s3_act_early = df_feat["s3_act"].iloc[:10].mean()
    s3_act_late  = df_feat["s3_act"].iloc[-10:].mean()

    # Weight norm trends (first vs last for key layers)
    wnorm_keys = [c for c in df_wnorm.columns if c != "epoch"]
    wnorm_early = {k: df_wnorm[k].iloc[:5].mean() for k in wnorm_keys}
    wnorm_late  = {k: df_wnorm[k].iloc[-5:].mean() for k in wnorm_keys}
    wnorm_trends = {}
    for k in wnorm_keys:
        if wnorm_early[k] > 1e-8:
            wnorm_trends[k] = (wnorm_late[k] - wnorm_early[k]) / wnorm_early[k]
        else:
            wnorm_trends[k] = 0.0

    print(f"  Epoch 1 D_loss: {epoch1_d_loss:.4f}")
    print(f"    =2.0 -> SIGNAL STARVATION (activation decay too fast w/o norm)")
    print(f"    <1.5 -> signal survives, training viable")
    print()
    print(f"  D_real range: [{dr_min:+.2f}, {dr_max:+.2f}], std={dr_std:.2f}")
    print(f"    DR always<0       -> D suppresses G")
    print(f"    DR always>+2.0    -> D crushes G (output 'too fake')")
    print(f"    DR oscillating>1.5 -> VARIANCE DRIFT (no norm)")
    print(f"    DR in [0,+2.0]    -> HEALTHY")
    print()
    print(f"  D_loss range: [{d_loss_min:.4f}, {d_loss_max:.4f}]")
    print(f"    Epochs near collapse (D_loss>1.95): {near_collapse}/{EPOCHS}")
    if near_collapse > 0:
        first_bad = df_loss[df_loss["D_loss"] > 1.95]["epoch"].min()
        print(f"    First near-collapse at epoch {int(first_bad)}")
    print()
    print(f"  Per-stage feature stats (early -> late):")
    print(f"    Stage1: std {s1_std_early:.4f}->{s1_std_late:.4f} ({s1_std_trend:+.1%})  "
          f"act {s1_act_early:.3f}->{s1_act_late:.3f}")
    print(f"    Stage3: std {s3_std_early:.4f}->{s3_std_late:.4f} ({s3_std_trend:+.1%})  "
          f"act {s3_act_early:.3f}->{s3_act_late:.3f}")
    print(f"    Stage5: std {s5_std_early:.4f}->{s5_std_late:.4f} ({s5_std_trend:+.1%})")

    # Activation ratio diagnostics
    if s1_act_late < 0.3 or s3_act_late < 0.3:
        print(f"    SIGINT: Low activation ratio (<0.3) -> too many dead features")
        print(f"            LeakyReLU(0.2) may be insufficient. Try larger slope or PReLU.")
    if s1_act_late > 0.8:
        print(f"    SIGINT: High activation ratio (>0.8) -> LeakyReLU not filtering")
        print(f"            (expected ~0.5 for zero-mean input with LeakyReLU)")

    # Feature std diagnostics
    if s1_std_trend < -0.5 or s3_std_trend < -0.5:
        print(f"    SIGINT: Feature std collapsing -> signal starvation in early layers")
        print(f"            Fix: add LayerNorm at Stage1-3 positions")
    if s5_std_trend > 1.0:
        print(f"    SIGINT: Feature std exploding -> variance drift in late layers")
        print(f"            Fix: residual scaling x0.2, or InstanceNorm at Stage4-5")

    # Weight norm diagnostics
    print()
    print(f"  Weight norm trends (key layers):")
    max_wnorm_growth = 0.0
    max_wnorm_name = ""
    for k in sorted(wnorm_trends.keys()):
        if abs(wnorm_trends[k]) > 0.3:  # >30% change
            print(f"    {k}: {wnorm_early[k]:.4f} -> {wnorm_late[k]:.4f} ({wnorm_trends[k]:+.1%})")
        if wnorm_trends[k] > max_wnorm_growth:
            max_wnorm_growth = wnorm_trends[k]
            max_wnorm_name = k
    if max_wnorm_growth > 3.0:
        print(f"    WARNING: {max_wnorm_name} weight norm grew {max_wnorm_growth:+.1%} -> possible explosion")
    elif max_wnorm_growth < -0.7:
        print(f"    WARNING: {max_wnorm_name} weight norm shrank {max_wnorm_growth:+.1%} -> possible vanishing")

    # Overall verdict
    print()
    print(f"  VERDICT:")
    if epoch1_d_loss is not None and epoch1_d_loss > 1.95:
        print(f"  [FAIL] Epoch 1 signal starvation -> normalization failure.")
        print(f"         Fix: add LayerNorm/InstanceNorm at key positions.")
    elif near_collapse > 50:
        print(f"  [FAIL] Frequent near-collapse ({near_collapse} epochs) -> training unstable.")
        print(f"         Check feature_stats.csv for the layer where std collapses.")
    elif dr_std > 1.0:
        print(f"  [WARN] High DR variance (std={dr_std:.2f}) -> possible variance drift.")
        print(f"         Compare per-stage std trends in feature_stats.csv.")
    elif final_dr > 2.0:
        print(f"  [WARN] D dominating at end (DR={final_dr:+.2f}).")
        print(f"         Future 08 SA could help G catch up.")
    else:
        print(f"  [PASS] Training stable. BN-free base architecture is VIABLE.")
        print(f"         Ready for 08: +SAGAN Attention (SN on f/g/h, gamma=0).")
    print(f"{'='*60}\n")

    # Loss curves (4-panel: G_loss, D_loss, D_real/D_fake, feature std)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0][0].plot(df_loss["epoch"], df_loss["G_loss"], color="#e74c3c", lw=1.5)
    axes[0][0].set(xlabel="Epoch", ylabel="Loss", title="Generator Loss (Hinge)")
    axes[0][0].grid(True, alpha=0.3)

    axes[0][1].plot(df_loss["epoch"], df_loss["D_loss"], color="#3498db", lw=1.5)
    axes[0][1].axhline(2.0, color="red", ls="--", alpha=0.3, label="Collapse (D_loss=2.0)")
    axes[0][1].set(xlabel="Epoch", ylabel="Loss", title="Discriminator Loss (Hinge)")
    axes[0][1].legend(fontsize=7); axes[0][1].grid(True, alpha=0.3)

    axes[0][2].plot(df_loss["epoch"], df_loss["D_real"], color="#2ecc71", lw=1.5, label="D(Real)")
    axes[0][2].plot(df_loss["epoch"], df_loss["D_fake"], color="#e67e22", lw=1.5, label="D(Fake)")
    axes[0][2].axhline(0.0, color="gray", ls="--", alpha=0.4)
    axes[0][2].set(xlabel="Epoch", ylabel="Mean Logit", title="D(Real) vs D(Fake)")
    axes[0][2].legend(fontsize=8); axes[0][2].grid(True, alpha=0.3)

    # Feature std trajectory (key diagnostic plot)
    for s, color in [(1, "#9b59b6"), (2, "#3498db"), (3, "#2ecc71"), (4, "#e67e22"), (5, "#e74c3c")]:
        axes[1][0].plot(df_feat["epoch"], df_feat[f"s{s}_std"], lw=1.0, color=color, alpha=0.8,
                        label=f"S{s} ({['512@4x4','256@8x8','128@16x16','64@32x32','3@64x64(pre-Tanh)'][s-1]})")
    axes[1][0].set(xlabel="Epoch", ylabel="Feature Std", title="Per-Stage Feature Std (BN-free diagnostic)")
    axes[1][0].legend(fontsize=6, loc="upper right"); axes[1][0].grid(True, alpha=0.3)

    # Feature mean trajectory
    for s, color in [(1, "#9b59b6"), (2, "#3498db"), (3, "#2ecc71"), (4, "#e67e22"), (5, "#e74c3c")]:
        axes[1][1].plot(df_feat["epoch"], df_feat[f"s{s}_mean"], lw=1.0, color=color, alpha=0.8,
                        label=f"S{s}")
    axes[1][1].set(xlabel="Epoch", ylabel="Feature Mean", title="Per-Stage Feature Mean")
    axes[1][1].legend(fontsize=6); axes[1][1].grid(True, alpha=0.3)
    axes[1][1].axhline(0.0, color="gray", ls="--", alpha=0.4)

    # Feature activation ratio (fraction of LeakyReLU outputs > 0)
    for s, color in [(1, "#9b59b6"), (2, "#3498db"), (3, "#2ecc71"), (4, "#e67e22"), (5, "#e74c3c")]:
        axes[1][2].plot(df_feat["epoch"], df_feat[f"s{s}_act"], lw=1.0, color=color, alpha=0.8,
                        label=f"S{s}")
    axes[1][2].axhline(0.5, color="gray", ls="--", alpha=0.4, label="~0.5 (healthy)")
    axes[1][2].axhline(0.2, color="red", ls="--", alpha=0.3, label="<0.3 (dead)")
    axes[1][2].set(xlabel="Epoch", ylabel="Activation Ratio (>0)", title="Per-Stage Activation Ratio (LeakyReLU)")
    axes[1][2].legend(fontsize=5, loc="lower right"); axes[1][2].grid(True, alpha=0.3)

    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, "loss_curves.png"), dpi=150); plt.close()

    # Extra figure: weight norm trends
    fig2, ax2 = plt.subplots(1, 1, figsize=(14, 6))
    for k in wnorm_keys:
        ax2.plot(df_wnorm["epoch"], df_wnorm[k], lw=1.0, alpha=0.7, label=k)
    ax2.set(xlabel="Epoch", ylabel="L2 Norm", title="Weight Norm Trajectory (BN-free diagnostic)")
    ax2.legend(fontsize=5, loc="upper left", ncol=2); ax2.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, "weight_norms.png"), dpi=150); plt.close()

    # Final metrics
    print(f"{'='*60}\n  Computing final metrics for {EXPERIMENT_NAME}\n{'='*60}")
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_ds = AnimeDataset(image_paths, transform=eval_tf)
    eval_dl = DataLoader(eval_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2)

    print("Final FID ...")
    try:
        fid = FIDCalculator(DEVICE).compute_fid(G, eval_dl, n=N_FID)
        print(f"  FID: {fid:.2f}")
    except Exception as e:
        fid = -1; print(f"  FID FAILED: {e}"); gc.collect()

    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("LPIPS + Diversity + Laplacian + Edge Density ...")
    lpips_calc = LPIPSCalculator(DEVICE); rb, fb = [], []; generated = 0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE) + 1) / 2.0)
    rb = torch.cat(rb, dim=0)[:500]

    with torch.no_grad():
        while generated < 500:
            bs = min(32, 500 - generated)
            noise = torch.randn(bs, NOISE_DIM, device=DEVICE)
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

    # Failure mode diagnosis from feature stats
    if epoch1_d_loss is not None and epoch1_d_loss > 1.95:
        failure_mode = "signal_starvation"
    elif near_collapse > 50:
        failure_mode = "frequent_collapse"
    elif s1_std_trend < -0.5 or s3_std_trend < -0.5:
        failure_mode = "feature_std_collapsing"
    elif s5_std_trend > 1.0:
        failure_mode = "feature_std_exploding"
    elif dr_std > 1.0:
        failure_mode = "variance_drift"
    elif final_dr > 2.5:
        failure_mode = "d_dominating"
    else:
        failure_mode = "none_stable"

    # FID progress summary
    df_fid = pd.read_csv(os.path.join(EXP_DIR, "fid_progress.csv"))
    fid_values = {int(row["epoch"]): row["FID"] for _, row in df_fid.iterrows() if row["FID"] > 0}

    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "type": "base_verification",
        "step": "1 of 2: PS+NoBN base. Step 2 (08): +SAGAN Attention (SN on f/g/h, gamma=0)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_size": actual_dataset_size,
        "technique": (
            "07 Base Verification: PixelShuffle + BN-Free architecture. "
            "TWO changes from 03: "
            "(1) Upsampling: ConvTranspose2d(k=4,s=2)->Conv2d(k=3)+PixelShuffle(2). "
            "(2) Normalization: BatchNorm2d->None (zero normalization). "
            "Activation: ReLU->LeakyReLU(0.2) (compensates for no BN signal recovery). "
            "NO Self-Attention (reserved for 08 single-variable addition). "
            f"Channels: 512->256->128->64->3 (03: 768->384->192->96->3). "
            f"G params: {gp:,} (03: 7,774,947, delta={gp-7774947:+,}). "
            f"BN: {bn_count} (ZERO). ConvTranspose: {convt_count} (ZERO). "
            "Diagnostics: per-stage feature mean/std logged every epoch "
            "(feature_stats.csv). FID tracked every {SAMPLE_INTERVAL} epochs "
            "(fid_progress.csv). "
            "SUCCESS IF: training stable + FID < ~68 (FID~60 = success, "
            "proves BN-free backbone viable, opens door for 08 SA)."
        ),
        "FID": round(fid, 2),
        "FID_progress": fid_values,
        "LPIPS": round(lpips_mean, 4),
        "Diversity": round(div, 4),
        "Laplacian_Variance": round(lap, 2),
        "Edge_Density_Fake": round(fake_edge, 4),
        "Edge_Density_Real": round(real_edge, 4),
        "Edge_Density_Ratio": round(edge_ratio, 4),
        "final_G_loss": round(fgl, 4),
        "final_D_loss": round(fdl, 4),
        "D_real": round(fdr, 4),
        "D_fake": round(fdf, 4),
        # Diagnostic
        "epoch1_D_loss": round(epoch1_d_loss, 4) if epoch1_d_loss else None,
        "D_real_max": round(dr_max, 2),
        "D_real_min": round(dr_min, 2),
        "D_real_std": round(dr_std, 4),
        "D_loss_max": round(d_loss_max, 4),
        "epochs_near_collapse": int(near_collapse),
        "failure_mode_diagnosis": failure_mode,
        # Per-stage feature stats summary
        "feature_std_s1_early": round(s1_std_early, 6),
        "feature_std_s1_late": round(s1_std_late, 6),
        "feature_std_s1_trend": round(s1_std_trend, 4),
        "feature_std_s3_early": round(s3_std_early, 6),
        "feature_std_s3_late": round(s3_std_late, 6),
        "feature_std_s3_trend": round(s3_std_trend, 4),
        "feature_std_s5_early": round(s5_std_early, 6),
        "feature_std_s5_late": round(s5_std_late, 6),
        "feature_std_s5_trend": round(s5_std_trend, 4),
        "feature_act_s1_early": round(s1_act_early, 4),
        "feature_act_s1_late": round(s1_act_late, 4),
        "feature_act_s3_early": round(s3_act_early, 4),
        "feature_act_s3_late": round(s3_act_late, 4),
        "weight_norm_max_growth": round(max_wnorm_growth, 4),
        "weight_norm_max_growth_layer": max_wnorm_name,
        "G_architecture": (
            "PixelShuffle+BN-Free Base (NO Self-Attention): "
            "Stage1=Linear(128->8192)->Reshape(512,4,4)->LeakyReLU(0.2). "
            "Stage2=Conv2d(512->1024,k3,p1)->PixelShuffle(2)->LeakyReLU(0.2). "
            "Stage3=Conv2d(256->512,k3,p1)->PixelShuffle(2)->LeakyReLU(0.2). "
            "Stage4=Conv2d(128->256,k3,p1)->PixelShuffle(2)->LeakyReLU(0.2). "
            "Stage5=Conv2d(64->12,k3,p1)->PixelShuffle(2)->Tanh."
        ),
        "G_params": gp,
        "G_params_vs_03": f"{gp-7774947:+,}",
        "BN_count": bn_count,
        "ConvTranspose_count": convt_count,
        "PixelShuffle_stages": 4,
        "channels": "512->256->128->64->3",
        "vs_03_channels": "768->384->192->96->3",
        "steps_per_epoch": steps_per_epoch,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    fid_f.close()

    print(f"\n  Complete: {EXPERIMENT_NAME}  FID={fid:.2f}  Diagnosis: {failure_mode}")


if __name__ == "__main__":
    main()
