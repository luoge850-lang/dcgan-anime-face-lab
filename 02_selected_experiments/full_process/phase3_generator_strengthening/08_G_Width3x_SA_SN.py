"""
================================================================================
 G强化实验 08_G_Width3x_SA_SN: 03 + SAGAN Attention (SN约束)
================================================================================
 SCIENTIFIC QUESTION:
   在03(Width×3, FID=59)的最优基座上, 于16×16分辨率插入带Spectral
   Normalization约束的Self-Attention, 能否让G学习到卷积无法捕获的
   长距离结构依赖(双眼对称、五官布局、脸型), 从而将FID从59降至50-55?

 WHY THIS EXPERIMENT MATTERS (INDUSTRIAL PERSPECTIVE):
   之前03+SA的实验(旧08)崩溃了——f/g/h缺少SN导致softmax退化为one-hot,
   D碾压G(DR=+2.89), 模式坍缩. 但失败的是"无SN的SA", 不是"SA本身".
   SAGAN原论文明确: "We apply spectral normalization to BOTH G and D."
   本实验是第一次在正确约束下测试03+SA组合.

 WHY 03 BACKBONE (NOT BN-FREE 07):
   07证明了BN-free可训练但FID=90(弱于03的59). 在工业目标下,
   应该在已验证最优的基座上叠加改进, 而非在弱基座上追赶.

 PREVIOUS 08 SA FAILURE — ROOT CAUSE & FIX:
   ┌────────────────────┬──────────────────────────────────┐
   │ 旧08 SA (失败)      │ 新08 SA (本实验)                  │
   ├────────────────────┼──────────────────────────────────┤
   │ f/g/h: 无SN        │ f/g/h: spectral_norm()包裹       │
   │ softmax: 无温度缩放 │ softmax(f^T·g / sqrt(C/8))      │
   │ gamma: 0 init ✓    │ gamma: 0 init ✓                 │
   │ 结果: DR=+2.89, 坍缩│ 预期: DR平稳, FID改善             │
   └────────────────────┴──────────────────────────────────┘

 SINGLE VARIABLE vs 03:
   仅增加一个SelfAttention模块(46K参数, G的0.6%).
   不改变: 通道(768→384→192→96→3), BN数量(4), ConvTranspose数量(5),
   D架构(SN+Hinge), 优化器(lr=1e-4, Adam β), batch=32, 数据=10K, epoch=200.

 ARCHITECTURE:
   03:  z → ConvT(128→768,k4) → BN → ReLU  [1→4×4]
        → ConvT(768→384,k4,s2) → BN → ReLU [4→8×8]
        → ConvT(384→192,k4,s2) → BN → ReLU [8→16×16]
        → [SA插入点: 192ch @ 16×16, 256个空间位置]
        → ConvT(192→96,k4,s2) → BN → ReLU  [16→32×32]
        → ConvT(96→3,k4,s2) → Tanh          [32→64×64]

 SELF-ATTENTION MODULE (SAGAN standard):
   输入: x ∈ R^{B×192×16×16}, N = 256 空间位置

   f(x) = SN(Conv1×1(192→24))  → [B, 24, 256]   query投影, C/8=24
   g(x) = SN(Conv1×1(192→24))  → [B, 24, 256]   key投影
   h(x) = SN(Conv1×1(192→192)) → [B, 192, 256]  value投影(保持通道)

   attn = softmax( f^T · g / sqrt(24) )  → [B, 256, 256]
          温度缩放防止点积过大导致softmax饱和(Transformer标准做法)

   out = h × attn^T  → [B, 192, 256] → reshape → [B, 192, 16, 16]

   y = γ * out + x  残差连接, γ初始化为0
     γ=0时: y=x, 训练起点完全等价于03
     训练中γ逐渐增长: 网络先建立局部特征基础, 再引入全局注意力

   SN约束机制:
     每个1×1卷积的权重矩阵被归一化为谱范数≤1
     → ||f(x)||, ||g(x)|| 被约束
     → f^T·g 的内积有界 → softmax不会退化为one-hot
     → 防止08旧版的"假图指纹"问题和D碾压G

 PARAMETERS (SA only):
   f: 192×24 = 4,608
   g: 192×24 = 4,608
   h: 192×192 = 36,864
   γ: 1
   Total: 46,081 (03的0.59%, 几乎可忽略)

 TRAINING DIAGNOSTICS:
   loss.csv:  epoch, D_loss, G_loss, D_real, D_fake, gamma
   采样:      epoch_050/100/150/200.png
   关键观察:  gamma值轨迹(应缓慢增长, 非跳变)
             DR范围(应<+2.0, 旧08曾达+2.89)
             D_loss(应<2.0, 不应出现坍塌)

 EXPECTED RESULT:
   🟢 乐观(35%): FID 48~54. SA弥补了卷积缺失的全局结构,
                 gamma学到有意义的注意力模式. 突破03天花板.

   🟡 中性(45%): FID 54~59. 训练稳定, SA有小幅改善或持平.
                 10K数据可能限制了全局注意力的学习.

   🔴 悲观(20%): FID 59~65 或 DR异常.
                 SA在10K数据上无法学到有意义的注意力,
                 γ全程≈0(SA退化)或γ增长+DR飙升(SA引入噪声).

   和旧08SA的区别性诊断:
     旧08(失败): D碾压, DR=+2.89持续, 生成样本全相同
     新08(成功): DR在[0,+1.5]区间, 生成样本多样化
     若新08也出现DR>2.5: SN约束不够, 考虑增加SN到G的ConvT层

 REFERENCE:
   Zhang et al.(ICML 2019) — SAGAN: Self-Attention GAN
     - Table 1: SA@feat32(对应64×64的16×16)最优, FID 22.96→18.28
     - Sec 3.1: "We apply spectral normalization to both G and D"
   Yu et al.(ICCV 2021) — Dual Contrastive Loss and Attention for GANs
     - G-only SA消融: CelebA 9.84→9.35, AnimalFace 36.55→34.83
     - D-only SA警告: CelebA 9.84→10.49(反而变差)
   Wang et al.(CVPR 2018) — Non-local Neural Networks
     - SA module原始设计: f/g/h 1×1投影 + 残差连接
   Vaswani et al.(NeurIPS 2017) — Scaled Dot-Product Attention
================================================================================
"""
import os, csv, json, random, gc, math
from pathlib import Path; from datetime import datetime
import numpy as np; import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageFilter; import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models; from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader; from scipy import linalg

EXPERIMENT_NAME = "08_G_Width3x_SA_SN"
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
# [08] Self-Attention Module — SAGAN style + SN on ALL projections + temperature
#
#   CRITICAL DIFFERENCES from old 08 SA (which collapsed):
#     OLD: f/g/h WITHOUT SpectralNorm → weight growth → f^T·g explosion
#          → softmax near one-hot → "fake fingerprint" → D crushes G (DR=+2.89)
#     NEW: f/g/h WITH SpectralNorm → Lipschitz ≤ 1 → f^T·g bounded
#          + temperature scaling (÷√(C/8)) → softmax stays smooth
#          + gamma=0 init → training starts =03 (identity), then gradually attends
#
#   SAGAN Eq. 1-4 with Transformer-style scaled dot-product attention:
#     f(x) = SN(W_f * x)    query  [B, C/8, N]
#     g(x) = SN(W_g * x)    key    [B, C/8, N]
#     h(x) = SN(W_h * x)    value  [B, C, N]
#     β_ji = softmax(f(x_i)^T · g(x_j) / √(C/8))
#     o_j  = Σ_i β_ji · h(x_i)
#     y    = γ · o + x       residual, γ init=0
# =============================================================================

class SelfAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        reduced = in_ch // 8                     # 192/8 = 24
        self.scale = reduced ** 0.5               # √24 ≈ 4.899, temperature

        # [KEY] SN on all three 1×1 projections
        # Init Conv2d weights BEFORE wrapping with spectral_norm, avoiding
        # parametrization ordering issues with parent Generator's self.apply()
        fc, gc, hc = nn.Conv2d(in_ch, reduced, 1), nn.Conv2d(in_ch, reduced, 1), nn.Conv2d(in_ch, in_ch, 1)
        for c in (fc, gc, hc):
            nn.init.orthogonal_(c.weight)
            if c.bias is not None: nn.init.constant_(c.bias, 0)
        self.f = nn.utils.spectral_norm(fc)   # query
        self.g = nn.utils.spectral_norm(gc)   # key
        self.h = nn.utils.spectral_norm(hc)   # value

        # gamma init=0: network first relies on local conv features,
        # then gradually learns global attention (SAGAN key design)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W                                # 16×16 = 256

        # Project to query/key/value, reshape to [B, ch, N]
        f = self.f(x).view(B, -1, N)             # [B, 24, 256]
        g = self.g(x).view(B, -1, N)             # [B, 24, 256]
        h = self.h(x).view(B, -1, N)             # [B, 192, 256]

        # Scaled dot-product attention
        # SN bounds weight spectra → f^T·g bounded; /scale prevents saturation
        attn = torch.bmm(f.permute(0, 2, 1), g)   # [B, 256, 256]
        attn = F.softmax(attn / self.scale, dim=-1)

        # Apply attention to value features
        out = torch.bmm(h, attn.permute(0, 2, 1)) # [B, 192, 256]
        out = out.view(B, C, H, W)

        # Residual: gamma starts at 0 → SA is identity at initialization
        return self.gamma * out + x


# =============================================================================
# [08] Generator — 03 Width×3 + Self-Attention @16×16
#
#   Unpacked from 03's nn.Sequential to allow SA insertion.
#   Stage1-3, Stage4-5: bit-for-bit identical to 03.
#   Only addition: SA module between Stage3 output and Stage4 input.
#
#   03: z → S1→S2→S3→S4→S5 → RGB
#   08: z → S1→S2→S3→[SA]→S4→S5 → RGB
#                     ↑ 唯一新增: 46K参数, 0.6% of G
# =============================================================================

class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()

        # Stage 1: noise[128] → 4×4[768]  (=03, bit-for-bit)
        self.s1_conv = nn.ConvTranspose2d(nd, 768, 4)
        self.s1_bn = nn.BatchNorm2d(768)

        # Stage 2: 4×4[768] → 8×8[384]  (=03)
        self.s2_conv = nn.ConvTranspose2d(768, 384, 4, 2, 1)
        self.s2_bn = nn.BatchNorm2d(384)

        # Stage 3: 8×8[384] → 16×16[192]  (=03)
        self.s3_conv = nn.ConvTranspose2d(384, 192, 4, 2, 1)
        self.s3_bn = nn.BatchNorm2d(192)

        # [NEW] Self-Attention @16×16, 192 channels
        self.sa = SelfAttention(192)

        # Stage 4: 16×16[192] → 32×32[96]  (=03)
        self.s4_conv = nn.ConvTranspose2d(192, 96, 4, 2, 1)
        self.s4_bn = nn.BatchNorm2d(96)

        # Stage 5: 32×32[96] → 64×64[3]  (=03)
        self.s5_conv = nn.ConvTranspose2d(96, 3, 4, 2, 1)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage 1: 1×1 → 4×4  (=03)
        x = self.s1_conv(x)
        x = F.relu(self.s1_bn(x))

        # Stage 2: 4×4 → 8×8  (=03)
        x = self.s2_conv(x)
        x = F.relu(self.s2_bn(x))

        # Stage 3: 8×8 → 16×16  (=03)
        x = self.s3_conv(x)
        x = F.relu(self.s3_bn(x))

        # [NEW] Self-Attention @16×16
        x = self.sa(x)

        # Stage 4: 16×16 → 32×32  (=03)
        x = self.s4_conv(x)
        x = F.relu(self.s4_bn(x))

        # Stage 5: 32×32 → 64×64  (=03)
        x = self.s5_conv(x)
        return torch.tanh(x)


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
    # SN count: spectral_norm() is a function, not a class, so isinstance() fails.
    # Count by checking for _orig weight params created by SN parametrization.
    sn_count = sum(1 for n, _ in g_tmp.named_parameters()
                   if '_param_original' in n or n.endswith('_orig'))
    convt_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.ConvTranspose2d))
    conv2d_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.Conv2d))
    sa_count = sum(1 for m in g_tmp.modules() if isinstance(m, SelfAttention))
    sa_params = sum(p.numel() for m in g_tmp.modules()
                   if isinstance(m, SelfAttention) for p in m.parameters())

    with torch.no_grad():
        test_out = g_tmp(torch.randn(4, NOISE_DIM, 1, 1))
    del g_tmp, d_tmp

    steps_per_epoch = len(dl)

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT_NAME}")
    print(f"  08: 03 Width×3 + Self-Attention @16×16 (SN-constrained)")
    print(f"  {'─'*50}")
    print(f"  G: {gp:,} params (03: 7,774,947, +{gp-7774947:,})")
    print(f"    SA portion: {sa_params:,} ({100*sa_params/gp:.2f}% of G)")
    print(f"  D: {dp:,} params (=03, SN+Hinge, unchanged)")
    print(f"  ConvTranspose: {convt_count} (=03, 5 stages)")
    print(f"  Conv2d (SA only): {conv2d_count} (f/g/h 1×1 projections)")
    print(f"  BN: {bn_count} (=03, unchanged)")
    print(f"  SN in G: {sn_count} wrappers (f/g/h, fixes old 08 collapse)")
    print(f"  SA modules: {sa_count} (@16×16, 192ch, G-only)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  vs old 08 SA: f/g/h NOW with spectral_norm + temperature scaling")
    print(f"  Old 08 collapsed (DR=+2.89). This is the FIXED version.")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Channels: 768→384→192→96→3 (=03, unchanged)")
    print(f"  Output: {list(test_out.shape)} (expected [4,3,64,64])")
    print(f"  Range:  [{test_out.min():.3f}, {test_out.max():.3f}]")
    print(f"  Batch: {BATCH_SIZE} | Data: {actual_dataset_size:,} | Epochs: {EPOCHS}")
    print(f"{'='*60}\n")

    G = Generator().to(DEVICE); D = Discriminator().to(DEVICE)
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    csv_f = open(os.path.join(EXP_DIR, "loss.csv"), "w", newline="")
    w = csv.writer(csv_f)
    w.writerow(["epoch", "D_loss", "G_loss", "D_real", "D_fake", "gamma"])
    fdl, fgl, fdr, fdf = 0.0, 0.0, 0.0, 0.0

    epoch1_d_loss = None
    gamma_history = []

    print("Training ...\n")
    for ep in range(1, EPOCHS + 1):
        for img in dl:
            real = img.to(DEVICE); bs = real.size(0)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            with torch.no_grad(): fake = G(noise)
            d_real = D(real); d_fake = D(fake)
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            g_loss = -D(G(noise)).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v = d_loss.item(); gl_v = g_loss.item()
        dr_v = d_real.mean().item(); df_v = d_fake.mean().item()
        gamma_v = G.sa.gamma.item()
        w.writerow([ep, dl_v, gl_v, dr_v, df_v, gamma_v])
        fdl, fgl, fdr, fdf = dl_v, gl_v, dr_v, df_v
        gamma_history.append(gamma_v)

        if ep == 1:
            epoch1_d_loss = dl_v

        # Gamma warning (early spike = SA learning too fast)
        gamma_warn = ""
        if gamma_v > 0.3 and ep < 30:
            gamma_warn = " [GAMMA SPIKE!]"
        elif gamma_v > 0.6:
            gamma_warn = " [GAMMA HIGH]"

        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  "
              f"DR:{dr_v:+.2f}  DF:{df_v:+.2f}  gamma:{gamma_v:+.4f}{gamma_warn}")

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
    d_loss_min = df_loss["D_loss"].min()
    near_collapse = (df_loss["D_loss"] > 1.95).sum()

    gamma_max = df_loss["gamma"].max()
    gamma_final = df_loss["gamma"].iloc[-1]

    final_d_loss = df_loss["D_loss"].iloc[-1]
    final_dr = df_loss["D_real"].iloc[-1]

    print(f"  Epoch 1 D_loss: {epoch1_d_loss:.4f}")
    print(f"    =2.0 → SIGNAL STARVATION (unlikely with BN×4=03)")
    print(f"    <1.5 → normal 03-like start (gamma=0 = identity)")
    print()
    print(f"  D_real range: [{dr_min:+.2f}, {dr_max:+.2f}], std={dr_std:.2f}")
    print(f"    DR>+2.0 sustained → D CRUSHING (old 08 failure mode, DR=+2.89)")
    print(f"    DR in [0,+1.5]    → HEALTHY (03-like equilibrium)")
    print()
    print(f"  D_loss range: [{d_loss_min:.4f}, {d_loss_max:.4f}]")
    print(f"    Epochs near collapse (D_loss>1.95): {near_collapse}/{EPOCHS}")
    print()
    print(f"  Gamma trajectory:")
    print(f"    Max: {gamma_max:.4f}  Final: {gamma_final:.4f}")
    print(f"    gamma≈0 at end → SA never activated (10K data insufficient?)")
    print(f"    gamma>1.0        → SA output dominates residual (attention overfitting)")
    print(f"    gamma 0.1~0.8    → SA learning meaningful global attention")

    # Gamma growth rate in first 50 epochs
    if len(gamma_history) >= 50:
        gamma_early_growth = (gamma_history[49] - gamma_history[0]) / 50
        print(f"    Early growth rate: {gamma_early_growth:+.5f}/epoch")
        if gamma_early_growth > 0.01:
            print(f"    WARNING: Gamma growing fast — attention may destabilize.")

    # Compare with old 08 SA failure signature
    print()
    print(f"  vs old 08 SA (collapsed, no SN):")
    print(f"    Old 08: DR max=+2.89, mode collapse at epoch ~100")
    print(f"    New 08: DR max={dr_max:+.2f}")
    if dr_max < 2.0:
        print(f"    [OK] SN constraint is working — D not crushing G")

    # Verdict
    print()
    print(f"  VERDICT:")
    if epoch1_d_loss is not None and epoch1_d_loss > 1.95:
        print(f"  [FAIL] Epoch 1 collapse — unexpected with BN×4 baseline.")
    elif near_collapse > 20:
        print(f"  [FAIL] Frequent collapse ({near_collapse} epochs) — training unstable.")
    elif dr_max > 2.5:
        print(f"  [FAIL] D crushing G (DR={dr_max:+.2f}) — SN on f/g/h insufficient?")
    elif gamma_final < 0.001:
        print(f"  [NEUTRAL] SA never activated (gamma≈0). 10K data may be insufficient.")
        print(f"            FID should still ≈59 (same as 03) — SA did no harm.")
    elif gamma_final > 0.8 and dr_max > 2.0:
        print(f"  [WARN] SA activated but D reacting aggressively.")
        print(f"         SA may be learning useful attention, monitor sample quality.")
    else:
        print(f"  [PASS] SA integrated successfully. Check FID vs 03 (59).")
    print(f"{'='*60}\n")

    # Loss curves (4-panel: G_loss, D_loss, DR/DF, gamma)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].plot(df_loss["epoch"], df_loss["G_loss"], color="#e74c3c", lw=1.5)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Generator Loss (Hinge)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df_loss["epoch"], df_loss["D_loss"], color="#3498db", lw=1.5)
    axes[1].axhline(2.0, color="red", ls="--", alpha=0.3, label="Collapse (D_loss=2.0)")
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Discriminator Loss (Hinge)")
    axes[1].legend(fontsize=7); axes[1].grid(True, alpha=0.3)

    axes[2].plot(df_loss["epoch"], df_loss["D_real"], color="#2ecc71", lw=1.5, label="D(Real)")
    axes[2].plot(df_loss["epoch"], df_loss["D_fake"], color="#e67e22", lw=1.5, label="D(Fake)")
    axes[2].axhline(0.0, color="gray", ls="--", alpha=0.4)
    axes[2].axhline(2.0, color="red", ls="--", alpha=0.3, label="D crushing (>2.0)")
    axes[2].set(xlabel="Epoch", ylabel="Mean Logit", title="D(Real) vs D(Fake)")
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)

    axes[3].plot(df_loss["epoch"], df_loss["gamma"], color="#9b59b6", lw=2.0)
    axes[3].axhline(0.0, color="gray", ls="--", alpha=0.4)
    axes[3].set(xlabel="Epoch", ylabel="γ", title="SA Gamma (learnable, init=0)")
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

    print("LPIPS + Diversity + Laplacian + Edge Density ...")
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

    # Compare with 03 baseline
    fid_delta_vs_03 = round(fid - 59.0, 2) if fid > 0 else None

    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "type": "single_variable",
        "base": "03 Width×3 (FID=59.0)",
        "change": "+Self-Attention @16×16 (192ch, SN on f/g/h, temperature scaling, gamma=0 init)",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "dataset_size": actual_dataset_size,
        "technique": (
            "08: 03 Width×3 backbone + SAGAN Self-Attention @16×16 (192ch). "
            "Single variable: only SA module added (46,081 params, 0.59% of G). "
            "SA: f/g/h 1×1 conv projections with spectral_norm (CRITICAL: old 08 "
            "collapsed without SN — f^T·g exploded → softmax one-hot → D crushed G). "
            "Temperature scaling: softmax(attn / sqrt(C/8)). "
            "Gamma init=0 → training starts identical to 03. "
            "G-only SA (no SA in D, per Yu et al. ICCV 2021). "
            "All ConvTranspose/BN/channels/D/hyperparams =03 unchanged."
        ),
        "FID": round(fid, 2),
        "FID_delta_vs_03": fid_delta_vs_03,
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
        "gamma_initial": 0.0,
        "gamma_final": round(gamma_final, 6),
        "gamma_max": round(gamma_max, 6),
        "epoch1_D_loss": round(epoch1_d_loss, 4) if epoch1_d_loss else None,
        "D_real_max": round(dr_max, 2),
        "D_real_min": round(dr_min, 2),
        "D_real_std": round(dr_std, 4),
        "epochs_near_collapse": int(near_collapse),
        "vs_old_08_SA": (
            "Old 08 (no SN): DR max +2.89, mode collapse epoch ~100. "
            "New 08 (SN): f/g/h spectral_norm constrained. "
            f"DR max {dr_max:+.2f}. "
            "Root cause of old failure: unconstrained 1×1 conv weights → "
            "f^T·g explosion → softmax near one-hot → 'fake fingerprint' → D crush."
        ),
        "G_architecture": (
            "03 Width×3 + SA@16×16: "
            "Stage1=ConvT(128→768,k4)→BN→ReLU. "
            "Stage2=ConvT(768→384,k4,s2)→BN→ReLU. "
            "Stage3=ConvT(384→192,k4,s2)→BN→ReLU. "
            "SA(192ch, C/8=24, SN on f/g/h, temp÷√24, gamma=0). "
            "Stage4=ConvT(192→96,k4,s2)→BN→ReLU. "
            "Stage5=ConvT(96→3,k4,s2)→Tanh."
        ),
        "G_params": gp,
        "G_params_vs_03": f"+{gp-7774947}",
        "SA_params": sa_params,
        "SA_pct_of_G": round(100 * sa_params / gp, 2),
        "BN_count": bn_count,
        "SN_in_G_count": sn_count,
        "ConvTranspose_count": convt_count,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Complete: {EXPERIMENT_NAME}  FID={fid:.2f}  vs 03: {fid_delta_vs_03}")


if __name__ == "__main__":
    main()
