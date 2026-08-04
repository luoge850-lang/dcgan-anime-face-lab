"""
================================================================================
 G强化实验 11: 09 + DiffAugment + EMA — 20K候选稳定基线
================================================================================
 DESIGN:
   Keep Exp09's proven architecture, dataset, optimizer, batch size and
   200-epoch schedule unchanged. Add only two low-risk training/evaluation
   techniques that have direct GAN evidence:

 COMPONENT 1: DiffAugment on D inputs (Zhao et al., NeurIPS 2020)
   policy=color,translation,cutout. The SAME policy is applied to real and
   fake images. During the G update it remains differentiable, so gradients
   pass through the augmentation into G. This reduces D memorization of the
   20K training set and encourages more stable, useful gradients.

 COMPONENT 2: EMA on Generator (Yazici et al., ICLR 2019)
   decay=0.9999, effective window ≈ 10K steps ≈ 16 epochs.
   Tracks BOTH parameters AND buffers (BN running_mean/var).
   Reduces GAN oscillation noise by averaging weights along the training
   trajectory. Used for ALL evaluation and final sampling.
   ★ generator_ema_final.pth is what CLIP fine-tuning loads.

 WHY NOT:
   - R1 removed: Exp05 already tested R1 on the same Width3x family and
     FID regressed 59.00 -> 59.57. Stacking R1 with DiffAugment risks
     over-regularizing an already balanced discriminator.
   - D noise removed: it can weaken local-detail discrimination.
   - No SA (fingerprint, Exp 08)
   - No residual/deep/Laplacian/BN-free (all previously falsified)

 SINGLE VARIABLE vs 09:
   Training change: differentiable augmentation at the D boundary.
   Evaluation/deployment change: EMA weights.
   Unchanged: G architecture, G loss, D architecture, data, hyperparams.

 IMPORTANT:
   An FID improvement is a hypothesis, not a promise. The script saves raw,
   EMA and a full resume checkpoint so the result can be audited and used
   as the starting point for later CLIP fine-tuning.
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

EXPERIMENT_NAME = "11_G_DiffAug_EMA_20K"
OUTPUT_DIR = "/kaggle/working/dcgan_output"
DATASET_PATH = "/kaggle/input/gananime-lite"
DATASET_LIMIT = 20000
IMAGE_SIZE = 64; BATCH_SIZE = 32; NOISE_DIM = 128
LR = 1e-4; BETAS = (0.5, 0.99); SEED = 42
EPOCHS = 200; SAMPLE_INTERVAL = 50; N_FID = 10000

# [NEW] Differentiable augmentation at the discriminator boundary.
# Color jitter is global/per-image, so D still sees and penalizes local color
# blocks. Translation and cutout reduce memorization of exact training images.
DIFFAUG_POLICY = "color,translation,cutout"
DIFFAUG_TRANSLATION_RATIO = 0.125
DIFFAUG_CUTOUT_RATIO = 0.35

# [NEW] EMA on Generator weights
EMA_DECAY = 0.9999          # Effective window ≈ 10K steps ≈ 16 epochs at 20K data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

# Evaluation models are loaded once, only when metrics begin. This removes the
# former five-retry startup loop. On Kaggle, enable Internet once or attach the
# torchvision weight files as a Dataset so they resolve from cache.

def set_all_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def find_all_images(root_dir):
    exts = {".png", ".jpg", ".jpeg"}; files = []
    if not os.path.exists(root_dir): return files
    for dp, _, fn in os.walk(root_dir):
        for f in fn:
            if Path(f).suffix.lower() in exts: files.append(os.path.join(dp, f))
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
    def __init__(self, paths, transform=None): self.paths, self.tf = paths, transform
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
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.RandomHorizontalFlip(p=0.5),
        EdgeSharpen(prob=0.2), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


# =============================================================================
# DiffAugment — adapted from the official MIT Han Lab implementation.
# Anime-specific conservative changes: cutout=0.35 and replicated translation
# borders instead of a large zero-padded region.
# Augmentations operate on batched tensors and preserve gradients to G.
# =============================================================================
def _rand_brightness(x):
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) - 0.5)


def _rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0
    return (x - x_mean) * scale + x_mean


def _rand_contrast(x):
    x_mean = x.mean(dim=(1, 2, 3), keepdim=True)
    scale = torch.rand(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype) + 0.5
    return (x - x_mean) * scale + x_mean


def _rand_translation(x, ratio=DIFFAUG_TRANSLATION_RATIO):
    shift_x = int(x.size(2) * ratio + 0.5)
    shift_y = int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(
        -shift_x, shift_x + 1, (x.size(0), 1, 1), device=x.device
    )
    translation_y = torch.randint(
        -shift_y, shift_y + 1, (x.size(0), 1, 1), device=x.device
    )
    batch_idx = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(2), device=x.device),
        torch.arange(x.size(3), device=x.device),
        indexing="ij",
    )
    # Replicated border padding avoids the black bands created by zero padding.
    # Both real and fake still receive the identical augmentation distribution.
    grid_x = grid_x.unsqueeze(0) + translation_x + shift_x
    grid_y = grid_y.unsqueeze(0) + translation_y + shift_y
    x_pad = F.pad(x, (shift_y, shift_y, shift_x, shift_x), mode="replicate")
    return x_pad.permute(0, 2, 3, 1)[batch_idx, grid_x, grid_y].permute(0, 3, 1, 2)


def _rand_cutout(x, ratio=DIFFAUG_CUTOUT_RATIO):
    cutout_h = int(x.size(2) * ratio + 0.5)
    cutout_w = int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(
        0, x.size(2) + (1 - cutout_h % 2), (x.size(0), 1, 1), device=x.device
    )
    offset_y = torch.randint(
        0, x.size(3) + (1 - cutout_w % 2), (x.size(0), 1, 1), device=x.device
    )
    grid_x, grid_y = torch.meshgrid(
        torch.arange(cutout_h, device=x.device),
        torch.arange(cutout_w, device=x.device),
        indexing="ij",
    )
    grid_x = torch.clamp(
        grid_x.unsqueeze(0) + offset_x - cutout_h // 2, 0, x.size(2) - 1
    )
    grid_y = torch.clamp(
        grid_y.unsqueeze(0) + offset_y - cutout_w // 2, 0, x.size(3) - 1
    )
    mask = torch.ones(
        x.size(0), x.size(2), x.size(3), device=x.device, dtype=x.dtype
    )
    batch_idx = torch.arange(x.size(0), device=x.device).view(-1, 1, 1)
    mask[batch_idx, grid_x, grid_y] = 0
    return x * mask.unsqueeze(1)


def diff_augment(x, policy=DIFFAUG_POLICY):
    for item in [p.strip() for p in policy.split(",") if p.strip()]:
        if item == "color":
            x = _rand_brightness(x)
            x = _rand_saturation(x)
            x = _rand_contrast(x)
        elif item == "translation":
            x = _rand_translation(x)
        elif item == "cutout":
            x = _rand_cutout(x)
        else:
            raise ValueError(f"Unknown DiffAugment policy item: {item}")
    return x.contiguous()


# =============================================================================
# Generator — bit-for-bit =09 (UNCHANGED)
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
        ); self.apply(self._init)
    def _init(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)

# =============================================================================
# Discriminator — architecture =09 (SN unchanged), training modified
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


# =============================================================================
# EMA — Exponential Moving Average of Generator weights
# Yazıcı et al., ICLR 2019: "The Unusual Effectiveness of Averaging in GAN"
# StyleGAN/BigGAN/ProGAN all use EMA as default for final sampling.
# Does NOT affect training — only used for evaluation and final saving.
# =============================================================================
class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        # Track parameters and every buffer. Floating tensors are averaged;
        # integer buffers such as num_batches_tracked copy the latest value.
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().detach()
        for name, buf in model.named_buffers():
            self.shadow[name] = buf.data.clone().detach()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)
        for name, buf in model.named_buffers():
            if torch.is_floating_point(buf):
                self.shadow[name].mul_(self.decay).add_(buf.data, alpha=1 - self.decay)
            else:
                self.shadow[name].copy_(buf.data)

    @torch.no_grad()
    def apply_to(self, model):
        """Copy EMA weights and all buffers into model for evaluation/export."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])
        for name, buf in model.named_buffers():
            buf.data.copy_(self.shadow[name])

    @torch.no_grad()
    def model_state_dict(self, model):
        """Return a directly loadable Generator state_dict containing EMA."""
        result = {}
        for name, tensor in model.state_dict().items():
            result[name] = self.shadow.get(name, tensor).detach().cpu().clone()
        return result

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


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
        imgs = (imgs - 0.5) / 0.5; return self.inc(imgs).cpu().numpy()
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


def laplacian_variances(imgs):
    vars_ = []
    for img in imgs:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); vars_.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return np.asarray(vars_, dtype=np.float64)


def compute_laplacian_variance(imgs):
    return float(laplacian_variances(imgs).mean())


def compute_edge_density(imgs, real_imgs=None):
    densities = []
    for img in imgs:
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); edges = cv2.Canny(gray, 50, 150); densities.append((edges > 0).mean())
    fake_density = float(np.mean(densities))
    if real_imgs is not None:
        real_densities = []
        for img in real_imgs:
            arr = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY); edges = cv2.Canny(gray, 50, 150); real_densities.append((edges > 0).mean())
        real_density = float(np.mean(real_densities)); ratio = fake_density / max(real_density, 1e-8)
        return fake_density, real_density, ratio
    return fake_density, None, None


def main():
    set_all_seeds(SEED); os.makedirs(EXP_DIR, exist_ok=True)

    dataset_path, image_paths = load_dataset()
    if DATASET_LIMIT and len(image_paths) > DATASET_LIMIT:
        set_all_seeds(SEED); image_paths = random.sample(image_paths, DATASET_LIMIT)
        print(f"Subsampled to {DATASET_LIMIT} images (seed={SEED})")
    actual_dataset_size = len(image_paths)

    set_all_seeds(SEED); fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)
    ds = AnimeDataset(image_paths, transform=get_transform())
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=2)

    g_tmp, d_tmp = Generator(), Discriminator()
    gp = sum(p.numel() for p in g_tmp.parameters()); dp = sum(p.numel() for p in d_tmp.parameters())
    bn_count = sum(1 for m in g_tmp.modules() if isinstance(m, nn.BatchNorm2d))
    del g_tmp, d_tmp
    steps_per_epoch = len(dl)
    total_steps = steps_per_epoch * EPOCHS

    print(f"\n{'='*60}")
    print(f"  {EXPERIMENT_NAME}")
    print(f"  09 + DiffAugment ({DIFFAUG_POLICY}) + EMA")
    print(f"  {'─'*50}")
    print(f"  G: {gp:,} params (=09, UNCHANGED)")
    print(f"  D: {dp:,} params (=09, UNCHANGED architecture)")
    print(f"  BN: {bn_count} | Data: {actual_dataset_size:,} | Epochs: {EPOCHS}")
    print(f"  DiffAugment: {DIFFAUG_POLICY} (real + fake + G path)")
    print(f"  EMA: decay={EMA_DECAY}, used for all eval and sampling")
    print(f"  Baseline 09: FID=49.17, LapVar=10833")
    print(f"{'='*60}\n")

    G = Generator().to(DEVICE); D = Discriminator().to(DEVICE)
    ema = EMA(G, decay=EMA_DECAY)
    g_opt = torch.optim.Adam(G.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(D.parameters(), lr=LR, betas=BETAS)

    # Save training config
    config = {
        "experiment": EXPERIMENT_NAME, "base": "09 (768ch, 20K, Hinge)",
        "components": [f"DiffAugment ({DIFFAUG_POLICY}) on real/fake/G path",
                        f"EMA on G (decay={EMA_DECAY})"],
        "seed": SEED, "epochs": EPOCHS, "batch_size": BATCH_SIZE,
        "dataset_limit": DATASET_LIMIT, "noise_dim": NOISE_DIM,
        "lr": LR, "betas": BETAS, "image_size": IMAGE_SIZE,
        "diffaugment_translation_ratio": DIFFAUG_TRANSLATION_RATIO,
        "diffaugment_cutout_ratio": DIFFAUG_CUTOUT_RATIO,
        "diffaugment_translation_padding": "replicate",
        "clip_handoff": {
            "generator": "generator_ema_final.pth (direct Generator state_dict)",
            "discriminator": "discriminator_final.pth",
            "full_resume": "checkpoint_final.pth"
        }
    }
    with open(os.path.join(EXP_DIR, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(EXP_DIR, "dataset_manifest.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(image_paths))

    csv_f = open(os.path.join(EXP_DIR, "loss.csv"), "w", newline="")
    w = csv.writer(csv_f)
    w.writerow([
        "epoch", "D_loss", "G_loss",
        "D_real_aug", "D_fake_aug", "D_real_raw", "D_fake_raw"
    ])
    fdl, fgl, fdr, fdf, fdrr, fdfr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    print("Training ...\n")
    for ep in range(1, EPOCHS + 1):
        for img in dl:
            real = img.to(DEVICE); bs = real.size(0)

            # === D update: 09 Hinge loss at a differentiable augmentation boundary ===
            d_opt.zero_grad()

            noise_z = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            with torch.no_grad(): fake = G(noise_z)

            d_real = D(diff_augment(real))
            d_fake = D(diff_augment(fake))
            d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
            d_loss.backward()
            d_opt.step()

            # === G update: same 09 Hinge objective; DiffAugment keeps gradients ===
            # D gradients are not needed here; disabling them preserves the exact
            # G gradient while reducing memory and compute.
            D.requires_grad_(False)
            noise_z = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake_for_g = G(noise_z)
            g_loss = -D(diff_augment(fake_for_g)).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
            D.requires_grad_(True)
            ema.update(G)  # EMA: shadow weights = 0.9999*shadow + 0.0001*current

        dl_v = d_loss.item(); gl_v = g_loss.item()
        dr_v = d_real.mean().item(); df_v = d_fake.mean().item()
        # Raw logits are diagnostics only. Eval mode prevents this extra pass
        # from updating spectral-normalization buffers.
        D.eval()
        with torch.no_grad():
            dr_raw_v = D(real).mean().item()
            df_raw_v = D(fake).mean().item()
        D.train()
        w.writerow([ep, dl_v, gl_v, dr_v, df_v, dr_raw_v, df_raw_v])
        fdl, fgl, fdr, fdf, fdrr, fdfr = (
            dl_v, gl_v, dr_v, df_v, dr_raw_v, df_raw_v
        )

        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  "
              f"DR/DF_aug:{dr_v:+.2f}/{df_v:+.2f}  "
              f"DR/DF_raw:{dr_raw_v:+.2f}/{df_raw_v:+.2f}")

        if ep % SAMPLE_INTERVAL == 0:
            # Use EMA weights for sampling (restore training weights after)
            train_sd = {k: v.clone() for k, v in G.state_dict().items()}
            ema.apply_to(G)
            G.eval(); samples = G(fixed_noise); G.train()
            save_image_grid(samples, os.path.join(EXP_DIR, f"epoch_{ep:03d}.png"))
            G.load_state_dict(train_sd); del train_sd  # restore training weights
            torch.save(G.state_dict(), os.path.join(EXP_DIR, f"generator_epoch_{ep:03d}.pth"))
            torch.save(
                ema.model_state_dict(G),
                os.path.join(EXP_DIR, f"generator_ema_epoch_{ep:03d}.pth"),
            )

    csv_f.close()
    # Save raw training weights (last step — may have oscillation noise)
    torch.save(G.state_dict(), os.path.join(EXP_DIR, "generator_raw_final.pth"))
    torch.save(D.state_dict(), os.path.join(EXP_DIR, "discriminator_final.pth"))
    # Directly loadable deployment/CLIP weights plus the EMA tracker state.
    torch.save(ema.model_state_dict(G), os.path.join(EXP_DIR, "generator_ema_final.pth"))
    torch.save(ema.state_dict(), os.path.join(EXP_DIR, "ema_state_final.pth"))
    # Save optimizer states for potential resumption
    torch.save(g_opt.state_dict(), os.path.join(EXP_DIR, "optimizer_G.pth"))
    torch.save(d_opt.state_dict(), os.path.join(EXP_DIR, "optimizer_D.pth"))
    torch.save(
        {
            "epoch": EPOCHS,
            "global_step": total_steps,
            "generator": G.state_dict(),
            "discriminator": D.state_dict(),
            "generator_ema": ema.model_state_dict(G),
            "ema": ema.state_dict(),
            "optimizer_G": g_opt.state_dict(),
            "optimizer_D": d_opt.state_dict(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "config": config,
        },
        os.path.join(EXP_DIR, "checkpoint_final.pth"),
    )
    save_image_grid(next(iter(dl))[:64], os.path.join(EXP_DIR, "real_images.png"))

    # =========================================================================
    # Diagnostics
    # =========================================================================
    print(f"\n{'='*60}\n  DIAGNOSTICS: {EXPERIMENT_NAME}\n{'='*60}")
    df_loss = pd.read_csv(os.path.join(EXP_DIR, "loss.csv"))

    df_loss["D_gap_aug"] = df_loss["D_real_aug"] - df_loss["D_fake_aug"]
    df_loss["D_gap_raw"] = df_loss["D_real_raw"] - df_loss["D_fake_raw"]
    low_gap_epochs = int((df_loss.loc[df_loss["epoch"] > 20, "D_gap_aug"] < 0.10).sum())
    print(f"  Final augmented D gap: {df_loss['D_gap_aug'].iloc[-1]:+.3f}")
    print(f"  Final raw D gap: {df_loss['D_gap_raw'].iloc[-1]:+.3f}")
    print(f"  Low-gap epochs after warmup: {low_gap_epochs}/{max(EPOCHS - 20, 1)}")
    print("  Note: augmented logits are not directly comparable with Exp09 raw logits.")
    print(f"{'='*60}\n")

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(df_loss["epoch"], df_loss["G_loss"], color="#e74c3c", lw=1.5)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Generator Loss (Hinge)"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df_loss["epoch"], df_loss["D_loss"], color="#3498db", lw=1.5)
    axes[1].axhline(2.0, color="red", ls="--", alpha=0.3); axes[1].set(xlabel="Epoch", ylabel="Loss", title="D Loss"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(df_loss["epoch"], df_loss["D_real_aug"], color="#2ecc71", lw=1.5, label="D(Real aug)")
    axes[2].plot(df_loss["epoch"], df_loss["D_fake_aug"], color="#e67e22", lw=1.5, label="D(Fake aug)")
    axes[2].plot(df_loss["epoch"], df_loss["D_real_raw"], color="#27ae60", lw=1.0, ls="--", label="D(Real raw)")
    axes[2].plot(df_loss["epoch"], df_loss["D_fake_raw"], color="#d35400", lw=1.0, ls="--", label="D(Fake raw)")
    axes[2].axhline(0.0, color="gray", ls="--", alpha=0.4); axes[2].set(xlabel="Epoch", ylabel="Logit", title="Augmented D logits")
    axes[2].legend(fontsize=7); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, "loss_curves.png"), dpi=150); plt.close()

    # Metrics
    print(f"\n{'='*60}\n  Computing Metrics for {EXPERIMENT_NAME}\n{'='*60}")
    eval_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    eval_ds = AnimeDataset(image_paths, transform=eval_tf)
    eval_dl = DataLoader(eval_ds, batch_size=64, shuffle=True, drop_last=True, num_workers=2)

    print("FID (EMA, legacy project protocol for direct comparison with Exp09) ...")
    train_sd = {k: v.clone() for k, v in G.state_dict().items()}
    ema.apply_to(G)
    try:
        fid = FIDCalculator(DEVICE).compute_fid(G, eval_dl, n=N_FID)
        print(f"  FID (EMA): {fid:.2f}")
    except Exception as e:
        fid = -1; print(f"  FID FAILED: {e}"); gc.collect()
    G.load_state_dict(train_sd); del train_sd
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    print("LPIPS + Diversity + Laplacian + Edge (EMA) ...")
    train_sd2 = {k: v.clone() for k, v in G.state_dict().items()}
    ema.apply_to(G)
    lpips_calc = LPIPSCalculator(DEVICE); rb, fb = [], []; generated = 0
    real_count = 0
    for imgs in eval_dl:
        rb.append((imgs.to(DEVICE) + 1) / 2.0)
        real_count += imgs.size(0)
        if real_count >= 500:
            break
    rb = torch.cat(rb, dim=0)[:500]
    with torch.no_grad():
        while generated < 500:
            bs = min(32, 500 - generated); noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fb.append((G(noise) + 1) / 2.0); generated += bs
    fb = torch.cat(fb, dim=0)[:500]
    # Compute Diversity while G still has EMA weights
    div = compute_diversity(G, lpips_calc, ns=300)
    G.load_state_dict(train_sd2); del train_sd2  # restore training weights

    lpips_scores = []
    for i in range(0, 500, 50):
        lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i + 50, 500)], rb[i:min(i + 50, 500)]))
    lpips_mean = float(np.mean(lpips_scores))
    fake_lap_values = laplacian_variances(fb[:200])
    real_lap_values = laplacian_variances(rb[:200])
    lap = float(fake_lap_values.mean())
    real_lap = float(real_lap_values.mean())
    blur_threshold = float(np.percentile(real_lap_values, 10))
    blur_rate = float((fake_lap_values < blur_threshold).mean())
    fake_edge, real_edge, edge_ratio = compute_edge_density(fb[:200], rb[:200])
    with torch.no_grad():
        eval_d_real = float(D(rb[:200] * 2.0 - 1.0).mean().item())
        eval_d_fake = float(D(fb[:200] * 2.0 - 1.0).mean().item())
    print(f"  AlexFeat: {lpips_mean:.4f}  Diversity: {div:.4f}  LapVar: {lap:.2f}")
    print(f"  BlurRate@real-p10: {blur_rate:.3f}  EdgeRatio: {edge_ratio:.4f}")
    del lpips_calc; gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()

    fid_delta = round(fid - 49.17, 2) if fid > 0 else None
    lapvar_delta = round(lap - 10833, 2)

    metrics = {
        "experiment_name": EXPERIMENT_NAME, "type": "data_efficient_stability",
        "base": "09 (FID=49.17, LapVar=10833, 768ch, 20K, Hinge)",
        "changes": [
            f"DiffAugment: {DIFFAUG_POLICY}, applied symmetrically to real/fake and on G path",
            f"EMA on G: decay={EMA_DECAY} (Yazici et al., ICLR 2019)"
        ],
        "r1": "disabled because Exp05 regressed FID on the Width3x family",
        "epochs": EPOCHS, "batch_size": BATCH_SIZE, "dataset_size": actual_dataset_size,
        "FID": round(fid, 2), "FID_delta_vs_09": fid_delta,
        "FID_protocol": (
            "Legacy project torchvision Inception-v3 pipeline, 10K real/10K fake. "
            "Comparable with Exp09 inside this project; not directly comparable "
            "with published clean-fid/torch-fidelity values."
        ),
        "LPIPS_legacy_AlexNet_feature_distance": round(lpips_mean, 4),
        "metric_note": "Legacy LPIPS field is AlexNet feature MSE, not calibrated LPIPS.",
        "Diversity": round(div, 4),
        "Laplacian_Variance": round(lap, 2), "LapVar_delta_vs_09": lapvar_delta,
        "Laplacian_Variance_Real": round(real_lap, 2),
        "Blur_Threshold_Real_P10": round(blur_threshold, 2),
        "Blur_Rate_Fake_Below_Real_P10": round(blur_rate, 4),
        "Edge_Density_Fake": round(fake_edge, 4), "Edge_Density_Real": round(real_edge, 4),
        "Edge_Density_Ratio": round(edge_ratio, 4),
        "final_G_loss": round(fgl, 4), "final_D_loss": round(fdl, 4),
        "D_real_aug": round(fdr, 4), "D_fake_aug": round(fdf, 4),
        "D_real_raw_final_epoch": round(fdrr, 4),
        "D_fake_raw_final_epoch": round(fdfr, 4),
        "D_real_unaug_eval": round(eval_d_real, 4),
        "D_fake_unaug_eval": round(eval_d_fake, 4),
        "low_D_gap_epochs_after_warmup": low_gap_epochs,
        "G_params": gp, "D_params": dp,
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f: json.dump(metrics, f, indent=2)
    print(f"\n  Complete: {EXPERIMENT_NAME}  FID={fid:.2f}  vs 09(49.17): {fid_delta}  "
          f"LapVar vs 09(10833): {lapvar_delta}")

if __name__ == "__main__":
    main()
