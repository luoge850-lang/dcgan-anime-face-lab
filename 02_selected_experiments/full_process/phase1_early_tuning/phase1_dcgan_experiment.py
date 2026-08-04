"""
================================================================================
 DCGAN Experiment Runner — Flat Output, Standard Metrics
================================================================================
 Per-experiment output (ONE flat folder, no subdirectories):

   exp2_a_flip/
     metrics.json          ← FID, LPIPS, Diversity, Laplacian Variance, Loss
     loss.csv              ← epoch, D_loss, G_loss, D_real, D_fake
     loss_curves.png       ← 3-panel plot
     epoch_050.png         ← 64-image grid from fixed noise
     epoch_100.png
     real_images.png
     generator_final.pth
     discriminator_final.pth

 Metrics (all standard implementations):
   1. FID               — InceptionV3 pool3, 10000 real vs 10000 fake
   2. LPIPS             — AlexNet 5-layer multi-scale perceptual distance
   3. Diversity         — Pairwise LPIPS among 500 generated images
   4. Laplacian Variance — Sharpness measure (cv2.Laplacian variance)
   5. Loss curves       — G Loss, D Loss, D(Real), D(Fake) per epoch
   6. 50-epoch samples  — 64-image grid from fixed latent noise

 Architecture & training: UNCHANGED from model64.py + main.py.
================================================================================
"""

import os, csv, json, random, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from scipy import linalg

# =============================================================================
# >>> CONFIG — MODIFY ONLY THIS BLOCK FOR EACH EXPERIMENT <<<
# =============================================================================
EXPERIMENT_NAME = "baseline_epoch100"
EXPERIMENT_GROUP = "exp1_epoch_study"

OUTPUT_DIR      = "/kaggle/working/dcgan_output"
DATASET_15K_DIR = "/kaggle/working/dataset_15k"

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS     = 100
NOISE_DIM  = 100
LR         = 1e-4
BETAS      = (0.5, 0.99)
SEED       = 42
SAMPLE_INTERVAL = 50

# Derived — do NOT modify
EXP_DIR = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# >>> TRANSFORM — Modify ONLY for Exp 2 & Exp 3 <<<
# =============================================================================
def get_transform():
    """
    Baseline: no augmentation.
    Exp 2/3: insert augmentation transforms between Resize and ToTensor.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # >>> INSERT AUGMENTATION HERE <<<
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


# =============================================================================
# Custom Augmentation Classes
# =============================================================================
class EdgeSharpen:
    """Unsharp Mask: output = input + alpha * (input - blurred)."""
    def __init__(self, prob=0.3, alpha=0.3):
        self.prob, self.alpha = prob, alpha
    def __call__(self, img):
        if random.random() < self.prob:
            arr = np.array(img, dtype=np.float32) / 255.0
            blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=1.5)), dtype=np.float32) / 255.0
            sharp = arr + self.alpha * (arr - blurred)
            return Image.fromarray(np.clip(sharp * 255, 0, 255).astype(np.uint8))
        return img


class MedianDenoise:
    """Median filter for noise reduction."""
    def __init__(self, prob=0.2, kernel_size=3):
        self.prob, self.ksize = prob, kernel_size
    def __call__(self, img):
        if random.random() < self.prob:
            return img.filter(ImageFilter.MedianFilter(size=self.ksize))
        return img


# =============================================================================
# Reproducibility & Dataset utilities
# =============================================================================
def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_all_images(root_dir):
    extensions = {".png", ".jpg", ".jpeg"}
    files = []
    if not os.path.exists(root_dir): return files
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def prepare_dataset_15k():
    if os.path.exists(DATASET_15K_DIR):
        existing = find_all_images(DATASET_15K_DIR)
        if len(existing) >= 15000:
            print(f"[SKIP] {DATASET_15K_DIR} ready ({len(existing)} images)")
            return DATASET_15K_DIR, None, len(existing)
    print("Scanning /kaggle/input/ for images ...")
    all_images = []
    for sub in sorted(os.listdir("/kaggle/input")):
        sp = os.path.join("/kaggle/input", sub)
        if os.path.isdir(sp):
            imgs = find_all_images(sp)
            if imgs: all_images.extend(imgs); print(f"  {sp}: {len(imgs)} images")
    if not all_images:
        raise FileNotFoundError("No images found. Use 'Add Data' to attach dataset.")
    original_size = len(all_images)
    set_all_seeds(SEED)
    sampled = random.sample(all_images, min(15000, original_size))
    os.makedirs(DATASET_15K_DIR, exist_ok=True)
    print(f"Copying {len(sampled)} images to {DATASET_15K_DIR} ...")
    for i, src in enumerate(sampled):
        shutil.copy2(src, os.path.join(DATASET_15K_DIR, f"image{i:05d}{Path(src).suffix}"))
    return DATASET_15K_DIR, original_size, len(sampled)


class AnimeDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths, self.transform = image_paths, transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img) if self.transform else img


# =============================================================================
# Model: Generator (model64.py — UNCHANGED)
# =============================================================================
class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, 4), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Tanh(),
        )
    def forward(self, x): return self.net(x)


# =============================================================================
# Model: Discriminator (model64.py — UNCHANGED)
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4*4*256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x): return self.net(x).view(-1)


# =============================================================================
# Image utility
# =============================================================================
def save_image_grid(tensor, filepath, nrow=8):
    grid = make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    Image.fromarray(ndarr).save(filepath)


# =============================================================================
# Metric 1: FID (Fréchet Inception Distance)
# Standard: InceptionV3 pool3 (2048-dim), 10000 real vs 10000 fake
# Ref: Heusel et al., NeurIPS 2017
# =============================================================================
class FIDCalculator:
    def __init__(self, device):
        self.device = device
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
        inception.fc = nn.Identity()
        inception.eval()
        self.inception = inception.to(device)
        for p in self.inception.parameters(): p.requires_grad = False

    @torch.no_grad()
    def _get_features(self, images):
        images = (images + 1) / 2.0
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images - 0.5) / 0.5
        return self.inception(images).cpu().numpy()

    @torch.no_grad()
    def compute_fid(self, generator, real_loader, num_fake=10000):
        generator.eval()
        real_feats, fake_feats = [], []
        collected = 0
        for imgs in real_loader:
            imgs = imgs.to(self.device)
            real_feats.append(self._get_features(imgs))
            collected += imgs.size(0)
            if collected >= num_fake: break
        real_feats = np.concatenate(real_feats, axis=0)[:num_fake]
        generated = 0
        while generated < num_fake:
            bs = min(64, num_fake - generated)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=self.device)
            fake_feats.append(self._get_features(generator(noise)))
            generated += bs
        fake_feats = np.concatenate(fake_feats, axis=0)[:num_fake]
        mu_r, sigma_r = np.mean(real_feats, axis=0), np.cov(real_feats, rowvar=False)
        mu_f, sigma_f = np.mean(fake_feats, axis=0), np.cov(fake_feats, rowvar=False)
        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
        if np.iscomplexobj(covmean): covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)
        generator.train()
        return float(fid)


# =============================================================================
# Metric 2: LPIPS (Learned Perceptual Image Patch Similarity)
# Standard: AlexNet 5-layer multi-scale, L2-normalized, unit-weighted
# Ref: Zhang et al., CVPR 2018
# =============================================================================
class LPIPSCalculator:
    def __init__(self, device):
        self.device = device
        alexnet = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
        alexnet.eval()
        # Standard LPIPS: conv1, conv2, conv3, conv4, conv5 of AlexNet
        self.layers = nn.ModuleList([
            alexnet.features[:3],    # conv1  (64 channels)
            alexnet.features[:6],    # conv2  (192 channels)
            alexnet.features[:9],    # conv3  (384 channels)
            alexnet.features[:12],   # conv4  (256 channels)
            alexnet.features,        # conv5  (256 channels)
        ]).to(device)
        for p in self.parameters(): p.requires_grad = False

    def _normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return (x - mean) / std

    @torch.no_grad()
    def compute_lpips(self, img1, img2):
        """Multi-scale LPIPS: average L2 across 5 AlexNet layers, unit-normalized."""
        img1 = self._normalize(img1)
        img2 = self._normalize(img2)
        total = 0.0
        for layer in self.layers:
            f1 = layer(img1)
            f2 = layer(img2)
            # Unit-normalize per channel (standard LPIPS normalization)
            f1 = f1 / (f1.norm(dim=1, keepdim=True) + 1e-10)
            f2 = f2 / (f2.norm(dim=1, keepdim=True) + 1e-10)
            total += (f1 - f2).pow(2).mean(dim=[1, 2, 3])
        return (total / len(self.layers)).cpu().numpy()

    def parameters(self):
        return self.layers.parameters()


# =============================================================================
# Metric 3: Diversity (pairwise LPIPS among generated images)
# =============================================================================
@torch.no_grad()
def compute_diversity(generator, lpips_calc, num_samples=500):
    generator.eval()
    all_imgs = []
    generated = 0
    while generated < num_samples:
        bs = min(32, num_samples - generated)
        noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
        fake = generator(noise)
        all_imgs.append((fake + 1) / 2.0)
        generated += bs
    all_imgs = torch.cat(all_imgs, dim=0)[:num_samples]
    num_pairs = 2000
    idx1 = torch.randint(0, num_samples, (num_pairs,))
    idx2 = torch.randint(0, num_samples, (num_pairs,))
    scores = []
    for i in range(0, num_pairs, 50):
        end = min(i + 50, num_pairs)
        scores.extend(lpips_calc.compute_lpips(all_imgs[idx1[i:end]].to(DEVICE), all_imgs[idx2[i:end]].to(DEVICE)))
    generator.train()
    return float(np.mean(scores))


# =============================================================================
# Metric 4: Laplacian Variance (Sharpness)
# Higher variance = more edges = sharper image
# =============================================================================
def compute_laplacian_variance(images_denormed):
    """Compute mean Laplacian variance over a batch of images in [0,1]."""
    import cv2
    variances = []
    for img in images_denormed:
        np_img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        variances.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    return float(np.mean(variances))


# =============================================================================
# Main
# =============================================================================
def main():
    set_all_seeds(SEED)
    os.makedirs(EXP_DIR, exist_ok=True)

    # Dataset (shared across all experiments via seed=42)
    dataset_path, original_size, sample_size = prepare_dataset_15k()
    image_paths = find_all_images(dataset_path)

    # Fixed noise for cross-epoch visual comparison
    set_all_seeds(SEED)
    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)

    # DataLoader
    transform = get_transform()
    dataset = AnimeDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Info
    g_tmp, d_tmp = Generator(), Discriminator()
    g_p = sum(p.numel() for p in g_tmp.parameters())
    d_p = sum(p.numel() for p in d_tmp.parameters())
    del g_tmp, d_tmp
    print(f"\n{'='*55}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")
    print(f"  Device:      {DEVICE}")
    if DEVICE.type == "cuda": print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Generator:   {g_p:,} params  |  Discriminator: {d_p:,} params")
    print(f"  Dataset:     {len(image_paths)} images  |  {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch:       {BATCH_SIZE}  |  Steps/epoch: {len(dataloader)}")
    print(f"  Epochs:      {EPOCHS}  |  Seed: {SEED}")
    print(f"  Optimizer:   Adam lr={LR} betas={BETAS}  |  Loss: BCELoss")
    print(f"{'='*55}\n")

    # Models
    generator     = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)
    criterion     = nn.BCELoss()
    g_opt = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    d_opt = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    # Loss log
    csv_file = open(os.path.join(EXP_DIR, "loss.csv"), "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "D_loss", "G_loss", "D_real", "D_fake"])
    final_dl = final_gl = final_dr = final_df = 0.0

    # ---- Training ----
    print("Training ...\n")
    for epoch in range(1, EPOCHS + 1):
        for img in dataloader:
            real_img = img.to(DEVICE)
            bs = real_img.size(0)
            rl, fl = torch.ones(bs, device=DEVICE), torch.zeros(bs, device=DEVICE)
            # D
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake_img = generator(noise)
            ro, fo = discriminator(real_img), discriminator(fake_img.detach())
            d_loss = criterion(ro, rl) + criterion(fo, fl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            # G
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            g_loss = criterion(discriminator(generator(noise)), rl)
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl, gl, dr, df = d_loss.item(), g_loss.item(), ro.mean().item(), fo.mean().item()
        writer.writerow([epoch, dl, gl, dr, df])
        final_dl, final_gl, final_dr, final_df = dl, gl, dr, df
        print(f"Epoch [{epoch:3d}/{EPOCHS}]  D:{dl:.4f}  G:{gl:.4f}  DR:{dr:.4f}  DF:{df:.4f}")

        # Save 50-epoch sample
        if epoch % SAMPLE_INTERVAL == 0:
            generator.eval()
            with torch.no_grad(): samples = generator(fixed_noise)
            generator.train()
            save_image_grid(samples, os.path.join(EXP_DIR, f"epoch_{epoch:03d}.png"))
            torch.save(generator.state_dict(), os.path.join(EXP_DIR, f"generator_epoch_{epoch:03d}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(EXP_DIR, f"discriminator_epoch_{epoch:03d}.pth"))

    csv_file.close()

    # Final weights
    torch.save(generator.state_dict(), os.path.join(EXP_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(EXP_DIR, "discriminator_final.pth"))

    # Save real images
    save_image_grid(next(iter(dataloader))[:64], os.path.join(EXP_DIR, "real_images.png"))

    # ---- Loss curves ----
    df = pd.read_csv(os.path.join(EXP_DIR, "loss.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(df["epoch"], df["G_loss"], color="#e74c3c", linewidth=1.5)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Generator Loss"); axes[0].grid(True, alpha=0.3)
    axes[1].plot(df["epoch"], df["D_loss"], color="#3498db", linewidth=1.5)
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Discriminator Loss"); axes[1].grid(True, alpha=0.3)
    axes[2].plot(df["epoch"], df["D_real"], color="#2ecc71", linewidth=1.5, label="D(Real)")
    axes[2].plot(df["epoch"], df["D_fake"], color="#e67e22", linewidth=1.5, label="D(Fake)")
    axes[2].axhline(0.5, color="gray", ls="--", alpha=0.4)
    axes[2].set(xlabel="Epoch", ylabel="Mean", title="D(Real) vs D(Fake)")
    axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR, "loss_curves.png"), dpi=150); plt.close()

    # ===== Post-Training Metrics =====
    print(f"\n{'='*55}\n  Computing Evaluation Metrics\n{'='*55}")

    # Shared eval loader (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    eval_dataset = AnimeDataset(image_paths, transform=eval_transform)
    eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=True, drop_last=True)

    # --- FID ---
    print("Computing FID ...")
    fid_score = FIDCalculator(DEVICE).compute_fid(generator, eval_loader, num_fake=10000)
    print(f"  FID: {fid_score:.2f}")
    import gc; gc.collect(); torch.cuda.empty_cache()

    # --- LPIPS + Diversity + Laplacian ---
    print("Computing LPIPS, Diversity, Laplacian Variance ...")
    lpips_calc = LPIPSCalculator(DEVICE)

    real_batch_list, fake_batch_list = [], []
    generated = 0
    for imgs in eval_loader:
        real_batch_list.append((imgs.to(DEVICE) + 1) / 2.0)
        if sum(r.size(0) for r in real_batch_list) >= 500: break
    real_batch = torch.cat(real_batch_list, dim=0)[:500]

    with torch.no_grad():
        while generated < 500:
            bs = min(32, 500 - generated)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake = generator(noise)
            fake_batch_list.append((fake + 1) / 2.0)
            generated += bs
    fake_batch = torch.cat(fake_batch_list, dim=0)[:500]

    # LPIPS
    lpips_scores = []
    for i in range(0, 500, 50):
        end = min(i + 50, 500)
        lpips_scores.extend(lpips_calc.compute_lpips(fake_batch[i:end], real_batch[i:end]))
    lpips_mean = float(np.mean(lpips_scores))
    print(f"  LPIPS: {lpips_mean:.4f}")

    # Diversity
    div_score = compute_diversity(generator, lpips_calc, num_samples=300)
    print(f"  Diversity: {div_score:.4f}")

    # Laplacian Variance
    lap_var = compute_laplacian_variance(fake_batch[:200])
    print(f"  Laplacian Variance: {lap_var:.2f}")

    del lpips_calc; gc.collect(); torch.cuda.empty_cache()

    # ===== Save metrics.json =====
    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "experiment_group": EXPERIMENT_GROUP,
        "epochs": EPOCHS,
        "dataset_size": len(image_paths),
        "FID": round(fid_score, 2),
        "LPIPS": round(lpips_mean, 4),
        "Diversity": round(div_score, 4),
        "Laplacian_Variance": round(lap_var, 2),
        "final_G_loss": round(final_gl, 4),
        "final_D_loss": round(final_dl, 4),
        "D_real": round(final_dr, 4),
        "D_fake": round(final_df, 4),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(EXP_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Complete: {EXPERIMENT_NAME}")
    print(f"  FID={fid_score:.2f}  LPIPS={lpips_mean:.4f}  Div={div_score:.4f}  LapVar={lap_var:.2f}")
    print(f"  All results: {os.path.abspath(EXP_DIR)}/")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
