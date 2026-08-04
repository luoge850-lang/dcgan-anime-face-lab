"""
================================================================================
 DCGAN Baseline Experiment — Standardized Experimental Framework
================================================================================
 Purpose: Single-file, strictly reproducible DCGAN baseline for Kaggle.
          All subsequent experiments (epoch study, augmentation ablation,
          combination augmentation) build on this foundation.

 Architecture & training logic: UNCHANGED from original model64.py + main.py.
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
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from scipy import linalg


# =============================================================================
# Part 1: Experiment Configuration
# =============================================================================
EXPERIMENT_NAME = "baseline_epoch100"
DATASET_PATH    = "/kaggle/input/anime-face-dataset"
OUTPUT_DIR      = "/kaggle/working/dcgan_output"
DATASET_15K_DIR = "/kaggle/working/dataset_15k"

IMAGE_SIZE = 64
BATCH_SIZE = 32
EPOCHS     = 100
NOISE_DIM  = 100
LR         = 1e-4
BETAS      = (0.5, 0.99)
SEED       = 42

# Derived paths — do NOT modify
EXP_DIR      = os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)
SAMPLE_DIR   = os.path.join(EXP_DIR, "samples")
MODEL_DIR    = os.path.join(EXP_DIR, "models")
LOG_DIR      = os.path.join(EXP_DIR, "logs")
FID_DIR      = os.path.join(EXP_DIR, "fid_images", "fake")
EVAL_DIR     = os.path.join(EXP_DIR, "evaluation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Part 3: Reproducibility
# =============================================================================
def set_all_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Part 2: Dataset — recursive image discovery
# =============================================================================
def find_all_images(root_dir):
    extensions = {".png", ".jpg", ".jpeg"}
    files = []
    if not os.path.exists(root_dir):
        return files
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if Path(f).suffix.lower() in extensions:
                files.append(os.path.join(dirpath, f))
    return sorted(files)


def prepare_dataset_15k():
    """Randomly sample 15k images from Kaggle input with fixed seed=42."""
    if os.path.exists(DATASET_15K_DIR):
        existing = find_all_images(DATASET_15K_DIR)
        if len(existing) >= 15000:
            print(f"[SKIP] {DATASET_15K_DIR} ready ({len(existing)} images)")
            return DATASET_15K_DIR, None, len(existing)

    # Scan Kaggle input
    print(f"Scanning /kaggle/input/ for images ...")
    all_images = []
    for sub in sorted(os.listdir("/kaggle/input")):
        sp = os.path.join("/kaggle/input", sub)
        if os.path.isdir(sp):
            imgs = find_all_images(sp)
            if imgs:
                all_images.extend(imgs)
                print(f"  {sp}: {len(imgs)} images")

    if not all_images:
        raise FileNotFoundError("No images found. Use 'Add Data' to attach dataset.")

    original_size = len(all_images)
    set_all_seeds(SEED)
    sampled = random.sample(all_images, min(15000, original_size))

    os.makedirs(DATASET_15K_DIR, exist_ok=True)
    print(f"Copying {len(sampled)} images to {DATASET_15K_DIR} ...")
    for i, src in enumerate(sampled):
        dst = os.path.join(DATASET_15K_DIR, f"image{i:05d}{Path(src).suffix}")
        shutil.copy2(src, dst)
    print(f"Done.")
    return DATASET_15K_DIR, original_size, len(sampled)


def save_dataset_info(original_size, sample_size):
    info = {
        "original_size": original_size,
        "sample_size": sample_size,
        "seed": SEED,
        "dataset_path": DATASET_15K_DIR,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = os.path.join(EXP_DIR, "dataset_info.json")
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"dataset_info.json saved")


# =============================================================================
# Dataset Class
# =============================================================================
class AnimeDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# =============================================================================
# Model: Generator (model64.py — UNCHANGED)
# =============================================================================
class Generator(nn.Module):
    def __init__(self, noise_dim=NOISE_DIM):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, 256, 4),          # 1x1 -> 4x4
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),          # 4x4 -> 8x8
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),           # 8x8 -> 16x16
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),            # 16x16 -> 32x32
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),             # 32x32 -> 64x64
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# Model: Discriminator (model64.py — UNCHANGED)
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.LeakyReLU(0.2),  # 64 -> 32
            nn.Conv2d(32, 64, 3, 2, 1), nn.LeakyReLU(0.2),  # 32 -> 16
            nn.Conv2d(64, 128, 3, 2, 1), nn.LeakyReLU(0.2), # 16 -> 8
            nn.Conv2d(128, 256, 3, 2, 1), nn.LeakyReLU(0.2),# 8  -> 4
            nn.Flatten(),
            nn.Linear(4 * 4 * 256, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


# =============================================================================
# Image utilities
# =============================================================================
def save_image_grid(tensor, filepath, nrow=8):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    grid = make_grid(tensor, nrow=nrow, normalize=True, value_range=(-1, 1))
    ndarr = grid.mul(255).clamp(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    Image.fromarray(ndarr).save(filepath)


# =============================================================================
# Part 6: FID Calculator
# =============================================================================
class FIDCalculator:
    """Standard FID using Inception-v3 pool3 features (2048-dim)."""
    def __init__(self, device):
        self.device = device
        inception = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
        )
        inception.fc = nn.Identity()
        inception.eval()
        self.inception = inception.to(device)
        for p in self.inception.parameters():
            p.requires_grad = False

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

        # Real features
        collected = 0
        for imgs in real_loader:
            imgs = imgs.to(self.device)
            real_feats.append(self._get_features(imgs))
            collected += imgs.size(0)
            if collected >= num_fake:
                break
        real_feats = np.concatenate(real_feats, axis=0)[:num_fake]

        # Fake features
        batch_size = 64
        generated = 0
        while generated < num_fake:
            bs = min(batch_size, num_fake - generated)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=self.device)
            fake = generator(noise)
            fake_feats.append(self._get_features(fake))
            generated += bs
        fake_feats = np.concatenate(fake_feats, axis=0)[:num_fake]

        # Statistics
        mu_r = np.mean(real_feats, axis=0)
        sigma_r = np.cov(real_feats, rowvar=False)
        mu_f = np.mean(fake_feats, axis=0)
        sigma_f = np.cov(fake_feats, rowvar=False)

        diff = mu_r - mu_f
        covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_f), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        fid = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)
        generator.train()
        return float(fid)


# =============================================================================
# Part 5B: Generate FID fake images to disk
# =============================================================================
def generate_fid_images(generator, device):
    os.makedirs(FID_DIR, exist_ok=True)
    generator.eval()
    batch_size = 64
    count = 0
    print(f"Generating 10000 fake images to {FID_DIR} ...")
    with torch.no_grad():
        while count < 10000:
            bs = min(batch_size, 10000 - count)
            noise = torch.randn(bs, NOISE_DIM, 1, 1, device=device)
            fake = generator(noise)
            for j in range(bs):
                # Denormalize [-1,1] -> [0,1] -> uint8
                img = (fake[j] + 1) / 2.0
                img = img.clamp(0, 1).mul(255).permute(1, 2, 0).cpu().byte().numpy()
                Image.fromarray(img).save(os.path.join(FID_DIR, f"{count+1:06d}.png"))
                count += 1
    generator.train()
    print(f"  {count} images saved.")


# =============================================================================
# Main
# =============================================================================
def main():
    set_all_seeds(SEED)

    # Create output directories
    for d in [EXP_DIR, SAMPLE_DIR, MODEL_DIR, LOG_DIR, FID_DIR, EVAL_DIR]:
        os.makedirs(d, exist_ok=True)

    # ===== Part 2: Dataset preparation =====
    dataset_path, original_size, sample_size = prepare_dataset_15k()
    save_dataset_info(original_size, sample_size)

    image_paths = find_all_images(dataset_path)

    # ===== Part 4: Fixed noise for visualization =====
    set_all_seeds(SEED)
    fixed_noise = torch.randn(64, NOISE_DIM, 1, 1, device=DEVICE)

    # ===== Data transform — NO augmentation (baseline) =====
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = AnimeDataset(image_paths, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
    )

    # ===== Print info =====
    g_tmp = Generator()
    d_tmp = Discriminator()
    g_params = sum(p.numel() for p in g_tmp.parameters())
    d_params = sum(p.numel() for p in d_tmp.parameters())
    del g_tmp, d_tmp

    print(f"\n{'='*55}")
    print(f"  Experiment:  {EXPERIMENT_NAME}")
    print(f"  Device:      {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  Generator:   {g_params:,} params")
    print(f"  Discriminator: {d_params:,} params")
    print(f"  Dataset:     {len(image_paths)} images | {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Batch:       {BATCH_SIZE} | Steps/epoch: {len(dataloader)}")
    print(f"  Epochs:      {EPOCHS} | Total iters: ~{len(dataloader)*EPOCHS}")
    print(f"  Optimizer:   Adam | lr={LR} | betas={BETAS}")
    print(f"  Loss:        BCELoss")
    print(f"  Augmentation: None (baseline)")
    print(f"  Seed:        {SEED}")
    print(f"{'='*55}\n")

    # ===== Initialize models =====
    generator     = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    criterion   = nn.BCELoss()
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

    # ===== Training log =====
    csv_path = os.path.join(LOG_DIR, "loss.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    writer.writerow(["epoch", "D_loss", "G_loss", "D_real", "D_fake"])

    final_d_loss = final_g_loss = final_dr = final_df = 0.0

    # ===== Training Loop =====
    print("Training ...\n")
    for epoch in range(1, EPOCHS + 1):
        for img in dataloader:
            real_img = img.to(DEVICE)
            bs = real_img.size(0)

            real_label = torch.ones(bs, device=DEVICE)
            fake_label = torch.zeros(bs, device=DEVICE)

            # -- Train Discriminator --
            noise    = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake_img = generator(noise)

            real_out = discriminator(real_img)
            fake_out = discriminator(fake_img.detach())

            d_loss = criterion(real_out, real_label) + criterion(fake_out, fake_label)
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # -- Train Generator --
            noise    = torch.randn(bs, NOISE_DIM, 1, 1, device=DEVICE)
            fake_img = generator(noise)
            output   = discriminator(fake_img)

            g_loss = criterion(output, real_label)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # Log
        dl, gl, dr, df = d_loss.item(), g_loss.item(), real_out.mean().item(), fake_out.mean().item()
        writer.writerow([epoch, dl, gl, dr, df])
        final_d_loss, final_g_loss, final_dr, final_df = dl, gl, dr, df

        print(
            f"Epoch [{epoch:3d}/{EPOCHS}]  "
            f"D: {dl:.4f}  G: {gl:.4f}  "
            f"D(real): {dr:.4f}  D(fake): {df:.4f}"
        )

        # ===== Part 5A: Save visualization sample every 50 epochs =====
        if epoch % 50 == 0:
            generator.eval()
            with torch.no_grad():
                samples = generator(fixed_noise)
            generator.train()
            save_image_grid(samples, os.path.join(SAMPLE_DIR, f"epoch_{epoch:03d}.png"))

            # ===== Part 9: Save model checkpoint every 50 epochs =====
            torch.save(
                generator.state_dict(),
                os.path.join(MODEL_DIR, f"generator_epoch_{epoch:03d}.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(MODEL_DIR, f"discriminator_epoch_{epoch:03d}.pth"),
            )

    csv_file.close()

    # ===== Final model save =====
    torch.save(generator.state_dict(),     os.path.join(MODEL_DIR, "generator_final.pth"))
    torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, "discriminator_final.pth"))

    # ===== Part 8: Loss curves =====
    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(df["epoch"], df["G_loss"], color="#e74c3c", linewidth=1.5)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Generator Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(df["epoch"], df["D_loss"], color="#3498db", linewidth=1.5)
    axes[1].set(xlabel="Epoch", ylabel="Loss", title="Discriminator Loss")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(df["epoch"], df["D_real"], color="#2ecc71", linewidth=1.5, label="D(Real)")
    axes[2].plot(df["epoch"], df["D_fake"], color="#e67e22", linewidth=1.5, label="D(Fake)")
    axes[2].axhline(0.5, color="gray", ls="--", alpha=0.4)
    axes[2].set(xlabel="Epoch", ylabel="Mean", title="D(Real) vs D(Fake)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "loss_curves.png"), dpi=150)
    plt.close()

    # ===== Save real images reference =====
    save_image_grid(next(iter(dataloader))[:64], os.path.join(SAMPLE_DIR, "real_images.png"))

    # ===== Part 5B + 6 + 7: FID Evaluation =====
    print(f"\n{'='*55}")
    print(f"  FID Evaluation")
    print(f"{'='*55}")

    # Generate 10000 fake images to disk
    generate_fid_images(generator, DEVICE)

    # Prepare FID real loader (no augmentation)
    fid_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    fid_dataset = AnimeDataset(image_paths, transform=fid_transform)
    fid_loader = DataLoader(fid_dataset, batch_size=64, shuffle=True, drop_last=True)

    fid_calc = FIDCalculator(DEVICE)
    fid_score = fid_calc.compute_fid(generator, fid_loader, num_fake=10000)
    print(f"  FID Score: {fid_score:.2f}")

    # ===== Part 7: Save evaluation metrics =====
    metrics = {
        "experiment_name": EXPERIMENT_NAME,
        "epochs": EPOCHS,
        "dataset_size": len(image_paths),
        "FID": round(fid_score, 2),
        "final_G_loss": round(final_g_loss, 4),
        "final_D_loss": round(final_d_loss, 4),
        "D_real": round(final_dr, 4),
        "D_fake": round(final_df, 4),
        "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    metrics_path = os.path.join(EVAL_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  metrics.json saved: {metrics_path}")

    # ===== Final summary =====
    print(f"\n{'='*55}")
    print(f"  Experiment Complete: {EXPERIMENT_NAME}")
    print(f"{'='*55}")
    print(f"  Output: {os.path.abspath(EXP_DIR)}/")
    print(f"    dataset_info.json")
    print(f"    samples/          — epoch_050.png, epoch_100.png, real_images.png")
    print(f"    models/           — generator_epoch_*.pth, generator_final.pth")
    print(f"    logs/             — loss.csv, loss_curves.png")
    print(f"    fid_images/fake/  — 10000 generated PNGs for FID")
    print(f"    evaluation/       — metrics.json (FID={fid_score:.2f})")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
