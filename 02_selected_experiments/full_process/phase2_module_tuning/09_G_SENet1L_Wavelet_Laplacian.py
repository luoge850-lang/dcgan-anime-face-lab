"""
================================================================================
 Final Model v1: SENet 1L + Wavelet Loss + Laplacian Pyramid Loss [FAILED]
================================================================================
 Result: FID=104.93 — WORSE than SENet 1L alone (96.76) and Laplacian alone (98.67)

 Root cause: Wavelet (4-subband frequency) and Laplacian (3-level scale) decompose
   the SAME image information in two different bases. Combined they double-penalize
   the same structural errors → excessive L1 regularization → over-constrained G.

   Wavelet LL ≈ Laplacian coarse scale (global structure)
   Wavelet LH/HL/HH ≈ Laplacian fine scales (edges/details)

 Status: FAILED. Archived for reference. Replaced by v2 (SENet 1L + Laplacian only).

 Architecture:
   - Generator: model64.py + 1 SEBlock at L3 (+1K params)
   - Discriminator: model64.py UNCHANGED
   - Generator Loss: BCE_adv + 0.1×Wavelet_L1 + 0.1×Laplacian_L1

 References:
   - Hu et al., "SENet." CVPR 2018.
   - Mallat, "A Theory for Multiresolution Signal Decomposition." IEEE TPAMI 1989.
   - Burt & Adelson, "The Laplacian Pyramid as a Compact Image Code." 1983.
================================================================================
"""

import os, csv, json, random, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
import cv2

import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import transforms, models
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from scipy import linalg

# =============================================================================
EXPERIMENT_NAME = "09_G_SENet1L_Wavelet_Laplacian"
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"; DATASET_LIMIT=8000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=100; LR=1e-4; BETAS=(0.5,0.99); SEED=42
EPOCHS=200; SAMPLE_INTERVAL=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR=os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

# ----- SENet hyperparameters -----
SE_REDUCTION = 16
SE_MIN_BOTTLENECK = 8

# ----- Wavelet Loss hyperparameters -----
WAVELET_LAMBDA = 0.1
W_LL = 1.0
W_LH = 0.5
W_HL = 0.5
W_HH = 0.25
W_LOSS = "l1"

# ----- Laplacian Pyramid hyperparameters -----
PYRAMID_LEVELS = 3
LAPLACIAN_LAMBDA = 0.1
PYRAMID_WEIGHTS = [1.0, 0.5, 0.25]

# =============================================================================
# Pre-download model weights (retry on Kaggle DNS failure)
# =============================================================================
import time as _time
def _download_with_retry(model_fn, name):
    for attempt in range(5):
        try:
            return model_fn()
        except Exception as e:
            if attempt < 4:
                wait = 2 ** attempt * 5
                print(f"  {name} attempt {attempt+1} failed, retry in {wait}s...")
                _time.sleep(wait)
            else:
                raise e

print("Pre-loading models...")
try:
    _inc = _download_with_retry(lambda: models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False), "InceptionV3")
    del _inc; gc.collect()
    print("  InceptionV3: OK")
except: print("  InceptionV3: FAILED — FID will use -1")

try:
    _anet = _download_with_retry(lambda: models.alexnet(weights=models.AlexNet_Weights.DEFAULT), "AlexNet")
    del _anet; gc.collect()
    print("  AlexNet: OK")
except: print("  AlexNet: FAILED — LPIPS/Diversity will use -1")
print()

# =============================================================================
def set_all_seeds(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def find_all_images(root_dir):
    exts={".png",".jpg",".jpeg"}; files=[]
    if not os.path.exists(root_dir): return files
    for dp,_,fn in os.walk(root_dir):
        for f in fn:
            if Path(f).suffix.lower() in exts: files.append(os.path.join(dp,f))
    return sorted(files)

def load_dataset():
    if os.path.exists(DATASET_PATH):
        imgs=find_all_images(DATASET_PATH)
        if imgs: print(f"Dataset: {DATASET_PATH} ({len(imgs)} images)"); return DATASET_PATH,imgs
    print("Scanning /kaggle/input/ ...")
    for sub in sorted(os.listdir("/kaggle/input")):
        sp=os.path.join("/kaggle/input",sub)
        if os.path.isdir(sp):
            imgs=find_all_images(sp)
            if imgs: print(f"Dataset: {sp} ({len(imgs)} images)"); return sp,imgs
    raise FileNotFoundError("GANAnime Lite not found. Add via Kaggle 'Add Data'")

class AnimeDataset(Dataset):
    def __init__(self,paths,transform=None): self.paths,self.tf=paths,transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        img=Image.open(self.paths[i]).convert("RGB")
        return self.tf(img) if self.tf else img

def save_image_grid(tensor,fp,nrow=8):
    grid=make_grid(tensor,nrow=nrow,normalize=True,value_range=(-1,1))
    ndarr=grid.mul(255).clamp(0,255).permute(1,2,0).to("cpu",torch.uint8).numpy()
    Image.fromarray(ndarr).save(fp)

# =============================================================================
# Augmentation: Flip(p=0.5) + EdgeSharpen(p=0.2)
# =============================================================================
class EdgeSharpen:
    def __init__(self,prob=0.2,alpha=0.3): self.prob,self.alpha=prob,alpha
    def __call__(self,img):
        if random.random()<self.prob:
            arr=np.array(img,dtype=np.float32)/255.0
            blurred=np.array(img.filter(ImageFilter.GaussianBlur(radius=1.5)),dtype=np.float32)/255.0
            sharp=arr+self.alpha*(arr-blurred)
            return Image.fromarray(np.clip(sharp*255,0,255).astype(np.uint8))
        return img

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        EdgeSharpen(prob=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

# =============================================================================
# [Module 1] SEBlock — Squeeze-and-Excitation Channel Attention
# =============================================================================
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16, min_bottleneck=8):
        super().__init__()
        hidden = max(in_channels // reduction, min_bottleneck)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# =============================================================================
# Generator with SENet 1-Layer — SEBlock only at L3(16×16, 64ch)
# =============================================================================
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(nd,256,4), nn.BatchNorm2d(256), nn.ReLU())
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU())
        self.se3 = SEBlock(64, reduction=SE_REDUCTION, min_bottleneck=SE_MIN_BOTTLENECK)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU())
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh())

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.se3(self.up3(x))
        x = self.up4(x)
        x = self.up5(x)
        return x

# =============================================================================
# Discriminator (model64.py — UNCHANGED)
# =============================================================================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,32,3,2,1),nn.LeakyReLU(0.2),
            nn.Conv2d(32,64,3,2,1),nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,3,2,1),nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,3,2,1),nn.LeakyReLU(0.2),
            nn.Flatten(),nn.Linear(4*4*256,256),nn.LeakyReLU(0.2),
            nn.Linear(256,1),nn.Sigmoid())
    def forward(self,x): return self.net(x).view(-1)

# =============================================================================
# [Module 2] Haar DWT + Wavelet Loss
# =============================================================================
def haar_dwt_2d(x):
    lo = torch.tensor([1.0, 1.0], device=x.device) / np.sqrt(2)
    hi = torch.tensor([1.0, -1.0], device=x.device) / np.sqrt(2)
    lo_2d = lo.view(1, 1, 1, 2); hi_2d = hi.view(1, 1, 1, 2)
    b, c, h, w = x.size()
    if h % 2 != 0: x = F.pad(x, (0,0,0,1))
    if w % 2 != 0: x = F.pad(x, (0,1,0,0))
    x = x.reshape(b * c, 1, h, w)
    x_lo = F.conv2d(x, lo_2d, stride=(1, 2))
    x_hi = F.conv2d(x, hi_2d, stride=(1, 2))
    x_lo = x_lo.reshape(b * c, 1, h, -1).transpose(2, 3)
    x_hi = x_hi.reshape(b * c, 1, h, -1).transpose(2, 3)
    pad_h = 1 if x_lo.size(3) % 2 != 0 else 0
    if pad_h: x_lo = F.pad(x_lo, (0, pad_h)); x_hi = F.pad(x_hi, (0, pad_h))
    LL = F.conv2d(x_lo, lo_2d, stride=(1, 2)).transpose(2, 3)
    LH = F.conv2d(x_lo, hi_2d, stride=(1, 2)).transpose(2, 3)
    HL = F.conv2d(x_hi, lo_2d, stride=(1, 2)).transpose(2, 3)
    HH = F.conv2d(x_hi, hi_2d, stride=(1, 2)).transpose(2, 3)
    out = torch.stack([LL, LH, HL, HH], dim=2).view(b, c, 4, -1, LL.size(3))
    return out

def wavelet_loss(fake_imgs, real_imgs):
    fake = (fake_imgs + 1) / 2.0; real = (real_imgs + 1) / 2.0
    fake_dwt = haar_dwt_2d(fake); real_dwt = haar_dwt_2d(real)
    if W_LOSS == "l2":
        loss_ll = F.mse_loss(fake_dwt[:,:,0], real_dwt[:,:,0])
        loss_lh = F.mse_loss(fake_dwt[:,:,1], real_dwt[:,:,1])
        loss_hl = F.mse_loss(fake_dwt[:,:,2], real_dwt[:,:,2])
        loss_hh = F.mse_loss(fake_dwt[:,:,3], real_dwt[:,:,3])
    else:
        loss_ll = F.l1_loss(fake_dwt[:,:,0], real_dwt[:,:,0])
        loss_lh = F.l1_loss(fake_dwt[:,:,1], real_dwt[:,:,1])
        loss_hl = F.l1_loss(fake_dwt[:,:,2], real_dwt[:,:,2])
        loss_hh = F.l1_loss(fake_dwt[:,:,3], real_dwt[:,:,3])
    total = W_LL * loss_ll + W_LH * loss_lh + W_HL * loss_hl + W_HH * loss_hh
    return total * WAVELET_LAMBDA

# =============================================================================
# [Module 3] Laplacian Pyramid Loss
# =============================================================================
def gaussian_pyramid(img, levels):
    pyramid = [img]
    for _ in range(levels):
        img = F.avg_pool2d(img, kernel_size=2, stride=2)
        pyramid.append(img)
    return pyramid

def laplacian_pyramid(gaussian_pyr):
    laplacian = []
    for k in range(len(gaussian_pyr) - 1):
        upsampled = F.interpolate(gaussian_pyr[k + 1],
                                   size=gaussian_pyr[k].shape[2:],
                                   mode="bilinear", align_corners=False)
        lap = gaussian_pyr[k] - upsampled
        laplacian.append(lap)
    return laplacian

def laplacian_loss(fake_imgs, real_imgs):
    fake = (fake_imgs + 1) / 2.0; real = (real_imgs + 1) / 2.0
    fake_gpyr = gaussian_pyramid(fake, PYRAMID_LEVELS)
    real_gpyr = gaussian_pyramid(real, PYRAMID_LEVELS)
    fake_lpyr = laplacian_pyramid(fake_gpyr)
    real_lpyr = laplacian_pyramid(real_gpyr)
    total_loss = 0.0
    for k, (w, fl, rl) in enumerate(zip(PYRAMID_WEIGHTS, fake_lpyr, real_lpyr)):
        total_loss += w * F.l1_loss(fl, rl)
    return total_loss * LAPLACIAN_LAMBDA

# =============================================================================
# Metrics
# =============================================================================
class FIDCalculator:
    def __init__(self,device):
        self.device=device
        inc=models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,transform_input=False)
        inc.fc=nn.Identity(); inc.eval(); self.inc=inc.to(device)
        for p in self.inc.parameters(): p.requires_grad=False
    @torch.no_grad()
    def _feat(self,imgs):
        imgs=(imgs+1)/2.0; imgs=F.interpolate(imgs,size=(299,299),mode="bilinear",align_corners=False)
        imgs=(imgs-0.5)/0.5; return self.inc(imgs).cpu().numpy()
    @torch.no_grad()
    def compute_fid(self,G,real_loader,n=10000):
        G.eval(); rf,ff=[],[]; c=0
        for imgs in real_loader:
            rf.append(self._feat(imgs.to(self.device))); c+=imgs.size(0)
            if c>=n: break
        rf=np.concatenate(rf,axis=0)[:n]; g=0
        while g<n:
            bs=min(64,n-g); noise=torch.randn(bs,NOISE_DIM,1,1,device=self.device)
            ff.append(self._feat(G(noise))); g+=bs
        ff=np.concatenate(ff,axis=0)[:n]
        mr,sr=np.mean(rf,axis=0),np.cov(rf,rowvar=False); mf,sf=np.mean(ff,axis=0),np.cov(ff,rowvar=False)
        d=mr-mf; cm=linalg.sqrtm(sr.dot(sf))
        if np.iscomplexobj(cm): cm=cm.real
        G.train(); return float(d.dot(d)+np.trace(sr+sf-2*cm))

class LPIPSCalculator:
    def __init__(self,device):
        self.device=device; anet=models.alexnet(weights=models.AlexNet_Weights.DEFAULT); anet.eval()
        self.layers=nn.ModuleList([anet.features[:3],anet.features[:6],anet.features[:9],anet.features[:12],anet.features]).to(device)
        for p in self.layers.parameters(): p.requires_grad=False
    def _norm(self,x):
        m=torch.tensor([0.485,0.456,0.406],device=self.device).view(1,3,1,1)
        s=torch.tensor([0.229,0.224,0.225],device=self.device).view(1,3,1,1); return (x-m)/s
    @torch.no_grad()
    def compute_lpips(self,a,b):
        a,b=self._norm(a),self._norm(b); total=0.0
        for L in self.layers: f1,f2=L(a),L(b); total+=(f1-f2).pow(2).mean(dim=[1,2,3])
        return (total/len(self.layers)).cpu().numpy()

@torch.no_grad()
def compute_diversity(G,lpips_calc,ns=500):
    G.eval(); imgs=[]; g=0
    while g<ns:
        bs=min(32,ns-g); noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
        imgs.append((G(noise)+1)/2.0); g+=bs
    imgs=torch.cat(imgs,dim=0)[:ns]; npairs=2000
    i1=torch.randint(0,ns,(npairs,)); i2=torch.randint(0,ns,(npairs,))
    scores=[]
    for i in range(0,npairs,50):
        e=min(i+50,npairs)
        scores.extend(lpips_calc.compute_lpips(imgs[i1[i:e]].to(DEVICE),imgs[i2[i:e]].to(DEVICE)))
    G.train(); return float(np.mean(scores))

def compute_laplacian_variance(imgs):
    vars_=[]
    for img in imgs:
        arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        vars_.append(cv2.Laplacian(gray,cv2.CV_64F).var())
    return float(np.mean(vars_))

def compute_edge_density(imgs, real_imgs=None):
    densities=[]
    for img in imgs:
        arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(gray,50,150); densities.append((edges>0).mean())
    fake_density=float(np.mean(densities))
    if real_imgs is not None:
        real_densities=[]
        for img in real_imgs:
            arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
            edges=cv2.Canny(gray,50,150); real_densities.append((edges>0).mean())
        real_density=float(np.mean(real_densities)); ratio=fake_density/max(real_density,1e-8)
        return fake_density, real_density, ratio
    return fake_density, None, None

# =============================================================================
# Main
# =============================================================================
def main():
    set_all_seeds(SEED); os.makedirs(EXP_DIR,exist_ok=True)

    dataset_path,image_paths=load_dataset()
    if DATASET_LIMIT and len(image_paths)>DATASET_LIMIT:
        set_all_seeds(SEED); image_paths=random.sample(image_paths,DATASET_LIMIT)
        print(f"Subsampled to {DATASET_LIMIT} images (seed={SEED})")

    set_all_seeds(SEED); fixed_noise=torch.randn(64,NOISE_DIM,1,1,device=DEVICE)

    ds=AnimeDataset(image_paths,transform=get_transform())
    dl=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=2)

    g_tmp,d_tmp=Generator(),Discriminator()
    gp=sum(p.numel() for p in g_tmp.parameters())
    dp=sum(p.numel() for p in d_tmp.parameters())
    del g_tmp,d_tmp

    print(f"\n{'='*55}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Status: FAILED — FID=104.93, worse than single-module baselines")
    print(f"  Architecture: Generator + 1-layer SE(L3@16×16,64ch)")
    print(f"  Generator Loss: BCE_adv + {WAVELET_LAMBDA}×Wavelet + {LAPLACIAN_LAMBDA}×Laplacian")
    print(f"  Wavelet: λ={WAVELET_LAMBDA}, LL={W_LL} LH={W_LH} HL={W_HL} HH={W_HH}")
    print(f"  Laplacian: λ={LAPLACIAN_LAMBDA}, levels={PYRAMID_LEVELS}, w={PYRAMID_WEIGHTS}")
    print(f"  Augmentation: Flip(p=0.5) + EdgeSharpen(p=0.2)")
    print(f"  Device: {DEVICE}")
    if DEVICE.type=="cuda": print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  G: {gp:,} params (+{gp-1100707:,} vs baseline) | D: {dp:,} params")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Steps/epoch: {len(dl)}")
    print(f"{'='*55}\n")

    G=Generator().to(DEVICE); D=Discriminator().to(DEVICE); crit=nn.BCELoss()
    g_opt=torch.optim.Adam(G.parameters(),lr=LR,betas=BETAS)
    d_opt=torch.optim.Adam(D.parameters(),lr=LR,betas=BETAS)

    csv_f=open(os.path.join(EXP_DIR,"loss.csv"),"w",newline="")
    w=csv.writer(csv_f)
    w.writerow(["epoch","D_loss","G_loss","G_adv","G_wavelet","G_laplacian","D_real","D_fake"])
    fdl,fgl,fdr,fdf=0.0,0.0,0.0,0.0

    print("Training ...\n")
    for ep in range(1,EPOCHS+1):
        for img in dl:
            real=img.to(DEVICE); bs=real.size(0)
            rl,fl=torch.ones(bs,device=DEVICE),torch.zeros(bs,device=DEVICE)

            # Train D
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fake=G(noise)
            ro,fo=D(real),D(fake.detach())
            d_loss=crit(ro,rl)+crit(fo,fl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # Train G — BCE + Wavelet + Laplacian
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fake=G(noise)
            g_adv  = crit(D(fake), rl)
            g_wav  = wavelet_loss(fake, real)
            g_lap  = laplacian_loss(fake, real)
            g_loss = g_adv + g_wav + g_lap
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v,gl_v,dr_v,df_v = d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        g_adv_v=g_adv.item(); g_wav_v=g_wav.item(); g_lap_v=g_lap.item()
        w.writerow([ep,dl_v,gl_v,g_adv_v,g_wav_v,g_lap_v,dr_v,df_v])
        fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}(adv={g_adv_v:.4f} wav={g_wav_v:.4f} lap={g_lap_v:.4f})  DR:{dr_v:.4f}  DF:{df_v:.4f}")

        if ep%SAMPLE_INTERVAL==0:
            G.eval()
            with torch.no_grad(): samples=G(fixed_noise)
            G.train()
            save_image_grid(samples,os.path.join(EXP_DIR,f"epoch_{ep:03d}.png"))
            torch.save(G.state_dict(),os.path.join(EXP_DIR,f"generator_epoch_{ep:03d}.pth"))

    csv_f.close()
    torch.save(G.state_dict(),os.path.join(EXP_DIR,"generator_final.pth"))
    torch.save(D.state_dict(),os.path.join(EXP_DIR,"discriminator_final.pth"))
    save_image_grid(next(iter(dl))[:64],os.path.join(EXP_DIR,"real_images.png"))

    df=pd.read_csv(os.path.join(EXP_DIR,"loss.csv"))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].plot(df["epoch"],df["G_loss"],color="#e74c3c",lw=1.5)
    axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss (adv+wav+lap)"); axes[0].grid(True,alpha=0.3)
    axes[1].plot(df["epoch"],df["D_loss"],color="#3498db",lw=1.5)
    axes[1].set(xlabel="Epoch",ylabel="Loss",title="Discriminator Loss"); axes[1].grid(True,alpha=0.3)
    axes[2].plot(df["epoch"],df["D_real"],color="#2ecc71",lw=1.5,label="D(Real)")
    axes[2].plot(df["epoch"],df["D_fake"],color="#e67e22",lw=1.5,label="D(Fake)")
    axes[2].axhline(0.5,color="gray",ls="--",alpha=0.4)
    axes[2].set(xlabel="Epoch",ylabel="Mean",title="D(Real) vs D(Fake)"); axes[2].legend(fontsize=8); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,"loss_curves.png"),dpi=150); plt.close()

    print(f"\n{'='*55}\n  Computing Metrics for {EXPERIMENT_NAME}\n{'='*55}")
    eval_tf=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    eval_ds=AnimeDataset(image_paths,transform=eval_tf)
    eval_dl=DataLoader(eval_ds,batch_size=64,shuffle=True,drop_last=True,num_workers=2)

    print("FID ...")
    try: fid=FIDCalculator(DEVICE).compute_fid(G,eval_dl,n=10000); print(f"  FID: {fid:.2f}")
    except Exception as e: fid=-1; print(f"  FID FAILED (network): {e}"); gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

    print("LPIPS + Diversity + Laplacian + Edge Density ...")
    rb,fb=[],[]; generated=0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE)+1)/2.0)
    rb=torch.cat(rb,dim=0)[:500]
    with torch.no_grad():
        while generated<500:
            bs=min(32,500-generated)
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fb.append((G(noise)+1)/2.0); generated+=bs
    fb=torch.cat(fb,dim=0)[:500]

    print("  Laplacian + Edge Density (cv2) ...")
    lap=compute_laplacian_variance(fb[:200])
    fake_edge, real_edge, edge_ratio=compute_edge_density(fb[:200],rb[:200])
    print(f"  LapVar: {lap:.2f}  EdgeRatio: fake={fake_edge:.4f} real={real_edge:.4f} ratio={edge_ratio:.4f}")

    lpips_mean=-1; div=-1
    try:
        print("  LPIPS + Diversity (AlexNet) ...")
        lpips_calc=LPIPSCalculator(DEVICE)
        lpips_scores=[]
        for i in range(0,500,50): lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i+50,500)],rb[i:min(i+50,500)]))
        lpips_mean=float(np.mean(lpips_scores)); div=compute_diversity(G,lpips_calc,ns=300)
        print(f"  LPIPS: {lpips_mean:.4f}  Diversity: {div:.4f}")
        del lpips_calc; gc.collect()
    except Exception as e:
        print(f"  LPIPS/Diversity FAILED (network): {e}")
        gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

    metrics={
        "experiment_name":EXPERIMENT_NAME,"epochs":EPOCHS,"dataset_size":len(image_paths),
        "augmentation":"Flip(p=0.5)+EdgeSharpen(p=0.2)",
        "architecture":"Generator + 1-layer SE(L3@16×16) + Wavelet Loss + Laplacian Pyramid Loss",
        "technique":"SENet 1L(r=16) + Haar DWT(λ=0.1, LL×1/LH×0.5/HL×0.5/HH×0.25) + Laplacian(λ=0.1, 3lev, w=[1,0.5,0.25])",
        "status":"FAILED — FID=104.93 worse than SENet 1L(96.76). Root cause: Wavelet+Laplacian double-penalize same structural errors.",
        "FID":round(fid,2),"LPIPS":round(lpips_mean,4),"Diversity":round(div,4),
        "Laplacian_Variance":round(lap,2),
        "Edge_Density_Fake":round(fake_edge,4),"Edge_Density_Real":round(real_edge,4),
        "Edge_Density_Ratio":round(edge_ratio,4),
        "final_G_loss":round(fgl,4),"final_D_loss":round(fdl,4),
        "D_real":round(fdr,4),"D_fake":round(fdf,4),
        "completed_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(EXP_DIR,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)

    print(f"\n{'='*55}")
    print(f"  Complete: {EXPERIMENT_NAME}")
    print(f"  FID={fid:.2f}  LPIPS={lpips_mean:.4f}  Div={div:.4f}  LapVar={lap:.2f}  EdgeRatio={edge_ratio:.4f}")
    print(f"  Results: {os.path.abspath(EXP_DIR)}/")
    print(f"{'='*55}")

if __name__=="__main__": main()
