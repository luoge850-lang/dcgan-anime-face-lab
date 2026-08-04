"""
================================================================================
 Deep Tuning 05_G_Wavelet: BCE + Haar DWT Loss | 200 Epochs
================================================================================
 Wavelet: Discrete Wavelet Transform (DWT) for multi-frequency supervision.

 Principle:
   Standard DCGAN only uses BCE adversarial loss — the Generator never
   directly sees "what a real face looks like" at the pixel level. It only
   gets indirect feedback through the Discriminator's gradients.

   Wavelet loss adds DIRECT structural supervision:
     1. Apply Haar DWT to both real and generated images
     2. Decompose into 4 sub-bands at half resolution:
        LL: Low-Low (coarse structure, face shape, color distribution)
        LH: Low-High (horizontal edges — eye lines, mouth line)
        HL: High-Low (vertical edges — face contour, nose bridge)
        HH: High-High (diagonal texture — hair details)
     3. Compute weighted L1/L2 loss between corresponding sub-bands
     4. Add to Generator's training objective

   Why it works:
     Adversarial loss tells G "be more convincing" — but not HOW.
     Wavelet loss tells G "match these specific frequency components."
     The LL band loss directly forces face structure to match real faces.
     The LH/HL band losses force edges and contours to be sharp.
     This is complementary supervision that BCE alone cannot provide.

   Why Haar wavelet specifically:
     - Simplest wavelet basis (just averages and differences of 2x2 blocks)
     - Computationally trivial — no learned parameters
     - Perfectly sufficient for 64x64 anime faces
     - Implemented in pure PyTorch — zero external dependencies

 Integration:
   Generator loss = BCE(G(z), real_label) + λ * wavelet_L1(G(z), real)
   λ = 0.1 (recommended starting point)
   Sub-band weights: LL=1.0, LH=0.5, HL=0.5, HH=0.25

 Controlled variables (identical to 00_Baseline):
   - Dataset, Architecture, Discriminator, Augmentation — ALL identical
   - ONLY change: Generator loss includes wavelet auxiliary term

 Expected improvement:
   - FID: 3-6 points lower than baseline (109.40 → ~103-106)
   - Laplacian Variance: should increase (edge sub-bands force sharper contours)
   - Edge Density: should approach 1.0 more closely
   - Primary benefit: better facial structure and edge sharpness

 Reference:
   - Mallat, "A Theory for Multiresolution Signal Decomposition." IEEE TPAMI 1989.
   - Wavelet-based losses proven effective in pix2pixHD, SRGAN, etc.
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
EXPERIMENT_NAME = "05_G_Wavelet"
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"; DATASET_LIMIT=8000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=100; LR=1e-4; BETAS=(0.5,0.99); SEED=42
EPOCHS=200; SAMPLE_INTERVAL=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR=os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

# Wavelet loss hyperparameters
WAVELET_LAMBDA = 0.1       # Overall wavelet loss weight
W_LL  = 1.0                # LL band weight (structure — most important)
W_LH  = 0.5                # LH band weight (horizontal edges)
W_HL  = 0.5                # HL band weight (vertical edges)
W_HH  = 0.25               # HH band weight (diagonal texture — least important)
W_LOSS = "l1"              # "l1" or "l2" — L2 more sensitive to structural errors

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
# Augmentation: Flip(p=0.5) + EdgeSharpen(p=0.2) — Identical to baseline
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
# [核心模块] Haar Discrete Wavelet Transform (DWT)
# =============================================================================
# 原理: 对图像 (B,C,H,W) 的每个通道做一次 2D Haar 小波分解。
#   1. 沿 rows 做 low-pass / high-pass 滤波
#   2. 沿 cols 做 low-pass / high-pass 滤波
#   3. 组合产生 4 个子带: LL, LH, HL, HH（每个 H/2 × W/2）
#
# 实现: 纯 PyTorch 操作，无外部依赖。2D Haar DWT 的核心就是
#   两次卷积——先用 [1,1]/√2 和 [1,-1]/√2 沿行滤波，
#   再用同样滤波器沿列滤波。
#
# 为什么在 DCGAN 训练中用 Wavelet Loss:
#   BCE Loss 只告诉 Generator "更像真的"——但不告诉它哪里不像。
#   Wavelet Loss 直接在频率域对比真图和假图的结构差异：
#     LL 差异大 → 脸型、五官位置不对 → Generator 被显式惩罚
#     LH/HL 差异大 → 轮廓模糊、边缘不锐 → Generator 被显式惩罚
#   这是 BCE 无法提供的"结构化监督信号"
#
# 为什么用 Haar 不用 Daubechies/Bior:
#   Haar 最简单（2×2 块的平均和差），计算零成本。
#   64×64 动漫头像不需要更复杂的小波基——Haar 已经够用。
#   更高的阶数引入更长滤波器，对 64px 图像反而是过度平滑。
# =============================================================================

def haar_dwt_2d(x):
    """
    2D Haar Discrete Wavelet Transform.
    Input:  (B, C, H, W)  — image tensor (assumed H=W)
    Output: (B, C, 4, H//2, W//2) — LL, LH, HL, HH sub-bands
    """
    # Low-pass and high-pass Haar filters
    lo = torch.tensor([1.0, 1.0], device=x.device) / np.sqrt(2)
    hi = torch.tensor([1.0, -1.0], device=x.device) / np.sqrt(2)

    # Reshape as 2D convolution kernels: (out_ch, in_ch, kernel_h, kernel_w)
    lo_2d = lo.view(1, 1, 1, 2)    # row filter: (1,1,1,2)
    hi_2d = hi.view(1, 1, 1, 2)

    b, c, h, w = x.size()

    # Pad if odd dimensions
    if h % 2 != 0: x = F.pad(x, (0,0,0,1))
    if w % 2 != 0: x = F.pad(x, (0,1,0,0))

    # Apply along rows (width), then permute, apply along columns (height)
    x = x.reshape(b * c, 1, h, w)

    # Row transform: low-pass and high-pass
    x_lo = F.conv2d(x, lo_2d, stride=(1, 2))   # (B*C, 1, H, W/2)
    x_hi = F.conv2d(x, hi_2d, stride=(1, 2))

    # Column transform on each
    x_lo = x_lo.reshape(b * c, 1, h, -1).transpose(2, 3)  # -> (B*C, 1, W/2, H)
    x_hi = x_hi.reshape(b * c, 1, h, -1).transpose(2, 3)

    pad_h = 1 if x_lo.size(3) % 2 != 0 else 0
    if pad_h: x_lo = F.pad(x_lo, (0, pad_h)); x_hi = F.pad(x_hi, (0, pad_h))

    LL = F.conv2d(x_lo, lo_2d, stride=(1, 2)).transpose(2, 3)  # (B*C, 1, H/2, W/2)
    LH = F.conv2d(x_lo, hi_2d, stride=(1, 2)).transpose(2, 3)
    HL = F.conv2d(x_hi, lo_2d, stride=(1, 2)).transpose(2, 3)
    HH = F.conv2d(x_hi, hi_2d, stride=(1, 2)).transpose(2, 3)

    # Stack: (B, C, 4, H/2, W/2)
    out = torch.stack([LL, LH, HL, HH], dim=2).view(b, c, 4, -1, LL.size(3))
    return out


def wavelet_loss(fake_imgs, real_imgs):
    """
    Compute weighted multi-scale wavelet loss between fake and real.
    Denormalizes from [-1,1] to [0,1] before DWT.

    fake_imgs, real_imgs: (B, 3, 64, 64) in [-1, 1]
    Returns: scalar loss
    """
    # Denormalize: [-1,1] -> [0,1]
    fake = (fake_imgs + 1) / 2.0
    real = (real_imgs + 1) / 2.0

    # DWT decomposition
    fake_dwt = haar_dwt_2d(fake)  # (B, 3, 4, 32, 32)
    real_dwt = haar_dwt_2d(real)

    # Per-sub-band loss
    # LL (index 0): structure — use L2 for sensitivity to shape errors
    if W_LOSS == "l2":
        loss_ll = F.mse_loss(fake_dwt[:, :, 0], real_dwt[:, :, 0])
        loss_lh = F.mse_loss(fake_dwt[:, :, 1], real_dwt[:, :, 1])
        loss_hl = F.mse_loss(fake_dwt[:, :, 2], real_dwt[:, :, 2])
        loss_hh = F.mse_loss(fake_dwt[:, :, 3], real_dwt[:, :, 3])
    else:
        loss_ll = F.l1_loss(fake_dwt[:, :, 0], real_dwt[:, :, 0])
        loss_lh = F.l1_loss(fake_dwt[:, :, 1], real_dwt[:, :, 1])
        loss_hl = F.l1_loss(fake_dwt[:, :, 2], real_dwt[:, :, 2])
        loss_hh = F.l1_loss(fake_dwt[:, :, 3], real_dwt[:, :, 3])

    total = W_LL * loss_ll + W_LH * loss_lh + W_HL * loss_hl + W_HH * loss_hh
    return total * WAVELET_LAMBDA


# =============================================================================
# Models (model64.py — UNCHANGED from baseline)
# =============================================================================
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(nd,256,4),nn.BatchNorm2d(256),nn.ReLU(),
            nn.ConvTranspose2d(256,128,4,2,1),nn.BatchNorm2d(128),nn.ReLU(),
            nn.ConvTranspose2d(128,64,4,2,1),nn.BatchNorm2d(64),nn.ReLU(),
            nn.ConvTranspose2d(64,32,4,2,1),nn.BatchNorm2d(32),nn.ReLU(),
            nn.ConvTranspose2d(32,3,4,2,1),nn.Tanh())
    def forward(self,x): return self.net(x)

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
# Metrics (identical to baseline)
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
        for imgs in real_loader: rf.append(self._feat(imgs.to(self.device))); c+=imgs.size(0)
        rf=np.concatenate(rf,axis=0)[:n]; g=0
        while g<n: bs=min(64,n-g); noise=torch.randn(bs,NOISE_DIM,1,1,device=self.device); ff.append(self._feat(G(noise))); g+=bs
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
    while g<ns: bs=min(32,ns-g); noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); imgs.append((G(noise)+1)/2.0); g+=bs
    imgs=torch.cat(imgs,dim=0)[:ns]; npairs=2000; i1=torch.randint(0,ns,(npairs,)); i2=torch.randint(0,ns,(npairs,))
    scores=[]
    for i in range(0,npairs,50): e=min(i+50,npairs); scores.extend(lpips_calc.compute_lpips(imgs[i1[i:e]].to(DEVICE),imgs[i2[i:e]].to(DEVICE)))
    G.train(); return float(np.mean(scores))

def compute_laplacian_variance(imgs):
    vars_=[]
    for img in imgs: arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8); gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); vars_.append(cv2.Laplacian(gray,cv2.CV_64F).var())
    return float(np.mean(vars_))

def compute_edge_density(imgs, real_imgs=None):
    densities=[]
    for img in imgs: arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8); gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(gray,50,150); densities.append((edges>0).mean())
    fake_density=float(np.mean(densities))
    if real_imgs is not None:
        real_densities=[]
        for img in real_imgs: arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8); gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(gray,50,150); real_densities.append((edges>0).mean())
        real_density=float(np.mean(real_densities)); ratio=fake_density/max(real_density,1e-8)
        return fake_density, real_density, ratio
    return fake_density, None, None

# =============================================================================
# Main — ONLY change: wavelet_loss added to Generator objective
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
    gp=sum(p.numel() for p in g_tmp.parameters()); dp=sum(p.numel() for p in d_tmp.parameters())
    del g_tmp,d_tmp
    print(f"\n{'='*55}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Technique: Haar Wavelet Loss (lambda={WAVELET_LAMBDA}, weights LL={W_LL} LH={W_LH} HL={W_HL} HH={W_HH})")
    print(f"  Augmentation: Flip(p=0.5) + EdgeSharpen(p=0.2)")
    print(f"  Device: {DEVICE}")
    if DEVICE.type=="cuda": print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  G: {gp:,} params | D: {dp:,} params | Images: {len(image_paths)}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Steps/epoch: {len(dl)}")
    print(f"{'='*55}\n")

    G=Generator().to(DEVICE); D=Discriminator().to(DEVICE); crit=nn.BCELoss()
    g_opt=torch.optim.Adam(G.parameters(),lr=LR,betas=BETAS)
    d_opt=torch.optim.Adam(D.parameters(),lr=LR,betas=BETAS)
    csv_f=open(os.path.join(EXP_DIR,"loss.csv"),"w",newline="")
    w=csv.writer(csv_f); w.writerow(["epoch","D_loss","G_loss","G_adv","G_wavelet","D_real","D_fake"])
    fdl,fgl,fdr,fdf=0.0,0.0,0.0,0.0

    print("Training ...\n")
    for ep in range(1,EPOCHS+1):
        for img in dl:
            real=img.to(DEVICE); bs=real.size(0)
            rl,fl=torch.ones(bs,device=DEVICE),torch.zeros(bs,device=DEVICE)

            # ===== Train Discriminator (unchanged) =====
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fake=G(noise)
            ro,fo=D(real),D(fake.detach())
            d_loss=crit(ro,rl)+crit(fo,fl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # ===== Train Generator — [修改点] 加入 Wavelet Loss =====
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fake=G(noise)
            g_adv=crit(D(fake),rl)                        # Adversarial loss (original)
            g_wave=wavelet_loss(fake, real)               # Wavelet structural loss (NEW)
            g_loss=g_adv + g_wave                         # Combined objective
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        g_adv_v=g_adv.item(); g_wave_v=g_wave.item()
        w.writerow([ep,dl_v,gl_v,g_adv_v,g_wave_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}(adv={g_adv_v:.4f} wav={g_wave_v:.4f})  DR:{dr_v:.4f}  DF:{df_v:.4f}")

        if ep%SAMPLE_INTERVAL==0:
            G.eval()
            with torch.no_grad(): samples=G(fixed_noise)
            G.train(); save_image_grid(samples,os.path.join(EXP_DIR,f"epoch_{ep:03d}.png"))
            torch.save(G.state_dict(),os.path.join(EXP_DIR,f"generator_epoch_{ep:03d}.pth"))

    csv_f.close()
    torch.save(G.state_dict(),os.path.join(EXP_DIR,"generator_final.pth"))
    torch.save(D.state_dict(),os.path.join(EXP_DIR,"discriminator_final.pth"))
    save_image_grid(next(iter(dl))[:64],os.path.join(EXP_DIR,"real_images.png"))

    df=pd.read_csv(os.path.join(EXP_DIR,"loss.csv"))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].plot(df["epoch"],df["G_loss"],color="#e74c3c",lw=1.5)
    axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss (adv+wavelet)"); axes[0].grid(True,alpha=0.3)
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
    lpips_calc=LPIPSCalculator(DEVICE); rb,fb=[],[]; generated=0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE)+1)/2.0)
    rb=torch.cat(rb,dim=0)[:500]
    with torch.no_grad():
        while generated<500: bs=min(32,500-generated); noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fb.append((G(noise)+1)/2.0); generated+=bs
    fb=torch.cat(fb,dim=0)[:500]
    lpips_scores=[]
    for i in range(0,500,50): lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i+50,500)],rb[i:min(i+50,500)]))
    lpips_mean=float(np.mean(lpips_scores)); div=compute_diversity(G,lpips_calc,ns=300); lap=compute_laplacian_variance(fb[:200])
    fake_edge, real_edge, edge_ratio=compute_edge_density(fb[:200],rb[:200])
    print(f"  LPIPS: {lpips_mean:.4f}  Diversity: {div:.4f}  LapVar: {lap:.2f}")
    print(f"  Edge Density: fake={fake_edge:.4f}  real={real_edge:.4f}  ratio={edge_ratio:.4f}")
    del lpips_calc; gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

    metrics={"experiment_name":EXPERIMENT_NAME,"epochs":EPOCHS,"dataset_size":len(image_paths),
        "augmentation":"Flip(p=0.5)+EdgeSharpen(p=0.2)",
        "technique":"Haar Wavelet Loss (lambda=0.1, LL=1.0 LH=0.5 HL=0.5 HH=0.25)",
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
