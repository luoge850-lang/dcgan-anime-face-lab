"""
================================================================================
 Deep Tuning 06_G_FFT: BCE + FFT Frequency Loss | 200 Epochs
================================================================================
 FFT: Fast Fourier Transform for frequency-domain supervision.

 Principle:
   Standard DCGAN only uses BCE adversarial loss — the Generator never
   receives direct feedback about the frequency content of its output.
   Blurry images lack high frequencies; noisy images have excess high
   frequencies. The Discriminator may or may not catch this.

   FFT loss adds direct frequency-domain supervision:
     1. Apply 2D FFT to both real and generated images
     2. Take the magnitude spectrum (|FFT|)
     3. Apply log scaling: log(1 + magnitude) to balance frequency bands
        (without log, the DC component dominates 99% of the loss)
     4. Compute L1 loss between real and fake log-magnitude spectra
     5. Add to Generator's training objective

   Why log scaling is essential:
     The DC component (frequency 0, image mean brightness) is typically
     100-1000x larger than high-frequency components. Without log scaling,
     the FFT loss would only care about "is the average brightness right?"
     and completely ignore texture/sharpness. Log compresses the dynamic
     range so ALL frequency bands contribute to the loss.

   Why FFT vs Wavelet:
     FFT analyzes the ENTIRE frequency spectrum at once — every frequency
     from the global structure (low freq) to the finest texture (high freq).
     Wavelet splits into 4 discrete bands. FFT is more granular — it can
     detect and penalize specific frequency deficiencies, e.g., "this
     generated image is missing the 0.3-0.5 Nyquist range that real images
     have." Wavelet can't isolate specific frequency ranges within a band.

   Why magnitude-only (not phase):
     Phase encodes position information (where edges are located). Magnitude
     encodes strength information (how sharp edges are). For DCGAN training,
     edge strength (sharpness) matters more than exact edge position —
     the adversarial loss already handles position alignment.

 Integration:
   Generator loss = BCE(G(z), real_label) + λ * FFT_L1(G(z), real)
   λ = 0.05 (FFT loss values are larger than wavelet, needs smaller weight)
   Log scaling: log(1 + |FFT|) applied to both real and fake spectra

 Controlled variables (identical to 00_Baseline):
   - Dataset, Architecture, Discriminator, Augmentation — ALL identical
   - ONLY change: Generator loss includes FFT frequency term

 Expected improvement:
   - FID: 3-6 points lower than baseline (109.40 -> ~103-106)
   - Laplacian Variance: should increase (high-freq penalty pushes sharpness)
   - Primary benefit: corrects frequency imbalances that BCE alone misses
   - vs Wavelet: FFT is more fine-grained (continuous spectrum vs 4 bands)
     but more sensitive to noise. Wavelet is coarser but more robust.

 Key risk:
   - FFT loss is global — it compares the entire spectrum. If the Generator
     shifts a face by 1 pixel, the FFT magnitude barely changes (translation
     invariance of magnitude). This is actually GOOD — it means the FFT loss
     doesn't penalize small positional shifts, only frequency content.

 Reference:
   - Focal Frequency Loss, Jiang et al., ICCV 2021
   - torch.fft.fft2 — PyTorch built-in, zero external dependencies
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
EXPERIMENT_NAME = "06_G_FFT"
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"; DATASET_LIMIT=8000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=100; LR=1e-4; BETAS=(0.5,0.99); SEED=42
EPOCHS=200; SAMPLE_INTERVAL=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR=os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

# FFT loss hyperparameters — tuned for 64x64 anime faces
FFT_LAMBDA = 0.05          # Overall FFT loss weight (smaller than wavelet — FFT values are larger)
FFT_LOG_SCALE = True        # log(1+|FFT|) — critical for balancing low/high frequency contributions

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
# [核心模块] 2D FFT Frequency Loss
# =============================================================================
# 原理: 对图像做 2D FFT → 取幅度谱 → log 压缩 → L1 Loss
#
# 为什么需要 log 压缩:
#   图像的 FFT 幅度谱动态范围极大——DC 分量（频率0）的值通常是高频分量的
#   100-1000 倍。如果不做 log 压缩，Loss 完全由 DC 分量主导——
#   "平均亮度对不对"占据了 99% 的梯度，"纹理够不够锐"几乎没有信号。
#   log(1+|FFT|) 压缩了动态范围，让低频和高频对 Loss 的贡献更均衡。
#
# 为什么取幅度不取相位:
#   幅度谱 = "这个频率有多强" → 决定图像的清晰度和纹理
#   相位谱 = "这个频率出现在哪个位置" → 决定图像的空间结构
#   对于 DCGAN 训练，我们主要关心"Generator 是否生成了足够强的高频"
#   （=图够不够清晰），而不是"高频出现在哪个像素位置"。
#   相位由对抗 Loss 隐式处理——D 会惩罚位置不对的特征。
#
# 为什么不用 Focal Frequency Loss (ICCV 2021):
#   FFL 比基础 FFT Loss 多了一步"动态加权"——对生成困难的频率成分
#   给予更高权重。但 FFL 需要调额外超参数(alpha)且计算量更大。
#   基础 FFT Loss + log 压缩已经能覆盖 64x64 DCGAN 的需求。
#   FFL 更适合高清图像（256+），对 64px 来说过度设计。
#
# 和 Wavelet 的对比:
#   FFT: 连续频谱，精细到每个频率分量 → 更敏感，但更易受噪声影响
#   Wavelet: 4 个离散子带，粗粒度 → 更鲁棒，但可能漏掉特定频率缺陷
# =============================================================================

def fft_loss(fake_imgs, real_imgs):
    """
    Compute log-magnitude FFT loss between fake and real images.

    fake_imgs, real_imgs: (B, 3, 64, 64) in [-1, 1]
    Returns: scalar loss
    """
    # Denormalize: [-1,1] -> [0,1]
    fake = (fake_imgs + 1) / 2.0
    real = (real_imgs + 1) / 2.0

    # 2D FFT per image, per channel
    # rfft2 returns half the spectrum (Hermitian symmetry — the other half is conjugate)
    # Shape: (B, C, H, W//2+1) for real input
    fake_fft = torch.fft.rfft2(fake, norm="ortho")
    real_fft = torch.fft.rfft2(real, norm="ortho")

    # Magnitude spectrum: |FFT| = sqrt(real^2 + imag^2)
    fake_mag = torch.abs(fake_fft)
    real_mag = torch.abs(real_fft)

    # Log scaling: compress dynamic range so high frequencies contribute
    if FFT_LOG_SCALE:
        fake_mag = torch.log1p(fake_mag)   # log(1 + x), numerically stable
        real_mag = torch.log1p(real_mag)

    # L1 loss between spectra
    loss = F.l1_loss(fake_mag, real_mag)

    return loss * FFT_LAMBDA


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
# Main — ONLY change: fft_loss added to Generator objective
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
    print(f"  Technique: FFT Frequency Loss (lambda={FFT_LAMBDA}, log_scale={FFT_LOG_SCALE})")
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
    w=csv.writer(csv_f); w.writerow(["epoch","D_loss","G_loss","G_adv","G_fft","D_real","D_fake"])
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

            # ===== Train Generator — [修改点] 加入 FFT Frequency Loss =====
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fake=G(noise)
            g_adv=crit(D(fake),rl)                  # Adversarial loss (original)
            g_fft=fft_loss(fake, real)               # FFT frequency loss (NEW)
            g_loss=g_adv + g_fft                     # Combined objective
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        g_adv_v=g_adv.item(); g_fft_v=g_fft.item()
        w.writerow([ep,dl_v,gl_v,g_adv_v,g_fft_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}(adv={g_adv_v:.4f} fft={g_fft_v:.4f})  DR:{dr_v:.4f}  DF:{df_v:.4f}")

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
    axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss (adv+fft)"); axes[0].grid(True,alpha=0.3)
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
        "technique":"FFT Frequency Loss (lambda=0.05, log_scale=True)",
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
