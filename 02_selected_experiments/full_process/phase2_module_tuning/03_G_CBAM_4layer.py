"""
================================================================================
 Deep Tuning 03_G_CBAM_4layer: Generator + 4-Layer CBAM | 200 Epochs
================================================================================
 CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018)

 Principle:
   CBAM extends SENet by adding spatial attention AFTER channel attention.

   Stage 1 — Channel Attention (similar to SENet, but dual-pool):
     AvgPool(C,H,W)->(C,1,1) + MaxPool(C,H,W)->(C,1,1)
     → Shared FC(C->C/r->C) → Add → Sigmoid → channel weights
     The dual-pool design captures both average activation AND peak activation
     per channel, providing richer channel descriptors than SENet alone.

   Stage 2 — Spatial Attention (unique to CBAM):
     AvgPool(C,H,W)->(1,H,W) + MaxPool(C,H,W)->(1,H,W)
     → Concat → Conv2d(2->1, 7x7) → Sigmoid → spatial weight map
     This tells the network "WHERE to look" — face regions get high weight,
     background regions get low weight.

 Integration:
   CBAM blocks inserted after each BN+ReLU of Generator's intermediate layers
   (channels 256, 128, 64, 32), same positions as SENet.

 Controlled variables (identical to 00_Baseline):
   - Dataset, Discriminator, Training, Augmentation, Epochs — ALL identical

 Independent variable:
   - Generator: CBAM blocks added (reduction=16, min_bottleneck=8)

 Reference:
   Woo, Park, Lee, Kweon. "CBAM: Convolutional Block Attention Module." ECCV 2018.
   https://arxiv.org/abs/1807.06521

 SENet vs CBAM:
   - SENet: channel attention only ("WHAT matters")
   - CBAM:  channel attention + spatial attention ("WHAT matters" + "WHERE matters")
   - For 64x64 anime faces: spatial attention may help Generator focus on
     face region (~30% of image area) vs background. This is CBAM's advantage.
   - SENet adds ~11K params; CBAM adds ~12K params (extra 7x7 conv in spatial attn)
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
EXPERIMENT_NAME = "03_G_CBAM_4layer"
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"; DATASET_LIMIT=8000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=100; LR=1e-4; BETAS=(0.5,0.99); SEED=42
EPOCHS=200; SAMPLE_INTERVAL=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR=os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

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
# [模块位置] CBAM — Convolutional Block Attention Module
# =============================================================================
# 论文: Woo, Park, Lee, Kweon. "CBAM: Convolutional Block Attention Module." ECCV 2018.
# 链接: https://arxiv.org/abs/1807.06521
#
# === 原理（两个顺序阶段）===
#
# Stage 1 — Channel Attention（通道注意力，SENet的升级版）
#   和 SENet 一样的目标：给每个通道打分。但 CBAM 用双池化：
#     - AvgPool: 每个通道的平均激活水平（类似SENet）
#     - MaxPool: 每个通道的峰值激活（SENet没有的）
#   两个池化结果分别通过同一个 Shared MLP（C->C/r->C），
#   然后 element-wise 相加，过 Sigmoid 得到 0~1 的通道权重。
#   为什么加 MaxPool？AvgPool 只看到"平均"，如果某个通道只有
#   一个很小的区域激活（如眼睛高光），AvgPool 会把它平均掉。
#   MaxPool 能捕捉这种"局部峰值"，补 AvgPool 的盲区。
#
# Stage 2 — Spatial Attention（空间注意力，SENet没有的）
#   对 Stage 1 输出的特征图，沿通道维度做 AvgPool 和 MaxPool，
#   得到两个 (1, H, W) 的空间描述符。拼接成 (2, H, W)，
#   过 7×7 卷积 + Sigmoid，输出 (1, H, W) 的空间权重图。
#   这个权重的含义是"图的每个位置有多重要"——
#   人脸区域（眼睛、鼻子、嘴）高权重，纯色背景低权重。
#   在 64×64 动漫头像中，脸只占 ~30% 的画面，
#   空间注意力能明确告诉 Generator "重点画这里"。
#
# === 为什么在 DCGAN Generator 中用 CBAM ===
# SENet 只回答"哪个通道重要"（WHAT）。
# CBAM 额外回答"哪个位置重要"（WHERE）。
# 对于 64×64 的动漫头像生成——脸占画面比例不大，背景占了大部分像素——
# "位置"信息和"通道"信息同样重要。Generator 如果能把更多算力
# 集中在人脸区域而非背景，生成质量理应更好。
#
# === 实现位置（和 SENet 相同）===
# CBAM 块插入在 Generator 的第 1~4 层转置卷积的 BN+ReLU 之后：
#   Layer 1: ConvTranspose(100→256) + BN + ReLU → CBAM(256)
#   Layer 2: ConvTranspose(256→128) + BN + ReLU → CBAM(128)
#   Layer 3: ConvTranspose(128→64)  + BN + ReLU → CBAM(64)
#   Layer 4: ConvTranspose(64→32)   + BN + ReLU → CBAM(32)
#   Layer 5: ConvTranspose(32→3)    + Tanh              → 不加CBAM（输出层）
#
# === 参数选择 ===
# reduction=16, min_bottleneck=8: 同SENet，防止浅层过度压缩
# spatial_kernel=7: 标准值。7×7 的感受野在 8×8~32×32 的特征图上
#   足以覆盖人脸区域（相对于背景）的尺度差异
#   3×3 太小（看不清人脸 vs 背景），11×11 太大（计算浪费）
#
# === 额外参数量 ===
# Channel Attention（同SENet）: ~11.8K
# Spatial Attention: 每个 CBAM 一个 Conv2d(2,1,7) = 98 参数 × 4 = 392
# 总计增加: ~12.2K 参数（仍 < Generator 的 1.2%）
# =============================================================================

class ChannelAttention(nn.Module):
    """Stage 1: Channel Attention with dual-pool (Avg + Max)."""
    def __init__(self, in_channels, reduction=16, min_bottleneck=8):
        super().__init__()
        hidden = max(in_channels // reduction, min_bottleneck)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # Shared MLP between avg and max branches
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Avg branch
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        # Max branch — captures peak activations that AvgPool would smooth out
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        # Element-wise sum + Sigmoid
        weight = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * weight


class SpatialAttention(nn.Module):
    """Stage 2: Spatial Attention with 7x7 convolution."""
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average along channel axis: "which spatial positions are generally active?"
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        # Max along channel axis: "which positions have peak activation?"
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        # Concat -> 7x7 Conv -> Sigmoid -> spatial weight map
        attn = torch.cat([avg_out, max_out], dim=1)     # (B, 2, H, W)
        attn = self.sigmoid(self.conv(attn))             # (B, 1, H, W)
        return x * attn


class CBAMBlock(nn.Module):
    """
    Convolutional Block Attention Module.
    Sequentially applies Channel Attention then Spatial Attention.

    Input:  (B, C, H, W)
    Output: (B, C, H, W) — same shape, refined features
    """
    def __init__(self, in_channels, reduction=16, min_bottleneck=8, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels, reduction, min_bottleneck)
        self.spatial_attn = SpatialAttention(kernel_size)

    def forward(self, x):
        # Stage 1: What matters? (channel weighting)
        x = self.channel_attn(x)
        # Stage 2: Where matters? (spatial weighting)
        x = self.spatial_attn(x)
        return x


# =============================================================================
# [修改位置] Generator with CBAM — 在中间4层加CBAM Block
#
# 和 Baseline 的区别:
#   +4 个 CBAMBlock 实例 (cbam1~cbam4)
#   +约 12.2K 参数
#   forward() 中每层多了 self.cbamN(...) 调用
# =============================================================================
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()

        # Layer 1: 1x1 -> 4x4, 256 channels
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(nd,256,4), nn.BatchNorm2d(256), nn.ReLU())
        self.cbam1 = CBAMBlock(256)

        # Layer 2: 4x4 -> 8x8, 128 channels
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.cbam2 = CBAMBlock(128)

        # Layer 3: 8x8 -> 16x16, 64 channels
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbam3 = CBAMBlock(64)

        # Layer 4: 16x16 -> 32x32, 32 channels
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU())
        self.cbam4 = CBAMBlock(32)

        # Layer 5: 32x32 -> 64x64, RGB output — no CBAM
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh())

    def forward(self, x):
        x = self.cbam1(self.up1(x))
        x = self.cbam2(self.up2(x))
        x = self.cbam3(self.up3(x))
        x = self.cbam4(self.up4(x))
        x = self.up5(x)
        return x


# =============================================================================
# Discriminator (model64.py — UNCHANGED from baseline)
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
# Metric 1: FID (InceptionV3 pool3)
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

# =============================================================================
# Metric 2-5: LPIPS, Diversity, Laplacian, Edge Density (identical to baseline)
# =============================================================================
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
    gp=sum(p.numel() for p in g_tmp.parameters()); dp=sum(p.numel() for p in d_tmp.parameters())
    del g_tmp,d_tmp
    print(f"\n{'='*55}")
    print(f"  Experiment: {EXPERIMENT_NAME}")
    print(f"  Architecture: Generator + CBAM Blocks (Channel + Spatial Attn)")
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
    w=csv.writer(csv_f); w.writerow(["epoch","D_loss","G_loss","D_real","D_fake"])
    fdl,fgl,fdr,fdf=0.0,0.0,0.0,0.0

    print("Training ...\n")
    for ep in range(1,EPOCHS+1):
        for img in dl:
            real=img.to(DEVICE); bs=real.size(0)
            rl,fl=torch.ones(bs,device=DEVICE),torch.zeros(bs,device=DEVICE)
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fake=G(noise)
            ro,fo=D(real),D(fake.detach())
            d_loss=crit(ro,rl)+crit(fo,fl); d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            g_loss=crit(D(G(noise)),rl); g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        w.writerow([ep,dl_v,gl_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  DR:{dr_v:.4f}  DF:{df_v:.4f}")
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
    axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss"); axes[0].grid(True,alpha=0.3)
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

    print("FID ..."); fid=FIDCalculator(DEVICE).compute_fid(G,eval_dl,n=10000); print(f"  FID: {fid:.2f}"); gc.collect()
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
        "architecture":"Generator + CBAM Blocks (Channel+Spatial Attn)",
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
