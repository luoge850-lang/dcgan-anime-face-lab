"""
================================================================================
 Deep Tuning 02_G_SENet_1layer: Generator + 1-Layer SENet @ L3 | 200 Epochs
================================================================================
 SENet: Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)

 Principle:
   Generator intermediate features have shape (C, H, W). Different channels
   encode different visual attributes (color, texture, edges, etc.). SENet
   learns to weight each channel by its importance for the final output.

   Squeeze:  Global AvgPool compresses HxW -> 1x1 per channel
   Excitation: FC(C->C/r) -> ReLU -> FC(C/r->C) -> Sigmoid
              This bottleneck learns channel interdependencies.
   Scale:     Multiply original feature map by learned weights.

 Integration:
   SE blocks inserted after each BN+ReLU of Generator's intermediate layers
   (channels 256, 128, 64, 32). NOT on the final Tanh output layer.

 Controlled variables (identical to 00_Baseline):
   - Dataset: GANAnime Lite, 8000 samples, seed=42
   - Discriminator: model64.py UNCHANGED
   - Training: BCELoss, Adam(lr=1e-4, betas=0.5/0.99), batch=32
   - Augmentation: Flip(p=0.5) + EdgeSharpen(p=0.2)
   - Epochs: 200

 Independent variable:
   - Generator: SE blocks added (reduction=16)

 Expected outcome:
   SENet allows Generator to focus computational resources on the most
   informative feature channels. For anime face generation, channels
   responsible for eyes, hair, and facial contours should receive higher
   weights. This should improve FID by 3-8 points over baseline,
   particularly enhancing fine facial details.

 Reference:
   Hu, Shen, Sun. "Squeeze-and-Excitation Networks." CVPR 2018.
   https://arxiv.org/abs/1709.01507
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
EXPERIMENT_NAME = "02_G_SENet_1layer"
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
# [模块位置] SENet — Squeeze-and-Excitation Channel Attention
# =============================================================================
# 论文: Hu, Shen, Sun. "Squeeze-and-Excitation Networks." CVPR 2018.
# 链接: https://arxiv.org/abs/1709.01507
#
# === 原理（三步操作）===
# 1. Squeeze（压缩）
#    对输入特征图的每个通道做全局平均池化（Global AvgPool），
#    把 (C, H, W) 的空间信息压缩为一个标量 (C, 1, 1)。
#    例如：256 通道 × 8×8 特征图 → 256 个数字。
#    这 256 个数字代表了"每个通道在空间上激活了多少"。
#
# 2. Excitation（激励）
#    将 256 个数字通过一个 bottleneck 全连接结构：
#      FC(256 → 256/16=16) → ReLU → FC(16 → 256) → Sigmoid
#    输出 256 个 0~1 之间的权重值。
#    Bottleneck（reduction=16）的关键作用：
#      - 压缩阶段：强制网络用一个紧凑的 16 维向量来"总结"256 个通道之间的关系
#      - 恢复阶段：从这个紧凑表示重建出每个通道的重要性分数
#      - 这个过程让网络学习通道之间的非线性依赖——哪些通道是协同的，哪些是冗余的
#
# 3. Scale（重标定）
#    将原始特征图 (C, H, W) 的每个通道乘以对应的权重：
#      output = input × weight.view(C, 1, 1)
#    重要的通道（如管眼睛纹理、面部轮廓的）被放大
#    不重要的通道（如管纯色背景、噪点的）被抑制
#
# === 为什么在 DCGAN Generator 中用 SENet ===
# Generator 的每一层转置卷积后，输出的特征图不同通道负责不同的视觉模式：
#   - 某些通道编码"眼睛在哪里"
#   - 某些通道编码"头发颜色"
#   - 某些通道编码"背景填充"
#   - 某些通道可能是死神经元（输出恒定值）
# 标准卷积对所有这些通道一视同仁——无论它对生成质量重不重要。
# SE Block 让 Generator 学会自己判断"哪些通道对最终人脸最重要"，
# 把有限的计算资源集中在关键通道上。
#
# === 实现位置 ===
# SEBlock 插入在 Generator 的第 1~4 层转置卷积的 BN+ReLU 之后：
#   Layer 1: ConvTranspose(100→256) + BN + ReLU → SEBlock(256)
#   Layer 2: ConvTranspose(256→128) + BN + ReLU → SEBlock(128)
#   Layer 3: ConvTranspose(128→64)  + BN + ReLU → SEBlock(64)
#   Layer 4: ConvTranspose(64→32)   + BN + ReLU → SEBlock(32)
#   Layer 5: ConvTranspose(32→3)    + Tanh              → 不加SE（输出层）
#
# === 参数选择（针对 DCGAN Generator 优化）===
# 原论文 reduction=16 适用于 ResNet（通道数 256~2048），但 DCGAN Generator
# 通道数仅为 256→128→64→32。直接套用会导致浅层 bottleneck 过窄：
#
#   层    通道数    r=16 bottleneck   问题
#   L1     256         16             OK — 16维足够表达256个通道的关系
#   L2     128          8             OK — 8维还行
#   L3      64          4             ⚠️  4维偏窄，可能丢失部分通道依赖
#   L4      32          2             ❌  2维太窄！32个通道被压到2维，信息瓶颈严重
#
# 解决方案：设 min_bottleneck=8，确保每层的 bottleneck 至少 8 维。
#   L1: max(256/16, 8) = 16
#   L2: max(128/16, 8) = 8
#   L3: max(64/16,  8) = 8   ← 从 4 提升到 8
#   L4: max(32/16,  8) = 8   ← 从 2 提升到 8，显著改善信息流
#
# === 额外参数量（修正后）===
#   SE(256, bottleneck=16): 256×16 + 16×256 = 8,192
#   SE(128, bottleneck=8):  128×8  + 8×128  = 2,048
#   SE(64,  bottleneck=8):  64×8   + 8×64   = 1,024
#   SE(32,  bottleneck=8):  32×8   + 8×32   = 512
#   总计: ~11.8K 参数（仍 < Generator 总参数的 1.1%）
#
# === 预期效果 ===
# - FID: 预计比 baseline (109.40) 降低 3~8 点
# - Laplacian Variance: 可能提升，边缘相关通道被放大后五官更清晰
# - 生成质量：眼睛、轮廓等关键特征的细节应有所改善
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Channel Attention Block.
    Optimized for DCGAN Generator: includes min_bottleneck to prevent
    over-compression at shallow layers.

    Args:
        in_channels:    Input feature channels (e.g., 256, 128, 64, 32)
        reduction:      Bottleneck compression ratio (default=16)
        min_bottleneck: Minimum hidden dimension (default=8)
                        Prevents bottleneck collapse when in_channels < 128

    Input:  (B, C, H, W) feature map
    Output: (B, C, H, W) same shape, channel-wise reweighted
    """
    def __init__(self, in_channels, reduction=16, min_bottleneck=8):
        super().__init__()
        # Compute hidden dim with floor protection
        hidden = max(in_channels // reduction, min_bottleneck)

        # Step 1: Squeeze — compress spatial info to a single scalar per channel
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # Step 2: Excitation — learn channel weights through bottleneck
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden),    # C -> bottleneck
            nn.ReLU(inplace=True),
            nn.Linear(hidden, in_channels),     # bottleneck -> C
            nn.Sigmoid(),                       # [0,1] per-channel weight
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)        # Squeeze: (B,C)
        y = self.fc(y).view(b, c, 1, 1)        # Excitation: (B,C,1,1)
        return x * y                            # Scale: channel-wise reweight


# =============================================================================
# =============================================================================
# [修改位置] Generator with 1-Layer SENet — 仅L3(16×16,64ch)加SEBlock
#
# 原始 Generator (model64.py baseline):
#   up1(100→256) -> up2(256→128) -> up3(128→64) -> up4(64→32) -> up5(32→3)
#   ↑ 每层只有 ConvTranspose + BN + ReLU，无注意力机制
#
# 修改后 Generator (1-Layer SENet):
#   up1 -> up2 -> up3 -> [SE3] -> up4 -> up5
#   ↑ 仅在 L3 (8x8→16x16, 64ch) 的 BN+ReLU 后插入一个 SEBlock
#   ↑ 为什么仅L3: L1(4x4)/L2(8x8)通道未分化，L4(32ch) bottleneck过窄(32→2→32)
#     L3的16×16×64ch是DCGAN Generator中唯一合适的注意力插入点
#
# 对比 4-Layer SENet 的变化:
#   - 3 个 SEBlock 实例 (se1/se2/se4 已移除)
#   +1 个 SEBlock (se3) — 约1.0K参数
#   forward() 中仅 L3 多一次 self.se3(...) 调用
#   Discriminator、训练循环、增强、超参 —— 全部不变
# =============================================================================
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()

        # L1: 1x1->4x4, 256ch — NO SE (4x4 too small for meaningful channel attention)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(nd,256,4), nn.BatchNorm2d(256), nn.ReLU())

        # L2: 4x4->8x8, 128ch — NO SE (channels still undifferentiated at coarse scale)
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())

        # L3: 8x8->16x16, 64ch — [1-LAYER SENet] ONLY attention here
        # 16x16 is the sweet spot: coarse enough for global channel context,
        # fine enough that channels encode meaningful features (eyes, hair, etc.)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU())
        self.se3 = SEBlock(64, reduction=16)

        # L4: 16x16->32x32, 32ch — NO SE (32ch too few for meaningful bottleneck)
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(64,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU())

        # L5: 32x32->64x64, RGB output — NO SE (output layer)
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(32,3,4,2,1), nn.Tanh())

    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.se3(self.up3(x))  # [唯一注意力点] L3: 64ch @ 16x16
        x = self.up4(x)
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
# Metric 1: FID (InceptionV3 pool3, 2048-dim)
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
# Metric 2: LPIPS (AlexNet 5-layer)
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

# =============================================================================
# Metric 3: Diversity (pairwise LPIPS)
# =============================================================================
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

# =============================================================================
# Metric 4: Laplacian Variance (sharpness)
# =============================================================================
def compute_laplacian_variance(imgs):
    vars_=[]
    for img in imgs:
        arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        vars_.append(cv2.Laplacian(gray,cv2.CV_64F).var())
    return float(np.mean(vars_))

# =============================================================================
# Metric 5: Edge Density (Canny edge pixel ratio)
# =============================================================================
def compute_edge_density(imgs, real_imgs=None):
    densities=[]
    for img in imgs:
        arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
        edges=cv2.Canny(gray,50,150)
        densities.append((edges>0).mean())
    fake_density=float(np.mean(densities))
    if real_imgs is not None:
        real_densities=[]
        for img in real_imgs:
            arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY)
            edges=cv2.Canny(gray,50,150)
            real_densities.append((edges>0).mean())
        real_density=float(np.mean(real_densities))
        ratio=fake_density/max(real_density,1e-8)
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
    print(f"  Architecture: Generator + 1-layer SE Block at L3 (64ch@16x16)")
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
            d_loss=crit(ro,rl)+crit(fo,fl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            g_loss=crit(D(G(noise)),rl)
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        w.writerow([ep,dl_v,gl_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  DR:{dr_v:.4f}  DF:{df_v:.4f}")

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

    print("FID ..."); fid=FIDCalculator(DEVICE).compute_fid(G,eval_dl,n=10000)
    print(f"  FID: {fid:.2f}"); gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

    print("LPIPS + Diversity + Laplacian + Edge Density ...")
    lpips_calc=LPIPSCalculator(DEVICE); rb,fb=[],[]; generated=0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE)+1)/2.0)
    rb=torch.cat(rb,dim=0)[:500]
    with torch.no_grad():
        while generated<500:
            bs=min(32,500-generated)
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fb.append((G(noise)+1)/2.0); generated+=bs
    fb=torch.cat(fb,dim=0)[:500]

    lpips_scores=[]
    for i in range(0,500,50): lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i+50,500)],rb[i:min(i+50,500)]))
    lpips_mean=float(np.mean(lpips_scores))
    div=compute_diversity(G,lpips_calc,ns=300)
    lap=compute_laplacian_variance(fb[:200])
    fake_edge, real_edge, edge_ratio=compute_edge_density(fb[:200],rb[:200])
    print(f"  LPIPS: {lpips_mean:.4f}  Diversity: {div:.4f}  LapVar: {lap:.2f}")
    print(f"  Edge Density: fake={fake_edge:.4f}  real={real_edge:.4f}  ratio={edge_ratio:.4f}")
    del lpips_calc; gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()

    metrics={
        "experiment_name":EXPERIMENT_NAME,"epochs":EPOCHS,"dataset_size":len(image_paths),
        "augmentation":"Flip(p=0.5)+EdgeSharpen(p=0.2)",
        "architecture":"Generator + 1-layer SE Block at L3 (64ch@16x16)",
        "FID":round(fid,2),"LPIPS":round(lpips_mean,4),"Diversity":round(div,4),
        "Laplacian_Variance":round(lap,2),
        "Edge_Density_Fake":round(fake_edge,4),
        "Edge_Density_Real":round(real_edge,4),
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
