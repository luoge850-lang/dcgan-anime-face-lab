"""
================================================================================
 Deep Tuning 07_G_Canny: BCE + Canny Edge Loss | 200 Epochs
================================================================================
 Canny: Multi-stage edge detection for structural supervision.

 Principle:
   Canny edge detector extracts binary edge maps from images through four stages:
     1. Gaussian blur (sigma=1.4) — smooth noise
     2. Sobel gradient (magnitude + direction)
     3. Non-Maximum Suppression — thin edges to 1px
     4. Hysteresis thresholding (low=50, high=150) — connect edge fragments

   For DCGAN training, Canny edge maps provide explicit STRUCTURAL supervision:
     - The Generator's BCE loss only gets indirect "real/fake" feedback
     - Canny loss tells G directly: "these edges MUST be here"
     - Eyes, mouth, face contour — all have strong Canny edges
     - The edge map is a binary mask — it tells G WHERE the important pixels are

   Integration:
     1. Before training: run Canny on ALL real images, save edge maps (CPU, once)
        BUT — with augmentation (Flip/Sharpen), the real images change each epoch!
        So we compute Canny on-the-fly during training.
     2. During G training: run Canny on fake and real images
     3. Compute L1 loss between edge maps: L1(Canny(fake), Canny(real))
     4. Add to Generator loss with weight lambda=0.1

   Why NOT precompute: Augmentation (Flip+Sharpen) changes real images each epoch.
   If we precomputed edges from raw images, the edge loss would be out of sync
   with the augmented images the Generator actually sees.

   Canny is NOT differentiable (thresholds break autograd).
   Solution: use Canny output as a detached weight mask for L1 loss.
   The gradient flows through the L1 between raw pixel values weighted by edge locations.
   Or simpler: just use L1 loss between generated image edges and real image edges
   as a pure auxiliary loss term. The Generator gets structural feedback without
   Canny needing to be differentiable.

   Reference:
     Canny, "A Computational Approach to Edge Detection." IEEE TPAMI 1986.
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
EXPERIMENT_NAME = "07_G_Canny"
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"; DATASET_LIMIT=8000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=100; LR=1e-4; BETAS=(0.5,0.99); SEED=42
EPOCHS=200; SAMPLE_INTERVAL=50
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR=os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

# Canny loss hyperparameters
CANNY_LAMBDA = 0.1       # Edge loss weight
CANNY_LOW  = 50           # Hysteresis low threshold
CANNY_HIGH = 150          # Hysteresis high threshold

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
# Augmentation (identical to baseline)
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
# [核心模块] Canny Edge Loss
# =============================================================================
# 原理: Canny 边缘检测提取图像的边缘图（0=非边缘, 1=边缘）。
#   对真图和假图各跑一次 Canny，比较两张边缘图的差异。
#   Generator 被强迫生成和真图相同边缘位置/密度的输出。
#
# 为什么 Canny 不参与反向传播（不可微）却还能有效:
#   Canny Loss 加到 Generator 的 Loss 里——G 在更新参数时，
#   不仅被 BCE Loss（"更像真的"）驱动，还被边缘 L1（"边缘要画对位置"）驱动。
#   Canny 边缘图本身不需要梯度——它只是告诉 G "哪里重要"。
#   真正的梯度来自边缘图的 L1 Loss——G 要生成边缘位置和真图匹配的图。
#
# 为什么阈值选 (50, 150):
#   这是 Canny 的经典默认值。低于 50 的梯度不是边缘（过滤噪声），
#   高于 150 的梯度一定是边缘（强制保留）。50-150 之间的弱边缘
#   只有连接到强边缘时才保留——避免断裂的边缘碎片。
#   对 64x64 动漫头像：这个范围恰好捕捉五官轮廓但忽略浅色渐变。
#
# 和 Edge Density 指标的区别:
#   Edge Density 只统计边缘像素占比——是"量"。
#   Canny Loss 比较边缘图的 L1——是"位置"+"量"。
#   后者更严格，也更能引导 Generator 画对位置。
# =============================================================================

@torch.no_grad()
def canny_edges(imgs):
    """
    Extract Canny edge maps from a batch of images (always detached).
    imgs: (B, 3, 64, 64) in [0, 1]
    Returns: (B, 1, 64, 64) binary edge maps — no grad, serves as weight mask
    """
    batch_edges = []
    for img in imgs:
        np_img = (img.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
        gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
        edge_tensor = torch.from_numpy((edges > 0).astype(np.float32)).unsqueeze(0)
        batch_edges.append(edge_tensor)
    return torch.stack(batch_edges, dim=0).to(imgs.device)


def canny_loss(fake_imgs, real_imgs):
    """
    Edge-weighted L1 loss: penalize pixel differences MORE at edge locations.

    Canny is non-differentiable, so we use it as a weight mask, NOT as a
    comparison target. The gradient flows through the raw pixel L1(loss),
    and the Canny mask tells the loss "focus gradient on these edge pixels."

    fake_imgs, real_imgs: (B, 3, 64, 64) in [-1, 1]
    Returns: scalar loss
    """
    fake = (fake_imgs + 1) / 2.0
    real = (real_imgs + 1) / 2.0

    # Edge mask from real images (detached — no grad through Canny)
    # (B, 1, 64, 64) binary: 1=edge, 0=background
    real_edge_mask = canny_edges(real)

    # Weighted L1: edge pixels get 10x more penalty than background
    # Fake has grad, real_edge_mask is detached — gradients flow through (fake - real)
    pixel_diff = (fake - real).abs()                     # (B, 3, 64, 64)
    edge_weight = 1.0 + 9.0 * real_edge_mask             # (B, 1, 64, 64): 10x at edges
    weighted_l1 = (edge_weight * pixel_diff).mean()      # broadcast: (B,1) x (B,3) = (B,3)

    return weighted_l1 * CANNY_LAMBDA


# =============================================================================
# Models (model64.py — UNCHANGED)
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
    print(f"  Technique: Canny Edge Loss (lambda={CANNY_LAMBDA}, thresh={CANNY_LOW}/{CANNY_HIGH})")
    print(f"  Device: {DEVICE}")
    if DEVICE.type=="cuda": print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  G: {gp:,} params | D: {dp:,} params | Images: {len(image_paths)}")
    print(f"  Epochs: {EPOCHS}  |  Batch: {BATCH_SIZE}  |  Steps/epoch: {len(dl)}")
    print(f"{'='*55}\n")

    G=Generator().to(DEVICE); D=Discriminator().to(DEVICE); crit=nn.BCELoss()
    g_opt=torch.optim.Adam(G.parameters(),lr=LR,betas=BETAS)
    d_opt=torch.optim.Adam(D.parameters(),lr=LR,betas=BETAS)
    csv_f=open(os.path.join(EXP_DIR,"loss.csv"),"w",newline="")
    w=csv.writer(csv_f); w.writerow(["epoch","D_loss","G_loss","G_adv","G_canny","D_real","D_fake"])
    fdl,fgl,fdr,fdf=0.0,0.0,0.0,0.0

    print("Training ...\n")
    for ep in range(1,EPOCHS+1):
        for img in dl:
            real=img.to(DEVICE); bs=real.size(0)
            rl,fl=torch.ones(bs,device=DEVICE),torch.zeros(bs,device=DEVICE)

            # Train D (unchanged)
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fake=G(noise)
            ro,fo=D(real),D(fake.detach())
            d_loss=crit(ro,rl)+crit(fo,fl)
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()

            # Train G — [修改点] 加入 Canny Edge Loss
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fake=G(noise)
            g_adv=crit(D(fake),rl)
            g_canny=canny_loss(fake, real)
            g_loss=g_adv + g_canny
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()

        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        g_adv_v=g_adv.item(); g_canny_v=g_canny.item()
        w.writerow([ep,dl_v,gl_v,g_adv_v,g_canny_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}(adv={g_adv_v:.4f} canny={g_canny_v:.4f})  DR:{dr_v:.4f}  DF:{df_v:.4f}")

        if ep%SAMPLE_INTERVAL==0:
            G.eval(); samples=G(fixed_noise) if fixed_noise is not None else None
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
    axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss (adv+canny)"); axes[0].grid(True,alpha=0.3)
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
    except Exception as e: fid=-1; print(f"  FID FAILED (network/DNS): {e}"); gc.collect()
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
        "technique":f"Canny Edge Loss (lambda={CANNY_LAMBDA}, thresh={CANNY_LOW}/{CANNY_HIGH})",
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
