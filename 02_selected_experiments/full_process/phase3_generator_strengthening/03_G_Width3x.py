"""
================================================================================
 G强化实验 03_G_Width3x: 通道×3 — 容量边际收益测试
================================================================================
 PROBLEM: 01_Width×2(FID=78.75, 4.4M)证明2×容量有效。
          ×3是否还有边际收益？还是×2已达饱和？

 VARIABLE: 通道 ×3 (768→384→192→96, ~10M params, G:D≈8:1)
          架构结构 = Baseline ConvTranspose+BN+ReLU (bit-for-bit, 5层)
          NO Skip Connection. NO architecture change.

 RISK: G:D=8:1 → 可能反向失衡(G压倒D, D无法有效判别)
       GAN文献中 G:D > 5:1 通常需要特殊训练策略
       如果崩溃 → 说明 01(4:1) 已达最优比例

 PREDICTION: FID 73~85
   最优: FID < 76 → 容量方向还有空间
   中性: FID ≈ 78 → ×2 已达饱和
   最差: FID > 82 或 G压倒D → ×2 是最优容量点
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

EXPERIMENT_NAME = "03_G_Width3x"
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"
DATASET_LIMIT = 10000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=128; LR=1e-4; BETAS=(0.5,0.99); SEED=42
EPOCHS=200; SAMPLE_INTERVAL=50; N_FID=10000
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXP_DIR=os.path.join(OUTPUT_DIR, EXPERIMENT_NAME)

import time as _time
def _download_with_retry(model_fn, name):
    for attempt in range(5):
        try: return model_fn()
        except Exception as e:
            if attempt < 4: wait=2**attempt*5; print(f"  {name} retry in {wait}s..."); _time.sleep(wait)
            else: raise e
print("Pre-loading models...")
try: _inc=_download_with_retry(lambda:models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT,transform_input=False),"InceptionV3"); del _inc; gc.collect(); print("  InceptionV3: OK")
except: print("  InceptionV3: FAILED")
try: _anet=_download_with_retry(lambda:models.alexnet(weights=models.AlexNet_Weights.DEFAULT),"AlexNet"); del _anet; gc.collect(); print("  AlexNet: OK")
except: print("  AlexNet: FAILED")
print()

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
    raise FileNotFoundError("No dataset found.")

class AnimeDataset(Dataset):
    def __init__(self,paths,transform=None): self.paths,self.tf=paths,transform
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        for _ in range(10):
            try: img=Image.open(self.paths[i]).convert("RGB"); return self.tf(img) if self.tf else img
            except (OSError, IOError): i=random.randint(0,len(self.paths)-1)
        raise RuntimeError("Failed to load any image after 10 retries")

def save_image_grid(tensor,fp,nrow=8):
    grid=make_grid(tensor,nrow=nrow,normalize=True,value_range=(-1,1))
    ndarr=grid.mul(255).clamp(0,255).permute(1,2,0).to("cpu",torch.uint8).numpy()
    Image.fromarray(ndarr).save(fp)

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
    return transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.RandomHorizontalFlip(p=0.5),EdgeSharpen(prob=0.2),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# =============================================================================
# [03 CHANGE] Generator Width ×3 — 所有中间层通道翻三倍
#   768→384→192→96 (vs Baseline 256→128→64→32, vs Width×2 512→256→128→64)
#   Architecture IDENTICAL to 00/01 — 5 ConvTranspose+BN+ReLU. NO Skip.
# =============================================================================
class Generator(nn.Module):
    def __init__(self, nd=NOISE_DIM):
        super().__init__()
        self.net=nn.Sequential(
            nn.ConvTranspose2d(nd,768,4),nn.BatchNorm2d(768),nn.ReLU(),
            nn.ConvTranspose2d(768,384,4,2,1),nn.BatchNorm2d(384),nn.ReLU(),
            nn.ConvTranspose2d(384,192,4,2,1),nn.BatchNorm2d(192),nn.ReLU(),
            nn.ConvTranspose2d(192,96,4,2,1),nn.BatchNorm2d(96),nn.ReLU(),
            nn.ConvTranspose2d(96,3,4,2,1),nn.Tanh())
        self.apply(self._init)
    def _init(self,m):
        if isinstance(m,(nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias,0)
    def forward(self,x): return self.net(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3,32,3,2,1)),nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(32,64,3,2,1)),nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64,128,3,2,1)),nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128,256,3,2,1)),nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(4*4*256,256)),nn.LeakyReLU(0.2),
            nn.Linear(256,1))
    def forward(self,x): return self.net(x).view(-1)

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
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); vars_.append(cv2.Laplacian(gray,cv2.CV_64F).var())
    return float(np.mean(vars_))

def compute_edge_density(imgs, real_imgs=None):
    densities=[]
    for img in imgs:
        arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
        gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(gray,50,150); densities.append((edges>0).mean())
    fake_density=float(np.mean(densities))
    if real_imgs is not None:
        real_densities=[]
        for img in real_imgs:
            arr=(img.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
            gray=cv2.cvtColor(arr,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(gray,50,150); real_densities.append((edges>0).mean())
        real_density=float(np.mean(real_densities)); ratio=fake_density/max(real_density,1e-8)
        return fake_density, real_density, ratio
    return fake_density, None, None

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
    print(f"  {EXPERIMENT_NAME}: Width×3 (768→384→192→96, ~10M params)")
    print(f"  G: {gp:,} params (5 ConvT, ×3 channels) | D: {dp:,} params (Baseline)")
    print(f"  G:D ratio = {gp/dp:.1f}:1 | No Skip. Same arch as 00/01.")
    print(f"{'='*55}\n")
    G=Generator().to(DEVICE); D=Discriminator().to(DEVICE)
    g_opt=torch.optim.Adam(G.parameters(),lr=LR,betas=BETAS)
    d_opt=torch.optim.Adam(D.parameters(),lr=LR,betas=BETAS)
    csv_f=open(os.path.join(EXP_DIR,"loss.csv"),"w",newline="")
    w=csv.writer(csv_f); w.writerow(["epoch","D_loss","G_loss","D_real","D_fake"])
    fdl,fgl,fdr,fdf=0.0,0.0,0.0,0.0
    print("Training ...\n")
    for ep in range(1,EPOCHS+1):
        for img in dl:
            real=img.to(DEVICE); bs=real.size(0)
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            with torch.no_grad(): fake=G(noise)
            d_real=D(real); d_fake=D(fake)
            d_loss=F.relu(1.0-d_real).mean()+F.relu(1.0+d_fake).mean()
            d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            g_loss=-D(G(noise)).mean()
            g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),d_real.mean().item(),d_fake.mean().item()
        w.writerow([ep,dl_v,gl_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{EPOCHS}]  D:{dl_v:.4f}  G:{gl_v:.4f}  DR:{dr_v:+.2f}  DF:{df_v:+.2f}")
        if ep%SAMPLE_INTERVAL==0:
            G.eval(); samples=G(fixed_noise); G.train()
            save_image_grid(samples,os.path.join(EXP_DIR,f"epoch_{ep:03d}.png"))
            torch.save(G.state_dict(),os.path.join(EXP_DIR,f"generator_epoch_{ep:03d}.pth"))
    csv_f.close()
    torch.save(G.state_dict(),os.path.join(EXP_DIR,"generator_final.pth"))
    torch.save(D.state_dict(),os.path.join(EXP_DIR,"discriminator_final.pth"))
    save_image_grid(next(iter(dl))[:64],os.path.join(EXP_DIR,"real_images.png"))
    df=pd.read_csv(os.path.join(EXP_DIR,"loss.csv"))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].plot(df["epoch"],df["G_loss"],color="#e74c3c",lw=1.5); axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss (Hinge)"); axes[0].grid(True,alpha=0.3)
    axes[1].plot(df["epoch"],df["D_loss"],color="#3498db",lw=1.5); axes[1].set(xlabel="Epoch",ylabel="Loss",title="Discriminator Loss (Hinge)"); axes[1].grid(True,alpha=0.3)
    axes[2].plot(df["epoch"],df["D_real"],color="#2ecc71",lw=1.5,label="D(Real)")
    axes[2].plot(df["epoch"],df["D_fake"],color="#e67e22",lw=1.5,label="D(Fake)")
    axes[2].axhline(0.0,color="gray",ls="--",alpha=0.4); axes[2].set(xlabel="Epoch",ylabel="Mean Logit",title="D(Real) vs D(Fake)")
    axes[2].legend(fontsize=8); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(EXP_DIR,"loss_curves.png"),dpi=150); plt.close()
    print(f"\n{'='*55}\n  Computing Metrics for {EXPERIMENT_NAME}\n{'='*55}")
    eval_tf=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    eval_ds=AnimeDataset(image_paths,transform=eval_tf)
    eval_dl=DataLoader(eval_ds,batch_size=64,shuffle=True,drop_last=True,num_workers=2)
    print("FID ...")
    try: fid=FIDCalculator(DEVICE).compute_fid(G,eval_dl,n=N_FID); print(f"  FID: {fid:.2f}")
    except Exception as e: fid=-1; print(f"  FID FAILED: {e}"); gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    print("LPIPS + Diversity + Laplacian + Edge Density ...")
    lpips_calc=LPIPSCalculator(DEVICE); rb,fb=[],[]; generated=0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE)+1)/2.0)
    rb=torch.cat(rb,dim=0)[:500]
    with torch.no_grad():
        while generated<500:
            bs=min(32,500-generated); noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE)
            fb.append((G(noise)+1)/2.0); generated+=bs
    fb=torch.cat(fb,dim=0)[:500]
    lpips_scores=[]
    for i in range(0,500,50): lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i+50,500)],rb[i:min(i+50,500)]))
    lpips_mean=float(np.mean(lpips_scores)); div=compute_diversity(G,lpips_calc,ns=300)
    lap=compute_laplacian_variance(fb[:200])
    fake_edge, real_edge, edge_ratio=compute_edge_density(fb[:200],rb[:200])
    print(f"  LPIPS: {lpips_mean:.4f}  Diversity: {div:.4f}  LapVar: {lap:.2f}  EdgeRatio: {edge_ratio:.4f}")
    del lpips_calc; gc.collect()
    if DEVICE.type=="cuda": torch.cuda.empty_cache()
    metrics={"experiment_name":EXPERIMENT_NAME,"epochs":EPOCHS,"dataset_size":DATASET_LIMIT,
        "technique":"G Width×3: 768→384→192→96, ~10M params (8:1 G:D). 5 ConvT+BN+ReLU, no Skip. Tests capacity marginal benefit beyond ×2.",
        "FID":round(fid,2),"LPIPS":round(lpips_mean,4),"Diversity":round(div,4),
        "Laplacian_Variance":round(lap,2),"Edge_Density_Fake":round(fake_edge,4),
        "Edge_Density_Real":round(real_edge,4),"Edge_Density_Ratio":round(edge_ratio,4),
        "final_G_loss":round(fgl,4),"final_D_loss":round(fdl,4),
        "D_real":round(fdr,4),"D_fake":round(fdf,4),
        "completed_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(EXP_DIR,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    print(f"\n  Complete: {EXPERIMENT_NAME}  FID={fid:.2f}")

if __name__=="__main__": main()
