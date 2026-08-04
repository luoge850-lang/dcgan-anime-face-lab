"""
================================================================================
 DCGAN Experiment 2: Augmentation Ablation (A:Flip / B:Color)
================================================================================
 Runs 2 augmentation configurations sequentially in ONE Kaggle session.
 Each produces a flat output folder under /kaggle/working/dcgan_output/.

 Output:
   exp2_a_flip/
   exp2_b_color/
   exp2_c_structure/
     metrics.json          ← FID, LPIPS, Diversity, Laplacian Variance, Loss
     loss.csv, loss_curves.png, epoch_050.png, epoch_100.png, real_images.png
     generator_final.pth
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
OUTPUT_DIR="/kaggle/working/dcgan_output"; DATASET_PATH="/kaggle/input/gananime-lite"; DATASET_LIMIT=8000
IMAGE_SIZE=64; BATCH_SIZE=32; NOISE_DIM=100; LR=1e-4; BETAS=(0.5,0.99); SEED=42
SAMPLE_INTERVAL=50; DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    """Auto-discover GANAnime Lite dataset (25k anime faces)."""
    if os.path.exists(DATASET_PATH):
        imgs=find_all_images(DATASET_PATH)
        if imgs: print(f"Dataset: {DATASET_PATH} ({len(imgs)} images)"); return DATASET_PATH,imgs
    print("Scanning /kaggle/input/ ...")
    for sub in sorted(os.listdir("/kaggle/input")):
        sp=os.path.join("/kaggle/input",sub)
        if os.path.isdir(sp):
            imgs=find_all_images(sp)
            if imgs: print(f"Dataset: {sp} ({len(imgs)} images)"); return sp,imgs
    raise FileNotFoundError("GanAnime Lite not found. Add via Kaggle 'Add Data' -> prasoonkottarathil/gananime-lite")

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
class EdgeSharpen:
    def __init__(self,prob=0.3,alpha=0.3): self.prob,self.alpha=prob,alpha
    def __call__(self,img):
        if random.random()<self.prob:
            arr=np.array(img,dtype=np.float32)/255.0
            blurred=np.array(img.filter(ImageFilter.GaussianBlur(radius=1.5)),dtype=np.float32)/255.0
            sharp=arr+self.alpha*(arr-blurred)
            return Image.fromarray(np.clip(sharp*255,0,255).astype(np.uint8))
        return img

class MedianDenoise:
    def __init__(self,prob=0.2,ksize=3): self.prob,self.ksize=prob,ksize
    def __call__(self,img):
        if random.random()<self.prob: return img.filter(ImageFilter.MedianFilter(size=self.ksize))
        return img

# =============================================================================
class Generator(nn.Module):
    def __init__(self,nd=NOISE_DIM):
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
        d=mr-mf; cm = linalg.sqrtm(sr.dot(sf))
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

# =============================================================================
def run_experiment(name,group,epochs,transform_fn,image_paths,fixed_noise):
    exp_dir=os.path.join(OUTPUT_DIR,name); os.makedirs(exp_dir,exist_ok=True); set_all_seeds(SEED)
    ds=AnimeDataset(image_paths,transform=transform_fn()); dl=DataLoader(ds,batch_size=BATCH_SIZE,shuffle=True,drop_last=True,num_workers=2,pin_memory=True)
    g_tmp,d_tmp=Generator(),Discriminator(); gp=sum(p.numel() for p in g_tmp.parameters()); dp=sum(p.numel() for p in d_tmp.parameters()); del g_tmp,d_tmp
    print(f"\n{'='*55}\n  Experiment: {name}  |  Group: {group}\n  Epochs: {epochs}  |  Device: {DEVICE}")
    if DEVICE.type=="cuda": print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  G: {gp:,} params  |  D: {dp:,} params  |  Images: {len(image_paths)}  |  Steps/epoch: {len(dl)}\n{'='*55}\n")
    G=Generator().to(DEVICE); D=Discriminator().to(DEVICE); crit=nn.BCELoss()
    g_opt=torch.optim.Adam(G.parameters(),lr=LR,betas=BETAS); d_opt=torch.optim.Adam(D.parameters(),lr=LR,betas=BETAS)
    csv_f=open(os.path.join(exp_dir,"loss.csv"),"w",newline=""); w=csv.writer(csv_f); w.writerow(["epoch","D_loss","G_loss","D_real","D_fake"])
    fdl,fgl,fdr,fdf=0.0,0.0,0.0,0.0
    print("Training ...\n")
    for ep in range(1,epochs+1):
        for img in dl:
            real=img.to(DEVICE); bs=real.size(0); rl,fl=torch.ones(bs,device=DEVICE),torch.zeros(bs,device=DEVICE)
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fake=G(noise); ro,fo=D(real),D(fake.detach())
            d_loss=crit(ro,rl)+crit(fo,fl); d_opt.zero_grad(); d_loss.backward(); d_opt.step()
            noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); g_loss=crit(D(G(noise)),rl); g_opt.zero_grad(); g_loss.backward(); g_opt.step()
        dl_v,gl_v,dr_v,df_v=d_loss.item(),g_loss.item(),ro.mean().item(),fo.mean().item()
        w.writerow([ep,dl_v,gl_v,dr_v,df_v]); fdl,fgl,fdr,fdf=dl_v,gl_v,dr_v,df_v
        print(f"Epoch [{ep:3d}/{epochs}]  D:{dl_v:.4f}  G:{gl_v:.4f}  DR:{dr_v:.4f}  DF:{df_v:.4f}")
        if ep%SAMPLE_INTERVAL==0:
            G.eval();
            with torch.no_grad(): samples=G(fixed_noise)
            G.train(); save_image_grid(samples,os.path.join(exp_dir,f"epoch_{ep:03d}.png"))
            torch.save(G.state_dict(),os.path.join(exp_dir,f"generator_epoch_{ep:03d}.pth"))
    csv_f.close(); torch.save(G.state_dict(),os.path.join(exp_dir,"generator_final.pth")); torch.save(D.state_dict(),os.path.join(exp_dir,"discriminator_final.pth"))
    save_image_grid(next(iter(dl))[:64],os.path.join(exp_dir,"real_images.png"))
    df=pd.read_csv(os.path.join(exp_dir,"loss.csv"))
    fig,axes=plt.subplots(1,3,figsize=(16,5))
    axes[0].plot(df["epoch"],df["G_loss"],color="#e74c3c",lw=1.5); axes[0].set(xlabel="Epoch",ylabel="Loss",title="Generator Loss"); axes[0].grid(True,alpha=0.3)
    axes[1].plot(df["epoch"],df["D_loss"],color="#3498db",lw=1.5); axes[1].set(xlabel="Epoch",ylabel="Loss",title="Discriminator Loss"); axes[1].grid(True,alpha=0.3)
    axes[2].plot(df["epoch"],df["D_real"],color="#2ecc71",lw=1.5,label="D(Real)"); axes[2].plot(df["epoch"],df["D_fake"],color="#e67e22",lw=1.5,label="D(Fake)")
    axes[2].axhline(0.5,color="gray",ls="--",alpha=0.4); axes[2].set(xlabel="Epoch",ylabel="Mean",title="D(Real) vs D(Fake)"); axes[2].legend(fontsize=8); axes[2].grid(True,alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(exp_dir,"loss_curves.png"),dpi=150); plt.close()
    print(f"\n{'='*55}\n  Computing Metrics for {name}\n{'='*55}")
    eval_tf=transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    eval_ds=AnimeDataset(image_paths,transform=eval_tf); eval_dl=DataLoader(eval_ds,batch_size=64,shuffle=True,drop_last=True,num_workers=2,pin_memory=True)
    print("FID ..."); fid=FIDCalculator(DEVICE).compute_fid(G,eval_dl,n=10000); print(f"  FID: {fid:.2f}"); gc.collect(); torch.cuda.empty_cache()
    print("LPIPS + Diversity + Laplacian ..."); lpips_calc=LPIPSCalculator(DEVICE); rb,fb=[],[]; generated=0
    for imgs in eval_dl: rb.append((imgs.to(DEVICE)+1)/2.0)
    rb=torch.cat(rb,dim=0)[:500]
    with torch.no_grad():
        while generated<500: bs=min(32,500-generated); noise=torch.randn(bs,NOISE_DIM,1,1,device=DEVICE); fb.append((G(noise)+1)/2.0); generated+=bs
    fb=torch.cat(fb,dim=0)[:500]
    lpips_scores=[]
    for i in range(0,500,50): lpips_scores.extend(lpips_calc.compute_lpips(fb[i:min(i+50,500)],rb[i:min(i+50,500)]))
    lpips_mean=float(np.mean(lpips_scores)); div=compute_diversity(G,lpips_calc,ns=300); lap=compute_laplacian_variance(fb[:200])
    print(f"  LPIPS: {lpips_mean:.4f}  Diversity: {div:.4f}  LapVar: {lap:.2f}"); del lpips_calc; gc.collect(); torch.cuda.empty_cache()
    metrics={"experiment_name":name,"experiment_group":group,"epochs":epochs,"dataset_size":len(image_paths),"FID":round(fid,2),"LPIPS":round(lpips_mean,4),"Diversity":round(div,4),"Laplacian_Variance":round(lap,2),"final_G_loss":round(fgl,4),"final_D_loss":round(fdl,4),"D_real":round(fdr,4),"D_fake":round(fdf,4),"completed_at":datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(exp_dir,"metrics.json"),"w") as f: json.dump(metrics,f,indent=2)
    print(f"  Complete: {name}  FID={fid:.2f}  LPIPS={lpips_mean:.4f}  Div={div:.4f}  LapVar={lap:.2f}\n")
    del G,D; gc.collect(); torch.cuda.empty_cache()

# =============================================================================
def main():
    set_all_seeds(SEED); dataset_path,image_paths=load_dataset()
    if DATASET_LIMIT and len(image_paths)>DATASET_LIMIT:
        set_all_seeds(SEED); image_paths=random.sample(image_paths,DATASET_LIMIT)
        print(f"Subsampled to {DATASET_LIMIT} images (seed={SEED})")
    set_all_seeds(SEED); fixed_noise=torch.randn(64,NOISE_DIM,1,1,device=DEVICE)

    def tf_flip(): return transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
    def tf_color(): return transforms.Compose([transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),transforms.ColorJitter(brightness=0.2,contrast=0.2),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

    run_experiment("exp2_a_flip","exp2_augmentation_ablation",100,tf_flip,image_paths,fixed_noise)
    run_experiment("exp2_b_color","exp2_augmentation_ablation",100,tf_color,image_paths,fixed_noise)

    print(f"\n{'='*55}\n  ALL EXPERIMENTS COMPLETE\n  Results: {os.path.abspath(OUTPUT_DIR)}/\n{'='*55}")

if __name__=="__main__": main()
