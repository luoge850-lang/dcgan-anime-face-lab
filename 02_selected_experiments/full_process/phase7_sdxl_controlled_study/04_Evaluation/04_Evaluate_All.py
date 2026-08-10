"""
Unified Evaluation — Colab 粘贴即跑。
================================================================================
统一评估 A0, A10, A20 三组: Legacy FID, Coverage, 特征多样性, 模糊率, 边缘密度。

使用方法:
  1. colab.research.google.com → New Notebook → T4 GPU
  2. 拖入所有结果 zip 到 /content/:
     - A0_5K_results.zip
     - A10_5K_results.zip
     - A20_5K_results.zip
  3. 本文件全文粘贴到一个 cell → Ctrl+Enter
  4. 下载 comparison.csv + comparison.json

评估规则:
  - 有效组: FID < A0 FID AND Coverage >= A0 Coverage × 95%
  - Coverage < A0 × 90% → 标记为硬失败
  - 最终推荐: 有效组中 FID 最低者
"""

from __future__ import annotations
import csv, json, os, random, shutil, subprocess, sys, time, zipfile
import importlib.util
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# 0. 依赖 + CUDA
# ═══════════════════════════════════════════════════════════════════
def _install() -> None:
    pkgs = [
        ("torch>=2.0.0", "torch"), ("torchvision>=0.15.0", "torchvision"),
        ("numpy", "numpy"), ("Pillow>=10.0.0", "PIL"), ("scipy", "scipy"),
    ]
    missing = [p for p, m in pkgs if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_install()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from PIL import Image
from scipy import linalg


def _ensure_cuda() -> None:
    need_fix = False
    if not torch.cuda.is_available():
        need_fix = True
    else:
        try:
            a = torch.randn(2000, 2000, device="cuda"); b = torch.randn(2000, 2000, device="cuda")
            torch.cuda.synchronize(); t0 = time.perf_counter(); _ = a @ b; torch.cuda.synchronize()
            if time.perf_counter() - t0 > 0.5: need_fix = True
        except Exception: need_fix = True
    if need_fix:
        print("[cuda] Reinstalling CUDA torch...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
            "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu121"])

_ensure_cuda()
assert torch.cuda.is_available(), "Enable T4 GPU!"
print(f"CUDA OK — {torch.cuda.get_device_name(0)}")


# ═══════════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════════
IMAGE_SIZE = 64; NOISE_DIM = 128; SEED = 42
EVAL_N = int(os.environ.get("EVAL_N", "5000"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = Path("/content/evaluation")

GROUPS = ["A0", "A10", "A20"]
WEIGHT_FILES = {
    "A0":  "generator_ema_final.pth",
    "A10": "generator_ema_best.pth",
    "A20": "generator_ema_best.pth",
}


# ═══════════════════════════════════════════════════════════════════
# 2. Generator
# ═══════════════════════════════════════════════════════════════════
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 768, 4), nn.BatchNorm2d(768), nn.ReLU(),
            nn.ConvTranspose2d(768, 384, 4, 2, 1), nn.BatchNorm2d(384), nn.ReLU(),
            nn.ConvTranspose2d(384, 192, 4, 2, 1), nn.BatchNorm2d(192), nn.ReLU(),
            nn.ConvTranspose2d(192, 96, 4, 2, 1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.ConvTranspose2d(96, 3, 4, 2, 1), nn.Tanh())
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.orthogonal_(m.weight)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
    def forward(self, x): return self.net(x)


# ═══════════════════════════════════════════════════════════════════
# 3. 工具函数
# ═══════════════════════════════════════════════════════════════════
def find_weight(zip_path: Path, weight_name: str) -> Path:
    """从 zip 中提取权重文件, 返回解压后的 .pth 路径"""
    extract_root = Path("/content") / f"_eval_{zip_path.stem}"
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_root)
    matches = list(extract_root.rglob(weight_name))
    if not matches:
        # 尝试查找任何 generator_ema*.pth
        matches = list(extract_root.rglob("generator_ema_*.pth")) + \
                  list(extract_root.rglob("generator_ema_final.pth")) + \
                  list(extract_root.rglob("generator_ema_best.pth"))
    if not matches:
        raise FileNotFoundError(f"'{weight_name}' not found in {zip_path}")
    # 返回文件大小最大的 (可能是正确的)
    return max(matches, key=lambda p: p.stat().st_size)


class EvalDataset(Dataset):
    def __init__(self, paths: list[str]):
        self.paths = paths
        self.tf = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.5,)*3, (0.5,)*3)])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def inception_features(inc, images):
    x = F.interpolate((images + 1) / 2, size=(299, 299), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5; return inc(x).detach().cpu().numpy()


def feature_batch(inc, paths, n):
    ds = EvalDataset(paths[:n]); out = []
    for i in range(0, len(ds), 64):
        out.append(inception_features(inc, torch.stack([ds[j] for j in range(i, min(i + 64, len(ds)))]).to(DEVICE)))
    return np.concatenate(out, axis=0)


def compute_stats(real, fake):
    mr, mf = real.mean(0), fake.mean(0)
    sr, sf = np.cov(real, rowvar=False), np.cov(fake, rowvar=False)
    cm = linalg.sqrtm(sr.dot(sf)); cm = cm.real if np.iscomplexobj(cm) else cm
    fid = float((mr - mf).dot(mr - mf) + np.trace(sr + sf - 2 * cm))

    sample = real[: min(512, len(real))]
    dist_rr = np.linalg.norm(sample[:, None, :] - sample[None, :, :], axis=2)
    dist_rr += np.eye(len(sample)) * 1e9
    threshold = float(np.percentile(np.min(dist_rr, axis=1), 95))
    coverage = float((np.min(np.linalg.norm(sample[:, None, :] - fake[None, :, :], axis=2), axis=1) <= threshold).mean())

    centered = fake - fake.mean(0); cov = np.cov(centered, rowvar=False)
    eig = np.linalg.eigvalsh(cov).clip(min=1e-12); p = eig / eig.sum()
    effective_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    return {"fid_legacy_inception_v3": fid, "coverage": coverage,
            "fake_feature_effective_rank": effective_rank, "coverage_threshold": threshold}


def image_metrics(images):
    arr = ((images + 1) / 2).clamp(0, 1).cpu().numpy()
    gray = arr.mean(1)
    p = np.pad(gray, ((0, 0), (1, 1), (1, 1)), mode="edge")
    lap = p[:, :-2, 1:-1] + p[:, 2:, 1:-1] + p[:, 1:-1, :-2] + p[:, 1:-1, 2:] - 4 * p[:, 1:-1, 1:-1]
    gx = gray[:, :, 1:] - gray[:, :, :-1]; gy = gray[:, 1:, :] - gray[:, :-1, :]
    lap_values = lap.reshape(len(lap), -1).var(1)
    edge_values = (np.abs(gx[:, :, :-1]) + np.abs(gy[:, :-1, :])) > 0.08
    return lap_values, edge_values.reshape(len(edge_values), -1).mean(1)


# ═══════════════════════════════════════════════════════════════════
# 4. 主评估流程
# ═══════════════════════════════════════════════════════════════════
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # —— 检测可用 zip ——
    available = {}
    for group in GROUPS:
        zip_path = Path(f"/content/{group}_5K_results.zip")
        if zip_path.exists():
            available[group] = zip_path
            print(f"[found] {group}: {zip_path.name} ({zip_path.stat().st_size/1e6:.0f}MB)")
        else:
            print(f"[missing] {group}_5K_results.zip — skipping")
    if "A0" not in available:
        raise RuntimeError("A0_5K_results.zip required! Drag it to /content/")

    # —— 加载 A0 参考图 (从 A0 zip 提取 training_images.zip) ——
    print("\n[setup] Extracting A0 reference images ...")
    with zipfile.ZipFile(available["A0"]) as z:
        # 查找 training_images.zip 或 dataset_manifest.txt
        has_training_zip = any("training_images.zip" in n for n in z.namelist())
        if has_training_zip:
            # 提取 training_images.zip 中的图片作为参考
            extract_root = Path("/content/_eval_ref")
            if extract_root.exists(): shutil.rmtree(extract_root)
            extract_root.mkdir(parents=True, exist_ok=True)
            z.extractall(extract_root)
            # 再解压 training_images.zip
            train_zip = list(extract_root.rglob("training_images.zip"))
            if train_zip:
                ref_extract = Path("/content/_eval_ref_images")
                if ref_extract.exists(): shutil.rmtree(ref_extract)
                ref_extract.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(train_zip[0]) as tz:
                    tz.extractall(ref_extract)
                real_paths = sorted([str(p) for p in ref_extract.rglob("*")
                                    if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}])
            else:
                # Fallback: 直接找 64x64 图片
                real_paths = sorted([str(extract_root / n) for n in z.namelist()
                                    if n.lower().endswith(('.png', '.jpg')) and 'training_images' in n])
        else:
            # 没有 training_images.zip — 需要用户提供
            raise RuntimeError(
                "A0 zip must contain training_images.zip with the 5000 reference images. "
                "Re-run 00_A0_Baseline_5K.py to generate it."
            )

    if len(real_paths) < EVAL_N:
        print(f"[warning] Only {len(real_paths)} reference images (need {EVAL_N})")
    real_paths = real_paths[:EVAL_N]
    print(f"[ref] {len(real_paths)} reference images")

    # —— InceptionV3 ——
    inc = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
    inc.fc = nn.Identity(); inc.eval().to(DEVICE)

    # 参考图特征
    print("[eval] Computing real reference features ...")
    real_feat = feature_batch(inc, real_paths, EVAL_N)

    # 参考图图像指标
    real_ds = EvalDataset(real_paths[:EVAL_N])
    real_lap, real_edge = [], []
    for i in range(0, len(real_ds), 64):
        batch = torch.stack([real_ds[j] for j in range(i, min(i + 64, len(real_ds)))])
        lv, ev = image_metrics(batch); real_lap.extend(lv.tolist()); real_edge.extend(ev.tolist())
    blur_threshold = float(np.percentile(real_lap, 10))
    real_lap_mean = float(np.mean(real_lap))
    real_edge_mean = float(np.mean(real_edge))

    # —— 固定噪声 ——
    z = torch.randn(EVAL_N, NOISE_DIM, 1, 1,
                    generator=torch.Generator(device=DEVICE).manual_seed(SEED), device=DEVICE)

    # —— 评估每组 ——
    rows = []
    for group in GROUPS:
        if group not in available:
            continue
        print(f"\n[eval] {group} ...")
        weight_path = find_weight(available[group], WEIGHT_FILES[group])

        G = Generator().to(DEVICE)
        state = torch.load(weight_path, map_location=DEVICE)
        # 处理不同的保存格式
        if isinstance(state, dict) and "generator_ema" in state:
            state = state["generator_ema"]
        elif isinstance(state, dict) and "shadow" in state:
            # EMA state_dict — 直接使用 shadow 权重
            state = {k.replace("shadow.", ""): v for k, v in state.items() if "shadow" in k}
        G.load_state_dict(state, strict=True)
        G.eval()

        fake_feat, blur_vals, edge_vals = [], [], []
        with torch.no_grad():
            for i in range(0, EVAL_N, 64):
                fake = G(z[i:i + 64])
                fake_feat.append(inception_features(inc, fake))
                lap, edge = image_metrics(fake)
                blur_vals.extend(lap.tolist()); edge_vals.extend(edge.tolist())

        fake_feat = np.concatenate(fake_feat, axis=0)
        metric = compute_stats(real_feat, fake_feat)
        metric.update({
            "group": group, "weight": str(weight_path),
            "fake_laplacian_mean": float(np.mean(blur_vals)),
            "real_laplacian_mean": real_lap_mean,
            "blur_threshold_real_p10": blur_threshold,
            "fake_blur_rate": float(np.mean(np.asarray(blur_vals) < blur_threshold)),
            "real_edge_density": real_edge_mean,
            "fake_edge_density": float(np.mean(edge_vals)),
            "edge_density_ratio": float(np.mean(edge_vals) / max(real_edge_mean, 1e-9)),
        })
        rows.append(metric)

    # —— 相对于 A0 的 delta ——
    baseline = next(r for r in rows if r["group"] == "A0")
    baseline_cov = baseline["coverage"]
    for r in rows:
        r["fid_delta_vs_A0"] = r["fid_legacy_inception_v3"] - baseline["fid_legacy_inception_v3"]
        r["coverage_ratio_vs_A0"] = r["coverage"] / max(baseline_cov, 1e-9)
        r["coverage_update_eligible"] = bool(r["coverage_ratio_vs_A0"] >= 0.95)
        r["coverage_abort"] = bool(r["coverage_ratio_vs_A0"] < 0.90)
        r["passes_criteria"] = bool(r["fid_delta_vs_A0"] < 0 and r["coverage_ratio_vs_A0"] >= 0.95)

    # —— 输出 ——
    with (OUTPUT_DIR / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (OUTPUT_DIR / "comparison.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    # —— 打印结论 ——
    print(f"\n{'='*60}")
    print(f"  EVALUATION RESULTS")
    print(f"  {'─'*50}")
    for r in rows:
        status = "✅ PASS" if r["passes_criteria"] else ("❌ ABORT" if r["coverage_abort"] else "⚠ MARGINAL")
        print(f"  {r['group']}: FID={r['fid_legacy_inception_v3']:.2f} "
              f"(Δ={r['fid_delta_vs_A0']:+.2f})  "
              f"Coverage={r['coverage']:.4f} ({r['coverage_ratio_vs_A0']:.2%})  "
              f"{status}")
    print(f"  {'─'*50}")
    passing = [r for r in rows if r["passes_criteria"]]
    if passing:
        best = min(passing, key=lambda r: r["fid_legacy_inception_v3"])
        print(f"  🏆 Best: {best['group']}  FID={best['fid_legacy_inception_v3']:.2f}")
    else:
        print(f"  ⚠ No group passes all criteria")
    print(f"  ─────────────────────────────────────────────")
    print(f"  💾 下载: Colab 左侧文件浏览器")
    print(f"     → 右键 evaluation/ → Download folder as zip")
    print(f"     或下载 comparison.csv + comparison.json")
    print(f"{'='*60}")

    # 打包 evaluation 结果
    zip_path = Path("/content/evaluation_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                zf.write(str(f), str(Path("evaluation") / f.relative_to(OUTPUT_DIR)))
    print(f"\n  Download: evaluation_results.zip ({zip_path.stat().st_size/1e3:.0f}KB)")


if __name__ == "__main__":
    main()
