"""
SDXL Cleaning — Kaggle 版 —— 自动检测 /kaggle/input/ 中的 SDXL 候选图
================================================================================
两轮清洗流程:
  1. 自动筛选 (SHA-256, aHash 近重复, Laplacian 模糊, 纯色)
     → 输出 review_template.csv (含 auto_flags, auto_pass 列)
  2. 人工审核 (下载 review_template.csv → Excel 标记 manual_keep=1/0
     → 保存为 review_checked.csv → 上传为 Kaggle Dataset → 本脚本复跑)

输入 (自动检测 /kaggle/input/):
  - SDXL 候选 zip 文件 (含 candidate_manifest.csv 或 images_64/ 或 .png/.jpg 图片)
  - 第二轮: review_checked.csv

输出 (/kaggle/working/SDXL_Cleaned/):
  - accepted_64/          ← 通过清洗的 64×64 图片
  - accepted_manifest.csv ← 清洗通过清单
  - rejected_manifest.csv ← 被淘汰清单 (含淘汰原因)
  - review_template.csv   ← 人工审核模板
  - cleaning_summary.json ← 清洗统计

使用方法:
  1. Kaggle Notebook → T4×2 GPU (CPU 也行), Internet OFF 即可
  2. 把 SDXL 候选 zip 上传为 Kaggle Dataset → attach
  3. 本文件粘贴到一个 cell → Run
  4. 第一轮自动筛选完成 → 下载 review_template.csv
  5. Excel 打开 → 标记 manual_keep 列: 1=保留 0=丢弃 → 保存为 review_checked.csv
  6. 把 review_checked.csv 上传为新 Kaggle Dataset → attach → 复跑本脚本
  7. 下载输出的 accepted_64/ + accepted_manifest.csv (或直接下载 cleaned_accepted.zip)
"""

from __future__ import annotations
import csv, hashlib, json, os, random, shutil, sys, time, zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════════
OUTPUT_DIR = Path("/kaggle/working/SDXL_Cleaned")
INPUT_ZIP_GLOB = "/kaggle/input/**/*.zip"        # 自动搜索所有 zip
INPUT_DIR_SEARCH = "/kaggle/input/"               # 也搜索解压后的目录

REVIEW_CSV_NAME = "review_checked.csv"            # 第二轮: /kaggle/input/ 中的文件名
BLUR_THRESHOLD = 0                                # 0 = 从候选分布自动校准 P10
AHASH_THRESHOLD = 8                               # 汉明距离 ≤ 8 = 近重复
SEED = 42

# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════
def find_files(root: Path, exts: set[str] | None = None) -> list[Path]:
    """递归找所有文件, 跳过隐藏目录"""
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    if not root.exists():
        return files
    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in exts:
            # 跳过隐藏目录
            parts = item.relative_to(root).parts
            if any(p.startswith(".") or p.startswith("__") for p in parts):
                continue
            files.append(item)
    return sorted(files)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_hash(path: Path, size: int = 16) -> str:
    """感知哈希: 缩放到 size×size → 灰度 → 大于均值的像素为1 → 64位hex"""
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    bits = (a > a.mean()).astype(np.uint8).ravel()
    # 转为 hex 字符串
    return hashlib.sha1(bits.tobytes()).hexdigest()


def hamming_distance(h1: str, h2: str) -> int:
    """两个 hex 哈希的汉明距离 (基于 hex→bytes 转换)"""
    b1 = bytes.fromhex(h1[:40])  # sha1 hex = 40 chars = 20 bytes
    b2 = bytes.fromhex(h2[:40])
    return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))


def laplacian_score(path: Path) -> float:
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    p = np.pad(a, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    return float(lap.var())


# ═══════════════════════════════════════════════════════════════
# 输入发现
# ═══════════════════════════════════════════════════════════════
def discover_candidates() -> list[dict]:
    """
    自动发现 SDXL 候选图。
    策略:
      1. 搜索 /kaggle/input/ 下的 candidate_manifest.csv → 读取
      2. 搜索 zip 文件 → 解压 → 找 candidate_manifest.csv
      3. 都没有 → 直接搜索 /kaggle/input/ 下所有图片文件
    返回: [{"sample_id": ..., "image_path": Path, ...}, ...]
    """
    candidates = []
    seen_ids = set()

    # ── 策略 1 & 2: 找 manifest ──
    # 先解压所有 zip
    extracted_roots = []
    for zp in sorted(Path("/kaggle/input").rglob("*.zip")):
        if "cleaned" in zp.name.lower() or "accepted" in zp.name.lower():
            continue  # 跳过输出文件
        extract_root = Path("/kaggle/working/_extracted") / zp.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            print(f"[input] Extracting {zp.name} ({zp.stat().st_size/1e6:.0f}MB) ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(extract_root)
        extracted_roots.append(extract_root)

    # 搜索所有 candidate_manifest.csv
    all_manifests = list(Path("/kaggle/input").rglob("candidate_manifest.csv"))
    for root in extracted_roots:
        all_manifests.extend(root.rglob("candidate_manifest.csv"))

    for mf in sorted(set(all_manifests)):
        print(f"[input] Loading manifest: {mf}")
        img_root = mf.parent
        with mf.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                sid = row.get("sample_id", "")
                if sid in seen_ids:
                    sid = f"{sid}_{img_root.name}"
                seen_ids.add(sid)
                img_rel = row.get("image_64_path", row.get("preview_path", ""))
                img_path = img_root / img_rel
                candidates.append({
                    "sample_id": sid,
                    "image_path": str(img_path),
                    "target_pose": row.get("target_pose", ""),
                    "target_hair": row.get("target_hair", ""),
                    "source_manifest": str(mf),
                })

    # ── 策略 3: 直接找图片 (没有 manifest) ──
    if not candidates:
        print("[input] No manifest found, scanning for images directly...")
        all_imgs = find_files(Path("/kaggle/input"))
        for i, p in enumerate(all_imgs, 1):
            sid = f"IMG_{i:05d}"
            candidates.append({
                "sample_id": sid,
                "image_path": str(p),
                "target_pose": "unknown",
                "target_hair": "unknown",
                "source_manifest": "direct_scan",
            })

    # 去重路径
    uniq = []
    seen_p = set()
    for c in candidates:
        p = c["image_path"]
        if p not in seen_p:
            seen_p.add(p)
            uniq.append(c)
    print(f"[input] Total: {len(uniq)} unique candidates from "
          f"{len(set(c['source_manifest'] for c in uniq))} source(s)")
    return uniq


# ═══════════════════════════════════════════════════════════════
# 自动筛选
# ═══════════════════════════════════════════════════════════════
def calibrate_blur_threshold(rows: list[dict], max_samples: int = 2000) -> float:
    """用候选图 Laplacian 方差的 P10 作为模糊阈值"""
    scores = []
    for r in rows[:min(len(rows), max_samples)]:
        p = Path(r["image_path"])
        if p.exists():
            try:
                scores.append(laplacian_score(p))
            except Exception:
                pass
    if not scores:
        return 80.0
    p10 = float(np.percentile(scores, 10))
    print(f"[blur] Auto-calibrated threshold: P10={p10:.1f} (from {len(scores)} candidates)")
    return p10


def automatic_flags(rows: list[dict]) -> tuple[list[dict], float]:
    threshold = BLUR_THRESHOLD if BLUR_THRESHOLD > 0 else calibrate_blur_threshold(rows)

    seen_sha: dict[str, str] = {}
    seen_ahash: list[tuple[str, str, str]] = []  # (ahash, sample_id, hex)

    for row in rows:
        p = Path(row["image_path"])
        flags: list[str] = []

        if not p.exists():
            flags.append("missing")
            row["sha256_64"] = ""
            row["average_hash"] = ""
            row["laplacian_score"] = "0"
        else:
            try:
                digest = sha256_file(p)
                ah = average_hash(p)
                score = laplacian_score(p)
                row["sha256_64"] = digest
                row["average_hash"] = ah
                row["laplacian_score"] = f"{score:.2f}"

                # SHA-256 精确重复
                if digest in seen_sha:
                    flags.append(f"exact_dup:{seen_sha[digest]}")
                else:
                    seen_sha[digest] = row["sample_id"]

                # aHash 近重复
                for prev_ah, prev_sid, prev_hex in seen_ahash:
                    if hamming_distance(ah, prev_ah) <= AHASH_THRESHOLD:
                        flags.append(f"near_dup:{prev_sid}")
                        break
                seen_ahash.append((ah, row["sample_id"], ah[:16]))

                # 模糊
                if score < threshold:
                    flags.append("blur")

                # 纯色
                arr = np.asarray(Image.open(p).convert("RGB"))
                if arr.std() < 3:
                    flags.append("near_blank")

            except Exception as exc:
                flags.append(f"read_error:{type(exc).__name__}")

        row["auto_flags"] = "|".join(flags)
        row["auto_pass"] = "1" if not flags else "0"
        row.setdefault("manual_keep", "")
        row.setdefault("reject_reason", "")

    return rows, threshold


# ═══════════════════════════════════════════════════════════════
# 人工审核
# ═══════════════════════════════════════════════════════════════
def find_review_csv() -> Path | None:
    """在 /kaggle/input/ 中找 review_checked.csv"""
    for p in Path("/kaggle/input").rglob(REVIEW_CSV_NAME):
        print(f"[review] Found review_checked.csv: {p}")
        return p
    return None


def write_review_template(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "review_template.csv"
    fields = ["sample_id", "target_pose", "target_hair",
              "image_path", "auto_flags", "auto_pass", "manual_keep", "reject_reason"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def read_review(rows: list[dict], review_path: Path) -> list[dict]:
    decisions = {}
    with review_path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            decisions[r["sample_id"]] = r
    for r in rows:
        d = decisions.get(r["sample_id"], {})
        r["manual_keep"] = d.get("manual_keep", r.get("manual_keep", "")).strip()
        r["reject_reason"] = d.get("reject_reason", r.get("reject_reason", "")).strip()
    return rows


# ═══════════════════════════════════════════════════════════════
# 分层采样 → Accepted
# ═══════════════════════════════════════════════════════════════
def stratified_select(rows: list[dict]) -> list[dict]:
    """
    从 eligible 中分层采样, 尽量保持 (pose, hair) 类别均匀。
    如果 eligible 数量已经不多, 全部保留。
    """
    # 第一轮: 仅 auto_pass
    # 第二轮: auto_pass + manual_keep == 1
    has_manual = any(r.get("manual_keep", "").strip() == "1" for r in rows)

    eligible = []
    for r in rows:
        auto_ok = r.get("auto_pass", "") == "1"
        if not auto_ok:
            continue
        if has_manual:
            if r.get("manual_keep", "").strip() == "1":
                eligible.append(r)
        else:
            eligible.append(r)

    print(f"[select] Eligible: {len(eligible)}/{len(rows)} "
          f"({'auto+manual' if has_manual else 'auto only'})")
    return eligible


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED)

    candidates = discover_candidates()
    if not candidates:
        print("[error] No images found. Upload SDXL candidate zip as Kaggle Dataset and attach.")
        return

    # ── 自动筛选 ──
    rows, threshold = automatic_flags(candidates)
    auto_pass = sum(1 for r in rows if r.get("auto_pass") == "1")
    blur_n = sum(1 for r in rows if "blur" in (r.get("auto_flags", "")))
    dup_n = sum(1 for r in rows if "dup" in (r.get("auto_flags", "")))
    blank_n = sum(1 for r in rows if "near_blank" in (r.get("auto_flags", "")))
    print(f"[auto] pass={auto_pass}/{len(rows)}  blur={blur_n}  dup={dup_n}  blank={blank_n}  "
          f"threshold={threshold:.1f}  ({time.time()-t0:.0f}s)")

    # ── 检查是否有 review_checked.csv ──
    review_path = find_review_csv()
    if review_path:
        rows = read_review(rows, review_path)

    # ── 无人工审核: 输出模板, 停止 ──
    if not review_path:
        template = write_review_template(rows)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "cleaning_state.json").write_text(json.dumps({
            "stage": "waiting_for_review",
            "candidate_count": len(rows),
            "auto_pass_count": auto_pass,
            "blur_count": blur_n, "dup_count": dup_n, "blank_count": blank_n,
            "blur_threshold": threshold,
            "next": "Download review_template.csv from Kaggle output → mark manual_keep → "
                    "save as review_checked.csv → upload as Kaggle Dataset → re-run this script.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"  ⏸  WAITING FOR REVIEW")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Auto-pass: {auto_pass}/{len(rows)} candidates")
        print(f"  ─────────────────────────────────────────────")
        print(f"  1. 从 Kaggle Output 下载 review_template.csv")
        print(f"  2. Excel 打开 → 只改 manual_keep 列 (1=保留, 0=丢弃)")
        print(f"  3. 保存为 review_checked.csv")
        print(f"  4. 上传为 Kaggle Dataset → attach → 复跑本脚本")
        print(f"{'='*60}")
        return

    # ── 有人工审核: 生成 accepted ──
    accepted = stratified_select(rows)

    accepted_dir = OUTPUT_DIR / "accepted_64"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows = []
    for i, r in enumerate(accepted, 1):
        src = Path(r["image_path"])
        dst = accepted_dir / f"A{i:04d}.png"
        shutil.copy2(str(src), str(dst))
        accepted_rows.append({
            "accepted_id": f"A{i:04d}",
            "source_sample_id": r["sample_id"],
            "target_pose": r.get("target_pose", ""),
            "target_hair": r.get("target_hair", ""),
            "path": f"accepted_64/A{i:04d}.png",
            "sha256_64": sha256_file(dst),
            "auto_flags": r.get("auto_flags", ""),
        })

    selected_ids = {r["sample_id"] for r in accepted}

    # 保存 manifests
    with (OUTPUT_DIR / "accepted_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(accepted_rows[0].keys()))
        w.writeheader(); w.writerows(accepted_rows)

    with (OUTPUT_DIR / "rejected_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "target_pose", "target_hair",
                                           "auto_flags", "manual_keep", "reject_reason"])
        w.writeheader()
        for r in rows:
            if r["sample_id"] not in selected_ids:
                w.writerow({k: r.get(k, "") for k in w.fieldnames})

    # 总结
    (OUTPUT_DIR / "cleaning_summary.json").write_text(json.dumps({
        "stage": "complete",
        "candidate_count": len(rows),
        "accepted_count": len(accepted_rows),
        "rejected_count": len(rows) - len(accepted_rows),
        "blur_threshold": threshold,
        "runtime_seconds": round(time.time() - t0, 1),
        "pose_hair_distribution": {
            f"{k[0]}_{k[1]}": len([r for r in accepted_rows
                                   if r["target_pose"] == k[0] and r["target_hair"] == k[1]])
            for k in sorted(set((r["target_pose"], r["target_hair"]) for r in accepted_rows))
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打包
    zip_path = Path("/kaggle/working/cleaned_accepted.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                z.write(str(f), str(f.relative_to(OUTPUT_DIR)))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"  ✅ CLEANING DONE")
    print(f"  Accepted: {len(accepted_rows)} / {len(rows)} ({len(accepted_rows)*100/max(len(rows),1):.1f}%)")
    print(f"  Rejected: {len(rows) - len(accepted_rows)}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Zip: cleaned_accepted.zip ({size_mb:.0f}MB)")
    print(f"{'='*60}")
    print(f"\n  下一步:")
    print(f"  1. 下载 cleaned_accepted.zip")
    print(f"  2. 上传为 Kaggle Dataset (for 消融训练)")
    print(f"  3. 在消融 notebook 中 attach")


if __name__ == "__main__":
    main()
