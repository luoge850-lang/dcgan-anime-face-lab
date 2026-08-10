"""
人工复核 rejected — 在 v2 清洗结果基础上手动捞回
================================================================================
前提: v2 脚本已运行, /kaggle/working/SDXL_Cleaned/ 中有:
  - accepted/  (自动通过的)
  - rejected/  (被淘汰的, 按原因分文件夹)
  - rejected_manifest.csv

第一轮: 本脚本生成 review_rejected.csv → 下载 → Excel 标记 → 保存为
        review_rejected_checked.csv → 上传 → 复跑本脚本
第二轮: 把 manual_keep=1 的从 rejected 搬到 accepted → 重新打包 zip
"""

import csv, json, os, shutil, zipfile
from pathlib import Path

OUTPUT_DIR = Path("/kaggle/working/SDXL_Cleaned")
REJECTED_CSV = OUTPUT_DIR / "rejected_manifest.csv"
REVIEW_TEMPLATE = Path("/kaggle/working/review_rejected.csv")
REVIEW_CHECKED_NAME = "review_rejected_checked.csv"
MANUAL_KEEP_DIR = Path("/kaggle/working/SDXL_Cleaned/rejected/manual_keep")

# ── 检查是否有 review_rejected_checked.csv ──
checked_path = None
for p in Path("/kaggle/input").rglob(REVIEW_CHECKED_NAME):
    checked_path = p; break

if checked_path:
    # ═══════════════════════════════════════
    # 第二轮: 合并 manual_keep=1
    # ═══════════════════════════════════════
    print(f"[review] Found {checked_path}")

    decisions = {}
    with checked_path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            decisions[r["source"]] = r.get("manual_keep", "").strip()

    keep_count = sum(1 for v in decisions.values() if v == "1")
    print(f"[review] manual_keep=1: {keep_count}/{len(decisions)}")

    # 复制 manual_keep=1 的图片
    MANUAL_KEEP_DIR.mkdir(parents=True, exist_ok=True)
    acc_dir = OUTPUT_DIR / "accepted"
    existing = set(f.stem for f in acc_dir.iterdir())
    next_id = max([int(f.stem[1:]) for f in acc_dir.iterdir() if f.stem.startswith("A")], default=0) + 1

    added = 0
    for r in csv.DictReader(REJECTED_CSV.open(newline="", encoding="utf-8-sig")):
        src_name = r["source"]
        if decisions.get(src_name) == "1":
            src = next(
                (p for p in Path("/kaggle/working/SDXL_Cleaned/rejected").rglob(src_name)
                 if p.is_file()), None)
            if src:
                aid = f"A{next_id:04d}"
                shutil.copy2(src, acc_dir / f"{aid}.png")
                shutil.copy2(src, MANUAL_KEEP_DIR / f"{aid}_{src_name}"[:240])
                next_id += 1; added += 1

    print(f"[review] Added {added} images to accepted/")

    # 重新打包
    acc_files = sorted(acc_dir.iterdir())
    zip_path = Path("/kaggle/working/cleaned_accepted_final.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for p in acc_files:
            z.write(p, p.name)

    print(f"\n{'='*60}")
    print(f"  FINAL: {len(acc_files)} accepted images")
    print(f"  Auto-pass + Manual-keep = {len(acc_files) - added} + {added}")
    print(f"  Zip: cleaned_accepted_final.zip ({zip_path.stat().st_size/1e6:.1f}MB)")
    print(f"{'='*60}")

else:
    # ═══════════════════════════════════════
    # 第一轮: 生成审核模板
    # ═══════════════════════════════════════
    print("[review] Generating manual review template for rejected images...")

    rows = []
    for cat_dir in sorted(Path("/kaggle/working/SDXL_Cleaned/rejected").iterdir()):
        if not cat_dir.is_dir() or cat_dir.name == "manual_keep":
            continue
        for p in sorted(cat_dir.iterdir()):
            if p.suffix.lower() == ".png":
                rows.append({
                    "reject_reason": cat_dir.name,
                    "source": p.name,
                    "file_size_kb": round(p.stat().st_size / 1024, 1),
                    "manual_keep": "",  # ← 你填 1=保留 或 0=丢弃
                })

    with REVIEW_TEMPLATE.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["reject_reason", "source", "file_size_kb", "manual_keep"])
        w.writeheader(); w.writerows(rows)

    print(f"\n{'='*60}")
    print(f"  REVIEW TEMPLATE: {len(rows)} rejected images")
    for cat in sorted(set(r["reject_reason"] for r in rows)):
        n = sum(1 for r in rows if r["reject_reason"] == cat)
        print(f"    {cat}: {n}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  1. 从 Output 下载 review_rejected.csv")
    print(f"  2. Excel 打开 → 只改 manual_keep 列 (1=保留, 0=丢弃)")
    print(f"  3. 保存为 review_rejected_checked.csv")
    print(f"  4. 上传为 Kaggle Dataset → attach → 复跑本脚本")
    print(f"{'='*60}")
