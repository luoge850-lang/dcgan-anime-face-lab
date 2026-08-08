"""
SDXL ?? v2 ? ????? + ?? accepted/rejected
================================================================================
Kaggle ????, ????????

?? (/kaggle/working/SDXL_Cleaned/):
  accepted/     ? ??????? (?????)
  rejected/     ? ?????? (???????)
  accepted_manifest.csv
  rejected_manifest.csv
  cleaning_summary.json
  cleaned_accepted.zip ? ????
"""

import csv, hashlib, json, os, random, shutil, time, zipfile
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT_DIR = Path("/kaggle/working/SDXL_Cleaned")
AHASH_THRESHOLD = 8
SEED = 42

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()

def average_hash(p, size=16):
    im = Image.open(p).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    bits = (a > a.mean()).astype(np.uint8).ravel()
    return hashlib.sha1(bits.tobytes()).hexdigest()

def hamming_distance(h1, h2):
    b1, b2 = bytes.fromhex(h1[:40]), bytes.fromhex(h2[:40])
    return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))

def laplacian_score(p):
    a = np.asarray(Image.open(p).convert("L"), dtype=np.float32)
    p2 = np.pad(a, 1, mode="edge")
    lap = p2[:-2, 1:-1] + p2[2:, 1:-1] + p2[1:-1, :-2] + p2[1:-1, 2:] - 4 * p2[1:-1, 1:-1]
    return float(lap.var())

# ???????????????????????????????????????
# Step 1: ??????
# ???????????????????????????????????????
print("[1/4] Discovering images from /kaggle/input/ ...")
candidates = []
for zp in sorted(Path("/kaggle/input").rglob("*.zip")):
    extract_root = Path("/kaggle/working/_extracted") / zp.stem
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {zp.name} ...")
    with zipfile.ZipFile(zp) as z: z.extractall(extract_root)

img_files = []
for root in [Path("/kaggle/working/_extracted"), Path("/kaggle/input")]:
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for p in root.rglob(ext):
            if p.is_file() and not any(x.startswith(".") or x.startswith("__") for x in p.parts):
                img_files.append(p)

img_files = sorted(set(img_files))
print(f"  Found {len(img_files)} images")

for i, p in enumerate(img_files, 1):
    candidates.append({"sample_id": f"S{i:05d}", "image_path": str(p),
                       "source": p.parent.name, "filename": p.name})

# ???????????????????????????????????????
# Step 2: ???? + ??
# ???????????????????????????????????????
print("\n[2/4] Computing image hashes & scores ...")

# ??????
sample_n = min(len(candidates), 1000)
lap_scores = []
for r in candidates[:sample_n]:
    try: lap_scores.append(laplacian_score(r["image_path"]))
    except Exception: pass
blur_threshold = float(np.percentile(lap_scores, 10)) if lap_scores else 80.0
print(f"  Blur threshold (P10): {blur_threshold:.1f}")

seen_sha, seen_ahash = {}, []
accepted, rejected = [], []

for r in candidates:
    p = r["image_path"]
    flags = []
    try:
        r["sha256"] = sha256_file(p)
        r["ahash"] = average_hash(p)
        r["lap_score"] = round(laplacian_score(p), 2)
        r["file_size_kb"] = round(os.path.getsize(p) / 1024, 1)

        # SHA-256 ??
        if r["sha256"] in seen_sha:
            flags.append(f"exact_dup:{seen_sha[r['sha256']]}")
        else:
            seen_sha[r["sha256"]] = r["sample_id"]

        # aHash ???
        if not flags:
            for prev_ah, prev_sid in seen_ahash:
                if hamming_distance(r["ahash"], prev_ah) <= AHASH_THRESHOLD:
                    flags.append(f"near_dup:{prev_sid}")
                    break
            seen_ahash.append((r["ahash"], r["sample_id"]))

        # ??
        if r["lap_score"] < blur_threshold:
            flags.append("blur")

        # ??/???
        arr = np.asarray(Image.open(p).convert("RGB"))
        if arr.std() < 3:
            flags.append("near_blank")

    except Exception as e:
        flags.append(f"read_error:{type(e).__name__}")
        r["sha256"], r["ahash"], r["lap_score"], r["file_size_kb"] = "", "", 0, 0

    r["flags"] = "|".join(flags) if flags else "clean"
    r["status"] = "rejected" if flags else "accepted"
    (rejected if flags else accepted).append(r)

# ???????????????????????????????????????
# Step 3: ?????????
# ???????????????????????????????????????
print(f"\n[3/4] Saving results: {len(accepted)} accepted / {len(rejected)} rejected")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
acc_dir = OUTPUT_DIR / "accepted"
rej_dir = OUTPUT_DIR / "rejected"
acc_dir.mkdir(exist_ok=True); rej_dir.mkdir(exist_ok=True)

# ?? accepted
accepted_rows = []
for i, r in enumerate(accepted, 1):
    aid = f"A{i:04d}"
    dst = acc_dir / f"{aid}.png"
    shutil.copy2(r["image_path"], dst)
    accepted_rows.append({"accepted_id": aid, "source": r["filename"],
                          "lap_score": r["lap_score"], "sha256": r["sha256"]})

# ??????????? rejected
reject_categories = {"exact_dup": [], "near_dup": [], "blur": [], "near_blank": [], "read_error": [], "other": []}
for r in rejected:
    f = r["flags"]
    if   "exact_dup" in f: reject_categories["exact_dup"].append(r)
    elif "near_dup"  in f: reject_categories["near_dup"].append(r)
    elif "blur"      in f: reject_categories["blur"].append(r)
    elif "near_blank" in f: reject_categories["near_blank"].append(r)
    elif "read_error" in f: reject_categories["read_error"].append(r)
    else:                   reject_categories["other"].append(r)

rejected_rows = []
for cat, rows in reject_categories.items():
    if not rows: continue
    cat_dir = rej_dir / cat
    cat_dir.mkdir(exist_ok=True)
    for i, r in enumerate(rows, 1):
        dst = cat_dir / f"R{i:04d}_{r['filename']}"[:240]  # ??????
        shutil.copy2(r["image_path"], dst)
        rejected_rows.append({"reject_reason": cat, "source": r["filename"],
                              "flags": r["flags"], "lap_score": r["lap_score"]})

# manifests
with (OUTPUT_DIR / "accepted_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["accepted_id", "source", "lap_score", "sha256"])
    w.writeheader(); w.writerows(accepted_rows)

with (OUTPUT_DIR / "rejected_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=["reject_reason", "source", "flags", "lap_score"])
    w.writeheader(); w.writerows(rejected_rows)

# summary
blur_n = sum(1 for r in rejected if "blur" in r.get("flags",""))
dup_n = sum(1 for r in rejected if "dup" in r.get("flags",""))
blank_n = sum(1 for r in rejected if "near_blank" in r.get("flags",""))

with (OUTPUT_DIR / "cleaning_summary.json").open("w", encoding="utf-8") as f:
    json.dump({
        "candidate_count": len(candidates), "accepted_count": len(accepted),
        "rejected_count": len(rejected), "blur_count": blur_n, "dup_count": dup_n,
        "blank_count": blank_n, "blur_threshold": blur_threshold,
        "reject_breakdown": {k: len(v) for k, v in reject_categories.items()},
    }, f, ensure_ascii=False, indent=2)

# ???????????????????????????????????????
# Step 4: ?? accepted zip
# ???????????????????????????????????????
print("\n[4/4] Packaging cleaned_accepted.zip ...")
zip_path = Path("/kaggle/working/cleaned_accepted.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
    for p in acc_dir.iterdir():
        z.write(p, p.name)

size_mb = zip_path.stat().st_size / 1e6
print(f"\n{'='*60}")
print(f"  CLEANING DONE")
print(f"  Accepted: {len(accepted)}/{len(candidates)} ({len(accepted)*100/max(len(candidates),1):.1f}%)")
print(f"  Rejected: {len(rejected)}")
print(f"    - exact_dup:  {reject_categories['exact_dup'].__len__()}")
print(f"    - near_dup:   {reject_categories['near_dup'].__len__()}")
print(f"    - blur:       {reject_categories['blur'].__len__()}")
print(f"    - near_blank: {reject_categories['near_blank'].__len__()}")
print(f"    - read_error: {reject_categories['read_error'].__len__()}")
print(f"  ?????????????????????????????????????????????")
print(f"  cleaned_accepted.zip: {size_mb:.1f} MB")
print(f"{'='*60}")
print(f"\n??????:")
print(f"  SDXL_Cleaned/accepted/   ? ???????")
print(f"  SDXL_Cleaned/rejected/   ? ???????????")
print(f"  cleaned_accepted.zip     ? ??????????")
