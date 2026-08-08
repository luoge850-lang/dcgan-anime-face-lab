"""
SDXL Cleaning ? Kaggle ? ?? ???? /kaggle/input/ ?? SDXL ???
================================================================================
??????:
  1. ???? (SHA-256, aHash ???, Laplacian ??, ??)
     ? ?? review_template.csv (? auto_flags, auto_pass ?)
  2. ???? (?? review_template.csv ? Excel ?? manual_keep=1/0
     ? ??? review_checked.csv ? ??? Kaggle Dataset ? ?????)

?? (???? /kaggle/input/):
  - SDXL ?? zip ?? (? candidate_manifest.csv ? images_64/ ? .png/.jpg ??)
  - ???: review_checked.csv

?? (/kaggle/working/SDXL_Cleaned/):
  - accepted_64/          ? ????? 64?64 ??
  - accepted_manifest.csv ? ??????
  - rejected_manifest.csv ? ????? (?????)
  - review_template.csv   ? ??????
  - cleaning_summary.json ? ????

????:
  1. Kaggle Notebook ? T4?2 GPU (CPU ??), Internet OFF ??
  2. ? SDXL ?? zip ??? Kaggle Dataset ? attach
  3. ???????? cell ? Run
  4. ????????? ? ?? review_template.csv
  5. Excel ?? ? ?? manual_keep ?: 1=?? 0=?? ? ??? review_checked.csv
  6. ? review_checked.csv ???? Kaggle Dataset ? attach ? ?????
  7. ????? accepted_64/ + accepted_manifest.csv (????? cleaned_accepted.zip)
"""

from __future__ import annotations
import csv, hashlib, json, os, random, shutil, sys, time, zipfile
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image

# ???????????????????????????????????????????????????????????????
# ??
# ???????????????????????????????????????????????????????????????
OUTPUT_DIR = Path("/kaggle/working/SDXL_Cleaned")
INPUT_ZIP_GLOB = "/kaggle/input/**/*.zip"        # ?????? zip
INPUT_DIR_SEARCH = "/kaggle/input/"               # ?????????

REVIEW_CSV_NAME = "review_checked.csv"            # ???: /kaggle/input/ ?????
BLUR_THRESHOLD = 0                                # 0 = ????????? P10
AHASH_THRESHOLD = 8                               # ???? ? 8 = ???
SEED = 42

# ???????????????????????????????????????????????????????????????
# ????
# ???????????????????????????????????????????????????????????????
def find_files(root: Path, exts: set[str] | None = None) -> list[Path]:
    """???????, ??????"""
    if exts is None:
        exts = {".png", ".jpg", ".jpeg", ".webp"}
    files = []
    if not root.exists():
        return files
    for item in root.rglob("*"):
        if item.is_file() and item.suffix.lower() in exts:
            # ??????
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
    """????: ??? size?size ? ?? ? ????????1 ? 64?hex"""
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    bits = (a > a.mean()).astype(np.uint8).ravel()
    # ?? hex ???
    return hashlib.sha1(bits.tobytes()).hexdigest()


def hamming_distance(h1: str, h2: str) -> int:
    """?? hex ??????? (?? hex?bytes ??)"""
    b1 = bytes.fromhex(h1[:40])  # sha1 hex = 40 chars = 20 bytes
    b2 = bytes.fromhex(h2[:40])
    return sum(bin(x ^ y).count("1") for x, y in zip(b1, b2))


def laplacian_score(path: Path) -> float:
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    p = np.pad(a, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    return float(lap.var())


# ???????????????????????????????????????????????????????????????
# ????
# ???????????????????????????????????????????????????????????????
def discover_candidates() -> list[dict]:
    """
    ???? SDXL ????
    ??:
      1. ?? /kaggle/input/ ?? candidate_manifest.csv ? ??
      2. ?? zip ?? ? ?? ? ? candidate_manifest.csv
      3. ??? ? ???? /kaggle/input/ ???????
    ??: [{"sample_id": ..., "image_path": Path, ...}, ...]
    """
    candidates = []
    seen_ids = set()

    # ?? ?? 1 & 2: ? manifest ??
    # ????? zip
    extracted_roots = []
    for zp in sorted(Path("/kaggle/input").rglob("*.zip")):
        if "cleaned" in zp.name.lower() or "accepted" in zp.name.lower():
            continue  # ??????
        extract_root = Path("/kaggle/working/_extracted") / zp.stem
        if not extract_root.exists():
            extract_root.mkdir(parents=True, exist_ok=True)
            print(f"[input] Extracting {zp.name} ({zp.stat().st_size/1e6:.0f}MB) ...")
            with zipfile.ZipFile(zp) as z:
                z.extractall(extract_root)
        extracted_roots.append(extract_root)

    # ???? candidate_manifest.csv
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

    # ?? ?? 3: ????? (?? manifest) ??
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

    # ????
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


# ???????????????????????????????????????????????????????????????
# ????
# ???????????????????????????????????????????????????????????????
def calibrate_blur_threshold(rows: list[dict], max_samples: int = 2000) -> float:
    """???? Laplacian ??? P10 ??????"""
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

                # SHA-256 ????
                if digest in seen_sha:
                    flags.append(f"exact_dup:{seen_sha[digest]}")
                else:
                    seen_sha[digest] = row["sample_id"]

                # aHash ???
                for prev_ah, prev_sid, prev_hex in seen_ahash:
                    if hamming_distance(ah, prev_ah) <= AHASH_THRESHOLD:
                        flags.append(f"near_dup:{prev_sid}")
                        break
                seen_ahash.append((ah, row["sample_id"], ah[:16]))

                # ??
                if score < threshold:
                    flags.append("blur")

                # ??
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


# ???????????????????????????????????????????????????????????????
# ????
# ???????????????????????????????????????????????????????????????
def find_review_csv() -> Path | None:
    """? /kaggle/input/ ?? review_checked.csv"""
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


# ???????????????????????????????????????????????????????????????
# ???? ? Accepted
# ???????????????????????????????????????????????????????????????
def stratified_select(rows: list[dict]) -> list[dict]:
    """
    ? eligible ?????, ???? (pose, hair) ?????
    ?? eligible ??????, ?????
    """
    # ???: ? auto_pass
    # ???: auto_pass + manual_keep == 1
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


# ???????????????????????????????????????????????????????????????
# ???
# ???????????????????????????????????????????????????????????????
def main():
    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED)

    candidates = discover_candidates()
    if not candidates:
        print("[error] No images found. Upload SDXL candidate zip as Kaggle Dataset and attach.")
        return

    # ?? ???? ??
    rows, threshold = automatic_flags(candidates)
    auto_pass = sum(1 for r in rows if r.get("auto_pass") == "1")
    blur_n = sum(1 for r in rows if "blur" in (r.get("auto_flags", "")))
    dup_n = sum(1 for r in rows if "dup" in (r.get("auto_flags", "")))
    blank_n = sum(1 for r in rows if "near_blank" in (r.get("auto_flags", "")))
    print(f"[auto] pass={auto_pass}/{len(rows)}  blur={blur_n}  dup={dup_n}  blank={blank_n}  "
          f"threshold={threshold:.1f}  ({time.time()-t0:.0f}s)")

    # ?? ????? review_checked.csv ??
    review_path = find_review_csv()
    if review_path:
        rows = read_review(rows, review_path)

    # ?? ?????: ????, ?? ??
    if not review_path:
        template = write_review_template(rows)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "cleaning_state.json").write_text(json.dumps({
            "stage": "waiting_for_review",
            "candidate_count": len(rows),
            "auto_pass_count": auto_pass,
            "blur_count": blur_n, "dup_count": dup_n, "blank_count": blank_n,
            "blur_threshold": threshold,
            "next": "Download review_template.csv from Kaggle output ? mark manual_keep ? "
                    "save as review_checked.csv ? upload as Kaggle Dataset ? re-run this script.",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"  ?  WAITING FOR REVIEW")
        print(f"  ?????????????????????????????????????????????")
        print(f"  Auto-pass: {auto_pass}/{len(rows)} candidates")
        print(f"  ?????????????????????????????????????????????")
        print(f"  1. ? Kaggle Output ?? review_template.csv")
        print(f"  2. Excel ?? ? ?? manual_keep ? (1=??, 0=??)")
        print(f"  3. ??? review_checked.csv")
        print(f"  4. ??? Kaggle Dataset ? attach ? ?????")
        print(f"{'='*60}")
        return

    # ?? ?????: ?? accepted ??
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

    # ?? manifests
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

    # ??
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

    # ??
    zip_path = Path("/kaggle/working/cleaned_accepted.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                z.write(str(f), str(f.relative_to(OUTPUT_DIR)))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"  ? CLEANING DONE")
    print(f"  Accepted: {len(accepted_rows)} / {len(rows)} ({len(accepted_rows)*100/max(len(rows),1):.1f}%)")
    print(f"  Rejected: {len(rows) - len(accepted_rows)}")
    print(f"  ?????????????????????????????????????????????")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Zip: cleaned_accepted.zip ({size_mb:.0f}MB)")
    print(f"{'='*60}")
    print(f"\n  ???:")
    print(f"  1. ?? cleaned_accepted.zip")
    print(f"  2. ??? Kaggle Dataset (for ????)")
    print(f"  3. ??? notebook ? attach")


if __name__ == "__main__":
    main()
