"""
SDXL Cleaning — Colab 粘贴即跑。
================================================================================
两轮清洗流程:
  1. 自动筛选 (SHA-256 去重, aHash 近重复, Laplacian 模糊, 纯色检测)
  2. 人工审核 (生成 review_template.csv → 人工标记 → 重新运行)

输入:
  把 SDXL 候选 zip 拖入 Colab 文件浏览器 → /content/
  (从 Kaggle 下载的 all_candidates_cumulative.zip 或类似)

输出:
  cleaned_accepted.zip  — 通过清洗的 64×64 图片 + accepted_manifest.csv

使用方法:
  1. colab.research.google.com → New Notebook → T4 GPU
  2. 把候选 zip 拖入 /content/
  3. 本文件全文粘贴到一个 cell → Ctrl+Enter
  4. 第一轮: 自动筛选 → 下载 review_template.csv, 标记 manual_keep 列
  5. 第二轮: 把标记好的 review_checked.csv 拖入, 重新运行本脚本
  6. 下载 cleaned_accepted.zip
"""

from __future__ import annotations
import csv, hashlib, json, os, random, shutil, subprocess, sys, time, zipfile
import importlib.util
from pathlib import Path
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# 0. 依赖
# ═══════════════════════════════════════════════════════════════════
def _install() -> None:
    pkgs = [("numpy", "numpy"), ("Pillow>=10.0.0", "PIL")]
    missing = [p for p, m in pkgs if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_install()
import numpy as np
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# 1. 配置
# ═══════════════════════════════════════════════════════════════════
# 支持多个输入 zip（用逗号分隔）：SDXL_INPUT_ZIPS=/content/kaggle_1600.zip,/content/colab_1000.zip
# 或者用通配符：SDXL_INPUT_GLOB=/content/*candidate*.zip
# 也可以直接放目录：SDXL_INPUT_DIR=/content/SDXL_Production_v3
INPUT_ZIPS = os.environ.get("SDXL_INPUT_ZIPS", "")
INPUT_ZIP  = Path(os.environ.get("SDXL_INPUT_ZIP", "/content/all_candidates_cumulative.zip"))
INPUT_DIR  = Path(os.environ.get("SDXL_INPUT_DIR", "/content/SDXL_Production_v3"))
INPUT_GLOB = os.environ.get("SDXL_INPUT_GLOB", "/content/*SDXL*.zip")
OUTPUT_DIR = Path(os.environ.get("SDXL_CLEAN_OUTPUT", "/content/SDXL_Cleaned"))

REVIEW_CSV      = os.environ.get("SDXL_REVIEW_CSV", "")  # 第二轮: review_checked.csv 路径
TARGET_ACCEPTED = int(os.environ.get("SDXL_TARGET", "1100"))  # 1600 candidates × 70% ≈ 1120
REQUIRE_MANUAL  = os.environ.get("SDXL_REQUIRE_MANUAL", "1") == "1"
BLUR_THRESHOLD  = float(os.environ.get("SDXL_BLUR_THRESHOLD", "0"))  # 0 = 自动校准
SEED            = int(os.environ.get("SDXL_CLEAN_SEED", "42"))


# ═══════════════════════════════════════════════════════════════════
# 2. 工具函数
# ═══════════════════════════════════════════════════════════════════
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_hash(path: Path, size: int = 16) -> str:
    im = Image.open(path).convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.float32)
    return hashlib.sha1((a > a.mean()).astype(np.uint8).tobytes()).hexdigest()


def laplacian_score(path: Path) -> float:
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
    p = np.pad(a, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    return float(lap.var())


def _collect_zips() -> list[Path]:
    """收集所有输入 zip 路径"""
    zips: list[Path] = []

    # 1. SDXL_INPUT_ZIPS: 逗号分隔的显式列表
    if INPUT_ZIPS:
        for p in INPUT_ZIPS.split(","):
            p = p.strip()
            if p:
                zp = Path(p)
                if zp.exists():
                    zips.append(zp)
                else:
                    print(f"[input] WARNING: zip not found: {p}")

    # 2. SDXL_INPUT_ZIP: 单个 zip (兼容旧版)
    if INPUT_ZIP.exists() and INPUT_ZIP not in zips:
        zips.append(INPUT_ZIP)

    # 3. SDXL_INPUT_GLOB: 通配符匹配
    if INPUT_GLOB and not zips:
        import glob as _glob
        for p in sorted(_glob.glob(INPUT_GLOB)):
            zp = Path(p)
            if zp not in zips:
                zips.append(zp)

    return zips


def resolve_input() -> list[tuple[Path, Path]]:
    """收集所有输入源 → 返回 [(manifest_path, image_root), ...]"""
    results: list[tuple[Path, Path]] = []

    # A. 已有目录
    if INPUT_DIR.exists():
        manifests = sorted(INPUT_DIR.rglob("candidate_manifest.csv"))
        for mf in manifests:
            results.append((mf, mf.parent))
            print(f"[input] Directory: {mf}")

    # B. 多个 zip
    zips = _collect_zips()
    for zp in zips:
        extract_root = OUTPUT_DIR.parent / f"_extracted_{zp.stem}"
        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        print(f"[input] Extracting {zp.name} ({zp.stat().st_size/1e6:.0f}MB) ...")
        with zipfile.ZipFile(zp) as z:
            z.extractall(extract_root)
        inner_manifests = sorted(extract_root.rglob("candidate_manifest.csv"))
        for mf in inner_manifests:
            results.append((mf, mf.parent))
            print(f"[input] Zip '{zp.name}' → manifest: {mf}")

    if not results:
        raise FileNotFoundError(
            f"No input found. Options:\n"
            f"  1. SDXL_INPUT_ZIPS=/content/a.zip,/content/b.zip\n"
            f"  2. SDXL_INPUT_GLOB=/content/*SDXL*.zip\n"
            f"  3. SDXL_INPUT_ZIP=/content/all_candidates_cumulative.zip\n"
            f"  4. SDXL_INPUT_DIR=/content/SDXL_Production_v3"
        )
    return results


def calibrate_blur_threshold(rows: list[dict], input_root: Path) -> float:
    """用候选图自身的分布校准模糊阈值 (P10)"""
    scores = []
    for r in rows[: min(len(rows), 2000)]:
        p = input_root / r.get("image_64_path", "")
        if p.exists():
            try:
                scores.append(laplacian_score(p))
            except Exception:
                pass
    if not scores:
        return 80.0  # 默认值
    p10 = float(np.percentile(scores, 10))
    print(f"[blur] Auto threshold: p10={p10:.1f} (from {len(scores)} candidates)")
    return p10


def automatic_flags(rows: list[dict], input_root: Path) -> tuple[list[dict], float]:
    threshold = BLUR_THRESHOLD if BLUR_THRESHOLD > 0 else calibrate_blur_threshold(rows, input_root)
    if threshold == 0:
        threshold = 80.0

    seen_sha: dict[str, str] = {}
    seen_hash: dict[str, str] = {}

    for row in rows:
        p = input_root / row.get("image_64_path", "")
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
                    flags.append(f"exact_duplicate_of:{seen_sha[digest]}")
                else:
                    seen_sha[digest] = row["sample_id"]

                # aHash 近重复
                if ah in seen_hash:
                    flags.append(f"near_duplicate_of:{seen_hash[ah]}")
                else:
                    seen_hash[ah] = row["sample_id"]

                # 模糊
                if score < threshold:
                    flags.append("blur")

                # 纯色/近空白
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


def write_review_template(rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "review_template.csv"
    fields = ["sample_id", "target_pose", "target_hair", "preview_path",
              "image_64_path", "auto_flags", "auto_pass", "manual_keep", "reject_reason"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    return path


def read_review(rows: list[dict]) -> list[dict]:
    if not REVIEW_CSV:
        return rows
    review_path = Path(REVIEW_CSV)
    if not review_path.exists():
        raise FileNotFoundError(f"SDXL_REVIEW_CSV does not exist: {review_path}")
    decisions = {}
    with review_path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            decisions[r["sample_id"]] = r
    for r in rows:
        d = decisions.get(r["sample_id"], {})
        r["manual_keep"] = d.get("manual_keep", "").strip()
        r["reject_reason"] = d.get("reject_reason", "").strip()
    return rows


def stratified_select(rows: list[dict]) -> list[dict]:
    """分层采样: 每个 (pose, hair) 组合均匀选取, 尽量达到 TARGET"""
    eligible = []
    for r in rows:
        manual_ok = r.get("manual_keep", "").strip() == "1"
        auto_ok = r.get("auto_pass", "") == "1"
        if REQUIRE_MANUAL:
            if auto_ok and manual_ok:
                eligible.append(r)
        else:
            if auto_ok:
                eligible.append(r)

    if len(eligible) < TARGET_ACCEPTED:
        # 不强制报错 — 只是警告并返回所有 eligible
        print(f"[warning] Only {len(eligible)} eligible candidates; target was {TARGET_ACCEPTED}")
        print(f"  Will use all {len(eligible)}. Generate more candidates if more needed.")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in eligible:
        groups[(r.get("target_pose", ""), r.get("target_hair", ""))].append(r)

    rng = random.Random(SEED)
    for group in groups.values():
        rng.shuffle(group)

    keys = sorted(groups)
    selected: list[dict] = []
    actual_target = min(TARGET_ACCEPTED, len(eligible))
    while len(selected) < actual_target:
        changed = False
        for key in keys:
            if groups[key] and len(selected) < actual_target:
                selected.append(groups[key].pop())
                changed = True
        if not changed:
            break

    return selected


# ═══════════════════════════════════════════════════════════════════
# 3. 主流程
# ═══════════════════════════════════════════════════════════════════
def main():
    input_sources = resolve_input()

    # 合并所有 manifest
    all_rows: list[dict] = []
    seen_ids: dict[str, int] = {}  # sample_id → row_index (0-first)
    collision_count = 0

    for manifest_path, image_root in input_sources:
        with manifest_path.open(newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["_image_root"] = str(image_root)
            row["_manifest_path"] = str(manifest_path)
            sid = row.get("sample_id", "")
            if sid and sid in seen_ids:
                collision_count += 1
                # 自动重命名冲突的 sample_id
                new_sid = f"{sid}_{manifest_path.parent.name}"
                print(f"  [collision] {sid} in {manifest_path.parent.name} → renamed to {new_sid}")
                row["sample_id"] = new_sid
                row["_original_sample_id"] = sid
            if row["sample_id"]:
                seen_ids[row["sample_id"]] = len(all_rows)
            all_rows.append(row)
        print(f"[input] {len(rows)} rows from {manifest_path}")

    if collision_count:
        print(f"[input] ⚠ {collision_count} sample_id collisions resolved by renaming")

    print(f"[input] Total: {len(all_rows)} candidates from {len(input_sources)} source(s)")

    # 统一解析 image_64_path
    for row in all_rows:
        img_path = row.get("image_64_path", "")
        img_root = Path(row["_image_root"])
        row["_path"] = str(img_root / img_path)

    rows = all_rows

    # 自动筛选
    rows, threshold = automatic_flags(rows, input_root)
    auto_pass = sum(1 for r in rows if r.get("auto_pass") == "1")
    blur_count = sum(1 for r in rows if "blur" in (r.get("auto_flags", "")))
    dup_count = sum(1 for r in rows if "duplicate" in (r.get("auto_flags", "")))
    print(f"[auto] pass={auto_pass}/{len(rows)}  blur={blur_count}  dup={dup_count}  threshold={threshold:.1f}")

    # 生成审核模板
    review_path = write_review_template(rows)

    # 读取人工审核 (如果提供)
    rows = read_review(rows)

    if not REVIEW_CSV:
        # 第一轮: 仅自动筛选, 输出审核模板
        (OUTPUT_DIR / "cleaning_summary.json").write_text(json.dumps({
            "stage": "automatic_screen_only",
            "candidate_count": len(rows),
            "auto_pass_count": auto_pass,
            "blur_count": blur_count,
            "dup_count": dup_count,
            "blur_threshold": threshold,
            "target_accepted": TARGET_ACCEPTED,
            "next": "Fill review_template.csv → save as review_checked.csv → re-run with SDXL_REVIEW_CSV=/content/SDXL_Cleaned/review_checked.csv"
        }, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\n{'='*60}")
        print(f"  REVIEW REQUIRED")
        print(f"  ─────────────────────────────────────────────")
        print(f"  1. 从 Colab 文件浏览器下载 review_template.csv")
        print(f"  2. 用 Excel/Google Sheets 打开")
        print(f"  3. 标记 manual_keep 列: 1=保留, 0=丢弃")
        print(f"  4. 保存为 review_checked.csv")
        print(f"  5. 拖入 Colab, 重新运行本脚本")
        print(f"     (会自动读取 SDXL_REVIEW_CSV)")
        print(f"{'='*60}")
        return

    # 第二轮: 分层采样 → 生成 accepted
    selected = stratified_select(rows)
    accepted_dir = OUTPUT_DIR / "accepted_64"
    accepted_dir.mkdir(parents=True, exist_ok=True)

    accepted_rows = []
    selected_ids = {r["sample_id"] for r in selected}
    for i, r in enumerate(selected, 1):
        src = Path(r["_path"])
        dst = accepted_dir / f"A{i:04d}.png"
        shutil.copy2(str(src), str(dst))
        accepted_rows.append({
            "accepted_id": f"A{i:04d}",
            "source_sample_id": r["sample_id"],
            "target_pose": r.get("target_pose", ""),
            "target_hair": r.get("target_hair", ""),
            "path": f"accepted_64/A{i:04d}.png",
            "sha256_64": sha256_file(dst),
            "source_average_hash": r.get("average_hash", ""),
        })

    # 保存 accepted manifest
    with (OUTPUT_DIR / "accepted_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(accepted_rows[0].keys()))
        writer.writeheader(); writer.writerows(accepted_rows)

    # 保存 rejected manifest
    with (OUTPUT_DIR / "rejected_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "auto_flags", "manual_keep", "reject_reason"])
        writer.writeheader()
        for r in rows:
            if r["sample_id"] not in selected_ids:
                writer.writerow({k: r.get(k, "") for k in ["sample_id", "auto_flags", "manual_keep", "reject_reason"]})

    # 总结
    (OUTPUT_DIR / "cleaning_summary.json").write_text(json.dumps({
        "stage": "accepted_manifest_ready",
        "candidate_count": len(rows),
        "accepted_count": len(accepted_rows),
        "target_accepted": TARGET_ACCEPTED,
        "blur_threshold": threshold,
        "require_manual": REQUIRE_MANUAL,
        "pose_hair_distribution": {
            f"{k[0]}_{k[1]}": len([r for r in accepted_rows if r["target_pose"] == k[0] and r["target_hair"] == k[1]])
            for k in sorted(set((r["target_pose"], r["target_hair"]) for r in accepted_rows))
        }
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打包
    zip_path = OUTPUT_DIR.parent / "cleaned_accepted.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for f in OUTPUT_DIR.rglob("*"):
            if f.is_file():
                z.write(str(f), str(Path("SDXL_Cleaned") / f.relative_to(OUTPUT_DIR)))

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"  CLEANING DONE")
    print(f"  Accepted: {len(accepted_rows)} / {len(rows)} candidates")
    print(f"  ─────────────────────────────────────────────")
    print(f"  💾 下载: Colab 左侧文件浏览器")
    print(f"     → 右键 cleaned_accepted.zip ({size_mb:.0f}MB)")
    print(f"     → Download")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
