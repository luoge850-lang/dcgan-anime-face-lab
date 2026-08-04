"""
SDXL数据增强实验 Step 0：只读审计Kaggle Anime Faces原始数据。

本脚本不删除文件、不训练模型、不修改输入数据。
它只生成：
  - original_manifest.csv：每个图片的相对路径、尺寸、文件大小、SHA-256
  - duplicate_groups.json：内容完全相同的文件分组
  - audit_summary.json：数量、尺寸、重复数和审计结论

Kaggle运行方式：
  1. 只挂载原始 anime-faces 数据集；
  2. 运行本脚本；
  3. 下载 /kaggle/working/sdxl_data_audit；
  4. 确认路径数约为 21K，并分别记录唯一 SHA-256 数量后再进入B1训练。
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


DATASET_PATH = os.environ.get(
    "ANIME_FACES_DATASET",
    "/kaggle/input/anime-faces",
)
OUTPUT_ROOT = Path(
    os.environ.get("AUDIT_OUTPUT_ROOT", "/kaggle/working/sdxl_data_audit")
)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
CHUNK_SIZE = 1024 * 1024


def find_images(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    paths = []
    nested_copy = root / "data"
    for path in root.rglob("*"):
        # In the uploaded Kaggle dataset, data/data is a second copy of data.
        # Ignore it while preserving any other legitimate subdirectories.
        if nested_copy in path.parents:
            continue
        if path.is_file() and path.suffix.lower() in EXTENSIONS:
            paths.append(path)
    return sorted(paths)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(CHUNK_SIZE)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_size(path: Path) -> tuple[int, int, str | None]:
    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height), image.mode
    except Exception as exc:  # keep the audit complete and report bad files
        return -1, -1, f"ERROR:{type(exc).__name__}"


def main() -> None:
    root = Path(DATASET_PATH).resolve()
    # The Kaggle dataset mount contains ``data/`` and, in some uploads, a
    # nested ``data/data/`` copy.  The formal audit must use exactly one
    # top-level data folder; scanning the parent recursively would count both
    # copies and silently change the training distribution.
    scan_root = root / "data" if (root / "data").is_dir() else root
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = find_images(scan_root)
    if not paths:
        raise RuntimeError(f"No image files found under {scan_root}")

    rows: list[dict[str, object]] = []
    groups: dict[str, list[str]] = defaultdict(list)
    size_counter: Counter[tuple[int, int]] = Counter()
    bad_files: list[str] = []

    for index, path in enumerate(paths, 1):
        width, height, mode = inspect_size(path)
        digest = sha256_file(path)
        relative = path.relative_to(scan_root).as_posix()
        row = {
            "relative_path": relative,
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
            "mode": mode,
            "sha256": digest,
        }
        rows.append(row)
        groups[digest].append(relative)
        if width > 0 and height > 0:
            size_counter[(width, height)] += 1
        else:
            bad_files.append(relative)
        if index % 1000 == 0 or index == len(paths):
            print(f"[audit] {index}/{len(paths)} files")

    duplicate_groups = {
        digest: members for digest, members in groups.items() if len(members) > 1
    }
    duplicate_file_count = sum(len(members) - 1 for members in duplicate_groups.values())
    unique_count = len(groups)
    summary = {
        "dataset_root": str(root),
        "scan_root": str(scan_root),
        "nested_data_copy_detected": (root / "data" / "data").is_dir(),
        "image_count_before_dedup": len(paths),
        "unique_sha256_count": unique_count,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count_excluding_one_copy_per_group": duplicate_file_count,
        "bad_file_count": len(bad_files),
        "top_sizes": [
            {"size": f"{width}x{height}", "count": count}
            for (width, height), count in size_counter.most_common(20)
        ],
        "expected_project_condition": {
            "approximately_21k_paths": 19000 <= len(paths) <= 23000,
            "approximately_21k_unique_images": 19000 <= unique_count <= 23000,
            "exact_duplicates_present": duplicate_file_count > 0,
        },
        "notes": [
            "This is an audit only; no input file is deleted or changed.",
            "Use SHA-256 equality to decide exact duplicate removal before the cleaned-mixture runs.",
            "The formal baseline and all mixture runs use one path per SHA-256 group.",
            "Near-duplicates still require a later perceptual/semantic dedup step.",
        ],
    }

    with (OUTPUT_ROOT / "original_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Formal training uses one representative path per exact-content hash.
    # The path-full manifest is retained only as an audit record.
    seen_hashes: set[str] = set()
    unique_rows = []
    for row in rows:
        digest = str(row["sha256"])
        if digest not in seen_hashes:
            seen_hashes.add(digest)
            unique_rows.append(row)
    with (OUTPUT_ROOT / "original_unique_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(unique_rows)

    (OUTPUT_ROOT / "duplicate_groups.json").write_text(
        json.dumps(duplicate_groups, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "bad_files.json").write_text(
        json.dumps(bad_files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUTPUT_ROOT / "audit_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
