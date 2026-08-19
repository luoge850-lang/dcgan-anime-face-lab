"""
Task 3 / 03A
Prepare a reproducible calibration and evaluation protocol for FP32/FP16/INT8 PTQ.

Important distinction:
  - real anime images are reference/evaluation data;
  - latent vectors are the actual calibration inputs for a Generator.

This script does not build an engine and does not compute FID.  It creates one
portable package that the later FP16, INT8 and evaluation scripts can reuse.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SEED = 20260811


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", default="", help="Image root; auto-detect under /kaggle/input when omitted")
    parser.add_argument("--output-root", default="/kaggle/working/dcgan_output/Deployment_Optimization_Results/03_Quantization/03A_Protocol")
    parser.add_argument("--calibration-count", type=int, default=100)
    parser.add_argument("--real-eval-count", type=int, default=5000)
    parser.add_argument("--latent-calibration-count", type=int, default=512)
    parser.add_argument("--latent-eval-count", type=int, default=5000)
    parser.add_argument("--z-dim", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-zip", action="store_true", help="Do not create the portable ZIP package")
    args, _unknown = parser.parse_known_args()
    return args


def iter_images(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for name in sorted(filenames):
            path = Path(directory) / name
            if path.suffix.lower() in IMAGE_EXTS:
                yield path


def detect_dataset(explicit: str) -> tuple[Path, list[Path]]:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        paths = sorted(iter_images(root))
        if not paths:
            raise FileNotFoundError(f"No supported images found under --dataset-path: {root}")
        return root, paths

    candidates: list[tuple[int, Path, list[Path]]] = []
    input_root = Path("/kaggle/input")
    roots: list[Path] = []
    if input_root.is_dir():
        roots.extend(sorted(p for p in input_root.iterdir() if p.is_dir()))
        roots.append(input_root)
    roots.append(Path.cwd())
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve())
        if key in seen or not root.exists():
            continue
        seen.add(key)
        paths = sorted(iter_images(root))
        if paths:
            candidates.append((len(paths), root, paths))
    if not candidates:
        raise FileNotFoundError("No image dataset found. Set --dataset-path explicitly.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    count, root, paths = candidates[0]
    print(f"[dataset] auto-detected {root} ({count} images)")
    return root, paths


def normalize_image(path: Path, image_size: int) -> tuple[bytes, Image.Image]:
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (image_size, image_size):
            image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        image.load()
    return image.tobytes(), image


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_unique_records(paths: Sequence[Path], image_size: int) -> tuple[list[dict], int, int]:
    records: list[dict] = []
    seen: set[str] = set()
    failed = 0
    for index, path in enumerate(paths, 1):
        try:
            pixels, _image = normalize_image(path, image_size)
            normalized_hash = sha256_bytes(pixels)
        except Exception as exc:  # keep audit running and record the failure count
            failed += 1
            print(f"[skip] {path}: {exc}")
            continue
        if normalized_hash in seen:
            continue
        seen.add(normalized_hash)
        records.append({
            "source_path": str(path),
            "source_name": path.name,
            "normalized_sha256": normalized_hash,
            "source_index": index,
        })
        if index % 1000 == 0:
            print(f"[audit] scanned={index} unique={len(records)} failed={failed}")
    records.sort(key=lambda row: (row["normalized_sha256"], row["source_path"]))
    return records, len(paths), failed


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def save_normalized_copy(record: dict, destination: Path, image_size: int, output_name: str) -> dict:
    pixels, image = normalize_image(Path(record["source_path"]), image_size)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / output_name
    image.save(target, format="PNG", optimize=False)
    return {
        "output_name": output_name,
        "source_path": record["source_path"],
        "normalized_sha256": sha256_bytes(pixels),
        "width": image_size,
        "height": image_size,
    }


def make_latents(count: int, z_dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((count, z_dim, 1, 1), dtype=np.float32)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def zip_directory(directory: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path != target:
                archive.write(path, path.relative_to(directory).as_posix())


def main() -> None:
    args = parse_args()
    if args.calibration_count <= 0 or args.real_eval_count <= 0:
        raise ValueError("calibration-count and real-eval-count must be positive")
    if args.latent_calibration_count <= 0 or args.latent_eval_count <= 0:
        raise ValueError("latent counts must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    dataset_root, paths = detect_dataset(args.dataset_path)
    records, scanned_count, failed_count = build_unique_records(paths, args.image_size)
    required = args.calibration_count + args.real_eval_count
    if len(records) < required:
        raise RuntimeError(
            f"Only {len(records)} unique normalized images available, but {required} are required "
            "for disjoint calibration and real-eval sets. Reduce --real-eval-count or check the dataset."
        )

    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists():
        # This script owns only its output root. Existing files are not deleted;
        # a timestamped run should be used when a clean protocol is required.
        print(f"[warn] output exists; files with the same names will be replaced: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    selected = rng.sample(records, required)
    calibration_records = selected[: args.calibration_count]
    real_eval_records = selected[args.calibration_count :]

    cal_rows = []
    for idx, record in enumerate(calibration_records):
        cal_rows.append(save_normalized_copy(record, output_root / "real_calibration_100", args.image_size, f"cal_{idx:05d}.png"))
    eval_rows = []
    for idx, record in enumerate(real_eval_records):
        eval_rows.append(save_normalized_copy(record, output_root / "real_eval", args.image_size, f"real_{idx:05d}.png"))

    write_csv(output_root / "real_calibration_100_manifest.csv", cal_rows[0].keys(), cal_rows)
    write_csv(output_root / "real_eval_manifest.csv", eval_rows[0].keys(), eval_rows)

    calibration_latents = make_latents(args.latent_calibration_count, args.z_dim, args.seed + 1)
    eval_latents = make_latents(args.latent_eval_count, args.z_dim, args.seed + 2)
    np.save(output_root / "latent_calibration.npy", calibration_latents)
    np.save(output_root / "latent_eval.npy", eval_latents)
    write_csv(
        output_root / "latent_manifest.csv",
        ["split", "count", "z_dim", "shape", "seed"],
        [
            {"split": "calibration", "count": len(calibration_latents), "z_dim": args.z_dim, "shape": list(calibration_latents.shape), "seed": args.seed + 1},
            {"split": "evaluation", "count": len(eval_latents), "z_dim": args.z_dim, "shape": list(eval_latents.shape), "seed": args.seed + 2},
        ],
    )

    protocol = {
        "protocol": "Task3_Quantization_03A",
        "seed": args.seed,
        "dataset_root": str(dataset_root),
        "scanned_file_count": scanned_count,
        "unique_normalized_image_count": len(records),
        "decode_failed_count": failed_count,
        "image_size": args.image_size,
        "calibration_real_count": len(cal_rows),
        "real_eval_count": len(eval_rows),
        "latent_calibration_count": len(calibration_latents),
        "latent_eval_count": len(eval_latents),
        "latent_shape": list(calibration_latents.shape),
        "normalization": "RGB; resize to 64x64 using PIL LANCZOS; PNG output",
        "quantization_input_rule": "Generator receives latent arrays; real images are evaluation/reference data, not generator input calibration",
        "fid_policy": {
            "historical": "preserve the project's existing legacy Inception FID for direct comparability",
            "supplemental_standard": "03D must compute the standard pytorch-fid Inception-v3 pool3 2048-d FID; missing dependencies are a run failure, not a silent fallback",
        },
        "next_steps": ["03B build FP16", "03C TensorRT INT8 PTQ", "03D fixed-latent quality evaluation", "03E report"],
    }
    write_json(output_root / "calibration_manifest.json", protocol)
    (output_root / "protocol_readme.txt").write_text(
        "真实图像用于评估参考；latent_calibration.npy 才是 Generator 的 INT8 激活校准输入。\n"
        "所有精度必须复用 latent_eval.npy，确保 FP32/FP16/INT8 逐样本公平比较。\n",
        encoding="utf-8",
    )

    if not args.no_zip:
        zip_path = output_root.parent / "Task3_03A_Quantization_Protocol.zip"
        zip_directory(output_root, zip_path)
        print(f"[zip] {zip_path}")
    print(f"[done] protocol={output_root}")
    print(f"[done] real_calibration={len(cal_rows)} real_eval={len(eval_rows)} latent_calibration={len(calibration_latents)} latent_eval={len(eval_latents)}")


if __name__ == "__main__":
    main()
