"""
Task 3 / 03D
Evaluate FP32, FP16 and INT8 TensorRT Generator engines with identical latent
inputs and an identical real reference set.

Required inputs:
  generator_trt_fp32.engine
  generator_trt_fp16.engine
  generator_trt_int8.engine
  Task3_03A_Quantization_Protocol.zip

The report keeps the project's historical torchvision Inception-v3 FID for
backward comparability and also computes the public ``pytorch-fid`` protocol
(2048-dim Inception-v3 pool3 features) as a separate standard-FID column.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw


IMAGE_SIZE = 64
NOISE_DIM = 128
TRT_LOGGER = None


def parse_args():
    parser = argparse.ArgumentParser(description="Task 3D quality evaluation")
    parser.add_argument("--fp32-engine", default=os.getenv("FP32_ENGINE_PATH", ""))
    parser.add_argument("--fp16-engine", default=os.getenv("FP16_ENGINE_PATH", ""))
    parser.add_argument("--int8-engine", default=os.getenv("INT8_ENGINE_PATH", ""))
    # 03A may be attached as the original ZIP or as an already extracted
    # Kaggle Dataset folder. Keep --protocol-zip as a backwards-compatible
    # alias, but treat both forms through one protocol-path argument.
    parser.add_argument(
        "--protocol-path",
        "--protocol-zip",
        dest="protocol_path",
        default=os.getenv("PROTOCOL_PATH", os.getenv("PROTOCOL_ZIP_PATH", "")),
        help="03A ZIP or extracted directory containing latent_eval.npy and real_eval/",
    )
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/03_Quantization/03D_Evaluation"))
    parser.add_argument("--n-fid", type=int, default=5000)
    parser.add_argument("--n-image-eval", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument("--lpips-pairs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--save-all-fakes", action="store_true", help="Save every generated PNG; otherwise save only a 64-image audit subset")
    args, _unknown = parser.parse_known_args()
    if min(args.n_fid, args.n_image_eval, args.batch_size, args.lpips_pairs) <= 0:
        raise ValueError("n-fid, n-image-eval, batch-size and lpips-pairs must be positive")
    if args.n_image_eval > args.n_fid:
        raise ValueError("n-image-eval cannot exceed n-fid")
    return args


def get_logger(trt):
    global TRT_LOGGER
    if TRT_LOGGER is None:
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return TRT_LOGGER


def ensure_tensorrt():
    try:
        import tensorrt as trt
        print(f"[TensorRT] {trt.__version__}")
        return trt
    except (ImportError, ModuleNotFoundError):
        # Kaggle images are not identical: some contain TensorRT already, while
        # others provide CUDA/PyTorch but omit the Python TensorRT bindings.
        # Install the CUDA-matched wheel once, then import it in this process.
        cuda_version = getattr(torch.version, "cuda", None)
        cuda_major = cuda_version.split(".", 1)[0] if cuda_version else ""
        candidates = []
        if cuda_major:
            candidates.append(f"tensorrt-cu{cuda_major}")
        # The metapackage is a safe fallback for runtimes where the CUDA
        # specific package name is not published.
        candidates.append("tensorrt")
        errors = []
        for package in candidates:
            print(f"[TensorRT] Python bindings not found; installing {package} ...")
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "--quiet",
                        "--no-cache-dir",
                        "--upgrade",
                        package,
                    ]
                )
                importlib.invalidate_caches()
                import tensorrt as trt
                print(f"[TensorRT] installed {trt.__version__}")
                return trt
            except Exception as exc:  # try the next compatible package name
                errors.append(f"{package}: {exc}")
        raise RuntimeError(
            "TensorRT is unavailable. Enable a Kaggle GPU and Internet, then "
            "restart the session. Automatic installation attempts failed: "
            + " | ".join(errors)
        )


def _engine_search_roots():
    # Kaggle normally has the uploaded Dataset under /kaggle/input and any
    # extracted/copied result under /kaggle/working.  The order is deliberate:
    # an uploaded engine is the preferred source when duplicate copies exist.
    return (
        ("/kaggle/input", Path("/kaggle/input")),
        ("/kaggle/working", Path("/kaggle/working")),
        ("cwd", Path.cwd()),
    )


def _engine_candidates(root: Path, name: str):
    if not root.exists():
        return []
    found = list(root.rglob(name))
    # Linux/Kaggle is case-sensitive.  This small fallback handles an upload
    # whose filename differs only in case without scanning every file twice.
    if not found:
        lowered = name.lower()
        suffix_pattern = re.compile(
            rf"^{re.escape(Path(name).stem)}(?: \(\d+\))?{re.escape(Path(name).suffix)}$",
            re.IGNORECASE,
        )
        target_normalized = re.sub(r"[^a-z0-9]", "", lowered)

        def normalized_filename(path):
            value = path.name.lower()
            value = re.sub(r"\s*\(\d+\)(?=\.[^.]+$)", "", value)
            return re.sub(r"[^a-z0-9]", "", value)

        try:
            found = [
                path for path in root.rglob("*")
                if path.is_file()
                and (
                    path.name.lower() == lowered
                    or suffix_pattern.match(path.name)
                    or normalized_filename(path) == target_normalized
                )
            ]
        except PermissionError:
            found = []
    return sorted({path.resolve() for path in found if path.is_file()})


def _engine_bundle_manifest(name: str):
    if name in {"generator_trt_fp32.engine", "generator_trt_fp16.engine"}:
        return "fp32_fp16_build_manifest"
    if name == "generator_trt_int8.engine":
        return "int8_build_manifest"
    return None


def _is_bundled_engine(path: Path, manifest_stem: str | None) -> bool:
    if not manifest_stem:
        return False
    # The exact name is used by the scripts; the wildcard also accepts a
    # Windows/Kaggle download renamed with " (1)" without losing the bundle.
    return any(path.parent.glob(f"{manifest_stem}*.json"))


def _nearby_build_manifest(path: Path, stem: str):
    for parent in (path.parent, *path.parents):
        matches = sorted(parent.glob(f"{stem}*.json"))
        if matches:
            return matches[0]
        if parent == parent.anchor:
            break
    return None


def validate_engine_provenance(engines):
    """Validate hashes when an engine is accompanied by its build manifest."""
    rules = {
        "FP32": ("fp32_fp16_build_manifest", "engines", "FP32"),
        "FP16": ("fp32_fp16_build_manifest", "engines", "FP16"),
        "INT8": ("int8_build_manifest", "engine", None),
    }
    records = {}
    for label, path in engines.items():
        stem, section, key = rules[label]
        manifest_path = _nearby_build_manifest(path, stem)
        record = {"path": str(path), "sha256": sha256(path), "manifest": None, "hash_match": None}
        if manifest_path:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = payload.get(section, {})
            if key is not None:
                expected = expected.get(key, {})
            expected_hash = expected.get("sha256")
            record["manifest"] = str(manifest_path)
            record["expected_sha256"] = expected_hash
            record["hash_match"] = record["sha256"] == expected_hash
            if expected_hash and not record["hash_match"]:
                raise RuntimeError(
                    f"{label} engine hash does not match its nearby build manifest: "
                    f"{path} actual={record['sha256']} expected={expected_hash}. "
                    "Use the engine produced by the same 03B/03C build or pass an explicit verified path."
                )
        records[label] = record
    return records


def locate_unique(name: str, explicit: str = "") -> Path:
    if explicit:
        supplied = Path(explicit).expanduser().resolve()
        if not supplied.is_file():
            raise FileNotFoundError(f"Explicit engine path does not exist: {supplied}")
        print(f"[engine] {name}: explicit={supplied}")
        return supplied

    grouped = []
    for root_name, root in _engine_search_roots():
        for path in _engine_candidates(root, name):
            grouped.append((root_name, path))
    # A notebook cwd can be inside /kaggle/input or /kaggle/working, so the
    # same resolved file may be discovered through two roots. Keep the first
    # (higher-priority) occurrence only for a clean provenance record.
    seen_paths = set()
    grouped = [
        (root_name, path)
        for root_name, path in grouped
        if not (path in seen_paths or seen_paths.add(path))
    ]
    if not grouped:
        raise FileNotFoundError(f"{name} not found. Attach the required Kaggle Dataset.")

    bundle_manifest = _engine_bundle_manifest(name)
    bundled = [(root_name, path) for root_name, path in grouped if _is_bundled_engine(path, bundle_manifest)]
    if bundled:
        grouped = bundled
        print(f"[engine] {name}: restricted to {bundle_manifest} bundle ({len(grouped)} path(s))")

    # Multiple paths may be the same uploaded engine copied to working.  Hash
    # the files so byte-identical copies become one logical candidate.
    by_digest = {}
    for root_name, path in grouped:
        digest = sha256(path)
        by_digest.setdefault(digest, []).append((root_name, path))

    # Prefer a single logical engine in the highest-priority root.  If the
    # input Dataset contains conflicting engines, never guess silently.
    for preferred_root in ("/kaggle/input", "/kaggle/working", "cwd"):
        root_digests = [digest for digest, items in by_digest.items() if any(root == preferred_root for root, _ in items)]
        if len(root_digests) == 1:
            digest = root_digests[0]
            candidates = by_digest[digest]
            chosen = next(path for root, path in candidates if root == preferred_root)
            copies = ", ".join(str(path) for _root, path in candidates)
            print(f"[engine] {name}: {chosen} (sha256={digest[:12]}, identical_copies={copies})")
            return chosen
        if len(root_digests) > 1:
            details = []
            for digest in root_digests:
                for root, path in by_digest[digest]:
                    details.append(f"- {path} [{root}, sha256={digest[:12]}, bytes={path.stat().st_size}]")
            raise RuntimeError(
                f"Multiple different {name} files found under {preferred_root}. "
                "Set the explicit path for this precision.\n" + "\n".join(details)
            )

    details = []
    for digest, items in sorted(by_digest.items()):
        for root_name, path in items:
            details.append(f"- {path} [{root_name}, sha256={digest[:12]}, bytes={path.stat().st_size}]")
    raise RuntimeError(
        f"Multiple different {name} files found. Set the explicit path for this precision.\n"
        + "\n".join(details)
        + "\nExample: --fp32-engine /kaggle/input/<dataset>/generator_trt_fp32.engine"
    )


def _valid_protocol_dir(path: Path) -> bool:
    return path.is_dir() and (path / "latent_eval.npy").is_file() and (path / "real_eval").is_dir()


def _protocol_dirs_under(root: Path):
    """Find extracted 03A roots, including one extra Dataset subdirectory."""
    candidates = []
    if not root.exists():
        return candidates
    if _valid_protocol_dir(root):
        candidates.append(root.resolve())
    try:
        for latent in root.rglob("latent_eval.npy"):
            parent = latent.parent
            if _valid_protocol_dir(parent):
                candidates.append(parent.resolve())
    except PermissionError:
        pass
    return sorted(set(candidates))


def locate_protocol(explicit: str = "") -> Path:
    """Locate either Task3_03A_Quantization_Protocol.zip or an extracted folder."""
    if explicit:
        supplied = Path(explicit).expanduser()
        if supplied.exists():
            if supplied.is_file() and supplied.suffix.lower() == ".zip":
                print(f"[auto-detect] 03A protocol ZIP: {supplied.resolve()}")
                return supplied.resolve()
            if supplied.is_dir():
                dirs = _protocol_dirs_under(supplied)
                if len(dirs) == 1:
                    print(f"[auto-detect] 03A protocol folder: {dirs[0]}")
                    return dirs[0]
                raise RuntimeError(
                    f"Explicit protocol directory does not contain exactly one valid 03A root: {supplied}"
                )
        raise FileNotFoundError(f"Explicit 03A protocol path does not exist: {supplied}")

    roots = [Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()]
    # Prefer attached input datasets. This avoids accidentally selecting an
    # older extracted copy left in /kaggle/working from a previous run.
    for root in roots:
        if not root.exists():
            continue
        zip_candidates = sorted({p.resolve() for p in root.rglob("Task3_03A_Quantization_Protocol.zip")})
        if len(zip_candidates) == 1:
            print(f"[auto-detect] 03A protocol ZIP: {zip_candidates[0]}")
            return zip_candidates[0]
        if len(zip_candidates) > 1:
            raise RuntimeError("Multiple 03A protocol ZIP files found. Set --protocol-path explicitly.")
        dirs = _protocol_dirs_under(root)
        if len(dirs) == 1:
            print(f"[auto-detect] 03A protocol folder: {dirs[0]}")
            return dirs[0]
        if len(dirs) > 1:
            raise RuntimeError("Multiple extracted 03A protocol folders found. Set --protocol-path explicitly.")
    raise FileNotFoundError(
        "03A protocol not found. Attach Task3_03A_Quantization_Protocol.zip or an "
        "extracted folder containing latent_eval.npy and real_eval/."
    )


def _resolve_extracted_protocol(root: Path) -> Path:
    dirs = _protocol_dirs_under(root)
    if len(dirs) != 1:
        raise FileNotFoundError(
            f"Extracted 03A protocol must contain exactly one folder with latent_eval.npy and real_eval/: {root}"
        )
    return dirs[0]


def extract_protocol(source: Path, output_dir: Path) -> Path:
    if source.is_dir():
        return _resolve_extracted_protocol(source)
    protocol_dir = output_dir / "protocol_03a"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    marker = protocol_dir / "latent_eval.npy"
    if not marker.exists():
        with zipfile.ZipFile(source) as archive:
            archive.extractall(protocol_dir)
    return _resolve_extracted_protocol(protocol_dir)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def io_names(engine, trt):
    if hasattr(engine, "num_io_tensors"):
        names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        input_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT)
        output_name = next(name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT)
        return input_name, output_name
    names = [engine.get_binding_name(i) for i in range(engine.num_bindings)]
    input_name = next(name for i, name in enumerate(names) if engine.binding_is_input(i))
    output_name = next(name for i, name in enumerate(names) if not engine.binding_is_input(i))
    return input_name, output_name


class TensorRTEngine:
    def __init__(self, path: Path, trt, device: torch.device):
        self.path = path
        self.trt = trt
        self.device = device
        runtime = trt.Runtime(get_logger(trt))
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {path}")
        self.context = self.engine.create_execution_context()
        self.input_name, self.output_name = io_names(self.engine, trt)
        self.input_np_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name)) if hasattr(self.engine, "get_tensor_dtype") else np.float32
        self.output_np_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name)) if hasattr(self.engine, "get_tensor_dtype") else np.float32
        self.input_torch_dtype = torch.float16 if self.input_np_dtype == np.float16 else torch.float32
        self.output_torch_dtype = torch.float16 if self.output_np_dtype == np.float16 else torch.float32
        self.stream = torch.cuda.Stream()
        print(f"[engine] {path.name}: input={self.input_name}/{self.input_np_dtype} output={self.output_name}/{self.output_np_dtype}")

    def infer(self, latent: np.ndarray) -> torch.Tensor:
        batch = latent.shape[0]
        host = np.ascontiguousarray(latent.astype(self.input_np_dtype, copy=False))
        input_tensor = torch.from_numpy(host).to(self.device, dtype=self.input_torch_dtype, non_blocking=False)
        output_tensor = torch.empty((batch, 3, IMAGE_SIZE, IMAGE_SIZE), device=self.device, dtype=self.output_torch_dtype)
        stream = self.stream.cuda_stream
        if hasattr(self.context, "set_input_shape"):
            ok = self.context.set_input_shape(self.input_name, tuple(input_tensor.shape))
            if ok is False:
                raise RuntimeError(f"Could not set dynamic input shape {tuple(input_tensor.shape)}")
            self.context.set_tensor_address(self.input_name, int(input_tensor.data_ptr()))
            self.context.set_tensor_address(self.output_name, int(output_tensor.data_ptr()))
            ok = self.context.execute_async_v3(stream)
        else:
            input_index = self.engine.get_binding_index(self.input_name)
            output_index = self.engine.get_binding_index(self.output_name)
            self.context.set_binding_shape(input_index, tuple(input_tensor.shape))
            bindings = [0] * self.engine.num_bindings
            bindings[input_index] = int(input_tensor.data_ptr())
            bindings[output_index] = int(output_tensor.data_ptr())
            ok = self.context.execute_async_v2(bindings, stream)
        if not ok:
            raise RuntimeError(f"TensorRT execution failed for {self.path.name}")
        self.stream.synchronize()
        return output_tensor.float().detach().cpu()


class LegacyFID:
    """The same Inception-v3 preprocessing used by prior project evaluations."""

    def __init__(self, device):
        from torchvision import models
        self.device = device
        self.net = models.inception_v3(
            weights=models.Inception_V3_Weights.DEFAULT,
            transform_input=False,
        )
        self.net.fc = nn.Identity()
        self.net.eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images01: torch.Tensor) -> np.ndarray:
        images = images01.to(self.device, non_blocking=True)
        images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        images = (images - 0.5) / 0.5
        values = self.net(images)
        if hasattr(values, "logits"):
            values = values.logits
        if isinstance(values, tuple):
            values = values[0]
        return values.detach().cpu().numpy()


class StandardFID:
    """Canonical public FID audit using the ``pytorch-fid`` implementation.

    ``pytorch-fid`` supplies the TensorFlow-compatible Inception-v3 pool3
    weights and preprocessing.  It is intentionally separate from the
    project's historical torchvision metric so the two protocols cannot be
    confused in the final table.
    """

    def __init__(self, device):
        try:
            from pytorch_fid.inception import InceptionV3
        except ImportError:
            print("[standard-fid] pytorch-fid not found; installing it ...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pytorch-fid"])
                importlib.invalidate_caches()
                from pytorch_fid.inception import InceptionV3
            except Exception as exc:
                raise RuntimeError(
                    "Canonical standard FID requires pytorch-fid and its Inception weights. "
                    "Enable Kaggle Internet or preinstall pytorch-fid before running 03D."
                ) from exc
        self.device = device
        # pytorch-fid releases differ: older releases expose the module-level
        # BLOCK_INDEX_BY_DIM mapping, while newer releases keep only the class
        # DEFAULT_BLOCK_INDEX.  The canonical 2048-d pool3 block is index 3 in
        # both APIs, so use the class constant when available and fall back to
        # the stable value instead of importing a version-specific symbol.
        pool3_block = int(getattr(InceptionV3, "DEFAULT_BLOCK_INDEX", 3))
        self.net = InceptionV3([pool3_block]).eval().to(device)
        for parameter in self.net.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def features(self, images01: torch.Tensor) -> np.ndarray:
        # pytorch-fid expects [0, 1] tensors and performs its own resize and
        # [-1, 1] normalization inside InceptionV3.
        images = images01.to(self.device, non_blocking=True)
        values = self.net(images)[0]
        return values.reshape(values.shape[0], -1).detach().cpu().numpy()


def load_real_images(real_dir: Path, count: int, batch_size: int):
    paths = sorted(real_dir.glob("*.png"))[:count]
    if len(paths) < count:
        raise RuntimeError(f"real_eval contains {len(paths)} PNG files, but {count} are required")
    for start in range(0, count, batch_size):
        batch_paths = paths[start : start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            images.append(torch.from_numpy(array).permute(2, 0, 1))
        yield torch.stack(images, dim=0), batch_paths


def stats(features: np.ndarray):
    return features.mean(axis=0), np.cov(features, rowvar=False)


def fid_from_stats(real_mu, real_cov, fake_mu, fake_cov):
    from scipy import linalg
    diff = real_mu - fake_mu
    covmean = linalg.sqrtm(real_cov.dot(fake_cov))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(real_cov + fake_cov - 2.0 * covmean))


def laplacian_values(images01: torch.Tensor) -> np.ndarray:
    values = []
    for image in images01:
        array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        gray = array.mean(axis=2).astype(np.float32)
        padded = np.pad(gray, ((1, 1), (1, 1)), mode="edge")
        lap = padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4.0 * gray
        values.append(float(lap.var()))
    return np.asarray(values, dtype=np.float64)


def edge_density(images01: torch.Tensor) -> float:
    values = []
    for image in images01:
        array = image.permute(1, 2, 0).numpy().astype(np.float32)
        gray = array.mean(axis=2)
        gx = np.diff(gray, axis=1, prepend=gray[:, :1])
        gy = np.diff(gray, axis=0, prepend=gray[:1, :])
        values.append(float((np.sqrt(gx * gx + gy * gy) > 0.12).mean()))
    return float(np.mean(values))


def haar_subbands(images01: torch.Tensor) -> torch.Tensor:
    """Return one-level orthonormal Haar LL/LH/HL/HH bands."""
    x = images01.float().cpu()
    a = x[:, :, 0::2, 0::2]
    b = x[:, :, 0::2, 1::2]
    c = x[:, :, 1::2, 0::2]
    d = x[:, :, 1::2, 1::2]
    ll = (a + b + c + d) * 0.5
    lh = (a - b + c - d) * 0.5
    hl = (a + b - c - d) * 0.5
    hh = (a - b - c + d) * 0.5
    return torch.stack((ll, lh, hl, hh), dim=2)


def frequency_error_rows(error_images: dict[str, torch.Tensor]) -> list[dict]:
    """Measure FP16/INT8 error per Haar subband against FP32."""
    baseline = haar_subbands(error_images["FP32"])
    rows = []
    labels = ("LL", "LH", "HL", "HH")
    for precision in ("FP16", "INT8"):
        current = haar_subbands(error_images[precision])
        diff = current - baseline
        for index, band in enumerate(labels):
            band_diff = diff[:, :, index]
            band_base = baseline[:, :, index]
            rows.append({
                "comparison": f"{precision}_vs_FP32",
                "subband": band,
                "mae_01": float(band_diff.abs().mean()),
                "rmse_01": float(torch.sqrt((band_diff ** 2).mean())),
                "baseline_mean_abs": float(band_base.abs().mean()),
                "error_to_baseline_ratio": float(band_diff.abs().mean() / max(float(band_base.abs().mean()), 1e-12)),
            })
    return rows


def lpips_diversity(images01: torch.Tensor, device: torch.device, pairs: int, seed: int) -> float:
    try:
        import lpips
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips"])
        import lpips
    metric = lpips.LPIPS(net="alex").eval().to(device)
    rng = np.random.default_rng(seed)
    n = len(images01)
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    equal = i == j
    j[equal] = (j[equal] + 1) % n
    values = []
    with torch.no_grad():
        for start in range(0, pairs, 64):
            ii = torch.from_numpy(i[start : start + 64]).long()
            jj = torch.from_numpy(j[start : start + 64]).long()
            a = images01[ii].to(device) * 2.0 - 1.0
            b = images01[jj].to(device) * 2.0 - 1.0
            values.append(metric(a, b).flatten().cpu())
    return float(torch.cat(values).mean())


def lpips_fake_real(fake01: torch.Tensor, real01: torch.Tensor, device: torch.device) -> float:
    """Compute fixed-index fake/real LPIPS as a supplementary distortion score."""
    try:
        import lpips
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "lpips"])
        import lpips
    count = min(len(fake01), len(real01))
    metric = lpips.LPIPS(net="alex").eval().to(device)
    values = []
    with torch.no_grad():
        for start in range(0, count, 64):
            fake = fake01[start : start + 64].to(device) * 2.0 - 1.0
            real = real01[start : start + 64].to(device) * 2.0 - 1.0
            values.append(metric(fake, real).flatten().cpu())
    return float(torch.cat(values).mean())


def save_contact_sheet(images_by_label: dict[str, torch.Tensor], path: Path, count: int = 64):
    tile = 64
    cols = 8
    rows_per_label = (count + cols - 1) // cols
    label_height = 24
    sheet = Image.new("RGB", (cols * tile, len(images_by_label) * (rows_per_label * tile + label_height)), "white")
    draw = ImageDraw.Draw(sheet)
    offset_y = 0
    for label, images in images_by_label.items():
        draw.text((4, offset_y + 4), label, fill="black")
        y0 = offset_y + label_height
        for index, image in enumerate(images[:count]):
            x = (index % cols) * tile
            y = y0 + (index // cols) * tile
            array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
            sheet.paste(Image.fromarray(array), (x, y))
        offset_y += rows_per_label * tile + label_height
    sheet.save(path)


def save_images(images01: torch.Tensor, directory: Path, prefix: str):
    directory.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images01):
        array = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(array).save(directory / f"{prefix}_{index:05d}.png")


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a Kaggle GPU accelerator.")
    device = torch.device("cuda")
    trt = ensure_tensorrt()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    engines = {
        "FP32": locate_unique("generator_trt_fp32.engine", args.fp32_engine),
        "FP16": locate_unique("generator_trt_fp16.engine", args.fp16_engine),
        "INT8": locate_unique("generator_trt_int8.engine", args.int8_engine),
    }
    engine_provenance = validate_engine_provenance(engines)
    protocol_source = locate_protocol(args.protocol_path)
    protocol_dir = extract_protocol(protocol_source, output_dir)
    latent_eval = np.load(protocol_dir / "latent_eval.npy").astype(np.float32, copy=False)
    if latent_eval.ndim != 4 or tuple(latent_eval.shape[1:]) != (NOISE_DIM, 1, 1):
        raise ValueError(f"Unexpected latent_eval shape: {latent_eval.shape}")
    n_fid = min(args.n_fid, len(latent_eval))
    if n_fid < args.n_fid:
        print(f"[warn] requested n_fid={args.n_fid}, using available {n_fid}")
    real_dir = protocol_dir / "real_eval"
    real_batches = list(load_real_images(real_dir, n_fid, args.batch_size))
    fid_net = LegacyFID(device)
    real_feature_parts = [fid_net.features(images) for images, _paths in real_batches]
    real_features = np.concatenate(real_feature_parts, axis=0)[:n_fid]
    real_mu, real_cov = stats(real_features)
    np.savez(output_dir / "real_fid_legacy_stats.npz", mean=real_mu, covariance=real_cov, count=n_fid)
    standard_fid_net = StandardFID(device)
    standard_real_feature_parts = [standard_fid_net.features(images) for images, _paths in real_batches]
    standard_real_features = np.concatenate(standard_real_feature_parts, axis=0)[:n_fid]
    standard_real_mu, standard_real_cov = stats(standard_real_features)
    np.savez(output_dir / "real_fid_standard_stats.npz", mean=standard_real_mu, covariance=standard_real_cov, count=n_fid)
    if standard_real_features.shape[1] != 2048:
        raise RuntimeError(f"Standard FID expected 2048-dim Inception-v3 features, got {standard_real_features.shape[1]}")
    real_metric = torch.cat([images for images, _paths in real_batches], dim=0)[: args.n_image_eval]
    real_lap = laplacian_values(real_metric)
    blur_threshold = float(np.percentile(real_lap, 10))

    all_metric_images = {}
    feature_rows = []
    metric_rows = []
    error_images = {}
    pair_seed = args.seed + 700001
    for label, engine_path in engines.items():
        engine = TensorRTEngine(engine_path, trt, device)
        for _ in range(args.warmup_batches):
            engine.infer(latent_eval[: min(args.batch_size, n_fid)])
        fake_feature_parts = []
        standard_fake_feature_parts = []
        metric_parts = []
        sample_parts = []
        timings = []
        for start in range(0, n_fid, args.batch_size):
            batch = latent_eval[start : start + args.batch_size]
            t0 = time.perf_counter()
            fake = engine.infer(batch)
            timings.append((time.perf_counter() - t0) * 1000.0)
            fake01 = ((fake + 1.0) * 0.5).clamp(0.0, 1.0)
            fake_feature_parts.append(fid_net.features(fake01))
            standard_fake_feature_parts.append(standard_fid_net.features(fake01))
            if start < args.n_image_eval:
                keep = min(len(fake01), args.n_image_eval - start)
                metric_parts.append(fake01[:keep])
            if start < 64:
                sample_parts.append(fake01[: min(len(fake01), 64 - start)])
            if args.save_all_fakes:
                save_images(fake01, output_dir / f"fake_{label.lower()}", label.lower())
        fake_features = np.concatenate(fake_feature_parts, axis=0)[:n_fid]
        fake_mu, fake_cov = stats(fake_features)
        standard_fake_features = np.concatenate(standard_fake_feature_parts, axis=0)[:n_fid]
        standard_fake_mu, standard_fake_cov = stats(standard_fake_features)
        fake_metric = torch.cat(metric_parts, dim=0)[: args.n_image_eval]
        fake_sample = torch.cat(sample_parts, dim=0)[:64]
        all_metric_images[label] = fake_sample
        error_images[label] = fake_metric
        fake_lap = laplacian_values(fake_metric)
        mean_ms = float(np.mean(timings))
        metric_rows.append({
            "precision": label,
            "fid_legacy_inception_v3": fid_from_stats(real_mu, real_cov, fake_mu, fake_cov),
            "fid_standard_inception_v3": fid_from_stats(standard_real_mu, standard_real_cov, standard_fake_mu, standard_fake_cov),
            "blur_threshold_real_p10": blur_threshold,
            "fake_blur_rate": float((fake_lap < blur_threshold).mean()),
            "real_blur_rate_by_definition": float((real_lap < blur_threshold).mean()),
            "fake_laplacian_mean": float(fake_lap.mean()),
            "real_laplacian_mean": float(real_lap.mean()),
            "fake_edge_density": edge_density(fake_metric),
            "real_edge_density": edge_density(real_metric),
            "edge_density_ratio": edge_density(fake_metric) / max(edge_density(real_metric), 1e-12),
            "lpips_alex_diversity": lpips_diversity(fake_metric, device, args.lpips_pairs, pair_seed),
            "lpips_alex_fake_real": lpips_fake_real(fake_metric, real_metric, device),
            "inference_mean_ms_per_batch": mean_ms,
            "inference_images_per_s": float(args.batch_size / (mean_ms / 1000.0)),
            "status": "ok",
        })
        feature_rows.append({
            "precision": label,
            "count": n_fid,
            "legacy_feature_dim": int(fake_features.shape[1]),
            "standard_feature_dim": int(standard_fake_features.shape[1]),
        })
        print(
            f"[ok] {label}: legacy_fid={metric_rows[-1]['fid_legacy_inception_v3']:.4f} "
            f"standard_fid={metric_rows[-1]['fid_standard_inception_v3']:.4f} "
            f"blur={metric_rows[-1]['fake_blur_rate']:.4f}"
        )

    write_csv(output_dir / "fp32_fp16_int8_metrics.csv", list(metric_rows[0].keys()), metric_rows)
    write_csv(output_dir / "feature_manifest.csv", list(feature_rows[0].keys()), feature_rows)
    baseline = error_images["FP32"]
    error_rows = []
    for label in ("FP16", "INT8"):
        diff = error_images[label] - baseline
        error_rows.append({"comparison": f"{label}_vs_FP32", "count": len(diff), "mae_01": float(np.abs(diff.numpy()).mean()), "rmse_01": float(np.sqrt((diff.numpy() ** 2).mean())), "max_abs_01": float(np.abs(diff.numpy()).max())})
    write_csv(output_dir / "quantization_error.csv", list(error_rows[0].keys()), error_rows)
    frequency_rows = frequency_error_rows(error_images)
    write_csv(output_dir / "frequency_band_error.csv", list(frequency_rows[0].keys()), frequency_rows)
    save_contact_sheet(all_metric_images, output_dir / "quality_contact_sheet.png", count=64)
    manifest = {
        "task": "Task3_03D_Quality_Evaluation",
        "protocol_source": str(protocol_source),
        "protocol_source_type": "zip" if protocol_source.is_file() else "extracted_directory",
        "protocol_source_sha256": sha256(protocol_source) if protocol_source.is_file() else None,
        "engines": {
            label: {**engine_provenance[label], "bytes": path.stat().st_size}
            for label, path in engines.items()
        },
        "latent_eval_count": n_fid,
        "latent_shape": list(latent_eval.shape),
        "real_eval_count": n_fid,
        "metric_image_count": args.n_image_eval,
        "lpips_pairs": args.lpips_pairs,
        "batch_size": args.batch_size,
        "warmup_batches": args.warmup_batches,
        "seed": args.seed,
        "fid_protocol": "project legacy torchvision Inception-v3: resize 299, normalize (x-0.5)/0.5, pool3/FC identity features",
        "standard_fid_protocol": "pytorch-fid Inception-v3 pool3 block: input [0,1], library TensorFlow-compatible resize/normalization, 2048 dimensions",
        "blur_protocol": "real_eval Laplacian p10 threshold; same threshold applied to FP32/FP16/INT8",
        "lpips_protocol": "LPIPS-Alex pairwise diversity on fixed pairs plus fixed-index fake-real supplementary distortion",
        "standard_fid": "computed for FP32/FP16/INT8; see fid_standard_inception_v3 in fp32_fp16_int8_metrics.csv",
        "standard_fid_computed": True,
        "standard_fid_feature_dim": 2048,
        "standard_fid_implementation": "pytorch-fid InceptionV3 pool3 block",
        "standard_fid_stats_file": "real_fid_standard_stats.npz",
        "frequency_error_protocol": "one-level orthonormal Haar LL/LH/HL/HH error on the same fixed metric images against FP32",
        "frequency_band_error_file": "frequency_band_error.csv",
        "note": "All three engines receive identical latent_eval arrays; FP32/FP16/INT8 differences are attributed to inference precision.",
    }
    (output_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    archive = output_dir.parent / "Task3_03D_Evaluation.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        # Include optional fake_* directories when --save-all-fakes is used.
        # The previous top-level-only loop silently omitted those PNGs.
        for path in output_dir.rglob("*"):
            if path.is_file() and path.resolve() != archive.resolve():
                z.write(path, arcname=str(path.relative_to(output_dir)))
    print(f"[zip] {archive}")
    print(f"[done] {output_dir}")


if __name__ == "__main__":
    main()
