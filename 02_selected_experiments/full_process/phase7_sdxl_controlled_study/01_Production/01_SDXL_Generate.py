"""
SDXL ?????? ? Colab ???????????

? ?????? SDXL ??????????
   ?????? 1600 ?? Kaggle ??????, ???? 02_Cleaning?

????:
  1. colab.research.google.com ? New Notebook ? T4 GPU
  2. ?????????? cell ? Ctrl+Enter
  3. ????? HuggingFace ?? (~6GB), ??????

?? (??):
  ? all_candidates_cumulative.zip ?? Colab ??????? ? /content/
  ????????????????

????:
  Colab ??????? ? ?? all_candidates_cumulative.zip ? Download

??:
  all_candidates_cumulative.zip ? ?? 256px ???64px ????1024px ???contact sheets
"""

from __future__ import annotations
import csv, hashlib, json, os, shutil, subprocess, sys, time, zipfile, math
import importlib.util
from pathlib import Path


# ???????????????????????????????????????????????????????????????????
# 0. ?? + CUDA
# ???????????????????????????????????????????????????????????????????
def _install() -> None:
    pkgs = [
        ("diffusers>=0.30.0", "diffusers"),
        ("transformers>=4.44.0", "transformers"),
        ("accelerate>=0.33.0", "accelerate"),
        ("safetensors", "safetensors"),
        ("Pillow>=10.0.0", "PIL"),
    ]
    missing = [p for p, m in pkgs if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

# ?? ?????????? PyTorch ????????? OOM
import os as _os
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_install()

import numpy as np
import torch
import gc


def _ensure_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available. Enable GPU: Runtime ? Change runtime type ? T4 GPU"
        )
    # ??????????????? ?? Colab ?? GPU ???????????
    # ??????????diffusers ???? TORCH_LIBRARY ???????
    try:
        a = torch.randn(2000, 2000, device="cuda")
        b = torch.randn(2000, 2000, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = a @ b
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        if ms > 500:
            print(f"[cuda] ? GPU matmul slow ({ms:.0f}ms) ? may be shared GPU or first-run JIT. Continuing anyway.")
    except Exception as e:
        raise RuntimeError(f"CUDA test failed: {e}")

_ensure_cuda()
print(f"CUDA OK ? {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")


# ???????????????????????????????????????????????????????????????????
# 1. ??
# ???????????????????????????????????????????????????????????????????
MODEL_ID        = "cagliostrolab/animagine-xl-4.0"
SEED_BASE       = 20260806
TARGET_ACCEPTED = 5000
EXPECTED_PASS   = 0.70
N_CANDIDATES    = int(math.ceil(TARGET_ACCEPTED / max(EXPECTED_PASS, 0.05) * 1.05))

# ?? ??????: ???? Kaggle ???????????????
# Kaggle 1600 ? = ?? 1600????? Kaggle ??????????????
# 0 ????? resume zip ??; ??????????? SDXL ????
FORCE_START_INDEX = int(os.environ.get("SDXL_FORCE_START", "1600"))

WIDTH, HEIGHT   = 1024, 1024
STEPS, CFG      = 20, 7.0
BATCH_SIZE      = 2

ZIP_EVERY       = 500
MAX_PER_SESSION = 2500
BLUR_THRESHOLD  = 80

OUTPUT_ROOT = Path("/content/SDXL_Production_v3")
OUT         = OUTPUT_ROOT / "current_batch"
PROMPT_VERSION = "production_v3_colab"

# ??: ? all_candidates_cumulative.zip ?? Colab ?????????
RESUME_ZIP = Path("/content/all_candidates_cumulative.zip")
if RESUME_ZIP.exists():
    print(f"[setup] Resume zip detected: {RESUME_ZIP.stat().st_size/1e9:.1f}GB")
    print(f"[setup] Will resume from checkpoint.")
else:
    print(f"[setup] No resume zip ? starting fresh (0?{N_CANDIDATES})")
    print(f"[setup] Tip: drag all_candidates_cumulative.zip into Colab file browser to resume.")

from PIL import Image, ImageDraw, ImageFont
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


# ???????????????????????????????????????????????????????????????????
# 2. ????? Kaggle ??? v3 ?????
# ???????????????????????????????????????????????????????????????????
QUALITY = "masterpiece, best quality, newest"

COMPOSITION = (
    "1girl, solo, head focus, portrait, upper body, "
    "simple background, plain background, light background, "
    "anime coloring, cel shading, flat color"
)

NEGATIVE = (
    "lowres, worst quality, low quality, normal quality, "
    "bad anatomy, bad face, deformed, malformed, mutated, disfigured, "
    "3d, realistic, photorealistic, western artstyle, "
    "multiple girls, 2girls, multiple views, "
    "text, watermark, signature, username, logo, "
    "blurry, jpeg artifacts, motion blur, depth of field, "
    "cropped, out of frame, partial face, "
    "extra eyes, extra ears, extra limbs, "
    "chibi, sketch, unfinished, rough"
)

TARGETS = [
    ("profile", "purple hair",
     "from side, profile, looking to the side, one eye visible, ear visible, "
     "nose profile, chin line, sideburns"),
    ("profile", "silver hair",
     "from side, profile, looking to the side, one eye visible, ear visible, "
     "nose profile, chin line, sideburns"),
    ("profile", "blonde hair",
     "from side, profile, looking to the side, one eye visible, ear visible, "
     "nose profile, chin line, sideburns"),
    ("profile", "blue hair",
     "from side, profile, looking to the side, one eye visible, ear visible, "
     "nose profile, chin line, sideburns"),
    ("three_quarter", "purple hair",
     "from side, head turn, turned head, both eyes, looking at viewer, "
     "nose visible, mouth visible"),
    ("three_quarter", "blonde hair",
     "from side, head turn, turned head, both eyes, looking at viewer, "
     "nose visible, mouth visible"),
    ("three_quarter", "blue hair",
     "from side, head turn, turned head, both eyes, looking at viewer, "
     "nose visible, mouth visible"),
    ("three_quarter", "silver hair",
     "from side, head turn, turned head, both eyes, looking at viewer, "
     "nose visible, mouth visible"),
    ("front", "black hair",
     "from front, facing viewer, looking at viewer, "
     "both eyes, symmetrical face, nose, mouth"),
    ("front", "purple hair",
     "from front, facing viewer, looking at viewer, "
     "both eyes, symmetrical face, nose, mouth"),
    ("front", "blonde hair",
     "from front, facing viewer, looking at viewer, "
     "both eyes, symmetrical face, nose, mouth"),
    ("profile", "purple hair",
     "from side, profile, looking to the side, one eye visible, ear visible, "
     "nose profile, chin line, glasses, black framed glasses, eyewear"),
    ("three_quarter", "silver hair",
     "from side, head turn, turned head, both eyes, looking at viewer, "
     "nose visible, mouth visible, glasses, black framed glasses, eyewear"),
    ("profile", "blonde hair",
     "from side, profile, looking to the side, closed eyes, eyelashes, "
     "serene expression, calm, one eye closed, ear visible, "
     "nose profile, chin line"),
    ("three_quarter", "purple hair",
     "from side, head turn, turned head, both eyes, looking at viewer, "
     "open mouth, teeth, talking, nose visible"),
    ("three_quarter", "silver hair",
     "looking up, from below, head tilt, "
     "both eyes, looking at viewer, nose visible, mouth visible"),
]


# ???????????????????????????????????????????????????????????????????
# 3. ????
# ???????????????????????????????????????????????????????????????????
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def average_hash(img: Image.Image, size: int = 16) -> str:
    a = np.asarray(img.convert("L").resize((size, size), Image.Resampling.LANCZOS), dtype=np.float32)
    return "".join("1" if b else "0" for b in (a > a.mean()).ravel())


def laplacian_variance(img_rgb: Image.Image, size: int = 256) -> float:
    gray = img_rgb.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    a = np.asarray(gray, dtype=np.float32)
    p = np.pad(a, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * p[1:-1, 1:-1]
    return float(lap.var())


def ensure_dirs() -> None:
    for d in ["previews_256", "images_64", "masters_1024", "contact_sheets"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)


def load_pipe():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[pipe] Downloading {MODEL_ID} (~6GB, one-time)...")
    t0 = time.time()
    kw = {"torch_dtype": torch.float16, "use_safetensors": True}
    try:
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **kw, variant="fp16")
    except (OSError, ValueError):
        pipe = DiffusionPipeline.from_pretrained(MODEL_ID, **kw)
    pipe.to(dev)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_vae_tiling()
        pipe.enable_vae_slicing()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    print(f"[pipe] Loaded in {time.time() - t0:.0f}s | DPM++ 2M Karras | VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return pipe, dev


def generate_batch(pipe, batch: list[dict], dev: str) -> list[Image.Image]:
    prompts = [j["prompt"] for j in batch]
    negatives = [j["negative"] for j in batch]
    generators = [torch.Generator(device=dev).manual_seed(j["seed"]) for j in batch]
    try:
        with torch.inference_mode():
            return pipe(prompt=prompts, negative_prompt=negatives,
                        width=WIDTH, height=HEIGHT, num_inference_steps=STEPS,
                        guidance_scale=CFG, generator=generators).images
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(batch) == 1:
            raise
        print(f"  [oom] batch={len(batch)} ? single-image fallback")
        torch.cuda.empty_cache()
        imgs = []
        for j in batch:
            g = torch.Generator(device=dev).manual_seed(j["seed"])
            with torch.inference_mode():
                imgs.extend(pipe(prompt=j["prompt"], negative_prompt=j["negative"],
                                 width=WIDTH, height=HEIGHT, num_inference_steps=STEPS,
                                 guidance_scale=CFG, generator=g).images)
        return imgs


def load_existing_rows() -> list[dict]:
    mf = OUT / "candidate_manifest.csv"
    if not mf.exists():
        return []
    with mf.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if (OUT / r.get("image_64_path", "")).is_file()]


def write_manifest(rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for e in ["manual_keep", "reject_reason"]:
        if e not in fields:
            fields.append(e)
    with (OUT / "candidate_manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_checkpoint(rows: list[dict], status: str, start_idx: int) -> None:
    (OUT / "checkpoint_state.json").write_text(json.dumps({
        "status": status, "prompt_version": PROMPT_VERSION,
        "session_completed": len(rows), "global_start_idx": start_idx,
        "total_target": N_CANDIDATES, "model_id": MODEL_ID,
        "width": WIDTH, "height": HEIGHT, "steps": STEPS, "cfg": CFG,
        "blur_threshold": BLUR_THRESHOLD,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def make_cumulative_zip() -> Path:
    zp = OUTPUT_ROOT / "all_candidates_cumulative.zip"
    tmp = zp.with_suffix(".zip.tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for d in sorted(OUTPUT_ROOT.glob("batch_*")):
            for f in d.rglob("*"):
                if f.is_file():
                    z.write(f, f.relative_to(OUTPUT_ROOT))
        for f in OUT.rglob("*"):
            if f.is_file():
                z.write(f, f.relative_to(OUTPUT_ROOT))
    tmp.replace(zp)
    return zp


def determine_start_idx() -> int:
    if FORCE_START_INDEX > 0:
        print(f"[setup] FORCE_START_INDEX={FORCE_START_INDEX} ? ??????, ??? {FORCE_START_INDEX} ??")
        print(f"[setup] sample_id ? S{FORCE_START_INDEX+1:05d} ??, seed ? {SEED_BASE + FORCE_START_INDEX} ??")
        return FORCE_START_INDEX
    if RESUME_ZIP.exists():
        return _restore_from(RESUME_ZIP)
    existing = sorted(OUTPUT_ROOT.glob("batch_*"))
    if existing:
        max_idx = 0
        for d in existing:
            mf = d / "candidate_manifest.csv"
            if mf.exists():
                with mf.open(newline="", encoding="utf-8-sig") as f:
                    max_idx = max(max_idx, sum(1 for _ in csv.DictReader(f)))
            cp = d / "checkpoint_state.json"
            if cp.exists():
                try:
                    s = json.loads(cp.read_text(encoding="utf-8"))
                    max_idx = max(max_idx, s.get("global_start_idx", 0) + s.get("session_completed", 0))
                except Exception:
                    pass
        if max_idx > 0:
            print(f"[resume] Found {len(existing)} batches, next_idx={max_idx}")
            return max_idx
    return 0


def _restore_from(zip_path: Path) -> int:
    print(f"[resume] Extracting {zip_path.name} ...")
    root = OUTPUT_ROOT.resolve()
    with zipfile.ZipFile(zip_path, "r") as z:
        for info in z.infolist():
            target = (OUTPUT_ROOT / info.filename).resolve()
            try:
                target.relative_to(root)
            except ValueError:
                continue
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(info) as src, target.open("wb") as dst:
                    dst.write(src.read())
    total = 0
    for d in sorted(OUTPUT_ROOT.glob("batch_*")):
        mf = d / "candidate_manifest.csv"
        if mf.exists():
            with mf.open(newline="", encoding="utf-8-sig") as f:
                total += sum(1 for _ in csv.DictReader(f))
    print(f"[resume] Restored {total} existing candidates ? next starts at {total}")
    return total


def make_contact_sheets(rows: list[dict]) -> None:
    tile, cols, page_size = 160, 10, 100
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except OSError:
        font = ImageFont.load_default()
    for ps in range(0, len(rows), page_size):
        page = rows[ps:ps + page_size]
        sheet = Image.new("RGB", (cols * tile, 10 * (tile + 25)), "white")
        draw = ImageDraw.Draw(sheet)
        for j, r in enumerate(page):
            im = Image.open(OUT / r["preview_path"]).convert("RGB")
            im.thumbnail((tile - 4, tile - 4))
            x = (j % cols) * tile; y = (j // cols) * (tile + 25)
            sheet.paste(im, (x + (tile - im.width) // 2, y))
            label = f"{r['sample_id']} {r['target_pose'][:4]}-{r['target_hair'].replace(' hair','')[:6]}"
            if r.get("auto_blur") == "1":
                label += " [B]"
            draw.text((x + 2, y + tile + 2), label, fill="black", font=font)
        sheet.save(OUT / "contact_sheets" / f"sheet_{ps // page_size:03d}.jpg", quality=92)


def build_schedule(start_idx: int, limit: int) -> list[dict]:
    jobs = []
    end_idx = min(start_idx + limit, N_CANDIDATES) if limit > 0 else N_CANDIDATES
    for global_idx in range(start_idx, end_idx):
        ti = global_idx % len(TARGETS)
        pose, hair, pose_tags = TARGETS[ti]
        seed = SEED_BASE + global_idx
        sid = f"S{global_idx + 1:05d}"
        prompt = f"{QUALITY}, {pose_tags}, {hair}, {COMPOSITION}"
        neg = NEGATIVE
        if pose == "profile":
            neg += ", from front, facing viewer, both eyes, looking at viewer, symmetrical face"
        elif pose == "three_quarter":
            neg += ", from front, exactly facing viewer, straight on, symmetrical face, profile"
        elif pose == "front":
            neg += ", from side, profile, one eye, asymmetrical face, looking away, looking to the side"
        jobs.append({
            "sample_id": sid, "seed": seed, "prompt": prompt, "negative": neg,
            "target_pose": pose, "target_hair": hair,
        })
    return jobs


# ???????????????????????????????????????????????????????????????????
# 4. ???
# ???????????????????????????????????????????????????????????????????
def main() -> None:
    t_total_start = time.perf_counter()

    start_idx = determine_start_idx()
    if start_idx >= N_CANDIDATES:
        print(f"[done] All {N_CANDIDATES} done. Download from Colab file browser ? all_candidates_cumulative.zip")
        return

    session_limit = min(MAX_PER_SESSION, N_CANDIDATES - start_idx) if MAX_PER_SESSION > 0 else (N_CANDIDATES - start_idx)
    global OUT
    session_end = min(start_idx + session_limit, N_CANDIDATES)
    OUT = OUTPUT_ROOT / f"batch_{start_idx:05d}_{session_end:05d}"
    ensure_dirs()

    torch.manual_seed(SEED_BASE)
    np.random.seed(SEED_BASE)

    rows = load_existing_rows()
    existing_ids = {r["sample_id"] for r in rows}
    hash_groups: dict[str, list[str]] = {}
    blur_count = sum(1 for r in rows if r.get("auto_blur") == "1")

    if len(rows) >= session_limit:
        print(f"[done] Batch already complete ({len(rows)} entries).")
        pipe = None; dev = "cuda"
    else:
        pipe, dev = load_pipe()
        print("[warmup] Compiling CUDA kernels...")
        _wg = torch.Generator(device=dev).manual_seed(42)
        with torch.inference_mode():
            _ = pipe(prompt="1girl, head focus", negative_prompt=NEGATIVE,
                     width=WIDTH, height=HEIGHT, num_inference_steps=STEPS,
                     guidance_scale=CFG, generator=_wg).images[0]
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        print("[warmup] Done\n")

    schedule = build_schedule(start_idx, session_limit)
    schedule = [j for j in schedule if j["sample_id"] not in existing_ids]
    remaining = len(schedule)
    est_h = remaining * 10 / 3600

    print(f"{'='*60}")
    print(f"  SDXL Colab ? {start_idx}?{session_end} ({remaining} new / {N_CANDIDATES} total)")
    print(f"  {WIDTH}?{HEIGHT} | {STEPS} steps DPM++2M | CFG={CFG} | batch={BATCH_SIZE}")
    print(f"  Est: {est_h:.0f}h | Save every {ZIP_EVERY} | Blur < {BLUR_THRESHOLD}")
    print(f"  ?????????????????????????????????????????????")
    print(f"  ?? ??: Colab ??????? ? ?? zip ? Download")
    print(f"{'='*60}\n")

    device_name = dev if dev != "cpu" else "cpu"
    last_zip_count = len(rows)

    for batch_start in range(0, len(schedule), BATCH_SIZE):
        batch = schedule[batch_start:batch_start + BATCH_SIZE]
        if not batch:
            continue

        n_done = len(rows)
        pct = (start_idx + n_done) * 100 // N_CANDIDATES
        print(f"[gen] {start_idx + n_done + 1}-{start_idx + n_done + len(batch)}/"
              f"{N_CANDIDATES} ({pct}%)", end=" ")

        t_infer = time.perf_counter()
        images = generate_batch(pipe, batch, device_name)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        infer_sec = time.perf_counter() - t_infer

        # ? 10 ? batch ?????????
        if (batch_start // BATCH_SIZE) % 10 == 0 and batch_start > 0:
            torch.cuda.empty_cache()

        batch_blur = 0
        for j, im in zip(batch, images):
            im = im.convert("RGB")
            sid = j["sample_id"]
            im256 = im.resize((256, 256), Image.Resampling.LANCZOS)
            lap_var = laplacian_variance(im)
            is_blur = "1" if lap_var < BLUR_THRESHOLD else "0"
            if is_blur == "1":
                batch_blur += 1

            preview_rel = f"previews_256/{sid}.jpg"
            img64_rel = f"images_64/{sid}.png"
            master_rel = f"masters_1024/{sid}.jpg"

            im256.save(OUT / preview_rel, quality=92)
            im256.resize((64, 64), Image.Resampling.LANCZOS).save(OUT / img64_rel, optimize=True)
            im.save(OUT / master_rel, quality=95)
            ah = average_hash(Image.open(OUT / img64_rel).convert("RGB"))
            hash_groups.setdefault(ah, []).append(sid)

            rows.append({
                "sample_id": sid, "target_pose": j["target_pose"], "target_hair": j["target_hair"],
                "seed": j["seed"], "prompt": j["prompt"], "negative_prompt": j["negative"],
                "model_id": MODEL_ID, "preview_path": preview_rel,
                "image_64_path": img64_rel, "master_path": master_rel,
                "sha256_64": sha256_file(OUT / img64_rel), "average_hash": ah,
                "lapvar_256": f"{lap_var:.1f}", "auto_blur": is_blur,
            })

        blur_count += batch_blur
        print(f"infer={infer_sec:.1f}s ({infer_sec/len(batch):.1f}s/img) blur={batch_blur}/{len(batch)}")

        write_manifest(rows)
        trigger_early = (last_zip_count == 0 and len(rows) >= 100)
        trigger_normal = (len(rows) - last_zip_count >= ZIP_EVERY)
        if trigger_early or trigger_normal:
            last_zip_count = len(rows)
            write_checkpoint(rows, "running", start_idx)
            cumulative = make_cumulative_zip()
            tag = "[early]" if trigger_early else ""
            print(f"  [save{tag}] @{start_idx+len(rows)}/{N_CANDIDATES} ? "
                  f"all_candidates_cumulative.zip ({cumulative.stat().st_size/1e6:.0f}MB)")

    # ?? Finalize ??
    write_manifest(rows)
    write_checkpoint(rows, "completed", start_idx)

    blur_rows = [r for r in rows if r.get("auto_blur") == "1"]
    if blur_rows:
        with (OUT / "blur_report.csv").open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["sample_id", "target_pose", "target_hair", "lapvar_256"])
            w.writeheader()
            for r in blur_rows:
                w.writerow({k: r[k] for k in ["sample_id", "target_pose", "target_hair", "lapvar_256"]})

    make_contact_sheets(rows)
    total_sec = time.perf_counter() - t_total_start
    session_pct = (start_idx + len(rows)) * 100 // N_CANDIDATES

    (OUT / "production_config.json").write_text(json.dumps({
        "prompt_version": PROMPT_VERSION, "model_id": MODEL_ID,
        "session_range": f"{start_idx}-{start_idx + len(rows)}",
        "session_completed": len(rows), "global_progress_pct": session_pct,
        "total_target": N_CANDIDATES, "auto_blur_count": blur_count,
        "auto_blur_rate": round(blur_count / max(len(rows), 1), 3),
        "batch_size": BATCH_SIZE, "width": WIDTH, "height": HEIGHT,
        "steps": STEPS, "cfg": CFG, "blur_threshold": BLUR_THRESHOLD,
        "runtime_h": round(total_sec / 3600, 1),
        "seconds_per_image": round(total_sec / max(len(rows), 1), 2),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    cumulative_zip = make_cumulative_zip()

    print(f"\n{'='*60}")
    print(f"  DONE: {len(rows)} images | {total_sec/60:.0f}min | {total_sec/max(len(rows),1):.1f}s/img")
    print(f"  Global: {start_idx + len(rows)}/{N_CANDIDATES} ({session_pct}%)")
    print(f"  Blur: {blur_count}/{len(rows)} ({blur_count*100/max(len(rows),1):.1f}%)")
    print(f"  ?????????????????????????????????????????????")
    print(f"  ?? ????:")
    print(f"     Colab ?? ? ?? ?????")
    print(f"     ? ?? all_candidates_cumulative.zip")
    print(f"     ? Download")
    print(f"     ({cumulative_zip.stat().st_size/1e6:.0f}MB)")
    print(f"{'='*60}")

    if start_idx + len(rows) < N_CANDIDATES:
        next_idx = start_idx + len(rows)
        print(f"\n  ?? Colab ??:")
        print(f"  1. ?? all_candidates_cumulative.zip ?????")
        print(f"  2. ? Colab ? ? zip ??????")
        print(f"  3. ????? ? ??? {next_idx} ??")
    else:
        print(f"\n  ?? ALL {N_CANDIDATES} DONE!")
        print(f"  ?? zip ? ?? 03_Cleaning ?? ? ????")


if __name__ == "__main__":
    main()
