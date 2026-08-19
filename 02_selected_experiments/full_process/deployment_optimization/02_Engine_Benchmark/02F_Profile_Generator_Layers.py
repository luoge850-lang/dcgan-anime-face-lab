"""
Task 2F - map PyTorch profiler time to concrete Generator ``net.*`` layers.

The existing 02D trace identifies operator classes. This script adds a
record_function scope around every Sequential layer, so the Top-3 result can
name net.0/net.3/... instead of only saying "ConvTranspose".
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path


NOISE_DIM = 128


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Profile concrete DCGAN Generator layers")
    parser.add_argument("--g-path", default=os.getenv("G_PATH", ""))
    parser.add_argument(
        "--output-dir",
        default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark/02F_Layer_Profile"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args, _unknown = parser.parse_known_args(argv)
    if min(args.batch_size, args.warmup, args.steps) <= 0:
        raise ValueError("batch-size, warmup and steps must be positive")
    return args


def locate_weight(explicit: str) -> Path:
    if explicit and Path(explicit).exists():
        return Path(explicit).resolve()
    found = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            found.extend(root.rglob("generator_ema_final.pth"))
    unique = sorted({p.resolve() for p in found})
    if len(unique) == 1:
        print(f"[weights] {unique[0]}")
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError("Multiple generator_ema_final.pth files found; set --g-path explicitly.")
    raise FileNotFoundError("generator_ema_final.pth not found.")


def make_generator(torch):
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(128, 768, 4, 1, 0, bias=True), nn.BatchNorm2d(768), nn.ReLU(True),
                nn.ConvTranspose2d(768, 384, 4, 2, 1, bias=True), nn.BatchNorm2d(384), nn.ReLU(True),
                nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=True), nn.BatchNorm2d(192), nn.ReLU(True),
                nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=True), nn.BatchNorm2d(96), nn.ReLU(True),
                nn.ConvTranspose2d(96, 3, 4, 2, 1, bias=True), nn.Tanh(),
            )
        def forward_profiled(self, z, record_function):
            x = z
            for index, layer in enumerate(self.net):
                with record_function(f"Generator.net.{index}.{layer.__class__.__name__}"):
                    x = layer(x)
            return x
    return Generator()


def load_state(torch, generator, path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    generator.load_state_dict(state, strict=True)


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    try:
        import torch
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
        import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = locate_weight(args.g_path)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(torch).to(device).eval()
    load_state(torch, generator, weight_path, device)
    z = torch.randn(args.batch_size, NOISE_DIM, 1, 1, device=device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with torch.inference_mode():
        for _ in range(args.warmup):
            generator.forward_profiled(z, record_function)
    if device.type == "cuda":
        torch.cuda.synchronize()
    trace_path = output_dir / "layer_profiler_trace.json"
    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=False) as prof:
        for _ in range(args.steps):
            generator.forward_profiled(z, record_function)
            prof.step()
    prof.export_chrome_trace(str(trace_path))
    rows = []
    for item in prof.key_averages():
        if not item.key.startswith("Generator.net."):
            continue
        cpu_ms = float(item.self_cpu_time_total / 1000.0)
        device_ms = float(getattr(item, "self_device_time_total", 0.0) / 1000.0)
        rows.append([item.key, item.count, cpu_ms, device_ms, int(getattr(item, "self_cpu_memory_usage", 0)), int(getattr(item, "self_device_memory_usage", 0))])
    rows.sort(key=lambda row: max(row[2], row[3]), reverse=True)
    write_csv(output_dir / "layer_operator_summary.csv", ["layer", "calls", "self_cpu_ms", "self_device_ms", "self_cpu_memory_bytes", "self_device_memory_bytes"], rows)
    archive = output_dir.parent / "Task2_02F_Layer_Profile.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in output_dir.rglob("*"):
            if path.is_file():
                zf.write(path, arcname=f"02F_Layer_Profile/{path.relative_to(output_dir).as_posix()}")
    print(f"[done] layer rows: {len(rows)}")
    print(f"[zip] {archive}")


if __name__ == "__main__":
    main()
