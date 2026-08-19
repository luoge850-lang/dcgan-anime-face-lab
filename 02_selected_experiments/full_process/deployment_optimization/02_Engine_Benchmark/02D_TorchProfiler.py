"""
Task 2D — PyTorch reference profiling for Chrome Trace.

This is separate from ORT/TensorRT/OpenVINO: torch.profiler profiles the
original PyTorch generator and produces the required Chrome Trace. It uses
the same Exp11 generator architecture and the EMA state_dict.
"""

import argparse
import csv
import os
import subprocess
import sys
import zipfile
from pathlib import Path


NOISE_DIM = 128


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 2D torch.profiler trace")
    p.add_argument("--g-path", default=os.getenv("G_PATH", ""))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/kaggle/working/dcgan_output/Deployment_Optimization_Results/02_Engine_Benchmark/02D_Torch_Reference"))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=20)
    args, _ = p.parse_known_args(argv)
    return args


def install_if_missing():
    try:
        import torch  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])


def locate_weight(path_text):
    if path_text and Path(path_text).exists():
        return Path(path_text)
    found = []
    for root in (Path("/kaggle/input"), Path("/kaggle/working"), Path.cwd()):
        if root.exists():
            found.extend(root.rglob("generator_ema_final.pth"))
    unique = sorted({p.resolve() for p in found})
    if len(unique) == 1:
        print("[weights] auto-detected", unique[0])
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError("Multiple generator_ema_final.pth files found; set G_PATH explicitly:\n" + "\n".join(map(str, unique)))
    raise FileNotFoundError("generator_ema_final.pth not found. Attach the Exp11 weight Dataset.")


def make_generator(torch):
    import torch.nn as nn

    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.ConvTranspose2d(128, 768, 4, 1, 0, bias=True),
                nn.BatchNorm2d(768), nn.ReLU(True),
                nn.ConvTranspose2d(768, 384, 4, 2, 1, bias=True),
                nn.BatchNorm2d(384), nn.ReLU(True),
                nn.ConvTranspose2d(384, 192, 4, 2, 1, bias=True),
                nn.BatchNorm2d(192), nn.ReLU(True),
                nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=True),
                nn.BatchNorm2d(96), nn.ReLU(True),
                nn.ConvTranspose2d(96, 3, 4, 2, 1, bias=True),
                nn.Tanh(),
            )
        def forward(self, z):
            return self.net(z)
    return Generator()


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    install_if_missing()
    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    if args.batch_size < 1 or args.steps < 1:
        raise ValueError("batch-size and steps must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight_path = locate_weight(args.g_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = make_generator(torch).to(device).eval()
    checkpoint = torch.load(weight_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    generator.load_state_dict(checkpoint, strict=True)
    z = torch.randn(args.batch_size, NOISE_DIM, 1, 1, device=device)
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    trace_path = output_dir / "torch_trace.json"
    # Warm-up must be outside the profiler.  CUDA module loading, cuDNN
    # autotuning and allocator setup are one-time costs, not generator
    # operators.  Keeping them in the trace would make the Top-3 list
    # misleading (for example, "Runtime Triggered Module Loading").
    with torch.inference_mode():
        for _ in range(args.warmup):
            with record_function("Exp11_Generator_Warmup"):
                _ = generator(z)
    if device.type == "cuda":
        torch.cuda.synchronize()
    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=False) as prof:
        for step in range(args.steps):
            with record_function("Exp11_Generator_Inference"):
                _ = generator(z)
            prof.step()
    prof.export_chrome_trace(str(trace_path))
    rows = []
    for item in prof.key_averages():
        cpu_ms = float(item.self_cpu_time_total / 1000.0)
        cuda_ms = float(getattr(item, "self_device_time_total", 0.0) / 1000.0)
        rows.append([item.key, item.count, cpu_ms, cuda_ms, int(getattr(item, "self_cpu_memory_usage", 0)), int(getattr(item, "self_device_memory_usage", 0))])
    rows.sort(key=lambda row: max(row[2], row[3]), reverse=True)
    write_csv(output_dir / "torch_operator_summary.csv", ["operator", "calls", "self_cpu_ms", "self_device_ms", "self_cpu_memory_bytes", "self_device_memory_bytes"], rows)
    archive = output_dir.parent / "Task2_02D_Torch_Reference.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in output_dir.rglob("*"):
            if file_path.is_file():
                zf.write(file_path, arcname=f"02D_Torch_Reference/{file_path.relative_to(output_dir).as_posix()}")
    print("[done] trace:", trace_path)
    print("[zip]", archive)


if __name__ == "__main__":
    main()
