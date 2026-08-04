"""Build interview-facing SVG figures from the curated results table.

The generator uses only the Python standard library. It does not retrain a
model and it deliberately keeps incompatible historical protocols out of the
headline figures.
"""

from pathlib import Path
import csv
from html import escape


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results_summary.csv"
OUT = ROOT / "04_visual_assets"


def load_rows():
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find(rows, experiment):
    return next(row for row in rows if row["experiment"] == experiment)


def text(x, y, value, size=16, anchor="start", weight="400", fill="#0f172a"):
    return f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}px" text-anchor="{anchor}" font-weight="{weight}" fill="{fill}">{escape(str(value))}</text>'


def line(x1, y1, x2, y2, stroke="#cbd5e1", width=1, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def roadmap(rows):
    milestones = [
        ("00 baseline", "00_baseline"),
        ("D SN+Hinge", "13_D_SN_Hinge"),
        ("G Width x3", "03_G_Width3x"),
        ("20K data", "09_G_Width3x_20K"),
        ("DiffAug + EMA", "11_G_DiffAug_EMA_20K"),
        ("C0 control", "C0_continue_L0"),
        ("C1 CLIP λ=.01", "C1_clip_L001"),
    ]
    labels = [label for label, _ in milestones]
    values = [float(find(rows, key)["fid_legacy_project"]) for _, key in milestones]
    colors = ["#64748b", "#0f766e", "#2563eb", "#2563eb", "#7c3aed", "#475569", "#ea580c"]
    width, height = 1240, 570
    left, right, top, bottom = 105, 1185, 95, 455
    lo, hi = 30, 120

    def sx(i):
        return left + i * (right - left) / (len(values) - 1)

    def sy(v):
        return top + (v - lo) * (bottom - top) / (hi - lo)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">']
    parts += ['<title id="title">DCGAN experiment roadmap</title>', '<desc id="desc">Legacy project FID decreases from the baseline through discriminator stabilization, generator scaling, data scaling, DiffAugment plus EMA, and a matched continuation control.</desc>']
    parts += ['<rect width="100%" height="100%" fill="#ffffff"/>', text(width / 2, 42, "DCGAN experiment roadmap: stabilization, scale, and controlled continuation", 23, "middle", "700")]
    parts += [text(left, 72, "Legacy project FID (lower is better)", 15, "start", "500", "#475569")]
    parts += [line(left, top, left, bottom, "#334155", 1.5), line(left, bottom, right, bottom, "#334155", 1.5)]
    for tick in (40, 60, 80, 100, 120):
        y = sy(tick)
        parts += [line(left, y, right, y, "#cbd5e1", 1, "4 5"), text(left - 14, y + 5, tick, 13, "end", "400", "#64748b")]
    path = " ".join(("M" if i == 0 else "L") + f" {sx(i):.1f} {sy(v):.1f}" for i, v in enumerate(values))
    parts += [f'<path d="{path}" fill="none" stroke="#94a3b8" stroke-width="3"/>']
    for i, (label, value, color) in enumerate(zip(labels, values, colors)):
        x, y = sx(i), sy(value)
        parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" stroke="#ffffff" stroke-width="3"/>']
        label_y = y - 18 if i not in (0, 5) else y + 28
        parts += [text(x, label_y, f"{value:.2f}", 13, "middle", "700")]
        parts += [text(x, bottom + 28, label, 13, "middle", "500")]
    parts += [text(left, height - 32, "Same project protocol where indicated; legacy FID is not a clean-fid benchmark.", 12, "start", "400", "#64748b")]
    parts += ["</svg>"]
    return "\n".join(parts)


def clip_sweep(rows):
    keys = ["E0_exp11_formal_eval", "C0_continue_L0", "C1_clip_L001", "C2_clip_L0025", "C3_clip_L005", "C4_clip_L010"]
    labels = ["E0\neval", "C0\nλ=0", "C1\nλ=.01", "C2\nλ=.025", "C3\nλ=.05", "C4\nλ=.10"]
    selected = [find(rows, key) for key in keys]
    fids = [float(row["fid_legacy_project"]) for row in selected]
    mmd = [None] + [float(row["clip_mmd2_unbiased"]) for row in selected[1:]]
    colors = ["#64748b", "#475569", "#ea580c", "#f59e0b", "#f59e0b", "#16a34a"]
    width, height = 1240, 520
    panels = [(80, 555, "Legacy FID", "lower is better", fids, 30, 40, labels, colors), (680, 1155, "CLIP feature MMD² (unbiased)", "lower is better", mmd[1:], 0.040, 0.0445, labels[1:], colors[1:])]
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">']
    parts += ['<title id="title">CLIP controlled continuation sweep</title>', '<desc id="desc">Side-by-side bars compare legacy FID and CLIP feature MMD squared for an evaluation control, a no-CLIP continuation control, and four CLIP regularization strengths.</desc>']
    parts += ['<rect width="100%" height="100%" fill="#ffffff"/>', text(width / 2, 38, "Matched 50-epoch continuation: metric trade-off, not a single winner", 22, "middle", "700")]
    for left, right, title, ylabel, values, lo, hi, panel_labels, panel_colors in panels:
        top, bottom = 90, 405
        parts += [text((left + right) / 2, 70, title, 17, "middle", "700"), text(left, bottom + 45, ylabel, 12, "start", "400", "#64748b")]
        parts += [line(left, top, left, bottom, "#334155", 1.5), line(left, bottom, right, bottom, "#334155", 1.5)]
        for tick_i in range(5):
            tick = lo + (hi - lo) * tick_i / 4
            y = bottom - (tick - lo) * (bottom - top) / (hi - lo)
            parts += [line(left, y, right, y, "#cbd5e1", 1, "4 5"), text(left - 10, y + 4, f"{tick:.2f}" if hi - lo > 1 else f"{tick:.5f}", 11, "end", "400", "#64748b")]
        step = (right - left) / len(values)
        bar_w = step * 0.58
        for i, (value, label, color) in enumerate(zip(values, panel_labels, panel_colors)):
            if value is None:
                continue
            x = left + step * i + (step - bar_w) / 2
            y = bottom - (value - lo) * (bottom - top) / (hi - lo)
            parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bottom-y:.1f}" rx="4" fill="{color}"/>']
            parts += [text(x + bar_w / 2, y - 8, f"{value:.2f}" if hi - lo > 1 else f"{value:.5f}", 11, "middle", "700")]
            first, *rest = label.split("\n")
            parts += [text(x + bar_w / 2, bottom + 22, first, 12, "middle", "500")]
            if rest:
                parts += [text(x + bar_w / 2, bottom + 37, rest[0], 12, "middle", "500")]
    parts += [text(width / 2, height - 25, "C0 is the no-CLIP control; C4 minimizes CLIP MMD² while C1 has the lowest FID in this local sweep.", 12, "middle", "400", "#64748b"), "</svg>"]
    return "\n".join(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    (OUT / "interview_results_roadmap.svg").write_text(roadmap(rows), encoding="utf-8")
    (OUT / "clip_control_sweep.svg").write_text(clip_sweep(rows), encoding="utf-8")
    print("Wrote interview_results_roadmap.svg and clip_control_sweep.svg")


if __name__ == "__main__":
    main()
