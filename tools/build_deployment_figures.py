"""Build public deployment figures from normalized snapshot tables.

This script only reads curated CSV evidence. It does not train, evaluate, or
invent missing measurements. It uses the standard library so the figures can
be rebuilt without a GPU.
"""

from csv import DictReader
from html import escape
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_metrics_and_logs" / "deployment_optimization"
OUT = ROOT / "04_visual_assets"


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(DictReader(handle))


def text(x, y, value, size=14, anchor="start", weight="400", fill="#0f172a"):
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
        f'font-size="{size}px" text-anchor="{anchor}" font-weight="{weight}" '
        f'fill="{fill}">{escape(str(value))}</text>'
    )


def line(x1, y1, x2, y2, stroke="#cbd5e1", width=1, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def frame(title, desc, width, height):
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc" viewBox="0 0 {width} {height}">',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(desc)}</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        text(width / 2, 38, title, 22, "middle", "700"),
    ]


def deployment_quality_speed():
    rows = read_csv(DATA / "deployment_quantization_summary.csv")
    rows = [row for row in rows if row["standard_fid"]]
    colors = {"FP32": "#64748b", "FP16": "#2563eb", "INT8 PTQ": "#dc2626", "Mixed_PTQ": "#16a34a", "QAT_INT8": "#ea580c"}
    labels = {"FP32": "FP32", "FP16": "FP16", "INT8": "INT8 PTQ", "Mixed_PTQ": "Mixed PTQ", "QAT_INT8": "QAT INT8"}
    width, height = 1240, 620
    parts = frame(
        "Deployment quality-speed snapshot",
        "Standard FID and throughput from the archived quantization tables; benchmark scope labels are preserved in the repository CSV.",
        width,
        height,
    )
    panels = [(90, 570, "Standard FID", "lower is better", "standard_fid", 28.0, 37.0), (660, 1140, "Throughput", "images/s, higher is better", "throughput_images_per_s", 0.0, 27000.0)]
    keys = ["FP32", "FP16", "INT8", "Mixed_PTQ", "QAT_INT8"]
    selected = {}
    for row in rows:
        key = row["label"]
        if key not in selected:
            selected[key] = row
    for left, right, title, axis_label, field, lo, hi in panels:
        top, bottom = 100, 470
        parts += [text((left + right) / 2, 73, title, 17, "middle", "700"), text(left, bottom + 48, axis_label, 12, "start", "400", "#64748b")]
        parts += [line(left, top, left, bottom, "#334155", 1.5), line(left, bottom, right, bottom, "#334155", 1.5)]
        for i in range(5):
            value = lo + (hi - lo) * i / 4
            y = bottom - (value - lo) * (bottom - top) / (hi - lo)
            parts += [line(left, y, right, y, "#cbd5e1", 1, "4 5"), text(left - 10, y + 4, f"{value:.0f}" if hi > 100 else f"{value:.1f}", 11, "end", "400", "#64748b")]
        step = (right - left) / len(keys)
        bar_w = step * 0.58
        for i, key in enumerate(keys):
            row = selected.get(key)
            if not row:
                continue
            value = float(row[field])
            x = left + step * i + (step - bar_w) / 2
            y = bottom - (value - lo) * (bottom - top) / (hi - lo)
            color = colors.get(key, "#475569")
            parts += [f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bottom-y:.1f}" fill="{color}"/>']
            parts += [text(x + bar_w / 2, y - 8, f"{value:.2f}" if hi < 100 else f"{value:,.0f}", 11, "middle", "700")]
            parts += [text(x + bar_w / 2, bottom + 22, labels[key], 11, "middle", "500")]
    parts += [text(width / 2, height - 24, "Task 3/4/5 benchmark scopes are documented separately; do not treat every latency row as one identical protocol.", 12, "middle", "400", "#64748b"), "</svg>"]
    return "\n".join(parts)


def service_stress():
    rows = read_csv(DATA / "06_Service_Stress" / "service_stress_summary.csv")
    width, height = 1240, 560
    parts = frame(
        "Staged service stress: concurrency, P99, and throughput",
        "Locust staged-load evidence on a Tesla T4. All tested stages returned zero failed requests.",
        width,
        height,
    )
    panels = [(90, 570, "P99 latency", "milliseconds", "p99_ms", 0, 520, "#dc2626"), (660, 1140, "Throughput", "requests/s", "rps", 230, 350, "#2563eb")]
    x_values = [int(row["concurrency"]) for row in rows]
    for left, right, title, axis_label, field, lo, hi, color in panels:
        top, bottom = 100, 425
        parts += [text((left + right) / 2, 73, title, 17, "middle", "700"), text(left, bottom + 48, axis_label, 12, "start", "400", "#64748b")]
        parts += [line(left, top, left, bottom, "#334155", 1.5), line(left, bottom, right, bottom, "#334155", 1.5)]
        for i in range(5):
            value = lo + (hi - lo) * i / 4
            y = bottom - (value - lo) * (bottom - top) / (hi - lo)
            parts += [line(left, y, right, y, "#cbd5e1", 1, "4 5"), text(left - 10, y + 4, f"{value:.0f}", 11, "end", "400", "#64748b")]
        def sx(value):
            return left + (value - min(x_values)) * (right - left) / (max(x_values) - min(x_values))
        def sy(value):
            return bottom - (value - lo) * (bottom - top) / (hi - lo)
        points = [(sx(int(row["concurrency"])), sy(float(row[field]))) for row in rows]
        path = " ".join(("M" if i == 0 else "L") + f" {x:.1f} {y:.1f}" for i, (x, y) in enumerate(points))
        parts += [f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>']
        for row, (x, y) in zip(rows, points):
            parts += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>', text(x, y - 12, f"{float(row[field]):.0f}", 11, "middle", "700")]
        for value in (1, 16, 32, 64, 128):
            parts += [text(sx(value), bottom + 22, value, 11, "middle", "500")]
        parts += [text(right, bottom + 22, "concurrency", 11, "end", "500")]
    parts += [text(width / 2, height - 24, "Staged run: concurrency 1–128, zero failures; no hard crash observed at the tested maximum.", 12, "middle", "400", "#64748b"), "</svg>"]
    return "\n".join(parts)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "deployment_quality_speed.svg").write_text(deployment_quality_speed(), encoding="utf-8")
    (OUT / "service_stress_summary.svg").write_text(service_stress(), encoding="utf-8")
    print("Wrote deployment_quality_speed.svg and service_stress_summary.svg")


if __name__ == "__main__":
    main()
