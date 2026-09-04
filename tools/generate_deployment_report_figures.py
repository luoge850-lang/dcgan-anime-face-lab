from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path.cwd()
RESULT_ROOT = PROJECT_ROOT / "results" / "Deployment_Optimization_Results"
OUT = RESULT_ROOT / "figures"
EVIDENCE_07 = RESULT_ROOT / "07" / "07_MLOps_Observability" / "evidence"
EVIDENCE_08 = RESULT_ROOT / "08_Model_Hot_Update_AB" / "evidence"

W, H = 1200, 700
L, R, T, B = 150, 70, 95, 135
PLOT_W, PLOT_H = W - L - R, H - T - B
FONT = "Microsoft YaHei, SimHei, Arial, sans-serif"
BLUE = "#4472C4"
ORANGE = "#ED7D31"
GREEN = "#70AD47"
GRAY = "#A5A5A5"
RED = "#C00000"
GRID = "#D9E2F3"
TEXT = "#222222"
PNG_W, PNG_H = 1600, 900
PNG_L, PNG_R, PNG_T, PNG_B = 190, 90, 145, 175
PNG_PLOT_W, PNG_PLOT_H = PNG_W - PNG_L - PNG_R, PNG_H - PNG_T - PNG_B


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def public_source(path: Path) -> str:
    """Return a repository-relative source path for the public manifest."""
    try:
        return path.resolve().relative_to(RESULT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def text(x, y, value, size=18, anchor="middle", weight="400", fill=TEXT, extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" font-size="{size}px" font-weight="{weight}" text-anchor="{anchor}" fill="{fill}" {extra}>{esc(value)}</text>'


def line(x1, y1, x2, y2, stroke=TEXT, width=1, dash=""):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def rect(x, y, width, height, fill, stroke="none", stroke_width=1, opacity=1.0):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{height:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" opacity="{opacity}"/>'


def circle(x, y, radius, fill, stroke="none", stroke_width=1):
    return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'


def base_svg(title: str, subtitle: str = "") -> list[str]:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        f'<title>{esc(title)}</title>',
        f'<desc>{esc(subtitle or title)}</desc>',
        rect(0, 0, W, H, "#FFFFFF"),
        text(W / 2, 48, title, 25, weight="600"),
    ]
    if subtitle:
        parts.append(text(W / 2, 76, subtitle, 14, fill="#666666"))
    return parts


def finish_svg(parts: list[str]) -> str:
    parts.append("</svg>")
    return "\n".join(parts)


def axes(parts: list[str], y_max: float, y_ticks: list[float], x_label: str, y_label: str, x_labels: list[str] | None = None):
    x0, y0 = L, T + PLOT_H
    x1, y1 = L + PLOT_W, T
    for tick in y_ticks:
        y = y0 - (tick / y_max) * PLOT_H
        parts.append(line(x0, y, x1, y, GRID, 1, "4 4"))
        parts.append(text(x0 - 14, y + 5, tick, 14, anchor="end", fill="#555555"))
    parts.append(line(x0, y0, x1, y0, TEXT, 1.5))
    parts.append(line(x0, y0, x0, y1, TEXT, 1.5))
    parts.append(text((x0 + x1) / 2, H - 35, x_label, 16))
    parts.append(text(42, (y0 + y1) / 2, y_label, 16, extra=f'transform="rotate(-90 42 {(y0 + y1) / 2:.1f})"'))
    if x_labels:
        step = PLOT_W / len(x_labels)
        for i, label in enumerate(x_labels):
            x = x0 + step * (i + 0.5)
            parts.append(line(x, y0, x, y0 + 7, TEXT, 1))
            parts.append(text(x, y0 + 31, label, 15))


def value_label(parts: list[str], x: float, y: float, value: float, suffix="", color=TEXT):
    parts.append(text(x, y - 10, f"{value:.2f}{suffix}", 14, fill=color))


def png_font(size: int, bold: bool = False):
    candidates = [
        Path(r"C:\Windows\Fonts\msyhbd.ttc") if bold else Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def png_text(draw, xy, value, size=22, anchor="mm", fill=TEXT, bold=False):
    draw.text(xy, str(value), font=png_font(size, bold), fill=fill, anchor=anchor)


def png_axes(draw, y_max, y_ticks, x_labels, x_label, y_label):
    x0, y0 = PNG_L, PNG_T + PNG_PLOT_H
    x1, y1 = PNG_L + PNG_PLOT_W, PNG_T
    for tick in y_ticks:
        y = y0 - (tick / y_max) * PNG_PLOT_H
        draw.line((x0, y, x1, y), fill=GRID, width=2)
        png_text(draw, (x0 - 18, y), tick, 19, anchor="rm", fill="#555555")
    draw.line((x0, y0, x1, y0), fill=TEXT, width=3)
    draw.line((x0, y0, x0, y1), fill=TEXT, width=3)
    png_text(draw, ((x0 + x1) / 2, PNG_H - 58), x_label, 22)
    draw.text((96, (y0 + y1) / 2), y_label, font=png_font(22), fill=TEXT, anchor="mm", angle=90)
    step = PNG_PLOT_W / len(x_labels)
    for i, label in enumerate(x_labels):
        x = x0 + step * (i + 0.5)
        draw.line((x, y0, x, y0 + 10), fill=TEXT, width=2)
        png_text(draw, (x, y0 + 42), label, 21)


def png_bar(draw, x, bottom, width, value, y_max, color):
    height = value / y_max * PNG_PLOT_H
    top = bottom - height
    draw.rectangle((x, top, x + width, bottom), fill=color)
    return top


def png_figure_02(path: Path, audit: dict) -> None:
    image = Image.new("RGB", (PNG_W, PNG_H), "white")
    draw = ImageDraw.Draw(image)
    png_text(draw, (PNG_W / 2, 55), "图2  队列积压告警触发与恢复记录", 30, bold=True)
    png_text(draw, (PNG_W / 2, 98), "受控模拟：队列深度=60，告警阈值>50，规则持续时间=30秒；事件日志跨度约10秒", 19, fill="#666666")
    x0, x1, y = 245, 1355, 440
    draw.line((x0, y, x1, y), fill="#8FAADC", width=18)
    draw.ellipse((x0 - 16, y - 16, x0 + 16, y + 16), fill=RED)
    draw.ellipse((x1 - 16, y - 16, x1 + 16, y + 16), fill=GREEN)
    png_text(draw, (x0, y - 74), "firing：告警触发", 23, fill=RED, bold=True)
    png_text(draw, (x1, y - 74), "resolved：告警恢复", 23, fill=GREEN, bold=True)
    png_text(draw, (x0, y + 66), audit["starts_at"].replace("T", " ").replace("Z", " UTC"), 19, fill="#555555")
    png_text(draw, (x1, y + 66), audit["ends_at"].replace("T", " ").replace("Z", " UTC"), 19, fill="#555555")
    png_text(draw, (PNG_W / 2, y + 4), "DCGANQueueBacklog", 25, bold=True)
    png_text(draw, (PNG_W / 2, PNG_H - 82), f"实际记录持续约 {audit['duration_seconds']:.0f} 秒；状态由 firing 变为 resolved", 19, fill="#555555")
    image.save(path, format="PNG", dpi=(300, 300))


def png_figure_03(path: Path, audit: dict) -> None:
    image = Image.new("RGB", (PNG_W, PNG_H), "white")
    draw = ImageDraw.Draw(image)
    png_text(draw, (PNG_W / 2, 55), "图3  A/B 灰度发布目标流量比例与实际流量比例对比", 30, bold=True)
    png_text(draw, (PNG_W / 2, 98), "实际比例来自灰度请求日志，误差均小于5个百分点", 19, fill="#666666")
    labels = [f"{x:.0f}%" for x in audit["target_ratio_percent"]]
    png_axes(draw, 120, [0, 20, 40, 60, 80, 100, 120], labels, "B版本目标流量比例", "流量（%）")
    step = PNG_PLOT_W / len(labels)
    width = 92
    bottom = PNG_T + PNG_PLOT_H
    for i, (target, actual, error) in enumerate(zip(audit["target_ratio_percent"], audit["observed_ratio_percent"], audit["error_pp"])):
        center = PNG_L + step * (i + 0.5)
        top_t = png_bar(draw, center - width - 8, bottom, width, target, 120, GRAY)
        top_a = png_bar(draw, center + 8, bottom, width, actual, 120, GREEN)
        png_text(draw, (center - width / 2 - 8, top_t - 18), f"{target:.1f}%", 18)
        png_text(draw, (center + width / 2 + 8, top_a - 18), f"{actual:.1f}%", 18)
        png_text(draw, (center, min(top_t, top_a) - 56), f"误差 {error:.1f}pp", 18, fill="#555555")
    draw.rectangle((PNG_W - 450, 128, PNG_W - 425, 153), fill=GRAY)
    png_text(draw, (PNG_W - 405, 140), "目标比例", 18, anchor="lm")
    draw.rectangle((PNG_W - 260, 128, PNG_W - 235, 153), fill=GREEN)
    png_text(draw, (PNG_W - 215, 140), "实际比例", 18, anchor="lm")
    image.save(path, format="PNG", dpi=(300, 300))


def png_figure_04(path: Path, audit: dict) -> None:
    image = Image.new("RGB", (PNG_W, PNG_H), "white")
    draw = ImageDraw.Draw(image)
    png_text(draw, (PNG_W / 2, 55), "图4  PTQ-INT8 与 QAT-INT8 的 P50、P95 和 P99 延迟对比", 30, bold=True)
    png_text(draw, (PNG_W / 2, 98), "延迟越低越好；虚线为200ms告警阈值", 19, fill="#666666")
    labels = ["P50", "P95", "P99"]
    png_axes(draw, 230, [0, 50, 100, 150, 200, 230], labels, "延迟分位数", "延迟（ms）")
    bottom = PNG_T + PNG_PLOT_H
    y_threshold = bottom - (200 / 230) * PNG_PLOT_H
    draw.line((PNG_L, y_threshold, PNG_L + PNG_PLOT_W, y_threshold), fill=RED, width=3)
    png_text(draw, (PNG_L + PNG_PLOT_W - 8, y_threshold - 18), "200 ms 阈值", 18, anchor="rm", fill=RED)
    step = PNG_PLOT_W / len(labels)
    width = 92
    for i, (va, vb) in enumerate(zip(audit["ptq_int8_ms"], audit["qat_int8_ms"])):
        center = PNG_L + step * (i + 0.5)
        top_a = png_bar(draw, center - width - 8, bottom, width, va, 230, BLUE)
        top_b = png_bar(draw, center + 8, bottom, width, vb, 230, ORANGE)
        png_text(draw, (center - width / 2 - 8, top_a - 18), f"{va:.2f}", 17)
        png_text(draw, (center + width / 2 + 8, top_b - 18), f"{vb:.2f}", 17)
    draw.rectangle((PNG_W - 450, 128, PNG_W - 425, 153), fill=BLUE)
    png_text(draw, (PNG_W - 405, 140), "A：PTQ-INT8", 18, anchor="lm")
    draw.rectangle((PNG_W - 260, 128, PNG_W - 235, 153), fill=ORANGE)
    png_text(draw, (PNG_W - 215, 140), "B：QAT-INT8", 18, anchor="lm")
    image.save(path, format="PNG", dpi=(300, 300))


def png_figure_05(path: Path, audit: dict) -> None:
    image = Image.new("RGB", (PNG_W, PNG_H), "white")
    draw = ImageDraw.Draw(image)
    png_text(draw, (PNG_W / 2, 55), "图5  PTQ-INT8 与 QAT-INT8 的 FID 对比", 30, bold=True)
    png_text(draw, (PNG_W / 2, 98), "FID越低越好；A/B各使用5000张生成图像进行同协议抽样评估", 19, fill="#666666")
    labels = audit["labels"]
    png_axes(draw, 45, [0, 10, 20, 30, 40, 45], labels, "模型版本", "FID（越低越好）")
    bottom = PNG_T + PNG_PLOT_H
    step = PNG_PLOT_W / 2
    width = 150
    for i, value in enumerate(audit["fid"]):
        center = PNG_L + step * (i + 0.5)
        top = png_bar(draw, center - width / 2, bottom, width, value, 45, (BLUE, ORANGE)[i])
        png_text(draw, (center, top - 20), f"{value:.4f}", 19)
    png_text(draw, (PNG_W / 2, PNG_H - 82), "同一轮评估：版本A与版本B各抽样5000张图像", 19, fill="#555555")
    image.save(path, format="PNG", dpi=(300, 300))


def figure_02() -> tuple[dict, str]:
    source = EVIDENCE_07 / "alert_webhook_events.jsonl"
    events = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    firing = next(item for item in events if item.get("status") == "firing")
    resolved = next(item for item in events if item.get("status") == "resolved")
    start = datetime.fromisoformat(firing["alerts"][0]["startsAt"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(resolved["alerts"][0]["endsAt"].replace("Z", "+00:00"))
    duration = (end - start).total_seconds()
    parts = base_svg("图2  队列积压告警触发与恢复记录", "受控模拟：队列深度=60，告警阈值>50，规则持续时间=30秒；事件日志跨度约10秒")
    x0, y0 = L, T + PLOT_H / 2
    x1 = L + PLOT_W
    parts.append(line(x0, y0, x1, y0, "#8FAADC", 10))
    parts.append(circle(x0, y0, 12, RED))
    parts.append(circle(x1, y0, 12, GREEN))
    parts.append(text(x0, y0 - 42, "firing：告警触发", 18, weight="600", fill=RED))
    parts.append(text(x1, y0 - 42, "resolved：告警恢复", 18, weight="600", fill=GREEN))
    parts.append(text(x0, y0 + 42, start.strftime("%H:%M:%S UTC"), 15, fill="#555555"))
    parts.append(text(x1, y0 + 42, end.strftime("%H:%M:%S UTC"), 15, fill="#555555"))
    parts.append(text(W / 2, y0 + 6, "DCGANQueueBacklog", 19, weight="600"))
    parts.append(text(W / 2, H - 68, f"实际记录持续约 {duration:.0f} 秒；状态由 firing 变为 resolved", 15, fill="#555555"))
    audit = {
        "source": public_source(source), "source_sha256": sha256(source), "alert": "DCGANQueueBacklog",
        "starts_at": firing["alerts"][0]["startsAt"], "ends_at": resolved["alerts"][0]["endsAt"],
        "duration_seconds": duration, "simulation": True,
    }
    return audit, finish_svg(parts)


def figure_03() -> tuple[dict, str]:
    source = EVIDENCE_08 / "08_traffic_split.csv"
    rows = read_csv(source)
    labels = [f"{float(row['target_ratio_b']):.0%}" for row in rows]
    target = [float(row["target_ratio_b"]) * 100 for row in rows]
    actual = [float(row["observed_ratio_b"]) * 100 for row in rows]
    errors = [float(row["error_pp"]) for row in rows]
    parts = base_svg("图3  A/B 灰度发布目标流量比例与实际流量比例对比", "实际比例来自灰度请求日志，比例误差均小于5个百分点")
    axes(parts, 120, [0, 20, 40, 60, 80, 100, 120], "B版本目标流量比例", "流量比例（%）", labels)
    step = PLOT_W / len(labels)
    width = 72
    for i, (t, a, e) in enumerate(zip(target, actual, errors)):
        center = L + step * (i + 0.5)
        h_t = (t / 120) * PLOT_H
        h_a = (a / 120) * PLOT_H
        parts.append(rect(center - width - 5, T + PLOT_H - h_t, width, h_t, GRAY))
        parts.append(rect(center + 5, T + PLOT_H - h_a, width, h_a, GREEN))
        value_label(parts, center - width / 2 - 5, T + PLOT_H - h_t, t, "%")
        value_label(parts, center + width / 2 + 5, T + PLOT_H - h_a, a, "%")
        parts.append(text(center, T + PLOT_H - max(h_t, h_a) - 42, f"误差 {e:.1f}pp", 14, fill="#555555"))
    parts.append(rect(W - 370, 105, 18, 18, GRAY))
    parts.append(text(W - 340, 121, "目标比例", 15, anchor="start"))
    parts.append(rect(W - 220, 105, 18, 18, GREEN))
    parts.append(text(W - 190, 121, "实际比例", 15, anchor="start"))
    audit = {
        "source": public_source(source), "source_sha256": sha256(source),
        "target_ratio_percent": target, "observed_ratio_percent": actual, "error_pp": errors,
    }
    return audit, finish_svg(parts)


def figure_04() -> tuple[dict, str]:
    source = EVIDENCE_08 / "08_latency_by_version.csv"
    rows = read_csv(source)
    a = next(row for row in rows if row["version"] == "A")
    b = next(row for row in rows if row["version"] == "B")
    labels = ["P50", "P95", "P99"]
    vals_a = [float(a[key]) for key in ("p50_ms", "p95_ms", "p99_ms")]
    vals_b = [float(b[key]) for key in ("p50_ms", "p95_ms", "p99_ms")]
    parts = base_svg("图4  PTQ-INT8 与 QAT-INT8 的 P50、P95 和 P99 延迟对比", "延迟越低越好；虚线为200ms告警阈值")
    axes(parts, 230, [0, 50, 100, 150, 200, 230], "延迟分位数", "延迟（ms）", labels)
    y_threshold = T + PLOT_H - (200 / 230) * PLOT_H
    parts.append(line(L, y_threshold, L + PLOT_W, y_threshold, RED, 2, "8 5"))
    parts.append(text(L + PLOT_W - 4, y_threshold - 10, "200 ms 阈值", 14, anchor="end", fill=RED))
    step = PLOT_W / len(labels)
    width = 72
    for i, (va, vb) in enumerate(zip(vals_a, vals_b)):
        center = L + step * (i + 0.5)
        ha = (va / 230) * PLOT_H
        hb = (vb / 230) * PLOT_H
        parts.append(rect(center - width - 5, T + PLOT_H - ha, width, ha, BLUE))
        parts.append(rect(center + 5, T + PLOT_H - hb, width, hb, ORANGE))
        value_label(parts, center - width / 2 - 5, T + PLOT_H - ha, va)
        value_label(parts, center + width / 2 + 5, T + PLOT_H - hb, vb)
    parts.append(rect(W - 375, 105, 18, 18, BLUE))
    parts.append(text(W - 345, 121, "A：PTQ-INT8", 15, anchor="start"))
    parts.append(rect(W - 220, 105, 18, 18, ORANGE))
    parts.append(text(W - 190, 121, "B：QAT-INT8", 15, anchor="start"))
    audit = {
        "source": public_source(source), "source_sha256": sha256(source),
        "ptq_int8_ms": vals_a, "qat_int8_ms": vals_b, "threshold_ms": 200,
    }
    return audit, finish_svg(parts)


def figure_05() -> tuple[dict, str]:
    source = EVIDENCE_08 / "08_fid_sample.csv"
    rows = read_csv(source)
    a = next(row for row in rows if row["version"] == "A")
    b = next(row for row in rows if row["version"] == "B")
    values = [float(a["fid_sample"]), float(b["fid_sample"])]
    labels = ["PTQ-INT8", "QAT-INT8"]
    parts = base_svg("图5  PTQ-INT8 与 QAT-INT8 的 FID 对比", "FID越低越好；A/B各使用5000张生成图像进行同协议抽样评估")
    y_max = 45
    axes(parts, y_max, [0, 10, 20, 30, 40, 45], "模型版本", "FID（越低越好）", labels)
    step = PLOT_W / 2
    width = 120
    for i, value in enumerate(values):
        center = L + step * (i + 0.5)
        height = value / y_max * PLOT_H
        parts.append(rect(center - width / 2, T + PLOT_H - height, width, height, (BLUE, ORANGE)[i]))
        value_label(parts, center, T + PLOT_H - height, value)
    parts.append(text(W / 2, H - 68, "同一轮评估：版本A与版本B各抽样5000张图像", 15, fill="#555555"))
    audit = {
        "source": public_source(source), "source_sha256": sha256(source), "labels": labels,
        "fid": values, "sample_count_each": 5000, "lower_is_better": True,
    }
    return audit, finish_svg(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild the Stage 7/8 deployment figures from curated evidence."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing results/ (default: current directory)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="Override the deployment result root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the directory receiving SVG, PNG, and manifest outputs",
    )
    args = parser.parse_args()

    global PROJECT_ROOT, RESULT_ROOT, OUT, EVIDENCE_07, EVIDENCE_08
    PROJECT_ROOT = args.project_root.resolve()
    RESULT_ROOT = (args.results_root or (PROJECT_ROOT / "results" / "Deployment_Optimization_Results")).resolve()
    OUT = (args.output_dir or (RESULT_ROOT / "figures")).resolve()
    EVIDENCE_07 = RESULT_ROOT / "07" / "07_MLOps_Observability" / "evidence"
    EVIDENCE_08 = RESULT_ROOT / "08_Model_Hot_Update_AB" / "evidence"

    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("fig02_queue_alert_lifecycle.svg", figure_02),
        ("fig03_ab_traffic_split.svg", figure_03),
        ("fig04_latency_ptq_vs_qat.svg", figure_04),
        ("fig05_fid_ptq_vs_qat.svg", figure_05),
    ]
    manifest = {"generated_at_local": datetime.now().astimezone().isoformat(), "figures": {}}
    for name, builder in jobs:
        audit, svg = builder()
        (OUT / name).write_text(svg, encoding="utf-8")
        manifest["figures"][name] = audit
    png_figure_02(OUT / "fig02_queue_alert_lifecycle.png", manifest["figures"]["fig02_queue_alert_lifecycle.svg"])
    png_figure_03(OUT / "fig03_ab_traffic_split.png", manifest["figures"]["fig03_ab_traffic_split.svg"])
    png_figure_04(OUT / "fig04_latency_ptq_vs_qat.png", manifest["figures"]["fig04_latency_ptq_vs_qat.svg"])
    png_figure_05(OUT / "fig05_fid_ptq_vs_qat.png", manifest["figures"]["fig05_fid_ptq_vs_qat.svg"])
    manifest["png_outputs"] = [
        "fig02_queue_alert_lifecycle.png",
        "fig03_ab_traffic_split.png",
        "fig04_latency_ptq_vs_qat.png",
        "fig05_fid_ptq_vs_qat.png",
    ]
    (OUT / "figure_generation_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
