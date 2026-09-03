"""Build small, public-safe Stage 8 SVGs from the curated evidence CSVs.

This intentionally uses only the Python standard library so it can run locally
without re-installing the Kaggle training environment. It creates one metric per
figure and never embeds the raw request log or engine binaries.
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path


WIDTH = 1400
HEIGHT = 760
LEFT = 150
RIGHT = 90
TOP = 120
BOTTOM = 130
PLOT_W = WIDTH - LEFT - RIGHT
PLOT_H = HEIGHT - TOP - BOTTOM
FONT = '"Microsoft YaHei","Noto Sans CJK SC","SimHei",Arial,sans-serif'


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def header(title: str, subtitle: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="title desc">',
        f'<title id="title">{esc(title)}</title><desc id="desc">{esc(subtitle)}</desc>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#FFFFFF"/>',
        f'<style>text{{font-family:{FONT};fill:#1F2937}}.title{{font-size:30px;font-weight:700}}.subtitle{{font-size:17px;fill:#4B5563}}.axis{{font-size:18px}}.tick{{font-size:15px;fill:#4B5563}}.value{{font-size:18px;font-weight:700}}.note{{font-size:13px;fill:#4B5563}}.grid{{stroke:#D1D5DB;stroke-width:1}}.frame{{stroke:#374151;stroke-width:1.2;fill:none}}</style>',
        f'<text x="{WIDTH / 2:.1f}" y="48" text-anchor="middle" class="title">{esc(title)}</text>',
        f'<text x="{WIDTH / 2:.1f}" y="78" text-anchor="middle" class="subtitle">{esc(subtitle)}</text>',
    ]


def finish(parts: list[str], note: str) -> str:
    parts.append(f'<text x="{LEFT}" y="{HEIGHT - 28}" class="note">数据源：Stage 8 curated evidence CSV；单节点 Kaggle 运行；结果不能外推为多副本生产集群</text>')
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def bar_chart(path: Path, title: str, subtitle: str, labels: list[str], values: list[float], y_label: str, max_y: float, decimals: int = 1) -> None:
    parts = header(title, subtitle)
    base_y = TOP + PLOT_H
    parts.append(f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_W}" height="{PLOT_H}" class="frame"/>')
    for i in range(5):
        value = max_y * i / 4
        y = base_y - PLOT_H * i / 4
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{LEFT - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.{decimals}f}</text>')
    slot = PLOT_W / max(len(labels), 1)
    bar_w = min(220, slot * 0.48)
    colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
    for index, (label, value) in enumerate(zip(labels, values)):
        x = LEFT + slot * (index + 0.5) - bar_w / 2
        height = max(0.0, min(PLOT_H, PLOT_H * value / max_y))
        y = base_y - height
        color = colors[index % len(colors)]
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{height:.1f}" rx="6" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 12:.1f}" text-anchor="middle" class="value">{value:.{decimals}f}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{base_y + 38}" text-anchor="middle" class="tick">{esc(label)}</text>')
    parts.append(f'<text x="{LEFT + PLOT_W / 2:.1f}" y="{HEIGHT - 70}" text-anchor="middle" class="axis">版本</text>')
    parts.append(f'<text x="34" y="{TOP - 18}" class="axis">{esc(y_label)}</text>')
    path.write_text(finish(parts, subtitle), encoding="utf-8")


def split_chart(path: Path, rows: list[dict[str, str]]) -> None:
    title = "Stage 8：A/B 目标流量与实际流量"
    subtitle = "验证 10%、50%、100% 灰度目标的路由误差"
    parts = header(title, subtitle)
    base_y = TOP + PLOT_H
    max_y = 1.0
    for i in range(5):
        value = max_y * i / 4
        y = base_y - PLOT_H * i / 4
        parts.append(f'<line x1="{LEFT}" y1="{y:.1f}" x2="{LEFT + PLOT_W}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{LEFT - 14}" y="{y + 5:.1f}" text-anchor="end" class="tick">{value:.0%}</text>')
    parts.append(f'<rect x="{LEFT}" y="{TOP}" width="{PLOT_W}" height="{PLOT_H}" class="frame"/>')
    xs = [LEFT + PLOT_W * (i + 0.5) / len(rows) for i in range(len(rows))]
    points_by_series: list[tuple[str, list[float], str]] = []
    for name, color in (("目标 B 比例", "#6B7280"), ("实际 B 比例", "#0072B2")):
        vals = [float(r["target_ratio_b"]) if name.startswith("目标") else float(r["observed_ratio_b"]) for r in rows]
        points_by_series.append((name, vals, color))
    for name, vals, color in points_by_series:
        points = " ".join(f'{x:.1f},{base_y - PLOT_H * val:.1f}' for x, val in zip(xs, vals))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="4"/>')
        for x, val in zip(xs, vals):
            y = base_y - PLOT_H * val
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{color}"/><text x="{x:.1f}" y="{y - 16:.1f}" text-anchor="middle" class="value">{val:.1%}</text>')
    for x, row in zip(xs, rows):
        parts.append(f'<text x="{x:.1f}" y="{base_y + 38}" text-anchor="middle" class="tick">目标 {float(row["target_ratio_b"]):.0%}</text>')
    parts.append(f'<line x1="{WIDTH - 380}" y1="145" x2="{WIDTH - 340}" y2="145" stroke="#6B7280" stroke-width="4"/><text x="{WIDTH - 330}" y="151" class="tick">目标 B 比例</text>')
    parts.append(f'<line x1="{WIDTH - 380}" y1="180" x2="{WIDTH - 340}" y2="180" stroke="#0072B2" stroke-width="4"/><text x="{WIDTH - 330}" y="186" class="tick">实际 B 比例</text>')
    parts.append(f'<text x="{LEFT + PLOT_W / 2:.1f}" y="{HEIGHT - 70}" text-anchor="middle" class="axis">灰度阶段</text>')
    parts.append(f'<text x="34" y="{TOP - 18}" class="axis">B 流量比例</text>')
    path.write_text(finish(parts, subtitle), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    latency = read_csv(args.evidence / "08_latency_by_version.csv")
    fid = read_csv(args.evidence / "08_fid_sample.csv")
    split = read_csv(args.evidence / "08_traffic_split.csv")
    by_version = {row["version"]: row for row in latency}
    fid_by_version = {row["version"]: row for row in fid}
    bar_chart(args.output / "08_AB_P99.svg", "Stage 8：A/B 版本 P99 延迟", "候选 B 的尾延迟约为稳定版本 A 的 1.9 倍，应作为灰度门槛披露", ["A · PTQ INT8", "B · QAT INT8"], [float(by_version["A"]["p99_ms"]), float(by_version["B"]["p99_ms"])], "P99 延迟（ms）", 220.0, 1)
    bar_chart(args.output / "08_AB_FID.svg", "Stage 8：A/B 抽样 FID", "5000 张生成样本；独立于 Stage 3/5 的 canonical FID 表", ["A · PTQ INT8", "B · QAT INT8"], [float(fid_by_version["A"]["fid_sample"]), float(fid_by_version["B"]["fid_sample"])], "抽样 FID", 40.0, 2)
    split_chart(args.output / "08_AB_流量比例.svg", split)


if __name__ == "__main__":
    main()
