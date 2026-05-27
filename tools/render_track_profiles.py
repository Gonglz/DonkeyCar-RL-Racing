#!/usr/bin/env python3
"""Render DonkeyCar track-profile JSON files into a README-friendly PNG.

The script intentionally avoids notebooks and simulator dependencies. It reads
the calibrated centerline and boundary coordinates from `module/track_data` and
writes `assets/track-profiles.png`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
PROFILES = [
    (ROOT / "module" / "track_data" / "manual_width_waveshare.json", "Waveshare"),
    (ROOT / "module" / "track_data" / "manual_width_generated_track.json", "Generated Track"),
]


def _fit_transform(points: list[list[float]], x0: int, y0: int, width: int, height: int):
    xs = [float(p[0]) for p in points]
    zs = [float(p[1]) for p in points]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    scale = min(width / max(max_x - min_x, 1e-6), height / max(max_z - min_z, 1e-6)) * 0.92
    cx = (min_x + max_x) / 2.0
    cz = (min_z + max_z) / 2.0

    def project(point: list[float]) -> tuple[float, float]:
        return (
            x0 + width / 2.0 + (float(point[0]) - cx) * scale,
            y0 + height / 2.0 - (float(point[1]) - cz) * scale,
        )

    return project


def _draw_polyline(draw: ImageDraw.ImageDraw, points: Iterable[tuple[float, float]], color: tuple[int, int, int], width: int) -> None:
    mapped = list(points)
    if len(mapped) > 1:
        draw.line(mapped + [mapped[0]], fill=color, width=width, joint="curve")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    canvas_w, canvas_h = 1400, 700
    pad = 70
    panel_w = (canvas_w - pad * 3) // 2
    panel_h = canvas_h - pad * 2
    image = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(image)

    for i, (path, title) in enumerate(PROFILES):
        data = json.loads(path.read_text(encoding="utf-8"))
        outline = data["outline"]
        center = outline["fine_track_xz"]
        left = outline["left_boundary_xz"]
        right = outline["right_boundary_xz"]
        x0 = pad + i * (panel_w + pad)
        y0 = pad
        draw.rounded_rectangle(
            (x0 - 18, y0 - 18, x0 + panel_w + 18, y0 + panel_h + 18),
            radius=18,
            outline=(220, 225, 230),
            width=2,
            fill=(250, 252, 255),
        )
        project = _fit_transform(center + left + right, x0, y0, panel_w, panel_h)
        _draw_polyline(draw, (project(p) for p in left), (31, 110, 180), 4)
        _draw_polyline(draw, (project(p) for p in right), (31, 110, 180), 4)
        _draw_polyline(draw, (project(p) for p in center), (230, 126, 34), 3)
        draw.text((x0, y0 - 45), title, fill=(20, 30, 40))

        summary = data.get("manual_width_probe", {}).get("summary", {})
        total_width = summary.get("estimated_total_width_sim")
        width_label = f"{float(total_width):.3f} sim units" if total_width is not None else "not recorded"
        draw.text((x0, y0 + panel_h + 30), f"Profile: {path.name}; estimated width: {width_label}", fill=(70, 80, 90))

    image.save(ASSETS / "track-profiles.png", optimize=True)


if __name__ == "__main__":
    main()
