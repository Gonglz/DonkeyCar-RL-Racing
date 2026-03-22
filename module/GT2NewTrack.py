#!/usr/bin/env python3
"""
Generated-track to new-track line-color transfer.

Detects white edge lines and yellow center lines in generated_track images,
then recolors them to match new_track's color scheme:
  - white edge lines  -> blue  (BGR 94, 61, 48)
  - yellow center line -> orange (BGR 59, 97, 146)
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .green_vehicle_detect import GreenVehicleDetector

_green_detector = GreenVehicleDetector()


# ---------------------------------------------------------------------------
# Target colors sampled from new_track (BGR, top-20% saturation median)
# ---------------------------------------------------------------------------
TARGET_BLUE_BGR = np.array([94, 61, 48], dtype=np.uint8)    # edge line
TARGET_ORANGE_BGR = np.array([59, 97, 146], dtype=np.uint8)  # center line


# ---------------------------------------------------------------------------
# Shape-based denoising
# ---------------------------------------------------------------------------
def _denoise_shape(
    mask: np.ndarray,
    min_area: int = 80,
    min_elongated_area: int = 10,
    min_aspect: float = 2.0,
    skip_open: bool = False,
) -> np.ndarray:
    """Remove small non-elongated blobs from a binary mask.

    skip_open: if True, skip the morphological OPEN step which destroys
               thin (1-2px wide) lines.  Use for white-line detection.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    if not skip_open:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / (min(w, h) + 1e-5)
        if area >= min_area or (area >= min_elongated_area and aspect >= min_aspect):
            clean[labels == i] = 255
    return clean


# ---------------------------------------------------------------------------
# Road support mask
# ---------------------------------------------------------------------------
def detect_gt_road_support(img_bgr: np.ndarray) -> np.ndarray:
    """Estimate the drivable road neighborhood and exclude grass/off-road regions."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    base = (s <= 70) & (v >= 35) & (v <= 235)
    greenish = (h_ch >= 30) & (h_ch <= 95) & (s >= 45)
    base &= ~greenish

    road = (base.astype(np.uint8) * 255)
    road = cv2.morphologyEx(
        road, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    )
    road = cv2.morphologyEx(
        road, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    n, labels, stats, _ = cv2.connectedComponentsWithStats(road, connectivity=8)
    keep = np.zeros_like(road)
    h, w = road.shape
    min_area = max(150, int(0.01 * h * w))
    lower_y = int(0.55 * h)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        top = int(stats[i, cv2.CC_STAT_TOP])
        height = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        if top + height < lower_y and area < int(0.12 * h * w):
            continue
        keep[labels == i] = 255

    if not np.any(keep):
        keep = road

    support = cv2.dilate(
        keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2
    )
    support = cv2.morphologyEx(
        support, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    )
    return support


def _apply_road_support(mask: np.ndarray, road_support: Optional[np.ndarray]) -> np.ndarray:
    if road_support is None or not np.any(road_support):
        return mask
    return cv2.bitwise_and(mask, road_support)


# ---------------------------------------------------------------------------
# White-line local-contrast supplement (borrowed from RRL-style detection)
# ---------------------------------------------------------------------------
def _split_clusters(vals: np.ndarray, split_gap: int = 5):
    if len(vals) == 0:
        return []
    gaps = np.where(np.diff(vals) > split_gap)[0]
    if len(gaps) == 0:
        return [vals]
    return list(np.split(vals, gaps + 1))


def _row_multi_centroid_encode(
    mask: np.ndarray,
    min_band: int = 3,
    pad: int = 1,
    min_pixels: int = 1,
    max_width_ratio: float = 0.15,
    split_gap: int = 5,
) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    max_span = max(1, int(round(w * max_width_ratio)))
    min_half = max(1, min_band // 2)

    for row in range(h):
        cols = np.flatnonzero(mask[row])
        if len(cols) < min_pixels:
            continue
        clusters = _split_clusters(cols, split_gap)
        for cl in clusters:
            if len(cl) < min_pixels:
                continue
            span = cl[-1] - cl[0] + 1
            if span > max_span:
                continue
            center_x = int(np.round(cl.mean()))
            half = max(min_half, span // 2 + pad)
            left = max(0, center_x - half)
            right = min(w, center_x + half + 1)
            out[row, left:right] = True
    return out


def _vertical_continuity_filter(mask: np.ndarray, min_run: int = 4) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.where(mask, 255, 0).astype(np.uint8), connectivity=8
    )
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_HEIGHT]) >= min_run:
            out[labels == i] = True
    return out


def _recover_far_thin_segments(
    encoded_mask: np.ndarray,
    strict_mask: np.ndarray,
    h_img: int,
    w_img: int,
) -> np.ndarray:
    """Recover faint upper-frame segments without reopening broad near-field noise."""
    loose = _vertical_continuity_filter(encoded_mask, min_run=4)
    extra = loose & ~strict_mask
    if not np.any(extra):
        return extra

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.where(extra, 255, 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros_like(extra, dtype=bool)
    for i in range(1, n):
        top = int(stats[i, cv2.CC_STAT_TOP])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cy = top + hh // 2
        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        if cy > int(0.55 * h_img):
            continue
        if hh < 5 or area < 6:
            continue
        if ww > int(0.18 * w_img) and hh < 8:
            continue
        if area < 14 and aspect < 1.5:
            continue
        keep |= labels == i
    return keep


def _detect_horizontal_line_supplement(
    candidate_mask: np.ndarray,
    min_support_ratio: float = 0.62,
) -> np.ndarray:
    if not np.any(candidate_mask):
        return candidate_mask.copy()

    h, w = candidate_mask.shape
    cand_u8 = np.where(candidate_mask, 255, 0).astype(np.uint8)
    lines = cv2.HoughLinesP(
        cand_u8,
        1,
        np.pi / 180,
        threshold=10,
        minLineLength=max(16, int(0.16 * w)),
        maxLineGap=10,
    )
    if lines is None:
        return candidate_mask.copy() & False

    drawn = np.zeros_like(cand_u8)
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        if length < 0.18 * w:
            continue
        if angle > 18.0:
            continue
        if max(y1, y2) < 0.40 * h:
            continue
        line_mask = np.zeros_like(cand_u8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=5)
        support_ratio = float(
            np.count_nonzero((cand_u8 > 0) & (line_mask > 0))
        ) / float(max(1, np.count_nonzero(line_mask)))
        if support_ratio < min_support_ratio:
            continue
        cv2.line(drawn, (x1, y1), (x2, y2), 255, thickness=5)

    drawn = cv2.bitwise_and(drawn, cand_u8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(drawn, connectivity=8)
    keep = np.zeros_like(candidate_mask, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = ww / float(max(1, hh))
        if area < 60:
            continue
        if ww < max(18, int(0.12 * w)):
            continue
        if hh > int(0.18 * h):
            continue
        if aspect < 3.0:
            continue
        keep |= labels == i
    return keep


def _detect_white_line_local_contrast(
    img_bgr: np.ndarray,
    road_support: Optional[np.ndarray] = None,
) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = img_bgr.shape[:2]
    row_y = np.arange(h_img, dtype=np.int32)[:, None]
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1]

    local_mean = cv2.GaussianBlur(v, (0, 0), sigmaX=12, sigmaY=12)
    contrast = v - local_mean
    cand = (
        ((contrast > 32) & (s < 45))
        | ((contrast > 22) & (s < 28) & (v >= 145))
        | (((contrast > 18) & (s < 24) & (v >= 135)) & (row_y < int(0.52 * h_img)))
        | ((v >= 215) & (s < 20))
    ).astype(np.uint8) * 255
    cand = _apply_road_support(cand, road_support)
    cand = cv2.morphologyEx(
        cand,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    encoded = _row_multi_centroid_encode(
        cand > 0,
        min_band=3,
        pad=1,
        min_pixels=1,
        max_width_ratio=0.18,
        split_gap=6,
    )
    filtered = _vertical_continuity_filter(encoded, min_run=5)
    far_thin = _recover_far_thin_segments(encoded, filtered, h_img, w_img)
    horizontal = _detect_horizontal_line_supplement(cand > 0)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        np.where(filtered, 255, 0).astype(np.uint8), connectivity=8
    )
    clean = np.zeros_like(cand)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        cy = int(stats[i, cv2.CC_STAT_TOP]) + int(stats[i, cv2.CC_STAT_HEIGHT]) // 2
        ch_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        keep = False
        if cy < h_img * 0.45:
            # Perspective-compressed far white lines can be only 4-8px tall.
            keep = (ch_i >= 4 and ww >= max(12, int(0.09 * w_img)) and area >= 20)
        else:
            min_h = 12 if cy > h_img * 0.55 else 10
            keep = ch_i >= min_h
        if keep:
            clean[labels == i] = 255
    if np.any(far_thin):
        clean[far_thin] = 255
    if np.any(horizontal):
        clean[horizontal] = 255
    return clean


# ---------------------------------------------------------------------------
# White line detector  (HSV: S<30, V>200)
# ---------------------------------------------------------------------------
def detect_white_line(
    img_bgr: np.ndarray, road_support: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return binary mask of white edge lines."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    mask = _apply_road_support(mask, road_support)
    contrast_mask = _detect_white_line_local_contrast(img_bgr, road_support=road_support)
    merged = cv2.bitwise_or(mask, contrast_mask)
    return _denoise_shape(merged, min_area=40, min_elongated_area=10, min_aspect=2.0, skip_open=True)


# ---------------------------------------------------------------------------
# Yellow line detector  (LAB: b>140, L>50, R>G)
# ---------------------------------------------------------------------------
def detect_yellow_line(
    img_bgr: np.ndarray, road_support: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return binary mask of yellow center line."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    R = img_bgr[:, :, 2].astype(np.float32)
    G = img_bgr[:, :, 1].astype(np.float32)
    lab_b_mask = ((lab[:, :, 2] > 140) & (lab[:, :, 0] > 50)).astype(np.uint8) * 255
    rg_mask = (R > G).astype(np.uint8) * 255
    hsv_yellow = cv2.inRange(hsv, (10, 45, 70), (45, 255, 255))
    mask = cv2.bitwise_and(cv2.bitwise_and(lab_b_mask, rg_mask), hsv_yellow)
    mask = _apply_road_support(mask, road_support)
    return _denoise_shape(mask)


# ---------------------------------------------------------------------------
# Recolor: replace detected line pixels with target colors
# ---------------------------------------------------------------------------
def recolor_lines(
    img_bgr: np.ndarray,
    white_mask: np.ndarray,
    yellow_mask: np.ndarray,
    blue_bgr: np.ndarray = TARGET_BLUE_BGR,
    orange_bgr: np.ndarray = TARGET_ORANGE_BGR,
) -> np.ndarray:
    """Return a copy of img_bgr with line pixels replaced by target colors."""
    out = img_bgr.copy()
    out[white_mask > 0] = blue_bgr
    out[yellow_mask > 0] = orange_bgr
    return out


# ---------------------------------------------------------------------------
# NT-style road surface rendering
# ---------------------------------------------------------------------------
_CLOTH_HSV = np.array([128.0, 24.0, 172.0], dtype=np.float32)
_BG_HSV = np.array([125.0, 20.0, 120.0], dtype=np.float32)


def render_road_surface(img_bgr: np.ndarray) -> np.ndarray:
    """Transform road/bg to NT cloth texture. Lines are NOT modified."""
    src_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    src_v = src_hsv[:, :, 2]
    h_img, w_img = img_bgr.shape[:2]

    road_support = detect_gt_road_support(img_bgr)
    white_mask = detect_white_line(img_bgr, road_support) > 0
    yellow_mask = detect_yellow_line(img_bgr, road_support) > 0
    line_mask = white_mask | yellow_mask

    # Road = road_support minus lines
    road_region = (road_support > 0) & ~line_mask
    bg_region = (road_support == 0) & ~line_mask

    if not np.any(road_region):
        return img_bgr

    sigma = max(3.0, 0.10 * float(min(h_img, w_img)))
    shade = cv2.GaussianBlur(src_v, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hf = src_v - shade
    road_ref = float(np.mean(shade[road_region]))

    out_hsv = src_hsv.copy()
    out_hsv[:, :, 0][road_region] = float(_CLOTH_HSV[0])
    out_hsv[:, :, 1][road_region] = np.clip(
        float(_CLOTH_HSV[1]) + 0.22 * 0.3 * hf[road_region], 0.0, 60.0)
    out_hsv[:, :, 2][road_region] = np.clip(
        float(_CLOTH_HSV[2]) + 0.40 * (shade[road_region] - road_ref)
        + 0.22 * hf[road_region], 0.0, 255.0)

    if np.any(bg_region):
        bg_ref = float(np.mean(shade[bg_region]))
        out_hsv[:, :, 0][bg_region] = float(_BG_HSV[0])
        out_hsv[:, :, 1][bg_region] = np.clip(
            float(_BG_HSV[1]) + 0.18 * 0.3 * hf[bg_region], 0.0, 60.0)
        out_hsv[:, :, 2][bg_region] = np.clip(
            float(_BG_HSV[2]) + 0.25 * (shade[bg_region] - bg_ref)
            + 0.18 * hf[bg_region], 0.0, 255.0)

    return cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_directory(
    raw_dir: str,
    out_dir: str,
    save_masks: bool = False,
) -> None:
    """Detect lines and recolor for all images in raw_dir."""
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_masks:
        (out_path / "white_mask").mkdir(exist_ok=True)
        (out_path / "yellow_mask").mkdir(exist_ok=True)

    files = sorted(raw_path.glob("*.png"))
    if not files:
        files = sorted(raw_path.glob("*.jpg"))
    print(f"Processing {len(files)} images from {raw_dir}")

    for i, fpath in enumerate(files):
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            continue

        road_support = detect_gt_road_support(img)
        road_img = render_road_surface(img)
        white = detect_white_line(img, road_support=road_support)
        yellow = detect_yellow_line(img, road_support=road_support)
        result = recolor_lines(road_img, white, yellow)

        # 保留绿车原始像素，不被道路渲染覆盖
        green = _green_detector.detect(img)
        if green.detected:
            result[green.mask > 0] = img[green.mask > 0]

        cv2.imwrite(str(out_path / fpath.name), result)
        if save_masks:
            cv2.imwrite(str(out_path / "white_mask" / fpath.name), white)
            cv2.imwrite(str(out_path / "yellow_mask" / fpath.name), yellow)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} done")

    print(f"All done. Output: {out_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recolor generated_track lines to match new_track colors"
    )
    parser.add_argument(
        "--raw_dir",
        default="data/scene_samples/generated_track/raw",
        help="Input directory with raw generated_track images",
    )
    parser.add_argument(
        "--out_dir",
        default="data/scene_samples/generated_track/GT2NewTrack",
        help="Output directory for recolored images",
    )
    parser.add_argument(
        "--save_masks",
        action="store_true",
        help="Also save white/yellow mask sub-directories",
    )
    args = parser.parse_args()
    process_directory(args.raw_dir, args.out_dir, args.save_masks)


if __name__ == "__main__":
    main()
