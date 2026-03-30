#!/usr/bin/env python3
"""
Waveshare to newtrack transfer helpers.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .green_vehicle_detect import GreenVehicleDetector

_green_detector = GreenVehicleDetector()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CLS_ORDER = ["blue", "yellow", "white"]
CLS_ORDER = _CLS_ORDER

# Match the sampled newtrack targets from module/GT2NewTrack.py instead of a
# brighter canonical blue/orange.
_CANON_BLUE_HSV = np.array([112.0, 125.0, 94.0], dtype=np.float32)
_CANON_CENTER_HSV = np.array([13.0, 152.0, 146.0], dtype=np.float32)
_WS_CENTER_BLEND = (0.95, 0.78, 0.80)
_WS_CENTER_V_BOOST = 2.0

_WS_YELLOW_BASE: Dict[str, float] = {
    "lane_min_area": 8, "lane_min_height": 3, "edge_half_width": 1,
    "edge_close_h": 5, "edge_close_w": 3, "final_close_k": 3,
    "final_dilate_iter": 0, "top_min_area": 3, "main_min_area": 8,
    "blue_alpha_h": 0.96, "blue_alpha_s": 0.88, "blue_alpha_v": 0.62,
}
WS_YELLOW_PRESETS: Dict[str, Dict[str, float]] = {
    "case01": {**_WS_YELLOW_BASE},
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _natural_key(p: Path) -> Tuple[int, str]:
    m = re.search(r"(\d+)", p.stem)
    return (int(m.group(1)) if m else 10**12, p.name)


def list_images(path_str: str) -> List[Path]:
    p = Path(path_str)
    for sub in ("processed", "images"):
        if (p / sub).exists():
            p = p / sub
            break
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        files.extend(p.glob(ext))
    return sorted(files, key=_natural_key)


def load_images(paths: List[Path]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is not None:
            out.append(img)
    return out


def sample_paths(paths: List[Path], max_images: int) -> List[Path]:
    if len(paths) <= max_images:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_images, dtype=np.int32)
    return [paths[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _u8(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _k(kh: int, kw: Optional[int] = None) -> np.ndarray:
    return np.ones((max(1, int(kh)), max(1, int(kw or kh))), np.uint8)


def _ws_green_text_mask(hsv: np.ndarray) -> np.ndarray:
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    return (h_ch >= 38) & (h_ch <= 95) & (s_ch >= 70) & (v_ch >= 45)


def _filter_centerline_components(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    seed_d = cv2.dilate(_u8(seed), _k(3), iterations=1) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(_u8(mask), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        comp = labels == i
        if np.any(comp & seed_d):
            if area >= 3:
                out |= comp
        elif area >= 8:
            out |= comp
    return out


# ---------------------------------------------------------------------------
# Semantic masks & road support
# ---------------------------------------------------------------------------
def semantic_masks(hsv: np.ndarray) -> Dict[str, np.ndarray]:
    blue = cv2.inRange(hsv, np.array([90, 40, 40]), np.array([140, 255, 255])) > 0
    yellow = cv2.inRange(hsv, np.array([12, 55, 60]), np.array([45, 255, 255])) > 0
    white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([179, 70, 255])) > 0
    yellow &= ~blue
    white &= ~(blue | yellow)
    return {"blue": blue, "yellow": yellow, "white": white}


def road_support_mask(hsv: np.ndarray) -> np.ndarray:
    s, v, h = hsv[:, :, 1], hsv[:, :, 2], hsv[:, :, 0]
    base = (s <= 105) & (v >= 35) & ~((h >= 35) & (h <= 95) & (s >= 80))
    base = cv2.morphologyEx(base.astype(np.uint8), cv2.MORPH_OPEN, _k(3))
    return cv2.dilate(base, _k(5), iterations=1) > 0


def overlay_semantics(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    masks = semantic_masks(hsv)
    out = img_bgr.copy()
    colors = {"blue": (255, 80, 0), "yellow": (0, 220, 255), "white": (255, 255, 255)}
    for cls in _CLS_ORDER:
        m = masks[cls]
        if not np.any(m):
            continue
        c = np.zeros_like(out); c[:, :] = colors[cls]
        out[m] = cv2.addWeighted(out, 0.55, c, 0.45, 0.0)[m]
    return out


def compute_stats(images: List[np.ndarray]) -> Dict[str, object]:
    if not images:
        raise ValueError("No images to analyze")
    ratios = {c: [] for c in _CLS_ORDER}
    proto_sum = {c: np.zeros(3, dtype=np.float64) for c in _CLS_ORDER}
    proto_cnt = {c: 0 for c in _CLS_ORDER}
    center_sum, center_cnt = np.zeros(3, dtype=np.float64), 0

    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        total = float(h * w)
        masks = semantic_masks(hsv)
        for cls in _CLS_ORDER:
            m = masks[cls]
            ratios[cls].append(float(np.count_nonzero(m) / total))
            cnt = int(np.count_nonzero(m))
            if cnt > 0:
                proto_sum[cls] += hsv[m].astype(np.float64).sum(axis=0)
                proto_cnt[cls] += cnt
        hch, sch, vch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        warm = ((hch <= 30) | (hch >= 150)) & (sch >= 40) & (vch >= 55)
        if np.any(masks["white"]):
            wd = cv2.distanceTransform(masks["white"].astype(np.uint8) * 255, cv2.DIST_L2, 3)
            wl = masks["white"] & (wd <= 2.6)
        else:
            wl = masks["white"]
        cand = wl | masks["yellow"] | warm
        cx = 0.5 * (w - 1)
        for y in range(int(0.25 * h), h):
            xs = np.flatnonzero(cand[y])
            if xs.size == 0:
                continue
            x = int(xs[int(np.argmin(np.abs(xs - cx)))])
            if abs(x - cx) <= max(3.0, 0.23 * w):
                center_sum += hsv[y, x].astype(np.float64)
                center_cnt += 1

    protos: Dict[str, Optional[List[float]]] = {}
    for cls in _CLS_ORDER:
        protos[cls] = (proto_sum[cls] / float(proto_cnt[cls])).tolist() if proto_cnt[cls] >= 1000 else None
    return {
        "ratios": {c: float(np.mean(v)) for c, v in ratios.items()},
        "prototypes_hsv": protos,
        "centerline_hsv": (center_sum / float(center_cnt)).tolist() if center_cnt >= 80 else None,
    }


# ---------------------------------------------------------------------------
# Tophat-based white dash extraction (improved from white_dash_extractor v5)
# ---------------------------------------------------------------------------
def _multiscale_tophat(gray: np.ndarray,
                       kernel_sizes: List[Tuple[int, int]] = [(9, 9), (15, 15), (21, 21)]
                       ) -> np.ndarray:
    """Multi-scale White Top-Hat: pixel-wise max across different kernel sizes."""
    result = np.zeros_like(gray, dtype=np.float32)
    for ksize in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        result = np.maximum(result, tophat.astype(np.float32))
    return result.astype(np.uint8)


def _filter_dash_shape(mask: np.ndarray) -> np.ndarray:
    """Shape filter: perspective-aware size/aspect + solidity/extent checks."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h_img = mask.shape[0]
    mid_y = h_img * 0.5
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]
        cy = stats[i, cv2.CC_STAT_TOP] + hh // 2
        if cy >= mid_y:
            if area < 2 or area > 500 or ww > 28:
                continue
            if hh > 0 and ww / max(hh, 1) > 6:
                continue
            if ww > 0 and hh / max(ww, 1) > 10 and area > 30:
                continue
        else:
            if area < 2 or area > 200 or ww > 16:
                continue
            if hh > 0 and ww / max(hh, 1) > 5:
                continue
            if ww > 0 and hh / max(ww, 1) > 8 and area > 20:
                continue
        if area >= 6:
            comp_mask = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            cnt = contours[0]
            if area < 30:
                min_sol, min_ext = 0.35, 0.25
            elif area < 80:
                min_sol, min_ext = 0.50, 0.35
            else:
                min_sol, min_ext = 0.60, 0.40
            hull_area = cv2.contourArea(cv2.convexHull(cnt))
            if hull_area > 0 and area / hull_area < min_sol:
                continue
            if len(cnt) >= 5:
                rect = cv2.minAreaRect(cnt)
                rect_area = rect[1][0] * rect[1][1]
                if rect_area > 0 and area / rect_area < min_ext:
                    continue
        out[labels == i] = 255
    return out


def _check_dark_neighborhood(mask: np.ndarray, gray: np.ndarray, radius: int = 6) -> np.ndarray:
    """Reject blobs surrounded by too many dark pixels (shelf/furniture edges)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    h_img, w_img = gray.shape[:2]
    dark_pixels = (gray < 65).astype(np.uint8)
    for i in range(1, n):
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        ww, hh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x1, y1 = max(0, x - radius), max(0, y - radius)
        x2, y2 = min(w_img, x + ww + radius), min(h_img, y + hh + radius)
        comp = (labels[y1:y2, x1:x2] == i)
        surround_total = (y2 - y1) * (x2 - x1) - np.sum(comp)
        if surround_total <= 0:
            out[labels == i] = 255
            continue
        surround_dark = dark_pixels[y1:y2, x1:x2].copy()
        surround_dark[comp] = 0
        if np.sum(surround_dark) / surround_total < 0.25:
            out[labels == i] = 255
    return out


def _extract_white_dashes_tophat(
    bgr: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
    return_confidence: bool = False,
) -> np.ndarray:
    """
    Improved white dash extraction using multi-scale Top-Hat morphology.
    Returns a binary mask (uint8, 0/255) of detected white dash pixels.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = gray.shape[:2]

    # Road mask: use provided or build from bottom-seed growth
    if road_mask is not None:
        road = _u8(road_mask)
        road = cv2.dilate(cv2.morphologyEx(road, cv2.MORPH_CLOSE, _k(5)), _k(5), iterations=1)
    else:
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        gray_road = ((s_ch <= 40) & (v_ch >= 150) & (v_ch <= 220)).astype(np.uint8)
        white_line = ((s_ch <= 35) & (v_ch >= 210)).astype(np.uint8)
        yellow_line = ((h_ch >= 10) & (h_ch <= 45) & (s_ch >= 50) & (v_ch >= 130)).astype(np.uint8)
        brown_edge = ((h_ch >= 0) & (h_ch <= 15) & (s_ch >= 60) & (v_ch >= 80) & (v_ch <= 180)).astype(np.uint8)
        road_cand = (gray_road | white_line | yellow_line | brown_edge).astype(np.uint8) * 255
        green = ((h_ch >= 35) & (h_ch <= 85) & (s_ch >= 60)).astype(np.uint8) * 255
        road_cand = cv2.bitwise_and(road_cand, cv2.bitwise_not(green))
        road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_CLOSE, _k(7))
        road_cand = cv2.morphologyEx(road_cand, cv2.MORPH_OPEN, _k(3))
        seed_rows = max(2, int(h_img * 0.15))
        n_cc, lbl_cc, st_cc, _ = cv2.connectedComponentsWithStats(road_cand, connectivity=8)
        if n_cc > 1:
            bottom_labels = lbl_cc[h_img - seed_rows:, :]
            seed_labels = set(np.unique(bottom_labels)) - {0}
            road = np.zeros_like(road_cand)
            for lbl in seed_labels:
                road[lbl_cc == lbl] = 255
            road = cv2.dilate(road, _k(3), iterations=2)
        else:
            road = road_cand

    # Multi-scale Top-Hat
    tophat = _multiscale_tophat(gray)
    _, mask = cv2.threshold(tophat, 20, 255, cv2.THRESH_BINARY)

    # Two-tier HSV white confirmation
    mask_white_strict = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([179, 60, 255]))
    mask_white_loose = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([179, 60, 255]))
    strong_th = (tophat > 30).astype(np.uint8) * 255
    weak_th = cv2.bitwise_and(mask, cv2.bitwise_not(strong_th))
    mask = cv2.bitwise_or(
        cv2.bitwise_and(strong_th, cv2.bitwise_and(mask, mask_white_loose)),
        cv2.bitwise_and(weak_th, mask_white_strict),
    )

    # Road constraint
    mask = cv2.bitwise_and(mask, road)

    # Exclude yellow & high saturation
    yellow_exc = cv2.inRange(hsv, np.array([10, 60, 50]), np.array([45, 255, 255]))
    yellow_exc = cv2.dilate(yellow_exc, _k(3), iterations=2)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(yellow_exc))
    high_sat = cv2.inRange(hsv, np.array([0, 80, 0]), np.array([179, 255, 255]))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(high_sat))

    # Shape filter + neighborhood check
    mask = _filter_dash_shape(mask)
    mask = _check_dark_neighborhood(mask, gray)

    # Tophat-aware brightness verification with position awareness
    v_ch = hsv[:, :, 2]
    s_ch = hsv[:, :, 1]
    upper_y = int(h_img * 0.35)
    very_top_y = int(h_img * 0.28)   # wall / map-edge zone — only very obvious white passes
    side_band = int(w_img * 0.18)    # left/right side columns: wall/shelf noise zone
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    conf = np.zeros_like(mask, dtype=np.float32)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        bx = stats[i, cv2.CC_STAT_LEFT]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        cx = bx + bw // 2
        cy = stats[i, cv2.CC_STAT_TOP] + bh // 2
        comp = (labels == i)
        mean_v = float(np.mean(v_ch[comp]))
        mean_s = float(np.mean(s_ch[comp]))
        mean_tophat = float(np.mean(tophat[comp]))
        # Top 28%: shelves / furniture / wall edges — reject unless very large & very bright
        if cy < very_top_y and (area < 40 or mean_tophat < 38 or mean_s > 25):
            continue
        # Upper 40% + side bands: furniture/wall corner noise
        if cy < int(h_img * 0.40) and (cx < side_band or cx > w_img - side_band):
            if area < 50 or mean_tophat < 35:
                continue
        touches_edge = (bx == 0 or bx + bw >= w_img)
        if touches_edge and area > 20 and (mean_v < 200 or mean_s > 30):
            continue
        v_boost = 30 if cy < upper_y else 0
        in_upper = cy < upper_y
        keep_comp = False
        if area < 8:
            if (mean_v >= 195 and mean_s <= 30) or \
               (mean_tophat >= 30 and mean_v >= 140 + v_boost and mean_s <= 35):
                keep_comp = True
        elif area <= 50:
            if (mean_tophat >= 30 and mean_v >= 145 + v_boost and mean_s <= 40) or \
               mean_v >= 190 + v_boost:
                keep_comp = True
        else:
            if (mean_tophat >= 30 and mean_v >= 180 + v_boost and mean_s <= 35) or \
               (mean_v >= 200 + v_boost and mean_s <= 40):
                keep_comp = True

        if not keep_comp:
            continue

        out[comp] = 255
        th_n = float(np.clip((mean_tophat - 20.0) / 20.0, 0.0, 1.0))
        v_ref = float(140.0 + (20.0 if in_upper else 0.0))
        v_n = float(np.clip((mean_v - v_ref) / 80.0, 0.0, 1.0))
        s_n = float(np.clip((48.0 - mean_s) / 48.0, 0.0, 1.0))
        area_n = float(np.clip(np.log1p(float(area)) / np.log1p(80.0), 0.0, 1.0))
        score = 0.45 * th_n + 0.35 * v_n + 0.15 * s_n + 0.05 * area_n
        if touches_edge:
            score *= 0.85
        if in_upper:
            score *= 0.92
        conf[comp] = np.maximum(conf[comp], np.float32(np.clip(score, 0.0, 1.0)))

    if return_confidence:
        conf = cv2.GaussianBlur(conf, (3, 3), sigmaX=0.8, sigmaY=0.8)
        conf = np.clip(conf * (out > 0).astype(np.float32), 0.0, 1.0).astype(np.float32)
        return out, conf
    return out


# ---------------------------------------------------------------------------
# Line extraction primitives
# ---------------------------------------------------------------------------
def _extract_line_mask(
    line_mask: np.ndarray, mode: str = "center",
    min_y_ratio: float = 0.25, offset_ratio: float = 0.23, half_width: int = 2,
) -> np.ndarray:
    h, w = line_mask.shape
    cx = 0.5 * (w - 1)
    y0 = int(h * min_y_ratio)
    thresh = max(3.0, offset_ratio * w)
    out = np.zeros_like(line_mask, dtype=np.uint8)
    for y in range(y0, h):
        xs = np.flatnonzero(line_mask[y])
        if xs.size == 0:
            continue
        if mode == "center":
            x = int(xs[int(np.argmin(np.abs(xs - cx)))])
            if abs(x - cx) > thresh:
                continue
        else:
            x = int(xs[-1] if mode == "right" else xs[0])
            if abs(x - cx) < thresh:
                continue
        out[y, max(0, x - half_width):min(w, x + half_width + 1)] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((5, 3), np.uint8))
    return out.astype(bool) & line_mask


def _split_row_clusters(cols: np.ndarray, gap: int) -> List[np.ndarray]:
    if len(cols) == 0:
        return []
    gaps = np.where(np.diff(cols) > gap)[0]
    return [cols] if len(gaps) == 0 else list(np.split(cols, gaps + 1))


def _row_centroid_encode(mask: np.ndarray, band_width: int, min_pixels: int,
                         max_width_ratio: float, split_gap: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    half = max(1, band_width // 2)
    max_span = max(1, int(round(w * max_width_ratio)))
    for row in range(h):
        cols = np.flatnonzero(mask[row])
        if len(cols) < min_pixels:
            continue
        viable = [c for c in _split_row_clusters(cols, split_gap)
                  if len(c) >= min_pixels and (c[-1] - c[0] + 1) <= max_span]
        if not viable:
            if (cols[-1] - cols[0] + 1) <= max_span:
                viable = [cols]
            else:
                continue
        cx = int(np.round(max(viable, key=len).mean()))
        out[row, max(0, cx - half):min(w, cx + half + 1)] = True
    return out


def _center_band_from_yellow(yellow_mask: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    if not np.any(road_mask):
        return road_mask.copy()
    h, w = road_mask.shape
    cx = 0.5 * (w - 1)
    y0 = int(0.18 * h)
    band = np.zeros_like(road_mask, dtype=bool)
    prev_mid: Optional[float] = None
    max_mid_jump = max(3.0, 0.07 * w)
    for y in range(y0, h):
        y_ratio = float(y) / max(1.0, float(h - 1))
        xs = np.flatnonzero(yellow_mask[y] & road_mask[y])
        if xs.size >= 2:
            left = float(xs[0])
            right = float(xs[-1])
            lane_w = max(6.0, right - left)
            mid = 0.5 * (left + right)
            half_w = max(5.0, (0.14 + 0.12 * y_ratio) * lane_w)
        elif xs.size == 1:
            x0 = float(xs[0])
            mid = x0 + 0.28 * w if x0 < cx else x0 - 0.28 * w
            half_w = (0.11 + 0.06 * y_ratio) * w
        else:
            mid = cx
            half_w = (0.12 + 0.07 * y_ratio) * w
        if prev_mid is not None:
            mid = float(np.clip(mid, prev_mid - max_mid_jump, prev_mid + max_mid_jump))
            mid = 0.65 * prev_mid + 0.35 * mid
        prev_mid = mid
        x0 = max(0, int(round(mid - half_w)))
        x1 = min(w, int(round(mid + half_w)) + 1)
        band[y, x0:x1] = True
    return band & road_mask


def _center_band_from_seed(seed_mask: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    if not np.any(seed_mask) or not np.any(road_mask):
        return road_mask.copy()
    h, w = seed_mask.shape
    y0 = int(0.18 * h)
    band = np.zeros_like(seed_mask, dtype=bool)
    prev_mid: Optional[float] = None
    for y in range(y0, h):
        xs = np.flatnonzero(seed_mask[y] & road_mask[y])
        if xs.size > 0:
            cur_mid = float(xs.mean())
            prev_mid = cur_mid if prev_mid is None else (0.7 * prev_mid + 0.3 * cur_mid)
        elif prev_mid is None:
            continue
        y_ratio = float(y) / max(1.0, float(h - 1))
        half_w = int(round(5.0 + 5.0 * y_ratio))
        x0 = max(0, int(round(prev_mid - half_w)))
        x1 = min(w, int(round(prev_mid + half_w)) + 1)
        band[y, x0:x1] = True
    return band & road_mask


def _white_center_support_band(
    seed_mask: np.ndarray,
    road_mask: np.ndarray,
    yellow_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    # Prefer yellow-guided center geometry, but relax the lower rows with a
    # seed-guided fallback so near white blocks are not rejected too early.
    seed_band = _center_band_from_seed(seed_mask, road_mask)
    if yellow_mask is None or not np.any(yellow_mask):
        return seed_band
    yellow_band = _center_band_from_yellow(yellow_mask, road_mask)
    out = yellow_band.copy()
    h = out.shape[0]
    near_y0 = int(0.55 * h)
    out[near_y0:] |= seed_band[near_y0:]
    # If yellow guidance misses the tracked center path entirely on a row, fall
    # back to the seed-derived band instead of dropping the near white block.
    for y in range(h):
        if np.any(seed_mask[y]) and not np.any(out[y] & seed_mask[y]):
            out[y] |= seed_band[y]
    return out & road_mask


def _expand_white_seed_to_center_blocks(
    white_clean: np.ndarray,
    white_soft: np.ndarray,
    seed: np.ndarray,
    road_mask: np.ndarray,
    yellow_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not np.any(seed):
        return seed.copy()

    support_band = _white_center_support_band(seed, road_mask, yellow_mask)
    cand = white_clean & road_mask & support_band

    seed_d = cv2.dilate(_u8(seed), _k(3), iterations=1) > 0
    yellow_near = (
        cv2.dilate(_u8(yellow_mask), _k(3), iterations=1) > 0
        if yellow_mask is not None and np.any(yellow_mask)
        else np.zeros_like(cand, dtype=bool)
    )
    h, w = cand.shape
    max_w = max(14, int(round(0.18 * w)))
    max_h = max(24, int(round(0.60 * h)))
    max_area = max(180, int(round(0.06 * h * w)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(_u8(cand), connectivity=8)
    out = np.zeros_like(cand, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 2 or area > max_area or ww > max_w or hh > max_h:
            continue
        if ww >= 12 and ww > int(2.8 * max(1, hh)):
            continue
        comp = labels == i
        if not np.any(comp & seed_d):
            continue
        if np.any(yellow_near) and np.count_nonzero(comp & yellow_near) > 0.42 * max(1, area):
            continue
        out |= comp

    if not np.any(out):
        out = seed.copy()
    else:
        out |= seed

    grow = white_soft & road_mask & support_band & (cv2.dilate(_u8(out), _k(3), iterations=1) > 0)
    out |= grow
    out = cv2.morphologyEx(_u8(out), cv2.MORPH_CLOSE, _k(3)) > 0
    return _clip_center_fill_width(out, seed, road_mask)


def _clip_center_fill_width(mask: np.ndarray, seed: np.ndarray, road_mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    prev_c: Optional[float] = None
    for y in range(h):
        xs_mask = np.flatnonzero(mask[y] & road_mask[y])
        if xs_mask.size == 0:
            continue
        xs_seed = np.flatnonzero(seed[y])
        if xs_seed.size > 0:
            cur_c = float(xs_seed.mean())
            prev_c = cur_c if prev_c is None else (0.65 * prev_c + 0.35 * cur_c)
        elif prev_c is None:
            prev_c = float(xs_mask.mean())
        y_ratio = float(y) / max(1.0, float(h - 1))
        half_w = int(round(3.0 + 3.0 * y_ratio))
        x0 = max(0, int(round(prev_c - half_w)))
        x1 = min(w, int(round(prev_c + half_w)) + 1)
        out[y, x0:x1] = mask[y, x0:x1]
    out |= seed
    out = cv2.morphologyEx(_u8(out), cv2.MORPH_CLOSE, _k(3)) > 0
    return out & road_mask


def _add_center_white_edge_ring(
    mask: np.ndarray,
    white_soft: np.ndarray,
    road_mask: np.ndarray,
    yellow_mask: Optional[np.ndarray] = None,
    anchor_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    ring = (cv2.dilate(_u8(mask), _k(3), iterations=1) > 0) & ~mask
    ring &= white_soft & road_mask
    ring &= _white_center_support_band(mask if anchor_mask is None else anchor_mask, road_mask, yellow_mask)
    if not np.any(ring):
        return mask.copy()
    merged = cv2.morphologyEx(_u8(mask | ring), cv2.MORPH_CLOSE, _k(3)) > 0
    # Use the pre-ring mask as geometric anchor so edge fill only recovers thin
    # anti-aliased boundaries instead of creating broad near-field blocks.
    return _clip_center_fill_width(merged, mask, road_mask)


# ---------------------------------------------------------------------------
# Proto / color blending
# ---------------------------------------------------------------------------
def _get_proto(tgt_stats: Dict, cls: str, fallback: np.ndarray) -> np.ndarray:
    p = tgt_stats.get("prototypes_hsv", {}).get(cls)
    return np.asarray(p, dtype=np.float32) if p is not None else fallback.copy()


def _get_center_proto(tgt_stats: Dict) -> np.ndarray:
    p = tgt_stats.get("centerline_hsv")
    if p is not None:
        return np.asarray(p, dtype=np.float32)
    p = tgt_stats.get("prototypes_hsv", {}).get("yellow")
    return np.asarray(p, dtype=np.float32) if p is not None else _CANON_CENTER_HSV.copy()


def _sanitize_blue(proto: np.ndarray) -> np.ndarray:
    out = proto.copy()
    if not (90.0 <= out[0] <= 145.0):
        out[0] = float(_CANON_BLUE_HSV[0])
    out[1] = np.clip(out[1], 95.0, 145.0)
    out[2] = np.clip(out[2], 88.0, 138.0)
    return out


def _resolve_tgt_blue(tgt_stats: Dict) -> np.ndarray:
    raw = _get_proto(tgt_stats, "blue", _CANON_BLUE_HSV)
    return _sanitize_blue(0.55 * raw + 0.45 * _CANON_BLUE_HSV)


def _sanitize_center(proto: np.ndarray) -> np.ndarray:
    out = proto.copy()
    is_warm = (out[0] <= 45.0) or (out[0] >= 150.0)
    if (not is_warm) or out[1] < 60.0:
        return _CANON_CENTER_HSV.copy()
    out[1] = np.clip(out[1], 118.0, 172.0)
    out[2] = np.clip(out[2], 135.0, 176.0)
    return out


def _resolve_tgt_center(tgt_stats: Dict) -> np.ndarray:
    raw = _get_center_proto(tgt_stats)
    if (90.0 <= raw[0] <= 145.0) or raw[1] < 70.0:
        return _CANON_CENTER_HSV.copy()
    return _sanitize_center(0.60 * raw + 0.40 * _CANON_CENTER_HSV)


def _blend_to_proto(hsv: np.ndarray, mask: np.ndarray, proto: np.ndarray,
                    ah: float, a_s: float, av: float) -> None:
    if not np.any(mask):
        return
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    th, ts, tv = float(proto[0]), float(proto[1]), float(proto[2])
    h_old = h[mask]
    d = th - h_old
    d = np.where(d > 90, d - 180, d)
    d = np.where(d < -90, d + 180, d)
    h[mask] = np.mod(h_old + ah * d, 180.0)
    s[mask] = np.clip(s[mask] * (1 - a_s) + ts * a_s, 0, 255)
    v[mask] = np.clip(v[mask] * (1 - av) + tv * av, 0, 255)


def _apply_newtrack_blend(src_hsv: np.ndarray, line_mask: np.ndarray, center_mask: np.ndarray,
                          tgt_stats: Dict, line_a: Tuple[float, float, float],
                          center_a: Tuple[float, float, float], v_boost: float) -> np.ndarray:
    hsv = src_hsv.astype(np.float32)
    _blend_to_proto(hsv, line_mask, _resolve_tgt_blue(tgt_stats), *line_a)
    _blend_to_proto(hsv, center_mask, _resolve_tgt_center(tgt_stats), *center_a)
    if np.any(center_mask):
        hsv[:, :, 2][center_mask] = np.clip(hsv[:, :, 2][center_mask] + v_boost, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Waveshare pipeline
# ---------------------------------------------------------------------------
def extract_ws_yellow_mask(
    src_hsv: np.ndarray,
    road_mask: np.ndarray,
    preset_name: str = "case01",
    return_confidence: bool = False,
) -> Union[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
]:
    # Keep the parameter for backward compatibility, but only case01 is active.
    preset = WS_YELLOW_PRESETS["case01"]
    masks = semantic_masks(src_hsv)
    yellow_raw = masks["yellow"] & ~_ws_green_text_mask(src_hsv)
    yellow_on_road = yellow_raw & road_mask
    if not np.any(yellow_raw):
        empty = yellow_raw.copy()
        if return_confidence:
            return empty, empty, yellow_on_road, np.zeros_like(src_hsv[:, :, 0], dtype=np.float32)
        return empty, empty, yellow_on_road

    h, w = yellow_on_road.shape
    road_relaxed = road_mask | (cv2.dilate(_u8(yellow_raw), _k(3), iterations=1) > 0)
    yellow_work = yellow_raw & road_relaxed

    n, labels, stats, _ = cv2.connectedComponentsWithStats(_u8(yellow_work), connectivity=8)
    yellow_lane = np.zeros_like(yellow_work, dtype=bool)
    for i in range(1, n):
        y, ww, hh, area = (int(stats[i, cv2.CC_STAT_TOP]), int(stats[i, cv2.CC_STAT_WIDTH]),
                           int(stats[i, cv2.CC_STAT_HEIGHT]), int(stats[i, cv2.CC_STAT_AREA]))
        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        if area < int(preset["lane_min_area"]) or hh < int(preset["lane_min_height"]):
            continue
        if area < 18 and aspect < 1.7:
            continue
        if y + hh < int(0.06 * h) or (ww > int(0.75 * w) and hh < int(0.12 * h)):
            continue
        yellow_lane |= labels == i
    if not np.any(yellow_lane):
        yellow_lane = yellow_work.copy()

    ehw = int(preset["edge_half_width"])
    yellow_edge = (_extract_line_mask(yellow_lane, "left", 0.10, 0.04, ehw)
                   | _extract_line_mask(yellow_lane, "right", 0.10, 0.04, ehw))
    yellow_edge = cv2.morphologyEx(
        _u8(yellow_edge), cv2.MORPH_CLOSE,
        np.ones((int(preset["edge_close_h"]), int(preset["edge_close_w"])), np.uint8),
    ) > 0

    seed = yellow_lane | yellow_edge
    yellow_full = cv2.morphologyEx(_u8(seed | yellow_work), cv2.MORPH_CLOSE, _k(int(preset["final_close_k"]))) > 0
    if int(preset["final_dilate_iter"]) > 0:
        yellow_full = cv2.dilate(_u8(yellow_full), _k(3), iterations=int(preset["final_dilate_iter"])) > 0

    seed_d = cv2.dilate(_u8(seed), _k(5), iterations=1) > 0
    yellow_near = cv2.dilate(_u8(yellow_work), _k(3), iterations=1) > 0
    n2, labels2, stats2, _ = cv2.connectedComponentsWithStats(_u8(yellow_full), connectivity=8)
    out = np.zeros_like(yellow_full, dtype=bool)
    for i in range(1, n2):
        y, ww, hh, area = (int(stats2[i, cv2.CC_STAT_TOP]), int(stats2[i, cv2.CC_STAT_WIDTH]),
                           int(stats2[i, cv2.CC_STAT_HEIGHT]), int(stats2[i, cv2.CC_STAT_AREA]))
        top_comp = (y + hh) < int(0.62 * h)
        ma = int(preset["top_min_area"] if top_comp else preset["main_min_area"])
        if area < ma or hh < 2 or y + hh < int(0.06 * h) or (ww > int(0.75 * w) and hh < int(0.12 * h)):
            continue
        comp = labels2 == i
        if np.any(comp & seed_d):
            out |= comp
        elif top_comp and np.any(comp & yellow_near):
            out |= comp

    yellow_full = out if np.any(out) else (seed | yellow_work)
    yellow_full = cv2.morphologyEx(_u8(yellow_full), cv2.MORPH_CLOSE, _k(3)) > 0
    yellow_full &= road_relaxed
    if not return_confidence:
        return yellow_full, yellow_edge, yellow_on_road

    # Confidence map: hue/sat/value evidence inside extracted yellow geometry.
    h_ch = src_hsv[:, :, 0].astype(np.float32)
    s_ch = src_hsv[:, :, 1].astype(np.float32)
    v_ch = src_hsv[:, :, 2].astype(np.float32)
    hue_center = 24.0
    hue_delta = np.minimum(np.abs(h_ch - hue_center), 180.0 - np.abs(h_ch - hue_center))
    hue_conf = np.clip(1.0 - hue_delta / 18.0, 0.0, 1.0)
    sat_conf = np.clip((s_ch - 55.0) / 145.0, 0.0, 1.0)
    val_conf = np.clip((v_ch - 70.0) / 150.0, 0.0, 1.0)
    edge_hint = cv2.GaussianBlur(yellow_edge.astype(np.float32), (3, 3), sigmaX=0.8, sigmaY=0.8)
    onroad_hint = yellow_on_road.astype(np.float32)
    base = yellow_full.astype(np.float32)
    yellow_conf = base * (
        0.48 * hue_conf
        + 0.30 * sat_conf
        + 0.14 * val_conf
        + 0.08 * edge_hint
    )
    yellow_conf = np.maximum(yellow_conf, 0.40 * onroad_hint * (0.65 * hue_conf + 0.35 * sat_conf))
    yellow_conf = cv2.GaussianBlur(yellow_conf, (3, 3), sigmaX=0.8, sigmaY=0.8)
    yellow_conf = np.clip(yellow_conf * base, 0.0, 1.0).astype(np.float32)
    return yellow_full, yellow_edge, yellow_on_road, yellow_conf


def extract_ws_white_case10_line(
    hsv: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
    yellow_mask: Optional[np.ndarray] = None,
    src_bgr: Optional[np.ndarray] = None,
    return_confidence: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract the WS white center dashes that are recolored into newtrack's warm
    centerline:
      1) tophat-based white dash detection (improved v5),
      2) thin center seed extraction,
      3) recover connected center-white blocks in the support band,
      4) add a thin soft-white ring for anti-aliased borders.

    Always uses improved multi-scale tophat extraction from
    src/test_ws2newtrack_white.py path. If src_bgr is None, reconstruct BGR
    from HSV so legacy callers still run the same extractor.
    """
    road_u8 = (
        ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 65)).astype(np.uint8) * 255
        if road_mask is None
        else _u8(road_mask)
    )
    road_u8 = cv2.dilate(cv2.morphologyEx(road_u8, cv2.MORPH_CLOSE, _k(5)), _k(5), iterations=1)
    road = road_u8 > 0

    # Unified path: always use tophat-based white dash extraction.
    if src_bgr is None:
        src_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    white_seed_conf: Optional[np.ndarray] = None
    if return_confidence:
        tophat_mask, white_seed_conf = _extract_white_dashes_tophat(
            src_bgr,
            road_mask=road_mask,
            return_confidence=True,
        )
    else:
        tophat_mask = _extract_white_dashes_tophat(src_bgr, road_mask=road_mask)
    white_clean = tophat_mask > 0

    seed = _row_centroid_encode(
        white_clean, band_width=9, min_pixels=1, max_width_ratio=0.18, split_gap=5,
    )
    white_soft = (
        ((hsv[:, :, 2] >= 182) & (hsv[:, :, 1] <= 55))
        | ((hsv[:, :, 2] >= 168) & (hsv[:, :, 1] <= 72))
        | ((hsv[:, :, 2] >= 160) & (hsv[:, :, 1] <= 80))
    )
    base = _expand_white_seed_to_center_blocks(
        white_clean, white_soft, seed, road, yellow_mask=yellow_mask,
    )
    center = _add_center_white_edge_ring(
        base, white_soft, road, yellow_mask=yellow_mask, anchor_mask=seed,
    )
    center_mask = _filter_centerline_components(center & road, seed)

    # Fill gaps between dashes → connect into solid white line.
    # Tall narrow CLOSE bridges the inter-dash gaps without merging left/right lines.
    if np.any(center_mask):
        _h = center_mask.shape[0]
        close_h = max(5, int(_h * 0.13))   # ~15px for 120px frame
        fill_ker = np.ones((close_h, 3), np.uint8)
        filled_mask = cv2.morphologyEx(_u8(center_mask), cv2.MORPH_CLOSE, fill_ker) > 0
        # Restrict fill to within reach of the originally detected dashes
        near_orig = cv2.dilate(_u8(center_mask), np.ones((close_h + 2, 7), np.uint8)) > 0
        center_mask = (filled_mask & near_orig & road)

    if not return_confidence:
        return center_mask

    # Confidence map used for WS white channel visualization/probability:
    # keep tophat confidence as primary evidence, then softly propagate to
    # connected centerline pixels recovered by morphology.
    v_ch = hsv[:, :, 2].astype(np.float32)
    s_ch = hsv[:, :, 1].astype(np.float32)
    white_evidence = (
        np.clip((v_ch - 148.0) / 107.0, 0.0, 1.0)
        * np.clip((88.0 - s_ch) / 88.0, 0.0, 1.0)
    ).astype(np.float32)
    if white_seed_conf is None:
        seed_conf = white_evidence * white_clean.astype(np.float32)
    else:
        seed_conf = np.maximum(
            np.clip(white_seed_conf.astype(np.float32), 0.0, 1.0),
            0.45 * white_evidence * white_clean.astype(np.float32),
        ).astype(np.float32)

    seed_spread = cv2.dilate(
        (seed_conf * 255.0).astype(np.uint8),
        _k(3),
        iterations=1,
    ).astype(np.float32) / 255.0
    seed_spread = cv2.GaussianBlur(seed_spread, (3, 3), sigmaX=0.8, sigmaY=0.8)
    center_soft = cv2.GaussianBlur(center_mask.astype(np.float32), (3, 3), sigmaX=0.8, sigmaY=0.8)
    center_conf = np.maximum(seed_conf, 0.55 * seed_spread + 0.45 * (white_evidence * center_soft))
    if yellow_mask is not None:
        center_conf *= (1.0 - np.clip(yellow_mask.astype(np.float32), 0.0, 1.0))
    center_conf = np.clip(center_conf * center_mask.astype(np.float32), 0.0, 1.0).astype(np.float32)
    return center_mask, center_conf


# ---------------------------------------------------------------------------
# Observation-space WS edge/white/yellow enhancement (used by obv wrapper)
# ---------------------------------------------------------------------------
def _obs_filter_edge_components(
    edge_hard: np.ndarray,
    min_area: int,
    min_height: int,
    max_compactness: float,
    min_aspect: float,
) -> np.ndarray:
    m = (edge_hard > 0).astype(np.uint8)
    if int(np.count_nonzero(m)) == 0:
        return np.zeros_like(edge_hard, dtype=np.float32)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < int(min_area) or h < int(min_height):
            continue
        compact = float(area) / float(max(1, w * h))
        max_side = float(max(w, h))
        min_side = float(max(1, min(w, h)))
        elongated = max_side >= float(min_aspect) * min_side
        if compact > float(max_compactness) and not elongated:
            continue
        out[labels == i] = 1
    return out.astype(np.float32)


def _obs_filter_line_components(
    mask: np.ndarray,
    min_area: int,
    min_height: int,
    max_compactness: float,
    tall_ratio: float,
) -> np.ndarray:
    m = (mask > 0.0).astype(np.uint8)
    if int(np.count_nonzero(m)) == 0:
        return np.zeros_like(mask, dtype=np.float32)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m, dtype=np.uint8)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < int(min_area) or h < int(min_height):
            continue
        compact = float(area) / float(max(1, w * h))
        tall_enough = float(h) >= float(tall_ratio) * float(max(1, w))
        if compact > float(max_compactness) and not tall_enough:
            continue
        out[labels == i] = 1
    return out.astype(np.float32)


def _obs_edge_support_mask(edge_hard: np.ndarray, dilate_iter: int) -> np.ndarray:
    edge_u8 = (edge_hard > 0).astype(np.uint8) * 255
    if int(dilate_iter) > 0:
        edge_u8 = cv2.dilate(
            edge_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=int(dilate_iter),
        )
    return edge_u8 > 0


def _obs_edge_strength_gate(
    edge_raw: np.ndarray,
    edge_hard: np.ndarray,
    min_strength: float,
    power: float,
) -> np.ndarray:
    denom = max(1e-6, 1.0 - float(min_strength))
    edge_strength = np.clip((edge_raw - float(min_strength)) / denom, 0.0, 1.0)
    if abs(float(power) - 1.0) > 1e-6:
        edge_strength = np.power(edge_strength, float(power))
    return (edge_strength * edge_hard).astype(np.float32)


def _obs_close_line_gaps(mask: np.ndarray, kernel: int, iterations: int) -> np.ndarray:
    if int(iterations) <= 0 or int(kernel) <= 1:
        return mask.astype(np.float32)
    k = int(kernel)
    if (k % 2) == 0:
        k += 1
    m = (mask > 0.0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=int(iterations))
    return m.astype(np.float32)


def _obs_ws_dash_chain_filter(
    mask: np.ndarray,
    close_h: int,
    close_w: int,
    min_height: int,
    top_min_height: int,
    max_width: int,
    fill_weight: float,
) -> np.ndarray:
    m = (mask > 0.0).astype(np.uint8)
    if int(np.count_nonzero(m)) == 0:
        return m.astype(np.float32)

    ker = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (int(max(1, close_w)), int(max(1, close_h))),
    )
    bridge = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bridge, connectivity=8)
    keep_bridge = np.zeros_like(bridge, dtype=np.uint8)
    h_img = bridge.shape[0]
    top_y = int(0.45 * h_img)
    for i in range(1, n):
        top = int(stats[i, cv2.CC_STAT_TOP])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cy = top + hh // 2
        min_h = int(top_min_height) if cy < top_y else int(min_height)
        max_w = int(max_width * (1.15 if cy < top_y else 1.0))
        if hh < min_h:
            continue
        if ww > max_w and hh < int(1.2 * min_h):
            continue
        if area < max(6, min_h * 2):
            continue
        keep_bridge[labels == i] = 1

    if int(np.count_nonzero(keep_bridge)) == 0:
        return m.astype(np.float32)
    base = ((m > 0) & (keep_bridge > 0)).astype(np.float32)
    if float(fill_weight) <= 1e-6:
        return base
    fill = ((keep_bridge > 0) & (base < 0.5)).astype(np.float32)
    return np.clip(base + float(fill_weight) * fill, 0.0, 1.0).astype(np.float32)


def _obs_ws_white_floor_map(h: int, floor: float, far_boost: float) -> np.ndarray:
    y = np.linspace(0.0, 1.0, int(h), dtype=np.float32)[:, None]
    far_relax = 1.0 - y
    out = float(floor) + float(far_boost) * far_relax
    return np.clip(out, 0.0, 0.98).astype(np.float32)


def _obs_ws_center_corridor_mask(seed_mask: np.ndarray, near_w: int, far_w: int) -> np.ndarray:
    h, w = seed_mask.shape
    cx = 0.5 * (w - 1)
    out = np.zeros_like(seed_mask, dtype=np.float32)
    prev_mid: Optional[float] = None
    for y in range(h - 1, -1, -1):
        xs = np.flatnonzero(seed_mask[y] > 0.0)
        if xs.size > 0:
            cur_mid = float(xs.mean())
            if prev_mid is None:
                mid = cur_mid
            else:
                max_jump = 0.08 * w
                cur_mid = float(np.clip(cur_mid, prev_mid - max_jump, prev_mid + max_jump))
                mid = 0.70 * prev_mid + 0.30 * cur_mid
        else:
            mid = cx if prev_mid is None else prev_mid
        prev_mid = mid
        y_ratio = float(y) / max(1.0, float(h - 1))
        half_w = int(near_w) + (1.0 - y_ratio) * (int(far_w) - int(near_w))
        x0 = max(0, int(round(mid - half_w)))
        x1 = min(w, int(round(mid + half_w)) + 1)
        out[y, x0:x1] = 1.0
    return out


def _obs_edge_guided_color_prob(
    color_prob: np.ndarray,
    edge_geom: np.ndarray,
    filter_floor: float,
    boost_gain: float,
) -> np.ndarray:
    edge_soft = cv2.GaussianBlur(edge_geom.astype(np.float32), (5, 5), sigmaX=1.2)
    gate = float(filter_floor) + (1.0 - float(filter_floor)) * edge_soft
    boosted = color_prob * gate * (1.0 + float(boost_gain) * edge_soft)
    return np.clip(boosted, 0.0, 1.0)


def build_ws_observation_line_probs(
    raw_bgr: np.ndarray,
    raw_y: np.ndarray,
    *,
    prev_y: Optional[np.ndarray] = None,       # kept for compat, unused
    prev_white_prob: Optional[np.ndarray] = None,  # kept for compat, unused
    edge_preblur_sigma: float = 1.0,
    edge_support_thresh: float = 0.035,
    edge_support_dilate: int = 1,
    edge_comp_min_area: int = 10,
    edge_comp_min_height: int = 3,
    edge_comp_max_compactness: float = 0.80,
    edge_comp_aspect_ratio: float = 1.3,
    **_kwargs,  # absorb deprecated ws_white_* / line_* params
) -> Dict[str, np.ndarray]:
    """
    Build WS observation channels (edge + white/yellow prob).
    white_prob / yellow_prob = raw detection confidence (no blur, no dilation).
    Edge channel = Sobel on raw_y.
    """
    # ── Edge channel ────────────────────────────────────────────────
    y_for_edge = raw_y.astype(np.float32)
    if float(edge_preblur_sigma) > 1e-6:
        y_for_edge = cv2.GaussianBlur(
            y_for_edge, (0, 0),
            sigmaX=float(edge_preblur_sigma),
            sigmaY=float(edge_preblur_sigma),
        )
    gx = cv2.Sobel(y_for_edge, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_for_edge, cv2.CV_32F, 0, 1, ksize=3)
    edge_raw = np.clip(np.sqrt(gx ** 2 + gy ** 2) / 4.0, 0.0, 1.0)
    edge_hard = _obs_filter_edge_components(
        (edge_raw > float(edge_support_thresh)).astype(np.float32),
        min_area=int(edge_comp_min_area),
        min_height=int(edge_comp_min_height),
        max_compactness=float(edge_comp_max_compactness),
        min_aspect=float(edge_comp_aspect_ratio),
    )
    edge = (edge_raw * edge_hard).astype(np.float32)

    # ── WS detection → confidence (no dilation, no blur) ────────────
    hsv_ws = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    road_ws = road_support_mask(hsv_ws) > 0

    yellow_out = extract_ws_yellow_mask(
        hsv_ws, road_ws, preset_name="case01", return_confidence=True,
    )
    if isinstance(yellow_out, tuple) and len(yellow_out) == 4:
        yellow_full, _, _, yellow_conf = yellow_out
        yellow_conf = np.clip(yellow_conf.astype(np.float32), 0.0, 1.0)
    else:
        yellow_full, _, _ = yellow_out
        yellow_conf = yellow_full.astype(np.float32)

    white_out = extract_ws_white_case10_line(
        hsv_ws, road_mask=road_ws, yellow_mask=yellow_full,
        src_bgr=raw_bgr, return_confidence=True,
    )
    if isinstance(white_out, tuple):
        _, white_conf = white_out
        white_conf = np.clip(white_conf.astype(np.float32), 0.0, 1.0)
    else:
        white_conf = (white_out > 0).astype(np.float32)

    return {
        "edge": edge,
        "white_prob": white_conf,
        "yellow_prob": yellow_conf,
    }


_WS_CLOTH_HSV = np.array([128.0, 24.0, 172.0], dtype=np.float32)
_WS_BG_HSV = np.array([125.0, 20.0, 120.0], dtype=np.float32)


def _render_ws_road_surface(img_bgr: np.ndarray) -> np.ndarray:
    """Transform WS road surface to NT cloth texture. Lines are NOT modified."""
    src_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    src_v = src_hsv[:, :, 2]
    hsv_u8 = src_hsv.astype(np.uint8)

    road = road_support_mask(hsv_u8)
    masks = semantic_masks(hsv_u8)
    line_mask = masks["yellow"] | masks["white"] | masks["blue"]
    road_region = road & ~line_mask
    bg_region = ~road & ~line_mask

    if not np.any(road_region):
        return img_bgr

    h, w = img_bgr.shape[:2]
    sigma = max(3.0, 0.10 * float(min(h, w)))
    shade = cv2.GaussianBlur(src_v, (0, 0), sigmaX=sigma, sigmaY=sigma)
    hf = src_v - shade
    road_ref = float(np.mean(shade[road_region]))

    out_hsv = src_hsv.copy()
    out_hsv[:, :, 0][road_region] = float(_WS_CLOTH_HSV[0])
    out_hsv[:, :, 1][road_region] = np.clip(
        float(_WS_CLOTH_HSV[1]) + 0.22 * 0.3 * hf[road_region], 0.0, 60.0)
    out_hsv[:, :, 2][road_region] = np.clip(
        float(_WS_CLOTH_HSV[2]) + 0.40 * (shade[road_region] - road_ref)
        + 0.22 * hf[road_region], 0.0, 255.0)

    if np.any(bg_region):
        bg_ref = float(np.mean(shade[bg_region]))
        out_hsv[:, :, 0][bg_region] = float(_WS_BG_HSV[0])
        out_hsv[:, :, 1][bg_region] = np.clip(
            float(_WS_BG_HSV[1]) + 0.18 * 0.3 * hf[bg_region], 0.0, 60.0)
        out_hsv[:, :, 2][bg_region] = np.clip(
            float(_WS_BG_HSV[2]) + 0.25 * (shade[bg_region] - bg_ref)
            + 0.18 * hf[bg_region], 0.0, 255.0)

    return cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def transform_ws_to_newtrack(
    img_bgr: np.ndarray, tgt_stats: Dict, yellow_preset_name: str = "case01",
) -> np.ndarray:
    # Road surface -> NT cloth texture first
    road_img = _render_ws_road_surface(img_bgr)
    # Then apply line blending on top
    src_hsv = cv2.cvtColor(road_img, cv2.COLOR_BGR2HSV)
    road = road_support_mask(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    masks = semantic_masks(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV))
    # Keep parameter for backward compatibility; line extraction currently uses case01 only.
    preset = WS_YELLOW_PRESETS["case01"]
    yellow_full, _, _ = extract_ws_yellow_mask(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), road, preset_name=yellow_preset_name)
    # WS center white becomes the newtrack warm centerline.
    center_white = extract_ws_white_case10_line(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), road_mask=road, yellow_mask=yellow_full,
        src_bgr=img_bgr)
    result = _apply_newtrack_blend(
        src_hsv, yellow_full | (masks["blue"] & road), center_white, tgt_stats,
        line_a=(preset["blue_alpha_h"], preset["blue_alpha_s"], preset["blue_alpha_v"]),
        center_a=_WS_CENTER_BLEND, v_boost=_WS_CENTER_V_BOOST)

    # 保留绿车原始像素，不被道路渲染覆盖
    green = _green_detector.detect(img_bgr)
    if green.detected:
        result[green.mask > 0] = img_bgr[green.mask > 0]
    return result


# ---------------------------------------------------------------------------
__all__ = [
    "CLS_ORDER", "WS_YELLOW_PRESETS",
    "build_ws_observation_line_probs",
    "compute_stats", "extract_ws_white_case10_line", "extract_ws_yellow_mask",
    "list_images", "load_images", "overlay_semantics",
    "road_support_mask", "sample_paths", "semantic_masks",
    "transform_ws_to_newtrack",
]
