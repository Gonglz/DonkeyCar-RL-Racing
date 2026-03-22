#!/usr/bin/env python3
"""
Waveshare to newtrack transfer helpers.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    "case04": {**_WS_YELLOW_BASE, "lane_min_area": 12, "main_min_area": 10},
    "case07": {**_WS_YELLOW_BASE, "lane_min_area": 10, "edge_half_width": 2,
               "blue_alpha_s": 0.90, "blue_alpha_v": 0.60},
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


def _remove_small(mask: np.ndarray, min_area: int = 24, max_area: Optional[int] = None) -> np.ndarray:
    if not np.any(mask):
        return mask.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(_u8(mask), connectivity=8)
    out = np.zeros_like(mask, dtype=bool)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a >= min_area and (max_area is None or a <= max_area):
            out |= labels == i
    return out


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
    src_hsv: np.ndarray, road_mask: np.ndarray, preset_name: str = "case01",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preset = WS_YELLOW_PRESETS[preset_name.lower()]
    masks = semantic_masks(src_hsv)
    yellow_raw = masks["yellow"] & ~_ws_green_text_mask(src_hsv)
    yellow_on_road = yellow_raw & road_mask
    if not np.any(yellow_raw):
        empty = yellow_raw.copy()
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
    return yellow_full, yellow_edge, yellow_on_road


def extract_ws_white_case10_line(
    hsv: np.ndarray,
    road_mask: Optional[np.ndarray] = None,
    yellow_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extract the WS white center dashes that are recolored into newtrack's warm
    centerline:
      1) strict white detection on road,
      2) thin center seed extraction,
      3) recover connected center-white blocks in the support band,
      4) add a thin soft-white ring for anti-aliased borders.
    """
    road_u8 = (
        ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] >= 65)).astype(np.uint8) * 255
        if road_mask is None
        else _u8(road_mask)
    )
    road_u8 = cv2.dilate(cv2.morphologyEx(road_u8, cv2.MORPH_CLOSE, _k(5)), _k(7), iterations=2)
    road = road_u8 > 0
    white_raw = cv2.bitwise_and(
        cv2.inRange(hsv, np.array([0, 0, 200], np.uint8), np.array([179, 35, 255], np.uint8)),
        road_u8,
    ) > 0
    white_clean = cv2.morphologyEx(_u8(white_raw), cv2.MORPH_CLOSE, _k(3)) > 0
    white_clean = _remove_small(white_clean, min_area=4)
    seed = _row_centroid_encode(
        white_clean, band_width=7, min_pixels=1, max_width_ratio=0.14, split_gap=5,
    )
    white_soft = (
        ((hsv[:, :, 2] >= 182) & (hsv[:, :, 1] <= 55))
        | ((hsv[:, :, 2] >= 168) & (hsv[:, :, 1] <= 72))
    )
    base = _expand_white_seed_to_center_blocks(
        white_clean, white_soft, seed, road, yellow_mask=yellow_mask,
    )
    center = _add_center_white_edge_ring(
        base, white_soft, road, yellow_mask=yellow_mask, anchor_mask=seed,
    )
    return _filter_centerline_components(center & road, seed)


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
    preset = WS_YELLOW_PRESETS[yellow_preset_name.lower()]
    yellow_full, _, _ = extract_ws_yellow_mask(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), road, preset_name=yellow_preset_name)
    # WS center white becomes the newtrack warm centerline.
    center_white = extract_ws_white_case10_line(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV), road_mask=road, yellow_mask=yellow_full)
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
    "compute_stats", "extract_ws_white_case10_line", "extract_ws_yellow_mask",
    "list_images", "load_images", "overlay_semantics",
    "road_support_mask", "sample_paths", "semantic_masks",
    "transform_ws_to_newtrack",
]
