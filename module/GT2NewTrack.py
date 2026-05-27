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


# Observation presets aligned with validate_channels.py GT branch.
GT_OBS_PRESETS: Dict[str, Dict[str, float]] = {
    "tight": {
        "edge_preblur_sigma": 1.0,
        "edge_support_thresh": 0.035,
        "edge_support_dilate": 1,
        "edge_comp_min_area": 10,
        "edge_comp_min_height": 3,
        "edge_comp_max_compactness": 0.80,
        "edge_comp_aspect_ratio": 1.3,
        "line_comp_min_area": 8,
        "line_comp_min_height": 2,
        "line_comp_max_compactness": 0.72,
        "line_comp_tall_ratio": 1.8,
        "line_edge_filter_floor": 0.0,
        "line_edge_boost_gain": 0.8,
        "line_edge_min_strength": 0.06,
        "line_edge_strength_power": 1.4,
        "line_edge_hard_floor": 0.25,
        "line_close_kernel": 1,
        "line_close_iter": 0,
        "line_prob_min": 0.001,
    },
    "relax1": {
        "edge_preblur_sigma": 0.9,
        "edge_support_thresh": 0.030,
        "edge_support_dilate": 1,
        "edge_comp_min_area": 8,
        "edge_comp_min_height": 2,
        "edge_comp_max_compactness": 0.84,
        "edge_comp_aspect_ratio": 1.25,
        "line_comp_min_area": 6,
        "line_comp_min_height": 2,
        "line_comp_max_compactness": 0.80,
        "line_comp_tall_ratio": 1.6,
        "line_edge_filter_floor": 0.02,
        "line_edge_boost_gain": 0.9,
        "line_edge_min_strength": 0.05,
        "line_edge_strength_power": 1.3,
        "line_edge_hard_floor": 0.30,
        "line_close_kernel": 1,
        "line_close_iter": 0,
        "line_prob_min": 0.001,
    },
}


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
# GT edge-geometry enhancement (for GT2NewTrack rendering path)
# ---------------------------------------------------------------------------
def _filter_edge_components(
    edge_hard_u8: np.ndarray,
    min_area: int = 10,
    min_height: int = 3,
    max_compactness: float = 0.80,
    min_aspect: float = 1.3,
) -> np.ndarray:
    m = (edge_hard_u8 > 0).astype(np.uint8)
    if int(np.count_nonzero(m)) == 0:
        return np.zeros_like(edge_hard_u8, dtype=np.uint8)

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
        out[labels == i] = 255
    return out


def _gt_edge_support_mask(
    img_bgr: np.ndarray,
    road_support: Optional[np.ndarray] = None,
    preblur_sigma: float = 1.0,
    edge_thresh: float = 0.035,
    dilate_iter: int = 1,
) -> np.ndarray:
    y = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0].astype(np.float32) / 255.0
    if float(preblur_sigma) > 1e-6:
        y = cv2.GaussianBlur(y, (0, 0), sigmaX=float(preblur_sigma), sigmaY=float(preblur_sigma))
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    edge_raw = np.clip(np.sqrt(gx**2 + gy**2) / 4.0, 0.0, 1.0)

    edge_hard = np.where(edge_raw > float(edge_thresh), 255, 0).astype(np.uint8)
    edge_hard = _filter_edge_components(edge_hard)

    if road_support is not None and np.any(road_support):
        edge_hard = cv2.bitwise_and(edge_hard, road_support)

    if int(dilate_iter) > 0:
        edge_hard = cv2.dilate(
            edge_hard,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=int(dilate_iter),
        )
    return edge_hard


# ---------------------------------------------------------------------------
# Observation-space edge/color fusion helpers (shared by obv wrapper)
# ---------------------------------------------------------------------------
def _edge_support_mask(edge_hard: np.ndarray, dilate_iter: int = 1) -> np.ndarray:
    edge_u8 = np.where(edge_hard > 0, 255, 0).astype(np.uint8)
    if int(dilate_iter) > 0:
        edge_u8 = cv2.dilate(
            edge_u8,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=int(dilate_iter),
        )
    return edge_u8 > 0


def _filter_line_components(
    mask: np.ndarray,
    min_area: int = 8,
    min_height: int = 2,
    max_compactness: float = 0.72,
    tall_ratio: float = 1.8,
) -> np.ndarray:
    m = np.where(mask > 0.0, 255, 0).astype(np.uint8)
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
        compactness = float(area) / float(max(1, w * h))
        tall_enough = float(h) >= float(tall_ratio) * float(max(1, w))
        if compactness > float(max_compactness) and not tall_enough:
            continue
        out[labels == i] = 255
    return (out > 0).astype(np.float32)


def _edge_guided_color_prob(
    color_prob: np.ndarray,
    edge_geom: np.ndarray,
    filter_floor: float = 0.0,
    boost_gain: float = 0.8,
) -> np.ndarray:
    edge_soft = cv2.GaussianBlur(edge_geom.astype(np.float32), (5, 5), sigmaX=1.2)
    gate = float(filter_floor) + (1.0 - float(filter_floor)) * edge_soft
    boosted = color_prob * gate * (1.0 + float(boost_gain) * edge_soft)
    return np.clip(boosted, 0.0, 1.0)


def _edge_strength_gate(
    edge_raw: np.ndarray,
    edge_hard: np.ndarray,
    min_strength: float = 0.06,
    strength_power: float = 1.4,
) -> np.ndarray:
    denom = max(1e-6, 1.0 - float(min_strength))
    edge_strength = np.clip((edge_raw - float(min_strength)) / denom, 0.0, 1.0)
    if abs(float(strength_power) - 1.0) > 1e-6:
        edge_strength = np.power(edge_strength, float(strength_power))
    return (edge_strength * edge_hard).astype(np.float32)


def _close_line_gaps(mask: np.ndarray, kernel: int = 1, iterations: int = 0) -> np.ndarray:
    if int(iterations) <= 0 or int(kernel) <= 1:
        return mask.astype(np.float32)
    k = int(kernel)
    if (k % 2) == 0:
        k += 1
    m = np.where(mask > 0.0, 255, 0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker, iterations=int(iterations))
    return (m > 0).astype(np.float32)


def _gt_line_confidence_maps(
    raw_bgr: np.ndarray,
    road_support: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build GT color-confidence maps for white/yellow lines (0~1)."""
    hsv = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]

    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local = cv2.GaussianBlur(gray, (0, 0), sigmaX=9.0, sigmaY=9.0)
    contrast = gray - local

    white_vs = (
        np.clip((v_ch - 150.0) / 105.0, 0.0, 1.0)
        * np.clip((78.0 - s_ch) / 78.0, 0.0, 1.0)
    )
    white_ct = np.clip((contrast + 2.0) / 30.0, 0.0, 1.0)
    white_conf = np.clip(
        0.72 * white_vs + 0.28 * white_ct * np.clip((90.0 - s_ch) / 90.0, 0.0, 1.0),
        0.0,
        1.0,
    ).astype(np.float32)

    hue_center = 27.0
    hue_delta = np.minimum(np.abs(h_ch - hue_center), 180.0 - np.abs(h_ch - hue_center))
    yellow_h = np.clip(1.0 - hue_delta / 22.0, 0.0, 1.0)
    yellow_s = np.clip((s_ch - 55.0) / 150.0, 0.0, 1.0)
    yellow_v = np.clip((v_ch - 60.0) / 170.0, 0.0, 1.0)
    yellow_conf = np.clip(
        (0.60 * yellow_h + 0.25 * yellow_s + 0.15 * yellow_v) * yellow_h,
        0.0,
        1.0,
    ).astype(np.float32)

    if road_support is not None and np.any(road_support):
        road_f = (road_support > 0).astype(np.float32)
        white_conf *= (0.25 + 0.75 * road_f)
        yellow_conf *= (0.35 + 0.65 * road_f)

    return white_conf.astype(np.float32), yellow_conf.astype(np.float32)


def build_gt_observation_line_probs(
    raw_bgr: np.ndarray,
    raw_y: np.ndarray,
    *,
    preset_name: str = "relax1",
    line_conf_mode: str = "none",   # kept for compat, unused
    line_conf_blend: float = 0.75,  # kept for compat, unused
    line_conf_prob_floor: float = 0.30,  # kept for compat, unused
    **overrides: float,
) -> Dict[str, np.ndarray]:
    """
    Build GT observation channels.
    white_prob / yellow_prob = raw detection masks (no blur, no dilation).
    Edge channel = Sobel on raw_y.
    """
    key = str(preset_name).lower()
    if key not in GT_OBS_PRESETS:
        raise ValueError(f"Unknown GT observation preset: {preset_name}")
    cfg = dict(GT_OBS_PRESETS[key])
    cfg.update(overrides)

    road_support = detect_gt_road_support(raw_bgr)
    white_mask  = (detect_white_line(raw_bgr, road_support=road_support, edge_enhance=True) > 0)
    yellow_mask = (detect_yellow_line(raw_bgr, road_support=road_support) > 0)

    # Sobel edge channel.
    y_fe = raw_y.astype(np.float32)
    sigma = float(cfg.get("edge_preblur_sigma", 1.0))
    if sigma > 1e-6:
        y_fe = cv2.GaussianBlur(y_fe, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gx = cv2.Sobel(y_fe, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_fe, cv2.CV_32F, 0, 1, ksize=3)
    edge_raw = np.clip(np.sqrt(gx ** 2 + gy ** 2) / 4.0, 0.0, 1.0)
    edge_hard = (edge_raw > float(cfg.get("edge_support_thresh", 0.035))).astype(np.float32)
    edge = (edge_raw * edge_hard).astype(np.float32)

    # White confidence: multi-scale tophat response weights each detected pixel.
    # Line width is unchanged — tophat only modulates intensity, no spatial expansion.
    gray = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY)
    th = np.zeros_like(gray, dtype=np.float32)
    for ks in [(11, 11), (17, 17), (25, 25)]:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ks)
        th = np.maximum(th, cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, k).astype(np.float32))
    th_norm = np.clip(th / 80.0, 0.0, 1.0)
    white_prob = np.clip(
        white_mask.astype(np.float32) * (0.40 + 0.60 * th_norm), 0.0, 1.0
    ).astype(np.float32)

    # Yellow confidence: saturation × brightness.
    hsv_conf = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2HSV)
    s_n = hsv_conf[:, :, 1].astype(np.float32) / 255.0
    v_n = hsv_conf[:, :, 2].astype(np.float32) / 255.0
    yellow_prob = np.clip(
        yellow_mask.astype(np.float32) * np.clip(s_n * v_n * 2.5, 0.0, 1.0), 0.0, 1.0
    ).astype(np.float32)

    return {
        "edge":        edge,
        "edge_raw":    edge_raw,
        "white_prob":  white_prob,
        "yellow_prob": yellow_prob,
        "white_raw":   white_mask.astype(np.float32),
        "yellow_raw":  yellow_mask.astype(np.float32),
    }


def build_observation_line_probs(
    raw_y: np.ndarray,
    white_raw: np.ndarray,
    yellow_raw: np.ndarray,
    raw_bgr: Optional[np.ndarray] = None,
    *,
    line_conf_mode: str = "none",
    line_conf_blend: float = 0.75,
    line_conf_prob_floor: float = 0.30,
    edge_preblur_sigma: float = 1.0,
    edge_support_thresh: float = 0.035,
    edge_support_dilate: int = 1,
    edge_comp_min_area: int = 10,
    edge_comp_min_height: int = 3,
    edge_comp_max_compactness: float = 0.80,
    edge_comp_aspect_ratio: float = 1.3,
    line_comp_min_area: int = 8,
    line_comp_min_height: int = 2,
    line_comp_max_compactness: float = 0.72,
    line_comp_tall_ratio: float = 1.8,
    line_edge_filter_floor: float = 0.0,
    line_edge_boost_gain: float = 0.8,
    line_edge_min_strength: float = 0.06,
    line_edge_strength_power: float = 1.4,
    line_edge_hard_floor: float = 0.25,
    line_close_kernel: int = 1,
    line_close_iter: int = 0,
    line_prob_min: float = 0.001,
) -> Dict[str, np.ndarray]:
    """
    Build observation-space edge/white/yellow channels from raw masks.
    This keeps low-level morphology/gating out of module/obv.py.
    """
    y_for_edge = raw_y.astype(np.float32)
    if float(edge_preblur_sigma) > 1e-6:
        y_for_edge = cv2.GaussianBlur(
            y_for_edge,
            (0, 0),
            sigmaX=float(edge_preblur_sigma),
            sigmaY=float(edge_preblur_sigma),
        )
    gx = cv2.Sobel(y_for_edge, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y_for_edge, cv2.CV_32F, 0, 1, ksize=3)
    edge_raw = np.clip(np.sqrt(gx ** 2 + gy ** 2) / 4.0, 0.0, 1.0)

    edge_hard_u8 = np.where(edge_raw > float(edge_support_thresh), 255, 0).astype(np.uint8)
    edge_hard_u8 = _filter_edge_components(
        edge_hard_u8,
        min_area=int(edge_comp_min_area),
        min_height=int(edge_comp_min_height),
        max_compactness=float(edge_comp_max_compactness),
        min_aspect=float(edge_comp_aspect_ratio),
    )
    edge_hard = (edge_hard_u8 > 0).astype(np.float32)
    edge = (edge_raw * edge_hard).astype(np.float32)
    edge_support = _edge_support_mask(edge_hard, dilate_iter=int(edge_support_dilate)).astype(np.float32)

    edge_strength = _edge_strength_gate(
        edge_raw,
        edge_hard,
        min_strength=float(line_edge_min_strength),
        strength_power=float(line_edge_strength_power),
    )
    edge_gate = np.maximum(edge_strength, float(line_edge_hard_floor) * edge_hard).astype(np.float32)

    white_mask = _filter_line_components(
        np.asarray(white_raw, dtype=np.float32) * edge_support,
        min_area=int(line_comp_min_area),
        min_height=int(line_comp_min_height),
        max_compactness=float(line_comp_max_compactness),
        tall_ratio=float(line_comp_tall_ratio),
    )
    yellow_mask = _filter_line_components(
        np.asarray(yellow_raw, dtype=np.float32) * edge_support,
        min_area=int(line_comp_min_area),
        min_height=int(line_comp_min_height),
        max_compactness=float(line_comp_max_compactness),
        tall_ratio=float(line_comp_tall_ratio),
    )
    white_mask = _close_line_gaps(white_mask, kernel=int(line_close_kernel), iterations=int(line_close_iter))
    yellow_mask = _close_line_gaps(yellow_mask, kernel=int(line_close_kernel), iterations=int(line_close_iter))

    use_gt_conf = (str(line_conf_mode).lower() == "gt") and (raw_bgr is not None)
    white_seed_conf = np.zeros_like(white_mask, dtype=np.float32)
    yellow_seed_conf = np.zeros_like(yellow_mask, dtype=np.float32)
    white_color_input = white_mask.copy()
    yellow_color_input = yellow_mask.copy()
    if use_gt_conf:
        road_hint = detect_gt_road_support(raw_bgr)
        white_color_conf, yellow_color_conf = _gt_line_confidence_maps(raw_bgr, road_support=road_hint)
        white_near = cv2.dilate(
            (np.asarray(white_raw, dtype=np.float32) > 0.0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(np.float32)
        yellow_near = cv2.dilate(
            (np.asarray(yellow_raw, dtype=np.float32) > 0.0).astype(np.uint8),
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        ).astype(np.float32)
        white_seed_conf = np.clip(
            white_color_conf * edge_support * (0.18 + 0.82 * white_near),
            0.0,
            1.0,
        ).astype(np.float32)
        yellow_seed_conf = np.clip(
            yellow_color_conf * edge_support * (0.20 + 0.80 * yellow_near),
            0.0,
            1.0,
        ).astype(np.float32)
        white_seed_conf = np.maximum(white_seed_conf, 0.35 * white_mask).astype(np.float32)
        yellow_seed_conf = np.maximum(yellow_seed_conf, 0.35 * yellow_mask).astype(np.float32)
        white_color_input = np.maximum(
            white_mask,
            float(np.clip(line_conf_blend, 0.0, 1.0)) * white_seed_conf,
        ).astype(np.float32)
        yellow_color_input = np.maximum(
            yellow_mask,
            float(np.clip(line_conf_blend, 0.0, 1.0)) * yellow_seed_conf,
        ).astype(np.float32)

    white_color = cv2.GaussianBlur(white_color_input, (5, 5), sigmaX=1.5) * edge_support
    yellow_color = cv2.GaussianBlur(yellow_color_input, (5, 5), sigmaX=1.5) * edge_support

    white_prob = _edge_guided_color_prob(
        white_color,
        edge,
        filter_floor=float(line_edge_filter_floor),
        boost_gain=float(line_edge_boost_gain),
    ) * edge_gate
    yellow_prob = _edge_guided_color_prob(
        yellow_color,
        edge,
        filter_floor=float(line_edge_filter_floor),
        boost_gain=float(line_edge_boost_gain),
    ) * edge_gate
    if use_gt_conf:
        conf_floor = float(np.clip(line_conf_prob_floor, 0.0, 1.0))
        white_prob = np.maximum(white_prob, conf_floor * white_seed_conf * edge_gate).astype(np.float32)
        yellow_prob = np.maximum(yellow_prob, conf_floor * yellow_seed_conf * edge_gate).astype(np.float32)

    if float(line_prob_min) > 0.0:
        white_prob = np.where(white_prob >= float(line_prob_min), white_prob, 0.0).astype(np.float32)
        yellow_prob = np.where(yellow_prob >= float(line_prob_min), yellow_prob, 0.0).astype(np.float32)

    return {
        "edge": edge.astype(np.float32),
        "edge_raw": edge_raw.astype(np.float32),
        "edge_hard": edge_hard.astype(np.float32),
        "edge_support": edge_support.astype(np.float32),
        "edge_gate": edge_gate.astype(np.float32),
        "white_conf": np.clip(white_seed_conf, 0.0, 1.0).astype(np.float32),
        "yellow_conf": np.clip(yellow_seed_conf, 0.0, 1.0).astype(np.float32),
        "white_prob": np.clip(white_prob, 0.0, 1.0).astype(np.float32),
        "yellow_prob": np.clip(yellow_prob, 0.0, 1.0).astype(np.float32),
    }


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
    img_bgr: np.ndarray,
    road_support: Optional[np.ndarray] = None,
    edge_enhance: bool = True,
) -> np.ndarray:
    """Return binary mask of white edge lines."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]
    h_ch, s_ch = hsv[:, :, 0], hsv[:, :, 1]

    mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    mask = _apply_road_support(mask, road_support)
    contrast_mask = _detect_white_line_local_contrast(img_bgr, road_support=road_support)
    merged = cv2.bitwise_or(mask, contrast_mask)
    if edge_enhance:
        support = _gt_edge_support_mask(img_bgr, road_support=road_support)
        if np.any(support):
            merged = cv2.bitwise_and(merged, support)

    # Exclude yellow-hued pixels — wider range + lower S to catch faded far-end yellow.
    yellow_excl = cv2.inRange(hsv, np.array([8, 12, 50]), np.array([50, 255, 255]))
    yellow_excl = cv2.dilate(yellow_excl, np.ones((3, 3), np.uint8), iterations=1)
    merged = cv2.bitwise_and(merged, cv2.bitwise_not(yellow_excl))

    # Erode road support to strip grass/road boundary pixels.
    if road_support is not None and np.any(road_support):
        road_tight = cv2.erode(
            np.where(road_support > 0, 255, 0).astype(np.uint8),
            np.ones((3, 3), np.uint8), iterations=1,
        )
        merged = cv2.bitwise_and(merged, road_tight)

    out = _denoise_shape(merged, min_area=40, min_elongated_area=8, min_aspect=1.5, skip_open=True)

    # Per-CC shape + grass border check.
    if np.any(out):
        green_strict = ((h_ch >= 30) & (h_ch <= 95) & (s_ch >= 50)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            ww   = int(stats[i, cv2.CC_STAT_WIDTH])
            hh   = int(stats[i, cv2.CC_STAT_HEIGHT])
            cy   = int(stats[i, cv2.CC_STAT_TOP]) + hh // 2
            aspect = max(ww, hh) / float(max(1, min(ww, hh)))

            # Reject compact blobs (road surface noise): real edge lines are elongated.
            if area < 400 and aspect < 1.5:
                out[labels == i] = 0
                continue

            # Green grass border check (near zone only).
            if cy < int(h_img * 0.40):
                continue
            comp_u8 = (labels == i).astype(np.uint8) * 255
            border = cv2.dilate(comp_u8, np.ones((3, 3), np.uint8)) - comp_u8
            border_px = int(np.count_nonzero(border))
            if border_px > 0 and float(np.sum(green_strict[border > 0])) / border_px > 0.25:
                out[labels == i] = 0

    return out


# ---------------------------------------------------------------------------
# Yellow line detector  (HSV near/far + perspective CC + far-zone validation)
# ---------------------------------------------------------------------------
def detect_yellow_line(
    img_bgr: np.ndarray,
    road_support: Optional[np.ndarray] = None,
    edge_enhance: bool = False,
) -> np.ndarray:
    """
    Return binary mask of yellow center line.

    Migrated from src/gt_line_extractor.py yellow_A_hsv:
    near/far HSV thresholds + perspective-aware connected-component filtering +
    far-zone color re-check.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_img, w_img = hsv.shape[:2]

    if road_support is None:
        road_u8 = detect_gt_road_support(img_bgr)
    else:
        road_u8 = np.where(road_support > 0, 255, 0).astype(np.uint8)

    # Standard (near) and relaxed (far) yellow thresholds.
    mask_std = cv2.inRange(hsv, np.array([12, 50, 70], np.uint8), np.array([45, 255, 255], np.uint8))
    mask_far = cv2.inRange(hsv, np.array([10, 35, 50], np.uint8), np.array([48, 255, 255], np.uint8))

    far_zone = np.zeros((h_img, w_img), dtype=np.uint8)
    far_boundary = int(h_img * 0.45)
    far_zone[:far_boundary, :] = 255
    mask = cv2.bitwise_or(mask_std, cv2.bitwise_and(mask_far, far_zone))

    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Exclude orange cones.
    orange = ((h_ch >= 0) & (h_ch <= 18) & (s_ch >= 120) & (v_ch >= 100)).astype(np.uint8) * 255
    orange = cv2.dilate(orange, np.ones((3, 3), np.uint8), iterations=2)

    # Erode road mask to strip grass/edge boundary pixels (no dilation of exclusion zone).
    road_tight = cv2.erode(road_u8, np.ones((5, 5), np.uint8), iterations=1)

    mask = cv2.bitwise_and(mask, road_tight)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(orange))
    # Small noise removal only — no CLOSE to preserve natural line width.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    # Perspective-aware CC filtering: far zone accepts smaller components.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        x0 = int(stats[i, cv2.CC_STAT_LEFT])
        cy = int(stats[i, cv2.CC_STAT_TOP]) + hh // 2
        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        # Reject greenish-yellow (H>42 ≈ grass edge): true yellow line H≈15-30
        comp_px = labels == i
        mean_h_v = float(np.mean(h_ch[comp_px]))
        if mean_h_v > 42:
            continue
        # Reject small CCs confined to edge columns (grass boundary artifacts)
        if area < 200 and (x0 + ww <= 18 or x0 >= w_img - 18):
            continue
        if cy < far_boundary:
            if area >= 5 or (area >= 3 and aspect >= 1.2):
                out[labels == i] = 255
        else:
            if area >= 40 or (area >= 10 and aspect >= 1.8):
                out[labels == i] = 255

    # Far-zone hue/sat validation to suppress relaxed-threshold drift.
    if np.any(out[:far_boundary]):
        n2, labels2, _, _ = cv2.connectedComponentsWithStats(out[:far_boundary].copy(), connectivity=8)
        for i in range(1, n2):
            comp = labels2 == i
            mean_h = float(np.mean(h_ch[:far_boundary][comp]))
            mean_s = float(np.mean(s_ch[:far_boundary][comp]))
            if not (10.0 <= mean_h <= 42.0 and mean_s >= 25.0):
                out[:far_boundary][comp] = 0

    # Remove near-end CCs whose border is predominantly adjacent to green grass.
    # Uses strict green (S>=50) and requires >30% of CC border to be green before rejecting.
    green_strict = ((h_ch >= 30) & (h_ch <= 95) & (s_ch >= 50)).astype(np.uint8)
    n3, labels3, stats3, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    near_boundary = int(h_img * 0.40)
    for i in range(1, n3):
        cy_i = int(stats3[i, cv2.CC_STAT_TOP]) + int(stats3[i, cv2.CC_STAT_HEIGHT]) // 2
        if cy_i <= near_boundary:
            continue  # far-end — skip
        comp_u8 = (labels3 == i).astype(np.uint8) * 255
        border = cv2.dilate(comp_u8, np.ones((3, 3), np.uint8)) - comp_u8
        border_px = np.count_nonzero(border)
        if border_px > 0 and np.sum(green_strict[border > 0]) / border_px > 0.30:
            out[labels3 == i] = 0

    return out


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
    white_mask = detect_white_line(img_bgr, road_support, edge_enhance=True) > 0
    yellow_mask = detect_yellow_line(img_bgr, road_support, edge_enhance=True) > 0
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
        white = detect_white_line(img, road_support=road_support, edge_enhance=True)
        yellow = detect_yellow_line(img, road_support=road_support, edge_enhance=True)
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
