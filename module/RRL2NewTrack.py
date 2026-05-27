#!/usr/bin/env python3
"""
RRL (sim racing) to NewTrack style transfer.

Converts rrl scene images to match new_track's visual style:
  - dark asphalt road  -> white/light cloth floor
  - white edge lines   -> blue tape lines  (all lines: near, far, straight, curved)
  - black/white checker -> white floor with blue accents

Uses local-contrast detection (V - local_mean_V) instead of absolute
brightness thresholds so that distant thin lines are captured.
"""

import argparse
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .green_vehicle_detect import GreenVehicleDetector

_green_detector = GreenVehicleDetector()
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RRL_RAW_DIR = REPO_ROOT / "data/scene_samples/rrl/raw"
DEFAULT_RRL_OUT_DIR = REPO_ROOT / "data/scene_samples/rrl/RRL2NewTrack"


# ---------------------------------------------------------------------------
# Target HSV prototypes (from new_track)
# ---------------------------------------------------------------------------
TARGET_BLUE_HSV = np.array([112.0, 185.0, 175.0], dtype=np.float32)
TARGET_ORANGE_HSV = np.array([18.0, 185.0, 200.0], dtype=np.float32)
TARGET_CLOTH_HSV = np.array([128.0, 24.0, 172.0], dtype=np.float32)
TARGET_BG_HSV = np.array([125.0, 20.0, 120.0], dtype=np.float32)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mask_u8(mask: np.ndarray) -> np.ndarray:
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _kernel(kh: int, kw: Optional[int] = None) -> np.ndarray:
    kw = kh if kw is None else kw
    return np.ones((max(1, int(kh)), max(1, int(kw))), np.uint8)


# ---------------------------------------------------------------------------
# White line detection: local-contrast approach
# ---------------------------------------------------------------------------
def _split_row_clusters(cols: np.ndarray, split_gap: int = 5):
    """Split a sorted array of column indices into clusters by gap."""
    if len(cols) == 0:
        return []
    gaps = np.where(np.diff(cols) > split_gap)[0]
    if len(gaps) == 0:
        return [cols]
    return list(np.split(cols, gaps + 1))


def _row_multi_centroid_encode(
    mask: np.ndarray,
    min_band: int = 3,
    pad: int = 1,
    min_pixels: int = 1,
    max_width_ratio: float = 0.15,
    split_gap: int = 5,
) -> np.ndarray:
    """Like WS2NewTrack's _row_centroid_encode, but keeps ALL line clusters
    per row instead of only the strongest. This handles multiple parallel lines.

    For each row, finds all clusters of white pixels and fills the full
    cluster span (+ padding). For thin clusters (1-2px), uses min_band
    to ensure continuity. This covers both thick near lines and thin far lines.
    """
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    max_span = max(1, int(round(w * max_width_ratio)))
    min_half = max(1, min_band // 2)

    for row in range(h):
        cols = np.flatnonzero(mask[row])
        if len(cols) < min_pixels:
            continue
        clusters = _split_row_clusters(cols, split_gap)
        for cl in clusters:
            if len(cl) < min_pixels:
                continue
            span = cl[-1] - cl[0] + 1
            if span > max_span:
                continue
            # Use actual cluster extent + padding, with minimum band
            center_x = int(np.round(cl.mean()))
            half = max(min_half, span // 2 + pad)
            left = max(0, center_x - half)
            right = min(w, center_x + half + 1)
            out[row, left:right] = True
    return out


def _vertical_continuity_filter(
    mask: np.ndarray,
    min_run: int = 4,
) -> np.ndarray:
    """Remove columns of mask that don't have vertical continuity.

    For each column, require at least `min_run` consecutive True rows
    to keep that column's contribution. This removes isolated noise rows
    while preserving actual lines (which span many consecutive rows).
    """
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    # For efficiency, process via connected components
    mask_u8 = _mask_u8(mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    for i in range(1, n):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        if ch >= min_run:
            out[labels == i] = True
    return out


def _filter_rrl_candidate_components(
    candidate_mask: np.ndarray,
    value_channel: np.ndarray,
    contrast_map: np.ndarray,
) -> np.ndarray:
    """Reject large compact bright blobs before row-wise encoding.

    RRL wall highlights often enter the raw candidate mask as big chunky
    upper-frame patches. The row encoder then turns those patches into false
    vertical bands. Real lane lines are usually either:
    - sparse wide bands (lower/mid frame), or
    - narrow/tall traces (upper frame).
    """
    if not np.any(candidate_mask):
        return candidate_mask.copy()

    h, w = candidate_mask.shape
    cand_u8 = _mask_u8(candidate_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    keep = np.zeros_like(candidate_mask, dtype=bool)

    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        bottom = y + hh
        cy = y + hh // 2
        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        fill_ratio = area / float(max(1, ww * hh))
        mean_v = float(np.mean(value_channel[comp]))
        mean_contrast = float(np.mean(contrast_map[comp]))

        # Upper bright wall patches: big, compact, and not line-like.
        bulky_upper = (
            cy < int(0.42 * h)
            and area >= max(120, int(0.010 * h * w))
            and fill_ratio >= 0.58
            and aspect < 2.2
        )
        huge_upper_blob = (
            bottom < int(0.48 * h)
            and area >= max(420, int(0.022 * h * w))
            and fill_ratio >= 0.30
            and aspect < 2.0
        )
        compact_mid_blob = (
            cy < int(0.55 * h)
            and area >= max(180, int(0.012 * h * w))
            and fill_ratio >= 0.72
            and aspect < 2.6
            and mean_v >= 185.0
        )
        if bulky_upper or huge_upper_blob or compact_mid_blob:
            # Preserve narrow/tall traces and sparse wide bands that still
            # look like actual lane paint rather than filled wall patches.
            narrow_tall = ww <= max(16, int(0.10 * w)) and hh >= max(20, int(0.16 * h))
            sparse_wide = ww >= max(28, int(0.16 * w)) and fill_ratio <= 0.40
            high_conf = mean_contrast >= 42.0 and fill_ratio <= 0.65
            if not (narrow_tall or sparse_wide or high_conf):
                continue

        keep |= comp

    return keep


def _detect_rrl_horizontal_supplement(
    candidate_mask: np.ndarray,
    value_channel: np.ndarray,
    contrast_map: np.ndarray,
    guide_mask: Optional[np.ndarray] = None,
    min_support_ratio: float = 0.42,
) -> np.ndarray:
    """Recover rare near-horizontal white lines when the vertical pass is weak.

    Some RRL views bring the near white curb almost horizontally across the
    image. The row-wise encoder is intentionally biased toward perspective
    lanes, so those frames can end up under-detected. We only use this as a
    low-coverage fallback and keep the geometry narrow via Hough line segments
    instead of filling the full bright blob.
    """
    if not np.any(candidate_mask):
        return candidate_mask.copy()

    h, w = candidate_mask.shape
    if guide_mask is not None and np.any(guide_mask):
        lower_guide = guide_mask[int(0.45 * h):, :]
        # This path is meant to be a fallback for frames where the main
        # detector under-covers the lower horizontal curb. If the main path
        # already has decent lower-half support, adding another horizontal pass
        # tends to over-thicken or flatten the line geometry.
        if np.count_nonzero(lower_guide) >= int(0.030 * h * w):
            return candidate_mask.copy() & False

    cand_u8 = _mask_u8(candidate_mask)
    lines = cv2.HoughLinesP(
        cand_u8,
        1,
        np.pi / 180,
        threshold=10,
        minLineLength=max(14, int(0.14 * w)),
        maxLineGap=12,
    )
    if lines is None:
        return candidate_mask.copy() & False

    accepted_lines = []
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = map(int, line)
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        if length < 0.10 * w:
            continue
        # Only treat genuinely near-horizontal segments here. Larger-angle
        # oblique lines should stay on the main perspective-biased path.
        if angle > 18.0:
            continue
        if max(y1, y2) < 0.42 * h:
            continue
        line_mask = np.zeros_like(cand_u8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
        support_ratio = float(
            np.count_nonzero((cand_u8 > 0) & (line_mask > 0))
        ) / float(max(1, np.count_nonzero(line_mask)))
        if support_ratio < min_support_ratio:
            continue

        accepted_lines.append(
            (
                support_ratio * length,
                0.5 * float(y1 + y2),
                angle,
                x1, y1, x2, y2,
            )
        )

    if not accepted_lines:
        return candidate_mask.copy() & False

    accepted_lines.sort(reverse=True)
    seed = np.zeros_like(candidate_mask, dtype=bool)
    kept_lines = []
    for score, cy_line, angle, x1, y1, x2, y2 in accepted_lines:
        duplicate = False
        for _, kept_cy, kept_angle, kx1, ky1, kx2, ky2 in kept_lines:
            overlap_x = min(max(x1, x2), max(kx1, kx2)) - max(min(x1, x2), min(kx1, kx2))
            if abs(cy_line - kept_cy) <= 5.0 and abs(angle - kept_angle) <= 4.0 and overlap_x >= 0.35 * w:
                duplicate = True
                break
        if duplicate:
            continue

        line_mask = np.zeros_like(cand_u8)
        cv2.line(line_mask, (x1, y1), (x2, y2), 255, thickness=3)
        seed |= line_mask > 0
        kept_lines.append((score, cy_line, angle, x1, y1, x2, y2))
        if len(kept_lines) >= 3:
            break

    seed &= candidate_mask
    if not np.any(seed):
        return candidate_mask.copy() & False

    seed_u8 = _mask_u8(seed)
    seed_d = cv2.dilate(seed_u8, _kernel(3, 9), iterations=1) > 0
    guide_d = None
    if guide_mask is not None and np.any(guide_mask):
        guide_d = cv2.dilate(_mask_u8(guide_mask), _kernel(5), iterations=1) > 0
    narrow_candidates = candidate_mask & seed_d
    n, labels, stats, _ = cv2.connectedComponentsWithStats(_mask_u8(narrow_candidates), connectivity=8)
    keep = np.zeros_like(candidate_mask, dtype=bool)
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        aspect = ww / float(max(1, hh))
        y = int(stats[i, cv2.CC_STAT_TOP])
        if area < 20:
            continue
        if ww < max(12, int(0.08 * w)):
            continue
        if hh > int(0.14 * h):
            continue
        if aspect < 2.4:
            continue
        comp = labels == i
        touch_seed = np.any(comp & seed_d)
        touch_guide = guide_d is not None and np.any(comp & guide_d)
        if not touch_seed and not touch_guide:
            continue
        mean_v = float(np.mean(value_channel[comp]))
        mean_contrast = float(np.mean(contrast_map[comp]))
        if mean_v < 190.0 and mean_contrast < 32.0:
            continue
        if y + hh < int(0.42 * h):
            # Upper-frame horizontal pieces are only accepted when they are
            # attached to an already-detected white line; this recovers frames
            # like rrl_00157 without reopening free-floating top clutter.
            if not touch_guide:
                continue
            if mean_v < 175.0 and mean_contrast < 80.0:
                continue
        keep |= comp
    return keep


def _recover_rrl_upper_adjacent_segments(
    result_mask: np.ndarray,
    clean_mask: np.ndarray,
    value_channel: np.ndarray,
    contrast_map: np.ndarray,
) -> np.ndarray:
    """Keep short upper-curve segments when they sit right next to a kept line.

    RRL frames like rrl_00157 can produce a thin upper arc that survives the
    row/continuity pass but is only ~8px tall, so the generic min-height cleanup
    drops it. We only recover segments that are:
    - already present in the vertically continuous `result_mask`
    - near the very top of the frame
    - wide, bright/high-contrast, and horizontally adjacent to a kept segment
    """
    if not np.any(result_mask) or not np.any(clean_mask):
        return clean_mask

    h, w = result_mask.shape
    n, labels, stats, _ = cv2.connectedComponentsWithStats(result_mask, connectivity=8)
    recovered = clean_mask.copy()
    accepted_boxes = []
    accepted_ids = set()

    for i in range(1, n):
        comp = labels == i
        if np.any(comp & (clean_mask > 0)):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            ww = int(stats[i, cv2.CC_STAT_WIDTH])
            hh = int(stats[i, cv2.CC_STAT_HEIGHT])
            accepted_boxes.append((x, y, ww, hh))
            accepted_ids.add(i)

    for i in range(1, n):
        if i in accepted_ids:
            continue

        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 70:
            continue
        if ww < max(24, int(0.10 * w)):
            continue
        if hh > 12:
            continue
        if y > int(0.20 * h):
            continue

        comp = labels == i
        mean_v = float(np.mean(value_channel[comp]))
        mean_contrast = float(np.mean(contrast_map[comp]))
        if mean_v < 185.0 and mean_contrast < 60.0:
            continue

        close_to_kept = False
        for ax, ay, aw, ah in accepted_boxes:
            overlap_y = min(y + hh, ay + ah) - max(y, ay)
            x_gap = max(0, max(ax - (x + ww), x - (ax + aw)))
            if overlap_y >= 2 and x_gap <= 18:
                close_to_kept = True
                break

        if not close_to_kept:
            continue

        recovered[comp] = 255
        accepted_boxes.append((x, y, ww, hh))

    return recovered


def _recover_rrl_curve_continuation_segments(
    encoded_mask: np.ndarray,
    clean_mask: np.ndarray,
    value_channel: np.ndarray,
    contrast_map: np.ndarray,
) -> np.ndarray:
    """Recover short upper/mid arc fragments that continue a kept lower line.

    Some curved RRL curb lines appear as a set of short 3-6px-high bands in the
    upper half. They are too short for the main min_run=8 pass, but they still
    form a plausible continuation of an already-kept lower arc. We only recover
    fragments that are line-like and geometrically adjacent to an accepted line,
    then iteratively extend that chain upward/sideways.
    """
    if not np.any(encoded_mask) or not np.any(clean_mask):
        return clean_mask

    h, w = encoded_mask.shape
    loose = _vertical_continuity_filter(encoded_mask, min_run=3)
    extra = loose & ~(clean_mask > 0)
    if not np.any(extra):
        return clean_mask

    recovered = clean_mask.copy()
    comp_u8 = _mask_u8(extra)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(comp_u8, connectivity=8)

    pending = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 18:
            continue
        if ww < max(16, int(0.10 * w)):
            continue
        if hh > max(14, int(0.12 * h)):
            continue
        if y >= int(0.78 * h):
            continue

        comp = labels == i
        fill_ratio = area / float(max(1, ww * hh))
        mean_v = float(np.mean(value_channel[comp]))
        mean_contrast = float(np.mean(contrast_map[comp]))
        aspect = ww / float(max(1, hh))
        if fill_ratio > 0.62 and aspect < 2.4:
            continue
        if mean_v < 145.0 and mean_contrast < 24.0:
            continue
        pending.append((i, area, x, y, ww, hh))

    if not pending:
        return recovered

    accepted_ids = set()
    for _ in range(3):
        nr, rlabels, rstats, _ = cv2.connectedComponentsWithStats(recovered, connectivity=8)
        accepted_boxes = []
        for j in range(1, nr):
            accepted_boxes.append(
                (
                    int(rstats[j, cv2.CC_STAT_LEFT]),
                    int(rstats[j, cv2.CC_STAT_TOP]),
                    int(rstats[j, cv2.CC_STAT_WIDTH]),
                    int(rstats[j, cv2.CC_STAT_HEIGHT]),
                )
            )

        changed = False
        for i, area, x, y, ww, hh in pending:
            if i in accepted_ids:
                continue

            comp = labels == i
            cx = x + 0.5 * ww
            cy = y + 0.5 * hh
            attach = False
            for ax, ay, aw, ah in accepted_boxes:
                acx = ax + 0.5 * aw
                x_overlap = min(x + ww, ax + aw) - max(x, ax)
                y_overlap = min(y + hh, ay + ah) - max(y, ay)
                x_gap = max(0, max(ax - (x + ww), x - (ax + aw)))
                gap_above = ay - (y + hh)
                center_gap = abs(cx - acx)

                vertical_attach = (
                    0 <= gap_above <= max(26, int(0.22 * h))
                    and (
                        x_overlap >= max(4, int(0.12 * min(ww, aw)))
                        or x_gap <= max(10, int(0.06 * w))
                        or center_gap <= max(18, int(0.16 * w))
                    )
                )
                lateral_attach = (
                    y_overlap >= 2
                    and x_gap <= max(12, int(0.08 * w))
                    and cy <= ay + ah + max(8, int(0.06 * h))
                )
                if vertical_attach or lateral_attach:
                    attach = True
                    break

            if not attach:
                continue

            recovered[comp] = 255
            accepted_ids.add(i)
            changed = True

        if not changed:
            break

    return recovered


def _recover_rrl_candidate_arc_bands(
    candidate_mask: np.ndarray,
    clean_mask: np.ndarray,
    value_channel: np.ndarray,
    contrast_map: np.ndarray,
) -> np.ndarray:
    """Recover wide-but-thin upper arc bands skipped by row encoding.

    The row encoder rejects very wide row spans on purpose, but some curved
    curb lines appear as a sparse 2-4px-high arc band in the upper-middle
    frame. We only recover such bands when they sit directly above an already
    accepted lower line and can be chained sideways across neighboring bands.
    """
    if not np.any(candidate_mask) or not np.any(clean_mask):
        return clean_mask

    h, w = candidate_mask.shape
    cand_u8 = _mask_u8(candidate_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand_u8, connectivity=8)
    pending = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 24:
            continue
        if ww < max(18, int(0.12 * w)):
            continue
        if hh > max(6, int(0.06 * h)):
            continue
        if not (int(0.25 * h) <= y <= int(0.72 * h)):
            continue

        comp = labels == i
        fill_ratio = area / float(max(1, ww * hh))
        mean_v = float(np.mean(value_channel[comp]))
        mean_contrast = float(np.mean(contrast_map[comp]))
        if fill_ratio > 0.72:
            continue
        if mean_v < 180.0 and mean_contrast < 45.0:
            continue
        pending.append((i, x, y, ww, hh))

    if not pending:
        return clean_mask

    recovered = clean_mask.copy()
    accepted_ids = set()
    for _ in range(3):
        nr, rlabels, rstats, _ = cv2.connectedComponentsWithStats(recovered, connectivity=8)
        accepted_boxes = []
        for j in range(1, nr):
            accepted_boxes.append(
                (
                    int(rstats[j, cv2.CC_STAT_LEFT]),
                    int(rstats[j, cv2.CC_STAT_TOP]),
                    int(rstats[j, cv2.CC_STAT_WIDTH]),
                    int(rstats[j, cv2.CC_STAT_HEIGHT]),
                )
            )

        changed = False
        for i, x, y, ww, hh in pending:
            if i in accepted_ids:
                continue

            cx = x + 0.5 * ww
            cy = y + 0.5 * hh
            attach = False
            for ax, ay, aw, ah in accepted_boxes:
                acx = ax + 0.5 * aw
                x_overlap = min(x + ww, ax + aw) - max(x, ax)
                x_gap = max(0, max(ax - (x + ww), x - (ax + aw)))
                y_overlap = min(y + hh, ay + ah) - max(y, ay)
                gap_above = ay - (y + hh)
                center_gap = abs(cx - acx)

                upward_attach = (
                    0 <= gap_above <= max(30, int(0.25 * h))
                    and (
                        x_overlap >= max(4, int(0.10 * min(ww, aw)))
                        or x_gap <= max(10, int(0.06 * w))
                        or center_gap <= max(20, int(0.18 * w))
                    )
                )
                lateral_chain = (
                    abs(cy - (ay + 0.5 * ah)) <= max(8, int(0.06 * h))
                    and y_overlap >= 1
                    and x_gap <= max(12, int(0.08 * w))
                )
                if upward_attach or lateral_chain:
                    attach = True
                    break

            if not attach:
                continue

            recovered[labels == i] = 255
            accepted_ids.add(i)
            changed = True

        if not changed:
            break

    return recovered


def _bootstrap_rrl_empty_detection(
    filtered_mask: np.ndarray,
    value_channel: np.ndarray,
    contrast_map: np.ndarray,
    road_mask: np.ndarray,
) -> np.ndarray:
    """Bootstrap a seed line when the normal cleanup rejects everything.

    This is intentionally a last-resort path. It only runs when the main
    detector produced an empty mask, and it reconstructs at most the strongest
    one or two connected line-like chains from the vertically continuous mask.
    """
    if not np.any(filtered_mask):
        return _mask_u8(filtered_mask) & 0

    h, w = filtered_mask.shape
    filt_u8 = _mask_u8(filtered_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(filt_u8, connectivity=8)

    comps = []
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 40:
            continue

        road_overlap = float(np.count_nonzero(comp & road_mask)) / float(max(1, area))
        mean_v = float(np.mean(value_channel[comp]))
        mean_contrast = float(np.mean(contrast_map[comp]))
        fill_ratio = area / float(max(1, ww * hh))
        aspect = max(ww, hh) / float(max(1, min(ww, hh)))
        if road_overlap < 0.45:
            continue
        if mean_v < 165.0 and mean_contrast < 18.0:
            continue
        if fill_ratio > 0.82 and aspect < 1.8:
            continue

        score = (
            float(area)
            + 1.4 * mean_contrast
            + 0.35 * mean_v
            + 8.0 * aspect
            - 0.55 * float(y)
        )
        comps.append(
            {
                "id": i,
                "area": area,
                "x": x,
                "y": y,
                "w": ww,
                "h": hh,
                "score": score,
            }
        )

    if not comps:
        return np.zeros_like(filt_u8)

    # Build loose chains of vertically or diagonally adjacent components.
    neighbors = {c["id"]: set() for c in comps}
    for idx, a in enumerate(comps):
        ax0, ay0, ax1, ay1 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
        acx = a["x"] + 0.5 * a["w"]
        for b in comps[idx + 1:]:
            bx0, by0, bx1, by1 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
            bcx = b["x"] + 0.5 * b["w"]
            x_overlap = min(ax1, bx1) - max(ax0, bx0)
            y_overlap = min(ay1, by1) - max(ay0, by0)
            x_gap = max(0, max(ax0 - bx1, bx0 - ax1))
            y_gap = max(0, max(ay0 - by1, by0 - ay1))
            center_gap = abs(acx - bcx)
            linked = (
                (y_gap <= max(12, int(0.10 * h)) and (x_overlap >= 4 or x_gap <= max(12, int(0.08 * w))))
                or (y_overlap >= 1 and x_gap <= max(16, int(0.10 * w)))
                or (y_gap <= max(18, int(0.15 * h)) and center_gap <= max(24, int(0.16 * w)))
            )
            if linked:
                neighbors[a["id"]].add(b["id"])
                neighbors[b["id"]].add(a["id"])

    comp_by_id = {c["id"]: c for c in comps}
    groups = []
    seen = set()
    for comp in comps:
        cid = comp["id"]
        if cid in seen:
            continue
        stack = [cid]
        group_ids = []
        seen.add(cid)
        while stack:
            cur = stack.pop()
            group_ids.append(cur)
            for nxt in neighbors[cur]:
                if nxt in seen:
                    continue
                seen.add(nxt)
                stack.append(nxt)

        xs = [comp_by_id[i]["x"] for i in group_ids]
        ys = [comp_by_id[i]["y"] for i in group_ids]
        x2s = [comp_by_id[i]["x"] + comp_by_id[i]["w"] for i in group_ids]
        y2s = [comp_by_id[i]["y"] + comp_by_id[i]["h"] for i in group_ids]
        total_area = sum(comp_by_id[i]["area"] for i in group_ids)
        total_score = sum(comp_by_id[i]["score"] for i in group_ids)
        group_w = max(x2s) - min(xs)
        group_h = max(y2s) - min(ys)
        groups.append(
            {
                "ids": group_ids,
                "area": total_area,
                "score": total_score,
                "x": min(xs),
                "y": min(ys),
                "w": group_w,
                "h": group_h,
            }
        )

    groups.sort(key=lambda g: (g["score"], g["area"]), reverse=True)
    out = np.zeros_like(filt_u8)
    kept_groups = 0
    for group in groups:
        if kept_groups >= 2:
            break
        group_bottom = group["y"] + group["h"]
        if group_bottom < int(0.42 * h):
            continue
        if group["area"] < 180:
            continue
        if group["h"] < 18:
            continue
        if group["score"] < 260.0:
            continue
        for cid in group["ids"]:
            out[labels == cid] = 255
        kept_groups += 1

    return out


def detect_rrl_white_lines(img_bgr: np.ndarray) -> np.ndarray:
    """Detect ALL white lane lines (near, far, straight, curved).

    Inspired by WS2NewTrack's row-centroid approach:
    1. Local-contrast thresholding to find bright-vs-surroundings pixels
    2. Row-by-row multi-cluster encoding (keeps ALL lines, not just one)
       - No morphological opening that would destroy 1-2px thin lines
       - Each cluster gets a band_width to form a continuous stripe
    3. Vertical continuity filtering to reject noise (real lines span
       many consecutive rows; noise is isolated)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1]
    road_mask = cv2.dilate(detect_rrl_road_surface(img_bgr), _kernel(5), iterations=1) > 0
    checker_mask = cv2.dilate(detect_rrl_checker(img_bgr), _kernel(3), iterations=1) > 0

    # --- Local contrast ---
    local_mean = cv2.GaussianBlur(v, (0, 0), sigmaX=12, sigmaY=12)
    contrast = v - local_mean

    # --- Candidate pixels ---
    candidates = (
        ((contrast > 35) & (s < 35))          # standard contrast
        | ((contrast > 25) & (s < 25) & (v >= 140))  # faint far lines
        | ((v >= 210) & (s < 20))             # absolute bright
    )
    candidates = _filter_rrl_candidate_components(candidates, v, contrast)

    # --- Close tiny gaps (1-2px) within lines, but NO opening ---
    # (opening destroys thin lines)
    cand_u8 = _mask_u8(candidates)
    cand_u8 = cv2.morphologyEx(cand_u8, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                iterations=1)

    # --- Row-by-row multi-cluster encoding ---
    # This finds ALL line clusters per row and draws a band around each
    encoded = _row_multi_centroid_encode(
        cand_u8 > 0,
        min_band=3,
        pad=1,
        min_pixels=1,
        max_width_ratio=0.15,
        split_gap=6,
    )

    # --- Vertical continuity filter ---
    # Real lines span many consecutive rows; road texture noise is short
    filtered = _vertical_continuity_filter(encoded, min_run=8)

    # --- Final cleanup: remove short/small components ---
    result = _mask_u8(filtered)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
    clean = np.zeros_like(result)
    road_touch = cv2.dilate(_mask_u8(road_mask), _kernel(7), iterations=1) > 0
    for i in range(1, n):
        comp = labels == i
        area = int(stats[i, cv2.CC_STAT_AREA])
        x_i = int(stats[i, cv2.CC_STAT_LEFT])
        y_i = int(stats[i, cv2.CC_STAT_TOP])
        ww_i = int(stats[i, cv2.CC_STAT_WIDTH])
        hh_i = int(stats[i, cv2.CC_STAT_HEIGHT])
        road_overlap = float(np.count_nonzero(comp & road_touch)) / float(max(1, area))
        fill_ratio = area / float(max(1, ww_i * hh_i))
        mean_v = float(np.mean(v[comp]))
        mean_contrast = float(np.mean(contrast[comp]))
        if road_overlap < 0.20:
            continue
        border_hug = x_i <= 2 or (x_i + ww_i) >= (w - 2)
        if border_hug and (y_i + hh_i) < int(0.55 * h):
            aspect = max(ww_i, hh_i) / float(max(1, min(ww_i, hh_i)))
            raw_overlap = float(np.count_nonzero(comp & road_mask)) / float(max(1, area))
            if raw_overlap < 0.55:
                continue
            if aspect < 1.6:
                continue
        cy = y_i + hh_i // 2
        ch_i = hh_i
        aspect = max(ww_i, hh_i) / float(max(1, min(ww_i, hh_i)))
        lower_lane_like = (
            area >= 55
            and road_overlap >= 0.58
            and (mean_v >= 145.0 or mean_contrast >= 18.0)
        )
        wide_sparse = (
            ww_i >= max(24, int(0.12 * w))
            and fill_ratio <= 0.42
        )
        horiz_ok = (
            ww_i >= max(28, int(0.14 * w))
            and hh_i >= 3
            and area >= 32
            and ww_i / float(max(1, hh_i)) >= 2.8
            and fill_ratio <= 0.60
        )
        upper_narrow = (
            ww_i <= max(16, int(0.10 * w))
            and aspect >= 2.4
            and (
                hh_i >= max(36, int(0.30 * h))
                or (border_hug and hh_i >= max(28, int(0.23 * h)))
            )
        )

        # Bottom region (curb area): require taller components,
        # but allow wide near-horizontal bands even if short in height.
        if cy > h * 0.50:
            keep = lower_lane_like and (
                ch_i >= 14
                or wide_sparse
                or horiz_ok
            )
        elif cy < h * 0.35:
            keep = lower_lane_like and (
                upper_narrow
                or (wide_sparse and aspect >= 2.0)
                or (y_i >= int(0.10 * h) and ch_i >= 22 and aspect >= 2.2 and fill_ratio <= 0.72)
            )
        else:
            keep = lower_lane_like and (
                ch_i >= 14
                or wide_sparse
                or (ww_i >= max(20, int(0.10 * w)) and area >= 70 and aspect >= 1.6)
            )
        if keep:
            clean[comp] = 255

    clean = _recover_rrl_upper_adjacent_segments(result, clean, v, contrast)
    clean = _recover_rrl_curve_continuation_segments(encoded, clean, v, contrast)
    clean = _recover_rrl_candidate_arc_bands(cand_u8 > 0, clean, v, contrast)

    horizontal = _detect_rrl_horizontal_supplement(
        (cand_u8 > 0) & road_mask & ~checker_mask,
        value_channel=v,
        contrast_map=contrast,
        guide_mask=clean > 0,
    )
    if np.any(horizontal):
        clean[horizontal] = 255
    if not np.any(clean):
        clean = _bootstrap_rrl_empty_detection(result > 0, v, contrast, road_mask)

    return clean


def _rrl_blue_line_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Use the detected white-line width directly instead of globally thickening it.

    `detect_rrl_white_lines()` already encodes each row by its actual cluster
    span (+ a tiny padding), so near thick lines stay thick and far thin lines
    stay thin. A later fixed dilation was making all lines too wide.
    """
    return detect_rrl_white_lines(img_bgr) > 0


# ---------------------------------------------------------------------------
# Checker detection
# ---------------------------------------------------------------------------
def detect_rrl_checker(img_bgr: np.ndarray) -> np.ndarray:
    """Detect black/white checkered start/finish line.

    Requires a horizontal band with both very dark and very bright
    low-saturation pixels and >= 6 B/W transitions per row, sustained
    over >= 5 consecutive rows.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1]

    checker_rows = np.zeros(h, dtype=bool)
    for y in range(int(0.3 * h), h):
        row_s = s[y, :]
        row_v = v[y, :]
        low_sat = row_s < 30

        n_bright = np.sum((row_v > 200) & low_sat)
        n_dark = np.sum((row_v < 50) & low_sat)
        if n_bright < 0.08 * w or n_dark < 0.08 * w:
            continue

        low_sat_idx = np.where(low_sat)[0]
        if len(low_sat_idx) < 0.4 * w:
            continue
        row_binary = (row_v[low_sat_idx] > 120).astype(np.int8)
        transitions = np.sum(np.abs(np.diff(row_binary)))
        if transitions >= 6:
            checker_rows[y] = True

    # Require >= 5 consecutive checker rows
    checker_mask = np.zeros((h, w), dtype=np.uint8)
    run_start = -1
    for y in range(h + 1):
        in_run = y < h and checker_rows[y]
        if in_run:
            if run_start < 0:
                run_start = y
        else:
            if run_start >= 0 and (y - run_start) >= 5:
                for yy in range(run_start, min(y, h)):
                    row_s = s[yy, :]
                    row_v = v[yy, :]
                    is_checker = (row_s < 30) & ((row_v > 180) | (row_v < 60))
                    checker_mask[yy, is_checker] = 255
            run_start = -1

    if np.any(checker_mask):
        checker_mask = cv2.morphologyEx(checker_mask, cv2.MORPH_CLOSE, _kernel(5, 3))
    return checker_mask


# ---------------------------------------------------------------------------
# Road surface detection
# ---------------------------------------------------------------------------
def detect_rrl_road_surface(img_bgr: np.ndarray) -> np.ndarray:
    """Detect the dark asphalt road. Low saturation, wide V range."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    road = ((s < 50) & (v >= 25) & (v <= 200)).astype(np.uint8) * 255
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, _kernel(7))
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, _kernel(3))
    return road


# ---------------------------------------------------------------------------
# Colored background detection
# ---------------------------------------------------------------------------
def detect_rrl_colored_bg(img_bgr: np.ndarray) -> np.ndarray:
    """Detect colored background (blue barriers, green areas, etc.)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    blue_bg = cv2.inRange(hsv, (85, 35, 30), (135, 255, 200))
    green_bg = cv2.inRange(hsv, (35, 35, 30), (85, 255, 200))
    red_bg = cv2.inRange(hsv, (0, 50, 30), (12, 255, 200))
    bg = cv2.bitwise_or(blue_bg, green_bg)
    bg = cv2.bitwise_or(bg, red_bg)
    return bg


# ---------------------------------------------------------------------------
# Shade field
# ---------------------------------------------------------------------------
def _lowfreq_value_field(v_channel: np.ndarray) -> np.ndarray:
    h, w = v_channel.shape
    sigma = max(3.0, 0.10 * float(min(h, w)))
    return cv2.GaussianBlur(
        v_channel.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma
    )


# ---------------------------------------------------------------------------
# HSV rendering
# ---------------------------------------------------------------------------
def _paint_flat(
    out_hsv: np.ndarray,
    mask: np.ndarray,
    base_hsv: np.ndarray,
    shade: np.ndarray,
    shade_ref: float,
    value_gain: float,
) -> None:
    if not np.any(mask):
        return
    out_hsv[:, :, 0][mask] = float(base_hsv[0])
    out_hsv[:, :, 1][mask] = float(base_hsv[1])
    out_hsv[:, :, 2][mask] = np.clip(
        float(base_hsv[2]) + value_gain * (shade[mask] - shade_ref), 0.0, 255.0
    )


def _paint_textured(
    out_hsv: np.ndarray,
    mask: np.ndarray,
    base_hsv: np.ndarray,
    src_v: np.ndarray,
    shade: np.ndarray,
    shade_ref: float,
    value_gain: float = 0.52,
    texture_scale: float = 0.30,
) -> None:
    """Paint with target color but preserve scaled high-frequency texture.

    Extracts high-freq detail from source V channel and adds it (scaled down)
    on top of the target color + low-freq shading. This gives a cloth-like
    appearance instead of the flat paint look.
    """
    if not np.any(mask):
        return
    # High-freq texture = original V - local mean
    hf = src_v - shade
    out_hsv[:, :, 0][mask] = float(base_hsv[0])
    out_hsv[:, :, 1][mask] = np.clip(
        float(base_hsv[1]) + texture_scale * 0.3 * hf[mask], 0.0, 60.0
    )
    out_hsv[:, :, 2][mask] = np.clip(
        float(base_hsv[2])
        + value_gain * (shade[mask] - shade_ref)
        + texture_scale * hf[mask],
        0.0, 255.0,
    )


def _blend_to_proto(
    hsv: np.ndarray,
    mask: np.ndarray,
    proto_hsv: np.ndarray,
    alpha_h: float,
    alpha_s: float,
    alpha_v: float,
) -> None:
    if not np.any(mask):
        return
    h_ch = hsv[:, :, 0]
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    th, ts, tv = float(proto_hsv[0]), float(proto_hsv[1]), float(proto_hsv[2])
    h_old = h_ch[mask]
    d = th - h_old
    d = np.where(d > 90.0, d - 180.0, d)
    d = np.where(d < -90.0, d + 180.0, d)
    h_ch[mask] = np.mod(h_old + alpha_h * d, 180.0)
    s_ch[mask] = np.clip(s_ch[mask] * (1.0 - alpha_s) + ts * alpha_s, 0.0, 255.0)
    v_ch[mask] = np.clip(v_ch[mask] * (1.0 - alpha_v) + tv * alpha_v, 0.0, 255.0)


# ---------------------------------------------------------------------------
# Main transforms
# ---------------------------------------------------------------------------
def transform_rrl_to_newtrack(img_bgr: np.ndarray) -> np.ndarray:
    """Full style transfer: re-render entire image in new_track style."""
    src_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, w = img_bgr.shape[:2]

    # Detect semantic regions
    white_mask = _rrl_blue_line_mask(img_bgr)
    checker_mask = detect_rrl_checker(img_bgr) > 0
    colored_bg = detect_rrl_colored_bg(img_bgr) > 0
    road_raw = detect_rrl_road_surface(img_bgr) > 0

    # 绿车区域：最高优先级，先从所有语义区域中排除
    green_result = _green_detector.detect(img_bgr)
    green_mask = green_result.mask > 0

    # Priority-based classification
    assigned = green_mask.copy()   # 绿车像素不参与后续语义渲染

    blue_line_region = white_mask & ~assigned
    assigned |= blue_line_region

    checker_region = checker_mask & ~assigned
    assigned |= checker_region

    road_region = road_raw & ~assigned
    assigned |= road_region

    bg_colored = colored_bg & ~assigned
    assigned |= bg_colored

    bg_remaining = ~assigned
    bg_mask = bg_colored | bg_remaining

    # Shade field
    shade = _lowfreq_value_field(src_hsv[:, :, 2])
    road_ref = float(np.mean(shade[road_region])) if np.any(road_region) else float(np.mean(shade))
    bg_ref = float(np.mean(shade[bg_mask])) if np.any(bg_mask) else road_ref

    # Build output
    out_hsv = np.zeros_like(src_hsv, dtype=np.float32)
    src_v = src_hsv[:, :, 2]

    _paint_textured(out_hsv, bg_mask, TARGET_BG_HSV, src_v, shade, bg_ref,
                    value_gain=0.25, texture_scale=0.18)
    _paint_textured(out_hsv, road_region, TARGET_CLOTH_HSV, src_v, shade, road_ref,
                    value_gain=0.40, texture_scale=0.22)
    _paint_flat(out_hsv, blue_line_region, TARGET_BLUE_HSV, shade, road_ref, value_gain=0.16)

    if np.any(checker_region):
        checker_bright = checker_region & (src_v > 130)
        checker_dark = checker_region & (src_v <= 130)
        _paint_textured(out_hsv, checker_bright, TARGET_CLOTH_HSV, src_v, shade, road_ref,
                        value_gain=0.3, texture_scale=0.20)
        _paint_flat(out_hsv, checker_dark, TARGET_BLUE_HSV, shade, road_ref, value_gain=0.16)

    result = cv2.cvtColor(out_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 回填绿车原始像素
    if np.any(green_mask):
        result[green_mask] = img_bgr[green_mask]
    return result


def transform_rrl_lines_only(img_bgr: np.ndarray) -> np.ndarray:
    """Minimal: only recolor white lines to blue, keep everything else."""
    src_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    white_mask = _rrl_blue_line_mask(img_bgr)

    _blend_to_proto(
        src_hsv, white_mask, TARGET_BLUE_HSV,
        alpha_h=0.97, alpha_s=0.90, alpha_v=0.65,
    )
    return cv2.cvtColor(src_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------
def process_directory(
    raw_dir: str,
    out_dir: str,
    mode: str = "full",
    save_masks: bool = False,
    save_preview: bool = True,
) -> None:
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if save_masks:
        for d in ["white_mask", "checker_mask"]:
            (out_path / d).mkdir(exist_ok=True)

    files = sorted(raw_path.glob("*.png"))
    if not files:
        files = sorted(raw_path.glob("*.jpg"))
    print(f"Processing {len(files)} images from {raw_dir} (mode={mode})")

    transform_fn = transform_rrl_to_newtrack if mode == "full" else transform_rrl_lines_only

    previews = []
    for i, fpath in enumerate(files):
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            continue

        result = transform_fn(img)
        cv2.imwrite(str(out_path / fpath.name), result)

        if save_masks:
            white = detect_rrl_white_lines(img)
            checker = detect_rrl_checker(img)
            cv2.imwrite(str(out_path / "white_mask" / fpath.name), white)
            cv2.imwrite(str(out_path / "checker_mask" / fpath.name), checker)

        if save_preview and i in (0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1):
            previews.append((fpath.stem, img, result))

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} done")

    if save_preview and previews:
        _save_preview(previews, out_path / "preview_compare.jpg")
    print(f"All done. Output: {out_dir}")


def _save_preview(previews, out_path):
    rows = []
    for name, orig, conv in previews:
        orig_labeled = orig.copy()
        conv_labeled = conv.copy()
        cv2.putText(orig_labeled, f"RRL: {name}", (5, 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(conv_labeled, f"NT-style: {name}", (5, 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        row = np.hstack([orig_labeled, conv_labeled])
        rows.append(row)
    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)
    print(f"  Preview saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform rrl images to new_track style"
    )
    parser.add_argument("--raw_dir", default=str(DEFAULT_RRL_RAW_DIR))
    parser.add_argument("--out_dir", default=str(DEFAULT_RRL_OUT_DIR))
    parser.add_argument("--mode", choices=["full", "lines"], default="full")
    parser.add_argument("--save_masks", action="store_true")
    parser.add_argument("--no_preview", action="store_true")
    args = parser.parse_args()
    process_directory(
        args.raw_dir, args.out_dir,
        mode=args.mode,
        save_masks=args.save_masks,
        save_preview=not args.no_preview,
    )


if __name__ == "__main__":
    main()
