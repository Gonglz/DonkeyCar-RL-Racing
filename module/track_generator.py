#!/usr/bin/env python3
"""
Track Point Generator — module-local GT / WS profiles
=====================================================
Loads the module-local GT / WS ``manual_width_*.json`` profiles and exposes
centre-line, left/right boundaries, and a CTE calculator that matches the
DonkeySimulator real-time telemetry value.

Verified conventions
--------------------
  coord_scale = 8.0  for every scene (sim telemetry CTE multiplier)

  sim_cte (telemetry) = -(signed_dist_from_centreline_sim) × coord_scale

  Sign  : positive = car is to the RIGHT of the centreline
          negative = car is to the LEFT
  Error : ≤ 0.1 CTE units on-track (quantisation from discrete centreline)
          ~3 % off-track (continuous sim geometry vs discrete points)

Scene ↔ gym-env mapping
-----------------------
  donkey-generated-track-v0          → generated_track
  donkey-waveshare-v0                → waveshare

Quick usage
-----------
  from module.track_generator import load_track

  tp = load_track("donkey-generated-track-v0")      # by gym env name
  tp = load_track("waveshare")                      # by scene name
  tp = load_track("/abs/path/to/manual_width_x.json") # by file path

  cte  = tp.sim_cte(car_x, car_z)         # matches info['cte']
  dist = tp.dist_sim(car_x, car_z)        # signed physical distance (sim)
  idx  = tp.nearest_idx(car_x, car_z)     # nearest centreline index

  for x, z in tp.centerline():  ...
  for x, z in tp.left_boundary():  ...
  for x, z in tp.right_boundary():  ...
  for i, cx,cz,lx,lz,rx,rz in tp.points_with_boundaries():  ...
"""

from __future__ import annotations

import json
import math
import os
from typing import List, Tuple

# ── KD-tree for fast nearest-neighbour (falls back to linear scan) ──────────
try:
    from scipy.spatial import KDTree as _KDTree
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

# ── Registry ─────────────────────────────────────────────────────────────────
# Module-local canonical GT / WS track data
_SCENE_TO_FILE: dict[str, str] = {
    "generated_track":       "manual_width_generated_track.json",
    "waveshare":             "manual_width_waveshare.json",
}

_GYM_ENV_TO_SCENE: dict[str, str] = {
    "donkey-generated-track-v0":         "generated_track",
    "donkey-waveshare-v0":               "waveshare",
}

_DEFAULT_TRACK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "track_data")


def load_track(name: str, track_dir: str = _DEFAULT_TRACK_DIR) -> "TrackProfile":
    """Load a TrackProfile by gym env name, scene name, or absolute JSON path.

    Examples
    --------
    load_track("donkey-generated-track-v0")
    load_track("waveshare")
    load_track("/path/to/manual_width_waveshare.json")
    """
    # 1. absolute / relative path
    if os.path.isfile(name):
        return TrackProfile(name)
    # 2. gym env id
    if name in _GYM_ENV_TO_SCENE:
        name = _GYM_ENV_TO_SCENE[name]
    # 3. scene name
    if name in _SCENE_TO_FILE:
        path = os.path.join(track_dir, _SCENE_TO_FILE[name])
        return TrackProfile(path)
    raise ValueError(
        f"Unknown track '{name}'. "
        f"Valid gym env ids: {sorted(_GYM_ENV_TO_SCENE)}\n"
        f"Valid scene names: {sorted(_SCENE_TO_FILE)}"
    )


def available_scenes() -> List[str]:
    """Return list of scene names that have a JSON profile."""
    return sorted(_SCENE_TO_FILE.keys())


def available_gym_envs() -> List[str]:
    """Return list of gym env ids that have a JSON profile."""
    return sorted(_GYM_ENV_TO_SCENE.keys())


# ── Core class ────────────────────────────────────────────────────────────────

class TrackProfile:
    """Centreline, boundaries, and CTE computation for one DonkeySim scene."""

    def __init__(self, json_path: str):
        with open(json_path) as f:
            raw = json.load(f)

        self.scene: str = raw["scene"]
        # CTE multiplier: |sim_cte| = physical_dist_sim × coord_scale
        self.coord_scale: float = raw["coord_scale"]

        outline = raw["outline"]
        self._center: List[Tuple[float, float]] = outline["fine_track_xz"]
        self._left:   List[Tuple[float, float]] = outline["left_boundary_xz"]
        self._right:  List[Tuple[float, float]] = outline["right_boundary_xz"]
        self._polygon:List[Tuple[float, float]] = outline["track_polygon_xz"]

        probe_summary = raw["manual_width_probe"]["summary"]
        self.left_width_sim:  float = probe_summary["outline_left_width_sim"]
        self.right_width_sim: float = probe_summary["outline_right_width_sim"]
        self.total_width_sim: float = self.left_width_sim + self.right_width_sim
        self.symmetric: bool = probe_summary["outline_inferred_symmetric"]

        self.num_points: int = len(self._center)

        # Perimeter: sum of all arc segments (handles non-uniform spacing)
        self.perimeter_sim: float = sum(
            math.sqrt(
                (self._center[i][0] - self._center[i-1][0]) ** 2 +
                (self._center[i][1] - self._center[i-1][1]) ** 2
            )
            for i in range(1, self.num_points)
        )

        # Typical step size (median to avoid outliers from non-uniform tracks)
        if self.num_points >= 2:
            dx = self._center[1][0] - self._center[0][0]
            dz = self._center[1][1] - self._center[0][1]
            _step0 = math.sqrt(dx*dx + dz*dz)
            # use arc / n as a better estimate when the first step is tiny
            _arc_avg = self.perimeter_sim / max(self.num_points - 1, 1)
            self.step_sim: float = _arc_avg if _step0 < _arc_avg * 0.1 else _step0
        else:
            self.step_sim = 0.0

        # Build KD-tree for O(log n) nearest-point lookup
        pts = list(self._center)
        if _HAVE_SCIPY:
            import numpy as np
            self._kdtree = _KDTree(np.array(pts, dtype=float))
        else:
            self._kdtree = None

    # ── Nearest-point lookup ──────────────────────────────────────────────────

    def nearest_idx(self, x: float, z: float) -> int:
        """Index of the centreline point nearest to (x, z)."""
        if self._kdtree is not None:
            _, idx = self._kdtree.query([x, z])
            return int(idx)
        # linear fallback
        best_i, best_d2 = 0, float("inf")
        for i, (cx, cz) in enumerate(self._center):
            d2 = (cx - x)**2 + (cz - z)**2
            if d2 < best_d2:
                best_d2, best_i = d2, i
        return best_i

    # ── CTE computation ───────────────────────────────────────────────────────

    def dist_sim(self, x: float, z: float) -> Tuple[float, int]:
        """(signed_dist_sim, nearest_idx).

        signed_dist > 0  →  car is to the LEFT  of the centreline (CCW).
        Negate and multiply by coord_scale to get sim telemetry CTE.
        """
        idx = self.nearest_idx(x, z)
        n = self.num_points
        # centred tangent
        px, pz = self._center[(idx - 1) % n]
        nx, nz = self._center[(idx + 1) % n]
        tx, tz = nx - px, nz - pz
        tlen = math.sqrt(tx*tx + tz*tz)
        if tlen > 1e-12:
            tx, tz = tx / tlen, tz / tlen
        # CCW left-normal
        lnx, lnz = -tz, tx
        cx, cz = self._center[idx]
        vx, vz = x - cx, z - cz
        unsigned = math.sqrt(vx*vx + vz*vz)
        sign = 1.0 if (vx*lnx + vz*lnz) >= 0 else -1.0
        return sign * unsigned, idx

    def sim_cte(self, x: float, z: float) -> float:
        """CTE value in the same units as DonkeySimulator's info['cte'].

        Positive = car to the RIGHT.  Negative = car to the LEFT.
        """
        d, _ = self.dist_sim(x, z)
        return -d * self.coord_scale

    # ── Generators ────────────────────────────────────────────────────────────

    def centerline(self):
        """Yield (x, z) sim coordinates along the track centreline."""
        yield from self._center

    def left_boundary(self):
        """Yield (x, z) sim coordinates along the left track boundary."""
        yield from self._left

    def right_boundary(self):
        """Yield (x, z) sim coordinates along the right track boundary."""
        yield from self._right

    def polygon(self):
        """Yield (x, z) coordinates of the closed track polygon."""
        yield from self._polygon

    def points_with_boundaries(self):
        """Yield (idx, cx, cz, lx, lz, rx, rz) for every track index."""
        for i, ((cx, cz), (lx, lz), (rx, rz)) in enumerate(
            zip(self._center, self._left, self._right)
        ):
            yield i, cx, cz, lx, lz, rx, rz

    # ── Track geometry helpers ────────────────────────────────────────────────

    def tangent_at(self, idx: int) -> Tuple[float, float]:
        """Unit tangent (tx, tz) at centreline index idx."""
        n = self.num_points
        px, pz = self._center[(idx - 1) % n]
        nx, nz = self._center[(idx + 1) % n]
        dx, dz = nx - px, nz - pz
        length = math.sqrt(dx*dx + dz*dz)
        return (dx/length, dz/length) if length > 1e-12 else (0.0, 0.0)

    def normal_left_at(self, idx: int) -> Tuple[float, float]:
        """Unit left-normal (CCW from tangent) at centreline index idx."""
        tx, tz = self.tangent_at(idx)
        return (-tz, tx)

    def arc_length_to(self, idx: int) -> float:
        """Cumulative arc length from index 0 to idx (sim units)."""
        total = 0.0
        for i in range(1, idx + 1):
            px, pz = self._center[i - 1]
            cx_, cz_ = self._center[i]
            total += math.sqrt((cx_ - px)**2 + (cz_ - pz)**2)
        return total

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [
            f"Scene         : {self.scene}",
            f"JSON source   : (loaded)",
            f"coord_scale   : {self.coord_scale}  "
            f"(sim_cte = -dist_sim × {self.coord_scale})",
            f"Points        : {self.num_points}",
            f"Typical step  : {self.step_sim:.5f} sim units",
            f"Perimeter     : {self.perimeter_sim:.3f} sim units",
            f"Width  left   : {self.left_width_sim:.4f} sim units"
            f"  ({self.left_width_sim * self.coord_scale:.2f} CTE units)",
            f"Width  right  : {self.right_width_sim:.4f} sim units"
            f"  ({self.right_width_sim * self.coord_scale:.2f} CTE units)",
            f"Total width   : {self.total_width_sim:.4f} sim units",
            f"Symmetric     : {self.symmetric}",
            f"KD-tree       : {'yes (scipy)' if self._kdtree is not None else 'no (linear fallback)'}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"TrackProfile(scene={self.scene!r}, pts={self.num_points})"


# ── CLI demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== Available scenes ===")
    for s in available_scenes():
        print(f"  {s}")

    print("\n=== Available gym env ids ===")
    for e in available_gym_envs():
        print(f"  {e}")

    # Load a specific track from CLI arg or default
    name = sys.argv[1] if len(sys.argv) > 1 else "generated_track"
    print(f"\n=== Loading '{name}' ===")
    tp = load_track(name)
    print(tp.summary())

    print("\n--- First 5 centreline points ---")
    for i, (x, z) in enumerate(tp.centerline()):
        if i >= 5:
            break
        tx, tz = tp.tangent_at(i)
        cte_here = tp.sim_cte(x, z)
        print(f"  [{i:4d}]  ({x:.4f}, {z:.4f})  tan=({tx:.4f},{tz:.4f})  cte_at_centre={cte_here:.4f}")

    print("\n--- First 3 cross-sections (centre / left / right) ---")
    for i, cx, cz, lx, lz, rx, rz in tp.points_with_boundaries():
        if i >= 3:
            break
        dl = math.sqrt((lx-cx)**2 + (lz-cz)**2)
        dr = math.sqrt((rx-cx)**2 + (rz-cz)**2)
        cte_l = tp.sim_cte(lx, lz)
        cte_r = tp.sim_cte(rx, rz)
        print(f"  [{i}]  C=({cx:.3f},{cz:.3f})"
              f"  L=({lx:.3f},{lz:.3f}) d={dl:.3f} cte={cte_l:.2f}"
              f"  R=({rx:.3f},{rz:.3f}) d={dr:.3f} cte={cte_r:.2f}")
