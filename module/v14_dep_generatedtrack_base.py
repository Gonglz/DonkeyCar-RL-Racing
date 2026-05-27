#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V11.2: generated_track note + fixed2 staticobstacletraining
(target-controller + note + RecurrentPPO/LSTM)

notegoal:
1) note reset/note Ego/NPC note
2) note generated_track note(telemetrynote)
3) notestablenote telemetry/info noterewardnote
4) note RecurrentPPO(CnnLstmPolicy) notetraining

note:
  # note(notetraining)
  python ppo_generatedtrack_v11_2_randomspawn_fixednpc.py --mode calibrate

  # resetnote(notespawnstablenote)
  python ppo_generatedtrack_v11_2_randomspawn_fixednpc.py --mode reset-stress --stress-resets 200

  # fixed2 training(note sb3_contrib==1.8.0)
  python ppo_generatedtrack_v11_2_randomspawn_fixednpc.py --mode train --v11-2-mode fixed2 --total-steps 300000
"""

import os
import sys
import time
import math
import json
import random
import csv
import argparse
import glob
import re
import hashlib
import platform
import subprocess
import importlib
from dataclasses import dataclass, asdict
from collections import deque, Counter
from typing import Optional

import numpy as np
import gym
import gym_donkeycar  # noqa: F401
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# note v9 note(note/NPCcontrol/note/CNN)
from.v14_dep_sim_core import (  # noqa: E402
    SimExtendedAPI,
    TrackNodeCache,
    NPCController,
    OvertakeTrainingWrapper,
    LightweightCNN,
)


OBS_FIELD_CAPS = {
    "stable": ["cte", "speed", "hit", "lap_count", "activeNode", "pos"],
    "fallbacks": {
        "pos": ["pos", "pos_x/pos_z"],
        "activeNode": ["activeNode", "last_active_node"],
    },
}

DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK = {
    "scene": "generated_track",
    "coord_system": "telemetry",
    "lap_length_sim": None,
    "avg_node_gap_sim": None,
    "avg_fine_gap_sim": None,
    # note(notedefault)
    "spawn_min_gap_sim": 3.5,
    "spawn_min_gap_progress": 30,   # fine_track index gap
    "danger_close_sim": 1.2,
    "follow_safe_min_sim": 1.6,
    "follow_safe_max_sim": 4.0,
    "pass_window_sim": 2.2,
    # v10.1: lane offset note(notetracknote)
    "lane_offset_center_sim": 0.0,
    "lane_offset_left_sim": 0.55,
    "lane_offset_right_sim": -0.55,
    "lane_offset_safe_jitter_sim": 0.12,
    "lane_offset_calibration_fallback": True,
    # v10.1: spawn note pair type note(None note)
    "spawn_min_gap_sim_ego_npc": None,
    "spawn_min_gap_progress_ego_npc": None,
    "spawn_min_gap_sim_npc_npc": None,
    "spawn_min_gap_progress_npc_npc": None,
}


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def _parse_float_list(raw, default_values):
    """Parse comma-separated float list from CLI string."""
    if isinstance(raw, (list, tuple)):
        vals = []
        for it in raw:
            try:
                vals.append(float(it))
            except Exception:
                pass
        if vals:
            return vals
    s = str(raw or "").strip()
    if not s:
        return [float(v) for v in default_values]
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except Exception:
            continue
    if not out:
        out = [float(v) for v in default_values]
    return out


def _apply_global_seeds(seed):
    s = int(_safe_float(seed, 42))
    random.seed(s)
    np.random.seed(s)
    try:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
    except Exception:
        pass
    return s


def _sha256_file(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _safe_module_version(name):
    try:
        mod = importlib.import_module(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception:
        return "unavailable"


def _git_metadata_for_path(path):
    out = {
        "repo_root": "",
        "commit": "",
        "branch": "",
        "dirty": None,
    }
    p = os.path.abspath(path)
    start = os.path.dirname(p) if os.path.isfile(p) else p
    try:
        root = subprocess.check_output(
            ["git", "-C", start, "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        out["repo_root"] = str(root)
        out["commit"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "HEAD"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        out["branch"] = subprocess.check_output(
            ["git", "-C", root, "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root, "status", "--porcelain"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        out["dirty"] = bool(status.strip())
    except Exception:
        pass
    return out


def _write_json(path, payload):
    if not path:
        return
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"🧾 noterowsnotedatanote: {abs_path}")
    except Exception as e:
        print(f"⚠️ noterowsnotedatafailed({path}): {e}")


class ManualWidthSpawnSampler(object):
    """Sample in-track spawn poses from manual width profile outline."""

    def __init__(self, profile_path, scene_name="generated_track"):
        self.profile_path = str(profile_path)
        self.scene_name = str(scene_name)
        self.loaded = False
        self.error = None

        self.coord_scale = float(SimExtendedAPI.COORD_SCALE)
        self.fine_track = np.zeros((0, 2), dtype=np.float64)
        self.left_boundary = np.zeros((0, 2), dtype=np.float64)
        self.right_boundary = np.zeros((0, 2), dtype=np.float64)
        self.width_sim = np.zeros((0,), dtype=np.float64)
        self.half_width_cte = np.zeros((0,), dtype=np.float64)
        # note/note(CTEnote): tracknoteontracknote
        self.half_width_left_cte = np.zeros((0,), dtype=np.float64)   # note->note
        self.half_width_right_cte = np.zeros((0,), dtype=np.float64)  # note->note
        self.half_width_narrow_cte = np.zeros((0,), dtype=np.float64) # min(left, right): note
        self.half_width_wide_cte = np.zeros((0,), dtype=np.float64)   # max(left, right): note
        self.fine_gap_sim = 0.025
        self.width_median_sim = 1.0

        self._load()

    @staticmethod
    def _as_points(arr):
        if not isinstance(arr, list):
            return np.zeros((0, 2), dtype=np.float64)
        out = []
        for p in arr:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                x = _safe_float(p[0], np.nan)
                z = _safe_float(p[1], np.nan)
                if np.isfinite(x) and np.isfinite(z):
                    out.append((float(x), float(z)))
        if not out:
            return np.zeros((0, 2), dtype=np.float64)
        return np.asarray(out, dtype=np.float64)

    def _load(self):
        try:
            with open(self.profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            scene = str(data.get("scene", self.scene_name))
            if scene!= self.scene_name:
                raise ValueError("scene mismatch: %s!= %s" % (scene, self.scene_name))
            outline = data.get("outline", {})
            self.fine_track = self._as_points(outline.get("fine_track_xz", []))
            self.left_boundary = self._as_points(outline.get("left_boundary_xz", []))
            self.right_boundary = self._as_points(outline.get("right_boundary_xz", []))
            self.coord_scale = float(_safe_float(data.get("coord_scale", SimExtendedAPI.COORD_SCALE), SimExtendedAPI.COORD_SCALE))

            n = int(self.fine_track.shape[0])
            if n <= 8:
                raise ValueError("fine_track too short")
            if self.left_boundary.shape[0]!= n or self.right_boundary.shape[0]!= n:
                raise ValueError("boundary size mismatch")

            seg = np.roll(self.fine_track, -1, axis=0) - self.fine_track
            seglen = np.sqrt(np.sum(seg * seg, axis=1))
            seglen = seglen[np.isfinite(seglen)]
            if seglen.size == 0:
                raise ValueError("invalid fine gaps")
            self.fine_gap_sim = float(max(1e-4, np.median(seglen)))

            width = np.sqrt(np.sum((self.left_boundary - self.right_boundary) ** 2, axis=1))
            width = np.where(np.isfinite(width), width, 0.0)
            width = np.maximum(width, 1e-3)
            self.width_sim = width.astype(np.float64)
            self.width_median_sim = float(max(1e-3, np.median(self.width_sim)))
            # note sim note(outline_xz), note coord_scale note CTE note
            # info["cte"] note CTE note, ontrack note
            self.half_width_cte = (0.5 * self.width_sim * self.coord_scale).astype(np.float64)

            # note/note: note->note, note->note(CTEnote=sim*coord_scale)
            hw_left  = np.sqrt(np.sum((self.left_boundary  - self.fine_track) ** 2, axis=1))
            hw_right = np.sqrt(np.sum((self.right_boundary - self.fine_track) ** 2, axis=1))
            hw_left  = np.where(np.isfinite(hw_left),  hw_left,  0.5 * self.width_sim)
            hw_right = np.where(np.isfinite(hw_right), hw_right, 0.5 * self.width_sim)
            # sim->CTE: note coord_scale
            self.half_width_left_cte   = (hw_left  * self.coord_scale).astype(np.float64)
            self.half_width_right_cte  = (hw_right * self.coord_scale).astype(np.float64)
            self.half_width_narrow_cte = np.minimum(self.half_width_left_cte, self.half_width_right_cte).astype(np.float64)
            self.half_width_wide_cte   = np.maximum(self.half_width_left_cte, self.half_width_right_cte).astype(np.float64)

            self.loaded = True
            self.error = None
        except Exception as e:
            self.loaded = False
            self.error = str(e)

    def local_window_idx(self, window_m, min_idx):
        n = int(self.fine_track.shape[0])
        if n <= 1:
            return int(max(4, min_idx))
        w = int(round(float(window_m) / max(1e-4, float(self.fine_gap_sim))))
        w = max(int(min_idx), w)
        w = min(w, max(4, n // 2))
        return int(w)

    def nearest_fine_idx_local(self, x, z, seed_idx, window_idx):
        n = int(self.fine_track.shape[0])
        if n <= 0:
            return 0, float("inf")
        seed = int(seed_idx) % n
        w = max(4, min(int(window_idx), n // 2))
        best_i = seed
        best_d2 = float("inf")
        for off in range(-w, w + 1):
            i = (seed + off) % n
            dx = float(self.fine_track[i, 0]) - float(x)
            dz = float(self.fine_track[i, 1]) - float(z)
            d2 = dx * dx + dz * dz
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return int(best_i), float(math.sqrt(best_d2))

    def tangent(self, fi):
        n = int(self.fine_track.shape[0])
        i = int(fi) % n
        p0 = self.fine_track[(i - 1) % n]
        p1 = self.fine_track[(i + 1) % n]
        tx = float(p1[0] - p0[0])
        tz = float(p1[1] - p0[1])
        norm = math.sqrt(tx * tx + tz * tz)
        if norm < 1e-9:
            return 0.0, 1.0
        return tx / norm, tz / norm

    def estimate_kappa(self, fi):
        n = int(self.fine_track.shape[0])
        i = int(fi) % n
        a = self.fine_track[(i - 1) % n]
        b = self.fine_track[i]
        c = self.fine_track[(i + 1) % n]
        bax = float(a[0] - b[0])
        baz = float(a[1] - b[1])
        bcx = float(c[0] - b[0])
        bcz = float(c[1] - b[1])
        la = math.sqrt(bax * bax + baz * baz)
        lc = math.sqrt(bcx * bcx + bcz * bcz)
        acx = float(c[0] - a[0])
        acz = float(c[1] - a[1])
        l_ac = math.sqrt(acx * acx + acz * acz)
        denom = max(1e-6, la * lc * l_ac)
        cross = abs(bax * bcz - baz * bcx)
        return float(2.0 * cross / denom)

    def half_width_cte_at(self, fi):
        if self.half_width_cte.size <= 0:
            return 1.0
        return float(self.half_width_cte[int(fi) % int(self.half_width_cte.shape[0])])

    def half_width_narrow_cte_at(self, fi):
        """note(min(note,note)), noteontracknote."""
        if self.half_width_narrow_cte.size > 0:
            return float(max(1e-3, self.half_width_narrow_cte[int(fi) % int(self.half_width_narrow_cte.shape[0])]))
        return self.half_width_cte_at(fi)

    def half_width_wide_cte_at(self, fi):
        """note(max(note,note)), notemax_ctenote."""
        if self.half_width_wide_cte.size > 0:
            return float(max(1e-3, self.half_width_wide_cte[int(fi) % int(self.half_width_wide_cte.shape[0])]))
        return self.half_width_cte_at(fi)

    def nearest_fine_idx_global(self, x, z):
        n = int(self.fine_track.shape[0])
        if n <= 0:
            return 0, float("inf")
        dx = self.fine_track[:, 0] - float(x)
        dz = self.fine_track[:, 1] - float(z)
        d2 = dx * dx + dz * dz
        i = int(np.argmin(d2))
        return i, float(math.sqrt(float(d2[i])))

    def classify_point(self, x, z, margin_ratio=0.0, seed_idx=None, window_idx=None):
        """Classify whether point lies inside cross-section corridor at nearest fine index."""
        if not self.loaded:
            return {"inside": False, "reason": "profile_not_loaded"}
        n = int(self.fine_track.shape[0])
        if n <= 0:
            return {"inside": False, "reason": "empty_fine_track"}

        if seed_idx is None:
            fi, center_dist = self.nearest_fine_idx_global(x, z)
        else:
            w_idx = self.local_window_idx(2.0, 20) if window_idx is None else int(max(4, window_idx))
            fi, center_dist = self.nearest_fine_idx_local(x, z, int(seed_idx) % n, w_idx)

        rx, rz = self.right_boundary[fi]
        lx, lz = self.left_boundary[fi]
        vx = float(lx - rx)
        vz = float(lz - rz)
        vv = vx * vx + vz * vz
        if vv < 1e-12:
            return {
                "inside": False,
                "reason": "degenerate_cross_section",
                "fine_idx": int(fi),
                "center_dist_sim": float(center_dist),
            }

        px = float(x) - float(rx)
        pz = float(z) - float(rz)
        lane_ratio = (px * vx + pz * vz) / vv
        lane_ratio_clip = float(np.clip(lane_ratio, 0.0, 1.0))
        proj_x = float(rx) + lane_ratio_clip * vx
        proj_z = float(rz) + lane_ratio_clip * vz
        seg_dist = math.sqrt((float(x) - proj_x) ** 2 + (float(z) - proj_z) ** 2)

        margin = float(np.clip(float(margin_ratio), 0.0, 0.48))
        inside_lane = (lane_ratio >= margin) and (lane_ratio <= (1.0 - margin))
        # notegeometrynote: note, notefinenote
        seg_dist_cap = max(0.40, 0.35 * float(self.width_sim[fi]))
        inside = bool(inside_lane and (seg_dist <= seg_dist_cap))

        return {
            "inside": bool(inside),
            "reason": "ok" if inside else ("lane_ratio_outside" if not inside_lane else "far_from_cross_section"),
            "fine_idx": int(fi),
            "lane_ratio": float(lane_ratio),
            "lane_ratio_clipped": float(lane_ratio_clip),
            "center_dist_sim": float(center_dist),
            "cross_section_dist_sim": float(seg_dist),
            "cross_section_dist_cap_sim": float(seg_dist_cap),
            "signed_offset_sim": float((lane_ratio - 0.5) * float(self.width_sim[fi])),
            "width_sim": float(self.width_sim[fi]),
        }

    def bounds_summary(self):
        def _bounds(arr):
            if not isinstance(arr, np.ndarray) or arr.size <= 0:
                return {}
            return {
                "x_min": float(np.min(arr[:, 0])),
                "x_max": float(np.max(arr[:, 0])),
                "z_min": float(np.min(arr[:, 1])),
                "z_max": float(np.max(arr[:, 1])),
            }
        return {
            "centerline": _bounds(self.fine_track),
            "left_boundary": _bounds(self.left_boundary),
            "right_boundary": _bounds(self.right_boundary),
        }

    def export_intrack_samples(self, y_tel=0.0625, node_count=108, lane_fracs=None,
                               step_idx=3, margin_ratio=0.08, kappa_max=None):
        """Build deterministic in-track sample table from left/right boundaries."""
        if not self.loaded:
            return []
        n = int(self.fine_track.shape[0])
        if n <= 0:
            return []
        if lane_fracs is None:
            lane_fracs = [0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88]
        step_idx = max(1, int(step_idx))
        margin = float(np.clip(float(margin_ratio), 0.0, 0.45))

        cleaned_lane_fracs = []
        for u in lane_fracs:
            uu = float(np.clip(float(u), margin, 1.0 - margin))
            cleaned_lane_fracs.append(uu)
        cleaned_lane_fracs = sorted(set([round(u, 6) for u in cleaned_lane_fracs]))

        out = []
        for fi in range(0, n, step_idx):
            if kappa_max is not None and abs(self.estimate_kappa(fi)) > float(kappa_max):
                continue
            rx, rz = self.right_boundary[fi]
            lx, lz = self.left_boundary[fi]
            tx, tz = self.tangent(fi)
            yaw_deg = float(math.degrees(math.atan2(tx, tz)))
            qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
            node_idx = int(round(float(fi) * float(node_count) / float(max(1, n)))) % max(1, int(node_count))

            for ratio in cleaned_lane_fracs:
                x = float((1.0 - ratio) * float(rx) + ratio * float(lx))
                z = float((1.0 - ratio) * float(rz) + ratio * float(lz))
                dn = float((ratio - 0.5) * float(self.width_sim[fi]))
                out.append({
                    "fine_idx": int(fi),
                    "node_idx": int(node_idx),
                    "lane_ratio": float(ratio),
                    "lane_side": ("left" if ratio > 0.58 else ("right" if ratio < 0.42 else "center")),
                    "tel": [x, float(y_tel), z],
                    "yaw_deg": float(yaw_deg),
                    "node_coords": [
                        float(x * self.coord_scale), float(y_tel * self.coord_scale), float(z * self.coord_scale),
                        float(qx), float(qy), float(qz), float(qw),
                    ],
                    "dn_offset_sim": float(dn),
                    "kappa": float(self.estimate_kappa(fi)),
                    "width_sim": float(self.width_sim[fi]),
                })
        return out

    def sample_pose(self, y_tel, node_count, margin_ratio, kappa_max, window_m, window_min_idx,
                    anchor_fi=None, anchor_window_idx=None):
        if not self.loaded:
            return None
        n = int(self.fine_track.shape[0])
        w_idx = self.local_window_idx(window_m, window_min_idx)
        anchor_idx = None if anchor_fi is None else int(anchor_fi) % n
        anchor_w = None
        if anchor_idx is not None:
            if anchor_window_idx is None:
                anchor_w = max(4, int(round(3.0 / max(1e-4, float(self.fine_gap_sim)))))
            else:
                anchor_w = max(4, int(anchor_window_idx))
            anchor_w = min(anchor_w, max(4, n // 2))
        margin = float(np.clip(float(margin_ratio), 0.02, 0.45))
        for _ in range(max(16, n // 12)):
            if anchor_idx is None:
                fi = random.randint(0, n - 1)
            else:
                fi = (anchor_idx + random.randint(-anchor_w, anchor_w)) % n
            kappa = self.estimate_kappa(fi)
            if abs(kappa) > float(kappa_max):
                continue
            u = random.uniform(margin, 1.0 - margin)
            rx, rz = self.right_boundary[fi]
            lx, lz = self.left_boundary[fi]
            x = float((1.0 - u) * rx + u * lx)
            z = float((1.0 - u) * rz + u * lz)
            fi_chk, d = self.nearest_fine_idx_local(x, z, fi, w_idx)
            if d > max(0.35, 0.45 * self.width_median_sim):
                continue
            # v11.7: note, note"notetracknote"note.
            cls = self.classify_point(
                x, z,
                margin_ratio=max(0.03, margin * 0.70),
                seed_idx=fi_chk,
                window_idx=max(16, w_idx),
            )
            if not bool(cls.get("inside", False)):
                continue
            tx, tz = self.tangent(fi_chk)
            yaw_deg = math.degrees(math.atan2(tx, tz))
            qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
            if int(node_count) > 0:
                node_idx = int(round(float(fi_chk) * float(node_count) / float(n))) % int(node_count)
            else:
                node_idx = int(fi_chk)
            return {
                "node_idx": int(node_idx),
                "fine_idx": int(fi_chk),
                "tel": (x, float(y_tel), z),
                "node_coords": (
                    float(x * self.coord_scale),
                    float(y_tel * self.coord_scale),
                    float(z * self.coord_scale),
                    float(qx),
                    float(qy),
                    float(qz),
                    float(qw),
                ),
                "yaw_deg": float(yaw_deg),
                "lane_side": "center",
                "dn_offset_sim": float((u - 0.5) * self.width_sim[fi_chk]),
                "fine_dist_sim": float(d),
                "spawn_kappa": float(kappa),
                "spawn_margin_ratio": float(margin),
                "spawn_local_window_idx": int(w_idx),
            }
        return None


class FixedSuccessGuardState(object):
    """Windowed state for fixed2 success guard."""

    def __init__(self, hit_window_steps=200):
        self.hit_window_steps = int(max(10, hit_window_steps))
        self.hit_window = deque(maxlen=self.hit_window_steps)
        self.episode_time_sec = 0.0
        self.last_offtrack_ts = None
        self.offtrack_run_sec = 0.0

    def reset(self, hit_window_steps=None):
        if hit_window_steps is not None:
            self.hit_window_steps = int(max(10, hit_window_steps))
        self.hit_window = deque(maxlen=self.hit_window_steps)
        self.episode_time_sec = 0.0
        self.last_offtrack_ts = None
        self.offtrack_run_sec = 0.0

    def update(self, dt, hit_event, offtrack_flag):
        dtt = max(0.0, _safe_float(dt, 0.0))
        self.episode_time_sec += dtt
        self.hit_window.append(bool(hit_event))
        if bool(offtrack_flag):
            self.offtrack_run_sec += dtt
            self.last_offtrack_ts = float(self.episode_time_sec)
        else:
            self.offtrack_run_sec = 0.0

    def no_hit_window_ok(self):
        if not self.hit_window:
            return True
        return not any(self.hit_window)

    def no_offtrack_window_ok(self, seconds_required):
        req = max(0.0, _safe_float(seconds_required, 0.0))
        if self.last_offtrack_ts is None:
            return True
        return (self.episode_time_sec - float(self.last_offtrack_ts)) >= req


def build_dist_scale_profile(track_cache, scene_name="generated_track"):
    """note query note nodes/fine_track generatenote."""
    nodes = track_cache.nodes.get(scene_name, [])
    fine = track_cache.fine_track.get(scene_name, [])
    profile = dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK)

    if len(nodes) >= 2:
        S = SimExtendedAPI.COORD_SCALE
        node_gaps = []
        lap_length = 0.0
        for i in range(len(nodes)):
            j = (i + 1) % len(nodes)
            dx = (nodes[j][0] - nodes[i][0]) / S
            dz = (nodes[j][2] - nodes[i][2]) / S
            d = math.sqrt(dx * dx + dz * dz)
            node_gaps.append(d)
            lap_length += d
        profile["lap_length_sim"] = lap_length
        profile["avg_node_gap_sim"] = float(np.mean(node_gaps)) if node_gaps else None

    if len(fine) >= 2:
        fine_gaps = []
        for i in range(len(fine)):
            j = (i + 1) % len(fine)
            dx = fine[j][0] - fine[i][0]
            dz = fine[j][1] - fine[i][1]
            fine_gaps.append(math.sqrt(dx * dx + dz * dz))
        profile["avg_fine_gap_sim"] = float(np.mean(fine_gaps)) if fine_gaps else None

    # notetracknotedefaultnote(note sim note)
    avg_node_gap = profile.get("avg_node_gap_sim") or 0.5
    avg_fine_gap = profile.get("avg_fine_gap_sim") or 0.05

    # note(note)
    profile["spawn_min_gap_sim"] = max(3.0, 6.0 * avg_node_gap)
    # fine_tracknote, note
    profile["spawn_min_gap_progress"] = int(max(20, round(profile["spawn_min_gap_sim"] / max(avg_fine_gap, 1e-3))))

    # note/note, note
    profile["danger_close_sim"] = max(0.9, 2.0 * avg_node_gap)
    profile["follow_safe_min_sim"] = max(profile["danger_close_sim"] + 0.2, 2.8 * avg_node_gap)
    profile["follow_safe_max_sim"] = max(profile["follow_safe_min_sim"] + 1.5, 6.5 * avg_node_gap)
    profile["pass_window_sim"] = max(1.8, 4.0 * avg_node_gap)

    # v10.1: note lane offset notedefault(note)
    lane_half = max(0.45, 2.2 * avg_node_gap)
    profile["lane_offset_center_sim"] = 0.0
    profile["lane_offset_left_sim"] = +lane_half
    profile["lane_offset_right_sim"] = -lane_half
    profile["lane_offset_safe_jitter_sim"] = max(0.08, min(0.20, lane_half * 0.25))
    profile["lane_offset_calibration_fallback"] = True

    # v10.1: pair-wise spawn note(note B2 note)
    profile["spawn_min_gap_sim_ego_npc"] = profile["spawn_min_gap_sim"]
    profile["spawn_min_gap_progress_ego_npc"] = max(30, int(round(profile["spawn_min_gap_progress"] * 0.70)))
    profile["spawn_min_gap_sim_npc_npc"] = max(1.0, profile["spawn_min_gap_sim"] * 0.45)
    profile["spawn_min_gap_progress_npc_npc"] = max(12, int(round(profile["spawn_min_gap_progress"] * 0.28)))

    return profile


class StageConfig(object):
    def __init__(self, sid, name, reward_mode, npc_count=0, npc_mode="offtrack",
                 enable_overtake_reward=False, p_ego_behind=0.0, npc_speed_range=(0.0, 0.0),
                 min_stage_steps=20000):
        self.sid = sid
        self.name = name
        self.reward_mode = reward_mode
        self.npc_count = npc_count
        self.npc_mode = npc_mode
        self.enable_overtake_reward = enable_overtake_reward
        self.p_ego_behind = p_ego_behind
        self.npc_speed_range = npc_speed_range
        self.min_stage_steps = min_stage_steps


STAGES = {
    1: StageConfig(1, "Anote", "drive_only", npc_count=0, npc_mode="offtrack", min_stage_steps=80000),
    2: StageConfig(2, "B1staticnote2note", "avoid_static", npc_count=2, npc_mode="static", min_stage_steps=120000),
    3: StageConfig(3, "B2note2note", "avoid_static", npc_count=2, npc_mode="wobble", min_stage_steps=160000),
    4: StageConfig(4, "Cdynamicnote", "follow_only", npc_count=1, npc_mode="slow_policy",
                   p_ego_behind=0.8, npc_speed_range=(0.10, 0.22), min_stage_steps=240000),
    5: StageConfig(5, "Dnote", "overtake", npc_count=1, npc_mode="slow_policy",
                   enable_overtake_reward=True, p_ego_behind=0.8, npc_speed_range=(0.12, 0.30), min_stage_steps=320000),
}


def _max_npc_count_from_stage(start_stage):
    sid = int(max(1, min(int(start_stage), max(STAGES.keys()))))
    candidates = [int(cfg.npc_count) for s, cfg in STAGES.items() if int(s) >= sid]
    if not candidates:
        return int(STAGES[sid].npc_count)
    return int(max(candidates))


@dataclass
class StageTrainProfile(object):
    max_throttle: Optional[float] = None
    max_cte_limit: Optional[float] = None
    progress_reward_scale: Optional[float] = None
    progress_milestone_lap: Optional[float] = None
    progress_milestone_reward: Optional[float] = None
    progress_reward_decay_min: Optional[float] = None
    penalty_decay_min: Optional[float] = None
    progress_backward_penalty_scale: Optional[float] = None
    reverse_onset_penalty: Optional[float] = None
    reverse_streak_penalty_scale: Optional[float] = None
    reverse_backdist_penalty_scale: Optional[float] = None
    reverse_event_penalty: Optional[float] = None
    random_start_enabled: Optional[bool] = None
    npc_count: Optional[int] = None
    npc_layout_reset_policy: Optional[str] = None
    npc_layout_reset_every: Optional[int] = None
    npc_layout_reset_every_laps: Optional[int] = None
    npc_lane_offset_left_sim: Optional[float] = None
    npc_lane_offset_right_sim: Optional[float] = None
    npc_lane_jitter_sim: Optional[float] = None
    spawn_min_gap_sim_ego_npc: Optional[float] = None
    spawn_min_gap_progress_ego_npc: Optional[int] = None
    spawn_min_gap_sim_npc_npc: Optional[float] = None
    spawn_min_gap_progress_npc_npc: Optional[int] = None
    ent_coef: Optional[float] = None
    lr: Optional[float] = None
    stuck_counter_limit: Optional[int] = None
    offtrack_counter_limit: Optional[int] = None
    reward_clip_abs: Optional[float] = None
    disable_cte_auto_adjust: Optional[bool] = None
    reverse_reset_steps: Optional[int] = None
    negative_throttle_reset_speed_max: Optional[float] = None
    progress_full_lap_reward: Optional[float] = None


@dataclass
class StageEvalGate(object):
    min_stage_steps: int
    eval_every_steps: int = 20000
    eval_episodes: int = 3
    consecutive_passes_required: int = 2
    min_relative_progress_laps: float = 0.0
    require_lap_count_at_least: int = 0
    max_collisions: int = 0
    max_rear_end: int = 0
    allow_persistent_offtrack: bool = False
    allow_stuck: bool = False
    max_reverse_events: Optional[int] = None
    min_npc_encounters: int = 0
    min_safe_overtakes: int = 0
    max_unsafe_cutin: Optional[int] = None
    max_unsafe_follow_ratio: Optional[float] = None


STAGE_TRAIN_PROFILES = {
    1: StageTrainProfile(
        npc_count=0, random_start_enabled=False, max_throttle=0.80, max_cte_limit=8.0,
        progress_reward_scale=1.40, progress_milestone_lap=0.04, progress_milestone_reward=2.00,
        progress_reward_decay_min=0.70, penalty_decay_min=0.85,
        progress_backward_penalty_scale=0.55, reverse_onset_penalty=0.18,
        reverse_streak_penalty_scale=0.06, reverse_backdist_penalty_scale=2.00, reverse_event_penalty=1.20,
        ent_coef=0.008, lr=1.5e-4,
        stuck_counter_limit=25, offtrack_counter_limit=25, reward_clip_abs=10.0,
        disable_cte_auto_adjust=True, reverse_reset_steps=15,
        negative_throttle_reset_speed_max=0.60, progress_full_lap_reward=5.00,
    ),
    2: StageTrainProfile(
        npc_count=2, random_start_enabled=True, max_throttle=0.80, max_cte_limit=8.0,
        progress_reward_scale=1.05, progress_milestone_lap=0.08, progress_milestone_reward=1.20,
        progress_reward_decay_min=0.60, penalty_decay_min=0.85,
        progress_backward_penalty_scale=0.60, reverse_onset_penalty=0.18,
        reverse_streak_penalty_scale=0.06, reverse_backdist_penalty_scale=2.10, reverse_event_penalty=1.20,
        npc_layout_reset_policy="hybrid", npc_layout_reset_every=5, npc_layout_reset_every_laps=1,
        npc_lane_offset_left_sim=0.42, npc_lane_offset_right_sim=-0.42, npc_lane_jitter_sim=0.08,
        spawn_min_gap_sim_ego_npc=0.8, spawn_min_gap_progress_ego_npc=40,
        spawn_min_gap_sim_npc_npc=1.5, spawn_min_gap_progress_npc_npc=80,
        stuck_counter_limit=25, offtrack_counter_limit=25, reward_clip_abs=10.0,
        disable_cte_auto_adjust=False, reverse_reset_steps=15,
        negative_throttle_reset_speed_max=0.60, progress_full_lap_reward=2.50,
        ent_coef=0.010, lr=3e-4,
    ),
    3: StageTrainProfile(
        npc_count=2, random_start_enabled=True, max_throttle=0.80, max_cte_limit=8.0,
        progress_reward_scale=0.95, progress_milestone_lap=0.10, progress_milestone_reward=1.00,
        progress_reward_decay_min=0.55, penalty_decay_min=0.80,
        progress_backward_penalty_scale=0.65, reverse_onset_penalty=0.20,
        reverse_streak_penalty_scale=0.06, reverse_backdist_penalty_scale=2.20, reverse_event_penalty=1.30,
        npc_layout_reset_policy="hybrid", npc_layout_reset_every=5, npc_layout_reset_every_laps=1,
        npc_lane_offset_left_sim=0.36, npc_lane_offset_right_sim=-0.36, npc_lane_jitter_sim=0.06,
        spawn_min_gap_sim_ego_npc=0.8, spawn_min_gap_progress_ego_npc=40,
        spawn_min_gap_sim_npc_npc=1.5, spawn_min_gap_progress_npc_npc=60,
        stuck_counter_limit=25, offtrack_counter_limit=25, reward_clip_abs=10.0,
        disable_cte_auto_adjust=False, reverse_reset_steps=15,
        negative_throttle_reset_speed_max=0.60, progress_full_lap_reward=2.50,
        ent_coef=0.009, lr=3e-4,
    ),
    4: StageTrainProfile(
        npc_count=1, random_start_enabled=False, max_throttle=0.80, max_cte_limit=8.0,
        progress_reward_scale=0.80, progress_milestone_lap=0.12, progress_milestone_reward=0.80,
        progress_reward_decay_min=0.50, penalty_decay_min=0.80,
        progress_backward_penalty_scale=0.70, reverse_onset_penalty=0.20,
        reverse_streak_penalty_scale=0.07, reverse_backdist_penalty_scale=2.30, reverse_event_penalty=1.30,
        stuck_counter_limit=25, offtrack_counter_limit=25, reward_clip_abs=10.0,
        disable_cte_auto_adjust=False, reverse_reset_steps=15,
        negative_throttle_reset_speed_max=0.60, progress_full_lap_reward=2.00,
        ent_coef=0.006, lr=2.5e-4,
    ),
    5: StageTrainProfile(
        npc_count=1, random_start_enabled=True, max_throttle=0.80, max_cte_limit=8.0,
        progress_reward_scale=0.70, progress_milestone_lap=0.15, progress_milestone_reward=0.60,
        progress_reward_decay_min=0.45, penalty_decay_min=0.75,
        progress_backward_penalty_scale=0.75, reverse_onset_penalty=0.22,
        reverse_streak_penalty_scale=0.08, reverse_backdist_penalty_scale=2.40, reverse_event_penalty=1.40,
        stuck_counter_limit=25, offtrack_counter_limit=25, reward_clip_abs=10.0,
        disable_cte_auto_adjust=False, reverse_reset_steps=15,
        negative_throttle_reset_speed_max=0.60, progress_full_lap_reward=2.00,
        ent_coef=0.004, lr=2.0e-4,
    ),
}


# notestagenote: note, note
STAGE_EVAL_GATES = {
    1: StageEvalGate(
        # stage1note: note(1.2->0.9), noteofftracknote
        min_stage_steps=80000, eval_every_steps=20000, eval_episodes=3, consecutive_passes_required=1,
        min_relative_progress_laps=0.9, require_lap_count_at_least=1, max_collisions=1, max_rear_end=0,
        allow_persistent_offtrack=True, allow_stuck=False, max_reverse_events=2,
    ),
    2: StageEvalGate(
        # stage2note: note1.3->1.0, notemin_npc_encounters(note), noteofftrack
        min_stage_steps=120000, eval_every_steps=20000, eval_episodes=3, consecutive_passes_required=1,
        min_relative_progress_laps=1.0, require_lap_count_at_least=1, max_collisions=1, max_rear_end=0,
        allow_persistent_offtrack=True, allow_stuck=False, max_reverse_events=2,
    ),
    3: StageEvalGate(
        # stage3note: note1.2->1.0, notemin_npc_encounters(notemin_npc_encounters=1noteallow=Truenote)
        min_stage_steps=160000, eval_every_steps=20000, eval_episodes=3, consecutive_passes_required=1,
        min_relative_progress_laps=1.0, require_lap_count_at_least=1, max_collisions=1, max_rear_end=0,
        allow_persistent_offtrack=True, allow_stuck=False, max_reverse_events=3,
    ),
    4: StageEvalGate(
        # stage4note: note0.9note, noteofftracknote
        min_stage_steps=240000, eval_every_steps=20000, eval_episodes=3, consecutive_passes_required=1,
        min_relative_progress_laps=0.9, require_lap_count_at_least=1, max_collisions=1, max_rear_end=0,
        allow_persistent_offtrack=True, allow_stuck=False, max_reverse_events=3,
        max_unsafe_follow_ratio=0.18,
    ),
    5: StageEvalGate(
        # stage5note: note
        min_stage_steps=320000, eval_every_steps=20000, eval_episodes=3, consecutive_passes_required=1,
        min_relative_progress_laps=1.0, require_lap_count_at_least=1, max_collisions=1, max_rear_end=0,
        allow_persistent_offtrack=False, allow_stuck=False, max_reverse_events=3,
        min_safe_overtakes=1, max_unsafe_cutin=1,
    ),
}


_TRAIN_PROFILE_CLI_KEYS = {
    "max_throttle": "max_throttle",
    "progress_reward_scale": "progress_reward_scale",
    "progress_milestone_lap": "progress_milestone_lap",
    "progress_milestone_reward": "progress_milestone_reward",
    "progress_reward_decay_min": "progress_reward_decay_min",
    "penalty_decay_min": "penalty_decay_min",
    "progress_backward_penalty_scale": "progress_backward_penalty_scale",
    "reverse_onset_penalty": "reverse_onset_penalty",
    "reverse_streak_penalty_scale": "reverse_streak_penalty_scale",
    "reverse_backdist_penalty_scale": "reverse_backdist_penalty_scale",
    "reverse_event_penalty": "reverse_event_penalty",
    "npc_layout_reset_policy": "npc_layout_reset_policy",
    "npc_layout_reset_every": "npc_layout_reset_every",
    "npc_lane_offset_left_sim": "npc_lane_offset_left_sim",
    "npc_lane_offset_right_sim": "npc_lane_offset_right_sim",
    "npc_lane_jitter_sim": "npc_lane_jitter_sim",
    "spawn_min_gap_sim_ego_npc": "spawn_min_gap_sim_ego_npc",
    "spawn_min_gap_progress_ego_npc": "spawn_min_gap_progress_ego_npc",
    "spawn_min_gap_sim_npc_npc": "spawn_min_gap_sim_npc_npc",
    "spawn_min_gap_progress_npc_npc": "spawn_min_gap_progress_npc_npc",
    "ent_coef": "ent_coef",
    "lr": "lr",
    "stuck_counter_limit": "stuck_counter_limit",
    "offtrack_counter_limit": "offtrack_counter_limit",
    "reverse_reset_steps": "reverse_reset_steps",
}


def _collect_cli_overrides(argv):
    """note CLI note(dest note)."""
    out = set()
    for raw in argv:
        if not raw.startswith("--"):
            continue
        tok = raw[2:]
        if "=" in tok:
            tok = tok.split("=", 1)[0]
        if tok.startswith("no-"):
            tok = tok[3:]
        out.add(tok.replace("-", "_"))
    return out


def _cli_overrode(cli_overrides, dest_name):
    return bool(dest_name and dest_name in (cli_overrides or set()))


def _set_model_hparams_from_profile(model, profile, cli_overrides):
    if model is None or profile is None:
        return
    if profile.ent_coef is not None and not _cli_overrode(cli_overrides, "ent_coef"):
        try:
            model.ent_coef = float(profile.ent_coef)
        except Exception:
            pass
    if profile.lr is not None and not _cli_overrode(cli_overrides, "lr"):
        try:
            lr = float(profile.lr)
            model.learning_rate = lr
            model.lr_schedule = (lambda _: lr)
            if hasattr(model, "policy") and getattr(model.policy, "optimizer", None) is not None:
                for group in model.policy.optimizer.param_groups:
                    group["lr"] = lr
        except Exception as e:
            print(f"⚠️ notestagenotefailed: {e}")


def apply_stage_train_profile(stage_id, curriculum_stage_ref, args=None, wrapper=None, model=None,
                              cli_overrides=None, verbose=False):
    profile = STAGE_TRAIN_PROFILES.get(int(stage_id))
    if profile is None:
        return None
    p = asdict(profile)

    # note curriculum ref(wrapper note ref notedynamicnote)
    if p.get("npc_count") is not None:
        curriculum_stage_ref["npc_count"] = int(p["npc_count"])
    if p.get("random_start_enabled") is not None:
        curriculum_stage_ref["stage_random_start_enabled"] = bool(p["random_start_enabled"])
    if p.get("max_cte_limit") is not None:
        curriculum_stage_ref["stage_cte_reset_limit"] = float(p["max_cte_limit"])
        curriculum_stage_ref["stage1_cte_reset_limit"] = float(p["max_cte_limit"])
    if p.get("max_throttle") is not None:
        curriculum_stage_ref["max_throttle"] = float(p["max_throttle"])

    ref_keys = [
        "progress_reward_scale", "progress_milestone_lap", "progress_milestone_reward",
        "progress_reward_decay_min", "penalty_decay_min", "progress_backward_penalty_scale",
        "reverse_onset_penalty", "reverse_streak_penalty_scale", "reverse_backdist_penalty_scale",
        "reverse_event_penalty", "npc_layout_reset_policy", "npc_layout_reset_every",
        "npc_layout_reset_every_laps",
        "npc_lane_offset_left_sim", "npc_lane_offset_right_sim", "npc_lane_jitter_sim",
        "spawn_min_gap_sim_ego_npc", "spawn_min_gap_progress_ego_npc",
        "spawn_min_gap_sim_npc_npc", "spawn_min_gap_progress_npc_npc",
        "stuck_counter_limit", "offtrack_counter_limit",
        "reward_clip_abs", "disable_cte_auto_adjust", "reverse_reset_steps",
        "negative_throttle_reset_speed_max", "progress_full_lap_reward",
    ]
    for key in ref_keys:
        if p.get(key) is not None:
            curriculum_stage_ref[key] = p[key]

    # args defaultnote(note CLI note)
    if args is not None:
        for pkey, akey in _TRAIN_PROFILE_CLI_KEYS.items():
            val = p.get(pkey)
            if val is None or _cli_overrode(cli_overrides, akey):
                continue
            if hasattr(args, akey):
                setattr(args, akey, val)

    # noterowsnote wrapper note
    if wrapper is not None:
        for attr in [
            "max_throttle",
            "progress_reward_scale", "progress_milestone_lap", "progress_milestone_reward",
            "progress_reward_decay_min", "penalty_decay_min", "progress_backward_penalty_scale",
            "reverse_onset_penalty", "reverse_streak_penalty_scale", "reverse_backdist_penalty_scale",
            "reverse_event_penalty",
            "stuck_counter_limit", "offtrack_counter_limit", "reward_clip_abs", "reverse_reset_steps",
            "negative_throttle_reset_speed_max", "progress_full_lap_reward",
        ]:
            if p.get(attr) is not None and hasattr(wrapper, attr):
                setattr(wrapper, attr, p[attr])
        if p.get("max_cte_limit") is not None and hasattr(wrapper, "current_max_cte"):
            # notecurrentstagenote; reset() notestagenote
            wrapper.current_max_cte = float(p["max_cte_limit"])
            if hasattr(wrapper, "stage1_cte_reset_limit"):
                wrapper.stage1_cte_reset_limit = float(p["max_cte_limit"])
        if p.get("npc_lane_jitter_sim") is not None and hasattr(wrapper, "npc_lane_jitter_sim"):
            wrapper.npc_lane_jitter_sim = float(p["npc_lane_jitter_sim"])
        if hasattr(wrapper, "_lane_offset_overrides"):
            if p.get("npc_lane_offset_left_sim") is not None:
                wrapper._lane_offset_overrides["left"] = float(p["npc_lane_offset_left_sim"])
            if p.get("npc_lane_offset_right_sim") is not None:
                wrapper._lane_offset_overrides["right"] = float(p["npc_lane_offset_right_sim"])

    _set_model_hparams_from_profile(model, profile, cli_overrides)

    if verbose:
        summary = {
            "max_throttle": p.get("max_throttle"),
            "max_cte_limit": p.get("max_cte_limit"),
            "progress_reward_scale": p.get("progress_reward_scale"),
            "progress_milestone_lap": p.get("progress_milestone_lap"),
            "progress_milestone_reward": p.get("progress_milestone_reward"),
            "random_start_enabled": p.get("random_start_enabled"),
            "npc_count": p.get("npc_count"),
            "lane_offsets": [p.get("npc_lane_offset_left_sim"), p.get("npc_lane_offset_right_sim")],
            "spawn_gap_ego_npc": [p.get("spawn_min_gap_sim_ego_npc"), p.get("spawn_min_gap_progress_ego_npc")],
            "ent_coef": p.get("ent_coef"),
            "lr": p.get("lr"),
        }
        print(f"🧩 StageProfile(stage={int(stage_id)}): {json.dumps(summary, ensure_ascii=False)}")
    return profile


def stage_eval_gate_pass(eval_out, gate):
    if gate is None:
        return bool(eval_out.get("success", False))
    episodes = eval_out.get("episodes", [])
    if not episodes:
        return False
    total_collisions = sum(1 for e in episodes if e.get("collision"))
    total_rear_end = sum(1 for e in episodes if e.get("rear_end"))
    total_reverse_events = sum(int(e.get("reverse_events", 0)) for e in episodes)
    total_npc_encounters = sum(int(e.get("npc_encounters", 0)) for e in episodes)
    total_safe_overtakes = sum(int(e.get("safe_overtakes", 0)) for e in episodes)
    total_unsafe_cutin = sum(int(e.get("unsafe_cutin", 0)) for e in episodes)
    avg_unsafe_follow_ratio = float(np.mean([float(e.get("unsafe_follow_ratio", 0.0)) for e in episodes]))
    min_laps = min(int(e.get("laps", 0)) for e in episodes)
    min_rel_prog = min(float(e.get("relative_progress_laps", 0.0)) for e in episodes)
    avg_laps = float(np.mean([int(e.get("laps", 0)) for e in episodes]))
    avg_rel_prog = float(np.mean([float(e.get("relative_progress_laps", 0.0)) for e in episodes]))
    term_reasons = [str(e.get("termination_reason")) for e in episodes]
    good_eps = 0
    for e in episodes:
        if e.get("collision") or e.get("rear_end"):
            continue
        if (not gate.allow_persistent_offtrack) and e.get("termination_reason") == "persistent_offtrack":
            continue
        if (not gate.allow_stuck) and e.get("termination_reason") == "stuck":
            continue
        if gate.max_reverse_events is not None and int(e.get("reverse_events", 0)) > int(gate.max_reverse_events):
            continue
        if int(e.get("laps", 0)) < int(gate.require_lap_count_at_least) and float(e.get("relative_progress_laps", 0.0)) < float(gate.min_relative_progress_laps):
            continue
        good_eps += 1
    required_good_eps = max(1, int(math.ceil(len(episodes) * 0.67)))

    if total_collisions > int(gate.max_collisions):
        return False
    if total_rear_end > int(gate.max_rear_end):
        return False
    if gate.max_reverse_events is not None and total_reverse_events > int(gate.max_reverse_events) * max(1, len(episodes)):
        return False
    if gate.min_npc_encounters and total_npc_encounters < int(gate.min_npc_encounters):
        return False
    if gate.min_safe_overtakes and total_safe_overtakes < int(gate.min_safe_overtakes):
        return False
    if gate.max_unsafe_cutin is not None and total_unsafe_cutin > int(gate.max_unsafe_cutin):
        return False
    if gate.max_unsafe_follow_ratio is not None and avg_unsafe_follow_ratio > float(gate.max_unsafe_follow_ratio):
        return False
    if avg_rel_prog < float(gate.min_relative_progress_laps):
        return False
    if avg_laps < float(gate.require_lap_count_at_least):
        return False
    if good_eps < required_good_eps:
        return False
    return True


class GeneratedTrackV10Wrapper(OvertakeTrainingWrapper):
    """V10 wrapper: note generated_track + spawnnote + noterewardnote."""

    def __init__(self, env, npc_controllers=None, track_cache=None, dist_scale_profile=None,
                 curriculum_stage_ref=None, spawn_debug=True, target_size=(120, 160),
                 spawn_jitter_s_sim=0.30, spawn_jitter_d_sim=0.25, spawn_yaw_jitter_deg=6.0,
                 spawn_verify_tol_sim=0.90, spawn_verify_cte_threshold=8.0,
                 npc_layout_debug=False,
                 npc_lane_balance_mode="balanced_lr",
                 npc_layout_segments=3,
                 npc_lane_offset_left_sim=None,
                 npc_lane_offset_right_sim=None,
                 npc_lane_offset_center_sim=None,
                 npc_lane_jitter_sim=None,
                 npc_npc_collision_guard_enable=True,
                 npc_npc_collision_guard_dist_sim=1.10,
                 npc_npc_collision_guard_progress_window=120,
                 npc_npc_collision_guard_brake_steps=7,
                 npc_npc_collision_guard_cooldown_steps=14,
                 npc_npc_contact_reset_enable=False,
                 npc_npc_contact_dist_sim=0.45,
                 npc_npc_contact_progress_window=26,
                 npc_npc_contact_cooldown_steps=45,
                 npc_npc_spawn_hard_check=True,
                 spawn_min_gap_sim_ego_npc_hard_floor=1.40,
                 spawn_min_gap_progress_ego_npc_hard_floor=60,
                 npc_persist_across_agent_resets=True,
                 npc_stuck_reset_enable=True,
                 npc_stuck_speed_thresh=0.10,
                 npc_stuck_disp_thresh_sim=0.012,
                 npc_stuck_steps=80,
                 npc_stuck_cooldown_steps=120,
                 spawn_debug_violations_limit=5,
                 lazy_connect_npcs=False,
                 **kwargs):
        super().__init__(
            env,
            npc_controllers=npc_controllers,
            track_cache=track_cache,
            scene_name="generated_track",
            curriculum_stage_ref=curriculum_stage_ref,
            target_size=target_size,
            **kwargs,
        )
        self.dist_scale = dist_scale_profile or dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK)
        self.spawn_debug = spawn_debug
        self.spawn_jitter_s_sim = float(spawn_jitter_s_sim)
        self.spawn_jitter_d_sim = float(spawn_jitter_d_sim)
        self.spawn_yaw_jitter_deg = float(spawn_yaw_jitter_deg)
        self.spawn_verify_tol_sim = float(spawn_verify_tol_sim)
        self.spawn_verify_cte_threshold = float(spawn_verify_cte_threshold)
        self.npc_layout_debug = bool(npc_layout_debug)
        self.spawn_debug_violations_limit = int(spawn_debug_violations_limit)
        self.lazy_connect_npcs = bool(lazy_connect_npcs)
        self.reset_debug_history = deque(maxlen=200)
        self._reset_index = 0
        self._bad_nodes = set()  # note
        self._obs_caps_logged = False
        self.follow_distance_history = deque(maxlen=400)
        self._last_npc_dists = {}
        self._last_spawn_debug = None
        self._last_step_npc_metrics = {}
        self.success_laps_target = int(self.curriculum_stage_ref.get("success_laps_target", 2))
        self._last_episode_success_2laps = False
        self._last_episode_collision = False

        # v10.1: NPC note
        self.npc_layout_id = 0
        self.npc_layout_reset_count = 0
        self.npc_layout_age_agent_resets = 0
        self.npc_layout_last_reset_reason = "init"
        self.npc_layout_cached_poses = []          # [{npc_id, pose(dict), lane_side, segment_id}]
        self.npc_layout_lane_assignments = []
        self.npc_layout_segment_assignments = []
        self._npc_layout_stage_id = None
        self._npc_layout_active_count = 0
        self._consecutive_layout_failures = 0
        # v11.3: note NPC note
        self._total_laps_completed = 0             # note
        self._npc_layout_last_reset_at_laps = 0    # note NPC note
        self._mid_episode_laps_already_counted = 0  # step()note, resetnote
        self.spawn_precheck_fail_stats = Counter()
        self.spawn_failure_reason_stats = Counter()
        self._lane_side_seen_counter = Counter()
        self._segment_seen_counter = Counter()
        self._pending_npc_connect = []  # [(npc, body_rgb),...] for lazy connect
        self._spawn_bin_counts = Counter()
        self._spawn_bins = int(self.curriculum_stage_ref.get("spawn_bins", 8))
        self._last_progress_milestone = 0
        # note NPC note"note", note.
        self._inactive_npc_hide_pose = (0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        self.npc_npc_collision_guard_enable = bool(npc_npc_collision_guard_enable)
        self.npc_npc_collision_guard_dist_sim = float(npc_npc_collision_guard_dist_sim)
        self.npc_npc_collision_guard_progress_window = int(npc_npc_collision_guard_progress_window)
        self.npc_npc_collision_guard_brake_steps = int(npc_npc_collision_guard_brake_steps)
        self.npc_npc_collision_guard_cooldown_steps = int(npc_npc_collision_guard_cooldown_steps)
        self.npc_npc_contact_reset_enable = bool(npc_npc_contact_reset_enable)
        self.npc_npc_contact_dist_sim = float(npc_npc_contact_dist_sim)
        self.npc_npc_contact_progress_window = int(npc_npc_contact_progress_window)
        self.npc_npc_contact_cooldown_steps = int(npc_npc_contact_cooldown_steps)
        self.npc_npc_spawn_hard_check = bool(npc_npc_spawn_hard_check)
        self.spawn_min_gap_sim_ego_npc_hard_floor = float(spawn_min_gap_sim_ego_npc_hard_floor)
        self.spawn_min_gap_progress_ego_npc_hard_floor = int(spawn_min_gap_progress_ego_npc_hard_floor)
        self.npc_persist_across_agent_resets = bool(npc_persist_across_agent_resets)
        self.npc_stuck_reset_enable = bool(npc_stuck_reset_enable)
        self.npc_stuck_speed_thresh = float(npc_stuck_speed_thresh)
        self.npc_stuck_disp_thresh_sim = float(npc_stuck_disp_thresh_sim)
        self.npc_stuck_steps = int(npc_stuck_steps)
        self.npc_stuck_cooldown_steps = int(npc_stuck_cooldown_steps)
        self._npc_npc_guard_pair_cooldown = {}
        self._npc_npc_guard_events_episode = 0
        self._npc_npc_contact_reset_cooldown_left = 0
        self._npc_npc_contact_events_episode = 0
        self._npc_stuck_state = {}
        self._npc_stuck_reset_cooldown_left = 0
        self._force_npc_layout_refresh_next = False

        # v10.1: lane offset note profile note CLI note
        self._lane_offset_overrides = {
            "left": npc_lane_offset_left_sim,
            "right": npc_lane_offset_right_sim,
            "center": npc_lane_offset_center_sim,
        }
        self.npc_lane_jitter_sim = float(
            self.dist_scale.get("lane_offset_safe_jitter_sim", 0.12)
            if npc_lane_jitter_sim is None else npc_lane_jitter_sim
        )
        # note wrapper defaultnote, note curriculum_stage_ref note
        self.default_npc_lane_balance_mode = str(npc_lane_balance_mode)
        self.default_npc_layout_segments = int(npc_layout_segments)

        # v10.2: note(notetracknote, note)
        self.reverse_progress_counter = 0
        self.reverse_progress_accum = 0.0
        self._prev_ego_fine_idx_for_reverse = None
        # P0-2 fix: note _fine_gap_sim, note property note dist_scale read(note property)
        self.reverse_penalty_steps = int(self.curriculum_stage_ref.get("reverse_penalty_steps", 5))
        self.reverse_progress_step_thresh = int(self.curriculum_stage_ref.get("reverse_progress_step_thresh", 1))
        self.reverse_progress_dist_thresh = float(self.curriculum_stage_ref.get("reverse_progress_dist_thresh", 0.05))
        # defaultnote reverse gate: notereward/notemodelnote
        self.reverse_gate_enabled = bool(self.curriculum_stage_ref.get("reverse_gate_enabled", False))
        self.reverse_gate_lock_speed = float(self.curriculum_stage_ref.get("reverse_gate_lock_speed", 0.22))
        self.reverse_gate_brake_speed = float(self.curriculum_stage_ref.get("reverse_gate_brake_speed", 0.60))
        self.reverse_gate_block_penalty = float(self.curriculum_stage_ref.get("reverse_gate_block_penalty", 0.05))
        self.reverse_escape_steps = int(self.curriculum_stage_ref.get("reverse_escape_steps", 12))
        self.reverse_escape_cooldown_steps = int(self.curriculum_stage_ref.get("reverse_escape_cooldown_steps", 30))
        self.reverse_escape_min_episode_steps = int(self.curriculum_stage_ref.get("reverse_escape_min_episode_steps", 70))
        self.reverse_escape_stuck_counter = int(self.curriculum_stage_ref.get("reverse_escape_stuck_counter", 35))
        self.reverse_escape_offtrack_counter = int(self.curriculum_stage_ref.get("reverse_escape_offtrack_counter", 14))
        self.reverse_escape_low_progress_counter = int(self.curriculum_stage_ref.get("reverse_escape_low_progress_counter", 80))
        self.reverse_escape_low_progress_step_sim = float(self.curriculum_stage_ref.get("reverse_escape_low_progress_step_sim", 0.01))
        # v10.7: reward-based anti-reverse shaping(note)
        self.reverse_onset_penalty = float(self.curriculum_stage_ref.get("reverse_onset_penalty", 0.12))
        self.reverse_streak_penalty_scale = float(self.curriculum_stage_ref.get("reverse_streak_penalty_scale", 0.03))
        self.reverse_backdist_penalty_scale = float(self.curriculum_stage_ref.get("reverse_backdist_penalty_scale", 1.4))
        self.reverse_event_penalty = float(self.curriculum_stage_ref.get("reverse_event_penalty", 0.8))
        self._reverse_escape_steps_left = 0
        self._reverse_escape_cooldown_left = 0
        self._low_progress_counter = 0

        # v10.3: tracknote(fine_track)note/reward, noteresetnote
        self.progress_reward_scale = float(self.curriculum_stage_ref.get("progress_reward_scale", 0.85))
        self.progress_backward_penalty_scale = float(self.curriculum_stage_ref.get("progress_backward_penalty_scale", 0.25))
        self.progress_milestone_lap = float(self.curriculum_stage_ref.get("progress_milestone_lap", 0.125))
        self.progress_milestone_reward = float(self.curriculum_stage_ref.get("progress_milestone_reward", 0.2))
        # first 0.1 note progress rewardnote(notefirstnote)
        self.early_progress_boost_lap = float(self.curriculum_stage_ref.get("early_progress_boost_lap", 0.10))
        self.early_progress_boost_factor = float(self.curriculum_stage_ref.get("early_progress_boost_factor", 2.0))
        self.progress_reward_decay_min = float(self.curriculum_stage_ref.get("progress_reward_decay_min", 0.35))
        self.penalty_decay_min = float(self.curriculum_stage_ref.get("penalty_decay_min", 0.55))
        self.random_start_from_stage = int(self.curriculum_stage_ref.get("random_start_from_stage", 5))
        self.stage1_cte_reset_limit = float(self.curriculum_stage_ref.get("stage1_cte_reset_limit", 6.0))
        self.startup_grace_steps = int(self.curriculum_stage_ref.get("startup_grace_steps", 70))
        self.motion_arm_speed_thresh = float(self.curriculum_stage_ref.get("motion_arm_speed_thresh", 0.35))
        self.motion_arm_progress_steps = int(self.curriculum_stage_ref.get("motion_arm_progress_steps", 2))
        self.motion_arm_speed_streak_steps = int(self.curriculum_stage_ref.get("motion_arm_speed_streak_steps", 3))
        self.motion_arm_displacement_sim = float(self.curriculum_stage_ref.get("motion_arm_displacement_sim", 0.18))
        self.motion_arm_forward_progress_sim = float(self.curriculum_stage_ref.get("motion_arm_forward_progress_sim", 0.12))
        self.no_motion_timeout_steps = int(self.curriculum_stage_ref.get("no_motion_timeout_steps", 10))
        # note: resetnotefirstnote, notefirstnote
        self.startup_force_throttle_steps = int(self.curriculum_stage_ref.get("startup_force_throttle_steps", 10))
        self.startup_force_throttle = float(self.curriculum_stage_ref.get("startup_force_throttle", 0.20))
        self.stuck_guard_progress_laps = float(self.curriculum_stage_ref.get("stuck_guard_progress_laps", 0.05))
        self.offtrack_guard_progress_laps = float(self.curriculum_stage_ref.get("offtrack_guard_progress_laps", 0.03))
        self.stuck_counter_limit = int(self.curriculum_stage_ref.get("stuck_counter_limit", 90))
        self.offtrack_counter_limit = int(self.curriculum_stage_ref.get("offtrack_counter_limit", 35))
        self.reward_clip_abs = float(self.curriculum_stage_ref.get("reward_clip_abs", 0.0))
        self.disable_cte_auto_adjust = bool(self.curriculum_stage_ref.get("disable_cte_auto_adjust", False))
        self.reverse_reset_steps = int(self.curriculum_stage_ref.get("reverse_reset_steps", 10))
        self.negative_throttle_reset_speed_max = float(self.curriculum_stage_ref.get("negative_throttle_reset_speed_max", 0.60))
        self.progress_full_lap_reward = float(self.curriculum_stage_ref.get("progress_full_lap_reward", 0.8))
        self._progress_step_clip = int(self.curriculum_stage_ref.get("progress_step_clip", 24))
        self._reverse_reset_streak = 0
        self._negative_throttle_streak = 0
        self._episode_prev_fine_idx = None
        self._episode_progress_fine_signed = 0.0
        self._episode_progress_fine_forward = 0.0
        self._episode_progress_fine_backward = 0.0
        self._episode_motion_armed = False
        self._motion_speed_streak = 0
        self._episode_spawn_x = None
        self._episode_spawn_z = None
        self._last_full_lap_bonus = 0

        # noteepisode, note(generated_track)
        if self.max_episode_steps < 1500:
            self.max_episode_steps = 1500

        # note steer/throttle note [-1, 1]
        try:
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            print("   PASS V10 note: steer∈[-1,1], throttle∈[-1,1]")
        except Exception as e:
            print(f"   ⚠️ notefailed: {e}")

    # ---------------- telemetry / info helpers ----------------
    def _extract_pos(self, info):
        pos = info.get("pos")
        if isinstance(pos, (tuple, list)) and len(pos) >= 3:
            return float(pos[0]), float(pos[1]), float(pos[2])
        if "pos_x" in info and "pos_z" in info:
            return float(info.get("pos_x", 0.0)), float(info.get("pos_y", 0.0)), float(info.get("pos_z", 0.0))
        # fallbacknoteenv handler(telemetry)
        try:
            h = self.env.viewer.handler
            return float(h.x), float(h.y), float(h.z)
        except Exception:
            return 0.0, 0.0, 0.0

    def _clear_handler_over(self):
        try:
            self.env.viewer.handler.over = False
        except Exception:
            pass

    def _idle_step_for_telemetry(self, n=1):
        obs = None
        info = {}
        for _ in range(max(1, n)):
            try:
                obs, _, _, info = self.env.step(np.array([0.0, 0.0], dtype=np.float32))
                self._clear_handler_over()
            except Exception:
                break
            time.sleep(0.03)
        return obs, info

    def _log_obs_caps_once(self, info):
        if self._obs_caps_logged:
            return
        keys = sorted(list(info.keys())) if isinstance(info, dict) else []
        print("🧾 OBS_FIELD_CAPS")
        print("   stablenote(goal):", OBS_FIELD_CAPS["stable"])
        print("   noteinfonote:", keys[:50])
        self._obs_caps_logged = True

    def _format_progress_bar(self, ratio, width=16):
        ratio = float(np.clip(ratio, 0.0, 1.0))
        filled = int(round(ratio * width))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    # ---------------- reward debug helpers ----------------
    def _rt_add(self, info, key, value):
        """Accumulate reward term decomposition into info['reward_terms']."""
        if not isinstance(info, dict):
            return
        v = float(_safe_float(value, 0.0))
        if abs(v) < 1e-12:
            return
        terms = info.get("reward_terms")
        if not isinstance(terms, dict):
            terms = {}
            info["reward_terms"] = terms
        terms[key] = float(_safe_float(terms.get(key, 0.0), 0.0) + v)

    def _rt_finalize(self, info, final_reward):
        if not isinstance(info, dict):
            return
        terms = info.get("reward_terms")
        if not isinstance(terms, dict):
            return
        total = float(sum(float(_safe_float(v, 0.0)) for v in terms.values()))
        info["reward_terms_total"] = float(total)
        info["reward_terms_gap"] = float(final_reward - total)

    def _on_episode_end(self, avg_cte, term_reason):
        """Subclass hook called in reset() before episode stats are reinitialized."""
        _ = avg_cte
        _ = term_reason

    @property
    def _fine_gap_sim(self):
        """P0-2 fix: note dist_scale read, note dist_scale note."""
        return float(self.dist_scale.get("avg_fine_gap_sim") or 0.025)

    def _maybe_open_reverse_escape_window(self):
        """note, note."""
        if not self.reverse_gate_enabled:
            return False
        if self._reverse_escape_steps_left > 0 or self._reverse_escape_cooldown_left > 0:
            return False
        if self.episode_step < self.reverse_escape_min_episode_steps:
            return False
        progress_laps_est = float(self.episode_stats.get('progress_laps_est', 0.0))
        severe_offtrack = False  # offtrack_counter note, note CTE note done
        stuck_like = self.stuck_counter >= self.reverse_escape_stuck_counter
        low_progress_stall = self._low_progress_counter >= self.reverse_escape_low_progress_counter
        if (severe_offtrack or stuck_like or low_progress_stall) and progress_laps_est <= 0.25:
            self._reverse_escape_steps_left = max(1, self.reverse_escape_steps)
            self.episode_stats['reverse_escape_windows'] = self.episode_stats.get('reverse_escape_windows', 0) + 1
            return True
        return False

    def _apply_reverse_gate(self, throttle_cmd_raw):
        """
        note: note, notedefaultnote.
        note.
        """
        throttle_cmd = float(np.clip(throttle_cmd_raw, -1.0, 1.0))
        meta = {
            "blocked": False,
            "escape_active": False,
            "escape_opened": False,
            "reverse_request": bool(throttle_cmd < 0.0),
            "last_speed": 0.0,
            "scaled": False,
        }

        # note cooldown/escape note(noteticknote)
        if self._reverse_escape_steps_left <= 0 and self._reverse_escape_cooldown_left > 0:
            self._reverse_escape_cooldown_left -= 1

        if not self.reverse_gate_enabled or throttle_cmd >= 0.0:
            return throttle_cmd, meta

        last_speed = abs(float(self.speed_history[-1])) if len(self.speed_history) > 0 else 0.0
        meta["last_speed"] = last_speed

        # notestagenote
        if last_speed >= self.reverse_gate_brake_speed:
            return throttle_cmd, meta

        # note(note)
        if self._reverse_escape_steps_left <= 0:
            meta["escape_opened"] = self._maybe_open_reverse_escape_window()

        if self._reverse_escape_steps_left > 0:
            meta["escape_active"] = True
            self._reverse_escape_steps_left -= 1
            if self._reverse_escape_steps_left <= 0:
                self._reverse_escape_steps_left = 0
                self._reverse_escape_cooldown_left = max(self._reverse_escape_cooldown_left, self.reverse_escape_cooldown_steps)
            return throttle_cmd, meta

        # note: note, note
        if self.reverse_gate_lock_speed < last_speed < self.reverse_gate_brake_speed:
            denom = max(1e-6, self.reverse_gate_brake_speed - self.reverse_gate_lock_speed)
            scale = (last_speed - self.reverse_gate_lock_speed) / denom
            gated = float(throttle_cmd * np.clip(scale, 0.0, 1.0))
            meta["scaled"] = True
            if abs(gated) < 1e-3:
                gated = 0.0
                meta["blocked"] = True
            return gated, meta

        # note: note
        meta["blocked"] = True
        return 0.0, meta

    def _shaping_decay_factor(self):
        """note: notestagetrainingnote, note shaping reward/note.
        first500knote(1.0), 500k-1000knote."""
        decay_start = 500000
        decay_end   = 1000000
        steps = self.total_train_steps
        if steps <= decay_start:
            t = 0.0
        else:
            t = float(np.clip((steps - decay_start) / float(decay_end - decay_start), 0.0, 1.0))
        # note=1.0, note
        reward_scale = 1.0 - (1.0 - self.progress_reward_decay_min) * t
        penalty_scale = 1.0 - (1.0 - self.penalty_decay_min) * t
        return reward_scale, penalty_scale

    def _update_episode_progress(self, info, speed=None):
        """note fine_track notetracknote, notecurrentnote(simnote)."""
        info = info if isinstance(info, dict) else {}
        fine = self._track_fine()
        fine_total = len(fine)
        if not self.track_cache or fine_total <= 1:
            info['progress_step_sim'] = 0.0
            info['episode_progress_laps_est'] = 0.0
            info['track_progress_ratio'] = 0.0
            info['episode_progress_ratio_to_goal'] = 0.0
            return 0, 0.0, 0

        lx, ly, lz = self._extract_pos(info)
        cur_fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, lx, lz)
        self.learner_fine_idx = int(cur_fi)
        info['learner_fine_idx'] = int(cur_fi)

        if self._episode_prev_fine_idx is None:
            self._episode_prev_fine_idx = int(cur_fi)
            lap_ratio = float(cur_fi) / float(max(1, fine_total))
            info['progress_step_sim'] = 0.0
            info['track_progress_ratio'] = lap_ratio
            info['episode_progress_laps_est'] = 0.0
            info['episode_progress_ratio_to_goal'] = 0.0
            return 0, 0.0, fine_total

        dfi = int(self.track_cache.progress_diff(self.scene_name, int(cur_fi), int(self._episode_prev_fine_idx)))
        self._episode_prev_fine_idx = int(cur_fi)

        # notenearestnote, notereward/note
        dfi = int(np.clip(dfi, -self._progress_step_clip, self._progress_step_clip))
        step_sim = float(dfi) * max(self._fine_gap_sim, 1e-3)

        self._episode_progress_fine_signed += float(dfi)
        if dfi > 0:
            self._episode_progress_fine_forward += float(dfi)
        elif dfi < 0:
            self._episode_progress_fine_backward += float(-dfi)

        # note(v10.4): notenearest fine-tracknote"note"
        try:
            spd = abs(float(speed)) if speed is not None else 0.0
        except Exception:
            spd = 0.0
        if spd >= self.motion_arm_speed_thresh:
            self._motion_speed_streak += 1
        else:
            self._motion_speed_streak = 0

        disp_from_spawn = 0.0
        if self._episode_spawn_x is not None and self._episode_spawn_z is not None:
            dxs = float(lx) - float(self._episode_spawn_x)
            dzs = float(lz) - float(self._episode_spawn_z)
            disp_from_spawn = math.sqrt(dxs * dxs + dzs * dzs)
        fwd_progress_dist = float(self._episode_progress_fine_forward) * max(self._fine_gap_sim, 1e-3)

        if (self._motion_speed_streak >= self.motion_arm_speed_streak_steps or
                disp_from_spawn >= self.motion_arm_displacement_sim or
                fwd_progress_dist >= self.motion_arm_forward_progress_sim):
            self._episode_motion_armed = True

        lap_ratio = float(cur_fi) / float(max(1, fine_total))
        laps_est = float(self._episode_progress_fine_forward) / float(max(1, fine_total))
        goal_ratio = min(1.0, float(self._episode_progress_fine_forward) / float(max(1, fine_total * max(1, self.success_laps_target))))
        info['progress_step_fi'] = dfi
        info['progress_step_sim'] = step_sim
        info['track_progress_ratio'] = lap_ratio
        info['episode_progress_laps_est'] = laps_est
        info['episode_progress_ratio_to_goal'] = goal_ratio
        info['motion_armed'] = bool(self._episode_motion_armed)
        info['motion_speed_streak'] = int(self._motion_speed_streak)
        info['motion_disp_from_spawn_sim'] = float(disp_from_spawn)
        info['motion_fwd_progress_sim'] = float(fwd_progress_dist)
        return dfi, step_sim, fine_total

    def _ensure_npc_connections_for_stage(self):
        """v10.2: trainingnoteNPC, stagenote."""
        if not self.lazy_connect_npcs:
            return
        target = int(self.curriculum_stage_ref.get("npc_count", 0))
        if target <= 0:
            return
        connected = [n for n in self.npc_controllers if n.connected]
        if len(connected) >= target:
            return

        need = target - len(connected)
        connected_now = 0
        for item in self._pending_npc_connect:
            npc = item.get("npc")
            if npc is None or npc.connected:
                continue
            color = item.get("body_rgb", (255, 100, 100))
            try:
                ok = npc.connect(body_rgb=color)
                if ok:
                    self._hide_npc_offtrack(npc)
                    connected_now += 1
                    time.sleep(0.3)
            except Exception as e:
                print(f"⚠️ note NPC_{getattr(npc, 'npc_id', '?')} failed: {e}")
            if connected_now >= need:
                break
        if connected_now > 0:
            print(f"🚗 noteNPCnote: +{connected_now} (goalstagenote {target} note)")

    # ---------------- geometry / spawn helpers ----------------
    def _node_tel(self, node):
        S = SimExtendedAPI.COORD_SCALE
        return node[0] / S, node[1] / S, node[2] / S

    def _track_nodes(self):
        return self.track_cache.nodes.get(self.scene_name, []) if self.track_cache else []

    def _track_fine(self):
        return self.track_cache.fine_track.get(self.scene_name, []) if self.track_cache else []

    def _hide_npc_offtrack(self, npc):
        """note NPC note, note."""
        if npc is None:
            return
        try:
            if getattr(npc, "running", False):
                npc.stop_driving()
            npc.set_mode("static", 0.0)
            if hasattr(npc, "set_speed_cap"):
                npc.set_speed_cap(None)
            px, py, pz, qx, qy, qz, qw = self._inactive_npc_hide_pose
            npc.set_position_node_coords(px, py, pz, qx, qy, qz, qw)
            setattr(npc, "_hidden_offtrack", True)
        except Exception:
            pass

    def _geometry_track_check(self, x_tel, z_tel, seed_fi=None, margin_scale=0.35, centerline_slack=0.15):
        """notegeometrynotecurrentnotetracknote(noteCTE)."""
        fine_idx = int(seed_fi) if seed_fi is not None else 0
        fine_dist = float("inf")
        if self.track_cache:
            fine_idx, fine_dist = self.track_cache.find_nearest_fine_track(self.scene_name, float(x_tel), float(z_tel))
        _, _, _, centerline_cap = self._lane_offset_stage_limits()
        center_ok = bool(fine_dist <= (float(centerline_cap) + float(centerline_slack)))

        manual_ok = True
        reason = "ok"
        try:
            if bool(getattr(getattr(self, "manual_spawn", None), "loaded", False)):
                margin = max(0.02, float(self.spawn_inside_margin_ratio) * float(margin_scale))
                cls = self.manual_spawn.classify_point(
                    float(x_tel), float(z_tel),
                    margin_ratio=margin,
                    seed_idx=int(fine_idx),
                    window_idx=max(16, self._spawn_window_idx()),
                )
                manual_ok = bool(cls.get("inside", False))
                reason = str(cls.get("reason", "ok"))
                fine_idx = int(cls.get("fine_idx", fine_idx))
        except Exception:
            manual_ok = True
            reason = "manual_check_exception"

        ok = bool(center_ok and manual_ok)
        return ok, {
            "fine_idx": int(fine_idx),
            "fine_dist_sim": float(fine_dist),
            "centerline_cap_sim": float(centerline_cap),
            "manual_reason": reason,
        }

    def _active_npcs(self):
        target = int(self.curriculum_stage_ref.get("npc_count", 0))
        return [n for n in self.npc_controllers if n.connected][:target]

    def _inactive_npcs(self):
        target = int(self.curriculum_stage_ref.get("npc_count", 0))
        return [n for n in self.npc_controllers if n.connected][target:]

    def _apply_active_npc_runtime_mode(self, active_npcs, npc_mode, npc_speed_min, npc_speed_max, freeze_for_spawn=False):
        """unifiednote NPC noterowsnote; spawnnote, succeedednote."""
        mode = self._normalize_npc_mode(npc_mode)
        vmin = float(npc_speed_min)
        vmax = float(npc_speed_max)
        for npc in active_npcs:
            if npc is None or (not getattr(npc, "connected", False)):
                continue
            if freeze_for_spawn:
                try:
                    if getattr(npc, "running", False):
                        npc.stop_driving()
                    npc.set_mode("static", 0.0)
                    try:
                        npc.handler.send_control(0, 0, 1.0)
                        time.sleep(0.03)
                        npc.handler.send_control(0, 0, 1.0)
                    except Exception:
                        pass
                except Exception:
                    pass
                continue

            if mode == "static":
                npc.set_mode("static", 0.0)
                try:
                    npc.handler.send_control(0, 0, 1.0)
                    time.sleep(0.03)
                    npc.handler.send_control(0, 0, 1.0)
                except Exception:
                    pass
                continue

            if mode == "wobble":
                npc.set_mode("wobble", 0.0)
                if not getattr(npc, "running", False):
                    npc.start_driving()
                continue

            if mode in ("slow", "random", "chaos"):
                if vmax > vmin:
                    thr = random.uniform(vmin, vmax)
                else:
                    thr = max(0.12, vmax if vmax > 0.0 else 0.20)
                npc.set_mode(mode, thr)
                if not getattr(npc, "running", False):
                    npc.start_driving()
                continue

            # unknown mode fallback
            npc.set_mode("static", 0.0)

    def _sync_npc_speed_cap_with_learner(self, info):
        """note NPC note learner currentnote(dynamicnote)."""
        if not isinstance(info, dict):
            return
        learner_speed = abs(_safe_float(info.get("speed", 0.0), 0.0))
        cap = max(0.10, float(learner_speed))
        for npc in self._active_npcs():
            try:
                if hasattr(npc, "set_speed_cap"):
                    npc.set_speed_cap(cap)
            except Exception:
                continue

    def _current_stage_id(self):
        return int(self.curriculum_stage_ref.get("stage", 1))

    def _is_static_layout_stage(self):
        return self._current_stage_id() in (2, 3)

    def _layout_reset_policy(self):
        # note: default NPC note learner note(note)
        raw = self.curriculum_stage_ref.get("npc_layout_reset_policy")
        if self._npc_persist_mode_enabled():
            return "agent_only"
        if raw:
            return str(raw)
        return "hybrid" if self._is_static_layout_stage() else "every_episode"

    def _layout_reset_every(self):
        return int(self.curriculum_stage_ref.get("npc_layout_reset_every", 5))

    def _layout_reset_on_collision(self):
        return bool(self.curriculum_stage_ref.get("npc_layout_reset_on_collision", True))

    def _layout_reset_on_success(self):
        return bool(self.curriculum_stage_ref.get("npc_layout_reset_on_success", True))

    def _layout_fail_refresh_threshold(self):
        return int(self.curriculum_stage_ref.get("npc_layout_force_refresh_after_failures", 3))

    def _npc_persist_mode_enabled(self):
        raw = self.curriculum_stage_ref.get("npc_persist_across_agent_resets", None)
        return bool(self.npc_persist_across_agent_resets if raw is None else raw)

    def _npc_layout_segments(self):
        return int(self.curriculum_stage_ref.get("npc_layout_segments", self.default_npc_layout_segments))

    def _npc_lane_balance_mode(self):
        return str(self.curriculum_stage_ref.get("npc_lane_balance_mode", self.default_npc_lane_balance_mode))

    @staticmethod
    def _normalize_npc_mode(raw_mode):
        m = str(raw_mode or "static").strip().lower()
        if m == "slow_policy":
            return "slow"
        if m in ("random_reverse", "chaos_reverse"):
            return "chaos"
        return m

    def _lane_offset_value(self, side):
        if side not in ("left", "right", "center"):
            side = "center"
        override = self._lane_offset_overrides.get(side)
        if override is not None:
            return float(override)
        if side == "center":
            return float(self.dist_scale.get("lane_offset_center_sim", 0.0))
        if side == "left":
            return float(self.dist_scale.get("lane_offset_left_sim", +0.55))
        return float(self.dist_scale.get("lane_offset_right_sim", -0.55))

    def _lane_offset_stage_limits(self):
        """notestagenote, notestaticobstaclenotetracknote."""
        stage = self._current_stage_id()
        if stage == 2:   # B1 staticnote1note
            return 0.80, 0.65, 0.48, 0.55  # offset_scale, jitter_scale, abs_cap, centerline_cap
        if stage == 3:   # B2 staticnote3note(note)
            return 0.68, 0.55, 0.42, 0.50
        return 1.00, 1.00, 0.70, 0.70

    def _node_to_fine_idx(self, node_idx):
        fine = self._track_fine()
        nodes = self._track_nodes()
        n_fine = len(fine)
        n_nodes = len(nodes)
        if n_fine <= 0:
            return 0
        if n_nodes <= 0:
            return int(node_idx) % n_fine
        return int(round(float(int(node_idx) % n_nodes) * float(n_fine) / float(max(1, n_nodes)))) % n_fine

    def _spawn_ego_verify_retries(self):
        retries = int(self.curriculum_stage_ref.get("spawn_ego_verify_retries", 3))
        retries = max(2, retries)
        mode = str(getattr(self, "v11_2_mode", "")).lower()
        if mode == "fixed2":
            # v11.5: note, teleport stablenote
            retries = max(retries, 3)
            if int(getattr(self, "_consecutive_layout_failures", 0)) >= 2:
                retries = max(retries, 4)
        return int(retries)

    def _precheck_pose_viable(self, pose):
        """v11.5: note - notetracknote.
        notefailednoterows teleport(note).
        note True note teleport.
        """
        if pose is None:
            return False
        tel = pose.get("tel", None)
        if not (isinstance(tel, (list, tuple)) and len(tel) >= 3):
            return False
        px, _, pz = tel
        seed_fi = int(_safe_float(pose.get("fine_idx", 0), 0))
        ok, _meta = self._geometry_track_check(
            px, pz,
            seed_fi=seed_fi,
            margin_scale=0.55,     # note verify stagenote, note teleport
            centerline_slack=0.10,
        )
        return bool(ok)

    def _npc_npc_guard_enabled(self):
        raw = self.curriculum_stage_ref.get("npc_npc_collision_guard_enable", None)
        return bool(self.npc_npc_collision_guard_enable if raw is None else raw)

    def _npc_npc_guard_dist(self):
        raw = self.curriculum_stage_ref.get("npc_npc_collision_guard_dist_sim", None)
        val = float(self.npc_npc_collision_guard_dist_sim if raw is None else raw)
        return max(0.20, val)

    def _npc_npc_guard_progress_window(self):
        raw = self.curriculum_stage_ref.get("npc_npc_collision_guard_progress_window", None)
        val = int(self.npc_npc_collision_guard_progress_window if raw is None else raw)
        return max(0, val)

    def _npc_npc_guard_brake_steps(self):
        raw = self.curriculum_stage_ref.get("npc_npc_collision_guard_brake_steps", None)
        val = int(self.npc_npc_collision_guard_brake_steps if raw is None else raw)
        return max(1, val)

    def _npc_npc_guard_cooldown_steps(self):
        raw = self.curriculum_stage_ref.get("npc_npc_collision_guard_cooldown_steps", None)
        val = int(self.npc_npc_collision_guard_cooldown_steps if raw is None else raw)
        return max(1, val)

    def _spawn_fail_on_npc_npc(self):
        raw = self.curriculum_stage_ref.get("npc_npc_spawn_hard_check", None)
        default = bool(self.npc_npc_spawn_hard_check)
        return bool(default if raw is None else raw)

    def _npc_stuck_reset_enabled(self):
        raw = self.curriculum_stage_ref.get("npc_stuck_reset_enable", None)
        return bool(self.npc_stuck_reset_enable if raw is None else raw)

    def _npc_stuck_speed_threshold(self):
        raw = self.curriculum_stage_ref.get("npc_stuck_speed_thresh", None)
        val = float(self.npc_stuck_speed_thresh if raw is None else raw)
        return max(0.02, val)

    def _npc_stuck_disp_threshold(self):
        raw = self.curriculum_stage_ref.get("npc_stuck_disp_thresh_sim", None)
        val = float(self.npc_stuck_disp_thresh_sim if raw is None else raw)
        return max(0.001, val)

    def _npc_stuck_steps_threshold(self):
        raw = self.curriculum_stage_ref.get("npc_stuck_steps", None)
        val = int(self.npc_stuck_steps if raw is None else raw)
        return max(10, val)

    def _npc_stuck_cooldown(self):
        raw = self.curriculum_stage_ref.get("npc_stuck_cooldown_steps", None)
        val = int(self.npc_stuck_cooldown_steps if raw is None else raw)
        return max(10, val)

    def _npc_npc_contact_reset_enabled(self):
        raw = self.curriculum_stage_ref.get("npc_npc_contact_reset_enable", None)
        return bool(self.npc_npc_contact_reset_enable if raw is None else raw)

    def _npc_npc_contact_dist(self):
        raw = self.curriculum_stage_ref.get("npc_npc_contact_dist_sim", None)
        val = float(self.npc_npc_contact_dist_sim if raw is None else raw)
        return max(0.20, val)

    def _npc_npc_contact_progress_window(self):
        raw = self.curriculum_stage_ref.get("npc_npc_contact_progress_window", None)
        val = int(self.npc_npc_contact_progress_window if raw is None else raw)
        return max(0, val)

    def _npc_npc_contact_cooldown_steps(self):
        raw = self.curriculum_stage_ref.get("npc_npc_contact_cooldown_steps", None)
        val = int(self.npc_npc_contact_cooldown_steps if raw is None else raw)
        return max(1, val)

    def _ego_npc_spawn_hard_floor_sim(self):
        raw = self.curriculum_stage_ref.get("spawn_min_gap_sim_ego_npc_hard_floor", None)
        val = float(self.spawn_min_gap_sim_ego_npc_hard_floor if raw is None else raw)
        return max(0.0, val)

    def _ego_npc_spawn_hard_floor_progress(self):
        raw = self.curriculum_stage_ref.get("spawn_min_gap_progress_ego_npc_hard_floor", None)
        val = int(self.spawn_min_gap_progress_ego_npc_hard_floor if raw is None else raw)
        return max(0, val)

    def _detect_npc_npc_contact(self):
        """detection NPC-NPC note(note NPC telemetry, note learner hit)."""
        active = [n for n in self._active_npcs() if getattr(n, "connected", False)]
        if len(active) < 2:
            return None
        dist_th = self._npc_npc_contact_dist()
        progress_win = self._npc_npc_contact_progress_window()
        best = None
        for i in range(len(active)):
            ai = active[i]
            ax, ay, az = ai.get_telemetry_position()
            afi = int(getattr(ai, "fine_track_idx", 0))
            if self.track_cache:
                afi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(ax), float(az))
                ai.fine_track_idx = int(afi)
            for j in range(i + 1, len(active)):
                bj = active[j]
                bx, by, bz = bj.get_telemetry_position()
                bfi = int(getattr(bj, "fine_track_idx", 0))
                if self.track_cache:
                    bfi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(bx), float(bz))
                    bj.fine_track_idx = int(bfi)
                d = math.sqrt((float(ax) - float(bx)) ** 2 + (float(az) - float(bz)) ** 2)
                if d > dist_th:
                    continue
                gap = None
                if self.track_cache:
                    gap = abs(self.track_cache.progress_diff(self.scene_name, int(afi), int(bfi)))
                    if progress_win > 0 and gap > progress_win:
                        continue
                rec = {
                    "pair": (int(ai.npc_id), int(bj.npc_id)),
                    "dist": float(d),
                    "progress_gap": (None if gap is None else int(gap)),
                }
                if best is None or rec["dist"] < best["dist"]:
                    best = rec
        return best

    def _maybe_handle_npc_npc_contact_reset(self, info=None):
        """note: notedetectionnote NPC-NPC note, note learner note, note NPC note."""
        if not self._npc_npc_contact_reset_enabled():
            return False
        if self._npc_npc_contact_reset_cooldown_left > 0:
            self._npc_npc_contact_reset_cooldown_left -= 1
            return False
        hit_v = str((info or {}).get("hit", "none")).lower() if isinstance(info, dict) else "none"
        if hit_v not in ("none", "", "null"):
            # learner note NPC-only note, note learner note
            return False
        contact = self._detect_npc_npc_contact()
        if not contact:
            return False

        self._npc_npc_contact_reset_cooldown_left = self._npc_npc_contact_cooldown_steps()
        ok = bool(self._refresh_npc_layout_mid_episode())

        if isinstance(info, dict):
            info["npc_npc_contact_detected"] = True
            info["npc_npc_contact_pair"] = contact.get("pair")
            info["npc_npc_contact_dist"] = float(contact.get("dist", -1.0))
            if contact.get("progress_gap") is not None:
                info["npc_npc_contact_progress_gap"] = int(contact.get("progress_gap"))
            info["npc_npc_contact_layout_reset"] = bool(ok)

        if ok:
            self._npc_npc_contact_events_episode += 1
            self.npc_layout_last_reset_reason = "npc_npc_contact"
            if isinstance(self.episode_stats, dict):
                self.episode_stats["npc_npc_contact_resets"] = int(self.episode_stats.get("npc_npc_contact_resets", 0)) + 1
            print(
                f"🚨 NPC-NPCnote: pair={contact.get('pair')} dist={float(contact.get('dist', 0.0)):.3f} "
                f"gap={contact.get('progress_gap')} cooldown={self._npc_npc_contact_reset_cooldown_left}"
            )
        return bool(ok)

    def _capture_runtime_npc_records(self, active_npcs):
        """notecurrentNPCnote(noteteleport), note agent-only learner reset."""
        if not active_npcs:
            return True, [], None
        records = []
        _, _, _, centerline_cap = self._lane_offset_stage_limits()
        offtrack_cap = float(centerline_cap) + 0.20
        for npc in active_npcs:
            if npc is None or (not getattr(npc, "connected", False)):
                return False, records, "runtime_npc_disconnected"
            nx, ny, nz = npc.get_telemetry_position()
            fi = int(getattr(npc, "fine_track_idx", 0))
            fd = 0.0
            if self.track_cache:
                fi, fd = self.track_cache.find_nearest_fine_track(self.scene_name, float(nx), float(nz))
                npc.fine_track_idx = int(fi)
            if self.track_cache and float(fd) > float(offtrack_cap):
                return False, records, "runtime_npc_offtrack"
            records.append({
                "npc_id": int(npc.npc_id),
                "npc_spawn_idx": -1,
                "npc_fine_idx": int(fi),
                "dist_err": 0.0,
                "tel": (float(nx), float(ny), float(nz)),
                "target_pose": {"tel": (float(nx), float(ny), float(nz)), "fine_idx": int(fi), "node_idx": -1},
            })
        return True, records, None

    def _maybe_reset_stuck_npcs(self, info=None):
        """noteNPCnoteNPC-onlynote, notelearnernote."""
        if not self._npc_stuck_reset_enabled():
            return False
        if self._npc_stuck_reset_cooldown_left > 0:
            self._npc_stuck_reset_cooldown_left -= 1
            return False
        active = [n for n in self._active_npcs() if getattr(n, "connected", False)]
        if len(active) <= 0:
            return False
        speed_th = self._npc_stuck_speed_threshold()
        disp_th = self._npc_stuck_disp_threshold()
        steps_th = self._npc_stuck_steps_threshold()
        stuck_ids = []
        for npc in active:
            st = self._npc_stuck_state.get(int(npc.npc_id))
            x, y, z = npc.get_telemetry_position()
            mode = str(getattr(npc, "mode", "static")).lower()
            v = abs(float(getattr(npc, "speed", 0.0)))
            if st is None:
                self._npc_stuck_state[int(npc.npc_id)] = {
                    "x": float(x),
                    "z": float(z),
                    "still_steps": 0,
                }
                continue
            dx = float(x) - float(st.get("x", x))
            dz = float(z) - float(st.get("z", z))
            disp = math.sqrt(dx * dx + dz * dz)
            st["x"] = float(x)
            st["z"] = float(z)
            if mode in ("static",):
                st["still_steps"] = 0
                continue
            if v <= speed_th and disp <= disp_th:
                st["still_steps"] = int(st.get("still_steps", 0)) + 1
            else:
                st["still_steps"] = max(0, int(st.get("still_steps", 0)) - 2)
            if int(st.get("still_steps", 0)) >= steps_th:
                stuck_ids.append(int(npc.npc_id))

        if not stuck_ids:
            return False
        ok = bool(self._refresh_npc_layout_mid_episode())
        self._npc_stuck_reset_cooldown_left = self._npc_stuck_cooldown()
        if not ok:
            self._force_npc_layout_refresh_next = True
        for sid in stuck_ids:
            if sid in self._npc_stuck_state:
                self._npc_stuck_state[sid]["still_steps"] = 0
        if isinstance(info, dict):
            info["npc_stuck_detected"] = True
            info["npc_stuck_ids"] = list(stuck_ids)
            info["npc_stuck_layout_reset"] = bool(ok)
        if isinstance(self.episode_stats, dict):
            self.episode_stats["npc_stuck_resets"] = int(self.episode_stats.get("npc_stuck_resets", 0)) + int(bool(ok))
        if ok:
            self.npc_layout_last_reset_reason = "npc_stuck"
            print(f"🚨 NPCnote: ids={stuck_ids} cooldown={self._npc_stuck_reset_cooldown_left}")
        return bool(ok)

    def _apply_npc_npc_collision_guard(self, info=None):
        """noterowsnote NPC-NPC note: note, note."""
        if not self._npc_npc_guard_enabled():
            return 0
        active = [n for n in self._active_npcs() if getattr(n, "connected", False)]
        if len(active) < 2:
            return 0

        guard_dist = self._npc_npc_guard_dist()
        progress_win = self._npc_npc_guard_progress_window()
        brake_steps = self._npc_npc_guard_brake_steps()
        cooldown_steps = self._npc_npc_guard_cooldown_steps()
        now_step = int(self.episode_step)

        pair_events = 0
        min_pair_dist = float("inf")
        for i in range(len(active)):
            npc_i = active[i]
            xi, yi, zi = npc_i.get_telemetry_position()
            fi_i = int(getattr(npc_i, "fine_track_idx", 0))
            if self.track_cache:
                fi_i, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(xi), float(zi))
                npc_i.fine_track_idx = int(fi_i)
            for j in range(i + 1, len(active)):
                npc_j = active[j]
                xj, yj, zj = npc_j.get_telemetry_position()
                fi_j = int(getattr(npc_j, "fine_track_idx", 0))
                if self.track_cache:
                    fi_j, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(xj), float(zj))
                    npc_j.fine_track_idx = int(fi_j)

                dist = math.sqrt((float(xi) - float(xj)) ** 2 + (float(zi) - float(zj)) ** 2)
                min_pair_dist = min(min_pair_dist, float(dist))
                if dist >= guard_dist:
                    continue

                progress_gap = None
                if self.track_cache:
                    progress_gap = abs(self.track_cache.progress_diff(self.scene_name, int(fi_i), int(fi_j)))
                    if progress_win > 0 and progress_gap > progress_win:
                        continue

                pair_key = tuple(sorted((int(npc_i.npc_id), int(npc_j.npc_id))))
                if int(self._npc_npc_guard_pair_cooldown.get(pair_key, -1)) > now_step:
                    continue

                targets = []
                if self.track_cache:
                    dfi = self.track_cache.progress_diff(self.scene_name, int(fi_i), int(fi_j))
                    if dfi < 0:
                        targets = [npc_i]   # i note
                    elif dfi > 0:
                        targets = [npc_j]   # j note
                    else:
                        targets = [npc_i, npc_j]
                else:
                    vi = abs(float(getattr(npc_i, "speed", 0.0)))
                    vj = abs(float(getattr(npc_j, "speed", 0.0)))
                    targets = [npc_i] if vi >= vj else [npc_j]

                for npc in targets:
                    if hasattr(npc, "set_emergency_brake"):
                        npc.set_emergency_brake(brake_steps)
                    else:
                        try:
                            npc.handler.send_control(0, 0, 1.0)
                        except Exception:
                            pass

                self._npc_npc_guard_pair_cooldown[pair_key] = now_step + cooldown_steps
                pair_events += 1
                if isinstance(info, dict):
                    info["npc_npc_guard_trigger"] = True
                    info["npc_npc_guard_last_pair"] = pair_key
                    info["npc_npc_guard_last_dist"] = float(dist)
                    if progress_gap is not None:
                        info["npc_npc_guard_last_progress_gap"] = int(progress_gap)

        if isinstance(info, dict):
            info["npc_npc_guard_events_step"] = int(pair_events)
            if min_pair_dist < float("inf"):
                info["npc_npc_min_dist"] = float(min_pair_dist)
        if pair_events > 0:
            self._npc_npc_guard_events_episode += int(pair_events)
            if isinstance(self.episode_stats, dict):
                self.episode_stats["npc_npc_guard_events"] = int(self.episode_stats.get("npc_npc_guard_events", 0)) + int(pair_events)
            if isinstance(info, dict):
                info["npc_npc_guard_events_ep"] = int(self._npc_npc_guard_events_episode)
        return int(pair_events)

    def _spawn_constraints(self):
        """v10.1: pair-wise note(note CLI/curriculum note)"""
        ds = self.dist_scale
        def _cfg(name, default):
            v = self.curriculum_stage_ref.get(name, None)
            return default if v is None else v
        def _is_overridden(name):
            return self.curriculum_stage_ref.get(name, None) is not None
        c = {
            "ego_npc": {
                "min_euclid": float(_cfg(
                    "spawn_min_gap_sim_ego_npc",
                    ds.get("spawn_min_gap_sim_ego_npc", ds.get("spawn_min_gap_sim", 3.0)),
                )),
                "min_progress_gap": int(_cfg(
                    "spawn_min_gap_progress_ego_npc",
                    ds.get("spawn_min_gap_progress_ego_npc", ds.get("spawn_min_gap_progress", 30)),
                )),
                # v11.5: False = note euclid AND progress note
                # (ego-first note NPC notefirstnote 40+ fine points, note <2m note)
                "require_both": False,
            },
            "npc_npc": {
                "min_euclid": float(_cfg(
                    "spawn_min_gap_sim_npc_npc",
                    ds.get("spawn_min_gap_sim_npc_npc", max(1.0, ds.get("spawn_min_gap_sim", 3.0) * 0.45)),
                )),
                "min_progress_gap": int(_cfg(
                    "spawn_min_gap_progress_npc_npc",
                    ds.get("spawn_min_gap_progress_npc_npc", max(12, int(ds.get("spawn_min_gap_progress", 30) * 0.3))),
                )),
                # B2 staticnotedefaultnote NPC-NPC note progress sanity check, note hard fail
                "require_progress": bool(_cfg(
                    "spawn_require_progress_npc_npc",
                    not self._is_static_layout_stage()
                )),
            },
        }
        # v11.4: note STAGE_TRAIN_PROFILES notecontrol, note.
        # note CLI note.
        # LiDARnote: noteobstaclenoteEgonote
        if bool(self.curriculum_stage_ref.get("npc_side_by_side_start", False)):
            c["ego_npc"]["min_progress_gap"] = 0
            # side-by-side note ego-npc note, notespawnfailednote/note
            c["ego_npc"]["min_euclid"] = 0.0
        # note: note spawn note NPC note learner note(note)
        c["ego_npc"]["min_euclid"] = max(float(c["ego_npc"]["min_euclid"]), self._ego_npc_spawn_hard_floor_sim())
        c["ego_npc"]["min_progress_gap"] = max(int(c["ego_npc"]["min_progress_gap"]), self._ego_npc_spawn_hard_floor_progress())
        return c

    def assign_npc_lane_sides(self, active_npc_count):
        mode = self._npc_lane_balance_mode()
        count = int(active_npc_count)
        if count <= 0:
            return []
        if mode == "random":
            choices = ["left", "right", "center"]
            if count <= 2:
                choices = ["left", "right"]
            return [random.choice(choices) for _ in range(count)]

        if mode == "balanced_lr_center" and count >= 3:
            sides = ["left", "right", "center"]
            while len(sides) < count:
                sides.append(random.choice(["left", "right", "center"]))
            random.shuffle(sides)
            return sides[:count]

        # default balanced_lr
        if count == 1:
            return [random.choice(["left", "right"])]
        if count == 2:
            sides = ["left", "right"]
            random.shuffle(sides)
            return sides
        sides = ["left", "right", random.choice(["left", "right"])]
        while len(sides) < count:
            sides.append(random.choice(["left", "right"]))
        random.shuffle(sides)
        return sides[:count]

    def _select_segment_ids(self, active_npc_count):
        count = int(active_npc_count)
        if count <= 0:
            return []
        seg_n = max(count, self._npc_layout_segments())
        if seg_n == count:
            segs = list(range(seg_n))
            random.shuffle(segs)
            return segs[:count], seg_n

        # note
        stride = float(seg_n) / float(count)
        segs = []
        for i in range(count):
            base = int(round(i * stride)) % seg_n
            window = [base, (base + 1) % seg_n, (base - 1) % seg_n]
            pick = None
            for cand in window:
                if cand not in segs:
                    pick = cand
                    break
            if pick is None:
                for cand in range(seg_n):
                    if cand not in segs:
                        pick = cand
                        break
            segs.append(pick if pick is not None else base)
        random.shuffle(segs)
        return segs, seg_n

    def build_segmented_npc_layout(self, active_npc_count):
        """v10.1: B2 staticnote, note precheck failednote."""
        nodes = self._track_nodes()
        n = len(nodes)
        if n <= 0 or active_npc_count <= 0:
            return [], []

        lane_sides = self.assign_npc_lane_sides(active_npc_count)
        seg_ids, seg_n = self._select_segment_ids(active_npc_count)
        # v11.3: bad_nodes note, note
        candidates = list(range(n))
        # note, note NPC note
        start_exclusion = 5
        candidates = [c for c in candidates if min(c, n - c) >= start_exclusion]
        if len(candidates) < max(10, active_npc_count * 3):
            candidates = list(range(n))  # note

        poses = []
        meta = []
        for k in range(active_npc_count):
            seg_id = seg_ids[k]
            start = int(seg_id * n / seg_n)
            end = max(start, int((seg_id + 1) * n / seg_n) - 1)
            seg_candidates = [i for i in candidates if start <= i <= end]
            if not seg_candidates:
                seg_candidates = list(range(start, end + 1)) if end >= start else [start % n]
            # note, note(note)
            if len(seg_candidates) >= 5:
                seg_candidates = seg_candidates[1:-1] or seg_candidates
            anchor = random.choice(seg_candidates) % n
            pose = self._sample_pose_from_node(anchor, jitter_scale=1.0, exact=False, lane_side=lane_sides[k])
            if pose is None:
                continue
            pose["segment_id"] = seg_id
            poses.append(pose)
            meta.append({"segment_id": seg_id, "lane_side": lane_sides[k]})
        return poses, meta

    def _build_npc_target_poses(self, active_npc_count):
        """notestagegenerate NPC goalnote(v10.1: B2 note segmented + lane-balanced)."""
        if active_npc_count <= 0:
            return [], []
        stage = self._current_stage_id()
        if stage == 3:  # B2: staticnote3note
            poses, meta = self.build_segmented_npc_layout(active_npc_count)
            if len(poses) == active_npc_count:
                return poses, meta

        # fallback: note anchor note, note lane-side note
        ego_anchor, npc_anchors = self._pick_ego_and_npc_anchor_nodes(active_npc_count)
        lane_sides = self.assign_npc_lane_sides(active_npc_count)
        poses, meta = [], []
        for i, a in enumerate(npc_anchors):
            side = lane_sides[i] if i < len(lane_sides) else "center"
            pose = self._sample_pose_from_node(a, jitter_scale=1.0, exact=False, lane_side=side)
            if pose is None:
                continue
            pose["segment_id"] = None
            poses.append(pose)
            meta.append({"segment_id": None, "lane_side": side})
        return poses, meta

    def _cache_npc_layout(self, npc_records, npc_target_poses):
        cached = []
        lane_assignments = []
        seg_assignments = []
        pose_map = {}
        for p in npc_target_poses:
            pose_map[int(p.get("node_idx", -1))] = p
        for rec in npc_records:
            target_pose = rec.get("target_pose")
            if target_pose is None:
                target_pose = pose_map.get(int(rec.get("npc_spawn_idx", -1)))
            lane_side = (target_pose or {}).get("lane_side", "center")
            segment_id = (target_pose or {}).get("segment_id")
            cached.append({
                "npc_id": rec["npc_id"],
                "pose": dict(target_pose) if target_pose else None,
                "lane_side": lane_side,
                "segment_id": segment_id,
            })
            lane_assignments.append(lane_side)
            seg_assignments.append(segment_id)
            self._lane_side_seen_counter[lane_side] += 1
            if segment_id is not None:
                self._segment_seen_counter[int(segment_id)] += 1
        self.npc_layout_cached_poses = cached
        self.npc_layout_lane_assignments = lane_assignments
        self.npc_layout_segment_assignments = seg_assignments
        self._npc_layout_stage_id = self._current_stage_id()
        self._npc_layout_active_count = len(cached)

    def _apply_cached_npc_layout(self, active_npcs):
        """notecurrentnoteNPC(noteteleport), note."""
        if not self.npc_layout_cached_poses:
            return False, [], "no_cached_layout"
        if len(active_npcs)!= len(self.npc_layout_cached_poses):
            return False, [], "cached_count_mismatch"
        records = []
        by_id = {c["npc_id"]: c for c in self.npc_layout_cached_poses if c.get("pose")}
        for npc in active_npcs:
            c = by_id.get(npc.npc_id)
            if not c or not c.get("pose"):
                return False, [], "cached_pose_missing"
            pose = c["pose"]
            # note env.reset note NPC notegoalnote, note; note
            nx, ny, nz = npc.get_telemetry_position()
            tx, ty, tz = pose["tel"]
            d = math.sqrt((nx - tx) ** 2 + (nz - tz) ** 2)
            if d > self.spawn_verify_tol_sim * 1.2:
                npc_res = self._apply_spawn_and_verify_npc(npc, pose, hold_static=True)
            else:
                fi = pose.get("fine_idx", 0)
                if self.track_cache:
                    fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, nx, nz)
                npc.fine_track_idx = fi
                npc_res = {
                    "ok": True,
                    "dist_err": d,
                    "stabilize_steps": 0,
                    "tel": (nx, ny, nz),
                    "fine_idx": fi,
                }
            if not npc_res.get("ok"):
                return False, records, "cached_npc_verify_failed"
            records.append({
                "npc_id": npc.npc_id,
                "npc_spawn_idx": int(pose.get("node_idx", -1)),
                "npc_fine_idx": int(npc_res.get("fine_idx", pose.get("fine_idx", 0))),
                "dist_err": float(npc_res.get("dist_err", 0.0)),
                "tel": npc_res.get("tel", pose["tel"]),
                "target_pose": pose,
            })
        return True, records, None

    def should_refresh_npc_layout(self, active_npcs):
        count = len(active_npcs)
        if count <= 0:
            return False, "no_active_npc"
        stage = self._current_stage_id()
        policy = self._layout_reset_policy()

        if bool(getattr(self, "_force_npc_layout_refresh_next", False)):
            self._force_npc_layout_refresh_next = False
            return True, "force_refresh_flag"

        if not self.npc_layout_cached_poses:
            return True, "no_cached_layout"
        if self._npc_layout_active_count!= count:
            return True, "active_count_changed"
        if self._npc_layout_stage_id!= stage:
            return True, "stage_changed"
        if self._consecutive_layout_failures >= self._layout_fail_refresh_threshold():
            return True, "consecutive_layout_failures"

        if policy == "every_episode":
            return True, "policy_every_episode"
        if policy == "agent_only":
            return False, "policy_agent_only"

        # hybrid
        if not self._is_static_layout_stage():
            # dynamicstagedefaultnote(note policy note)
            return True, "hybrid_dynamic_default"
        # v11.3: note N note(default2note)note NPC note
        reset_every_laps = int(self.curriculum_stage_ref.get("npc_layout_reset_every_laps", 2))
        if reset_every_laps > 0:
            laps_since_reset = self._total_laps_completed - self._npc_layout_last_reset_at_laps
            if laps_since_reset >= reset_every_laps:
                return True, f"lap_count_threshold({laps_since_reset}laps)"
        if self.npc_layout_age_agent_resets >= self._layout_reset_every():
            return True, "layout_age_threshold"
        if self._last_episode_collision and self._layout_reset_on_collision():
            return True, "last_episode_collision"
        if self._last_episode_success_2laps and self._layout_reset_on_success():
            return True, "last_episode_success"
        return False, "hybrid_reuse"

    def _refresh_npc_layout_mid_episode(self):
        """v11.3: noterowsnote, note NPC(note ego)."""
        active_npcs = self._active_npcs()
        if not active_npcs:
            return False
        npc_count = len(active_npcs)

        # note NPC note ego currentnote
        try:
            h = self.env.viewer.handler
            ego_tel = (float(h.x), float(h.y), float(h.z))
        except Exception:
            return False
        ego_fi = 0
        if self.track_cache:
            ego_fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, ego_tel[0], ego_tel[2])

        # v11.4: note ego-first note(NPC note ego firstnote)
        use_ego_first = (str(getattr(self, "v11_2_mode", "")).lower() == "fixed2"
                         and bool(getattr(getattr(self, "manual_spawn", None), "loaded", False))
                         and self._current_stage_id() in (2, 3))
        if use_ego_first:
            npc_target_poses = self._build_npc_ahead_of_ego(npc_count, ego_fi)
        else:
            npc_target_poses, _meta = self._build_npc_target_poses(npc_count)

        if len(npc_target_poses)!= npc_count:
            print(f"⚠️ mid-episode NPCnote: notefailed ({len(npc_target_poses)}/{npc_count})")
            return False

        ego_pose_for_check = {"tel": ego_tel, "fine_idx": ego_fi}
        # mid-episode note: ego noterowsnote, note NPC note
        # note 2.5m, progress_gap note 60(track 1080 fine note)
        mid_constraints = self._spawn_constraints()
        mid_ego_npc = mid_constraints["ego_npc"]
        hard_e = self._ego_npc_spawn_hard_floor_sim()
        hard_g = self._ego_npc_spawn_hard_floor_progress()
        mid_ego_npc["min_euclid"] = max(hard_e, min(float(mid_ego_npc["min_euclid"]), 2.5))
        mid_ego_npc["min_progress_gap"] = max(hard_g, min(int(mid_ego_npc["min_progress_gap"]), 60))

        # note 10 note
        for attempt in range(10):
            valid, violations = self.validate_spawn_separation(
                ego_pose_for_check, npc_target_poses, constraints=mid_constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc())
            if valid:
                break
            npc_target_poses, npc_layout_meta = self._build_npc_target_poses(npc_count)
            if len(npc_target_poses)!= npc_count:
                return False
        else:
            print(f"⚠️ mid-episode NPCnote: notefailed, note")
            return False

        # teleport NPC note
        npc_records = []
        npc_mode = self._normalize_npc_mode(self.curriculum_stage_ref.get("npc_mode", "static"))
        for i, (npc, pose) in enumerate(zip(active_npcs, npc_target_poses)):
            if not npc.connected:
                continue
            npc_res = self._apply_spawn_and_verify_npc(npc, pose, hold_static=(npc_mode == "static"))
            if not npc_res.get("ok"):
                continue
            npc_records.append({
                "npc_id": npc.npc_id,
                "npc_spawn_idx": int(pose["node_idx"]),
                "npc_fine_idx": int(npc_res.get("fine_idx", pose.get("fine_idx", 0))),
                "dist_err": float(npc_res.get("dist_err", 0.0)),
                "tel": npc_res.get("tel", pose["tel"]),
                "target_pose": pose,
            })

        if npc_records:
            self._cache_npc_layout(npc_records, npc_target_poses)
            self.npc_layout_age_agent_resets = 0
            self._npc_layout_last_reset_at_laps = self._total_laps_completed
            self.npc_layout_id += 1
            self.npc_layout_reset_count += 1
            # note wobble note, note NPC note
            if npc_mode == "wobble":
                for npc in active_npcs:
                    npc.set_mode("wobble", 0.0)
                    if not npc.running:
                        npc.start_driving()
            print(f"🔄 mid-episode NPCnote: {len(npc_records)}note (layout_id={self.npc_layout_id}, total_laps={self._total_laps_completed})")
            return True
        return False

    def _spawn_agent_only_against_existing_npcs(self, active_npcs):
        """v10.1: note Agent, note(noteteleport)note NPC note."""
        spawn_debug = {
            "reset_idx": self._reset_index,
            "stage": self._current_stage_id(),
            "ego_spawn_idx": None,
            "ego_fine_idx": None,
            "npc": [],
            "spawn_retries": 0,
            "spawn_validation_pass": False,
            "spawn_fail_reason": None,
            "ego_npc_min_dist": None,
            "ego_npc_progress_gap": None,
            "post_spawn_cte": None,
            "stabilize_steps": 0,
            "layout_reused": True,
            "npc_layout_id": self.npc_layout_id,
            "npc_layout_age_agent_resets": self.npc_layout_age_agent_resets,
            "npc_layout_last_reset_reason": self.npc_layout_last_reset_reason,
            "attempt_fail_counts": {},
            "bad_nodes_count": len(self._bad_nodes),
            "safe_anchor_count": 0,
        }
        attempt_fail_counts = Counter()

        if self._npc_persist_mode_enabled():
            ok_cached, npc_records, reason = self._capture_runtime_npc_records(active_npcs)
        else:
            ok_cached, npc_records, reason = self._apply_cached_npc_layout(active_npcs)
        if not ok_cached:
            spawn_debug["spawn_fail_reason"] = reason
            if reason in ("runtime_npc_offtrack", "runtime_npc_disconnected"):
                self._force_npc_layout_refresh_next = True
            return {"ok": False, "obs": None, "info": {}, "debug": spawn_debug}

        constraints = self._spawn_constraints()
        ego_constraints = constraints["ego_npc"]
        safe_ego_anchors = []
        fixed2_mode = str(getattr(self, "v11_2_mode", "")).lower() == "fixed2"
        manual_spawn_loaded = bool(getattr(getattr(self, "manual_spawn", None), "loaded", False))
        if fixed2_mode and manual_spawn_loaded and npc_records:
            safe_ego_anchors = self._fixed2_collect_safe_ego_anchors(
                [{"tel": r.get("tel"), "fine_idx": r.get("npc_fine_idx", 0)} for r in npc_records],
                constraints,
            )
        spawn_debug["safe_anchor_count"] = int(len(safe_ego_anchors))
        max_attempts = max(8, int(self.curriculum_stage_ref.get("spawn_max_attempts", 8)))
        final_obs, final_info = None, {}
        last_violations = []

        nodes = self._track_nodes()
        if not nodes:
            spawn_debug["spawn_fail_reason"] = "no_track_nodes"
            attempt_fail_counts["no_track_nodes"] += 1
            spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
            return {"ok": False, "obs": None, "info": {}, "debug": spawn_debug}

        for attempt in range(max_attempts):
            spawn_debug["spawn_retries"] = attempt
            # Agent-only note: note, failednote
            exact = attempt < 2
            jitter_scale = 0.7 if attempt < (max_attempts // 2) else 1.0
            if safe_ego_anchors:
                ego_anchor = random.choice(safe_ego_anchors)
            else:
                ego_anchor, _ = self._pick_ego_and_npc_anchor_nodes(0)
            if ego_anchor is None:
                spawn_debug["spawn_fail_reason"] = "no_ego_anchor"
                attempt_fail_counts["no_ego_anchor"] += 1
                break
            ego_pose = self._sample_pose_from_node(ego_anchor, jitter_scale=jitter_scale, exact=exact, lane_side="center")
            if ego_pose is None:
                continue

            # v11.5 note
            if not self._precheck_pose_viable(ego_pose):
                attempt_fail_counts["ego_precheck_off_track"] = attempt_fail_counts.get("ego_precheck_off_track", 0) + 1
                continue

            valid_sep, last_violations = self.validate_spawn_separation(
                ego_pose,
                [{"tel": r["tel"], "fine_idx": r["npc_fine_idx"]} for r in npc_records],
                constraints=constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc(),
            )
            if not valid_sep:
                spawn_debug["spawn_fail_reason"] = "precheck_separation_agent_only"
                attempt_fail_counts["precheck_separation_agent_only"] += 1
                for v in last_violations:
                    self.spawn_precheck_fail_stats[v.get("type", "unknown")] += 1
                continue

            ego_res = self._apply_spawn_and_verify_ego(ego_pose, retries_step=self._spawn_ego_verify_retries())
            final_obs, final_info = ego_res.get("obs"), ego_res.get("info", {})
            spawn_debug["stabilize_steps"] += ego_res.get("stabilize_steps", 0)
            spawn_debug["post_spawn_cte"] = ego_res.get("cte_after")
            if not ego_res.get("ok"):
                spawn_debug["spawn_fail_reason"] = "ego_verify_failed"
                attempt_fail_counts["ego_verify_failed"] += 1
                # v11.3: note ego_verify_failed note node--
                # teleport note CTE note, note.
                # note safe_ego_anchors note, note.
                if safe_ego_anchors and ego_pose["node_idx"] is not None:
                    bad_idx = int(ego_pose["node_idx"])
                    safe_ego_anchors = [a for a in safe_ego_anchors if int(a)!= bad_idx]
                spawn_debug["bad_nodes_count"] = len(self._bad_nodes)
                continue

            self.last_active_node = int(ego_pose["node_idx"])
            spawn_debug["ego_spawn_idx"] = int(ego_pose["node_idx"])
            spawn_debug["ego_fine_idx"] = int(ego_pose.get("fine_idx", 0))

            # note
            ex, ey, ez = self._extract_pos(final_info)
            ef = ego_pose.get("fine_idx", 0)
            if self.track_cache:
                ef, _ = self.track_cache.find_nearest_fine_track(self.scene_name, ex, ez)
            ego_real = {"tel": (ex, ey, ez), "fine_idx": ef}
            valid_sep2, last_violations = self.validate_spawn_separation(
                ego_real,
                [{"tel": r["tel"], "fine_idx": r["npc_fine_idx"]} for r in npc_records],
                constraints=constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc(),
            )
            if not valid_sep2:
                spawn_debug["spawn_fail_reason"] = "postcheck_separation_agent_only"
                attempt_fail_counts["postcheck_separation_agent_only"] += 1
                for v in last_violations:
                    self.spawn_precheck_fail_stats["post_" + v.get("type", "unknown")] += 1
                continue

            min_dist, min_gap = None, None
            for r in npc_records:
                nx, ny, nz = r["tel"]
                d = math.sqrt((ex - nx) ** 2 + (ez - nz) ** 2)
                g = abs(self.track_cache.progress_diff(self.scene_name, ef, r["npc_fine_idx"])) if self.track_cache else 0
                min_dist = d if min_dist is None else min(min_dist, d)
                min_gap = g if min_gap is None else min(min_gap, g)
            spawn_debug["npc"] = npc_records
            spawn_debug["ego_npc_min_dist"] = min_dist
            spawn_debug["ego_npc_progress_gap"] = min_gap
            spawn_debug["spawn_validation_pass"] = True
            spawn_debug["spawn_fail_reason"] = None
            spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
            if self.npc_layout_debug:
                print(f"🧩 NPCnote id={self.npc_layout_id} age={self.npc_layout_age_agent_resets} lanes={self.npc_layout_lane_assignments} segs={self.npc_layout_segment_assignments}")
            return {"ok": True, "obs": final_obs, "info": final_info, "debug": spawn_debug}

        spawn_debug["spawn_fail_reason"] = spawn_debug.get("spawn_fail_reason") or "agent_only_spawn_exhausted"
        attempt_fail_counts[spawn_debug["spawn_fail_reason"]] += 1
        spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
        if last_violations:
            spawn_debug["violations"] = last_violations[: self.spawn_debug_violations_limit]
        self.spawn_failure_reason_stats[spawn_debug["spawn_fail_reason"]] += 1
        return {"ok": False, "obs": final_obs, "info": final_info, "debug": spawn_debug}

    def _sample_pose_from_node(self, node_idx, jitter_scale=1.0, exact=False, lane_side=None):
        nodes = self._track_nodes()
        if not nodes:
            return None
        n = len(nodes)
        node_idx = int(node_idx) % n
        node = nodes[node_idx]
        prev_node = nodes[(node_idx - 1) % n]
        next_node = nodes[(node_idx + 1) % n]

        x0, y0, z0 = self._node_tel(node)
        xp, _, zp = self._node_tel(prev_node)
        xn, _, zn = self._node_tel(next_node)

        tx = xn - xp
        tz = zn - zp
        norm = math.sqrt(tx * tx + tz * tz)
        if norm < 1e-6:
            tx, tz = 0.0, 1.0
        else:
            tx, tz = tx / norm, tz / norm
        nx, nz = -tz, tx

        if exact:
            ds = 0.0
            dn = 0.0
            yaw_jitter = 0.0
        else:
            ds = random.uniform(-self.spawn_jitter_s_sim, self.spawn_jitter_s_sim) * jitter_scale
            dn = random.uniform(-self.spawn_jitter_d_sim, self.spawn_jitter_d_sim) * jitter_scale
            yaw_jitter = random.uniform(-self.spawn_yaw_jitter_deg, self.spawn_yaw_jitter_deg) * jitter_scale

        dn_base = 0.0
        if lane_side is not None:
            lane_side = str(lane_side)
            off_scale, jit_scale, abs_cap, centerline_cap = self._lane_offset_stage_limits()
            dn_base = self._lane_offset_value(lane_side) * off_scale
            dn += random.uniform(-self.npc_lane_jitter_sim, self.npc_lane_jitter_sim) * max(0.4, jitter_scale) * jit_scale

        dn_total = dn_base + dn
        if lane_side is not None:
            _, _, abs_cap, _ = self._lane_offset_stage_limits()
            dn_total = float(np.clip(dn_total, -abs_cap, abs_cap))
        tel_x = x0 + ds * tx + dn_total * nx
        tel_z = z0 + ds * tz + dn_total * nz
        tel_y = y0

        yaw_deg = math.degrees(math.atan2(tx, tz)) + yaw_jitter
        qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)

        S = SimExtendedAPI.COORD_SCALE
        fine_idx = 0
        fine_dist = 0.0
        if self.track_cache:
            fine_idx, fine_dist = self.track_cache.find_nearest_fine_track(self.scene_name, tel_x, tel_z)
            # notetracknote, note(notetracknote)
            if lane_side is not None and (not exact):
                _, _, _, centerline_cap = self._lane_offset_stage_limits()
                if fine_dist > centerline_cap:
                    dn_total *= 0.5
                    tel_x = x0 + ds * tx + dn_total * nx
                    tel_z = z0 + ds * tz + dn_total * nz
                    fine_idx, fine_dist = self.track_cache.find_nearest_fine_track(self.scene_name, tel_x, tel_z)

        return {
            "node_idx": node_idx,
            "fine_idx": fine_idx,
            "tel": (tel_x, tel_y, tel_z),
            "node_coords": (tel_x * S, tel_y * S, tel_z * S, qx, qy, qz, qw),
            "yaw_deg": yaw_deg,
            "lane_side": lane_side or "center",
            "dn_offset_sim": dn_total,
            "fine_dist_sim": fine_dist,
        }

    def _apply_spawn_and_verify_ego(self, pose, retries_step=2, skip_cte_check=False):
        h = self.env.viewer.handler
        px, py, pz, qx, qy, qz, qw = pose["node_coords"]
        SimExtendedAPI.send_set_position(h, px, py, pz, qx, qy, qz, qw)

        post_info = {}
        obs = None
        stabilize_steps = 0
        confirmed = False
        dist_err = 999.0
        cte_after = 999.0
        resend_count = 0
        geom_ok = False
        geom_meta = {}
        cte_waived = False
        max_loops = max(1, retries_step)

        def _collect_metrics(info):
            lx, _, lz = self._extract_pos(info)
            tx, _, tz = pose["tel"]
            d_err = math.sqrt((lx - tx) ** 2 + (lz - tz) ** 2)
            cte_val = abs(_safe_float(info.get("cte", 0.0), 0.0))
            g_ok, g_meta = self._geometry_track_check(
                lx, lz,
                seed_fi=int(_safe_float(pose.get("fine_idx", 0), 0)),
                margin_scale=0.35,
                centerline_slack=0.18,
            )
            speed_abs = abs(_safe_float(info.get("speed", 0.0), 0.0))
            return lx, lz, tx, tz, d_err, cte_val, g_ok, g_meta, speed_abs

        for i in range(max_loops):
            obs, post_info = self._idle_step_for_telemetry(1)
            stabilize_steps += 1
            lx, lz, tx, tz, dist_err, cte_after, geom_ok, geom_meta, _speed_abs = _collect_metrics(post_info)
            # v11.5: NPC note CTE datanote(23/33 note),
            # note dist_err(teleport note), note CTE note
            # v11.5 fix: skip_cte_check note CTE (> 15.0) - note 1-step note
            if skip_cte_check:
                cte_ok = cte_after <= 15.0
            else:
                cte_ok = cte_after <= self.spawn_verify_cte_threshold

                # geometrynote + teleportnote, note1note, note CTE note.
                if (not cte_ok) and geom_ok and dist_err <= self.spawn_verify_tol_sim:
                    if (i + 1) < max_loops:
                        obs, post_info = self._idle_step_for_telemetry(1)
                        stabilize_steps += 1
                        lx, lz, tx, tz, dist_err, cte_after, geom_ok, geom_meta, _speed_abs = _collect_metrics(post_info)
                        cte_ok = cte_after <= self.spawn_verify_cte_threshold

                # v11.7: geometrynotetracknote CTE note(note).
                if (not cte_ok) and geom_ok and dist_err <= self.spawn_verify_tol_sim:
                    cte_soft_cap = min(9.5, float(self.spawn_verify_cte_threshold) + 1.0)
                    cte_ok = cte_after <= cte_soft_cap
                # CTE telemetry notefirstnote; geometrynoteteleportnoterows.
                if (not cte_ok) and geom_ok and dist_err <= self.spawn_verify_tol_sim and _speed_abs <= 0.35:
                    cte_ok = True
                    cte_waived = True
            if dist_err <= self.spawn_verify_tol_sim and cte_ok:
                confirmed = True
                break
            # v11.8: note dist_err note teleport, note(1.0~2.0)notefailednote.
            if dist_err > self.spawn_verify_tol_sim and resend_count < 3:
                resend_count += 1
                time.sleep(0.08)
                SimExtendedAPI.send_set_position(h, px, py, pz, qx, qy, qz, qw)
                time.sleep(0.08)
        # v11.5 note: failednote
        if not confirmed:
            print(f"    🔍 ego_verify FAIL node={pose.get('node_idx')} dist_err={dist_err:.3f}(tol={self.spawn_verify_tol_sim:.2f}) "
                  f"cte={cte_after:.2f}(tol={self.spawn_verify_cte_threshold:.1f}) steps={stabilize_steps} "
                  f"actual=({lx:.3f},{lz:.3f}) target=({tx:.3f},{tz:.3f}) skip_cte={skip_cte_check} "
                  f"geom_ok={geom_ok} geom_fd={_safe_float((geom_meta or {}).get('fine_dist_sim', -1.0), -1.0):.3f} "
                  f"cte_waived={cte_waived}")
        return {
            "ok": confirmed,
            "obs": obs,
            "info": post_info,
            "cte_after": cte_after,
            "dist_err": dist_err,
            "stabilize_steps": stabilize_steps,
            "cte_waived": cte_waived,
        }

    def _apply_spawn_and_verify_npc(self, npc, pose, hold_static=True):
        px, py, pz, qx, qy, qz, qw = pose["node_coords"]
        npc.set_position_node_coords(px, py, pz, qx, qy, qz, qw)
        if hold_static and npc.mode == "static":
            try:
                npc.handler.send_control(0, 0, 1.0)
                time.sleep(0.05)
                npc.handler.send_control(0, 0, 1.0)
            except Exception:
                pass
        time.sleep(0.12)
        nx, ny, nz = npc.get_telemetry_position()
        tx, ty, tz = pose["tel"]
        dist_err = math.sqrt((nx - tx) ** 2 + (nz - tz) ** 2)
        fine_idx = pose.get("fine_idx", 0)
        fine_dist = float(pose.get("fine_dist_sim", 0.0))
        if self.track_cache:
            fine_idx, fine_dist = self.track_cache.find_nearest_fine_track(self.scene_name, nx, nz)
        npc.fine_track_idx = fine_idx
        # v10.12: NPC note"notetracknote", notetracknote
        _, _, _, centerline_cap = self._lane_offset_stage_limits()
        # noteNPCnote, notetrack
        npc_centerline_cap = max(float(centerline_cap), 0.45) + 0.08
        on_track_ok = float(fine_dist) <= float(npc_centerline_cap)
        return {
            "ok": (dist_err <= self.spawn_verify_tol_sim * 1.5) and on_track_ok,
            "dist_err": dist_err,
            "stabilize_steps": 1,
            "tel": (nx, ny, nz),
            "fine_idx": fine_idx,
            "fine_dist_sim": float(fine_dist),
            "on_track_ok": bool(on_track_ok),
        }

    def validate_spawn_separation(self, ego_pose, npc_poses, min_euclid=None, min_progress_gap=None,
                                  constraints=None, fail_on_npc_npc=True):
        """v10.1: pair-wise note, noteclassnote."""
        if not self.track_cache:
            return True, []

        # note
        if constraints is None:
            if min_euclid is None:
                min_euclid = self.dist_scale.get("spawn_min_gap_sim", 3.5)
            if min_progress_gap is None:
                min_progress_gap = self.dist_scale.get("spawn_min_gap_progress", 30)
            constraints = {
                "ego_npc": {
                    "min_euclid": float(min_euclid),
                    "min_progress_gap": int(min_progress_gap),
                    "require_both": True,
                },
                "npc_npc": {
                    "min_euclid": float(min_euclid),
                    "min_progress_gap": int(min_progress_gap),
                    "require_progress": True,
                },
            }

        c_ego = constraints.get("ego_npc", {})
        c_npc = constraints.get("npc_npc", {})

        ex, _, ez = ego_pose["tel"]
        ef = ego_pose.get("fine_idx", 0)
        violations = []

        for i, p in enumerate(npc_poses):
            nx, _, nz = p["tel"]
            nf = p.get("fine_idx", 0)
            euclid = math.sqrt((ex - nx) ** 2 + (ez - nz) ** 2)
            gap = abs(self.track_cache.progress_diff(self.scene_name, ef, nf))
            min_e = float(c_ego.get("min_euclid", 0.0))
            min_g = int(c_ego.get("min_progress_gap", 0))
            bad_e = euclid < min_e
            bad_g = gap < min_g
            if bad_e or bad_g:
                if bool(c_ego.get("require_both", True)):
                    violations.append({
                        "type": "ego_npc_both" if (bad_e and bad_g) else ("ego_npc_euclid" if bad_e else "ego_npc_progress"),
                        "npc_i": i,
                        "euclid": euclid,
                        "progress_gap": gap,
                        "target_euclid": min_e,
                        "target_progress_gap": min_g,
                    })
                else:
                    # note: note
                    if bad_e and bad_g:
                        violations.append({
                            "type": "ego_npc_both",
                            "npc_i": i,
                            "euclid": euclid,
                            "progress_gap": gap,
                            "target_euclid": min_e,
                            "target_progress_gap": min_g,
                        })

        for i in range(len(npc_poses)):
            for j in range(i + 1, len(npc_poses)):
                ax, _, az = npc_poses[i]["tel"]
                bx, _, bz = npc_poses[j]["tel"]
                af = npc_poses[i].get("fine_idx", 0)
                bf = npc_poses[j].get("fine_idx", 0)
                euclid = math.sqrt((ax - bx) ** 2 + (az - bz) ** 2)
                gap = abs(self.track_cache.progress_diff(self.scene_name, af, bf))
                min_e = float(c_npc.get("min_euclid", 0.0))
                min_g = int(c_npc.get("min_progress_gap", 0))
                require_progress = bool(c_npc.get("require_progress", False))

                # NPC-NPC note hard-fail note fail_on_npc_npc control(noteconfigurationnote).
                if euclid < min_e and fail_on_npc_npc:
                    violations.append({
                        "type": "npc_npc_overlap",
                        "npc_pair": [i, j],
                        "euclid": euclid,
                        "progress_gap": gap,
                        "target_euclid": min_e,
                        "target_progress_gap": min_g,
                    })
                elif require_progress and gap < min_g and fail_on_npc_npc:
                    violations.append({
                        "type": "npc_npc_progress",
                        "npc_pair": [i, j],
                        "euclid": euclid,
                        "progress_gap": gap,
                        "target_euclid": min_e,
                        "target_progress_gap": min_g,
                    })

        return len(violations) == 0, violations

    def _pick_ego_and_npc_anchor_nodes(self, active_npc_count):
        """notestagenote anchor notedistribution, dynamicstagenote ego note."""
        nodes = self._track_nodes()
        n = len(nodes)
        if n == 0:
            return None, []

        stage = int(self.curriculum_stage_ref.get("stage", 1))
        prefer_ego_behind = random.random() < float(self.curriculum_stage_ref.get("p_ego_behind", 0.0))
        min_gap_progress = int(self._spawn_constraints()["ego_npc"]["min_progress_gap"])
        fine = self._track_fine()
        # note fine_gap -> node_gap
        approx_node_gap = max(3, int(round(min_gap_progress / max(1, len(fine) / max(1, n))))) if fine else 8

        # v11.3: bad_nodes note(note ego_verify_failed note), note
        candidates = list(range(n))
        # v10.5: notetracknote ego note, note
        num_bins = max(4, int(self._spawn_bins))
        bins = [[] for _ in range(num_bins)]
        for c in candidates:
            b = min(num_bins - 1, int((c / float(max(1, n))) * num_bins))
            bins[b].append(c)
        non_empty_bins = [i for i, arr in enumerate(bins) if arr]
        if non_empty_bins:
            min_count = min(self._spawn_bin_counts.get(b, 0) for b in non_empty_bins)
            least_bins = [b for b in non_empty_bins if self._spawn_bin_counts.get(b, 0) == min_count]
            chosen_bin = random.choice(least_bins)
            ego_anchor = random.choice(bins[chosen_bin])
            self._spawn_bin_counts[chosen_bin] += 1
        else:
            ego_anchor = random.choice(candidates)
        npc_anchors = []

        if active_npc_count <= 0:
            return ego_anchor, npc_anchors

        # note: noteNPCnoteEgonote, notelane offsetnote
        if bool(self.curriculum_stage_ref.get("npc_side_by_side_start", False)):
            npc_anchors.append(ego_anchor)
            while len(npc_anchors) < active_npc_count:
                npc_anchors.append((ego_anchor + 6 * len(npc_anchors)) % n)
            return ego_anchor, npc_anchors

        # note: noteNPCnoteEgonotefirstnote(note, note)
        if bool(self.curriculum_stage_ref.get("npc_front_start", False)):
            front_off_cfg = int(self.curriculum_stage_ref.get("npc_front_offset_nodes", 8))
            front_off = max(3, max(front_off_cfg, approx_node_gap))
            npc_anchors.append((ego_anchor + front_off) % n)
            while len(npc_anchors) < active_npc_count:
                step = front_off + 4 * len(npc_anchors)
                npc_anchors.append((ego_anchor + step) % n)
            return ego_anchor, npc_anchors

        if stage in (4, 5) and prefer_ego_behind:
            # dynamicstage: note ego note, NPC notefirstnote
            base_offsets = [random.randint(max(5, approx_node_gap // 2), max(12, approx_node_gap + 4))]
            while len(base_offsets) < active_npc_count:
                base_offsets.append(base_offsets[-1] + random.randint(3, 6))
            for off in base_offsets[:active_npc_count]:
                npc_anchors.append((ego_anchor + off) % n)
        else:
            # staticstage/note: noteNPCnote, note ego note(note)
            # note(node 0 ±5), note NPC note
            start_exclusion = max(5, approx_node_gap // 2)
            npc_candidates = [c for c in candidates
                              if min(c, n - c) >= start_exclusion]
            if len(npc_candidates) < active_npc_count + 2:
                npc_candidates = candidates  # note: note
            used = {ego_anchor}
            for i in range(active_npc_count):
                for _ in range(20):
                    cand = random.choice(npc_candidates)
                    if cand in used:
                        continue
                    # note: note(note)
                    delta = abs(cand - ego_anchor)
                    delta = min(delta, n - delta)
                    if delta < max(3, approx_node_gap // 2):
                        continue
                    if any(min(abs(cand - u), n - abs(cand - u)) < 3 for u in used):
                        continue
                    npc_anchors.append(cand)
                    used.add(cand)
                    break
            # note
            while len(npc_anchors) < active_npc_count:
                npc_anchors.append((ego_anchor + (len(npc_anchors) + 1) * max(4, approx_node_gap)) % n)

        return ego_anchor, npc_anchors

    def _spawn_episode_layout(self, refresh_npc_layout=True, refresh_reason=None):
        """v10.1 unifiedspawnnote: note NPC layout note(agent-only/hybrid)."""
        active_npcs = self._active_npcs()
        inactive_npcs = self._inactive_npcs()
        for npc in inactive_npcs:
            self._hide_npc_offtrack(npc)

        if not self.track_cache or not self._track_nodes():
            return {"ok": False, "reason": "no_track_nodes"}

        for npc in active_npcs:
            setattr(npc, "_hidden_offtrack", False)

        # v11.8: spawnstagenoteNPC, notesucceedednoterowsnote, note/note
        npc_mode = self._normalize_npc_mode(self.curriculum_stage_ref.get("npc_mode", "offtrack"))
        npc_speed_min = float(self.curriculum_stage_ref.get("npc_speed_min", 0.0))
        npc_speed_max = float(self.curriculum_stage_ref.get("npc_speed_max", 0.0))
        should_freeze_npcs_for_spawn = not ((not refresh_npc_layout) and self._npc_persist_mode_enabled())
        if should_freeze_npcs_for_spawn:
            self._apply_active_npc_runtime_mode(
                active_npcs,
                npc_mode,
                npc_speed_min,
                npc_speed_max,
                freeze_for_spawn=True,
            )

        # noteNPCnote, note; note should_refresh_npc_layout note
        constraints = self._spawn_constraints()
        max_attempts = int(self.curriculum_stage_ref.get("spawn_max_attempts", 8))

        spawn_debug = {
            "reset_idx": self._reset_index,
            "stage": int(self.curriculum_stage_ref.get("stage", 1)),
            "ego_spawn_idx": None,
            "ego_fine_idx": None,
            "npc": [],
            "spawn_retries": 0,
            "spawn_validation_pass": False,
            "spawn_fail_reason": None,
            "ego_npc_min_dist": None,
            "ego_npc_progress_gap": None,
            "post_spawn_cte": None,
            "stabilize_steps": 0,
            "layout_reused": (not refresh_npc_layout),
            "npc_layout_id": self.npc_layout_id,
            "npc_layout_last_reset_reason": self.npc_layout_last_reset_reason,
            "npc_layout_age_agent_resets": self.npc_layout_age_agent_resets,
            "refresh_reason": refresh_reason,
            "attempt_fail_counts": {},
            "bad_nodes_count": len(self._bad_nodes),
            "safe_anchor_count": 0,
        }
        attempt_fail_counts = Counter()

        # noteNPCnote Ego(noteunifiedworkflow, note)
        if len(active_npcs) == 0:
            final_obs = None
            final_info = {}
            last_violations = []
            for attempt in range(max_attempts):
                spawn_debug["spawn_retries"] = attempt
                ego_anchor, _ = self._pick_ego_and_npc_anchor_nodes(0)
                if ego_anchor is None:
                    spawn_debug["spawn_fail_reason"] = "no_ego_anchor"
                    attempt_fail_counts["no_ego_anchor"] += 1
                    break
                exact = attempt < 2
                jitter_scale = 0.7 if attempt < max_attempts // 2 else 1.0
                ego_pose = self._sample_pose_from_node(ego_anchor, jitter_scale=jitter_scale, exact=exact, lane_side="center")
                if ego_pose is None:
                    continue
                # v11.5 note
                if not self._precheck_pose_viable(ego_pose):
                    attempt_fail_counts["ego_precheck_off_track"] = attempt_fail_counts.get("ego_precheck_off_track", 0) + 1
                    continue
                ego_result = self._apply_spawn_and_verify_ego(ego_pose, retries_step=self._spawn_ego_verify_retries())
                final_obs, final_info = ego_result.get("obs"), ego_result.get("info", {})
                spawn_debug["stabilize_steps"] += ego_result.get("stabilize_steps", 0)
                spawn_debug["post_spawn_cte"] = ego_result.get("cte_after")
                if not ego_result.get("ok"):
                    spawn_debug["spawn_fail_reason"] = "ego_verify_failed"
                    attempt_fail_counts["ego_verify_failed"] += 1
                    # v11.3: note ego_verify_failed note node
                    spawn_debug["bad_nodes_count"] = len(self._bad_nodes)
                    continue
                self.last_active_node = int(ego_pose["node_idx"])
                spawn_debug["ego_spawn_idx"] = int(ego_pose["node_idx"])
                spawn_debug["ego_fine_idx"] = int(ego_pose.get("fine_idx", 0))
                spawn_debug["spawn_validation_pass"] = True
                spawn_debug["spawn_fail_reason"] = None
                spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
                self._last_spawn_debug = spawn_debug
                return {"ok": True, "obs": final_obs, "info": final_info, "debug": spawn_debug}

            spawn_debug["spawn_fail_reason"] = spawn_debug.get("spawn_fail_reason") or "ego_only_spawn_exhausted"
            attempt_fail_counts[spawn_debug["spawn_fail_reason"]] += 1
            spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
            self.spawn_failure_reason_stats[spawn_debug["spawn_fail_reason"]] += 1
            self._last_spawn_debug = spawn_debug
            return {"ok": False, "obs": None, "info": {}, "debug": spawn_debug}

        if not refresh_npc_layout:
            result = self._spawn_agent_only_against_existing_npcs(active_npcs)
            dbg = result.get("debug") or {}
            if not result.get("ok"):
                fail_reason = dbg.get("spawn_fail_reason", "agent_only_spawn_failed")
                self.spawn_failure_reason_stats[fail_reason] += 1
                self._consecutive_layout_failures += 1
            else:
                if should_freeze_npcs_for_spawn:
                    self._apply_active_npc_runtime_mode(
                        active_npcs,
                        npc_mode,
                        npc_speed_min,
                        npc_speed_max,
                        freeze_for_spawn=False,
                    )
                self._consecutive_layout_failures = 0
            self._last_spawn_debug = dbg
            return result

        final_obs = None
        final_info = {}
        last_violations = []
        exact_fallback = False

        # v11.4: "ego-first" spawn note
        # fixed2 note NPC note, note ego note,
        # dynamicnote NPC note ego firstnote(note agent noteobstacle).
        use_ego_first = (str(getattr(self, "v11_2_mode", "")).lower() == "fixed2"
                         and bool(getattr(getattr(self, "manual_spawn", None), "loaded", False))
                         and self._current_stage_id() in (2, 3))

        if not use_ego_first:
            # noteworkflow: note NPC -> note ego
            npc_target_poses, npc_layout_meta = self._build_npc_target_poses(len(active_npcs))
            if len(npc_target_poses)!= len(active_npcs):
                spawn_debug["spawn_fail_reason"] = "npc_layout_build_failed"
                attempt_fail_counts["npc_layout_build_failed"] += 1
                spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
                self.spawn_failure_reason_stats[spawn_debug["spawn_fail_reason"]] += 1
                self._consecutive_layout_failures += 1
                self._last_spawn_debug = spawn_debug
                return {"ok": False, "obs": None, "info": {}, "debug": spawn_debug}
        else:
            npc_target_poses = []
            npc_layout_meta = []

        if self.npc_layout_debug and npc_target_poses:
            dbg_lanes = [p.get("lane_side", "center") for p in npc_target_poses]
            dbg_segs = [p.get("segment_id") for p in npc_target_poses]
            print(f"🧩 NPCnote pending -> lanes={dbg_lanes} segs={dbg_segs}")

        safe_ego_anchors = []
        fixed2_mode = str(getattr(self, "v11_2_mode", "")).lower() == "fixed2"
        manual_spawn_loaded = bool(getattr(getattr(self, "manual_spawn", None), "loaded", False))
        if fixed2_mode and manual_spawn_loaded and not use_ego_first:
            safe_ego_anchors = self._fixed2_collect_safe_ego_anchors(npc_target_poses, constraints)
        spawn_debug["safe_anchor_count"] = int(len(safe_ego_anchors))

        for attempt in range(max_attempts):
            spawn_debug["spawn_retries"] = attempt
            jitter_scale = 1.0
            if attempt >= max_attempts // 2:
                jitter_scale = 0.6
            if attempt >= max_attempts - 2:
                exact_fallback = True

            if safe_ego_anchors:
                ego_anchor = random.choice(safe_ego_anchors)
            else:
                ego_anchor, _npc_anchors_unused = self._pick_ego_and_npc_anchor_nodes(0)
            if ego_anchor is None:
                spawn_debug["spawn_fail_reason"] = "no_anchor"
                attempt_fail_counts["no_anchor"] += 1
                break

            ego_pose = self._sample_pose_from_node(ego_anchor, jitter_scale=jitter_scale, exact=exact_fallback, lane_side="center")
            if ego_pose is None:
                spawn_debug["spawn_fail_reason"] = "ego_pose_sample_failed"
                attempt_fail_counts["ego_pose_sample_failed"] += 1
                continue

            # v11.5 note: notetracknote, note(note teleport note)
            if not self._precheck_pose_viable(ego_pose):
                spawn_debug["spawn_fail_reason"] = "ego_precheck_off_track"
                attempt_fail_counts["ego_precheck_off_track"] = attempt_fail_counts.get("ego_precheck_off_track", 0) + 1
                continue

            # v11.4 ego-first: note ego note, note ego firstnotedynamicgenerate NPC note
            if use_ego_first:
                ego_fi = int(ego_pose.get("fine_idx", 0))
                npc_target_poses = self._build_npc_ahead_of_ego(len(active_npcs), ego_fi)
                if len(npc_target_poses)!= len(active_npcs):
                    spawn_debug["spawn_fail_reason"] = "ego_first_npc_build_failed"
                    attempt_fail_counts["ego_first_npc_build_failed"] += 1
                    continue

            npc_poses = [dict(p) for p in npc_target_poses]
            npc_precheck_bad = False
            for _p in npc_poses:
                if not self._precheck_pose_viable(_p):
                    npc_precheck_bad = True
                    break
            if npc_precheck_bad:
                spawn_debug["spawn_fail_reason"] = "npc_precheck_off_track"
                attempt_fail_counts["npc_precheck_off_track"] += 1
                continue

            # notegeometrynote(notefailed)
            valid_sep, last_violations = self.validate_spawn_separation(
                ego_pose,
                npc_poses,
                constraints=constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc(),
            )
            if not valid_sep:
                spawn_debug["spawn_fail_reason"] = "precheck_separation"
                attempt_fail_counts["precheck_separation"] += 1
                for v in last_violations:
                    self.spawn_precheck_fail_stats[v.get("type", "unknown")] += 1
                    print(f"  FAIL separation_violation: {v}")
                continue

            # 1) Ego teleport + note
            ego_result = self._apply_spawn_and_verify_ego(ego_pose, retries_step=self._spawn_ego_verify_retries())
            final_obs, final_info = ego_result.get("obs"), ego_result.get("info", {})
            spawn_debug["stabilize_steps"] += ego_result.get("stabilize_steps", 0)
            spawn_debug["post_spawn_cte"] = ego_result.get("cte_after")
            if not ego_result.get("ok"):
                spawn_debug["spawn_fail_reason"] = "ego_verify_failed"
                attempt_fail_counts["ego_verify_failed"] += 1
                # v11.3: note ego_verify_failed note node(teleportnote)
                if safe_ego_anchors and ego_pose["node_idx"] is not None:
                    bad_idx = int(ego_pose["node_idx"])
                    safe_ego_anchors = [a for a in safe_ego_anchors if int(a)!= bad_idx]
                spawn_debug["bad_nodes_count"] = len(self._bad_nodes)
                continue

            self.last_active_node = int(ego_pose["node_idx"])
            spawn_debug["ego_spawn_idx"] = int(ego_pose["node_idx"])
            spawn_debug["ego_fine_idx"] = int(ego_pose.get("fine_idx", 0))

            # 2) NPCnote teleport + note
            npc_records = []
            fail = False
            for i, (npc, pose) in enumerate(zip(active_npcs, npc_poses)):
                if not npc.connected:
                    continue
                npc_res = self._apply_spawn_and_verify_npc(npc, pose, hold_static=True)
                spawn_debug["stabilize_steps"] += npc_res.get("stabilize_steps", 0)
                if not npc_res.get("ok"):
                    fail = True
                    spawn_debug["spawn_fail_reason"] = "npc_verify_failed"
                    attempt_fail_counts["npc_verify_failed"] += 1
                    break
                npc_records.append({
                    "npc_id": npc.npc_id,
                    "npc_spawn_idx": int(pose["node_idx"]),
                    "npc_fine_idx": int(npc_res.get("fine_idx", pose.get("fine_idx", 0))),
                    "dist_err": float(npc_res.get("dist_err", 0.0)),
                    "tel": npc_res.get("tel", pose["tel"]),
                    "target_pose": pose,
                })

            if fail:
                continue

            # 3) note"notetelemetrynote"note(note telem lag / note)
            ego_real = {
                "tel": self._extract_pos(final_info),
                "fine_idx": self.track_cache.find_nearest_fine_track(self.scene_name, *self._extract_pos(final_info)[::2])[0]
                if self.track_cache else ego_pose.get("fine_idx", 0),
            }
            npc_real_poses = []
            for rec in npc_records:
                tx, ty, tz = rec["tel"]
                fi = rec["npc_fine_idx"]
                npc_real_poses.append({"tel": (tx, ty, tz), "fine_idx": fi})

            valid_sep2, last_violations = self.validate_spawn_separation(
                ego_real,
                npc_real_poses,
                constraints=constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc(),
            )
            if not valid_sep2:
                spawn_debug["spawn_fail_reason"] = "postcheck_separation"
                attempt_fail_counts["postcheck_separation"] += 1
                for v in last_violations:
                    self.spawn_precheck_fail_stats["post_" + v.get("type", "unknown")] += 1
                continue

            # succeeded, note
            min_dist = None
            min_gap = None
            ex, _, ez = ego_real["tel"]
            ef = ego_real["fine_idx"]
            for rec in npc_records:
                nx, ny, nz = rec["tel"]
                nf = rec["npc_fine_idx"]
                d = math.sqrt((ex - nx) ** 2 + (ez - nz) ** 2)
                g = abs(self.track_cache.progress_diff(self.scene_name, ef, nf)) if self.track_cache else 0
                min_dist = d if min_dist is None else min(min_dist, d)
                min_gap = g if min_gap is None else min(min_gap, g)
            spawn_debug["ego_npc_min_dist"] = min_dist
            spawn_debug["ego_npc_progress_gap"] = min_gap
            spawn_debug["npc"] = npc_records
            spawn_debug["spawn_validation_pass"] = True
            spawn_debug["spawn_fail_reason"] = None
            spawn_debug["npc_layout_id"] = self.npc_layout_id + 1
            spawn_debug["npc_lanes"] = [p.get("lane_side", "center") for p in npc_target_poses]
            spawn_debug["npc_segments"] = [p.get("segment_id") for p in npc_target_poses]
            spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
            self.npc_layout_id += 1
            self.npc_layout_reset_count += 1
            self.npc_layout_last_reset_reason = refresh_reason or "episode_refresh"
            self.npc_layout_age_agent_resets = 0
            self._cache_npc_layout(npc_records, npc_target_poses)
            self._apply_active_npc_runtime_mode(
                active_npcs,
                npc_mode,
                npc_speed_min,
                npc_speed_max,
                freeze_for_spawn=False,
            )
            self._consecutive_layout_failures = 0
            self._last_spawn_debug = spawn_debug
            return {"ok": True, "obs": final_obs, "info": final_info, "debug": spawn_debug}

        # notefailed: note(note)
        spawn_debug["spawn_fail_reason"] = spawn_debug.get("spawn_fail_reason") or "max_attempts_exhausted"
        attempt_fail_counts[spawn_debug["spawn_fail_reason"]] += 1
        spawn_debug["attempt_fail_counts"] = dict(attempt_fail_counts)
        if last_violations:
            spawn_debug["violations"] = last_violations[: self.spawn_debug_violations_limit]
        self.spawn_failure_reason_stats[spawn_debug["spawn_fail_reason"]] += 1
        self._consecutive_layout_failures += 1
        self._last_spawn_debug = spawn_debug
        return {"ok": False, "obs": final_obs, "info": final_info, "debug": spawn_debug}

    def _fallback_spawn_relaxed(self, active_npcs):
        """v11.4/v11.5 note spawn.

        note _spawn_episode_layout notefailednote.
        v11.5 note: note(note), note teleport,
        note teleport note.
        """
        nodes = self._track_nodes()
        n = len(nodes) if nodes else 0
        if n == 0:
            return {"ok": False, "obs": None, "info": {}, "debug": None}

        MAX_CANDIDATES = 12   # note(notecompute, note)
        MAX_TELEPORTS = 2     # noterowsnote teleport note
        handler = self.env.viewer.handler
        final_obs = None
        final_info = {}

        # Phase 1: note + note(note teleport, note)
        viable_poses = []
        tried = set()
        for _ in range(MAX_CANDIDATES):
            ego_anchor = random.randint(0, n - 1)
            while ego_anchor in tried and len(tried) < n:
                ego_anchor = random.randint(0, n - 1)
            tried.add(ego_anchor)
            # note(center), note jitter note
            exact = len(viable_poses) == 0  # note, note jitter
            ego_pose = self._sample_pose_from_node(ego_anchor, jitter_scale=0.3, exact=exact, lane_side="center")
            if ego_pose is None:
                continue
            if self._precheck_pose_viable(ego_pose):
                viable_poses.append(ego_pose)
                if len(viable_poses) >= MAX_TELEPORTS:
                    break

        # Phase 2: note teleport(note MAX_TELEPORTS note)
        # skip_cte_check=True: NPC note CTE datanote(23/33),
        # note fine_dist notetracknote, note teleport note
        for ego_pose in viable_poses:
            ego_result = self._apply_spawn_and_verify_ego(ego_pose, retries_step=3, skip_cte_check=True)
            final_obs = ego_result.get("obs")
            final_info = ego_result.get("info", {})
            if ego_result.get("ok"):
                self.last_active_node = int(ego_pose["node_idx"])
                if len(active_npcs) > 0:
                    self._fallback_place_npcs(active_npcs, int(ego_pose["node_idx"]),
                                              int(ego_pose.get("fine_idx", 0)))
                print(f"  PASS v11.5notesucceeded ego_idx={ego_pose['node_idx']} cte={ego_result.get('cte_after', '?'):.2f} "
                      f"note={len(viable_poses)} teleport={viable_poses.index(ego_pose)+1}")
                return {"ok": True, "obs": final_obs, "info": final_info, "debug": None}
            else:
                print(f"  FAIL v11.5noteteleportfailed ego_idx={ego_pose['node_idx']} "
                      f"cte={ego_result.get('cte_after', '?'):.2f} dist_err={ego_result.get('dist_err', '?'):.3f} "
                      f"fine_dist={ego_pose.get('fine_dist_sim', '?')}")

        # notefailed: teleport note node 0(note), note CTE note
        if nodes:
            node0 = nodes[0]
            # v11.5 fix: note teleport note node0, noterows
            for _retry_n0 in range(3):
                SimExtendedAPI.send_set_position(handler,
                    node0[0], node0[1], node0[2], node0[3], node0[4], node0[5], node0[6])
                time.sleep(0.15)
                final_obs, final_info = self._idle_step_for_telemetry(1)
                lx, _, lz = self._extract_pos(final_info)
                S = SimExtendedAPI.COORD_SCALE
                tx, tz = node0[0] / S, node0[2] / S
                if math.sqrt((lx - tx) ** 2 + (lz - tz) ** 2) <= 1.5:
                    break
            try:
                handler.over = False
            except Exception:
                pass
            self.last_active_node = 0
            if len(active_npcs) > 0:
                self._fallback_place_npcs(active_npcs, 0, 0)
            print(f"  ⚠️ v11.5note: teleportnotenode0 (note={len(viable_poses)})")
        return {"ok": True, "obs": final_obs, "info": final_info, "debug": None}

    def _fallback_spawn_relaxed_agent_only(self, active_npcs):
        """agent-only note: note learner, note NPC note."""
        ok_cached, npc_records, reason = self._capture_runtime_npc_records(active_npcs)
        if not ok_cached:
            return {"ok": False, "obs": None, "info": {}, "debug": {"spawn_fail_reason": reason}}

        nodes = self._track_nodes()
        n = len(nodes) if nodes else 0
        if n == 0:
            return {"ok": False, "obs": None, "info": {}, "debug": {"spawn_fail_reason": "no_track_nodes"}}

        constraints = self._spawn_constraints()
        fixed2_mode = str(getattr(self, "v11_2_mode", "")).lower() == "fixed2"
        manual_spawn_loaded = bool(getattr(getattr(self, "manual_spawn", None), "loaded", False))
        safe_ego_anchors = []
        if fixed2_mode and manual_spawn_loaded and npc_records:
            safe_ego_anchors = self._fixed2_collect_safe_ego_anchors(
                [{"tel": r.get("tel"), "fine_idx": r.get("npc_fine_idx", 0)} for r in npc_records],
                constraints,
            )

        max_candidates = max(12, int(self.curriculum_stage_ref.get("spawn_max_attempts", 8)) * 2)
        max_teleports = 3
        candidate_poses = []
        tried = set()
        last_violations = []

        for attempt in range(max_candidates):
            if safe_ego_anchors:
                ego_anchor = random.choice(safe_ego_anchors)
            else:
                if len(tried) >= n:
                    break
                ego_anchor = random.randint(0, n - 1)
                while ego_anchor in tried and len(tried) < n:
                    ego_anchor = random.randint(0, n - 1)
                tried.add(ego_anchor)

            exact = attempt < 2
            jitter_scale = 0.6 if attempt < (max_candidates // 2) else 1.0
            ego_pose = self._sample_pose_from_node(ego_anchor, jitter_scale=jitter_scale, exact=exact, lane_side="center")
            if ego_pose is None:
                continue
            if not self._precheck_pose_viable(ego_pose):
                continue

            valid_sep, last_violations = self.validate_spawn_separation(
                ego_pose,
                [{"tel": r["tel"], "fine_idx": r["npc_fine_idx"]} for r in npc_records],
                constraints=constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc(),
            )
            if not valid_sep:
                continue
            candidate_poses.append(ego_pose)
            if len(candidate_poses) >= max_teleports:
                break

        final_obs = None
        final_info = {}
        for ego_pose in candidate_poses:
            ego_result = self._apply_spawn_and_verify_ego(
                ego_pose,
                retries_step=max(3, self._spawn_ego_verify_retries()),
                skip_cte_check=True,
            )
            final_obs = ego_result.get("obs")
            final_info = ego_result.get("info", {})
            if not ego_result.get("ok"):
                continue

            ex, ey, ez = self._extract_pos(final_info)
            ef = ego_pose.get("fine_idx", 0)
            if self.track_cache:
                ef, _ = self.track_cache.find_nearest_fine_track(self.scene_name, ex, ez)
            valid_sep2, last_violations = self.validate_spawn_separation(
                {"tel": (ex, ey, ez), "fine_idx": ef},
                [{"tel": r["tel"], "fine_idx": r["npc_fine_idx"]} for r in npc_records],
                constraints=constraints,
                fail_on_npc_npc=self._spawn_fail_on_npc_npc(),
            )
            if not valid_sep2:
                continue

            self.last_active_node = int(ego_pose.get("node_idx", 0))
            return {
                "ok": True,
                "obs": final_obs,
                "info": final_info,
                "debug": {
                    "reset_idx": self._reset_index,
                    "stage": int(self.curriculum_stage_ref.get("stage", 1)),
                    "spawn_validation_pass": True,
                    "spawn_fail_reason": None,
                    "layout_reused": True,
                    "npc_layout_id": int(self.npc_layout_id),
                    "npc_layout_age_agent_resets": int(self.npc_layout_age_agent_resets),
                    "refresh_reason": "fallback_agent_only",
                    "safe_anchor_count": int(len(safe_ego_anchors)),
                },
            }

        fail_reason = "fallback_agent_only_no_candidate"
        if candidate_poses:
            fail_reason = "fallback_agent_only_verify_failed"
        dbg = {
            "reset_idx": self._reset_index,
            "stage": int(self.curriculum_stage_ref.get("stage", 1)),
            "spawn_validation_pass": False,
            "spawn_fail_reason": fail_reason,
            "layout_reused": True,
            "npc_layout_id": int(self.npc_layout_id),
            "npc_layout_age_agent_resets": int(self.npc_layout_age_agent_resets),
            "refresh_reason": "fallback_agent_only",
        }
        if last_violations:
            dbg["violations"] = last_violations[: self.spawn_debug_violations_limit]
        return {"ok": False, "obs": final_obs, "info": final_info, "debug": dbg}

    def _fallback_place_npcs(self, active_npcs, ego_node_idx, ego_fine_idx):
        """v11.4: note spawn note NPC note ego note."""
        nodes = self._track_nodes()
        n = len(nodes) if nodes else 0
        if n == 0:
            return

        npc_mode = self._normalize_npc_mode(self.curriculum_stage_ref.get("npc_mode", "static"))
        placed = []
        for i, npc in enumerate(active_npcs):
            if not npc.connected:
                continue
            # note ego note + note
            offset = n // 2 + random.randint(-n // 6, n // 6)
            npc_node = (ego_node_idx + offset + i * (n // (len(active_npcs) + 1))) % n
            # note NPC note
            for _ in range(5):
                too_close = any(abs(npc_node - p) < 3 or abs(npc_node - p) > n - 3 for p in placed)
                if not too_close:
                    break
                npc_node = (npc_node + random.randint(3, 8)) % n

            npc_pose = self._sample_pose_from_node(npc_node, jitter_scale=0.3, exact=True, lane_side="center")
            if npc_pose is None:
                # note node note
                node = nodes[npc_node]
                npc.set_position_node_coords(node[0], node[1], node[2], node[3], node[4], node[5], node[6])
            else:
                px, py, pz, qx, qy, qz, qw = npc_pose["node_coords"]
                npc.set_position_node_coords(px, py, pz, qx, qy, qz, qw)

            placed.append(npc_node)
            time.sleep(0.12)

            # note NPC note
            if npc_mode == "static":
                npc.set_mode("static", 0.0)
                try:
                    npc.handler.send_control(0, 0, 1.0)
                    time.sleep(0.05)
                    npc.handler.send_control(0, 0, 1.0)
                except Exception:
                    pass
            elif npc_mode == "wobble":
                npc.set_mode("wobble", 0.0)
                if not npc.running:
                    npc.start_driving()
            elif npc_mode == "slow":
                thr = float(self.curriculum_stage_ref.get("npc_speed_max", 0.12))
                thr = max(0.10, thr if thr > 0.0 else 0.12)
                npc.set_mode("slow", thr)
                if not npc.running:
                    npc.start_driving()
            elif npc_mode in ("random", "chaos"):
                thr = float(self.curriculum_stage_ref.get("npc_speed_max", 0.20))
                thr = max(0.12, thr if thr > 0.0 else 0.20)
                npc.set_mode(npc_mode, thr)
                if not npc.running:
                    npc.start_driving()

            # note fine_track_idx
            if self.track_cache:
                nx, ny, nz = npc.get_telemetry_position()
                npc.fine_track_idx, _ = self.track_cache.find_nearest_fine_track(self.scene_name, nx, nz)

    def _print_spawn_debug(self, dbg):
        if not self.spawn_debug or not dbg:
            return
        fail_counts = dbg.get("attempt_fail_counts") or {}
        flash_trig = "-"
        if isinstance(fail_counts, dict) and fail_counts:
            top_items = sorted(fail_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
            flash_trig = ",".join([f"{k}:{v}" for k, v in top_items[:3]])
        stage_name = STAGES.get(int(dbg.get("stage", 1)), StageConfig(0, "?", "drive_only")).name
        print(
            "🧪 ResetSpawn |"
            " stage=%s" % stage_name,
            " reset=%s" % dbg.get("reset_idx"),
            " ok=%s" % dbg.get("spawn_validation_pass"),
            " retries=%s" % dbg.get("spawn_retries"),
            " ego_idx=%s" % dbg.get("ego_spawn_idx"),
            " ego_fi=%s" % dbg.get("ego_fine_idx"),
            " min_d=%s" % ("%.2f" % dbg["ego_npc_min_dist"] if dbg.get("ego_npc_min_dist") is not None else "-"),
            " min_gap=%s" % (dbg.get("ego_npc_progress_gap") if dbg.get("ego_npc_progress_gap") is not None else "-"),
            " cte=%.2f" % _safe_float(dbg.get("post_spawn_cte", 0.0), 0.0),
            " stabilize=%s" % dbg.get("stabilize_steps"),
            " safe_anchors=%s" % dbg.get("safe_anchor_count", "-"),
            " layout=%s" % ("reuse" if dbg.get("layout_reused") else "new"),
            " layout_id=%s" % dbg.get("npc_layout_id"),
            " bad_nodes=%s" % dbg.get("bad_nodes_count"),
            " flash_trig=%s" % flash_trig,
            " fail=%s" % dbg.get("spawn_fail_reason"),
        )
        if self.npc_layout_debug:
            if dbg.get("npc_lanes") or dbg.get("npc_segments"):
                print(f"   🧩 lanes={dbg.get('npc_lanes')} segs={dbg.get('npc_segments')} refresh_reason={dbg.get('refresh_reason')}")
            if dbg.get("violations"):
                print(f"   ❗violations={dbg.get('violations')[: self.spawn_debug_violations_limit]}")

    def _park_unused_npcs(self):
        for npc in self._inactive_npcs():
            self._hide_npc_offtrack(npc)

    # ---------------- reset / step overrides ----------------
    def reset(self, **kwargs):
        self._reset_index += 1

        # note base note(note, note reset note spawn)
        if self.episode_stats.get("steps", 0) > 0:
            avg_cte = self.episode_stats['cte_sum'] / max(1, self.episode_stats['steps'])
            term_reason = self.episode_stats.get('termination_reason', 'max_steps')
            self._last_episode_collision = bool(self.episode_stats.get('collision', False))
            self._last_episode_success_2laps = bool(
                self.episode_stats.get('termination_reason') in ('two_laps_success', 'success_laps_target')
                or self.episode_stats.get('success_2laps', False)
            )
            # v11.3: note
            # note step() note, note
            episode_laps = int(self.episode_stats.get('full_lap_bonus_count', 0))
            remaining_laps = episode_laps - self._mid_episode_laps_already_counted
            if remaining_laps > 0:
                self._total_laps_completed += remaining_laps
            self.recent_episodes.append({
                'avg_cte': avg_cte,
                'collision': self.episode_stats['collision'],
                'steps': self.episode_stats['steps'],
                'reward': self.episode_stats['total_reward'],
            })
            prev_stage_for_cte = int(self.curriculum_stage_ref.get('stage', 1))
            if not (prev_stage_for_cte <= 1 and bool(self.disable_cte_auto_adjust)):
                self._evaluate_and_adjust_cte()
            self.total_episodes += 1
            try:
                self._on_episode_end(avg_cte, term_reason)
            except Exception as e:
                print(f"⚠️ episode end hook failed: {e}")
            stage = int(self.curriculum_stage_ref.get('stage', 1))
            stage_name = STAGES.get(stage, STAGES[1]).name
            laps_est = float(self.episode_stats.get('progress_laps_est', 0.0))
            progress_goal = float(self.episode_stats.get('progress_goal_ratio', 0.0))
            pbar = self._format_progress_bar(progress_goal, width=14)
            rev_blk = int(self.episode_stats.get('reverse_gate_blocks', 0))
            rev_esc = int(self.episode_stats.get('reverse_escape_windows', 0))
            print(f"[{stage_name}] note{self.total_train_steps:,} | note{self.total_episodes:,}: "
                  f"{self.episode_stats['steps']}note R={self.episode_stats['total_reward']:.1f} "
                  f"CTE={avg_cte:.2f} v_max={self.episode_stats['max_speed']:.1f} "
                  f"Prog={laps_est:.2f}note {pbar} {progress_goal*100:.0f}% "
                  f"RevBlk={rev_blk} RevEsc={rev_esc} "
                  f"note={term_reason}")

        obs = self.env.reset(**kwargs)
        time.sleep(0.15)
        self._clear_handler_over()

        # v10.2: stagenote NPC, note Stage A note
        self._ensure_npc_connections_for_stage()

        stage = int(self.curriculum_stage_ref.get("stage", 1))
        # note: note, notestagenote
        stage_random_start_override = self.curriculum_stage_ref.get("stage_random_start_enabled", None)
        if stage_random_start_override is None:
            stage_random_start = bool(self.use_random_start and stage >= self.random_start_from_stage)
        else:
            stage_random_start = bool(self.use_random_start and bool(stage_random_start_override))
        if stage <= 1:
            # Stage1 noteCTEnote, noteclassdynamicnote
            self.current_max_cte = max(float(getattr(self, "current_max_cte", 0.0)), self.stage1_cte_reset_limit)
        stage_cte_limit = self.curriculum_stage_ref.get("stage_cte_reset_limit", None)
        if stage_cte_limit is not None:
            try:
                self.current_max_cte = float(stage_cte_limit)
            except Exception:
                pass

        # noteNPCnotetrack
        self._park_unused_npcs()

        spawn_result = {"ok": True, "obs": None, "info": {}, "debug": None}
        active_npcs = self._active_npcs()
        # note: note"note", noteNPCnotestagenoterowstracknotespawn,
        # noteNPCnote(note)notelearnernote.
        need_track_spawn_layout = bool(self.track_cache and self._track_nodes() and (stage_random_start or len(active_npcs) > 0))
        if need_track_spawn_layout:
            refresh_layout, refresh_reason = self.should_refresh_npc_layout(active_npcs)
            spawn_result = self._spawn_episode_layout(refresh_npc_layout=refresh_layout, refresh_reason=refresh_reason)
            npc_state_invalid = False
            attempted_agent_only_fallback = False
            if (not spawn_result.get("ok")) and (not refresh_layout) and len(active_npcs) > 0:
                # persistnote learner-only note, note learner reset note NPC note.
                _dbg = spawn_result.get("debug") or {}
                _fail = str(_dbg.get("spawn_fail_reason", "") or "")
                npc_state_invalid = _fail in (
                    "runtime_npc_offtrack",
                    "runtime_npc_disconnected",
                    "cached_count_mismatch",
                    "cached_pose_missing",
                    "cached_npc_verify_failed",
                )
                if npc_state_invalid:
                    force_reason = "force_refresh_after_agent_spawn_fail"
                    self.npc_layout_last_reset_reason = force_reason
                    spawn_result = self._spawn_episode_layout(refresh_npc_layout=True, refresh_reason=force_reason)
                elif self._npc_persist_mode_enabled():
                    attempted_agent_only_fallback = True
                    print("⚠️ agent-only spawnfailed, note learner-only note(noteNPCnote)")
                    spawn_result = self._fallback_spawn_relaxed_agent_only(active_npcs)
            if not spawn_result.get("ok"):
                # v11.4 note: noteclassnote _randomize_start_position(note node note,
                # CTE note >6 note, note1note).
                # note: note fixed2 note fine-point spawn, note.
                _dbg = spawn_result.get("debug") or {}
                _fail = _dbg.get("spawn_fail_reason", "?")
                _fc = _dbg.get("attempt_fail_counts", {})
                if self._npc_persist_mode_enabled() and len(active_npcs) > 0 and (not npc_state_invalid) and (not attempted_agent_only_fallback):
                    attempted_agent_only_fallback = True
                    print(f"⚠️ v10 spawnnotefailed, note learner-only note(noteNPCnote)fail={_fail} counts={_fc}")
                    spawn_result = self._fallback_spawn_relaxed_agent_only(active_npcs)
                if (not spawn_result.get("ok")) and self._npc_persist_mode_enabled() and len(active_npcs) > 0 and (not npc_state_invalid):
                    # note: note NPC, note env.reset notedefault learner note, note NPC note.
                    print(f"⚠️ learnernotefailed, noteNPCnote; notedefaultnote fail={_fail} counts={_fc}")
                    dbg_keep = dict(_dbg)
                    dbg_keep.setdefault("reset_idx", self._reset_index)
                    dbg_keep.setdefault("stage", int(self.curriculum_stage_ref.get("stage", 1)))
                    dbg_keep.setdefault("layout_reused", True)
                    dbg_keep.setdefault("npc_layout_id", int(self.npc_layout_id))
                    dbg_keep.setdefault("npc_layout_age_agent_resets", int(self.npc_layout_age_agent_resets))
                    dbg_keep.setdefault("spawn_fail_reason", str(_fail))
                    dbg_keep.setdefault("spawn_validation_pass", False)
                    spawn_result = {"ok": True, "obs": obs, "info": {}, "debug": dbg_keep}
                elif not spawn_result.get("ok"):
                    print(f"⚠️ v10 spawnnotefailed, note v11.5 note(note+noteteleport)fail={_fail} counts={_fc}")
                    spawn_result = self._fallback_spawn_relaxed(active_npcs)
            if spawn_result.get("ok"):
                # notesucceededspawnnote
                if len(active_npcs) > 0:
                    if spawn_result.get("debug", {}).get("layout_reused"):
                        self.npc_layout_age_agent_resets += 1
                    else:
                        # NPC note: note, note
                        self.npc_layout_age_agent_resets = 0
                        self._npc_layout_last_reset_at_laps = self._total_laps_completed
                self._consecutive_layout_failures = 0
        else:
            if self.curriculum_stage_ref.get("npc_count", 0) <= 0:
                self._park_npcs_offtrack_lazy()

        self._print_spawn_debug(spawn_result.get("debug"))
        if spawn_result.get("debug"):
            self.reset_debug_history.append(spawn_result["debug"])

        # note
        final_obs = spawn_result.get("obs")
        final_info = spawn_result.get("info", {})
        if final_obs is None:
            final_obs, final_info = self._idle_step_for_telemetry(2)
        else:
            # note, noteteleportnote
            final_obs, final_info = self._idle_step_for_telemetry(1)

        self._log_obs_caps_once(final_info if isinstance(final_info, dict) else {})

        # reset state (notev9note + v10note)
        self.episode_step = 0
        self.nodes_passed = 0
        self.speed_history.clear()
        self.stuck_counter = 0
        self.offtrack_counter = 0
        self._collision_persist_counter = 0
        self.learner_fine_idx = 0
        self._episode_prev_fine_idx = None
        self._episode_progress_fine_signed = 0.0
        self._episode_progress_fine_forward = 0.0
        self._episode_progress_fine_backward = 0.0
        self._episode_motion_armed = False
        self._motion_speed_streak = 0
        self._episode_spawn_x = None
        self._episode_spawn_z = None
        self._last_progress_milestone = 0
        self._last_full_lap_bonus = 0
        self._mid_episode_laps_already_counted = 0   # note
        self._bad_nodes = set()  # v11.4: note, noteclassnote
        self._reverse_escape_steps_left = 0
        self._reverse_escape_cooldown_left = 0
        self._low_progress_counter = 0
        self._npc_overtake_state = {}
        self._last_npc_dists = {}
        self._last_step_npc_metrics = {}
        self.reverse_progress_counter = 0
        self.reverse_progress_accum = 0.0
        self._reverse_reset_streak = 0
        self._negative_throttle_streak = 0
        self._prev_ego_fine_idx_for_reverse = None
        self._npc_npc_guard_pair_cooldown = {}
        self._npc_npc_guard_events_episode = 0
        self._npc_npc_contact_reset_cooldown_left = 0
        self._npc_npc_contact_events_episode = 0

        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0
        self.prev_rgb = None
        self._ep_prev_ctrl_steer_exec = None
        self._ep_prev_pilot_kappa_ref = None

        self.episode_stats = {
            'cte_sum': 0, 'steps': 0, 'max_speed': 0,
            'speed_sum': 0.0,
            'collision': False, 'rear_end': False,
            'total_reward': 0, 'termination_reason': 'max_steps',
            'unsafe_follow_steps': 0,
            'success_2laps': False,
            'reverse_penalty_events': 0,
            'reverse_reset_terminations': 0,
            'reverse_gate_blocks': 0,
            'reverse_escape_windows': 0,
            'progress_laps_est': 0.0,
            'progress_laps_max': 0.0,
            'progress_goal_ratio': 0.0,
            'full_lap_bonus_count': 0,
            'lap_count_max': 0,
            'ctrl_steer_sat_steps': 0,
            'ctrl_dt_clipped_steps': 0,
            'ctrl_dsteer_abs_samples': [],
            'ctrl_dkappa_abs_samples': [],
            'stall_penalty_sum': 0.0,
            'stall_penalty_steps': 0,
            'npc_npc_guard_events': 0,
            'npc_npc_contact_resets': 0,
            'npc_stuck_resets': 0,
            'spawn_source': '',
            'npc_layout_hash': '',
            'npc_layout_refresh_reason': '',
        }

        if final_obs is None:
            final_obs = obs
        if isinstance(final_info, dict):
            sx, sy, sz = self._extract_pos(final_info)
            self._episode_spawn_x = float(sx)
            self._episode_spawn_z = float(sz)
        return self._process_observation(final_obs)

    def _extract_active_node(self, info):
        active_node = info.get('activeNode', self.last_active_node)
        if isinstance(active_node, str):
            try:
                active_node = int(active_node)
            except Exception:
                active_node = self.last_active_node
        try:
            return int(active_node)
        except Exception:
            return int(self.last_active_node)

    def _base_drive_reward(self, info, safe_action, actual_delta, prev_delta_steer, rate_excess_bounded):
        cte = abs(_safe_float(info.get('cte', 0.0), 0.0))
        speed = _safe_float(info.get('speed', 0.0), 0.0)
        hit = info.get('hit', 'none')
        lap_count = int(_safe_float(info.get('lap_count', 0), 0)) if str(info.get('lap_count', 0)).isdigit() else int(_safe_float(info.get('lap_count', 0), 0))
        dfi, progress_step_sim, fine_total = self._update_episode_progress(info, speed=speed)
        info['episode_progress_step_sim'] = float(progress_step_sim)
        self.speed_history.append(speed)
        self.episode_stats['cte_sum'] += cte
        self.episode_stats['max_speed'] = max(self.episode_stats['max_speed'], speed)
        progress_laps_est = float(info.get('episode_progress_laps_est', 0.0))
        info['progress_lapcount_gap'] = float(progress_laps_est - float(lap_count))
        self.episode_stats['progress_laps_est'] = progress_laps_est
        self.episode_stats['progress_goal_ratio'] = float(info.get('episode_progress_ratio_to_goal', 0.0))
        reward_decay_scale, penalty_decay_scale = self._shaping_decay_factor()

        # tracknote(CTEnote)
        _half_w = self._half_width_avg_cte_at_fine(self.learner_fine_idx)   # note
        _wide_w = self._half_width_wide_cte_at_fine(self.learner_fine_idx)  # note, CTEnote
        ontrack = float(cte <= _wide_w)  # noteontracknote(note), CTEnote
        reward = 0.0
        done = False
        rt = self._rt_add

        # ═══════════════════════════════════════════════════
        # [R1] notereward: note +0.10
        # ═══════════════════════════════════════════════════
        reward += 0.10
        rt(info, "base_alive", 0.10)

        # ═══════════════════════════════════════════════════
        # [R2] notereward + note
        #      v_norm note max_speed note, reward +0.8 x v_norm
        #      speed > safe_hi -> note(note)
        #      speed < safe_lo -> note(note)
        #      safe_lo <= speed <= safe_hi -> note, note
        # ═══════════════════════════════════════════════════
        _speed_max = 2.0
        _speed_safe_hi = 1.5
        _speed_safe_lo = 0.5
        v_norm = np.clip(speed, 0.0, _speed_max) / _speed_max
        speed_reward = 0.8 * v_norm
        reward += speed_reward
        rt(info, "base_speed", speed_reward)

        # note: speed > 1.5 note, note speed=2.0 note -0.5/step
        if speed > _speed_safe_hi:
            _over_ratio = min(1.0, (speed - _speed_safe_hi) / max(1e-6, _speed_max - _speed_safe_hi))
            _over_pen = -0.5 * _over_ratio * penalty_decay_scale
            reward += _over_pen
            rt(info, "speed_overspeed_penalty", _over_pen)
        # note: speed < 0.5 note, note speed=0 note -0.3/step
        elif speed < _speed_safe_lo and self.episode_step > 30:
            _slow_ratio = min(1.0, (_speed_safe_lo - speed) / max(1e-6, _speed_safe_lo))
            _slow_pen = -0.3 * _slow_ratio * penalty_decay_scale
            reward += _slow_pen
            rt(info, "speed_tooslow_penalty", _slow_pen)

        # ═══════════════════════════════════════════════════
        # [R3] tracknotereward(ontrack keeping bonus)
        #      cte < 0.3x_wide_w -> +0.25 (note)
        #      cte < 0.6x_wide_w -> +0.15 (note)
        #      cte < _wide_w     -> +0.05 (note)
        # ═══════════════════════════════════════════════════
        if cte < 0.3 * _wide_w:
            _keeping = 0.25
            reward += _keeping
            rt(info, "ontrack_keeping_core", _keeping)
        elif cte < 0.6 * _wide_w:
            _keeping = 0.15
            reward += _keeping
            rt(info, "ontrack_keeping_comfort", _keeping)
        elif cte < _wide_w:
            _keeping = 0.05
            reward += _keeping
            rt(info, "ontrack_keeping_edge", _keeping)

        # ═══════════════════════════════════════════════════
        # [R4] CTE note + max_cte note done
        #      cte > max_cte -> note + done
        #      cte > _wide_w -> note(note)
        # ═══════════════════════════════════════════════════
        cte_for_done = float(cte)
        if cte_for_done >= self.current_max_cte and self.episode_step <= 8:
            lx, _, lz = self._extract_pos(info)
            geom_ok, _geom_meta = self._geometry_track_check(
                lx, lz,
                seed_fi=int(getattr(self, "learner_fine_idx", 0)),
                margin_scale=0.45,
                centerline_slack=0.22,
            )
            if geom_ok:
                info['spawn_cte_waived'] = True
                info['spawn_cte_waived_raw'] = float(cte_for_done)
                cte_for_done = float(min(cte_for_done, self.spawn_verify_cte_threshold))

        if cte_for_done >= self.current_max_cte:
            exceed_ratio = (cte_for_done - self.current_max_cte) / max(1e-6, self.current_max_cte)
            cte_term = -penalty_decay_scale * (0.8 + 3.5 * exceed_ratio)
            reward += cte_term
            rt(info, "cte_exceed_penalty", cte_term)
            done = True
            info['termination_reason'] = 'cte_exceed'
            self.episode_stats['termination_reason'] = 'cte_exceed'
            rt(info, "event_cte_exceed_done", 0.0)
        elif cte_for_done > _wide_w:
            edge_ratio = (cte_for_done - _wide_w) / max(1e-6, self.current_max_cte - _wide_w)
            cte_term = -penalty_decay_scale * 0.3 * edge_ratio
            reward += cte_term
            rt(info, "cte_edge_penalty", cte_term)

        # note: notestagenote, notespawnnote/note"note"
        collision_guard_ok = bool(self._episode_motion_armed) or (self.episode_step > max(120, self.startup_grace_steps))
        if hit!= 'none' and self.episode_step > 20 and collision_guard_ok:
            self.episode_stats['collision'] = True
            # unifiednote: noteNPC, note + notedone
            reward -= 5.0
            rt(info, "event_collision", -5.0)
            done = True
            info['termination_reason'] = 'collision'
            self.episode_stats['termination_reason'] = 'collision'
        else:
            # note, note
            self._collision_persist_counter = 0

        # [R5] notereward: ontracknote, offtracknote50%
        if progress_step_sim > 0:
            _prog_scale = 1.0 if ontrack else 0.5
            progress_boost = 1.0
            if self.early_progress_boost_lap > 1e-6 and progress_laps_est < self.early_progress_boost_lap:
                remain = 1.0 - (progress_laps_est / self.early_progress_boost_lap)
                progress_boost += (max(1.0, self.early_progress_boost_factor) - 1.0) * float(np.clip(remain, 0.0, 1.0))
                info['early_progress_boost'] = float(progress_boost)
            prog_term = _prog_scale * (self.progress_reward_scale * reward_decay_scale * progress_boost) * progress_step_sim
            reward += prog_term
            rt(info, "progress_forward", prog_term)
        elif progress_step_sim < 0:
            prog_term = -(self.progress_backward_penalty_scale * penalty_decay_scale) * abs(progress_step_sim)
            reward += prog_term
            rt(info, "progress_backward_penalty", prog_term)
        # notereward: noterewardnote(note)
        # B1 fix: noterewardnotefirstnote(_episode_progress_fine_forward),
        # note laps_net(V11.1 signednote, notereward).
        milestone_progress = float(self._episode_progress_fine_forward) / float(max(1, fine_total))
        milestone_lap = max(0.02, self.progress_milestone_lap)
        cur_milestone = int(milestone_progress / milestone_lap)
        if cur_milestone > self._last_progress_milestone:
            gained = cur_milestone - self._last_progress_milestone
            milestone_term = float(gained) * (self.progress_milestone_reward * reward_decay_scale)
            reward += milestone_term
            rt(info, "progress_milestone_bonus", milestone_term)
            self._last_progress_milestone = cur_milestone
            info['progress_milestone'] = cur_milestone
            info['progress_reward_decay_scale'] = float(reward_decay_scale)

        # notereward: notereward(notelap_count)
        full_laps = int(max(0.0, milestone_progress))
        if full_laps > self._last_full_lap_bonus:
            gained_laps = full_laps - self._last_full_lap_bonus
            lap_bonus = float(gained_laps) * (self.progress_full_lap_reward * reward_decay_scale)
            reward += lap_bonus
            rt(info, "progress_full_lap_bonus", lap_bonus)
            self._last_full_lap_bonus = full_laps
            self.episode_stats['full_lap_bonus_count'] = int(self._last_full_lap_bonus)
            info['full_lap_bonus_laps'] = int(gained_laps)
            info['full_lap_bonus_reward'] = float(lap_bonus)

            # v11.3: noterowsnote, note NPC note
            reset_every_laps = int(self.curriculum_stage_ref.get("npc_layout_reset_every_laps", 0))
            if reset_every_laps > 0 and gained_laps >= reset_every_laps:
                self._total_laps_completed += int(gained_laps)
                self._mid_episode_laps_already_counted += int(gained_laps)
                if not self._npc_persist_mode_enabled():
                    self._refresh_npc_layout_mid_episode()

        # reverse gate note"note"note
        if self.episode_step > 20 and abs(progress_step_sim) <= self.reverse_escape_low_progress_step_sim and abs(speed) < max(0.35, self.reverse_gate_brake_speed):
            self._low_progress_counter += 1
        elif progress_step_sim > self.reverse_escape_low_progress_step_sim * 1.5:
            self._low_progress_counter = 0
        else:
            self._low_progress_counter = max(0, self._low_progress_counter - 1)
        info['low_progress_counter'] = int(self._low_progress_counter)

        # stuck
        stuck_guard_ok = bool(self._episode_motion_armed) and (progress_laps_est >= self.stuck_guard_progress_laps)
        if self.episode_step > 20 and stuck_guard_ok and ontrack and speed < 0.1:
            self.stuck_counter += 1
            # note, notemodelnote
            _stuck_per_step = -0.5
            reward += _stuck_per_step
            rt(info, "stuck_per_step_penalty", _stuck_per_step)
        else:
            self.stuck_counter = 0
        if self.stuck_counter > self.stuck_counter_limit:
            done = True
            rt(info, "event_stuck_done", 0.0)
            info['termination_reason'] = 'stuck'
            self.episode_stats['termination_reason'] = 'stuck'

        # [P4] note
        abs_delta = abs(actual_delta)
        abs_jerk = abs(actual_delta - prev_delta_steer)
        delta_term = -(self.w_d * penalty_decay_scale) * abs_delta
        jerk_term = -(self.w_dd * penalty_decay_scale) * abs_jerk
        sat_term = -(self.w_sat * penalty_decay_scale) * rate_excess_bounded
        reward += delta_term
        reward += jerk_term
        reward += sat_term
        rt(info, "smooth_delta_penalty", delta_term)
        rt(info, "smooth_jerk_penalty", jerk_term)
        rt(info, "smooth_sat_penalty", sat_term)

        # activeNode note(rewardnote fine_track progress note)
        active_node = self._extract_active_node(info)
        if active_node!= self.last_active_node:
            self.nodes_passed += 1
            self.last_active_node = active_node

        # notereward(noterelative progress + lap_countnote), note teleport note
        lap1_valid = (lap_count >= 1) and (progress_laps_est >= 0.60)
        lap2_valid = (lap_count >= 2) and (progress_laps_est >= 1.70)
        success_target_valid = (
            lap_count >= int(self.success_laps_target)
            and progress_laps_est >= max(1.70, float(self.success_laps_target) - 0.30)
        )
        if lap1_valid and not info.get('task_success', False):
            # notelap_countnote, notereward
            info['laps_completed'] = max(int(_safe_float(info.get('lap_count', 0), 0)), 1)
        if lap2_valid:
            info['two_lap_progress_success'] = True
            info['laps_completed'] = max(int(_safe_float(info.get('lap_count', 0), 0)), 2)
            # v10.12: trainingnote2note, note"note"noteepisode_stats,
            # noteresetnotehybridnoteNPCnote.
            if not bool(self.episode_stats.get('collision', False)):
                self.episode_stats['success_2laps'] = True
                info['episode_success_2laps'] = True
        if success_target_valid:
            if not info.get('task_success', False):
                info['task_success'] = True
                info['laps_completed'] = int(_safe_float(info.get('lap_count', 0), 0))
            if bool(self.curriculum_stage_ref.get("terminate_on_success_laps", True)):
                info['termination_reason'] = 'success_laps_target'

        if isinstance(self.episode_stats, dict):
            self.episode_stats['speed_sum'] = float(self.episode_stats.get('speed_sum', 0.0)) + float(speed)
            self.episode_stats['lap_count_max'] = max(
                int(self.episode_stats.get('lap_count_max', 0)),
                int(max(0, lap_count)),
            )
            self.episode_stats['progress_laps_max'] = max(
                float(self.episode_stats.get('progress_laps_max', 0.0)),
                float(max(0.0, progress_laps_est)),
            )

        return reward, done

    def _npc_distance_features(self, info):
        if not self.npc_controllers:
            self._last_step_npc_metrics = {"count": 0, "min_dist": None, "encounter": False}
            return []
        lx, ly, lz = self._extract_pos(info)
        learner_speed = _safe_float(info.get('speed', 0.0), 0.0)
        features = []
        if self.track_cache:
            learner_fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, lx, lz)
        else:
            learner_fi = 0

        for npc in self._active_npcs():
            nx, ny, nz = npc.get_telemetry_position()
            dist = math.sqrt((lx - nx) ** 2 + (lz - nz) ** 2)
            prev_dist = self._last_npc_dists.get(npc.npc_id, dist)
            closing_speed = max(0.0, prev_dist - dist) / 0.05  # note(notestepnote50msnote, noterewardnote)
            self._last_npc_dists[npc.npc_id] = dist
            npc_fi = npc.fine_track_idx
            progress_diff = self.track_cache.progress_diff(self.scene_name, learner_fi, npc_fi) if self.track_cache else 0
            features.append({
                'npc': npc,
                'dist': dist,
                'closing_speed': closing_speed,
                'progress_diff': progress_diff,
                'learner_speed': learner_speed,
                'npc_speed': npc.speed,
                'learner_fine_idx': learner_fi,
                'npc_fine_idx': npc_fi,
            })
        if features:
            min_dist = min(f['dist'] for f in features)
            encounter_thresh = float(self.dist_scale.get('follow_safe_max_sim', 4.0)) * 1.35
            self._last_step_npc_metrics = {
                "count": len(features),
                "min_dist": float(min_dist),
                "encounter": bool(min_dist <= encounter_thresh),
            }
        else:
            self._last_step_npc_metrics = {"count": 0, "min_dist": None, "encounter": False}
        return features

    def _apply_reverse_displacement_penalty(self, reward, info):
        """
        v10.2: note(note)
        note fine_track notetrackfirstnote.
        v10.7: note reward-based shaping(defaultnote reverse gate).
        """
        if not self.track_cache:
            return reward
        if self.episode_step <= 5:
            return reward

        lx, ly, lz = self._extract_pos(info)
        cur_fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, lx, lz)

        if self._prev_ego_fine_idx_for_reverse is None:
            self._prev_ego_fine_idx_for_reverse = cur_fi
            return reward

        dfi = self.track_cache.progress_diff(self.scene_name, cur_fi, self._prev_ego_fine_idx_for_reverse)
        self._prev_ego_fine_idx_for_reverse = cur_fi

        # forward: dfi>0, backward: dfi<0
        if dfi < -self.reverse_progress_step_thresh:
            back_dist = abs(dfi) * max(self._fine_gap_sim, 1e-3)
            self.reverse_progress_counter += 1
            self.reverse_progress_accum += back_dist
            self._reverse_reset_streak += 1
            info['reverse_reset_streak'] = int(self._reverse_reset_streak)

            # reward-based anti-reverse: note, note, note
            _, penalty_decay_scale = self._shaping_decay_factor()
            backdist_term = -(self.reverse_backdist_penalty_scale * penalty_decay_scale) * back_dist
            reward += backdist_term
            self._rt_add(info, "reverse_backdist_penalty", backdist_term)
            if self.reverse_progress_counter == 1:
                onset_term = -self.reverse_onset_penalty * penalty_decay_scale
                reward += onset_term
                self._rt_add(info, "reverse_onset_penalty", onset_term)
                info['reverse_onset_penalty'] = True
            else:
                # note, note
                streak_extra = self.reverse_streak_penalty_scale * penalty_decay_scale * min(8, self.reverse_progress_counter - 1)
                reward -= streak_extra
                self._rt_add(info, "reverse_streak_penalty", -streak_extra)
                info['reverse_streak_penalty'] = float(streak_extra)
        else:
            self.reverse_progress_counter = 0
            self.reverse_progress_accum = 0.0
            self._reverse_reset_streak = 0

        if (self.reverse_progress_counter >= self.reverse_penalty_steps and
                self.reverse_progress_accum >= self.reverse_progress_dist_thresh):
            _, penalty_decay_scale = self._shaping_decay_factor()
            event_term = -self.reverse_event_penalty * penalty_decay_scale
            reward += event_term
            self._rt_add(info, "reverse_event_penalty", event_term)
            self.episode_stats['reverse_penalty_events'] = self.episode_stats.get('reverse_penalty_events', 0) + 1
            info['reverse_penalty'] = True
            info['reverse_progress_counter'] = self.reverse_progress_counter
            info['reverse_progress_accum'] = self.reverse_progress_accum
            # note, note
            self.reverse_progress_counter = 0
            self.reverse_progress_accum = 0.0

        if self._reverse_reset_streak >= max(1, int(self.reverse_reset_steps)):
            info['reverse_streak_reset'] = True
            info['reverse_streak_reset_steps'] = int(self._reverse_reset_streak)

        return reward

    def _reward_avoid_static(self, reward, done, info):
        feats = self._npc_distance_features(info)
        if not feats:
            return reward, done

        d_danger = float(self.dist_scale.get('danger_close_sim', 1.2))
        d_safe_min = float(self.dist_scale.get('follow_safe_min_sim', 1.6))
        d_safe_max = float(self.dist_scale.get('follow_safe_max_sim', 4.0))
        d_pass = float(self.dist_scale.get('pass_window_sim', 2.2))
        rt = self._rt_add

        for f in feats:
            dist = f['dist']
            speed = f['learner_speed']
            closing = float(f.get('closing_speed', 0.0))

            # note/noterowsnotereward: note, note
            if d_safe_min <= dist <= d_safe_max and 0.30 < speed < 1.60:
                reward += 0.06
                rt(info, "avoid_static_safe_follow_bonus", 0.06)

            # note(note"note")
            # note, note, note.
            if dist < d_danger:
                danger_ratio = float(np.clip((d_danger - dist) / max(1e-6, d_danger), 0.0, 1.0))
                pen = 0.35 + 0.85 * danger_ratio
                if speed > 0.8:
                    pen += 0.20 * float(np.clip((speed - 0.8) / 0.8, 0.0, 1.0))
                if closing > 0.35:
                    pen += 0.20 * float(np.clip((closing - 0.35) / 1.0, 0.0, 1.0))
                reward -= pen
                rt(info, "avoid_static_danger_penalty", -pen)
                # note, note"note"note
                if dist < 0.65 * d_danger:
                    reward -= 0.25
                    rt(info, "avoid_static_emergency_gap_penalty", -0.25)
            elif dist < d_safe_min:
                # note: note, noterows
                near_ratio = float(np.clip((d_safe_min - dist) / max(1e-6, d_safe_min - d_danger), 0.0, 1.0))
                soft_pen = 0.08 * near_ratio
                reward -= soft_pen
                rt(info, "avoid_static_near_penalty", -soft_pen)

            # notereward: note, note
            if max(d_safe_min, d_danger) < dist < d_pass and 0.20 < speed < 1.80:
                reward += 0.05
                rt(info, "avoid_static_pass_window_bonus", 0.05)
        return reward, done

    def _reward_follow_only(self, reward, done, info):
        feats = self._npc_distance_features(info)
        if not feats:
            return reward, done

        d_danger = float(self.dist_scale.get('danger_close_sim', 1.2))
        d_safe_min = float(self.dist_scale.get('follow_safe_min_sim', 1.6))
        d_safe_max = float(self.dist_scale.get('follow_safe_max_sim', 4.0))
        rt = self._rt_add

        for f in feats:
            dist = f['dist']
            closing = f['closing_speed']
            progress_diff = f['progress_diff']
            # note learner note/note
            if progress_diff < 0:
                if d_safe_min <= dist <= d_safe_max:
                    reward += 0.18
                    rt(info, "follow_safe_band_bonus", 0.18)
                elif dist < d_danger:
                    reward -= 0.9
                    rt(info, "follow_danger_penalty", -0.9)
                    self.episode_stats['unsafe_follow_steps'] += 1
                    # note: note + note
                    if closing > 0.7:
                        reward -= 0.6
                        rt(info, "follow_rear_end_risk_penalty", -0.6)
                        info['rear_end_risk'] = True
                elif dist < d_safe_min:
                    reward -= 0.25
                    rt(info, "follow_too_close_penalty", -0.25)
                # note(note basenote, note)
                if dist < d_safe_max and abs(self.delta_steer_prev) > 0.18:
                    reward -= 0.12
                    rt(info, "follow_aggressive_steer_penalty", -0.12)
            # noteNPCnote
            if abs(progress_diff) < max(8, int(self.dist_scale.get('spawn_min_gap_progress', 30) * 0.25)) and dist < d_safe_min:
                reward -= 0.15
                rt(info, "follow_parallel_close_penalty", -0.15)
            self.follow_distance_history.append(dist)

        return reward, done

    def _reward_overtake_mode(self, reward, done, info):
        # note, notev9noterewardnote
        reward, done = self._reward_follow_only(reward, done, info)

        # note
        feats = self._npc_distance_features(info)
        safe_gate = True
        d_danger = float(self.dist_scale.get('danger_close_sim', 1.2))
        for f in feats:
            if f['dist'] < d_danger:
                safe_gate = False
                break

        if safe_gate:
            reward, done = self._compute_overtake_reward(reward, done, info)
        return reward, done

    def step(self, action):
        self.episode_step += 1
        self.total_train_steps += 1

        # ===== ActionSafety (notev9) =====
        steer_raw = float(action[0])
        throttle_raw = float(action[1])

        delta = steer_raw - self.steer_prev_limited
        rate_excess_raw = max(0.0, abs(delta) - self.delta_max) / max(self.delta_max, 1e-6)
        if abs(delta) > self.delta_max:
            delta = np.clip(delta, -self.delta_max, self.delta_max)
        steer_limited = self.steer_prev_limited + delta

        if self.enable_lpf:
            steer_exec = (1 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        else:
            steer_exec = steer_limited
        steer_exec = np.clip(steer_exec, -1.0, 1.0)

        actual_delta = steer_exec - self.steer_prev_exec
        rate_excess_bounded = float(np.tanh(rate_excess_raw))
        prev_delta_steer = self.delta_steer_prev
        self.delta_steer_prev = actual_delta
        self.steer_prev_limited = steer_limited
        self.steer_prev_exec = steer_exec

        # v10.2/v10.6: throttle action space [-1, 1] + reverse gate(note, note)
        throttle_cmd = float(np.clip(throttle_raw, -1.0, 1.0))
        startup_forced = False
        neg_throttle_request = False
        if self.episode_step <= max(0, self.startup_force_throttle_steps):
            throttle_cmd = float(max(throttle_cmd, abs(self.startup_force_throttle)))
            startup_forced = True
            info_hint = True
        else:
            info_hint = False
        if (not startup_forced) and throttle_cmd < 0.0:
            neg_throttle_request = True
        gated_throttle_cmd, reverse_gate_meta = self._apply_reverse_gate(throttle_cmd)
        throttle_exec = float(np.clip(gated_throttle_cmd, -self.max_throttle, self.max_throttle))
        safe_action = np.array([steer_exec, throttle_exec], dtype=np.float32)

        obs, _base_reward, gym_done, info = self.env.step(safe_action)
        done = False
        if gym_done:
            self._clear_handler_over()

        processed_obs = self._process_observation(obs)
        if not isinstance(info, dict):
            info = {}
        if self.curriculum_stage_ref.get('npc_count', 0) > 1:
            self._apply_npc_npc_collision_guard(info)
            self._maybe_handle_npc_npc_contact_reset(info)
        if self.curriculum_stage_ref.get('npc_count', 0) > 0:
            self._maybe_reset_stuck_npcs(info)
            self._sync_npc_speed_cap_with_learner(info)
        info['reverse_gate_blocked'] = bool(reverse_gate_meta.get("blocked", False))
        info['reverse_escape_active'] = bool(reverse_gate_meta.get("escape_active", False))
        info['reverse_escape_opened'] = bool(reverse_gate_meta.get("escape_opened", False))
        info['reverse_gate_last_speed'] = float(reverse_gate_meta.get("last_speed", 0.0))
        info['reverse_gate_scaled'] = bool(reverse_gate_meta.get("scaled", False))
        info['negative_throttle_streak'] = int(self._negative_throttle_streak)
        if info_hint:
            info['startup_force_throttle'] = float(np.clip(abs(self.startup_force_throttle), 0.0, self.max_throttle))

        reward, done = self._base_drive_reward(info, safe_action, actual_delta, prev_delta_steer, rate_excess_bounded)

        # note reset(note), notefirstnote, note
        speed_now = abs(_safe_float(info.get('speed', 0.0), 0.0))
        progress_step_now = _safe_float(info.get('episode_progress_step_sim', 0.0), 0.0)
        neg_reset_gate = bool(neg_throttle_request and speed_now <= self.negative_throttle_reset_speed_max and progress_step_now <= 0.005)
        if neg_reset_gate:
            self._negative_throttle_streak += 1
        else:
            self._negative_throttle_streak = 0

        if info.get('reverse_gate_blocked', False):
            reward -= self.reverse_gate_block_penalty
            self.episode_stats['reverse_gate_blocks'] = self.episode_stats.get('reverse_gate_blocks', 0) + 1

        reward_mode = self.curriculum_stage_ref.get('reward_mode', 'drive_only')
        if self.curriculum_stage_ref.get('npc_count', 0) > 0:
            if reward_mode == 'avoid_static':
                reward, done = self._reward_avoid_static(reward, done, info)
            elif reward_mode == 'follow_only':
                reward, done = self._reward_follow_only(reward, done, info)
            elif reward_mode == 'overtake':
                reward, done = self._reward_overtake_mode(reward, done, info)
            npc_metrics = getattr(self, "_last_step_npc_metrics", {}) or {}
            info['npc_count_seen'] = int(npc_metrics.get("count", 0))
            info['npc_encounter'] = bool(npc_metrics.get("encounter", False))
            if npc_metrics.get("min_dist") is not None:
                info['npc_min_dist'] = float(npc_metrics.get("min_dist"))

        # v10.2: note(noteNPCnote)
        reward = self._apply_reverse_displacement_penalty(reward, info)
        if self._negative_throttle_streak >= max(1, int(self.reverse_reset_steps)):
            done = True
            reward -= 2.0
            info['negative_throttle_reset'] = True
            info['negative_throttle_reset_steps'] = int(self._negative_throttle_streak)
            info['termination_reason'] = 'negative_throttle_streak_reset'
            self.episode_stats['termination_reason'] = 'negative_throttle_streak_reset'
            self.episode_stats['reverse_reset_terminations'] = self.episode_stats.get('reverse_reset_terminations', 0) + 1

        # note, note(note)
        if info.get('reverse_streak_reset', False):
            info['reverse_streak_reset_armed'] = True

        # note(note+note)
        if self.episode_stats.get('collision'):
            feats = self._npc_distance_features(info)
            for f in feats:
                if f['dist'] <= float(self.dist_scale.get('danger_close_sim', 1.2)) * 1.25 and f['progress_diff'] < 0:
                    info['rear_end'] = True
                    self.episode_stats['rear_end'] = True
                    break

        # rear-end note(note, note)
        if info.get('rear_end_risk'):
            info['rear_end'] = False

        if (
            bool(self.curriculum_stage_ref.get("terminate_on_success_laps", True))
            and info.get('task_success')
            and int(_safe_float(info.get('lap_count', 0), 0)) >= self.success_laps_target
        ):
            done = True
            info['episode_success_2laps'] = True
            self.episode_stats['termination_reason'] = 'success_laps_target'
            self.episode_stats['success_2laps'] = True

        if self.episode_step >= self.max_episode_steps:
            done = True
            if self.episode_stats.get('termination_reason') == 'max_steps':
                self.episode_stats['termination_reason'] = 'max_steps'

        if self.reward_clip_abs and self.reward_clip_abs > 0:
            raw_reward = float(reward)
            reward = float(np.clip(reward, -self.reward_clip_abs, self.reward_clip_abs))
            if reward!= raw_reward:
                info['reward_clipped'] = True
                info['raw_reward_unclipped'] = raw_reward

        self.episode_stats['total_reward'] += reward
        self.episode_stats['steps'] += 1

        return processed_obs, reward, done, info

    # ---------------- debug utilities ----------------
    def reset_stress_test(self, n_resets=50, rollout_steps=20):
        """note debug: note reset note spawn notefirst20note."""
        stress_constraints = self._spawn_constraints()
        near_spawn_thresh = float(stress_constraints["ego_npc"]["min_euclid"])
        stats = {
            'resets': 0,
            'spawn_fail': 0,
            'near_spawn': 0,
            'collision_in_20': 0,
            'offtrack_in_20': 0,
            'min_dist_samples': [],
            'layout_reused_count': 0,
            'layout_new_count': 0,
            'stress_constraints': stress_constraints,
        }
        for i in range(int(n_resets)):
            self.reset()
            dbg = self._last_spawn_debug or {}
            stats['resets'] += 1
            if not dbg.get('spawn_validation_pass', False):
                stats['spawn_fail'] += 1
            md = dbg.get('ego_npc_min_dist')
            if md is not None:
                stats['min_dist_samples'].append(float(md))
                if md < near_spawn_thresh:
                    stats['near_spawn'] += 1
            if dbg.get('layout_reused'):
                stats['layout_reused_count'] += 1
            else:
                stats['layout_new_count'] += 1

            collided = False
            offtrack = False
            for _ in range(int(rollout_steps)):
                _, _, done, info = self.step(np.array([0.0, 0.15], dtype=np.float32))
                cte = abs(_safe_float(info.get('cte', 0.0), 0.0))
                if info.get('hit', 'none')!= 'none':
                    collided = True
                if cte > self.current_max_cte * 1.5:
                    offtrack = True
                if done:
                    break
            if collided:
                stats['collision_in_20'] += 1
            if offtrack:
                stats['offtrack_in_20'] += 1

        if stats['min_dist_samples']:
            stats['min_dist_min'] = float(np.min(stats['min_dist_samples']))
            stats['min_dist_mean'] = float(np.mean(stats['min_dist_samples']))
        stats['npc_layout_reset_count'] = int(self.npc_layout_reset_count)
        stats['npc_layout_age_agent_resets'] = int(self.npc_layout_age_agent_resets)
        stats['lane_side_coverage'] = dict(self._lane_side_seen_counter)
        stats['segment_coverage'] = dict(self._segment_seen_counter)
        stats['precheck_fail_types'] = dict(self.spawn_precheck_fail_stats)
        stats['spawn_failure_reasons'] = dict(self.spawn_failure_reason_stats)
        return stats


class GeneratedTrackV11_1Wrapper(GeneratedTrackV10Wrapper):
    """V11.1 wrapper: single-car target-controller with anti-no-motion shaping."""

    def __init__(self, *args, **kwargs):
        # target space / controller params
        self.v_ref_min = float(kwargs.pop("v_ref_min", 0.05))
        self.v_ref_max = float(kwargs.pop("v_ref_max", 1.7))
        self.kappa_ref_max = float(kwargs.pop("kappa_ref_max", 2.1))
        self.v_ref_rate_max = float(kwargs.pop("v_ref_rate_max", 0.8))
        self.kappa_ref_rate_max = float(kwargs.pop("kappa_ref_rate_max", 4.0))
        self.steer_ff_headroom = float(kwargs.pop("steer_ff_headroom", 0.85))
        self.steer_softsat_gain = float(kwargs.pop("steer_softsat_gain", 1.0))
        self.steer_slew_rate_max = float(kwargs.pop("steer_slew_rate_max", 3.0))

        self.ctrl_dt_min = float(kwargs.pop("ctrl_dt_min", 0.01))
        self.ctrl_dt_max = float(kwargs.pop("ctrl_dt_max", 0.2))
        self.ctrl_dt_fallback = float(kwargs.pop("ctrl_dt_fallback", 0.05))

        self.gyro_lpf_tau = float(kwargs.pop("gyro_lpf_tau", 0.10))
        self.v_fb_min = float(kwargs.pop("v_fb_min", 0.25))
        self.yaw_kp = float(kwargs.pop("yaw_kp", 0.35))
        self.yaw_kd = float(kwargs.pop("yaw_kd", 0.0))
        self.yaw_ff_k = float(kwargs.pop("yaw_ff_k", 0.52))

        self.speed_kp = float(kwargs.pop("speed_kp", 0.9))
        self.speed_ki = float(kwargs.pop("speed_ki", 0.35))
        self.speed_kaw = float(kwargs.pop("speed_kaw", 0.5))
        self.throttle_brake_min = float(kwargs.pop("throttle_brake_min", 0.0))
        self.v_stop_eps = float(kwargs.pop("v_stop_eps", 0.08))
        self.speed_i_leak = float(kwargs.pop("speed_i_leak", 3.0))
        self.speed_i_fwd_max = float(kwargs.pop("speed_i_fwd_max", 0.40))
        self.speed_i_brake_max = float(kwargs.pop("speed_i_brake_max", 0.25))
        # B2 fix: notesavenote, super().__init__() (V10) note
        # curriculum_stage_ref note self.startup_force_throttle_steps(default10),
        # super() note V11.1 note.
        _startup_force_throttle_steps = int(kwargs.pop("startup_force_throttle_steps", 24))
        _startup_force_throttle = float(kwargs.pop("startup_force_throttle", 0.20))
        self.startup_force_throttle_steps = _startup_force_throttle_steps
        self.startup_force_throttle = _startup_force_throttle
        self.no_motion_penalty_start_steps = int(kwargs.pop("no_motion_penalty_start_steps", 40))
        self.no_motion_penalty_speed_thresh = float(kwargs.pop("no_motion_penalty_speed_thresh", 0.15))
        self.no_motion_penalty_progress_thresh = float(kwargs.pop("no_motion_penalty_progress_thresh", 0.002))
        self.no_motion_penalty_per_step = float(kwargs.pop("no_motion_penalty_per_step", 0.25))
        self.progress_local_window = int(kwargs.pop("progress_local_window", 120))
        self.progress_local_recover_dist = float(kwargs.pop("progress_local_recover_dist", 2.5))
        self.progress_idle_freeze_speed = float(kwargs.pop("progress_idle_freeze_speed", 0.12))
        self.progress_idle_freeze_dfi_abs = int(kwargs.pop("progress_idle_freeze_dfi_abs", 1))
        self.enable_progress_heading_filter = bool(kwargs.pop("enable_progress_heading_filter", False))
        self.progress_heading_dot_min = float(kwargs.pop("progress_heading_dot_min", -0.20))

        # sign check / unit check
        self.unit_calibrate = bool(kwargs.pop("unit_calibrate", True))
        self.unit_calib_steps = int(kwargs.pop("unit_calib_steps", 300))
        self.unit_calib_ratio_min = float(kwargs.pop("unit_calib_ratio_min", 2.0))
        self.unit_calib_strict = bool(kwargs.pop("unit_calib_strict", False))
        self.sign_check_min_steer = float(kwargs.pop("sign_check_min_steer", 0.08))
        self.sign_check_min_speed = float(kwargs.pop("sign_check_min_speed", 0.25))
        self.sign_check_max_lag = int(kwargs.pop("sign_check_max_lag", 15))
        self.sign_check_min_corr = float(kwargs.pop("sign_check_min_corr", 0.10))
        self.sign_check_min_samples = int(kwargs.pop("sign_check_min_samples", 40))

        super().__init__(*args, **kwargs)

        # B2 fix: super().__init__() (V10) note curriculum_stage_ref note,
        # notedefaultnote10note V11.1 note kwargs note(default24).
        # note super() note, note V11.1 note kwargs note.
        self.startup_force_throttle_steps = _startup_force_throttle_steps
        self.startup_force_throttle = _startup_force_throttle

        # controller state
        self.v_ref = 0.0
        self.kappa_ref = 0.0
        self.speed_i = 0.0
        self.gyro_z_f = 0.0
        self.prev_yaw_rate_err = 0.0
        self.last_speed_est = 0.0
        self._last_exec_steer = 0.0

        # one-step delayed timing source (no extra env.step probes in step())
        self.ctrl_last_info = {}
        self.ctrl_last_time = None
        self.ctrl_last_dt = float(np.clip(self.ctrl_dt_fallback, self.ctrl_dt_min, self.ctrl_dt_max))

        # sign/unit calibration state
        self.gyro_sign = 1.0
        self.ctrl_sign_flip_applied = False
        self._sign_check_locked = False
        self._sign_samples = deque(maxlen=max(200, self.sign_check_min_samples * 3))
        self._unit_samples = deque(maxlen=max(600, self.unit_calib_steps + 80))
        self._unit_calib_done = False
        self._unit_calib_ratio = None

        print("   PASS V11.1 controlnote: policy->[v_ref,kappa_ref]->controller->[steer,throttle]")
        print(f"      steer_softsat_gain={self.steer_softsat_gain:.2f}, ff_headroom={self.steer_ff_headroom:.2f}, "
              f"steer_slew_rate_max={self.steer_slew_rate_max:.2f}/s, kappa_ref_rate_max={self.kappa_ref_rate_max:.2f}/s")

    # ---------------- V11 helpers ----------------
    def _handler_info_snapshot(self):
        snap = {}
        try:
            h = self.env.viewer.handler
        except Exception:
            return snap

        snap["time"] = _safe_float(getattr(h, "time_received", time.time()), time.time())
        snap["speed"] = _safe_float(getattr(h, "speed", 0.0), 0.0)
        gx = _safe_float(getattr(h, "gyro_x", 0.0), 0.0)
        gy = _safe_float(getattr(h, "gyro_y", 0.0), 0.0)
        gz = _safe_float(getattr(h, "gyro_z", 0.0), 0.0)
        snap["gyro"] = (gx, gy, gz)
        snap["gyro_z"] = gz
        if hasattr(h, "yaw"):
            snap["yaw"] = _safe_float(getattr(h, "yaw", 0.0), 0.0)
        if hasattr(h, "steering_angle"):
            snap["steering_angle"] = _safe_float(getattr(h, "steering_angle", 0.0), 0.0)
        elif hasattr(h, "steering"):
            snap["steering_angle"] = _safe_float(getattr(h, "steering", 0.0), 0.0)
        return snap

    def _extract_time(self, info):
        t = None
        if isinstance(info, dict):
            t = info.get("time")
        if t is not None:
            tf = _safe_float(t, np.nan)
            if np.isfinite(tf) and tf > 0.0:
                return float(tf)
        try:
            tf = _safe_float(getattr(self.env.viewer.handler, "time_received"), np.nan)
            if np.isfinite(tf) and tf > 0.0:
                return float(tf)
        except Exception:
            pass
        return float(time.time())

    def _extract_gyro_z(self, info, apply_sign=True):
        raw = None
        if isinstance(info, dict):
            if info.get("gyro_z") is not None:
                raw = _safe_float(info.get("gyro_z"), np.nan)
            if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
                g = info.get("gyro")
                if isinstance(g, (tuple, list)) and len(g) >= 3:
                    raw = _safe_float(g[2], np.nan)
        if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
            try:
                raw = _safe_float(getattr(self.env.viewer.handler, "gyro_z"), np.nan)
            except Exception:
                raw = np.nan
        if not np.isfinite(raw):
            raw = 0.0
        raw = float(raw)
        if apply_sign:
            raw *= float(self.gyro_sign)
        return raw

    def _extract_speed(self, info):
        speed = np.nan
        if isinstance(info, dict) and info.get("speed") is not None:
            speed = _safe_float(info.get("speed"), np.nan)
        if not np.isfinite(speed):
            try:
                speed = _safe_float(getattr(self.env.viewer.handler, "speed"), np.nan)
            except Exception:
                speed = np.nan
        if not np.isfinite(speed):
            return float(self.last_speed_est)
        speed = float(max(0.0, speed))
        self.last_speed_est = speed
        return speed

    def _extract_yaw_deg(self, info):
        if isinstance(info, dict):
            if info.get("yaw") is not None:
                yaw = _safe_float(info.get("yaw"), np.nan)
                if np.isfinite(yaw):
                    return float(yaw)
            car = info.get("car")
            if isinstance(car, (tuple, list)) and len(car) >= 3:
                yaw = _safe_float(car[2], np.nan)
                if np.isfinite(yaw):
                    return float(yaw)
        try:
            yaw = _safe_float(getattr(self.env.viewer.handler, "yaw"), np.nan)
            if np.isfinite(yaw):
                return float(yaw)
        except Exception:
            pass
        return 0.0

    def _extract_steering_measure(self, info, fallback):
        if isinstance(info, dict):
            for key in ("steering_angle", "steering"):
                if info.get(key) is not None:
                    sv = _safe_float(info.get(key), np.nan)
                    if np.isfinite(sv):
                        return float(sv)
        try:
            h = self.env.viewer.handler
            for key in ("steering_angle", "steering"):
                if hasattr(h, key):
                    sv = _safe_float(getattr(h, key), np.nan)
                    if np.isfinite(sv):
                        return float(sv)
        except Exception:
            pass
        return float(fallback)

    def _clip_dt(self, dt_raw):
        dt = _safe_float(dt_raw, np.nan)
        if not np.isfinite(dt) or dt <= 0.0:
            fallback = float(np.clip(self.ctrl_dt_fallback, self.ctrl_dt_min, self.ctrl_dt_max))
            return fallback, True
        dt_clip = float(np.clip(dt, self.ctrl_dt_min, self.ctrl_dt_max))
        clipped = bool(abs(dt_clip - dt) > 1e-9)
        return dt_clip, clipped

    def _rate_limit(self, prev_val, cmd_val, rate_max, dt):
        dv_lim = max(0.0, float(rate_max)) * max(0.0, float(dt))
        return float(prev_val + np.clip(float(cmd_val) - float(prev_val), -dv_lim, dv_lim))

    @staticmethod
    def _corr(a, b):
        if len(a) < 8:
            return float("nan")
        aa = np.asarray(a, dtype=np.float64)
        bb = np.asarray(b, dtype=np.float64)
        aa = aa - np.mean(aa)
        bb = bb - np.mean(bb)
        sa = np.std(aa)
        sb = np.std(bb)
        if sa < 1e-12 or sb < 1e-12:
            return float("nan")
        return float(np.mean(aa * bb) / (sa * sb))

    def _maybe_update_sign_check(self, info, steer_exec):
        if self._sign_check_locked:
            return
        speed = self._extract_speed(info)
        steer_m = self._extract_steering_measure(info, steer_exec)
        if abs(steer_m) < self.sign_check_min_steer or speed < self.sign_check_min_speed:
            return
        gz_raw = self._extract_gyro_z(info, apply_sign=False)
        if not np.isfinite(gz_raw):
            return
        self._sign_samples.append((float(steer_m), float(gz_raw)))
        if len(self._sign_samples) < self.sign_check_min_samples:
            return

        arr = np.asarray(self._sign_samples, dtype=np.float64)
        steer = arr[:, 0]
        gyro = arr[:, 1]
        best = None
        max_lag = max(0, int(self.sign_check_max_lag))
        for lag in range(-max_lag, max_lag + 1):
            if lag >= 0:
                a = steer[:-lag or None]
                b = gyro[lag:]
            else:
                a = steer[-lag:]
                b = gyro[:lag or None]
            n = min(len(a), len(b))
            if n < self.sign_check_min_samples:
                continue
            a = a[:n]
            b = b[:n]
            for sign in (1, -1):
                c = self._corr(a, sign * b)
                if not np.isfinite(c):
                    continue
                score = abs(c)
                if best is None or score > best[0]:
                    best = (score, lag, sign, c, n)

        if best is None:
            return
        score, lag, sign, corr, n = best
        if score < self.sign_check_min_corr:
            return
        if sign < 0:
            self.gyro_sign = -1.0
            self.ctrl_sign_flip_applied = True
            print(f"⚠️ V11.1 sign-check: gyro_z sign flipped (corr={corr:.3f}, lag={lag}, n={n})")
        else:
            print(f"PASS V11.1 sign-check: gyro_z sign aligned (corr={corr:.3f}, lag={lag}, n={n})")
        self._sign_check_locked = True

    def _update_unit_calibration(self, info):
        if (not self.unit_calibrate) or self._unit_calib_done:
            return
        t = self._extract_time(info)
        yaw = self._extract_yaw_deg(info)
        gz = self._extract_gyro_z(info, apply_sign=False)
        spd = self._extract_speed(info)
        if not (np.isfinite(t) and np.isfinite(yaw) and np.isfinite(gz)):
            return
        self._unit_samples.append((float(t), float(yaw), float(gz), float(spd)))
        if len(self._unit_samples) < max(30, self.unit_calib_steps):
            return

        arr = np.asarray(self._unit_samples, dtype=np.float64)
        ts = arr[:, 0]
        yaw_deg = arr[:, 1]
        gyro = arr[:, 2]
        spd = arr[:, 3]

        dt = np.diff(ts)
        dyaw = np.diff(yaw_deg)
        dyaw = (dyaw + 180.0) % 360.0 - 180.0
        valid = np.isfinite(dt) & (dt > self.ctrl_dt_min * 0.5) & (dt < self.ctrl_dt_max * 2.0)
        valid = valid & (np.abs(spd[1:]) > 0.1)
        if int(np.sum(valid)) < max(20, self.unit_calib_steps // 5):
            return

        raddot = np.deg2rad(dyaw) / np.clip(dt, 1e-6, None)
        g = gyro[1:]
        rmse_rad = float(np.sqrt(np.mean((g[valid] - raddot[valid]) ** 2)))
        rmse_deg = float(np.sqrt(np.mean((g[valid] - np.rad2deg(raddot[valid])) ** 2)))
        ratio = rmse_deg / max(rmse_rad, 1e-6)
        self._unit_calib_done = True
        self._unit_calib_ratio = float(ratio)
        if ratio < self.unit_calib_ratio_min:
            msg = f"⚠️ V11.1 unit-check weak: rmse_deg/rmse_rad={ratio:.2f} (< {self.unit_calib_ratio_min:.2f})"
            if self.unit_calib_strict:
                raise RuntimeError(msg)
            print(msg)
        else:
            print(f"PASS V11.1 unit-check pass: rmse_deg/rmse_rad={ratio:.2f}")

    def _reset_controller_state(self):
        self.v_ref = float(self.v_ref_min)
        self.kappa_ref = 0.0
        self.speed_i = 0.0
        self.gyro_z_f = 0.0
        self.prev_yaw_rate_err = 0.0
        self._last_exec_steer = 0.0
        self.ctrl_last_dt = float(np.clip(self.ctrl_dt_fallback, self.ctrl_dt_min, self.ctrl_dt_max))
        self.steer_prev_limited = 0.0
        self.steer_prev_exec = 0.0
        self.delta_steer_prev = 0.0

    def _find_nearest_fine_track_local(self, tel_x, tel_z, prev_fi):
        fine = self._track_fine()
        n = len(fine)
        if not self.track_cache or n <= 1 or prev_fi is None:
            fi, dist = self.track_cache.find_nearest_fine_track(self.scene_name, tel_x, tel_z)
            return int(fi), float(dist), True

        center = int(prev_fi) % n
        w = max(4, min(int(self.progress_local_window), n // 2))
        best_i = center
        best_d2 = float("inf")
        for off in range(-w, w + 1):
            i = (center + off) % n
            fx, fz = fine[i]
            d2 = (tel_x - fx) ** 2 + (tel_z - fz) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        local_dist = math.sqrt(best_d2)

        # Teleport/relocalize protection: local result is too far, fallback to global nearest.
        if local_dist > self.progress_local_recover_dist:
            fi, dist = self.track_cache.find_nearest_fine_track(self.scene_name, tel_x, tel_z)
            return int(fi), float(dist), True
        return int(best_i), float(local_dist), False

    def _heading_dot_with_track_tangent(self, info, fi):
        fine = self._track_fine()
        n = len(fine)
        if n <= 2:
            return 1.0
        i = int(fi) % n
        px, pz = fine[(i - 1) % n]
        nx, nz = fine[(i + 1) % n]
        tx = nx - px
        tz = nz - pz
        tn = math.sqrt(tx * tx + tz * tz)
        if tn < 1e-9:
            return 1.0
        tx /= tn
        tz /= tn

        # Donkey telemetry yaw is degree-like heading; in x-z plane, yaw=0 is roughly +z.
        yaw_deg = self._extract_yaw_deg(info)
        yaw_rad = math.radians(float(yaw_deg))
        hx = math.sin(yaw_rad)
        hz = math.cos(yaw_rad)
        hn = math.sqrt(hx * hx + hz * hz)
        if hn < 1e-9:
            return 1.0
        hx /= hn
        hz /= hn
        return float(hx * tx + hz * tz)

    def _update_episode_progress(self, info, speed=None):
        """
        V11.1 progress estimator:
        1) local-window nearest fine index to reduce jump-segment errors
        2) net progress laps as primary estimate (forward-only kept as diagnostic)
        3) clip ratio / idle drift diagnostics for health monitoring
        """
        info = info if isinstance(info, dict) else {}
        fine = self._track_fine()
        fine_total = len(fine)
        if not self.track_cache or fine_total <= 1:
            info['progress_step_sim'] = 0.0
            info['episode_progress_laps_est'] = 0.0
            info['episode_progress_laps_signed_est'] = 0.0
            info['episode_progress_laps_forward_est'] = 0.0
            info['track_progress_ratio'] = 0.0
            info['episode_progress_ratio_to_goal'] = 0.0
            return 0, 0.0, 0

        lx, ly, lz = self._extract_pos(info)
        cur_fi, nearest_dist, global_recover = self._find_nearest_fine_track_local(
            lx, lz, self._episode_prev_fine_idx
        )
        heading_dot = 1.0
        heading_reject = False
        if self.enable_progress_heading_filter and self._episode_prev_fine_idx is not None:
            heading_dot = self._heading_dot_with_track_tangent(info, cur_fi)
            if heading_dot < self.progress_heading_dot_min:
                cur_fi = int(self._episode_prev_fine_idx)
                heading_reject = True

        self.learner_fine_idx = int(cur_fi)
        info['learner_fine_idx'] = int(cur_fi)
        info['progress_nearest_dist_sim'] = float(nearest_dist)
        info['progress_nearest_global_recover'] = bool(global_recover)
        info['progress_heading_dot'] = float(heading_dot)
        info['progress_heading_reject'] = bool(heading_reject)

        if self._episode_prev_fine_idx is None:
            self._episode_prev_fine_idx = int(cur_fi)
            lap_ratio = float(cur_fi) / float(max(1, fine_total))
            info['progress_step_fi_raw'] = 0
            info['progress_step_fi'] = 0
            info['progress_step_clipped'] = False
            info['progress_idle_freeze'] = False
            info['progress_step_sim'] = 0.0
            info['track_progress_ratio'] = lap_ratio
            info['episode_progress_laps_est'] = 0.0
            info['episode_progress_laps_signed_est'] = 0.0
            info['episode_progress_laps_forward_est'] = 0.0
            info['episode_progress_ratio_to_goal'] = 0.0
            return 0, 0.0, fine_total

        raw_dfi = int(self.track_cache.progress_diff(self.scene_name, int(cur_fi), int(self._episode_prev_fine_idx)))
        dfi = int(np.clip(raw_dfi, -self._progress_step_clip, self._progress_step_clip))
        clipped = bool(dfi!= raw_dfi)
        idle_freeze = False
        try:
            spd = abs(float(speed)) if speed is not None else abs(float(_safe_float(info.get('speed', 0.0), 0.0)))
        except Exception:
            spd = 0.0
        if spd < self.progress_idle_freeze_speed and abs(dfi) <= self.progress_idle_freeze_dfi_abs:
            dfi = 0
            idle_freeze = True

        self._episode_prev_fine_idx = int(cur_fi)
        step_sim = float(dfi) * max(self._fine_gap_sim, 1e-3)

        # accumulated progress stats
        self._episode_progress_fine_signed += float(dfi)
        if dfi > 0:
            self._episode_progress_fine_forward += float(dfi)
        elif dfi < 0:
            self._episode_progress_fine_backward += float(-dfi)

        if spd >= self.motion_arm_speed_thresh:
            self._motion_speed_streak += 1
        else:
            self._motion_speed_streak = 0

        disp_from_spawn = 0.0
        if self._episode_spawn_x is not None and self._episode_spawn_z is not None:
            dxs = float(lx) - float(self._episode_spawn_x)
            dzs = float(lz) - float(self._episode_spawn_z)
            disp_from_spawn = math.sqrt(dxs * dxs + dzs * dzs)
        fwd_progress_dist = float(self._episode_progress_fine_forward) * max(self._fine_gap_sim, 1e-3)
        if (
            self._motion_speed_streak >= self.motion_arm_speed_streak_steps
            or disp_from_spawn >= self.motion_arm_displacement_sim
            or fwd_progress_dist >= self.motion_arm_forward_progress_sim
        ):
            self._episode_motion_armed = True

        lap_ratio = float(cur_fi) / float(max(1, fine_total))
        laps_signed = float(self._episode_progress_fine_signed) / float(max(1, fine_total))
        laps_net = max(0.0, laps_signed)
        laps_forward = float(self._episode_progress_fine_forward) / float(max(1, fine_total))
        goal_ratio = min(1.0, laps_net / float(max(1, self.success_laps_target)))

        info['progress_step_fi_raw'] = int(raw_dfi)
        info['progress_step_fi'] = int(dfi)
        info['progress_step_clipped'] = bool(clipped)
        info['progress_idle_freeze'] = bool(idle_freeze)
        info['progress_step_sim'] = step_sim
        info['track_progress_ratio'] = lap_ratio
        info['episode_progress_laps_est'] = float(laps_net)
        info['episode_progress_laps_signed_est'] = float(laps_signed)
        info['episode_progress_laps_forward_est'] = float(laps_forward)
        info['episode_progress_ratio_to_goal'] = float(goal_ratio)
        info['motion_armed'] = bool(self._episode_motion_armed)
        info['motion_speed_streak'] = int(self._motion_speed_streak)
        info['motion_disp_from_spawn_sim'] = float(disp_from_spawn)
        info['motion_fwd_progress_sim'] = float(fwd_progress_dist)

        # health metrics for progress quality
        if isinstance(self.episode_stats, dict):
            self.episode_stats['progress_steps_total'] = int(self.episode_stats.get('progress_steps_total', 0)) + 1
            self.episode_stats['progress_clip_events'] = int(self.episode_stats.get('progress_clip_events', 0)) + int(clipped)
            self.episode_stats['progress_idle_freeze_steps'] = int(self.episode_stats.get('progress_idle_freeze_steps', 0)) + int(idle_freeze)
            self.episode_stats['progress_global_recover_events'] = int(self.episode_stats.get('progress_global_recover_events', 0)) + int(global_recover)
            self.episode_stats['progress_heading_reject_events'] = int(self.episode_stats.get('progress_heading_reject_events', 0)) + int(heading_reject)
            total = max(1, int(self.episode_stats.get('progress_steps_total', 1)))
            clip_ratio = float(self.episode_stats.get('progress_clip_events', 0)) / float(total)
            info['progress_clip_ratio'] = clip_ratio

        return dfi, step_sim, fine_total

    # ---------------- V11 reset / step ----------------
    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self._reset_controller_state()
        snap = self._handler_info_snapshot()
        self.ctrl_last_info = dict(snap) if isinstance(snap, dict) else {}
        self.ctrl_last_time = self._extract_time(self.ctrl_last_info)
        self.last_speed_est = self._extract_speed(self.ctrl_last_info)
        if isinstance(self.episode_stats, dict):
            self.episode_stats.setdefault('progress_steps_total', 0)
            self.episode_stats.setdefault('progress_clip_events', 0)
            self.episode_stats.setdefault('progress_idle_freeze_steps', 0)
            self.episode_stats.setdefault('progress_global_recover_events', 0)
            self.episode_stats.setdefault('progress_heading_reject_events', 0)
        return obs

    def step(self, action):
        self.episode_step += 1
        self.total_train_steps += 1

        # current control tick uses one-step delayed dt/info (no dummy env.step)
        prev_info = self.ctrl_last_info if isinstance(self.ctrl_last_info, dict) else {}
        if not prev_info:
            prev_info = self._handler_info_snapshot()
        dt_used, _ = self._clip_dt(self.ctrl_last_dt)
        v_est = self._extract_speed(prev_info)
        gyro_z_raw = self._extract_gyro_z(prev_info, apply_sign=True)
        lpf_alpha = float(np.clip(dt_used / max(1e-6, (self.gyro_lpf_tau + dt_used)), 0.0, 1.0))
        self.gyro_z_f = float((1.0 - lpf_alpha) * self.gyro_z_f + lpf_alpha * gyro_z_raw)

        # policy output => target refs
        a0 = float(np.clip(float(action[0]), -1.0, 1.0))
        a1 = float(np.clip(float(action[1]), -1.0, 1.0))
        v_cmd = self.v_ref_min + 0.5 * (a0 + 1.0) * (self.v_ref_max - self.v_ref_min)
        kappa_cmd = a1 * self.kappa_ref_max
        self.v_ref = self._rate_limit(self.v_ref, v_cmd, self.v_ref_rate_max, dt_used)
        self.kappa_ref = self._rate_limit(self.kappa_ref, kappa_cmd, self.kappa_ref_rate_max, dt_used)
        kappa_ref_raw = float(self.kappa_ref)
        ff_headroom_clipped = False

        # Feedforward headroom: cap FF steer and back-write kappa_ref to avoid frequent edge-hitting.
        ff_headroom = float(np.clip(self.steer_ff_headroom, 0.05, 0.99))
        yaw_ff_abs = abs(float(self.yaw_ff_k))
        if yaw_ff_abs > 1e-6:
            kappa_ff_limit = ff_headroom / yaw_ff_abs
            if abs(self.kappa_ref) > kappa_ff_limit:
                self.kappa_ref = float(np.clip(self.kappa_ref, -kappa_ff_limit, kappa_ff_limit))
                ff_headroom_clipped = True

        # lateral controller
        yaw_rate_ref = float(v_est * self.kappa_ref)
        yaw_rate_err = float(yaw_rate_ref - self.gyro_z_f)
        de = float((yaw_rate_err - self.prev_yaw_rate_err) / max(1e-6, dt_used))
        self.prev_yaw_rate_err = yaw_rate_err
        yaw_kp_eff = self.yaw_kp if v_est >= self.v_fb_min else 0.0
        steer_ff = float(np.clip(self.yaw_ff_k * self.kappa_ref, -ff_headroom, ff_headroom))
        steer_fb = float(yaw_kp_eff * yaw_rate_err + self.yaw_kd * de)
        steer_raw = float(steer_ff + steer_fb)
        softsat_gain = max(1e-3, float(self.steer_softsat_gain))
        steer_soft = float(np.tanh(softsat_gain * steer_raw))
        steer_exec = float(self._rate_limit(self.steer_prev_exec, steer_soft, self.steer_slew_rate_max, dt_used))
        steer_exec = float(np.clip(steer_exec, -1.0, 1.0))
        steer_sat = bool(abs(steer_soft) > 0.95)

        prev_delta_steer = float(self.delta_steer_prev)
        actual_delta = float(steer_exec - self.steer_prev_exec)
        self.delta_steer_prev = actual_delta
        self.steer_prev_limited = steer_exec
        self.steer_prev_exec = steer_exec
        self._last_exec_steer = steer_exec
        # P0-4 fix: V11.1 noterowsnote steer_slew_rate_max(1/s)xdt note,
        # rate_excess note controller note(per-step), note V9 delta_max(note).
        # note w_sat notecontrolnote: note _rate_limit note.
        allowed_per_step = max(float(self.steer_slew_rate_max) * float(dt_used), 1e-6)
        rate_excess_raw = max(0.0, abs(actual_delta) - allowed_per_step) / allowed_per_step
        rate_excess_bounded = float(np.tanh(rate_excess_raw))

        # longitudinal controller: PI + anti-windup + near-stop no-reverse clamp
        speed_err = float(self.v_ref - v_est)
        u_unsat = float(self.speed_kp * speed_err + self.speed_i)
        u_sat = float(np.clip(u_unsat, self.throttle_brake_min, self.max_throttle))
        near_stop_clamp = bool(v_est < self.v_stop_eps and u_sat < 0.0)
        if near_stop_clamp:
            u_sat = 0.0
            if speed_err < 0.0:
                self.speed_i *= max(0.0, 1.0 - self.speed_i_leak * dt_used)
        else:
            self.speed_i += float((self.speed_ki * speed_err + self.speed_kaw * (u_sat - u_unsat)) * dt_used)
        self.speed_i = float(np.clip(self.speed_i, -self.speed_i_brake_max, self.speed_i_fwd_max))
        throttle_exec = float(u_sat)
        if self.episode_step <= max(0, self.startup_force_throttle_steps):
            throttle_exec = float(max(throttle_exec, self.startup_force_throttle))

        safe_action = np.array([steer_exec, throttle_exec], dtype=np.float32)
        obs, _base_reward, gym_done, info = self.env.step(safe_action)
        done = False
        if gym_done:
            self._clear_handler_over()
        if not isinstance(info, dict):
            info = {}
        if self.curriculum_stage_ref.get('npc_count', 0) > 1:
            self._apply_npc_npc_collision_guard(info)
            self._maybe_handle_npc_npc_contact_reset(info)
        if self.curriculum_stage_ref.get('npc_count', 0) > 0:
            self._maybe_reset_stuck_npcs(info)
        processed_obs = self._process_observation(obs)

        # update next-step timing state
        info_time = self._extract_time(info)
        dt_raw_next = float(info_time - self.ctrl_last_time) if self.ctrl_last_time is not None else float(self.ctrl_dt_fallback)
        dt_next, dt_clipped_next = self._clip_dt(dt_raw_next)
        self.ctrl_last_time = float(info_time)
        self.ctrl_last_dt = float(dt_next)
        self.ctrl_last_info = dict(info)

        # online sign/unit checks (uses real rollout telemetry, no extra env.step)
        self._maybe_update_sign_check(info, steer_exec)
        self._update_unit_calibration(info)

        # diagnostics
        info['policy_action_0'] = float(a0)
        info['policy_action_1'] = float(a1)
        info['pilot_v_ref_cmd'] = float(v_cmd)
        info['pilot_kappa_cmd'] = float(kappa_cmd)
        info['ctrl_dt_raw'] = float(dt_raw_next)
        info['ctrl_dt'] = float(dt_used)
        info['ctrl_dt_clipped'] = bool(dt_clipped_next)
        info['pilot_v_ref'] = float(self.v_ref)
        info['pilot_kappa_ref_raw'] = float(kappa_ref_raw)
        info['pilot_kappa_ref'] = float(self.kappa_ref)
        info['ctrl_ff_headroom_clipped'] = bool(ff_headroom_clipped)
        info['ctrl_steer_exec'] = float(steer_exec)
        info['ctrl_steer_raw'] = float(steer_raw)
        info['ctrl_steer_soft'] = float(steer_soft)
        info['ctrl_steer_ff'] = float(steer_ff)
        info['ctrl_steer_fb'] = float(steer_fb)
        info['ctrl_steer_sat'] = bool(steer_sat)
        info['ctrl_yaw_rate_err'] = float(yaw_rate_err)
        info['ctrl_speed_err'] = float(speed_err)
        info['ctrl_gyro_z_raw'] = float(gyro_z_raw)
        info['ctrl_gyro_z_f'] = float(self.gyro_z_f)
        info['ctrl_throttle_exec'] = float(throttle_exec)
        info['ctrl_throttle_unsat'] = float(u_unsat)
        info['ctrl_throttle_sat'] = float(u_sat)
        info['ctrl_near_stop_clamp'] = bool(near_stop_clamp)
        info['ctrl_sign_flip_applied'] = bool(self.ctrl_sign_flip_applied)

        # keep reverse-gate diagnostics stable for downstream tools
        info['reverse_gate_blocked'] = False
        info['reverse_escape_active'] = False
        info['reverse_escape_opened'] = False
        info['reverse_gate_last_speed'] = float(v_est)
        info['reverse_gate_scaled'] = False

        reward, done = self._base_drive_reward(info, safe_action, actual_delta, prev_delta_steer, rate_excess_bounded)

        # negative throttle streak reset guard (kept for compatibility)
        speed_now = abs(_safe_float(info.get('speed', 0.0), 0.0))
        progress_step_now = _safe_float(info.get('episode_progress_step_sim', 0.0), 0.0)
        if (
            self.episode_step >= self.no_motion_penalty_start_steps
            and speed_now < self.no_motion_penalty_speed_thresh
            and abs(progress_step_now) <= self.no_motion_penalty_progress_thresh
        ):
            reward -= float(self.no_motion_penalty_per_step)
            self._rt_add(info, "no_motion_step_penalty", -float(self.no_motion_penalty_per_step))
            info['no_motion_step_penalty'] = float(self.no_motion_penalty_per_step)

        neg_request = bool(throttle_exec < 0.0)
        neg_reset_gate = bool(neg_request and speed_now <= self.negative_throttle_reset_speed_max and progress_step_now <= 0.005)
        if neg_reset_gate:
            self._negative_throttle_streak += 1
        else:
            self._negative_throttle_streak = 0
        info['negative_throttle_streak'] = int(self._negative_throttle_streak)

        reward_mode = self.curriculum_stage_ref.get('reward_mode', 'drive_only')
        if self.curriculum_stage_ref.get('npc_count', 0) > 0:
            if reward_mode == 'avoid_static':
                reward, done = self._reward_avoid_static(reward, done, info)
            elif reward_mode == 'follow_only':
                reward, done = self._reward_follow_only(reward, done, info)
            elif reward_mode == 'overtake':
                reward, done = self._reward_overtake_mode(reward, done, info)
            npc_metrics = getattr(self, "_last_step_npc_metrics", {}) or {}
            info['npc_count_seen'] = int(npc_metrics.get("count", 0))
            info['npc_encounter'] = bool(npc_metrics.get("encounter", False))
            if npc_metrics.get("min_dist") is not None:
                info['npc_min_dist'] = float(npc_metrics.get("min_dist"))

        reward = self._apply_reverse_displacement_penalty(reward, info)
        if self._negative_throttle_streak >= max(1, int(self.reverse_reset_steps)):
            done = True
            reward -= 2.0
            self._rt_add(info, "negative_throttle_reset_penalty", -2.0)
            info['negative_throttle_reset'] = True
            info['negative_throttle_reset_steps'] = int(self._negative_throttle_streak)
            info['termination_reason'] = 'negative_throttle_streak_reset'
            self.episode_stats['termination_reason'] = 'negative_throttle_streak_reset'
            self.episode_stats['reverse_reset_terminations'] = self.episode_stats.get('reverse_reset_terminations', 0) + 1

        if info.get('reverse_streak_reset', False):
            info['reverse_streak_reset_armed'] = True

        if self.episode_stats.get('collision'):
            feats = self._npc_distance_features(info)
            for f in feats:
                if f['dist'] <= float(self.dist_scale.get('danger_close_sim', 1.2)) * 1.25 and f['progress_diff'] < 0:
                    info['rear_end'] = True
                    self.episode_stats['rear_end'] = True
                    break
        if info.get('rear_end_risk'):
            info['rear_end'] = False

        if (
            bool(self.curriculum_stage_ref.get("terminate_on_success_laps", True))
            and info.get('task_success')
            and int(_safe_float(info.get('lap_count', 0), 0)) >= self.success_laps_target
        ):
            done = True
            info['episode_success_2laps'] = True
            self.episode_stats['termination_reason'] = 'success_laps_target'
            self.episode_stats['success_2laps'] = True

        if self.episode_step >= self.max_episode_steps:
            done = True
            if self.episode_stats.get('termination_reason') == 'max_steps':
                self.episode_stats['termination_reason'] = 'max_steps'

        if self.reward_clip_abs and self.reward_clip_abs > 0:
            raw_reward = float(reward)
            reward = float(np.clip(reward, -self.reward_clip_abs, self.reward_clip_abs))
            if reward!= raw_reward:
                info['reward_clipped'] = True
                info['raw_reward_unclipped'] = raw_reward
                self._rt_add(info, "reward_clip_adjust", float(reward - raw_reward))

        if isinstance(self.episode_stats, dict):
            self.episode_stats['ctrl_steer_sat_steps'] = int(self.episode_stats.get('ctrl_steer_sat_steps', 0)) + int(bool(steer_sat))
            self.episode_stats['ctrl_dt_clipped_steps'] = int(self.episode_stats.get('ctrl_dt_clipped_steps', 0)) + int(bool(info.get('ctrl_dt_clipped', False)))
            prev_steer = self._ep_prev_ctrl_steer_exec
            if prev_steer is not None:
                self.episode_stats.setdefault('ctrl_dsteer_abs_samples', []).append(abs(float(steer_exec) - float(prev_steer)))
            self._ep_prev_ctrl_steer_exec = float(steer_exec)
            prev_kappa = self._ep_prev_pilot_kappa_ref
            if prev_kappa is not None:
                self.episode_stats.setdefault('ctrl_dkappa_abs_samples', []).append(abs(float(self.kappa_ref) - float(prev_kappa)))
            self._ep_prev_pilot_kappa_ref = float(self.kappa_ref)

        self.episode_stats['total_reward'] += reward
        self.episode_stats['steps'] += 1
        self._rt_finalize(info, reward)
        return processed_obs, reward, done, info


class GeneratedTrackV11_2Wrapper(GeneratedTrackV11_1Wrapper):
    """V11.2: fixed2 mode with manual-width in-track spawn and fixed NPC refresh-on-success."""

    def __init__(self, *args, **kwargs):
        self.v11_2_mode = str(kwargs.pop("v11_2_mode", "fixed2"))
        self.manual_width_profile = str(
            kwargs.pop(
                "manual_width_profile",
                os.path.join(os.path.dirname(os.path.dirname(__file__)), "track_profiles", "manual_width_generated_track.json"),
            )
        )
        self.fixed_npc_count = int(kwargs.pop("fixed_npc_count", 2))
        self.fixed_success_progress = float(kwargs.pop("fixed_success_progress", 2.0))
        self.fixed_success_laps = int(kwargs.pop("fixed_success_laps", 2))
        self.success_no_hit_steps = int(kwargs.pop("success_no_hit_steps", 200))
        self.success_no_offtrack_seconds = float(kwargs.pop("success_no_offtrack_seconds", 3.0))
        self.success_cte_norm_margin = float(kwargs.pop("success_cte_norm_margin", 0.08))
        self.success_min_steps = int(kwargs.pop("success_min_steps", 300))
        self.success_progress_int_laps = int(kwargs.pop("success_progress_int_laps", 2))
        self.success_use_milestone_fallback = bool(kwargs.pop("success_use_milestone_fallback", False))

        self.stall_penalty_enable = bool(kwargs.pop("stall_penalty_enable", True))
        self.stall_speed_thresh = float(kwargs.pop("stall_speed_thresh", 0.15))
        self.stall_progress_step_thresh = float(kwargs.pop("stall_progress_step_thresh", 0.003))
        self.stall_penalty_per_step = float(kwargs.pop("stall_penalty_per_step", 0.002))
        self.stall_penalty_warmup_steps = int(kwargs.pop("stall_penalty_warmup_steps", 50000))

        self.spawn_local_window_m = float(kwargs.pop("spawn_local_window_m", 2.0))
        self.spawn_local_window_min_idx = int(kwargs.pop("spawn_local_window_min_idx", 20))
        self.spawn_anchor_window_m = float(kwargs.pop("spawn_anchor_window_m", 4.5))
        self.spawn_anchor_window_min_idx = int(kwargs.pop("spawn_anchor_window_min_idx", 48))
        self.spawn_safe_anchor_pool = int(kwargs.pop("spawn_safe_anchor_pool", 72))
        self.spawn_kappa_max = float(kwargs.pop("spawn_kappa_max", 1.6))
        self.spawn_inside_margin_ratio = float(kwargs.pop("spawn_inside_margin_ratio", 0.10))
        self.spawn_inside_max_attempts = int(kwargs.pop("spawn_inside_max_attempts", 24))

        self.npc_min_track_width_ratio = float(kwargs.pop("npc_min_track_width_ratio", 0.75))
        self.npc_min_track_width_abs_min = float(kwargs.pop("npc_min_track_width_abs_min", 0.8))
        self.layout_age_hard_cap = int(kwargs.pop("layout_age_hard_cap", 80))
        self.episode_summary_jsonl = str(kwargs.pop("episode_summary_jsonl", "") or "").strip()
        self.episode_summary_csv = str(kwargs.pop("episode_summary_csv", "") or "").strip()
        self.event_trace_jsonl = str(kwargs.pop("event_trace_jsonl", "") or "").strip()
        self.event_trace_window_steps = int(kwargs.pop("event_trace_window_steps", 60))
        self.event_trace_max_per_episode = int(kwargs.pop("event_trace_max_per_episode", 6))
        self.event_dt_spike_thresh = float(kwargs.pop("event_dt_spike_thresh", 0.08))
        self.event_sat_spike_cooldown_steps = int(kwargs.pop("event_sat_spike_cooldown_steps", 40))
        self.event_generic_cooldown_steps = int(kwargs.pop("event_generic_cooldown_steps", 40))
        self.event_reward_spike_abs = float(kwargs.pop("event_reward_spike_abs", 8.0))
        self.event_reward_gap_spike_abs = float(kwargs.pop("event_reward_gap_spike_abs", 2.0))
        self.event_progress_jump_fi = int(kwargs.pop("event_progress_jump_fi", 12))
        self.event_progress_clip_ratio_high = float(kwargs.pop("event_progress_clip_ratio_high", 0.25))
        # P3 fix: defaultnote 4.4 note 3.2, note spawn_min_gap_sim_ego_npc=4.0,
        # note(4.0~4.4)note spawn_near note.
        self.event_spawn_near_dist = float(kwargs.pop("event_spawn_near_dist", 3.2))
        self.event_periodic_sample_steps = int(kwargs.pop("event_periodic_sample_steps", 20000))

        super().__init__(*args, **kwargs)

        if self._fixed2_enabled():
            # noteNPCnotestagenotefixed2noteNPC/rewardrowsnote; stage1(npc_count=0)note
            _init_npc_count = int(self.curriculum_stage_ref.get("npc_count", 0))
            if _init_npc_count > 0:
                # npc_count notestageconfigurationnote, note fixed_npc_count note
                self.curriculum_stage_ref["npc_count"] = _init_npc_count
                # npc_mode notestageconfiguration(static / wobble note)
                if not self.curriculum_stage_ref.get("npc_mode"):
                    self.curriculum_stage_ref["npc_mode"] = "static"
                self.curriculum_stage_ref["reward_mode"] = "avoid_static"
                if not self.curriculum_stage_ref.get("npc_layout_reset_on_collision"):
                    self.curriculum_stage_ref["npc_layout_reset_on_collision"] = False
                if not self.curriculum_stage_ref.get("npc_layout_reset_on_success"):
                    self.curriculum_stage_ref["npc_layout_reset_on_success"] = False
                if not self.curriculum_stage_ref.get("npc_layout_reset_policy"):
                    self.curriculum_stage_ref["npc_layout_reset_policy"] = "agent_only"
            self.curriculum_stage_ref["terminate_on_success_laps"] = False
            # note STAGE_TRAIN_PROFILES[stage].random_start_enabled control, note

        self.manual_spawn = ManualWidthSpawnSampler(self.manual_width_profile, self.scene_name)
        if self._fixed2_enabled():
            if self.manual_spawn.loaded:
                print(
                    "   PASS V11.2 manual-width spawn loaded:",
                    f"fine={len(self.manual_spawn.fine_track)}",
                    f"fine_gap={self.manual_spawn.fine_gap_sim:.4f}",
                    f"width_med={self.manual_spawn.width_median_sim:.3f}",
                )
            else:
                print(f"   ⚠️ V11.2 manual-width spawn disabled: {self.manual_spawn.error}")

        self._fixed_guard = FixedSuccessGuardState(self.success_no_hit_steps)
        self._fixed_force_refresh_next = False
        self._fixed_episode_max_progress = 0.0
        self._fixed_episode_max_lap = 0
        self._fixed_episode_max_milestone = 0
        self._fixed_last_spawn_debug = {}
        self._fixed_last_spawn_source = "unknown"
        self._fixed_last_spawn_meta = {}
        self._fixed_last_refresh_reason = "init"
        self._fixed_last_refresh_was_refresh = False
        self._fixed_layout_hash = ""
        self._fixed_last_fine_idx = 0
        self._trace_recent_steps = deque(maxlen=max(10, int(self.event_trace_window_steps)))
        self._trace_pending_events = []
        self._trace_event_seq = 0
        self._trace_events_this_episode = 0
        self._trace_prev_sat = False
        self._trace_prev_dt_clip = False
        self._trace_last_sat_event_step = -10**9
        self._trace_last_dt_event_step = -10**9
        self._trace_last_periodic_bucket = -1
        self._trace_last_generic_event_step = {}
        self._summary_csv_header_written = False

        for p in (self.episode_summary_jsonl, self.episode_summary_csv, self.event_trace_jsonl):
            if p:
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
                except Exception:
                    pass

    def _p95(self, values):
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, dtype=np.float32), 95))

    def _append_jsonl(self, path, payload):
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"⚠️ write jsonl failed ({path}): {e}")

    def _append_csv_row(self, path, row):
        if not path:
            return
        try:
            write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        except Exception:
            write_header = True
        try:
            with open(path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            print(f"⚠️ write csv failed ({path}): {e}")

    def _on_episode_end(self, avg_cte, term_reason):
        stats = self.episode_stats if isinstance(self.episode_stats, dict) else {}
        steps = int(stats.get("steps", 0))
        if steps <= 0:
            return
        progress_steps = max(1, int(stats.get("progress_steps_total", 0)))
        speed_mean = float(stats.get("speed_sum", 0.0)) / float(max(1, steps))
        row = {
            "ts_unix": int(time.time()),
            "episode": int(self.total_episodes),
            "global_step": int(self.total_train_steps),
            "episode_steps": steps,
            "termination_reason": str(term_reason),
            "reward_total": float(stats.get("total_reward", 0.0)),
            "speed_mean": float(speed_mean),
            "speed_max": float(stats.get("max_speed", 0.0)),
            "cte_mean": float(avg_cte),
            "progress_laps_max": float(max(stats.get("progress_laps_max", 0.0), stats.get("progress_laps_est", 0.0))),
            "laps_max": int(stats.get("lap_count_max", 0)),
            "collision": int(bool(stats.get("collision", False))),
            "rear_end": int(bool(stats.get("rear_end", False))),
            "sat_ratio": float(stats.get("ctrl_steer_sat_steps", 0)) / float(max(1, steps)),
            "dsteer_abs_p95": float(self._p95(stats.get("ctrl_dsteer_abs_samples", []))),
            "dkappa_abs_p95": float(self._p95(stats.get("ctrl_dkappa_abs_samples", []))),
            "progress_clip_ratio": float(stats.get("progress_clip_events", 0)) / float(progress_steps),
            "global_recover_ratio": float(stats.get("progress_global_recover_events", 0)) / float(progress_steps),
            "stall_ratio": float(stats.get("stall_penalty_steps", 0)) / float(max(1, steps)),
            "dt_clipped_ratio": float(stats.get("ctrl_dt_clipped_steps", 0)) / float(max(1, steps)),
            "spawn_source": str(stats.get("spawn_source", "")),
            "npc_layout_hash": str(stats.get("npc_layout_hash", "")),
            "refresh_reason": str(stats.get("npc_layout_refresh_reason", "")),
            "spawn_fail_reason": str(stats.get("spawn_fail_reason", "")),
            "spawn_retries": int(stats.get("spawn_retries", 0)),
            "spawn_safe_anchor_count": int(stats.get("spawn_safe_anchor_count", 0)),
            "ego_npc_min_dist_spawn": float(_safe_float(stats.get("ego_npc_min_dist_spawn", -1.0), -1.0)),
        }
        self._append_jsonl(self.episode_summary_jsonl, row)
        self._append_csv_row(self.episode_summary_csv, row)
        # notepost-windownote, note.
        self._trace_flush_pending(force=True)

    def _trace_build_step_snapshot(self, info, reward, done):
        pos = info.get("pos", (np.nan, np.nan, np.nan))
        if not isinstance(pos, (list, tuple)) or len(pos) < 3:
            pos = (np.nan, np.nan, np.nan)
        yaw_deg = _safe_float(info.get("yaw", np.nan), np.nan)
        if not np.isfinite(yaw_deg):
            try:
                yaw_deg = float(self._extract_yaw_deg(info))
            except Exception:
                yaw_deg = float("nan")
        terms = info.get("reward_terms", {})
        if not isinstance(terms, dict):
            terms = {}
        terms_slim = {}
        for k, v in terms.items():
            vv = _safe_float(v, np.nan)
            if np.isfinite(vv):
                terms_slim[str(k)] = float(vv)
        return {
            "global_step": int(self.total_train_steps),
            "episode_step": int(self.episode_step),
            "reward": float(_safe_float(reward, 0.0)),
            "done": bool(done),
            "termination_reason": str(info.get("termination_reason", "")),
            "cte": float(_safe_float(info.get("cte", 0.0), 0.0)),
            "speed": float(_safe_float(info.get("speed", 0.0), 0.0)),
            "pos_x": float(_safe_float(pos[0], np.nan)),
            "pos_z": float(_safe_float(pos[2], np.nan)),
            "yaw_deg": float(yaw_deg) if np.isfinite(yaw_deg) else None,
            "lap_count": int(_safe_float(info.get("lap_count", 0), 0)),
            "progress_step_sim": float(_safe_float(info.get("episode_progress_step_sim", 0.0), 0.0)),
            "progress_laps": float(_safe_float(info.get("episode_progress_laps_est", 0.0), 0.0)),
            "progress_step_fi_raw": int(_safe_float(info.get("progress_step_fi_raw", 0), 0)),
            "progress_step_fi": int(_safe_float(info.get("progress_step_fi", 0), 0)),
            "progress_step_clipped": bool(info.get("progress_step_clipped", False)),
            "progress_clip_ratio": float(_safe_float(info.get("progress_clip_ratio", 0.0), 0.0)),
            "policy_action_0": float(_safe_float(info.get("policy_action_0", 0.0), 0.0)),
            "policy_action_1": float(_safe_float(info.get("policy_action_1", 0.0), 0.0)),
            "pilot_v_ref_cmd": float(_safe_float(info.get("pilot_v_ref_cmd", 0.0), 0.0)),
            "pilot_kappa_cmd": float(_safe_float(info.get("pilot_kappa_cmd", 0.0), 0.0)),
            "pilot_v_ref": float(_safe_float(info.get("pilot_v_ref", 0.0), 0.0)),
            "pilot_kappa_ref": float(_safe_float(info.get("pilot_kappa_ref", 0.0), 0.0)),
            "ctrl_steer_exec": float(_safe_float(info.get("ctrl_steer_exec", 0.0), 0.0)),
            "ctrl_steer_ff": float(_safe_float(info.get("ctrl_steer_ff", 0.0), 0.0)),
            "ctrl_steer_fb": float(_safe_float(info.get("ctrl_steer_fb", 0.0), 0.0)),
            "ctrl_steer_sat": bool(info.get("ctrl_steer_sat", False)),
            "ctrl_throttle_exec": float(_safe_float(info.get("ctrl_throttle_exec", 0.0), 0.0)),
            "ctrl_yaw_rate_err": float(_safe_float(info.get("ctrl_yaw_rate_err", 0.0), 0.0)),
            "ctrl_speed_err": float(_safe_float(info.get("ctrl_speed_err", 0.0), 0.0)),
            "ctrl_gyro_z_raw": float(_safe_float(info.get("ctrl_gyro_z_raw", 0.0), 0.0)),
            "ctrl_gyro_z_f": float(_safe_float(info.get("ctrl_gyro_z_f", 0.0), 0.0)),
            "ctrl_dt_raw": float(_safe_float(info.get("ctrl_dt_raw", 0.0), 0.0)),
            "ctrl_dt": float(_safe_float(info.get("ctrl_dt", 0.0), 0.0)),
            "ctrl_dt_clipped": bool(info.get("ctrl_dt_clipped", False)),
            "npc_min_dist": float(_safe_float(info.get("npc_min_dist", -1.0), -1.0)),
            "hit": str(info.get("hit", "none")),
            "spawn_validation_pass": bool(info.get("spawn_validation_pass", True)),
            "spawn_fail_reason": str(info.get("spawn_fail_reason", "")),
            "ego_npc_min_dist": float(_safe_float(info.get("ego_npc_min_dist", -1.0), -1.0)),
            "negative_throttle_streak": int(_safe_float(info.get("negative_throttle_streak", 0), 0)),
            "reward_terms_total": float(_safe_float(info.get("reward_terms_total", 0.0), 0.0)),
            "reward_terms_gap": float(_safe_float(info.get("reward_terms_gap", 0.0), 0.0)),
            "reward_terms": terms_slim,
        }

    def _trace_emit_event(self, event):
        payload = {
            "event_id": str(event.get("event_id", "")),
            "event_type": str(event.get("event_type", "")),
            "event_reason": str(event.get("event_reason", "")),
            "trigger_global_step": int(event.get("trigger_global_step", 0)),
            "trigger_episode_step": int(event.get("trigger_episode_step", 0)),
            "window_size": int(self.event_trace_window_steps),
            "post_truncated": bool(event.get("post_truncated", False)),
            "pre_steps": event.get("pre_steps", []),
            "trigger_step": event.get("trigger_step", {}),
            "post_steps": event.get("post_steps", []),
        }
        self._append_jsonl(self.event_trace_jsonl, payload)

    def _trace_start_event(self, event_type, event_reason, trigger_snapshot):
        if not self.event_trace_jsonl:
            return
        if int(self._trace_events_this_episode) >= int(max(1, self.event_trace_max_per_episode)):
            return
        self._trace_event_seq += 1
        event = {
            "event_id": f"ev_{int(time.time())}_{int(self._trace_event_seq)}",
            "event_type": str(event_type),
            "event_reason": str(event_reason),
            "trigger_global_step": int(trigger_snapshot.get("global_step", 0)),
            "trigger_episode_step": int(trigger_snapshot.get("episode_step", 0)),
            "pre_steps": list(self._trace_recent_steps),
            "trigger_step": dict(trigger_snapshot),
            "post_steps": [],
            "post_remaining": int(max(1, self.event_trace_window_steps)),
            "post_truncated": False,
        }
        self._trace_pending_events.append(event)
        self._trace_events_this_episode += 1

    def _trace_cooldown_ok(self, key, cooldown_steps):
        cd = int(max(1, cooldown_steps))
        cur = int(self.total_train_steps)
        last = int(self._trace_last_generic_event_step.get(str(key), -10**9))
        if (cur - last) < cd:
            return False
        self._trace_last_generic_event_step[str(key)] = cur
        return True

    def _trace_flush_pending(self, force=False):
        if not self.event_trace_jsonl:
            self._trace_pending_events = []
            return
        keep = []
        for ev in self._trace_pending_events:
            if force:
                ev["post_truncated"] = bool(ev.get("post_remaining", 0) > 0)
                self._trace_emit_event(ev)
            else:
                keep.append(ev)
        self._trace_pending_events = keep

    def _trace_on_step(self, info, reward, done):
        if not self.event_trace_jsonl:
            return
        snap = self._trace_build_step_snapshot(info, reward, done)

        # First, append current step to existing pending events as post-window samples.
        next_pending = []
        for ev in self._trace_pending_events:
            if int(snap["global_step"]) > int(ev.get("trigger_global_step", -1)):
                ev["post_steps"].append(dict(snap))
                ev["post_remaining"] = int(ev.get("post_remaining", 0)) - 1
            if int(ev.get("post_remaining", 0)) <= 0 or bool(done):
                ev["post_truncated"] = bool(done and int(ev.get("post_remaining", 0)) > 0)
                self._trace_emit_event(ev)
            else:
                next_pending.append(ev)
        self._trace_pending_events = next_pending

        # Event triggers
        term = str(info.get("termination_reason", ""))
        if term in (
            "collision",
            "persistent_offtrack",
            "stuck",
            "no_motion_timeout",
            "negative_throttle_streak_reset",
            "success_avoidance_2laps",
            "success_laps_target",
        ):
            self._trace_start_event(term, term, snap)

        if str(info.get("npc_layout_refresh_reason", "")) == "age_cap" and (not bool(info.get("npc_layout_reused", True))):
            self._trace_start_event("layout_refresh", "age_cap", snap)

        # spawn diagnostics (focus on episode start)
        if int(snap.get("episode_step", 0)) <= 2:
            spawn_ok = bool(info.get("spawn_validation_pass", True))
            spawn_fail_reason = str(info.get("spawn_fail_reason", "") or "")
            ego_spawn_dist = float(_safe_float(info.get("ego_npc_min_dist", -1.0), -1.0))
            if (not spawn_ok) or spawn_fail_reason:
                if self._trace_cooldown_ok("spawn_fail", self.event_generic_cooldown_steps):
                    self._trace_start_event("spawn_fail", spawn_fail_reason or "spawn_validation_failed", snap)
            elif ego_spawn_dist >= 0.0 and ego_spawn_dist < float(self.event_spawn_near_dist):
                if self._trace_cooldown_ok("spawn_near", self.event_generic_cooldown_steps):
                    self._trace_start_event("spawn_near", "ego_npc_min_dist_low", snap)

        sat_now = bool(info.get("ctrl_steer_sat", False))
        if sat_now and (not self._trace_prev_sat):
            if int(self.total_train_steps - self._trace_last_sat_event_step) >= int(max(1, self.event_sat_spike_cooldown_steps)):
                self._trace_start_event("sat_spike", "ctrl_steer_sat_rise", snap)
                self._trace_last_sat_event_step = int(self.total_train_steps)
        self._trace_prev_sat = sat_now

        dt_clipped_now = bool(info.get("ctrl_dt_clipped", False))
        dt_raw = float(_safe_float(info.get("ctrl_dt_raw", 0.0), 0.0))
        dt_used = float(_safe_float(info.get("ctrl_dt", 0.0), 0.0))
        dt_spike = dt_clipped_now or (abs(dt_raw - dt_used) >= float(max(0.0, self.event_dt_spike_thresh)))
        if dt_spike and (not self._trace_prev_dt_clip):
            if int(self.total_train_steps - self._trace_last_dt_event_step) >= int(max(1, self.event_sat_spike_cooldown_steps)):
                self._trace_start_event("dt_spike", "dt_clipped_or_gap", snap)
                self._trace_last_dt_event_step = int(self.total_train_steps)
        self._trace_prev_dt_clip = dt_spike

        reward_abs = abs(float(_safe_float(snap.get("reward", 0.0), 0.0)))
        reward_gap_abs = abs(float(_safe_float(info.get("reward_terms_gap", 0.0), 0.0)))
        if reward_abs >= float(self.event_reward_spike_abs) or reward_gap_abs >= float(self.event_reward_gap_spike_abs):
            if self._trace_cooldown_ok("reward_spike", self.event_generic_cooldown_steps):
                self._trace_start_event("reward_spike", "reward_or_gap_spike", snap)

        raw_dfi_abs = abs(int(_safe_float(info.get("progress_step_fi_raw", 0), 0)))
        clip_ratio = float(_safe_float(info.get("progress_clip_ratio", 0.0), 0.0))
        progress_spike = bool(info.get("progress_step_clipped", False)) or raw_dfi_abs >= int(max(1, self.event_progress_jump_fi))
        progress_spike = progress_spike or (clip_ratio >= float(self.event_progress_clip_ratio_high))
        if progress_spike:
            if self._trace_cooldown_ok("progress_anomaly", self.event_generic_cooldown_steps):
                self._trace_start_event("progress_anomaly", "progress_jump_or_clip", snap)

        periodic_every = int(max(0, self.event_periodic_sample_steps))
        if periodic_every > 0:
            bucket = int(self.total_train_steps // periodic_every)
            if bucket > self._trace_last_periodic_bucket:
                self._trace_last_periodic_bucket = int(bucket)
                if bucket > 0:
                    self._trace_start_event("periodic_sample", f"every_{periodic_every}_steps", snap)

        self._trace_recent_steps.append(dict(snap))
        if done:
            self._trace_flush_pending(force=True)

    def _update_episode_aux_stats(self, info):
        if not isinstance(self.episode_stats, dict) or not isinstance(info, dict):
            return
        if "spawn_source" in info:
            self.episode_stats["spawn_source"] = str(info.get("spawn_source", ""))
        if "npc_layout_hash" in info:
            self.episode_stats["npc_layout_hash"] = str(info.get("npc_layout_hash", ""))
        if "npc_layout_refresh_reason" in info:
            self.episode_stats["npc_layout_refresh_reason"] = str(info.get("npc_layout_refresh_reason", ""))
        if "spawn_fail_reason" in info:
            self.episode_stats["spawn_fail_reason"] = str(info.get("spawn_fail_reason", ""))
        if "spawn_retries" in info:
            self.episode_stats["spawn_retries"] = int(_safe_float(info.get("spawn_retries", 0), 0))
        if "spawn_safe_anchor_count" in info:
            self.episode_stats["spawn_safe_anchor_count"] = int(_safe_float(info.get("spawn_safe_anchor_count", 0), 0))
        # P4 fix: ego_npc_min_dist_spawn note episode note(spawn note)note,
        # note NPC noterowsnote, note.
        if "ego_npc_min_dist" in info and float(_safe_float(self.episode_stats.get("ego_npc_min_dist_spawn", -1.0), -1.0)) < 0.0:
            self.episode_stats["ego_npc_min_dist_spawn"] = float(_safe_float(info.get("ego_npc_min_dist", -1.0), -1.0))

    def _fixed2_enabled(self):
        return str(self.v11_2_mode).lower() == "fixed2"

    def _compute_layout_hash(self):
        poses = []
        for item in (self.npc_layout_cached_poses or []):
            pose = item.get("pose") or {}
            tel = pose.get("tel", (0.0, 0.0, 0.0))
            fi = int(_safe_float(pose.get("fine_idx", -1), -1))
            nid = int(_safe_float(item.get("npc_id", -1), -1))
            lane = str(item.get("lane_side", "center"))
            poses.append((nid, fi, round(_safe_float(tel[0], 0.0), 3), round(_safe_float(tel[2], 0.0), 3), lane))
        raw = json.dumps(sorted(poses), ensure_ascii=False)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10] if poses else ""

    def _spawn_window_idx(self):
        n = len(self._track_fine())
        if n <= 1:
            return max(4, int(self.spawn_local_window_min_idx))
        fine_gap = self.manual_spawn.fine_gap_sim if self.manual_spawn.loaded else max(1e-3, float(self._fine_gap_sim))
        w = int(round(float(self.spawn_local_window_m) / max(1e-4, fine_gap)))
        w = max(int(self.spawn_local_window_min_idx), w)
        w = min(w, max(4, n // 2))
        return int(w)

    def _spawn_anchor_window_idx(self):
        n = len(self._track_fine())
        if n <= 1:
            return max(8, int(self.spawn_anchor_window_min_idx))
        fine_gap = self.manual_spawn.fine_gap_sim if self.manual_spawn.loaded else max(1e-3, float(self._fine_gap_sim))
        w = int(round(float(self.spawn_anchor_window_m) / max(1e-4, fine_gap)))
        w = max(int(self.spawn_anchor_window_min_idx), w)
        w = min(w, max(8, n // 2))
        return int(w)

    def _nearest_fine_idx_local(self, x, z, seed_idx):
        fine = self._track_fine()
        n = len(fine)
        if n <= 0:
            return 0, float("inf")
        seed = int(seed_idx) % n
        w = self._spawn_window_idx()
        best_i = seed
        best_d2 = float("inf")
        for off in range(-w, w + 1):
            i = (seed + off) % n
            fx, fz = fine[i]
            d2 = (float(fx) - float(x)) ** 2 + (float(fz) - float(z)) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
        return int(best_i), float(math.sqrt(best_d2))

    def _track_kappa(self, fi):
        if self.manual_spawn.loaded and len(self.manual_spawn.fine_track) > 8:
            return float(self.manual_spawn.estimate_kappa(fi))
        fine = self._track_fine()
        n = len(fine)
        if n <= 2:
            return 0.0
        i = int(fi) % n
        a = fine[(i - 1) % n]
        b = fine[i]
        c = fine[(i + 1) % n]
        bax = float(a[0] - b[0])
        baz = float(a[1] - b[1])
        bcx = float(c[0] - b[0])
        bcz = float(c[1] - b[1])
        la = math.sqrt(bax * bax + baz * baz)
        lc = math.sqrt(bcx * bcx + bcz * bcz)
        acx = float(c[0] - a[0])
        acz = float(c[1] - a[1])
        l_ac = math.sqrt(acx * acx + acz * acz)
        denom = max(1e-6, la * lc * l_ac)
        cross = abs(bax * bcz - baz * bcx)
        return float(2.0 * cross / denom)

    def _half_width_cte_at_fine(self, fi):
        """note(min(note,note)), note."""
        if self.manual_spawn.loaded:
            return float(max(1e-3, self.manual_spawn.half_width_narrow_cte_at(fi)))
        return float(max(1e-3, self.current_max_cte))

    def _half_width_avg_cte_at_fine(self, fi):
        """note((note+note)/2), noteCTEnoteontracknote."""
        if self.manual_spawn.loaded:
            return float(max(1e-3, self.manual_spawn.half_width_cte_at(fi)))
        return float(max(1e-3, self.current_max_cte))

    def _half_width_wide_cte_at_fine(self, fi):
        """note(max(note,note)), noteCTEnote."""
        if self.manual_spawn.loaded:
            return float(max(1e-3, self.manual_spawn.half_width_wide_cte_at(fi)))
        return float(max(1e-3, self.current_max_cte))

    def _fixed2_collect_safe_ego_anchors(self, npc_poses, constraints):
        if (not self._fixed2_enabled()) or (not self.manual_spawn.loaded) or (not self.track_cache):
            return []
        nodes = self._track_nodes()
        n = len(nodes)
        if n <= 0 or (not npc_poses):
            return []

        c_ego = constraints.get("ego_npc", {}) if isinstance(constraints, dict) else {}
        min_e = float(c_ego.get("min_euclid", 0.0))
        min_g = int(c_ego.get("min_progress_gap", 0))
        max_keep = max(8, int(self.spawn_safe_anchor_pool))

        # v11.3: bad_nodes note, note
        candidates = list(range(n))
        random.shuffle(candidates)

        safe = []
        for idx in candidates:
            fi = self._node_to_fine_idx(idx)
            if abs(self._track_kappa(fi)) > float(self.spawn_kappa_max):
                continue
            x, _, z = self._node_tel(nodes[int(idx) % n])
            ok = True
            for pose in npc_poses:
                tel = pose.get("tel", None) if isinstance(pose, dict) else None
                if not (isinstance(tel, (list, tuple)) and len(tel) >= 3):
                    continue
                nx, _, nz = tel
                euclid = math.sqrt((float(x) - float(nx)) ** 2 + (float(z) - float(nz)) ** 2)
                if euclid < min_e:
                    ok = False
                    break
                nf = int(_safe_float((pose or {}).get("fine_idx", fi), fi))
                gap = abs(int(self.track_cache.progress_diff(self.scene_name, int(fi), int(nf))))
                if gap < min_g:
                    ok = False
                    break
            if ok:
                safe.append(int(idx))
                if len(safe) >= max_keep:
                    break
        return safe

    def _sample_pose_from_node(self, node_idx, jitter_scale=1.0, exact=False, lane_side=None):
        if self._fixed2_enabled() and lane_side == "center" and self.manual_spawn.loaded:
            nodes = self._track_nodes()
            y_tel = self._node_tel(nodes[int(node_idx) % len(nodes)])[1] if nodes else 0.0625
            anchor_fi = self._node_to_fine_idx(node_idx)
            anchor_w = self._spawn_anchor_window_idx()
            pose = None
            for _ in range(max(1, int(self.spawn_inside_max_attempts))):
                pose = self.manual_spawn.sample_pose(
                    y_tel=y_tel,
                    node_count=len(nodes),
                    margin_ratio=self.spawn_inside_margin_ratio,
                    kappa_max=self.spawn_kappa_max,
                    window_m=self.spawn_local_window_m,
                    window_min_idx=self.spawn_local_window_min_idx,
                    anchor_fi=anchor_fi,
                    anchor_window_idx=anchor_w,
                )
                if pose is None:
                    continue
                if self.track_cache:
                    fi_local, fd = self._nearest_fine_idx_local(pose["tel"][0], pose["tel"][2], pose["fine_idx"])
                    pose["fine_idx"] = int(fi_local)
                    pose["fine_dist_sim"] = float(fd)
                self._fixed_last_spawn_source = "manual_width"
                self._fixed_last_spawn_meta = {
                    "spawn_source": "manual_width",
                    "spawn_margin_ratio": float(pose.get("spawn_margin_ratio", self.spawn_inside_margin_ratio)),
                    "spawn_kappa": float(pose.get("spawn_kappa", 0.0)),
                    "spawn_fine_idx": int(pose.get("fine_idx", 0)),
                    "spawn_local_window_idx": int(pose.get("spawn_local_window_idx", self._spawn_window_idx())),
                    "spawn_anchor_fine_idx": int(anchor_fi),
                    "spawn_anchor_window_idx": int(anchor_w),
                }
                return pose
            self._fixed_last_spawn_source = "fallback_centerline_safe"
            self._fixed_last_spawn_meta = {"spawn_source": "fallback_centerline_safe"}
            return super()._sample_pose_from_node(node_idx, jitter_scale=0.3, exact=True, lane_side=lane_side)
        return super()._sample_pose_from_node(node_idx, jitter_scale=jitter_scale, exact=exact, lane_side=lane_side)

    def _fixed2_build_npc_pose(self, fi, lane_side):
        nodes = self._track_nodes()
        if not nodes:
            return None
        if not self.manual_spawn.loaded:
            return None
        n = len(self.manual_spawn.fine_track)
        i = int(fi) % n
        ratio = 0.78 if str(lane_side) == "left" else 0.22
        rx, rz = self.manual_spawn.right_boundary[i]
        lx, lz = self.manual_spawn.left_boundary[i]
        x = float((1.0 - ratio) * rx + ratio * lx)
        z = float((1.0 - ratio) * rz + ratio * lz)
        tx, tz = self.manual_spawn.tangent(i)
        yaw_deg = math.degrees(math.atan2(tx, tz))
        qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
        fi_local, fd = self._nearest_fine_idx_local(x, z, i)
        node_idx = int(round(float(fi_local) * float(len(nodes)) / float(max(1, n)))) % len(nodes)
        y_tel = self._node_tel(nodes[node_idx])[1]
        S = float(SimExtendedAPI.COORD_SCALE)
        return {
            "node_idx": int(node_idx),
            "fine_idx": int(fi_local),
            "tel": (x, float(y_tel), z),
            "node_coords": (x * S, y_tel * S, z * S, qx, qy, qz, qw),
            "yaw_deg": float(yaw_deg),
            "lane_side": str(lane_side),
            "dn_offset_sim": float((ratio - 0.5) * self.manual_spawn.width_sim[fi_local]),
            "fine_dist_sim": float(fd),
            "segment_id": None,
        }

    def _build_npc_target_poses(self, active_npc_count):
        if not self._fixed2_enabled():
            return super()._build_npc_target_poses(active_npc_count)
        count = int(active_npc_count)
        if count <= 0:
            return [], []
        if not self.manual_spawn.loaded:
            return super()._build_npc_target_poses(active_npc_count)

        constraints = self._spawn_constraints()
        c_npc = constraints.get("npc_npc", {})
        min_e = float(c_npc.get("min_euclid", 1.2))
        min_g = int(c_npc.get("min_progress_gap", 24))
        width_min = max(
            float(self.npc_min_track_width_abs_min),
            float(self.npc_min_track_width_ratio) * float(self.manual_spawn.width_median_sim),
        )
        n = len(self.manual_spawn.fine_track)

        # v11.4: note ego note fine_idx note(note), note NPC note ego firstnote.
        # note agent note NPC, note.
        ego_fi_hint = getattr(self, 'learner_fine_idx', None)
        if ego_fi_hint is None or not isinstance(ego_fi_hint, int):
            ego_fi_hint = None

        # note fine points
        all_candidates = [
            i for i in range(n)
            if float(self.manual_spawn.width_sim[i]) >= width_min and abs(self._track_kappa(i)) <= float(self.spawn_kappa_max)
        ]
        if len(all_candidates) < count:
            return [], []

        # v11.4: note ego note, note ego firstnote 40-250 fine points note
        # (note 1-6m tracknote), NPC note agent note, note.
        NPC_AHEAD_MIN = 40    # note (~1m), note ego note
        NPC_AHEAD_MAX = 250   # note (~6m), note
        candidates = all_candidates  # fallback
        if ego_fi_hint is not None:
            ahead_candidates = []
            for fi in all_candidates:
                ahead_dist = (fi - ego_fi_hint) % n  # note(firstnote)note
                if NPC_AHEAD_MIN <= ahead_dist <= NPC_AHEAD_MAX:
                    ahead_candidates.append(fi)
            if len(ahead_candidates) >= count:
                candidates = ahead_candidates
            # else: fallback note candidates

        lane_sides = ["left", "right"] + ["center"] * max(0, count - 2)
        random.shuffle(lane_sides)
        chosen = []
        for _ in range(240):
            chosen = []
            random.shuffle(candidates)
            for fi in candidates:
                ok = True
                for cj in chosen:
                    gap = abs(int(self.track_cache.progress_diff(self.scene_name, int(fi), int(cj)))) if self.track_cache else abs(fi - cj)
                    gap = min(gap, n - gap)
                    px, pz = self.manual_spawn.fine_track[fi]
                    qx, qz = self.manual_spawn.fine_track[cj]
                    euclid = math.sqrt((float(px) - float(qx)) ** 2 + (float(pz) - float(qz)) ** 2)
                    if gap < min_g or euclid < min_e:
                        ok = False
                        break
                if ok:
                    chosen.append(int(fi))
                if len(chosen) >= count:
                    break
            if len(chosen) >= count:
                break
        if len(chosen) < count:
            return [], []

        poses, meta = [], []
        for fi, side in zip(chosen[:count], lane_sides[:count]):
            pose = self._fixed2_build_npc_pose(fi, side)
            if pose is None:
                continue
            poses.append(pose)
            meta.append({"segment_id": None, "lane_side": side})
        return poses, meta

    def _build_npc_ahead_of_ego(self, npc_count, ego_fine_idx):
        """v11.4 ego-first NPC note: note ego firstnote 40-250 fine points note NPC note.

        note agent notefirstnote NPC obstacle, note.
        NPC 1 note ego firstnote 40-120 fine points(note, note)
        NPC 2 note ego firstnote 130-250 fine points(note, noteobstacle)
        """
        if not self.manual_spawn.loaded:
            return []
        n = len(self.manual_spawn.fine_track)
        count = int(npc_count)
        if count <= 0 or n <= 0:
            return []

        constraints = self._spawn_constraints()
        c_npc = constraints.get("npc_npc", {})
        min_g = int(c_npc.get("min_progress_gap", 24))

        width_min = max(
            float(self.npc_min_track_width_abs_min),
            float(self.npc_min_track_width_ratio) * float(self.manual_spawn.width_median_sim),
        )

        # note NPC notefirstnote(fine points offset from ego)
        # NPC 1: note [40, 120]   -> agent note 10-30 note
        # NPC 2: note [130, 250]  -> agent note 30-60 note
        # note NPC: notefirst
        npc_zones = []
        for i in range(count):
            zone_start = 40 + i * 90   # 40, 130, 220,...
            zone_end = 120 + i * 90    # 120, 210, 300,...
            zone_end = min(zone_end, n // 2)  # note
            if zone_start >= zone_end:
                zone_start = max(40, n // (count + 1) * i)
                zone_end = min(zone_start + 80, n // 2)
            npc_zones.append((zone_start, zone_end))

        lane_sides = ["left", "right"] + ["center"] * max(0, count - 2)
        random.shuffle(lane_sides)

        chosen_fis = []
        for zone_idx, (zs, ze) in enumerate(npc_zones):
            zone_candidates = []
            for offset in range(zs, ze):
                fi = (ego_fine_idx + offset) % n
                if float(self.manual_spawn.width_sim[fi]) < width_min:
                    continue
                if abs(self._track_kappa(fi)) > float(self.spawn_kappa_max):
                    continue
                # note NPC note
                ok = True
                for cj in chosen_fis:
                    gap = abs(fi - cj)
                    gap = min(gap, n - gap)
                    if gap < min_g:
                        ok = False
                        break
                if ok:
                    zone_candidates.append(fi)

            if zone_candidates:
                chosen_fis.append(random.choice(zone_candidates))
            else:
                # zone note, note [20, n//2]
                for offset in range(20, n // 2):
                    fi = (ego_fine_idx + offset) % n
                    if float(self.manual_spawn.width_sim[fi]) < width_min:
                        continue
                    if abs(self._track_kappa(fi)) > float(self.spawn_kappa_max):
                        continue
                    ok = True
                    for cj in chosen_fis:
                        gap = abs(fi - cj)
                        gap = min(gap, n - gap)
                        if gap < min_g:
                            ok = False
                            break
                    if ok:
                        chosen_fis.append(fi)
                        break

            if len(chosen_fis) > count:
                break

        if len(chosen_fis) < count:
            return []

        poses = []
        for fi, side in zip(chosen_fis[:count], lane_sides[:count]):
            pose = self._fixed2_build_npc_pose(fi, side)
            if pose is not None:
                poses.append(pose)
        return poses

    def should_refresh_npc_layout(self, active_npcs):
        if not self._fixed2_enabled():
            return super().should_refresh_npc_layout(active_npcs)
        count = len(active_npcs)
        stage = self._current_stage_id()
        persist_mode = bool(self._npc_persist_mode_enabled())
        refresh = False
        reason = "fixed2_reuse"
        if bool(getattr(self, "_force_npc_layout_refresh_next", False)):
            refresh = True
            reason = "force_refresh_flag"
            self._force_npc_layout_refresh_next = False
        elif count <= 0:
            refresh = False
            reason = "no_active_npc"
        elif not self.npc_layout_cached_poses:
            refresh = True
            reason = "no_cached_layout"
        elif self._npc_layout_active_count!= count:
            refresh = True
            reason = "active_count_changed"
        elif self._npc_layout_stage_id!= stage:
            refresh = True
            reason = "stage_changed"
        elif self._consecutive_layout_failures >= self._layout_fail_refresh_threshold():
            refresh = True
            reason = "consecutive_layout_failures"
        elif self._fixed_force_refresh_next:
            refresh = not persist_mode
            reason = "success_avoidance_2laps" if refresh else "success_refresh_suppressed"
            self._fixed_force_refresh_next = False
        elif (not persist_mode) and self.npc_layout_age_agent_resets >= int(max(1, self.layout_age_hard_cap)):
            refresh = True
            reason = "age_cap"

        self._fixed_last_refresh_reason = str(reason)
        self._fixed_last_refresh_was_refresh = bool(refresh)
        return bool(refresh), str(reason)

    def reset(self, **kwargs):
        self._fixed_last_spawn_source = "unknown"
        self._fixed_last_spawn_meta = {}
        obs = super().reset(**kwargs)
        self._fixed_last_spawn_debug = dict(self._last_spawn_debug) if isinstance(self._last_spawn_debug, dict) else {}
        self._trace_recent_steps.clear()
        self._trace_pending_events = []
        self._trace_events_this_episode = 0
        self._trace_prev_sat = False
        self._trace_prev_dt_clip = False
        self._fixed_guard.reset(self.success_no_hit_steps)
        self._fixed_episode_max_progress = 0.0
        self._fixed_episode_max_lap = 0
        self._fixed_episode_max_milestone = 0
        self._fixed_layout_hash = self._compute_layout_hash()
        if isinstance(self.episode_stats, dict):
            self.episode_stats.setdefault("stall_penalty_sum", 0.0)
            self.episode_stats.setdefault("stall_penalty_steps", 0)
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if not isinstance(info, dict):
            info = {}
        self._update_episode_aux_stats(info)
        if not self._fixed2_enabled():
            self._trace_on_step(info, reward, done)
            return obs, reward, done, info

        base_reward = float(reward)
        dt = max(0.0, _safe_float(info.get("ctrl_dt", self.ctrl_dt_fallback), self.ctrl_dt_fallback))
        hit_v = str(info.get("hit", "none"))
        hit_event = (hit_v!= "none")

        progress_laps = float(_safe_float(info.get("episode_progress_laps_est", 0.0), 0.0))
        lap_count = int(_safe_float(info.get("lap_count", 0), 0))
        milestone = int(_safe_float(info.get("progress_milestone", 0), 0))
        self._fixed_episode_max_progress = max(float(self._fixed_episode_max_progress), progress_laps)
        self._fixed_episode_max_lap = max(int(self._fixed_episode_max_lap), lap_count)
        self._fixed_episode_max_milestone = max(int(self._fixed_episode_max_milestone), milestone)

        fi = info.get("learner_fine_idx", None)
        if fi is None:
            lx, _, lz = self._extract_pos(info)
            fi, _ = self._nearest_fine_idx_local(lx, lz, self._fixed_last_fine_idx)
        fi = int(_safe_float(fi, self._fixed_last_fine_idx))
        self._fixed_last_fine_idx = int(fi)
        half_w_cte = self._half_width_cte_at_fine(fi)
        cte = abs(_safe_float(info.get("cte", 0.0), 0.0))
        cte_norm = float(cte / max(1e-6, half_w_cte))
        offtrack_flag = bool(cte_norm > (1.0 + float(self.success_cte_norm_margin)))
        self._fixed_guard.update(dt=dt, hit_event=hit_event, offtrack_flag=offtrack_flag)

        no_hit_window = bool(self._fixed_guard.no_hit_window_ok())
        no_offtrack_window = bool(self._fixed_guard.no_offtrack_window_ok(self.success_no_offtrack_seconds))

        if self.stall_penalty_enable:
            speed_now = abs(_safe_float(info.get("speed", 0.0), 0.0))
            progress_step_now = abs(_safe_float(info.get("episode_progress_step_sim", 0.0), 0.0))
            stall_active = bool(speed_now < self.stall_speed_thresh and progress_step_now < self.stall_progress_step_thresh)
            if stall_active:
                warm = 1.0
                if int(self.stall_penalty_warmup_steps) > 0:
                    warm = float(np.clip(float(self.total_train_steps) / float(self.stall_penalty_warmup_steps), 0.0, 1.0))
                pen = float(self.stall_penalty_per_step) * float(warm)
                reward -= pen
                self._rt_add(info, "stall_penalty", -pen)
                info["stall_penalty"] = float(pen)
                info["stall_penalty_active"] = True
                info["stall_penalty_warmup_scale"] = float(warm)
                if isinstance(self.episode_stats, dict):
                    self.episode_stats["stall_penalty_sum"] = float(self.episode_stats.get("stall_penalty_sum", 0.0)) + float(pen)
                    self.episode_stats["stall_penalty_steps"] = int(self.episode_stats.get("stall_penalty_steps", 0)) + 1
            else:
                info["stall_penalty_active"] = False

        progress_ok = bool(self._fixed_episode_max_progress >= float(self.fixed_success_progress))
        lap_ok = bool(self._fixed_episode_max_lap >= int(self.fixed_success_laps))
        steps_ok = bool(self.episode_step >= int(self.success_min_steps))
        progress_int_ok = bool(int(math.floor(self._fixed_episode_max_progress + 1e-9)) >= int(self.success_progress_int_laps))
        milestone_ok = bool(self._fixed_episode_max_milestone >= int(self.success_progress_int_laps))

        success_main = bool(progress_ok and lap_ok and no_hit_window and no_offtrack_window and steps_ok)
        success_progress_only = bool(progress_int_ok and no_hit_window and no_offtrack_window and steps_ok)
        success_milestone = bool(self.success_use_milestone_fallback and milestone_ok and no_hit_window and no_offtrack_window and steps_ok)
        success_guard_passed = bool(success_main or success_progress_only or success_milestone)

        if (not done) and success_guard_passed:
            done = True
            info["task_success"] = True
            info["episode_success_2laps"] = True
            info["termination_reason"] = "success_avoidance_2laps"
            self._fixed_force_refresh_next = True
            if isinstance(self.episode_stats, dict):
                self.episode_stats["termination_reason"] = "success_avoidance_2laps"
                self.episode_stats["success_2laps"] = True

        info["success_guard_progress"] = bool(progress_ok)
        info["success_guard_lap"] = bool(lap_ok)
        info["success_guard_no_hit"] = bool(no_hit_window)
        info["success_guard_no_offtrack"] = bool(no_offtrack_window)
        info["success_guard_progress_int"] = bool(progress_int_ok)
        info["success_guard_milestone"] = bool(success_milestone)
        info["success_guard_passed"] = bool(success_guard_passed)
        info["success_guard_offtrack_run_sec"] = float(self._fixed_guard.offtrack_run_sec)
        info["success_guard_cte_norm"] = float(cte_norm)
        info["spawn_source"] = str(self._fixed_last_spawn_source)
        info["spawn_margin_ratio"] = float(self._fixed_last_spawn_meta.get("spawn_margin_ratio", self.spawn_inside_margin_ratio))
        info["spawn_kappa"] = float(self._fixed_last_spawn_meta.get("spawn_kappa", 0.0))
        info["spawn_fine_idx"] = int(_safe_float(self._fixed_last_spawn_meta.get("spawn_fine_idx", fi), fi))
        info["spawn_local_window_idx"] = int(_safe_float(self._fixed_last_spawn_meta.get("spawn_local_window_idx", self._spawn_window_idx()), self._spawn_window_idx()))
        info["spawn_anchor_fine_idx"] = int(_safe_float(self._fixed_last_spawn_meta.get("spawn_anchor_fine_idx", fi), fi))
        info["spawn_anchor_window_idx"] = int(_safe_float(self._fixed_last_spawn_meta.get("spawn_anchor_window_idx", self._spawn_anchor_window_idx()), self._spawn_anchor_window_idx()))
        spawn_dbg = self._fixed_last_spawn_debug if isinstance(self._fixed_last_spawn_debug, dict) else {}
        info["spawn_validation_pass"] = bool(spawn_dbg.get("spawn_validation_pass", True))
        info["spawn_fail_reason"] = str(spawn_dbg.get("spawn_fail_reason", "") or "")
        info["spawn_retries"] = int(_safe_float(spawn_dbg.get("spawn_retries", 0), 0))
        info["spawn_safe_anchor_count"] = int(_safe_float(spawn_dbg.get("safe_anchor_count", 0), 0))
        if spawn_dbg.get("ego_npc_min_dist") is not None:
            info["ego_npc_min_dist"] = float(_safe_float(spawn_dbg.get("ego_npc_min_dist", -1.0), -1.0))
        if spawn_dbg.get("ego_npc_progress_gap") is not None:
            info["ego_npc_progress_gap"] = float(_safe_float(spawn_dbg.get("ego_npc_progress_gap", -1.0), -1.0))
        info["npc_layout_hash"] = str(self._fixed_layout_hash)
        info["npc_layout_reused"] = bool(not self._fixed_last_refresh_was_refresh)
        info["npc_layout_refresh_reason"] = str(self._fixed_last_refresh_reason)
        info["npc_layout_refresh_on_success"] = bool(self._fixed_last_refresh_reason == "success_avoidance_2laps")
        info["npc_layout_age"] = int(self.npc_layout_age_agent_resets)

        delta_reward = float(reward) - float(base_reward)
        if abs(delta_reward) > 1e-12 and isinstance(self.episode_stats, dict):
            self.episode_stats["total_reward"] = float(self.episode_stats.get("total_reward", 0.0)) + float(delta_reward)
        self._update_episode_aux_stats(info)
        self._rt_finalize(info, reward)
        self._trace_on_step(info, reward, done)

        return obs, float(reward), bool(done), info


class CurriculumManager(object):
    """trainingnote(learn chunknote)noterowsnote."""

    def __init__(self, curriculum_stage_ref, eval_freq_steps=20000, eval_episodes=2,
                 success_laps=2, consecutive_success_required=2,
                 args=None, cli_overrides=None, enable_stage_profiles=True,
                 eval_max_steps=1000):
        self.ref = curriculum_stage_ref
        self.eval_freq_steps = int(eval_freq_steps)
        self.eval_episodes = int(eval_episodes)
        self.success_laps = int(success_laps)
        self.consecutive_success_required = int(consecutive_success_required)
        self.args = args
        self.cli_overrides = cli_overrides or set()
        self.enable_stage_profiles = bool(enable_stage_profiles)
        self.eval_max_steps = int(eval_max_steps)
        self.last_eval_at = 0
        self.stage_enter_step = 0
        self.consecutive_eval_success = 0

    def set_stage(self, stage_id, global_step, wrapper=None, model=None):
        stage = STAGES.get(int(stage_id), STAGES[1])
        self.ref['stage'] = stage.sid
        self.ref['reward_mode'] = stage.reward_mode
        self.ref['npc_count'] = stage.npc_count
        self.ref['npc_mode'] = stage.npc_mode
        self.ref['enable_overtake_reward'] = stage.enable_overtake_reward
        self.ref['p_ego_behind'] = stage.p_ego_behind
        self.ref['npc_speed_min'], self.ref['npc_speed_max'] = stage.npc_speed_range
        self.ref['spawn_max_attempts'] = 8
        self.consecutive_eval_success = 0
        self.stage_enter_step = int(global_step)
        profile = None
        if self.enable_stage_profiles:
            profile = apply_stage_train_profile(
                stage.sid, self.ref, args=self.args, wrapper=wrapper, model=model,
                cli_overrides=self.cli_overrides, verbose=False
            )
        # P0-1 fix: apply_stage_train_profile note npc_layout_reset_policy note fixed2 note;
        # note set_stage note fixed2 note, note.
        if self.args is not None:
            _apply_v11_2_curriculum_overrides(self.args, self.ref)
            _apply_npc_mode_override(self.args, self.ref)
        gate = STAGE_EVAL_GATES.get(stage.sid)
        print(f"🎓 notestage -> {stage.sid}: {stage.name} | reward={stage.reward_mode} | npc={self.ref.get('npc_count', stage.npc_count)}")
        if profile is not None:
            print("   StageProfile:",
                  f"throttle={self.ref.get('max_throttle', getattr(wrapper, 'max_throttle', 'n/a')) if wrapper else getattr(self.args, 'max_throttle', 'n/a')}",
                  f"prog={self.ref.get('progress_reward_scale')}",
                  f"milestone=({self.ref.get('progress_milestone_lap')},{self.ref.get('progress_milestone_reward')})",
                  f"random_start={self.ref.get('stage_random_start_enabled')}",
                  f"gaps ego-npc=({self.ref.get('spawn_min_gap_sim_ego_npc')},{self.ref.get('spawn_min_gap_progress_ego_npc')})")
        if gate is not None:
            print("   EvalGate:",
                  f"min_steps={gate.min_stage_steps}",
                  f"eval_every={gate.eval_every_steps}",
                  f"eps={gate.eval_episodes}",
                  f"passes={gate.consecutive_passes_required}",
                  f"min_prog={gate.min_relative_progress_laps}",
                  f"lap>={gate.require_lap_count_at_least}")

    def maybe_eval_and_promote(self, model, wrapper, global_step):
        stage_id = int(self.ref.get('stage', 1))
        stage = STAGES.get(stage_id, STAGES[1])
        gate = STAGE_EVAL_GATES.get(stage_id) if self.enable_stage_profiles else None
        eval_every = int(gate.eval_every_steps) if gate is not None else self.eval_freq_steps
        min_stage_steps = int(gate.min_stage_steps) if gate is not None else int(stage.min_stage_steps)
        eval_episodes = int(gate.eval_episodes) if gate is not None else self.eval_episodes
        required_passes = int(gate.consecutive_passes_required) if gate is not None else self.consecutive_success_required

        if global_step - self.last_eval_at < eval_every:
            return None
        if global_step - self.stage_enter_step < min_stage_steps:
            self.last_eval_at = global_step
            print(f"⏳ stage{stage_id}note {min_stage_steps}, note")
            return None

        self.last_eval_at = global_step
        early_stop_laps = self.success_laps
        if gate is not None and int(gate.require_lap_count_at_least) > 0:
            early_stop_laps = max(1, int(gate.require_lap_count_at_least))
        result = evaluate_stage_on_same_env(
            model, wrapper, self.ref, n_episodes=eval_episodes,
            success_laps=early_stop_laps, eval_gate=gate,
            eval_max_steps=self.eval_max_steps,
            stop_on_laps=False,
        )
        ok = bool(result.get('gate_passed', result.get('success', False)))
        if ok:
            self.consecutive_eval_success += 1
        else:
            self.consecutive_eval_success = 0

        print(f"🧪 Eval(stage={stage_id}) success={ok} laps={result.get('avg_laps', 0):.2f} "
              f"rel_prog={result.get('avg_relative_progress_laps', 0):.2f} "
              f"collision_rate={result.get('collision_rate', 0):.0%} rear_end_rate={result.get('rear_end_rate', 0):.0%} "
              f"consecutive={self.consecutive_eval_success}/{required_passes}")

        if self.consecutive_eval_success >= required_passes and stage_id < max(STAGES.keys()):
            self.set_stage(stage_id + 1, global_step, wrapper=wrapper, model=model)
            result['promoted_to'] = stage_id + 1
        return result


def evaluate_stage_on_same_env(model, wrapper, curriculum_stage_ref, n_episodes=2, success_laps=2,
                               eval_gate=None, eval_max_steps=1000, stop_on_laps=False):
    """notetrainingnote(note).note RecurrentPPO predict note."""
    out = {
        'episodes': [],
        'success': False,
        'gate_passed': False,
        'avg_laps': 0.0,
        'avg_relative_progress_laps': 0.0,
        'collision_rate': 0.0,
        'rear_end_rate': 0.0,
        'npc_layout_reset_count': 0,
        'npc_layout_age_mean': 0.0,
        'lane_side_coverage': {},
        'segment_coverage': {},
        'avg_unsafe_follow_ratio': 0.0,
        'total_npc_encounters': 0,
        'total_safe_overtakes': 0,
        'total_reverse_events': 0,
    }

    for ep in range(int(n_episodes)):
        obs = wrapper.reset()
        lstm_state = None
        episode_start = np.array([True], dtype=bool)
        done = False
        ep_steps = 0
        collisions = False
        rear_end = False
        laps = 0
        term_reason = None
        max_rel_prog = 0.0
        npc_encounters = 0
        safe_overtakes = 0
        unsafe_cutin = 0
        encounter_active = False

        max_eval_steps = int(eval_max_steps) if int(eval_max_steps) > 0 else int(wrapper.max_episode_steps)
        max_eval_steps = min(int(wrapper.max_episode_steps), max_eval_steps)
        while not done and ep_steps < max_eval_steps:
            # RecurrentPPO note PPO notepredictnote(note/note)
            action, lstm_state = model.predict(obs, state=lstm_state, episode_start=episode_start, deterministic=True)
            obs, reward, done, info = wrapper.step(action)
            episode_start = np.array([done], dtype=bool)
            ep_steps += 1
            laps = max(laps, int(_safe_float(info.get('lap_count', 0), 0)))
            max_rel_prog = max(max_rel_prog, float(_safe_float(info.get('episode_progress_laps_est', 0.0), 0.0)))
            if info.get('hit', 'none')!= 'none':
                collisions = True
            if info.get('rear_end', False):
                rear_end = True
            has_enc = bool(info.get('npc_encounter', False))
            if has_enc and not encounter_active:
                npc_encounters += 1
            encounter_active = has_enc
            if info.get('overtake_success', False):
                safe_overtakes += 1
            if info.get('unsafe_cutin', False):
                unsafe_cutin += 1
            term_reason = info.get('termination_reason', term_reason)
            if stop_on_laps and laps >= success_laps and not collisions:
                # notesucceedednotefirstnote
                done = True

        # B3 fix: fixed2 notesucceedednote "success_avoidance_2laps",
        # note lap_count note success_progress_only/milestone pathnote success_laps,
        # notesucceedednote.
        fixed2_success = (term_reason == "success_avoidance_2laps") and (not collisions) and (not rear_end)
        success = (fixed2_success or (laps >= success_laps)) and (not collisions) and (term_reason not in ('persistent_offtrack', 'stuck')) and (not rear_end)
        ep_stats = getattr(wrapper, "episode_stats", {}) or {}
        reverse_events = int(ep_stats.get('reverse_penalty_events', 0))
        unsafe_follow_steps = int(ep_stats.get('unsafe_follow_steps', 0))
        unsafe_follow_ratio = (float(unsafe_follow_steps) / float(max(1, ep_steps)))
        out['episodes'].append({
            'success': success,
            'laps': laps,
            'relative_progress_laps': max_rel_prog,
            'collision': collisions,
            'rear_end': rear_end,
            'steps': ep_steps,
            'termination_reason': term_reason,
            'reverse_events': reverse_events,
            'unsafe_follow_ratio': unsafe_follow_ratio,
            'npc_encounters': npc_encounters,
            'safe_overtakes': safe_overtakes,
            'unsafe_cutin': unsafe_cutin,
        })

    if out['episodes']:
        out['avg_laps'] = float(np.mean([e['laps'] for e in out['episodes']]))
        out['avg_relative_progress_laps'] = float(np.mean([e['relative_progress_laps'] for e in out['episodes']]))
        out['collision_rate'] = float(np.mean([1.0 if e['collision'] else 0.0 for e in out['episodes']]))
        out['rear_end_rate'] = float(np.mean([1.0 if e['rear_end'] else 0.0 for e in out['episodes']]))
        out['avg_unsafe_follow_ratio'] = float(np.mean([e['unsafe_follow_ratio'] for e in out['episodes']]))
        out['total_npc_encounters'] = int(sum(e['npc_encounters'] for e in out['episodes']))
        out['total_safe_overtakes'] = int(sum(e['safe_overtakes'] for e in out['episodes']))
        out['total_reverse_events'] = int(sum(e['reverse_events'] for e in out['episodes']))
        out['success'] = all(e['success'] for e in out['episodes'])
    out['gate_passed'] = stage_eval_gate_pass(out, eval_gate)
    out['npc_layout_reset_count'] = int(getattr(wrapper, 'npc_layout_reset_count', 0))
    out['npc_layout_age_mean'] = float(getattr(wrapper, 'npc_layout_age_agent_resets', 0))
    out['lane_side_coverage'] = dict(getattr(wrapper, '_lane_side_seen_counter', {}))
    out['segment_coverage'] = dict(getattr(wrapper, '_segment_seen_counter', {}))
    return out


def _extract_step_from_ckpt_path(path):
    m = re.search(r"(?:step)(\d+)", os.path.basename(path))
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    return -1


def _checkpoint_eval_sort_key(item):
    """
    note:
    1) gate_passed
    2) avg_relative_progress_laps
    3) avg_laps
    4) note collision_rate / rear_end_rate
    5) checkpoint step(note)
    """
    r = item.get("result", {}) or {}
    return (
        1 if r.get("gate_passed", False) else 0,
        float(r.get("avg_relative_progress_laps", 0.0)),
        float(r.get("avg_laps", 0.0)),
        -float(r.get("collision_rate", 1.0)),
        -float(r.get("rear_end_rate", 1.0)),
        int(item.get("step", -1)),
    )


def _apply_v11_2_curriculum_overrides(args, curriculum_stage_ref):
    """Apply fixed2 mode overrides. Does NOT force stage to 2.
    Each stage's npc_count/reward_mode is governed by STAGES and STAGE_TRAIN_PROFILES."""
    if str(getattr(args, "v11_2_mode", "fixed2")).lower()!= "fixed2":
        return
    # note stage = 2, note --start-stage notecontrolstage.
    current_npc_count = int(curriculum_stage_ref.get("npc_count", 0))

    # noteNPCnotestagenotereward_mode/npc_modenotefixed2staticnote
    if current_npc_count > 0:
        curriculum_stage_ref["reward_mode"] = "avoid_static"
        # npc_count notestageconfigurationnote, note fixed_npc_count note
        # fixed_npc_count notecontrol NPC controlnote
        curriculum_stage_ref["npc_count"] = current_npc_count
        # npc_mode notestageconfiguration(static / wobble note), note static
        if not curriculum_stage_ref.get("npc_mode"):
            curriculum_stage_ref["npc_mode"] = "static"
        curriculum_stage_ref["npc_speed_min"] = 0.0
        curriculum_stage_ref["npc_speed_max"] = 0.0
        curriculum_stage_ref["p_ego_behind"] = 0.0
        # npc_layout_reset_policy / spawn gaps note StageTrainProfile control, notedefaultnote
        if not curriculum_stage_ref.get("npc_layout_reset_policy"):
            curriculum_stage_ref["npc_layout_reset_policy"] = "agent_only"
        if not curriculum_stage_ref.get("npc_layout_reset_on_collision"):
            curriculum_stage_ref["npc_layout_reset_on_collision"] = False
        if not curriculum_stage_ref.get("npc_layout_reset_on_success"):
            curriculum_stage_ref["npc_layout_reset_on_success"] = False
        if not curriculum_stage_ref.get("spawn_min_gap_sim_ego_npc"):
            curriculum_stage_ref["spawn_min_gap_sim_ego_npc"] = 0.8
        if not curriculum_stage_ref.get("spawn_min_gap_progress_ego_npc"):
            curriculum_stage_ref["spawn_min_gap_progress_ego_npc"] = 40
        curriculum_stage_ref.setdefault("spawn_min_gap_sim_ego_npc_hard_floor", 1.40)
        curriculum_stage_ref.setdefault("spawn_min_gap_progress_ego_npc_hard_floor", 60)
        curriculum_stage_ref.setdefault("npc_persist_across_agent_resets", True)
        curriculum_stage_ref.setdefault("npc_stuck_reset_enable", True)
        if not curriculum_stage_ref.get("spawn_min_gap_sim_npc_npc"):
            curriculum_stage_ref["spawn_min_gap_sim_npc_npc"] = 1.8
        if not curriculum_stage_ref.get("spawn_min_gap_progress_npc_npc"):
            curriculum_stage_ref["spawn_min_gap_progress_npc_npc"] = 36
        if current_npc_count > 1:
            curriculum_stage_ref.setdefault("npc_npc_spawn_hard_check", True)
            curriculum_stage_ref.setdefault("npc_npc_collision_guard_enable", True)
            curriculum_stage_ref.setdefault("npc_npc_contact_reset_enable", False)

    curriculum_stage_ref["success_laps_target"] = int(max(1, int(getattr(args, "fixed_success_laps", 2))))
    curriculum_stage_ref["terminate_on_success_laps"] = False
    # note stage_random_start_enabled=True; note STAGE_TRAIN_PROFILES[stage].random_start_enabled control
    # fixed2notestagenotestageconfiguration, note


def _apply_npc_mode_override(args, curriculum_stage_ref):
    """noteCLInote NPC rowsnote(notestagenote)."""
    if args is None or curriculum_stage_ref is None:
        return
    raw = str(getattr(args, "npc_mode_override", "none") or "none").strip().lower()
    if raw in ("", "none", "auto"):
        return
    mode = raw
    if mode == "slow_policy":
        mode = "slow"
    if mode in ("random_reverse", "chaos_reverse"):
        mode = "chaos"
    if mode not in ("static", "wobble", "slow", "random", "chaos"):
        return
    if int(curriculum_stage_ref.get("npc_count", 0)) <= 0:
        return
    curriculum_stage_ref["npc_mode"] = mode
    if mode in ("random", "chaos"):
        vmin = float(_safe_float(curriculum_stage_ref.get("npc_speed_min", 0.10), 0.10))
        vmax = float(_safe_float(curriculum_stage_ref.get("npc_speed_max", 0.24), 0.24))
        if vmax <= 0.0:
            vmin, vmax = 0.12, 0.30
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        curriculum_stage_ref["npc_speed_min"] = max(0.08, float(vmin))
        curriculum_stage_ref["npc_speed_max"] = max(float(curriculum_stage_ref["npc_speed_min"]), float(vmax))

def eval_checkpoints_mode(args):
    """note4: notecheckpointnotecurrentnotemodel."""
    try:
        from sb3_contrib import RecurrentPPO
    except Exception as e:
        raise RuntimeError(
            "note sb3_contrib(note RecurrentPPO).note: pip install sb3_contrib==1.8.0"
        ) from e

    stage_id = int(args.eval_ckpt_stage or args.start_stage)
    stage_id = max(1, min(stage_id, max(STAGES.keys())))
    gate = STAGE_EVAL_GATES.get(stage_id)
    cli_overrides = _collect_cli_overrides(sys.argv[1:])

    curriculum_stage_ref = {
        'stage': stage_id,
        'reward_mode': STAGES[stage_id].reward_mode,
        'npc_count': STAGES[stage_id].npc_count,
        'npc_mode': STAGES[stage_id].npc_mode,
        'enable_overtake_reward': STAGES[stage_id].enable_overtake_reward,
        'p_ego_behind': STAGES[stage_id].p_ego_behind,
        'npc_speed_min': STAGES[stage_id].npc_speed_range[0],
        'npc_speed_max': STAGES[stage_id].npc_speed_range[1],
        'spawn_max_attempts': 8,
        'success_laps_target': 8,
        'terminate_on_success_laps': True,
        'npc_layout_reset_policy': args.npc_layout_reset_policy,
        'npc_layout_reset_every': args.npc_layout_reset_every,
        'npc_layout_reset_on_collision': args.npc_layout_reset_on_collision,
        'npc_layout_reset_on_success': args.npc_layout_reset_on_success,
        'npc_layout_segments': args.npc_layout_segments,
        'npc_lane_balance_mode': args.npc_lane_balance_mode,
        'spawn_min_gap_sim_ego_npc': args.spawn_min_gap_sim_ego_npc,
        'spawn_min_gap_progress_ego_npc': args.spawn_min_gap_progress_ego_npc,
        'spawn_min_gap_sim_npc_npc': args.spawn_min_gap_sim_npc_npc,
        'spawn_min_gap_progress_npc_npc': args.spawn_min_gap_progress_npc_npc,
        'npc_npc_spawn_hard_check': args.npc_npc_spawn_hard_check,
        'npc_npc_collision_guard_enable': args.npc_npc_collision_guard,
        'npc_npc_collision_guard_dist_sim': args.npc_npc_guard_dist_sim,
        'npc_npc_collision_guard_progress_window': args.npc_npc_guard_progress_window,
        'npc_npc_collision_guard_brake_steps': args.npc_npc_guard_brake_steps,
        'npc_npc_collision_guard_cooldown_steps': args.npc_npc_guard_cooldown_steps,
        'npc_npc_contact_reset_enable': args.npc_npc_contact_reset,
        'npc_npc_contact_dist_sim': args.npc_npc_contact_dist_sim,
        'npc_npc_contact_progress_window': args.npc_npc_contact_progress_window,
        'npc_npc_contact_cooldown_steps': args.npc_npc_contact_cooldown_steps,
        'spawn_min_gap_sim_ego_npc_hard_floor': args.spawn_min_gap_sim_ego_npc_hard_floor,
        'spawn_min_gap_progress_ego_npc_hard_floor': args.spawn_min_gap_progress_ego_npc_hard_floor,
        'npc_persist_across_agent_resets': args.npc_persist_across_agent_resets,
        'npc_stuck_reset_enable': args.npc_stuck_reset_enable,
        'npc_stuck_speed_thresh': args.npc_stuck_speed_thresh,
        'npc_stuck_disp_thresh_sim': args.npc_stuck_disp_thresh_sim,
        'npc_stuck_steps': args.npc_stuck_steps,
        'npc_stuck_cooldown_steps': args.npc_stuck_cooldown_steps,
        'reverse_penalty_steps': args.reverse_penalty_steps,
        'reverse_progress_step_thresh': args.reverse_progress_step_thresh,
        'reverse_progress_dist_thresh': args.reverse_progress_dist_thresh,
        'reverse_onset_penalty': args.reverse_onset_penalty,
        'reverse_streak_penalty_scale': args.reverse_streak_penalty_scale,
        'reverse_backdist_penalty_scale': args.reverse_backdist_penalty_scale,
        'reverse_event_penalty': args.reverse_event_penalty,
        'reverse_gate_enabled': args.enable_reverse_gate,
        'progress_reward_scale': args.progress_reward_scale,
        'progress_backward_penalty_scale': args.progress_backward_penalty_scale,
        'progress_milestone_lap': args.progress_milestone_lap,
        'progress_milestone_reward': args.progress_milestone_reward,
        'spawn_bins': args.spawn_bins,
        'progress_reward_decay_min': args.progress_reward_decay_min,
        'penalty_decay_min': args.penalty_decay_min,
        'random_start_from_stage': args.random_start_from_stage,
        'stage1_cte_reset_limit': args.stage1_cte_reset_limit,
        'npc_side_by_side_start': args.npc_side_by_side_start,
        'npc_front_start': args.npc_front_start,
        'npc_front_offset_nodes': args.npc_front_offset_nodes,
    }
    if not getattr(args, "disable_stage_profiles", False):
        apply_stage_train_profile(stage_id, curriculum_stage_ref, args=args, cli_overrides=cli_overrides, verbose=True)
    _apply_v11_2_curriculum_overrides(args, curriculum_stage_ref)
    _apply_npc_mode_override(args, curriculum_stage_ref)
    stage_id = int(curriculum_stage_ref.get("stage", stage_id))

    # stagenote/noteNPC
    stage_npc_count = int(curriculum_stage_ref.get('npc_count', 0))
    lazy_connect_npcs = bool(stage_npc_count > 0 and int(args.num_npc) > 0)
    saved_num_npc = args.num_npc
    if stage_npc_count <= 0:
        args.num_npc = 0
    vec_env, wrapper, npcs, track_cache = create_generated_env_and_npcs(
        args, curriculum_stage_ref, dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK),
        lazy_connect_npcs=lazy_connect_npcs,
    )
    args.num_npc = saved_num_npc
    wrapper.dist_scale = build_dist_scale_profile(track_cache, "generated_track")

    pattern = args.eval_ckpt_glob
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise RuntimeError(f"notecheckpoint: {pattern}")
    if args.eval_ckpt_limit and int(args.eval_ckpt_limit) > 0:
        # defaultnotestepnoteNnote, note"notecurrentnote"note
        ckpts = sorted(ckpts, key=_extract_step_from_ckpt_path)[-int(args.eval_ckpt_limit):]

    print(f"\n🔎 notecheckpoint | stage={stage_id} ({STAGES[stage_id].name})")
    print(f"   files={len(ckpts)} pattern={pattern}")
    if gate is not None:
        print(f"   gate: min_prog={gate.min_relative_progress_laps} lap>={gate.require_lap_count_at_least} "
              f"max_collision={gate.max_collisions} passes={gate.consecutive_passes_required} (note)")

    rows = []
    try:
        for i, path in enumerate(ckpts, 1):
            print(f"\n[{i}/{len(ckpts)}] note: {os.path.basename(path)}")
            model = RecurrentPPO.load(path, env=vec_env)
            early_stop_laps = max(1, int(gate.require_lap_count_at_least)) if gate else int(args.eval_success_laps)
            result = evaluate_stage_on_same_env(
                model, wrapper, curriculum_stage_ref,
                n_episodes=(int(gate.eval_episodes) if gate else int(args.eval_episodes)),
                success_laps=early_stop_laps,
                eval_gate=gate,
                eval_max_steps=int(args.eval_max_steps),
                stop_on_laps=False,
            )
            row = {
                "path": path,
                "step": _extract_step_from_ckpt_path(path),
                "result": result,
            }
            rows.append(row)
            print(
                f"   gate_passed={result.get('gate_passed', False)} "
                f"avg_rel_prog={result.get('avg_relative_progress_laps', 0):.2f} "
                f"avg_laps={result.get('avg_laps', 0):.2f} "
                f"collision={result.get('collision_rate', 0):.0%} "
                f"rear_end={result.get('rear_end_rate', 0):.0%} "
                f"unsafe_follow={result.get('avg_unsafe_follow_ratio', 0):.2f}"
            )
    finally:
        cleanup_envs(vec_env, npcs)

    rows_sorted = sorted(rows, key=_checkpoint_eval_sort_key, reverse=True)
    print("\n🏆 Checkpoint Ranking (Top 10)")
    for rank, row in enumerate(rows_sorted[:10], 1):
        r = row["result"]
        print(
            f"{rank:2d}. step={row['step']:>7} gate={str(r.get('gate_passed', False)):<5} "
            f"rel_prog={r.get('avg_relative_progress_laps', 0):.2f} laps={r.get('avg_laps', 0):.2f} "
            f"col={r.get('collision_rate', 0):.0%} rear={r.get('rear_end_rate', 0):.0%} "
            f"| {os.path.basename(row['path'])}"
        )

    best = rows_sorted[0]
    best_r = best["result"]
    print("\nPASS notecheckpoint")
    print(json.dumps({
        "path": best["path"],
        "step": best["step"],
        "stage": stage_id,
        "gate_passed": best_r.get("gate_passed", False),
        "avg_relative_progress_laps": best_r.get("avg_relative_progress_laps", 0.0),
        "avg_laps": best_r.get("avg_laps", 0.0),
        "collision_rate": best_r.get("collision_rate", 0.0),
        "rear_end_rate": best_r.get("rear_end_rate", 0.0),
    }, indent=2, ensure_ascii=False))


def force_reload_scene(handler, scene_name="generated_track", load_timeout=25.0,
                       exit_wait=1.5, settle_wait=1.0, max_attempts=2):
    """
    note: note exit_scene note load_scene, notedefaultnote.
    notesucceeded(handler.loaded=True).
    """
    if handler is None:
        return False

    target = str(scene_name)
    timeout = max(1.0, float(load_timeout))
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        print(f"🗺️ note attempt={attempt}: exit_scene -> load_scene({target})")

        try:
            handler.loaded = False
        except Exception:
            pass

        try:
            handler.blocking_send({"msg_type": "exit_scene"})
            time.sleep(max(0.2, float(exit_wait)))
        except Exception as e:
            print(f"⚠️ exit_scene notefailed: {e}")

        try:
            handler.loaded = False
        except Exception:
            pass

        try:
            handler.blocking_send({"msg_type": "load_scene", "scene_name": target})
        except Exception as e:
            print(f"⚠️ load_scene({target}) notefailed: {e}")
            continue

        deadline = time.time() + timeout
        while time.time() < deadline:
            if bool(getattr(handler, "loaded", False)):
                time.sleep(max(0.1, float(settle_wait)))
                print(f"PASS note: {target}")
                return True
            time.sleep(0.2)

        print(f"⚠️ load_scene({target}) note {timeout:.1f}s, note")

    return bool(getattr(handler, "loaded", False))


def create_generated_env_and_npcs(args, curriculum_stage_ref, dist_scale_profile, track_cache=None, lazy_connect_npcs=False):
    """note generated_track note(note)."""
    track_cache = track_cache or TrackNodeCache()
    body_rgb = tuple(int(np.clip(v, 0, 255)) for v in args.body_rgb)
    conf = {
        "host": args.host,
        "port": args.port,
        "body_style": args.body_style,
        "body_rgb": body_rgb,
        "car_name": args.car_name,
        "racer_name": "v11_2_learner",
        "country": "US",
        "bio": "",
        "guid": f"v11_2_learner_{int(time.time())}",
        "font_size": 50,
        "level": "generated_track",
        "max_cte": args.max_cte,
        # v10.2: note(note), notewrappernoterewardnotecontrol
        "throttle_min": -1.0,
        "throttle_max": 1.0,
        "cam_resolution": (120, 160, 3),
        "cam_encode": "JPG",
        "log_level": 20,
    }
    if args.enable_sim_lidar:
        conf["lidar_config"] = {
            "deg_per_sweep_inc": float(args.lidar_deg_per_sweep_inc),
            "deg_ang_down": float(args.lidar_deg_ang_down),
            "deg_ang_delta": float(args.lidar_deg_ang_delta),
            "num_sweeps_levels": int(args.lidar_num_sweeps_levels),
            "max_range": float(args.lidar_max_range),
            "noise": float(args.lidar_noise),
            "offset_x": float(args.lidar_offset_x),
            "offset_y": float(args.lidar_offset_y),
            "offset_z": float(args.lidar_offset_z),
            "rot_x": float(args.lidar_rot_x),
        }
    if args.exe_path:
        conf['exe_path'] = args.exe_path

    env = gym.make("donkey-generated-track-v0", conf=conf)
    time.sleep(1.5)

    # note + fine_track
    handler = env.viewer.handler
    if bool(getattr(args, "force_scene_reload", True)):
        ok = force_reload_scene(
            handler,
            scene_name="generated_track",
            load_timeout=float(getattr(args, "scene_load_timeout", 25.0)),
            max_attempts=2,
        )
        if not ok:
            raise RuntimeError("notefailed: note generated_track(noterows exit_scene -> load_scene)")
    track_cache.query_nodes(handler, "generated_track", total_nodes=108)
    if not track_cache.fine_track.get("generated_track"):
        track_cache._build_fine_track("generated_track")

    # note NPC(note args.num_npc, notestagecontrol)
    npc_controllers = []
    npc_colors = [(255, 100, 100), (100, 255, 100), (255, 255, 100), (100, 100, 255), (255, 165, 0), (180, 0, 255)]
    for i in range(int(args.num_npc)):
        npc = NPCController(
            npc_id=i + 1,
            host=args.host,
            port=args.port,
            scene='generated_track',
            track_cache=track_cache,
        )
        body_rgb = npc_colors[i % len(npc_colors)]
        npc._desired_body_rgb = body_rgb
        if lazy_connect_npcs:
            # note: note controller note, notestagenote wrapper.reset() note
            npc_controllers.append(npc)
        else:
            if npc.connect(body_rgb=body_rgb):
                npc.set_mode('static', 0.0)
                npc.set_position_node_coords(0.0, -500.0, 0.0, 0.0, 0.0, 0.0, 1.0)
                npc_controllers.append(npc)
            time.sleep(0.6)

    wrapper = GeneratedTrackV11_2Wrapper(
        env,
        npc_controllers=npc_controllers,
        track_cache=track_cache,
        dist_scale_profile=dist_scale_profile,
        curriculum_stage_ref=curriculum_stage_ref,
        use_random_start=args.random_start,
        max_episode_steps=args.max_episode_steps,
        enable_dr=args.enable_dr,
        max_throttle=args.max_throttle,
        delta_max=args.delta_max,
        enable_lpf=args.enable_lpf,
        beta=args.beta,
        w_d=args.w_d,
        w_dd=args.w_dd,
        w_sat=args.w_sat,
        spawn_debug=(not args.no_spawn_debug),
        spawn_jitter_s_sim=args.spawn_jitter_s_sim,
        spawn_jitter_d_sim=args.spawn_jitter_d_sim,
        spawn_yaw_jitter_deg=args.spawn_yaw_jitter_deg,
        spawn_verify_tol_sim=args.spawn_verify_tol_sim,
        spawn_verify_cte_threshold=args.spawn_verify_cte_threshold,
        npc_layout_debug=args.npc_layout_debug,
        npc_lane_balance_mode=args.npc_lane_balance_mode,
        npc_layout_segments=args.npc_layout_segments,
        npc_lane_offset_left_sim=args.npc_lane_offset_left_sim,
        npc_lane_offset_right_sim=args.npc_lane_offset_right_sim,
        npc_lane_offset_center_sim=args.npc_lane_offset_center_sim,
        npc_lane_jitter_sim=args.npc_lane_jitter_sim,
        npc_npc_collision_guard_enable=args.npc_npc_collision_guard,
        npc_npc_collision_guard_dist_sim=args.npc_npc_guard_dist_sim,
        npc_npc_collision_guard_progress_window=args.npc_npc_guard_progress_window,
        npc_npc_collision_guard_brake_steps=args.npc_npc_guard_brake_steps,
        npc_npc_collision_guard_cooldown_steps=args.npc_npc_guard_cooldown_steps,
        npc_npc_contact_reset_enable=args.npc_npc_contact_reset,
        npc_npc_contact_dist_sim=args.npc_npc_contact_dist_sim,
        npc_npc_contact_progress_window=args.npc_npc_contact_progress_window,
        npc_npc_contact_cooldown_steps=args.npc_npc_contact_cooldown_steps,
        npc_npc_spawn_hard_check=args.npc_npc_spawn_hard_check,
        spawn_min_gap_sim_ego_npc_hard_floor=args.spawn_min_gap_sim_ego_npc_hard_floor,
        spawn_min_gap_progress_ego_npc_hard_floor=args.spawn_min_gap_progress_ego_npc_hard_floor,
        npc_persist_across_agent_resets=args.npc_persist_across_agent_resets,
        npc_stuck_reset_enable=args.npc_stuck_reset_enable,
        npc_stuck_speed_thresh=args.npc_stuck_speed_thresh,
        npc_stuck_disp_thresh_sim=args.npc_stuck_disp_thresh_sim,
        npc_stuck_steps=args.npc_stuck_steps,
        npc_stuck_cooldown_steps=args.npc_stuck_cooldown_steps,
        spawn_debug_violations_limit=args.spawn_debug_violations_limit,
        lazy_connect_npcs=lazy_connect_npcs,
        v_ref_min=args.v_ref_min,
        v_ref_max=args.v_ref_max,
        kappa_ref_max=args.kappa_ref_max,
        v_ref_rate_max=args.v_ref_rate_max,
        kappa_ref_rate_max=args.kappa_ref_rate_max,
        steer_ff_headroom=args.steer_ff_headroom,
        steer_softsat_gain=args.steer_softsat_gain,
        steer_slew_rate_max=args.steer_slew_rate_max,
        ctrl_dt_min=args.ctrl_dt_min,
        ctrl_dt_max=args.ctrl_dt_max,
        ctrl_dt_fallback=args.ctrl_dt_fallback,
        gyro_lpf_tau=args.gyro_lpf_tau,
        v_fb_min=args.v_fb_min,
        yaw_kp=args.yaw_kp,
        yaw_kd=args.yaw_kd,
        yaw_ff_k=args.yaw_ff_k,
        speed_kp=args.speed_kp,
        speed_ki=args.speed_ki,
        speed_kaw=args.speed_kaw,
        throttle_brake_min=args.throttle_brake_min,
        v_stop_eps=args.v_stop_eps,
        speed_i_leak=args.speed_i_leak,
        speed_i_fwd_max=args.speed_i_fwd_max,
        speed_i_brake_max=args.speed_i_brake_max,
        unit_calibrate=args.unit_calibrate,
        unit_calib_steps=args.unit_calib_steps,
        unit_calib_ratio_min=args.unit_calib_ratio_min,
        unit_calib_strict=args.unit_calib_strict,
        sign_check_min_steer=args.sign_check_min_steer,
        sign_check_min_speed=args.sign_check_min_speed,
        sign_check_max_lag=args.sign_check_max_lag,
        sign_check_min_corr=args.sign_check_min_corr,
        sign_check_min_samples=args.sign_check_min_samples,
        startup_force_throttle_steps=args.startup_force_throttle_steps,
        startup_force_throttle=args.startup_force_throttle,
        no_motion_penalty_start_steps=args.no_motion_penalty_start_steps,
        no_motion_penalty_speed_thresh=args.no_motion_penalty_speed_thresh,
        no_motion_penalty_progress_thresh=args.no_motion_penalty_progress_thresh,
        no_motion_penalty_per_step=args.no_motion_penalty_per_step,
        progress_local_window=args.progress_local_window,
        progress_local_recover_dist=args.progress_local_recover_dist,
        progress_idle_freeze_speed=args.progress_idle_freeze_speed,
        progress_idle_freeze_dfi_abs=args.progress_idle_freeze_dfi_abs,
        enable_progress_heading_filter=args.enable_progress_heading_filter,
        progress_heading_dot_min=args.progress_heading_dot_min,
        v11_2_mode=args.v11_2_mode,
        manual_width_profile=args.manual_width_profile,
        fixed_npc_count=args.fixed_npc_count,
        fixed_success_progress=args.fixed_success_progress,
        fixed_success_laps=args.fixed_success_laps,
        success_no_hit_steps=args.success_no_hit_steps,
        success_no_offtrack_seconds=args.success_no_offtrack_seconds,
        success_cte_norm_margin=args.success_cte_norm_margin,
        success_min_steps=args.success_min_steps,
        success_progress_int_laps=args.success_progress_int_laps,
        success_use_milestone_fallback=args.success_use_milestone_fallback,
        stall_penalty_enable=args.stall_penalty_enable,
        stall_speed_thresh=args.stall_speed_thresh,
        stall_progress_step_thresh=args.stall_progress_step_thresh,
        stall_penalty_per_step=args.stall_penalty_per_step,
        stall_penalty_warmup_steps=args.stall_penalty_warmup_steps,
        spawn_local_window_m=args.spawn_local_window_m,
        spawn_local_window_min_idx=args.spawn_local_window_min_idx,
        spawn_anchor_window_m=args.spawn_anchor_window_m,
        spawn_anchor_window_min_idx=args.spawn_anchor_window_min_idx,
        spawn_safe_anchor_pool=args.spawn_safe_anchor_pool,
        spawn_kappa_max=args.spawn_kappa_max,
        spawn_inside_margin_ratio=args.spawn_inside_margin_ratio,
        spawn_inside_max_attempts=args.spawn_inside_max_attempts,
        npc_min_track_width_ratio=args.npc_min_track_width_ratio,
        npc_min_track_width_abs_min=args.npc_min_track_width_abs_min,
        layout_age_hard_cap=args.layout_age_hard_cap,
        episode_summary_jsonl=args.episode_summary_jsonl,
        episode_summary_csv=args.episode_summary_csv,
        event_trace_jsonl=args.event_trace_jsonl,
        event_trace_window_steps=args.event_trace_window_steps,
        event_trace_max_per_episode=args.event_trace_max_per_episode,
        event_dt_spike_thresh=args.event_dt_spike_thresh,
        event_sat_spike_cooldown_steps=args.event_sat_spike_cooldown_steps,
        event_generic_cooldown_steps=args.event_generic_cooldown_steps,
        event_reward_spike_abs=args.event_reward_spike_abs,
        event_reward_gap_spike_abs=args.event_reward_gap_spike_abs,
        event_progress_jump_fi=args.event_progress_jump_fi,
        event_progress_clip_ratio_high=args.event_progress_clip_ratio_high,
        event_spawn_near_dist=args.event_spawn_near_dist,
        event_periodic_sample_steps=args.event_periodic_sample_steps,
    )
    if lazy_connect_npcs:
        wrapper._pending_npc_connect = [{"npc": n, "body_rgb": getattr(n, "_desired_body_rgb", (255, 100, 100))}
                                        for n in npc_controllers]
    monitored = Monitor(wrapper, filename=None, allow_early_resets=True)
    vec_env = DummyVecEnv([lambda: monitored])
    return vec_env, wrapper, npc_controllers, track_cache


def calibrate_only(args):
    """note1: note."""
    _apply_global_seeds(getattr(args, "seed", 42))
    curriculum_stage_ref = {
        'stage': 1, 'reward_mode': 'drive_only', 'npc_count': 0,
        'npc_mode': 'offtrack', 'p_ego_behind': 0.0,
        'npc_speed_min': 0.0, 'npc_speed_max': 0.0,
        'success_laps_target': 8, 'terminate_on_success_laps': True,
        'npc_layout_reset_policy': args.npc_layout_reset_policy,
        'npc_layout_reset_every': args.npc_layout_reset_every,
        'npc_layout_reset_on_collision': args.npc_layout_reset_on_collision,
        'npc_layout_reset_on_success': args.npc_layout_reset_on_success,
        'npc_layout_segments': args.npc_layout_segments,
        'npc_lane_balance_mode': args.npc_lane_balance_mode,
    }
    saved_num_npc = args.num_npc
    try:
        args.num_npc = 0  # v10.1: calibrate defaultnoteNPC
        vec_env, wrapper, npcs, track_cache = create_generated_env_and_npcs(
            args, curriculum_stage_ref, dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK)
        )
    finally:
        args.num_npc = saved_num_npc
    profile = build_dist_scale_profile(track_cache, "generated_track")
    print("\n📏 DIST_SCALE_PROFILE_GENERATED_TRACK")
    print(json.dumps(profile, indent=2, ensure_ascii=False))
    if args.scale_profile_out:
        with open(args.scale_profile_out, 'w') as f:
            json.dump(profile, f, indent=2)
        print(f"saved notesavenote: {args.scale_profile_out}")
    cleanup_envs(vec_env, npcs)


def reset_stress_mode(args):
    """note2: note reset note, note spawn stablenote."""
    _apply_global_seeds(getattr(args, "seed", 42))
    profile = dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK)
    curriculum_stage_ref = {
        'stage': int(args.start_stage),
        'reward_mode': STAGES.get(int(args.start_stage), STAGES[1]).reward_mode,
        'npc_count': STAGES.get(int(args.start_stage), STAGES[1]).npc_count,
        'npc_mode': STAGES.get(int(args.start_stage), STAGES[1]).npc_mode,
        'p_ego_behind': STAGES.get(int(args.start_stage), STAGES[1]).p_ego_behind,
        'npc_speed_min': STAGES.get(int(args.start_stage), STAGES[1]).npc_speed_range[0],
        'npc_speed_max': STAGES.get(int(args.start_stage), STAGES[1]).npc_speed_range[1],
        'spawn_max_attempts': 8,
        'success_laps_target': 8, 'terminate_on_success_laps': True,
        'npc_layout_reset_policy': ('agent_only' if args.stress_layout_mode == 'agent-only'
                                    else ('every_episode' if args.stress_layout_mode == 'full' else args.npc_layout_reset_policy)),
        'npc_layout_reset_every': args.npc_layout_reset_every,
        'npc_layout_reset_on_collision': args.npc_layout_reset_on_collision,
        'npc_layout_reset_on_success': args.npc_layout_reset_on_success,
        'npc_layout_segments': args.npc_layout_segments,
        'npc_lane_balance_mode': args.npc_lane_balance_mode,
        'spawn_min_gap_sim_ego_npc': args.spawn_min_gap_sim_ego_npc,
        'spawn_min_gap_progress_ego_npc': args.spawn_min_gap_progress_ego_npc,
        'spawn_min_gap_sim_npc_npc': args.spawn_min_gap_sim_npc_npc,
        'spawn_min_gap_progress_npc_npc': args.spawn_min_gap_progress_npc_npc,
        'npc_npc_spawn_hard_check': args.npc_npc_spawn_hard_check,
        'npc_npc_collision_guard_enable': args.npc_npc_collision_guard,
        'npc_npc_collision_guard_dist_sim': args.npc_npc_guard_dist_sim,
        'npc_npc_collision_guard_progress_window': args.npc_npc_guard_progress_window,
        'npc_npc_collision_guard_brake_steps': args.npc_npc_guard_brake_steps,
        'npc_npc_collision_guard_cooldown_steps': args.npc_npc_guard_cooldown_steps,
        'npc_npc_contact_reset_enable': args.npc_npc_contact_reset,
        'npc_npc_contact_dist_sim': args.npc_npc_contact_dist_sim,
        'npc_npc_contact_progress_window': args.npc_npc_contact_progress_window,
        'npc_npc_contact_cooldown_steps': args.npc_npc_contact_cooldown_steps,
        'spawn_min_gap_sim_ego_npc_hard_floor': args.spawn_min_gap_sim_ego_npc_hard_floor,
        'spawn_min_gap_progress_ego_npc_hard_floor': args.spawn_min_gap_progress_ego_npc_hard_floor,
        'npc_persist_across_agent_resets': args.npc_persist_across_agent_resets,
        'npc_stuck_reset_enable': args.npc_stuck_reset_enable,
        'npc_stuck_speed_thresh': args.npc_stuck_speed_thresh,
        'npc_stuck_disp_thresh_sim': args.npc_stuck_disp_thresh_sim,
        'npc_stuck_steps': args.npc_stuck_steps,
        'npc_stuck_cooldown_steps': args.npc_stuck_cooldown_steps,
        'reverse_penalty_steps': args.reverse_penalty_steps,
        'reverse_progress_step_thresh': args.reverse_progress_step_thresh,
        'reverse_progress_dist_thresh': args.reverse_progress_dist_thresh,
        'reverse_onset_penalty': args.reverse_onset_penalty,
        'reverse_streak_penalty_scale': args.reverse_streak_penalty_scale,
        'reverse_backdist_penalty_scale': args.reverse_backdist_penalty_scale,
        'reverse_event_penalty': args.reverse_event_penalty,
        'reverse_gate_enabled': args.enable_reverse_gate,
        'progress_reward_scale': args.progress_reward_scale,
        'progress_backward_penalty_scale': args.progress_backward_penalty_scale,
        'progress_milestone_lap': args.progress_milestone_lap,
        'progress_milestone_reward': args.progress_milestone_reward,
        'spawn_bins': args.spawn_bins,
        'progress_reward_decay_min': args.progress_reward_decay_min,
        'penalty_decay_min': args.penalty_decay_min,
        'random_start_from_stage': args.random_start_from_stage,
        'stage1_cte_reset_limit': args.stage1_cte_reset_limit,
        'npc_side_by_side_start': args.npc_side_by_side_start,
        'npc_front_start': args.npc_front_start,
        'npc_front_offset_nodes': args.npc_front_offset_nodes,
    }
    _apply_v11_2_curriculum_overrides(args, curriculum_stage_ref)
    _apply_npc_mode_override(args, curriculum_stage_ref)
    # Stage1 note NPC: note/note, note"noteNPC"
    saved_num_npc = args.num_npc
    try:
        need_npc = _max_npc_count_from_stage(int(curriculum_stage_ref.get('stage', args.start_stage)))
        args.num_npc = min(int(saved_num_npc), int(need_npc))
        vec_env, wrapper, npcs, track_cache = create_generated_env_and_npcs(args, curriculum_stage_ref, profile)
    finally:
        args.num_npc = saved_num_npc
    wrapper.dist_scale = build_dist_scale_profile(track_cache, "generated_track")
    stats = wrapper.reset_stress_test(n_resets=args.stress_resets, rollout_steps=args.stress_rollout_steps)
    print("\n🧪 Reset Stress Result")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    cleanup_envs(vec_env, npcs)

def export_spawn_table_mode(args):
    """note5: note manual width profile notetracknote."""
    sampler = ManualWidthSpawnSampler(args.manual_width_profile, "generated_track")
    if not sampler.loaded:
        raise RuntimeError(f"manual width profile notefailed: {sampler.error}")

    lane_fracs = _parse_float_list(
        getattr(args, "spawn_table_lane_fracs", ""),
        [0.12, 0.25, 0.38, 0.50, 0.62, 0.75, 0.88],
    )
    samples = sampler.export_intrack_samples(
        y_tel=float(args.spawn_table_y_tel),
        node_count=int(args.spawn_table_node_count),
        lane_fracs=lane_fracs,
        step_idx=int(args.spawn_table_step_idx),
        margin_ratio=float(args.spawn_table_margin_ratio),
        kappa_max=(None if (args.spawn_table_kappa_max is None or float(args.spawn_table_kappa_max) <= 0.0)
                   else float(args.spawn_table_kappa_max)),
    )

    x_vals = [float(p["tel"][0]) for p in samples] if samples else []
    z_vals = [float(p["tel"][2]) for p in samples] if samples else []
    summary = {
        "scene": "generated_track",
        "profile_path": os.path.abspath(args.manual_width_profile),
        "generated_at_unix": int(time.time()),
        "coord_scale": float(sampler.coord_scale),
        "fine_points": int(len(sampler.fine_track)),
        "width_median_sim": float(sampler.width_median_sim),
        "bounds": sampler.bounds_summary(),
        "sampling": {
            "step_idx": int(args.spawn_table_step_idx),
            "margin_ratio": float(args.spawn_table_margin_ratio),
            "lane_fracs": [float(v) for v in lane_fracs],
            "kappa_max": (None if (args.spawn_table_kappa_max is None or float(args.spawn_table_kappa_max) <= 0.0)
                          else float(args.spawn_table_kappa_max)),
            "y_tel": float(args.spawn_table_y_tel),
            "node_count": int(args.spawn_table_node_count),
        },
        "exported_point_count": int(len(samples)),
        "sample_bounds": {
            "x_min": float(min(x_vals)) if x_vals else None,
            "x_max": float(max(x_vals)) if x_vals else None,
            "z_min": float(min(z_vals)) if z_vals else None,
            "z_max": float(max(z_vals)) if z_vals else None,
        },
    }

    payload = dict(summary)
    payload["points"] = samples
    _write_json(args.spawn_table_out, payload)

    if args.spawn_table_csv:
        csv_path = os.path.abspath(args.spawn_table_csv)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "fine_idx", "node_idx", "lane_ratio", "lane_side",
                "x_tel", "y_tel", "z_tel", "yaw_deg", "dn_offset_sim", "kappa", "width_sim",
            ])
            for p in samples:
                tel = p.get("tel", [0.0, 0.0, 0.0])
                w.writerow([
                    int(_safe_float(p.get("fine_idx", 0), 0)),
                    int(_safe_float(p.get("node_idx", 0), 0)),
                    float(_safe_float(p.get("lane_ratio", 0.5), 0.5)),
                    str(p.get("lane_side", "center")),
                    float(_safe_float(tel[0] if len(tel) > 0 else 0.0, 0.0)),
                    float(_safe_float(tel[1] if len(tel) > 1 else 0.0, 0.0)),
                    float(_safe_float(tel[2] if len(tel) > 2 else 0.0, 0.0)),
                    float(_safe_float(p.get("yaw_deg", 0.0), 0.0)),
                    float(_safe_float(p.get("dn_offset_sim", 0.0), 0.0)),
                    float(_safe_float(p.get("kappa", 0.0), 0.0)),
                    float(_safe_float(p.get("width_sim", 0.0), 0.0)),
                ])
        print(f"🧾 tracknoteCSVnote: {csv_path}")

    print("\n🗺️ Spawn Table Export Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


class V11ControlTBCallback(BaseCallback):
    """Collect control-layer diagnostics from env info and write to TensorBoard."""

    def __init__(self, log_every=1000, verbose=0):
        super().__init__(verbose=verbose)
        self.log_every = max(50, int(log_every))
        self.last_log_step = 0
        self.prev_steer = {}
        self.prev_kappa = {}
        self._reset_acc()

    def _reset_acc(self):
        self.acc = {
            "steer_abs": [],
            "dsteer_abs": [],
            "kappa_abs": [],
            "dkappa_abs": [],
            "ff_abs": [],
            "fb_abs": [],
            "sat": [],
            "yaw_err_abs": [],
            "dt": [],
            "dt_raw": [],
            "headroom_clip": [],
            "near_stop": [],
            "success_guard_passed": [],
            "stall_active": [],
            "spawn_fallback": [],
            "layout_refresh_any": [],
            "layout_refresh_on_success": [],
            "layout_refresh_age_cap": [],
            "episode_success": [],
        }

    def _append(self, key, value):
        v = _safe_float(value, np.nan)
        if np.isfinite(v):
            self.acc[key].append(float(v))

    def _record_mean(self, key, tag):
        vals = self.acc.get(key, [])
        if vals:
            self.logger.record(tag, float(np.mean(vals)))

    def _record_p95(self, key, tag):
        vals = self.acc.get(key, [])
        if vals:
            self.logger.record(tag, float(np.percentile(vals, 95)))

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if infos is None:
            infos = []

        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            steer = _safe_float(info.get("ctrl_steer_exec"), np.nan)
            kappa = _safe_float(info.get("pilot_kappa_ref"), np.nan)
            ff = _safe_float(info.get("ctrl_steer_ff"), np.nan)
            fb = _safe_float(info.get("ctrl_steer_fb"), np.nan)
            yaw_err = _safe_float(info.get("ctrl_yaw_rate_err"), np.nan)
            dt = _safe_float(info.get("ctrl_dt"), np.nan)
            dt_raw = _safe_float(info.get("ctrl_dt_raw"), np.nan)

            self._append("steer_abs", abs(steer))
            self._append("kappa_abs", abs(kappa))
            self._append("ff_abs", abs(ff))
            self._append("fb_abs", abs(fb))
            self._append("yaw_err_abs", abs(yaw_err))
            self._append("dt", dt)
            self._append("dt_raw", dt_raw)
            self._append("sat", 1.0 if bool(info.get("ctrl_steer_sat", False)) else 0.0)
            self._append("headroom_clip", 1.0 if bool(info.get("ctrl_ff_headroom_clipped", False)) else 0.0)
            self._append("near_stop", 1.0 if bool(info.get("ctrl_near_stop_clamp", False)) else 0.0)
            self._append("success_guard_passed", 1.0 if bool(info.get("success_guard_passed", False)) else 0.0)
            self._append("stall_active", 1.0 if bool(info.get("stall_penalty_active", False)) else 0.0)
            self._append("spawn_fallback", 1.0 if str(info.get("spawn_source", "")) == "fallback_centerline_safe" else 0.0)
            if "npc_layout_reused" in info:
                self._append("layout_refresh_any", 1.0 if (not bool(info.get("npc_layout_reused", True))) else 0.0)
            rr = str(info.get("npc_layout_refresh_reason", ""))
            self._append("layout_refresh_on_success", 1.0 if rr == "success_avoidance_2laps" else 0.0)
            self._append("layout_refresh_age_cap", 1.0 if rr == "age_cap" else 0.0)

            if np.isfinite(steer):
                prev_steer = self.prev_steer.get(env_idx, None)
                if prev_steer is not None:
                    self.acc["dsteer_abs"].append(abs(float(steer) - float(prev_steer)))
                self.prev_steer[env_idx] = float(steer)
            if np.isfinite(kappa):
                prev_kappa = self.prev_kappa.get(env_idx, None)
                if prev_kappa is not None:
                    self.acc["dkappa_abs"].append(abs(float(kappa) - float(prev_kappa)))
                self.prev_kappa[env_idx] = float(kappa)

            if dones is not None and env_idx < len(dones) and bool(dones[env_idx]):
                self._append("episode_success", 1.0 if bool(info.get("episode_success_2laps", False)) else 0.0)
                self.prev_steer.pop(env_idx, None)
                self.prev_kappa.pop(env_idx, None)

        if (self.num_timesteps - self.last_log_step) >= self.log_every:
            self._record_mean("steer_abs", "ctrl_tb/steer_abs_mean")
            self._record_p95("steer_abs", "ctrl_tb/steer_abs_p95")
            self._record_mean("dsteer_abs", "ctrl_tb/dsteer_abs_mean")
            self._record_p95("dsteer_abs", "ctrl_tb/dsteer_abs_p95")
            self._record_mean("kappa_abs", "ctrl_tb/kappa_abs_mean")
            self._record_p95("kappa_abs", "ctrl_tb/kappa_abs_p95")
            self._record_mean("dkappa_abs", "ctrl_tb/dkappa_abs_mean")
            self._record_p95("dkappa_abs", "ctrl_tb/dkappa_abs_p95")
            self._record_mean("sat", "ctrl_tb/steer_sat_ratio")
            self._record_mean("yaw_err_abs", "ctrl_tb/yaw_rate_err_abs_mean")
            self._record_p95("yaw_err_abs", "ctrl_tb/yaw_rate_err_abs_p95")
            self._record_mean("dt", "ctrl_tb/dt_mean")
            self._record_p95("dt_raw", "ctrl_tb/dt_raw_p95")
            self._record_mean("headroom_clip", "ctrl_tb/ff_headroom_clip_ratio")
            self._record_mean("near_stop", "ctrl_tb/near_stop_clamp_ratio")
            self._record_mean("ff_abs", "ctrl_tb/steer_ff_abs_mean")
            self._record_mean("fb_abs", "ctrl_tb/steer_fb_abs_mean")
            if self.acc["ff_abs"] and self.acc["fb_abs"]:
                ff_abs_mean = float(np.mean(self.acc["ff_abs"]))
                fb_abs_mean = float(np.mean(self.acc["fb_abs"]))
                self.logger.record("ctrl_tb/fb_over_ff_abs_ratio", fb_abs_mean / max(ff_abs_mean, 1e-6))
            self.logger.record("ctrl_tb/window_size", float(len(self.acc["steer_abs"])))

            self._record_mean("success_guard_passed", "fixed2/success_guard_pass_rate")
            self._record_mean("episode_success", "fixed2/episode_success_rate")
            self._record_mean("stall_active", "fixed2/stall_ratio")
            self._record_mean("spawn_fallback", "fixed2/spawn_fallback_ratio")
            self._record_mean("layout_refresh_any", "fixed2/layout_refresh_ratio")
            self._record_mean("layout_refresh_on_success", "fixed2/layout_refresh_success_ratio")
            self._record_mean("layout_refresh_age_cap", "fixed2/layout_refresh_age_cap_ratio")

            self.last_log_step = int(self.num_timesteps)
            self._reset_acc()
        return True


def train_mode(args):
    """note3: RecurrentPPO(LSTM) notetraining + notestagenote."""
    try:
        from sb3_contrib import RecurrentPPO
    except Exception as e:
        raise RuntimeError(
            "note sb3_contrib(note RecurrentPPO).note: pip install sb3_contrib==1.8.0"
        ) from e

    seed = _apply_global_seeds(getattr(args, "seed", 42))
    print(f"🎲 note: {seed}")

    stage_id = int(args.start_stage)
    stage_id = max(1, min(stage_id, max(STAGES.keys())))
    cli_overrides = _collect_cli_overrides(sys.argv[1:])
    curriculum_stage_ref = {
        'stage': stage_id,
        'reward_mode': STAGES[stage_id].reward_mode,
        'npc_count': STAGES[stage_id].npc_count,
        'npc_mode': STAGES[stage_id].npc_mode,
        'enable_overtake_reward': STAGES[stage_id].enable_overtake_reward,
        'p_ego_behind': STAGES[stage_id].p_ego_behind,
        'npc_speed_min': STAGES[stage_id].npc_speed_range[0],
        'npc_speed_max': STAGES[stage_id].npc_speed_range[1],
        'spawn_max_attempts': 8,
        'success_laps_target': 8,
        'terminate_on_success_laps': True,
        'npc_layout_reset_policy': args.npc_layout_reset_policy,
        'npc_layout_reset_every': args.npc_layout_reset_every,
        'npc_layout_reset_on_collision': args.npc_layout_reset_on_collision,
        'npc_layout_reset_on_success': args.npc_layout_reset_on_success,
        'npc_layout_segments': args.npc_layout_segments,
        'npc_lane_balance_mode': args.npc_lane_balance_mode,
        'spawn_min_gap_sim_ego_npc': args.spawn_min_gap_sim_ego_npc,
        'spawn_min_gap_progress_ego_npc': args.spawn_min_gap_progress_ego_npc,
        'spawn_min_gap_sim_npc_npc': args.spawn_min_gap_sim_npc_npc,
        'spawn_min_gap_progress_npc_npc': args.spawn_min_gap_progress_npc_npc,
        'npc_npc_spawn_hard_check': args.npc_npc_spawn_hard_check,
        'npc_npc_collision_guard_enable': args.npc_npc_collision_guard,
        'npc_npc_collision_guard_dist_sim': args.npc_npc_guard_dist_sim,
        'npc_npc_collision_guard_progress_window': args.npc_npc_guard_progress_window,
        'npc_npc_collision_guard_brake_steps': args.npc_npc_guard_brake_steps,
        'npc_npc_collision_guard_cooldown_steps': args.npc_npc_guard_cooldown_steps,
        'npc_npc_contact_reset_enable': args.npc_npc_contact_reset,
        'npc_npc_contact_dist_sim': args.npc_npc_contact_dist_sim,
        'npc_npc_contact_progress_window': args.npc_npc_contact_progress_window,
        'npc_npc_contact_cooldown_steps': args.npc_npc_contact_cooldown_steps,
        'spawn_min_gap_sim_ego_npc_hard_floor': args.spawn_min_gap_sim_ego_npc_hard_floor,
        'spawn_min_gap_progress_ego_npc_hard_floor': args.spawn_min_gap_progress_ego_npc_hard_floor,
        'npc_persist_across_agent_resets': args.npc_persist_across_agent_resets,
        'npc_stuck_reset_enable': args.npc_stuck_reset_enable,
        'npc_stuck_speed_thresh': args.npc_stuck_speed_thresh,
        'npc_stuck_disp_thresh_sim': args.npc_stuck_disp_thresh_sim,
        'npc_stuck_steps': args.npc_stuck_steps,
        'npc_stuck_cooldown_steps': args.npc_stuck_cooldown_steps,
        'reverse_penalty_steps': args.reverse_penalty_steps,
        'reverse_progress_step_thresh': args.reverse_progress_step_thresh,
        'reverse_progress_dist_thresh': args.reverse_progress_dist_thresh,
        'reverse_onset_penalty': args.reverse_onset_penalty,
        'reverse_streak_penalty_scale': args.reverse_streak_penalty_scale,
        'reverse_backdist_penalty_scale': args.reverse_backdist_penalty_scale,
        'reverse_event_penalty': args.reverse_event_penalty,
        'reverse_gate_enabled': args.enable_reverse_gate,
        'progress_reward_scale': args.progress_reward_scale,
        'progress_backward_penalty_scale': args.progress_backward_penalty_scale,
        'progress_milestone_lap': args.progress_milestone_lap,
        'progress_milestone_reward': args.progress_milestone_reward,
        'spawn_bins': args.spawn_bins,
        'progress_reward_decay_min': args.progress_reward_decay_min,
        'penalty_decay_min': args.penalty_decay_min,
        'random_start_from_stage': args.random_start_from_stage,
        'stage1_cte_reset_limit': args.stage1_cte_reset_limit,
        'npc_side_by_side_start': args.npc_side_by_side_start,
        'npc_front_start': args.npc_front_start,
        'npc_front_offset_nodes': args.npc_front_offset_nodes,
    }
    if not getattr(args, "disable_stage_profiles", False):
        apply_stage_train_profile(
            stage_id, curriculum_stage_ref, args=args, cli_overrides=cli_overrides, verbose=True
        )
    _apply_v11_2_curriculum_overrides(args, curriculum_stage_ref)
    _apply_npc_mode_override(args, curriculum_stage_ref)
    stage_id = int(curriculum_stage_ref.get("stage", stage_id))

    # v10.2: trainingnote(noteStage A)note/noteNPC, stagenotewrappernote.
    # note: note Stage1 npc_count=0, note num_npc>0 note NPC controlnote(lazynote),
    # note Stage2+ note _pending_npc_connect note, notesucceedednote.
    stage_npc_count = int(curriculum_stage_ref.get('npc_count', 0))
    saved_num_npc = args.num_npc
    lazy_connect_npcs = bool(int(saved_num_npc) > 0)   # note NPC note lazy note
    # note args.num_npc, note create_generated_env_and_npcs note NPC note
    vec_env, wrapper, npcs, track_cache = create_generated_env_and_npcs(
        args, curriculum_stage_ref, dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK),
        lazy_connect_npcs=lazy_connect_npcs,
    )
    args.num_npc = saved_num_npc
    wrapper.dist_scale = build_dist_scale_profile(track_cache, "generated_track")
    print("\n📏 note:")
    print(json.dumps(wrapper.dist_scale, indent=2, ensure_ascii=False))
    try:
        vec_env.seed(seed)
    except Exception:
        pass

    run_meta = {
        "ts_unix": int(time.time()),
        "mode": "train",
        "script_path": os.path.abspath(__file__),
        "script_sha256": _sha256_file(__file__),
        "git": _git_metadata_for_path(__file__),
        "runtime": {
            "python": str(sys.version),
            "platform": str(platform.platform()),
            "numpy": _safe_module_version("numpy"),
            "torch": _safe_module_version("torch"),
            "stable_baselines3": _safe_module_version("stable_baselines3"),
            "sb3_contrib": _safe_module_version("sb3_contrib"),
            "gym_donkeycar": _safe_module_version("gym_donkeycar"),
            "gym": _safe_module_version("gym"),
        },
        "seed": int(seed),
        "args": dict(vars(args)),
        "curriculum_stage_ref": dict(curriculum_stage_ref),
        "dist_scale_profile": dict(wrapper.dist_scale),
    }
    _write_json(getattr(args, "run_meta_json", ""), run_meta)

    policy_kwargs = dict(
        features_extractor_class=LightweightCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"build notemodel: {args.pretrained_model}")
        model = RecurrentPPO.load(args.pretrained_model, env=vec_env, tensorboard_log=args.tb_log)
        try:
            model.set_random_seed(seed)
        except Exception:
            pass
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            vec_env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.tb_log,
            seed=seed,
        )

    print("\n🚀 V11.2 trainingnote(note + fixed2staticNPC, goalnote+controlnote, generated_track + LSTM)")
    print(f"   note: {args.total_steps:,}")
    print(f"   notestage: {stage_id} ({STAGES[stage_id].name})")
    print(f"   v11_2_mode: {args.v11_2_mode}")
    print(f"   auto_promote: {args.auto_promote}")
    print(f"   eval_freq_steps: {args.eval_freq_steps:,}")

    curriculum = CurriculumManager(
        curriculum_stage_ref,
        eval_freq_steps=args.eval_freq_steps,
        eval_episodes=args.eval_episodes,
        success_laps=args.eval_success_laps,
        consecutive_success_required=args.eval_consecutive_success,
        args=args,
        cli_overrides=cli_overrides,
        enable_stage_profiles=(not getattr(args, "disable_stage_profiles", False)),
        eval_max_steps=args.eval_max_steps,
    )
    curriculum.set_stage(stage_id, global_step=0, wrapper=wrapper, model=model)

    total_done = 0
    chunk = max(1000, int(args.train_chunk_steps))
    control_tb_callback = V11ControlTBCallback(log_every=args.ctrl_tb_log_every)

    try:
        while total_done < int(args.total_steps):
            steps = min(chunk, int(args.total_steps) - total_done)
            model.learn(
                total_timesteps=steps,
                reset_num_timesteps=False,
                tb_log_name="v11_2_generatedtrack_fixed2",
                callback=control_tb_callback,
            )
            total_done += steps

            if total_done % args.save_freq == 0 or total_done >= int(args.total_steps):
                cur_stage = int(curriculum_stage_ref.get('stage', stage_id))
                path = os.path.join(save_dir, f"v11_2_fixed2_lstm_stage{cur_stage}_step{total_done}")
                model.save(path)
                print(f"saved notesave: {path}.zip")

            if args.auto_promote:
                curriculum.maybe_eval_and_promote(model, wrapper, total_done)
                try:
                    vec_env.reset()  # notewrappernote, noteVecEnvnote
                except Exception as e:
                    print(f"⚠️ vec_env.reset() notefailed: {e}")

    finally:
        final_stage = int(curriculum_stage_ref.get('stage', stage_id))
        final_path = os.path.join(save_dir, f"v11_2_fixed2_lstm_final_stage{final_stage}")
        try:
            model.save(final_path)
            print(f"saved notemodel: {final_path}.zip")
        except Exception as e:
            print(f"⚠️ savenotemodelfailed: {e}")
        cleanup_envs(vec_env, npcs)


def cleanup_envs(vec_env, npc_controllers):
    for npc in npc_controllers:
        try:
            npc.close()
        except Exception:
            pass
    try:
        vec_env.close()
    except Exception:
        pass


def parse_args():
    p = argparse.ArgumentParser(description="V11.2 generated_track fixed2/curriculum RecurrentPPO(LSTM) trainingnote")
    p.add_argument('--mode', choices=['calibrate', 'reset-stress', 'train', 'eval-checkpoints', 'export-spawn-table'], default='calibrate')
    p.add_argument('--v11-2-mode', choices=['fixed2', 'curriculum'], default='fixed2',
                   help='fixed2=note+notestaticNPC(succeedednote); curriculum=note')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=9091)
    p.add_argument('--seed', type=int, default=42, help='note(python/numpy/torch/sb3)')
    p.add_argument('--exe-path', type=str, default=None)
    p.add_argument('--force-scene-reload', action='store_true', default=True,
                   help='noterows exit_scene -> load_scene(generated_track)')
    p.add_argument('--no-force-scene-reload', action='store_false', dest='force_scene_reload')
    p.add_argument('--scene-load-timeout', type=float, default=25.0,
                   help='note(note)')

    # note generated_track(note)
    p.add_argument('--max-cte', type=float, default=10.0)
    p.add_argument('--random-start', action='store_true', default=True)
    p.add_argument('--no-random-start', action='store_false', dest='random_start')
    p.add_argument('--num-npc', type=int, default=2, help='noteNPCcontrolnote(fixed2notefixed_npc_count)')
    p.add_argument('--fixed-npc-count', type=int, default=2, help='fixed2noteNPCnote(default2, notestagenote)')
    p.add_argument('--max-episode-steps', type=int, default=1500)
    p.add_argument('--body-style', choices=['donkey', 'bare', 'car01', 'f1', 'cybertruck'], default='donkey',
                   help='Learner note(note simulator car_config note)')
    p.add_argument('--body-rgb', type=int, nargs=3, metavar=('R', 'G', 'B'), default=[128, 128, 255],
                   help='Learner note, 0-255')
    p.add_argument('--car-name', type=str, default='V11_2_Learner', help='Learner note(note)')
    p.add_argument('--enable-sim-lidar', action='store_true', default=False,
                   help='note simulator LiDAR(note GYM conf note lidar_config)')
    p.add_argument('--lidar-deg-per-sweep-inc', type=float, default=2.0)
    p.add_argument('--lidar-deg-ang-down', type=float, default=0.0)
    p.add_argument('--lidar-deg-ang-delta', type=float, default=-1.0)
    p.add_argument('--lidar-num-sweeps-levels', type=int, default=1)
    p.add_argument('--lidar-max-range', type=float, default=50.0)
    p.add_argument('--lidar-noise', type=float, default=0.5)
    p.add_argument('--lidar-offset-x', type=float, default=0.0, help='LiDAR note')
    p.add_argument('--lidar-offset-y', type=float, default=1.14, help='LiDAR note(note)')
    p.add_argument('--lidar-offset-z', type=float, default=0.5, help='LiDAR firstnote')
    p.add_argument('--lidar-rot-x', type=float, default=0.0, help='LiDAR note')
    p.add_argument('--npc-side-by-side-start', action='store_true', default=False,
                   help='note: noteNPCnoteLearnernote(notelane offsetnote)')
    p.add_argument('--npc-front-start', action='store_true', default=False,
                   help='note: noteNPCnoteLearnernotefirstnote(note)')
    p.add_argument('--npc-front-offset-nodes', type=int, default=8,
                   help='NPCnotefirstnote, noteNPCnoteLearnernotefirstnote')
    p.add_argument('--npc-mode-override',
                   choices=['none', 'static', 'wobble', 'slow', 'slow_policy', 'random', 'chaos', 'random_reverse'],
                   default='none',
                   help='notestagenotenpc_mode; random_reversenotechaos(note)')

    # spawn / reset debug
    p.add_argument('--spawn-jitter-s-sim', type=float, default=0.30)
    p.add_argument('--spawn-jitter-d-sim', type=float, default=0.25)
    p.add_argument('--spawn-yaw-jitter-deg', type=float, default=6.0)
    p.add_argument('--spawn-verify-tol-sim', type=float, default=0.90)
    p.add_argument('--spawn-verify-cte-threshold', type=float, default=8.0)
    p.add_argument('--no-spawn-debug', action='store_true')
    p.add_argument('--spawn-debug-violations-limit', type=int, default=5)
    p.add_argument('--npc-layout-debug', action='store_true', help='noteNPCnote/note')

    # v10.1 NPC layout note / note / notecontrol
    p.add_argument('--npc-layout-reset-policy', choices=['hybrid', 'agent_only', 'every_episode'], default='hybrid')
    p.add_argument('--npc-layout-reset-every', type=int, default=5)
    p.add_argument('--npc-layout-reset-on-collision', action='store_true', default=True)
    p.add_argument('--no-npc-layout-reset-on-collision', action='store_false', dest='npc_layout_reset_on_collision')
    p.add_argument('--npc-layout-reset-on-success', action='store_true', default=True)
    p.add_argument('--no-npc-layout-reset-on-success', action='store_false', dest='npc_layout_reset_on_success')
    p.add_argument('--npc-layout-segments', type=int, default=3)
    p.add_argument('--npc-lane-balance-mode', choices=['balanced_lr', 'random', 'balanced_lr_center'], default='balanced_lr')
    p.add_argument('--npc-lane-offset-left-sim', type=float, default=None)
    p.add_argument('--npc-lane-offset-right-sim', type=float, default=None)
    p.add_argument('--npc-lane-offset-center-sim', type=float, default=None)
    p.add_argument('--npc-lane-jitter-sim', type=float, default=None)

    # v10.1 pair-wise spawn note(note)
    p.add_argument('--spawn-min-gap-sim-ego-npc', type=float, default=None)
    p.add_argument('--spawn-min-gap-progress-ego-npc', type=int, default=None)
    p.add_argument('--spawn-min-gap-sim-npc-npc', type=float, default=None)
    p.add_argument('--spawn-min-gap-progress-npc-npc', type=int, default=None)
    p.add_argument('--npc-npc-spawn-hard-check', action='store_true', default=True,
                   help='spawnnote/noteNPC-NPCnote(note)')
    p.add_argument('--no-npc-npc-spawn-hard-check', action='store_false', dest='npc_npc_spawn_hard_check')
    p.add_argument('--npc-npc-collision-guard', action='store_true', default=True,
                   help='noterowsnoteNPC-NPCnote(note)')
    p.add_argument('--no-npc-npc-collision-guard', action='store_false', dest='npc_npc_collision_guard')
    p.add_argument('--npc-npc-guard-dist-sim', type=float, default=1.10,
                   help='noterowsnoteNPC-NPCnote(sim)')
    p.add_argument('--npc-npc-guard-progress-window', type=int, default=120,
                   help='noteNPCnotetracknoterowsnote(fine idx)')
    p.add_argument('--npc-npc-guard-brake-steps', type=int, default=7,
                   help='noterowsnote, NPCnote')
    p.add_argument('--npc-npc-guard-cooldown-steps', type=int, default=14,
                   help='noteNPCnote')
    p.add_argument('--npc-npc-contact-reset', action='store_true', default=False,
                   help='note: detectionnoteNPC-NPCnoteNPCnote(notelearner)')
    p.add_argument('--no-npc-npc-contact-reset', action='store_false', dest='npc_npc_contact_reset')
    p.add_argument('--npc-npc-contact-dist-sim', type=float, default=0.45,
                   help='noteNPC-NPCnote(sim)')
    p.add_argument('--npc-npc-contact-progress-window', type=int, default=26,
                   help='noteNPC-NPCnote(fine idx)')
    p.add_argument('--npc-npc-contact-cooldown-steps', type=int, default=45,
                   help='NPC-NPCnote, note')
    p.add_argument('--spawn-min-gap-sim-ego-npc-hard-floor', type=float, default=1.40,
                   help='note: notespawnnoteNPCnotelearnernote(sim)')
    p.add_argument('--spawn-min-gap-progress-ego-npc-hard-floor', type=int, default=60,
                   help='note: notespawnnoteNPCnotelearnernote(fine idx)')
    p.add_argument('--npc-persist-across-agent-resets', action='store_true', default=True,
                   help='defaultnote: learner reset noteNPCnote/note(note)')
    p.add_argument('--no-npc-persist-across-agent-resets', action='store_false', dest='npc_persist_across_agent_resets')
    p.add_argument('--npc-stuck-reset-enable', action='store_true', default=True,
                   help='detectionnoteNPCnoterowsNPC-onlynote(notelearner)')
    p.add_argument('--no-npc-stuck-reset-enable', action='store_false', dest='npc_stuck_reset_enable')
    p.add_argument('--npc-stuck-speed-thresh', type=float, default=0.10,
                   help='NPCnote: note(sim)')
    p.add_argument('--npc-stuck-disp-thresh-sim', type=float, default=0.012,
                   help='NPCnote: note(sim)')
    p.add_argument('--npc-stuck-steps', type=int, default=80,
                   help='NPCnote: note')
    p.add_argument('--npc-stuck-cooldown-steps', type=int, default=120,
                   help='NPCnote')

    # reward / control (notev9note)
    p.add_argument('--enable-dr', action='store_true', default=True)
    p.add_argument('--no-dr', action='store_false', dest='enable_dr')
    p.add_argument('--max-throttle', type=float, default=0.30)
    p.add_argument('--delta-max', type=float, default=0.10)
    p.add_argument('--enable-lpf', action='store_true', default=True)
    p.add_argument('--no-lpf', action='store_false', dest='enable_lpf')
    p.add_argument('--beta', type=float, default=0.6)
    p.add_argument('--w-d', type=float, default=0.01)
    p.add_argument('--w-dd', type=float, default=0.01)
    p.add_argument('--w-sat', type=float, default=0.01)
    p.add_argument('--progress-reward-scale', type=float, default=0.85)
    p.add_argument('--progress-backward-penalty-scale', type=float, default=0.25)
    p.add_argument('--progress-milestone-lap', type=float, default=0.125,
                   help='note, notereward')
    p.add_argument('--progress-milestone-reward', type=float, default=0.2)
    p.add_argument('--progress-reward-decay-min', type=float, default=0.35,
                   help='note progress shaping rewardnote')
    p.add_argument('--penalty-decay-min', type=float, default=0.55,
                   help='note shaping note')
    p.add_argument('--random-start-from-stage', type=int, default=5,
                   help='notestagenote(notefirstnote)')
    p.add_argument('--stage1-cte-reset-limit', type=float, default=6.0,
                   help='Stage1 note CTE reset note')
    p.add_argument('--spawn-bins', type=int, default=8,
                   help='notetracknote, note(note)')
    p.add_argument('--reverse-penalty-steps', type=int, default=5,
                   help='note(note)')
    p.add_argument('--reverse-progress-step-thresh', type=int, default=1,
                   help='fine_tracknote(note)')
    p.add_argument('--reverse-progress-dist-thresh', type=float, default=0.05,
                   help='note(sim distance)')
    p.add_argument('--reverse-onset-penalty', type=float, default=0.12,
                   help='note')
    p.add_argument('--reverse-streak-penalty-scale', type=float, default=0.03,
                   help='note')
    p.add_argument('--reverse-backdist-penalty-scale', type=float, default=1.4,
                   help='note(reward-based anti-reverse)')
    p.add_argument('--reverse-event-penalty', type=float, default=0.8,
                   help='note')
    p.add_argument('--enable-reverse-gate', action='store_true', default=False,
                   help='notecontrolnotereverse gate(defaultnote, noterewardnote)')

    # v11 target + controller
    p.add_argument('--v-ref-min', type=float, default=0.05)
    p.add_argument('--v-ref-max', type=float, default=1.7)
    p.add_argument('--kappa-ref-max', type=float, default=2.1)
    p.add_argument('--v-ref-rate-max', type=float, default=0.8)
    p.add_argument('--kappa-ref-rate-max', type=float, default=4.0)
    p.add_argument('--steer-ff-headroom', type=float, default=0.85,
                   help='noteFFnote(|steer_ff|note), notekappa_ref')
    p.add_argument('--steer-softsat-gain', type=float, default=1.0,
                   help='steer soft saturation: tanh(gain * steer_raw)')
    p.add_argument('--steer-slew-rate-max', type=float, default=3.0,
                   help='noterowsnote(1/s)')
    p.add_argument('--ctrl-dt-min', type=float, default=0.01)
    p.add_argument('--ctrl-dt-max', type=float, default=0.2)
    p.add_argument('--ctrl-dt-fallback', type=float, default=0.05)
    p.add_argument('--gyro-lpf-tau', type=float, default=0.10)
    p.add_argument('--v-fb-min', type=float, default=0.25)
    p.add_argument('--yaw-kp', type=float, default=0.35)
    p.add_argument('--yaw-kd', type=float, default=0.0)
    p.add_argument('--yaw-ff-k', type=float, default=0.52)
    p.add_argument('--speed-kp', type=float, default=0.9)
    p.add_argument('--speed-ki', type=float, default=0.35)
    p.add_argument('--speed-kaw', type=float, default=0.5)
    p.add_argument('--throttle-brake-min', type=float, default=0.0)
    p.add_argument('--v-stop-eps', type=float, default=0.08)
    p.add_argument('--speed-i-leak', type=float, default=3.0)
    p.add_argument('--speed-i-fwd-max', type=float, default=0.40)
    p.add_argument('--speed-i-brake-max', type=float, default=0.25)
    p.add_argument('--startup-force-throttle-steps', type=int, default=24)
    p.add_argument('--startup-force-throttle', type=float, default=0.20)
    p.add_argument('--no-motion-penalty-start-steps', type=int, default=40)
    p.add_argument('--no-motion-penalty-speed-thresh', type=float, default=0.15)
    p.add_argument('--no-motion-penalty-progress-thresh', type=float, default=0.002)
    p.add_argument('--no-motion-penalty-per-step', type=float, default=0.25)
    p.add_argument('--progress-local-window', type=int, default=120,
                   help='note(fine_tracknote)')
    p.add_argument('--progress-local-recover-dist', type=float, default=2.5,
                   help='note(sim)')
    p.add_argument('--progress-idle-freeze-speed', type=float, default=0.12,
                   help='note(notedfi)')
    p.add_argument('--progress-idle-freeze-dfi-abs', type=int, default=1,
                   help='note|dfi|note(<=note0)')
    p.add_argument('--enable-progress-heading-filter', action='store_true', default=False,
                   help='notetracknote(note)')
    p.add_argument('--progress-heading-dot-min', type=float, default=-0.20,
                   help='note, note(note)')

    # v11 unit/sign checks
    p.add_argument('--unit-calibrate', action='store_true', default=True)
    p.add_argument('--no-unit-calibrate', action='store_false', dest='unit_calibrate')
    p.add_argument('--unit-calib-steps', type=int, default=300)
    p.add_argument('--unit-calib-ratio-min', type=float, default=2.0,
                   help='rmse_deg_hyp/rmse_rad_hyp note')
    p.add_argument('--unit-calib-strict', action='store_true', default=False,
                   help='notetraining')
    p.add_argument('--sign-check-min-steer', type=float, default=0.08)
    p.add_argument('--sign-check-min-speed', type=float, default=0.25)
    p.add_argument('--sign-check-max-lag', type=int, default=15)
    p.add_argument('--sign-check-min-corr', type=float, default=0.10)
    p.add_argument('--sign-check-min-samples', type=int, default=40)

    # v11.2 fixed2 guards / stall / spawn / npc feasibility
    p.add_argument('--manual-width-profile', type=str,
                   default=os.path.join(
                       os.path.dirname(os.path.dirname(__file__)),
                       'track_profiles',
                       'manual_width_generated_track.json',
                   ))
    p.add_argument('--spawn-table-out', type=str,
                   default='track_profiles/generated_track_intrack_spawn_table.json',
                   help='--mode export-spawn-table noteoutputJSONpath')
    p.add_argument('--spawn-table-csv', type=str,
                   default='track_profiles/generated_track_intrack_spawn_table.csv',
                   help='--mode export-spawn-table noteoutputCSVpath; note')
    p.add_argument('--spawn-table-step-idx', type=int, default=3,
                   help='notefinenote(note)')
    p.add_argument('--spawn-table-lane-fracs', type=str, default='0.12,0.25,0.38,0.50,0.62,0.75,0.88',
                   help='notetracknote(0=note,1=note), note')
    p.add_argument('--spawn-table-margin-ratio', type=float, default=0.08,
                   help='notemarginnote')
    p.add_argument('--spawn-table-kappa-max', type=float, default=1.6,
                   help='note; <=0note')
    p.add_argument('--spawn-table-y-tel', type=float, default=0.0625,
                   help='notetelemetry ynote')
    p.add_argument('--spawn-table-node-count', type=int, default=108,
                   help='notefine_idx->node_idxnote')
    p.add_argument('--fixed-success-progress', type=float, default=2.0)
    p.add_argument('--fixed-success-laps', type=int, default=2)
    p.add_argument('--success-no-hit-steps', type=int, default=200)
    p.add_argument('--success-no-offtrack-seconds', type=float, default=3.0)
    p.add_argument('--success-cte-norm-margin', type=float, default=0.08)
    p.add_argument('--success-min-steps', type=int, default=300)
    p.add_argument('--success-progress-int-laps', type=int, default=2)
    p.add_argument('--success-use-milestone-fallback', action='store_true', default=False)

    p.add_argument('--stall-penalty-enable', action='store_true', default=True)
    p.add_argument('--no-stall-penalty-enable', action='store_false', dest='stall_penalty_enable')
    p.add_argument('--stall-speed-thresh', type=float, default=0.15)
    p.add_argument('--stall-progress-step-thresh', type=float, default=0.003)
    p.add_argument('--stall-penalty-per-step', type=float, default=0.002)
    p.add_argument('--stall-penalty-warmup-steps', type=int, default=50000)

    p.add_argument('--spawn-local-window-m', type=float, default=2.0)
    p.add_argument('--spawn-local-window-min-idx', type=int, default=20)
    p.add_argument('--spawn-anchor-window-m', type=float, default=4.5,
                   help='fixed2noteEgonoteanchornote(note)')
    p.add_argument('--spawn-anchor-window-min-idx', type=int, default=48,
                   help='fixed2noteEgo anchornotefinenote')
    p.add_argument('--spawn-safe-anchor-pool', type=int, default=72,
                   help='fixed2noteNPCnoteEgonoteanchornote')
    p.add_argument('--spawn-kappa-max', type=float, default=1.6)
    p.add_argument('--spawn-inside-margin-ratio', type=float, default=0.10)
    p.add_argument('--spawn-inside-max-attempts', type=int, default=24)

    p.add_argument('--npc-min-track-width-ratio', type=float, default=0.75)
    p.add_argument('--npc-min-track-width-abs-min', type=float, default=0.8)
    p.add_argument('--layout-age-hard-cap', type=int, default=80)

    # calibrate / stress
    p.add_argument('--scale-profile-out', type=str, default='dist_scale_profile_generated_track.json')
    p.add_argument('--stress-resets', type=int, default=200)
    p.add_argument('--stress-rollout-steps', type=int, default=20)
    p.add_argument('--stress-layout-mode', choices=['full', 'agent-only'], default='agent-only')
    p.add_argument('--start-stage', type=int, default=2, choices=[1, 2, 3, 4, 5])

    # train (RecurrentPPO)
    p.add_argument('--disable-stage-profiles', action='store_true',
                   help='noteStageProfile/StageEvalGate, noteCLInote')
    p.add_argument('--total-steps', type=int, default=1000000)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--n-steps', type=int, default=2048)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--n-epochs', type=int, default=4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-range', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.01)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)
    p.add_argument('--pretrained-model', type=str, default=None)
    p.add_argument('--save-dir', type=str, default='models/v11_2_randomspawn_fixednpc_lstm')
    p.add_argument('--save-freq', type=int, default=20000)
    p.add_argument('--tb-log', type=str, default='./logs/v11_2_randomspawn_fixednpc_lstm/')
    p.add_argument('--ctrl-tb-log-every', type=int, default=1000,
                   help='controlnoteTensorBoardnote')
    p.add_argument('--episode-summary-jsonl', type=str, default='./logs/v11_2_episode_summary.jsonl',
                   help='noteJSONL, note')
    p.add_argument('--episode-summary-csv', type=str, default='./logs/v11_2_episode_summary.csv',
                   help='noteCSV, note')
    p.add_argument('--run-meta-json', type=str, default='./logs/v11_2_run_meta.json',
                   help='noterowsnotedatanote(configuration/note/seed/git)JSON, note')
    p.add_argument('--event-trace-jsonl', type=str, default='./logs/v11_2_event_trace.jsonl',
                   help='notetrace JSONL(notefirstnote), note')
    p.add_argument('--event-trace-window-steps', type=int, default=60,
                   help='notetracefirstnote')
    p.add_argument('--event-trace-max-per-episode', type=int, default=6,
                   help='notetrace')
    p.add_argument('--event-dt-spike-thresh', type=float, default=0.08,
                   help='|dt_raw-dt|notedt_spikenote')
    p.add_argument('--event-sat-spike-cooldown-steps', type=int, default=40,
                   help='sat/dt spikenote(note)')
    p.add_argument('--event-generic-cooldown-steps', type=int, default=40,
                   help='reward/progress/spawn note(note)')
    p.add_argument('--event-reward-spike-abs', type=float, default=8.0,
                   help='|reward|notereward_spikenote')
    p.add_argument('--event-reward-gap-spike-abs', type=float, default=2.0,
                   help='|reward_terms_gap|notereward_spikenote')
    p.add_argument('--event-progress-jump-fi', type=int, default=12,
                   help='|progress_step_fi_raw|noteprogress_anomalynote')
    p.add_argument('--event-progress-clip-ratio-high', type=float, default=0.25,
                   help='progress_clip_rationoteprogress_anomalynote')
    p.add_argument('--event-spawn-near-dist', type=float, default=4.4,
                   help='note ego_npc_min_dist notespawn_nearnote')
    p.add_argument('--event-periodic-sample-steps', type=int, default=20000,
                   help='noteNnoteperiodic_samplenote, 0note')
    p.add_argument('--train-chunk-steps', type=int, default=50000)

    # auto eval / promotion
    p.add_argument('--auto-promote', action='store_true', default=False)
    p.add_argument('--no-auto-promote', action='store_false', dest='auto_promote')
    p.add_argument('--eval-freq-steps', type=int, default=20000)
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--eval-success-laps', type=int, default=2)
    p.add_argument('--eval-consecutive-success', type=int, default=2)
    p.add_argument('--eval-max-steps', type=int, default=1000,
                   help='note; defaultnotesucceedednotefirstnote, note')

    # batch checkpoint eval
    p.add_argument('--eval-ckpt-glob', type=str, default='models/v11_2_randomspawn_fixednpc_lstm/*stage2*.zip',
                   help='notecheckpointnoteglobnote(note)')
    p.add_argument('--eval-ckpt-stage', type=int, default=None, choices=[1, 2, 3, 4, 5],
                   help='notestagenote/rewardconfigurationnotecheckpoint; defaultnote --start-stage')
    p.add_argument('--eval-ckpt-limit', type=int, default=10,
                   help='noteNnotecheckpoint(notestepnote); 0note')

    return p.parse_args()


def _enforce_v11_2_mode(args):
    mode = str(getattr(args, "v11_2_mode", "fixed2")).lower()
    changed = []
    if mode == "fixed2":
        # fixed_npc_count note >= 1
        if int(getattr(args, "fixed_npc_count", 2)) < 1:
            changed.append(f"fixed_npc_count:{args.fixed_npc_count}->2")
            args.fixed_npc_count = 2
        # NPC controlnote"notestagenote"note, noteNPCnote.
        max_needed = _max_npc_count_from_stage(int(getattr(args, "start_stage", 1)))
        desired_npc = int(max_needed)
        if int(getattr(args, "num_npc", 0))!= desired_npc:
            changed.append(f"num_npc:{args.num_npc}->{desired_npc}")
            args.num_npc = desired_npc
        # notestart_stagenote, notestage(notestage1)
        # notestagenoteSTAGE_TRAIN_PROFILESnotenpc_count
        if not bool(getattr(args, "random_start", True)):
            changed.append("random_start:False->True")
            args.random_start = True
        # noteauto_promotenote, note
        if getattr(args, "eval_ckpt_stage", None) not in (None, 2):
            changed.append(f"eval_ckpt_stage:{args.eval_ckpt_stage}->2")
            args.eval_ckpt_stage = 2
    elif mode!= "curriculum":
        changed.append(f"v11_2_mode:{mode}->fixed2")
        args.v11_2_mode = "fixed2"

    if changed:
        print(f"🔒 V11.2 note({mode}):", ", ".join(changed))


def main():
    args = parse_args()
    _enforce_v11_2_mode(args)
    print("=" * 80)
    print("V11.2 generated_track random-spawn + fixed2 NPC + target-controller + LSTM")
    print("=" * 80)
    print(f"note: {args.mode} | v11_2_mode={args.v11_2_mode} | host={args.host}:{args.port}")
    print("note:", OBS_FIELD_CAPS['stable'])

    if args.mode == 'calibrate':
        calibrate_only(args)
    elif args.mode == 'reset-stress':
        reset_stress_mode(args)
    elif args.mode == 'train':
        train_mode(args)
    elif args.mode == 'eval-checkpoints':
        eval_checkpoints_mode(args)
    elif args.mode == 'export-spawn-table':
        export_spawn_table_mode(args)
    else:
        raise ValueError(args.mode)


if __name__ == '__main__':
    main()
