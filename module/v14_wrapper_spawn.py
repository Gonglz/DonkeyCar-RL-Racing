#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 wrapper spawn/reset mixin."""

import math
import random
import time

import numpy as np

from .v14_dep_sim_core import SimExtendedAPI
from .utils import _safe_float


class V14SpawnResetMixin:
    def _v14_track_fine(self):
        fine = self._track_fine() if hasattr(self, "_track_fine") else []
        if fine:
            return fine
        if self.manual_spawn_v14.loaded:
            return [tuple(p) for p in self.manual_spawn_v14.fine_track.tolist()]
        return []
    def _v14_tangent_at_fi(self, fi, fine=None):
        if self.manual_spawn_v14.loaded:
            try:
                tx, tz = self.manual_spawn_v14.tangent(int(fi))
                norm = math.sqrt(float(tx) * float(tx) + float(tz) * float(tz))
                if norm > 1e-8:
                    return float(tx / norm), float(tz / norm)
            except Exception:
                pass
        fine = fine if fine is not None else self._v14_track_fine()
        n = len(fine)
        if n <= 2:
            return 0.0, 1.0
        i = int(fi) % n
        p0 = fine[(i - 1) % n]
        p1 = fine[(i + 1) % n]
        dx = float(p1[0] - p0[0])
        dz = float(p1[1] - p0[1])
        norm = math.sqrt(dx * dx + dz * dz)
        if norm <= 1e-9:
            return 0.0, 1.0
        return dx / norm, dz / norm
    def _v14_pose_from_fi(self, fi, lateral_offset_sim=0.0, yaw_deg_override=None):
        fine = self._v14_track_fine()
        n = len(fine)
        if n <= 0:
            return None
        i = int(fi) % n
        cx, cz = fine[i]
        tx, tz = self._v14_tangent_at_fi(i, fine=fine)
        nx, nz = -tz, tx
        tel_x = float(cx) + float(lateral_offset_sim) * float(nx)
        tel_z = float(cz) + float(lateral_offset_sim) * float(nz)
        tel_y = 0.0625
        if yaw_deg_override is None:
            yaw_deg = math.degrees(math.atan2(tx, tz))
        else:
            yaw_deg = float(yaw_deg_override)
        qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
        return {
            "fine_idx": int(i),
            "tel": (float(tel_x), float(tel_y), float(tel_z)),
            "quat": (float(qx), float(qy), float(qz), float(qw)),
            "yaw_deg": float(yaw_deg),
        }
    def _v14_npc_yaw_for_spawn(self, fallback_yaw_deg):
        if bool(self.v14_npc_random_heading):
            return float(random.uniform(-180.0, 180.0))
        return float(fallback_yaw_deg)
    def _v14_register_npc_anchor(self, npc, tel_x, tel_y, tel_z, yaw_deg):
        if npc is None:
            return
        npc_id = int(getattr(npc, "npc_id", -1))
        if npc_id < 0:
            return
        self._v14_npc_anchor_state[npc_id] = {
            "x": float(tel_x),
            "y": float(tel_y),
            "z": float(tel_z),
            "yaw_deg": float(yaw_deg),
            "phase_seed": float(random.uniform(0.0, 2.0 * math.pi)),
        }
    def _v14_forget_npc_anchor(self, npc):
        if npc is None:
            return
        npc_id = int(getattr(npc, "npc_id", -1))
        if npc_id < 0:
            return
        self._v14_npc_anchor_state.pop(npc_id, None)
    def _v14_teleport_learner_to_fi(self, fi, settle_steps=3):
        pose = self._v14_pose_from_fi(fi, lateral_offset_sim=0.0)
        if pose is None:
            return {"ok": False, "reason": "no_pose", "obs": None, "info": {}, "fine_idx": int(fi)}
        handler = getattr(getattr(self.env, "viewer", None), "handler", None)
        if handler is None:
            return {"ok": False, "reason": "no_handler", "obs": None, "info": {}, "fine_idx": int(fi)}
        tel_x, tel_y, tel_z = pose["tel"]
        qx, qy, qz, qw = pose["quat"]
        try:
            for _ in range(2):
                SimExtendedAPI.send_set_position_from_telemetry(handler, tel_x, tel_y, tel_z, qx, qy, qz, qw)
                time.sleep(0.03)
            obs, info = self._idle_step_for_telemetry(max(2, int(settle_steps)))
            if not isinstance(info, dict):
                info = {}
            lx, ly, lz = self._extract_pos(info)
            fi_real = int(pose["fine_idx"])
            if self.track_cache:
                try:
                    fi_real, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(lx), float(lz))
                except Exception:
                    pass
            cte_abs = abs(_safe_float(info.get("cte", 0.0), 0.0))
            geom_ok, _ = self._geometry_track_check(
                float(lx),
                float(lz),
                seed_fi=int(fi_real),
                margin_scale=0.45,
                centerline_slack=0.22,
            )
            return {
                "ok": bool(geom_ok),
                "reason": None if bool(geom_ok) else "geometry_check_failed",
                "obs": obs,
                "info": info,
                "fine_idx": int(fi_real),
                "cte_after": float(cte_abs),
                "stabilize_steps": max(2, int(settle_steps)),
            }
        except Exception as e:
            return {"ok": False, "reason": f"teleport_exception:{e}", "obs": None, "info": {}, "fine_idx": int(fi)}
    def _v14_spawn_learner_fixed_start(self):
        fine = self._v14_track_fine()
        n = len(fine)
        if n <= 0:
            return {"ok": False, "reason": "no_fine_track", "obs": None, "info": {}, "fine_idx": 0}
        # 固定起点附近多次尝试，避免偶发物理抖动导致CTE爆表
        candidates = [0, 1, 2, n - 1, 3]
        candidates = [int(c) % n for c in candidates]
        cte_ok_th = max(1.0, min(float(self.current_max_cte) * 0.75, float(self.current_max_cte) - 0.2))
        best = None
        for k, fi in enumerate(candidates):
            res = self._v14_teleport_learner_to_fi(fi, settle_steps=3 + min(2, k))
            if best is None:
                best = res
            else:
                best_cte = _safe_float(best.get("cte_after", 1e9), 1e9)
                cur_cte = _safe_float(res.get("cte_after", 1e9), 1e9)
                if cur_cte < best_cte:
                    best = res
            cte_now = _safe_float(res.get("cte_after", 1e9), 1e9)
            if bool(res.get("ok", False)) and cte_now <= cte_ok_th:
                return res
        return best if isinstance(best, dict) else {"ok": False, "reason": "no_spawn_result", "obs": None, "info": {}, "fine_idx": 0}
    def _v14_record_learner_trace(self, info):
        fi = None
        if isinstance(info, dict):
            v = info.get("learner_fine_idx", None)
            if v is not None:
                fi = int(_safe_float(v, 0))
        if fi is None:
            fi = int(getattr(self, "learner_fine_idx", 0))
        if self.track_cache:
            fine = self._track_fine()
            n = len(fine)
            if n > 0:
                fi = int(fi) % int(n)
        if len(self._v14_learner_trace_fi) == 0 or int(self._v14_learner_trace_fi[-1]) != int(fi):
            self._v14_learner_trace_fi.append(int(fi))
    def _v14_place_npc_from_trace_window(self, npc, learner_fi, learner_pos):
        if not self.track_cache:
            return False
        n = len(self._track_fine())
        if n <= 0:
            return False
        hist = list(self._v14_learner_trace_fi)
        if len(hist) < max(24, int(self.v14_trace_min_points)):
            return False
        window = hist[-max(64, int(self.v14_trace_window)):]
        if not window:
            return False
        ahead_min_idx = self._sim_to_ahead_idx(self.npc_spawn_ahead_min_sim, default_idx=120)
        ahead_max_idx = self._sim_to_ahead_idx(self.npc_spawn_ahead_max_sim, default_idx=260)
        min_learner_dist = max(0.5, float(self.npc_spawn_min_learner_dist_sim))
        min_npc_dist = max(0.3, float(self.npc_spawn_min_npc_dist_sim))
        min_npc_gap_idx = max(1, int(self.npc_spawn_min_npc_progress_gap_idx))
        lx, _, lz = learner_pos
        other_npcs = [x for x in self._active_npcs() if x is not npc]

        uniq = []
        seen = set()
        for fi in reversed(window):
            key = int(fi) % n
            if key in seen:
                continue
            seen.add(key)
            uniq.append(key)
        candidate_fis = []
        for candidate_fi in uniq:
            pdiff = int(self.track_cache.progress_diff(self.scene_name, int(learner_fi), int(candidate_fi)))
            ahead_gap = -pdiff
            if pdiff < 0 and ahead_min_idx <= ahead_gap <= ahead_max_idx:
                candidate_fis.append(int(candidate_fi))
        if not candidate_fis:
            return False
        random.shuffle(candidate_fis)

        S = SimExtendedAPI.COORD_SCALE
        best_relaxed_pose = None
        best_relaxed_score = -1e9
        min_learner_dist_hard = max(2.8, min_learner_dist * 0.65)
        min_npc_dist_hard = max(1.8, min_npc_dist * 0.65)
        min_npc_gap_hard = max(30, int(min_npc_gap_idx * 0.50))
        tried = 0
        max_tries = max(8, int(self.npc_spawn_candidate_tries) * 2)
        for candidate_fi in candidate_fis:
            if tried >= max_tries:
                break
            tried += 1
            lateral_offset = random.uniform(-0.03, 0.03)
            pose_ref = self._v14_pose_from_fi(candidate_fi, lateral_offset_sim=lateral_offset)
            if pose_ref is None:
                continue
            spawn_yaw = self._v14_npc_yaw_for_spawn(float(pose_ref.get("yaw_deg", 0.0)))
            pose = self._v14_pose_from_fi(
                candidate_fi,
                lateral_offset_sim=lateral_offset,
                yaw_deg_override=spawn_yaw,
            )
            if pose is None:
                continue
            tel_x, tel_y, tel_z = pose["tel"]
            if math.hypot(tel_x - lx, tel_z - lz) < min_learner_dist:
                continue
            ok = True
            min_d_npc = 999.0
            min_gap_npc = 99999
            for other in other_npcs:
                ox, _, oz = other.get_telemetry_position()
                d_npc = math.hypot(tel_x - ox, tel_z - oz)
                min_d_npc = min(min_d_npc, d_npc)
                if d_npc < min_npc_dist:
                    ok = False
                ofi = int(getattr(other, "fine_track_idx", candidate_fi))
                pgap = abs(int(self.track_cache.progress_diff(self.scene_name, int(candidate_fi), int(ofi))))
                min_gap_npc = min(min_gap_npc, pgap)
                if pgap < min_npc_gap_idx:
                    ok = False
            if not ok:
                relaxed_ok = (
                    math.hypot(tel_x - lx, tel_z - lz) >= min_learner_dist_hard
                    and min_d_npc >= min_npc_dist_hard
                    and min_gap_npc >= min_npc_gap_hard
                )
                if relaxed_ok:
                    score = (
                        math.hypot(tel_x - lx, tel_z - lz)
                        + 0.6 * min_d_npc
                        + 0.01 * min_gap_npc
                    )
                    if score > best_relaxed_score:
                        best_relaxed_score = score
                        best_relaxed_pose = (tel_x, tel_y, tel_z, pose["quat"], int(candidate_fi))
                continue
            qx, qy, qz, qw = pose["quat"]
            try:
                npc.set_position_node_coords(tel_x * S, tel_y * S, tel_z * S, qx, qy, qz, qw)
                npc.fine_track_idx = int(candidate_fi)
                self._v14_register_npc_anchor(npc, tel_x, tel_y, tel_z, float(spawn_yaw))
                return True
            except Exception:
                continue
        if best_relaxed_pose is not None:
            tel_x, tel_y, tel_z, quat, candidate_fi = best_relaxed_pose
            qx, qy, qz, qw = quat
            try:
                npc.set_position_node_coords(tel_x * S, tel_y * S, tel_z * S, qx, qy, qz, qw)
                npc.fine_track_idx = int(candidate_fi)
                try:
                    yaw_from_quat = 2.0 * math.degrees(math.atan2(float(qz), float(qw)))
                except Exception:
                    yaw_from_quat = 0.0
                self._v14_register_npc_anchor(npc, tel_x, tel_y, tel_z, float(yaw_from_quat))
                return True
            except Exception:
                pass
        return False
    def _v14_place_active_npcs(self, active_npcs, learner_fi, learner_pos, trace_only=False):
        if not active_npcs:
            return True, []
        ahead_min_idx = self._sim_to_ahead_idx(self.npc_spawn_ahead_min_sim, default_idx=120)
        ahead_max_idx = self._sim_to_ahead_idx(self.npc_spawn_ahead_max_sim, default_idx=260)
        has_trace = len(self._v14_learner_trace_fi) >= max(24, int(self.v14_trace_min_points))
        prefer_random_spawn = bool(self.v14_npc_spawn_randomize)
        records = []
        for npc in active_npcs:
            ok = False
            for _ in range(4):
                if prefer_random_spawn:
                    ok = self._place_npc_on_learner_path(
                        npc,
                        learner_fi=int(learner_fi),
                        ahead_range=(ahead_min_idx, ahead_max_idx),
                        learner_pos=learner_pos,
                    )
                if (not ok) and has_trace:
                    ok = self._v14_place_npc_from_trace_window(
                        npc,
                        learner_fi=int(learner_fi),
                        learner_pos=learner_pos,
                    )
                if (not ok) and (not trace_only):
                    ok = self._place_npc_on_learner_path(
                        npc,
                        learner_fi=int(learner_fi),
                        ahead_range=(ahead_min_idx, ahead_max_idx),
                        learner_pos=learner_pos,
                    )
                if ok:
                    break
            if not ok:
                self._hide_npc_offtrack(npc)
                self._v14_forget_npc_anchor(npc)
                continue
            nx, ny, nz = npc.get_telemetry_position()
            fi = int(getattr(npc, "fine_track_idx", 0))
            if self.track_cache:
                try:
                    fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(nx), float(nz))
                    npc.fine_track_idx = int(fi)
                except Exception:
                    pass
            st = self._v14_npc_anchor_state.get(int(getattr(npc, "npc_id", -1)))
            if st is None:
                self._v14_register_npc_anchor(npc, nx, ny, nz, float(getattr(npc, "yaw", 0.0)))
            records.append({
                "npc_id": int(getattr(npc, "npc_id", -1)),
                "npc_fine_idx": int(fi),
                "tel": (float(nx), float(ny), float(nz)),
            })
        return bool(records), records
    def should_refresh_npc_layout(self, active_npcs):
        # V14自写reset：每回合直接按当前learner状态重采样，无复用缓存布局
        if not active_npcs:
            return False, "v14_no_active_npc"
        return True, "v14_custom_reset_each_episode"
    def _spawn_episode_layout(self, refresh_npc_layout=True, refresh_reason=None):
        active_npcs = self._active_npcs()
        for npc in self._inactive_npcs():
            self._hide_npc_offtrack(npc)
            self._v14_forget_npc_anchor(npc)
        global_warmup = bool(self.v14_global_warmup_no_npc and (not self._v14_global_init_done))
        if global_warmup:
            for npc in active_npcs:
                self._hide_npc_offtrack(npc)
                self._v14_forget_npc_anchor(npc)
            active_npcs = []
        force_refresh = bool(self._v14_force_collision_spawn_refresh)
        spawn_debug = {
            "reset_idx": int(getattr(self, "_reset_index", 0)),
            "stage": int(self.curriculum_stage_ref.get("stage", 1)),
            "ego_spawn_idx": 0,
            "ego_fine_idx": None,
            "npc": [],
            "spawn_retries": 0,
            "spawn_validation_pass": False,
            "spawn_fail_reason": None,
            "ego_npc_min_dist": None,
            "ego_npc_progress_gap": None,
            "post_spawn_cte": None,
            "stabilize_steps": 0,
            "layout_reused": False,
            "npc_layout_id": int(getattr(self, "npc_layout_id", 0)),
            "npc_layout_last_reset_reason": "v14_custom_reset",
            "npc_layout_age_agent_resets": 0,
            "refresh_reason": "v14_custom_pipeline",
            "attempt_fail_counts": {},
            "bad_nodes_count": 0,
            "safe_anchor_count": 0,
            "global_warmup_no_npc": bool(global_warmup),
            "collision_forced_refresh": bool(force_refresh),
            "trace_points": int(len(self._v14_learner_trace_fi)),
        }

        npc_mode = self._normalize_npc_mode(self.curriculum_stage_ref.get("npc_mode", "offtrack"))
        npc_speed_min = float(self.curriculum_stage_ref.get("npc_speed_min", 0.0))
        npc_speed_max = float(self.curriculum_stage_ref.get("npc_speed_max", 0.0))
        self._apply_active_npc_runtime_mode(
            active_npcs, npc_mode, npc_speed_min, npc_speed_max, freeze_for_spawn=True
        )

        learner_res = self._v14_spawn_learner_fixed_start()
        obs = learner_res.get("obs")
        info = learner_res.get("info", {}) if isinstance(learner_res.get("info"), dict) else {}
        learner_fi = int(learner_res.get("fine_idx", 0))
        spawn_debug["ego_fine_idx"] = int(learner_fi)
        spawn_debug["post_spawn_cte"] = _safe_float(learner_res.get("cte_after", 0.0), 0.0)
        spawn_debug["stabilize_steps"] = int(learner_res.get("stabilize_steps", 0))
        if not bool(learner_res.get("ok", False)):
            reason = str(learner_res.get("reason", "learner_spawn_failed"))
            spawn_debug["spawn_fail_reason"] = reason
            spawn_debug["attempt_fail_counts"] = {reason: 1}

        learner_pos = self._extract_pos(info) if isinstance(info, dict) else (0.0, 0.0, 0.0)
        use_trace_only = (not global_warmup) and (not force_refresh)
        npc_ok, npc_records = self._v14_place_active_npcs(
            active_npcs, learner_fi, learner_pos, trace_only=use_trace_only
        )
        spawn_debug["npc"] = npc_records
        if not npc_ok:
            spawn_debug["spawn_fail_reason"] = spawn_debug.get("spawn_fail_reason") or "npc_place_failed"
            for npc in active_npcs:
                self._hide_npc_offtrack(npc)
                self._v14_forget_npc_anchor(npc)
            active_npcs = []

        # 启动NPC运行模式（仅对成功放置的活跃NPC）
        if active_npcs:
            if bool(self.v14_npc_wobble_in_place) and str(npc_mode).strip().lower() == "wobble":
                self._v14_apply_custom_wobble_runtime_mode(active_npcs)
            else:
                self._apply_active_npc_runtime_mode(
                    active_npcs, npc_mode, npc_speed_min, npc_speed_max, freeze_for_spawn=False
                )

        obs2, info2 = self._idle_step_for_telemetry(1)
        if obs2 is not None:
            obs = obs2
        if isinstance(info2, dict) and info2:
            info = info2

        if npc_records and isinstance(info, dict):
            ex, _, ez = self._extract_pos(info)
            ego_fi = learner_fi
            if self.track_cache:
                try:
                    ego_fi, _ = self.track_cache.find_nearest_fine_track(self.scene_name, float(ex), float(ez))
                except Exception:
                    pass
            dists = []
            gaps = []
            for rec in npc_records:
                nx, _, nz = rec["tel"]
                dists.append(math.hypot(float(ex) - float(nx), float(ez) - float(nz)))
                if self.track_cache:
                    try:
                        pgap = abs(int(self.track_cache.progress_diff(self.scene_name, int(ego_fi), int(rec["npc_fine_idx"]))))
                        gaps.append(pgap)
                    except Exception:
                        pass
            if dists:
                spawn_debug["ego_npc_min_dist"] = float(min(dists))
            if gaps:
                spawn_debug["ego_npc_progress_gap"] = int(min(gaps))

        spawn_debug["spawn_validation_pass"] = bool(spawn_debug.get("post_spawn_cte", 0.0) <= float(self.current_max_cte))
        if spawn_debug["spawn_validation_pass"] and (not spawn_debug.get("spawn_fail_reason")):
            spawn_debug["spawn_fail_reason"] = None

        # 同步布局状态，避免父类进入旧的layout-reuse判断链路
        self.npc_layout_id = int(getattr(self, "npc_layout_id", 0)) + 1
        self.npc_layout_last_reset_reason = "v14_custom_reset"
        self.npc_layout_age_agent_resets = 0
        self._consecutive_layout_failures = 0
        self._force_npc_layout_refresh_next = False
        self._last_spawn_debug = dict(spawn_debug)
        self._v14_global_init_done = True
        self._v14_force_collision_spawn_refresh = False
        if isinstance(info, dict):
            self._v14_record_learner_trace(info)

        # 不返回失败，避免父类触发旧版fallback链路
        return {"ok": True, "obs": obs, "info": info, "debug": spawn_debug}
    def _refresh_npc_layout_mid_episode(self):
        # V14自写：禁用父类中途重排，避免闪烁/飞车
        return False


__all__ = ["V14SpawnResetMixin"]
