#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 wrapper NPC runtime mixin."""

import math
import random

import numpy as np

from .v14_dep_sim_core import SimExtendedAPI
from .utils import _safe_float


class V14NpcRuntimeMixin:
    def _maybe_reset_stuck_npcs(self, info=None):
        """V14自写: NPC卡住不动时，触发NPC重置。"""
        if not self._npc_stuck_reset_enabled():
            return False
        if not self._v14_npc_reset_enabled():
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
            npc_id = int(getattr(npc, "npc_id", -1))
            if npc_id < 0:
                continue
            st = self._npc_stuck_state.get(npc_id)
            x, _, z = npc.get_telemetry_position()
            mode = str(getattr(npc, "mode", "static")).lower()
            v = abs(float(getattr(npc, "speed", 0.0)))
            if st is None:
                self._npc_stuck_state[npc_id] = {
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
                stuck_ids.append(int(npc_id))

        if not stuck_ids:
            return False

        ok = bool(self._v14_reset_npcs_after_stage34_collision(info, reason="v14_npc_stuck"))
        self._npc_stuck_reset_cooldown_left = self._npc_stuck_cooldown()
        if not ok:
            self._v14_force_collision_spawn_refresh = True
        for sid in stuck_ids:
            if sid in self._npc_stuck_state:
                self._npc_stuck_state[sid]["still_steps"] = 0

        if isinstance(info, dict):
            info["npc_stuck_detected"] = True
            info["npc_stuck_ids"] = list(stuck_ids)
            info["npc_stuck_layout_reset"] = bool(ok)
            info["v14_npc_stuck_reset"] = bool(ok)
        if isinstance(self.episode_stats, dict):
            self.episode_stats["npc_stuck_resets"] = int(
                self.episode_stats.get("npc_stuck_resets", 0)
            ) + int(bool(ok))
        if ok:
            self.npc_layout_last_reset_reason = "v14_npc_stuck"
            print(f"🚨 V14 NPC卡住重置: ids={stuck_ids} cooldown={self._npc_stuck_reset_cooldown_left}")
        return bool(ok)
    def _maybe_handle_npc_npc_contact_reset(self, info=None):
        if isinstance(info, dict):
            info["v14_disable_legacy_npc_contact_reset"] = True
        return False
    def _v14_npc_reset_enabled(self):
        stage_id = int(self.curriculum_stage_ref.get("stage", 1))
        npc_count = int(self.curriculum_stage_ref.get("npc_count", 0))
        return bool(stage_id >= 2 and npc_count > 0 and (not self._episode_npc_free))
    def _v14_stage34_enabled(self):
        stage_id = int(self.curriculum_stage_ref.get("stage", 1))
        return bool(stage_id in (3, 4) and (not self._episode_npc_free))
    def _v14_stage34_speed_ratio_bounds(self):
        rmin = float(self.v14_stage34_npc_speed_ratio_min)
        rmax = float(self.v14_stage34_npc_speed_ratio_max)
        if (not np.isfinite(rmin)) or rmin < 0.0:
            rmin = 0.30
        if (not np.isfinite(rmax)) or rmax <= 0.0:
            rmax = 0.80
        if rmax < rmin:
            rmin, rmax = rmax, rmin
        return float(rmin), float(rmax)
    def _v14_sync_stage34_npc_speed_cap(self, info):
        if not self._v14_stage34_enabled():
            return
        if not isinstance(info, dict):
            return
        active_npcs = self._active_npcs()
        if not active_npcs:
            return

        learner_speed = abs(_safe_float(info.get("speed", getattr(self, "last_speed_est", 0.0)), 0.0))
        if not np.isfinite(learner_speed):
            learner_speed = 0.0
        rmin, rmax = self._v14_stage34_speed_ratio_bounds()
        cap_min = max(0.03, learner_speed * rmin)
        cap_max = max(cap_min, learner_speed * rmax)
        ref_speed = max(0.5, float(getattr(self, "v_ref_max", 1.7)))
        t_min = float(np.clip(cap_min / ref_speed, 0.05, 0.85))
        t_max = float(np.clip(cap_max / ref_speed, t_min, 0.90))

        for npc in active_npcs:
            try:
                if hasattr(npc, "set_speed_cap"):
                    npc.set_speed_cap(cap_max)
            except Exception:
                pass
            try:
                mode = str(getattr(npc, "mode", "")).strip().lower()
                if mode in ("slow", "random", "chaos"):
                    base_thr = float(getattr(npc, "base_throttle", 0.20))
                    npc.base_throttle = float(np.clip(base_thr, t_min, t_max))
            except Exception:
                pass

        info["v14_stage34_npc_speed_cap_min"] = float(cap_min)
        info["v14_stage34_npc_speed_cap_max"] = float(cap_max)
        info["v14_stage34_npc_speed_ratio_min"] = float(rmin)
        info["v14_stage34_npc_speed_ratio_max"] = float(rmax)
    def _v14_detect_stage34_collision(self, done, info):
        if (not self._v14_npc_reset_enabled()) or (not bool(self.v14_stage34_collision_reset_npc)):
            return False
        if not isinstance(info, dict):
            return False
        hit_v = str(info.get("hit", "none")).strip().lower()
        if hit_v not in ("none", "", "null"):
            return True
        if bool(done) and str(info.get("termination_reason", "")).strip().lower() == "collision":
            return True
        if bool(getattr(self, "episode_stats", {}).get("collision", False)):
            return True
        return False
    def _v14_reset_npcs_after_stage34_collision(self, info=None, reason="stage34_collision"):
        if not self._v14_npc_reset_enabled():
            return False
        active_npcs = self._active_npcs()
        if not active_npcs:
            return False

        learner_fi = int(getattr(self, "learner_fine_idx", 0))
        if isinstance(info, dict) and ("learner_fine_idx" in info):
            learner_fi = int(_safe_float(info.get("learner_fine_idx", learner_fi), learner_fi))
        learner_pos = self._extract_pos(info) if isinstance(info, dict) else (0.0, 0.0, 0.0)
        has_trace = len(self._v14_learner_trace_fi) >= max(24, int(self.v14_trace_min_points))
        use_trace_only = bool(self._v14_global_init_done and has_trace)
        ok, records = self._v14_place_active_npcs(
            active_npcs, learner_fi, learner_pos, trace_only=use_trace_only
        )
        if (not ok) or (len(records) < len(active_npcs)):
            ok2, records2 = self._v14_place_active_npcs(
                active_npcs, learner_fi, learner_pos, trace_only=False
            )
            if ok2:
                ok = True
                records = records2

        if ok:
            npc_mode = self._normalize_npc_mode(self.curriculum_stage_ref.get("npc_mode", "offtrack"))
            npc_speed_min = float(self.curriculum_stage_ref.get("npc_speed_min", 0.0))
            npc_speed_max = float(self.curriculum_stage_ref.get("npc_speed_max", 0.0))
            if bool(self.v14_npc_wobble_in_place) and str(npc_mode).strip().lower() == "wobble":
                self._v14_apply_custom_wobble_runtime_mode(active_npcs)
            else:
                self._apply_active_npc_runtime_mode(
                    active_npcs, npc_mode, npc_speed_min, npc_speed_max, freeze_for_spawn=False
                )
            self.npc_layout_id = int(getattr(self, "npc_layout_id", 0)) + 1
            self.npc_layout_last_reset_reason = str(reason)
            self._v14_last_reposition_step = {}

        if isinstance(info, dict):
            info["v14_stage34_npc_collision_reset"] = bool(ok)
            info["v14_stage34_npc_collision_reset_reason"] = str(reason)
            info["v14_stage34_npc_collision_reset_count"] = int(len(records))
        return bool(ok)
    def _v14_maybe_reset_npcs_on_stage34_contact(self, done, info):
        if (not self._v14_npc_reset_enabled()) or (not bool(self.v14_stage34_collision_reset_npc)):
            return False
        if bool(done):
            return False
        active_npcs = self._active_npcs()
        if len(active_npcs) < 2:
            return False

        if self._v14_stage34_contact_reset_cooldown_left > 0:
            self._v14_stage34_contact_reset_cooldown_left -= 1
            return False
        contact = self._detect_npc_npc_contact()
        if not contact:
            return False

        ok = self._v14_reset_npcs_after_stage34_collision(
            info, reason="stage34_npc_npc_contact"
        )
        self._v14_stage34_contact_reset_cooldown_left = int(
            max(1, self.v14_stage34_npc_contact_reset_cooldown_steps)
        )
        if isinstance(info, dict):
            info["v14_stage34_npc_contact_pair"] = contact.get("pair")
            info["v14_stage34_npc_contact_dist"] = float(contact.get("dist", -1.0))
            if contact.get("progress_gap") is not None:
                info["v14_stage34_npc_contact_progress_gap"] = int(contact.get("progress_gap"))
        return bool(ok)
    def _v14_apply_custom_wobble_runtime_mode(self, active_npcs):
        for npc in active_npcs:
            if npc is None or (not getattr(npc, "connected", False)):
                continue
            try:
                if getattr(npc, "running", False):
                    npc.stop_driving()
            except Exception:
                pass
            try:
                npc.set_mode("static", 0.0)
            except Exception:
                pass
            try:
                npc.handler.send_control(0, 0, 1.0)
            except Exception:
                pass
    def _v14_apply_npc_inplace_wobble(self):
        if not bool(self.v14_npc_wobble_in_place):
            return
        mode = self._normalize_npc_mode(self.curriculum_stage_ref.get("npc_mode", "offtrack"))
        if mode != "wobble":
            return
        active_npcs = self._active_npcs()
        if not active_npcs:
            return
        self._v14_wobble_tick += 1
        tick = int(self._v14_wobble_tick)
        update_every = max(1, int(self.v14_npc_wobble_update_every_steps))
        if (tick % update_every) != 0:
            return
        period = max(12, int(self.v14_npc_wobble_period_steps))
        radius = max(0.02, float(self.v14_npc_wobble_radius_sim))
        yaw_jitter = float(max(0.0, self.v14_npc_wobble_yaw_jitter_deg))
        S = SimExtendedAPI.COORD_SCALE
        for npc in active_npcs:
            npc_id = int(getattr(npc, "npc_id", -1))
            st = self._v14_npc_anchor_state.get(npc_id)
            if st is None:
                try:
                    nx, ny, nz = npc.get_telemetry_position()
                except Exception:
                    continue
                st = {
                    "x": float(nx),
                    "y": float(ny if np.isfinite(ny) else 0.0625),
                    "z": float(nz),
                    "yaw_deg": float(getattr(npc, "yaw", 0.0)),
                    "phase_seed": float(random.uniform(0.0, 2.0 * math.pi)),
                }
                self._v14_npc_anchor_state[npc_id] = st

            phase0 = float(st.get("phase_seed", 0.0))
            angle = (2.0 * math.pi * float(tick) / float(period)) + phase0
            dx = radius * math.sin(angle)
            dz = (radius * 0.72) * math.cos(angle * 1.13)
            yaw_deg = float(st.get("yaw_deg", 0.0)) + yaw_jitter * math.sin(angle * 0.83)
            qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
            tx = float(st.get("x", 0.0)) + float(dx)
            ty = float(st.get("y", 0.0625))
            tz = float(st.get("z", 0.0)) + float(dz)
            try:
                npc.set_position_node_coords(tx * S, ty * S, tz * S, qx, qy, qz, qw)
                if self.track_cache:
                    try:
                        fi, _ = self.track_cache.find_nearest_fine_track(
                            self.scene_name, float(tx), float(tz)
                        )
                        npc.fine_track_idx = int(fi)
                    except Exception:
                        npc.fine_track_idx = int(getattr(npc, "fine_track_idx", 0))
                else:
                    npc.fine_track_idx = int(getattr(npc, "fine_track_idx", 0))
            except Exception:
                continue
    def _compute_learner_npc_lateral_dist(self, info, npc_feat):
        """
        计算learner与NPC的真实横向距离（垂直于赛道切线方向的投影距离）。

        用赛道切线向量将learner-NPC位移分解为:
        - longitudinal: 沿赛道方向的距离（前后）
        - lateral: 垂直于赛道方向的距离（左右）
        返回: (lateral_dist, longitudinal_dist)
        """
        lx, _, lz = self._extract_pos(info)
        npc = npc_feat.get('npc')
        if npc is None:
            return 0.0, 0.0
        nx, _, nz = npc.get_telemetry_position()

        # 赛道切线方向（在learner位置处）
        learner_fi = npc_feat.get('learner_fine_idx', getattr(self, 'learner_fine_idx', 0))
        if self.manual_spawn_v14.loaded:
            tx, tz = self.manual_spawn_v14.tangent(learner_fi)
        elif self.track_cache:
            fine = self.track_cache.fine_track.get(self.scene_name, [])
            n = len(fine)
            if n > 2:
                i = int(learner_fi) % n
                p0 = fine[(i - 1) % n]
                p1 = fine[(i + 1) % n]
                dx = p1[0] - p0[0]
                dz = p1[1] - p0[1]
                norm = math.sqrt(dx * dx + dz * dz)
                tx, tz = (dx / norm, dz / norm) if norm > 1e-9 else (0.0, 1.0)
            else:
                tx, tz = 0.0, 1.0
        else:
            tx, tz = 0.0, 1.0

        # learner→NPC 位移向量
        dx = nx - lx
        dz = nz - lz

        # 投影: longitudinal = dot(delta, tangent), lateral = cross(tangent, delta)
        longitudinal = dx * tx + dz * tz
        lateral = abs(dx * (-tz) + dz * tx)  # 法向分量的绝对值

        return lateral, longitudinal
    def _avg_fine_gap_sim(self):
        return float(max(1e-4, _safe_float(self.dist_scale.get('avg_fine_gap_sim', 0.025), 0.025)))
    def _sim_to_ahead_idx(self, sim_dist, default_idx=40):
        try:
            return max(1, int(round(float(sim_dist) / self._avg_fine_gap_sim())))
        except Exception:
            return int(max(1, default_idx))
    def _place_npc_on_learner_path(self, npc, learner_fi, ahead_range=None, learner_pos=None):
        """
        将NPC放置在learner前方路径上（赛车线位置）。

        Args:
            npc: NPCController
            learner_fi: learner当前的fine_track索引
            ahead_range: (min_idx, max_idx) 前方fine_track索引偏移范围
        """
        if not self.manual_spawn_v14.loaded:
            return False
        ms = self.manual_spawn_v14
        n = int(ms.fine_track.shape[0])
        if n <= 0:
            return False

        if ahead_range is None:
            ahead_min = self._sim_to_ahead_idx(self.npc_spawn_ahead_min_sim, default_idx=120)
            ahead_max = self._sim_to_ahead_idx(self.npc_spawn_ahead_max_sim, default_idx=260)
        else:
            ahead_min, ahead_max = int(ahead_range[0]), int(ahead_range[1])
        ahead_min = max(1, int(ahead_min))
        ahead_max = max(ahead_min, int(ahead_max))

        if learner_pos is None:
            lx, _, lz = 0.0, 0.0, 0.0
        else:
            lx, _, lz = learner_pos

        other_npcs = [x for x in self._active_npcs() if x is not npc]
        candidates = list(range(ahead_min, ahead_max + 1))
        random.shuffle(candidates)
        tries = max(6, int(self.npc_spawn_candidate_tries))
        tried = 0

        min_learner_dist = max(0.5, float(self.npc_spawn_min_learner_dist_sim))
        min_npc_dist = max(0.3, float(self.npc_spawn_min_npc_dist_sim))
        min_npc_gap_idx = max(1, int(self.npc_spawn_min_npc_progress_gap_idx))

        fallback_pose = None
        fallback_score = -1e9
        S = SimExtendedAPI.COORD_SCALE
        for ahead in candidates:
            if tried >= tries:
                break
            tried += 1
            target_fi = (int(learner_fi) + int(ahead)) % n
            if self.racing_line.loaded:
                racing_offset = self.racing_line.get_offset_cte(target_fi)
                racing_offset_sim = racing_offset / max(1e-3, ms.coord_scale)
            else:
                racing_offset_sim = 0.0

            cx, cz = ms.fine_track[target_fi]
            ttx, ttz = ms.tangent(target_fi)
            normal_x, normal_z = -ttz, ttx
            jitter = random.uniform(-0.03, 0.03)
            dn = racing_offset_sim + jitter
            tel_x = cx + dn * normal_x
            tel_z = cz + dn * normal_z
            tel_y = 0.0625

            d_learner = math.hypot(tel_x - lx, tel_z - lz)
            if d_learner < min_learner_dist:
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
                    break
                if self.track_cache:
                    ofi = int(getattr(other, "fine_track_idx", target_fi))
                    pgap = abs(int(self.track_cache.progress_diff(self.scene_name, target_fi, ofi)))
                    min_gap_npc = min(min_gap_npc, pgap)
                    if pgap < min_npc_gap_idx:
                        ok = False
                        break
            if not ok:
                score = d_learner + 0.5 * min_d_npc + 0.01 * min_gap_npc
                if score > fallback_score:
                    fallback_score = score
                    fallback_pose = (tel_x, tel_y, tel_z, ttx, ttz, target_fi)
                continue

            yaw_deg = math.degrees(math.atan2(ttx, ttz))
            yaw_deg = self._v14_npc_yaw_for_spawn(float(yaw_deg))
            qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
            try:
                npc.set_position_node_coords(tel_x * S, tel_y * S, tel_z * S, qx, qy, qz, qw)
                npc.fine_track_idx = target_fi
                self._v14_register_npc_anchor(npc, tel_x, tel_y, tel_z, float(yaw_deg))
                return True
            except Exception:
                continue

        if fallback_pose is not None:
            tel_x, tel_y, tel_z, ttx, ttz, target_fi = fallback_pose
            yaw_deg = math.degrees(math.atan2(ttx, ttz))
            yaw_deg = self._v14_npc_yaw_for_spawn(float(yaw_deg))
            qx, qy, qz, qw = SimExtendedAPI.yaw_to_quaternion(yaw_deg)
            try:
                npc.set_position_node_coords(tel_x * S, tel_y * S, tel_z * S, qx, qy, qz, qw)
                npc.fine_track_idx = target_fi
                self._v14_register_npc_anchor(npc, tel_x, tel_y, tel_z, float(yaw_deg))
                return True
            except Exception:
                return False
        return False
    def _maybe_reposition_npcs_on_path(self, info):
        """
        Stage 2/4: 检查NPC是否在learner前方路径上。
        如果NPC已被agent超过或距离过远，则重新放置到前方路径上。
        """
        stage_id = int(self.curriculum_stage_ref.get('stage', 1))
        if stage_id not in (2, 4) or self._episode_npc_free:
            return

        feats = self._npc_distance_features(info)
        if not feats:
            return

        learner_fi = getattr(self, 'learner_fine_idx', 0)
        learner_pos = self._extract_pos(info)
        ahead_min_idx = self._sim_to_ahead_idx(self.npc_spawn_ahead_min_sim, default_idx=120)
        ahead_max_idx = self._sim_to_ahead_idx(self.npc_spawn_ahead_max_sim, default_idx=260)
        overpass_gap_idx = max(20, int(self.v14_reposition_overpass_gap_idx))
        min_step_gap = max(0, int(self.v14_reposition_min_step_gap))
        cur_step = int(getattr(self, "episode_step", 0))
        for f in feats:
            progress_diff = f.get('progress_diff', 0)
            dist = f.get('dist', float('inf'))
            npc = f.get('npc')
            if npc is None:
                continue
            npc_id = int(getattr(npc, "npc_id", -1))
            if npc_id >= 0 and min_step_gap > 0:
                last_step = int(self._v14_last_reposition_step.get(npc_id, -10**9))
                if cur_step - last_step < min_step_gap:
                    continue

            # NPC在agent后方(被超过) 或 距离过远 → 重新放置到前方
            d_reposition = float(self.dist_scale.get('follow_safe_max_sim', 4.0)) * max(2.0, float(self.v14_reposition_far_factor))
            if progress_diff > overpass_gap_idx or dist > d_reposition:
                ok = self._place_npc_on_learner_path(
                    npc,
                    learner_fi,
                    ahead_range=(ahead_min_idx, ahead_max_idx),
                    learner_pos=learner_pos,
                )
                if ok and npc_id >= 0:
                    self._v14_last_reposition_step[npc_id] = int(cur_step)


__all__ = ["V14NpcRuntimeMixin"]
