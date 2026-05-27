"""
module/reward.py
DonkeyRewardWrapper: DonkeyCar unifiedrewardnote.

noterewardnotefilenote, note.
notedescriptionnote docs/reward.md.
"""

import math
from collections import deque
from typing import Any, Dict, Optional, Tuple

import gym
import numpy as np


# ============================================================
# unifiedrewardnote
# ============================================================
class DonkeyRewardWrapper(gym.Wrapper):
    """
    DonkeyCar unifiedrewardnote.

    rewardnote:
      survival      = survival_reward_scale * speed_gate * 1[progress>0]
      speed         = 0.25 * ontrack * speed_gate * center_factor * 1[progress>0]
      progress      = progress_reward_scale * signed_progress_ratio(notetracknote)
      cte           = note | notereward(x speed_gate_cte)
      lap           = notereward
      center        = -w_center * |lat_err_norm|
      heading       = -w_heading * |heading_err| / pi
      speed_ref     = -w_speed_ref * ((v-v_ref(kappa))/v_ref_max)^2
      time          = -w_time
      near_offtrack = notefirstnote(cte note out note)
      near_collision= notefirstnote(obstacle/heading/speed/control)
      overtake      = succeedednoteobstaclenote bonus
      collision     = note/stuck/offtrack note
      smooth        = -w_d  * |Δsteer_exec|
      jerk          = -w_dd * |jerk|
      mismatch      = -w_m  * |steer_raw - steer_exec|
      sat           = -w_sat * tanh(rate_excess)

    CTE note: cte_left > 0(note), cte_right < 0(note), note lat_err note.
    step() note lat_err_cte = lat_err * coord_scale note.
    notedescriptionnote docs/reward.md.
    """

    def __init__(
        self,
        env,
        total_timesteps: int = 200000,
        action_safety_wrapper=None,
        w_d: float = 0.0,
        w_dd: float = 0.0,
        w_m: float = 0.0,
        w_sat: float = 0.0,
        w_time: float = 0.0,
        w_center: float = 0.0,
        w_heading: float = 0.0,
        w_speed_ref: float = 0.0,
        speed_ref_vmin: float = 0.35,
        speed_ref_vmax: float = 2.2,
        speed_ref_kappa_ref: float = 0.15,
        lap_reward_scale: float = 1.0,
        progress_reward_scale: float = 80.0,
        progress_curve_boost: float = 0.35,
        progress_kappa_ref: float = 0.15,
        progress_center_gate_min: float = 0.10,
        progress_center_gate_power: float = 1.0,
        smooth_curve_relief: float = 0.5,
        throttle_penalty_threshold: float = 1.0,
        throttle_penalty_amount: float = 0.0,
        survival_reward_scale: float = 0.2,
        collision_penalty_base: float = 8.0,
        offtrack_penalty_base: float = 6.0,
        w_near_offtrack: float = 0.40,
        near_offtrack_start_ratio: float = 0.45,
        w_near_collision: float = 0.35,
        near_collision_start_ratio: float = 0.65,
        overtake_success_bonus: float = 2.5,
        cte_left: float = 5.0,
        cte_right: float = -5.0,
        cte_left_out: Optional[float] = None,
        cte_right_out: Optional[float] = None,
        coord_scale: float = 8.0,
        offtrack_leniency_ratio: float = 0.25,
        offtrack_leniency_mult: float = 2.5,
        track_geometry=None,
        scene_key: str = "",
        logging_key: str = "",
        cte_half_width: float = 4.6,
        cte_norm_scale: Optional[float] = None,
        reward_decay_ref_steps: int = 0,
        enable_step_diagnostics: bool = False,
        step_diagnostics_first_steps: int = 3,
        step_diagnostics_every_episodes: int = 0,
        reset_env_done_grace_steps: int = 0,
        reset_collision_grace_steps: int = 0,
    ):
        super().__init__(env)
        self.total_timesteps = total_timesteps
        self.current_step = 0
        self.action_safety_wrapper = action_safety_wrapper
        # notegeometry: notecompute lat_err_cte = lat_err * coord_scale(note, note CTE note)
        self._track_geometry = track_geometry
        self._scene_key = scene_key
        self._logging_key = str(logging_key or scene_key or "")
        self._prev_track_idx = None
        self.coord_scale = float(max(coord_scale, 1e-3))
        self.enable_step_diagnostics = bool(enable_step_diagnostics)
        self.step_diagnostics_first_steps = max(1, int(step_diagnostics_first_steps))
        self.step_diagnostics_every_episodes = max(0, int(step_diagnostics_every_episodes))
        self.reset_env_done_grace_steps = max(0, int(reset_env_done_grace_steps))
        self.reset_collision_grace_steps = max(0, int(reset_collision_grace_steps))
        self._episode_index = 0

        # reward decay: note episode noterewardnote, note
        self.reward_decay_ref_steps = max(0, int(reward_decay_ref_steps))

        # offtrack done note: first leniency_ratio note mult note 1.0 note
        # note: CTE note(cte_term)note, note
        self._leniency_steps = int(total_timesteps * float(np.clip(offtrack_leniency_ratio, 0.0, 0.5)))
        self._leniency_mult  = float(max(1.0, offtrack_leniency_mult))

        self.w_d   = float(w_d)
        self.w_dd  = float(w_dd)
        self.w_m   = float(w_m)
        self.w_sat = float(w_sat)
        self.w_time = float(max(0.0, w_time))
        self.w_center = float(max(0.0, w_center))
        self.w_heading = float(max(0.0, w_heading))
        self.w_speed_ref = float(max(0.0, w_speed_ref))
        self.speed_ref_vmin = float(max(0.0, speed_ref_vmin))
        self.speed_ref_vmax = float(max(self.speed_ref_vmin + 1e-3, speed_ref_vmax))
        self.speed_ref_kappa_ref = float(max(1e-6, speed_ref_kappa_ref))
        self.lap_reward_scale = float(max(0.0, lap_reward_scale))
        self.progress_reward_scale = float(progress_reward_scale)
        self.progress_curve_boost = float(max(0.0, progress_curve_boost))
        self.progress_kappa_ref = float(max(1e-6, progress_kappa_ref))
        self.progress_center_gate_min = float(np.clip(progress_center_gate_min, 0.0, 1.0))
        self.progress_center_gate_power = float(max(0.1, progress_center_gate_power))
        self.smooth_curve_relief = float(np.clip(smooth_curve_relief, 0.0, 0.9))
        self.throttle_penalty_threshold = float(np.clip(throttle_penalty_threshold, 0.0, 1.0))
        self.throttle_penalty_amount = float(max(0.0, throttle_penalty_amount))
        self.survival_reward_scale = float(max(0.0, survival_reward_scale))
        self.collision_penalty_base = float(max(0.0, collision_penalty_base))
        self.offtrack_penalty_base = float(max(0.0, offtrack_penalty_base))
        self.w_near_offtrack = float(max(0.0, w_near_offtrack))
        self.near_offtrack_start_ratio = float(np.clip(near_offtrack_start_ratio, 0.0, 0.98))
        self.w_near_collision = float(max(0.0, w_near_collision))
        self.near_collision_start_ratio = float(np.clip(near_collision_start_ratio, 0.0, 0.98))
        self.overtake_success_bonus = float(max(0.0, overtake_success_bonus))
        # note 10 note: note, note1note10note.
        self.near_penalty_ramp_steps = 10
        self._near_offtrack_ramp_step = 0
        self._near_collision_ramp_step = 0
        self.overtake_arm_longitudinal_min_m = 0.8
        self.overtake_arm_planar_max_m = 8.0
        self.overtake_pass_longitudinal_threshold_m = -0.8
        self.overtake_pass_planar_min_m = 0.9
        self.overtake_min_front_steps = 4
        self.overtake_rearm_cooldown_steps = 12

        self.smooth_stats: deque = deque(maxlen=1000)

        # CTE note(note)
        # cte_left/right = left_in_max_sim: notetracknote CTE(noterewardnote)
        # cte_left_out/right_out = left_out_first_sim: note CTE(note done note)
        self.cte_left      = float(max(cte_left,  0.1))
        self.cte_right     = float(min(cte_right, -0.1))   # note: note
        self.cte_left_out  = float(max(cte_left_out,  self.cte_left))  if cte_left_out  is not None else self.cte_left  * 1.1
        self.cte_right_out = float(min(cte_right_out, self.cte_right)) if cte_right_out is not None else self.cte_right * 1.1

        # CTE rewardnote: notetrack CTE note「note」note
        # note: cte_abs / cte_boundary note [0,1] note,
        #   note cte_boundary note「noterewardnote」.
        #   notetrack boundary note -> noterewardnote -> note CTE rewardnote;
        #   notetrack boundary note -> noterewardnote -> note CTE rewardnote.
        #   notegoal: notetracknotetracknote.
        #   currentnote, notetrack(note ws)note, notetracknote.
        CTE_REF_HALF_WIDTH = 4.6
        self.cte_half_width = float(max(cte_half_width, 0.5))
        if cte_norm_scale is not None:
            self.cte_norm_scale = float(np.clip(cte_norm_scale, 0.1, 2.0))
        else:
            self.cte_norm_scale = float(np.clip(self.cte_half_width / CTE_REF_HALF_WIDTH, 0.75, 1.15))

        self.stuck_counter = 0
        self.offtrack_counter = 0
        self.episode_stats: Dict[str, Any] = self._zero_episode_stats()
        self.prev_lap_count = 0
        self._episode_overtake_count = 0
        self._overtake_front_steps = 0
        self._overtake_armed = False
        self._overtake_cooldown_steps_left = 0
        self._overtake_last_longitudinal: Optional[float] = None

        _side_method = "lat_err * coord_scale(note)" if track_geometry and scene_key else "-sim_cte(note fallback)"
        print(
            f"DonkeyRewardWrapper: w_d={w_d}, w_dd={w_dd}, w_m={w_m}, w_sat={w_sat}, "
            f"w_time={self.w_time:.3f}, w_center={self.w_center:.3f}, "
            f"w_heading={self.w_heading:.3f}, w_speed_ref={self.w_speed_ref:.3f}, "
            f"w_near_offtrack={self.w_near_offtrack:.3f}, w_near_collision={self.w_near_collision:.3f}, "
            f"survival_scale={self.survival_reward_scale:.3f}, "
            f"term(collision={self.collision_penalty_base:.2f}, offtrack={self.offtrack_penalty_base:.2f}), "
            f"lap_scale={self.lap_reward_scale:.2f}, prog_scale={self.progress_reward_scale:.2f}, "
            f"prog_curve_boost={self.progress_curve_boost:.2f}, "
            f"prog_gate(min={self.progress_center_gate_min:.2f}, p={self.progress_center_gate_power:.2f})"
        )
        print(
            f"   speed_ref: vmin={self.speed_ref_vmin:.2f}, vmax={self.speed_ref_vmax:.2f}, "
            f"kappa_ref={self.speed_ref_kappa_ref:.3f}"
        )
        print(f"   CTE in-note: left=+{self.cte_left:.3f}, right={self.cte_right:.3f}  out-note: left=+{self.cte_left_out:.3f}, right={self.cte_right_out:.3f}  coord_scale={self.coord_scale:.1f}")
        print(f"   CTE note: half_width={self.cte_half_width:.2f}, norm_scale={self.cte_norm_scale:.3f} (ref={CTE_REF_HALF_WIDTH})")
        print(f"   note: {_side_method}")
        print(f"   offtrack done note: first {self._leniency_steps:,} note donenote {self._leniency_mult:.1f}x -> 1.0x  (note)")
        if self.overtake_success_bonus > 0.0:
            print(
                f"   overtake_bonus: +{self.overtake_success_bonus:.2f} "
                f"(front>={self.overtake_arm_longitudinal_min_m:.1f}m for {self.overtake_min_front_steps} steps, "
                f"behind<={self.overtake_pass_longitudinal_threshold_m:.1f}m)"
            )
        if self.reward_decay_ref_steps > 0:
            print(f"   reward_decay: ref_steps={self.reward_decay_ref_steps} (noterewardnote ref/step note)")
        if self.reset_env_done_grace_steps > 0 or self.reset_collision_grace_steps > 0:
            print(
                f"   resetnote: env_donefirst{self.reset_env_done_grace_steps}note, "
                f"collisionfirst{self.reset_collision_grace_steps}note"
            )

        # rewardnote(note episode note)-- note Monitor -> PerSceneStatsCallback note
        self._reward_parts_episode: Dict[str, float] = self._zero_reward_parts()
        # note(note episode note)-- note
        self._episode_diag: Dict[str, Any] = self._zero_episode_diag()

    @staticmethod
    def _extract_obstacle_risk(info: Dict[str, Any]) -> float:
        """
        note info note"obstaclenote"[0,1].

        note:
        1) note(note)
        2) note(2.0 note)
        3) lidar(note)note
        4) note -1(noteobstaclenote)
        """
        # note(note)
        risk_keys = (
            "obstacle_risk",
            "collision_risk",
            "hit_risk",
            "near_hit_risk",
            "risk_collision",
        )
        for k in risk_keys:
            if k in info:
                try:
                    v = float(info.get(k, 0.0))
                    if np.isfinite(v):
                        info["obstacle_risk_source"] = f"direct:{k}"
                        return float(np.clip(v, 0.0, 1.0))
                except Exception:
                    pass

        # note(note)
        # note:
        # - d >= 4.0: note(note0)
        # - d <= 0.5: note(note1)
        # - note, note, note.
        dist_keys = (
            "obstacle_dist",
            "obstacle_distance",
            "nearest_obstacle_dist",
            "closest_obstacle_dist",
            "front_obstacle_dist",
            "wall_dist",
            "distance_to_obstacle",
        )
        d_risk_start = 4.0
        d_risk_full = 0.5
        risk_exp = 4.0

        def _distance_to_exp_risk(d: float) -> float:
            if d >= d_risk_start:
                return 0.0
            if d <= d_risk_full:
                return 1.0
            x = (d_risk_start - d) / max(1e-6, (d_risk_start - d_risk_full))
            num = math.exp(risk_exp * x) - 1.0
            den = math.exp(risk_exp) - 1.0
            return float(np.clip(num / max(den, 1e-6), 0.0, 1.0))

        for k in dist_keys:
            if k in info:
                try:
                    d = float(info.get(k, np.inf))
                    if np.isfinite(d):
                        risk = _distance_to_exp_risk(d)
                        info["obstacle_risk_source"] = f"distance:{k}"
                        info.setdefault("obstacle_dist", float(d))
                        return risk
                except Exception:
                    pass

        # lidar note: note 5% note, note.
        lidar = info.get("lidar", None)
        if lidar is not None:
            try:
                arr = np.asarray(lidar, dtype=np.float32).reshape(-1)
                valid = arr[np.isfinite(arr) & (arr > 0.0)]
                if valid.size > 0:
                    d_lidar = float(np.percentile(valid, 5))
                    risk = _distance_to_exp_risk(d_lidar)
                    info["obstacle_risk_source"] = "lidar:p5"
                    info["obstacle_dist"] = float(d_lidar)
                    return risk
            except Exception:
                pass

        info["obstacle_risk_source"] = "none"
        return -1.0

    @staticmethod
    def _zero_reward_parts() -> Dict[str, float]:
        return {
            "survival": 0.0, "speed": 0.0, "cte": 0.0, "collision": 0.0,
            "near_offtrack": 0.0, "near_collision": 0.0,
            "progress": 0.0, "lap": 0.0, "lap_raw": 0.0, "overtake": 0.0, "smooth": 0.0, "jerk": 0.0,
            "mismatch": 0.0, "center": 0.0, "heading": 0.0, "speed_ref": 0.0, "time": 0.0,
            "sat": 0.0, "total": 0.0,
        }

    @staticmethod
    def _zero_episode_diag() -> Dict[str, Any]:
        return {
            "steps_total": 0,
            "cte_abs_samples": [],
            "progress_ratio_signed_sum": 0.0,
            "progress_ratio_forward_sum": 0.0,
            "steps_cte_over_in": 0,
            "steps_cte_over_out": 0,
            "steps_rate_limit_hit": 0,
            "steps_steer_clip_hit": 0,
            "steps_throttle_high_penalty_hit": 0,
            "offtrack_counter_max": 0,
            "stuck_counter_max": 0,
        }

    @staticmethod
    def _signed_arc_ratio(g, idx_prev: int, idx_now: int) -> float:
        """notetracknote, firstnote, note."""
        n = int(g.center.shape[0])
        i0 = int(idx_prev) % n
        i1 = int(idx_now) % n

        if i1 >= i0:
            ds_fwd = float(g.cum_len[i1] - g.cum_len[i0])
        else:
            ds_fwd = float((g.loop_len - g.cum_len[i0]) + g.cum_len[i1])

        if i0 >= i1:
            ds_back = float(g.cum_len[i0] - g.cum_len[i1])
        else:
            ds_back = float((g.loop_len - g.cum_len[i1]) + g.cum_len[i0])

        ds_signed = ds_fwd if ds_fwd <= ds_back else -ds_back
        if not np.isfinite(ds_signed) or g.loop_len <= 1e-6:
            return 0.0

        # note(reset/note)
        max_reasonable = max(3.0, 0.03 * float(g.loop_len))
        if abs(ds_signed) > max_reasonable:
            return 0.0

        return float(ds_signed / float(g.loop_len))

    def _zero_episode_stats(self) -> Dict[str, Any]:
        return {
            "steps": 0,
            "max_speed": 0.0,
            "collision": False,
            "total_reward": 0.0,
            "cte_violations": 0,
            "overtake_count": 0,
        }

    @staticmethod
    def _extract_obstacle_relative_state(info: Dict[str, Any]) -> Tuple[float, float, float, float]:
        try:
            present = float(info.get("obstacle_present", 0.0) or 0.0)
        except Exception:
            present = 0.0
        try:
            longitudinal = float(info.get("obstacle_longitudinal", np.nan))
        except Exception:
            longitudinal = float("nan")
        try:
            lateral = float(info.get("obstacle_lateral", np.nan))
        except Exception:
            lateral = float("nan")
        try:
            planar_distance = float(info.get("obstacle_dist", np.nan))
        except Exception:
            planar_distance = float("nan")
        if (not np.isfinite(planar_distance)) and np.isfinite(longitudinal) and np.isfinite(lateral):
            planar_distance = float(math.hypot(longitudinal, lateral))
        return float(present), float(longitudinal), float(lateral), float(planar_distance)

    def _unwrap_base_env(self):
        base = self.env
        depth = 0
        while hasattr(base, "env") and depth < 32:
            base = base.env
            depth += 1
        return base

    def _clear_base_handler_over(self) -> None:
        try:
            base = self._unwrap_base_env()
            viewer = getattr(base, "viewer", None)
            handler = getattr(viewer, "handler", None)
            if handler is not None:
                handler.over = False
        except Exception:
            pass

    def reset(self, **kwargs):
        self.episode_stats = self._zero_episode_stats()
        self.prev_lap_count = 0
        self._soft_lap_progress = 0.0   # notefirstnote, >=1.0 note
        self._soft_lap_count = 0        # notedetectionnote
        self.stuck_counter  = 0
        self.offtrack_counter = 0
        self._prev_track_idx = None
        self._near_offtrack_ramp_step = 0
        self._near_collision_ramp_step = 0
        self._episode_overtake_count = 0
        self._overtake_front_steps = 0
        self._overtake_armed = False
        self._overtake_cooldown_steps_left = 0
        self._overtake_last_longitudinal = None
        self._reward_parts_episode = self._zero_reward_parts()
        self._episode_diag = self._zero_episode_diag()
        self._episode_index += 1

        obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.current_step += 1

        # note done note(note)
        env_done_before_processing = done
        term_reasons = []
        prev_reason = str(info.get("termination_reason", "") or "").strip()
        if prev_reason and prev_reason!= "normal":
            term_reasons.append(prev_reason)
        episode_step = int(self.episode_stats["steps"]) + 1
        reset_env_done_grace_active = (
            self.reset_env_done_grace_steps > 0
            and episode_step <= self.reset_env_done_grace_steps
        )
        reset_collision_grace_active = (
            self.reset_collision_grace_steps > 0
            and episode_step <= self.reset_collision_grace_steps
        )
        env_done_masked = False
        collision_masked = False
        if env_done_before_processing and reset_env_done_grace_active:
            done = False
            env_done_masked = True
            self._clear_base_handler_over()

        cte_signed = float(info.get("cte", 0))
        cte_abs    = abs(cte_signed)
        speed      = float(info.get("speed", 0) or 0)
        hit        = info.get("hit", "none")
        lap_count  = int(info.get("lap_count", 0) or 0)
        if "ctrl/throttle_cmd_exec" in info:
            throttle_cmd = float(info.get("ctrl/throttle_cmd_exec", 0.0) or 0.0)
        else:
            throttle_cmd = float(action[1]) if len(action) > 1 else 0.0
        prev_track_idx = self._prev_track_idx
        curr_track_idx = None
        kappa_abs = 0.0

        self.episode_stats["max_speed"] = max(self.episode_stats["max_speed"], speed)

        # notegeometrycomputenote lat_err_cte(note _SCENE_CTE_TABLE note)
        # lat_err_cte > 0 = tracknote, lat_err_cte < 0 = tracknote
        # notegeometrynote -cte_signed note(lat_err * coord_scale ~ -sim_cte)
        lat_err_cte = -cte_signed   # fallback
        lat_err_norm = 0.0
        heading_err_abs = 0.0
        if self._track_geometry is not None and self._scene_key:
            try:
                pos = info.get("pos", (0.0, 0.0, 0.0))
                x, z = float(pos[0]), float(pos[2])
                car = info.get("car", (0.0, 0.0, 0.0))
                yaw_deg = float(car[2]) if len(car) >= 3 else 0.0
                yaw_rad = math.radians(yaw_deg)
                geo = self._track_geometry.query(
                    self._scene_key, x=x, z=z, yaw_rad=yaw_rad,
                    prev_idx=self._prev_track_idx,
                )
                curr_track_idx = int(geo["idx"])
                self._prev_track_idx = curr_track_idx
                lat_err = geo["lat_err"]
                lat_err_cte = lat_err * self.coord_scale  # note, note CTE note
                lat_err_norm = float(geo.get("lat_err_norm", 0.0))
                heading_err_abs = abs(float(math.atan2(
                    float(geo.get("heading_err_sin", 0.0)),
                    float(geo.get("heading_err_cos", 1.0)),
                )))
                kappa_abs = abs(float(geo.get("kappa_lookahead", 0.0)))
            except Exception:
                pass

        # note(note)
        lat_err_cte_abs = abs(lat_err_cte)
        is_left_side = (lat_err_cte >= 0)  # note

        # note CTE note(note, note)
        cte_boundary     = abs(self.cte_left     if is_left_side else self.cte_right)
        cte_out_boundary = abs(self.cte_left_out if is_left_side else self.cte_right_out)
        # ontrack: note; note over_in / over_out
        ontrack = float(lat_err_cte_abs <= cte_boundary)
        cte_over_in = float(lat_err_cte_abs > cte_boundary)
        cte_over_out = float(lat_err_cte_abs > cte_out_boundary)

        self._episode_diag["steps_total"] += 1
        self._episode_diag["cte_abs_samples"].append(float(lat_err_cte_abs))
        self._episode_diag["steps_cte_over_in"] += int(cte_over_in > 0.5)
        self._episode_diag["steps_cte_over_out"] += int(cte_over_out > 0.5)

        speed_gate = float(np.clip(speed / 0.5, 0.0, 1.0))
        v_normalized = float(np.clip(speed, 0.0, 4.0) / 4.0)
        cte_norm = lat_err_cte_abs / max(1e-6, cte_boundary)
        center_factor = float(np.clip(1.0 - cte_norm * cte_norm, 0.0, 1.0))

        # Progress reward(notetrackgeometrynotecomputenote, firstnote, note)
        progress_reward = 0.0
        progress_reward_raw = 0.0
        progress_ratio = 0.0
        progress_ratio_unclipped = 0.0  # note, notedetection
        progress_center_gate = 1.0
        progress_forward_gain = 1.0
        if (
            self._track_geometry is not None
            and self._scene_key
            and prev_track_idx is not None
            and curr_track_idx is not None
        ):
            try:
                g = self._track_geometry.scenes[self._scene_key]
                progress_ratio_unclipped = self._signed_arc_ratio(g, int(prev_track_idx), int(curr_track_idx))
                progress_ratio = float(np.clip(progress_ratio_unclipped, -0.02, 0.02))
                curve_ratio = float(np.clip(kappa_abs / self.progress_kappa_ref, 0.0, 1.0))
                if progress_ratio > 0.0:
                    # note: noterewardnotecontrolnote, note"note"CTE/note
                    progress_center_gate = float(max(
                        self.progress_center_gate_min,
                        center_factor ** self.progress_center_gate_power,
                    ))
                    # note, note"note+note"note
                    progress_forward_gain = float(1.0 + self.progress_curve_boost * curve_ratio * center_factor)
                    progress_reward_raw = ontrack * self.progress_reward_scale * progress_ratio * progress_forward_gain
                    progress_reward = progress_reward_raw * progress_center_gate
                else:
                    # note, note
                    progress_reward = ontrack * self.progress_reward_scale * progress_ratio
                    progress_reward_raw = progress_reward
            except Exception:
                progress_ratio = 0.0
                progress_reward_raw = 0.0
                progress_reward = 0.0
        # notegeometrynote(note progress_reward_scale note, notedynamicnotesucceedednote)
        self._episode_diag["progress_ratio_signed_sum"] += float(progress_ratio)
        self._episode_diag["progress_ratio_forward_sum"] += float(max(0.0, progress_ratio))

        # ── notedetection(note WS notetracknote starting-line trigger note)──
        # note unclipped note, notefirstnote >= 1.0 note
        self._soft_lap_progress += float(progress_ratio_unclipped)
        # note
        self._soft_lap_progress = max(-0.5, self._soft_lap_progress)
        while self._soft_lap_progress >= 1.0:
            self._soft_lap_count += 1
            self._soft_lap_progress -= 1.0

        # Survival / speed: note"notefirstnote", note.
        alive_forward_gate = float(progress_ratio > 1e-6)
        survival_reward = self.survival_reward_scale * speed_gate * alive_forward_gate
        speed_reward = 0.25 * ontrack * speed_gate * center_factor * alive_forward_gate

        # note: notepath / note / notegoalnote / note
        center_penalty = -self.w_center * abs(float(lat_err_norm))
        heading_penalty = -self.w_heading * (float(heading_err_abs) / math.pi)
        curve_ratio_speed = float(np.clip(kappa_abs / self.speed_ref_kappa_ref, 0.0, 1.0))
        v_ref = float(
            self.speed_ref_vmax
            - (self.speed_ref_vmax - self.speed_ref_vmin) * curve_ratio_speed
        )
        speed_err_norm = float((speed - v_ref) / max(self.speed_ref_vmax, 1e-6))
        speed_ref_penalty = -self.w_speed_ref * (speed_err_norm * speed_err_norm)
        time_penalty = -self.w_time

        # ★ CTE reward(BUG FIXED: note cte_abs + cte_boundary)
        # V3 note: note cte_norm_scale notetrack CTE note
        #   norm_scale = cte_half_width / REF -> notetrack <1(note), notetrack >1(note)
        if lat_err_cte_abs > cte_boundary:
            # note(note)= lat_err_cte - note(note=note, note=note)
            # note cte_half_width(note)note boundary,
            # notetrack(wt/gt/wh note)note
            exceed_ratio = (lat_err_cte_abs - cte_boundary) / max(1e-6, self.cte_half_width)
            # clip exceed_ratio note
            exceed_ratio = min(exceed_ratio, 2.0)
            cte_term = -(1.0 + 4.0 * exceed_ratio) * self.cte_norm_scale
            self.episode_stats["cte_violations"] += 1
        else:
            cte_base = 0.3 * (1.0 - lat_err_cte_abs / max(1e-6, cte_boundary))
            speed_gate_cte = float(np.clip(speed / 0.3, 0.0, 1.0))
            cte_term = cte_base * speed_gate_cte * self.cte_norm_scale

        # Terminal penalty(note; near_* note)
        terminal_penalty = 0.0
        if hit!= "none":
            if reset_collision_grace_active:
                collision_masked = True
                if env_done_before_processing:
                    done = False
                    env_done_masked = True
                self._clear_base_handler_over()
            else:
                terminal_penalty = -self.collision_penalty_base
                self.episode_stats["collision"] = True
                done = True
                term_reasons.append("collision")

        # Lap reward(note sim note, note)
        effective_lap_count = max(lap_count, self._soft_lap_count)
        lap_reward = 0.0
        lap_reward_raw = 0.0
        # WSnoteobstacleepisodenote:
        # noteobstaclenoteep_len~200note, noteobstaclenoteep_len~1300note, note(noteobstaclenote18%note)
        # noteobstacleWSnote3notedone, noteclassepisodenote, noteobstaclenote
        obstacle_active = float(info.get("obstacle_runtime_active", 1.0))
        if self._scene_key == "waveshare" and obstacle_active < 0.5:
            MAX_LAPS_FOR_REWARD = 3  # WSnoteobstacle: 3notedone
        else:
            MAX_LAPS_FOR_REWARD = 5  # WSnoteobstacle / GT: 5notedone
        if effective_lap_count > self.prev_lap_count and effective_lap_count <= MAX_LAPS_FOR_REWARD:
            # note 1 note, noterewardnote
            laps_completed_raw = effective_lap_count - self.prev_lap_count
            laps_completed = int(max(0, min(laps_completed_raw, 1)))
            lap_reward_raw = 6.0 * laps_completed
            lap_reward = lap_reward_raw * self.lap_reward_scale
            self.prev_lap_count = effective_lap_count
            lap_source = "sim" if lap_count >= self._soft_lap_count else "soft"
            print(
                f"\n🎉 [{self._logging_key}] note {effective_lap_count} note ({lap_source})! "
                f"reward +{lap_reward:.1f} (raw={lap_reward_raw:.1f}, "
                f"scale={self.lap_reward_scale:.2f}, "
                f"sim_lap={lap_count}, soft_lap={self._soft_lap_count})"
            )
        elif effective_lap_count > MAX_LAPS_FOR_REWARD:
            # note, notereward, noteprev_lap_countnote
            self.prev_lap_count = effective_lap_count

        # ★ note episode
        if effective_lap_count >= MAX_LAPS_FOR_REWARD:
            done = True
            term_reasons.append("max_laps_reached")

        # Stuck detection(note < 0.1, note 30 note)
        if ontrack and speed < 0.1:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        if self.stuck_counter > 30:
            done = True
            stuck_penalty_increment = min(3.0, 0.3 * (self.stuck_counter - 30))
            terminal_penalty -= stuck_penalty_increment
            term_reasons.append("stuck")

        # note: done notefirstnote, note; CTE note(cte_term)note
        if self._leniency_steps > 0 and self.current_step < self._leniency_steps:
            _progress = self.current_step / self._leniency_steps          # 0 -> 1
            _leniency = self._leniency_mult * (1.0 - _progress) + _progress  # mult -> 1.0
        else:
            _leniency = 1.0
        _effective_out = cte_out_boundary * _leniency

        if lat_err_cte_abs > _effective_out:
            self.offtrack_counter += 1
        else:
            self.offtrack_counter = 0

        # Offtrack done(note -6 note, done note)
        if self.offtrack_counter >= 3:
            terminal_penalty -= self.offtrack_penalty_base
            done = True
            term_reasons.append("offtrack")

        # note"note": note/notefirstnote, note.
        # note: note"note out note", note done note leniency note.
        cte_out_ratio_done = float(np.clip(lat_err_cte_abs / max(_effective_out, 1e-6), 0.0, 2.0))
        cte_out_ratio_risk = float(np.clip(lat_err_cte_abs / max(cte_out_boundary, 1e-6), 0.0, 2.0))

        near_offtrack_ratio = float(np.clip(
            (cte_out_ratio_risk - self.near_offtrack_start_ratio)
            / max(1e-6, 1.0 - self.near_offtrack_start_ratio),
            0.0,
            1.0,
        ))
        if near_offtrack_ratio > 1e-6:
            self._near_offtrack_ramp_step = min(
                self.near_penalty_ramp_steps,
                self._near_offtrack_ramp_step + 1,
            )
        else:
            self._near_offtrack_ramp_step = 0
        near_offtrack_ramp_scale = float(self._near_offtrack_ramp_step / max(1, self.near_penalty_ramp_steps))
        near_offtrack_penalty = -self.w_near_offtrack * near_offtrack_ratio * near_offtrack_ramp_scale

        heading_risk = float(np.clip(heading_err_abs / (0.70 * math.pi), 0.0, 1.0))
        speed_risk = float(np.clip(speed / max(self.speed_ref_vmax, 1e-6), 0.0, 1.0))
        near_collision_ratio = float(np.clip(
            (cte_out_ratio_risk - self.near_collision_start_ratio)
            / max(1e-6, 1.0 - self.near_collision_start_ratio),
            0.0,
            1.0,
        ))
        control_risk = 0.0
        if self.action_safety_wrapper is not None:
            try:
                diag = self.action_safety_wrapper.diag
                control_risk = float(max(
                    float(diag.get("rate_excess_bounded", 0.0)),
                    float(diag.get("steer_clip_hit", 0.0)),
                ))
            except Exception:
                control_risk = 0.0
        obstacle_risk = self._extract_obstacle_risk(info)
        has_obstacle_signal = float(obstacle_risk >= 0.0)
        if obstacle_risk < 0.0:
            obstacle_risk = 0.0
        proxy_collision_risk = float(np.clip(
            0.30 * near_collision_ratio + 0.30 * heading_risk + 0.20 * speed_risk + 0.20 * control_risk,
            0.0,
            1.0,
        ))
        # noteobstaclenote; note.
        if has_obstacle_signal > 0.5:
            near_collision_risk_raw = float(np.clip(
                0.75 * obstacle_risk + 0.25 * proxy_collision_risk,
                0.0,
                1.0,
            ))
        else:
            near_collision_risk_raw = 0.50 * proxy_collision_risk

        near_collision_trigger = float(max(
            0.50 * near_collision_ratio,
            heading_risk,
            control_risk,
            obstacle_risk,
        ))
        if near_collision_trigger > 1e-6:
            self._near_collision_ramp_step = min(
                self.near_penalty_ramp_steps,
                self._near_collision_ramp_step + 1,
            )
        else:
            self._near_collision_ramp_step = 0

        near_collision_ramp_scale = float(self._near_collision_ramp_step / max(1, self.near_penalty_ramp_steps))
        near_collision_risk = near_collision_risk_raw * near_collision_ramp_scale
        near_collision_penalty = -self.w_near_collision * near_collision_risk

        info["reward_debug/offtrack_leniency"]  = _leniency
        info["reward_debug/effective_cte_out"]  = _effective_out
        info["reward_debug/cte_out_ratio"] = float(cte_out_ratio_risk)
        info["reward_debug/cte_out_ratio_done"] = float(cte_out_ratio_done)
        info["reward_debug/near_offtrack_ratio"] = float(near_offtrack_ratio)
        info["reward_debug/near_offtrack_ramp_scale"] = float(near_offtrack_ramp_scale)
        info["reward_debug/near_collision_ramp_scale"] = float(near_collision_ramp_scale)
        info["reward_debug/near_offtrack_ramp_step"] = float(self._near_offtrack_ramp_step)
        info["reward_debug/near_collision_ramp_step"] = float(self._near_collision_ramp_step)
        info["reward_debug/r_near_offtrack"] = float(near_offtrack_penalty)
        info["reward_debug/near_collision_proxy_risk"] = float(proxy_collision_risk)
        info["reward_debug/near_collision_obstacle_risk"] = float(obstacle_risk)
        info["reward_debug/near_collision_has_obstacle_signal"] = float(has_obstacle_signal)
        info["reward_debug/obstacle_dist"] = float(info.get("obstacle_dist", np.nan))
        info["reward_debug/obstacle_risk_source"] = str(info.get("obstacle_risk_source", "none"))
        info["reward_debug/near_collision_risk"] = float(near_collision_risk)
        info["reward_debug/r_near_collision"] = float(near_collision_penalty)

        overtake_bonus = 0.0
        overtake_success = False
        obstacle_present, obstacle_longitudinal, _obstacle_lateral, obstacle_planar_distance = (
            self._extract_obstacle_relative_state(info)
        )
        obstacle_available = bool(obstacle_present > 0.5 and np.isfinite(obstacle_longitudinal))
        obstacle_planar_ok = bool(np.isfinite(obstacle_planar_distance) and obstacle_planar_distance > 0.0)
        if self._overtake_cooldown_steps_left > 0:
            self._overtake_cooldown_steps_left -= 1

        encounter_front = bool(
            obstacle_available
            and obstacle_longitudinal >= self.overtake_arm_longitudinal_min_m
            and (
                (not obstacle_planar_ok)
                or obstacle_planar_distance <= self.overtake_arm_planar_max_m
            )
        )
        if encounter_front:
            self._overtake_front_steps += 1
            if (
                self._overtake_front_steps >= self.overtake_min_front_steps
                and self._overtake_cooldown_steps_left <= 0
            ):
                self._overtake_armed = True
        elif not self._overtake_armed:
            self._overtake_front_steps = 0

        safe_overtake_state = bool(
            (not done)
            and (cte_over_out <= 0.5)
            and (speed > 0.2)
            and ("collision" not in term_reasons)
            and ("offtrack" not in term_reasons)
            and ("stuck" not in term_reasons)
        )
        passed_to_back = bool(
            obstacle_available
            and obstacle_longitudinal <= self.overtake_pass_longitudinal_threshold_m
            and (
                (not obstacle_planar_ok)
                or obstacle_planar_distance >= self.overtake_pass_planar_min_m
            )
        )
        crossed_from_front = bool(
            self._overtake_last_longitudinal is not None
            and self._overtake_last_longitudinal >= 0.0
        )
        if (
            self.overtake_success_bonus > 0.0
            and self._overtake_armed
            and safe_overtake_state
            and passed_to_back
            and (crossed_from_front or self._overtake_front_steps >= self.overtake_min_front_steps)
        ):
            overtake_bonus = float(self.overtake_success_bonus)
            overtake_success = True
            self._episode_overtake_count += 1
            self.episode_stats["overtake_count"] = int(self._episode_overtake_count)
            self._overtake_armed = False
            self._overtake_front_steps = 0
            self._overtake_cooldown_steps_left = self.overtake_rearm_cooldown_steps
        elif done and (("collision" in term_reasons) or ("offtrack" in term_reasons) or ("stuck" in term_reasons)):
            self._overtake_armed = False
            self._overtake_front_steps = 0

        self._overtake_last_longitudinal = (
            float(obstacle_longitudinal) if obstacle_available else None
        )
        info["overtake_success"] = bool(overtake_success)
        info["overtake_bonus"] = float(overtake_bonus)
        info["overtake_count"] = int(self._episode_overtake_count)
        info["reward_debug/overtake_armed"] = float(self._overtake_armed)
        info["reward_debug/overtake_front_steps"] = float(self._overtake_front_steps)
        info["reward_debug/overtake_cooldown"] = float(self._overtake_cooldown_steps_left)
        info["reward_debug/overtake_obstacle_longitudinal"] = float(obstacle_longitudinal)
        info["reward_debug/overtake_obstacle_planar_distance"] = float(obstacle_planar_distance)
        info["reward_debug/r_overtake"] = float(overtake_bonus)

        # note
        info["reward_debug/survival"]         = survival_reward
        info["reward_debug/speed_gate"]       = speed_gate
        info["reward_debug/alive_forward_gate"] = alive_forward_gate
        info["reward_debug/center_factor"]    = center_factor
        info["reward_debug/stuck_counter"]    = self.stuck_counter
        info["reward_debug/cte_boundary"]     = cte_boundary
        info["reward_debug/cte_out_boundary"] = cte_out_boundary
        info["reward_debug/offtrack_counter"] = self.offtrack_counter
        info["reward_debug/lat_err_cte"]      = float(lat_err_cte)
        info["reward_debug/cte_abs"]          = float(lat_err_cte_abs)
        info["reward_debug/cte_over_in"]      = float(cte_over_in)
        info["reward_debug/cte_over_out"]     = float(cte_over_out)
        info["reward_debug/reset_env_done_grace_active"] = float(reset_env_done_grace_active)
        info["reward_debug/reset_collision_grace_active"] = float(reset_collision_grace_active)
        info["reward_debug/reset_env_done_masked"] = float(env_done_masked)
        info["reward_debug/reset_collision_masked"] = float(collision_masked)
        self._episode_diag["offtrack_counter_max"] = max(
            int(self._episode_diag["offtrack_counter_max"]),
            int(self.offtrack_counter),
        )
        self._episode_diag["stuck_counter_max"] = max(
            int(self._episode_diag["stuck_counter_max"]),
            int(self.stuck_counter),
        )

        # note
        smooth_penalty = 0.0
        jerk_penalty   = 0.0
        mismatch_penalty = 0.0
        sat_penalty    = 0.0
        rate_limit_hit = 0.0
        steer_clip_hit = 0.0
        curve_ratio_for_penalty = float(np.clip(kappa_abs / self.progress_kappa_ref, 0.0, 1.0))
        curve_penalty_scale = float(max(0.35, 1.0 - self.smooth_curve_relief * curve_ratio_for_penalty))
        if self.action_safety_wrapper is not None:
            diag = self.action_safety_wrapper.diag
            abs_delta             = abs(diag["delta_steer"])
            abs_jerk              = abs(diag["delta_steer"] - diag["delta_steer_prev"])
            abs_mismatch          = abs(diag["mismatch"])
            rate_excess_bounded   = float(diag["rate_excess_bounded"])
            rate_limit_hit        = float(diag["rate_limit_hit"])
            steer_clip_hit        = float(diag["steer_clip_hit"])

            # note, note"note"
            smooth_penalty = -self.w_d   * abs_delta * curve_penalty_scale
            jerk_penalty   = -self.w_dd  * abs_jerk * curve_penalty_scale
            mismatch_penalty = -self.w_m * abs_mismatch * curve_penalty_scale
            sat_penalty    = -self.w_sat * rate_excess_bounded
            self._episode_diag["steps_rate_limit_hit"] += int(rate_limit_hit > 0.5)
            self._episode_diag["steps_steer_clip_hit"] += int(steer_clip_hit > 0.5)

            self.smooth_stats.append({
                "abs_delta":            abs_delta,
                "abs_jerk":             abs_jerk,
                "abs_mismatch":         abs_mismatch,
                "rate_limit_hit":       rate_limit_hit,
                "rate_excess_raw":      float(diag["rate_excess_raw"]),
                "rate_excess_bounded":  rate_excess_bounded,
                "steer_clip_hit":       steer_clip_hit,
            })
            info["smooth/abs_delta_steer"]      = abs_delta
            info["smooth/rate_limit_hit"]        = rate_limit_hit
            info["smooth/rate_excess_raw"]       = float(diag["rate_excess_raw"])
            info["smooth/rate_excess_bounded"]   = rate_excess_bounded
            info["smooth/steer_clip_hit"]        = steer_clip_hit
            info["smooth/abs_mismatch"]          = abs_mismatch
            info["smooth/abs_jerk"]              = abs_jerk
            info["smooth/hairpin_relax_active"]  = float(diag.get("hairpin_relax_active", 0.0))

        info["reward_debug/progress_ratio"] = float(progress_ratio)
        info["reward_debug/progress_reward_raw"] = float(progress_reward_raw)
        info["reward_debug/progress_reward"] = float(progress_reward)
        info["reward_debug/progress_center_gate"] = float(progress_center_gate)
        info["reward_debug/progress_forward_gain"] = float(progress_forward_gain)
        info["reward_debug/progress_curve_ratio"] = float(curve_ratio_for_penalty)
        info["reward_debug/curve_penalty_scale"] = float(curve_penalty_scale)
        info["reward_debug/lat_err_norm"] = float(lat_err_norm)
        info["reward_debug/heading_err_abs"] = float(heading_err_abs)
        info["reward_debug/v_ref"] = float(v_ref)
        info["reward_debug/speed_ref_err_norm"] = float(speed_err_norm)
        info["reward_debug/r_center"] = float(center_penalty)
        info["reward_debug/r_heading"] = float(heading_penalty)
        info["reward_debug/r_speed_ref"] = float(speed_ref_penalty)
        info["reward_debug/r_time"] = float(time_penalty)

        throttle_high_penalty = 0.0
        if throttle_cmd > self.throttle_penalty_threshold:
            speed_norm_for_penalty = float(np.clip(speed / 4.0, 0.0, 2.0))
            # note: note(note, note)
            throttle_high_penalty = -self.throttle_penalty_amount * (1.0 + speed_norm_for_penalty)
        else:
            speed_norm_for_penalty = float(np.clip(speed / 4.0, 0.0, 2.0))
        throttle_high_penalty_hit = float(throttle_high_penalty < 0.0)
        self._episode_diag["steps_throttle_high_penalty_hit"] += int(throttle_high_penalty_hit > 0.5)
        info["reward_debug/throttle_cmd"] = float(throttle_cmd)
        info["reward_debug/speed_norm_for_throttle_penalty"] = float(speed_norm_for_penalty)
        info["reward_debug/throttle_high_penalty"] = float(throttle_high_penalty)
        info["reward_debug/throttle_high_penalty_hit"] = throttle_high_penalty_hit

        total_reward = (
            survival_reward + speed_reward + progress_reward + cte_term +
            center_penalty + heading_penalty + speed_ref_penalty + time_penalty +
            terminal_penalty + near_offtrack_penalty + near_collision_penalty + lap_reward +
            overtake_bonus + smooth_penalty + jerk_penalty + mismatch_penalty + sat_penalty +
            throttle_high_penalty
        )

        # reward decay: note ref_steps note ref/step notereward
        ep_steps = self.episode_stats["steps"] + 1   # currentnote(note1note)
        if self.reward_decay_ref_steps > 0 and ep_steps > self.reward_decay_ref_steps:
            total_reward /= (ep_steps / self.reward_decay_ref_steps)

        self.episode_stats["total_reward"] += total_reward
        self.episode_stats["steps"] += 1

        # ── rewardnote(note ep_info_buffer -> PerSceneStatsCallback note)──
        self._reward_parts_episode["survival"]  += survival_reward
        self._reward_parts_episode["speed"]     += speed_reward
        self._reward_parts_episode["progress"]  += progress_reward
        self._reward_parts_episode["cte"]       += cte_term
        self._reward_parts_episode["center"]    += center_penalty
        self._reward_parts_episode["heading"]   += heading_penalty
        self._reward_parts_episode["speed_ref"] += speed_ref_penalty
        self._reward_parts_episode["time"]      += time_penalty
        self._reward_parts_episode["collision"] += terminal_penalty
        self._reward_parts_episode["near_offtrack"] += near_offtrack_penalty
        self._reward_parts_episode["near_collision"] += near_collision_penalty
        self._reward_parts_episode["lap"]       += lap_reward
        self._reward_parts_episode["lap_raw"]   += lap_reward_raw
        self._reward_parts_episode["overtake"]  += overtake_bonus
        self._reward_parts_episode["smooth"]    += smooth_penalty
        self._reward_parts_episode["jerk"]      += jerk_penalty
        self._reward_parts_episode["mismatch"]  += mismatch_penalty
        self._reward_parts_episode["sat"]       += sat_penalty
        self._reward_parts_episode["total"]     += total_reward

        if done:
            info["ep_r_survival"]  = self._reward_parts_episode["survival"]
            info["ep_r_speed"]     = self._reward_parts_episode["speed"]
            info["ep_r_progress"]  = self._reward_parts_episode["progress"]
            info["ep_r_cte"]       = self._reward_parts_episode["cte"]
            info["ep_r_center"]    = self._reward_parts_episode["center"]
            info["ep_r_heading"]   = self._reward_parts_episode["heading"]
            info["ep_r_speed_ref"] = self._reward_parts_episode["speed_ref"]
            info["ep_r_time"]      = self._reward_parts_episode["time"]
            info["ep_r_collision"] = self._reward_parts_episode["collision"]
            info["ep_r_near_offtrack"] = self._reward_parts_episode["near_offtrack"]
            info["ep_r_near_collision"] = self._reward_parts_episode["near_collision"]
            info["ep_r_lap"]       = self._reward_parts_episode["lap"]
            info["ep_r_lap_raw"]   = self._reward_parts_episode["lap_raw"]
            info["ep_r_overtake"]  = self._reward_parts_episode["overtake"]
            info["ep_overtake_count"] = int(self._episode_overtake_count)
            info["ep_soft_lap_count"] = self._soft_lap_count
            info["ep_r_smooth"]    = self._reward_parts_episode["smooth"]
            info["ep_r_jerk"]      = self._reward_parts_episode["jerk"]
            info["ep_r_mismatch"]  = self._reward_parts_episode["mismatch"]
            info["ep_r_sat"]       = self._reward_parts_episode["sat"]
            info["ep_r_total"]     = self._reward_parts_episode["total"]
            diag_steps = max(1, int(self._episode_diag["steps_total"]))
            cte_samples = np.asarray(self._episode_diag["cte_abs_samples"], dtype=np.float64)
            if cte_samples.size > 0:
                info["ep_cte_abs_p50"] = float(np.percentile(cte_samples, 50))
                info["ep_cte_abs_p90"] = float(np.percentile(cte_samples, 90))
                info["ep_cte_abs_p99"] = float(np.percentile(cte_samples, 99))
            else:
                info["ep_cte_abs_p50"] = 0.0
                info["ep_cte_abs_p90"] = 0.0
                info["ep_cte_abs_p99"] = 0.0
            info["ep_cte_over_in_rate"] = float(self._episode_diag["steps_cte_over_in"] / diag_steps)
            info["ep_cte_over_out_rate"] = float(self._episode_diag["steps_cte_over_out"] / diag_steps)
            info["ep_rate_limit_hit_rate"] = float(self._episode_diag["steps_rate_limit_hit"] / diag_steps)
            info["ep_steer_clip_hit_rate"] = float(self._episode_diag["steps_steer_clip_hit"] / diag_steps)
            info["ep_throttle_high_penalty_hit_rate"] = float(
                self._episode_diag["steps_throttle_high_penalty_hit"] / diag_steps
            )
            info["ep_offtrack_counter_max"] = float(self._episode_diag["offtrack_counter_max"])
            info["ep_stuck_counter_max"] = float(self._episode_diag["stuck_counter_max"])
            info["ep_progress_ratio_signed_sum"] = float(self._episode_diag["progress_ratio_signed_sum"])
            info["ep_progress_ratio_forward_sum"] = float(self._episode_diag["progress_ratio_forward_sum"])
            info["ep_progress_reward_scale"] = float(self.progress_reward_scale)

        # note: firstnote(defaultnote, notetrainingnote)
        _diag_episode_hit = (
            self.step_diagnostics_every_episodes <= 0
            or (self._episode_index % self.step_diagnostics_every_episodes == 0)
        )
        if (
            self.enable_step_diagnostics
            and _diag_episode_hit
            and self.episode_stats["steps"] <= self.step_diagnostics_first_steps
        ):
            side_str = "L" if is_left_side else "R"
            reason_preview = (
                prev_reason
                if prev_reason
                else ("env_done" if env_done_before_processing else "normal")
            )
            print(
                f"🔍 [{self._logging_key}] ep={self._episode_index} step={self.episode_stats['steps']}: "
                f"lat_err_cte={lat_err_cte:.3f} side={side_str} "
                f"(in={cte_boundary:.2f}, out={cte_out_boundary:.2f}), "
                f"speed={speed:.2f}, hit={hit}, done={done}, "
                f"env_done={env_done_before_processing}, reason={reason_preview}"
            )

        if term_reasons:
            dedup = []
            for r in term_reasons:
                if r and r not in dedup:
                    dedup.append(r)
            info["termination_reason"] = "+".join(dedup)
        else:
            if env_done_before_processing and (not env_done_masked):
                info.setdefault("termination_reason", "env_done")
            else:
                info.setdefault("termination_reason", "normal")
        if done:
            reason_tokens = set(str(info.get("termination_reason", "normal")).split("+"))
            info["ep_term_collision"] = float("collision" in reason_tokens)
            info["ep_term_stuck"] = float("stuck" in reason_tokens)
            info["ep_term_offtrack"] = float("offtrack" in reason_tokens)
            info["ep_term_env_done"] = float("env_done" in reason_tokens)
            info["ep_term_normal"] = float(
                ("normal" in reason_tokens)
                and ("collision" not in reason_tokens)
                and ("stuck" not in reason_tokens)
                and ("offtrack" not in reason_tokens)
                and ("env_done" not in reason_tokens)
            )
        return obs, total_reward, done, info



# ---------------------------------------------------------------------------
# note(note ppo_waveshare_v8/v9/test note)
# ---------------------------------------------------------------------------
ImprovedRewardWrapperV3 = DonkeyRewardWrapper
V9DomainRewardWrapper   = DonkeyRewardWrapper
