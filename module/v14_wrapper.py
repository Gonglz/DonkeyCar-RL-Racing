#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 wrapper main class assembled from mixins."""

import math
import random
import time
from collections import deque
from typing import Optional

import numpy as np

from .v14_dep_generatedtrack_base import (
    ManualWidthSpawnSampler,
    GeneratedTrackV11_1Wrapper,
)

from .v14_racing_line import RacingLineComputer
from .v14_distillation import PolicyDistillationManager
from .v14_stage_profiles import STAGE_TRAIN_PROFILES_V14
from .v14_wrapper_spawn import V14SpawnResetMixin
from .v14_wrapper_npc import V14NpcRuntimeMixin
from .v14_wrapper_reward import V14RewardMixin
from .v14_paths import default_generated_track_profile


class GeneratedTrackV14Wrapper(V14RewardMixin, V14NpcRuntimeMixin, V14SpawnResetMixin, GeneratedTrackV11_1Wrapper):
    """
    V14 Wrapper: 继承V11_1的控制层，新增:
    - 赛车线奖励 (Stage 1)
    - 主动避障奖励 (Stage 2)
    - 动态避让奖励 (Stage 3)
    - 混沌+防遗忘 (Stage 4)
    - 混合采样逻辑

    MRO 固定顺序:
    V14RewardMixin -> V14NpcRuntimeMixin -> V14SpawnResetMixin -> GeneratedTrackV11_1Wrapper
    """

    def __init__(self, *args, **kwargs):
        # V14 特有参数
        self.manual_width_profile = str(
            kwargs.pop("manual_width_profile",
                       default_generated_track_profile())
        )
        self.racing_line_lookahead = int(kwargs.pop("racing_line_lookahead", 20))
        self.racing_line_smoothing = int(kwargs.pop("racing_line_smoothing", 10))
        self.racing_line_max_offset_ratio = float(kwargs.pop("racing_line_max_offset_ratio", 0.35))
        self.proactive_zone_scale = float(kwargs.pop("proactive_zone_scale", 2.0))
        self.proactive_reward_scale = float(kwargs.pop("proactive_reward_scale", 0.30))
        self.close_call_penalty = float(kwargs.pop("close_call_penalty", 0.30))
        self.safe_passage_bonus = float(kwargs.pop("safe_passage_bonus", 0.40))
        self.dynamic_overtake_bonus = float(kwargs.pop("dynamic_overtake_bonus", 1.0))
        self.v14_overtake_bonus_growth = float(kwargs.pop("v14_overtake_bonus_growth", 0.35))
        self.v14_overtake_bonus_max_mult = float(kwargs.pop("v14_overtake_bonus_max_mult", 4.0))
        self.v14_collision_extra_penalty = float(kwargs.pop("v14_collision_extra_penalty", 2.0))
        self.v14_lap_complete_bonus = float(kwargs.pop("v14_lap_complete_bonus", 1.2))
        self.v14_lap_time_reward_scale = float(kwargs.pop("v14_lap_time_reward_scale", 1.2))
        self.v14_lap_time_ref_sec = float(kwargs.pop("v14_lap_time_ref_sec", 23.0))
        self.v14_lap_time_reward_max_ratio = float(kwargs.pop("v14_lap_time_reward_max_ratio", 2.5))
        self.v14_speed_maintain_bonus_scale = float(kwargs.pop("v14_speed_maintain_bonus_scale", 0.12))
        self.rear_end_penalty = float(kwargs.pop("rear_end_penalty", 1.2))
        self.racing_line_weight_stage4 = float(kwargs.pop("racing_line_weight_stage4", 0.5))
        self.curvature_penalty_scale = float(kwargs.pop("curvature_penalty_scale", 0.10))
        self.curvature_penalty_speed_thresh = float(kwargs.pop("curvature_penalty_speed_thresh", 0.5))
        self.kappa_ref_max_for_penalty = float(kwargs.pop("kappa_ref_max_for_penalty", 2.0))
        self.racing_line_reward_scale = float(kwargs.pop("racing_line_reward_scale", 0.15))
        self.p_npc_free = float(kwargs.pop("p_npc_free", 0.0))
        self.lap_time_target_speed = float(kwargs.pop("lap_time_target_speed", 1.0))
        self.npc_spawn_ahead_min_sim = float(kwargs.pop("npc_spawn_ahead_min_sim", 6.0))
        self.npc_spawn_ahead_max_sim = float(kwargs.pop("npc_spawn_ahead_max_sim", 12.0))
        self.npc_spawn_min_learner_dist_sim = float(kwargs.pop("npc_spawn_min_learner_dist_sim", 3.2))
        self.npc_spawn_min_npc_dist_sim = float(kwargs.pop("npc_spawn_min_npc_dist_sim", 2.8))
        self.npc_spawn_min_npc_progress_gap_idx = int(kwargs.pop("npc_spawn_min_npc_progress_gap_idx", 90))
        self.npc_spawn_candidate_tries = int(kwargs.pop("npc_spawn_candidate_tries", 36))
        self.v14_global_warmup_no_npc = bool(kwargs.pop("v14_global_warmup_no_npc", True))
        self.v14_trace_buffer_maxlen = int(kwargs.pop("v14_trace_buffer_maxlen", 6000))
        self.v14_trace_window = int(kwargs.pop("v14_trace_window", 1600))
        self.v14_trace_min_points = int(kwargs.pop("v14_trace_min_points", 40))
        self.v14_npc_spawn_randomize = bool(kwargs.pop("v14_npc_spawn_randomize", True))
        self.v14_cte_done_relax = float(kwargs.pop("v14_cte_done_relax", 0.0))
        self.v14_avoidance_success_bonus = float(kwargs.pop("v14_avoidance_success_bonus", 0.45))
        self.v14_npc_random_heading = bool(kwargs.pop("v14_npc_random_heading", True))
        self.v14_npc_wobble_in_place = bool(kwargs.pop("v14_npc_wobble_in_place", False))
        self.v14_npc_wobble_radius_sim = float(kwargs.pop("v14_npc_wobble_radius_sim", 0.10))
        self.v14_npc_wobble_period_steps = int(kwargs.pop("v14_npc_wobble_period_steps", 32))
        self.v14_npc_wobble_update_every_steps = int(kwargs.pop("v14_npc_wobble_update_every_steps", 6))
        self.v14_npc_wobble_yaw_jitter_deg = float(kwargs.pop("v14_npc_wobble_yaw_jitter_deg", 18.0))
        self.v14_reposition_overpass_gap_idx = int(kwargs.pop("v14_reposition_overpass_gap_idx", 90))
        self.v14_reposition_far_factor = float(kwargs.pop("v14_reposition_far_factor", 5.0))
        self.v14_reposition_min_step_gap = int(kwargs.pop("v14_reposition_min_step_gap", 80))
        self.v14_reward_forward_only = bool(kwargs.pop("v14_reward_forward_only", True))
        self.v14_npc_radius_penalty_radius_sim = float(
            kwargs.pop("v14_npc_radius_penalty_radius_sim", 1.5)
        )
        self.v14_npc_radius_penalty_per_step = float(
            kwargs.pop("v14_npc_radius_penalty_per_step", 0.12)
        )
        self.v14_stage34_npc_speed_ratio_min = float(kwargs.pop("v14_stage34_npc_speed_ratio_min", 0.30))
        self.v14_stage34_npc_speed_ratio_max = float(kwargs.pop("v14_stage34_npc_speed_ratio_max", 0.80))
        self.v14_stage34_collision_reset_npc = bool(kwargs.pop("v14_stage34_collision_reset_npc", True))
        self.v14_stage34_npc_contact_reset_cooldown_steps = int(
            kwargs.pop("v14_stage34_npc_contact_reset_cooldown_steps", 60)
        )
        if self.npc_spawn_ahead_max_sim < self.npc_spawn_ahead_min_sim:
            self.npc_spawn_ahead_min_sim, self.npc_spawn_ahead_max_sim = (
                self.npc_spawn_ahead_max_sim,
                self.npc_spawn_ahead_min_sim,
            )
        if (not np.isfinite(self.v14_stage34_npc_speed_ratio_min)) or self.v14_stage34_npc_speed_ratio_min < 0.0:
            self.v14_stage34_npc_speed_ratio_min = 0.30
        if (not np.isfinite(self.v14_stage34_npc_speed_ratio_max)) or self.v14_stage34_npc_speed_ratio_max <= 0.0:
            self.v14_stage34_npc_speed_ratio_max = 0.80
        if self.v14_stage34_npc_speed_ratio_max < self.v14_stage34_npc_speed_ratio_min:
            self.v14_stage34_npc_speed_ratio_min, self.v14_stage34_npc_speed_ratio_max = (
                self.v14_stage34_npc_speed_ratio_max,
                self.v14_stage34_npc_speed_ratio_min,
            )
        if (not np.isfinite(self.v14_npc_radius_penalty_radius_sim)) or self.v14_npc_radius_penalty_radius_sim < 0.0:
            self.v14_npc_radius_penalty_radius_sim = 1.5
        if (not np.isfinite(self.v14_npc_radius_penalty_per_step)) or self.v14_npc_radius_penalty_per_step < 0.0:
            self.v14_npc_radius_penalty_per_step = 0.12
        self.v14_stage34_npc_contact_reset_cooldown_steps = max(
            1, int(self.v14_stage34_npc_contact_reset_cooldown_steps)
        )
        if (not np.isfinite(self.v14_overtake_bonus_growth)) or self.v14_overtake_bonus_growth < 0.0:
            self.v14_overtake_bonus_growth = 0.0
        if (not np.isfinite(self.v14_overtake_bonus_max_mult)) or self.v14_overtake_bonus_max_mult < 1.0:
            self.v14_overtake_bonus_max_mult = 1.0
        if (not np.isfinite(self.v14_collision_extra_penalty)) or self.v14_collision_extra_penalty < 0.0:
            self.v14_collision_extra_penalty = 0.0
        if (not np.isfinite(self.v14_lap_complete_bonus)) or self.v14_lap_complete_bonus < 0.0:
            self.v14_lap_complete_bonus = 0.0
        if (not np.isfinite(self.v14_lap_time_reward_scale)) or self.v14_lap_time_reward_scale < 0.0:
            self.v14_lap_time_reward_scale = 0.0
        if (not np.isfinite(self.v14_lap_time_ref_sec)) or self.v14_lap_time_ref_sec <= 0.0:
            self.v14_lap_time_ref_sec = 23.0
        if (not np.isfinite(self.v14_lap_time_reward_max_ratio)) or self.v14_lap_time_reward_max_ratio < 0.5:
            self.v14_lap_time_reward_max_ratio = 2.5
        if (not np.isfinite(self.v14_speed_maintain_bonus_scale)) or self.v14_speed_maintain_bonus_scale < 0.0:
            self.v14_speed_maintain_bonus_scale = 0.12

        super().__init__(*args, **kwargs)

        # 赛车线计算器
        self.manual_spawn_v14 = ManualWidthSpawnSampler(self.manual_width_profile, self.scene_name)
        self.racing_line = RacingLineComputer(
            self.manual_spawn_v14,
            lookahead_window=self.racing_line_lookahead,
            smoothing_window=self.racing_line_smoothing,
            max_offset_ratio=self.racing_line_max_offset_ratio,
        )
        if self.racing_line.loaded:
            print(f"   ✅ V14 RacingLine loaded: {self.manual_spawn_v14.fine_track.shape[0]} points")
        else:
            print(f"   ⚠️ V14 RacingLine not loaded")

        # 主动避障跟踪状态
        self._npc_lateral_dist_history = deque(maxlen=10)  # 兼容字段（弃用）
        self._npc_lateral_dist_histories = {}
        self._episode_npc_free = False
        self._episode_overtake_count = 0
        self._episode_avoidance_success = True
        self._episode_lap_start_time = None
        self._episode_lap_times = []
        self._prev_progress_diff = 0
        self._v14_global_init_done = False
        self._v14_learner_trace_fi = deque(maxlen=max(200, int(self.v14_trace_buffer_maxlen)))
        self._v14_npc_anchor_state = {}
        self._v14_wobble_tick = 0
        self._v14_last_reposition_step = {}
        self._v14_stage34_contact_reset_cooldown_left = 0
        self._v14_force_collision_spawn_refresh = False

        # 策略蒸馏引用 (由外部设置)
        self.distillation_manager: Optional[PolicyDistillationManager] = None
        self.distillation_model_ref = None  # 当前model引用

    def set_distillation(self, manager: PolicyDistillationManager, model):
        """设置策略蒸馏管理器和model引用。"""
        self.distillation_manager = manager
        self.distillation_model_ref = model

    def _half_width_cte_at_fine(self, fi):
        """兼容V11基类：返回窄侧半宽（CTE单位）。"""
        manual = getattr(self, 'manual_spawn', None)
        if manual is not None and getattr(manual, 'loaded', False):
            try:
                return float(max(1e-3, manual.half_width_narrow_cte_at(fi)))
            except Exception:
                pass
        return float(max(1e-3, getattr(self, 'current_max_cte', 10.0)))

    def _half_width_avg_cte_at_fine(self, fi):
        """兼容V11基类：返回平均半宽（CTE单位）。"""
        manual = getattr(self, 'manual_spawn', None)
        if manual is not None and getattr(manual, 'loaded', False):
            try:
                return float(max(1e-3, manual.half_width_cte_at(fi)))
            except Exception:
                pass
        return float(max(1e-3, getattr(self, 'current_max_cte', 10.0)))

    def _half_width_wide_cte_at_fine(self, fi):
        """兼容V11基类：返回宽侧半宽（CTE单位）。"""
        manual = getattr(self, 'manual_spawn', None)
        if manual is not None and getattr(manual, 'loaded', False):
            try:
                return float(max(1e-3, manual.half_width_wide_cte_at(fi)))
            except Exception:
                pass
        return float(max(1e-3, getattr(self, 'current_max_cte', 10.0)))

    def reset(self):
        """Reset with mixed sampling support for Stage 4."""
        # 混合采样: Stage 4 以 p_npc_free 概率运行无NPC episode
        stage_id = int(self.curriculum_stage_ref.get('stage', 1))
        p_free = float(self.curriculum_stage_ref.get('p_npc_free', self.p_npc_free))
        if stage_id == 4 and p_free > 0 and random.random() < p_free:
            self._episode_npc_free = True
            # 临时将npc_count设为0
            self._saved_npc_count = int(self.curriculum_stage_ref.get('npc_count', 0))
            self.curriculum_stage_ref['npc_count'] = 0
        else:
            self._episode_npc_free = False

        obs = super().reset()

        # 恢复npc_count
        if self._episode_npc_free and hasattr(self, '_saved_npc_count'):
            self.curriculum_stage_ref['npc_count'] = self._saved_npc_count

        # 重置V14状态
        self._npc_lateral_dist_history.clear()
        self._npc_lateral_dist_histories = {}
        self._episode_overtake_count = 0
        self._episode_avoidance_success = True
        self._episode_lap_start_time = time.time()
        self._episode_lap_times = []
        self._v14_last_reposition_step = {}
        self._v14_stage34_contact_reset_cooldown_left = 0
        base_cte = None
        for key in ("stage_cte_reset_limit", "stage1_cte_reset_limit", "max_cte_limit"):
            raw = self.curriculum_stage_ref.get(key, None)
            if raw is None:
                continue
            try:
                base_cte = float(raw)
                break
            except Exception:
                continue
        if base_cte is None:
            base_cte = float(getattr(self, "current_max_cte", 8.0))

        # Stage3/4 的出界done收紧到 Stage1/2 口径（不比早期阶段更宽松）
        if int(stage_id) >= 3:
            stage12_cte = []
            for sid in (1, 2):
                p = STAGE_TRAIN_PROFILES_V14.get(sid)
                if p is not None and p.max_cte_limit is not None:
                    try:
                        stage12_cte.append(float(p.max_cte_limit))
                    except Exception:
                        pass
            if stage12_cte:
                base_cte = float(min(float(base_cte), min(stage12_cte)))

        cte_relax = float(self.v14_cte_done_relax)
        if int(stage_id) >= 3:
            cte_relax = min(0.0, cte_relax)  # 收紧模式下不允许放宽
        self.current_max_cte = float(max(0.5, float(base_cte) + cte_relax))

        return obs


__all__ = ["GeneratedTrackV14Wrapper"]
