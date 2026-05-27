#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 stage configs, train profiles and eval gates."""

from dataclasses import dataclass, asdict
from typing import Optional


class StageConfigV14:
    def __init__(self, sid, name, reward_mode, npc_count=0, npc_mode="offtrack",
                 npc_speed_range=(0.0, 0.0), min_stage_steps=50000,
                 p_npc_free=0.0):
        self.sid = sid
        self.name = name
        self.reward_mode = reward_mode
        self.npc_count = npc_count
        self.npc_mode = npc_mode
        self.npc_speed_range = npc_speed_range
        self.min_stage_steps = min_stage_steps
        self.p_npc_free = p_npc_free  # 混合采样: 无NPC episode概率


@dataclass
class StageTrainProfileV14:
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
    npc_lane_offset_left_sim: Optional[float] = None
    npc_lane_offset_right_sim: Optional[float] = None
    npc_lane_jitter_sim: Optional[float] = None
    ent_coef: Optional[float] = None
    lr: Optional[float] = None
    stuck_counter_limit: Optional[int] = None
    offtrack_counter_limit: Optional[int] = None
    reward_clip_abs: Optional[float] = None
    progress_full_lap_reward: Optional[float] = None


@dataclass
class StageEvalGateV14:
    min_stage_steps: int
    eval_every_steps: int = 20000
    eval_episodes: int = 3
    consecutive_passes_required: int = 2
    min_relative_progress_laps: float = 0.0
    require_lap_count_at_least: int = 0
    max_collisions: int = 0
    min_avoidance_success_rate: float = 0.0


STAGES_V14 = {
    1: StageConfigV14(
        1,
        "赛道引导-赛车线+最小曲率",
        "racing_line",
        npc_count=0,
        npc_mode="offtrack",
        min_stage_steps=100000,
    ),
    2: StageConfigV14(
        2,
        "主动避障-抖动NPC",
        "proactive_avoid",
        npc_count=2,
        npc_mode="wobble",
        min_stage_steps=150000,
    ),
    3: StageConfigV14(
        3,
        "规则NPC-动态避让",
        "dynamic_avoid",
        npc_count=1,
        npc_mode="slow_policy",
        npc_speed_range=(0.3, 0.8),
        min_stage_steps=200000,
    ),
    4: StageConfigV14(
        4,
        "混沌NPC+防遗忘",
        "chaos_robust",
        npc_count=2,
        npc_mode="chaos",
        min_stage_steps=250000,
        p_npc_free=0.15,
    ),
}


STAGE_TRAIN_PROFILES_V14 = {
    1: StageTrainProfileV14(
        npc_count=0,
        random_start_enabled=False,
        max_throttle=0.8,
        max_cte_limit=8.0,
        progress_reward_scale=1.4,
        progress_milestone_lap=0.04,
        progress_milestone_reward=2.0,
        progress_reward_decay_min=0.7,
        penalty_decay_min=0.85,
        progress_backward_penalty_scale=0.55,
        reverse_onset_penalty=0.18,
        reverse_streak_penalty_scale=0.06,
        reverse_backdist_penalty_scale=2.0,
        reverse_event_penalty=1.2,
        ent_coef=0.008,
        lr=0.00015,
        stuck_counter_limit=25,
        offtrack_counter_limit=25,
        reward_clip_abs=10.0,
        progress_full_lap_reward=5.0,
    ),
    2: StageTrainProfileV14(
        npc_count=2,
        random_start_enabled=True,
        max_throttle=0.8,
        max_cte_limit=8.0,
        progress_reward_scale=1.05,
        progress_milestone_lap=0.08,
        progress_milestone_reward=1.2,
        progress_reward_decay_min=0.6,
        penalty_decay_min=0.85,
        progress_backward_penalty_scale=0.6,
        reverse_onset_penalty=0.18,
        reverse_streak_penalty_scale=0.06,
        reverse_backdist_penalty_scale=2.1,
        reverse_event_penalty=1.2,
        npc_layout_reset_policy="hybrid",
        npc_layout_reset_every=5,
        npc_lane_offset_left_sim=0.0,
        npc_lane_offset_right_sim=0.0,
        npc_lane_jitter_sim=0.06,
        stuck_counter_limit=25,
        offtrack_counter_limit=25,
        reward_clip_abs=10.0,
        progress_full_lap_reward=2.5,
        ent_coef=0.01,
        lr=0.0003,
    ),
    3: StageTrainProfileV14(
        npc_count=1,
        random_start_enabled=True,
        max_throttle=0.8,
        max_cte_limit=8.0,
        progress_reward_scale=0.9,
        progress_milestone_lap=0.1,
        progress_milestone_reward=1.0,
        progress_reward_decay_min=0.55,
        penalty_decay_min=0.8,
        progress_backward_penalty_scale=0.65,
        reverse_onset_penalty=0.18,
        reverse_streak_penalty_scale=0.06,
        reverse_backdist_penalty_scale=2.1,
        reverse_event_penalty=1.2,
        npc_layout_reset_policy="hybrid",
        npc_layout_reset_every=3,
        stuck_counter_limit=25,
        offtrack_counter_limit=25,
        reward_clip_abs=10.0,
        progress_full_lap_reward=2.0,
        ent_coef=0.006,
        lr=0.00025,
    ),
    4: StageTrainProfileV14(
        npc_count=2,
        random_start_enabled=True,
        max_throttle=0.8,
        max_cte_limit=8.0,
        progress_reward_scale=0.8,
        progress_milestone_lap=0.12,
        progress_milestone_reward=0.8,
        progress_reward_decay_min=0.5,
        penalty_decay_min=0.75,
        progress_backward_penalty_scale=0.7,
        reverse_onset_penalty=0.18,
        reverse_streak_penalty_scale=0.06,
        reverse_backdist_penalty_scale=2.1,
        reverse_event_penalty=1.2,
        npc_layout_reset_policy="hybrid",
        npc_layout_reset_every=3,
        stuck_counter_limit=25,
        offtrack_counter_limit=25,
        reward_clip_abs=10.0,
        progress_full_lap_reward=1.5,
        ent_coef=0.004,
        lr=0.0002,
    ),
}


STAGE_EVAL_GATES_V14 = {
    1: StageEvalGateV14(
        min_stage_steps=100000,
        eval_every_steps=25000,
        eval_episodes=3,
        consecutive_passes_required=2,
        min_relative_progress_laps=1.0,
        require_lap_count_at_least=1,
        max_collisions=0,
    ),
    2: StageEvalGateV14(
        min_stage_steps=150000,
        eval_every_steps=30000,
        eval_episodes=3,
        consecutive_passes_required=2,
        min_relative_progress_laps=1.0,
        max_collisions=1,
    ),
    3: StageEvalGateV14(
        min_stage_steps=200000,
        eval_every_steps=30000,
        eval_episodes=3,
        consecutive_passes_required=2,
        min_relative_progress_laps=1.0,
        require_lap_count_at_least=1,
        min_avoidance_success_rate=0.8,
    ),
    4: StageEvalGateV14(
        min_stage_steps=250000,
        eval_every_steps=50000,
        eval_episodes=3,
        consecutive_passes_required=99,
    ),
}


def apply_stage_train_profile_v14(stage_id, curriculum_stage_ref, args=None, wrapper=None,
                                  model=None, verbose=True):
    """应用阶段训练参数到 curriculum_stage_ref 和 wrapper/model。"""
    profile = STAGE_TRAIN_PROFILES_V14.get(int(stage_id))
    if profile is None:
        return None
    d = asdict(profile)
    for k, v in d.items():
        if v is not None:
            curriculum_stage_ref[k] = v
    # 兼容V11基类字段：max_cte_limit 需要桥接到 stage_cte_reset_limit / stage1_cte_reset_limit
    if d.get("max_cte_limit") is not None:
        try:
            cte_lim = float(d["max_cte_limit"])
            curriculum_stage_ref["stage_cte_reset_limit"] = cte_lim
            curriculum_stage_ref["stage1_cte_reset_limit"] = cte_lim
        except Exception:
            pass
    # 兼容V11基类字段：random_start_enabled 需要桥接到 stage_random_start_enabled
    if d.get("random_start_enabled") is not None:
        try:
            curriculum_stage_ref["stage_random_start_enabled"] = bool(d["random_start_enabled"])
        except Exception:
            pass
    # 同步到wrapper
    if wrapper is not None:
        for attr in ['max_throttle', 'progress_reward_scale', 'progress_milestone_lap',
                      'progress_milestone_reward', 'progress_reward_decay_min',
                      'penalty_decay_min', 'progress_backward_penalty_scale',
                      'reverse_onset_penalty', 'reverse_streak_penalty_scale',
                      'reverse_backdist_penalty_scale', 'reverse_event_penalty',
                      'stuck_counter_limit', 'offtrack_counter_limit',
                      'reward_clip_abs', 'progress_full_lap_reward']:
            val = d.get(attr)
            if val is not None and hasattr(wrapper, attr):
                setattr(wrapper, attr, val)
        cte_lim = d.get('max_cte_limit')
        if cte_lim is not None and hasattr(wrapper, 'current_max_cte'):
            wrapper.current_max_cte = float(cte_lim)
            if hasattr(wrapper, "stage1_cte_reset_limit"):
                wrapper.stage1_cte_reset_limit = float(cte_lim)
    # 同步 lr / ent_coef 到 model
    if model is not None:
        if d.get('lr') is not None:
            try:
                model.learning_rate = float(d['lr'])
            except Exception:
                pass
        if d.get('ent_coef') is not None:
            try:
                model.ent_coef = float(d['ent_coef'])
            except Exception:
                pass
    if verbose:
        print(f"   📋 V14 StageProfile({stage_id}): lr={d.get('lr')}, ent={d.get('ent_coef')}, "
              f"prog_scale={d.get('progress_reward_scale')}")
    return d


__all__ = [
    "StageConfigV14",
    "StageTrainProfileV14",
    "StageEvalGateV14",
    "STAGES_V14",
    "STAGE_TRAIN_PROFILES_V14",
    "STAGE_EVAL_GATES_V14",
    "apply_stage_train_profile_v14",
]
