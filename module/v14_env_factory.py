#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 environment and NPC factory."""

import time

import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from .v14_dep_sim_core import TrackNodeCache, NPCController
from .v14_wrapper import GeneratedTrackV14Wrapper
from .v14_paths import default_generated_track_profile

def _query_generated_track_cache_from_env(env, scene_name="generated_track", total_nodes=108):
    """从已创建env中查询赛道节点并构建fine track缓存。"""
    track_cache = TrackNodeCache()
    handler = getattr(getattr(env, "viewer", None), "handler", None)
    if handler is None:
        raise RuntimeError("无法获取 DonkeySim handler，不能查询赛道节点")
    track_cache.query_nodes(handler, scene_name, total_nodes=int(total_nodes))
    if not track_cache.fine_track.get(scene_name):
        track_cache._build_fine_track(scene_name)
    return track_cache


def create_v14_env_and_npcs(args, curriculum_stage_ref, dist_scale_default,
                             lazy_connect_npcs=False):
    """创建V14训练环境 + NPC控制器。"""
    import gym_donkeycar  # noqa

    conf = {
        "exe_path": getattr(args, 'exe_path', None) or "manual",
        "host": str(args.host),
        "port": int(args.port),
        "body_style": str(getattr(args, 'body_style', 'donkey')),
        "body_rgb": tuple(getattr(args, 'body_rgb', (128, 128, 255))),
        "car_name": str(getattr(args, 'car_name', 'V14_Learner')),
        "font_size": 32,
        "racer_name": "V14_PPO",
        "country": "CN",
        "bio": "V14 CNN+CBAM+Distillation",
        "guid": "",
        "max_cte": float(args.max_cte),
        "frame_skip": 1,
    }

    scene_name = "generated_track"
    env = gym.make("donkey-generated-track-v0", conf=conf)
    env._max_episode_steps = int(args.max_episode_steps)
    time.sleep(1.5)

    # 创建 track cache
    track_cache = TrackNodeCache()
    try:
        track_cache = _query_generated_track_cache_from_env(
            env, scene_name=scene_name, total_nodes=108
        )
    except Exception as e:
        print(f"⚠️ track cache query failed: {e}")

    # NPC 控制器
    num_npc = int(getattr(args, 'num_npc', 2))
    npc_colors = [
        (255, 100, 100), (100, 255, 100), (255, 255, 100),
        (100, 100, 255), (255, 165, 0), (180, 0, 255),
    ]
    npcs = []
    for i in range(num_npc):
        npc = NPCController(
            npc_id=i + 1,
            host=str(args.host),
            port=int(args.port),
            scene=scene_name,
            track_cache=track_cache,
        )
        if not lazy_connect_npcs:
            try:
                color = npc_colors[i % len(npc_colors)]
                ok = npc.connect(body_rgb=color)
                if ok:
                    npc.set_mode('static', 0.0)
                    npc.set_position_node_coords(0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 1.0)
            except Exception as e:
                print(f"⚠️ NPC {i} connect failed: {e}")
        npcs.append(npc)

    # Wrapper 链
    wrapper = GeneratedTrackV14Wrapper(
        env,
        npc_controllers=npcs,
        track_cache=track_cache,
        dist_scale_profile=dist_scale_default,
        curriculum_stage_ref=curriculum_stage_ref,
        target_size=(120, 160),
        manual_width_profile=str(getattr(args, 'manual_width_profile',
                                          default_generated_track_profile())),
        racing_line_lookahead=int(getattr(args, 'racing_line_lookahead', 20)),
        racing_line_smoothing=int(getattr(args, 'racing_line_smoothing', 10)),
        racing_line_max_offset_ratio=float(getattr(args, 'racing_line_max_offset_ratio', 0.35)),
        proactive_zone_scale=float(getattr(args, 'proactive_zone_scale', 2.0)),
        proactive_reward_scale=float(getattr(args, 'proactive_reward_scale', 0.30)),
        close_call_penalty=float(getattr(args, 'close_call_penalty', 0.30)),
        safe_passage_bonus=float(getattr(args, 'safe_passage_bonus', 0.40)),
        dynamic_overtake_bonus=float(getattr(args, 'dynamic_overtake_bonus', 1.0)),
        v14_overtake_bonus_growth=float(getattr(args, 'v14_overtake_bonus_growth', 0.35)),
        v14_overtake_bonus_max_mult=float(getattr(args, 'v14_overtake_bonus_max_mult', 4.0)),
        v14_collision_extra_penalty=float(getattr(args, 'v14_collision_extra_penalty', 2.0)),
        v14_lap_complete_bonus=float(getattr(args, 'v14_lap_complete_bonus', 1.2)),
        v14_lap_time_reward_scale=float(getattr(args, 'v14_lap_time_reward_scale', 1.2)),
        v14_lap_time_ref_sec=float(getattr(args, 'v14_lap_time_ref_sec', 23.0)),
        v14_lap_time_reward_max_ratio=float(getattr(args, 'v14_lap_time_reward_max_ratio', 2.5)),
        v14_speed_maintain_bonus_scale=float(getattr(args, 'v14_speed_maintain_bonus_scale', 0.12)),
        rear_end_penalty=float(getattr(args, 'rear_end_penalty', 1.2)),
        racing_line_weight_stage4=float(getattr(args, 'racing_line_weight_stage4', 0.5)),
        curvature_penalty_scale=float(getattr(args, 'curvature_penalty_scale', 0.10)),
        p_npc_free=float(getattr(args, 'p_npc_free', 0.15)),
        npc_spawn_ahead_min_sim=float(getattr(args, 'npc_spawn_ahead_min_sim', 6.0)),
        npc_spawn_ahead_max_sim=float(getattr(args, 'npc_spawn_ahead_max_sim', 12.0)),
        npc_spawn_min_learner_dist_sim=float(getattr(args, 'npc_spawn_min_learner_dist_sim', 3.2)),
        npc_spawn_min_npc_dist_sim=float(getattr(args, 'npc_spawn_min_npc_dist_sim', 2.8)),
        npc_spawn_min_npc_progress_gap_idx=int(getattr(args, 'npc_spawn_min_npc_progress_gap_idx', 90)),
        npc_spawn_candidate_tries=int(getattr(args, 'npc_spawn_candidate_tries', 36)),
        v14_global_warmup_no_npc=bool(getattr(args, 'v14_global_warmup_no_npc', True)),
        v14_trace_buffer_maxlen=int(getattr(args, 'v14_trace_buffer_maxlen', 6000)),
        v14_trace_window=int(getattr(args, 'v14_trace_window', 1600)),
        v14_trace_min_points=int(getattr(args, 'v14_trace_min_points', 40)),
        v14_npc_spawn_randomize=bool(getattr(args, 'v14_npc_spawn_randomize', True)),
        v14_cte_done_relax=float(getattr(args, 'v14_cte_done_relax', 0.0)),
        v14_avoidance_success_bonus=float(getattr(args, 'v14_avoidance_success_bonus', 0.45)),
        v14_npc_random_heading=bool(getattr(args, 'v14_npc_random_heading', True)),
        v14_npc_wobble_in_place=bool(getattr(args, 'v14_npc_wobble_in_place', False)),
        v14_npc_wobble_radius_sim=float(getattr(args, 'v14_npc_wobble_radius_sim', 0.10)),
        v14_npc_wobble_period_steps=int(getattr(args, 'v14_npc_wobble_period_steps', 32)),
        v14_npc_wobble_update_every_steps=int(getattr(args, 'v14_npc_wobble_update_every_steps', 6)),
        v14_npc_wobble_yaw_jitter_deg=float(getattr(args, 'v14_npc_wobble_yaw_jitter_deg', 18.0)),
        v14_reposition_overpass_gap_idx=int(getattr(args, 'v14_reposition_overpass_gap_idx', 90)),
        v14_reposition_far_factor=float(getattr(args, 'v14_reposition_far_factor', 5.0)),
        v14_reposition_min_step_gap=int(getattr(args, 'v14_reposition_min_step_gap', 80)),
        v14_reward_forward_only=bool(getattr(args, 'v14_reward_forward_only', True)),
        v14_npc_radius_penalty_radius_sim=float(getattr(args, 'v14_npc_radius_penalty_radius_sim', 1.5)),
        v14_npc_radius_penalty_per_step=float(getattr(args, 'v14_npc_radius_penalty_per_step', 0.12)),
        v14_stage34_npc_speed_ratio_min=float(getattr(args, 'v14_stage34_npc_speed_ratio_min', 0.30)),
        v14_stage34_npc_speed_ratio_max=float(getattr(args, 'v14_stage34_npc_speed_ratio_max', 0.80)),
        v14_stage34_collision_reset_npc=bool(getattr(args, 'v14_stage34_collision_reset_npc', True)),
        v14_stage34_npc_contact_reset_cooldown_steps=int(
            getattr(args, 'v14_stage34_npc_contact_reset_cooldown_steps', 60)
        ),
        # V11_1 控制层参数
        v_ref_min=float(getattr(args, 'v_ref_min', 0.05)),
        v_ref_max=float(getattr(args, 'v_ref_max', 2.0)),
        kappa_ref_max=float(getattr(args, 'kappa_ref_max', 2.1)),
        startup_force_throttle_steps=int(getattr(args, 'startup_force_throttle_steps', 60)),
        startup_force_throttle=float(getattr(args, 'startup_force_throttle', 0.28)),
        # V10 参数
        spawn_jitter_s_sim=float(getattr(args, 'spawn_jitter_s_sim', 0.30)),
        spawn_jitter_d_sim=float(getattr(args, 'spawn_jitter_d_sim', 0.25)),
        spawn_yaw_jitter_deg=float(getattr(args, 'spawn_yaw_jitter_deg', 6.0)),
        lazy_connect_npcs=lazy_connect_npcs,
        # Reward / Control
        max_throttle=float(args.max_throttle),
        delta_max=float(getattr(args, 'delta_max', 0.10)),
        enable_lpf=bool(getattr(args, 'enable_lpf', True)),
        beta=float(getattr(args, 'beta', 0.6)),
        enable_dr=bool(getattr(args, 'enable_dr', True)),
        max_episode_steps=int(args.max_episode_steps),
    )

    monitored = Monitor(wrapper)
    vec_env = DummyVecEnv([lambda: monitored])

    return vec_env, wrapper, npcs, track_cache


__all__ = [
    "_query_generated_track_cache_from_env",
    "create_v14_env_and_npcs",
]
