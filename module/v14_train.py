#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 training entry."""

import json
import os

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None
    print("⚠️ sb3_contrib 未安装，RecurrentPPO 不可用。请安装: pip install sb3_contrib==1.8.0")

from .v14_dep_generatedtrack_base import (
    build_dist_scale_profile,
    DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK,
)
from .utils import _apply_global_seeds

from .v14_attention import AttentionCNN
from .v14_stage_profiles import STAGES_V14
from .v14_distillation import PolicyDistillationManager
from .v14_curriculum import CurriculumManagerV14
from .v14_callbacks import V14ControlTBCallback
from .v14_env_factory import create_v14_env_and_npcs

def train_v14(args):
    """V14 主训练函数。"""
    if RecurrentPPO is None:
        print("❌ 需要 sb3_contrib。安装: pip install sb3_contrib==1.8.0")
        return

    seed = _apply_global_seeds(args.seed)
    stage_id = int(args.start_stage)
    stage_id = max(1, min(stage_id, max(STAGES_V14.keys())))

    curriculum_stage_ref = {
        'stage': stage_id,
        'reward_mode': STAGES_V14[stage_id].reward_mode,
        'npc_count': STAGES_V14[stage_id].npc_count,
        'npc_mode': STAGES_V14[stage_id].npc_mode,
        'npc_speed_min': STAGES_V14[stage_id].npc_speed_range[0],
        'npc_speed_max': STAGES_V14[stage_id].npc_speed_range[1],
        'p_npc_free': STAGES_V14[stage_id].p_npc_free,
        'success_laps_target': int(max(1, int(args.v14_force_reset_laps))),
        'terminate_on_success_laps': True,
        'max_throttle': float(args.max_throttle),
    }

    # NPC lazy connect (Stage 1 不需要NPC，后续晋级时连接)
    lazy_connect = bool(STAGES_V14[stage_id].npc_count == 0 and int(args.num_npc) > 0)

    vec_env, wrapper, npcs, track_cache = create_v14_env_and_npcs(
        args, curriculum_stage_ref, dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK),
        lazy_connect_npcs=lazy_connect,
    )
    wrapper.dist_scale = build_dist_scale_profile(track_cache, "generated_track")
    print("\n📏 比例尺档案:")
    print(json.dumps(wrapper.dist_scale, indent=2, ensure_ascii=False))

    try:
        vec_env.seed(seed)
    except Exception:
        pass

    # 策略蒸馏管理器
    distillation = PolicyDistillationManager(
        kl_coef_initial=float(args.kl_coef_initial),
        kl_decay=float(args.kl_decay),
        kl_min=float(args.kl_min),
    )

    # Policy kwargs: AttentionCNN
    policy_kwargs = dict(
        features_extractor_class=AttentionCNN,
        features_extractor_kwargs=dict(features_dim=64),
    )

    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    if args.pretrained_model and os.path.exists(args.pretrained_model):
        print(f"📦 加载模型: {args.pretrained_model}")
        model = RecurrentPPO.load(args.pretrained_model, env=vec_env, tensorboard_log=args.tb_log)
    else:
        model = RecurrentPPO(
            "CnnLstmPolicy",
            vec_env,
            learning_rate=float(args.lr),
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            n_epochs=int(args.n_epochs),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            ent_coef=float(args.ent_coef),
            vf_coef=float(args.vf_coef),
            max_grad_norm=float(args.max_grad_norm),
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.tb_log,
            seed=seed,
        )

    # 连接蒸馏
    wrapper.set_distillation(distillation, model)

    # 课程管理器
    curriculum = CurriculumManagerV14(
        curriculum_stage_ref,
        distillation_manager=distillation,
        eval_freq_steps=int(args.eval_freq_steps),
        eval_episodes=int(args.eval_episodes),
        consecutive_success_required=int(args.eval_consecutive_success),
        args=args,
        eval_max_steps=int(args.eval_max_steps),
    )
    curriculum.set_stage(stage_id, global_step=0, wrapper=wrapper, model=model)

    print("\n🚀 V14 训练启动 (CNN+CBAM + 4阶段课程 + 策略蒸馏)")
    print(f"   总步数: {args.total_steps:,}")
    print(f"   起始阶段: {stage_id} ({STAGES_V14[stage_id].name})")
    print(f"   KL蒸馏: coef={args.kl_coef_initial}, decay={args.kl_decay}, min={args.kl_min}")
    print(f"   混合采样(Stage4): p_npc_free={STAGES_V14[4].p_npc_free}")
    print(f"   NPC前向放置: {args.npc_spawn_ahead_min_sim:.1f}~{args.npc_spawn_ahead_max_sim:.1f} sim | "
          f"与Learner最小距离={args.npc_spawn_min_learner_dist_sim:.1f} sim | "
          f"NPC间最小距离={args.npc_spawn_min_npc_dist_sim:.1f} sim")
    print(f"   V14首回合暖机: {'ON' if args.v14_global_warmup_no_npc else 'OFF'} | "
          f"轨迹窗口={args.v14_trace_window} | 最少轨迹点={args.v14_trace_min_points} | "
          f"CTE放宽+{args.v14_cte_done_relax:.2f} | NPC出生随机={'ON' if args.v14_npc_spawn_randomize else 'OFF'}")
    print(f"   NPC朝向随机: {'ON' if args.v14_npc_random_heading else 'OFF'} | "
          f"原地晃动: {'ON' if args.v14_npc_wobble_in_place else 'OFF'} "
          f"(半径={args.v14_npc_wobble_radius_sim:.2f} sim, 周期={int(args.v14_npc_wobble_period_steps)}步, "
          f"刷新={int(args.v14_npc_wobble_update_every_steps)}步)")
    print(f"   奖励主判据: {'仅前方NPC' if args.v14_reward_forward_only else '全向NPC'} | "
          f"重放阈值={int(args.v14_reposition_overpass_gap_idx)}idx, "
          f"重放冷却={int(args.v14_reposition_min_step_gap)}步")
    print(f"   NPC近距步惩罚: 半径={float(args.v14_npc_radius_penalty_radius_sim):.2f} sim | "
          f"每步扣分={float(args.v14_npc_radius_penalty_per_step):.3f}")
    print(f"   强制回合重置: lap_count >= {int(args.v14_force_reset_laps)}")
    print(f"   Stage3/4 NPC速度比例: {float(args.v14_stage34_npc_speed_ratio_min):.2f}~"
          f"{float(args.v14_stage34_npc_speed_ratio_max):.2f} × learner | "
          f"碰撞后重置NPC: {'ON' if args.v14_stage34_collision_reset_npc else 'OFF'} "
          f"(冷却={int(args.v14_stage34_npc_contact_reset_cooldown_steps)}步)")
    print(f"   额外惩罚/奖励: collision_pen={float(args.v14_collision_extra_penalty):.2f} | "
          f"lap_bonus={float(args.v14_lap_complete_bonus):.2f} + "
          f"{float(args.v14_lap_time_reward_scale):.2f}*(ref={float(args.v14_lap_time_ref_sec):.1f}s/lap_time)")
    print(f"   速度激励: speed_keep_scale={float(args.v14_speed_maintain_bonus_scale):.3f} | "
          f"v_ref_max={float(args.v_ref_max):.2f} | startup_throttle={float(args.startup_force_throttle):.2f} "
          f"({int(args.startup_force_throttle_steps)} steps)")

    total_done = 0
    chunk = max(1000, int(args.train_chunk_steps))
    tb_callback = V14ControlTBCallback(log_every=int(getattr(args, 'ctrl_tb_log_every', 500)))

    try:
        while total_done < int(args.total_steps):
            steps = min(chunk, int(args.total_steps) - total_done)
            model.learn(
                total_timesteps=steps,
                reset_num_timesteps=False,
                tb_log_name="v14_cbam_distill",
                callback=tb_callback,
            )
            total_done += steps

            # 保存checkpoint
            if total_done % int(args.save_freq) == 0 or total_done >= int(args.total_steps):
                cur_stage = int(curriculum_stage_ref.get('stage', stage_id))
                path = os.path.join(save_dir, f"v14_cbam_stage{cur_stage}_step{total_done}")
                model.save(path)
                print(f"💾 已保存: {path}.zip")

            # 评估 & 晋级
            if args.auto_promote:
                curriculum.maybe_eval_and_promote(model, wrapper, total_done)
                try:
                    vec_env.reset()
                except Exception as e:
                    print(f"⚠️ vec_env.reset() 同步失败: {e}")

    finally:
        final_stage = int(curriculum_stage_ref.get('stage', stage_id))
        final_path = os.path.join(save_dir, f"v14_cbam_final_stage{final_stage}")
        try:
            model.save(final_path)
            print(f"💾 最终模型: {final_path}.zip")
        except Exception as e:
            print(f"⚠️ 保存最终模型失败: {e}")
        # cleanup
        for npc in npcs:
            try:
                npc.close()
            except Exception:
                pass
        try:
            vec_env.close()
        except Exception:
            pass


__all__ = ["RecurrentPPO", "train_v14"]
