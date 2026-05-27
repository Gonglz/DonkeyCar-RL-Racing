#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 training entry."""

import json
import os

try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None
    print("⚠️ sb3_contrib note, RecurrentPPO note.note: pip install sb3_contrib==1.8.0")

from.v14_dep_generatedtrack_base import (
    build_dist_scale_profile,
    DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK,
)
from.utils import _apply_global_seeds

from.v14_attention import AttentionCNN
from.v14_stage_profiles import STAGES_V14
from.v14_distillation import PolicyDistillationManager
from.v14_curriculum import CurriculumManagerV14
from.v14_callbacks import V14ControlTBCallback
from.v14_env_factory import create_v14_env_and_npcs

def train_v14(args):
    """V14 notetrainingfunction."""
    if RecurrentPPO is None:
        print("FAIL note sb3_contrib.note: pip install sb3_contrib==1.8.0")
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

    # NPC lazy connect (Stage 1 noteNPC, note)
    lazy_connect = bool(STAGES_V14[stage_id].npc_count == 0 and int(args.num_npc) > 0)

    vec_env, wrapper, npcs, track_cache = create_v14_env_and_npcs(
        args, curriculum_stage_ref, dict(DEFAULT_DIST_SCALE_PROFILE_GENERATED_TRACK),
        lazy_connect_npcs=lazy_connect,
    )
    wrapper.dist_scale = build_dist_scale_profile(track_cache, "generated_track")
    print("\n📏 note:")
    print(json.dumps(wrapper.dist_scale, indent=2, ensure_ascii=False))

    try:
        vec_env.seed(seed)
    except Exception:
        pass

    # note
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
        print(f"build notemodel: {args.pretrained_model}")
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

    # note
    wrapper.set_distillation(distillation, model)

    # note
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

    print("\n🚀 V14 trainingnote (CNN+CBAM + 4stagenote + note)")
    print(f"   note: {args.total_steps:,}")
    print(f"   notestage: {stage_id} ({STAGES_V14[stage_id].name})")
    print(f"   KLnote: coef={args.kl_coef_initial}, decay={args.kl_decay}, min={args.kl_min}")
    print(f"   note(Stage4): p_npc_free={STAGES_V14[4].p_npc_free}")
    print(f"   NPCfirstnote: {args.npc_spawn_ahead_min_sim:.1f}~{args.npc_spawn_ahead_max_sim:.1f} sim | "
          f"noteLearnernote={args.npc_spawn_min_learner_dist_sim:.1f} sim | "
          f"NPCnote={args.npc_spawn_min_npc_dist_sim:.1f} sim")
    print(f"   V14note: {'ON' if args.v14_global_warmup_no_npc else 'OFF'} | "
          f"note={args.v14_trace_window} | note={args.v14_trace_min_points} | "
          f"CTEnote+{args.v14_cte_done_relax:.2f} | NPCnote={'ON' if args.v14_npc_spawn_randomize else 'OFF'}")
    print(f"   NPCnote: {'ON' if args.v14_npc_random_heading else 'OFF'} | "
          f"note: {'ON' if args.v14_npc_wobble_in_place else 'OFF'} "
          f"(note={args.v14_npc_wobble_radius_sim:.2f} sim, note={int(args.v14_npc_wobble_period_steps)}note, "
          f"note={int(args.v14_npc_wobble_update_every_steps)}note)")
    print(f"   rewardnote: {'notefirstnoteNPC' if args.v14_reward_forward_only else 'noteNPC'} | "
          f"note={int(args.v14_reposition_overpass_gap_idx)}idx, "
          f"note={int(args.v14_reposition_min_step_gap)}note")
    print(f"   NPCnote: note={float(args.v14_npc_radius_penalty_radius_sim):.2f} sim | "
          f"note={float(args.v14_npc_radius_penalty_per_step):.3f}")
    print(f"   note: lap_count >= {int(args.v14_force_reset_laps)}")
    print(f"   Stage3/4 NPCnote: {float(args.v14_stage34_npc_speed_ratio_min):.2f}~"
          f"{float(args.v14_stage34_npc_speed_ratio_max):.2f} x learner | "
          f"noteNPC: {'ON' if args.v14_stage34_collision_reset_npc else 'OFF'} "
          f"(note={int(args.v14_stage34_npc_contact_reset_cooldown_steps)}note)")
    print(f"   note/reward: collision_pen={float(args.v14_collision_extra_penalty):.2f} | "
          f"lap_bonus={float(args.v14_lap_complete_bonus):.2f} + "
          f"{float(args.v14_lap_time_reward_scale):.2f}*(ref={float(args.v14_lap_time_ref_sec):.1f}s/lap_time)")
    print(f"   note: speed_keep_scale={float(args.v14_speed_maintain_bonus_scale):.3f} | "
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

            # savecheckpoint
            if total_done % int(args.save_freq) == 0 or total_done >= int(args.total_steps):
                cur_stage = int(curriculum_stage_ref.get('stage', stage_id))
                path = os.path.join(save_dir, f"v14_cbam_stage{cur_stage}_step{total_done}")
                model.save(path)
                print(f"saved notesave: {path}.zip")

            # note & note
            if args.auto_promote:
                curriculum.maybe_eval_and_promote(model, wrapper, total_done)
                try:
                    vec_env.reset()
                except Exception as e:
                    print(f"⚠️ vec_env.reset() notefailed: {e}")

    finally:
        final_stage = int(curriculum_stage_ref.get('stage', stage_id))
        final_path = os.path.join(save_dir, f"v14_cbam_final_stage{final_stage}")
        try:
            model.save(final_path)
            print(f"saved notemodel: {final_path}.zip")
        except Exception as e:
            print(f"⚠️ savenotemodelfailed: {e}")
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
