#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 curriculum manager."""

from collections import deque

import numpy as np

from .utils import _safe_float
from .v14_stage_profiles import (
    STAGES_V14,
    STAGE_EVAL_GATES_V14,
    apply_stage_train_profile_v14,
)

class CurriculumManagerV14:
    """V14 课程管理器: 4阶段升级 + 策略蒸馏触发 + 混合采样控制。"""

    def __init__(self, curriculum_stage_ref, distillation_manager=None,
                 eval_freq_steps=25000, eval_episodes=3,
                 consecutive_success_required=2,
                 args=None, eval_max_steps=1000):
        self.ref = curriculum_stage_ref
        self.distillation = distillation_manager
        self.eval_freq_steps = int(eval_freq_steps)
        self.eval_episodes = int(eval_episodes)
        self.consecutive_success_required = int(consecutive_success_required)
        self.args = args
        self.eval_max_steps = int(eval_max_steps)
        self.last_eval_at = 0
        self.stage_enter_step = 0
        self.consecutive_eval_success = 0
        # 避让成功率跟踪 (滚动窗口)
        self._avoidance_results = deque(maxlen=20)
        self._lap_times = deque(maxlen=20)

    def set_stage(self, stage_id, global_step, wrapper=None, model=None):
        """切换到指定阶段，触发策略蒸馏快照。"""
        stage = STAGES_V14.get(int(stage_id), STAGES_V14[1])
        old_stage = int(self.ref.get('stage', 0))
        target_laps = int(max(1, _safe_float(
            getattr(self.args, "v14_force_reset_laps", self.ref.get("success_laps_target", 10)),
            self.ref.get("success_laps_target", 10)
        )))

        # 策略蒸馏: 切换前快照当前策略
        if model is not None and self.distillation is not None and old_stage > 0:
            self.distillation.snapshot_policy(model, old_stage)

        # 更新 curriculum_stage_ref
        self.ref['stage'] = stage.sid
        self.ref['reward_mode'] = stage.reward_mode
        self.ref['npc_count'] = stage.npc_count
        self.ref['npc_mode'] = stage.npc_mode
        self.ref['npc_speed_min'], self.ref['npc_speed_max'] = stage.npc_speed_range
        self.ref['p_npc_free'] = stage.p_npc_free
        self.ref['success_laps_target'] = int(target_laps)
        self.ref['terminate_on_success_laps'] = True
        self.ref['spawn_max_attempts'] = 8

        self.consecutive_eval_success = 0
        self.stage_enter_step = int(global_step)

        # 应用阶段训练参数
        apply_stage_train_profile_v14(
            stage.sid, self.ref, args=self.args, wrapper=wrapper, model=model
        )

        # NPC速度范围同步到wrapper
        if wrapper is not None and stage.npc_speed_range != (0.0, 0.0):
            self.ref['npc_speed_min'] = stage.npc_speed_range[0]
            self.ref['npc_speed_max'] = stage.npc_speed_range[1]
        if wrapper is not None:
            wrapper.success_laps_target = int(target_laps)

        gate = STAGE_EVAL_GATES_V14.get(stage.sid)
        print(f"\n🎓 V14 切换阶段 -> {stage.sid}: {stage.name}")
        print(f"   reward_mode={stage.reward_mode} | npc={stage.npc_count} | "
              f"npc_mode={stage.npc_mode} | p_npc_free={stage.p_npc_free} | "
              f"success_laps_target={int(target_laps)}")
        if gate is not None:
            print(f"   EvalGate: min_steps={gate.min_stage_steps}, "
                  f"eval_every={gate.eval_every_steps}, passes={gate.consecutive_passes_required}")

    def maybe_eval_and_promote(self, model, wrapper, global_step):
        """评估当前阶段，满足条件则晋级。"""
        stage_id = int(self.ref.get('stage', 1))
        stage = STAGES_V14.get(stage_id, STAGES_V14[1])
        gate = STAGE_EVAL_GATES_V14.get(stage_id)
        if gate is None:
            return None

        eval_every = int(gate.eval_every_steps)
        min_stage_steps = int(gate.min_stage_steps)

        if global_step - self.last_eval_at < eval_every:
            return None
        if global_step - self.stage_enter_step < min_stage_steps:
            self.last_eval_at = global_step
            print(f"⏳ V14 阶段{stage_id} 未达最小步数 {min_stage_steps}, 跳过评估")
            return None

        self.last_eval_at = global_step

        # 执行评估
        result = self._evaluate(model, wrapper, gate)

        ok = bool(result.get('gate_passed', False))
        if ok:
            self.consecutive_eval_success += 1
        else:
            self.consecutive_eval_success = 0

        print(f"🧪 V14 Eval(stage={stage_id}) passed={ok} "
              f"progress={result.get('avg_progress', 0):.2f} "
              f"collision_rate={result.get('collision_rate', 0):.0%} "
              f"avoidance_rate={result.get('avoidance_rate', 0):.0%} "
              f"consecutive={self.consecutive_eval_success}/{gate.consecutive_passes_required}")

        if (self.consecutive_eval_success >= gate.consecutive_passes_required
                and stage_id < max(STAGES_V14.keys())):
            self.set_stage(stage_id + 1, global_step, wrapper=wrapper, model=model)
            result['promoted_to'] = stage_id + 1

        # KL系数衰减
        if self.distillation is not None:
            self.distillation.decay_kl_coef()

        return result

    def _evaluate(self, model, wrapper, gate):
        """在训练间隙用同一环境做评估。"""
        n_episodes = int(gate.eval_episodes)
        max_steps = int(self.eval_max_steps) if self.eval_max_steps > 0 else int(wrapper.max_episode_steps)
        max_steps = min(int(wrapper.max_episode_steps), max_steps)

        episodes = []
        for ep in range(n_episodes):
            obs = wrapper.reset()
            lstm_state = None
            episode_start = np.array([True], dtype=bool)
            done = False
            ep_steps = 0
            collisions = False
            laps = 0
            max_progress = 0.0
            avoidance_ok = True

            while not done and ep_steps < max_steps:
                action, lstm_state = model.predict(
                    obs, state=lstm_state, episode_start=episode_start, deterministic=True
                )
                obs, reward, done, info = wrapper.step(action)
                episode_start = np.array([done], dtype=bool)
                ep_steps += 1
                laps = max(laps, int(_safe_float(info.get('lap_count', 0), 0)))
                max_progress = max(max_progress, float(_safe_float(
                    info.get('episode_progress_laps_est', 0.0), 0.0)))
                if info.get('hit', 'none') != 'none':
                    collisions = True
                    avoidance_ok = False
                term = info.get('termination_reason', '')
                if term in ('stuck', 'persistent_offtrack'):
                    avoidance_ok = False

            episodes.append({
                'laps': laps,
                'progress': max_progress,
                'collision': collisions,
                'avoidance_ok': avoidance_ok,
                'steps': ep_steps,
            })
            self._avoidance_results.append(avoidance_ok)

        # 汇总
        result = {
            'episodes': episodes,
            'avg_progress': float(np.mean([e['progress'] for e in episodes])),
            'avg_laps': float(np.mean([e['laps'] for e in episodes])),
            'collision_rate': float(np.mean([1.0 if e['collision'] else 0.0 for e in episodes])),
            'avoidance_rate': float(np.mean([1.0 if e['avoidance_ok'] else 0.0 for e in episodes])),
        }

        # Gate check
        passed = True
        if gate.min_relative_progress_laps > 0:
            if result['avg_progress'] < gate.min_relative_progress_laps:
                passed = False
        if gate.require_lap_count_at_least > 0:
            if result['avg_laps'] < gate.require_lap_count_at_least:
                passed = False
        if gate.max_collisions >= 0:
            total_collisions = sum(1 for e in episodes if e['collision'])
            if total_collisions > gate.max_collisions:
                passed = False
        if gate.min_avoidance_success_rate > 0:
            if result['avoidance_rate'] < gate.min_avoidance_success_rate:
                passed = False

        result['gate_passed'] = passed
        return result


__all__ = ["CurriculumManagerV14"]
