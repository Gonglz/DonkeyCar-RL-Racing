#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 tensorboard callback."""

from stable_baselines3.common.callbacks import BaseCallback

class V14ControlTBCallback(BaseCallback):
    """V14 TensorBoard callback: 记录阶段、奖励组件、避让指标。"""

    def __init__(self, log_every=500, verbose=0):
        super().__init__(verbose)
        self.log_every = max(1, int(log_every))

    def _on_step(self):
        if self.num_timesteps % self.log_every != 0:
            return True
        try:
            infos = self.locals.get('infos', [{}])
            if not infos:
                return True
            info = infos[-1] if isinstance(infos, list) else infos
            if not isinstance(info, dict):
                return True

            prefix = "v14/"
            for key in ['v14_reward_mode', 'v14_episode_npc_free', 'v14_overtake_count',
                         'v14_avoidance_success', 'v14_avg_lap_time']:
                if key in info:
                    val = info[key]
                    if isinstance(val, bool):
                        val = float(val)
                    elif isinstance(val, str):
                        continue
                    self.logger.record(prefix + key, float(val))

            # reward terms
            terms = info.get('reward_terms', {})
            if isinstance(terms, dict):
                for k, v in terms.items():
                    if k.startswith(('racing_line', 'curvature', 'proactive', 'safe_passage',
                                     'danger_', 'close_call', 'dynamic_', 'rear_end',
                                     'emergency', 'avoidance_success', 'kl_distillation',
                                     'collision_', 'lap_')):
                        self.logger.record(prefix + "reward/" + k, float(v))

        except Exception:
            pass
        return True


__all__ = ["V14ControlTBCallback"]
