#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 CLI and main entry."""

import argparse
import json
import time

import gym
import gym_donkeycar  # noqa: F401

from .v14_dep_generatedtrack_base import build_dist_scale_profile
from .v14_env_factory import _query_generated_track_cache_from_env
from .v14_train import train_v14
from .v14_paths import default_generated_track_profile
from .utils import _write_json

def parse_args():
    p = argparse.ArgumentParser(description="V14 CNN+CBAM + 4-stage Curriculum + Policy Distillation")

    p.add_argument('--mode', choices=['calibrate', 'train'], default='train')
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=9091)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exe-path', type=str, default=None)

    # 地图
    p.add_argument('--max-cte', type=float, default=10.0)
    p.add_argument('--num-npc', type=int, default=2)
    p.add_argument('--max-episode-steps', type=int, default=1500)
    p.add_argument('--body-style', default='donkey')
    p.add_argument('--body-rgb', type=int, nargs=3, default=[128, 128, 255])
    p.add_argument('--car-name', type=str, default='V14_Learner')
    p.add_argument('--manual-width-profile', type=str,
                   default=default_generated_track_profile())

    # 训练
    p.add_argument('--total-steps', type=int, default=700000)
    p.add_argument('--train-chunk-steps', type=int, default=4096)
    p.add_argument('--save-freq', type=int, default=50000)
    p.add_argument('--save-dir', type=str, default='models/v14')
    p.add_argument('--tb-log', type=str, default='logs/v14')
    p.add_argument('--pretrained-model', type=str, default='')
    p.add_argument('--start-stage', type=int, default=1)
    p.add_argument('--auto-promote', action='store_true', default=True)
    p.add_argument('--no-auto-promote', action='store_false', dest='auto_promote')

    # PPO超参
    p.add_argument('--lr', type=float, default=1.5e-4)
    p.add_argument('--n-steps', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--n-epochs', type=int, default=4)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--gae-lambda', type=float, default=0.95)
    p.add_argument('--clip-range', type=float, default=0.2)
    p.add_argument('--ent-coef', type=float, default=0.008)
    p.add_argument('--vf-coef', type=float, default=0.5)
    p.add_argument('--max-grad-norm', type=float, default=0.5)

    # 控制层
    p.add_argument('--max-throttle', type=float, default=0.30)
    p.add_argument('--delta-max', type=float, default=0.10)
    p.add_argument('--enable-lpf', action='store_true', default=True)
    p.add_argument('--no-lpf', action='store_false', dest='enable_lpf')
    p.add_argument('--beta', type=float, default=0.6)
    p.add_argument('--enable-dr', action='store_true', default=True)
    p.add_argument('--no-dr', action='store_false', dest='enable_dr')
    p.add_argument('--v-ref-min', type=float, default=0.05)
    p.add_argument('--v-ref-max', type=float, default=2.0)
    p.add_argument('--kappa-ref-max', type=float, default=2.1)
    p.add_argument('--startup-force-throttle-steps', type=int, default=60)
    p.add_argument('--startup-force-throttle', type=float, default=0.28)

    # spawn
    p.add_argument('--spawn-jitter-s-sim', type=float, default=0.30)
    p.add_argument('--spawn-jitter-d-sim', type=float, default=0.25)
    p.add_argument('--spawn-yaw-jitter-deg', type=float, default=6.0)
    p.add_argument('--npc-spawn-ahead-min-sim', type=float, default=6.0)
    p.add_argument('--npc-spawn-ahead-max-sim', type=float, default=12.0)
    p.add_argument('--npc-spawn-min-learner-dist-sim', type=float, default=3.2)
    p.add_argument('--npc-spawn-min-npc-dist-sim', type=float, default=2.8)
    p.add_argument('--npc-spawn-min-npc-progress-gap-idx', type=int, default=90)
    p.add_argument('--npc-spawn-candidate-tries', type=int, default=36)
    p.add_argument('--v14-global-warmup-no-npc', action='store_true', default=True)
    p.add_argument('--no-v14-global-warmup-no-npc', action='store_false', dest='v14_global_warmup_no_npc')
    p.add_argument('--v14-trace-buffer-maxlen', type=int, default=6000)
    p.add_argument('--v14-trace-window', type=int, default=1600)
    p.add_argument('--v14-trace-min-points', type=int, default=40)
    p.add_argument('--v14-npc-spawn-randomize', action='store_true', default=True)
    p.add_argument('--no-v14-npc-spawn-randomize', action='store_false', dest='v14_npc_spawn_randomize')
    p.add_argument('--v14-cte-done-relax', type=float, default=0.0)
    p.add_argument('--v14-avoidance-success-bonus', type=float, default=0.45)
    p.add_argument('--v14-npc-random-heading', action='store_true', default=True)
    p.add_argument('--no-v14-npc-random-heading', action='store_false', dest='v14_npc_random_heading')
    p.add_argument('--v14-npc-wobble-in-place', action='store_true', default=False)
    p.add_argument('--no-v14-npc-wobble-in-place', action='store_false', dest='v14_npc_wobble_in_place')
    p.add_argument('--v14-npc-wobble-radius-sim', type=float, default=0.10)
    p.add_argument('--v14-npc-wobble-period-steps', type=int, default=32)
    p.add_argument('--v14-npc-wobble-update-every-steps', type=int, default=6)
    p.add_argument('--v14-npc-wobble-yaw-jitter-deg', type=float, default=18.0)
    p.add_argument('--v14-reposition-overpass-gap-idx', type=int, default=90)
    p.add_argument('--v14-reposition-far-factor', type=float, default=5.0)
    p.add_argument('--v14-reposition-min-step-gap', type=int, default=80)
    p.add_argument('--v14-reward-forward-only', action='store_true', default=True)
    p.add_argument('--no-v14-reward-forward-only', action='store_false', dest='v14_reward_forward_only')
    p.add_argument('--v14-npc-radius-penalty-radius-sim', type=float, default=1.5)
    p.add_argument('--v14-npc-radius-penalty-per-step', type=float, default=0.12)
    p.add_argument('--v14-force-reset-laps', type=int, default=10)
    p.add_argument('--v14-stage34-npc-speed-ratio-min', type=float, default=0.30)
    p.add_argument('--v14-stage34-npc-speed-ratio-max', type=float, default=0.80)
    p.add_argument('--v14-stage34-collision-reset-npc', action='store_true', default=True)
    p.add_argument('--no-v14-stage34-collision-reset-npc', action='store_false', dest='v14_stage34_collision_reset_npc')
    p.add_argument('--v14-stage34-npc-contact-reset-cooldown-steps', type=int, default=60)

    # V14 赛车线
    p.add_argument('--racing-line-lookahead', type=int, default=20)
    p.add_argument('--racing-line-smoothing', type=int, default=10)
    p.add_argument('--racing-line-max-offset-ratio', type=float, default=0.35)
    p.add_argument('--racing-line-reward-scale', type=float, default=0.15)
    p.add_argument('--curvature-penalty-scale', type=float, default=0.10)

    # V14 主动避障
    p.add_argument('--proactive-zone-scale', type=float, default=2.0)
    p.add_argument('--proactive-reward-scale', type=float, default=0.30)
    p.add_argument('--close-call-penalty', type=float, default=0.30)
    p.add_argument('--safe-passage-bonus', type=float, default=0.40)
    p.add_argument('--dynamic-overtake-bonus', type=float, default=1.0)
    p.add_argument('--v14-overtake-bonus-growth', type=float, default=0.35)
    p.add_argument('--v14-overtake-bonus-max-mult', type=float, default=4.0)
    p.add_argument('--v14-collision-extra-penalty', type=float, default=2.0)
    p.add_argument('--v14-lap-complete-bonus', type=float, default=1.2)
    p.add_argument('--v14-lap-time-reward-scale', type=float, default=1.2)
    p.add_argument('--v14-lap-time-ref-sec', type=float, default=23.0)
    p.add_argument('--v14-lap-time-reward-max-ratio', type=float, default=2.5)
    p.add_argument('--v14-speed-maintain-bonus-scale', type=float, default=0.12)
    p.add_argument('--rear-end-penalty', type=float, default=1.2)
    p.add_argument('--racing-line-weight-stage4', type=float, default=0.5)

    # V14 策略蒸馏
    p.add_argument('--kl-coef-initial', type=float, default=0.5)
    p.add_argument('--kl-decay', type=float, default=0.995)
    p.add_argument('--kl-min', type=float, default=0.05)

    # V14 混合采样
    p.add_argument('--p-npc-free', type=float, default=0.15,
                   help='Stage4 混合采样: 无NPC episode概率')

    # 评估
    p.add_argument('--eval-freq-steps', type=int, default=25000)
    p.add_argument('--eval-episodes', type=int, default=3)
    p.add_argument('--eval-consecutive-success', type=int, default=2)
    p.add_argument('--eval-max-steps', type=int, default=1000)
    p.add_argument('--ctrl-tb-log-every', type=int, default=500)

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == 'calibrate':
        print("📏 V14 标定模式")
        env = None
        try:
            conf = {
                "exe_path": getattr(args, 'exe_path', None) or "manual",
                "host": str(args.host),
                "port": int(args.port),
                "body_style": str(getattr(args, 'body_style', 'donkey')),
                "body_rgb": tuple(getattr(args, 'body_rgb', (128, 128, 255))),
                "car_name": "V14_Calibrate",
                "font_size": 32,
                "racer_name": "V14_Calibrate",
                "country": "CN",
                "bio": "V14 calibration",
                "guid": "",
                "max_cte": float(args.max_cte),
                "frame_skip": 1,
            }
            env = gym.make("donkey-generated-track-v0", conf=conf)
            env._max_episode_steps = int(args.max_episode_steps)
            time.sleep(1.5)
            track_cache = _query_generated_track_cache_from_env(
                env, scene_name="generated_track", total_nodes=108
            )
            profile = build_dist_scale_profile(track_cache, "generated_track")
            print(json.dumps(profile, indent=2, ensure_ascii=False))
            out_path = "dist_scale_profile_generated_track.json"
            _write_json(out_path, profile)
        except Exception as e:
            print(f"❌ 标定失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        return

    if args.mode == 'train':
        train_v14(args)
        return


__all__ = ["parse_args", "main"]
