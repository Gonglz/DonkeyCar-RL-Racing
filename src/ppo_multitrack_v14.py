#!/usr/bin/env python3
"""
DonkeyCar PPO V14 (multi-track)
- 2-domain (ws/gt) multi-scene training
- 6-channel semantic observation + 7D sensor+adapter state
- 3D action: [Δsteer, speed_scale, line_bias] via ActionAdapterWrapper
- RecurrentPPO + MultiInputLstmPolicy + FiLMFeatureExtractor
- yaw_rate dampened (/8.0) for narrow-track stability
- WS cte_norm_scale=0.50 to reduce CTE penalty on narrow track
"""

import os
import sys
import math
import json
import copy
import random
import re
import time
import logging
import socket
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import gym
import gym_donkeycar
import numpy as np

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import torch

try:
    from sb3_contrib import RecurrentPPO
except Exception:
    RecurrentPPO = None

from module.utils import (
    load_config,
    _seed_everything, _safe_seed_env, _find_latest_checkpoint,
)
from module.track import TrackGeometryManager
from module.callbacks import (
    PTHExportCallback, BestModelCallback, DomainAwareBestModelCallback,
    PerSceneStatsCallback, PerDomainStatsCallback,
    AdaptiveLearningRateCallback,
    TrainingMetricsFileLoggerCallback,
    ShortEpisodeLoggerCallback, TqdmProgressCallback,
    SceneSchedulerLoggingCallback, CrashRecoveryCallback,
)
from module.multi_scene_env import (
    MultiSceneEnvV13, MultiInputObsWrapper,
)
from module.actor import FiLMFeatureExtractor


# ============================================================
# Scene/Track Config
# ============================================================
SCENE_SPECS: Dict[str, Dict[str, Any]] = {
    "donkey-waveshare-v0": {
        "scene_key": "waveshare",
        "logging_key": "ws",
        "level_name": "waveshare",
        "track_file": "manual_width_waveshare.json",
        "domain": "ws",
        "reward_overrides": {
            "cte_norm_scale": 0.50,
        },
    },
    "donkey-generated-track-v0": {
        "scene_key": "generated_track",
        "logging_key": "gt",
        "level_name": "generated_track",
        "track_file": "manual_width_generated_track.json",
        "domain": "gt",
    },
    "donkey-warehouse-v0": {
        "scene_key": "warehouse",
        "logging_key": "wh",
        "level_name": "warehouse",
        "track_file": "manual_width_warehouse.json",
        "domain": "gt",
    },
    "donkey-mountain-track-v0": {
        "scene_key": "mountain_track",
        "logging_key": "mt",
        "level_name": "mountain_track",
        "track_file": "manual_width_mountain_track.json",
        "domain": "gt",
    },
    "donkey-minimonaco-track-v0": {
        "scene_key": "mini_monaco",
        "logging_key": "mm",
        "level_name": "mini_monaco",
        "track_file": "manual_width_mini_monaco.json",
        "domain": "gt",
    },
    # --- RRL 场景暂不启用（V14 仅 ws+gt 双场景训练） ---
    # "donkey-roboracingleague-track-v0": {
    #     "scene_key": "roboracingleague_track",
    #     "logging_key": "rrl",
    #     "level_name": "roboracingleague_1",
    #     "track_file": "manual_width_roboracingleague_track.json",
    #     "domain": "rrl",
    #     "max_cte": 6.0,
    #     "reward_overrides": {
    #         "near_offtrack_start_ratio": 0.50,
    #         "w_near_offtrack": 0.45,
    #         "w_center": 0.05,
    #         "collision_penalty_base": 10.0,
    #         "progress_reward_scale": 100.0,
    #     },
    # },
    "donkey-warren-track-v0": {
        "scene_key": "warren_track",
        "logging_key": "wt",
        "level_name": "warren",
        "track_file": "manual_width_warren_track.json",
        "domain": "gt",
    },
    "donkey-circuit-launch-track-v0": {
        "scene_key": "circuit_launch",
        "logging_key": "cl",
        "level_name": "circuit_launch",
        "track_file": "manual_width_circuit_launch.json",
        "domain": "gt",
    },
}

DEFAULT_ENV_IDS: List[str] = [
    "donkey-waveshare-v0",
    "donkey-generated-track-v0",
]

# ============================================================
# Curriculum stages — V14 仅 ws+gt，RRL 已注释
# ============================================================
STAGE_ENV_IDS: Dict[str, List[str]] = {
    "S1": [
        "donkey-waveshare-v0",
        "donkey-generated-track-v0",
        # "donkey-roboracingleague-track-v0",   # rrl 暂不启用
    ],
    "S2": [
        "donkey-waveshare-v0",
        "donkey-generated-track-v0",
        # "donkey-roboracingleague-track-v0",
    ],
}

STAGE_SCENE_WEIGHTS: Dict[str, List[float]] = {
    "S1": [0.55, 0.45],   # ws+gt only
    "S2": [0.55, 0.45],
}

STAGE_STEP_BALANCE_MASK: Dict[str, List[bool]] = {
    "S1": [True, True],
    "S2": [True, True],
}

# Per-scene 动态调权上限：顺序与 STAGE_ENV_IDS 对应: [ws, gt]
STAGE_DYNAMIC_WEIGHT_MAX: Dict[str, List[float]] = {
    "S1": [0.55, 0.55],
    "S2": [0.55, 0.55],
}

CURRICULUM_STAGE_ADVANCE_RULES: Dict[str, Dict[str, Any]] = {
    "S1": {
        # 总步数 30% 后开始检查：ws/gt 最近 10 局都至少 2 lap，则提前晋级到 S2。
        "advance_after_ratio": 0.30,
        "required_logging_keys": ["ws", "gt"],
        "recent_episodes": 10,
        "min_laps_per_episode": 2.0,
    },
}


# ============================================================
# V13 Stats Logging Callback
# ============================================================
class V13StatsLoggingCallback(BaseCallback):
    def __init__(self, log_freq: int = 500, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = int(max(1, log_freq))
        self.buffer: Dict[str, deque] = {
            # speed controller
            "ctrl/v_target": deque(maxlen=self.log_freq),
            "ctrl/v_meas": deque(maxlen=self.log_freq),
            "ctrl/v_err": deque(maxlen=self.log_freq),
            "ctrl/throttle_pi": deque(maxlen=self.log_freq),
            # V13 adapter internals (ctrl/ prefix, unified)
            "ctrl/steer_core": deque(maxlen=self.log_freq),
            "ctrl/bias_smooth": deque(maxlen=self.log_freq),
            "ctrl/bias_offset": deque(maxlen=self.log_freq),
            "ctrl/v_base": deque(maxlen=self.log_freq),
            "ctrl/delta_steer_input": deque(maxlen=self.log_freq),
            "ctrl/speed_scale_input": deque(maxlen=self.log_freq),
            "ctrl/line_bias_input": deque(maxlen=self.log_freq),
            # reward debug
            "reward_debug/cte_abs": deque(maxlen=self.log_freq),
            "reward_debug/cte_over_in": deque(maxlen=self.log_freq),
            "reward_debug/cte_over_out": deque(maxlen=self.log_freq),
            "reward_debug/offtrack_counter": deque(maxlen=self.log_freq),
            "reward_debug/stuck_counter": deque(maxlen=self.log_freq),
            "reward_debug/progress_reward_raw": deque(maxlen=self.log_freq),
            "reward_debug/progress_center_gate": deque(maxlen=self.log_freq),
            "reward_debug/progress_forward_gain": deque(maxlen=self.log_freq),
            "reward_debug/r_center": deque(maxlen=self.log_freq),
            "reward_debug/r_heading": deque(maxlen=self.log_freq),
            "reward_debug/r_speed_ref": deque(maxlen=self.log_freq),
            "reward_debug/r_time": deque(maxlen=self.log_freq),
            "reward_debug/r_near_offtrack": deque(maxlen=self.log_freq),
            "reward_debug/r_near_collision": deque(maxlen=self.log_freq),
            "reward_debug/near_collision_risk": deque(maxlen=self.log_freq),
            "reward_debug/cte_out_ratio": deque(maxlen=self.log_freq),
            "reward_debug/cte_out_ratio_done": deque(maxlen=self.log_freq),
            "reward_debug/near_offtrack_ratio": deque(maxlen=self.log_freq),
            "reward_debug/near_offtrack_ramp_scale": deque(maxlen=self.log_freq),
            "reward_debug/near_collision_ramp_scale": deque(maxlen=self.log_freq),
            "reward_debug/near_collision_proxy_risk": deque(maxlen=self.log_freq),
            "reward_debug/near_collision_obstacle_risk": deque(maxlen=self.log_freq),
            "reward_debug/near_collision_has_obstacle_signal": deque(maxlen=self.log_freq),
            "reward_debug/obstacle_dist": deque(maxlen=self.log_freq),
            "reward_debug/throttle_high_penalty_hit": deque(maxlen=self.log_freq),
            # smooth / safety
            "smooth/rate_limit_hit": deque(maxlen=self.log_freq),
            "smooth/steer_clip_hit": deque(maxlen=self.log_freq),
            "smooth/hairpin_relax_active": deque(maxlen=self.log_freq),
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            for k in self.buffer:
                if k in info:
                    try:
                        self.buffer[k].append(float(info[k]))
                    except Exception:
                        pass

        if self.n_calls % self.log_freq == 0:
            for k, dq in self.buffer.items():
                if len(dq) > 0:
                    self.logger.record(k, float(np.mean(dq)))
            cte_dq = self.buffer.get("reward_debug/cte_abs")
            if cte_dq:
                arr = np.asarray(cte_dq, dtype=np.float64)
                if arr.size > 0:
                    self.logger.record("reward_debug/cte_abs_p50", float(np.percentile(arr, 50)))
                    self.logger.record("reward_debug/cte_abs_p90", float(np.percentile(arr, 90)))
                    self.logger.record("reward_debug/cte_abs_p99", float(np.percentile(arr, 99)))
        return True


class CurriculumStageAdvanceCallback(BaseCallback):
    """Early-advance gate for curriculum stages based on recent per-scene laps."""

    def __init__(
        self,
        stage_name: str,
        required_logging_keys: List[str],
        min_total_timesteps: int,
        recent_episodes: int = 10,
        min_laps_per_episode: float = 2.0,
        max_total_timesteps: Optional[int] = None,
        lap_reward_unit: float = 6.0,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.stage_name = str(stage_name)
        self.required_logging_keys = [str(x) for x in required_logging_keys]
        self.min_total_timesteps = max(0, int(min_total_timesteps))
        self.recent_episodes = max(1, int(recent_episodes))
        self.min_laps_per_episode = float(max(0.0, min_laps_per_episode))
        self.max_total_timesteps = (
            None if max_total_timesteps is None else max(1, int(max_total_timesteps))
        )
        self.lap_reward_unit = float(max(1e-6, lap_reward_unit))
        self.recent_laps: Dict[str, deque] = {
            key: deque(maxlen=self.recent_episodes) for key in self.required_logging_keys
        }
        self.triggered = False
        self.stop_reason = ""
        self.stop_num_timesteps = 0
        self.stop_stage_timesteps = 0
        self._start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self._start_num_timesteps = int(getattr(self.model, "num_timesteps", 0))
        if self.verbose > 0:
            joined = "+".join(self.required_logging_keys)
            max_txt = (
                f", fallback@{self.max_total_timesteps} total steps"
                if self.max_total_timesteps is not None else ""
            )
            print(
                f"🎓 课程晋级门控[{self.stage_name}]: "
                f"after {self.min_total_timesteps} total steps, "
                f"{joined} recent {self.recent_episodes} eps all >= "
                f"{self.min_laps_per_episode:.1f} laps{max_txt}"
            )

    def _stage_timesteps(self) -> int:
        return max(0, int(self.num_timesteps) - int(self._start_num_timesteps))

    def _extract_laps(self, info: Dict[str, Any]) -> float:
        lap_raw = 0.0
        try:
            lap_raw = float(info.get("ep_r_lap_raw", 0.0) or 0.0)
        except Exception:
            lap_raw = 0.0
        if not np.isfinite(lap_raw):
            lap_raw = 0.0
        return max(0.0, lap_raw / self.lap_reward_unit)

    def _all_recent_windows_ready(self) -> bool:
        for key in self.required_logging_keys:
            dq = self.recent_laps.get(key)
            if dq is None or len(dq) < self.recent_episodes:
                return False
            if any(float(v) + 1e-9 < self.min_laps_per_episode for v in dq):
                return False
        return True

    def summary(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "triggered": bool(self.triggered),
            "stop_reason": str(self.stop_reason),
            "min_total_timesteps": int(self.min_total_timesteps),
            "max_total_timesteps": (
                None if self.max_total_timesteps is None else int(self.max_total_timesteps)
            ),
            "stop_num_timesteps": int(self.stop_num_timesteps),
            "stop_stage_timesteps": int(self.stop_stage_timesteps),
            "recent_laps": {
                key: [float(x) for x in self.recent_laps.get(key, [])]
                for key in self.required_logging_keys
            },
        }

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not isinstance(info, dict):
                continue
            if "episode" not in info:
                continue
            logging_key = str(info.get("logging_key", "") or "")
            if logging_key not in self.recent_laps:
                continue
            self.recent_laps[logging_key].append(self._extract_laps(info))

        if int(self.num_timesteps) >= self.min_total_timesteps and self._all_recent_windows_ready():
            self.triggered = True
            self.stop_reason = "recent_laps_ready"
            self.stop_num_timesteps = int(self.num_timesteps)
            self.stop_stage_timesteps = self._stage_timesteps()
            if self.verbose > 0:
                snapshot = {
                    key: [float(x) for x in self.recent_laps.get(key, [])]
                    for key in self.required_logging_keys
                }
                print(
                    f"🎯 课程提前晋级[{self.stage_name}]："
                    f"total_steps={self.stop_num_timesteps}, "
                    f"stage_steps={self.stop_stage_timesteps}, recent_laps={snapshot}"
                )
            return False

        if self.max_total_timesteps is not None and int(self.num_timesteps) >= self.max_total_timesteps:
            self.triggered = False
            self.stop_reason = "max_total_timesteps_reached"
            self.stop_num_timesteps = int(self.num_timesteps)
            self.stop_stage_timesteps = self._stage_timesteps()
            if self.verbose > 0:
                print(
                    f"⏱️  课程阶段[{self.stage_name}]未触发提前晋级，"
                    f"达到总步数上限 {self.max_total_timesteps}，切换下一阶段"
                )
            return False

        return True


class StrictStageTimestepsStopCallback(BaseCallback):
    """Hard-stop a stage after an exact relative step budget."""

    def __init__(self, stage_name: str, max_stage_timesteps: int, verbose: int = 1):
        super().__init__(verbose)
        self.stage_name = str(stage_name)
        self.max_stage_timesteps = max(1, int(max_stage_timesteps))
        self.stop_reason = ""
        self.stop_num_timesteps = 0
        self.stop_stage_timesteps = 0
        self._start_num_timesteps = 0

    def _on_training_start(self) -> None:
        self._start_num_timesteps = int(getattr(self.model, "num_timesteps", 0))
        if self.verbose > 0:
            print(
                f"⛔ 严格阶段步数门控[{self.stage_name}]: "
                f"max_stage_timesteps={self.max_stage_timesteps}"
            )

    def _stage_timesteps(self) -> int:
        return max(0, int(self.num_timesteps) - int(self._start_num_timesteps))

    def summary(self) -> Dict[str, Any]:
        return {
            "stage_name": self.stage_name,
            "stop_reason": str(self.stop_reason),
            "max_stage_timesteps": int(self.max_stage_timesteps),
            "stop_num_timesteps": int(self.stop_num_timesteps),
            "stop_stage_timesteps": int(self.stop_stage_timesteps),
        }

    def _on_step(self) -> bool:
        stage_timesteps = self._stage_timesteps()
        if stage_timesteps >= self.max_stage_timesteps:
            self.stop_reason = "max_stage_timesteps_reached"
            self.stop_num_timesteps = int(self.num_timesteps)
            self.stop_stage_timesteps = int(stage_timesteps)
            if self.verbose > 0:
                print(
                    f"⛔ 严格阶段步数门控[{self.stage_name}]命中: "
                    f"stage_steps={self.stop_stage_timesteps} >= {self.max_stage_timesteps}"
                )
            return False
        return True


# ============================================================
# Offline Tests
# ============================================================
def run_offline_track_checks(track_geometry: TrackGeometryManager) -> None:
    summaries = track_geometry.scene_summary()
    if len(summaries) < 1:
        raise RuntimeError("No scene geometry loaded")

    for scene, s in summaries.items():
        if int(s["points"]) <= 100:
            raise AssertionError(f"scene={scene} has too few points: {s['points']}")

    for scene in track_geometry.list_scenes():
        g = track_geometry.scenes[scene]
        n = g.center.shape[0]
        for idx in [0, n // 4, n // 2, (3 * n) // 4]:
            x, z = g.center[idx]
            out = track_geometry.query(scene, x=float(x), z=float(z), yaw_rad=0.0, prev_idx=idx)
            vals = [
                out["lat_err_norm"],
                out["heading_err_sin"],
                out["heading_err_cos"],
                out["kappa_lookahead"],
                out["width_norm"],
            ]
            if not all(np.isfinite(vals)):
                raise AssertionError(f"Non-finite geometry output at scene={scene}, idx={idx}")


def run_v13_contract_tests(obs_size: int = 128) -> None:
    """V13 wrapper chain contract tests using a DummyBaseEnv."""
    from module.wrappers import CanonicalSemanticWrapper
    from module.action_adapter import ActionAdapterWrapper
    from module.control import ActionSafetyWrapper
    from module.reward import DonkeyRewardWrapper
    from module.multi_scene_env import MultiInputObsWrapper, _build_state_v13

    class DummyBaseEnv(gym.Env):
        """Minimal env mimicking DonkeyEnv output for contract tests."""
        def __init__(self, obs_size_: int):
            self.obs_size = int(obs_size_)
            self.observation_space = gym.spaces.Box(
                low=0, high=255,
                shape=(self.obs_size, self.obs_size, 3),
                dtype=np.uint8,
            )
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self._step_count = 0

        def reset(self):
            self._step_count = 0
            return np.random.randint(0, 255, (self.obs_size, self.obs_size, 3), dtype=np.uint8)

        def step(self, action):
            self._step_count += 1
            obs = np.random.randint(0, 255, (self.obs_size, self.obs_size, 3), dtype=np.uint8)
            info = {
                "speed": 0.7,
                "gyro": (0.0, 0.0, 0.2),
                "accel": (0.1, 0.5, 9.7),
                "car": (3.0, -2.0, 30.0),
                "pos": (0.0, 0.0, 0.0),
                "cte": 0.1,
            }
            done = self._step_count >= 50
            return obs, 0.0, done, info

    print("🧪 V13 Contract Tests")

    # Build full wrapper chain
    base = DummyBaseEnv(obs_size)
    semantic = CanonicalSemanticWrapper(base, domain="ws", obs_size=obs_size, augment=False)

    reward_wrapper = DonkeyRewardWrapper(
        semantic, total_timesteps=100000,
        action_safety_wrapper=None,
        w_d=0.04, w_dd=0.01, w_m=0.0, w_sat=0.0,
        w_time=0.01, w_center=0.0, w_heading=0.0, w_speed_ref=0.0,
        progress_reward_scale=62.0, survival_reward_scale=0.30,
        collision_penalty_base=12.0, offtrack_penalty_base=8.0,
    )

    action_safety = ActionSafetyWrapper(reward_wrapper, delta_max=0.35)
    reward_wrapper.action_safety_wrapper = action_safety

    adapter = ActionAdapterWrapper(
        action_safety,
        k_delta=0.10, lambda_bias=0.20, k_bias=0.10,
        v_nominal=1.4, max_throttle=0.3,
    )

    env = MultiInputObsWrapper(
        adapter,
        track_geometry=None, scene_key="waveshare", logging_key="ws",
        domain="ws",
        obs_size=obs_size, image_channels=6,
        include_cte_in_obs=False, speed_vmax=2.2,
        control_wrapper=adapter, action_safety_wrapper=action_safety,
        state_builder=_build_state_v13, state_dim=7,
    )

    # --- Test 1: obs/action dimension contract ---
    obs = env.reset()
    assert isinstance(obs, dict), f"obs should be dict, got {type(obs)}"
    assert obs["image"].shape == (6, obs_size, obs_size), \
        f"image shape {obs['image'].shape} != (6, {obs_size}, {obs_size})"
    assert obs["state"].shape == (7,), f"state shape {obs['state'].shape} != (7,)"
    assert env.action_space.shape == (3,), f"action shape {env.action_space.shape} != (3,)"
    print("  ✅ Test 1: obs/action dimension contract passed")

    # --- Test 2: reset contract (no state leakage) ---
    for _ in range(5):
        env.step(np.array([0.5, 0.3, 0.8], dtype=np.float32))
    obs = env.reset()
    assert adapter.steer_core == 0.0, f"steer_core={adapter.steer_core} after reset"
    assert adapter.bias_smooth == 0.0, f"bias_smooth={adapter.bias_smooth} after reset"
    assert adapter.i_term == 0.0, f"i_term={adapter.i_term} after reset"
    assert adapter.last_speed_mps == 0.0, f"last_speed_mps={adapter.last_speed_mps} after reset"
    assert np.allclose(adapter.last_low_level_action, [0.0, 0.0]), \
        f"last_low_level_action={adapter.last_low_level_action} after reset"
    assert abs(obs["state"][3]) < 1e-6, f"prev_steer_exec={obs['state'][3]} after reset"
    assert abs(obs["state"][4]) < 1e-6, f"prev_throttle_exec={obs['state'][4]} after reset"
    assert abs(obs["state"][5]) < 1e-6, f"steer_core={obs['state'][5]} after reset"
    assert abs(obs["state"][6]) < 1e-6, f"bias_smooth={obs['state'][6]} after reset"
    print("  ✅ Test 2: reset contract passed (no state leakage)")

    # --- Test 3: one-step wrapper chain contract ---
    obs = env.reset()
    action = np.array([0.5, -0.3, 0.7], dtype=np.float32)
    obs2, r, done, info = env.step(action)
    assert adapter.steer_core != 0.0, "steer_core should change after Δsteer=0.5"
    assert adapter.bias_smooth != 0.0, "bias_smooth should change after line_bias=0.7"
    assert "steer_exec" in action_safety.diag, "safety should write steer_exec to diag"
    assert abs(obs2["state"][5] - adapter.steer_core) < 1e-6, \
        f"state[5]={obs2['state'][5]} != adapter.steer_core={adapter.steer_core}"
    assert abs(obs2["state"][6] - adapter.bias_smooth) < 1e-6, \
        f"state[6]={obs2['state'][6]} != adapter.bias_smooth={adapter.bias_smooth}"
    for k in ["ctrl/v_target", "ctrl/steer_core", "ctrl/bias_smooth"]:
        assert k in info, f"info missing key: {k}"
    print("  ✅ Test 3: one-step wrapper chain contract passed")

    print("🧪 All V13 contract tests passed\n")


def run_preflight_tests(
    track_geometry: TrackGeometryManager,
    obs_size: int = 128,
) -> None:
    print("\n🔍 Running preflight checks...")
    run_offline_track_checks(track_geometry)
    print("  ✅ Track geometry checks passed")
    run_v13_contract_tests(obs_size=obs_size)
    print("✅ All preflight checks passed\n")


def _probe_sim_tcp(host: str, port: int, timeout_s: float = 1.0) -> Tuple[bool, str]:
    """Quick TCP reachability probe for simulator endpoint."""
    try:
        with socket.create_connection((str(host), int(port)), timeout=float(timeout_s)):
            return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def _install_sim_wait_timeout_patch(timeout_s: float = 35.0, resend_scene_names_s: float = 3.0) -> bool:
    """
    Patch gym_donkeycar wait_until_loaded to avoid infinite hang.
    When stuck, periodically resend get_scene_names and raise timeout.
    """
    try:
        from gym_donkeycar.envs.donkey_sim import DonkeyUnitySimContoller
    except Exception as e:
        print(f"⚠️  cannot patch sim wait timeout: {type(e).__name__}: {e}")
        return False

    timeout_s = float(max(5.0, timeout_s))
    resend_scene_names_s = float(max(0.0, resend_scene_names_s))
    sim_logger = logging.getLogger("gym_donkeycar.envs.donkey_sim")

    def _wait_until_loaded_with_timeout(self) -> None:
        time.sleep(0.1)
        start_t = time.time()
        last_resend_t = 0.0
        while not self.handler.loaded:
            elapsed = time.time() - start_t
            sim_logger.warning("waiting for sim to start..")
            if elapsed >= timeout_s:
                host = getattr(self, "address", ("localhost", -1))[0]
                port = getattr(self, "address", ("localhost", -1))[1]
                raise TimeoutError(
                    f"sim load handshake timeout after {timeout_s:.0f}s for {host}:{port}. "
                    f"Please restart DonkeySim and retry."
                )

            if resend_scene_names_s > 0.0 and (elapsed - last_resend_t) >= resend_scene_names_s:
                try:
                    if hasattr(self, "handler") and self.handler is not None:
                        self.handler.send_get_scene_names()
                except Exception:
                    pass
                last_resend_t = elapsed

            time.sleep(1.0)
        sim_logger.info("sim started!")

    _wait_until_loaded_with_timeout._v13_timeout_patch = True
    DonkeyUnitySimContoller.wait_until_loaded = _wait_until_loaded_with_timeout
    return True


# ============================================================
# Train V13
# ============================================================
def train_v13(
    env_ids: Optional[List[str]] = None,
    scene_weights: Optional[List[float]] = None,
    track_dir: str = "/home/longzhao/track",
    sim_path: str = "remote",
    total_timesteps: int = 2_000_000,
    save_dir: str = "models/v13_multi_scene",
    port: int = 9091,
    # obs
    obs_size: int = 128,
    augment: bool = False,
    yellow_dropout_prob: float = 0.20,
    dropout_start_step: int = 0,
    dropout_ramp_steps: int = 200_000,
    save_scene_start_snapshots: bool = True,
    scene_start_snapshot_steps: int = 10,
    # LSTM
    use_lstm: bool = True,
    lstm_hidden_size: int = 128,
    lstm_layers: int = 1,
    # ActionAdapter
    adapter_k_delta: float = 0.15,
    adapter_lambda_bias: float = 0.20,
    adapter_k_bias: float = 0.15,
    adapter_steer_core_decay: float = 0.0,
    adapter_v_nominal: float = 1.4,
    adapter_k_turn: float = 0.5,
    adapter_k_bias_speed: float = 0.0,
    adapter_alpha_speed: float = 0.25,
    adapter_v_min: float = 0.6,
    adapter_v_max: float = 1.8,
    adapter_max_throttle: float = 0.3,
    # PI+FF speed controller (inside adapter)
    speed_vmax: float = 2.2,
    speed_kp: float = 0.35,
    speed_ki: float = 0.08,
    speed_kff: float = 0.10,
    allow_reverse: bool = False,
    control_dt: float = 0.05,
    # PPO
    learning_rate: float = 8e-5,
    ent_coef: float = 0.01,
    ppo_n_steps: int = 2048,
    ppo_batch_size: int = 128,
    ppo_n_epochs: int = 4,
    ppo_clip_range: float = 0.2,
    target_kl: Optional[float] = 0.01,
    # switching / balancing
    min_episodes_per_scene: int = 5,
    max_steps_per_scene: int = 640,
    enable_dynamic_scene_weights: bool = True,
    dynamic_weight_update_episodes: int = 24,
    dynamic_weight_window: int = 50,
    dynamic_min_samples_per_scene: int = 6,
    dynamic_weight_alpha: float = 1.6,
    dynamic_length_beta: float = 1.0,
    dynamic_weight_smoothing: float = 0.35,
    dynamic_weight_min: float = 0.02,
    dynamic_weight_max = 0.55,  # float or List[float] for per-scene max
    dynamic_success_mode: str = "scene_adaptive",
    dynamic_success_warmup_episodes: int = 1200,
    dynamic_success_post_warmup_scale: float = 0.20,
    dynamic_success_deficit_mix: float = 0.85,
    enable_step_balance_sampling: bool = True,
    step_balance_sampling_mix: float = 0.3,
    step_balance_mask: Optional[List[bool]] = None,
    # reward / control safety
    delta_max: float = 0.35,
    enable_lpf: bool = True,
    beta: float = 0.6,
    w_d: float = 0.04,
    w_dd: float = 0.01,
    w_m: float = 0.0,
    w_sat: float = 0.0,
    w_time: float = 0.01,
    w_center: float = 0.03,
    w_heading: float = 0.015,
    w_speed_ref: float = 0.0,
    speed_ref_vmin: float = 0.35,
    speed_ref_vmax: float = 2.2,
    speed_ref_kappa_ref: float = 0.15,
    lap_reward_scale: float = 1.0,
    progress_reward_scale: float = 48.0,
    survival_reward_scale: float = 0.30,
    collision_penalty_base: float = 8.0,
    offtrack_penalty_base: float = 5.0,
    adaptive_delta_max: bool = True,
    curve_delta_boost: float = 1.0,
    curve_kappa_ref: float = 0.15,
    steer_intent_boost: float = 0.30,
    hairpin_curve_ratio: float = 0.85,
    hairpin_min_delta_max: float = 0.45,
    hairpin_max_delta_max: float = 0.85,
    w_near_offtrack: float = 0.55,
    near_offtrack_start_ratio: float = 0.45,
    w_near_collision: float = 0.20,
    near_collision_start_ratio: float = 0.65,
    offtrack_leniency_ratio: float = 0.25,
    offtrack_leniency_mult: float = 2.5,
    sim_loaded_timeout_s: float = 35.0,
    sim_wait_resend_scene_names_s: float = 3.0,
    # infra
    seed: Optional[int] = None,
    exp_tag: Optional[str] = None,
    resume_latest: bool = False,
    resume_path: Optional[str] = None,
    run_preflight_checks: bool = True,
    enable_file_metrics_log: bool = True,
    file_metrics_log_freq: int = 1000,
    file_metrics_log_name: str = "train_metrics.jsonl",
    enable_auto_lr_decay: bool = True,
    auto_lr_check_freq: int = 1000,
    auto_lr_decay_factor: float = 0.92,
    auto_lr_min: float = 7e-5,
    auto_lr_high_kl: float = 0.05,
    auto_lr_high_kl_patience: int = 3,
    auto_lr_balanced_drop: float = 12.0,
    auto_lr_balanced_patience: int = 12,
    auto_lr_cooldown_checks: int = 15,
    auto_lr_warmup_steps: int = 250000,
    auto_lr_best_window: int = 50,
    extra_callbacks: Optional[List[BaseCallback]] = None,
):
    if RecurrentPPO is None:
        raise ImportError("sb3_contrib not available, please install sb3-contrib==1.8.0")

    env_ids = list(DEFAULT_ENV_IDS if env_ids is None else env_ids)
    for eid in env_ids:
        if eid not in SCENE_SPECS:
            raise KeyError(f"Unsupported env_id for V13: {eid}")

    if scene_weights is None:
        scene_weights = [1.0 / len(env_ids)] * len(env_ids)
    else:
        if len(scene_weights) != len(env_ids):
            raise ValueError("scene_weights length must match env_ids")
        total_w = float(sum(scene_weights))
        if total_w <= 0:
            raise ValueError("scene_weights sum must be > 0")
        scene_weights = [float(w) / total_w for w in scene_weights]

    if not use_lstm:
        raise ValueError("V13 requires RecurrentPPO MultiInputLstmPolicy; set --use-lstm")

    obs_size = int(max(32, obs_size))
    ppo_n_steps = int(max(64, ppo_n_steps))
    ppo_batch_size = int(max(1, ppo_batch_size))
    ppo_n_epochs = int(max(1, ppo_n_epochs))
    lstm_hidden_size = int(max(8, lstm_hidden_size))
    lstm_layers = int(max(1, lstm_layers))

    if target_kl is not None:
        target_kl = float(target_kl)
        if (not np.isfinite(target_kl)) or target_kl <= 0:
            target_kl = None

    print("\n" + "=" * 76)
    print("🚀 DonkeyCar PPO V13 - 3-Domain Multi-Scene Recurrent Training")
    print("=" * 76)
    print(f"maps: {len(env_ids)}")
    print(f"obs: Dict(image=6x{obs_size}x{obs_size}, state=7)")
    print(f"action: [Δsteer, speed_scale, line_bias], allow_reverse={allow_reverse}")
    print(
        f"adapter: k_delta={adapter_k_delta}, lambda_bias={adapter_lambda_bias}, "
        f"k_bias={adapter_k_bias}, decay={adapter_steer_core_decay}"
    )
    print(
        f"speed: v_nominal={adapter_v_nominal}, k_turn={adapter_k_turn}, "
        f"max_thr={adapter_max_throttle}, kp={speed_kp}, ki={speed_ki}, kff={speed_kff}"
    )
    print(
        f"ppo: lr={learning_rate}, ent={ent_coef}, n_steps={ppo_n_steps}, batch={ppo_batch_size}, "
        f"epochs={ppo_n_epochs}, target_kl={target_kl}"
    )
    print(
        f"balance: min_eps={min_episodes_per_scene}, max_steps={max_steps_per_scene}, "
        f"dyn={'on' if enable_dynamic_scene_weights else 'off'} alpha={dynamic_weight_alpha}, "
        f"succ_mode={dynamic_success_mode}"
    )
    print(
        f"reward: prog={progress_reward_scale}, w_d={w_d}, w_near_off={w_near_offtrack}, "
        f"w_near_col={w_near_collision}, w_center={w_center}, w_heading={w_heading}"
    )

    _seed_everything(seed)

    os.makedirs(save_dir, exist_ok=True)
    snapshot_dir = ""
    if save_scene_start_snapshots and int(scene_start_snapshot_steps) > 0:
        snapshot_dir = os.path.join(save_dir, "scene_start_snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

    # V13 obs/actor don't depend on TrackGeometry, but reward CTE boundaries
    # and track checks still use it. Default: initialize.
    track_geometry = TrackGeometryManager(track_dir=track_dir, env_ids=env_ids, scene_specs=SCENE_SPECS)

    if run_preflight_checks:
        run_preflight_tests(
            track_geometry=track_geometry,
            obs_size=obs_size,
        )

    _launch_sim = bool(sim_path and sim_path not in ("", "remote", "none"))

    try:
        cfg = load_config(myconfig="/home/longzhao/mysim/myconfig.py")
        if cfg is not None and hasattr(cfg, "GYM_CONF"):
            conf = cfg.GYM_CONF.copy()
            _update = {
                "port": port,
                "car_name": "waveshare_v13",
                "racer_name": "V13-ActionAdapter",
                "country": "CN",
                "bio": "V13 3D action adapter with FiLM",
                "guid": "waveshare-v13-multi",
                "max_cte": 8.0,
            }
            if _launch_sim:
                _update["exe_path"] = sim_path
            conf.update(_update)
        else:
            raise ValueError("myconfig load failed")
    except Exception as e:
        print(f"⚠️  load myconfig failed: {e}; using fallback conf")
        conf = {
            "port": port,
            "body_style": "donkey",
            "body_rgb": (128, 128, 128),
            "car_name": "waveshare_v13",
            "font_size": 50,
            "racer_name": "V13-ActionAdapter",
            "country": "CN",
            "bio": "V13 3D action adapter with FiLM",
            "guid": "waveshare-v13-multi",
            "max_cte": 8.0,
        }
        if _launch_sim:
            conf["exe_path"] = sim_path

    sim_host = str(conf.get("host", "localhost"))
    sim_port = int(conf.get("port", port))

    if sim_loaded_timeout_s > 0:
        if _install_sim_wait_timeout_patch(
            timeout_s=float(sim_loaded_timeout_s),
            resend_scene_names_s=float(sim_wait_resend_scene_names_s),
        ):
            print(
                f"🔧 sim wait patch enabled: timeout={float(sim_loaded_timeout_s):.1f}s, "
                f"resend_scene_names_every={float(sim_wait_resend_scene_names_s):.1f}s"
            )
    else:
        print("ℹ️  sim wait timeout patch disabled (sim_loaded_timeout_s <= 0)")

    if not _launch_sim:
        print("ℹ️  sim_path 为空/remote，不自动启动模拟器，请手动启动")
        ok, err = _probe_sim_tcp(sim_host, sim_port, timeout_s=1.0)
        if ok:
            print(f"✅ sim tcp reachable: {sim_host}:{sim_port}")
        else:
            print(f"⚠️  sim tcp not reachable: {sim_host}:{sim_port} ({err})")

    # Dummy env for model initialization
    from gym import spaces

    dummy_obs_dict: Dict[str, gym.spaces.Space] = {
        "image": spaces.Box(
            low=np.full((6, obs_size, obs_size), 0.0, dtype=np.float32),
            high=np.full((6, obs_size, obs_size), 1.0, dtype=np.float32),
            dtype=np.float32,
        ),
        "state": spaces.Box(
            low=np.full((7,), -3.0, dtype=np.float32),
            high=np.full((7,), 3.0, dtype=np.float32),
            dtype=np.float32,
        ),
    }
    dummy_obs_space = spaces.Dict(dummy_obs_dict)

    dummy_act_space = spaces.Box(
        low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        dtype=np.float32,
    )

    class DummyEnv(gym.Env):
        def __init__(self):
            self.observation_space = dummy_obs_space
            self.action_space = dummy_act_space

        def reset(self):
            return {
                "image": np.zeros((6, obs_size, obs_size), dtype=np.float32),
                "state": np.zeros((7,), dtype=np.float32),
            }

        def step(self, action):
            return self.reset(), 0.0, False, {}

    dummy_vec_env = DummyVecEnv([lambda: DummyEnv()])
    _safe_seed_env(dummy_vec_env, seed, label="dummy_v13_env")

    policy_kwargs = dict(
        features_extractor_class=FiLMFeatureExtractor,
        features_extractor_kwargs=dict(
            image_feat_dim=128,
            state_feat_dim=32,
        ),
        lstm_hidden_size=lstm_hidden_size,
        n_lstm_layers=lstm_layers,
        shared_lstm=False,
        enable_critic_lstm=True,
    )

    resume_ckpt_path = None
    if resume_path:
        resume_ckpt_path = resume_path
    elif resume_latest:
        resume_ckpt_path = _find_latest_checkpoint(save_dir, name_prefix="v13")
        if resume_ckpt_path is None:
            raise FileNotFoundError(f"No v13 checkpoint found in {save_dir}")

    if resume_ckpt_path is not None:
        print(f"🔄 Resume from: {resume_ckpt_path}")
        model = RecurrentPPO.load(resume_ckpt_path, env=dummy_vec_env)
        model.learning_rate = float(learning_rate)
        model.lr_schedule = lambda _p, _v=float(learning_rate): _v
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = float(learning_rate)
        model.ent_coef = float(ent_coef)
        model.n_steps = int(ppo_n_steps)
        model.batch_size = int(ppo_batch_size)
        model.n_epochs = int(ppo_n_epochs)
        model.clip_range = lambda _p, _v=float(ppo_clip_range): _v
        model.target_kl = (None if target_kl is None else float(target_kl))
    else:
        model = RecurrentPPO(
            "MultiInputLstmPolicy",
            dummy_vec_env,
            policy_kwargs=policy_kwargs,
            learning_rate=float(learning_rate),
            n_steps=int(ppo_n_steps),
            batch_size=int(ppo_batch_size),
            n_epochs=int(ppo_n_epochs),
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=float(ppo_clip_range),
            clip_range_vf=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=target_kl,
            ent_coef=float(ent_coef),
            verbose=1,
            tensorboard_log=(
                os.path.join(save_dir, "tensorboard", exp_tag)
                if exp_tag
                else os.path.join(save_dir, "tensorboard")
            ),
            seed=(None if seed is None else int(seed)),
        )
    model_start_timesteps = int(getattr(model, "num_timesteps", 0))

    dummy_vec_env.close()

    print("🏗️  Building multi-scene V13 env")

    def make_env():
        return MultiSceneEnvV13(
            env_ids=env_ids,
            conf=conf,
            scene_weights=scene_weights,
            scene_specs=SCENE_SPECS,
            track_geometry=track_geometry,
            obs_size=obs_size,
            augment=augment,
            yellow_dropout_prob=yellow_dropout_prob,
            dropout_start_step=dropout_start_step,
            dropout_ramp_steps=dropout_ramp_steps,
            # ActionAdapter
            adapter_k_delta=adapter_k_delta,
            adapter_lambda_bias=adapter_lambda_bias,
            adapter_k_bias=adapter_k_bias,
            adapter_steer_core_decay=adapter_steer_core_decay,
            adapter_v_nominal=adapter_v_nominal,
            adapter_k_turn=adapter_k_turn,
            adapter_k_bias_speed=adapter_k_bias_speed,
            adapter_alpha_speed=adapter_alpha_speed,
            adapter_v_min=adapter_v_min,
            adapter_v_max=adapter_v_max,
            # PI+FF (passed through kwargs to V12 base)
            speed_vmax=speed_vmax,
            speed_kp=speed_kp,
            speed_ki=speed_ki,
            speed_kff=speed_kff,
            allow_reverse=allow_reverse,
            max_throttle=adapter_max_throttle,
            control_dt=control_dt,
            # Reward
            total_timesteps=total_timesteps,
            delta_max=delta_max,
            enable_lpf=enable_lpf,
            beta=beta,
            w_d=w_d, w_dd=w_dd, w_m=w_m, w_sat=w_sat,
            w_time=w_time, w_center=w_center,
            w_heading=w_heading, w_speed_ref=w_speed_ref,
            speed_ref_vmin=speed_ref_vmin, speed_ref_vmax=speed_ref_vmax,
            speed_ref_kappa_ref=speed_ref_kappa_ref,
            lap_reward_scale=lap_reward_scale,
            progress_reward_scale=progress_reward_scale,
            survival_reward_scale=survival_reward_scale,
            collision_penalty_base=collision_penalty_base,
            offtrack_penalty_base=offtrack_penalty_base,
            offtrack_leniency_ratio=offtrack_leniency_ratio,
            offtrack_leniency_mult=offtrack_leniency_mult,
            # ActionSafety
            adaptive_delta_max=adaptive_delta_max,
            curve_delta_boost=curve_delta_boost,
            curve_kappa_ref=curve_kappa_ref,
            steer_intent_boost=steer_intent_boost,
            hairpin_curve_ratio=hairpin_curve_ratio,
            hairpin_min_delta_max=hairpin_min_delta_max,
            hairpin_max_delta_max=hairpin_max_delta_max,
            w_near_offtrack=w_near_offtrack,
            near_offtrack_start_ratio=near_offtrack_start_ratio,
            w_near_collision=w_near_collision,
            near_collision_start_ratio=near_collision_start_ratio,
            snapshot_dir=snapshot_dir,
            snapshot_max_steps=scene_start_snapshot_steps,
            # Dynamic sampling
            min_episodes_per_scene=min_episodes_per_scene,
            max_steps_per_scene=max_steps_per_scene,
            enable_dynamic_scene_weights=enable_dynamic_scene_weights,
            dynamic_weight_update_episodes=dynamic_weight_update_episodes,
            dynamic_weight_window=dynamic_weight_window,
            dynamic_min_samples_per_scene=dynamic_min_samples_per_scene,
            dynamic_weight_alpha=dynamic_weight_alpha,
            dynamic_length_beta=dynamic_length_beta,
            dynamic_weight_smoothing=dynamic_weight_smoothing,
            dynamic_weight_min=dynamic_weight_min,
            dynamic_weight_max=dynamic_weight_max,
            dynamic_success_mode=dynamic_success_mode,
            dynamic_success_warmup_episodes=dynamic_success_warmup_episodes,
            dynamic_success_post_warmup_scale=dynamic_success_post_warmup_scale,
            dynamic_success_deficit_mix=dynamic_success_deficit_mix,
            enable_step_balance_sampling=enable_step_balance_sampling,
            step_balance_sampling_mix=step_balance_sampling_mix,
            step_balance_mask=step_balance_mask,
        )

    env = DummyVecEnv([make_env])
    _safe_seed_env(env, seed, label="v13_train_env")
    model.set_env(env)

    callbacks: List[BaseCallback] = [
        PTHExportCallback(save_path=save_dir, save_freq=20000, name_prefix="v13", verbose=1),
        BestModelCallback(
            save_path=save_dir,
            check_freq=1000,
            metric_mode="per_scene_min",
            min_episodes_per_scene_for_save=10,
            save_separate_per_scene_best=True,
            scene_keys=None,
            save_balanced_from_training_buffer=False,
            verbose=1,
        ),
        PerSceneStatsCallback(check_freq=1000, short_episode_threshold=15, verbose=1),
        SceneSchedulerLoggingCallback(check_freq=1000, verbose=0),
        ShortEpisodeLoggerCallback(save_dir=save_dir, threshold=15, verbose=1),
        V13StatsLoggingCallback(log_freq=500, verbose=0),
        TqdmProgressCallback(total_timesteps=total_timesteps, update_freq=2048),
        CrashRecoveryCallback(
            save_dir=save_dir,
            check_freq=2000,
            rolling_window=30,
            crash_ratio=0.25,        # rolling < peak * 0.25 → 判定崩溃
            min_peak_len=80.0,       # peak < 80 的场景不触发（RRL 等早期场景）
            cooldown_steps=50000,    # 回滚后 50k 步冷却
            min_warmup_steps=30000,  # 开训 30k 步内不检测
            verbose=1,
        ),
    ]

    if enable_auto_lr_decay:
        callbacks.append(
            AdaptiveLearningRateCallback(
                check_freq=auto_lr_check_freq,
                scene_keys=None,
                min_episodes_per_domain=10,
                balanced_drop_threshold=auto_lr_balanced_drop,
                balanced_drop_patience=auto_lr_balanced_patience,
                high_kl_threshold=auto_lr_high_kl,
                high_kl_patience=auto_lr_high_kl_patience,
                decay_factor=auto_lr_decay_factor,
                min_lr=auto_lr_min,
                cooldown_checks=auto_lr_cooldown_checks,
                warmup_steps=auto_lr_warmup_steps,
                best_window=auto_lr_best_window,
                verbose=1,
            )
        )

    if enable_file_metrics_log:
        callbacks.append(
            TrainingMetricsFileLoggerCallback(
                save_dir=save_dir,
                log_freq=file_metrics_log_freq,
                filename=file_metrics_log_name,
                exp_tag=exp_tag,
                verbose=1,
            )
        )

    if extra_callbacks:
        callbacks.extend(extra_callbacks)

    print("\n" + "=" * 76)
    print("🚦 Start V13 training")
    print("=" * 76)

    start_time = time.time()

    try:
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=False,
                reset_num_timesteps=(resume_ckpt_path is None),
            )
        except Exception as e:
            print(f"\n⚠️  learn() 异常: {e}")
            raise
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    elapsed = time.time() - start_time
    model_end_timesteps = int(getattr(model, "num_timesteps", model_start_timesteps))
    trained_timesteps = max(0, model_end_timesteps - model_start_timesteps)

    final_model_path = os.path.join(save_dir, "final_model")
    model.save(final_model_path)
    final_pth_path = os.path.join(save_dir, "final_model_policy.pth")
    torch.save(model.policy.state_dict(), final_pth_path)

    curriculum_advance = None
    strict_stage_stop = None
    if extra_callbacks:
        for cb in extra_callbacks:
            if isinstance(cb, CurriculumStageAdvanceCallback):
                curriculum_advance = cb.summary()
            elif isinstance(cb, StrictStageTimestepsStopCallback):
                strict_stage_stop = cb.summary()

    config = {
        "version": "V13",
        "timestamp": datetime.now().isoformat(),
        "env_ids": env_ids,
        "scene_weights": scene_weights,
        "track_dir": track_dir,
        "track_geometry_summary": track_geometry.scene_summary(),
        "sim": {
            "sim_path": sim_path,
            "launch_sim": bool(_launch_sim),
            "host": sim_host,
            "port": sim_port,
            "loaded_timeout_s": float(sim_loaded_timeout_s),
            "wait_resend_scene_names_s": float(sim_wait_resend_scene_names_s),
        },
        "observation": {
            "image_shape": [6, obs_size, obs_size],
            "scene_start_snapshots": {
                "enabled": bool(save_scene_start_snapshots),
                "steps_per_scene": int(scene_start_snapshot_steps),
                "dir": snapshot_dir,
            },
            "image_channels": [
                "raw_Y", "blue_prob", "yellow_prob",
                "sobel_edge", "vehicle_prob", "motion_residual",
            ],
            "state_dim": 7,
            "state_features": [
                "v_long_norm", "yaw_rate_norm", "accel_x_norm",
                "prev_steer_exec", "prev_throttle_exec",
                "steer_core", "bias_smooth",
            ],
        },
        "action": {
            "space": "[delta_steer, speed_scale, line_bias]",
            "allow_reverse": allow_reverse,
            "adapter": {
                "k_delta": adapter_k_delta,
                "lambda_bias": adapter_lambda_bias,
                "k_bias": adapter_k_bias,
                "steer_core_decay": adapter_steer_core_decay,
                "v_nominal": adapter_v_nominal,
                "k_turn": adapter_k_turn,
                "k_bias_speed": adapter_k_bias_speed,
                "alpha_speed": adapter_alpha_speed,
                "v_min": adapter_v_min,
                "v_max": adapter_v_max,
                "max_throttle": adapter_max_throttle,
            },
            "speed_controller": {
                "vmax": speed_vmax,
                "kp": speed_kp,
                "ki": speed_ki,
                "kff": speed_kff,
                "dt": control_dt,
            },
        },
        "ppo": {
            "policy": "MultiInputLstmPolicy",
            "learning_rate": learning_rate,
            "ent_coef": ent_coef,
            "n_steps": ppo_n_steps,
            "batch_size": ppo_batch_size,
            "n_epochs": ppo_n_epochs,
            "clip_range": ppo_clip_range,
            "target_kl": target_kl,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_layers": lstm_layers,
        },
        "balancing": {
            "min_episodes_per_scene": min_episodes_per_scene,
            "max_steps_per_scene": max_steps_per_scene,
            "enable_dynamic_scene_weights": enable_dynamic_scene_weights,
            "dynamic_weight_update_episodes": dynamic_weight_update_episodes,
            "dynamic_weight_window": dynamic_weight_window,
            "dynamic_min_samples_per_scene": dynamic_min_samples_per_scene,
            "dynamic_weight_alpha": dynamic_weight_alpha,
            "dynamic_length_beta": dynamic_length_beta,
            "dynamic_weight_smoothing": dynamic_weight_smoothing,
            "dynamic_weight_min": dynamic_weight_min,
            "dynamic_weight_max": dynamic_weight_max,
            "dynamic_success_mode": dynamic_success_mode,
            "dynamic_success_warmup_episodes": dynamic_success_warmup_episodes,
            "dynamic_success_post_warmup_scale": dynamic_success_post_warmup_scale,
            "dynamic_success_deficit_mix": dynamic_success_deficit_mix,
            "enable_step_balance_sampling": enable_step_balance_sampling,
            "step_balance_sampling_mix": step_balance_sampling_mix,
            "step_balance_mask": (list(step_balance_mask) if step_balance_mask is not None else None),
        },
        "reward": {
            "lap_reward_scale": lap_reward_scale,
            "progress_reward_scale": progress_reward_scale,
            "survival_reward_scale": survival_reward_scale,
            "collision_penalty_base": collision_penalty_base,
            "offtrack_penalty_base": offtrack_penalty_base,
            "w_d": w_d,
            "w_dd": w_dd,
            "w_m": w_m,
            "w_sat": w_sat,
            "w_time": w_time,
            "w_center": w_center,
            "w_heading": w_heading,
            "w_speed_ref": w_speed_ref,
            "speed_ref_vmin": speed_ref_vmin,
            "speed_ref_vmax": speed_ref_vmax,
            "speed_ref_kappa_ref": speed_ref_kappa_ref,
            "w_near_offtrack": w_near_offtrack,
            "near_offtrack_start_ratio": near_offtrack_start_ratio,
            "w_near_collision": w_near_collision,
            "near_collision_start_ratio": near_collision_start_ratio,
            "offtrack_leniency_ratio": offtrack_leniency_ratio,
            "offtrack_leniency_mult": offtrack_leniency_mult,
        },
        "action_safety": {
            "delta_max": delta_max,
            "adaptive_delta_max": adaptive_delta_max,
            "curve_delta_boost": curve_delta_boost,
            "curve_kappa_ref": curve_kappa_ref,
            "steer_intent_boost": steer_intent_boost,
            "hairpin_curve_ratio": hairpin_curve_ratio,
            "hairpin_min_delta_max": hairpin_min_delta_max,
            "hairpin_max_delta_max": hairpin_max_delta_max,
        },
        "resume": {
            "enabled": resume_ckpt_path is not None,
            "path": resume_ckpt_path,
            "resume_latest": resume_latest,
        },
        "run_summary": {
            "trained_timesteps": int(trained_timesteps),
            "num_timesteps_total": int(model_end_timesteps),
            "curriculum_advance": curriculum_advance,
            "strict_stage_stop": strict_stage_stop,
        },
        "seed": seed,
        "exp_tag": exp_tag,
        "training_time_hours": elapsed / 3600.0,
        "final_model_zip": final_model_path + ".zip",
        "final_model_pth": final_pth_path,
    }

    config_path = os.path.join(save_dir, "v13_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("✅ V13 training finished")
    print("=" * 76)
    print(f"elapsed: {elapsed / 3600.0:.2f} h")
    print(f"model: {final_model_path}.zip")
    print(f"policy: {final_pth_path}")
    print(f"config: {config_path}")

    env.close()
    return {
        "trained_timesteps": int(trained_timesteps),
        "num_timesteps_total": int(model_end_timesteps),
        "final_model_zip": final_model_path + ".zip",
        "final_model_pth": final_pth_path,
        "config_path": config_path,
        "curriculum_advance": curriculum_advance,
        "strict_stage_stop": strict_stage_stop,
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="V13: 3-domain recurrent PPO with 6ch semantic obs, 7D state, 3D action adapter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--train-profile", type=str, default="v13_clean",
                        choices=["v13_clean", "v13_softsafe"],
                        help="训练配置档位：v13_clean(默认baseline) / v13_softsafe(安全项更软)")
    parser.add_argument("--env-ids", nargs="+", type=str, default=None)
    parser.add_argument("--scene-weights", nargs="+", type=float, default=None)
    parser.add_argument("--stage", type=str, default=None, choices=["S1", "S2"],
                        help="分阶段训练：S1(ws+gt+rrl热身) / S2(ws+gt+rrl)。优先级低于 --env-ids")
    parser.add_argument("--track-dir", type=str, default="/home/longzhao/track")

    parser.add_argument("--sim", type=str, default="remote",
                        help="模拟器路径，设为 remote/none/空 则不自动启动 (默认: remote)")
    parser.add_argument("--steps", type=int, default=2000000)
    parser.add_argument("--save-dir", type=str, default="models/v13_multi_scene")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--sim-loaded-timeout-s", type=float, default=35.0,
                        help="sim 握手最长等待秒数；超时后直接报错而不是无限等待")
    parser.add_argument("--sim-wait-resend-scene-names-s", type=float, default=3.0,
                        help="等待 sim 握手时，重发 get_scene_names 的间隔秒数")

    parser.add_argument("--obs-size", type=int, default=128)
    parser.add_argument("--no-augment", action="store_false", dest="augment", default=False)
    parser.add_argument("--augment", action="store_true", dest="augment")
    parser.add_argument("--yellow-dropout-prob", type=float, default=0.20)
    parser.add_argument("--dropout-start-step", type=int, default=0)
    parser.add_argument("--dropout-ramp-steps", type=int, default=200000)

    parser.add_argument("--use-lstm", action="store_true", default=True)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)

    # ActionAdapter
    parser.add_argument("--adapter-k-delta", type=float, default=0.15)
    parser.add_argument("--adapter-lambda-bias", type=float, default=0.20)
    parser.add_argument("--adapter-k-bias", type=float, default=0.15)
    parser.add_argument("--adapter-steer-core-decay", type=float, default=0.0)
    parser.add_argument("--adapter-v-nominal", type=float, default=1.4)
    parser.add_argument("--adapter-k-turn", type=float, default=0.5)
    parser.add_argument("--adapter-k-bias-speed", type=float, default=0.0)
    parser.add_argument("--adapter-alpha-speed", type=float, default=0.25)
    parser.add_argument("--adapter-v-min", type=float, default=0.6)
    parser.add_argument("--adapter-v-max", type=float, default=1.8)
    parser.add_argument("--adapter-max-throttle", type=float, default=0.3,
                        help="Adapter 内部速度控制器最大油门")

    # PI+FF speed controller
    parser.add_argument("--speed-vmax", type=float, default=2.2)
    parser.add_argument("--speed-kp", type=float, default=0.35)
    parser.add_argument("--speed-ki", type=float, default=0.08)
    parser.add_argument("--speed-kff", type=float, default=0.10)
    parser.add_argument("--allow-reverse", action="store_true", default=False)
    parser.add_argument("--control-dt", type=float, default=0.05)

    # PPO
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--ppo-n-steps", type=int, default=2048)
    parser.add_argument("--ppo-batch-size", type=int, default=128)
    parser.add_argument("--ppo-n-epochs", type=int, default=6)
    parser.add_argument("--ppo-clip-range", type=float, default=0.2)
    parser.add_argument("--target-kl", type=float, default=0.015)

    # Balancing
    parser.add_argument("--min-episodes-per-scene", type=int, default=5)
    parser.add_argument("--max-steps-per-scene", type=int, default=640)
    parser.add_argument("--disable-dynamic-scene-weights", action="store_true", default=False)
    parser.add_argument("--dynamic-weight-update-episodes", type=int, default=24)
    parser.add_argument("--dynamic-weight-window", type=int, default=50)
    parser.add_argument("--dynamic-min-samples-per-scene", type=int, default=6)
    parser.add_argument("--dynamic-weight-alpha", type=float, default=1.6)
    parser.add_argument("--dynamic-length-beta", type=float, default=1.0)
    parser.add_argument("--dynamic-weight-smoothing", type=float, default=0.35)
    parser.add_argument("--dynamic-weight-min", type=float, default=0.02)
    parser.add_argument("--dynamic-weight-max", type=float, default=0.55)
    parser.add_argument("--dynamic-success-mode", type=str, default="scene_adaptive",
                        choices=["env_done", "scene_adaptive", "scene_strict"])
    parser.add_argument("--dynamic-success-warmup-episodes", type=int, default=1200)
    parser.add_argument("--dynamic-success-post-warmup-scale", type=float, default=0.20)
    parser.add_argument("--dynamic-success-deficit-mix", type=float, default=0.85)
    parser.add_argument("--disable-step-balance-sampling", action="store_true", default=False)
    parser.add_argument("--step-balance-sampling-mix", type=float, default=0.3)

    # Safety
    parser.add_argument("--delta-max", type=float, default=0.35)
    parser.add_argument("--enable-lpf", action="store_true", default=True)
    parser.add_argument("--no-lpf", action="store_false", dest="enable_lpf")
    parser.add_argument("--beta", type=float, default=0.6)

    # Reward
    parser.add_argument("--w-d", type=float, default=0.04)
    parser.add_argument("--w-dd", type=float, default=0.01)
    parser.add_argument("--w-m", type=float, default=0.0)
    parser.add_argument("--w-sat", type=float, default=0.0)
    parser.add_argument("--w-time", type=float, default=0.01)
    parser.add_argument("--w-center", type=float, default=0.03,
                        help="居中惩罚权重（轻量引导，依赖 TrackGeometry）")
    parser.add_argument("--w-heading", type=float, default=0.015,
                        help="航向惩罚权重（轻量引导，依赖 TrackGeometry）")
    parser.add_argument("--w-speed-ref", type=float, default=0.0,
                        help="速度参考惩罚权重（V13 默认 0：adapter 内部管速度）")
    parser.add_argument("--speed-ref-vmin", type=float, default=0.35)
    parser.add_argument("--speed-ref-vmax", type=float, default=2.2)
    parser.add_argument("--speed-ref-kappa-ref", type=float, default=0.15)
    parser.add_argument("--lap-reward-scale", type=float, default=1.0)
    parser.add_argument("--progress-reward-scale", type=float, default=48.0)
    parser.add_argument("--survival-reward-scale", type=float, default=0.30)
    parser.add_argument("--collision-penalty-base", type=float, default=8.0)
    parser.add_argument("--offtrack-penalty-base", type=float, default=5.0)
    parser.add_argument("--disable-adaptive-delta-max", action="store_true", default=False)
    parser.add_argument("--curve-delta-boost", type=float, default=1.0)
    parser.add_argument("--curve-kappa-ref", type=float, default=0.15)
    parser.add_argument("--steer-intent-boost", type=float, default=0.30)
    parser.add_argument("--hairpin-curve-ratio", type=float, default=0.85)
    parser.add_argument("--hairpin-min-delta-max", type=float, default=0.45)
    parser.add_argument("--hairpin-max-delta-max", type=float, default=0.85)
    parser.add_argument("--w-near-offtrack", type=float, default=0.55)
    parser.add_argument("--near-offtrack-start-ratio", type=float, default=0.45)
    parser.add_argument("--w-near-collision", type=float, default=0.20)
    parser.add_argument("--near-collision-start-ratio", type=float, default=0.65)
    parser.add_argument("--offtrack-leniency-ratio", type=float, default=0.25)
    parser.add_argument("--offtrack-leniency-mult", type=float, default=2.5)

    # Infra
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--exp-tag", type=str, default=None)
    parser.add_argument("--resume-latest", action="store_true", default=False)
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--disable-preflight-checks", action="store_true", default=False)
    parser.add_argument("--disable-file-metrics-log", action="store_true", default=False)
    parser.add_argument("--file-metrics-log-freq", type=int, default=1000)
    parser.add_argument("--file-metrics-log-name", type=str, default="train_metrics.jsonl")
    parser.add_argument("--disable-auto-lr-decay", action="store_true", default=False)
    parser.add_argument("--auto-lr-check-freq", type=int, default=1000)
    parser.add_argument("--auto-lr-decay-factor", type=float, default=0.92)
    parser.add_argument("--auto-lr-min", type=float, default=7e-5)
    parser.add_argument("--auto-lr-high-kl", type=float, default=0.05)
    parser.add_argument("--auto-lr-high-kl-patience", type=int, default=3)
    parser.add_argument("--auto-lr-balanced-drop", type=float, default=12.0)
    parser.add_argument("--auto-lr-balanced-patience", type=int, default=12)
    parser.add_argument("--auto-lr-cooldown-checks", type=int, default=15)
    parser.add_argument("--auto-lr-warmup-steps", type=int, default=250000)
    parser.add_argument("--auto-lr-best-window", type=int, default=50)

    # Curriculum
    parser.add_argument("--curriculum-auto", action="store_true", default=False,
                        help="自动按课程阶段连续训练（默认 S1 30%后可提前晋级，60%兜底切到 S2）")
    parser.add_argument("--no-curriculum-auto", action="store_false", dest="curriculum_auto")
    parser.add_argument("--curriculum-stages", type=str, default="S1,S2",
                        help="自动课程阶段顺序，例如 S1,S2")
    parser.add_argument("--curriculum-ratios", type=str, default="0.60,0.40",
                        help="各阶段步数占比，逗号分隔")

    args = parser.parse_args()

    raw_argv = list(sys.argv[1:])

    def _has_cli_flag(flag: str) -> bool:
        return any(tok == flag or tok.startswith(flag + "=") for tok in raw_argv)

    def _apply_if_not_set(attr: str, value, flag: str) -> None:
        if not _has_cli_flag(flag):
            setattr(args, attr, value)

    # Profile overrides
    profile_overrides = {
        "v13_clean": [
            ("--w-center", "w_center", 0.03),
            ("--w-heading", "w_heading", 0.015),
            ("--w-m", "w_m", 0.0),
            ("--w-sat", "w_sat", 0.0),
            ("--w-speed-ref", "w_speed_ref", 0.0),
            ("--w-near-offtrack", "w_near_offtrack", 0.55),
            ("--w-near-collision", "w_near_collision", 0.20),
            ("--progress-reward-scale", "progress_reward_scale", 48.0),
            ("--adapter-max-throttle", "adapter_max_throttle", 0.3),
            ("--adapter-k-delta", "adapter_k_delta", 0.15),
            ("--adapter-k-bias", "adapter_k_bias", 0.15),
            ("--ent-coef", "ent_coef", 0.01),
            ("--near-offtrack-start-ratio", "near_offtrack_start_ratio", 0.45),
            ("--near-collision-start-ratio", "near_collision_start_ratio", 0.65),
            ("--collision-penalty-base", "collision_penalty_base", 8.0),
            ("--offtrack-penalty-base", "offtrack_penalty_base", 5.0),
            ("--offtrack-leniency-ratio", "offtrack_leniency_ratio", 0.25),
            ("--offtrack-leniency-mult", "offtrack_leniency_mult", 2.5),
            ("--min-episodes-per-scene", "min_episodes_per_scene", 5),
            ("--max-steps-per-scene", "max_steps_per_scene", 640),
            ("--dynamic-weight-alpha", "dynamic_weight_alpha", 1.6),
            ("--dynamic-weight-max", "dynamic_weight_max", 0.55),
            ("--learning-rate", "learning_rate", 8e-5),
        ],
        "v13_softsafe": [
            ("--w-center", "w_center", 0.03),
            ("--w-heading", "w_heading", 0.015),
            ("--w-m", "w_m", 0.0),
            ("--w-sat", "w_sat", 0.0),
            ("--w-speed-ref", "w_speed_ref", 0.0),
            ("--w-near-offtrack", "w_near_offtrack", 0.35),
            ("--w-near-collision", "w_near_collision", 0.20),
            ("--progress-reward-scale", "progress_reward_scale", 48.0),
            ("--adapter-max-throttle", "adapter_max_throttle", 0.3),
            ("--adapter-k-delta", "adapter_k_delta", 0.15),
            ("--adapter-k-bias", "adapter_k_bias", 0.15),
            ("--ent-coef", "ent_coef", 0.01),
            ("--near-offtrack-start-ratio", "near_offtrack_start_ratio", 0.50),
            ("--near-collision-start-ratio", "near_collision_start_ratio", 0.65),
            ("--collision-penalty-base", "collision_penalty_base", 8.0),
            ("--offtrack-penalty-base", "offtrack_penalty_base", 5.0),
            ("--offtrack-leniency-ratio", "offtrack_leniency_ratio", 0.25),
            ("--offtrack-leniency-mult", "offtrack_leniency_mult", 2.5),
            ("--min-episodes-per-scene", "min_episodes_per_scene", 5),
            ("--max-steps-per-scene", "max_steps_per_scene", 640),
            ("--dynamic-weight-alpha", "dynamic_weight_alpha", 1.6),
            ("--dynamic-weight-max", "dynamic_weight_max", 0.55),
            ("--learning-rate", "learning_rate", 8e-5),
        ],
    }

    for flag, attr, value in profile_overrides.get(args.train_profile, []):
        _apply_if_not_set(attr, value, flag)

    print(
        f"🆕 V13 profile={args.train_profile} | "
        f"prog={args.progress_reward_scale:.1f}, w_d={args.w_d:.2f}, "
        f"near_off={args.w_near_offtrack:.2f}, near_col={args.w_near_collision:.2f}, "
        f"thr={args.adapter_max_throttle:.2f}, ent={args.ent_coef:.4f}"
    )

    def _run_single(
        env_ids: Optional[List[str]],
        scene_weights: Optional[List[float]],
        total_timesteps: int,
        save_dir: str,
        exp_tag: Optional[str],
        resume_latest: bool,
        resume_path: Optional[str],
        step_balance_mask: Optional[List[bool]] = None,
        dynamic_weight_max=None,
        extra_callbacks: Optional[List[BaseCallback]] = None,
    ) -> Dict[str, Any]:
        _dwm = dynamic_weight_max if dynamic_weight_max is not None else args.dynamic_weight_max
        return train_v13(
            env_ids=env_ids,
            scene_weights=scene_weights,
            track_dir=args.track_dir,
            sim_path=args.sim,
            total_timesteps=total_timesteps,
            save_dir=save_dir,
            port=args.port,
            obs_size=args.obs_size,
            augment=args.augment,
            yellow_dropout_prob=args.yellow_dropout_prob,
            dropout_start_step=args.dropout_start_step,
            dropout_ramp_steps=args.dropout_ramp_steps,
            use_lstm=args.use_lstm,
            lstm_hidden_size=args.lstm_hidden_size,
            lstm_layers=args.lstm_layers,
            adapter_k_delta=args.adapter_k_delta,
            adapter_lambda_bias=args.adapter_lambda_bias,
            adapter_k_bias=args.adapter_k_bias,
            adapter_steer_core_decay=args.adapter_steer_core_decay,
            adapter_v_nominal=args.adapter_v_nominal,
            adapter_k_turn=args.adapter_k_turn,
            adapter_k_bias_speed=args.adapter_k_bias_speed,
            adapter_alpha_speed=args.adapter_alpha_speed,
            adapter_v_min=args.adapter_v_min,
            adapter_v_max=args.adapter_v_max,
            adapter_max_throttle=args.adapter_max_throttle,
            speed_vmax=args.speed_vmax,
            speed_kp=args.speed_kp,
            speed_ki=args.speed_ki,
            speed_kff=args.speed_kff,
            allow_reverse=args.allow_reverse,
            control_dt=args.control_dt,
            learning_rate=args.learning_rate,
            ent_coef=args.ent_coef,
            ppo_n_steps=args.ppo_n_steps,
            ppo_batch_size=args.ppo_batch_size,
            ppo_n_epochs=args.ppo_n_epochs,
            ppo_clip_range=args.ppo_clip_range,
            target_kl=args.target_kl,
            min_episodes_per_scene=args.min_episodes_per_scene,
            max_steps_per_scene=args.max_steps_per_scene,
            enable_dynamic_scene_weights=not args.disable_dynamic_scene_weights,
            dynamic_weight_update_episodes=args.dynamic_weight_update_episodes,
            dynamic_weight_window=args.dynamic_weight_window,
            dynamic_min_samples_per_scene=args.dynamic_min_samples_per_scene,
            dynamic_weight_alpha=args.dynamic_weight_alpha,
            dynamic_length_beta=args.dynamic_length_beta,
            dynamic_weight_smoothing=args.dynamic_weight_smoothing,
            dynamic_weight_min=args.dynamic_weight_min,
            dynamic_weight_max=_dwm,
            dynamic_success_mode=args.dynamic_success_mode,
            dynamic_success_warmup_episodes=args.dynamic_success_warmup_episodes,
            dynamic_success_post_warmup_scale=args.dynamic_success_post_warmup_scale,
            dynamic_success_deficit_mix=args.dynamic_success_deficit_mix,
            enable_step_balance_sampling=not args.disable_step_balance_sampling,
            step_balance_sampling_mix=args.step_balance_sampling_mix,
            step_balance_mask=step_balance_mask,
            delta_max=args.delta_max,
            enable_lpf=args.enable_lpf,
            beta=args.beta,
            w_d=args.w_d,
            w_dd=args.w_dd,
            w_m=args.w_m,
            w_sat=args.w_sat,
            w_time=args.w_time,
            w_center=args.w_center,
            w_heading=args.w_heading,
            w_speed_ref=args.w_speed_ref,
            speed_ref_vmin=args.speed_ref_vmin,
            speed_ref_vmax=args.speed_ref_vmax,
            speed_ref_kappa_ref=args.speed_ref_kappa_ref,
            lap_reward_scale=args.lap_reward_scale,
            progress_reward_scale=args.progress_reward_scale,
            survival_reward_scale=args.survival_reward_scale,
            collision_penalty_base=args.collision_penalty_base,
            offtrack_penalty_base=args.offtrack_penalty_base,
            adaptive_delta_max=not args.disable_adaptive_delta_max,
            curve_delta_boost=args.curve_delta_boost,
            curve_kappa_ref=args.curve_kappa_ref,
            steer_intent_boost=args.steer_intent_boost,
            hairpin_curve_ratio=args.hairpin_curve_ratio,
            hairpin_min_delta_max=args.hairpin_min_delta_max,
            hairpin_max_delta_max=args.hairpin_max_delta_max,
            w_near_offtrack=args.w_near_offtrack,
            near_offtrack_start_ratio=args.near_offtrack_start_ratio,
            w_near_collision=args.w_near_collision,
            near_collision_start_ratio=args.near_collision_start_ratio,
            offtrack_leniency_ratio=args.offtrack_leniency_ratio,
            offtrack_leniency_mult=args.offtrack_leniency_mult,
            sim_loaded_timeout_s=args.sim_loaded_timeout_s,
            sim_wait_resend_scene_names_s=args.sim_wait_resend_scene_names_s,
            seed=args.seed,
            exp_tag=exp_tag,
            resume_latest=resume_latest,
            resume_path=resume_path,
            run_preflight_checks=not args.disable_preflight_checks,
            enable_file_metrics_log=not args.disable_file_metrics_log,
            file_metrics_log_freq=args.file_metrics_log_freq,
            file_metrics_log_name=args.file_metrics_log_name,
            enable_auto_lr_decay=not args.disable_auto_lr_decay,
            auto_lr_check_freq=args.auto_lr_check_freq,
            auto_lr_decay_factor=args.auto_lr_decay_factor,
            auto_lr_min=args.auto_lr_min,
            auto_lr_high_kl=args.auto_lr_high_kl,
            auto_lr_high_kl_patience=args.auto_lr_high_kl_patience,
            auto_lr_balanced_drop=args.auto_lr_balanced_drop,
            auto_lr_balanced_patience=args.auto_lr_balanced_patience,
            auto_lr_cooldown_checks=args.auto_lr_cooldown_checks,
            auto_lr_warmup_steps=args.auto_lr_warmup_steps,
            auto_lr_best_window=args.auto_lr_best_window,
            extra_callbacks=extra_callbacks,
        )

    use_curriculum_auto = bool(
        args.curriculum_auto and args.env_ids is None and args.stage is None
    )
    if use_curriculum_auto:
        stages = [s.strip().upper() for s in args.curriculum_stages.split(",") if s.strip()]
        if not stages:
            raise ValueError("--curriculum-stages 为空")
        for s in stages:
            if s not in STAGE_ENV_IDS:
                raise ValueError(f"未知课程阶段: {s}")

        raw_ratios = [float(x.strip()) for x in args.curriculum_ratios.split(",") if x.strip()]
        if len(raw_ratios) != len(stages):
            raise ValueError(
                f"--curriculum-ratios 数量({len(raw_ratios)})必须与阶段数量({len(stages)})一致"
            )
        ratio_sum = float(sum(raw_ratios))
        if ratio_sum <= 0:
            raise ValueError("--curriculum-ratios 总和必须 > 0")
        ratios = [r / ratio_sum for r in raw_ratios]

        stage_steps = [max(1, int(args.steps * r)) for r in ratios]
        stage_steps[-1] = int(args.steps) - sum(stage_steps[:-1])
        if stage_steps[-1] <= 0:
            stage_steps[-1] = 1

        print("\n📚 自动课程学习已启用")
        print(f"   阶段顺序: {' -> '.join(stages)}")
        print(f"   总步数: {args.steps}")
        print(f"   阶段最大步数: {stage_steps}")
        if "S1" in stages and "S1" in CURRICULUM_STAGE_ADVANCE_RULES:
            _rule = CURRICULUM_STAGE_ADVANCE_RULES["S1"]
            print(
                "   S1 提前晋级: "
                f"总步数>{int(args.steps * float(_rule['advance_after_ratio']))} 后, "
                f"{'+'.join(_rule['required_logging_keys'])} 最近{int(_rule['recent_episodes'])}局"
                f"每局 >= {float(_rule['min_laps_per_episode']):.1f} lap"
            )
        if args.scene_weights is not None:
            print("⚠️  自动课程模式下忽略 --scene-weights（每阶段用预设权重）")

        prev_resume_path = args.resume_path
        prev_resume_latest = bool(args.resume_latest and (prev_resume_path is None))
        consumed_total_steps = 0
        for i, stage_name in enumerate(stages):
            remaining_total_steps = max(0, int(args.steps) - int(consumed_total_steps))
            if remaining_total_steps <= 0:
                print(f"⏹️  总步数已用尽，跳过后续阶段: {stage_name}")
                break
            stage_env_ids = STAGE_ENV_IDS[stage_name]
            stage_scene_weights = STAGE_SCENE_WEIGHTS.get(stage_name)
            stage_mask = STAGE_STEP_BALANCE_MASK.get(stage_name)
            stage_wmax = STAGE_DYNAMIC_WEIGHT_MAX.get(stage_name, args.dynamic_weight_max)
            stage_dir = os.path.join(args.save_dir, f"stage_{stage_name}")
            stage_exp_tag = (
                (f"{args.exp_tag}_stage_{stage_name}" if args.exp_tag else f"stage_{stage_name}")
            )
            stage_short = [SCENE_SPECS[eid]["logging_key"] for eid in stage_env_ids]
            stage_budget = remaining_total_steps if (i >= len(stages) - 1) else min(stage_steps[i], remaining_total_steps)
            stage_extra_callbacks: List[BaseCallback] = []
            stage_gate_cb: Optional[CurriculumStageAdvanceCallback] = None
            stage_hard_stop_cb: Optional[StrictStageTimestepsStopCallback] = None
            if i == 0 and stage_name == "S1":
                stage_rule = CURRICULUM_STAGE_ADVANCE_RULES.get(stage_name)
                if stage_rule is not None:
                    stage_gate_cb = CurriculumStageAdvanceCallback(
                        stage_name=stage_name,
                        required_logging_keys=list(stage_rule["required_logging_keys"]),
                        min_total_timesteps=int(args.steps * float(stage_rule["advance_after_ratio"])),
                        recent_episodes=int(stage_rule["recent_episodes"]),
                        min_laps_per_episode=float(stage_rule["min_laps_per_episode"]),
                        max_total_timesteps=int(args.steps * ratios[i]),
                        verbose=1,
                    )
                    stage_extra_callbacks.append(stage_gate_cb)
            if stage_gate_cb is None:
                stage_hard_stop_cb = StrictStageTimestepsStopCallback(
                    stage_name=stage_name,
                    max_stage_timesteps=stage_budget,
                    verbose=1,
                )
                stage_extra_callbacks.append(stage_hard_stop_cb)
            print(
                f"\n🎯 课程阶段 {stage_name} | maps={'+'.join(stage_short)} | "
                f"steps={stage_budget} | save_dir={stage_dir}"
            )

            stage_result = _run_single(
                env_ids=stage_env_ids,
                scene_weights=stage_scene_weights,
                total_timesteps=stage_budget,
                save_dir=stage_dir,
                exp_tag=stage_exp_tag,
                resume_latest=prev_resume_latest,
                resume_path=prev_resume_path,
                step_balance_mask=stage_mask,
                dynamic_weight_max=stage_wmax,
                extra_callbacks=stage_extra_callbacks,
            )

            stage_total_timesteps = int(stage_result.get("num_timesteps_total", consumed_total_steps))
            stage_trained_timesteps = int(stage_result.get("trained_timesteps", 0))
            if stage_total_timesteps > 0:
                consumed_total_steps = stage_total_timesteps
            else:
                consumed_total_steps += max(0, stage_trained_timesteps)

            prev_resume_path = str(stage_result.get("final_model_zip", os.path.join(stage_dir, "final_model.zip")))
            prev_resume_latest = False
            if stage_gate_cb is not None:
                gate_summary = stage_gate_cb.summary()
                print(
                    f"   阶段结果[{stage_name}]: "
                    f"trained={stage_trained_timesteps}, total={consumed_total_steps}, "
                    f"stop={gate_summary['stop_reason'] or 'natural_end'}"
                )
            elif stage_hard_stop_cb is not None:
                stop_summary = stage_hard_stop_cb.summary()
                print(
                    f"   阶段结果[{stage_name}]: "
                    f"trained={stage_trained_timesteps}, total={consumed_total_steps}, "
                    f"stop={stop_summary['stop_reason'] or 'natural_end'}"
                )
    else:
        effective_env_ids = args.env_ids
        effective_scene_weights = args.scene_weights
        effective_mask = None
        if effective_env_ids is None and args.stage is None:
            print("ℹ️  未指定 --env-ids/--stage：将使用 DEFAULT_ENV_IDS (ws+gt+rrl 3 场景)。")
        effective_dynamic_weight_max = args.dynamic_weight_max  # CLI global default
        if effective_env_ids is None and args.stage is not None:
            effective_env_ids = STAGE_ENV_IDS[args.stage]
            effective_mask = STAGE_STEP_BALANCE_MASK.get(args.stage)
            stage_names = [SCENE_SPECS[eid]["logging_key"] for eid in effective_env_ids]
            print(f"📋 Stage {args.stage} 训练: {'+'.join(stage_names)} ({len(effective_env_ids)} 场景)")
            if effective_scene_weights is None:
                effective_scene_weights = STAGE_SCENE_WEIGHTS.get(args.stage)
                if effective_scene_weights is not None:
                    wtxt = ",".join(f"{w:.2f}" for w in effective_scene_weights)
                    print(f"🎚️  Stage {args.stage} 默认权重: [{wtxt}]")
            # per-scene dynamic_weight_max
            _stage_wmax = STAGE_DYNAMIC_WEIGHT_MAX.get(args.stage)
            if _stage_wmax is not None:
                effective_dynamic_weight_max = _stage_wmax
                print(f"🎚️  Stage {args.stage} per-scene weight_max: {[f'{x:.2f}' for x in _stage_wmax]}")

        _run_single(
            env_ids=effective_env_ids,
            scene_weights=effective_scene_weights,
            total_timesteps=args.steps,
            save_dir=args.save_dir,
            exp_tag=args.exp_tag,
            resume_latest=args.resume_latest,
            resume_path=args.resume_path,
            step_balance_mask=effective_mask,
            dynamic_weight_max=effective_dynamic_weight_max,
        )
