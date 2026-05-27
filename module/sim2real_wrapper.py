"""
module/sim2real_wrapper.py

Sim2Real note Wrapper - trainingnote----
notetrainingstagenote Sim note, note PPO note.

note(MultiSceneEnvV16._create_env note):
    base_env (gym DonkeyEnv)
         ↓
    ScenarioObstacleWrapper
         ↓
    Sim2RealActionWrapper   <- note, note steer / throttle
         ↓
    CanonicalSemanticWrapper / DonkeyRewardWrapper / ActionSafetyWrapper / ActionAdapterWrapper...

note
--------
note mysim/tools/calibrate_sim2real.py note wm_real + wm_sim notecomputenote JSON:

    {
      "throttle_gain_ratio": 0.115,
      "steer_gain_ratio": 0.715,
      "steer_tau_s": 0.0,
      "throttle_tau_s": 0.0,
      "source": "wm_calibration",
      "calibrated_at": "2026-04-19"
    }

notefilenote
-----------
mysim/models/world_model/dynamics_alignment_wm.json
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Union

import gym
import numpy as np


class Sim2RealActionWrapper(gym.ActionWrapper):
    """
    note gym.ActionWrapper.

    note DonkeySim note [steer, throttle] note + note,
    note Sim note.

    Parameters
    ----------
    env: gym.Env
        note.
    throttle_gain: float
        note.< 1.0 note sim note; note 0.115.
    steer_gain: float
        note.< 1.0 note sim note; note 0.715.
    steer_tau_s: float
        note(note).0.0 = note.
    throttle_tau_s: float
        note(note).0.0 = note.
    """

    def __init__(
        self,
        env: gym.Env,
        throttle_gain: float = 1.0,
        steer_gain: float = 1.0,
        steer_tau_s: float = 0.0,
        throttle_tau_s: float = 0.0,
    ):
        super().__init__(env)
        self.throttle_gain  = float(max(0.01, throttle_gain))
        self.steer_gain     = float(max(0.01, steer_gain))
        self.steer_tau_s    = float(max(0.0, steer_tau_s))
        self.throttle_tau_s = float(max(0.0, throttle_tau_s))

        self._filtered_steer    = 0.0
        self._filtered_throttle = 0.0
        self._last_t: Optional[float] = None

        print(
            f"[Sim2RealActionWrapper] "
            f"throttle_gain={self.throttle_gain:.4f}, "
            f"steer_gain={self.steer_gain:.4f}, "
            f"steer_tau={self.steer_tau_s:.3f}s, "
            f"throttle_tau={self.throttle_tau_s:.3f}s"
        )

    @classmethod
    def from_json(cls, env: gym.Env, json_path: Union[str, Path]) -> "Sim2RealActionWrapper":
        """note JSON filenote."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Sim2Real calibration JSON not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            params = json.load(f)
        inst = cls(
            env,
            throttle_gain  = float(params.get("throttle_gain_ratio", 1.0)),
            steer_gain     = float(params.get("steer_gain_ratio",    1.0)),
            steer_tau_s    = float(params.get("steer_tau_s",    0.0)),
            throttle_tau_s = float(params.get("throttle_tau_s", 0.0)),
        )
        src = params.get("source", "unknown")
        cal = params.get("calibrated_at", "?")
        print(f"  loaded from {path.name}  (source={src}, date={cal})")
        return inst

    def action(self, action: np.ndarray) -> np.ndarray:
        steer    = float(action[0])
        throttle = float(action[1])

        # note
        steer    *= self.steer_gain
        throttle *= self.throttle_gain

        # note
        now = time.monotonic()
        dt  = 0.05 if self._last_t is None else max(now - self._last_t, 1e-3)
        self._last_t = now

        if self.steer_tau_s > 1e-4:
            alpha = 1.0 - math.exp(-dt / self.steer_tau_s)
            self._filtered_steer += alpha * (steer - self._filtered_steer)
            steer = self._filtered_steer

        if self.throttle_tau_s > 1e-4:
            alpha_t = 1.0 - math.exp(-dt / self.throttle_tau_s)
            self._filtered_throttle += alpha_t * (throttle - self._filtered_throttle)
            throttle = self._filtered_throttle

        steer    = float(np.clip(steer,    -1.0, 1.0))
        throttle = float(np.clip(throttle, -1.0, 1.0))

        out = action.copy() if isinstance(action, np.ndarray) else np.array(action, dtype=np.float32)
        out[0] = steer
        out[1] = throttle
        return out

    def reset(self, **kwargs):
        self._filtered_steer    = 0.0
        self._filtered_throttle = 0.0
        self._last_t = None
        return self.env.reset(**kwargs)


__all__ = ["Sim2RealActionWrapper"]
