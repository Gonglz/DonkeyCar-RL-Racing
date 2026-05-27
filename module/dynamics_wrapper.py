"""Dynamics-aligned gym environment wrapper for sim2real transfer.

Canonical module-local copy of the simulator-side action alignment wrapper.
It mirrors ``donkeycar.parts.dgym.DonkeyGymEnv`` while transforming actions
before they are sent to DonkeySim.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, Optional


def _is_exe(fpath: str) -> bool:
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def _import_gym_donkeycar():
    try:
        import gym  # type: ignore
        import gym_donkeycar  # noqa: F401  # type: ignore
        return gym
    except ImportError:
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src", "donkeycar"))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)
        import gym  # type: ignore
        import gym_donkeycar  # noqa: F401  # type: ignore
        return gym


class DynamicsAlignedGymEnv:
    """Drop-in replacement for ``DonkeyGymEnv`` with action-side alignment."""

    def __init__(
        self,
        sim_path: str,
        host: str = "127.0.0.1",
        port: int = 9091,
        headless: int = 0,
        env_name: str = "donkey-generated-track-v0",
        sync: str = "asynchronous",
        conf: Optional[Dict] = None,
        dynamics_json: str = "",
        record_location: bool = False,
        record_gyroaccel: bool = False,
        record_velocity: bool = False,
        record_lidar: bool = False,
        delay: int = 0,
    ):
        del headless, sync  # kept for interface compatibility with DonkeyGymEnv

        conf = dict(conf or {})

        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise FileNotFoundError(f"Simulator path does not exist: {sim_path}")
            if not _is_exe(sim_path):
                raise PermissionError(f"Simulator path is not executable: {sim_path}")

        gym = _import_gym_donkeycar()
        conf["exe_path"] = sim_path
        conf["host"] = host
        conf["port"] = port
        conf["guid"] = 0
        conf["frame_skip"] = 1

        self.env = gym.make(env_name, conf=conf)
        self.frame = self.env.reset()
        self.action = [0.0, 0.0, 0.0]
        self.running = True
        self.info: Dict = {
            "pos": (0.0, 0.0, 0.0),
            "speed": 0,
            "cte": 0,
            "gyro": (0.0, 0.0, 0.0),
            "accel": (0.0, 0.0, 0.0),
            "vel": (0.0, 0.0, 0.0),
        }
        self.delay = float(delay) / 1000.0
        self.record_location = bool(record_location)
        self.record_gyroaccel = bool(record_gyroaccel)
        self.record_velocity = bool(record_velocity)
        self.record_lidar = bool(record_lidar)
        self.buffer = []

        self._load_dynamics(dynamics_json)
        self._filtered_steer = 0.0
        self._filtered_throttle = 0.0
        self._last_time: Optional[float] = None

    def _load_dynamics(self, json_path: str) -> None:
        self.dynamics_enabled = False

        if not json_path or not os.path.isfile(json_path):
            print(
                f"[dynamics_wrapper] no dynamics JSON at {json_path!r}, wrapper pass-through mode"
            )
            self._steer_gain_ratio = 1.0
            self._tau_steer_extra = 0.0
            self._hw_deadband = 0.0
            self._throttle_gain_ratio = 1.0
            self._tau_throttle_extra = 0.0
            return

        with open(json_path, "r", encoding="utf-8") as fp:
            params = json.load(fp)

        alignment = params.get("alignment", {})
        hw = params.get("hardware", {})

        self._steer_gain_ratio = float(alignment.get("steer_gain_ratio", 1.0))
        self._tau_steer_extra = float(max(0.0, alignment.get("steer_tau_delta", 0.0)))
        self._hw_deadband = float(hw.get("deadband_steer", 0.0))

        self._throttle_gain_ratio = float(alignment.get("throttle_gain_ratio", 1.0))
        self._tau_throttle_extra = float(max(0.0, alignment.get("throttle_tau_delta", 0.0)))

        self.dynamics_enabled = True
        print(f"[dynamics_wrapper] loaded from {json_path}")
        print(
            f"  steer_gain_ratio={self._steer_gain_ratio:.4f}, "
            f"tau_steer_extra={self._tau_steer_extra:.4f}s, "
            f"deadband={self._hw_deadband:.3f}"
        )
        print(
            f"  throttle_gain_ratio={self._throttle_gain_ratio:.4f}, "
            f"tau_throttle_extra={self._tau_throttle_extra:.4f}s"
        )

    def _transform_action(self, steering: float, throttle: float) -> tuple:
        if not self.dynamics_enabled:
            return steering, throttle

        now = time.time()
        dt = 0.05 if self._last_time is None else max(now - self._last_time, 1e-3)
        self._last_time = now

        if abs(steering) < self._hw_deadband:
            effective_steer = 0.0
        else:
            effective_steer = float(steering)

        scaled_steer = effective_steer * self._steer_gain_ratio
        if self._tau_steer_extra > 1e-3:
            alpha = 1.0 - math.exp(-dt / self._tau_steer_extra)
            self._filtered_steer += alpha * (scaled_steer - self._filtered_steer)
        else:
            self._filtered_steer = scaled_steer
        out_steer = max(-1.0, min(1.0, self._filtered_steer))

        scaled_throttle = float(throttle) * self._throttle_gain_ratio
        if self._tau_throttle_extra > 1e-3:
            alpha_t = 1.0 - math.exp(-dt / self._tau_throttle_extra)
            self._filtered_throttle += alpha_t * (scaled_throttle - self._filtered_throttle)
        else:
            self._filtered_throttle = scaled_throttle
        out_throttle = max(-1.0, min(1.0, self._filtered_throttle))

        return out_steer, out_throttle

    def delay_buffer(self, frame, info) -> None:
        now = time.time()
        self.buffer.append((now, frame, info))
        num_to_remove = 0
        for buf in self.buffer:
            if now - buf[0] >= self.delay:
                num_to_remove += 1
                self.frame = buf[1]
            else:
                break
        del self.buffer[:num_to_remove]

    def update(self) -> None:
        while self.running:
            steering, throttle = self._transform_action(self.action[0], self.action[1])
            brake = self.action[2] if len(self.action) > 2 else 0.0
            transformed_action = [steering, throttle, brake]

            if self.delay > 0.0:
                current_frame, _, _, current_info = self.env.step(transformed_action)
                self.delay_buffer(current_frame, current_info)
            else:
                self.frame, _, _, self.info = self.env.step(transformed_action)

    def run_threaded(self, steering, throttle, brake=None):
        if steering is None or throttle is None:
            steering = 0.0
            throttle = 0.0
        if brake is None:
            brake = 0.0

        self.action = [steering, throttle, brake]

        outputs = [self.frame]
        if self.record_location:
            outputs += (
                self.info["pos"][0],
                self.info["pos"][1],
                self.info["pos"][2],
                self.info["speed"],
                self.info["cte"],
            )
        if self.record_gyroaccel:
            outputs += (
                self.info["gyro"][0],
                self.info["gyro"][1],
                self.info["gyro"][2],
                self.info["accel"][0],
                self.info["accel"][1],
                self.info["accel"][2],
            )
        if self.record_velocity:
            outputs += (
                self.info["vel"][0],
                self.info["vel"][1],
                self.info["vel"][2],
            )
        if self.record_lidar:
            outputs += self.info["lidar"]
        if len(outputs) == 1:
            return self.frame
        return outputs

    def shutdown(self) -> None:
        self.running = False
        time.sleep(0.2)
        self.env.close()
