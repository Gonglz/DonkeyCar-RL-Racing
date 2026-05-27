"""
module/predictive_safety_filter.py

H note - V1

note
--------
note ActionAdapterWrapper output(steer_target, throttle)note,
ActionSafetyWrapper notefirst:

    PPO -> ActionAdapterWrapper
                ↓ [steer_target, throttle]
         PredictiveSafetyFilter.check()     <- note
                ↓ note: note
         ActionSafetyWrapper -> DonkeyEnv

notestage
--------
Phase 1(log-only, currentdefault):
    note, note.note H=3 notefirstnote, note.
    noteresult(note, note, note)note.

Phase 2(intervene, note):
    note steer_target, note throttle, note.

note
--------
- NeuralPhysicsDynamics(wm_real.pth): firstnote
- ActionSafetyWrapper note: delta_max=0.5, beta=0.6
  note wrapper note, note.

note
--------
    from module import PredictiveSafetyFilter, PhysState

    flt = PredictiveSafetyFilter(
        model_path="models/world_model/wm_real.pth",
        horizon=3,
        mode="log",
        log_path="safety_filter_events.jsonl",
    )

    # note episode note
    flt.reset()

    # note
    triggered, preds, diag = flt.check(steer_target, throttle, phys, dt_ms=50.0)

    # env.step noterowsnote, note safety wrapper note
    flt.sync(safety_wrapper.steer_prev_limited, safety_wrapper.steer_prev_exec)

    # note
    flt.print_stats()
    flt.close()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from.world_model import NeuralPhysicsDynamics


# ─── note(note _build_state_v13 note) ────────────────────

V_MAX    = 2.2   # m/s, v_long_norm = speed / V_MAX
GYRO_MAX = 4.0   # rad/s, yaw_rate_norm = -gyro_z / GYRO_MAX
ACCEL_MAX = 9.8  # m/s², accel_x_norm = accel_x / ACCEL_MAX


# ─── notedataclass ────────────────────────────────────────────────

@dataclass
class PhysState:
    """
    note(notemodelinputnote).

    note
    ----
    v_long: float  ∈ [0,  2]   speed / 2.2
    yaw_rate: float  ∈ [-2, 2]   -gyro_z / 4.0
    accel_x: float  ∈ [-2, 2]   accel_x / 9.8
    """
    v_long: float
    yaw_rate: float
    accel_x: float

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [self.v_long, self.yaw_rate, self.accel_x], dtype=torch.float32
        )

    @classmethod
    def from_raw(cls, speed_mps: float, gyro_z: float, accel_x: float) -> "PhysState":
        """note(note)."""
        import numpy as np
        return cls(
            v_long   = float(np.clip(speed_mps / V_MAX,     0.0,  2.0)),
            yaw_rate = float(np.clip(-gyro_z   / GYRO_MAX, -2.0,  2.0)),
            accel_x  = float(np.clip(accel_x   / ACCEL_MAX,-2.0,  2.0)),
        )


# ─── note ActionSafetyWrapper note ──────────────────────────────────

@dataclass
class _ShadowSafetyState:
    """
    note ActionSafetyWrapper note, notefirstnote.
    note control.py::ActionSafetyWrapper note.
    """
    steer_prev_limited: float = 0.0
    steer_prev_exec: float = 0.0
    delta_max: float = 0.5
    beta: float = 0.6

    def step(self, steer_target: float) -> Tuple[float, float]:
        """
        noterowsnote, note (steer_exec, steer_limited), note.
        note ActionSafetyWrapper.action() note.
        """
        delta = steer_target - self.steer_prev_limited
        if abs(delta) > self.delta_max:
            delta = max(-self.delta_max, min(self.delta_max, delta))
        steer_limited = self.steer_prev_limited + delta
        steer_exec = (1.0 - self.beta) * self.steer_prev_exec + self.beta * steer_limited
        steer_exec = max(-1.0, min(1.0, steer_exec))
        self.steer_prev_limited = steer_limited
        self.steer_prev_exec = steer_exec
        return steer_exec, steer_limited

    def copy(self) -> "_ShadowSafetyState":
        return _ShadowSafetyState(
            steer_prev_limited=self.steer_prev_limited,
            steer_prev_exec=self.steer_prev_exec,
            delta_max=self.delta_max,
            beta=self.beta,
        )


# ─── note ─────────────────────────────────────────────────────

class PredictiveSafetyFilter:
    """
    H note.

    Parameters
    ----------
    model_path: str
        wm_real.pth notepath.
    horizon: int
        firstnote, default 3.
    delta_max: float
        ActionSafetyWrapper note, note wrapper note(default 0.5).
    beta: float
        ActionSafetyWrapper note LPF note, note wrapper note(default 0.6).
    yaw_thresh: float or None
        |yaw_rate_norm| note.None = note(Phase 1 note).
    decel_thresh: float or None
        note Δv_norm < -decel_thresh note(note).None = note.
    mode: "log" or "intervene"
        "log" = Phase 1, note;
        "intervene" = Phase 2, note.
    log_path: str
        note JSONL notepath.note = note.
    device: str
        note(default "cpu", goal < 0.3 ms / 3 notefirstnote).
    """

    def __init__(
        self,
        model_path: str,
        horizon: int = 3,
        delta_max: float = 0.5,
        beta: float = 0.6,
        yaw_thresh: Optional[float] = None,
        decel_thresh: Optional[float] = None,
        mode: str = "log",
        log_path: str = "safety_filter_events.jsonl",
        device: str = "cpu",
    ):
        assert mode in ("log", "intervene"), (
            f"mode must be 'log' or 'intervene', got {mode!r}"
        )

        self.horizon     = int(horizon)
        self.mode        = mode
        self.yaw_thresh  = yaw_thresh
        self.decel_thresh = decel_thresh
        self.device      = device

        # notemodel
        self.model = NeuralPhysicsDynamics.load_checkpoint(model_path, device=device)
        self.model.eval()

        # note safety wrapper note
        self._shadow = _ShadowSafetyState(delta_max=delta_max, beta=beta)

        # note
        self._total_steps    = 0
        self._triggered_steps = 0
        self._episode        = 0

        # note
        self._log_path = log_path
        self._log_file = open(log_path, "a", encoding="utf-8") if log_path else None

        print(
            f"[PredictiveSafetyFilter] mode={mode}, H={horizon}, "
            f"delta_max={delta_max}, beta={beta}"
        )
        print(f"  yaw_thresh={yaw_thresh}, decel_thresh={decel_thresh}")
        print(f"  model: {model_path}")
        if log_path:
            print(f"  log: {log_path}")

    # ── note ─────────────────────────────────────────────────────

    def check(
        self,
        steer_target: float,
        throttle: float,
        phys: PhysState,
        dt_ms: float = 50.0,
    ) -> Tuple[bool, List[Dict], Dict]:
        """
        note H notefirstnote.

        Parameters
        ----------
        steer_target: float
            ActionAdapterWrapper outputnotegoalnote(note safety wrapper note).
        throttle: float
            ActionAdapterWrapper outputnote.
        phys: PhysState
            currentnote.
        dt_ms: float
            controlnote(ms), note dt_norm compute.

        Returns
        -------
        triggered: bool
            note.Phase 1 note, notecontrol.
        predictions: list[dict]
            H noteresult, note {h, v_long, yaw_rate, accel_x, delta_v, steer_exec}.
        diag: dict
            note: triggered, trigger_dims, trigger_rate, total_steps.
        """
        self._total_steps += 1

        # note(note)
        shadow = self._shadow.copy()
        phys_cur = phys.to_tensor().to(self.device)
        dt_norm = dt_ms / 50.0

        prev_steer_exec = shadow.steer_prev_exec
        prev_throttle   = float(throttle)

        predictions: List[Dict] = []
        triggered    = False
        trigger_dims: List[str] = []

        for h in range(self.horizon):
            # 1. note ActionSafetyWrapper, note steer_exec
            steer_exec, _ = shadow.step(float(steer_target))

            # 2. note 8D inputnote
            x = torch.tensor(
                [[
                    float(phys_cur[0]),   # v_long
                    float(phys_cur[1]),   # yaw_rate
                    float(phys_cur[2]),   # accel_x
                    steer_exec,
                    float(throttle),
                    prev_steer_exec,
                    prev_throttle,
                    dt_norm,
                ]],
                dtype=torch.float32,
                device=self.device,
            )

            # 3. notemodelfirstnote
            with torch.no_grad():
                delta, phys_next = self.model(x, phys_cur.unsqueeze(0))

            phys_next = phys_next.squeeze(0)
            delta     = delta.squeeze(0)

            step_pred = {
                "h":          h + 1,
                "v_long":     float(phys_next[0]),
                "yaw_rate":   float(phys_next[1]),
                "accel_x":    float(phys_next[2]),
                "delta_v":    float(delta[0]),
                "steer_exec": steer_exec,
            }
            predictions.append(step_pred)

            # 4. note(None = note, note)
            if (self.yaw_thresh is not None
                    and abs(float(phys_next[1])) > self.yaw_thresh):
                triggered = True
                if "yaw_rate" not in trigger_dims:
                    trigger_dims.append("yaw_rate")

            if (self.decel_thresh is not None
                    and float(delta[0]) < -self.decel_thresh):
                triggered = True
                if "v_long_decel" not in trigger_dims:
                    trigger_dims.append("v_long_decel")

            # 5. note
            prev_steer_exec = steer_exec
            prev_throttle   = float(throttle)
            phys_cur        = phys_next

        if triggered:
            self._triggered_steps += 1

        diag = {
            "triggered":    triggered,
            "trigger_dims": trigger_dims,
            "trigger_rate": self._triggered_steps / max(self._total_steps, 1),
            "total_steps":  self._total_steps,
        }

        # note(note + note 500 note)
        if triggered or (self._total_steps % 500 == 0):
            self._write_log(steer_target, throttle, phys, predictions, trigger_dims)

        return triggered, predictions, diag

    def sync(self, steer_prev_limited: float, steer_prev_exec: float) -> None:
        """
        note env.step() noterowsnote, note ActionSafetyWrapper note.

        note:
            flt.sync(
                safety_wrapper.steer_prev_limited,
                safety_wrapper.steer_prev_exec,
            )

        note, note.
        """
        self._shadow.steer_prev_limited = float(steer_prev_limited)
        self._shadow.steer_prev_exec    = float(steer_prev_exec)

    def reset(self, episode: Optional[int] = None) -> None:
        """
        note episode note.note episode note.

        Parameters
        ----------
        episode: int or None
            note episode note; None = note.
        """
        self._shadow.steer_prev_limited = 0.0
        self._shadow.steer_prev_exec    = 0.0
        self._episode = int(episode) if episode is not None else self._episode + 1

    # ── note ──────────────────────────────────────────────────────

    def stats(self) -> Dict:
        """notecurrentnote."""
        return {
            "total_steps":     self._total_steps,
            "triggered_steps": self._triggered_steps,
            "trigger_rate":    self._triggered_steps / max(self._total_steps, 1),
            "mode":            self.mode,
            "horizon":         self.horizon,
            "yaw_thresh":      self.yaw_thresh,
            "decel_thresh":    self.decel_thresh,
        }

    def print_stats(self) -> None:
        s = self.stats()
        print(
            f"[SafetyFilter] steps={s['total_steps']:,}  "
            f"triggered={s['triggered_steps']:,}  "
            f"rate={s['trigger_rate']:.3%}  "
            f"mode={s['mode']}"
        )

    # ── note ──────────────────────────────────────────────────────

    def _write_log(
        self,
        steer_target: float,
        throttle: float,
        phys: PhysState,
        predictions: List[Dict],
        trigger_dims: List[str],
    ) -> None:
        if self._log_file is None:
            return
        record = {
            "ts":          time.time(),
            "episode":     self._episode,
            "step":        self._total_steps,
            "phys":        {"v_long": phys.v_long, "yaw_rate": phys.yaw_rate,
                            "accel_x": phys.accel_x},
            "action":      {"steer_target": steer_target, "throttle": throttle},
            "predictions": predictions,
            "trigger_dims": trigger_dims,
        }
        self._log_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._log_file.flush()

    def close(self) -> None:
        """notefile."""
        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def __del__(self):
        self.close()


__all__ = [
    "PredictiveSafetyFilter",
    "PhysState",
]
