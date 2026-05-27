"""
module/world_model.py

notemodel - V1(note, note)

notedescription
--------
note"notemodel":
  - note(parse): ActionAdapter + ActionSafetyWrapper note,
                    note.
  - note(notefile): note 3 note
                      [Δv_long, Δyaw_rate, Δaccel_x]

modelinput(8 note, 5 note baseline)
--------------------------------------
  [v_long_t,       yaw_rate_t,   accel_x_t,      # currentnote
   steer_exec_t,   throttle_t,                    # currentnoterowsnote(note SafetyWrapper)
   prev_steer_exec, prev_throttle, dt_norm]        # note + note(note)

  dt_norm = dt_ms / 50.0(note 50 ms note)

modeloutput(3 note)
--------------------
  [Δv_long, Δyaw_rate, Δaccel_x]
  next_state = clip(current + delta, lo, hi)

note(note _build_state_v13 note)
------------------------------------------
  v_long:   [0, 2]   = clip(speed / 2.2)
  yaw_rate: [-2, 2]  = clip(-gyro_z / 4.0)
  accel_x:  [-2, 2]  = clip(accel_x / 9.8)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple, Optional

# note
PHYS_DIM = 3   # [v_long, yaw_rate, accel_x]

# inputnote
INPUT_DIM_BASELINE = 5   # [v, ω, a, steer_exec, throttle]
INPUT_DIM_FULL     = 8   # + [prev_steer_exec, prev_throttle, dt_norm]

# note(note: v_long, yaw_rate, accel_x)
STATE_LO = torch.tensor([0.0,  -2.0, -2.0], dtype=torch.float32)
STATE_HI = torch.tensor([2.0,   2.0,  2.0], dtype=torch.float32)


class NeuralPhysicsDynamics(nn.Module):
    """
    note MLP.

    note s_{t+1}[:3] = clip(s_t[:3] + f(x_t), lo, hi)
    note f note, notetrainingnote.

    Parameters
    ----------
    input_dim: int
        5(baseline sanity check)note 8(note, note).
    hidden_dim: int
        note, default 128.
    dropout: float
        Dropout note, note eval() note.

    Usage
    -----
    model = NeuralPhysicsDynamics(input_dim=8)
    x = torch.zeros(B, 8)       # [v, ω, a, steer, thr, prev_steer, prev_thr, dt]
    phys_t = x[:,:3]           # currentnote
    delta, s_next = model(x, phys_t)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM_FULL,
        hidden_dim: int = 128,
        dropout: float = 0.05,
    ):
        super().__init__()
        assert input_dim in (INPUT_DIM_BASELINE, INPUT_DIM_FULL), (
            f"input_dim must be {INPUT_DIM_BASELINE} or {INPUT_DIM_FULL}, got {input_dim}"
        )
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
        )

        # noteoutputnote: trainingnote(note)
        self.head = nn.Linear(64, PHYS_DIM)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self,
        x: torch.Tensor,
        phys_t: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x: (..., input_dim)  float32
            noteinputnote.
        phys_t: (..., 3) or None
            currentnote [v, ω, a].note None, note x[:3] note.

        Returns
        -------
        delta: (..., 3)  note
        s_next: (..., 3)  note(note clip)
        """
        if phys_t is None:
            phys_t = x[...,:PHYS_DIM]

        h     = self.trunk(x)
        delta = self.head(h)
        s_next = phys_t + delta

        lo = STATE_LO.to(s_next.device, dtype=s_next.dtype)
        hi = STATE_HI.to(s_next.device, dtype=s_next.dtype)
        s_next = torch.clamp(s_next, lo, hi)

        return delta, s_next

    # ─── save / note ────────────────────────────────────────────

    def save_checkpoint(self, path: str, extra: Optional[dict] = None) -> None:
        """savenote(note)."""
        ckpt = {
            "model_state": self.state_dict(),
            "model_kwargs": {
                "input_dim":  self.input_dim,
                "hidden_dim": self.hidden_dim,
                "dropout":    0.0,   # note dropout
            },
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu") -> "NeuralPhysicsDynamics":
        """notemodel(eval note)."""
        ckpt = torch.load(path, map_location="cpu")
        model = cls(**ckpt["model_kwargs"])
        model.load_state_dict(ckpt["model_state"])
        model = model.to(device)
        model.eval()
        return model


# ─── notefunction: noteinputnote ─────────────────────────────────

def build_input_5d(
    v: float, yaw: float, accel: float,
    steer_exec: float, throttle: float,
) -> torch.Tensor:
    """note 5D baseline input(note, note shape (1, 5))."""
    return torch.tensor(
        [[v, yaw, accel, steer_exec, throttle]], dtype=torch.float32
    )


def build_input_8d(
    v: float, yaw: float, accel: float,
    steer_exec: float, throttle: float,
    prev_steer_exec: float, prev_throttle: float,
    dt_ms: float,
) -> torch.Tensor:
    """note 8D noteinput(note, note shape (1, 8))."""
    dt_norm = dt_ms / 50.0
    return torch.tensor(
        [[v, yaw, accel, steer_exec, throttle,
          prev_steer_exec, prev_throttle, dt_norm]],
        dtype=torch.float32,
    )


__all__ = [
    "NeuralPhysicsDynamics",
    "build_input_5d",
    "build_input_8d",
    "INPUT_DIM_BASELINE",
    "INPUT_DIM_FULL",
    "PHYS_DIM",
    "STATE_LO",
    "STATE_HI",
]
