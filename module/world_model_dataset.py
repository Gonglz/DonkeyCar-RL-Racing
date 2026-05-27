"""
module/world_model_dataset.py

notemodeldatanote - V1

noteclassdatanote
--------------
1. CatalogTransitionDataset  -  note DonkeyCar catalog(JSONL)
2. SimTransitionDataset      -  Sim note CSV(note collect_sim_transitions.py generate)
3. CombinedTransitionDataset -  note, note domain_weight(notedatanote)

datanote(note)
--------------------
  x: float32 tensor, shape (input_dim,)   - 8D note or 5D baseline
  delta: float32 tensor, shape (3,)           - goalnote [Δv, Δω, Δa]

inputnote(8D)
-------------------
  [v_long_t, yaw_rate_t, accel_x_t,      # currentnote(note)
   steer_exec_t, throttle_t,              # currentnoterowsnote
   prev_steer_exec, prev_throttle,        # noterowsnote(note)
   dt_norm]                              # dt_ms / 50.0

note(note obv.py:_build_state_v13 note)
---------------------------------------------
  V_MAX    = 2.2   m/s
  GYRO_MAX = 4.0   rad/s   (yaw_rate = clip(-gyro_z / GYRO_MAX, -2, 2))
  ACCEL_MAX= 9.8   m/s²
  DT_REF   = 50.0  ms
  DT_MAX   = 200   ms(note session note, note)

note: note catalog note -rp2040/gyro_z note yaw_rate(note!)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# ─── note ──────────────────────────────────────────────────
V_MAX     = 2.2
GYRO_MAX  = 4.0
ACCEL_MAX = 9.8
DT_REF    = 50.0   # ms, note
DT_MAX_MS = 200    # ms, note(session note / note)


def _norm_v(v: float) -> float:
    return float(np.clip(v / V_MAX, 0.0, 2.0))


def _norm_yaw(gyro_z_raw: float) -> float:
    """note yaw_rate = -gyro_z(note, note _build_state_v13 note)."""
    return float(np.clip(-gyro_z_raw / GYRO_MAX, -2.0, 2.0))


def _norm_accel(accel_x: float) -> float:
    return float(np.clip(accel_x / ACCEL_MAX, -2.0, 2.0))


def _safe_float(val, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


# ─── Catalog dataset(note)────────────────────────────────────

class CatalogTransitionDataset(Dataset):
    """
    note DonkeyCar JSONL catalog directorynote (x_8d, delta_3d) trainingnote.

    directorynote catalog_*.catalog file(JSONL, noterowsnote).
    notedirectorynote, note.

    datanotedescription
    ------------
    catalog note(user mode), user/angle note, note ActionAdapter note ActionSafetyWrapper,
    note steer_exec, note.

    Parameters
    ----------
    catalog_dirs: list of str
        note catalog_*.catalog filenotedirectorypathnote.
    input_dim: int
        5(baseline)note 8(note).
    augment_noise: float
        trainingnoteinputnote.0 = note.
    """

    def __init__(
        self,
        catalog_dirs: List[str],
        input_dim: int = 8,
        augment_noise: float = 0.005,
    ):
        assert input_dim in (5, 8), f"input_dim must be 5 or 8, got {input_dim}"
        self.input_dim      = input_dim
        self.augment_noise  = augment_noise
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for d in catalog_dirs:
            self._load_dir(d)

    def _load_dir(self, directory: str) -> None:
        d = Path(directory)
        catalog_files = sorted(
            [p for p in d.glob("catalog_*.catalog") if "manifest" not in p.name],
            key=lambda p: int(p.stem.split("_")[1]),
        )
        if not catalog_files:
            return

        records = []
        for cf in catalog_files:
            with open(cf, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

        # note
        records.sort(key=lambda r: r.get("_timestamp_ms", 0))
        self._build_pairs(records)

    def _build_pairs(self, records: list) -> None:
        prev_rec     = None
        prev_session = None

        for rec in records:
            session = rec.get("_session_id", "")
            ts_ms   = _safe_float(rec.get("_timestamp_ms", 0))

            # session note: note
            if session!= prev_session:
                prev_rec     = rec
                prev_session = session
                continue

            # note
            dt_ms = ts_ms - _safe_float(prev_rec.get("_timestamp_ms", 0))
            if dt_ms <= 0 or dt_ms > DT_MAX_MS:
                prev_rec = rec
                continue

            x, delta = self._make_sample(prev_rec, rec, dt_ms)
            if x is not None:
                self.samples.append((x, delta))

            prev_rec = rec

    def _make_sample(
        self, r_t: dict, r_t1: dict, dt_ms: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """note (x, delta)."""
        # ─ currentnote ─
        v_t   = _norm_v(_safe_float(r_t.get("rp2040/speed_odom")))
        yaw_t = _norm_yaw(_safe_float(r_t.get("rp2040/gyro_z")))
        a_t   = _norm_accel(_safe_float(r_t.get("rp2040/accel_x")))

        # ─ currentnoterowsnote ─
        steer_t   = float(np.clip(_safe_float(r_t.get("user/angle")),   -1.0, 1.0))
        thr_t     = float(np.clip(_safe_float(r_t.get("user/throttle")),  0.0, 0.3))

        # ─ note ─
        v_t1   = _norm_v(_safe_float(r_t1.get("rp2040/speed_odom")))
        yaw_t1 = _norm_yaw(_safe_float(r_t1.get("rp2040/gyro_z")))
        a_t1   = _norm_accel(_safe_float(r_t1.get("rp2040/accel_x")))

        delta = np.array([v_t1 - v_t, yaw_t1 - yaw_t, a_t1 - a_t], dtype=np.float32)

        if self.input_dim == 5:
            x = np.array([v_t, yaw_t, a_t, steer_t, thr_t], dtype=np.float32)
        else:
            # firstnote(note r_t notefirstnote, note r_t note
            # note prev; note prev note _build_pairs note, note 3 note)
            # note: prev_steer ~ steer_t(note prev = current, note)
            # note, note _build_pairs_3frame note
            dt_norm = dt_ms / DT_REF
            x = np.array(
                [v_t, yaw_t, a_t, steer_t, thr_t, steer_t, thr_t, dt_norm],
                dtype=np.float32,
            )

        return x, delta

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, delta = self.samples[idx]
        x = x.copy()
        if self.augment_noise > 0.0 and self.training_mode:
            x += np.random.normal(0, self.augment_noise, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(delta)

    # training_mode note(Dataset note train/val)
    @property
    def training_mode(self) -> bool:
        return getattr(self, "_training_mode", True)

    def train(self):
        self._training_mode = True
        return self

    def eval(self):
        self._training_mode = False
        return self


class CatalogTransitionDatasetV2(CatalogTransitionDataset):
    """
    note 8D note: note 3 note prev_steer_exec / prev_throttle.
    note V1 note, note _build_pairs.
    """

    def _build_pairs(self, records: list) -> None:
        prev_prev_rec = None
        prev_rec      = None
        prev_session  = None

        for rec in records:
            session = rec.get("_session_id", "")
            ts_ms   = _safe_float(rec.get("_timestamp_ms", 0))

            if session!= prev_session:
                prev_prev_rec = None
                prev_rec      = rec
                prev_session  = session
                continue

            # note
            dt_ms = ts_ms - _safe_float(prev_rec.get("_timestamp_ms", 0))
            if dt_ms <= 0 or dt_ms > DT_MAX_MS:
                prev_prev_rec = None
                prev_rec      = rec
                continue

            if prev_prev_rec is not None and self.input_dim == 8:
                x, delta = self._make_sample_v2(prev_prev_rec, prev_rec, rec, dt_ms)
                if x is not None:
                    self.samples.append((x, delta))
            elif self.input_dim == 5:
                x, delta = self._make_sample(prev_rec, rec, dt_ms)
                if x is not None:
                    self.samples.append((x, delta))

            prev_prev_rec = prev_rec
            prev_rec      = rec

    def _make_sample_v2(
        self, r_tm1: dict, r_t: dict, r_t1: dict, dt_ms: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """note 3 note: r_tm1=t-1, r_t=t, r_t1=t+1."""
        v_t   = _norm_v(_safe_float(r_t.get("rp2040/speed_odom")))
        yaw_t = _norm_yaw(_safe_float(r_t.get("rp2040/gyro_z")))
        a_t   = _norm_accel(_safe_float(r_t.get("rp2040/accel_x")))

        steer_t   = float(np.clip(_safe_float(r_t.get("user/angle")),    -1.0, 1.0))
        thr_t     = float(np.clip(_safe_float(r_t.get("user/throttle")),  0.0, 0.3))
        prev_steer = float(np.clip(_safe_float(r_tm1.get("user/angle")),  -1.0, 1.0))
        prev_thr   = float(np.clip(_safe_float(r_tm1.get("user/throttle")), 0.0, 0.3))

        v_t1   = _norm_v(_safe_float(r_t1.get("rp2040/speed_odom")))
        yaw_t1 = _norm_yaw(_safe_float(r_t1.get("rp2040/gyro_z")))
        a_t1   = _norm_accel(_safe_float(r_t1.get("rp2040/accel_x")))

        delta  = np.array([v_t1 - v_t, yaw_t1 - yaw_t, a_t1 - a_t], dtype=np.float32)
        dt_norm = dt_ms / DT_REF
        x = np.array(
            [v_t, yaw_t, a_t, steer_t, thr_t, prev_steer, prev_thr, dt_norm],
            dtype=np.float32,
        )
        return x, delta


# ─── Sim CSV dataset ──────────────────────────────────────────────

class SimTransitionDataset(Dataset):
    """
    note collect_sim_transitions.py generatenote CSV filenotedata.

    CSV note(note, noteread):
      v_t, yaw_t, accel_t, steer_exec_t, throttle_t,
      prev_steer_exec, prev_throttle, dt_ms,
      v_t1, yaw_t1, accel_t1,
      policy_type, episode_id      <- note, notetraining

    Parameters
    ----------
    csv_dirs: list of str
        note *.csv filenotedirectorynote.
    input_dim: int
        5 note 8.
    augment_noise: float
        trainingnoteinputnote.
    """

    def __init__(
        self,
        csv_dirs: List[str],
        input_dim: int = 8,
        augment_noise: float = 0.005,
    ):
        assert input_dim in (5, 8)
        self.input_dim     = input_dim
        self.augment_noise = augment_noise
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []

        for d in csv_dirs:
            self._load_dir(d)

    def _load_dir(self, directory: str) -> None:
        for csv_path in sorted(Path(directory).glob("*.csv")):
            self._load_csv(csv_path)

    def _load_csv(self, csv_path: Path) -> None:
        with open(csv_path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    v_t    = float(row["v_t"])
                    yaw_t  = float(row["yaw_t"])
                    a_t    = float(row["accel_t"])
                    s_t    = float(row["steer_exec_t"])
                    thr_t  = float(row["throttle_t"])
                    v_t1   = float(row["v_t1"])
                    yaw_t1 = float(row["yaw_t1"])
                    a_t1   = float(row["accel_t1"])

                    delta = np.array(
                        [v_t1 - v_t, yaw_t1 - yaw_t, a_t1 - a_t],
                        dtype=np.float32,
                    )

                    if self.input_dim == 5:
                        x = np.array([v_t, yaw_t, a_t, s_t, thr_t], dtype=np.float32)
                    else:
                        ps_t   = float(row.get("prev_steer_exec", s_t))
                        pthr_t = float(row.get("prev_throttle", thr_t))
                        dt_ms  = float(row.get("dt_ms", 50.0))
                        dt_norm = dt_ms / DT_REF
                        x = np.array(
                            [v_t, yaw_t, a_t, s_t, thr_t, ps_t, pthr_t, dt_norm],
                            dtype=np.float32,
                        )

                    self.samples.append((x, delta))
                except (KeyError, ValueError):
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, delta = self.samples[idx]
        x = x.copy()
        if self.augment_noise > 0.0 and getattr(self, "_training_mode", True):
            x += np.random.normal(0, self.augment_noise, x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(delta)

    def train(self):
        self._training_mode = True
        return self

    def eval(self):
        self._training_mode = False
        return self


# ─── notedataset ──────────────────────────────────────────────────

class CombinedTransitionDataset(Dataset):
    """
    note catalog(note)note sim dataset.

    domain_weight controlnotedatanote(note is_real).
    note train_world_model.py notefunctionnoteimplement.

    Parameters
    ----------
    real_dataset: CatalogTransitionDataset or None
    sim_dataset: SimTransitionDataset or None
    """

    def __init__(
        self,
        real_dataset: Optional[Dataset] = None,
        sim_dataset:  Optional[Dataset] = None,
    ):
        assert real_dataset is not None or sim_dataset is not None
        self.real_ds   = real_dataset
        self.sim_ds    = sim_dataset
        self.real_len  = len(real_dataset) if real_dataset is not None else 0
        self.sim_len   = len(sim_dataset)  if sim_dataset  is not None else 0

    def __len__(self) -> int:
        return self.real_len + self.sim_len

    def __getitem__(self, idx: int):
        """Returns (x, delta, is_real) where is_real=1 for real-car data."""
        if idx < self.real_len:
            x, delta = self.real_ds[idx]
            is_real  = torch.tensor(1, dtype=torch.float32)
        else:
            x, delta = self.sim_ds[idx - self.real_len]
            is_real  = torch.tensor(0, dtype=torch.float32)
        return x, delta, is_real

    def train(self):
        if self.real_ds is not None:
            self.real_ds.train()
        if self.sim_ds is not None:
            self.sim_ds.train()
        return self

    def eval(self):
        if self.real_ds is not None:
            self.real_ds.eval()
        if self.sim_ds is not None:
            self.sim_ds.eval()
        return self


# ─── note: note ──────────────────────────────────────────────

def chronological_split(
    dataset: Dataset,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
):
    """
    note(note)notedataset, note.

    Returns
    -------
    train_set, val_set, test_set: Subset
    """
    from torch.utils.data import Subset

    n       = len(dataset)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    n_test  = n - n_train - n_val

    idx = list(range(n))
    return (
        Subset(dataset, idx[:n_train]),
        Subset(dataset, idx[n_train: n_train + n_val]),
        Subset(dataset, idx[n_train + n_val:]),
    )


__all__ = [
    "CatalogTransitionDataset",
    "CatalogTransitionDatasetV2",
    "SimTransitionDataset",
    "CombinedTransitionDataset",
    "chronological_split",
    "V_MAX", "GYRO_MAX", "ACCEL_MAX", "DT_REF", "DT_MAX_MS",
]
