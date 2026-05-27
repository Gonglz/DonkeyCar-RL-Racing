"""
tools/train_world_model.py

notemodel - notetrainingnotetrainingnote(--mode):
  real   - note catalog data, training wm_real
  sim    - note sim CSV data, training wm_sim
  mixed  - note, notedata 3x note, training wm_mixed

trainingnote: notetraining(real / sim), note cross-domain note
sim2real gap, note gap note.

note
--------
# notedatanote wm_real(note 5D sanity check, note 8D note)
python3 tools/train_world_model.py \\
    --mode real \\
    --catalog-dirs data \\
    --input-dim 8 \\
    --output-dir models/world_model \\
    --epochs 150

# sim datanote wm_sim
python3 tools/train_world_model.py \\
    --mode sim \\
    --sim-dirs dynamics_data/sim_transitions \\
    --input-dim 8 \\
    --output-dir models/world_model \\
    --epochs 150

# notetraining wm_mixed(real data 3x note)
python3 tools/train_world_model.py \\
    --mode mixed \\
    --catalog-dirs data \\
    --sim-dirs dynamics_data/sim_transitions \\
    --input-dim 8 \\
    --real-weight 3.0 \\
    --output-dir models/world_model \\
    --epochs 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# ─── pathnote ────────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_DIR))

from module.world_model import NeuralPhysicsDynamics, PHYS_DIM
from module.world_model_dataset import (
    CatalogTransitionDatasetV2,
    SimTransitionDataset,
    CombinedTransitionDataset,
    chronological_split,
)


# ─── note(v_long note, accel_x note)────────────────────
DIM_WEIGHTS = torch.tensor([3.0, 2.0, 0.5], dtype=torch.float32)
DIM_NAMES   = ["v_long", "yaw_rate", "accel_x"]


def weighted_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    domain_mask: Optional[torch.Tensor] = None,
    real_weight: float = 1.0,
) -> torch.Tensor:
    """
    Per-dim note MSE.
    domain_mask: (B,) float, 1=notedata, 0=sim data, notedatanote.
    """
    err = (pred - target) ** 2                         # (B, 3)
    loss = (err * weights.to(pred.device)).sum(dim=-1)  # (B,)

    if domain_mask is not None and real_weight!= 1.0:
        sample_weights = 1.0 + (real_weight - 1.0) * domain_mask.to(pred.device)
        loss = loss * sample_weights

    return loss.mean()


# ─── note epoch training ───────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, real_weight=1.0, is_combined=False):
    model.train()
    total_loss = 0.0
    for batch in loader:
        if is_combined:
            x, delta, is_real = batch
            is_real = is_real.to(device)
        else:
            x, delta = batch
            is_real  = None

        x, delta = x.to(device), delta.to(device)
        pred_delta, _ = model(x, x[:,:PHYS_DIM])

        loss = weighted_mse(
            pred_delta, delta, DIM_WEIGHTS,
            domain_mask=is_real, real_weight=real_weight,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


# ─── note ───────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, is_combined=False):
    model.eval()
    total_loss  = 0.0
    per_dim_se  = torch.zeros(PHYS_DIM)
    n_batches   = 0

    for batch in loader:
        if is_combined:
            x, delta, _ = batch
        else:
            x, delta = batch

        x, delta = x.to(device), delta.to(device)
        pred_delta, _ = model(x, x[:,:PHYS_DIM])

        total_loss += weighted_mse(pred_delta, delta, DIM_WEIGHTS).item()
        per_dim_se += ((pred_delta.cpu() - delta.cpu()) ** 2).mean(dim=0)
        n_batches  += 1

    n_batches = max(n_batches, 1)
    return total_loss / n_batches, per_dim_se / n_batches


# ─── notefunction ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train world model")
    parser.add_argument("--mode", choices=["real", "sim", "mixed"], required=True)
    parser.add_argument("--catalog-dirs", nargs="+", default=[],
                        help="note catalog directory(--mode real/mixed note)")
    parser.add_argument("--sim-dirs", nargs="+", default=[],
                        help="Sim CSV directory(--mode sim/mixed note)")
    parser.add_argument("--input-dim", type=int, choices=[5, 8], default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--real-weight", type=float, default=3.0,
                        help="notetrainingnotedatanote")
    parser.add_argument("--output-dir", type=str, default="models/world_model")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}, input_dim: {args.input_dim}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ─ notedataset ─
    real_ds = sim_ds = None

    if args.mode in ("real", "mixed"):
        if not args.catalog_dirs:
            parser.error("--catalog-dirs required for mode real/mixed")
        real_ds = CatalogTransitionDatasetV2(
            args.catalog_dirs, input_dim=args.input_dim, augment_noise=0.005
        )
        print(f"Real data: {len(real_ds):,} samples from {args.catalog_dirs}")

    if args.mode in ("sim", "mixed"):
        if not args.sim_dirs:
            parser.error("--sim-dirs required for mode sim/mixed")
        sim_ds = SimTransitionDataset(
            args.sim_dirs, input_dim=args.input_dim, augment_noise=0.005
        )
        print(f"Sim data: {len(sim_ds):,} samples from {args.sim_dirs}")

    # ─ notedataset(note)─
    is_combined = (args.mode == "mixed")

    if is_combined:
        base_ds = CombinedTransitionDataset(real_ds, sim_ds)
    elif args.mode == "real":
        base_ds = real_ds
    else:
        base_ds = sim_ds

    total = len(base_ds)
    print(f"Total samples: {total:,}")
    if total < 100:
        print("WARNING: Very few samples. Check data paths.")

    train_set, val_set, test_set = chronological_split(base_ds, 0.8, 0.1)
    print(f"Split: train={len(train_set):,} val={len(val_set):,} test={len(test_set):,}")

    # ─ DataLoaders ─
    def make_loader(subset, shuffle):
        return DataLoader(
            subset, batch_size=args.batch_size,
            shuffle=shuffle, num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )

    train_loader = make_loader(train_set, shuffle=True)
    val_loader   = make_loader(val_set,   shuffle=False)
    test_loader  = make_loader(test_set,  shuffle=False)

    # ─ model ─
    model = NeuralPhysicsDynamics(
        input_dim=args.input_dim, hidden_dim=args.hidden_dim, dropout=0.05
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ─ note + note ─
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    total_steps  = args.epochs * max(len(train_loader), 1)
    warmup_steps = min(5 * max(len(train_loader), 1), total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        total_steps=total_steps,
        pct_start=warmup_steps / max(total_steps, 1),
        anneal_strategy="cos",
    )

    # ─ trainingnote ─
    best_val_loss = float("inf")
    history       = []
    ckpt_name     = f"wm_{args.mode}.pth"
    ckpt_path     = os.path.join(args.output_dir, ckpt_name)
    t0            = time.time()

    for epoch in range(1, args.epochs + 1):
        if is_combined:
            base_ds.train()
        elif hasattr(base_ds, "train"):
            base_ds.train()

        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            real_weight=args.real_weight, is_combined=is_combined,
        )
        scheduler.step()

        if is_combined:
            base_ds.eval()
        elif hasattr(base_ds, "eval"):
            base_ds.eval()

        val_loss, per_dim = evaluate(model, val_loader, device, is_combined)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        row.update({f"val_mse_{n}": float(v) for n, v in zip(DIM_NAMES, per_dim)})
        history.append(row)

        if epoch % 10 == 0 or epoch == 1 or epoch == args.epochs:
            elapsed = time.time() - t0
            rmse_str = "  ".join(
                f"{n}={math.sqrt(float(v)):.4f}" for n, v in zip(DIM_NAMES, per_dim)
            )
            print(
                f"Epoch {epoch:3d}/{args.epochs} "
                f"[{elapsed:5.0f}s]  "
                f"train={train_loss:.5f}  val={val_loss:.5f} | "
                f"RMSE: {rmse_str}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_checkpoint(
                ckpt_path,
                extra={
                    "epoch":        epoch,
                    "val_loss":     val_loss,
                    "mode":         args.mode,
                    "input_dim":    args.input_dim,
                    "dim_weights":  DIM_WEIGHTS.tolist(),
                    "catalog_dirs": args.catalog_dirs,
                    "sim_dirs":     args.sim_dirs,
                },
            )

    # ─ note test note ─
    best_model = NeuralPhysicsDynamics.load_checkpoint(ckpt_path, device=str(device))
    test_loss, test_per_dim = evaluate(best_model, test_loader, device, is_combined)

    print(f"\n{'='*60}")
    print(f"Test loss: {test_loss:.5f}")
    print("Per-dim test RMSE (note):")
    for name, val in zip(DIM_NAMES, test_per_dim):
        rmse = math.sqrt(float(val))
        ok   = "PASS" if rmse < (0.05 if name!= "accel_x" else 0.10) else "✗"
        print(f"  {name:12s}: RMSE = {rmse:.5f}  {ok}")

    print(f"\nBest checkpoint saved -> {ckpt_path}")

    # ─ savetrainingnote ─
    hist_path = os.path.join(args.output_dir, f"history_{args.mode}.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history -> {hist_path}")


if __name__ == "__main__":
    main()
