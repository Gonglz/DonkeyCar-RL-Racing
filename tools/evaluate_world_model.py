"""
tools/evaluate_world_model.py

notemodelnote:
  1. note RMSE(per-dim, note)
  2. k note(k = 1, 2, 4, 8, 16), note sim note
  3. Cross-domain note(note sim2real gap):
       wm_sim notedatanote  -> sim->real gap
       wm_real note sim datanote -> real->sim gap

note
--------
# notemodelnote
python3 tools/evaluate_world_model.py \\
    --model models/world_model/wm_real.pth \\
    --catalog-dirs data \\
    --mode single

# Cross-domain gap note
python3 tools/evaluate_world_model.py \\
    --model-real  models/world_model/wm_real.pth \\
    --model-sim   models/world_model/wm_sim.pth \\
    --catalog-dirs data \\
    --sim-dirs dynamics_data/sim_transitions \\
    --mode cross_domain
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

_SCRIPT_DIR = Path(__file__).parent
_REPO_DIR   = _SCRIPT_DIR.parent
sys.path.insert(0, str(_REPO_DIR))

from module.world_model import NeuralPhysicsDynamics, PHYS_DIM, STATE_LO, STATE_HI
from module.world_model_dataset import (
    CatalogTransitionDatasetV2,
    SimTransitionDataset,
    chronological_split,
)

DIM_NAMES   = ["v_long", "yaw_rate", "accel_x"]
DIM_TARGETS = [0.05, 0.05, 0.10]   # RMSE goalnote


# ─── note ────────────────────────────────────────────────────

@torch.no_grad()
def eval_single_step(
    model: NeuralPhysicsDynamics,
    loader: DataLoader,
    device: str,
) -> dict:
    model.eval()
    sq_errs = torch.zeros(PHYS_DIM)
    n = 0
    for batch in loader:
        x, delta = batch[0].to(device), batch[1].to(device)
        pred_delta, _ = model(x, x[:,:PHYS_DIM])
        sq_errs += ((pred_delta.cpu() - delta.cpu()) ** 2).sum(dim=0)
        n += x.shape[0]
    n = max(n, 1)
    rmse = {name: math.sqrt(float(sq_errs[i] / n))
            for i, name in enumerate(DIM_NAMES)}
    return rmse


# ─── k note ──────────────────────────────────────────────

@torch.no_grad()
def eval_kstep_rollout(
    model: NeuralPhysicsDynamics,
    dataset,
    device: str,
    k_list: List[int] = (1, 2, 4, 8, 16),
    n_seeds: int = 200,
) -> dict:
    """
    note n_seeds note, notefirstnote max(k_list) note.
    note k note per-dim RMSE note.

    note: k note, note"note":
    note phys_t note, action note
    (note"open-loop"note, action note ground truth).
    """
    model.eval()
    max_k  = max(k_list)
    n      = len(dataset)
    seeds  = np.random.choice(max(n - max_k - 1, 1), size=min(n_seeds, n - max_k - 1), replace=False)

    # note k, note
    sq_errs = {k: torch.zeros(PHYS_DIM) for k in k_list}
    counts  = {k: 0 for k in k_list}

    for seed in seeds:
        # note
        x0, _ = dataset[int(seed)]
        phys_cur = x0[:PHYS_DIM].clone().to(device)

        for step in range(1, max_k + 1):
            idx = int(seed) + step
            if idx >= n:
                break
            x_step, delta_step = dataset[idx]
            x_step = x_step.to(device)

            # note(open-loop: note action note)
            x_pred = x_step.clone()
            x_pred[:PHYS_DIM] = phys_cur
            pred_delta, phys_next = model(x_pred.unsqueeze(0), phys_cur.unsqueeze(0))
            phys_cur = phys_next.squeeze(0).detach()

            if step in k_list:
                # goal: note
                true_phys = x_step[:PHYS_DIM] + delta_step.to(device)
                true_phys = torch.clamp(
                    true_phys,
                    STATE_LO.to(device),
                    STATE_HI.to(device),
                )
                sq_errs[step] += ((phys_cur.cpu() - true_phys.cpu()) ** 2)
                counts[step]  += 1

    result = {}
    for k in k_list:
        c = max(counts[k], 1)
        rmse_per_dim = {name: math.sqrt(float(sq_errs[k][i] / c))
                        for i, name in enumerate(DIM_NAMES)}
        result[k] = rmse_per_dim
    return result


# ─── noteresult ────────────────────────────────────────────────────

def print_rmse_table(rmse: dict, title: str, targets: Optional[List[float]] = None):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")
    for i, (name, val) in enumerate(rmse.items()):
        tgt = targets[i] if targets else None
        ok  = ""
        if tgt is not None:
            ok = "  PASS" if val < tgt else f"  ✗ (target < {tgt:.2f})"
        print(f"  {name:12s}: RMSE = {val:.5f}{ok}")


def print_kstep_table(kstep: dict):
    print(f"\n{'─'*60}")
    print(f"  k-step note RMSE(open-loop, ground-truth action)")
    print(f"{'─'*60}")
    print(f"  {'k':>3}  {'v_long':>9}  {'yaw_rate':>9}  {'accel_x':>9}")
    k1_rmse = {n: kstep[1][n] for n in DIM_NAMES} if 1 in kstep else None
    for k, rmse in sorted(kstep.items()):
        ratio_str = ""
        if k1_rmse is not None and k > 1:
            ratios = [f"{rmse[n] / max(k1_rmse[n], 1e-9):.1f}x" for n in DIM_NAMES]
            ratio_str = "  (" + " / ".join(ratios) + " vs k=1)"
        print(
            f"  {k:>3}  {rmse['v_long']:>9.5f}  "
            f"{rmse['yaw_rate']:>9.5f}  {rmse['accel_x']:>9.5f}{ratio_str}"
        )


# ─── notefunction ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate world model")
    parser.add_argument("--mode", choices=["single", "cross_domain"], default="single")
    parser.add_argument("--model",      default="", help="notemodelpath(--mode single)")
    parser.add_argument("--model-real", default="", help="wm_real.pth(--mode cross_domain)")
    parser.add_argument("--model-sim",  default="", help="wm_sim.pth(--mode cross_domain)")
    parser.add_argument("--catalog-dirs", nargs="+", default=[],
                        help="note catalog directorynote")
    parser.add_argument("--sim-dirs",   nargs="+", default=[],
                        help="Sim CSV directorynote")
    parser.add_argument("--input-dim",  type=int, choices=[5, 8], default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--kstep-seeds", type=int, default=200,
                        help="k note")
    parser.add_argument("--output-json", default="",
                        help="noteresultsavepath(note)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─ notedataset ─
    real_ds = sim_ds = None
    if args.catalog_dirs:
        real_ds = CatalogTransitionDatasetV2(args.catalog_dirs, input_dim=args.input_dim, augment_noise=0.0)
        real_ds.eval()
        _, _, real_test = chronological_split(real_ds, 0.8, 0.1)
        print(f"Real test samples: {len(real_test):,}")

    if args.sim_dirs:
        sim_ds = SimTransitionDataset(args.sim_dirs, input_dim=args.input_dim, augment_noise=0.0)
        sim_ds.eval()
        _, _, sim_test = chronological_split(sim_ds, 0.8, 0.1)
        print(f"Sim  test samples: {len(sim_test):,}")

    def make_loader(subset):
        return DataLoader(subset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_results = {}

    if args.mode == "single":
        if not args.model:
            parser.error("--model required for --mode single")
        model = NeuralPhysicsDynamics.load_checkpoint(args.model, device=device)
        print(f"\nModel: {args.model}  (input_dim={model.input_dim})")

        if real_ds is not None:
            rmse = eval_single_step(model, make_loader(real_test), device)
            print_rmse_table(rmse, "note RMSE - note", DIM_TARGETS)
            all_results["real_single_step"] = rmse

            kstep = eval_kstep_rollout(model, real_test.dataset,
                                       device, n_seeds=args.kstep_seeds)
            print_kstep_table(kstep)
            all_results["real_kstep"] = {str(k): v for k, v in kstep.items()}

        if sim_ds is not None:
            rmse = eval_single_step(model, make_loader(sim_test), device)
            print_rmse_table(rmse, "note RMSE - Sim note", DIM_TARGETS)
            all_results["sim_single_step"] = rmse

            kstep = eval_kstep_rollout(model, sim_test.dataset,
                                       device, n_seeds=args.kstep_seeds)
            print_kstep_table(kstep)
            all_results["sim_kstep"] = {str(k): v for k, v in kstep.items()}

    elif args.mode == "cross_domain":
        if not args.model_real or not args.model_sim:
            parser.error("--model-real and --model-sim required for cross_domain mode")
        if real_ds is None or sim_ds is None:
            parser.error("--catalog-dirs and --sim-dirs both required for cross_domain mode")

        wm_real = NeuralPhysicsDynamics.load_checkpoint(args.model_real, device=device)
        wm_sim  = NeuralPhysicsDynamics.load_checkpoint(args.model_sim,  device=device)

        print(f"\nwm_real: {args.model_real}")
        print(f"wm_sim:  {args.model_sim}")

        # wm_real note(note)
        rmse_rr = eval_single_step(wm_real, make_loader(real_test), device)
        print_rmse_table(rmse_rr, "wm_real -> notedata(note, note)", DIM_TARGETS)

        # wm_sim note(sim->real gap)
        rmse_sr = eval_single_step(wm_sim, make_loader(real_test), device)
        print_rmse_table(rmse_sr, "wm_sim  -> notedata(sim->real gap)", DIM_TARGETS)

        # wm_sim note sim note(note)
        rmse_ss = eval_single_step(wm_sim, make_loader(sim_test), device)
        print_rmse_table(rmse_ss, "wm_sim  -> Sim data(note, note)", DIM_TARGETS)

        # wm_real note sim note(real->sim gap)
        rmse_rs = eval_single_step(wm_real, make_loader(sim_test), device)
        print_rmse_table(rmse_rs, "wm_real -> Sim data(real->sim gap)", DIM_TARGETS)

        # Gap note
        print(f"\n{'─'*60}")
        print("  Sim2Real Gap note(note 1.0 note)")
        print(f"{'─'*60}")
        print(f"  {'note':12s}  {'sim->real/baseline':>18}  {'real->sim/baseline':>18}")
        for name in DIM_NAMES:
            ratio_sr = rmse_sr[name] / max(rmse_rr[name], 1e-9)
            ratio_rs = rmse_rs[name] / max(rmse_ss[name], 1e-9)
            ok_sr    = "PASS" if ratio_sr < 1.5 else "✗"
            ok_rs    = "PASS" if ratio_rs < 1.5 else "✗"
            print(
                f"  {name:12s}  "
                f"{ratio_sr:>18.3f} {ok_sr}  "
                f"{ratio_rs:>18.3f} {ok_rs}"
            )
        print()
        print("  note < 1.5 -> gap note, notetraining wm_mixed")
        print("  note >= 1.5 -> gap note, note dual-head note domain embedding")

        all_results = {
            "wm_real_on_real": rmse_rr,
            "wm_sim_on_real":  rmse_sr,
            "wm_sim_on_sim":   rmse_ss,
            "wm_real_on_sim":  rmse_rs,
        }

    # ─ save JSON ─
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved -> {args.output_json}")


if __name__ == "__main__":
    main()
