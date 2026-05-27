#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 policy distillation manager."""

import copy
from typing import Dict

import torch

class PolicyDistillationManager:
    """
    note: notestagenote, notestagenoteKLnote.

    implementnote: reward-based KL penalty (noteSB3note)
    - note: savenotestate_dict
    - KLcompute: notecurrentnoteobservationnoteactiondistribution
    - note: -kl_coef * KL(current || snapshot) notestep reward
    """

    def __init__(self, kl_coef_initial=0.5, kl_decay=0.995, kl_min=0.05):
        self.snapshots: Dict[int, dict] = {}  # {stage_id: state_dict}
        self.kl_coef = float(kl_coef_initial)
        self.kl_coef_initial = float(kl_coef_initial)
        self.kl_decay = float(kl_decay)
        self.kl_min = float(kl_min)
        self._snapshot_policy_ref = None  # notepolicynote
        self._snapshot_device = None

    def snapshot_policy(self, model, stage_id):
        """notestagenote."""
        try:
            state_dict = copy.deepcopy(model.policy.state_dict())
            self.snapshots[int(stage_id)] = state_dict
            self.kl_coef = self.kl_coef_initial  # noteKLnote
            # notepolicy
            self._rebuild_snapshot_policy(model)
            print(f"   📸 notesave (stage={stage_id}, params={len(state_dict)})")
        except Exception as e:
            print(f"   ⚠️ notefailed: {e}")

    def _rebuild_snapshot_policy(self, model):
        """notepolicynote."""
        if not self.snapshots:
            self._snapshot_policy_ref = None
            return
        latest_stage = max(self.snapshots.keys())
        try:
            snapshot_policy = copy.deepcopy(model.policy)
            snapshot_policy.load_state_dict(self.snapshots[latest_stage])
            snapshot_policy.eval()
            for p in snapshot_policy.parameters():
                p.requires_grad = False
            self._snapshot_policy_ref = snapshot_policy
            self._snapshot_device = next(model.policy.parameters()).device
        except Exception as e:
            print(f"   ⚠️ notepolicynotefailed: {e}")
            self._snapshot_policy_ref = None

    def decay_kl_coef(self):
        """notetrainingchunknoteKLnote."""
        self.kl_coef = max(self.kl_min, self.kl_coef * self.kl_decay)

    def compute_kl_penalty(self, model, obs_tensor):
        """
        computecurrentnoteKLnote.

        Args:
            model: currentPPOmodel
            obs_tensor: notetensor (B, C, H, W)

        Returns:
            kl_penalty: float, noterewardnote (-kl_coef * kl_div)
        """
        if self._snapshot_policy_ref is None or self.kl_coef < 1e-6:
            return 0.0

        try:
            with torch.no_grad():
                # currentnotedistribution
                current_dist = model.policy.get_distribution(obs_tensor)
                current_mean = current_dist.distribution.mean
                current_std = current_dist.distribution.stddev

                # notedistribution
                snapshot_dist = self._snapshot_policy_ref.get_distribution(obs_tensor)
                snapshot_mean = snapshot_dist.distribution.mean
                snapshot_std = snapshot_dist.distribution.stddev

                # notedistributionKLnote: KL(p || q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
                var_current = current_std ** 2
                var_snapshot = snapshot_std ** 2
                kl = (torch.log(snapshot_std / current_std)
                      + (var_current + (current_mean - snapshot_mean) ** 2) / (2 * var_snapshot)
                      - 0.5)
                kl_mean = float(kl.mean().item())
                return float(-self.kl_coef * max(0.0, kl_mean))
        except Exception:
            return 0.0

    @property
    def has_snapshot(self):
        return bool(self.snapshots) and self._snapshot_policy_ref is not None


__all__ = ["PolicyDistillationManager"]
