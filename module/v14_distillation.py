#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 policy distillation manager."""

import copy
from typing import Dict

import torch

class PolicyDistillationManager:
    """
    策略蒸馏防遗忘: 在阶段切换时快照策略网络，后续阶段通过KL散度惩罚防止遗忘。

    实现方式: reward-based KL penalty (不修改SB3内部)
    - 快照: 保存策略网络的state_dict
    - KL计算: 比较当前策略和快照策略对相同observation的action分布
    - 惩罚: -kl_coef * KL(current || snapshot) 加入step reward
    """

    def __init__(self, kl_coef_initial=0.5, kl_decay=0.995, kl_min=0.05):
        self.snapshots: Dict[int, dict] = {}  # {stage_id: state_dict}
        self.kl_coef = float(kl_coef_initial)
        self.kl_coef_initial = float(kl_coef_initial)
        self.kl_decay = float(kl_decay)
        self.kl_min = float(kl_min)
        self._snapshot_policy_ref = None  # 用于推理的快照policy副本
        self._snapshot_device = None

    def snapshot_policy(self, model, stage_id):
        """在阶段切换时深拷贝策略网络参数。"""
        try:
            state_dict = copy.deepcopy(model.policy.state_dict())
            self.snapshots[int(stage_id)] = state_dict
            self.kl_coef = self.kl_coef_initial  # 重置KL系数
            # 构建一个用于推理的快照policy
            self._rebuild_snapshot_policy(model)
            print(f"   📸 策略快照已保存 (stage={stage_id}, params={len(state_dict)})")
        except Exception as e:
            print(f"   ⚠️ 策略快照失败: {e}")

    def _rebuild_snapshot_policy(self, model):
        """从最新快照重建推理用的policy副本。"""
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
            print(f"   ⚠️ 快照policy重建失败: {e}")
            self._snapshot_policy_ref = None

    def decay_kl_coef(self):
        """每训练chunk衰减KL系数。"""
        self.kl_coef = max(self.kl_min, self.kl_coef * self.kl_decay)

    def compute_kl_penalty(self, model, obs_tensor):
        """
        计算当前策略与快照策略的KL散度。

        Args:
            model: 当前PPO模型
            obs_tensor: 观测tensor (B, C, H, W)

        Returns:
            kl_penalty: float, 应加到reward上的惩罚 (-kl_coef * kl_div)
        """
        if self._snapshot_policy_ref is None or self.kl_coef < 1e-6:
            return 0.0

        try:
            with torch.no_grad():
                # 当前策略的动作分布
                current_dist = model.policy.get_distribution(obs_tensor)
                current_mean = current_dist.distribution.mean
                current_std = current_dist.distribution.stddev

                # 快照策略的动作分布
                snapshot_dist = self._snapshot_policy_ref.get_distribution(obs_tensor)
                snapshot_mean = snapshot_dist.distribution.mean
                snapshot_std = snapshot_dist.distribution.stddev

                # 高斯分布KL散度: KL(p || q) = log(σ_q/σ_p) + (σ_p² + (μ_p - μ_q)²) / (2σ_q²) - 0.5
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
