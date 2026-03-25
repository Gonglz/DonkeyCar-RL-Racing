#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 wrapper reward and step mixin."""

from collections import deque
import time

import numpy as np
import torch

from .utils import _safe_float


class V14RewardMixin:
    def _forward_gap_sim_from_progress_diff(self, progress_diff):
        """Convert progress-diff to forward distance in simulator units."""
        try:
            pdiff = int(_safe_float(progress_diff, 0))
        except Exception:
            pdiff = 0
        forward_gap_idx = max(0, -pdiff)
        return float(forward_gap_idx) * float(self._avg_fine_gap_sim())

    def _ahead_idx_range_from_sim(self, ahead_min_sim, ahead_max_sim):
        """Convert ahead distance range (sim units) to fine-track index range."""
        gap = float(max(1e-4, self._avg_fine_gap_sim()))
        amin = max(0.0, _safe_float(ahead_min_sim, 0.0))
        amax = max(0.0, _safe_float(ahead_max_sim, 0.0))
        if amax < amin:
            amin, amax = amax, amin
        min_idx = max(1, int(round(amin / gap)))
        max_idx = max(min_idx, int(round(amax / gap)))
        return int(min_idx), int(max_idx)

    def _forward_gap_sim_from_idx(self, learner_fi, npc_fi):
        """Forward distance in sim units between learner and NPC by fine index."""
        tc = getattr(self, "track_cache", None)
        if tc is None:
            return 0.0
        try:
            pdiff = int(tc.progress_diff(self.scene_name, int(learner_fi), int(npc_fi)))
        except Exception:
            return 0.0
        return float(self._forward_gap_sim_from_progress_diff(pdiff))

    def _longitudinal_gap_penalty(self, forward_gap_sim, scale=1.0):
        """
        Piecewise-linear proximity penalty:
        - gap >= start: no penalty
        - gap <= danger: max penalty and trigger danger flag
        - otherwise: linear ramp
        """
        gap = float(max(0.0, _safe_float(forward_gap_sim, 0.0)))
        start = float(max(1e-6, _safe_float(getattr(self, "npc_longitudinal_penalty_start_sim", 2.5), 2.5)))
        danger = float(max(0.0, _safe_float(getattr(self, "npc_longitudinal_danger_sim", 0.5), 0.5)))
        max_pen = float(max(0.0, _safe_float(getattr(self, "npc_proximity_penalty_max", 1.2), 1.2)))
        w = float(max(0.0, _safe_float(scale, 1.0)))

        if gap >= start:
            return 0.0, False

        if start <= danger + 1e-6:
            pen = max_pen * w if gap <= danger else 0.0
            return float(pen), bool(gap <= danger)

        ratio = (start - gap) / max(1e-6, (start - danger))
        ratio = float(np.clip(ratio, 0.0, 1.0))
        pen = float(max_pen * ratio * w)
        return pen, bool(gap <= danger)

    def _turn_away_bonus(self, forward_gap_sim, lateral_delta, steer_exec, scale=1.0):
        """Small dense bonus when steering away while lateral clearance is increasing."""
        gap = float(_safe_float(forward_gap_sim, 0.0))
        lat_delta = float(_safe_float(lateral_delta, 0.0))
        steer = abs(float(_safe_float(steer_exec, 0.0)))
        w = float(max(0.0, _safe_float(scale, 1.0)))
        if gap <= 0.0 or lat_delta <= 0.0 or steer < 0.05 or w <= 0.0:
            return 0.0

        lat_safe = float(max(1e-3, _safe_float(getattr(self, "npc_lateral_safe_sim", 0.8), 0.8)))
        start = float(max(1e-3, _safe_float(getattr(self, "npc_longitudinal_penalty_start_sim", 2.5), 2.5)))
        lat_ratio = float(np.clip(lat_delta / lat_safe, 0.0, 1.0))
        gap_ratio = float(np.clip(gap / start, 0.0, 1.0))
        bonus = 0.12 * lat_ratio * steer * (0.5 + 0.5 * gap_ratio) * w
        return float(max(0.0, bonus))

    def _reward_racing_line(self, reward, info, weight=1.0):
        """
        Stage 1 & 4: 赛车线奖励 + 曲率惩罚。

        奖励沿最优赛车线行驶，惩罚急转向。
        """
        rt = self._rt_add
        fi = getattr(self, 'learner_fine_idx', 0)

        if self.racing_line.loaded:
            # 赛车线偏移奖励
            racing_offset = self.racing_line.get_offset_cte(fi)
            cte_raw = _safe_float(info.get('cte', 0.0), 0.0)  # 带符号CTE
            hw_cte = self._half_width_avg_cte_at_fine(fi)
            hw_cte = max(0.1, hw_cte)

            # 距离赛车线的偏差 (归一化)
            deviation = abs(cte_raw - racing_offset) / hw_cte
            racing_reward = self.racing_line_reward_scale * weight * max(0.0, 1.0 - deviation)
            reward += racing_reward
            rt(info, "racing_line_bonus", racing_reward)

        # 曲率惩罚: 速度足够时惩罚大曲率转向
        speed = _safe_float(info.get('speed', 0.0), 0.0)
        if speed > self.curvature_penalty_speed_thresh:
            kappa = abs(getattr(self, 'kappa_ref', 0.0))
            kappa_norm = min(1.0, kappa / max(0.1, self.kappa_ref_max_for_penalty))
            curv_pen = -self.curvature_penalty_scale * weight * kappa_norm
            reward += curv_pen
            rt(info, "curvature_penalty", curv_pen)

        return reward
    def _reward_proactive_avoid(self, reward, done, info, scale=1.0):
        """
        Stage 2 & 3: 密集连续的主动避障奖励。

        修复v14初版三个问题:
        1) 用真实learner-NPC横向距离代替CTE（CTE是离中心线距离，不是离NPC距离）
        2) 距离梯度奖励全程连续（不只在proactive_zone内）—— 远距离弱信号、近距离强信号
        3) Stage2每步调用_maybe_reposition_npcs_on_path确保NPC在前方路径上
        """
        # 确保NPC在learner前方路径上
        self._maybe_reposition_npcs_on_path(info)

        feats = self._npc_distance_features(info)
        if not feats:
            return reward, done

        d_danger = float(self.dist_scale.get('danger_close_sim', 1.2))
        d_safe_min = float(self.dist_scale.get('follow_safe_min_sim', 1.6))
        d_safe_max = float(self.dist_scale.get('follow_safe_max_sim', 4.0))
        d_proactive_near = d_danger * self.proactive_zone_scale
        d_awareness_far = d_safe_max * 2.5  # 感知范围：更远就开始给微弱信号
        rt = self._rt_add
        radius_penalty_radius = max(0.0, float(self.v14_npc_radius_penalty_radius_sim))
        radius_penalty_step = max(0.0, float(self.v14_npc_radius_penalty_per_step))

        for f in feats:
            dist = float(f['dist'])
            if radius_penalty_step > 0.0 and radius_penalty_radius > 0.0 and dist <= radius_penalty_radius:
                radius_pen = float(radius_penalty_step * scale)
                reward -= radius_pen
                rt(info, "danger_radius_penalty", -radius_pen)
            progress_diff = int(f.get('progress_diff', 0))
            forward_gap_idx = max(0, -progress_diff)
            forward_gap_sim = float(forward_gap_idx) * self._avg_fine_gap_sim()
            if bool(self.v14_reward_forward_only) and forward_gap_idx <= 0:
                continue
            dist_metric = float(forward_gap_sim if bool(self.v14_reward_forward_only) else dist)
            if (not np.isfinite(dist_metric)) or dist_metric <= 0.0:
                dist_metric = float(max(1e-3, dist))
            speed = f.get('learner_speed', 0.0)
            closing = float(f.get('closing_speed', 0.0))

            # 计算真实横向距离（垂直于赛道方向）
            lateral_dist, longitudinal_dist = self._compute_learner_npc_lateral_dist(info, f)
            self._npc_lateral_dist_history.append(lateral_dist)
            npc = f.get('npc')
            npc_id = int(getattr(npc, "npc_id", -1)) if npc is not None else -1
            if npc_id >= 0:
                hist = self._npc_lateral_dist_histories.get(npc_id)
                if hist is None:
                    hist = deque(maxlen=10)
                    self._npc_lateral_dist_histories[npc_id] = hist
                hist.append(float(lateral_dist))
            else:
                hist = self._npc_lateral_dist_history

            # ═══════════════════════════════════════════════════
            # [密集连续] 距离梯度奖励: 全程有信号
            # 远距离(>d_safe_max): 微弱正奖励，鼓励保持进度
            # 中距离(d_proactive_near ~ d_safe_max): 横向距离越大奖励越高
            # 近距离(<d_proactive_near): 急需避让，横向距离直接决定奖励/惩罚
            # ═══════════════════════════════════════════════════

            if dist_metric > d_awareness_far:
                # 超出感知范围: 只给微弱存活信号（与NPC无关）
                pass

            elif dist_metric > d_safe_max:
                # 远感知区: NPC在远处，微弱连续信号
                # 鼓励维持横向距离（预防性）
                awareness_ratio = 1.0 - float(np.clip(
                    (dist_metric - d_safe_max) / max(1e-6, d_awareness_far - d_safe_max), 0.0, 1.0))
                # 横向距离越大越好
                lateral_norm = float(np.clip(lateral_dist / max(0.1, d_danger), 0.0, 2.0))
                awareness_r = 0.03 * scale * awareness_ratio * min(1.0, lateral_norm)
                reward += awareness_r
                rt(info, "npc_awareness_bonus", awareness_r)

            elif dist_metric > d_proactive_near:
                # ═══ Proactive Zone: 主动避让区 (核心密集奖励) ═══
                # 距离越近、横向偏移越大 → 奖励越高
                proximity_ratio = 1.0 - float(np.clip(
                    (dist_metric - d_proactive_near) / max(1e-6, d_safe_max - d_proactive_near), 0.0, 1.0))
                # 横向距离归一化: lateral_dist / danger_close 作为"安全程度"
                lateral_safety = float(np.clip(lateral_dist / max(0.1, d_danger), 0.0, 2.0))

                # 基础奖励: 横向距离足够 + 在proactive zone
                proactive_r = self.proactive_reward_scale * scale * (0.3 + 0.7 * proximity_ratio) * min(1.0, lateral_safety)
                reward += proactive_r
                rt(info, "proactive_avoid_bonus", proactive_r)

                # 横向距离增长奖励: 检查横向距离是否在增加（agent正在远离NPC）
                if len(hist) >= 3:
                    prev_lateral = hist[-3]
                    lateral_delta = lateral_dist - prev_lateral
                    if lateral_delta > 0.01:
                        # 横向距离在增加（正在避让）
                        steer_away_r = 0.12 * scale * float(np.clip(lateral_delta / 0.15, 0.0, 1.0))
                        reward += steer_away_r
                        rt(info, "steer_away_bonus", steer_away_r)

            elif dist_metric > d_danger:
                # ═══ 近距离通过区 ═══
                # 横向距离决定: 距离够=安全通过奖励, 距离不够=惩罚
                lateral_safety = float(np.clip(lateral_dist / max(0.1, d_danger * 0.5), 0.0, 2.0))
                if lateral_safety > 0.8:
                    # 安全通过: 横向距离足够
                    passage_r = self.safe_passage_bonus * scale * min(1.0, lateral_safety - 0.5)
                    reward += passage_r
                    rt(info, "safe_passage_bonus", passage_r)
                else:
                    # 横向距离不足: 逼近惩罚（梯度连续，越近越重）
                    close_ratio = 1.0 - lateral_safety
                    close_pen = 0.15 * scale * close_ratio
                    reward -= close_pen
                    rt(info, "insufficient_lateral_penalty", -close_pen)

            else:
                # ═══ Danger Zone: 已进入碰撞危险区 ═══
                danger_ratio = float(np.clip((d_danger - dist_metric) / max(1e-6, d_danger), 0.0, 1.0))
                pen = 0.25 + 0.50 * danger_ratio
                if speed > 0.8:
                    pen += 0.15 * float(np.clip((speed - 0.8) / 0.8, 0.0, 1.0))
                if closing > 0.35:
                    pen += 0.15 * float(np.clip((closing - 0.35) / 1.0, 0.0, 1.0))
                reward -= pen * scale
                rt(info, "danger_penalty", -pen * scale)

                # Close Call Penalty: 进入danger zone时横向距离不够
                if lateral_dist < d_danger * 0.4:
                    reward -= self.close_call_penalty * scale
                    rt(info, "close_call_penalty", -self.close_call_penalty * scale)

                # 即使在danger zone，横向距离够大也给正向信号（鼓励侧向避让而非停车）
                if lateral_dist > d_danger * 0.6:
                    lateral_save_r = 0.15 * scale * float(np.clip(
                        (lateral_dist - d_danger * 0.6) / max(0.1, d_danger * 0.4), 0.0, 1.0))
                    reward += lateral_save_r
                    rt(info, "danger_lateral_save_bonus", lateral_save_r)

                self._episode_avoidance_success = False

                # 极近距离紧急惩罚
                if dist < 0.65 * d_danger:
                    reward -= 0.20 * scale
                    rt(info, "emergency_gap_penalty", -0.20 * scale)

            # ═══ 安全跟车/绕行窗口 (在所有距离段之外的通用信号) ═══
            if d_safe_min <= dist_metric <= d_safe_max and 0.30 < speed < 1.60:
                reward += 0.12 * scale
                rt(info, "safe_follow_bonus", 0.12 * scale)

            # ═══ 速度维持奖励: 鼓励在避障时保持前进，防止学到"停车"策略 ═══
            if speed > 0.4 and dist_metric > d_danger:
                speed_keep_r = float(max(0.0, self.v14_speed_maintain_bonus_scale)) * scale * float(
                    np.clip(speed / 1.2, 0.0, 1.0)
                )
                reward += speed_keep_r
                rt(info, "speed_maintain_bonus", speed_keep_r)

        return reward, done
    def _reward_dynamic_avoid(self, reward, done, info):
        """
        Stage 3: 动态避让奖励 (规则NPC跟车+超车)。

        在proactive_avoid基础上增加: 安全跟车带、动态超车奖励、追尾惩罚。
        """
        # 先应用主动避障 (scale=0.6, 因为动态场景需要更多跟车奖励)
        reward, done = self._reward_proactive_avoid(reward, done, info, scale=0.6)

        feats = self._npc_distance_features(info)
        if not feats:
            return reward, done

        d_danger = float(self.dist_scale.get('danger_close_sim', 1.2))
        d_safe_min = float(self.dist_scale.get('follow_safe_min_sim', 1.6))
        d_safe_max = float(self.dist_scale.get('follow_safe_max_sim', 4.0))
        rt = self._rt_add

        for f in feats:
            dist = f['dist']
            closing = f.get('closing_speed', 0.0)
            progress_diff = f.get('progress_diff', 0)

            # 安全跟车带奖励 (agent在NPC后方)
            if progress_diff < 0:
                if d_safe_min <= dist <= d_safe_max:
                    reward += 0.18
                    rt(info, "dynamic_follow_safe_band", 0.18)
                elif dist < d_danger and closing > 0.7:
                    reward -= self.rear_end_penalty
                    rt(info, "rear_end_risk_penalty", -self.rear_end_penalty)
                    info['rear_end_risk'] = True

            # 安全超车检测
            if hasattr(self, '_prev_progress_diff'):
                if self._prev_progress_diff < 0 and progress_diff > 0 and dist > d_danger:
                    next_count = int(self._episode_overtake_count) + 1
                    mult = 1.0 + float(self.v14_overtake_bonus_growth) * float(max(0, next_count - 1))
                    mult = float(min(float(self.v14_overtake_bonus_max_mult), mult))
                    overtake_reward = float(self.dynamic_overtake_bonus) * mult
                    reward += overtake_reward
                    rt(info, "dynamic_overtake_bonus", overtake_reward)
                    self._episode_overtake_count = int(next_count)
                    info['overtake_success'] = True
                    info['overtake_bonus_multiplier'] = float(mult)
            self._prev_progress_diff = progress_diff

        return reward, done
    def _reward_chaos_robust(self, reward, done, info):
        """
        Stage 4: 混沌NPC + 防遗忘奖励。

        组合proactive_avoid + 赛车线回忆 + KL蒸馏惩罚。
        """
        if self._episode_npc_free:
            # 无NPC episode: 使用Stage 1赛车线奖励
            reward = self._reward_racing_line(reward, info, weight=1.0)
        else:
            # 有NPC: 主动避障 + 动态避让
            reward, done = self._reward_proactive_avoid(reward, done, info, scale=0.8)

            # 当附近无NPC时，恢复赛车线奖励 (防遗忘)
            feats = self._npc_distance_features(info)
            npc_nearby = False
            d_safe_max = float(self.dist_scale.get('follow_safe_max_sim', 4.0))
            for f in (feats or []):
                if f['dist'] < d_safe_max:
                    npc_nearby = True
                    break
            if not npc_nearby:
                reward = self._reward_racing_line(
                    reward, info, weight=self.racing_line_weight_stage4
                )

        return reward, done
    def step(self, action):
        """V14 step: 调用parent step, 然后叠加V14奖励。"""
        obs, reward, done, info = super().step(action)
        if not isinstance(info, dict):
            info = {}
        rt = self._rt_add
        self._v14_sync_stage34_npc_speed_cap(info)
        self._v14_maybe_reset_npcs_on_stage34_contact(done, info)
        if self._v14_detect_stage34_collision(done, info):
            info["v14_stage34_collision_detected"] = True
            self._v14_force_collision_spawn_refresh = True
            reset_reason = "stage34_done_collision" if bool(done) else "stage34_runtime_collision"
            self._v14_reset_npcs_after_stage34_collision(info, reason=reset_reason)
        self._v14_record_learner_trace(info)
        self._v14_apply_npc_inplace_wobble()

        reward_mode = self.curriculum_stage_ref.get('reward_mode', 'racing_line')

        # ═══ Stage-specific rewards ═══
        # 赛车线奖励在所有阶段都保留（权重递减），防止驾驶能力退化
        if reward_mode == 'racing_line':
            reward = self._reward_racing_line(reward, info, weight=1.0)

        elif reward_mode == 'proactive_avoid':
            reward = self._reward_racing_line(reward, info, weight=0.5)
            reward, done = self._reward_proactive_avoid(reward, done, info, scale=1.0)

        elif reward_mode == 'dynamic_avoid':
            reward = self._reward_racing_line(reward, info, weight=0.35)
            reward, done = self._reward_dynamic_avoid(reward, done, info)

        elif reward_mode == 'chaos_robust':
            reward, done = self._reward_chaos_robust(reward, done, info)

        # 额外碰撞惩罚（比基础惩罚更强）
        collision_pen = float(max(0.0, self.v14_collision_extra_penalty))
        if collision_pen > 0.0:
            hit_v = str(info.get("hit", "none")).strip().lower()
            term_v = str(info.get("termination_reason", "")).strip().lower()
            collided = bool(hit_v not in ("none", "", "null") or term_v == "collision")
            if collided:
                reward -= collision_pen
                rt(info, "collision_extra_penalty", -collision_pen)

        # ═══ KL蒸馏惩罚 (所有Stage 2+) ═══
        if (self.distillation_manager is not None
                and self.distillation_manager.has_snapshot
                and self.distillation_model_ref is not None):
            try:
                obs_t = torch.as_tensor(obs[None]).float()
                device = next(self.distillation_model_ref.policy.parameters()).device
                obs_t = obs_t.to(device)
                kl_pen = self.distillation_manager.compute_kl_penalty(
                    self.distillation_model_ref, obs_t
                )
                if abs(kl_pen) > 1e-8:
                    reward += kl_pen
                    self._rt_add(info, "kl_distillation_penalty", kl_pen)
            except Exception:
                pass

        # 避障成功终局奖励（避免稀疏、强化“安全绕行后完成回合”）
        if bool(done) and bool(self._episode_avoidance_success):
            term = str(info.get('termination_reason', ''))
            if term in ('success_avoidance_2laps', 'success_laps_target', 'two_laps_success'):
                bonus = float(max(0.0, self.v14_avoidance_success_bonus))
                if bonus > 0.0:
                    reward += bonus
                    self._rt_add(info, "avoidance_success_bonus", bonus)

        # ═══ 圈速追踪 ═══
        lap_count = int(_safe_float(info.get('lap_count', 0), 0))
        if lap_count > len(self._episode_lap_times):
            now = time.time()
            if self._episode_lap_start_time is not None:
                lap_time = max(0.1, now - self._episode_lap_start_time)
                self._episode_lap_times.append(lap_time)
                info['last_lap_time'] = float(lap_time)
                base_bonus = float(max(0.0, self.v14_lap_complete_bonus))
                lap_ratio = float(np.clip(
                    float(self.v14_lap_time_ref_sec) / max(0.1, float(lap_time)),
                    0.2,
                    float(max(0.5, self.v14_lap_time_reward_max_ratio)),
                ))
                lap_time_bonus = float(max(0.0, self.v14_lap_time_reward_scale)) * lap_ratio
                lap_total_bonus = float(base_bonus + lap_time_bonus)
                reward += lap_total_bonus
                rt(info, "lap_complete_bonus", base_bonus)
                rt(info, "lap_time_speed_bonus", lap_time_bonus)
                info['v14_lap_reward_total'] = float(lap_total_bonus)
                info['v14_lap_time_ratio'] = float(lap_ratio)
            self._episode_lap_start_time = now

        # ═══ V14 info ═══
        info['v14_reward_mode'] = str(reward_mode)
        info['v14_episode_npc_free'] = bool(self._episode_npc_free)
        info['v14_overtake_count'] = int(self._episode_overtake_count)
        info['v14_avoidance_success'] = bool(self._episode_avoidance_success)
        if self._episode_lap_times:
            info['v14_avg_lap_time'] = float(np.mean(self._episode_lap_times))

        return obs, float(reward), bool(done), info


__all__ = ["V14RewardMixin"]
