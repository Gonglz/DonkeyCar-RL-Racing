#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V14 racing line computer."""

import math
import numpy as np

from .v14_dep_generatedtrack_base import ManualWidthSpawnSampler

class RacingLineComputer:
    """
    从赛道曲率 profile 预计算最优赛车线偏移。

    基本原理: 弯道中 racing line 走 外-内-外 路径以最小化曲率。
    对每个 fine_track 点，根据前方窗口内的平均曲率和方向，计算横向偏移量。
    正偏移=左，负偏移=右（相对赛道前进方向）。
    """

    def __init__(self, manual_spawn: ManualWidthSpawnSampler, lookahead_window=20,
                 smoothing_window=10, max_offset_ratio=0.35):
        """
        Args:
            manual_spawn: ManualWidthSpawnSampler, 提供fine_track和边界
            lookahead_window: 前方曲率平均窗口（fine_track索引数）
            smoothing_window: 赛车线偏移平滑窗口
            max_offset_ratio: 最大偏移占半宽的比例 (0~0.5)
        """
        self.manual_spawn = manual_spawn
        self.lookahead_window = int(max(5, lookahead_window))
        self.smoothing_window = int(max(3, smoothing_window))
        self.max_offset_ratio = float(np.clip(max_offset_ratio, 0.1, 0.48))
        self.offsets_cte = np.zeros(0, dtype=np.float64)  # 赛车线偏移 (CTE单位)
        self.kappa_profile = np.zeros(0, dtype=np.float64)  # 曲率 profile
        self.loaded = False

        if manual_spawn.loaded:
            self._compute()

    def _compute(self):
        """预计算赛车线偏移。"""
        ms = self.manual_spawn
        n = int(ms.fine_track.shape[0])
        if n < 10:
            self.loaded = False
            return

        # Step 1: 计算每点的曲率
        kappa = np.zeros(n, dtype=np.float64)
        for i in range(n):
            kappa[i] = ms.estimate_kappa(i)
        self.kappa_profile = kappa.copy()

        # Step 2: 计算前方窗口平均带符号曲率
        # 符号: 用cross product判断弯道方向 (左弯=正, 右弯=负)
        signed_kappa = np.zeros(n, dtype=np.float64)
        for i in range(n):
            a = ms.fine_track[(i - 1) % n]
            b = ms.fine_track[i]
            c = ms.fine_track[(i + 1) % n]
            bax = float(a[0] - b[0])
            baz = float(a[1] - b[1])
            bcx = float(c[0] - b[0])
            bcz = float(c[1] - b[1])
            cross = bax * bcz - baz * bcx  # >0=左弯, <0=右弯
            signed_kappa[i] = float(np.sign(cross)) * kappa[i]

        # Step 3: 前方lookahead平均 + 平滑
        lookahead_avg = np.zeros(n, dtype=np.float64)
        for i in range(n):
            indices = [(i + j) % n for j in range(self.lookahead_window)]
            lookahead_avg[i] = np.mean(signed_kappa[indices])

        # 移动平均平滑
        smoothed = np.zeros(n, dtype=np.float64)
        hw = self.smoothing_window // 2
        for i in range(n):
            indices = [(i + j - hw) % n for j in range(self.smoothing_window)]
            smoothed[i] = np.mean(lookahead_avg[indices])

        # Step 4: 映射到横向偏移 (CTE单位)
        # 左弯(positive kappa) → racing line偏向内侧(右, negative offset)
        # 右弯(negative kappa) → racing line偏向内侧(左, positive offset)
        max_kappa = float(np.percentile(np.abs(kappa[kappa > 0]) if np.any(kappa > 0) else [1.0], 90))
        max_kappa = max(0.1, max_kappa)

        offsets = np.zeros(n, dtype=np.float64)
        for i in range(n):
            hw_cte = float(ms.half_width_cte[i]) if ms.half_width_cte.size > 0 else 1.0
            ratio = float(np.clip(-smoothed[i] / max_kappa, -1.0, 1.0))
            offsets[i] = ratio * self.max_offset_ratio * hw_cte

        self.offsets_cte = offsets
        self.loaded = True

        offset_abs_mean = float(np.mean(np.abs(offsets)))
        offset_max = float(np.max(np.abs(offsets)))
        print(f"   🏎️ RacingLine: n={n}, mean_offset={offset_abs_mean:.4f}, max_offset={offset_max:.4f}, "
              f"max_kappa_90p={max_kappa:.3f}")

    def get_offset_cte(self, fine_idx):
        """返回指定fine_track点的赛车线横向偏移 (CTE单位)。"""
        if not self.loaded or self.offsets_cte.size == 0:
            return 0.0
        return float(self.offsets_cte[int(fine_idx) % int(self.offsets_cte.shape[0])])

    def get_kappa(self, fine_idx):
        """返回指定fine_track点的曲率。"""
        if not self.loaded or self.kappa_profile.size == 0:
            return 0.0
        return float(self.kappa_profile[int(fine_idx) % int(self.kappa_profile.shape[0])])


__all__ = ["RacingLineComputer"]
