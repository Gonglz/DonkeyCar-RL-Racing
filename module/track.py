"""
module/track.py
赛道几何管理：SceneGeometry 数据类 + TrackGeometryManager 查询器。
从 ppo_waveshare_v12 提取，支持 9 地图非对称 CTE 边界 + 角点采样。
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

MODULE_TRACK_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "track_data"))

def _wrap_pi(x: float) -> float:
    return float((float(x) + math.pi) % (2.0 * math.pi) - math.pi)


def _clip_float(x: float, lo: float, hi: float) -> float:
    return float(min(max(float(x), float(lo)), float(hi)))


# 三条已标定赛道的真实出界 CTE 阈值（仅针对已知的 3 个场景硬编码）
# 来源：manual_width_probe.summary 中的 sim 偏移值 × coord_scale=8
# 符号约定：lat_err > 0 = 赛道左侧（cte_left* 为正），lat_err < 0 = 赛道右侧（cte_right* 为负）
# 模拟器参考中线偏离赛道几何中心，故 left/right 幅值不对称
_SCENE_CTE_TABLE: Dict[str, Dict[str, float]] = {
    "generated_track": {
        "cte_left":      +6.60,   # 0.825 * 8
        "cte_right":     -2.60,   # 0.325 * 8
        "cte_left_out":  +6.80,   # 0.850 * 8
        "cte_right_out": -2.80,   # 0.350 * 8
    },
    "roboracingleague_track": {
        "cte_left":      +4.40,   # 0.550 * 8
        "cte_right":     -4.80,   # 0.600 * 8
        "cte_left_out":  +4.60,   # 0.575 * 8
        "cte_right_out": -5.00,   # 0.625 * 8
    },
    "waveshare": {
        "cte_left":      +2.40,   # 0.300 * 8
        "cte_right":     -2.00,   # 0.250 * 8
        "cte_left_out":  +2.60,   # 0.325 * 8
        "cte_right_out": -2.20,   # 0.275 * 8
    },
}


@dataclass
class SceneGeometry:
    """单条赛道的离线几何信息。"""
    scene_key: str
    center: np.ndarray        # (N, 2) x-z
    left: np.ndarray          # (N, 2)
    right: np.ndarray         # (N, 2)
    tangent: np.ndarray       # (N, 2) 单位切向量
    seg_len: np.ndarray       # (N,)   每段长度
    cum_len: np.ndarray       # (N,)   累积弧长
    loop_len: float           # 整圈弧长
    width: np.ndarray         # (N,)   局部赛道宽度
    width_median: float
    cte_left: float           # 左侧在界最大 CTE（正值，lat_err > 0 方向）
    cte_right: float          # 右侧在界最大 CTE（负值，lat_err < 0 方向）
    cte_left_out: float       # 左侧首次确认出界 CTE（正值，>= cte_left）
    cte_right_out: float      # 右侧首次确认出界 CTE（负值，<= cte_right）
    cte_half_width: float     # (cte_left - cte_right) / 2 —— CTE 奖励归一化因子（正值）
    coord_scale: float        # sim 坐标缩放系数（lat_err * coord_scale ≈ -sim_cte，与 CTE 表同单位）
    corner_nodes: List[int]   # Top-20% 高曲率节点索引（角点优先采样用）


class TrackGeometryManager:
    """加载多赛道 JSON 并提供实时局部几何查询（x-z 平面）。"""

    def __init__(
        self,
        track_dir: str,
        env_ids: List[str],
        scene_specs: Dict[str, Dict[str, str]],
        lookahead_points: int = 12,
    ):
        """
        Args:
            track_dir:    赛道 JSON 目录（.../track/）。
            env_ids:      需要加载的 gym env_id 列表。
            scene_specs:  SCENE_SPECS 字典（来自 v12 的顶层配置）。
            lookahead_points: 前视节点数，用于计算 kappa_lookahead。
        """
        self.track_dir = track_dir
        self.lookahead_points = int(max(2, lookahead_points))
        self.scenes: Dict[str, SceneGeometry] = {}

        for env_id in env_ids:
            if env_id not in scene_specs:
                raise KeyError(f"Unknown env_id in scene_specs: {env_id}")
            spec = scene_specs[env_id]
            scene_key = spec["scene_key"]
            if scene_key in self.scenes:
                continue
            path = os.path.join(track_dir, spec["track_file"])
            self.scenes[scene_key] = self._load_scene_geometry(scene_key, path)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_scene_geometry(self, scene_key: str, path: str) -> SceneGeometry:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Track JSON missing: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        outline = data.get("outline", {})
        center = np.asarray(outline.get("fine_track_xz", []), dtype=np.float64)
        left   = np.asarray(outline.get("left_boundary_xz", []), dtype=np.float64)
        right  = np.asarray(outline.get("right_boundary_xz", []), dtype=np.float64)

        if center.ndim != 2 or center.shape[1] != 2:
            raise ValueError(f"Invalid centerline in {path}")
        if left.shape != center.shape or right.shape != center.shape:
            raise ValueError(f"Boundary shape mismatch in {path}")
        if center.shape[0] <= 100:
            raise ValueError(f"Too few points in {path}: {center.shape[0]}")

        center_next = np.roll(center, -1, axis=0)
        d = center_next - center
        seg_len = np.linalg.norm(d, axis=1)
        seg_len = np.maximum(seg_len, 1e-6)
        tangent = d / seg_len[:, None]

        cum_len = np.zeros(center.shape[0], dtype=np.float64)
        cum_len[1:] = np.cumsum(seg_len[:-1])
        loop_len = float(np.sum(seg_len))

        width = np.linalg.norm(left - right, axis=1)
        width = np.maximum(width, 1e-4)
        width_median = float(np.median(width))

        # 读取左右 CTE 出界边界
        # 优先使用硬编码标定表（3 个已知场景）；其余场景回退到 JSON probe summary
        if scene_key in _SCENE_CTE_TABLE:
            tb = _SCENE_CTE_TABLE[scene_key]
            cte_left      = tb["cte_left"]       # 正值
            cte_right     = tb["cte_right"]      # 负值
            cte_left_out  = tb["cte_left_out"]   # 正值，>= cte_left
            cte_right_out = tb["cte_right_out"]  # 负值，<= cte_right
            coord_scale   = 8.0                  # 已标定场景均使用 coord_scale=8
        else:
            # 回退：从 JSON probe summary 动态计算；right 取反使其为负
            coord_scale = float(data.get("coord_scale", 1.0))
            probe   = data.get("manual_width_probe", {})
            summary = probe.get("summary", {})
            cte_left  = float(summary.get("left_in_max_sim",   1.0)) * coord_scale
            cte_right = -float(summary.get("right_in_max_sim", 1.0)) * coord_scale
            cte_left_out  = float(summary.get("left_out_first_sim",
                                               abs(cte_left)  / coord_scale * 1.1)) * coord_scale
            cte_right_out = -float(summary.get("right_out_first_sim",
                                                abs(cte_right) / coord_scale * 1.1)) * coord_scale

        # 计算曲率，找出 Top-20% 高曲率角点
        curvature = self._compute_curvature(center)
        n_nodes = len(center)
        n_corners = max(4, int(np.ceil(n_nodes * 0.2)))
        corner_indices = np.argsort(curvature)[-n_corners:]
        corner_nodes = sorted(int(i) for i in corner_indices)

        _cte_left_final  = max(cte_left,  0.1)
        _cte_right_final = min(cte_right, -0.1)
        return SceneGeometry(
            scene_key=scene_key,
            center=center,
            left=left,
            right=right,
            tangent=tangent,
            seg_len=seg_len,
            cum_len=cum_len,
            loop_len=loop_len,
            width=width,
            width_median=max(width_median, 1e-3),
            cte_left=_cte_left_final,
            cte_right=_cte_right_final,
            cte_left_out=max(cte_left_out, _cte_left_final),
            cte_right_out=min(cte_right_out, _cte_right_final),
            cte_half_width=0.5 * (_cte_left_final - _cte_right_final),
            coord_scale=float(max(coord_scale, 1e-3)),
            corner_nodes=corner_nodes,
        )

    def _compute_curvature(self, center: np.ndarray) -> np.ndarray:
        """每个节点处的曲率（相邻向量叉积绝对值）。"""
        n = len(center)
        curvature = np.zeros(n, dtype=np.float64)
        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            v1 = center[i] - center[prev_idx]
            v2 = center[next_idx] - center[i]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1e-6 or norm2 < 1e-6:
                curvature[i] = 0.0
            else:
                u1 = v1 / norm1
                u2 = v2 / norm2
                cross = abs(u1[0] * u2[1] - u1[1] * u2[0])
                curvature[i] = float(cross)
        return curvature

    def _circular_arc_delta(self, g: SceneGeometry, i0: int, i1: int) -> float:
        if i1 >= i0:
            ds = g.cum_len[i1] - g.cum_len[i0]
        else:
            ds = (g.loop_len - g.cum_len[i0]) + g.cum_len[i1]
        return float(max(ds, 1e-6))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def list_scenes(self) -> List[str]:
        return sorted(self.scenes.keys())

    def scene_summary(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for k, g in self.scenes.items():
            out[k] = {
                "points":     int(g.center.shape[0]),
                "loop_len":   float(g.loop_len),
                "width_mean": float(np.mean(g.width)),
                "width_min":  float(np.min(g.width)),
                "width_max":  float(np.max(g.width)),
                "cte_left":      float(g.cte_left),
                "cte_right":     float(g.cte_right),
                "cte_left_out":  float(g.cte_left_out),
                "cte_right_out": float(g.cte_right_out),
                "cte_half_width": float(g.cte_half_width),
                "n_corners":     int(len(g.corner_nodes)),
            }
        return out

    def query(
        self,
        scene_key: str,
        x: float,
        z: float,
        yaw_rad: float,
        prev_idx: Optional[int] = None,
        local_window: int = 180,
    ) -> Dict[str, float]:
        """
        查询车辆在赛道上的局部几何信息。

        Returns:
            dict with keys: idx, lat_err_norm, heading_err_sin, heading_err_cos,
                            kappa_lookahead, width_norm
        """
        g = self.scenes[scene_key]
        n = g.center.shape[0]

        if prev_idx is None:
            dx = g.center[:, 0] - float(x)
            dz = g.center[:, 1] - float(z)
            idx = int(np.argmin(dx * dx + dz * dz))
        else:
            base = int(prev_idx) % n
            offsets = np.arange(-local_window, local_window + 1)
            cands = (base + offsets) % n
            c = g.center[cands]
            dx = c[:, 0] - float(x)
            dz = c[:, 1] - float(z)
            idx = int(cands[int(np.argmin(dx * dx + dz * dz))])

        cx, cz = g.center[idx]
        tx, tz = g.tangent[idx]
        nx, nz = -tz, tx

        ex = float(x) - float(cx)
        ez = float(z) - float(cz)
        lat_err = ex * nx + ez * nz

        local_width = float(g.width[idx])
        half_w = max(0.5 * local_width, 1e-3)
        lat_err_norm = _clip_float(lat_err / half_w, -3.0, 3.0)

        track_yaw = math.atan2(float(tz), float(tx))
        heading_err = _wrap_pi(float(yaw_rad) - track_yaw)
        heading_err_sin = math.sin(heading_err)
        heading_err_cos = math.cos(heading_err)

        j = (idx + self.lookahead_points) % n
        tx2, tz2 = g.tangent[j]
        yaw2 = math.atan2(float(tz2), float(tx2))
        dpsi = _wrap_pi(yaw2 - track_yaw)
        ds = self._circular_arc_delta(g, idx, j)
        kappa = _clip_float(dpsi / ds, -2.0, 2.0)

        width_norm = _clip_float(local_width / g.width_median, 0.0, 3.0)

        return {
            "idx":                float(idx),
            "lat_err":            float(lat_err),
            "lat_err_norm":       float(lat_err_norm),
            "heading_err_sin":    float(heading_err_sin),
            "heading_err_cos":    float(heading_err_cos),
            "kappa_lookahead":    float(kappa),
            "width_norm":         float(width_norm),
        }
