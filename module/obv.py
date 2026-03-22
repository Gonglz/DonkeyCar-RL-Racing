"""
module/obv.py
V13 观测空间：6 通道语义图像 + 5 维纯传感器状态。

观测结构
--------
image: (6, obs_size, obs_size) float32 [0, 1]
  ch0  raw_Y             原图 Y 亮度通道
  ch1  blue_boundary_prob  蓝色边界软概率（*2NewTrack 规范图）
  ch2  yellow_line_prob    中心线软概率（*2NewTrack，渐进 dropout）
  ch3  sobel_edge          Sobel 边缘图（raw_Y）
  ch4  vehicle_prob        动态目标软概率（GreenVehicleDetector）
  ch5  motion_residual     |Y_t - Y_{t-1}|，reset 后置零

state: (5,) float32
  v_long_norm      clip(speed / v_max, 0, 2)
  yaw_rate_norm    clip(gyro_z / 4.0, -2, 2)
  accel_x_norm     clip(accel_x / 9.8, -2, 2)   ← 纵向加速度（上线前需 sanity check）
  prev_steer       ∈ [-1, 1]
  prev_throttle    ∈ [-1, 1]

设计原则
--------
- 不输入整张 mapped 图；*2NewTrack 仅用于提取 ch1/ch2 概率通道
- 软概率（Gaussian blur）而非硬二值，通道 dropout 提升鲁棒性
- motion_residual 替代帧堆叠；配合 RecurrentPPO LSTM 处理长时依赖
- 状态向量完全来自传感器 info，无 TrackGeometryManager 依赖

RGB/BGR 约定
------------
- DonkeyEnv 输出 RGB
- *2NewTrack 和 GreenVehicleDetector 期望 BGR
- cv2.resize 参数顺序：(width, height)，即 (W, H)
"""

from typing import Any, Dict, Optional

import cv2
import gym
import numpy as np


# ============================================================
# WS 规范化颜色常量（无需加载参考图）
# ============================================================
_CANONICAL_TGT_STATS: Dict[str, Any] = {
    "prototypes_hsv": {
        "blue":   [112.0, 125.0,  94.0],
        "yellow": [ 13.0, 152.0, 146.0],
        "white":  [  0.0,   0.0, 200.0],
    },
    "centerline_hsv": [13.0, 152.0, 146.0],
}


# ============================================================
# CanonicalSemanticWrapper
# ============================================================
class CanonicalSemanticWrapper(gym.ObservationWrapper):
    """
    V13 专用：6 通道观测 = 原图辅助流 + *2NewTrack 语义流。

    通道布局 (6, obs_size, obs_size) float32 [0,1]:
      ch0  raw_Y               原图 Y 亮度通道
      ch1  blue_boundary_prob  蓝色边界软概率（规范图）
      ch2  yellow_line_prob    中心线软概率（规范图，渐进 dropout）
      ch3  sobel_edge          Sobel 边缘图（raw_Y）
      ch4  vehicle_prob        动态目标软概率（GreenVehicleDetector）
      ch5  motion_residual     |Y_t - Y_{t-1}|，reset 后清零
    """

    def __init__(
        self,
        env,
        domain: str = "ws",
        obs_size: int = 128,
        augment: bool = False,
        dropout_start_step: int = 0,
        dropout_ramp_steps: int = 200_000,
        dropout_max_prob: float = 0.20,
    ):
        super().__init__(env)
        self._domain = str(domain).lower()
        self.obs_size = int(obs_size)
        self.H = self.obs_size
        self.W = self.obs_size
        self.augment = bool(augment)
        self.dropout_start_step = int(max(0, dropout_start_step))
        self.dropout_ramp_steps = int(max(1, dropout_ramp_steps))
        self.dropout_max_prob = float(np.clip(dropout_max_prob, 0.0, 1.0))

        self._step = 0
        self._prev_y: Optional[np.ndarray] = None

        from .green_vehicle_detect import GreenVehicleDetector
        self._green_det = GreenVehicleDetector()

        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0,
            shape=(6, self.obs_size, self.obs_size),
            dtype=np.float32,
        )
        print(
            f"✅ CanonicalSemanticWrapper [{domain}]: "
            f"(6,{obs_size},{obs_size}) float32, "
            f"dropout_max={dropout_max_prob:.2f} ramp={dropout_ramp_steps}"
        )

    # ------------------------------------------------------------------
    # Dropout 渐进 schedule
    # ------------------------------------------------------------------
    def _get_yellow_dropout(self) -> float:
        """渐进式 dropout 概率：step < start → 0；线性增至 dropout_max_prob。"""
        if self._step < self.dropout_start_step:
            return 0.0
        ramp = float(self._step - self.dropout_start_step) / self.dropout_ramp_steps
        return float(np.clip(ramp * self.dropout_max_prob, 0.0, self.dropout_max_prob))

    # ------------------------------------------------------------------
    # 域适配：各域原图 BGR → 规范 BGR
    # ------------------------------------------------------------------
    def _canonical_transform(self, img_bgr: np.ndarray) -> np.ndarray:
        if self._domain == "ws":
            from .WS2NewTrack import transform_ws_to_newtrack
            return transform_ws_to_newtrack(img_bgr, _CANONICAL_TGT_STATS)
        elif self._domain == "rrl":
            from .RRL2NewTrack import transform_rrl_to_newtrack
            return transform_rrl_to_newtrack(img_bgr)
        else:  # gt
            from .GT2NewTrack import (
                detect_gt_road_support, render_road_surface,
                detect_white_line, detect_yellow_line, recolor_lines,
            )
            rs = detect_gt_road_support(img_bgr)
            ri = render_road_surface(img_bgr)
            return recolor_lines(
                ri,
                detect_white_line(img_bgr, road_support=rs),
                detect_yellow_line(img_bgr, road_support=rs),
            )

    # ------------------------------------------------------------------
    # 主观测构建
    # ------------------------------------------------------------------
    def observation(self, obs: np.ndarray) -> np.ndarray:
        """
        输入: DonkeyEnv 原始 RGB 帧 (H, W, C) uint8
        输出: (6, obs_size, obs_size) float32 [0, 1]
        """
        self._step += 1
        img = np.asarray(obs, dtype=np.uint8)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]

        # DonkeyEnv 输出 RGB；cv2.resize 参数为 (width, height)
        raw_rgb = cv2.resize(img, (self.W, self.H), interpolation=cv2.INTER_LINEAR)

        # ch0: raw Y  —— RGB→YCrCb，取第 0 分量（亮度）
        raw_y = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32) / 255.0

        # *2NewTrack 和 GreenDetector 均期望 BGR
        raw_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)

        # ch1, ch2: 规范图（BGR）→ HSV → 语义 mask → 软概率
        canonical_bgr = self._canonical_transform(raw_bgr)
        hsv = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2HSV)
        from .WS2NewTrack import semantic_masks
        smasks = semantic_masks(hsv)

        blue_prob = cv2.GaussianBlur(
            smasks["blue"].astype(np.float32), (5, 5), sigmaX=1.5)

        # *2NewTrack 已将所有中线统一渲染为黄/橙；white ≈ 空集，勿合并
        yellow_raw = smasks["yellow"].astype(np.float32)
        yellow_prob = cv2.GaussianBlur(yellow_raw, (5, 5), sigmaX=1.5)
        if np.max(yellow_raw) < 0.01:
            # 地图先天无中线 → 全零，与 augment 无关
            yellow_prob = np.zeros_like(yellow_prob)
        elif self.augment and np.random.random() < self._get_yellow_dropout():
            yellow_prob = np.zeros_like(yellow_prob)   # 渐进 dropout

        # ch3: Sobel 边缘（在 raw_Y 上，颜色无关）
        gx = cv2.Sobel(raw_y, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(raw_y, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.clip(np.sqrt(gx ** 2 + gy ** 2) / 4.0, 0.0, 1.0)

        # ch4: vehicle prob（GreenVehicleDetector，期望 BGR）
        det = self._green_det.detect(raw_bgr)
        if det.detected:
            veh = (det.mask > 0).astype(np.float32)
            veh_prob = cv2.GaussianBlur(veh, (5, 5), sigmaX=1.5)
        else:
            veh_prob = np.zeros((self.H, self.W), dtype=np.float32)

        # ch5: motion residual |Y_t - Y_{t-1}|，放大小运动
        if self._prev_y is not None:
            motion = np.clip(np.abs(raw_y - self._prev_y) * 4.0, 0.0, 1.0)
        else:
            motion = np.zeros_like(raw_y)
        self._prev_y = raw_y.copy()

        return np.stack(
            [raw_y, blue_prob, yellow_prob, edge, veh_prob, motion], axis=0
        ).astype(np.float32)

    def reset(self, **kwargs):
        self._prev_y = None   # reset 时清除运动历史
        return super().reset(**kwargs)


# ============================================================
# _build_state_v13
# ============================================================
def _build_state_v13(
    info: Dict[str, Any],
    action_safety_wrapper,
    control_wrapper,
    v_max: float = 2.2,
) -> np.ndarray:
    """
    返回 7 维状态向量，完全来自传感器 info + adapter 内部状态，无 TrackGeometryManager 依赖：
      [v_long_norm, yaw_rate_norm, accel_x_norm, prev_steer, prev_throttle,
       steer_core, bias_smooth]

    归一化范围：
      v_long_norm      = clip(speed / v_max,      0,   2)
      yaw_rate_norm    = clip(gyro_z / 4.0,      -2,   2)
      accel_x_norm     = clip(accel_x / 9.8,     -2,   2)   ← 纵向加减速
      prev_steer       ∈ [-1, 1]   ← 上一已执行低层 steer（safety 约束后）
      prev_throttle    ∈ [-1, 1]   ← 上一已执行低层 throttle（adapter 输出）
      steer_core       ∈ [-1, 1]   ← ActionAdapterWrapper 积分器内部状态
      bias_smooth      ∈ [-1, 1]   ← ActionAdapterWrapper 占位意图（低通滤波后原始值）

    ⚠️  accel_x 上线前需 sanity check（直线大油门起步应显著为正）。
        若不可信，可替换为 delta_speed 或退化为 6 维状态。

    兼容性：若 control_wrapper 无 steer_core/bias_smooth 属性（V12 path），
    对应维度退化为 0.0。
    """
    v = float(info.get("speed", 0.0) or 0.0)
    gyro  = info.get("gyro",  (0.0, 0.0, 0.0))
    accel = info.get("accel", (0.0, 0.0, 0.0))
    try:
        gz = float(gyro[2])
    except Exception:
        gz = 0.0
    try:
        ax = float(accel[0])
    except Exception:
        ax = 0.0

    prev_steer = 0.0
    prev_throttle = 0.0
    if action_safety_wrapper is not None:
        try:
            prev_steer = float(action_safety_wrapper.diag.get("steer_exec", 0.0))
        except Exception:
            pass
    if control_wrapper is not None:
        try:
            prev_throttle = float(control_wrapper.last_low_level_action[1])
            if action_safety_wrapper is None:
                prev_steer = float(control_wrapper.last_low_level_action[0])
        except Exception:
            pass

    # adapter internal state (V13 ActionAdapterWrapper exposes these)
    steer_core = 0.0
    bias_smooth = 0.0
    if control_wrapper is not None:
        try:
            steer_core = float(getattr(control_wrapper, "steer_core", 0.0))
        except Exception:
            steer_core = 0.0
        try:
            bias_smooth = float(getattr(control_wrapper, "bias_smooth", 0.0))
        except Exception:
            bias_smooth = 0.0

    return np.array([
        float(np.clip(v   / v_max, 0.0,  2.0)),       # v_long_norm
        float(np.clip(gz  / 4.0,  -2.0,  2.0)),       # yaw_rate_norm
        float(np.clip(ax  / 9.8,  -2.0,  2.0)),       # accel_x_norm
        float(np.clip(prev_steer,    -1.0,  1.0)),     # prev_steer_exec
        float(np.clip(prev_throttle, -1.0,  1.0)),     # prev_throttle_exec
        float(np.clip(steer_core,    -1.0,  1.0)),     # steer_core
        float(np.clip(bias_smooth,   -1.0,  1.0)),     # bias_smooth
    ], dtype=np.float32)
