"""
module/wrappers.py
所有 Gym 观测增强包装器：V8 obs wrappers + V9 域对齐包装器 + V12 RGB 预处理。
控制相关 wrapper 已迁移到 module/control.py。
"""

import random
from collections import deque
from typing import Any, Dict, Optional, Tuple

import cv2
import gym
import numpy as np

# V9 鲁棒检测器
from .robust_lane_detector import RobustLaneDetector, RobustYellowLaneEnhancer

from .control import (
    ActionSafetyWrapper,
    ThrottleControlWrapper,
    CurvatureAwareThrottleWrapper,
)


# ============================================================
# V8 泛化增强包装器
# ============================================================
class GeneralizationWrapper(gym.ObservationWrapper):
    """
    泛化性增强包装器 - 在 enable_step 步后对 RGB 通道施加亮度/噪声扰动。
    V8: 自管理步数，不依赖外部更新。输入: int16 HWC 8通道。
    """

    def __init__(self, env, enable_step: int = 100000):
        super().__init__(env)
        self.enable_step = enable_step
        self.current_step = 0
        print(f"🆕 泛化性增强: {enable_step:,} 步后启用（只对 RGB 通道）")

    def _random_brightness_contrast(self, rgb: np.ndarray) -> np.ndarray:
        rgb = rgb.astype(np.float32)
        if np.random.random() > 0.5:
            rgb = np.clip(rgb + np.random.uniform(-30, 30), 0, 255)
        if np.random.random() > 0.5:
            rgb = np.clip(rgb * np.random.uniform(0.7, 1.3), 0, 255)
        return rgb.astype(np.int16)

    def _random_noise(self, rgb: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.7:
            noise = np.random.normal(0, np.random.uniform(5, 15), rgb.shape)
            return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.int16)
        return rgb

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self.current_step += 1
        if self.current_step == self.enable_step:
            print(f"\n🎯 泛化性增强在第 {self.current_step} 步启用！")
        if self.current_step >= self.enable_step:
            rgb = obs[:, :, :3].copy()
            if np.random.random() > 0.3:
                rgb = self._random_brightness_contrast(rgb)
            if np.random.random() > 0.3:
                rgb = self._random_noise(rgb)
            result = obs.copy()
            result[:, :, :3] = rgb
            return result
        return obs

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# ============================================================
# V8 HWC→CHW 转换
# ============================================================
class TransposeWrapper(gym.ObservationWrapper):
    """HWC → CHW（PyTorch 格式）。"""

    def __init__(self, env):
        super().__init__(env)
        old = env.observation_space.shape
        new_shape = (old[2], old[0], old[1])
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )
        print(f"✅ TransposeWrapper: {old} -> {new_shape}")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return np.transpose(obs, (2, 0, 1))


# ============================================================
# V8 归一化包装器
# ============================================================
class NormalizeWrapper(gym.ObservationWrapper):
    """
    V8 8 通道归一化（V9.4 修正 per-channel bounds）：
      通道 0-2 (RGB): /255 → [0, 1]
      通道 3-5 (DiffRGB): /255 → [-1, 1]
      通道 6 (Mask): /255 → [0, 1]
      通道 7 (Edges): /255 → [0, 1]
    输入: CHW int16，输出: CHW float32。
    """

    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape  # (C, H, W)
        C, H, W = old_shape
        low  = np.zeros(old_shape, dtype=np.float32)
        high = np.ones(old_shape,  dtype=np.float32)
        low[3:6]  = -1.0
        high[3:6] =  1.0
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        print(f"✅ NormalizeWrapper: RGB/Mask/Edges→[0,1], DiffRGB→[-1,1] (V9.4 per-channel bounds)")

    def observation(self, obs: np.ndarray) -> np.ndarray:
        result = obs.astype(np.float32)
        result[0:3] /= 255.0
        result[3:6] /= 255.0
        result[6:7] /= 255.0
        result[7:8] /= 255.0
        return result


# 说明：
# ActionSafetyWrapper / ThrottleControlWrapper / CurvatureAwareThrottleWrapper
# 已迁移到 module/control.py，本文件仅保留与视觉预处理相关的 wrapper。


# ============================================================
# V9 域对齐黄线/边界检测包装器
# ============================================================
class V9YellowLaneWrapper(gym.ObservationWrapper):
    """
    V9.4 三域对齐增强包装器：per-domain edge detection + 行质心编码 + Coverage Dropout。
    接口与 V8 完全兼容（8通道 HWC int16）。
    """

    VALID_DIFF_MODES = ("dr_rgb", "off", "clean_rgb", "clean_gray")
    VALID_MASK_MODES = ("lane", "zero")

    def __init__(
        self,
        env,
        target_size: Tuple[int, int] = (128, 128),
        enable_dr: bool = False,
        dr_prob: float = 0.6,
        domain: str = "ws",
        detector_kwargs: Optional[Dict[str, Any]] = None,
        diff_mode: str = "dr_rgb",
        mask_mode: str = "lane",
        mask_scale: float = 1.0,
    ):
        super().__init__(env)
        self.target_size = target_size
        self.domain = domain
        self.diff_mode = str(diff_mode).lower()
        if self.diff_mode not in self.VALID_DIFF_MODES:
            raise ValueError(f"无效 diff_mode={diff_mode}, 可选: {self.VALID_DIFF_MODES}")
        self.mask_mode = str(mask_mode).lower()
        if self.mask_mode not in self.VALID_MASK_MODES:
            raise ValueError(f"无效 mask_mode={mask_mode}, 可选: {self.VALID_MASK_MODES}")
        self.mask_scale = float(np.clip(mask_scale, 0.0, 1.0))

        det_kwargs = dict(detector_kwargs or {})
        det_kwargs["domain"] = domain
        detector = RobustLaneDetector(**det_kwargs)
        self.enhancer = RobustYellowLaneEnhancer(
            enable_dr=enable_dr, dr_prob=dr_prob, detector=detector
        )
        self.prev_rgb_dr    = None
        self.prev_rgb_clean = None
        self.prev_gray_clean = None

        self._coverage_history: deque = deque(maxlen=200)

        self.observation_space = gym.spaces.Box(
            low=-255, high=255,
            shape=(target_size[0], target_size[1], 8),
            dtype=np.int16,
        )
        print(f"✅ V9.4 V9YellowLaneWrapper [{domain}]: {target_size} -> 8ch HWC, diff={diff_mode}, mask={mask_mode}")

    # ------ helpers ------

    def _compute_coverage(self, lane_mask: np.ndarray) -> float:
        h = lane_mask.shape[0]
        nonzero_rows = np.count_nonzero(np.any(lane_mask > 0, axis=1))
        return nonzero_rows / h if h > 0 else 0.0

    def get_coverage_stats(self) -> Dict[str, float]:
        if len(self._coverage_history) == 0:
            return {"mean": 0.0, "std": 0.0}
        arr = np.array(self._coverage_history)
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr))}

    def _apply_mask_ablation(self, lane_mask: np.ndarray) -> np.ndarray:
        if self.mask_mode == "zero":
            return np.zeros_like(lane_mask, dtype=np.uint8)
        if self.mask_scale >= 0.999999:
            return lane_mask
        if self.mask_scale <= 0.0:
            return np.zeros_like(lane_mask, dtype=np.uint8)
        return np.clip(
            np.round(lane_mask.astype(np.float32) * self.mask_scale), 0, 255
        ).astype(np.uint8)

    def _compute_diff_rgb(self, rgb_dr: np.ndarray, rgb_clean: np.ndarray) -> np.ndarray:
        if self.diff_mode == "off":
            return np.zeros_like(rgb_dr, dtype=np.int16)
        if self.diff_mode == "dr_rgb":
            if self.prev_rgb_dr is None:
                diff = np.zeros_like(rgb_dr, dtype=np.int16)
            else:
                diff = np.clip(
                    rgb_dr.astype(np.float32) - self.prev_rgb_dr.astype(np.float32),
                    -255, 255,
                ).astype(np.int16)
            self.prev_rgb_dr = rgb_dr.copy()
            return diff
        if self.diff_mode == "clean_rgb":
            if self.prev_rgb_clean is None:
                diff = np.zeros_like(rgb_clean, dtype=np.int16)
            else:
                diff = np.clip(
                    rgb_clean.astype(np.float32) - self.prev_rgb_clean.astype(np.float32),
                    -255, 255,
                ).astype(np.int16)
            self.prev_rgb_clean = rgb_clean.copy()
            return diff
        # clean_gray
        gray = cv2.cvtColor(rgb_clean, cv2.COLOR_RGB2GRAY)
        if self.prev_gray_clean is None:
            diff_gray = np.zeros_like(gray, dtype=np.int16)
        else:
            diff_gray = np.clip(
                gray.astype(np.float32) - self.prev_gray_clean.astype(np.float32),
                -255, 255,
            ).astype(np.int16)
        self.prev_gray_clean = gray.copy()
        return np.repeat(diff_gray[:, :, np.newaxis], 3, axis=2)

    def _reset_diff_states(self):
        self.prev_rgb_dr    = None
        self.prev_rgb_clean = None
        self.prev_gray_clean = None

    def _build_observation(self, obs: np.ndarray, is_reset: bool = False) -> np.ndarray:
        resized = cv2.resize(
            obs, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_LINEAR
        )
        rgb, lane_mask, edges = self.enhancer.enhance(resized, apply_dr=True)
        lane_mask = self._apply_mask_ablation(lane_mask)
        if is_reset:
            self._reset_diff_states()
        diff_rgb = self._compute_diff_rgb(rgb, resized)
        self._coverage_history.append(self._compute_coverage(lane_mask))

        return np.dstack([
            rgb.astype(np.int16),
            diff_rgb,
            lane_mask.astype(np.int16)[:, :, np.newaxis],
            edges.astype(np.int16)[:, :, np.newaxis],
        ])

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._build_observation(obs, is_reset=True)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return self._build_observation(obs, is_reset=False)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        info["domain"] = self.domain
        if len(self._coverage_history) > 0:
            info["mask_coverage"] = self._coverage_history[-1]
        return obs, reward, done, info


# ============================================================
# V9 GT reset 扰动包装器
# ============================================================
class GTResetPerturbWrapper(gym.Wrapper):
    """
    GT 场景 reset 后执行少量随机动作，防止策略"记脚本"固定起点时序。
    放在 RewardWrapper 之前，避免暖启动步数污染奖励统计。
    """

    def __init__(
        self,
        env,
        enabled: bool = False,
        steps_lo: int = 0,
        steps_hi: int = 0,
        steer_abs_max: float = 0.25,
        throttle_lo: float = 0.08,
        throttle_hi: float = 0.16,
        zero_steer_prob: float = 0.35,
    ):
        super().__init__(env)
        self.enabled      = bool(enabled)
        self.steps_lo     = max(0, int(steps_lo))
        self.steps_hi     = max(self.steps_lo, int(steps_hi))
        self.steer_abs_max = float(max(0.0, steer_abs_max))
        self.throttle_lo  = float(max(0.0, throttle_lo))
        self.throttle_hi  = float(max(self.throttle_lo, throttle_hi))
        self.zero_steer_prob = float(np.clip(zero_steer_prob, 0.0, 1.0))
        self._last_n = 0

        if self.enabled and self.steps_hi > 0:
            print(
                f"🎲 GTResetPerturb: {self.steps_lo}-{self.steps_hi} 步, "
                f"steer±{self.steer_abs_max:.2f}, throttle[{self.throttle_lo:.2f},{self.throttle_hi:.2f}]"
            )

    def _sample_action(self) -> np.ndarray:
        steer = float(np.random.uniform(-self.steer_abs_max, self.steer_abs_max))
        if np.random.rand() < self.zero_steer_prob:
            steer *= float(np.random.uniform(0.0, 0.3))
        throttle = float(np.random.uniform(self.throttle_lo, self.throttle_hi))
        return np.array([steer, throttle], dtype=np.float32)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if not self.enabled or self.steps_hi <= 0:
            self._last_n = 0
            return obs
        n_steps = int(np.random.randint(self.steps_lo, self.steps_hi + 1))
        self._last_n = n_steps
        for _ in range(n_steps):
            action = self._sample_action()
            obs, _, done, _ = self.env.step(action)
            if done:
                obs = self.env.reset(**kwargs)
                break
        return obs


# ============================================================
# V12 轻量 RGB 预处理包装器（替代 V9YellowLaneWrapper 8 通道链路）
# ============================================================
class RGBResizeWrapper(gym.ObservationWrapper):
    """
    V12 专用：将 DonkeyEnv 原始 uint8 HWC 图像 (H, W, 3)
    resize → (obs_size, obs_size, 3) → CHW float32 归一化到 [0, 1]。
    循序渐进数据增强：从 augment_start_step 开始，逐步引入不同增强方法，强度线性增加。
    """

    def __init__(
        self,
        env,
        obs_size: int = 96,
        augment: bool = False,
        max_steps: int = 500000,
        augment_start_step: int = 0,
    ):
        super().__init__(env)
        self.obs_size = int(obs_size)
        self.augment = bool(augment)
        self.max_steps = int(max_steps)
        self.augment_start_step = int(max(0, augment_start_step))
        self._step = 0

        # 增强方法启用时间表（按“增强阶段进度”百分比）
        self.augment_schedule = {
            0.00: "brightness_contrast",  # 亮度/对比度
            0.08: "rotation",              # 旋转
            0.16: "scale_crop",            # 缩放
            0.24: "translation",           # 平移
            0.35: "hsv_jitter",            # 色彩抖动
            0.45: "blur",                  # 模糊
            0.58: "noise_gamma",           # 噪声和 Gamma
            0.72: "occlusion",             # 遮挡
        }
        self._enabled_augments = set()
        self._last_log_step = 0

        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(3, self.obs_size, self.obs_size),
            dtype=np.float32,
        )
        print(f"✅ RGBResizeWrapper: raw→(3,{self.obs_size},{self.obs_size}) float32 [0,1]"
              f"{f', 循序渐进增强(start={self.augment_start_step})' if augment else ''}")

    def _get_progress_ratio(self) -> float:
        """当前增强阶段进度比例 [0, 1]。"""
        if self.max_steps <= self.augment_start_step:
            return 1.0 if self._step >= self.augment_start_step else 0.0
        if self._step < self.augment_start_step:
            return 0.0
        ratio = float(self._step - self.augment_start_step) / float(self.max_steps - self.augment_start_step)
        return float(np.clip(ratio, 0.0, 1.0))

    def _update_enabled_augments(self):
        """根据当前进度更新启用的增强方法"""
        progress = self._get_progress_ratio()
        for threshold, augment_name in self.augment_schedule.items():
            if progress >= threshold:
                if augment_name not in self._enabled_augments:
                    self._enabled_augments.add(augment_name)
                    log_step = int(self.augment_start_step + threshold * max(1, (self.max_steps - self.augment_start_step)))
                    if self._step - self._last_log_step > 1000:  # 每 1000 步最多打 1 次
                        print(f"  🎯 {log_step:7d} 步 (augment阶段 {100*threshold:5.1f}%): 启用 {augment_name}")
                        self._last_log_step = self._step

    def _get_augment_strength(self, augment_name: str) -> float:
        """获取增强方法的当前强度 [0, 1]，从启用后线性增加到 1.0"""
        threshold = None
        for t, name in self.augment_schedule.items():
            if name == augment_name:
                threshold = t
                break
        if threshold is None or augment_name not in self._enabled_augments:
            return 0.0
        progress = self._get_progress_ratio()
        # 从启用点线性增加到最后 (1.0)
        if progress >= 1.0:
            return 1.0
        if progress < threshold:
            return 0.0
        # 线性插值：从 threshold 到 1.0，强度从 0 到 1
        return (progress - threshold) / (1.0 - threshold)

    def _augment(self, rgb: np.ndarray) -> np.ndarray:
        """
        循序渐进的数据增强，强度随进度线性增加。
        不含翻转，包含：旋转、缩放、平移、色彩、模糊、噪声、Gamma、遮挡。
        """
        self._update_enabled_augments()
        img = rgb.astype(np.float32)
        h, w = img.shape[:2]

        # 1. 亮度/对比度（基础，强度最早达到 1.0）
        if "brightness_contrast" in self._enabled_augments:
            strength = self._get_augment_strength("brightness_contrast")
            if strength > 0 and np.random.random() < strength * 0.5:
                delta = np.random.uniform(-30, 30) * strength
                img = np.clip(img + delta, 0, 255)
            if strength > 0 and np.random.random() < strength * 0.5:
                factor = np.random.uniform(0.85, 1.15) if strength >= 1.0 else np.random.uniform(1.0 - 0.15*strength, 1.0 + 0.15*strength)
                img = np.clip(img * factor, 0, 255)

        # 2. 旋转
        if "rotation" in self._enabled_augments:
            strength = self._get_augment_strength("rotation")
            if strength > 0 and np.random.random() < strength * 0.6:
                max_angle = 15.0 * strength
                angle = np.random.uniform(-max_angle, max_angle)
                center = (w / 2, h / 2)
                mat = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 3. 缩放 + 裁剪
        if "scale_crop" in self._enabled_augments:
            strength = self._get_augment_strength("scale_crop")
            if strength > 0 and np.random.random() < strength * 0.6:
                scale_range = 0.3 * strength  # 从 0% 到 30%
                scale = np.random.uniform(1.0 - scale_range, 1.0 + scale_range)
                new_h, new_w = int(h * scale), int(w * scale)
                img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                if scale < 1.0:  # 缩小了，需要填充到原尺寸
                    pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
                    img_pad = np.full((h, w, 3), 128, dtype=np.float32)
                    img_pad[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = img_scaled
                    img = img_pad
                else:  # 放大了，需要裁剪中心部分
                    pad_h, pad_w = (new_h - h) // 2, (new_w - w) // 2
                    img = img_scaled[pad_h:pad_h+h, pad_w:pad_w+w]

        # 4. 平移
        if "translation" in self._enabled_augments:
            strength = self._get_augment_strength("translation")
            if strength > 0 and np.random.random() < strength * 0.5:
                max_trans = 0.05 * strength  # 从 0% 到 5%
                dx = int(np.random.uniform(-w*max_trans, w*max_trans))
                dy = int(np.random.uniform(-h*max_trans, h*max_trans))
                mat = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # 5. HSV 色彩抖动
        if "hsv_jitter" in self._enabled_augments:
            strength = self._get_augment_strength("hsv_jitter")
            if strength > 0 and np.random.random() < strength * 0.6:
                img_hsv = cv2.cvtColor(np.uint8(np.clip(img, 0, 255)), cv2.COLOR_RGB2HSV).astype(np.float32)
                h_shift = np.random.uniform(-20, 20) * strength
                s_scale = np.random.uniform(1.0 - 0.2*strength, 1.0 + 0.2*strength)
                v_scale = np.random.uniform(1.0 - 0.2*strength, 1.0 + 0.2*strength)
                img_hsv[:, :, 0] = np.clip(img_hsv[:, :, 0] + h_shift, 0, 180)
                img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * s_scale, 0, 255)
                img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * v_scale, 0, 255)
                img = cv2.cvtColor(np.uint8(img_hsv), cv2.COLOR_HSV2RGB).astype(np.float32)

        # 6. 模糊（高斯或运动）
        if "blur" in self._enabled_augments:
            strength = self._get_augment_strength("blur")
            if strength > 0 and np.random.random() < strength * 0.4:
                max_sigma = 1.5 * strength
                sigma = np.random.uniform(0.5, max_sigma)
                if sigma > 0.3:
                    kernel_size = int(2 * np.ceil(2 * sigma)) + 1
                    kernel_size = min(kernel_size, 7)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

        # 7. 噪声（高斯 + 椒盐）和 Gamma
        if "noise_gamma" in self._enabled_augments:
            strength = self._get_augment_strength("noise_gamma")
            # 高斯噪声
            if strength > 0 and np.random.random() < strength * 0.5:
                max_sigma = 20.0 * strength
                sigma = np.random.uniform(5, max_sigma)
                img = np.clip(img + np.random.normal(0, sigma, img.shape), 0, 255)
            # 椒盐噪声
            if strength > 0 and np.random.random() < strength * 0.35:
                noise_ratio = 0.03 * strength
                noise_pixels = int(h * w * noise_ratio)
                if noise_pixels > 0:
                    coords = np.random.choice(h * w, min(noise_pixels, h*w), replace=False)
                    for coord in coords:
                        y, x = divmod(coord, w)
                        color = 0 if np.random.random() < 0.5 else 255
                        img[y, x] = color
            # Gamma 校正
            if strength > 0 and np.random.random() < strength * 0.4:
                gamma_range = 0.5 * strength  # 从 [1.0, 1.0] 到 [0.8, 1.2]
                gamma = np.random.uniform(1.0 - gamma_range/2, 1.0 + gamma_range/2)
                img = np.clip(np.power(img / 255.0, 1.0 / gamma) * 255.0, 0, 255)

        # 8. 随机遮挡
        if "occlusion" in self._enabled_augments:
            strength = self._get_augment_strength("occlusion")
            if strength > 0 and np.random.random() < strength * 0.25:
                num_occlude = int(1 + strength * 2)  # 1-3 个遮挡块
                for _ in range(num_occlude):
                    patch_h = int(np.random.uniform(0.05, 0.15) * h)
                    patch_w = int(np.random.uniform(0.05, 0.15) * w)
                    y_start = int(np.random.uniform(0, max(1, h - patch_h)))
                    x_start = int(np.random.uniform(0, max(1, w - patch_w)))
                    color = np.random.uniform(50, 200)
                    img[y_start:y_start+patch_h, x_start:x_start+patch_w] = color

        return np.uint8(np.clip(img, 0, 255))

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._step += 1
        img = np.asarray(obs, dtype=np.uint8)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]  # 截取 RGB
        img = cv2.resize(img, (self.obs_size, self.obs_size), interpolation=cv2.INTER_LINEAR)
        if self.augment and self._step >= self.augment_start_step:
            img = self._augment(img)
        # HWC → CHW, normalize to [0,1]
        return img.transpose(2, 0, 1).astype(np.float32) / 255.0

    def reset(self, **kwargs):
        return super().reset(**kwargs)


# ============================================================
# V13 规范化语义包装器（实现在 obv.py，此处仅重导出）
# ============================================================
from .obv import CanonicalSemanticWrapper, _CANONICAL_TGT_STATS  # noqa: F401, E402
