#!/usr/bin/env python3
"""
🔧 鲁棒车道线检测器 - V9.4 三域对齐版本
基于 V9.3c，修复三大风险：

V9.4 修复（基于 V9.3c）：
  Fix A — Coverage Dropout 消除域泄漏：
    - 训练时对 mask 逐行随机置零 p~U(0.15, 0.45)
    - 三域 coverage 分布从 [0.37, 0.56] 模糊到 [0.2, 0.5]
    - 推理时不 dropout（apply_dr=False）
  Fix B — DR 与 mask/edges 解耦：
    - mask/edges 始终从原始 RGB 计算（语义干净）
    - DR 只影响返回的 RGB 通道 0-2
    - HSV 扰动从 H±8 缩小到 H±4
  Fix C — 归一化/CNN 改进（在 ppo_waveshare_v8.py 中）

V9.3c 保留：
  三域检测策略（per-domain edge detection）：
  - WS:   HSV 黄色检测 H15-40, S≥60, V≥80 + road_mask
  - Real: HSV 黄色检测 H12-45, S≥35, V≥70 + road_mask
  - GT:   几何边界检测 — road_mask 左右边界 ±8px，V≥120

  行质心编码 (Row Centroid Encoding)：
  1. per-domain 边沿线检测
  2. CLOSE-only 形态学清理
  3. 逐行质心，固定 7px 宽标记
  4. 跨度 >15% 丢弃 + 聚类回退
"""

import numpy as np
import cv2
import random
from typing import Tuple


class RobustLaneDetector:
    """
    鲁棒车道线检测器 V9.3c - per-domain 边沿线检测 + 行质心编码
    
    解决三域（Real/WS/GT）边沿线语义不一致问题。
    
    原理：
    1. 路面检测：低饱和度灰色区域 → 路面掩码
    2. ★per-domain 边沿线检测：
       - WS: 黄色 HSV（V9.2 参数不变）
       - Real: 放宽黄色 HSV（H12-45, S≥35, V≥70 捕捉细线）
       - GT: 几何边界检测（road_mask 边界 ±8px 内找亮像素，99.1% 精度）
    3. CLOSE-only 形态学（无 OPEN，保留 Real 细线）
    4. 行质心编码: 固定 7px 宽，跨度 >15% 丢弃 + 聚类回退
    5. 路面边缘：bilateral+Canny+road_mask
    """
    
    def __init__(self, 
                 # ★ V9.3c: 域参数
                 domain: str = 'ws',         # 'ws' / 'real' / 'gt'
                 # 路面检测参数
                 road_s_max: int = 50,       # 路面饱和度上限（灰色=低饱和度）
                 road_v_min: int = 70,       # 路面亮度下限
                 road_v_max: int = 255,      # 路面亮度上限
                 # 黄线检测参数（WS/Real 使用）
                 yellow_h_range: Tuple[int, int] = (15, 40),
                 yellow_s_min: int = 60,     # WS: 60, Real: 35
                 yellow_v_min: int = 80,     # WS: 80, Real: 70
                 # GT 几何检测参数
                 gt_search_radius: int = 8,  # road boundary 搜索半径
                 gt_min_bright: int = 120,   # GT 边沿线最小亮度
                 # 行质心编码参数（核心方案）
                 centroid_band_width: int = 7,    # ★ 固定输出线宽（像素），所有域一致
                 centroid_min_pixels: int = 1,    # ★ V9.3c: 1（保留细线，V9.2 用 3）
                 centroid_max_width_ratio: float = 0.15,  # ★ 跨度上限（占图像宽度）
                 # 白线检测参数（默认禁用）
                 white_s_max: int = 40,
                 white_v_min: int = 180,
                 enable_white_detection: bool = False,
                 # 边缘检测参数
                 canny_low: int = 60,
                 canny_high: int = 180,
                 use_bilateral_filter: bool = True,
                 bilateral_d: int = 5,
                 bilateral_sigma_color: int = 50,
                 bilateral_sigma_space: int = 50,
                 # 掩码控制
                 use_road_mask_for_edges: bool = True,
                 use_road_mask_for_yellow: bool = True,
                 # ★ V9.4: Coverage dropout — 消除三域 coverage 差异泄漏 domain
                 coverage_dropout_range: Tuple[float, float] = (0.15, 0.45)):
        
        self.domain = domain
        self.coverage_dropout_range = coverage_dropout_range
        
        self.road_s_max = road_s_max
        self.road_v_min = road_v_min
        self.road_v_max = road_v_max
        
        self.yellow_h_range = yellow_h_range
        self.yellow_s_min = yellow_s_min
        self.yellow_v_min = yellow_v_min
        
        # GT 几何检测参数
        self.gt_search_radius = gt_search_radius
        self.gt_min_bright = gt_min_bright
        
        # 行质心编码参数
        self.centroid_band_width = centroid_band_width
        self.centroid_min_pixels = centroid_min_pixels
        self.centroid_max_width_ratio = centroid_max_width_ratio
        
        self.white_s_max = white_s_max
        self.white_v_min = white_v_min
        self.enable_white_detection = enable_white_detection
        
        self.canny_low = canny_low
        self.canny_high = canny_high
        
        self.use_bilateral_filter = use_bilateral_filter
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        self.use_road_mask_for_edges = use_road_mask_for_edges
        self.use_road_mask_for_yellow = use_road_mask_for_yellow
        
        # 形态学核
        self._close_kernel = np.ones((5, 5), np.uint8)
        self._dilate_kernel = np.ones((7, 7), np.uint8)
        self._line_kernel = np.ones((3, 3), np.uint8)
        
        domain_desc = {'ws': 'WS(yellow H15-40,S60+)', 'real': 'Real(yellow H12-45,S35+)', 'gt': 'GT(geometric boundary)'}
        print(f"🔧 鲁棒车道线检测器 V9.3c 初始化 domain={domain}")
        print(f"   检测方式: {domain_desc.get(domain, domain)}")
        print(f"   行质心: band={centroid_band_width}px, min_px={centroid_min_pixels}, max_width={centroid_max_width_ratio:.0%}")
        print(f"   形态学: CLOSE-only（无OPEN，保留细线）")
        print(f"   边缘: bilateral({bilateral_d},{bilateral_sigma_color},{bilateral_sigma_space}), Canny({canny_low},{canny_high})")
    
    def detect_road(self, hsv: np.ndarray) -> np.ndarray:
        """
        检测路面区域
        
        路面特征：低饱和度 + 中高亮度（灰色/白色沥青）
        这在 waveshare（灰色桌面）和 generated_track（灰色柏油路）上都成立。
        
        Returns:
            road_mask: uint8 (H, W)，255=路面
        """
        # 低饱和度 + 适当亮度 = 路面
        s_mask = hsv[:, :, 1] < self.road_s_max
        v_mask = (hsv[:, :, 2] >= self.road_v_min) & (hsv[:, :, 2] <= self.road_v_max)
        road_mask = (s_mask & v_mask).astype(np.uint8) * 255
        
        # 形态学清理：闭运算填洞（V9.3c: 无OPEN，保留细线特征）
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, self._close_kernel)
        
        # 膨胀路面掩码（让边缘和车道线被包含）
        road_expanded = cv2.dilate(road_mask, self._dilate_kernel, iterations=2)
        
        return road_mask, road_expanded
    
    def detect_lane_lines(self, rgb: np.ndarray, hsv: np.ndarray,
                          road_mask: np.ndarray,
                          road_expanded: np.ndarray) -> np.ndarray:
        """
        V9.3c 边沿线检测 - per-domain + 行质心编码
        
        三域检测策略：
        - WS:   HSV 黄色 H15-40, S≥60, V≥80 + road_mask
        - Real: HSV 黄色 H12-45, S≥35, V≥70 + road_mask（放宽，捕捉细线）
        - GT:   几何边界检测 — road_mask 左右边界 ±8px 内找亮像素
                排除绿草(H35-85,S>100)，精度 99.1%
        
        Returns:
            lane_mask: uint8 (H, W)，255=边沿线质心标记
        """
        h, w = hsv.shape[:2]
        
        if self.domain == 'gt':
            # === GT: 几何边界检测 ===
            lane_raw = self._detect_gt_geometric(hsv, road_mask, h, w)
        else:
            # === WS / Real: HSV 黄色检测 ===
            yellow_mask = cv2.inRange(
                hsv,
                np.array([self.yellow_h_range[0], self.yellow_s_min, self.yellow_v_min]),
                np.array([self.yellow_h_range[1], 255, 255])
            )
            lane_raw = yellow_mask
            
            # 路面掩码过滤
            if self.use_road_mask_for_yellow:
                lane_raw = cv2.bitwise_and(lane_raw, road_expanded)
        
        # === 形态学清理：CLOSE-only（V9.3c: 无OPEN，保留Real细线） ===
        lane_raw = cv2.morphologyEx(lane_raw, cv2.MORPH_CLOSE, self._line_kernel)
        
        # === ★ 行质心编码：将任意宽度转为固定宽度 ===
        result = self._row_centroid_encode(lane_raw, h, w)
        
        return result
    
    def apply_coverage_dropout(self, lane_mask: np.ndarray) -> np.ndarray:
        """
        ★ V9.4 Coverage Dropout: 按行随机置零，模糊三域 coverage 差异。
        
        问题：GT coverage≈0.37, WS/Real≈0.56，策略可通过 mask 稀疏度推断 domain。
        方案：训练时对每行以 p~U(dropout_range) 概率随机置零，
              使三域 coverage 分布重叠到 [0.2, 0.5] 范围。
        推理时（apply_dr=False）不调用此函数。
        """
        if self.coverage_dropout_range is None:
            return lane_mask
        
        lo, hi = self.coverage_dropout_range
        h = lane_mask.shape[0]
        # 每行独立决定是否 dropout
        drop_prob = np.random.uniform(lo, hi)
        drop_mask = np.random.random(h) < drop_prob
        result = lane_mask.copy()
        result[drop_mask] = 0
        return result
    
    def _detect_gt_geometric(self, hsv: np.ndarray, road_mask: np.ndarray,
                             h: int, w: int) -> np.ndarray:
        """
        GT 几何边界检测：从 road_mask 定位左右边界，
        在边界 ±search_radius px 窗口内找亮像素(V≥min_bright)。
        排除绿草(H35-85, S>100)。

        V9.5 修复：改为检测所有 road_mask 过渡点（内部边界），
        不再只取 road_cols[0]/[-1]，解决 generated_track 路面铺满
        整幅图像导致两个端点均落在图像边缘被跳过（coverage≈0.02）的问题。
        """
        result = np.zeros((h, w), dtype=np.uint8)
        v_chan = hsv[:, :, 2].astype(np.float32)
        sr = self.gt_search_radius
        mb = self.gt_min_bright

        for row in range(h):
            road_row = road_mask[row] > 0
            if road_row.sum() < 10:
                continue

            # ── 找所有过渡点（非路面→路面 和 路面→非路面）──────────────
            diffs = np.diff(road_row.astype(np.int8))
            # non-road→road: diff=+1, 边界列 = transition_col
            # road→non-road: diff=-1, 边界列 = transition_col
            boundaries = list(np.where(diffs != 0)[0] + 1)

            # 若路面从 col=0 开始（无左过渡），加入左边缘附近
            if road_row[0]:
                boundaries.append(0)
            # 若路面到 col=w-1 结束（无右过渡），加入右边缘附近
            if road_row[-1]:
                boundaries.append(w - 1)

            if not boundaries:
                continue

            for boundary in boundaries:
                left_s = max(0, boundary - sr)
                right_s = min(w, boundary + sr + 1)

                wv = v_chan[row, left_s:right_s]
                wh = hsv[row, left_s:right_s, 0]
                ws = hsv[row, left_s:right_s, 1]

                # 亮像素 且 非绿草
                bright_mask = (wv >= mb) & ~((wh >= 35) & (wh <= 85) & (ws > 100))

                for c in np.where(bright_mask)[0]:
                    result[row, left_s + c] = 255

        return result
    
    def _row_centroid_encode(self, lane_raw: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        行质心编码：逐行计算像素质心，画固定宽度标记。
        超宽行尝试聚类回退，取最窄簇。
        """
        result = np.zeros((h, w), dtype=np.uint8)
        max_span = int(w * self.centroid_max_width_ratio)
        half = self.centroid_band_width // 2
        min_px = self.centroid_min_pixels
        
        for row in range(h):
            cols = np.where(lane_raw[row] > 0)[0]
            if len(cols) < min_px:
                continue
            
            span = cols[-1] - cols[0] + 1
            if span > max_span:
                # 聚类回退：尝试分割为多个簇，取最窄的
                gaps = np.where(np.diff(cols) > 5)[0]
                if len(gaps) > 0:
                    clusters = np.split(cols, gaps + 1)
                    best = min(clusters,
                               key=lambda c: (c[-1] - c[0] + 1) if len(c) >= min_px else 9999)
                    if len(best) >= min_px and (best[-1] - best[0] + 1) <= max_span:
                        cx = int(np.mean(best))
                        result[row, max(0, cx - half):min(w, cx + half + 1)] = 255
                continue
            
            cx = int(np.mean(cols))
            result[row, max(0, cx - half):min(w, cx + half + 1)] = 255
        
        return result
    
    def detect_road_edges(self, rgb: np.ndarray, road_mask: np.ndarray,
                          road_expanded: np.ndarray) -> np.ndarray:
        """
        路面结构边缘检测（V9.2: 轻度双边滤波 + 中等 Canny + 路面掩码）
        
        V9.2 改进：
        - bilateral(5,50,50)：比V9.1(9,75,75)更轻，保留Real细节
        - Canny(60,180)：比V9.1(100,250)更低，捕获Real较弱边缘
        - 路面掩码：排除路面外区域的边缘
        
        三域验证: max ratio=1.6x（V8原: 3.9x）
        
        Returns:
            edges: uint8 (H, W)，255=边缘
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # ★ 双边滤波：保留边缘结构，平滑纹理细节（草地、路面粗糙度）
        if self.use_bilateral_filter:
            gray = cv2.bilateralFilter(
                gray, self.bilateral_d, 
                self.bilateral_sigma_color, self.bilateral_sigma_space
            )
        
        # CLAHE 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)
        
        # ★ 高阈值 Canny（只保留强边缘）
        edges = cv2.Canny(gray_enhanced, self.canny_low, self.canny_high)
        
        # ★ 路面掩码过滤（排除草地纹理等路面外边缘）
        if self.use_road_mask_for_edges:
            edges = cv2.bitwise_and(edges, road_expanded)
        
        return edges
    
    def detect(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整检测管线
        
        Args:
            rgb: RGB uint8 (H, W, 3)
        
        Returns:
            lane_mask: uint8 (H, W)，255=车道线
            edges: uint8 (H, W)，255=路面边缘
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        
        # 1. 路面检测
        road_mask, road_expanded = self.detect_road(hsv)
        
        # 2. 边沿线检测（per-domain + 行质心编码）
        lane_mask = self.detect_lane_lines(rgb, hsv, road_mask, road_expanded)
        
        # 3. 路面边缘检测
        edges = self.detect_road_edges(rgb, road_mask, road_expanded)
        
        return lane_mask, edges
    
    def detect_with_debug(self, rgb: np.ndarray) -> dict:
        """
        带调试信息的完整检测（用于可视化和调参）
        
        Returns:
            dict: 包含所有中间结果
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        road_mask, road_expanded = self.detect_road(hsv)
        
        # 原始黄线候选（不含路面过滤）
        yellow_raw = cv2.inRange(
            hsv,
            np.array([self.yellow_h_range[0], self.yellow_s_min, self.yellow_v_min]),
            np.array([self.yellow_h_range[1], 255, 255])
        )
        
        # V8原始方法（用于对比）
        yellow_v8 = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([40, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        yellow_v8 = cv2.morphologyEx(yellow_v8, cv2.MORPH_CLOSE, kernel)
        yellow_v8 = cv2.morphologyEx(yellow_v8, cv2.MORPH_OPEN, kernel)
        
        lane_mask = self.detect_lane_lines(rgb, hsv, road_mask, road_expanded)
        edges = self.detect_road_edges(rgb, road_mask, road_expanded)
        
        # 全图 Canny（V8方法，用于对比）
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        edges_v8 = cv2.Canny(clahe.apply(gray), 40, 120)
        
        return {
            "rgb": rgb,
            "hsv": hsv,
            "road_mask": road_mask,
            "road_expanded": road_expanded,
            "yellow_raw": yellow_raw,
            "yellow_v8": yellow_v8,
            "lane_mask": lane_mask,
            "edges": edges,
            "edges_v8": edges_v8,
            "stats": {
                "road_ratio": road_mask.sum() / (255.0 * road_mask.size),
                "yellow_v8_ratio": yellow_v8.sum() / (255.0 * yellow_v8.size),
                "lane_mask_ratio": lane_mask.sum() / (255.0 * lane_mask.size),
                "edges_v8_density": edges_v8.sum() / (255.0 * edges_v8.size),
                "edges_new_density": edges.sum() / (255.0 * edges.size),
            }
        }


class RobustYellowLaneEnhancer:
    """
    V9.3c 鲁棒增强器 - per-domain 边沿线检测
    
    核心改动（相对 V9.2）：
    - per-domain 边沿线检测：WS/Real=黄色, GT=几何边界
    - CLOSE-only 形态学（保留 Real 细线）
    - min_pixels=1（V9.2 用 3，丢细线）
    - 聚类回退（超宽行取最窄簇而非丢弃）
    - DR 逻辑只作用于 RGB
    """
    
    def __init__(self, enable_dr=False, dr_prob=0.6, detector: RobustLaneDetector = None):
        self.enable_dr = enable_dr
        self.dr_prob = dr_prob
        self.detector = detector or RobustLaneDetector(domain='ws')
        
        print(f"🔧 V9.3c 鲁棒增强器初始化 domain={self.detector.domain}")
        print(f"   RGB-only DR: {'启用' if enable_dr else '禁用'} (prob={dr_prob})")
    
    def _dr_brightness_contrast(self, rgb):
        rgb = rgb.astype(np.float32)
        if random.random() < 0.5:
            b = random.uniform(-30, 30)
            rgb = np.clip(rgb + b, 0, 255)
        if random.random() < 0.5:
            c = random.uniform(0.75, 1.25)
            rgb = np.clip(rgb * c, 0, 255)
        return rgb.astype(np.uint8)
    
    def _dr_blur(self, rgb):
        if random.random() < 0.3:
            k = random.choice([1, 3, 5])
            if k > 1:
                rgb = cv2.GaussianBlur(rgb, (k, k), 0)
        return rgb
    
    def _dr_noise(self, rgb):
        if random.random() < 0.25:
            sigma = random.uniform(3, 12)
            noise = np.random.normal(0, sigma, rgb.shape)
            rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return rgb
    
    def _dr_hsv(self, rgb):
        # ★ V9.4: H±4（从±8缩小），降低 HSV 扰动对 mask 的间接影响
        if random.random() < 0.4:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.uniform(-4, 4), 0, 179)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.85, 1.15), 0, 255)
            hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.uniform(-15, 15), 0, 255)
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return rgb
    
    def _apply_dr(self, rgb):
        if self.enable_dr and random.random() < self.dr_prob:
            rgb = self._dr_brightness_contrast(rgb)
            rgb = self._dr_hsv(rgb)
            rgb = self._dr_blur(rgb)
            rgb = self._dr_noise(rgb)
        return rgb
    
    def enhance(self, img, apply_dr=True):
        """
        鲁棒黄线增强处理（V9.4 修复）
        
        ★ V9.4 关键修复：mask/edges 始终从原始 RGB 计算，DR 只影响返回的 RGB 通道。
        
        之前：DR→RGB'，然后从 RGB' 计算 mask/edges → mask 随 DR 闪烁
        现在：从原始 RGB 计算 mask/edges（语义干净），DR 只改通道 0-2
        
        与 V8 接口完全兼容：
        输入: RGB (H, W, 3)
        输出: (rgb, lane_mask, edges)
        """
        rgb_clean = img.copy()
        
        # ① mask/edges 始终从原始 RGB 计算（语义干净，不受 DR 扰动）
        lane_mask, edges = self.detector.detect(rgb_clean)
        
        # ② DR 只改 RGB 通道（返回给网络的通道 0-2）
        if apply_dr:
            rgb_out = self._apply_dr(rgb_clean)
            # ③ Coverage dropout：训练时模糊三域 coverage 差异
            lane_mask = self.detector.apply_coverage_dropout(lane_mask)
        else:
            rgb_out = rgb_clean
        
        return rgb_out, lane_mask, edges


# ============================================================
# 离线验证工具
# ============================================================
def validate_on_samples(ws_dir: str = "data/scene_samples/waveshare/processed",
                        gt_dir: str = "data/scene_samples/generated_track/processed",
                        output_dir: str = "data/lane_detection_validation",
                        max_images: int = 50):
    """
    在已有采样图像上验证新检测器 vs V8旧检测器的效果
    
    用法:
        python -m module.robust_lane_detector --validate
    """
    import os
    from pathlib import Path
    
    os.makedirs(output_dir, exist_ok=True)
    
    detector = RobustLaneDetector()
    
    results = {"waveshare": [], "generated_track": []}
    
    for scene, img_dir in [("waveshare", ws_dir), ("generated_track", gt_dir)]:
        images = sorted(Path(img_dir).glob("*.png"))[:max_images]
        if not images:
            print(f"⚠️  {img_dir} 中没有图像")
            continue
        
        print(f"\n📊 验证 {scene}: {len(images)} 张图像")
        
        for i, img_path in enumerate(images):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            debug = detector.detect_with_debug(img_rgb)
            results[scene].append(debug["stats"])
            
            # 每10张保存一张对比图
            if i % 10 == 0:
                _save_comparison(debug, scene, i, output_dir)
        
        # 统计
        stats = results[scene]
        print(f"   V8 黄线 ratio: {np.mean([s['yellow_v8_ratio'] for s in stats])*100:.2f}%")
        print(f"   V9 车道线 ratio: {np.mean([s['lane_mask_ratio'] for s in stats])*100:.2f}%")
        print(f"   V8 边缘密度: {np.mean([s['edges_v8_density'] for s in stats])*100:.2f}%")
        print(f"   V9 边缘密度: {np.mean([s['edges_new_density'] for s in stats])*100:.2f}%")
        print(f"   路面占比: {np.mean([s['road_ratio'] for s in stats])*100:.2f}%")
    
    # 域差异对比
    if results["waveshare"] and results["generated_track"]:
        ws = results["waveshare"]
        gt = results["generated_track"]
        
        print("\n" + "="*60)
        print("📏 域差异对比 (V8 vs V9)")
        print("="*60)
        
        ws_v8_yr = np.mean([s['yellow_v8_ratio'] for s in ws])
        gt_v8_yr = np.mean([s['yellow_v8_ratio'] for s in gt])
        ws_v9_yr = np.mean([s['lane_mask_ratio'] for s in ws])
        gt_v9_yr = np.mean([s['lane_mask_ratio'] for s in gt])
        
        v8_ratio = max(ws_v8_yr, gt_v8_yr) / max(min(ws_v8_yr, gt_v8_yr), 1e-6)
        v9_ratio = max(ws_v9_yr, gt_v9_yr) / max(min(ws_v9_yr, gt_v9_yr), 1e-6)
        
        print(f"\n  车道线 mask:")
        print(f"   V8: WS={ws_v8_yr*100:.2f}% vs GT={gt_v8_yr*100:.2f}% → {v8_ratio:.1f}x 差异")
        print(f"   V9: WS={ws_v9_yr*100:.2f}% vs GT={gt_v9_yr*100:.2f}% → {v9_ratio:.1f}x 差异")
        print(f"   改善: {v8_ratio:.1f}x → {v9_ratio:.1f}x")
        
        ws_v8_ed = np.mean([s['edges_v8_density'] for s in ws])
        gt_v8_ed = np.mean([s['edges_v8_density'] for s in gt])
        ws_v9_ed = np.mean([s['edges_new_density'] for s in ws])
        gt_v9_ed = np.mean([s['edges_new_density'] for s in gt])
        
        v8_ed_ratio = max(ws_v8_ed, gt_v8_ed) / max(min(ws_v8_ed, gt_v8_ed), 1e-6)
        v9_ed_ratio = max(ws_v9_ed, gt_v9_ed) / max(min(ws_v9_ed, gt_v9_ed), 1e-6)
        
        print(f"\n  边缘密度:")
        print(f"   V8: WS={ws_v8_ed*100:.2f}% vs GT={gt_v8_ed*100:.2f}% → {v8_ed_ratio:.1f}x 差异")
        print(f"   V9: WS={ws_v9_ed*100:.2f}% vs GT={gt_v9_ed*100:.2f}% → {v9_ed_ratio:.1f}x 差异")
        print(f"   改善: {v8_ed_ratio:.1f}x → {v9_ed_ratio:.1f}x")
    
    print(f"\n✅ 验证图保存到: {output_dir}/")
    return results


def _save_comparison(debug: dict, scene: str, idx: int, output_dir: str):
    """保存对比图（V8 vs V9）"""
    import os
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{scene} #{idx} - V8 vs V9 对比", fontsize=14)
    
    # Row 1: V8
    axes[0, 0].imshow(cv2.cvtColor(debug["rgb"], cv2.COLOR_RGB2BGR)[:, :, ::-1])
    axes[0, 0].set_title("原图 (RGB)")
    
    axes[0, 1].imshow(debug["yellow_v8"], cmap='gray')
    axes[0, 1].set_title(f"V8 黄线 mask\nratio={debug['stats']['yellow_v8_ratio']*100:.2f}%")
    
    axes[0, 2].imshow(debug["edges_v8"], cmap='gray')
    axes[0, 2].set_title(f"V8 Canny\ndensity={debug['stats']['edges_v8_density']*100:.2f}%")
    
    axes[0, 3].imshow(debug["road_mask"], cmap='gray')
    axes[0, 3].set_title(f"路面掩码\nratio={debug['stats']['road_ratio']*100:.2f}%")
    
    # Row 2: V9
    axes[1, 0].imshow(debug["road_expanded"], cmap='gray')
    axes[1, 0].set_title("路面扩展区域")
    
    axes[1, 1].imshow(debug["lane_mask"], cmap='gray')
    axes[1, 1].set_title(f"V9 车道线 mask\nratio={debug['stats']['lane_mask_ratio']*100:.2f}%")
    
    axes[1, 2].imshow(debug["edges"], cmap='gray')
    axes[1, 2].set_title(f"V9 路面边缘\ndensity={debug['stats']['edges_new_density']*100:.2f}%")
    
    # V8 vs V9 叠加对比
    overlay = debug["rgb"].copy()
    overlay[debug["lane_mask"] > 0] = [255, 255, 0]  # 黄色叠加
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title("V9 车道线叠加在原图")
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{scene}_{idx:03d}_comparison.png")
    plt.savefig(out_path, dpi=100)
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="鲁棒车道线检测器")
    parser.add_argument("--validate", action="store_true",
                       help="在采样图像上验证检测效果")
    parser.add_argument("--ws-dir", type=str, 
                       default="data/scene_samples/waveshare/processed")
    parser.add_argument("--gt-dir", type=str,
                       default="data/scene_samples/generated_track/processed")
    parser.add_argument("--output-dir", type=str,
                       default="data/lane_detection_validation")
    parser.add_argument("--max-images", type=int, default=50)
    
    args = parser.parse_args()
    
    if args.validate:
        validate_on_samples(args.ws_dir, args.gt_dir, args.output_dir, args.max_images)
    else:
        print("用法: python -m module.robust_lane_detector --validate")
        print("      在采样图像上对比 V8 vs V9 检测效果")
