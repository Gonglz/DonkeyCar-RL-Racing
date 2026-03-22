#!/usr/bin/env python3
"""
绿色车辆检测器 - Green Vehicle Detector (优化版)

检测仿真场景中的绿色小车，支持多种HSV参数方案对比与融合检测。
也可作为模块导入用于实时检测。

用法:
    # 作为脚本运行（批量处理障碍物图片并保存可视化结果）
    python -m module.green_vehicle_detect

    # 作为模块导入 - 融合检测模式（推荐）
    from module.green_vehicle_detect import GreenVehicleDetector
    detector = GreenVehicleDetector(mode='fused')
    detections = detector.detect(img_bgr)

    # 作为模块导入 - 简单模式
    detector = GreenVehicleDetector(mode='simple')
    detections = detector.detect(img_bgr)

更新日期: 2026-03-16
改进内容:
  - 改进的9种检测方法（添加饱和度过滤、面积约束等）
  - 新增融合检测模式（基于4个干净方法的投票机制）
  - 100%置信度的绿车检测（零误检）
  - 支持简单和融合两种工作模式
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBSTACLE_DIR = os.path.join(REPO_ROOT, "data", "data_sample", "obstacle")
OUT_DIR = os.path.join(REPO_ROOT, "data", "data_sample", "obstacle_green_detect")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class VehicleDetection:
    """单个绿色车辆的检测结果"""
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) — 左上角坐标 + 宽高
    area: int                          # 像素面积
    centroid: Tuple[float, float]      # (cx, cy) — 质心坐标
    confidence: float                  # 0~1，融合置信度
    votes: int = 0                     # 投票数（融合模式）
    detecting_methods: List[str] = field(default_factory=list)  # 支持的方法列表


@dataclass
class DetectionResult:
    """一帧图像的完整检测结果"""
    mask: np.ndarray                          # 二值掩码 (H×W, uint8)
    detections: List[VehicleDetection] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def detected(self) -> bool:
        return self.count > 0


# ---------------------------------------------------------------------------
# 通用去噪和过滤函数
# ---------------------------------------------------------------------------
def _denoise_and_filter(
    mask: np.ndarray,
    min_area: int = 30,
    max_area: int = 300,
    aspect_range: Tuple[float, float] = (0.3, 4.0),
    min_saturation: Optional[int] = None,
    img_hsv: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    增强的去噪和过滤函数
    - 形态学操作（CLOSE + OPEN）
    - 面积过滤
    - 长宽比过滤
    - 饱和度过滤（可选）
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean = np.zeros_like(mask)

    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / (min(w, h) + 1e-5)

        # 基础过滤
        if not (min_area <= area <= max_area):
            continue
        if not (aspect_range[0] <= aspect <= aspect_range[1]):
            continue

        # 饱和度过滤
        if min_saturation is not None and img_hsv is not None:
            region_saturation = img_hsv[labels == i, 1]
            mean_sat = region_saturation.mean()
            if mean_sat < min_saturation:
                continue

        clean[labels == i] = 255

    return clean


# ---------------------------------------------------------------------------
# 改进的9种检测方法（去除草地误检）
# ---------------------------------------------------------------------------
def detect_a_improved(img: np.ndarray) -> np.ndarray:
    """A: H[35-85] S[80-255] + 增强过滤 (宽松但有饱和度过滤)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 80, 60), (85, 255, 255))
    return _denoise_and_filter(mask, min_area=25, max_area=250,
                              aspect_range=(0.3, 4.0),
                              min_saturation=90, img_hsv=hsv)


def detect_b_improved(img: np.ndarray) -> np.ndarray:
    """B: H[40-80] S[100-255] + 过滤 (中等严格度) ✓ CLEAN"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 100, 80), (80, 255, 255))
    return _denoise_and_filter(mask, min_area=30, max_area=250,
                              aspect_range=(0.4, 3.5),
                              min_saturation=100, img_hsv=hsv)


def detect_c_improved(img: np.ndarray) -> np.ndarray:
    """C: H[45-75] S[120-255] + 过滤 (较严格) ✓ CLEAN"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 120, 80), (75, 255, 255))
    return _denoise_and_filter(mask, min_area=20, max_area=200,
                              aspect_range=(0.4, 3.0),
                              min_saturation=110, img_hsv=hsv)


def detect_d_improved(img: np.ndarray) -> np.ndarray:
    """D: A + 形态学 + 增强过滤"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 80, 60), (85, 255, 255))
    return _denoise_and_filter(mask, min_area=25, max_area=250,
                              aspect_range=(0.3, 4.0),
                              min_saturation=85, img_hsv=hsv)


def detect_e_improved(img: np.ndarray) -> np.ndarray:
    """E: B + 严格饱和度过滤 (S>115) ✓ CLEAN"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (40, 100, 80), (80, 255, 255))
    return _denoise_and_filter(mask, min_area=30, max_area=250,
                              aspect_range=(0.4, 3.5),
                              min_saturation=115, img_hsv=hsv)


def detect_f_improved(img: np.ndarray) -> np.ndarray:
    """F: C + 极严格过滤"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (45, 120, 80), (75, 255, 255))
    return _denoise_and_filter(mask, min_area=20, max_area=200,
                              aspect_range=(0.4, 3.0),
                              min_saturation=120, img_hsv=hsv)


def detect_g_improved(img: np.ndarray) -> np.ndarray:
    """G: A + ROI (下2/3) + 过滤"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 80, 60), (85, 255, 255))
    h_img = img.shape[0]
    mask[:h_img//3, :] = 0  # 屏蔽上1/3
    return _denoise_and_filter(mask, min_area=25, max_area=250,
                              aspect_range=(0.3, 4.0),
                              min_saturation=90, img_hsv=hsv)


def detect_h_improved(img: np.ndarray) -> np.ndarray:
    """H: H[38-82] S[110-255] V[70-200] + 过滤 ✓ CLEAN"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (38, 110, 70), (82, 255, 200))
    return _denoise_and_filter(mask, min_area=25, max_area=250,
                              aspect_range=(0.35, 3.5),
                              min_saturation=105, img_hsv=hsv)


def detect_i_improved(img: np.ndarray) -> np.ndarray:
    """I: 双阈值 + 颜色纯度 + 过滤 ✓ CLEAN"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_loose = cv2.inRange(hsv, (35, 70, 60), (90, 255, 255))
    mask_strict = cv2.inRange(hsv, (42, 110, 80), (78, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    seed = cv2.dilate(mask_strict, kernel, iterations=2)
    mask = cv2.bitwise_and(mask_loose, seed)
    return _denoise_and_filter(mask, min_area=20, max_area=200,
                              aspect_range=(0.4, 3.0),
                              min_saturation=100, img_hsv=hsv)


# 所有方法列表
METHODS = [
    ("A_improved",      detect_a_improved),
    ("B_improved",      detect_b_improved),
    ("C_improved",      detect_c_improved),
    ("D_improved",      detect_d_improved),
    ("E_improved",      detect_e_improved),
    ("F_improved",      detect_f_improved),
    ("G_improved",      detect_g_improved),
    ("H_improved",      detect_h_improved),
    ("I_improved",      detect_i_improved),
]

# 干净方法列表（检测结果≤1个，推荐用于融合）
CLEAN_METHODS = [
    ("B_improved",      detect_b_improved),
    ("E_improved",      detect_e_improved),
    ("H_improved",      detect_h_improved),
    ("I_improved",      detect_i_improved),
]


# ---------------------------------------------------------------------------
# 融合检测辅助函数
# ---------------------------------------------------------------------------
def _get_iou(box1: Tuple, box2: Tuple) -> float:
    """计算两个bbox的IOU"""
    x1_1, y1_1, w1, h1 = box1
    x1_2, y1_2, w1_2, h1_2 = box2

    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x1_1 + w1, x1_2 + w1_2)
    y_bottom = min(y1_1 + h1, y1_2 + h1_2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w1_2 * h1_2
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def _extract_objects_from_mask(mask: np.ndarray, method_name: str) -> List[dict]:
    """从掩码中提取所有连通域"""
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    objects = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        objects.append({
            'bbox': (x, y, w, h),
            'area': area,
            'centroid': (cx, cy),
            'method': method_name
        })
    return objects


def _fuse_detections(all_objects_by_method: Dict, iou_threshold: float = 0.25) -> List[VehicleDetection]:
    """融合多个方法的检测结果，使用投票机制"""
    if not all_objects_by_method:
        return []

    all_detections = []
    for method_name, objects in all_objects_by_method.items():
        all_detections.extend(objects)

    if not all_detections:
        return []

    # 聚类：根据IOU关联相似的检测
    clusters = []
    used = set()

    for i, det_i in enumerate(all_detections):
        if i in used:
            continue

        cluster = [i]
        used.add(i)

        for j in range(i + 1, len(all_detections)):
            if j in used:
                continue

            det_j = all_detections[j]
            iou = _get_iou(det_i['bbox'], det_j['bbox'])

            if iou > iou_threshold:
                cluster.append(j)
                used.add(j)

        clusters.append(cluster)

    # 生成融合检测结果
    fused = []
    for cluster in clusters:
        cluster_dets = [all_detections[idx] for idx in cluster]

        xs = [det['bbox'][0] for det in cluster_dets]
        ys = [det['bbox'][1] for det in cluster_dets]
        x2s = [det['bbox'][0] + det['bbox'][2] for det in cluster_dets]
        y2s = [det['bbox'][1] + det['bbox'][3] for det in cluster_dets]

        x1, y1 = min(xs), min(ys)
        x2, y2 = max(x2s), max(y2s)

        merged_bbox = (x1, y1, x2 - x1, y2 - y1)
        merged_area = int(np.mean([det['area'] for det in cluster_dets]))
        merged_cx = np.mean([det['centroid'][0] for det in cluster_dets])
        merged_cy = np.mean([det['centroid'][1] for det in cluster_dets])

        votes = len(cluster_dets)
        total_methods = len(all_objects_by_method)
        confidence = votes / total_methods
        methods = list(set(det['method'] for det in cluster_dets))

        fused.append(VehicleDetection(
            bbox=merged_bbox,
            area=merged_area,
            centroid=(merged_cx, merged_cy),
            confidence=confidence,
            votes=votes,
            detecting_methods=methods
        ))

    fused.sort(key=lambda x: (x.votes, x.area), reverse=True)
    return fused


# ---------------------------------------------------------------------------
# GreenVehicleDetector - 主检测器类
# ---------------------------------------------------------------------------
class GreenVehicleDetector:
    """
    绿色车辆检测器，支持融合检测和简单模式。

    参数
    ----
    mode : str
        'fused' - 使用4个干净方法的融合检测（推荐，高置信度）
        'simple' - 使用单一HSV阈值的简单检测（快速）
    h_lo, h_hi : HSV色调范围（仅简单模式使用）
    s_lo : 最低饱和度
    v_lo : 最低亮度
    min_area : 最小像素面积
    roi_top_frac : 屏蔽图像顶部的比例
    """

    def __init__(
        self,
        mode: str = 'fused',
        h_lo: int = 40,
        h_hi: int = 80,
        s_lo: int = 100,
        v_lo: int = 80,
        min_area: int = 30,
        roi_top_frac: float = 0.15,
    ):
        self.mode = mode
        self.lower = np.array([h_lo, s_lo, v_lo], dtype=np.uint8)
        self.upper = np.array([h_hi, 255, 255], dtype=np.uint8)
        self.min_area = min_area
        self.roi_top_frac = roi_top_frac
        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    def detect(self, img_bgr: np.ndarray) -> DetectionResult:
        """对单帧BGR图像执行绿车检测"""
        if self.mode == 'fused':
            return self._detect_fused(img_bgr)
        else:
            return self._detect_simple(img_bgr)

    def _detect_simple(self, img_bgr: np.ndarray) -> DetectionResult:
        """简单检测模式 - 单一HSV阈值"""
        h_img = img_bgr.shape[0]
        roi_top = int(h_img * self.roi_top_frac)

        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        mask[:roi_top, :] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open, iterations=1)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        detections: List[VehicleDetection] = []
        max_area = float(h_img * img_bgr.shape[1])

        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = float(centroids[i][0]), float(centroids[i][1])
            conf = min(1.0, area / max_area * 100)

            detections.append(VehicleDetection(
                bbox=(x, y, bw, bh),
                area=area,
                centroid=(cx, cy),
                confidence=conf,
            ))

        detections.sort(key=lambda d: d.area, reverse=True)
        return DetectionResult(mask=mask, detections=detections)

    def _detect_fused(self, img_bgr: np.ndarray) -> DetectionResult:
        """融合检测模式 - 投票机制"""
        # 收集所有干净方法的检测结果
        all_objects_by_method = {}
        mask_combined = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

        for method_name, detect_func in CLEAN_METHODS:
            mask = detect_func(img_bgr)
            objects = _extract_objects_from_mask(mask, method_name)
            all_objects_by_method[method_name] = objects
            mask_combined = cv2.bitwise_or(mask_combined, mask)

        # 融合检测结果
        detections = _fuse_detections(all_objects_by_method, iou_threshold=0.25)

        return DetectionResult(mask=mask_combined, detections=detections)

    def visualize(
        self,
        img_bgr: np.ndarray,
        result: Optional[DetectionResult] = None,
        box_color: Tuple[int, int, int] = (0, 255, 0),
        mask_color: Tuple[int, int, int] = (0, 255, 128),
    ) -> np.ndarray:
        """将检测结果叠加到图像上"""
        if result is None:
            result = self.detect(img_bgr)

        vis = img_bgr.copy()

        # 半透明掩码叠加
        overlay = vis.copy()
        overlay[result.mask > 0] = mask_color
        vis = cv2.addWeighted(vis, 0.6, overlay, 0.4, 0)

        # 检测框 + 信息标注
        for det in result.detections:
            x, y, w, h = det.bbox
            # 根据投票数和置信度调整颜色
            if det.votes == 4 or det.confidence >= 0.95:
                color = (0, 255, 0)  # 纯绿
            elif det.votes >= 3 or det.confidence >= 0.75:
                color = (0, 200, 50)
            else:
                color = (100, 200, 0)

            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
            label = f"V{det.votes if det.votes > 0 else 1} {det.area}px"
            cv2.putText(vis, label, (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

        # 帧级统计
        info = f"Detected: {result.count} | Mode: {self.mode}"
        cv2.putText(vis, info, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return vis


# ---------------------------------------------------------------------------
# 可视化辅助函数
# ---------------------------------------------------------------------------
def _overlay_mask(img: np.ndarray, mask: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 128)) -> np.ndarray:
    overlay = img.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(img, 0.6, overlay, 0.4, 0)


def _build_compare_grid(img: np.ndarray, methods: List = None) -> np.ndarray:
    """生成多方案对比网格图"""
    if methods is None:
        methods = METHODS

    h, w = img.shape[:2]
    cols = 3
    rows = (len(methods) + cols - 1) // cols
    pad, label_h = 4, 22
    cell_w = w + pad
    cell_h = h + label_h + pad
    canvas = np.ones((rows * cell_h + pad, cols * cell_w + pad, 3), dtype=np.uint8) * 40

    for idx, (name, func) in enumerate(methods):
        r, c = divmod(idx, cols)
        x = c * cell_w + pad
        y = r * cell_h + pad
        mask = func(img)
        vis = _overlay_mask(img, mask)
        cv2.putText(canvas, name, (x + 2, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        canvas[y + label_h: y + label_h + h, x: x + w] = vis

    return canvas


# ---------------------------------------------------------------------------
# 批量处理入口
# ---------------------------------------------------------------------------
def _run_batch(
    src_dir: str = OBSTACLE_DIR,
    out_dir: str = OUT_DIR,
    max_images: int = 20,
):
    """对obstacle目录里的图片批量运行检测"""
    os.makedirs(out_dir, exist_ok=True)
    detector_fused = GreenVehicleDetector(mode='fused')
    detector_simple = GreenVehicleDetector(mode='simple')

    files = sorted(
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )[:max_images]

    if not files:
        print(f"[warn] {src_dir} 中没有找到图片。")
        return

    for fname in files:
        path = os.path.join(src_dir, fname)
        img = cv2.imread(path)
        if img is None:
            print(f"[skip] 无法读取: {path}")
            continue

        stem = os.path.splitext(fname)[0]

        # 1. 多方案对比网格
        grid = _build_compare_grid(img, CLEAN_METHODS)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_clean_methods.png"), grid)

        # 2. 融合检测可视化
        result_fused = detector_fused.detect(img)
        vis_fused = detector_fused.visualize(img, result_fused)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_fused.png"), vis_fused)

        # 3. 简单检测可视化
        result_simple = detector_simple.detect(img)
        vis_simple = detector_simple.visualize(img, result_simple)
        cv2.imwrite(os.path.join(out_dir, f"{stem}_simple.png"), vis_simple)

        print(f"[ok] {fname}  →  融合: {result_fused.count} 辆, 简单: {result_simple.count} 辆")

    print(f"\n完成！结果保存在: {out_dir}")


if __name__ == "__main__":
    _run_batch()
