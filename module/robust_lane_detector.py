#!/usr/bin/env python3
"""
config notedetectionnote - V9.4 noteV9.3c, note:

V9.4 note(note V9.3c):
  Fix A - Coverage Dropout note:
    - trainingnote mask noterowsnote p~U(0.15, 0.45)
    - note coverage distributionnote [0.37, 0.56] note [0.2, 0.5]
    - note dropout(apply_dr=False)
  Fix B - DR note mask/edges note:
    - mask/edges note RGB compute(note)
    - DR note RGB note 0-2
    - HSV note H±8 note H±4
  Fix C - note/CNN note(note ppo_waveshare_v8.py note)

V9.3c note:
  notedetectionnote(per-domain edge detection):
  - WS:   HSV notedetection H15-40, S>=60, V>=80 + road_mask
  - Real: HSV notedetection H12-45, S>=35, V>=70 + road_mask
  - GT:   geometrynotedetection - road_mask note ±8px, V>=120

  rowsnote (Row Centroid Encoding):
  1. per-domain notedetection
  2. CLOSE-only notecleanup
  3. noterowsnote, note 7px note
  4. note >15% note + noteclassnote
"""

import numpy as np
import cv2
import random
from typing import Tuple


class RobustLaneDetector:
    """
    notedetectionnote V9.3c - per-domain notedetection + rowsnote(Real/WS/GT)note.

    note:
    1. notedetection: note -> note
    2. ★per-domain notedetection:
       - WS: note HSV(V9.2 note)
       - Real: note HSV(H12-45, S>=35, V>=70 note)
       - GT: geometrynotedetection(road_mask note ±8px note, 99.1% note)
    3. CLOSE-only note(note OPEN, note Real note)
    4. rowsnote: note 7px note, note >15% note + noteclassnote
    5. note: bilateral+Canny+road_mask
    """

    def __init__(self,
                 # ★ V9.3c: note
                 domain: str = 'ws',         # 'ws' / 'real' / 'gt'
                 # notedetectionnote
                 road_s_max: int = 50,       # note(note=note)
                 road_v_min: int = 70,       # note
                 road_v_max: int = 255,      # note
                 # notedetectionnote(WS/Real note)
                 yellow_h_range: Tuple[int, int] = (15, 40),
                 yellow_s_min: int = 60,     # WS: 60, Real: 35
                 yellow_v_min: int = 80,     # WS: 80, Real: 70
                 # GT geometrydetectionnote
                 gt_search_radius: int = 8,  # road boundary note
                 gt_min_bright: int = 120,   # GT note
                 # rowsnote(note)
                 centroid_band_width: int = 7,    # ★ noteoutputnote(note), note
                 centroid_min_pixels: int = 1,    # ★ V9.3c: 1(note, V9.2 note 3)
                 centroid_max_width_ratio: float = 0.15,  # ★ note(note)
                 # notedetectionnote(defaultnote)
                 white_s_max: int = 40,
                 white_v_min: int = 180,
                 enable_white_detection: bool = False,
                 # notedetectionnote
                 canny_low: int = 60,
                 canny_high: int = 180,
                 use_bilateral_filter: bool = True,
                 bilateral_d: int = 5,
                 bilateral_sigma_color: int = 50,
                 bilateral_sigma_space: int = 50,
                 # notecontrol
                 use_road_mask_for_edges: bool = True,
                 use_road_mask_for_yellow: bool = True,
                 # ★ V9.4: Coverage dropout - note coverage note domain
                 coverage_dropout_range: Tuple[float, float] = (0.15, 0.45)):

        self.domain = domain
        self.coverage_dropout_range = coverage_dropout_range

        self.road_s_max = road_s_max
        self.road_v_min = road_v_min
        self.road_v_max = road_v_max

        self.yellow_h_range = yellow_h_range
        self.yellow_s_min = yellow_s_min
        self.yellow_v_min = yellow_v_min

        # GT geometrydetectionnote
        self.gt_search_radius = gt_search_radius
        self.gt_min_bright = gt_min_bright

        # rowsnote
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

        # note
        self._close_kernel = np.ones((5, 5), np.uint8)
        self._dilate_kernel = np.ones((7, 7), np.uint8)
        self._line_kernel = np.ones((3, 3), np.uint8)

        domain_desc = {'ws': 'WS(yellow H15-40,S60+)', 'real': 'Real(yellow H12-45,S35+)', 'gt': 'GT(geometric boundary)'}
        print(f"config notedetectionnote V9.3c note domain={domain}")
        print(f"   detectionnote: {domain_desc.get(domain, domain)}")
        print(f"   rowsnote: band={centroid_band_width}px, min_px={centroid_min_pixels}, max_width={centroid_max_width_ratio:.0%}")
        print(f"   note: CLOSE-only(noteOPEN, note)")
        print(f"   note: bilateral({bilateral_d},{bilateral_sigma_color},{bilateral_sigma_space}), Canny({canny_low},{canny_high})")

    def detect_road(self, hsv: np.ndarray) -> np.ndarray:
        """
        detectionnote: note + note(note/note)
        note waveshare(note)note generated_track(note)note.

        Returns:
            road_mask: uint8 (H, W), 255=note
        """
        # note + note = note
        s_mask = hsv[:,:, 1] < self.road_s_max
        v_mask = (hsv[:,:, 2] >= self.road_v_min) & (hsv[:,:, 2] <= self.road_v_max)
        road_mask = (s_mask & v_mask).astype(np.uint8) * 255

        # notecleanup: note(V9.3c: noteOPEN, note)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, self._close_kernel)

        # note(note)
        road_expanded = cv2.dilate(road_mask, self._dilate_kernel, iterations=2)

        return road_mask, road_expanded

    def detect_lane_lines(self, rgb: np.ndarray, hsv: np.ndarray,
                          road_mask: np.ndarray,
                          road_expanded: np.ndarray) -> np.ndarray:
        """
        V9.3c notedetection - per-domain + rowsnotedetectionnote:
        - WS:   HSV note H15-40, S>=60, V>=80 + road_mask
        - Real: HSV note H12-45, S>=35, V>=70 + road_mask(note, note)
        - GT:   geometrynotedetection - road_mask note ±8px note(H35-85,S>100), note 99.1%

        Returns:
            lane_mask: uint8 (H, W), 255=note
        """
        h, w = hsv.shape[:2]

        if self.domain == 'gt':
            # === GT: geometrynotedetection ===
            lane_raw = self._detect_gt_geometric(hsv, road_mask, h, w)
        else:
            # === WS / Real: HSV notedetection ===
            yellow_mask = cv2.inRange(
                hsv,
                np.array([self.yellow_h_range[0], self.yellow_s_min, self.yellow_v_min]),
                np.array([self.yellow_h_range[1], 255, 255])
            )
            lane_raw = yellow_mask

            # note
            if self.use_road_mask_for_yellow:
                lane_raw = cv2.bitwise_and(lane_raw, road_expanded)

        # === notecleanup: CLOSE-only(V9.3c: noteOPEN, noteRealnote) ===
        lane_raw = cv2.morphologyEx(lane_raw, cv2.MORPH_CLOSE, self._line_kernel)

        # === ★ rowsnote: note ===
        result = self._row_centroid_encode(lane_raw, h, w)

        return result

    def apply_coverage_dropout(self, lane_mask: np.ndarray) -> np.ndarray:
        """
        ★ V9.4 Coverage Dropout: noterowsnote, note coverage note.

        note: GT coverage~0.37, WS/Real~0.56, note mask note domain.
        note: trainingnoterowsnote p~U(dropout_range) note,
              note coverage distributionnote [0.2, 0.5] note.
        note(apply_dr=False)notefunction.
        """
        if self.coverage_dropout_range is None:
            return lane_mask

        lo, hi = self.coverage_dropout_range
        h = lane_mask.shape[0]
        # noterowsnote dropout
        drop_prob = np.random.uniform(lo, hi)
        drop_mask = np.random.random(h) < drop_prob
        result = lane_mask.copy()
        result[drop_mask] = 0
        return result

    def _detect_gt_geometric(self, hsv: np.ndarray, road_mask: np.ndarray,
                             h: int, w: int) -> np.ndarray:
        """
        GT geometrynotedetection: note road_mask note,
        note ±search_radius px note(V>=min_bright).
        note(H35-85, S>100).

        V9.5 note: notedetectionnote road_mask note(note),
        note road_cols[0]/[-1], note generated_track note(coverage~0.02)note.
        """
        result = np.zeros((h, w), dtype=np.uint8)
        v_chan = hsv[:,:, 2].astype(np.float32)
        sr = self.gt_search_radius
        mb = self.gt_min_bright

        for row in range(h):
            road_row = road_mask[row] > 0
            if road_row.sum() < 10:
                continue

            # ── note(note->note->note)──────────────
            diffs = np.diff(road_row.astype(np.int8))
            # non-road->road: diff=+1, note = transition_col
            # road->non-road: diff=-1, note = transition_col
            boundaries = list(np.where(diffs!= 0)[0] + 1)

            # note col=0 note(note), note
            if road_row[0]:
                boundaries.append(0)
            # note col=w-1 note(note), note
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

                # notebright_mask = (wv >= mb) & ~((wh >= 35) & (wh <= 85) & (ws > 100))

                for c in np.where(bright_mask)[0]:
                    result[row, left_s + c] = 255

        return result

    def _row_centroid_encode(self, lane_raw: np.ndarray, h: int, w: int) -> np.ndarray:
        """
        rowsnote: noterowscomputenote, note.
        noterowsnoteclassnote, note.
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
                # noteclassnote: note, note
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
        notedetection(V9.2: note + note Canny + note)

        V9.2 note:
        - bilateral(5,50,50): noteV9.1(9,75,75)note, noteRealnote
        - Canny(60,180): noteV9.1(100,250)note, noteRealnote
        - note: note: max ratio=1.6x(V8note: 3.9x)

        Returns:
            edges: uint8 (H, W), 255=note
        """
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        # ★ note: note, note(note, note)
        if self.use_bilateral_filter:
            gray = cv2.bilateralFilter(
                gray, self.bilateral_d,
                self.bilateral_sigma_color, self.bilateral_sigma_space
            )

        # CLAHE note
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_enhanced = clahe.apply(gray)

        # ★ note Canny(note)
        edges = cv2.Canny(gray_enhanced, self.canny_low, self.canny_high)

        # ★ note(note)
        if self.use_road_mask_for_edges:
            edges = cv2.bitwise_and(edges, road_expanded)

        return edges

    def detect(self, rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        notedetectionnote

        Args:
            rgb: RGB uint8 (H, W, 3)

        Returns:
            lane_mask: uint8 (H, W), 255=note
            edges: uint8 (H, W), 255=note
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

        # 1. notedetection
        road_mask, road_expanded = self.detect_road(hsv)

        # 2. notedetection(per-domain + rowsnote)
        lane_mask = self.detect_lane_lines(rgb, hsv, road_mask, road_expanded)

        # 3. notedetection
        edges = self.detect_road_edges(rgb, road_mask, road_expanded)

        return lane_mask, edges

    def detect_with_debug(self, rgb: np.ndarray) -> dict:
        """
        notedetection(note)

        Returns:
            dict: noteresult
        """
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        road_mask, road_expanded = self.detect_road(hsv)

        # note(note)
        yellow_raw = cv2.inRange(
            hsv,
            np.array([self.yellow_h_range[0], self.yellow_s_min, self.yellow_v_min]),
            np.array([self.yellow_h_range[1], 255, 255])
        )

        # V8note(note)
        yellow_v8 = cv2.inRange(hsv, np.array([15, 60, 60]), np.array([40, 255, 255]))
        kernel = np.ones((3, 3), np.uint8)
        yellow_v8 = cv2.morphologyEx(yellow_v8, cv2.MORPH_CLOSE, kernel)
        yellow_v8 = cv2.morphologyEx(yellow_v8, cv2.MORPH_OPEN, kernel)

        lane_mask = self.detect_lane_lines(rgb, hsv, road_mask, road_expanded)
        edges = self.detect_road_edges(rgb, road_mask, road_expanded)

        # note Canny(V8note, note)
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
    V9.3c note - per-domain notedetection

    note(note V9.2):
    - per-domain notedetection: WS/Real=note, GT=geometrynote
    - CLOSE-only note(note Real note)
    - min_pixels=1(V9.2 note 3, note)
    - noteclassnote(noterowsnote)
    - DR note RGB
    """

    def __init__(self, enable_dr=False, dr_prob=0.6, detector: RobustLaneDetector = None):
        self.enable_dr = enable_dr
        self.dr_prob = dr_prob
        self.detector = detector or RobustLaneDetector(domain='ws')

        print(f"config V9.3c note domain={self.detector.domain}")
        print(f"   RGB-only DR: {'note' if enable_dr else 'note'} (prob={dr_prob})")

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
        # ★ V9.4: H±4(note±8note), note HSV note mask note
        if random.random() < 0.4:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:,:, 0] = np.clip(hsv[:,:, 0] + random.uniform(-4, 4), 0, 179)
            hsv[:,:, 1] = np.clip(hsv[:,:, 1] * random.uniform(0.85, 1.15), 0, 255)
            hsv[:,:, 2] = np.clip(hsv[:,:, 2] + random.uniform(-15, 15), 0, 255)
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
        note(V9.4 note)

        ★ V9.4 note: mask/edges note RGB compute, DR note RGB note.

        notefirst: DR->RGB', note RGB' compute mask/edges -> mask note DR note: note RGB compute mask/edges(note), DR note 0-2

        note V8 note:
        input: RGB (H, W, 3)
        output: (rgb, lane_mask, edges)
        """
        rgb_clean = img.copy()

        # ① mask/edges note RGB compute(note, note DR note)
        lane_mask, edges = self.detector.detect(rgb_clean)

        # ② DR note RGB note(note 0-2)
        if apply_dr:
            rgb_out = self._apply_dr(rgb_clean)
            # ③ Coverage dropout: trainingnote coverage note
            lane_mask = self.detector.apply_coverage_dropout(lane_mask)
        else:
            rgb_out = rgb_clean

        return rgb_out, lane_mask, edges


# ============================================================
# note
# ============================================================
def validate_on_samples(ws_dir: str = "data/scene_samples/waveshare/processed",
                        gt_dir: str = "data/scene_samples/generated_track/processed",
                        output_dir: str = "data/lane_detection_validation",
                        max_images: int = 50):
    """
    notedetectionnote vs V8notedetectionnote:
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
            print(f"⚠️  {img_dir} note")
            continue

        print(f"\nmetrics note {scene}: {len(images)} note")

        for i, img_path in enumerate(images):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            debug = detector.detect_with_debug(img_rgb)
            results[scene].append(debug["stats"])

            # note10notesavenote
            if i % 10 == 0:
                _save_comparison(debug, scene, i, output_dir)

        # note
        stats = results[scene]
        print(f"   V8 note ratio: {np.mean([s['yellow_v8_ratio'] for s in stats])*100:.2f}%")
        print(f"   V9 note ratio: {np.mean([s['lane_mask_ratio'] for s in stats])*100:.2f}%")
        print(f"   V8 note: {np.mean([s['edges_v8_density'] for s in stats])*100:.2f}%")
        print(f"   V9 note: {np.mean([s['edges_new_density'] for s in stats])*100:.2f}%")
        print(f"   note: {np.mean([s['road_ratio'] for s in stats])*100:.2f}%")

    # note
    if results["waveshare"] and results["generated_track"]:
        ws = results["waveshare"]
        gt = results["generated_track"]

        print("\n" + "="*60)
        print("📏 note (V8 vs V9)")
        print("="*60)

        ws_v8_yr = np.mean([s['yellow_v8_ratio'] for s in ws])
        gt_v8_yr = np.mean([s['yellow_v8_ratio'] for s in gt])
        ws_v9_yr = np.mean([s['lane_mask_ratio'] for s in ws])
        gt_v9_yr = np.mean([s['lane_mask_ratio'] for s in gt])

        v8_ratio = max(ws_v8_yr, gt_v8_yr) / max(min(ws_v8_yr, gt_v8_yr), 1e-6)
        v9_ratio = max(ws_v9_yr, gt_v9_yr) / max(min(ws_v9_yr, gt_v9_yr), 1e-6)

        print(f"\n  note mask:")
        print(f"   V8: WS={ws_v8_yr*100:.2f}% vs GT={gt_v8_yr*100:.2f}% -> {v8_ratio:.1f}x note")
        print(f"   V9: WS={ws_v9_yr*100:.2f}% vs GT={gt_v9_yr*100:.2f}% -> {v9_ratio:.1f}x note")
        print(f"   note: {v8_ratio:.1f}x -> {v9_ratio:.1f}x")

        ws_v8_ed = np.mean([s['edges_v8_density'] for s in ws])
        gt_v8_ed = np.mean([s['edges_v8_density'] for s in gt])
        ws_v9_ed = np.mean([s['edges_new_density'] for s in ws])
        gt_v9_ed = np.mean([s['edges_new_density'] for s in gt])

        v8_ed_ratio = max(ws_v8_ed, gt_v8_ed) / max(min(ws_v8_ed, gt_v8_ed), 1e-6)
        v9_ed_ratio = max(ws_v9_ed, gt_v9_ed) / max(min(ws_v9_ed, gt_v9_ed), 1e-6)

        print(f"\n  note:")
        print(f"   V8: WS={ws_v8_ed*100:.2f}% vs GT={gt_v8_ed*100:.2f}% -> {v8_ed_ratio:.1f}x note")
        print(f"   V9: WS={ws_v9_ed*100:.2f}% vs GT={gt_v9_ed*100:.2f}% -> {v9_ed_ratio:.1f}x note")
        print(f"   note: {v8_ed_ratio:.1f}x -> {v9_ed_ratio:.1f}x")

    print(f"\nPASS notesavenote: {output_dir}/")
    return results


def _save_comparison(debug: dict, scene: str, idx: int, output_dir: str):
    """savenote(V8 vs V9)"""
    import os
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{scene} #{idx} - V8 vs V9 note", fontsize=14)

    # Row 1: V8
    axes[0, 0].imshow(cv2.cvtColor(debug["rgb"], cv2.COLOR_RGB2BGR)[:,:,::-1])
    axes[0, 0].set_title("note (RGB)")

    axes[0, 1].imshow(debug["yellow_v8"], cmap='gray')
    axes[0, 1].set_title(f"V8 note mask\nratio={debug['stats']['yellow_v8_ratio']*100:.2f}%")

    axes[0, 2].imshow(debug["edges_v8"], cmap='gray')
    axes[0, 2].set_title(f"V8 Canny\ndensity={debug['stats']['edges_v8_density']*100:.2f}%")

    axes[0, 3].imshow(debug["road_mask"], cmap='gray')
    axes[0, 3].set_title(f"note\nratio={debug['stats']['road_ratio']*100:.2f}%")

    # Row 2: V9
    axes[1, 0].imshow(debug["road_expanded"], cmap='gray')
    axes[1, 0].set_title("note")

    axes[1, 1].imshow(debug["lane_mask"], cmap='gray')
    axes[1, 1].set_title(f"V9 note mask\nratio={debug['stats']['lane_mask_ratio']*100:.2f}%")

    axes[1, 2].imshow(debug["edges"], cmap='gray')
    axes[1, 2].set_title(f"V9 note\ndensity={debug['stats']['edges_new_density']*100:.2f}%")

    # V8 vs V9 note
    overlay = debug["rgb"].copy()
    overlay[debug["lane_mask"] > 0] = [255, 255, 0]  # note
    axes[1, 3].imshow(overlay)
    axes[1, 3].set_title("V9 note")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{scene}_{idx:03d}_comparison.png")
    plt.savefig(out_path, dpi=100)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="notedetectionnote")
    parser.add_argument("--validate", action="store_true",
                       help="notedetectionnote")
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
        print("note: python -m module.robust_lane_detector --validate")
        print("      note V8 vs V9 detectionnote")
