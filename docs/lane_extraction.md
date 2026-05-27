# 赛道线提取技术文档

## 概述

本项目支持两种赛道场景的线条提取：

| 场景 | 代码模块 | 赛道特征 |
|------|----------|----------|
| **GT**（Generated Track，仿真赛道）| `module/GT2NewTrack.py` | 灰色柏油路面，黄色实线居中，白色实线双侧，绿草地，橙色锥桶 |
| **WS**（WaveShare，室内赛道）| `module/WS2NewTrack.py` | 浅灰地板，白色虚线居中，黄色实线双侧，墙壁/货架/家具背景 |

---

## GT 赛道

### 最终方案

#### 白线检测 `detect_white_line()`

**核心思路**：HSV 严格白色阈值 + 局部对比度补充，二者取并集，再用 Sobel 边缘支持过滤（AND），最后形状过滤去除噪点。

```
原图 BGR
  ├─ HSV 白色掩码（S≤30, V≥200）
  │     × road_support 掩码
  ├─ 局部对比度掩码（_detect_white_line_local_contrast）
  │     行质心编码 + 纵向连续性过滤
  │     × road_support 掩码
  │
  ├─ 合并（OR）→ 边缘支持 AND（edge_enhance=True）
  │
  ├─ 黄色色调排除 H∈[8,50] S≥12 V≥50 + 1px 膨胀
  ├─ 道路掩码腐蚀（3×3, iter=1）去除草地边界像素
  ├─ _denoise_shape（min_area=40, min_aspect=1.5）
  └─ 紧凑块拒绝（area<400 且 aspect<1.5）+ 绿色边界检查（>25% 边界为草 → 拒绝）
```

**置信度**：`tophat 多尺度响应（11×11 / 17×17 / 25×25）归一化到 [0.4, 1.0]`，仅加权强度，不扩展线宽。

#### 黄线检测 `detect_yellow_line()`

**核心思路**：透视自适应双阈值（近端严格 / 远端放宽），CC 面积+形状过滤，远端颜色验证。

```
原图 BGR
  ├─ 近端 HSV 掩码（H[12,45], S≥50, V≥70）
  ├─ 远端 HSV 掩码（H[10,35], S≥35, V≥50）× 上部 45% 区域
  ├─ 合并（OR）→ 道路掩码腐蚀 × 锥桶排除
  ├─ 仅小噪点去除（OPEN 2×2，保持线宽）
  ├─ CC 过滤
  │     远端：area≥5 或（area≥3 且 aspect≥1.2）
  │     近端：area≥40 或（area≥10 且 aspect≥1.8）
  │     mean_H > 42 → 拒绝（草地偏绿黄色）
  │     紧贴左右边缘小 CC → 拒绝（草地边界）
  ├─ 远端颜色验证：mean_H∈[10,42] 且 mean_S≥25
  └─ 近端绿色边界检查（>30% 边界为绿 → 拒绝）
```

**置信度**：`饱和度 × 亮度（S×V × 2.5）`。

---

### GT 尝试过的失败方案

#### 白线

| 方案 | 失败原因 |
|------|----------|
| **纯 HSV 严格阈值**（S≤30, V≥200）| 远端白线亮度低（V<200）被漏检；路面明亮纹理误检 |
| **White Top-Hat 单独使用** | 路面局部纹理、车身反光产生大量假阳性；无颜色区分 |
| **自适应阈值**（Gaussian, blockSize=31, C=−15）| 路面边缘（路面→草地高对比）也触发检测；对光照变化不够免疫 |
| **局部对比度 + 梯度**（V−GaussianBlur + Sobel）| 路面任意亮斑都产生响应；形状过滤困难 |
| **三路融合**（TopHat + 对比度 + 梯度投票≥2）| 噪点量与单路方案相当；参数调优维度高，泛化性差 |
| **边缘膨胀增强**（edge_enhance dilation）| 将检测到的线条向外膨胀 3×3，线宽超出原始值，导致观测空间失真 |
| **Gaussian Blur 软化**（5×5, σ=1.5）| 软化导致线宽扩大约 4-6px，与原始线宽不符 |

#### 黄线

| 方案 | 失败原因 |
|------|----------|
| **LAB + HSV + R>G 三重确认** | LAB b 通道对远端褪色黄线不敏感；R>G 条件在某些光照下误拒真实黄线 |
| **Top-Hat + Hue 确认** | 灰度 Top-Hat 不区分颜色，将路面高对比亮斑误认为黄线 |
| **宽松 HSV + 形状过滤** | H 上限放宽后捕获大量草地偏黄色（H≈43-50）假阳性；近端无法仅靠形状区分 |
| **固定远端阈值**（不区分近/远端）| 近端使用放宽阈值引入大量噪点；远端使用严格阈值漏检小面积线段 |

---

## WS 赛道

### 最终方案

#### 白线检测 `extract_ws_white_case10_line()`

**核心思路**：WS 白线是**虚线**（非实线），需要检测离散线段后填充间隙形成连续线。

```
原图 BGR
  ├─ 道路掩码（road_support_mask：底部种子生长，排除墙壁/非路面区域）
  ├─ 多尺度 White Top-Hat（9×9 / 15×15 / 21×21 椭圆核）
  │     取最大响应 → 局部比周围更亮的区域
  ├─ 黄线排除（避免黄线混入白线 seed）
  ├─ 行质心编码（_row_multi_centroid_encode）
  │     每行检测线段质心，宽度限制（max_width_ratio=0.18）
  ├─ 纵向连续性过滤（_vertical_continuity_filter, min_run=5）
  │     要求在连续多行出现才认为是线段
  ├─ 虚线间隙填充（MORPH_CLOSE，核 15×3）
  │     垂直方向闭合，将相邻虚线段连接为实线
  └─ 返回置信度（tophat 响应归一化）
```

#### 黄线检测 `extract_ws_yellow_mask()`

```
HSV 掩码（H∈[15,45], S≥55, V≥60）
  × 道路掩码
  → 形态学清理（CLOSE+OPEN）
  → CC 过滤（纵向连续性）
  → 返回置信度（S×V 归一化）
```

---

### WS 尝试过的失败方案

| 方案 | 失败原因 |
|------|----------|
| **纯 HSV 白色阈值**（S≤40, V≥180）| 墙壁、货架、家具与白色虚线 HSV 完全相同，无法区分 |
| **仅道路掩码过滤**（限制在路面内）| 道路掩码本身也包含墙壁区域（灰白色、低饱和度），道路边界模糊 |
| **Top-Hat + 固定阈值**（无行质心编码）| 虚线段检测分散，间隙未填充，输出为离散小块而非连续线 |
| **自适应阈值**（局部均值比较）| 墙角、货架边缘高对比区域产生大量假阳性；室内光照不均匀加剧误检 |
| **Gaussian Blur 软化概率图**（5×5, σ=1.5）| 扩展线宽；低置信度噪点被"漂白"为高置信，模糊了真实线条位置 |
| **edge_grow dilation**（grow×edge_support×0.45）| 检测到的白线向外扩张约 2-3px，不符合原始线宽要求 |
| **时序膨胀**（temporal dilate×1）| 将前帧线条向外膨胀后与当前帧融合，持续扩展线宽 |
| **top 22% 区域排除**（过于保守）| 漏检部分靠近顶部的白线段；噪声仍然严重 |
| **置信度 floor**（`max(0.3 × mask, conf)`）| 0.3 floor 使所有检测到的位置都有较高可见度，掩盖了低置信噪点应淡化的效果 |

---

## 当前视觉感知处理流程

```
DonkeyEnv
  │
  ▼
原始 RGB 帧 (H×W×3) uint8
  │
  ▼  CanonicalSemanticWrapper.observation()  [module/obv.py]
  │
  ├─ resize → (128×128) 或 (160×120)
  ├─ RGB → BGR（线提取器期望 BGR）
  ├─ RGB → YCrCb → 取 Y 通道 / 255.0  →  ch0: raw_Y
  │
  ├─ [GT 域] build_gt_observation_line_probs()  [GT2NewTrack.py]
  │     ├─ detect_gt_road_support()     → 道路掩码（底部种子生长）
  │     ├─ detect_white_line()          → 白线二值掩码（原始线宽）
  │     ├─ detect_yellow_line()         → 黄线二值掩码（原始线宽）
  │     ├─ 白线置信度 = 掩码 × TopHat 归一化（0.4~1.0）
  │     ├─ 黄线置信度 = 掩码 × S×V×2.5
  │     └─ Sobel 边缘图（raw_Y，Gaussian 预模糊 σ=1.0）
  │
  ├─ [WS 域] build_ws_observation_line_probs()  [WS2NewTrack.py]
  │     ├─ road_support_mask()          → 道路掩码
  │     ├─ extract_ws_yellow_mask()     → 黄线置信度（HSV+形态学）
  │     ├─ extract_ws_white_case10_line()→ 白线置信度（TopHat+行质心+间隙填充）
  │     └─ Sobel 边缘图（raw_Y）
  │
  ├─ ch1: white_prob    float32 [0,1]  （无线宽膨胀，原始线宽）
  ├─ ch2: yellow_prob   float32 [0,1]  （无线宽膨胀，原始线宽）
  ├─ ch3: sobel_edge    float32 [0,1]  （Sobel 边缘强度）
  ├─ ch4: vehicle_prob  float32 [0,1]  （GreenVehicleDetector）
  └─ ch5: motion_residual float32 [0,1] （|Y_t − Y_{t−1}| × 4，reset 后清零）
  │
  ▼
观测张量 (6, obs_size, obs_size) float32  →  PPO Agent
  │
  ▼
状态向量 (5~7,) float32
  ├─ v_long_norm       = clip(speed / v_max, 0, 2)
  ├─ yaw_rate_norm     = clip(gyro_y / 8.0, −2, 2)
  ├─ accel_x_norm      = clip(accel_x / 9.8, −2, 2)
  ├─ prev_steer        ∈ [−1, 1]
  └─ prev_throttle     ∈ [−1, 1]
```

---

## 关键设计原则

### 线宽一致性
所有方案明确**禁止线宽膨胀**：
- 不使用 `cv2.dilate` 扩展检测到的线条
- 不使用 `GaussianBlur` 对线掩码进行空间软化
- 置信度权重仅调整像素强度，不改变像素范围

### 近端/远端分级策略
透视效应使远端线条面积小、饱和度低：
- GT 黄线：`far_boundary = 45%`，远端放宽面积阈值（area≥5 vs 近端 area≥40）
- WS 白线：行质心编码天然处理透视压缩

### 颜色互斥
- 白线检测中排除黄色色调（H∈[8,50], S≥12）防止黄→白误检
- WS 白线提取明确排除已检测的黄线区域

### 输出格式
- 全部以 `float32 [0,1]` 置信度图输出
- 高置信 = 线条核心（颜色鲜艳/对比强）
- 低置信 = 线条边缘或条件不确定
- 可视化使用 inferno colormap（暗→黑，高→亮黄）

---

## 文件结构

```
module/
  GT2NewTrack.py          GT 全流程：道路掩码 / 白线 / 黄线 / obs 构建
  WS2NewTrack.py          WS 全流程：道路掩码 / 白线 / 黄线 / obs 构建
  obv.py                  CanonicalSemanticWrapper（6 通道 obs 封装）

src/
  gt_line_extractor.py    GT 线提取可视化（调用 GT2NewTrack，生成对比图）
  test_obv_output.py      GT/WS obs 通道可视化（inferno 热图）
  compare_ws2newtrack_lanes.py  WS 置信度可视化
  test_gt_obs_compare.py  GT 三方案对比（A/B/C）

output/
  gt_line_compare/        GT 线提取对比页（白线、黄线）
  obv_test/               GT/WS obs 通道热图
  ws2newtrack_lanes/      WS 置信度页
```
