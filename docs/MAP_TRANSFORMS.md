# 三个地图变换脚本说明

本文档说明 `module` 目录下这三个 NewTrack 风格映射脚本的职责、处理流程和使用方式：

- `WS2NewTrack.py`
- `GT2NewTrack.py`
- `RRL2NewTrack.py`

它们的共同目标都是把不同来源的赛道图像尽量映射到 `new_track` 的视觉风格，但三者的输入域差异很大，所以检测策略也不一样。

三个脚本均已集成 `green_vehicle_detect.py` 中的绿色车辆检测器，处理流程最末会保留绿车原始像素，防止被道路渲染覆盖。

## 1. 总览

| 脚本 | 输入域 | 主要输出风格 | 核心难点 | 当前入口 |
| --- | --- | --- | --- | --- |
| `WS2NewTrack.py` | Waveshare | 路面转浅色 cloth，边线转蓝，中间白线/黄线转暖橙 | 原图本来就有蓝线、黄线、白块，中心线几何容易丢 | 模块函数 |
| `GT2NewTrack.py` | generated_track | 路面转浅色 cloth，白线转蓝，黄线转橙 | 草地/场外误检成线，白线远处对比弱 | CLI + 模块函数 |
| `RRL2NewTrack.py` | rrl / sim racing | 深色柏油重渲染成 NT 白布地面，白线转蓝，棋盘格重绘 | 白线形态变化大，横线/弯线/近端粗线都要保住 | CLI + 模块函数 |

## 2. 共性设计

三个脚本虽然实现不同，但大体都遵循下面这条链路：

1. 先估计路面或语义支持区域，尽量减少场外误检。
2. 再提取需要保留的线条语义。
3. 把路面和背景渲染到更接近 `new_track` 的底色。
4. 最后把线条混合到目标蓝/橙风格。
5. **（新增）** 调用 `GreenVehicleDetector` 检测绿色车辆，将绿车像素从原图回填到输出，保留障碍物外观。

差异主要在第 1 步和第 2 步：

- `WS` 更依赖原图已有的颜色语义。
- `GT` 更依赖颜色阈值加一点局部对比补充。
- `RRL` 更依赖局部对比、连通性和几何规则。

绿车检测集成细节见 `green_vehicle_detect.py`。

## 3. WS2NewTrack.py

文件：`module/WS2NewTrack.py`

### 3.1 适用场景

这个脚本用于 Waveshare 域。它不是简单“重涂颜色”，而是先理解原图里已有的蓝线、黄线、白色中心块，再把它们柔和地映射到 NewTrack 风格。

### 3.2 主要入口

- `compute_stats(images)`
- `extract_ws_yellow_mask(...)`
- `extract_ws_white_case10_line(...)`
- `transform_ws_to_newtrack(img_bgr, tgt_stats, yellow_preset_name="case01")`

注意：这个脚本目前是模块式接口，没有 `main()` 命令行入口。通常要先准备 `tgt_stats`，再逐张调用 `transform_ws_to_newtrack(...)`。

### 3.3 核心流程

#### 1) 语义分离

- `semantic_masks(...)`
  - 先在 HSV 空间里分出 `blue / yellow / white` 三类。
- `road_support_mask(...)`
  - 粗略估计路面区域，避免把明显场外区域当成可处理区域。

#### 2) 目标颜色统计

- `compute_stats(...)`
  - 从参考图像里统计目标域的颜色比例和 HSV prototype。
  - `WS` 的颜色不是写死的纯 canonical 值，而是会参考 `tgt_stats` 再做安全裁剪。

#### 3) 黄线处理

- `extract_ws_yellow_mask(...)`
  - 从原图黄色区域里提取真正的 lane 部分。
  - 通过连通域筛选、边缘线提取和 closing，把零散黄块连成较稳定的边线。
  - 最终输出 `yellow_full`，它会用于蓝线方向的混色。

#### 4) 白线转暖色中心线

- `extract_ws_white_case10_line(...)`
  - 这是 `WS` 里最特殊的一段。
  - 它会先做严格白色检测，再抽出细的中心 seed。
  - 然后通过 `_white_center_support_band(...)`、`_expand_white_seed_to_center_blocks(...)`、`_add_center_white_edge_ring(...)` 逐步把中心白块补完整。
  - 当前逻辑里，近端允许 `seed-band fallback`，避免明明存在的白块被过窄的黄线走廊挡掉。
  - 同时又会经过 `_clip_center_fill_width(...)` 控宽，避免近端直接变成过大的黄块。

#### 5) 渲染和混色

- `_render_ws_road_surface(...)`
  - 先把原始路面渲染成更接近 NT 的浅色 cloth。
- `_apply_newtrack_blend(...)`
  - 再把边线和中心线分别混到目标蓝/橙上。
  - 蓝线和中心线都带有 HSV 混合系数，不是硬替换，所以保留了一些原图的亮度纹理。

#### 6) 绿车保留

- `_green_detector.detect(img_bgr)`
  - 在混色结果上做绿车检测，将绿车区域像素从原图回填，不被 cloth 渲染覆盖。

### 3.4 当前特点

- 优点：
  - 颜色方向最接近 NT。
  - 中心白块转暖色线的可控性最好。
- 代价：
  - 参数较多。
  - 需要 `tgt_stats` 支撑，调用方式比另外两个脚本更偏“模块化”。

### 3.5 典型调用

```python
from mysim.module.WS2NewTrack import compute_stats, transform_ws_to_newtrack

tgt_stats = compute_stats(nt_images)
result = transform_ws_to_newtrack(ws_img, tgt_stats, yellow_preset_name="case01")
```

## 4. GT2NewTrack.py

文件：`module/GT2NewTrack.py`

### 4.1 适用场景

这个脚本用于 `generated_track` 域。它的思路相对直接：先做路面限制，再检测白线和黄线，最后把线重着色、把地面渲染成 NT 风格。

### 4.2 主要入口

- `detect_gt_road_support(img_bgr)`
- `detect_white_line(img_bgr, road_support=None)`
- `detect_yellow_line(img_bgr, road_support=None)`
- `render_road_surface(img_bgr)`
- `process_directory(raw_dir, out_dir, save_masks=False)`

这个脚本有命令行入口，可直接批处理。

### 4.3 核心流程

#### 1) 路面支持区域

- `detect_gt_road_support(...)`
  - 先用 HSV 排掉高饱和绿色区域，减少草地和场外误检。
  - 再保留较大的主要连通域，并做适度膨胀，给真实边线留余量。

#### 2) 白线检测

- `detect_white_line(...)`
  - 白线主检测是 HSV：`S` 低、`V` 高。
  - 同时融合 `_detect_white_line_local_contrast(...)` 的局部对比补充，借鉴了 `RRL` 的思路，用来抓远处或偏灰的白线。
  - 最后再通过 `_denoise_shape(...)` 去掉小块噪声。

#### 3) 黄线检测

- `detect_yellow_line(...)`
  - 主要依赖 LAB 的 `b` 通道和 `R > G` 条件。
  - 再套上 `road_support`，避免草地/场外黄色块被误判成中线。

#### 4) 渲染和改色

- `render_road_surface(...)`
  - 先把 GT 的道路和背景渲染为 NT 近似的 cloth 和背景色。
- `recolor_lines(...)`
  - 最后把白线直接替换为蓝色，把黄线直接替换为橙色。

#### 5) 绿车保留

- `_green_detector.detect(img)`
  - 在 `process_directory` 循环内，`recolor_lines` 之后检测绿车并回填原始像素。

### 4.4 当前特点

- 优点：
  - 结构清晰，适合做批处理。
  - 路面限制后，对草地误检控制较好。
- 代价：
  - 黄线检测仍然更依赖颜色阈值，不像 `WS` 那样有更复杂的几何补线。
  - 视觉上通常比 `WS` 稍硬一些，因为线条是直接替换颜色。

### 4.5 命令行用法

```bash
python -m module.GT2NewTrack \
  --raw_dir data/scene_samples/generated_track/raw \
  --out_dir data/scene_samples/generated_track/GT2NewTrack \
  --save_masks
```

## 5. RRL2NewTrack.py

文件：`module/RRL2NewTrack.py`

### 5.1 适用场景

这个脚本用于 `rrl` 域。相对另外两个脚本，`RRL` 的输入更像真实/半真实赛道，白线形状变化大，背景复杂，还有棋盘格、彩色护栏等元素，所以它的规则最多。

### 5.2 主要入口

- `detect_rrl_white_lines(img_bgr)`
- `detect_rrl_checker(img_bgr)`
- `detect_rrl_road_surface(img_bgr)`
- `transform_rrl_to_newtrack(img_bgr)`
- `transform_rrl_lines_only(img_bgr)`
- `process_directory(raw_dir, out_dir, mode="full", save_masks=False, save_preview=True)`

这个脚本同时支持：

- `full`：整张图重渲染成 NT 风格。
- `lines`：只改线条颜色，保留原场景其余部分。

### 5.3 核心流程

#### 1) 白线检测

- `detect_rrl_white_lines(...)`
  - 先做局部对比：`V - local_mean(V)`。
  - 再按行找多个 cluster，通过 `_row_multi_centroid_encode(...)` 保住多条线。
  - 之后用 `_vertical_continuity_filter(...)` 去掉不连续噪声。

当前白线检测里还有两个补救分支：

- `_detect_rrl_horizontal_supplement(...)`
  - 专门补那种近似横着出现的白线。
  - 通过 Hough 种子线和亮度/对比度约束，避免把杂乱高亮区域都捞进来。
- `_recover_rrl_upper_adjacent_segments(...)`
  - 专门补像 `rrl_00157` 这种“已经连续但高度太短、被最终清理误删”的上方弯线段。

#### 2) 其他语义

- `detect_rrl_checker(...)`
  - 检测黑白棋盘格。
- `detect_rrl_road_surface(...)`
  - 检测深色柏油路面。
- `detect_rrl_colored_bg(...)`
  - 检测彩色背景，比如蓝色/绿色区域。

#### 3) 整体重渲染

- `transform_rrl_to_newtrack(...)`
  - 把白线转蓝。
  - 把柏油路面改成 NT 的白布风格。
  - 把背景、棋盘格分别重绘成更接近 NT 的外观。
  - 绿车区域在优先级分配阶段（`assigned`）最先被排除，不参与任何语义渲染，最后回填原始像素。

#### 4) 轻量模式

- `transform_rrl_lines_only(...)`
  - 只把白线混成蓝色，不重绘整张图。
  - 调试白线 mask 时很有用。

### 5.4 当前特点

- 优点：
  - 对复杂白线几何最鲁棒。
  - 支持整图风格重渲染，结果通常更像 NT 的整体气质。
- 代价：
  - 规则最多，维护复杂度最高。
  - 某些特殊帧需要定向补偿，例如横线、上方短弯段、棋盘格干扰等。

### 5.5 命令行用法

```bash
python -m module.RRL2NewTrack \
  --raw_dir data/scene_samples/rrl/raw \
  --out_dir data/scene_samples/rrl/RRL2NewTrack \
  --mode full \
  --save_masks
```

如果只想看线条改色：

```bash
python -m module.RRL2NewTrack \
  --raw_dir data/scene_samples/rrl/raw \
  --out_dir data/scene_samples/rrl/RRL2NewTrack_lines \
  --mode lines
```

## 6. 三者怎么选

如果只是看“当前最像 NT 的视觉方向”：

1. `WS2NewTrack.py`
2. `RRL2NewTrack.py`
3. `GT2NewTrack.py`

如果是看“维护难度”：

1. `GT2NewTrack.py` 最简单
2. `WS2NewTrack.py` 中等
3. `RRL2NewTrack.py` 最复杂

如果是看“对特殊线形的适应能力”：

1. `RRL2NewTrack.py` 最强
2. `WS2NewTrack.py`
3. `GT2NewTrack.py`

## 7. 绿色车辆检测集成

三个脚本共用 `module/green_vehicle_detect.py` 中的 `GreenVehicleDetector`，各自维护一个模块级单例 `_green_detector`。

| 脚本 | 集成位置 | 方式 |
|------|---------|------|
| `GT2NewTrack.py` | `process_directory()` 循环，`recolor_lines` 之后 | 检测 → 回填原始像素 |
| `WS2NewTrack.py` | `transform_ws_to_newtrack()` 返回前 | 检测 → 回填原始像素 |
| `RRL2NewTrack.py` | `transform_rrl_to_newtrack()` 语义分配阶段 + 返回前 | 绿车优先进入 `assigned`（不参与任何渲染）→ 回填原始像素 |

检测器默认参数：HSV H∈[35,85]、S≥75、V≥60，CLOSE 3×3 + OPEN 2×2，最小面积 25 px，屏蔽顶部 15%。
在 285 张 obstacle 测试图上 **0 漏检**。

如需调整灵敏度，修改 `_green_detector = GreenVehicleDetector(min_area=..., s_lo=...)` 即可。

## 8. 后续维护建议

- `WS` 优先关注：
  - 白块转暖色中心线的完整性和近端宽度控制。
  - 蓝线和中心线的饱和度是否继续贴近 NT。
- `GT` 优先关注：
  - 草地/场外黄色误检。
  - 白线远处连续性。
- `RRL` 优先关注：
  - 横线、上方短弯段、棋盘格附近的误检和漏检平衡。
- 绿车检测优先关注：
  - 如场景中出现其他绿色物体（草地、绿色标牌），适当收紧 `s_lo` 或 `roi_top_frac`。

## 9. 一句话总结

- `WS`：语义最强，颜色最像 NT，但依赖目标统计。
- `GT`：结构最直，适合稳定批处理，但风格表达相对保守。
- `RRL`：规则最复杂，但对难例白线最能打。
- `green_vehicle_detect`：三者共用的绿车保留层，确保障碍物不被道路渲染吞掉。
