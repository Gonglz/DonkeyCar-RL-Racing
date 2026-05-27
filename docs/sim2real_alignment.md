# Sim2Real 动力学对齐

Waveshare 赛道，2026-03-22。物理车数据来自 Jetson `manage.py drive --js` + RP2040 传感器板，模拟器数据来自 `donkey-waveshare-v0`。

## 1. 原始数据对比

### 1.1 采集概况

|  | 物理车 | 模拟器 |
|--|--------|--------|
| 记录数 | 2556 | 1664 |
| 时长 | 138.6s | 139.8s |
| 采样率 | 18.4 Hz | 11.9 Hz |
| 赛道 | Waveshare 实体地图 | donkey-waveshare-v0 |

### 1.2 转向和油门命令

```
                          物理车                     模拟器
                    min     max    mean   std     min     max    mean   std
user/angle       -1.000  1.000   0.295  0.762  -1.000  1.000  -0.070 0.481
user/throttle    -0.800  0.800   0.367  0.589   0.300  0.500   0.399 0.070
```

### 1.3 陀螺仪各轴 (rad/s)

```
模拟器 (Unity Y-up 坐标系):
  gyro_x: std=0.0635  range=[-0.656, 0.641]     ← 横滚 (roll)
  gyro_y: std=0.6641  range=[-2.055, 2.071]     ← 偏航 (yaw) ★
  gyro_z: std=0.0277  range=[-0.368, 0.356]     ← 俯仰 (pitch)

物理车 RP2040 (Z-up 坐标系):
  gyro_x: 横滚 (roll)
  gyro_y: 俯仰 (pitch)
  gyro_z: std=0.978   range=[-1.930, 2.475]     ← 偏航 (yaw) ★
           符号反转: 正转向 → 负 gyro_z
```

**关键发现**: 模拟器偏航角速度在 `gyro[1]`（gyro_y），不是 `gyro[2]`（gyro_z）。原因是 Unity 使用 Y-up 坐标系。

### 1.4 偏航角速度对比（修正轴后）

```
                                     min      max     mean     std
物理车 yaw_rate  (-gyro_z)        -2.475    1.930    0.287   0.978
模拟器 yaw_rate  (gyro_y)         -2.055    2.071   -0.050   0.664
  物理车/模拟器 std 比值                                      1.47x
```

### 1.5 转向增益

仅统计 `|steering| > 0.1` 且车在运动的样本：

```
|yaw_rate| / |steering|  (每单位转向命令产生的偏航角速度):
  物理车:   1.0723 rad/s  (1722 个样本)
  模拟器:   1.3095 rad/s  (1244 个样本)
  比值:     0.819
```

**结论**: 物理车同样转向命令产生的偏航角速度是模拟器的 82%。

### 1.6 加速度计 (m/s^2)

```
                     物理车                     模拟器
                 min     max   mean   std    min     max   mean   std
accel_x       -3.493  3.518  0.177  0.920  -3.868  4.089 -0.010 0.671
accel_z        0.176 14.305  9.219  1.166  -1.105  1.966  0.743 0.319
```

- `accel_x`: 范围接近，物理车噪声大 1.4 倍
- `accel_z`: 物理车 ~9.2（包含重力），模拟器 ~0.74（不含重力分量）— 坐标系差异

### 1.7 速度（单位不同）

```
物理车 speed_enc:  mean=84.5   max=120.0  (编码器 ticks)
模拟器 speed:      mean=1.029  max=2.720  (模拟器单位)
  换算比例:  1 模拟器单位 ≈ 78 编码器 ticks
```

---

## 2. Bug 修复: obv.py 偏航角速度轴错误

**文件**: `mysim/module/obv.py`, `_build_state_v13()`

**修复前**（错误）:
```python
gz = float(gyro[2])     # gyro_z — 在模拟器中几乎为零 (std=0.03)
yaw_rate_norm = clip(gz / 4.0, -2, 2)   # 实际上始终约等于 0
```

**修复后**（正确）:
```python
gy = float(gyro[1])     # gyro_y — 模拟器中真正的偏航角速度 (std=0.66)
yaw_rate_norm = clip(gy / 4.0, -2, 2)   # 有意义的偏航反馈
```

**影响**: 修复前 RL 策略训练时 yaw_rate_norm 输入几乎为零，策略无法学到转向反馈。

---

## 3. 对齐架构

DonkeySim 物理引擎封装在 Unity 二进制中，无法通过 API 修改质量、摩擦力、轮胎抓地力、转弯半径。对齐通过两层变换实现：

```
 策略输出                动作对齐                  模拟器
 [steer, throttle] --> dynamics_wrapper.py ------> DonkeySim
                        - steer × 0.82 (增益缩放)
                        - 一阶滞后滤波 (响应延迟)
                        - 死区处理

 模拟器输出             观测对齐                   状态向量
 info["gyro","speed"] --> obv.py ----------------> state(7,)
                          - gyro[1] 取偏航角速度
                          - speed / v_max 归一化
                          - accel[0] / 9.8 归一化
```

### 3.1 动作对齐 (`module/dynamics_wrapper.py`)

发送给模拟器之前变换动作，让模拟器的车**行为**像物理车：

| 参数 | 值 | 来源 |
|------|-----|------|
| `steer_gain_ratio` | **0.82** | 驾驶数据中物理车/模拟器偏航增益比 |
| `steer_tau_delta` | ~0.05s | 物理车舵机响应延迟（bench test 拟合） |
| `throttle_gain_ratio` | 1.0 | 速度单位不可比，保持 1:1 |
| `hw_deadband` | ~0.05 | 物理车转向死区 |

`myconfig.py` 配置:
```python
SIM_DYNAMICS_ALIGNMENT = True
SIM_DYNAMICS_JSON = "dynamics_data/params/dynamics_params.json"
```

### 3.2 观测对齐 (obv.py)

让模拟器和物理车的状态向量**数值分布**一致：

| state[i] | 模拟器来源 | 物理车来源 | 归一化 |
|----------|-----------|-----------|--------|
| v_long_norm | `info["speed"]` / 2.2 | `rp2040/speed_enc` / 120 | v_max 不同 |
| yaw_rate_norm | `info["gyro"][1]` / 4.0 | `-rp2040/gyro_z` / 4.0 | 轴映射 + 取反 |
| accel_x_norm | `info["accel"][0]` / 9.8 | `rp2040/accel_x` / 9.8 | 单位相同 |
| prev_steer | safety wrapper 输出 | `user/angle` | 已在 [-1,1] |
| prev_throttle | adapter 输出 | `user/throttle` | 已在 [-1,1] |
| steer_core | adapter 积分器 | `user/angle` | 手动驾驶 ≈ 直接转向 |
| bias_smooth | adapter LPF | 0.0 | 手动驾驶无 line_bias |

### 3.3 GYM_CONF（不能修复动力学）

只控制动作空间边界，不影响物理：

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `steer_limit` | 1.0 | 仅限制动作范围 |
| `throttle_min/max` | 0.0 / 1.0 | 仅限制动作范围 |
| `max_cte` | 8.0 | episode 终止条件 |
| `frame_skip` | 1 | 控制频率 |

### 3.4 域随机化（可选，提高泛化能力）

| 项目 | 范围 | 目的 |
|------|------|------|
| 陀螺仪/加速度计高斯噪声 | +50% 物理车噪声水平 | 物理车传感器噪声大 1.4 倍 |
| `steer_gain_ratio` 抖动 | ±10% | 适应车辆个体差异 |
| 控制延迟抖动 | ±20ms | 适应真实世界延迟波动 |

---

## 4. 相关文件

| 文件 | 用途 |
|------|------|
| `mysim/module/obv.py` | 状态向量构建（偏航轴修复） |
| `mysim/module/dynamics_wrapper.py` | 动作侧对齐 wrapper |
| `src/donkeycar/.../rp2040_sensor.py` | RP2040 传感器 DonkeyCar Part |
| `tools/dynamics_id.py` | 从驾驶数据做系统辨识 |
| `tools/dynamics_verify.py` | 验证图表和 RMSE 指标 |
| `mysim/myconfig.py` | `HAVE_RP2040`、`SIM_DYNAMICS_ALIGNMENT` 开关 |

## 5. 实车部署备忘

在物理车上部署模拟器训练的策略时，传感器适配层需要映射：

```
info["speed"]    ← rp2040/speed_enc  (或转换为模拟器单位: enc / 78)
info["gyro"][1]  ← -rp2040/gyro_z    (取反 + 轴重映射)
info["accel"][0] ← rp2040/accel_x    (直接使用，单位相同)
```

`v_max` 在模拟器 (2.2) 和物理车 (120) 之间不同，两种处理方式：
- (A) 在适配层将物理车速度转换为模拟器单位: `speed_sim = speed_enc / 78`
- (B) 模拟器和物理车使用不同的 `v_max` 配置

---

## 6. GT 地图米制施工图

`generated_track` 的施工图已导出到：

- `mysim/docs/track_profiles/generated_track_construction_annotated.jpg`

当前这版施工图按下面的工作假设标注：

- **1 sim unit ≈ 1 m**

这是基于 DonkeySim / Unity 场景几何常见尺度做的工程化假设，方便先施工和排版。  
如果后续拿到 Unity 场景的实测比例或人工测绘值，应再按真实比例整体缩放一次。

按该假设，`generated_track` 的关键尺寸为：

| 项目 | 数值 |
|------|------|
| 总体 X 向跨度 | **6.515 m** |
| 总体 Z 向跨度 | **11.920 m** |
| 中心线单圈长度 | **26.862 m** |
| 总赛道宽度 | **1.150 m** |
| 中线到左边界 | **0.825 m** |
| 中线到右边界 | **0.325 m** |

说明：

- GT 地图不是对称中线，左右边界宽度不同。
- 施工图中角度标注来自中心线高曲率节点的切向变化。
- 图中所有长度标注目前都已按“米”显示，但本质上仍然来源于 simulator X-Z 几何。
