# DonkeyRewardWrapper 奖励函数说明

## 概述

`DonkeyRewardWrapper` 是 DonkeyCar 仿真训练的统一奖励包装器，定义在 `module/reward.py`。

该包装器封装了底层 gym 环境，在每个 `step()` 调用时计算综合奖励信号，涵盖：
- 速度与生存奖励（鼓励车辆保持运动并快速行驶）
- 赛道进度奖励（鼓励沿赛道前进）
- 横向偏差（CTE）奖惩（鼓励居中行驶，惩罚超出边界）
- 姿态与速度参考惩罚（稠密引导信号）
- 出界/碰撞预警惩罚（提前惩罚危险状态）
- 终止惩罚（碰撞、出界、卡住）
- 平滑驾驶惩罚（惩罚急打方向盘）

旧名称 `ImprovedRewardWrapperV3` 和 `V9DomainRewardWrapper` 作为向后兼容别名保留，指向同一个类。

---

## CTE 边界符号约定

### 有符号横偏距 `lat_err_cte`

- **有轨迹几何时（精确）**：`lat_err_cte = lat_err * coord_scale`
  - `lat_err` 来自 `TrackGeometryManager.query()`，正值表示车辆偏向赛道左侧
  - `coord_scale`（默认 8.0）将归一化横偏量转换为与 `_SCENE_CTE_TABLE` 同单位的量
- **无几何时（近似 fallback）**：`lat_err_cte = -sim_cte`
  - `sim_cte` 为模拟器原始 CTE，符号取反以保持左正右负约定

### 边界选边逻辑

```
is_left_side = (lat_err_cte >= 0)
cte_boundary     = |cte_left|     if is_left_side else |cte_right|
cte_out_boundary = |cte_left_out| if is_left_side else |cte_right_out|
```

- `cte_left > 0`（左侧在界边界），`cte_right < 0`（右侧在界边界，数值为负）
- `cte_left_out > cte_left`（左侧出界边界），`cte_right_out < cte_right`（右侧出界边界）
- 所有 CTE 奖惩比较均使用 `lat_err_cte_abs = |lat_err_cte|` 与 `cte_boundary` 或 `cte_out_boundary` 比较

---

## 奖励项（正向）

### 1. `survival` — 生存奖励

```
speed_gate = clip(speed / 0.5, 0, 1)
survival   = survival_reward_scale * speed_gate
```

- 速度门控：低速（< 0.5 m/s）时奖励线性减少，静止时为 0
- 鼓励车辆保持运动状态

### 2. `speed` — 速度奖励

```
v_normalized  = clip(speed / 4.0, 0, 1)
cte_norm      = lat_err_cte_abs / cte_boundary
center_factor = clip(1 - cte_norm^2, 0, 1)
speed         = ontrack * v_normalized * center_factor
```

额外加成（满足以下全部条件时 +0.2）：
- `ontrack = 1`（在界内）
- `lat_err_cte_abs < 0.6 * cte_boundary`（偏差不超过边界 60%）
- `v_normalized > 0.3`（速度超过 1.2 m/s）

在界时鼓励快速且居中行驶；超界后 ontrack=0，速度奖励清零。

### 3. `progress` — 赛道进度奖励

```
progress_ratio        = 按赛道弧长的有向进度（前进正，后退负），clip 至 [-0.02, 0.02]
curve_ratio           = clip(kappa_abs / progress_kappa_ref, 0, 1)
center_factor         = 同上
progress_center_gate  = max(progress_center_gate_min, center_factor ^ progress_center_gate_power)
progress_forward_gain = 1 + progress_curve_boost * curve_ratio * center_factor
```

- 正向进度：`progress = ontrack * progress_reward_scale * progress_ratio * progress_forward_gain * progress_center_gate`
- 负向进度（后退）：`progress = ontrack * progress_reward_scale * progress_ratio`（不加门控，全额惩罚）

弯道增益（`progress_forward_gain`）受中心线因子调制，避免贴边时被高曲率额外鼓励。

### 4. `cte_in` — 在界居中奖励

```
cte_base      = 0.3 * (1 - lat_err_cte_abs / cte_boundary)
speed_gate_cte = clip(speed / 0.3, 0, 1)
cte_in        = cte_base * speed_gate_cte * cte_norm_scale
```

仅在 `lat_err_cte_abs <= cte_boundary` 时生效（在界时）。在界内越居中奖励越高，速度极低时奖励减弱。

### 5. `lap` — 完圈奖励

```
lap = 6.0 * laps_completed * lap_reward_scale
```

每次完圈奖励一次，每步最多计 1 圈（防止计数器抖动）。

---

## 惩罚项（负向）

### 6. `cte_out` — 超界惩罚

```
exceed_ratio = min((lat_err_cte_abs - cte_boundary) / cte_half_width, 2.0)
cte_out      = -(1 + 4 * exceed_ratio) * cte_norm_scale
```

仅在 `lat_err_cte_abs > cte_boundary` 时生效（超界时）。超界越深惩罚越重，最大约 `-9 * cte_norm_scale`（exceed_ratio=2 时）。

> 注：`cte_in` 与 `cte_out` 在代码中均通过同一个 `cte_term` 变量传递。

### 7. `center` — 居中惩罚

```
center = -w_center * |lat_err_norm|
```

- `lat_err_norm` 为归一化横向偏差，范围约 [-3, 3]
- 每步稠密惩罚，持续引导车辆向中心线靠拢

### 8. `heading` — 航向惩罚

```
heading = -w_heading * (heading_err_abs / pi)
```

- `heading_err_abs`：车辆朝向与赛道切向量的角度差绝对值（弧度，范围 [0, π]）
- 鼓励车辆对齐赛道行进方向

### 9. `speed_ref` — 速度参考惩罚

```
v_ref           = v_ref_max - (v_ref_max - v_ref_min) * curve_ratio_speed
curve_ratio_speed = clip(kappa_abs / speed_ref_kappa_ref, 0, 1)
speed_err_norm  = (speed - v_ref) / v_ref_max
speed_ref       = -w_speed_ref * speed_err_norm^2
```

弯道曲率越大，参考速度越低（从 `v_ref_max` 线性降至 `v_ref_min`）。惩罚对速度偏差平方，在弯道中过快或过慢均受惩罚。

### 10. `time` — 时间惩罚

```
time = -w_time
```

每步固定惩罚，促使策略尽快完成任务。

### 11. `near_offtrack` — 出界预警惩罚

```
cte_out_ratio_risk    = lat_err_cte_abs / cte_out_boundary
near_offtrack_ratio   = clip((cte_out_ratio_risk - near_offtrack_start_ratio) / (1 - near_offtrack_start_ratio), 0, 1)
near_offtrack_ramp    = 线性增长（触发后从第1步到第10步线性达到满惩罚）
near_offtrack_penalty = -w_near_offtrack * near_offtrack_ratio * ramp_scale
```

- 当 CTE 超过 out 边界的 `near_offtrack_start_ratio` 比例时开始惩罚
- 渐进 10 步线性增强，防止突然大惩罚
- 风险评估基于**真实 out 边界**（不随 done 阈值课程放宽）

### 12. `near_collision` — 碰撞预警惩罚

```
proxy_collision_risk  = clip(0.55 * near_collision_ratio + 0.20 * heading_risk + 0.15 * speed_risk + 0.10 * control_risk, 0, 1)
near_collision_risk   = proxy_collision_risk * ramp_scale   （无障碍物信号时）
                      = clip(0.65 * obstacle_risk + 0.35 * proxy_collision_risk, 0, 1) * ramp_scale  （有障碍物信号时）
near_collision_penalty = -w_near_collision * near_collision_risk
```

综合考虑：
- CTE 接近边界程度（`near_collision_ratio`）
- 航向偏差风险（`heading_risk`）
- 速度风险（`speed_risk`）
- 转向控制风险（`control_risk`）
- 障碍物距离风险（优先使用，若 info 中有障碍物信息）

### 13. `smooth` — 平滑惩罚（转向变化）

```
curve_penalty_scale = max(0.35, 1 - smooth_curve_relief * curve_ratio)
smooth = -w_d * |Δsteer_exec| * curve_penalty_scale
```

- `Δsteer_exec`：连续两帧执行转向量的差值
- 弯道中适度降低惩罚（`smooth_curve_relief`），避免策略在发夹弯"怕转向"

### 14. `jerk` — 抖动惩罚

```
jerk = -w_dd * |Δsteer_t - Δsteer_{t-1}| * curve_penalty_scale
```

惩罚转向速率的变化量（转向加速度），抑制方向盘抖动。

### 15. `mismatch` — 指令偏差惩罚

```
mismatch = -w_m * |steer_raw - steer_exec| * curve_penalty_scale
```

- `steer_raw`：策略输出的原始转向指令
- `steer_exec`：经安全包装器处理后实际执行的转向量
- 惩罚两者之间的偏差，促使策略学习安全约束范围内的动作

### 16. `sat` — 速率饱和惩罚

```
sat = -w_sat * rate_excess_bounded
```

- `rate_excess_bounded`：转向速率超出限制的程度（已归一化）
- 来自 `ActionSafetyWrapper.diag["rate_excess_bounded"]`

### 17. `throttle_high` — 高油门惩罚

```
throttle_high = -throttle_penalty_amount * (1 + speed_norm)   （当 throttle_cmd > throttle_penalty_threshold 时）
speed_norm = clip(speed / 4.0, 0, 2)
```

- 仅当执行油门超过阈值时触发
- 速度越高，高油门惩罚越重

---

## 终止条件

### 碰撞终止

- 触发条件：`info["hit"] != "none"`
- 效果：`done = True`，`collision_penalty -= collision_penalty_base`

### 出界终止（offtrack）

- 触发条件：连续 3 步满足 `lat_err_cte_abs >= effective_out`
- `effective_out = cte_out_boundary * leniency`（课程宽松系数，前期 > 1.0，后期 = 1.0）
- 效果：`done = True`，`collision_penalty -= offtrack_penalty_base`

### 卡住终止（stuck）

- 触发条件：`ontrack = 1` 且 `speed < 0.1` 连续 30+ 步
- 效果：`done = True`，附加递增惩罚 `min(3.0, 0.3 * (stuck_counter - 30))`

---

## 出界阈值课程（offtrack leniency）

前期训练中放宽 done 触发阈值，帮助策略从容学习基础控制：

```
leniency_steps = total_timesteps * offtrack_leniency_ratio
```

- 在前 `leniency_steps` 步内：`leniency` 从 `offtrack_leniency_mult` 线性降至 1.0
- 之后：`leniency = 1.0`（正常阈值）

**重要**：CTE 惩罚（`cte_term`）始终基于真实边界（不受 leniency 影响）；只有 done 触发阈值随课程放宽。

---

## CTE 归一化说明

各赛道宽度不同，为使奖惩量级在不同赛道间保持一致，引入 `cte_norm_scale`：

```
CTE_REF_HALF_WIDTH = 4.6
cte_half_width     = (cte_left - cte_right) / 2     # 始终为正值
cte_norm_scale     = cte_half_width / CTE_REF_HALF_WIDTH
```

- 窄赛道（`cte_half_width < 4.6`）：`cte_norm_scale < 1`，缩小奖惩幅度
- 宽赛道（`cte_half_width > 4.6`）：`cte_norm_scale > 1`，放大奖惩幅度
- 参考宽度 4.6 对应典型 WaveShare 赛道半宽

`cte_half_width` 参数在初始化时由调用方根据 `cte_left`/`cte_right` 或场景标定表传入。

---

## 边界逻辑流程（step 内）

1. 从 `info` 读取 `cte`（模拟器原始）、`speed`、`hit`、`lap_count`
2. 若有轨迹几何：调用 `track_geometry.query()` 获取精确 `lat_err`，计算 `lat_err_cte = lat_err * coord_scale`
3. 否则：`lat_err_cte = -cte`（近似 fallback）
4. `lat_err_cte_abs = |lat_err_cte|`，`is_left_side = (lat_err_cte >= 0)`
5. 根据 `is_left_side` 选取本侧的 `cte_boundary` 和 `cte_out_boundary`
6. `ontrack = float(lat_err_cte_abs < cte_boundary)`
7. 计算所有奖惩项，汇总为 `total_reward`
8. 检测 collision / offtrack / stuck → 触发 done

---

## 完整参数列表

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `total_timesteps` | int | 200000 | 总训练步数（用于课程计算） |
| `action_safety_wrapper` | object | None | ActionSafetyWrapper 引用，用于读取平滑惩罚诊断 |
| `w_d` | float | 0.0 | 转向变化（smooth）惩罚权重 |
| `w_dd` | float | 0.0 | 转向抖动（jerk）惩罚权重 |
| `w_m` | float | 0.0 | 转向指令偏差（mismatch）惩罚权重 |
| `w_sat` | float | 0.0 | 速率饱和（sat）惩罚权重 |
| `w_time` | float | 0.0 | 时间惩罚权重（每步固定扣分） |
| `w_center` | float | 0.0 | 居中惩罚权重 |
| `w_heading` | float | 0.0 | 航向惩罚权重 |
| `w_speed_ref` | float | 0.0 | 速度参考惩罚权重 |
| `speed_ref_vmin` | float | 0.35 | 弯道最低参考速度（m/s） |
| `speed_ref_vmax` | float | 2.2 | 直道最高参考速度（m/s） |
| `speed_ref_kappa_ref` | float | 0.15 | 速度参考曲率归一化参考值 |
| `lap_reward_scale` | float | 1.0 | 完圈奖励缩放系数 |
| `progress_reward_scale` | float | 80.0 | 进度奖励缩放系数 |
| `progress_curve_boost` | float | 0.6 | 弯道进度增益系数 |
| `progress_kappa_ref` | float | 0.15 | 进度弯道增益曲率归一化参考值 |
| `progress_center_gate_min` | float | 0.08 | 进度中心门控最小值（防止贴边策略） |
| `progress_center_gate_power` | float | 1.0 | 进度中心门控指数 |
| `smooth_curve_relief` | float | 0.5 | 弯道中平滑惩罚减弱系数 |
| `throttle_penalty_threshold` | float | 0.50 | 高油门惩罚触发阈值 |
| `throttle_penalty_amount` | float | 0.005 | 高油门每步基础惩罚量 |
| `survival_reward_scale` | float | 0.2 | 生存奖励缩放系数 |
| `collision_penalty_base` | float | 8.0 | 碰撞终止惩罚基础值 |
| `offtrack_penalty_base` | float | 6.0 | 出界终止惩罚基础值 |
| `w_near_offtrack` | float | 0.40 | 出界预警惩罚权重 |
| `near_offtrack_start_ratio` | float | 0.70 | 出界预警开始触发比例（占 out 边界） |
| `w_near_collision` | float | 0.35 | 碰撞预警惩罚权重 |
| `near_collision_start_ratio` | float | 0.65 | 碰撞预警开始触发比例 |
| `cte_left` | float | 5.0 | 左侧在界 CTE 边界（正值） |
| `cte_right` | float | -5.0 | 右侧在界 CTE 边界（负值） |
| `cte_left_out` | float\|None | None | 左侧出界 CTE 边界（默认 cte_left * 1.1） |
| `cte_right_out` | float\|None | None | 右侧出界 CTE 边界（默认 cte_right * 1.1） |
| `coord_scale` | float | 8.0 | lat_err 转换为 CTE 表同单位的缩放系数 |
| `offtrack_leniency_ratio` | float | 0.15 | 出界课程覆盖训练步数比例 |
| `offtrack_leniency_mult` | float | 1.8 | 出界课程初始宽松倍数 |
| `track_geometry` | object | None | TrackGeometryManager 实例（提供精确 lat_err） |
| `scene_key` | str | "" | 当前场景键名（用于查询轨迹几何） |
| `logging_key` | str | "" | 日志前缀标识 |
| `cte_half_width` | float | 4.6 | CTE 奖惩归一化参考半宽（默认等于 CTE_REF_HALF_WIDTH） |
| `enable_step_diagnostics` | bool | False | 是否打印前若干步的诊断日志 |
| `step_diagnostics_first_steps` | int | 3 | 每个 episode 打印诊断的前 N 步 |
| `step_diagnostics_every_episodes` | int | 0 | 每隔 N 个 episode 打印诊断（0=每次） |
