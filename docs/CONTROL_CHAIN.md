# Control Chain (Steer + Throttle)

本文档对应当前简化版 V12/V13 控制链实现。

## 1. 代码入口

- 控制主模块：`module/control.py`
- 包含全部油门/转向控制相关 wrapper：
  - `HighLevelControlWrapper`
  - `ActionSafetyWrapper`
  - `ThrottleControlWrapper`（兼容保留）
  - `CurvatureAwareThrottleWrapper`（兼容保留，默认不启用）

说明：
- `module/wrappers.py` 仍可导入这些类（兼容旧代码），但实现主入口已经统一到 `module/control.py`。

## 2. 当前训练主链（简化）

在 `_build_v12_wrapper_chain()` 中，当前默认动作链是：

1. `HighLevelControlWrapper`
2. `ActionSafetyWrapper`
3. Base Env

不再默认叠加：
- 曲率油门收紧（`CurvatureAwareThrottleWrapper`）
- 二次油门裁切（`ThrottleControlWrapper`）

油门限幅由 `HighLevelControlWrapper` 内部统一处理。

## 3. HighLevelControlWrapper

策略输出：
- `target_steer` in `[-1, 1]`
- `target_speed_norm` in `[0, 1]`（允许倒车时 `[-1, 1]`）

控制律：

```text
v_tgt = target_speed_norm * speed_vmax
v_err = v_tgt - v_meas
i_term = clip(i_term + v_err * dt, -integral_limit, +integral_limit)
throttle = kff*v_tgt + kp*v_err + ki*i_term
```

最终限幅：
- 不允许倒车：`throttle ∈ [0, max_throttle]`
- 允许倒车：`throttle ∈ [-max_throttle, max_throttle]`

## 4. ActionSafetyWrapper（简化版）

当前策略：
- 固定 `delta_max` 速率限制
- 可选 LPF（`beta`）
- 不启用曲率/意图自适应放宽

保留旧参数（`adaptive_delta_max` / `curve_*` / `hairpin_*`）仅为兼容，不参与执行逻辑。

关键诊断：
- `smooth/rate_limit_hit`
- `smooth/steer_clip_hit`
- `smooth/hairpin_relax_active`（简化模式下恒为 0）

## 5. 奖励侧“前10步渐进惩罚”

`ImprovedRewardWrapperV3` 已改成固定 10 步 ramp：

- 出界风险触发后：
  - `near_offtrack_ramp_step` 从 1 递增到 10
  - 惩罚按 `ramp_scale = step/10` 线性增强
- 碰撞风险触发后同理：
  - `near_collision_ramp_step` 从 1 到 10 递增

新增日志（用于验证 ramp）：
- `reward_debug/near_offtrack_ratio`
- `reward_debug/near_offtrack_ramp_scale`
- `reward_debug/near_collision_ramp_scale`
- `reward_debug/r_near_offtrack`
- `reward_debug/r_near_collision`

## 6. 关于 `target_speed_norm`

当前继续使用 `target_speed_norm` 的原因：
- 数值尺度固定在 `[-1,1]`/`[0,1]`，PPO 更稳定。
- 跨地图/不同 `speed_vmax` 时不需要改策略输出尺度。
- 与旧模型、旧日志、旧训练脚本保持兼容。

如需改成“直接输出目标速度（m/s）”，可以再做一个切换开关，但会改变动作空间语义并影响旧模型兼容。
