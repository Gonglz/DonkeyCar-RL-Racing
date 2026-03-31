# Dynamics Wrapper

## 概述

`dynamics_wrapper.py` 现在的模块内主版本在：

- `mysim/module/dynamics_wrapper.py`

它的作用是在动作发给 DonkeySim 之前，先做一层 sim2real 动力学对齐。  
这层对齐改的是“输入给模拟器的控制命令”，不是 Unity 内部物理参数。

## 解决的问题

DonkeySim 本体的质量、摩擦、轮胎抓地、转向几何不能直接通过 Python API 改。  
因此对齐方式不是改 simulator physics，而是：

`policy / joystick output -> dynamics wrapper -> DonkeySim`

wrapper 负责把策略输出变成更像物理车实际执行到电机和舵机上的动作。

## 当前接口

主类：

- `DynamicsAlignedGymEnv`

它保持和 `DonkeyGymEnv` 接近的接口：

- `update()`
- `run_threaded()`
- `shutdown()`

因此可以直接在 `manage.py` 里替换原来的 `DonkeyGymEnv`。

## 动作变换内容

### 转向

依次做：

1. `deadband_steer`
2. `steer_gain_ratio`
3. 一阶滞后 `steer_tau_delta`
4. 限幅到 `[-1, 1]`

### 油门

依次做：

1. `throttle_gain_ratio`
2. 一阶滞后 `throttle_tau_delta`
3. 限幅到 `[-1, 1]`

## 参数来源

wrapper 从 `dynamics_params.json` 读取参数，主要字段是：

```json
{
  "alignment": {
    "steer_gain_ratio": 0.82,
    "steer_tau_delta": 0.05,
    "throttle_gain_ratio": 1.0,
    "throttle_tau_delta": 0.0
  },
  "hardware": {
    "deadband_steer": 0.05
  }
}
```

注意：

- `steer_tau_delta` 和 `throttle_tau_delta` 在代码里会被 `max(0.0, ...)` 裁掉
- 如果 JSON 里给的是负值，运行时等价于 `0.0`

## 使用方式

### Python 直接实例化

```python
from module.dynamics_wrapper import DynamicsAlignedGymEnv

cam = DynamicsAlignedGymEnv(
    sim_path=cfg.DONKEY_SIM_PATH,
    host=cfg.SIM_HOST,
    port=9091,
    env_name=cfg.DONKEY_GYM_ENV_NAME,
    conf=cfg.GYM_CONF,
    dynamics_json=cfg.SIM_DYNAMICS_JSON,
    record_location=cfg.SIM_RECORD_LOCATION,
    record_gyroaccel=cfg.SIM_RECORD_GYROACCEL,
    record_velocity=cfg.SIM_RECORD_VELOCITY,
    record_lidar=cfg.SIM_RECORD_LIDAR,
    delay=cfg.SIM_ARTIFICIAL_LATENCY,
)
```

### `manage.py` 中切换

推荐方式：

- `SIM_DYNAMICS_ALIGNMENT = False` 时使用 `DonkeyGymEnv`
- `SIM_DYNAMICS_ALIGNMENT = True` 时使用 `DynamicsAlignedGymEnv`

这样可以在“不改训练/遥控主链”的前提下切换 sim2real 对齐层。

## 配置建议

Waveshare 对齐文档当前建议值见：

- `mysim/docs/sim2real_alignment.md`

核心动作参数建议是：

- `steer_gain_ratio ≈ 0.82`
- `steer_tau_delta ≈ 0.05`
- `deadband_steer ≈ 0.05`
- `throttle_gain_ratio = 1.0`

## 和观测对齐的关系

这个 wrapper 只解决“动作侧”问题。  
观测侧仍需要同时保证：

- `yaw_rate_norm <- gyro[1] / 4.0`
- `accel_x_norm <- accel[0] / 9.8`
- `v_long_norm <- speed / v_max`

也就是说，完整 sim2real 对齐是两层：

1. `module/dynamics_wrapper.py` 负责动作对齐
2. `module/obv.py` 负责观测对齐

## 备注

- 当前模块内这份 `dynamics_wrapper.py` 是 `mysim` 侧的主版本
- DonkeySim / DonkeyCar 源码树里若还保留旧副本，应以 `module` 里的版本为准
