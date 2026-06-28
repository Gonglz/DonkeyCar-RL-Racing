# Jetson Runtime

This folder contains edge-deployment helpers for the DonkeyCar / Jetson runtime experiments.

The strongest endpoint-deployment path is the V17 runtime, which combines camera input, LiDAR sector features, vehicle-state telemetry, recurrent actor inference, logging, and safety gates.

## Runtime components

| Component | Purpose |
|---|---|
| `runtime_monitor.py` | DonkeyCar runtime monitor, shadow/active wrapper, DataCollector logging, LiDAR reader, Jetson telemetry, safety preflight/gates |
| `v17_pilot.py` | V17 pilot wrapper, semantic observation construction, PyTorch/TensorRT backend selection |
| `v17_trt_runtime.py` | TensorRT actor runtime using CUDA runtime APIs for buffer/stream execution |
| `summarize_shadow_run.py` | CSV-to-summary utility for shadow validation runs |

Some files may be sanitized or excluded from the public repository if they contain local lab paths, model checkpoints, raw logs, or private hardware configuration.

## V17 runtime path

```text
CSI camera
  -> semantic image preprocessing
ROS /scan LiDAR
  -> 72 sector ranges + 72 valid-mask values
RP2040 / vehicle sensors
  -> state vector
image + state + lidar + lidar_meta + LSTM h/c
  -> V17 actor backend
  -> action + next LSTM state
  -> action adapter / safety gate
  -> DonkeyCar actuator path or shadow log
```

## Shadow mode

Shadow mode runs the actor and logs the output without taking actuator control.

Use it before any active run:

```bash
python runtime_monitor.py drive \
  --model /path/to/v17_model.zip \
  --type v17 \
  --js \
  --control-mode shadow \
  --shadow-duration 1200 \
  --log-dir /path/to/monitor_logs/manual_trt_shadow_20min \
  --run-label manual_trt_shadow_20min \
  --track-condition endpoint_deployment_repro \
  --shadow-engine /path/to/v17_actor_fp16.engine \
  --shadow-engine-metadata /path/to/v17_actor_export.json \
  --force-recording
```

## TensorRT smoke test

A TensorRT runtime smoke test should validate that the engine and metadata can be loaded and executed:

```bash
PYTHONPATH=/path/to/tools python tools/check_v17_trt_runtime.py \
  --engine /path/to/v17_actor_fp16.engine \
  --metadata /path/to/v17_actor_export.json
```

## PyTorch vs TensorRT action diff

Run an action-diff check before using TensorRT as a deployment backend:

```bash
PYTHONPATH=/path/to/tools python tools/compare_v17_torch_trt.py \
  --model /path/to/v17_model.zip \
  --engine /path/to/v17_actor_fp16.engine \
  --metadata /path/to/v17_actor_export.json \
  --pilot /path/to/v17_pilot.py \
  --tolerance 0.02
```

A representative checked result:

| Item | Value |
|---|---:|
| PyTorch action | `[-0.029122, -0.249469, -0.026623]` |
| TensorRT action | `[-0.029739, -0.249512, -0.026794]` |
| Max abs diff | `0.000617` |
| Tolerance | `0.02` |

## Safety gates

The runtime supports safety gates for endpoint validation:

| Option | Purpose |
|---|---|
| `--require-lidar` | Require LiDAR data before entering the loop |
| `--require-rp2040` | Require vehicle sensor data before entering the loop |
| `--max-lidar-age-ms` | Reject stale LiDAR data |
| `--max-rp2040-age-ms` | Reject stale RP2040 data |
| `--max-inference-ms` | Count or block slow inference |

Active mode should use conservative safety gates. Shadow mode is used for validation and diagnostics before actuator takeover is considered.

## Logged evidence

A complete shadow run should keep:

- `command.txt`;
- `run_context.txt`;
- `runtime.log`;
- `run_*.csv`;
- `summary.json`;
- `preflight_report.json`, if generated;
- `DONE` or `exit_code.txt`.

Large raw logs and model checkpoints should not be committed to Git. Use summarized metrics and small redacted examples instead.
