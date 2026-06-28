# V17 Safety Gate and Shadow Validation

This document summarizes the endpoint safety validation used for the V17 Jetson runtime.

## Safety principle

The runtime separates deployment validation from active autonomy validation.

Shadow validation means:

- the V17 policy runs on the Jetson;
- inference outputs, latency, LiDAR state, RP2040 state, telemetry, and safety counters are logged;
- the policy output does not control the vehicle;
- the actuator path remains under manual/user control.

This prevents an unvalidated actor backend from directly taking over the car.

## Preflight checks

The runtime performs fail-fast checks before the vehicle loop starts.

Preflight checks include:

- V17 model path exists;
- TensorRT engine path exists;
- TensorRT metadata exists and is parseable;
- metadata declares expected inputs and outputs;
- metadata shape matches the deployed V17 actor contract;
- TensorRT engine can be deserialized;
- engine binding names and shapes match metadata.

On failure, the runtime exits before the vehicle loop starts and writes a preflight report when possible.

## Startup safety gates

The endpoint runtime supports explicit startup gates:

| Gate | Purpose |
|---|---|
| `require_lidar` | Require LiDAR data before running active validation |
| `require_rp2040` | Require RP2040 / vehicle sensor data |
| `max_lidar_age_ms` | Reject stale LiDAR data |
| `max_rp2040_age_ms` | Reject stale vehicle-sensor data |
| `max_inference_ms` | Track or block slow inference |

Default intent:

- active mode should be conservative;
- shadow mode can be permissive for diagnostics and fault injection;
- safety counters should still be logged in shadow mode.

## Runtime safety monitor

The runtime safety monitor tracks:

- inference latency;
- LiDAR missing / stale state;
- RP2040 missing / stale state;
- safety block status;
- last observed sensor ages.

In shadow mode, violations are counted and logged but do not take over the actuator path. In active mode, violations can force safe output and stop the vehicle loop.

## Fault injection coverage

| Fault scenario | Expected behavior | Result |
|---|---|---|
| Engine missing | Exit before vehicle loop | Pass |
| Metadata missing | Exit before vehicle loop | Pass |
| RP2040 missing with `require_rp2040` | Exit before vehicle loop, no default serial-port bypass | Pass |
| LiDAR disabled with `require_lidar` | Exit before vehicle loop | Pass |
| LiDAR stale with `max_lidar_age_ms=350` | Exit before vehicle loop | Pass |
| Inference timeout | Count in shadow, block or stop in active path | Pass for counter path |
| Shadow non-takeover | Keep actuator path manual/user | Pass |

## Final 20-minute TensorRT shadow result

The final shadow run validated long-duration observability and runtime stability.

| Metric | Value |
|---|---:|
| Planned duration | 1200 s |
| Actual duration | 1199.55 s |
| Exit code | 0 |
| Backend | TensorRT FP16 actor |
| Control mode | shadow |
| Frames logged | 1985 |
| DataCollector p99 | 9.87 ms |

Safety counters:

| Counter | Value |
|---|---:|
| `safety_blocked` | false |
| `inference_timeout_count` | 0 |
| `lidar_missing_count` | 0 |
| `lidar_stale_count` | 0 |
| `rp2040_missing_count` | 0 |

## What this validates

The safety and shadow work validates:

- endpoint runtime stability;
- deployment preflight coverage;
- sensor availability checks;
- stale-sensor blocking logic;
- failure visibility through summary fields;
- non-blocking logging behavior;
- shadow-mode non-takeover.

## What this does not validate

This does not validate:

- obstacle avoidance success;
- active closed-loop control quality;
- lap completion;
- policy generalization;
- safe public-road autonomy.

The project should not claim those results unless a separate active validation protocol is run and documented.
