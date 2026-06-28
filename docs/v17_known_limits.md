# V17 Known Limits

This document records the limits of the V17 endpoint deployment so the repository does not overstate the project.

## Main boundary

The endpoint-deployment result is valid for:

- ONNX/TensorRT actor deployment;
- Jetson runtime integration;
- shadow-mode validation;
- LiDAR/DataCollector/telemetry runtime optimization;
- safety preflight and gate behavior;
- logging and summary reproducibility.

It is not evidence that the learned policy can reliably avoid obstacles or complete active closed-loop laps on the physical car.

## Do not claim

Do not claim:

- V17 has validated real-car obstacle avoidance.
- V17 has passed active closed-loop driving.
- V17 is a production autonomous-driving stack.
- The runtime reaches 20 Hz in the final 20-minute shadow test.
- TensorRT alone solved end-to-end latency.

## What the final shadow run actually says

The final 20-minute TensorRT shadow run shows that the deployment chain can run for about 20 minutes, log useful data, and preserve shadow non-takeover. It does not show that the car can drive autonomously.

Key final-run values:

| Metric | Value |
|---|---:|
| Duration | 1199.55 s |
| Effective FPS mean | 5.030 |
| V17 latency p95 | 281.205 ms |
| Loop dt p95 | 286.900 ms |
| LiDAR scan age p95 | 351.220 ms |
| DataCollector p99 | 9.87 ms |
| Safety blocked | false |

## Remaining bottlenecks

### 1. Visual semantic preprocessing

The visual frontend remains the main residual latency source. It includes resizing, color conversion, lane probability construction, obstacle probability construction, morphology, edge extraction, motion residuals, and tensor stacking.

This should be profiled and optimized without changing the semantic contract first. GPU/CUDA rewrites should not be the first step.

### 2. LiDAR freshness

Moving LiDAR sectorization out of the critical loop reduced compute overhead, but it did not materially improve LiDAR scan age. LiDAR freshness likely depends on the LiDAR device, ROS bridge, timestamp semantics, and callback design.

A future revision should separate:

- LaserScan header age;
- local receipt age;
- sectorization compute time;
- main-loop consumption age.

### 3. Active gate threshold

The final 20-minute shadow run reported LiDAR scan age p95 near 351 ms. If active mode uses `max_lidar_age_ms=350`, the threshold is conservative and may block valid runs. Before active smoke tests, either improve LiDAR freshness or tune the threshold based on measured age distribution.

### 4. PMIC telemetry

The PMIC reading reported 100 C while CPU/GPU/AO/PLL/Fan readings were not showing thermal runaway behavior. Treat this as a board-level telemetry risk or sensor-reporting issue, not direct evidence of V17 thermal failure.

### 5. ROS bridge shutdown noise

The final run observed intermittent ROS LiDAR bridge shutdown noise, but LiDAR CSV data remained available and missing counters stayed at zero. This should be cleaned up later, but it did not invalidate the endpoint-deployment result.

## Recommended next work

Priorities:

1. Add fine-grained visual-preprocess profiling.
2. Build a golden-image and action-diff regression set before changing the visual frontend.
3. Separate LiDAR header age from receipt age.
4. Run a longer 30-minute or 60-minute shadow validation if more reliability evidence is needed.
5. Re-evaluate active-mode LiDAR thresholds before any active smoke test.
6. Improve ROS bridge shutdown handling.
7. Keep policy-quality experiments separate from endpoint-deployment claims.

## Portfolio framing

The strongest public framing is:

> V17 is an endpoint-deployment and runtime-optimization case study for a multi-input recurrent robotic policy on Jetson.

The weakest public framing would be:

> V17 is an autonomous obstacle-avoidance car.

Use the first framing.
