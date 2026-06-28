# V17 Runtime Benchmark

This document consolidates the public benchmark story for the V17 Jetson endpoint runtime. It separates model-backend latency from full robotic runtime latency.

## Why separate actor latency from full runtime latency

TensorRT accelerates the actor network. A physical robot loop also contains camera capture, semantic preprocessing, LiDAR freshness, observation assembly, logging, telemetry, and DonkeyCar scheduling. Therefore:

- actor residual latency measures the model backend;
- V17 latency measures the pilot path;
- vehicle loop latency measures the runtime loop;
- DataCollector latency measures logging overhead.

Reporting only model-only inference would overstate the system-level result.

## Baseline bottleneck

Before the runtime optimizations, the main bottleneck was not only the actor backend. The high-latency path included:

- building LiDAR sector features from raw ranges in the main pilot path;
- synchronous DataCollector writes;
- slow Jetson telemetry reads from sysfs/proc/I2C paths;
- camera and semantic-image preprocessing;
- Python runtime and vehicle-loop jitter.

## LiDAR / DataCollector / telemetry optimization

The largest full-runtime improvement came from moving LiDAR sectorization out of the critical path, writing logs asynchronously, and caching telemetry.

| Metric | TensorRT baseline | Optimized TensorRT | Change |
|---|---:|---:|---:|
| V17 latency p50 | 168.570 ms | 87.153 ms | -48.3% |
| V17 latency p95 | 253.958 ms | 124.753 ms | -50.9% |
| Effective FPS mean | 4.270 | 10.641 | +149.2% |
| Vehicle loop p50 | 192.050 ms | 87.800 ms | -54.3% |
| Vehicle loop p95 | 696.175 ms | 130.800 ms | -81.2% |
| DataCollector p99 | 537.42 ms | 5.67 ms | -98.9% |

Interpretation:

- p50 and FPS improved mostly because LiDAR sectorization was moved out of the main loop.
- p95 improved mostly because DataCollector writes and telemetry reads stopped blocking the vehicle loop.
- LiDAR scan age did not materially improve because that depends on the LiDAR / ROS data source, not only the sectorization compute location.

## Optimized PyTorch vs TensorRT backend A/B

After LiDAR/DataCollector/telemetry optimization, the actor backend comparison becomes visible.

| Metric | Optimized PyTorch | Optimized TensorRT | Change |
|---|---:|---:|---:|
| V17 latency p50 | 106.325 ms | 94.603 ms | -11.0% |
| V17 latency p95 | 202.382 ms | 199.075 ms | -1.6% |
| Effective FPS mean | 7.964 | 9.476 | +19.0% |
| Loop dt p50 | 107.100 ms | 95.500 ms | -10.8% |
| Loop dt p95 | 203.020 ms | 199.820 ms | -1.6% |
| GPU load mean | 13.719% | 12.398% | -9.6% |

Actor residual split:

| Metric | PyTorch p50 | TensorRT p50 | Change |
|---|---:|---:|---:|
| Preprocess | 79.995 ms | 81.413 ms | +1.8% |
| Actor residual | 24.325 ms | 12.474 ms | -48.7% |

| Metric | PyTorch p95 | TensorRT p95 | Change |
|---|---:|---:|---:|
| Preprocess | 178.629 ms | 187.059 ms | +4.7% |
| Actor residual | 39.913 ms | 17.215 ms | -56.9% |

Interpretation:

- TensorRT roughly halves actor residual latency.
- End-to-end p50 improves, but p95 remains dominated by semantic preprocessing and runtime jitter.
- This is the key systems lesson: backend acceleration is necessary but not sufficient for real-time robotic control.

## Final 20-minute TensorRT shadow run

The final reproducibility run validates runtime stability and observability rather than active closed-loop autonomy.

| Metric | Value |
|---|---:|
| Duration | 1199.55 s |
| Frames logged | 1985 |
| Effective FPS mean | 5.030 |
| V17 latency p50 | 237.113 ms |
| V17 latency p95 | 281.205 ms |
| V17 latency p99 | 303.734 ms |
| V17 latency max | 555.380 ms |
| Loop dt p50 | 242.100 ms |
| Loop dt p95 | 286.900 ms |
| Loop dt p99 | 309.448 ms |
| LiDAR scan age p50 | 274.600 ms |
| LiDAR scan age p95 | 351.220 ms |
| LiDAR scan age p99 | 454.944 ms |
| CPU load mean | 77.810% |
| GPU load mean | 7.772% |
| Power in mean | 4172.058 mW |
| DataCollector p99 | 9.87 ms |

Safety counters in the final run:

| Counter | Value |
|---|---:|
| `safety_blocked` | false |
| `inference_timeout_count` | 0 |
| `lidar_missing_count` | 0 |
| `lidar_stale_count` | 0 |
| `rp2040_missing_count` | 0 |

## Remaining bottlenecks

The remaining latency bottlenecks are:

1. semantic image preprocessing;
2. LiDAR scan age / freshness;
3. Python runtime scheduling and vehicle-loop jitter;
4. camera input and memory-copy overhead;
5. TensorRT I/O overhead, likely smaller than the visual frontend issue.

## Benchmark conclusion

The strongest result is not simply "TensorRT is faster." The stronger result is:

> The runtime was profiled as a system, actor backend latency was separated from end-to-end control latency, and the largest full-loop improvement came from moving sensor/logging/telemetry work out of the critical path.

That is the intended hiring signal for Robotics Software / Edge AI Systems roles.
