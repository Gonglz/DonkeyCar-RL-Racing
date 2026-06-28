# V17 ONNX / TensorRT Deployment

This note documents the deployable actor path used by the V17 Jetson runtime.

## Deployment goal

The training artifact contains more than what is needed for real-time driving. The runtime only needs deterministic actor inference:

```text
image + state + lidar + lidar_meta + recurrent h/c
    -> action + next_h + next_c
```

The exported graph deliberately excludes:

- critic / value head;
- optimizer state;
- rollout buffer state;
- SB3 training wrappers;
- stochastic sampling logic used only during training.

This makes the deployment graph smaller, fixed-shape, and easier for TensorRT 7 on Jetson Nano to parse.

## ONNX export

ONNX is used here as an exchange format, not as the runtime accelerator. The acceleration comes from the TensorRT engine built from the ONNX graph.

Export characteristics:

| Item | Value |
|---|---|
| Export scope | deterministic actor only |
| Batch | fixed batch = 1 |
| Dynamic axes | disabled |
| Opset | 11 |
| Image input | `(1, 6, 128, 128)` |
| State input | `(1, 7)` |
| LiDAR input | `(1, 144)` |
| LiDAR metadata | `(1, 2)` |
| LSTM h/c | `(2, 1, 256)` |
| Output | `action`, `next_h`, `next_c` |

Compatibility decisions:

- Use fixed input shapes to reduce TensorRT runtime complexity.
- Replace dynamic LayerNorm-style expressions with fixed-axis equivalents where needed.
- Avoid ONNX expressions that are known to be fragile on older TensorRT 7 builds.
- Keep LSTM hidden and cell states explicit so the runtime can persist recurrent state.

## TensorRT engine

The Jetson engine is built from the actor-only ONNX graph using FP16.

Example build command:

```bash
/usr/src/tensorrt/bin/trtexec \
  --onnx=/path/to/v17_actor.onnx \
  --explicitBatch \
  --workspace=512 \
  --fp16 \
  --saveEngine=/path/to/v17_actor_fp16.engine \
  --verbose
```

The engine is tied to the target Jetson software stack. If the JetPack, CUDA, TensorRT, or device class changes, rebuild the engine.

## Runtime integration

The TensorRT runtime path does the following:

1. Deserialize the TensorRT engine.
2. Validate binding names and shapes against metadata.
3. Allocate device buffers once at startup.
4. Copy host inputs to device buffers.
5. Run `execute_async_v2` on a CUDA stream.
6. Copy `action`, `next_h`, and `next_c` back to host.
7. Persist `next_h` and `next_c` as the next recurrent state.

The implementation intentionally avoids making PyCUDA a required dependency. TensorRT already executes CUDA kernels; the project uses CUDA runtime APIs directly for buffer and stream management.

## Correctness check

Before using the TensorRT backend for runtime experiments, the actor output is checked against the PyTorch actor.

| Check | Result |
|---|---:|
| PyTorch action | `[-0.029122, -0.249469, -0.026623]` |
| TensorRT action | `[-0.029739, -0.249512, -0.026794]` |
| Max absolute difference | `0.000617` |
| Tolerance | `0.02` |

The action difference is well below the deployment tolerance. This validates the TensorRT actor as a numerically aligned deployment backend for the tested input.

## Important limitation

TensorRT makes the actor backend faster, but it does not remove the full robotic runtime bottleneck by itself. End-to-end latency still includes:

- camera capture;
- semantic image preprocessing;
- LiDAR freshness;
- observation construction;
- Python runtime contention;
- DonkeyCar vehicle loop scheduling;
- logging and telemetry.

The benchmark document separates actor residual latency from full runtime latency for this reason.
