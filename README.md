# DonkeyCar RL Racing

Portfolio showcase for reinforcement learning in DonkeyCar simulation. This repository contains the simulator-side training code, environment wrappers, curriculum logic, track-processing utilities, and deployment helpers used to train obstacle-aware autonomous driving policies.

## Project Highlights

- Multi-scene PPO training pipeline for DonkeyCar simulator tracks, including Waveshare-style narrow tracks and generated-track domains.
- Curriculum learning for obstacle avoidance, progressing from warmup driving to static obstacles, mixed dynamic obstacles, and lane-following NPC vehicles.
- Custom observation and control stack: semantic lane extraction, multi-input observations, high-level speed control, steering safety limits, and reward shaping.
- Sim-to-real support code for lane/track transformation, robust visual lane detection, green obstacle-vehicle detection, and Jetson runtime monitoring.
- Reproducibility-oriented docs covering reward design, map transforms, multisim training, dynamics wrappers, and V16 curriculum gates.

## Main Entry Points

| Path | Purpose |
|---|---|
| `src/ppo_multitrack_v16.py` | Current dual-domain obstacle curriculum training entrypoint. |
| `module/v14_train.py` and `module/v14_cli.py` | Modular V14 training flow and command-line interface. |
| `module/multi_scene_env.py` | Multi-scene environment switching and observation wrapper stack. |
| `module/reward.py` and `module/v14_wrapper_reward.py` | Reward shaping for progress, safety, off-track risk, and obstacle avoidance. |
| `module/control.py` | High-level steering/throttle control wrappers and safety limits. |
| `module/WS2NewTrack.py`, `module/GT2NewTrack.py`, `module/RRL2NewTrack.py` | Track image conversion and lane/road surface preprocessing tools. |
| `Jetson/runtime_monitor.py` | Runtime monitoring helper for edge deployment experiments. |

## Repository Structure

```text
.
├── src/                  # Training scripts and experiment entrypoints
├── module/               # RL environments, wrappers, rewards, control, perception utilities
├── module/track_data/    # Manual-width track geometry profiles
├── docs/                 # Design notes for reward, control, map transforms, sim-to-real, curriculum
├── tools/                # World-model training and evaluation utilities
├── Jetson/               # Edge runtime helper scripts
├── config.py             # DonkeyCar simulator configuration
├── myconfig.py           # Project-specific runtime configuration
└── train.py / manage.py  # Standard DonkeyCar project entrypoints
```

## Recommended Training Command

The current showcase path is the V16 dual-domain curriculum:

```bash
python src/ppo_multitrack_v16.py \
  --auto-curriculum \
  --sim remote \
  --port 9091 \
  --track-dir /path/to/track \
  --steps 6000000 \
  --save-dir models/v16_auto_curriculum \
  --exp-tag v16_auto \
  --file-metrics-log-freq 250
```

See `docs/V16_CURRICULUM.md` for stage gates, domain settings, and expected logs.

## Design Notes

- `docs/reward.md` explains the progress/safety reward terms and near-risk ramping behavior.
- `docs/CONTROL_CHAIN.md` documents the steering and throttle wrapper sequence.
- `docs/MULTISIM_TRAINING.md` covers the multi-scene training setup.
- `docs/sim2real_alignment.md` and `docs/lane_extraction.md` describe the perception and domain-alignment work.
- `module/README.md` is a detailed index of the modules, classes, and top-level functions.

## Scope

Large generated training outputs are intentionally not included: simulator logs, model checkpoints, tub data, generated models, and local Unity logs should stay outside Git. The archived development history remains available at https://github.com/Gonglz/DonkeyCar-RL-Racing-archive.
