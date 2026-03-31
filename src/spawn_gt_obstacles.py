#!/usr/bin/env python3
"""
兼容脚本：实际障碍车 preset / fleet 逻辑已下沉到 `module/obstacle.py`。

默认会在赛道范围内随机启动 2 台障碍车；
两台初始位置最少相隔 `3.0` 个 sim/world 坐标单位。
可通过 `--scene ws` 切到 waveshare。
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
import types
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = REPO_ROOT / "module"


def _load_obstacle_module():
    pkg_name = "_obstacle_runtime_cli"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(MODULE_DIR)]
    sys.modules[pkg_name] = pkg

    for name in ("track", "obstacle"):
        spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", MODULE_DIR / f"{name}.py")
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module: {name}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
    return sys.modules[f"{pkg_name}.obstacle"]


def _parse_layout(raw: str) -> List[Tuple[float, float]]:
    layout: List[Tuple[float, float]] = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid layout item: {item!r}")
        progress_s, lateral_s = item.split(":", 1)
        layout.append((float(progress_s), float(lateral_s)))
    return layout


def _format_pose(pose) -> str:
    if pose is None:
        return "pose=None"
    return (
        f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
        f"yaw={pose.yaw_deg:.1f}, speed={pose.speed:.3f}"
    )


def main() -> int:
    obstacle_mod = _load_obstacle_module()

    parser = argparse.ArgumentParser(description="Spawn GT / WS obstacle fleet")
    parser.add_argument("--scene", type=str, default="gt", choices=["gt", "ws"])
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9091)
    parser.add_argument("--layout", type=str, default="", help="Comma-separated progress:lateral pairs")
    parser.add_argument("--count", type=int, default=2, help="Random obstacle count when --layout is empty")
    parser.add_argument(
        "--min-separation-world",
        type=float,
        default=3.0,
        help="Minimum obstacle spacing in sim/world coordinates when --layout is empty",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible placement")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds to keep alive; 0 means until Ctrl-C")
    args = parser.parse_args()

    layout = _parse_layout(args.layout) if args.layout.strip() else None
    fleet = obstacle_mod.spawn_preset_obstacle_fleet(
        scene=args.scene,
        host=args.host,
        port=args.port,
        layout=layout,
        count=args.count,
        min_separation_world=args.min_separation_world,
        seed=args.seed,
    )
    try:
        for i, (target, pose) in enumerate(zip(fleet.targets, fleet.get_obstacle_poses()), start=1):
            print(
                f"[placed {i}] scene={fleet.preset.name} "
                f"target=({target.x:.3f}, {target.z:.3f}, yaw={target.yaw_deg:.1f}) "
                f"pose=({_format_pose(pose)})"
            )

        started = time.time()
        print(f"[ready] {len(fleet.cars)} obstacle cars active in {fleet.preset.scene_key}. Press Ctrl-C to stop.")
        while True:
            if args.duration > 0.0 and (time.time() - started) >= float(args.duration):
                break
            for i, (pose, err) in enumerate(zip(fleet.get_obstacle_poses(), fleet.last_errors()), start=1):
                if err:
                    print(f"[status {i}] error={err}")
                else:
                    print(f"[status {i}] {_format_pose(pose)}")
            print("---")
            time.sleep(2.0)
    except KeyboardInterrupt:
        pass
    finally:
        fleet.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
