# mysim_public

这个目录是从本地 `mysim` 工作区整理出来的公开版仓库，只保留适合上传到 GitHub 的核心代码与文档。

## 包含内容

- `module/`
- `docs/`
- `src/`

## 未包含内容

- `data/`
- `models/`
- `logs/`
- `backups/`
- 本地私有配置和其他实验脚本

## 说明

- `robust_lane_detector.py` 已整理到 `module/robust_lane_detector.py`，训练与观测链路统一走包内依赖。
- `src/ppo_waveshare_v13.py` 已改为优先使用仓库相对路径：
  - 赛道目录默认 `track/`
  - 配置文件默认 `myconfig.py`
- 如果你的本地路径不同，可以用环境变量覆盖：
  - `MYSIM_TRACK_DIR`
  - `MYSIM_MYCONFIG`
- 当前仓库是从更大的私有/本地实验目录中裁剪出的子集，所以未附带训练数据、模型权重和日志。

## 快速开始

在仓库根目录运行：

```bash
python -m module.GT2NewTrack --help
python -m module.RRL2NewTrack --help
python src/ppo_waveshare_v13.py --help
```
