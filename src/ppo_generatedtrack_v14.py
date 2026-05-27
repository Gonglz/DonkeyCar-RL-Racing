#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V14 facade entry (compatible):
- keep original module path: ppo_generatedtrack_v14.py
- re-export modularized symbols from module/v14_*.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_repo_root = str(REPO_ROOT)
while _repo_root in sys.path:
    sys.path.remove(_repo_root)
sys.path.insert(0, _repo_root)

from module.v14_attention import ChannelAttention, SpatialAttention, CBAMBlock, AttentionCNN
from module.v14_racing_line import RacingLineComputer
from module.v14_stage_profiles import (
    StageConfigV14,
    StageTrainProfileV14,
    StageEvalGateV14,
    STAGES_V14,
    STAGE_TRAIN_PROFILES_V14,
    STAGE_EVAL_GATES_V14,
    apply_stage_train_profile_v14,
)
from module.v14_distillation import PolicyDistillationManager
from module.v14_wrapper import GeneratedTrackV14Wrapper
from module.v14_curriculum import CurriculumManagerV14
from module.v14_callbacks import V14ControlTBCallback
from module.v14_env_factory import _query_generated_track_cache_from_env, create_v14_env_and_npcs
from module.v14_train import RecurrentPPO, train_v14
from module.v14_cli import parse_args, main


__all__ = [
    "ChannelAttention", "SpatialAttention", "CBAMBlock", "AttentionCNN",
    "RacingLineComputer",
    "StageConfigV14", "StageTrainProfileV14", "StageEvalGateV14",
    "STAGES_V14", "STAGE_TRAIN_PROFILES_V14", "STAGE_EVAL_GATES_V14",
    "apply_stage_train_profile_v14",
    "PolicyDistillationManager",
    "GeneratedTrackV14Wrapper",
    "CurriculumManagerV14",
    "V14ControlTBCallback",
    "_query_generated_track_cache_from_env", "create_v14_env_and_npcs",
    "RecurrentPPO", "train_v14",
    "parse_args", "main",
]


if __name__ == "__main__":
    main()
