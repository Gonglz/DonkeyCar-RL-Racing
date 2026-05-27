#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path helpers for V14 to avoid machine-specific hardcoded defaults."""

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def default_generated_track_profile():
    """Return default manual-width profile path for generated_track."""
    override = os.environ.get("V14_MANUAL_WIDTH_PROFILE", "").strip()
    if override:
        return override
    return str(REPO_ROOT / "track_profiles" / "manual_width_generated_track.json")


__all__ = ["REPO_ROOT", "default_generated_track_profile"]
