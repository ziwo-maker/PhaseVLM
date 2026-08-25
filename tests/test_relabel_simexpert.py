"""Tests for SimExpert relabeling phase-transition emulation."""

from __future__ import annotations

import sys
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.relabel_simexpert import (  # noqa: E402
    YELLOW_TIME_S,
    _candidate_phase_setpoints,
)


def test_keep_current_phase_has_no_yellow() -> None:
    setpoints = _candidate_phase_setpoints(base_phase=2, target_phase=2)
    assert setpoints == [(0, 2)]
    assert all(phase != 0 for _, phase in setpoints)


def test_switch_goes_through_yellow_first() -> None:
    setpoints = _candidate_phase_setpoints(base_phase=2, target_phase=3)
    assert setpoints[0] == (0, 0)  # yellow at second 0
    assert setpoints[1] == (YELLOW_TIME_S, 3)  # target green after yellow


def test_switch_yellow_duration_matches_env() -> None:
    setpoints = _candidate_phase_setpoints(base_phase=1, target_phase=4)
    assert setpoints[1][0] == 3  # YELLOW_TIME from utils/config.py


def test_setpoints_are_typed_int() -> None:
    setpoints = _candidate_phase_setpoints(base_phase=1, target_phase=2)
    for t, p in setpoints:
        assert isinstance(t, int) and isinstance(p, int)
