"""Tests for SimExpert determinism comparison (verify_simexpert_determinism)."""

from __future__ import annotations

import sys
import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.verify_simexpert_determinism import compare_relabel_records  # noqa: E402

from projection_train.action_space import ACTION_LIST  # noqa: E402


def _record(expert="ETWT", base="NTST", margin=0.1, wait=1.0,
            feasible=None) -> dict:
    if feasible is None:
        feasible = {a: True for a in ACTION_LIST}
    return {
        "time_s": 30.0,
        "intersection_id": "intersection_1_1",
        "expert_action": expert,
        "base_action": base,
        "expert_action_margin": margin,
        "candidate_action_waits": {a: wait for a in ACTION_LIST},
        "candidate_action_metrics": {a: {"att": 1.0, "aql": 2.0} for a in ACTION_LIST},
        "pareto_feasible": feasible,
        "constraint_violations": {a: [] for a in ACTION_LIST},
    }


def test_identical_records_are_bitwise_deterministic() -> None:
    a = [_record()]
    b = [_record()]
    rep = compare_relabel_records(a, b)
    assert rep["bitwise_deterministic"] is True
    assert rep["n_expert_action_mismatch"] == 0


def test_label_flip_is_detected() -> None:
    a = [_record(expert="ETWT")]
    b = [_record(expert="NTST")]
    rep = compare_relabel_records(a, b)
    assert rep["n_expert_action_mismatch"] == 1
    assert rep["bitwise_deterministic"] is False


def test_small_cost_diff_within_tolerance_is_ignored() -> None:
    a = [_record(wait=1.0)]
    b = [_record(wait=1.0 + 1e-9)]
    rep = compare_relabel_records(a, b, cost_atol=1e-6, cost_rtol=1e-6)
    assert rep["n_cost_diff"] == 0
    assert rep["bitwise_deterministic"] is True


def test_large_cost_diff_is_flagged() -> None:
    a = [_record(wait=1.0)]
    b = [_record(wait=2.0)]
    rep = compare_relabel_records(a, b, cost_atol=1e-6, cost_rtol=1e-6)
    assert rep["n_cost_diff"] == len(ACTION_LIST)
    assert rep["bitwise_deterministic"] is False


def test_feasibility_mask_mismatch_is_detected() -> None:
    fa = {a: True for a in ACTION_LIST}
    fb = dict(fa)
    fb["ETWT"] = False
    a = [_record(feasible=fa)]
    b = [_record(feasible=fb)]
    rep = compare_relabel_records(a, b)
    assert rep["n_feasibility_mismatch"] == 1
    assert rep["bitwise_deterministic"] is False


def test_near_tie_detection() -> None:
    a = [_record(margin=1e-9)]  # below tie_atol
    rep = compare_relabel_records(a, a)
    assert rep["n_near_tie"] == 1
    a2 = [_record(margin=0.5)]
    rep2 = compare_relabel_records(a2, a2)
    assert rep2["n_near_tie"] == 0


def test_record_count_mismatch_reported() -> None:
    rep = compare_relabel_records([_record(), _record()], [_record()])
    assert "error" in rep
