from projection_train.action_space import action_to_index
from projection_train.run_projection_online import ProjectionPolicyRunner


def _runner(max_wait_protect: float) -> ProjectionPolicyRunner:
    runner = object.__new__(ProjectionPolicyRunner)
    runner.max_phase_override_gap = 2.0
    runner.max_phase_override_left_wait_protect = 0.0
    runner.max_phase_override_max_wait_protect = max_wait_protect
    runner.max_phase_override_target_queue_floor = 0.0
    runner.max_phase_override_total_queue_floor = 0.0
    return runner


def _state() -> dict:
    return {
        "ET": {"queue_len": 2, "avg_wait_time": 30},
        "WT": {"queue_len": 2, "avg_wait_time": 30},
        "NT": {"queue_len": 5, "avg_wait_time": 10},
        "ST": {"queue_len": 5, "avg_wait_time": 10},
        "EL": {"queue_len": 0, "avg_wait_time": 0},
        "WL": {"queue_len": 0, "avg_wait_time": 0},
        "NL": {"queue_len": 0, "avg_wait_time": 0},
        "SL": {"queue_len": 0, "avg_wait_time": 0},
    }


def test_max_wait_protect_blocks_override_from_starved_phase() -> None:
    runner = _runner(max_wait_protect=90.0)
    indices, actions, diagnostics = runner._apply_max_phase_override(
        [_state()],
        [action_to_index("ETWT")],
        ["ETWT"],
        phase_wait_maxes=[{"ETWT": 120.0, "NTST": 10.0, "ELWL": 0.0, "NLSL": 0.0}],
    )

    assert indices == [action_to_index("ETWT")]
    assert actions == ["ETWT"]
    assert diagnostics[0]["overridden"] is False
    assert diagnostics[0]["protect_reason"].startswith("max_wait_protect:ETWT=")


def test_max_wait_protect_allows_override_below_threshold() -> None:
    runner = _runner(max_wait_protect=90.0)
    indices, actions, diagnostics = runner._apply_max_phase_override(
        [_state()],
        [action_to_index("ETWT")],
        ["ETWT"],
        phase_wait_maxes=[{"ETWT": 60.0, "NTST": 10.0, "ELWL": 0.0, "NLSL": 0.0}],
    )

    assert indices == [action_to_index("NTST")]
    assert actions == ["NTST"]
    assert diagnostics[0]["overridden"] is True
