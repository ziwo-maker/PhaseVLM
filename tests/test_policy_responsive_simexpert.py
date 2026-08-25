from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from projection_train.policy_responsive_simexpert import (
    RollingSequenceBank,
    compose_counterfactual_actions,
    force_target_action,
    load_snapshot_phase_indices,
    normalize_sequence,
    synchronize_intersection_phase,
)


def test_normalize_sequence_pads_clips_and_sanitizes_values() -> None:
    sequence = normalize_sequence(
        [[1, "2", None], [3, 4, float("nan")], [5, 6, 7]],
        seq_len=4,
        input_dim=3,
    )
    assert sequence == [[1.0, 2.0, 0.0], [1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 7.0]]


def test_load_snapshot_phases_maps_yellow_or_invalid_to_safe_base(tmp_path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                "trafficLights": {
                    "intersection_1_1": {"curPhaseIndex": 3},
                    "intersection_1_2": {"curPhaseIndex": 0},
                    "virtual": {"curPhaseIndex": 4},
                }
            }
        ),
        encoding="utf-8",
    )
    assert load_snapshot_phase_indices(str(path)) == {
        "intersection_1_1": 3,
        "intersection_1_2": 1,
    }


def test_synchronize_intersection_phase_clears_branch_local_state() -> None:
    intersection = SimpleNamespace(
        current_phase_index=-1,
        previous_phase_index=-1,
        next_phase_to_set_index=3,
        current_phase_duration=99,
        all_yellow_flag=True,
        all_red_flag=True,
        flicker=1,
        dic_lane_vehicle_previous_step={"x": ["vehicle"]},
        dic_lane_vehicle_previous_step_in={"x": ["vehicle"]},
        dic_lane_waiting_vehicle_count_previous_step={"x": 1},
        dic_vehicle_speed_previous_step={"vehicle": 0.0},
        dic_vehicle_distance_previous_step={"vehicle": 1.0},
        dic_lane_vehicle_current_step={"x": ["vehicle"]},
        dic_lane_vehicle_current_step_in={"x": ["vehicle"]},
        dic_lane_waiting_vehicle_count_current_step={"x": 1},
        dic_vehicle_speed_current_step={"vehicle": 0.0},
        dic_vehicle_distance_current_step={"vehicle": 1.0},
        list_lane_vehicle_previous_step_in=["vehicle"],
        list_lane_vehicle_current_step_in=["vehicle"],
        dic_vehicle_arrive_leave_time={"vehicle": {}},
    )
    synchronize_intersection_phase(intersection, 3)
    assert intersection.current_phase_index == 3
    assert intersection.previous_phase_index == 3
    assert intersection.current_phase_duration == 0
    assert not intersection.all_yellow_flag
    assert not intersection.all_red_flag
    assert intersection.next_phase_to_set_index is None
    assert intersection.dic_lane_vehicle_current_step == {}
    assert intersection.dic_vehicle_arrive_leave_time == {}


def test_force_target_action_preserves_source_network_response() -> None:
    assert force_target_action([0, 1, 2, 3], 2, "NTST") == [0, 1, 1, 3]


def test_counterfactual_actions_isolate_only_post_interval_neighbor_response() -> None:
    source_actions = [0, 2, 3, 1]
    snapshot_actions = [3, 1, 0, 2]

    initial = compose_counterfactual_actions(
        source_actions,
        snapshot_actions,
        1,
        neighbor_mode="responsive",
        target_action_name="ETWT",
    )
    responsive = compose_counterfactual_actions(
        source_actions,
        snapshot_actions,
        1,
        neighbor_mode="responsive",
    )
    frozen = compose_counterfactual_actions(
        source_actions,
        snapshot_actions,
        1,
        neighbor_mode="frozen",
    )

    assert initial == [3, 0, 0, 2]
    assert responsive == source_actions
    assert frozen == [3, 2, 0, 2]
    assert frozen[1] == responsive[1]
    assert frozen[0] == snapshot_actions[0]
    assert frozen[2:] == snapshot_actions[2:]


def test_rolling_sequence_bank_reads_only_new_cityflow_bins() -> None:
    records = [
        {"intersection_id": "intersection_1_1", "sequences": [[1, 2], [3, 4]]},
        {"intersection_id": "intersection_1_2", "sequences": [[5, 6], [7, 8]]},
    ]
    bank = RollingSequenceBank.from_records(
        records,
        ["intersection_1_1", "intersection_1_2"],
        seq_len=3,
        input_dim=2,
        interval_seconds=2,
        snapshot_time_s=4,
    )
    counts = np.zeros((5, 4), dtype=np.float32)
    counts[3] = [10, 11, 20, 21]
    counts[4] = [12, 13, 22, 23]
    env = SimpleNamespace(
        traffic_counts_by_interval={2: counts},
        traffic_intersection_ids=["intersection_1_1", "intersection_1_2"],
        get_current_time=lambda: 8,
    )
    assert bank.advance_from_cityflow(env) == 2
    assert bank.sequence("intersection_1_1") == [[3.0, 4.0], [10.0, 11.0], [12.0, 13.0]]
    assert bank.sequence("intersection_1_2") == [[7.0, 8.0], [20.0, 21.0], [22.0, 23.0]]
    assert bank.advance_from_cityflow(env) == 0
