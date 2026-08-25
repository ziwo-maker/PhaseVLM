from __future__ import annotations

from projection_train.action_space import ACTION_LIST
from projection_train.joint_counterfactual_simexpert import (
    adjacent_intersection_pairs,
    compose_joint_counterfactual_actions,
    force_joint_target_actions,
    intersection_grid_position,
    joint_min_marginals,
    select_joint_pairs,
)


def test_force_joint_target_actions_overrides_only_the_two_endpoints() -> None:
    assert force_joint_target_actions([0, 1, 2, 3], 1, "ELWL", 3, "ETWT") == [0, 2, 2, 0]


def test_joint_counterfactual_action_composition_separates_initial_and_continuation() -> None:
    source_actions = [0, 2, 3, 1]
    snapshot_actions = [3, 1, 0, 2]
    initial = compose_joint_counterfactual_actions(
        source_actions,
        snapshot_actions,
        1,
        3,
        neighbor_mode="responsive",
        first_action_name="ELWL",
        second_action_name="ETWT",
    )
    responsive = compose_joint_counterfactual_actions(
        source_actions,
        snapshot_actions,
        1,
        3,
        neighbor_mode="responsive",
    )
    frozen = compose_joint_counterfactual_actions(
        source_actions,
        snapshot_actions,
        1,
        3,
        neighbor_mode="frozen",
    )

    assert initial == [3, 2, 0, 0]
    assert responsive == source_actions
    assert frozen == [3, 2, 0, 1]


def test_grid_adjacency_and_pressure_selection_are_deterministic() -> None:
    ids = ["intersection_1_1", "intersection_1_2", "intersection_2_1", "intersection_2_2", "invalid"]
    assert intersection_grid_position("intersection_2_3") == (2, 3)
    assert intersection_grid_position("intersection_a_3") is None
    assert adjacent_intersection_pairs(ids) == [
        ("intersection_1_1", "intersection_1_2"),
        ("intersection_1_1", "intersection_2_1"),
        ("intersection_1_2", "intersection_2_2"),
        ("intersection_2_1", "intersection_2_2"),
    ]
    assert select_joint_pairs(
        ids,
        pressures={"intersection_1_2": 8.0, "intersection_2_2": 9.0},
        max_pairs=1,
    ) == [("intersection_1_2", "intersection_2_2")]


def test_joint_min_marginals_project_a_complete_4x4_cost_table() -> None:
    pair_costs = {
        (first_action, second_action): float(10 * first_index + second_index)
        for first_index, first_action in enumerate(ACTION_LIST)
        for second_index, second_action in enumerate(ACTION_LIST)
    }
    first_costs, second_costs, first_partners, second_partners = joint_min_marginals(pair_costs)

    assert first_costs == {action: float(10 * index) for index, action in enumerate(ACTION_LIST)}
    assert second_costs == {action: float(index) for index, action in enumerate(ACTION_LIST)}
    assert first_partners == {action: "ETWT" for action in ACTION_LIST}
    assert second_partners == {action: "ETWT" for action in ACTION_LIST}
