from __future__ import annotations

import pytest

from projection_train.reward_shaping import (
    OBJECTIVE_QUEUE,
    OBJECTIVE_TRAVEL,
    OBJECTIVE_WAIT,
    compute_local_shaped_reward,
    resolve_objective_weights,
)


def test_resolve_objective_weights_preserves_average_scale() -> None:
    weights = resolve_objective_weights(queue_weight=2.0, wait_weight=1.0, travel_weight=2.0)
    assert weights[OBJECTIVE_QUEUE] == pytest.approx(1.2)
    assert weights[OBJECTIVE_WAIT] == pytest.approx(0.6)
    assert weights[OBJECTIVE_TRAVEL] == pytest.approx(1.2)


def test_resolve_objective_weights_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        resolve_objective_weights(queue_weight=-1.0, wait_weight=1.0, travel_weight=1.0)
    with pytest.raises(ValueError):
        resolve_objective_weights(queue_weight=0.0, wait_weight=0.0, travel_weight=0.0)


def test_negative_penalty_coefficients_reward_improvements() -> None:
    weights = resolve_objective_weights(queue_weight=1.0, wait_weight=1.0, travel_weight=0.0)
    result = compute_local_shaped_reward(
        queue_delta=-5.0,
        wait_delta=-2.0,
        active_vehicle_delta=0.0,
        throughput_count=0.0,
        queue_reward_coef=-0.35,
        waiting_reward_coef=-0.01,
        system_vehicle_reward_coef=-0.01,
        throughput_reward_coef=0.02,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
    )
    assert result["queue_reward"] > 0.0
    assert result["wait_reward"] > 0.0
    assert result["reward"] > 0.0


def test_selection_weights_change_training_objective_mix() -> None:
    weights = resolve_objective_weights(queue_weight=2.0, wait_weight=1.0, travel_weight=2.0)
    result = compute_local_shaped_reward(
        queue_delta=-1.0,
        wait_delta=-1.0,
        active_vehicle_delta=-1.0,
        throughput_count=1.0,
        queue_reward_coef=-1.0,
        waiting_reward_coef=-1.0,
        system_vehicle_reward_coef=-1.0,
        throughput_reward_coef=1.0,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
    )
    assert result["queue_reward"] == pytest.approx(1.2)
    assert result["wait_reward"] == pytest.approx(0.6)
    assert result["system_vehicle_reward"] == pytest.approx(1.2)
    assert result["throughput_reward"] == pytest.approx(1.2)
