from __future__ import annotations

import pytest

from projection_train.reward_shaping import (
    OBJECTIVE_QUEUE,
    OBJECTIVE_TRAVEL,
    OBJECTIVE_WAIT,
    REWARD_MODE_STABLE_DELTA,
    WAIT_REDUCER_MAX,
    WAIT_REDUCER_MEAN,
    WAIT_REDUCER_MIN,
    WAIT_REDUCER_MIN_POSITIVE,
    compute_local_shaped_reward,
    reduce_wait_values,
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
        wait_value=0.0,
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
    assert result["reward"] > 0.0


def test_wait_reward_penalizes_absolute_delay() -> None:
    weights = resolve_objective_weights(queue_weight=0.0, wait_weight=1.0, travel_weight=0.0)
    low_wait = compute_local_shaped_reward(
        queue_delta=0.0,
        wait_value=10.0,
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
    high_wait = compute_local_shaped_reward(
        queue_delta=0.0,
        wait_value=20.0,
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
    assert low_wait["wait_reward"] < 0.0
    assert high_wait["wait_reward"] < low_wait["wait_reward"]
    assert high_wait["reward"] < low_wait["reward"]


def test_stable_delta_wait_ignores_absolute_delay_when_delta_is_zero() -> None:
    weights = resolve_objective_weights(queue_weight=0.0, wait_weight=1.0, travel_weight=0.0)
    low_absolute_wait = compute_local_shaped_reward(
        queue_delta=0.0,
        wait_value=30.0,
        wait_delta=0.0,
        active_vehicle_delta=0.0,
        throughput_count=0.0,
        queue_reward_coef=-0.35,
        waiting_reward_coef=-0.01,
        system_vehicle_reward_coef=-0.01,
        throughput_reward_coef=0.02,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
        reward_mode=REWARD_MODE_STABLE_DELTA,
    )
    high_absolute_wait = compute_local_shaped_reward(
        queue_delta=0.0,
        wait_value=600.0,
        wait_delta=0.0,
        active_vehicle_delta=0.0,
        throughput_count=0.0,
        queue_reward_coef=-0.35,
        waiting_reward_coef=-0.01,
        system_vehicle_reward_coef=-0.01,
        throughput_reward_coef=0.02,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
        reward_mode=REWARD_MODE_STABLE_DELTA,
    )
    assert low_absolute_wait["wait_reward"] == pytest.approx(0.0)
    assert high_absolute_wait["wait_reward"] == pytest.approx(low_absolute_wait["wait_reward"])
    assert high_absolute_wait["reward"] == pytest.approx(low_absolute_wait["reward"])


def test_stable_delta_wait_delta_penalizes_growth_and_rewards_reduction() -> None:
    weights = resolve_objective_weights(queue_weight=0.0, wait_weight=1.0, travel_weight=0.0)
    growing_wait = compute_local_shaped_reward(
        queue_delta=0.0,
        wait_value=600.0,
        wait_delta=30.0,
        active_vehicle_delta=0.0,
        throughput_count=0.0,
        queue_reward_coef=-0.35,
        waiting_reward_coef=-0.01,
        system_vehicle_reward_coef=-0.01,
        throughput_reward_coef=0.02,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
        reward_mode=REWARD_MODE_STABLE_DELTA,
        wait_delta_scale=30.0,
    )
    falling_wait = compute_local_shaped_reward(
        queue_delta=0.0,
        wait_value=600.0,
        wait_delta=-30.0,
        active_vehicle_delta=0.0,
        throughput_count=0.0,
        queue_reward_coef=-0.35,
        waiting_reward_coef=-0.01,
        system_vehicle_reward_coef=-0.01,
        throughput_reward_coef=0.02,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
        reward_mode=REWARD_MODE_STABLE_DELTA,
        wait_delta_scale=30.0,
    )
    assert growing_wait["wait_reward"] == pytest.approx(-1.0)
    assert falling_wait["wait_reward"] == pytest.approx(1.0)
    assert falling_wait["reward"] > growing_wait["reward"]


def test_stable_delta_reward_is_clipped_for_extreme_inputs() -> None:
    weights = resolve_objective_weights(queue_weight=1.0, wait_weight=1.0, travel_weight=1.0)
    result = compute_local_shaped_reward(
        queue_delta=-1000.0,
        wait_value=0.0,
        wait_delta=-1000.0,
        active_vehicle_delta=-1000.0,
        throughput_count=1000.0,
        queue_reward_coef=-0.35,
        waiting_reward_coef=-0.01,
        system_vehicle_reward_coef=-0.01,
        throughput_reward_coef=0.02,
        switch_penalty=0.0,
        starvation_penalty=0.0,
        objective_weights=weights,
        reward_mode=REWARD_MODE_STABLE_DELTA,
        reward_clip=0.5,
    )
    assert result["raw_reward"] > 0.5
    assert result["reward"] == pytest.approx(0.5)


def test_selection_weights_change_training_objective_mix() -> None:
    weights = resolve_objective_weights(queue_weight=2.0, wait_weight=1.0, travel_weight=2.0)
    result = compute_local_shaped_reward(
        queue_delta=-1.0,
        wait_value=1.0,
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
    assert result["wait_reward"] == pytest.approx(-0.6)
    assert result["system_vehicle_reward"] == pytest.approx(1.2)
    assert result["throughput_reward"] == pytest.approx(1.2)


def test_reduce_wait_values_supports_multiple_reducers() -> None:
    values = [0.0, 5.0, 20.0]
    assert reduce_wait_values(values, reducer=WAIT_REDUCER_MEAN) == pytest.approx(25.0 / 3.0)
    assert reduce_wait_values(values, reducer=WAIT_REDUCER_MAX) == pytest.approx(20.0)
    assert reduce_wait_values(values, reducer=WAIT_REDUCER_MIN) == pytest.approx(0.0)
    assert reduce_wait_values(values, reducer=WAIT_REDUCER_MIN_POSITIVE) == pytest.approx(5.0)


def test_reduce_wait_values_handles_empty_and_invalid_input() -> None:
    assert reduce_wait_values([], reducer=WAIT_REDUCER_MIN_POSITIVE) == pytest.approx(0.0)
    assert reduce_wait_values([0.0, 0.0], reducer=WAIT_REDUCER_MIN_POSITIVE) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        reduce_wait_values([1.0, 2.0], reducer="unsupported")
