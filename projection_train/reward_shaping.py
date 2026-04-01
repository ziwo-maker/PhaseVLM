from __future__ import annotations

from typing import Dict, Iterable, Mapping


OBJECTIVE_QUEUE = "queue"
OBJECTIVE_WAIT = "wait"
OBJECTIVE_TRAVEL = "travel"

WAIT_REDUCER_MEAN = "mean"
WAIT_REDUCER_MAX = "max"
WAIT_REDUCER_MIN = "min"
WAIT_REDUCER_MIN_POSITIVE = "min_positive"
WAIT_REDUCER_CHOICES = (
    WAIT_REDUCER_MEAN,
    WAIT_REDUCER_MAX,
    WAIT_REDUCER_MIN,
    WAIT_REDUCER_MIN_POSITIVE,
)


def resolve_objective_weights(
    queue_weight: float,
    wait_weight: float,
    travel_weight: float,
) -> Dict[str, float]:
    weights = {
        OBJECTIVE_QUEUE: float(queue_weight),
        OBJECTIVE_WAIT: float(wait_weight),
        OBJECTIVE_TRAVEL: float(travel_weight),
    }
    if any(value < 0.0 for value in weights.values()):
        raise ValueError("Objective weights must be non-negative.")

    active_weight_sum = sum(value for value in weights.values() if value > 0.0)
    active_weight_count = sum(1 for value in weights.values() if value > 0.0)
    if active_weight_count <= 0 or active_weight_sum <= 0.0:
        raise ValueError("At least one objective weight must be greater than zero.")

    # Preserve an average active weight of 1.0 so changing relative priorities
    # does not unexpectedly rescale the whole PPO reward.
    scale = float(active_weight_count) / float(active_weight_sum)
    return {name: float(value * scale) for name, value in weights.items()}


def reduce_wait_values(
    wait_values: Iterable[float],
    reducer: str = WAIT_REDUCER_MIN_POSITIVE,
) -> float:
    values = [max(0.0, float(value)) for value in wait_values]
    if not values:
        return 0.0

    resolved_reducer = str(reducer or WAIT_REDUCER_MIN_POSITIVE).strip().lower()
    if resolved_reducer == WAIT_REDUCER_MEAN:
        return float(sum(values) / len(values))
    if resolved_reducer == WAIT_REDUCER_MAX:
        return float(max(values))
    if resolved_reducer == WAIT_REDUCER_MIN:
        return float(min(values))
    if resolved_reducer == WAIT_REDUCER_MIN_POSITIVE:
        positive_values = [value for value in values if value > 0.0]
        if positive_values:
            return float(min(positive_values))
        return 0.0
    raise ValueError(
        f"Unsupported wait reducer: {reducer!r}. Expected one of {WAIT_REDUCER_CHOICES}."
    )


def compute_local_shaped_reward(
    queue_delta: float,
    wait_value: float,
    active_vehicle_delta: float,
    throughput_count: float,
    queue_reward_coef: float,
    waiting_reward_coef: float,
    system_vehicle_reward_coef: float,
    throughput_reward_coef: float,
    switch_penalty: float,
    starvation_penalty: float,
    objective_weights: Mapping[str, float],
) -> Dict[str, float]:
    queue_reward = float(objective_weights.get(OBJECTIVE_QUEUE, 0.0)) * float(queue_reward_coef) * float(queue_delta)
    wait_reward = float(objective_weights.get(OBJECTIVE_WAIT, 0.0)) * float(waiting_reward_coef) * float(wait_value)
    system_vehicle_reward = (
        float(objective_weights.get(OBJECTIVE_TRAVEL, 0.0))
        * float(system_vehicle_reward_coef)
        * float(active_vehicle_delta)
    )
    throughput_reward = (
        float(objective_weights.get(OBJECTIVE_TRAVEL, 0.0))
        * float(throughput_reward_coef)
        * float(throughput_count)
    )
    travel_reward = float(system_vehicle_reward + throughput_reward)
    reward = (
        float(queue_reward)
        + float(wait_reward)
        + float(travel_reward)
        - float(switch_penalty)
        - float(starvation_penalty)
    )
    return {
        "queue_reward": float(queue_reward),
        "wait_reward": float(wait_reward),
        "system_vehicle_reward": float(system_vehicle_reward),
        "throughput_reward": float(throughput_reward),
        "travel_reward": float(travel_reward),
        "reward": float(reward),
    }
