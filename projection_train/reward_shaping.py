from __future__ import annotations

from typing import Dict, Mapping


OBJECTIVE_QUEUE = "queue"
OBJECTIVE_WAIT = "wait"
OBJECTIVE_TRAVEL = "travel"


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


def compute_local_shaped_reward(
    queue_delta: float,
    wait_delta: float,
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
    wait_reward = float(objective_weights.get(OBJECTIVE_WAIT, 0.0)) * float(waiting_reward_coef) * float(wait_delta)
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
