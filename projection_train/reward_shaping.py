from __future__ import annotations

from typing import Dict, Iterable, Mapping


OBJECTIVE_QUEUE = "queue"
OBJECTIVE_WAIT = "wait"
OBJECTIVE_TRAVEL = "travel"

REWARD_MODE_LEGACY_ABSOLUTE = "legacy_absolute"
REWARD_MODE_STABLE_DELTA = "stable_delta"
REWARD_MODE_CHOICES = (
    REWARD_MODE_LEGACY_ABSOLUTE,
    REWARD_MODE_STABLE_DELTA,
)

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


def _clip(value: float, lower: float, upper: float) -> float:
    return float(min(float(upper), max(float(lower), float(value))))


def _positive_scale(value: float, fallback: float = 1.0) -> float:
    resolved = abs(float(value))
    if resolved <= 1e-8:
        return float(fallback)
    return resolved


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
    wait_delta: float | None = None,
    reward_mode: str = REWARD_MODE_LEGACY_ABSOLUTE,
    queue_delta_scale: float = 4.0,
    wait_delta_scale: float = 30.0,
    wait_abs_coef: float = 0.0,
    wait_abs_scale: float = 60.0,
    active_delta_scale: float = 4.0,
    throughput_scale: float = 2.0,
    reward_clip: float = 3.0,
    global_reward_scale: float = 1.0,
) -> Dict[str, float]:
    resolved_mode = str(reward_mode or REWARD_MODE_LEGACY_ABSOLUTE).strip().lower()
    if resolved_mode not in REWARD_MODE_CHOICES:
        raise ValueError(f"Unsupported reward_mode: {reward_mode!r}. Expected one of {REWARD_MODE_CHOICES}.")

    if resolved_mode == REWARD_MODE_STABLE_DELTA:
        queue_term = -_clip(
            float(queue_delta) / _positive_scale(queue_delta_scale),
            -1.0,
            1.0,
        )
        resolved_wait_delta = float(wait_value if wait_delta is None else wait_delta)
        wait_term = -_clip(
            resolved_wait_delta / _positive_scale(wait_delta_scale),
            -1.0,
            1.0,
        )
        wait_abs_term = -_clip(
            float(wait_value) / _positive_scale(wait_abs_scale),
            0.0,
            1.0,
        )
        active_vehicle_term = -_clip(
            float(active_vehicle_delta) / _positive_scale(active_delta_scale),
            -1.0,
            1.0,
        )
        throughput_term = _clip(
            float(throughput_count) / _positive_scale(throughput_scale),
            0.0,
            1.0,
        )

        # In stable_delta the sign is already correct (negative for costs we
        # want to decrease, positive for throughput).  The coefficient args
        # act as *magnitude* scalars (abs) so that e.g. --queue_reward_coef -0.7
        # makes the queue term 0.7× as strong, not -0.7× (which would flip the
        # intended direction).
        q_mag = abs(float(queue_reward_coef)) if abs(float(queue_reward_coef)) > 1e-8 else 0.0
        w_mag = abs(float(waiting_reward_coef)) if abs(float(waiting_reward_coef)) > 1e-8 else 0.0
        sv_mag = abs(float(system_vehicle_reward_coef)) if abs(float(system_vehicle_reward_coef)) > 1e-8 else 0.0
        th_mag = abs(float(throughput_reward_coef)) if abs(float(throughput_reward_coef)) > 1e-8 else 0.0

        queue_reward = float(objective_weights.get(OBJECTIVE_QUEUE, 0.0)) * q_mag * queue_term
        wait_reward = float(objective_weights.get(OBJECTIVE_WAIT, 0.0)) * w_mag * (
            wait_term + float(wait_abs_coef) * wait_abs_term
        )
        system_vehicle_reward = (
            float(objective_weights.get(OBJECTIVE_TRAVEL, 0.0))
            * sv_mag * 0.5
            * active_vehicle_term
        )
        throughput_reward = (
            float(objective_weights.get(OBJECTIVE_TRAVEL, 0.0))
            * th_mag * 0.5
            * throughput_term
        )
        travel_reward = float(system_vehicle_reward + throughput_reward)
        raw_reward = (
            float(queue_reward)
            + float(wait_reward)
            + float(travel_reward)
            - float(switch_penalty)
            - float(starvation_penalty)
        )
        scaled_reward = float(raw_reward * float(global_reward_scale))
        clip_value = abs(float(reward_clip))
        reward = _clip(scaled_reward, -clip_value, clip_value) if clip_value > 0.0 else scaled_reward
        return {
            "queue_reward": float(queue_reward),
            "wait_reward": float(wait_reward),
            "system_vehicle_reward": float(system_vehicle_reward),
            "throughput_reward": float(throughput_reward),
            "travel_reward": float(travel_reward),
            "raw_reward": float(raw_reward),
            "scaled_reward": float(scaled_reward),
            "clipped_reward": float(reward),
            "reward": float(reward),
            "queue_term": float(queue_term),
            "wait_term": float(wait_term),
            "wait_abs_term": float(wait_abs_term),
            "active_vehicle_term": float(active_vehicle_term),
            "throughput_term": float(throughput_term),
            "wait_delta": float(resolved_wait_delta),
        }

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
        "raw_reward": float(reward),
        "scaled_reward": float(reward),
        "clipped_reward": float(reward),
        "reward": float(reward),
        "queue_term": float(queue_reward),
        "wait_term": float(wait_reward),
        "wait_abs_term": float(0.0),
        "active_vehicle_term": float(system_vehicle_reward),
        "throughput_term": float(throughput_reward),
        "wait_delta": float(wait_value if wait_delta is None else wait_delta),
    }
