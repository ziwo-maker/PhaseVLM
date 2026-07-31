"""Queue-priority rule controller with left-wait protection."""

from __future__ import annotations

from typing import Any, Mapping

from projection_train.action_space import ACTION_LIST, action_to_index, index_to_action

from .agent import Agent


PHASE_LANES = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
LEFT_PHASES = {"ELWL", "NLSL"}
THROUGH_PHASES = {"ETWT", "NTST"}

_LANE_INDEX = {
    "WL": 0,
    "WT": 1,
    "EL": 3,
    "ET": 4,
    "NL": 6,
    "NT": 7,
    "SL": 9,
    "ST": 10,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _phase_from_index(index: int) -> str:
    return index_to_action(max(0, min(int(index), len(ACTION_LIST) - 1)))


def action_index_from_cur_phase(cur_phase: Any, fallback: int = 0) -> int:
    """Resolve CityFlow phase id or action index to a four-phase action index."""
    try:
        phase = int(cur_phase[0] if isinstance(cur_phase, (list, tuple)) else cur_phase)
    except (TypeError, ValueError, IndexError):
        return int(fallback)
    if 1 <= phase <= len(ACTION_LIST):
        return phase - 1
    if 0 <= phase < len(ACTION_LIST):
        return int(fallback) if phase == 0 else phase
    return int(fallback)


def phase_queue_wait_from_state(state: Mapping[str, Any] | None) -> tuple[dict[str, float], dict[str, float]]:
    """Extract phase queue/wait from detailed movement state or CityFlow feature state."""
    state = state or {}
    explicit_queue = state.get("phase_queue") if isinstance(state, Mapping) else None
    explicit_wait = state.get("phase_wait") if isinstance(state, Mapping) else None
    if isinstance(explicit_queue, Mapping):
        phase_queue = {phase: _safe_float(explicit_queue.get(phase, 0.0)) for phase in ACTION_LIST}
        phase_wait = (
            {phase: _safe_float(explicit_wait.get(phase, 0.0)) for phase in ACTION_LIST}
            if isinstance(explicit_wait, Mapping)
            else {phase: 0.0 for phase in ACTION_LIST}
        )
        return phase_queue, phase_wait

    movement_queue: dict[str, float] = {}
    movement_wait: dict[str, float] = {}
    for movement in _LANE_INDEX:
        item = state.get(movement) if isinstance(state, Mapping) else None
        if isinstance(item, Mapping):
            movement_queue[movement] = _safe_float(item.get("queue_len", item.get("queue", 0.0)))
            movement_wait[movement] = _safe_float(item.get("avg_wait_time", item.get("wait", 0.0)))

    if not movement_queue:
        lane_queues = state.get("lane_num_waiting_vehicle_in", []) if isinstance(state, Mapping) else []
        if isinstance(lane_queues, (list, tuple)):
            for movement, idx in _LANE_INDEX.items():
                movement_queue[movement] = _safe_float(lane_queues[idx] if idx < len(lane_queues) else 0.0)
                movement_wait[movement] = 0.0

    phase_queue: dict[str, float] = {}
    phase_wait: dict[str, float] = {}
    for phase, lanes in PHASE_LANES.items():
        phase_queue[phase] = float(sum(movement_queue.get(lane, 0.0) for lane in lanes))
        waits = [movement_wait.get(lane, 0.0) for lane in lanes]
        phase_wait[phase] = float(sum(waits) / len(waits)) if waits else 0.0
    return phase_queue, phase_wait


def choose_rule_action(
    phase_queue: Mapping[str, float],
    phase_wait: Mapping[str, float],
    original_action: str,
    rule_gap: float,
    left_wait_protect: float,
    target_queue_floor: float = 0.0,
    total_queue_floor: float = 0.0,
) -> tuple[int, str, dict[str, Any]]:
    """Apply queue-priority with left-turn waiting-time protection."""
    original = original_action if original_action in ACTION_LIST else ACTION_LIST[0]
    queues = {phase: _safe_float(phase_queue.get(phase, 0.0)) for phase in ACTION_LIST}
    waits = {phase: _safe_float(phase_wait.get(phase, 0.0)) for phase in ACTION_LIST}
    target = max(ACTION_LIST, key=lambda phase: queues.get(phase, 0.0))

    original_queue = float(queues.get(original, 0.0))
    target_queue = float(queues.get(target, 0.0))
    total_queue = float(sum(queues.values()))
    gap = float(target_queue - original_queue)
    should_switch = target != original and gap >= float(rule_gap)
    protect_reason = ""

    if should_switch and float(target_queue_floor or 0.0) > 0.0 and target_queue < float(target_queue_floor):
        should_switch = False
        protect_reason = f"target_queue_floor:{target}={target_queue:.3f}<{float(target_queue_floor):.3f}"

    if should_switch and float(total_queue_floor or 0.0) > 0.0 and total_queue < float(total_queue_floor):
        should_switch = False
        protect_reason = f"total_queue_floor:{total_queue:.3f}<{float(total_queue_floor):.3f}"

    if should_switch and float(left_wait_protect or 0.0) > 0.0:
        if original in LEFT_PHASES and target in THROUGH_PHASES:
            original_wait = float(waits.get(original, 0.0))
            if original_wait >= float(left_wait_protect):
                should_switch = False
                protect_reason = f"left_wait_protect:{original}={original_wait:.3f}>={float(left_wait_protect):.3f}"

    final_action = target if should_switch else original
    diag = {
        "enabled": True,
        "controller": "rule_only",
        "overridden": bool(should_switch),
        "protect_reason": protect_reason,
        "original_action": original,
        "final_action": final_action,
        "target_phase": target,
        "original_queue": original_queue,
        "target_queue": target_queue,
        "total_queue": total_queue,
        "gap": gap,
        "phase_queue": queues,
        "phase_wait": waits,
        "rule_gap": float(rule_gap),
        "left_wait_protect": float(left_wait_protect),
        "target_queue_floor": float(target_queue_floor or 0.0),
        "total_queue_floor": float(total_queue_floor or 0.0),
    }
    return action_to_index(final_action), final_action, diag


class RuleOnlyAgent(Agent):
    """Standard agent wrapper for the Rule-only controller."""

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, cnt_round, intersection_id):
        super(RuleOnlyAgent, self).__init__(dic_agent_conf, dic_traffic_env_conf, dic_path, intersection_id)
        self.action = 0
        self.rule_gap = _safe_float(dic_agent_conf.get("RULE_GAP", 2.0), 2.0)
        self.left_wait_protect = _safe_float(dic_agent_conf.get("LEFT_WAIT_PROTECT", 60.0), 60.0)
        self.target_queue_floor = _safe_float(dic_agent_conf.get("TARGET_QUEUE_FLOOR", 0.0), 0.0)
        self.total_queue_floor = _safe_float(dic_agent_conf.get("TOTAL_QUEUE_FLOOR", 0.0), 0.0)
        self.last_diagnostic: dict[str, Any] = {}

    def choose_action(self, count, state):
        if state.get("cur_phase", [0])[0] == -1:
            return self.action
        original_idx = action_index_from_cur_phase(state.get("cur_phase", [0]), fallback=self.action)
        phase_queue, phase_wait = phase_queue_wait_from_state(state)
        action_idx, _, diagnostic = choose_rule_action(
            phase_queue=phase_queue,
            phase_wait=phase_wait,
            original_action=_phase_from_index(original_idx),
            rule_gap=self.rule_gap,
            left_wait_protect=self.left_wait_protect,
            target_queue_floor=self.target_queue_floor,
            total_queue_floor=self.total_queue_floor,
        )
        self.action = action_idx
        self.last_diagnostic = diagnostic
        return self.action
