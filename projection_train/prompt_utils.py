from __future__ import annotations

from typing import Mapping, Sequence


DEFAULT_ACTION_TAGS: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENT_KEYS: tuple[str, ...] = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
PHASE_LANES: dict[str, tuple[str, str]] = {
    "ETWT": ("ET", "WT"),
    "NTST": ("NT", "ST"),
    "ELWL": ("EL", "WL"),
    "NLSL": ("NL", "SL"),
}
PROMPT_MODES: tuple[str, ...] = (
    "raw", "image_only_short", "queue_wait_short", "matched_minimal", "matched_structured",
    "information_matched",
)


def serialize_movement_history(sequence: Sequence[Sequence[float]], interval_s: int = 2) -> str:
    """Losslessly serialize the exact oldest-to-newest 16x8 policy input."""
    if len(sequence) != 16 or any(len(row) != len(MOVEMENT_KEYS) for row in sequence):
        raise ValueError("History-Text requires exactly 16 rows and 8 movement counts")
    if interval_s != 2:
        raise ValueError("The matched representation protocol requires a 2 s interval")

    def number(value: float) -> str:
        value = float(value)
        if not (value >= 0 and value < float("inf")):
            raise ValueError("Movement counts must be finite and nonnegative")
        return str(int(value)) if value.is_integer() else repr(value)

    lines = ["History every 2 s, oldest to newest; columns: " + ",".join(MOVEMENT_KEYS)]
    for index, row in enumerate(sequence):
        offset = (15 - index) * interval_s
        timestamp = f"t-{offset}s" if offset else "t"
        lines.append(f"{timestamp}: " + ",".join(number(value) for value in row))
    return "\n".join(lines)


def serialize_compact_movement_history(sequence: Sequence[Sequence[float]], interval_s: int = 2) -> str:
    if len(sequence) != 16 or any(len(row) != len(MOVEMENT_KEYS) for row in sequence):
        raise ValueError("Information-matched history requires exactly 16x8 values")
    if interval_s != 2:
        raise ValueError("Information-matched history requires a 2 s interval")
    rows = []
    for row in sequence:
        values = []
        for value in row:
            number = float(value)
            if not (0 <= number < float("inf")):
                raise ValueError("Movement counts must be finite and nonnegative")
            values.append(str(int(number)) if number.is_integer() else repr(number))
        rows.append(" ".join(values))
    return "H(16x8,2s/row,oldest->newest; " + " ".join(MOVEMENT_KEYS) + "):\n" + "\n".join(rows)


def _format_action_tags(action_tags: Sequence[str]) -> str:
    tags = [str(tag).strip() for tag in action_tags if str(tag).strip()]
    if not tags:
        tags = list(DEFAULT_ACTION_TAGS)
    return ", ".join(f"<signal>{tag}</signal>" for tag in tags)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def phase_queue_wait_from_state(
    state: Mapping[str, object] | None,
) -> tuple[dict[str, float], dict[str, float], dict[str, dict[str, float]]]:
    """Extract movement and phase queue/wait summaries from a traffic state."""
    movement: dict[str, dict[str, float]] = {}
    state = state or {}
    for key in MOVEMENT_KEYS:
        item = state.get(key) if isinstance(state, Mapping) else None
        lane_state = item if isinstance(item, Mapping) else {}
        movement[key] = {
            "queue": _safe_float(lane_state.get("queue_len", 0.0)),
            "wait": _safe_float(lane_state.get("avg_wait_time", 0.0)),
        }

    phase_queue: dict[str, float] = {}
    phase_wait: dict[str, float] = {}
    for phase, lanes in PHASE_LANES.items():
        queues = [movement[lane]["queue"] for lane in lanes]
        waits = [movement[lane]["wait"] for lane in lanes]
        phase_queue[phase] = float(sum(queues))
        phase_wait[phase] = float(sum(waits) / len(waits)) if waits else 0.0
    return phase_queue, phase_wait, movement


def _format_metric(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.1f}"


def build_phase_queue_wait_context(
    state: Mapping[str, object] | None = None,
    phase_queue: Mapping[str, object] | None = None,
    phase_wait: Mapping[str, object] | None = None,
    movement_queue_wait: Mapping[str, Mapping[str, object]] | None = None,
) -> str:
    if phase_queue is None or phase_wait is None or movement_queue_wait is None:
        state_phase_queue, state_phase_wait, state_movement = phase_queue_wait_from_state(state)
        if phase_queue is None:
            phase_queue = state_phase_queue
        if phase_wait is None:
            phase_wait = state_phase_wait
        if movement_queue_wait is None:
            movement_queue_wait = state_movement

    movement_parts: list[str] = []
    for key in MOVEMENT_KEYS:
        item = movement_queue_wait.get(key, {}) if isinstance(movement_queue_wait, Mapping) else {}
        queue = _safe_float(item.get("queue", item.get("queue_len", 0.0)) if isinstance(item, Mapping) else 0.0)
        wait = _safe_float(item.get("wait", item.get("avg_wait_time", 0.0)) if isinstance(item, Mapping) else 0.0)
        movement_parts.append(f"{key}={_format_metric(queue)}/{_format_metric(wait)}")

    phase_parts: list[str] = []
    phase_queue_values: dict[str, float] = {}
    phase_wait_values: dict[str, float] = {}
    for phase in DEFAULT_ACTION_TAGS:
        queue = _safe_float(phase_queue.get(phase, 0.0) if isinstance(phase_queue, Mapping) else 0.0)
        wait = _safe_float(phase_wait.get(phase, 0.0) if isinstance(phase_wait, Mapping) else 0.0)
        phase_queue_values[phase] = queue
        phase_wait_values[phase] = wait
        phase_parts.append(f"{phase}={_format_metric(queue)}/{_format_metric(wait)}")

    target_phase = max(DEFAULT_ACTION_TAGS, key=lambda phase: phase_queue_values.get(phase, 0.0))
    total_queue = sum(phase_queue_values.values())
    max_wait_phase = max(DEFAULT_ACTION_TAGS, key=lambda phase: phase_wait_values.get(phase, 0.0))

    return (
        "Traffic q/w(s): "
        f"lanes {','.join(movement_parts)}; "
        f"phases {','.join(phase_parts)}; "
        f"maxQ={target_phase}({_format_metric(phase_queue_values[target_phase])}); "
        f"maxW={max_wait_phase}({_format_metric(phase_wait_values[max_wait_phase])}); "
        f"totalQ={_format_metric(total_queue)}."
    )


def build_task_prompt(
    raw_prompt: str,
    mode: str = "raw",
    action_tags: Sequence[str] = DEFAULT_ACTION_TAGS,
    traffic_context: str | None = None,
) -> str:
    mode = str(mode or "raw").strip().lower()
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {mode!r}")

    if mode == "raw":
        return str(raw_prompt or "").rstrip()

    allowed = _format_action_tags(action_tags)
    if mode == "information_matched":
        context = str(traffic_context or "").strip()
        if not context:
            raise ValueError("Information-matched prompt requires queue/wait context")
        context = context.replace("Traffic q/w(s): lanes ", "Q/W(s): ").replace("; phases ", "; P: ")
        return (
            "Minimize near-term congestion; favor much larger queues; protect high-wait left turns.\n"
            f"{context}\n"
            "Tags: " + ",".join(action_tags) + ". Output <signal>TAG</signal>."
        )
    if mode in {"matched_minimal", "matched_structured"}:
        instruction = (
            "Control one four-phase traffic signal using the supplied recent movement history.\n"
            "Choose the next phase to reduce near-term congestion.\n"
        )
        if mode == "matched_structured":
            context = str(traffic_context or "").strip()
            if not context:
                raise ValueError("Matched structured prompt requires queue/wait context")
            instruction += (
                f"{context}\n"
                "Rule: reduce near-term congestion; switch to a much larger-queue phase when needed; "
                "avoid starving high-wait left turns.\n"
            )
        return instruction + f"Allowed: {allowed}. Return exactly one tag."
    if mode == "image_only_short":
        return (
            "You are controlling a four-phase traffic signal.\n"
            "The attached image is the only reliable observation of recent traffic dynamics at one intersection.\n"
            "Choose the single best next phase to reduce near-term congestion.\n"
            f"Allowed outputs: {allowed}.\n"
            "Return exactly one <signal>...</signal> tag and nothing else."
        )

    context = str(traffic_context or "").strip()
    if not context:
        context = "Structured traffic summary: unavailable."
    return (
        "Control one four-phase traffic signal using the image and summary.\n"
        f"{context}\n"
        "Rule: reduce near-term congestion; switch to a much larger-queue phase when needed; "
        "avoid starving high-wait left turns.\n"
        f"Allowed: {allowed}. Return exactly one tag."
    )


def append_prompt_suffix(prompt: str, suffix: str) -> str:
    base = str(prompt or "").rstrip()
    extra = str(suffix or "").strip()
    if base and extra:
        return f"{base}\n{extra}"
    if base:
        return base
    return extra
