from __future__ import annotations

from typing import Sequence


DEFAULT_ACTION_TAGS: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")
PROMPT_MODES: tuple[str, ...] = ("raw", "image_only_short")


def _format_action_tags(action_tags: Sequence[str]) -> str:
    tags = [str(tag).strip() for tag in action_tags if str(tag).strip()]
    if not tags:
        tags = list(DEFAULT_ACTION_TAGS)
    return ", ".join(f"<signal>{tag}</signal>" for tag in tags)


def build_task_prompt(
    raw_prompt: str,
    mode: str = "raw",
    action_tags: Sequence[str] = DEFAULT_ACTION_TAGS,
) -> str:
    mode = str(mode or "raw").strip().lower()
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt mode: {mode!r}")

    if mode == "raw":
        return str(raw_prompt or "").rstrip()

    allowed = _format_action_tags(action_tags)
    return (
        "You are controlling a four-phase traffic signal.\n"
        "The attached image is the only reliable observation of recent traffic dynamics at one intersection.\n"
        "Choose the single best next phase to reduce near-term congestion.\n"
        f"Allowed outputs: {allowed}.\n"
        "Return exactly one <signal>...</signal> tag and nothing else."
    )


def append_prompt_suffix(prompt: str, suffix: str) -> str:
    base = str(prompt or "").rstrip()
    extra = str(suffix or "").strip()
    if base and extra:
        return f"{base}\n{extra}"
    if base:
        return base
    return extra
