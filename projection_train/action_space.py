from __future__ import annotations

from typing import Sequence


ACTION_LIST: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")
ACTION_TO_INDEX: dict[str, int] = {action: idx for idx, action in enumerate(ACTION_LIST)}


def action_to_index(action: str) -> int:
    return ACTION_TO_INDEX.get(str(action or "").strip(), 0)


def index_to_action(index: int) -> str:
    idx = int(index)
    if idx < 0 or idx >= len(ACTION_LIST):
        return ACTION_LIST[0]
    return ACTION_LIST[idx]


def format_action_tag(action: str) -> str:
    return f"<signal>{str(action).strip()}</signal>"


def normalize_action_tags(tags: Sequence[str] | None = None) -> tuple[str, ...]:
    if not tags:
        return ACTION_LIST
    normalized = tuple(str(tag).strip() for tag in tags if str(tag).strip())
    return normalized or ACTION_LIST
