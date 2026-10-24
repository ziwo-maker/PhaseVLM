#!/usr/bin/env python3
"""Build queue/wait-aware Stage 2 distillation JSONL from online logs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projection_train.action_space import ACTION_LIST
from projection_train.prompt_utils import (
    MOVEMENT_KEYS,
    build_phase_queue_wait_context,
    phase_queue_wait_from_state,
)


SIGNAL_RE = re.compile(r"<signal>(.*?)</signal>")
LEFT_PHASES = {"ELWL", "NLSL"}
THROUGH_PHASES = {"ETWT", "NTST"}


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def _parse_action(value: Any) -> str:
    text = str(value or "").strip()
    if text in ACTION_LIST:
        return text
    matches = SIGNAL_RE.findall(text)
    if matches:
        candidate = matches[-1].strip()
        if candidate in ACTION_LIST:
            return candidate
    return ""


def _action_from_item(item: dict[str, Any], use_policy: bool) -> str:
    if use_policy:
        return (
            _parse_action(item.get("policy_response"))
            or _parse_action(item.get("policy_action"))
            or _parse_action(item.get("response"))
            or _parse_action(item.get("action"))
        )
    return (
        _parse_action(item.get("response"))
        or _parse_action(item.get("action"))
        or _parse_action(item.get("policy_response"))
        or _parse_action(item.get("policy_action"))
    )


def _float_map(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _resolve_phase_stats(item: dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
    diag = item.get("max_phase_override") or {}
    phase_queue = _float_map(item.get("phase_queue")) or _float_map(diag.get("phase_queue"))
    phase_wait = _float_map(item.get("phase_wait")) or _float_map(diag.get("phase_wait"))
    if phase_queue and phase_wait:
        return phase_queue, phase_wait
    derived_queue, derived_wait, _ = phase_queue_wait_from_state(item.get("state") or {})
    return phase_queue or derived_queue, phase_wait or derived_wait


def _movement_counts_from_state(item: dict[str, Any]) -> dict[str, float]:
    state = item.get("state") or {}
    counts: dict[str, float] = {}
    for key in MOVEMENT_KEYS:
        lane_state = state.get(key) if isinstance(state, dict) else {}
        if not isinstance(lane_state, dict):
            lane_state = {}
        try:
            counts[key] = float(lane_state.get("queue_len", 0.0) or 0.0)
        except (TypeError, ValueError):
            counts[key] = 0.0
    return counts


def _rule_action(
    item: dict[str, Any],
    gap: float,
    left_wait_protect: float,
) -> tuple[str, dict[str, Any]]:
    phase_queue, phase_wait = _resolve_phase_stats(item)
    original = _action_from_item(item, use_policy=True)
    if not original:
        original = _action_from_item(item, use_policy=False) or ACTION_LIST[0]

    target = max(ACTION_LIST, key=lambda phase: float(phase_queue.get(phase, 0.0)))
    original_queue = float(phase_queue.get(original, 0.0))
    target_queue = float(phase_queue.get(target, 0.0))
    total_queue = float(sum(float(phase_queue.get(phase, 0.0)) for phase in ACTION_LIST))
    missed_gap = target_queue - original_queue
    should_override = target != original and missed_gap >= float(gap)
    protect_reason = ""
    if should_override and left_wait_protect > 0.0:
        if original in LEFT_PHASES and target in THROUGH_PHASES:
            original_wait = float(phase_wait.get(original, 0.0))
            if original_wait >= float(left_wait_protect):
                should_override = False
                protect_reason = f"left_wait_protect:{original}={original_wait:.3f}>={left_wait_protect:.3f}"

    final = target if should_override else original
    diag = {
        "enabled": True,
        "synthetic": True,
        "overridden": bool(should_override),
        "protect_reason": protect_reason,
        "original_action": original,
        "final_action": final,
        "target_phase": target,
        "original_queue": original_queue,
        "target_queue": target_queue,
        "total_queue": total_queue,
        "gap": float(missed_gap),
        "phase_queue": phase_queue,
        "phase_wait": phase_wait,
        "rule_gap": float(gap),
        "left_wait_protect": float(left_wait_protect),
    }
    return final, diag


def _label_action(
    item: dict[str, Any],
    label_source: str,
    gap: float,
    left_wait_protect: float,
) -> tuple[str, dict[str, Any]]:
    if label_source == "rule":
        return _rule_action(item, gap=gap, left_wait_protect=left_wait_protect)

    diag = item.get("max_phase_override") or {}
    phase_queue, phase_wait = _resolve_phase_stats(item)
    if label_source == "policy":
        action = _action_from_item(item, use_policy=True)
    else:
        action = _parse_action(diag.get("final_action")) or _action_from_item(item, use_policy=False)
    if not action:
        action = ACTION_LIST[0]
    if isinstance(diag, dict) and diag:
        resolved_diag = dict(diag)
    else:
        resolved_diag = {
            "enabled": False,
            "synthetic": False,
            "overridden": False,
            "original_action": _action_from_item(item, use_policy=True) or action,
            "final_action": action,
            "target_phase": max(ACTION_LIST, key=lambda phase: float(phase_queue.get(phase, 0.0))),
            "phase_queue": phase_queue,
            "phase_wait": phase_wait,
        }
    return action, resolved_diag


def _augment_record(
    record: dict[str, Any],
    item: dict[str, Any],
    label_source: str,
    gap: float,
    left_wait_protect: float,
    source_path: Path,
) -> tuple[dict[str, Any], bool]:
    phase_queue, phase_wait = _resolve_phase_stats(item)
    context = str(item.get("prompt_context") or "").strip()
    if not context:
        context = build_phase_queue_wait_context(
            state=item.get("state") or {},
            phase_queue=phase_queue,
            phase_wait=phase_wait,
        )
    action, diag = _label_action(
        item,
        label_source=label_source,
        gap=gap,
        left_wait_protect=left_wait_protect,
    )
    base_action = (
        _parse_action(diag.get("original_action") if isinstance(diag, dict) else "")
        or _action_from_item(item, use_policy=True)
        or _action_from_item(item, use_policy=False)
    )

    augmented = dict(record)
    if not str(augmented.get("prompt") or "").strip():
        augmented["prompt"] = str(item.get("raw_prompt") or item.get("policy_prompt") or "")
    augmented["expert_action"] = action
    augmented["base_action"] = base_action
    augmented["prompt_context"] = context
    augmented["phase_queue"] = phase_queue
    augmented["phase_wait"] = phase_wait
    phase_wait_p95 = _float_map(item.get("phase_wait_p95"))
    if set(phase_wait_p95) == set(ACTION_LIST):
        augmented["phase_wait_p95"] = phase_wait_p95
    augmented["candidate_action_waits"] = {
        phase: -float(phase_queue.get(phase, 0.0)) for phase in ACTION_LIST
    }
    augmented["candidate_action_metrics"] = {
        phase: {
            "att": 0.0,
            "aql": -float(phase_queue.get(phase, 0.0)),
            "awt": -float(phase_wait.get(phase, 0.0)),
            "spillback": 0.0,
            **({"tail_wait": -float(phase_wait_p95[phase])} if phase in phase_wait_p95 else {}),
        }
        for phase in ACTION_LIST
    }
    augmented["max_phase_override"] = diag
    augmented["distill_source"] = {
        "state_action_path": str(source_path),
        "label_source": label_source,
    }
    meta = dict(augmented.get("meta") or {})
    meta["queue_wait_distill"] = {
        "schema": "queue_wait_distill_v1",
        "label_source": label_source,
        "rule_gap": float(gap),
        "left_wait_protect": float(left_wait_protect),
    }
    augmented["meta"] = meta
    return augmented, bool(isinstance(diag, dict) and diag.get("overridden"))


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _terminal_record(record: dict[str, Any]) -> bool:
    return bool(
        str(record.get("prompt") or "").strip()
        or str(record.get("expert_action") or "").strip()
        or str(record.get("clip_caption") or "").strip()
    )


def build_from_prompt_series(
    state_action_path: Path,
    prompt_series_path: Path,
    output_path: Path,
    label_source: str,
    gap: float,
    left_wait_protect: float,
) -> dict[str, Any]:
    state_action = _load_json(state_action_path)
    if not isinstance(state_action, list):
        raise ValueError(f"Expected state_action to be a list: {state_action_path}")

    inter_to_idx: dict[str, int] = {}
    step_by_inter: defaultdict[str, int] = defaultdict(int)
    action_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    total_rows = 0
    terminal_rows = 0
    overridden_rows = 0
    missing_state_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in _iter_jsonl(prompt_series_path):
            total_rows += 1
            if not _terminal_record(record):
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            inter_id = str(record.get("intersection_id") or record.get("road_id") or "")
            if inter_id not in inter_to_idx:
                inter_to_idx[inter_id] = len(inter_to_idx)
            inter_idx = inter_to_idx[inter_id]
            step_idx = step_by_inter[inter_id]
            step_by_inter[inter_id] += 1

            item = None
            if inter_idx < len(state_action) and isinstance(state_action[inter_idx], list):
                inter_log = state_action[inter_idx]
                if step_idx < len(inter_log) and isinstance(inter_log[step_idx], dict):
                    item = inter_log[step_idx]
            if item is None:
                missing_state_rows += 1
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            augmented, overridden = _augment_record(
                record,
                item,
                label_source=label_source,
                gap=gap,
                left_wait_protect=left_wait_protect,
                source_path=state_action_path,
            )
            terminal_rows += 1
            overridden_rows += int(overridden)
            action_counts[str(augmented.get("expert_action") or "")] += 1
            base_counts[str(augmented.get("base_action") or "")] += 1
            handle.write(json.dumps(augmented, ensure_ascii=False) + "\n")

    return {
        "output_path": str(output_path),
        "state_action_path": str(state_action_path),
        "prompt_series_path": str(prompt_series_path),
        "label_source": label_source,
        "rule_gap": float(gap),
        "left_wait_protect": float(left_wait_protect),
        "total_rows": total_rows,
        "terminal_rows": terminal_rows,
        "missing_state_rows": missing_state_rows,
        "overridden_rows": overridden_rows,
        "action_counts": dict(action_counts),
        "base_action_counts": dict(base_counts),
        "intersection_order": inter_to_idx,
    }


def build_from_state_action_only(
    state_action_path: Path,
    output_path: Path,
    label_source: str,
    gap: float,
    left_wait_protect: float,
) -> dict[str, Any]:
    state_action = _load_json(state_action_path)
    if not isinstance(state_action, list):
        raise ValueError(f"Expected state_action to be a list: {state_action_path}")

    action_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    rows = 0
    overridden_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for inter_idx, inter_log in enumerate(state_action):
            if not isinstance(inter_log, list):
                continue
            for step_idx, item in enumerate(inter_log):
                if not isinstance(item, dict):
                    continue
                record = {
                    "date": f"step_{step_idx:05d}",
                    "interval_s": 30,
                    "intersection_id": f"intersection_index_{inter_idx}",
                    "movement_counts": _movement_counts_from_state(item),
                    "prompt": str(item.get("raw_prompt") or item.get("policy_prompt") or ""),
                    "snapshot_path": str(item.get("snapshot_path") or ""),
                }
                augmented, overridden = _augment_record(
                    record,
                    item,
                    label_source=label_source,
                    gap=gap,
                    left_wait_protect=left_wait_protect,
                    source_path=state_action_path,
                )
                rows += 1
                overridden_rows += int(overridden)
                action_counts[str(augmented.get("expert_action") or "")] += 1
                base_counts[str(augmented.get("base_action") or "")] += 1
                handle.write(json.dumps(augmented, ensure_ascii=False) + "\n")

    return {
        "output_path": str(output_path),
        "state_action_path": str(state_action_path),
        "prompt_series_path": "",
        "label_source": label_source,
        "rule_gap": float(gap),
        "left_wait_protect": float(left_wait_protect),
        "terminal_rows": rows,
        "overridden_rows": overridden_rows,
        "action_counts": dict(action_counts),
        "base_action_counts": dict(base_counts),
    }


def build_from_canonical_decision_states(
    state_action_path: Path,
    decision_states_path: Path,
    output_path: Path,
    label_source: str,
    gap: float,
    left_wait_protect: float,
) -> dict[str, Any]:
    state_action = _load_json(state_action_path)
    if not isinstance(state_action, list):
        raise ValueError(f"Expected state_action to be a list: {state_action_path}")

    intersection_order: dict[str, int] = {}
    step_by_intersection: defaultdict[str, int] = defaultdict(int)
    action_counts: Counter[str] = Counter()
    base_counts: Counter[str] = Counter()
    rows = 0
    overridden_rows = 0
    missing_state_action_rows = 0
    invalid_sequence_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with decision_states_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as handle:
        for line in source:
            if not line.strip():
                continue
            decision = json.loads(line)
            intersection_id = str(decision.get("intersection_id") or "")
            if not intersection_id:
                raise ValueError(f"Missing intersection_id in {decision_states_path}")
            if intersection_id not in intersection_order:
                intersection_order[intersection_id] = len(intersection_order)
            intersection_index = intersection_order[intersection_id]
            step_index = step_by_intersection[intersection_id]
            step_by_intersection[intersection_id] += 1

            sequences = decision.get("sequences")
            if not (
                isinstance(sequences, list)
                and len(sequences) == 16
                and all(isinstance(row, list) and len(row) == len(MOVEMENT_KEYS) for row in sequences)
            ):
                invalid_sequence_rows += 1
                continue
            if intersection_index >= len(state_action) or not isinstance(
                state_action[intersection_index], list
            ) or step_index >= len(state_action[intersection_index]):
                missing_state_action_rows += 1
                continue

            item = state_action[intersection_index][step_index]
            if not isinstance(item, dict):
                missing_state_action_rows += 1
                continue

            record = dict(decision)
            record["sequence_source"] = "online_canonical"
            record["canonical_intersection_index"] = intersection_index
            record["canonical_step_index"] = step_index
            augmented, overridden = _augment_record(
                record,
                item,
                label_source=label_source,
                gap=gap,
                left_wait_protect=left_wait_protect,
                source_path=state_action_path,
            )
            rows += 1
            overridden_rows += int(overridden)
            action_counts[str(augmented.get("expert_action") or "")] += 1
            base_counts[str(augmented.get("base_action") or "")] += 1
            handle.write(json.dumps(augmented, ensure_ascii=False) + "\n")

    if invalid_sequence_rows or missing_state_action_rows:
        raise ValueError(
            "Canonical decision-state alignment failed: "
            f"invalid_sequence_rows={invalid_sequence_rows} "
            f"missing_state_action_rows={missing_state_action_rows}"
        )
    expected_rows = sum(len(items) for items in state_action if isinstance(items, list))
    if rows != expected_rows:
        raise ValueError(f"Canonical row count mismatch: rows={rows} expected={expected_rows}")

    return {
        "output_path": str(output_path),
        "state_action_path": str(state_action_path),
        "decision_states_path": str(decision_states_path),
        "label_source": label_source,
        "rule_gap": float(gap),
        "left_wait_protect": float(left_wait_protect),
        "terminal_rows": rows,
        "overridden_rows": overridden_rows,
        "invalid_sequence_rows": invalid_sequence_rows,
        "missing_state_action_rows": missing_state_action_rows,
        "intersection_count": len(intersection_order),
        "action_counts": dict(action_counts),
        "base_action_counts": dict(base_counts),
        "sequence_source": "online_canonical",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_action", required=True, type=Path)
    parser.add_argument("--decision_states", type=Path, default=None)
    parser.add_argument("--prompt_series", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--label_source",
        choices=["final", "policy", "rule"],
        default="final",
        help="final uses logged final action; policy uses pre-override action; rule synthesizes max-phase labels.",
    )
    parser.add_argument("--rule_gap", type=float, default=2.0)
    parser.add_argument("--left_wait_protect", type=float, default=60.0)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.decision_states is not None:
        if args.prompt_series is not None:
            raise ValueError("--decision_states cannot be combined with --prompt_series")
        summary = build_from_canonical_decision_states(
            state_action_path=args.state_action,
            decision_states_path=args.decision_states,
            output_path=args.output,
            label_source=args.label_source,
            gap=args.rule_gap,
            left_wait_protect=args.left_wait_protect,
        )
    elif args.prompt_series is not None:
        summary = build_from_prompt_series(
            state_action_path=args.state_action,
            prompt_series_path=args.prompt_series,
            output_path=args.output,
            label_source=args.label_source,
            gap=args.rule_gap,
            left_wait_protect=args.left_wait_protect,
        )
    else:
        summary = build_from_state_action_only(
            state_action_path=args.state_action,
            output_path=args.output,
            label_source=args.label_source,
            gap=args.rule_gap,
            left_wait_protect=args.left_wait_protect,
        )
    summary_path = args.summary or args.output.with_suffix(args.output.suffix + ".summary.json")
    _dump_json(summary_path, summary)
    print(json.dumps({"output": str(args.output), "summary": str(summary_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
