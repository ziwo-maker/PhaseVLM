#!/usr/bin/env python3
"""Aggregate expert-relabelled policy rollouts into a weighted Stage 2 JSONL.

The rollout input must retain every observation row. This script only removes
the expert label from easy decision rows, so GroupedTrafficActionDataset still
reconstructs the same per-intersection temporal context for hard examples.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projection_train.action_space import ACTION_LIST


_WORKER_CONFIG: dict[str, float] | None = None


def _init_rollout_worker(config: dict[str, float]) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config


def _float(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _float(item) for key, item in value.items() if math.isfinite(_float(item, float("nan")))}


def _candidate_costs(record: dict[str, Any]) -> dict[str, float]:
    metrics = record.get("candidate_action_metrics")
    if isinstance(metrics, dict):
        scores = {
            action: _float(value.get("score"), float("nan"))
            for action, value in metrics.items()
            if action in ACTION_LIST and isinstance(value, dict)
        }
        scores = {action: value for action, value in scores.items() if math.isfinite(value)}
        if len(scores) >= 2:
            return scores
    waits = _float_map(record.get("candidate_action_waits"))
    return {action: waits[action] for action in ACTION_LIST if action in waits}


def _behavior_action(record: dict[str, Any]) -> str:
    for key in ("policy_action", "base_action", "action"):
        action = str(record.get(key) or "").strip()
        if action in ACTION_LIST:
            return action
    meta = record.get("meta")
    if isinstance(meta, dict):
        dagger = meta.get("dagger")
        if isinstance(dagger, dict):
            action = str(dagger.get("policy_action") or "").strip()
            if action in ACTION_LIST:
                return action
    return ""


def dagger_priority(
    record: dict[str, Any],
    queue_scale: float,
    wait_scale: float,
    regret_scale: float,
    missed_gap_scale: float,
) -> tuple[float, dict[str, float]]:
    phase_queue = _float_map(record.get("phase_queue"))
    phase_wait = _float_map(record.get("phase_wait"))
    total_queue = sum(max(0.0, phase_queue.get(action, 0.0)) for action in ACTION_LIST)
    max_wait = max((max(0.0, phase_wait.get(action, 0.0)) for action in ACTION_LIST), default=0.0)
    costs = _candidate_costs(record)
    behavior = _behavior_action(record)
    regret = 0.0
    if behavior in costs and costs:
        regret = max(0.0, costs[behavior] - min(costs.values()))
    diagnostic = record.get("max_phase_override")
    missed_gap = _float(diagnostic.get("gap"), 0.0) if isinstance(diagnostic, dict) else 0.0
    components = {
        "total_queue": total_queue,
        "max_wait": max_wait,
        "action_regret": regret,
        "missed_queue_gap": max(0.0, missed_gap),
    }
    priority = (
        total_queue / max(queue_scale, 1e-8)
        + max_wait / max(wait_scale, 1e-8)
        + regret / max(regret_scale, 1e-8)
        + max(0.0, missed_gap) / max(missed_gap_scale, 1e-8)
    )
    return priority, components


def is_terminal_record(record: dict[str, Any]) -> bool:
    return bool(
        str(record.get("prompt") or "").strip()
        or str(record.get("expert_action") or "").strip()
        or str(record.get("clip_caption") or "").strip()
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield record


def prepare_dagger_record(
    record: dict[str, Any],
    source_path: Path,
    min_priority: float,
    queue_scale: float,
    wait_scale: float,
    regret_scale: float,
    missed_gap_scale: float,
    weight_scale: float,
    max_sample_weight: float,
) -> tuple[dict[str, Any], bool, dict[str, float]]:
    if not is_terminal_record(record):
        return dict(record), False, {}
    priority, components = dagger_priority(
        record,
        queue_scale=queue_scale,
        wait_scale=wait_scale,
        regret_scale=regret_scale,
        missed_gap_scale=missed_gap_scale,
    )
    expert_action = str(record.get("expert_action") or "").strip()
    selected = expert_action in ACTION_LIST and priority >= min_priority
    prepared = dict(record)
    if not selected:
        # Keep the observation row for history, but exclude it from Stage 2 loss.
        prepared["expert_action"] = ""
        return prepared, False, {"priority": priority, **components}

    sample_weight = min(max_sample_weight, max(1.0, 1.0 + priority * weight_scale))
    prepared["sample_weight"] = sample_weight
    meta = dict(prepared.get("meta") or {})
    meta["dagger"] = {
        "schema": "dagger_stage2_v1",
        "source_path": str(source_path),
        "priority": priority,
        "components": components,
        "behavior_action": _behavior_action(record),
        "sample_weight": sample_weight,
    }
    prepared["meta"] = meta
    return prepared, True, {"priority": priority, "sample_weight": sample_weight, **components}


def _prepare_rollout_record(item: tuple[dict[str, Any], str]) -> tuple[dict[str, Any], bool, bool]:
    """Prepare one rollout row in a pool worker while retaining input order via imap."""
    if _WORKER_CONFIG is None:
        raise RuntimeError("Rollout worker was not initialized")
    record, source_path = item
    terminal = is_terminal_record(record)
    prepared, selected, _ = prepare_dagger_record(
        record,
        source_path=Path(source_path),
        min_priority=_WORKER_CONFIG["min_priority"],
        queue_scale=_WORKER_CONFIG["queue_scale"],
        wait_scale=_WORKER_CONFIG["wait_scale"],
        regret_scale=_WORKER_CONFIG["regret_scale"],
        missed_gap_scale=_WORKER_CONFIG["missed_gap_scale"],
        weight_scale=_WORKER_CONFIG["weight_scale"],
        max_sample_weight=_WORKER_CONFIG["max_sample_weight"],
    )
    return prepared, selected, terminal


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output_path = Path(args.output).resolve()
    workers = getattr(args, "workers", 1)
    worker_chunksize = getattr(args, "worker_chunksize", 256)
    inputs = [Path(path).resolve() for path in [*(args.base_data or []), *args.rollout_data]]
    if output_path in inputs:
        raise ValueError("--output must not overwrite an input JSONL")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    action_counts: Counter[str] = Counter()
    summary: dict[str, Any] = {
        "schema": "dagger_stage2_v1",
        "base_data": [str(Path(path).resolve()) for path in (args.base_data or [])],
        "rollout_data": [str(Path(path).resolve()) for path in args.rollout_data],
        "output": str(output_path),
        "base_rows": 0,
        "rollout_rows": 0,
        "rollout_terminal_rows": 0,
        "selected_dagger_rows": 0,
        "masked_easy_terminal_rows": 0,
        "workers": workers,
        "worker_chunksize": worker_chunksize,
        "action_counts": action_counts,
        "thresholds": {
            "min_priority": args.min_priority,
            "queue_scale": args.queue_scale,
            "wait_scale": args.wait_scale,
            "regret_scale": args.regret_scale,
            "missed_gap_scale": args.missed_gap_scale,
            "weight_scale": args.weight_scale,
            "max_sample_weight": args.max_sample_weight,
        },
    }
    with output_path.open("w", encoding="utf-8") as handle:
        for source in [Path(path) for path in args.base_data or []]:
            for record in _iter_jsonl(source):
                summary["base_rows"] += 1
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        worker_config = {
            "min_priority": args.min_priority,
            "queue_scale": args.queue_scale,
            "wait_scale": args.wait_scale,
            "regret_scale": args.regret_scale,
            "missed_gap_scale": args.missed_gap_scale,
            "weight_scale": args.weight_scale,
            "max_sample_weight": args.max_sample_weight,
        }
        pool = (
            mp.Pool(processes=workers, initializer=_init_rollout_worker, initargs=(worker_config,))
            if workers > 1
            else None
        )
        try:
            if pool is None:
                _init_rollout_worker(worker_config)
            for source in [Path(path) for path in args.rollout_data]:
                records = ((record, str(source)) for record in _iter_jsonl(source))
                prepared_records = (
                    pool.imap(_prepare_rollout_record, records, chunksize=worker_chunksize)
                    if pool is not None
                    else map(_prepare_rollout_record, records)
                )
                for prepared, selected, terminal in prepared_records:
                    summary["rollout_rows"] += 1
                    summary["rollout_terminal_rows"] += int(terminal)
                    if terminal and not selected:
                        summary["masked_easy_terminal_rows"] += 1
                    if selected:
                        summary["selected_dagger_rows"] += 1
                        action_counts[str(prepared["expert_action"])] += 1
                    handle.write(json.dumps(prepared, ensure_ascii=False, sort_keys=True) + "\n")
        finally:
            if pool is not None:
                pool.close()
                pool.join()

    summary["action_counts"] = dict(sorted(action_counts.items()))
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    summary["summary_path"] = str(summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-data", nargs="*", default=[])
    parser.add_argument("--rollout-data", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-priority", type=float, default=1.0)
    parser.add_argument("--queue-scale", type=float, default=10.0)
    parser.add_argument("--wait-scale", type=float, default=60.0)
    parser.add_argument("--regret-scale", type=float, default=1.0)
    parser.add_argument("--missed-gap-scale", type=float, default=5.0)
    parser.add_argument("--weight-scale", type=float, default=0.5)
    parser.add_argument("--max-sample-weight", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=1, help="Rollout processing workers; use 1 for sequential mode")
    parser.add_argument("--worker-chunksize", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    positive = (
        args.queue_scale,
        args.wait_scale,
        args.regret_scale,
        args.missed_gap_scale,
        args.max_sample_weight,
    )
    if min(positive) <= 0 or args.weight_scale < 0 or args.min_priority < 0:
        raise ValueError("Scales and max sample weight must be positive; priorities/weight scale non-negative")
    if args.workers < 1 or args.worker_chunksize < 1:
        raise ValueError("--workers and --worker-chunksize must be positive")
    print(json.dumps(build_dataset(args), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
