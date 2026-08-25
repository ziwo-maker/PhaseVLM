#!/usr/bin/env python3
"""Version an existing queue/wait-labelled dataset for direct urgency ranking.

This does not relabel actions or simulate alternative rollouts.  For each
already labelled decision, every phase receives a cost vector built only from
the queue and waiting time observed at that same decision point.  Lower cost
means that selecting the phase serves a more urgent observed demand.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ACTIONS = ("ETWT", "NTST", "ELWL", "NLSL")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_nonnegative(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if parsed >= 0.0 else 0.0


def direct_metrics(record: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    phase_queue = record.get("phase_queue") if isinstance(record.get("phase_queue"), dict) else {}
    phase_wait = record.get("phase_wait") if isinstance(record.get("phase_wait"), dict) else {}
    metrics = {
        action: {
            "att": 0.0,
            "aql": -finite_nonnegative(phase_queue.get(action)),
            "awt": -finite_nonnegative(phase_wait.get(action)),
            "spillback": 0.0,
        }
        for action in ACTIONS
    }
    # Legacy consumers use this field when raw metrics are unavailable.  Keep
    # it semantically aligned with its name rather than silently substituting
    # queue cost for waiting cost.
    waits = {action: metrics[action]["awt"] for action in ACTIONS}
    return metrics, waits


def build(input_path: Path, output_path: Path, summary_path: Path) -> dict[str, Any]:
    rows_written = 0
    terminal_labels = 0
    action_counts: Counter[str] = Counter()
    full_metric_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as destination:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"{input_path}:{line_number} is not a JSON object")
            expert_action = str(record.get("expert_action") or "")
            if expert_action:
                if expert_action not in ACTIONS:
                    raise ValueError(f"{input_path}:{line_number} has unsupported action {expert_action!r}")
                metrics, waits = direct_metrics(record)
                record["candidate_action_metrics"] = metrics
                record["candidate_action_waits"] = waits
                meta = copy.deepcopy(record.get("meta")) if isinstance(record.get("meta"), dict) else {}
                meta["direct_urgency_costs"] = {
                    "schema": "direct_phase_urgency_v1",
                    "label_source": "rule",
                    "semantics": "observed_phase_queue_wait_no_rollout",
                    "cost_direction": "lower_is_more_urgent_to_serve",
                    "aql": "negative_observed_phase_queue",
                    "awt": "negative_observed_phase_wait",
                }
                record["meta"] = meta
                terminal_labels += 1
                action_counts[expert_action] += 1
                full_metric_rows += int(len(metrics) == len(ACTIONS))
            destination.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            rows_written += 1

    summary = {
        "schema": "direct_phase_urgency_dataset_summary_v1",
        "input_path": str(input_path),
        "input_sha256": sha256_file(input_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "rows": rows_written,
        "terminal_labels": terminal_labels,
        "action_counts": dict(sorted(action_counts.items())),
        "full_candidate_metric_rows": full_metric_rows,
        "label_semantics": "existing_rule_labels; observed_phase_queue_wait_no_rollout",
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, dest="input_path")
    parser.add_argument("--output", required=True, type=Path, dest="output_path")
    parser.add_argument("--summary", required=True, type=Path, dest="summary_path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build(args.input_path, args.output_path, args.summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
