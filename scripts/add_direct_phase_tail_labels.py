#!/usr/bin/env python3
"""Attach directly observed per-phase tail waiting costs to a Stage 2 JSONL.

The input must already contain a rule-derived hard label and the current-state
phase queue/wait fields.  This script adds only ``tail_wait`` to each action's
cost vector; it never relabels actions or simulates alternatives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any


ACTIONS = ("ETWT", "NTST", "ELWL", "NLSL")


def finite_nonnegative(value: object, context: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: non-numeric tail wait") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{context}: invalid tail wait {number!r}")
    return number


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() or args.summary.exists():
        raise FileExistsError("Refusing to overwrite a tail-label artifact")

    rows = terminals = enriched = 0
    temporary = args.output.with_name(f"{args.output.name}.tmp.{os.getpid()}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with args.input.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as target:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    raise ValueError(f"{args.input}:{line_number}: blank JSONL row")
                record: dict[str, Any] = json.loads(raw_line)
                rows += 1
                if record.get("expert_action") in ACTIONS:
                    terminals += 1
                    p95 = record.get("phase_wait_p95")
                    if not isinstance(p95, dict) or set(p95) != set(ACTIONS):
                        raise ValueError(f"{args.input}:{line_number}: incomplete phase_wait_p95")
                    metrics = record.get("candidate_action_metrics")
                    if not isinstance(metrics, dict) or set(metrics) != set(ACTIONS):
                        raise ValueError(f"{args.input}:{line_number}: incomplete candidate_action_metrics")
                    for action in ACTIONS:
                        entry = metrics.get(action)
                        if not isinstance(entry, dict):
                            raise ValueError(f"{args.input}:{line_number}: invalid metric entry for {action}")
                        entry["tail_wait"] = -finite_nonnegative(p95[action], f"{args.input}:{line_number}:{action}")
                    meta = record.get("meta") if isinstance(record.get("meta"), dict) else {}
                    meta["direct_urgency_tail"] = {
                        "schema": "direct_phase_urgency_tail_v1",
                        "semantics": "negative_observed_phase_vehicle_wait_p95_no_rollout",
                        "source": "phase_wait_p95",
                    }
                    record["meta"] = meta
                    enriched += 1
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, args.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    summary = {
        "schema": "direct_phase_urgency_tail_dataset_summary_v1",
        "input_path": str(args.input.resolve()),
        "input_sha256": digest(args.input),
        "output_path": str(args.output.resolve()),
        "output_sha256": digest(args.output),
        "rows": rows,
        "terminal_labels": terminals,
        "tail_enriched_terminal_labels": enriched,
        "label_semantics": "rule_anchor; observed_phase_queue_wait_tail_no_rollout",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
