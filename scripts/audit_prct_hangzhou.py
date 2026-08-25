#!/usr/bin/env python3
"""Create a reproducible audit ledger for the single-dataset PRCT study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


ACTION_LIST = ("ETWT", "NTST", "ELWL", "NLSL")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_jsonl(path: Path, *, expected_neighbor_mode: str | None = None) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    snapshot_paths: set[str] = set()
    missing_snapshots = 0
    expert_actions: Counter[str] = Counter()
    raw_actions: Counter[str] = Counter()
    relabel_methods: Counter[str] = Counter()
    neighbor_modes: Counter[str] = Counter()
    counterfactual_modes: Counter[str] = Counter()
    schema_errors: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            records.append(record)
            snapshot = str(record.get("snapshot_path", ""))
            if snapshot:
                snapshot_paths.add(snapshot)
                if not Path(snapshot).is_file():
                    missing_snapshots += 1
            else:
                schema_errors.append(f"line {index}: missing snapshot_path")
            action = str(record.get("expert_action", ""))
            if action not in ACTION_LIST:
                schema_errors.append(f"line {index}: invalid expert_action={action!r}")
            else:
                expert_actions[action] += 1
            costs = record.get("candidate_action_waits", {})
            if not isinstance(costs, dict) or set(costs) != set(ACTION_LIST):
                schema_errors.append(f"line {index}: incomplete candidate_action_waits")
            else:
                raw_actions[min(costs, key=costs.get)] += 1
            method = str(record.get("relabel_method", ""))
            relabel_methods[method] += 1
            if expected_neighbor_mode is not None:
                neighbor_mode = record.get("neighbor_mode")
                counterfactual_mode = record.get("counterfactual_mode")
                neighbor_modes[str(neighbor_mode)] += 1
                counterfactual_modes[str(counterfactual_mode)] += 1
                if neighbor_mode != expected_neighbor_mode:
                    schema_errors.append(
                        f"line {index}: neighbor_mode={neighbor_mode!r}, expected {expected_neighbor_mode!r}"
                    )
                expected_method = f"simexpert_policy_{expected_neighbor_mode}_neighbors"
                if method != expected_method:
                    schema_errors.append(
                        f"line {index}: relabel_method={method!r}, expected {expected_method!r}"
                    )

    if not records:
        raise ValueError(f"No records in {path}")
    provenance = {
        key: records[0].get(key)
        for key in ("sim_horizon", "relabel_method", "cost_scope", "source_policy_checkpoint")
        if key in records[0]
    }
    return {
        "path": str(path),
        "sha256": sha256(path),
        "records": len(records),
        "unique_snapshots": len(snapshot_paths),
        "missing_snapshot_references": missing_snapshots,
        "expert_action_distribution": dict(expert_actions),
        "raw_best_distribution": dict(raw_actions),
        "relabel_method_distribution": dict(relabel_methods),
        "neighbor_mode_distribution": dict(neighbor_modes),
        "counterfactual_mode_distribution": dict(counterfactual_modes),
        "schema_error_count": len(schema_errors),
        "schema_error_examples": schema_errors[:20],
        "provenance": provenance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--responsive", required=True, type=Path)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--artifact", action="append", default=[], type=Path)
    args = parser.parse_args()

    for path in (args.input, args.responsive, args.frozen, *args.artifact):
        if not path.is_file():
            raise FileNotFoundError(path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "study": "PRCT single-dataset Jinan-to-Hangzhou matched teacher control",
        "input": inspect_jsonl(args.input),
        "responsive_h120": inspect_jsonl(args.responsive, expected_neighbor_mode="responsive"),
        "frozen_h120": inspect_jsonl(args.frozen, expected_neighbor_mode="frozen"),
        "artifacts": {str(path): sha256(path) for path in args.artifact},
    }
    for label in ("input", "responsive_h120", "frozen_h120"):
        entry = summary[label]
        if entry["schema_error_count"] or entry["missing_snapshot_references"]:
            raise RuntimeError(f"Audit failed for {label}: {entry}")
    if summary["responsive_h120"]["records"] != summary["frozen_h120"]["records"]:
        raise RuntimeError("Teacher record counts differ")
    if summary["responsive_h120"]["unique_snapshots"] != summary["frozen_h120"]["unique_snapshots"]:
        raise RuntimeError("Teacher snapshot counts differ")

    output_json = args.output_dir / "prct_result_ledger.json"
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_csv = args.output_dir / "teacher_action_distribution.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["action", "responsive_expert", "frozen_expert", "responsive_raw", "frozen_raw"])
        writer.writeheader()
        for action in ACTION_LIST:
            writer.writerow({
                "action": action,
                "responsive_expert": summary["responsive_h120"]["expert_action_distribution"].get(action, 0),
                "frozen_expert": summary["frozen_h120"]["expert_action_distribution"].get(action, 0),
                "responsive_raw": summary["responsive_h120"]["raw_best_distribution"].get(action, 0),
                "frozen_raw": summary["frozen_h120"]["raw_best_distribution"].get(action, 0),
            })
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
