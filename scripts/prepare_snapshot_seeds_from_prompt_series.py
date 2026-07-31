#!/usr/bin/env python3
"""Select high-load prompt-series snapshots for Stage 3 snapshot-reset PPO."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


MOVEMENTS = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
DEFAULT_TARGET_INTERSECTIONS = ("intersection_1_4", "intersection_4_4", "intersection_1_3")


def parse_seconds(date_s: str) -> int:
    hms = str(date_s or "").split()[-1]
    h, m, s = hms.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def load_action_rows(path: Path, start_s: int, end_s: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("prompt") or not record.get("snapshot_path"):
                continue
            try:
                seconds = parse_seconds(str(record.get("date") or ""))
            except Exception:
                continue
            if seconds < start_s or seconds > end_s:
                continue
            record["_seconds"] = seconds
            rows.append(record)
    return rows


def record_load(record: dict[str, Any]) -> tuple[float, float, float]:
    counts = record.get("movement_counts") or {}
    total = sum(float(counts.get(key, 0.0) or 0.0) for key in MOVEMENTS)
    through_ew = float(counts.get("ET", 0.0) or 0.0) + float(counts.get("WT", 0.0) or 0.0)
    left = (
        float(counts.get("EL", 0.0) or 0.0)
        + float(counts.get("WL", 0.0) or 0.0)
        + float(counts.get("NL", 0.0) or 0.0)
        + float(counts.get("SL", 0.0) or 0.0)
    )
    return total, through_ew, left


def score_group(records: list[dict[str, Any]], target_intersections: set[str]) -> float:
    best_target = 0.0
    best_any = 0.0
    target_bonus = 0.0
    for record in records:
        total, through_ew, left = record_load(record)
        score = total + 0.75 * through_ew + 0.25 * left
        best_any = max(best_any, score)
        if record.get("intersection_id") in target_intersections:
            best_target = max(best_target, score)
            target_bonus += 0.05 * score
    return 2.0 * best_target + 0.5 * best_any + target_bonus


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="traffic_prompt_series_*.jsonl")
    parser.add_argument("--output", required=True, help="Filtered snapshot seed jsonl")
    parser.add_argument("--start_s", type=int, default=900)
    parser.add_argument("--end_s", type=int, default=2700)
    parser.add_argument("--max_snapshots", type=int, default=48)
    parser.add_argument(
        "--target_intersections",
        type=str,
        default=",".join(DEFAULT_TARGET_INTERSECTIONS),
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    target_intersections = {
        item.strip() for item in str(args.target_intersections or "").split(",") if item.strip()
    }

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in load_action_rows(input_path, args.start_s, args.end_s):
        grouped[str(record.get("snapshot_path") or "")].append(record)

    ranked = sorted(
        grouped.items(),
        key=lambda item: score_group(item[1], target_intersections),
        reverse=True,
    )
    selected = ranked[: max(1, int(args.max_snapshots))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    selected_snapshot_paths = []
    with output_path.open("w", encoding="utf-8") as handle:
        for snapshot_path, records in selected:
            selected_snapshot_paths.append(snapshot_path)
            for record in sorted(records, key=lambda row: str(row.get("intersection_id") or "")):
                record = dict(record)
                record.pop("_seconds", None)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_rows += 1

    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "start_s": args.start_s,
        "end_s": args.end_s,
        "max_snapshots": args.max_snapshots,
        "target_intersections": sorted(target_intersections),
        "candidate_snapshots": len(grouped),
        "selected_snapshots": len(selected),
        "written_rows": written_rows,
        "selected_snapshot_paths_preview": selected_snapshot_paths[:10],
        "cwd": os.getcwd(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
