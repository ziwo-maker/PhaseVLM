#!/usr/bin/env python3
"""Aggregate the pre-registered direct-urgency Hangzhou Stage 2 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


ARMS = ("hard_bc", "equal_mixed_rank", "wait2_mixed_rank", "wait25_mixed_rank")
PRIMARY_TRAFFIC = "anon_4_4_hangzhou_real.json"
PRIMARY_EVAL_SEED = 2024
LOWER_IS_BETTER = {
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "state_p95_phase_wait",
    "state_max_phase_wait",
}
SUMMARY_METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "action_entropy",
    "state_p95_phase_wait",
    "state_max_phase_wait",
)


def read_status(arm_dir: Path) -> str:
    path = arm_dir / "status.txt"
    if not path.is_file():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("status="):
            return line.partition("=")[2].strip()
    return "unknown"


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low, high = math.floor(pos), math.ceil(pos)
    if low == high:
        return float(ordered[low])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (pos - low))


def state_action_safety(work_dir: Path) -> dict[str, Any]:
    path = work_dir / "state_action.json"
    if not path.is_file():
        return {"state_action_available": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    action_counts: Counter[str] = Counter()
    phase_waits: list[float] = []
    decisions = 0
    for intersection_log in payload if isinstance(payload, list) else []:
        for entry in intersection_log if isinstance(intersection_log, list) else []:
            if not isinstance(entry, dict):
                continue
            decisions += 1
            action = str(entry.get("action") or "")
            if action:
                action_counts[action] += 1
            waits = entry.get("phase_wait")
            if isinstance(waits, dict):
                for value in waits.values():
                    try:
                        phase_waits.append(float(value))
                    except (TypeError, ValueError):
                        pass
    total_actions = sum(action_counts.values())
    entropy = None
    if total_actions:
        probs = [count / total_actions for count in action_counts.values()]
        entropy = float(-sum(prob * math.log(prob) for prob in probs if prob > 0.0))
    return {
        "state_action_available": True,
        "decision_count": decisions,
        "action_counts": dict(sorted(action_counts.items())),
        "action_entropy": entropy,
        "state_p95_phase_wait": quantile(phase_waits, 0.95),
        "state_max_phase_wait": max(phase_waits) if phase_waits else None,
        "tail_metric_semantics": "p95/max over logged per-phase average wait at decision points; not vehicle-level wait quantiles",
    }


def numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "values": values,
    }


def paired_comparison(rows: list[dict[str, Any]], candidate: str, reference: str) -> dict[str, Any]:
    by_arm_seed: dict[tuple[str, int], dict[str, Any]] = {(row["arm"], row["training_seed"]): row for row in rows}
    shared = sorted(
        seed for seed in {seed for arm, seed in by_arm_seed if arm == candidate}
        if (reference, seed) in by_arm_seed
    )
    metrics: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        pairs = [
            (float(by_arm_seed[(candidate, seed)][metric]), float(by_arm_seed[(reference, seed)][metric]))
            for seed in shared
            if by_arm_seed[(candidate, seed)].get(metric) is not None
            and by_arm_seed[(reference, seed)].get(metric) is not None
        ]
        deltas = [left - right for left, right in pairs]
        favorable = [
            (delta < 0.0 if metric in LOWER_IS_BETTER else delta > 0.0)
            for delta in deltas
        ]
        metrics[metric] = {
            "pairs": len(pairs),
            "candidate_minus_reference": numeric_summary(deltas),
            "favorable_direction_count": int(sum(favorable)),
            "unfavorable_direction_count": int(len(favorable) - sum(favorable)),
            "two_sided_wilcoxon_p": exact_wilcoxon_p(deltas),
            "wilcoxon_note": "descriptive only; n=3 cannot support a strong significance claim",
        }
    return {
        "candidate": candidate,
        "reference": reference,
        "shared_training_seeds": shared,
        "metrics": metrics,
    }


def exact_wilcoxon_p(deltas: list[float]) -> float | None:
    nonzero = [abs(delta) for delta in deltas if abs(delta) > 1e-12]
    if not nonzero:
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(deltas, alternative="two-sided", method="exact").pvalue)
    except Exception:
        return None


def load_rows(run_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for arm in ARMS:
        arm_root = run_root / "arms" / arm
        for seed_dir in sorted(arm_root.glob("seed_*")) if arm_root.is_dir() else []:
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                excluded.append({"path": str(seed_dir), "reason": "invalid_seed_directory"})
                continue
            result_path = seed_dir / "evaluation" / "result.json"
            status = read_status(seed_dir)
            if status != "success" or not result_path.is_file():
                excluded.append({"path": str(seed_dir), "arm": arm, "training_seed": seed, "reason": f"status={status}; result_present={result_path.is_file()}"})
                continue
            result = json.loads(result_path.read_text(encoding="utf-8"))
            args = result.get("args") if isinstance(result.get("args"), dict) else {}
            if args.get("traffic_file") != PRIMARY_TRAFFIC or int(args.get("seed", -1)) != PRIMARY_EVAL_SEED:
                excluded.append({"path": str(seed_dir), "arm": arm, "training_seed": seed, "reason": "non_primary_evaluation_protocol"})
                continue
            if any(float(args.get(key, 0.0) or 0.0) != 0.0 for key in (
                "max_phase_override_gap",
                "max_phase_override_left_wait_protect",
                "max_phase_override_target_queue_floor",
                "max_phase_override_total_queue_floor",
            )):
                excluded.append({"path": str(seed_dir), "arm": arm, "training_seed": seed, "reason": "online_override_enabled"})
                continue
            metrics = result.get("results") if isinstance(result.get("results"), dict) else {}
            if not all(key in metrics for key in SUMMARY_METRICS[:4]):
                excluded.append({"path": str(seed_dir), "arm": arm, "training_seed": seed, "reason": "missing_closed_loop_metric"})
                continue
            work_dir = Path(str(result.get("work_dir") or ""))
            if not work_dir.is_absolute():
                work_dir = Path.cwd() / work_dir
            row: dict[str, Any] = {
                "arm": arm,
                "training_seed": seed,
                "result_path": str(result_path),
                "work_dir": str(work_dir),
                **{key: float(metrics[key]) for key in SUMMARY_METRICS[:4]},
                **state_action_safety(work_dir),
            }
            valid.append(row)
    return valid, excluded


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["arm", "training_seed", *SUMMARY_METRICS, "decision_count", "state_action_available", "result_path", "work_dir"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Hangzhou_real Direct-Urgency Stage 2 Aggregate",
        "",
        "Primary evaluation: `anon_4_4_hangzhou_real.json`, 3,600 seconds, fixed online seed `2024`, no max-phase override.",
        "Statistical unit: independent training seed. Lower is better for queue, wait, and travel; higher is better for reward.",
        "",
        "| Arm | n | Reward | Avg queue | Avg wait | Avg travel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        stats = aggregate["by_arm"].get(arm, {})
        def fmt(metric: str) -> str:
            item = stats.get(metric, {})
            mean, std = item.get("mean"), item.get("std")
            return "NA" if mean is None else f"{mean:.3f} +/- {std:.3f}"
        lines.append(f"| {arm} | {stats.get('n', 0)} | {fmt('test_reward_over')} | {fmt('test_avg_queue_len_over')} | {fmt('test_avg_waiting_time_over')} | {fmt('test_avg_travel_time_over')} |")
    lines += [
        "",
        "Tail metrics are p95/max of the logged per-phase *average* wait at decision points, not vehicle-level wait quantiles.",
        "Paired comparison details, protocol validation, raw rows, and excluded/incomplete runs are in `aggregate.json` and `raw_results.csv`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    rows, excluded = load_rows(root)
    by_arm: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in rows if row["arm"] == arm]
        by_arm[arm] = {"n": len(arm_rows), **{
            metric: numeric_summary([float(row[metric]) for row in arm_rows if row.get(metric) is not None])
            for metric in SUMMARY_METRICS
        }}
    aggregate = {
        "schema": "hangzhou_direct_urgency_aggregate_v1",
        "primary_protocol": {
            "traffic_file": PRIMARY_TRAFFIC,
            "run_counts": 3600,
            "online_seed": PRIMARY_EVAL_SEED,
            "online_phase_override": "disabled",
            "statistical_unit": "training_seed",
        },
        "completed_rows": rows,
        "excluded_or_incomplete": excluded,
        "by_arm": by_arm,
        "paired_comparisons": [
            paired_comparison(rows, "wait2_mixed_rank", "equal_mixed_rank"),
            paired_comparison(rows, "wait2_mixed_rank", "hard_bc"),
            paired_comparison(rows, "wait25_mixed_rank", "wait2_mixed_rank"),
        ],
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(root / "raw_results.csv", rows)
    write_markdown(root / "AGGREGATE.md", aggregate)
    print(json.dumps({"completed": len(rows), "incomplete_or_excluded": len(excluded), "by_arm": {arm: by_arm[arm]["n"] for arm in ARMS}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
