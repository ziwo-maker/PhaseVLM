#!/usr/bin/env python3
"""Aggregate the 9-seed, multi-demand DUCPD robustness campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("hard_bc", "wait2_mixed_rank")
SCENARIOS = (
    "primary_real",
    "source_5816",
    "synthetic_high",
    "time_early",
    "time_mid",
    "time_late",
)
SCENARIO_METADATA = {
    "primary_real": ("anon_4_4_hangzhou_real.json", 3600, "target full demand"),
    "source_5816": ("anon_4_4_hangzhou_real_5816.json", 3600, "training-source demand reference"),
    "synthetic_high": ("anon_4_4_hangzhou_synthetic_24000_60min.json", 3600, "synthetic high-demand stress"),
    "time_early": ("anon_4_4_hangzhou_real_t0000_1200.json", 1200, "target-real early time slice"),
    "time_mid": ("anon_4_4_hangzhou_real_t1200_2400.json", 1200, "target-real middle time slice"),
    "time_late": ("anon_4_4_hangzhou_real_t2400_3600.json", 1200, "target-real late time slice"),
}
METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
    "test_vehicle_wait_observed_vehicle_count",
)
LOWER_IS_BETTER = set(METRICS) - {"test_reward_over", "test_vehicle_wait_observed_vehicle_count"}


def status_of(path: Path) -> str:
    if not path.is_file():
        return "missing"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("status="):
            return line.partition("=")[2].strip()
    return "unknown"


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "values": values,
    }


def wilcoxon_p(deltas: list[float]) -> float | None:
    if not any(abs(delta) > 1e-12 for delta in deltas):
        return 1.0
    try:
        from scipy.stats import wilcoxon

        return float(wilcoxon(deltas, alternative="two-sided", method="exact").pvalue)
    except Exception:
        return None


def load_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        expected_traffic, expected_counts, _ = SCENARIO_METADATA[scenario]
        for arm in ARMS:
            arm_root = root / "evaluations" / scenario / arm
            for seed_dir in sorted(arm_root.glob("seed_*")) if arm_root.is_dir() else []:
                try:
                    seed = int(seed_dir.name.removeprefix("seed_"))
                except ValueError:
                    excluded.append({"path": str(seed_dir), "reason": "invalid_seed_directory"})
                    continue
                result_path = seed_dir / "result.json"
                if status_of(seed_dir / "status.txt") != "success" or not result_path.is_file():
                    excluded.append({"scenario": scenario, "arm": arm, "training_seed": seed, "path": str(seed_dir), "reason": "missing_or_unsuccessful_evaluation"})
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                args = result.get("args") if isinstance(result.get("args"), dict) else {}
                metrics = result.get("results") if isinstance(result.get("results"), dict) else {}
                if args.get("traffic_file") != expected_traffic or int(args.get("run_counts", -1)) != expected_counts:
                    excluded.append({"scenario": scenario, "arm": arm, "training_seed": seed, "path": str(seed_dir), "reason": "protocol_mismatch"})
                    continue
                if any(float(args.get(key, 0.0) or 0.0) != 0.0 for key in (
                    "max_phase_override_gap",
                    "max_phase_override_left_wait_protect",
                    "max_phase_override_target_queue_floor",
                    "max_phase_override_total_queue_floor",
                )):
                    excluded.append({"scenario": scenario, "arm": arm, "training_seed": seed, "path": str(seed_dir), "reason": "online_override_enabled"})
                    continue
                missing = [metric for metric in METRICS if metric not in metrics]
                if missing:
                    excluded.append({"scenario": scenario, "arm": arm, "training_seed": seed, "path": str(seed_dir), "reason": f"missing_metrics:{','.join(missing)}"})
                    continue
                rows.append({
                    "scenario": scenario,
                    "arm": arm,
                    "training_seed": seed,
                    "result_path": str(result_path),
                    "checkpoint": str(args.get("projection_ckpt") or ""),
                    **{metric: float(metrics[metric]) for metric in METRICS},
                })
    return rows, excluded


def paired(rows: list[dict[str, Any]], scenario: str) -> dict[str, Any]:
    table = {(row["arm"], row["training_seed"]): row for row in rows if row["scenario"] == scenario}
    seeds = sorted(seed for arm, seed in table if arm == "wait2_mixed_rank" and ("hard_bc", seed) in table)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        deltas = [float(table[("wait2_mixed_rank", seed)][metric]) - float(table[("hard_bc", seed)][metric]) for seed in seeds]
        favorable = [(delta < 0.0 if metric in LOWER_IS_BETTER else delta > 0.0) for delta in deltas]
        metrics[metric] = {
            "candidate_minus_reference": summary(deltas),
            "favorable_direction_count": int(sum(favorable)),
            "unfavorable_direction_count": int(len(favorable) - sum(favorable)),
            "two_sided_wilcoxon_p": wilcoxon_p(deltas),
            "wilcoxon_note": "descriptive unless the registered analysis declares otherwise",
        }
    return {"candidate": "wait2_mixed_rank", "reference": "hard_bc", "shared_training_seeds": seeds, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    rows, excluded = load_rows(root)
    by_scenario: dict[str, Any] = {}
    for scenario in SCENARIOS:
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arm_rows = [row for row in rows if row["scenario"] == scenario and row["arm"] == arm]
            arms[arm] = {"n": len(arm_rows), **{metric: summary([row[metric] for row in arm_rows]) for metric in METRICS}}
        by_scenario[scenario] = {"protocol": dict(zip(("traffic_file", "run_counts", "role"), SCENARIO_METADATA[scenario])), "by_arm": arms, "paired_wait2_minus_hard_bc": paired(rows, scenario)}

    aggregate = {
        "schema": "hangzhou_ducpd_robustness_aggregate_v1",
        "statistical_unit": "training_seed",
        "evaluation_type": "CityFlow closed-loop simulation",
        "vehicle_tail_semantics": "per unique vehicle maximum continuous wait observed at an action boundary; not a completed-trip wait distribution",
        "completed_rows": rows,
        "excluded_or_incomplete": excluded,
        "by_scenario": by_scenario,
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (root / "raw_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scenario", "arm", "training_seed", *METRICS, "checkpoint", "result_path"])
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Hangzhou DUCPD Robustness Aggregate", "", "Statistical unit: independently trained seed. All results are CityFlow closed-loop simulations.", "", "Vehicle tail: p95/max across each unique vehicle's maximum continuous wait observed at action boundaries.", ""]
    for scenario in SCENARIOS:
        item = by_scenario[scenario]
        protocol = item["protocol"]
        lines += [f"## {scenario}", "", f"`{protocol['traffic_file']}`, {protocol['run_counts']} s, {protocol['role']}.", "", "| Arm | n | Avg queue | Avg wait | Avg travel | Vehicle wait p95 | Vehicle wait max |", "|---|---:|---:|---:|---:|---:|---:|"]
        for arm in ARMS:
            stats = item["by_arm"][arm]
            def fmt(metric: str) -> str:
                value = stats[metric]
                return "NA" if value["mean"] is None else f"{value['mean']:.3f} +/- {value['std']:.3f}"
            lines.append(f"| {arm} | {stats['n']} | {fmt('test_avg_queue_len_over')} | {fmt('test_avg_waiting_time_over')} | {fmt('test_avg_travel_time_over')} | {fmt('test_vehicle_wait_p95_over')} | {fmt('test_vehicle_wait_max_over')} |")
        lines.append("")
    lines += ["Complete rows, paired deltas, and excluded tasks are in `aggregate.json` and `raw_results.csv`."]
    (root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"completed": len(rows), "excluded": len(excluded)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
