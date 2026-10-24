#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


ARMS = ("full", "zero_prefix", "shuffled_image", "constant_image")
METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_queuing_vehicle_num_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, required=True)
    parser.add_argument("--expected_seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected_seeds = sorted(set(args.expected_seeds))
    aggregate: dict[str, object] = {
        "run_root": str(args.run_root.resolve()),
        "expected_arms": list(ARMS),
        "expected_seeds": expected_seeds,
        "complete": True,
        "missing": [],
        "arms": {},
    }
    reference: dict[str, str] = {}

    for arm in ARMS:
        rows = []
        for seed in expected_seeds:
            result_path = args.run_root / "results" / arm / f"seed_{seed}" / "result.json"
            if not result_path.is_file():
                aggregate["complete"] = False
                aggregate["missing"].append(str(result_path))
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            payload_args = payload.get("args") or {}
            hashes = payload.get("artifact_hashes") or {}
            if payload_args.get("visual_ablation") != arm:
                raise ValueError(f"{result_path}: visual_ablation mismatch")
            invariant = {
                "projection_ckpt_sha256": str(hashes.get("projection_ckpt_sha256", "")),
                "prompt_mode": str(payload.get("prompt_mode", "")),
                "llm_path": str((payload.get("runtime") or {}).get("llm_path", "")),
            }
            if not reference:
                reference = invariant
            elif invariant != reference:
                raise ValueError(f"{result_path}: invariant mismatch {invariant} != {reference}")
            rows.append({"seed": seed, **(payload.get("results") or {})})

        arm_summary: dict[str, object] = {"n": len(rows), "runs": rows, "metrics": {}}
        for metric in METRICS:
            values = [float(row[metric]) for row in rows if metric in row]
            if not values:
                continue
            arm_summary["metrics"][metric] = {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
        aggregate["arms"][arm] = arm_summary

    aggregate["invariants"] = reference
    args.run_root.mkdir(parents=True, exist_ok=True)
    aggregate_path = args.run_root / "aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with (args.run_root / "aggregate.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "n", "metric", "mean", "std"])
        for arm in ARMS:
            arm_summary = aggregate["arms"][arm]
            for metric, values in arm_summary["metrics"].items():
                writer.writerow([arm, arm_summary["n"], metric, values["mean"], values["std"]])

    lines = [
        "# 8B Visual-Signal Ablation",
        "",
        f"Complete: `{aggregate['complete']}`",
        "",
        "| Arm | n | Reward | AQL | AWT | ATT |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    short_metrics = (
        "test_reward_over",
        "test_avg_queue_len_over",
        "test_avg_waiting_time_over",
        "test_avg_travel_time_over",
    )
    for arm in ARMS:
        arm_summary = aggregate["arms"][arm]
        rendered = []
        for metric in short_metrics:
            values = arm_summary["metrics"].get(metric)
            rendered.append("NA" if values is None else f"{values['mean']:.4f} +/- {values['std']:.4f}")
        lines.append(f"| {arm} | {arm_summary['n']} | " + " | ".join(rendered) + " |")
    (args.run_root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if aggregate["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
