#!/usr/bin/env python3
"""Aggregate the Jinan-real-only Stage-1 direct-urgency campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("hard_bc", "wait2_mixed_rank", "wait225_mixed_rank")
TRAINING_SEEDS = (2024, 42, 123)
EVAL_SEEDS = (2024, 42, 123)
PRIMARY_EVAL_SEED = 2024
FLOW = "anon_3_4_jinan_real.json"
METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
    "test_vehicle_wait_observed_vehicle_count",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid env manifest at {path}:{line_number}")
        values[key] = value
    return values


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "values": values,
    }


def load_result(
    path: Path,
    checkpoint: Path,
    checkpoint_hash: str,
    evaluator_hash: str,
    eval_seed: int,
) -> tuple[dict[str, float] | None, str | None]:
    if not path.is_file():
        return None, "missing_result"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        hashes = payload["artifact_hashes"]
        deterministic = payload["determinism"]
        assert args["traffic_file"] == FLOW
        assert int(args["seed"]) == eval_seed
        assert int(args["run_counts"]) == 3600
        assert Path(args["projection_ckpt"]).resolve() == checkpoint.resolve()
        assert hashes["projection_ckpt_sha256"] == checkpoint_hash
        assert hashes["online_evaluator_sha256"] == evaluator_hash
        assert args["deterministic"] is True
        assert deterministic["requested"] is True
        assert deterministic["torch_deterministic_algorithms"] is True
        assert deterministic["cudnn_deterministic"] is True
        assert deterministic["cudnn_benchmark"] is False
        assert deterministic["allow_tf32"] is False
        assert all(
            float(args.get(name, 0.0) or 0.0) == 0.0
            for name in (
                "max_phase_override_gap",
                "max_phase_override_left_wait_protect",
                "max_phase_override_target_queue_floor",
                "max_phase_override_total_queue_floor",
            )
        )
        values = {metric: float(payload["results"][metric]) for metric in METRICS}
        assert all(math.isfinite(value) for value in values.values())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_or_protocol_mismatched_result"
    return values, None


def paired(rows: list[dict[str, Any]], baseline: str, candidate: str) -> dict[str, Any]:
    by_arm_seed = {
        (row["arm"], int(row["training_seed"])): row
        for row in rows
        if int(row["eval_seed"]) == PRIMARY_EVAL_SEED
    }
    seeds = [
        seed
        for seed in TRAINING_SEEDS
        if (baseline, seed) in by_arm_seed and (candidate, seed) in by_arm_seed
    ]
    deltas: dict[str, Any] = {}
    wins: dict[str, int] = {}
    for metric in METRICS:
        values = [
            by_arm_seed[(candidate, seed)][metric] - by_arm_seed[(baseline, seed)][metric]
            for seed in seeds
        ]
        deltas[metric] = summary(values)
        if metric == "test_reward_over":
            wins[metric] = sum(1 for value in values if value > 0)
        else:
            wins[metric] = sum(1 for value in values if value < 0)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "paired_training_seeds": seeds,
        "metric_delta_candidate_minus_baseline": deltas,
        "seeds_favoring_candidate": wins,
    }


def fmt(summary_item: dict[str, Any]) -> str:
    mean = summary_item["mean"]
    return "NA" if mean is None else f"{mean:.3f} +/- {summary_item['std']:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    provenance = read_env(root / "provenance.env")
    if provenance.get("stage1_data_mode") != "jinan_real_only":
        raise ValueError("provenance does not identify a jinan_real_only Stage-1 run")
    if provenance.get("training_flow") != FLOW or provenance.get("evaluation_flow") != FLOW:
        raise ValueError("campaign violates same-flow protocol")
    evaluator_hash = provenance.get("file_projection_train_run_projection_online.py_sha256")
    if not evaluator_hash:
        raise ValueError("missing evaluator hash in provenance")

    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for arm in ARMS:
        for training_seed in TRAINING_SEEDS:
            arm_root = root / "arms" / arm / f"seed_{training_seed}"
            status_path = arm_root / "status.txt"
            checkpoint = arm_root / "stage2" / "best.pt"
            checkpoint_hash = sha256(checkpoint) if checkpoint.is_file() else ""
            reason: str | None = None
            if not status_path.is_file():
                reason = "missing_arm_status"
            elif not checkpoint.is_file():
                reason = "missing_checkpoint"
            else:
                status = read_env(status_path)
                if status.get("training_flow") != FLOW:
                    reason = "training_flow_mismatch"
                elif status.get("stage2_best_sha256") != checkpoint_hash:
                    reason = "checkpoint_hash_mismatch"
                elif status.get("online_status") != "success":
                    reason = "online_not_successful"

            for eval_seed in EVAL_SEEDS:
                result_path = (
                    arm_root
                    / "evaluation"
                    / FLOW.removesuffix(".json")
                    / f"seed_{eval_seed}"
                    / "result.json"
                )
                base = {
                    "arm": arm,
                    "training_seed": training_seed,
                    "flow": FLOW,
                    "eval_seed": eval_seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "result_path": str(result_path),
                }
                if reason:
                    incomplete.append({**base, "reason": reason})
                    continue
                values, failure = load_result(
                    result_path,
                    checkpoint,
                    checkpoint_hash,
                    evaluator_hash,
                    eval_seed,
                )
                if failure:
                    incomplete.append({**base, "reason": failure})
                else:
                    rows.append({**base, **(values or {})})

    primary = {
        arm: {
            metric: summary(
                [
                    row[metric]
                    for row in rows
                    if row["arm"] == arm and int(row["eval_seed"]) == PRIMARY_EVAL_SEED
                ]
            )
            for metric in METRICS
        }
        for arm in ARMS
    }
    descriptive = {
        arm: {
            metric: summary([row[metric] for row in rows if row["arm"] == arm])
            for metric in METRICS
        }
        for arm in ARMS
    }
    comparisons = {
        "wait2_vs_hard_bc": paired(rows, "hard_bc", "wait2_mixed_rank"),
        "wait225_vs_hard_bc": paired(rows, "hard_bc", "wait225_mixed_rank"),
        "wait225_vs_wait2": paired(rows, "wait2_mixed_rank", "wait225_mixed_rank"),
    }

    output = {
        "schema": "jinan_real_stage1_only_direct_urgency_aggregate_v1",
        "protocol": {
            "stage1_data_mode": "jinan_real_only",
            "training_flow": FLOW,
            "evaluation_flow": FLOW,
            "training_seeds": list(TRAINING_SEEDS),
            "primary_eval_seed": PRIMARY_EVAL_SEED,
            "descriptive_eval_seeds": list(EVAL_SEEDS),
            "run_counts": 3600,
            "online_phase_override": "disabled",
            "online_seed_role": "descriptive realization only; not independent training replicates",
        },
        "completed_rows": rows,
        "incomplete_rows": incomplete,
        "primary_by_arm": primary,
        "descriptive_all_online_seeds_by_arm": descriptive,
        "paired_primary_seed_comparisons": comparisons,
    }
    (root / "aggregate.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    result_fields = [
        "arm",
        "training_seed",
        "flow",
        "eval_seed",
        "checkpoint",
        "checkpoint_sha256",
        *METRICS,
        "result_path",
    ]
    incomplete_fields = [
        "arm",
        "training_seed",
        "flow",
        "eval_seed",
        "checkpoint",
        "checkpoint_sha256",
        "result_path",
        "reason",
    ]
    for name, contents, fields in (
        ("online_results.csv", rows, result_fields),
        ("incomplete_runs.csv", incomplete, incomplete_fields),
    ):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(contents)

    lines = [
        "# Jinan Real-Only Stage-1 Direct-Urgency Aggregate",
        "",
        "Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, deterministic online seed `2024`, no phase override.",
        "Stage 1 is trained only on `data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl`.",
        "Online realization seeds `42` and `123` are descriptive only.",
        "",
        "| Arm | n | Reward | AQL | AWT | ATT | P95 wait | Max wait |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        stats = primary[arm]
        lines.append(
            f"| `{arm}` | {stats['test_reward_over']['n']} | "
            f"{fmt(stats['test_reward_over'])} | "
            f"{fmt(stats['test_avg_queue_len_over'])} | "
            f"{fmt(stats['test_avg_waiting_time_over'])} | "
            f"{fmt(stats['test_avg_travel_time_over'])} | "
            f"{fmt(stats['test_vehicle_wait_p95_over'])} | "
            f"{fmt(stats['test_vehicle_wait_max_over'])} |"
        )
    lines.extend(
        (
            "",
            "Paired comparisons use matched training seeds and primary online seed `2024` only.",
            "Deltas are candidate minus baseline; positive Reward and negative AQL/AWT/ATT/P95/Max deltas are favorable.",
            "",
            "| Comparison | Paired n | Reward delta | AQL delta | AWT delta | ATT delta | P95 wait delta | Max wait delta |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        )
    )
    labels = {
        "wait2_vs_hard_bc": "`wait2_mixed_rank` - `hard_bc`",
        "wait225_vs_hard_bc": "`wait225_mixed_rank` - `hard_bc`",
        "wait225_vs_wait2": "`wait225_mixed_rank` - `wait2_mixed_rank`",
    }
    for name, comparison in comparisons.items():
        deltas = comparison["metric_delta_candidate_minus_baseline"]
        lines.append(
            f"| {labels[name]} | {len(comparison['paired_training_seeds'])} | "
            f"{fmt(deltas['test_reward_over'])} | "
            f"{fmt(deltas['test_avg_queue_len_over'])} | "
            f"{fmt(deltas['test_avg_waiting_time_over'])} | "
            f"{fmt(deltas['test_avg_travel_time_over'])} | "
            f"{fmt(deltas['test_vehicle_wait_p95_over'])} | "
            f"{fmt(deltas['test_vehicle_wait_max_over'])} |"
        )
    expected = len(ARMS) * len(TRAINING_SEEDS) * len(EVAL_SEEDS)
    audit = {
        "expected_completed_online_rows": expected,
        "completed_online_rows": len(rows),
        "incomplete_online_rows": len(incomplete),
        "status": "pass" if len(rows) == expected and not incomplete else "fail",
    }
    (root / "COMPLETION_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines.extend(
        (
            "",
            f"Completed online rows: {len(rows)}/{expected}.",
            f"Incomplete or invalid rows: {len(incomplete)}.",
            f"Completion audit: {audit['status']}.",
        )
    )
    (root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    if audit["status"] != "pass":
        raise SystemExit("completion audit failed")


if __name__ == "__main__":
    main()
