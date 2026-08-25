#!/usr/bin/env python3
"""Aggregate the Jinan Stage-1 x Stage-2 capacity factorial ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


FLOW = "anon_3_4_jinan_real.json"
TRAINING_SEEDS = (314, 9001, 2718)
EVAL_SEEDS = (2024, 42, 123)
PRIMARY_EVAL_SEED = 2024
EXPECTED_COMPLETED_ROWS = 36
METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
    "test_vehicle_wait_observed_vehicle_count",
)
CELLS = (
    {
        "cell": "stage1_pt8_stage2_pt8",
        "source": "source_root",
        "track": "seed_expansion",
        "arm": "wait225_mixed_rank",
        "stage1_arch": "pt8_hd64",
        "stage2_arch": "pt8_hd64",
    },
    {
        "cell": "stage1_pt8_stage2_pt16",
        "source": "run_root",
        "track": "stage1_pt8_stage2_pt16",
        "arm": "wait225_mixed_rank",
        "stage1_arch": "pt8_hd64",
        "stage2_arch": "pt16_hd128",
    },
    {
        "cell": "stage1_pt16_stage2_pt8",
        "source": "run_root",
        "track": "stage1_pt16_stage2_pt8",
        "arm": "wait225_mixed_rank",
        "stage1_arch": "pt16_hd128",
        "stage2_arch": "pt8_hd64",
    },
    {
        "cell": "stage1_pt16_stage2_pt16",
        "source": "source_root",
        "track": "capacity_pt16_hd128",
        "arm": "wait225_mixed_rank",
        "stage1_arch": "pt16_hd128",
        "stage2_arch": "pt16_hd128",
    },
)
COMPARISONS = (
    (
        "stage2_capacity_at_pt8_stage1",
        "stage1_pt8_stage2_pt8",
        "stage1_pt8_stage2_pt16",
        "Stage-2 capacity effect with Stage-1 fixed at pt8_hd64",
    ),
    (
        "stage1_capacity_at_pt8_stage2",
        "stage1_pt8_stage2_pt8",
        "stage1_pt16_stage2_pt8",
        "Stage-1 capacity effect with Stage-2 fixed at pt8_hd64",
    ),
    (
        "stage2_capacity_at_pt16_stage1",
        "stage1_pt16_stage2_pt8",
        "stage1_pt16_stage2_pt16",
        "Stage-2 capacity effect with Stage-1 fixed at pt16_hd128",
    ),
    (
        "stage1_capacity_at_pt16_stage2",
        "stage1_pt8_stage2_pt16",
        "stage1_pt16_stage2_pt16",
        "Stage-1 capacity effect with Stage-2 fixed at pt16_hd128",
    ),
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"invalid env manifest at {path}:{line_number}")
        values[key] = value
    return values


def stats(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "values": values,
    }


def load_online_result(
    path: Path,
    checkpoint: Path,
    checkpoint_hash: str,
    eval_seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, "missing_result"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        hashes = payload["artifact_hashes"]
        determinism = payload["determinism"]
        assert args["traffic_file"] == FLOW
        assert int(args["seed"]) == eval_seed and int(args["run_counts"]) == 3600
        assert Path(args["projection_ckpt"]).resolve() == checkpoint.resolve()
        assert hashes["projection_ckpt_sha256"] == checkpoint_hash
        assert args["deterministic"] is True and determinism["requested"] is True
        assert determinism["torch_deterministic_algorithms"] is True
        assert determinism["cudnn_deterministic"] is True
        assert determinism["cudnn_benchmark"] is False and determinism["allow_tf32"] is False
        assert all(
            float(args.get(name, 0.0) or 0.0) == 0.0
            for name in (
                "max_phase_override_gap",
                "max_phase_override_left_wait_protect",
                "max_phase_override_target_queue_floor",
                "max_phase_override_total_queue_floor",
            )
        )
        metrics = {name: float(payload["results"][name]) for name in METRICS}
        assert all(math.isfinite(value) for value in metrics.values())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_or_protocol_mismatched_result"
    return {**metrics, "evaluator_sha256": str(hashes["online_evaluator_sha256"])}, None


def online_status_matches_checkpoint(path: Path, checkpoint: Path, checkpoint_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        manifest = parse_env(path)
        return (
            manifest.get("status") == "success"
            and Path(manifest["checkpoint"]).resolve() == checkpoint.resolve()
            and manifest.get("checkpoint_sha256") == checkpoint_hash
            and manifest.get("evaluation_flow") == FLOW
            and manifest.get("deterministic") == "true"
        )
    except (KeyError, OSError, ValueError):
        return False


def format_stat(summary: dict[str, Any]) -> str:
    mean = summary["mean"]
    return "NA" if mean is None else f"{mean:.3f} +/- {summary['std']:.3f}"


def paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row["eval_seed"] != PRIMARY_EVAL_SEED:
            continue
        indexed.setdefault(row["cell"], {})[row["training_seed"]] = row

    output: dict[str, Any] = {}
    for name, baseline_cell, candidate_cell, description in COMPARISONS:
        baseline_rows = indexed.get(baseline_cell, {})
        candidate_rows = indexed.get(candidate_cell, {})
        paired_seeds = [seed for seed in TRAINING_SEEDS if seed in baseline_rows and seed in candidate_rows]
        output[name] = {
            "description": description,
            "baseline": baseline_cell,
            "candidate": candidate_cell,
            "paired_training_seeds": paired_seeds,
            "metric_delta_candidate_minus_baseline": {
                metric: stats([candidate_rows[seed][metric] - baseline_rows[seed][metric] for seed in paired_seeds])
                for metric in METRICS
            },
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run_root.resolve()
    source_root = args.source_root.resolve()

    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for cell_info in CELLS:
        base_root = source_root if cell_info["source"] == "source_root" else run_root
        for training_seed in TRAINING_SEEDS:
            arm_root = (
                base_root
                / cell_info["track"]
                / "arms"
                / cell_info["arm"]
                / f"seed_{training_seed}"
            )
            checkpoint = arm_root / "stage2" / "best.pt"
            status_path = arm_root / "status.txt"
            online_status_path = arm_root / "online.status"
            manifest: dict[str, str] = {}
            problem: str | None = None
            if not status_path.is_file():
                problem = "missing_arm_status"
            else:
                try:
                    manifest = parse_env(status_path)
                except ValueError:
                    problem = "invalid_arm_status"
            checkpoint_hash = digest(checkpoint) if checkpoint.is_file() else ""
            if problem is None and not checkpoint.is_file():
                problem = "missing_checkpoint"
            if problem is None and manifest.get("training_flow") != FLOW:
                problem = "training_flow_mismatch"
            if problem is None and manifest.get("stage2_best_sha256") not in (None, checkpoint_hash):
                problem = "checkpoint_hash_mismatch"
            if problem is None and not online_status_matches_checkpoint(
                online_status_path, checkpoint, checkpoint_hash
            ):
                problem = "missing_or_invalid_online_status"
            if problem is None and manifest.get("stage2_status") not in (None, "success"):
                problem = "stage2_not_successful"
            if problem is None and manifest.get("stage2_status") is None and manifest.get("status") != "success":
                if manifest.get("online_status") not in (None, "success"):
                    problem = "legacy_arm_not_recovered"

            for eval_seed in EVAL_SEEDS:
                result = arm_root / "evaluation" / FLOW.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
                base = {
                    "cell": cell_info["cell"],
                    "stage1_arch": cell_info["stage1_arch"],
                    "stage2_arch": cell_info["stage2_arch"],
                    "track": cell_info["track"],
                    "arm": cell_info["arm"],
                    "source_root": str(base_root),
                    "training_seed": training_seed,
                    "flow": FLOW,
                    "eval_seed": eval_seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_hash,
                    "result_path": str(result),
                    "extra_train_args": manifest.get("extra_train_args", ""),
                }
                if problem:
                    incomplete.append({**base, "reason": problem})
                    continue
                values, failure = load_online_result(result, checkpoint, checkpoint_hash, eval_seed)
                if failure:
                    incomplete.append({**base, "reason": failure})
                else:
                    rows.append({**base, **(values or {})})

    primary_by_cell: dict[str, Any] = {}
    descriptive_by_cell: dict[str, Any] = {}
    for cell_info in CELLS:
        cell = cell_info["cell"]
        primary = [row for row in rows if row["cell"] == cell and row["eval_seed"] == PRIMARY_EVAL_SEED]
        descriptive = [row for row in rows if row["cell"] == cell]
        primary_by_cell[cell] = {metric: stats([row[metric] for row in primary]) for metric in METRICS}
        descriptive_by_cell[cell] = {metric: stats([row[metric] for row in descriptive]) for metric in METRICS}

    comparisons = paired_comparisons(rows)
    output = {
        "schema": "jinan_capacity_stage1_factorial_v1",
        "protocol": {
            "training_flow": FLOW,
            "evaluation_flow": FLOW,
            "training_seeds": list(TRAINING_SEEDS),
            "primary_eval_seed": PRIMARY_EVAL_SEED,
            "descriptive_eval_seeds": list(EVAL_SEEDS),
            "run_counts": 3600,
            "online_phase_override": "disabled",
            "source_root": str(source_root),
            "interpretation_limits": {
                "cross_arch_transfer": (
                    "Mismatched Stage-1/Stage-2 architecture cells use "
                    "--allow_mismatched_align_projectors and copy overlapping "
                    "projector weights; they test transfer under architecture "
                    "change, not a pure same-shape initializer replacement."
                )
            },
        },
        "completed_rows": rows,
        "incomplete_rows": incomplete,
        "primary_by_cell": primary_by_cell,
        "descriptive_all_online_seeds_by_cell": descriptive_by_cell,
        "paired_primary_seed_comparisons": comparisons,
    }
    (run_root / "aggregate.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    result_fields = [
        "cell",
        "stage1_arch",
        "stage2_arch",
        "track",
        "arm",
        "source_root",
        "training_seed",
        "flow",
        "eval_seed",
        "checkpoint",
        "checkpoint_sha256",
        "extra_train_args",
        *METRICS,
        "evaluator_sha256",
        "result_path",
    ]
    incomplete_fields = [
        "cell",
        "stage1_arch",
        "stage2_arch",
        "track",
        "arm",
        "source_root",
        "training_seed",
        "flow",
        "eval_seed",
        "checkpoint",
        "checkpoint_sha256",
        "extra_train_args",
        "result_path",
        "reason",
    ]
    for name, contents, fields in (
        ("online_results.csv", rows, result_fields),
        ("incomplete_runs.csv", incomplete, incomplete_fields),
    ):
        with (run_root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(contents)

    lines = [
        "# Jinan Stage-1 x Stage-2 Capacity Factorial",
        "",
        "Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, deterministic online seed `2024`, no phase override.",
        "Rows include two reused source cells and two newly trained cross-architecture cells.",
        "",
        "| Cell | Stage 1 | Stage 2 | n | Reward | AQL | AWT | ATT | P95 wait |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell_info in CELLS:
        cell = cell_info["cell"]
        summary = primary_by_cell[cell]
        lines.append(
            f"| {cell} | {cell_info['stage1_arch']} | {cell_info['stage2_arch']} | "
            f"{summary['test_reward_over']['n']} | "
            f"{format_stat(summary['test_reward_over'])} | "
            f"{format_stat(summary['test_avg_queue_len_over'])} | "
            f"{format_stat(summary['test_avg_waiting_time_over'])} | "
            f"{format_stat(summary['test_avg_travel_time_over'])} | "
            f"{format_stat(summary['test_vehicle_wait_p95_over'])} |"
        )

    lines.extend(
        (
            "",
            "Paired comparisons use matched training seeds and primary online seed `2024` only.",
            "Metric deltas are candidate minus baseline; positive reward and negative delay/queue/tail deltas are favorable.",
            "Cross-architecture cells use overlapping projector weight transfer for mismatched tensor shapes.",
            "",
            "| Comparison | Paired n | Reward delta | AQL delta | AWT delta | ATT delta | P95 wait delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for comparison in comparisons.values():
        delta = comparison["metric_delta_candidate_minus_baseline"]
        lines.append(
            f"| {comparison['description']} | {len(comparison['paired_training_seeds'])} | "
            f"{format_stat(delta['test_reward_over'])} | "
            f"{format_stat(delta['test_avg_queue_len_over'])} | "
            f"{format_stat(delta['test_avg_waiting_time_over'])} | "
            f"{format_stat(delta['test_avg_travel_time_over'])} | "
            f"{format_stat(delta['test_vehicle_wait_p95_over'])} |"
        )

    audit = {
        "expected_completed_online_rows": EXPECTED_COMPLETED_ROWS,
        "completed_online_rows": len(rows),
        "incomplete_online_rows": len(incomplete),
        "status": "pass" if len(rows) == EXPECTED_COMPLETED_ROWS and not incomplete else "fail",
    }
    (run_root / "COMPLETION_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines.extend(
        (
            "",
            f"Completed online rows: {len(rows)}/{EXPECTED_COMPLETED_ROWS}.",
            f"Incomplete or invalid rows: {len(incomplete)}.",
            f"Completion audit: {audit['status']}.",
        )
    )
    (run_root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    if audit["status"] != "pass":
        raise SystemExit("completion audit failed")


if __name__ == "__main__":
    main()
