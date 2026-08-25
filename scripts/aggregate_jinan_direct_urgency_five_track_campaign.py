#!/usr/bin/env python3
"""Strict aggregate for the Jinan direct-urgency five-track campaign.

The campaign deliberately uses one traffic flow and deterministic online runs.
This reporter preserves that contract, records incomplete arms explicitly, and
does not treat online realization seeds as independent training replicates.
"""

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
EXPECTED_COMPLETED_ROWS = 45
METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
    "test_vehicle_wait_observed_vehicle_count",
)
TRACKS = {
    "seed_expansion": ("wait2_mixed_rank", "wait225_mixed_rank"),
    "tail_direct_labels": ("tail_p95_mixed_rank",),
    "stage1_sameflow": ("wait225_mixed_rank",),
    "capacity_pt16_hd128": ("wait225_mixed_rank",),
}
COMPARISONS = (
    # Candidate minus baseline.  These pair the same training seed and use
    # only PRIMARY_EVAL_SEED, so online realization seeds are never inflated
    # into extra independent replicates.
    (
        "wait225_vs_wait2",
        "seed_expansion/wait2_mixed_rank",
        "seed_expansion/wait225_mixed_rank",
        "AWT cost-weight refinement",
    ),
    (
        "tail_p95_vs_wait2",
        "seed_expansion/wait2_mixed_rank",
        "tail_direct_labels/tail_p95_mixed_rank",
        "direct P95-tail labels",
    ),
    (
        "stage1_sameflow_vs_wait225",
        "seed_expansion/wait225_mixed_rank",
        "stage1_sameflow/wait225_mixed_rank",
        "Stage-1 same-flow initializer",
    ),
    (
        "capacity_vs_wait225",
        "seed_expansion/wait225_mixed_rank",
        "capacity_pt16_hd128/wait225_mixed_rank",
        "larger Stage-1/Stage-2 architecture (matched three-flow Stage-1 data)",
    ),
    (
        "capacity_bundle_vs_stage1_sameflow",
        "stage1_sameflow/wait225_mixed_rank",
        "capacity_pt16_hd128/wait225_mixed_rank",
        "architecture plus three-flow Stage-1 versus same-flow Stage-1",
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


def checkpoint_reference(root: Path) -> dict[str, Any]:
    path = root / "checkpoint_tiebreak" / "epoch8_result.json"
    if not path.is_file():
        return {"result_path": str(path), "status": "missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        values = {metric: float(payload["results"][metric]) for metric in METRICS}
        assert all(math.isfinite(value) for value in values.values())
        return {
            "result_path": str(path),
            "status": "valid",
            "checkpoint": payload["args"]["projection_ckpt"],
            "checkpoint_sha256": payload["artifact_hashes"]["projection_ckpt_sha256"],
            "evaluator_sha256": payload["artifact_hashes"]["online_evaluator_sha256"],
            **values,
        }
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError, AssertionError):
        return {"result_path": str(path), "status": "invalid"}


def online_status_matches_checkpoint(path: Path, checkpoint: Path, checkpoint_hash: str) -> bool:
    """Accept legacy and current arm wrappers only with identical online evidence."""
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


def primary_rows_by_key(rows: list[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    indexed: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if row["eval_seed"] != PRIMARY_EVAL_SEED:
            continue
        key = f"{row['track']}/{row['arm']}"
        indexed.setdefault(key, {})[row["training_seed"]] = row
    return indexed


def paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = primary_rows_by_key(rows)
    output: dict[str, Any] = {}
    for name, baseline_key, candidate_key, description in COMPARISONS:
        baseline_rows = indexed.get(baseline_key, {})
        candidate_rows = indexed.get(candidate_key, {})
        paired_seeds = [seed for seed in TRAINING_SEEDS if seed in baseline_rows and seed in candidate_rows]
        output[name] = {
            "description": description,
            "baseline": baseline_key,
            "candidate": candidate_key,
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
    args = parser.parse_args()
    root = args.run_root.resolve()
    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []

    for track, arms in TRACKS.items():
        for arm in arms:
            for training_seed in TRAINING_SEEDS:
                arm_root = root / track / "arms" / arm / f"seed_{training_seed}"
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
                # Earlier arms recorded a single terminal `status=success` while
                # newer wrappers also expose `stage2_status=success`.  Both need
                # independent online-status evidence tied to this exact checkpoint.
                if problem is None and not online_status_matches_checkpoint(
                    online_status_path, checkpoint, checkpoint_hash
                ):
                    problem = "missing_or_invalid_online_status"
                if problem is None and manifest.get("stage2_status") not in (None, "success"):
                    problem = "stage2_not_successful"
                if problem is None and manifest.get("stage2_status") is None and manifest.get("status") != "success":
                    # A recovered legacy wrapper can still be accepted, but only
                    # because the checkpoint-bound online manifest above is valid.
                    if manifest.get("online_status") not in (None, "success"):
                        problem = "legacy_arm_not_recovered"

                for eval_seed in EVAL_SEEDS:
                    result = arm_root / "evaluation" / FLOW.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
                    base = {
                        "track": track,
                        "arm": arm,
                        "architecture": manifest.get("architecture", "unknown"),
                        "training_seed": training_seed,
                        "flow": FLOW,
                        "eval_seed": eval_seed,
                        "checkpoint": str(checkpoint),
                        "checkpoint_sha256": checkpoint_hash,
                        "result_path": str(result),
                    }
                    if problem:
                        incomplete.append({**base, "reason": problem})
                        continue
                    values, failure = load_online_result(result, checkpoint, checkpoint_hash, eval_seed)
                    if failure:
                        incomplete.append({**base, "reason": failure})
                    else:
                        rows.append({**base, **(values or {})})

    grouped_primary: dict[str, Any] = {}
    grouped_all: dict[str, Any] = {}
    for track, arms in TRACKS.items():
        for arm in arms:
            key = f"{track}/{arm}"
            primary = [row for row in rows if row["track"] == track and row["arm"] == arm and row["eval_seed"] == PRIMARY_EVAL_SEED]
            descriptive = [row for row in rows if row["track"] == track and row["arm"] == arm]
            grouped_primary[key] = {metric: stats([row[metric] for row in primary]) for metric in METRICS}
            grouped_all[key] = {metric: stats([row[metric] for row in descriptive]) for metric in METRICS}
    comparisons = paired_comparisons(rows)

    output = {
        "schema": "jinan_direct_urgency_five_track_aggregate_v1",
        "protocol": {
            "training_flow": FLOW,
            "evaluation_flow": FLOW,
            "training_seeds": list(TRAINING_SEEDS),
            "primary_eval_seed": PRIMARY_EVAL_SEED,
            "descriptive_eval_seeds": list(EVAL_SEEDS),
            "run_counts": 3600,
        "online_phase_override": "disabled",
        "online_seed_role": "descriptive realization only; not independent training replicates",
        "interpretation_limits": {
            "tail_direct_labels": "Observed current-state phase P95 waiting proxy; not counterfactual candidate-action rollout P95.",
            "capacity_pt16_hd128": "Matches the baseline's three-flow Stage-1 source data but enlarges Stage-1 and Stage-2 together; it is an architecture intervention, not a memory-only test.",
        },
        },
        "checkpoint_tiebreak": checkpoint_reference(root),
        "completed_rows": rows,
        "incomplete_rows": incomplete,
        "primary_by_track_arm": grouped_primary,
        "descriptive_all_online_seeds_by_track_arm": grouped_all,
        "paired_primary_seed_comparisons": comparisons,
    }
    (root / "aggregate.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_fields = [
        "track", "arm", "architecture", "training_seed", "flow", "eval_seed", "checkpoint", "checkpoint_sha256",
        *METRICS, "evaluator_sha256", "result_path",
    ]
    incomplete_fields = [
        "track", "arm", "architecture", "training_seed", "flow", "eval_seed", "checkpoint", "checkpoint_sha256", "result_path", "reason",
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
        "# Jinan Direct-Urgency Five-Track Aggregate",
        "",
        "Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, deterministic online seed `2024`, no phase override.",
        "Online seeds `42` and `123` are descriptive realizations, not independent training replicates.",
        "",
        "| Track / arm | n | Reward | AQL | AWT | ATT | P95 wait |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, summary_by_metric in grouped_primary.items():
        lines.append(
            f"| {key} | {summary_by_metric['test_reward_over']['n']} | "
            f"{format_stat(summary_by_metric['test_reward_over'])} | "
            f"{format_stat(summary_by_metric['test_avg_queue_len_over'])} | "
            f"{format_stat(summary_by_metric['test_avg_waiting_time_over'])} | "
            f"{format_stat(summary_by_metric['test_avg_travel_time_over'])} | "
            f"{format_stat(summary_by_metric['test_vehicle_wait_p95_over'])} |"
        )
    lines.extend(
        (
            "",
            "Paired comparisons use the same training seeds and primary online seed `2024` only.",
            "Metric deltas are candidate minus baseline; positive reward and negative delay/queue/tail deltas are favorable.",
            "Tail labels are observed current-state phase P95 proxies, not counterfactual candidate-action rollout labels.",
            "The capacity arm matches the baseline's three-flow Stage-1 source data but enlarges Stage-1 and Stage-2 together; it is not a memory-only test.",
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
    (root / "COMPLETION_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines.extend(("", f"Completed online rows: {len(rows)}/{EXPECTED_COMPLETED_ROWS}.", f"Incomplete or invalid rows: {len(incomplete)}.", f"Completion audit: {audit['status']}."))
    (root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(audit))
    if audit["status"] != "pass":
        raise SystemExit("completion audit failed")


if __name__ == "__main__":
    main()
