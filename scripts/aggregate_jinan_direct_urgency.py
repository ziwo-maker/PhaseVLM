#!/usr/bin/env python3
"""Aggregate primary and generalization results for the Jinan direct-urgency matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARMS = ("hard_bc", "equal_mixed_rank", "wait2_mixed_rank", "wait25_mixed_rank")
TRAINING_SEEDS = (2024, 42, 123)
EVAL_SEEDS = (2024, 42, 123)
FLOWS = (
    "anon_3_4_jinan_real.json",
    "anon_3_4_jinan_real_2000.json",
    "anon_3_4_jinan_real_2500.json",
)
PRIMARY_FLOW = "anon_3_4_jinan_real.json"
PRIMARY_EVAL_SEED = 2024
METRICS = (
    "test_reward_over",
    "test_avg_queue_len_over",
    "test_avg_waiting_time_over",
    "test_avg_travel_time_over",
    "test_vehicle_wait_p95_over",
    "test_vehicle_wait_max_over",
    "test_vehicle_wait_observed_vehicle_count",
)


def summary(values: list[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values) if values else None,
        "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
        "values": values,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_env_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"Invalid provenance manifest at {path}:{line_number}")
        values[key] = value
    return values


def load_result(
    path: Path,
    expected_flow: str,
    expected_seed: int,
    expected_checkpoint: Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    expected_online_evaluator_sha256: str | None = None,
    require_artifact_hashes: bool = False,
    require_deterministic_online: bool = False,
) -> tuple[dict[str, float] | None, str | None]:
    if not path.is_file():
        return None, "missing_result"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_result:{type(exc).__name__}"
    args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
    if args.get("traffic_file") != expected_flow or int(args.get("seed", -1)) != expected_seed:
        return None, "protocol_mismatch"
    if int(args.get("run_counts", 0)) != 3600:
        return None, "run_counts_mismatch"
    if any(float(args.get(key, 0.0) or 0.0) != 0.0 for key in (
        "max_phase_override_gap",
        "max_phase_override_left_wait_protect",
        "max_phase_override_target_queue_floor",
        "max_phase_override_total_queue_floor",
    )):
        return None, "online_override_enabled"
    if expected_checkpoint is not None:
        checkpoint_arg = str(args.get("projection_ckpt") or "")
        if not checkpoint_arg or Path(checkpoint_arg).resolve() != expected_checkpoint.resolve():
            return None, "checkpoint_mismatch"
    artifact_hashes = payload.get("artifact_hashes") if isinstance(payload.get("artifact_hashes"), dict) else {}
    if require_artifact_hashes:
        if not expected_checkpoint_sha256:
            return None, "missing_expected_checkpoint_hash"
        if artifact_hashes.get("projection_ckpt_sha256") != expected_checkpoint_sha256:
            return None, "checkpoint_hash_mismatch"
        if not expected_online_evaluator_sha256:
            return None, "missing_expected_evaluator_hash"
        if artifact_hashes.get("online_evaluator_sha256") != expected_online_evaluator_sha256:
            return None, "evaluator_hash_mismatch"
    if require_deterministic_online:
        determinism = payload.get("determinism") if isinstance(payload.get("determinism"), dict) else {}
        if not (
            args.get("deterministic") is True
            and determinism.get("requested") is True
            and determinism.get("torch_deterministic_algorithms") is True
            and determinism.get("cudnn_deterministic") is True
            and determinism.get("cudnn_benchmark") is False
            and determinism.get("allow_tf32") is False
        ):
            return None, "deterministic_online_not_verified"
    raw = payload.get("results") if isinstance(payload.get("results"), dict) else {}
    if not all(metric in raw for metric in METRICS):
        return None, "missing_closed_loop_metric"
    try:
        metrics = {metric: float(raw[metric]) for metric in METRICS}
    except (TypeError, ValueError):
        return None, "invalid_closed_loop_metric"
    if not all(math.isfinite(value) for value in metrics.values()):
        return None, "nonfinite_closed_loop_metric"
    return metrics, None


def arm_rows(
    root: Path,
    require_artifact_hashes: bool,
    require_deterministic_online: bool,
    expected_online_evaluator_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for arm in ARMS:
        for training_seed in TRAINING_SEEDS:
            arm_root = root / "arms" / arm / f"seed_{training_seed}"
            checkpoint = arm_root / "stage2" / "best.pt"
            checkpoint_sha256 = sha256_file(checkpoint) if checkpoint.is_file() else None
            checkpoint_reason: str | None = None
            expected_checkpoint_sha256 = checkpoint_sha256
            if require_artifact_hashes:
                status_path = arm_root / "status.txt"
                if not status_path.is_file():
                    checkpoint_reason = "missing_arm_status"
                else:
                    status = read_env_manifest(status_path)
                    status_checkpoint_sha256 = status.get("stage2_best_sha256")
                    if not status_checkpoint_sha256:
                        checkpoint_reason = "missing_stage2_checkpoint_hash"
                    elif checkpoint_sha256 != status_checkpoint_sha256:
                        checkpoint_reason = "checkpoint_file_hash_mismatch"
                    else:
                        expected_checkpoint_sha256 = status_checkpoint_sha256
            for flow in FLOWS:
                for eval_seed in EVAL_SEEDS:
                    path = arm_root / "evaluation" / flow.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
                    if checkpoint_reason:
                        metrics, reason = None, checkpoint_reason
                    else:
                        metrics, reason = load_result(
                            path,
                            flow,
                            eval_seed,
                            expected_checkpoint=checkpoint,
                            expected_checkpoint_sha256=expected_checkpoint_sha256,
                            expected_online_evaluator_sha256=expected_online_evaluator_sha256,
                            require_artifact_hashes=require_artifact_hashes,
                            require_deterministic_online=require_deterministic_online,
                        )
                    base = {
                        "arm": arm,
                        "training_seed": training_seed,
                        "flow": flow,
                        "eval_seed": eval_seed,
                        "result_path": str(path),
                    }
                    if reason:
                        incomplete.append({**base, "reason": reason})
                    else:
                        complete.append({**base, **(metrics or {})})
    return complete, incomplete


def baseline_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    complete: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for flow in FLOWS:
        for eval_seed in EVAL_SEEDS:
            path = root / "baseline" / "evaluation" / flow.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
            metrics, reason = load_result(path, flow, eval_seed)
            base = {"flow": flow, "eval_seed": eval_seed, "result_path": str(path)}
            if reason:
                incomplete.append({**base, "reason": reason})
            else:
                complete.append({**base, **(metrics or {})})
    return complete, incomplete


def by_arm_primary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        selected = [
            row for row in rows
            if row["arm"] == arm and row["flow"] == PRIMARY_FLOW and row["eval_seed"] == PRIMARY_EVAL_SEED
        ]
        result[arm] = {metric: summary([row[metric] for row in selected]) for metric in METRICS}
        result[arm]["n"] = len(selected)
    return result


def by_arm_flow(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in ARMS:
        result[arm] = {}
        for flow in FLOWS:
            selected = [row for row in rows if row["arm"] == arm and row["flow"] == flow]
            result[arm][flow] = {metric: summary([row[metric] for row in selected]) for metric in METRICS}
            result[arm][flow]["n"] = len(selected)
            result[arm][flow]["note"] = (
                "n combines three training seeds and three online realization seeds; "
                "training seed is the primary statistical unit and online seeds are descriptive only."
            )
    return result


def paired_primary(rows: list[dict[str, Any]], candidate: str, reference: str) -> dict[str, Any]:
    index = {(row["arm"], row["training_seed"]): row for row in rows if row["flow"] == PRIMARY_FLOW and row["eval_seed"] == PRIMARY_EVAL_SEED}
    shared = [seed for seed in TRAINING_SEEDS if (candidate, seed) in index and (reference, seed) in index]
    return {
        "candidate": candidate,
        "reference": reference,
        "shared_training_seeds": shared,
        "candidate_minus_reference": {
            metric: summary([index[(candidate, seed)][metric] - index[(reference, seed)][metric] for seed in shared])
            for metric in METRICS
        },
    }


def write_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    lines = [
        "# Jinan_real Direct-Urgency Stage 2 Aggregate",
        "",
        "Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, online seed `2024`, no phase override.",
        "Generalization protocol: all three Jinan flows (`real`, `real_2000`, `real_2500`) x online seeds `2024`, `42`, `123`.",
        "Online realization seeds are not independent statistical samples; all primary comparisons use the three training seeds.",
        "The `hard_bc` arm is the matched online baseline; the Stage 1 alignment checkpoint is not an action policy and is not evaluated online.",
        "",
        "| Arm | n | Reward | Avg queue | Avg wait | Avg travel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        stats = aggregate["primary_by_arm"][arm]
        def fmt(metric: str) -> str:
            value = stats[metric]
            return "NA" if value["mean"] is None else f"{value['mean']:.3f} +/- {value['std']:.3f}"
        lines.append(
            f"| {arm} | {stats['n']} | {fmt('test_reward_over')} | {fmt('test_avg_queue_len_over')} | {fmt('test_avg_waiting_time_over')} | {fmt('test_avg_travel_time_over')} |"
        )
    lines += [
        "",
        f"Completed online rows: {len(aggregate['completed_rows'])}/108.",
        "The JSON/CSV also include p95/max observed vehicle wait diagnostics, raw rows, and incomplete-run reasons.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument(
        "--allow-legacy-artifacts",
        action="store_true",
        help=(
            "Allow an aggregate without the hardened provenance, artifact-hash, "
            "and deterministic-runtime evidence. Use only for historical campaigns."
        ),
    )
    # Keep the former explicit switches accepted for existing wrappers.  New
    # invocations are strict by default; legacy compatibility is opt-in above.
    parser.add_argument("--require-provenance", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-artifact-hashes", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-deterministic-online", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.allow_legacy_artifacts and any((
        args.require_provenance,
        args.require_artifact_hashes,
        args.require_deterministic_online,
    )):
        parser.error("--allow-legacy-artifacts cannot be combined with hardened validation flags")
    require_provenance = not args.allow_legacy_artifacts
    require_artifact_hashes = not args.allow_legacy_artifacts
    require_deterministic_online = not args.allow_legacy_artifacts
    root = args.run_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    provenance_path = root / "provenance.env"
    if require_provenance and not provenance_path.is_file():
        raise FileNotFoundError(f"Missing required provenance manifest: {provenance_path}")
    provenance = read_env_manifest(provenance_path) if provenance_path.is_file() else {}
    expected_online_evaluator_sha256 = provenance.get(
        "file_projection_train_run_projection_online.py_sha256"
    )
    if require_artifact_hashes and not expected_online_evaluator_sha256:
        raise ValueError(
            "Provenance manifest is missing file_projection_train_run_projection_online.py_sha256"
        )
    rows, incomplete = arm_rows(
        root,
        require_artifact_hashes=require_artifact_hashes,
        require_deterministic_online=require_deterministic_online,
        expected_online_evaluator_sha256=expected_online_evaluator_sha256,
    )
    aggregate = {
        "schema": "jinan_real_direct_urgency_aggregate_v2",
        "protocol": {
            "training_data": "jinan_real direct observed phase queue/wait",
            "architecture": "pt8_hd64",
            "training_seeds": list(TRAINING_SEEDS),
            "primary_flow": PRIMARY_FLOW,
            "primary_eval_seed": PRIMARY_EVAL_SEED,
            "generalization_flows": list(FLOWS),
            "generalization_eval_seeds": list(EVAL_SEEDS),
            "run_counts": 3600,
            "online_phase_override": "disabled",
            "online_seed_role": "descriptive realization only; not an independent statistical unit",
            "artifact_validation_mode": "legacy" if args.allow_legacy_artifacts else "hardened",
            "deterministic_online_required": require_deterministic_online,
            "checkpoint_hash_verification_required": require_artifact_hashes,
            "provenance_manifest": str(provenance_path) if provenance_path.is_file() else None,
            "matched_online_baseline_arm": "hard_bc",
        },
        "completed_rows": rows,
        "incomplete_rows": incomplete,
        "primary_by_arm": by_arm_primary(rows),
        "generalization_by_arm_and_flow": by_arm_flow(rows),
        "paired_primary_comparisons": [
            paired_primary(rows, "wait2_mixed_rank", "equal_mixed_rank"),
            paired_primary(rows, "wait2_mixed_rank", "hard_bc"),
            paired_primary(rows, "wait25_mixed_rank", "wait2_mixed_rank"),
        ],
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = ["arm", "training_seed", "flow", "eval_seed", *METRICS, "result_path"]
    with (root / "online_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "incomplete_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "training_seed", "flow", "eval_seed", "result_path", "reason"])
        writer.writeheader()
        writer.writerows(incomplete)
    write_markdown(root / "AGGREGATE.md", aggregate)
    print(json.dumps({"completed_online_rows": len(rows), "incomplete_online_rows": len(incomplete)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
