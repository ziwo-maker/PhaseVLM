#!/usr/bin/env python3
"""Aggregate the same-flow frozen SimExpert reference campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any


ARM = "frozen_simexpert_mixed_rank"
TRAINING_SEEDS = (2024, 42, 123)
EVAL_SEEDS = (2024, 42, 123)
FLOW = "anon_3_4_jinan_real.json"
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
            raise ValueError(f"Invalid env manifest at {path}:{line_number}")
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
    checkpoint_sha256: str,
    evaluator_sha256: str,
    expected_seed: int,
) -> tuple[dict[str, float] | None, str | None]:
    if not path.is_file():
        return None, "missing_result"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        args = payload["args"]
        hashes = payload["artifact_hashes"]
        determinism = payload["determinism"]
        assert args["traffic_file"] == FLOW
        assert int(args["seed"]) == expected_seed
        assert int(args["run_counts"]) == 3600
        assert Path(args["projection_ckpt"]).resolve() == checkpoint.resolve()
        assert hashes["projection_ckpt_sha256"] == checkpoint_sha256
        assert hashes["online_evaluator_sha256"] == evaluator_sha256
        assert args["deterministic"] is True
        assert determinism["requested"] is True
        assert determinism["torch_deterministic_algorithms"] is True
        assert determinism["cudnn_deterministic"] is True
        assert determinism["cudnn_benchmark"] is False
        assert determinism["allow_tf32"] is False
        assert all(float(args.get(key, 0.0) or 0.0) == 0.0 for key in (
            "max_phase_override_gap",
            "max_phase_override_left_wait_protect",
            "max_phase_override_target_queue_floor",
            "max_phase_override_total_queue_floor",
        ))
        raw = payload["results"]
        assert isinstance(raw, dict)
        values = {metric: float(raw[metric]) for metric in METRICS}
        assert all(math.isfinite(value) for value in values.values())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_or_protocol_mismatched_result"
    return values, None


def collect_rows(root: Path, evaluator_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    for training_seed in TRAINING_SEEDS:
        arm_root = root / "arms" / ARM / f"seed_{training_seed}"
        checkpoint = arm_root / "stage2" / "best.pt"
        status_path = arm_root / "status.txt"
        checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else ""
        checkpoint_reason: str | None = None
        if not status_path.is_file():
            checkpoint_reason = "missing_arm_status"
        elif not checkpoint.is_file():
            checkpoint_reason = "missing_checkpoint"
        else:
            status = read_env_manifest(status_path)
            if status.get("stage2_best_sha256") != checkpoint_hash:
                checkpoint_reason = "checkpoint_hash_mismatch"
            elif status.get("training_flow") != FLOW:
                checkpoint_reason = "training_flow_mismatch"
        for eval_seed in EVAL_SEEDS:
            result_path = arm_root / "evaluation" / FLOW.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
            base = {
                "arm": ARM,
                "training_seed": training_seed,
                "flow": FLOW,
                "eval_seed": eval_seed,
                "result_path": str(result_path),
            }
            if checkpoint_reason:
                incomplete.append({**base, "reason": checkpoint_reason})
                continue
            values, reason = load_result(result_path, checkpoint, checkpoint_hash, evaluator_sha256, eval_seed)
            if reason:
                incomplete.append({**base, "reason": reason})
            else:
                rows.append({**base, **(values or {})})
    return rows, incomplete


def metric_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {metric: summary([row[metric] for row in rows]) for metric in METRICS}


def write_markdown(path: Path, aggregate: dict[str, Any]) -> None:
    primary = aggregate["primary_by_training_seed"]
    across = aggregate["descriptive_all_online_seeds"]
    lines = [
        "# Jinan_real Frozen SimExpert Reference Aggregate",
        "",
        "Protocol: frozen SimExpert rollout teacher, training and evaluation both on `anon_3_4_jinan_real.json`, 3,600 seconds, deterministic online evaluation, no phase override.",
        "Primary result: online seed `2024`; the three training seeds are the statistical units.",
        "The summaries across online seeds are descriptive realization checks, not independent statistical samples.",
        "",
        "| Primary metric | n | Mean | Std |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        value = primary[metric]
        mean = "NA" if value["mean"] is None else f"{value['mean']:.3f}"
        std = "NA" if value["std"] is None else f"{value['std']:.3f}"
        lines.append(f"| {metric} | {value['n']} | {mean} | {std} |")
    lines += [
        "",
        "| Descriptive all-online-seeds metric | n | Mean | Std |",
        "|---|---:|---:|---:|",
    ]
    for metric in METRICS:
        value = across[metric]
        mean = "NA" if value["mean"] is None else f"{value['mean']:.3f}"
        std = "NA" if value["std"] is None else f"{value['std']:.3f}"
        lines.append(f"| {metric} | {value['n']} | {mean} | {std} |")
    lines += ["", f"Completed online rows: {len(aggregate['completed_rows'])}/9."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    provenance_path = root / "provenance.env"
    if not provenance_path.is_file():
        raise FileNotFoundError(f"Missing provenance manifest: {provenance_path}")
    provenance = read_env_manifest(provenance_path)
    if provenance.get("training_flow") != FLOW or provenance.get("evaluation_flow") != FLOW:
        raise ValueError("Provenance does not declare the required same-flow protocol")
    evaluator_sha256 = provenance.get("file_projection_train_run_projection_online.py_sha256")
    if not evaluator_sha256:
        raise ValueError("Provenance is missing online evaluator hash")

    rows, incomplete = collect_rows(root, evaluator_sha256)
    primary_rows = [row for row in rows if row["eval_seed"] == PRIMARY_EVAL_SEED]
    aggregate = {
        "schema": "jinan_real_frozen_simexpert_reference_aggregate_v1",
        "protocol": {
            "arm": ARM,
            "frozen_teacher": "SimExpert rollout composite minmax cost: ATT=1, AQL=1, AWT=1, spillback=0",
            "training_seeds": list(TRAINING_SEEDS),
            "training_flow": FLOW,
            "evaluation_flow": FLOW,
            "primary_eval_seed": PRIMARY_EVAL_SEED,
            "descriptive_eval_seeds": list(EVAL_SEEDS),
            "run_counts": 3600,
            "online_phase_override": "disabled",
            "online_seed_role": "descriptive realization only; not an independent statistical unit",
            "provenance_manifest": str(provenance_path),
        },
        "completed_rows": rows,
        "incomplete_rows": incomplete,
        "primary_by_training_seed": metric_summaries(primary_rows),
        "descriptive_all_online_seeds": metric_summaries(rows),
    }
    (root / "aggregate.json").write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["arm", "training_seed", "flow", "eval_seed", *METRICS, "result_path"]
    with (root / "online_results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (root / "incomplete_runs.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "training_seed", "flow", "eval_seed", "result_path", "reason"])
        writer.writeheader()
        writer.writerows(incomplete)
    write_markdown(root / "AGGREGATE.md", aggregate)
    print(json.dumps({"completed_online_rows": len(rows), "incomplete_online_rows": len(incomplete)}))


if __name__ == "__main__":
    main()
