#!/usr/bin/env python3
"""Strict same-flow aggregate for local wait25 refinement arms."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ARMS = ("wait225_mixed_rank", "wait25_conservative_rank", "wait25_strong_rank")
TRAINING_SEEDS = (2024, 42, 123)
EVAL_SEEDS = (2024, 42, 123)
FLOW = "anon_3_4_jinan_real.json"
PRIMARY_EVAL_SEED = 2024
METRICS = ("test_reward_over", "test_avg_queue_len_over", "test_avg_waiting_time_over", "test_avg_travel_time_over", "test_vehicle_wait_p95_over", "test_vehicle_wait_max_over", "test_vehicle_wait_observed_vehicle_count")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def env(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in output:
            raise ValueError(f"Invalid env manifest at {path}:{number}")
        output[key] = value
    return output


def summary(values: list[float]) -> dict[str, Any]:
    return {"n": len(values), "mean": statistics.fmean(values) if values else None, "std": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None), "values": values}


def load(path: Path, checkpoint: Path, checkpoint_hash: str, evaluator_hash: str, seed: int) -> tuple[dict[str, float] | None, str | None]:
    if not path.is_file():
        return None, "missing_result"
    try:
        payload = json.loads(path.read_text(encoding="utf-8")); args = payload["args"]; hashes = payload["artifact_hashes"]; deterministic = payload["determinism"]
        assert args["traffic_file"] == FLOW and int(args["seed"]) == seed and int(args["run_counts"]) == 3600
        assert Path(args["projection_ckpt"]).resolve() == checkpoint.resolve()
        assert hashes["projection_ckpt_sha256"] == checkpoint_hash and hashes["online_evaluator_sha256"] == evaluator_hash
        assert args["deterministic"] is True and deterministic["requested"] is True and deterministic["torch_deterministic_algorithms"] is True and deterministic["cudnn_deterministic"] is True and deterministic["cudnn_benchmark"] is False and deterministic["allow_tf32"] is False
        assert all(float(args.get(key, 0.0) or 0.0) == 0.0 for key in ("max_phase_override_gap", "max_phase_override_left_wait_protect", "max_phase_override_target_queue_floor", "max_phase_override_total_queue_floor"))
        values = {metric: float(payload["results"][metric]) for metric in METRICS}
        assert all(math.isfinite(value) for value in values.values())
    except (AssertionError, KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None, "invalid_or_protocol_mismatched_result"
    return values, None


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True); args = parser.parse_args()
    root = args.run_root.resolve(); provenance_path = root / "provenance.env"
    provenance = env(provenance_path)
    if provenance.get("training_flow") != FLOW or provenance.get("evaluation_flow") != FLOW:
        raise ValueError("Provenance violates the required same-flow protocol")
    evaluator_hash = provenance.get("file_projection_train_run_projection_online.py_sha256")
    if not evaluator_hash:
        raise ValueError("Missing evaluator hash in provenance")
    rows: list[dict[str, Any]] = []; incomplete: list[dict[str, Any]] = []
    for arm in ARMS:
        for train_seed in TRAINING_SEEDS:
            arm_root = root / "arms" / arm / f"seed_{train_seed}"; checkpoint = arm_root / "stage2" / "best.pt"; status_path = arm_root / "status.txt"
            reason = None; checkpoint_hash = sha256(checkpoint) if checkpoint.is_file() else ""
            if not status_path.is_file(): reason = "missing_arm_status"
            elif not checkpoint.is_file(): reason = "missing_checkpoint"
            else:
                status = env(status_path)
                if status.get("stage2_best_sha256") != checkpoint_hash: reason = "checkpoint_hash_mismatch"
                elif status.get("training_flow") != FLOW: reason = "training_flow_mismatch"
            for eval_seed in EVAL_SEEDS:
                path = arm_root / "evaluation" / FLOW.removesuffix(".json") / f"seed_{eval_seed}" / "result.json"
                base = {"arm": arm, "training_seed": train_seed, "flow": FLOW, "eval_seed": eval_seed, "result_path": str(path)}
                if reason: incomplete.append({**base, "reason": reason}); continue
                values, failure = load(path, checkpoint, checkpoint_hash, evaluator_hash, eval_seed)
                (incomplete.append({**base, "reason": failure}) if failure else rows.append({**base, **(values or {})}))
    primary: dict[str, Any] = {}
    descriptive: dict[str, Any] = {}
    for arm in ARMS:
        arm_primary = [row for row in rows if row["arm"] == arm and row["eval_seed"] == PRIMARY_EVAL_SEED]
        arm_all = [row for row in rows if row["arm"] == arm]
        primary[arm] = {metric: summary([row[metric] for row in arm_primary]) for metric in METRICS}
        descriptive[arm] = {metric: summary([row[metric] for row in arm_all]) for metric in METRICS}
    output = {"schema": "jinan_real_wait25_improvement_aggregate_v1", "protocol": {"training_flow": FLOW, "evaluation_flow": FLOW, "training_seeds": list(TRAINING_SEEDS), "primary_eval_seed": PRIMARY_EVAL_SEED, "descriptive_eval_seeds": list(EVAL_SEEDS), "run_counts": 3600, "online_phase_override": "disabled", "online_seed_role": "descriptive realization only; not independent"}, "completed_rows": rows, "incomplete_rows": incomplete, "primary_by_arm": primary, "descriptive_all_online_seeds_by_arm": descriptive}
    (root / "aggregate.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["arm", "training_seed", "flow", "eval_seed", *METRICS, "result_path"]
    for name, contents, names in (("online_results.csv", rows, fields), ("incomplete_runs.csv", incomplete, ["arm", "training_seed", "flow", "eval_seed", "result_path", "reason"])):
        with (root / name).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=names); writer.writeheader(); writer.writerows(contents)
    lines = ["# Jinan_real Wait25 Improvement Aggregate", "", "Primary protocol: `anon_3_4_jinan_real.json`, 3,600 seconds, deterministic online seed `2024`, no phase override.", "Online realization seeds `42` and `123` are descriptive only.", "", "| Arm | n | Reward | AQL | AWT | ATT | P95 wait |", "|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        stats = primary[arm]
        def fmt(metric: str) -> str:
            value = stats[metric]; return "NA" if value["mean"] is None else f"{value['mean']:.3f} +/- {value['std']:.3f}"
        lines.append(f"| {arm} | {stats['test_reward_over']['n']} | {fmt('test_reward_over')} | {fmt('test_avg_queue_len_over')} | {fmt('test_avg_waiting_time_over')} | {fmt('test_avg_travel_time_over')} | {fmt('test_vehicle_wait_p95_over')} |")
    lines += ["", f"Completed online rows: {len(rows)}/27."]
    (root / "AGGREGATE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"completed_online_rows": len(rows), "incomplete_online_rows": len(incomplete)}))


if __name__ == "__main__":
    main()
