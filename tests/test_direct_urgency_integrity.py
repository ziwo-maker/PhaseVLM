from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_split import blocked_group_split


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_direct_urgency_fallback_waits_use_phase_wait() -> None:
    builder = _load_module("direct_urgency_builder", "scripts/build_direct_urgency_stage2_dataset.py")
    record = {
        "phase_queue": {"ETWT": 10, "NTST": 4, "ELWL": 2, "NLSL": 1},
        "phase_wait": {"ETWT": 30, "NTST": 9, "ELWL": 5, "NLSL": 2},
    }

    metrics, fallback_waits = builder.direct_metrics(record)

    assert metrics["ETWT"]["aql"] == -10.0
    assert metrics["ETWT"]["awt"] == -30.0
    assert fallback_waits == {"ETWT": -30.0, "NTST": -9.0, "ELWL": -5.0, "NLSL": -2.0}


def test_blocked_split_purges_boundary_history() -> None:
    samples = [SimpleNamespace(traffic_file="jinan_real") for _ in range(10)]

    train_indices, validation_indices = blocked_group_split(samples, val_ratio=0.2, purge_samples=2)

    assert train_indices == [0, 1, 2, 3, 4, 5]
    assert validation_indices == [8, 9]


def test_blocked_split_purges_each_intersection_trajectory() -> None:
    samples = [
        SimpleNamespace(
            source_path="data.jsonl",
            traffic_file="jinan_real",
            intersection_id=f"intersection_{intersection}",
        )
        for step in range(10)
        for intersection in (1, 2)
    ]

    train_indices, validation_indices = blocked_group_split(samples, val_ratio=0.2, purge_samples=3)

    # The input is time-interleaved.  Both trajectories lose their three
    # predecessors, rather than merely three global records before validation.
    assert train_indices == list(range(10))
    assert validation_indices == [16, 17, 18, 19]


def test_blocked_split_never_restores_purged_training_samples() -> None:
    samples = [SimpleNamespace(traffic_file="jinan_real", intersection_id="intersection_1") for _ in range(2)]

    with pytest.raises(ValueError, match="leaves no training samples"):
        blocked_group_split(samples, val_ratio=0.5, purge_samples=1)


def test_aggregate_rejects_wrong_checkpoint_hash(tmp_path) -> None:
    aggregate = _load_module("jinan_direct_urgency_aggregate", "scripts/aggregate_jinan_direct_urgency.py")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_hash = hashlib.sha256(b"checkpoint").hexdigest()
    result_path = tmp_path / "result.json"
    payload = {
        "args": {
            "traffic_file": "anon_3_4_jinan_real.json",
            "seed": 2024,
            "run_counts": 3600,
            "projection_ckpt": str(checkpoint),
            "deterministic": True,
            "max_phase_override_gap": 0.0,
            "max_phase_override_left_wait_protect": 0.0,
            "max_phase_override_target_queue_floor": 0.0,
            "max_phase_override_total_queue_floor": 0.0,
        },
        "artifact_hashes": {
            "projection_ckpt_sha256": checkpoint_hash,
            "online_evaluator_sha256": "evaluator",
        },
        "determinism": {
            "requested": True,
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "allow_tf32": False,
        },
        "results": {metric: 1.0 for metric in aggregate.METRICS},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    metrics, reason = aggregate.load_result(
        result_path,
        "anon_3_4_jinan_real.json",
        2024,
        expected_checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint_hash,
        expected_online_evaluator_sha256="evaluator",
        require_artifact_hashes=True,
        require_deterministic_online=True,
    )
    assert reason is None
    assert metrics is not None

    payload["artifact_hashes"]["projection_ckpt_sha256"] = "wrong"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    _, reason = aggregate.load_result(
        result_path,
        "anon_3_4_jinan_real.json",
        2024,
        expected_checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint_hash,
        expected_online_evaluator_sha256="evaluator",
        require_artifact_hashes=True,
        require_deterministic_online=True,
    )
    assert reason == "checkpoint_hash_mismatch"

    payload["artifact_hashes"]["projection_ckpt_sha256"] = checkpoint_hash
    payload["artifact_hashes"]["online_evaluator_sha256"] = "wrong-evaluator"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    _, reason = aggregate.load_result(
        result_path,
        "anon_3_4_jinan_real.json",
        2024,
        expected_checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint_hash,
        expected_online_evaluator_sha256="evaluator",
        require_artifact_hashes=True,
        require_deterministic_online=True,
    )
    assert reason == "evaluator_hash_mismatch"

    payload["artifact_hashes"]["online_evaluator_sha256"] = "evaluator"
    payload["results"]["test_reward_over"] = float("nan")
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    _, reason = aggregate.load_result(
        result_path,
        "anon_3_4_jinan_real.json",
        2024,
        expected_checkpoint=checkpoint,
        expected_checkpoint_sha256=checkpoint_hash,
        expected_online_evaluator_sha256="evaluator",
        require_artifact_hashes=True,
        require_deterministic_online=True,
    )
    assert reason == "nonfinite_closed_loop_metric"


def test_aggregate_cli_requires_hardened_artifacts_by_default(tmp_path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts/aggregate_jinan_direct_urgency.py"

    strict = subprocess.run(
        [sys.executable, str(script), "--run-root", str(tmp_path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert strict.returncode != 0
    assert "Missing required provenance manifest" in strict.stderr

    legacy = subprocess.run(
        [sys.executable, str(script), "--run-root", str(tmp_path), "--allow-legacy-artifacts"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert legacy.returncode == 0, legacy.stderr
    aggregate_payload = json.loads((tmp_path / "aggregate.json").read_text(encoding="utf-8"))
    assert aggregate_payload["protocol"]["artifact_validation_mode"] == "legacy"
