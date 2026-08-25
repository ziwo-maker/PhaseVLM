from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from projection_train.grouped_dataset import GroupedTrafficActionDataset


def _load_builder():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_dagger_stage2_dataset.py"
    spec = importlib.util.spec_from_file_location("dagger_builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dagger_builder_keeps_history_but_masks_easy_label(tmp_path):
    builder = _load_builder()
    source = tmp_path / "rollout.jsonl"
    output = tmp_path / "dagger.jsonl"
    rows = [
        {"intersection_id": "A", "movement_counts": {"NT": 1}},
        {
            "intersection_id": "A",
            "movement_counts": {"NT": 2},
            "prompt": "choose",
            "expert_action": "ETWT",
            "base_action": "ETWT",
            "phase_queue": {"ETWT": 1, "NTST": 0},
            "phase_wait": {"ETWT": 1},
            "candidate_action_metrics": {"ETWT": {"score": 0.0}, "NTST": {"score": 0.1}},
        },
        {
            "intersection_id": "A",
            "movement_counts": {"NT": 3},
            "prompt": "choose",
            "expert_action": "NTST",
            "base_action": "ETWT",
            "phase_queue": {"ETWT": 20, "NTST": 5},
            "phase_wait": {"ETWT": 90},
            "candidate_action_metrics": {"ETWT": {"score": 1.2}, "NTST": {"score": 0.0}},
        },
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    args = builder.argparse.Namespace(
        base_data=[], rollout_data=[str(source)], output=str(output), min_priority=1.0,
        queue_scale=10.0, wait_scale=60.0, regret_scale=1.0, missed_gap_scale=5.0,
        weight_scale=0.5, max_sample_weight=4.0, workers=1, worker_chunksize=1,
    )
    summary = builder.build_dataset(args)
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert summary["selected_dagger_rows"] == 1
    assert written[0]["movement_counts"]["NT"] == 1
    assert written[1]["expert_action"] == ""
    assert written[2]["expert_action"] == "NTST"
    assert written[2]["sample_weight"] > 1.0

    dataset = GroupedTrafficActionDataset(str(output), seq_len=3)
    assert len(dataset) == 1
    assert [row[0] for row in dataset.samples[0].sequence] == [1.0, 2.0, 3.0]


def test_dagger_builder_parallel_output_matches_sequential(tmp_path):
    builder = _load_builder()
    source = tmp_path / "rollout.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(
                {
                    "intersection_id": "A",
                    "prompt": "choose",
                    "expert_action": "NTST" if index % 2 else "ETWT",
                    "base_action": "ETWT",
                    "phase_queue": {"ETWT": index + 1, "NTST": index},
                    "phase_wait": {"ETWT": 30 + index},
                    "candidate_action_metrics": {"ETWT": {"score": float(index)}, "NTST": {"score": 0.0}},
                }
            )
            for index in range(8)
        )
        + "\n",
        encoding="utf-8",
    )
    common = {
        "base_data": [], "rollout_data": [str(source)], "min_priority": 0.0,
        "queue_scale": 10.0, "wait_scale": 60.0, "regret_scale": 1.0,
        "missed_gap_scale": 5.0, "weight_scale": 0.5, "max_sample_weight": 4.0,
        "worker_chunksize": 2,
    }
    sequential = tmp_path / "sequential.jsonl"
    parallel = tmp_path / "parallel.jsonl"
    sequential_summary = builder.build_dataset(
        builder.argparse.Namespace(**common, output=str(sequential), workers=1)
    )
    parallel_summary = builder.build_dataset(
        builder.argparse.Namespace(**common, output=str(parallel), workers=2)
    )
    assert parallel.read_bytes() == sequential.read_bytes()
    assert parallel_summary["action_counts"] == sequential_summary["action_counts"]
    assert parallel_summary["selected_dagger_rows"] == sequential_summary["selected_dagger_rows"]
