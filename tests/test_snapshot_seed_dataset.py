import json

from projection_train.snapshot_seed_dataset import SnapshotSeedDataset


def _write_record(handle, snapshot_path, intersection_id):
    handle.write(
        json.dumps(
            {
                "snapshot_path": str(snapshot_path),
                "intersection_id": intersection_id,
                "prompt": "choose a phase",
                "movement_counts": {"NT": 1, "NL": 2},
            }
        )
        + "\n"
    )


def test_snapshot_seed_dataset_skips_unparseable_snapshots(tmp_path):
    recorded_dir = tmp_path / "anon_4_4_hangzhou_real_5816.json_run"
    snapshot_dir = recorded_dir / "snapshots"
    snapshot_dir.mkdir(parents=True)
    valid_snapshot = snapshot_dir / "snapshot_valid.json"
    invalid_snapshot = snapshot_dir / "snapshot_invalid.json"
    jsonl_path = recorded_dir / "traffic_prompt_series_2s.jsonl"

    valid_snapshot.write_text(json.dumps({"currentTime": 123}), encoding="utf-8")
    invalid_snapshot.write_text('{"currentTime": "truncated}', encoding="utf-8")
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        _write_record(handle, valid_snapshot, "intersection_1_1")
        _write_record(handle, valid_snapshot, "intersection_1_2")
        _write_record(handle, invalid_snapshot, "intersection_1_1")

    dataset = SnapshotSeedDataset(str(jsonl_path), seq_len=4)

    assert len(dataset) == 1
    assert dataset.skipped_snapshot_count() == 1
    assert str(invalid_snapshot) in dataset.skipped_snapshot_errors
    assert dataset[0].intersection_ids == ["intersection_1_1", "intersection_1_2"]
