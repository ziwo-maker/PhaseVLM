import json

from projection_train.grouped_dataset import GroupedTrafficActionDataset


def _record(intersection_id, nt, expert_action=""):
    return {
        "intersection_id": intersection_id,
        "movement_counts": {"NT": nt},
        "prompt": "choose a phase",
        "expert_action": expert_action,
    }


def test_grouped_dataset_keeps_per_intersection_history_across_unlabeled_records(tmp_path):
    jsonl_path = tmp_path / "traffic_prompt_series_2s.jsonl"
    records = [
        _record("intersection_1_1", 1),
        _record("intersection_1_2", 10),
        _record("intersection_1_1", 2),
        _record("intersection_1_2", 20),
        _record("intersection_1_1", 3, "ETWT"),
        _record("intersection_1_2", 30, "NTST"),
    ]
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    dataset = GroupedTrafficActionDataset(str(jsonl_path), seq_len=3)

    assert len(dataset) == 2
    assert [sample.expert_action for sample in dataset.samples] == ["ETWT", "NTST"]
    assert [row[0] for row in dataset.samples[0].sequence] == [1.0, 2.0, 3.0]
    assert [row[0] for row in dataset.samples[1].sequence] == [10.0, 20.0, 30.0]
