import json

from dataset import TrafficClipJsonlDataset


def test_decision_only_keeps_interleaved_intersection_histories(tmp_path):
    records = [
        {"intersection_id": "A", "movement_counts": {"NT": 1}, "clip_caption": "A0"},
        {"intersection_id": "B", "movement_counts": {"NT": 10}, "clip_caption": "B0"},
        {
            "intersection_id": "A",
            "movement_counts": {"NT": 2},
            "clip_caption": "A1",
            "expert_action": "NTST",
        },
        {
            "intersection_id": "B",
            "movement_counts": {"NT": 20},
            "clip_caption": "B1",
            "expert_action": "ETWT",
        },
    ]
    path = tmp_path / "interleaved.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    dataset = TrafficClipJsonlDataset(str(path), seq_len=3, decision_only=True)

    assert len(dataset) == 2
    assert dataset[0]["intersection_id"] == "A"
    assert dataset[0]["sequence"][:, 0].tolist() == [1.0, 1.0, 2.0]
    assert dataset[1]["intersection_id"] == "B"
    assert dataset[1]["sequence"][:, 0].tolist() == [10.0, 10.0, 20.0]
