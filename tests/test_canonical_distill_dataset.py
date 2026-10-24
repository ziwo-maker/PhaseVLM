import json

from scripts.build_queue_wait_distill_dataset import build_from_canonical_decision_states
from projection_train.grouped_dataset import GroupedTrafficActionDataset


def test_canonical_builder_preserves_online_sequence_and_final_label(tmp_path) -> None:
    state_action_path = tmp_path / "state_action.json"
    decision_states_path = tmp_path / "decision_states.jsonl"
    output_path = tmp_path / "dataset.jsonl"
    sequence = [[float(step + column) for column in range(8)] for step in range(16)]

    item = {
        "action": "NTST",
        "policy_action": "ETWT",
        "raw_prompt": "raw",
        "policy_prompt": "policy",
        "prompt_context": "context",
        "phase_queue": {"ETWT": 1, "NTST": 3, "ELWL": 0, "NLSL": 0},
        "phase_wait": {"ETWT": 1, "NTST": 2, "ELWL": 0, "NLSL": 0},
        "max_phase_override": {
            "enabled": True,
            "overridden": True,
            "original_action": "ETWT",
            "final_action": "NTST",
            "phase_queue": {"ETWT": 1, "NTST": 3, "ELWL": 0, "NLSL": 0},
            "phase_wait": {"ETWT": 1, "NTST": 2, "ELWL": 0, "NLSL": 0},
        },
    }
    state_action_path.write_text(json.dumps([[item]], ensure_ascii=False), encoding="utf-8")
    decision_states_path.write_text(
        json.dumps(
            {
                "intersection_id": "intersection_1_1",
                "time_s": 0,
                "movement_counts": {key: 0 for key in ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")},
                "prompt": "prompt",
                "prompt_context": "context",
                "sequences": sequence,
                "snapshot_path": "snapshot.json",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = build_from_canonical_decision_states(
        state_action_path=state_action_path,
        decision_states_path=decision_states_path,
        output_path=output_path,
        label_source="final",
        gap=2.0,
        left_wait_protect=60.0,
    )

    dataset = GroupedTrafficActionDataset(str(output_path), seq_len=16)
    assert summary["terminal_rows"] == 1
    assert summary["sequence_source"] == "online_canonical"
    assert dataset.samples[0].has_explicit_sequence is True
    assert dataset.samples[0].sequence == sequence
    assert dataset.samples[0].expert_action == "NTST"
    assert dataset.samples[0].base_action == "ETWT"
