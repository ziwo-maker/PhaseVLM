import pytest

from projection_train.prompt_utils import (
    MOVEMENT_KEYS,
    build_task_prompt,
    serialize_movement_history,
)


def test_history_serializer_preserves_every_cell_in_fixed_order():
    matrix = [[float(t * 8 + d) for d in range(8)] for t in range(16)]
    text = serialize_movement_history(matrix)
    lines = text.splitlines()
    assert lines[0].endswith(",".join(MOVEMENT_KEYS))
    assert len(lines) == 17
    assert lines[1] == "t-30s: 0,1,2,3,4,5,6,7"
    assert lines[-1] == "t: 120,121,122,123,124,125,126,127"
    assert text == serialize_movement_history(matrix)


def test_matched_prompt_only_adds_cues_in_structured_condition():
    cue = "Traffic q/w(s): lanes NT=2/5."
    minimal = build_task_prompt("ignored current state", mode="matched_minimal", traffic_context=cue)
    structured = build_task_prompt("ignored current state", mode="matched_structured", traffic_context=cue)
    assert cue not in minimal
    assert cue in structured
    for tag in ("ETWT", "NTST", "ELWL", "NLSL"):
        assert f"<signal>{tag}</signal>" in minimal
        assert f"<signal>{tag}</signal>" in structured


def test_serializer_rejects_missing_history():
    with pytest.raises(ValueError):
        serialize_movement_history([[0] * 8] * 15)
