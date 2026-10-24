from projection_train.action_space import ACTION_LIST
from projection_train.prompt_utils import build_task_prompt, serialize_compact_movement_history
from projection_train.train_projection_bc import format_prompts


def test_compact_history_retains_all_128_values_and_order():
    sequence = [[float(t * 8 + i) for i in range(8)] for t in range(16)]
    text = serialize_compact_movement_history(sequence)
    rows = [[float(value) for value in line.split()] for line in text.splitlines()[1:]]
    assert rows == sequence
    assert len(rows) == 16


def test_both_arms_have_identical_cues_rules_and_candidates():
    sequence = [[float(t + i) for i in range(8)] for t in range(16)]
    cues = "Traffic q/w(s): lanes NT=2/5; phases ETWT=3/7; maxQ=ETWT(3); maxW=ETWT(7); totalQ=3."
    common = build_task_prompt("unused", "information_matched", ACTION_LIST, cues)
    text = format_prompts(["unused"], "information_matched", [cues], [sequence], "history_text")[0]
    ts2i = format_prompts(["unused"], "information_matched", [cues], [sequence], "ts2i")[0]
    assert ts2i == common
    assert text == common + "\n" + serialize_compact_movement_history(sequence)
    assert "maxQ=" in common and "maxW=" in common and "totalQ=" in common
    assert all(action in common for action in ACTION_LIST)
