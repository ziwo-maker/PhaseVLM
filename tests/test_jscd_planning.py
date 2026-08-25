from __future__ import annotations

from scripts.analyze_jscd_self_consistency import analyze
from scripts.split_jscd_snapshot_pool import partition_snapshot_paths


def _record(expert: str, scores: dict[str, float]) -> dict:
    return {
        "expert_action": expert,
        "candidate_action_waits": scores,
    }


def test_partition_snapshot_paths_is_disjoint_and_deterministic() -> None:
    snapshots = [f"snapshot-{index}" for index in range(10)]
    holdout_a, train_a = partition_snapshot_paths(snapshots, holdout_count=3, seed=7)
    holdout_b, train_b = partition_snapshot_paths(snapshots, holdout_count=3, seed=7)
    assert (holdout_a, train_a) == (holdout_b, train_b)
    assert len(holdout_a) == 3
    assert set(holdout_a).isdisjoint(train_a)
    assert sorted(holdout_a + train_a) == snapshots


def test_self_consistency_gate_passes_for_material_action_and_rank_change() -> None:
    pi0 = {}
    pi1 = {}
    for index in range(10):
        key = (f"snapshot-{index // 2}", f"intersection-{index}")
        pi0[key] = _record("ETWT", {"ETWT": 0.0, "NTST": 1.0, "ELWL": 2.0, "NLSL": 3.0})
        if index < 3:
            pi1[key] = _record("NTST", {"ETWT": 2.0, "NTST": 0.0, "ELWL": 1.0, "NLSL": 3.0})
        else:
            pi1[key] = _record("ETWT", {"ETWT": 0.0, "NTST": 1.0, "ELWL": 2.0, "NLSL": 3.0})
    report = analyze(
        pi0,
        pi1,
        min_action_change_rate=0.2,
        max_mean_kendall_tau=0.8,
        min_informative_rank_rate=0.5,
    )
    assert report["top1_action_change_rate"] == 0.3
    assert report["gate_pass"]


def test_self_consistency_gate_rejects_identical_rankings() -> None:
    scores = {"ETWT": 0.0, "NTST": 1.0, "ELWL": 2.0, "NLSL": 3.0}
    pi0 = {("snapshot", "intersection"): _record("ETWT", scores)}
    pi1 = {("snapshot", "intersection"): _record("ETWT", scores)}
    report = analyze(
        pi0,
        pi1,
        min_action_change_rate=0.2,
        max_mean_kendall_tau=0.8,
        min_informative_rank_rate=0.5,
    )
    assert not report["gate_pass"]
    assert not report["gate_components"]["action_change_rate"]
