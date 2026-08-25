"""Tests for OOD-guided snapshot selection (Stage 3 main method)."""

from __future__ import annotations

import numpy as np
import pytest

from projection_train.ood_selection import (
    ACTION_LIST,
    FEATURE_DIM,
    MOVEMENT_KEYS,
    compute_source_stats,
    extract_intersection_features,
    group_snapshots,
    mahalanobis_sq,
    method_uses_seed,
    score_snapshots,
    select_snapshots,
    snapshot_mean_features,
    subset_records_by_times,
)


def _make_record(t: float, queue_val: float = 0.0, wait_val: float = 0.0) -> dict:
    return {
        "time_s": float(t),
        "movement_counts": {k: queue_val for k in MOVEMENT_KEYS},
        "phase_queue": {a: queue_val for a in ACTION_LIST},
        "phase_wait": {a: wait_val for a in ACTION_LIST},
    }


def test_extract_intersection_features_shape_and_values() -> None:
    rec = _make_record(0.0, queue_val=2.0, wait_val=3.0)
    vec = extract_intersection_features(rec)
    assert vec is not None
    assert vec.shape == (FEATURE_DIM,)
    assert FEATURE_DIM == len(MOVEMENT_KEYS) + 2 * len(ACTION_LIST)
    # First 8 = movement counts, next 4 = queue, last 4 = wait
    assert np.allclose(vec[:8], 2.0)
    assert np.allclose(vec[8:12], 2.0)
    assert np.allclose(vec[12:16], 3.0)


def test_extract_intersection_features_incomplete_returns_none() -> None:
    # Missing phase_queue/phase_wait must NOT be zero-imputed — strict None
    rec = {"time_s": 0.0, "movement_counts": {"NT": 5.0}}
    assert extract_intersection_features(rec) is None
    rec2 = {"time_s": 0.0, "movement_counts": {"NT": 5.0},
            "phase_queue": {}, "phase_wait": {}}
    assert extract_intersection_features(rec2) is None


def test_extract_intersection_features_malformed_returns_none() -> None:
    assert extract_intersection_features({"time_s": 0.0, "movement_counts": None}) is None


def test_group_snapshots() -> None:
    records = [_make_record(0.0), _make_record(0.0), _make_record(30.0)]
    groups = group_snapshots(records)
    assert set(groups.keys()) == {0.0, 30.0}
    assert len(groups[0.0]) == 2
    assert len(groups[30.0]) == 1


def test_mahalanobis_scales_by_std() -> None:
    stats = {"mean": np.zeros(4), "std": np.ones(4)}
    feats = np.array([[2.0, 0.0, 0.0, 0.0]])
    assert mahalanobis_sq(feats, stats)[0] == pytest.approx(4.0)
    stats2 = {"mean": np.zeros(4), "std": np.full(4, 2.0)}
    assert mahalanobis_sq(feats, stats2)[0] == pytest.approx(1.0)


def test_ood_score_ranks_far_snapshots_higher() -> None:
    source_records = [_make_record(t, queue_val=10.0, wait_val=5.0) for t in range(5)]
    stats = compute_source_stats(source_records, min_records=1)
    # In-distribution snapshot
    close = [_make_record(0.0, queue_val=10.0, wait_val=5.0),
             _make_record(0.0, queue_val=10.0, wait_val=5.0)]
    # Far snapshot
    far = [_make_record(30.0, queue_val=100.0, wait_val=90.0)]
    scores = score_snapshots(close + far, stats)
    assert scores[30.0] > scores[0.0]


def test_compute_source_stats_skips_incomplete_records() -> None:
    complete = [_make_record(t, queue_val=10.0, wait_val=5.0) for t in range(3)]
    incomplete = [{"time_s": 99.0, "movement_counts": {"NT": 1.0}}] * 100
    stats = compute_source_stats(complete + incomplete, min_records=1)
    assert stats["n_records"] == 3
    assert stats["coverage"] == pytest.approx(3 / 103)


def test_compute_source_stats_rejects_low_coverage() -> None:
    with pytest.raises(ValueError):
        compute_source_stats([_make_record(0.0)], min_records=10)


def test_select_ood_top_k_and_nesting() -> None:
    source_records = [_make_record(t, queue_val=10.0, wait_val=5.0) for t in range(5)]
    stats = compute_source_stats(source_records, min_records=1)
    records = []
    for i, t in enumerate(range(0, 300, 30)):  # 10 snapshots
        # Snapshots with increasing deviation from source
        records.append(_make_record(t, queue_val=10.0 + i * 20.0, wait_val=5.0 + i * 10.0))

    selected3, meta3 = select_snapshots(records, "ood", 3, source_stats=stats)
    selected5, meta5 = select_snapshots(records, "ood", 5, source_stats=stats)
    assert len(selected3) == 3 and len(selected5) == 5
    assert set(selected3) <= set(selected5)  # nested: top-3 subset of top-5
    assert meta3["method"] == "ood"
    # Highest OOD scores are the most deviant (last) snapshots
    assert selected3 == [210.0, 240.0, 270.0]
    assert selected5 == [150.0, 180.0, 210.0, 240.0, 270.0]


def test_select_random_is_seeded_and_exhaustive() -> None:
    records = [_make_record(t) for t in range(0, 300, 30)]
    sel1, _ = select_snapshots(records, "random", 4, seed=2024)
    sel2, _ = select_snapshots(records, "random", 4, seed=2024)
    sel3, _ = select_snapshots(records, "random", 4, seed=42)
    assert sel1 == sel2  # same seed → same selection
    assert sel1 != sel3  # different seed → (almost surely) different
    assert len(sel1) == 4
    assert len(set(sel1)) == 4


def test_select_rejects_k_larger_than_pool() -> None:
    records = [_make_record(t) for t in range(0, 60, 30)]
    with pytest.raises(ValueError):
        select_snapshots(records, "random", 5)


def test_select_ood_requires_source_stats() -> None:
    with pytest.raises(ValueError):
        select_snapshots([_make_record(0.0)], "ood", 1, source_stats=None)


def test_subset_records_by_times() -> None:
    records = [_make_record(0.0), _make_record(0.0), _make_record(30.0)]
    sub = subset_records_by_times(records, [30.0])
    assert len(sub) == 1 and sub[0]["time_s"] == 30.0


# ── Diversity selection methods (#2) ──────────────────────────────────────


def _make_varied_snapshots() -> list:
    """10 snapshots at 30s spacing with increasing deviation from source."""
    records = []
    for i, t in enumerate(range(0, 300, 30)):
        records.append(_make_record(t, queue_val=10.0 + i * 20.0,
                                    wait_val=5.0 + i * 10.0))
    return records


def _source_stats() -> dict:
    src = [_make_record(t, queue_val=10.0, wait_val=5.0) for t in range(5)]
    return compute_source_stats(src, min_records=1)


def test_select_ood_diverse_returns_k_distinct() -> None:
    records = _make_varied_snapshots()
    stats = _source_stats()
    selected, meta = select_snapshots(records, "ood_diverse", 4, source_stats=stats)
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert meta["method"] == "ood_diverse"
    # candidate pool is 3x K (capped by pool size)
    assert meta["candidate_pool"] >= 4


def test_select_ood_diverse_picks_high_ood_seed() -> None:
    records = _make_varied_snapshots()
    stats = _source_stats()
    # Highest-OOD snapshot (270.0) should be the farthest-first seed point
    selected, _ = select_snapshots(records, "ood_diverse", 2, source_stats=stats)
    assert 270.0 in selected


def test_select_ood_window_quota_spans_windows() -> None:
    # Snapshots across 3600s (4 x 900s windows)
    records = [(_make_record(t, queue_val=10.0 + (t / 30.0), wait_val=5.0))
               for t in range(30, 3601, 30)]
    stats = _source_stats()
    selected, meta = select_snapshots(records, "ood_window_quota", 8,
                                      source_stats=stats)
    assert len(selected) == 8
    assert meta["method"] == "ood_window_quota"
    windows = {int(t // 900.0) for t in selected}
    assert len(windows) >= 3  # spread across most temporal windows


def test_select_random_window_quota_is_seeded() -> None:
    records = [(_make_record(t, queue_val=10.0 + (t / 30.0), wait_val=5.0))
               for t in range(30, 3601, 30)]
    sel1, _ = select_snapshots(records, "random_window_quota", 8, seed=2024)
    sel2, _ = select_snapshots(records, "random_window_quota", 8, seed=2024)
    sel3, _ = select_snapshots(records, "random_window_quota", 8, seed=42)
    assert sel1 == sel2
    assert sel1 != sel3
    assert len(set(sel1)) == 8


def test_method_uses_seed() -> None:
    assert method_uses_seed("random") is True
    assert method_uses_seed("random_window_quota") is True
    assert method_uses_seed("ood") is False
    assert method_uses_seed("ood_diverse") is False
    assert method_uses_seed("ood_window_quota") is False


def test_snapshot_mean_features_standardizes() -> None:
    records = _make_varied_snapshots()
    stats = _source_stats()
    feats = snapshot_mean_features(records, stats)
    assert set(feats.keys()) == {float(t) for t in range(0, 300, 30)}
    for t, v in feats.items():
        assert v is not None and v.shape == (FEATURE_DIM,)
