#!/usr/bin/env python3
"""OOD-guided snapshot selection for Stage 3 cross-network adaptation.

Selects which target-network decision snapshots to send to SimExpert for
counterfactual relabeling, using distance-to-source-distribution scoring
computed from collection-time features ONLY (no oracle information).

Feature space (per intersection record):
  8 movement counts (NT/NL/ST/SL/ET/EL/WT/WL)
  + 4 phase queue lengths (one per 4-phase action)
  + 4 phase waiting times (one per 4-phase action)
  = 16 dimensions, all available at collection time from the canonical
    evaluator hook (run_projection_online.py --collect_states_output).

Source distribution: per-intersection feature mean/std computed from
COMPLETE records in the source-network training JSONL (all 16 fields
present; coverage reported and enforced). A snapshot's OOD score is the mean
over its intersections of the squared Mahalanobis distance to the source
distribution (diagonal covariance, scale-aware variance floor).

Selection methods:
  ood                — top-K snapshots by OOD score
  random             — seeded uniform random sample of K snapshots
  ood_diverse        — OOD top-(3K) candidates, then farthest-first (core-set)
                       over standardized state features
  ood_window_quota   — per-15min-window quota, top OOD within each window
  random_window_quota— per-15min-window quota, random within each window

Reference: refine-logs/EXPERIMENT_RESULTS.md (Phase 2) — OOD was the best
acquisition method; FINAL_REPORT gates on the pt32/hd256 architecture.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from projection_train.action_space import ACTION_LIST

MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
FEATURE_DIM = len(MOVEMENT_KEYS) + 2 * len(ACTION_LIST)  # 8 + 8 = 16

EPS = 1e-6


# ── Feature extraction ────────────────────────────────────────────────────


def extract_intersection_features(record: Dict) -> Optional[np.ndarray]:
    """Extract the 16-dim collection-time feature vector from one record.

    STRICT: all three feature groups must be present (movement_counts,
    phase_queue, phase_wait). Returns None for incomplete records instead of
    imputing zeros — zero-imputation corrupts the source distribution with
    artificial vectors (e.g., non-decision records in the source JSONL).
    """
    movement_counts = record.get("movement_counts", {})
    phase_queue = record.get("phase_queue", {})
    phase_wait = record.get("phase_wait", {})
    if not isinstance(movement_counts, dict) or not movement_counts:
        return None
    if not isinstance(phase_queue, dict) or not phase_queue:
        return None
    if not isinstance(phase_wait, dict) or not phase_wait:
        return None

    feats = []
    for key in MOVEMENT_KEYS:
        try:
            feats.append(float(movement_counts.get(key, 0.0) or 0.0))
        except (TypeError, ValueError):
            feats.append(0.0)

    for action in ACTION_LIST:
        try:
            feats.append(float((phase_queue or {}).get(action, 0.0) or 0.0))
        except (TypeError, ValueError):
            feats.append(0.0)
    for action in ACTION_LIST:
        try:
            feats.append(float((phase_wait or {}).get(action, 0.0) or 0.0))
        except (TypeError, ValueError):
            feats.append(0.0)

    arr = np.array(feats, dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def load_records(paths: Sequence[str], max_records: int = 0) -> List[Dict]:
    """Load JSONL records, skipping malformed lines."""
    records: List[Dict] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    continue
                if max_records and len(records) >= max_records:
                    return records
    return records


# ── Source distribution ───────────────────────────────────────────────────


def compute_source_stats(
    source_records: Sequence[Dict],
    min_records: int = 500,
) -> Dict[str, np.ndarray]:
    """Compute source feature mean/std from complete per-intersection records.

    Only records with all 16 fields are used; coverage is reported and a
    minimum count is enforced so the distribution is not built from imputed
    garbage. Variance uses a scale-aware floor: std of a count/wait feature is
    floored at max(EPS, 1e-3 * max(1, |mean|)) — degenerate constant
    dimensions still contribute ~0 weight via the large denominator.
    """
    feats = []
    n_total = 0
    for rec in source_records:
        n_total += 1
        vec = extract_intersection_features(rec)
        if vec is not None:
            feats.append(vec)
    if not feats:
        raise ValueError("No usable source records for feature statistics")
    if len(feats) < min_records:
        raise ValueError(
            f"Only {len(feats)}/{n_total} source records have complete features "
            f"(need >= {min_records}) — refusing to build a corrupted source distribution."
        )
    stack = np.stack(feats, axis=0)
    mean = stack.mean(axis=0, dtype=np.float64)
    std = stack.std(axis=0, dtype=np.float64)
    # Scale-aware floor instead of a bare EPS
    floor = np.maximum(EPS, 1e-3 * np.maximum(1.0, np.abs(mean)))
    std = np.maximum(std, floor)
    return {
        "mean": mean,
        "std": std,
        "n_records": int(len(feats)),
        "n_total_seen": int(n_total),
        "coverage": float(len(feats)) / max(1, n_total),
    }


def mahalanobis_sq(features: np.ndarray, source_stats: Dict[str, np.ndarray]) -> np.ndarray:
    """Squared Mahalanobis distance to source distribution (diagonal covariance).

    Args:
        features: [N, FEATURE_DIM]
        source_stats: {"mean": [D], "std": [D]} from compute_source_stats
    Returns:
        [N] squared distances
    """
    mean = source_stats["mean"].astype(np.float64)
    std = source_stats["std"].astype(np.float64)
    diff = features.astype(np.float64) - mean[None, :]
    return np.sum((diff / std[None, :]) ** 2, axis=-1)


# ── Snapshot grouping & scoring ───────────────────────────────────────────


def group_snapshots(records: Sequence[Dict]) -> Dict[float, List[Dict]]:
    """Group records by decision time (time_s). Each snapshot = one decision
    step across all intersections."""
    groups: Dict[float, List[Dict]] = {}
    for rec in records:
        t = rec.get("time_s")
        if t is None:
            continue
        groups.setdefault(float(t), []).append(rec)
    return groups


def score_snapshots(
    records: Sequence[Dict],
    source_stats: Dict[str, np.ndarray],
) -> Dict[float, float]:
    """Score each snapshot by mean OOD distance over its intersections.

    Returns {time_s: ood_score}. Unscoreable snapshots get -inf so they rank
    LAST under descending-order selection.
    """
    groups = group_snapshots(records)
    scores: Dict[float, float] = {}
    for t, recs in groups.items():
        per_inter = []
        for rec in recs:
            vec = extract_intersection_features(rec)
            if vec is not None:
                per_inter.append(vec)
        if not per_inter:
            scores[t] = float("-inf")  # unscoreable snapshots rank last
            continue
        stack = np.stack(per_inter, axis=0)
        dists = mahalanobis_sq(stack, source_stats)
        scores[t] = float(np.mean(dists))
    return scores


# ── Diversity helpers ─────────────────────────────────────────────────────


def snapshot_mean_features(
    records: Sequence[Dict],
    source_stats: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[float, Optional[np.ndarray]]:
    """Per-snapshot mean feature vector (optionally standardized to source).

    Returns {time_s: mean_16dim_vector}. Snapshots with no scoreable
    intersections map to None.
    """
    groups = group_snapshots(records)
    out: Dict[float, Optional[np.ndarray]] = {}
    for t, recs in groups.items():
        vecs = [extract_intersection_features(r) for r in recs]
        vecs = [v for v in vecs if v is not None]
        if not vecs:
            out[t] = None
            continue
        mean = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float64)
        if source_stats is not None:
            std = source_stats["std"].astype(np.float64)
            mean = (mean - source_stats["mean"].astype(np.float64)) / std
        out[t] = mean
    return out


def _farthest_first_selection(
    candidate_times: Sequence[float],
    features: Dict[float, Optional[np.ndarray]],
    K: int,
) -> List[float]:
    """Greedy farthest-first (core-set style) over standardized features.

    Starts from the first (highest-priority) candidate, then iteratively adds
    the candidate maximizing its minimum L2 distance to the already-selected
    set. Deterministic given the candidate ordering and feature vectors.
    """
    cand = [t for t in candidate_times if features.get(t) is not None]
    if not cand:
        cand = list(candidate_times)
    if len(cand) <= K:
        return sorted(cand)
    X = {t: np.asarray(features[t], dtype=np.float64) for t in cand}
    selected = [cand[0]]
    selected_vecs = [X[cand[0]]]
    remaining = set(cand[1:])
    while len(selected) < K and remaining:
        best_t: Optional[float] = None
        best_d = -1.0
        for t in sorted(remaining):
            d = min(float(np.linalg.norm(X[t] - sv)) for sv in selected_vecs)
            if d > best_d:
                best_d = d
                best_t = t
        if best_t is None:
            break
        selected.append(best_t)
        selected_vecs.append(X[best_t])
        remaining.remove(best_t)
    return sorted(selected)


def _split_time_windows(
    times: Sequence[float],
    n_windows: int = 4,
    window_seconds: float = 900.0,
) -> Dict[int, List[float]]:
    """Split decision times into fixed-duration temporal windows."""
    windows: Dict[int, List[float]] = {}
    for t in sorted(times):
        w = min(n_windows - 1, max(0, int(float(t) // window_seconds)))
        windows.setdefault(w, []).append(t)
    return windows


def _window_quota_selection(
    times: Sequence[float],
    K: int,
    scores: Optional[Dict[float, float]],
    seed: int,
    n_windows: int = 4,
    window_seconds: float = 900.0,
) -> List[float]:
    """Select K snapshots with an even per-time-window quota.

    When `scores` is provided, the top-scoring snapshots within each window
    are chosen (OOD-per-window); otherwise a seeded random sample per window
    is drawn. Round-robin distributes the K quota over non-empty windows and
    backfills any shortfall (small windows) from the remaining pool.
    """
    times = sorted(set(float(t) for t in times))
    if K >= len(times):
        return times
    windows = _split_time_windows(times, n_windows, window_seconds)
    nonempty = sorted(windows.keys())
    quotas: Dict[int, int] = {w: 0 for w in nonempty}
    for i in range(K):
        quotas[nonempty[i % len(nonempty)]] += 1

    rng = np.random.RandomState(seed)
    selected: List[float] = []
    for w in nonempty:
        pool = windows[w]
        q = min(quotas[w], len(pool))
        if scores is not None:
            chosen = sorted(pool, key=lambda t: scores[t], reverse=True)[:q]
        else:
            chosen = sorted(rng.choice(pool, size=q, replace=False).tolist())
        selected.extend(chosen)

    selected_set = set(selected)
    if len(selected_set) < K:
        remaining = [t for t in times if t not in selected_set]
        need = K - len(selected_set)
        if scores is not None:
            backfill = sorted(remaining, key=lambda t: scores[t], reverse=True)[:need]
        else:
            backfill = sorted(
                rng.choice(remaining, size=min(need, len(remaining)),
                           replace=False).tolist()
            )
        selected.extend(backfill)
    return sorted(set(selected))


# Method names whose selection depends on the RNG seed (vs. deterministic).
SEEDED_METHODS = {"random", "random_window_quota"}


def method_uses_seed(method: str) -> bool:
    """True if a selection method's output varies with the RNG seed."""
    return str(method).lower().strip() in SEEDED_METHODS


# ── Selection ─────────────────────────────────────────────────────────────


def select_snapshots(
    records: Sequence[Dict],
    method: str,
    K: int,
    source_stats: Optional[Dict[str, np.ndarray]] = None,
    seed: int = 2024,
) -> Tuple[List[float], Dict[str, object]]:
    """Select K decision snapshots from records.

    Args:
        records: collected target-network state records (with time_s)
        method: "ood" or "random"
        K: number of snapshots to select
        source_stats: required for "ood"
        seed: RNG seed for random selection

    Returns:
        (selected_times, meta) where selected_times is sorted ascending and
        meta holds method/K/scores for audit.
    """
    groups = group_snapshots(records)
    all_times = sorted(groups.keys())
    if K > len(all_times):
        raise ValueError(
            f"K={K} exceeds available snapshots ({len(all_times)}). "
            f"Each 3600s episode at 30s intervals yields at most 120 snapshots."
        )

    method = str(method).lower().strip()
    if method == "ood":
        if source_stats is None:
            raise ValueError("ood selection requires source_stats")
        scores = score_snapshots(records, source_stats)
        ranked = sorted(all_times, key=lambda t: scores[t], reverse=True)
        selected = sorted(ranked[:K])
        meta: Dict[str, object] = {
            "method": "ood",
            "K": K,
            "selected_scores": {str(t): scores[t] for t in selected},
            "score_stats": {
                "min": float(min(scores.values())),
                "max": float(max(scores.values())),
                "mean": float(np.mean(list(scores.values()))),
            },
        }
    elif method == "random":
        rng = np.random.RandomState(seed)
        selected = sorted(rng.choice(all_times, size=K, replace=False).tolist())
        meta = {"method": "random", "K": K, "seed": seed}
    elif method == "ood_diverse":
        if source_stats is None:
            raise ValueError("ood_diverse selection requires source_stats")
        scores = score_snapshots(records, source_stats)
        ranked = sorted(all_times, key=lambda t: scores[t], reverse=True)
        candidate_n = min(len(all_times), max(K, int(round(K * 3))))
        candidates = ranked[:candidate_n]
        features = snapshot_mean_features(records, source_stats)
        selected = _farthest_first_selection(candidates, features, K)
        meta = {
            "method": "ood_diverse",
            "K": K,
            "candidate_pool": candidate_n,
            "selected_scores": {str(t): scores[t] for t in selected},
        }
    elif method == "ood_window_quota":
        if source_stats is None:
            raise ValueError("ood_window_quota selection requires source_stats")
        scores = score_snapshots(records, source_stats)
        selected = _window_quota_selection(all_times, K, scores, seed)
        meta = {
            "method": "ood_window_quota",
            "K": K,
            "selected_scores": {str(t): scores[t] for t in selected},
        }
    elif method == "random_window_quota":
        selected = _window_quota_selection(all_times, K, None, seed)
        meta = {"method": "random_window_quota", "K": K, "seed": seed}
    else:
        raise ValueError(
            f"Unknown selection method: {method} (choose ood/random/ood_diverse/"
            f"ood_window_quota/random_window_quota)"
        )

    return selected, meta


def subset_records_by_times(records: Sequence[Dict], times: Sequence[float]) -> List[Dict]:
    """Return the intersection records belonging to the given decision times."""
    time_set = set(float(t) for t in times)
    return [rec for rec in records if float(rec.get("time_s", -1.0)) in time_set]
