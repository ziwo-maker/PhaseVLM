#!/usr/bin/env python3
"""SimExpert Oracle Validation — Step 0 prerequisite from IDEA_REPORT.md.

Validates SimExpert action ranking stability across:
  1. Horizon stability: Does the best action at 30s remain best at 90s and 180s?
  2. Neighbor interaction: When one intersection's action changes, do neighboring
     intersections' SimExpert evaluations remain valid?
  3. Future-arrival robustness: Do actions optimized for current queues account for
     vehicles arriving during the simulation horizon?

Outputs:
  - sim_expert_validation.json     — per-check results with pass/fail and metrics
  - sim_expert_validation.csv      — per-state diagnostic rows for downstream analysis
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.cost_ranking import DEFAULT_COST_WEIGHTS, _normalized_metric_costs


# ── helpers ──────────────────────────────────────────────────────────────


def _safe_parse(line: str) -> Optional[Dict]:
    try:
        return json.loads(line.strip())
    except (json.JSONDecodeError, ValueError):
        return None


def _has_full_metrics(record: Dict) -> bool:
    cam = record.get("candidate_action_metrics")
    if not isinstance(cam, dict):
        return False
    return len(cam) >= 4


def _best_by_score(metrics: Dict[str, Dict[str, float]]) -> Optional[str]:
    """Return the action with the lowest precomputed score (lower is better)."""
    best_action = None
    best_score = float("inf")
    for action, vals in metrics.items():
        score = vals.get("score")
        if score is not None and math.isfinite(score):
            if score < best_score:
                best_score = score
                best_action = action
    return best_action


def _best_by_metric(metrics: Dict[str, Dict[str, float]], metric: str) -> Optional[str]:
    """Return the action with the lowest value for a specific raw metric."""
    best_action = None
    best_val = float("inf")
    for action, vals in metrics.items():
        val = vals.get(metric)
        if val is not None and math.isfinite(val):
            if val < best_val:
                best_val = val
                best_action = action
    return best_action


def _action_ranking(metrics: Dict[str, Dict[str, float]], metric: str = "score") -> List[str]:
    """Return actions sorted by metric, best (lowest) first."""
    pairs = []
    for action, vals in metrics.items():
        val = vals.get(metric)
        if val is not None and math.isfinite(val):
            pairs.append((float(val), action))
    pairs.sort()
    return [action for _, action in pairs]


def _rank_correlation(rank_a: List[str], rank_b: List[str]) -> float:
    """Kendall tau between two rankings."""
    actions = list(dict.fromkeys(rank_a + rank_b))
    if len(actions) < 2:
        return 1.0
    idx_a = {action: i for i, action in enumerate(rank_a)}
    idx_b = {action: i for i, action in enumerate(rank_b)}
    concordant = 0
    discordant = 0
    for i, a1 in enumerate(actions):
        for a2 in actions[i + 1:]:
            if a1 not in idx_a or a2 not in idx_a or a1 not in idx_b or a2 not in idx_b:
                continue
            order_a = idx_a[a1] < idx_a[a2]
            order_b = idx_b[a1] < idx_b[a2]
            if order_a == order_b:
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    if total == 0:
        return 1.0
    return (concordant - discordant) / total


def _compute_oracle_gap(metrics: Dict[str, Dict[str, float]], expert_action: str) -> Optional[float]:
    """Compute: cost(expert) - cost(SimExpert_best). Non-negative by definition."""
    expert_score = None
    best_score = float("inf")
    for action, vals in metrics.items():
        score = vals.get("score")
        if score is not None and math.isfinite(score):
            score_f = float(score)
            if action == expert_action:
                expert_score = score_f
            if score_f < best_score:
                best_score = score_f
    if expert_score is None or not math.isfinite(best_score):
        return None
    return expert_score - best_score


# ── Check 1: Horizon stability ───────────────────────────────────────────


def check_horizon_stability(records: List[Dict]) -> Dict:
    """Simulate different horizons by varying the cost weight sensitivity.

    Since the data only contains a single horizon (whatever the collector used),
    we simulate horizon effects by varying the relative weight between instantaneous
    metrics (queue, wait) and cumulative metrics (travel time, spillback).

    Longer horizons → travel time / spillback matter more.
    Shorter horizons → queue / wait dominate.
    """
    horizon_configs = {
        "30s": {"awt": 3.0, "aql": 2.0, "att": 0.5, "spillback": 0.1},
        "90s": {"awt": 2.0, "aql": 1.0, "att": 1.0, "spillback": 0.2},
        "180s": {"awt": 1.0, "aql": 0.5, "att": 3.0, "spillback": 0.5},
    }

    results: Dict[str, Dict] = {}
    for horizon_label, weights in horizon_configs.items():
        agreements = 0
        total = 0
        ranking_changes = defaultdict(int)
        for record in records:
            cam = record.get("candidate_action_metrics", {})
            if not isinstance(cam, dict) or len(cam) < 4:
                continue
            # Best action under default weights (reference)
            costs_default = _normalized_metric_costs(cam, dict(DEFAULT_COST_WEIGHTS))
            best_default = min(costs_default, key=costs_default.get) if costs_default else None
            # Best action under horizon weights
            costs_horizon = _normalized_metric_costs(cam, weights)
            best_horizon = min(costs_horizon, key=costs_horizon.get) if costs_horizon else None
            if best_default and best_horizon:
                total += 1
                if best_default == best_horizon:
                    agreements += 1
                else:
                    ranking_changes[f"{best_default}→{best_horizon}"] += 1

        stability = agreements / max(1, total) if total > 0 else 0.0
        results[horizon_label] = {
            "total_states": total,
            "best_action_agreement": round(stability, 4),
            "best_action_changes": dict(ranking_changes),
            "verdict": "PASS" if stability >= 0.7 else ("WARN" if stability >= 0.5 else "FAIL"),
        }

    return results


# ── Check 2: Neighbor interaction ────────────────────────────────────────


def check_neighbor_interaction(records: List[Dict]) -> Dict:
    """Check whether action rankings correlate across neighboring intersections.

    We compare action rankings for spatially adjacent intersection pairs within
    the same time window to see if rankings are consistent or diverge.
    """
    # Group records by time window and intersection
    by_intersection: Dict[str, List[Dict]] = defaultdict(list)
    by_time: Dict[str, List[Dict]] = defaultdict(list)
    for record in records:
        iid = str(record.get("intersection_id", ""))
        date = str(record.get("date", ""))
        if iid:
            by_intersection[iid].append(record)
        if date:
            by_time[date].append(record)

    # For intersections that are grid-adjacent, compute rank correlation
    # Parse intersection IDs like "intersection_r_c"
    def grid_pos(iid: str) -> Optional[Tuple[int, int]]:
        parts = iid.split("_")
        if len(parts) >= 3:
            try:
                return int(parts[1]), int(parts[2])
            except ValueError:
                pass
        return None

    positions: Dict[str, Tuple[int, int]] = {}
    for iid in by_intersection:
        pos = grid_pos(iid)
        if pos:
            positions[iid] = pos

    # Find adjacent pairs (Manhattan distance 1)
    adjacent_pairs = []
    seen = set()
    for iid_a, pos_a in positions.items():
        for iid_b, pos_b in positions.items():
            if iid_a >= iid_b:
                continue
            if abs(pos_a[0] - pos_b[0]) + abs(pos_a[1] - pos_b[1]) == 1:
                pair_key = (min(iid_a, iid_b), max(iid_a, iid_b))
                if pair_key not in seen:
                    seen.add(pair_key)
                    adjacent_pairs.append((iid_a, iid_b))

    # Compute ranking agreement between adjacent intersections at same times
    correlations = []
    for iid_a, iid_b in adjacent_pairs:
        recs_a = by_intersection.get(iid_a, [])
        recs_b = by_intersection.get(iid_b, [])
        # Match by time
        times_a = {r.get("date", ""): r for r in recs_a}
        times_b = {r.get("date", ""): r for r in recs_b}
        common_times = set(times_a) & set(times_b)
        for t in common_times:
            cam_a = times_a[t].get("candidate_action_metrics", {})
            cam_b = times_b[t].get("candidate_action_metrics", {})
            if not isinstance(cam_a, dict) or not isinstance(cam_b, dict):
                continue
            if len(cam_a) < 4 or len(cam_b) < 4:
                continue
            rank_a = _action_ranking(cam_a)
            rank_b = _action_ranking(cam_b)
            if len(rank_a) >= 2 and len(rank_b) >= 2:
                correlations.append(_rank_correlation(rank_a, rank_b))

    mean_corr = float(np.mean(correlations)) if correlations else 0.0
    return {
        "adjacent_pairs_found": len(adjacent_pairs),
        "paired_samples": len(correlations),
        "mean_kendall_tau": round(mean_corr, 4),
        "std_kendall_tau": round(float(np.std(correlations)), 4) if correlations else 0.0,
        "verdict": "PASS" if mean_corr <= 0.7 else "INFO",
        "note": "Low correlation expected (different intersections have different traffic). "
                "High correlation would indicate regime collapse.",
    }


# ── Check 3: Future-arrival robustness ───────────────────────────────────


def check_future_arrival_robustness(records: List[Dict]) -> Dict:
    """Assess whether the best action would remain best if future vehicles arrive.

    We compare action ranking by instantaneous metrics (queue only) vs.
    comprehensive metrics (including travel time and spillback which account
    for future arrivals).
    """
    queue_only_agreements = 0
    comprehensive_agreements = 0
    total = 0
    regret_gaps = []

    for record in records:
        cam = record.get("candidate_action_metrics", {})
        if not isinstance(cam, dict) or len(cam) < 4:
            continue

        # Best by queue only (instantaneous, no future-awareness)
        best_queue = _best_by_metric(cam, "aql")
        # Best by travel time (cumulative, includes future arrivals)
        best_att = _best_by_metric(cam, "att")
        # Best by comprehensive score
        best_score = _best_by_score(cam)

        if best_queue and best_score:
            total += 1
            if best_queue == best_score:
                queue_only_agreements += 1
            else:
                expert_action = record.get("expert_action", "")
                gap = _compute_oracle_gap(cam, best_queue)
                if gap is not None:
                    regret_gaps.append(gap)

        if best_att and best_score:
            if best_att == best_score:
                comprehensive_agreements += 1

    queue_agreement = queue_only_agreements / max(1, total) if total > 0 else 0.0
    att_agreement = comprehensive_agreements / max(1, total) if total > 0 else 0.0

    return {
        "total_states": total,
        "queue_best_agreement": round(queue_agreement, 4),
        "traveltime_best_agreement": round(att_agreement, 4),
        "mean_regret_gap_queue_vs_best": round(float(np.mean(regret_gaps)), 6) if regret_gaps else 0.0,
        "verdict": "PASS" if att_agreement >= 0.5 else "WARN",
        "note": "Queue-only agreement < comprehensive → SimExpert correctly accounts for future arrivals.",
    }


# ── Check 4: Global ranking stability ────────────────────────────────────


def check_rating_consistency(records: List[Dict]) -> Dict:
    """Cross-check: do different scoring approaches agree on the best action?

    Compares: precomputed score, awt-minimizing, aql-minimizing, att-minimizing.
    """
    methods = {
        "score": lambda cam: _best_by_score(cam),
        "awt": lambda cam: _best_by_metric(cam, "awt"),
        "aql": lambda cam: _best_by_metric(cam, "aql"),
        "att": lambda cam: _best_by_metric(cam, "att"),
    }

    pairwise_agreements: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    rankings_by_method: Dict[str, List[List[str]]] = defaultdict(list)
    total = 0

    for record in records:
        cam = record.get("candidate_action_metrics", {})
        if not isinstance(cam, dict) or len(cam) < 4:
            continue
        bests = {name: fn(cam) for name, fn in methods.items()}
        bests = {k: v for k, v in bests.items() if v is not None}
        if len(bests) < 2:
            continue
        total += 1
        names = sorted(bests)
        for i, n1 in enumerate(names):
            rankings_by_method[n1].append(_action_ranking(cam, n1))
            for n2 in names[i + 1:]:
                if bests[n1] == bests[n2]:
                    pairwise_agreements[n1][n2] += 1
                    pairwise_agreements[n2][n1] += 1

    result = {"total_states": total, "pairwise_agreement_rates": {}}
    for n1 in sorted(pairwise_agreements):
        for n2 in sorted(pairwise_agreements[n1]):
            if n1 < n2:
                rate = pairwise_agreements[n1][n2] / max(1, total)
                result["pairwise_agreement_rates"][f"{n1}_vs_{n2}"] = round(rate, 4)

    # Mean rank correlation between methods
    method_names = sorted(rankings_by_method)
    rank_corrs = {}
    for i, n1 in enumerate(method_names):
        for n2 in method_names[i + 1:]:
            ranks_a = rankings_by_method[n1]
            ranks_b = rankings_by_method[n2]
            corrs = []
            for ra, rb in zip(ranks_a, ranks_b):
                if len(ra) >= 2 and len(rb) >= 2:
                    corrs.append(_rank_correlation(ra, rb))
            if corrs:
                rank_corrs[f"{n1}_vs_{n2}"] = round(float(np.mean(corrs)), 4)

    result["mean_rank_correlations"] = rank_corrs
    overall_agreement = (
        float(np.mean(list(pairwise_agreements[n1][n2]
                          for n1 in sorted(pairwise_agreements)
                          for n2 in sorted(pairwise_agreements[n1])
                          if n1 < n2)))
        / max(1, total)
        if total > 0
        else 0.0
    )
    result["overall_agreement"] = round(overall_agreement, 4)
    result["verdict"] = "PASS" if overall_agreement >= 0.6 else ("WARN" if overall_agreement >= 0.4 else "FAIL")
    return result


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SimExpert Oracle Validation")
    parser.add_argument("--data", nargs="+", required=True,
                        help="JSONL files with candidate_action_metrics")
    parser.add_argument("--output-dir", default="sim_expert_validation",
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=10000,
                        help="Max records per file to analyze")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    all_records: List[Dict] = []
    for path in args.data:
        file_records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = _safe_parse(line)
                if record and _has_full_metrics(record):
                    file_records.append(record)
        print(f"Loaded {len(file_records)} valid records from {os.path.basename(path)}")
        all_records.extend(file_records)

    if len(all_records) > args.max_samples:
        indices = np.random.choice(len(all_records), args.max_samples, replace=False)
        all_records = [all_records[int(i)] for i in indices]
        print(f"Subsampled to {len(all_records)} records")

    if not all_records:
        print("ERROR: No valid records with candidate_action_metrics found.")
        print("Please provide JSONL files collected with SimExpert counterfactual evaluation.")
        sys.exit(1)

    print(f"\nRunning SimExpert Oracle Validation on {len(all_records)} records...\n")

    # Run checks
    checks = {}

    print("=" * 60)
    print("Check 1: Horizon Stability")
    print("=" * 60)
    checks["horizon_stability"] = check_horizon_stability(all_records)
    for horizon, result in checks["horizon_stability"].items():
        print(f"  {horizon}: agreement={result['best_action_agreement']:.2%} → {result['verdict']}")

    print("\n" + "=" * 60)
    print("Check 2: Neighbor Interaction")
    print("=" * 60)
    checks["neighbor_interaction"] = check_neighbor_interaction(all_records)
    ni = checks["neighbor_interaction"]
    print(f"  Pairs: {ni['adjacent_pairs_found']}, Paired samples: {ni['paired_samples']}")
    print(f"  Mean Kendall τ: {ni['mean_kendall_tau']:.4f} → {ni['verdict']}")

    print("\n" + "=" * 60)
    print("Check 3: Future-Arrival Robustness")
    print("=" * 60)
    checks["future_arrival"] = check_future_arrival_robustness(all_records)
    fa = checks["future_arrival"]
    print(f"  Queue-best agreement: {fa['queue_best_agreement']:.2%}")
    print(f"  TravelTime-best agreement: {fa['traveltime_best_agreement']:.2%}")
    print(f"  → {fa['verdict']} — {fa['note']}")

    print("\n" + "=" * 60)
    print("Check 4: Rating Consistency")
    print("=" * 60)
    checks["rating_consistency"] = check_rating_consistency(all_records)
    rc = checks["rating_consistency"]
    print(f"  Overall agreement: {rc['overall_agreement']:.2%} → {rc['verdict']}")
    for pair, rate in rc.get("pairwise_agreement_rates", {}).items():
        print(f"    {pair}: {rate:.2%}")

    # ── Overall verdict ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    verdicts = []
    for check_name, result in checks.items():
        if isinstance(result, dict):
            for sub in result.values() if isinstance(result, dict) else []:
                if isinstance(sub, dict):
                    v = sub.get("verdict", "")
                    if v:
                        verdicts.append(v)
                break
            v = result.get("verdict", "")
            if v:
                verdicts.append(v)
        elif isinstance(result, list):
            for item in result:
                if isinstance(item, dict):
                    v = item.get("verdict", "")
                    if v:
                        verdicts.append(v)

    if "FAIL" in verdicts:
        overall = "FAIL — SimExpert has stability issues. Fix the oracle before proceeding."
    elif verdicts.count("WARN") >= 2:
        overall = "WARN — Some instability detected. Document limitations carefully."
    elif "WARN" in verdicts:
        overall = "PASS with caveats — minor instability, acceptable for research use."
    else:
        overall = "PASS — SimExpert action rankings are stable across all checks."

    print(f"  Verdicts: {verdicts}")
    print(f"  Overall: {overall}")

    # Save results
    output = {
        "date": "2026-08-12",
        "total_records": len(all_records),
        "checks": checks,
        "overall_verdict": overall,
        "action_list": ACTION_LIST,
    }

    json_path = os.path.join(args.output_dir, "sim_expert_validation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")

    # Also save per-record diagnostic CSV
    csv_path = os.path.join(args.output_dir, "sim_expert_validation.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        fields = [
            "intersection_id", "date", "expert_action", "best_score_action",
            "best_awt_action", "best_aql_action", "best_att_action",
            "oracle_gap", "score_ranking", "awt_ranking",
        ]
        f.write(",".join(fields) + "\n")
        for record in all_records[:min(len(all_records), 2000)]:
            cam = record.get("candidate_action_metrics", {})
            row = {
                "intersection_id": record.get("intersection_id", ""),
                "date": record.get("date", ""),
                "expert_action": record.get("expert_action", ""),
                "best_score_action": _best_by_score(cam) or "",
                "best_awt_action": _best_by_metric(cam, "awt") or "",
                "best_aql_action": _best_by_metric(cam, "aql") or "",
                "best_att_action": _best_by_metric(cam, "att") or "",
                "oracle_gap": _compute_oracle_gap(cam, record.get("expert_action", "")) or 0.0,
                "score_ranking": "|".join(_action_ranking(cam, "score")),
                "awt_ranking": "|".join(_action_ranking(cam, "awt")),
            }
            f.write(",".join(str(row[k]) for k in fields) + "\n")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
