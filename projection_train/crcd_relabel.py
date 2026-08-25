"""CRCD (Reliable Cross-Horizon Counterfactual Distillation) relabeler.

Pure python — no cityflow, no torch. Phase 7 (refine-logs/EXPERIMENT_FINAL_REPORT.md).

Takes the three frozen-neighbor SimExpert relabel files over the SAME snapshot
pool (30s / 120s / 300s) and produces a teacher-variant training dataset. The
core contribution is a *reliability gate* that decides, per (snapshot,
intersection) state, whether the long-horizon counterfactual label should be
distilled into the policy (reliable) or whether the policy should revert to the
source/base action (unreliable).

Teacher variants (the Phase 7 evaluation table, minus policy-responsive which
needs a new cityflow rollout):

  c30         30s frozen labels              (existing teacher baseline)
  c300        300s frozen labels             (longer horizon, no gate)
  crcd_nogate multi-horizon majority vote    (no reliability filter, no KL)
  crcd        reliability gate + KL anchor   (full proposed method)

Reliability gate (all must hold for a state to be "reliable"):

  1. best_300 is a through action (ETWT / NTST)  — left-turn labels are ~95%
     short-horizon artifacts (teacher-validity finding).
  2. best_120 == best_300                          — action stable across the two
     longer horizons.
  3. adv_300(best_300) > adv_margin                — long-horizon advantage over
     the base action exceeds a predeclared practical margin.
  4. adv_120(best_300) > 0                          — advantage sign confirmed at
     >= 120s.
  5. best_300 is Pareto-feasible and satisfies the wait/spillback constraints at
     300s.

advantage(a) = score(base) - score(a) on the per-snapshot min-max-normalized
composite score (positive => a beats the base action).

Reliable state     -> expert_action = best_300, soft targets from 300s costs.
Unreliable state   -> expert_action = base_action, empty candidate costs (so the
                      soft/ranking loss is disabled) and the KL-to-base anchor
                      applies via train_projection_bc.py --base_policy_kl_mode
                      expert_match.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

ACTION_LIST: Tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")
THROUGH_ACTIONS: Tuple[str, ...] = ("ETWT", "NTST")

TEACHER_VARIANTS: Tuple[str, ...] = ("c30", "c300", "crcd_nogate", "crcd", "prcat")


def _safe_float(v: Any, default: float = 0.0) -> Optional[float]:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return None


def load_relabeled(path: str) -> Dict[Tuple[str, str], Dict]:
    """Load a relabeled JSONL keyed by (snapshot_path, intersection_id)."""
    out: Dict[Tuple[str, str], Dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sp = str(rec.get("snapshot_path", ""))
            iid = str(rec.get("intersection_id", ""))
            if not sp or not iid:
                continue
            out[(sp, iid)] = rec
    return out


def load_aligned(
    f30: str, f120: str, f300: str
) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict]], List[str]]:
    """Align the three horizon files by (snapshot, intersection).

    Returns (aligned, snapshot_order). aligned[key] = {"30": rec, "120": rec, "300": rec}.
    snapshot_order is the sorted list of unique snapshot paths present in all
    three files (intersection-complete).
    """
    by_horizon = {
        "30": load_relabeled(f30),
        "120": load_relabeled(f120),
        "300": load_relabeled(f300),
    }
    keys = set(by_horizon["30"]) & set(by_horizon["120"]) & set(by_horizon["300"])
    aligned: Dict[Tuple[str, str], Dict[str, Dict]] = {}
    for key in sorted(keys):
        aligned[key] = {h: by_horizon[h][key] for h in ("30", "120", "300")}

    snapshots: Dict[str, int] = defaultdict(int)
    for (sp, _), _ in aligned.items():
        snapshots[sp] += 1
    snapshot_order = sorted(snapshots.keys())
    return aligned, snapshot_order


def best_action(rec: Dict) -> Optional[str]:
    """Argmin over candidate_action_waits (composite score, lower better)."""
    waits = rec.get("candidate_action_waits", {})
    if not isinstance(waits, dict) or not waits:
        return None
    best: Optional[str] = None
    best_score = float("inf")
    for a in ACTION_LIST:
        s = _safe_float(waits.get(a))
        if s is not None and s < best_score:
            best_score = s
            best = a
    return best


def advantage(rec: Dict, action: str) -> Optional[float]:
    """score(base) - score(action); positive => action beats base."""
    waits = rec.get("candidate_action_waits", {})
    if not isinstance(waits, dict):
        return None
    base = rec.get("base_action")
    base_s = _safe_float(waits.get(base))
    act_s = _safe_float(waits.get(action))
    if base_s is None or act_s is None:
        return None
    return base_s - act_s


def majority_vote(actions: List[str]) -> Optional[str]:
    """Majority vote over the three horizon best actions; tie -> longest horizon
    wins (order 30,120,300 => later entries override earlier on a tie)."""
    valid = [a for a in actions if a]
    if not valid:
        return None
    counts = Counter(valid)
    best, best_n = None, -1
    for a in valid:  # preserve order for deterministic tie-break
        if counts[a] > best_n:
            best, best_n = a, counts[a]
        elif counts[a] == best_n and a != best:
            # tie: prefer the action that appears later in `actions`
            # (longer horizon comes last).
            if actions.index(a) > actions.index(best):
                best = a
    return best


def is_reliable(
    rec30: Dict, rec120: Dict, rec300: Dict, params: Dict[str, Any]
) -> Tuple[bool, str]:
    """Apply the CRCD reliability gate. Returns (reliable, reason)."""
    b300 = best_action(rec300)
    b120 = best_action(rec120)
    if b300 is None or b120 is None:
        return False, "no_best"

    if params.get("require_through", True) and b300 not in THROUGH_ACTIONS:
        return False, "non_through_300"

    if b120 != b300:
        return False, "unstable_120_300"

    adv_margin = float(params.get("adv_margin", 0.05))
    a300 = advantage(rec300, b300)
    a120 = advantage(rec120, b300)
    if a300 is None or a120 is None:
        return False, "no_advantage"

    if a300 <= adv_margin:
        return False, "margin_too_small"

    if a120 <= 0.0:
        return False, "sign_flips_at_120"

    if params.get("require_feasible", True):
        pareto = rec300.get("pareto_feasible", {})
        viol = rec300.get("constraint_violations", {})
        if isinstance(pareto, dict) and pareto.get(b300) is False:
            return False, "infeasible_300"
        if isinstance(viol, dict) and viol.get(b300):
            return False, "constraint_violation_300"

    return True, "ok"


def _base_record(rec30: Dict) -> Dict:
    """Start from the 30s record (full collection fields) and strip label keys
    we will overwrite per teacher, so no stale 30s label leaks into a variant."""
    rec = dict(rec30)
    for key in (
        "candidate_action_metrics",
        "candidate_action_waits",
        "expert_action",
        "expert_action_margin",
        "pareto_feasible",
        "constraint_violations",
        "sample_weight",
        "sim_horizon",
    ):
        rec.pop(key, None)
    return rec


def _attach_label(
    rec: Dict,
    expert: str,
    horizon_rec: Optional[Dict],
    sample_weight: float,
    reliable: bool,
    teacher: str,
    advantage_value: Optional[float],
) -> Dict:
    rec["expert_action"] = expert
    rec["base_action"] = str(horizon_rec.get("base_action", rec.get("base_action", expert))) if horizon_rec else rec.get("base_action", expert)
    if horizon_rec is not None:
        rec["candidate_action_metrics"] = horizon_rec.get("candidate_action_metrics", {})
        rec["candidate_action_waits"] = horizon_rec.get("candidate_action_waits", {})
        rec["pareto_feasible"] = horizon_rec.get("pareto_feasible", {})
        rec["constraint_violations"] = horizon_rec.get("constraint_violations", {})
        rec["sim_horizon"] = horizon_rec.get("sim_horizon")
        rec["expert_action_margin"] = horizon_rec.get("expert_action_margin")
    else:
        # Unreliable -> no candidate costs, so soft/ranking loss is disabled and
        # only the hard CE to base + KL-to-base anchor apply.
        rec["candidate_action_metrics"] = {}
        rec["candidate_action_waits"] = {}
        rec["pareto_feasible"] = {}
        rec["constraint_violations"] = {}
        rec["sim_horizon"] = None
        rec["expert_action_margin"] = 0.0
    rec["sample_weight"] = float(sample_weight)
    rec["crcd_teacher"] = teacher
    rec["crcd_reliable"] = bool(reliable)
    if advantage_value is not None:
        rec["crcd_advantage"] = float(advantage_value)
    return rec


def build_teacher_records(
    aligned: Dict[Tuple[str, str], Dict[str, Dict]],
    teacher: str,
    params: Dict[str, Any],
) -> List[Dict]:
    """Produce one output record per aligned state for the given teacher."""
    unreliable_weight = float(params.get("unreliable_weight", 0.3))
    prcat_min_weight = float(params.get("prcat_min_weight", 0.15))
    if not 0.0 < prcat_min_weight <= 1.0:
        raise ValueError("prcat_min_weight must be in (0, 1]")
    out: List[Dict] = []
    for key in sorted(aligned):
        recs = aligned[key]
        rec30, rec120, rec300 = recs["30"], recs["120"], recs["300"]
        base = rec30.get("base_action") or rec300.get("base_action")
        base = base if base in ACTION_LIST else ACTION_LIST[0]

        if teacher == "c30":
            b = best_action(rec30) or base
            out.append(_attach_label(_base_record(rec30), b, rec30, 1.0, True, teacher, advantage(rec30, b)))
        elif teacher == "c300":
            b = best_action(rec300) or base
            out.append(_attach_label(_base_record(rec30), b, rec300, 1.0, True, teacher, advantage(rec300, b)))
        elif teacher == "crcd_nogate":
            b30, b120, b300 = best_action(rec30), best_action(rec120), best_action(rec300)
            b = majority_vote([b30 or base, b120 or base, b300 or base]) or base
            # Use the 300s costs so soft targets are long-horizon; the hard CE
            # target is the majority vote.
            out.append(_attach_label(_base_record(rec30), b, rec300, 1.0, True, teacher, advantage(rec300, b)))
        elif teacher == "crcd":
            reliable, reason = is_reliable(rec30, rec120, rec300, params)
            if reliable:
                b = best_action(rec300) or base
                out.append(_attach_label(_base_record(rec30), b, rec300, 1.0, True, teacher, advantage(rec300, b)))
            else:
                out.append(_attach_label(_base_record(rec30), base, None, unreliable_weight, False, teacher, None))
        elif teacher == "prcat":
            # Candidate costs are min-max normalized per state, so a positive
            # source-relative advantage is already bounded by one.
            b = best_action(rec120) or base
            adv = advantage(rec120, b)
            bounded_adv = max(0.0, min(1.0, float(adv or 0.0)))
            weight = prcat_min_weight + (1.0 - prcat_min_weight) * bounded_adv
            rec = _attach_label(_base_record(rec120), b, rec120, weight, True, teacher, adv)
            rec["prcat_advantage_weight"] = float(weight)
            rec["prcat_source_deviation"] = bool(b != base)
            out.append(rec)
        else:
            raise ValueError(f"Unknown teacher: {teacher}")
    return out


# ---------------------------------------------------------------------------
# Selection + split (pure python)
# ---------------------------------------------------------------------------


def select_snapshot_times(
    snapshot_order: List[str], K: int, seed: int
) -> Tuple[List[str], List[str]]:
    """Randomly select K snapshots (seeded). Returns (selected, unselected)."""
    import random

    rng = random.Random(seed)
    order = list(snapshot_order)
    rng.shuffle(order)
    k = min(int(K), len(order))
    selected = sorted(order[:k])
    unselected = sorted(order[k:])
    return selected, unselected


def split_selected_times(
    selected: List[str], val_ratio: float, training_seed: int
) -> Tuple[List[str], List[str]]:
    """Snapshot-level train/val split over the selected snapshots."""
    import random

    rng = random.Random(training_seed)
    times = list(selected)
    rng.shuffle(times)
    n_val = max(1, int(round(len(times) * val_ratio)))
    if n_val >= len(times):
        n_val = max(1, len(times) // 2)
    val = sorted(times[:n_val])
    train = sorted(times[n_val:])
    return train, val


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def gate_stats(aligned: Dict[Tuple[str, str], Dict[str, Dict]], params: Dict[str, Any]) -> Dict:
    reasons: Counter = Counter()
    n_reliable = 0
    n = len(aligned)
    advs: List[float] = []
    for key in sorted(aligned):
        recs = aligned[key]
        reliable, reason = is_reliable(recs["30"], recs["120"], recs["300"], params)
        reasons[reason] += 1
        if reliable:
            n_reliable += 1
        a = advantage(recs["300"], best_action(recs["300"]) or "")
        if a is not None:
            advs.append(a)
    import statistics

    return {
        "n_states": n,
        "n_reliable": n_reliable,
        "reliable_fraction": (n_reliable / n) if n else 0.0,
        "reasons": dict(reasons),
        "adv_300_best": {
            "mean": statistics.fmean(advs) if advs else 0.0,
            "p25": _quantile(advs, 0.25),
            "p50": _quantile(advs, 0.50),
            "p75": _quantile(advs, 0.75),
            "min": min(advs) if advs else 0.0,
            "max": max(advs) if advs else 0.0,
        },
    }


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, int(q * (len(s) - 1)))
    return s[idx]


def build_dataset(
    aligned: Dict[Tuple[str, str], Dict[str, Dict]],
    snapshot_order: List[str],
    teacher: str,
    K: int,
    selection_seed: int,
    training_seed: int,
    val_ratio: float,
    params: Dict[str, Any],
    out_dir: str,
) -> Dict[str, Any]:
    selected, _ = select_snapshot_times(snapshot_order, K, selection_seed)
    train_times, val_times = split_selected_times(selected, val_ratio, training_seed)
    train_set, val_set = set(train_times), set(val_times)

    records = build_teacher_records(aligned, teacher, params)
    train_recs = [r for r in records if r.get("snapshot_path") in train_set]
    val_recs = [r for r in records if r.get("snapshot_path") in val_set]

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train_data.jsonl")
    val_path = os.path.join(out_dir, "val_data.jsonl")
    _write_jsonl(train_path, train_recs)
    _write_jsonl(val_path, val_recs)

    n_rel_train = sum(1 for r in train_recs if r.get("crcd_reliable"))
    meta = {
        "teacher": teacher,
        "K": K,
        "selection_seed": selection_seed,
        "training_seed": training_seed,
        "n_snapshots_selected": len(selected),
        "n_snapshots_train": len(train_times),
        "n_snapshots_val": len(val_times),
        "n_train_records": len(train_recs),
        "n_val_records": len(val_recs),
        "n_reliable_train": n_rel_train,
        "train_action_dist": dict(Counter(r["expert_action"] for r in train_recs)),
        "train_data_path": train_path,
        "val_data_path": val_path,
    }
    with open(os.path.join(out_dir, "dataset_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def _write_jsonl(path: str, records: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="CRCD relabeler / gate")
    parser.add_argument("--f30", required=True)
    parser.add_argument("--f120", required=True)
    parser.add_argument("--f300", required=True)
    parser.add_argument("--mode", choices=["gate_stats", "build"], default="gate_stats")
    parser.add_argument("--teacher", choices=TEACHER_VARIANTS, default="crcd")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--selection_seed", type=int, default=2024)
    parser.add_argument("--training_seed", type=int, default=2024)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--adv_margin", type=float, default=0.05)
    parser.add_argument("--unreliable_weight", type=float, default=0.3)
    parser.add_argument("--no_require_through", action="store_true")
    parser.add_argument("--no_require_feasible", action="store_true")
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    aligned, snapshot_order = load_aligned(args.f30, args.f120, args.f300)
    print(f"Aligned states: {len(aligned)}; snapshots: {len(snapshot_order)}")

    params = {
        "adv_margin": args.adv_margin,
        "unreliable_weight": args.unreliable_weight,
        "require_through": not args.no_require_through,
        "require_feasible": not args.no_require_feasible,
    }

    if args.mode == "gate_stats":
        stats = gate_stats(aligned, params)
        print(json.dumps(stats, indent=2))
        out_dir = args.out_dir or "records/pipeline_runs/crcd_gate_stats"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "gate_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Saved gate_stats to {out_dir}/gate_stats.json")
    else:
        if args.out_dir is None:
            args.out_dir = (
                f"records/pipeline_runs/crcd_dataset/{args.teacher}/"
                f"k{args.K}_sel{args.selection_seed}_tr{args.training_seed}"
            )
        meta = build_dataset(
            aligned, snapshot_order, args.teacher, args.K, args.selection_seed,
            args.training_seed, args.val_ratio, params, args.out_dir,
        )
        print(json.dumps(meta, indent=2))
        print(f"Wrote dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
