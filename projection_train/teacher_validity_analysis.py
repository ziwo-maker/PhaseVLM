#!/usr/bin/env python3
"""Teacher-validity analysis: cross-horizon agreement of the SimExpert teacher.

Experiment #1 from refine-logs/EXPERIMENT_FINAL_REPORT.md Phase 7 (Required
Validation, steps 1 & 3): does the *short-horizon* (30s, frozen-neighbor)
counterfactual teacher — the label source used by all Stage-3 adaptation so
far — agree with the same counterfactual run at longer horizons (120s, 300s)?

If 30s disagrees substantially with 300s, then short-horizon label unreliability
is real (the Phase 7 premise) and motivates CRCD's multi-horizon reliability
gate. If they agree, horizon is not the failure mode and the objective-mismatch
story needs re-examination.

Input: one relabeled JSONL per horizon (output of scripts/relabel_simexpert.py),
keyed by (snapshot_path, intersection_id). The 30s file is the existing
Stage-3 teacher (`records/pipeline_runs/stage3_ood_20260813/ood/k118/relabeled_states.jsonl`).

Pure python — no cityflow needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

ACTION_LIST: Tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return None


def load_relabeled(path: str) -> Dict[Tuple[str, str], Dict]:
    """Load a relabeled JSONL keyed by (snapshot_path, intersection_id).

    Keeps the last record for each key (records are per-intersection; a key
    should be unique but dedup defensively).
    """
    out: Dict[Tuple[str, str], Dict] = {}
    n = 0
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
            n += 1
    return out


def scores(rec: Dict) -> Dict[str, float]:
    """Raw composite score per action (lower better), from candidate metrics."""
    cam = rec.get("candidate_action_metrics", {})
    out: Dict[str, float] = {}
    if isinstance(cam, dict):
        for a, vals in cam.items():
            s = _safe_float(vals.get("score")) if isinstance(vals, dict) else None
            if s is not None:
                out[a] = s
    return out


def best_raw(rec: Dict) -> Optional[str]:
    sc = scores(rec)
    if not sc:
        return None
    return min(sc, key=sc.get)


def ranking(sc: Dict[str, float]) -> List[str]:
    return [a for a, _ in sorted(sc.items(), key=lambda kv: kv[1])]


def kendall_tau(rank_a: List[str], rank_b: List[str]) -> Optional[float]:
    actions = list(dict.fromkeys(rank_a + rank_b))
    if len(actions) < 2:
        return 1.0
    idx_a = {a: i for i, a in enumerate(rank_a)}
    idx_b = {a: i for i, a in enumerate(rank_b)}
    conc = disc = 0
    for i, a1 in enumerate(actions):
        for a2 in actions[i + 1:]:
            if a1 not in idx_a or a2 not in idx_a or a1 not in idx_b or a2 not in idx_b:
                continue
            oa = idx_a[a1] < idx_a[a2]
            ob = idx_b[a1] < idx_b[a2]
            if oa == ob:
                conc += 1
            else:
                disc += 1
    total = conc + disc
    if total == 0:
        return 1.0
    return (conc - disc) / total


def pair_agreement(recs_a: Dict, recs_b: Dict, tag_a: str, tag_b: str) -> Dict:
    common = set(recs_a) & set(recs_b)
    if not common:
        return {"n": 0}

    raw_agree = expert_agree = 0
    tau_list: List[float] = []
    flip_pairs: Counter = Counter()
    adv_sign_flip = adv_sign_total = 0
    margin_abs: List[float] = []

    for key in common:
        ra, rb = recs_a[key], recs_b[key]
        ba = best_raw(ra)
        bb = best_raw(rb)
        ea = ra.get("expert_action")
        eb = rb.get("expert_action")

        if ba is not None and bb is not None:
            if ba == bb:
                raw_agree += 1
            flip_pairs[f"{ba}->{bb}"] += 1
            sc_a, sc_b = scores(ra), scores(rb)
            ra_rank, rb_rank = ranking(sc_a), ranking(sc_b)
            tau = kendall_tau(ra_rank, rb_rank)
            if tau is not None:
                tau_list.append(tau)

            base = ra.get("base_action") or rb.get("base_action")
            # advantage of best_raw@a over base, evaluated at both horizons
            if base in sc_a and base in sc_b and ba in sc_a and ba in sc_b:
                adv_a = sc_a[base] - sc_a[ba]   # >0 => ba better than base at H_a
                adv_b = sc_b[base] - sc_b[ba]   # same action evaluated at H_b
                if adv_a != 0.0 and adv_b != 0.0:
                    adv_sign_total += 1
                    if (adv_a > 0) != (adv_b > 0):
                        adv_sign_flip += 1
            ma = ra.get("expert_action_margin")
            if ma is not None:
                margin_abs.append(abs(float(ma)))

        if ea is not None and eb is not None:
            if ea == eb:
                expert_agree += 1

    n = len(common)
    return {
        "n": n,
        "raw_best_agreement": raw_agree / n,
        "expert_label_agreement": expert_agree / n,
        "mean_kendall_tau": float(np.mean(tau_list)) if tau_list else None,
        "kendall_tau_std": float(np.std(tau_list)) if tau_list else None,
        "adv_sign_flip_rate": (adv_sign_flip / adv_sign_total) if adv_sign_total else None,
        "adv_sign_total": adv_sign_total,
        "top_flip_transitions": dict(flip_pairs.most_common(10)),
        "mean_abs_margin_at_a": float(np.mean(margin_abs)) if margin_abs else None,
    }


def reliability(recs_by_horizon: Dict[str, Dict]) -> Dict:
    """Fraction of (snapshot, intersection) states stable across all horizons."""
    horizons = list(recs_by_horizon)
    common = set.intersection(*(set(d) for d in recs_by_horizon.values())) if recs_by_horizon else set()
    if not common or len(horizons) < 2:
        return {"n": 0}

    raw_stable = expert_stable = 0
    n = len(common)
    for key in common:
        raws = [best_raw(recs_by_horizon[h][key]) for h in horizons]
        experts = [recs_by_horizon[h][key].get("expert_action") for h in horizons]
        if raws and all(a == raws[0] for a in raws):
            raw_stable += 1
        if experts and all(a == experts[0] for a in experts):
            expert_stable += 1

    return {
        "n": n,
        "horizons": horizons,
        "raw_best_stable_fraction": raw_stable / n,
        "expert_label_stable_fraction": expert_stable / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-horizon teacher-validity analysis")
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Relabeled JSONLs, in the form path=horizon_label (e.g. x.jsonl=30)")
    parser.add_argument("--output-dir", default="records/pipeline_runs/teacher_validity_20260813")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    recs_by_horizon: Dict[str, Dict] = {}
    horizon_order: List[str] = []
    for spec in args.inputs:
        if "=" in spec:
            path, label = spec.rsplit("=", 1)
        else:
            path, label = spec, os.path.basename(spec)
        if not os.path.exists(path):
            print(f"[skip] missing {path}")
            continue
        recs = load_relabeled(path)
        print(f"[load] {label}: {len(recs)} states from {path}")
        recs_by_horizon[label] = recs
        horizon_order.append(label)

    if len(recs_by_horizon) < 2:
        print("ERROR: need at least two horizon files.")
        return

    summary = {"pairwise": {}, "reliability": {}, "per_horizon": {}}
    for label, recs in recs_by_horizon.items():
        n = len(recs)
        raw_counts = Counter(best_raw(r) for r in recs.values() if best_raw(r))
        expert_counts = Counter(r.get("expert_action") for r in recs.values())
        summary["per_horizon"][label] = {
            "n_states": n,
            "raw_best_dist": dict(raw_counts),
            "expert_label_dist": dict(expert_counts),
        }

    # Pairwise against the shortest horizon (the current teacher), plus adjacent
    base_horizon = min(horizon_order, key=lambda h: float(h) if h.replace(".", "").isdigit() else 1e9)
    for i, h2 in enumerate(horizon_order):
        if h2 == base_horizon:
            continue
        pair = pair_agreement(recs_by_horizon[base_horizon], recs_by_horizon[h2], base_horizon, h2)
        summary["pairwise"][f"{base_horizon}->{h2}"] = pair
        print(f"\n=== {base_horizon}s -> {h2}s (n={pair.get('n')}) ===")
        print(f"  raw best-action agreement:    {pair.get('raw_best_agreement', 0):.3f}")
        print(f"  expert-label agreement:       {pair.get('expert_label_agreement', 0):.3f}")
        print(f"  mean Kendall tau:             {pair.get('mean_kendall_tau')}")
        print(f"  advantage-sign flip rate:     {pair.get('adv_sign_flip_rate')}  (n={pair.get('adv_sign_total')})")

    summary["reliability"] = reliability(recs_by_horizon)
    rel = summary["reliability"]
    print(f"\n=== Reliability across {rel.get('horizons')} (n={rel.get('n')}) ===")
    print(f"  raw-best stable fraction:     {rel.get('raw_best_stable_fraction', 0):.3f}")
    print(f"  expert-label stable fraction: {rel.get('expert_label_stable_fraction', 0):.3f}")

    summary_path = os.path.join(args.output_dir, "teacher_validity_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")

    # Per-state CSV (first vs last horizon in order)
    if len(horizon_order) >= 2:
        h_first, h_last = horizon_order[0], horizon_order[-1]
        common = sorted(set(recs_by_horizon[h_first]) & set(recs_by_horizon[h_last]))
        csv_path = os.path.join(args.output_dir, "teacher_validity_per_state.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("snapshot,intersection,base_action," + ",".join(
                f"{h}_rawbest,{h}_expert,{h}_margin" for h in horizon_order) + "\n")
            for key in common:
                sp, iid = key
                row = [sp, iid]
                r0 = recs_by_horizon[h_first][key]
                row.append(r0.get("base_action", ""))
                for h in horizon_order:
                    r = recs_by_horizon[h][key]
                    row.append(best_raw(r) or "")
                    row.append(r.get("expert_action", ""))
                    row.append(f"{r.get('expert_action_margin', 0.0):.4f}")
                f.write(",".join(str(x) for x in row) + "\n")
        print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
