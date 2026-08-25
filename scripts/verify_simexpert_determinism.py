#!/usr/bin/env python3
"""Verify SimExpert relabeling is bitwise deterministic across replays.

The SimExpert oracle is used as a variance-control baseline in Stage-3
experiments. If identical snapshots relabeled in separate work directories
produce different labels/costs, then "training variance" and "oracle
non-determinism" are confounded.

This script replays `relabel_simexpert.py` N times on the SAME input, each
into a fresh work directory, then compares the outputs record-by-record:

  - expert_action / base_action   : exact-equality mismatch count
  - expert_action_margin          : max absolute difference
  - candidate_action_waits (score): tolerance-aware cost difference count
  - candidate_action_metrics      : max per-metric absolute difference
  - pareto_feasible / constraint_violations : exact-equality mismatch count
  - near-tied records             : |best - second-best| < tie tolerance

Tolerance-aware handling: candidate costs are compared with absolute+relative
tolerance so near-tied float jitter is reported separately from real label
disagreement. "bitwise_deterministic" is True only when every label and every
cost entry is within tolerance across all replay pairs.

Usage:
  /home/sysu/anaconda3/envs/hnr5090/bin/python scripts/verify_simexpert_determinism.py \
      --input records/pipeline_runs/stage3_ood_20260813/random/k8/selected_states.jsonl \
      --roadnet data/Hangzhou/4_4/roadnet_4_4.json \
      --traffic_file data/Hangzhou/4_4/anon_4_4_hangzhou_real.json \
      --dataset hangzhou \
      --output_dir records/determinism_check \
      --n_replays 3 --max_snapshots 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from projection_train.action_space import ACTION_LIST  # noqa: E402

_CONDA_PYTHON = "/home/sysu/anaconda3/envs/hnr5090/bin/python"

METRIC_KEYS = ["att", "aql", "awt", "spillback",
               "spillback_occupancy", "spillback_blocked_ratio", "score"]


def _resolve_python() -> str:
    for candidate in (_CONDA_PYTHON, sys.executable):
        if not os.path.isfile(candidate):
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", "import cityflow; print(cityflow.Engine)"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and "Engine" in result.stdout:
                return candidate
        except Exception:
            continue
    raise RuntimeError("No Python interpreter with a real cityflow found.")


def _record_key(rec: Dict) -> tuple:
    return (float(rec.get("time_s", -1.0)), str(rec.get("intersection_id", "")))


def compare_relabel_records(
    records_a: Sequence[Dict],
    records_b: Sequence[Dict],
    cost_atol: float = 1e-6,
    cost_rtol: float = 1e-6,
    tie_atol: float = 1e-6,
) -> Dict[str, Any]:
    """Compare two relabel outputs record-by-record.

    Returns a report with mismatch counts, max differences, and a
    bitwise-determinism verdict. Candidate costs are compared tolerance-aware
    so near-tied float jitter is tracked separately from label disagreement.
    """
    if len(records_a) != len(records_b):
        return {"error": "record count mismatch",
                "n_a": len(records_a), "n_b": len(records_b)}
    keys_a = [_record_key(r) for r in records_a]
    keys_b = [_record_key(r) for r in records_b]
    if keys_a != keys_b:
        return {"error": "record key order mismatch",
                "first_diff_a": keys_a[:3], "first_diff_b": keys_b[:3]}

    n = len(records_a)
    n_expert_mismatch = 0
    n_base_mismatch = 0
    n_margin_diff = 0
    n_cost_diff = 0
    n_metric_diff = 0
    n_feasibility_mismatch = 0
    n_constraint_mismatch = 0
    n_near_tie = 0
    max_margin_abs = 0.0
    max_cost_abs = 0.0
    max_metric_abs = 0.0

    for a, b in zip(records_a, records_b):
        if a.get("expert_action") != b.get("expert_action"):
            n_expert_mismatch += 1
        if a.get("base_action") != b.get("base_action"):
            n_base_mismatch += 1

        # Near-tied best/second-best (from replay A margins)
        if float(a.get("expert_action_margin", 0.0) or 0.0) < tie_atol:
            n_near_tie += 1

        ma = float(a.get("expert_action_margin", 0.0) or 0.0)
        mb = float(b.get("expert_action_margin", 0.0) or 0.0)
        d = abs(ma - mb)
        if d > cost_atol:
            n_margin_diff += 1
            max_margin_abs = max(max_margin_abs, d)

        wa = a.get("candidate_action_waits", {})
        wb = b.get("candidate_action_waits", {})
        wa = wa if isinstance(wa, dict) else {}
        wb = wb if isinstance(wb, dict) else {}
        for action in ACTION_LIST:
            sa = float(wa.get(action, 0.0) or 0.0)
            sb = float(wb.get(action, 0.0) or 0.0)
            dd = abs(sa - sb)
            if dd > cost_atol + cost_rtol * abs(sb):
                n_cost_diff += 1
                max_cost_abs = max(max_cost_abs, dd)

        ca = a.get("candidate_action_metrics", {})
        cb = b.get("candidate_action_metrics", {})
        ca = ca if isinstance(ca, dict) else {}
        cb = cb if isinstance(cb, dict) else {}
        for action in ACTION_LIST:
            ma_ = ca.get(action, {})
            mb_ = cb.get(action, {})
            ma_ = ma_ if isinstance(ma_, dict) else {}
            mb_ = mb_ if isinstance(mb_, dict) else {}
            for m in METRIC_KEYS:
                va = float(ma_.get(m, 0.0) or 0.0)
                vb = float(mb_.get(m, 0.0) or 0.0)
                dm = abs(va - vb)
                if dm > cost_atol + cost_rtol * abs(vb):
                    n_metric_diff += 1
                    max_metric_abs = max(max_metric_abs, dm)

        if a.get("pareto_feasible") != b.get("pareto_feasible"):
            n_feasibility_mismatch += 1
        if a.get("constraint_violations") != b.get("constraint_violations"):
            n_constraint_mismatch += 1

    bitwise = (n_expert_mismatch == 0 and n_base_mismatch == 0
               and n_feasibility_mismatch == 0 and n_constraint_mismatch == 0
               and n_cost_diff == 0 and n_margin_diff == 0
               and n_metric_diff == 0)
    return {
        "n_records": n,
        "n_near_tie": n_near_tie,
        "n_expert_action_mismatch": n_expert_mismatch,
        "n_base_action_mismatch": n_base_mismatch,
        "n_margin_diff": n_margin_diff,
        "max_margin_abs": max_margin_abs,
        "n_cost_diff": n_cost_diff,
        "max_cost_abs": max_cost_abs,
        "n_metric_diff": n_metric_diff,
        "max_metric_abs": max_metric_abs,
        "n_feasibility_mismatch": n_feasibility_mismatch,
        "n_constraint_mismatch": n_constraint_mismatch,
        "bitwise_deterministic": bitwise,
    }


def _load_jsonl(path: str) -> List[Dict]:
    return [json.loads(line) for line in open(path, "r", encoding="utf-8")
            if line.strip()]


def _run(cmd: List[str], desc: str) -> None:
    print(f"[{desc}] {' '.join(cmd[:8])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO)
    if result.returncode != 0:
        print(f"  STDOUT (last 500): {result.stdout[-500:]}")
        print(f"  STDERR (last 500): {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd[:6])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--roadnet", required=True)
    parser.add_argument("--traffic_file", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_replays", type=int, default=3)
    parser.add_argument("--max_snapshots", type=int, default=0,
                        help="Limit to first N snapshots (0 = all) for a fast check")
    parser.add_argument("--sim_horizon", type=int, default=30)
    parser.add_argument("--cost_atol", type=float, default=1e-6)
    parser.add_argument("--cost_rtol", type=float, default=1e-6)
    parser.add_argument("--tie_atol", type=float, default=1e-6)
    parser.add_argument("--python", default="")
    args = parser.parse_args()

    python = args.python or _resolve_python()
    print(f"Interpreter: {python}")

    os.makedirs(args.output_dir, exist_ok=True)
    relabel_script = os.path.join(_REPO, "scripts", "relabel_simexpert.py")

    # Optionally subsample to the first `max_snapshots` distinct snapshot times
    input_path = os.path.abspath(args.input)
    if args.max_snapshots > 0:
        records = _load_jsonl(input_path)
        seen: List[float] = []
        kept: List[Dict] = []
        for rec in records:
            t = float(rec.get("time_s", -1.0))
            if t not in seen:
                seen.append(t)
                if len(seen) > args.max_snapshots:
                    break
            kept.append(rec)
        input_path = os.path.join(args.output_dir, "_subsampled_input.jsonl")
        with open(input_path, "w", encoding="utf-8") as f:
            for rec in kept:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Subsampled to {len(kept)} records / {len(seen)} snapshots")

    outputs: List[str] = []
    for i in range(args.n_replays):
        out_path = os.path.join(args.output_dir, f"relabel_r{i}.jsonl")
        work_dir = os.path.join(args.output_dir, f"work_r{i}")
        cmd = [
            python, relabel_script,
            "--input", input_path,
            "--output", out_path,
            "--roadnet", os.path.abspath(args.roadnet),
            "--traffic_file", os.path.abspath(args.traffic_file),
            "--dataset", args.dataset,
            "--work_dir", os.path.abspath(work_dir),
            "--sim_horizon", str(args.sim_horizon),
        ]
        _run(cmd, desc=f"Replay {i + 1}/{args.n_replays}")
        outputs.append(out_path)

    all_records = [_load_jsonl(p) for p in outputs]
    print(f"Loaded {[len(r) for r in all_records]} records per replay")

    # Compare every replay against replay 0
    pair_reports: Dict[str, Dict[str, Any]] = {}
    for i in range(1, args.n_replays):
        rep = compare_relabel_records(
            all_records[0], all_records[i],
            cost_atol=args.cost_atol, cost_rtol=args.cost_rtol,
            tie_atol=args.tie_atol,
        )
        pair_reports[f"r0_vs_r{i}"] = rep
        print(f"\n=== replay 0 vs replay {i} ===")
        print(json.dumps(rep, indent=2))

    all_bitwise = all(r.get("bitwise_deterministic", False)
                      for r in pair_reports.values())
    report = {
        "input": args.input,
        "n_replays": args.n_replays,
        "sim_horizon": args.sim_horizon,
        "cost_atol": args.cost_atol,
        "cost_rtol": args.cost_rtol,
        "tie_atol": args.tie_atol,
        "pairs": pair_reports,
        "all_bitwise_deterministic": all_bitwise,
    }
    report_path = os.path.join(args.output_dir, "determinism_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")
    print(f"Verdict: all_bitwise_deterministic = {all_bitwise}")


if __name__ == "__main__":
    main()
