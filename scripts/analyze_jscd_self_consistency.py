#!/usr/bin/env python3
"""Audit whether a current policy materially changes counterfactual rankings."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ACTIONS = ("ETWT", "NTST", "ELWL", "NLSL")


def load_records(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record: dict[str, Any] = json.loads(line)
            key = (str(record.get("snapshot_path", "")), str(record.get("intersection_id", "")))
            if not all(key):
                raise ValueError(f"{path}:{line_number}: missing snapshot_path or intersection_id")
            if key in records:
                raise ValueError(f"{path}:{line_number}: duplicate decision state {key}")
            records[key] = record
    if not records:
        raise ValueError(f"No decision states found in {path}")
    return records


def score_vector(record: dict[str, Any]) -> dict[str, float]:
    raw = record.get("candidate_action_waits")
    if not isinstance(raw, dict):
        raise ValueError("candidate_action_waits must be a mapping")
    scores: dict[str, float] = {}
    for action in ACTIONS:
        try:
            score = float(raw[action])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing/non-numeric candidate score for {action}") from exc
        if not math.isfinite(score):
            raise ValueError(f"non-finite candidate score for {action}")
        scores[action] = score
    return scores


def top1_action(record: dict[str, Any], scores: dict[str, float]) -> str:
    expert = str(record.get("expert_action", ""))
    if expert in ACTIONS:
        return expert
    return min(ACTIONS, key=lambda action: (scores[action], action))


def kendall_tau(scores_a: dict[str, float], scores_b: dict[str, float]) -> float | None:
    concordant = 0
    discordant = 0
    for index, left in enumerate(ACTIONS):
        for right in ACTIONS[index + 1 :]:
            difference_a = scores_a[left] - scores_a[right]
            difference_b = scores_b[left] - scores_b[right]
            if abs(difference_a) <= 1e-12 or abs(difference_b) <= 1e-12:
                continue
            if (difference_a > 0) == (difference_b > 0):
                concordant += 1
            else:
                discordant += 1
    pair_count = concordant + discordant
    if pair_count == 0:
        return None
    return (concordant - discordant) / pair_count


def analyze(
    pi0_records: dict[tuple[str, str], dict[str, Any]],
    pi1_records: dict[tuple[str, str], dict[str, Any]],
    *,
    min_action_change_rate: float,
    max_mean_kendall_tau: float,
    min_informative_rank_rate: float,
) -> dict[str, Any]:
    if set(pi0_records) != set(pi1_records):
        missing_from_pi1 = sorted(set(pi0_records) - set(pi1_records))[:10]
        missing_from_pi0 = sorted(set(pi1_records) - set(pi0_records))[:10]
        raise ValueError(
            "Decision-state keys differ between continuation labels: "
            f"missing_from_pi1={missing_from_pi1}, missing_from_pi0={missing_from_pi0}"
        )

    changed_states = 0
    changed_by_snapshot: dict[str, int] = defaultdict(int)
    state_count_by_snapshot: dict[str, int] = defaultdict(int)
    taus: list[float] = []
    for key in sorted(pi0_records):
        pi0_score = score_vector(pi0_records[key])
        pi1_score = score_vector(pi1_records[key])
        if top1_action(pi0_records[key], pi0_score) != top1_action(pi1_records[key], pi1_score):
            changed_states += 1
            changed_by_snapshot[key[0]] += 1
        state_count_by_snapshot[key[0]] += 1
        tau = kendall_tau(pi0_score, pi1_score)
        if tau is not None:
            taus.append(tau)

    state_count = len(pi0_records)
    snapshot_count = len(state_count_by_snapshot)
    action_change_rate = changed_states / state_count
    rank_state_rate = len(taus) / state_count
    mean_tau = sum(taus) / len(taus) if taus else None
    snapshot_change_rate = sum(count > 0 for count in changed_by_snapshot.values()) / snapshot_count
    gates = {
        "action_change_rate": action_change_rate >= min_action_change_rate,
        "informative_rank_rate": rank_state_rate >= min_informative_rank_rate,
        "mean_kendall_tau": mean_tau is not None and mean_tau <= max_mean_kendall_tau,
    }
    return {
        "unit": "decision state (snapshot_path, intersection_id); training seed is not inferred here",
        "state_count": state_count,
        "snapshot_count": snapshot_count,
        "changed_top1_states": changed_states,
        "top1_action_change_rate": action_change_rate,
        "snapshots_with_any_top1_change": sum(count > 0 for count in changed_by_snapshot.values()),
        "snapshot_any_change_rate": snapshot_change_rate,
        "informative_rank_states": len(taus),
        "informative_rank_state_rate": rank_state_rate,
        "mean_kendall_tau": mean_tau,
        "thresholds": {
            "min_action_change_rate": min_action_change_rate,
            "max_mean_kendall_tau": max_mean_kendall_tau,
            "min_informative_rank_rate": min_informative_rank_rate,
        },
        "gate_components": gates,
        "gate_pass": all(gates.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pi0", required=True, type=Path)
    parser.add_argument("--pi1", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--min-action-change-rate", type=float, default=0.20)
    parser.add_argument("--max-mean-kendall-tau", type=float, default=0.80)
    parser.add_argument("--min-informative-rank-rate", type=float, default=0.50)
    args = parser.parse_args()
    if not 0.0 <= args.min_action_change_rate <= 1.0:
        raise ValueError("--min-action-change-rate must be in [0, 1]")
    if not -1.0 <= args.max_mean_kendall_tau <= 1.0:
        raise ValueError("--max-mean-kendall-tau must be in [-1, 1]")
    if not 0.0 <= args.min_informative_rank_rate <= 1.0:
        raise ValueError("--min-informative-rank-rate must be in [0, 1]")

    report = analyze(
        load_records(args.pi0),
        load_records(args.pi1),
        min_action_change_rate=args.min_action_change_rate,
        max_mean_kendall_tau=args.max_mean_kendall_tau,
        min_informative_rank_rate=args.min_informative_rank_rate,
    )
    report["pi0_label_path"] = str(args.pi0.resolve())
    report["pi1_label_path"] = str(args.pi1.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["gate_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
