"""Adjacent-intersection joint counterfactual SimExpert relabeling.

This teacher evaluates both endpoints of one adjacent road-network edge under
every joint phase pair.  Every one of the 16 candidates is restored from the
same CityFlow snapshot.  The first control interval forces both candidate
phases, then the existing source-policy continuation controller takes over.

The Stage-2 trainer consumes per-intersection action costs, so a joint 4x4
cost table is projected to each endpoint with a min-marginal:

``m_i(a_i) = min_{a_j} C_ij(a_i, a_j)``.

The full edge table remains in record provenance for later coordination and
counterfactual analyses.  This is an implementation of established
counterfactual multi-agent credit-assignment ideas, not a novelty claim.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.policy_responsive_simexpert import (
    PolicyResponsiveSimExpertRelabeler,
    RollingSequenceBank,
    _finite_float,
    load_snapshot_phase_indices,
)
from scripts.relabel_simexpert import _metric_mean, _metric_sum


JointAction = Tuple[str, str]
JointPair = Tuple[str, str]


def force_joint_target_actions(
    source_actions: Sequence[int],
    first_index: int,
    first_action_name: str,
    second_index: int,
    second_action_name: str,
) -> List[int]:
    """Override exactly two distinct positions in a network action vector."""
    if first_index == second_index:
        raise ValueError("Joint counterfactual endpoints must be distinct")
    actions = [int(action) for action in source_actions]
    for index in (first_index, second_index):
        if index < 0 or index >= len(actions):
            raise IndexError(f"target index {index} out of range for {len(actions)} actions")
    actions[first_index] = action_to_index(first_action_name)
    actions[second_index] = action_to_index(second_action_name)
    return actions


def compose_joint_counterfactual_actions(
    source_actions: Sequence[int],
    snapshot_actions: Sequence[int],
    first_index: int,
    second_index: int,
    *,
    neighbor_mode: str,
    first_action_name: Optional[str] = None,
    second_action_name: Optional[str] = None,
) -> List[int]:
    """Build actions for one step of a two-intersection intervention.

    During the first interval both endpoint candidates are forced while every
    other intersection keeps the snapshot phase.  Later, responsive mode lets
    the source policy control the full network; frozen mode only lets the two
    intervened endpoints replan.
    """
    if len(source_actions) != len(snapshot_actions):
        raise ValueError(
            "source_actions and snapshot_actions must have the same length: "
            f"{len(source_actions)} != {len(snapshot_actions)}"
        )
    if first_index == second_index:
        raise ValueError("Joint counterfactual endpoints must be distinct")
    for index in (first_index, second_index):
        if index < 0 or index >= len(source_actions):
            raise IndexError(f"target index {index} out of range for {len(source_actions)} actions")
    if neighbor_mode not in {"responsive", "frozen"}:
        raise ValueError(f"Unsupported neighbor_mode={neighbor_mode!r}")
    if (first_action_name is None) != (second_action_name is None):
        raise ValueError("Both joint target actions must be provided for the initial interval")

    if first_action_name is not None and second_action_name is not None:
        return force_joint_target_actions(
            snapshot_actions,
            first_index,
            first_action_name,
            second_index,
            second_action_name,
        )
    if neighbor_mode == "responsive":
        return [int(action) for action in source_actions]

    actions = [int(action) for action in snapshot_actions]
    actions[first_index] = int(source_actions[first_index])
    actions[second_index] = int(source_actions[second_index])
    return actions


def intersection_grid_position(intersection_id: str) -> Optional[Tuple[int, int]]:
    """Parse a CityFlow grid ID such as ``intersection_2_3``."""
    parts = str(intersection_id).split("_")
    if len(parts) < 3 or parts[0] != "intersection":
        return None
    try:
        return int(parts[-2]), int(parts[-1])
    except ValueError:
        return None


def adjacent_intersection_pairs(intersection_ids: Iterable[str]) -> List[JointPair]:
    """Return deterministic Manhattan-adjacent pairs among known grid IDs."""
    positions = {
        intersection_id: position
        for intersection_id in sorted({str(value) for value in intersection_ids if str(value)})
        if (position := intersection_grid_position(intersection_id)) is not None
    }
    identifiers = sorted(positions)
    pairs: List[JointPair] = []
    for first_offset, first_id in enumerate(identifiers):
        first_position = positions[first_id]
        for second_id in identifiers[first_offset + 1 :]:
            second_position = positions[second_id]
            if abs(first_position[0] - second_position[0]) + abs(first_position[1] - second_position[1]) == 1:
                pairs.append((first_id, second_id))
    return pairs


def select_joint_pairs(
    intersection_ids: Iterable[str],
    *,
    pressures: Optional[Mapping[str, float]] = None,
    max_pairs: int = 1,
    selection: str = "max_pressure",
) -> List[JointPair]:
    """Choose non-overlapping adjacent edges with deterministic tie-breaking."""
    if max_pairs < 0:
        raise ValueError("max_pairs must be non-negative")
    if selection not in {"max_pressure", "lexicographic"}:
        raise ValueError(f"Unsupported joint pair selection: {selection!r}")
    pairs = adjacent_intersection_pairs(intersection_ids)
    if selection == "max_pressure":
        pressure_map = pressures or {}
        pairs.sort(
            key=lambda pair: (
                -(_finite_float(pressure_map.get(pair[0])) + _finite_float(pressure_map.get(pair[1]))),
                pair,
            )
        )

    selected: List[JointPair] = []
    occupied: set[str] = set()
    for pair in pairs:
        if max_pairs and len(selected) >= max_pairs:
            break
        if pair[0] in occupied or pair[1] in occupied:
            continue
        selected.append(pair)
        occupied.update(pair)
    return selected


def joint_min_marginals(
    pair_costs: Mapping[JointAction, float],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str], Dict[str, str]]:
    """Compute endpoint min-marginals and the selected partner action.

    Returns ``(first_costs, second_costs, first_partners, second_partners)``.
    Ties use ``ACTION_LIST`` order, which keeps JSONL output reproducible.
    """
    first_costs: Dict[str, float] = {}
    second_costs: Dict[str, float] = {}
    first_partners: Dict[str, str] = {}
    second_partners: Dict[str, str] = {}

    for first_action in ACTION_LIST:
        candidates = [
            (float(pair_costs[(first_action, second_action)]), action_to_index(second_action), second_action)
            for second_action in ACTION_LIST
            if (first_action, second_action) in pair_costs
        ]
        if candidates:
            score, _, partner = min(candidates)
            first_costs[first_action] = score
            first_partners[first_action] = partner

    for second_action in ACTION_LIST:
        candidates = [
            (float(pair_costs[(first_action, second_action)]), action_to_index(first_action), first_action)
            for first_action in ACTION_LIST
            if (first_action, second_action) in pair_costs
        ]
        if candidates:
            score, _, partner = min(candidates)
            second_costs[second_action] = score
            second_partners[second_action] = partner
    return first_costs, second_costs, first_partners, second_partners


def _record_pressure(record: Mapping[str, Any]) -> float:
    """Extract a stable congestion proxy for edge selection from a state record."""
    for key in ("phase_queue", "movement_counts", "traffic_counts"):
        values = record.get(key)
        if isinstance(values, Mapping):
            return sum(max(0.0, _finite_float(value)) for value in values.values())
    return 0.0


class JointCounterfactualSimExpertRelabeler(PolicyResponsiveSimExpertRelabeler):
    """Policy-responsive teacher with adjacent-pair joint action enumeration."""

    def __init__(
        self,
        *args: Any,
        max_joint_pairs_per_snapshot: int = 1,
        joint_pair_selection: str = "max_pressure",
        joint_marginalization: str = "min",
        joint_only: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if int(max_joint_pairs_per_snapshot) < 1:
            raise ValueError("max_joint_pairs_per_snapshot must be positive")
        if joint_pair_selection not in {"max_pressure", "lexicographic"}:
            raise ValueError("joint_pair_selection must be 'max_pressure' or 'lexicographic'")
        if joint_marginalization != "min":
            raise ValueError("Only min joint marginalization is currently supported")
        self.max_joint_pairs_per_snapshot = int(max_joint_pairs_per_snapshot)
        self.joint_pair_selection = str(joint_pair_selection)
        self.joint_marginalization = str(joint_marginalization)
        self.joint_only = bool(joint_only)

    @staticmethod
    def _snapshot_actions(intersection_ids: Sequence[str], phases: Mapping[str, int]) -> List[int]:
        return [
            action_to_index(ACTION_LIST[max(1, min(len(ACTION_LIST), int(phases.get(intersection_id, 1)))) - 1])
            for intersection_id in intersection_ids
        ]

    def _simulate_joint_candidate(
        self,
        *,
        snapshot_path: str,
        snapshot_records: Sequence[Mapping[str, Any]],
        first_intersection_id: str,
        second_intersection_id: str,
        first_action_name: str,
        second_action_name: str,
        snapshot_time_s: float,
    ) -> Tuple[Dict[str, float], Dict[str, int], JointAction]:
        """Evaluate one ordered joint phase pair from a restored snapshot."""
        env, phases = self._restore_snapshot(snapshot_path)
        intersection_ids = [intersection.inter_name for intersection in env.list_intersection]
        runner = self._runner
        assert runner is not None
        sequences = RollingSequenceBank.from_records(
            snapshot_records,
            intersection_ids,
            seq_len=runner.runtime.seq_len,
            input_dim=runner.runtime.input_dim,
            interval_seconds=self.traffic_series_interval,
            snapshot_time_s=snapshot_time_s,
        )
        first_index = intersection_ids.index(first_intersection_id)
        second_index = intersection_ids.index(second_intersection_id)
        source_actions, response_actions = self._policy_actions(env, sequences)
        base_actions = (
            ACTION_LIST[int(source_actions[first_index])],
            ACTION_LIST[int(source_actions[second_index])],
        )
        snapshot_actions = self._snapshot_actions(intersection_ids, phases)
        actions = compose_joint_counterfactual_actions(
            source_actions,
            snapshot_actions,
            first_index,
            second_index,
            neighbor_mode=self.neighbor_mode,
            first_action_name=first_action_name,
            second_action_name=second_action_name,
        )

        traces: Dict[str, List[float]] = defaultdict(list)
        for elapsed in range(self.sim_horizon):
            if elapsed > 0 and elapsed % self.min_action_time == 0:
                sequences.advance_from_cityflow(env)
                source_actions, counts = self._policy_actions(env, sequences)
                response_actions.update(counts)
                actions = compose_joint_counterfactual_actions(
                    source_actions,
                    snapshot_actions,
                    first_index,
                    second_index,
                    neighbor_mode=self.neighbor_mode,
                )
            env._inner_step(actions)
            first_metrics = self._measure_branch(env, first_intersection_id)
            second_metrics = self._measure_branch(env, second_intersection_id)
            for metric in ("att", "aql", "awt", "spillback"):
                traces[f"first_local_{metric}"].append(first_metrics[f"local_{metric}"])
                traces[f"second_local_{metric}"].append(second_metrics[f"local_{metric}"])
                traces[f"global_{metric}"].append(first_metrics[f"global_{metric}"])

        metrics: Dict[str, float] = {}
        for metric in ("att", "aql", "awt", "spillback"):
            aggregate = _metric_sum if metric == "awt" else _metric_mean
            metrics[f"first_local_{metric}"] = aggregate(traces[f"first_local_{metric}"])
            metrics[f"second_local_{metric}"] = aggregate(traces[f"second_local_{metric}"])
            metrics[f"global_{metric}"] = aggregate(traces[f"global_{metric}"])
            metrics[metric] = (
                metrics[f"global_{metric}"]
                if self.cost_scope == "global"
                else metrics[f"first_local_{metric}"] + metrics[f"second_local_{metric}"]
            )
        metrics["joint_score_scope"] = self.cost_scope
        return metrics, dict(response_actions), base_actions

    @staticmethod
    def _pair_table(values: Mapping[JointAction, Any]) -> Dict[str, Dict[str, Any]]:
        return {
            first_action: {
                second_action: values[(first_action, second_action)]
                for second_action in ACTION_LIST
                if (first_action, second_action) in values
            }
            for first_action in ACTION_LIST
            if any((first_action, second_action) in values for second_action in ACTION_LIST)
        }

    def _score_joint_candidates(
        self,
        raw_metrics: Mapping[JointAction, Mapping[str, float]],
    ) -> Tuple[Dict[JointAction, float], Dict[JointAction, Dict[str, float]]]:
        """Score all 16 branches jointly using the existing normalized objective."""
        tags = {pair: f"{pair[0]}::{pair[1]}" for pair in raw_metrics}
        scoring_metrics = {
            tags[pair]: {metric: values[metric] for metric in ("att", "aql", "awt", "spillback")}
            for pair, values in raw_metrics.items()
        }
        tag_scores, tag_metrics = self._score_candidates(scoring_metrics)
        scores: Dict[JointAction, float] = {}
        enriched: Dict[JointAction, Dict[str, float]] = {}
        for pair, tag in tags.items():
            scores[pair] = tag_scores[tag]
            enriched[pair] = {
                **tag_metrics[tag],
                **raw_metrics[pair],
                "score": tag_scores[tag],
                "joint_action_first": pair[0],
                "joint_action_second": pair[1],
                "score_scope": self.cost_scope,
            }
        return scores, enriched

    @staticmethod
    def _select_best_action(
        scores: Mapping[str, float],
        pareto_feasible: Mapping[str, bool],
        base_action: str,
    ) -> Tuple[str, float]:
        best = base_action if base_action in scores else ACTION_LIST[0]
        best_score = float("inf")
        for action in ACTION_LIST:
            if action not in scores or not pareto_feasible.get(action, True):
                continue
            score = float(scores[action])
            if score < best_score - 1e-9 or (
                abs(score - best_score) <= 1e-9 and action == base_action
            ):
                best, best_score = action, score
        return best, best_score

    def _endpoint_result(
        self,
        *,
        scores: Mapping[str, float],
        metrics: Mapping[str, Mapping[str, float]],
        base_action: str,
    ) -> Dict[str, Any]:
        copied_metrics = {action: dict(values) for action, values in metrics.items()}
        pareto, violations, _ = self._check_pareto_feasibility(copied_metrics, base_action)
        best, _ = self._select_best_action(scores, pareto, base_action)
        feasible = sorted(
            float(scores[action])
            for action in ACTION_LIST
            if action in scores and pareto.get(action, True)
        )
        margin = feasible[1] - feasible[0] if len(feasible) >= 2 else 0.0
        base_score = scores.get(base_action)
        advantages = {
            action: (float(base_score - score) if base_score is not None else 0.0)
            for action, score in scores.items()
        }
        return {
            "candidate_action_metrics": copied_metrics,
            "candidate_action_waits": dict(scores),
            "expert_action": best,
            "expert_action_margin": margin,
            "base_action": base_action,
            "pareto_feasible": pareto,
            "constraint_violations": dict(violations),
            "candidate_advantages": advantages,
        }

    def _label_unilateral(
        self,
        *,
        snapshot_path: str,
        snapshot_records: Sequence[Mapping[str, Any]],
        target_intersection_id: str,
        snapshot_time_s: float,
    ) -> Dict[str, Any]:
        """Generate the existing individual teacher label for a non-pair endpoint."""
        raw_metrics: Dict[str, Dict[str, float]] = {}
        response_counts: Dict[str, Dict[str, int]] = {}
        base_action: Optional[str] = None
        for action in ACTION_LIST:
            metrics, counts, candidate_base_action = self._simulate_candidate(
                snapshot_path=snapshot_path,
                snapshot_records=snapshot_records,
                target_intersection_id=target_intersection_id,
                target_action_name=action,
                snapshot_time_s=snapshot_time_s,
            )
            raw_metrics[action] = metrics
            response_counts[action] = counts
            if base_action is None:
                base_action = candidate_base_action
            elif base_action != candidate_base_action:
                raise RuntimeError(
                    "Source-policy action changed across deterministic candidate branches: "
                    f"{base_action} != {candidate_base_action}"
                )
        if base_action is None:
            raise RuntimeError(f"No source-policy action resolved for {target_intersection_id}")

        scoring_metrics = {
            action: {metric: values[metric] for metric in ("att", "aql", "awt", "spillback")}
            for action, values in raw_metrics.items()
        }
        scores, enriched = self._score_candidates(scoring_metrics)
        for action, values in enriched.items():
            values.update(raw_metrics[action])
            values["score_scope"] = self.cost_scope
            values["response_action_counts"] = response_counts[action]
        result = self._endpoint_result(scores=scores, metrics=enriched, base_action=base_action)
        result.update(
            {
                "joint_counterfactual": False,
                "joint_fallback": "unilateral_policy_responsive",
            }
        )
        return result

    def _label_joint_pair(
        self,
        *,
        snapshot_path: str,
        snapshot_records: Sequence[Mapping[str, Any]],
        pair: JointPair,
        snapshot_time_s: float,
    ) -> Dict[str, Dict[str, Any]]:
        first_id, second_id = pair
        raw_metrics: Dict[JointAction, Dict[str, float]] = {}
        response_counts: Dict[JointAction, Dict[str, int]] = {}
        base_actions: Optional[JointAction] = None
        for first_action in ACTION_LIST:
            for second_action in ACTION_LIST:
                action_pair = (first_action, second_action)
                metrics, counts, candidate_base_actions = self._simulate_joint_candidate(
                    snapshot_path=snapshot_path,
                    snapshot_records=snapshot_records,
                    first_intersection_id=first_id,
                    second_intersection_id=second_id,
                    first_action_name=first_action,
                    second_action_name=second_action,
                    snapshot_time_s=snapshot_time_s,
                )
                raw_metrics[action_pair] = metrics
                response_counts[action_pair] = counts
                if base_actions is None:
                    base_actions = candidate_base_actions
                elif base_actions != candidate_base_actions:
                    raise RuntimeError(
                        "Source-policy actions changed across deterministic joint candidate branches: "
                        f"{base_actions} != {candidate_base_actions}"
                    )
        if base_actions is None:
            raise RuntimeError(f"No source-policy actions resolved for joint pair {pair}")

        pair_scores, pair_metrics = self._score_joint_candidates(raw_metrics)
        first_scores, second_scores, first_partners, second_partners = joint_min_marginals(pair_scores)
        first_metrics: Dict[str, Dict[str, float]] = {}
        second_metrics: Dict[str, Dict[str, float]] = {}
        edge_id = f"{first_id}__{second_id}"
        for action, partner in first_partners.items():
            action_pair = (action, partner)
            first_metrics[action] = {
                **pair_metrics[action_pair],
                "joint_partner_action": partner,
                "joint_pair_score": pair_scores[action_pair],
                "joint_marginalization": self.joint_marginalization,
                "joint_edge_id": edge_id,
                "response_action_counts": response_counts[action_pair],
            }
        for action, partner in second_partners.items():
            action_pair = (partner, action)
            second_metrics[action] = {
                **pair_metrics[action_pair],
                "joint_partner_action": partner,
                "joint_pair_score": pair_scores[action_pair],
                "joint_marginalization": self.joint_marginalization,
                "joint_edge_id": edge_id,
                "response_action_counts": response_counts[action_pair],
            }

        best_pair = min(
            pair_scores,
            key=lambda action_pair: (
                pair_scores[action_pair],
                0 if action_pair == base_actions else 1,
                action_to_index(action_pair[0]),
                action_to_index(action_pair[1]),
            ),
        )
        pair_cost_table = self._pair_table(pair_scores)
        pair_metric_table = self._pair_table(pair_metrics)
        common = {
            "joint_counterfactual": True,
            "joint_edge_id": edge_id,
            "joint_pair_costs": pair_cost_table,
            "joint_pair_metrics": pair_metric_table,
            "joint_pair_best_actions": {first_id: best_pair[0], second_id: best_pair[1]},
            "joint_pair_best_score": pair_scores[best_pair],
            "joint_pair_base_actions": {first_id: base_actions[0], second_id: base_actions[1]},
            "joint_marginalization": self.joint_marginalization,
            "joint_score_scope": self.cost_scope,
        }
        first_result = self._endpoint_result(
            scores=first_scores,
            metrics=first_metrics,
            base_action=base_actions[0],
        )
        second_result = self._endpoint_result(
            scores=second_scores,
            metrics=second_metrics,
            base_action=base_actions[1],
        )
        first_result.update({**common, "joint_partner_intersection_id": second_id})
        second_result.update({**common, "joint_partner_intersection_id": first_id})
        return {first_id: first_result, second_id: second_result}

    def relabel(self, input_path: str, output_path: str) -> str:
        """Create train-compatible labels with joint labels on selected edges."""
        records: List[Dict[str, Any]] = []
        snapshot_groups: Dict[str, List[int]] = defaultdict(list)
        with open(input_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                records.append(record)
                snapshot_path = str(record.get("snapshot_path", ""))
                if snapshot_path and os.path.exists(snapshot_path):
                    snapshot_groups[snapshot_path].append(len(records) - 1)
        if not snapshot_groups:
            raise RuntimeError("No records with existing snapshot_path values were found.")

        all_snapshot_items = sorted(snapshot_groups.items())
        selected_snapshot_items = (
            all_snapshot_items[: self.max_snapshots] if self.max_snapshots else all_snapshot_items
        )
        max_time = max(
            self._snapshot_time([records[index] for index in indices])
            for _, indices in selected_snapshot_items
        )
        self._ensure_runner(max_time)
        coverage: Dict[str, Any] = {
            "total_records": len(records),
            "total_snapshots": len(all_snapshot_items),
            "selected_snapshots": len(selected_snapshot_items),
            "processed_snapshots": 0,
            "skipped_records": 0,
            "candidate_rollouts": 0,
            "joint_candidate_rollouts": 0,
            "unilateral_candidate_rollouts": 0,
            "joint_pairs": 0,
            "joint_pair_selection": self.joint_pair_selection,
            "max_joint_pairs_per_snapshot": self.max_joint_pairs_per_snapshot,
            "joint_marginalization": self.joint_marginalization,
            "joint_only": self.joint_only,
            "cost_scope": self.cost_scope,
            "neighbor_mode": self.neighbor_mode,
            "source_policy_checkpoint": self.source_policy_ckpt,
            "sim_horizon": self.sim_horizon,
            "min_action_time": self.min_action_time,
            "traffic_series_interval": self.traffic_series_interval,
        }
        selected_snapshot_paths = {snapshot_path for snapshot_path, _ in selected_snapshot_items}
        for snapshot_path, indices in all_snapshot_items:
            snapshot_records = [records[index] for index in indices]
            if snapshot_path not in selected_snapshot_paths:
                for index in indices:
                    records[index]["relabel_method"] = "joint_counterfactual_not_selected"
                coverage["skipped_records"] += len(indices)
                continue

            phases = load_snapshot_phase_indices(snapshot_path)
            snapshot_time = self._snapshot_time(snapshot_records)
            pressures: Dict[str, float] = {}
            eligible_ids: List[str] = []
            for record in snapshot_records:
                intersection_id = str(record.get("intersection_id", ""))
                if not intersection_id or intersection_id not in phases:
                    continue
                if self.target_intersections and intersection_id not in self.target_intersections:
                    continue
                if intersection_id not in pressures:
                    eligible_ids.append(intersection_id)
                    pressures[intersection_id] = 0.0
                pressures[intersection_id] = max(pressures[intersection_id], _record_pressure(record))
            selected_pairs = select_joint_pairs(
                eligible_ids,
                pressures=pressures,
                max_pairs=self.max_joint_pairs_per_snapshot,
                selection=self.joint_pair_selection,
            )
            results: Dict[str, Dict[str, Any]] = {}
            pair_endpoints: set[str] = set()
            for pair in selected_pairs:
                results.update(
                    self._label_joint_pair(
                        snapshot_path=snapshot_path,
                        snapshot_records=snapshot_records,
                        pair=pair,
                        snapshot_time_s=snapshot_time,
                    )
                )
                pair_endpoints.update(pair)
                coverage["candidate_rollouts"] += len(ACTION_LIST) ** 2
                coverage["joint_candidate_rollouts"] += len(ACTION_LIST) ** 2
                coverage["joint_pairs"] += 1
            if not self.joint_only:
                for intersection_id in eligible_ids:
                    if intersection_id in pair_endpoints:
                        continue
                    results[intersection_id] = self._label_unilateral(
                        snapshot_path=snapshot_path,
                        snapshot_records=snapshot_records,
                        target_intersection_id=intersection_id,
                        snapshot_time_s=snapshot_time,
                    )
                    coverage["candidate_rollouts"] += len(ACTION_LIST)
                    coverage["unilateral_candidate_rollouts"] += len(ACTION_LIST)

            for index in indices:
                record = records[index]
                intersection_id = str(record.get("intersection_id", ""))
                result = results.get(intersection_id)
                if result is None:
                    is_unselected_target = bool(
                        self.target_intersections and intersection_id not in self.target_intersections
                    )
                    record.update(
                        {
                            "relabel_method": "joint_counterfactual_not_selected",
                            "candidate_action_metrics": {},
                            "candidate_action_waits": {},
                            "expert_action": "" if self.joint_only or is_unselected_target else record.get("policy_action", ACTION_LIST[0]),
                            "expert_action_margin": 0.0,
                            "base_action": record.get("policy_action", ACTION_LIST[0]),
                            "pareto_feasible": {},
                            "constraint_violations": {},
                            "joint_counterfactual": False,
                        }
                    )
                    coverage["skipped_records"] += 1
                    continue
                record.update(result)
                record.update(
                    {
                        "relabel_method": (
                            "simexpert_joint_counterfactual"
                            if result.get("joint_counterfactual")
                            else f"simexpert_policy_{self.neighbor_mode}_fallback"
                        ),
                        "counterfactual_mode": (
                            f"joint_policy_{self.neighbor_mode}_neighbors"
                            if result.get("joint_counterfactual")
                            else f"policy_{self.neighbor_mode}_neighbors"
                        ),
                        "neighbor_mode": self.neighbor_mode,
                        "source_policy_checkpoint": self.source_policy_ckpt,
                        "sim_horizon": self.sim_horizon,
                        "response_min_action_time": self.min_action_time,
                        "response_traffic_series_interval": self.traffic_series_interval,
                        "cost_scope": self.cost_scope,
                        "cost_weights": {
                            "att": self.cost_att_weight,
                            "aql": self.cost_aql_weight,
                            "awt": self.cost_awt_weight,
                            "spillback": self.cost_spillback_weight,
                        },
                    }
                )
            coverage["processed_snapshots"] += 1
            if coverage["processed_snapshots"] % 5 == 0:
                print(
                    "[JointCounterfactualSimExpert] processed "
                    f"{coverage['processed_snapshots']}/{coverage['selected_snapshots']} snapshots",
                    flush=True,
                )

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        coverage_path = output_path.replace(".jsonl", "_coverage.json")
        with open(coverage_path, "w", encoding="utf-8") as handle:
            json.dump(coverage, handle, indent=2)
        print(f"[JointCounterfactualSimExpert] wrote {len(records)} records to {output_path}", flush=True)
        return output_path


def add_joint_args(parser: argparse.ArgumentParser) -> None:
    """Add joint-edge controls after the shared responsive-label arguments."""
    parser.add_argument(
        "--max_joint_pairs_per_snapshot",
        type=int,
        default=1,
        help="Maximum non-overlapping adjacent pairs per snapshot (default: 1).",
    )
    parser.add_argument(
        "--joint_pair_selection",
        choices=["max_pressure", "lexicographic"],
        default="max_pressure",
        help="How to select non-overlapping adjacent pairs when several exist.",
    )
    parser.add_argument(
        "--joint_marginalization",
        choices=["min"],
        default="min",
        help="Projection of the 4x4 pair cost table into endpoint action costs.",
    )
    parser.add_argument(
        "--joint_only",
        action="store_true",
        help="Leave records outside selected pairs unlabeled instead of using unilateral fallback labels.",
    )


def relabel_joint_from_args(args: argparse.Namespace) -> str:
    """Instantiate the joint teacher from the shared CLI namespace."""
    relabeler = JointCounterfactualSimExpertRelabeler(
        roadnet_path=os.path.abspath(args.roadnet),
        traffic_file=os.path.abspath(args.traffic_file),
        work_dir=os.path.abspath(args.work_dir),
        num_rows=args.num_rows,
        num_cols=args.num_cols,
        source_policy_ckpt=args.source_policy_ckpt,
        llm_path=args.llm_path,
        dataset=args.dataset,
        sim_horizon=args.sim_horizon,
        min_action_time=args.min_action_time,
        traffic_series_interval=args.traffic_series_interval,
        inference_batch_size=args.inference_batch_size,
        device=args.device,
        bf16=args.bf16 or not args.fp16,
        fp16=args.fp16,
        seq_len=args.seq_len,
        image_size=args.image_size,
        periodicity=args.periodicity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens,
        prefix_mode=args.prefix_mode,
        max_length=args.max_length,
        cost_scope=args.cost_scope,
        max_snapshots=args.max_snapshots,
        neighbor_mode=args.neighbor_mode,
        target_intersections=args.target_intersections,
        cost_att_weight=args.cost_att_weight,
        cost_aql_weight=args.cost_aql_weight,
        cost_awt_weight=args.cost_awt_weight,
        cost_spillback_weight=args.cost_spillback_weight,
        awt_constraint_epsilon=args.awt_constraint_epsilon,
        spillback_constraint_epsilon=args.spillback_constraint_epsilon,
        enable_pareto_filter=not args.disable_pareto_filter,
        enable_constraint_filter=not args.disable_constraint_filter,
        max_joint_pairs_per_snapshot=args.max_joint_pairs_per_snapshot,
        joint_pair_selection=args.joint_pair_selection,
        joint_marginalization=args.joint_marginalization,
        joint_only=args.joint_only,
    )
    return relabeler.relabel(os.path.abspath(args.input), os.path.abspath(args.output))
