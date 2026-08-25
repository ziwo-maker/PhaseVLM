#!/usr/bin/env python3
"""Standalone SimExpert relabeling: counterfactual simulation per candidate action.

Reads collected states from Stage 3 state collection (run_projection_online.py
--collect_states_output), loads each unique CityFlow snapshot, and runs
per-candidate counterfactual simulations to produce:

  candidate_action_metrics  – {action: {att, aql, awt, spillback, ...}}
  candidate_action_waits    – {action: composite_score}
  base_action               – action inferred from snapshot phase state
  expert_action             – best feasible action per SimExpert scoring
  expert_action_margin      – score gap between best and second-best
  pareto_feasible           – {action: bool}
  constraint_violations     – {action: [violation_labels]}

Metric semantics (cost PROXIES, not the online evaluator's ATT/AWT):
  att  = mean number of vehicles on the intersection's entering lanes over
         the horizon (a throughput proxy, NOT average travel time)
  aql  = mean queued (waiting) vehicles on entering lanes (queue proxy)
  awt  = accumulated queued vehicle-seconds (sum, NOT mean waiting time)
  spillback = exiting-lane occupancy + weighted blocked-lane ratio
Online evaluation (run_projection_online.py) reports the true ATT/AQL/AWT;
these keys only rank candidate actions for relabeling/training.

Counterfactual rollouts emulate the canonical evaluator's phase semantics:
a phase switch passes through phase 0 (all-yellow) for YELLOW_TIME_S seconds
before the target green takes effect.

Requires a Python environment with cityflow installed (e.g. conda env hnr5090).
"""

from __future__ import annotations

import sys
# Prevent stale .pyc from masking source edits
sys.dont_write_bytecode = True

import argparse
import json
import os
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ACTION_LIST: tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")

# Phase index in CityFlow → action index in ACTION_LIST.
# CityFlow phase indices: 1=WT_ET, 2=NT_ST, 3=WL_EL, 4=NL_SL
# ACTION_LIST: ETWT(0), NTST(1), ELWL(2), NLSL(3)
PHASE_TO_ACTION: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3}
ACTION_TO_PHASE: Dict[int, int] = {0: 1, 1: 2, 2: 3, 3: 4}

SPILLBACK_JAM_SPACING_M = 7.5
SPILLBACK_BLOCK_THRESHOLD = 0.8
SPILLBACK_BLOCK_WEIGHT = 0.5

# Canonical evaluator semantics (utils/config.py YELLOW_TIME): a phase switch
# passes through phase 0 (all-yellow) for 3 seconds before the target green
# takes effect. Counterfactual rollouts must emulate this or switched actions
# get systematically optimistic costs.
YELLOW_TIME_S = 3


def _candidate_phase_setpoints(
    base_phase: int, target_phase: int, yellow_time: int = YELLOW_TIME_S
) -> List[Tuple[int, int]]:
    """Phase set-points (second, phase_idx) for the target intersection.

    Canonical evaluator semantics: a phase switch passes through phase 0
    (all-yellow) for yellow_time seconds before the target green takes
    effect; keeping the current phase triggers no transition.
    """
    if int(target_phase) == int(base_phase):
        return [(0, int(target_phase))]
    return [(0, 0), (int(yellow_time), int(target_phase))]

# NOTE: rows × cols of REAL (non-virtual) intersections in each roadnet.
# The "3_4" directory name means 3 COLUMNS × 4 ROWS (roadnet_3_4.json has
# intersections intersection_1_1 .. intersection_4_3).
NETWORK_GRIDS: Dict[str, Tuple[int, int]] = {
    "jinan": (4, 3),
    "hangzhou": (4, 4),
    "jinan_3_4": (4, 3),
    "hangzhou_4_4": (4, 4),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_network_grid(dataset: str) -> Tuple[int, int]:
    dataset_lower = dataset.lower().replace("-", "_")
    for key, grid in NETWORK_GRIDS.items():
        if key in dataset_lower:
            return grid
    raise ValueError(f"Unknown dataset grid: {dataset}. Known: {list(NETWORK_GRIDS)}")


def _action_name(phase_index: int) -> str:
    """Convert CityFlow phase index (1-4) to ACTION_LIST name."""
    return ACTION_LIST[PHASE_TO_ACTION.get(phase_index, 0)]


def _phase_index(action_name: str) -> int:
    """Convert ACTION_LIST name to CityFlow phase index (1-4)."""
    for i, name in enumerate(ACTION_LIST):
        if name == action_name:
            return ACTION_TO_PHASE[i]
    return 1


def _intersection_name(row: int, col: int) -> str:
    return f"intersection_{row}_{col}"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _metric_mean(values: List[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _metric_sum(values: List[float]) -> float:
    return float(np.sum(values)) if values else 0.0


def _compute_road_length(road: Dict) -> float:
    """Compute road length from its geometry points (Euclidean distance)."""
    pts = road.get("points", [])
    if not pts or len(pts) < 2:
        # Fallback: try explicit length field
        explicit = road.get("length")
        if explicit is not None:
            return float(explicit)
        return 200.0  # conservative default
    total = 0.0
    for i in range(len(pts) - 1):
        dx = float(pts[i + 1].get("x", 0.0)) - float(pts[i].get("x", 0.0))
        dy = float(pts[i + 1].get("y", 0.0)) - float(pts[i].get("y", 0.0))
        total += float(np.sqrt(dx * dx + dy * dy))
    return max(1.0, total)


# ---------------------------------------------------------------------------
# Roadnet parsing
# ---------------------------------------------------------------------------

def load_roadnet(roadnet_path: str) -> Dict[str, Any]:
    with open(roadnet_path, "r") as f:
        return json.load(f)


def build_intersection_lane_map(
    roadnet: Dict, num_rows: int, num_cols: int
) -> Dict[str, Dict[str, List[str]]]:
    """Build per-intersection entering/exiting lane lists from roadnet.

    Returns:
        {intersection_name: {"entering": [...], "exiting": [...], "lane_length": {...}}}
    """
    inter_nodes = roadnet.get("intersections", []) or roadnet.get("nodes", [])
    inter_ids = {node.get("id", ""): node for node in inter_nodes if "intersection" in str(node.get("id", ""))}

    # Build road → (from, to) mapping
    roads = roadnet.get("roads", []) or roadnet.get("edges", [])
    road_map: Dict[str, Dict] = {}
    for road in roads:
        rid = road.get("id", "")
        if rid:
            road_map[rid] = road

    # Build intersection lane map
    lane_map: Dict[str, Dict[str, List[str]]] = {}
    for row in range(1, num_rows + 1):
        for col in range(1, num_cols + 1):
            iname = _intersection_name(row, col)
            entering = []
            exiting = []
            lane_length: Dict[str, float] = {}

            # Find roads ending at this intersection (entering)
            for rid, road in road_map.items():
                end_node = str(road.get("endIntersection", "") or road.get("to", ""))
                start_node = str(road.get("startIntersection", "") or road.get("from", ""))
                if end_node == iname:
                    # Road goes INTO this intersection
                    road_lanes = road.get("lanes", [])
                    road_len = _compute_road_length(road)
                    for li, lane in enumerate(road_lanes):
                        lane_id = f"{rid}_{li}"
                        entering.append(lane_id)
                        lane_length[lane_id] = road_len
                if start_node == iname:
                    # Road goes OUT of this intersection
                    road_lanes = road.get("lanes", [])
                    road_len = _compute_road_length(road)
                    for li in range(len(road_lanes)):
                        lane_id = f"{rid}_{li}"
                        exiting.append(lane_id)
                        lane_length[lane_id] = road_len

            lane_map[iname] = {
                "entering": entering,
                "exiting": exiting,
                "lane_length": lane_length,
            }

    return lane_map


# ---------------------------------------------------------------------------
# SimExpert Counterfactual Simulation
# ---------------------------------------------------------------------------

class SimExpertRelabeler:
    """Runs counterfactual simulations on CityFlow snapshots."""

    def __init__(
        self,
        roadnet_path: str,
        traffic_file: str,
        work_dir: str,
        num_rows: int,
        num_cols: int,
        sim_horizon: int = 30,
        sim_interval: float = 1.0,
        cost_att_weight: float = 1.0,
        cost_aql_weight: float = 1.0,
        cost_awt_weight: float = 1.0,
        cost_spillback_weight: float = 0.2,
        awt_constraint_epsilon: float = 5.0,
        spillback_constraint_epsilon: float = 0.05,
        enable_pareto_filter: bool = True,
        enable_constraint_filter: bool = True,
    ):
        self.roadnet_path = roadnet_path
        self.traffic_file = traffic_file
        self.work_dir = work_dir
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.sim_horizon = sim_horizon
        self.sim_interval = sim_interval
        self.cost_att_weight = cost_att_weight
        self.cost_aql_weight = cost_aql_weight
        self.cost_awt_weight = cost_awt_weight
        self.cost_spillback_weight = cost_spillback_weight
        self.awt_constraint_epsilon = awt_constraint_epsilon
        self.spillback_constraint_epsilon = spillback_constraint_epsilon
        self.enable_pareto_filter = enable_pareto_filter
        self.enable_constraint_filter = enable_constraint_filter

        self._roadnet_data: Optional[Dict] = None
        self._lane_map: Optional[Dict] = None
        self._eng = None
        self._engine_initialized = False

    # ---- Engine lifecycle ----

    def _ensure_engine(self) -> None:
        if self._engine_initialized:
            return
        import cityflow as engine

        os.makedirs(self.work_dir, exist_ok=True)

        # Copy roadnet and traffic file to work dir
        import shutil
        roadnet_basename = os.path.basename(self.roadnet_path)
        traffic_basename = os.path.basename(self.traffic_file)
        shutil.copy2(self.roadnet_path, os.path.join(self.work_dir, roadnet_basename))
        shutil.copy2(self.traffic_file, os.path.join(self.work_dir, traffic_basename))

        config = {
            "interval": self.sim_interval,
            "seed": 0,
            "laneChange": True,
            "dir": self.work_dir + "/",
            "roadnetFile": roadnet_basename,
            "flowFile": traffic_basename,
            "rlTrafficLight": True,
            "saveReplay": False,
            "roadnetLogFile": "roadnetLog.json",
            "replayLogFile": "replayLog.txt",
        }
        config_path = os.path.join(self.work_dir, "cityflow.config")
        with open(config_path, "w") as f:
            json.dump(config, f)

        self._eng = engine.Engine(config_path, thread_num=1)
        # Pin the engine RNG so counterfactual rollouts are bitwise
        # deterministic across replays. The config "seed" field controls flow
        # generation; the simulation dynamics RNG is seeded here explicitly
        # (default seed would otherwise be process-dependent, producing tiny
        # non-deterministic cost/margin differences across identical replays).
        try:
            self._eng.set_random_seed(0)
        except Exception:
            pass
        self._engine_initialized = True

        # Parse roadnet
        self._roadnet_data = load_roadnet(self.roadnet_path)
        self._lane_map = build_intersection_lane_map(
            self._roadnet_data, self.num_rows, self.num_cols
        )

    def _read_snapshot_phase(self, snapshot_path: str) -> Dict[str, int]:
        """Read curPhaseIndex for each intersection from a snapshot JSON."""
        with open(snapshot_path, "r") as f:
            data = json.load(f)
        tl_data = data.get("trafficLights", {})
        phases: Dict[str, int] = {}
        for name, info in tl_data.items():
            if "intersection" in name:
                phases[name] = int(info.get("curPhaseIndex", 0) or 0)
        return phases

    def _simulate_candidate(
        self,
        inter_name: str,
        target_phase_idx: int,
        base_actions: Dict[str, int],
    ) -> Dict[str, float]:
        """Run counterfactual simulation: set inter to target_phase, others to base, simulate H steps.

        Returns metrics: att, aql, awt, spillback, spillback_occupancy, spillback_blocked_ratio.
        """
        eng = self._eng
        lm = self._lane_map
        if lm is None or inter_name not in lm:
            return {"att": 0.0, "aql": 0.0, "awt": 0.0, "spillback": 0.0,
                    "spillback_occupancy": 0.0, "spillback_blocked_ratio": 0.0}

        lane_info = lm[inter_name]
        entering_lanes = lane_info["entering"]
        exiting_lanes = lane_info["exiting"]
        lane_lengths = lane_info["lane_length"]

        # Apply actions with yellow-transition emulation (canonical env
        # semantics: a phase switch passes through phase 0 for YELLOW_TIME_S
        # seconds). Base intersections keep their snapshot phase (frozen, as
        # before); the target intersection transitions yellow→target when the
        # candidate differs from its current phase.
        base_phase = int(base_actions.get(inter_name, 1))
        setpoints = _candidate_phase_setpoints(base_phase, int(target_phase_idx))
        for name, base_p in base_actions.items():
            if name == inter_name:
                continue
            eng.set_tl_phase(name, int(base_p))
        eng.set_tl_phase(inter_name, setpoints[0][1])

        occupancy_trace: List[float] = []
        queue_trace: List[float] = []
        wait_trace: List[float] = []
        spillback_occupancy_trace: List[float] = []
        spillback_blocked_trace: List[float] = []

        interval = max(0.01, self.sim_interval)
        substeps = max(1, int(1.0 / interval))

        for second in range(self.sim_horizon):
            for t_set, phase in setpoints[1:]:
                if second == t_set:
                    eng.set_tl_phase(inter_name, phase)
            for _ in range(substeps):
                eng.next_step()

            lane_vehicle_counts = eng.get_lane_vehicle_count()
            lane_waiting_counts = eng.get_lane_waiting_vehicle_count()

            queue_total = sum(
                _safe_float(lane_waiting_counts.get(lane, 0)) for lane in entering_lanes
            )
            entering_total = sum(
                _safe_float(lane_vehicle_counts.get(lane, 0)) for lane in entering_lanes
            )
            queue_trace.append(queue_total)
            wait_trace.append(queue_total * interval)
            occupancy_trace.append(entering_total)

            exiting_occs = []
            exiting_blocked = []
            for lane in exiting_lanes:
                lane_length = _safe_float(lane_lengths.get(lane, 200.0), 200.0)
                jam_capacity = max(1.0, lane_length / SPILLBACK_JAM_SPACING_M)
                lane_count = _safe_float(lane_vehicle_counts.get(lane, 0))
                occupancy = min(1.0, lane_count / jam_capacity)
                lane_waiting = _safe_float(lane_waiting_counts.get(lane, 0))
                exiting_occs.append(occupancy)
                exiting_blocked.append(
                    1.0 if occupancy >= SPILLBACK_BLOCK_THRESHOLD and lane_waiting > 0 else 0.0
                )

            spillback_occupancy_trace.append(_metric_mean(exiting_occs) if exiting_occs else 0.0)
            spillback_blocked_trace.append(_metric_mean(exiting_blocked) if exiting_blocked else 0.0)

        spillback_occ = _metric_mean(spillback_occupancy_trace)
        spillback_blocked = _metric_mean(spillback_blocked_trace)

        return {
            "att": _metric_mean(occupancy_trace),
            "aql": _metric_mean(queue_trace),
            "awt": _metric_sum(wait_trace),  # accumulated vehicle-seconds, NOT mean
            "spillback": spillback_occ + SPILLBACK_BLOCK_WEIGHT * spillback_blocked,
            "spillback_occupancy": spillback_occ,
            "spillback_blocked_ratio": spillback_blocked,
        }

    def _minmax_normalize(
        self, values: Dict[str, float]
    ) -> Dict[str, float]:
        if len(values) < 2:
            return {k: 0.0 for k in values}
        vv = list(values.values())
        lo, hi = min(vv), max(vv)
        spread = hi - lo
        if spread <= 1e-8:
            return {k: 0.0 for k in values}
        return {k: (v - lo) / spread for k, v in values.items()}

    def _score_candidates(
        self,
        raw_metrics: Dict[str, Dict[str, float]],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """Score candidates by min-max normalizing each metric then weighting.

        Returns (composite_scores, enriched_metrics).
        """
        action_names = list(raw_metrics.keys())
        if not action_names:
            return {}, {}

        raw_by_metric: Dict[str, Dict[str, float]] = defaultdict(dict)
        for metric in ["att", "aql", "awt", "spillback"]:
            for action in action_names:
                raw_by_metric[metric][action] = raw_metrics[action].get(metric, 0.0)

        norm_by_metric = {}
        weights = {
            "att": self.cost_att_weight,
            "aql": self.cost_aql_weight,
            "awt": self.cost_awt_weight,
            "spillback": self.cost_spillback_weight,
        }
        for metric in ["att", "aql", "awt", "spillback"]:
            norm_by_metric[metric] = self._minmax_normalize(raw_by_metric[metric])

        composite: Dict[str, float] = {}
        for action in action_names:
            composite[action] = sum(
                weights[m] * norm_by_metric[m][action]
                for m in ["att", "aql", "awt", "spillback"]
            )

        enriched: Dict[str, Dict[str, float]] = {}
        for action in action_names:
            enriched[action] = {
                **{m: raw_metrics[action].get(m, 0.0) for m in ["att", "aql", "awt", "spillback"]},
                "spillback_occupancy": raw_metrics[action].get("spillback_occupancy", 0.0),
                "spillback_blocked_ratio": raw_metrics[action].get("spillback_blocked_ratio", 0.0),
                "score": composite[action],
                "score_mode": "composite_spillback",
                "score_norm": "minmax",
                "weight_att": weights["att"],
                "weight_aql": weights["aql"],
                "weight_awt": weights["awt"],
                "weight_spillback": weights["spillback"],
                **{f"norm_{m}": norm_by_metric[m][action] for m in ["att", "aql", "awt", "spillback"]},
            }

        return composite, enriched

    def _check_pareto_feasibility(
        self,
        enriched_metrics: Dict[str, Dict[str, float]],
        base_action_name: str,
    ) -> Tuple[Dict[str, bool], Dict[str, List[str]], Dict[str, bool]]:
        """Determine which actions are Pareto-optimal and constraint-satisfying.

        Returns (pareto_feasible, constraint_violations, dominated).
        """
        action_names = list(enriched_metrics.keys())
        pareto_feasible: Dict[str, bool] = {a: True for a in action_names}
        constraint_violations: Dict[str, List[str]] = defaultdict(list)
        dominated: Dict[str, bool] = {a: False for a in action_names}

        # ---- Constraint filtering ----
        if self.enable_constraint_filter and base_action_name in enriched_metrics:
            base_metrics = enriched_metrics[base_action_name]
            base_awt = base_metrics.get("awt", 0.0)
            base_spill = base_metrics.get("spillback", 0.0)

            for action in action_names:
                m = enriched_metrics[action]
                awt = m.get("awt", 0.0)
                spill = m.get("spillback", 0.0)
                if awt > base_awt + self.awt_constraint_epsilon:
                    constraint_violations[action].append("awt_exceeded")
                    pareto_feasible[action] = False
                if spill > base_spill + self.spillback_constraint_epsilon:
                    constraint_violations[action].append("spillback_exceeded")
                    pareto_feasible[action] = False

        # ---- Pareto dominance filtering ----
        if self.enable_pareto_filter:
            metrics_to_check = ["att", "aql", "awt", "spillback"]
            for i, a_i in enumerate(action_names):
                if not pareto_feasible[a_i]:
                    continue
                for j, a_j in enumerate(action_names):
                    if i == j:
                        continue
                    m_i = enriched_metrics[a_i]
                    m_j = enriched_metrics[a_j]
                    # a_j dominates a_i if it's no worse on all metrics and strictly better on at least one
                    no_worse = all(m_j.get(m, float("inf")) <= m_i.get(m, float("inf")) for m in metrics_to_check)
                    strictly_better = any(m_j.get(m, float("inf")) < m_i.get(m, float("inf")) - 1e-8 for m in metrics_to_check)
                    if no_worse and strictly_better:
                        dominated[a_i] = True
                        pareto_feasible[a_i] = False
                        break

        return pareto_feasible, constraint_violations, dominated

    # ---- Main entry point ----

    def relabel(
        self,
        input_path: str,
        output_path: str,
    ) -> str:
        """Run SimExpert relabeling on collected states.

        Returns output_path on success.
        """
        import cityflow as engine

        self._ensure_engine()
        eng = self._eng
        lm = self._lane_map

        assert eng is not None
        assert lm is not None

        # ---- Phase 1: Read and group records ----
        print(f"[SimExpert] Reading collected states from {input_path}")
        records: List[Dict[str, Any]] = []
        snapshot_groups: Dict[str, List[int]] = defaultdict(list)

        with open(input_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records.append(rec)
                sp = rec.get("snapshot_path", "")
                if sp and os.path.exists(sp):
                    snapshot_groups[sp].append(len(records) - 1)

        print(f"[SimExpert] Loaded {len(records)} records across {len(snapshot_groups)} unique snapshots")

        if not snapshot_groups:
            raise RuntimeError(
                "No valid snapshot paths found in collected states. "
                "Ensure state collection was run with record_snapshots=True."
            )

        # ---- Phase 2: Process each snapshot ----
        total_intersections = self.num_rows * self.num_cols
        processed_snapshots = 0
        coverage_stats = {
            "total_records": len(records),
            "total_snapshots": len(snapshot_groups),
            "candidates_with_metrics": 0,
            "candidates_with_2plus_valid": 0,
            "constraint_violations_total": 0,
            "pareto_dominated_total": 0,
        }
        skipped_snapshots: List[str] = []
        max_skip = max(10, int(len(snapshot_groups) * 0.50))  # tolerate up to 50% corrupted

        for sp_path, record_indices in sorted(snapshot_groups.items()):
            try:
                # Load snapshot into engine
                eng.load_from_file(sp_path)

                # Read phase state from snapshot JSON
                snapshot_phases = self._read_snapshot_phase(sp_path)

                # Determine base actions for all intersections
                base_actions: Dict[str, int] = {}
                for row in range(1, self.num_rows + 1):
                    for col in range(1, self.num_cols + 1):
                        iname = _intersection_name(row, col)
                        cur_phase = snapshot_phases.get(iname, 0)
                        if 1 <= cur_phase <= 4:
                            base_actions[iname] = cur_phase
                        else:
                            base_actions[iname] = 1  # default to phase 1

                # ---- For each intersection, simulate all candidates ----
                # We need to process unique (snapshot, intersection) pairs
                # But since records are per-intersection, we process all intersections
                # and distribute results to the corresponding records.

                inter_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
                inter_scores: Dict[str, Dict[str, float]] = {}
                inter_pareto: Dict[str, Dict[str, bool]] = {}
                inter_violations: Dict[str, Dict[str, List[str]]] = {}
                inter_best: Dict[str, str] = {}
                inter_margins: Dict[str, float] = {}

                for row in range(1, self.num_rows + 1):
                    for col in range(1, self.num_cols + 1):
                        iname = _intersection_name(row, col)
                        if iname not in lm:
                            continue

                        base_phase = base_actions[iname]
                        base_action_name = _action_name(base_phase)

                        # Run counterfactual for each candidate
                        raw_metrics: Dict[str, Dict[str, float]] = {}
                        for cand_phase in range(1, 5):
                            cand_name = _action_name(cand_phase)
                            # Reload snapshot before each candidate
                            eng.load_from_file(sp_path)
                            metrics = self._simulate_candidate(
                                iname, cand_phase, base_actions
                            )
                            raw_metrics[cand_name] = metrics

                        # Score candidates
                        scores, enriched = self._score_candidates(raw_metrics)

                        # Pareto feasibility
                        pareto_feasible, constraint_violations, dominated = (
                            self._check_pareto_feasibility(enriched, base_action_name)
                        )

                        # Best feasible action (lowest composite score among feasible)
                        # Tie-break: prefer base_action when scores are equal
                        best_action = base_action_name
                        best_score = float("inf")
                        for action in ACTION_LIST:
                            if action in scores and pareto_feasible.get(action, True):
                                s = scores[action]
                                if s < best_score - 1e-9:
                                    best_score = s
                                    best_action = action
                                elif abs(s - best_score) < 1e-9 and action == base_action_name:
                                    best_action = action  # tie-break to base

                        # Margin: gap between best and second-best feasible scores
                        feasible_scores = sorted(
                            [scores[a] for a in ACTION_LIST if a in scores and pareto_feasible.get(a, True)]
                        )
                        margin = feasible_scores[1] - feasible_scores[0] if len(feasible_scores) >= 2 else 0.0

                        inter_metrics[iname] = enriched
                        inter_scores[iname] = scores
                        inter_pareto[iname] = pareto_feasible
                        inter_violations[iname] = dict(constraint_violations)
                        inter_best[iname] = best_action
                        inter_margins[iname] = margin

                        # Coverage stats
                        if enriched:
                            coverage_stats["candidates_with_metrics"] += 1
                            n_feasible = sum(1 for a in ACTION_LIST if pareto_feasible.get(a, True))
                            if n_feasible >= 2:
                                coverage_stats["candidates_with_2plus_valid"] += 1
                            coverage_stats["constraint_violations_total"] += sum(
                                len(v) for v in constraint_violations.values()
                            )
                            coverage_stats["pareto_dominated_total"] += sum(
                                1 for d in dominated.values() if d
                            )

                # ---- Distribute results to records ----
                for idx in record_indices:
                    rec = records[idx]
                    inter_id = str(rec.get("intersection_id", ""))

                    if inter_id in inter_metrics:
                        rec["candidate_action_metrics"] = inter_metrics[inter_id]
                        # Convert scores to "waits" (legacy naming: lower = better)
                        rec["candidate_action_waits"] = inter_scores[inter_id]
                        rec["expert_action"] = inter_best[inter_id]
                        rec["expert_action_margin"] = inter_margins[inter_id]
                        rec["base_action"] = _action_name(base_actions.get(inter_id, 1))
                        rec["relabel_method"] = "simexpert_counterfactual"
                        rec["pareto_feasible"] = inter_pareto.get(inter_id, {})
                        rec["constraint_violations"] = inter_violations.get(inter_id, {})
                        rec["sim_horizon"] = self.sim_horizon
                        rec["cost_weights"] = {
                            "att": self.cost_att_weight,
                            "aql": self.cost_aql_weight,
                            "awt": self.cost_awt_weight,
                            "spillback": self.cost_spillback_weight,
                        }
                    else:
                        rec["relabel_method"] = "simexpert_no_lane_map"
                        rec["candidate_action_metrics"] = {}
                        rec["candidate_action_waits"] = {}
                        rec["expert_action"] = rec.get("policy_action", ACTION_LIST[0])
                        rec["expert_action_margin"] = 0.0
                        rec["base_action"] = rec.get("policy_action", ACTION_LIST[0])
                        rec["pareto_feasible"] = {}
                        rec["constraint_violations"] = {}

                processed_snapshots += 1
                if processed_snapshots % 10 == 0:
                    print(f"[SimExpert] Processed {processed_snapshots}/{len(snapshot_groups)} snapshots")

            except (json.JSONDecodeError, RuntimeError) as e:
                # Corrupted snapshot (truncated file, parse error) — skip with warning
                msg = str(e)[:120]
                print(f"[SimExpert] WARNING: Skipping corrupted snapshot {sp_path}: {msg}")
                skipped_snapshots.append(sp_path)
                for idx in record_indices:
                    rec = records[idx]
                    rec["relabel_method"] = "simexpert_skipped_corrupted_snapshot"
                    rec["candidate_action_metrics"] = {}
                    rec["candidate_action_waits"] = {}
                    rec["expert_action"] = rec.get("policy_action", ACTION_LIST[0])
                    rec["expert_action_margin"] = 0.0
                    rec["base_action"] = rec.get("policy_action", ACTION_LIST[0])
                    rec["pareto_feasible"] = {}
                    rec["constraint_violations"] = {}
                continue
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"SimExpert relabeling FAILED on snapshot {sp_path}: {e}\n"
                    f"This is a FATAL error (not a JSON parse issue). "
                    f"Fix the underlying problem or use --allow_ruleonly_fallback for smoke tests."
                ) from e

        # ---- Validate skipped snapshots ----
        if len(skipped_snapshots) > max_skip:
            print(
                f"[SimExpert] WARNING: {len(skipped_snapshots)} corrupted snapshots "
                f"(max allowed: {max_skip}). Saving partial results with policy-action fallback "
                f"for skipped snapshots. First 5: {skipped_snapshots[:5]}"
            )
            # DO NOT raise — save what we have. Records from skipped snapshots
            # already have policy_action as expert_action (set in the except block).

        # ---- Phase 3: Write output ----
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # ---- Coverage report ----
        print(f"\n[SimExpert] Relabeling complete: {len(records)} records -> {output_path}")
        print(f"[SimExpert] Coverage stats:")
        print(f"  Total records:           {coverage_stats['total_records']}")
        print(f"  Unique snapshots:        {coverage_stats['total_snapshots']}")
        print(f"  Skipped (corrupted):     {len(skipped_snapshots)}")
        print(f"  Candidates with metrics: {coverage_stats['candidates_with_metrics']}")
        print(f"  Candidates 2+ feasible:  {coverage_stats['candidates_with_2plus_valid']}")
        print(f"  Constraint violations:   {coverage_stats['constraint_violations_total']}")
        print(f"  Pareto dominated actions:{coverage_stats['pareto_dominated_total']}")

        # Save coverage stats
        stats_path = output_path.replace(".jsonl", "_coverage.json")
        with open(stats_path, "w") as f:
            json.dump(coverage_stats, f, indent=2)
        print(f"[SimExpert] Coverage stats saved to {stats_path}")

        return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SimExpert counterfactual relabeling for Stage 3 cross-network adaptation"
    )
    parser.add_argument("--input", required=True, help="Collected states JSONL (from run_projection_online)")
    parser.add_argument("--output", required=True, help="Output relabeled JSONL path")
    parser.add_argument("--roadnet", required=True, help="Path to target network roadnet JSON")
    parser.add_argument("--traffic_file", required=True, help="Path to traffic flow JSON")
    parser.add_argument("--dataset", required=True, choices=["jinan", "hangzhou", "jinan_3_4", "hangzhou_4_4"],
                        help="Target dataset name (determines grid size)")
    parser.add_argument("--work_dir", default=None, help="Working directory for cityflow (default: auto)")
    parser.add_argument("--sim_horizon", type=int, default=30, help="Counterfactual simulation horizon (seconds)")
    parser.add_argument("--sim_interval", type=float, default=1.0, help="Simulation step interval")
    parser.add_argument("--cost_att_weight", type=float, default=1.0)
    parser.add_argument("--cost_aql_weight", type=float, default=1.0)
    parser.add_argument("--cost_awt_weight", type=float, default=1.0,
                        help="AWT weight in composite score (default 1.0; SimExpert scoring)")
    parser.add_argument("--cost_spillback_weight", type=float, default=0.2)
    parser.add_argument("--awt_constraint_epsilon", type=float, default=5.0,
                        help="Allowed AWT slack over base action before constraint violation")
    parser.add_argument("--spillback_constraint_epsilon", type=float, default=0.05,
                        help="Allowed spillback slack before constraint violation")
    parser.add_argument("--disable_pareto_filter", action="store_true",
                        help="Disable Pareto dominance filtering")
    parser.add_argument("--disable_constraint_filter", action="store_true",
                        help="Disable AWT/spillback constraint filtering")
    parser.add_argument("--python_path", default=sys.executable,
                        help="Python executable with cityflow (for subprocess; default=current)")

    args = parser.parse_args()

    num_rows, num_cols = _resolve_network_grid(args.dataset)

    work_dir = args.work_dir or os.path.join(
        os.path.dirname(args.output) or ".", "simexpert_work"
    )

    relabeler = SimExpertRelabeler(
        roadnet_path=os.path.abspath(args.roadnet),
        traffic_file=os.path.abspath(args.traffic_file),
        work_dir=os.path.abspath(work_dir),
        num_rows=num_rows,
        num_cols=num_cols,
        sim_horizon=args.sim_horizon,
        sim_interval=args.sim_interval,
        cost_att_weight=args.cost_att_weight,
        cost_aql_weight=args.cost_aql_weight,
        cost_awt_weight=args.cost_awt_weight,
        cost_spillback_weight=args.cost_spillback_weight,
        awt_constraint_epsilon=args.awt_constraint_epsilon,
        spillback_constraint_epsilon=args.spillback_constraint_epsilon,
        enable_pareto_filter=not args.disable_pareto_filter,
        enable_constraint_filter=not args.disable_constraint_filter,
    )

    relabeler.relabel(
        input_path=os.path.abspath(args.input),
        output_path=os.path.abspath(args.output),
    )


if __name__ == "__main__":
    main()
