"""Policy-responsive SimExpert counterfactual relabeling.

The existing SimExpert oracle changes one traffic signal while every other
intersection remains fixed at its snapshot phase.  This module instead applies
one candidate action at the target intersection for one decision interval, then
lets a frozen source policy control every intersection for the remaining rollout
steps.  It deliberately emits the same training-record schema as
``scripts/relabel_simexpert.py`` so the established Stage-3 training code can
consume responsive labels without a special dataset path.

CityFlow and the projection model are imported lazily.  The sequence and phase
helpers remain importable in a lightweight test environment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from projection_train.action_space import ACTION_LIST, action_to_index
from scripts.relabel_simexpert import (
    SPILLBACK_BLOCK_THRESHOLD,
    SPILLBACK_BLOCK_WEIGHT,
    SPILLBACK_JAM_SPACING_M,
    SimExpertRelabeler,
    _action_name,
    _metric_mean,
    _metric_sum,
)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if math.isfinite(parsed) else float(default)


def normalize_sequence(
    raw_sequence: Any,
    *,
    seq_len: int,
    input_dim: int,
) -> List[List[float]]:
    """Normalize a saved canonical sequence to the projection-model shape."""
    rows: List[List[float]] = []
    if isinstance(raw_sequence, (list, tuple)):
        for raw_row in raw_sequence:
            if not isinstance(raw_row, (list, tuple)):
                continue
            row = [_finite_float(value) for value in raw_row[:input_dim]]
            if len(row) < input_dim:
                row.extend([0.0] * (input_dim - len(row)))
            rows.append(row)

    if not rows:
        rows = [[0.0] * input_dim]
    rows = rows[-seq_len:]
    if len(rows) < seq_len:
        rows = [list(rows[0]) for _ in range(seq_len - len(rows))] + rows
    return rows


def load_snapshot_phase_indices(snapshot_path: str) -> Dict[str, int]:
    """Return real (non-yellow) CityFlow phase indices recorded in a snapshot."""
    with open(snapshot_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    phases: Dict[str, int] = {}
    for name, value in (payload.get("trafficLights") or {}).items():
        if not isinstance(value, Mapping) or "intersection" not in str(name):
            continue
        phase = int(_finite_float(value.get("curPhaseIndex"), 1.0))
        phases[str(name)] = phase if 1 <= phase <= len(ACTION_LIST) else 1
    return phases


def synchronize_intersection_phase(intersection: Any, phase_index: int) -> None:
    """Synchronize CityFlowEnv's signal cache after ``engine.load_from_file``.

    ``load_from_file`` restores engine state but the Python ``Intersection``
    objects retain phase/yellow bookkeeping from the preceding candidate branch.
    Resetting that cache is necessary for the next branch to use the same yellow
    transition as a canonical online decision.
    """
    phase = int(phase_index)
    if phase < 1 or phase > len(ACTION_LIST):
        phase = 1
    intersection.current_phase_index = phase
    intersection.previous_phase_index = phase
    intersection.next_phase_to_set_index = None
    intersection.current_phase_duration = 0
    intersection.all_yellow_flag = False
    intersection.all_red_flag = False
    intersection.flicker = 0

    # Dynamic measurement caches refer to the previous candidate branch.  They
    # must not synthesize vehicle arrivals/leaves while refreshing a snapshot.
    for attr in (
        "dic_lane_vehicle_previous_step",
        "dic_lane_vehicle_previous_step_in",
        "dic_lane_waiting_vehicle_count_previous_step",
        "dic_vehicle_speed_previous_step",
        "dic_vehicle_distance_previous_step",
        "dic_lane_vehicle_current_step",
        "dic_lane_vehicle_current_step_in",
        "dic_lane_waiting_vehicle_count_current_step",
        "dic_vehicle_speed_current_step",
        "dic_vehicle_distance_current_step",
    ):
        setattr(intersection, attr, {})
    intersection.list_lane_vehicle_previous_step_in = []
    intersection.list_lane_vehicle_current_step_in = []
    intersection.dic_vehicle_arrive_leave_time = {}


def force_target_action(
    source_actions: Sequence[int],
    target_index: int,
    target_action_name: str,
) -> List[int]:
    """Keep source-policy actions everywhere except the intervened signal."""
    if target_index < 0 or target_index >= len(source_actions):
        raise IndexError(f"target index {target_index} out of range for {len(source_actions)} actions")
    actions = [int(action) for action in source_actions]
    actions[target_index] = action_to_index(target_action_name)
    return actions


def compose_counterfactual_actions(
    source_actions: Sequence[int],
    snapshot_actions: Sequence[int],
    target_index: int,
    *,
    neighbor_mode: str,
    target_action_name: Optional[str] = None,
) -> List[int]:
    """Build one rollout action vector under a controlled teacher intervention.

    The initial interval is identical for both teachers: preserve every
    snapshot-phase neighbor and force the target candidate. After that
    interval, the target always follows the source policy. The only treatment
    variable is whether the non-target intersections stay frozen or replan.
    """
    if len(source_actions) != len(snapshot_actions):
        raise ValueError(
            "source_actions and snapshot_actions must have the same length: "
            f"{len(source_actions)} != {len(snapshot_actions)}"
        )
    if target_index < 0 or target_index >= len(source_actions):
        raise IndexError(f"target index {target_index} out of range for {len(source_actions)} actions")
    if neighbor_mode not in {"responsive", "frozen"}:
        raise ValueError(f"Unsupported neighbor_mode={neighbor_mode!r}")

    if target_action_name is not None:
        return force_target_action(snapshot_actions, target_index, target_action_name)
    if neighbor_mode == "responsive":
        return [int(action) for action in source_actions]

    actions = [int(action) for action in snapshot_actions]
    actions[target_index] = int(source_actions[target_index])
    return actions


@dataclass
class RollingSequenceBank:
    """Saved policy histories advanced from CityFlow movement counters."""

    rows: Dict[str, List[List[float]]]
    seq_len: int
    input_dim: int
    interval_seconds: int
    last_bin: int

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, Any]],
        intersection_ids: Sequence[str],
        *,
        seq_len: int,
        input_dim: int,
        interval_seconds: int,
        snapshot_time_s: float,
    ) -> "RollingSequenceBank":
        by_intersection = {
            str(record.get("intersection_id", "")): record
            for record in records
            if str(record.get("intersection_id", ""))
        }
        rows = {
            inter_id: normalize_sequence(
                by_intersection.get(inter_id, {}).get("sequences", []),
                seq_len=seq_len,
                input_dim=input_dim,
            )
            for inter_id in intersection_ids
        }
        safe_interval = max(1, int(interval_seconds))
        return cls(
            rows=rows,
            seq_len=int(seq_len),
            input_dim=int(input_dim),
            interval_seconds=safe_interval,
            last_bin=int(float(snapshot_time_s)) // safe_interval,
        )

    def sequence(self, intersection_id: str) -> List[List[float]]:
        return [list(row) for row in self.rows[intersection_id][-self.seq_len:]]

    def advance_from_cityflow(self, env: Any) -> int:
        """Append newly completed traffic-counter bins from the active branch."""
        current_bin = int(float(env.get_current_time())) // self.interval_seconds
        if current_bin <= self.last_bin:
            return 0
        counts = getattr(env, "traffic_counts_by_interval", {}).get(self.interval_seconds)
        inter_ids = list(getattr(env, "traffic_intersection_ids", []) or [])
        if counts is None or not inter_ids:
            self.last_bin = current_bin
            return 0

        index_by_intersection = {name: index for index, name in enumerate(inter_ids)}
        appended = 0
        for bin_index in range(self.last_bin + 1, current_bin + 1):
            if bin_index >= counts.shape[0]:
                break
            for inter_id, history in self.rows.items():
                inter_index = index_by_intersection.get(inter_id)
                if inter_index is None:
                    continue
                start = inter_index * self.input_dim
                values = counts[bin_index, start : start + self.input_dim]
                row = [_finite_float(value) for value in values]
                if len(row) < self.input_dim:
                    row.extend([0.0] * (self.input_dim - len(row)))
                history.append(row)
                if len(history) > self.seq_len:
                    del history[:-self.seq_len]
            appended += 1
        self.last_bin = current_bin
        return appended


class PolicyResponsiveSimExpertRelabeler(SimExpertRelabeler):
    """Counterfactual teacher with source-policy network response."""

    def __init__(
        self,
        *,
        roadnet_path: str,
        traffic_file: str,
        work_dir: str,
        num_rows: int,
        num_cols: int,
        source_policy_ckpt: str,
        llm_path: str = "",
        dataset: str = "hangzhou",
        sim_horizon: int = 30,
        sim_interval: float = 1.0,
        min_action_time: int = 30,
        traffic_series_interval: int = 2,
        inference_batch_size: int = 0,
        device: str = "auto",
        bf16: bool = True,
        fp16: bool = False,
        seq_len: int = 0,
        image_size: int = 0,
        periodicity: int = 0,
        input_dim: int = 0,
        hidden_dim: int = 0,
        prefix_tokens: int = 0,
        prefix_mode: str = "",
        max_length: int = 0,
        cost_scope: str = "global",
        max_snapshots: int = 0,
        neighbor_mode: str = "responsive",
        target_intersections: Sequence[str] | None = None,
        cost_att_weight: float = 1.0,
        cost_aql_weight: float = 1.0,
        cost_awt_weight: float = 1.0,
        cost_spillback_weight: float = 0.2,
        awt_constraint_epsilon: float = 5.0,
        spillback_constraint_epsilon: float = 0.05,
        enable_pareto_filter: bool = True,
        enable_constraint_filter: bool = True,
    ) -> None:
        if int(sim_horizon) <= 0:
            raise ValueError("sim_horizon must be positive")
        if int(min_action_time) <= 3:
            raise ValueError("min_action_time must exceed the canonical 3-second yellow phase")
        if str(cost_scope) not in {"local", "global"}:
            raise ValueError("cost_scope must be 'local' or 'global'")
        if str(neighbor_mode) not in {"responsive", "frozen"}:
            raise ValueError("neighbor_mode must be 'responsive' or 'frozen'")
        if bool(bf16) and bool(fp16):
            raise ValueError("Choose only one of bf16/fp16")

        super().__init__(
            roadnet_path=roadnet_path,
            traffic_file=traffic_file,
            work_dir=work_dir,
            num_rows=num_rows,
            num_cols=num_cols,
            sim_horizon=sim_horizon,
            sim_interval=sim_interval,
            cost_att_weight=cost_att_weight,
            cost_aql_weight=cost_aql_weight,
            cost_awt_weight=cost_awt_weight,
            cost_spillback_weight=cost_spillback_weight,
            awt_constraint_epsilon=awt_constraint_epsilon,
            spillback_constraint_epsilon=spillback_constraint_epsilon,
            enable_pareto_filter=enable_pareto_filter,
            enable_constraint_filter=enable_constraint_filter,
        )
        self.source_policy_ckpt = os.path.abspath(source_policy_ckpt)
        self.llm_path = str(llm_path or "")
        self.dataset = str(dataset)
        self.min_action_time = int(min_action_time)
        self.traffic_series_interval = max(1, int(traffic_series_interval))
        self.inference_batch_size = max(0, int(inference_batch_size))
        self.device_text = str(device)
        self.use_bf16 = bool(bf16)
        self.use_fp16 = bool(fp16)
        self.runtime_overrides = {
            "seq_len": int(seq_len),
            "image_size": int(image_size),
            "periodicity": int(periodicity),
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "prefix_tokens": int(prefix_tokens),
            "prefix_mode": str(prefix_mode or ""),
            "max_length": int(max_length),
        }
        self.cost_scope = str(cost_scope)
        self.max_snapshots = max(0, int(max_snapshots))
        self.neighbor_mode = str(neighbor_mode)
        self.target_intersections = {
            str(intersection).strip()
            for intersection in (target_intersections or [])
            if str(intersection).strip()
        }
        self._runner: Any = None

    def _ensure_runner(self, max_snapshot_time_s: float) -> Any:
        if self._runner is not None:
            return self._runner
        if not os.path.isfile(self.source_policy_ckpt):
            raise FileNotFoundError(f"Source policy checkpoint not found: {self.source_policy_ckpt}")

        # Importing the canonical runner imports CityFlow and model dependencies,
        # so keep it out of module import time for lightweight tests.
        import torch

        from projection_train.run_projection_online import (
            ProjectionPolicyRunner,
            _resolve_device,
            _resolve_projection_runtime,
            _resolve_torch_dtype,
            merge,
        )
        from utils.config import dic_traffic_env_conf

        runtime_args = SimpleNamespace(
            projection_ckpt=self.source_policy_ckpt,
            llm_path=self.llm_path,
            fp16=self.use_fp16,
            bf16=self.use_bf16,
            prompt_mode="",
            **self.runtime_overrides,
        )
        runtime, prompt_mode = _resolve_projection_runtime(runtime_args)
        max_run = int(math.ceil(max_snapshot_time_s)) + self.sim_horizon + self.min_action_time + 1
        num_intersections = self.num_rows * self.num_cols
        os.makedirs(self.work_dir, exist_ok=True)
        runtime_dir = os.path.join(self.work_dir, "policy_controller_runtime")
        model_dir = os.path.join(self.work_dir, "policy_controller_model")
        if os.path.exists(runtime_dir) or os.path.exists(model_dir):
            raise FileExistsError(
                "Policy-responsive runtime directory already exists. Use a fresh --work_dir "
                "to avoid sharing CityFlow state across relabeling runs."
            )

        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = _resolve_device(self.device_text)
        dtype = _resolve_torch_dtype(runtime_args, device)
        if device.type == "cuda":
            torch.cuda.set_device(device)

        roadnet_name = os.path.basename(self.roadnet_path)
        traffic_name = os.path.basename(self.traffic_file)
        agent_conf = {
            "LLM_PATH": runtime.llm_path,
            "LLM_MODEL": os.path.basename(runtime.llm_path.rstrip("/")) or "projection-policy",
            "LOG_DIR": os.path.join(self.work_dir, "logs"),
            "TRAFFIC_SERIES_INTERVAL": self.traffic_series_interval,
            "TRAFFIC_SERIES_WINDOW": max(
                self.traffic_series_interval,
                (runtime.seq_len - 1) * self.traffic_series_interval,
            ),
            "TRAFFIC_SERIES_OUTPUT_DIR": "",
            "PROJECTION_CKPT": self.source_policy_ckpt,
            "PROMPT_MODE": prompt_mode,
            "FIXED_TIME": [self.min_action_time for _ in ACTION_LIST],
        }
        env_conf = merge(
            dic_traffic_env_conf,
            {
                "NUM_AGENTS": num_intersections,
                "NUM_INTERSECTIONS": num_intersections,
                "MODEL_NAME": "policy-responsive-simexpert",
                "PROJECT_NAME": "Phase7-policy-responsive",
                "RUN_COUNTS": max_run,
                "NUM_ROUNDS": 1,
                "NUM_ROW": self.num_rows,
                "NUM_COL": self.num_cols,
                "CITYFLOW_SEED": 0,
                "MIN_ACTION_TIME": self.min_action_time,
                "MEASURE_TIME": self.min_action_time,
                "SAVE_REPLAY": False,
                "ENABLE_TRAFFIC_COUNT": True,
                "TRAFFIC_COUNT_MODE": "intersection_movement",
                "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
                "TRAFFIC_COUNT_INTERVALS": sorted({self.traffic_series_interval, 5, 60, 3600}),
                "TRAFFIC_COUNT_ADD_TOTAL": True,
                "TRAFFIC_COUNT_BASENAME": "traffic_counts",
                "TRAFFIC_FILE": traffic_name,
                "ROADNET_FILE": roadnet_name,
                "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
                "DIC_REWARD_INFO": {"queue_length": -0.25},
            },
        )
        runner = ProjectionPolicyRunner(
            dic_agent_conf=agent_conf,
            dic_traffic_env_conf_local=env_conf,
            dic_path={
                "PATH_TO_MODEL": model_dir,
                "PATH_TO_WORK_DIRECTORY": runtime_dir,
                "PATH_TO_DATA": os.path.dirname(self.roadnet_path),
            },
            roadnet=f"policy-responsive-{self.dataset}",
            trafficflow=os.path.splitext(traffic_name)[0],
            runtime=runtime,
            projection_ckpt=self.source_policy_ckpt,
            prompt_mode=prompt_mode,
            device=device,
            torch_dtype=dtype,
            use_wandb=False,
            logger=None,
            record_snapshots=False,
            record_traffic_series=False,
            dump_cityflow_env_args=False,
            inference_batch_size=self.inference_batch_size,
        )
        self._runner = runner
        return runner

    @staticmethod
    def _refresh_env_measurements(env: Any) -> None:
        env.system_states = {
            "get_lane_vehicles": env.eng.get_lane_vehicles(),
            "get_lane_waiting_vehicle_count": env.eng.get_lane_waiting_vehicle_count(),
            "get_vehicle_speed": env.eng.get_vehicle_speed(),
            "get_vehicle_distance": env.eng.get_vehicle_distance(),
        }
        for intersection in env.list_intersection:
            intersection.update_current_measurements(env.system_states)
        env._update_traffic_counts()

    def _restore_snapshot(self, snapshot_path: str) -> Tuple[Any, Dict[str, int]]:
        runner = self._runner
        assert runner is not None and runner.env is not None
        env = runner.env
        env.eng.load_from_file(snapshot_path)
        try:
            env.eng.set_random_seed(0)
        except Exception:
            pass
        phases = load_snapshot_phase_indices(snapshot_path)
        for intersection in env.list_intersection:
            synchronize_intersection_phase(intersection, phases.get(intersection.inter_name, 1))
        env.waiting_vehicle_list = {}
        self._refresh_env_measurements(env)
        return env, phases

    @staticmethod
    def _lane_metrics(lanes: Sequence[str], vehicle_counts: Mapping[str, Any], waiting_counts: Mapping[str, Any]) -> Tuple[float, float]:
        occupancy = sum(_finite_float(vehicle_counts.get(lane, 0.0)) for lane in lanes)
        queue = sum(_finite_float(waiting_counts.get(lane, 0.0)) for lane in lanes)
        return occupancy, queue

    @staticmethod
    def _spillback(
        lanes: Sequence[str],
        lane_lengths: Mapping[str, Any],
        vehicle_counts: Mapping[str, Any],
        waiting_counts: Mapping[str, Any],
    ) -> float:
        occupancies: List[float] = []
        blocked: List[float] = []
        for lane in lanes:
            capacity = max(1.0, _finite_float(lane_lengths.get(lane, 200.0), 200.0) / SPILLBACK_JAM_SPACING_M)
            occupancy = min(1.0, _finite_float(vehicle_counts.get(lane, 0.0)) / capacity)
            occupancies.append(occupancy)
            blocked.append(1.0 if occupancy >= SPILLBACK_BLOCK_THRESHOLD and _finite_float(waiting_counts.get(lane, 0.0)) > 0 else 0.0)
        return _metric_mean(occupancies) + SPILLBACK_BLOCK_WEIGHT * _metric_mean(blocked)

    def _measure_branch(self, env: Any, target_intersection_id: str) -> Dict[str, float]:
        vehicle_counts = env.eng.get_lane_vehicle_count()
        waiting_counts = env.eng.get_lane_waiting_vehicle_count()
        target = next(inter for inter in env.list_intersection if inter.inter_name == target_intersection_id)
        local_att, local_aql = self._lane_metrics(target.list_entering_lanes, vehicle_counts, waiting_counts)
        local_spillback = self._spillback(
            target.list_exiting_lanes,
            target.lane_length,
            vehicle_counts,
            waiting_counts,
        )

        entering_lanes: List[str] = []
        exiting_lanes: List[str] = []
        lane_lengths: Dict[str, Any] = {}
        for intersection in env.list_intersection:
            entering_lanes.extend(intersection.list_entering_lanes)
            exiting_lanes.extend(intersection.list_exiting_lanes)
            lane_lengths.update(intersection.lane_length)
        global_att, global_aql = self._lane_metrics(entering_lanes, vehicle_counts, waiting_counts)
        global_spillback = self._spillback(exiting_lanes, lane_lengths, vehicle_counts, waiting_counts)
        return {
            "local_att": local_att,
            "local_aql": local_aql,
            "local_awt": local_aql * self.sim_interval,
            "local_spillback": local_spillback,
            "global_att": global_att,
            "global_aql": global_aql,
            "global_awt": global_aql * self.sim_interval,
            "global_spillback": global_spillback,
        }

    def _policy_actions(self, env: Any, sequences: RollingSequenceBank) -> Tuple[List[int], Counter]:
        runner = self._runner
        assert runner is not None
        from projection_train.prompt_utils import build_phase_queue_wait_context, build_task_prompt
        from utils.my_utils import get_state_detail

        states: List[Dict[str, Any]] = []
        prompts: List[str] = []
        ordered_ids: List[str] = []
        for intersection in env.list_intersection:
            roads = copy.deepcopy(env.intersection_dict[intersection.inter_name]["roads"])
            state, _, _ = get_state_detail(roads, env)
            phase_queue, phase_wait = runner._phase_queue_wait(state)
            context = build_phase_queue_wait_context(
                state=state,
                phase_queue=phase_queue,
                phase_wait=phase_wait,
            )
            prompts.append(
                build_task_prompt(
                    runner._build_raw_prompt(state),
                    mode=runner.prompt_mode,
                    action_tags=ACTION_LIST,
                    traffic_context=context,
                )
            )
            states.append(state)
            ordered_ids.append(intersection.inter_name)

        output = runner.predict_from_sequences(
            [sequences.sequence(intersection_id) for intersection_id in ordered_ids],
            prompts,
        )
        actions = [int(action) for action in output["action_indices"]]
        return actions, Counter(output["predicted_actions"])

    def _simulate_candidate(
        self,
        *,
        snapshot_path: str,
        snapshot_records: Sequence[Mapping[str, Any]],
        target_intersection_id: str,
        target_action_name: str,
        snapshot_time_s: float,
    ) -> Tuple[Dict[str, float], Dict[str, int], str]:
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
        target_index = intersection_ids.index(target_intersection_id)
        source_actions, response_actions = self._policy_actions(env, sequences)
        base_action_name = ACTION_LIST[0]
        if target_index < len(source_actions):
            base_action_name = ACTION_LIST[int(source_actions[target_index])]
        snapshot_actions = [
            action_to_index(ACTION_LIST[max(1, min(len(ACTION_LIST), int(phases.get(intersection_id, 1)))) - 1])
            for intersection_id in intersection_ids
        ]
        actions = compose_counterfactual_actions(
            source_actions,
            snapshot_actions,
            target_index,
            neighbor_mode=self.neighbor_mode,
            target_action_name=target_action_name,
        )

        traces: Dict[str, List[float]] = defaultdict(list)
        for elapsed in range(self.sim_horizon):
            if elapsed > 0 and elapsed % self.min_action_time == 0:
                sequences.advance_from_cityflow(env)
                source_actions, counts = self._policy_actions(env, sequences)
                response_actions.update(counts)
                actions = compose_counterfactual_actions(
                    source_actions,
                    snapshot_actions,
                    target_index,
                    neighbor_mode=self.neighbor_mode,
                )
            env._inner_step(actions)
            measured = self._measure_branch(env, target_intersection_id)
            for name, value in measured.items():
                traces[name].append(value)

        metrics = {
            "local_att": _metric_mean(traces["local_att"]),
            "local_aql": _metric_mean(traces["local_aql"]),
            "local_awt": _metric_sum(traces["local_awt"]),
            "local_spillback": _metric_mean(traces["local_spillback"]),
            "global_att": _metric_mean(traces["global_att"]),
            "global_aql": _metric_mean(traces["global_aql"]),
            "global_awt": _metric_sum(traces["global_awt"]),
            "global_spillback": _metric_mean(traces["global_spillback"]),
        }
        prefix = "global" if self.cost_scope == "global" else "local"
        metrics.update(
            {
                "att": metrics[f"{prefix}_att"],
                "aql": metrics[f"{prefix}_aql"],
                "awt": metrics[f"{prefix}_awt"],
                "spillback": metrics[f"{prefix}_spillback"],
            }
        )
        return metrics, dict(response_actions), base_action_name

    @staticmethod
    def _snapshot_time(records: Sequence[Mapping[str, Any]]) -> float:
        for record in records:
            if "time_s" in record:
                return _finite_float(record.get("time_s"))
        return 0.0

    def relabel(self, input_path: str, output_path: str) -> str:
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
            all_snapshot_items[: self.max_snapshots]
            if self.max_snapshots
            else all_snapshot_items
        )
        max_time = max(
            self._snapshot_time([records[index] for index in indices])
            for _, indices in selected_snapshot_items
        )
        self._ensure_runner(max_time)
        coverage = {
            "total_records": len(records),
            "total_snapshots": len(all_snapshot_items),
            "selected_snapshots": len(selected_snapshot_items),
            "processed_snapshots": 0,
            "skipped_snapshots": 0,
            "skipped_records": 0,
            "candidate_rollouts": 0,
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
                    records[index]["relabel_method"] = f"policy_{self.neighbor_mode}_not_selected"
                coverage["skipped_records"] += len(indices)
                continue
            try:
                phases = load_snapshot_phase_indices(snapshot_path)
                snapshot_time = self._snapshot_time(snapshot_records)
                by_intersection: Dict[str, Dict[str, Any]] = {}
                target_ids = dict.fromkeys(
                    str(record.get("intersection_id", "")) for record in snapshot_records
                )
                for target_id in target_ids:
                    if not target_id or target_id not in phases:
                        continue
                    if self.target_intersections and target_id not in self.target_intersections:
                        continue
                    raw_metrics: Dict[str, Dict[str, float]] = {}
                    response_counts: Dict[str, Dict[str, int]] = {}
                    base_action: Optional[str] = None
                    for action in ACTION_LIST:
                        metrics, counts, candidate_base_action = self._simulate_candidate(
                            snapshot_path=snapshot_path,
                            snapshot_records=snapshot_records,
                            target_intersection_id=target_id,
                            target_action_name=action,
                            snapshot_time_s=snapshot_time,
                        )
                        raw_metrics[action] = metrics
                        response_counts[action] = counts
                        coverage["candidate_rollouts"] += 1
                        if base_action is None:
                            base_action = candidate_base_action
                        elif base_action != candidate_base_action:
                            raise RuntimeError(
                                "Source-policy action changed across deterministic candidate branches: "
                                f"{base_action} != {candidate_base_action}"
                            )

                    if base_action is None:
                        raise RuntimeError(f"No source-policy action resolved for {target_id}")

                    scoring_metrics = {
                        action: {metric: values[metric] for metric in ("att", "aql", "awt", "spillback")}
                        for action, values in raw_metrics.items()
                    }
                    scores, enriched = self._score_candidates(scoring_metrics)
                    for action, values in enriched.items():
                        values.update(raw_metrics[action])
                        values["score_scope"] = self.cost_scope
                        values["response_action_counts"] = response_counts[action]
                    pareto, violations, _ = self._check_pareto_feasibility(enriched, base_action)
                    best = base_action
                    best_score = float("inf")
                    for action in ACTION_LIST:
                        if pareto.get(action, True) and action in scores:
                            score = scores[action]
                            if score < best_score - 1e-9 or (
                                abs(score - best_score) <= 1e-9 and action == base_action
                            ):
                                best, best_score = action, score
                    feasible = sorted(
                        scores[action] for action in ACTION_LIST if pareto.get(action, True) and action in scores
                    )
                    margin = feasible[1] - feasible[0] if len(feasible) >= 2 else 0.0
                    base_score = scores.get(base_action)
                    advantages = {
                        action: (float(base_score - score) if base_score is not None else 0.0)
                        for action, score in scores.items()
                    }
                    by_intersection[target_id] = {
                        "candidate_action_metrics": enriched,
                        "candidate_action_waits": scores,
                        "expert_action": best,
                        "expert_action_margin": margin,
                        "base_action": base_action,
                        "pareto_feasible": pareto,
                        "constraint_violations": dict(violations),
                        "candidate_advantages": advantages,
                    }

                for index in indices:
                    record = records[index]
                    target_id = str(record.get("intersection_id", ""))
                    result = by_intersection.get(target_id)
                    if result is None:
                        record.update(
                            {
                                "relabel_method": (
                                    f"policy_{self.neighbor_mode}_not_selected"
                                    if self.target_intersections and target_id not in self.target_intersections
                                    else f"policy_{self.neighbor_mode}_no_intersection"
                                ),
                                "candidate_action_metrics": {},
                                "candidate_action_waits": {},
                                "expert_action": record.get("policy_action", ACTION_LIST[0]),
                                "expert_action_margin": 0.0,
                                "base_action": record.get("policy_action", ACTION_LIST[0]),
                                "pareto_feasible": {},
                                "constraint_violations": {},
                            }
                        )
                        coverage["skipped_records"] += 1
                        continue
                    record.update(result)
                    record.update(
                        {
                            "relabel_method": f"simexpert_policy_{self.neighbor_mode}_neighbors",
                            "counterfactual_mode": f"policy_{self.neighbor_mode}_neighbors",
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
                        "[PolicyResponsiveSimExpert] processed "
                        f"{coverage['processed_snapshots']}/{coverage['total_snapshots']} snapshots",
                        flush=True,
                    )
            except Exception:
                coverage["skipped_snapshots"] += 1
                raise

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        coverage_path = output_path.replace(".jsonl", "_coverage.json")
        with open(coverage_path, "w", encoding="utf-8") as handle:
            json.dump(coverage, handle, indent=2)
        print(f"[PolicyResponsiveSimExpert] wrote {len(records)} records to {output_path}", flush=True)
        return output_path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=True, help="Collected decision-state JSONL with snapshots and sequences")
    parser.add_argument("--output", required=True, help="Output responsive-label JSONL")
    parser.add_argument("--roadnet", required=True, help="Target roadnet JSON")
    parser.add_argument("--traffic_file", required=True, help="Target flow JSON")
    parser.add_argument("--dataset", default="hangzhou", help="Dataset name retained in output provenance")
    parser.add_argument("--num_rows", type=int, required=True)
    parser.add_argument("--num_cols", type=int, required=True)
    parser.add_argument("--source_policy_ckpt", required=True, help="Frozen source projection-policy checkpoint")
    parser.add_argument("--llm_path", default="", help="Optional override; defaults to checkpoint metadata")
    parser.add_argument("--work_dir", required=True, help="Fresh runtime directory for CityFlow/model initialization")
    parser.add_argument("--sim_horizon", type=int, default=30)
    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--traffic_series_interval", type=int, default=2)
    parser.add_argument("--inference_batch_size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true", default=False)
    precision.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--cost_scope", choices=["local", "global"], default="global")
    parser.add_argument(
        "--neighbor-mode",
        choices=["responsive", "frozen"],
        default="responsive",
        help="Counterfactual controller after the forced target interval.",
    )
    parser.add_argument(
        "--max_snapshots",
        type=int,
        default=0,
        help="Process only the first N sorted snapshots; intended for smoke tests, not training labels.",
    )
    parser.add_argument(
        "--target_intersections",
        nargs="*",
        default=None,
        help="Optional intersection IDs for a small smoke test. Unselected records are marked and unusable for training.",
    )
    parser.add_argument("--cost_att_weight", type=float, default=1.0)
    parser.add_argument("--cost_aql_weight", type=float, default=1.0)
    parser.add_argument("--cost_awt_weight", type=float, default=1.0)
    parser.add_argument("--cost_spillback_weight", type=float, default=0.2)
    parser.add_argument("--awt_constraint_epsilon", type=float, default=5.0)
    parser.add_argument("--spillback_constraint_epsilon", type=float, default=0.05)
    parser.add_argument("--disable_pareto_filter", action="store_true")
    parser.add_argument("--disable_constraint_filter", action="store_true")
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--periodicity", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--prefix_tokens", type=int, default=0)
    parser.add_argument("--prefix_mode", default="")
    parser.add_argument("--max_length", type=int, default=0)


def relabel_from_args(args: argparse.Namespace) -> str:
    relabeler = PolicyResponsiveSimExpertRelabeler(
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
    )
    return relabeler.relabel(os.path.abspath(args.input), os.path.abspath(args.output))
