from __future__ import annotations

import json
import os
import shutil
from collections import deque
from copy import deepcopy
from functools import lru_cache
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from projection_train.action_space import ACTION_LIST
from projection_train.prompt_utils import build_task_prompt
from projection_train.reward_shaping import (
    REWARD_MODE_CHOICES,
    REWARD_MODE_STABLE_DELTA,
    WAIT_REDUCER_CHOICES,
    WAIT_REDUCER_MEAN,
    compute_local_shaped_reward,
    reduce_wait_values,
    resolve_objective_weights,
)
from projection_train.snapshot_seed_dataset import SnapshotSeedSample
from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf

def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    merged.update(override)
    return merged


@lru_cache(maxsize=1024)
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_existing_path(path: str) -> str:
    text = str(path or "").strip()
    if not text:
        raise FileNotFoundError("Empty path")
    candidates = [text]
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    candidates.append(os.path.join(repo_root, text))
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError(f"Path not found: {path}")


def _infer_grid_shape(roadnet_path: str) -> Tuple[int, int]:
    roadnet = _load_json(roadnet_path)
    max_col = 0
    max_row = 0
    for intersection in roadnet.get("intersections", []) or []:
        if not intersection.get("roadLinks"):
            continue
        inter_id = str(intersection.get("id") or "")
        parts = inter_id.split("_")
        if len(parts) != 3 or parts[0] != "intersection":
            continue
        try:
            col = int(parts[1])
            row = int(parts[2])
        except ValueError:
            continue
        max_col = max(max_col, col)
        max_row = max(max_row, row)
    if max_col <= 0 or max_row <= 0:
        raise ValueError(f"Failed to infer grid shape from roadnet: {roadnet_path}")
    return max_col, max_row


def _build_phase_time_lookup(roadnet_path: str) -> Dict[str, Dict[int, int]]:
    roadnet = _load_json(roadnet_path)
    lookup: Dict[str, Dict[int, int]] = {}
    for intersection in roadnet.get("intersections", []) or []:
        inter_id = str(intersection.get("id") or "")
        phases = {}
        for idx, phase in enumerate(intersection.get("trafficLight", {}).get("lightphases", []) or []):
            try:
                phases[idx] = int(phase.get("time", 0) or 0)
            except (TypeError, ValueError):
                phases[idx] = 0
        lookup[inter_id] = phases
    return lookup


def _build_neighbor_intersection_lookup(roadnet_path: str) -> Dict[str, List[str]]:
    roadnet = _load_json(roadnet_path)
    controlled_ids = {
        str(intersection.get("id") or "")
        for intersection in roadnet.get("intersections", []) or []
        if intersection.get("roadLinks")
    }
    neighbors: Dict[str, set[str]] = {inter_id: set() for inter_id in controlled_ids}
    for road in roadnet.get("roads", []) or []:
        start_id = str(road.get("startIntersection") or "")
        end_id = str(road.get("endIntersection") or "")
        if start_id not in controlled_ids or end_id not in controlled_ids or start_id == end_id:
            continue
        neighbors.setdefault(start_id, set()).add(end_id)
        neighbors.setdefault(end_id, set()).add(start_id)
    return {
        inter_id: sorted(adjacent_ids)
        for inter_id, adjacent_ids in neighbors.items()
    }


class TrajectoryCityFlowEnv:
    def __init__(
        self,
        roadnet_path: str,
        traffic_root: str,
        work_root: str,
        default_traffic_file: str,
        seq_len: int = 16,
        movement_keys: Optional[Sequence[str]] = None,
        prompt_mode: str = "image_only_short",
        traffic_series_interval: int = 2,
        window_s: int = 30,
        min_action_time: int = 30,
        yellow_time: int = 3,
        interval: float = 1.0,
        reward_alpha: float = 0.1,
        switch_penalty_beta: float = 0.02,
        queue_reward_coef: float = -0.25,
        waiting_reward_coef: float = -0.02,
        system_vehicle_reward_coef: float = -0.01,
        throughput_reward_coef: float = 0.02,
        starvation_penalty_coef: float = 0.05,
        starvation_threshold_steps: int = 2,
        starvation_queue_threshold: float = 8.0,
        starvation_wait_threshold: float = 45.0,
        wait_reward_scope: str = "neighborhood",
        wait_reward_reducer: str = WAIT_REDUCER_MEAN,
        reward_mode: str = REWARD_MODE_STABLE_DELTA,
        queue_delta_scale: float = 4.0,
        wait_delta_scale: float = 30.0,
        wait_abs_coef: float = 0.0,
        wait_abs_scale: float = 60.0,
        active_delta_scale: float = 4.0,
        throughput_scale: float = 2.0,
        reward_clip: float = 3.0,
        global_reward_scale: float = 1.0,
        selection_queue_weight: float = 1.0,
        selection_wait_weight: float = 1.0,
        selection_travel_weight: float = 1.0,
        reward_info: Optional[Dict[str, float]] = None,
        run_counts: int = 3600,
        env_seed: Optional[int] = None,
        advance_seed_on_reset: bool = False,
    ) -> None:
        self.roadnet_path = _resolve_existing_path(roadnet_path)
        self.traffic_root = _resolve_existing_path(traffic_root)
        self.work_root = work_root
        self.default_traffic_file = str(default_traffic_file)
        self.seq_len = int(seq_len)
        self.movement_keys = list(movement_keys or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"])
        self.prompt_mode = str(prompt_mode or "image_only_short")
        self.traffic_series_interval = int(traffic_series_interval)
        self.window_s = int(window_s)
        self.min_action_time = int(min_action_time)
        self.yellow_time = int(yellow_time)
        self.interval = float(interval)
        self.reward_alpha = float(reward_alpha)
        self.switch_penalty_beta = float(switch_penalty_beta)
        self.queue_reward_coef = float(queue_reward_coef)
        self.waiting_reward_coef = float(waiting_reward_coef)
        self.system_vehicle_reward_coef = float(system_vehicle_reward_coef)
        self.throughput_reward_coef = float(throughput_reward_coef)
        self.starvation_penalty_coef = float(starvation_penalty_coef)
        self.starvation_threshold_steps = max(1, int(starvation_threshold_steps))
        self.starvation_queue_threshold = max(1e-6, float(starvation_queue_threshold))
        self.starvation_wait_threshold = max(1e-6, float(starvation_wait_threshold))
        resolved_wait_scope = str(wait_reward_scope or "neighborhood").strip().lower()
        if resolved_wait_scope not in {"local", "neighborhood"}:
            raise ValueError("wait_reward_scope must be one of {'local', 'neighborhood'}.")
        resolved_wait_reducer = str(wait_reward_reducer or WAIT_REDUCER_MEAN).strip().lower()
        if resolved_wait_reducer not in WAIT_REDUCER_CHOICES:
            raise ValueError(
                f"wait_reward_reducer must be one of {WAIT_REDUCER_CHOICES}, got {wait_reward_reducer!r}."
            )
        resolved_reward_mode = str(reward_mode or REWARD_MODE_STABLE_DELTA).strip().lower()
        if resolved_reward_mode not in REWARD_MODE_CHOICES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODE_CHOICES}, got {reward_mode!r}.")
        self.wait_reward_scope = resolved_wait_scope
        self.wait_reward_reducer = resolved_wait_reducer
        self.reward_mode = resolved_reward_mode
        self.queue_delta_scale = max(1e-8, float(queue_delta_scale))
        self.wait_delta_scale = max(1e-8, float(wait_delta_scale))
        self.wait_abs_coef = float(wait_abs_coef)
        self.wait_abs_scale = max(1e-8, float(wait_abs_scale))
        self.active_delta_scale = max(1e-8, float(active_delta_scale))
        self.throughput_scale = max(1e-8, float(throughput_scale))
        self.reward_clip = max(0.0, float(reward_clip))
        self.global_reward_scale = float(global_reward_scale)
        self.objective_weights = resolve_objective_weights(
            queue_weight=selection_queue_weight,
            wait_weight=selection_wait_weight,
            travel_weight=selection_travel_weight,
        )
        self.reward_info = dict(reward_info or {"queue_length": -0.25})
        self.run_counts = int(run_counts)
        self.base_env_seed = None if env_seed is None else int(env_seed)
        self.advance_seed_on_reset = bool(advance_seed_on_reset)
        self._reset_seed_counts: Dict[str, int] = {}

        self.num_col, self.num_row = _infer_grid_shape(self.roadnet_path)
        self.phase_time_lookup = _build_phase_time_lookup(self.roadnet_path)
        self.neighbor_intersection_lookup = _build_neighbor_intersection_lookup(self.roadnet_path)
        self._env_cache: Dict[str, CityFlowEnv] = {}

        self.env: Optional[CityFlowEnv] = None
        self.current_traffic_file = self.default_traffic_file
        self.intersection_ids: List[str] = []
        self.history_buffers: Dict[str, Deque[List[float]]] = {}
        self.prev_actions: List[int] = []
        self.current_time_s: float = 0.0
        self.done: bool = False
        self.last_reset_mode: str = "full"
        self.active_vehicle_ids: set[str] = set()
        self.movement_lanes_by_intersection: Dict[str, Dict[str, List[str]]] = {}
        self.phase_lanes_by_intersection: Dict[str, Dict[int, List[str]]] = {}
        self.enter_lane_to_intersection: Dict[str, str] = {}
        self.enter_lane_to_phase_indices: Dict[str, List[int]] = {}
        self.starvation_counters: Dict[str, List[int]] = {}
        self.prev_local_metrics_by_intersection: Dict[str, Dict[str, Any]] = {}
        self.policy_prompt = build_task_prompt("", mode=self.prompt_mode, action_tags=ACTION_LIST)

    def _prepare_env_workspace(self, traffic_file: str) -> Tuple[str, str]:
        traffic_src = os.path.join(self.traffic_root, traffic_file)
        if not os.path.exists(traffic_src):
            raise FileNotFoundError(f"Traffic file not found: {traffic_src}")

        traffic_stem = traffic_file.replace(".json", "")
        work_dir = os.path.join(self.work_root, traffic_stem)
        log_dir = os.path.join(work_dir, "logs")
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        shutil.copy2(self.roadnet_path, os.path.join(work_dir, os.path.basename(self.roadnet_path)))
        shutil.copy2(traffic_src, os.path.join(work_dir, traffic_file))
        return work_dir, log_dir

    def _create_env(self, traffic_file: str) -> CityFlowEnv:
        work_dir, log_dir = self._prepare_env_workspace(traffic_file)
        env_conf = _merge_configs(
            dic_traffic_env_conf,
            {
                "MODEL_NAME": "projection-ppo-traj",
                "PROJECT_NAME": "projection-ppo-traj",
                "NUM_AGENTS": self.num_col * self.num_row,
                "NUM_INTERSECTIONS": self.num_col * self.num_row,
                "NUM_COL": self.num_col,
                "NUM_ROW": self.num_row,
                "RUN_COUNTS": self.run_counts,
                "MIN_ACTION_TIME": self.min_action_time,
                "MEASURE_TIME": self.min_action_time,
                "YELLOW_TIME": self.yellow_time,
                "INTERVAL": self.interval,
                "TRAFFIC_FILE": traffic_file,
                "ROADNET_FILE": os.path.basename(self.roadnet_path),
                "ENABLE_TRAFFIC_COUNT": True,
                "TRAFFIC_COUNT_MODE": "intersection_movement",
                "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
                "TRAFFIC_COUNT_INTERVALS": sorted({self.traffic_series_interval, 5, 60, 3600}),
                "TRAFFIC_COUNT_MOVEMENT_KEYS": list(self.movement_keys),
                "TRAFFIC_COUNT_ADD_TOTAL": True,
                "TRAFFIC_COUNT_BASENAME": "traffic_counts",
                "DIC_REWARD_INFO": dict(self.reward_info),
                "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
                "CITYFLOW_SEED": self._resolve_cityflow_seed(traffic_file),
            },
        )
        dic_path = {
            "PATH_TO_MODEL": work_dir,
            "PATH_TO_WORK_DIRECTORY": work_dir,
            "PATH_TO_DATA": work_dir,
        }
        env = CityFlowEnv(
            path_to_log=log_dir,
            path_to_work_directory=work_dir,
            dic_traffic_env_conf=env_conf,
            dic_path=dic_path,
        )
        return env

    def get_env(self, traffic_file: str) -> CityFlowEnv:
        if traffic_file not in self._env_cache:
            self._env_cache[traffic_file] = self._create_env(traffic_file)
        return self._env_cache[traffic_file]

    def _resolve_cityflow_seed(self, traffic_file: str) -> Optional[int]:
        if self.base_env_seed is None:
            return None
        offset = self._reset_seed_counts.get(traffic_file, 0) if self.advance_seed_on_reset else 0
        return int(self.base_env_seed + offset)

    def _apply_cityflow_seed(self, env: CityFlowEnv, traffic_file: str) -> None:
        seed = self._resolve_cityflow_seed(traffic_file)
        if seed is None:
            env.dic_traffic_env_conf.pop("CITYFLOW_SEED", None)
            return
        env.dic_traffic_env_conf["CITYFLOW_SEED"] = int(seed)

    def _mark_env_reset(self, traffic_file: str) -> None:
        if self.base_env_seed is None or not self.advance_seed_on_reset:
            return
        self._reset_seed_counts[traffic_file] = self._reset_seed_counts.get(traffic_file, 0) + 1

    def _restore_phase_state(self, env: CityFlowEnv, snapshot_path: str) -> None:
        snapshot = _load_json(snapshot_path)
        traffic_lights = snapshot.get("trafficLights", {}) or {}
        for inter in env.list_intersection:
            light_state = traffic_lights.get(inter.inter_name) or {}
            phase_idx = int(light_state.get("curPhaseIndex", 1) or 0)
            remain = float(light_state.get("remainDuration", 0.0) or 0.0)
            phase_time = self.phase_time_lookup.get(inter.inter_name, {}).get(
                phase_idx,
                self.yellow_time if phase_idx == 0 else self.min_action_time,
            )
            elapsed = max(1, int(round(max(0.0, phase_time - remain))))

            if phase_idx == 0:
                inter.current_phase_index = inter.all_yellow_phase_index
                inter.previous_phase_index = inter.all_yellow_phase_index
                inter.next_phase_to_set_index = None
                inter.current_phase_duration = elapsed
                inter.all_yellow_flag = True
                inter.all_red_flag = False
                inter.flicker = 1
            else:
                inter.current_phase_index = phase_idx
                inter.previous_phase_index = phase_idx
                inter.next_phase_to_set_index = phase_idx
                inter.current_phase_duration = elapsed
                inter.all_yellow_flag = False
                inter.all_red_flag = False
                inter.flicker = 0

    @staticmethod
    def _base_action_for_intersection(inter) -> int:
        if isinstance(inter.current_phase_index, int) and inter.current_phase_index > 0:
            candidate = inter.current_phase_index - 1
            if 0 <= candidate < len(ACTION_LIST):
                return candidate
        next_idx = inter.next_phase_to_set_index
        if isinstance(next_idx, int) and next_idx > 0:
            candidate = next_idx - 1
            if 0 <= candidate < len(ACTION_LIST):
                return candidate
        return 0

    def _refresh_measurements(self, env: CityFlowEnv) -> None:
        env.system_states = {
            "get_lane_vehicles": env.eng.get_lane_vehicles(),
            "get_lane_waiting_vehicle_count": env.eng.get_lane_waiting_vehicle_count(),
            "get_vehicle_speed": env.eng.get_vehicle_speed(),
            "get_vehicle_distance": env.eng.get_vehicle_distance(),
        }
        for inter in env.list_intersection:
            inter.update_current_measurements(env.system_states)
        if env.intersection_dict is None:
            env.create_intersection_dict()

    def _make_empty_history(self) -> Deque[List[float]]:
        zero = [0.0 for _ in self.movement_keys]
        return deque([list(zero) for _ in range(self.seq_len)], maxlen=self.seq_len)

    def _init_history_from_seed(self, seed_sample: Optional[SnapshotSeedSample]) -> None:
        self.history_buffers = {}
        for inter_id in self.intersection_ids:
            history = self._make_empty_history()
            if seed_sample is not None and inter_id in seed_sample.sequences_by_intersection:
                history.clear()
                rows = seed_sample.sequences_by_intersection[inter_id][-self.seq_len :]
                if len(rows) < self.seq_len:
                    pad = list(rows[0]) if rows else [0.0 for _ in self.movement_keys]
                    for _ in range(self.seq_len - len(rows)):
                        history.append(list(pad))
                for row in rows:
                    history.append(list(row))
            self.history_buffers[inter_id] = history

    def _counts_matrix(self, env: CityFlowEnv):
        return getattr(env, "traffic_counts_by_interval", {}).get(self.traffic_series_interval)

    def _intersection_offset(self, env: CityFlowEnv, intersection_id: str) -> Optional[int]:
        intersections = list(getattr(env, "traffic_intersection_ids", []) or [])
        if intersection_id not in intersections:
            return None
        return intersections.index(intersection_id) * len(self.movement_keys)

    def _read_bin_counts(self, env: CityFlowEnv, intersection_id: str, bin_idx: int) -> List[float]:
        counts = self._counts_matrix(env)
        if counts is None or getattr(counts, "size", 0) <= 0:
            return [0.0 for _ in self.movement_keys]
        if bin_idx < 0:
            return [0.0 for _ in self.movement_keys]
        if bin_idx >= counts.shape[0]:
            bin_idx = counts.shape[0] - 1
        offset = self._intersection_offset(env, intersection_id)
        if offset is None:
            return [0.0 for _ in self.movement_keys]
        row = counts[bin_idx, offset : offset + len(self.movement_keys)]
        return [float(value) for value in row.tolist()]

    def _append_env_history(self, env: CityFlowEnv, prev_time_s: float, current_time_s: float) -> None:
        prev_bin = int(prev_time_s) // self.traffic_series_interval
        current_bin = int(current_time_s) // self.traffic_series_interval
        if current_bin < prev_bin:
            return
        for inter_id in self.intersection_ids:
            history = self.history_buffers.setdefault(inter_id, self._make_empty_history())
            for bin_idx in range(prev_bin + 1, current_bin + 1):
                history.append(self._read_bin_counts(env, inter_id, bin_idx))

    def _collect_active_vehicle_ids(self, env: Optional[CityFlowEnv] = None) -> set[str]:
        current_env = env or self.env
        if current_env is None:
            return set()
        lane_vehicles = (getattr(current_env, "system_states", {}) or {}).get("get_lane_vehicles", {}) or {}
        vehicle_ids: set[str] = set()
        for items in lane_vehicles.values():
            for veh in items or []:
                if "shadow" not in str(veh):
                    vehicle_ids.add(str(veh))
        return vehicle_ids

    @staticmethod
    def _action_movement_keys(action_idx: int) -> tuple[str, str]:
        action_idx = int(action_idx)
        if action_idx == 0:
            return ("ET", "WT")
        if action_idx == 1:
            return ("NT", "ST")
        if action_idx == 2:
            return ("EL", "WL")
        if action_idx == 3:
            return ("NL", "SL")
        return ("ET", "WT")

    @staticmethod
    def _location_to_prefix(location: str) -> str:
        text = str(location or "").strip().lower()
        mapping = {
            "north": "N",
            "south": "S",
            "east": "E",
            "west": "W",
        }
        return mapping.get(text, "")

    def _ensure_intersection_topology(self, env: CityFlowEnv) -> None:
        current_ids = [inter.inter_name for inter in env.list_intersection]
        if (
            self.movement_lanes_by_intersection
            and set(self.movement_lanes_by_intersection.keys()) == set(current_ids)
            and self.phase_lanes_by_intersection
        ):
            return
        if env.intersection_dict is None:
            env.create_intersection_dict()

        movement_lanes_by_intersection: Dict[str, Dict[str, List[str]]] = {}
        phase_lanes_by_intersection: Dict[str, Dict[int, List[str]]] = {}
        enter_lane_to_intersection: Dict[str, str] = {}
        enter_lane_to_phase_indices: Dict[str, List[int]] = {}

        for inter in env.list_intersection:
            inter_id = inter.inter_name
            roads = ((env.intersection_dict or {}).get(inter_id) or {}).get("roads") or {}
            movement_lanes = {key: [] for key in self.movement_keys}
            for road_id, road_info in roads.items():
                prefix = self._location_to_prefix(road_info.get("location"))
                if not prefix:
                    continue
                lanes = road_info.get("lanes") or {}
                for lane_idx in lanes.get("go_straight", []) or []:
                    lane_id = f"{road_id}_{int(lane_idx)}"
                    movement_key = f"{prefix}T"
                    if movement_key in movement_lanes:
                        movement_lanes[movement_key].append(lane_id)
                for lane_idx in lanes.get("turn_left", []) or []:
                    lane_id = f"{road_id}_{int(lane_idx)}"
                    movement_key = f"{prefix}L"
                    if movement_key in movement_lanes:
                        movement_lanes[movement_key].append(lane_id)
            for movement_key in movement_lanes:
                movement_lanes[movement_key] = sorted(set(movement_lanes[movement_key]))

            phase_lanes: Dict[int, List[str]] = {}
            for action_idx in range(len(ACTION_LIST)):
                lane_ids: List[str] = []
                for movement_key in self._action_movement_keys(action_idx):
                    lane_ids.extend(movement_lanes.get(movement_key, []) or [])
                deduped_lane_ids = sorted(set(lane_ids))
                phase_lanes[action_idx] = deduped_lane_ids
                for lane_id in deduped_lane_ids:
                    enter_lane_to_phase_indices.setdefault(lane_id, []).append(action_idx)

            movement_lanes_by_intersection[inter_id] = movement_lanes
            phase_lanes_by_intersection[inter_id] = phase_lanes
            for lane_id in inter.list_entering_lanes:
                enter_lane_to_intersection[lane_id] = inter_id

        self.movement_lanes_by_intersection = movement_lanes_by_intersection
        self.phase_lanes_by_intersection = phase_lanes_by_intersection
        self.enter_lane_to_intersection = enter_lane_to_intersection
        self.enter_lane_to_phase_indices = enter_lane_to_phase_indices

    def _collect_local_metrics(self, env: Optional[CityFlowEnv] = None) -> Dict[str, Dict[str, Any]]:
        current_env = env or self.env
        if current_env is None:
            return {}
        self._ensure_intersection_topology(current_env)

        waiting_values_by_intersection = {inter_id: [] for inter_id in self.intersection_ids}
        waiting_values_by_phase = {
            inter_id: {action_idx: [] for action_idx in range(len(ACTION_LIST))}
            for inter_id in self.intersection_ids
        }
        active_vehicle_ids_by_intersection = {inter_id: set() for inter_id in self.intersection_ids}
        for vehicle_info in (current_env.waiting_vehicle_list or {}).values():
            lane_id = str(vehicle_info.get("link") or "")
            inter_id = self.enter_lane_to_intersection.get(lane_id)
            if inter_id is None:
                continue
            wait_time = float(vehicle_info.get("time") or 0.0)
            waiting_values_by_intersection.setdefault(inter_id, []).append(wait_time)
            for action_idx in self.enter_lane_to_phase_indices.get(lane_id, []):
                waiting_values_by_phase.setdefault(inter_id, {}).setdefault(action_idx, []).append(wait_time)

        lane_waiting = ((getattr(current_env, "system_states", {}) or {}).get("get_lane_waiting_vehicle_count", {})) or {}
        lane_vehicles = ((getattr(current_env, "system_states", {}) or {}).get("get_lane_vehicles", {})) or {}
        local_metrics: Dict[str, Dict[str, Any]] = {}
        for inter in current_env.list_intersection:
            inter_id = inter.inter_name
            phase_queue_sums: Dict[int, float] = {}
            phase_wait_means: Dict[int, float] = {}
            for action_idx in range(len(ACTION_LIST)):
                lane_ids = self.phase_lanes_by_intersection.get(inter_id, {}).get(action_idx, []) or []
                phase_queue_sums[action_idx] = float(
                    sum(float(lane_waiting.get(lane_id, 0.0) or 0.0) for lane_id in lane_ids)
                )
                phase_waits = waiting_values_by_phase.get(inter_id, {}).get(action_idx, [])
                phase_wait_means[action_idx] = float(np.mean(phase_waits) if phase_waits else 0.0)
            for lane_id in inter.list_entering_lanes:
                for veh in lane_vehicles.get(lane_id, []) or []:
                    if "shadow" in str(veh):
                        continue
                    active_vehicle_ids_by_intersection.setdefault(inter_id, set()).add(str(veh))
            inter_waits = waiting_values_by_intersection.get(inter_id, [])
            active_vehicle_ids = set(active_vehicle_ids_by_intersection.get(inter_id, set()))
            local_metrics[inter_id] = {
                "queue_sum": float(
                    sum((inter.dic_feature or {}).get("lane_num_waiting_vehicle_in", []) or [])
                ),
                "waiting_time_mean": float(np.mean(inter_waits) if inter_waits else 0.0),
                "waiting_values": [float(value) for value in inter_waits],
                "active_vehicle_count": float(len(active_vehicle_ids)),
                "active_vehicle_ids": active_vehicle_ids,
                "phase_queue_sums": phase_queue_sums,
                "phase_wait_means": phase_wait_means,
            }
        return local_metrics

    def _wait_reward_intersection_ids(self, intersection_id: str) -> List[str]:
        if self.wait_reward_scope == "local":
            return [intersection_id]
        return list(
            dict.fromkeys([intersection_id, *self.neighbor_intersection_lookup.get(intersection_id, [])])
        )

    def _resolve_wait_reward_value(
        self,
        local_metrics_by_intersection: Dict[str, Dict[str, Any]],
        intersection_id: str,
    ) -> float:
        wait_values: List[float] = []
        for candidate_id in self._wait_reward_intersection_ids(intersection_id):
            candidate_metrics = local_metrics_by_intersection.get(candidate_id) or {}
            wait_values.extend(float(value) for value in (candidate_metrics.get("waiting_values") or []))
        return reduce_wait_values(wait_values, reducer=self.wait_reward_reducer)

    def _phase_demand_scale(self, local_metrics: Dict[str, Any], action_idx: int) -> float:
        phase_queue = float((local_metrics.get("phase_queue_sums") or {}).get(int(action_idx), 0.0))
        phase_wait = float((local_metrics.get("phase_wait_means") or {}).get(int(action_idx), 0.0))
        queue_scale = phase_queue / self.starvation_queue_threshold if phase_queue >= self.starvation_queue_threshold else 0.0
        wait_scale = phase_wait / self.starvation_wait_threshold if phase_wait >= self.starvation_wait_threshold else 0.0
        return float(max(queue_scale, wait_scale))

    def _update_starvation_penalty(
        self,
        intersection_id: str,
        selected_action: int,
        local_metrics: Dict[str, Any],
    ) -> tuple[float, List[int], Dict[int, float]]:
        counters = list(self.starvation_counters.get(intersection_id, [0] * len(ACTION_LIST)))
        demand_scales: Dict[int, float] = {}
        starvation_penalty = 0.0
        selected_action = int(selected_action)

        for action_idx in range(len(ACTION_LIST)):
            demand_scale = self._phase_demand_scale(local_metrics, action_idx)
            demand_scales[action_idx] = demand_scale
            if action_idx == selected_action:
                counters[action_idx] = 0
                continue
            if demand_scale > 0.0:
                counters[action_idx] += 1
            else:
                counters[action_idx] = 0
            overdue_steps = max(0, counters[action_idx] - self.starvation_threshold_steps + 1)
            if overdue_steps > 0 and demand_scale > 0.0:
                starvation_penalty = max(
                    starvation_penalty,
                    self.starvation_penalty_coef * demand_scale * float(overdue_steps),
                )

        self.starvation_counters[intersection_id] = counters
        return float(starvation_penalty), counters, demand_scales

    def _vehicle_travel_time_sums(self, resolved_end_time: Optional[float] = None) -> Dict[str, float]:
        if self.env is None:
            return {}
        fallback_time = float(
            self.current_time_s if resolved_end_time is None else resolved_end_time
        )
        vehicle_travel_times: Dict[str, float] = {}
        for inter in self.env.list_intersection:
            arrive_left_times = getattr(inter, "dic_vehicle_arrive_leave_time", {}) or {}
            for veh, times in arrive_left_times.items():
                if "shadow" in str(veh):
                    continue
                enter_time = times.get("enter_time")
                leave_time = times.get("leave_time")
                if enter_time is None or np.isnan(enter_time):
                    continue
                resolved_leave = leave_time if leave_time is not None and not np.isnan(leave_time) else fallback_time
                vehicle_travel_times[str(veh)] = vehicle_travel_times.get(str(veh), 0.0) + max(
                    0.0,
                    float(resolved_leave) - float(enter_time),
                )
        return vehicle_travel_times

    def episode_travel_time_mean(self, resolved_end_time: Optional[float] = None) -> float:
        vehicle_travel_times = self._vehicle_travel_time_sums(resolved_end_time=resolved_end_time)
        if not vehicle_travel_times:
            return 0.0
        return float(np.mean(list(vehicle_travel_times.values())))

    def _set_runtime_state(self, env: CityFlowEnv, traffic_file: str, reset_mode: str) -> None:
        self.env = env
        self.current_traffic_file = traffic_file
        self.intersection_ids = [inter.inter_name for inter in env.list_intersection]
        self._ensure_intersection_topology(env)
        self.prev_actions = [self._base_action_for_intersection(inter) for inter in env.list_intersection]
        self.current_time_s = float(env.get_current_time())
        self.done = False
        self.last_reset_mode = reset_mode
        self.active_vehicle_ids = self._collect_active_vehicle_ids(env)
        self.starvation_counters = {
            inter_id: [0 for _ in range(len(ACTION_LIST))]
            for inter_id in self.intersection_ids
        }
        self.prev_local_metrics_by_intersection = self._collect_local_metrics(env)

    def reset(self, traffic_file: str = "") -> Dict[str, Any]:
        resolved_traffic_file = str(traffic_file or self.default_traffic_file)
        env = self.get_env(resolved_traffic_file)
        self._apply_cityflow_seed(env, resolved_traffic_file)
        env.reset()
        self._mark_env_reset(resolved_traffic_file)
        self._refresh_measurements(env)
        self._set_runtime_state(env, resolved_traffic_file, "full")
        self._init_history_from_seed(seed_sample=None)
        return self.observe()

    def reset_from_snapshot(self, seed_sample: SnapshotSeedSample) -> Dict[str, Any]:
        env = self.get_env(seed_sample.traffic_file)
        self._apply_cityflow_seed(env, seed_sample.traffic_file)
        env.reset()
        self._mark_env_reset(seed_sample.traffic_file)
        snapshot_path = _resolve_existing_path(seed_sample.snapshot_path)
        env.eng.load_from_file(snapshot_path)
        self._restore_phase_state(env, snapshot_path)
        self._refresh_measurements(env)
        self._set_runtime_state(env, seed_sample.traffic_file, "snapshot")
        self._init_history_from_seed(seed_sample=seed_sample)
        return self.observe()

    def observe(self) -> Dict[str, Any]:
        sequences = [list(self.history_buffers[inter_id]) for inter_id in self.intersection_ids]
        prompts = [self.policy_prompt for _ in self.intersection_ids]
        return {
            "sequences": sequences,
            "prompts": prompts,
            "intersection_ids": list(self.intersection_ids),
            "current_time_s": float(self.current_time_s),
            "traffic_file": self.current_traffic_file,
            "reset_mode": self.last_reset_mode,
        }

    def current_metrics(self) -> Dict[str, float]:
        if self.env is None:
            return {"queue_sum": 0.0, "waiting_time_mean": 0.0, "active_vehicle_count": 0.0}
        queue_sum = float(
            sum(
                sum((inter.dic_feature or {}).get("lane_num_waiting_vehicle_in", []) or [])
                for inter in self.env.list_intersection
            )
        )
        waiting_times = [self.env.waiting_vehicle_list[veh]["time"] for veh in self.env.waiting_vehicle_list]
        waiting_time_mean = float(np.mean(waiting_times) if waiting_times else 0.0)
        waiting_time_p95 = float(np.percentile(waiting_times, 95) if waiting_times else 0.0)
        waiting_time_max = float(max(waiting_times) if waiting_times else 0.0)
        active_vehicle_count = float(len(self._collect_active_vehicle_ids(self.env)))
        return {
            "queue_sum": queue_sum,
            "waiting_time_mean": waiting_time_mean,
            "waiting_time_p95": waiting_time_p95,
            "waiting_time_max": waiting_time_max,
            "active_vehicle_count": active_vehicle_count,
        }

    def step(self, actions: Sequence[int]) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("TrajectoryCityFlowEnv is not reset.")
        if not self.prev_local_metrics_by_intersection:
            self.prev_local_metrics_by_intersection = self._collect_local_metrics(self.env)
        prev_active_vehicle_ids = set(self.active_vehicle_ids)
        prev_time_s = float(self.env.get_current_time())
        _, _, done, avg_rewards = self.env.step(list(actions))
        current_time_s = float(self.env.get_current_time())
        self._append_env_history(self.env, prev_time_s=prev_time_s, current_time_s=current_time_s)

        prev_actions = np.asarray(self.prev_actions, dtype=np.int64)
        action_arr = np.asarray(actions, dtype=np.int64)
        switch_penalty = (action_arr != prev_actions).astype(np.float32)
        current_local_metrics = self._collect_local_metrics(self.env)
        left_turn_action_indices = {
            action_idx for action_idx, action_name in enumerate(ACTION_LIST) if "L" in action_name
        }
        phase_wait_values: List[float] = []
        phase_queue_values: List[float] = []
        left_turn_wait_values: List[float] = []
        left_turn_queue_values: List[float] = []
        for local_metrics in current_local_metrics.values():
            phase_waits = local_metrics.get("phase_wait_means") or {}
            phase_queues = local_metrics.get("phase_queue_sums") or {}
            for action_idx in range(len(ACTION_LIST)):
                wait_value_for_phase = float(phase_waits.get(action_idx, 0.0) or 0.0)
                queue_value_for_phase = float(phase_queues.get(action_idx, 0.0) or 0.0)
                phase_wait_values.append(wait_value_for_phase)
                phase_queue_values.append(queue_value_for_phase)
                if action_idx in left_turn_action_indices:
                    left_turn_wait_values.append(wait_value_for_phase)
                    left_turn_queue_values.append(queue_value_for_phase)
        positive_phase_waits = [value for value in phase_wait_values if value > 0.0]
        positive_left_turn_waits = [value for value in left_turn_wait_values if value > 0.0]
        current_metrics = self.current_metrics()
        current_metrics["travel_time_mean"] = float(self.episode_travel_time_mean(resolved_end_time=current_time_s))
        current_active_vehicle_ids = self._collect_active_vehicle_ids(self.env)
        departed_vehicle_count = float(len(prev_active_vehicle_ids - current_active_vehicle_ids))
        rewards: List[float] = []
        queue_rewards: List[float] = []
        wait_rewards: List[float] = []
        travel_rewards: List[float] = []
        system_vehicle_rewards: List[float] = []
        throughput_rewards: List[float] = []
        starvation_penalties: List[float] = []
        switch_penalties: List[float] = []
        wait_signal_values: List[float] = []
        wait_delta_values: List[float] = []
        local_throughput_counts: List[float] = []
        queue_terms: List[float] = []
        wait_terms: List[float] = []
        active_vehicle_terms: List[float] = []
        throughput_terms: List[float] = []
        raw_rewards: List[float] = []
        clipped_rewards: List[float] = []

        for idx, intersection_id in enumerate(self.intersection_ids):
            previous_local = self.prev_local_metrics_by_intersection.get(intersection_id) or {}
            current_local = current_local_metrics.get(intersection_id) or {}
            queue_delta = float(current_local.get("queue_sum", 0.0)) - float(previous_local.get("queue_sum", 0.0))
            wait_value = self._resolve_wait_reward_value(current_local_metrics, intersection_id)
            previous_wait_value = self._resolve_wait_reward_value(
                self.prev_local_metrics_by_intersection,
                intersection_id,
            )
            wait_delta = float(wait_value - previous_wait_value)
            prev_active_count = float(previous_local.get("active_vehicle_count", 0.0))
            current_active_count = float(current_local.get("active_vehicle_count", 0.0))
            active_vehicle_delta = current_active_count - prev_active_count
            previous_active_ids = set(previous_local.get("active_vehicle_ids") or set())
            current_active_ids = set(current_local.get("active_vehicle_ids") or set())
            throughput_count = float(len(previous_active_ids - current_active_ids))
            starvation_penalty, _, _ = self._update_starvation_penalty(
                intersection_id=intersection_id,
                selected_action=int(action_arr[idx]),
                local_metrics=current_local,
            )
            shaped = compute_local_shaped_reward(
                queue_delta=queue_delta,
                wait_value=wait_value,
                active_vehicle_delta=active_vehicle_delta,
                throughput_count=throughput_count,
                queue_reward_coef=self.queue_reward_coef,
                waiting_reward_coef=self.waiting_reward_coef,
                system_vehicle_reward_coef=self.system_vehicle_reward_coef,
                throughput_reward_coef=self.throughput_reward_coef,
                switch_penalty=float(self.switch_penalty_beta * float(switch_penalty[idx])),
                starvation_penalty=starvation_penalty,
                objective_weights=self.objective_weights,
                wait_delta=wait_delta,
                reward_mode=self.reward_mode,
                queue_delta_scale=self.queue_delta_scale,
                wait_delta_scale=self.wait_delta_scale,
                wait_abs_coef=self.wait_abs_coef,
                wait_abs_scale=self.wait_abs_scale,
                active_delta_scale=self.active_delta_scale,
                throughput_scale=self.throughput_scale,
                reward_clip=self.reward_clip,
                global_reward_scale=self.global_reward_scale,
            )
            rewards.append(float(shaped["reward"]))
            queue_rewards.append(float(shaped["queue_reward"]))
            wait_rewards.append(float(shaped["wait_reward"]))
            travel_rewards.append(float(shaped["travel_reward"]))
            system_vehicle_rewards.append(float(shaped["system_vehicle_reward"]))
            throughput_rewards.append(float(shaped["throughput_reward"]))
            starvation_penalties.append(float(starvation_penalty))
            switch_penalties.append(float(self.switch_penalty_beta * float(switch_penalty[idx])))
            wait_signal_values.append(float(wait_value))
            wait_delta_values.append(float(wait_delta))
            local_throughput_counts.append(float(throughput_count))
            queue_terms.append(float(shaped["queue_term"]))
            wait_terms.append(float(shaped["wait_term"]))
            active_vehicle_terms.append(float(shaped["active_vehicle_term"]))
            throughput_terms.append(float(shaped["throughput_term"]))
            raw_rewards.append(float(shaped["raw_reward"]))
            clipped_rewards.append(float(shaped["clipped_reward"]))

        self.prev_actions = [int(item) for item in action_arr.tolist()]
        self.current_time_s = current_time_s
        self.done = bool(done or current_time_s >= self.run_counts)
        self.active_vehicle_ids = current_active_vehicle_ids
        self.prev_local_metrics_by_intersection = current_local_metrics
        info = {
            "base_rewards": np.asarray(avg_rewards, dtype=np.float32).tolist(),
            "switch_penalty": switch_penalties,
            "queue_rewards": queue_rewards,
            "queue_reward_mean": float(np.mean(queue_rewards) if queue_rewards else 0.0),
            "wait_rewards": wait_rewards,
            "wait_reward_mean": float(np.mean(wait_rewards) if wait_rewards else 0.0),
            "travel_rewards": travel_rewards,
            "travel_reward_mean": float(np.mean(travel_rewards) if travel_rewards else 0.0),
            "system_vehicle_rewards": system_vehicle_rewards,
            "system_vehicle_reward_mean": float(
                np.mean(system_vehicle_rewards) if system_vehicle_rewards else 0.0
            ),
            "throughput_rewards": throughput_rewards,
            "throughput_reward_mean": float(np.mean(throughput_rewards) if throughput_rewards else 0.0),
            "starvation_penalties": starvation_penalties,
            "starvation_penalty_mean": float(
                np.mean(starvation_penalties) if starvation_penalties else 0.0
            ),
            "wait_signal_values": wait_signal_values,
            "wait_signal_mean": float(np.mean(wait_signal_values) if wait_signal_values else 0.0),
            "wait_delta_values": wait_delta_values,
            "wait_delta_mean": float(np.mean(wait_delta_values) if wait_delta_values else 0.0),
            "phase_wait_mean": float(np.mean(positive_phase_waits) if positive_phase_waits else 0.0),
            "phase_wait_max": float(max(phase_wait_values) if phase_wait_values else 0.0),
            "phase_queue_mean": float(np.mean(phase_queue_values) if phase_queue_values else 0.0),
            "phase_queue_max": float(max(phase_queue_values) if phase_queue_values else 0.0),
            "left_turn_wait_mean": float(
                np.mean(positive_left_turn_waits) if positive_left_turn_waits else 0.0
            ),
            "left_turn_wait_max": float(max(left_turn_wait_values) if left_turn_wait_values else 0.0),
            "left_turn_queue_mean": float(
                np.mean(left_turn_queue_values) if left_turn_queue_values else 0.0
            ),
            "left_turn_queue_max": float(max(left_turn_queue_values) if left_turn_queue_values else 0.0),
            "local_throughput_counts": local_throughput_counts,
            "local_throughput_mean": float(np.mean(local_throughput_counts) if local_throughput_counts else 0.0),
            "queue_terms": queue_terms,
            "queue_term_mean": float(np.mean(queue_terms) if queue_terms else 0.0),
            "wait_terms": wait_terms,
            "wait_term_mean": float(np.mean(wait_terms) if wait_terms else 0.0),
            "active_vehicle_terms": active_vehicle_terms,
            "active_vehicle_term_mean": float(np.mean(active_vehicle_terms) if active_vehicle_terms else 0.0),
            "throughput_terms": throughput_terms,
            "throughput_term_mean": float(np.mean(throughput_terms) if throughput_terms else 0.0),
            "raw_rewards": raw_rewards,
            "raw_reward_mean": float(np.mean(raw_rewards) if raw_rewards else 0.0),
            "clipped_rewards": clipped_rewards,
            "clipped_reward_mean": float(np.mean(clipped_rewards) if clipped_rewards else 0.0),
            "reward_mode": self.reward_mode,
            "wait_reward_scope": self.wait_reward_scope,
            "wait_reward_reducer": self.wait_reward_reducer,
            "departed_vehicle_count": departed_vehicle_count,
            "objective_weights": dict(self.objective_weights),
            **current_metrics,
        }
        return self.observe(), rewards, self.done, info
