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
        yellow_time: int = 5,
        interval: float = 1.0,
        reward_alpha: float = 0.1,
        switch_penalty_beta: float = 0.02,
        reward_info: Optional[Dict[str, float]] = None,
        run_counts: int = 3600,
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
        self.reward_info = dict(reward_info or {"queue_length": -0.25})
        self.run_counts = int(run_counts)

        self.num_col, self.num_row = _infer_grid_shape(self.roadnet_path)
        self.phase_time_lookup = _build_phase_time_lookup(self.roadnet_path)
        self._env_cache: Dict[str, CityFlowEnv] = {}

        self.env: Optional[CityFlowEnv] = None
        self.current_traffic_file = self.default_traffic_file
        self.intersection_ids: List[str] = []
        self.history_buffers: Dict[str, Deque[List[float]]] = {}
        self.prev_actions: List[int] = []
        self.current_time_s: float = 0.0
        self.done: bool = False
        self.last_reset_mode: str = "full"
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
        env.reset()
        return env

    def get_env(self, traffic_file: str) -> CityFlowEnv:
        if traffic_file not in self._env_cache:
            self._env_cache[traffic_file] = self._create_env(traffic_file)
        return self._env_cache[traffic_file]

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

    def _set_runtime_state(self, env: CityFlowEnv, traffic_file: str, reset_mode: str) -> None:
        self.env = env
        self.current_traffic_file = traffic_file
        self.intersection_ids = [inter.inter_name for inter in env.list_intersection]
        self.prev_actions = [self._base_action_for_intersection(inter) for inter in env.list_intersection]
        self.current_time_s = float(env.get_current_time())
        self.done = False
        self.last_reset_mode = reset_mode

    def reset(self, traffic_file: str = "") -> Dict[str, Any]:
        resolved_traffic_file = str(traffic_file or self.default_traffic_file)
        env = self.get_env(resolved_traffic_file)
        env.reset()
        self._refresh_measurements(env)
        self._set_runtime_state(env, resolved_traffic_file, "full")
        self._init_history_from_seed(seed_sample=None)
        return self.observe()

    def reset_from_snapshot(self, seed_sample: SnapshotSeedSample) -> Dict[str, Any]:
        env = self.get_env(seed_sample.traffic_file)
        env.reset()
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
            return {"queue_sum": 0.0, "waiting_time_mean": 0.0}
        queue_sum = float(
            sum(
                sum((inter.dic_feature or {}).get("lane_num_waiting_vehicle_in", []) or [])
                for inter in self.env.list_intersection
            )
        )
        waiting_times = [self.env.waiting_vehicle_list[veh]["time"] for veh in self.env.waiting_vehicle_list]
        waiting_time_mean = float(np.mean(waiting_times) if waiting_times else 0.0)
        return {
            "queue_sum": queue_sum,
            "waiting_time_mean": waiting_time_mean,
        }

    def step(self, actions: Sequence[int]) -> Tuple[Dict[str, Any], List[float], bool, Dict[str, Any]]:
        if self.env is None:
            raise RuntimeError("TrajectoryCityFlowEnv is not reset.")
        prev_time_s = float(self.env.get_current_time())
        _, _, done, avg_rewards = self.env.step(list(actions))
        current_time_s = float(self.env.get_current_time())
        self._append_env_history(self.env, prev_time_s=prev_time_s, current_time_s=current_time_s)

        local_rewards = np.asarray(avg_rewards, dtype=np.float32)
        global_mean_reward = float(local_rewards.mean()) if local_rewards.size > 0 else 0.0
        prev_actions = np.asarray(self.prev_actions, dtype=np.int64)
        action_arr = np.asarray(actions, dtype=np.int64)
        switch_penalty = (action_arr != prev_actions).astype(np.float32)
        shaped_rewards = local_rewards + self.reward_alpha * global_mean_reward - self.switch_penalty_beta * switch_penalty

        self.prev_actions = [int(item) for item in action_arr.tolist()]
        self.current_time_s = current_time_s
        self.done = bool(done or current_time_s >= self.run_counts)
        info = {
            "local_rewards": local_rewards.tolist(),
            "global_mean_reward": global_mean_reward,
            "switch_penalty": switch_penalty.tolist(),
            **self.current_metrics(),
        }
        return self.observe(), shaped_rewards.tolist(), self.done, info

