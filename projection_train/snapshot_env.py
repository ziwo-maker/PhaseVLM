from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from functools import lru_cache
from typing import Dict, Tuple

from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf


def _merge_configs(base: Dict, override: Dict) -> Dict:
    merged = deepcopy(base)
    merged.update(override)
    return merged


@lru_cache(maxsize=1024)
def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


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


class SnapshotActionEvaluator:
    def __init__(
        self,
        roadnet_path: str,
        traffic_root: str,
        work_root: str,
        min_action_time: int = 30,
        yellow_time: int = 5,
        interval: float = 1.0,
        wait_speed: float = 1.0,
        horizon_s: int = 30,
    ) -> None:
        self.roadnet_path = roadnet_path
        self.traffic_root = traffic_root
        self.work_root = work_root
        self.min_action_time = int(min_action_time)
        self.yellow_time = int(yellow_time)
        self.interval = float(interval)
        self.wait_speed = float(wait_speed)
        self.horizon_s = int(horizon_s)
        self.num_col, self.num_row = _infer_grid_shape(roadnet_path)
        self.phase_time_lookup = _build_phase_time_lookup(roadnet_path)
        self._env_cache: Dict[str, CityFlowEnv] = {}

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
                "MODEL_NAME": "projection-ppo-online",
                "PROJECT_NAME": "projection-ppo-online",
                "NUM_AGENTS": self.num_col * self.num_row,
                "NUM_INTERSECTIONS": self.num_col * self.num_row,
                "NUM_COL": self.num_col,
                "NUM_ROW": self.num_row,
                "RUN_COUNTS": max(self.min_action_time, self.horizon_s),
                "MIN_ACTION_TIME": self.min_action_time,
                "MEASURE_TIME": self.min_action_time,
                "YELLOW_TIME": self.yellow_time,
                "INTERVAL": self.interval,
                "TRAFFIC_FILE": traffic_file,
                "ROADNET_FILE": os.path.basename(self.roadnet_path),
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
            if 0 <= candidate < 4:
                return candidate
        next_idx = inter.next_phase_to_set_index
        if isinstance(next_idx, int) and next_idx > 0:
            candidate = next_idx - 1
            if 0 <= candidate < 4:
                return candidate
        return 0

    def _apply_action_no_log(self, inter, action: int) -> None:
        action = int(action)
        if inter.all_yellow_flag:
            inter.flicker = 0
            if inter.current_phase_duration >= self.yellow_time:
                next_idx = inter.next_phase_to_set_index
                if isinstance(next_idx, int):
                    inter.current_phase_index = next_idx
                    inter.eng.set_tl_phase(inter.inter_name, inter.current_phase_index)
                inter.all_yellow_flag = False
            return

        inter.next_phase_to_set_index = action + 1
        if inter.current_phase_index == inter.next_phase_to_set_index:
            return

        inter.eng.set_tl_phase(inter.inter_name, 0)
        inter.current_phase_index = inter.all_yellow_phase_index
        inter.all_yellow_flag = True
        inter.flicker = 1

    def simulate_action(
        self,
        snapshot_path: str,
        traffic_file: str,
        intersection_id: str,
        action_index: int,
    ) -> Dict[str, float]:
        env = self.get_env(traffic_file)
        if intersection_id not in env.id_to_index:
            raise KeyError(f"Intersection not found in environment: {intersection_id}")

        env.eng.load_from_file(snapshot_path)
        self._restore_phase_state(env, snapshot_path)

        inter_idx = env.id_to_index[intersection_id]
        base_actions = [self._base_action_for_intersection(inter) for inter in env.list_intersection]
        action_list = list(base_actions)
        action_list[inter_idx] = int(action_index)

        target_inter = env.list_intersection[inter_idx]
        entering_lanes = list(target_inter.list_entering_lanes)
        substeps = max(1, int(round(1.0 / max(self.interval, 1e-6))))
        total_wait = 0.0
        unique_waiting = set()

        for _ in range(self.horizon_s):
            for inter in env.list_intersection:
                inter.previous_phase_index = inter.current_phase_index

            for idx, inter in enumerate(env.list_intersection):
                self._apply_action_no_log(inter, action_list[idx])

            for _ in range(substeps):
                env.eng.next_step()

            for inter in env.list_intersection:
                if inter.current_phase_index == inter.previous_phase_index:
                    inter.current_phase_duration += 1
                else:
                    inter.current_phase_duration = 1

            lane_vehicles = env.eng.get_lane_vehicles()
            vehicle_speed = env.eng.get_vehicle_speed()
            for lane_id in entering_lanes:
                for vehicle_id in lane_vehicles.get(lane_id, []):
                    if float(vehicle_speed.get(vehicle_id, 0.0) or 0.0) < self.wait_speed:
                        total_wait += 1.0
                        unique_waiting.add(vehicle_id)

        avg_wait = total_wait / len(unique_waiting) if unique_waiting else 0.0
        reward = -avg_wait / max(1, self.horizon_s)
        return {"reward": reward, "avg_wait": avg_wait}
