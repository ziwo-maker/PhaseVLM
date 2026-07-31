"""Run the standalone Rule-only online traffic signal controller."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
import shutil
import time
from typing import Any

import numpy as np

from models.rule_only_agent import (
    action_index_from_cur_phase,
    choose_rule_action,
    phase_queue_wait_from_state,
)
from projection_train.action_space import format_action_tag, index_to_action
from utils import error
from utils.config import dic_traffic_env_conf

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, **kwargs):
        return iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default="RuleOnly")
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS-extreme")
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--rule_gap", type=float, default=2.0)
    parser.add_argument("--left_wait_protect", type=float, default=60.0)
    parser.add_argument("--target_queue_floor", type=float, default=0.0)
    parser.add_argument("--total_queue_floor", type=float, default=0.0)
    parser.add_argument("--traffic_series_interval", type=int, default=30)
    parser.add_argument("--result_json", type=str, default="")
    parser.add_argument("--disable_replay", action="store_true", default=False)
    parser.add_argument("--disable_progress", action="store_true", default=False)
    return parser.parse_args()


def _dataset_spec(dataset: str) -> tuple[int, str, list[str], str]:
    dataset = str(dataset).strip().lower()
    if dataset == "jinan":
        return (
            3600,
            "3_4",
            [
                "anon_3_4_jinan_real.json",
                "anon_3_4_jinan_real_2000.json",
                "anon_3_4_jinan_real_2500.json",
                "anon_3_4_jinan_synthetic_24000_60min.json",
                "anon_3_4_jinan_synthetic_24h_6000.json",
            ],
            "Jinan",
        )
    if dataset == "hangzhou":
        return (
            3600,
            "4_4",
            [
                "anon_4_4_hangzhou_real.json",
                "anon_4_4_hangzhou_real_5816.json",
                "anon_4_4_hangzhou_synthetic_24000_60min.json",
            ],
            "Hangzhou",
        )
    if dataset == "newyork_16x3":
        return (3600, "16_3", ["anon_16_3_newyork_real.json"], "NewYork")
    if dataset == "newyork_28x7":
        return (
            3600,
            "28_7",
            ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"],
            "NewYork",
        )
    raise error.flowFileException(f"Unknown dataset: {dataset}")


def _merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    merged.update(extra)
    return merged


def path_check(dic_path: dict[str, str]) -> None:
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError(dic_path["PATH_TO_WORK_DIRECTORY"])
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])

    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError(dic_path["PATH_TO_MODEL"])
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(
    dic_path: dict[str, str],
    dic_agent_conf: dict[str, Any],
    dic_traffic_env_conf_local: dict[str, Any],
) -> None:
    with open(os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "agent.conf"), "w", encoding="utf-8") as handle:
        json.dump(dic_agent_conf, handle, indent=4)
    with open(os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "traffic_env.conf"), "w", encoding="utf-8") as handle:
        json.dump(dic_traffic_env_conf_local, handle, indent=4)


def copy_cityflow_file(dic_path: dict[str, str], dic_traffic_env_conf_local: dict[str, Any]) -> None:
    shutil.copy(
        os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf_local["TRAFFIC_FILE"]),
        os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf_local["TRAFFIC_FILE"]),
    )
    shutil.copy(
        os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf_local["ROADNET_FILE"]),
        os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], dic_traffic_env_conf_local["ROADNET_FILE"]),
    )


def setup_logger(log_path: str) -> logging.Logger:
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    run_tag = os.path.basename(os.path.dirname(log_path)) or "ruleonly"
    logger = logging.getLogger(f"ruleonly.{run_tag}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _write_json(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _build_paths(args: argparse.Namespace, template: str, road_net: str) -> dict[str, str]:
    timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    return {
        "PATH_TO_MODEL": os.path.join("model", args.memo, f"{args.traffic_file}_{timestamp}"),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, f"{args.traffic_file}_{timestamp}"),
        "PATH_TO_DATA": os.path.join("data", template, road_net),
    }


def run_ruleonly(args: argparse.Namespace) -> dict[str, Any]:
    from utils.cityflow_env import CityFlowEnv
    from utils.my_utils import dump_json, eight_phase_list, get_state_detail

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    default_count, road_net, traffic_files, template = _dataset_spec(args.dataset)
    if args.traffic_file not in traffic_files:
        raise error.flowFileException(f"Flow file does not exist: {args.traffic_file}")

    run_counts = int(args.run_counts or default_count)
    if "24h" in args.traffic_file:
        run_counts = 86400

    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])
    num_intersections = num_row * num_col
    min_action_time = int(args.min_action_time or 30)
    traffic_series_interval = max(1, int(args.traffic_series_interval or min_action_time))

    dic_agent_conf = {
        "FIXED_TIME": [min_action_time for _ in range(4)],
        "RULE_GAP": float(args.rule_gap),
        "LEFT_WAIT_PROTECT": float(args.left_wait_protect),
        "TARGET_QUEUE_FLOOR": float(args.target_queue_floor or 0.0),
        "TOTAL_QUEUE_FLOOR": float(args.total_queue_floor or 0.0),
    }
    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "MODEL_NAME": "RuleOnly",
        "PROJECT_NAME": args.proj_name,
        "RUN_COUNTS": run_counts,
        "NUM_ROUNDS": int(args.num_rounds),
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "CITYFLOW_SEED": int(args.seed),
        "SAVE_REPLAY": not bool(args.disable_replay),
        "MIN_ACTION_TIME": min_action_time,
        "MEASURE_TIME": min_action_time,
        "ENABLE_TRAFFIC_COUNT": True,
        "TRAFFIC_COUNT_MODE": "intersection_movement",
        "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
        "TRAFFIC_COUNT_INTERVALS": sorted({traffic_series_interval, min_action_time, 60, 3600}),
        "TRAFFIC_COUNT_ADD_TOTAL": True,
        "TRAFFIC_COUNT_BASENAME": "traffic_counts",
        "TRAFFIC_FILE": args.traffic_file,
        "ROADNET_FILE": f"roadnet_{road_net}.json",
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "time_this_phase",
            "lane_num_waiting_vehicle_in",
            "traffic_movement_pressure_queue",
        ],
        "DIC_REWARD_INFO": {"queue_length": -0.25},
    }
    dic_traffic_env_conf_local = _merge(dic_traffic_env_conf, dic_traffic_env_conf_extra)
    dic_path = _build_paths(args, template=template, road_net=road_net)

    path_check(dic_path)
    copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf_local)
    copy_cityflow_file(dic_path, dic_traffic_env_conf_local)
    log_path = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "ruleonly_test.log")
    logger = setup_logger(log_path)
    logger.info(
        "Initialized Rule-only run: dataset=%s traffic=%s gap=%.3f protect=%.3f",
        args.dataset,
        args.traffic_file,
        float(args.rule_gap),
        float(args.left_wait_protect),
    )

    env = CityFlowEnv(
        path_to_log=dic_path["PATH_TO_WORK_DIRECTORY"],
        path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
        dic_traffic_env_conf=dic_traffic_env_conf_local,
        dic_path=dic_path,
    )

    done = False
    state = env.reset()
    current_time = env.get_current_time()
    total_reward = 0.0
    queue_length_episode: list[float] = []
    waiting_time_episode: list[float] = []
    state_action_log = [[] for _ in range(num_intersections)]
    last_actions = [0 for _ in range(num_intersections)]
    total_steps = int(run_counts / min_action_time)
    start_time = time.time()

    for step_num in tqdm(range(total_steps), disable=bool(args.disable_progress)):
        if done or current_time >= run_counts:
            break

        action_list: list[int] = []
        for idx in range(num_intersections):
            intersection = env.intersection_dict[env.list_intersection[idx].inter_name]
            roads = copy.deepcopy(intersection["roads"])
            statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, env)
            phase_queue, phase_wait = phase_queue_wait_from_state(statistic_state)
            original_idx = action_index_from_cur_phase(state[idx].get("cur_phase", [0]), fallback=last_actions[idx])
            action_idx, action_name, diagnostic = choose_rule_action(
                phase_queue=phase_queue,
                phase_wait=phase_wait,
                original_action=index_to_action(original_idx),
                rule_gap=float(args.rule_gap),
                left_wait_protect=float(args.left_wait_protect),
                target_queue_floor=float(args.target_queue_floor or 0.0),
                total_queue_floor=float(args.total_queue_floor or 0.0),
            )
            action_list.append(action_idx)
            last_actions[idx] = action_idx
            state_action_log[idx].append(
                {
                    "state": statistic_state,
                    "state_incoming": statistic_state_incoming,
                    "approaching_speed": mean_speed,
                    "response": format_action_tag(action_name),
                    "action": eight_phase_list[action_idx],
                    "rule_only": diagnostic,
                }
            )

        next_state, reward, done, _ = env.step(action_list)
        current_time = env.get_current_time()
        state = next_state

        total_reward += sum(reward)
        queue_length_inter = [sum(inter.dic_feature["lane_num_waiting_vehicle_in"]) for inter in env.list_intersection]
        queue_length_episode.append(sum(queue_length_inter))
        waiting_times = [env.waiting_vehicle_list[veh]["time"] for veh in env.waiting_vehicle_list]
        waiting_time_episode.append(np.mean(waiting_times) if waiting_times else 0.0)
        logger.info("step=%d time=%s queued_sum=%s", step_num, current_time, sum(queue_length_episode))

    vehicle_travel_times: dict[str, list[float]] = {}
    for inter in env.list_intersection:
        for veh, times in inter.dic_vehicle_arrive_leave_time.items():
            if "shadow" in veh:
                continue
            enter_time = times["enter_time"]
            leave_time = times["leave_time"]
            if not np.isnan(enter_time):
                resolved_leave = leave_time if not np.isnan(leave_time) else dic_traffic_env_conf_local["RUN_COUNTS"]
                vehicle_travel_times.setdefault(veh, []).append(resolved_leave - enter_time)

    total_travel_time = (
        np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
        if vehicle_travel_times
        else 0.0
    )
    results = {
        "test_reward_over": float(total_reward),
        "test_avg_queue_len_over": float(np.mean(queue_length_episode) if queue_length_episode else 0.0),
        "test_queuing_vehicle_num_over": float(np.sum(queue_length_episode) if queue_length_episode else 0.0),
        "test_avg_waiting_time_over": float(np.mean(waiting_time_episode) if waiting_time_episode else 0.0),
        "test_avg_travel_time_over": float(total_travel_time),
    }
    logger.info("Test results: %s", json.dumps(results, ensure_ascii=False, sort_keys=True))

    state_action_path = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
    dump_json(state_action_log, state_action_path)
    env.batch_log_2()
    env.end_cityflow()
    logger.info("Testing time: %.4f", time.time() - start_time)

    payload = {
        "results": results,
        "args": vars(args),
        "work_dir": dic_path["PATH_TO_WORK_DIRECTORY"],
        "model_dir": dic_path["PATH_TO_MODEL"],
        "data_dir": dic_path["PATH_TO_DATA"],
        "log_path": log_path,
        "state_action_path": state_action_path,
    }
    _write_json(args.result_json, payload)
    return payload


if __name__ == "__main__":
    run_ruleonly(parse_args())
