import argparse
import copy
import os
import random
import time

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

import torch

from projection_train.action_space import ACTION_LIST
from projection_train.run_projection_online import (
    ProjectionPolicyRunner,
    _dataset_spec,
    _resolve_device,
    _resolve_projection_runtime,
    _resolve_torch_dtype,
)
from utils import error
from utils.config import DIC_FIXEDTIME_AGENT_CONF, dic_traffic_env_conf


def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default="ProjectionLLMRun")
    parser.add_argument("--llm_model", type=str, default="")
    parser.add_argument("--projection_ckpt", type=str, required=True)
    parser.add_argument("--llm_path", type=str, default="")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS-extreme")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--prompt_mode", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--periodicity", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--prefix_tokens", type=int, default=0)
    parser.add_argument("--prefix_mode", type=str, default="")
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--traffic_series_interval", type=int, default=2)
    parser.add_argument("--traffic_series_window", type=int, default=0)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    return parser.parse_args()


def _resolve_fixed_time(num_actions: int, min_action_time: int) -> list[int]:
    configured = list(DIC_FIXEDTIME_AGENT_CONF.get("FIXED_TIME", []) or [])
    if len(configured) == num_actions:
        try:
            resolved = [int(item) for item in configured]
        except (TypeError, ValueError):
            resolved = []
        if resolved and all(item > 0 for item in resolved):
            return resolved
    return [int(min_action_time) for _ in range(num_actions)]


def main(in_args):
    if in_args.fp16 and in_args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")
    if in_args.eightphase:
        raise ValueError("Projection checkpoints only support the four-phase action space from utils/config.py.")

    random.seed(2024)
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

    default_count, road_net, traffic_file_list, template = _dataset_spec(in_args.dataset)
    if in_args.traffic_file not in traffic_file_list:
        raise error.flowFileException("Flow file does not exist.")

    runtime, prompt_mode = _resolve_projection_runtime(in_args)
    run_counts = int(in_args.run_counts or default_count)
    if "24h" in in_args.traffic_file:
        run_counts = 86400

    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])
    num_intersections = num_row * num_col

    min_action_time = int(dic_traffic_env_conf.get("MIN_ACTION_TIME", 30) or 30)
    measure_time = int(dic_traffic_env_conf.get("MEASURE_TIME", min_action_time) or min_action_time)
    traffic_series_interval = max(1, int(in_args.traffic_series_interval or 2))
    traffic_series_window = int(in_args.traffic_series_window or 0)
    if traffic_series_window <= 0:
        traffic_series_window = max(traffic_series_interval, (runtime.seq_len - 1) * traffic_series_interval)

    llm_model_name = str(in_args.llm_model or "").strip()
    if not llm_model_name:
        llm_model_name = os.path.basename(runtime.llm_path.rstrip("/")) or "projection-llm"

    dic_agent_conf_extra = {
        "LLM_PATH": runtime.llm_path,
        "LLM_MODEL": llm_model_name,
        "LOG_DIR": f"./{llm_model_name}_logs",
        "FIXED_TIME": _resolve_fixed_time(len(ACTION_LIST), min_action_time),
        "TRAFFIC_SERIES_INTERVAL": traffic_series_interval,
        "TRAFFIC_SERIES_WINDOW": traffic_series_window,
        "PROJECTION_CKPT": in_args.projection_ckpt,
        "PROMPT_MODE": prompt_mode,
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "MODEL_NAME": f"{in_args.memo}-{llm_model_name}",
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": run_counts,
        "NUM_ROUNDS": in_args.num_rounds,
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "ENABLE_TRAFFIC_COUNT": True,
        "TRAFFIC_COUNT_MODE": "intersection_movement",
        "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
        "TRAFFIC_COUNT_INTERVALS": sorted({traffic_series_interval, measure_time, 60, 3600}),
        "TRAFFIC_COUNT_ADD_TOTAL": True,
        "TRAFFIC_COUNT_BASENAME": "traffic_counts",
        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": f"roadnet_{road_net}.json",
        "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
        "DIC_REWARD_INFO": {"queue_length": -0.25},
    }

    timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", in_args.memo, in_args.traffic_file + "_" + timestamp),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", in_args.memo, in_args.traffic_file + "_" + timestamp),
        "PATH_TO_DATA": os.path.join("data", template, road_net),
    }

    device = _resolve_device(in_args.device)
    torch_dtype = _resolve_torch_dtype(in_args, device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    runner = ProjectionPolicyRunner(
        dic_agent_conf=dic_agent_conf_extra,
        dic_traffic_env_conf_local=merge(dic_traffic_env_conf, dic_traffic_env_conf_extra),
        dic_path=dic_path_extra,
        roadnet=f"{template}-{road_net}",
        trafficflow=in_args.traffic_file.split(".")[0],
        runtime=runtime,
        projection_ckpt=in_args.projection_ckpt,
        prompt_mode=prompt_mode,
        device=device,
        torch_dtype=torch_dtype,
        use_wandb=in_args.use_wandb,
        record_snapshots=False,
        record_traffic_series=False,
        dump_cityflow_env_args=False,
    )
    runner.train_test()


if __name__ == "__main__":
    args = parse_args()
    main(args)
