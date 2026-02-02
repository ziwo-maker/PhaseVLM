"""
基于 PPO 对时间序列到图像的可学习适配器进行投影训练。
流程：读取交通数据样本 -> 构建单路口 CityFlow 环境 -> 生成图像 -> LLM 策略输出动作 -> 计算等待时间奖励 -> PPO 更新。
"""
import cityflow
import argparse
import json
import os
import random
import shutil
import sys
import time
import math
from copy import deepcopy
from functools import lru_cache
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

# Ensure repo root is on sys.path so `import src.*` works when run from subdir.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.TimeVLM.ts_image_adapter import TimeSeriesImageAdapter
from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf
from utils.my_utils import merge

from projection_train.jsonl_dataset import TrafficPromptSeriesDataset
from projection_train.policy import GemmaVLPolicy, ACTION_LIST


def seed_everything(seed: int) -> None:
    """固定随机种子，保证可复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_paths(base_dir: str, traffic_file: str, timestamp: Optional[str] = None) -> dict:
    """创建模型与日志目录，并返回 CityFlow 所需的路径字典。"""
    if timestamp is None:
        timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    model_dir = os.path.join("model", "projection_train", f"{traffic_file}_{timestamp}")
    work_dir = os.path.join("records", "projection_train", f"{traffic_file}_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)
    return {
        "PATH_TO_MODEL": model_dir,
        "PATH_TO_WORK_DIRECTORY": work_dir,
        "PATH_TO_DATA": base_dir,
    }


def resolve_rank_work_dir(base_work_dir: str, local_rank: int) -> str:
    """为每个进程生成独立的 CityFlow 工作目录，避免并发写冲突。"""
    if dist.is_initialized():
        return os.path.join(base_work_dir, f"rank{dist.get_rank()}")
    if local_rank >= 0:
        return os.path.join(base_work_dir, f"rank{local_rank}")
    return os.path.join(base_work_dir, f"pid{os.getpid()}")


def copy_cityflow_files(dic_path: dict, dic_traffic_env_conf_local: dict) -> None:
    """将 roadnet/traffic 文件复制到 CityFlow 工作目录。"""
    dst = dic_path["PATH_TO_WORK_DIRECTORY"]
    src = dic_path["PATH_TO_DATA"]
    os.makedirs(dst, exist_ok=True)
    shutil.copy(os.path.join(src, dic_traffic_env_conf_local["TRAFFIC_FILE"]), dst)
    shutil.copy(os.path.join(src, dic_traffic_env_conf_local["ROADNET_FILE"]), dst)


def build_env_conf(
    traffic_file: str,
    roadnet_file: str,
    run_counts: int,
    min_action_time: int,
    num_row: int,
    num_col: int,
    num_intersections: int,
) -> dict:
    """构建仿真配置，并覆盖默认配置中的关键字段。"""
    override = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "MODEL_NAME": "projection-train-grid",
        "PROJECT_NAME": "projection-train-grid",
        "RUN_COUNTS": run_counts,
        "NUM_ROUNDS": 1,
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": roadnet_file,
        "MIN_ACTION_TIME": min_action_time,
    }
    return merge(dic_traffic_env_conf, override)


def _load_vehicle_template(src_traffic: str) -> dict:
    if src_traffic and os.path.exists(src_traffic):
        try:
            with open(src_traffic, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list) and data and isinstance(data[0], dict):
                vehicle = data[0].get("vehicle")
                if isinstance(vehicle, dict) and vehicle:
                    return deepcopy(vehicle)
        except Exception:
            pass
    return {
        "length": 5.0,
        "width": 2.0,
        "maxPosAcc": 2.0,
        "maxNegAcc": 4.5,
        "usualPosAcc": 2.0,
        "usualNegAcc": 4.5,
        "minGap": 2.5,
        "maxSpeed": 11.111,
        "headwayTime": 2,
    }


def _build_lane_route_map(roadnet: dict, intersection_id: str) -> dict:
    inter = None
    for item in roadnet.get("intersections", []):
        if item.get("id") == intersection_id:
            inter = item
            break
    if inter is None:
        raise ValueError(f"Intersection not found in roadnet: {intersection_id}")
    lane_route = {}
    for link in inter.get("roadLinks", []) or []:
        start_road = link.get("startRoad")
        end_road = link.get("endRoad")
        if not start_road or not end_road:
            continue
        for lane_link in link.get("laneLinks", []) or []:
            lane_idx = lane_link.get("startLaneIndex")
            if lane_idx is None:
                continue
            lane_id = f"{start_road}_{lane_idx}"
            lane_route.setdefault(lane_id, [start_road, end_road])
    return lane_route


def _infer_grid_size(intersection_ids: list[str]) -> tuple[int, int]:
    max_row = 0
    max_col = 0
    for inter_id in intersection_ids:
        if not inter_id:
            continue
        parts = str(inter_id).split("_")
        if len(parts) >= 3 and parts[0] == "intersection":
            try:
                row = int(parts[1])
                col = int(parts[2])
            except ValueError:
                continue
            max_row = max(max_row, row)
            max_col = max(max_col, col)
    if max_row <= 0 or max_col <= 0:
        # fallback to 1x1 if ids are not in expected format
        return 1, 1
    return max_row, max_col


def _flows_from_lane_details(lane_details: dict, lane_route_map: dict, vehicle_template: dict) -> tuple[list, list]:
    flows = []
    missing = []
    if not isinstance(lane_details, dict):
        return flows, missing
    for movement, lane_map in lane_details.items():
        if not isinstance(lane_map, dict):
            continue
        for lane_id, veh_list in lane_map.items():
            count = len(veh_list or [])
            if count <= 0:
                continue
            route = lane_route_map.get(lane_id)
            if not route:
                missing.append(lane_id)
                continue
            for _ in range(count):
                flows.append(
                    {
                        "vehicle": deepcopy(vehicle_template),
                        "route": route,
                        "interval": 1.0,
                        "startTime": 0,
                        "endTime": 0,
                    }
                )
    return flows, missing


def _write_traffic_file(path: str, flows: list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(flows, handle, ensure_ascii=False, indent=2)


@lru_cache(maxsize=32)
def _load_incoming_lane_ids(intersection_id: str) -> list[str]:
    """Load incoming lane ids for an intersection from the fixed roadnet file."""
    roadnet_path = os.path.join(_REPO_ROOT, 'data', 'Jinan', '3_4', 'roadnet_3_4.json')
    with open(roadnet_path, 'r', encoding='utf-8') as handle:
        roadnet = json.load(handle)
    if not any(inter.get('id') == intersection_id for inter in roadnet.get('intersections', []) or []):
        raise ValueError(f'Intersection not found in roadnet: {intersection_id}')
    lane_ids = []
    for road in roadnet.get('roads', []) or []:
        if road.get('endIntersection') != intersection_id:
            continue
        road_id = road.get('id')
        if not road_id:
            continue
        lanes = road.get('lanes') or []
        for idx in range(len(lanes)):
            lane_ids.append(f'{road_id}_{idx}')
    return lane_ids


def mean_waiting_time(env, intersection_id: str) -> float:
    """Compute per-step average waiting ratio over [0, 30) with dt=1s."""
    lane_ids = _load_incoming_lane_ids(intersection_id)
    total_wait = 0.0
    unique_vehicles = set()
    dt = 1.0
    for _ in range(30):
        lane_vehicles = env.get_lane_vehicles()
        vehicle_speed = env.get_vehicle_speed()
        vehicles = set()
        for lane_id in lane_ids:
            vehicles.update(lane_vehicles.get(lane_id, []))
        unique_vehicles.update(vehicles)
        for veh_id in vehicles:
            speed = vehicle_speed.get(veh_id, 0.0)
            if speed < 1:
                total_wait += dt
        env.next_step()
    total_count = len(unique_vehicles)
    if total_count == 0:
        return 0.0
    return total_wait / total_count


def discounted_returns(rewards: list[float], gamma: float) -> torch.Tensor:
    """根据奖励序列计算折扣回报（从后往前累加）。"""
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--jsonl_path",
        type=str,
        default="/home/code/LLMSTSC/data/recorded/anon_3_4_jinan_real.json_01_30_04_11_52/traffic_prompt_series_2s.jsonl",
    )
    parser.add_argument("--intersection_id", type=str, default="intersection_1_1")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--interval_s", type=int, default=2)
    parser.add_argument("--llm_path", type=str, default="/home/data/model/gemma-12B/")
    parser.add_argument(
        "--train_device_ids",
        type=str,
        default="1",
        help="Comma-separated CUDA device ids for training the time-series image adapter.",
    )
    parser.add_argument(
        "--llm_device_ids",
        type=str,
        default="0,1",
        help="Comma-separated CUDA device ids for LLM loading/sharding.",
    )
    parser.add_argument("--seed", type=int, default=2024)

    parser.add_argument(
        "--roadnet_src",
        type=str,
        default="/home/code/LLMSTSC/data/Jinan/3_4/roadnet_3_4.json",
    )
    parser.add_argument("--traffic_src", type=str, default="data/Jinan/3_4/anon_3_4_jinan_real.json")
    parser.add_argument("--output_dir", type=str, default="data/Jinan/1_1")
    parser.add_argument("--roadnet_out", type=str, default="roadnet_1_1.json")
    parser.add_argument("--traffic_out", type=str, default="traffic_1_1.json")
    parser.add_argument("--flow_interval", type=float, default=2.0)

    parser.add_argument("--min_action_time", type=int, default=60)
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument(
        "--ppo_model_path",
        type=str,
        default="/home/code/LLMSTSC/model/projection_train/anon_3_4_jinan_real.json_01_30_09_42_15/round_0/ts_image_adapter.pt",
        help="PPO projection model path. If set, skip training and run evaluation only.",
    )

    parser.add_argument("--image_size", type=int, default=896)
    parser.add_argument("--periodicity", type=int, default=24)
    parser.add_argument("--norm_const", type=float, default=0.4)
    parser.add_argument("--three_channel", action="store_true", default=True)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--mini_batch_size", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.2)

    parser.add_argument("--max_new_tokens", type=int, default=1500)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="每隔多少步输出一次中文进度日志（采样与 PPO 更新共用）。",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _shorten(text: str, max_len: int = 60) -> str:
    text = str(text)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _is_rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def _broadcast_object(value):
    if not dist.is_initialized():
        return value
    obj_list = [value]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _resolve_snapshot_path(snapshot_path: str, jsonl_path: str) -> Optional[str]:
    if not snapshot_path:
        return None
    candidates = []
    if os.path.isabs(snapshot_path):
        candidates.append(snapshot_path)
    else:
        candidates.append(snapshot_path)
        if jsonl_path:
            candidates.append(os.path.join(os.path.dirname(jsonl_path), snapshot_path))
        candidates.append(os.path.join(_REPO_ROOT, snapshot_path))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def _resolve_model_path(model_path: str, jsonl_path: str) -> Optional[str]:
    if not model_path:
        return None
    candidates = []
    if os.path.isabs(model_path):
        candidates.append(model_path)
    else:
        candidates.append(model_path)
        if jsonl_path:
            candidates.append(os.path.join(os.path.dirname(jsonl_path), model_path))
        candidates.append(os.path.join(_REPO_ROOT, model_path))
    for cand in candidates:
        if cand and os.path.exists(cand):
            return cand
    return None


def _split_indices(total: int, test_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if total <= 0:
        return [], []
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)
    test_size = max(1, int(round(total * test_ratio)))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    if not train_indices:
        train_indices = test_indices
        test_indices = indices[: max(1, int(round(total * test_ratio)))]
    return train_indices, test_indices


def evaluate_policy(
    dataset: TrafficPromptSeriesDataset,
    indices: list[int],
    ts_adapter: TimeSeriesImageAdapter,
    policy: GemmaVLPolicy,
    device: torch.device,
    eng,
    intersection_ids: list[str],
    intersection_id_to_index: dict,
    intersections_count: int,
    generation_kwargs: dict,
    log_interval: int,
) -> dict:
    rewards = []
    total_samples = len(indices)
    sample_counter = 0
    eval_start = time.time()
    for start in range(0, len(indices), 1):
        idx = indices[start]
        sample = dataset[idx]
        prompt = sample.prompt
        seq = sample.sequence
        snapshot_path = sample.snapshot_path
        intersection_id = sample.intersection_id

        ts_tensor = torch.tensor([seq], dtype=torch.float32, device=device)
        with torch.no_grad():
            images = ts_adapter.generate(ts_tensor)
            batch_actions, _ = policy.generate_actions(
                [prompt],
                images,
                generation_kwargs,
            )
        action = batch_actions[0]

        if snapshot_path:
            eng.load_from_file(snapshot_path)
        target_index = intersection_id_to_index.get(intersection_id)
        if target_index is None:
            raise ValueError(f"intersection_id not found in roadnet: {intersection_id}")
        action_list = [
            random.randrange(len(ACTION_LIST))
            for _ in range(intersections_count)
        ]
        action_list[target_index] = action
        for iid, phase in zip(intersection_ids, action_list):
            eng.set_tl_phase(iid, int(phase))
        eng.next_step()
        reward = -mean_waiting_time(eng, intersection_id)
        rewards.append(reward)

        sample_counter += 1
        if log_interval > 0 and (
            sample_counter == 1
            or sample_counter == total_samples
            or sample_counter % log_interval == 0
        ):
            avg_reward_so_far = float(np.mean(rewards)) if rewards else 0.0
            elapsed = time.time() - eval_start
            print(
                "[测试进度] {}/{} | 即时奖励(负等待)={:.4f} | 平均奖励={:.4f} | 动作={} | 用时{:.1f}s".format(
                    sample_counter,
                    total_samples,
                    reward,
                    avg_reward_so_far,
                    _shorten(action),
                    elapsed,
                )
            )
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_waiting": -avg_reward,
        "num_samples": len(rewards),
    }




def main() -> None:
    args = parse_args()
    # 统一随机种子，避免结果波动
    seed_everything(args.seed)

    local_rank = int(args.local_rank)
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    use_cuda = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if use_cuda else 0

    def _parse_device_ids(value: str) -> list[int]:
        text = str(value or "").strip()
        if not text or text.lower() in ("none", "auto", "cpu"):
            return []
        seen = set()
        ids: list[int] = []
        for chunk in text.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            did = int(chunk)
            if did in seen:
                continue
            seen.add(did)
            ids.append(did)
        return ids

    train_device_ids = _parse_device_ids(args.train_device_ids)
    llm_device_ids = _parse_device_ids(args.llm_device_ids)

    if use_cuda:
        for label, ids in (("training", train_device_ids), ("LLM", llm_device_ids)):
            for did in ids:
                if did < 0 or did >= gpu_count:
                    raise RuntimeError(
                        f"Requested {label} device cuda:{did} is not available. "
                        f"Visible GPU count: {gpu_count}."
                    )
    else:
        if (train_device_ids or llm_device_ids) and _is_rank0():
            print("Warning: CUDA is not available; falling back to CPU.")
        train_device_ids = []
        llm_device_ids = []

    if use_cuda and train_device_ids:
        primary_device_id = train_device_ids[0]
        torch.cuda.set_device(primary_device_id)
        device = torch.device(f"cuda:{primary_device_id}")
    else:
        primary_device_id = None
        device = torch.device("cpu")

    if use_cuda and train_device_ids and llm_device_ids:
        overlap = set(train_device_ids) & set(llm_device_ids)
        if overlap and _is_rank0():
            print(
                "Warning: training devices overlap with LLM devices: "
                f"{sorted(overlap)}."
            )

    use_multi_gpu = use_cuda and len(train_device_ids) > 1

    def _build_max_memory(device_ids: list[int]) -> dict[int, str]:
        max_memory = {}
        for did in device_ids:
            props = torch.cuda.get_device_properties(did)
            total_gib = int(props.total_memory / (1024**3) * 0.9)
            total_gib = max(total_gib, 1)
            max_memory[did] = f"{total_gib}GiB"
        return max_memory

    llm_device_map = None
    llm_max_memory = None
    llm_primary_device = None
    if use_cuda and llm_device_ids:
        llm_primary_device = torch.device(f"cuda:{llm_device_ids[0]}")
        if len(llm_device_ids) > 1:
            llm_device_map = "auto"
            llm_max_memory = _build_max_memory(llm_device_ids)
    else:
        llm_primary_device = torch.device("cpu")

    # 加载时间序列提示词样本（用于生成动作）
    dataset = TrafficPromptSeriesDataset(
        path=args.jsonl_path,
        intersection_id=args.intersection_id,
        seq_len=args.seq_len,
    )
    if len(dataset) == 0:
        raise RuntimeError("No samples found for the specified intersection.")
    print(
        "数据集加载完成：样本数={}, 序列长度={}, 交叉口={}。".format(
            len(dataset), args.seq_len, args.intersection_id
        )
    )
    train_indices, test_indices = _split_indices(len(dataset), 0.9, args.seed)
    print(
        "数据集拆分完成：训练样本数={}，测试样本数={}（20%测试集）。".format(
            len(train_indices), len(test_indices)
        )
    )

    # 直接读取原始 roadnet/traffic（不再生成单路口版本）
    roadnet_path = args.roadnet_src
    traffic_path = args.traffic_src
    if not os.path.exists(roadnet_path):
        raise FileNotFoundError(f"Roadnet not found: {roadnet_path}")
    with open(roadnet_path, "r", encoding="utf-8") as handle:
        roadnet_single = json.load(handle)
    vehicle_template = _load_vehicle_template(traffic_path)
    lane_route_map = _build_lane_route_map(roadnet_single, args.intersection_id)
    intersection_ids = [
        inter.get("id") for inter in roadnet_single.get("intersections", []) or []
    ]
    intersection_id_to_index = {
        inter_id: idx for idx, inter_id in enumerate(intersection_ids) if inter_id
    }
    intersections_count = len(intersection_ids)
    num_row, num_col = _infer_grid_size(intersection_ids)


    # 根据样本长度设定仿真步数
    run_counts = args.run_counts or (args.min_action_time * len(dataset))
    env_conf = build_env_conf(
        traffic_file=os.path.basename(traffic_path),
        roadnet_file=os.path.basename(roadnet_path),
        run_counts=run_counts,
        min_action_time=args.min_action_time,
        num_row=num_row,
        num_col=num_col,
        num_intersections=intersections_count,
    )

    # 准备训练输出目录与 CityFlow 工作目录
    timestamp = None
    if _is_rank0():
        timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    timestamp = _broadcast_object(timestamp)
    dic_path = prepare_paths(
        os.path.dirname(roadnet_path) or ".",
        os.path.basename(traffic_path),
        timestamp=timestamp,
    )
    rank_work_dir = resolve_rank_work_dir(dic_path["PATH_TO_WORK_DIRECTORY"], local_rank)
    dic_path["PATH_TO_WORK_DIRECTORY"] = rank_work_dir
    os.makedirs(rank_work_dir, exist_ok=True)
    copy_cityflow_files(dic_path, env_conf)
    _barrier()

    # 初始化 CityFlow 环境（从保存的参数加载）
    cityflow_env_args_path = (
        "/home/code/LLMSTSC/records/LLMTLCSRun/"
        "anon_3_4_jinan_real.json_01_29_17_15_14/"
        "cityflow_env_init/cityflow_env_args.json"
    )
    with open(cityflow_env_args_path, "r", encoding="utf-8") as handle:
        cityflow_env_payload = json.load(handle)
    cityflow_env_args = cityflow_env_payload.get("cityflow_env_args", cityflow_env_payload)
    env = CityFlowEnv(**cityflow_env_args)
    env.reset()
    eng = env.eng
    # 初始化可学习的时间序列图像适配器
    ts_adapter = TimeSeriesImageAdapter(
        image_size=args.image_size,
        seq_len=args.seq_len,
        periodicity=args.periodicity,
        norm_const=args.norm_const,
        learnable_image=True,
        three_channel_image=args.three_channel,
        save_images=False,
        device=device,
    )
    if ts_adapter.learnable_image_module is None:
        raise RuntimeError("TimeSeriesImageAdapter learnable_image_module is not initialized.")
    if use_multi_gpu:
        device_ids = list(train_device_ids)
        if primary_device_id in device_ids:
            device_ids.remove(primary_device_id)
            device_ids.insert(0, primary_device_id)
        ts_adapter.learnable_image_module = torch.nn.DataParallel(
            ts_adapter.learnable_image_module,
            device_ids=device_ids,
            output_device=device_ids[0],
        )
        if _is_rank0():
            print(f"Using DataParallel on GPUs {device_ids} for learnable image adapter.")
    elif use_cuda:
        if _is_rank0():
            print(f"Training device set to cuda:{primary_device_id}.")

    # 初始化 LLM 策略（负责生成动作）
    policy = GemmaVLPolicy(
        model_path=args.llm_path,
        device=llm_primary_device,
        device_map=llm_device_map,
        max_memory=llm_max_memory,
    )
    # 冻结 LLM 参数并开启显存优化选项
    if policy.model is not None:
        policy.model.requires_grad_(False)
        policy.model.eval()
        try:
            policy.model.gradient_checkpointing_enable()
        except Exception as exc:
            if _is_rank0():
                print(f"Warning: failed to enable gradient checkpointing: {exc}")
        if hasattr(policy.model, "config") and policy.model.config is not None:
            policy.model.config.use_cache = False

    # 只优化可学习的图像投影模块
    optimizer = torch.optim.AdamW(
        ts_adapter.learnable_image_module.parameters(),
        lr=args.learning_rate,
    )
    loss_device = device

    # LLM 生成超参
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": True,
        "pad_token_id": policy.tokenizer.pad_token_id,
        "eos_token_id": policy.tokenizer.eos_token_id,
    }

    cache_dir = os.path.join("records", "projection_train", "sampling_cache")
    cache_key = os.path.basename(args.jsonl_path)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pt")
    if _is_rank0():
        os.makedirs(cache_dir, exist_ok=True)
    _barrier()

    log_lines = []
    log_dir = os.path.join("records", "projection_train", "model")
    if _is_rank0():
        os.makedirs(log_dir, exist_ok=True)
    log_date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(log_dir, f"projection_ppo_{log_date}.txt")
    if _is_rank0():
        log_lines.append(
            "数据集拆分：训练样本数={}，测试样本数={}（20%测试集）。".format(
                len(train_indices), len(test_indices)
            )
        )

    model_path = _resolve_model_path(args.ppo_model_path, args.jsonl_path)
    if args.ppo_model_path and model_path is None:
        raise FileNotFoundError(f"ppo_model_path not found: {args.ppo_model_path}")

    if model_path:
        if _is_rank0():
            print(f"检测到 ppo_model_path，跳过训练，加载模型进行测试：{model_path}")
            log_lines.append(f"Skip training. Load model: {model_path}")
        module = ts_adapter.learnable_image_module
        state_dict = torch.load(model_path, map_location=device)
        if isinstance(module, torch.nn.DataParallel):
            module.module.load_state_dict(state_dict, strict=True)
        else:
            module.load_state_dict(state_dict, strict=True)
        test_result = evaluate_policy(
            dataset=dataset,
            indices=test_indices,
            ts_adapter=ts_adapter,
            policy=policy,
            device=device,
            eng=eng,
            intersection_ids=intersection_ids,
            intersection_id_to_index=intersection_id_to_index,
            intersections_count=intersections_count,
            generation_kwargs=generation_kwargs,
            log_interval=args.log_interval,
        )
        if _is_rank0():
            summary = (
                f"[测试汇总] 样本数={test_result['num_samples']} | "
                f"平均奖励(负等待)={test_result['avg_reward']:.4f} | "
                f"估计平均等待时间={test_result['avg_waiting']:.4f}"
            )
            print(summary)
            log_lines.append(summary)
            with open(log_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(log_lines))
            print(f"[日志保存] 训练/测试日志已保存到：{log_path}")
        return

    for round_idx in range(args.num_rounds):
        print(
            "\n======== 开始第 {}/{} 轮训练 ========".format(
                round_idx + 1, args.num_rounds
            )
        )
        
        # 收集一轮的轨迹数据
        prompts = []
        sequences = []
        actions = []
        old_logprobs = []
        
        rewards = []
        use_cached = False

      
        use_cached=False
        if not use_cached:

            total_samples = len(train_indices)
            collect_start = time.time()

            indices = list(train_indices)
            random.shuffle(indices)
            sample_counter = 0
            batch_size = max(1, int(args.batch_size))
            random.shuffle(indices)
            for batch_start in range(0, len(indices), batch_size):
                
                batch_indices = indices[batch_start : batch_start + batch_size]
                batch_samples = [dataset[i] for i in batch_indices]
                batch_prompts = [sample.prompt for sample in batch_samples]
                batch_seqs = [sample.sequence for sample in batch_samples]
                batch_lane_details = [sample.lane_details for sample in batch_samples]
                batch_snapshot_paths = [sample.snapshot_path for sample in batch_samples]
                batch_intersection_ids = [sample.intersection_id for sample in batch_samples]
                
                # 将时间序列编码为图像输入
                ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    images = ts_adapter.generate(ts_tensor)
             
                    # 用 LLM 策略生成动作（文本形式）
                    batch_actions, _ = policy.generate_actions(
                        batch_prompts,
                        images,
                        generation_kwargs,
                    )
                    # 计算当前动作的对数概率（旧策略）
                    batch_logprobs = (
                        policy.action_logprob(batch_prompts, images, batch_actions)
                        .detach()
                        .cpu()
                        .tolist()
                    )

                for prompt, seq, action, logprob, lane_details, snapshot_path, intersection_id in zip(
                    batch_prompts,
                    batch_seqs,
                    batch_actions,
                    batch_logprobs,
                    batch_lane_details,
                    batch_snapshot_paths,
                    batch_intersection_ids,
                ):
                    
            
                    
                    if snapshot_path:
                        eng.load_from_file(snapshot_path)
                        print("SUCCESS load")
                    target_index = intersection_id_to_index.get(intersection_id)
                    if target_index is None:
                        raise ValueError(
                            f"intersection_id not found in roadnet: {intersection_id}"
                        )
                    action_list = [
                        random.randrange(len(ACTION_LIST))
                        for _ in range(intersections_count)
                    ]
                    action_list[target_index] = action
                    # 在仿真环境中执行动作
                    for iid, phase in zip(intersection_ids, action_list):
                        eng.set_tl_phase(iid, int(phase))   # 逐路口设置相位
                    eng.next_step()  
                    # 奖励：负的平均等待时间（越小越好）
                    reward = -mean_waiting_time(eng, intersection_id)
                    print(action, -reward)
                    prompts.append(prompt)
                    print("*********")
                    sequences.append(seq)
                    actions.append(action)
                    old_logprobs.append(logprob)
                    rewards.append(reward)

                    sample_counter += 1
                    if args.log_interval > 0 and (
                        sample_counter == 1
                        or sample_counter == total_samples
                        or sample_counter % args.log_interval == 0
                    ):
                        avg_reward_so_far = float(np.mean(rewards)) if rewards else 0.0
                        elapsed = time.time() - collect_start
                        print(
                            "[采样进度] {}/{} | 即时奖励(负等待)={:.4f} | 平均奖励={:.4f} | 动作={} | 用时{:.1f}s".format(
                                sample_counter,
                                total_samples,
                                reward,
                                avg_reward_so_far,
                                _shorten(action),
                                elapsed,
                            )
                        )

            if round_idx == 0:
                if _is_rank0():
                    torch.save(
                        {
                            "prompts": prompts,
                            "sequences": sequences,
                            "actions": actions,
                            "old_logprobs": old_logprobs,
                            "rewards": rewards,
                        },
                        cache_path,
                    )
                    print(
                        "[采样缓存] 已保存缓存结果：{} | 样本数={}".format(
                            cache_path, len(prompts)
                        )
                    )
                _barrier()

        # 计算回报与优势（用于 PPO 更新）
        returns = discounted_returns(rewards, args.gamma).to(loss_device)
        advantages = returns.clone()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        indices = list(range(len(prompts)))
        minibatch_total = math.ceil(len(indices) / max(1, args.mini_batch_size))
        for epoch_idx in range(args.ppo_epochs):
            # random.shuffle(indices)
            epoch_losses = []
            epoch_clip_fracs = []
            for start in range(0, len(indices), args.mini_batch_size):
                batch_idx = indices[start : start + args.mini_batch_size]
                minibatch_idx = start // args.mini_batch_size + 1
                batch_prompts = [prompts[i] for i in batch_idx]
                batch_seqs = [sequences[i] for i in batch_idx]
                batch_actions = [actions[i] for i in batch_idx]

                # 重新计算图像与新策略的对数概率
                ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                images = ts_adapter.generate(ts_tensor)

                new_logprobs = policy.action_logprob(batch_prompts, images, batch_actions)
                old_logprob_tensor = torch.tensor(
                    [old_logprobs[i] for i in batch_idx],
                    dtype=new_logprobs.dtype,
                    device=new_logprobs.device,
                )
                advantage_tensor = advantages[batch_idx].to(new_logprobs.device)

                # PPO 裁剪目标
                ratio = torch.exp(new_logprobs - old_logprob_tensor)
                unclipped = ratio * advantage_tensor
                clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantage_tensor
                policy_loss = -torch.min(unclipped, clipped).mean()
                clip_frac = (
                    (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item()
                )

                # 反向传播更新可学习投影模块
                optimizer.zero_grad(set_to_none=True)
                policy_loss.backward()
                optimizer.step()

                epoch_losses.append(float(policy_loss.detach().cpu().item()))
                epoch_clip_fracs.append(clip_frac)

                if args.log_interval > 0 and (
                    minibatch_idx == 1
                    or minibatch_idx == minibatch_total
                    or minibatch_idx % args.log_interval == 0
                ):
                    print(
                        "[更新进度] Epoch {}/{} | Batch {}/{} | 策略损失loss={:.6f}(越小越稳定) | clip占比={:.3f}".format(
                            epoch_idx + 1,
                            args.ppo_epochs,
                            minibatch_idx,
                            minibatch_total,
                            epoch_losses[-1],
                            clip_frac,
                        )
                    )

            if epoch_losses:
                print(
                    "[Epoch汇总] {}/{} | 平均loss={:.6f} | 平均clip占比={:.3f}".format(
                        epoch_idx + 1,
                        args.ppo_epochs,
                        float(np.mean(epoch_losses)),
                        float(np.mean(epoch_clip_fracs)) if epoch_clip_fracs else 0.0,
                    )
                )

        # 保存当前轮次的投影模块权重
        ckpt_dir = os.path.join(dic_path["PATH_TO_MODEL"], f"round_{round_idx}")
        os.makedirs(ckpt_dir, exist_ok=True)
        if _is_rank0():
            module = ts_adapter.learnable_image_module
            state_dict = (
                module.module.state_dict()
                if isinstance(module, torch.nn.DataParallel)
                else module.state_dict()
            )
            torch.save(state_dict, os.path.join(ckpt_dir, "ts_image_adapter.pt"))

        # 输出本轮平均奖励
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_waiting = -avg_reward
        print(
            "第 {} 轮完成：平均奖励(负等待)={:.4f}，估计平均等待时间={:.4f}。模型已保存到 {}。".format(
                round_idx + 1, avg_reward, avg_waiting, ckpt_dir
            )
        )
        if _is_rank0():
            log_lines.append(
                "第 {} 轮完成：平均奖励(负等待)={:.4f}，估计平均等待时间={:.4f}。模型已保存到 {}。".format(
                    round_idx + 1, avg_reward, avg_waiting, ckpt_dir
                )
            )

    test_result = evaluate_policy(
        dataset=dataset,
        indices=test_indices,
        ts_adapter=ts_adapter,
        policy=policy,
        device=device,
        eng=eng,
        intersection_ids=intersection_ids,
        intersection_id_to_index=intersection_id_to_index,
        intersections_count=intersections_count,
        generation_kwargs=generation_kwargs,
        log_interval=args.log_interval,
    )
    if _is_rank0():
        summary = (
            f"[测试汇总] 样本数={test_result['num_samples']} | "
            f"平均奖励(负等待)={test_result['avg_reward']:.4f} | "
            f"估计平均等待时间={test_result['avg_waiting']:.4f}"
        )
        print(summary)
        log_lines.append(summary)
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines))
        print(f"[日志保存] 训练/测试日志已保存到：{log_path}")


if __name__ == "__main__":
    main()
