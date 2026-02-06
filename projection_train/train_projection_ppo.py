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
from typing import Any, Optional
# --- add near imports ---
import re
import numpy as np
import torch
import torch.distributed as dist

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
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
        default="/home/code/LLMSTSC/data/recorded/anon_3_4_jinan_real.json_02_04_08_41_17/traffic_prompt_series_2s.jsonl",
    )
    parser.add_argument("--intersection_id", type=str, default="intersection_1_1")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--interval_s", type=int, default=2)
    parser.add_argument("--llm_path", type=str, default="/home/data/model/gemma-12B/")
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

    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument(
        "--ppo_model_path",
        type=str,
        default="",
        help="PPO projection model path. If set, skip training and run evaluation only.",
    )
    # gemma 
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


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _get_rank_world_size() -> tuple[int, int]:
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _init_distributed(local_rank: int) -> torch.device:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        if local_rank >= 0:
            device = torch.device(f"cuda:{local_rank}")
            torch.cuda.set_device(device)
        else:
            device = torch.device("cuda:0")
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if use_cuda else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return device


def _shard_indices(indices: list[int], rank: int, world_size: int) -> list[int]:
    if world_size <= 1:
        return indices
    return [idx for pos, idx in enumerate(indices) if pos % world_size == rank]


def _broadcast_object(value):
    if not dist.is_initialized():
        return value
    obj_list = [value]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def _unwrap_module(module: nn.Module) -> nn.Module:
    if isinstance(module, (DDP, torch.nn.DataParallel)):
        return module.module
    return module


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


# --- add somewhere (e.g., above main / above training loop) ---
def parse_action_id(action: Any, num_actions: int, default: int = 0) -> int:
    """

    """
    if action is None:
        print("action is Null")
        return default

    # already numeric
    if isinstance(action, (int,)):
        a = action
    elif isinstance(action, float):
        a = int(round(action))
    elif isinstance(action, str):
        s = action.strip()
        # exact digit string
        if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
            a = int(s)
        else:
            # pick the first integer appearing in the string
            m = re.search(r"-?\d+", s)
            a = int(m.group(0)) if m else default
    else:
        # try best-effort cast
        try:
            a = int(action)
        except Exception:
            print("action 不能被解析");
            a = default

   
    return a


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
        action = batch_actions[0]  # 不一定能获得动作
        # --- in your sampling loop, after you get `action` from policy.generate_actions(...) ---
        # NOTE: ACTION_LIST length is your discrete action space size
        # action_id = parse_action_id(action, num_actions=len(ACTION_LIST), default=0)  #这里不能用action_id，logprob 必须对应“实际执行的动作”


        if snapshot_path:
            eng.load_from_file(snapshot_path)
        target_index = intersection_id_to_index.get(intersection_id)
        if target_index is None:
            raise ValueError(f"intersection_id not found in roadnet: {intersection_id}")
        eng.set_tl_phase(intersection_id, action)
        
        
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


class ImageCritic(nn.Module):
    """
    V(s) baseline: 输入是 adapter 生成的图像 [B, 3, H, W]，输出每个样本的 state value [B]
    """
    def __init__(self, in_ch: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.v_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, H, W]
        h = self.net(images)
        v = self.v_head(h).squeeze(-1)   # [B]
        return v

def main() -> None:
    args = parse_args()
    # 统一随机种子，避免结果波动
    seed_everything(args.seed)

    local_rank = int(args.local_rank)
    if local_rank < 0:
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))

    device = _init_distributed(local_rank)
    rank, world_size = _get_rank_world_size()
    use_cuda = device.type == "cuda"
    if use_cuda and local_rank < 0 and world_size > 1:
        local_rank = rank
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    llm_device_map = None
    llm_max_memory = None
    llm_primary_device = device

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
    train_indices, test_indices = _split_indices(len(dataset), 0.3, args.seed)
    train_indices = _shard_indices(train_indices, rank, world_size)
    if dist.is_initialized():
        local_len = torch.tensor(len(train_indices), device=device)
        dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
        min_len = int(local_len.item())
        train_indices = train_indices[:min_len]
        if min_len == 0:
            raise RuntimeError("No training samples left after DDP sharding.")
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
   
    intersection_ids = [
        inter.get("id") for inter in roadnet_single.get("intersections", []) or []
    ]
    intersection_id_to_index = {
        inter_id: idx for idx, inter_id in enumerate(intersection_ids) if inter_id
    }
    intersections_count = len(intersection_ids)



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

    _barrier()

    # 初始化 CityFlow 环境（从保存的参数加载）
    cityflow_env_args_path = (
        "/home/code/LLMSTSC/records/LLMTLCSRun/"
        "anon_3_4_jinan_real.json_02_04_08_41_17/"
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
    loss_device = device
    critic = ImageCritic(in_ch=3, hidden=128).to(loss_device)
    critic.train()

    # 价值函数系数（常用 0.5）
    vf_coef = getattr(args, "vf_coef", 0.5)
    if ts_adapter.learnable_image_module is None:
        raise RuntimeError("TimeSeriesImageAdapter learnable_image_module is not initialized.")
    if dist.is_initialized():
        ddp_device_ids = [device.index] if device.type == "cuda" else None
        ts_adapter.learnable_image_module = DDP(
            ts_adapter.learnable_image_module,
            device_ids=ddp_device_ids,
            output_device=device.index if device.type == "cuda" else None,
        )
        critic = DDP(
            critic,
            device_ids=ddp_device_ids,
            output_device=device.index if device.type == "cuda" else None,
        )

    # 如果你原来 optimizer 只优化 adapter，这里把 critic 参数也加进去
    optimizer = torch.optim.Adam(
        list(ts_adapter.learnable_image_module.parameters()) + list(critic.parameters()),
        lr=args.learning_rate
    )

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
    cache_path = os.path.join(cache_dir, f"{cache_key}_rank{rank}.pt")
    os.makedirs(cache_dir, exist_ok=True)
    _barrier()

    log_lines = []
    log_dir = os.path.join("records", "projection_train", "model")
    os.makedirs(log_dir, exist_ok=True)
    log_date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(log_dir, f"projection_ppo_{log_date}_rank{rank}.txt")
    log_lines.append(f"DDP rank={rank}, world_size={world_size}, device={device}.")
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
        if _is_rank0():
            module = ts_adapter.learnable_image_module
            state_dict = torch.load(model_path, map_location=device)
            _unwrap_module(module).load_state_dict(state_dict, strict=True)
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
            summary = (
                f"[测试汇总] 样本数={test_result['num_samples']} | "
                f"平均奖励(负等待)={test_result['avg_reward']:.4f} | "
                f"估计平均等待时间={test_result['avg_waiting']:.4f}"
            )
            print(summary)
            log_lines.append(summary)
        _barrier()
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(log_lines))
        print(f"[日志保存] 训练/测试日志已保存到：{log_path}")
        if dist.is_initialized():
            dist.destroy_process_group()
        return
    minibatch_idx=12
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
            ts_image_time = 0.0
            llm_time = 0.0

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
                    _sync_if_cuda(device)
                    ts_start = time.perf_counter()
                    images = ts_adapter.generate(ts_tensor)
                    _sync_if_cuda(device)
                    ts_image_time += time.perf_counter() - ts_start
             
                    # 用 LLM 策略生成动作（文本形式）
                    _sync_if_cuda(device)
                    llm_start = time.perf_counter()
                    batch_actions, _ = policy.generate_actions(
                        batch_prompts,
                        images,
                        generation_kwargs,
                    )
                    _sync_if_cuda(device)
                    llm_time += time.perf_counter() - llm_start
                    # 计算当前动作的对数概率（旧策略）
                    _sync_if_cuda(device)
                    llm_start = time.perf_counter()
                    batch_logprobs = (
                        policy.action_logprob(batch_prompts, images, batch_actions)
                        .detach()
                        .cpu()
                        .tolist()
                    )
                    _sync_if_cuda(device)
                    llm_time += time.perf_counter() - llm_start

                for prompt, seq, action, logprob,  snapshot_path, intersection_id in zip(
                    batch_prompts,
                    batch_seqs,
                    batch_actions,
                    batch_logprobs,
        
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
                    # action_list = [
                    #     random.randrange(len(ACTION_LIST))
                    #     for _ in range(intersections_count)
                    # ]
                    # action_list[target_index] = action
                    # # 在仿真环境中执行动作
                    # for iid, phase in zip(intersection_ids, action_list):
                    #     eng.set_tl_phase(iid, int(phase))   # 逐路口设置相位
                    # action_id = parse_action_id(action, num_actions=len(ACTION_LIST), default=0)
                    eng.set_tl_phase(intersection_id, action)
                    
                  
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
                        ts_avg = ts_image_time / max(1, sample_counter)
                        llm_avg = llm_time / max(1, sample_counter)
                        print(
                            "[采样进度] {}/{} | 即时奖励(负等待)={:.4f} | 平均奖励={:.4f} | 动作={} | 用时{:.1f}s | ts_images累计{:.1f}s(均{:.3f}s/样本) | LLM累计{:.1f}s(均{:.3f}s/样本)".format(
                                sample_counter,
                                total_samples,
                                reward,
                                avg_reward_so_far,
                                _shorten(action),
                                elapsed,
                                ts_image_time,
                                ts_avg,
                                llm_time,
                                llm_avg,
                            )
                        )

            if round_idx == 0:
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
        

        print("[步骤1] 全局计算values和advantages...")
        
        # 构建returns tensor
        returns = torch.tensor(rewards, dtype=torch.float32, device=loss_device)
        
        # 一次性计算所有样本的values（用于advantage计算）
        all_values = []
        batch_size_for_value = 8  # 防止OOM，可根据GPU显存调整
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size_for_value):
                batch_seqs = sequences[i : i + batch_size_for_value]
                ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                images = ts_adapter.generate(ts_tensor)
                batch_values = critic(images)  # [batch_size_for_value]
                all_values.append(batch_values.to(loss_device))
        
        values_all = torch.cat(all_values, dim=0)  # [total_samples]
        
        # 全局计算advantages（重要：只归一化一次）
        advantages = returns - values_all
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        print(f"  - Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
        print(f"  - Values: mean={values_all.mean():.4f}, std={values_all.std():.4f}")
        print(f"  - Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")

        indices = list(range(len(prompts)))
        minibatch_total = math.ceil(len(indices) / max(1, args.mini_batch_size))
        for epoch_idx in range(args.ppo_epochs):
            random.shuffle(indices)  # 启用shuffle以打破时间相关性
            epoch_losses = []
            epoch_clip_fracs = []
            # PPO更新时直接索引
            for start in range(0, len(indices), args.mini_batch_size):
                batch_idx = indices[start : start + args.mini_batch_size]
                batch_prompts = [prompts[i] for i in batch_idx]
                batch_actions = [actions[i] for i in batch_idx]
                batch_seqs = [sequences[i] for i in batch_idx]
                ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                images = ts_adapter.generate(ts_tensor)
                new_logprobs = policy.action_logprob(batch_prompts, images, batch_actions)

                old_logprob_tensor = torch.tensor(
                    [old_logprobs[i] for i in batch_idx],
                    dtype=new_logprobs.dtype,
                    device=new_logprobs.device,
                )  # [B]

                returns_b = returns[batch_idx].to(new_logprobs.device)  # [B]
                
                # 从全局advantages中索引当前batch（已经归一化过，无需重复）
                advantages_b = advantages[batch_idx].to(new_logprobs.device)  # [B]
                
                # 从全局values中索引当前batch（用于value clipping）
                old_values_b = values_all[batch_idx].to(new_logprobs.device)  # [B]

                # critic
                values = critic(images)  # [B]

                # PPO ratio
                ratio = torch.exp(new_logprobs - old_logprob_tensor)  # [B]

                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # ============================================================
                # Value Loss (带Clipping，防止value更新过大)
                # ============================================================
                
                # Clip value predictions
                values_clipped = old_values_b + torch.clamp(
                    values - old_values_b,
                    -args.clip_range,
                    args.clip_range,
                )
                
          
                # 方法2：更简洁（OpenAI标准实现）
                vpredclipped = old_values_b + torch.clamp(values - old_values_b, -args.clip_range, args.clip_range)
                vf_loss1 = (values - returns_b).pow(2)
                vf_loss2 = (vpredclipped - returns_b).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))
                # ============================================================
                # Entropy Loss (鼓励探索)
                # ============================================================
                
                # 计算熵（需要policy类支持compute_entropy方法）
                try:
                    entropy = policy.compute_entropy(batch_prompts, images)
                    entropy_loss = -entropy.mean()  # 负号因为我们要最大化熵
                    ent_coef = 0.01  # 可调超参数
                except AttributeError:
                    # 如果policy没有实现entropy，跳过
                    entropy_loss = torch.tensor(0.0, device=new_logprobs.device)
                    ent_coef = 0.0
                    if epoch_idx == 0 :
                        print("[警告] Policy未实现compute_entropy方法，跳过entropy bonus")
                
                
                # ============================================================
                # Total Loss
                # ============================================================
                
                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # 梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(
                    list(ts_adapter.learnable_image_module.parameters()) + list(critic.parameters()),
                    max_norm=0.5
                )
                optimizer.step()

                clip_frac = (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item()



                

                epoch_losses.append({
                    'policy_loss': float(policy_loss.detach().cpu().item()),
                    'value_loss': float(value_loss.detach().cpu().item()),
                    'entropy_loss': float(entropy_loss.detach().cpu().item()) if isinstance(entropy_loss, torch.Tensor) else 0.0,
                    'total_loss': float(loss.detach().cpu().item()),
                })
                epoch_clip_fracs.append(clip_frac)

              
            if epoch_losses:
                avg_policy = np.mean([l['policy_loss'] for l in epoch_losses])
                avg_value = np.mean([l['value_loss'] for l in epoch_losses])
                avg_entropy = np.mean([l['entropy_loss'] for l in epoch_losses])
                avg_total = np.mean([l['total_loss'] for l in epoch_losses])
                print(
                    "[Epoch汇总] {}/{} | avg_policy={:.6f} | avg_value={:.6f} | "
                    "avg_entropy={:.6f} | avg_total={:.6f} | avg_clip={:.3f}".format(
                        epoch_idx + 1,
                        args.ppo_epochs,
                        avg_policy,
                        avg_value,
                        avg_entropy,
                        avg_total,
                        float(np.mean(epoch_clip_fracs)) if epoch_clip_fracs else 0.0,
                    )
                )

        ckpt_dir = os.path.join(dic_path["PATH_TO_MODEL"], f"round_{round_idx}")
        os.makedirs(ckpt_dir, exist_ok=True)
        module = ts_adapter.learnable_image_module
        state_dict = _unwrap_module(module).state_dict()
        torch.save(
            state_dict,
            os.path.join(ckpt_dir, f"ts_image_adapter_rank{rank}.pt"),
        )

        # 输出本轮平均奖励
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_waiting = -avg_reward
        print(
            "第 {} 轮完成：平均奖励(负等待)={:.4f}，估计平均等待时间={:.4f}。模型已保存到 {}。".format(
                round_idx + 1, avg_reward, avg_waiting, ckpt_dir
            )
        )
        log_lines.append(
            "第 {} 轮完成：平均奖励(负等待)={:.4f}，估计平均等待时间={:.4f}。模型已保存到 {}。".format(
                round_idx + 1, avg_reward, avg_waiting, ckpt_dir
            )
        )
    # 每个epoch打印
        print(f"ratio stats: mean={ratio.mean():.3f}, std={ratio.std():.3f}, "
            f"min={ratio.min():.3f}, max={ratio.max():.3f}")
        print(f"old_logprobs: mean={old_logprob_tensor.mean():.3f}")
        print(f"new_logprobs: mean={new_logprobs.mean():.3f}")
    if _is_rank0():
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
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
