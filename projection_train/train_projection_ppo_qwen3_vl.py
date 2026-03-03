"""
基于 PPO 对时间序列到图像的可学习适配器进行投影训练（Qwen3-VL 版本）。
流程：读取交通数据样本 -> 构建单路口 CityFlow 环境 -> 生成图像 -> Qwen3-VL 策略输出动作 -> 计算等待时间奖励 -> PPO 更新。
"""
import cityflow
import argparse
import csv
import hashlib
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
from projection_train.policy_qwen3_vl import Qwen3VLPolicy, ACTION_LIST

_SIGNAL_ONLY_SUFFIX = (
    "\n\n[输出格式要求]\n"
    "你只能输出信号灯相位，且必须严格使用 <signal>相位</signal> 格式。\n"
    "不要输出任何解释、分析、标点说明、代码块或额外文本。"
)

def _configure_cuda_allocator() -> None:
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"


def _append_signal_only_instruction(prompt: str) -> str:
    """Append strict output-format instruction for signal-only response."""
    base = prompt.rstrip()
    return f"{base}{_SIGNAL_ONLY_SUFFIX}"


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
@lru_cache(maxsize=64)
def _load_incoming_lane_ids(roadnet_path: str, intersection_id: str) -> list[str]:
    """Load incoming lane ids for an intersection from the specified roadnet file."""
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


def mean_waiting_time(env, intersection_id: str, roadnet_path: str) -> float:
    """Compute per-step average waiting ratio over [0, 30) with dt=1s."""
    lane_ids = _load_incoming_lane_ids(roadnet_path, intersection_id)
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
    parser.add_argument(
        "--dataset_keyword",
        type=str,
        default="jinan",
        help="按关键词匹配 recorded 子目录并合并其中的 jsonl；支持逗号分隔多个关键词。",
    )
    parser.add_argument(
        "--recorded_root",
        type=str,
        default="/home/code/LLMSTSC/data/recorded",
        help="recorded 数据根目录（与 --dataset_keyword 搭配使用）。", 
    )
    parser.add_argument(
        "--dataset_filename",
        type=str,
        default="traffic_prompt_series_2s.jsonl",
        help="每个匹配子目录中需要收集的 jsonl 文件名。",
    )
    parser.add_argument(
        "--llmtlcs_run_root",
        type=str,
        default="/home/code/LLMSTSC/records/LLMTLCSRun",
        help="与 recorded 子目录同名的 LLMTLCSRun 根目录。",
    )
    parser.add_argument("--intersection_id", type=str, default="intersection_1_1")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--interval_s", type=int, default=2)
    # parser.add_argument("--llm_path", type=str, default="/home/data/model/gemma-12B/")
    parser.add_argument("--llm_path", type=str, default="/home/data/model/qwen-vl-8B/")
    parser.add_argument(
        "--llm_fsdp",
        type=str,
        default="on",
        choices=["auto", "on", "off"],
        help="LLM 是否使用 FSDP：auto=多卡分布式时启用，on=强制启用，off=关闭。",
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

    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument(
        "--ppo_model_path",
        type=str,
        default="",
        help="PPO projection model path. If set, skip training and run evaluation only.",
    )
    # qwen3-vl
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--periodicity", type=int, default=24)
    parser.add_argument("--norm_const", type=float, default=0.4)
    parser.add_argument("--three_channel", action="store_true", default=True)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.01,
        help="测试集占比（默认 0.2，即 20%%）。",
    )
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.00)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument(
        "--early_stop_loss_threshold",
        type=float,
        default=None,
        help="达到该 avg_total_loss 阈值时提前停止当前 round（例如 0.01）。",
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=3,
        help="连续若干个 epoch 无改进时提前停止当前 round。",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=1e-4,
        help="判定改进所需的最小损失下降幅度。",
    )
    parser.add_argument(
        "--target_kl",
        type=float,
        default=1.2,
        help="minibatch 的近似 KL 超过该值时提前结束当轮 epoch。",
    )
    parser.add_argument(
        "--stop_on_nan_inf",
        action="store_true",
        default=True,
        help="训练 loss 出现 NaN/Inf 时立即停止。",
    )
    parser.add_argument("--use_lr_scheduler", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_p", type=float, default=1.0) 
    parser.add_argument("--do_sample", action="store_true", default=True)
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


def _ddp_allreduce_max(value: float, device: torch.device) -> float:
    if not dist.is_initialized():
        return float(value)
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return float(tensor.item())


def _ddp_any_true(flag: bool, device: torch.device) -> bool:
    if not dist.is_initialized():
        return bool(flag)
    tensor = torch.tensor(1 if flag else 0, device=device, dtype=torch.int32)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return bool(int(tensor.item()))


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


def verify_freeze_and_trainable(
    policy,
    ts_adapter,
    optimizer: torch.optim.Optimizer,
    loss: Optional[torch.Tensor] = None,
    rank0_only: bool = True,
) -> None:
    """Verify LLM is frozen and adapter is trainable, plus optional gradient checks."""
    should_print = True
    if rank0_only and dist.is_initialized():
        should_print = dist.get_rank() == 0

    adapter_module = getattr(ts_adapter, "learnable_image_module", None)
    if adapter_module is None:
        if should_print:
            print("[验证] ts_adapter.learnable_image_module 未初始化，无法检查。")
        return
    if hasattr(adapter_module, "module"):
        adapter_module = adapter_module.module

    llm_model = getattr(policy, "model", None)
    if llm_model is not None and hasattr(llm_model, "module"):
        llm_model = llm_model.module

    if llm_model is None and should_print:
        print("[验证] policy.model 为 None，跳过 LLM 相关检查。")

    optimizer_param_ids: set[int] = set()
    for group in optimizer.param_groups:
        for param in group.get("params", []):
            if param is not None:
                optimizer_param_ids.add(id(param))

    llm_params = list(llm_model.parameters()) if llm_model is not None else []
    adapter_params = list(adapter_module.parameters())

    llm_trainable_numel = sum(p.numel() for p in llm_params if p.requires_grad)
    adapter_trainable_numel = sum(p.numel() for p in adapter_params if p.requires_grad)

    llm_in_optimizer = any(id(p) in optimizer_param_ids for p in llm_params)
    adapter_in_optimizer = bool(adapter_params) and all(id(p) in optimizer_param_ids for p in adapter_params)

    if should_print:
        print(f"[验证] LLM trainable param count: {llm_trainable_numel}")
        print(f"[验证] Adapter trainable param count: {adapter_trainable_numel}")
        print(f"[验证] LLM params in optimizer: {'yes' if llm_in_optimizer else 'no'}")
        print(f"[验证] Adapter params in optimizer: {'yes' if adapter_in_optimizer else 'no'}")

    if llm_trainable_numel > 0 and should_print:
        print("[警告][冻结检查] LLM 仍有 requires_grad=True 的参数，未完全冻结。")
    if llm_in_optimizer and should_print:
        print("[警告][优化器检查] LLM 参数出现在 optimizer.param_groups 中。")
    if any(not p.requires_grad for p in adapter_params) and should_print:
        print("[警告][冻结检查] Adapter 存在 requires_grad=False 的参数，未完全可训练。")
    if not adapter_in_optimizer and should_print:
        print("[警告][优化器检查] Adapter 参数未完整加入 optimizer.param_groups。")

    if loss is not None:
        if should_print:
            print(
                "[验证] 注意：verify_freeze_and_trainable 将执行 loss.backward(retain_graph=True)。"
                "如果主流程还会 backward，请在主 backward 之后调用并传 loss=None。"
            )
        try:
            loss.backward(retain_graph=True)
        except Exception as exc:
            if should_print:
                print(f"[验证] backward 失败，跳过梯度检查：{exc}")
            return

    grads = [p.grad for p in adapter_params]
    has_any_grad = any(g is not None for g in grads)
    if not has_any_grad:
        if should_print:
            print("无法检查梯度，只检查 requires_grad 与 optimizer param_groups")
        return
    if loss is None and should_print:
        print("[验证] loss=None，使用已有梯度进行检查（未执行 backward）。")

    none_count = sum(1 for g in grads if g is None)
    total_params = len(grads)
    none_ratio = (none_count / total_params) if total_params > 0 else 1.0

    grad_abs_sum = 0.0
    grad_numel = 0
    for grad in grads:
        if grad is None:
            continue
        grad_abs_sum += float(grad.detach().abs().sum().item())
        grad_numel += grad.numel()

    grad_mean_abs = grad_abs_sum / grad_numel if grad_numel > 0 else 0.0

    if should_print:
        print(
            f"[验证] Adapter grad None ratio: {none_ratio:.3f} ({none_count}/{total_params})"
        )
        print(f"[验证] Adapter grad mean(|grad|): {grad_mean_abs:.6e}")

    all_grad_none = none_count == total_params
    all_grad_zero = grad_numel > 0 and grad_abs_sum == 0.0
    if (all_grad_none or all_grad_zero) and should_print:
        print(
            "[警告][梯度检查] Adapter 梯度全 None 或全 0，可能在 "
            "policy.prepare_inputs / policy.action_logprob_from_prepared 内部发生了 "
            ".detach() / torch.no_grad() / 转 tensor 断梯度。"
        )


def _format_training_config(
    args: argparse.Namespace,
    clip_value_loss_enabled: bool,
    clip_value_range: float,
    mixed_precision_enabled: bool,
    mixed_precision_dtype: Optional[str],
    grad_scaler_enabled: bool,
) -> str:
    config = {
        "clip_eps_or_clip_range": args.clip_range,
        "ppo_epochs": args.ppo_epochs,
        "mini_batch_size": args.mini_batch_size,
        "lr": args.learning_rate,
        "lr_scheduler": "enabled" if args.use_lr_scheduler else "none",
        "vf_coef": args.vf_coef,
        "ent_coef": args.ent_coef,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "max_grad_norm": args.max_grad_norm,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
        },
        "mixed_precision": {
            "enabled": mixed_precision_enabled,
            "dtype": mixed_precision_dtype,
            "grad_scaler": grad_scaler_enabled,
        },
        "clip_value_loss": {
            "enabled": clip_value_loss_enabled,
            "range": clip_value_range,
        },
        "early_stop": {
            "loss_threshold": args.early_stop_loss_threshold,
            "patience": args.early_stop_patience,
            "min_delta": args.early_stop_min_delta,
            "target_kl": args.target_kl,
            "stop_on_nan_inf": args.stop_on_nan_inf,
        },
    }
    return json.dumps(config, ensure_ascii=True, indent=2)


_STAGE_LOG_FIELDS = [
    "ts",
    "rank",
    "round",
    "stage",
    "batch_id",
    "sample_counter",
    "total_samples",
    "batch_size",
    "reward",
    "avg_reward",
    "ts_image_s",
    "llm_generate_s",
    "llm_logprob_s",
    "env_load_s",
    "env_set_phase_s",
    "reward_calc_s",
    "critic_s",
    "entropy_s",
    "backward_opt_s",
    "batch_total_s",
]


_EPOCH_LOG_FIELDS = [
    "ts",
    "rank",
    "round",
    "epoch",
    "num_minibatches",
    "avg_policy_loss",
    "avg_value_loss",
    "avg_entropy_loss",
    "avg_total_loss",
    "avg_clip_frac",
    "avg_approx_kl",
    "test_num_samples",
    "test_skipped_samples",
    "test_avg_reward",
    "test_avg_waiting",
    "best_so_far",
    "early_stop_reason",
]


def _append_stage_log(csv_path: str, row: dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_STAGE_LOG_FIELDS)
        writer.writerow({key: row.get(key, "") for key in _STAGE_LOG_FIELDS})


def _append_epoch_log(csv_path: str, row: dict) -> None:
    with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_EPOCH_LOG_FIELDS)
        writer.writerow({key: row.get(key, "") for key in _EPOCH_LOG_FIELDS})


def _safe_token(text: str, fallback: str = "na", max_len: int = 48) -> str:
    if not text:
        return fallback
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text).strip())
    token = token.strip("._-")
    if not token:
        token = fallback
    if len(token) > max_len:
        token = token[:max_len]
    return token


def _short_hash(text: str) -> str:
    payload = text.encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _build_sampling_cache_path(
    args: argparse.Namespace,
    cache_dir: str,
    rank: int,
    world_size: int,
) -> str:
    llm_abs = os.path.abspath(args.llm_path)
    llm_name = os.path.basename(os.path.normpath(args.llm_path))
    keyword_raw = args.dataset_keyword or "none"
    projection_cfg = {
        "seq_len": args.seq_len,
        "image_size": args.image_size,
        "periodicity": args.periodicity,
        "norm_const": args.norm_const,
        "three_channel": bool(args.three_channel),
    }
    proj_payload = json.dumps(projection_cfg, ensure_ascii=True, sort_keys=True)

    cache_key = (
        f"llm_{_safe_token(llm_name)}_{_short_hash(llm_abs)}__"
        f"kw_{_safe_token(keyword_raw)}_{_short_hash(keyword_raw.lower().strip())}__"
        f"proj_{_short_hash(proj_payload)}__"
        f"ws_{int(world_size)}"
    )
    cache_subdir = os.path.join(cache_dir, cache_key)
    os.makedirs(cache_subdir, exist_ok=True)
    cache_path = os.path.join(cache_subdir, f"round_0_rank{rank}_ws{int(world_size)}.pt")
    return cache_path


def _save_projection_checkpoint(
    output_dir: str,
    rank: int,
    ts_adapter_module: nn.Module,
    critic_module: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        _unwrap_module(ts_adapter_module).state_dict(),
        os.path.join(output_dir, f"ts_image_adapter_rank{rank}.pt"),
    )
    torch.save(
        _unwrap_module(critic_module).state_dict(),
        os.path.join(output_dir, f"critic_rank{rank}.pt"),
    )
    torch.save(
        optimizer.state_dict(),
        os.path.join(output_dir, f"optimizer_rank{rank}.pt"),
    )


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


def _parse_keywords(keyword_text: str) -> list[str]:
    if not keyword_text:
        return []
    return [item.strip().lower() for item in keyword_text.split(",") if item.strip()]


def _collect_recorded_jsonl_paths(
    recorded_root: str,
    dataset_filename: str,
    keywords: list[str],
) -> tuple[list[str], list[str]]:
    if not os.path.isdir(recorded_root):
        raise FileNotFoundError(f"recorded_root not found: {recorded_root}")
    matched_dirs = []
    jsonl_paths = []
    for entry in sorted(os.scandir(recorded_root), key=lambda item: item.name):
        if not entry.is_dir():
            continue
        dirname_lower = entry.name.lower()
        if keywords and not any(keyword in dirname_lower for keyword in keywords):
            continue
        candidate = os.path.join(entry.path, dataset_filename)
        if os.path.isfile(candidate):
            matched_dirs.append(entry.path)
            jsonl_paths.append(candidate)
    return matched_dirs, jsonl_paths


def _merge_jsonl_files(input_paths: list[str], output_path: str) -> tuple[int, int]:
    raw_lines = 0
    kept_lines = 0
    with open(output_path, "w", encoding="utf-8") as out_file:
        for src_path in input_paths:
            src_dir = os.path.dirname(src_path)
            with open(src_path, "r", encoding="utf-8") as in_file:
                for line in in_file:
                    if not line.strip():
                        continue
                    raw_lines += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(record, dict):
                        continue
                    snapshot_raw = record.get("snapshot_path")
                    if isinstance(snapshot_raw, str) and snapshot_raw.strip():
                        snapshot_clean = snapshot_raw.strip()
                        candidates = []
                        if os.path.isabs(snapshot_clean):
                            candidates.append(snapshot_clean)
                        else:
                            # 1) 相对当前 jsonl 文件目录
                            candidates.append(os.path.normpath(os.path.join(src_dir, snapshot_clean)))
                            # 2) 常见写法：data/recorded/...（相对仓库根目录）
                            candidates.append(os.path.normpath(os.path.join(_REPO_ROOT, snapshot_clean)))
                            # 3) 仅文件名时，默认在当前目录 snapshots/ 下
                            candidates.append(
                                os.path.normpath(os.path.join(src_dir, "snapshots", os.path.basename(snapshot_clean)))
                            )
                        resolved = None
                        for cand in candidates:
                            if cand and os.path.isfile(cand):
                                resolved = cand
                                break
                        if resolved is not None:
                            record["snapshot_path"] = resolved
                        else:
                            # 保留原值，后续运行时仍可按现有逻辑尝试加载
                            record["snapshot_path"] = snapshot_clean
                    out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept_lines += 1
    return raw_lines, kept_lines


def _sanitize_jsonl_file(input_path: str, output_path: str) -> tuple[int, int]:
    """Validate jsonl line-by-line and normalize snapshot_path."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"jsonl not found: {input_path}")
    raw, kept = _merge_jsonl_files([input_path], output_path)
    return raw, kept


def _extract_recorded_run_name(snapshot_path: str, recorded_root: str) -> str:
    if not snapshot_path:
        return ""
    snap_abs = os.path.abspath(snapshot_path)
    recorded_abs = os.path.abspath(recorded_root)
    prefix = recorded_abs + os.sep
    if not snap_abs.startswith(prefix):
        return ""
    rel = os.path.relpath(snap_abs, recorded_abs)
    parts = rel.split(os.sep)
    return parts[0] if parts else ""


def _resolve_roadnet_path_from_env_args(cityflow_env_args: dict) -> Optional[str]:
    dic_path = cityflow_env_args.get("dic_path") or {}
    env_conf = cityflow_env_args.get("dic_traffic_env_conf") or {}
    base = dic_path.get("PATH_TO_DATA")
    roadnet = env_conf.get("ROADNET_FILE")
    if not base or not roadnet:
        return None
    candidate = os.path.join(base, roadnet)
    candidate_abs = os.path.abspath(candidate)
    if os.path.isfile(candidate_abs):
        return candidate_abs
    return None


def _get_env_bundle_for_run(
    run_name: str,
    args: argparse.Namespace,
    env_cache: dict[str, dict],
) -> dict:
    if not run_name:
        raise RuntimeError(
            "Cannot infer recorded run name from snapshot_path; "
            "unable to dynamically map to LLMTLCSRun env args."
        )
    cached = env_cache.get(run_name)
    if cached is not None:
        return cached
    cityflow_env_args_path = os.path.join(
        args.llmtlcs_run_root,
        run_name,
        "cityflow_env_init",
        "cityflow_env_args.json",
    )
    if not os.path.isfile(cityflow_env_args_path):
        raise FileNotFoundError(f"cityflow_env_args not found: {cityflow_env_args_path}")
    with open(cityflow_env_args_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    cityflow_env_args = payload.get("cityflow_env_args", payload)
    env = CityFlowEnv(**cityflow_env_args)
    env.reset()
    bundle = {
        "run_name": run_name,
        "env": env,
        "eng": env.eng,
        "roadnet_path": _resolve_roadnet_path_from_env_args(cityflow_env_args),
        "cityflow_env_args_path": cityflow_env_args_path,
    }
    env_cache[run_name] = bundle
    return bundle


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
    policy: Qwen3VLPolicy,
    device: torch.device,
    args: argparse.Namespace,
    env_cache: dict[str, dict],
    generation_kwargs: dict,
    log_interval: int,
) -> dict:
    rewards = []
    total_samples = len(indices)
    sample_counter = 0
    skipped_samples = 0
    eval_start = time.time()
    infer_time_total = 0.0
    for start in range(0, len(indices), 1):
        try:
            idx = indices[start]
            sample = dataset[idx]
            prompt = _append_signal_only_instruction(sample.prompt)
            seq = sample.sequence
            snapshot_path = sample.snapshot_path
            intersection_id = sample.intersection_id
        except Exception as exc:
            skipped_samples += 1
            if log_interval > 0:
                print(f"[测试跳过] idx={indices[start]} | 样本读取失败: {exc}")
            continue

        ts_tensor = torch.tensor([seq], dtype=torch.float32, device=device)
        with torch.no_grad():
            images = ts_adapter.generate(ts_tensor)
            _sync_if_cuda(device)
            infer_start = time.perf_counter()
            prepared_inputs = policy.prepare_inputs([prompt], images)
            batch_actions, _ = policy.generate_actions_from_prepared(
                prepared_inputs,
                generation_kwargs,
            )
            _sync_if_cuda(device)
            infer_elapsed = time.perf_counter() - infer_start
            infer_time_total += infer_elapsed
        action = batch_actions[0]  # 不一定能获得动作
        # --- in your sampling loop, after you get `action` from policy.generate_actions(...) ---
        # NOTE: ACTION_LIST length is your discrete action space size
        # action_id = parse_action_id(action, num_actions=len(ACTION_LIST), default=0)  #这里不能用action_id，logprob 必须对应“实际执行的动作”


        try:
            if not snapshot_path:
                raise RuntimeError("empty snapshot_path")
            run_name = _extract_recorded_run_name(snapshot_path, args.recorded_root)
            env_bundle = _get_env_bundle_for_run(run_name, args, env_cache)
            eng = env_bundle["eng"]
            roadnet_path = env_bundle.get("roadnet_path")

            eng.load_from_file(snapshot_path)
            eng.set_tl_phase(intersection_id, action)

            if not roadnet_path:
                raise RuntimeError(
                    f"roadnet_path not found in env args for run={run_name}, "
                    f"args_file={env_bundle.get('cityflow_env_args_path')}"
                )
            reward = -mean_waiting_time(eng, intersection_id, roadnet_path)
        except Exception as exc:
            skipped_samples += 1
            if log_interval > 0:
                print(f"[测试跳过] idx={idx} | snapshot={_shorten(snapshot_path, 80)} | 原因: {exc}")
            continue
        rewards.append(reward)

        sample_counter += 1
        if log_interval > 0 and (
            sample_counter == 1
            or sample_counter == total_samples
            or sample_counter % log_interval == 0
        ):
            avg_reward_so_far = float(np.mean(rewards)) if rewards else 0.0
            elapsed = time.time() - eval_start
            infer_avg = infer_time_total / max(1, sample_counter)
            print(
                "[测试进度] {}/{} | 即时奖励(负等待)={:.4f} | 平均奖励={:.4f} | 动作={} | 用时{:.1f}s | 推理{:.0f}ms | 推理累计{:.1f}s(均{:.3f}s/样本)".format(
                    sample_counter,
                    total_samples,
                    reward,
                    avg_reward_so_far,
                    _shorten(action),
                    elapsed,
                    infer_elapsed * 1000.0,
                    infer_time_total,
                    infer_avg,
                )
            )
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    return {
        "avg_reward": avg_reward,
        "avg_waiting": -avg_reward,
        "num_samples": len(rewards),
        "skipped_samples": skipped_samples,
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
    _configure_cuda_allocator()
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

    dataset_keywords = _parse_keywords(args.dataset_keyword)
    dataset_path = args.jsonl_path
    dataset_source_desc = f"single={dataset_path}"
    if dataset_keywords:
        dataset_payload = None
        if _is_rank0():
            matched_dirs, jsonl_paths = _collect_recorded_jsonl_paths(
                recorded_root=args.recorded_root,
                dataset_filename=args.dataset_filename,
                keywords=dataset_keywords,
            )
            if not jsonl_paths:
                raise RuntimeError(
                    "No matched dataset files. "
                    f"recorded_root={args.recorded_root}, "
                    f"keywords={dataset_keywords}, filename={args.dataset_filename}"
                )
            merged_dir = os.path.join(
                _REPO_ROOT, "records", "projection_train", "merged_datasets"
            )
            os.makedirs(merged_dir, exist_ok=True)
            merged_name = "merged_{}_{}.jsonl".format(
                "_".join(dataset_keywords),
                time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            )
            merged_path = os.path.join(merged_dir, merged_name)
            raw_line_count, line_count = _merge_jsonl_files(jsonl_paths, merged_path)
            dataset_payload = {
                "dataset_path": merged_path,
                "line_count": line_count,
                "matched_dirs": matched_dirs,
                "jsonl_paths": jsonl_paths,
                "raw_line_count": raw_line_count,
            }
        dataset_payload = _broadcast_object(dataset_payload)
        dataset_path = dataset_payload["dataset_path"]
        dataset_source_desc = "keyword={} matched_dirs={} files={} raw_lines={} kept_lines={} merged_path={}".format(
            ",".join(dataset_keywords),
            len(dataset_payload["matched_dirs"]),
            len(dataset_payload["jsonl_paths"]),
            dataset_payload["raw_line_count"],
            dataset_payload["line_count"],
            dataset_path,
        )
        if _is_rank0():
            print("[数据集聚合] " + dataset_source_desc)
    else:
        dataset_payload = None
        if _is_rank0():
            sanitized_dir = os.path.join(
                _REPO_ROOT, "records", "projection_train", "merged_datasets"
            )
            os.makedirs(sanitized_dir, exist_ok=True)
            sanitized_name = "sanitized_{}_{}.jsonl".format(
                os.path.splitext(os.path.basename(dataset_path))[0],
                time.strftime("%Y%m%d_%H%M%S", time.localtime()),
            )
            sanitized_path = os.path.join(sanitized_dir, sanitized_name)
            raw_lines, kept_lines = _sanitize_jsonl_file(dataset_path, sanitized_path)
            dataset_payload = {
                "dataset_path": sanitized_path,
                "raw_line_count": raw_lines,
                "line_count": kept_lines,
            }
        dataset_payload = _broadcast_object(dataset_payload)
        dataset_path = dataset_payload["dataset_path"]
        dataset_source_desc = "single={} raw_lines={} kept_lines={} sanitized_path={}".format(
            args.jsonl_path,
            dataset_payload["raw_line_count"],
            dataset_payload["line_count"],
            dataset_path,
        )
        if _is_rank0():
            print("[数据集清洗] " + dataset_source_desc)

    # 加载时间序列提示词样本（用于生成动作）
    dataset = TrafficPromptSeriesDataset(
        path=dataset_path,
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
    if _is_rank0():
        print(f"数据集来源：{dataset_source_desc}")
    test_ratio = max(0.0, min(0.9, float(args.test_ratio)))
    train_indices, test_indices = _split_indices(len(dataset), test_ratio, args.seed)
    train_indices = _shard_indices(train_indices, rank, world_size)
    if dist.is_initialized():
        local_len = torch.tensor(len(train_indices), device=device)
        dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
        min_len = int(local_len.item())
        train_indices = train_indices[:min_len]
        if min_len == 0:
            raise RuntimeError("No training samples left after DDP sharding.")
    print(
        "数据集拆分完成：训练样本数={}，测试样本数={}（{:.0f}%测试集）。".format(
            len(train_indices), len(test_indices), test_ratio * 100.0
        )
    )

    # 目录仅用于训练日志和模型输出；仿真环境将按 snapshot 动态加载。
    roadnet_path = args.roadnet_src
    traffic_path = args.traffic_src
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

    # 按 snapshot_path -> recorded 子目录 -> LLMTLCSRun 同名目录 动态加载并缓存 CityFlowEnv
    env_cache: dict[str, dict] = {}
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
    clip_value_loss_enabled = True
    clip_value_range = args.clip_range
    mixed_precision_enabled = True
    mixed_precision_dtype = None
    grad_scaler_enabled = False
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
    llm_fsdp_enabled = (
        args.llm_fsdp == "on"
        or (
            args.llm_fsdp == "auto"
            and dist.is_initialized()
            and world_size > 1
            and device.type == "cuda"
        )
    )
    policy = Qwen3VLPolicy(
        model_path=args.llm_path,
        device=llm_primary_device,
        device_map=llm_device_map,
        max_memory=llm_max_memory,
        torch_dtype=torch.bfloat16,
        fsdp_enabled=llm_fsdp_enabled,
        fsdp_sharding_strategy="full_shard",
    )
    # 冻结 LLM 参数并开启推理缓存（关闭 gradient checkpointing）
    if policy.model is not None:
        model_for_setup = policy.model.module if hasattr(policy.model, "module") else policy.model
        model_for_setup.requires_grad_(False)
        model_for_setup.eval()
        policy.set_gradient_checkpointing(False)
        policy.set_use_cache(True)

    def _set_llm_train_mode(enabled: bool) -> None:
        policy.set_gradient_checkpointing(enabled)
        policy.set_use_cache(not enabled)



    # LLM 生成超参
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": args.do_sample,
        "pad_token_id": policy.tokenizer.pad_token_id,
        "eos_token_id": policy.tokenizer.eos_token_id,
    }

    cache_dir = os.path.join("records", "projection_train", "sampling_cache_qwen3_vl")
    cache_path = _build_sampling_cache_path(args, cache_dir, rank, world_size)
    os.makedirs(cache_dir, exist_ok=True)
    _barrier()

    log_lines = []
    log_dir = dic_path["PATH_TO_MODEL"]
    os.makedirs(log_dir, exist_ok=True)
    log_date = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join(log_dir, f"projection_ppo_qwen3_vl_{log_date}_rank{rank}.txt")
    stage_csv_path = os.path.join(log_dir, f"projection_ppo_qwen3_vl_stage_{log_date}_rank{rank}.csv")
    epoch_csv_path = os.path.join(log_dir, f"projection_ppo_qwen3_vl_epoch_{log_date}_rank{rank}.csv")
    with open(stage_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_STAGE_LOG_FIELDS)
        writer.writeheader()
    with open(epoch_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=_EPOCH_LOG_FIELDS)
        writer.writeheader()
    training_config_text = _format_training_config(
        args=args,
        clip_value_loss_enabled=clip_value_loss_enabled,
        clip_value_range=clip_value_range,
        mixed_precision_enabled=mixed_precision_enabled,
        mixed_precision_dtype=mixed_precision_dtype,
        grad_scaler_enabled=grad_scaler_enabled,
    )
    if _is_rank0():
        print("[训练配置与超参]\n" + training_config_text)
    log_lines.append("[训练配置与超参]\n" + training_config_text)
    log_lines.append(f"DDP rank={rank}, world_size={world_size}, device={device}.")
    log_lines.append(
        f"LLM family={policy.model_family}, multimodal={policy.is_multimodal}, fsdp={policy.is_fsdp}."
    )
    log_lines.append(
        "数据集拆分：训练样本数={}，测试样本数={}（{:.0f}%测试集）。".format(
            len(train_indices), len(test_indices), test_ratio * 100.0
        )
    )
    log_lines.append("数据集来源：" + dataset_source_desc)
    log_lines.append(
        f"动态环境加载：recorded_root={args.recorded_root}, llmtlcs_run_root={args.llmtlcs_run_root}"
    )
    log_lines.append(f"采样缓存（round_0复用）：{cache_path}")
    log_lines.append(f"阶段耗时日志：csv={stage_csv_path}")
    log_lines.append(f"Epoch日志：csv={epoch_csv_path}")

    model_path = _resolve_model_path(args.ppo_model_path, dataset_path)
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
                args=args,
                env_cache=env_cache,
                generation_kwargs=generation_kwargs,
                log_interval=args.log_interval,
            )
            summary = (
                f"[测试汇总] 样本数={test_result['num_samples']} | "
                f"跳过样本数={test_result.get('skipped_samples', 0)} | "
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
    for round_idx in range(args.num_rounds):
        round_no = round_idx + 1
        print(f"\n======== 开始第 {round_no}/{args.num_rounds} 轮训练 ========")
        log_lines.append(f"[Round {round_no}] 开始训练。")

        round_dir = os.path.join(dic_path["PATH_TO_MODEL"], f"round_{round_idx}")
        epoch_ckpt_dir = os.path.join(round_dir, "epochs")
        best_ckpt_dir = os.path.join(round_dir, "best")
        latest_ckpt_dir = os.path.join(round_dir, "latest")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        os.makedirs(best_ckpt_dir, exist_ok=True)
        os.makedirs(latest_ckpt_dir, exist_ok=True)

        prompts: list[str] = []
        sequences: list[list[float]] = []
        actions: list[str] = []
        old_logprobs: list[float] = []
        rewards: list[float] = []
        skipped_samples = 0

        use_cached = False
        if round_idx == 0:
            cache_available = os.path.isfile(cache_path)
            if dist.is_initialized():
                cache_flag = torch.tensor(1 if cache_available else 0, device=device)
                dist.all_reduce(cache_flag, op=dist.ReduceOp.MIN)
                cache_available = bool(int(cache_flag.item()))
            if cache_available:
                load_ok = True
                load_error = None
                try:
                    cached_payload = torch.load(cache_path, map_location="cpu")
                    prompts = list(cached_payload.get("prompts", []))
                    sequences = list(cached_payload.get("sequences", []))
                    actions = list(cached_payload.get("actions", []))
                    old_logprobs = list(cached_payload.get("old_logprobs", []))
                    rewards = list(cached_payload.get("rewards", []))
                    if not (
                        len(prompts)
                        == len(sequences)
                        == len(actions)
                        == len(old_logprobs)
                        == len(rewards)
                    ):
                        raise RuntimeError("cached tensors have inconsistent lengths")
                    load_ok = len(prompts) > 0
                except Exception as exc:
                    load_ok = False
                    load_error = exc
                if dist.is_initialized():
                    load_flag = torch.tensor(1 if load_ok else 0, device=device)
                    dist.all_reduce(load_flag, op=dist.ReduceOp.MIN)
                    load_ok = bool(int(load_flag.item()))
                if load_ok:
                    use_cached = True
                    if _is_rank0():
                        print(f"[采样缓存] 命中 round_0 缓存：{cache_path} | 样本数={len(prompts)}")
                    log_lines.append(
                        f"[Round {round_no}] 复用 round_0 采样缓存：{cache_path} | 样本数={len(prompts)}"
                    )
                else:
                    if _is_rank0():
                        detail = f"{load_error}" if load_error is not None else "some rank failed"
                        print(f"[采样缓存] 全局一致性检查失败，将重新采样：{detail}")
                    log_lines.append(
                        f"[Round {round_no}] 缓存一致性失败，改为重新采样：{load_error if load_error else 'some rank failed'}"
                    )
                    prompts = []
                    sequences = []
                    actions = []
                    old_logprobs = []
                    rewards = []
                    use_cached = False
            else:
                if _is_rank0():
                    print("[采样缓存] 未命中全局一致缓存，将重新采样。")
                log_lines.append(f"[Round {round_no}] 缓存未全员可用，改为重新采样。")

        if not use_cached:
            total_samples = len(train_indices)
            collect_start = time.time()
            ts_image_time = 0.0
            llm_time = 0.0
            llm_infer_time = 0.0
            llm_logprob_time = 0.0
            env_load_time = 0.0
            env_phase_time = 0.0
            reward_calc_time = 0.0

            indices = list(train_indices)
            
            random.shuffle(indices)
            # indices=indices[:20]
            sample_counter = 0
            batch_size = max(1, int(args.batch_size))

            for batch_start in range(0, len(indices), batch_size):
                batch_total_start = time.perf_counter()
                batch_indices = indices[batch_start : batch_start + batch_size]
                batch_samples = [dataset[i] for i in batch_indices]
                batch_prompts = [_append_signal_only_instruction(sample.prompt) for sample in batch_samples]
                batch_seqs = [sample.sequence for sample in batch_samples]
                batch_snapshot_paths = [sample.snapshot_path for sample in batch_samples]
                batch_intersection_ids = [sample.intersection_id for sample in batch_samples]

                batch_ts_image_s = 0.0
                batch_llm_generate_s = 0.0
                batch_llm_logprob_s = 0.0
                batch_env_load_s = 0.0
                batch_env_set_phase_s = 0.0
                batch_reward_calc_s = 0.0

                ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                with torch.no_grad():
                    _sync_if_cuda(device)
                    ts_start = time.perf_counter()
                    images = ts_adapter.generate(ts_tensor)
                    _sync_if_cuda(device)
                    ts_elapsed = time.perf_counter() - ts_start
                    ts_image_time += ts_elapsed
                    batch_ts_image_s += ts_elapsed

                    _sync_if_cuda(device)
                    llm_start = time.perf_counter()
                    prepared_inputs = policy.prepare_inputs(batch_prompts, images, use_tensor_preprocess=True)
                    batch_actions, _ = policy.generate_actions_from_prepared(prepared_inputs, generation_kwargs)
                    _sync_if_cuda(device)
                    llm_elapsed = time.perf_counter() - llm_start
                    llm_time += llm_elapsed
                    llm_infer_time += llm_elapsed
                    batch_llm_generate_s += llm_elapsed

                    _sync_if_cuda(device)
                    llm_start = time.perf_counter()
                    batch_logprobs = (
                        policy.action_logprob_from_prepared(prepared_inputs, batch_actions).detach().cpu().tolist()
                    )
                    _sync_if_cuda(device)
                    llm_elapsed = time.perf_counter() - llm_start
                    llm_time += llm_elapsed
                    llm_logprob_time += llm_elapsed
                    batch_llm_logprob_s += llm_elapsed

                for prompt, seq, action, logprob, snapshot_path, intersection_id in zip(
                    batch_prompts,
                    batch_seqs,
                    batch_actions,
                    batch_logprobs,
                    batch_snapshot_paths,
                    batch_intersection_ids,
                ):
                    load_start = time.perf_counter()
                    try:
                        if not snapshot_path:
                            raise RuntimeError("empty snapshot_path")
                        run_name = _extract_recorded_run_name(snapshot_path, args.recorded_root)
                        env_bundle = _get_env_bundle_for_run(run_name, args, env_cache)
                        eng = env_bundle["eng"]
                        roadnet_path_local = env_bundle.get("roadnet_path")
                        eng.load_from_file(snapshot_path)
                    except Exception as exc:
                        skipped_samples += 1
                        if args.log_interval > 0:
                            print(f"[采样跳过] snapshot={_shorten(snapshot_path, 80)} | 原因: {exc}")
                        continue
                    load_elapsed = time.perf_counter() - load_start
                    env_load_time += load_elapsed
                    batch_env_load_s += load_elapsed

                    set_phase_start = time.perf_counter()
                    eng.set_tl_phase(intersection_id, action)
                    set_phase_elapsed = time.perf_counter() - set_phase_start
                    env_phase_time += set_phase_elapsed
                    batch_env_set_phase_s += set_phase_elapsed

                    reward_start = time.perf_counter()
                    if not roadnet_path_local:
                        skipped_samples += 1
                        if args.log_interval > 0:
                            print(
                                "[采样跳过] roadnet_path not found in env args: "
                                f"run={run_name}, args_file={env_bundle.get('cityflow_env_args_path')}"
                            )
                        continue
                    reward = -mean_waiting_time(eng, intersection_id, roadnet_path_local)
                    reward_elapsed = time.perf_counter() - reward_start
                    reward_calc_time += reward_elapsed
                    batch_reward_calc_s += reward_elapsed

                    prompts.append(prompt)
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
                        llm_infer_avg = llm_infer_time / max(1, sample_counter)
                        llm_logprob_avg = llm_logprob_time / max(1, sample_counter)
                        print(
                            "[采样进度] {}/{} | 即时奖励(负等待)={:.4f} | 平均奖励={:.4f} | 动作={} | 用时{:.1f}s | "
                            "ts_images累计{:.1f}s(均{:.3f}s/样本) | LLM累计{:.1f}s(均{:.3f}s/样本) | "
                            "推理累计{:.1f}s(均{:.3f}s/样本) | logprob累计{:.1f}s(均{:.3f}s/样本) | "
                            "env加载累计{:.1f}s | 奖励累计{:.1f}s".format(
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
                                llm_infer_time,
                                llm_infer_avg,
                                llm_logprob_time,
                                llm_logprob_avg,
                                env_load_time,
                                reward_calc_time,
                            )
                        )

                batch_total_elapsed = time.perf_counter() - batch_total_start
                latest_rewards = rewards[-len(batch_indices) :] if rewards else []
                stage_row = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "rank": rank,
                    "round": round_no,
                    "stage": "sampling_batch",
                    "batch_id": batch_start // batch_size + 1,
                    "sample_counter": sample_counter,
                    "total_samples": total_samples,
                    "batch_size": len(batch_indices),
                    "reward": float(np.mean(latest_rewards)) if latest_rewards else 0.0,
                    "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
                    "ts_image_s": round(batch_ts_image_s, 6),
                    "llm_generate_s": round(batch_llm_generate_s, 6),
                    "llm_logprob_s": round(batch_llm_logprob_s, 6),
                    "env_load_s": round(batch_env_load_s, 6),
                    "env_set_phase_s": round(batch_env_set_phase_s, 6),
                    "reward_calc_s": round(batch_reward_calc_s, 6),
                    "critic_s": "",
                    "entropy_s": "",
                    "backward_opt_s": "",
                    "batch_total_s": round(batch_total_elapsed, 6),
                }
                _append_stage_log(stage_csv_path, stage_row)
                if _is_rank0() and (
                    batch_start == 0
                    or sample_counter == total_samples
                    or ((batch_start // batch_size + 1) % max(1, args.log_interval) == 0)
                ):
                    print("[采样阶段统计] " + json.dumps(stage_row, ensure_ascii=False, sort_keys=True))

            if skipped_samples > 0 and _is_rank0():
                print(f"[采样阶段] 已跳过异常样本数={skipped_samples}")

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
                if _is_rank0():
                    print(f"[采样缓存] 已保存 round_0 缓存：{cache_path} | 样本数={len(prompts)}")
                log_lines.append(f"[Round {round_no}] 已保存 round_0 采样缓存：{cache_path} | 样本数={len(prompts)}")
                _barrier()

        # 同步各 rank 的样本数，避免后续 PPO/collectives 进入不同步
        if dist.is_initialized():
            local_len = torch.tensor(len(prompts), device=device)
            dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
            min_len = int(local_len.item())
            if len(prompts) != min_len:
                prompts = prompts[:min_len]
                sequences = sequences[:min_len]
                actions = actions[:min_len]
                old_logprobs = old_logprobs[:min_len]
                rewards = rewards[:min_len]

        if len(prompts) == 0:
            raise RuntimeError("All collected samples were skipped due to invalid jsonl/snapshot/env mapping.")

        print("[步骤1] 全局计算values和advantages...")
        returns = torch.tensor(rewards, dtype=torch.float32, device=loss_device)

        all_values = []
        batch_size_for_value = 8
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size_for_value):
                batch_seqs = sequences[i : i + batch_size_for_value]
                ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                images = ts_adapter.generate(ts_tensor)
                batch_values = critic(images)
                all_values.append(batch_values.to(loss_device))

        values_all = torch.cat(all_values, dim=0)
        advantages = returns - values_all
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        print(f"  - Returns: mean={returns.mean():.4f}, std={returns.std():.4f}")
        print(f"  - Values: mean={values_all.mean():.4f}, std={values_all.std():.4f}")
        print(f"  - Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")

        indices = list(range(len(prompts)))
        minibatch_total = math.ceil(len(indices) / max(1, args.mini_batch_size))
        best_epoch_loss = float("inf")
        best_epoch = -1
        epochs_no_improve = 0
        round_stop_reason = ""

        _set_llm_train_mode(True)
        
        for epoch_idx in range(args.ppo_epochs):
            epoch_no = epoch_idx + 1
            if _is_rank0():
                print(f"[PPO] Epoch {epoch_no}/{args.ppo_epochs} | mini-batches={minibatch_total}")
            random.shuffle(indices)
            
            epoch_losses = []
            epoch_clip_fracs = []
            epoch_approx_kls = []
            stop_this_epoch_kl = False
            nan_or_inf_triggered = False

            for start in range(0, len(indices), args.mini_batch_size):
                mb_total_start = time.perf_counter()
                mb_idx = start // args.mini_batch_size + 1
                debug_mb = bool(_is_rank0() and epoch_idx == 0 and mb_idx <= 2)
                batch_idx = indices[start : start + args.mini_batch_size]
                batch_prompts = [prompts[i] for i in batch_idx]
                batch_actions = [actions[i] for i in batch_idx]
                batch_seqs = [sequences[i] for i in batch_idx]

                critic_elapsed = 0.0
                entropy_elapsed = 0.0
                backward_opt_elapsed = 0.0

                # 确保训练路径不处于 inference_mode，避免 inference tensor 参与反向传播
                with torch.inference_mode(False):
                    ts_start = time.perf_counter()
                    ts_tensor = torch.tensor(batch_seqs, dtype=torch.float32, device=device)
                    images = ts_adapter.generate(ts_tensor)
                    ts_elapsed = time.perf_counter() - ts_start
                    if debug_mb:
                        print(f"[验证] images.requires_grad: {images.requires_grad}")

                    logprob_start = time.perf_counter()
                    prepared_inputs = policy.prepare_inputs(batch_prompts, images, use_tensor_preprocess=True)
                    new_logprobs = policy.action_logprob_from_prepared(prepared_inputs, batch_actions)
                    logprob_elapsed = time.perf_counter() - logprob_start
                if debug_mb:
                    grad_fn_name = (
                        type(new_logprobs.grad_fn).__name__ if new_logprobs.grad_fn is not None else None
                    )
                    print(
                        "[验证] new_logprobs requires_grad: {} grad_fn: {}".format(
                            new_logprobs.requires_grad, grad_fn_name
                        )
                    )

                old_logprob_tensor = torch.tensor(
                    [old_logprobs[i] for i in batch_idx],
                    dtype=new_logprobs.dtype,
                    device=new_logprobs.device,
                )
                returns_b = returns[batch_idx].to(new_logprobs.device)
                advantages_b = advantages[batch_idx].to(new_logprobs.device)
                old_values_b = values_all[batch_idx].to(new_logprobs.device)

                critic_start = time.perf_counter()
                values = critic(images)
                critic_elapsed = time.perf_counter() - critic_start

                ratio = torch.exp(new_logprobs - old_logprob_tensor)
                if debug_mb:
                    delta = (new_logprobs - old_logprob_tensor).abs().max().item()
                    print(f"[验证] max|new-old|: {delta:.6e}")
                    print(
                        "[验证] ratio mean/std/min/max: {:.6f} {:.6f} {:.6f} {:.6f}".format(
                            ratio.mean().item(),
                            ratio.std().item(),
                            ratio.min().item(),
                            ratio.max().item(),
                        )
                    )
                with torch.no_grad():
                    log_ratio = new_logprobs - old_logprob_tensor
                    approx_kl_local = torch.mean((torch.exp(log_ratio) - 1.0) - log_ratio).item()
                approx_kl = _ddp_allreduce_max(approx_kl_local, new_logprobs.device)

                surr1 = ratio * advantages_b
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages_b
                policy_loss = -torch.min(surr1, surr2).mean()

                vpredclipped = old_values_b + torch.clamp(values - old_values_b, -args.clip_range, args.clip_range)
                vf_loss1 = (values - returns_b).pow(2)
                vf_loss2 = (vpredclipped - returns_b).pow(2)
                value_loss = 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))

                entropy_loss = torch.zeros((), device=new_logprobs.device, dtype=new_logprobs.dtype)
                ent_coef = float(args.ent_coef)
                if ent_coef > 0.0:
                    try:
                        entropy_start = time.perf_counter()
                        entropy = policy.compute_entropy(
                            batch_prompts,
                            images,
                            prepared_inputs=prepared_inputs,
                        )
                        entropy_elapsed = time.perf_counter() - entropy_start
                        entropy_loss = -entropy.mean()
                    except AttributeError:
                        ent_coef = 0.0
                        if epoch_idx == 0:
                            print("[警告] Policy未实现compute_entropy方法，跳过entropy bonus")

                loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
                local_bad_loss = bool(args.stop_on_nan_inf and not torch.isfinite(loss))
                global_bad_loss = _ddp_any_true(local_bad_loss, loss.device)
                if global_bad_loss:
                    nan_or_inf_triggered = True
                    round_stop_reason = "nan_or_inf_loss"
                    if _is_rank0():
                        print(f"[早停] Epoch {epoch_no} mb {mb_idx} 出现 NaN/Inf loss，停止训练。")
                    break

                backward_start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                adapter_param_before = None
                if debug_mb:
                    try:
                        adapter_param_before = (
                            next(ts_adapter.learnable_image_module.parameters()).detach().clone()
                        )
                    except StopIteration:
                        adapter_param_before = None
                loss.backward()
                if debug_mb:
                    grads = [p.grad for p in ts_adapter.learnable_image_module.parameters()]
                    if grads:
                        none_count = sum(1 for g in grads if g is None)
                        print(
                            f"[验证] adapter grad None ratio: {none_count}/{len(grads)}"
                        )
                    else:
                        print("[验证] adapter grad None ratio: n/a (no params)")
                if epoch_idx == 0 and mb_idx == 1:
                    verify_freeze_and_trainable(
                        policy=policy,
                        ts_adapter=ts_adapter,
                        optimizer=optimizer,
                        loss=None,
                        rank0_only=True,
                    )
                torch.nn.utils.clip_grad_norm_(
                    list(ts_adapter.learnable_image_module.parameters()) + list(critic.parameters()),
                    max_norm=args.max_grad_norm,
                )
                optimizer.step()
                if debug_mb and adapter_param_before is not None:
                    try:
                        adapter_param_after = next(ts_adapter.learnable_image_module.parameters()).detach()
                        adapter_delta = (adapter_param_after - adapter_param_before).abs().mean().item()
                        print(f"[验证] adapter param delta mean: {adapter_delta:.6e}")
                    except StopIteration:
                        print("[验证] adapter param delta mean: n/a (no params)")
                backward_opt_elapsed = time.perf_counter() - backward_start

                clip_frac = (torch.abs(ratio - 1.0) > args.clip_range).float().mean().item()
                policy_loss_v = float(policy_loss.detach().cpu().item())
                value_loss_v = float(value_loss.detach().cpu().item())
                entropy_loss_v = float(entropy_loss.detach().cpu().item()) if isinstance(entropy_loss, torch.Tensor) else 0.0
                total_loss_v = float(loss.detach().cpu().item())
                mb_total_elapsed = time.perf_counter() - mb_total_start
                minibatch_rewards = [rewards[i] for i in batch_idx]
                minibatch_reward = float(np.mean(minibatch_rewards)) if minibatch_rewards else 0.0
                avg_reward_all = float(np.mean(rewards)) if rewards else 0.0

                stage_row = {
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                    "rank": rank,
                    "round": round_no,
                    "stage": "ppo_minibatch",
                    "batch_id": mb_idx,
                    "sample_counter": len(batch_idx),
                    "total_samples": len(indices),
                    "batch_size": len(batch_idx),
                    "reward": minibatch_reward,
                    "avg_reward": avg_reward_all,
                    "ts_image_s": round(ts_elapsed, 6),
                    "llm_generate_s": "",
                    "llm_logprob_s": round(logprob_elapsed, 6),
                    "env_load_s": "",
                    "env_set_phase_s": "",
                    "reward_calc_s": 0.0,
                    "critic_s": round(critic_elapsed, 6),
                    "entropy_s": round(entropy_elapsed, 6),
                    "backward_opt_s": round(backward_opt_elapsed, 6),
                    "batch_total_s": round(mb_total_elapsed, 6),
                }
                _append_stage_log(stage_csv_path, stage_row)

                if _is_rank0() and args.log_interval > 0 and (
                    mb_idx == 1 or mb_idx % args.log_interval == 0 or mb_idx == minibatch_total
                ):
                    print(
                        "[PPO] Epoch {}/{} | mb {}/{} | loss={:.6f} | policy={:.6f} | value={:.6f} | "
                        "entropy={:.6f} | clip={:.3f} | approx_kl={:.6f}".format(
                            epoch_no,
                            args.ppo_epochs,
                            mb_idx,
                            minibatch_total,
                            total_loss_v,
                            policy_loss_v,
                            value_loss_v,
                            entropy_loss_v,
                            clip_frac,
                            approx_kl,
                        )
                    )
                    print("[PPO阶段统计] " + json.dumps(stage_row, ensure_ascii=False, sort_keys=True))

                epoch_losses.append(
                    {
                        "policy_loss": policy_loss_v,
                        "value_loss": value_loss_v,
                        "entropy_loss": entropy_loss_v,
                        "total_loss": total_loss_v,
                    }
                )
                epoch_clip_fracs.append(clip_frac)
                epoch_approx_kls.append(float(approx_kl))

                if args.target_kl > 0 and approx_kl > args.target_kl:
                    stop_this_epoch_kl = True
                    if _is_rank0():
                        print(
                            f"[PPO早停-epoch] Epoch {epoch_no} mb {mb_idx} "
                            f"approx_kl={approx_kl:.6f} > target_kl={args.target_kl:.6f}，提前结束该 epoch。"
                        )
                    break

            if nan_or_inf_triggered:
                break
            no_valid_local = len(epoch_losses) == 0
            no_valid_global = _ddp_any_true(no_valid_local, device)
            if no_valid_global:
                if not round_stop_reason:
                    round_stop_reason = "no_valid_minibatch"
                break

            avg_policy = float(np.mean([item["policy_loss"] for item in epoch_losses]))
            avg_value = float(np.mean([item["value_loss"] for item in epoch_losses]))
            avg_entropy = float(np.mean([item["entropy_loss"] for item in epoch_losses]))
            avg_total = float(np.mean([item["total_loss"] for item in epoch_losses]))
            avg_clip = float(np.mean(epoch_clip_fracs)) if epoch_clip_fracs else 0.0
            avg_kl = float(np.mean(epoch_approx_kls)) if epoch_approx_kls else 0.0

            epoch_output_dir = os.path.join(epoch_ckpt_dir, f"epoch_{epoch_no}")
            _save_projection_checkpoint(
                output_dir=epoch_output_dir,
                rank=rank,
                ts_adapter_module=ts_adapter.learnable_image_module,
                critic_module=critic,
                optimizer=optimizer,
            )
            _save_projection_checkpoint(
                output_dir=latest_ckpt_dir,
                rank=rank,
                ts_adapter_module=ts_adapter.learnable_image_module,
                critic_module=critic,
                optimizer=optimizer,
            )

            state_payload = None
            if _is_rank0():
                is_best = avg_total < (best_epoch_loss - args.early_stop_min_delta)
                if is_best:
                    best_epoch_loss = avg_total
                    best_epoch = epoch_no
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if args.early_stop_loss_threshold is not None and avg_total <= args.early_stop_loss_threshold:
                    round_stop_reason = (
                        f"loss_threshold(avg_total={avg_total:.6f} <= {args.early_stop_loss_threshold:.6f})"
                    )
                elif args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
                    round_stop_reason = f"patience_exhausted({epochs_no_improve})"

                state_payload = {
                    "is_best": is_best,
                    "best_epoch_loss": best_epoch_loss,
                    "best_epoch": best_epoch,
                    "epochs_no_improve": epochs_no_improve,
                    "round_stop_reason": round_stop_reason,
                }

            state_payload = _broadcast_object(state_payload)
            is_best = bool(state_payload["is_best"])
            best_epoch_loss = float(state_payload["best_epoch_loss"])
            best_epoch = int(state_payload["best_epoch"])
            epochs_no_improve = int(state_payload["epochs_no_improve"])
            round_stop_reason = str(state_payload["round_stop_reason"] or "")

            if is_best:
                _save_projection_checkpoint(
                    output_dir=best_ckpt_dir,
                    rank=rank,
                    ts_adapter_module=ts_adapter.learnable_image_module,
                    critic_module=critic,
                    optimizer=optimizer,
                )

            epoch_stop_reasons = []
            if stop_this_epoch_kl:
                epoch_stop_reasons.append("target_kl_exceeded")
            if round_stop_reason:
                epoch_stop_reasons.append(round_stop_reason)

            epoch_log_row = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "rank": rank,
                "round": round_no,
                "epoch": epoch_no,
                "num_minibatches": len(epoch_losses),
                "avg_policy_loss": round(avg_policy, 8),
                "avg_value_loss": round(avg_value, 8),
                "avg_entropy_loss": round(avg_entropy, 8),
                "avg_total_loss": round(avg_total, 8),
                "avg_clip_frac": round(avg_clip, 8),
                "avg_approx_kl": round(avg_kl, 8),
                "test_num_samples": "",
                "test_skipped_samples": "",
                "test_avg_reward": "",
                "test_avg_waiting": "",
                "best_so_far": "yes" if is_best else "no",
                "early_stop_reason": ";".join(epoch_stop_reasons),
            }

            _barrier()
            run_eval_all_ranks = dist.is_initialized() and world_size > 1
            test_result = None
            if run_eval_all_ranks or _is_rank0():
                _set_llm_train_mode(False)
                adapter_module = ts_adapter.learnable_image_module
                prev_training = adapter_module.training
                adapter_module.eval()
                with torch.no_grad():
                    test_result = evaluate_policy(
                        dataset=dataset,
                        indices=test_indices,
                        ts_adapter=ts_adapter,
                        policy=policy,
                        device=device,
                        args=args,
                        env_cache=env_cache,
                        generation_kwargs=generation_kwargs,
                        log_interval=args.log_interval,
                    )
                if prev_training:
                    adapter_module.train()
                if not round_stop_reason and epoch_idx + 1 < args.ppo_epochs:
                    _set_llm_train_mode(True)
            if dist.is_initialized():
                if run_eval_all_ranks:
                    test_result = _broadcast_object(test_result if _is_rank0() else None)
                else:
                    test_result = _broadcast_object(test_result)
            if test_result is not None:
                epoch_log_row["test_num_samples"] = int(test_result["num_samples"])
                epoch_log_row["test_skipped_samples"] = int(test_result.get("skipped_samples", 0))
                epoch_log_row["test_avg_reward"] = round(float(test_result["avg_reward"]), 8)
                epoch_log_row["test_avg_waiting"] = round(float(test_result["avg_waiting"]), 8)
                if _is_rank0():
                    test_summary = (
                        f"[Epoch测试] round={round_no} epoch={epoch_no}/{args.ppo_epochs} | "
                        f"样本数={test_result['num_samples']} | "
                        f"跳过样本数={test_result.get('skipped_samples', 0)} | "
                        f"平均奖励(负等待)={test_result['avg_reward']:.4f} | "
                        f"估计平均等待时间={test_result['avg_waiting']:.4f}"
                    )
                    print(test_summary)
                    log_lines.append(test_summary)
            _barrier()
            _append_epoch_log(epoch_csv_path, epoch_log_row)

            epoch_summary = (
                f"[Epoch汇总] {epoch_no}/{args.ppo_epochs} | avg_policy={avg_policy:.6f} | "
                f"avg_value={avg_value:.6f} | avg_entropy={avg_entropy:.6f} | avg_total={avg_total:.6f} | "
                f"avg_clip={avg_clip:.3f} | avg_kl={avg_kl:.6f} | best={'Y' if is_best else 'N'}"
            )
            print(epoch_summary)
            log_lines.append(f"[Round {round_no}] {epoch_summary}")

            if round_stop_reason:
                print(f"[Round早停] round={round_no} | reason={round_stop_reason}")
                log_lines.append(f"[Round {round_no}] 触发早停：{round_stop_reason}")
                break

        _set_llm_train_mode(False)

        _save_projection_checkpoint(
            output_dir=round_dir,
            rank=rank,
            ts_adapter_module=ts_adapter.learnable_image_module,
            critic_module=critic,
            optimizer=optimizer,
        )

        best_adapter_path = os.path.join(best_ckpt_dir, f"ts_image_adapter_rank{rank}.pt")
        if os.path.isfile(best_adapter_path):
            shutil.copyfile(
                best_adapter_path,
                os.path.join(round_dir, f"best_ts_image_adapter_rank{rank}.pt"),
            )

        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_waiting = -avg_reward
        summary_line = (
            f"第 {round_no} 轮完成：平均奖励(负等待)={avg_reward:.4f}，估计平均等待时间={avg_waiting:.4f}。"
            f"模型已保存到 {round_dir}。best_epoch={best_epoch if best_epoch > 0 else 'NA'}。"
        )
        print(summary_line)
        log_lines.append(summary_line)
    run_eval_all_ranks = dist.is_initialized() and world_size > 1
    test_result = None
    if run_eval_all_ranks or _is_rank0():
        adapter_module = ts_adapter.learnable_image_module
        prev_training = adapter_module.training
        adapter_module.eval()
        with torch.no_grad():
            test_result = evaluate_policy(
                dataset=dataset,
                indices=test_indices,
                ts_adapter=ts_adapter,
                policy=policy,
                device=device,
                args=args,
                env_cache=env_cache,
                generation_kwargs=generation_kwargs,
                log_interval=args.log_interval,
            )
        if prev_training:
            adapter_module.train()
    if dist.is_initialized():
        if run_eval_all_ranks:
            test_result = _broadcast_object(test_result if _is_rank0() else None)
        else:
            test_result = _broadcast_object(test_result)
    if _is_rank0() and test_result is not None:
        summary = (
            f"[测试汇总] 样本数={test_result['num_samples']} | "
            f"跳过样本数={test_result.get('skipped_samples', 0)} | "
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
