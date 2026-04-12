from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from contextlib import nullcontext
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.elastic.multiprocessing.errors import record

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.reward_shaping import (
    WAIT_REDUCER_CHOICES,
    WAIT_REDUCER_MIN_POSITIVE,
    resolve_objective_weights,
)
from projection_train.rollout_buffer import TrajectoryRolloutBuffer
from projection_train.snapshot_seed_dataset import SnapshotSeedDataset
from projection_train.trajectory_env import TrajectoryCityFlowEnv

_CURRENT_LOGGER: logging.Logger | None = None


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except TypeError:
                reconfigure(line_buffering=True)


def setup_logger(output_dir: str, rank: int) -> logging.Logger:
    run_tag = os.path.basename(os.path.normpath(str(output_dir or ""))) or "stage3_ppo"
    logger = logging.getLogger(f"projection_train.ppo.{run_tag}.rank{rank}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if rank == 0:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    os.makedirs(output_dir, exist_ok=True)
    log_name = "train_progress.log" if rank == 0 else f"train_progress.rank{rank}.log"
    file_handler = logging.FileHandler(os.path.join(output_dir, log_name), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_current_logger(logger: logging.Logger) -> None:
    global _CURRENT_LOGGER
    _CURRENT_LOGGER = logger


def get_logger() -> logging.Logger:
    if _CURRENT_LOGGER is not None:
        return _CURRENT_LOGGER
    return logging.getLogger("projection_train.ppo")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return True, rank, world_size, device
    device = resolve_device("auto")
    return False, 0, 1, device


def is_main_process() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            dist.barrier(device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier()


def resolve_rank_work_dir(base_dir: str, rank: int) -> str:
    text = str(base_dir or "").strip()
    if not text:
        return text
    return os.path.join(text, f"rank{rank}")


def sync_gradients(parameters: Sequence[torch.nn.Parameter]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    world_size = dist.get_world_size()
    if world_size <= 1:
        return
    for param in parameters:
        if param.grad is None:
            continue
        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        param.grad.div_(world_size)


def sync_metrics(
    metrics: Dict[str, float],
    device: torch.device,
    sum_keys: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    if not metrics or not (dist.is_available() and dist.is_initialized()):
        return dict(metrics)
    keys = list(metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    sum_key_set = set(sum_keys or [])
    world_size = dist.get_world_size()
    resolved: Dict[str, float] = {}
    for idx, key in enumerate(keys):
        item = float(values[idx].item())
        if key not in sum_key_set:
            item /= max(1, world_size)
        resolved[key] = item
    return resolved


def resolve_shared_transition_count(local_count: int, device: torch.device) -> int:
    resolved_local = max(0, int(local_count))
    if not (dist.is_available() and dist.is_initialized()):
        return resolved_local
    min_count = torch.tensor(resolved_local, device=device, dtype=torch.long)
    max_count = torch.tensor(resolved_local, device=device, dtype=torch.long)
    dist.all_reduce(min_count, op=dist.ReduceOp.MIN)
    dist.all_reduce(max_count, op=dist.ReduceOp.MAX)
    shared_count = int(min_count.item())
    if int(max_count.item()) != shared_count and is_main_process():
        get_logger().warning(
            f"[warn] rank-local PPO transition counts diverged across workers; "
            f"trimming each rank to the shared minimum {shared_count}."
        )
    return shared_count


def _dataset_spec(dataset: str) -> tuple[int, str, str]:
    dataset = str(dataset).strip().lower()
    if dataset == "jinan":
        return 3600, "3_4", "Jinan"
    if dataset == "hangzhou":
        return 3600, "4_4", "Hangzhou"
    if dataset == "newyork_16x3":
        return 3600, "16_3", "NewYork"
    if dataset == "newyork_28x7":
        return 3600, "28_7", "NewYork"
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def _normalize_model_path(model_path: Any) -> str:
    text = str(model_path or "").strip()
    if not text:
        return ""
    trimmed = text.rstrip("/\\")
    return trimmed or text


def _load_checkpoint_args(ckpt_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_positive_int(value: Any, default: int) -> int:
    resolved = _safe_int(value, default)
    if resolved <= 0:
        return _safe_int(default, 0)
    return resolved


def _first_positive_int(*values: Any, fallback: int) -> int:
    for value in values:
        resolved = _safe_int(value, 0)
        if resolved > 0:
            return resolved
    return _safe_positive_int(fallback, 1)


def _str_to_bool(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", ""}:
        return False
    raise ValueError(f"Unsupported boolean flag: {value!r}")


def _default_cuda_device() -> torch.device:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return torch.device(f"cuda:{int(local_rank)}")
        except ValueError:
            pass
    try:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    except Exception:
        return torch.device("cuda:0")


def resolve_device(device_text: str) -> torch.device:
    text = str(device_text or "").strip().lower()
    if not text or text == "auto":
        return _default_cuda_device() if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(text)
    if device.type == "cuda" and device.index is None:
        return _default_cuda_device()
    return device


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return None


def resolve_runtime(args: argparse.Namespace) -> Dict[str, Any]:
    ckpt_args = _load_checkpoint_args(args.stage2_ckpt)
    llm_path = _normalize_model_path(
        args.llm_path or ckpt_args.get("llm_path") or ckpt_args.get("model_path")
    )
    if not llm_path:
        raise ValueError("Failed to resolve llm_path. Pass --llm_path or use a checkpoint with saved args.")

    runtime = {
        "llm_path": llm_path,
        "seq_len": _first_positive_int(args.seq_len, ckpt_args.get("seq_len"), fallback=16),
        "image_size": _first_positive_int(args.image_size, ckpt_args.get("image_size"), fallback=224),
        "periodicity": _first_positive_int(args.periodicity, ckpt_args.get("periodicity"), fallback=16),
        "input_dim": _first_positive_int(args.input_dim, ckpt_args.get("input_dim"), fallback=8),
        "hidden_dim": _first_positive_int(args.hidden_dim, ckpt_args.get("hidden_dim"), fallback=64),
        "prefix_tokens": _first_positive_int(args.prefix_tokens, ckpt_args.get("prefix_tokens"), fallback=16),
        "prefix_mode": str(args.prefix_mode or ckpt_args.get("prefix_mode") or "prefix"),
        "max_length": _first_positive_int(args.max_length, ckpt_args.get("max_length"), fallback=128),
    }
    return runtime


def resolve_snapshot_jsonl_paths(args: argparse.Namespace) -> List[str]:
    if args.snapshot_jsonl:
        return [str(item) for item in args.snapshot_jsonl if str(item).strip()]
    return []


def build_env_paths(args: argparse.Namespace) -> tuple[str, str, int]:
    default_count, road_net, template = _dataset_spec(args.dataset)
    traffic_root = args.traffic_root or os.path.join("data", template, road_net)
    roadnet_path = args.roadnet_path or os.path.join(traffic_root, f"roadnet_{road_net}.json")
    run_counts = int(args.run_counts or default_count)
    if "24h" in str(args.traffic_file or ""):
        run_counts = 86400
    return roadnet_path, traffic_root, run_counts


def build_policy(
    runtime: Dict[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
    with_value_head: bool,
    llm_fsdp_enabled: bool = False,
    fsdp_sharding_strategy: str = "full_shard",
    backbone: Any = None,
) -> PrefixActionPolicy:
    return PrefixActionPolicy(
        model_path=runtime["llm_path"],
        seq_len=runtime["seq_len"],
        image_size=runtime["image_size"],
        periodicity=runtime["periodicity"],
        input_dim=runtime["input_dim"],
        hidden_dim=runtime["hidden_dim"],
        prefix_tokens=runtime["prefix_tokens"],
        prefix_mode=runtime["prefix_mode"],
        max_length=runtime["max_length"],
        torch_dtype=torch_dtype,
        device=device,
        with_value_head=with_value_head,
        value_hidden_dim=runtime.get("value_hidden_dim", 512),
        backbone=backbone,
        backbone_fsdp_enabled=llm_fsdp_enabled,
        backbone_fsdp_sharding_strategy=fsdp_sharding_strategy,
    )


def set_policy_mode(policy: PrefixActionPolicy, is_train: bool) -> None:
    policy.backbone.model.eval()
    policy.ts_projector.train(mode=is_train)
    policy.vision_projector.train(mode=is_train)
    if policy.value_head is not None:
        policy.value_head.train(mode=is_train)


def rollout_once(
    policy: PrefixActionPolicy,
    env: TrajectoryCityFlowEnv,
    device: torch.device,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    sample_actions: bool,
    reference_policy: PrefixActionPolicy | None = None,
    episode_reset_fn: Callable[[], Dict[str, Any]] | None = None,
) -> tuple[TrajectoryRolloutBuffer, Dict[str, float]]:
    buffer = TrajectoryRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
    set_policy_mode(policy, is_train=False)
    if reference_policy is not None:
        set_policy_mode(reference_policy, is_train=False)
    obs = env.observe()
    episode_reward = 0.0
    queue_trace: List[float] = []
    wait_trace: List[float] = []
    travel_time_trace: List[float] = []
    active_trace: List[float] = []
    departed_trace: List[float] = []
    queue_reward_trace: List[float] = []
    wait_reward_trace: List[float] = []
    wait_signal_trace: List[float] = []
    travel_reward_trace: List[float] = []
    starvation_penalty_trace: List[float] = []
    local_throughput_trace: List[float] = []

    for step_idx in range(int(rollout_steps)):
        seq_tensor = torch.tensor(obs["sequences"], dtype=torch.float32, device=device)
        prompts = obs["prompts"]
        with torch.no_grad():
            act_out = policy.act(seq_tensor, prompts, sample=sample_actions)
            reference_logits = None
            if reference_policy is not None:
                reference_logits = reference_policy(seq_tensor, prompts).detach().cpu().float()
        actions = act_out["actions"]
        values = act_out["values"]
        if values is None:
            raise RuntimeError("Trajectory PPO requires PrefixActionPolicy(with_value_head=True).")
        next_obs, rewards, done, info = env.step(actions.tolist())
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        done_tensor = torch.full((len(rewards),), float(done), dtype=torch.float32)
        buffer.add(
            sequences=seq_tensor,
            prompts=prompts,
            actions=actions,
            log_probs=act_out["log_probs"],
            values=values,
            rewards=reward_tensor,
            dones=done_tensor,
            reference_logits=reference_logits,
        )
        episode_reward += float(sum(rewards))
        queue_trace.append(float(info.get("queue_sum", 0.0)))
        wait_trace.append(float(info.get("waiting_time_mean", 0.0)))
        travel_time_trace.append(float(info.get("travel_time_mean", 0.0)))
        active_trace.append(float(info.get("active_vehicle_count", 0.0)))
        departed_trace.append(float(info.get("departed_vehicle_count", 0.0)))
        queue_reward_trace.append(float(info.get("queue_reward_mean", 0.0)))
        wait_reward_trace.append(float(info.get("wait_reward_mean", 0.0)))
        wait_signal_trace.append(float(info.get("wait_signal_mean", 0.0)))
        travel_reward_trace.append(float(info.get("travel_reward_mean", 0.0)))
        starvation_penalty_trace.append(float(info.get("starvation_penalty_mean", 0.0)))
        local_throughput_trace.append(float(info.get("local_throughput_mean", 0.0)))
        obs = next_obs
        if done and step_idx + 1 < int(rollout_steps):
            if episode_reset_fn is not None:
                obs = episode_reset_fn()
            else:
                obs = env.reset(env.current_traffic_file)

    if env.done:
        last_values = torch.zeros((len(env.intersection_ids),), dtype=torch.float32)
    else:
        seq_tensor = torch.tensor(obs["sequences"], dtype=torch.float32, device=device)
        with torch.no_grad():
            last_values = policy.predict_values(seq_tensor, obs["prompts"]).detach().cpu().float()

    buffer.compute_returns_and_advantages(last_values)
    metrics = {
        "rollout_reward": episode_reward,
        "rollout_avg_queue": float(np.mean(queue_trace) if queue_trace else 0.0),
        "rollout_avg_waiting": float(np.mean(wait_trace) if wait_trace else 0.0),
        "rollout_avg_travel_time": float(np.mean(travel_time_trace) if travel_time_trace else 0.0),
        "rollout_avg_active_vehicles": float(np.mean(active_trace) if active_trace else 0.0),
        "rollout_avg_departed_vehicles": float(np.mean(departed_trace) if departed_trace else 0.0),
        "rollout_avg_queue_reward": float(np.mean(queue_reward_trace) if queue_reward_trace else 0.0),
        "rollout_avg_wait_reward": float(np.mean(wait_reward_trace) if wait_reward_trace else 0.0),
        "rollout_avg_wait_signal": float(np.mean(wait_signal_trace) if wait_signal_trace else 0.0),
        "rollout_avg_travel_reward": float(np.mean(travel_reward_trace) if travel_reward_trace else 0.0),
        "rollout_avg_starvation_penalty": float(
            np.mean(starvation_penalty_trace) if starvation_penalty_trace else 0.0
        ),
        "rollout_avg_local_throughput": float(
            np.mean(local_throughput_trace) if local_throughput_trace else 0.0
        ),
        "rollout_steps": float(buffer.num_steps),
        "rollout_transitions": float(buffer.num_transitions),
    }
    return buffer, metrics


def ppo_update(
    policy: PrefixActionPolicy,
    optimizer: torch.optim.Optimizer,
    buffer: TrajectoryRolloutBuffer,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Dict[str, float]:
    set_policy_mode(policy, is_train=True)

    stats = {
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "kl": [],
        "ratio": [],
        "clip_frac": [],
        "total_loss": [],
    }
    def _autocast_context():
        if amp_dtype is not None and device.type == "cuda":
            return torch.amp.autocast("cuda", dtype=amp_dtype)
        return nullcontext()

    shared_transition_count = resolve_shared_transition_count(buffer.num_transitions, device=device)
    if shared_transition_count <= 0:
        raise RuntimeError("Resolved zero shared transitions for PPO update.")

    for _ in range(int(args.ppo_epochs)):
        for batch in buffer.mini_batches(
            args.mini_batch_size,
            shuffle=True,
            max_items=shared_transition_count,
        ):
            sequences = batch["sequences"].to(device)
            prompts = batch["prompts"]
            actions = batch["actions"].to(device)
            old_log_probs = batch["old_log_probs"].to(device)
            old_values = batch["old_values"].to(device)
            advantages = batch["advantages"].to(device)
            returns = batch["returns"].to(device)
            if "reference_logits" not in batch:
                raise RuntimeError("PPO update requires cached reference_logits from rollout.")
            reference_logits = batch["reference_logits"].to(device).float()

            optimizer.zero_grad(set_to_none=True)
            with _autocast_context():
                evaluated = policy.evaluate_actions(sequences, prompts, actions)
                new_log_probs = evaluated["log_probs"]
                entropy = evaluated["entropy"].mean()
                values = evaluated["values"]
                if values is None:
                    raise RuntimeError("Missing value predictions during PPO update.")
                logits = evaluated["logits"].float()
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred_clipped = old_values + torch.clamp(
                    values - old_values,
                    -args.clip_range,
                    args.clip_range,
                )
                value_loss_1 = (values - returns).pow(2)
                value_loss_2 = (value_pred_clipped - returns).pow(2)
                value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()

                current_log_probs = F.log_softmax(logits, dim=-1)
                reference_log_probs = F.log_softmax(reference_logits, dim=-1)
                current_probs = current_log_probs.exp()
                kl = torch.sum(current_probs * (current_log_probs - reference_log_probs), dim=-1).mean()

                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    - args.entropy_coef * entropy
                    + args.kl_coef * kl
                )

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                sync_gradients(policy.trainable_parameters())
                torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                sync_gradients(policy.trainable_parameters())
                torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), args.max_grad_norm)
                optimizer.step()

            clip_frac = ((ratio - 1.0).abs() > args.clip_range).float().mean()
            stats["policy_loss"].append(float(policy_loss.detach().cpu().item()))
            stats["value_loss"].append(float(value_loss.detach().cpu().item()))
            stats["entropy"].append(float(entropy.detach().cpu().item()))
            stats["kl"].append(float(kl.detach().cpu().item()))
            stats["ratio"].append(float(ratio.detach().mean().cpu().item()))
            stats["clip_frac"].append(float(clip_frac.detach().cpu().item()))
            stats["total_loss"].append(float(loss.detach().cpu().item()))

    return {
        key: float(np.mean(values) if values else 0.0)
        for key, values in stats.items()
    }


@torch.no_grad()
def evaluate_full_episode(
    policy: PrefixActionPolicy,
    env: TrajectoryCityFlowEnv,
    device: torch.device,
    traffic_file: str,
) -> Dict[str, float]:
    set_policy_mode(policy, is_train=False)
    obs = env.reset(traffic_file)
    done = False
    total_reward = 0.0
    queue_trace: List[float] = []
    wait_trace: List[float] = []
    step_count = 0

    while not done:
        sequences = torch.tensor(obs["sequences"], dtype=torch.float32, device=device)
        prompts = obs["prompts"]
        with torch.no_grad():
            act_out = policy.act(sequences, prompts, sample=False)
        obs, rewards, done, info = env.step(act_out["actions"].tolist())
        total_reward += float(sum(rewards))
        queue_trace.append(float(info.get("queue_sum", 0.0)))
        wait_trace.append(float(info.get("waiting_time_mean", 0.0)))
        step_count += 1
        if env.current_time_s >= env.run_counts:
            done = True

    return {
        "eval_reward": total_reward,
        "eval_avg_queue": float(np.mean(queue_trace) if queue_trace else 0.0),
        "eval_avg_waiting": float(np.mean(wait_trace) if wait_trace else 0.0),
        "eval_avg_travel_time": float(env.episode_travel_time_mean(resolved_end_time=env.run_counts)),
        "eval_steps": float(step_count),
    }


def _safe_metric_ratio(current: float, baseline: float) -> float:
    current_value = float(current)
    baseline_value = float(baseline)
    if abs(baseline_value) < 1e-6:
        return 1.0 if abs(current_value) < 1e-6 else 1.0 + abs(current_value)
    return current_value / baseline_value


def add_selection_metrics(
    eval_metrics: Dict[str, float],
    baseline_eval_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> Dict[str, float]:
    queue_ratio = _safe_metric_ratio(
        eval_metrics.get("eval_avg_queue", 0.0),
        baseline_eval_metrics.get("eval_avg_queue", 0.0),
    )
    wait_ratio = _safe_metric_ratio(
        eval_metrics.get("eval_avg_waiting", 0.0),
        baseline_eval_metrics.get("eval_avg_waiting", 0.0),
    )
    travel_ratio = _safe_metric_ratio(
        eval_metrics.get("eval_avg_travel_time", 0.0),
        baseline_eval_metrics.get("eval_avg_travel_time", 0.0),
    )
    weight_sum = max(
        1e-6,
        float(args.selection_queue_weight)
        + float(args.selection_wait_weight)
        + float(args.selection_travel_weight),
    )
    composite_score = -(
        float(args.selection_queue_weight) * queue_ratio
        + float(args.selection_wait_weight) * wait_ratio
        + float(args.selection_travel_weight) * travel_ratio
    ) / weight_sum
    enriched = dict(eval_metrics)
    enriched.update(
        {
            "eval_queue_ratio": float(queue_ratio),
            "eval_wait_ratio": float(wait_ratio),
            "eval_travel_ratio": float(travel_ratio),
            "eval_composite_score": float(composite_score),
            "eval_all_metrics_improved": float(
                queue_ratio < 1.0 and wait_ratio < 1.0 and travel_ratio < 1.0
            ),
        }
    )
    return enriched


def resolve_selection_score(eval_metrics: Dict[str, float], args: argparse.Namespace) -> float:
    if args.model_selection_metric == "reward":
        return float(eval_metrics.get("eval_reward", float("-inf")))
    return float(eval_metrics.get("eval_composite_score", float("-inf")))


def save_checkpoint(
    output_dir: str,
    filename: str,
    policy: PrefixActionPolicy,
    optimizer: torch.optim.Optimizer,
    update_idx: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "ts_projector": policy.ts_projector.state_dict(),
        "vision_projector": policy.vision_projector.state_dict(),
        "value_head": policy.value_head.state_dict() if policy.value_head is not None else None,
        "optimizer": optimizer.state_dict(),
        "update": int(update_idx),
        "metrics": metrics,
        "args": vars(args),
    }
    output_path = os.path.join(output_dir, filename)
    torch.save(payload, output_path)
    return output_path


def log_configuration_warnings(
    logger: logging.Logger,
    args: argparse.Namespace,
    snapshot_jsonl_paths: Sequence[str],
) -> None:
    if not snapshot_jsonl_paths and float(args.snapshot_reset_prob) > 0.0:
        logger.warning(
            "snapshot_reset_prob=%.3f but no snapshot_jsonl was provided; training will only reset when an episode ends.",
            float(args.snapshot_reset_prob),
        )
    if int(args.rollout_steps) < 4:
        logger.warning(
            "rollout_steps=%d is unusually small for PPO and often causes unstable updates.",
            int(args.rollout_steps),
        )
    if int(args.mini_batch_size) <= 1:
        logger.warning(
            "mini_batch_size=%d gives the optimizer almost no averaging and can destabilize PPO.",
            int(args.mini_batch_size),
        )
    if str(args.wait_reward_reducer or "").strip().lower() == "min":
        logger.warning(
            "wait_reward_reducer=min can make the wait signal collapse toward zero when any nearby approach has a fresh vehicle."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="records/projection_ppo_traj")
    parser.add_argument("--snapshot_jsonl", type=str, nargs="+", default=None)
    parser.add_argument("--snapshot_traffic_file", type=str, default="")
    parser.add_argument("--snapshot_max_samples", type=int, default=None)
    parser.add_argument("--snapshot_reset_prob", type=float, default=0.8)
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--eval_traffic_file", type=str, default="")
    parser.add_argument("--roadnet_path", type=str, default="")
    parser.add_argument("--traffic_root", type=str, default="")
    parser.add_argument("--work_root", type=str, default="records/projection_ppo_traj/env_work")
    parser.add_argument("--llm_path", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--periodicity", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--prefix_tokens", type=int, default=0)
    parser.add_argument("--prefix_mode", type=str, default="")
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--value_hidden_dim", type=int, default=512)
    parser.add_argument("--prompt_mode", type=str, default="image_only_short")
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--traffic_series_interval", type=int, default=2)
    parser.add_argument("--window_s", type=int, default=30)
    parser.add_argument("--num_updates", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--kl_coef", type=float, default=0.02)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--reward_alpha", type=float, default=0.1)
    parser.add_argument("--switch_penalty_beta", type=float, default=0.02)
    parser.add_argument(
        "--queue_reward_coef",
        type=float,
        default=-0.25,
        help="Coefficient on queue change (current - previous). Negative values reward queue reductions.",
    )
    parser.add_argument("--pressure_reward_coef", type=float, default=0.0)
    parser.add_argument(
        "--waiting_reward_coef",
        type=float,
        default=-0.02,
        help="Coefficient on the resolved wait statistic. Negative values reward lower wait signals.",
    )
    parser.add_argument(
        "--system_vehicle_reward_coef",
        type=float,
        default=-0.01,
        help="Coefficient on local active-vehicle-count change (current - previous). Negative values reward clearing vehicles.",
    )
    parser.add_argument(
        "--throughput_reward_coef",
        type=float,
        default=0.02,
        help="Coefficient on local throughput count. Positive values reward vehicles clearing the controlled approaches.",
    )
    parser.add_argument("--starvation_penalty_coef", type=float, default=0.05)
    parser.add_argument("--starvation_threshold_steps", type=int, default=2)
    parser.add_argument("--starvation_queue_threshold", type=float, default=8.0)
    parser.add_argument("--starvation_wait_threshold", type=float, default=45.0)
    parser.add_argument(
        "--wait_reward_scope",
        type=str,
        default="neighborhood",
        choices=["local", "neighborhood"],
        help="Whether the wait reward uses only the current intersection or the current intersection plus neighbors.",
    )
    parser.add_argument(
        "--wait_reward_reducer",
        type=str,
        default=WAIT_REDUCER_MIN_POSITIVE,
        choices=list(WAIT_REDUCER_CHOICES),
        help="Reducer applied to the collected wait values before the wait reward coefficient is applied.",
    )
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument(
        "--model_selection_metric",
        type=str,
        default="composite",
        choices=["composite", "reward"],
    )
    parser.add_argument(
        "--selection_queue_weight",
        type=float,
        default=1.0,
        help="Relative queue objective weight used in PPO reward shaping and composite checkpoint selection.",
    )
    parser.add_argument(
        "--selection_wait_weight",
        type=float,
        default=1.0,
        help="Relative waiting-time objective weight used in PPO reward shaping and composite checkpoint selection.",
    )
    parser.add_argument(
        "--selection_travel_weight",
        type=float,
        default=1.0,
        help="Relative travel proxy weight used in PPO reward shaping and composite checkpoint selection.",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--llm_fsdp", type=str, default="off", choices=["off", "on"])
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
    )
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    return parser.parse_args()


@record
def main() -> None:
    configure_stdio()
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")
    if not (0.0 <= args.snapshot_reset_prob <= 1.0):
        raise ValueError("--snapshot_reset_prob must be within [0, 1].")
    objective_weights = resolve_objective_weights(
        queue_weight=args.selection_queue_weight,
        wait_weight=args.selection_wait_weight,
        travel_weight=args.selection_travel_weight,
    )

    roadnet_path, traffic_root, run_counts = build_env_paths(args)
    runtime = resolve_runtime(args)
    runtime["value_hidden_dim"] = int(args.value_hidden_dim)

    ddp, rank, world_size, device = init_distributed()
    logger = setup_logger(args.output_dir, rank)
    set_current_logger(logger)
    if str(args.device or "").strip().lower() not in {"", "auto"}:
        requested_device = resolve_device(args.device)
        if requested_device.type != device.type or requested_device.index != device.index:
            raise ValueError(
                f"--device={args.device} conflicts with distributed device assignment {device}. "
                "Use --device auto when launching with torchrun."
            )
    llm_fsdp_enabled = _str_to_bool(args.llm_fsdp)
    if llm_fsdp_enabled and not ddp:
        raise ValueError("--llm_fsdp on requires distributed launch via torchrun.")
    if llm_fsdp_enabled and device.type != "cuda":
        raise ValueError("--llm_fsdp on requires CUDA devices.")
    amp_dtype = resolve_amp_dtype(args, device)
    model_dtype = amp_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    seed_everything(args.seed + rank)
    logger.info(
        "Starting Stage 3 PPO training: rank=%d world_size=%d device=%s output_dir=%s",
        rank,
        world_size,
        device,
        args.output_dir,
    )
    logger.info("Training arguments: %s", vars(args))
    logger.info("Resolved runtime: %s", runtime)
    logger.info("Resolved PPO objective weights: %s", objective_weights)

    reward_info = {
        "queue_length": float(args.queue_reward_coef),
        "pressure": float(args.pressure_reward_coef),
    }

    policy = build_policy(
        runtime,
        device,
        model_dtype,
        with_value_head=True,
        llm_fsdp_enabled=llm_fsdp_enabled,
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
    )
    policy.load_checkpoint(args.stage2_ckpt, load_value_head=False)
    reference_policy = build_policy(
        runtime,
        device,
        model_dtype,
        with_value_head=False,
        backbone=policy.backbone,
    )
    reference_policy.load_checkpoint(args.stage2_ckpt, load_value_head=False)
    reference_policy.eval()
    for param in reference_policy.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(
        policy.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))

    train_env = TrajectoryCityFlowEnv(
        roadnet_path=roadnet_path,
        traffic_root=traffic_root,
        work_root=os.path.join(resolve_rank_work_dir(args.work_root, rank), "train"),
        default_traffic_file=args.traffic_file,
        seq_len=runtime["seq_len"],
        movement_keys=["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"],
        prompt_mode=args.prompt_mode,
        traffic_series_interval=args.traffic_series_interval,
        window_s=args.window_s,
        min_action_time=args.min_action_time,
        reward_alpha=args.reward_alpha,
        switch_penalty_beta=args.switch_penalty_beta,
        queue_reward_coef=args.queue_reward_coef,
        waiting_reward_coef=args.waiting_reward_coef,
        system_vehicle_reward_coef=args.system_vehicle_reward_coef,
        throughput_reward_coef=args.throughput_reward_coef,
        starvation_penalty_coef=args.starvation_penalty_coef,
        starvation_threshold_steps=args.starvation_threshold_steps,
        starvation_queue_threshold=args.starvation_queue_threshold,
        starvation_wait_threshold=args.starvation_wait_threshold,
        wait_reward_scope=args.wait_reward_scope,
        wait_reward_reducer=args.wait_reward_reducer,
        selection_queue_weight=args.selection_queue_weight,
        selection_wait_weight=args.selection_wait_weight,
        selection_travel_weight=args.selection_travel_weight,
        reward_info=reward_info,
        run_counts=run_counts,
    )
    eval_env = TrajectoryCityFlowEnv(
        roadnet_path=roadnet_path,
        traffic_root=traffic_root,
        work_root=os.path.join(resolve_rank_work_dir(args.work_root, rank), "eval"),
        default_traffic_file=args.eval_traffic_file or args.traffic_file,
        seq_len=runtime["seq_len"],
        movement_keys=["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"],
        prompt_mode=args.prompt_mode,
        traffic_series_interval=args.traffic_series_interval,
        window_s=args.window_s,
        min_action_time=args.min_action_time,
        reward_alpha=args.reward_alpha,
        switch_penalty_beta=args.switch_penalty_beta,
        queue_reward_coef=args.queue_reward_coef,
        waiting_reward_coef=args.waiting_reward_coef,
        system_vehicle_reward_coef=args.system_vehicle_reward_coef,
        throughput_reward_coef=args.throughput_reward_coef,
        starvation_penalty_coef=args.starvation_penalty_coef,
        starvation_threshold_steps=args.starvation_threshold_steps,
        starvation_queue_threshold=args.starvation_queue_threshold,
        starvation_wait_threshold=args.starvation_wait_threshold,
        wait_reward_scope=args.wait_reward_scope,
        wait_reward_reducer=args.wait_reward_reducer,
        selection_queue_weight=args.selection_queue_weight,
        selection_wait_weight=args.selection_wait_weight,
        selection_travel_weight=args.selection_travel_weight,
        reward_info=reward_info,
        run_counts=run_counts,
    )

    snapshot_jsonl_paths = resolve_snapshot_jsonl_paths(args)
    log_configuration_warnings(logger, args, snapshot_jsonl_paths)
    snapshot_dataset = None
    if snapshot_jsonl_paths:
        snapshot_dataset = SnapshotSeedDataset(
            path=snapshot_jsonl_paths,
            seq_len=runtime["seq_len"],
            movement_keys=["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"],
            max_snapshots=args.snapshot_max_samples,
            min_intersections=1,
        )
        if len(snapshot_dataset) == 0:
            snapshot_dataset = None
            if is_main_process():
                logger.warning(
                    "Ignoring snapshot reset because no valid snapshot seeds were loaded from %s.",
                    snapshot_jsonl_paths,
                )
        elif is_main_process():
            logger.info("Loaded snapshot seeds: %d from %s", len(snapshot_dataset), snapshot_jsonl_paths)

    if is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
    barrier()
    best_metric = float("-inf")
    eval_traffic_file = args.eval_traffic_file or args.traffic_file
    rng = random.Random(args.seed + rank)

    if is_main_process():
        logger.info(
            "[init] device=%s world_size=%d ddp=%s llm_fsdp_requested=%s "
            "llm_fsdp_active=%s shared_reference_backbone=True",
            device,
            world_size,
            ddp,
            llm_fsdp_enabled,
            getattr(policy.backbone, "is_fsdp", False),
        )

    baseline_eval_metrics = evaluate_full_episode(
        policy=policy,
        env=eval_env,
        device=device,
        traffic_file=eval_traffic_file,
    )
    baseline_eval_metrics = sync_metrics(baseline_eval_metrics, device=device)
    baseline_eval_metrics = add_selection_metrics(
        baseline_eval_metrics,
        baseline_eval_metrics,
        args,
    )
    if is_main_process():
        baseline_score = resolve_selection_score(baseline_eval_metrics, args)
        logger.info(
            "[eval 0] reward=%.4f avg_queue=%.4f avg_wait=%.4f avg_travel=%.4f score=%.4f",
            baseline_eval_metrics["eval_reward"],
            baseline_eval_metrics["eval_avg_queue"],
            baseline_eval_metrics["eval_avg_waiting"],
            baseline_eval_metrics["eval_avg_travel_time"],
            baseline_score,
        )
        save_checkpoint(
            output_dir=args.output_dir,
            filename="baseline.pt",
            policy=policy,
            optimizer=optimizer,
            update_idx=0,
            args=args,
            metrics=baseline_eval_metrics,
        )
        save_checkpoint(
            output_dir=args.output_dir,
            filename="best.pt",
            policy=policy,
            optimizer=optimizer,
            update_idx=0,
            args=args,
            metrics=baseline_eval_metrics,
        )
        save_checkpoint(
            output_dir=args.output_dir,
            filename="last.pt",
            policy=policy,
            optimizer=optimizer,
            update_idx=0,
            args=args,
            metrics=baseline_eval_metrics,
        )
        best_metric = float(baseline_score)

    def _reset_train_env() -> Dict[str, Any]:
        use_snapshot = snapshot_dataset is not None and rng.random() < args.snapshot_reset_prob
        if use_snapshot:
            seed_sample = snapshot_dataset.sample(rng=rng, traffic_file=args.snapshot_traffic_file)
            return train_env.reset_from_snapshot(seed_sample)
        return train_env.reset(args.traffic_file)

    for update_idx in range(1, args.num_updates + 1):
        # Keep rolling forward inside the same episode and only reset when an
        # episode finishes. Otherwise, full-reset PPO keeps oversampling t=0.
        if train_env.env is None or train_env.done:
            _reset_train_env()

        rollout_buffer, rollout_metrics = rollout_once(
            policy=policy,
            env=train_env,
            device=device,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            sample_actions=True,
            reference_policy=reference_policy,
            episode_reset_fn=_reset_train_env,
        )
        update_metrics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            buffer=rollout_buffer,
            device=device,
            args=args,
            amp_dtype=amp_dtype,
            scaler=scaler if amp_dtype == torch.float16 else None,
        )

        merged_metrics = sync_metrics(
            {**rollout_metrics, **update_metrics},
            device=device,
            sum_keys=("rollout_steps", "rollout_transitions"),
        )
        if is_main_process():
            logger.info(
                "[update %d/%d] reward=%.4f queue=%.4f wait=%.4f wait_sig=%.4f travel=%.4f active=%.4f "
                "rq=%.4f rw=%.4f rt=%.4f starve=%.4f policy=%.4f value=%.4f entropy=%.4f kl=%.4f "
                "transitions=%.0f",
                update_idx,
                args.num_updates,
                merged_metrics["rollout_reward"],
                merged_metrics["rollout_avg_queue"],
                merged_metrics["rollout_avg_waiting"],
                merged_metrics["rollout_avg_wait_signal"],
                merged_metrics["rollout_avg_travel_time"],
                merged_metrics["rollout_avg_active_vehicles"],
                merged_metrics["rollout_avg_queue_reward"],
                merged_metrics["rollout_avg_wait_reward"],
                merged_metrics["rollout_avg_travel_reward"],
                merged_metrics["rollout_avg_starvation_penalty"],
                merged_metrics["policy_loss"],
                merged_metrics["value_loss"],
                merged_metrics["entropy"],
                merged_metrics["kl"],
                merged_metrics["rollout_transitions"],
            )

        if is_main_process():
            save_checkpoint(
                output_dir=args.output_dir,
                filename=f"epoch_{update_idx:04d}.pt",
                policy=policy,
                optimizer=optimizer,
                update_idx=update_idx,
                args=args,
                metrics=merged_metrics,
            )
            save_checkpoint(
                output_dir=args.output_dir,
                filename="last.pt",
                policy=policy,
                optimizer=optimizer,
                update_idx=update_idx,
                args=args,
                metrics=merged_metrics,
            )
        barrier()

        should_eval = (args.eval_every > 0 and update_idx % args.eval_every == 0) or update_idx == args.num_updates
        if should_eval:
            eval_metrics = evaluate_full_episode(
                policy=policy,
                env=eval_env,
                device=device,
                traffic_file=eval_traffic_file,
            )
            eval_metrics = sync_metrics(eval_metrics, device=device)
            eval_metrics = add_selection_metrics(eval_metrics, baseline_eval_metrics, args)
            merged_metrics.update(eval_metrics)
            if is_main_process():
                logger.info(
                    "[eval %d] reward=%.4f avg_queue=%.4f avg_wait=%.4f avg_travel=%.4f score=%.4f",
                    update_idx,
                    eval_metrics["eval_reward"],
                    eval_metrics["eval_avg_queue"],
                    eval_metrics["eval_avg_waiting"],
                    eval_metrics["eval_avg_travel_time"],
                    resolve_selection_score(eval_metrics, args),
                )
                save_checkpoint(
                    output_dir=args.output_dir,
                    filename="last.pt",
                    policy=policy,
                    optimizer=optimizer,
                    update_idx=update_idx,
                    args=args,
                    metrics=merged_metrics,
                )
                selection_score = resolve_selection_score(eval_metrics, args)
                if selection_score > best_metric:
                    best_metric = selection_score
                    save_checkpoint(
                        output_dir=args.output_dir,
                        filename="best.pt",
                        policy=policy,
                        optimizer=optimizer,
                        update_idx=update_idx,
                        args=args,
                        metrics=merged_metrics,
                    )
        barrier()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
