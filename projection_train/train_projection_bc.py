from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from data_split import blocked_group_subset, stratified_label_subset
from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.cost_ranking import (
    DEFAULT_COST_WEIGHTS,
    base_action_kl_loss,
    build_candidate_costs,
    cost_soft_targets,
    pairwise_cost_ranking_loss,
)
from projection_train.grouped_dataset import GroupedTrafficActionDataset, grouped_action_collate
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.prompt_utils import PROMPT_MODES, build_task_prompt


BASELINE_ACTION_DISTRIBUTIONS: Dict[str, Dict[str, float]] = {
    "stage2_bc_val02": {"ETWT": 36.88, "NTST": 31.46, "ELWL": 15.00, "NLSL": 16.67},
    "baseline": {"ETWT": 36.88, "NTST": 31.46, "ELWL": 15.00, "NLSL": 16.67},
}


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except TypeError:
                reconfigure(line_buffering=True)


def setup_logger(output_dir: str, rank: int) -> logging.Logger:
    logger = logging.getLogger(f"projection_train.bc.rank{rank}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(output_dir, "train_progress.log"),
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed(timeout_seconds: int) -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                init_method="env://",
                timeout=timedelta(seconds=max(1, int(timeout_seconds))),
            )
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return True, rank, world_size, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def barrier() -> None:
    if not dist.is_available() or not dist.is_initialized():
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def shard_eval_subset(eval_set, ddp: bool, rank: int, world_size: int):
    if eval_set is None or not ddp or world_size <= 1:
        return eval_set
    return Subset(eval_set, list(range(rank, len(eval_set), world_size)))


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return None


def split_dataset(
    dataset,
    val_ratio: float,
    seed: int,
    split_strategy: str,
    validation_purge_samples: int = 0,
) -> Tuple[Subset, Subset | None]:
    if len(dataset) == 0:
        raise RuntimeError("No grouped samples found for Stage 2 training.")
    if split_strategy == "blocked":
        return blocked_group_subset(
            dataset,
            val_ratio=val_ratio,
            purge_samples=validation_purge_samples,
        )
    if split_strategy == "stratified":
        return stratified_label_subset(
            dataset,
            val_ratio=val_ratio,
            label_attr="expert_action",
            seed=seed,
        )
    raise ValueError(f"Unsupported split strategy: {split_strategy}")


def dataset_samples(dataset) -> List:
    if isinstance(dataset, Subset):
        base_samples = list(getattr(dataset.dataset, "samples", []))
        return [base_samples[int(idx)] for idx in dataset.indices]
    return list(getattr(dataset, "samples", []))


def action_distribution(dataset) -> Dict[str, int]:
    counts = {action: 0 for action in ACTION_LIST}
    extra_counts: Dict[str, int] = {}
    for sample in dataset_samples(dataset):
        action = str(getattr(sample, "expert_action", "") or "")
        if action in counts:
            counts[action] += 1
        else:
            extra_counts[action] = extra_counts.get(action, 0) + 1
    counts.update({key: value for key, value in sorted(extra_counts.items())})
    return counts


def format_action_distribution(counts: Dict[str, int]) -> str:
    return ", ".join(f"{action}={int(counts.get(action, 0))}" for action in counts)


def compute_class_weights(train_set) -> torch.Tensor:
    counts = torch.zeros(len(ACTION_LIST), dtype=torch.float32)
    for sample in dataset_samples(train_set):
        label = action_to_index(getattr(sample, "expert_action", ""))
        counts[label] += 1.0
    counts = counts.clamp_min(1.0)
    weights = counts.sum() / (counts * float(len(ACTION_LIST)))
    return weights / weights.mean().clamp_min(1e-8)


def _parse_global_action_weights(raw: str | None) -> torch.Tensor | None:
    """Parse comma-separated per-action weights e.g. '0.5,1.0,1.0,2.0'."""
    if raw is None or not str(raw).strip():
        return None
    parts = [p.strip() for p in str(raw).split(",")]
    if len(parts) != len(ACTION_LIST):
        raise ValueError(
            f"--global_action_weights expects {len(ACTION_LIST)} values, got {len(parts)}"
        )
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def _compute_action_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    """Entropy regularizer: penalize non-uniform predicted action distribution."""
    probs = F.softmax(logits, dim=-1)
    batch_probs = probs.mean(dim=0)
    eps = 1e-8
    entropy = -(batch_probs * torch.log(batch_probs + eps)).sum()
    max_entropy = torch.log(
        torch.tensor(len(ACTION_LIST), device=logits.device, dtype=torch.float32)
    )
    return max_entropy - entropy


def _compute_overconf_penalty(logits: torch.Tensor) -> torch.Tensor:
    """Penalize when any action's mean probability exceeds uniform."""
    probs = F.softmax(logits, dim=-1)
    mean_probs = probs.mean(dim=0)
    if mean_probs.numel() < 2:
        return torch.tensor(0.0, device=logits.device)
    uniform = 1.0 / len(ACTION_LIST)
    excess = torch.relu(mean_probs - uniform)
    return excess.sum()


def base_policy_kl_mask(
    base_actions: Sequence[str],
    expert_actions: Sequence[str],
    sample_weights: torch.Tensor,
    mode: str,
    low_weight_threshold: float,
    device: torch.device,
) -> torch.Tensor:
    """Select rows where the conservative base-action prior should apply."""
    if mode == "all":
        return torch.ones((len(base_actions),), dtype=torch.bool, device=device)

    match_mask = torch.tensor(
        [
            str(base_action) in ACTION_LIST and str(base_action) == str(expert_action)
            for base_action, expert_action in zip(base_actions, expert_actions)
        ],
        dtype=torch.bool,
        device=device,
    )
    if mode == "expert_match":
        return match_mask
    if mode == "low_weight":
        return sample_weights <= float(low_weight_threshold)
    if mode == "match_or_low_weight":
        return match_mask | (sample_weights <= float(low_weight_threshold))
    raise ValueError(f"Unsupported base policy KL mode: {mode}")


def build_soft_targets(
    raw_waits_batch: Sequence[Dict[str, float]],
    device: torch.device,
    temperature: float,
) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
    if not raw_waits_batch:
        return None, None
    wait_rows = []
    valid_rows = []
    for raw_waits in raw_waits_batch:
        if not isinstance(raw_waits, dict) or not raw_waits:
            wait_rows.append([0.0] * len(ACTION_LIST))
            valid_rows.append(False)
            continue
        row = []
        valid_count = 0
        for action in ACTION_LIST:
            value = raw_waits.get(action)
            if value is None:
                row.append(float("inf"))
            else:
                try:
                    row.append(float(value))
                    valid_count += 1
                except (TypeError, ValueError):
                    row.append(float("inf"))
        wait_rows.append(row)
        valid_rows.append(valid_count >= 2 and any(torch.isfinite(torch.tensor(row)).tolist()))
    valid_mask = torch.tensor(valid_rows, device=device, dtype=torch.bool)
    if not bool(valid_mask.any().item()):
        return None, None
    wait_tensor = torch.tensor(wait_rows, device=device, dtype=torch.float32)
    finite_mask = torch.isfinite(wait_tensor)
    safe_waits = wait_tensor.clone()
    safe_waits[~finite_mask] = 0.0
    min_waits = torch.where(
        finite_mask,
        wait_tensor,
        torch.full_like(wait_tensor, float("inf")),
    ).amin(dim=-1, keepdim=True)
    min_waits = torch.where(torch.isfinite(min_waits), min_waits, torch.zeros_like(min_waits))
    scaled = -(safe_waits - min_waits) / max(float(temperature), 1e-6)
    scaled = scaled.masked_fill(~finite_mask, float("-inf"))
    soft_targets = torch.softmax(scaled, dim=-1)
    return soft_targets, valid_mask


def weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def format_prompts(
    raw_prompts: Sequence[str],
    prompt_mode: str,
    prompt_contexts: Sequence[str] | None = None,
) -> List[str]:
    contexts = list(prompt_contexts or [])
    return [
        build_task_prompt(
            prompt,
            mode=prompt_mode,
            action_tags=ACTION_LIST,
            traffic_context=contexts[idx] if idx < len(contexts) else "",
        )
        for idx, prompt in enumerate(raw_prompts)
    ]


def update_class_stats(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
    class_pred_total: torch.Tensor,
) -> None:
    preds_cpu = preds.detach().view(-1).cpu()
    labels_cpu = labels.detach().view(-1).cpu()
    class_total += torch.bincount(labels_cpu, minlength=len(ACTION_LIST))
    class_pred_total += torch.bincount(preds_cpu, minlength=len(ACTION_LIST))
    correct_mask = preds_cpu == labels_cpu
    if correct_mask.any():
        class_correct += torch.bincount(labels_cpu[correct_mask], minlength=len(ACTION_LIST))


def macro_f1(
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
    class_pred_total: torch.Tensor,
) -> float:
    precision = class_correct.float() / class_pred_total.clamp(min=1).float()
    recall = class_correct.float() / class_total.clamp(min=1).float()
    denom = (precision + recall).clamp(min=1e-12)
    f1 = 2.0 * precision * recall / denom
    return float(f1.mean().item())


def sync_epoch_stats(
    ddp: bool,
    device: torch.device,
    total_loss: float,
    total_correct: int,
    total_count: int,
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
    class_pred_total: torch.Tensor,
    cost_stats: torch.Tensor,
) -> Tuple[float, int, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not ddp:
        return total_loss, total_correct, total_count, class_correct, class_total, class_pred_total, cost_stats

    stats = torch.tensor([total_loss, float(total_correct), float(total_count)], device=device, dtype=torch.float64)
    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    tensors = []
    for item in (class_correct, class_total, class_pred_total):
        tensor = item.to(device=device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensors.append(tensor.cpu())
    synced_cost_stats = cost_stats.to(device=device, dtype=torch.float64)
    dist.all_reduce(synced_cost_stats, op=dist.ReduceOp.SUM)
    return (
        float(stats[0].item()),
        int(stats[1].item()),
        int(stats[2].item()),
        tensors[0],
        tensors[1],
        tensors[2],
        synced_cost_stats.cpu(),
    )


def save_checkpoint(
    output_dir: str,
    filename: str,
    model: PrefixActionPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
    selection: Dict[str, Any] | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint = {
        "ts_projector": model.ts_projector.state_dict(),
        "vision_projector": model.vision_projector.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    if not getattr(args, "checkpoint_slim", False):
        checkpoint["optimizer"] = optimizer.state_dict()
    if model.topology_encoder is not None:
        checkpoint["topology_encoder"] = model.topology_encoder.state_dict()
    if selection is not None:
        checkpoint["checkpoint_selection"] = selection
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)
    return path


def append_metrics_jsonl(
    output_dir: str,
    epoch: int,
    metrics: Dict[str, float],
    selection: Dict[str, Any],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.jsonl")
    payload = {"epoch": int(epoch), "selection": selection}
    payload.update({key: float(value) for key, value in sorted(metrics.items())})
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")
    return path


def resolve_best_monitor(metrics: Dict[str, float], args: argparse.Namespace) -> Tuple[str, str, float]:
    metric_name = str(args.best_metric or "val_macro_f1").strip()
    if metric_name not in metrics:
        fallback_name = ""
        if metric_name.startswith("val_"):
            candidate = "train_" + metric_name[len("val_") :]
            if candidate in metrics:
                fallback_name = candidate
        if not fallback_name:
            available = ", ".join(sorted(metrics))
            raise KeyError(f"Best checkpoint metric '{metric_name}' is not available. Available metrics: {available}")
        metric_name = fallback_name

    mode = str(args.best_mode or "auto").strip().lower()
    if mode == "auto":
        mode = "min" if metric_name.endswith(("_loss", "_cost", "_regret")) else "max"
    if mode not in {"max", "min"}:
        raise ValueError(f"Unsupported best checkpoint mode: {args.best_mode}")
    return metric_name, mode, float(metrics[metric_name])


def is_better_monitor(value: float, best_value: float | None, mode: str, min_delta: float) -> bool:
    if best_value is None:
        return True
    delta = max(0.0, float(min_delta))
    if mode == "min":
        return value < best_value - delta
    return value > best_value + delta


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0


def build_lr_scheduler(args: argparse.Namespace, steps_per_epoch: int):
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0

    def lr_scale(global_step: int) -> float:
        if warmup_steps > 0 and global_step < warmup_steps:
            return float(global_step + 1) / max(1, warmup_steps)
        progress = (global_step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if args.lr_schedule == "constant":
            return 1.0
        if args.lr_schedule == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return lr_scale


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
    lr_scale_fn,
    global_step_start: int,
    epoch: int,
    ddp: bool,
    is_train: bool,
    rank: int,
    class_weights: torch.Tensor | None,
    logger: logging.Logger | None,
) -> Tuple[Dict[str, float], int]:
    base_model: PrefixActionPolicy = model.module if hasattr(model, "module") else model
    base_model.backbone.model.eval()
    if is_train:
        base_model.ts_projector.train()
        base_model.vision_projector.train()
    else:
        base_model.ts_projector.eval()
        base_model.vision_projector.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    class_correct = torch.zeros(len(ACTION_LIST), dtype=torch.long)
    class_total = torch.zeros(len(ACTION_LIST), dtype=torch.long)
    class_pred_total = torch.zeros(len(ACTION_LIST), dtype=torch.long)
    cost_stats = torch.zeros(9, dtype=torch.float64)  # [8] = invalid-pred count
    global_step = global_step_start
    num_batches = len(loader)
    epoch_started_at = time.time()

    if is_train:
        optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        sequences = batch["sequence"].to(device)
        labels = torch.tensor(
            [action_to_index(item) for item in batch["expert_action"]],
            dtype=torch.long,
            device=device,
        )
        prompts = format_prompts(
            batch["raw_prompt"],
            args.prompt_mode,
            batch.get("prompt_context"),
        )

        sync_step = ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(loader))
        sync_context = (
            nullcontext()
            if (not is_train or not ddp or sync_step or args.grad_accum_steps == 1)
            else model.no_sync()
        )

        with sync_context:
            with torch.set_grad_enabled(is_train):
                with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                    intersection_ids = batch.get("intersection_id", None)
                    traffic_files = batch.get("traffic_file", None)
                    # Derive dataset hints for topology adapter
                    dataset_hints = None
                    _model = model.module if hasattr(model, "module") else model
                    if traffic_files and _model.enable_topology_adapter:
                        from projection_train.topology_adapter import IntersectionTopologyEncoder
                        dataset_hints = IntersectionTopologyEncoder.dataset_hints_from_batch(
                            list(traffic_files), intersection_ids
                        )
                    logits = model(
                        sequences, prompts,
                        intersection_ids=intersection_ids,
                        dataset_hints=dataset_hints,
                    )
                    raw_sample_weights = batch.get("sample_weight")
                    if raw_sample_weights is None:
                        sample_weight_values = torch.ones(labels.size(0), device=device, dtype=torch.float32)
                    else:
                        sample_weight_values = raw_sample_weights.to(device=device, dtype=torch.float32).clamp_min(0.0)
                    loss_weights = (
                        sample_weight_values
                        if is_train
                        else torch.ones(labels.size(0), device=device, dtype=torch.float32)
                    )
                    hard_loss_all = F.cross_entropy(
                        logits,
                        labels,
                        weight=class_weights,
                        reduction="none",
                    )
                    raw_loss_all = hard_loss_all
                    candidate_costs = None
                    candidate_valid = None
                    if args.target_mode != "hard":
                        candidate_costs, candidate_valid = build_candidate_costs(
                            batch.get("candidate_action_metrics", []),
                            batch["candidate_action_waits"],
                            device=device,
                            weights={
                                "att": args.cost_att_weight,
                                "aql": args.cost_aql_weight,
                                "awt": args.cost_awt_weight,
                                "spillback": args.cost_spillback_weight,
                                "tail_wait": args.cost_tail_wait_weight,
                            },
                            pareto_feasible_batch=batch.get("pareto_feasible"),
                        )
                        soft_valid = candidate_valid.sum(dim=-1) >= 2
                        if bool(soft_valid.any().item()):
                            soft_targets = cost_soft_targets(
                                candidate_costs,
                                candidate_valid,
                                temperature=args.soft_label_temperature,
                            )
                            log_probs = F.log_softmax(logits, dim=-1)
                            soft_loss_all = -(soft_targets * log_probs).sum(dim=-1)
                            if args.target_mode == "soft":
                                raw_loss_all = torch.where(soft_valid, soft_loss_all, hard_loss_all)
                            else:
                                alpha = float(args.soft_target_alpha)
                                mixed_loss = (1.0 - alpha) * hard_loss_all + alpha * soft_loss_all
                                raw_loss_all = torch.where(soft_valid, mixed_loss, hard_loss_all)

                    if args.ranking_loss_weight > 0.0:
                        if candidate_costs is None or candidate_valid is None:
                            candidate_costs, candidate_valid = build_candidate_costs(
                                batch.get("candidate_action_metrics", []),
                                batch["candidate_action_waits"],
                                device=device,
                                weights={
                                    "att": args.cost_att_weight,
                                    "aql": args.cost_aql_weight,
                                    "awt": args.cost_awt_weight,
                                    "spillback": args.cost_spillback_weight,
                                    "tail_wait": args.cost_tail_wait_weight,
                                },
                                pareto_feasible_batch=batch.get("pareto_feasible"),
                            )
                        ranking_loss_all, ranking_valid = pairwise_cost_ranking_loss(
                            logits,
                            candidate_costs,
                            candidate_valid,
                            min_gap=args.ranking_min_cost_gap,
                            margin=args.ranking_margin,
                        )
                        raw_loss_all = raw_loss_all + args.ranking_loss_weight * torch.where(
                            ranking_valid,
                            ranking_loss_all,
                            torch.zeros_like(ranking_loss_all),
                        )

                    if args.base_policy_kl_weight > 0.0:
                        kl_loss_all, kl_valid = base_action_kl_loss(
                            logits,
                            batch.get("base_action", []),
                            prior_mass=args.base_policy_prior_mass,
                        )
                        kl_mask = base_policy_kl_mask(
                            batch.get("base_action", []),
                            batch.get("expert_action", []),
                            sample_weight_values,
                            mode=args.base_policy_kl_mode,
                            low_weight_threshold=args.base_policy_kl_low_weight_threshold,
                            device=device,
                        )
                        kl_valid = kl_valid & kl_mask
                        raw_loss_all = raw_loss_all + args.base_policy_kl_weight * torch.where(
                            kl_valid,
                            kl_loss_all,
                            torch.zeros_like(kl_loss_all),
                        )

                    raw_loss = weighted_mean(raw_loss_all, loss_weights)
                    if getattr(args, "action_balance_weight", 0.0) > 0:
                        raw_loss = raw_loss + args.action_balance_weight * _compute_action_balance_loss(logits)
                    if getattr(args, "overconf_penalty_weight", 0.0) > 0:
                        raw_loss = raw_loss + args.overconf_penalty_weight * _compute_overconf_penalty(logits)
                    loss = raw_loss / max(1, args.grad_accum_steps)
                if is_train:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

        if is_train and sync_step:
            lr_scale = lr_scale_fn(global_step)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(base_model.trainable_parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(base_model.trainable_parameters(), args.max_grad_norm)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            total_loss += float(raw_loss.item()) * labels.size(0)
            total_correct += int((preds == labels).sum().item())
            total_count += int(labels.size(0))
            update_class_stats(preds, labels, class_correct, class_total, class_pred_total)
            if candidate_costs is None or candidate_valid is None:
                candidate_costs, candidate_valid = build_candidate_costs(
                    batch.get("candidate_action_metrics", []),
                    batch["candidate_action_waits"],
                    device=device,
                    weights={
                        "att": args.cost_att_weight,
                        "aql": args.cost_aql_weight,
                        "awt": args.cost_awt_weight,
                        "spillback": args.cost_spillback_weight,
                        "tail_wait": args.cost_tail_wait_weight,
                    },
                    pareto_feasible_batch=batch.get("pareto_feasible"),
                )
            metric_valid = candidate_valid.sum(dim=-1) >= 2
            if bool(metric_valid.any().item()):
                safe_costs = candidate_costs.masked_fill(~candidate_valid, float("inf"))
                oracle_cost = safe_costs.amin(dim=-1)
                pred_cost_raw = candidate_costs.gather(1, preds.view(-1, 1)).squeeze(1)
                label_cost = candidate_costs.gather(1, labels.view(-1, 1)).squeeze(1)
                pred_valid = candidate_valid.gather(1, preds.view(-1, 1)).squeeze(1)
                label_valid = candidate_valid.gather(1, labels.view(-1, 1)).squeeze(1)
                # Infeasible predictions are penalized, NOT dropped: cost =
                # worst feasible cost + one feasible-range gap + 1.0. Dropping
                # them lets a model game the selection metric by predicting
                # infeasible actions (they disappear from the regret).
                worst_feasible = (
                    candidate_costs.masked_fill(~candidate_valid, float("-inf")).amax(dim=-1)
                )
                infeasible_penalty = (
                    (worst_feasible - oracle_cost).clamp_min(0.0) + 1.0
                )
                pred_cost = torch.where(
                    pred_valid, pred_cost_raw, worst_feasible + infeasible_penalty
                )
                probs = F.softmax(logits.float(), dim=-1)
                valid_probs = probs * candidate_valid.to(dtype=probs.dtype)
                valid_prob_mass = valid_probs.sum(dim=-1).clamp_min(1e-8)
                expected_cost = (
                    valid_probs * candidate_costs.masked_fill(~candidate_valid, 0.0)
                ).sum(dim=-1) / valid_prob_mass
                metric_valid = (
                    metric_valid
                    & label_valid
                    & torch.isfinite(oracle_cost)
                    & torch.isfinite(pred_cost)
                    & torch.isfinite(label_cost)
                    & torch.isfinite(expected_cost)
                )
                if bool(metric_valid.any().item()):
                    selected_oracle = oracle_cost[metric_valid].double()
                    selected_top1 = pred_cost[metric_valid].double()
                    selected_label = label_cost[metric_valid].double()
                    selected_expected = expected_cost[metric_valid].double()
                    selected_pred_valid = pred_valid[metric_valid]
                    cost_stats[0] += float(metric_valid.sum().item())
                    cost_stats[1] += selected_top1.sum().cpu()
                    cost_stats[2] += selected_oracle.sum().cpu()
                    cost_stats[3] += selected_label.sum().cpu()
                    cost_stats[4] += selected_expected.sum().cpu()
                    cost_stats[5] += (selected_top1 - selected_oracle).sum().cpu()
                    cost_stats[6] += (selected_expected - selected_oracle).sum().cpu()
                    cost_stats[7] += (selected_label - selected_oracle).sum().cpu()
                    cost_stats[8] += float((~selected_pred_valid).sum().item())

        should_log_step = (
            rank == 0
            and is_train
            and logger is not None
            and (
                step == 0
                or (args.log_interval > 0 and (step + 1) % args.log_interval == 0)
                or (step + 1 == num_batches)
            )
        )
        if should_log_step:
            avg_loss = total_loss / max(1, total_count)
            avg_acc = total_correct / max(1, total_count)
            logger.info(
                "[train epoch %d] step %d/%d (%.1f%%) loss=%.4f acc=%.4f lr=%.6g elapsed=%.1fs global_step=%d",
                epoch,
                step + 1,
                num_batches,
                100.0 * float(step + 1) / max(1, num_batches),
                avg_loss,
                avg_acc,
                current_lr(optimizer),
                time.time() - epoch_started_at,
                global_step,
            )

    total_loss, total_correct, total_count, class_correct, class_total, class_pred_total, cost_stats = sync_epoch_stats(
        ddp=ddp,
        device=device,
        total_loss=total_loss,
        total_correct=total_correct,
        total_count=total_count,
        class_correct=class_correct,
        class_total=class_total,
        class_pred_total=class_pred_total,
        cost_stats=cost_stats,
    )
    metrics = {
        "loss": total_loss / max(1, total_count),
        "acc": total_correct / max(1, total_count),
        "macro_f1": macro_f1(class_correct, class_total, class_pred_total),
    }
    cost_count = float(cost_stats[0].item())
    if cost_count > 0.0:
        metrics.update(
            {
                "cost_count": cost_count,
                "top1_cost": float(cost_stats[1].item() / cost_count),
                "oracle_cost": float(cost_stats[2].item() / cost_count),
                "label_cost": float(cost_stats[3].item() / cost_count),
                "expected_cost": float(cost_stats[4].item() / cost_count),
                "top1_cost_regret": float(cost_stats[5].item() / cost_count),
                "expected_cost_regret": float(cost_stats[6].item() / cost_count),
                "label_cost_regret": float(cost_stats[7].item() / cost_count),
                "invalid_action_rate": float(cost_stats[8].item() / cost_count),
            }
        )
    return metrics, global_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, nargs="+", default=None)
    parser.add_argument("--train_data_path", type=str, nargs="+", default=None)
    parser.add_argument("--val_data_path", type=str, nargs="+", default=None)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--align_ckpt", type=str, required=True)
    parser.add_argument(
        "--allow_mismatched_align_projectors",
        action="store_true",
        default=False,
        help=(
            "Allow cross-architecture alignment transfer by skipping incompatible "
            "projector tensors. Intended only for explicit capacity ablations."
        ),
    )
    parser.add_argument(
        "--init_policy_ckpt", type=str, default="",
        help="Optional Stage 2 checkpoint to initialize trainable weights from. "
             "Loads prefix/TS2I weights (not optimizer state) for few-shot adaptation.",
    )
    parser.add_argument("--output_dir", type=str, default="records/projection_bc")
    parser.add_argument("--intersection_id", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--periodicity", type=int, default=16)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--prefix_tokens", type=int, default=16)
    parser.add_argument("--prefix_mode", type=str, default="prefix", choices=["single", "prefix"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prompt_mode", type=str, default="image_only_short", choices=list(PROMPT_MODES))
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument(
        "--validation_purge_samples",
        type=int,
        default=0,
        help=(
            "For blocked validation, discard this many boundary samples before "
            "the held-out suffix so temporal histories cannot cross the split."
        ),
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="stratified",
        choices=["stratified", "blocked"],
        help="Dataset split strategy. stratified keeps expert_action classes represented in validation.",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--min_action_margin", type=float, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--checkpoint_slim",
        action="store_true",
        default=False,
        help="Omit optimizer state from saved checkpoints (reproduce/eval only, no resume).",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="val_macro_f1",
        choices=[
            "val_macro_f1",
            "val_acc",
            "val_loss",
            "val_top1_cost_regret",
            "val_expected_cost_regret",
            "val_top1_cost",
            "train_macro_f1",
            "train_acc",
            "train_loss",
            "train_top1_cost_regret",
            "train_expected_cost_regret",
            "train_top1_cost",
        ],
        help="Metric used to update best.pt during offline BC training.",
    )
    parser.add_argument(
        "--best_mode",
        type=str,
        default="auto",
        choices=["auto", "max", "min"],
        help="Whether larger or smaller best_metric is better. auto minimizes loss and maximizes other metrics.",
    )
    parser.add_argument(
        "--best_min_delta",
        type=float,
        default=0.0,
        help="Minimum metric improvement required before overwriting best.pt.",
    )
    parser.add_argument(
        "--target_mode",
        type=str,
        default="mixed",
        choices=["hard", "soft", "mixed"],
    )
    parser.add_argument("--soft_target_alpha", type=float, default=0.3)
    parser.add_argument("--soft_label_temperature", type=float, default=0.35)
    parser.add_argument(
        "--enable_class_balance",
        action="store_true",
        default=False,
        help="Opt in to inverse-frequency class weights. Disabled by default to preserve demand-conditioned priors.",
    )
    parser.add_argument(
        "--disable_class_balance",
        action="store_true",
        default=False,
        help="Deprecated compatibility flag; class balancing is already disabled by default.",
    )
    parser.add_argument("--action_balance_weight", type=float, default=0.0)
    parser.add_argument("--overconf_penalty_weight", type=float, default=0.0)
    parser.add_argument("--global_action_weights", type=str, default=None)
    parser.add_argument("--ranking_loss_weight", type=float, default=0.15)
    parser.add_argument("--ranking_margin", type=float, default=0.0)
    parser.add_argument("--ranking_min_cost_gap", type=float, default=0.02)
    parser.add_argument("--base_policy_kl_weight", type=float, default=0.02)
    parser.add_argument("--base_policy_prior_mass", type=float, default=0.70)
    parser.add_argument(
        "--base_policy_kl_mode",
        type=str,
        default="match_or_low_weight",
        choices=["all", "expert_match", "low_weight", "match_or_low_weight"],
        help="Rows where KL(policy || base-action prior) is applied.",
    )
    parser.add_argument(
        "--base_policy_kl_low_weight_threshold",
        type=float,
        default=1.0,
        help="Sample-weight threshold treated as non-DAgger/low-priority for conditional KL.",
    )
    parser.add_argument("--cost_att_weight", type=float, default=DEFAULT_COST_WEIGHTS["att"])
    parser.add_argument("--cost_aql_weight", type=float, default=DEFAULT_COST_WEIGHTS["aql"])
    parser.add_argument("--cost_awt_weight", type=float, default=DEFAULT_COST_WEIGHTS["awt"])
    parser.add_argument("--cost_spillback_weight", type=float, default=DEFAULT_COST_WEIGHTS["spillback"])
    parser.add_argument("--cost_tail_wait_weight", type=float, default=DEFAULT_COST_WEIGHTS["tail_wait"])
    parser.add_argument("--ddp_timeout_seconds", type=int, default=1800)
    parser.add_argument("--enable_topology_adapter", action="store_true", default=False,
                        help="Enable topology-conditioned intersection adapter (Ablation E)")
    parser.add_argument("--topology_roadnet_configs", type=str, default="",
                        help="JSON string mapping dataset names to {rows, cols} for topology adapter")
    return parser.parse_args()


def main() -> None:
    configure_stdio()
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")
    if args.min_lr < 0 or args.min_lr > args.lr:
        raise ValueError("--min_lr must be within [0, lr]")
    if args.validation_purge_samples < 0:
        raise ValueError("--validation_purge_samples must be non-negative")
    if args.enable_class_balance and args.disable_class_balance:
        raise ValueError("Choose at most one of --enable_class_balance and --disable_class_balance")
    if not 0.25 <= args.base_policy_prior_mass < 1.0:
        raise ValueError("--base_policy_prior_mass must be in [0.25, 1.0)")
    if min(
        args.ranking_loss_weight,
        args.ranking_margin,
        args.ranking_min_cost_gap,
        args.base_policy_kl_weight,
        args.cost_att_weight,
        args.cost_aql_weight,
        args.cost_awt_weight,
        args.cost_spillback_weight,
        args.cost_tail_wait_weight,
        args.base_policy_kl_low_weight_threshold,
    ) < 0.0:
        raise ValueError("Ranking, KL, and cost weights must be non-negative")
    if bool(args.data_path) == bool(args.train_data_path):
        raise ValueError("Provide exactly one of --data_path or --train_data_path")
    if args.val_data_path and not args.train_data_path:
        raise ValueError("--val_data_path requires --train_data_path")

    ddp, rank, world_size, device = init_distributed(args.ddp_timeout_seconds)
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(args.output_dir, rank)
    seed_everything(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if rank == 0:
        logger.info(
            "Stage 2 BC start: device=%s world_size=%d output_dir=%s log_interval=%d grad_accum_steps=%d",
            device,
            world_size,
            args.output_dir,
            args.log_interval,
            args.grad_accum_steps,
        )
        logger.info(
            "Loss config: target_mode=%s soft_alpha=%.3f soft_temp=%.3f ranking_weight=%.3f "
            "base_kl_weight=%.3f base_kl_mode=%s low_weight_threshold=%.3f",
            args.target_mode,
            args.soft_target_alpha,
            args.soft_label_temperature,
            args.ranking_loss_weight,
            args.base_policy_kl_weight,
            args.base_policy_kl_mode,
            args.base_policy_kl_low_weight_threshold,
        )
        logger.info("Distributed timeout: %ds", args.ddp_timeout_seconds)
        if args.train_data_path:
            logger.info("Loading explicit train dataset from: %s", ", ".join(args.train_data_path))
            if args.val_data_path:
                logger.info("Loading explicit val dataset from: %s", ", ".join(args.val_data_path))
        else:
            logger.info("Loading grouped dataset from: %s", ", ".join(args.data_path or []))

    explicit_eval = bool(args.train_data_path)
    if explicit_eval:
        train_set = GroupedTrafficActionDataset(
            path=args.train_data_path,
            seq_len=args.seq_len,
            intersection_id=args.intersection_id,
            require_expert_action=True,
            valid_actions=ACTION_LIST,
            max_samples=args.max_samples,
            min_action_margin=args.min_action_margin,
        )
        val_set = (
            GroupedTrafficActionDataset(
                path=args.val_data_path,
                seq_len=args.seq_len,
                intersection_id=args.intersection_id,
                require_expert_action=True,
                valid_actions=ACTION_LIST,
                max_samples=None,
                min_action_margin=None,
            )
            if args.val_data_path
            else None
        )
        dataset = None
        if rank == 0:
            logger.info(
                "Loaded explicit datasets: train_samples=%d val_samples=%d",
                len(train_set),
                len(val_set) if val_set is not None else 0,
            )
    else:
        dataset = GroupedTrafficActionDataset(
            path=args.data_path,
            seq_len=args.seq_len,
            intersection_id=args.intersection_id,
            require_expert_action=True,
            valid_actions=ACTION_LIST,
            max_samples=args.max_samples,
            min_action_margin=args.min_action_margin,
        )
        if rank == 0:
            logger.info("Loaded dataset with %d samples before split.", len(dataset))
        train_set, val_set = split_dataset(
            dataset,
            val_ratio=args.val_ratio,
            seed=args.seed,
            split_strategy=args.split_strategy,
            validation_purge_samples=args.validation_purge_samples,
        )
    if rank == 0:
        train_distribution = action_distribution(train_set)
        val_distribution = action_distribution(val_set) if val_set is not None else {}
        if explicit_eval:
            logger.info("Split strategy: explicit train/val files")
        else:
            total_distribution = action_distribution(dataset)
            logger.info(
                "Split strategy: %s val_ratio=%.4f seed=%d validation_purge_samples=%d",
                args.split_strategy,
                args.val_ratio,
                args.seed,
                args.validation_purge_samples,
            )
            logger.info("Action distribution total: %s", format_action_distribution(total_distribution))
        logger.info("Action distribution train: %s", format_action_distribution(train_distribution))
        if val_set is not None:
            logger.info("Action distribution val: %s", format_action_distribution(val_distribution))
            missing_val_actions = [action for action in ACTION_LIST if int(val_distribution.get(action, 0)) == 0]
            if missing_val_actions:
                logger.warning(
                    "Validation split is missing action classes: %s",
                    ",".join(missing_val_actions),
                )
        else:
            logger.info("Action distribution val: none")
    class_weights = None
    if args.enable_class_balance:
        class_weights = compute_class_weights(train_set)
    elif rank == 0:
        logger.info("Class balancing: disabled (default demand-conditioned policy prior).")
    if args.global_action_weights:
        global_w = _parse_global_action_weights(args.global_action_weights)
        if global_w is not None:
            if class_weights is not None:
                class_weights = class_weights * global_w
            else:
                class_weights = global_w
    eval_set = shard_eval_subset(val_set, ddp=ddp, rank=rank, world_size=world_size)

    if ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=grouped_action_collate,
            drop_last=False,
        )
        val_loader = None
        if eval_set is not None:
            val_loader = DataLoader(
                eval_set,
                batch_size=args.batch_size,
                sampler=SequentialSampler(eval_set),
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=grouped_action_collate,
                drop_last=False,
            )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_set),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=grouped_action_collate,
            drop_last=False,
        )
        val_loader = (
            DataLoader(
                eval_set,
                batch_size=args.batch_size,
                sampler=SequentialSampler(eval_set),
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=grouped_action_collate,
                drop_last=False,
            )
            if eval_set is not None
            else None
        )

    if rank == 0:
        logger.info(
            "Prepared dataloaders: train_samples=%d val_samples=%d local_val_samples=%d train_batches=%d local_val_batches=%d batch_size=%d num_workers=%d",
            len(train_set),
            len(val_set) if val_set is not None else 0,
            len(eval_set) if eval_set is not None else 0,
            len(train_loader),
            len(val_loader) if val_loader is not None else 0,
            args.batch_size,
            args.num_workers,
        )

    amp_dtype = resolve_amp_dtype(args, device)
    torch_dtype = amp_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    if rank == 0:
        logger.info(
            "Building PrefixActionPolicy from llm_path=%s align_ckpt=%s amp_dtype=%s",
            args.llm_path,
            args.align_ckpt,
            amp_dtype,
        )
    model = PrefixActionPolicy(
        model_path=args.llm_path,
        seq_len=args.seq_len,
        image_size=args.image_size,
        periodicity=args.periodicity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens,
        prefix_mode=args.prefix_mode,
        max_length=args.max_length,
        torch_dtype=torch_dtype,
        device=device,
        with_value_head=False,
        enable_topology_adapter=args.enable_topology_adapter,
        topology_roadnet_configs=(
            json.loads(args.topology_roadnet_configs)
            if args.topology_roadnet_configs
            else None
        ),
    )
    model.load_checkpoint(
        args.align_ckpt,
        load_value_head=False,
        allow_mismatched_projectors=args.allow_mismatched_align_projectors,
    )
    if rank == 0:
        logger.info(
            "Loaded alignment checkpoint and initialized trainable projectors "
            "(allow_mismatched_align_projectors=%s).",
            args.allow_mismatched_align_projectors,
        )

    # Optional: initialize trainable weights from a source Stage 2 checkpoint
    # (used for few-shot target-network adaptation)
    if args.init_policy_ckpt:
        if rank == 0:
            logger.info("Loading source policy weights from %s", args.init_policy_ckpt)
        model.load_checkpoint(args.init_policy_ckpt, load_value_head=False)
        if rank == 0:
            logger.info("Initialized trainable weights from source Stage 2 checkpoint.")

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16 and device.type == "cuda"))
    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )
    class_weights = class_weights.to(device) if class_weights is not None else None

    lr_scale_fn = build_lr_scheduler(args, max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps))))
    best_monitor_value: float | None = None
    global_step = 0
    if rank == 0:
        logger.info(
            "Best checkpoint selection: metric=%s mode=%s min_delta=%.6g",
            args.best_metric,
            args.best_mode,
            args.best_min_delta,
        )

    for epoch in range(args.epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            logger.info("Epoch %d/%d started.", epoch + 1, args.epochs)

        train_metrics, global_step = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            amp_dtype=amp_dtype,
            scaler=scaler if amp_dtype == torch.float16 else None,
            lr_scale_fn=lr_scale_fn,
            global_step_start=global_step,
            epoch=epoch + 1,
            ddp=ddp,
            is_train=True,
            rank=rank,
            class_weights=class_weights,
            logger=logger,
        )

        if rank == 0:
            logger.info(
                "[epoch %d] train_loss=%.4f train_acc=%.4f train_f1=%.4f",
                epoch + 1,
                train_metrics["loss"],
                train_metrics["acc"],
                train_metrics["macro_f1"],
            )

        val_metrics = None
        if val_loader is not None:
            if rank == 0:
                logger.info("Epoch %d/%d validation started.", epoch + 1, args.epochs)
            eval_model: PrefixActionPolicy = model.module if hasattr(model, "module") else model
            with torch.no_grad():
                val_metrics, _ = run_epoch(
                    model=eval_model,
                    loader=val_loader,
                    optimizer=optimizer,
                    device=device,
                    args=args,
                    amp_dtype=amp_dtype,
                    scaler=None,
                    lr_scale_fn=lr_scale_fn,
                    global_step_start=global_step,
                    epoch=epoch + 1,
                    ddp=ddp,
                    is_train=False,
                    rank=rank,
                    class_weights=class_weights,
                    logger=logger,
                )
            if rank == 0 and val_metrics is not None:
                logger.info(
                    "[epoch %d] val_loss=%.4f val_acc=%.4f val_f1=%.4f",
                    epoch + 1,
                    val_metrics["loss"],
                    val_metrics["acc"],
                    val_metrics["macro_f1"],
                )
                if "top1_cost_regret" in val_metrics:
                    logger.info(
                        "[epoch %d] val_top1_cost_regret=%.6f val_expected_cost_regret=%.6f val_top1_cost=%.6f",
                        epoch + 1,
                        val_metrics["top1_cost_regret"],
                        val_metrics["expected_cost_regret"],
                        val_metrics["top1_cost"],
                    )

        if rank == 0:
            base_model: PrefixActionPolicy = model.module if hasattr(model, "module") else model
            metrics = {f"train_{key}": value for key, value in train_metrics.items()}
            if val_metrics is not None:
                metrics.update({f"val_{key}": value for key, value in val_metrics.items()})
            monitor_metric, monitor_mode, monitor_value = resolve_best_monitor(metrics, args)
            selection = {
                "type": "offline_metric",
                "metric": monitor_metric,
                "mode": monitor_mode,
                "value": monitor_value,
                "min_delta": float(args.best_min_delta),
            }
            metrics_path = append_metrics_jsonl(
                output_dir=args.output_dir,
                epoch=epoch + 1,
                metrics=metrics,
                selection=selection,
            )
            logger.info("Wrote metrics: %s", metrics_path)

            last_path = save_checkpoint(
                output_dir=args.output_dir,
                filename="last.pt",
                model=base_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                args=args,
                metrics=metrics,
                selection=selection,
            )
            logger.info("Saved checkpoint: %s", last_path)
            if (epoch + 1) % args.save_every == 0:
                epoch_path = save_checkpoint(
                    output_dir=args.output_dir,
                    filename=f"epoch_{epoch + 1}.pt",
                    model=base_model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    args=args,
                    metrics=metrics,
                    selection=selection,
                )
                logger.info("Saved checkpoint: %s", epoch_path)
            if is_better_monitor(
                value=monitor_value,
                best_value=best_monitor_value,
                mode=monitor_mode,
                min_delta=args.best_min_delta,
            ):
                previous_best = best_monitor_value
                best_monitor_value = monitor_value
                best_selection = dict(selection)
                best_selection["previous_best_value"] = previous_best
                best_path = save_checkpoint(
                    output_dir=args.output_dir,
                    filename="best.pt",
                    model=base_model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    args=args,
                    metrics=metrics,
                    selection=best_selection,
                )
                logger.info(
                    "Updated best checkpoint: %s (%s=%.6f mode=%s previous=%s)",
                    best_path,
                    monitor_metric,
                    monitor_value,
                    monitor_mode,
                    "none" if previous_best is None else f"{previous_best:.6f}",
                )

        if ddp:
            barrier()

    if rank == 0:
        logger.info("Stage 2 BC training finished.")

    if ddp:
        barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
