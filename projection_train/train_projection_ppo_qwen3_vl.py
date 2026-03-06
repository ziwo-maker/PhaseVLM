from __future__ import annotations

import argparse
import os
import random
import sys
from typing import Dict, List, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from torch.utils.data.distributed import DistributedSampler

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.action_space import ACTION_LIST, action_to_index
from projection_train.grouped_dataset import GroupedTrafficActionDataset, grouped_action_collate
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.prompt_utils import build_task_prompt
from projection_train.snapshot_env import SnapshotActionEvaluator


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, torch.device]:
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def resolve_amp_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype | None:
    if device.type != "cuda":
        return None
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return None


def split_dataset(dataset, val_ratio: float, seed: int) -> tuple[Subset, Subset | None]:
    total = len(dataset)
    if total == 0:
        raise RuntimeError("No grouped samples found for Stage 3 training.")
    val_size = int(total * val_ratio)
    if total > 1:
        val_size = max(1, val_size) if val_ratio > 0 else 0
        val_size = min(val_size, total - 1)
    else:
        val_size = 0
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), (Subset(dataset, val_indices) if val_size > 0 else None)


def format_prompts(raw_prompts: Sequence[str], prompt_mode: str) -> List[str]:
    return [build_task_prompt(prompt, mode=prompt_mode, action_tags=ACTION_LIST) for prompt in raw_prompts]


def unwrap_module(module: torch.nn.Module | None) -> torch.nn.Module | None:
    if module is None:
        return None
    return module.module if hasattr(module, "module") else module


def wrap_trainable_modules(model: PrefixActionPolicy, device: torch.device, ddp: bool) -> None:
    if not ddp:
        return
    device_ids = [device.index] if device.type == "cuda" and device.index is not None else None
    model.ts_projector = DistributedDataParallel(
        model.ts_projector,
        device_ids=device_ids,
        find_unused_parameters=False,
    )
    model.vision_projector = DistributedDataParallel(
        model.vision_projector,
        device_ids=device_ids,
        find_unused_parameters=False,
    )
    if model.value_head is not None:
        model.value_head = DistributedDataParallel(
            model.value_head,
            device_ids=device_ids,
            find_unused_parameters=False,
        )


def build_evaluator_work_root(output_dir: str, rank: int, world_size: int) -> str:
    base = os.path.join(output_dir, "cityflow_cache")
    if world_size <= 1:
        return base
    return os.path.join(base, f"rank_{rank}")


def sync_train_metrics(
    ddp: bool,
    device: torch.device,
    total_reward: float,
    total_wait: float,
    total_acc: float,
    total_count: int,
    total_policy_loss: float,
    total_value_loss: float,
    total_entropy: float,
) -> Dict[str, float]:
    stats = torch.tensor(
        [
            total_reward,
            total_wait,
            total_acc,
            float(total_count),
            total_policy_loss,
            total_value_loss,
            total_entropy,
        ],
        device=device,
        dtype=torch.float64,
    )
    if ddp:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    count = max(1.0, float(stats[3].item()))
    return {
        "reward": float(stats[0].item() / count),
        "avg_wait": float(stats[1].item() / count),
        "expert_acc": float(stats[2].item() / count),
        "policy_loss": float(stats[4].item() / count),
        "value_loss": float(stats[5].item() / count),
        "entropy": float(stats[6].item() / count),
        "count": float(stats[3].item()),
    }


def save_checkpoint(
    output_dir: str,
    filename: str,
    model: PrefixActionPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts_projector = unwrap_module(model.ts_projector)
    vision_projector = unwrap_module(model.vision_projector)
    value_head = unwrap_module(model.value_head)
    checkpoint = {
        "ts_projector": ts_projector.state_dict() if ts_projector is not None else None,
        "vision_projector": vision_projector.state_dict() if vision_projector is not None else None,
        "value_head": value_head.state_dict() if value_head is not None else None,
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }
    path = os.path.join(output_dir, filename)
    torch.save(checkpoint, path)
    return path


def collect_bandit_batch(
    model: PrefixActionPolicy,
    batch: Dict,
    evaluator: SnapshotActionEvaluator,
    prompt_mode: str,
    device: torch.device,
    deterministic: bool = False,
) -> Dict[str, torch.Tensor | List[str] | List[float]]:
    sequences = batch["sequence"].to(device)
    prompts = format_prompts(batch["raw_prompt"], prompt_mode)
    with torch.no_grad():
        prepared = model.prepare_policy_batch(sequences, prompts)
        action_logits = model.get_action_logits_from_prepared(prepared)
        values = model.get_values_from_prepared(prepared)
        dist = torch.distributions.Categorical(logits=action_logits)
        actions = action_logits.argmax(dim=-1) if deterministic else dist.sample()
        old_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()

    rewards: List[float] = []
    avg_waits: List[float] = []
    for snapshot_path, traffic_file, intersection_id, action in zip(
        batch["snapshot_path"],
        batch["traffic_file"],
        batch["intersection_id"],
        actions.tolist(),
    ):
        result = evaluator.simulate_action(
            snapshot_path=snapshot_path,
            traffic_file=traffic_file,
            intersection_id=intersection_id,
            action_index=int(action),
        )
        rewards.append(float(result["reward"]))
        avg_waits.append(float(result["avg_wait"]))

    reward_tensor = torch.tensor(rewards, device=device, dtype=torch.float32)
    returns = reward_tensor
    advantages = returns - values.detach().float()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)

    expert_actions = torch.tensor(
        [action_to_index(item) for item in batch["expert_action"]],
        dtype=torch.long,
        device=device,
    )
    expert_acc = float((actions == expert_actions).float().mean().item())
    return {
        "sequences": sequences,
        "prompts": prompts,
        "actions": actions.detach(),
        "old_logprobs": old_logprobs.detach(),
        "returns": returns.detach(),
        "advantages": advantages.detach(),
        "reward_tensor": reward_tensor.detach(),
        "entropy": entropy.detach(),
        "avg_waits": avg_waits,
        "expert_acc": expert_acc,
    }


def ppo_update(
    model: PrefixActionPolicy,
    optimizer: torch.optim.Optimizer,
    rollout: Dict[str, torch.Tensor | List[str] | List[float]],
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
) -> Dict[str, float]:
    sequences = rollout["sequences"]
    prompts = rollout["prompts"]
    actions = rollout["actions"]
    old_logprobs = rollout["old_logprobs"]
    returns = rollout["returns"]
    advantages = rollout["advantages"]

    batch_size = int(actions.shape[0])
    mini_batch_size = min(args.mini_batch_size, batch_size)
    policy_losses: List[float] = []
    value_losses: List[float] = []
    entropies: List[float] = []

    for _ in range(args.ppo_epochs):
        order = torch.randperm(batch_size, device=actions.device)
        for start in range(0, batch_size, mini_batch_size):
            batch_index = order[start : start + mini_batch_size]
            seq_mb = sequences.index_select(0, batch_index)
            act_mb = actions.index_select(0, batch_index)
            old_logprob_mb = old_logprobs.index_select(0, batch_index)
            returns_mb = returns.index_select(0, batch_index)
            adv_mb = advantages.index_select(0, batch_index)
            prompt_mb = [prompts[int(idx)] for idx in batch_index.tolist()]

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                prepared = model.prepare_policy_batch(seq_mb, prompt_mb)
                outputs = model.evaluate_actions_from_prepared(prepared, act_mb)
                new_logprob = outputs["logprob"]
                entropy = outputs["entropy"].mean()
                values = outputs["values"]
                ratio = torch.exp(new_logprob - old_logprob_mb)
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.float(), returns_mb.float())
                total_loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), args.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.detach().item()))
            value_losses.append(float(value_loss.detach().item()))
            entropies.append(float(entropy.detach().item()))

    return {
        "policy_loss": sum(policy_losses) / max(1, len(policy_losses)),
        "value_loss": sum(value_losses) / max(1, len(value_losses)),
        "entropy": sum(entropies) / max(1, len(entropies)),
    }


def evaluate_policy(
    model: PrefixActionPolicy,
    loader: DataLoader,
    evaluator: SnapshotActionEvaluator,
    prompt_mode: str,
    device: torch.device,
    max_samples: int,
) -> Dict[str, float]:
    total_reward = 0.0
    total_wait = 0.0
    total_acc = 0.0
    total_count = 0

    for batch in loader:
        rollout = collect_bandit_batch(
            model=model,
            batch=batch,
            evaluator=evaluator,
            prompt_mode=prompt_mode,
            device=device,
            deterministic=True,
        )
        reward_tensor = rollout["reward_tensor"]
        total_reward += float(reward_tensor.sum().item())
        total_wait += float(sum(rollout["avg_waits"]))
        total_acc += float(rollout["expert_acc"]) * reward_tensor.numel()
        total_count += int(reward_tensor.numel())
        if total_count >= max_samples:
            break

    return {
        "reward": total_reward / max(1, total_count),
        "avg_wait": total_wait / max(1, total_count),
        "expert_acc": total_acc / max(1, total_count),
        "count": float(total_count),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, nargs="+")
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="records/projection_ppo")
    parser.add_argument("--init_adapter_ckpt", type=str, default="")
    parser.add_argument("--align_ckpt", type=str, default="")
    parser.add_argument("--intersection_id", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--periodicity", type=int, default=16)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--prefix_tokens", type=int, default=16)
    parser.add_argument("--prefix_mode", type=str, default="prefix", choices=["single", "prefix"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--prompt_mode", type=str, default="image_only_short", choices=["raw", "image_only_short"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=64)
    parser.add_argument("--roadnet_path", type=str, default="data/Jinan/3_4/roadnet_3_4.json")
    parser.add_argument("--traffic_root", type=str, default="data/Jinan/3_4")
    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--yellow_time", type=int, default=5)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--sim_horizon_s", type=int, default=30)
    parser.add_argument("--sim_wait_speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--llm_fsdp", type=str, default="off", choices=["off", "on"])
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
    )
    parser.add_argument("--log_interval", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")
    ddp, rank, world_size, device = init_distributed()
    seed_everything(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    dataset = GroupedTrafficActionDataset(
        path=args.data_path,
        seq_len=args.seq_len,
        intersection_id=args.intersection_id,
        require_expert_action=True,
        valid_actions=ACTION_LIST,
        max_samples=args.max_samples,
    )
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)
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
        if val_set is not None and rank == 0:
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                sampler=SequentialSampler(val_set),
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
                val_set,
                batch_size=args.batch_size,
                sampler=SequentialSampler(val_set),
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                collate_fn=grouped_action_collate,
                drop_last=False,
            )
            if val_set is not None
            else None
        )

    amp_dtype = resolve_amp_dtype(args, device)
    torch_dtype = amp_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
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
        with_value_head=True,
        llm_fsdp=(args.llm_fsdp == "on"),
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
    )
    init_ckpt = args.init_adapter_ckpt or args.align_ckpt
    if init_ckpt:
        model.load_checkpoint(init_ckpt, load_value_head=False)
    wrap_trainable_modules(model, device=device, ddp=ddp)

    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    evaluator = SnapshotActionEvaluator(
        roadnet_path=args.roadnet_path,
        traffic_root=args.traffic_root,
        work_root=build_evaluator_work_root(args.output_dir, rank, world_size),
        min_action_time=args.min_action_time,
        yellow_time=args.yellow_time,
        interval=args.interval,
        wait_speed=args.sim_wait_speed,
        horizon_s=args.sim_horizon_s,
    )

    if rank == 0:
        print(
            f"[stage3] device={device} world_size={world_size} "
            f"llm_fsdp_requested={args.llm_fsdp} llm_fsdp_active={getattr(model.backbone, 'is_fsdp', False)}"
        )

    best_reward = float("-inf")

    for epoch in range(args.epochs):
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        model.ts_projector.train()
        model.vision_projector.train()
        if model.value_head is not None:
            model.value_head.train()
        model.backbone.model.eval()

        epoch_reward = 0.0
        epoch_wait = 0.0
        epoch_acc = 0.0
        epoch_count = 0
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0

        for batch_idx, batch in enumerate(train_loader):
            rollout = collect_bandit_batch(
                model=model,
                batch=batch,
                evaluator=evaluator,
                prompt_mode=args.prompt_mode,
                device=device,
                deterministic=False,
            )
            update_stats = ppo_update(
                model=model,
                optimizer=optimizer,
                rollout=rollout,
                args=args,
                amp_dtype=amp_dtype,
            )

            reward_tensor = rollout["reward_tensor"]
            batch_size = int(reward_tensor.numel())
            epoch_reward += float(reward_tensor.sum().item())
            epoch_wait += float(sum(rollout["avg_waits"]))
            epoch_acc += float(rollout["expert_acc"]) * batch_size
            epoch_count += batch_size
            epoch_policy_loss += update_stats["policy_loss"] * batch_size
            epoch_value_loss += update_stats["value_loss"] * batch_size
            epoch_entropy += update_stats["entropy"] * batch_size

            if rank == 0 and args.log_interval > 0 and (batch_idx + 1) % args.log_interval == 0:
                print(
                    f"[epoch {epoch + 1} step {batch_idx + 1}/{len(train_loader)}] "
                    f"reward={epoch_reward / max(1, epoch_count):.4f} "
                    f"avg_wait={epoch_wait / max(1, epoch_count):.4f} "
                    f"expert_acc={epoch_acc / max(1, epoch_count):.4f}"
                )

        train_metrics = sync_train_metrics(
            ddp=ddp,
            device=device,
            total_reward=epoch_reward,
            total_wait=epoch_wait,
            total_acc=epoch_acc,
            total_count=epoch_count,
            total_policy_loss=epoch_policy_loss,
            total_value_loss=epoch_value_loss,
            total_entropy=epoch_entropy,
        )
        if rank == 0:
            print(
                f"[epoch {epoch + 1}] train_reward={train_metrics['reward']:.4f} "
                f"train_wait={train_metrics['avg_wait']:.4f} train_expert_acc={train_metrics['expert_acc']:.4f} "
                f"policy_loss={train_metrics['policy_loss']:.4f} value_loss={train_metrics['value_loss']:.4f}"
            )

        eval_metrics = None
        if rank == 0 and val_loader is not None and args.eval_max_samples > 0:
            model.ts_projector.eval()
            model.vision_projector.eval()
            if model.value_head is not None:
                model.value_head.eval()
            with torch.no_grad():
                eval_metrics = evaluate_policy(
                    model=model,
                    loader=val_loader,
                    evaluator=evaluator,
                    prompt_mode=args.prompt_mode,
                    device=device,
                    max_samples=args.eval_max_samples,
                )
            print(
                f"[epoch {epoch + 1}] val_reward={eval_metrics['reward']:.4f} "
                f"val_wait={eval_metrics['avg_wait']:.4f} val_expert_acc={eval_metrics['expert_acc']:.4f} "
                f"val_count={int(eval_metrics['count'])}"
            )

        if rank == 0:
            metrics = {
                "train_reward": train_metrics["reward"],
                "train_avg_wait": train_metrics["avg_wait"],
                "train_expert_acc": train_metrics["expert_acc"],
                "policy_loss": train_metrics["policy_loss"],
                "value_loss": train_metrics["value_loss"],
                "entropy": train_metrics["entropy"],
            }
            if eval_metrics is not None:
                metrics.update(
                    {
                        "val_reward": eval_metrics["reward"],
                        "val_avg_wait": eval_metrics["avg_wait"],
                        "val_expert_acc": eval_metrics["expert_acc"],
                    }
                )

            save_checkpoint(
                output_dir=args.output_dir,
                filename="last.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                args=args,
                metrics=metrics,
            )
            save_checkpoint(
                output_dir=args.output_dir,
                filename=f"epoch_{epoch + 1}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                args=args,
                metrics=metrics,
            )
            monitor = eval_metrics["reward"] if eval_metrics is not None else train_metrics["reward"]
            if monitor > best_reward:
                best_reward = monitor
                save_checkpoint(
                    output_dir=args.output_dir,
                    filename="best.pt",
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    args=args,
                    metrics=metrics,
                )

        if ddp:
            dist.barrier()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
