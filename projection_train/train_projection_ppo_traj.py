from __future__ import annotations

import argparse
import math
import os
import random
import sys
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.rollout_buffer import TrajectoryRolloutBuffer
from projection_train.snapshot_seed_dataset import SnapshotSeedDataset
from projection_train.trajectory_env import TrajectoryCityFlowEnv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
        "seq_len": _safe_positive_int(args.seq_len, ckpt_args.get("seq_len", 16)),
        "image_size": _safe_positive_int(args.image_size, ckpt_args.get("image_size", 224)),
        "periodicity": _safe_positive_int(args.periodicity, ckpt_args.get("periodicity", 16)),
        "input_dim": _safe_positive_int(args.input_dim, ckpt_args.get("input_dim", 8)),
        "hidden_dim": _safe_positive_int(args.hidden_dim, ckpt_args.get("hidden_dim", 64)),
        "prefix_tokens": _safe_positive_int(args.prefix_tokens, ckpt_args.get("prefix_tokens", 16)),
        "prefix_mode": str(args.prefix_mode or ckpt_args.get("prefix_mode") or "prefix"),
        "max_length": _safe_positive_int(args.max_length, ckpt_args.get("max_length", 128)),
    }
    return runtime


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
) -> tuple[TrajectoryRolloutBuffer, Dict[str, float]]:
    buffer = TrajectoryRolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
    obs = env.observe()
    episode_reward = 0.0
    queue_trace: List[float] = []
    wait_trace: List[float] = []

    for _ in range(int(rollout_steps)):
        seq_tensor = torch.tensor(obs["sequences"], dtype=torch.float32, device=device)
        prompts = obs["prompts"]
        with torch.no_grad():
            act_out = policy.act(seq_tensor, prompts, sample=sample_actions)
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
        )
        episode_reward += float(sum(rewards))
        queue_trace.append(float(info.get("queue_sum", 0.0)))
        wait_trace.append(float(info.get("waiting_time_mean", 0.0)))
        obs = next_obs
        if done:
            break

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
        "rollout_steps": float(buffer.num_steps),
        "rollout_transitions": float(buffer.num_transitions),
    }
    return buffer, metrics


def ppo_update(
    policy: PrefixActionPolicy,
    reference_policy: PrefixActionPolicy,
    optimizer: torch.optim.Optimizer,
    buffer: TrajectoryRolloutBuffer,
    device: torch.device,
    args: argparse.Namespace,
    amp_dtype: torch.dtype | None,
    scaler: torch.cuda.amp.GradScaler | None,
) -> Dict[str, float]:
    set_policy_mode(policy, is_train=True)
    set_policy_mode(reference_policy, is_train=False)

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
            return torch.cuda.amp.autocast(dtype=amp_dtype)
        return nullcontext()

    for _ in range(int(args.ppo_epochs)):
        for batch in buffer.mini_batches(args.mini_batch_size, shuffle=True):
            sequences = batch["sequences"].to(device)
            prompts = batch["prompts"]
            actions = batch["actions"].to(device)
            old_log_probs = batch["old_log_probs"].to(device)
            old_values = batch["old_values"].to(device)
            advantages = batch["advantages"].to(device)
            returns = batch["returns"].to(device)

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

                with torch.no_grad():
                    reference_logits = reference_policy(sequences, prompts).float()
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
                torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
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
        "eval_steps": float(step_count),
    }


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
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--value_hidden_dim", type=int, default=512)
    parser.add_argument("--prompt_mode", type=str, default="image_only_short")
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--traffic_series_interval", type=int, default=2)
    parser.add_argument("--window_s", type=int, default=30)
    parser.add_argument("--num_updates", type=int, default=100)
    parser.add_argument("--rollout_steps", type=int, default=8)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--mini_batch_size", type=int, default=32)
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
    parser.add_argument("--queue_reward_coef", type=float, default=-0.25)
    parser.add_argument("--pressure_reward_coef", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")
    if not (0.0 <= args.snapshot_reset_prob <= 1.0):
        raise ValueError("--snapshot_reset_prob must be within [0, 1].")

    roadnet_path, traffic_root, run_counts = build_env_paths(args)
    runtime = resolve_runtime(args)
    runtime["value_hidden_dim"] = int(args.value_hidden_dim)

    device = resolve_device(args.device)
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

    seed_everything(args.seed)

    reward_info = {
        "queue_length": float(args.queue_reward_coef),
        "pressure": float(args.pressure_reward_coef),
    }

    policy = build_policy(runtime, device, model_dtype, with_value_head=True)
    policy.load_checkpoint(args.stage2_ckpt, load_value_head=False)
    reference_policy = build_policy(runtime, device, model_dtype, with_value_head=False)
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
        work_root=os.path.join(args.work_root, "train"),
        default_traffic_file=args.traffic_file,
        seq_len=runtime["seq_len"],
        movement_keys=["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"],
        prompt_mode=args.prompt_mode,
        traffic_series_interval=args.traffic_series_interval,
        window_s=args.window_s,
        min_action_time=args.min_action_time,
        reward_alpha=args.reward_alpha,
        switch_penalty_beta=args.switch_penalty_beta,
        reward_info=reward_info,
        run_counts=run_counts,
    )
    eval_env = TrajectoryCityFlowEnv(
        roadnet_path=roadnet_path,
        traffic_root=traffic_root,
        work_root=os.path.join(args.work_root, "eval"),
        default_traffic_file=args.eval_traffic_file or args.traffic_file,
        seq_len=runtime["seq_len"],
        movement_keys=["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"],
        prompt_mode=args.prompt_mode,
        traffic_series_interval=args.traffic_series_interval,
        window_s=args.window_s,
        min_action_time=args.min_action_time,
        reward_alpha=args.reward_alpha,
        switch_penalty_beta=args.switch_penalty_beta,
        reward_info=reward_info,
        run_counts=run_counts,
    )

    snapshot_dataset = None
    if args.snapshot_jsonl:
        snapshot_dataset = SnapshotSeedDataset(
            path=args.snapshot_jsonl,
            seq_len=runtime["seq_len"],
            movement_keys=["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"],
            max_snapshots=args.snapshot_max_samples,
            min_intersections=1,
        )
        if len(snapshot_dataset) == 0:
            raise RuntimeError("snapshot_jsonl was provided but no valid snapshot seeds were loaded.")
        print(f"Loaded snapshot seeds: {len(snapshot_dataset)}")

    os.makedirs(args.output_dir, exist_ok=True)
    best_metric = float("-inf")
    eval_traffic_file = args.eval_traffic_file or args.traffic_file
    rng = random.Random(args.seed)

    for update_idx in range(1, args.num_updates + 1):
        use_snapshot = snapshot_dataset is not None and rng.random() < args.snapshot_reset_prob
        if use_snapshot:
            seed_sample = snapshot_dataset.sample(rng=rng, traffic_file=args.snapshot_traffic_file)
            train_env.reset_from_snapshot(seed_sample)
        else:
            train_env.reset(args.traffic_file)

        rollout_buffer, rollout_metrics = rollout_once(
            policy=policy,
            env=train_env,
            device=device,
            rollout_steps=args.rollout_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            sample_actions=True,
        )
        update_metrics = ppo_update(
            policy=policy,
            reference_policy=reference_policy,
            optimizer=optimizer,
            buffer=rollout_buffer,
            device=device,
            args=args,
            amp_dtype=amp_dtype,
            scaler=scaler if amp_dtype == torch.float16 else None,
        )

        merged_metrics = {**rollout_metrics, **update_metrics}
        print(
            f"[update {update_idx}/{args.num_updates}] "
            f"mode={train_env.last_reset_mode} reward={merged_metrics['rollout_reward']:.4f} "
            f"queue={merged_metrics['rollout_avg_queue']:.4f} wait={merged_metrics['rollout_avg_waiting']:.4f} "
            f"policy={merged_metrics['policy_loss']:.4f} value={merged_metrics['value_loss']:.4f} "
            f"entropy={merged_metrics['entropy']:.4f} kl={merged_metrics['kl']:.4f}"
        )

        if update_idx % args.save_every == 0 or update_idx == args.num_updates:
            save_checkpoint(
                output_dir=args.output_dir,
                filename="last.pt",
                policy=policy,
                optimizer=optimizer,
                update_idx=update_idx,
                args=args,
                metrics=merged_metrics,
            )

        if update_idx % args.eval_every == 0 or update_idx == args.num_updates:
            eval_metrics = evaluate_full_episode(
                policy=policy,
                env=eval_env,
                device=device,
                traffic_file=eval_traffic_file,
            )
            print(
                f"[eval {update_idx}] reward={eval_metrics['eval_reward']:.4f} "
                f"avg_queue={eval_metrics['eval_avg_queue']:.4f} avg_wait={eval_metrics['eval_avg_waiting']:.4f}"
            )
            merged_metrics.update(eval_metrics)
            save_checkpoint(
                output_dir=args.output_dir,
                filename="last.pt",
                policy=policy,
                optimizer=optimizer,
                update_idx=update_idx,
                args=args,
                metrics=merged_metrics,
            )
            if eval_metrics["eval_reward"] > best_metric:
                best_metric = eval_metrics["eval_reward"]
                save_checkpoint(
                    output_dir=args.output_dir,
                    filename="best.pt",
                    policy=policy,
                    optimizer=optimizer,
                    update_idx=update_idx,
                    args=args,
                    metrics=merged_metrics,
                )


if __name__ == "__main__":
    main()

