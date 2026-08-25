#!/usr/bin/env python3
"""Run one isolated Stage-1 -> Stage-2 -> online-evaluation coreset task."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence


REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/sysu/anaconda3/envs/hnr5090/bin/python")
TORCHRUN = PYTHON.parent / "torchrun"
LLM = "/home/data/model/qwen-vl-8B"


def prune_intermediate_checkpoints(stage1_dir: Path, stage2_dir: Path) -> list[str]:
    """Keep the two best checkpoints needed for reproducibility and evaluation."""
    candidates = [
        *stage1_dir.glob("align_epoch_*.pt"),
        stage1_dir / "last.pt",
        *stage2_dir.glob("epoch_*.pt"),
        stage2_dir / "last.pt",
    ]
    removed: list[str] = []
    for path in candidates:
        if path.is_file() and path.name != "best.pt":
            path.unlink()
            removed.append(str(path))
    return removed


def write_status(path: Path, **updates: Any) -> None:
    previous: dict[str, Any] = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous = {}
    previous.update(updates)
    previous["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")


def run(command: Sequence[str], log_path: Path, env: dict[str, str]) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(
            list(command), cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}: {command[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--stage1_data", required=True)
    parser.add_argument("--stage2_data", required=True)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[2024, 42, 123])
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    stage1_dir = output_dir / "stage1"
    stage2_dir = output_dir / "stage2"
    eval_dir = output_dir / "online"
    status_path = output_dir / "status.json"
    log_path = output_dir / "job.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    if (stage2_dir / "best.pt").is_file() and all(
        (eval_dir / f"seed_{seed}.json").is_file() for seed in args.eval_seeds
    ):
        removed = prune_intermediate_checkpoints(stage1_dir, stage2_dir)
        write_status(status_path, state="complete", skipped=True, pruned_checkpoint_count=len(removed))
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    base_port = 30000 + (int(args.gpu) * 100) + (int(args.seed) % 97)
    write_status(
        status_path,
        state="running",
        task=args.task,
        flow=args.flow,
        gpu=str(args.gpu),
        train_seed=args.seed,
        stage1_data=str(Path(args.stage1_data).resolve()),
        stage2_data=str(Path(args.stage2_data).resolve()),
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    try:
        if not (stage1_dir / "best.pt").is_file():
            stage1_dir.mkdir(parents=True, exist_ok=True)
            write_status(status_path, state="stage1")
            run(
                [
                    str(TORCHRUN), "--standalone", "--nproc_per_node=1", "--master_port", str(base_port),
                    "train_align.py", "--data_path", args.stage1_data, "--model_path", LLM,
                    "--output_dir", str(stage1_dir), "--decision_only", "--seq_len", "16",
                    "--image_size", "224", "--periodicity", "16", "--input_dim", "8",
                    "--hidden_dim", "256", "--prefix_tokens", "32", "--prefix_mode", "prefix",
                    "--temperature", "0.07", "--batch_size", "2", "--grad_accum_steps", "2",
                    "--lr", "1e-4", "--min_lr", "1e-5", "--warmup_ratio", "0.1",
                    "--lr_schedule", "cosine", "--epochs", str(args.epochs), "--num_workers", "4",
                    "--val_ratio", "0.1", "--eval_max_samples", "2000", "--log_every_steps", "50",
                    "--save_every_epochs", "1", "--seed", str(args.seed), "--bf16",
                ],
                log_path,
                env,
            )
        if not (stage1_dir / "best.pt").is_file():
            raise RuntimeError("Stage 1 ended without best.pt")

        if not (stage2_dir / "best.pt").is_file():
            stage2_dir.mkdir(parents=True, exist_ok=True)
            write_status(status_path, state="stage2")
            run(
                [
                    str(TORCHRUN), "--standalone", "--nproc_per_node=1", "--master_port", str(base_port + 1),
                    "projection_train/train_projection_bc.py", "--data_path", args.stage2_data,
                    "--llm_path", LLM, "--align_ckpt", str(stage1_dir / "best.pt"),
                    "--output_dir", str(stage2_dir), "--seq_len", "16", "--image_size", "224",
                    "--periodicity", "16", "--input_dim", "8", "--hidden_dim", "256",
                    "--prefix_tokens", "32", "--prefix_mode", "prefix", "--max_length", "64",
                    "--prompt_mode", "queue_wait_short", "--batch_size", "2", "--grad_accum_steps", "8",
                    "--epochs", str(args.epochs), "--lr", "1e-4", "--min_lr", "1e-5",
                    "--warmup_ratio", "0.1", "--lr_schedule", "cosine", "--weight_decay", "1e-4",
                    "--max_grad_norm", "1.0", "--val_ratio", "0.1", "--split_strategy", "stratified",
                    "--num_workers", "4", "--seed", str(args.seed), "--bf16", "--ranking_loss_weight", "0.5",
                    "--cost_awt_weight", "2.0", "--cost_aql_weight", "1.0", "--best_metric",
                    "val_top1_cost_regret", "--best_mode", "min", "--enable_class_balance",
                    "--overconf_penalty_weight", "0.1", "--save_every", "3", "--checkpoint_slim",
                    "--ddp_timeout_seconds", "1800",
                ],
                log_path,
                env,
            )
        if not (stage2_dir / "best.pt").is_file():
            raise RuntimeError("Stage 2 ended without best.pt")
        # Stage 2 has loaded the Stage-1 best checkpoint; recovery snapshots are no longer needed.
        prune_intermediate_checkpoints(stage1_dir, Path("/nonexistent"))

        eval_dir.mkdir(parents=True, exist_ok=True)
        write_status(status_path, state="online_eval")
        for eval_seed in args.eval_seeds:
            result_path = eval_dir / f"seed_{eval_seed}.json"
            if result_path.is_file():
                continue
            run(
                [
                    str(PYTHON), "projection_train/run_projection_online.py", "--projection_ckpt",
                    str(stage2_dir / "best.pt"), "--llm_path", LLM, "--memo",
                    f"joint_coreset_{args.task}_s{args.seed}_e{eval_seed}", "--dataset", "jinan",
                    "--traffic_file", args.flow, "--run_counts", "3600", "--seed", str(eval_seed),
                    "--prompt_mode", "queue_wait_short", "--seq_len", "16", "--image_size", "224",
                    "--periodicity", "16", "--input_dim", "8", "--hidden_dim", "256",
                    "--prefix_tokens", "32", "--prefix_mode", "prefix", "--max_length", "64",
                    "--min_action_time", "30", "--traffic_series_interval", "2", "--bf16",
                    "--result_json", str(result_path),
                ],
                log_path,
                env,
            )
        removed = prune_intermediate_checkpoints(stage1_dir, stage2_dir)
        write_status(
            status_path,
            state="complete",
            completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            pruned_checkpoint_count=len(removed),
        )
    except Exception as exc:
        write_status(status_path, state="failed", error=str(exc), failed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        raise


if __name__ == "__main__":
    main()
