import argparse
import json
import math
import os
import random
import shlex
import socket
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from data_split import blocked_group_subset
from dataset import TrafficClipJsonlDataset, traffic_clip_collate
from eval_retrieval import evaluate_retrieval
from losses import clip_contrastive_loss
from models import AlignmentModel


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except TypeError:
                reconfigure(line_buffering=True)


def _format_timestamp(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
        handle.write("\n")


def _log(rank: int, message: str) -> None:
    if rank == 0:
        print(message, flush=True)


def _log_block(rank: int, title: str, payload: Any) -> None:
    if rank != 0:
        return
    print(f"\n===== {title} =====", flush=True)
    if isinstance(payload, str):
        print(payload, flush=True)
    else:
        print(_json_text(payload), flush=True)


def _precision_mode(args) -> str:
    if args.fp16:
        return "fp16"
    if args.bf16:
        return "bf16"
    return "fp32"


def _peak_memory_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
    try:
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
    except Exception:
        allocated = 0.0
        reserved = 0.0
    return {
        "peak_allocated_gb": round(float(allocated), 3),
        "peak_reserved_gb": round(float(reserved), 3),
    }


def setup_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_distributed() -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return True, rank, world_size, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def split_dataset(dataset, val_ratio: float, seed: int):
    train_set, val_set = blocked_group_subset(dataset, val_ratio=val_ratio)
    if val_set is None:
        return train_set, Subset(dataset, [])
    return train_set, val_set


def _parse_temperature_sweep(raw: Optional[str], default_value: float) -> list[float]:
    if raw is None:
        return [default_value]
    temps: list[float] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        temps.append(float(chunk))
    if not temps:
        raise ValueError("--temperature_sweep must contain at least one value")
    return temps


def _resolve_output_dir(base_dir: str, temperature_value: float, multi_run: bool) -> str:
    if not multi_run:
        return base_dir
    return os.path.join(base_dir, f"temp_{temperature_value:.4f}")


def _run_training(
    args,
    temperature_value: float,
    output_dir: str,
    ddp: bool,
    rank: int,
    world_size: int,
    device: torch.device,
) -> Optional[dict[str, Any]]:
    setup_seed(args.seed)
    output_path = Path(output_dir)
    metrics_path = output_path / "metrics.jsonl"
    run_config_path = output_path / "run_config.json"
    run_summary_path = output_path / "run_summary.json"
    run_started_at = datetime.now()
    run_started_clock = perf_counter()
    saved_checkpoints: list[str] = []
    last_epoch_record: dict[str, Any] = {}

    dataset = TrafficClipJsonlDataset(
        path=args.data_path,
        seq_len=args.seq_len,
        movement_keys=None,
        max_samples=args.max_samples,
        shuffle_samples=False,
    )
    train_set, val_set = split_dataset(dataset, args.val_ratio, args.seed)

    if ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=traffic_clip_collate,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = None
        if rank == 0:
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=traffic_clip_collate,
                pin_memory=True,
            )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=traffic_clip_collate,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=traffic_clip_collate,
            pin_memory=True,
        )

    amp_dtype = None
    if device.type == "cuda":
        if args.fp16:
            amp_dtype = torch.float16
        elif args.bf16:
            amp_dtype = torch.bfloat16

    model = AlignmentModel(
        model_path=args.model_path,
        seq_len=args.seq_len,
        image_size=args.image_size,
        periodicity=args.periodicity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens,
        prefix_mode=args.prefix_mode,
        init_temperature=temperature_value,
        learnable_logit_scale=args.learnable_logit_scale,
        torch_dtype=amp_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32),
        device=device,
    )

    with torch.no_grad():
        init_logit = math.log(1.0 / max(float(temperature_value), 1e-6))
        model.logit_scale.fill_(init_logit)
        model.logit_scale.requires_grad_(bool(args.learnable_logit_scale))

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum_steps)))
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0.0
    global_step = 0
    ddp_grad_scale = world_size if ddp else 1
    trainable_param_count = sum(int(p.numel()) for p in trainable_params)
    total_param_count = sum(int(p.numel()) for p in model.parameters())
    run_context = {
        "started_at": _format_timestamp(run_started_at),
        "output_dir": str(output_path),
        "command": shlex.join(sys.argv),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.executable,
        "torch_version": torch.__version__,
        "ddp": ddp,
        "rank": rank,
        "world_size": world_size,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "local_rank": os.environ.get("LOCAL_RANK", ""),
        "precision": _precision_mode(args),
        "temperature": float(temperature_value),
        "learnable_logit_scale": bool(args.learnable_logit_scale),
        "contrastive_mode": "distributed_global_batch" if ddp else "local_batch",
        "data_path": list(args.data_path),
        "dataset_size": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "shuffle_before_split": False,
        "batch_size_per_device": int(args.batch_size),
        "effective_global_batch_size": int(args.batch_size * world_size * max(1, args.grad_accum_steps)),
        "contrastive_global_batch_size": int(args.batch_size * world_size),
        "grad_accum_steps": int(args.grad_accum_steps),
        "num_workers": int(args.num_workers),
        "epochs": int(args.epochs),
        "train_batches_per_epoch": len(train_loader),
        "val_batches": len(val_loader) if val_loader is not None else 0,
        "optimizer_steps_per_epoch": steps_per_epoch,
        "total_optimizer_steps": total_steps,
        "warmup_steps": warmup_steps,
        "lr": float(args.lr),
        "min_lr": float(args.min_lr),
        "lr_schedule": str(args.lr_schedule),
        "val_ratio": float(args.val_ratio),
        "max_samples": args.max_samples,
        "eval_max_samples": int(args.eval_max_samples),
        "save_every_epochs": int(args.save_every_epochs),
        "log_every_steps": int(args.log_every_steps),
        "trainable_params": trainable_param_count,
        "total_params": total_param_count,
    }
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        if metrics_path.exists():
            metrics_path.unlink()
        _write_json(run_config_path, run_context)
        _log_block(rank, "Align Run Config", run_context)

    def _lr_scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        if args.lr_schedule == "constant":
            return 1.0
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        if args.lr_schedule == "linear":
            return min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    def _get_logit_scale():
        if not args.learnable_logit_scale:
            return None
        base_model = model.module if hasattr(model, "module") else model
        return base_model.logit_scale.exp().clamp(min=args.logit_scale_min, max=args.logit_scale_max)

    def _current_temp_and_scale():
        if args.learnable_logit_scale:
            with torch.no_grad():
                scale_value = float(_get_logit_scale().detach().item())
                temp_value = 1.0 / max(scale_value, 1e-6)
                return temp_value, scale_value
        scale_value = 1.0 / max(float(temperature_value), 1e-6)
        return float(temperature_value), scale_value

    def _current_lr() -> float:
        if not optimizer.param_groups:
            return 0.0
        return float(optimizer.param_groups[0].get("lr", 0.0))

    for epoch in range(args.epochs):
        epoch_started_at = datetime.now()
        epoch_started_clock = perf_counter()
        if device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(device)
            except Exception:
                pass
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader):
            sequences = batch["sequence"].to(device)
            captions = batch["caption"]
            do_sync = ((step + 1) % args.grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            sync_context = (
                nullcontext()
                if (not ddp or do_sync or args.grad_accum_steps == 1)
                else model.no_sync()
            )
            with sync_context:
                with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                    image_vec, text_vec = model(sequences, captions)
                    logit_scale = _get_logit_scale()
                    loss = clip_contrastive_loss(
                        image_vec,
                        text_vec,
                        temperature=temperature_value,
                        logit_scale=logit_scale,
                    )
                    loss_for_backward = (loss * ddp_grad_scale) / max(1, args.grad_accum_steps)
                if args.fp16:
                    scaler.scale(loss_for_backward).backward()
                else:
                    loss_for_backward.backward()

            if do_sync:
                lr_scale = _lr_scale(global_step)
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * lr_scale
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            running_loss += loss.item()
            step_count += 1

            if rank == 0 and args.log_every_steps > 0 and (step + 1) % args.log_every_steps == 0:
                avg_loss = running_loss / max(1, step_count)
                temp_value, logit_scale_value = _current_temp_and_scale()
                lr_value = _current_lr()
                elapsed_seconds = perf_counter() - epoch_started_clock
                avg_step_seconds = elapsed_seconds / max(1, step_count)
                remaining_steps = max(0, len(train_loader) - (step + 1))
                eta_seconds = remaining_steps * avg_step_seconds
                _log(
                    rank,
                    f"[epoch {epoch+1}/{args.epochs}] step {step+1}/{len(train_loader)} "
                    f"| loss={avg_loss:.4f} "
                    f"temp={temp_value:.4f} logit_scale={logit_scale_value:.4f} "
                    f"lr={lr_value:.6g} "
                    f"elapsed={_format_duration(elapsed_seconds)} "
                    f"eta={_format_duration(eta_seconds)}"
                )

        if rank == 0:
            eval_metrics: dict[str, float] = {}
            eval_duration_seconds = 0.0
            avg_loss = running_loss / max(1, step_count)
            temp_value, logit_scale_value = _current_temp_and_scale()
            lr_value = _current_lr()
            epoch_duration_seconds = perf_counter() - epoch_started_clock
            total_elapsed_seconds = perf_counter() - run_started_clock
            memory_stats = _peak_memory_stats(device)
            _log(
                rank,
                f"[epoch {epoch+1}/{args.epochs}] train_loss={avg_loss:.4f} "
                f"temp={temp_value:.4f} logit_scale={logit_scale_value:.4f} "
                f"lr={lr_value:.6g} "
                f"epoch_time={_format_duration(epoch_duration_seconds)} "
                f"total_elapsed={_format_duration(total_elapsed_seconds)} "
                f"peak_mem={memory_stats['peak_allocated_gb']:.2f}GB/"
                f"{memory_stats['peak_reserved_gb']:.2f}GB"
            )
            if val_loader is not None:
                eval_started_clock = perf_counter()
                base_model = model.module if hasattr(model, "module") else model
                eval_metrics = evaluate_retrieval(
                    base_model,
                    val_loader,
                    device=device,
                    ks=(1, 5, 10),
                    max_samples=args.eval_max_samples,
                )
                eval_duration_seconds = perf_counter() - eval_started_clock
                metrics_str = " ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
                _log(
                    rank,
                    f"[epoch {epoch+1}/{args.epochs}] eval {metrics_str} "
                    f"eval_time={_format_duration(eval_duration_seconds)}"
                )

            checkpoint_path = ""
            checkpoint_duration_seconds = 0.0
            if (epoch + 1) % args.save_every_epochs == 0:
                checkpoint_started_clock = perf_counter()
                base_model = model.module if hasattr(model, "module") else model
                run_args = dict(vars(args))
                run_args["temperature"] = temperature_value
                run_args["output_dir"] = output_dir
                ckpt = {
                    "ts_projector": base_model.ts_projector.state_dict(),
                    "vision_projector": base_model.vision_projector.state_dict(),
                    "logit_scale": base_model.logit_scale.detach().cpu(),
                    "args": run_args,
                }
                checkpoint_path = os.path.join(output_dir, f"align_epoch_{epoch+1}.pt")
                torch.save(ckpt, checkpoint_path)
                checkpoint_duration_seconds = perf_counter() - checkpoint_started_clock
                saved_checkpoints.append(checkpoint_path)
                _log(
                    rank,
                    f"[epoch {epoch+1}/{args.epochs}] checkpoint_saved={checkpoint_path} "
                    f"save_time={_format_duration(checkpoint_duration_seconds)}"
                )

            last_epoch_record = {
                "epoch": epoch + 1,
                "epoch_started_at": _format_timestamp(epoch_started_at),
                "epoch_ended_at": _format_timestamp(datetime.now()),
                "epoch_duration": _format_duration(epoch_duration_seconds),
                "epoch_duration_seconds": round(epoch_duration_seconds, 3),
                "total_elapsed": _format_duration(total_elapsed_seconds),
                "total_elapsed_seconds": round(total_elapsed_seconds, 3),
                "optimizer_step": global_step,
                "train_loss": round(float(avg_loss), 6),
                "lr": round(float(lr_value), 10),
                "temperature": round(float(temp_value), 6),
                "logit_scale": round(float(logit_scale_value), 6),
                "eval_metrics": {key: round(float(value), 6) for key, value in eval_metrics.items()},
                "eval_duration": _format_duration(eval_duration_seconds),
                "eval_duration_seconds": round(eval_duration_seconds, 3),
                "checkpoint_path": checkpoint_path,
                "checkpoint_save_duration": _format_duration(checkpoint_duration_seconds),
                "checkpoint_save_duration_seconds": round(checkpoint_duration_seconds, 3),
                "peak_allocated_gb": memory_stats["peak_allocated_gb"],
                "peak_reserved_gb": memory_stats["peak_reserved_gb"],
            }
            _append_jsonl(metrics_path, last_epoch_record)

    if rank != 0:
        return None

    ended_at = datetime.now()
    duration_seconds = perf_counter() - run_started_clock
    run_summary = {
        "status": "success",
        "started_at": _format_timestamp(run_started_at),
        "ended_at": _format_timestamp(ended_at),
        "duration": _format_duration(duration_seconds),
        "duration_seconds": round(duration_seconds, 3),
        "output_dir": str(output_path),
        "run_config_path": str(run_config_path),
        "metrics_path": str(metrics_path),
        "summary_path": str(run_summary_path),
        "epochs": int(args.epochs),
        "optimizer_steps": int(global_step),
        "temperature": float(temperature_value),
        "learnable_logit_scale": bool(args.learnable_logit_scale),
        "precision": _precision_mode(args),
        "dataset_size": len(dataset),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "saved_checkpoints": saved_checkpoints,
        "last_epoch": last_epoch_record,
    }
    _write_json(run_summary_path, run_summary)
    _log_block(rank, "Align Run Summary", run_summary)
    return run_summary


def main():
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, nargs="+")
    parser.add_argument("--model_path", type=str, default="/home/data/model/qwen-vl-8B/")
    parser.add_argument("--output_dir", type=str, default="records/align")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--periodicity", type=int, default=16)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--prefix_tokens", type=int, default=8)
    parser.add_argument("--prefix_mode", type=str, default="prefix", choices=["single", "prefix"])
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--temperature_sweep", type=str, default=None)
    parser.add_argument("--learnable_logit_scale", action="store_true")
    parser.add_argument("--logit_scale_min", type=float, default=1e-3)
    parser.add_argument("--logit_scale_max", type=float, default=100.0)
    parser.add_argument("--batch_size", type=int, default=16, help="per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=2000)
    parser.add_argument("--log_every_steps", type=int, default=50)
    parser.add_argument("--save_every_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16")
    if args.learnable_logit_scale and args.temperature_sweep:
        raise ValueError("Choose either --learnable_logit_scale or --temperature_sweep")
    if args.logit_scale_min <= 0 or args.logit_scale_max <= 0:
        raise ValueError("--logit_scale_min and --logit_scale_max must be > 0")
    if args.logit_scale_min > args.logit_scale_max:
        raise ValueError("--logit_scale_min must be <= --logit_scale_max")
    if args.min_lr < 0:
        raise ValueError("--min_lr must be >= 0")
    if args.min_lr > args.lr:
        raise ValueError("--min_lr must be <= --lr")
    if args.log_every_steps < 0:
        raise ValueError("--log_every_steps must be >= 0")
    if args.save_every_epochs <= 0:
        raise ValueError("--save_every_epochs must be > 0")

    ddp, rank, world_size, device = init_distributed()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    temperatures = _parse_temperature_sweep(args.temperature_sweep, float(args.temperature))
    multi_run = len(temperatures) > 1
    all_run_summaries: list[dict[str, Any]] = []
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
    for temperature_value in temperatures:
        output_dir = _resolve_output_dir(args.output_dir, temperature_value, multi_run)
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        run_summary = _run_training(args, temperature_value, output_dir, ddp, rank, world_size, device)
        if rank == 0 and run_summary is not None:
            all_run_summaries.append(run_summary)
        if ddp:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if rank == 0 and multi_run:
        sweep_summary_path = Path(args.output_dir) / "temperature_sweep_summary.json"
        sweep_summary = {
            "started_at": all_run_summaries[0]["started_at"] if all_run_summaries else "",
            "ended_at": all_run_summaries[-1]["ended_at"] if all_run_summaries else "",
            "runs": all_run_summaries,
        }
        _write_json(sweep_summary_path, sweep_summary)
        _log_block(rank, "Temperature Sweep Summary", sweep_summary)

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
