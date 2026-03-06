import argparse
import math
import os
import random
from contextlib import nullcontext
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

from dataset import TrafficClipJsonlDataset, traffic_clip_collate
from eval_retrieval import evaluate_retrieval
from losses import clip_contrastive_loss
from models import AlignmentModel


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
    total = len(dataset)
    val_size = max(1, int(total * val_ratio))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


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
) -> None:
    setup_seed(args.seed)

    dataset = TrafficClipJsonlDataset(
        path=args.data_path,
        seq_len=args.seq_len,
        movement_keys=None,
        max_samples=args.max_samples,
        shuffle_samples=True,
        shuffle_seed=args.seed,
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
                sampler=RandomSampler(val_set),
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
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        optimizer.zero_grad()
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(train_loader):
            sequences = batch["sequence"].to(device)
            captions = batch["caption"]
            do_sync = ((step + 1) % args.grad_accum_steps == 0)
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
                    loss = loss / max(1, args.grad_accum_steps)
                if args.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

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

            if rank == 0 and (step + 1) % 50 == 0:
                avg_loss = running_loss / max(1, step_count)
                temp_value, logit_scale_value = _current_temp_and_scale()
                lr_value = _current_lr()
                print(
                    f"[epoch {epoch+1}] step {step+1} | loss={avg_loss:.4f} "
                    f"temp={temp_value:.4f} logit_scale={logit_scale_value:.4f} "
                    f"lr={lr_value:.6g}"
                )

        if rank == 0:
            avg_loss = running_loss / max(1, step_count)
            temp_value, logit_scale_value = _current_temp_and_scale()
            lr_value = _current_lr()
            print(
                f"[epoch {epoch+1}] train_loss={avg_loss:.4f} "
                f"temp={temp_value:.4f} logit_scale={logit_scale_value:.4f} "
                f"lr={lr_value:.6g}"
            )
            if val_loader is not None:
                base_model = model.module if hasattr(model, "module") else model
                metrics = evaluate_retrieval(
                    base_model,
                    val_loader,
                    device=device,
                    ks=(1, 5, 10),
                    max_samples=args.eval_max_samples,
                )
                metrics_str = " ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
                print(f"[epoch {epoch+1}] {metrics_str}")

            if (epoch + 1) % 20 == 0:
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
                ckpt_path = os.path.join(output_dir, f"align_epoch_{epoch+1}.pt")
                torch.save(ckpt, ckpt_path)


def main():
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
    parser.add_argument("--warmup_ratio", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant"],
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_max_samples", type=int, default=2000)
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
    for temperature_value in temperatures:
        output_dir = _resolve_output_dir(args.output_dir, temperature_value, multi_run)
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
        _run_training(args, temperature_value, output_dir, ddp, rank, world_size, device)
        if ddp:
            dist.barrier()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
