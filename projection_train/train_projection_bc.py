import argparse
import json
import math
import os
import random
import sys
import time
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Subset
from torch.utils.data.distributed import DistributedSampler

# Ensure repo root is on sys.path so `import models` works when run from subdir.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models import Qwen3VLBackbone, TimeSeriesProjector, VisionPrefixProjector
from projection_train.jsonl_dataset import TrafficPromptSeriesDataset
from projection_train.policy import ACTION_LIST, action_to_index

try:
    import deepspeed
    from deepspeed.runtime.zero import GatheredParameters
except Exception:  # pragma: no cover - optional dependency
    deepspeed = None
    GatheredParameters = None


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_distributed(use_deepspeed: bool = False) -> Tuple[bool, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if use_deepspeed and deepspeed is not None:
            if not dist.is_initialized():
                try:
                    deepspeed.init_distributed(dist_backend=backend)
                except TypeError:
                    deepspeed.init_distributed()
        else:
            if not dist.is_initialized():
                dist.init_process_group(backend=backend, init_method="env://")
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        return True, rank, world_size, device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return False, 0, 1, device


def _resolve_amp_dtype(args, device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return None


def _digit_token_ids(tokenizer, digits: Iterable[int]) -> List[int]:
    token_ids: List[int] = []
    for digit in digits:
        ids = tokenizer(str(digit), add_special_tokens=False).input_ids
        if len(ids) != 1:
            raise ValueError(f"Digit token '{digit}' is not a single token: {ids}")
        token_ids.append(ids[0])
    return token_ids


def _build_prompt(prompt: str, suffix: str) -> str:
    base = (prompt or "").rstrip()
    suffix = (suffix or "").strip()
    if base and suffix:
        return f"{base}\n{suffix}"
    if base:
        return base
    return suffix


def make_collate_fn(prompt_suffix: str):
    def _collate(batch):
        sequences = torch.stack(
            [torch.tensor(sample.sequence, dtype=torch.float32) for sample in batch], dim=0
        )
        prompts = [_build_prompt(sample.prompt, prompt_suffix) for sample in batch]
        labels = torch.tensor(
            [action_to_index(sample.expert_action) for sample in batch], dtype=torch.long
        )
        return {"sequence": sequences, "prompt": prompts, "label": labels}

    return _collate


def _get_lr(optimizer: Optional[torch.optim.Optimizer]) -> float:
    if optimizer is None:
        return 0.0
    for group in optimizer.param_groups:
        return float(group.get("lr", 0.0))
    return 0.0


def _update_class_stats(
    preds: torch.Tensor,
    labels: torch.Tensor,
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
    class_pred_total: torch.Tensor,
    num_classes: int,
) -> None:
    preds_cpu = preds.detach().view(-1).cpu()
    labels_cpu = labels.detach().view(-1).cpu()
    class_total += torch.bincount(labels_cpu, minlength=num_classes)
    class_pred_total += torch.bincount(preds_cpu, minlength=num_classes)
    correct_mask = preds_cpu == labels_cpu
    if correct_mask.any():
        class_correct += torch.bincount(labels_cpu[correct_mask], minlength=num_classes)


def _format_class_accuracy(
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
    action_labels: Sequence[str],
) -> str:
    parts = []
    for idx, label in enumerate(action_labels):
        total = int(class_total[idx].item())
        correct = int(class_correct[idx].item())
        acc = (correct / total) if total > 0 else 0.0
        parts.append(f"{label}:{acc:.3f}({correct}/{total})")
    return " ".join(parts)


def _macro_f1(
    class_correct: torch.Tensor,
    class_total: torch.Tensor,
    class_pred_total: torch.Tensor,
) -> float:
    precision = class_correct / class_pred_total.clamp(min=1)
    recall = class_correct / class_total.clamp(min=1)
    denom = (precision + recall).clamp(min=1e-12)
    f1 = 2 * precision * recall / denom
    return float(f1.mean().item())


class BCProjectionModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        align_ckpt: str,
        seq_len: int,
        image_size: int,
        periodicity: int,
        input_dim: int,
        hidden_dim: int,
        prefix_tokens: int,
        prefix_mode: str,
        max_length: int,
        torch_dtype: torch.dtype,
        device: torch.device,
        fsdp_enabled: bool = False,
        fsdp_sharding_strategy: str = "full_shard",
    ) -> None:
        super().__init__()
        self.device = device
        self.backbone = Qwen3VLBackbone(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device=device,
            max_length=max_length,
            fsdp_enabled=fsdp_enabled,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
        )
        if self.backbone.tokenizer is not None:
            self.backbone.tokenizer.padding_side = "left"

        self.ts_projector = TimeSeriesProjector(
            seq_len=seq_len,
            image_size=image_size,
            periodicity=periodicity,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=device,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=device)
            vision_tokens = self.backbone.encode_vision_tokens(dummy)
            vision_dim = int(vision_tokens.shape[-1])

        self.vision_projector = VisionPrefixProjector(
            vision_dim=vision_dim,
            text_dim=self.backbone.text_dim,
            prefix_tokens=prefix_tokens,
            mode=prefix_mode,
        )
        proj_dtype = getattr(self.backbone.model, "dtype", torch_dtype)
        self.vision_projector.to(device=device, dtype=proj_dtype)
        self.ts_projector.to(device=device)

        if align_ckpt:
            self._load_align_ckpt(align_ckpt)

        self._freeze_modules()
        self.to(device)

    def _load_align_ckpt(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ts_state = ckpt.get("ts_projector", None) if isinstance(ckpt, dict) else None
        if ts_state is None:
            ts_state = ckpt
        try:
            self.ts_projector.load_state_dict(ts_state, strict=True)
        except RuntimeError:
            if hasattr(self.ts_projector, "image_module") and isinstance(
                self.ts_projector.image_module, nn.Module
            ):
                self.ts_projector.image_module.load_state_dict(ts_state, strict=True)
            else:
                raise

        if isinstance(ckpt, dict) and "vision_projector" in ckpt:
            vision_state = ckpt["vision_projector"]
            if not isinstance(vision_state, dict) or "proj.weight" not in vision_state:
                self.vision_projector.load_state_dict(vision_state, strict=True)
                return

            ckpt_weight = vision_state["proj.weight"]
            current_weight = self.vision_projector.proj.weight
            if tuple(ckpt_weight.shape) == tuple(current_weight.shape):
                self.vision_projector.load_state_dict(vision_state, strict=True)
                return

            ckpt_vision_dim = ckpt_weight.shape[1]
            current_vision_dim = current_weight.shape[1]
            if ckpt_vision_dim != current_vision_dim:
                msg = [
                    "Align checkpoint 与当前模型的 VisionPrefixProjector 维度不一致（vision_dim 不匹配）。",
                    f"ckpt proj.weight={tuple(ckpt_weight.shape)}",
                    f"current proj.weight={tuple(current_weight.shape)}",
                    "请使用与当前 Qwen3-VL 视觉输出维度一致的配置重新跑 train_align.py。",
                ]
                raise ValueError("\n".join(msg))

            if self.vision_projector.mode == "single":
                msg = [
                    "Align checkpoint 与当前模型的 VisionPrefixProjector 维度不一致（single 模式不支持自动适配）。",
                    f"ckpt proj.weight={tuple(ckpt_weight.shape)}",
                    f"current proj.weight={tuple(current_weight.shape)}",
                    "请使用一致的配置重新跑 train_align.py。",
                ]
                raise ValueError("\n".join(msg))

            text_dim = int(self.backbone.text_dim)
            ckpt_out_dim = ckpt_weight.shape[0]
            if ckpt_out_dim % text_dim != 0:
                msg = [
                    "Align checkpoint 与当前模型的 VisionPrefixProjector 维度不一致（text_dim 不匹配）。",
                    f"ckpt proj.weight={tuple(ckpt_weight.shape)}",
                    f"current proj.weight={tuple(current_weight.shape)}",
                    "请使用与当前 Qwen3-VL text_dim 一致的配置重新跑 train_align.py。",
                ]
                raise ValueError("\n".join(msg))

            ckpt_prefix_tokens = ckpt_out_dim // text_dim
            current_prefix_tokens = int(self.vision_projector.prefix_tokens)
            tokens_to_copy = min(ckpt_prefix_tokens, current_prefix_tokens)

            with torch.no_grad():
                ckpt_w = ckpt_weight.view(ckpt_prefix_tokens, text_dim, ckpt_vision_dim)
                cur_w = current_weight.view(current_prefix_tokens, text_dim, current_vision_dim)
                cur_w[:tokens_to_copy].copy_(ckpt_w[:tokens_to_copy])
                if "proj.bias" in vision_state and self.vision_projector.proj.bias is not None:
                    ckpt_bias = vision_state["proj.bias"]
                    cur_bias = self.vision_projector.proj.bias
                    if ckpt_bias.numel() == ckpt_prefix_tokens * text_dim:
                        ckpt_b = ckpt_bias.view(ckpt_prefix_tokens, text_dim)
                        cur_b = cur_bias.view(current_prefix_tokens, text_dim)
                        cur_b[:tokens_to_copy].copy_(ckpt_b[:tokens_to_copy])

            filtered_state = dict(vision_state)
            filtered_state.pop("proj.weight", None)
            filtered_state.pop("proj.bias", None)
            if filtered_state:
                self.vision_projector.load_state_dict(filtered_state, strict=False)

            rank = int(os.environ.get("RANK", "0"))
            if rank == 0:
                print(
                    "Warning: align_ckpt 的 prefix_tokens 与当前模型不一致，已执行截断/填充适配。"
                )

    def _freeze_modules(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.vision_projector.parameters():
            p.requires_grad = False
        for p in self.ts_projector.parameters():
            p.requires_grad = True
        self.backbone.model.eval()
        self.vision_projector.eval()

    def forward(self, sequences: torch.Tensor, prompts: List[str], digit_token_ids: List[int]) -> torch.Tensor:
        # Keep the time-series -> image path in float32 to avoid bf16 conv1d issues.
        if sequences.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                images = self.ts_projector(sequences.float())
        else:
            images = self.ts_projector(sequences.float())
        vision_tokens = self.backbone.encode_vision_tokens(images)
        if vision_tokens.dim() == 2:
            vision_tokens = vision_tokens.unsqueeze(0)
        vision_pooled = vision_tokens.mean(dim=1)
        _, prefix_tokens = self.vision_projector(vision_pooled)

        encoding = self.backbone.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.backbone.max_length,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            text_embeds = self.backbone.text_embed(input_ids)
        if text_embeds.dtype != prefix_tokens.dtype:
            text_embeds = text_embeds.to(dtype=prefix_tokens.dtype)

        inputs_embeds = torch.cat([prefix_tokens, text_embeds], dim=1)
        prefix_mask = torch.ones(
            (attention_mask.size(0), prefix_tokens.size(1)),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        outputs = self.backbone.model(inputs_embeds=inputs_embeds, attention_mask=full_mask)
        logits = outputs.logits
        last_index = full_mask.sum(dim=1) - 1
        batch_idx = torch.arange(logits.size(0), device=logits.device)
        logits_last = logits[batch_idx, last_index]

        digit_ids = torch.tensor(digit_token_ids, device=logits.device)
        action_logits = logits_last.index_select(dim=-1, index=digit_ids)
        return action_logits


def _unwrap_model(model: nn.Module) -> BCProjectionModel:
    return model.module if hasattr(model, "module") else model


def _load_deepspeed_config(path: str, args: argparse.Namespace, world_size: int) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    micro_bs = int(args.batch_size)
    grad_accum = int(cfg.get("gradient_accumulation_steps", 1))
    cfg["train_micro_batch_size_per_gpu"] = micro_bs
    cfg["gradient_accumulation_steps"] = grad_accum
    cfg["train_batch_size"] = micro_bs * grad_accum * max(1, int(world_size))
    if "bf16" not in cfg and args.bf16:
        cfg["bf16"] = {"enabled": True}
    if "fp16" not in cfg and args.fp16:
        cfg["fp16"] = {"enabled": True}
    return cfg


def _gather_module_state(module: nn.Module) -> Optional[dict]:
    if GatheredParameters is None or not dist.is_initialized():
        return module.state_dict()
    params = [p for p in module.parameters()]
    if not params:
        return module.state_dict()
    with GatheredParameters(params, modifier_rank=0):
        if dist.get_rank() == 0:
            return module.state_dict()
    return None


def _save_deepspeed_checkpoint(
    output_dir: str,
    epoch: int,
    engine: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    args: argparse.Namespace,
) -> Optional[str]:
    tag = f"bc_epoch_{epoch}"
    engine.save_checkpoint(output_dir, tag=tag)
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    base_model = _unwrap_model(engine)
    ts_state = _gather_module_state(base_model.ts_projector)
    vision_state = _gather_module_state(base_model.vision_projector)
    if ts_state is None or vision_state is None:
        return None
    ckpt = {
        "ts_projector": ts_state,
        "vision_projector": vision_state,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": epoch,
        "args": vars(args),
        "deepspeed_tag": tag,
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"bc_epoch_{epoch}.pt")
    torch.save(ckpt, path)
    return path


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    digit_token_ids: List[int],
    num_classes: int,
    action_labels: Sequence[str],
    amp_dtype: Optional[torch.dtype] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    log_interval: int = 20,
    use_deepspeed: bool = False,
    is_main: bool = True,
) -> Tuple[float, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = _unwrap_model(model)
    base_model.ts_projector.train()
    base_model.backbone.model.eval()
    base_model.vision_projector.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_pred_total = torch.zeros(num_classes, dtype=torch.long)
    total_steps = len(loader)
    start_time = time.time()

    for step, batch in enumerate(loader):
        sequences = batch["sequence"].to(device)
        labels = batch["label"].to(device)
        prompts = batch["prompt"]

        if use_deepspeed:
            try:
                model.zero_grad(set_to_none=True)
            except TypeError:
                model.zero_grad()
        else:
            optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
            logits = model(sequences, prompts, digit_token_ids)
            loss = F.cross_entropy(logits, labels)

        if use_deepspeed:
            model.backward(loss)
            model.step()
        else:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            _update_class_stats(
                preds, labels, class_correct, class_total, class_pred_total, num_classes
            )

        if is_main and log_interval > 0 and (step + 1) % log_interval == 0:
            avg_loss = total_loss / max(1, total_count)
            avg_acc = total_correct / max(1, total_count)
            elapsed = max(1e-6, time.time() - start_time)
            sps = total_count / elapsed
            steps_per_sec = (step + 1) / elapsed
            eta_sec = (total_steps - (step + 1)) / max(1e-6, steps_per_sec)
            lr = _get_lr(optimizer)
            print(
                f"[train step {step+1}/{total_steps}] "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} "
                f"lr={lr:.6g} sps={sps:.1f} eta={eta_sec/60:.1f}m"
            )

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    macro_f1 = _macro_f1(class_correct, class_total, class_pred_total)
    if is_main and log_interval > 0 and total_count > 0:
        per_class = _format_class_accuracy(class_correct, class_total, action_labels)
        print(f"[train epoch] per_class_acc {per_class}")
    return avg_loss, avg_acc, macro_f1, class_correct, class_total, class_pred_total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    digit_token_ids: List[int],
    num_classes: int,
    action_labels: Sequence[str],
    amp_dtype: Optional[torch.dtype] = None,
    is_main: bool = True,
) -> Tuple[float, float, float, torch.Tensor, torch.Tensor, torch.Tensor]:
    base_model = _unwrap_model(model)
    base_model.ts_projector.eval()
    base_model.backbone.model.eval()
    base_model.vision_projector.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    class_correct = torch.zeros(num_classes, dtype=torch.long)
    class_total = torch.zeros(num_classes, dtype=torch.long)
    class_pred_total = torch.zeros(num_classes, dtype=torch.long)

    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequence"].to(device)
            labels = batch["label"].to(device)
            prompts = batch["prompt"]

            with torch.cuda.amp.autocast(enabled=amp_dtype is not None, dtype=amp_dtype):
                logits = model(sequences, prompts, digit_token_ids)
                loss = F.cross_entropy(logits, labels)

            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            _update_class_stats(
                preds, labels, class_correct, class_total, class_pred_total, num_classes
            )

    avg_loss = total_loss / max(1, total_count)
    avg_acc = total_correct / max(1, total_count)
    macro_f1 = _macro_f1(class_correct, class_total, class_pred_total)
    if is_main and total_count > 0:
        per_class = _format_class_accuracy(class_correct, class_total, action_labels)
        print(f"[val epoch] per_class_acc {per_class}")
    return avg_loss, avg_acc, macro_f1, class_correct, class_total, class_pred_total


def _save_checkpoint(
    output_dir: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> str:
    base_model = _unwrap_model(model)
    os.makedirs(output_dir, exist_ok=True)
    ckpt = {
        "ts_projector": base_model.ts_projector.state_dict(),
        "vision_projector": base_model.vision_projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
    }
    path = os.path.join(output_dir, f"bc_epoch_{epoch}.pt")
    torch.save(ckpt, path)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, nargs="+")
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--align_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="records/projection_bc")
    parser.add_argument("--intersection_id", type=str, default="intersection_1_1")
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--periodicity", type=int, default=24)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--prefix_tokens", type=int, default=8)
    parser.add_argument("--prefix_mode", type=str, default="prefix", choices=["single", "prefix"])
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="Only output the phase id (0-3). Phase:",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default=None,
        help="DeepSpeed 配置文件路径（启用 ZeRO 分片）。",
    )
    parser.add_argument(
        "--llm_fsdp",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="是否启用 FSDP 分片 LLM 权重：auto=多卡分布式时启用。",
    )
    parser.add_argument(
        "--fsdp_sharding_strategy",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op", "no_shard"],
        help="FSDP 分片策略。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_deepspeed = args.deepspeed_config is not None
    if use_deepspeed and deepspeed is None:
        raise RuntimeError("DeepSpeed not installed but --deepspeed_config was provided.")
    ddp, rank, world_size, device = _init_distributed(use_deepspeed=use_deepspeed)
    seed_everything(args.seed + rank)
    amp_dtype = _resolve_amp_dtype(args, device)
    scaler = None
    if not use_deepspeed and amp_dtype == torch.float16 and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()

    dataset = TrafficPromptSeriesDataset(
        path=args.data_path,
        intersection_id=args.intersection_id,
        seq_len=args.seq_len,
        require_expert_action=True,
        valid_actions=ACTION_LIST,
    )
    if len(dataset) == 0:
        raise RuntimeError("No samples found with expert_action.")

    total = len(dataset)
    val_size = int(total * args.val_ratio)
    if total > 1:
        val_size = max(1, val_size)
        val_size = min(val_size, total - 1)
    else:
        val_size = 0
    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(total, generator=generator).tolist()
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices) if val_size > 0 else None

    collate_fn = make_collate_fn(args.prompt_suffix)

    if ddp:
        train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = None
        if val_set is not None and rank == 0:
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                sampler=RandomSampler(val_set),
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            sampler=RandomSampler(train_set),
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )
        val_loader = None
        if val_set is not None:
            val_loader = DataLoader(
                val_set,
                batch_size=args.batch_size,
                sampler=RandomSampler(val_set),
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )

    torch_dtype = amp_dtype or (torch.bfloat16 if device.type == "cuda" else torch.float32)
    if use_deepspeed and args.llm_fsdp != "off":
        raise RuntimeError("FSDP and DeepSpeed cannot be enabled at the same time.")
    llm_fsdp_enabled = False
    if not use_deepspeed:
        llm_fsdp_enabled = (
            args.llm_fsdp == "on"
            or (
                args.llm_fsdp == "auto"
                and ddp
                and world_size > 1
                and device.type == "cuda"
            )
        )
    model = BCProjectionModel(
        model_path=args.llm_path,
        align_ckpt=args.align_ckpt,
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
        fsdp_enabled=llm_fsdp_enabled,
        fsdp_sharding_strategy=args.fsdp_sharding_strategy,
    )

    digit_ids = _digit_token_ids(_unwrap_model(model).backbone.tokenizer, range(len(ACTION_LIST)))
    action_labels = [str(a) for a in ACTION_LIST]

    if ddp and not use_deepspeed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    ds_config = None
    if use_deepspeed:
        ds_config = _load_deepspeed_config(args.deepspeed_config, args, world_size)
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            model_parameters=trainable_params,
            config_params=ds_config,
        )

    if rank == 0:
        print(
            f"[data] total={total} train={len(train_set)} val={len(val_set) if val_set is not None else 0}"
        )
        if total > 0:
            actual_ratio = (len(val_set) / total) if val_set is not None else 0.0
            print(f"[data] val_ratio={actual_ratio:.3f}")
        print(f"[train] device={device} amp_dtype={amp_dtype} ddp={ddp} world_size={world_size}")
        print(f"[train] trainable_params={sum(p.numel() for p in trainable_params)}")
        print(
            f"[train] llm_fsdp={llm_fsdp_enabled} sharding={args.fsdp_sharding_strategy}"
        )
        print(f"[train] deepspeed={use_deepspeed}")

    for epoch in range(1, args.epochs + 1):
        if ddp:
            train_loader.sampler.set_epoch(epoch)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.time()
        train_loss, train_acc, train_f1, _, _, _ = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            digit_ids,
            len(ACTION_LIST),
            action_labels,
            amp_dtype=amp_dtype,
            scaler=scaler,
            log_interval=args.log_interval,
            use_deepspeed=use_deepspeed,
            is_main=(rank == 0),
        )
        if rank == 0:
            epoch_time = time.time() - epoch_start
            lr = _get_lr(optimizer)
            mem = ""
            if device.type == "cuda":
                mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                mem = f" max_mem={mem_gb:.2f}GB"
            print(
                f"[epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"train_f1={train_f1:.4f} lr={lr:.6g} time={epoch_time:.1f}s{mem}"
            )

        if val_loader is not None and rank == 0:
            val_loss, val_acc, val_f1, _, _, _ = evaluate(
                model,
                val_loader,
                device,
                digit_ids,
                len(ACTION_LIST),
                action_labels,
                amp_dtype=amp_dtype,
                is_main=True,
            )
            print(
                f"[epoch {epoch}] val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"val_f1={val_f1:.4f}"
            )

        if args.save_every > 0 and epoch % args.save_every == 0:
            if use_deepspeed:
                ckpt_path = _save_deepspeed_checkpoint(args.output_dir, epoch, model, optimizer, args)
                if rank == 0 and ckpt_path is not None:
                    print(f"[ckpt] saved {ckpt_path}")
            elif rank == 0:
                ckpt_path = _save_checkpoint(args.output_dir, epoch, model, optimizer, args)
                print(f"[ckpt] saved {ckpt_path}")

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
