import math
from typing import Any, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoProcessor, AutoTokenizer

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover - optional dependency
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - optional dependency
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForCausalLM
except Exception:  # pragma: no cover - optional dependency
    AutoModelForCausalLM = None

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
except Exception:  # pragma: no cover - optional dependency
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None

from src.TimeVLM.ts_image_adapter import TimeSeriesImageAdapter


def _normalize_pretrained_model_path(model_path: str) -> str:
    text = str(model_path or "").strip()
    if not text:
        raise ValueError("model_path must not be empty.")
    trimmed = text.rstrip("/\\")
    return trimmed or text


class TimeSeriesProjector(nn.Module):
    """Learnable time-series -> image adapter."""

    def __init__(
        self,
        seq_len: int = 15,
        image_size: int = 224,
        periodicity: int = 30,
        input_dim: int = 8,
        hidden_dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.adapter = TimeSeriesImageAdapter(
            image_size=image_size,
            seq_len=seq_len,
            periodicity=periodicity,
            learnable_image=True,
            three_channel_image=True,
            save_images=False,
            device=device,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )
        # Register the learnable module so parameters are tracked.
        self.image_module = self.adapter.learnable_image_module

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        # sequences: [B, L, D]
        return self.adapter.generate(sequences)


class VisionPrefixProjector(nn.Module):
    """Small linear projection from vision pooled features to text space."""

    def __init__(self, vision_dim: int, text_dim: int, prefix_tokens: int = 8, mode: str = "prefix") -> None:
        super().__init__()
        self.mode = mode
        self.prefix_tokens = int(prefix_tokens)
        if mode == "single":
            self.proj = nn.Linear(vision_dim, text_dim)
        else:
            self.proj = nn.Linear(vision_dim, self.prefix_tokens * text_dim)

    def forward(self, vision_pooled: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if vision_pooled.dtype != self.proj.weight.dtype:
            vision_pooled = vision_pooled.to(dtype=self.proj.weight.dtype)
        if self.mode == "single":
            vec = self.proj(vision_pooled)
            tokens = vec.unsqueeze(1)
            return vec, tokens
        out = self.proj(vision_pooled)
        bsz = out.shape[0]
        tokens = out.view(bsz, self.prefix_tokens, -1)
        vec = tokens.mean(dim=1)
        return vec, tokens


class Qwen3VLBackbone(nn.Module):
    """Frozen Qwen3-VL backbone for vision + text embedding extraction."""

    def __init__(
        self,
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
        max_length: int = 128,
        fsdp_enabled: bool = False,
        fsdp_sharding_strategy: str = "full_shard",
    ) -> None:
        super().__init__()
        model_path = _normalize_pretrained_model_path(model_path)
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.max_length = int(max_length)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_cls = self._select_model_class()
        if model_cls is None:
            raise RuntimeError("No compatible model class found for Qwen3-VL.")
        self.model = model_cls.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.is_fsdp = False
        if fsdp_enabled:
            self._maybe_wrap_fsdp(torch_dtype=torch_dtype, sharding=fsdp_sharding_strategy)
        # Ensure parameters/buffers are on the right device (FSDP shards or full params).
        self.model.to(self.device)
        self.model.eval()
        model_for_freeze = self.model.module if self.is_fsdp and hasattr(self.model, "module") else self.model
        for p in model_for_freeze.parameters():
            p.requires_grad = False

        model_for_setup = self.model.module if self.is_fsdp and hasattr(self.model, "module") else self.model
        self.text_embed = model_for_setup.get_input_embeddings()
        text_embed_mod = self.text_embed
        if FSDP is not None and isinstance(self.text_embed, FSDP) and hasattr(self.text_embed, "module"):
            text_embed_mod = self.text_embed.module
        self.text_dim = int(text_embed_mod.embedding_dim)
        self.vision_tower = self._find_vision_tower(model_for_setup)
        if self.vision_tower is None:
            raise RuntimeError("Failed to locate vision tower in Qwen3-VL model.")

    def _select_model_class(self):
        if Qwen3VLForConditionalGeneration is not None:
            return Qwen3VLForConditionalGeneration
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText
        return AutoModelForCausalLM

    @staticmethod
    def _find_vision_tower(model: nn.Module) -> Optional[nn.Module]:
        # Try common attributes / accessors.
        if hasattr(model, "get_vision_tower"):
            try:
                vt = model.get_vision_tower()
                if isinstance(vt, (list, tuple)):
                    return vt[0] if vt else None
                if isinstance(vt, nn.Module):
                    return vt
            except Exception:
                pass
        for attr in (
            "vision_tower",
            "vision_encoder",
            "vision_model",
            "visual_encoder",
            "visual_model",
            "vision",
            "visual",
        ):
            if hasattr(model, attr):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module):
                    return mod
        # Try nested model.
        inner = getattr(model, "model", None)
        if isinstance(inner, nn.Module):
            return Qwen3VLBackbone._find_vision_tower(inner)
        return None

    @staticmethod
    def _collect_transformer_block_classes(model: nn.Module) -> Set[type]:
        cls_set: Set[type] = set()
        block_name_keywords = (
            "decoderlayer",
            "decoder_layer",
            "transformerblock",
            "transformer_block",
            "qwen2vldecoderlayer",
            "qwen3vldecoderlayer",
            "llamalayer",
            "mistrallayer",
            "block",
        )
        for module in model.modules():
            cls = module.__class__
            name = cls.__name__.lower()
            if any(key in name for key in block_name_keywords):
                if len(list(module.children())) == 0:
                    continue
                cls_set.add(cls)
        return cls_set

    def _maybe_wrap_fsdp(self, torch_dtype: torch.dtype, sharding: str) -> None:
        if FSDP is None or ShardingStrategy is None:
            print("Warning: torch FSDP unavailable, fallback to non-FSDP loading.")
            return
        if not (dist.is_available() and dist.is_initialized()):
            print("Warning: FSDP requested but torch.distributed not initialized, skip FSDP.")
            return
        if self.device.type != "cuda":
            print("Warning: FSDP requested on non-CUDA device, skip FSDP.")
            return

        strategy_map = {
            "full_shard": ShardingStrategy.FULL_SHARD,
            "shard_grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "no_shard": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = strategy_map.get(str(sharding).lower(), ShardingStrategy.FULL_SHARD)

        mp_policy = None
        if MixedPrecision is not None:
            mp_policy = MixedPrecision(
                param_dtype=torch_dtype,
                reduce_dtype=torch_dtype,
                buffer_dtype=torch_dtype,
            )

        block_classes = self._collect_transformer_block_classes(self.model)
        use_block_wrap = bool(block_classes)
        min_num_params = 20_000_000

        def _wrap_modules(module: nn.Module) -> int:
            wrapped_count = 0
            for name, child in list(module.named_children()):
                if isinstance(child, FSDP):
                    continue
                should_wrap = False
                if use_block_wrap:
                    if isinstance(child, tuple(block_classes)):
                        should_wrap = True
                else:
                    num_params = sum(p.numel() for p in child.parameters())
                    if num_params >= min_num_params:
                        should_wrap = True
                if should_wrap:
                    setattr(
                        module,
                        name,
                        FSDP(
                            child,
                            sharding_strategy=sharding_strategy,
                            mixed_precision=mp_policy,
                            use_orig_params=False,
                            sync_module_states=True,
                            limit_all_gathers=True,
                            device_id=self.device,
                        ),
                    )
                    wrapped_count += 1
                else:
                    wrapped_count += _wrap_modules(child)
            return wrapped_count

        # Keep base model on CPU before sharding to avoid full GPU materialization.
        self.model = self.model.cpu()
        wrapped = _wrap_modules(self.model)
        if wrapped == 0:
            print("Warning: FSDP enabled but no modules were wrapped; check model class names.")
        else:
            print(f"[FSDP] Wrapped {wrapped} submodules for sharding.")
        self.is_fsdp = wrapped > 0

    @staticmethod
    def _smart_resize(
        height: int,
        width: int,
        factor: int,
        min_pixels: int,
        max_pixels: int,
    ) -> Tuple[int, int]:
        if max(height, width) / max(1, min(height, width)) > 200:
            raise ValueError(
                "absolute aspect ratio must be smaller than 200, got "
                f"{max(height, width) / max(1, min(height, width))}"
            )
        h_bar = round(height / factor) * factor
        w_bar = round(width / factor) * factor
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = max(factor, math.floor(height / beta / factor) * factor)
            w_bar = max(factor, math.floor(width / beta / factor) * factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = math.ceil(height * beta / factor) * factor
            w_bar = math.ceil(width * beta / factor) * factor
        return int(h_bar), int(w_bar)

    def _normalize_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 4:
            raise ValueError("Expected pixel_values with shape [B, C, H, W].")
        if pixel_values.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            pixel_values = pixel_values.float()
        if pixel_values.size(1) == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)
        pixel_values = pixel_values.to(self.device)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image_processor = getattr(self.processor, "image_processor", None)
        if image_processor is not None:
            mean = list(getattr(image_processor, "image_mean", mean))
            std = list(getattr(image_processor, "image_std", std))
        mean_t = torch.tensor(mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        return (pixel_values - mean_t) / std_t

    def _patchify(
        self, pixel_values: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Qwen2/3-VL style patchify.
        model_cfg = getattr(self.model, "config", None)
        vision_cfg = getattr(model_cfg, "vision_config", None)
        patch_size = int(getattr(vision_cfg, "patch_size", 16))
        temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2))
        merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))

        height, width = pixel_values.shape[-2:]
        factor = patch_size * merge_size
        image_processor = getattr(self.processor, "image_processor", None)
        size = getattr(image_processor, "size", {}) if image_processor is not None else {}
        if hasattr(size, "get"):
            min_pixels = int(size.get("shortest_edge", factor * factor))
            max_pixels = int(size.get("longest_edge", height * width))
        else:
            min_pixels = int(factor * factor)
            max_pixels = int(height * width)
        try:
            resized_h, resized_w = self._smart_resize(height, width, factor, min_pixels, max_pixels)
        except Exception:
            resized_h, resized_w = height, width

        if (resized_h, resized_w) != (height, width):
            pixel_values = F.interpolate(
                pixel_values,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )

        height, width = pixel_values.shape[-2:]
        pad_h = (factor - height % factor) % factor
        pad_w = (factor - width % factor) % factor
        if pad_h or pad_w:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h))

        batch_size, channels, height, width = pixel_values.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches_list = []
        grid_thw_list = []
        for idx in range(batch_size):
            frames = pixel_values[idx : idx + 1]
            if frames.shape[0] % temporal_patch_size != 0:
                repeat_num = temporal_patch_size - (frames.shape[0] % temporal_patch_size)
                frames = torch.cat([frames, frames[-1:].repeat(repeat_num, 1, 1, 1)], dim=0)
            grid_t = frames.shape[0] // temporal_patch_size

            patches = frames.reshape(
                grid_t,
                temporal_patch_size,
                channels,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            flatten = patches.reshape(
                grid_t * grid_h * grid_w,
                channels * temporal_patch_size * patch_size * patch_size,
            )
            patches_list.append(flatten)
            grid_thw_list.append([grid_t, grid_h, grid_w])

        model_dtype = getattr(self.model, "dtype", pixel_values.dtype)
        pixel_values_out = torch.cat(patches_list, dim=0).to(self.device, dtype=model_dtype)
        image_grid_thw = torch.tensor(
            grid_thw_list,
            device=self.device,
            dtype=torch.long,
        )
        return pixel_values_out, image_grid_thw

    def _vision_forward(
        self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor]
    ) -> torch.Tensor:
        vt = self.vision_tower
        try:
            out = vt(pixel_values=pixel_values, image_grid_thw=image_grid_thw, return_dict=True)
        except Exception:
            try:
                out = vt(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
            except Exception:
                try:
                    out = vt(pixel_values)
                except Exception:
                    out = vt(pixel_values, image_grid_thw)

        if isinstance(out, torch.Tensor):
            return out
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            return out.hidden_states[-1]
        if isinstance(out, (tuple, list)) and len(out) > 0:
            return out[0]
        raise RuntimeError("Unable to parse vision tower outputs.")

    @staticmethod
    def _reshape_vision_tokens(
        tokens: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        batch_size: int,
        merge_size: int = 1,
    ) -> torch.Tensor:
        """Return vision tokens in [B, T, D] format."""
        if tokens.ndim == 3:
            if tokens.shape[0] == batch_size:
                return tokens
            if tokens.shape[1] == batch_size:
                return tokens.permute(1, 0, 2).contiguous()
        elif tokens.ndim == 2:
            if tokens.shape[0] == batch_size:
                return tokens.unsqueeze(1)
        if image_grid_thw is None:
            if batch_size == 1:
                return tokens.unsqueeze(0) if tokens.ndim == 2 else tokens
            raise ValueError(
                f"Unable to reshape vision tokens without grid info: tokens={tokens.shape}, batch={batch_size}"
            )
        if tokens.ndim != 2:
            raise ValueError(
                f"Unsupported vision token shape for grid split: tokens={tokens.shape}, batch={batch_size}"
            )
        counts = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()
        total = sum(int(c) for c in counts)
        extra_per_sample = 0
        chosen_counts = counts
        if tokens.shape[0] == total + batch_size:
            extra_per_sample = 1
        elif tokens.shape[0] != total:
            merged_counts = None
            if merge_size > 1:
                grid_t = image_grid_thw[:, 0]
                grid_h = image_grid_thw[:, 1] // merge_size
                grid_w = image_grid_thw[:, 2] // merge_size
                merged_counts = (grid_t * grid_h * grid_w).tolist()
                merged_total = sum(int(c) for c in merged_counts)
                if tokens.shape[0] == merged_total + batch_size:
                    extra_per_sample = 1
                    chosen_counts = merged_counts
                elif tokens.shape[0] == merged_total:
                    chosen_counts = merged_counts
                else:
                    merged_counts = None
            if merged_counts is None:
                raise ValueError(
                    "Vision token count mismatch: "
                    f"tokens={tokens.shape[0]}, expected={total} (or {total + batch_size})."
                )
        chunks = []
        start = 0
        for count in chosen_counts:
            count = int(count) + extra_per_sample
            end = start + count
            chunks.append(tokens[start:end])
            start = end
        max_len = max(chunk.shape[0] for chunk in chunks)
        dim = chunks[0].shape[-1]
        padded = tokens.new_zeros((batch_size, max_len, dim))
        for idx, chunk in enumerate(chunks):
            padded[idx, : chunk.shape[0]] = chunk
        return padded

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        encoding = self.tokenizer(
            captions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            token_embeds = self.text_embed(input_ids)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (token_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return pooled

    def encode_vision_tokens(self, images: torch.Tensor) -> torch.Tensor:
        normalized = self._normalize_images(images)
        batch_size = normalized.shape[0]
        vision_cfg = getattr(getattr(self.model, "config", None), "vision_config", None)
        merge_size = max(1, int(getattr(vision_cfg, "spatial_merge_size", 1)))
        try:
            pixel_values_out, image_grid_thw = self._patchify(normalized)
            tokens = self._vision_forward(pixel_values_out, image_grid_thw)
            return self._reshape_vision_tokens(tokens, image_grid_thw, batch_size, merge_size=merge_size)
        except Exception:
            # Fallback: feed normalized image directly (for towers expecting raw pixel_values)
            tokens = self._vision_forward(normalized, None)
            return self._reshape_vision_tokens(tokens, None, batch_size, merge_size=merge_size)


class AlignmentModel(nn.Module):
    def __init__(
        self,
        model_path: str,
        seq_len: int = 15,
        image_size: int = 224,
        periodicity: int = 30,
        input_dim: int = 8,
        hidden_dim: int = 64,
        prefix_tokens: int = 8,
        prefix_mode: str = "prefix",
        init_temperature: float = 0.07,
        learnable_logit_scale: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        init_logit = math.log(1.0 / max(float(init_temperature), 1e-6))
        self.logit_scale = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32),
            requires_grad=bool(learnable_logit_scale),
        )
        self.backbone = Qwen3VLBackbone(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device=self.device,
        )
        self.ts_projector = TimeSeriesProjector(
            seq_len=seq_len,
            image_size=image_size,
            periodicity=periodicity,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )

        # Infer vision dimension from a dummy forward on a single fake image.
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=self.device)
            tokens = self.backbone.encode_vision_tokens(dummy)
            if tokens.dim() == 2:
                vision_dim = tokens.shape[-1]
            else:
                vision_dim = tokens.shape[-1]
        self.vision_projector = VisionPrefixProjector(
            vision_dim=vision_dim,
            text_dim=self.backbone.text_dim,
            prefix_tokens=prefix_tokens,
            mode=prefix_mode,
        )
        projector_dtype = getattr(self.backbone.model, "dtype", torch_dtype)
        self.vision_projector.to(device=self.device, dtype=projector_dtype)
        self.to(self.device)

    def encode_image(self, sequences: torch.Tensor) -> torch.Tensor:
        images = self.ts_projector(sequences)
        vision_tokens = self.backbone.encode_vision_tokens(images)
        if vision_tokens.dim() == 2:
            vision_tokens = vision_tokens.unsqueeze(0)
        vision_pooled = vision_tokens.mean(dim=1)
        image_vec, _ = self.vision_projector(vision_pooled)
        image_vec = F.normalize(image_vec, dim=-1)
        return image_vec

    def encode_text(self, captions: list[str]) -> torch.Tensor:
        text_vec = self.backbone.encode_text(captions)
        text_vec = F.normalize(text_vec, dim=-1)
        return text_vec

    def forward(self, sequences: torch.Tensor, captions: list[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        image_vec = self.encode_image(sequences)
        text_vec = self.encode_text(captions)
        return image_vec, text_vec
