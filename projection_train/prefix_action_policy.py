from __future__ import annotations

import inspect
import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from models import Qwen3VLBackbone, TimeSeriesProjector, VisionPrefixProjector
from projection_train.action_space import ACTION_LIST, action_to_index


VISUAL_ABLATION_MODES = (
    "full",
    "zero_prefix",
    "shuffled_image",
    "constant_image",
)


def apply_visual_image_ablation(images: torch.Tensor, mode: str) -> torch.Tensor:
    """Apply an inference-time visual intervention without changing tensor shape."""
    resolved_mode = str(mode or "full").strip().lower()
    if resolved_mode not in VISUAL_ABLATION_MODES:
        raise ValueError(
            f"Unsupported visual ablation mode {mode!r}; expected one of {VISUAL_ABLATION_MODES}."
        )
    if resolved_mode in {"full", "zero_prefix"}:
        return images
    if resolved_mode == "constant_image":
        return torch.zeros_like(images)
    if images.shape[0] < 2:
        raise ValueError("shuffled_image requires an inference batch with at least two samples.")
    return torch.roll(images, shifts=1, dims=0)


def _resolve_device(device: Optional[torch.device | str]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str) and device.strip():
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _strip_state_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    resolved: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            resolved[key[len(prefix) :]] = value
    return resolved


def _maybe_load_module(
    module: nn.Module,
    checkpoint: Dict[str, Any],
    key: str,
    load: bool = True,
) -> bool:
    if not load:
        return False
    state = checkpoint.get(key)
    def _load_exact(state_dict: Dict[str, torch.Tensor]) -> bool:
        incompatible = module.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Checkpoint module {key!r} is incompatible: "
                f"missing_keys={incompatible.missing_keys}, "
                f"unexpected_keys={incompatible.unexpected_keys}."
            )
        return True

    if isinstance(state, dict):
        return _load_exact(state)
    if isinstance(checkpoint.get("state_dict"), dict):
        nested = _strip_state_prefix(checkpoint["state_dict"], f"{key}.")
        if nested:
            return _load_exact(nested)
    nested = _strip_state_prefix(checkpoint, f"{key}.")
    if nested:
        return _load_exact(nested)
    module_prefix = _strip_state_prefix(checkpoint, f"module.{key}.")
    if module_prefix:
        return _load_exact(module_prefix)
    return False


def _extract_module_state(checkpoint: Dict[str, Any], key: str) -> Dict[str, torch.Tensor]:
    state = checkpoint.get(key)
    if isinstance(state, dict):
        return state
    if isinstance(checkpoint.get("state_dict"), dict):
        nested = _strip_state_prefix(checkpoint["state_dict"], f"{key}.")
        if nested:
            return nested
    nested = _strip_state_prefix(checkpoint, f"{key}.")
    if nested:
        return nested
    return _strip_state_prefix(checkpoint, f"module.{key}.")


def _load_module_with_shape_intersection(
    module: nn.Module,
    checkpoint: Dict[str, Any],
    key: str,
) -> list[str]:
    source_state = _extract_module_state(checkpoint, key)
    if not source_state:
        return [f"{key}:missing_source_state"]

    current_state = module.state_dict()
    adapted_state: Dict[str, torch.Tensor] = {}
    notes: list[str] = []
    for name, current_tensor in current_state.items():
        source_tensor = source_state.get(name)
        if not isinstance(source_tensor, torch.Tensor):
            adapted_state[name] = current_tensor
            notes.append(f"{key}.{name}:kept_current_missing_source")
            continue
        if tuple(source_tensor.shape) == tuple(current_tensor.shape):
            adapted_state[name] = source_tensor.to(dtype=current_tensor.dtype)
            continue
        if source_tensor.ndim != current_tensor.ndim:
            adapted_state[name] = current_tensor
            notes.append(
                f"{key}.{name}:kept_current_rank_mismatch "
                f"source={tuple(source_tensor.shape)} current={tuple(current_tensor.shape)}"
            )
            continue
        target = current_tensor.detach().cpu().clone()
        source_cpu = source_tensor.detach().cpu().to(dtype=target.dtype)
        slices = tuple(slice(0, min(int(src), int(dst))) for src, dst in zip(source_cpu.shape, target.shape))
        target[slices].copy_(source_cpu[slices])
        adapted_state[name] = target
        notes.append(
            f"{key}.{name}:copied_overlap "
            f"source={tuple(source_tensor.shape)} current={tuple(current_tensor.shape)}"
        )
    incompatible = module.load_state_dict(adapted_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Adapted checkpoint module {key!r} is incompatible: "
            f"missing_keys={incompatible.missing_keys}, "
            f"unexpected_keys={incompatible.unexpected_keys}."
        )
    return notes


class PrefixActionPolicy(nn.Module):
    def __init__(
        self,
        model_path: str,
        seq_len: int = 16,
        image_size: int = 224,
        periodicity: int = 16,
        input_dim: int = 8,
        hidden_dim: int = 64,
        prefix_tokens: int = 16,
        prefix_mode: str = "prefix",
        max_length: int = 128,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device | str] = None,
        with_value_head: bool = False,
        value_hidden_dim: int = 512,
        backbone: Optional[Qwen3VLBackbone] = None,
        backbone_fsdp_enabled: bool = False,
        backbone_fsdp_sharding_strategy: str = "full_shard",
        action_eval_chunk_size: int = 0,
        visual_ablation: str = "full",
        representation_mode: str = "ts2i",
        enable_topology_adapter: bool = False,
        topology_roadnet_configs: Optional[Dict[str, Dict]] = None,
    ) -> None:
        super().__init__()
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.seq_len = int(seq_len)
        self.image_size = int(image_size)
        self.input_dim = int(input_dim)
        self.with_value_head = bool(with_value_head)
        self.action_eval_chunk_size = max(0, int(action_eval_chunk_size))
        self.visual_ablation = str(visual_ablation or "full").strip().lower()
        self.representation_mode = str(representation_mode or "ts2i").strip().lower()
        if self.representation_mode not in {"ts2i", "history_text"}:
            raise ValueError(f"Unsupported representation mode: {representation_mode!r}")
        if self.visual_ablation not in VISUAL_ABLATION_MODES:
            raise ValueError(
                f"Unsupported visual ablation mode {visual_ablation!r}; "
                f"expected one of {VISUAL_ABLATION_MODES}."
            )
        self.backbone_gradient_checkpointing = False
        self._vision_feature_mode = "last_hidden_state"

        if backbone is None:
            self.backbone = Qwen3VLBackbone(
                model_path=model_path,
                torch_dtype=torch_dtype,
                device=self.device,
                max_length=self.max_length,
                fsdp_enabled=bool(backbone_fsdp_enabled),
                fsdp_sharding_strategy=str(backbone_fsdp_sharding_strategy or "full_shard"),
            )
        else:
            self.backbone = backbone
        try:
            self._supports_logits_to_keep = "logits_to_keep" in inspect.signature(
                self.backbone.model.forward
            ).parameters
        except Exception:
            self._supports_logits_to_keep = False
        self.ts_projector = TimeSeriesProjector(
            seq_len=seq_len,
            image_size=image_size,
            periodicity=periodicity,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            device=self.device,
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size, device=self.device)
            vision_tokens = self.backbone.encode_vision_tokens(dummy)
            vision_dim = int(vision_tokens.shape[-1])

        self.vision_projector = VisionPrefixProjector(
            vision_dim=vision_dim,
            text_dim=self.backbone.text_dim,
            prefix_tokens=prefix_tokens,
            mode=prefix_mode,
        )
        projector_dtype = getattr(self.backbone.model, "dtype", torch_dtype)
        self.vision_projector.to(device=self.device, dtype=projector_dtype)
        self.ts_projector.to(self.device)
        self.history_prefix = nn.Parameter(
            torch.empty(prefix_tokens, self.backbone.text_dim, device=self.device, dtype=projector_dtype).normal_(0, 0.02),
            requires_grad=self.representation_mode == "history_text",
        )
        if self.representation_mode == "history_text":
            self.ts_projector.requires_grad_(False)
            self.vision_projector.requires_grad_(False)

        # Topology-conditioned adapter for cross-network adaptation
        self.enable_topology_adapter = bool(enable_topology_adapter)
        self.topology_encoder: Optional[nn.Module] = None
        self._vision_adapter: Optional[nn.Module] = None

        if self.enable_topology_adapter:
            from projection_train.topology_adapter import IntersectionTopologyEncoder
            self.topology_encoder = IntersectionTopologyEncoder(
                vision_dim=vision_dim,
                hidden_dim=32,
                roadnet_configs=topology_roadnet_configs,
            )
            self.topology_encoder.to(device=self.device, dtype=projector_dtype)
            # Concat fusion: vision_pooled [vision_dim] + topo_emb [vision_dim] → adapter
            adapter_in = vision_dim * 2
            adapter_out = vision_dim
            self._vision_adapter = nn.Linear(adapter_in, adapter_out, bias=False)
            self._vision_adapter.to(device=self.device, dtype=projector_dtype)

        tokenizer = self.backbone.tokenizer
        tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._action_tag_token_ids: List[List[int]] = []
        for action in ACTION_LIST:
            tag = f"<signal>{action}</signal>"
            self._action_tag_token_ids.append(
                tokenizer(tag, add_special_tokens=False).input_ids
            )

        self.value_head: Optional[nn.Module] = None
        if self.with_value_head:
            self.value_head = nn.Sequential(
                nn.LayerNorm(self.backbone.text_dim),
                nn.Linear(self.backbone.text_dim, int(value_hidden_dim)),
                nn.Tanh(),
                nn.Linear(int(value_hidden_dim), 1),
            ).to(self.device)

        if backbone is None:
            self.to(self.device)

    def _model_dtype(self) -> torch.dtype:
        return getattr(self.backbone.model, "dtype", torch.float32)

    def _format_prompts(self, prompts: Sequence[str]) -> List[str]:
        tokenizer = self.backbone.tokenizer
        formatted: List[str] = []
        for prompt in prompts:
            text = str(prompt or "").rstrip()
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        [{"role": "user", "content": text}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            formatted.append(text)
        return formatted

    def _encode_sequences(
        self,
        sequences: torch.Tensor,
        intersection_ids: Optional[List[str]] = None,
        dataset_hints: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if sequences.device != self.device:
            sequences = sequences.to(self.device)
        images = self.ts_projector(sequences)
        images = apply_visual_image_ablation(images, self.visual_ablation)
        if self.backbone_gradient_checkpointing and torch.is_grad_enabled():
            vision_tokens = activation_checkpoint(
                lambda batch: self.backbone.encode_vision_tokens(
                    batch,
                    output_mode=self._vision_feature_mode,
                ),
                images,
                use_reentrant=False,
            )
        else:
            vision_tokens = self.backbone.encode_vision_tokens(
                images,
                output_mode=self._vision_feature_mode,
            )
        if vision_tokens.ndim == 2:
            vision_tokens = vision_tokens.unsqueeze(1)
        vision_pooled = vision_tokens.mean(dim=1)
        # Topology conditioning with concat fusion (handles dimension mismatch
        # between vision_dim=1152 and adapter_out=4096 for old checkpoints).
        if self.topology_encoder is not None and intersection_ids is not None:
            topo_emb = self.topology_encoder(
                intersection_ids,
                dataset_hints=dataset_hints,
            )
            topo_emb = topo_emb.to(device=vision_pooled.device, dtype=vision_pooled.dtype)
            vision_pooled = torch.cat([vision_pooled, topo_emb], dim=-1)
        if self._vision_adapter is not None:
            vision_pooled = self._vision_adapter(vision_pooled)
        prefix_vec, prefix_tokens = self.vision_projector(vision_pooled)
        if self.visual_ablation == "zero_prefix":
            prefix_vec = torch.zeros_like(prefix_vec)
            prefix_tokens = torch.zeros_like(prefix_tokens)
        prefix_tokens = prefix_tokens.to(device=self.device, dtype=self._model_dtype())
        return {
            "images": images,
            "vision_pooled": vision_pooled,
            "prefix_vec": prefix_vec,
            "prefix_tokens": prefix_tokens,
        }

    def _encode_prompts(self, prompts: Sequence[str]) -> Dict[str, Any]:
        formatted = self._format_prompts(prompts)
        tokenizer = self.backbone.tokenizer
        encoding = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=self.representation_mode != "history_text",
            max_length=self.max_length,
            add_special_tokens=True,
        )
        if self.representation_mode == "history_text" and encoding["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                f"History-Text prompt has {encoding['input_ids'].shape[1]} tokens, "
                f"exceeding max_length={self.max_length}; refusing to discard history"
            )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        with torch.no_grad():
            prompt_embeds = self.backbone.text_embed(input_ids)
        prompt_embeds = prompt_embeds.to(device=self.device, dtype=self._model_dtype())
        return {
            "formatted_prompts": formatted,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_embeds": prompt_embeds,
            "prompt_lengths": [int(x) for x in lengths],
        }

    def prepare_policy_batch(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        intersection_ids: Optional[List[str]] = None,
        dataset_hints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if self.representation_mode == "history_text":
            prefix_tokens = self.history_prefix.unsqueeze(0).expand(sequences.shape[0], -1, -1)
            seq_features = {
                "prefix_tokens": prefix_tokens,
                "prefix_vec": prefix_tokens.mean(dim=1),
            }
        else:
            seq_features = self._encode_sequences(
                sequences,
                intersection_ids=intersection_ids,
                dataset_hints=dataset_hints,
            )
        prompt_features = self._encode_prompts(prompts)
        prepared = {**seq_features, **prompt_features}
        if self.value_head is not None:
            prepared["values"] = self.value_head(prepared["prefix_vec"].float()).squeeze(-1)
        else:
            prepared["values"] = None
        return prepared

    def _action_tag_embed(self, action_index: int) -> torch.Tensor:
        action_index = int(action_index)
        if action_index < 0 or action_index >= len(ACTION_LIST):
            action_index = 0
        token_ids = torch.tensor(
            self._action_tag_token_ids[action_index],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            embeds = self.backbone.text_embed(token_ids.unsqueeze(0)).squeeze(0)
        return embeds.to(dtype=self._model_dtype())

    def _build_tagged_inputs(
        self,
        prepared_batch: Dict[str, Any],
        actions: Sequence[int],
    ) -> Dict[str, torch.Tensor]:
        prefix_tokens = prepared_batch["prefix_tokens"]
        prompt_embeds = prepared_batch["prompt_embeds"]
        prompt_lengths = prepared_batch["prompt_lengths"]
        batch_size = prefix_tokens.shape[0]
        prefix_len = int(prefix_tokens.shape[1])
        embed_dim = int(prefix_tokens.shape[2])

        tag_embeds = [self._action_tag_embed(int(action)) for action in actions]
        tag_lens = [int(item.shape[0]) for item in tag_embeds]
        total_lens = [
            prefix_len + int(prompt_lengths[idx]) + tag_lens[idx]
            for idx in range(batch_size)
        ]
        max_total_len = max(total_lens)

        full_embeds = prefix_tokens.new_zeros((batch_size, max_total_len, embed_dim))
        full_mask = torch.zeros((batch_size, max_total_len), device=self.device, dtype=torch.long)
        tag_token_ids = torch.full(
            (batch_size, max(tag_lens)),
            fill_value=int(self.backbone.tokenizer.pad_token_id),
            device=self.device,
            dtype=torch.long,
        )
        tag_start_pos = torch.zeros((batch_size,), device=self.device, dtype=torch.long)
        tag_len_tensor = torch.tensor(tag_lens, device=self.device, dtype=torch.long)

        for idx in range(batch_size):
            cur = 0
            full_embeds[idx, cur : cur + prefix_len] = prefix_tokens[idx]
            cur += prefix_len

            prompt_len = int(prompt_lengths[idx])
            if prompt_len > 0:
                full_embeds[idx, cur : cur + prompt_len] = prompt_embeds[idx, :prompt_len]
                cur += prompt_len

            tag_start_pos[idx] = cur
            full_embeds[idx, cur : cur + tag_lens[idx]] = tag_embeds[idx]
            tag_token_ids[idx, : tag_lens[idx]] = torch.tensor(
                self._action_tag_token_ids[int(actions[idx])],
                device=self.device,
                dtype=torch.long,
            )
            cur += tag_lens[idx]
            full_mask[idx, :cur] = 1

        return {
            "inputs_embeds": full_embeds,
            "attention_mask": full_mask,
            "tag_token_ids": tag_token_ids,
            "tag_start_pos": tag_start_pos,
            "tag_lens": tag_len_tensor,
        }

    def _forward_lm_logits(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        logits_to_keep: Optional[Any] = None,
    ) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)

        def _run_lm(
            cur_inputs_embeds: torch.Tensor,
            cur_attention_mask: torch.Tensor,
            cur_position_ids: torch.Tensor,
        ) -> torch.Tensor:
            forward_kwargs = {
                "inputs_embeds": cur_inputs_embeds,
                "attention_mask": cur_attention_mask,
                "position_ids": cur_position_ids,
                "use_cache": False,
                "return_dict": True,
            }
            if logits_to_keep is not None and self._supports_logits_to_keep:
                forward_kwargs["logits_to_keep"] = logits_to_keep
            try:
                outputs = self.backbone.model(**forward_kwargs)
            except TypeError:
                forward_kwargs.pop("position_ids", None)
                outputs = self.backbone.model(**forward_kwargs)
            logits = getattr(outputs, "logits", None)
            if logits is None:
                raise RuntimeError("Missing logits from language-model forward.")
            return logits

        if self.backbone_gradient_checkpointing and torch.is_grad_enabled():
            return activation_checkpoint(
                _run_lm,
                inputs_embeds,
                attention_mask,
                position_ids,
                use_reentrant=False,
            )

        return _run_lm(inputs_embeds, attention_mask, position_ids)

    def action_tag_logprob_from_prepared(
        self,
        prepared_batch: Dict[str, Any],
        actions: Sequence[int],
    ) -> torch.Tensor:
        tagged = self._build_tagged_inputs(prepared_batch, actions)
        logits = self._forward_lm_logits(
            inputs_embeds=tagged["inputs_embeds"],
            attention_mask=tagged["attention_mask"],
            logits_to_keep=(
                tagged["tag_start_pos"][0]
                + torch.arange(
                    int(tagged["tag_token_ids"].shape[1]),
                    device=self.device,
                    dtype=torch.long,
                )
                - 1
            ).clamp_min(0)
            if self._supports_logits_to_keep and int(tagged["inputs_embeds"].shape[0]) == 1
            else None,
        )
        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        max_tag_len = int(tagged["tag_token_ids"].shape[1])
        pos_offset = torch.arange(max_tag_len, device=logits.device).view(1, -1)
        flat_tag_ids = tagged["tag_token_ids"].reshape(-1)
        if self._supports_logits_to_keep and batch_size == 1:
            selected_rows = logits.reshape(max_tag_len, vocab_size)
        else:
            seq_len = logits.shape[1]
            batch_idx = torch.arange(batch_size, device=logits.device).view(-1, 1)
            gather_pos = tagged["tag_start_pos"].view(-1, 1) + pos_offset - 1
            gather_pos = gather_pos.clamp_min(0)
            flat_logits = logits.reshape(-1, vocab_size)
            flat_pos = (batch_idx * seq_len + gather_pos).reshape(-1)
            selected_rows = flat_logits.index_select(0, flat_pos)
        selected_logits = selected_rows.gather(1, flat_tag_ids.view(-1, 1)).view(batch_size, max_tag_len)
        selected_log_norm = torch.logsumexp(selected_rows, dim=-1).view(batch_size, max_tag_len)
        chosen = selected_logits - selected_log_norm
        mask = pos_offset < tagged["tag_lens"].view(-1, 1)
        return (chosen * mask).sum(dim=1)

    def get_action_logits_from_prepared(self, prepared_batch: Dict[str, Any]) -> torch.Tensor:
        batch_size = int(prepared_batch["prefix_tokens"].shape[0])
        num_actions = len(ACTION_LIST)
        if batch_size == 0:
            return torch.empty((0, num_actions), device=self.device)
        chunk_size = self.action_eval_chunk_size
        if chunk_size <= 0:
            chunk_size = num_actions
        chunk_size = max(1, min(int(chunk_size), num_actions))

        chunk_logits: List[torch.Tensor] = []
        for start in range(0, num_actions, chunk_size):
            action_chunk = list(range(start, min(num_actions, start + chunk_size)))
            sample_indices = torch.arange(batch_size, device=self.device).repeat_interleave(len(action_chunk))
            expanded_batch = {
                "prefix_tokens": prepared_batch["prefix_tokens"].index_select(0, sample_indices),
                "prompt_embeds": prepared_batch["prompt_embeds"].index_select(0, sample_indices),
                "prompt_lengths": [
                    prepared_batch["prompt_lengths"][int(idx)]
                    for idx in sample_indices.detach().cpu().tolist()
                ],
            }
            expanded_actions = [
                action_idx
                for _sample_idx in range(batch_size)
                for action_idx in action_chunk
            ]
            scores = self.action_tag_logprob_from_prepared(expanded_batch, expanded_actions)
            chunk_logits.append(scores.view(batch_size, len(action_chunk)))
        return torch.cat(chunk_logits, dim=1)

    def get_action_logits(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
        intersection_ids: Optional[List[str]] = None,
        dataset_hints: Optional[List[str]] = None,
    ) -> torch.Tensor:
        prepared = prepared_batch or self.prepare_policy_batch(
            sequences, prompts,
            intersection_ids=intersection_ids,
            dataset_hints=dataset_hints,
        )
        return self.get_action_logits_from_prepared(prepared)

    def forward(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
        intersection_ids: Optional[List[str]] = None,
        dataset_hints: Optional[List[str]] = None,
    ) -> torch.Tensor:
        return self.get_action_logits(
            sequences, prompts,
            prepared_batch=prepared_batch,
            intersection_ids=intersection_ids,
            dataset_hints=dataset_hints,
        )

    def action_distribution(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> tuple[Categorical, Dict[str, Any]]:
        prepared = prepared_batch or self.prepare_policy_batch(sequences, prompts)
        logits = self.get_action_logits_from_prepared(prepared).float()
        return Categorical(logits=logits), prepared

    def predict_values_from_prepared(self, prepared_batch: Dict[str, Any]) -> torch.Tensor:
        values = prepared_batch.get("values")
        if values is None:
            raise RuntimeError("value_head is disabled for this PrefixActionPolicy.")
        return values

    def predict_values(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        prepared = prepared_batch or self.prepare_policy_batch(sequences, prompts)
        return self.predict_values_from_prepared(prepared)

    def act(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        sample: bool = True,
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        dist, prepared = self.action_distribution(
            sequences=sequences,
            prompts=prompts,
            prepared_batch=prepared_batch,
        )
        if sample:
            actions = dist.sample()
        else:
            actions = torch.argmax(dist.logits, dim=-1)
        log_probs = dist.log_prob(actions)
        return {
            "actions": actions,
            "log_probs": log_probs,
            "values": prepared.get("values"),
            "entropy": dist.entropy(),
            "logits": dist.logits,
            "prepared_batch": prepared,
        }

    def evaluate_actions(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        actions: Sequence[int] | torch.Tensor,
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        dist, prepared = self.action_distribution(
            sequences=sequences,
            prompts=prompts,
            prepared_batch=prepared_batch,
        )
        if not torch.is_tensor(actions):
            action_tensor = torch.tensor(actions, device=self.device, dtype=torch.long)
        else:
            action_tensor = actions.to(device=self.device, dtype=torch.long)
        return {
            "logits": dist.logits,
            "log_probs": dist.log_prob(action_tensor),
            "entropy": dist.entropy(),
            "values": prepared.get("values"),
        }

    def action_logprob(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        actions: Sequence[int] | torch.Tensor,
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        evaluated = self.evaluate_actions(
            sequences=sequences,
            prompts=prompts,
            actions=actions,
            prepared_batch=prepared_batch,
        )
        return evaluated["log_probs"]

    def compute_entropy(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        dist, _ = self.action_distribution(
            sequences=sequences,
            prompts=prompts,
            prepared_batch=prepared_batch,
        )
        return dist.entropy()

    def trainable_parameters(self):
        if self.representation_mode == "history_text":
            return [self.history_prefix]
        params: List[nn.Parameter] = []
        params.extend(list(self.ts_projector.parameters()))
        params.extend(list(self.vision_projector.parameters()))
        if self._vision_adapter is not None:
            params.extend(list(self._vision_adapter.parameters()))
        if self.topology_encoder is not None:
            params.extend(list(self.topology_encoder.parameters()))
        if self.value_head is not None:
            params.extend(list(self.value_head.parameters()))
        return params

    def set_backbone_gradient_checkpointing(self, enabled: bool) -> None:
        self.backbone_gradient_checkpointing = bool(enabled)
        self.backbone.set_gradient_checkpointing(enabled)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_value_head: bool = True,
        allow_mismatched_projectors: bool = False,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")
        saved_mode = str((checkpoint.get("args") or {}).get("representation_mode") or "ts2i")
        if saved_mode != self.representation_mode:
            raise ValueError(f"Checkpoint representation {saved_mode} != policy {self.representation_mode}")
        if self.representation_mode == "history_text":
            stored_prefix = checkpoint.get("history_prefix")
            if stored_prefix is None or tuple(stored_prefix.shape) != tuple(self.history_prefix.shape):
                raise ValueError("History-Text checkpoint is missing a compatible history_prefix")
            with torch.no_grad():
                self.history_prefix.copy_(stored_prefix.to(self.history_prefix))
            return checkpoint

        # A Qwen3-VL API change exposed raw visual states (1152) as
        # ``last_hidden_state`` while historical checkpoints were trained on the
        # vision merger output (4096).  Select a real backbone feature path that
        # matches the serialized projector; never install an untrained adapter.
        vp_state = checkpoint.get("vision_projector")
        if isinstance(vp_state, dict) and "proj.weight" in vp_state:
            stored_weight = vp_state["proj.weight"]
            current_weight = self.vision_projector.proj.weight
            if stored_weight.shape != current_weight.shape:
                stored_in_dim = int(stored_weight.shape[1])
                stored_out_dim = int(stored_weight.shape[0])
                stored_prefix_tokens = 1
                if self.vision_projector.mode == "prefix":
                    checkpoint_args = checkpoint.get("args")
                    if isinstance(checkpoint_args, dict):
                        stored_prefix_tokens = int(
                            checkpoint_args.get("prefix_tokens", self.vision_projector.prefix_tokens)
                        )
                    if stored_prefix_tokens <= 0 or stored_out_dim % stored_prefix_tokens != 0:
                        raise RuntimeError(
                            "Vision projector prefix-token metadata is incompatible with the checkpoint: "
                            f"prefix_tokens={stored_prefix_tokens}, output_dim={stored_out_dim}."
                        )
                stored_text_dim = stored_out_dim // stored_prefix_tokens
                if stored_text_dim != self.backbone.text_dim:
                    raise RuntimeError(
                        "Vision projector text dimension does not match the loaded backbone: "
                        f"checkpoint={stored_text_dim}, backbone={self.backbone.text_dim}."
                    )

                if stored_in_dim != int(current_weight.shape[1]):
                    with torch.no_grad():
                        dummy = torch.zeros(
                            1,
                            3,
                            self.image_size,
                            self.image_size,
                            device=self.device,
                        )
                        pooled_dim = int(
                            self.backbone.encode_vision_tokens(
                                dummy,
                                output_mode="pooler_output",
                            ).shape[-1]
                        )
                    if pooled_dim != stored_in_dim:
                        raise RuntimeError(
                            "Checkpoint vision dimension is incompatible with this backbone: "
                            f"checkpoint={stored_in_dim}, raw={current_weight.shape[1]}, "
                            f"pooler_output={pooled_dim}. Refusing to use an untrained adapter."
                        )

                    self._vision_feature_mode = "pooler_output"
                    logging.getLogger("projection_train.policy").info(
                        "Using Qwen3-VL pooler_output (%d dims) for historical vision projector.",
                        pooled_dim,
                    )
                    new_projector = VisionPrefixProjector(
                        vision_dim=stored_in_dim,
                        text_dim=stored_text_dim,
                        prefix_tokens=self.vision_projector.prefix_tokens,
                        mode=self.vision_projector.mode,
                    )
                    new_projector.to(
                        device=self.vision_projector.proj.weight.device,
                        dtype=self.vision_projector.proj.weight.dtype,
                    )
                    self.vision_projector = new_projector

                current_prefix_tokens = self.vision_projector.prefix_tokens
                if (
                    self.vision_projector.mode == "prefix"
                    and stored_prefix_tokens != current_prefix_tokens
                ):
                    copied_tokens = min(stored_prefix_tokens, current_prefix_tokens)
                    with torch.no_grad():
                        source_weight = stored_weight.reshape(
                            stored_prefix_tokens, stored_text_dim, stored_in_dim
                        )
                        target_weight = self.vision_projector.proj.weight.reshape(
                            current_prefix_tokens, stored_text_dim, stored_in_dim
                        )
                        target_weight[:copied_tokens].copy_(
                            source_weight[:copied_tokens].to(
                                device=target_weight.device,
                                dtype=target_weight.dtype,
                            )
                        )
                        if "proj.bias" in vp_state:
                            source_bias = vp_state["proj.bias"].reshape(
                                stored_prefix_tokens, stored_text_dim
                            )
                            target_bias = self.vision_projector.proj.bias.reshape(
                                current_prefix_tokens, stored_text_dim
                            )
                            target_bias[:copied_tokens].copy_(
                                source_bias[:copied_tokens].to(
                                    device=target_bias.device,
                                    dtype=target_bias.dtype,
                                )
                            )

                    # _maybe_load_module below is strict about tensor shapes, so replace
                    # the checkpoint state with the prefix-compatible initialization.
                    checkpoint = dict(checkpoint)
                    checkpoint["vision_projector"] = {
                        **vp_state,
                        "proj.weight": self.vision_projector.proj.weight.detach().cpu().clone(),
                        "proj.bias": self.vision_projector.proj.bias.detach().cpu().clone(),
                    }
                    logging.getLogger("projection_train.policy").info(
                        "Adapted vision projector prefix tokens from %d to %d; copied %d token blocks.",
                        stored_prefix_tokens,
                        current_prefix_tokens,
                        copied_tokens,
                    )

        try:
            _maybe_load_module(self.ts_projector, checkpoint, "ts_projector", load=True)
        except RuntimeError:
            if not allow_mismatched_projectors:
                raise
            notes = _load_module_with_shape_intersection(self.ts_projector, checkpoint, "ts_projector")
            logging.getLogger("projection_train.policy").warning(
                "Loaded ts_projector from %s with shape-intersection adaptation: %s",
                checkpoint_path,
                "; ".join(notes),
            )
        try:
            _maybe_load_module(self.vision_projector, checkpoint, "vision_projector", load=True)
        except RuntimeError:
            if not allow_mismatched_projectors:
                raise
            notes = _load_module_with_shape_intersection(self.vision_projector, checkpoint, "vision_projector")
            logging.getLogger("projection_train.policy").warning(
                "Loaded vision_projector from %s with shape-intersection adaptation: %s",
                checkpoint_path,
                "; ".join(notes),
            )
        if self.topology_encoder is not None:
            _maybe_load_module(self.topology_encoder, checkpoint, "topology_encoder", load=True)
        if self.value_head is not None:
            _maybe_load_module(self.value_head, checkpoint, "value_head", load=load_value_head)
        return checkpoint

    @torch.no_grad()
    def reference_kl(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        reference_policy: "PrefixActionPolicy",
    ) -> torch.Tensor:
        current_logits = self(sequences, prompts).float()
        reference_logits = reference_policy(sequences, prompts).float()
        current_log_probs = F.log_softmax(current_logits, dim=-1)
        reference_log_probs = F.log_softmax(reference_logits, dim=-1)
        current_probs = current_log_probs.exp()
        return torch.sum(
            current_probs * (current_log_probs - reference_log_probs),
            dim=-1,
        )

    def greedy_action_names(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
    ) -> List[str]:
        logits = self(sequences, prompts)
        indices = torch.argmax(logits, dim=-1).tolist()
        return [ACTION_LIST[int(idx)] for idx in indices]

    def normalize_actions(
        self,
        actions: Sequence[int | str] | torch.Tensor,
    ) -> torch.Tensor:
        if torch.is_tensor(actions):
            return actions.to(device=self.device, dtype=torch.long)
        normalized = []
        for action in actions:
            if isinstance(action, str):
                normalized.append(action_to_index(action))
            else:
                normalized.append(int(action))
        return torch.tensor(normalized, device=self.device, dtype=torch.long)
