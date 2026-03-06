from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from models import Qwen3VLBackbone, TimeSeriesProjector, VisionPrefixProjector
from projection_train.action_space import ACTION_LIST, action_to_index, format_action_tag


def _strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    return {
        (key[len(prefix) :] if key.startswith(prefix) else key): value
        for key, value in state_dict.items()
    }


def _normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = dict(state_dict)
    while state and all(key.startswith("module.") for key in state):
        state = _strip_prefix(state, "module.")
    return state


def _unwrap_module(module: nn.Module | None) -> nn.Module | None:
    if module is None:
        return None
    return module.module if hasattr(module, "module") else module


@dataclass
class PreparedPolicyBatch:
    context_embeds: torch.Tensor
    context_mask: torch.Tensor
    context_lengths: List[int]


class PrefixActionPolicy(nn.Module):
    def __init__(
        self,
        model_path: str,
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
        with_value_head: bool = False,
        llm_fsdp: bool = False,
        fsdp_sharding_strategy: str = "full_shard",
    ) -> None:
        super().__init__()
        self.device = device
        self.backbone = Qwen3VLBackbone(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device=device,
            max_length=max_length,
            fsdp_enabled=llm_fsdp,
            fsdp_sharding_strategy=fsdp_sharding_strategy,
        )
        self.backbone.tokenizer.padding_side = "right"
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
        projector_dtype = getattr(self.backbone.model, "dtype", torch_dtype)
        self.vision_projector.to(device=device, dtype=projector_dtype)
        self.ts_projector.to(device=device)

        self.value_head = None
        if with_value_head:
            self.value_head = nn.Linear(self.backbone.text_dim, 1)
            self.value_head.to(device=device, dtype=projector_dtype)

        self._action_tag_token_ids: List[List[int]] = []
        for action in ACTION_LIST:
            token_ids = self.backbone.tokenizer(
                format_action_tag(action),
                add_special_tokens=False,
            ).input_ids
            if not token_ids:
                raise ValueError(f"Tokenizer produced empty ids for action tag: {action}")
            self._action_tag_token_ids.append(token_ids)

        self._freeze_backbone()
        if not getattr(self.backbone, "is_fsdp", False):
            self.to(device)

    def _freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.model.eval()
        for parameter in self.ts_projector.parameters():
            parameter.requires_grad = True
        for parameter in self.vision_projector.parameters():
            parameter.requires_grad = True
        if self.value_head is not None:
            for parameter in self.value_head.parameters():
                parameter.requires_grad = True

    def trainable_parameters(self) -> List[nn.Parameter]:
        params = list(self.ts_projector.parameters()) + list(self.vision_projector.parameters())
        if self.value_head is not None:
            params.extend(self.value_head.parameters())
        return [parameter for parameter in params if parameter.requires_grad]

    def _project_sequences(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                images = self.ts_projector(sequences.float())
        else:
            images = self.ts_projector(sequences.float())
        vision_tokens = self.backbone.encode_vision_tokens(images)
        if vision_tokens.dim() == 2:
            vision_tokens = vision_tokens.unsqueeze(1)
        vision_pooled = vision_tokens.mean(dim=1)
        _, prefix_tokens = self.vision_projector(vision_pooled)
        return prefix_tokens

    def prepare_policy_batch(self, sequences: torch.Tensor, prompts: Sequence[str]) -> PreparedPolicyBatch:
        prefix_tokens = self._project_sequences(sequences.to(self.device))
        encoding = self.backbone.tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.backbone.max_length,
            add_special_tokens=True,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        with torch.no_grad():
            padded_prompt_embeds = self.backbone.text_embed(input_ids)

        context_tensors: List[torch.Tensor] = []
        context_masks: List[torch.Tensor] = []
        context_lengths: List[int] = []
        for idx in range(input_ids.size(0)):
            prompt_len = int(attention_mask[idx].sum().item())
            prompt_embeds = padded_prompt_embeds[idx, :prompt_len]
            prefix = prefix_tokens[idx]
            if prefix.dtype != prompt_embeds.dtype:
                prefix = prefix.to(dtype=prompt_embeds.dtype)
            context = torch.cat([prefix, prompt_embeds], dim=0)
            context_tensors.append(context)
            context_masks.append(torch.ones(context.shape[0], dtype=torch.long, device=self.device))
            context_lengths.append(int(context.shape[0]))

        return PreparedPolicyBatch(
            context_embeds=pad_sequence(context_tensors, batch_first=True),
            context_mask=pad_sequence(context_masks, batch_first=True),
            context_lengths=context_lengths,
        )

    def _context_slices(self, prepared: PreparedPolicyBatch) -> List[torch.Tensor]:
        return [
            prepared.context_embeds[idx, : prepared.context_lengths[idx]]
            for idx in range(prepared.context_embeds.size(0))
        ]

    def get_values_from_prepared(self, prepared: PreparedPolicyBatch) -> torch.Tensor:
        if self.value_head is None:
            raise RuntimeError("Value head is disabled for this policy.")
        outputs = self.backbone.model(
            inputs_embeds=prepared.context_embeds,
            attention_mask=prepared.context_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        last_index = prepared.context_mask.sum(dim=1) - 1
        batch_index = torch.arange(hidden_states.size(0), device=hidden_states.device)
        pooled = hidden_states[batch_index, last_index]
        value_head = _unwrap_module(self.value_head)
        if value_head is None:
            raise RuntimeError("Value head is disabled for this policy.")
        if pooled.dtype != value_head.weight.dtype:
            pooled = pooled.to(dtype=value_head.weight.dtype)
        return self.value_head(pooled).squeeze(-1)

    def get_values(self, sequences: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        prepared = self.prepare_policy_batch(sequences, prompts)
        return self.get_values_from_prepared(prepared)

    def action_logprob_from_prepared(
        self,
        prepared: PreparedPolicyBatch,
        actions: Sequence[Union[int, str]],
    ) -> torch.Tensor:
        context_slices = self._context_slices(prepared)
        tag_id_tensors: List[torch.Tensor] = []
        for action in actions:
            action_idx = int(action) if isinstance(action, int) else action_to_index(str(action))
            tag_id_tensors.append(
                torch.tensor(
                    self._action_tag_token_ids[action_idx],
                    device=self.device,
                    dtype=torch.long,
                )
            )

        with torch.no_grad():
            tag_embed_tensors = [
                self.backbone.text_embed(tag_ids.unsqueeze(0)).squeeze(0)
                for tag_ids in tag_id_tensors
            ]

        full_tensors: List[torch.Tensor] = []
        full_masks: List[torch.Tensor] = []
        for context, tag_embeds in zip(context_slices, tag_embed_tensors):
            if context.dtype != tag_embeds.dtype:
                tag_embeds = tag_embeds.to(dtype=context.dtype)
            full = torch.cat([context, tag_embeds], dim=0)
            full_tensors.append(full)
            full_masks.append(torch.ones(full.shape[0], dtype=torch.long, device=self.device))

        padded_full = pad_sequence(full_tensors, batch_first=True)
        padded_mask = pad_sequence(full_masks, batch_first=True)
        outputs = self.backbone.model(
            inputs_embeds=padded_full,
            attention_mask=padded_mask,
            return_dict=True,
        )
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        results: List[torch.Tensor] = []
        for row_idx, tag_ids in enumerate(tag_id_tensors):
            context_len = prepared.context_lengths[row_idx]
            total = log_probs.new_zeros(())
            for offset, token_id in enumerate(tag_ids):
                position = context_len + offset
                total = total + log_probs[row_idx, position - 1, token_id]
            results.append(total)
        return torch.stack(results, dim=0)

    def action_logprob(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        actions: Sequence[Union[int, str]],
    ) -> torch.Tensor:
        prepared = self.prepare_policy_batch(sequences, prompts)
        return self.action_logprob_from_prepared(prepared, actions)

    def get_action_logits_from_prepared(self, prepared: PreparedPolicyBatch) -> torch.Tensor:
        all_logprobs = []
        batch_size = prepared.context_embeds.size(0)
        for action in ACTION_LIST:
            logprob = self.action_logprob_from_prepared(prepared, [action] * batch_size)
            all_logprobs.append(logprob)
        return torch.stack(all_logprobs, dim=-1)

    def get_action_logits(self, sequences: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        prepared = self.prepare_policy_batch(sequences, prompts)
        return self.get_action_logits_from_prepared(prepared)

    def evaluate_actions_from_prepared(
        self,
        prepared: PreparedPolicyBatch,
        action_indices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        logits = self.get_action_logits_from_prepared(prepared)
        dist = torch.distributions.Categorical(logits=logits)
        values = self.get_values_from_prepared(prepared) if self.value_head is not None else None
        return {
            "logits": logits,
            "logprob": dist.log_prob(action_indices),
            "entropy": dist.entropy(),
            "values": values,
        }

    def predict_actions(self, sequences: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        logits = self.get_action_logits(sequences, prompts)
        return logits.argmax(dim=-1)

    def _load_ts_projector_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        state = _normalize_state_dict(state_dict)
        try:
            self.ts_projector.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            pass

        image_module_state = state
        if any(key.startswith("image_module.") for key in image_module_state):
            image_module_state = _strip_prefix(image_module_state, "image_module.")
        if hasattr(self.ts_projector, "image_module") and isinstance(self.ts_projector.image_module, nn.Module):
            self.ts_projector.image_module.load_state_dict(image_module_state, strict=True)
            return
        self.ts_projector.load_state_dict(state, strict=True)

    def _load_vision_projector_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        state = _normalize_state_dict(state_dict)
        try:
            self.vision_projector.load_state_dict(state, strict=True)
            return
        except RuntimeError:
            pass

        if "proj.weight" not in state:
            self.vision_projector.load_state_dict(state, strict=False)
            return

        current_weight = self.vision_projector.proj.weight
        saved_weight = state["proj.weight"]
        if saved_weight.shape[1] != current_weight.shape[1]:
            raise ValueError(
                "Vision projector input dim mismatch: "
                f"checkpoint={tuple(saved_weight.shape)}, current={tuple(current_weight.shape)}"
            )

        text_dim = int(self.backbone.text_dim)
        if saved_weight.shape[0] % text_dim != 0 or current_weight.shape[0] % text_dim != 0:
            raise ValueError(
                "Vision projector output dim mismatch: "
                f"checkpoint={tuple(saved_weight.shape)}, current={tuple(current_weight.shape)}"
            )

        saved_prefix = saved_weight.shape[0] // text_dim
        current_prefix = current_weight.shape[0] // text_dim
        tokens_to_copy = min(saved_prefix, current_prefix)

        with torch.no_grad():
            saved_view = saved_weight.view(saved_prefix, text_dim, saved_weight.shape[1])
            current_view = current_weight.view(current_prefix, text_dim, current_weight.shape[1])
            current_view[:tokens_to_copy].copy_(saved_view[:tokens_to_copy])
            if "proj.bias" in state and self.vision_projector.proj.bias is not None:
                saved_bias = state["proj.bias"].view(saved_prefix, text_dim)
                current_bias = self.vision_projector.proj.bias.view(current_prefix, text_dim)
                current_bias[:tokens_to_copy].copy_(saved_bias[:tokens_to_copy])

    def load_checkpoint(self, ckpt_path: str, load_value_head: bool = False) -> Dict[str, str]:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            self._load_ts_projector_state(checkpoint)
            return {"source": "ts_projector_state_dict"}

        ts_state = checkpoint.get("ts_projector")
        vision_state = checkpoint.get("vision_projector")
        value_state = checkpoint.get("value_head")

        if ts_state is None and any(
            key.startswith("image_module.") or key.startswith("module.image_module.")
            for key in checkpoint.keys()
        ):
            ts_state = checkpoint

        if ts_state is not None:
            self._load_ts_projector_state(ts_state)
        if vision_state is not None:
            self._load_vision_projector_state(vision_state)
        if load_value_head and self.value_head is not None and isinstance(value_state, dict):
            self.value_head.load_state_dict(_normalize_state_dict(value_state), strict=True)

        return {
            "source": "checkpoint",
            "has_ts_projector": str(ts_state is not None),
            "has_vision_projector": str(vision_state is not None),
            "has_value_head": str(load_value_head and value_state is not None),
        }

    def forward(self, sequences: torch.Tensor, prompts: Sequence[str]) -> torch.Tensor:
        return self.get_action_logits(sequences, prompts)
