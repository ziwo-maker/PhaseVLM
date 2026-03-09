from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models import Qwen3VLBackbone, TimeSeriesProjector, VisionPrefixProjector
from projection_train.action_space import ACTION_LIST, action_to_index


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
    if isinstance(state, dict):
        module.load_state_dict(state, strict=False)
        return True
    if isinstance(checkpoint.get("state_dict"), dict):
        nested = _strip_state_prefix(checkpoint["state_dict"], f"{key}.")
        if nested:
            module.load_state_dict(nested, strict=False)
            return True
    nested = _strip_state_prefix(checkpoint, f"{key}.")
    if nested:
        module.load_state_dict(nested, strict=False)
        return True
    module_prefix = _strip_state_prefix(checkpoint, f"module.{key}.")
    if module_prefix:
        module.load_state_dict(module_prefix, strict=False)
        return True
    return False


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
    ) -> None:
        super().__init__()
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.seq_len = int(seq_len)
        self.input_dim = int(input_dim)
        self.with_value_head = bool(with_value_head)

        self.backbone = Qwen3VLBackbone(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device=self.device,
            max_length=self.max_length,
        )
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

    def _encode_sequences(self, sequences: torch.Tensor) -> Dict[str, torch.Tensor]:
        if sequences.device != self.device:
            sequences = sequences.to(self.device)
        images = self.ts_projector(sequences)
        vision_tokens = self.backbone.encode_vision_tokens(images)
        if vision_tokens.ndim == 2:
            vision_tokens = vision_tokens.unsqueeze(1)
        vision_pooled = vision_tokens.mean(dim=1)
        prefix_vec, prefix_tokens = self.vision_projector(vision_pooled)
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
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
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
    ) -> Dict[str, Any]:
        seq_features = self._encode_sequences(sequences)
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

    def _forward_lm(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        forward_kwargs = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": False,
            "return_dict": True,
        }
        try:
            return self.backbone.model(**forward_kwargs)
        except TypeError:
            forward_kwargs.pop("position_ids", None)
            return self.backbone.model(**forward_kwargs)

    def action_tag_logprob_from_prepared(
        self,
        prepared_batch: Dict[str, Any],
        actions: Sequence[int],
    ) -> torch.Tensor:
        tagged = self._build_tagged_inputs(prepared_batch, actions)
        outputs = self._forward_lm(
            inputs_embeds=tagged["inputs_embeds"],
            attention_mask=tagged["attention_mask"],
        )
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        batch_size = logits.shape[0]
        max_tag_len = int(tagged["tag_token_ids"].shape[1])
        batch_idx = torch.arange(batch_size, device=logits.device).view(-1, 1)
        pos_offset = torch.arange(max_tag_len, device=logits.device).view(1, -1)
        gather_pos = tagged["tag_start_pos"].view(-1, 1) + pos_offset - 1
        gather_pos = gather_pos.clamp_min(0)
        chosen = log_probs[batch_idx, gather_pos, tagged["tag_token_ids"]]
        mask = pos_offset < tagged["tag_lens"].view(-1, 1)
        return (chosen * mask).sum(dim=1)

    def get_action_logits_from_prepared(self, prepared_batch: Dict[str, Any]) -> torch.Tensor:
        all_scores = []
        for action_idx in range(len(ACTION_LIST)):
            actions = [action_idx] * prepared_batch["prefix_tokens"].shape[0]
            score = self.action_tag_logprob_from_prepared(prepared_batch, actions)
            all_scores.append(score)
        return torch.stack(all_scores, dim=-1)

    def get_action_logits(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        prepared = prepared_batch or self.prepare_policy_batch(sequences, prompts)
        return self.get_action_logits_from_prepared(prepared)

    def forward(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        prepared_batch: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        return self.get_action_logits(sequences, prompts, prepared_batch=prepared_batch)

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
        params: List[nn.Parameter] = []
        params.extend(list(self.ts_projector.parameters()))
        params.extend(list(self.vision_projector.parameters()))
        if self.value_head is not None:
            params.extend(list(self.value_head.parameters()))
        return params

    def load_checkpoint(
        self,
        checkpoint_path: str,
        load_value_head: bool = True,
    ) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        _maybe_load_module(self.ts_projector, checkpoint, "ts_projector", load=True)
        _maybe_load_module(self.vision_projector, checkpoint, "vision_projector", load=True)
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
