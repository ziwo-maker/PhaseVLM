from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover - optional dependency
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - optional dependency
    AutoModelForImageTextToText = None

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    process_vision_info = None

from projection_train.policy import ACTION_LIST, parse_signal, action_to_index


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape [C, H, W].")

    image = image.detach().float().cpu()
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError("Expected 1 or 3 channels for image tensor.")

    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    image = image.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image, mode="RGB")


class Qwen3VLPolicy:
    """Qwen3-VL policy with explicit processor/vision preprocessing."""

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        device_map: Optional[Union[str, dict]] = None,
        max_memory: Optional[Dict[Union[int, str], str]] = None,
        trust_remote_code: bool = True,
        image_patch_size: int = 16,
    ) -> None:
        resolved_device = self._resolve_device(device)
        if resolved_device is None:
            resolved_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = resolved_device
        self.max_length = int(max_length)
        self.device_map = device_map
        self.image_patch_size = int(image_patch_size)

        self.model = None
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if hasattr(self.processor, "tokenizer"):
            try:
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token_id is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            except Exception:
                pass

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        if Qwen3VLForConditionalGeneration is not None:
            model_cls = Qwen3VLForConditionalGeneration
        elif AutoModelForImageTextToText is not None:
            model_cls = AutoModelForImageTextToText
        else:
            model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(model_path, **model_kwargs)
        self.model.config.use_cache = False
        if device_map is None:
            self.model.to(self.device)

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
        if device is None:
            return None
        if isinstance(device, torch.device):
            resolved = device
        else:
            text = str(device).strip().lower()
            if text in ("", "none", "auto", "null"):
                return None
            try:
                resolved = torch.device(text)
            except (TypeError, ValueError):
                return None
        if resolved.type == "cuda":
            if not torch.cuda.is_available():
                return None
            if resolved.index is None:
                resolved = torch.device("cuda:0")
            if resolved.index >= torch.cuda.device_count():
                return None
        return resolved

    def _build_messages(self, prompt: str, image_pil: Optional[Image.Image]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        if image_pil is not None:
            content.append({"type": "image", "image": image_pil})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _prepare_batch_inputs(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
    ) -> Tuple[List[str], List[List[Any]], List[List[Any]], Dict[str, Any]]:
        texts: List[str] = []
        images: List[List[Any]] = []
        videos: List[List[Any]] = []
        video_kwargs_list: List[Dict[str, Any]] = []

        if pixel_values is None:
            pixel_values = torch.empty((len(prompts), 0, 0, 0))

        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)

        for prompt, img_tensor in zip(prompts, pixel_values):
            image_pil = _tensor_to_pil(img_tensor)
            messages = self._build_messages(prompt, image_pil)
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            texts.append(text)

            if process_vision_info is not None:
                try:
                    image_inputs, video_inputs, video_kwargs = process_vision_info(
                        messages,
                        image_patch_size=self.image_patch_size,
                        return_video_kwargs=True,
                        return_video_metadata=True,
                    )
                except TypeError:
                    image_inputs, video_inputs = process_vision_info(messages)
                    video_kwargs = {}
                images.append(image_inputs or [])
                videos.append(video_inputs or [])
                video_kwargs_list.append(video_kwargs or {})
            else:
                images.append([image_pil])
                videos.append([])
                video_kwargs_list.append({})

        merged_video_kwargs: Dict[str, Any] = {}
        if video_kwargs_list:
            keys = set().union(*[vk.keys() for vk in video_kwargs_list])
            for key in keys:
                values = [vk.get(key) for vk in video_kwargs_list if key in vk]
                if values and all(val == values[0] for val in values):
                    merged_video_kwargs[key] = values[0]

        return texts, images, videos, merged_video_kwargs

    def encode_inputs(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        texts, images, videos, video_kwargs = self._prepare_batch_inputs(prompts, pixel_values)

        has_images = any(len(imgs) > 0 for imgs in images)
        has_videos = any(len(vids) > 0 for vids in videos)
        encoding = self.processor(
            text=texts,
            images=images if has_images else None,
            videos=videos if has_videos else None,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            **video_kwargs,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        image_grid_thw = encoding.get("image_grid_thw")
        pixel_values_out = encoding.get("pixel_values")
        return input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out

    def generate_actions(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        generation_kwargs: dict,
    ) -> Tuple[List[int], List[str]]:
        input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out = self.encode_inputs(
            prompts, pixel_values
        )
        if pixel_values_out is not None:
            pixel_values_out = pixel_values_out.to(self.device, dtype=self.model.dtype)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        generate_kwargs = dict(generation_kwargs)
        if image_grid_thw is not None:
            generate_kwargs["image_grid_thw"] = image_grid_thw

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_out,
            **generate_kwargs,
        )

        responses: List[str] = []
        actions: List[int] = []
        for idx, seq in enumerate(output_ids):
            prompt_len = int(lengths[idx])
            gen_ids = seq[prompt_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response)
            signal = parse_signal(response)
            actions.append(action_to_index(signal))
        return actions, responses

    def action_logprob(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        actions: List[Union[int, str]],
    ) -> torch.Tensor:
        input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out = self.encode_inputs(
            prompts, pixel_values
        )
        if pixel_values_out is not None:
            pixel_values_out = pixel_values_out.to(self.device, dtype=self.model.dtype)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        pad_id = int(self.tokenizer.pad_token_id)
        tag_ids_list = []
        for action in actions:
            if isinstance(action, str):
                idx = action_to_index(action)
            else:
                try:
                    idx = int(action)
                except (TypeError, ValueError):
                    idx = 0
            if idx < 0 or idx >= len(ACTION_LIST):
                idx = 0
            signal = ACTION_LIST[idx]
            tag_ids = self.tokenizer(
                f"<signal>{signal}</signal>",
                add_special_tokens=False,
            ).input_ids
            tag_ids_list.append(tag_ids)

        max_len = max(lengths[i] + len(tag_ids_list[i]) for i in range(len(actions)))
        batch_size = input_ids.size(0)
        full_ids = torch.full(
            (batch_size, max_len),
            pad_id,
            device=self.device,
            dtype=torch.long,
        )
        full_mask = torch.zeros((batch_size, max_len), device=self.device, dtype=attention_mask.dtype)

        for i in range(batch_size):
            prompt_len = int(lengths[i])
            tag_ids = tag_ids_list[i]
            total_len = prompt_len + len(tag_ids)
            full_ids[i, :prompt_len] = input_ids[i, :prompt_len]
            full_ids[i, prompt_len:total_len] = torch.tensor(tag_ids, device=self.device)
            full_mask[i, :total_len] = 1

        forward_kwargs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "pixel_values": pixel_values_out,
        }
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        outputs = self.model(**forward_kwargs)
        logits = outputs.logits
        logsumexp = torch.logsumexp(logits, dim=-1)
        logp = torch.zeros(batch_size, device=self.device, dtype=logits.dtype)
        for i in range(batch_size):
            prompt_len = int(lengths[i])
            tag_ids = tag_ids_list[i]
            for offset, token_id in enumerate(tag_ids):
                pos = prompt_len + offset
                if pos == 0:
                    continue
                logp[i] = logp[i] + (logits[i, pos - 1, token_id] - logsumexp[i, pos - 1])
        return logp

    def get_action_logits(self, prompts: List[str], images: torch.Tensor) -> torch.Tensor:
        batch_size = len(prompts)
        num_actions = len(ACTION_LIST)
        all_logprobs = []
        for action_idx in range(num_actions):
            action = ACTION_LIST[action_idx]
            actions = [action] * batch_size
            logprobs = self.action_logprob(prompts, images, actions)
            all_logprobs.append(logprobs)
        logits = torch.stack(all_logprobs, dim=-1)
        return logits

    def compute_entropy(self, prompts: List[str], images: torch.Tensor) -> torch.Tensor:
        logits = self.get_action_logits(prompts, images)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
