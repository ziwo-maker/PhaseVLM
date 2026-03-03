from functools import partial
import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# 可选依赖：根据不同模型族/多模态实现按需加载，失败时降级到可用实现。
try:
    from transformers import Qwen3VLForConditionalGeneration
except Exception:  # pragma: no cover - optional dependency
    Qwen3VLForConditionalGeneration = None

try:
    from transformers import Gemma3ForConditionalGeneration
except Exception:  # pragma: no cover - optional dependency
    Gemma3ForConditionalGeneration = None

try:
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - optional dependency
    AutoModelForImageTextToText = None

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    process_vision_info = None

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        transformer_auto_wrap_policy,
    )
except Exception:  # pragma: no cover - optional dependency
    FSDP = None
    MixedPrecision = None
    ShardingStrategy = None
    size_based_auto_wrap_policy = None
    transformer_auto_wrap_policy = None

from projection_train.policy import ACTION_LIST, action_to_index


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """将 [C, H, W] 或 [1, C, H, W] 的 Tensor 转成 PIL.Image。"""
    if image.ndim == 4:
        # 批次输入仅取第一张
        image = image[0]
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape [C, H, W].")
    # 统一到 CPU/float，避免带梯度或不支持的 dtype
    image = image.detach().float().cpu()
    if image.shape[0] == 1:
        # 灰度图扩展成 3 通道
        image = image.repeat(3, 1, 1)
    if image.shape[0] != 3:
        raise ValueError("Expected 1 or 3 channels for image tensor.")
    # 归一化到 [0, 1] 并量化到 0~255 的 uint8
    image = image.clamp(0.0, 1.0)
    image = (image * 255.0).round().to(torch.uint8)
    # CHW -> HWC
    image = image.permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(image, mode="RGB")


class Qwen3VLPolicy:
    """Unified VLM/Text policy wrapper for Qwen3-VL and Gemma family."""

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
        enforce_action_format: bool = True,
        fsdp_enabled: bool = False,
        fsdp_sharding_strategy: str = "full_shard",
    ) -> None:
        # 解析设备；若未指定则优先使用 CUDA
        resolved_device = self._resolve_device(device)
        if resolved_device is None:
            resolved_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = resolved_device
        self.max_length = int(max_length)
        self.image_patch_size = int(image_patch_size)
        self.enforce_action_format = bool(enforce_action_format)
        self.model = None
        self.is_fsdp = False

        config = None
        try:
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        except Exception:
            config = None
        # 推断模型族，决定是否走多模态路径
        self.model_family = self._infer_model_family(model_path, config)
        self.is_multimodal = self.model_family in {"qwen3_vl", "gemma3_vl"}

        if self.is_multimodal:
            # 多模态模型通常需要 processor 统一处理图文输入
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
            )
            self.tokenizer = getattr(self.processor, "tokenizer", None)
        else:
            self.processor = None
            self.tokenizer = None

        if self.tokenizer is None:
            # 纯文本或 processor 不包含 tokenizer 的场景
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache tokenized action tags to avoid repeated tokenizer calls during PPO.
        # 预先缓存动作 tag 的 token id，加速 PPO 中的频繁计算
        self._action_tag_token_ids: List[List[int]] = []
        self._action_tag_token_tensors: List[torch.Tensor] = []
        for signal in ACTION_LIST:
            tag_ids = self.tokenizer(
                f"<signal>{signal}</signal>",
                add_special_tokens=False,
            ).input_ids
            self._action_tag_token_ids.append(tag_ids)
            self._action_tag_token_tensors.append(
                torch.tensor(tag_ids, device=self.device, dtype=torch.long)
            )

        if self.processor is not None and hasattr(self.processor, "tokenizer"):
            try:
                # 对齐 processor 内部 tokenizer 的 padding 设置
                self.processor.tokenizer.padding_side = "left"
                if self.processor.tokenizer.pad_token_id is None:
                    self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            except Exception:
                pass

        if fsdp_enabled and device_map is not None:
            # FSDP 与 device_map 同时使用会冲突，这里强制禁用 device_map
            print("Warning: FSDP enabled, ignore device_map to avoid conflict.")
            device_map = None

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        # 选择合适的模型类并加载权重
        model_cls = self._select_model_class(self.model_family)
        self.model = model_cls.from_pretrained(model_path, **model_kwargs)
        self.model.config.use_cache = False

        if fsdp_enabled:
            self._maybe_wrap_fsdp(torch_dtype=torch_dtype, sharding=fsdp_sharding_strategy)
        # If FSDP was requested but not actually enabled (e.g. no dist init),
        # fall back to the single-device move.
        if not self.is_fsdp and device_map is None:
            self.model.to(self.device)

        print(
            f"[LLM Load] family={self.model_family}, multimodal={self.is_multimodal}, fsdp={self.is_fsdp}"
        )

    @staticmethod
    def _resolve_device(device: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
        # 支持 "auto"/"none" 等空值写法，并验证 CUDA 设备有效性
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

    @staticmethod
    def _infer_model_family(model_path: str, config: Optional[Any]) -> str:
        # 通过模型路径 + config 中的 model_type/architectures 猜测模型族
        text = str(model_path or "").lower()
        model_type = str(getattr(config, "model_type", "") or "").lower()
        archs = [str(x).lower() for x in (getattr(config, "architectures", None) or [])]
        hint = " ".join([model_type] + archs + [text])

        if "qwen3_vl" in hint or "qwen3-vl" in hint or "qwen-vl" in hint:
            return "qwen3_vl"
        if "gemma3" in hint and ("vision" in hint or "conditionalgeneration" in hint or "vl" in hint):
            return "gemma3_vl"
        if "gemma" in hint:
            return "gemma_text"
        return "auto"

    @staticmethod
    def _select_model_class(model_family: str):
        # 不同模型族对应的加载类；优先使用专用多模态类，降级到通用因果 LM
        if model_family == "qwen3_vl":
            if Qwen3VLForConditionalGeneration is not None:
                return Qwen3VLForConditionalGeneration
            if AutoModelForImageTextToText is not None:
                return AutoModelForImageTextToText
            return AutoModelForCausalLM
        if model_family == "gemma3_vl":
            if Gemma3ForConditionalGeneration is not None:
                return Gemma3ForConditionalGeneration
            if AutoModelForImageTextToText is not None:
                return AutoModelForImageTextToText
            return AutoModelForCausalLM
        if model_family == "gemma_text":
            return AutoModelForCausalLM
        if AutoModelForImageTextToText is not None:
            return AutoModelForImageTextToText
        return AutoModelForCausalLM

    def _maybe_wrap_fsdp(self, torch_dtype: torch.dtype, sharding: str) -> None:
        # FSDP 只在分布式/可用 CUDA 时启用
        if (
            FSDP is None
            or ShardingStrategy is None
            or transformer_auto_wrap_policy is None
            or size_based_auto_wrap_policy is None
        ):
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

        # Keep base model on CPU before FSDP wrapping; avoid accidental full-GPU materialization.
        self.model = self.model.cpu()

        mp_policy = None
        if MixedPrecision is not None:
            # FSDP 混合精度策略
            mp_policy = MixedPrecision(
                param_dtype=torch_dtype,
                reduce_dtype=torch_dtype,
                buffer_dtype=torch_dtype,
            )

        # 优先按 transformer block 自动切分，找不到则按大模块切分
        block_classes = self._collect_transformer_block_classes(self.model)
        if block_classes:
            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=tuple(block_classes),
            )
            class_names = ", ".join(sorted(cls.__name__ for cls in block_classes))
            print(f"[FSDP] transformer auto wrap classes: {class_names}")
        else:
            # Fallback: wrap large submodules only, to avoid one-shot flatten over whole model.
            auto_wrap_policy = partial(size_based_auto_wrap_policy, min_num_params=20_000_000)
            print("[FSDP] transformer blocks not detected; fallback to size-based auto wrap.")

        self.model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mp_policy,
            auto_wrap_policy=auto_wrap_policy,
            use_orig_params=True,
            sync_module_states=True,
            device_id=self.device,
            limit_all_gathers=True,
        )
        self.is_fsdp = True

    @staticmethod
    def _collect_transformer_block_classes(model: torch.nn.Module) -> Set[type]:
        # 通过类名关键字收集 block 类，用于 FSDP 自动 wrap
        cls_set: Set[type] = set()
        block_name_keywords = (
            "decoderlayer",
            "decoder_layer",
            "transformerblock",
            "transformer_block",
            "gemmadecoderlayer",
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
                # Skip top container classes to keep wrap granularity at block/layer level.
                if len(list(module.children())) == 0:
                    continue
                cls_set.add(cls)
        return cls_set

    def _augment_prompt(self, prompt: str) -> str:
        # 可选地强制输出固定 tag 格式，便于解析动作
        if not self.enforce_action_format:
            return prompt
        return (
            f"{prompt}\n\n"
            "Output exactly one tag and nothing else.\n"
            "Allowed tags: <signal>ETWT</signal>, <signal>NTST</signal>, "
            "<signal>ELWL</signal>, <signal>NLSL</signal>."
        )

    def _build_messages(self, prompt: str, image_pil: Optional[Image.Image]) -> List[Dict[str, Any]]:
        # 构建多模态 messages，供 processor/chat template 使用
        content: List[Dict[str, Any]] = []
        if image_pil is not None:
            if self.model_family == "qwen3_vl":
                content.append({"type": "image", "image": image_pil})
            else:
                content.append({"type": "image"})
        content.append({"type": "text", "text": self._augment_prompt(prompt)})
        return [{"role": "user", "content": content}]

    def _build_messages_template(self, prompt: str, include_image: bool) -> List[Dict[str, Any]]:
        # 构建仅用于模板的 messages（图像占位）
        content: List[Dict[str, Any]] = []
        if include_image:
            # For chat template, we only need the image placeholder.
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _format_prompt_with_template(self, prompt: str, include_image: bool) -> Tuple[str, bool]:
        # 尝试使用 processor/tokenizer 的 chat template，失败则回退原始 prompt
        prompt = self._augment_prompt(prompt)
        if include_image and self.processor is not None and hasattr(self.processor, "apply_chat_template"):
            try:
                messages = self._build_messages_template(prompt, include_image=True)
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return text, True
            except Exception:
                pass
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                content = prompt
                if include_image:
                    # Keep image placeholder for multimodal prompts.
                    messages = self._build_messages_template(prompt, include_image=True)
                    content = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    content = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                return content, True
            except Exception:
                pass
        return prompt, False

    def _prepare_texts(self, prompts: List[str], include_image: bool) -> Tuple[List[str], bool]:
        # 批量应用模板并标记是否使用了模板
        texts: List[str] = []
        used_template = False
        for prompt in prompts:
            text, used = self._format_prompt_with_template(prompt, include_image)
            texts.append(text)
            used_template = used_template or used
        return texts, used_template

    def _prepare_text_only_inputs(
        self,
        prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 文本-only 输入编码（用于非多模态模型或纯文本推理）
        texts: List[str] = []
        for prompt in prompts:
            prompt = self._augment_prompt(prompt)
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    prompt = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    pass
            texts.append(prompt)
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, lengths, None, None

    def _prepare_batch_inputs(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
    ) -> Tuple[List[str], List[List[Any]], List[List[Any]], Dict[str, Any]]:
        # 多模态输入：将每个样本的图像 tensor -> PIL，再交给 processor 处理
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

            if self.model_family == "qwen3_vl" and process_vision_info is not None:
                try:
                    # Qwen3-VL 使用专用的视觉信息处理，返回图像/视频及额外参数
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
            # 将各样本一致的 video_kwargs 合并成 batch 级参数
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
        # processor 路径：依赖 HF 处理器生成 input_ids / pixel_values
        if not self.is_multimodal:
            return self._prepare_text_only_inputs(prompts)

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
            add_special_tokens=False,
            **video_kwargs,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        image_grid_thw = encoding.get("image_grid_thw")
        pixel_values_out = encoding.get("pixel_values")
        return input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out

    def _prepare_pixel_values_tensor(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 纯 tensor 路径：不依赖 processor 的 image_processor，自己完成归一化/patchify
        if pixel_values is None:
            return None, None
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.ndim != 4:
            raise ValueError("Expected pixel_values with shape [B, C, H, W].")

        # Ensure float and 3-channel
        if pixel_values.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            pixel_values = pixel_values.float()
        if pixel_values.size(1) == 1:
            pixel_values = pixel_values.repeat(1, 3, 1, 1)

        # 放到目标 device，后续操作保持同设备
        pixel_values = pixel_values.to(self.device)

        # Normalize using processor stats when available.
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        image_processor = getattr(self.processor, "image_processor", None) if self.processor is not None else None
        if image_processor is not None:
            mean = list(getattr(image_processor, "image_mean", mean))
            std = list(getattr(image_processor, "image_std", std))
        mean_t = torch.tensor(mean, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        std_t = torch.tensor(std, device=pixel_values.device, dtype=pixel_values.dtype).view(1, 3, 1, 1)
        pixel_values = (pixel_values - mean_t) / std_t

        # Resolve vision config (Qwen2/3-VL style).
        model_cfg = getattr(self.model.module, "config", None) if self.is_fsdp else getattr(self.model, "config", None)
        vision_cfg = getattr(model_cfg, "vision_config", None)
        patch_size = int(getattr(vision_cfg, "patch_size", self.image_patch_size))
        temporal_patch_size = int(getattr(vision_cfg, "temporal_patch_size", 2))
        merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))

        # Resize with smart_resize (differentiable), as in HF image processor.
        height, width = pixel_values.shape[-2:]
        factor = patch_size * merge_size
        image_processor = getattr(self.processor, "image_processor", None) if self.processor is not None else None
        min_pixels = int(getattr(image_processor, "size", {}).get("shortest_edge", factor * factor))
        max_pixels = int(getattr(image_processor, "size", {}).get("longest_edge", height * width))
        try:
            # 根据目标像素范围自适应缩放
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

        # Ensure divisibility by factor.
        height, width = pixel_values.shape[-2:]
        pad_h = (factor - height % factor) % factor
        pad_w = (factor - width % factor) % factor
        if pad_h or pad_w:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h))

        # Patchify exactly like HF Qwen2/3-VL image processor.
        batch_size, channels, height, width = pixel_values.shape
        grid_h = height // patch_size
        grid_w = width // patch_size
        patches_list = []
        grid_thw_list = []
        for idx in range(batch_size):
            # 这里按“视频帧”的维度处理，单张图像视作 T=1
            frames = pixel_values[idx : idx + 1]  # [T, C, H, W], T=1 for image
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

        pixel_values_out = torch.cat(patches_list, dim=0).to(self.device, dtype=self._model_dtype())
        image_grid_thw = torch.tensor(
            grid_thw_list,
            device=self.device,
            dtype=torch.long,
        )
        return pixel_values_out, image_grid_thw

    def _expand_image_tokens(
        self,
        texts: List[str],
        image_grid_thw: Optional[torch.Tensor],
    ) -> List[str]:
        # 将占位符 image_token 展开成与视觉 patch 数量匹配的 token 数
        if image_grid_thw is None or not texts:
            return texts

        image_token = None
        if self.processor is not None and hasattr(self.processor, "image_token"):
            image_token = getattr(self.processor, "image_token")
        if image_token is None and hasattr(self.tokenizer, "image_token"):
            image_token = getattr(self.tokenizer, "image_token")
        if image_token is None:
            image_token = "<|image_pad|>"

        if not any(image_token in text for text in texts):
            if any("<image>" in text for text in texts):
                image_token = "<image>"
            else:
                return texts

        merge_size = None
        if self.processor is not None:
            image_processor = getattr(self.processor, "image_processor", None)
            merge_size = getattr(image_processor, "merge_size", None)
        if merge_size is None:
            model_cfg = getattr(self.model.module, "config", None) if self.is_fsdp else getattr(self.model, "config", None)
            vision_cfg = getattr(model_cfg, "vision_config", None)
            merge_size = int(getattr(vision_cfg, "spatial_merge_size", 2))
        merge_length = int(merge_size) ** 2

        placeholder = "<|placeholder|>"
        grid_list = image_grid_thw.tolist() if torch.is_tensor(image_grid_thw) else image_grid_thw
        image_index = 0
        expanded_texts: List[str] = []
        for text in texts:
            while image_token in text:
                if image_index >= len(grid_list):
                    break
                grid_t, grid_h, grid_w = grid_list[image_index]
                num_image_tokens = int(grid_t * grid_h * grid_w // merge_length)
                if num_image_tokens <= 0:
                    num_image_tokens = 1
                text = text.replace(image_token, placeholder * num_image_tokens, 1)
                image_index += 1
            expanded_texts.append(text.replace(placeholder, image_token))
        return expanded_texts

    def encode_inputs_tensor(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], Optional[torch.Tensor], Optional[torch.Tensor]]:
        # 纯 tensor 路径：自己处理图像 patch + tokenizer 编码
        texts, used_template = self._prepare_texts(prompts, include_image=True)
        pixel_values_out, image_grid_thw = self._prepare_pixel_values_tensor(pixel_values)
        texts = self._expand_image_tokens(texts, image_grid_thw)
        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=not used_template,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out

    def _model_dtype(self) -> torch.dtype:
        # 兼容 FSDP 包装后的 model.dtype
        if self.is_fsdp and hasattr(self.model, "module"):
            return getattr(self.model.module, "dtype", torch.float32)
        return getattr(self.model, "dtype", torch.float32)

    def _get_base_model(self) -> Optional[torch.nn.Module]:
        if self.model is None:
            return None
        return self.model.module if hasattr(self.model, "module") else self.model

    def set_use_cache(self, enabled: bool) -> None:
        model = self._get_base_model()
        if model is None:
            return
        config = getattr(model, "config", None)
        if config is not None:
            config.use_cache = bool(enabled)

    def set_gradient_checkpointing(self, enabled: bool) -> None:
        model = self._get_base_model()
        if model is None:
            return
        try:
            if enabled:
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
        except Exception as exc:
            action = "enable" if enabled else "disable"
            print(f"Warning: failed to {action} gradient checkpointing: {exc}")

    def _get_visual_module(self) -> Optional[torch.nn.Module]:
        # 提取视觉模块（若存在）
        model = self.model.module if hasattr(self.model, "module") else self.model
        return getattr(model, "visual", None)

    @staticmethod
    def _smart_resize(
        height: int,
        width: int,
        factor: int,
        min_pixels: int,
        max_pixels: int,
    ) -> tuple[int, int]:
        # 与 HF image processor 的 smart_resize 逻辑保持一致
        # Matches Qwen2/3-VL smart_resize behavior from HF image processor.
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

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_p: float,
    ) -> torch.Tensor:
        # 采样/贪心选择下一个 token
        if not do_sample:
            return torch.argmax(logits, dim=-1)

        temp = max(float(temperature), 1e-5)
        probs = torch.softmax(logits / temp, dim=-1)

        if 0.0 < float(top_p) < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            keep_mask = cumsum_probs <= float(top_p)
            keep_mask[..., 0] = True
            filtered_probs = torch.where(keep_mask, sorted_probs, torch.zeros_like(sorted_probs))
            filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled_sorted = torch.multinomial(filtered_probs, num_samples=1).squeeze(-1)
            return sorted_idx.gather(dim=-1, index=sampled_sorted.unsqueeze(-1)).squeeze(-1)

        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    @staticmethod
    def _normalize_eos_ids(eos_token_id: Optional[Union[int, List[int], Tuple[int, ...]]]) -> List[int]:
        # 统一 eos_token_id 为 List[int]
        if eos_token_id is None:
            return []
        if isinstance(eos_token_id, int):
            return [eos_token_id]
        return [int(x) for x in eos_token_id]

    def _forward(self, **kwargs):
        # 统一 forward 入口，便于子类/封装扩展
        return self.model(**kwargs)

    @staticmethod
    def _is_inference_tensor(tensor: Optional[torch.Tensor]) -> bool:
        if tensor is None:
            return False
        torch_is_inference = getattr(torch, "is_inference", None)
        if callable(torch_is_inference):
            try:
                return bool(torch_is_inference(tensor))
            except Exception:
                pass
        attr = getattr(tensor, "is_inference", None)
        if callable(attr):
            try:
                return bool(attr())
            except Exception:
                return False
        if isinstance(attr, bool):
            return attr
        return False

    def _ensure_non_inference_tensor(
        self, tensor: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if self._is_inference_tensor(tensor):
            # Clone to get a normal tensor for autograd.
            return tensor.clone()
        return tensor

    def prepare_inputs(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        use_tensor_preprocess: Optional[bool] = None,
    ) -> Dict[str, Any]:
        # 根据模型类型与是否需要梯度选择输入准备路径
        if not self.is_multimodal:
            input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out = self._prepare_text_only_inputs(
                prompts
            )
        else:
            if use_tensor_preprocess is None:
                use_tensor_preprocess = torch.is_grad_enabled()
            if use_tensor_preprocess:
                input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out = self.encode_inputs_tensor(
                    prompts, pixel_values
                )
            else:
                input_ids, attention_mask, lengths, image_grid_thw, pixel_values_out = self.encode_inputs(
                    prompts, pixel_values
                )
                if pixel_values_out is not None:
                    pixel_values_out = pixel_values_out.to(self.device, dtype=self._model_dtype())
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.to(self.device)
        if torch.is_grad_enabled():
            pixel_values_out = self._ensure_non_inference_tensor(pixel_values_out)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "lengths": lengths,
            "image_grid_thw": image_grid_thw,
            "pixel_values_out": pixel_values_out,
        }

    def generate_actions_from_prepared(
        self,
        prepared_inputs: Dict[str, Any],
        generation_kwargs: dict,
    ) -> Tuple[List[int], List[str]]:
        # 使用已准备好的输入进行生成，并解析为动作
        input_ids = prepared_inputs["input_ids"]
        attention_mask = prepared_inputs["attention_mask"]
        image_grid_thw = prepared_inputs["image_grid_thw"]
        pixel_values_out = prepared_inputs["pixel_values_out"]

        generate_kwargs = dict(generation_kwargs or {})
        generate_kwargs["use_cache"] = bool(generate_kwargs.get("use_cache", True))

        if self.is_fsdp:
            # FSDP 下使用手动 token loop，避免 generate() 与分布式冲突
            max_new_tokens = int(generate_kwargs.get("max_new_tokens", 128))
            do_sample = bool(generate_kwargs.get("do_sample", True))
            temperature = float(generate_kwargs.get("temperature", 1.0))
            top_p = float(generate_kwargs.get("top_p", 1.0))

            model_for_cfg = self.model.module if hasattr(self.model, "module") else self.model
            model_cfg = getattr(model_for_cfg, "config", None)
            eos_ids = self._normalize_eos_ids(
                generate_kwargs.get(
                    "eos_token_id",
                    getattr(model_cfg, "eos_token_id", self.tokenizer.eos_token_id),
                )
            )
            eos_set = set(eos_ids)

            pad_token_id = generate_kwargs.get("pad_token_id", self.tokenizer.pad_token_id)
            if pad_token_id is None:
                pad_token_id = self.tokenizer.eos_token_id
            if pad_token_id is None:
                pad_token_id = 0
            pad_token_id = int(pad_token_id)

            # 逐步生成，自己维护缓存与 attention_mask
            cur_input_ids = input_ids
            finished = torch.zeros(cur_input_ids.size(0), dtype=torch.bool, device=self.device)
            generated_chunks: List[torch.Tensor] = []
            model_for_prepare = self.model.module if hasattr(self.model, "module") else self.model
            model_kwargs: Dict[str, Any] = {
                "past_key_values": None,
                "attention_mask": attention_mask,
                "use_cache": True,
                "cache_position": torch.arange(cur_input_ids.size(1), device=self.device, dtype=torch.long),
            }
            if pixel_values_out is not None:
                model_kwargs["pixel_values"] = pixel_values_out
            if image_grid_thw is not None:
                model_kwargs["image_grid_thw"] = image_grid_thw

            with torch.inference_mode():
                for _ in range(max_new_tokens):
                    if hasattr(model_for_prepare, "prepare_inputs_for_generation"):
                        forward_kwargs = model_for_prepare.prepare_inputs_for_generation(
                            cur_input_ids,
                            **model_kwargs,
                        )
                    else:
                        # 兼容不提供 prepare_inputs_for_generation 的模型
                        forward_kwargs = {
                            "input_ids": cur_input_ids,
                            "attention_mask": model_kwargs["attention_mask"],
                            "past_key_values": model_kwargs.get("past_key_values"),
                            "use_cache": True,
                        }
                        if model_kwargs.get("cache_position") is not None:
                            forward_kwargs["cache_position"] = model_kwargs["cache_position"]
                        if model_kwargs.get("pixel_values") is not None:
                            forward_kwargs["pixel_values"] = model_kwargs["pixel_values"]
                        if model_kwargs.get("image_grid_thw") is not None:
                            forward_kwargs["image_grid_thw"] = model_kwargs["image_grid_thw"]

                    outputs = self._forward(**forward_kwargs)
                    model_kwargs["past_key_values"] = getattr(outputs, "past_key_values", None)
                    next_logits = outputs.logits[:, -1, :]
                    sampled_tokens = self._sample_next_token(
                        next_logits,
                        do_sample=do_sample,
                        temperature=temperature,
                        top_p=top_p,
                    )

                    prev_finished = finished
                    next_tokens = torch.where(
                        prev_finished,
                        torch.full_like(sampled_tokens, pad_token_id),
                        sampled_tokens,
                    )

                    if eos_set:
                        # 命中 eos 的样本标记完成
                        reached_eos = torch.zeros_like(prev_finished)
                        for eos_id in eos_set:
                            reached_eos = reached_eos | ((next_tokens == eos_id) & (~prev_finished))
                        finished = prev_finished | reached_eos

                    step_mask = (~prev_finished).to(model_kwargs["attention_mask"].dtype).unsqueeze(1)
                    model_kwargs["attention_mask"] = torch.cat([model_kwargs["attention_mask"], step_mask], dim=1)
                    if model_kwargs.get("cache_position") is not None:
                        model_kwargs["cache_position"] = model_kwargs["cache_position"][-1:] + 1
                    else:
                        model_kwargs["cache_position"] = torch.tensor(
                            [model_kwargs["attention_mask"].shape[1] - 1],
                            device=self.device,
                            dtype=torch.long,
                        )
                    generated_chunks.append(next_tokens.unsqueeze(1))
                    cur_input_ids = next_tokens.unsqueeze(1)

                    if eos_set:
                        # 分布式场景下需要全局同步是否已完成
                        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                            all_done = torch.tensor(
                                int(torch.all(finished).item()),
                                device=self.device,
                                dtype=torch.int32,
                            )
                            dist.all_reduce(all_done, op=dist.ReduceOp.MIN)
                            if int(all_done.item()) == 1:
                                break
                        elif torch.all(finished):
                            break

            if generated_chunks:
                generated_ids = torch.cat(generated_chunks, dim=1)
            else:
                generated_ids = input_ids.new_zeros((input_ids.size(0), 0))
        else:
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                generate_kwargs["synced_gpus"] = True
            generate_inputs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                **generate_kwargs,
            }
            if pixel_values_out is not None:
                generate_inputs["pixel_values"] = pixel_values_out
            if image_grid_thw is not None:
                generate_inputs["image_grid_thw"] = image_grid_thw

            with torch.inference_mode():
                if hasattr(self.model, "generate"):
                    # 标准 HF generate 路径
                    output_ids = self.model.generate(**generate_inputs)
                elif hasattr(self.model, "module") and hasattr(self.model.module, "generate"):
                    output_ids = self.model.module.generate(**generate_inputs)
                else:
                    raise AttributeError("Current model does not expose generate().")

            prompt_width = int(input_ids.shape[1])
            generated_ids = output_ids[:, prompt_width:]

        # 将生成的文本解析为动作 id
        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        actions: List[int] = []
        for response in responses:
            matches = re.findall(r"<signal>(.*?)</signal>", response)
            signal = matches[-1].strip() if matches else ACTION_LIST[0]
            actions.append(action_to_index(signal))

        return actions, responses

    def generate_actions(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        generation_kwargs: dict,
    ) -> Tuple[List[int], List[str]]:
        # 组合接口：先准备输入再生成
        prepared_inputs = self.prepare_inputs(prompts, pixel_values)
        return self.generate_actions_from_prepared(prepared_inputs, generation_kwargs)

    def action_logprob_from_prepared(
        self,
        prepared_inputs: Dict[str, Any],
        actions: List[Union[int, str]],
    ) -> torch.Tensor:
        # 计算给定动作 tag 的对数概率（按 tag token 求和）
        input_ids = prepared_inputs["input_ids"]
        attention_mask = prepared_inputs["attention_mask"]
        image_grid_thw = prepared_inputs["image_grid_thw"]
        pixel_values_out = prepared_inputs["pixel_values_out"]
        input_width = int(input_ids.shape[1])

        pad_id = int(self.tokenizer.pad_token_id)
        tag_ids_list = []
        tag_tensor_list = []
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
            tag_ids = self._action_tag_token_ids[idx]
            tag_ids_list.append(tag_ids)
            tag_tensor_list.append(self._action_tag_token_tensors[idx])

        # 拼接 prompt + action tag，形成完整输入序列
        max_len = input_width + max(len(tag_ids) for tag_ids in tag_ids_list)
        batch_size = input_ids.size(0)
        full_ids = torch.full(
            (batch_size, max_len),
            pad_id,
            device=self.device,
            dtype=torch.long,
        )
        full_mask = torch.zeros((batch_size, max_len), device=self.device, dtype=attention_mask.dtype)

        for i in range(batch_size):
            tag_ids = tag_ids_list[i]
            total_len = input_width + len(tag_ids)
            full_ids[i, :input_width] = input_ids[i, :input_width]
            full_ids[i, input_width:total_len] = tag_tensor_list[i]
            full_mask[i, :input_width] = attention_mask[i, :input_width]
            full_mask[i, input_width:total_len] = 1

        forward_kwargs = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
        }
        if pixel_values_out is not None:
            forward_kwargs["pixel_values"] = pixel_values_out
        if image_grid_thw is not None:
            forward_kwargs["image_grid_thw"] = image_grid_thw

        outputs = self._forward(**forward_kwargs)
        logits = outputs.logits
        if input_width <= 0:
            return torch.zeros(batch_size, device=self.device, dtype=logits.dtype)

        # 只取 action tag 位置的 log prob 并累加
        max_tag_len = max(len(tag_ids) for tag_ids in tag_ids_list)
        tag_ids_tensor = torch.full(
            (batch_size, max_tag_len),
            fill_value=pad_id,
            device=logits.device,
            dtype=torch.long,
        )
        tag_lens = torch.tensor(
            [len(tag_ids) for tag_ids in tag_ids_list],
            device=logits.device,
            dtype=torch.long,
        )
        for i, tag_ids in enumerate(tag_ids_list):
            tag_ids_tensor[i, : len(tag_ids)] = torch.tensor(tag_ids, device=logits.device, dtype=torch.long)

        log_probs = F.log_softmax(logits, dim=-1)
        base = input_width - 1
        pos_idx = base + torch.arange(max_tag_len, device=logits.device).view(1, -1)
        pos_idx = pos_idx.expand(batch_size, -1)
        batch_idx = torch.arange(batch_size, device=logits.device).view(-1, 1)
        log_probs_at_pos = log_probs[batch_idx, pos_idx, tag_ids_tensor]

        mask = (
            torch.arange(max_tag_len, device=logits.device).view(1, -1)
            < tag_lens.view(-1, 1)
        )
        logp = (log_probs_at_pos * mask).sum(dim=1)
        return logp

    def action_logprob(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        actions: List[Union[int, str]],
    ) -> torch.Tensor:
        # 组合接口：先准备输入，再计算动作 log prob
        prepared_inputs = self.prepare_inputs(prompts, pixel_values)
        return self.action_logprob_from_prepared(prepared_inputs, actions)

    def get_action_logits(
        self,
        prompts: List[str],
        images: torch.Tensor,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # 为每个动作计算 log prob，堆叠成 logits
        batch_size = len(prompts)
        num_actions = len(ACTION_LIST)
        all_logprobs = []
        if prepared_inputs is None:
            prepared_inputs = self.prepare_inputs(prompts, images)
        for action_idx in range(num_actions):
            action = ACTION_LIST[action_idx]
            actions = [action] * batch_size
            logprobs = self.action_logprob_from_prepared(prepared_inputs, actions)
            all_logprobs.append(logprobs)
        logits = torch.stack(all_logprobs, dim=-1)
        return logits

    def compute_entropy(
        self,
        prompts: List[str],
        images: torch.Tensor,
        prepared_inputs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # 根据动作 logits 计算策略熵
        logits = self.get_action_logits(prompts, images, prepared_inputs=prepared_inputs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy
