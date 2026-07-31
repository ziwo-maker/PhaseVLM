"""
Qwen3-VL-8B 加载与输入构建工具。

参考官方 Qwen3-VL 文档/模型卡的多模态使用方式：
- 使用 `AutoModelForImageTextToText` / `AutoProcessor` 加载模型与处理器
- 通过 `processor.apply_chat_template` 构造多模态提示
- 使用 `qwen_vl_utils.process_vision_info` 解析图片/视频输入（可选）
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from qwen_vl_utils import process_vision_info  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    process_vision_info = None


DEFAULT_QWEN3_VL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_qwen3_vl_8b(
    model_id: str = DEFAULT_QWEN3_VL_ID,
    *,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: Optional[Dict[str, Any]] = "auto",
    trust_remote_code: bool = True,
    attn_implementation: Optional[str] = None,
) -> tuple[torch.nn.Module, Any]:
    """加载 Qwen3-VL-8B 模型与处理器。"""
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return model, processor


def _collect_media_from_messages(messages: Sequence[Dict[str, Any]]) -> tuple[list[Any], list[Any]]:
    images: list[Any] = []
    videos: list[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, (list, tuple)):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "image" and "image" in item:
                images.append(item["image"])
            elif item_type == "video" and "video" in item:
                videos.append(item["video"])
    return images, videos


def build_qwen3_vl_inputs(
    messages: Sequence[Dict[str, Any]],
    processor: Any,
    *,
    image_patch_size: int = 16,
    device: Optional[torch.device] = None,
    return_video_metadata: bool = True,
) -> Dict[str, torch.Tensor]:
    """根据 Qwen3-VL 推荐流程构建多模态输入。"""
    text = processor.apply_chat_template(
        list(messages),
        tokenize=False,
        add_generation_prompt=True,
    )

    if process_vision_info is not None:
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=image_patch_size,
            return_video_kwargs=True,
            return_video_metadata=return_video_metadata,
        )
    else:
        image_inputs, video_inputs = _collect_media_from_messages(messages)
        video_kwargs = {}

    inputs = processor(
        text=[text],
        images=image_inputs if image_inputs else None,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    if device is not None:
        inputs = inputs.to(device)
    return inputs


def generate_qwen3_vl(
    model: torch.nn.Module,
    processor: Any,
    messages: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 128,
    device: Optional[torch.device] = None,
    image_patch_size: int = 16,
) -> str:
    """便捷生成接口，返回单条文本输出。"""
    inputs = build_qwen3_vl_inputs(
        messages,
        processor,
        image_patch_size=image_patch_size,
        device=device,
    )
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    decoded = processor.batch_decode(outputs, skip_special_tokens=True)
    return decoded[0] if decoded else ""
