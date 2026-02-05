import re
from typing import List, Optional, Tuple, Union, Dict

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
ACTION_LIST = ["ETWT", "NTST", "ELWL", "NLSL"]
ACTION_SET = set(ACTION_LIST)


def parse_signal(text: str) -> str:
    """从模型输出文本中提取最后一个 <signal>...</signal> 作为动作信号。"""
    matches = re.findall(r"<signal>(.*?)</signal>", text)
    if not matches:
        return "ETWT"
    signal = matches[-1].strip()
    return signal if signal in ACTION_SET else "ETWT"


def action_to_index(signal: str) -> int:
    """将动作信号字符串映射为索引，若非法则回退到 0。"""
    try:
        return ACTION_LIST.index(signal)
    except ValueError:
        return 0


def _format_prompts(tokenizer, prompts: List[str]) -> List[str]:
    """按多模态模型的输入格式构造提示词（回退方案）。"""
    formatted = []
    for prompt in prompts:
        content = f"<start_of_image> {prompt}"
        if hasattr(tokenizer, "apply_chat_template"):
            formatted.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        else:
            formatted.append(content)
    return formatted


class GemmaVLPolicy:
    """基于 Gemma 视觉-语言模型的策略封装。

    主要功能：
    1) 将提示词与图像特征送入 LLM，生成包含动作标签的文本；
    2) 从文本中解析动作信号并映射为离散动作索引；
    3) 计算给定动作的对数概率，供 PPO 等算法优化可学习模块。
    """

    def __init__(
        self,
        model_path: str,
        max_length: int = 2048,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Optional[Union[str, torch.device]] = None,
        device_map: Optional[Union[str, dict]] = None,
        max_memory: Optional[Dict[Union[int, str], str]] = None,
        trust_remote_code: bool = True,
    ) -> None:
        # 设备与最大长度配置
        resolved_device = self._resolve_device(device)
        if resolved_device is None:
            resolved_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.device = resolved_device
        self.max_length = int(max_length)
        self.device_map = device_map
        self.model = None
        self.vision_device = None
        # 加载分词器并保证存在 pad token
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # 加载多模态处理器（用于自动插入图像占位符）
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
        )

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        self.model.config.use_cache = False
        if device_map is None:
            self.model.to(self.device)

    
    def compute_entropy(self, prompts: list[str], images: torch.Tensor) -> torch.Tensor:
        """
        计算策略的熵（用于entropy bonus）
        
        Args:
            prompts: 提示文本列表 [B]
            images: 图像tensor [B, C, H, W]
        
        Returns:
            entropy: 熵值 [B]
        """
        # 获取动作logits
        logits = self.get_action_logits(prompts, images)  # [B, num_actions]
        
        # 计算概率和log概率
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # 计算熵: H = -sum(p * log(p))
        entropy = -(probs * log_probs).sum(dim=-1)  # [B]
        
        return entropy
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

    def _maybe_place_vision_tower(self, vision_device: Optional[Union[str, torch.device]]) -> None:
        if self.model is None:
            return
        target_device = self._resolve_device(vision_device)
        if target_device is None:
            return

        moved = []
        seen = set()

        def _add_module(name: str, module: torch.nn.Module) -> None:
            if not isinstance(module, torch.nn.Module):
                return
            module_id = id(module)
            if module_id in seen:
                return
            seen.add(module_id)
            moved.append((name, module))

        def _collect_from_root(root, prefix: str) -> None:
            if root is None:
                return
            if hasattr(root, "get_vision_tower"):
                try:
                    vt = root.get_vision_tower()
                except Exception:
                    vt = None
                if isinstance(vt, (list, tuple)):
                    for idx, mod in enumerate(vt):
                        _add_module(f"{prefix}get_vision_tower[{idx}]", mod)
                elif isinstance(vt, torch.nn.Module):
                    _add_module(f"{prefix}get_vision_tower", vt)

            for attr in (
                "vision_tower",
                "vision_encoder",
                "vision_model",
                "visual_encoder",
                "visual_model",
                "vision",
            ):
                if hasattr(root, attr):
                    _add_module(f"{prefix}{attr}", getattr(root, attr))

            for attr in (
                "mm_projector",
                "multi_modal_projector",
                "vision_projector",
                "vision_proj",
                "image_projector",
                "image_encoder",
                "vision_adapter",
            ):
                if hasattr(root, attr):
                    _add_module(f"{prefix}{attr}", getattr(root, attr))

        _collect_from_root(self.model, "")
        _collect_from_root(getattr(self.model, "model", None), "model.")

        if not moved:
            keywords = ("vision", "visual", "image_encoder")
            for name, module in self.model.named_modules():
                if not name:
                    continue
                lower = name.lower()
                if any(key in lower for key in keywords):
                    _add_module(name, module)

        if not moved:
            print(
                f"Warning: no vision modules found to move to {target_device}; "
                "vision_device request ignored."
            )
            return

        moved_names = []
        for name, module in moved:
            try:
                module.to(target_device)
                moved_names.append(name)
            except Exception as exc:
                print(f"Warning: failed to move {name} to {target_device}: {exc}")

        if moved_names:
            self.vision_device = target_device
            print(
                f"Placed vision modules on {target_device}: {', '.join(moved_names)}"
            )

    def _model_forward(self, **kwargs):
        return self.model(**kwargs)

    def _format_multimodal_prompts(self, prompts: List[str]) -> List[str]:
        if hasattr(self.processor, "apply_chat_template"):
            formatted = []
            for prompt in prompts:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                formatted.append(
                    self.processor.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            return formatted
        return _format_prompts(self.tokenizer, prompts)

    def encode_inputs(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
        """将文本提示词编码为 input_ids/attention_mask，并返回每条提示词的长度。"""
        formatted = self._format_multimodal_prompts(prompts)

        # processor 会插入图像占位符；其输出的 pixel_values 不用于前向，以保留梯度链路
        images_for_processor = None
        if pixel_values is not None:
            if torch.is_tensor(pixel_values):
                if pixel_values.ndim == 3:
                    pixel_values = pixel_values.unsqueeze(0)
                images_for_processor = [img.detach().cpu() for img in pixel_values]
            else:
                images_for_processor = pixel_values
        images_for_processor = [[img] for img in images_for_processor]
        encoding = self.processor(
            text=formatted,
            images=images_for_processor,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        lengths = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, lengths

    def generate_actions(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        generation_kwargs: dict,
    ) -> Tuple[List[int], List[str]]:
        """根据提示词与图像生成动作及模型原始回答文本。"""
        input_ids, attention_mask, lengths = self.encode_inputs(prompts, pixel_values)
        # 将图像张量移动到模型设备并对齐数据类型
        if pixel_values is not None:
            pixel_device = self.vision_device or self.device
            pixel_values = pixel_values.to(pixel_device, dtype=self.model.dtype)
        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            **generation_kwargs,
        )

        responses = []
        actions = []
        for idx, seq in enumerate(output_ids):
            # 只取生成的新 tokens，避免重复 prompt
            prompt_len = int(lengths[idx])
            gen_ids = seq[prompt_len:]
            response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            responses.append(response)
            # 解析动作标签并映射为离散动作
            signal = parse_signal(response)
            actions.append(action_to_index(signal))
        return actions, responses

    def action_logprob(
        self,
        prompts: List[str],
        pixel_values: torch.Tensor,
        actions: List[int],
    ) -> torch.Tensor:
        """计算给定动作对应的对数概率（按动作标签 token 的对数似然求和）。"""
        input_ids, attention_mask, lengths = self.encode_inputs(prompts, pixel_values)
        pixel_device = self.vision_device or self.device
        pixel_values = pixel_values.to(pixel_device, dtype=self.model.dtype)
        pad_id = int(self.tokenizer.pad_token_id)

        # 将动作索引转换为带 <signal> 标签的 token 序列
        tag_ids_list = [
            self.tokenizer(
                f"<signal>{ACTION_LIST[action]}</signal>",
                add_special_tokens=False,
            ).input_ids
            for action in actions
        ]

        # 构造拼接后的输入（prompt + 动作标签），并补齐到同一长度
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
            # 拷贝 prompt，再拼接动作标签
            full_ids[i, :prompt_len] = input_ids[i, :prompt_len]
            full_ids[i, prompt_len:total_len] = torch.tensor(tag_ids, device=self.device)
            full_mask[i, :total_len] = 1

        # 前向推理得到每个位置的 token logits
        outputs = self._model_forward(
            input_ids=full_ids,
            attention_mask=full_mask,
            pixel_values=pixel_values,
        )
        logits = outputs.logits  # [B, T, V], bf16/fp16
        logsumexp = torch.logsumexp(logits, dim=-1)  # [B, T]
        # 累加动作标签 token 的对数概率（保持可反传）
        logp = torch.zeros(batch_size, device=self.device, dtype=logits.dtype)
        for i in range(batch_size):
            prompt_len = int(lengths[i])
            tag_ids = tag_ids_list[i]
            for offset, token_id in enumerate(tag_ids):
                pos = prompt_len + offset
                if pos == 0:
                    continue
                # 预测位置 pos 的 token 来自 logits 的 pos-1
                logp[i] = logp[i] + (logits[i, pos - 1, token_id] - logsumexp[i, pos - 1])
        return logp




def get_action_logits(self, prompts: list[str], images: torch.Tensor) -> torch.Tensor:
    """
    获取所有动作的logits
    
    Args:
        prompts: 提示文本列表 [B]
        images: 图像tensor [B, C, H, W]
    
    Returns:
        logits: 所有动作的logits [B, num_actions]
    """
    batch_size = len(prompts)
    num_actions = len(ACTION_LIST)
    
    # 为每个动作计算log概率
    all_logprobs = []
    for action_idx in range(num_actions):
        action = ACTION_LIST[action_idx]
        actions = [action] * batch_size
        logprobs = self.action_logprob(prompts, images, actions)  # [B]
        all_logprobs.append(logprobs)
    
    logits = torch.stack(all_logprobs, dim=-1)  # [B, num_actions]
    
    return logits