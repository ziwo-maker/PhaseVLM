import os
from typing import Any, Optional, Literal, Tuple, Union, List

import torch
from PIL import Image


from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage,LearnableTimeSeriesToImage_hot
from layers.TimeSeries_To_Image import time_series_to_simple_image
class TimeSeriesImageAdapter:
    """
    Convert time series inputs into image tensors.

    ✅ 关键改动：
    1) 训练/推理默认返回 float 图像（可导），范围通常为 [0,1] 或者保持原始浮点（由 image_output 决定）
    2) uint8 仅用于保存/可视化（PIL、保存到磁盘），不再在主流水线里强制 to(uint8)
    3) 支持更稳的缩放方式：per-image minmax / fixed-range / log1p+fixed-range
    """

    def __init__(
        self,
        image_size: int = 896,
        seq_len: Optional[int] = 16,
        periodicity: int = 30,
        norm_const: float = 1.0,
        learnable_image: bool = False,
        three_channel_image: bool = True,
        save_images: bool = True,
        device: Optional[torch.device] = None,
        input_dim: int = 8,
        hidden_dim: int = 64,
        # --- 新增：图像输出与缩放策略 ---
        image_output: Literal["float01", "float"] = "float01",
        image_scaling: Literal["per_image_minmax", "fixed_range", "log1p_fixed_range"] = "per_image_minmax",
        fixed_range: Tuple[float, float] = (0.0, 3.1),  # 用于 fixed_range / log1p_fixed_range
        eps: float = 1e-5,
    ) -> None:
        self.image_size = int(image_size)
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.periodicity = int(periodicity)
        self.norm_const = float(norm_const)
        self.learnable_image = bool(learnable_image)
        self.three_channel_image = bool(three_channel_image)
        self.save_images = bool(save_images)
        self.device = device
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)

        self.image_output = image_output
        self.image_scaling = image_scaling
        self.fixed_range = (float(fixed_range[0]), float(fixed_range[1]))
        self.eps = float(eps)

        self.learnable_image_module = None

        if self.learnable_image:
            out_channels = 3 if self.three_channel_image else 1
            self.learnable_image_module = LearnableTimeSeriesToImage_hot(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
               
                image_size=self.image_size,
             
            )
            if self.device is not None:
                self.learnable_image_module = self.learnable_image_module.to(self.device)

    def __call__(self, data: Any) -> torch.Tensor:
        return self.generate(data)

    # ---------------------------
    # 输入处理
    # ---------------------------
    def _ensure_tensor(self, data: Any) -> torch.Tensor:
        # 确保输入是 float32 张量，形状为 (B, L, C)，并放到目标设备上。
        if torch.is_tensor(data):
            x_enc = data
        else:
            x_enc = torch.as_tensor(data, dtype=torch.float32)

        # (L, C) -> (1, L, C)
        if x_enc.ndim == 2:
            x_enc = x_enc.unsqueeze(0)
        if x_enc.ndim != 3:
            raise ValueError("Expected input shape (L, C) or (B, L, C).")

        if x_enc.dtype != torch.float32:
            x_enc = x_enc.float()

        if self.device is not None and x_enc.device != self.device:
            x_enc = x_enc.to(self.device)

        return x_enc

    def _align_device(self, x_enc: torch.Tensor) -> torch.Tensor:
        # 若存在可学习图像模块，保证输入与模块在同一 device 上。
        if self.learnable_image_module is None:
            return x_enc
        module_device = next(self.learnable_image_module.parameters()).device
        if x_enc.device != module_device:
            x_enc = x_enc.to(module_device)
        return x_enc

    def _context_len(self, x_enc: torch.Tensor) -> int:
        # 使用实际序列长度，允许与配置的 seq_len 不一致。
        if self.seq_len is None or self.seq_len <= 0:
            return x_enc.shape[1]
        # 允许输入长度不等于 seq_len（例如在线控制窗口变化）
        return x_enc.shape[1]

    def _normalize_input(self, x_enc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对输入时序做标准化（沿时间维 L），并保持可导。
        """
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + self.eps)
        # norm_const 越小，stdev /= norm_const 越大，x/stdev 越小
        stdev = stdev / max(self.norm_const, self.eps)
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    # ---------------------------
    # 图像缩放（可导）
    # ---------------------------
    def _scale_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: float tensor [B, C, H, W]（来自 learnable_image 或 time_series_to_simple_image）
        输出：float（默认 [0,1]，除非 image_output="float"）
        """
        if self.image_scaling == "per_image_minmax":
            # 每张图单独 min-max（可导），注意：会弱化绝对规模信息
            min_vals = images.amin(dim=(1, 2, 3), keepdim=True)
            max_vals = images.amax(dim=(1, 2, 3), keepdim=True)
            scale = (max_vals - min_vals).clamp_min(self.eps)
            out = (images - min_vals) / scale
            return out if self.image_output == "float01" else images

        elif self.image_scaling == "fixed_range":
            lo, hi = self.fixed_range
            if hi <= lo:
                raise ValueError(f"fixed_range invalid: {self.fixed_range}")
            out = (images - lo) / (hi - lo)
            out = out.clamp(0.0, 1.0)
            return out if self.image_output == "float01" else images

        elif self.image_scaling == "log1p_fixed_range":
            # 适合“计数型”输入：压缩大值、保留量级差异
            lo, hi = self.fixed_range
            if hi <= lo:
                raise ValueError(f"fixed_range invalid: {self.fixed_range}")
            out = torch.log1p(images.clamp_min(0.0))
            out = (out - lo) / (hi - lo)
            out = out.clamp(0.0, 1.0)
            return out if self.image_output == "float01" else images

        else:
            raise ValueError(f"Unknown image_scaling: {self.image_scaling}")

    @staticmethod
    def _to_uint8(images_float01: torch.Tensor) -> torch.Tensor:
        """
        将 [0,1] 的 float 图像转成 uint8（仅用于保存/可视化，不用于训练反传）。
        """
        return (images_float01 * 255.0).clamp(0.0, 255.0).to(torch.uint8)

    # ---------------------------
    # 主生成逻辑
    # ---------------------------
    def vision_augmented_learner(
        self,
        x_enc: torch.Tensor,
        image_size: int,
        context_len: int,
        periodicity: int,
    ) -> torch.Tensor:
        """
        返回 float 图像（默认 [0,1]），保持可导（尤其是 learnable_image=True 时）。
        """
        
        # 通过可学习模块或固定映射将时序转为图像。
        if self.learnable_image:
            if self.learnable_image_module is None:
                out_channels = 3 if self.three_channel_image else 1
        
                self.learnable_image_module = LearnableTimeSeriesToImage_hot(
                    input_dim=self.input_dim,
                    hidden_dim=self.hidden_dim,
                    image_size=image_size,
                )
            images = self.learnable_image_module(x_enc)  # 期望 float: [B,C,H,W]
        else:
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)

        # ✅ 关键：缩放保持 float 可导（不再强制 uint8）
        images_scaled = images

        # 可选保存（保存时才转 uint8）
        if self.save_images:
            self.save_images_to_disk(images_scaled)

        return images_scaled

    def generate(self, data: Any) -> torch.Tensor:
        """
        默认：对输入时序做标准化，然后生成 float 图像（默认范围 [0,1]）。
        """
        x_enc = self._ensure_tensor(data)
        x_enc = self._align_device(x_enc)
        # 在转图像前，沿时间维进行标准化。
        # x_enc, _, _ = self._normalize_input(x_enc)  #自定义类不需要标准化
        context_len = self._context_len(x_enc)
        return self.vision_augmented_learner(x_enc, self.image_size, context_len, self.periodicity)

    def generate_from_normalized(self, data: Any) -> torch.Tensor:
        """
        输入已标准化时序时使用，不再重复 _normalize_input。
        """
        x_enc = self._ensure_tensor(data)
        x_enc = self._align_device(x_enc)
        context_len = self._context_len(x_enc)
        return self.vision_augmented_learner(x_enc, self.image_size, context_len, self.periodicity)

    def generate_images(
        self,
        data: Any,
        normalized: bool = False,
        return_tensor: bool = False,
        as_uint8: bool = False,
    ):
        """
        - return_tensor=True: 返回 torch.Tensor（float 或 uint8）
        - return_tensor=False: 返回 PIL.Image list
        - as_uint8=True: 将 float01 转 uint8（仅用于可视化/存盘/某些 processor 需要 uint8 时）
        """
        images = self.generate_from_normalized(data) if normalized else self.generate(data)

        if as_uint8:
            # 仅当 images 是 float01 时才合理
            if images.dtype != torch.uint8:
                images = self._to_uint8(images)

        if return_tensor:
            return images
        return self.to_pil(images)

    def to_pil(self, images: torch.Tensor) -> Optional[List[Image.Image]]:
        if not torch.is_tensor(images):
            return None

        images = images.detach().cpu()

        # 如果是 float01，先转 uint8 再做 PIL
        if images.dtype != torch.uint8:
            images = self._to_uint8(images)

        pil_images: List[Image.Image] = []
        for img in images:
            if img.shape[0] == 1:
                array = img.squeeze(0).numpy()
                mode = "L"
            else:
                array = img.permute(1, 2, 0).numpy()
                mode = "RGB"
            pil_images.append(Image.fromarray(array, mode=mode))
        return pil_images

    def save_images_to_disk(self, images: torch.Tensor) -> None:
        """
        保存时才转 uint8；训练主链路仍保持 float 可导。
        """
        save_dir = "ts-images/timevlm"
        os.makedirs(save_dir, exist_ok=True)

        # 保存统一用 uint8
        if images.dtype != torch.uint8:
            images_to_save = self._to_uint8(images.detach())
        else:
            images_to_save = images.detach()

        images_to_save = images_to_save.cpu()

        for i, img_tensor in enumerate(images_to_save):
            if img_tensor.shape[0] == 3:
                img_tensor = img_tensor.permute(1, 2, 0)
                mode = "RGB"
            elif img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
                mode = "L"
            else:
                continue

            try:
                img = Image.fromarray(img_tensor.numpy(), mode=mode)
                img.save(os.path.join(save_dir, f"image_{i}.png"))
            except Exception:
                continue
