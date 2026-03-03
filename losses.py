from typing import Optional, Union

import torch
import torch.nn.functional as F


def clip_contrastive_loss(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    temperature: float = 0.07,
    logit_scale: Optional[Union[torch.Tensor, float]] = None,
) -> torch.Tensor:
    """CLIP-style symmetric InfoNCE loss."""
    if image_embeds.ndim != 2 or text_embeds.ndim != 2:
        raise ValueError("Expected 2D embeddings [B, D].")
    if image_embeds.shape[0] != text_embeds.shape[0]:
        raise ValueError("Image and text batch size must match.")
    if logit_scale is None:
        logit_scale = image_embeds.new_tensor(1.0 / max(float(temperature), 1e-6))
    else:
        if not torch.is_tensor(logit_scale):
            logit_scale = image_embeds.new_tensor(float(logit_scale))
        else:
            logit_scale = logit_scale.to(device=image_embeds.device, dtype=image_embeds.dtype)
    logit_scale = logit_scale.clamp(min=1e-3, max=100.0)
    logits = logit_scale * image_embeds @ text_embeds.t()
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)
