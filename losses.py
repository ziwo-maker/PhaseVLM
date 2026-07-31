from typing import Optional, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F


def _resolve_logit_scale(
    image_embeds: torch.Tensor,
    temperature: float,
    logit_scale: Optional[Union[torch.Tensor, float]],
) -> torch.Tensor:
    if logit_scale is None:
        logit_scale = image_embeds.new_tensor(1.0 / max(float(temperature), 1e-6))
    elif not torch.is_tensor(logit_scale):
        logit_scale = image_embeds.new_tensor(float(logit_scale))
    else:
        logit_scale = logit_scale.to(device=image_embeds.device, dtype=image_embeds.dtype)
    return logit_scale.clamp(min=1e-3, max=100.0)


def _gather_for_distributed_contrastive(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    if not dist.is_available() or not dist.is_initialized():
        return image_embeds, text_embeds, 1

    world_size = dist.get_world_size()
    if world_size <= 1:
        return image_embeds, text_embeds, 1

    rank = dist.get_rank()
    gathered_images = [torch.zeros_like(image_embeds) for _ in range(world_size)]
    gathered_texts = [torch.zeros_like(text_embeds) for _ in range(world_size)]
    dist.all_gather(gathered_images, image_embeds.detach())
    dist.all_gather(gathered_texts, text_embeds.detach())

    # Keep the local slice attached to autograd so each rank backprops through
    # its own features while still contrasting against the global batch.
    gathered_images[rank] = image_embeds
    gathered_texts[rank] = text_embeds
    return torch.cat(gathered_images, dim=0), torch.cat(gathered_texts, dim=0), world_size


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
    logit_scale = _resolve_logit_scale(image_embeds, temperature=temperature, logit_scale=logit_scale)
    all_image_embeds, all_text_embeds, world_size = _gather_for_distributed_contrastive(image_embeds, text_embeds)
    logits = logit_scale * all_image_embeds @ all_text_embeds.t()
    labels = torch.arange(image_embeds.size(0), device=image_embeds.device)
    if world_size > 1:
        global_batch = all_image_embeds.size(0)
        labels = torch.arange(global_batch, device=image_embeds.device)

    loss_i = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.t(), labels)
    return 0.5 * (loss_i + loss_t)
