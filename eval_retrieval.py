from typing import Dict, Iterable, List, Optional

import torch


def _chunked_topk(
    queries: torch.Tensor,
    keys: torch.Tensor,
    ks: Iterable[int],
    chunk_size: int = 1024,
) -> Dict[int, float]:
    max_k = max(ks)
    n = queries.size(0)
    hits = {k: 0 for k in ks}
    keys_t = keys.t()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = queries[start:end] @ keys_t
        topk = sims.topk(max_k, dim=1).indices
        labels = torch.arange(start, end, device=queries.device).unsqueeze(1)
        match = topk.eq(labels)
        for k in ks:
            hits[k] += match[:, :k].any(dim=1).sum().item()
    return {k: hits[k] / float(n) for k in ks}


def compute_recall_at_k(
    image_embeds: torch.Tensor,
    text_embeds: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
    chunk_size: int = 1024,
) -> Dict[str, float]:
    if image_embeds.size(0) != text_embeds.size(0):
        raise ValueError("image_embeds and text_embeds must have same length")
    image_embeds = image_embeds.to(torch.float32)
    text_embeds = text_embeds.to(torch.float32)
    # image -> text
    i2t = _chunked_topk(image_embeds, text_embeds, ks, chunk_size=chunk_size)
    # text -> image
    t2i = _chunked_topk(text_embeds, image_embeds, ks, chunk_size=chunk_size)
    metrics = {}
    for k in ks:
        metrics[f"R@{k}_i2t"] = i2t[k]
        metrics[f"R@{k}_t2i"] = t2i[k]
    return metrics


def collect_embeddings(
    model,
    dataloader,
    device: torch.device,
    max_samples: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    model.eval()
    image_embeds: List[torch.Tensor] = []
    text_embeds: List[torch.Tensor] = []
    seen = 0
    with torch.no_grad():
        for batch in dataloader:
            sequences = batch["sequence"].to(device)
            captions = batch["caption"]
            img_vec = model.encode_image(sequences)
            txt_vec = model.encode_text(captions)
            image_embeds.append(img_vec.cpu())
            text_embeds.append(txt_vec.cpu())
            seen += sequences.size(0)
            if max_samples is not None and seen >= max_samples:
                break
    image_embeds = torch.cat(image_embeds, dim=0)
    text_embeds = torch.cat(text_embeds, dim=0)
    if max_samples is not None:
        image_embeds = image_embeds[:max_samples]
        text_embeds = text_embeds[:max_samples]
    return {"image": image_embeds, "text": text_embeds}


def evaluate_retrieval(
    model,
    dataloader,
    device: torch.device,
    ks: Iterable[int] = (1, 5, 10),
    max_samples: Optional[int] = None,
) -> Dict[str, float]:
    embeds = collect_embeddings(model, dataloader, device, max_samples=max_samples)
    return compute_recall_at_k(embeds["image"], embeds["text"], ks=ks)
