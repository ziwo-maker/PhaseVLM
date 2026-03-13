from __future__ import annotations

from typing import Any, Iterable, List, Sequence, Tuple

from torch.utils.data import Subset


def _sample_group_key(sample: Any) -> str:
    traffic_file = str(getattr(sample, "traffic_file", "") or "").strip()
    if traffic_file:
        return traffic_file
    source_path = str(getattr(sample, "source_path", "") or "").strip()
    if source_path:
        return source_path
    return "__default__"


def blocked_group_split(
    samples: Sequence[Any],
    val_ratio: float,
) -> Tuple[List[int], List[int]]:
    total = len(samples)
    if total == 0:
        return [], []
    if total == 1 or val_ratio <= 0:
        return list(range(total)), []

    grouped_indices: dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        grouped_indices.setdefault(_sample_group_key(sample), []).append(idx)

    val_indices: List[int] = []
    for indices in grouped_indices.values():
        if len(indices) <= 1:
            continue
        val_count = max(1, int(len(indices) * val_ratio))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[-val_count:])

    if not val_indices:
        val_count = max(1, int(total * val_ratio))
        val_count = min(val_count, total - 1)
        val_indices = list(range(total - val_count, total))

    val_index_set = set(val_indices)
    train_indices = [idx for idx in range(total) if idx not in val_index_set]
    if not train_indices:
        train_indices = list(range(total - 1))
        val_indices = [total - 1]
    return train_indices, sorted(val_indices)


def blocked_group_subset(dataset: Any, val_ratio: float) -> Tuple[Subset, Subset | None]:
    samples = list(getattr(dataset, "samples", []))
    train_indices, val_indices = blocked_group_split(samples, val_ratio=val_ratio)
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None
    return train_subset, val_subset
