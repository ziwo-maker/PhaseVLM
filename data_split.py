from __future__ import annotations

import random
from typing import Any, Iterable, List, Sequence, Tuple

from torch.utils.data import Subset


def _sample_group_key(sample: Any) -> str:
    """Return a chronological trajectory key for blocked validation.

    GroupedTrafficActionDataset preserves input-file order, but records from
    different intersections are interleaved.  A flow-level suffix therefore
    cannot safely purge the sequence history of each intersection.
    """
    traffic_file = str(getattr(sample, "traffic_file", "") or "").strip()
    source_path = str(getattr(sample, "source_path", "") or "").strip()
    intersection_id = str(getattr(sample, "intersection_id", "") or "").strip()
    if traffic_file or source_path or intersection_id:
        return "\x1f".join((source_path, traffic_file, intersection_id))
    return "__default__"


def _sample_label(sample: Any, label_attr: str) -> str:
    label = str(getattr(sample, label_attr, "") or "").strip()
    return label or "__missing__"


def blocked_group_split(
    samples: Sequence[Any],
    val_ratio: float,
    purge_samples: int = 0,
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
    purged_indices: set[int] = set()
    purge_samples = max(0, int(purge_samples))
    for indices in grouped_indices.values():
        if len(indices) <= 1:
            continue
        val_count = max(1, int(len(indices) * val_ratio))
        val_count = min(val_count, len(indices) - 1)
        val_start = len(indices) - val_count
        purge_start = max(0, val_start - purge_samples)
        val_indices.extend(indices[val_start:])
        purged_indices.update(indices[purge_start:val_start])

    if not val_indices:
        val_count = max(1, int(total * val_ratio))
        val_count = min(val_count, total - 1)
        val_start = total - val_count
        purge_start = max(0, val_start - purge_samples)
        val_indices = list(range(val_start, total))
        purged_indices.update(range(purge_start, val_start))

    val_index_set = set(val_indices)
    train_indices = [
        idx for idx in range(total)
        if idx not in val_index_set and idx not in purged_indices
    ]
    if not train_indices:
        raise ValueError(
            "Blocked validation leaves no training samples after the purge buffer; "
            "reduce --validation_purge_samples or --val_ratio."
        )
    return train_indices, sorted(val_indices)


def stratified_label_split(
    samples: Sequence[Any],
    val_ratio: float,
    label_attr: str = "expert_action",
    seed: int = 0,
) -> Tuple[List[int], List[int]]:
    total = len(samples)
    if total == 0:
        return [], []
    if total == 1 or val_ratio <= 0:
        return list(range(total)), []

    grouped_indices: dict[str, List[int]] = {}
    for idx, sample in enumerate(samples):
        grouped_indices.setdefault(_sample_label(sample, label_attr), []).append(idx)

    rng = random.Random(seed)
    val_indices: List[int] = []
    for label in sorted(grouped_indices):
        indices = list(grouped_indices[label])
        if len(indices) <= 1:
            continue
        rng.shuffle(indices)
        val_count = max(1, int(len(indices) * val_ratio))
        val_count = min(val_count, len(indices) - 1)
        val_indices.extend(indices[:val_count])

    if not val_indices:
        val_count = max(1, int(total * val_ratio))
        val_count = min(val_count, total - 1)
        shuffled = list(range(total))
        rng.shuffle(shuffled)
        val_indices = shuffled[:val_count]

    val_index_set = set(val_indices)
    train_indices = [idx for idx in range(total) if idx not in val_index_set]
    if not train_indices:
        train_indices = list(range(total - 1))
        val_indices = [total - 1]
    return train_indices, sorted(val_indices)


def blocked_group_subset(
    dataset: Any,
    val_ratio: float,
    purge_samples: int = 0,
) -> Tuple[Subset, Subset | None]:
    samples = list(getattr(dataset, "samples", []))
    train_indices, val_indices = blocked_group_split(
        samples,
        val_ratio=val_ratio,
        purge_samples=purge_samples,
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None
    return train_subset, val_subset


def stratified_label_subset(
    dataset: Any,
    val_ratio: float,
    label_attr: str = "expert_action",
    seed: int = 0,
) -> Tuple[Subset, Subset | None]:
    samples = list(getattr(dataset, "samples", []))
    train_indices, val_indices = stratified_label_split(
        samples,
        val_ratio=val_ratio,
        label_attr=label_attr,
        seed=seed,
    )
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices) if val_indices else None
    return train_subset, val_subset
