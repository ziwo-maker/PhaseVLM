import json
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from utils.my_utils import location_direction_dict


@dataclass
class TrafficClipSample:
    sequence: List[List[float]]
    caption: str
    intersection_id: str
    meta: Dict
    image_path: str = ""
    npy_path: str = ""


class TrafficClipJsonlDataset(Dataset):
    """Build (sequence, caption) samples from traffic_prompt_series jsonl.

    Each group ends when a record has non-empty clip_caption. All records in
    the group contribute to the time series history.
    """

    def __init__(
        self,
        path: Union[str, Sequence[str]],
        seq_len: int = 15,
        movement_keys: Optional[Iterable[str]] = None,
        max_samples: Optional[int] = None,
        max_lines: Optional[int] = None,
        shuffle_samples: bool = False,
        shuffle_seed: Optional[int] = None,
    ) -> None:
        if isinstance(path, (list, tuple)):
            self.paths = [str(p) for p in path]
        else:
            self.paths = [str(path)]
        self.seq_len = int(seq_len)
        self.movement_keys = list(movement_keys) if movement_keys is not None else list(location_direction_dict)
        self.max_samples = max_samples
        self.max_lines = max_lines
        self.shuffle_samples = shuffle_samples
        self.shuffle_seed = shuffle_seed
        self.samples: List[TrafficClipSample] = []
        self._load()

    def _load(self) -> None:
        for path in self.paths:
            history: List[Dict[str, float]] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line_idx, line in enumerate(handle):
                    if self.max_lines is not None and line_idx >= self.max_lines:
                        break
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    movement_counts = record.get("movement_counts") or {}
                    history.append(movement_counts)

                    clip_caption = record.get("clip_caption") or ""
                    if not clip_caption:
                        continue

                    seq = self._build_sequence(history)
                    intersection_id = str(record.get("intersection_id") or record.get("road_id") or "")
                    meta = record.get("meta") or {}
                    image_path = str(record.get("image_path") or record.get("snapshot_path") or "")
                    npy_path = str(record.get("npy_path") or record.get("image_npy_path") or record.get("tensor_path") or "")
                    self.samples.append(
                        TrafficClipSample(
                            sequence=seq,
                            caption=str(clip_caption),
                            intersection_id=intersection_id,
                            meta=meta,
                            image_path=image_path,
                            npy_path=npy_path,
                        )
                    )
                    history = []

                    if self.max_samples is not None and len(self.samples) >= self.max_samples:
                        break
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break
        if self.shuffle_samples:
            rng = random.Random(self.shuffle_seed)
            rng.shuffle(self.samples)

    def _build_sequence(self, history: List[Dict[str, float]]) -> List[List[float]]:
        # Build fixed-length sequence from history.
        seq: List[List[float]] = []
        start_index = max(0, len(history) - self.seq_len)
        window = history[start_index:]
        if len(window) < self.seq_len:
            pad = window[0] if window else {}
            for _ in range(self.seq_len - len(window)):
                seq.append(self._counts_to_vector(pad))
        for counts in window:
            seq.append(self._counts_to_vector(counts))
        return seq

    def _counts_to_vector(self, counts: Dict[str, float]) -> List[float]:
        return [float(counts.get(key, 0.0) or 0.0) for key in self.movement_keys]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        seq = torch.tensor(sample.sequence, dtype=torch.float32)
        return {
            "sequence": seq,
            "caption": sample.caption,
            "intersection_id": sample.intersection_id,
            "meta": sample.meta,
            "image_path": sample.image_path,
            "npy_path": sample.npy_path,
        }


def traffic_clip_collate(batch):
    sequences = torch.stack([item["sequence"] for item in batch], dim=0)
    captions = [item["caption"] for item in batch]
    meta = [item.get("meta", {}) for item in batch]
    intersection_ids = [item.get("intersection_id", "") for item in batch]
    image_paths = [item.get("image_path", "") for item in batch]
    npy_paths = [item.get("npy_path", "") for item in batch]
    return {
        "sequence": sequences,
        "caption": captions,
        "meta": meta,
        "intersection_id": intersection_ids,
        "image_path": image_paths,
        "npy_path": npy_paths,
    }
