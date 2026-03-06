from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Union

import torch
from torch.utils.data import Dataset

from projection_train.action_space import ACTION_LIST


DEFAULT_MOVEMENT_KEYS: tuple[str, ...] = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")


def infer_traffic_file_from_snapshot_path(snapshot_path: str) -> str:
    if not snapshot_path:
        return ""
    recorded_dir = os.path.basename(os.path.dirname(os.path.dirname(snapshot_path)))
    if not recorded_dir:
        return ""
    parts = recorded_dir.split("_")
    for end in range(len(parts), 0, -1):
        candidate = "_".join(parts[:end])
        if candidate.endswith(".json"):
            return candidate
    return ""


@dataclass
class TrafficActionSample:
    sequence: List[List[float]]
    raw_prompt: str
    expert_action: str
    clip_caption: str
    date: str
    intersection_id: str
    snapshot_path: str
    traffic_file: str
    meta: Dict


class GroupedTrafficActionDataset(Dataset):
    def __init__(
        self,
        path: Union[str, Sequence[str]],
        seq_len: int = 16,
        movement_keys: Optional[Iterable[str]] = None,
        intersection_id: str = "",
        require_expert_action: bool = True,
        valid_actions: Optional[Iterable[str]] = ACTION_LIST,
        max_samples: Optional[int] = None,
    ) -> None:
        self.paths = [str(item) for item in path] if isinstance(path, (list, tuple)) else [str(path)]
        self.seq_len = int(seq_len)
        self.movement_keys = list(movement_keys) if movement_keys is not None else list(DEFAULT_MOVEMENT_KEYS)
        self.intersection_id = str(intersection_id or "")
        self.require_expert_action = bool(require_expert_action)
        self.valid_actions = set(valid_actions) if valid_actions is not None else None
        self.max_samples = max_samples
        self.samples: List[TrafficActionSample] = []
        self._load()

    def _load(self) -> None:
        for path in self.paths:
            history: List[Dict[str, float]] = []
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    raw_intersection_id = record.get("intersection_id") or record.get("road_id") or ""
                    sample_intersection_id = str(raw_intersection_id)
                    if self.intersection_id and sample_intersection_id != self.intersection_id:
                        history = []
                        continue

                    history.append(record.get("movement_counts") or {})

                    raw_prompt = str(record.get("prompt") or "")
                    expert_action = str(record.get("expert_action") or "")
                    clip_caption = str(record.get("clip_caption") or "")
                    if not (raw_prompt or expert_action or clip_caption):
                        continue

                    if self.require_expert_action and not expert_action:
                        history = []
                        continue
                    if self.valid_actions is not None and expert_action and expert_action not in self.valid_actions:
                        history = []
                        continue

                    self.samples.append(
                        TrafficActionSample(
                            sequence=self._build_sequence(history),
                            raw_prompt=raw_prompt,
                            expert_action=expert_action,
                            clip_caption=clip_caption,
                            date=str(record.get("date") or ""),
                            intersection_id=sample_intersection_id,
                            snapshot_path=str(record.get("snapshot_path") or ""),
                            traffic_file=infer_traffic_file_from_snapshot_path(
                                str(record.get("snapshot_path") or "")
                            ),
                            meta=record.get("meta") or {},
                        )
                    )
                    history = []

                    if self.max_samples is not None and len(self.samples) >= self.max_samples:
                        return

    def _build_sequence(self, history: List[Dict[str, float]]) -> List[List[float]]:
        if not history:
            return [[0.0 for _ in self.movement_keys] for _ in range(self.seq_len)]

        window = history[-self.seq_len :]
        if len(window) < self.seq_len:
            pad_item = window[0]
            window = [pad_item for _ in range(self.seq_len - len(window))] + window
        return [self._counts_to_vector(item) for item in window]

    def _counts_to_vector(self, counts: Dict[str, float]) -> List[float]:
        return [float(counts.get(key, 0.0) or 0.0) for key in self.movement_keys]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        sample = self.samples[index]
        return {
            "sequence": torch.tensor(sample.sequence, dtype=torch.float32),
            "raw_prompt": sample.raw_prompt,
            "expert_action": sample.expert_action,
            "clip_caption": sample.clip_caption,
            "date": sample.date,
            "intersection_id": sample.intersection_id,
            "snapshot_path": sample.snapshot_path,
            "traffic_file": sample.traffic_file,
            "meta": sample.meta,
        }


def grouped_action_collate(batch: Sequence[Dict]) -> Dict:
    return {
        "sequence": torch.stack([item["sequence"] for item in batch], dim=0),
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "expert_action": [item["expert_action"] for item in batch],
        "clip_caption": [item["clip_caption"] for item in batch],
        "date": [item["date"] for item in batch],
        "intersection_id": [item["intersection_id"] for item in batch],
        "snapshot_path": [item["snapshot_path"] for item in batch],
        "traffic_file": [item["traffic_file"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
