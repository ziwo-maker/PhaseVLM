from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from projection_train.action_space import ACTION_LIST
from projection_train.grouped_dataset import DEFAULT_MOVEMENT_KEYS, GroupedTrafficActionDataset


@dataclass
class SnapshotSeedSample:
    snapshot_path: str
    traffic_file: str
    intersection_ids: List[str]
    sequences_by_intersection: Dict[str, List[List[float]]]
    prompts_by_intersection: Dict[str, str]
    meta_by_intersection: Dict[str, Dict]
    movement_keys: List[str]


class SnapshotSeedDataset:
    def __init__(
        self,
        path: str | Sequence[str],
        seq_len: int = 16,
        movement_keys: Optional[Iterable[str]] = None,
        max_snapshots: Optional[int] = None,
        min_intersections: int = 1,
    ) -> None:
        self.paths = [str(item) for item in path] if isinstance(path, (list, tuple)) else [str(path)]
        self.seq_len = int(seq_len)
        self.movement_keys = list(movement_keys) if movement_keys is not None else list(DEFAULT_MOVEMENT_KEYS)
        self.max_snapshots = max_snapshots
        self.min_intersections = int(min_intersections)
        self.samples: List[SnapshotSeedSample] = []
        self._load()

    def _load(self) -> None:
        grouped = GroupedTrafficActionDataset(
            path=self.paths,
            seq_len=self.seq_len,
            movement_keys=self.movement_keys,
            require_expert_action=False,
            valid_actions=ACTION_LIST,
        )
        grouped_by_snapshot: Dict[tuple[str, str], Dict[str, Dict]] = {}
        for sample in grouped.samples:
            snapshot_path = str(sample.snapshot_path or "").strip()
            traffic_file = str(sample.traffic_file or "").strip()
            intersection_id = str(sample.intersection_id or "").strip()
            if not snapshot_path or not traffic_file or not intersection_id:
                continue
            key = (traffic_file, snapshot_path)
            bucket = grouped_by_snapshot.setdefault(key, {})
            bucket[intersection_id] = {
                "sequence": [list(row) for row in sample.sequence],
                "prompt": str(sample.raw_prompt or ""),
                "meta": dict(sample.meta or {}),
            }

        for (traffic_file, snapshot_path), intersections in grouped_by_snapshot.items():
            if len(intersections) < self.min_intersections:
                continue
            ordered_ids = sorted(intersections.keys())
            self.samples.append(
                SnapshotSeedSample(
                    snapshot_path=snapshot_path,
                    traffic_file=traffic_file,
                    intersection_ids=ordered_ids,
                    sequences_by_intersection={
                        inter_id: intersections[inter_id]["sequence"] for inter_id in ordered_ids
                    },
                    prompts_by_intersection={
                        inter_id: intersections[inter_id]["prompt"] for inter_id in ordered_ids
                    },
                    meta_by_intersection={
                        inter_id: intersections[inter_id]["meta"] for inter_id in ordered_ids
                    },
                    movement_keys=list(self.movement_keys),
                )
            )
            if self.max_snapshots is not None and len(self.samples) >= self.max_snapshots:
                break

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SnapshotSeedSample:
        return self.samples[index]

    def sample(
        self,
        rng: Optional[random.Random] = None,
        traffic_file: str = "",
    ) -> SnapshotSeedSample:
        if not self.samples:
            raise RuntimeError("SnapshotSeedDataset is empty.")
        candidates = self.samples
        if traffic_file:
            candidates = [item for item in self.samples if item.traffic_file == traffic_file]
            if not candidates:
                raise RuntimeError(f"No snapshot seeds found for traffic_file={traffic_file!r}.")
        chooser = rng or random
        return chooser.choice(candidates)

    def traffic_files(self) -> List[str]:
        return sorted({sample.traffic_file for sample in self.samples})
