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
    prompt_context: str
    expert_action: str
    clip_caption: str
    date: str
    intersection_id: str
    snapshot_path: str
    traffic_file: str
    source_path: str
    candidate_action_waits: Dict[str, float]
    candidate_action_metrics: Dict[str, Dict[str, float]]
    expert_action_margin: Optional[float]
    base_action: str
    sample_weight: float
    phase_queue: Dict[str, float]
    phase_wait: Dict[str, float]
    pareto_feasible: Dict[str, bool]  # Per-action Pareto feasibility from SimExpert
    meta: Dict
    has_explicit_sequence: bool = False


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
        min_action_margin: Optional[float] = None,
    ) -> None:
        self.paths = [str(item) for item in path] if isinstance(path, (list, tuple)) else [str(path)]
        self.seq_len = int(seq_len)
        self.movement_keys = list(movement_keys) if movement_keys is not None else list(DEFAULT_MOVEMENT_KEYS)
        self.intersection_id = str(intersection_id or "")
        self.require_expert_action = bool(require_expert_action)
        self.valid_actions = set(valid_actions) if valid_actions is not None else None
        self.max_samples = max_samples
        self.min_action_margin = None if min_action_margin is None else float(min_action_margin)
        self.samples: List[TrafficActionSample] = []
        self._load()

    @staticmethod
    def _parse_candidate_waits(raw_waits: object) -> Dict[str, float]:
        if not isinstance(raw_waits, dict):
            return {}
        parsed: Dict[str, float] = {}
        for key, value in raw_waits.items():
            try:
                parsed[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return parsed

    @classmethod
    def _parse_candidate_metrics(cls, raw_metrics: object) -> Dict[str, Dict[str, float]]:
        if not isinstance(raw_metrics, dict):
            return {}
        parsed: Dict[str, Dict[str, float]] = {}
        for action, raw_values in raw_metrics.items():
            values = cls._parse_candidate_waits(raw_values)
            if values:
                parsed[str(action)] = values
        return parsed

    @staticmethod
    def _parse_pareto_feasible(raw_pareto: object) -> Dict[str, bool]:
        """Parse pareto_feasible dict from JSONL record."""
        if not isinstance(raw_pareto, dict):
            return {}
        parsed: Dict[str, bool] = {}
        for action, value in raw_pareto.items():
            if isinstance(value, bool):
                parsed[str(action)] = value
            elif isinstance(value, (int, float)):
                parsed[str(action)] = bool(value)
        return parsed

    @staticmethod
    def _sample_weight(record: Dict) -> float:
        raw_weight = record.get("sample_weight", 1.0)
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            weight = 1.0
        return max(0.0, min(weight, 100.0))

    @staticmethod
    def _resolve_margin(record: Dict, candidate_waits: Dict[str, float]) -> Optional[float]:
        explicit = record.get("expert_action_margin")
        if explicit is not None:
            try:
                return float(explicit)
            except (TypeError, ValueError):
                return None
        if len(candidate_waits) < 2:
            return None
        values = sorted(candidate_waits.values())
        return float(values[1] - values[0])

    def _load(self) -> None:
        for path in self.paths:
            # Records are interleaved across intersections and only some carry an
            # expert action. Keep each intersection's rolling observation window
            # so an unlabeled observation never breaks the next labeled sample.
            histories: Dict[str, List[Dict[str, float]]] = {}
            with open(path, "r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    raw_intersection_id = record.get("intersection_id") or record.get("road_id") or ""
                    sample_intersection_id = str(raw_intersection_id)
                    if self.intersection_id and sample_intersection_id != self.intersection_id:
                        continue

                    history = histories.setdefault(sample_intersection_id, [])
                    history.append(record.get("movement_counts") or {})
                    if len(history) > self.seq_len:
                        del history[:-self.seq_len]

                    # Prefer explicit canonical sequence (from evaluator hook).
                    # This guarantees parity: the dataset feeds the same tensor
                    # the policy received during collection.
                    explicit_sequence: Optional[List[List[float]]] = None
                    raw_sequences = record.get("sequences")
                    if isinstance(raw_sequences, list) and len(raw_sequences) > 0:
                        parsed_rows: List[List[float]] = []
                        for row in raw_sequences:
                            if not isinstance(row, list):
                                parsed_rows = []
                                break
                            try:
                                parsed_rows.append([float(v) for v in row])
                            except (TypeError, ValueError):
                                parsed_rows = []
                                break
                        if (
                            parsed_rows
                            and len(parsed_rows) == self.seq_len
                            and len(parsed_rows[0]) == len(self.movement_keys)
                        ):
                            explicit_sequence = parsed_rows

                    raw_prompt = str(record.get("prompt") or "")
                    prompt_context = str(record.get("prompt_context") or record.get("traffic_context") or "")
                    expert_action = str(record.get("expert_action") or "")
                    clip_caption = str(record.get("clip_caption") or "")
                    candidate_waits = self._parse_candidate_waits(record.get("candidate_action_waits"))
                    candidate_metrics = self._parse_candidate_metrics(record.get("candidate_action_metrics"))
                    pareto_feasible = self._parse_pareto_feasible(record.get("pareto_feasible"))
                    phase_queue = self._parse_candidate_waits(record.get("phase_queue"))
                    phase_wait = self._parse_candidate_waits(record.get("phase_wait"))
                    action_margin = self._resolve_margin(record, candidate_waits)
                    if not (raw_prompt or expert_action or clip_caption):
                        continue

                    if self.require_expert_action and not expert_action:
                        continue
                    if self.valid_actions is not None and expert_action and expert_action not in self.valid_actions:
                        continue
                    if self.min_action_margin is not None:
                        if action_margin is None or action_margin < self.min_action_margin:
                            continue

                    self.samples.append(
                        TrafficActionSample(
                            sequence=explicit_sequence if explicit_sequence is not None
                                     else self._build_sequence(history),
                            raw_prompt=raw_prompt,
                            prompt_context=prompt_context,
                            expert_action=expert_action,
                            clip_caption=clip_caption,
                            date=str(record.get("date") or ""),
                            intersection_id=sample_intersection_id,
                            snapshot_path=str(record.get("snapshot_path") or ""),
                            traffic_file=infer_traffic_file_from_snapshot_path(
                                str(record.get("snapshot_path") or "")
                            ),
                            source_path=str(path),
                            candidate_action_waits=candidate_waits,
                            candidate_action_metrics=candidate_metrics,
                            expert_action_margin=action_margin,
                            base_action=str(record.get("base_action") or ""),
                            sample_weight=self._sample_weight(record),
                            phase_queue=phase_queue,
                            phase_wait=phase_wait,
                            pareto_feasible=pareto_feasible,
                            meta=record.get("meta") or {},
                            has_explicit_sequence=explicit_sequence is not None,
                        )
                    )

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
            "prompt_context": sample.prompt_context,
            "expert_action": sample.expert_action,
            "clip_caption": sample.clip_caption,
            "date": sample.date,
            "intersection_id": sample.intersection_id,
            "snapshot_path": sample.snapshot_path,
            "traffic_file": sample.traffic_file,
            "source_path": sample.source_path,
            "candidate_action_waits": sample.candidate_action_waits,
            "candidate_action_metrics": sample.candidate_action_metrics,
            "expert_action_margin": sample.expert_action_margin,
            "base_action": sample.base_action,
            "sample_weight": sample.sample_weight,
            "phase_queue": sample.phase_queue,
            "phase_wait": sample.phase_wait,
            "pareto_feasible": sample.pareto_feasible,
            "meta": sample.meta,
        }


def grouped_action_collate(batch: Sequence[Dict]) -> Dict:
    return {
        "sequence": torch.stack([item["sequence"] for item in batch], dim=0),
        "raw_prompt": [item["raw_prompt"] for item in batch],
        "prompt_context": [item["prompt_context"] for item in batch],
        "expert_action": [item["expert_action"] for item in batch],
        "clip_caption": [item["clip_caption"] for item in batch],
        "date": [item["date"] for item in batch],
        "intersection_id": [item["intersection_id"] for item in batch],
        "snapshot_path": [item["snapshot_path"] for item in batch],
        "traffic_file": [item["traffic_file"] for item in batch],
        "source_path": [item["source_path"] for item in batch],
        "candidate_action_waits": [item["candidate_action_waits"] for item in batch],
        "candidate_action_metrics": [item["candidate_action_metrics"] for item in batch],
        "pareto_feasible": [item["pareto_feasible"] for item in batch],
        "expert_action_margin": [item["expert_action_margin"] for item in batch],
        "base_action": [item["base_action"] for item in batch],
        "sample_weight": torch.tensor([item["sample_weight"] for item in batch], dtype=torch.float32),
        "phase_queue": [item["phase_queue"] for item in batch],
        "phase_wait": [item["phase_wait"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }
