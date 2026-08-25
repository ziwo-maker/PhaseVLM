#!/usr/bin/env python3
"""Build matched small Stage-1 and Stage-2 SimExpert datasets.

The selection unit is one caption-terminated decision group.  Stage 1 keeps
the original group structure; Stage 2 writes only the selected terminal rows
with an explicit 16-step sequence, so removing other rows cannot truncate its
temporal input.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np


ACTIONS: Tuple[str, ...] = ("ETWT", "NTST", "ELWL", "NLSL")
MOVEMENTS: Tuple[str, ...] = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
LABEL_KEYS: Tuple[str, ...] = (
    "candidate_action_metrics",
    "candidate_action_waits",
    "expert_action",
    "expert_action_margin",
    "base_action",
    "pareto_feasible",
    "constraint_violations",
    "sim_horizon",
    "cost_weights",
    "relabel_method",
)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def movement_vector(record: Dict[str, Any]) -> List[float]:
    counts = record.get("movement_counts") or {}
    return [finite(counts.get(key, 0.0)) for key in MOVEMENTS]


def normalized_percentile(values: Sequence[float], value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if ordered[-1] - ordered[0] <= 1e-9:
        return 0.5
    rank = sum(item <= value for item in ordered) - 1
    return max(0.0, min(1.0, rank / max(1, len(ordered) - 1)))


def candidate_best(record: Dict[str, Any]) -> Tuple[str, float, float, bool]:
    waits = record.get("candidate_action_waits") or {}
    pareto = record.get("pareto_feasible") or {}
    feasible = [
        (action, finite(waits.get(action), float("inf")))
        for action in ACTIONS
        if action in waits and pareto.get(action, True)
    ]
    if not feasible:
        return "", 0.0, 0.0, False
    feasible.sort(key=lambda item: item[1])
    best_action, best_score = feasible[0]
    base_action = str(record.get("base_action") or "")
    base_score = finite(waits.get(base_action), best_score)
    margin = feasible[1][1] - best_score if len(feasible) > 1 else 0.0
    violations = record.get("constraint_violations") or {}
    safe = not bool(violations.get(best_action))
    return best_action, base_score - best_score, margin, safe


@dataclass
class Group:
    index: int
    records: List[Dict[str, Any]]
    terminal: Dict[str, Any]
    sequence: List[List[float]]
    action: str
    intersection: str
    score: float = 0.0
    reliable: bool = False
    reason: str = "static"
    vector: np.ndarray | None = None

    @property
    def key(self) -> Tuple[str, str]:
        return (str(self.terminal.get("snapshot_path") or ""), self.intersection)


def read_horizon_labels(path: str | None) -> Dict[Tuple[str, str], Dict[str, Any]]:
    if not path:
        return {}
    labels: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("clip_caption") or not record.get("expert_action"):
                continue
            key = (str(record.get("snapshot_path") or ""), str(record.get("intersection_id") or ""))
            if key[0] and key[1]:
                labels[key] = record
    return labels


def iter_groups(path: str, seq_len: int) -> Iterable[Group]:
    histories: Dict[str, List[List[float]]] = defaultdict(list)
    group_records: List[Dict[str, Any]] = []
    group_index = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            iid = str(record.get("intersection_id") or record.get("road_id") or "")
            if iid:
                history = histories[iid]
                history.append(movement_vector(record))
                if len(history) > seq_len:
                    del history[:-seq_len]
            group_records.append(record)
            caption = str(record.get("clip_caption") or "").strip()
            if not caption:
                continue

            terminal = record
            action = str(terminal.get("expert_action") or "")
            if iid and action in ACTIONS:
                history = list(histories[iid])
                if history:
                    history = [list(history[0]) for _ in range(max(0, seq_len - len(history)))] + history
                sequence = history[-seq_len:]
                if len(sequence) == seq_len:
                    yield Group(
                        index=group_index,
                        records=[dict(item) for item in group_records],
                        terminal=dict(terminal),
                        sequence=sequence,
                        action=action,
                        intersection=iid,
                    )
                    group_index += 1
            # This matches TrafficClipJsonlDataset's caption-terminated group contract.
            group_records = []


def merge_h300(group: Group, h300: Dict[Tuple[str, str], Dict[str, Any]]) -> None:
    replacement = h300.get(group.key)
    if replacement is None:
        return
    merged = dict(group.terminal)
    for key in LABEL_KEYS:
        if key in replacement:
            merged[key] = replacement[key]
    group.terminal = merged
    group.action = str(merged.get("expert_action") or group.action)
    group.records[-1] = merged


def set_reliability(
    groups: Sequence[Group],
    h120: Dict[Tuple[str, str], Dict[str, Any]],
    h300: Dict[Tuple[str, str], Dict[str, Any]],
    min_margin: float,
    require_horizon: bool,
) -> None:
    margins = [finite(group.terminal.get("expert_action_margin")) for group in groups]
    for group in groups:
        current_best, current_adv, current_margin, current_safe = candidate_best(group.terminal)
        if require_horizon:
            record120 = h120.get(group.key)
            record300 = h300.get(group.key)
            if record120 is None or record300 is None:
                group.reliable, group.reason = False, "missing_horizon"
                continue
            best120, adv120, _, safe120 = candidate_best(record120)
            best300, adv300, margin300, safe300 = candidate_best(record300)
            group.reliable = bool(
                best120
                and best120 == best300
                and adv120 > 0.0
                and adv300 > 0.0
                and margin300 >= min_margin
                and safe120
                and safe300
            )
            group.reason = "horizon_reliable" if group.reliable else "horizon_unstable_or_unsafe"
            quality_margin = margin300
            quality_adv = adv300
        else:
            group.reliable = bool(
                current_best and current_adv >= 0.0 and current_margin >= min_margin and current_safe
            )
            group.reason = "static_reliable" if group.reliable else "static_low_quality"
            quality_margin = current_margin
            quality_adv = current_adv

        caption_words = len(str(group.terminal.get("clip_caption") or "").split())
        sequence = np.asarray(group.sequence, dtype=np.float32)
        variation = float(np.abs(np.diff(sequence.sum(axis=1))).sum()) if len(sequence) > 1 else 0.0
        last_total = float(sequence[-1].sum())
        group.vector = np.concatenate(
            [sequence[-1], np.asarray([variation, last_total, quality_margin, quality_adv], dtype=np.float32)]
        )
        group.score = (
            0.45 * normalized_percentile(margins, quality_margin)
            + 0.20 * min(1.0, max(0.0, quality_adv) / 3.0)
            + 0.20 * min(1.0, variation / 80.0)
            + 0.10 * min(1.0, last_total / 80.0)
            + 0.05 * min(1.0, caption_words / 30.0)
        )


def action_quotas(groups: Sequence[Group], target: int) -> Dict[str, int]:
    counts = Counter(group.action for group in groups if group.action in ACTIONS)
    total = max(1, sum(counts.values()))
    quotas = {action: int(math.floor(target * counts[action] / total)) for action in ACTIONS}
    for action in ACTIONS:
        if counts[action] and quotas[action] == 0:
            quotas[action] = 1
    while sum(quotas.values()) < target:
        action = max(ACTIONS, key=lambda item: counts[item] / max(1, quotas[item]))
        quotas[action] += 1
    while sum(quotas.values()) > target:
        action = max(ACTIONS, key=lambda item: quotas[item])
        quotas[action] -= 1
    return quotas


def select_random(groups: Sequence[Group], target: int, seed: int) -> List[Group]:
    rng = random.Random(seed)
    quotas = action_quotas(groups, target)
    selected: List[Group] = []
    by_action: Dict[str, List[Group]] = defaultdict(list)
    for group in groups:
        by_action[group.action].append(group)
    for action in ACTIONS:
        candidates = list(by_action[action])
        rng.shuffle(candidates)
        selected.extend(candidates[: quotas[action]])
    return sorted(selected, key=lambda item: item.index)


def select_quality_diverse(groups: Sequence[Group], target: int, novelty_weight: float) -> List[Group]:
    quotas = action_quotas(groups, target)
    candidates = list(groups)
    vectors = np.stack([group.vector for group in candidates if group.vector is not None])
    mean = vectors.mean(axis=0)
    std = np.maximum(vectors.std(axis=0), 1e-6)
    for group in candidates:
        assert group.vector is not None
        group.vector = (group.vector - mean) / std

    inter_count = len({group.intersection for group in candidates})
    per_intersection_cap = max(1, math.ceil(target * 1.5 / max(1, inter_count)))
    selected: List[Group] = []
    matrix = np.stack([group.vector for group in candidates])
    min_distances = np.full((len(candidates),), np.inf, dtype=np.float32)
    used_actions: Counter[str] = Counter()
    used_intersections: Counter[str] = Counter()
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < target:
        best_index = None
        best_value = float("-inf")
        for index in sorted(remaining):
            group = candidates[index]
            if used_actions[group.action] >= quotas[group.action]:
                continue
            if used_intersections[group.intersection] >= per_intersection_cap:
                continue
            if selected:
                novelty = min(1.0, float(min_distances[index]) / math.sqrt(len(group.vector)))
            else:
                novelty = 1.0
            value = group.score + novelty_weight * novelty
            if best_index is None or value > best_value or (
                value == best_value and group.index < candidates[best_index].index
            ):
                best_index, best_value = index, value
        if best_index is None:
            # Quotas can become infeasible after per-intersection capping; relax only the cap.
            for index in sorted(remaining):
                group = candidates[index]
                if used_actions[group.action] < quotas[group.action]:
                    best_index = index
                    break
        if best_index is None:
            break
        group = candidates[best_index]
        selected.append(group)
        used_actions[group.action] += 1
        used_intersections[group.intersection] += 1
        remaining.remove(best_index)
        distances = np.linalg.norm(matrix - group.vector[None, :], axis=1)
        min_distances = np.minimum(min_distances, distances)
    return sorted(selected, key=lambda item: item.index)


def write_dataset(groups: Sequence[Group], output_dir: Path, summary: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "stage1.jsonl"
    stage2_path = output_dir / "stage2.jsonl"
    with stage1_path.open("w", encoding="utf-8") as stage1:
        for group in groups:
            for record in group.records:
                stage1.write(json.dumps(record, ensure_ascii=False) + "\n")
    with stage2_path.open("w", encoding="utf-8") as stage2:
        for group in groups:
            record = dict(group.terminal)
            record["sequences"] = group.sequence
            record["joint_coreset_selected"] = True
            record["joint_coreset_score"] = round(float(group.score), 8)
            record["joint_coreset_reliable"] = bool(group.reliable)
            stage2.write(json.dumps(record, ensure_ascii=False) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build matched Stage-1/Stage-2 SimExpert coresets.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mode", choices=("full", "random", "selected"), required=True)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--min_margin", type=float, default=0.05)
    parser.add_argument("--novelty_weight", type=float, default=0.25)
    parser.add_argument("--h120", default="")
    parser.add_argument("--h300", default="")
    parser.add_argument("--require_horizon", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("--ratio must be in (0, 1]")
    if args.require_horizon and (not args.h120 or not args.h300):
        raise ValueError("--require_horizon needs both --h120 and --h300")

    groups = list(iter_groups(args.input, args.seq_len))
    if not groups:
        raise RuntimeError("No caption-terminated action groups found in input")
    h120 = read_horizon_labels(args.h120)
    h300 = read_horizon_labels(args.h300)
    if h300:
        for group in groups:
            merge_h300(group, h300)
    set_reliability(groups, h120, h300, args.min_margin, args.require_horizon)

    target = len(groups) if args.mode == "full" else max(len(ACTIONS), int(round(len(groups) * args.ratio)))
    eligible = [group for group in groups if group.reliable] if args.require_horizon else list(groups)
    if args.mode == "full":
        selected = list(groups)
    elif len(eligible) < target:
        raise RuntimeError(
            f"Only {len(eligible)} eligible groups for target={target}; relax the gate or increase the pool."
        )
    elif args.mode == "random":
        selected = select_random(eligible, target, args.seed)
    else:
        selected = select_quality_diverse(eligible, target, args.novelty_weight)

    summary = {
        "input": str(Path(args.input).resolve()),
        "mode": args.mode,
        "ratio": args.ratio,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "require_horizon": bool(args.require_horizon),
        "h120": str(args.h120),
        "h300": str(args.h300),
        "total_groups": len(groups),
        "reliable_groups": sum(group.reliable for group in groups),
        "eligible_groups": len(eligible),
        "selected_groups": len(selected),
        "selected_actions": dict(Counter(group.action for group in selected)),
        "selected_intersections": dict(sorted(Counter(group.intersection for group in selected).items())),
        "selected_reliability_reasons": dict(Counter(group.reason for group in selected)),
        "selected_stage1_rows": sum(len(group.records) for group in selected),
    }
    if not args.dry_run:
        write_dataset(selected, Path(args.output_dir), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
