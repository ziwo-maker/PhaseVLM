from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from projection_train.action_space import ACTION_LIST
from utils.clean_align_jsonl import (
    DEFAULT_MOVEMENT_KEYS,
    _action_margin_from_record,
    _is_similar_caption,
    _make_group_entry,
    _parse_movement_keys,
    _parse_rebalance_targets,
    _rebalance_groups,
    _resolve_rebalance_targets,
    _sequence_profile,
)


@dataclass
class Stage2Group:
    lines: List[str]
    final_record: Dict
    profile: Dict
    source_path: str
    group_index: int
    sample_id: tuple[str, str, str, str]
    content_hash: str

    @property
    def expert_action(self) -> str:
        return str(self.final_record.get("expert_action") or "").strip()

    @property
    def action_margin(self) -> float | None:
        return _action_margin_from_record(self.final_record)


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except TypeError:
                reconfigure(line_buffering=True)


def _canonical_json(record: Dict) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True)


def _sample_id(record: Dict, source_path: str, group_index: int) -> tuple[str, str, str, str]:
    snapshot_path = str(record.get("snapshot_path") or "").strip()
    intersection_id = str(record.get("intersection_id") or record.get("road_id") or "").strip()
    date_text = str(record.get("date") or "").strip()
    expert_action = str(record.get("expert_action") or "").strip()
    if snapshot_path or intersection_id or date_text:
        return snapshot_path, intersection_id, date_text, expert_action
    return source_path, str(group_index), "", expert_action


def _parse_candidate_waits(record: Dict) -> Dict[str, float] | None:
    raw = record.get("candidate_action_waits")
    if not isinstance(raw, dict):
        return None
    parsed: Dict[str, float] = {}
    for action in ACTION_LIST:
        value = _safe_float(raw.get(action))
        if value is None:
            return None
        parsed[action] = float(value)
    return parsed


def _has_complete_metrics(record: Dict) -> bool:
    raw = record.get("candidate_action_metrics")
    if not isinstance(raw, dict):
        return False
    for action in ACTION_LIST:
        if not isinstance(raw.get(action), dict):
            return False
    return True


def _write_groups(path: str, groups: Sequence[Stage2Group]) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    written = 0
    with open(path, "w", encoding="utf-8") as handle:
        for group in groups:
            for line in group.lines:
                handle.write(line)
                handle.write("\n")
                written += 1
    return written


def _label_counts(groups: Sequence[Stage2Group]) -> Dict[str, int]:
    counts = Counter(group.expert_action for group in groups)
    return {action: int(counts.get(action, 0)) for action in ACTION_LIST}


def _split_group_indices(
    groups: Sequence[Stage2Group],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[List[int], List[int], List[int]]:
    if not groups:
        return [], [], []
    grouped: Dict[str, List[int]] = {}
    for idx, group in enumerate(groups):
        grouped.setdefault(group.expert_action, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []
    for action in ACTION_LIST:
        indices = list(grouped.get(action, []))
        if not indices:
            continue
        rng.shuffle(indices)
        total = len(indices)
        val_count = int(total * val_ratio)
        test_count = int(total * test_ratio)
        if val_ratio > 0 and total >= 3 and val_count == 0:
            val_count = 1
        if test_ratio > 0 and total >= 3 and test_count == 0:
            test_count = 1
        while total - val_count - test_count < 1:
            if val_count >= test_count and val_count > 0:
                val_count -= 1
            elif test_count > 0:
                test_count -= 1
            else:
                break
        val_slice = indices[:val_count]
        test_slice = indices[val_count : val_count + test_count]
        train_slice = indices[val_count + test_count :]
        train_indices.extend(train_slice)
        val_indices.extend(val_slice)
        test_indices.extend(test_slice)
    return sorted(train_indices), sorted(val_indices), sorted(test_indices)


def _undersample_train(groups: Sequence[Stage2Group], seed: int) -> List[Stage2Group]:
    if not groups:
        return []
    by_action: Dict[str, List[Stage2Group]] = {action: [] for action in ACTION_LIST}
    for group in groups:
        by_action.setdefault(group.expert_action, []).append(group)
    active_counts = [len(items) for items in by_action.values() if items]
    if not active_counts:
        return list(groups)
    target = min(active_counts)
    rng = random.Random(seed)
    selected: List[Stage2Group] = []
    for action in ACTION_LIST:
        items = list(by_action.get(action, []))
        if not items:
            continue
        if len(items) <= target:
            chosen = items
        else:
            chosen = rng.sample(items, target)
        chosen.sort(key=lambda item: (item.source_path, item.group_index))
        selected.extend(chosen)
    selected.sort(key=lambda item: (item.source_path, item.group_index))
    return selected


def _oversample_train(
    groups: Sequence[Stage2Group],
    rebalance_targets: Iterable[str],
    rebalance_reference: str,
    rebalance_max_extra_copies_per_group: int,
    rebalance_seed: int,
) -> tuple[List[Stage2Group], Dict]:
    base_entries = [_make_group_entry(group.lines, group.final_record, group.profile) for group in groups]
    resolved_targets = _resolve_rebalance_targets(rebalance_targets)
    output_entries, summary = _rebalance_groups(
        groups=base_entries,
        targets=resolved_targets,
        reference=rebalance_reference,
        max_extra_copies_per_group=rebalance_max_extra_copies_per_group,
        seed=rebalance_seed,
    )
    hash_to_group: Dict[str, Stage2Group] = {group.content_hash: group for group in groups}
    output_groups: List[Stage2Group] = []
    for entry in output_entries:
        final_record = json.loads(str(entry["lines"][-1]))
        content_hash = hashlib.md5(
            "\n".join(_canonical_json(json.loads(line)) for line in entry["lines"]).encode("utf-8")
        ).hexdigest()
        group = hash_to_group.get(content_hash)
        if group is None:
            group = Stage2Group(
                lines=list(entry["lines"]),
                final_record=final_record,
                profile=_sequence_profile(
                    [json.loads(line).get("movement_counts") or {} for line in entry["lines"]],
                    seq_len=16,
                    movement_keys=DEFAULT_MOVEMENT_KEYS,
                ),
                source_path="",
                group_index=-1,
                sample_id=_sample_id(final_record, "", -1),
                content_hash=content_hash,
            )
        output_groups.append(group)
    return output_groups, summary


def build_stage2_splits(
    inputs: Sequence[str],
    train_output: str,
    val_output: str,
    test_output: str,
    seq_len: int,
    movement_keys: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    split_seed: int,
    dedup_similar_captions: bool,
    caption_hamming_threshold: int,
    caption_bands: int,
    train_min_action_margin: float | None,
    train_balance_mode: str,
    rebalance_targets: Sequence[str],
    rebalance_reference: str,
    rebalance_max_extra_copies_per_group: int,
    rebalance_seed: int,
) -> Dict:
    stats = {
        "total_lines": 0,
        "bad_lines": 0,
        "groups_seen": 0,
        "groups_kept": 0,
        "groups_dropped_zero_seq": 0,
        "groups_dropped_incomplete": 0,
        "groups_dropped_invalid_action": 0,
        "groups_dropped_missing_prompt": 0,
        "groups_dropped_missing_waits": 0,
        "groups_dropped_missing_metrics": 0,
        "groups_dropped_missing_snapshot": 0,
        "groups_dropped_exact_duplicate": 0,
        "groups_dropped_similar_caption": 0,
        "train_groups_dropped_low_margin": 0,
    }
    seen_hashes: List[int] = []
    caption_bucket: Dict[tuple[int, int], List[int]] = defaultdict(list)
    seen_exact_captions: set[str] = set()
    exact_hashes: set[str] = set()
    groups: List[Stage2Group] = []
    movement_keys = list(movement_keys)
    for source_path in inputs:
        current_group: List[Dict] = []
        group_index = 0
        with open(source_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stats["total_lines"] += 1
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    stats["bad_lines"] += 1
                    continue
                if not isinstance(record, dict):
                    stats["bad_lines"] += 1
                    continue
                current_group.append(record)
                clip_caption = str(record.get("clip_caption") or "").strip()
                if not clip_caption:
                    continue
                group_index += 1
                stats["groups_seen"] += 1
                prompt = str(record.get("prompt") or "").strip()
                expert_action = str(record.get("expert_action") or "").strip()
                snapshot_path = str(record.get("snapshot_path") or "").strip()
                profile = _sequence_profile(
                    [item.get("movement_counts") or {} for item in current_group],
                    seq_len=seq_len,
                    movement_keys=movement_keys,
                )
                if profile.get("sequence_total") == 0.0:
                    stats["groups_dropped_zero_seq"] += 1
                    current_group = []
                    continue
                if not prompt:
                    stats["groups_dropped_missing_prompt"] += 1
                    current_group = []
                    continue
                if expert_action not in ACTION_LIST:
                    stats["groups_dropped_invalid_action"] += 1
                    current_group = []
                    continue
                if not snapshot_path:
                    stats["groups_dropped_missing_snapshot"] += 1
                    current_group = []
                    continue
                if _parse_candidate_waits(record) is None:
                    stats["groups_dropped_missing_waits"] += 1
                    current_group = []
                    continue
                if not _has_complete_metrics(record):
                    stats["groups_dropped_missing_metrics"] += 1
                    current_group = []
                    continue
                output_lines = [json.dumps(item, ensure_ascii=False) for item in current_group]
                canonical_lines = [_canonical_json(item) for item in current_group]
                content_hash = hashlib.md5("\n".join(canonical_lines).encode("utf-8")).hexdigest()
                if content_hash in exact_hashes:
                    stats["groups_dropped_exact_duplicate"] += 1
                    current_group = []
                    continue
                if dedup_similar_captions:
                    is_duplicate_caption = _is_similar_caption(
                        clip_caption,
                        expert_action,
                        seen_hashes=seen_hashes,
                        bucket=caption_bucket,
                        seen_exact=seen_exact_captions,
                        hamming_threshold=caption_hamming_threshold,
                        bands=caption_bands,
                    )
                    if is_duplicate_caption:
                        stats["groups_dropped_similar_caption"] += 1
                        current_group = []
                        continue
                exact_hashes.add(content_hash)
                groups.append(
                    Stage2Group(
                        lines=output_lines,
                        final_record=record,
                        profile=profile,
                        source_path=str(source_path),
                        group_index=group_index,
                        sample_id=_sample_id(record, str(source_path), group_index),
                        content_hash=content_hash,
                    )
                )
                stats["groups_kept"] += 1
                current_group = []
        if current_group:
            stats["groups_dropped_incomplete"] += 1

    train_indices, val_indices, test_indices = _split_group_indices(
        groups=groups,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=split_seed,
    )
    train_groups = [groups[idx] for idx in train_indices]
    val_groups = [groups[idx] for idx in val_indices]
    test_groups = [groups[idx] for idx in test_indices]

    if train_min_action_margin is not None:
        filtered_train: List[Stage2Group] = []
        for group in train_groups:
            margin = group.action_margin
            if margin is None or margin < float(train_min_action_margin):
                stats["train_groups_dropped_low_margin"] += 1
                continue
            filtered_train.append(group)
        train_groups = filtered_train

    rebalance_summary = None
    if train_balance_mode == "undersample":
        train_groups = _undersample_train(train_groups, seed=rebalance_seed)
    elif train_balance_mode == "oversample":
        train_groups, rebalance_summary = _oversample_train(
            groups=train_groups,
            rebalance_targets=rebalance_targets,
            rebalance_reference=rebalance_reference,
            rebalance_max_extra_copies_per_group=rebalance_max_extra_copies_per_group,
            rebalance_seed=rebalance_seed,
        )

    train_written_lines = _write_groups(train_output, train_groups)
    val_written_lines = _write_groups(val_output, val_groups)
    test_written_lines = _write_groups(test_output, test_groups)
    overlap_train_val = len({group.sample_id for group in train_groups} & {group.sample_id for group in val_groups})
    overlap_train_test = len({group.sample_id for group in train_groups} & {group.sample_id for group in test_groups})
    overlap_val_test = len({group.sample_id for group in val_groups} & {group.sample_id for group in test_groups})
    return {
        **stats,
        "split_seed": int(split_seed),
        "train_ratio": float(train_ratio),
        "val_ratio": float(val_ratio),
        "test_ratio": float(test_ratio),
        "train_balance_mode": str(train_balance_mode),
        "train_min_action_margin": train_min_action_margin,
        "rebalance_summary": rebalance_summary,
        "train_groups": len(train_groups),
        "val_groups": len(val_groups),
        "test_groups": len(test_groups),
        "train_written_lines": train_written_lines,
        "val_written_lines": val_written_lines,
        "test_written_lines": test_written_lines,
        "train_label_counts": _label_counts(train_groups),
        "val_label_counts": _label_counts(val_groups),
        "test_label_counts": _label_counts(test_groups),
        "overlap_train_val": overlap_train_val,
        "overlap_train_test": overlap_train_test,
        "overlap_val_test": overlap_val_test,
    }


def main() -> None:
    _configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--train_output", type=str, required=True)
    parser.add_argument("--val_output", type=str, required=True)
    parser.add_argument("--test_output", type=str, required=True)
    parser.add_argument("--summary_output", type=str, default="")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--movement_keys", type=str, default=",".join(DEFAULT_MOVEMENT_KEYS))
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--split_seed", type=int, default=2024)
    parser.add_argument("--dedup_similar_captions", action="store_true", default=True)
    parser.add_argument("--caption_hamming_threshold", type=int, default=3)
    parser.add_argument("--caption_bands", type=int, default=4)
    parser.add_argument("--train_min_action_margin", type=float, default=None)
    parser.add_argument(
        "--train_balance_mode",
        type=str,
        default="none",
        choices=("none", "undersample", "oversample"),
    )
    parser.add_argument(
        "--rebalance_targets",
        type=str,
        default="east_west_dominant,left_dominant,ELWL,NLSL",
    )
    parser.add_argument("--rebalance_reference", choices=("second_largest", "max"), default="second_largest")
    parser.add_argument("--rebalance_max_extra_copies_per_group", type=int, default=4)
    parser.add_argument("--rebalance_seed", type=int, default=2026)
    args = parser.parse_args()

    total_ratio = float(args.train_ratio) + float(args.val_ratio) + float(args.test_ratio)
    if total_ratio <= 0:
        raise SystemExit("train/val/test ratios must sum to a positive number.")
    train_ratio = float(args.train_ratio) / total_ratio
    val_ratio = float(args.val_ratio) / total_ratio
    test_ratio = float(args.test_ratio) / total_ratio
    summary = build_stage2_splits(
        inputs=args.inputs,
        train_output=args.train_output,
        val_output=args.val_output,
        test_output=args.test_output,
        seq_len=args.seq_len,
        movement_keys=_parse_movement_keys(args.movement_keys),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        split_seed=args.split_seed,
        dedup_similar_captions=bool(args.dedup_similar_captions),
        caption_hamming_threshold=args.caption_hamming_threshold,
        caption_bands=args.caption_bands,
        train_min_action_margin=args.train_min_action_margin,
        train_balance_mode=args.train_balance_mode,
        rebalance_targets=_parse_rebalance_targets(args.rebalance_targets),
        rebalance_reference=args.rebalance_reference,
        rebalance_max_extra_copies_per_group=args.rebalance_max_extra_copies_per_group,
        rebalance_seed=args.rebalance_seed,
    )
    summary_output = str(args.summary_output or "").strip()
    if summary_output:
        os.makedirs(os.path.dirname(summary_output) or ".", exist_ok=True)
        with open(summary_output, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
