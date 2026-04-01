from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional


DEFAULT_MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
DEFAULT_REBALANCE_TARGETS = ["east_west_dominant", "left_dominant", "ELWL", "NLSL"]
REBALANCE_FAMILY_ALIASES = {
    "dominance": "dominance",
    "dominant_axis": "dominance",
    "turn": "turn_balance",
    "turn_balance": "turn_balance",
    "turn_bias": "turn_balance",
    "action": "action",
    "expert_action": "action",
}
REBALANCE_FAMILY_LABELS = {
    "dominance": {"balanced", "east_west_dominant", "north_south_dominant"},
    "turn_balance": {"balanced", "left_dominant", "through_dominant"},
    "action": {"ETWT", "NTST", "ELWL", "NLSL"},
}


def _normalize_text(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokenize(text: str) -> List[str]:
    words = text.split()
    if not words:
        return []
    tokens = list(words)
    tokens.extend(f"{words[idx]}_{words[idx + 1]}" for idx in range(len(words) - 1))
    return tokens


def _simhash(tokens: Iterable[str]) -> int:
    vec = [0] * 64
    for token in tokens:
        hashed = int(hashlib.md5(token.encode("utf-8")).hexdigest()[:16], 16)
        for bit in range(64):
            if (hashed >> bit) & 1:
                vec[bit] += 1
            else:
                vec[bit] -= 1
    output = 0
    for bit, value in enumerate(vec):
        if value >= 0:
            output |= 1 << bit
    return output


def _hamming(a: int, b: int) -> int:
    value = a ^ b
    try:
        return value.bit_count()
    except AttributeError:
        return bin(value).count("1")


def _band_keys(hash64: int, bands: int) -> List[tuple[int, int]]:
    bands = max(1, int(bands))
    bits_per_band = max(1, 64 // bands)
    keys: List[tuple[int, int]] = []
    for band_idx in range(bands):
        start = band_idx * bits_per_band
        end = 64 if band_idx == bands - 1 else min(64, (band_idx + 1) * bits_per_band)
        mask = (1 << max(1, end - start)) - 1
        keys.append((band_idx, (hash64 >> start) & mask))
    return keys


def _action_margin_from_record(record: Dict) -> Optional[float]:
    explicit = record.get("expert_action_margin")
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            return None
    waits = record.get("candidate_action_waits")
    if not isinstance(waits, dict) or len(waits) < 2:
        return None
    values: List[float] = []
    for value in waits.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if len(values) < 2:
        return None
    values.sort()
    return float(values[1] - values[0])


def _is_similar_caption(
    caption: str,
    expert_action: str,
    seen_hashes: List[int],
    bucket: Dict[tuple[int, int], List[int]],
    seen_exact: set[str],
    hamming_threshold: int,
    bands: int,
) -> bool:
    normalized_caption = _normalize_text(caption)
    normalized_action = _normalize_text(expert_action)
    normalized = f"{normalized_action} || {normalized_caption}".strip()
    if not normalized:
        return False
    if normalized in seen_exact:
        return True
    tokens = _tokenize(normalized)
    if not tokens:
        return False
    hash64 = _simhash(tokens)
    candidate_ids = set()
    for key in _band_keys(hash64, bands):
        candidate_ids.update(bucket.get(key, []))
    for idx in candidate_ids:
        if _hamming(hash64, seen_hashes[idx]) <= hamming_threshold:
            return True
    idx = len(seen_hashes)
    seen_hashes.append(hash64)
    seen_exact.add(normalized)
    for key in _band_keys(hash64, bands):
        bucket[key].append(idx)
    return False


def _parse_movement_keys(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_MOVEMENT_KEYS)
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    return keys if keys else list(DEFAULT_MOVEMENT_KEYS)


def _parse_rebalance_targets(raw: str) -> List[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _flat_target_key(family: str, label: str) -> str:
    return f"{family}:{label}"


def _safe_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _sequence_window(counts_history: List[dict], seq_len: int) -> List[dict]:
    if not counts_history:
        return []
    window = list(counts_history[-seq_len:])
    if len(window) < seq_len:
        pad = window[0] if window else {}
        window = [pad] * (seq_len - len(window)) + window
    return window


def _sequence_profile(
    counts_history: List[dict],
    seq_len: int,
    movement_keys: Iterable[str],
) -> dict:
    window = _sequence_window(counts_history, seq_len=seq_len)
    total = 0.0
    ew_sum = 0.0
    ns_sum = 0.0
    through_sum = 0.0
    left_sum = 0.0
    for counts in window:
        for key in movement_keys:
            value = _safe_float((counts or {}).get(key, 0.0))
            total += value
            if key.startswith(("E", "W")):
                ew_sum += value
            if key.startswith(("N", "S")):
                ns_sum += value
            if key.endswith("T"):
                through_sum += value
            if key.endswith("L"):
                left_sum += value
    direction_total = max(ew_sum + ns_sum, 1.0)
    ew_share = ew_sum / direction_total
    ns_share = ns_sum / direction_total
    if abs(ew_share - ns_share) < 0.1:
        dominance_label = "balanced"
    elif ew_share > ns_share:
        dominance_label = "east_west_dominant"
    else:
        dominance_label = "north_south_dominant"
    if through_sum > left_sum * 1.2:
        turn_label = "through_dominant"
    elif left_sum > through_sum * 1.2:
        turn_label = "left_dominant"
    else:
        turn_label = "balanced"
    return {
        "sequence_total": float(total),
        "dominance_label": dominance_label,
        "turn_label": turn_label,
    }


def _sequence_total(
    counts_history: List[dict],
    seq_len: int,
    movement_keys: Iterable[str],
) -> float:
    return float(_sequence_profile(counts_history, seq_len, movement_keys)["sequence_total"])


def _should_keep(
    total: float,
    drop_zero_seq: bool,
    min_total: Optional[float],
    max_total: Optional[float],
) -> bool:
    if drop_zero_seq and total == 0.0:
        return False
    if min_total is not None and total < min_total:
        return False
    if max_total is not None and total > max_total:
        return False
    return True


def _resolve_rebalance_target(raw: str) -> tuple[str, str]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty rebalance target")
    if ":" in text:
        family_raw, label = text.split(":", 1)
        family = REBALANCE_FAMILY_ALIASES.get(family_raw.strip().lower())
        label = label.strip()
        if not family:
            raise ValueError(f"unknown rebalance target family: {family_raw}")
        if label not in REBALANCE_FAMILY_LABELS[family]:
            raise ValueError(f"unknown rebalance target label: {text}")
        return family, label
    if text in REBALANCE_FAMILY_LABELS["action"]:
        return "action", text
    if text in {"east_west_dominant", "north_south_dominant"}:
        return "dominance", text
    if text in {"left_dominant", "through_dominant"}:
        return "turn_balance", text
    if text == "balanced":
        raise ValueError("ambiguous rebalance target 'balanced'; use dominance:balanced or turn_balance:balanced")
    raise ValueError(f"unsupported rebalance target: {text}")


def _resolve_rebalance_targets(raw_targets: Iterable[str]) -> List[tuple[str, str]]:
    resolved: List[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_targets:
        for item in str(raw or "").split(","):
            item = item.strip()
            if not item:
                continue
            target = _resolve_rebalance_target(item)
            if target in seen:
                continue
            resolved.append(target)
            seen.add(target)
    return resolved


def _make_group_entry(
    group_lines: List[str],
    final_record: Dict,
    profile: dict,
) -> dict:
    return {
        "lines": list(group_lines),
        "labels": {
            "dominance": str(profile.get("dominance_label") or "balanced"),
            "turn_balance": str(profile.get("turn_label") or "balanced"),
            "action": str(final_record.get("expert_action") or "").strip(),
        },
    }


def _reference_target_count(counter: Counter, reference: str) -> int:
    counts = sorted((int(value) for value in counter.values()), reverse=True)
    if not counts:
        return 0
    if reference == "max":
        return counts[0]
    if reference == "second_largest":
        return counts[1] if len(counts) > 1 else counts[0]
    raise ValueError(f"unsupported rebalance reference: {reference}")


def _target_count_snapshot(
    label_counts: Dict[str, Counter],
    targets: Iterable[tuple[str, str]],
) -> Dict[str, int]:
    return {
        _flat_target_key(family, label): int(label_counts.get(family, Counter()).get(label, 0))
        for family, label in targets
    }


def _active_target_match_count(labels: Dict[str, str], deficits: Dict[tuple[str, str], int]) -> int:
    score = 0
    for (family, label), deficit in deficits.items():
        if deficit > 0 and labels.get(family) == label:
            score += 1
    return score


def _inactive_target_match_count(labels: Dict[str, str], deficits: Dict[tuple[str, str], int]) -> int:
    score = 0
    for (family, label), deficit in deficits.items():
        if deficit <= 0 and labels.get(family) == label:
            score += 1
    return score


def _rebalance_groups(
    groups: List[dict],
    targets: List[tuple[str, str]],
    reference: str,
    max_extra_copies_per_group: int,
    seed: int,
) -> tuple[List[dict], dict]:
    base_groups = list(groups)
    label_counts: Dict[str, Counter] = {
        "dominance": Counter(),
        "turn_balance": Counter(),
        "action": Counter(),
    }
    target_buckets: Dict[tuple[str, str], List[int]] = defaultdict(list)
    for group_idx, group in enumerate(base_groups):
        labels = group.get("labels") or {}
        for family, label in labels.items():
            if label:
                label_counts[family][label] += 1
        for target in targets:
            family, label = target
            if labels.get(family) == label:
                target_buckets[target].append(group_idx)

    before_snapshot = _target_count_snapshot(label_counts, targets)
    desired_targets: Dict[tuple[str, str], int] = {}
    max_extra_copies_per_group = max(0, int(max_extra_copies_per_group))
    for family, label in targets:
        current = int(label_counts.get(family, Counter()).get(label, 0))
        family_reference = _reference_target_count(label_counts.get(family, Counter()), reference=reference)
        reachable = current * (1 + max_extra_copies_per_group)
        desired_targets[(family, label)] = min(family_reference, reachable) if current > 0 else 0

    deficits = {
        key: max(0, desired_targets[key] - before_snapshot.get(_flat_target_key(*key), 0))
        for key in desired_targets
    }
    rng = random.Random(seed)
    added_group_indices: List[int] = []
    added_per_group: Counter = Counter()
    while True:
        active_choices: List[tuple[tuple[str, str], int, List[int]]] = []
        for key, deficit in deficits.items():
            if deficit <= 0:
                continue
            candidates = [
                idx
                for idx in target_buckets.get(key, [])
                if added_per_group[idx] < max_extra_copies_per_group
            ]
            if candidates:
                active_choices.append((key, deficit, candidates))
        if not active_choices:
            break
        active_choices.sort(key=lambda item: (-item[1], item[0][0], item[0][1]))
        _, _, candidates = active_choices[0]
        best_score = max(
            _active_target_match_count(base_groups[idx].get("labels") or {}, deficits) for idx in candidates
        )
        best_candidates = [
            idx
            for idx in candidates
            if _active_target_match_count(base_groups[idx].get("labels") or {}, deficits) == best_score
        ]
        lowest_overshoot = min(
            _inactive_target_match_count(base_groups[idx].get("labels") or {}, deficits)
            for idx in best_candidates
        )
        best_candidates = [
            idx
            for idx in best_candidates
            if _inactive_target_match_count(base_groups[idx].get("labels") or {}, deficits) == lowest_overshoot
        ]
        chosen_idx = rng.choice(best_candidates)
        added_group_indices.append(chosen_idx)
        added_per_group[chosen_idx] += 1
        chosen_labels = base_groups[chosen_idx].get("labels") or {}
        for key in list(deficits):
            family, label = key
            if deficits[key] > 0 and chosen_labels.get(family) == label:
                deficits[key] -= 1

    output_groups = list(base_groups)
    after_counts: Dict[str, Counter] = {
        family: Counter(counter)
        for family, counter in label_counts.items()
    }
    for group_idx in added_group_indices:
        output_groups.append(base_groups[group_idx])
        labels = base_groups[group_idx].get("labels") or {}
        for family, label in labels.items():
            if label:
                after_counts[family][label] += 1

    return output_groups, {
        "base_groups": int(len(base_groups)),
        "added_groups": int(len(added_group_indices)),
        "output_groups": int(len(output_groups)),
        "target_counts_before": _target_count_snapshot(label_counts, targets),
        "target_counts_after": _target_count_snapshot(after_counts, targets),
        "target_counts_target": {
            _flat_target_key(family, label): int(desired_targets[(family, label)])
            for family, label in targets
        },
    }


def _write_group_lines(fout, group_lines: List[str]) -> int:
    written = 0
    for line in group_lines:
        fout.write(line + "\n")
        written += 1
    return written


def clean_jsonl(
    inputs: List[str],
    output_path: str,
    seq_len: int = 16,
    movement_keys: Optional[List[str]] = None,
    drop_zero_seq: bool = True,
    min_total: Optional[float] = None,
    max_total: Optional[float] = None,
    keep_incomplete: bool = False,
    min_action_margin: Optional[float] = None,
    dedup_similar_captions: bool = False,
    caption_hamming_threshold: int = 3,
    caption_bands: int = 4,
    rebalance_targets: Optional[List[str]] = None,
    rebalance_reference: str = "second_largest",
    rebalance_max_extra_copies_per_group: int = 4,
    rebalance_seed: int = 7,
) -> dict:
    movement_keys = movement_keys or list(DEFAULT_MOVEMENT_KEYS)
    resolved_rebalance_targets = _resolve_rebalance_targets(rebalance_targets or [])
    rebalance_enabled = bool(resolved_rebalance_targets)

    stats = {
        "total_lines": 0,
        "bad_lines": 0,
        "total_groups": 0,
        "kept_groups": 0,
        "dropped_zero_groups": 0,
        "dropped_range_groups": 0,
        "written_lines": 0,
        "incomplete_groups_dropped": 0,
        "incomplete_groups_kept": 0,
        "dropped_low_margin_groups": 0,
        "dropped_duplicate_caption_groups": 0,
        "rebalance_enabled": rebalance_enabled,
        "rebalance_targets": [_flat_target_key(family, label) for family, label in resolved_rebalance_targets],
        "rebalance_base_groups": 0,
        "rebalance_added_groups": 0,
        "rebalance_output_groups": 0,
        "rebalance_target_counts_before": {},
        "rebalance_target_counts_after": {},
        "rebalance_target_counts_target": {},
    }
    seen_hashes: List[int] = []
    caption_bucket: Dict[tuple[int, int], List[int]] = defaultdict(list)
    seen_exact_captions: set[str] = set()
    kept_groups_buffer: List[dict] = []

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for src in inputs:
            current_group: List[dict] = []
            current_group_lines: List[str] = []
            counts_history: List[dict] = []
            with open(src, "r", encoding="utf-8", errors="replace") as fin:
                for line in fin:
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
                    current_group_lines.append(json.dumps(record, ensure_ascii=False))
                    counts_history.append(record.get("movement_counts") or {})
                    clip_caption = record.get("clip_caption") or ""
                    if not clip_caption:
                        continue

                    stats["total_groups"] += 1
                    profile = _sequence_profile(counts_history, seq_len, movement_keys)
                    total = float(profile["sequence_total"])
                    keep = _should_keep(total, drop_zero_seq, min_total, max_total)
                    margin = _action_margin_from_record(record)
                    if keep and min_action_margin is not None:
                        if margin is None or float(margin) < float(min_action_margin):
                            keep = False
                            stats["dropped_low_margin_groups"] += 1
                    if keep and dedup_similar_captions:
                        if _is_similar_caption(
                            clip_caption,
                            str(record.get("expert_action") or ""),
                            seen_hashes=seen_hashes,
                            bucket=caption_bucket,
                            seen_exact=seen_exact_captions,
                            hamming_threshold=caption_hamming_threshold,
                            bands=caption_bands,
                        ):
                            keep = False
                            stats["dropped_duplicate_caption_groups"] += 1
                    if keep:
                        stats["kept_groups"] += 1
                        if rebalance_enabled:
                            kept_groups_buffer.append(_make_group_entry(current_group_lines, record, profile))
                        else:
                            stats["written_lines"] += _write_group_lines(fout, current_group_lines)
                    else:
                        if drop_zero_seq and total == 0.0:
                            stats["dropped_zero_groups"] += 1
                        else:
                            stats["dropped_range_groups"] += 1

                    current_group = []
                    current_group_lines = []
                    counts_history = []

            if current_group:
                if keep_incomplete:
                    profile = _sequence_profile(counts_history, seq_len, movement_keys)
                    total = float(profile["sequence_total"])
                    keep = _should_keep(total, drop_zero_seq, min_total, max_total)
                    if keep:
                        if rebalance_enabled:
                            kept_groups_buffer.append(_make_group_entry(current_group_lines, current_group[-1], profile))
                        else:
                            stats["written_lines"] += _write_group_lines(fout, current_group_lines)
                        stats["incomplete_groups_kept"] += 1
                    else:
                        stats["incomplete_groups_dropped"] += 1
                else:
                    stats["incomplete_groups_dropped"] += 1

        if rebalance_enabled:
            output_groups, rebalance_summary = _rebalance_groups(
                groups=kept_groups_buffer,
                targets=resolved_rebalance_targets,
                reference=rebalance_reference,
                max_extra_copies_per_group=rebalance_max_extra_copies_per_group,
                seed=rebalance_seed,
            )
            for group in output_groups:
                stats["written_lines"] += _write_group_lines(fout, group.get("lines") or [])
            stats["rebalance_base_groups"] = int(rebalance_summary["base_groups"])
            stats["rebalance_added_groups"] = int(rebalance_summary["added_groups"])
            stats["rebalance_output_groups"] = int(rebalance_summary["output_groups"])
            stats["rebalance_target_counts_before"] = dict(rebalance_summary["target_counts_before"])
            stats["rebalance_target_counts_after"] = dict(rebalance_summary["target_counts_after"])
            stats["rebalance_target_counts_target"] = dict(rebalance_summary["target_counts_target"])

    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="input jsonl paths")
    parser.add_argument(
        "--output",
        default="data/recorded/align_train.cleaned.jsonl",
        help="output jsonl path",
    )
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument(
        "--movement_keys",
        type=str,
        default=",".join(DEFAULT_MOVEMENT_KEYS),
        help="comma-separated movement keys",
    )
    parser.add_argument("--drop_zero_seq", action="store_true", default=True)
    parser.add_argument("--min_total", type=float, default=None)
    parser.add_argument("--max_total", type=float, default=None)
    parser.add_argument("--keep_incomplete", action="store_true")
    parser.add_argument("--min_action_margin", type=float, default=None)
    parser.add_argument("--dedup_similar_captions", action="store_true")
    parser.add_argument("--caption_hamming_threshold", type=int, default=3)
    parser.add_argument("--caption_bands", type=int, default=4)
    parser.add_argument(
        "--rebalance_targets",
        type=str,
        default="",
        help=(
            "comma-separated targets to oversample, e.g. "
            f"'{','.join(DEFAULT_REBALANCE_TARGETS)}'; supports action labels or family:label"
        ),
    )
    parser.add_argument(
        "--rebalance_reference",
        choices=("second_largest", "max"),
        default="second_largest",
        help="target count reference within the same label family",
    )
    parser.add_argument(
        "--rebalance_max_extra_copies_per_group",
        type=int,
        default=4,
        help="max additional copies allowed for the same sequence group",
    )
    parser.add_argument(
        "--rebalance_seed",
        type=int,
        default=7,
        help="random seed used by deterministic oversampling",
    )
    args = parser.parse_args()

    stats = clean_jsonl(
        inputs=args.inputs,
        output_path=args.output,
        seq_len=args.seq_len,
        movement_keys=_parse_movement_keys(args.movement_keys),
        drop_zero_seq=args.drop_zero_seq,
        min_total=args.min_total,
        max_total=args.max_total,
        keep_incomplete=args.keep_incomplete,
        min_action_margin=args.min_action_margin,
        dedup_similar_captions=args.dedup_similar_captions,
        caption_hamming_threshold=args.caption_hamming_threshold,
        caption_bands=args.caption_bands,
        rebalance_targets=_parse_rebalance_targets(args.rebalance_targets),
        rebalance_reference=args.rebalance_reference,
        rebalance_max_extra_copies_per_group=args.rebalance_max_extra_copies_per_group,
        rebalance_seed=args.rebalance_seed,
    )

    print("done:")
    for key in (
        "total_lines",
        "bad_lines",
        "total_groups",
        "kept_groups",
        "dropped_zero_groups",
        "dropped_range_groups",
        "written_lines",
        "incomplete_groups_dropped",
        "incomplete_groups_kept",
        "dropped_low_margin_groups",
        "dropped_duplicate_caption_groups",
    ):
        print(f"  {key}={stats[key]}")
    if stats.get("rebalance_enabled"):
        print(f"  rebalance_targets={','.join(stats['rebalance_targets'])}")
        print(f"  rebalance_base_groups={stats['rebalance_base_groups']}")
        print(f"  rebalance_added_groups={stats['rebalance_added_groups']}")
        print(f"  rebalance_output_groups={stats['rebalance_output_groups']}")
        print(
            "  rebalance_target_counts_before="
            + json.dumps(stats["rebalance_target_counts_before"], ensure_ascii=False, sort_keys=True)
        )
        print(
            "  rebalance_target_counts_target="
            + json.dumps(stats["rebalance_target_counts_target"], ensure_ascii=False, sort_keys=True)
        )
        print(
            "  rebalance_target_counts_after="
            + json.dumps(stats["rebalance_target_counts_after"], ensure_ascii=False, sort_keys=True)
        )


if __name__ == "__main__":
    main()
