#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable


DEFAULT_MOVEMENT_KEYS = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")


def _normalize_text(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"[^\w\s]+", " ", cleaned, flags=re.UNICODE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _sequence_window(
    history: list[dict[str, Any]],
    seq_len: int,
    movement_keys: Iterable[str],
) -> list[tuple[float, ...]]:
    window = list(history[-seq_len:])
    if len(window) < seq_len:
        pad = window[0] if window else {}
        window = [pad for _ in range(seq_len - len(window))] + window
    return [
        tuple(float((counts or {}).get(key, 0.0) or 0.0) for key in movement_keys)
        for counts in window
    ]


def _summarize_numeric(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(float(value) for value in values)
    total = len(ordered)

    def quantile(q: float) -> float:
        if total == 1:
            return ordered[0]
        index = (total - 1) * q
        lower = int(math.floor(index))
        upper = int(math.ceil(index))
        if lower == upper:
            return ordered[lower]
        weight = index - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": float(total),
        "mean": round(mean(ordered), 3),
        "min": round(ordered[0], 3),
        "p10": round(quantile(0.10), 3),
        "p50": round(quantile(0.50), 3),
        "p90": round(quantile(0.90), 3),
        "p99": round(quantile(0.99), 3),
        "max": round(ordered[-1], 3),
    }


def _margin_from_record(record: dict[str, Any]) -> float | None:
    explicit = record.get("expert_action_margin")
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            return None
    waits = record.get("candidate_action_waits")
    if not isinstance(waits, dict):
        return None
    values: list[float] = []
    for value in waits.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if len(values) < 2:
        return None
    values.sort()
    return float(values[1] - values[0])


def _safe_preview(text: str, limit: int = 96) -> str:
    compact = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _counter_to_items(counter: Counter, top_k: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for key, value in counter.most_common(top_k):
        items.append({"value": key, "count": int(value)})
    return items


def analyze_dataset(
    path: Path,
    seq_len: int,
    movement_keys: tuple[str, ...],
    top_k: int,
) -> dict[str, Any]:
    total_lines = 0
    bad_lines = 0
    dangling_lines = 0
    total_groups = 0

    line_field_counts: Counter[str] = Counter()
    group_field_counts: Counter[str] = Counter()
    interval_counter: Counter[int] = Counter()
    group_size_counter: Counter[int] = Counter()
    line_intersections: Counter[str] = Counter()
    group_intersections: Counter[str] = Counter()
    action_counter: Counter[str] = Counter()
    movement_counter: Counter[str] = Counter()
    dominance_counter: Counter[str] = Counter()
    turn_balance_counter: Counter[str] = Counter()
    exact_caption_counter: Counter[str] = Counter()
    normalized_caption_counter: Counter[str] = Counter()
    caption_prefix_counter: Counter[str] = Counter()

    group_sizes: list[float] = []
    caption_char_lens: list[float] = []
    caption_word_lens: list[float] = []
    sequence_totals: list[float] = []
    last_step_totals: list[float] = []
    delta_totals: list[float] = []
    variation_totals: list[float] = []
    action_margins: list[float] = []

    sequence_counter: Counter[tuple[tuple[float, ...], ...]] = Counter()
    sequence_to_captions: defaultdict[tuple[tuple[float, ...], ...], set[str]] = defaultdict(set)
    caption_to_sequences: defaultdict[str, set[tuple[tuple[float, ...], ...]]] = defaultdict(set)

    current_group: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            total_lines += 1
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if not isinstance(record, dict):
                bad_lines += 1
                continue

            for field in ("prompt", "clip_caption", "expert_action", "candidate_action_waits", "snapshot_path"):
                value = record.get(field)
                if isinstance(value, dict):
                    if value:
                        line_field_counts[field] += 1
                elif str(value or "").strip():
                    line_field_counts[field] += 1

            intersection_id = str(record.get("intersection_id") or record.get("road_id") or "")
            line_intersections[intersection_id or "__missing__"] += 1
            interval_value = record.get("interval_s")
            try:
                interval_counter[int(interval_value)] += 1
            except (TypeError, ValueError):
                interval_counter[-1] += 1

            movement_counts = record.get("movement_counts") or {}
            for key in movement_keys:
                movement_counter[key] += float(movement_counts.get(key, 0.0) or 0.0)

            current_group.append(record)
            if not str(record.get("clip_caption") or "").strip():
                continue

            total_groups += 1
            group_sizes.append(float(len(current_group)))
            group_size_counter[len(current_group)] += 1

            final_record = current_group[-1]
            for field in ("prompt", "clip_caption", "expert_action", "candidate_action_waits", "snapshot_path"):
                value = final_record.get(field)
                if isinstance(value, dict):
                    if value:
                        group_field_counts[field] += 1
                elif str(value or "").strip():
                    group_field_counts[field] += 1

            group_intersection_values = {
                str(item.get("intersection_id") or item.get("road_id") or "") or "__missing__"
                for item in current_group
            }
            if len(group_intersection_values) == 1:
                group_intersections[next(iter(group_intersection_values))] += 1
            else:
                group_intersections["__mixed__"] += 1

            history = [item.get("movement_counts") or {} for item in current_group]
            window = _sequence_window(history, seq_len=seq_len, movement_keys=movement_keys)
            sequence_signature = tuple(window)
            sequence_counter[sequence_signature] += 1

            caption = str(final_record.get("clip_caption") or "").strip()
            normalized_caption = _normalize_text(caption)
            exact_caption_counter[caption] += 1
            normalized_caption_counter[normalized_caption] += 1
            sequence_to_captions[sequence_signature].add(normalized_caption)
            caption_to_sequences[normalized_caption].add(sequence_signature)

            prefix_tokens = normalized_caption.split()
            prefix = " ".join(prefix_tokens[:6]) if prefix_tokens else "__empty__"
            caption_prefix_counter[prefix] += 1

            caption_char_lens.append(float(len(caption)))
            caption_word_lens.append(float(len(caption.split())))

            sequence_total = float(sum(sum(step) for step in window))
            first_total = float(sum(window[0])) if window else 0.0
            last_total = float(sum(window[-1])) if window else 0.0
            variation_total = 0.0
            for prev, curr in zip(window, window[1:]):
                variation_total += abs(sum(curr) - sum(prev))
            sequence_totals.append(sequence_total)
            last_step_totals.append(last_total)
            delta_totals.append(last_total - first_total)
            variation_totals.append(variation_total)

            ew_sum = 0.0
            ns_sum = 0.0
            through_sum = 0.0
            left_sum = 0.0
            for step in window:
                for idx, key in enumerate(movement_keys):
                    value = step[idx]
                    if key.startswith(("E", "W")):
                        ew_sum += value
                    if key.startswith(("N", "S")):
                        ns_sum += value
                    if key.endswith("T"):
                        through_sum += value
                    if key.endswith("L"):
                        left_sum += value
            total_dir = max(ew_sum + ns_sum, 1.0)
            ew_share = ew_sum / total_dir
            ns_share = ns_sum / total_dir
            if abs(ew_share - ns_share) < 0.1:
                dominance_counter["balanced"] += 1
            elif ew_share > ns_share:
                dominance_counter["east_west_dominant"] += 1
            else:
                dominance_counter["north_south_dominant"] += 1
            if through_sum > left_sum * 1.2:
                turn_balance_counter["through_dominant"] += 1
            elif left_sum > through_sum * 1.2:
                turn_balance_counter["left_dominant"] += 1
            else:
                turn_balance_counter["balanced"] += 1

            action = str(final_record.get("expert_action") or "").strip()
            if action:
                action_counter[action] += 1
            margin = _margin_from_record(final_record)
            if margin is not None:
                action_margins.append(float(margin))

            current_group = []

    dangling_lines = len(current_group)

    repeated_sequence_groups = int(sum(count - 1 for count in sequence_counter.values() if count > 1))
    repeated_caption_groups = int(sum(count - 1 for count in normalized_caption_counter.values() if count > 1))
    sequence_keys_with_multi_caption = sum(
        1 for captions in sequence_to_captions.values() if len([c for c in captions if c]) > 1
    )
    caption_keys_with_multi_sequence = sum(
        1 for sequences in caption_to_sequences.values() if len(sequences) > 1
    )

    movement_total_sum = float(sum(movement_counter.values()))
    movement_share = {
        key: round(float(movement_counter[key]) / movement_total_sum, 4) if movement_total_sum > 0 else 0.0
        for key in movement_keys
    }

    summary = {
        "path": str(path),
        "seq_len": int(seq_len),
        "movement_keys": list(movement_keys),
        "basic": {
            "total_lines": int(total_lines),
            "bad_lines": int(bad_lines),
            "total_groups": int(total_groups),
            "dangling_lines_without_terminal_clip_caption": int(dangling_lines),
        },
        "line_fields": {
            "non_empty_counts": {key: int(value) for key, value in sorted(line_field_counts.items())},
            "coverage_ratio": {
                key: round(float(value) / max(total_lines, 1), 4)
                for key, value in sorted(line_field_counts.items())
            },
        },
        "group_terminal_fields": {
            "non_empty_counts": {key: int(value) for key, value in sorted(group_field_counts.items())},
            "coverage_ratio": {
                key: round(float(value) / max(total_groups, 1), 4)
                for key, value in sorted(group_field_counts.items())
            },
        },
        "group_structure": {
            "group_size_summary": _summarize_numeric(group_sizes),
            "group_size_counts_top": _counter_to_items(group_size_counter, top_k=top_k),
            "groups_exactly_seq_len": int(group_size_counter.get(seq_len, 0)),
            "groups_shorter_than_seq_len": int(sum(v for k, v in group_size_counter.items() if k < seq_len)),
            "groups_longer_than_seq_len": int(sum(v for k, v in group_size_counter.items() if k > seq_len)),
        },
        "interval_distribution": _counter_to_items(interval_counter, top_k=top_k),
        "intersection_distribution": {
            "line_top": _counter_to_items(line_intersections, top_k=top_k),
            "group_top": _counter_to_items(group_intersections, top_k=top_k),
        },
        "action_distribution": _counter_to_items(action_counter, top_k=top_k),
        "action_margin_summary": _summarize_numeric(action_margins),
        "caption": {
            "char_length_summary": _summarize_numeric(caption_char_lens),
            "word_length_summary": _summarize_numeric(caption_word_lens),
            "unique_exact": int(len(exact_caption_counter)),
            "unique_normalized": int(len(normalized_caption_counter)),
            "repeated_normalized_groups": repeated_caption_groups,
            "top_repeated_normalized": [
                {"caption": _safe_preview(text), "count": int(count)}
                for text, count in normalized_caption_counter.most_common(top_k)
            ],
            "top_prefixes": [
                {"prefix": _safe_preview(prefix), "count": int(count)}
                for prefix, count in caption_prefix_counter.most_common(top_k)
            ],
        },
        "traffic": {
            "sequence_total_summary": _summarize_numeric(sequence_totals),
            "last_step_total_summary": _summarize_numeric(last_step_totals),
            "delta_total_summary": _summarize_numeric(delta_totals),
            "variation_total_summary": _summarize_numeric(variation_totals),
            "movement_share": movement_share,
            "direction_balance": _counter_to_items(dominance_counter, top_k=top_k),
            "turn_balance": _counter_to_items(turn_balance_counter, top_k=top_k),
            "zero_total_groups": int(sum(1 for value in sequence_totals if value == 0.0)),
        },
        "duplicates": {
            "unique_sequences": int(len(sequence_counter)),
            "repeated_sequence_groups": repeated_sequence_groups,
            "sequence_keys_with_multiple_captions": int(sequence_keys_with_multi_caption),
            "caption_keys_with_multiple_sequences": int(caption_keys_with_multi_sequence),
        },
    }
    return summary


def _print_text_report(summary: dict[str, Any], top_k: int) -> None:
    basic = summary["basic"]
    group_structure = summary["group_structure"]
    caption = summary["caption"]
    traffic = summary["traffic"]
    duplicates = summary["duplicates"]

    print("== Basic ==")
    print(json.dumps(basic, ensure_ascii=False, indent=2))

    print("\n== Group Structure ==")
    print(json.dumps(group_structure, ensure_ascii=False, indent=2))

    print("\n== Line Fields ==")
    print(json.dumps(summary["line_fields"], ensure_ascii=False, indent=2))

    print("\n== Group Terminal Fields ==")
    print(json.dumps(summary["group_terminal_fields"], ensure_ascii=False, indent=2))

    print("\n== Intersections ==")
    print(json.dumps(summary["intersection_distribution"], ensure_ascii=False, indent=2))

    print("\n== Actions ==")
    print(json.dumps(summary["action_distribution"][:top_k], ensure_ascii=False, indent=2))
    print(json.dumps(summary["action_margin_summary"], ensure_ascii=False, indent=2))

    print("\n== Captions ==")
    print(json.dumps(caption, ensure_ascii=False, indent=2))

    print("\n== Traffic ==")
    print(json.dumps(traffic, ensure_ascii=False, indent=2))

    print("\n== Duplicates ==")
    print(json.dumps(duplicates, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze align/stage dataset distribution from JSONL groups.")
    parser.add_argument("--input", type=str, required=True, help="Path to cleaned align jsonl with clip_caption groups.")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--movement_keys",
        type=str,
        default=",".join(DEFAULT_MOVEMENT_KEYS),
        help="Comma-separated movement key order.",
    )
    parser.add_argument("--format", type=str, choices=("text", "json"), default="text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movement_keys = tuple(key.strip() for key in args.movement_keys.split(",") if key.strip()) or DEFAULT_MOVEMENT_KEYS
    summary = analyze_dataset(
        path=Path(args.input),
        seq_len=int(args.seq_len),
        movement_keys=movement_keys,
        top_k=int(args.top_k),
    )
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    _print_text_report(summary, top_k=int(args.top_k))


if __name__ == "__main__":
    main()
