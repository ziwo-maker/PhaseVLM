#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


MOVEMENT_KEYS = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    idx = (len(ordered) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return ordered[lo]
    weight = idx - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def _rank_ratio(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    clipped = min(max(float(value), low), high)
    return (clipped - low) / (high - low)


def _normalize_text(text: str) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9\s]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _counts_to_vector(counts: dict[str, Any]) -> tuple[float, ...]:
    return tuple(float((counts or {}).get(key, 0.0) or 0.0) for key in MOVEMENT_KEYS)


def _action_margin(record: dict[str, Any]) -> float:
    explicit = record.get("expert_action_margin")
    if explicit is not None:
        try:
            return float(explicit)
        except (TypeError, ValueError):
            return 0.0
    waits = record.get("candidate_action_waits")
    if not isinstance(waits, dict):
        return 0.0
    values: list[float] = []
    for value in waits.values():
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if len(values) < 2:
        return 0.0
    values.sort()
    return float(values[1] - values[0])


@dataclass
class GroupSample:
    raw_lines: list[str]
    records: list[dict[str, Any]]
    final_record: dict[str, Any]
    intersection_id: str
    caption: str
    normalized_caption: str
    expert_action: str
    window: list[tuple[float, ...]]
    last_total: float
    seq_total: float
    delta_total: float
    variation_total: float
    ew_share: float
    through_share: float
    dominance: str
    turn_balance: str
    action_margin: float
    caption_words: int
    source_index: int
    volume_bin: str = ""
    variation_bin: str = ""
    trend_bin: str = ""
    coarse_bucket: tuple[str, ...] = field(default_factory=tuple)
    sequence_signature: tuple[tuple[float, ...], ...] = field(default_factory=tuple)
    score: float = 0.0


def _read_groups(path: Path, seq_len: int) -> list[GroupSample]:
    groups: list[GroupSample] = []
    current_records: list[dict[str, Any]] = []
    current_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_idx, raw_line in enumerate(handle):
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            if not isinstance(record, dict):
                continue
            current_records.append(record)
            current_lines.append(raw_line.rstrip("\n"))
            caption = str(record.get("clip_caption") or "").strip()
            if not caption:
                continue

            history = [item.get("movement_counts") or {} for item in current_records]
            window_counts = history[-seq_len:]
            if len(window_counts) < seq_len:
                pad = window_counts[0] if window_counts else {}
                window_counts = [pad] * (seq_len - len(window_counts)) + window_counts
            window = [_counts_to_vector(counts) for counts in window_counts]
            totals = [sum(step) for step in window]
            ew_sum = sum(step[4] + step[5] + step[6] + step[7] for step in window)
            ns_sum = sum(step[0] + step[1] + step[2] + step[3] for step in window)
            through_sum = sum(step[0] + step[2] + step[4] + step[6] for step in window)
            left_sum = sum(step[1] + step[3] + step[5] + step[7] for step in window)
            axis_total = max(ew_sum + ns_sum, 1.0)
            turn_total = max(through_sum + left_sum, 1.0)
            ew_share = ew_sum / axis_total
            through_share = through_sum / turn_total
            if abs(ew_share - (1.0 - ew_share)) < 0.1:
                dominance = "balanced"
            elif ew_share > 0.5:
                dominance = "east_west_dominant"
            else:
                dominance = "north_south_dominant"
            if through_share > 0.75:
                turn_balance = "through_dominant"
            elif left_sum / turn_total > 0.4:
                turn_balance = "left_dominant"
            else:
                turn_balance = "balanced"

            groups.append(
                GroupSample(
                    raw_lines=list(current_lines),
                    records=list(current_records),
                    final_record=record,
                    intersection_id=str(record.get("intersection_id") or record.get("road_id") or ""),
                    caption=caption,
                    normalized_caption=_normalize_text(caption),
                    expert_action=str(record.get("expert_action") or "").strip(),
                    window=window,
                    last_total=float(totals[-1] if totals else 0.0),
                    seq_total=float(sum(totals)),
                    delta_total=float((totals[-1] - totals[0]) if totals else 0.0),
                    variation_total=float(sum(abs(b - a) for a, b in zip(totals, totals[1:]))),
                    ew_share=float(ew_share),
                    through_share=float(through_share),
                    dominance=dominance,
                    turn_balance=turn_balance,
                    action_margin=_action_margin(record),
                    caption_words=len(caption.split()),
                    source_index=len(groups),
                    sequence_signature=tuple(window),
                )
            )
            current_records = []
            current_lines = []
    return groups


def _assign_bins(groups: list[GroupSample]) -> None:
    last_totals = [g.last_total for g in groups]
    variations = [g.variation_total for g in groups]
    margins = [g.action_margin for g in groups]
    word_lens = [float(g.caption_words) for g in groups]
    abs_deltas = [abs(g.delta_total) for g in groups]

    last_lo, last_hi = _quantile(last_totals, 0.1), _quantile(last_totals, 0.9)
    var_lo, var_hi = _quantile(variations, 0.1), _quantile(variations, 0.9)
    margin_lo, margin_hi = _quantile(margins, 0.1), _quantile(margins, 0.9)
    words_lo, words_hi = _quantile(word_lens, 0.1), _quantile(word_lens, 0.9)
    delta_lo, delta_hi = _quantile(abs_deltas, 0.1), _quantile(abs_deltas, 0.9)

    last_q1, last_q3 = _quantile(last_totals, 0.25), _quantile(last_totals, 0.75)
    var_q1, var_q3 = _quantile(variations, 0.25), _quantile(variations, 0.75)

    for sample in groups:
        if sample.last_total <= last_q1:
            sample.volume_bin = "low"
        elif sample.last_total >= last_q3:
            sample.volume_bin = "high"
        else:
            sample.volume_bin = "mid"

        if sample.variation_total <= var_q1:
            sample.variation_bin = "low_var"
        elif sample.variation_total >= var_q3:
            sample.variation_bin = "high_var"
        else:
            sample.variation_bin = "mid_var"

        if sample.delta_total <= -5.0:
            sample.trend_bin = "easing"
        elif sample.delta_total >= 5.0:
            sample.trend_bin = "growing"
        else:
            sample.trend_bin = "stable"

        info_score = (
            0.35 * _rank_ratio(sample.last_total, last_lo, last_hi)
            + 0.30 * _rank_ratio(sample.variation_total, var_lo, var_hi)
            + 0.15 * _rank_ratio(abs(sample.delta_total), delta_lo, delta_hi)
            + 0.10 * _rank_ratio(sample.action_margin, margin_lo, margin_hi)
            + 0.10 * _rank_ratio(float(sample.caption_words), words_lo, words_hi)
        )
        if sample.volume_bin == "low" and sample.variation_bin == "low_var" and sample.trend_bin == "stable":
            info_score -= 0.15
        sample.score = float(info_score)
        sample.coarse_bucket = (
            sample.intersection_id,
            sample.dominance,
            sample.turn_balance,
            sample.volume_bin,
            sample.trend_bin,
        )


def _dedup_by_sequence(groups: Iterable[GroupSample]) -> list[GroupSample]:
    best_by_sequence: dict[tuple[tuple[float, ...], ...], GroupSample] = {}
    for sample in groups:
        existing = best_by_sequence.get(sample.sequence_signature)
        if existing is None or sample.score > existing.score or (
            sample.score == existing.score and sample.source_index < existing.source_index
        ):
            best_by_sequence[sample.sequence_signature] = sample
    return list(best_by_sequence.values())


def _select_subset(
    groups: list[GroupSample],
    target_groups: int,
    min_per_bucket: int,
    per_caption_cap: int,
    per_bucket_cap: int,
    per_intersection_cap: int,
) -> list[GroupSample]:
    ordered = sorted(groups, key=lambda g: (-g.score, g.source_index))
    selected: list[GroupSample] = []
    seen_ids: set[int] = set()
    caption_counts: Counter[str] = Counter()
    bucket_counts: Counter[tuple[str, ...]] = Counter()
    intersection_counts: Counter[str] = Counter()

    bucket_to_items: defaultdict[tuple[str, ...], list[GroupSample]] = defaultdict(list)
    for sample in ordered:
        bucket_to_items[sample.coarse_bucket].append(sample)

    for bucket, items in sorted(bucket_to_items.items()):
        kept = 0
        for sample in items:
            if len(selected) >= target_groups:
                break
            if kept >= min_per_bucket:
                break
            if caption_counts[sample.normalized_caption] >= per_caption_cap:
                continue
            if intersection_counts[sample.intersection_id] >= per_intersection_cap:
                continue
            selected.append(sample)
            seen_ids.add(sample.source_index)
            caption_counts[sample.normalized_caption] += 1
            bucket_counts[bucket] += 1
            intersection_counts[sample.intersection_id] += 1
            kept += 1

    for sample in ordered:
        if len(selected) >= target_groups:
            break
        if sample.source_index in seen_ids:
            continue
        if caption_counts[sample.normalized_caption] >= per_caption_cap:
            continue
        if bucket_counts[sample.coarse_bucket] >= per_bucket_cap:
            continue
        if intersection_counts[sample.intersection_id] >= per_intersection_cap:
            continue
        selected.append(sample)
        seen_ids.add(sample.source_index)
        caption_counts[sample.normalized_caption] += 1
        bucket_counts[sample.coarse_bucket] += 1
        intersection_counts[sample.intersection_id] += 1

    if len(selected) < target_groups:
        for sample in ordered:
            if len(selected) >= target_groups:
                break
            if sample.source_index in seen_ids:
                continue
            selected.append(sample)
            seen_ids.add(sample.source_index)

    return sorted(selected, key=lambda g: g.source_index)


def _summary(groups: list[GroupSample]) -> dict[str, Any]:
    return {
        "groups": len(groups),
        "unique_sequences": len({g.sequence_signature for g in groups}),
        "unique_normalized_captions": len({g.normalized_caption for g in groups}),
        "action_margin": _margin_summary([g.action_margin for g in groups]),
        "intersection_counts": dict(sorted(Counter(g.intersection_id for g in groups).items())),
        "dominance_counts": dict(sorted(Counter(g.dominance for g in groups).items())),
        "turn_balance_counts": dict(sorted(Counter(g.turn_balance for g in groups).items())),
        "volume_bin_counts": dict(sorted(Counter(g.volume_bin for g in groups).items())),
        "trend_bin_counts": dict(sorted(Counter(g.trend_bin for g in groups).items())),
        "action_counts": dict(sorted(Counter(g.expert_action for g in groups).items())),
        "top_caption_prefixes": Counter(" ".join(g.normalized_caption.split()[:6]) for g in groups).most_common(20),
    }


def _margin_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": round(float(ordered[0]), 6),
        "p25": round(float(_quantile(ordered, 0.25)), 6),
        "median": round(float(_quantile(ordered, 0.5)), 6),
        "p75": round(float(_quantile(ordered, 0.75)), 6),
        "max": round(float(ordered[-1]), 6),
        "mean": round(float(sum(ordered) / len(ordered)), 6),
        "le_0": int(sum(value <= 0.0 for value in ordered)),
        "le_0_5": int(sum(value <= 0.5 for value in ordered)),
        "le_1": int(sum(value <= 1.0 for value in ordered)),
    }


def build_compact_dataset(
    input_path: Path,
    output_path: Path,
    seq_len: int,
    target_groups: int,
    min_per_bucket: int,
    per_caption_cap: int,
    per_bucket_cap: int,
    per_intersection_cap: int | None,
    min_action_margin: float,
) -> dict[str, Any]:
    groups = _read_groups(input_path, seq_len=seq_len)
    if not groups:
        raise SystemExit(f"No terminal clip_caption groups found in {input_path}")
    _assign_bins(groups)
    filtered = [sample for sample in groups if sample.action_margin >= float(min_action_margin)]
    if not filtered:
        raise SystemExit(
            f"No groups remain after applying --min_action_margin={min_action_margin} to {input_path}"
        )
    deduped = _dedup_by_sequence(filtered)
    intersection_count = len({g.intersection_id for g in deduped}) or 1
    resolved_intersection_cap = (
        int(per_intersection_cap)
        if per_intersection_cap is not None and int(per_intersection_cap) > 0
        else max(8, math.ceil(target_groups / intersection_count * 1.35))
    )
    selected = _select_subset(
        deduped,
        target_groups=min(target_groups, len(deduped)),
        min_per_bucket=min_per_bucket,
        per_caption_cap=per_caption_cap,
        per_bucket_cap=per_bucket_cap,
        per_intersection_cap=resolved_intersection_cap,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in selected:
            for line in sample.raw_lines:
                handle.write(line)
                handle.write("\n")

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "seq_len": int(seq_len),
        "target_groups": int(target_groups),
        "min_action_margin": float(min_action_margin),
        "per_intersection_cap": int(resolved_intersection_cap),
        "before": _summary(groups),
        "after_margin_filter": _summary(filtered),
        "after_dedup_sequence": _summary(deduped),
        "after_selection": _summary(selected),
        "selected_lines": int(sum(len(sample.raw_lines) for sample in selected)),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact Stage-1 alignment subset from jsonl groups.")
    parser.add_argument("--input", required=True, help="input jsonl path")
    parser.add_argument("--output", required=True, help="output jsonl path")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--target_groups", type=int, default=2500)
    parser.add_argument("--min_per_bucket", type=int, default=1)
    parser.add_argument("--per_caption_cap", type=int, default=3)
    parser.add_argument("--per_bucket_cap", type=int, default=18)
    parser.add_argument("--per_intersection_cap", type=int, default=0)
    parser.add_argument(
        "--min_action_margin",
        type=float,
        default=0.0,
        help="Drop terminal groups whose expert_action_margin is below this value.",
    )
    args = parser.parse_args()

    summary = build_compact_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        seq_len=args.seq_len,
        target_groups=args.target_groups,
        min_per_bucket=args.min_per_bucket,
        per_caption_cap=args.per_caption_cap,
        per_bucket_cap=args.per_bucket_cap,
        per_intersection_cap=args.per_intersection_cap,
        min_action_margin=args.min_action_margin,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
