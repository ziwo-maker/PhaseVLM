from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional


DEFAULT_MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]


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


def _sequence_total(
    counts_history: List[dict],
    seq_len: int,
    movement_keys: Iterable[str],
) -> float:
    if not counts_history:
        return 0.0
    window = counts_history[-seq_len:]
    if len(window) < seq_len:
        pad = window[0] if window else {}
        window = [pad] * (seq_len - len(window)) + window
    total = 0.0
    for counts in window:
        total += sum(float(counts.get(k, 0.0) or 0.0) for k in movement_keys)
    return total


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
) -> dict:
    movement_keys = movement_keys or list(DEFAULT_MOVEMENT_KEYS)

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
    }
    seen_hashes: List[int] = []
    caption_bucket: Dict[tuple[int, int], List[int]] = defaultdict(list)
    seen_exact_captions: set[str] = set()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fout:
        for src in inputs:
            current_group: List[dict] = []
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
                    counts_history.append(record.get("movement_counts") or {})
                    clip_caption = record.get("clip_caption") or ""
                    if not clip_caption:
                        continue

                    stats["total_groups"] += 1
                    total = _sequence_total(counts_history, seq_len, movement_keys)
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
                        for item in current_group:
                            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stats["written_lines"] += 1
                        stats["kept_groups"] += 1
                    else:
                        if drop_zero_seq and total == 0.0:
                            stats["dropped_zero_groups"] += 1
                        else:
                            stats["dropped_range_groups"] += 1

                    current_group = []
                    counts_history = []

            if current_group:
                if keep_incomplete:
                    total = _sequence_total(counts_history, seq_len, movement_keys)
                    keep = _should_keep(total, drop_zero_seq, min_total, max_total)
                    if keep:
                        for item in current_group:
                            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
                            stats["written_lines"] += 1
                        stats["incomplete_groups_kept"] += 1
                    else:
                        stats["incomplete_groups_dropped"] += 1
                else:
                    stats["incomplete_groups_dropped"] += 1

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


if __name__ == "__main__":
    main()
