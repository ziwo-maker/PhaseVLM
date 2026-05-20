#!/usr/bin/env python3
from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

try:
    from utils.backfill_llm_captions_api import (
        DEFAULT_API_BASE,
        DEFAULT_CAPTION_LANGUAGE,
        DEFAULT_CAPTION_TEMPERATURE,
        DEFAULT_CAPTION_TOP_P,
        DEFAULT_DIVERSITY_HAMMING_THRESHOLD,
        DEFAULT_MAX_CAPTION_RETRIES,
        DEFAULT_MODEL,
        DEFAULT_THINKING_MODE,
        DATA_DUMP_PATTERNS,
        DEFAULT_MOVEMENT_KEYS,
        _accept_caption_hash,
        _call_deepseek_api,
        _caption_is_too_similar,
        _clean_clip_caption,
        _extract_history_metrics,
        _movement_label,
        _summarize_history_for_caption,
    )
except (ImportError, ModuleNotFoundError):
    from backfill_llm_captions_api import (  # type: ignore
        DEFAULT_API_BASE,
        DEFAULT_CAPTION_LANGUAGE,
        DEFAULT_CAPTION_TEMPERATURE,
        DEFAULT_CAPTION_TOP_P,
        DEFAULT_DIVERSITY_HAMMING_THRESHOLD,
        DEFAULT_MAX_CAPTION_RETRIES,
        DEFAULT_MODEL,
        DEFAULT_THINKING_MODE,
        DATA_DUMP_PATTERNS,
        DEFAULT_MOVEMENT_KEYS,
        _accept_caption_hash,
        _call_deepseek_api,
        _caption_is_too_similar,
        _clean_clip_caption,
        _extract_history_metrics,
        _movement_label,
        _summarize_history_for_caption,
    )


STRICT_DATA_DUMP_PATTERNS = [
    *DATA_DUMP_PATTERNS,
    r"\b\d+\b",
    r"\bvehicles?\s+(?:in\s+total|counted|at\s+the\s+end)\b",
    r"\b(?:nt|nl|st|sl|et|el|wt|wl)\s*=\s*\d+\b",
]


def _resolve_api_key(value: str, read_stdin: bool) -> str:
    if value.strip():
        return value.strip()
    env_value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_value:
        return env_value
    if read_stdin:
        if sys.stdin.isatty():
            stdin_value = getpass.getpass("").strip()
        else:
            stdin_value = sys.stdin.readline().strip()
        if stdin_value:
            return stdin_value
    raise SystemExit("DeepSeek API key is required. Use --api_key, DEEPSEEK_API_KEY, or --api_key_stdin.")


def _iter_grouped_records(path: Path) -> list[list[dict[str, Any]]]:
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            current.append(record)
            if str(record.get("clip_caption") or "").strip():
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    return groups


def _sentence_end_count(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", str(text or "").strip()))


def _caption_issue(caption: str) -> str | None:
    text = " ".join(str(caption or "").strip().split())
    if not text:
        return "empty"
    if len(text.split()) < 12:
        return "too_short"
    if not re.search(r"[.!?]$", text):
        return "missing_terminal_punctuation"
    if _sentence_end_count(text) != 2:
        return "not_two_sentences"
    if text.count("(") != text.count(")"):
        return "unbalanced_parentheses"
    lowered = text.lower()
    for pattern in STRICT_DATA_DUMP_PATTERNS:
        if re.search(pattern, lowered):
            return "data_dump_or_numeric"
    if re.search(r"\b(?:and|or|the|with|while|as|in|on|for|to|from)\s*[.!?]?$", text, re.I):
        return "suspicious_trailing_word"
    return None


def _movement_keys(group: list[dict[str, Any]]) -> list[str]:
    last = group[-1] if group else {}
    meta = last.get("meta") or {}
    keys = meta.get("movement_keys") if isinstance(meta, dict) else None
    if isinstance(keys, list) and keys:
        return [str(key) for key in keys]
    counts = last.get("movement_counts") or {}
    if isinstance(counts, dict) and counts:
        return [str(key) for key in counts.keys()]
    return list(DEFAULT_MOVEMENT_KEYS)


def _history(group: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "date": item.get("date"),
            "movement_counts": item.get("movement_counts") or {},
        }
        for item in group
    ]


def _repair_prompt(group: list[dict[str, Any]], language: str) -> str:
    last = group[-1] if group else {}
    meta = last.get("meta") or {}
    ts_image_meta = meta.get("ts_image") if isinstance(meta, dict) else {}
    if not isinstance(ts_image_meta, dict):
        ts_image_meta = {}
    interval_s = int(last.get("interval_s") or ts_image_meta.get("interval_s") or 2)
    window_s = int(ts_image_meta.get("window_s") or max(interval_s * max(len(group) - 1, 1), interval_s))
    keys = _movement_keys(group)
    summary = _summarize_history_for_caption(_history(group), keys, interval_s, window_s)
    target_language = "English" if str(language).lower() in ("en", "eng", "english") else language
    return (
        "You are writing a concise visual caption for one traffic-camera clip.\n"
        f"Write in {target_language}. Return exactly two short sentences on one line, no quotes.\n"
        "Total length must be 26 to 42 words. End both sentences with periods.\n"
        "Describe visible traffic, queues, busy approaches, clear approaches, and current concentration.\n"
        "You may mention movement abbreviations like ST or WL, but do not use parentheses unless balanced.\n"
        "Do not use digits, numeric counts, timestamps, or the words data, table, chart, summary, total, peak, average.\n"
        "Do not say how many vehicles were counted.\n\n"
        f"Traffic context:\n{summary}\n\nCaption:"
    )


def _visual_fallback(group: list[dict[str, Any]]) -> str:
    keys = _movement_keys(group)
    metrics = _extract_history_metrics(_history(group), keys)
    dominant_axis = str(metrics.get("dominant_axis") or "balanced")
    trend = str(metrics.get("trend") or "steady")
    turn_bias = str(metrics.get("turn_bias") or "balanced").replace("-", " ")
    top = [item for item in metrics.get("movement_stats", []) if int(item.get("total", 0)) > 0]
    latest = [item for item in metrics.get("latest_nonzero", []) if int(item.get("end", 0)) > 0]
    quiet = list(metrics.get("quiet_movements") or [])

    if dominant_axis == "balanced":
        first = f"Traffic looks balanced across the intersection, with {turn_bias} demand and a {trend} flow pattern."
    else:
        first = f"{dominant_axis.capitalize()} traffic looks busiest, with {turn_bias} demand and a {trend} flow pattern."

    busy_item = latest[0] if latest else (top[0] if top else None)
    quiet_label = ""
    if quiet:
        quiet_label = f", while {_movement_label(str(quiet[0]))} looks mostly clear"
    if busy_item:
        second = f"The {_movement_label(str(busy_item.get('key')))} lane appears most crowded now{quiet_label}."
    else:
        second = "The approaches look mostly clear, with no obvious queue forming in the latest view."
    return f"{first} {second}"


def _generate_repair(
    group: list[dict[str, Any]],
    api_key: str,
    args: argparse.Namespace,
    seen_hashes: list[int],
) -> tuple[str, str]:
    for attempt in range(args.max_caption_retries + 1):
        temperature = min(1.0, float(args.temperature) + 0.1 * attempt)
        try:
            caption = _call_deepseek_api(
                prompt=_repair_prompt(group, args.language),
                api_key=api_key,
                api_base=args.api_base,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=temperature,
                top_p=args.top_p,
                thinking=args.thinking,
                max_retries=3,
            )
        except Exception as exc:
            print(f"  [REPAIR ERROR] attempt={attempt + 1}: {exc}", flush=True)
            caption = ""

        caption = _clean_clip_caption(caption)
        if _caption_issue(caption) is None and not _caption_is_too_similar(
            caption, seen_hashes, threshold=args.diversity_hamming_threshold
        ):
            _accept_caption_hash(caption, seen_hashes)
            return caption, "llm_quality_repair" if attempt == 0 else f"llm_quality_repair_retry_{attempt}"

    caption = _visual_fallback(group)
    _accept_caption_hash(caption, seen_hashes)
    return caption, "visual_fallback_quality_repair"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--api_key_stdin", action="store_true")
    parser.add_argument("--api_base", default=DEFAULT_API_BASE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default=DEFAULT_CAPTION_LANGUAGE)
    parser.add_argument("--max_tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CAPTION_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_CAPTION_TOP_P)
    parser.add_argument("--thinking", default=DEFAULT_THINKING_MODE, choices=("enabled", "disabled"))
    parser.add_argument("--max_caption_retries", type=int, default=DEFAULT_MAX_CAPTION_RETRIES)
    parser.add_argument("--diversity_hamming_threshold", type=int, default=DEFAULT_DIVERSITY_HAMMING_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0, help="Repair at most N bad groups, for smoke tests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = _resolve_api_key(args.api_key, args.api_key_stdin)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    groups = _iter_grouped_records(input_path)
    seen_hashes: list[int] = []
    stats: Counter[str] = Counter()
    repaired = 0

    print(f"Caption quality repair: input={input_path}", flush=True)
    print(
        f"  model={args.model}, max_tokens={args.max_tokens}, temperature={args.temperature}, "
        f"top_p={args.top_p}, max_retries={args.max_caption_retries}, diversity={args.diversity_hamming_threshold}",
        flush=True,
    )

    with output_path.open("w", encoding="utf-8") as handle:
        for group_idx, group in enumerate(groups, 1):
            updated_group = [dict(record) for record in group]
            last = updated_group[-1] if updated_group else {}
            current_caption = str(last.get("clip_caption") or "").strip()
            issue = _caption_issue(current_caption)
            if issue is None:
                _accept_caption_hash(current_caption, seen_hashes)
                stats["kept"] += 1
            elif args.limit and repaired >= args.limit:
                stats[f"left_unrepaired_{issue}"] += 1
            else:
                caption, mode = _generate_repair(updated_group, api_key, args, seen_hashes)
                if current_caption and "pre_repair_clip_caption" not in last:
                    last["pre_repair_clip_caption"] = current_caption
                last["clip_caption"] = caption
                meta = last.get("meta") or {}
                if not isinstance(meta, dict):
                    meta = {}
                meta = dict(meta)
                meta["caption_repair_source"] = "deepseek_api_quality_repair"
                meta["caption_repair_model"] = f"deepseek_api_{args.model}"
                meta["caption_repair_reason"] = issue
                meta["caption_backfill_previous_mode"] = str(meta.get("caption_backfill_mode") or "")
                meta["caption_backfill_mode"] = mode
                last["meta"] = meta
                repaired += 1
                stats[f"repaired_{issue}"] += 1
                stats[f"mode_{mode}"] += 1

            for record in updated_group:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if group_idx % 100 == 0 or group_idx == len(groups):
                print(
                    f"  processed {group_idx}/{len(groups)} groups, repaired {repaired}",
                    flush=True,
                )

    print(f"\nDone! Wrote repaired JSONL to {output_path}", flush=True)
    print(json.dumps(dict(stats), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
