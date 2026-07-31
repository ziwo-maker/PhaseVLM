"""
LLM Caption Backfill using DeepSeek API (deepseek-v4-flash).

Usage:
    python utils/backfill_llm_captions_api.py \
      --input records/sim_expert_validation_search/anon_3_4_jinan_real_2000/best/align_train.cleaned.rebalanced.jsonl \
      --output records/sim_expert_validation_search/anon_3_4_jinan_real_2000/best/align_train.cleaned.rebalanced.llm_caption_ds.jsonl \
      --api_key sk-xxx \
      --api_base https://api.deepseek.com \
      --model deepseek-v4-flash \
      --batch_size 8 \
      --language en \
      --max_tokens 64 \
      --temperature 0.8 \
      --top_p 0.95 \
      --max_caption_retries 3 \
      --diversity_hamming_threshold 4
"""

import argparse
import hashlib
import json
import os
import random
import re
import time
from itertools import chain
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests

DEFAULT_MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
DEFAULT_CAPTION_LANGUAGE = "en"
DEFAULT_CAPTION_MAX_NEW_TOKENS = 2048
DEFAULT_CAPTION_TEMPERATURE = 0.8
DEFAULT_CAPTION_TOP_P = 0.95
DEFAULT_MAX_CAPTION_RETRIES = 3
DEFAULT_DIVERSITY_HAMMING_THRESHOLD = 4
DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_THINKING_MODE = "disabled"

TRAFFIC_SCENE_TERMS = {
    "approach",
    "approaches",
    "backed",
    "busy",
    "clear",
    "clearing",
    "congested",
    "congestion",
    "empty",
    "flow",
    "free-flowing",
    "intersection",
    "lane",
    "lanes",
    "light",
    "moderate",
    "movement",
    "movements",
    "queue",
    "queues",
    "road",
    "roads",
    "traffic",
    "vehicle",
    "vehicles",
}

DATA_DUMP_PATTERNS = [
    r"\b\d+\s+vehicles?\b",
    r"\baverage\s*/?\s*bin\b",
    r"\bper\s+bin\b",
    r"\bpeak\s+bin\b",
    r"\bstart\s+\d+\b",
    r"\bend\s+\d+\b",
    r"\btotal\s+\d+\b",
    r"\b(nt|nl|st|sl|et|el|wt|wl)\s*=\s*\d+\b",
    r"\bdata\b",
    r"\btable\b",
    r"\bchart\b",
    r"\bsummary\b",
    r"\btimestamp\b",
]

MOVEMENT_LABELS = {
    "NT": "north through",
    "NL": "north left",
    "ST": "south through",
    "SL": "south left",
    "ET": "east through",
    "EL": "east left",
    "WT": "west through",
    "WL": "west left",
}


def _safe_int(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _movement_label(key: str) -> str:
    normalized = str(key or "").strip().upper()
    return f"{MOVEMENT_LABELS.get(normalized, normalized.lower())} ({normalized})"


def _extract_history_metrics(history: List[Dict], movement_keys: List[str]) -> Dict[str, Any]:
    keys = list(movement_keys or [])
    if not keys and history:
        keys = list((history[-1].get("movement_counts") or {}).keys())

    series_by_key: Dict[str, List[int]] = {key: [] for key in keys}
    per_bin_totals: List[int] = []
    timestamps: List[str] = []

    for entry in history:
        counts = entry.get("movement_counts") or {}
        timestamps.append(str(entry.get("date") or ""))
        total = 0
        for key in keys:
            value = _safe_int(counts.get(key, 0))
            series_by_key[key].append(value)
            total += value
        per_bin_totals.append(total)

    total_sum = sum(per_bin_totals)
    avg_total = total_sum / max(len(per_bin_totals), 1)
    first_total = per_bin_totals[0] if per_bin_totals else 0
    last_total = per_bin_totals[-1] if per_bin_totals else 0
    peak_idx = max(range(len(per_bin_totals)), key=lambda idx: per_bin_totals[idx]) if per_bin_totals else 0
    peak_total = per_bin_totals[peak_idx] if per_bin_totals else 0
    peak_time = timestamps[peak_idx] if timestamps else ""
    trend = _clip_trend_label(first_total, last_total)

    movement_stats: List[Dict[str, Any]] = []
    for key in keys:
        series = series_by_key.get(key, [])
        total = int(sum(series))
        start = series[0] if series else 0
        end = series[-1] if series else 0
        peak_value = max(series) if series else 0
        peak_step = max(range(len(series)), key=lambda idx: series[idx]) if series else 0
        active_bins = sum(1 for value in series if value > 0)
        movement_stats.append(
            {
                "key": key,
                "label": _movement_label(key),
                "total": total,
                "start": start,
                "end": end,
                "delta": end - start,
                "trend": _clip_trend_label(start, end),
                "peak": peak_value,
                "peak_time": timestamps[peak_step] if timestamps else "",
                "active_bins": active_bins,
                "series": series,
            }
        )

    movement_stats.sort(
        key=lambda item: (item["total"], item["end"], item["active_bins"], item["peak"]),
        reverse=True,
    )

    through_sum = sum(item["total"] for item in movement_stats if str(item["key"]).endswith("T"))
    left_sum = sum(item["total"] for item in movement_stats if str(item["key"]).endswith("L"))
    ew_sum = sum(item["total"] for item in movement_stats if str(item["key"]).startswith(("E", "W")))
    ns_sum = sum(item["total"] for item in movement_stats if str(item["key"]).startswith(("N", "S")))
    total_dir = max(ew_sum + ns_sum, 1)
    total_turn = max(through_sum + left_sum, 1)
    ew_share = ew_sum / total_dir
    ns_share = ns_sum / total_dir
    through_share = through_sum / total_turn
    left_share = left_sum / total_turn

    if abs(ew_share - ns_share) < 0.1:
        dominance = "balanced"
        dominant_axis = "balanced"
    else:
        dominance = "east-west dominant" if ew_share > ns_share else "north-south dominant"
        dominant_axis = "east-west" if ew_share > ns_share else "north-south"

    if abs(through_share - left_share) < 0.12:
        turn_bias = "balanced"
    else:
        turn_bias = "through-dominant" if through_share > left_share else "left-turn-dominant"

    latest_nonzero = [
        item
        for item in sorted(movement_stats, key=lambda row: (row["end"], row["total"]), reverse=True)
        if item["end"] > 0
    ]
    quiet_movements = [item["key"] for item in movement_stats if item["total"] <= 0]
    growing_movements = [item for item in movement_stats if item["delta"] > 0]
    growing_movements.sort(key=lambda item: (item["delta"], item["end"], item["total"]), reverse=True)

    return {
        "timestamps": timestamps,
        "per_bin_totals": per_bin_totals,
        "total_sum": total_sum,
        "avg_total": avg_total,
        "first_total": first_total,
        "last_total": last_total,
        "trend": trend,
        "peak_total": peak_total,
        "peak_time": peak_time,
        "movement_stats": movement_stats,
        "latest_nonzero": latest_nonzero,
        "growing_movements": growing_movements,
        "quiet_movements": quiet_movements,
        "dominance": dominance,
        "dominant_axis": dominant_axis,
        "turn_bias": turn_bias,
        "ew_share": ew_share,
        "ns_share": ns_share,
        "through_share": through_share,
        "left_share": left_share,
        "through_sum": through_sum,
        "left_sum": left_sum,
    }


def _format_movement_stat_lines(items: List[Dict[str, Any]], limit: int = 4) -> str:
    selected = [item for item in items if int(item.get("total", 0)) > 0][:limit]
    if not selected:
        return "none"
    lines = []
    for item in selected:
        lines.append(
            f"{item['label']}: total {int(item['total'])}, start {int(item['start'])}, "
            f"end {int(item['end'])}, trend {item['trend']}, peak {int(item['peak'])}, "
            f"active {int(item['active_bins'])}/{max(len(item.get('series') or []), 1)}"
        )
    return "; ".join(lines)


def _format_latest_load(items: List[Dict[str, Any]], limit: int = 4) -> str:
    selected = items[:limit]
    if not selected:
        return "none"
    return ", ".join(f"{item['key']}={int(item['end'])}" for item in selected)


def _build_structured_caption(metrics: Dict[str, Any]) -> str:
    top_movements = [item for item in metrics.get("movement_stats", []) if int(item.get("total", 0)) > 0][:2]
    latest_nonzero = list(metrics.get("latest_nonzero", []))
    dominant_axis = str(metrics.get("dominant_axis", "balanced") or "balanced")
    if dominant_axis == "balanced":
        sentence_one = (
            f"Traffic is balanced across directions, with {metrics.get('turn_bias', 'balanced')} demand and "
            f"{metrics.get('trend', 'steady')} total volume across the window."
        )
    else:
        sentence_one = (
            f"{dominant_axis.capitalize()} traffic dominates, with {metrics.get('turn_bias', 'balanced')} demand and "
            f"{metrics.get('trend', 'steady')} total volume across the window."
        )

    if len(top_movements) >= 2:
        first, second = top_movements[:2]
        detail_text = (
            f"Load is concentrated on {_movement_label(first['key'])} with {int(first['total'])} vehicles in total "
            f"and {int(first['end'])} in the latest frame, plus {_movement_label(second['key'])} with "
            f"{int(second['total'])} in total and {int(second['end'])} at the end"
        )
    elif len(top_movements) == 1:
        first = top_movements[0]
        quiet = list(metrics.get("quiet_movements", []))[:2]
        quiet_text = f", while {', '.join(quiet)} stay empty" if quiet else ""
        detail_text = (
            f"Load is concentrated on {_movement_label(first['key'])} with {int(first['total'])} vehicles in total "
            f"and {int(first['end'])} in the latest frame{quiet_text}"
        )
    elif latest_nonzero:
        first = latest_nonzero[0]
        detail_text = f"The latest load centers on {_movement_label(first['key'])} with {int(first['end'])} vehicles"
    else:
        detail_text = "Traffic stays very light with no persistent hotspot"

    latest_phrase = _format_latest_load(latest_nonzero, limit=3)
    sentence_two = (
        f"{detail_text}; the latest load is {latest_phrase} and the peak reaches "
        f"{int(metrics.get('peak_total', 0))} vehicles."
    )
    return " ".join(part.strip() for part in (sentence_one, sentence_two) if part.strip())


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


def _simhash(tokens: List[str]) -> int:
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


def _caption_min_hamming_distance(caption: str, seen_hashes: List[int]) -> Optional[int]:
    if not seen_hashes:
        return None
    normalized = _normalize_text(caption)
    if not normalized:
        return None
    tokens = _tokenize(normalized)
    if not tokens:
        return None
    h = _simhash(tokens)
    return min(_hamming(h, ref) for ref in seen_hashes)


def _caption_rejection_reason(text: str) -> Optional[str]:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return "empty"
    lowered = normalized.lower()
    words = normalized.split()
    if len(words) < 12:
        return "too_short"
    normalized_tokens = set(_tokenize(_normalize_text(normalized)))
    if not (normalized_tokens & TRAFFIC_SCENE_TERMS):
        return "missing_traffic_scene_terms"
    for pattern in DATA_DUMP_PATTERNS:
        if re.search(pattern, lowered):
            return "data_dump"
    return None


def _caption_needs_fallback(text: str) -> bool:
    return _caption_rejection_reason(text) is not None


def _caption_is_too_similar(
    caption: str,
    seen_hashes: List[int],
    threshold: int = DEFAULT_DIVERSITY_HAMMING_THRESHOLD,
) -> bool:
    if not seen_hashes:
        return False
    min_dist = _caption_min_hamming_distance(caption, seen_hashes)
    if min_dist is None:
        return False
    return min_dist <= threshold


def _accept_caption_hash(caption: str, seen_hashes: List[int]) -> None:
    normalized = _normalize_text(caption)
    if not normalized:
        return
    tokens = _tokenize(normalized)
    if not tokens:
        return
    seen_hashes.append(_simhash(tokens))


def _build_structured_caption_safe(metrics: Dict[str, Any]) -> str:
    return _build_structured_caption(metrics)


def _clip_trend_label(first_total: float, last_total: float) -> str:
    delta = last_total - first_total
    threshold = max(2.0, 0.2 * max(first_total, 1.0))
    if delta >= threshold:
        return "rising"
    if delta <= -threshold:
        return "declining"
    return "steady"


def _summarize_history_for_caption(
    history: List[Dict],
    movement_keys: List[str],
    interval_s: int,
    window_s: int,
) -> str:
    if not history:
        return "No observations."
    keys = list(movement_keys or [])
    if not keys:
        keys = list((history[-1].get("movement_counts") or {}).keys())
    if not keys:
        return "No movements."

    metrics = _extract_history_metrics(history, keys)
    start_time = history[0].get("date")
    end_time = history[-1].get("date")
    lines = [
        f"Window: {start_time} to {end_time} (interval {int(interval_s)}s, span {int(window_s)}s).",
        f"Overall load: total {int(metrics['total_sum'])}, average/bin {metrics['avg_total']:.1f}, "
        f"start {int(metrics['first_total'])}, end {int(metrics['last_total'])}, trend {metrics['trend']}.",
        f"Peak bin: {metrics['peak_time']} with {int(metrics['peak_total'])} vehicles.",
        f"Direction split: NS {metrics['ns_share']:.0%}, EW {metrics['ew_share']:.0%} ({metrics['dominance']}).",
        f"Turn split: through {metrics['through_share']:.0%}, left {metrics['left_share']:.0%} "
        f"({metrics['turn_bias']}).",
        f"Top cumulative movements: {_format_movement_stat_lines(metrics['movement_stats'], limit=4)}.",
        f"Fastest-growing movements: {_format_movement_stat_lines(metrics['growing_movements'], limit=2)}.",
        f"Latest non-zero movements: {_format_latest_load(metrics['latest_nonzero'], limit=4)}.",
        f"Zero-demand movements over the full window: {', '.join(metrics['quiet_movements'][:4]) or 'none'}.",
    ]
    return "\n".join(lines)


def _clip_caption_instruction_series(language: str) -> str:
    lang = str(language or DEFAULT_CAPTION_LANGUAGE).strip().lower()
    target_language = "English" if lang in ("en", "eng", "english") else lang

    templates = [
        (
            "You are looking at a traffic camera feed at one intersection. "
            "Describe what you SEE in the latest frame, based on the traffic summary below. "
            f"Write in {target_language}. Output exactly two sentences on one line, with no quotes. "
            "Sentence 1: describe the visual scene — which approaches look busy or clear, "
            "whether queues appear to be growing, shrinking, or stable. Use concrete visual "
            "language like 'queue forming', 'clear road', 'steady flow', 'bumper-to-bumper'. "
            "Sentence 2: describe the most noticeable details — where vehicles are concentrated "
            "right now, which lanes appear congested or empty. Mention specific movements "
            "using their abbreviations in parentheses ONLY when they help identify the lane. "
            "Do NOT recite raw vehicle counts, timestamps, or statistics. "
            "Do NOT mention data, tables, summaries, or charts."
        ),
        (
            "A traffic camera has been recording one intersection for the past 30 seconds. "
            "Based on the summary below, write a natural-language description of the scene. "
            f"Write in {target_language}. Output exactly two sentences on one line, with no quotes. "
            "Sentence 1: describe how traffic has changed over the observation window — "
            "is congestion building up, clearing out, or staying the same? Use vivid "
            "descriptors like 'queues are growing', 'traffic is easing', 'intersection is clearing'. "
            "Sentence 2: describe the current state of the busiest lanes and the quietest lanes. "
            "Mention movement abbreviations (e.g., ST, NT) naturally in context. "
            "Do NOT list numbers or use phrases like 'X vehicles counted'. "
            "Do NOT mention data, tables, summaries, or timestamps."
        ),
        (
            "Describe the traffic scene at this intersection as if narrating a traffic report. "
            f"Write in {target_language}. Output exactly two sentences on one line, with no quotes. "
            "Sentence 1: identify which direction (north-south or east-west) dominates the scene "
            "and whether through traffic or left turns are more prominent. Use visual terms like "
            "'dominates', 'flows freely', 'is backed up', 'is nearly empty'. "
            "Sentence 2: zoom in on the most congested movement direction right now and contrast "
            "it with the emptiest. Mention specific movement abbreviations accurately. "
            "Do NOT recite raw vehicle counts or tabular data. "
            "Do NOT mention data, tables, summaries, or timestamps."
        ),
        (
            "Summarize the latest traffic camera image at this intersection. "
            f"Write in {target_language}. Output exactly two sentences on one line, with no quotes. "
            "Sentence 1: in plain visual language, describe the overall traffic picture — "
            "which approaches are busy, which are clear, and whether the situation is "
            "improving or worsening. Sentence 2: point out the most striking visual detail, "
            "such as a long queue in a specific lane or an unusually empty approach. "
            "Include relevant movement abbreviations. Avoid numerical counts. "
            "Do NOT mention data, tables, summaries, or timestamps."
        ),
        (
            "Based on the traffic summary, paint a picture of what the latest camera frame shows. "
            f"Write in {target_language}. Output exactly two sentences on one line, with no quotes. "
            "Sentence 1: describe the congestion pattern — where vehicles are clustered, "
            "which movements have the heaviest load, and whether the pattern is typical or unusual. "
            "Use descriptors like 'heavy', 'light', 'moderate', 'backed up', 'free-flowing'. "
            "Sentence 2: highlight the movement with the most vehicles waiting right now, "
            "and note any movement that stands out as unexpectedly empty. "
            "Include movement abbreviations for precision. "
            "Do NOT recite raw vehicle counts. "
            "Do NOT mention data, tables, summaries, or timestamps."
        ),
    ]
    return random.choice(templates)


def _clean_clip_caption(text: str) -> str:
    if text is None:
        return ""
    cleaned = str(text).strip()
    if not cleaned:
        return ""
    cleaned = cleaned.splitlines()[0].strip()
    for prefix in ("caption:", "description:", "answer:", "output:", "response:"):
        if cleaned.lower().startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break
    return " ".join(cleaned.split())


def _iter_grouped_records(path: str) -> Iterator[List[Dict]]:
    current_group: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            current_group.append(record)
            if record.get("clip_caption"):
                yield current_group
                current_group = []
    if current_group:
        yield current_group


def _iter_group_batches(path: str, batch_size: int) -> Iterator[List[List[Dict]]]:
    batch: List[List[Dict]] = []
    batch_size = max(1, int(batch_size))
    for group in _iter_grouped_records(path):
        batch.append(group)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _group_prompt(group: List[Dict], language: str = DEFAULT_CAPTION_LANGUAGE) -> str:
    last_record = group[-1] if group else {}
    meta = last_record.get("meta") or {}
    if not isinstance(meta, dict):
        meta = {}
    ts_image_meta = meta.get("ts_image") or {}
    if not isinstance(ts_image_meta, dict):
        ts_image_meta = {}

    movement_keys = meta.get("movement_keys")
    if not isinstance(movement_keys, list) or not movement_keys:
        movement_keys = list((last_record.get("movement_counts") or {}).keys()) or list(DEFAULT_MOVEMENT_KEYS)
    interval_s = int(last_record.get("interval_s") or ts_image_meta.get("interval_s") or 2)
    window_s = int(ts_image_meta.get("window_s") or max(interval_s * max(len(group) - 1, 1), interval_s))
    history = [
        {
            "date": item.get("date"),
            "movement_counts": item.get("movement_counts") or {},
        }
        for item in group
    ]
    summary = _summarize_history_for_caption(history, movement_keys, interval_s, window_s)
    instruction = _clip_caption_instruction_series(language=language)
    return f"{instruction}\n\nTraffic summary:\n{summary}\n\nCaption:"


def _call_deepseek_api(
    prompt: str,
    api_key: str,
    api_base: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: str = DEFAULT_THINKING_MODE,
    max_retries: int = 3,
) -> str:
    """Call DeepSeek API to generate a caption from the prompt."""
    url = f"{api_base.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if str(thinking or "").strip():
        payload["thinking"] = {"type": str(thinking).strip()}

    last_error = None
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            if response.status_code == 200:
                data = response.json()
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
                if not content and choice.get("finish_reason") == "length":
                    last_error = "empty content because generation reached max_tokens"
                    time.sleep(1)
                    continue
                return _clean_clip_caption(content)
            elif response.status_code == 429:
                # Rate limited - wait and retry
                wait_time = min(2 ** attempt, 30)
                time.sleep(wait_time)
                last_error = f"HTTP 429 (rate limited, attempt {attempt + 1})"
            else:
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"
                time.sleep(1)
        except requests.exceptions.Timeout:
            last_error = f"Timeout (attempt {attempt + 1})"
            time.sleep(2)
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            time.sleep(2)

    raise RuntimeError(f"DeepSeek API call failed after {max_retries} retries: {last_error}")


def _generate_captions_api(
    prompts: List[str],
    api_key: str,
    api_base: str,
    model: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    thinking: str = DEFAULT_THINKING_MODE,
    max_workers: int = 8,
) -> List[str]:
    """Generate captions by calling DeepSeek API concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    total = len(prompts)
    if total == 0:
        return []

    # Process prompts concurrently
    results: List[Optional[str]] = [None] * total

    def _process_one(idx: int, prompt: str) -> tuple:
        try:
            caption = _call_deepseek_api(
                prompt=prompt,
                api_key=api_key,
                api_base=api_base,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                thinking=thinking,
            )
            return idx, caption
        except Exception as e:
            print(f"  [ERROR] prompt {idx + 1}/{total}: {e}", flush=True)
            return idx, ""  # Empty -> will trigger retry or fallback

    workers = min(max_workers, total)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process_one, i, p): i for i, p in enumerate(prompts)}
        done_count = 0
        for future in as_completed(futures):
            idx, caption = future.result()
            results[idx] = caption
            done_count += 1
            if done_count % 10 == 0 or done_count == total:
                print(f"  progress: {done_count}/{total} prompts", flush=True)

    return [r if r is not None else "" for r in results]


def _infer_model_name(api_base: str, model: str) -> str:
    return f"deepseek_api_{model}"


def _resolve_api_key(value: str = "") -> str:
    api_key = str(value or "").strip()
    if api_key:
        return api_key
    for env_name in ("DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        api_key = str(os.environ.get(env_name) or "").strip()
        if api_key:
            return api_key
    raise SystemExit("DeepSeek API key is required. Pass --api_key or set DEEPSEEK_API_KEY.")


def _rewrite_groups(
    groups: List[List[Dict]],
    captions: List[str],
    llm_model_name: str,
    caption_modes: List[str],
) -> List[Dict]:
    rewritten: List[Dict] = []
    for group, caption, caption_mode in zip(groups, captions, caption_modes):
        updated_group = [dict(item) for item in group]
        if updated_group:
            last_record = updated_group[-1]
            previous_caption = str(last_record.get("clip_caption") or "").strip()
            if previous_caption and previous_caption != caption and "original_clip_caption" not in last_record:
                last_record["original_clip_caption"] = previous_caption
            last_record["clip_caption"] = caption or previous_caption
            meta = last_record.get("meta") or {}
            if not isinstance(meta, dict):
                meta = {}
            meta = dict(meta)
            meta["caption_source"] = "llm_backfill"
            meta["caption_model"] = str(llm_model_name or "")
            meta["caption_style"] = "clip_caption_v3_specific_backfill"
            meta["caption_backfill_mode"] = str(caption_mode or "llm")
            last_record["meta"] = meta
        rewritten.extend(updated_group)
    return rewritten


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="DeepSeek API key. If omitted, DEEPSEEK_API_KEY or ANTHROPIC_API_KEY is used.",
    )
    parser.add_argument("--api_base", type=str, default=DEFAULT_API_BASE, help="DeepSeek API base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name (e.g. deepseek-v4-flash)")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of groups to batch together")
    parser.add_argument("--workers", type=int, default=8, help="Max concurrent API workers")
    parser.add_argument("--language", type=str, default=DEFAULT_CAPTION_LANGUAGE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_CAPTION_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CAPTION_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_CAPTION_TOP_P)
    parser.add_argument(
        "--thinking",
        type=str,
        default=DEFAULT_THINKING_MODE,
        choices=("enabled", "disabled"),
        help="DeepSeek thinking mode. Caption backfill defaults to disabled so max_tokens produce final content.",
    )
    parser.add_argument(
        "--max_caption_retries",
        type=int,
        default=DEFAULT_MAX_CAPTION_RETRIES,
        help="max regeneration attempts per caption for quality/diversity (0 = no retry)",
    )
    parser.add_argument(
        "--diversity_hamming_threshold",
        type=int,
        default=DEFAULT_DIVERSITY_HAMMING_THRESHOLD,
        help="SimHash Hamming distance below which captions are considered too similar",
    )
    return parser.parse_args()


def backfill_captions_api(
    input_path: str,
    output_path: str,
    api_key: str,
    api_base: str = DEFAULT_API_BASE,
    model: str = DEFAULT_MODEL,
    batch_size: int = 8,
    workers: int = 8,
    language: str = DEFAULT_CAPTION_LANGUAGE,
    max_tokens: int = DEFAULT_CAPTION_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_CAPTION_TEMPERATURE,
    top_p: float = DEFAULT_CAPTION_TOP_P,
    thinking: str = DEFAULT_THINKING_MODE,
    max_caption_retries: int = DEFAULT_MAX_CAPTION_RETRIES,
    diversity_hamming_threshold: int = DEFAULT_DIVERSITY_HAMMING_THRESHOLD,
) -> dict:
    max_retries = max(0, int(max_caption_retries))
    batch_size = max(1, int(batch_size))
    workers = max(1, int(workers))
    thinking = str(thinking or DEFAULT_THINKING_MODE).strip()
    diversity_threshold = int(diversity_hamming_threshold)
    num_groups = 0
    num_records = 0
    fallback_count = 0
    retry_count = 0
    diversity_retry_count = 0
    first_pass_accepted_count = 0
    retry_accepted_count = 0
    llm_model_name = _infer_model_name(api_base, model)
    seen_hashes: List[int] = []
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    group_batches = _iter_group_batches(input_path, batch_size=batch_size)
    first_batch = next(group_batches, None)
    if first_batch is None:
        raise SystemExit(f"No caption groups found in {input_path}.")

    with open(output_path, "w", encoding="utf-8") as handle:
        for groups in chain([first_batch], group_batches):
            prompts = [_group_prompt(group, language=language) for group in groups]
            raw_captions = _generate_captions_api(
                prompts,
                api_key=api_key,
                api_base=api_base,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                thinking=thinking,
                max_workers=workers,
            )
            captions: List[str] = [""] * len(groups)
            caption_modes: List[str] = ["llm"] * len(groups)
            retry_indices: List[int] = []

            for idx, (group, llm_caption) in enumerate(zip(groups, raw_captions)):
                needs_retry = False
                rejection_reason = _caption_rejection_reason(llm_caption)
                if rejection_reason is not None:
                    needs_retry = True
                    retry_count += 1
                elif _caption_is_too_similar(llm_caption, seen_hashes, threshold=diversity_threshold):
                    needs_retry = True
                    diversity_retry_count += 1

                if needs_retry:
                    retry_indices.append(idx)
                else:
                    captions[idx] = llm_caption
                    caption_modes[idx] = "llm"
                    first_pass_accepted_count += 1
                    _accept_caption_hash(llm_caption, seen_hashes)

            retry_round = 0
            while retry_indices and retry_round < max_retries:
                retry_round += 1
                next_retry_indices: List[int] = []
                retry_temperature = min(1.0, float(temperature) + 0.1 * retry_round)

                for local_idx, idx in enumerate(retry_indices):
                    group = groups[idx]
                    retry_prompt = _group_prompt(group, language=language)
                    try:
                        llm_caption = _call_deepseek_api(
                            prompt=retry_prompt,
                            api_key=api_key,
                            api_base=api_base,
                            model=model,
                            max_tokens=max_tokens,
                            temperature=retry_temperature,
                            top_p=top_p,
                            thinking=thinking,
                        )
                    except Exception as e:
                        print(f"  [RETRY ERROR] group retry round {retry_round}: {e}", flush=True)
                        llm_caption = ""

                    needs_retry = False
                    rejection_reason = _caption_rejection_reason(llm_caption)
                    if rejection_reason is not None:
                        needs_retry = True
                        retry_count += 1
                    elif _caption_is_too_similar(llm_caption, seen_hashes, threshold=diversity_threshold):
                        needs_retry = True
                        diversity_retry_count += 1

                    if needs_retry:
                        next_retry_indices.append(idx)
                    else:
                        captions[idx] = llm_caption
                        caption_modes[idx] = f"llm_retry_{retry_round}"
                        retry_accepted_count += 1
                        _accept_caption_hash(llm_caption, seen_hashes)

                retry_indices = next_retry_indices

            for idx in retry_indices:
                group = groups[idx]
                last_record = group[-1] if group else {}
                meta = last_record.get("meta") or {}
                if not isinstance(meta, dict):
                    meta = {}
                movement_keys = meta.get("movement_keys")
                if not isinstance(movement_keys, list) or not movement_keys:
                    movement_keys = list(
                        (last_record.get("movement_counts") or {}).keys()
                    ) or list(DEFAULT_MOVEMENT_KEYS)
                history = [
                    {
                        "date": item.get("date"),
                        "movement_counts": item.get("movement_counts") or {},
                    }
                    for item in group
                ]
                metrics = _extract_history_metrics(history, movement_keys)
                captions[idx] = _build_structured_caption_safe(metrics)
                caption_modes[idx] = "structured_fallback_exhausted"
                fallback_count += 1

            if len(captions) != len(groups):
                raise SystemExit(
                    f"Caption count mismatch: generated {len(captions)} captions for {len(groups)} groups."
                )

            rewritten_records = _rewrite_groups(
                groups, captions, llm_model_name=llm_model_name, caption_modes=caption_modes
            )
            for record in rewritten_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            num_groups += len(groups)
            num_records += len(rewritten_records)
            print(f"  processed {num_groups} groups, {num_records} records so far", flush=True)

    return {
        "input_path": input_path,
        "output_path": output_path,
        "model": model,
        "llm_model_name": llm_model_name,
        "num_groups": int(num_groups),
        "num_records": int(num_records),
        "language": language,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "thinking": thinking,
        "max_caption_retries": int(max_retries),
        "diversity_hamming_threshold": int(diversity_threshold),
        "num_llm_first_pass": int(first_pass_accepted_count),
        "num_llm_retry_captions": int(retry_accepted_count),
        "num_quality_retries": int(retry_count),
        "num_diversity_retries": int(diversity_retry_count),
        "num_structured_fallback_captions": int(fallback_count),
        "num_llm_captions": int(num_groups - fallback_count),
    }


def main():
    args = parse_args()
    print(f"DeepSeek API Backfill: model={args.model}, input={args.input}", flush=True)
    print(f"  temperature={args.temperature}, top_p={args.top_p}, max_tokens={args.max_tokens}", flush=True)
    print(
        f"  thinking={args.thinking}, max_retries={args.max_caption_retries}, "
        f"diversity_threshold={args.diversity_hamming_threshold}",
        flush=True,
    )

    api_key = _resolve_api_key(args.api_key)
    result = backfill_captions_api(
        input_path=args.input,
        output_path=args.output,
        api_key=api_key,
        api_base=args.api_base,
        model=args.model,
        batch_size=args.batch_size,
        workers=args.workers,
        language=args.language,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        thinking=args.thinking,
        max_caption_retries=args.max_caption_retries,
        diversity_hamming_threshold=args.diversity_hamming_threshold,
    )
    print(f"\nDone! Wrote {result['num_groups']} caption groups to {args.output}")
    print(f"  first-pass LLM accepted: {result['num_llm_first_pass']}")
    print(f"  retry LLM accepted: {result['num_llm_retry_captions']}")
    print(f"  quality retries: {result['num_quality_retries']}")
    print(f"  diversity retries: {result['num_diversity_retries']}")
    print(f"  exhausted fallbacks: {result['num_structured_fallback_captions']}")


if __name__ == "__main__":
    main()
