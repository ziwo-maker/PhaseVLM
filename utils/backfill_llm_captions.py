import argparse
import json
import os
import re
from typing import Any, Dict, List

DEFAULT_MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
DEFAULT_CAPTION_LANGUAGE = "en"
DEFAULT_CAPTION_MAX_NEW_TOKENS = 64
DEFAULT_CAPTION_TEMPERATURE = 0.2
DEFAULT_CAPTION_TOP_P = 0.9
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9
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
        key=lambda item: (
            item["total"],
            item["end"],
            item["active_bins"],
            item["peak"],
        ),
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
        item for item in sorted(movement_stats, key=lambda row: (row["end"], row["total"]), reverse=True) if item["end"] > 0
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


def _caption_needs_fallback(text: str) -> bool:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        return True
    lowered = normalized.lower()
    movement_mentions = re.findall(r"\b(nt|nl|st|sl|et|el|wt|wl)\b", lowered)
    if len(normalized.split()) < 18:
        return True
    if len(movement_mentions) < 2:
        return True
    if not any(token in lowered for token in ("rising", "declining", "steady", "increasing", "decreasing")):
        return True
    return False


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
    return (
        "Convert the following structured traffic summary into a specific visual caption of one intersection. "
        f"Write in {target_language}. Output exactly two sentences on one line, with no quotes. "
        "Sentence 1 must state the dominant axis, whether through traffic or left turns dominate, and whether "
        "overall demand is rising, declining, or steady. Sentence 2 must mention at least two explicit "
        "movements using their abbreviations in parentheses, such as south through (ST), and say where the "
        "latest load is concentrated. Avoid vague openings like 'moderate traffic' or 'heavy traffic' unless "
        "they are followed by concrete movement details. Do not mention data, tables, summaries, charts, or timestamps."
    )


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


def _infer_tensor_parallel_size() -> int:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        ids = [item.strip() for item in visible.split(",") if item.strip()]
        if ids:
            return len(ids)
    count = torch.cuda.device_count()
    return count if count > 0 else 1


def _load_caption_model(args):
    import torch
    import vllm
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    llm_path_lower = str(args.llm_path or "").strip().lower()
    if "qwen" in llm_path_lower and "vl" in llm_path_lower:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.llm_path,
            dtype="auto",
            device_map="auto",
        )
        tokenizer = AutoProcessor.from_pretrained(args.llm_path, padding_side="left")
        try:
            tokenizer.pad_token_id = 0
        except Exception:
            pass
        return "transformers", model, tokenizer

    tensor_parallel_size = _infer_tensor_parallel_size()
    available_gpus = torch.cuda.device_count()
    if available_gpus > 0 and tensor_parallel_size > available_gpus:
        tensor_parallel_size = available_gpus

    model = vllm.LLM(
        model=args.llm_path,
        tokenizer=args.llm_path,
        dtype=torch.bfloat16,
        tensor_parallel_size=max(1, tensor_parallel_size),
        max_model_len=DEFAULT_MAX_MODEL_LEN,
        max_num_seqs=max(1, int(args.batch_size)),
        max_num_batched_tokens=max(
            DEFAULT_MAX_MODEL_LEN,
            int(args.batch_size) * max(1, int(getattr(args, "max_tokens", DEFAULT_CAPTION_MAX_NEW_TOKENS))),
        ),
        gpu_memory_utilization=DEFAULT_GPU_MEMORY_UTILIZATION,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_path,
        padding_side="left",
        padding=True,
    )
    try:
        tokenizer.pad_token_id = 0
    except Exception:
        pass
    return "vllm", model, tokenizer


def _generate_captions(prompts: List[str], args, backend: str, model, tokenizer) -> List[str]:
    import torch
    from vllm import SamplingParams

    if not prompts:
        return []
    if backend == "transformers":
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=DEFAULT_MAX_MODEL_LEN,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        gen_kwargs = {
            "max_new_tokens": max(1, int(getattr(args, "max_tokens", DEFAULT_CAPTION_MAX_NEW_TOKENS))),
            "temperature": float(getattr(args, "temperature", DEFAULT_CAPTION_TEMPERATURE)),
            "top_p": float(getattr(args, "top_p", DEFAULT_CAPTION_TOP_P)),
            "do_sample": True,
            "pad_token_id": getattr(tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        }
        with torch.no_grad():
            output_ids = model.generate(**inputs, **gen_kwargs)
        decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        captions = []
        for prompt, text in zip(prompts, decoded):
            if text.startswith(prompt):
                text = text[len(prompt) :]
            captions.append(_clean_clip_caption(text))
        return captions

    sampling_params = SamplingParams(
        temperature=float(getattr(args, "temperature", DEFAULT_CAPTION_TEMPERATURE)),
        top_p=float(getattr(args, "top_p", DEFAULT_CAPTION_TOP_P)),
        max_tokens=max(1, int(getattr(args, "max_tokens", DEFAULT_CAPTION_MAX_NEW_TOKENS))),
    )
    captions = []
    batch_size = max(1, int(args.batch_size))
    try:
        from tqdm import tqdm

        batch_iter = tqdm(range(0, len(prompts), batch_size), desc="caption batches")
    except Exception:
        batch_iter = range(0, len(prompts), batch_size)
    for start in batch_iter:
        batch = prompts[start : start + batch_size]
        outputs = model.generate(prompts=batch, sampling_params=sampling_params)
        for output in outputs:
            text = output.outputs[0].text if output.outputs else ""
            captions.append(_clean_clip_caption(text))
    return captions


def _read_grouped_records(path: str) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    current_group: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            current_group.append(record)
            if record.get("clip_caption"):
                groups.append(current_group)
                current_group = []
    if current_group:
        groups.append(current_group)
    return groups


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


def _infer_model_name_from_path(llm_path: str) -> str:
    normalized = os.path.normpath(str(llm_path or "").strip())
    if not normalized:
        return "llm_caption_refiner"
    return os.path.basename(normalized) or "llm_caption_refiner"


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
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--language", type=str, default=DEFAULT_CAPTION_LANGUAGE)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_CAPTION_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CAPTION_TEMPERATURE)
    parser.add_argument("--top_p", type=float, default=DEFAULT_CAPTION_TOP_P)
    return parser.parse_args()


def backfill_captions(
    input_path: str,
    output_path: str,
    llm_path: str,
    batch_size: int = 8,
    language: str = DEFAULT_CAPTION_LANGUAGE,
    max_tokens: int = DEFAULT_CAPTION_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_CAPTION_TEMPERATURE,
    top_p: float = DEFAULT_CAPTION_TOP_P,
) -> dict:
    args = argparse.Namespace(
        input=str(input_path),
        output=str(output_path),
        llm_path=str(llm_path),
        batch_size=int(batch_size),
        language=str(language or DEFAULT_CAPTION_LANGUAGE),
        max_tokens=int(max_tokens),
        temperature=float(temperature),
        top_p=float(top_p),
    )
    groups = _read_grouped_records(args.input)
    if not groups:
        raise SystemExit(f"No caption groups found in {args.input}.")

    prompts = [_group_prompt(group, language=args.language) for group in groups]
    backend, model, tokenizer = _load_caption_model(args)
    raw_captions = _generate_captions(prompts, args, backend=backend, model=model, tokenizer=tokenizer)
    captions: List[str] = []
    caption_modes: List[str] = []
    fallback_count = 0
    for group, llm_caption in zip(groups, raw_captions):
        last_record = group[-1] if group else {}
        meta = last_record.get("meta") or {}
        if not isinstance(meta, dict):
            meta = {}
        movement_keys = meta.get("movement_keys")
        if not isinstance(movement_keys, list) or not movement_keys:
            movement_keys = list((last_record.get("movement_counts") or {}).keys()) or list(DEFAULT_MOVEMENT_KEYS)
        history = [
            {
                "date": item.get("date"),
                "movement_counts": item.get("movement_counts") or {},
            }
            for item in group
        ]
        metrics = _extract_history_metrics(history, movement_keys)
        if _caption_needs_fallback(llm_caption):
            captions.append(_build_structured_caption(metrics))
            caption_modes.append("structured_fallback")
            fallback_count += 1
        else:
            captions.append(llm_caption)
            caption_modes.append("llm")
    if len(captions) != len(groups):
        raise SystemExit(
            f"Caption count mismatch: generated {len(captions)} captions for {len(groups)} groups."
        )

    rewritten_records = _rewrite_groups(
        groups,
        captions,
        llm_model_name=_infer_model_name_from_path(args.llm_path),
        caption_modes=caption_modes,
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in rewritten_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "input_path": args.input,
        "output_path": args.output,
        "llm_path": args.llm_path,
        "llm_model_name": _infer_model_name_from_path(args.llm_path),
        "num_groups": len(groups),
        "num_records": len(rewritten_records),
        "language": args.language,
        "max_tokens": int(args.max_tokens),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "num_structured_fallback_captions": int(fallback_count),
        "num_llm_captions": int(len(groups) - fallback_count),
    }


def main():
    args = parse_args()
    result = backfill_captions(
        input_path=args.input,
        output_path=args.output,
        llm_path=args.llm_path,
        batch_size=args.batch_size,
        language=args.language,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(f"Wrote {result['num_groups']} caption groups to {args.output}")


if __name__ == "__main__":
    main()
