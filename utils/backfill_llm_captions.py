import argparse
import json
import os
from typing import Dict, List

DEFAULT_MOVEMENT_KEYS = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
DEFAULT_CAPTION_LANGUAGE = "en"
DEFAULT_CAPTION_MAX_NEW_TOKENS = 64
DEFAULT_CAPTION_TEMPERATURE = 0.2
DEFAULT_CAPTION_TOP_P = 0.9
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


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

    per_bin_totals = []
    window_counts = {k: 0 for k in keys}
    for entry in history:
        counts = entry.get("movement_counts") or {}
        total = 0
        for key in keys:
            try:
                value = int(counts.get(key, 0) or 0)
            except (TypeError, ValueError):
                value = 0
            window_counts[key] += value
            total += value
        per_bin_totals.append(total)

    total_sum = sum(per_bin_totals)
    avg_total = total_sum / max(len(per_bin_totals), 1)
    first_total = per_bin_totals[0] if per_bin_totals else 0
    last_total = per_bin_totals[-1] if per_bin_totals else 0
    trend = _clip_trend_label(first_total, last_total)
    peak_idx = max(range(len(per_bin_totals)), key=lambda i: per_bin_totals[i]) if per_bin_totals else 0
    peak_total = per_bin_totals[peak_idx] if per_bin_totals else 0
    peak_time = history[peak_idx].get("date") if history else ""

    sorted_keys = sorted(keys, key=lambda k: window_counts.get(k, 0), reverse=True)
    top_keys = [k for k in sorted_keys if window_counts.get(k, 0) > 0][:3]
    top_phrase = ", ".join(f"{k}({window_counts.get(k, 0)})" for k in top_keys) if top_keys else "none"

    through_sum = sum(window_counts.get(k, 0) for k in keys if str(k).endswith("T"))
    left_sum = sum(window_counts.get(k, 0) for k in keys if str(k).endswith("L"))
    ew_sum = sum(window_counts.get(k, 0) for k in keys if str(k).startswith(("E", "W")))
    ns_sum = sum(window_counts.get(k, 0) for k in keys if str(k).startswith(("N", "S")))
    total_dir = max(ew_sum + ns_sum, 1)
    ew_share = ew_sum / total_dir
    ns_share = ns_sum / total_dir
    if abs(ew_share - ns_share) < 0.1:
        dominance = "balanced"
    else:
        dominance = "east-west dominant" if ew_share > ns_share else "north-south dominant"

    start_time = history[0].get("date")
    end_time = history[-1].get("date")
    lines = [
        f"Window: {start_time} to {end_time} (interval {int(interval_s)}s, span {int(window_s)}s).",
        f"Total vehicles in window: {total_sum}, average per bin: {avg_total:.1f}, trend: {trend}.",
        f"Peak bin: {peak_time} with {peak_total} vehicles.",
        f"Top movements: {top_phrase}.",
        f"Direction balance: EW {ew_share:.0%}, NS {ns_share:.0%} ({dominance}).",
        f"Turn balance: through {through_sum}, left {left_sum}.",
    ]
    return "\n".join(lines)


def _clip_caption_instruction_series(language: str) -> str:
    lang = str(language or DEFAULT_CAPTION_LANGUAGE).strip().lower()
    target_language = "English" if lang in ("en", "eng", "english") else lang
    return (
        "Convert the following structured traffic time-series summary into a concise visual description "
        "of the scene, as if captioning a single image. Focus on vehicle volume, dominant directions, "
        "queues, and flow. Do not mention data, time series, or charts. "
        f"Write in {target_language}. Output one sentence with no quotes."
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
        max_num_batched_tokens=max(DEFAULT_MAX_MODEL_LEN, int(args.batch_size) * DEFAULT_CAPTION_MAX_NEW_TOKENS),
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
            "max_new_tokens": DEFAULT_CAPTION_MAX_NEW_TOKENS,
            "temperature": DEFAULT_CAPTION_TEMPERATURE,
            "top_p": DEFAULT_CAPTION_TOP_P,
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
        temperature=DEFAULT_CAPTION_TEMPERATURE,
        top_p=DEFAULT_CAPTION_TOP_P,
        max_tokens=DEFAULT_CAPTION_MAX_NEW_TOKENS,
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


def _group_prompt(group: List[Dict]) -> str:
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
    instruction = _clip_caption_instruction_series(language=DEFAULT_CAPTION_LANGUAGE)
    return f"{instruction}\n\nTraffic summary:\n{summary}\n\nCaption:"


def _infer_model_name_from_path(llm_path: str) -> str:
    normalized = os.path.normpath(str(llm_path or "").strip())
    if not normalized:
        return "llm_caption_refiner"
    return os.path.basename(normalized) or "llm_caption_refiner"


def _rewrite_groups(groups: List[List[Dict]], captions: List[str], llm_model_name: str) -> List[Dict]:
    rewritten: List[Dict] = []
    for group, caption in zip(groups, captions):
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
            meta["caption_style"] = "clip_caption_v2_llm_backfill"
            last_record["meta"] = meta
        rewritten.extend(updated_group)
    return rewritten


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--llm_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    groups = _read_grouped_records(args.input)
    if not groups:
        raise SystemExit(f"No caption groups found in {args.input}.")

    prompts = [_group_prompt(group) for group in groups]
    backend, model, tokenizer = _load_caption_model(args)
    captions = _generate_captions(prompts, args, backend=backend, model=model, tokenizer=tokenizer)
    if len(captions) != len(groups):
        raise SystemExit(
            f"Caption count mismatch: generated {len(captions)} captions for {len(groups)} groups."
        )

    rewritten_records = _rewrite_groups(
        groups,
        captions,
        llm_model_name=_infer_model_name_from_path(args.llm_path),
    )
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for record in rewritten_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(groups)} caption groups to {args.output}")


if __name__ == "__main__":
    main()
