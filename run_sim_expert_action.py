import os
import time
import argparse
import json
from copy import deepcopy

from utils.clean_align_jsonl import clean_jsonl
from utils.backfill_llm_captions import backfill_captions
from utils import error
from utils.config import *


def _merge_dict(base, extra):
    merged = deepcopy(base)
    merged.update(extra)
    return merged


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default="SimExpertRun")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS-extreme")
    parser.add_argument("--llm_model", type=str, default="gemma-3-27B")
    parser.add_argument("--llm_path", type=str, default="/home/data/model/gemma-3-27B/")
    parser.add_argument("--new_max_tokens", type=int, default=1024)
    parser.add_argument("--tensor_parallel_size", type=int, default=0)
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--dataset", type=str, default="template")
    parser.add_argument(
        "--traffic_file",
        type=str,
        nargs="+",
        default=None,
        help="One or more traffic files, or 'all' to run every file for the dataset.",
    )
    parser.add_argument("--ts_image_seq_len", type=int, default=25)
    parser.add_argument("--sim_wait_speed", type=float, default=1.0)
    parser.add_argument("--sim_horizon_s", type=int, default=0)
    parser.add_argument("--clip_caption_mode", type=str, default="prompt_llm")
    parser.add_argument("--clip_caption_language", type=str, default="en")
    parser.add_argument("--clip_caption_max_tokens", type=int, default=2048)
    parser.add_argument("--clip_caption_temperature", type=float, default=0.2)
    parser.add_argument("--clip_caption_top_p", type=float, default=0.9)
    parser.add_argument("--expert_att_weight", type=float, default=1.0)
    parser.add_argument("--expert_aql_weight", type=float, default=1.0)
    parser.add_argument("--expert_awt_weight", type=float, default=1.0)
    parser.add_argument("--expert_spillback_weight", type=float, default=0.2)
    parser.add_argument("--clean_seq_len", type=int, default=16)
    parser.add_argument(
        "--clean_output",
        type=str,
        default="",
        help="Optional cleaned jsonl output path. Defaults to <raw>.cleaned.jsonl.",
    )
    parser.add_argument(
        "--backfill_output",
        type=str,
        default="",
        help="Optional LLM caption output path. Defaults to <clean_or_raw>.llm_caption.jsonl.",
    )
    parser.add_argument("--skip_clean", action="store_true", default=False)
    parser.add_argument("--skip_llm_backfill", action="store_true", default=False)
    parser.add_argument(
        "--backfill_llm_path",
        type=str,
        default="",
        help="LLM path for caption backfill. Defaults to --llm_path when omitted.",
    )
    parser.add_argument("--backfill_batch_size", type=int, default=8)

    return parser.parse_args()


def _get_dataset_spec(dataset):
    if dataset == "jinan":
        count = 3600
        road_net = "3_4"
        traffic_file_list = [
            "anon_3_4_jinan_real.json",
            "anon_3_4_jinan_real_2000.json",
            "anon_3_4_jinan_real_2500.json",
            "anon_3_4_jinan_synthetic_24000_60min.json",
            "anon_3_4_jinan_synthetic_24h_6000.json",
        ]
        template = "Jinan"
    elif dataset == "hangzhou":
        count = 3600
        road_net = "4_4"
        traffic_file_list = [
            "anon_4_4_hangzhou_real.json",
            "anon_4_4_hangzhou_real_5816.json",
            "anon_4_4_hangzhou_synthetic_24000_60min.json",
        ]
        template = "Hangzhou"
    elif dataset == "newyork_16x3":
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        template = "NewYork"
    elif dataset == "newyork_28x7":
        count = 3600
        road_net = "28_7"
        traffic_file_list = [
            "anon_28_7_newyork_real_double.json",
            "anon_28_7_newyork_real_triple.json",
        ]
        template = "NewYork"
    else:
        raise error.flowFileException("Unknown dataset.")
    return {
        "count": count,
        "road_net": road_net,
        "traffic_file_list": traffic_file_list,
        "template": template,
    }


def _normalize_requested_traffic_files(raw_traffic_files):
    if not raw_traffic_files:
        return []
    normalized = []
    for item in raw_traffic_files:
        if item is None:
            continue
        for part in str(item).split(","):
            token = part.strip()
            if token:
                normalized.append(token)
    return normalized


def _resolve_traffic_files(requested_files, available_files):
    requested = _normalize_requested_traffic_files(requested_files)
    if not requested:
        raise error.flowFileException(
            "Flow file does not exist. Please provide --traffic_file <file1> [file2 ...] or --traffic_file all."
        )

    lowered = {item.lower() for item in requested}
    if "all" in lowered or "*" in lowered:
        return list(available_files)

    resolved = []
    seen = set()
    for traffic_file in requested:
        if traffic_file not in available_files:
            raise error.flowFileException(f"Flow file does not exist: {traffic_file}")
        if traffic_file in seen:
            continue
        seen.add(traffic_file)
        resolved.append(traffic_file)
    return resolved


def _resolve_run_count(default_count, traffic_file):
    return 86400 if "24h" in str(traffic_file) else int(default_count)


def _build_single_run_paths(memo, traffic_file, run_tag):
    return {
        "PATH_TO_MODEL": os.path.join("model", memo, f"{traffic_file}_{run_tag}"),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, f"{traffic_file}_{run_tag}"),
    }


def _build_batch_run_paths(memo, traffic_file, batch_tag):
    traffic_stem = os.path.splitext(os.path.basename(str(traffic_file)))[0]
    run_name = f"{traffic_stem}_{batch_tag}"
    return {
        "PATH_TO_MODEL": os.path.join("model", memo, f"batch_{batch_tag}", run_name),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", memo, f"batch_{batch_tag}", run_name),
    }


def _write_batch_manifest(memo, batch_tag, batch_summary):
    if not batch_summary:
        return
    batch_work_root = os.path.join("records", memo, f"batch_{batch_tag}")
    batch_model_root = os.path.join("model", memo, f"batch_{batch_tag}")
    os.makedirs(batch_work_root, exist_ok=True)
    os.makedirs(batch_model_root, exist_ok=True)
    manifest_path = os.path.join(batch_work_root, "batch_manifest.json")
    payload = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())),
        "memo": memo,
        "batch_tag": batch_tag,
        "num_files": len(batch_summary),
        "runs": batch_summary,
    }
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)
    print("batch_manifest:", manifest_path)


def _extract_eval_metrics(test_results):
    if not isinstance(test_results, dict):
        return {}

    metrics = {}
    mapping = {
        "ATT": "test_avg_travel_time_over",
        "AQL": "test_avg_queue_len_over",
        "AWT": "test_avg_waiting_time_over",
    }
    for output_key, source_key in mapping.items():
        value = test_results.get(source_key)
        try:
            metrics[output_key] = float(value)
        except (TypeError, ValueError):
            continue
    return metrics


def _print_eval_metrics_by_file(batch_summary):
    if not batch_summary:
        return
    print("sim_eval_metrics_by_file:")
    for item in batch_summary:
        traffic_file = str(item.get("traffic_file", "") or "").strip()
        metrics = item.get("evaluation_metrics") or {}
        att = metrics.get("ATT")
        aql = metrics.get("AQL")
        awt = metrics.get("AWT")
        if att is None and aql is None and awt is None:
            print(f"  {traffic_file}: ATT=N/A | AQL=N/A | AWT=N/A")
            continue
        att_text = f"{float(att):.4f}" if att is not None else "N/A"
        aql_text = f"{float(aql):.4f}" if aql is not None else "N/A"
        awt_text = f"{float(awt):.4f}" if awt is not None else "N/A"
        print(f"  {traffic_file}: ATT={att_text} | AQL={aql_text} | AWT={awt_text}")


def _merge_batch_training_sets(batch_tag, batch_summary):
    if not batch_summary:
        return ""
    output_dir = os.path.join("data", "recorded", f"batch_{batch_tag}")
    os.makedirs(output_dir, exist_ok=True)
    merged_output_path = os.path.join(output_dir, "traffic_prompt_series_2s.jsonl")
    merged_count = 0
    with open(merged_output_path, "w", encoding="utf-8") as fout:
        for item in batch_summary:
            input_path = str(item.get("traffic_series_output_path", "") or "").strip()
            if not input_path or not os.path.exists(input_path):
                continue
            with open(input_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    fout.write(line)
                    merged_count += 1
    print("merged_traffic_series_output_path:", merged_output_path)
    print("merged_traffic_series_records:", merged_count)
    return merged_output_path


def _strip_jsonl_suffix(path):
    normalized = str(path or "").strip()
    if normalized.lower().endswith(".jsonl"):
        return normalized[:-6]
    return normalized


def _resolve_clean_output_path(raw_input_path, requested_output):
    requested = str(requested_output or "").strip()
    if requested:
        return requested
    return f"{_strip_jsonl_suffix(raw_input_path)}.cleaned.jsonl"


def _resolve_backfill_output_path(backfill_input_path, requested_output):
    requested = str(requested_output or "").strip()
    if requested:
        return requested
    return f"{_strip_jsonl_suffix(backfill_input_path)}.llm_caption.jsonl"


def _resolve_backfill_llm_path(in_args):
    llm_path = str(in_args.backfill_llm_path or in_args.llm_path or "").strip()
    if not llm_path:
        raise SystemExit("LLM backfill requires --backfill_llm_path or --llm_path.")
    return llm_path


def _print_clean_stats(stats):
    print("clean_stats:")
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


def _run_postprocess_pipeline(in_args, raw_input_path):
    if not raw_input_path or not os.path.exists(raw_input_path):
        raise SystemExit(f"Pipeline input jsonl not found: {raw_input_path}")

    result = {
        "pipeline_input_path": raw_input_path,
        "clean_output_path": "",
        "llm_caption_output_path": "",
        "clean_stats": None,
        "backfill_stats": None,
    }

    current_input_path = raw_input_path
    if not in_args.skip_clean:
        clean_output_path = _resolve_clean_output_path(raw_input_path, in_args.clean_output)
        clean_stats = clean_jsonl(
            inputs=[raw_input_path],
            output_path=clean_output_path,
            seq_len=in_args.clean_seq_len,
        )
        print("clean_output_path:", clean_output_path)
        _print_clean_stats(clean_stats)
        result["clean_output_path"] = clean_output_path
        result["clean_stats"] = clean_stats
        current_input_path = clean_output_path

    if not in_args.skip_llm_backfill:
        backfill_output_path = _resolve_backfill_output_path(current_input_path, in_args.backfill_output)
        backfill_stats = backfill_captions(
            input_path=current_input_path,
            output_path=backfill_output_path,
            llm_path=_resolve_backfill_llm_path(in_args),
            batch_size=in_args.backfill_batch_size,
            language=in_args.clip_caption_language,
            max_tokens=in_args.clip_caption_max_tokens,
            temperature=in_args.clip_caption_temperature,
            top_p=in_args.clip_caption_top_p,
        )
        print("llm_caption_output_path:", backfill_output_path)
        print(
            "llm_caption_stats:",
            json.dumps(backfill_stats, ensure_ascii=False),
        )
        result["llm_caption_output_path"] = backfill_output_path
        result["backfill_stats"] = backfill_stats

    return result


def _run_single_traffic_file(in_args, dataset_spec, traffic_file, run_tag, batch_tag=None):
    from utils.sim_expert_collector import SimExpertCollector

    count = _resolve_run_count(dataset_spec["count"], traffic_file)
    road_net = dataset_spec["road_net"]
    template = dataset_spec["template"]
    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])
    num_intersections = num_row * num_col

    print("num_intersections:", num_intersections)
    print("traffic_file:", traffic_file)

    in_args.model = in_args.memo

    dic_agent_conf_extra = {
        "LLM_MODEL": in_args.llm_model,
        "LLM_PATH": in_args.llm_path,
        "NEW_MAX_TOKENS": in_args.new_max_tokens,
        "TENSOR_PARALLEL_SIZE": in_args.tensor_parallel_size,
        "SIM_WAIT_SPEED": in_args.sim_wait_speed,
        "CLIP_CAPTION_MODE": in_args.clip_caption_mode,
        "CLIP_CAPTION_LANGUAGE": in_args.clip_caption_language,
        "CLIP_CAPTION_MAX_TOKENS": in_args.clip_caption_max_tokens,
        "CLIP_CAPTION_TEMPERATURE": in_args.clip_caption_temperature,
        "CLIP_CAPTION_TOP_P": in_args.clip_caption_top_p,
        "EXPERT_ATT_WEIGHT": in_args.expert_att_weight,
        "EXPERT_AQL_WEIGHT": in_args.expert_aql_weight,
        "EXPERT_AWT_WEIGHT": in_args.expert_awt_weight,
        "EXPERT_SPILLBACK_WEIGHT": in_args.expert_spillback_weight,
    }
    if in_args.sim_horizon_s and int(in_args.sim_horizon_s) > 0:
        dic_agent_conf_extra["SIM_HORIZON_S"] = int(in_args.sim_horizon_s)
    if in_args.ts_image_seq_len and int(in_args.ts_image_seq_len) > 0:
        dic_agent_conf_extra["TS_IMAGE_SEQ_LEN"] = int(in_args.ts_image_seq_len)

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['LLM_MODEL']}",
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": count,
        "NUM_ROUNDS": in_args.num_rounds,
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "ENABLE_TRAFFIC_COUNT": True,
        "TRAFFIC_COUNT_MODE": "intersection_movement",
        "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
        "TRAFFIC_COUNT_INTERVALS": [2, 5, 60, 3600],
        "TRAFFIC_COUNT_ADD_TOTAL": True,
        "TRAFFIC_COUNT_BASENAME": "traffic_counts",
        "TRAFFIC_FILE": traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],
        "DIC_REWARD_INFO": {"queue_length": -0.25},
    }

    if in_args.eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0],
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = [
            "WT_ET",
            "NT_ST",
            "WL_EL",
            "NL_SL",
            "WL_WT",
            "EL_ET",
            "SL_ST",
            "NL_NT",
        ]
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30, 30, 30, 30, 30]
    else:
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
    if batch_tag is None:
        dic_path_extra = _build_single_run_paths(in_args.memo, traffic_file, run_tag)
    else:
        dic_path_extra = _build_batch_run_paths(in_args.memo, traffic_file, batch_tag)
    dic_path_extra["PATH_TO_DATA"] = os.path.join("data", template, str(road_net))

    trainer = SimExpertCollector(
        dic_agent_conf_extra,
        _merge_dict(dic_traffic_env_conf, dic_traffic_env_conf_extra),
        dic_path_extra,
        f"{template}-{road_net}",
        traffic_file.split(".")[0],
    )
    test_results = trainer.train_test()
    traffic_series_run_tag = os.path.basename(dic_path_extra["PATH_TO_WORK_DIRECTORY"])
    traffic_series_output_path = os.path.join(
        "data",
        "recorded",
        traffic_series_run_tag,
        "traffic_prompt_series_2s.jsonl",
    )
    return {
        "traffic_file": traffic_file,
        "run_counts": int(count),
        "path_to_model": dic_path_extra["PATH_TO_MODEL"],
        "path_to_work_directory": dic_path_extra["PATH_TO_WORK_DIRECTORY"],
        "traffic_series_output_path": traffic_series_output_path,
        "evaluation_metrics": _extract_eval_metrics(test_results),
    }


def main(in_args):
    dataset_spec = _get_dataset_spec(in_args.dataset)

    try:
        resolved_traffic_files = _resolve_traffic_files(
            in_args.traffic_file,
            dataset_spec["traffic_file_list"],
        )
    except error.flowFileException as e:
        print(e)
        return

    batch_tag = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
    run_tag = batch_tag
    print("resolved_traffic_files:", resolved_traffic_files)

    batch_summary = []
    multi_file_mode = len(resolved_traffic_files) > 1
    merged_output_path = ""
    for idx, traffic_file in enumerate(resolved_traffic_files, start=1):
        print(f"===== [{idx}/{len(resolved_traffic_files)}] start {traffic_file} =====")
        result = _run_single_traffic_file(
            in_args=in_args,
            dataset_spec=dataset_spec,
            traffic_file=traffic_file,
            run_tag=run_tag,
            batch_tag=batch_tag if multi_file_mode else None,
        )
        batch_summary.append(result)

    _print_eval_metrics_by_file(batch_summary)

    if multi_file_mode:
        merged_output_path = _merge_batch_training_sets(batch_tag, batch_summary)
        for item in batch_summary:
            item["merged_traffic_series_output_path"] = merged_output_path

    pipeline_input_path = merged_output_path if multi_file_mode else batch_summary[0]["traffic_series_output_path"]
    postprocess_result = _run_postprocess_pipeline(in_args, pipeline_input_path)
    print("pipeline_input_path:", pipeline_input_path)
    if postprocess_result.get("clean_output_path"):
        print("pipeline_clean_output_path:", postprocess_result["clean_output_path"])
    if postprocess_result.get("llm_caption_output_path"):
        print("pipeline_llm_caption_output_path:", postprocess_result["llm_caption_output_path"])

    if multi_file_mode:
        for item in batch_summary:
            item["pipeline_input_path"] = postprocess_result["pipeline_input_path"]
            item["clean_output_path"] = postprocess_result["clean_output_path"]
            item["llm_caption_output_path"] = postprocess_result["llm_caption_output_path"]
        _write_batch_manifest(in_args.memo, batch_tag, batch_summary)


if __name__ == "__main__":
    args = parse_args()
    main(args)
