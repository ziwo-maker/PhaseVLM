import os
import sys
import time
import argparse
import torch
from utils import error
from utils.llm_aft_trainer import LLM_Inference_VLLM
from utils.config import *
from utils.utils import merge, location_direction_dict
from src.TimeVLM.ts_image_adapter import TimeSeriesImageAdapter

_PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default='LLMTLCSRun')
    parser.add_argument("--llm_model", type=str, default="gemma-3-27B")
    parser.add_argument("--llm_path", type=str, default="/home/data/model/gemma-3-27B/")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--new_max_tokens", type=int, default=1024)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS-extreme")
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument("--multi_process", action="store_true", default=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=0)
    parser.add_argument("--dataset", type=str, default="template")
    parser.add_argument("--traffic_file", type=str, default="flow_main_stream.json")
    parser.add_argument("--ts_image_service_url", type=str, default="")
    parser.add_argument("--ts_image_seq_len", type=int, default=25)
    parser.add_argument("--ts_image_token", type=str, default="<image>")
    parser.add_argument("--ts_image_prompt_prefix", type=str, default="")
    parser.add_argument("--ts_feature_mode", type=str, default="movement_total")
    parser.add_argument("--ts_image_timeout", type=float, default=5.0)

    return parser.parse_args()

def _resolve_ts_adapter_device():
    if not torch.cuda.is_available():
        return None
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            local_rank = int(local_rank)
        except ValueError:
            local_rank = 0
        return f"cuda:{local_rank}"
    return "cuda:0"


def _normalize_llm_model_name(llm_model, llm_path):
    raw_name = str(llm_model or "").strip()
    if not raw_name:
        raw_name = os.path.basename(os.path.normpath(str(llm_path or "").strip()))

    alias = raw_name.lower().replace("_", "-")
    alias_map = {
        "qwen3-8b": "qwen3-vl-8B",
        "qwen3-vl-8b": "qwen3-vl-8B",
        "qwen3-vl-8b-instruct": "qwen3-vl-8B",
        "qwen-vl-8b": "qwen3-vl-8B",
    }
    return alias_map.get(alias, raw_name)


def _project_path(*parts):
    return os.path.join(_PROJECT_ROOT, *parts)


def _prepare_project_output_root(dirname):
    base_path = _project_path(dirname)
    if os.path.islink(base_path):
        target_path = os.path.realpath(base_path)
        try:
            os.makedirs(target_path, exist_ok=True)
            return base_path
        except OSError as exc:
            fallback_path = _project_path(f"{dirname}_local")
            os.makedirs(fallback_path, exist_ok=True)
            print(
                f"[run_open_LLM_with_vllm] Warning: failed to use '{base_path}' "
                f"(symlink target: '{target_path}'). Reason: {exc}. "
                f"Falling back to '{fallback_path}'.",
                file=sys.stderr,
                flush=True,
            )
            return fallback_path
    os.makedirs(base_path, exist_ok=True)
    return base_path


def main(in_args):
    in_args.llm_path = os.path.normpath(os.path.expanduser(str(in_args.llm_path).strip()))
    in_args.llm_model = _normalize_llm_model_name(in_args.llm_model, in_args.llm_path)
    traffic_file_list = []

    if in_args.dataset == 'jinan':
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real.json", "anon_3_4_jinan_real_2000.json",
                             "anon_3_4_jinan_real_2500.json", "anon_3_4_jinan_synthetic_24000_60min.json",
                             "anon_3_4_jinan_synthetic_24h_6000.json"]
        template = "Jinan"
    elif in_args.dataset == 'hangzhou':
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json", "anon_4_4_hangzhou_real_5816.json",
                             "anon_4_4_hangzhou_synthetic_24000_60min.json"]
        template = "Hangzhou"
    elif in_args.dataset == 'newyork_16x3':
        count = 3600
        road_net = "16_3"
        traffic_file_list = ["anon_16_3_newyork_real.json"]
        template = "NewYork"
    elif in_args.dataset == 'newyork_28x7':
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        template = "NewYork"
    in_args.model = in_args.memo

    if "24h" in in_args.traffic_file:
        count = 86400

    # flow_file error
    try:
        if in_args.traffic_file not in traffic_file_list:
            raise error.flowFileException('Flow file does not exist.')
    except error.flowFileException as e:
        print(e)
        return
    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(in_args.traffic_file)

    dic_agent_conf_extra = {
        "LLM_PATH": in_args.llm_path,
        "LLM_MODEL": in_args.llm_model,
        "LOG_DIR": f"./{in_args.llm_model}_logs",
        "NEW_MAX_TOKENS": in_args.new_max_tokens,
        "TENSOR_PARALLEL_SIZE": in_args.tensor_parallel_size,
        "TS_IMAGE_SERVICE_URL": in_args.ts_image_service_url,
        "TS_IMAGE_SEQ_LEN": in_args.ts_image_seq_len,
        "TS_IMAGE_TOKEN": in_args.ts_image_token,
        "TS_IMAGE_PROMPT_PREFIX": in_args.ts_image_prompt_prefix,
        "TS_FEATURE_MODE": in_args.ts_feature_mode,
        "TS_IMAGE_TIMEOUT": in_args.ts_image_timeout,
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,

        "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['LLM_MODEL']}",
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": count,
        "NUM_ROUNDS": in_args.num_rounds,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,
        "ENABLE_TRAFFIC_COUNT": True,
        "TRAFFIC_COUNT_MODE": "intersection_movement",
        "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
        "TRAFFIC_COUNT_INTERVALS": [2, 5, 60, 3600],
        "TRAFFIC_COUNT_ADD_TOTAL": True,
        "TRAFFIC_COUNT_BASENAME": "traffic_counts",

        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),

        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],

        "DIC_REWARD_INFO": {
            'queue_length': -0.25
        }
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
            8: [0, 0, 0, 0, 1, 1, 0, 0]
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = ['WT_ET', 'NT_ST', 'WL_EL', 'NL_SL',
                                                    'WL_WT', 'EL_ET', 'SL_ST', 'NL_NT']
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30, 30, 30, 30, 30]

    else:
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra["NUM_INTERSECTIONS"]
    timestamp = time.strftime('%m_%d_%H_%M_%S', time.localtime(time.time()))
    model_root = _prepare_project_output_root("model")
    records_root = _prepare_project_output_root("records")
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join(model_root, in_args.memo, f"{in_args.traffic_file}_{timestamp}"),
        "PATH_TO_WORK_DIRECTORY": os.path.join(records_root, in_args.memo, f"{in_args.traffic_file}_{timestamp}"),
        "PATH_TO_DATA": _project_path("data", template, str(road_net))
    }

    ts_adapter = TimeSeriesImageAdapter(
        input_dim=len(location_direction_dict),
        seq_len=in_args.ts_image_seq_len,
        device=_resolve_ts_adapter_device()
    )
    trainer = LLM_Inference_VLLM(dic_agent_conf_extra,
                                 merge(dic_traffic_env_conf, dic_traffic_env_conf_extra),
                                 dic_path_extra,
                                 f'{template}-{road_net}', in_args.traffic_file.split(".")[0],
                                 ts_image_adapter=ts_adapter)

    trainer.train_test()


if __name__ == "__main__":
    # Ensure vLLM uses spawn to avoid CUDA re-init in forked subprocesses.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method already set by parent process/debugger.
        pass
    args = parse_args()
    main(args)
