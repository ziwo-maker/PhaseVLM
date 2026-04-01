from pyexpat.errors import messages
from utils.my_utils import dump_json, get_state_detail, state2text, getPrompt, action2code, code2action, eight_phase_list, four_phase_list, torch_gc, location_direction_dict
from vllm import SamplingParams
from src.TimeVLM.ts_image_adapter import TimeSeriesImageAdapter
from layers.TimeSeries_To_Image import time_series_to_simple_image
from layers.Learnable_TimeSeries_To_Image import LearnableTimeSeriesToImage
import vllm
import os
import fcntl
import time
import numpy as np
import wandb
from utils.cityflow_env import CityFlowEnv
import utils.config as config
from utils.aft_rank_loss_utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from tqdm import tqdm
import torch
from copy import deepcopy
import re
import json
import shutil
import copy
import random
from collections import deque
from datetime import datetime, timedelta
from contextlib import contextmanager
from types import SimpleNamespace
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
def merge(dic_tmp, dic_to_change):
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result

def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("1", "true", "yes", "y")

def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    return str(value)


@contextmanager
def _locked_file(path, mode="a+", encoding="utf-8"):
    handle = open(path, mode, encoding=encoding)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    except Exception:
        handle.close()
        raise
    try:
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def _validate_jsonl_handle(handle, path):
    handle.seek(0)
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            raise SystemExit(f"JSONL integrity check failed for {path}: empty line at {line_no}.")
        try:
            json.loads(line)
        except Exception as exc:
            raise SystemExit(
                f"JSONL integrity check failed for {path} at line {line_no}: {exc}"
            )


def _append_jsonl_records(path, records):
    if not path or not records:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with _locked_file(path, "a+", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        _validate_jsonl_handle(handle, path)


def _prepare_jsonl_file(path):
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with _locked_file(path, "a+", encoding="utf-8") as handle:
        _validate_jsonl_handle(handle, path)


def _shared_token_in_vocab(tokenizer, token):
    if not token or tokenizer is None:
        return False
    convert_fn = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert_fn is None:
        return False
    try:
        token_id = convert_fn(token)
    except Exception:
        return False
    if token_id is None:
        return False
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == unk_id:
        return False
    try:
        vocab = tokenizer.get_vocab()
    except Exception:
        vocab = None
    if vocab is not None and token not in vocab:
        return False
    return True


def _shared_extract_image_token_candidates(llm_path, tokenizer):
    candidates = []

    if llm_path:
        for filename in ("processor_config.json", "preprocessor_config.json", "tokenizer_config.json", "config.json"):
            path = os.path.join(llm_path, filename)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for key in (
                "image_token",
                "image_placeholder",
                "image_token_str",
                "image_token_string",
                "image_special_token",
            ):
                value = data.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)
            for nested_key in ("image_processor", "processor", "multimodal_config", "mm_config"):
                nested = data.get(nested_key)
                if isinstance(nested, dict):
                    for key in (
                        "image_token",
                        "image_placeholder",
                        "image_token_str",
                        "image_token_string",
                        "image_special_token",
                    ):
                        value = nested.get(key)
                        if isinstance(value, str) and value:
                            candidates.append(value)

    if tokenizer is not None:
        for attr in ("image_token", "image_placeholder", "image_token_str", "image_token_string"):
            value = getattr(tokenizer, attr, None)
            if isinstance(value, str) and value:
                candidates.append(value)
        extra = getattr(tokenizer, "additional_special_tokens", None) or []
        for token in extra:
            if isinstance(token, str) and token:
                candidates.append(token)

    return candidates


def _shared_resolve_image_token(instance):
    cached = getattr(instance, "_resolved_image_token", None)
    if cached:
        return cached

    config_token = str(getattr(instance, "ts_image_token", "") or "").strip()
    tokenizer = getattr(instance, "tokenizer", None)
    if config_token and _shared_token_in_vocab(tokenizer, config_token):
        instance._resolved_image_token = config_token
        instance.ts_image_token = config_token
        return config_token

    dic_agent_conf = getattr(instance, "dic_agent_conf", None)
    llm_path = ""
    if isinstance(dic_agent_conf, dict):
        llm_path = str(dic_agent_conf.get("LLM_PATH", "")).strip()
    candidates = []
    candidates.extend(_shared_extract_image_token_candidates(llm_path, tokenizer))
    candidates.extend(
        [
            "<start_of_image>",
            "<image>",
            "<image_placeholder>",
            "<image_start>",
            "<im_start>",
            "<img>",
        ]
    )
    if config_token:
        candidates.insert(0, config_token)

    seen = set()
    for token in candidates:
        token = token.strip()
        if not token or token in seen:
            continue
        seen.add(token)
        if _shared_token_in_vocab(tokenizer, token):
            instance._resolved_image_token = token
            instance.ts_image_token = token
            return token

    resolved = config_token or (candidates[0] if candidates else "<image>")
    instance._resolved_image_token = resolved
    instance.ts_image_token = resolved
    return resolved


def _clip_level(value: float, reference: float) -> str:
    if value <= 0:
        return "none"
    if reference <= 0:
        return "low"
    ratio = value / reference
    if ratio < 0.5:
        return "low"
    if ratio < 1.5:
        return "moderate"
    return "high"


def _clip_trend_label(first_total: float, last_total: float) -> str:
    delta = last_total - first_total
    threshold = max(2.0, 0.2 * max(first_total, 1.0))
    if delta >= threshold:
        return "rising"
    if delta <= -threshold:
        return "declining"
    return "steady"


def _build_clip_caption(history: list[dict], movement_keys: list[str]) -> str:
    if not history:
        return ""
    last = history[-1].get("movement_counts") or {}
    first = history[0].get("movement_counts") or {}
    keys = list(movement_keys or []) or list(last.keys())
    if not keys:
        return ""

    counts = {}
    for key in keys:
        try:
            counts[key] = int(last.get(key, 0) or 0)
        except (TypeError, ValueError):
            counts[key] = 0

    total_last = sum(counts.values())
    if total_last <= 0:
        return "No vehicles observed in the latest window."

    try:
        total_first = sum(int(first.get(k, 0) or 0) for k in keys)
    except (TypeError, ValueError):
        total_first = 0

    sorted_keys = sorted(keys, key=lambda k: counts.get(k, 0), reverse=True)
    top_keys = [k for k in sorted_keys if counts.get(k, 0) > 0][:2]
    if not top_keys:
        top_keys = sorted_keys[:1]
    top_phrase = " and ".join(f"{k}({counts.get(k, 0)})" for k in top_keys if k)
    top_sum = sum(counts.get(k, 0) for k in top_keys)
    top_share = top_sum / total_last if total_last > 0 else 0.0

    through_keys = [k for k in keys if str(k).endswith("T")]
    left_keys = [k for k in keys if str(k).endswith("L")]
    overall_avg = total_last / max(len(keys), 1)
    through_avg = (
        sum(counts.get(k, 0) for k in through_keys) / max(len(through_keys), 1)
        if through_keys
        else 0.0
    )
    left_avg = (
        sum(counts.get(k, 0) for k in left_keys) / max(len(left_keys), 1)
        if left_keys
        else 0.0
    )
    through_level = _clip_level(through_avg, overall_avg)
    left_level = _clip_level(left_avg, overall_avg)
    trend = _clip_trend_label(total_first, total_last)

    ew_sum = sum(counts.get(k, 0) for k in keys if str(k).startswith(("E", "W")))
    ns_sum = sum(counts.get(k, 0) for k in keys if str(k).startswith(("N", "S")))
    ew_share = ew_sum / total_last if total_last > 0 else 0.0
    ns_share = ns_sum / total_last if total_last > 0 else 0.0
    if abs(ew_share - ns_share) < 0.1:
        dominance = "balanced"
    else:
        dominance = "east-west dominant" if ew_share > ns_share else "north-south dominant"

    return (
        f"Highest demand on {top_phrase}, top-2 share {top_share:.0%}. "
        f"Through lanes {through_level}, left-turn lanes {left_level}; overall demand {trend}. "
        f"East-West share {ew_share:.0%}, North-South share {ns_share:.0%} ({dominance})."
    )


def _build_clip_meta(
    movement_keys: list[str],
    ts_feature_mode: str,
    ts_image_seq_len: int,
    interval_s: int,
    window_s: int,
    image_size: int,
    periodicity: int,
    norm_const: float,
    three_channel: bool,
    learnable: bool,
    caption_style: str = "clip_caption_v2",
) -> dict:
    return {
        "schema": "clip_v1",
        "movement_keys": list(movement_keys or []),
        "ts_feature_mode": str(ts_feature_mode or ""),
        "ts_image": {
            "seq_len": int(ts_image_seq_len or 0),
            "interval_s": int(interval_s or 0),
            "window_s": int(window_s or 0),
            "size": int(image_size or 0),
            "periodicity": int(periodicity or 0),
            "norm_const": float(norm_const or 0.0),
            "three_channel": bool(three_channel),
            "learnable": bool(learnable),
        },
        "caption_style": str(caption_style or "clip_caption_v1"),
    }


def _shared_maybe_take_snapshot(instance, current_time_s):
    import os
    import uuid

    snapshot_path = None
    interval = int(getattr(instance, "traffic_series_interval_s", 2) or 2)
    if interval <= 0:
        interval = 2
    current_bin = int(current_time_s) // interval
    last_written = getattr(instance, "_traffic_series_last_written_bin", -1)

    if current_bin > last_written:
        try:
            ensure = getattr(instance, "_ensure_traffic_series_config", None)
            if callable(ensure):
                ensure()

            output_path = getattr(instance, "_traffic_series_output_path", "") or ""
            output_dir = os.path.dirname(output_path)
            if not output_dir:
                dic_path = getattr(instance, "dic_path", {}) or {}
                output_dir = dic_path.get("PATH_TO_WORK_DIRECTORY", "") or "."

            snapshot_dir = os.path.join(output_dir, "snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)

            # 添加随机数/随机标识，避免同名覆盖
            rand_tag = uuid.uuid4().hex[:8]
            snapshot_name = f"snapshot_t{int(current_time_s)}_b{current_bin}_{rand_tag}.json"
            snapshot_path = os.path.join(snapshot_dir, snapshot_name)

            env = getattr(instance, "env", None)
            if env is None or not hasattr(env, "eng"):
                raise RuntimeError("CityFlow env is not initialized.")

            archive = env.eng.snapshot()
            archive.dump(snapshot_path)
            env.eng.load_from_file(snapshot_path)
            print(f"Snapshot success copy and loaded")
        except Exception as exc:    
            print(f"Snapshot failed at t={current_time_s}: {exc}")
            snapshot_path = None

    return snapshot_path


class LLM_Inference_VLLM:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow, ts_image_adapter=None):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self._resolved_image_token = None
        self.ts_image_client = None
        self.ts_image_adapter = ts_image_adapter
        self.ts_history = None
        self.ts_feature_dim = None
        self.ts_cells_len = None
        self._cityflow_env_args_dir = None
  
        try:
            self.ts_image_seq_len = int(self.dic_agent_conf.get("TS_IMAGE_SEQ_LEN", 0) or 0)
        except (TypeError, ValueError):
            self.ts_image_seq_len = 0
        if self.ts_image_adapter is not None and self.ts_image_seq_len <= 0:
            adapter_seq_len = getattr(self.ts_image_adapter, "seq_len", None)
            if adapter_seq_len is not None:
                self.ts_image_seq_len = int(adapter_seq_len)
        self.ts_image_token = str(self.dic_agent_conf.get("TS_IMAGE_TOKEN", "<image>"))
        self.ts_image_prompt_prefix = str(self.dic_agent_conf.get("TS_IMAGE_PROMPT_PREFIX", ""))
        self.ts_feature_mode = str(self.dic_agent_conf.get("TS_FEATURE_MODE", "movement_total")).strip().lower()
        try:
            self.ts_image_timeout = float(self.dic_agent_conf.get("TS_IMAGE_TIMEOUT", 5.0))
        except (TypeError, ValueError):
            self.ts_image_timeout = 5.0
        try:
            self.ts_image_size = int(self.dic_agent_conf.get("TS_IMAGE_SIZE", 56) or 56)
        except (TypeError, ValueError):
            self.ts_image_size = 56
        try:
            self.ts_image_periodicity = int(self.dic_agent_conf.get("TS_IMAGE_PERIODICITY", 24) or 24)
        except (TypeError, ValueError):
            self.ts_image_periodicity = 24
        self.ts_image_norm_const = float(self.dic_agent_conf.get("TS_IMAGE_NORM_CONST", 0.4) or 0.4)

        self.ts_image_norm_const = 0.4
        self.ts_image_learnable = _str2bool(self.dic_agent_conf.get("TS_IMAGE_LEARNABLE", False))
        self.ts_image_three_channel = _str2bool(self.dic_agent_conf.get("TS_IMAGE_THREE_CHANNEL", True))
        self._resolved_image_token = None
        self.save_cityflow_env_args = _str2bool(
            self.dic_agent_conf.get("SAVE_CITYFLOW_ENV_ARGS", True)
        )

        self._traffic_history_interval = None
        self._traffic_history_intersection_ids = None
        self._traffic_history_intersection_index = None
        self._traffic_history_movement_keys = None
        self._traffic_history_output_path = None
        self._traffic_history_last_written_bin = -1
        self._traffic_history_base_dt = None

        try:
            self.traffic_series_interval_s = int(self.dic_agent_conf.get("TRAFFIC_SERIES_INTERVAL", 2) or 2)
        except (TypeError, ValueError):
            self.traffic_series_interval_s = 2
        try:
            self.traffic_series_window_s = int(self.dic_agent_conf.get("TRAFFIC_SERIES_WINDOW", 30) or 30)
        except (TypeError, ValueError):
            self.traffic_series_window_s = 30
        self.traffic_series_output_dir = str(
            self.dic_agent_conf.get("TRAFFIC_SERIES_OUTPUT_DIR", "")
        ).strip()
        if not self.traffic_series_output_dir:
            self.traffic_series_output_dir = os.path.join("data", "recorded")
        self.traffic_series_basename = str(
            self.dic_agent_conf.get("TRAFFIC_SERIES_BASENAME", "traffic_prompt_series")
        ).strip() or "traffic_prompt_series"
        self._traffic_series_output_path = None
        self._traffic_series_initialized = False
        self._traffic_series_intersection_index = None
        self._traffic_series_movement_keys = None
        self._traffic_series_last_written_bin = -1
        self._traffic_series_base_dt = None
        self.clip_caption_mode = str(self.dic_agent_conf.get("CLIP_CAPTION_MODE", "traffic_summary")).strip().lower()
        self.clip_caption_language = str(self.dic_agent_conf.get("CLIP_CAPTION_LANGUAGE", "en")).strip().lower()
        self.clip_caption_prompt = str(self.dic_agent_conf.get("CLIP_CAPTION_PROMPT", "")).strip()
        try:
            self.clip_caption_max_tokens = int(self.dic_agent_conf.get("CLIP_CAPTION_MAX_TOKENS", 64) or 64)
        except (TypeError, ValueError):
            self.clip_caption_max_tokens = 64
        try:
            self.clip_caption_temperature = float(
                self.dic_agent_conf.get("CLIP_CAPTION_TEMPERATURE", 0.2) or 0.2
            )
        except (TypeError, ValueError):
            self.clip_caption_temperature = 0.2
        try:
            self.clip_caption_top_p = float(self.dic_agent_conf.get("CLIP_CAPTION_TOP_P", 0.9) or 0.9)
        except (TypeError, ValueError):
            self.clip_caption_top_p = 0.9
        try:
            self.clip_caption_batch_size = int(self.dic_agent_conf.get("CLIP_CAPTION_BATCH_SIZE", 8) or 8)
        except (TypeError, ValueError):
            self.clip_caption_batch_size = 8
 
        if not os.path.exists("./fails"):
            os.mkdir("./fails")
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        
        if self.ts_image_adapter is None:
            self._init_ts_image_adapter()
        self._ensure_traffic_series_config()
        self.initialize()

    def _infer_tensor_parallel_size(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            ids = [x.strip() for x in visible.split(",") if x.strip() != ""]
            if len(ids) == 1 and "-" in ids[0]:
                parts = ids[0].split("-")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    start = int(parts[0])
                    end = int(parts[1])
                    if end >= start:
                        return end - start + 1
            if ids:
                return len(ids)

        count = torch.cuda.device_count()
        return count if count > 0 else 1



    def _init_ts_image_adapter(self):
        if self.ts_image_seq_len <= 0:
            return
        try:
            input_dim = self.ts_feature_dim or len(location_direction_dict)
            self.ts_image_adapter = TimeSeriesImageAdapter(
                image_size=self.ts_image_size,
                seq_len=self.ts_image_seq_len,
                periodicity=self.ts_image_periodicity,
                norm_const=self.ts_image_norm_const,
                learnable_image=self.ts_image_learnable,
                three_channel_image=self.ts_image_three_channel,
                input_dim=input_dim,
            )
        except Exception as exc:
            print(f"TS image adapter init error: {exc}")
            self.ts_image_adapter = None
    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            # vllm may not support Qwen-VL; fallback to transformers loader
            
            self.qwen=True
            print("Warning: detected Qwen-VL model in LLM_Inference_VLLM.initialize_llm — falling back to transformers loader")
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )
            self.processor = AutoProcessor.from_pretrained(llm_path, padding_side="left")
            # processor as tokenizer for text-only usage
            self.tokenizer = AutoProcessor.from_pretrained(llm_path)
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass

            # set generation kwargs for transformer usage (note: calling code expects vllm usage)
            self.generation_kwargs = {
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.1,
                "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"]
            }
            self.test_generation_kwargs = {
                "min_length": -1,
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.1,
                "do_sample": True,
                "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
                "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            }
        else:
            tensor_parallel_size = self.dic_agent_conf.get("TENSOR_PARALLEL_SIZE", 0)
            try:
                tensor_parallel_size = int(tensor_parallel_size)
            except (TypeError, ValueError):
                tensor_parallel_size = 0
            if tensor_parallel_size < 1:
                tensor_parallel_size = self._infer_tensor_parallel_size()

            available_gpus = torch.cuda.device_count()
            if available_gpus > 0 and tensor_parallel_size > available_gpus:
                print(f"Warning: TENSOR_PARALLEL_SIZE={tensor_parallel_size} > visible GPUs={available_gpus}; "
                      f"clamping to {available_gpus}.")
                tensor_parallel_size = available_gpus
            self.llm_model = vllm.LLM(
                model=llm_path,
                tokenizer=llm_path,
                dtype=torch.bfloat16,            # 先排除 bf16 兼容性
                tensor_parallel_size=4,         # 先排除 TP/NCCL
                max_model_len=2048,             # 先降低上下文
                max_num_seqs=4,                 # 先降低并发
                max_num_batched_tokens=2048,
                gpu_memory_utilization=0.9,    # 给 KV cache 足够空间
                swap_space=16,                   # 先排除 CPU offload 相关变量
            )
            self.processor = AutoProcessor.from_pretrained(llm_path, padding_side="left")

            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left",
                padding=True
            )
            self.tokenizer.pad_token_id = 0

            test_generation_kwargs = {
                "top_k": 50,
                "top_p": 1.0,
                "temperature": 0.1,
                "max_tokens": 2048 + self.dic_agent_conf["NEW_MAX_TOKENS"]
            }
            self.generation_kwargs = vllm.SamplingParams(**test_generation_kwargs)
            self.test_generation_kwargs = {}
            # Align image placeholder token with the loaded model, if possible.
            self._resolve_image_token()

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)
        if self.save_cityflow_env_args:
            self._dump_cityflow_env_args()

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def _dump_cityflow_env_args(self):
        run_dir = str(self.dic_path.get("PATH_TO_WORK_DIRECTORY", "")).strip()
        if not run_dir:
            return
        base_name = "cityflow_env_init"
        output_dir = os.path.join(run_dir, base_name)
        if os.path.exists(output_dir):
            suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(run_dir, f"{base_name}_{suffix}")
        os.makedirs(output_dir, exist_ok=True)

        env_args = {
            "path_to_log": self.dic_path.get("PATH_TO_WORK_DIRECTORY"),
            "path_to_work_directory": self.dic_path.get("PATH_TO_WORK_DIRECTORY"),
            "dic_traffic_env_conf": self.dic_traffic_env_conf,
            "dic_path": self.dic_path,
        }
        payload = {
            "saved_at": datetime.now().isoformat(),
            "cityflow_env_args": env_args,
        }
        dump_json(_json_safe(payload), os.path.join(output_dir, "cityflow_env_args.json"), indent=2)
        self._cityflow_env_args_dir = output_dir

    def _traffic_count_suffix(self, interval_seconds):
        if interval_seconds == 60:
            return "minute"
        if interval_seconds == 3600:
            return "hour"
        return f"{interval_seconds}s"

    def _resolve_traffic_history_output_path(self, interval_seconds):
        path = str(self.dic_traffic_env_conf.get("TRAFFIC_COUNT_HISTORY_PATH", "")).strip()
        if path:
            return path
        output_dir = self.dic_traffic_env_conf.get(
            "TRAFFIC_COUNT_OUTPUT_DIR", self.dic_path["PATH_TO_WORK_DIRECTORY"]
        )
        base_name = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASENAME", "traffic_counts")
        suffix = self._traffic_count_suffix(interval_seconds)
        return os.path.join(output_dir, f"{base_name}_{suffix}.jsonl")

    def _load_traffic_history(self):
        if self._traffic_history_interval is not None:
            return
        if self.env is None:
            return
        interval = int(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 1) or 1)
        current_time = self.env.get_current_time()
        needs_reinit = (
            not getattr(self.env, "traffic_count_enabled", False)
            or getattr(self.env, "traffic_count_mode", "") != "intersection_movement"
            or interval not in getattr(self.env, "traffic_count_intervals", [])
        )
        if needs_reinit and current_time in (None, 0, 0.0):
            self.dic_traffic_env_conf["ENABLE_TRAFFIC_COUNT"] = True
            self.dic_traffic_env_conf["TRAFFIC_COUNT_MODE"] = "intersection_movement"
            self.dic_traffic_env_conf["TRAFFIC_COUNT_OUTPUT_FORMAT"] = "jsonl"
            intervals = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_INTERVALS", None)
            if intervals is None:
                intervals = [interval]
            elif isinstance(intervals, int):
                intervals = [intervals]
            intervals = [int(x) for x in intervals if int(x) > 0]
            if interval not in intervals:
                intervals.append(interval)
            self.dic_traffic_env_conf["TRAFFIC_COUNT_INTERVALS"] = sorted(set(intervals))
            try:
                self.env._init_traffic_counter()
            except Exception as exc:
                print(f"Traffic counter init failed: {exc}")
        elif needs_reinit:
            print("Traffic counter not configured for MIN_ACTION_TIME; history may be empty.")

        if not getattr(self.env, "traffic_count_enabled", False):
            return
        if getattr(self.env, "traffic_count_mode", "") != "intersection_movement":
            return

        intervals = getattr(self.env, "traffic_count_intervals", [])
        if interval in intervals:
            self._traffic_history_interval = interval
        elif intervals:
            self._traffic_history_interval = min(intervals)
            print(
                f"Using traffic interval {self._traffic_history_interval} "
                f"(MIN_ACTION_TIME={interval} not available)."
            )
        else:
            self._traffic_history_interval = interval

        self._traffic_history_intersection_ids = list(getattr(self.env, "traffic_intersection_ids", []) or [])
        self._traffic_history_intersection_index = {
            inter_id: idx for idx, inter_id in enumerate(self._traffic_history_intersection_ids)
        }
        self._traffic_history_movement_keys = list(
            getattr(self.env, "traffic_movement_keys", []) or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        )
        self._traffic_history_last_written_bin = -1
        self._traffic_history_base_dt = getattr(self.env, "traffic_base_dt", None) or datetime(1970, 1, 1)
        output_path = self._resolve_traffic_history_output_path(self._traffic_history_interval)
        self._traffic_history_output_path = output_path
        if output_path:
            _prepare_jsonl_file(output_path)

    def _get_traffic_history_until(self, intersection_id, current_time_s):
        self._load_traffic_history()
        interval = self._traffic_history_interval
        if interval is None:
            return []
        counts = self.env.traffic_counts_by_interval.get(interval)
        if counts is None or counts.size == 0:
            return []
        inter_idx = self._traffic_history_intersection_index.get(intersection_id)
        if inter_idx is None:
            return []
        num_mv = len(self._traffic_history_movement_keys)
        current_bin = int(current_time_s) // interval
        if current_bin < 0:
            return []
        if current_bin >= counts.shape[0]:
            current_bin = counts.shape[0] - 1
        offset = inter_idx * num_mv
        slice_rows = counts[: current_bin + 1, offset : offset + num_mv]
        history = []
        for row in slice_rows:
            movement_counts = {
                key: int(row[idx]) for idx, key in enumerate(self._traffic_history_movement_keys)
            }
            history.append(movement_counts)
        return history

    
        self._load_traffic_history()
        interval = self._traffic_history_interval
        if interval is None or not self._traffic_history_output_path:
            return
        counts = self.env.traffic_counts_by_interval.get(interval)
        if counts is None or counts.size == 0:
            return
        current_bin = int(current_time_s) // interval
        if current_bin < 0:
            return
        if current_bin >= counts.shape[0]:
            current_bin = counts.shape[0] - 1
        if current_bin <= self._traffic_history_last_written_bin:
            return
        base_dt = self._traffic_history_base_dt or datetime(1970, 1, 1)
        num_mv = len(self._traffic_history_movement_keys)
        records = []
        for bin_idx in range(self._traffic_history_last_written_bin + 1, current_bin + 1):
            date_str = (base_dt + timedelta(seconds=bin_idx * interval)).strftime("%Y-%m-%d %H:%M:%S")
            row = counts[bin_idx, :]
            for inter_idx, inter_id in enumerate(self._traffic_history_intersection_ids or []):
                offset = inter_idx * num_mv
                movement_counts = {
                    key: int(row[offset + mv_i]) for mv_i, key in enumerate(self._traffic_history_movement_keys)
                }
                record = {
                    "date": date_str,
                    "interval_s": int(interval),
                    "intersection_id": inter_id,
                    "movement_counts": movement_counts,
                }
                records.append(record)
        _append_jsonl_records(self._traffic_history_output_path, records)
        self._traffic_history_last_written_bin = current_bin

    def _traffic_series_run_tag(self):
        run_dir = str(self.dic_path.get("PATH_TO_WORK_DIRECTORY", "")).strip()
        run_tag = os.path.basename(run_dir) if run_dir else ""
        if not run_tag:
            run_tag = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
        return run_tag

    def _ensure_traffic_series_config(self):
        if self._traffic_series_initialized:
            return
        base_dir = self.traffic_series_output_dir or os.path.join("data", "recorded")
        run_tag = self._traffic_series_run_tag()
        output_dir = os.path.join(base_dir, run_tag)
        os.makedirs(output_dir, exist_ok=True)
        suffix = self._traffic_count_suffix(self.traffic_series_interval_s)
        filename = f"{self.traffic_series_basename}_{suffix}.jsonl"
        self._traffic_series_output_path = os.path.join(output_dir, filename)
        _prepare_jsonl_file(self._traffic_series_output_path)
        self._traffic_series_initialized = True

    def _resolve_series_movement_keys(self):
        default_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        series_keys = list(self._traffic_series_movement_keys or [])
        if not series_keys:
            return default_keys, default_keys
        output_keys = list(series_keys)
        for key in default_keys:
            if key not in output_keys:
                output_keys.append(key)
        return series_keys, output_keys

    def _load_traffic_series(self):
        if self._traffic_series_movement_keys is not None:
            return
        if self.env is None:
            return
        self._ensure_traffic_series_config()
        interval = int(self.traffic_series_interval_s or 2)
        if interval <= 0:
            interval = 2

        if not getattr(self.env, "traffic_count_enabled", False):
            current_time = self.env.get_current_time()
            if current_time in (None, 0, 0.0):
                self.dic_traffic_env_conf["ENABLE_TRAFFIC_COUNT"] = True
                self.dic_traffic_env_conf["TRAFFIC_COUNT_MODE"] = "intersection_movement"
                self.dic_traffic_env_conf["TRAFFIC_COUNT_OUTPUT_FORMAT"] = "jsonl"

        intervals = list(getattr(self.env, "traffic_count_intervals", []) or [])
        if interval not in intervals:
            intervals.append(interval)
            self.dic_traffic_env_conf["TRAFFIC_COUNT_INTERVALS"] = sorted(set(intervals))
            current_time = self.env.get_current_time()
            if current_time in (None, 0, 0.0):
                try:
                    self.env._init_traffic_counter()
                except Exception as exc:
                    print(f"Traffic series counter init failed: {exc}")

        self._traffic_series_intersection_index = {
            inter_id: idx for idx, inter_id in enumerate(getattr(self.env, "traffic_intersection_ids", []) or [])
        }
        self._traffic_series_movement_keys = list(
            getattr(self.env, "traffic_movement_keys", []) or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        )
        self._traffic_series_base_dt = getattr(self.env, "traffic_base_dt", None) or datetime(1970, 1, 1)

    def _build_traffic_series_history(self, intersection_id, current_time_s):
        self._load_traffic_series()
        interval = int(self.traffic_series_interval_s or 2)
        if interval <= 0:
            interval = 2
        series_keys, output_keys = self._resolve_series_movement_keys()
        num_series_keys = len(series_keys)
        counts = getattr(self.env, "traffic_counts_by_interval", {}).get(interval)
        inter_idx = (self._traffic_series_intersection_index or {}).get(intersection_id)

        current_bin = int(current_time_s) // interval
        if counts is not None and counts.size > 0:
            if current_bin >= counts.shape[0]:
                current_bin = counts.shape[0] - 1
        if current_bin < 0:
            current_bin = 0

        window_bins = int(self.traffic_series_window_s // interval) if interval > 0 else 0
        start_bin = max(0, current_bin - window_bins)
        base_dt = self._traffic_series_base_dt or datetime(1970, 1, 1)
        history = []
        for bin_idx in range(start_bin, current_bin + 1):
            date_str = (base_dt + timedelta(seconds=bin_idx * interval)).strftime("%Y-%m-%d %H:%M:%S")
            movement_counts = {key: 0 for key in output_keys}
            if counts is not None and counts.size > 0 and inter_idx is not None and bin_idx >= 0:
                if bin_idx < counts.shape[0]:
                    offset = inter_idx * num_series_keys
                    row = counts[bin_idx, offset : offset + num_series_keys]
                    for mv_i, key in enumerate(series_keys):
                        if mv_i < len(row):
                            movement_counts[key] = int(row[mv_i])
            history.append({"date": date_str, "movement_counts": movement_counts})
        return history

    def _clip_caption_instruction(self):
        if self.clip_caption_prompt:
            return self.clip_caption_prompt
        lang = self.clip_caption_language or "en"
        language = "English"
        if lang not in ("en", "eng", "english"):
            language = lang
        return (
            "Rewrite the following traffic-control prompt into a concise visual description of the scene, "
            "as if captioning a single image. Focus on vehicles, directions, queues, and flow. "
            f"Write in {language}. Output one sentence with no quotes."
        )

    def _clean_clip_caption(self, text):
        if text is None:
            return ""
        text = str(text).strip()
        if not text:
            return ""
        text = text.splitlines()[0].strip()
        for prefix in ("caption:", "description:", "answer:", "output:", "response:"):
            if text.lower().startswith(prefix):
                text = text[len(prefix):].strip()
                break
        return " ".join(text.split())

    def _clip_caption_instruction_series(self):
        if self.clip_caption_prompt:
            return self.clip_caption_prompt
        lang = self.clip_caption_language or "en"
        language = "English"
        if lang not in ("en", "eng", "english"):
            language = lang
        return (
            "Convert the following structured traffic time-series summary into a concise visual description "
            "of the scene, as if captioning a single image. Focus on vehicle volume, dominant directions, "
            "queues, and flow. Do not mention data, time series, or charts. "
            f"Write in {language}. Output one sentence with no quotes."
        )

    def _use_llm_clip_caption(self):
        mode = str(self.clip_caption_mode or "").strip().lower()
        return mode in ("prompt_llm", "prompt", "llm", "prompt_caption")

    def _summarize_history_for_caption(self, history, movement_keys, interval_s, window_s):
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
            for k in keys:
                try:
                    value = int(counts.get(k, 0) or 0)
                except (TypeError, ValueError):
                    value = 0
                window_counts[k] += value
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

    def _generate_clip_captions(self, caption_prompts):
        if not caption_prompts:
            return []
        if self.llm_model is None:
            return ["" for _ in caption_prompts]
        try:
            if getattr(self, "qwen", False):
                inputs = self.tokenizer(
                    caption_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048,
                )
                inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
                gen_kwargs = {
                    "max_new_tokens": self.clip_caption_max_tokens,
                    "temperature": self.clip_caption_temperature,
                    "top_p": self.clip_caption_top_p,
                    "do_sample": True,
                    "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
                    "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
                }
                with torch.no_grad():
                    output_ids = self.llm_model.generate(**inputs, **gen_kwargs)
                decoded = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                captions = []
                for prompt, text in zip(caption_prompts, decoded):
                    if text.startswith(prompt):
                        text = text[len(prompt):]
                    captions.append(self._clean_clip_caption(text))
                return captions

            sampling_params = SamplingParams(
                temperature=self.clip_caption_temperature,
                top_p=self.clip_caption_top_p,
                max_tokens=self.clip_caption_max_tokens,
            )
            captions = []
            batch_size = max(1, int(self.clip_caption_batch_size or 1))
            for start in range(0, len(caption_prompts), batch_size):
                batch = caption_prompts[start:start + batch_size]
                outputs = self.llm_model.generate(prompts=batch, sampling_params=sampling_params)
                for out in outputs:
                    text = out.outputs[0].text if out.outputs else ""
                    captions.append(self._clean_clip_caption(text))
            return captions
        except Exception as exc:
            print(f"Clip caption generation failed: {exc}")
            return ["" for _ in caption_prompts]

    def _build_clip_caption_from_prompts(self, prompts):
        if not prompts:
            return []
        if not self._use_llm_clip_caption():
            return ["" for _ in prompts]
        instruction = self._clip_caption_instruction()
        caption_prompts = [f"{instruction}\n\nPrompt:\n{prompt}\n\nCaption:" for prompt in prompts]
        return self._generate_clip_captions(caption_prompts)

    def _build_clip_caption_from_history(self, histories, movement_keys, interval_s, window_s):
        if not histories:
            return []
        if not self._use_llm_clip_caption():
            return ["" for _ in histories]
        instruction = self._clip_caption_instruction_series()
        caption_prompts = []
        for history in histories:
            summary = self._summarize_history_for_caption(history, movement_keys, interval_s, window_s)
            caption_prompts.append(f"{instruction}\n\nTraffic summary:\n{summary}\n\nCaption:")
        return self._generate_clip_captions(caption_prompts)

    def _record_traffic_prompt_series(
        self,
        current_time_s,
        prompts,
        expert_actions=None,
        snapshot_paths=None,
        candidate_action_waits=None,
        candidate_action_metrics=None,
        expert_action_margins=None,
        base_actions=None,
    ):
        if not prompts:
            return
        self._load_traffic_series()
        if not self._traffic_series_output_path:
            return
        interval = int(self.traffic_series_interval_s or 2)
        if interval <= 0:
            interval = 2
        current_bin = int(current_time_s) // interval
        if current_bin <= self._traffic_series_last_written_bin:
            return

        if isinstance(snapshot_paths, str):
            snapshot_paths = [snapshot_paths] * len(prompts)
        if snapshot_paths is None:
            snapshot_paths = ["" for _ in range(len(prompts))]
        if expert_actions is None:
            expert_actions = ["" for _ in range(len(prompts))]
        if candidate_action_waits is None:
            candidate_action_waits = [{} for _ in range(len(prompts))]
        if candidate_action_metrics is None:
            candidate_action_metrics = [{} for _ in range(len(prompts))]
        if expert_action_margins is None:
            expert_action_margins = [None for _ in range(len(prompts))]
        if base_actions is None:
            base_actions = ["" for _ in range(len(prompts))]

        _, output_keys = self._resolve_series_movement_keys()
        meta_payload = _build_clip_meta(
            movement_keys=output_keys,
            ts_feature_mode=self.ts_feature_mode,
            ts_image_seq_len=self.ts_image_seq_len,
            interval_s=interval,
            window_s=self.traffic_series_window_s,
            image_size=self.ts_image_size,
            periodicity=self.ts_image_periodicity,
            norm_const=self.ts_image_norm_const,
            three_channel=self.ts_image_three_channel,
            learnable=self.ts_image_learnable,
        )

        histories = []
        for idx in range(len(prompts)):
            intersection_id = self.env.list_intersection[idx].inter_name
            histories.append(self._build_traffic_series_history(intersection_id, current_time_s))

        prompt_captions = self._build_clip_caption_from_history(
            histories,
            output_keys,
            interval,
            self.traffic_series_window_s,
        )

        records = []
        for idx, prompt in enumerate(prompts):
            intersection_id = self.env.list_intersection[idx].inter_name
            history = histories[idx] if idx < len(histories) else []
            clip_caption = ""
            if prompt_captions is not None and idx < len(prompt_captions):
                clip_caption = str(prompt_captions[idx] or "")
            if not clip_caption:
                clip_caption = _build_clip_caption(history, output_keys)
            expert_action = (
                expert_actions[idx] if idx < len(expert_actions) and expert_actions[idx] is not None else ""
            )
            snapshot_path = (
                snapshot_paths[idx] if idx < len(snapshot_paths) and snapshot_paths[idx] is not None else ""
            )
            action_waits = (
                candidate_action_waits[idx]
                if idx < len(candidate_action_waits) and isinstance(candidate_action_waits[idx], dict)
                else {}
            )
            action_metrics = (
                candidate_action_metrics[idx]
                if idx < len(candidate_action_metrics) and isinstance(candidate_action_metrics[idx], dict)
                else {}
            )
            action_margin = (
                expert_action_margins[idx]
                if idx < len(expert_action_margins)
                else None
            )
            base_action = (
                base_actions[idx] if idx < len(base_actions) and base_actions[idx] is not None else ""
            )
            for hist_idx, entry in enumerate(history):
                record = {
                    "date": entry.get("date"),
                    "interval_s": int(interval),
                    "intersection_id": intersection_id,
                    "movement_counts": entry.get("movement_counts", {}),
                    "prompt": prompt if hist_idx == len(history) - 1 else "",
                }
                if hist_idx == len(history) - 1:
                    record["clip_caption"] = clip_caption
                    record["expert_action"] = str(expert_action or "")
                    record["snapshot_path"] = str(snapshot_path or "")
                    if action_waits:
                        record["candidate_action_waits"] = {
                            str(key): float(value) for key, value in action_waits.items()
                        }
                    if action_metrics:
                        record["candidate_action_metrics"] = action_metrics
                    if action_margin is not None:
                        record["expert_action_margin"] = float(action_margin)
                    if base_action:
                        record["base_action"] = str(base_action)
                    record["meta"] = meta_payload
                records.append(record)
        _append_jsonl_records(self._traffic_series_output_path, records)
        self._traffic_series_last_written_bin = current_bin

    def _token_in_vocab(self, token):
        return _shared_token_in_vocab(self.tokenizer, token)

    def _extract_image_token_candidates(self, llm_path):
        return _shared_extract_image_token_candidates(llm_path, self.tokenizer)

    def _resolve_image_token(self):
        return _shared_resolve_image_token(self)

    def _inject_image_token(self, prompt_text):
        prefix = self.ts_image_prompt_prefix or ""
        token = self._resolve_image_token() if (self.ts_image_token or self.tokenizer) else (self.ts_image_token or "")
        if not token and not prefix:
            return prompt_text
        if token and token in prompt_text:
            return prompt_text
        return f"{prefix}{token}\n{prompt_text}"

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]
        self._load_traffic_history()

        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)

                state_action_log[i].append(
                    {
                        "state": statistic_state,
                        "state_incoming": statistic_state_incoming,
                        "approaching_speed": mean_speed,
                    }
                )
                current_states.append(statistic_state)

            prompts = []
            ts_images = []
            snapshot_paths = []

            snapshot_path = _shared_maybe_take_snapshot(self, current_time)

            movement_keys = self._traffic_history_movement_keys or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
            ts_adapter = self.ts_image_adapter
            if ts_adapter is None:
                raise RuntimeError(
                    "ts_image_adapter is not initialized; create it in run_open_LLM_with_vllm.py "
                    "and pass it to LLM_Inference_VLLM."
                )
            
            inferred_dim = None
            for idx, s in enumerate(current_states):
                state_action_log[idx].append({"state": s})
                intersection_id = self.env.list_intersection[idx].inter_name
                traffic_history = self._get_traffic_history_until(intersection_id, current_time)

                if not traffic_history:
                    raise ValueError(f"No traffic history for intersection {intersection_id} at t={current_time}.")

                ts_seq = [[float(step.get(key, 0.0)) for key in movement_keys] for step in traffic_history]

                try:
                    current_dim = len(ts_seq[0])
                except (TypeError, IndexError):
                    current_dim = self.ts_feature_dim or 3

                if inferred_dim is None:
                    inferred_dim = current_dim
                    if self.ts_feature_dim is None:
                        self.ts_feature_dim = inferred_dim

                    if getattr(ts_adapter, "learnable_image", False):
                        adapter_dim = getattr(ts_adapter, "input_dim", None)
                        if adapter_dim is not None and adapter_dim != inferred_dim:
                            raise ValueError(
                                "TS image adapter input_dim mismatch; reinitialize in main with input_dim="
                                f"{inferred_dim}."
                            )
                elif current_dim != inferred_dim:
                    raise ValueError(
                        f"Traffic history feature dim mismatch for {intersection_id}: "
                        f"expected {inferred_dim}, got {current_dim}."
                    )

                ts_images_local = ts_adapter.generate_images(ts_seq)
                if not ts_images_local:
                    raise RuntimeError(f"TS image adapter returned no images for intersection {intersection_id}.")

                ts_image = ts_images_local[0]
                ts_images.append(ts_image)

                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
                snapshot_paths.append(snapshot_path or "")

            inputs_text = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"<start_of_image> {p}"}],  # 这里 content 直接用字符串 p
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            



            # 2) vLLM 多模态输入：每条是一个 dict（prompt + multi_modal_data）
            #    ts_images: List[PIL.Image.Image]
            vllm_prompts = [
                {
                    "prompt": txt,
                    "multi_modal_data": {"image": img},  # 单图直接放 PIL.Image
                }
                for txt, img in zip(inputs_text, ts_images)
            ]

            # 3) SamplingParams：vLLM 用这个，不是 HF 的 generate kwargs
            sampling_params = self.generation_kwargs # 你的 generation_kwargs 需要是 vLLM 支持的字段

            responses = []
            bs = 16
            for start in range(0, len(inputs_text), bs):
                # batch = inputs_text[start:start + bs]
                batch = vllm_prompts[start:start + bs]

                # vLLM.generate 返回的是 RequestOutput 列表，不是 token ids
                outputs = self.llm_model.generate(prompts=batch, sampling_params=sampling_params)

                # 取每条的第一条 beam/候选文本
                for out in outputs:
                    responses.append(out.outputs[0].text)

                    

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            expert_actions = []
            for i, res in enumerate(responses):
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                raw_signal = signals[-1] if len(signals) > 0 else ""
                is_valid = raw_signal in four_phase_list
                signal_text = raw_signal if is_valid else "ERROR"
                expert_actions.append(signal_text)
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or not is_valid:
                    signal_text = "ERROR"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                state_action_log[i][-1]["response"] = res
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            self._record_traffic_prompt_series(
                current_time,
                prompts,
                expert_actions=expert_actions,
                snapshot_paths=snapshot_paths,
            )

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "test_reward_over": total_reward,
            "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "test_avg_travel_time_over": total_travel_time}
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results

    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        results = self.test(logger, 0)
        wandb.finish()
        return results

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards



def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


class LLM_Inference:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow, ts_image_adapter=None):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None

        self.ts_image_adapter = ts_image_adapter
        self.ts_history = None
        self.ts_feature_dim = None
        self.ts_cells_len = None
        self.ts_image_service_url = str(self.dic_agent_conf.get("TS_IMAGE_SERVICE_URL", "")).strip()
        try:
            self.ts_image_seq_len = int(self.dic_agent_conf.get("TS_IMAGE_SEQ_LEN", 0) or 0)
        except (TypeError, ValueError):
            self.ts_image_seq_len = 0
        if self.ts_image_adapter is not None and self.ts_image_seq_len <= 0:
            adapter_seq_len = getattr(self.ts_image_adapter, "seq_len", None)
            if adapter_seq_len is not None:
                self.ts_image_seq_len = int(adapter_seq_len)
        self.ts_image_token = str(self.dic_agent_conf.get("TS_IMAGE_TOKEN", "<image>"))
        self.ts_image_prompt_prefix = str(self.dic_agent_conf.get("TS_IMAGE_PROMPT_PREFIX", ""))
        self.ts_feature_mode = str(self.dic_agent_conf.get("TS_FEATURE_MODE", "movement_total")).strip().lower()
        try:
            self.ts_image_timeout = float(self.dic_agent_conf.get("TS_IMAGE_TIMEOUT", 5.0))
        except (TypeError, ValueError):
            self.ts_image_timeout = 5.0
        try:
            self.ts_image_size = int(self.dic_agent_conf.get("TS_IMAGE_SIZE", 56) or 56)
        except (TypeError, ValueError):
            self.ts_image_size = 56 
        try:
            self.ts_image_periodicity = int(self.dic_agent_conf.get("TS_IMAGE_PERIODICITY", 24) or 24)
        except (TypeError, ValueError):
            self.ts_image_periodicity = 24
        try:
            self.ts_image_norm_const = float(self.dic_agent_conf.get("TS_IMAGE_NORM_CONST", 0.4) or 0.4)
        except (TypeError, ValueError):
            return
        self.ts_image_norm_const = 0.4
        self.ts_image_learnable = _str2bool(self.dic_agent_conf.get("TS_IMAGE_LEARNABLE", False))
        self.ts_image_three_channel = _str2bool(self.dic_agent_conf.get("TS_IMAGE_THREE_CHANNEL", True))

        self._traffic_history_interval = None
        self._traffic_history_intersection_ids = None
        self._traffic_history_intersection_index = None
        self._traffic_history_movement_keys = None
        self._traffic_history_output_path = None
        self._traffic_history_last_written_bin = -1
        self._traffic_history_base_dt = None

        try:
            self.traffic_series_interval_s = int(self.dic_agent_conf.get("TRAFFIC_SERIES_INTERVAL", 2) or 2)
        except (TypeError, ValueError):
            self.traffic_series_interval_s = 2
        try:
            self.traffic_series_window_s = int(self.dic_agent_conf.get("TRAFFIC_SERIES_WINDOW", 30) or 30)
        except (TypeError, ValueError):
            self.traffic_series_window_s = 30
        self.traffic_series_output_dir = str(
            self.dic_agent_conf.get("TRAFFIC_SERIES_OUTPUT_DIR", "")
        ).strip()
        if not self.traffic_series_output_dir:
            self.traffic_series_output_dir = os.path.join("data", "recorded")
        self.traffic_series_basename = str(
            self.dic_agent_conf.get("TRAFFIC_SERIES_BASENAME", "traffic_prompt_series")
        ).strip() or "traffic_prompt_series"
        self._traffic_series_output_path = None
        self._traffic_series_initialized = False
        self._traffic_series_intersection_index = None
        self._traffic_series_movement_keys = None
        self._traffic_series_last_written_bin = -1
        self._traffic_series_base_dt = None

        if not os.path.exists("./fails"):
            os.mkdir("./fails")
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self._init_ts_image_client()
        if self.ts_image_adapter is None:
            self._init_ts_image_adapter()
        self._ensure_traffic_series_config()
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            self.qwen=True
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # processor as tokenizer for text-only usage
            self.processor = AutoProcessor.from_pretrained(llm_path, padding_side="left")
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left",
                padding=True
            )
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
                trust_remote_code=True
            )

            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left",
                padding=True
            )
            self.tokenizer.pad_token_id = 0

            self.processor = AutoProcessor.from_pretrained(llm_path, padding_side="left")

        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None)
        }
        self._resolve_image_token()

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def _init_ts_image_client(self):
        if not self.ts_image_service_url or self.ts_image_seq_len <= 0:
            return


    def _init_ts_image_adapter(self):
        if self.ts_image_seq_len <= 0:
            return
        try:
            input_dim = self.ts_feature_dim or len(location_direction_dict)
            self.ts_image_adapter = TimeSeriesImageAdapter(
                image_size=self.ts_image_size,
                seq_len=self.ts_image_seq_len,
                periodicity=self.ts_image_periodicity,
                norm_const=self.ts_image_norm_const,
                learnable_image=self.ts_image_learnable,
                three_channel_image=self.ts_image_three_channel,
                input_dim=input_dim,
            )
        except Exception as exc:
            print(f"TS image adapter init error: {exc}")
            self.ts_image_adapter = None

    def _ensure_ts_history(self, num_intersections):
        if self.ts_history is not None and len(self.ts_history) == num_intersections:
            return
        self.ts_history = [deque(maxlen=self.ts_image_seq_len) for _ in range(num_intersections)]

    def _traffic_series_run_tag(self):
        run_dir = str(self.dic_path.get("PATH_TO_WORK_DIRECTORY", "")).strip()
        run_tag = os.path.basename(run_dir) if run_dir else ""
        if not run_tag:
            run_tag = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))
        return run_tag

    def _ensure_traffic_series_config(self):
        if self._traffic_series_initialized:
            return
        base_dir = self.traffic_series_output_dir or os.path.join("data", "recorded")
        run_tag = self._traffic_series_run_tag()
        output_dir = os.path.join(base_dir, run_tag)
        os.makedirs(output_dir, exist_ok=True)
        suffix = self._traffic_count_suffix(self.traffic_series_interval_s)
        filename = f"{self.traffic_series_basename}_{suffix}.jsonl"
        self._traffic_series_output_path = os.path.join(output_dir, filename)
        _prepare_jsonl_file(self._traffic_series_output_path)
        self._traffic_series_initialized = True

    def _resolve_series_movement_keys(self):
        default_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        series_keys = list(self._traffic_series_movement_keys or [])
        if not series_keys:
            return default_keys, default_keys
        output_keys = list(series_keys)
        for key in default_keys:
            if key not in output_keys:
                output_keys.append(key)
        return series_keys, output_keys

    def _load_traffic_series(self):
        if self._traffic_series_movement_keys is not None:
            return
        if self.env is None:
            return
        self._ensure_traffic_series_config()
        interval = int(self.traffic_series_interval_s or 2)
        if interval <= 0:
            interval = 2

        if not getattr(self.env, "traffic_count_enabled", False):
            current_time = self.env.get_current_time()
            if current_time in (None, 0, 0.0):
                self.dic_traffic_env_conf["ENABLE_TRAFFIC_COUNT"] = True
                self.dic_traffic_env_conf["TRAFFIC_COUNT_MODE"] = "intersection_movement"
                self.dic_traffic_env_conf["TRAFFIC_COUNT_OUTPUT_FORMAT"] = "jsonl"

        intervals = list(getattr(self.env, "traffic_count_intervals", []) or [])
        if interval not in intervals:
            intervals.append(interval)
            self.dic_traffic_env_conf["TRAFFIC_COUNT_INTERVALS"] = sorted(set(intervals))
            current_time = self.env.get_current_time()
            if current_time in (None, 0, 0.0):
                try:
                    self.env._init_traffic_counter()
                except Exception as exc:
                    print(f"Traffic series counter init failed: {exc}")

        self._traffic_series_intersection_index = {
            inter_id: idx for idx, inter_id in enumerate(getattr(self.env, "traffic_intersection_ids", []) or [])
        }
        self._traffic_series_movement_keys = list(
            getattr(self.env, "traffic_movement_keys", []) or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        )
        self._traffic_series_base_dt = getattr(self.env, "traffic_base_dt", None) or datetime(1970, 1, 1)

    def _build_traffic_series_history(self, intersection_id, current_time_s):
        self._load_traffic_series()
        interval = int(self.traffic_series_interval_s or 2)
        if interval <= 0:
            interval = 2
        series_keys, output_keys = self._resolve_series_movement_keys()
        num_series_keys = len(series_keys)
        counts = getattr(self.env, "traffic_counts_by_interval", {}).get(interval)
        inter_idx = (self._traffic_series_intersection_index or {}).get(intersection_id)

        current_bin = int(current_time_s) // interval
        if counts is not None and counts.size > 0:
            if current_bin >= counts.shape[0]:
                current_bin = counts.shape[0] - 1
        if current_bin < 0:
            current_bin = 0

        window_bins = int(self.traffic_series_window_s // interval) if interval > 0 else 0
        start_bin = max(0, current_bin - window_bins)
        base_dt = self._traffic_series_base_dt or datetime(1970, 1, 1)
        history = []
        for bin_idx in range(start_bin, current_bin + 1):
            date_str = (base_dt + timedelta(seconds=bin_idx * interval)).strftime("%Y-%m-%d %H:%M:%S")
            movement_counts = {key: 0 for key in output_keys}
            if counts is not None and counts.size > 0 and inter_idx is not None and bin_idx >= 0:
                if bin_idx < counts.shape[0]:
                    offset = inter_idx * num_series_keys
                    row = counts[bin_idx, offset : offset + num_series_keys]
                    for mv_i, key in enumerate(series_keys):
                        if mv_i < len(row):
                            movement_counts[key] = int(row[mv_i])
            history.append({"date": date_str, "movement_counts": movement_counts})
        return history

    def _record_traffic_prompt_series(
        self,
        current_time_s,
        prompts,
        expert_actions=None,
        snapshot_paths=None,
        candidate_action_waits=None,
        candidate_action_metrics=None,
        expert_action_margins=None,
        base_actions=None,
    ):
        if not prompts:
            return
        self._load_traffic_series()
        if not self._traffic_series_output_path:
            return
        interval = int(self.traffic_series_interval_s or 2)
        if interval <= 0:
            interval = 2
        current_bin = int(current_time_s) // interval
        if current_bin <= self._traffic_series_last_written_bin:
            return

        if isinstance(snapshot_paths, str):
            snapshot_paths = [snapshot_paths] * len(prompts)
        if snapshot_paths is None:
            snapshot_paths = ["" for _ in range(len(prompts))]
        if expert_actions is None:
            expert_actions = ["" for _ in range(len(prompts))]
        if candidate_action_waits is None:
            candidate_action_waits = [{} for _ in range(len(prompts))]
        if candidate_action_metrics is None:
            candidate_action_metrics = [{} for _ in range(len(prompts))]
        if expert_action_margins is None:
            expert_action_margins = [None for _ in range(len(prompts))]
        if base_actions is None:
            base_actions = ["" for _ in range(len(prompts))]

        _, output_keys = self._resolve_series_movement_keys()
        meta_payload = _build_clip_meta(
            movement_keys=output_keys,
            ts_feature_mode=self.ts_feature_mode,
            ts_image_seq_len=self.ts_image_seq_len,
            interval_s=interval,
            window_s=self.traffic_series_window_s,
            image_size=self.ts_image_size,
            periodicity=self.ts_image_periodicity,
            norm_const=self.ts_image_norm_const,
            three_channel=self.ts_image_three_channel,
            learnable=self.ts_image_learnable,
        )

        records = []
        for idx, prompt in enumerate(prompts):
            intersection_id = self.env.list_intersection[idx].inter_name
            history = self._build_traffic_series_history(intersection_id, current_time_s)
            clip_caption = _build_clip_caption(history, output_keys)
            expert_action = (
                expert_actions[idx] if idx < len(expert_actions) and expert_actions[idx] is not None else ""
            )
            snapshot_path = (
                snapshot_paths[idx] if idx < len(snapshot_paths) and snapshot_paths[idx] is not None else ""
            )
            action_waits = (
                candidate_action_waits[idx]
                if idx < len(candidate_action_waits) and isinstance(candidate_action_waits[idx], dict)
                else {}
            )
            action_metrics = (
                candidate_action_metrics[idx]
                if idx < len(candidate_action_metrics) and isinstance(candidate_action_metrics[idx], dict)
                else {}
            )
            action_margin = (
                expert_action_margins[idx]
                if idx < len(expert_action_margins)
                else None
            )
            base_action = (
                base_actions[idx] if idx < len(base_actions) and base_actions[idx] is not None else ""
            )
            for hist_idx, entry in enumerate(history):
                record = {
                    "date": entry.get("date"),
                    "interval_s": int(interval),
                    "intersection_id": intersection_id,
                    "movement_counts": entry.get("movement_counts", {}),
                    "prompt": prompt if hist_idx == len(history) - 1 else "",
                }
                if hist_idx == len(history) - 1:
                    record["clip_caption"] = clip_caption
                    record["expert_action"] = str(expert_action or "")
                    record["snapshot_path"] = str(snapshot_path or "")
                    if action_waits:
                        record["candidate_action_waits"] = {
                            str(key): float(value) for key, value in action_waits.items()
                        }
                    if action_metrics:
                        record["candidate_action_metrics"] = action_metrics
                    if action_margin is not None:
                        record["expert_action_margin"] = float(action_margin)
                    if base_action:
                        record["base_action"] = str(base_action)
                    record["meta"] = meta_payload
                records.append(record)
        _append_jsonl_records(self._traffic_series_output_path, records)
        self._traffic_series_last_written_bin = current_bin

    def _state_to_features(self, state):
        mode = self.ts_feature_mode
        if mode not in ("movement_total", "movement_queue_cells", "movement_full"):
            mode = "movement_total"
        features = []
        for key in location_direction_dict:
            entry = state.get(key, {})
            queue_len = float(entry.get("queue_len", 0.0))
            cells = list(entry.get("cells", []) or [])
            avg_wait = float(entry.get("avg_wait_time", 0.0))
            if self.ts_cells_len is None:
                self.ts_cells_len = len(cells)
            if self.ts_cells_len:
                if len(cells) < self.ts_cells_len:
                    cells = cells + [0.0] * (self.ts_cells_len - len(cells))
                else:
                    cells = cells[:self.ts_cells_len]
            if mode == "movement_total":
                features.append(queue_len + sum(cells))
            elif mode == "movement_queue_cells":
                features.append(queue_len)
                features.extend(cells)
            else:
                features.append(queue_len)
                features.append(avg_wait)
                features.extend(cells)
        return features

    def _update_ts_history(self, current_states):
        self._ensure_ts_history(len(current_states))
        for i, state in enumerate(current_states):
            features = self._state_to_features(state)
            if self.ts_feature_dim is None:
                self.ts_feature_dim = len(features)
            self.ts_history[i].append(features)

    def _build_ts_batch(self):
        batch = []
        for history in self.ts_history:
            history_list = list(history)
            if not history_list:
                if self.ts_feature_dim is None:
                    return []
                pad = [0.0] * self.ts_feature_dim
                seq = [pad] * self.ts_image_seq_len
            elif len(history_list) < self.ts_image_seq_len:
                pad = history_list[0]
                seq = [pad] * (self.ts_image_seq_len - len(history_list)) + history_list
            else:
                seq = history_list
            batch.append(seq)
        return batch

    def _traffic_count_suffix(self, interval_seconds):
        if interval_seconds == 60:
            return "minute"
        if interval_seconds == 3600:
            return "hour"
        return f"{interval_seconds}s"

    def _resolve_traffic_history_output_path(self, interval_seconds):
        path = str(self.dic_traffic_env_conf.get("TRAFFIC_COUNT_HISTORY_PATH", "")).strip()
        if path:
            return path
        output_dir = self.dic_traffic_env_conf.get(
            "TRAFFIC_COUNT_OUTPUT_DIR", self.dic_path["PATH_TO_WORK_DIRECTORY"]
        )
        base_name = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_BASENAME", "traffic_counts")
        suffix = self._traffic_count_suffix(interval_seconds)
        return os.path.join(output_dir, f"{base_name}_{suffix}.jsonl")

    def _load_traffic_history(self):
        if self._traffic_history_interval is not None:
            return
        if self.env is None:
            return
        interval = int(self.dic_traffic_env_conf.get("MIN_ACTION_TIME", 1) or 1)
        current_time = self.env.get_current_time()
        needs_reinit = (
            not getattr(self.env, "traffic_count_enabled", False)
            or getattr(self.env, "traffic_count_mode", "") != "intersection_movement"
            or interval not in getattr(self.env, "traffic_count_intervals", [])
        )
        if needs_reinit and current_time in (None, 0, 0.0):
            self.dic_traffic_env_conf["ENABLE_TRAFFIC_COUNT"] = True
            self.dic_traffic_env_conf["TRAFFIC_COUNT_MODE"] = "intersection_movement"
            self.dic_traffic_env_conf["TRAFFIC_COUNT_OUTPUT_FORMAT"] = "jsonl"
            intervals = self.dic_traffic_env_conf.get("TRAFFIC_COUNT_INTERVALS", None)
            if intervals is None:
                intervals = [interval]
            elif isinstance(intervals, int):
                intervals = [intervals]
            intervals = [int(x) for x in intervals if int(x) > 0]
            if interval not in intervals:
                intervals.append(interval)
            self.dic_traffic_env_conf["TRAFFIC_COUNT_INTERVALS"] = sorted(set(intervals))
            try:
                self.env._init_traffic_counter()
            except Exception as exc:
                print(f"Traffic counter init failed: {exc}")
        elif needs_reinit:
            print("Traffic counter not configured for MIN_ACTION_TIME; history may be empty.")

        if not getattr(self.env, "traffic_count_enabled", False):
            return
        if getattr(self.env, "traffic_count_mode", "") != "intersection_movement":
            return

        intervals = getattr(self.env, "traffic_count_intervals", [])
        if interval in intervals:
            self._traffic_history_interval = interval
        elif intervals:
            self._traffic_history_interval = min(intervals)
            print(
                f"Using traffic interval {self._traffic_history_interval} "
                f"(MIN_ACTION_TIME={interval} not available)."
            )
        else:
            self._traffic_history_interval = interval

        self._traffic_history_intersection_ids = list(getattr(self.env, "traffic_intersection_ids", []) or [])
        self._traffic_history_intersection_index = {
            inter_id: idx for idx, inter_id in enumerate(self._traffic_history_intersection_ids)
        }
        self._traffic_history_movement_keys = list(
            getattr(self.env, "traffic_movement_keys", []) or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        )
        self._traffic_history_last_written_bin = -1
        self._traffic_history_base_dt = getattr(self.env, "traffic_base_dt", None) or datetime(1970, 1, 1)
        output_path = self._resolve_traffic_history_output_path(self._traffic_history_interval)
        self._traffic_history_output_path = output_path
        if output_path:
            _prepare_jsonl_file(output_path)

    def _get_traffic_history_until(self, intersection_id, current_time_s):
        self._load_traffic_history()
        interval = self._traffic_history_interval
        if interval is None:
            return []
        counts = self.env.traffic_counts_by_interval.get(interval)
        if counts is None or counts.size == 0:
            return []
        inter_idx = self._traffic_history_intersection_index.get(intersection_id)
        if inter_idx is None:
            return []
        num_mv = len(self._traffic_history_movement_keys)
        current_bin = int(current_time_s) // interval
        if current_bin < 0:
            return []
        if current_bin >= counts.shape[0]:
            current_bin = counts.shape[0] - 1
        offset = inter_idx * num_mv
        slice_rows = counts[: current_bin + 1, offset : offset + num_mv]
        history = []
        for row in slice_rows:
            movement_counts = {
                key: int(row[idx]) for idx, key in enumerate(self._traffic_history_movement_keys)
            }
            history.append(movement_counts)
        return history

   
    def _token_in_vocab(self, token):
        return _shared_token_in_vocab(self.tokenizer, token)

    def _extract_image_token_candidates(self, llm_path):
        return _shared_extract_image_token_candidates(llm_path, self.tokenizer)

    def _resolve_image_token(self):
        return _shared_resolve_image_token(self)

    def _inject_image_token(self, prompt_text):
        prefix = self.ts_image_prompt_prefix or ""
        token = self._resolve_image_token() if (self.ts_image_token or self.tokenizer) else (self.ts_image_token or "")
        if not token and not prefix:
            return prompt_text
        if token and token in prompt_text:
            return prompt_text
        return f"{prefix}{token}\n{prompt_text}"

    def _tensor_images_to_pil(self, images):
        if not torch.is_tensor(images):
            return None
        images = images.detach().cpu()
        if images.ndim != 4:
            return None
        pil_images = []
        for img in images:
            if img.shape[0] == 1:
                array = img.squeeze(0).numpy().astype(np.uint8)
                mode = "L"
            else:
                array = img.permute(1, 2, 0).numpy().astype(np.uint8)
                mode = "RGB"
            pil_images.append(Image.fromarray(array, mode=mode))
        return pil_images

    def _generate_ts_images_local(self, ts_batch):
        if self.ts_image_adapter is None:
            return None
        try:
            images = self.ts_image_adapter.generate(ts_batch)
        except Exception as exc:
            print(f"TS image adapter error: {exc}")
            return None
        return self._tensor_images_to_pil(images)

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]

        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []

        print("end reset")
        current_time = self.env.get_current_time()  # in seconds
        start_time = time.time()

        state_action_log = [[] for _ in range(len(state))]
        self._load_traffic_history()

        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf["MIN_ACTION_TIME"]))):
            if done or current_time >= total_run_cnt:
                break

            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)

                state_action_log[i].append(
                    {
                        "state": statistic_state,
                        "state_incoming": statistic_state_incoming,
                        "approaching_speed": mean_speed,
                    }
                )
                current_states.append(statistic_state)

            prompts = []
            snapshot_paths = []
            # ts_images = []

           
            snapshot_path = _shared_maybe_take_snapshot(self, current_time)

            movement_keys = self._traffic_history_movement_keys or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
            # ts_adapter = self.ts_image_adapter
            # if ts_adapter is None:
            #     raise RuntimeError(
            #         "ts_image_adapter is not initialized; create it in run_open_LLM.py and pass it to LLM_Inference."
            #     )

            # inferred_dim = None
            for idx, s in enumerate(current_states):
                state_action_log[idx].append({"state": s})
                intersection_id = self.env.list_intersection[idx].inter_name
                traffic_history = self._get_traffic_history_until(intersection_id, current_time)

            #     if not traffic_history:
            #         raise ValueError(f"No traffic history for intersection {intersection_id} at t={current_time}.")

            #     ts_seq = [[float(step.get(key, 0.0)) for key in movement_keys] for step in traffic_history]

            #     try:
            #         current_dim = len(ts_seq[0])
            #     except (TypeError, IndexError):
            #         current_dim = self.ts_feature_dim or 3

            #     if inferred_dim is None:
            #         inferred_dim = current_dim
            #         if self.ts_feature_dim is None:
            #             self.ts_feature_dim = inferred_dim

            #         if getattr(ts_adapter, "learnable_image", False):
            #             adapter_dim = getattr(ts_adapter, "input_dim", None)
            #             if adapter_dim is not None and adapter_dim != inferred_dim:
            #                 raise ValueError(
            #                     "TS image adapter input_dim mismatch; reinitialize in main with input_dim="
            #                     f"{inferred_dim}."
            #                 )

            #     elif current_dim != inferred_dim:
            #         raise ValueError(
            #             f"Traffic history feature dim mismatch for {intersection_id}: "
            #             f"expected {inferred_dim}, got {current_dim}."
            #         )

            #     ts_images_local = ts_adapter.generate_images(ts_seq)
                # if not ts_images_local:
                #     raise RuntimeError(f"TS image adapter returned no images for intersection {intersection_id}.")

                # ts_image = ts_images_local[0]
                # ts_images.append(ts_image)

                prompt = getPrompt(state2text(s))
                prompt = prompt[0]["content"] + "\n\n### Instruction:\n" + prompt[1]["content"] + "\n\n### Response:\n"
                prompts.append(prompt)
                snapshot_paths.append(snapshot_path or "")

            if hasattr(self, "qwen") and self.qwen is not None and self.qwen is True:
                inputs_text = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"<start_of_image> {p}"}],  # 这里 content 直接用字符串 p
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]

            else:
                inputs_text = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": f"<start_of_image> {p}"}],  # 这里 content 直接用字符串 p
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in prompts
                ]
            

            # images_batched = [[img] for img in ts_images]

            # inputs = self.processor(text=inputs_text, images=images_batched, padding=True, return_tensors="pt")
            inputs = self.processor(text=inputs_text, padding=True, return_tensors="pt")

            responses = []
            previous_flag = 0
            for i in range(len(current_states)):
                if (i + 1) % 16 == 0 or i + 1 >= len(current_states):
                    current_inputs = {k: v[previous_flag : i + 1] for k, v in inputs.items()}
                    
                    response_ids = self.llm_model.generate(**current_inputs, **self.test_generation_kwargs)
                    cur_response = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                    responses += cur_response
                    previous_flag = i + 1

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            critic_actions = []
            expert_actions = []

            for i, res in enumerate(responses):
                res = res[len(prompts[i]) :]
                signal_answer_pattern = r"<signal>(.*?)</signal>"
                signals = re.findall(signal_answer_pattern, res)
                raw_signal = signals[-1] if len(signals) > 0 else ""
                is_valid = raw_signal in four_phase_list
                signal_text = raw_signal if is_valid else "ETWT"
                expert_actions.append(signal_text)

                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)

                if len(signals) == 0 or not is_valid:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                state_action_log[i][-1]["response"] = res
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            self._record_traffic_prompt_series(
                current_time,
                prompts,
                expert_actions=expert_actions,
                snapshot_paths=snapshot_paths,
            )

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)  # my reward

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)

            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature["lane_num_waiting_vehicle_in"]))
            queue_length_episode.append(sum(queue_length_inter))

            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]["time"])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            # wandb logger
            vehicle_travel_times = {}
            for inter in self.env.list_intersection:
                arrive_left_times = inter.dic_vehicle_arrive_leave_time
                for veh in arrive_left_times:
                    if "shadow" in veh:
                        continue

                    enter_time = arrive_left_times[veh]["enter_time"]
                    leave_time = arrive_left_times[veh]["leave_time"]

                    if not np.isnan(enter_time):
                        leave_time = (
                            leave_time
                            if not np.isnan(leave_time)
                            else self.dic_traffic_env_conf["RUN_COUNTS"]
                        )
                        if veh not in vehicle_travel_times:
                            vehicle_travel_times[veh] = [leave_time - enter_time]
                        else:
                            vehicle_travel_times[veh].append(leave_time - enter_time)

            total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

            results = {
                "test_reward_over": total_reward,
                "test_avg_queue_len_over": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
                "test_queuing_vehicle_num_over": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
                "test_avg_waiting_time_over": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
                "test_avg_travel_time_over": total_travel_time,
            }
            logger.log(results)
            print("Test Round:", test_round, results)

        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)

        print("Testing time: ", time.time() - start_time)
        self.env.batch_log_2()
        return results


    def train_test(self):
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        self.test(logger, 0)
        wandb.finish()

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums

    def get_norm_reward(self, state):
        rewards = []

        for i in range(len(state)):
            vehicle_num = 0
            queue_length = 0

            intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
            roads = deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]['queue_len']

                vehicle_num += statistic_state[lane]['queue_len']
                for cell in range(len(statistic_state[lane]['cells'])):
                    vehicle_num += statistic_state[lane]['cells'][cell]

            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(reward)

        return rewards


class _LocalTSImageModel:
    def __init__(
        self,
        image_size,
        seq_len,
        periodicity,
        norm_const,
        learnable_image,
        three_channel_image,
        device=None,
    ):
        self.config = SimpleNamespace(
            image_size=int(image_size),
            seq_len=int(seq_len),
            periodicity=int(periodicity),
            norm_const=float(norm_const),
            learnable_image=bool(learnable_image),
            save_images=False,
            three_channel_image=bool(three_channel_image),
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.learnable_image_module = None
        if self.config.learnable_image:
            self.learnable_image_module = LearnableTimeSeriesToImage(
                input_dim=3,
                hidden_dim=48,
                output_channels=3 if self.config.three_channel_image else 1,
                image_size=self.config.image_size,
                periodicity=self.config.periodicity,
            ).to(self.device)
            self.learnable_image_module.eval()

    def _infer_device(self):
        if self.learnable_image_module is not None:
            try:
                return next(self.learnable_image_module.parameters()).device
            except StopIteration:
                pass
        return self.device

    def _normalize_input(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        stdev /= self.config.norm_const
        x_enc = x_enc / stdev
        return x_enc, means, stdev

    def vision_augmented_learner(self, x_enc, image_size, context_len, periodicity):
        if self.config.learnable_image:
            images = self.learnable_image_module(x_enc)
        else:
            images = time_series_to_simple_image(x_enc, image_size, context_len, periodicity)
        images = self._normalize_images(images)
        return images

    @staticmethod
    def _normalize_images(images):
        min_vals = images.reshape(images.size(0), -1).min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        max_vals = images.reshape(images.size(0), -1).max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
        scale = (max_vals - min_vals).clamp(min=1e-5)
        images = (images - min_vals) / scale
        images = (images * 255).clamp(0, 255).to(torch.uint8)
        return images

class LLM_CGPR_Collector:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.processor = None
        self.llm_model = None
        self.llm_ref_model = None
        self.critic_agents = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.data_buffer = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            # Qwen-VL specific loader (text-only usage; processor used as tokenizer)
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.qwen=True
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # use processor for text tokenization (no image handling here)
            self.tokenizer = AutoProcessor.from_pretrained(llm_path)
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                # processor may not expose pad_token_id; ignore if so
                pass
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                device_map=device_map,
            )

            # init tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                llm_path,
                padding_side="left"
            )
            self.tokenizer.pad_token_id = 0

        self.generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 1.0,
            "do_sample": False,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "num_beam_groups": 4,
            "diversity_penalty": 1.0,
            "num_beams": 4,
            "num_return_sequences": 4
        }

    def initialize_critic(self):
        round_num = 99
        traffic_file = self.dic_traffic_env_conf['TRAFFIC_FILE'].split('.')[0]

        dic_adv_colight_agent_conf_extra = {
            "MODEL_NAME": "AdvancedColight",
            "LIST_STATE_FEATURE": [
                "cur_phase",
                "traffic_movement_pressure_queue_efficient",
                "lane_enter_running_part",
                "adjacency_matrix",
            ],
            "CNN_layers": [[32, 32]],
        }
        dic_critic_agent_conf = merge(dic_adv_colight_agent_conf_extra, config.DIC_BASE_AGENT_CONF)

        if dic_critic_agent_conf["MODEL_NAME"] in self.dic_traffic_env_conf["LIST_MODEL_NEED_TO_UPDATE"]:
            dic_critic_agent_conf["EPSILON"] = 0
            dic_critic_agent_conf["MIN_EPSILON"] = 0

        critic_agents = []
        compare_dic_traffic_env_conf = deepcopy(self.dic_traffic_env_conf)
        compare_dic_traffic_env_conf["LIST_STATE_FEATURE"] = dic_critic_agent_conf["LIST_STATE_FEATURE"]
        for j in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            agent_name = dic_critic_agent_conf["MODEL_NAME"]
            agent = config.DIC_AGENTS[agent_name](
                dic_agent_conf=dic_critic_agent_conf,
                dic_traffic_env_conf=compare_dic_traffic_env_conf,
                dic_path=self.dic_path,
                cnt_round=0,
                intersection_id=str(j)
            )
            critic_agents.append(agent)

        for i in range(self.dic_traffic_env_conf['NUM_AGENTS']):
            critic_agents[i].load_network(f'round_{round_num}_inter_0', file_path=f"./model_weights/{dic_critic_agent_conf['MODEL_NAME']}/{traffic_file}/")

        self.critic_agents = critic_agents
        self.dic_critic_agent_conf = dic_critic_agent_conf

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    

    def _ensure_ts_history(self, num_intersections):
        if self.ts_history is not None and len(self.ts_history) == num_intersections:
            return
        self.ts_history = [deque(maxlen=self.ts_image_seq_len) for _ in range(num_intersections)]

    def _state_to_features(self, state):
        mode = self.ts_feature_mode
        if mode not in ("movement_total", "movement_queue_cells", "movement_full"):
            mode = "movement_total"
        features = []
        for key in location_direction_dict:
            entry = state.get(key, {})
            queue_len = float(entry.get("queue_len", 0.0))
            cells = list(entry.get("cells", []) or [])
            avg_wait = float(entry.get("avg_wait_time", 0.0))
            if self.ts_cells_len is None:
                self.ts_cells_len = len(cells)
            if self.ts_cells_len:
                if len(cells) < self.ts_cells_len:
                    cells = cells + [0.0] * (self.ts_cells_len - len(cells))
                else:
                    cells = cells[:self.ts_cells_len]
            if mode == "movement_total":
                features.append(queue_len + sum(cells))
            elif mode == "movement_queue_cells":
                features.append(queue_len)
                features.extend(cells)
            else:
                features.append(queue_len)
                features.append(avg_wait)
                features.extend(cells)
        return features

    def _update_ts_history(self, current_states):
        self._ensure_ts_history(len(current_states))
        for i, state in enumerate(current_states):
            features = self._state_to_features(state)
            if self.ts_feature_dim is None:
                self.ts_feature_dim = len(features)
            self.ts_history[i].append(features)

    def _build_ts_batch(self):
        batch = []
        for history in self.ts_history:
            history_list = list(history)
            if not history_list:
                if self.ts_feature_dim is None:
                    return []
                pad = [0.0] * self.ts_feature_dim
                seq = [pad] * self.ts_image_seq_len
            elif len(history_list) < self.ts_image_seq_len:
                pad = history_list[0]
                seq = [pad] * (self.ts_image_seq_len - len(history_list)) + history_list
            else:
                seq = history_list
            batch.append(seq)
        return batch

    def _inject_image_token(self, prompt_text):
        prefix = self.ts_image_prompt_prefix or ""
        token = self.ts_image_token or ""
        if not token and not prefix:
            return prompt_text
        return f"{prefix}{token}\n{prompt_text}"

    def _tensor_images_to_pil(self, images):
        if not torch.is_tensor(images):
            return None
        images = images.detach().cpu()
        if images.ndim != 4:
            return None
        pil_images = []
        for img in images:
            if img.shape[0] == 1:
                array = img.squeeze(0).numpy().astype(np.uint8)
                mode = "L"
            else:
                array = img.permute(1, 2, 0).numpy().astype(np.uint8)
                mode = "RGB"
            pil_images.append(Image.fromarray(array, mode=mode))
        return pil_images

    def _generate_ts_images_local(self, ts_batch):
        if self.ts_image_adapter is None:
            return None
        try:
            images = self.ts_image_adapter.generate(ts_batch)
        except Exception as exc:
            print(f"TS image adapter error: {exc}")
            return None
        return self._tensor_images_to_pil(images)
        self.initialize_critic()

    def collect(self):
        print("================ Start Training ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        state = self.env.reset()
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]

        # data buffer for training data collection
        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, return_tensors="pt", padding="longest")['input_ids'].to('cuda')

            response_ids = self.llm_model.generate(input_ids=inputs, **self.generation_kwargs)
            response_ids = response_ids.reshape(-1, 4, response_ids.size(1))
            responses = []
            for i in range(response_ids.size(0)):
                responses.append(self.tokenizer.batch_decode(response_ids[i], skip_special_tokens=True))

            rewards = []
            all_decoded_responses = []
            all_sampled_rewards = []
            critic_actions = []
            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                action_response = responses[i][random.randint(0, 3)][len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, action_response)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": action_response})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

                # critic agents
                one_state, _ = self.env.get_state(self.dic_critic_agent_conf["LIST_STATE_FEATURE"])
                critic_agent_action, q_value = self.critic_agents[i].choose_action_with_value(step_num, one_state)
                critic_actions.append(code2action(critic_agent_action[i]))
                rewards.append(q_value[i][action2code(signal_text)])

                # collect responses
                prompt_responses = []
                sampled_rewards = []
                for res_i in range(4):
                    sampled_response = res[res_i][len(prompts[i]):]
                    sampled_signals = re.findall(signal_answer_pattern, sampled_response)
                    sampled_signal_text = sampled_signals[-1] if len(sampled_signals) > 0 else "ETWT"
                    if len(sampled_signals) == 0 or sampled_signal_text not in four_phase_list:
                        sampled_rewards.append(0)
                    else:
                        sampled_rewards.append(float(q_value[i][action2code(sampled_signal_text)]))

                    prompt_responses.append(sampled_response)
                all_decoded_responses.append(prompt_responses)
                all_sampled_rewards.append(sampled_rewards)

            next_state, _, done, _ = self.env.step(action_list)

            for i, res in enumerate(responses):
                if vehicle_nums[i] > 0:
                    new_d = {"query": prompts[i],
                             "responses": all_decoded_responses[i],
                             "scores": all_sampled_rewards[i]}
                    com_score = new_d["scores"][0]
                    all_same = True
                    for s in new_d["scores"]:
                        if s != com_score:
                            all_same = False

                    if not all_same:
                        self.data_buffer.append(new_d)

            # log action
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            print("Rewards:", sum(rewards), "Fail Num:", fail_num)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

            if not os.path.exists("./data/cgpr"):
                os.makedirs("./data/cgpr")
            dump_json(self.data_buffer, f"./data/cgpr/cgpr_{self.dic_traffic_env_conf['TRAFFIC_FILE']}")
            torch_gc()

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "env/collect_reward": total_reward,
            "env/collect_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "env/collect_avg_travel_time": total_travel_time}
        print("Collect:", results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)

        print("Collection time: ", time.time() - start_time)
        self.env.batch_log_2()

        if not os.path.exists("./data/cgpr"):
            os.makedirs("./data/cgpr")
        dump_json(self.data_buffer, f"./data/cgpr/cgpr_{self.dic_traffic_env_conf['TRAFFIC_FILE']}")

    def train_test(self):
        print("================ Start Data Collection ================")
        self.collect()

    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums


class LLM_CGPR_Trainer:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, roadnet, trafficflow):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.agents = []
        self.env = None
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.models = []
        self.generation_kwargs = {}
        self.test_generation_kwargs = {}
        self.epoch_num = 0
        self.tokenizer = None
        self.llm_model = None
        self.llm_ref_model = None
        self.critic_agents = None
        self.dic_critic_agent_conf = None
        self.training_args = None
        self.trainer_built = False
        self.trainer = None
        self.device = None
        self.fail_log_file = f"./fails/{self.dic_agent_conf['LLM_MODEL']}-{self.dic_traffic_env_conf['TRAFFIC_FILE']}-{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        self.fail_logs = []
        self.initialize()

    def initialize_llm(self):
        device_map = "auto"

        # init LLM
        llm_path = self.dic_agent_conf["LLM_PATH"]
        llm_path_lower = llm_path.lower()
        if "qwen" in llm_path_lower and "vl" in llm_path_lower:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            self.qwen=True
            # load Qwen-VL (text-only usage)
            self.llm_model = Qwen3VLForConditionalGeneration.from_pretrained(
                llm_path,
                dtype="auto",
                device_map=device_map,
            )

            # processor as tokenizer for text-only
            self.tokenizer = AutoProcessor.from_pretrained(llm_path)
            try:
                self.tokenizer.pad_token_id = 0
            except Exception:
                pass
        else:
            self.llm_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.bfloat16,
                # load_in_8bit=True,
                device_map=device_map,
            )
        gradient_accumulation_steps = self.dic_agent_conf["BATCH_SIZE"] // self.dic_agent_conf["MINI_BATCH_SIZE"]
        self.training_args = TrainingArguments(output_dir=f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
                                               num_train_epochs=self.dic_agent_conf["EPOCHS"],
                                               per_device_train_batch_size=self.dic_agent_conf["MINI_BATCH_SIZE"],
                                               per_device_eval_batch_size=self.dic_agent_conf["MINI_BATCH_SIZE"],
                                               gradient_accumulation_steps=gradient_accumulation_steps,
                                               learning_rate=self.dic_agent_conf['LEARNING_RATE'],
                                               bf16=True,
                                               logging_steps=1,
                                               evaluation_strategy="steps",
                                               save_strategy="steps",
                                               eval_steps=50 if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else 10,
                                               save_steps=50 if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else 10,
                                               save_total_limit=3,
                                               load_best_model_at_end=True,
                                               model_max_length=2048)

        # init tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            padding_side="left",
            padding=True
        )
        self.tokenizer.pad_token_id = 0

        # init lora
        lora_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        # self.llm_model = prepare_model_for_kbit_training(self.llm_model)
        self.llm_model = get_peft_model(self.llm_model, lora_config)

        self.test_generation_kwargs = {
            "min_length": -1,
            "top_k": 50,
            "top_p": 1.0,
            "temperature": 0.1,
            "do_sample": True,
            "max_new_tokens": self.dic_agent_conf["NEW_MAX_TOKENS"],
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }

    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path
        )
        self.env.reset()
        self.initialize_llm()

    def train(self):
        print("================ Start Training ================")
        data = load_dataset("json", data_files=f"./data/cgpr/cgpr_{self.dic_agent_conf['CGPR_DATA_PATH']}")

        train_val = data["train"].train_test_split(test_size=0.1, shuffle=True, seed=2024)
        train_data = train_val["train"].shuffle(seed=2024)
        val_data = train_val["test"].shuffle(seed=2024)

        self.llm_model.train()
        data_module = make_supervised_data_module(self.tokenizer, train_data, val_data, mix=True if 'mix' in self.dic_agent_conf['CGPR_DATA_PATH'] else False)
        self.trainer = RankTrainer(model=self.llm_model, tokenizer=self.tokenizer, args=self.training_args, **data_module)

        # self.llm_model.config.use_cache = False
        # self.llm_model = torch.compile(self.llm_model)
        self.trainer.train()

    def test(self, logger, test_round):
        print("================ Start Test ================")
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        # initialize output streams
        done = False
        state = self.env.reset()
        total_reward = 0.0
        queue_length_episode = []
        waiting_time_episode = []
        print("end reset")
        current_time = self.env.get_current_time()  # in seconds

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]
        self._load_traffic_history()

        # self.llm_model.eval()
        for step_num in tqdm(range(int(total_run_cnt / self.dic_traffic_env_conf['MIN_ACTION_TIME']))):
            if done or current_time >= total_run_cnt:
                break
            action_list = []
            current_states = []

            for i in range(len(state)):
                # log statistic state
                intersection = self.env.intersection_dict[self.env.list_intersection[i].inter_name]
                roads = deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[i].append({"state": statistic_state, "state_incoming": statistic_state_incoming,
                                            "approaching_speed": mean_speed})
                current_states.append(statistic_state)

            prompts = []
            for s in current_states:
                prompt = getPrompt(state2text(s))
                prompt = prompt[0]['content'] + "\n\n### Instruction:\n" + prompt[1]['content'] + "\n\n### Response:\n"
                prompts.append(prompt)
            inputs = self.tokenizer(prompts, truncation=True, max_length=2048, padding=True, return_tensors='pt').to('cuda')

            # response_ids = self.llm_model.generate(input_ids=inputs["input_ids"], **self.test_generation_kwargs)
            # 构造全 0 的 token_type_ids (假设这是纯文本部分，或者模型默认处理)
            # 注意：如果是包含图片的输入，这里不能简单的全 0，必须沿用预处理时的逻辑
            current_token_type_ids = torch.zeros_like(current_input_ids)

            response_ids = self.llm_model.generate(
                input_ids=current_input_ids,
                token_type_ids=current_token_type_ids, 
                **self.test_generation_kwargs
            )
            responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            fail_num = 0
            vehicle_nums = self.get_vehicle_num(current_states)
            for i, res in enumerate(responses):
                res = res[len(prompts[i]):]
                signal_answer_pattern = r'<signal>(.*?)</signal>'
                signals = re.findall(signal_answer_pattern, res)
                signal_text = signals[-1] if len(signals) > 0 else "ETWT"
                action_list.append(action2code(signal_text) if signal_text in four_phase_list else 0)
                if len(signals) == 0 or signal_text not in four_phase_list:
                    signal_text = "ETWT"
                    if vehicle_nums[i] != 0:
                        self.fail_logs.append({"state": current_states[i], "response": res})
                        dump_json(self.fail_logs, self.fail_log_file)
                        fail_num += 1

            next_state, rewards, done, _ = self.env.step(action_list)

            # log action
            for i in range(len(state)):
                state_action_log[i][-1]["action"] = eight_phase_list[action_list[i]]

            current_time = self.env.get_current_time()  # in seconds
            state = next_state

            # calculate logger results
            total_reward += sum(rewards)
            queue_length_inter = []
            for inter in self.env.list_intersection:
                queue_length_inter.append(sum(inter.dic_feature['lane_num_waiting_vehicle_in']))
            queue_length_episode.append(sum(queue_length_inter))
            print("Fail Num:", fail_num, "Queuing Vehicles:", sum(queue_length_episode))

            # waiting time
            waiting_times = []
            for veh in self.env.waiting_vehicle_list:
                waiting_times.append(self.env.waiting_vehicle_list[veh]['time'])
            waiting_time_episode.append(np.mean(waiting_times) if len(waiting_times) > 0 else 0.0)

        # wandb logger
        vehicle_travel_times = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh in arrive_left_times:
                if "shadow" in veh:
                    continue
                enter_time = arrive_left_times[veh]["enter_time"]
                leave_time = arrive_left_times[veh]["leave_time"]
                if not np.isnan(enter_time):
                    leave_time = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    if veh not in vehicle_travel_times:
                        vehicle_travel_times[veh] = [leave_time - enter_time]
                    else:
                        vehicle_travel_times[veh].append(leave_time - enter_time)

        total_travel_time = np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])

        results = {
            "env/test_reward": total_reward,
            "env/test_avg_queue_len": np.mean(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_queuing_vehicle_num": np.sum(queue_length_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_avg_waiting_time": np.mean(waiting_time_episode) if len(queue_length_episode) > 0 else 0,
            "env/test_avg_travel_time": total_travel_time}
        logger.log(results)
        print("Test Round:", test_round, results)
        f_state_action = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, f_state_action)
        print("Testing time: ", time.time() - start_time)

        self.env.batch_log_2()

        return results

    def train_test(self):
        print("================ Start PPO Fine-Tuning ================")
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger = wandb.init(
            project=self.dic_traffic_env_conf['PROJECT_NAME'],
            group=f"{self.dic_traffic_env_conf['MODEL_NAME']}-{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases",
            name=f"{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}",
            config=all_config,
        )

        rounds = self.dic_traffic_env_conf["NUM_ROUNDS"]
        last_10_results = {"env/test_reward_over": [],
                           "env/test_avg_queue_len_over": [],
                           "env/test_queuing_vehicle_num_over": [],
                           "env/test_avg_waiting_time_over": [],
                           "env/test_avg_travel_time_over": []}
        for r in range(rounds):
            # train
            self.train()

            # test
            results = self.test(logger, r)
            for ele in last_10_results:
                last_10_results[ele].append(results[ele[:-5]])

            main_path = f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}"
            ckpt_path = f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}/my_checkpoint_{r}"
            if not os.path.isdir(main_path):
                os.mkdir(main_path)
            if not os.path.isdir(ckpt_path):
                os.mkdir(ckpt_path)

            self.llm_model.save_pretrained(
                f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}/my_checkpoint_{r}")

        logger.log(last_10_results)
        wandb.finish()

        self.llm_model.save_pretrained(f"{self.dic_agent_conf['LLM_OUTPUT_DIR']}_{self.dic_traffic_env_conf['TRAFFIC_FILE'].replace('.json', '')}")

    '''
    ======================= Class Utils =======================
    '''
    def get_vehicle_num(self, states):
        veh_nums = []

        for i in range(len(states)):
            vehicle_num = 0

            for lane in states[i]:
                vehicle_num += states[i][lane]['queue_len']
                for cell in range(len(states[i][lane]['cells'])):
                    vehicle_num += states[i][lane]['cells'][cell]

            veh_nums.append(vehicle_num)

        return veh_nums
