from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
from tqdm import tqdm

try:
    import wandb
except Exception:
    wandb = None

from projection_train.action_space import ACTION_LIST, action_to_index, format_action_tag, index_to_action
from projection_train.prompt_utils import MOVEMENT_KEYS, build_phase_queue_wait_context, build_task_prompt, serialize_movement_history, serialize_compact_movement_history
from utils import error
from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf
from utils.my_utils import (
    dump_json,
    eight_phase_list,
    getPrompt,
    get_state_detail,
    location_direction_dict,
    state2text,
)


def configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(line_buffering=True, write_through=True)
            except TypeError:
                reconfigure(line_buffering=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logger(log_path: str) -> logging.Logger:
    log_dir = os.path.dirname(log_path) or "."
    os.makedirs(log_dir, exist_ok=True)
    run_tag = os.path.basename(os.path.normpath(log_dir)) or "projection_test"
    logger = logging.getLogger(f"projection_train.online.{run_tag}")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


@dataclass
class ProjectionRuntimeConfig:
    llm_path: str
    seq_len: int
    image_size: int
    periodicity: int
    input_dim: int
    hidden_dim: int
    prefix_tokens: int
    prefix_mode: str
    max_length: int


def merge(dic_tmp: Dict[str, Any], dic_to_change: Dict[str, Any]) -> Dict[str, Any]:
    dic_result = copy.deepcopy(dic_tmp)
    dic_result.update(dic_to_change)
    return dic_result


def path_check(dic_path: Dict[str, str]) -> None:
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError(dic_path["PATH_TO_WORK_DIRECTORY"])
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])

    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError(dic_path["PATH_TO_MODEL"])
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(
    dic_path: Dict[str, str],
    dic_agent_conf: Dict[str, Any],
    dic_traffic_env_conf_local: Dict[str, Any],
    path: Optional[str] = None,
) -> None:
    output_dir = path or dic_path["PATH_TO_WORK_DIRECTORY"]
    with open(os.path.join(output_dir, "agent.conf"), "w", encoding="utf-8") as handle:
        json.dump(dic_agent_conf, handle, indent=4)
    with open(os.path.join(output_dir, "traffic_env.conf"), "w", encoding="utf-8") as handle:
        json.dump(dic_traffic_env_conf_local, handle, indent=4)


def copy_cityflow_file(
    dic_path: Dict[str, str],
    dic_traffic_env_conf_local: Dict[str, Any],
    path: Optional[str] = None,
) -> None:
    output_dir = path or dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(
        os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf_local["TRAFFIC_FILE"]),
        os.path.join(output_dir, dic_traffic_env_conf_local["TRAFFIC_FILE"]),
    )
    shutil.copy(
        os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf_local["ROADNET_FILE"]),
        os.path.join(output_dir, dic_traffic_env_conf_local["ROADNET_FILE"]),
    )


@contextmanager
def _locked_file(path: str, mode: str = "a+", encoding: str = "utf-8"):
    handle = open(path, mode, encoding=encoding)
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def _validate_jsonl_handle(handle, path: str) -> None:
    handle.seek(0)
    for line_no, line in enumerate(handle, 1):
        if not line.strip():
            raise RuntimeError(f"JSONL integrity check failed for {path}: empty line at {line_no}.")
        json.loads(line)


def _prepare_jsonl_file(path: str) -> None:
    if not path:
        return
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with _locked_file(path, "a+", encoding="utf-8") as handle:
        _validate_jsonl_handle(handle, path)


def _append_jsonl_records(path: str, records: Sequence[Dict[str, Any]]) -> None:
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

    counts: Dict[str, int] = {}
    for key in keys:
        try:
            counts[key] = int(last.get(key, 0) or 0)
        except (TypeError, ValueError):
            counts[key] = 0

    total_last = sum(counts.values())
    if total_last <= 0:
        return "No vehicles observed in the latest window."

    try:
        total_first = sum(int(first.get(key, 0) or 0) for key in keys)
    except (TypeError, ValueError):
        total_first = 0

    sorted_keys = sorted(keys, key=lambda key: counts.get(key, 0), reverse=True)
    top_keys = [key for key in sorted_keys if counts.get(key, 0) > 0][:2]
    if not top_keys:
        top_keys = sorted_keys[:1]
    top_phrase = " and ".join(f"{key}({counts.get(key, 0)})" for key in top_keys if key)
    top_sum = sum(counts.get(key, 0) for key in top_keys)
    top_share = top_sum / total_last if total_last > 0 else 0.0

    through_keys = [key for key in keys if str(key).endswith("T")]
    left_keys = [key for key in keys if str(key).endswith("L")]
    overall_avg = total_last / max(len(keys), 1)
    through_avg = (
        sum(counts.get(key, 0) for key in through_keys) / max(len(through_keys), 1)
        if through_keys
        else 0.0
    )
    left_avg = (
        sum(counts.get(key, 0) for key in left_keys) / max(len(left_keys), 1)
        if left_keys
        else 0.0
    )
    through_level = _clip_level(through_avg, overall_avg)
    left_level = _clip_level(left_avg, overall_avg)
    trend = _clip_trend_label(total_first, total_last)

    ew_sum = sum(counts.get(key, 0) for key in keys if str(key).startswith(("E", "W")))
    ns_sum = sum(counts.get(key, 0) for key in keys if str(key).startswith(("N", "S")))
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
        "caption_style": str(caption_style or "clip_caption_v2"),
    }


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_positive_int(value: Any, default: int) -> int:
    resolved = _safe_int(value, default)
    if resolved <= 0:
        return _safe_int(default, 0)
    return resolved


def _first_positive_int(*values: Any, fallback: int) -> int:
    for value in values:
        resolved = _safe_int(value, 0)
        if resolved > 0:
            return resolved
    return _safe_positive_int(fallback, 1)


def _default_cuda_device() -> torch.device:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return torch.device(f"cuda:{int(local_rank)}")
        except ValueError:
            pass
    try:
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    except Exception:
        return torch.device("cuda:0")

def _normalize_model_path(model_path: Any) -> str:
    text = str(model_path or "").strip()
    if not text:
        return ""
    trimmed = text.rstrip("/\\")
    return trimmed or text


def _load_checkpoint_args(ckpt_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}
    args = checkpoint.get("args")
    if isinstance(args, dict):
        return dict(args)
    if hasattr(args, "__dict__"):
        return dict(vars(args))
    return {}


def _resolve_device(device_text: str) -> torch.device:
    text = str(device_text or "").strip().lower()
    if not text or text == "auto":
        return _default_cuda_device() if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device(text)
    if device.type == "cuda" and device.index is None:
        return _default_cuda_device()
    return device

def _resolve_torch_dtype(args: argparse.Namespace, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    if args.fp16:
        return torch.float16
    if args.bf16:
        return torch.bfloat16
    return torch.bfloat16


def _resolve_projection_runtime(args: argparse.Namespace) -> tuple[ProjectionRuntimeConfig, str]:
    ckpt_args = _load_checkpoint_args(args.projection_ckpt)
    llm_path = _normalize_model_path(
        str(args.llm_path or "").strip()
        or str(ckpt_args.get("llm_path") or "").strip()
        or str(ckpt_args.get("model_path") or "").strip()
    )
    if not llm_path:
        raise ValueError(
            "Failed to resolve llm_path. Pass --llm_path explicitly or use a checkpoint "
            "that contains args.llm_path/model_path."
        )

    runtime = ProjectionRuntimeConfig(
        llm_path=llm_path,
        seq_len=_first_positive_int(args.seq_len, ckpt_args.get("seq_len"), fallback=16),
        image_size=_first_positive_int(args.image_size, ckpt_args.get("image_size"), fallback=224),
        periodicity=_first_positive_int(args.periodicity, ckpt_args.get("periodicity"), fallback=16),
        input_dim=_first_positive_int(
            args.input_dim,
            ckpt_args.get("input_dim"),
            fallback=len(location_direction_dict),
        ),
        hidden_dim=_first_positive_int(args.hidden_dim, ckpt_args.get("hidden_dim"), fallback=64),
        prefix_tokens=_first_positive_int(args.prefix_tokens, ckpt_args.get("prefix_tokens"), fallback=16),
        prefix_mode=str(args.prefix_mode or ckpt_args.get("prefix_mode") or "prefix"),
        max_length=_first_positive_int(args.max_length, ckpt_args.get("max_length"), fallback=128),
    )
    prompt_mode = str(args.prompt_mode or ckpt_args.get("prompt_mode") or "image_only_short")
    return runtime, prompt_mode


def _dataset_spec(dataset: str) -> tuple[int, str, list[str], str]:
    dataset = str(dataset).strip().lower()
    if dataset == "jinan":
        return (
            3600,
            "3_4",
            [
                "anon_3_4_jinan_real.json",
                "anon_3_4_jinan_real_2000.json",
                "anon_3_4_jinan_real_2500.json",
                "anon_3_4_jinan_synthetic_24000_60min.json",
                "anon_3_4_jinan_synthetic_24h_6000.json",
            ],
            "Jinan",
        )
    if dataset == "hangzhou":
        return (
            3600,
            "4_4",
            [
                "anon_4_4_hangzhou_real.json",
                "anon_4_4_hangzhou_real_5816.json",
                "anon_4_4_hangzhou_synthetic_24000_60min.json",
                "anon_4_4_hangzhou_real_t0000_1200.json",
                "anon_4_4_hangzhou_real_t1200_2400.json",
                "anon_4_4_hangzhou_real_t2400_3600.json",
            ],
            "Hangzhou",
        )
    if dataset == "newyork_16x3":
        return (3600, "16_3", ["anon_16_3_newyork_real.json"], "NewYork")
    if dataset == "newyork_28x7":
        return (
            3600,
            "28_7",
            ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"],
            "NewYork",
        )
    raise ValueError(f"Unsupported dataset: {dataset!r}")


class ProjectionPolicyRunner:
    def __init__(
        self,
        dic_agent_conf: Dict[str, Any],
        dic_traffic_env_conf_local: Dict[str, Any],
        dic_path: Dict[str, str],
        roadnet: str,
        trafficflow: str,
        runtime: ProjectionRuntimeConfig,
        projection_ckpt: str,
        prompt_mode: str,
        device: torch.device,
        torch_dtype: torch.dtype,
        use_wandb: bool,
        logger: logging.Logger | None = None,
        record_snapshots: bool = True,
        record_traffic_series: bool = True,
        dump_cityflow_env_args: bool = True,
        max_phase_override_gap: float = 0.0,
        max_phase_override_left_wait_protect: float = 0.0,
        max_phase_override_max_wait_protect: float = 0.0,
        max_phase_override_target_queue_floor: float = 0.0,
        max_phase_override_total_queue_floor: float = 0.0,
        max_phase_override_queue_weight: float = 1.0,
        max_phase_override_wait_weight: float = 0.0,
        max_phase_override_p95_wait_weight: float = 0.0,
        max_phase_override_wait_scale: float = 60.0,
        max_phase_override_p95_wait_scale: float = 300.0,
        inference_batch_size: int = 0,
        visual_ablation: str = "full",
        representation_mode: str = "ts2i",
        action_eval_chunk_size: int = 0,
        collect_states_output: str = "",
    ) -> None:
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf_local
        self.dic_path = dic_path
        self.roadnet = roadnet
        self.trafficflow = trafficflow
        self.runtime = runtime
        self.projection_ckpt = projection_ckpt
        self.prompt_mode = prompt_mode
        self.device = device
        self.torch_dtype = torch_dtype
        self.use_wandb = use_wandb
        self.logger = logger
        self.record_snapshots = bool(record_snapshots)
        self.record_traffic_series = bool(record_traffic_series)
        self.dump_cityflow_env_args = bool(dump_cityflow_env_args)
        self.max_phase_override_gap = float(max_phase_override_gap or 0.0)
        self.max_phase_override_left_wait_protect = float(max_phase_override_left_wait_protect or 0.0)
        self.max_phase_override_max_wait_protect = float(max_phase_override_max_wait_protect or 0.0)
        self.max_phase_override_target_queue_floor = float(max_phase_override_target_queue_floor or 0.0)
        self.max_phase_override_total_queue_floor = float(max_phase_override_total_queue_floor or 0.0)
        self.max_phase_override_queue_weight = float(max_phase_override_queue_weight)
        self.max_phase_override_wait_weight = float(max_phase_override_wait_weight or 0.0)
        self.max_phase_override_p95_wait_weight = float(max_phase_override_p95_wait_weight or 0.0)
        self.max_phase_override_wait_scale = max(1e-6, float(max_phase_override_wait_scale or 60.0))
        self.max_phase_override_p95_wait_scale = max(1e-6, float(max_phase_override_p95_wait_scale or 300.0))
        self.inference_batch_size = max(0, int(inference_batch_size or 0))
        self.visual_ablation = str(visual_ablation or "full").strip().lower()
        self.representation_mode = str(representation_mode or "ts2i").strip().lower()
        self.action_eval_chunk_size = max(0, int(action_eval_chunk_size))
        self.collect_states_output = str(collect_states_output or "")
        self._collect_states_fh = None
        self.env: Optional[CityFlowEnv] = None
        self.policy = None
        self.tokenizer = None
        self._movement_keys_checked = False

        self.ts_feature_mode = "movement_total"
        self.ts_image_seq_len = int(runtime.seq_len)
        self.ts_image_size = int(runtime.image_size)
        self.ts_image_periodicity = int(runtime.periodicity)
        self.ts_image_norm_const = 0.4
        self.ts_image_three_channel = True
        self.ts_image_learnable = True

        self._traffic_series_output_path = ""
        self._traffic_series_initialized = False
        self._traffic_series_intersection_index = None
        self._traffic_series_movement_keys = None
        self._traffic_series_last_written_bin = -1
        self._snapshot_last_written_bin = -1
        self._traffic_series_base_dt = None
        self.log_path = ""

        self.fail_logs: list[dict[str, Any]] = []
        os.makedirs("./fails", exist_ok=True)
        self.fail_log_file = (
            f"./fails/{self.dic_agent_conf['LLM_MODEL']}-"
            f"{self.dic_traffic_env_conf['TRAFFIC_FILE']}-"
            f"{self.dic_traffic_env_conf['ROADNET_FILE']}.json"
        )

        self.initialize()

    @staticmethod
    def _phase_queue_wait(state: Dict[str, Any]) -> tuple[dict[str, float], dict[str, float]]:
        phase_lanes = {
            "ETWT": ("ET", "WT"),
            "NTST": ("NT", "ST"),
            "ELWL": ("EL", "WL"),
            "NLSL": ("NL", "SL"),
        }
        queues: dict[str, float] = {}
        waits: dict[str, float] = {}
        for phase, lanes in phase_lanes.items():
            lane_queues = []
            lane_waits = []
            for lane in lanes:
                lane_state = state.get(lane) or {}
                lane_queues.append(float(lane_state.get("queue_len", 0.0) or 0.0))
                lane_waits.append(float(lane_state.get("avg_wait_time", 0.0) or 0.0))
            queues[phase] = float(sum(lane_queues))
            waits[phase] = float(sum(lane_waits) / len(lane_waits)) if lane_waits else 0.0
        return queues, waits

    def _phase_wait_tail(self, intersection_id: str) -> tuple[dict[str, float], dict[str, float]]:
        """Measure direct per-phase p95/max waits at the decision boundary."""
        assert self.env is not None
        roads = (self.env.intersection_dict.get(intersection_id) or {}).get("roads") or {}
        lane_vehicles = (self.env.system_states or {}).get("get_lane_vehicles") or {}
        waiting = self.env.waiting_vehicle_list or {}
        movement_waits: dict[str, list[float]] = {
            movement: [] for movement in ("ET", "WT", "NT", "ST", "EL", "WL", "NL", "SL")
        }
        direction_prefix = {"East": "E", "West": "W", "North": "N", "South": "S"}
        for road_id, road in roads.items():
            if road.get("type") != "outgoing":
                continue
            prefix = direction_prefix.get(str(road.get("location") or ""))
            if not prefix:
                continue
            for lane_kind, suffix in (("go_straight", "T"), ("turn_left", "L")):
                movement = f"{prefix}{suffix}"
                for lane_index in (road.get("lanes") or {}).get(lane_kind) or []:
                    lane_id = f"{road_id}_{lane_index}"
                    for vehicle_id in lane_vehicles.get(lane_id) or []:
                        info = waiting.get(vehicle_id)
                        if not isinstance(info, dict) or info.get("link") != lane_id:
                            continue
                        try:
                            wait_s = float(info.get("time", 0.0) or 0.0)
                        except (TypeError, ValueError):
                            continue
                        if np.isfinite(wait_s) and wait_s >= 0.0:
                            movement_waits[movement].append(wait_s)
        phase_lanes = {
            "ETWT": ("ET", "WT"), "NTST": ("NT", "ST"),
            "ELWL": ("EL", "WL"), "NLSL": ("NL", "SL"),
        }
        p95, maximum = {}, {}
        for phase, movements in phase_lanes.items():
            values = [value for movement in movements for value in movement_waits[movement]]
            p95[phase] = float(np.quantile(values, 0.95)) if values else 0.0
            maximum[phase] = float(max(values)) if values else 0.0
        return p95, maximum

    def _apply_max_phase_override(
        self,
        states: Sequence[Dict[str, Any]],
        action_indices: Sequence[int],
        predicted_actions: Sequence[str],
        phase_wait_p95s: Sequence[Dict[str, float]] | None = None,
        phase_wait_maxes: Sequence[Dict[str, float]] | None = None,
    ) -> tuple[list[int], list[str], list[dict[str, Any]]]:
        final_indices = [int(index) for index in action_indices]
        final_actions = [str(action) for action in predicted_actions]
        diagnostics: list[dict[str, Any]] = []
        if self.max_phase_override_gap <= 0.0:
            return final_indices, final_actions, diagnostics

        left_phases = {"ELWL", "NLSL"}
        through_phases = {"ETWT", "NTST"}
        wait_aware_target = (
            abs(self.max_phase_override_wait_weight) > 0.0
            or abs(self.max_phase_override_p95_wait_weight) > 0.0
        )
        for idx, state in enumerate(states):
            phase_queue, phase_wait = self._phase_queue_wait(state)
            if not phase_queue:
                continue
            phase_wait_p95 = (
                dict(phase_wait_p95s[idx])
                if phase_wait_p95s is not None and idx < len(phase_wait_p95s)
                else {}
            )
            phase_wait_max = (
                dict(phase_wait_maxes[idx])
                if phase_wait_maxes is not None and idx < len(phase_wait_maxes)
                else {}
            )
            original_action = final_actions[idx] if idx < len(final_actions) else index_to_action(final_indices[idx])
            original_queue = float(phase_queue.get(original_action, 0.0))
            total_queue = float(sum(phase_queue.values()))
            urgency_score: dict[str, float] = {}
            if wait_aware_target:
                for phase in ACTION_LIST:
                    urgency_score[phase] = (
                        self.max_phase_override_queue_weight * float(phase_queue.get(phase, 0.0))
                        + self.max_phase_override_wait_weight
                        * float(phase_wait.get(phase, 0.0))
                        / self.max_phase_override_wait_scale
                        + self.max_phase_override_p95_wait_weight
                        * float(phase_wait_p95.get(phase, 0.0))
                        / self.max_phase_override_p95_wait_scale
                    )
                target_phase = max(ACTION_LIST, key=lambda phase: urgency_score.get(phase, 0.0))
                gap = float(urgency_score.get(target_phase, 0.0) - urgency_score.get(original_action, 0.0))
            else:
                target_phase = max(ACTION_LIST, key=lambda phase: phase_queue.get(phase, 0.0))
                gap = float(phase_queue.get(target_phase, 0.0) - original_queue)
            target_queue = float(phase_queue.get(target_phase, 0.0))
            should_override = target_phase != original_action and gap >= self.max_phase_override_gap
            protect_reason = ""
            if should_override and self.max_phase_override_target_queue_floor > 0.0:
                if target_queue < self.max_phase_override_target_queue_floor:
                    should_override = False
                    protect_reason = (
                        f"target_queue_floor:{target_phase}="
                        f"{target_queue:.3f}<{self.max_phase_override_target_queue_floor:.3f}"
                    )
            if should_override and self.max_phase_override_total_queue_floor > 0.0:
                if total_queue < self.max_phase_override_total_queue_floor:
                    should_override = False
                    protect_reason = (
                        f"total_queue_floor:{total_queue:.3f}<"
                        f"{self.max_phase_override_total_queue_floor:.3f}"
                    )
            if should_override and self.max_phase_override_max_wait_protect > 0.0:
                original_wait_max = float(phase_wait_max.get(original_action, 0.0))
                if original_wait_max >= self.max_phase_override_max_wait_protect:
                    should_override = False
                    protect_reason = (
                        f"max_wait_protect:{original_action}="
                        f"{original_wait_max:.3f}>={self.max_phase_override_max_wait_protect:.3f}"
                    )
            if should_override and self.max_phase_override_left_wait_protect > 0.0:
                if original_action in left_phases and target_phase in through_phases:
                    original_wait = float(phase_wait.get(original_action, 0.0))
                    if original_wait >= self.max_phase_override_left_wait_protect:
                        should_override = False
                        protect_reason = (
                            f"left_wait_protect:{original_action}="
                            f"{original_wait:.3f}>={self.max_phase_override_left_wait_protect:.3f}"
                        )
            if should_override:
                final_actions[idx] = target_phase
                final_indices[idx] = action_to_index(target_phase)
            diagnostics.append(
                {
                    "enabled": True,
                    "overridden": bool(should_override),
                    "protect_reason": protect_reason,
                    "original_action": original_action,
                    "final_action": final_actions[idx],
                    "target_phase": target_phase,
                    "original_queue": original_queue,
                    "target_queue": target_queue,
                    "total_queue": total_queue,
                    "gap": float(gap),
                    "phase_queue": phase_queue,
                    "phase_wait": phase_wait,
                    "phase_wait_p95": phase_wait_p95,
                    "phase_wait_max": phase_wait_max,
                    "urgency_score": urgency_score,
                    "urgency_target_enabled": wait_aware_target,
                    "queue_weight": self.max_phase_override_queue_weight,
                    "wait_weight": self.max_phase_override_wait_weight,
                    "p95_wait_weight": self.max_phase_override_p95_wait_weight,
                    "wait_scale": self.max_phase_override_wait_scale,
                    "p95_wait_scale": self.max_phase_override_p95_wait_scale,
                    "max_wait_protect": self.max_phase_override_max_wait_protect,
                }
            )
        return final_indices, final_actions, diagnostics

    def initialize(self) -> None:
        path_check(self.dic_path)
        self.log_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "projection_test.log")
        if self.logger is None:
            self.logger = setup_logger(self.log_path)
        self.logger.info(
            "Initialized projection test workspace: work_dir=%s model_dir=%s data_dir=%s",
            self.dic_path["PATH_TO_WORK_DIRECTORY"],
            self.dic_path["PATH_TO_MODEL"],
            self.dic_path["PATH_TO_DATA"],
        )
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)
        if self.dump_cityflow_env_args:
            self._dump_cityflow_env_args()
        if self.record_traffic_series:
            self._ensure_traffic_series_config()

        self.env = CityFlowEnv(
            path_to_log=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=self.dic_traffic_env_conf,
            dic_path=self.dic_path,
        )
        self.env.reset()
        self._load_traffic_series()
        self.initialize_policy()
        self.logger.info("Loaded projection checkpoint: %s", self.projection_ckpt)

    def initialize_policy(self) -> None:
        try:
            from projection_train.prefix_action_policy import PrefixActionPolicy
        except Exception as exc:
            raise RuntimeError(
                "Failed to import PrefixActionPolicy. The local transformers/scipy runtime "
                "looks incomplete; please run this script in the same environment used for training."
            ) from exc

        self.policy = PrefixActionPolicy(
            model_path=self.runtime.llm_path,
            seq_len=self.runtime.seq_len,
            image_size=self.runtime.image_size,
            periodicity=self.runtime.periodicity,
            input_dim=self.runtime.input_dim,
            hidden_dim=self.runtime.hidden_dim,
            prefix_tokens=self.runtime.prefix_tokens,
            prefix_mode=self.runtime.prefix_mode,
            max_length=self.runtime.max_length,
            torch_dtype=self.torch_dtype,
            device=self.device,
            with_value_head=False,
            visual_ablation=self.visual_ablation,
            representation_mode=self.representation_mode,
            action_eval_chunk_size=self.action_eval_chunk_size,
        )
        self.policy.load_checkpoint(self.projection_ckpt, load_value_head=False)
        self.policy.eval()
        self.tokenizer = getattr(self.policy.backbone, "tokenizer", None)

    def _dump_cityflow_env_args(self) -> None:
        output_dir = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "cityflow_env_init")
        os.makedirs(output_dir, exist_ok=True)
        payload = {
            "saved_at": datetime.now().isoformat(),
            "cityflow_env_args": {
                "path_to_log": self.dic_path["PATH_TO_WORK_DIRECTORY"],
                "path_to_work_directory": self.dic_path["PATH_TO_WORK_DIRECTORY"],
                "dic_traffic_env_conf": self.dic_traffic_env_conf,
                "dic_path": self.dic_path,
            },
        }
        with open(os.path.join(output_dir, "cityflow_env_args.json"), "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _traffic_count_suffix(self, interval_seconds: int) -> str:
        if interval_seconds == 60:
            return "minute"
        if interval_seconds == 3600:
            return "hour"
        return f"{interval_seconds}s"

    def _traffic_series_run_tag(self) -> str:
        run_dir = str(self.dic_path.get("PATH_TO_WORK_DIRECTORY", "")).strip()
        run_tag = os.path.basename(run_dir) if run_dir else ""
        if not run_tag:
            run_tag = time.strftime("%m_%d_%H_%M_%S", time.localtime())
        return run_tag

    def _ensure_traffic_series_config(self) -> None:
        if not self.record_traffic_series:
            return
        if self._traffic_series_initialized:
            return
        base_dir = str(self.dic_agent_conf.get("TRAFFIC_SERIES_OUTPUT_DIR", "")).strip() or os.path.join(
            "data", "recorded"
        )
        run_tag = self._traffic_series_run_tag()
        output_dir = os.path.join(base_dir, run_tag)
        os.makedirs(output_dir, exist_ok=True)
        suffix = self._traffic_count_suffix(int(self.dic_agent_conf["TRAFFIC_SERIES_INTERVAL"]))
        filename = f"traffic_prompt_series_{suffix}.jsonl"
        self._traffic_series_output_path = os.path.join(output_dir, filename)
        _prepare_jsonl_file(self._traffic_series_output_path)
        self._traffic_series_initialized = True

    def _load_traffic_series(self) -> None:
        if self._traffic_series_movement_keys is not None:
            return
        if self.env is None:
            return

        interval = int(self.dic_agent_conf["TRAFFIC_SERIES_INTERVAL"])
        if not getattr(self.env, "traffic_count_enabled", False):
            current_time = self.env.get_current_time()
            if current_time in (None, 0, 0.0):
                self.dic_traffic_env_conf["ENABLE_TRAFFIC_COUNT"] = True
                self.dic_traffic_env_conf["TRAFFIC_COUNT_MODE"] = "intersection_movement"
                self.dic_traffic_env_conf["TRAFFIC_COUNT_OUTPUT_FORMAT"] = "jsonl"

        intervals = list(getattr(self.env, "traffic_count_intervals", []) or [])
        if interval not in intervals:
            intervals.append(interval)
            self.dic_traffic_env_conf["TRAFFIC_COUNT_INTERVALS"] = sorted(set(int(item) for item in intervals))
            current_time = self.env.get_current_time()
            if current_time in (None, 0, 0.0):
                self.env._init_traffic_counter()

        self._traffic_series_intersection_index = {
            inter_id: idx for idx, inter_id in enumerate(getattr(self.env, "traffic_intersection_ids", []) or [])
        }
        self._traffic_series_movement_keys = list(
            getattr(self.env, "traffic_movement_keys", []) or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        )
        self._traffic_series_base_dt = getattr(self.env, "traffic_base_dt", None) or datetime(1970, 1, 1)

    def _resolve_series_movement_keys(self) -> tuple[list[str], list[str]]:
        default_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
        series_keys = list(self._traffic_series_movement_keys or [])
        if not series_keys:
            return default_keys, default_keys
        output_keys = list(series_keys)
        for key in default_keys:
            if key not in output_keys:
                output_keys.append(key)
        return series_keys, output_keys

    def _build_traffic_series_history(self, intersection_id: str, current_time_s: float) -> list[dict[str, Any]]:
        self._load_traffic_series()
        interval = int(self.dic_agent_conf["TRAFFIC_SERIES_INTERVAL"])
        window_s = int(self.dic_agent_conf["TRAFFIC_SERIES_WINDOW"])
        series_keys, output_keys = self._resolve_series_movement_keys()
        num_series_keys = len(series_keys)
        counts = getattr(self.env, "traffic_counts_by_interval", {}).get(interval) if self.env is not None else None
        inter_idx = (self._traffic_series_intersection_index or {}).get(intersection_id)

        current_bin = int(current_time_s) // interval
        if counts is not None and getattr(counts, "size", 0) > 0 and current_bin >= counts.shape[0]:
            current_bin = counts.shape[0] - 1
        if current_bin < 0:
            current_bin = 0

        window_bins = int(window_s // interval) if interval > 0 else 0
        start_bin = max(0, current_bin - window_bins)
        base_dt = self._traffic_series_base_dt or datetime(1970, 1, 1)
        history = []
        for bin_idx in range(start_bin, current_bin + 1):
            date_str = (base_dt + timedelta(seconds=bin_idx * interval)).strftime("%Y-%m-%d %H:%M:%S")
            movement_counts = {key: 0 for key in output_keys}
            if counts is not None and getattr(counts, "size", 0) > 0 and inter_idx is not None and bin_idx < counts.shape[0]:
                offset = inter_idx * num_series_keys
                row = counts[bin_idx, offset : offset + num_series_keys]
                for mv_idx, key in enumerate(series_keys):
                    if mv_idx < len(row):
                        movement_counts[key] = int(row[mv_idx])
            history.append({"date": date_str, "movement_counts": movement_counts})
        return history

    def _record_traffic_prompt_series(
        self,
        current_time_s: float,
        prompts: Sequence[str],
        predicted_actions: Sequence[str],
        snapshot_paths: Sequence[str],
        prompt_contexts: Sequence[str] | None = None,
        phase_queues: Sequence[Dict[str, float]] | None = None,
        phase_waits: Sequence[Dict[str, float]] | None = None,
    ) -> None:
        if not self.record_traffic_series:
            return
        if not prompts:
            return
        current_bin = int(current_time_s) // int(self.dic_agent_conf["TRAFFIC_SERIES_INTERVAL"])
        if current_bin <= self._traffic_series_last_written_bin:
            return

        _, output_keys = self._resolve_series_movement_keys()
        interval = int(self.dic_agent_conf["TRAFFIC_SERIES_INTERVAL"])
        window_s = int(self.dic_agent_conf["TRAFFIC_SERIES_WINDOW"])
        meta_payload = _build_clip_meta(
            movement_keys=output_keys,
            ts_feature_mode=self.ts_feature_mode,
            ts_image_seq_len=self.ts_image_seq_len,
            interval_s=interval,
            window_s=window_s,
            image_size=self.ts_image_size,
            periodicity=self.ts_image_periodicity,
            norm_const=self.ts_image_norm_const,
            three_channel=self.ts_image_three_channel,
            learnable=self.ts_image_learnable,
        )

        records: list[dict[str, Any]] = []
        assert self.env is not None
        for idx, prompt in enumerate(prompts):
            intersection_id = self.env.list_intersection[idx].inter_name
            history = self._build_traffic_series_history(intersection_id, current_time_s)
            clip_caption = _build_clip_caption(history, output_keys)
            predicted_action = predicted_actions[idx] if idx < len(predicted_actions) else ""
            snapshot_path = snapshot_paths[idx] if idx < len(snapshot_paths) else ""
            prompt_context = prompt_contexts[idx] if prompt_contexts and idx < len(prompt_contexts) else ""
            phase_queue = phase_queues[idx] if phase_queues and idx < len(phase_queues) else {}
            phase_wait = phase_waits[idx] if phase_waits and idx < len(phase_waits) else {}
            for hist_idx, entry in enumerate(history):
                record = {
                    "date": entry.get("date"),
                    "interval_s": interval,
                    "intersection_id": intersection_id,
                    "movement_counts": entry.get("movement_counts", {}),
                    "prompt": prompt if hist_idx == len(history) - 1 else "",
                }
                if hist_idx == len(history) - 1:
                    record["clip_caption"] = clip_caption
                    record["expert_action"] = str(predicted_action or "")
                    record["snapshot_path"] = str(snapshot_path or "")
                    if prompt_context:
                        record["prompt_context"] = str(prompt_context)
                    if phase_queue:
                        record["phase_queue"] = dict(phase_queue)
                    if phase_wait:
                        record["phase_wait"] = dict(phase_wait)
                    record["meta"] = meta_payload
                records.append(record)
        _append_jsonl_records(self._traffic_series_output_path, records)
        self._traffic_series_last_written_bin = current_bin

    def _maybe_take_snapshot(self, current_time_s: float) -> Optional[str]:
        if not self.record_snapshots:
            return None
        interval = int(self.dic_agent_conf["TRAFFIC_SERIES_INTERVAL"])
        current_bin = int(current_time_s) // interval
        if current_bin <= self._snapshot_last_written_bin:
            return None
        if self.env is None:
            return None

        output_dir = os.path.dirname(self._traffic_series_output_path) or self.dic_path["PATH_TO_WORK_DIRECTORY"]
        snapshot_dir = os.path.join(output_dir, "snapshots")
        os.makedirs(snapshot_dir, exist_ok=True)

        # CityFlow's Archive.dump() is intermittently flaky (writes empty or
        # truncated files, e.g. 0-byte or mid-JSON cutoffs). Re-snapshot + re-dump
        # to a fresh path until we produce a complete, parseable JSON file so the
        # downstream relabel/train steps never read a corrupt snapshot.
        snapshot_path = None
        # Keep the historical default while allowing long counterfactual
        # collection jobs to tolerate intermittent CityFlow archive truncation.
        try:
            snapshot_dump_retries = max(1, int(os.environ.get("CITYFLOW_SNAPSHOT_DUMP_RETRIES", "5")))
        except ValueError:
            snapshot_dump_retries = 5
        for _attempt in range(snapshot_dump_retries):
            rand_tag = uuid.uuid4().hex[:8]
            candidate = os.path.join(
                snapshot_dir,
                f"snapshot_t{int(current_time_s)}_b{current_bin}_{rand_tag}.json",
            )
            archive = self.env.eng.snapshot()
            archive.dump(candidate)
            try:
                with open(candidate, "rb") as _fh:
                    json.load(_fh)
                snapshot_path = candidate
                break
            except Exception:
                continue

        if snapshot_path is None:
            snapshot_path = candidate
            self.logger.warning(
                "snapshot dump still invalid at t=%s after retries", current_time_s,
            )

        self._snapshot_last_written_bin = current_bin
        # self.env.eng.load_from_file(snapshot_path)
        return snapshot_path

    def _build_raw_prompt(self, state: Dict[str, Any]) -> str:
        prompt = getPrompt(state2text(state))
        return (
            prompt[0]["content"]
            + "\n\n### Instruction:\n"
            + prompt[1]["content"]
            + "\n\n### Response:\n"
        )

    def _history_to_sequence(self, history: Sequence[Dict[str, Any]], movement_keys: Sequence[str]) -> list[list[float]]:
        rows = [
            [float((entry.get("movement_counts") or {}).get(key, 0.0) or 0.0) for key in movement_keys]
            for entry in history
        ]
        if not rows:
            return [[0.0 for _ in movement_keys] for _ in range(self.runtime.seq_len)]
        window = rows[-self.runtime.seq_len :]
        if len(window) < self.runtime.seq_len:
            pad_item = window[0]
            window = [list(pad_item) for _ in range(self.runtime.seq_len - len(window))] + window
        return window

    def predict_from_sequences(
        self,
        sequences: Sequence[Sequence[Sequence[float]]],
        policy_prompts: Sequence[str],
    ) -> Dict[str, Any]:
        """Score action tags for policy inputs prepared outside the online loop.

        Counterfactual replay seeds each intersection's sequence from a saved
        decision record, then advances it in an alternate CityFlow branch.  It
        must invoke the same model, batching, and action decoding as online
        control, so the model call lives in this reusable method.
        """
        if len(sequences) != len(policy_prompts):
            raise ValueError(
                f"sequences/prompts length mismatch: {len(sequences)} != {len(policy_prompts)}"
            )
        if not sequences:
            return {
                "action_indices": [],
                "predicted_actions": [],
                "logits": [],
                "probs": [],
            }

        batch_size = self.inference_batch_size or len(sequences)
        action_indices: list[int] = []
        logits_payload: list[dict[str, float]] = []
        probs_payload: list[dict[str, float]] = []
        assert self.policy is not None
        for start_idx in range(0, len(sequences), batch_size):
            end_idx = min(start_idx + batch_size, len(sequences))
            sequence_tensor = torch.tensor(
                sequences[start_idx:end_idx], dtype=torch.float32, device=self.device
            )
            with torch.inference_mode():
                logits = self.policy(sequence_tensor, policy_prompts[start_idx:end_idx])
                probs = torch.softmax(logits.float(), dim=-1)
                batch_action_indices = logits.argmax(dim=-1)

            action_indices.extend(int(index) for index in batch_action_indices.tolist())
            logits_payload.extend(
                {
                    action: float(logits[row_idx, col_idx].float().item())
                    for col_idx, action in enumerate(ACTION_LIST)
                }
                for row_idx in range(logits.shape[0])
            )
            probs_payload.extend(
                {action: float(probs[row_idx, col_idx].item()) for col_idx, action in enumerate(ACTION_LIST)}
                for row_idx in range(probs.shape[0])
            )

        return {
            "action_indices": action_indices,
            "predicted_actions": [index_to_action(index) for index in action_indices],
            "logits": logits_payload,
            "probs": probs_payload,
        }

    def _predict_actions(
        self,
        current_states: Sequence[Dict[str, Any]],
        current_time_s: float,
    ) -> Dict[str, Any]:
        _, movement_keys = self._resolve_series_movement_keys()
        if self.prompt_mode in {"matched_minimal", "matched_structured", "information_matched"} and tuple(movement_keys) != MOVEMENT_KEYS:
            raise ValueError(f"Matched representation requires movement order {MOVEMENT_KEYS}; got {movement_keys}")
        if not self._movement_keys_checked:
            if len(movement_keys) != self.runtime.input_dim:
                raise ValueError(
                    f"Traffic series feature dim {len(movement_keys)} does not match "
                    f"projection input_dim {self.runtime.input_dim}."
                )
            self._movement_keys_checked = True

        raw_prompts: list[str] = []
        policy_prompts: list[str] = []
        prompt_contexts: list[str] = []
        phase_queues: list[dict[str, float]] = []
        phase_waits: list[dict[str, float]] = []
        phase_wait_p95s: list[dict[str, float]] = []
        phase_wait_maxes: list[dict[str, float]] = []
        sequences: list[list[list[float]]] = []
        assert self.env is not None
        for idx, state in enumerate(current_states):
            raw_prompt = self._build_raw_prompt(state)
            phase_queue, phase_wait = self._phase_queue_wait(state)
            phase_wait_p95, phase_wait_max = self._phase_wait_tail(self.env.list_intersection[idx].inter_name)
            prompt_context = build_phase_queue_wait_context(
                state=state,
                phase_queue=phase_queue,
                phase_wait=phase_wait,
            )
            policy_prompt = build_task_prompt(
                raw_prompt,
                mode=self.prompt_mode,
                action_tags=ACTION_LIST,
                traffic_context=prompt_context,
            )
            history = self._build_traffic_series_history(self.env.list_intersection[idx].inter_name, current_time_s)
            sequence = self._history_to_sequence(history, movement_keys)
            sequences.append(sequence)
            if self.representation_mode == "history_text":
                policy_prompt += "\n" + (serialize_compact_movement_history(sequence) if self.prompt_mode == "information_matched" else serialize_movement_history(sequence))
            raw_prompts.append(raw_prompt)
            policy_prompts.append(policy_prompt)
            prompt_contexts.append(prompt_context)
            phase_queues.append(phase_queue)
            phase_waits.append(phase_wait)
            phase_wait_p95s.append(phase_wait_p95)
            phase_wait_maxes.append(phase_wait_max)

        policy_output = self.predict_from_sequences(sequences, policy_prompts)
        return {
            "raw_prompts": raw_prompts,
            "policy_prompts": policy_prompts,
            "prompt_contexts": prompt_contexts,
            "phase_queues": phase_queues,
            "phase_waits": phase_waits,
            "phase_wait_p95s": phase_wait_p95s,
            "phase_wait_maxes": phase_wait_maxes,
            "sequences": sequences,
            **policy_output,
        }

    def get_norm_reward(self, state: Sequence[Dict[str, Any]]) -> list[float]:
        rewards = []
        assert self.env is not None
        for idx in range(len(state)):
            vehicle_num = 0
            queue_length = 0
            intersection = self.env.intersection_dict[self.env.list_intersection[idx].inter_name]
            roads = copy.deepcopy(intersection["roads"])
            statistic_state, _, _ = get_state_detail(roads, self.env)
            for lane in statistic_state:
                queue_length += statistic_state[lane]["queue_len"]
                vehicle_num += statistic_state[lane]["queue_len"]
                for cell_idx in range(len(statistic_state[lane]["cells"])):
                    vehicle_num += statistic_state[lane]["cells"][cell_idx]
            reward = -(queue_length / vehicle_num) if vehicle_num > 0.0 else -0.0
            rewards.append(float(reward))
        return rewards

    def test(self, logger: Any, test_round: int) -> Dict[str, float]:
        self.logger.info("================ Start Projection Test ================")
        assert self.env is not None
        total_run_cnt = self.dic_traffic_env_conf["RUN_COUNTS"]
        done = False
        state = self.env.reset()
        self._traffic_series_intersection_index = None
        self._traffic_series_movement_keys = None
        self._traffic_series_base_dt = None
        self._traffic_series_last_written_bin = -1
        self._snapshot_last_written_bin = -1
        self._load_traffic_series()
        total_reward = 0.0
        queue_length_episode: list[float] = []
        waiting_time_episode: list[float] = []
        # One value per vehicle: the longest continuous stopped interval that
        # CityFlow exposed at an action boundary. This is distinct from the
        # existing time-average over the current waiting set.
        vehicle_max_observed_wait: Dict[str, float] = {}
        current_time = self.env.get_current_time()

        start_time = time.time()
        state_action_log = [[] for _ in range(len(state))]
        total_steps = int(total_run_cnt / self.dic_traffic_env_conf["MIN_ACTION_TIME"])

        for _ in tqdm(range(total_steps)):
            if done or current_time >= total_run_cnt:
                break

            current_states = []
            for idx in range(len(state)):
                intersection = self.env.intersection_dict[self.env.list_intersection[idx].inter_name]
                roads = copy.deepcopy(intersection["roads"])
                statistic_state, statistic_state_incoming, mean_speed = get_state_detail(roads, self.env)
                state_action_log[idx].append(
                    {
                        "state": statistic_state,
                        "state_incoming": statistic_state_incoming,
                        "approaching_speed": mean_speed,
                    }
                )
                current_states.append(statistic_state)

            snapshot_path = self._maybe_take_snapshot(current_time)
            prediction = self._predict_actions(current_states, current_time)
            action_list = prediction["action_indices"]
            predicted_actions = prediction["predicted_actions"]
            policy_action_list = list(action_list)
            policy_predicted_actions = list(predicted_actions)
            phase_wait_maxes = prediction.get("phase_wait_maxes", [])
            phase_wait_p95s = prediction.get("phase_wait_p95s", [])
            action_list, predicted_actions, override_diagnostics = self._apply_max_phase_override(
                current_states,
                action_list,
                predicted_actions,
                phase_wait_p95s=phase_wait_p95s,
                phase_wait_maxes=phase_wait_maxes,
            )
            policy_prompts = prediction["policy_prompts"]
            raw_prompts = prediction["raw_prompts"]
            prompt_contexts = prediction.get("prompt_contexts", [])
            phase_queues = prediction.get("phase_queues", [])
            phase_waits = prediction.get("phase_waits", [])
            sequences = prediction.get("sequences", [])
            logits_payload = prediction["logits"]
            probs_payload = prediction["probs"]
            snapshot_paths = [snapshot_path or "" for _ in range(len(action_list))]

            for idx, action_name in enumerate(predicted_actions):
                state_action_log[idx][-1]["response"] = format_action_tag(action_name)
                state_action_log[idx][-1]["action"] = eight_phase_list[action_list[idx]]
                state_action_log[idx][-1]["policy_response"] = format_action_tag(policy_predicted_actions[idx])
                state_action_log[idx][-1]["policy_action"] = eight_phase_list[policy_action_list[idx]]
                state_action_log[idx][-1]["policy_prompt"] = policy_prompts[idx]
                state_action_log[idx][-1]["raw_prompt"] = raw_prompts[idx]
                if idx < len(prompt_contexts):
                    state_action_log[idx][-1]["prompt_context"] = prompt_contexts[idx]
                if idx < len(phase_queues):
                    state_action_log[idx][-1]["phase_queue"] = phase_queues[idx]
                if idx < len(phase_waits):
                    state_action_log[idx][-1]["phase_wait"] = phase_waits[idx]
                if idx < len(phase_wait_p95s):
                    state_action_log[idx][-1]["phase_wait_p95"] = phase_wait_p95s[idx]
                if idx < len(phase_wait_maxes):
                    state_action_log[idx][-1]["phase_wait_max"] = phase_wait_maxes[idx]
                state_action_log[idx][-1]["action_logits"] = logits_payload[idx]
                state_action_log[idx][-1]["action_probs"] = probs_payload[idx]
                if idx < len(override_diagnostics):
                    state_action_log[idx][-1]["max_phase_override"] = override_diagnostics[idx]
                if self.record_snapshots and snapshot_paths[idx]:
                    state_action_log[idx][-1]["snapshot_path"] = snapshot_paths[idx]

            self._record_traffic_prompt_series(
                current_time_s=current_time,
                prompts=policy_prompts,
                predicted_actions=predicted_actions,
                snapshot_paths=snapshot_paths,
                prompt_contexts=prompt_contexts,
                phase_queues=phase_queues,
                phase_waits=phase_waits,
            )

            # Collect states for Stage 3 cross-network adaptation (SimExpert relabeling)
            if self.collect_states_output:
                for idx in range(len(state)):
                    inter_id = self.env.list_intersection[idx].inter_name
                    # Serialize the EXACT canonical sequence fed to the policy
                    seq = sequences[idx] if idx < len(sequences) else []
                    # Also build movement_counts dict for SimExpert compatibility
                    movement_keys_list = (
                        self._traffic_series_movement_keys
                        or ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]
                    )
                    movement_counts = {}
                    if seq and len(seq) > 0 and len(seq[-1]) == len(movement_keys_list):
                        last_row = seq[-1]
                        for mk_idx, key in enumerate(movement_keys_list):
                            movement_counts[key] = float(last_row[mk_idx])
                    record = {
                        "intersection_id": inter_id,
                        "time_s": float(current_time),
                        "movement_counts": movement_counts,
                        "sequences": seq,  # The exact canonical 16x8 tensor rows
                        "movement_keys": movement_keys_list,
                        "prompt": policy_prompts[idx] if idx < len(policy_prompts) else "",
                        "prompt_context": prompt_contexts[idx] if idx < len(prompt_contexts) else "",
                        "phase_queue": phase_queues[idx] if idx < len(phase_queues) else {},
                        "phase_wait": phase_waits[idx] if idx < len(phase_waits) else {},
                        "phase_wait_p95": phase_wait_p95s[idx] if idx < len(phase_wait_p95s) else {},
                        "phase_wait_max": phase_wait_maxes[idx] if idx < len(phase_wait_maxes) else {},
                        "policy_action": policy_predicted_actions[idx] if idx < len(policy_predicted_actions) else "",
                        "snapshot_path": snapshot_paths[idx] if idx < len(snapshot_paths) else "",
                        "traffic_file": str(self.trafficflow),
                        "dataset": str(self.roadnet),
                    }
                    if self._collect_states_fh is None:
                        os.makedirs(os.path.dirname(self.collect_states_output) or ".", exist_ok=True)
                        self._collect_states_fh = open(self.collect_states_output, "a", encoding="utf-8")
                    self._collect_states_fh.write(json.dumps(record, ensure_ascii=False) + "\n")

            next_state, _, done, _ = self.env.step(action_list)
            rewards = self.get_norm_reward(next_state)
            current_time = self.env.get_current_time()
            state = next_state

            total_reward += sum(rewards)
            queue_length_inter = [
                sum(inter.dic_feature["lane_num_waiting_vehicle_in"]) for inter in self.env.list_intersection
            ]
            queue_length_episode.append(sum(queue_length_inter))

            waiting_times = [
                self.env.waiting_vehicle_list[veh]["time"] for veh in self.env.waiting_vehicle_list
            ]
            waiting_time_episode.append(np.mean(waiting_times) if waiting_times else 0.0)
            for vehicle_id, vehicle_info in self.env.waiting_vehicle_list.items():
                vehicle_id = str(vehicle_id)
                if "shadow" in vehicle_id:
                    continue
                wait_s = float(vehicle_info.get("time", 0.0) or 0.0)
                vehicle_max_observed_wait[vehicle_id] = max(
                    vehicle_max_observed_wait.get(vehicle_id, 0.0), wait_s
                )
            self.logger.info("Queuing Vehicles: %s", sum(queue_length_episode))

        vehicle_travel_times: Dict[str, list[float]] = {}
        for inter in self.env.list_intersection:
            arrive_left_times = inter.dic_vehicle_arrive_leave_time
            for veh, times in arrive_left_times.items():
                if "shadow" in veh:
                    continue
                enter_time = times["enter_time"]
                leave_time = times["leave_time"]
                if not np.isnan(enter_time):
                    resolved_leave = leave_time if not np.isnan(leave_time) else self.dic_traffic_env_conf["RUN_COUNTS"]
                    vehicle_travel_times.setdefault(veh, []).append(resolved_leave - enter_time)

        total_travel_time = (
            np.mean([sum(vehicle_travel_times[veh]) for veh in vehicle_travel_times])
            if vehicle_travel_times
            else 0.0
        )

        results = {
            "test_reward_over": float(total_reward),
            "test_avg_queue_len_over": float(np.mean(queue_length_episode) if queue_length_episode else 0.0),
            "test_queuing_vehicle_num_over": float(np.sum(queue_length_episode) if queue_length_episode else 0.0),
            "test_avg_waiting_time_over": float(np.mean(waiting_time_episode) if waiting_time_episode else 0.0),
            "test_avg_travel_time_over": float(total_travel_time),
            "test_vehicle_wait_p95_over": float(
                np.percentile(list(vehicle_max_observed_wait.values()), 95)
                if vehicle_max_observed_wait
                else 0.0
            ),
            "test_vehicle_wait_max_over": float(max(vehicle_max_observed_wait.values(), default=0.0)),
            "test_vehicle_wait_observed_vehicle_count": float(len(vehicle_max_observed_wait)),
        }
        self.logger.info(
            "Test results: %s",
            json.dumps(results, ensure_ascii=False, sort_keys=True),
        )
        if logger is not None:
            logger.log(results)
        self.logger.info("Test Round %d complete.", test_round)
        if self.max_phase_override_gap > 0.0:
            override_total = 0
            override_applied = 0
            for inter_log in state_action_log:
                for item in inter_log:
                    diag = item.get("max_phase_override") or {}
                    if diag.get("enabled"):
                        override_total += 1
                        override_applied += int(bool(diag.get("overridden")))
            self.logger.info(
                (
                    "Max-phase override summary: applied=%d checked=%d rate=%.4f "
                    "gap=%.3f left_wait_protect=%.3f max_wait_protect=%.3f "
                    "target_queue_floor=%.3f total_queue_floor=%.3f"
                ),
                override_applied,
                override_total,
                (override_applied / override_total) if override_total else 0.0,
                self.max_phase_override_gap,
                self.max_phase_override_left_wait_protect,
                self.max_phase_override_max_wait_protect,
                self.max_phase_override_target_queue_floor,
                self.max_phase_override_total_queue_floor,
            )

        state_action_path = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "state_action.json")
        dump_json(state_action_log, state_action_path)
        self.logger.info("Testing time: %.4f", time.time() - start_time)
        self.env.batch_log_2()
        if self._collect_states_fh is not None:
            self._collect_states_fh.close()
            self._collect_states_fh = None
            self.logger.info("Collected states written to %s", self.collect_states_output)
        return results

    def train_test(self) -> Dict[str, float]:
        all_config = merge(merge(self.dic_agent_conf, self.dic_path), self.dic_traffic_env_conf)
        logger: Any = None
        run = None
        if self.use_wandb:
            if wandb is None:
                self.logger.warning("wandb is unavailable, falling back to local logger output.")
            else:
                run = wandb.init(
                    project=self.dic_traffic_env_conf["PROJECT_NAME"],
                    group=(
                        f"{self.dic_traffic_env_conf['MODEL_NAME']}-"
                        f"{self.roadnet}-{self.trafficflow}-{len(self.dic_traffic_env_conf['PHASE'])}_Phases"
                    ),
                    name=self.dic_traffic_env_conf["TRAFFIC_FILE"].replace(".json", ""),
                    config=all_config,
                )
                logger = run
        try:
            return self.test(logger, 0)
        finally:
            if run is not None:
                wandb.finish()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memo", type=str, default="projection_train")
    parser.add_argument("--projection_ckpt", type=str, required=True)
    parser.add_argument("--llm_path", type=str, default="")
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--proj_name", type=str, default="LLM-TSCS-extreme")
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")
    parser.add_argument("--run_counts", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--prompt_mode", type=str, default="")
    parser.add_argument("--representation_mode", choices=("", "ts2i", "history_text"), default="")
    parser.add_argument("--seq_len", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=0)
    parser.add_argument("--periodicity", type=int, default=0)
    parser.add_argument("--input_dim", type=int, default=0)
    parser.add_argument("--hidden_dim", type=int, default=0)
    parser.add_argument("--prefix_tokens", type=int, default=0)
    parser.add_argument("--prefix_mode", type=str, default="")
    parser.add_argument("--max_length", type=int, default=0)
    parser.add_argument("--min_action_time", type=int, default=30)
    parser.add_argument("--traffic_series_interval", type=int, default=2)
    parser.add_argument("--traffic_series_window", type=int, default=0)
    parser.add_argument("--traffic_series_output_dir", type=str, default="")
    parser.add_argument("--result_json", type=str, default="")
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=0,
        help="Maximum intersections evaluated per model forward pass; 0 evaluates all at once.",
    )
    parser.add_argument("--action_eval_chunk_size", type=int, default=0)
    parser.add_argument(
        "--visual_ablation",
        choices=("full", "zero_prefix", "shuffled_image", "constant_image"),
        default="full",
        help="Inference-time visual intervention; all non-visual inputs remain unchanged.",
    )
    parser.add_argument(
        "--max_phase_override_gap",
        type=float,
        default=0.0,
        help="If >0, override the model action to the max-queue phase when queue gap reaches this threshold.",
    )
    parser.add_argument(
        "--max_phase_override_left_wait_protect",
        type=float,
        default=0.0,
        help="When overriding from a left-turn action to a through action, keep the left action if its wait exceeds this value.",
    )
    parser.add_argument(
        "--max_phase_override_max_wait_protect",
        type=float,
        default=0.0,
        help="Keep the current phase when its observed vehicle max wait reaches this value.",
    )
    parser.add_argument(
        "--max_phase_override_target_queue_floor",
        type=float,
        default=0.0,
        help="If >0, only override when the max-queue target phase has at least this queue.",
    )
    parser.add_argument(
        "--max_phase_override_total_queue_floor",
        type=float,
        default=0.0,
        help="If >0, only override when total phase queue at the intersection reaches this floor.",
    )
    parser.add_argument(
        "--max_phase_override_queue_weight",
        type=float,
        default=1.0,
        help="Queue weight for wait-aware override urgency scoring; default preserves max-queue behavior.",
    )
    parser.add_argument(
        "--max_phase_override_wait_weight",
        type=float,
        default=0.0,
        help="Average phase-wait weight for wait-aware override urgency scoring.",
    )
    parser.add_argument(
        "--max_phase_override_p95_wait_weight",
        type=float,
        default=0.0,
        help="Phase p95 wait weight for wait-aware override urgency scoring.",
    )
    parser.add_argument(
        "--max_phase_override_wait_scale",
        type=float,
        default=60.0,
        help="Seconds used to scale average phase wait in wait-aware urgency scoring.",
    )
    parser.add_argument(
        "--max_phase_override_p95_wait_scale",
        type=float,
        default=300.0,
        help="Seconds used to scale phase p95 wait in wait-aware urgency scoring.",
    )
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=False,
        help="Require deterministic PyTorch CUDA kernels for comparable online seed runs.",
    )
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--eightphase", action="store_true", default=False)
    parser.add_argument(
        "--collect_states_output",
        type=str,
        default="",
        help="If set, write decision-point states to this JSONL file for SimExpert relabeling.",
    )
    parser.add_argument(
        "--disable_snapshots",
        action="store_true",
        default=False,
        help="Skip CityFlow archive snapshots when the run only needs closed-loop metrics.",
    )
    return parser.parse_args()


def _write_result_json(path: str, payload: Dict[str, Any]) -> None:
    target = str(path or "").strip()
    if not target:
        return
    output_dir = os.path.dirname(target)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    temporary = f"{target}.tmp.{os.getpid()}"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(args: argparse.Namespace) -> Dict[str, Any]:
    configure_stdio()
    if args.fp16 and args.bf16:
        raise ValueError("Choose only one of --fp16 or --bf16.")
    if args.eightphase:
        raise ValueError("Stage 2/3 projection checkpoints only support the four-phase action space.")
    if args.deterministic:
        # Must be configured before the first CUDA kernel is launched.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    default_count, road_net, traffic_file_list, template = _dataset_spec(args.dataset)
    if args.traffic_file not in traffic_file_list:
        raise error.flowFileException("Flow file does not exist.")

    runtime, prompt_mode = _resolve_projection_runtime(args)
    checkpoint_args = _load_checkpoint_args(args.projection_ckpt)
    representation_mode = str(args.representation_mode or checkpoint_args.get("representation_mode") or "ts2i")
    if args.representation_mode and args.representation_mode != str(checkpoint_args.get("representation_mode") or "ts2i"):
        raise ValueError("Online representation_mode must match the checkpoint")

    run_counts = int(args.run_counts or default_count)
    if "24h" in args.traffic_file:
        run_counts = 86400

    min_action_time = int(args.min_action_time or 30)
    traffic_series_interval = int(args.traffic_series_interval or 2)
    traffic_series_window = int(args.traffic_series_window or 0)
    if traffic_series_window <= 0:
        traffic_series_window = max(traffic_series_interval, (runtime.seq_len - 1) * traffic_series_interval)

    num_row = int(road_net.split("_")[0])
    num_col = int(road_net.split("_")[1])
    num_intersections = num_row * num_col
    llm_model_name = os.path.basename(runtime.llm_path.rstrip("/")) or "qwen-3-vl-7b"
    ckpt_name = os.path.basename(args.projection_ckpt.rstrip("/"))

    dic_agent_conf_extra = {
        "LLM_PATH": runtime.llm_path,
        "LLM_MODEL": llm_model_name,
        "LOG_DIR": f"./{llm_model_name}_logs",
        "TRAFFIC_SERIES_INTERVAL": traffic_series_interval,
        "TRAFFIC_SERIES_WINDOW": traffic_series_window,
        "TRAFFIC_SERIES_OUTPUT_DIR": args.traffic_series_output_dir,
        "TS_IMAGE_SEQ_LEN": runtime.seq_len,
        "TS_IMAGE_SIZE": runtime.image_size,
        "TS_IMAGE_PERIODICITY": runtime.periodicity,
        "TS_IMAGE_NORM_CONST": 0.4,
        "TS_IMAGE_LEARNABLE": True,
        "TS_IMAGE_THREE_CHANNEL": True,
        "PROJECTION_CKPT": args.projection_ckpt,
        "PROMPT_MODE": prompt_mode,
        "SAVE_CITYFLOW_ENV_ARGS": True,
        "MAX_PHASE_OVERRIDE_GAP": float(args.max_phase_override_gap or 0.0),
        "MAX_PHASE_OVERRIDE_LEFT_WAIT_PROTECT": float(args.max_phase_override_left_wait_protect or 0.0),
        "MAX_PHASE_OVERRIDE_MAX_WAIT_PROTECT": float(args.max_phase_override_max_wait_protect or 0.0),
        "MAX_PHASE_OVERRIDE_TARGET_QUEUE_FLOOR": float(args.max_phase_override_target_queue_floor or 0.0),
        "MAX_PHASE_OVERRIDE_TOTAL_QUEUE_FLOOR": float(args.max_phase_override_total_queue_floor or 0.0),
        "MAX_PHASE_OVERRIDE_QUEUE_WEIGHT": float(args.max_phase_override_queue_weight),
        "MAX_PHASE_OVERRIDE_WAIT_WEIGHT": float(args.max_phase_override_wait_weight or 0.0),
        "MAX_PHASE_OVERRIDE_P95_WAIT_WEIGHT": float(args.max_phase_override_p95_wait_weight or 0.0),
        "MAX_PHASE_OVERRIDE_WAIT_SCALE": float(args.max_phase_override_wait_scale or 60.0),
        "MAX_PHASE_OVERRIDE_P95_WAIT_SCALE": float(args.max_phase_override_p95_wait_scale or 300.0),
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "MODEL_NAME": f"{args.memo}-{llm_model_name}-{ckpt_name}",
        "PROJECT_NAME": args.proj_name,
        "RUN_COUNTS": run_counts,
        "NUM_ROUNDS": args.num_rounds,
        "NUM_ROW": num_row,
        "NUM_COL": num_col,
        "CITYFLOW_SEED": seed,
        "MIN_ACTION_TIME": min_action_time,
        "MEASURE_TIME": min_action_time,
        "ENABLE_TRAFFIC_COUNT": True,
        "TRAFFIC_COUNT_MODE": "intersection_movement",
        "TRAFFIC_COUNT_OUTPUT_FORMAT": "jsonl",
        "TRAFFIC_COUNT_INTERVALS": sorted({traffic_series_interval, 5, 60, 3600}),
        "TRAFFIC_COUNT_ADD_TOTAL": True,
        "TRAFFIC_COUNT_BASENAME": "traffic_counts",
        "TRAFFIC_FILE": args.traffic_file,
        "ROADNET_FILE": f"roadnet_{road_net}.json",
        "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
        "DIC_REWARD_INFO": {"queue_length": -0.25},
    }

    dic_agent_conf_extra["FIXED_TIME"] = [min_action_time for _ in range(len(ACTION_LIST))]
    timestamp = time.strftime("%m_%d_%H_%M_%S", time.localtime())
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join("model", args.memo, args.traffic_file + "_" + timestamp),
        "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, args.traffic_file + "_" + timestamp),
        "PATH_TO_DATA": os.path.join("data", template, road_net),
    }

    device = _resolve_device(args.device)
    torch_dtype = _resolve_torch_dtype(args, device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = bool(args.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = not args.deterministic

    runner = ProjectionPolicyRunner(
        dic_agent_conf=dic_agent_conf_extra,
        dic_traffic_env_conf_local=merge(dic_traffic_env_conf, dic_traffic_env_conf_extra),
        dic_path=dic_path_extra,
        roadnet=f"{template}-{road_net}",
        trafficflow=args.traffic_file.split(".")[0],
        runtime=runtime,
        projection_ckpt=args.projection_ckpt,
        prompt_mode=prompt_mode,
        device=device,
        torch_dtype=torch_dtype,
        use_wandb=args.use_wandb,
        max_phase_override_gap=args.max_phase_override_gap,
        max_phase_override_left_wait_protect=args.max_phase_override_left_wait_protect,
        max_phase_override_max_wait_protect=args.max_phase_override_max_wait_protect,
        max_phase_override_target_queue_floor=args.max_phase_override_target_queue_floor,
        max_phase_override_total_queue_floor=args.max_phase_override_total_queue_floor,
        max_phase_override_queue_weight=args.max_phase_override_queue_weight,
        max_phase_override_wait_weight=args.max_phase_override_wait_weight,
        max_phase_override_p95_wait_weight=args.max_phase_override_p95_wait_weight,
        max_phase_override_wait_scale=args.max_phase_override_wait_scale,
        max_phase_override_p95_wait_scale=args.max_phase_override_p95_wait_scale,
        inference_batch_size=args.inference_batch_size,
        visual_ablation=args.visual_ablation,
        representation_mode=representation_mode,
        action_eval_chunk_size=args.action_eval_chunk_size,
        collect_states_output=args.collect_states_output,
        record_snapshots=not args.disable_snapshots,
    )
    if runtime.input_dim != len(location_direction_dict):
        runner.logger.warning(
            "Checkpoint input_dim=%d, online movement dim defaults to %d.",
            runtime.input_dim,
            len(location_direction_dict),
        )
    results = runner.train_test()
    payload = {
        "results": results,
        "metric_semantics": {
            "test_vehicle_wait_p95_over": (
                "p95 across unique non-shadow vehicles of each vehicle's maximum continuous "
                "waiting time observed at an action boundary"
            ),
            "test_vehicle_wait_max_over": (
                "maximum across unique non-shadow vehicles of maximum continuous waiting "
                "time observed at an action boundary"
            ),
            "test_vehicle_wait_observed_vehicle_count": (
                "unique non-shadow vehicles observed waiting at least once at an action boundary"
            ),
        },
        "args": vars(args),
        "artifact_hashes": {
            "projection_ckpt_sha256": sha256_file(args.projection_ckpt),
            "online_evaluator_sha256": sha256_file(__file__),
            "prefix_action_policy_sha256": sha256_file(
                os.path.join(os.path.dirname(__file__), "prefix_action_policy.py")
            ),
        },
        "determinism": {
            "requested": bool(args.deterministic),
            "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        },
        "runtime": asdict(runtime),
        "prompt_mode": prompt_mode,
        "representation_mode": representation_mode,
        "work_dir": runner.dic_path["PATH_TO_WORK_DIRECTORY"],
        "model_dir": runner.dic_path["PATH_TO_MODEL"],
        "data_dir": runner.dic_path["PATH_TO_DATA"],
        "log_path": runner.log_path,
    }
    _write_result_json(args.result_json, payload)
    return payload


if __name__ == "__main__":
    main(parse_args())
