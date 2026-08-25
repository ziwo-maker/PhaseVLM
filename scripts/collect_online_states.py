#!/usr/bin/env python3
"""Collect decision-point states during online evaluation for SimExpert relabeling.

This script runs a PhaseVLM policy in the standard CityFlowEnv (the paper's
canonical evaluator) and records the full observation at every decision step.
The collected states can later be relabeled by SimExpert and used for
few-shot cross-network adaptation.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/collect_online_states.py \
    --projection_ckpt records/.../best.pt \
    --dataset jinan --traffic_file anon_3_4_jinan_real.json \
    --output states.jsonl --max_steps 120
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from projection_train.action_space import ACTION_LIST, action_to_index, index_to_action
from projection_train.prompt_utils import (
    MOVEMENT_KEYS,
    build_phase_queue_wait_context,
    build_task_prompt,
)
from projection_train.prefix_action_policy import PrefixActionPolicy
from utils.cityflow_env import CityFlowEnv
from utils.config import dic_traffic_env_conf
from utils.my_utils import eight_phase_list


def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    merged.update(override)
    return merged


def _resolve_path(path: str) -> str:
    resolved = os.path.abspath(os.path.expanduser(path))
    if os.path.exists(resolved):
        return resolved
    repo_resolved = os.path.join(_REPO, path)
    if os.path.exists(repo_resolved):
        return os.path.abspath(repo_resolved)
    return resolved


def _load_checkpoint_args(ckpt_path: str) -> Dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        return {}
    ckpt_args = checkpoint.get("args")
    if isinstance(ckpt_args, dict):
        return dict(ckpt_args)
    if hasattr(ckpt_args, "__dict__"):
        return dict(vars(ckpt_args))
    return {}


def _collect_waiting_metrics(env: CityFlowEnv) -> Dict[str, Any]:
    """Extract per-phase queue length and waiting time from CityFlowEnv."""
    phase_queue: Dict[str, float] = {}
    phase_wait: Dict[str, float] = {}
    movement_queue_wait: Dict[str, Dict[str, float]] = {}

    for inter in env.list_intersection:
        features = inter.dic_feature or {}
        lane_waiting = features.get("lane_num_waiting_vehicle_in", [])
        # Map lanes to phases based on intersection structure
        pass

    # Simplified: use waiting_vehicle_list directly
    waiting_by_phase: Dict[int, List[float]] = {i: [] for i in range(len(ACTION_LIST))}
    for veh_info in (env.waiting_vehicle_list or {}).values():
        wait_time = float(veh_info.get("time", 0.0) or 0.0)
        # Assign to all phases conservatively
        for phase_idx in range(len(ACTION_LIST)):
            waiting_by_phase[phase_idx].append(wait_time)

    for idx, action_name in enumerate(ACTION_LIST):
        waits = waiting_by_phase.get(idx, [])
        phase_wait[action_name] = float(np.mean(waits) if waits else 0.0)
        phase_queue[action_name] = float(len(waits))

    return {
        "phase_queue": phase_queue,
        "phase_wait": phase_wait,
        "movement_queue_wait": movement_queue_wait,
    }


def collect_states(
    policy: PrefixActionPolicy,
    ckpt_path: str,
    llm_path: str,
    dataset: str,
    traffic_file: str,
    roadnet_file: str,
    output_path: str,
    run_counts: int = 3600,
    seq_len: int = 16,
    image_size: int = 224,
    periodicity: int = 16,
    input_dim: int = 8,
    hidden_dim: int = 64,
    prefix_tokens: int = 8,
    prefix_mode: str = "prefix",
    max_length: int = 64,
    prompt_mode: str = "queue_wait_short",
    bf16: bool = True,
    seed: int = 2024,
    max_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    """Run policy in standard CityFlowEnv and collect all decision-point states."""

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Resolve paths
    traffic_root = os.path.join(_REPO, "data", dataset.capitalize())
    roadnet_path = _resolve_path(roadnet_file or os.path.join("data", dataset.capitalize(), "roadnet.json"))
    resolved_traffic = traffic_file

    # Build CityFlow env config (matching run_projection_online.py)
    num_col, num_row = {"jinan": (3, 4), "hangzhou": (4, 4), "newyork_28x7": (28, 7)}.get(
        dataset.lower(), (3, 4)
    )

    env_conf = _merge_configs(
        dic_traffic_env_conf,
        {
            "MODEL_NAME": "phasevlm-collect",
            "PROJECT_NAME": "phasevlm-collect",
            "NUM_AGENTS": num_col * num_row,
            "NUM_INTERSECTIONS": num_col * num_row,
            "NUM_COL": num_col,
            "NUM_ROW": num_row,
            "RUN_COUNTS": run_counts,
            "MIN_ACTION_TIME": 30,
            "MEASURE_TIME": 30,
            "YELLOW_TIME": 3,
            "INTERVAL": 1.0,
            "TRAFFIC_FILE": resolved_traffic,
            "ROADNET_FILE": os.path.basename(roadnet_path),
            "LIST_STATE_FEATURE": ["cur_phase", "traffic_movement_pressure_queue"],
            "DIC_REWARD_INFO": {"queue_length": -0.25},
            "SAVE_REPLAY": False,
        },
    )

    # Prepare workspace
    work_dir = os.path.join("/tmp", f"collect_{dataset}_{int(time.time())}")
    log_dir = os.path.join(work_dir, "logs")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Copy roadnet and traffic files to workspace
    import shutil
    shutil.copy2(roadnet_path, os.path.join(work_dir, os.path.basename(roadnet_path)))
    traffic_src = os.path.join(traffic_root, resolved_traffic)
    if not os.path.exists(traffic_src):
        traffic_src = _resolve_path(resolved_traffic)
    shutil.copy2(traffic_src, os.path.join(work_dir, resolved_traffic))

    dic_path = {
        "PATH_TO_MODEL": work_dir,
        "PATH_TO_WORK_DIRECTORY": work_dir,
        "PATH_TO_DATA": work_dir,
    }

    env = CityFlowEnv(
        path_to_log=log_dir,
        path_to_work_directory=work_dir,
        dic_traffic_env_conf=env_conf,
        dic_path=dic_path,
    )

    # Load policy
    policy.load_checkpoint(ckpt_path, load_value_head=False)
    policy.eval()

    # Build movement keys
    movement_keys = ["NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL"]

    # We need traffic counts for TS2I — enable on the env
    env.traffic_count_enabled = True
    env.traffic_count_mode = "intersection_movement"
    env.traffic_count_intervals = [2]
    env.traffic_movement_keys = movement_keys
    env.traffic_intersection_ids = [
        inter.inter_name for inter in env.list_intersection
    ]
    env._init_traffic_counter()

    # Reset env
    state, _ = env.reset()

    # History buffer per intersection
    from collections import deque
    history_buffers: Dict[str, Any] = {}
    for inter in env.list_intersection:
        history_buffers[inter.inter_name] = deque(
            [[0.0] * len(movement_keys) for _ in range(seq_len)],
            maxlen=seq_len,
        )

    collected_states = []
    step_idx = 0
    done = False
    prev_time = 0.0

    while not done:
        if max_steps is not None and step_idx >= max_steps:
            break

        current_time = env.get_current_time()

        # Read traffic counts into history buffer
        if hasattr(env, "traffic_counts_by_interval") and 2 in env.traffic_counts_by_interval:
            counts_matrix = env.traffic_counts_by_interval[2]
            for inter_idx, inter in enumerate(env.list_intersection):
                inter_id = inter.inter_name
                offset = inter_idx * len(movement_keys)
                if counts_matrix.size > 0 and offset + len(movement_keys) <= counts_matrix.shape[1]:
                    # Read the latest bin
                    latest_bin = counts_matrix.shape[0] - 1
                    row = counts_matrix[latest_bin, offset:offset + len(movement_keys)]
                    history_buffers[inter_id].append([float(v) for v in row.tolist()])

        # Build observation
        sequences = []
        prompts = []
        phase_queues_all = []
        phase_waits_all = []

        for inter in env.list_intersection:
            inter_id = inter.inter_name
            history = list(history_buffers.get(inter_id, []))
            sequences.append(history)

            # Collect phase-level metrics
            phase_queue = {}
            phase_wait = {}
            lane_waiting = inter.dic_feature.get("lane_num_waiting_vehicle_in", [])
            for action_idx, action_name in enumerate(ACTION_LIST):
                phase_queue[action_name] = float(
                    sum(lane_waiting[i] for i in range(len(lane_waiting))
                        if i % len(ACTION_LIST) == action_idx)
                ) if lane_waiting else 0.0
                phase_wait[action_name] = 1.0  # placeholder — would need proper lane→phase mapping

            phase_queues_all.append(phase_queue)
            phase_waits_all.append(phase_wait)

            # Build prompt text
            traffic_context = build_phase_queue_wait_context(
                phase_queue=phase_queue,
                phase_wait=phase_wait,
                movement_queue_wait={},
            )
            prompt = build_task_prompt(
                "", mode=prompt_mode, action_tags=ACTION_LIST,
                traffic_context=traffic_context,
            )
            prompts.append(prompt)

        # Convert sequences to tensor and get policy action
        seq_tensor = torch.tensor(sequences, dtype=torch.float32, device=device)
        with torch.no_grad():
            act_out = policy.act(seq_tensor, prompts, sample=False)
        actions = act_out["actions"].tolist()

        # Record state BEFORE taking the action
        for inter_idx, inter in enumerate(env.list_intersection):
            inter_id = inter.inter_name
            state_record = {
                "step": step_idx,
                "time_s": float(current_time),
                "intersection_id": inter_id,
                "movement_counts": {
                    key: float(history_buffers[inter_id][-1][i])
                    for i, key in enumerate(movement_keys)
                } if history_buffers.get(inter_id) else {},
                "movement_keys": movement_keys,
                "prompt": prompts[inter_idx],
                "prompt_mode": prompt_mode,
                "policy_action": index_to_action(actions[inter_idx]),
                "policy_action_idx": actions[inter_idx],
                "phase_queue": phase_queues_all[inter_idx],
                "phase_wait": phase_waits_all[inter_idx],
                "traffic_file": resolved_traffic,
                "dataset": dataset,
            }
            collected_states.append(state_record)

        # Step the environment
        env.step(actions)
        step_idx += 1
        prev_time = current_time

        # Check done
        if current_time >= run_counts:
            done = True

    # Write output
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for state in collected_states:
            f.write(json.dumps(state, ensure_ascii=False) + "\n")

    print(f"Collected {len(collected_states)} states from {step_idx} steps")
    print(f"Saved to {output_path}")
    return collected_states


def main():
    parser = argparse.ArgumentParser(
        description="Collect decision-point states during online PhaseVLM evaluation"
    )
    parser.add_argument("--projection_ckpt", required=True, help="Path to Stage 2 best.pt")
    parser.add_argument("--llm_path", default="/home/data/model/qwen-vl-8B")
    parser.add_argument("--dataset", required=True, help="Dataset name: jinan, hangzhou, newyork_28x7")
    parser.add_argument("--traffic_file", required=True, help="Traffic flow file to collect on")
    parser.add_argument("--roadnet_file", default="", help="Roadnet file (auto-detected if empty)")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument("--run_counts", type=int, default=3600)
    parser.add_argument("--max_steps", type=int, default=None, help="Max decision steps (None = full episode)")
    parser.add_argument("--seq_len", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--periodicity", type=int, default=16)
    parser.add_argument("--input_dim", type=int, default=8)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--prefix_tokens", type=int, default=8)
    parser.add_argument("--prefix_mode", default="prefix")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--prompt_mode", default="queue_wait_short")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", default="cuda:0")

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.bfloat16 if args.bf16 and device.type == "cuda" else torch.float32

    print(f"Device: {device}, dtype: {torch_dtype}")

    # Build policy
    policy = PrefixActionPolicy(
        model_path=args.llm_path,
        seq_len=args.seq_len,
        image_size=args.image_size,
        periodicity=args.periodicity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens,
        prefix_mode=args.prefix_mode,
        max_length=args.max_length,
        device=device,
        with_value_head=False,
        torch_dtype=torch_dtype,
    )

    collect_states(
        policy=policy,
        ckpt_path=args.projection_ckpt,
        llm_path=args.llm_path,
        dataset=args.dataset,
        traffic_file=args.traffic_file,
        roadnet_file=args.roadnet_file,
        output_path=args.output,
        run_counts=args.run_counts,
        seq_len=args.seq_len,
        image_size=args.image_size,
        periodicity=args.periodicity,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        prefix_tokens=args.prefix_tokens,
        prefix_mode=args.prefix_mode,
        max_length=args.max_length,
        prompt_mode=args.prompt_mode,
        bf16=args.bf16,
        seed=args.seed,
        max_steps=args.max_steps,
        device=device,
    )


if __name__ == "__main__":
    main()
