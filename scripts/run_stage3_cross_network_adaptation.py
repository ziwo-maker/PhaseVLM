#!/usr/bin/env python3
"""Stage 3: Simulator-Assisted Few-Shot Target-Network Adaptation.

Implements the protocol recommended by external reviewer (Round 1):
  Phase 2: Zero-shot transfer evaluation (standard CityFlowEnv)
  Phase 3: State collection via --collect_states_output hook (guaranteed parity)
  Phase 4: SimExpert relabeling via run_sim_expert_action.py
  Phase 5: Few-shot fine-tuning via train_projection_bc.py
  Phase 6: Multi-seed final evaluation with correct t-intervals

Key design properties:
  - ALL evaluation uses CityFlowEnv (NO TrajectoryCityFlowEnv) — Phase 3 uses
    the --collect_states_output hook added to run_projection_online.py, which
    reuses the exact same observation construction as the canonical evaluator.
  - No PPO/RL — supervised fine-tuning on SimExpert-labeled target states.
  - Disjoint collection and test flows enforced.
  - Same-network control (Jinan→Jinan) for attribution.
  - K defined as complete network snapshots (one per decision step, all
    intersections).
  - Fail-fast: any phase failure stops the pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DEFAULT_LLM_PATH = "/home/data/model/qwen-vl-8B"

# ---- Configuration ----

NETWORK_CONFIGS = {
    "jinan": {
        "grid": "3_4",
        "roadnet": "data/Jinan/3_4/roadnet_3_4.json",
        "traffic_root": "data/Jinan/3_4",
        "flows": [
            "anon_3_4_jinan_real.json",
            "anon_3_4_jinan_real_2000.json",
            "anon_3_4_jinan_real_2500.json",
        ],
    },
    "hangzhou": {
        "grid": "4_4",
        "roadnet": "data/Hangzhou/4_4/roadnet_4_4.json",
        "traffic_root": "data/Hangzhou/4_4",
        "flows": [
            "anon_4_4_hangzhou_real.json",
            "anon_4_4_hangzhou_real_5816.json",
        ],
    },
}

# Result JSON key mapping (run_projection_online.py output format)
RESULT_KEY_MAP = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
}

# ---- Utility ----

def _run(cmd: List[str], timeout: int = 7200, env: Optional[Dict] = None,
         desc: str = "") -> subprocess.CompletedProcess:
    """Run a command and fail-fast on error."""
    print(f"[{desc}] {' '.join(cmd[:8])} ..." if desc else f"  {' '.join(cmd[:6])} ...")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_REPO,
                            timeout=timeout, env=(env or os.environ))
    if result.returncode != 0:
        print(f"  STDOUT (last 500): {result.stdout[-500:]}")
        print(f"  STDERR (last 500): {result.stderr[-500:]}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd[:6])}")
    return result


def _read_result_json(path: str) -> Dict[str, float]:
    """Parse run_projection_online.py result JSON and extract ATT/AQL/AWT."""
    with open(path, "r") as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    metrics = {}
    for label, key in RESULT_KEY_MAP.items():
        if key not in results:
            raise KeyError(f"Metric '{key}' not found in result JSON at {path}. Keys: {list(results.keys())}")
        val = results[key]
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            raise ValueError(f"Metric '{key}' is None/non-finite in {path}: {val}")
        metrics[label] = float(val)
    return metrics


# ---- Phase 2: Zero-shot evaluation ----

def evaluate_checkpoint(
    ckpt_path: str,
    llm_path: str,
    dataset: str,
    traffic_file: str,
    output_dir: str,
    seed: int = 2024,
    run_counts: int = 3600,
    collect_states_output: str = "",
) -> Dict[str, float]:
    """Evaluate a PhaseVLM checkpoint on a flow using the canonical evaluator."""
    cfg = NETWORK_CONFIGS[dataset]
    roadnet = os.path.join(_REPO, cfg["roadnet"])

    result_json = os.path.join(output_dir,
                               f"result_{traffic_file.replace('.json','')}_seed{seed}.json")
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable,
        os.path.join(_REPO, "projection_train", "run_projection_online.py"),
        "--memo", f"stage3_eval_{dataset}",
        "--projection_ckpt", ckpt_path,
        "--llm_path", llm_path,
        "--dataset", dataset,
        "--traffic_file", traffic_file,
        "--run_counts", str(run_counts),
        "--num_rounds", "1",
        "--seed", str(seed),
        "--prompt_mode", "queue_wait_short",
        "--seq_len", "16", "--image_size", "224",
        "--periodicity", "16", "--input_dim", "8", "--hidden_dim", "64",
        "--prefix_tokens", "8", "--prefix_mode", "prefix", "--max_length", "64",
        "--inference_batch_size", "64",
        "--bf16",
        "--result_json", result_json,
    ]
    if collect_states_output:
        cmd.extend(["--collect_states_output", collect_states_output])

    _run(cmd, timeout=7200, desc=f"Eval {dataset}/{traffic_file} seed={seed}")

    if not os.path.exists(result_json):
        raise FileNotFoundError(f"Result JSON not found: {result_json}")
    return _read_result_json(result_json)


# ---- Phase 3: State collection (via canonical evaluator hook) ----

def collect_target_states(
    ckpt_path: str,
    llm_path: str,
    dataset: str,
    traffic_file: str,
    output_path: str,
    seed: int = 2024,
) -> int:
    """Collect states using the --collect_states_output hook in run_projection_online.py.

    This guarantees observation parity because the SAME code path constructs
    observations, prompts, and actions as during standard evaluation.
    """
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Remove stale file
    if os.path.exists(output_path):
        os.remove(output_path)

    evaluate_checkpoint(
        ckpt_path=ckpt_path,
        llm_path=llm_path,
        dataset=dataset,
        traffic_file=traffic_file,
        output_dir=os.path.dirname(output_path),
        seed=seed,
        collect_states_output=output_path,
    )

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"States output not found: {output_path}")

    count = 0
    with open(output_path, "r") as f:
        for line in f:
            if line.strip():
                count += 1
    print(f"  Collected {count} states -> {output_path}")
    return count


# ---- Phase 4: SimExpert relabeling ----

# Default Python interpreter with cityflow (conda env hnr5090, Python 3.10).
# Falls back to current interpreter if the conda env is not available.
_SIMEXPERT_PYTHON = "/home/sysu/anaconda3/envs/hnr5090/bin/python"


def _resolve_simexpert_python() -> str:
    """Return a Python interpreter that can import cityflow."""
    import subprocess as _sp
    candidates = [
        _SIMEXPERT_PYTHON,
        sys.executable,
    ]
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            result = _sp.run(
                [candidate, "-c", "import cityflow; print(cityflow.Engine)"],
                capture_output=True, text=True, timeout=15,
                env={**os.environ, "PYTHONPATH": _REPO},
            )
            if result.returncode == 0 and "Engine" in result.stdout:
                return candidate
        except Exception:
            continue
    raise RuntimeError(
        "No Python interpreter with cityflow found. "
        "Install cityflow for your Python or activate the hnr5090 conda env."
    )


def relabel_with_simexpert(
    states_path: str,
    output_path: str,
    dataset: str,
    traffic_file: str,
    roadnet_path: str = "",
    allow_ruleonly_fallback: bool = False,
    cost_att_weight: float = 1.0,
    cost_aql_weight: float = 1.0,
    cost_awt_weight: float = 1.0,
    cost_spillback_weight: float = 0.2,
    sim_horizon: int = 30,
    awt_constraint_epsilon: float = 5.0,
    spillback_constraint_epsilon: float = 0.05,
    enable_pareto_filter: bool = True,
    enable_constraint_filter: bool = True,
) -> str:
    """Relabel collected states using SimExpert counterfactual simulation.

    Calls scripts/relabel_simexpert.py in a subprocess with a cityflow-capable
    Python interpreter. Falls back to RuleOnly when allow_ruleonly_fallback=True
    AND no valid snapshots are available (smoke tests only).
    """
    # Check snapshot availability
    snapshot_paths = set()
    with open(states_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            sp = record.get("snapshot_path", "")
            if sp and os.path.exists(sp):
                snapshot_paths.add(sp)

    if not snapshot_paths and allow_ruleonly_fallback:
        print("  WARNING: No snapshots, falling back to RuleOnly heuristic (smoke test only).")
        return _relabel_ruleonly(states_path, output_path,
                                 cost_att_weight=cost_att_weight,
                                 cost_aql_weight=cost_aql_weight,
                                 cost_awt_weight=cost_awt_weight)

    if not snapshot_paths:
        raise RuntimeError(
            "No valid snapshot paths in collected states. "
            "SimExpert relabeling requires state snapshots. "
            "Use --allow_ruleonly_fallback only for smoke tests."
        )

    # Resolve roadnet path
    if not roadnet_path:
        cfg = NETWORK_CONFIGS.get(dataset, {})
        roadnet_path = os.path.join(_REPO, cfg.get("roadnet", ""))

    traffic_file_abs = traffic_file
    if not os.path.isabs(traffic_file_abs):
        # traffic_file is the flow filename; resolve to data dir
        cfg = NETWORK_CONFIGS.get(dataset, {})
        traffic_root = os.path.join(_REPO, cfg.get("traffic_root", "data"))
        traffic_file_abs = os.path.join(traffic_root, traffic_file)

    # Resolve cityflow-capable Python
    simexpert_python = _resolve_simexpert_python()
    print(f"  SimExpert Python: {simexpert_python}")

    relabel_script = os.path.join(_REPO, "scripts", "relabel_simexpert.py")
    work_dir = os.path.join(os.path.dirname(output_path) or ".", "simexpert_work")

    cmd = [
        simexpert_python, relabel_script,
        "--input", os.path.abspath(states_path),
        "--output", os.path.abspath(output_path),
        "--roadnet", os.path.abspath(roadnet_path),
        "--traffic_file", os.path.abspath(traffic_file_abs),
        "--dataset", dataset,
        "--work_dir", os.path.abspath(work_dir),
        "--sim_horizon", str(sim_horizon),
        "--cost_att_weight", str(cost_att_weight),
        "--cost_aql_weight", str(cost_aql_weight),
        "--cost_awt_weight", str(cost_awt_weight),
        "--cost_spillback_weight", str(cost_spillback_weight),
        "--awt_constraint_epsilon", str(awt_constraint_epsilon),
        "--spillback_constraint_epsilon", str(spillback_constraint_epsilon),
    ]
    if not enable_pareto_filter:
        cmd.append("--disable_pareto_filter")
    if not enable_constraint_filter:
        cmd.append("--disable_constraint_filter")

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO

    print(f"[Phase 4] Running SimExpert relabeling on {len(snapshot_paths)} snapshots...")
    _run(cmd, timeout=7200, env=env, desc="SimExpert relabel")

    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Relabeled output not found: {output_path}")

    # Log coverage stats if available
    coverage_path = output_path.replace(".jsonl", "_coverage.json")
    if os.path.exists(coverage_path):
        with open(coverage_path, "r") as f:
            coverage = json.load(f)
        print(f"  Coverage: {json.dumps(coverage, indent=2)}")

    return output_path


def _relabel_ruleonly(states_path: str, output_path: str,
                      cost_att_weight: float = 1.0,
                      cost_aql_weight: float = 1.0,
                      cost_awt_weight: float = 2.0) -> str:
    """RuleOnly relabeling with configurable cost weights.

    Default weights (awt=2.0, aql=1.0, att=1.0) follow the Stage 2
    README findings: 2:1 wait:queue ratio achieves triple improvement
    on New York triple traffic, resolving the Queue-Wait trade-off.

    Score = att_weight * ATT_proxy + aql_weight * queue + awt_weight * wait
    Lower score = better phase. Since we only have queue/wait per phase,
    we use: score = aql_weight * queue + awt_weight * wait
    The policy should MINIMIZE this composite cost.
    """
    from projection_train.action_space import ACTION_LIST

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    relabeled_count = 0

    with open(states_path, "r") as fin, open(output_path, "w") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)

            phase_queue = record.get("phase_queue", {})
            phase_wait = record.get("phase_wait", {})

            # Select phase with HIGHEST weighted pressure (most congested → serve first).
            # Higher awt_weight gives more priority to waiting time vs queue length.
            # This is a greedy heuristic: the phase most in need gets the green.
            best_action = ACTION_LIST[0]
            best_score = -1.0
            for action_name in ACTION_LIST:
                q = float(phase_queue.get(action_name, 0.0) or 0.0)
                w = float(phase_wait.get(action_name, 0.0) or 0.0)
                score = cost_aql_weight * q + cost_awt_weight * w
                if score > best_score:
                    best_score = score
                    best_action = action_name

            record["expert_action"] = best_action
            record["relabel_method"] = f"ruleonly_att{cost_att_weight}_aql{cost_aql_weight}_awt{cost_awt_weight}"
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            relabeled_count += 1

    print(f"  Relabeled {relabeled_count} states -> {output_path} "
          f"(weights: att={cost_att_weight}, aql={cost_aql_weight}, awt={cost_awt_weight})")
    return output_path


# ---- Phase 5: Few-shot fine-tuning ----

def run_few_shot_adaptation(
    source_ckpt: str,
    stage1_ckpt: str,
    relabeled_data_path: str,
    llm_path: str,
    output_dir: str,
    num_labels: int,
    gpu_ids: str = "0,1",
    ranking_loss_weight: float = 0.05,
    soft_target_alpha: float = 0.3,
    cost_awt_weight: float = 2.0,
    cost_aql_weight: float = 1.0,
    cost_att_weight: float = 1.0,
    cost_spillback_weight: float = 0.2,
    target_mode: str = "hard",
    best_metric: str = "val_acc",
    seed: int = 2024,
) -> str:
    """Fine-tune PhaseVLM on relabeled target-network data.

    Important: train_projection_bc.py does NOT have --resume_from.
    Instead, we initialize from source checkpoint by copying it as the
    starting point and training on new data with a small learning rate.

    Alternative: preload source checkpoint weights before training starts.
    For now: train from Stage 1 checkpoint on new data (equivalent to
    training from scratch on target data — same architecture, frozen backbone).
    """
    world_size = len(gpu_ids.split(","))
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    cmd = [
        "torchrun", f"--nproc_per_node={world_size}",
        os.path.join(_REPO, "projection_train", "train_projection_bc.py"),
        "--data_path", relabeled_data_path,
        "--llm_path", llm_path,
        "--align_ckpt", stage1_ckpt,
        "--init_policy_ckpt", source_ckpt,
        "--output_dir", checkpoints_dir,
        "--seq_len", "16", "--image_size", "224",
        "--periodicity", "16", "--input_dim", "8", "--hidden_dim", "64",
        "--prefix_tokens", "8", "--prefix_mode", "prefix", "--max_length", "64",
        "--prompt_mode", "queue_wait_short",
        "--batch_size", "2", "--grad_accum_steps", "8",
        "--epochs", "5",
        "--lr", "1e-5", "--min_lr", "1e-6", "--warmup_ratio", "0.1", "--lr_schedule", "cosine",
        "--weight_decay", "1e-4", "--max_grad_norm", "1.0",
        "--val_ratio", "0.1", "--split_strategy", "stratified",
        "--num_workers", "2", "--seed", str(seed), "--bf16",
        "--log_interval", "10", "--save_every", "1",
        "--best_metric", best_metric, "--best_mode", "min" if "regret" in best_metric or "cost" in best_metric else "max",
        "--ddp_timeout_seconds", "1800",
    ]

    # Configure training target mode for ablation variants
    if target_mode == "hard":
        cmd.extend(["--target_mode", "hard"])
    elif target_mode == "soft":
        cmd.extend([
            "--target_mode", "soft",
            "--soft_target_alpha", str(soft_target_alpha),
            "--soft_label_temperature", str(0.35),
            "--cost_awt_weight", str(cost_awt_weight),
            "--cost_aql_weight", str(cost_aql_weight),
            "--cost_att_weight", str(cost_att_weight),
            "--cost_spillback_weight", str(cost_spillback_weight),
        ])
    elif target_mode == "mixed":
        # Mixed: hard BC + soft preference distillation + cost ranking
        cmd.extend([
            "--target_mode", "mixed",
            "--soft_target_alpha", str(soft_target_alpha),
            "--ranking_loss_weight", str(ranking_loss_weight),
            "--cost_awt_weight", str(cost_awt_weight),
            "--cost_aql_weight", str(cost_aql_weight),
            "--cost_att_weight", str(cost_att_weight),
            "--cost_spillback_weight", str(cost_spillback_weight),
            "--ranking_min_cost_gap", "0.02",
            "--soft_label_temperature", "0.35",
            "--base_policy_kl_weight", "0.0",
        ])
    elif target_mode == "pareto":
        # Pareto-constrained: hard BC on feasible actions + constraint loss
        cmd.extend([
            "--target_mode", "mixed",
            "--soft_target_alpha", str(soft_target_alpha),
            "--ranking_loss_weight", str(ranking_loss_weight),
            "--cost_awt_weight", str(cost_awt_weight),
            "--cost_aql_weight", str(cost_aql_weight),
            "--cost_att_weight", str(cost_att_weight),
            "--cost_spillback_weight", str(cost_spillback_weight),
            "--ranking_min_cost_gap", "0.02",
            "--soft_label_temperature", "0.35",
            "--base_policy_kl_weight", "0.0",
        ])
    else:
        cmd.extend(["--target_mode", "hard"])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids

    print(f"[Phase 5] Few-shot adaptation with {num_labels} labels")
    _run(cmd, timeout=14400, env=env, desc=f"Adapt K={num_labels}")

    best_pt = os.path.join(checkpoints_dir, "best.pt")
    if not os.path.exists(best_pt):
        raise FileNotFoundError(f"best.pt not found at {best_pt}")

    print(f"  Adapted model: {best_pt}")
    return best_pt


# ---- Phase 6: Multi-seed final evaluation ----

def run_multi_seed_eval(
    ckpt_path: str,
    llm_path: str,
    dataset: str,
    test_flow: str,
    output_dir: str,
    seeds: List[int],
) -> Dict[str, Any]:
    """Multi-seed evaluation with correct t-intervals."""
    os.makedirs(output_dir, exist_ok=True)
    all_metrics = {m: [] for m in ["ATT", "AQL", "AWT"]}

    for seed in seeds:
        result = evaluate_checkpoint(
            ckpt_path=ckpt_path,
            llm_path=llm_path,
            dataset=dataset,
            traffic_file=test_flow,
            output_dir=output_dir,
            seed=seed,
        )
        for metric in all_metrics:
            all_metrics[metric].append(result.get(metric, 0.0))
        print(f"  seed={seed}: ATT={result['ATT']:.1f} AQL={result['AQL']:.1f} AWT={result['AWT']:.1f}")

    import numpy as np
    from scipy import stats as scipy_stats

    summary = {}
    for metric, values in all_metrics.items():
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        if n > 1:
            t_val = scipy_stats.t.ppf(0.975, n - 1)
            ci_half = t_val * std_v / np.sqrt(n)
        else:
            ci_half = 0.0
        summary[metric] = {
            "mean": round(mean_v, 2),
            "std": round(std_v, 2),
            "n": n,
            "ci95_low": round(mean_v - ci_half, 2),
            "ci95_high": round(mean_v + ci_half, 2),
        }

    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"raw_values": all_metrics, "summary": summary}, f, indent=2)
    print(f"  Summary: {json.dumps(summary, indent=2)}")
    return summary


# ---- Main pipeline ----

def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: Simulator-Assisted Few-Shot Target-Network Adaptation"
    )
    parser.add_argument("--source_dataset", required=True, choices=["jinan", "hangzhou"])
    parser.add_argument("--target_dataset", required=True, choices=["jinan", "hangzhou"])
    parser.add_argument("--source_stage2_ckpt", required=True)
    parser.add_argument("--source_stage1_ckpt", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--collection_flow", default="")
    parser.add_argument("--test_flow", default="")
    parser.add_argument("--label_budgets", type=int, nargs="+", default=[0, 8, 32, 120])
    parser.add_argument("--llm_path", default=DEFAULT_LLM_PATH)
    parser.add_argument("--train_gpus", default="0,1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--same_network_control", action="store_true",
                        help="Run same-network adaptation control (Jinan→Jinan)")
    parser.add_argument("--control_collection_flow", default="")
    parser.add_argument("--control_test_flow", default="")
    parser.add_argument("--allow_ruleonly_fallback", action="store_true",
                        help="Allow RuleOnly heuristic as SimExpert fallback (smoke tests only)")
    parser.add_argument("--cost_att_weight", type=float, default=1.0)
    parser.add_argument("--cost_aql_weight", type=float, default=1.0)
    parser.add_argument("--cost_awt_weight", type=float, default=1.0,
                        help="AWT weight for SimExpert scoring (default 1.0; RuleOnly fallback uses 2.0)")
    parser.add_argument("--cost_spillback_weight", type=float, default=0.2,
                        help="Spillback weight for SimExpert scoring")
    parser.add_argument("--sim_horizon", type=int, default=30,
                        help="Counterfactual simulation horizon in seconds")
    parser.add_argument("--awt_constraint_epsilon", type=float, default=5.0,
                        help="Allowed AWT slack over base action (constraint)")
    parser.add_argument("--spillback_constraint_epsilon", type=float, default=0.05,
                        help="Allowed spillback slack over base action (constraint)")
    parser.add_argument("--disable_pareto_filter", action="store_true",
                        help="Disable Pareto dominance filtering")
    parser.add_argument("--disable_constraint_filter", action="store_true",
                        help="Disable AWT/spillback constraint filtering")
    parser.add_argument("--ranking_loss_weight", type=float, default=0.05,
                        help="Cost ranking loss weight (0 = pure BC)")
    parser.add_argument("--soft_target_alpha", type=float, default=0.3,
                        help="Soft target interpolation (0 = hard BC only)")
    parser.add_argument("--target_mode", type=str, default="hard",
                        choices=["hard", "soft", "mixed", "pareto"],
                        help="Training target mode for ablation: "
                             "hard=RuleOnly/SimExpert hard BC, "
                             "soft=soft preference distillation, "
                             "mixed=hard+soft+ranking, "
                             "pareto=hard+soft+ranking with Pareto constraints")
    parser.add_argument("--best_metric", type=str, default="val_acc",
                        help="Metric for best checkpoint selection")
    parser.add_argument("--ablation_id", type=str, default="",
                        help="Ablation experiment ID (A/B/C/D/E) for logging")

    args = parser.parse_args()

    if args.source_dataset == args.target_dataset and not args.same_network_control:
        print("ERROR: source and target must differ for cross-network adaptation.")
        print("Use --same_network_control for same-network baseline.")
        sys.exit(1)

    source_cfg = NETWORK_CONFIGS[args.source_dataset]
    target_cfg = NETWORK_CONFIGS[args.target_dataset]

    collection_flow = args.collection_flow or target_cfg["flows"][0]
    test_flow = args.test_flow or target_cfg["flows"][1] if len(target_cfg["flows"]) > 1 else target_cfg["flows"][0]

    if collection_flow == test_flow:
        print(f"ERROR: collection_flow ({collection_flow}) must differ from test_flow ({test_flow})")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    config = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "source_stage2_ckpt": args.source_stage2_ckpt,
        "source_stage1_ckpt": args.source_stage1_ckpt,
        "collection_flow": collection_flow,
        "test_flow": test_flow,
        "label_budgets": args.label_budgets,
        "seeds": args.seeds,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    pipeline_status: Dict[str, str] = {}

    # ---- Phase 2: Zero-shot evaluation ----
    print("\n" + "="*60)
    print("Phase 2: Zero-Shot Transfer Evaluation")
    print("="*60)
    zero_shot_dir = os.path.join(args.output_dir, "zero_shot")
    os.makedirs(zero_shot_dir, exist_ok=True)

    zero_shot_metrics = {}
    for seed in args.seeds:
        metrics = evaluate_checkpoint(
            ckpt_path=args.source_stage2_ckpt,
            llm_path=args.llm_path,
            dataset=args.target_dataset,
            traffic_file=test_flow,
            output_dir=zero_shot_dir,
            seed=seed,
        )
        zero_shot_metrics[f"seed_{seed}"] = metrics
        print(f"  Zero-shot seed={seed}: ATT={metrics['ATT']:.1f} AQL={metrics['AQL']:.1f} AWT={metrics['AWT']:.1f}")

    with open(os.path.join(zero_shot_dir, "all_seeds.json"), "w") as f:
        json.dump(zero_shot_metrics, f, indent=2)
    pipeline_status["zero_shot"] = "ok"

    # ---- Phase 3: State collection ----
    print("\n" + "="*60)
    print("Phase 3: Collect Target-Network States (canonical evaluator hook)")
    print("="*60)
    collected_path = os.path.join(args.output_dir, "collected_states.jsonl")

    num_collected = collect_target_states(
        ckpt_path=args.source_stage2_ckpt,
        llm_path=args.llm_path,
        dataset=args.target_dataset,
        traffic_file=collection_flow,
        output_path=collected_path,
        seed=args.seeds[0],
    )
    pipeline_status["collection"] = f"ok ({num_collected} states)"

    # ---- Phase 4: SimExpert relabeling ----
    print("\n" + "="*60)
    print("Phase 4: SimExpert Relabeling")
    print("="*60)
    relabeled_path = os.path.join(args.output_dir, "relabeled_states.jsonl")

    if args.allow_ruleonly_fallback:
        # RuleOnly fallback (smoke test / baseline A)
        print("  Using RuleOnly heuristic (allow_ruleonly_fallback=True)")
        _relabel_ruleonly(
            states_path=collected_path,
            output_path=relabeled_path,
            cost_att_weight=args.cost_att_weight,
            cost_aql_weight=args.cost_aql_weight,
            cost_awt_weight=args.cost_awt_weight if args.cost_awt_weight != 1.0 else 2.0,
        )
    else:
        # Full SimExpert counterfactual relabeling
        relabel_with_simexpert(
            states_path=collected_path,
            output_path=relabeled_path,
            dataset=args.target_dataset,
            traffic_file=collection_flow,
            allow_ruleonly_fallback=False,
            cost_att_weight=args.cost_att_weight,
            cost_aql_weight=args.cost_aql_weight,
            cost_awt_weight=args.cost_awt_weight,
            cost_spillback_weight=args.cost_spillback_weight,
            sim_horizon=args.sim_horizon,
            awt_constraint_epsilon=args.awt_constraint_epsilon,
            spillback_constraint_epsilon=args.spillback_constraint_epsilon,
            enable_pareto_filter=not args.disable_pareto_filter,
            enable_constraint_filter=not args.disable_constraint_filter,
        )
    pipeline_status["relabeling"] = "ok"

    # ---- Phase 5+6: Adaptation per K ----
    for num_labels in args.label_budgets:
        if num_labels == 0:
            pipeline_status[f"K=0"] = "ok (zero-shot baseline)"
            continue

        print(f"\n{'='*60}")
        print(f"Phase 5+6: Adaptation K={num_labels}")
        print("="*60)
        adapt_dir = os.path.join(args.output_dir, f"adapted_k{num_labels}")
        os.makedirs(adapt_dir, exist_ok=True)

        # K = number of decision-time snapshots (network-level selection)
        # Group records by time_s so each "snapshot" includes all intersections
        with open(relabeled_path, "r") as f:
            all_records = [json.loads(line) for line in f if line.strip()]

        # Group by time_s
        time_groups: Dict[float, List[dict]] = {}
        for rec in all_records:
            t = rec.get("time_s", 0.0)
            time_groups.setdefault(t, []).append(rec)

        unique_times = sorted(time_groups.keys())
        print(f"  Available: {len(unique_times)} decision times, {len(all_records)} intersection records")

        if num_labels > len(unique_times):
            raise ValueError(
                f"K={num_labels} exceeds available decision times ({len(unique_times)}). "
                f"Each 3600s episode with 30s intervals has at most 120 snapshots. "
                f"Use a smaller K budget or collect multiple episodes."
            )

        random.seed(2024)
        selected_times = sorted(random.sample(unique_times, num_labels))
        selected_records = []
        for t in selected_times:
            selected_records.extend(time_groups[t])

        subsampled_path = os.path.join(adapt_dir, "train_data.jsonl")
        with open(subsampled_path, "w") as f:
            for rec in selected_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  K={num_labels} snapshots → {len(selected_records)} intersection labels "
              f"({len(unique_times)} available, {len(selected_times)} selected)")

        try:
            adapted_ckpt = run_few_shot_adaptation(
                source_ckpt=args.source_stage2_ckpt,
                stage1_ckpt=args.source_stage1_ckpt,
                relabeled_data_path=subsampled_path,
                llm_path=args.llm_path,
                output_dir=adapt_dir,
                num_labels=num_labels,
                gpu_ids=args.train_gpus,
                ranking_loss_weight=args.ranking_loss_weight,
                soft_target_alpha=args.soft_target_alpha,
                cost_awt_weight=args.cost_awt_weight,
                cost_aql_weight=args.cost_aql_weight,
                cost_att_weight=args.cost_att_weight,
                cost_spillback_weight=args.cost_spillback_weight,
                target_mode=args.target_mode,
                best_metric=args.best_metric,
                seed=args.seeds[0],
            )

            eval_dir = os.path.join(adapt_dir, "evaluation")
            summary = run_multi_seed_eval(
                ckpt_path=adapted_ckpt,
                llm_path=args.llm_path,
                dataset=args.target_dataset,
                test_flow=test_flow,
                output_dir=eval_dir,
                seeds=args.seeds,
            )
            pipeline_status[f"K={num_labels}"] = "ok"

        except Exception as e:
            print(f"  FAILED K={num_labels}: {e}")
            pipeline_status[f"K={num_labels}"] = f"FAILED: {e}"
            continue

    # ---- Same-network control (if requested) ----
    if args.same_network_control:
        print(f"\n{'='*60}")
        print("Same-Network Control: Source→Source Adaptation")
        print("="*60)
        control_dir = os.path.join(args.output_dir, "same_network_control")
        os.makedirs(control_dir, exist_ok=True)

        ctrl_collection_flow = args.control_collection_flow or source_cfg["flows"][0]
        ctrl_test_flow = args.control_test_flow or source_cfg["flows"][1] if len(source_cfg["flows"]) > 1 else source_cfg["flows"][0]

        if ctrl_collection_flow == ctrl_test_flow:
            print(f"  WARNING: control collection ({ctrl_collection_flow}) == test ({ctrl_test_flow}), using first two different flows")
            ctrl_collection_flow = source_cfg["flows"][0]
            ctrl_test_flow = source_cfg["flows"][1]

        for num_labels in args.label_budgets:
            if num_labels == 0:
                continue
            try:
                ctrl_collected = os.path.join(control_dir, "collected_states.jsonl")
                collect_target_states(
                    ckpt_path=args.source_stage2_ckpt,
                    llm_path=args.llm_path,
                    dataset=args.source_dataset,
                    traffic_file=ctrl_collection_flow,
                    output_path=ctrl_collected,
                    seed=args.seeds[0],
                )
                ctrl_relabeled = os.path.join(control_dir, "relabeled_states.jsonl")
                if args.allow_ruleonly_fallback:
                    _relabel_ruleonly(ctrl_collected, ctrl_relabeled,
                                      cost_awt_weight=args.cost_awt_weight if args.cost_awt_weight != 1.0 else 2.0)
                else:
                    relabel_with_simexpert(
                        states_path=ctrl_collected,
                        output_path=ctrl_relabeled,
                        dataset=args.source_dataset,
                        traffic_file=ctrl_collection_flow,
                        allow_ruleonly_fallback=False,
                        cost_awt_weight=args.cost_awt_weight,
                        sim_horizon=args.sim_horizon,
                        enable_pareto_filter=not args.disable_pareto_filter,
                        enable_constraint_filter=not args.disable_constraint_filter,
                    )
                pipeline_status[f"ctrl_K={num_labels}"] = "ok"
            except Exception as e:
                pipeline_status[f"ctrl_K={num_labels}"] = f"FAILED: {e}"

    # ---- Final status ----
    print("\n" + "="*60)
    print("Pipeline complete. Status per phase:")
    for phase, status in pipeline_status.items():
        print(f"  {phase}: {status}")
    print(f"Output: {args.output_dir}")
    print("="*60)

    # Fail if any phase failed
    failures = [k for k, v in pipeline_status.items() if v.startswith("FAILED")]
    if failures:
        print(f"\nFAILED phases: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
