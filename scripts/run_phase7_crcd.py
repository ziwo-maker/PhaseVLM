#!/usr/bin/env python3
"""Phase 7 (CRCD) orchestrator: teacher-variant dataset build -> train -> online eval.

Implements the Phase 7 evaluation (refine-logs/EXPERIMENT_FINAL_REPORT.md) on
the Jinan -> Hangzhou Stage-3 task (pt32/hd256), comparing teacher variants on
identical selected snapshots:

  c30        30s frozen (existing teacher baseline)
  c300       300s frozen (longer horizon, no gate)
  crcd_nogate multi-horizon majority vote, no gate
  crcd       reliability gate + KL anchor (full CRCD)

Each (teacher, K, training_seed) builds a snapshot-level train/val split, runs
5-epoch BC fine-tune warm-started from the source Stage-2 checkpoint, and
evaluates online (3 simulator seeds). Training-replicate variance is aggregated.

Reuses:
  - projection_train/crcd_relabel.py  (gate + dataset build, pure python)
  - projection_train/train_projection_bc.py  (torchrun)
  - projection_train/run_projection_online.py  (canonical CityFlowEnv)

Usage:
  python3 scripts/run_phase7_crcd.py \
      --f30 .../ood/k118/relabeled_states.jsonl \
      --f120 .../teacher_validity_20260813/relabeled_h120.jsonl \
      --f300 .../teacher_validity_20260813/relabeled_h300.jsonl \
      --output_dir records/pipeline_runs/phase7_crcd_20260813 \
      --teachers c30 c300 crcd_nogate crcd \
      --label_budgets 8 32 118 \
      --training_seeds 2024 42 123
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
import zlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from projection_train import crcd_relabel  # noqa: E402

DEFAULT_LLM_PATH = "/home/data/model/qwen-vl-8B"
_CONDA_PYTHON = "/home/sysu/anaconda3/envs/hnr5090/bin/python"

# Source checkpoints (Jinan pt32/hd256).
SOURCE_STAGE2_CKPT = "records/pipeline_runs/stage2_pt32_hd256_20260812/best.pt"
SOURCE_STAGE1_CKPT = "records/pipeline_runs/stage1_pt32_hd256_20260812/best.pt"

# Hangzhou target.
DATASET = "hangzhou"
TEST_FLOW = "anon_4_4_hangzhou_real_5816.json"

ARCH_ARGS = {
    "seq_len": 16,
    "image_size": 224,
    "periodicity": 16,
    "input_dim": 8,
    "hidden_dim": 256,
    "prefix_tokens": 32,
    "prefix_mode": "prefix",
    "max_length": 64,
}

RESULT_KEY_MAP = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
}

# Teacher -> (base_policy_kl_weight, base_policy_kl_mode)
TEACHER_KL: Dict[str, tuple[float, str]] = {
    "c30": (0.0, "expert_match"),
    "c300": (0.0, "expert_match"),
    "crcd_nogate": (0.0, "expert_match"),
    "crcd": (0.5, "expert_match"),
    "prcat": (0.5, "low_weight"),
}

# Loss-config presets (Phase 7 follow-up: isolate the soft/ranking loss).
LOSS_ARGS: Dict[str, Dict[str, str]] = {
    "mixed": {
        "target_mode": "mixed",
        "soft_target_alpha": "0.3",
        "soft_label_temperature": "0.35",
        "ranking_loss_weight": "0.05",
        "ranking_min_cost_gap": "0.02",
    },
    "hard": {
        "target_mode": "hard",
        "soft_target_alpha": "0.3",
        "soft_label_temperature": "0.35",
        "ranking_loss_weight": "0.0",
        "ranking_min_cost_gap": "0.02",
    },
}


def _resolve_python() -> str:
    for candidate in (_CONDA_PYTHON, sys.executable):
        if not os.path.isfile(candidate):
            continue
        try:
            result = subprocess.run(
                [candidate, "-c", "import cityflow; print(cityflow.Engine)"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode == 0 and "Engine" in result.stdout:
                return candidate
        except Exception:
            continue
    raise RuntimeError(f"No Python with a real cityflow found (tried {_CONDA_PYTHON}).")


def _env(python: str, gpus: str = "") -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO
    env["PATH"] = os.path.dirname(python) + os.pathsep + env.get("PATH", "")
    # Fragmentation-resistant CUDA allocator: ml416 sequences make training peak
    # right at the 31GB edge; expandable_segments avoids the "38MiB alloc fails
    # with 2GiB reserved" fragmentation OOM.
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = gpus
    return env


def _run(cmd: List[str], timeout: int, env: Dict, log_path: str, desc: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as lf:
        lf.write(f"\n===== {desc} @ {datetime.now().isoformat()} =====\n")
        lf.write(" ".join(cmd) + "\n")
        result = subprocess.run(cmd, cwd=_REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed (exit {result.returncode}); log: {log_path}")


def _checkpoint_complete(best_pt: str) -> bool:
    """A complete best.pt is ~the (slim) model-param size; a truncated write
    (process killed mid-save) is much smaller. Compare against the Stage-1
    checkpoint, which is always slim (no optimizer state) like the trained
    checkpoints produced with --checkpoint_slim."""
    if not os.path.exists(best_pt):
        return False
    src = os.path.join(_REPO, SOURCE_STAGE1_CKPT)
    if not os.path.exists(src):
        return True  # cannot compare; trust existence
    return os.path.getsize(best_pt) >= 0.9 * os.path.getsize(src)


def _read_result_json(path: str) -> Dict[str, float]:
    with open(path) as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    metrics = {}
    for label, key in RESULT_KEY_MAP.items():
        if key not in results:
            raise KeyError(f"Metric '{key}' not in {path}. Keys: {list(results.keys())}")
        metrics[label] = float(results[key])
    return metrics


def build_dataset(
    aligned, snapshot_order, teacher, K, selection_seed, training_seed, adv_margin,
    unreliable_weight, out_dir,
) -> Dict:
    params = {
        "adv_margin": adv_margin,
        "unreliable_weight": unreliable_weight,
        "require_through": True,
        "require_feasible": True,
        "prcat_min_weight": 0.15,
    }
    return crcd_relabel.build_dataset(
        aligned, snapshot_order, teacher, K, selection_seed, training_seed,
        val_ratio=0.1, params=params, out_dir=out_dir,
    )


def train(
    python: str, train_data: str, val_data: str, ckpt_dir: str, training_seed: int,
    kl_weight: float, kl_mode: str, train_gpus: str, loss: str = "mixed",
) -> str:
    world_size = len([g for g in train_gpus.split(",") if g.strip()])
    torchrun = os.path.join(os.path.dirname(python), "torchrun")
    if not os.path.isfile(torchrun):
        raise RuntimeError(f"torchrun not found next to interpreter: {torchrun}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # Unique rendezvous port per config so a leftover training never collides.
    master_port = 29500 + (zlib.crc32(ckpt_dir.encode()) % 5000)
    cmd = [
        torchrun, f"--nproc_per_node={world_size}", f"--master_port={master_port}",
        os.path.join(_REPO, "projection_train", "train_projection_bc.py"),
        "--train_data_path", train_data,
        "--val_data_path", val_data,
        "--llm_path", DEFAULT_LLM_PATH,
        "--align_ckpt", os.path.join(_REPO, SOURCE_STAGE1_CKPT),
        "--init_policy_ckpt", os.path.join(_REPO, SOURCE_STAGE2_CKPT),
        "--output_dir", ckpt_dir,
        "--prompt_mode", "queue_wait_short",
        "--batch_size", "1", "--grad_accum_steps", "16",
        "--epochs", "5",
        "--lr", "1e-5", "--min_lr", "1e-6", "--warmup_ratio", "0.1",
        "--lr_schedule", "cosine",
        "--weight_decay", "1e-4", "--max_grad_norm", "1.0",
        "--num_workers", "2", "--seed", str(training_seed), "--bf16",
        "--log_interval", "10", "--save_every", "100", "--checkpoint_slim",
    ]
    for key, val in LOSS_ARGS[loss].items():
        cmd.extend([f"--{key}", val])
    cmd.extend([
        "--cost_att_weight", "1.0", "--cost_aql_weight", "1.0",
        "--cost_awt_weight", "1.0", "--cost_spillback_weight", "0.2",
        "--base_policy_kl_weight", str(kl_weight),
        "--base_policy_kl_mode", kl_mode,
        "--base_policy_kl_low_weight_threshold", "0.3" if kl_mode == "low_weight" else "1.0",
        "--best_metric", "val_top1_cost_regret", "--best_mode", "min",
        "--ddp_timeout_seconds", "1800",
    ])
    for key, val in ARCH_ARGS.items():
        cmd.extend([f"--{key}", str(val)])
    _run(cmd, timeout=14400, env=_env(python, train_gpus),
         log_path=os.path.join(ckpt_dir, "train.log"), desc=f"train seed={training_seed}")
    best_pt = os.path.join(ckpt_dir, "best.pt")
    if not os.path.exists(best_pt):
        raise RuntimeError(f"training done but best.pt missing at {best_pt}")
    return best_pt


def eval_online(
    python: str, ckpt: str, seeds: Sequence[int], eval_gpus: Sequence[int], eval_dir: str,
    dataset: str = DATASET, test_flow: str = TEST_FLOW,
) -> Dict[str, float]:
    os.makedirs(eval_dir, exist_ok=True)
    outcomes = {}
    if not eval_gpus:
        raise ValueError("eval_gpus must contain at least one GPU id")
    for batch_start in range(0, len(seeds), len(eval_gpus)):
        batch = list(seeds[batch_start: batch_start + len(eval_gpus)])
        info = {}
        for seed, gpu in zip(batch, eval_gpus):
            result_json = os.path.join(eval_dir, f"result_seed{seed}.json")
            memo = f"phase7_{uuid.uuid4().hex[:8]}"
            cmd = [
                python,
                os.path.join(_REPO, "projection_train", "run_projection_online.py"),
                "--memo", memo,
                "--projection_ckpt", ckpt,
                "--llm_path", DEFAULT_LLM_PATH,
                "--dataset", dataset,
                "--traffic_file", test_flow,
                "--run_counts", "3600",
                "--num_rounds", "1",
                "--seed", str(seed),
                "--prompt_mode", "queue_wait_short",
                "--inference_batch_size", "64",
                "--bf16",
                "--result_json", result_json,
            ]
            for key, val in ARCH_ARGS.items():
                cmd.extend([f"--{key}", str(val)])
            env = _env(python, str(gpu))
            log_path = os.path.join(eval_dir, f"eval_seed{seed}.log")
            lf = open(log_path, "a", encoding="utf-8")
            lf.write(f"\n===== seed={seed} gpu={gpu} @ {datetime.now().isoformat()} =====\n")
            lf.write(" ".join(cmd) + "\n")
            proc = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=lf, stderr=subprocess.STDOUT, text=True)
            lf.close()
            info[seed] = {"proc": proc, "result_json": result_json}
            print(f"    eval seed={seed} on GPU {gpu} (pid {proc.pid})")

        for seed, d in info.items():
            rc = d["proc"].wait()
            if rc != 0 or not os.path.exists(d["result_json"]):
                raise RuntimeError(f"eval seed={seed} failed (rc={rc}); log: {eval_dir}/eval_seed{seed}.log")
            outcomes[f"seed_{seed}"] = _read_result_json(d["result_json"])
    mean = {m: sum(v[m] for v in outcomes.values()) / len(outcomes) for m in RESULT_KEY_MAP}
    with open(os.path.join(eval_dir, "eval_summary.json"), "w") as f:
        json.dump({"mean": mean, "raw": outcomes}, f, indent=2)
    return mean


def main() -> None:
    global SOURCE_STAGE1_CKPT, SOURCE_STAGE2_CKPT
    parser = argparse.ArgumentParser()
    parser.add_argument("--f30", required=True)
    parser.add_argument("--f120", required=True)
    parser.add_argument("--f300", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--teachers", nargs="+", default=["c30", "c300", "crcd_nogate", "crcd"])
    parser.add_argument("--label_budgets", type=int, nargs="+", default=[8, 32, 118])
    parser.add_argument("--selection_seed", type=int, default=2024)
    parser.add_argument("--training_seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--eval_seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--adv_margin", type=float, default=0.05)
    parser.add_argument("--unreliable_weight", type=float, default=0.3)
    parser.add_argument("--loss", choices=list(LOSS_ARGS), default="mixed",
                        help="Loss preset: 'mixed' (hard CE + soft + ranking) or 'hard' (hard CE only).")
    parser.add_argument("--train_gpus", default="0,1")
    parser.add_argument("--eval_gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--only", nargs="*", default=None,
                        help="Optional filter 'teacher:k:seed' to run a subset.")
    parser.add_argument("--dataset", default=DATASET,
                        help="Target dataset name for online eval (jinan/hangzhou/...).")
    parser.add_argument("--test_flow", default=TEST_FLOW,
                        help="Target test traffic flow file for online eval.")
    parser.add_argument("--source_stage1_ckpt", default=SOURCE_STAGE1_CKPT,
                        help="Stage-1 alignment checkpoint (must match source architecture).")
    parser.add_argument("--source_stage2_ckpt", default=SOURCE_STAGE2_CKPT,
                        help="Stage-2 source policy checkpoint (warm-start init).")
    parser.add_argument("--hidden_dim", type=int, default=ARCH_ARGS["hidden_dim"],
                        help="Projector hidden dim (must match source checkpoints).")
    parser.add_argument("--prefix_tokens", type=int, default=ARCH_ARGS["prefix_tokens"],
                        help="Prefix token count (must match source checkpoints).")
    parser.add_argument("--max_length", type=int, default=ARCH_ARGS["max_length"],
                        help="Max prompt length (must match source checkpoints; e.g. 416 for exp_jinan ml416).")
    args = parser.parse_args()

    SOURCE_STAGE1_CKPT = args.source_stage1_ckpt
    SOURCE_STAGE2_CKPT = args.source_stage2_ckpt
    ARCH_ARGS["hidden_dim"] = args.hidden_dim
    ARCH_ARGS["prefix_tokens"] = args.prefix_tokens
    ARCH_ARGS["max_length"] = args.max_length

    python = _resolve_python()
    print(f"Interpreter: {python}")

    aligned, snapshot_order = crcd_relabel.load_aligned(args.f30, args.f120, args.f300)
    print(f"Aligned states: {len(aligned)}; snapshots: {len(snapshot_order)}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    status: Dict[str, str] = {}
    for teacher in args.teachers:
        kl_weight, kl_mode = TEACHER_KL[teacher]
        for K in args.label_budgets:
            for ts in args.training_seeds:
                if args.only:
                    tag = f"{teacher}:{K}:{ts}"
                    if tag not in args.only:
                        continue
                tag = f"{teacher}/k{K}/tr{ts}"
                tr_dir = os.path.join(args.output_dir, teacher, f"k{K}", f"tr{ts}")
                print(f"\n{'='*60}\n[{tag}] build -> train -> eval\n{'='*60}")

                # 1. dataset
                data_dir = os.path.join(tr_dir, "data")
                train_data = os.path.join(data_dir, "train_data.jsonl")
                val_data = os.path.join(data_dir, "val_data.jsonl")
                if not (os.path.exists(train_data) and os.path.exists(val_data)):
                    meta = build_dataset(
                        aligned, snapshot_order, teacher, K, args.selection_seed, ts,
                        args.adv_margin, args.unreliable_weight, data_dir,
                    )
                    print(f"    dataset: train={meta['n_train_records']} val={meta['n_val_records']} "
                          f"reliable_train={meta['n_reliable_train']}")
                else:
                    print("    dataset exists, skip build")

                # 2. train
                ckpt_dir = os.path.join(tr_dir, "checkpoints")
                best_pt = os.path.join(ckpt_dir, "best.pt")
                if not _checkpoint_complete(best_pt):
                    try:
                        train(python, train_data, val_data, ckpt_dir, ts, kl_weight, kl_mode, args.train_gpus, args.loss)
                    except Exception as e:
                        print(f"    TRAIN FAILED: {e}")
                        status[tag] = f"FAILED(train): {e}"
                        continue
                else:
                    print("    best.pt complete, skip train")

                # 3. eval
                eval_dir = os.path.join(tr_dir, "evaluation")
                missing = [s for s in args.eval_seeds
                           if not os.path.exists(os.path.join(eval_dir, f"result_seed{s}.json"))]
                if missing:
                    try:
                        mean = eval_online(python, best_pt, args.eval_seeds, args.eval_gpus, eval_dir,
                                           dataset=args.dataset, test_flow=args.test_flow)
                        print(f"    eval mean: ATT={mean['ATT']:.1f} AQL={mean['AQL']:.1f} AWT={mean['AWT']:.1f}")
                    except Exception as e:
                        print(f"    EVAL FAILED: {e}")
                        status[tag] = f"FAILED(eval): {e}"
                        continue
                else:
                    print("    eval exists, skip")
                status[tag] = "ok"

    # Aggregate
    print(f"\n{'='*60}\nAggregate (mean over training seeds)\n{'='*60}")
    agg = {}
    for teacher in args.teachers:
        for K in args.label_budgets:
            vals = {m: [] for m in RESULT_KEY_MAP}
            for ts in args.training_seeds:
                summ = os.path.join(args.output_dir, teacher, f"k{K}", f"tr{ts}", "evaluation", "eval_summary.json")
                if os.path.exists(summ):
                    with open(summ) as f:
                        mean = json.load(f)["mean"]
                    for m in RESULT_KEY_MAP:
                        vals[m].append(mean[m])
            if not vals["ATT"]:
                continue
            import statistics
            row = {m: {"mean": statistics.fmean(vals[m]), "std": statistics.pstdev(vals[m]) if len(vals[m]) > 1 else 0.0, "n": len(vals[m])} for m in RESULT_KEY_MAP}
            agg[f"{teacher}_k{K}"] = row
            print(f"{teacher} K={K} (n={row['ATT']['n']}): "
                  f"ATT={row['ATT']['mean']:.2f}±{row['ATT']['std']:.2f} "
                  f"AQL={row['AQL']['mean']:.2f}±{row['AQL']['std']:.2f} "
                  f"AWT={row['AWT']['mean']:.2f}±{row['AWT']['std']:.2f}")

    with open(os.path.join(args.output_dir, "status.json"), "w") as f:
        json.dump(status, f, indent=2)
    with open(os.path.join(args.output_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\nDone. Output: {args.output_dir}")


if __name__ == "__main__":
    main()
