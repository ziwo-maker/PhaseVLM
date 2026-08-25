#!/usr/bin/env python3
"""Evaluate BC and PPO checkpoints on multiple traffic flows within the same network.
Loads each checkpoint once, then evaluates sequentially on all specified flows.
Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_generalization.py \
    --bc_ckpt records/.../best.pt --ppo_ckpt records/.../best.pt \
    --dataset jinan --flows anon_3_4_jinan_real.json anon_3_4_jinan_real_2000.json anon_3_4_jinan_real_2500.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, _REPO)

import torch
from projection_train.prefix_action_policy import PrefixActionPolicy
from projection_train.trajectory_env import TrajectoryCityFlowEnv
from projection_train.train_projection_ppo_traj import evaluate_full_episode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bc_ckpt", required=True)
    parser.add_argument("--ppo_ckpt", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--flows", nargs="+", required=True)
    parser.add_argument("--llm_path", default="/home/data/model/qwen-vl-8B")
    parser.add_argument("--roadnet", default="")
    parser.add_argument("--traffic_root", default="")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024])
    parser.add_argument("--output_dir", default="records/pipeline_runs/stage3_generalization/eval")
    args = parser.parse_args()

    if not args.roadnet:
        net_map = {"jinan": "data/Jinan/3_4/roadnet_3_4.json",
                   "hangzhou": "data/Hangzhou/4_4/roadnet_4_4.json",
                   "newyork_28x7": "data/NewYork/28_7/roadnet_28_7.json"}
        args.roadnet = net_map[args.dataset]
    if not args.traffic_root:
        root_map = {"jinan": "data/Jinan/3_4", "hangzhou": "data/Hangzhou/4_4",
                    "newyork_28x7": "data/NewYork/28_7"}
        args.traffic_root = root_map[args.dataset]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Build policy once
    print("Loading policy...")
    policy = PrefixActionPolicy(
        model_path=args.llm_path, seq_len=16, image_size=224, periodicity=16,
        input_dim=8, hidden_dim=64, prefix_tokens=8, prefix_mode="prefix",
        max_length=64, device=device, with_value_head=False, torch_dtype=torch.bfloat16,
    )

    all_results = {}

    for ckpt_label, ckpt_path in [("bc", args.bc_ckpt), ("ppo", args.ppo_ckpt)]:
        print(f"\n{'='*60}\nLoading {ckpt_label}: {ckpt_path}")
        policy.load_checkpoint(ckpt_path, load_value_head=False)

        for flow in args.flows:
            flow_name = flow.replace(".json", "")
            for seed in args.seeds:
                key = f"{ckpt_label}_{flow_name}_seed{seed}"
                print(f"  Evaluating {key}...")
                t0 = time.time()
                env = TrajectoryCityFlowEnv(
                    roadnet_path=args.roadnet, traffic_root=args.traffic_root,
                    work_root=f"/tmp/eval_{key}", default_traffic_file=flow,
                    seq_len=16, movement_keys=["NT","NL","ST","SL","ET","EL","WT","WL"],
                    prompt_mode="queue_wait_short", traffic_series_interval=2,
                    window_s=30, min_action_time=30, reward_alpha=0.1,
                    switch_penalty_beta=0.02, queue_reward_coef=-1.0,
                    waiting_reward_coef=-1.0, system_vehicle_reward_coef=0.0,
                    throughput_reward_coef=0.0, starvation_penalty_coef=0.01,
                    reward_mode="stable_delta", run_counts=3600, env_seed=seed,
                    advance_seed_on_reset=False,
                )
                metrics = evaluate_full_episode(policy, env, device, flow)
                del env
                elapsed = time.time() - t0
                r = {
                    "avg_queue": round(metrics["eval_avg_queue"], 2),
                    "avg_waiting": round(metrics["eval_avg_waiting"], 2),
                    "avg_travel_time": round(metrics["eval_avg_travel_time"], 2),
                    "p95_wait": round(metrics.get("eval_waiting_p95", 0), 2),
                    "reward": round(metrics["eval_reward"], 2),
                    "elapsed_s": round(elapsed, 1),
                }
                all_results[key] = r
                print(f"    Q={r['avg_queue']:.1f} W={r['avg_waiting']:.1f} T={r['avg_travel_time']:.1f} ({elapsed:.0f}s)")

    # Save results
    out_path = os.path.join(args.output_dir, f"generalization_{args.dataset}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"Results saved to {out_path}")
    print(f"\n{'Flow':<35} {'BC Q':>8} {'PPO Q':>8} {'BC W':>8} {'PPO W':>8} {'BC T':>8} {'PPO T':>8}")
    print("-"*95)
    for flow in args.flows:
        fn = flow.replace(".json", "")
        for seed in args.seeds:
            bc = all_results.get(f"bc_{fn}_seed{seed}", {})
            pp = all_results.get(f"ppo_{fn}_seed{seed}", {})
            if bc and pp:
                print(f"{fn}_s{seed:<28} {bc.get('avg_queue',0):>8.1f} {pp.get('avg_queue',0):>8.1f} {bc.get('avg_waiting',0):>8.1f} {pp.get('avg_waiting',0):>8.1f} {bc.get('avg_travel_time',0):>8.1f} {pp.get('avg_travel_time',0):>8.1f}")

if __name__ == "__main__":
    main()
