#!/usr/bin/env python3
"""Run reverse Hangzhou→Jinan ablation matrix A-D with K=8,32,120."""
import json, os, random, subprocess, sys
from datetime import datetime

REPO = '/home/code/LLMSTSC'
os.chdir(REPO)
sys.path.insert(0, REPO)

REV_DIR = 'records/pipeline_runs/stage3_reverse_20260811_095618'
SOURCE_CKPT = 'records/pipeline_runs/exp_hangzhou_real/stage2_queue_wait_rule_gap2_p60_ml412_qw_20260530_115914/checkpoints/best.pt'
STAGE1_CKPT = 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt'
LLM_PATH = '/home/data/model/qwen-vl-8B'
TEST_FLOW = 'anon_3_4_jinan_real_2000.json'

ABLATIONS = [
    ('A', f'{REV_DIR}/A/relabeled_states.jsonl', 'hard', '0.0', '0.0', 'val_acc'),
    ('B', f'{REV_DIR}/relabeled_simexpert.jsonl', 'hard', '0.0', '0.0', 'val_acc'),
    ('C', f'{REV_DIR}/relabeled_simexpert.jsonl', 'soft', '0.0', '0.3', 'val_top1_cost_regret'),
    ('D', f'{REV_DIR}/relabeled_simexpert.jsonl', 'mixed', '0.05', '0.3', 'val_top1_cost_regret'),
]
KS = [8, 32, 120]
SEEDS = [2024, 42, 123]

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(f'{REV_DIR}/pipeline.log', 'a') as f:
        f.write(line + '\n')

def subsample(data_path, k, output_path):
    random.seed(2024)
    with open(data_path) as f:
        all_recs = [json.loads(l) for l in f if l.strip()]
    tg = {}
    for r in all_recs:
        tg.setdefault(r.get('time_s', 0), []).append(r)
    times = sorted(tg)
    sel = sorted(random.sample(times, min(k, len(times))))
    recs = [r for t in sel for r in tg[t]]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    return len(sel), len(recs)

def run_cmd(cmd, timeout=7200):
    log(f'  CMD: {" ".join(cmd[:6])}...')
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=REPO)
    if r.returncode != 0:
        log(f'  STDERR: {r.stderr[-300:]}')
        raise RuntimeError(f'Command failed: {r.returncode}')
    return r.stdout

def train(ablation, data_path, mode, rank_w, alpha, metric, k):
    k_dir = f'{REV_DIR}/{ablation}/adapted_k{k}'
    eval_json = f'{k_dir}/evaluation/eval_summary.json'
    if os.path.exists(eval_json):
        log(f'{ablation} K={k}: already done')
        return

    # Subsample
    train_data = f'{k_dir}/train_data.jsonl'
    n_snaps, n_recs = subsample(data_path, k, train_data)
    log(f'{ablation} K={k}: {n_snaps} snaps -> {n_recs} recs')

    # Train
    ckpt_dir = f'{k_dir}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)
    train_cmd = [
        sys.executable, 'projection_train/train_projection_bc.py',
        '--data_path', train_data, '--llm_path', LLM_PATH,
        '--align_ckpt', STAGE1_CKPT, '--init_policy_ckpt', SOURCE_CKPT,
        '--output_dir', ckpt_dir,
        '--seq_len', '16', '--image_size', '224',
        '--periodicity', '16', '--input_dim', '8', '--hidden_dim', '64',
        '--prefix_tokens', '8', '--prefix_mode', 'prefix', '--max_length', '64',
        '--prompt_mode', 'queue_wait_short',
        '--batch_size', '1', '--grad_accum_steps', '8',
        '--epochs', '5', '--lr', '1e-5', '--min_lr', '1e-6',
        '--warmup_ratio', '0.1', '--lr_schedule', 'cosine',
        '--weight_decay', '1e-4', '--max_grad_norm', '1.0',
        '--val_ratio', '0.1', '--split_strategy', 'stratified',
        '--num_workers', '2', '--seed', '2024', '--bf16',
        '--best_metric', metric, '--best_mode', 'min',
        '--target_mode', mode,
        '--soft_target_alpha', alpha, '--soft_label_temperature', '0.35',
        '--ranking_loss_weight', rank_w,
        '--cost_att_weight', '1.0', '--cost_aql_weight', '1.0',
        '--cost_awt_weight', '1.0', '--cost_spillback_weight', '0.2',
        '--ranking_min_cost_gap', '0.02', '--base_policy_kl_weight', '0.0',
    ]
    run_cmd(train_cmd, timeout=7200)

    # Eval
    best_pt = f'{ckpt_dir}/best.pt'
    assert os.path.exists(best_pt), f'No best.pt at {best_pt}'
    eval_dir = f'{k_dir}/evaluation'
    os.makedirs(eval_dir, exist_ok=True)

    for seed in SEEDS:
        result_json = f'{eval_dir}/result_seed{seed}.json'
        if os.path.exists(result_json):
            continue
        eval_cmd = [
            sys.executable, 'projection_train/run_projection_online.py',
            '--memo', f'stage3_rev_{ablation}_k{k}',
            '--projection_ckpt', best_pt,
            '--llm_path', LLM_PATH,
            '--dataset', 'jinan', '--traffic_file', TEST_FLOW,
            '--run_counts', '3600', '--num_rounds', '1', '--seed', str(seed),
            '--prompt_mode', 'queue_wait_short',
            '--seq_len', '16', '--image_size', '224',
            '--periodicity', '16', '--input_dim', '8', '--hidden_dim', '64',
            '--prefix_tokens', '8', '--prefix_mode', 'prefix', '--max_length', '64',
            '--inference_batch_size', '64', '--bf16',
            '--result_json', result_json,
        ]
        run_cmd(eval_cmd, timeout=7200)

    # Summarize
    import numpy as np
    from scipy import stats as scipy_stats
    mk = {'ATT': 'test_avg_travel_time_over', 'AQL': 'test_avg_queue_len_over', 'AWT': 'test_avg_waiting_time_over'}
    av = {m: [] for m in mk}
    for seed in SEEDS:
        p = f'{eval_dir}/result_seed{seed}.json'
        if os.path.exists(p):
            with open(p) as f:
                rr = json.load(f).get('results', json.load(f))
            for lb, key in mk.items():
                av[lb].append(float(rr.get(key, 0)))
    summary = {}
    for m, vs in av.items():
        a = np.array(vs, dtype=np.float64)
        n = len(a)
        mv = float(np.mean(a))
        sv = float(np.std(a, ddof=1)) if n > 1 else 0.0
        summary[m] = {'mean': round(mv, 2), 'std': round(sv, 2), 'n': n}
    with open(eval_json, 'w') as f:
        json.dump({'ablation': ablation, 'K': k, 'summary': summary}, f, indent=2)
    log(f'{ablation} K={k} DONE: {json.dumps(summary)}')

# ---- Main ----
log('=== Reverse Ablation Matrix Start ===')
for abl, data, mode, rank_w, alpha, metric in ABLATIONS:
    for k in KS:
        try:
            train(abl, data, mode, rank_w, alpha, metric, k)
        except Exception as e:
            log(f'FAILED {abl} K={k}: {e}')
            continue
log('=== Reverse Ablation Matrix Complete ===')
