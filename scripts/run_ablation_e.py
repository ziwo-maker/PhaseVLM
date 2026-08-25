#!/usr/bin/env python3
"""Ablation E: Topology-Conditioned Adapter for Cross-Network PhaseVLM."""
import json, os, random, subprocess, sys
from datetime import datetime

REPO = '/home/code/LLMSTSC'
os.chdir(REPO)
sys.path.insert(0, REPO)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
E_DIR = f'records/pipeline_runs/stage3_ablation_e_{TIMESTAMP}'
os.makedirs(E_DIR, exist_ok=True)

SOURCE_CKPT = 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage2/checkpoints/best.pt'
STAGE1_CKPT = 'records/pipeline_runs/jinan3_qwshort_targeted_20260805_qwshort_targeted/stage1/best.pt'
LLM_PATH = '/home/data/model/qwen-vl-8B'
TEST_FLOW = 'anon_4_4_hangzhou_real_5816.json'
DATA_PATH = 'records/pipeline_runs/stage3_ablation_20260810_204128/relabeled_simexpert.jsonl'
TOPO_CONFIG = '{"Jinan-3_4":{"rows":3,"cols":4},"Hangzhou-4_4":{"rows":4,"cols":4}}'

KS = [8, 32, 120]
SEEDS = [2024, 42, 123]

def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    line = f'[{ts}] {msg}'
    print(line, flush=True)
    with open(f'{E_DIR}/pipeline.log', 'a') as f:
        f.write(line + '\n')

def subsample(k, output_path):
    random.seed(2024)
    with open(DATA_PATH) as f:
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
        err = r.stderr[-300:] if r.stderr else '(none)'
        log(f'  STDERR: {err}')
        raise RuntimeError(f'Command failed: {r.returncode}')
    return r.stdout

def train_eval(ablation, k, extra_flags=None):
    """Train with topology adapter, then evaluate 3 seeds."""
    k_dir = f'{E_DIR}/{ablation}/adapted_k{k}'
    eval_json = f'{k_dir}/evaluation/eval_summary.json'
    if os.path.exists(eval_json):
        log(f'{ablation} K={k}: already done')
        return

    # Subsample
    train_data = f'{k_dir}/train_data.jsonl'
    n_snaps, n_recs = subsample(k, train_data)
    log(f'{ablation} K={k}: {n_snaps} snaps -> {n_recs} recs')

    # Build training command
    ckpt_dir = f'{k_dir}/checkpoints'
    os.makedirs(ckpt_dir, exist_ok=True)

    cmd = [
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
        '--best_metric', 'val_top1_cost_regret', '--best_mode', 'min',
        '--target_mode', 'mixed',
        '--soft_target_alpha', '0.3', '--soft_label_temperature', '0.35',
        '--ranking_loss_weight', '0.05',
        '--cost_att_weight', '1.0', '--cost_aql_weight', '1.0',
        '--cost_awt_weight', '1.0', '--cost_spillback_weight', '0.2',
        '--ranking_min_cost_gap', '0.02', '--base_policy_kl_weight', '0.0',
        '--enable_topology_adapter',
        '--topology_roadnet_configs', TOPO_CONFIG,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    run_cmd(cmd, timeout=7200)

    # Evaluate
    best_pt = f'{ckpt_dir}/best.pt'
    assert os.path.exists(best_pt), f'No best.pt at {best_pt}'
    eval_dir = f'{k_dir}/evaluation'
    os.makedirs(eval_dir, exist_ok=True)

    for seed in SEEDS:
        result_json = f'{eval_dir}/result_seed{seed}.json'
        if os.path.exists(result_json):
            # Verify non-empty
            if os.path.getsize(result_json) > 10:
                continue
        eval_cmd = [
            sys.executable, 'projection_train/run_projection_online.py',
            '--memo', f'stage3_e_{ablation}_k{k}',
            '--projection_ckpt', best_pt,
            '--llm_path', LLM_PATH,
            '--dataset', 'hangzhou', '--traffic_file', TEST_FLOW,
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
        if os.path.exists(p) and os.path.getsize(p) > 10:
            with open(p) as f:
                rr = json.load(f).get('results', {})
            for lb, key in mk.items():
                if rr.get(key) is not None:
                    av[lb].append(float(rr[key]))
    summary = {}
    for m, vs in av.items():
        if vs:
            a = np.array(vs, dtype=np.float64); n = len(a)
            summary[m] = {'mean': round(np.mean(a), 2), 'std': round(np.std(a, ddof=1), 2) if n > 1 else 0, 'n': n}
    with open(eval_json, 'w') as f:
        json.dump({'ablation': ablation, 'K': k, 'summary': summary}, f, indent=2)
    att = summary.get('ATT', {}).get('mean', '?')
    log(f'{ablation} K={k} DONE: ATT={att}')

# ---- E1: Topology Adapter Only (Jinan→Hangzhou, SimExpert+Pareto) ----
log('=== E1: Topology Adapter (SimExpert+Pareto data + topo) ===')
for k in KS:
    try:
        train_eval('E1', k)
    except Exception as e:
        log(f'FAILED E1 K={k}: {e}')

# ---- E2: Joint Jinan+Hangzhou Meta-Training ----
# First, collect Jinan states and relabel if needed
JN_COLLECTED = f'{E_DIR}/jinan_collected_states.jsonl'
JN_RELABELED = f'{E_DIR}/jinan_relabeled_simexpert.jsonl'

log('=== E2 Prep: Jinan State Collection ===')
if not os.path.exists(JN_COLLECTED):
    run_cmd([
        sys.executable, 'projection_train/run_projection_online.py',
        '--memo', 'stage3_e2_jn_collect',
        '--projection_ckpt', SOURCE_CKPT,
        '--llm_path', LLM_PATH,
        '--dataset', 'jinan', '--traffic_file', 'anon_3_4_jinan_real.json',
        '--run_counts', '3600', '--num_rounds', '1', '--seed', '2024',
        '--prompt_mode', 'queue_wait_short',
        '--seq_len', '16', '--image_size', '224',
        '--periodicity', '16', '--input_dim', '8', '--hidden_dim', '64',
        '--prefix_tokens', '8', '--prefix_mode', 'prefix', '--max_length', '64',
        '--inference_batch_size', '64', '--bf16',
        '--collect_states_output', JN_COLLECTED,
    ], timeout=7200)
log(f'Jinan states: {sum(1 for _ in open(JN_COLLECTED)) if os.path.exists(JN_COLLECTED) else 0} records')

if not os.path.exists(JN_RELABELED):
    log('SimExpert relabeling Jinan...')
    run_cmd([
        sys.executable, 'scripts/relabel_simexpert.py',
        '--input', JN_COLLECTED, '--output', JN_RELABELED,
        '--roadnet', 'data/Jinan/3_4/roadnet_3_4.json',
        '--traffic_file', 'data/Jinan/3_4/anon_3_4_jinan_real.json',
        '--dataset', 'jinan',
        '--work_dir', f'{E_DIR}/simexpert_jn_work',
        '--sim_horizon', '10',
        '--cost_att_weight', '1.0', '--cost_aql_weight', '1.0',
        '--cost_awt_weight', '1.0', '--cost_spillback_weight', '0.2',
    ], timeout=7200)

# Joint dataset
JOINT_DATA = f'{E_DIR}/joint_train.jsonl'
if not os.path.exists(JOINT_DATA):
    with open(JOINT_DATA, 'w') as out:
        for src in [DATA_PATH, JN_RELABELED]:
            if os.path.exists(src):
                with open(src) as inp:
                    out.write(inp.read())
    n = sum(1 for _ in open(JOINT_DATA))
    log(f'Joint dataset: {n} records')

# Phase M1: Joint pre-training with topology
log('=== E2 Phase M1: Joint Pre-Training ===')
JOINT_CKPT_DIR = f'{E_DIR}/E2_joint/checkpoints'
JOINT_BEST = f'{JOINT_CKPT_DIR}/best.pt'

if not os.path.exists(JOINT_BEST):
    os.makedirs(JOINT_CKPT_DIR, exist_ok=True)
    run_cmd([
        sys.executable, 'projection_train/train_projection_bc.py',
        '--data_path', JOINT_DATA, '--llm_path', LLM_PATH,
        '--align_ckpt', STAGE1_CKPT, '--init_policy_ckpt', SOURCE_CKPT,
        '--output_dir', JOINT_CKPT_DIR,
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
        '--best_metric', 'val_top1_cost_regret', '--best_mode', 'min',
        '--target_mode', 'mixed',
        '--soft_target_alpha', '0.3', '--soft_label_temperature', '0.35',
        '--ranking_loss_weight', '0.05',
        '--cost_att_weight', '1.0', '--cost_aql_weight', '1.0',
        '--cost_awt_weight', '1.0', '--cost_spillback_weight', '0.2',
        '--ranking_min_cost_gap', '0.02', '--base_policy_kl_weight', '0.0',
        '--enable_topology_adapter',
        '--topology_roadnet_configs', TOPO_CONFIG,
    ], timeout=14400)
    log('Joint pre-training complete')

# Phase M2: Few-shot adapt from joint checkpoint → Hangzhou
log('=== E2 Phase M2: Few-Shot Adapt (Joint→Hangzhou) ===')
for k in KS:
    try:
        train_eval('E2', k, extra_flags=['--init_policy_ckpt', JOINT_BEST])
    except Exception as e:
        log(f'FAILED E2 K={k}: {e}')

# ---- Summary ----
log('=== Ablation E Complete ===')
print(f'\nResults: {E_DIR}')
for abl in ['E1', 'E2']:
    for k in KS:
        p = f'{E_DIR}/{abl}/adapted_k{k}/evaluation/eval_summary.json'
        if os.path.exists(p):
            with open(p) as f:
                d = json.load(f)
            s = d.get('summary', {})
            att = s.get('ATT', {}).get('mean', '?')
            print(f'  {abl} K={k}: ATT={att}')
