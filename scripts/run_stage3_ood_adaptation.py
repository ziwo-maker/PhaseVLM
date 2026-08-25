#!/usr/bin/env python3
"""Stage 3: OOD-Guided SimExpert Adaptation for Cross-Network Transfer.

Main method: OOD-guided snapshot selection (refine-logs/EXPERIMENT_RESULTS.md
Phase 2 — OOD was the best acquisition method) applied end-to-end:

  P1 Zero-shot eval     — source pt32/hd256 policy on target test flow (K=0)
  P2 Target collection  — source policy deployed on target collection flow
                          (canonical evaluator hook → snapshot parity)
  P3 Per-(method, K) arm:
       select K snapshots (collection-time features only) → FULL SimExpert
       counterfactual relabeling of EXACTLY those K snapshots (honest K-query
       budget) → snapshot-level train/val split → BC fine-tune (mixed loss,
       warm-started from source checkpoint) → multi-seed online eval.

Design properties (addressing cross-model review round 1):
  - OOD scoring uses ONLY collection-time features; strict feature extraction
    (no zero-imputation); source stats from COMPLETE source records only.
  - Relabeling happens per (method, K) — a K-query run executes exactly K
    SimExpert queries; no cached-120-pool nesting.
  - Every relabeled record is verified: relabel_method must be
    simexpert_counterfactual with all four candidate metric entries.
  - Validation split is SNAPSHOT-level (a snapshot's intersections never
    straddle train/val), via --train_data_path/--val_data_path.
  - Unique per-process memo so parallel evals cannot collide on the
    memo+timestamp workspace of run_projection_online.py.
  - Config fingerprint stored in config.json and every sentinel; resume
    refuses on fingerprint mismatch (no silent stale-artifact reuse).
  - Evaluation = canonical CityFlowEnv on a flow-disjoint test set with fixed
    simulator seeds and 95% t-intervals. Ground truth is the simulator.

Same-network control is intentionally NOT included: all three Jinan flows
were used in source Stage-2 training, so no held-out Jinan flow exists.

Usage:
  python3 scripts/run_stage3_ood_adaptation.py \
      --source_dataset jinan --target_dataset hangzhou \
      --output_dir records/pipeline_runs/stage3_ood_20260813 \
      --methods random ood --label_budgets 8 32 120
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from projection_train.action_space import ACTION_LIST  # noqa: E402
from projection_train.ood_selection import (  # noqa: E402
    compute_source_stats,
    group_snapshots,
    method_uses_seed,
    select_snapshots,
    subset_records_by_times,
)

DEFAULT_LLM_PATH = "/home/data/model/qwen-vl-8B"

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

RESULT_KEY_MAP = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
}

# Source stats from the same training data the source policy saw.
SOURCE_STATS_DATA = [
    "data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
    "data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
    "data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl",
]

# New architecture (refine-logs/EXPERIMENT_FINAL_REPORT.md must-do retraining).
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

# ── Phase-scoped artifact fingerprints ─────────────────────────────────────
#
# Each pipeline phase binds its resume sentinel ONLY to the configuration and
# producer code that affect that phase's artifacts. A change to, e.g., the
# training script must not invalidate an expensive SimExpert relabeling.


def _ckpt_id(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.abspath(path)}:{st.st_mtime_ns}:{st.st_size}"


def _file_sha256(rel_paths: Sequence[str]) -> str:
    h = hashlib.sha256()
    for rel in rel_paths:
        p = os.path.join(_REPO, rel)
        if os.path.exists(p):
            with open(p, "rb") as f:
                h.update(f.read())
    return h.hexdigest()


def _phase_fp(payload: Dict[str, Any], files: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(payload, sort_keys=True, default=str).encode())
    h.update(_file_sha256(files).encode())
    return h.hexdigest()


def torch_version(python: str) -> str:
    result = subprocess.run(
        [python, "-c", "import torch; print(torch.__version__)"],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip() or "unknown"


def fp_eval(python: str, ckpt: str, dataset: str, flow: str,
            seeds: Sequence[int], tag: str) -> str:
    """Fingerprint for online-evaluation artifacts (also covers collection)."""
    return _phase_fp({
        "tag": tag, "ckpt": _ckpt_id(ckpt), "dataset": dataset, "flow": flow,
        "seeds": sorted(int(s) for s in seeds), "arch": ARCH_ARGS,
        "torch": torch_version(python),
    }, ["projection_train/run_projection_online.py"])


def fp_selection(args: argparse.Namespace, method: str, K: int,
                 pool_id: str, selection_seed: int) -> str:
    """Fingerprint for snapshot-selection artifacts."""
    return _phase_fp({
        "method": method, "K": K, "selection_seed": selection_seed,
        "pool": pool_id,
        "source_stats_data": [_ckpt_id(p) for p in args.source_stats_data],
    }, ["projection_train/ood_selection.py"])


def filter_usable_records(records: List[Dict]) -> Tuple[List[Dict], List[float]]:
    """Drop records whose CityFlow snapshot is missing/empty/unparseable.

    Returns (usable_records, dropped_times). A truncated (0-byte) snapshot
    file would otherwise surface as a relabeling failure — SimExpert cannot
    restore its state. Pre-filtering keeps the acquisition pool honest: K
    selected snapshots are all relabelable.
    """
    dropped_times: List[float] = []
    snapshot_cache: Dict[str, bool] = {}

    def _snapshot_ok(sp: str) -> bool:
        if sp in snapshot_cache:
            return snapshot_cache[sp]
        ok = False
        try:
            if sp and os.path.exists(sp) and os.path.getsize(sp) > 0:
                with open(sp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ok = isinstance(data, dict) and "trafficLights" in data
        except (json.JSONDecodeError, OSError, ValueError):
            ok = False
        snapshot_cache[sp] = ok
        return ok

    usable: List[Dict] = []
    dropped_times_set = set()
    for rec in records:
        sp = rec.get("snapshot_path", "")
        if _snapshot_ok(sp):
            usable.append(rec)
        else:
            dropped_times_set.add(float(rec.get("time_s", -1.0)))
    dropped_times = sorted(dropped_times_set)
    return usable, dropped_times


def fp_relabel(args: argparse.Namespace, dataset: str, flow: str) -> str:
    """Fingerprint for SimExpert relabeling artifacts."""
    return _phase_fp({
        "dataset": dataset, "flow": flow, "sim_horizon": args.sim_horizon,
        "cost_weights": {"att": 1.0, "aql": 1.0, "awt": 1.0, "spillback": 0.2},
        "awt_constraint_epsilon": 5.0, "spillback_constraint_epsilon": 0.05,
    }, ["scripts/relabel_simexpert.py", "projection_train/ood_selection.py"])


def fp_train(python: str, args: argparse.Namespace, train_data: str,
             val_data: str, align_ckpt: str, init_ckpt: str,
             training_seed: int) -> str:
    """Fingerprint for BC training artifacts."""
    return _phase_fp({
        "train_data": _ckpt_id(train_data), "val_data": _ckpt_id(val_data),
        "align_ckpt": _ckpt_id(align_ckpt), "init_ckpt": _ckpt_id(init_ckpt),
        "arch": ARCH_ARGS,
        "hparams": {
            "epochs": 5, "lr": 1e-5, "min_lr": 1e-6, "warmup_ratio": 0.1,
            "weight_decay": 1e-4, "max_grad_norm": 1.0,
            "batch_size": 2, "grad_accum_steps": 8,
            "target_mode": "mixed", "soft_target_alpha": 0.3,
            "soft_label_temperature": 0.35, "ranking_loss_weight": 0.05,
            "ranking_min_cost_gap": 0.02, "cost_att_weight": 1.0,
            "cost_aql_weight": 1.0, "cost_awt_weight": 1.0,
            "cost_spillback_weight": 0.2, "base_policy_kl_weight": 0.0,
            "best_metric": "val_top1_cost_regret",
        },
        "training_seed": training_seed,
        "torch": torch_version(python),
    }, ["projection_train/train_projection_bc.py"])


def _check_sentinel(sentinel_path: str, expected_fp: str) -> bool:
    """True if sentinel exists AND matches the current phase fingerprint.

    A mismatch raises — phase artifacts were produced under a different
    configuration/code; silently reusing them would corrupt results. Delete
    the sentinel (metadata only; artifacts are regenerated) to force a redo.
    """
    if not os.path.exists(sentinel_path):
        return False
    try:
        data = json.load(open(sentinel_path, "r", encoding="utf-8"))
        if data.get("fingerprint") != expected_fp:
            raise RuntimeError(
                f"Sentinel {sentinel_path} was produced by a different "
                f"configuration/code version. Refusing silent resume. "
                f"Delete the sentinel (or the output dir) to force regeneration."
            )
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _write_sentinel(sentinel_path: str, fingerprint: str) -> None:
    with open(sentinel_path, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint,
                   "timestamp": datetime.now().isoformat()}, f, indent=2)


# ── Interpreter resolution ────────────────────────────────────────────────

# The default /usr/bin/python3 resolves `cityflow` to a namespace stub with no
# Engine attribute. The real cityflow lives in the hnr5090 conda env (also has
# torch 2.11+cu128, transformers, torchrun). All subprocesses (online eval,
# state collection, SimExpert relabeling, BC training) must use that env.
_CONDA_PYTHON = "/home/sysu/anaconda3/envs/hnr5090/bin/python"


def _resolve_python() -> str:
    """Return a Python interpreter with a REAL cityflow extension."""
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
    raise RuntimeError(
        "No Python interpreter with a real cityflow found. "
        "Expected the hnr5090 conda env at " + _CONDA_PYTHON)


def _env_with_python(python: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO
    env["PATH"] = os.path.dirname(python) + os.pathsep + env.get("PATH", "")
    return env


# ── Utils ─────────────────────────────────────────────────────────────────


def _run(cmd: List[str], timeout: int = 7200, env: Optional[Dict] = None,
         desc: str = "", log_path: str = "") -> subprocess.CompletedProcess:
    """Run a command fail-fast, teeing output to log_path when given."""
    print(f"[{desc}] {' '.join(cmd[:8])} ..." if desc else f"  {' '.join(cmd[:6])} ...")
    kwargs: Dict[str, Any] = dict(cwd=_REPO, timeout=timeout,
                                  env=(env or os.environ))
    if log_path:
        with open(log_path, "a", encoding="utf-8") as lf:
            lf.write(f"\n===== {desc} @ {datetime.now().isoformat()} =====\n")
            lf.write(" ".join(cmd) + "\n")
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, text=True, **kwargs)
    else:
        result = subprocess.run(cmd, capture_output=True, text=True, **kwargs)
        if result.returncode != 0:
            print(f"  STDOUT (last 500): {result.stdout[-500:]}")
            print(f"  STDERR (last 500): {result.stderr[-500:]}")
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd[:6])}")
    return result


def _read_result_json(path: str) -> Dict[str, float]:
    with open(path, "r") as f:
        payload = json.load(f)
    results = payload.get("results", payload)
    metrics = {}
    for label, key in RESULT_KEY_MAP.items():
        if key not in results:
            raise KeyError(f"Metric '{key}' not found in {path}. Keys: {list(results.keys())}")
        val = results[key]
        if val is None or (isinstance(val, float) and not np.isfinite(val)):
            raise ValueError(f"Metric '{key}' is None/non-finite in {path}: {val}")
        metrics[label] = float(val)
    return metrics


def _build_eval_cmd(python: str, ckpt_path: str, dataset: str, traffic_file: str,
                    seed: int, result_json: str, collect_states_output: str = "",
                    tag: str = "eval",
                    llm_path: str = DEFAULT_LLM_PATH) -> List[str]:
    # Unique memo per process: parallel evals share a timestamp-second and
    # path_check() raises FileExistsError on workspace collision.
    memo = f"stage3_ood_{tag}_s{seed}_{uuid.uuid4().hex[:8]}"
    cmd = [
        python,
        os.path.join(_REPO, "projection_train", "run_projection_online.py"),
        "--memo", memo,
        "--projection_ckpt", ckpt_path,
        "--llm_path", llm_path,
        "--dataset", dataset,
        "--traffic_file", traffic_file,
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
    if collect_states_output:
        cmd.extend(["--collect_states_output", collect_states_output])
    return cmd


def _run_parallel_evals(
    python: str,
    ckpt_path: str,
    dataset: str,
    traffic_file: str,
    seeds: Sequence[int],
    eval_gpus: Sequence[int],
    output_dir: str,
    collect_states_output: str = "",
    llm_path: str = DEFAULT_LLM_PATH,
    tag: str = "",
) -> Dict[str, Dict[str, float]]:
    """Run one online evaluation per seed, in parallel across GPUs."""
    os.makedirs(output_dir, exist_ok=True)
    procs: List[subprocess.Popen] = []
    per_seed: Dict[int, Dict[str, Any]] = {}
    for i, seed in enumerate(seeds):
        gpu = eval_gpus[i % len(eval_gpus)]
        result_json = os.path.join(output_dir, f"result_seed{seed}.json")
        cmd = _build_eval_cmd(python, ckpt_path, dataset, traffic_file, seed,
                              result_json,
                              collect_states_output=collect_states_output,
                              tag=tag, llm_path=llm_path)
        env = _env_with_python(python)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        log_path = os.path.join(output_dir, f"eval_seed{seed}.log")
        lf = open(log_path, "a", encoding="utf-8")
        lf.write(f"\n===== {tag} seed={seed} gpu={gpu} @ {datetime.now().isoformat()} =====\n")
        lf.write(" ".join(cmd) + "\n")
        proc = subprocess.Popen(cmd, cwd=_REPO, env=env, stdout=lf,
                                stderr=subprocess.STDOUT, text=True)
        lf.close()
        procs.append(proc)
        per_seed[seed] = {"proc": proc, "result_json": result_json, "gpu": gpu}
        print(f"  launched {tag} seed={seed} on GPU {gpu} (pid {proc.pid})")

    outcomes: Dict[str, Dict[str, float]] = {}
    for seed, info in per_seed.items():
        rc = info["proc"].wait()
        if rc != 0 or not os.path.exists(info["result_json"]):
            raise RuntimeError(
                f"{tag} seed={seed} failed (rc={rc}); log: "
                f"{output_dir}/eval_seed{seed}.log")
        outcomes[f"seed_{seed}"] = _read_result_json(info["result_json"])
        m = outcomes[f"seed_{seed}"]
        print(f"  {tag} seed={seed}: ATT={m['ATT']:.1f} AQL={m['AQL']:.1f} AWT={m['AWT']:.1f}")
    return outcomes


def _summarize(outcomes: Dict[str, Dict[str, float]], output_dir: str,
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compute mean/std/95% t-interval across seeds and save."""
    from scipy import stats as scipy_stats
    all_values = {m: [] for m in RESULT_KEY_MAP}
    for payload in outcomes.values():
        for label in RESULT_KEY_MAP:
            all_values[label].append(payload[label])

    summary = {}
    for metric, values in all_values.items():
        arr = np.array(values, dtype=np.float64)
        n = len(arr)
        mean_v = float(np.mean(arr))
        std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        ci_half = float(scipy_stats.t.ppf(0.975, n - 1)) * std_v / np.sqrt(n) if n > 1 else 0.0
        summary[metric] = {
            "mean": round(mean_v, 2), "std": round(std_v, 2), "n": n,
            "ci95_low": round(mean_v - ci_half, 2),
            "ci95_high": round(mean_v + ci_half, 2),
        }

    payload: Dict[str, Any] = {
        "summary": summary,
        "raw_values": {m: [round(v, 2) for v in vs] for m, vs in all_values.items()},
    }
    if extra:
        payload.update(extra)
    with open(os.path.join(output_dir, "eval_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Summary: {json.dumps(summary)}")
    return summary


# ── Phases ────────────────────────────────────────────────────────────────


def collect_states(
    python: str, ckpt_path: str, dataset: str, traffic_file: str,
    output_path: str, seed: int, gpu: int, llm_path: str = DEFAULT_LLM_PATH,
) -> int:
    """Collect decision states with the canonical evaluator hook."""
    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    _run_parallel_evals(
        python=python, ckpt_path=ckpt_path, dataset=dataset,
        traffic_file=traffic_file, seeds=[seed], eval_gpus=[gpu],
        output_dir=os.path.dirname(output_path),
        collect_states_output=output_path, llm_path=llm_path, tag="collect",
    )
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"States output not found: {output_path}")
    count = sum(1 for line in open(output_path, "r", encoding="utf-8") if line.strip())
    print(f"  Collected {count} records -> {output_path}")
    return count


def relabel_simexpert(python: str, states_path: str, output_path: str,
                      dataset: str, traffic_file: str, output_dir: str,
                      sim_horizon: int = 30) -> str:
    """Full SimExpert counterfactual relabeling of the SELECTED states."""
    cfg = NETWORK_CONFIGS[dataset]
    roadnet = os.path.join(_REPO, cfg["roadnet"])
    traffic_abs = os.path.join(_REPO, cfg["traffic_root"], traffic_file)

    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        python,
        os.path.join(_REPO, "scripts", "relabel_simexpert.py"),
        "--input", os.path.abspath(states_path),
        "--output", os.path.abspath(output_path),
        "--roadnet", os.path.abspath(roadnet),
        "--traffic_file", os.path.abspath(traffic_abs),
        "--dataset", dataset,
        "--work_dir", os.path.abspath(os.path.join(output_dir, "simexpert_work")),
        "--sim_horizon", str(sim_horizon),
        "--cost_att_weight", "1.0",
        "--cost_aql_weight", "1.0",
        "--cost_awt_weight", "1.0",
        "--cost_spillback_weight", "0.2",
        "--awt_constraint_epsilon", "5.0",
        "--spillback_constraint_epsilon", "0.05",
    ]
    env = _env_with_python(python)
    log_path = os.path.join(output_dir, "relabel.log")
    n_records = sum(1 for line in open(states_path, "r", encoding="utf-8") if line.strip())
    print(f"[Relabel] {n_records} records ...")
    _run(cmd, timeout=21600, env=env, desc="SimExpert relabel", log_path=log_path)
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Relabeled output not found: {output_path}")
    coverage_path = output_path.replace(".jsonl", "_coverage.json")
    if os.path.exists(coverage_path):
        with open(coverage_path, "r") as f:
            print(f"  Coverage: {json.dumps(json.load(f))}")
    return output_path


def verify_relabeled(relabeled_path: str, expected_n: int) -> Dict[str, Any]:
    """Enforce full SimExpert coverage: no policy-action pseudo-labels."""
    records = [
        json.loads(line) for line in open(relabeled_path, "r", encoding="utf-8")
        if line.strip()
    ]
    bad: List[str] = []
    for rec in records:
        if rec.get("relabel_method") != "simexpert_counterfactual":
            bad.append(f"{rec.get('intersection_id')}@t={rec.get('time_s')}: method={rec.get('relabel_method')}")
            continue
        cam = rec.get("candidate_action_metrics", {})
        if not isinstance(cam, dict) or not all(a in cam for a in ACTION_LIST):
            bad.append(f"{rec.get('intersection_id')}@t={rec.get('time_s')}: missing candidate metrics")
            continue
        if rec.get("expert_action") not in ACTION_LIST:
            bad.append(f"{rec.get('intersection_id')}@t={rec.get('time_s')}: invalid expert_action")
    n_records = len(records)
    n_snapshots = len(group_snapshots(records))
    report = {
        "n_records": n_records,
        "n_snapshots": n_snapshots,
        "n_expect_snapshots": expected_n,
        "n_bad_records": len(bad),
        "bad_examples": bad[:10],
    }
    if bad or n_snapshots != expected_n:
        raise RuntimeError(
            f"Relabel verification FAILED: {len(bad)}/{n_records} records unusable, "
            f"{n_snapshots} snapshots (expected {expected_n}). {report['bad_examples']}"
        )
    print(f"  Verified relabeling: {n_records} records / {n_snapshots} snapshots, "
          f"all full SimExpert.")
    return report


def split_snapshots(relabeled_path: str, val_ratio: float,
                    training_seed: int) -> Dict[str, Any]:
    """Snapshot-level train/val split: a snapshot's intersections never
    straddle the split. Returns {train_snapshots, val_snapshots} times."""
    records = [
        json.loads(line) for line in open(relabeled_path, "r", encoding="utf-8")
        if line.strip()
    ]
    times = sorted(group_snapshots(records).keys())
    n_val = max(1, int(round(len(times) * val_ratio)))
    if n_val >= len(times):
        n_val = max(1, len(times) // 2)
    rng = np.random.RandomState(training_seed)
    val_times = set(rng.choice(times, size=n_val, replace=False).tolist())
    train_times = [t for t in times if t not in val_times]
    return {"times": times, "train": train_times, "val": sorted(val_times)}


def _run_selection_and_relabel(
    python: str,
    args: argparse.Namespace,
    method: str,
    K: int,
    selection_seed: int,
    collected_records: List[Dict],
    source_stats: Optional[Dict[str, Any]],
    dataset: str,
    collection_flow: str,
    sel_dir: str,
    pool_id: str,
) -> str:
    """Select K snapshots (seeded) and relabel them. Returns relabeled path.

    Lives under sel_dir (selection-seed scoped) so multiple training seeds
    share the same selection + relabeled data.
    """
    os.makedirs(sel_dir, exist_ok=True)
    fp_sel = fp_selection(args, method, K, pool_id, selection_seed)
    fp_rel = fp_relabel(args, dataset, collection_flow)
    # Bind relabel to the exact selection so a changed selection invalidates it.
    fp_rel_combined = f"{fp_rel}:{fp_sel}"

    selection_path = os.path.join(sel_dir, "selection.json")
    selected_states_path = os.path.join(sel_dir, "selected_states.jsonl")
    selection_fp_ok = False
    if os.path.exists(selection_path):
        try:
            selection_fp_ok = json.load(open(selection_path, "r", encoding="utf-8")).get(
                "fingerprint") == fp_sel
        except (json.JSONDecodeError, ValueError):
            selection_fp_ok = False
    if not selection_fp_ok:
        import shutil
        for stale in [selected_states_path,
                      os.path.join(sel_dir, "relabeled_states.jsonl"),
                      os.path.join(sel_dir, "relabeled.done.json")]:
            if os.path.exists(stale):
                os.remove(stale)
        for stale_dir in [os.path.join(sel_dir, "simexpert_work")]:
            if os.path.isdir(stale_dir):
                shutil.rmtree(stale_dir)
        print(f"  [{method} K={K} sel={selection_seed}] fingerprint changed — "
              f"invalidating selection/relabel artifacts")
    if not os.path.exists(selected_states_path):
        selected_times, meta = select_snapshots(
            records=collected_records, method=method, K=K,
            source_stats=source_stats, seed=selection_seed,
        )
        selected_records = subset_records_by_times(collected_records, selected_times)
        with open(selected_states_path, "w", encoding="utf-8") as f:
            for rec in selected_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(selection_path, "w", encoding="utf-8") as f:
            json.dump({
                **meta,
                "n_snapshots_available": len(group_snapshots(collected_records)),
                "n_records_selected": len(selected_records),
                "selected_times": [float(t) for t in selected_times],
                "fingerprint": fp_sel,
            }, f, indent=2)
        print(f"  [{method} K={K} sel={selection_seed}] selected "
              f"{len(selected_times)} snapshots ({len(selected_records)} records)")
    else:
        print(f"  [{method} K={K} sel={selection_seed}] selected_states exists, "
              f"skip selection")

    relabeled_path = os.path.join(sel_dir, "relabeled_states.jsonl")
    relabeled_done = os.path.join(sel_dir, "relabeled.done.json")
    if not _check_sentinel(relabeled_done, fp_rel_combined):
        if os.path.exists(relabeled_path):
            os.remove(relabeled_path)
        relabel_simexpert(
            python=python,
            states_path=selected_states_path, output_path=relabeled_path,
            dataset=dataset, traffic_file=collection_flow, output_dir=sel_dir,
            sim_horizon=args.sim_horizon,
        )
        report = verify_relabeled(relabeled_path, expected_n=K)
        _write_sentinel(relabeled_done, fp_rel_combined)
        with open(os.path.join(sel_dir, "relabel_verification.json"), "w") as f:
            json.dump(report, f, indent=2)
    else:
        print(f"  [{method} K={K} sel={selection_seed}] relabeled.done matches "
              f"fingerprint, skip relabel")
    return relabeled_path


def _run_training_replicate(
    python: str,
    args: argparse.Namespace,
    method: str,
    K: int,
    training_seed: int,
    relabeled_path: str,
    source_stage2_ckpt: str,
    source_stage1_ckpt: str,
    llm_path: str,
    dataset: str,
    test_flow: str,
    seeds: Sequence[int],
    train_gpus: str,
    eval_gpus: Sequence[int],
    tr_dir: str,
) -> None:
    """Snapshot-split → train → multi-seed eval for ONE training replicate."""
    os.makedirs(tr_dir, exist_ok=True)
    tag = f"{method}_k{K}_tr{training_seed}"

    # --- 3. snapshot-level train/val split ---
    train_data_path = os.path.join(tr_dir, "train_data.jsonl")
    val_data_path = os.path.join(tr_dir, "val_data.jsonl")
    if not os.path.exists(train_data_path):
        split = split_snapshots(relabeled_path, val_ratio=0.1,
                                training_seed=training_seed)
        records = [
            json.loads(line) for line in open(relabeled_path, "r", encoding="utf-8")
            if line.strip()
        ]
        train_recs = subset_records_by_times(records, split["train"])
        val_recs = subset_records_by_times(records, split["val"])
        with open(train_data_path, "w", encoding="utf-8") as f:
            for rec in train_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(val_data_path, "w", encoding="utf-8") as f:
            for rec in val_recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(os.path.join(tr_dir, "split.json"), "w", encoding="utf-8") as f:
            json.dump({"train_snapshots": [float(t) for t in split["train"]],
                       "val_snapshots": [float(t) for t in split["val"]],
                       "train_records": len(train_recs),
                       "val_records": len(val_recs)}, f, indent=2)
        print(f"  [{tag}] snapshot split: {len(split['train'])} train / "
              f"{len(split['val'])} val snapshots "
              f"({len(train_recs)}/{len(val_recs)} records)")
    else:
        print(f"  [{tag}] train/val split exists, skip")

    # --- 4. training (fingerprint-guarded resume) ---
    checkpoints_dir = os.path.join(tr_dir, "checkpoints")
    best_pt = os.path.join(checkpoints_dir, "best.pt")
    train_done = os.path.join(tr_dir, "train.done.json")
    fp_trn = fp_train(python, args, train_data_path, val_data_path,
                      source_stage1_ckpt, source_stage2_ckpt, training_seed)
    if not _check_sentinel(train_done, fp_trn) or not os.path.exists(best_pt):
        os.makedirs(checkpoints_dir, exist_ok=True)
        world_size = len(train_gpus.split(","))
        torchrun = os.path.join(os.path.dirname(python), "torchrun")
        if not os.path.isfile(torchrun):
            raise RuntimeError(f"torchrun not found next to interpreter: {torchrun}")
        cmd = [
            torchrun, f"--nproc_per_node={world_size}",
            os.path.join(_REPO, "projection_train", "train_projection_bc.py"),
            "--train_data_path", train_data_path,
            "--val_data_path", val_data_path,
            "--llm_path", llm_path,
            "--align_ckpt", source_stage1_ckpt,
            "--init_policy_ckpt", source_stage2_ckpt,
            "--output_dir", checkpoints_dir,
            "--prompt_mode", "queue_wait_short",
            "--batch_size", "2", "--grad_accum_steps", "8",
            "--epochs", "5",
            "--lr", "1e-5", "--min_lr", "1e-6", "--warmup_ratio", "0.1",
            "--lr_schedule", "cosine",
            "--weight_decay", "1e-4", "--max_grad_norm", "1.0",
            "--num_workers", "2", "--seed", str(training_seed), "--bf16",
            "--log_interval", "10", "--save_every", "1",
            "--target_mode", "mixed",
            "--soft_target_alpha", "0.3", "--soft_label_temperature", "0.35",
            "--ranking_loss_weight", "0.05", "--ranking_min_cost_gap", "0.02",
            "--cost_att_weight", "1.0", "--cost_aql_weight", "1.0",
            "--cost_awt_weight", "1.0", "--cost_spillback_weight", "0.2",
            "--base_policy_kl_weight", "0.0",
            "--best_metric", "val_top1_cost_regret", "--best_mode", "min",
            "--ddp_timeout_seconds", "1800",
        ]
        for key, val in ARCH_ARGS.items():
            cmd.extend([f"--{key}", str(val)])
        env = _env_with_python(python)
        env["CUDA_VISIBLE_DEVICES"] = train_gpus
        _run(cmd, timeout=14400, env=env, desc=f"Adapt {tag}",
             log_path=os.path.join(tr_dir, "train.log"))
        if not os.path.exists(best_pt):
            raise RuntimeError(f"Training finished but best.pt missing at {best_pt}")
        _write_sentinel(train_done, fp_trn)
    else:
        print(f"  [{tag}] train.done matches fingerprint, skip training")

    # --- 5. evaluation (seed-aware resume) ---
    eval_dir = os.path.join(tr_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    missing_seeds = [
        s for s in seeds
        if not os.path.exists(os.path.join(eval_dir, f"result_seed{s}.json"))
    ]
    if missing_seeds:
        outcomes = _run_parallel_evals(
            python=python,
            ckpt_path=best_pt, dataset=dataset, traffic_file=test_flow,
            seeds=missing_seeds, eval_gpus=eval_gpus, output_dir=eval_dir,
            llm_path=llm_path, tag=tag,
        )
        for s in seeds:
            path = os.path.join(eval_dir, f"result_seed{s}.json")
            if s not in missing_seeds and os.path.exists(path):
                outcomes[f"seed_{s}"] = _read_result_json(path)
        _summarize(outcomes, eval_dir, extra={
            "method": method, "K": K, "training_seed": training_seed,
            "fingerprint": fp_eval(python, best_pt, dataset, test_flow, seeds, tag),
        })
    else:
        print(f"  [{tag}] all eval seeds exist, skip eval")


def run_arm(
    python: str,
    args: argparse.Namespace,
    method: str,
    K: int,
    collected_records: List[Dict],
    source_stats: Optional[Dict[str, Any]],
    source_stage2_ckpt: str,
    source_stage1_ckpt: str,
    llm_path: str,
    dataset: str,
    collection_flow: str,
    test_flow: str,
    seeds: Sequence[int],
    train_gpus: str,
    eval_gpus: Sequence[int],
    output_dir: str,
    pool_id: str,
    selection_seeds: Sequence[int],
    training_seeds: Sequence[int],
) -> None:
    """One (method, K) arm across selection × training replicates.

    Selection + relabel are scoped per selection_seed (sel{ss}); snapshot
    split + train + eval are scoped per training_seed (tr{ts}) under the
    corresponding sel{ss}. Deterministic methods collapse selection_seeds to a
    single representative seed (their selection does not depend on the seed).
    """
    k_dir = os.path.join(output_dir, f"k{K}")
    os.makedirs(k_dir, exist_ok=True)

    eff_sel_seeds = selection_seeds if method_uses_seed(method) else selection_seeds[:1]

    for ss in eff_sel_seeds:
        sel_dir = os.path.join(k_dir, f"sel{ss}")
        relabeled_path = _run_selection_and_relabel(
            python=python, args=args, method=method, K=K, selection_seed=ss,
            collected_records=collected_records, source_stats=source_stats,
            dataset=dataset, collection_flow=collection_flow, sel_dir=sel_dir,
            pool_id=pool_id,
        )
        for ts in training_seeds:
            tr_dir = os.path.join(sel_dir, f"tr{ts}")
            _run_training_replicate(
                python=python, args=args, method=method, K=K, training_seed=ts,
                relabeled_path=relabeled_path,
                source_stage2_ckpt=source_stage2_ckpt,
                source_stage1_ckpt=source_stage1_ckpt,
                llm_path=llm_path, dataset=dataset, test_flow=test_flow,
                seeds=seeds, train_gpus=train_gpus, eval_gpus=eval_gpus,
                tr_dir=tr_dir,
            )


def aggregate_variance(
    output_dir: str,
    methods: Sequence[str],
    label_budgets: Sequence[int],
) -> Dict[str, Any]:
    """Aggregate per-training-replicate online metrics into mean/std/95% CI.

    The statistical unit is the independent training replicate (each (sel,tr)
    produces one checkpoint, evaluated over the same online seeds). Returns a
    per-(method, K) summary keyed by metric, plus per-replicate raw values.
    """
    from scipy import stats as scipy_stats

    result: Dict[str, Any] = {}
    for method in methods:
        for K in label_budgets:
            k_dir = os.path.join(output_dir, method, f"k{K}")
            if not os.path.isdir(k_dir):
                continue
            replicates: List[Dict[str, Any]] = []
            for sel_name in sorted(os.listdir(k_dir)):
                sel_path = os.path.join(k_dir, sel_name)
                if not sel_name.startswith("sel") or not os.path.isdir(sel_path):
                    continue
                for tr_name in sorted(os.listdir(sel_path)):
                    tr_path = os.path.join(sel_path, tr_name)
                    if not tr_name.startswith("tr") or not os.path.isdir(tr_path):
                        continue
                    summary_path = os.path.join(tr_path, "evaluation",
                                                "eval_summary.json")
                    if not os.path.exists(summary_path):
                        continue
                    with open(summary_path, "r", encoding="utf-8") as f:
                        s = json.load(f)
                    summary = s.get("summary", {})
                    replicates.append({
                        "selection_seed": sel_name,
                        "training_seed": tr_name,
                        "metrics": {m: summary[m]["mean"] for m in RESULT_KEY_MAP
                                    if m in summary},
                    })
            if not replicates:
                continue
            agg: Dict[str, Any] = {"n_training_replicates": len(replicates),
                                   "replicates": replicates}
            for metric in RESULT_KEY_MAP:
                vals = [r["metrics"][metric] for r in replicates if metric in r["metrics"]]
                if not vals:
                    continue
                arr = np.array(vals, dtype=np.float64)
                n = len(arr)
                mean_v = float(np.mean(arr))
                std_v = float(np.std(arr, ddof=1)) if n > 1 else 0.0
                ci_half = (float(scipy_stats.t.ppf(0.975, n - 1)) * std_v / np.sqrt(n)
                           if n > 1 else 0.0)
                agg[metric] = {
                    "mean": round(mean_v, 2), "std": round(std_v, 2), "n": n,
                    "ci95_low": round(mean_v - ci_half, 2),
                    "ci95_high": round(mean_v + ci_half, 2),
                }
            result[f"{method}_k{K}"] = agg
            with open(os.path.join(k_dir, "variance_summary.json"), "w",
                      encoding="utf-8") as f:
                json.dump(agg, f, indent=2)
            print(f"\n[Variance] {method} K={K} across "
                  f"{len(replicates)} training replicates:")
            for metric in RESULT_KEY_MAP:
                if metric in agg:
                    print(f"  {metric}: mean={agg[metric]['mean']} "
                          f"std={agg[metric]['std']} "
                          f"[{agg[metric]['ci95_low']}, {agg[metric]['ci95_high']}]")
    return result


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Stage 3: OOD-Guided SimExpert Cross-Network Adaptation")
    parser.add_argument("--source_dataset", default="jinan",
                        choices=["jinan", "hangzhou"])
    parser.add_argument("--target_dataset", default="hangzhou",
                        choices=["jinan", "hangzhou"])
    parser.add_argument("--source_stage2_ckpt", required=True)
    parser.add_argument("--source_stage1_ckpt", required=True)
    parser.add_argument("--source_stats_data", nargs="+", default=SOURCE_STATS_DATA,
                        help="Source-network JSONL for OOD distribution stats")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--collection_flow", default="")
    parser.add_argument("--test_flow", default="")
    parser.add_argument("--label_budgets", type=int, nargs="+", default=[8, 32, 120])
    parser.add_argument("--methods", nargs="+", default=["random", "ood"],
                        choices=["random", "ood", "ood_diverse",
                                 "ood_window_quota", "random_window_quota"])
    parser.add_argument("--llm_path", default=DEFAULT_LLM_PATH)
    parser.add_argument("--train_gpus", default="0,1")
    parser.add_argument("--eval_gpus", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--selection_seeds", type=int, nargs="+", default=[2024],
                        help="RNG seeds for snapshot selection (affects random / "
                             "random_window_quota; deterministic methods ignore it)")
    parser.add_argument("--training_seeds", type=int, nargs="+", default=[2024],
                        help="Independent training replicates (each seeds the "
                             "train/val split AND the torch/trainer RNG)")
    parser.add_argument("--sim_horizon", type=int, default=30)
    args = parser.parse_args()

    if args.source_dataset == args.target_dataset:
        print("ERROR: source and target must differ for cross-network adaptation. "
              "Same-network control is intentionally unsupported (no held-out "
              "Jinan flow exists — all three were used in source training).")
        sys.exit(1)

    tgt_cfg = NETWORK_CONFIGS[args.target_dataset]
    collection_flow = args.collection_flow or tgt_cfg["flows"][0]
    test_flow = args.test_flow or (tgt_cfg["flows"][1] if len(tgt_cfg["flows"]) > 1
                                   else tgt_cfg["flows"][0])
    if collection_flow == test_flow:
        print(f"ERROR: collection_flow must differ from test_flow (both {test_flow})")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Resolve cityflow-capable interpreter for ALL subprocesses
    python = _resolve_python()
    print(f"Subprocess interpreter: {python}")

    fp_zero_shot = fp_eval(python, args.source_stage2_ckpt, args.target_dataset,
                           test_flow, args.seeds, "zero_shot")
    fp_collect = fp_eval(python, args.source_stage2_ckpt, args.target_dataset,
                         collection_flow, [args.seeds[0]], "collect")

    config = {
        "source_dataset": args.source_dataset,
        "target_dataset": args.target_dataset,
        "source_stage2_ckpt": args.source_stage2_ckpt,
        "source_stage1_ckpt": args.source_stage1_ckpt,
        "collection_flow": collection_flow,
        "test_flow": test_flow,
        "label_budgets": args.label_budgets,
        "methods": args.methods,
        "seeds": args.seeds,
        "selection_seeds": args.selection_seeds,
        "training_seeds": args.training_seeds,
        "sim_horizon": args.sim_horizon,
        "arch": ARCH_ARGS,
        "interpreter": python,
        "torch": torch_version(python),
        "fingerprint_zero_shot": fp_zero_shot,
        "fingerprint_collection": fp_collect,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Zero-shot fingerprint: {fp_zero_shot[:16]}...")
    status: Dict[str, str] = {}

    # ── P1: Zero-shot (K=0 baseline, seed-aware resume) ──
    zero_shot_dir = os.path.join(args.output_dir, "zero_shot")
    print("\n" + "=" * 60 + "\nP1: Zero-shot evaluation\n" + "=" * 60)
    os.makedirs(zero_shot_dir, exist_ok=True)
    missing_seeds = [
        s for s in args.seeds
        if not os.path.exists(os.path.join(zero_shot_dir, f"result_seed{s}.json"))
    ]
    if missing_seeds:
        outcomes = _run_parallel_evals(
            python=python,
            ckpt_path=args.source_stage2_ckpt, dataset=args.target_dataset,
            traffic_file=test_flow, seeds=missing_seeds, eval_gpus=args.eval_gpus,
            output_dir=zero_shot_dir, llm_path=args.llm_path, tag="zero_shot",
        )
        for s in args.seeds:
            path = os.path.join(zero_shot_dir, f"result_seed{s}.json")
            if s not in missing_seeds and os.path.exists(path):
                outcomes[f"seed_{s}"] = _read_result_json(path)
        _summarize(outcomes, zero_shot_dir, extra={
            "method": "zero_shot", "K": 0, "fingerprint": fp_zero_shot,
        })
    else:
        print("  all zero-shot seed results exist, skip")
    status["zero_shot"] = "ok"

    # ── P2: Target state collection ──
    collected_path = os.path.join(args.output_dir, "collected_states.jsonl")
    collected_done = os.path.join(args.output_dir, "collection.done.json")
    print("\n" + "=" * 60 + "\nP2: Target state collection\n" + "=" * 60)
    if not _check_sentinel(collected_done, fp_collect):
        if os.path.exists(collected_path):
            os.remove(collected_path)  # discard partial file from a crashed run
        collect_states(
            python=python,
            ckpt_path=args.source_stage2_ckpt, dataset=args.target_dataset,
            traffic_file=collection_flow, output_path=collected_path,
            seed=args.seeds[0], gpu=args.eval_gpus[0], llm_path=args.llm_path,
        )
        _write_sentinel(collected_done, fp_collect)
    else:
        print("  collection.done matches fingerprint, skip")
    status["collection"] = "ok"

    # ── Source stats for OOD scoring (computed once, shared across arms) ──
    source_stats: Optional[Dict[str, Any]] = None
    if any(m in ("ood", "ood_diverse", "ood_window_quota") for m in args.methods):
        source_records: List[Dict] = []
        for p in args.source_stats_data:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        source_records.append(json.loads(line))
        source_stats = compute_source_stats(source_records)
        print(f"Source stats: {source_stats['n_records']} complete records / "
              f"{source_stats['n_total_seen']} seen "
              f"(coverage={source_stats['coverage']:.3f})")
        stats_path = os.path.join(args.output_dir, "source_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "mean": source_stats["mean"].tolist(),
                "std": source_stats["std"].tolist(),
                "n_records": source_stats["n_records"],
                "n_total_seen": source_stats["n_total_seen"],
                "coverage": source_stats["coverage"],
                "source_files": args.source_stats_data,
            }, f, indent=2)

    # ── P3: per (method, K) arms ──
    collected_records = [
        json.loads(line) for line in open(collected_path, "r", encoding="utf-8")
        if line.strip()
    ]
    # Pre-filter snapshots that SimExpert cannot restore (0-byte/corrupt files)
    collected_records, dropped_times = filter_usable_records(collected_records)
    n_avail = len(group_snapshots(collected_records))
    print(f"\nAvailable snapshots on collection flow: {n_avail}"
          + (f" (dropped {len(dropped_times)} corrupt: {dropped_times[:5]}...)"
             if dropped_times else ""))
    pool_id = f"{_ckpt_id(collected_path)}:filtered:{n_avail}"
    for K in args.label_budgets:
        if K > n_avail:
            print(f"  NOTE: K={K} > usable snapshots ({n_avail}); "
                  f"will use K_eff={n_avail} (recorded in selection/eval metadata)")

    for method in args.methods:
        for K in args.label_budgets:
            K_eff = min(K, n_avail)
            print("\n" + "=" * 60 +
                  f"\nArm: {method} K={K}" +
                  (f" (effective {K_eff})" if K_eff != K else "") + "\n" + "=" * 60)
            try:
                run_arm(
                    python=python,
                    args=args,
                    method=method, K=K_eff, collected_records=collected_records,
                    source_stats=source_stats,
                    source_stage2_ckpt=args.source_stage2_ckpt,
                    source_stage1_ckpt=args.source_stage1_ckpt,
                    llm_path=args.llm_path, dataset=args.target_dataset,
                    collection_flow=collection_flow, test_flow=test_flow,
                    seeds=args.seeds, train_gpus=args.train_gpus,
                    eval_gpus=args.eval_gpus,
                    output_dir=os.path.join(args.output_dir, method),
                    pool_id=pool_id,
                    selection_seeds=args.selection_seeds,
                    training_seeds=args.training_seeds,
                )
                status[f"{method}_k{K}"] = "ok"
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"  FAILED {method} K={K}: {e}")
                status[f"{method}_k{K}"] = f"FAILED: {e}"

    # ── Aggregate variance across training replicates ──
    print("\n" + "=" * 60 + "\nAggregating training-replicate variance\n" + "=" * 60)
    try:
        aggregate_variance(args.output_dir, args.methods, args.label_budgets)
        status["variance_aggregation"] = "ok"
    except Exception as e:
        import traceback
        traceback.print_exc()
        status["variance_aggregation"] = f"FAILED: {e}"

    # ── Final status ──
    with open(os.path.join(args.output_dir, "status.json"), "w") as f:
        json.dump(status, f, indent=2)
    print("\n" + "=" * 60 + "\nPipeline complete. Status:")
    for phase, s in status.items():
        print(f"  {phase}: {s}")
    print(f"Output: {args.output_dir}\n" + "=" * 60)

    failures = [k for k, v in status.items() if v.startswith("FAILED")]
    if failures:
        print(f"\nFAILED phases: {failures}")
        sys.exit(1)


if __name__ == "__main__":
    main()
