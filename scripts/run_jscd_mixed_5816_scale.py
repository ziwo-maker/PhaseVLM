#!/usr/bin/env python3
"""Run the JSCD P2 data-scale experiment with a balanced 5816 supplement.

This is deliberately a separate runner from ``run_phase7_crcd.py``.  It
reuses the immutable P2 primary train/validation files and only adds a common,
snapshot-balanced 5816 supervision supplement to the F/P/J second-pass arms.
The primary online test remains Hangzhou_real.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
P2_ROOT = REPO / "records/pipeline_runs/jscd_p2_pilot_20260819_141017"
FIRST_RUN = REPO / "records/pipeline_runs/prct_hangzhou_matched_h120_20260818_212203"
SUPPLEMENT = REPO / "data/processed/hangzhou_real_5816_stage3_queue_wait_rule_gap2_p60_20260531.jsonl"
SOURCE_STAGE1 = REPO / "records/pipeline_runs/stage1_pt32_hd256_20260812/best.pt"
LLM_PATH = "/home/data/model/qwen-vl-8B"
ACTION_LIST = ("ETWT", "NTST", "ELWL", "NLSL")
METRICS = {
    "ATT": "test_avg_travel_time_over",
    "AQL": "test_avg_queue_len_over",
    "AWT": "test_avg_waiting_time_over",
}
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
ARM_TO_P2_DIR = {
    "frozen": "train_frozen",
    "static_prct": "train_static_prct",
    "jscd": "train_jscd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Balanced JSCD P2 + Hangzhou_5816 data-scale experiment"
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p2_root", default=str(P2_ROOT))
    parser.add_argument("--first_run", default=str(FIRST_RUN))
    parser.add_argument("--supplement", default=str(SUPPLEMENT))
    parser.add_argument("--source_stage1_ckpt", default=str(SOURCE_STAGE1))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--llm_path", default=LLM_PATH)
    parser.add_argument("--training_seeds", type=int, nargs="+", default=[2024, 42, 123])
    parser.add_argument("--arms", nargs="+", choices=sorted(ARM_TO_P2_DIR),
                        default=["frozen", "static_prct", "jscd"])
    parser.add_argument("--supplement_snapshots", type=int, default=86)
    parser.add_argument("--selection_seed", type=int, default=20260820)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--train_gpus", default="0,1,2")
    parser.add_argument("--eval_gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_timeout_s", type=int, default=21600)
    parser.add_argument("--eval_timeout_s", type=int, default=7200)
    parser.add_argument("--prepare_only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            count += 1
    return count


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            records.append(record)
    return records


def canonical_sequence(history: Sequence[Mapping[str, Any]], seq_len: int = 16) -> List[List[float]]:
    movement_keys = ("NT", "NL", "ST", "SL", "ET", "EL", "WT", "WL")
    window = list(history[-seq_len:])
    if not window:
        window = [{}]
    if len(window) < seq_len:
        window = [window[0]] * (seq_len - len(window)) + window
    return [
        [float((row or {}).get(key, 0.0) or 0.0) for key in movement_keys]
        for row in window
    ]


def is_complete_label(record: Mapping[str, Any]) -> bool:
    if str(record.get("expert_action") or "") not in ACTION_LIST:
        return False
    if not str(record.get("snapshot_path") or ""):
        return False
    candidate_waits = record.get("candidate_action_waits")
    candidate_metrics = record.get("candidate_action_metrics")
    if not isinstance(candidate_waits, dict) or not isinstance(candidate_metrics, dict):
        return False
    for action in ACTION_LIST:
        if action not in candidate_waits or action not in candidate_metrics:
            return False
        if not isinstance(candidate_metrics[action], dict) or not candidate_metrics[action]:
            return False
        try:
            float(candidate_waits[action])
        except (TypeError, ValueError):
            return False
    return True


def load_complete_supplement(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Keep the original rolling movement history before filtering label rows.

    The older 5816 file carries unlabeled rows between labeled decision states.
    Injecting a canonical explicit sequence for each selected label preserves the
    sequence that GroupedTrafficActionDataset would have built from the source.
    """
    histories: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}") from exc
            if not isinstance(record, dict):
                raise RuntimeError(f"Expected object at {path}:{line_no}")
            intersection = str(record.get("intersection_id") or record.get("road_id") or "")
            history = histories[intersection]
            history.append(record.get("movement_counts") or {})
            if len(history) > ARCH_ARGS["seq_len"]:
                del history[:-ARCH_ARGS["seq_len"]]
            if not is_complete_label(record):
                continue
            snapshot = str(record["snapshot_path"])
            enriched = dict(record)
            enriched["sequences"] = canonical_sequence(history)
            enriched["traffic_file"] = "anon_4_4_hangzhou_real_5816"
            groups[snapshot].append(enriched)

    complete: Dict[str, List[Dict[str, Any]]] = {}
    for snapshot, records in groups.items():
        intersections = {str(record.get("intersection_id") or "") for record in records}
        if len(records) != 16 or len(intersections) != 16:
            continue
        if any(not is_complete_label(record) for record in records):
            continue
        complete[snapshot] = records
    if not complete:
        raise RuntimeError(f"No complete 16-intersection snapshots found in {path}")
    return complete


def choose_snapshots(groups: Mapping[str, List[Dict[str, Any]]], count: int, seed: int) -> List[str]:
    if count <= 0:
        raise ValueError("--supplement_snapshots must be positive")
    if count > len(groups):
        raise RuntimeError(f"Requested {count} supplement snapshots, but only {len(groups)} are complete")
    ordered = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    return sorted(ordered[:count])


def split_snapshots(snapshots: Sequence[str], seed: int, val_ratio: float) -> tuple[List[str], List[str]]:
    if not 0.0 < val_ratio < 0.5:
        raise ValueError("--val_ratio must be in (0, 0.5)")
    shuffled = list(snapshots)
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * val_ratio)))
    if n_val >= len(shuffled):
        raise RuntimeError("Validation split consumes all supplement snapshots")
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def primary_paths(p2_root: Path, arm: str, seed: int) -> tuple[Path, Path]:
    root = p2_root / ARM_TO_P2_DIR[arm] / "c30" / "k86" / f"tr{seed}" / "data"
    return root / "train_data.jsonl", root / "val_data.jsonl"


def init_checkpoint(first_run: Path, arm: str, seed: int) -> Path:
    if arm == "frozen":
        return first_run / "train_frozen_neighbor_h120" / "c30" / "k118" / f"tr{seed}" / "checkpoints/best.pt"
    return first_run / "train_prct_h120" / "c30" / "k118" / f"tr{seed}" / "checkpoints/best.pt"


def verify_python(python: str) -> Path:
    candidate = Path(python)
    if not candidate.is_file():
        raise RuntimeError(f"Python interpreter not found: {candidate}")
    check = subprocess.run(
        [str(candidate), "-c", "import cityflow; print(cityflow.Engine)"],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check.returncode != 0:
        raise RuntimeError(f"CityFlow unavailable in {candidate}: {check.stderr.strip()}")
    return candidate


def train_command(
    python: Path,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path],
    checkpoint_dir: Path,
    init_ckpt: Path,
    source_stage1: Path,
    llm_path: str,
    seed: int,
    gpus: str,
    epochs: int,
) -> List[str]:
    gpu_count = len([gpu for gpu in gpus.split(",") if gpu.strip()])
    torchrun = python.with_name("torchrun")
    if not torchrun.is_file():
        raise RuntimeError(f"torchrun not found beside {python}")
    port = 29500 + (int(hashlib.sha1(str(checkpoint_dir).encode()).hexdigest()[:8], 16) % 4000)
    command = [
        str(torchrun), f"--nproc_per_node={gpu_count}", f"--master_port={port}",
        str(REPO / "projection_train/train_projection_bc.py"),
        "--train_data_path", *[str(path) for path in train_paths],
        "--val_data_path", *[str(path) for path in val_paths],
        "--llm_path", llm_path,
        "--align_ckpt", str(source_stage1),
        "--init_policy_ckpt", str(init_ckpt),
        "--output_dir", str(checkpoint_dir),
        "--prompt_mode", "queue_wait_short",
        "--batch_size", "1", "--grad_accum_steps", "16",
        "--epochs", str(epochs),
        "--lr", "1e-5", "--min_lr", "1e-6", "--warmup_ratio", "0.1",
        "--lr_schedule", "cosine", "--weight_decay", "1e-4", "--max_grad_norm", "1.0",
        "--num_workers", "2", "--seed", str(seed), "--bf16",
        "--log_interval", "10", "--save_every", "100", "--checkpoint_slim",
        "--target_mode", "mixed", "--soft_target_alpha", "0.3",
        "--soft_label_temperature", "0.35", "--ranking_loss_weight", "0.05",
        "--ranking_min_cost_gap", "0.02",
        "--cost_att_weight", "1.0", "--cost_aql_weight", "1.0",
        "--cost_awt_weight", "1.0", "--cost_spillback_weight", "0.2",
        "--base_policy_kl_weight", "0.0", "--base_policy_kl_mode", "expert_match",
        "--base_policy_kl_low_weight_threshold", "1.0",
        "--best_metric", "val_top1_cost_regret", "--best_mode", "min",
        "--ddp_timeout_seconds", "1800",
    ]
    for key, value in ARCH_ARGS.items():
        command.extend([f"--{key}", str(value)])
    return command


def evaluation_command(
    python: Path, checkpoint: Path, result_json: Path, seed: int, llm_path: str
) -> List[str]:
    command = [
        str(python), str(REPO / "projection_train/run_projection_online.py"),
        "--memo", f"jscd_mix5816_{uuid.uuid4().hex[:8]}",
        "--projection_ckpt", str(checkpoint), "--llm_path", llm_path,
        "--dataset", "hangzhou", "--traffic_file", "anon_4_4_hangzhou_real.json",
        "--run_counts", "3600", "--num_rounds", "1", "--seed", str(seed),
        "--prompt_mode", "queue_wait_short", "--inference_batch_size", "64", "--bf16",
        "--result_json", str(result_json),
    ]
    for key, value in ARCH_ARGS.items():
        command.extend([f"--{key}", str(value)])
    return command


def run_logged(command: Sequence[str], log_path: Path, env: Mapping[str, str], timeout_s: int, desc: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== {desc} @ {datetime.now().isoformat()} =====\n")
        handle.write(" ".join(command) + "\n")
        completed = subprocess.run(
            list(command), cwd=REPO, env=dict(env), stdout=handle,
            stderr=subprocess.STDOUT, text=True, timeout=timeout_s,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"{desc} failed (exit {completed.returncode}); see {log_path}")


def checkpoint_complete(path: Path, source_stage1: Path) -> bool:
    return path.is_file() and path.stat().st_size >= int(0.9 * source_stage1.stat().st_size)


def read_metrics(path: Path) -> Dict[str, float]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    values = raw.get("results", raw)
    return {name: float(values[key]) for name, key in METRICS.items()}


def aggregate(values: Mapping[str, List[Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for arm, rows in values.items():
        if not rows:
            continue
        result[arm] = {
            metric: sum(row[metric] for row in rows) / len(rows)
            for metric in METRICS
        }
        result[arm]["n"] = float(len(rows))
    return result


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    p2_root = Path(args.p2_root).resolve()
    first_run = Path(args.first_run).resolve()
    supplement_path = Path(args.supplement).resolve()
    source_stage1 = Path(args.source_stage1_ckpt).resolve()
    existing = [path for path in output_dir.iterdir()] if output_dir.exists() else []
    if any(path.name != "launch.log" for path in existing):
        raise RuntimeError(f"Output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in (p2_root, first_run, supplement_path, source_stage1):
        if not path.exists():
            raise FileNotFoundError(path)
    python = verify_python(args.python)

    config = vars(args).copy()
    config.update({
        "created_at": datetime.now().isoformat(),
        "primary_online_test": "anon_4_4_hangzhou_real.json",
        "supplement_traffic": "anon_4_4_hangzhou_real_5816.json",
        "source_hashes": {
            "supplement": sha256_file(supplement_path),
            "source_stage1": sha256_file(source_stage1),
        },
    })
    write_json(output_dir / "config.json", config)

    groups = load_complete_supplement(supplement_path)
    selected = choose_snapshots(groups, args.supplement_snapshots, args.selection_seed)
    selection_meta: Dict[str, Any] = {
        "complete_supplement_snapshots": len(groups),
        "complete_supplement_records": sum(len(records) for records in groups.values()),
        "selected_snapshots": selected,
        "selected_count": len(selected),
        "records_per_snapshot": 16,
        "selection_seed": args.selection_seed,
        "supplement_sha256": sha256_file(supplement_path),
    }
    write_json(output_dir / "supplement_selection.json", selection_meta)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    prepared: Dict[tuple[str, int], Dict[str, Path]] = {}
    status: Dict[str, str] = {}
    for arm in args.arms:
        for seed in args.training_seeds:
            primary_train, primary_val = primary_paths(p2_root, arm, seed)
            init_ckpt = init_checkpoint(first_run, arm, seed)
            for path in (primary_train, primary_val, init_ckpt):
                if not path.is_file():
                    raise FileNotFoundError(path)
            train_snaps, val_snaps = split_snapshots(selected, seed, args.val_ratio)
            overlap = set(train_snaps).intersection(val_snaps)
            if overlap:
                raise RuntimeError(f"5816 train/validation snapshot overlap for seed {seed}: {sorted(overlap)}")
            data_dir = output_dir / f"train_{arm}" / "c30" / "k172" / f"tr{seed}" / "data"
            supplemental_train = data_dir / "supplement_5816_train.jsonl"
            supplemental_val = data_dir / "supplement_5816_val.jsonl"
            train_rows = [record for snapshot in train_snaps for record in groups[snapshot]]
            val_rows = [record for snapshot in val_snaps for record in groups[snapshot]]
            expected_train = len(train_snaps) * 16
            expected_val = len(val_snaps) * 16
            if write_jsonl(supplemental_train, train_rows) != expected_train:
                raise RuntimeError("Unexpected 5816 train row count")
            if write_jsonl(supplemental_val, val_rows) != expected_val:
                raise RuntimeError("Unexpected 5816 validation row count")
            primary_train_n = count_lines(primary_train)
            primary_val_n = count_lines(primary_val)
            if primary_train_n != expected_train or primary_val_n != expected_val:
                raise RuntimeError(
                    f"P2 primary split mismatch for {arm}/tr{seed}: "
                    f"train={primary_train_n}, val={primary_val_n}, expected={expected_train}/{expected_val}"
                )
            meta = {
                "arm": arm,
                "training_seed": seed,
                "primary_paths": {"train": str(primary_train), "val": str(primary_val)},
                "primary_counts": {"train": primary_train_n, "val": primary_val_n},
                "supplement_paths": {"train": str(supplemental_train), "val": str(supplemental_val)},
                "supplement_counts": {"train": expected_train, "val": expected_val},
                "combined_counts": {"train": primary_train_n + expected_train, "val": primary_val_n + expected_val},
                "supplement_train_snapshots": train_snaps,
                "supplement_val_snapshots": val_snaps,
                "init_checkpoint": str(init_ckpt),
                "init_checkpoint_sha256": sha256_file(init_ckpt),
                "source_hashes": {
                    "primary_train": sha256_file(primary_train),
                    "primary_val": sha256_file(primary_val),
                    "supplement_train": sha256_file(supplemental_train),
                    "supplement_val": sha256_file(supplemental_val),
                },
            }
            write_json(data_dir / "dataset_manifest.json", meta)
            from projection_train.grouped_dataset import GroupedTrafficActionDataset

            train_dataset = GroupedTrafficActionDataset([str(primary_train), str(supplemental_train)])
            val_dataset = GroupedTrafficActionDataset([str(primary_val), str(supplemental_val)])
            if len(train_dataset) != primary_train_n + expected_train:
                raise RuntimeError(
                    f"Mixed train loader dropped records for {arm}/tr{seed}: "
                    f"{len(train_dataset)} != {primary_train_n + expected_train}"
                )
            if len(val_dataset) != primary_val_n + expected_val:
                raise RuntimeError(
                    f"Mixed validation loader dropped records for {arm}/tr{seed}: "
                    f"{len(val_dataset)} != {primary_val_n + expected_val}"
                )
            prepared[(arm, seed)] = {
                "primary_train": primary_train,
                "primary_val": primary_val,
                "supplement_train": supplemental_train,
                "supplement_val": supplemental_val,
                "init_ckpt": init_ckpt,
            }
            status[f"{arm}/tr{seed}"] = "PREPARED"
    write_json(output_dir / "status.json", status)
    print(f"Prepared balanced mixed data: 86 primary + {len(selected)} supplement snapshots per arm/seed", flush=True)

    if args.prepare_only:
        return 0

    current_rows: Dict[str, List[Dict[str, float]]] = {arm: [] for arm in args.arms}
    baseline_rows: Dict[str, List[Dict[str, float]]] = {arm: [] for arm in args.arms}
    for arm in args.arms:
        for seed in args.training_seeds:
            tag = f"{arm}/tr{seed}"
            paths = prepared[(arm, seed)]
            run_dir = output_dir / f"train_{arm}" / "c30" / "k172" / f"tr{seed}"
            checkpoint_dir = run_dir / "checkpoints"
            best_pt = checkpoint_dir / "best.pt"
            try:
                if not checkpoint_complete(best_pt, source_stage1):
                    command = train_command(
                        python, [paths["primary_train"], paths["supplement_train"]],
                        [paths["primary_val"], paths["supplement_val"]], checkpoint_dir,
                        paths["init_ckpt"], source_stage1, args.llm_path,
                        seed, args.train_gpus, args.epochs,
                    )
                    train_env = dict(env)
                    train_env["CUDA_VISIBLE_DEVICES"] = args.train_gpus
                    print(f"[{datetime.now().isoformat()}] train {tag} on GPUs {args.train_gpus}", flush=True)
                    run_logged(command, checkpoint_dir / "train.log", train_env, args.train_timeout_s, f"train {tag}")
                else:
                    print(f"[{datetime.now().isoformat()}] existing complete checkpoint, skip train {tag}", flush=True)
                evaluation_dir = run_dir / "evaluation_hangzhou_real"
                result_json = evaluation_dir / "result_seed2024.json"
                if not result_json.is_file():
                    eval_env = dict(env)
                    eval_env["CUDA_VISIBLE_DEVICES"] = str(args.eval_gpu)
                    print(f"[{datetime.now().isoformat()}] evaluate {tag} on Hangzhou_real GPU {args.eval_gpu}", flush=True)
                    run_logged(
                        evaluation_command(python, best_pt, result_json, 2024, args.llm_path),
                        evaluation_dir / "eval_seed2024.log", eval_env, args.eval_timeout_s, f"eval {tag}",
                    )
                metrics = read_metrics(result_json)
                write_json(evaluation_dir / "eval_summary.json", {"mean": metrics, "seed": 2024})
                current_rows[arm].append(metrics)
                baseline_result = p2_root / ARM_TO_P2_DIR[arm] / "c30" / "k86" / f"tr{seed}" / "evaluation" / "result_seed2024.json"
                if baseline_result.is_file():
                    baseline_rows[arm].append(read_metrics(baseline_result))
                status[tag] = "OK"
            except Exception as exc:  # Keep independent arms/seeds running after one failure.
                status[tag] = f"FAILED: {type(exc).__name__}: {exc}"
                print(f"[{datetime.now().isoformat()}] {tag} FAILED: {exc}", flush=True)
            write_json(output_dir / "status.json", status)

    current_aggregate = aggregate(current_rows)
    baseline_aggregate = aggregate(baseline_rows)
    deltas: Dict[str, Dict[str, float]] = {}
    for arm in args.arms:
        if arm not in current_aggregate or arm not in baseline_aggregate:
            continue
        deltas[arm] = {
            metric: current_aggregate[arm][metric] - baseline_aggregate[arm][metric]
            for metric in METRICS
        }
    paired_jscd: Dict[str, List[float]] = {}
    if len(current_rows.get("jscd", [])) == len(current_rows.get("static_prct", [])):
        paired_jscd["jscd_minus_static_prct"] = [
            left["AWT"] - right["AWT"]
            for left, right in zip(current_rows["jscd"], current_rows["static_prct"])
        ]
    if len(current_rows.get("jscd", [])) == len(current_rows.get("frozen", [])):
        paired_jscd["jscd_minus_frozen"] = [
            left["AWT"] - right["AWT"]
            for left, right in zip(current_rows["jscd"], current_rows["frozen"])
        ]
    write_json(output_dir / "aggregate.json", {
        "current": current_aggregate,
        "p2_baseline": baseline_aggregate,
        "mixed_minus_p2": deltas,
        "paired_awt": paired_jscd,
        "metric_direction": "lower_is_better",
        "primary_online_test": "anon_4_4_hangzhou_real.json",
    })
    print(f"Done. Output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
