#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_DEFAULT = Path("/home/code/LLMSTSC")
DATA_RELATIVE = Path("data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl")
MODEL_DEFAULT = Path("/mnt/data/Qwen3-VL-8B-Instruct")
SEEDS = (2024, 2025, 2026)
PERCENTAGES = (10, 25, 50, 100)
STAGE2_EPOCHS = 15
STAGE1_EPOCHS = 15
SEQ_LEN = 16
SPLIT_GAP_TIMESTAMPS = SEQ_LEN - 1
REPRESENTATION_ARMS = ("normal_stage1", "random_initialization", "shuffled_pair_stage1")
REPRESENTATION_PROTOCOL_VERSION = "same_probe_initialization_v2"
STATUS_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def event(root: Path, stage: str, state: str, *, gpu: int | None = None,
          task: str = "", detail: str = "", path: str = "") -> None:
    row = {"time": utc_now(), "stage": stage, "status": state, "gpu": gpu,
           "task": task, "detail": detail, "path": path}
    append_jsonl(root / "status" / "events.jsonl", row)


def dump_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def write_data_splits(repo: Path, root: Path, source: Path) -> dict[str, Any]:
    splits_root = root / "splits"
    split_manifest_path = splits_root / "manifest.json"
    if split_manifest_path.exists():
        old = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        if old.get("source_sha256") != sha256(source):
            raise RuntimeError("The source dataset changed after split creation; refusing an unsafe resume.")
        if old.get("timestamp_counts"):
            return old

    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    records.append(item)
    timestamp_values = sorted({str(item.get("date") or "") for item in records})
    if len(timestamp_values) < 100:
        raise RuntimeError(f"Too few unique timestamps for blocked evaluation: {len(timestamp_values)}")

    usable = len(timestamp_values) - 2 * SPLIT_GAP_TIMESTAMPS
    train_count = int(round(usable * 0.70))
    val_count = int(round(usable * 0.15))
    test_count = usable - train_count - val_count
    train_times = timestamp_values[:train_count]
    val_start = train_count + SPLIT_GAP_TIMESTAMPS
    val_times = timestamp_values[val_start:val_start + val_count]
    test_start = val_start + val_count + SPLIT_GAP_TIMESTAMPS
    test_times = timestamp_values[test_start:]
    if len(test_times) != test_count:
        raise RuntimeError("Internal split accounting error.")
    assignments = {stamp: "train" for stamp in train_times}
    assignments.update({stamp: "validation" for stamp in val_times})
    assignments.update({stamp: "test" for stamp in test_times})
    time_indices = {stamp: i for i, stamp in enumerate(timestamp_values)}
    first_indices = {
        "train": time_indices[train_times[0]],
        "validation": time_indices[val_times[0]],
        "test": time_indices[test_times[0]],
    }
    split_rows: dict[str, list[dict[str, Any]]] = {key: [] for key in ("train", "validation", "test")}
    decision_ids: list[tuple[int, str]] = []
    split_line_counts = Counter()
    split_decision_counts = Counter()
    for record_index, raw in enumerate(records):
        stamp = str(raw.get("date") or "")
        split_name = assignments.get(stamp)
        if split_name is None:
            continue
        row = dict(raw)
        split_line_counts[split_name] += 1
        warmup = time_indices[stamp] - first_indices[split_name] < SPLIT_GAP_TIMESTAMPS
        if warmup:
            row["expert_action"] = ""
            row["clip_caption"] = ""
        split_rows[split_name].append(row)
        if split_name == "train" and not warmup and str(row.get("expert_action") or "").strip() and str(row.get("clip_caption") or "").strip():
            decision_ids.append((len(split_rows[split_name]) - 1, str(row["expert_action"]).strip()))
            split_decision_counts[split_name] += 1
        elif split_name in ("validation", "test") and not warmup and str(row.get("expert_action") or "").strip() and str(row.get("clip_caption") or "").strip():
            split_decision_counts[split_name] += 1

    if not decision_ids:
        raise RuntimeError("No labeled training decisions remain after temporal warmup purge.")
    splits_root.mkdir(parents=True, exist_ok=True)
    for name in ("validation", "test"):
        dump_jsonl(splits_root / f"{name}.jsonl", split_rows[name])

    per_seed: dict[str, Any] = {}
    row_to_action = {row_id: action for row_id, action in decision_ids}
    grouped_ids: dict[str, list[int]] = defaultdict(list)
    for row_id, action in decision_ids:
        grouped_ids[action].append(row_id)
    for seed in SEEDS:
        rng = random.Random(seed)
        class_order: dict[str, list[int]] = {}
        for action, ids in grouped_ids.items():
            ordered = list(ids)
            rng.shuffle(ordered)
            class_order[action] = ordered
        seed_root = splits_root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        shutil.copy2(splits_root / "validation.jsonl", seed_root / "validation.jsonl")
        shutil.copy2(splits_root / "test.jsonl", seed_root / "test.jsonl")
        per_seed[str(seed)] = {"train": {}, "source_seed": seed}
        for percentage in PERCENTAGES:
            selected: set[int] = set()
            for action, ordered in class_order.items():
                amount = len(ordered) if percentage == 100 else max(1, int(round(len(ordered) * percentage / 100.0)))
                selected.update(ordered[:amount])
            output_rows = []
            for row_id, source_row in enumerate(split_rows["train"]):
                row = dict(source_row)
                if row_id in row_to_action and row_id not in selected:
                    row["expert_action"] = ""
                    row["clip_caption"] = ""
                output_rows.append(row)
            out_path = seed_root / f"train_{percentage:03d}.jsonl"
            dump_jsonl(out_path, output_rows)
            per_seed[str(seed)]["train"][str(percentage)] = {
                "path": str(out_path), "selected_decision_samples": len(selected),
                "requested_percentage": percentage,
                "actual_percentage": round(100.0 * len(selected) / len(decision_ids), 4),
                "class_counts": dict(Counter(row_to_action[i] for i in selected)),
                "sha256": sha256(out_path),
            }

        full_train = seed_root / "train_100.jsonl"
        full_rows = [dict(row) for row in split_rows["train"]]
        paired_ids = [row_id for row_id, _ in decision_ids]
        captions = [str(full_rows[row_id].get("clip_caption") or "") for row_id in paired_ids]
        if len(captions) < 2:
            raise RuntimeError(f"Not enough caption pairs for shuffled control at seed {seed}.")
        candidate_shifts = range(1, len(captions))
        shift = min(candidate_shifts, key=lambda amount: sum(
            captions[i] == captions[(i + amount) % len(captions)] for i in range(len(captions))
        ))
        shuffled = captions[shift:] + captions[:shift]
        for row_id, caption in zip(paired_ids, shuffled):
            full_rows[row_id]["clip_caption"] = caption
        shuffled_path = seed_root / "train_100_shuffled.jsonl"
        dump_jsonl(shuffled_path, full_rows)
        per_seed[str(seed)]["shuffled_train"] = {
            "path": str(shuffled_path), "rotation": shift,
            "unchanged_pairs": sum(a == b for a, b in zip(captions, shuffled)),
            "pair_count": len(captions), "sha256": sha256(shuffled_path),
        }

    train_set, val_set, test_set = set(train_times), set(val_times), set(test_times)
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise RuntimeError("Timestamp split overlap detected.")
    manifest = {
        "schema": "stage1_validation_split_v1",
        "created_at": utc_now(),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_rows": len(records),
        "unique_timestamps": len(timestamp_values),
        "timestamp_counts": {
            "train": len(train_times), "validation": len(val_times),
            "test": len(test_times),
            "purged": len(timestamp_values) - len(train_times) - len(val_times) - len(test_times),
        },
        "timestamp_resolution_seconds": 2,
        "sequence_length": SEQ_LEN,
        "boundary_gap_timestamps": SPLIT_GAP_TIMESTAMPS,
        "boundary_gap_seconds": SPLIT_GAP_TIMESTAMPS * 2,
        "protocol": "chronological 70/15/15 with 30-second embargo at each split boundary and 30-second per-split history warmup",
        "timestamp_ranges": {
            "train": [train_times[0], train_times[-1]],
            "validation": [val_times[0], val_times[-1]],
            "test": [test_times[0], test_times[-1]],
            "purged": [stamp for stamp in timestamp_values if stamp not in assignments],
        },
        "rows": dict(split_line_counts),
        "decision_samples_after_warmup": dict(split_decision_counts),
        "validation_sha256": sha256(splits_root / "validation.jsonl"),
        "test_sha256": sha256(splits_root / "test.jsonl"),
        "seeds": per_seed,
    }
    write_json(split_manifest_path, manifest)
    return manifest


def detect_idle_gpus() -> list[dict[str, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    idle = []
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) != 3:
            continue
        gpu, memory, utilization = (int(value) for value in fields)
        if memory <= 4096 and utilization <= 10:
            idle.append({"index": gpu, "memory_used_mib": memory, "utilization_percent": utilization})
    if not idle:
        raise RuntimeError("No idle GPUs found (requires <=4096 MiB and <=10% utilization).")
    return idle


def attempt_directories(base: Path) -> list[Path]:
    return sorted(base.glob("attempt_*")) if base.exists() else []


def completed_alignment(base: Path) -> Path | None:
    for path in reversed(attempt_directories(base)):
        summary_path = path / "run_summary.json"
        if (path / "best.pt").is_file() and summary_path.is_file():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if summary.get("status") == "success":
                return path
    return None


def completed_stage2(base: Path) -> Path | None:
    for path in reversed(attempt_directories(base)):
        summary_path = path / "run_summary.json"
        if not ((path / "best.pt").is_file() and (path / "test_metrics.json").is_file() and summary_path.is_file()):
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("status") == "success":
            return path
    return None


def next_attempt(base: Path) -> Path:
    attempts = attempt_directories(base)
    number = max([int(path.name.split("_")[-1]) for path in attempts] + [0]) + 1
    path = base / f"attempt_{number:03d}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def resumable_attempt(base: Path, complete_check: Any) -> Path | None:
    for path in reversed(attempt_directories(base)):
        if complete_check(base) == path:
            return None
        if (path / "last.pt").is_file():
            return path
    return None


def run_bounded_command(command: list[str], repo: Path, output_dir: Path, gpu: int,
                        timeout_seconds: int = 12 * 60 * 60) -> tuple[int, float]:
    log_path = output_dir / "training.log"
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(repo),
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "4",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    })
    started = time.perf_counter()
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            code = 124
    return code, time.perf_counter() - started


def stage1_command(repo: Path, output: Path, train_path: Path, val_path: Path,
                   model_path: Path, seed: int) -> list[str]:
    return [
        sys.executable, str(repo / "train_align.py"),
        "--train_data_path", str(train_path), "--val_data_path", str(val_path),
        "--model_path", str(model_path), "--output_dir", str(output),
        "--decision_only", "--seq_len", str(SEQ_LEN), "--image_size", "224",
        "--periodicity", str(SEQ_LEN), "--input_dim", "8", "--hidden_dim", "256",
        "--prefix_tokens", "8", "--prefix_mode", "prefix", "--max_length", "512",
        "--temperature", "0.05", "--action_aux_weight", "0", "--batch_size", "16",
        "--grad_accum_steps", "1", "--epochs", str(STAGE1_EPOCHS), "--lr", "1e-4",
        "--min_lr", "1e-5", "--warmup_ratio", "0.1", "--lr_schedule", "cosine",
        "--num_workers", "2", "--eval_max_samples", "2000", "--save_every_epochs", "1",
        "--log_every_steps", "10", "--seed", str(seed), "--bf16",
    ]


def stage2_command(repo: Path, output: Path, train_path: Path, val_path: Path,
                   test_path: Path, model_path: Path, seed: int,
                   align_checkpoint: Path | None) -> list[str]:
    init_args = ["--init_mode", "random"] if align_checkpoint is None else [
        "--init_mode", "align", "--align_ckpt", str(align_checkpoint)
    ]
    return [
        sys.executable, str(repo / "projection_train" / "train_projection_bc.py"),
        "--train_data_path", str(train_path), "--val_data_path", str(val_path),
        "--test_data_path", str(test_path), "--llm_path", str(model_path), *init_args,
        "--output_dir", str(output), "--seq_len", str(SEQ_LEN), "--image_size", "224",
        "--periodicity", str(SEQ_LEN), "--input_dim", "8", "--hidden_dim", "256",
        "--prefix_tokens", "8", "--prefix_mode", "prefix", "--max_length", "512",
        "--prompt_mode", "queue_wait_short", "--action_eval_chunk_size", "0",
        "--batch_size", "1", "--grad_accum_steps", "8", "--epochs", str(STAGE2_EPOCHS),
        "--lr", "1e-4", "--min_lr", "1e-5", "--warmup_ratio", "0.1",
        "--lr_schedule", "cosine", "--weight_decay", "1e-4", "--max_grad_norm", "1",
        "--val_ratio", "0.1", "--split_strategy", "blocked", "--num_workers", "2",
        "--target_mode", "mixed", "--soft_target_alpha", "0.3",
        "--soft_label_temperature", "0.7", "--ranking_loss_weight", "0",
        "--base_policy_kl_weight", "0", "--resample_mode", "none",
        "--best_metric", "val_top1_cost_regret", "--best_mode", "min",
        "--log_interval", "25", "--save_every", "1", "--seed", str(seed),
        "--bf16", "--ddp_timeout_seconds", "1800",
    ]


def run_alignment(repo: Path, root: Path, base: Path, train: Path, val: Path,
                  model_path: Path, seed: int, gpu: int, task_name: str) -> Path:
    done = completed_alignment(base)
    if done:
        event(root, "stage1", "skipped_complete", gpu=gpu, task=task_name, path=str(done))
        return done
    output = resumable_attempt(base, completed_alignment) or next_attempt(base)
    command = stage1_command(repo, output, train, val, model_path, seed)
    if (output / "last.pt").is_file():
        command.extend(["--resume_from", str(output / "last.pt")])
    write_json(output / "command.json", {"command": command, "gpu": gpu, "seed": seed})
    event(root, "stage1", "running", gpu=gpu, task=task_name, path=str(output), detail="15 epochs; fixed train/validation partitions")
    code, duration = run_bounded_command(command, repo, output, gpu)
    if code != 0 or not (output / "best.pt").is_file() or not (output / "run_summary.json").is_file():
        event(root, "stage1", "failed", gpu=gpu, task=task_name,
              path=str(output), detail=f"exit_code={code}; wall_seconds={duration:.1f}")
        raise RuntimeError(f"Stage I failed; see {output / 'training.log'} (exit {code}).")
    event(root, "stage1", "complete", gpu=gpu, task=task_name,
          path=str(output), detail=f"wall_seconds={duration:.1f}")
    return output


def run_stage2(repo: Path, root: Path, base: Path, train: Path, val: Path, test: Path,
               model_path: Path, seed: int, gpu: int, task_name: str,
               align_checkpoint: Path | None) -> Path:
    done = completed_stage2(base)
    if done:
        event(root, "stage2", "skipped_complete", gpu=gpu, task=task_name, path=str(done))
        return done
    output = resumable_attempt(base, completed_stage2) or next_attempt(base)
    command = stage2_command(repo, output, train, val, test, model_path, seed, align_checkpoint)
    if (output / "last.pt").is_file():
        command.extend(["--resume_from", str(output / "last.pt")])
    write_json(output / "command.json", {"command": command, "gpu": gpu, "seed": seed,
                                           "align_checkpoint": str(align_checkpoint) if align_checkpoint else None})
    event(root, "stage2", "running", gpu=gpu, task=task_name, path=str(output), detail="15 epochs; validation selects best; test is held out")
    code, duration = run_bounded_command(command, repo, output, gpu)
    if code != 0 or not completed_stage2(output.parent):
        event(root, "stage2", "failed", gpu=gpu, task=task_name,
              path=str(output), detail=f"exit_code={code}; wall_seconds={duration:.1f}")
        raise RuntimeError(f"Stage II failed; see {output / 'training.log'} (exit {code}).")
    event(root, "stage2", "complete", gpu=gpu, task=task_name,
          path=str(output), detail=f"wall_seconds={duration:.1f}")
    return output


def task_root(root: Path, task: dict[str, Any]) -> Path:
    seed = int(task["seed"])
    if task["kind"] == "exp1":
        return root / "experiment1" / f"data_{int(task['percentage']):03d}pct" / f"seed_{seed}" / str(task["arm"])
    return root / "experiment2" / f"seed_{seed}" / "shuffled_stage1"


def process_training_task(repo: Path, root: Path, task: dict[str, Any], model_path: Path,
                          split_manifest: dict[str, Any], gpu: int) -> None:
    seed = int(task["seed"])
    seed_dir = root / "splits" / f"seed_{seed}"
    val_path, test_path = seed_dir / "validation.jsonl", seed_dir / "test.jsonl"
    base = task_root(root, task)
    label = str(task["key"])
    if task["kind"] == "exp1":
        percentage = int(task["percentage"])
        train_path = seed_dir / f"train_{percentage:03d}.jsonl"
        align_checkpoint = None
        if task["arm"] == "full":
            align_dir = run_alignment(
                repo, root, base / "stage1", train_path, val_path, model_path, seed, gpu, label
            )
            align_checkpoint = align_dir / "best.pt"
        stage2_dir = run_stage2(
            repo, root, base / "stage2", train_path, val_path, test_path,
            model_path, seed, gpu, label, align_checkpoint,
        )
        write_json(base / "task_result.json", {
            "status": "success", "kind": "exp1", "seed": seed, "percentage": percentage,
            "arm": task["arm"], "stage1_dir": str(align_dir) if task["arm"] == "full" else None,
            "stage2_dir": str(stage2_dir),
            "stage1_checkpoint_sha256": sha256(align_dir / "best.pt") if task["arm"] == "full" else None,
            "stage2_checkpoint_sha256": sha256(stage2_dir / "best.pt"),
            "train_samples": split_manifest["seeds"][str(seed)]["train"][str(percentage)]["selected_decision_samples"],
        })
    else:
        train_info = split_manifest["seeds"][str(seed)]["shuffled_train"]
        align_dir = run_alignment(
            repo, root, base, Path(train_info["path"]), val_path,
            model_path, seed, gpu, label,
        )
        write_json(base / "task_result.json", {
            "status": "success", "kind": "exp2_shuffled", "seed": seed,
            "stage1_dir": str(align_dir), "stage1_checkpoint_sha256": sha256(align_dir / "best.pt"),
            "train_samples": train_info["pair_count"],
        })


def build_task_list() -> list[dict[str, Any]]:
    tasks = []
    for percentage in PERCENTAGES:
        for seed in SEEDS:
            for arm in ("full", "without_stage1"):
                tasks.append({
                    "kind": "exp1", "percentage": percentage, "seed": seed,
                    "arm": arm, "key": f"exp1_p{percentage:03d}_s{seed}_{arm}",
                })
    for seed in SEEDS:
        tasks.append({"kind": "exp2_shuffle", "seed": seed,
                      "key": f"exp2_shuffled_stage1_s{seed}"})
    return tasks


def run_training_queue(repo: Path, root: Path, tasks: list[dict[str, Any]],
                       model_path: Path, split_manifest: dict[str, Any], gpu_ids: list[int]) -> list[str]:
    pending = deque(tasks)
    queue_lock = threading.Lock()
    failures: list[str] = []
    failure_lock = threading.Lock()

    def lane(gpu: int) -> None:
        while True:
            with queue_lock:
                if not pending:
                    return
                task = pending.popleft()
            event(root, "task", "started", gpu=gpu, task=task["key"])
            try:
                process_training_task(repo, root, task, model_path, split_manifest, gpu)
                event(root, "task", "complete", gpu=gpu, task=task["key"])
            except Exception as exc:
                with failure_lock:
                    failures.append(task["key"])
                event(root, "task", "failed", gpu=gpu, task=task["key"], detail=str(exc))

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(lane, gpu) for gpu in gpu_ids]
        for future in futures:
            future.result()
    return failures


def idle_gpu_inventory() -> list[dict[str, int]]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=10,
    )
    inventory = []
    for line in result.stdout.splitlines():
        parts = [value.strip() for value in line.split(",")]
        if len(parts) == 3:
            inventory.append({"index": int(parts[0]), "memory_used_mib": int(parts[1]), "utilization_percent": int(parts[2])})
    return inventory


def run_experiments(repo: Path, root: Path, model_path: Path, preflight_only: bool = False) -> int:
    source = repo / DATA_RELATIVE
    for required in (source, model_path / "config.json", repo / "train_align.py", repo / "projection_train" / "train_projection_bc.py"):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "manifest.json"
    if config_path.exists():
        manifest = json.loads(config_path.read_text(encoding="utf-8"))
        if manifest.get("source_sha256") != sha256(source) or manifest.get("model_path") != str(model_path):
            raise RuntimeError("Existing run manifest does not match current data/model; refusing to mix outputs.")
    else:
        code_paths = (
            "train_align.py", "dataset.py", "models.py", "src/TimeVLM/ts_image_adapter.py",
            "projection_train/train_projection_bc.py", "projection_train/grouped_dataset.py",
            "projection_train/prefix_action_policy.py", "projection_train/prompt_utils.py",
            "projection_train/cost_ranking.py", "recordlocal/stage1_validation.py",
        )
        package_versions = {}
        for package in ("torch", "transformers", "numpy", "matplotlib"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = None
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True,
                text=True, timeout=5, check=True,
            ).stdout.strip()
        except Exception:
            commit = None
        manifest = {
            "schema": "stage1_validation_run_v1", "created_at": utc_now(),
            "project": str(repo), "source": str(source), "source_sha256": sha256(source),
            "model_path": str(model_path), "model_config_sha256": sha256(model_path / "config.json"),
            "git_commit_at_launch": commit,
            "runtime": {"python": sys.version, "platform": platform.platform(), "packages": package_versions},
            "seeds": list(SEEDS), "training_percentages": list(PERCENTAGES),
            "experiment1_arms": ["full", "without_stage1"],
            "experiment2_arms": ["normal_stage1", "random_initialization", "shuffled_pair_stage1"],
            "protocol": {
                "sequence_length": SEQ_LEN, "image_size": 224, "hidden_dim": 256,
                "prefix_tokens": 8, "max_length": 512, "precision": "bf16",
                "stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS,
                "stage1_batch_per_gpu": 16, "stage2_batch_per_gpu": 1,
                "stage2_gradient_accumulation": 8,
                "stage2_checkpoint_selection": "minimum validation top-1 cost regret",
                "test_set_used_for_selection": False,
                "timing_includes_stage1": True,
                "probe_labels": {
                    "congestion_level": "tertiles of max phase queue, thresholds fit on training split only",
                    "queue_length": "max phase_queue per decision sample",
                    "waiting_degree": "max phase_wait per decision sample (seconds)",
                },
            },
            "gpu_inventory_at_launch": idle_gpu_inventory(),
            "code_sha256": {
                relative: sha256(repo / relative) for relative in code_paths
            },
        }
        write_json(config_path, manifest)

    tracked_code = list(manifest.get("code_sha256", {}))
    if "recordlocal/run_stage1_validation.sh" not in tracked_code:
        tracked_code.append("recordlocal/run_stage1_validation.sh")
    manifest["code_sha256"] = {
        relative: sha256(repo / relative)
        for relative in tracked_code if (repo / relative).is_file()
    }
    manifest["code_sha256_updated_at"] = utc_now()

    split_manifest = write_data_splits(repo, root, source)
    manifest["split_manifest"] = str(root / "splits" / "manifest.json")
    manifest["split_sha256"] = sha256(root / "splits" / "manifest.json")
    write_json(config_path, manifest)
    gpu_inventory = idle_gpu_inventory()
    gpu_ids = [item["index"] for item in gpu_inventory if item["memory_used_mib"] <= 4096 and item["utilization_percent"] <= 10]
    if not gpu_ids:
        raise RuntimeError(f"No idle GPUs available. Inventory: {gpu_inventory}")
    manifest["gpu_ids_used"] = gpu_ids
    manifest["gpu_inventory_before_run"] = gpu_inventory
    write_json(config_path, manifest)
    event(root, "orchestrator", "running", detail=f"gpu_ids={gpu_ids}; training_tasks={len(build_task_list())}")
    if preflight_only:
        print(json.dumps({"status": "preflight_ok", "gpu_ids": gpu_ids,
                          "training_tasks": len(build_task_list()),
                          "manifest": str(config_path)}, indent=2))
        return 0

    failures = run_training_queue(repo, root, build_task_list(), model_path, split_manifest, gpu_ids)
    if failures:
        event(root, "representation", "skipped", detail="Training prerequisites remain incomplete: " + ",".join(failures))
    else:
        rep_failures = run_representation_queue(repo, root, model_path, gpu_ids)
        failures.extend(rep_failures)

    summarize(root)
    final_state = "failed" if failures else "complete"
    event(root, "orchestrator", final_state, detail=";".join(failures) if failures else "all required runs and reports complete")
    write_json(root / "completion.json", {"status": final_state, "ended_at": utc_now(), "failed_tasks": failures})
    return 1 if failures else 0


def normalized_caption(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def macro_f1_values(predictions: list[int], labels: list[int], classes: int = 3) -> float:
    scores = []
    for label in range(classes):
        tp = sum(p == label and y == label for p, y in zip(predictions, labels))
        fp = sum(p == label and y != label for p, y in zip(predictions, labels))
        fn = sum(p != label and y == label for p, y in zip(predictions, labels))
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return sum(scores) / max(1, len(scores))


def multi_positive_recall(image: Any, text: Any, captions: list[str]) -> dict[str, float]:
    import torch

    image = torch.nn.functional.normalize(image.float(), dim=-1)
    text = torch.nn.functional.normalize(text.float(), dim=-1)
    positive = torch.tensor(
        [[normalized_caption(a) == normalized_caption(b) for b in captions] for a in captions],
        dtype=torch.bool,
    )
    similarity = image @ text.t()
    result: dict[str, float] = {}
    count = similarity.shape[0]
    for direction, scores, matches in (
        ("i2t", similarity, positive),
        ("t2i", similarity.t(), positive.t()),
    ):
        for k in (1, 5, 10):
            top = scores.topk(min(k, scores.shape[1]), dim=1).indices
            hits = matches.gather(1, top).any(dim=1).float().mean().item() if count else 0.0
            result[f"R@{k}_{direction}"] = float(hits)
    for k in (1, 5, 10):
        result[f"R@{k}"] = 0.5 * (result[f"R@{k}_i2t"] + result[f"R@{k}_t2i"])
    return result


def collect_frozen_features(model: Any, loader: Any, device: Any) -> dict[str, Any]:
    import torch

    features, text_features, captions, queues, waits = [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            sequences = batch["sequence"].to(device, non_blocking=True)
            texts = list(batch["clip_caption"])
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_vec = model.encode_image(sequences)
                text_vec = model.encode_text(texts)
            features.append(image_vec.float().cpu())
            text_features.append(text_vec.float().cpu())
            captions.extend(texts)
            queues.extend([
                max([float(v) for v in (item or {}).values()] or [0.0])
                for item in batch["phase_queue"]
            ])
            waits.extend([
                max([float(v) for v in (item or {}).values()] or [0.0])
                for item in batch["phase_wait"]
            ])
    if not features:
        raise RuntimeError("A representation split has no labeled image/text examples.")
    return {
        "features": torch.cat(features),
        "text_features": torch.cat(text_features),
        "captions": captions,
        "queue": queues,
        "wait": waits,
    }


def fit_linear_probes(train: dict[str, Any], validation: dict[str, Any],
                      test: dict[str, Any], seed: int, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch
    from torch import nn

    x_train = train["features"].float()
    x_val = validation["features"].float()
    x_test = test["features"].float()
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-3)
    x_train = ((x_train - mean) / std).to(device)
    x_val = ((x_val - mean) / std).to(device)
    x_test = ((x_test - mean) / std).to(device)

    q_train = np.asarray(train["queue"], dtype=np.float64)
    q_thresholds = np.quantile(q_train, [1.0 / 3.0, 2.0 / 3.0]).tolist()
    if q_thresholds[0] >= q_thresholds[1]:
        unique = np.unique(q_train)
        if len(unique) >= 3:
            q_thresholds = [float(unique[len(unique) // 3]), float(unique[(2 * len(unique)) // 3])]
    q1, q2 = float(q_thresholds[0]), float(q_thresholds[1])

    def class_ids(values: list[float]) -> list[int]:
        return [0 if value < q1 else (2 if value >= q2 else 1) for value in values]

    y_train_cls = torch.tensor(class_ids(train["queue"]), dtype=torch.long, device=device)
    y_val_cls = torch.tensor(class_ids(validation["queue"]), dtype=torch.long, device=device)
    y_test_cls = torch.tensor(class_ids(test["queue"]), dtype=torch.long, device=device)
    torch.manual_seed(seed)
    cls_head = nn.Linear(x_train.shape[1], 3).to(device)
    cls_optimizer = torch.optim.AdamW(cls_head.parameters(), lr=0.02, weight_decay=0.01)
    cls_loss = nn.CrossEntropyLoss()
    best_cls_loss = float("inf")
    best_cls_state = None
    best_cls_epoch = 0
    patience = 20
    stale = 0
    for epoch in range(1, 201):
        cls_head.train()
        cls_optimizer.zero_grad(set_to_none=True)
        loss = cls_loss(cls_head(x_train), y_train_cls)
        loss.backward()
        cls_optimizer.step()
        cls_head.eval()
        with torch.no_grad():
            val_loss = float(cls_loss(cls_head(x_val), y_val_cls).item())
        if val_loss < best_cls_loss - 1e-6:
            best_cls_loss = val_loss
            best_cls_state = {key: value.detach().cpu().clone() for key, value in cls_head.state_dict().items()}
            best_cls_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_cls_state is not None
    cls_head.load_state_dict(best_cls_state)
    cls_head.eval()
    with torch.no_grad():
        val_pred = cls_head(x_val).argmax(dim=-1).cpu().tolist()
        test_pred = cls_head(x_test).argmax(dim=-1).cpu().tolist()
    val_labels = y_val_cls.cpu().tolist()
    test_labels = y_test_cls.cpu().tolist()

    y_train_reg = torch.tensor(list(zip(train["queue"], train["wait"])), dtype=torch.float32, device=device)
    y_val_reg = torch.tensor(list(zip(validation["queue"], validation["wait"])), dtype=torch.float32, device=device)
    y_test_reg = torch.tensor(list(zip(test["queue"], test["wait"])), dtype=torch.float32, device=device)
    y_mean = y_train_reg.mean(dim=0, keepdim=True)
    y_std = y_train_reg.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-3)
    y_train_scaled = (y_train_reg - y_mean) / y_std
    y_val_scaled = (y_val_reg - y_mean) / y_std
    torch.manual_seed(seed + 1)
    reg_head = nn.Linear(x_train.shape[1], 2).to(device)
    reg_optimizer = torch.optim.AdamW(reg_head.parameters(), lr=0.02, weight_decay=0.01)
    mse = nn.MSELoss()
    best_reg_loss = float("inf")
    best_reg_state = None
    best_reg_epoch = 0
    stale = 0
    for epoch in range(1, 201):
        reg_head.train()
        reg_optimizer.zero_grad(set_to_none=True)
        loss = mse(reg_head(x_train), y_train_scaled)
        loss.backward()
        reg_optimizer.step()
        reg_head.eval()
        with torch.no_grad():
            val_loss = float(mse(reg_head(x_val), y_val_scaled).item())
        if val_loss < best_reg_loss - 1e-6:
            best_reg_loss = val_loss
            best_reg_state = {key: value.detach().cpu().clone() for key, value in reg_head.state_dict().items()}
            best_reg_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_reg_state is not None
    reg_head.load_state_dict(best_reg_state)
    reg_head.eval()
    with torch.no_grad():
        val_reg = (reg_head(x_val) * y_std + y_mean).cpu().numpy()
        test_reg = (reg_head(x_test) * y_std + y_mean).cpu().numpy()

    def regression_metrics(pred: Any, target_values: Any) -> dict[str, float]:
        actual = np.asarray(target_values, dtype=np.float64)
        errors = pred - actual
        mae = np.mean(np.abs(errors), axis=0)
        denom = np.sum((actual - np.mean(actual, axis=0, keepdims=True)) ** 2, axis=0)
        numer = np.sum(errors ** 2, axis=0)
        r2 = [None if d <= 1e-12 else float(1.0 - n / d) for n, d in zip(numer, denom)]
        return {
            "queue_mae": float(mae[0]), "queue_r2": r2[0],
            "waiting_mae": float(mae[1]), "waiting_r2": r2[1],
        }

    validation_metrics = {
        "congestion_macro_f1": macro_f1_values(val_pred, val_labels),
        **regression_metrics(val_reg, list(zip(validation["queue"], validation["wait"]))),
    }
    test_metrics = {
        "congestion_macro_f1": macro_f1_values(test_pred, test_labels),
        **regression_metrics(test_reg, list(zip(test["queue"], test["wait"]))),
    }
    saved_heads = {
        "congestion": best_cls_state,
        "queue_wait": best_reg_state,
        "feature_mean": mean.cpu(), "feature_std": std.cpu(),
        "target_mean": y_mean.cpu(), "target_std": y_std.cpu(),
        "congestion_thresholds": [q1, q2],
    }
    return {
        "validation": validation_metrics,
        "test": test_metrics,
        "congestion_head_best_epoch": best_cls_epoch,
        "regression_head_best_epoch": best_reg_epoch,
        "congestion_thresholds_train_only": [q1, q2],
    }, saved_heads


def completed_representation(path: Path) -> bool:
    metrics_path = path / "metrics.json"
    if not metrics_path.is_file() or not (path / "random_initialization.pt").is_file():
        return False
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "success" or payload.get("protocol_version") != REPRESENTATION_PROTOCOL_VERSION:
        return False
    arms = payload.get("arms", {})
    return set(arms) == set(REPRESENTATION_ARMS) and all(
        (path / f"{arm}_probe_heads.pt").is_file()
        and arms[arm].get("retrieval_test")
        and arms[arm].get("linear_probe", {}).get("test")
        for arm in REPRESENTATION_ARMS
    )


def representation_attempt(repo: Path, root: Path, model_path: Path, seed: int,
                           gpu: int) -> None:
    base = root / "experiment2" / f"seed_{seed}" / "representation"
    for path in reversed(attempt_directories(base)):
        if completed_representation(path):
            event(root, "representation", "skipped_complete", gpu=gpu,
                  task=f"exp2_s{seed}", path=str(path))
            return
    output = next_attempt(base)
    command = [
        sys.executable, str(Path(__file__).resolve()), "--mode", "representation-worker",
        "--repo", str(repo), "--root", str(root), "--model", str(model_path),
        "--seed", str(seed), "--gpu", str(gpu), "--output", str(output),
    ]
    write_json(output / "command.json", {"command": command, "gpu": gpu, "seed": seed})
    event(root, "representation", "running", gpu=gpu, task=f"exp2_s{seed}", path=str(output))
    code, duration = run_bounded_command(command, repo, output, gpu, timeout_seconds=6 * 60 * 60)
    if code != 0 or not completed_representation(output):
        event(root, "representation", "failed", gpu=gpu, task=f"exp2_s{seed}",
              path=str(output), detail=f"exit_code={code}; wall_seconds={duration:.1f}")
        raise RuntimeError(f"Representation evaluation failed; see {output / 'training.log'} (exit {code}).")
    event(root, "representation", "complete", gpu=gpu, task=f"exp2_s{seed}",
          path=str(output), detail=f"wall_seconds={duration:.1f}")


def representation_worker(repo: Path, root: Path, model_path: Path, seed: int,
                          output: Path, gpu: int) -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(repo))
    from models import AlignmentModel
    from projection_train.grouped_dataset import GroupedTrafficActionDataset, grouped_action_collate

    device = torch.device("cuda:0")
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = AlignmentModel(
        model_path=str(model_path), seq_len=SEQ_LEN, image_size=224,
        periodicity=SEQ_LEN, input_dim=8, hidden_dim=256, prefix_tokens=8,
        prefix_mode="prefix", max_length=512, init_temperature=0.05,
        torch_dtype=torch.bfloat16, device=device,
    )
    random_ts = {key: value.detach().cpu().clone() for key, value in model.ts_projector.state_dict().items()}
    random_vision = {key: value.detach().cpu().clone() for key, value in model.vision_projector.state_dict().items()}
    random_logit = model.logit_scale.detach().cpu().clone()
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ts_projector": random_ts,
        "vision_projector": random_vision,
        "logit_scale": random_logit,
        "seed": seed,
    }, output / "random_initialization.pt")

    seed_dir = root / "splits" / f"seed_{seed}"
    datasets = {}
    loaders = {}
    for split_name, filename in (("train", "train_100.jsonl"), ("validation", "validation.jsonl"), ("test", "test.jsonl")):
        dataset = GroupedTrafficActionDataset(
            path=[str(seed_dir / filename)], seq_len=SEQ_LEN,
            require_expert_action=True, valid_actions=("ETWT", "NTST", "ELWL", "NLSL"),
        )
        if len(dataset) == 0:
            raise RuntimeError(f"Representation split is empty: {split_name}")
        datasets[split_name] = dataset
        loaders[split_name] = DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=2,
            pin_memory=True, collate_fn=grouped_action_collate,
        )

    exp1_full = root / "experiment1" / "data_100pct" / f"seed_{seed}" / "full" / "stage1"
    normal_dir = completed_alignment(exp1_full)
    shuffled_dir = completed_alignment(root / "experiment2" / f"seed_{seed}" / "shuffled_stage1")
    if normal_dir is None or shuffled_dir is None:
        raise RuntimeError("Normal or shuffled Stage I checkpoint is missing for representation evaluation.")

    arms = {
        "normal_stage1": normal_dir / "best.pt",
        "random_initialization": None,
        "shuffled_pair_stage1": shuffled_dir / "best.pt",
    }
    results: dict[str, Any] = {
        "protocol_version": REPRESENTATION_PROTOCOL_VERSION,
        "seed": seed,
        "split_counts": {key: len(dataset) for key, dataset in datasets.items()},
        "arms": {},
        "probe_protocol": {
            "frozen_features": "AlignmentModel.encode_image output; all Stage-I/backbone parameters frozen",
            "head": "single affine layer; validation-loss early stopping; no nonlinear hidden layer",
            "head_initialization_seed": seed,
            "congestion_label": "low/medium/high from training-only max-phase-queue tertiles",
            "queue_target": "max phase_queue", "waiting_target": "max phase_wait in seconds",
        },
    }
    random_state = (random_ts, random_vision, random_logit)
    for arm, checkpoint_path in arms.items():
        if checkpoint_path is None:
            model.ts_projector.load_state_dict(random_state[0])
            model.vision_projector.load_state_dict(random_state[1])
            with torch.no_grad():
                model.logit_scale.copy_(random_state[2].to(model.logit_scale))
        else:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.ts_projector.load_state_dict(checkpoint["ts_projector"])
            model.vision_projector.load_state_dict(checkpoint["vision_projector"])
            if "logit_scale" in checkpoint:
                with torch.no_grad():
                    model.logit_scale.copy_(checkpoint["logit_scale"].to(model.logit_scale))
        extracted = {name: collect_frozen_features(model, loaders[name], device) for name in ("train", "validation", "test")}
        retrieval = multi_positive_recall(
            extracted["test"]["features"], extracted["test"]["text_features"], extracted["test"]["captions"]
        )
        probe, heads = fit_linear_probes(
            extracted["train"], extracted["validation"], extracted["test"], seed, device,
        )
        results["arms"][arm] = {
            "checkpoint": str(checkpoint_path) if checkpoint_path else "seeded_random_initialization",
            "retrieval_test": retrieval,
            "linear_probe": probe,
        }
        torch.save(heads, output / f"{arm}_probe_heads.pt")
    results["status"] = "success"
    results["completed_at"] = utc_now()
    write_json(output / "metrics.json", results)


def run_representation_queue(repo: Path, root: Path, model_path: Path,
                             gpu_ids: list[int]) -> list[str]:
    pending = deque(SEEDS)
    queue_lock = threading.Lock()
    failures: list[str] = []
    failure_lock = threading.Lock()

    def lane(gpu: int) -> None:
        while True:
            with queue_lock:
                if not pending:
                    return
                seed = pending.popleft()
            try:
                representation_attempt(repo, root, model_path, seed, gpu)
            except Exception as exc:
                with failure_lock:
                    failures.append(f"exp2_s{seed}")
                event(root, "representation", "failed", gpu=gpu,
                      task=f"exp2_s{seed}", detail=str(exc))
                seed_dir = root / "experiment2" / f"seed_{seed}"
                attempts = attempt_directories(seed_dir / "representation")
                if attempts:
                    write_json(attempts[-1] / "error.json", {"error": str(exc), "time": utc_now()})

    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
        futures = [executor.submit(lane, gpu) for gpu in gpu_ids]
        for future in futures:
            future.result()
    return failures


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    output = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                output.append(json.loads(line))
    return output


def format_mean_sd(values: list[float | None], digits: int = 4) -> str:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not clean:
        return "NA"
    avg = sum(clean) / len(clean)
    if len(clean) == 1:
        return f"{avg:.{digits}f}"
    variance = sum((value - avg) ** 2 for value in clean) / (len(clean) - 1)
    return f"{avg:.{digits}f} ± {math.sqrt(variance):.{digits}f}"


def collect_exp1_rows(root: Path) -> tuple[list[dict[str, Any]], dict[tuple[int, int, str], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    histories: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for percentage in PERCENTAGES:
        for seed in SEEDS:
            for arm in ("full", "without_stage1"):
                base = root / "experiment1" / f"data_{percentage:03d}pct" / f"seed_{seed}" / arm
                result_path = base / "task_result.json"
                if not result_path.is_file():
                    rows.append({"percentage": percentage, "seed": seed, "arm": arm, "status": "missing"})
                    continue
                task = json.loads(result_path.read_text(encoding="utf-8"))
                stage2_dir = Path(task["stage2_dir"])
                stage1_dir = Path(task["stage1_dir"]) if task.get("stage1_dir") else None
                metrics = read_jsonl(stage2_dir / "metrics.jsonl")
                histories[(percentage, seed, arm)] = metrics
                valid = [m for m in metrics if m.get("val_top1_cost_regret") is not None]
                if not valid:
                    rows.append({"percentage": percentage, "seed": seed, "arm": arm, "status": "missing_validation_metrics"})
                    continue
                best = min(valid, key=lambda item: float(item["val_top1_cost_regret"]))
                min_regret = float(best["val_top1_cost_regret"])
                threshold = min_regret + max(0.005, 0.05 * abs(min_regret))
                regret_hit = next((item for item in sorted(valid, key=lambda x: int(x["epoch"]))
                                   if float(item["val_top1_cost_regret"]) <= threshold), best)
                max_f1 = max(float(item.get("val_macro_f1", 0.0)) for item in valid)
                f1_hit = next((item for item in sorted(valid, key=lambda x: int(x["epoch"]))
                               if float(item.get("val_macro_f1", 0.0)) >= max_f1 - 0.01), best)
                stage2_summary = json.loads((stage2_dir / "run_summary.json").read_text(encoding="utf-8"))
                test_payload = json.loads((stage2_dir / "test_metrics.json").read_text(encoding="utf-8"))
                test_metrics = test_payload.get("metrics", {})
                stage1_seconds = 0.0
                if stage1_dir:
                    stage1_summary = json.loads((stage1_dir / "run_summary.json").read_text(encoding="utf-8"))
                    stage1_seconds = float(stage1_summary.get("duration_seconds", 0.0))
                stage2_seconds = float(stage2_summary.get("training_seconds", 0.0))
                train_samples = int(stage2_summary.get("train_samples", task.get("train_samples", 0)))
                steps_per_epoch = max(1, math.ceil(train_samples / 8))
                epoch_times = [float(item.get("epoch_elapsed_seconds", 0.0)) for item in sorted(metrics, key=lambda x: int(x["epoch"]))]
                regret_epoch, f1_epoch = int(regret_hit["epoch"]), int(f1_hit["epoch"])
                epoch_time_to_regret = sum(epoch_times[:regret_epoch])
                rows.append({
                    "percentage": percentage, "seed": seed, "arm": arm, "status": "complete",
                    "train_samples": train_samples, "best_epoch_by_val_regret": int(best["epoch"]),
                    "val_regret": min_regret, "val_macro_f1_at_best_regret": float(best.get("val_macro_f1", 0.0)),
                    "test_regret": test_metrics.get("top1_cost_regret"),
                    "test_macro_f1": test_metrics.get("macro_f1"),
                    "regret_convergence_epoch": regret_epoch,
                    "regret_convergence_optimizer_steps": regret_epoch * steps_per_epoch,
                    "regret_convergence_examples_seen": regret_epoch * train_samples,
                    "f1_convergence_epoch": f1_epoch,
                    "stage1_seconds": stage1_seconds, "stage2_seconds": stage2_seconds,
                    "total_training_seconds": stage1_seconds + stage2_seconds,
                    "training_seconds_to_regret_convergence": stage1_seconds + epoch_time_to_regret,
                    "stage2_dir": str(stage2_dir), "stage1_dir": str(stage1_dir) if stage1_dir else "",
                })
    return rows, histories


def collect_exp2_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for seed in SEEDS:
        base = root / "experiment2" / f"seed_{seed}" / "representation"
        metrics_files = [path / "metrics.json" for path in attempt_directories(base) if (path / "metrics.json").is_file()]
        if not metrics_files:
            rows.append({"seed": seed, "status": "missing"})
            continue
        payload = json.loads(metrics_files[-1].read_text(encoding="utf-8"))
        for arm, arm_result in payload.get("arms", {}).items():
            row = {"seed": seed, "arm": arm, "status": payload.get("status", "unknown")}
            row.update({f"retrieval_{key}": value for key, value in arm_result.get("retrieval_test", {}).items()})
            for scope, scope_result in arm_result.get("linear_probe", {}).items():
                if isinstance(scope_result, dict):
                    row.update({f"{scope}_{key}": value for key, value in scope_result.items()})
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(root: Path, exp1: list[dict[str, Any]], exp2: list[dict[str, Any]],
                 histories: dict[tuple[int, int, str], list[dict[str, Any]]]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_root = root / "figures"
    plot_root.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "legend.fontsize": 8, "figure.dpi": 150,
        "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
    })
    paths = []
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.8), sharex=True)
    colors = {"full": "#176B87", "without_stage1": "#D2691E"}
    for ax, percentage in zip(axes.flat, PERCENTAGES):
        for arm in ("full", "without_stage1"):
            curves = []
            for seed in SEEDS:
                entries = histories.get((percentage, seed, arm), [])
                entries = sorted([x for x in entries if x.get("val_top1_cost_regret") is not None], key=lambda x: int(x["epoch"]))
                if entries:
                    curves.append([float(item["val_top1_cost_regret"]) for item in entries])
            if not curves:
                continue
            length = min(len(curve) for curve in curves)
            data = np.asarray([curve[:length] for curve in curves], dtype=float)
            epochs = np.arange(1, length + 1)
            mean = data.mean(axis=0)
            std = data.std(axis=0, ddof=1) if len(curves) > 1 else np.zeros_like(mean)
            ax.plot(epochs, mean, color=colors[arm], label=arm.replace("_", " "), linewidth=1.6)
            ax.fill_between(epochs, mean - std, mean + std, color=colors[arm], alpha=0.16, linewidth=0)
        ax.set_title(f"{percentage}% training data")
        ax.set_xlabel("Stage II epoch")
        ax.set_ylabel("Validation top-1 cost regret")
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = plot_root / f"experiment1_convergence.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)

    complete = [row for row in exp1 if row.get("status") == "complete"]
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
    for ax, metric, label, color_sign in (
        (axes[0], "test_regret", "Held-out top-1 cost regret", True),
        (axes[1], "test_macro_f1", "Held-out Macro-F1", False),
    ):
        for arm in ("full", "without_stage1"):
            xs, means, stds = [], [], []
            for percentage in PERCENTAGES:
                group = [row for row in complete if row["percentage"] == percentage and row["arm"] == arm and row.get(metric) is not None]
                if not group:
                    continue
                xs.append(float(np.mean([row["train_samples"] for row in group])))
                values = np.asarray([float(row[metric]) for row in group], dtype=float)
                means.append(float(values.mean()))
                stds.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)
            if xs:
                ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3,
                            color=colors[arm], label=arm.replace("_", " "), linewidth=1.5)
        ax.set_xlabel("Training decision samples")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.22, linewidth=0.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = plot_root / f"experiment1_data_efficiency.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)

    if exp2:
        metrics = ("retrieval_R@1", "retrieval_R@5", "retrieval_R@10")
        arms = ("normal_stage1", "random_initialization", "shuffled_pair_stage1")
        fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.0))
        styles = {"normal_stage1": "#176B87", "random_initialization": "#7A8B45", "shuffled_pair_stage1": "#D2691E"}
        for arm in arms:
            group = [row for row in exp2 if row.get("arm") == arm and row.get("status") == "success"]
            if group:
                mean = [float(np.mean([float(row.get(metric, 0.0)) for row in group])) for metric in metrics]
                std = [float(np.std([float(row.get(metric, 0.0)) for row in group], ddof=1)) if len(group) > 1 else 0.0 for metric in metrics]
                axes[0, 0].errorbar([1, 5, 10], mean, yerr=std, marker="o", capsize=3, label=arm.replace("_", " "), color=styles[arm])
        axes[0, 0].set_xticks([1, 5, 10])
        axes[0, 0].set_xlabel("Retrieval k")
        axes[0, 0].set_ylabel("Recall")
        axes[0, 0].set_title("Bidirectional image-text retrieval")
        axes[0, 0].grid(True, alpha=0.22, linewidth=0.5)
        axes[0, 0].legend(frameon=False)
        probe_panels = (
            (axes[0, 1], "test_congestion_macro_f1", "Congestion Macro-F1"),
            (axes[1, 0], "test_queue_mae", "Queue length MAE"),
            (axes[1, 1], "test_waiting_mae", "Waiting degree MAE"),
        )
        for ax, metric, label in probe_panels:
            means, stds = [], []
            for arm in arms:
                values = [float(row[metric]) for row in exp2
                          if row.get("arm") == arm and row.get("status") == "success" and row.get(metric) is not None]
                means.append(float(np.mean(values)) if values else float("nan"))
                stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            ax.errorbar([0, 1, 2], means, yerr=stds, fmt="o", capsize=3, color="#343A40", linewidth=1.2)
            for x, arm, value in zip([0, 1, 2], arms, means):
                if math.isfinite(value):
                    ax.scatter([x], [value], color=styles[arm], zorder=3, s=28)
            ax.set_xticks([0, 1, 2], ["Normal", "Random", "Shuffled"])
            ax.set_ylabel(label + (" (higher is better)" if "f1" in metric else " (lower is better)"))
            ax.set_title(label)
            ax.grid(True, alpha=0.22, linewidth=0.5)
        fig.tight_layout()
        for ext in ("pdf", "png"):
            path = plot_root / f"experiment2_representation.{ext}"
            fig.savefig(path, bbox_inches="tight")
            paths.append(str(path))
        plt.close(fig)
    return paths


def summarize(root: Path) -> None:
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    exp1, histories = collect_exp1_rows(root)
    exp2 = collect_exp2_rows(root)
    write_csv(reports / "experiment1_per_run.csv", exp1)
    write_csv(reports / "experiment2_per_run.csv", exp2)
    complete_exp1 = [row for row in exp1 if row.get("status") == "complete"]
    aggregate_exp1 = []
    for percentage in PERCENTAGES:
        for arm in ("full", "without_stage1"):
            group = [row for row in complete_exp1 if row["percentage"] == percentage and row["arm"] == arm]
            aggregate_exp1.append({
                "percentage": percentage, "arm": arm, "n": len(group),
                "train_samples_mean": format_mean_sd([row.get("train_samples") for row in group], 1),
                "val_regret_mean_sd": format_mean_sd([row.get("val_regret") for row in group]),
                "test_regret_mean_sd": format_mean_sd([row.get("test_regret") for row in group]),
                "test_macro_f1_mean_sd": format_mean_sd([row.get("test_macro_f1") for row in group]),
                "regret_convergence_steps_mean_sd": format_mean_sd([row.get("regret_convergence_optimizer_steps") for row in group], 1),
                "stage1_seconds_mean_sd": format_mean_sd([row.get("stage1_seconds") for row in group], 1),
                "stage2_seconds_mean_sd": format_mean_sd([row.get("stage2_seconds") for row in group], 1),
                "total_seconds_mean_sd": format_mean_sd([row.get("total_training_seconds") for row in group], 1),
                "samples_seen_to_convergence_mean_sd": format_mean_sd([row.get("regret_convergence_examples_seen") for row in group], 1),
            })
    write_csv(reports / "experiment1_summary.csv", aggregate_exp1)

    complete_exp2 = [row for row in exp2 if row.get("status") == "success"]
    exp2_metric_names = (
        "retrieval_R@1", "retrieval_R@5", "retrieval_R@10",
        "test_congestion_macro_f1", "test_queue_mae", "test_queue_r2",
        "test_waiting_mae", "test_waiting_r2",
    )
    aggregate_exp2 = []
    for arm in ("normal_stage1", "random_initialization", "shuffled_pair_stage1"):
        group = [row for row in complete_exp2 if row.get("arm") == arm]
        payload = {"arm": arm, "n": len(group)}
        payload.update({metric: format_mean_sd([row.get(metric) for row in group]) for metric in exp2_metric_names})
        aggregate_exp2.append(payload)
    write_csv(reports / "experiment2_summary.csv", aggregate_exp2)
    figure_paths = plot_results(root, exp1, exp2, histories)

    def grouped(percentage: int, arm: str) -> list[dict[str, Any]]:
        return [row for row in complete_exp1 if row["percentage"] == percentage and row["arm"] == arm]

    main_full = grouped(100, "full")
    main_none = grouped(100, "without_stage1")
    full_r = [float(row["test_regret"]) for row in main_full if row.get("test_regret") is not None]
    none_r = [float(row["test_regret"]) for row in main_none if row.get("test_regret") is not None]
    deltas = [f["test_regret"] - n["test_regret"] for f in main_full for n in main_none
              if f.get("seed") == n.get("seed") and f.get("test_regret") is not None and n.get("test_regret") is not None]
    full_f1 = [float(row["test_macro_f1"]) for row in main_full if row.get("test_macro_f1") is not None]
    none_f1 = [float(row["test_macro_f1"]) for row in main_none if row.get("test_macro_f1") is not None]
    exp1_complete = len(complete_exp1)
    exp2_complete_seeds = len({row["seed"] for row in complete_exp2})

    if len(deltas) >= 2:
        dmean = sum(deltas) / len(deltas)
        dstd = math.sqrt(sum((item - dmean) ** 2 for item in deltas) / (len(deltas) - 1))
        delta_text = f"{dmean:.4f} ± {dstd:.4f} (paired seed differences, Full − w/o Stage I; negative favors Full)"
    elif deltas:
        delta_text = f"{deltas[0]:.4f} (one paired seed; descriptive only)"
    else:
        delta_text = "not available"
    by_condition = {(int(row["percentage"]), int(row["seed"]), str(row["arm"])): row
                    for row in complete_exp1}
    data_efficiency_seeds: dict[int, int] = {}
    control_efficiency_seeds: dict[int, int] = {}
    incremental_efficiency_seeds: dict[int, int] = {}

    def meets_high_data_target(candidate: dict[str, Any] | None,
                               baseline: dict[str, Any]) -> bool:
        return bool(
            candidate and candidate.get("test_regret") is not None
            and candidate.get("test_macro_f1") is not None
            and float(candidate["test_regret"]) <= float(baseline["test_regret"]) + 0.01
            and float(candidate["test_macro_f1"]) >= float(baseline["test_macro_f1"]) - 0.02
        )

    baseline_seeds = 0
    for seed in SEEDS:
        baseline = by_condition.get((100, seed, "without_stage1"))
        if not baseline or baseline.get("test_regret") is None or baseline.get("test_macro_f1") is None:
            continue
        baseline_seeds += 1
        for arm, matches in (("full", data_efficiency_seeds),
                             ("without_stage1", control_efficiency_seeds)):
            for percentage in (10, 25, 50):
                if meets_high_data_target(by_condition.get((percentage, seed, arm)), baseline):
                    matches[seed] = percentage
                    break
        full_fraction = data_efficiency_seeds.get(seed)
        control_fraction = control_efficiency_seeds.get(seed)
        if full_fraction is not None and (control_fraction is None or full_fraction < control_fraction):
            incremental_efficiency_seeds[seed] = full_fraction

    quality_time_pairs = []
    for percentage in PERCENTAGES:
        for seed in SEEDS:
            full = by_condition.get((percentage, seed, "full"))
            no_stage1 = by_condition.get((percentage, seed, "without_stage1"))
            if not full or not no_stage1:
                continue
            comparable = (
                full.get("test_regret") is not None and no_stage1.get("test_regret") is not None
                and full.get("test_macro_f1") is not None and no_stage1.get("test_macro_f1") is not None
                and float(full["test_regret"]) <= float(no_stage1["test_regret"]) + 0.01
                and float(full["test_macro_f1"]) >= float(no_stage1["test_macro_f1"]) - 0.02
            )
            quality_time_pairs.append(bool(
                comparable and float(full["total_training_seconds"]) < float(no_stage1["total_training_seconds"])
            ))
    required_data_seeds = math.ceil(2 * baseline_seeds / 3) if baseline_seeds else 0
    data_efficiency_supported = baseline_seeds >= 3 and len(data_efficiency_seeds) >= required_data_seeds
    incremental_efficiency_supported = (
        baseline_seeds >= 3 and len(incremental_efficiency_seeds) >= required_data_seeds
    )
    required_time_pairs = math.ceil(2 * len(quality_time_pairs) / 3) if quality_time_pairs else 0
    wall_time_supported = bool(quality_time_pairs) and sum(quality_time_pairs) >= required_time_pairs
    if baseline_seeds >= 3:
        efficient_seeds = ", ".join(f"{seed}: {percentage}%" for seed, percentage in sorted(data_efficiency_seeds.items())) or "none"
        control_seeds = ", ".join(f"{seed}: {percentage}%" for seed, percentage in sorted(control_efficiency_seeds.items())) or "none"
        low_data_verdict = "met" if data_efficiency_supported else "not met"
        incremental_verdict = "supported" if incremental_efficiency_supported else "not supported"
        time_verdict = "supported" if wall_time_supported else "not supported"
        exp1_statement = (
            f"**Predeclared low-data target: {low_data_verdict}.** Full Stage I reaches the w/o-Stage-I 100% held-out result "
            f"within 0.01 regret and 0.02 Macro-F1 using <=50% data for {len(data_efficiency_seeds)}/{baseline_seeds} seeds ({efficient_seeds}). "
            f"**Stage-I-specific data-efficiency: {incremental_verdict}.** The low-data w/o-Stage-I arm also meets that target "
            f"for {len(control_efficiency_seeds)}/{baseline_seeds} seeds ({control_seeds}); Full needs a strictly smaller data fraction "
            f"in {len(incremental_efficiency_seeds)}/{baseline_seeds} seeds. This additional attribution check requires two-thirds. "
            f"**Wall-clock verdict: {time_verdict}.** Full is both within those test tolerances and faster, after counting Stage I, "
            f"in {sum(quality_time_pairs)}/{len(quality_time_pairs)} paired same-data comparisons; the decision rule requires at least two-thirds."
        )
    else:
        exp1_statement = "Insufficient completed paired runs for the predeclared three-seed efficiency verdict."

    metric_messages = []
    exp2_metric_support: dict[str, bool] = {}
    if exp2_complete_seeds >= 3:
        arm_rows = {(int(row["seed"]), str(row["arm"])): row for row in complete_exp2}
        for metric, lower_better in (("retrieval_R@1", False), ("test_congestion_macro_f1", False),
                                    ("test_queue_mae", True), ("test_waiting_mae", True)):
            wins = {control: 0 for control in ("random_initialization", "shuffled_pair_stage1")}
            paired_counts = {control: 0 for control in wins}
            for seed in SEEDS:
                normal = arm_rows.get((seed, "normal_stage1"), {}).get(metric)
                if normal is None:
                    continue
                for control in wins:
                    comparison = arm_rows.get((seed, control), {}).get(metric)
                    if comparison is None:
                        continue
                    paired_counts[control] += 1
                    better = float(normal) < float(comparison) if lower_better else float(normal) > float(comparison)
                    if better:
                        wins[control] += 1
            supported = all(paired_counts[c] >= 3 and wins[c] >= 2 for c in wins)
            exp2_metric_support[metric] = supported
            metric_messages.append(
                f"- `{metric}`: Normal Stage I beats both controls in at least 2/3 paired seeds: "
                f"{'; '.join(f'{control} {wins[control]}/{paired_counts[control]}' for control in wins)}; "
                f"metric verdict **{'supported' if supported else 'not supported'}**."
            )
    retrieval_supported = exp2_metric_support.get("retrieval_R@1", False)
    probe_support_count = sum(exp2_metric_support.get(name, False) for name in (
        "test_congestion_macro_f1", "test_queue_mae", "test_waiting_mae"
    ))
    exp2_supported = exp2_complete_seeds >= 3 and retrieval_supported and probe_support_count >= 2
    if metric_messages:
        exp2_statement = "\n".join(metric_messages) + "\n\n**Overall visual-representation verdict: " + ("supported" if exp2_supported else "not supported under the stated criterion") + ".** The criterion requires R@1 and at least two of the three probe metrics to beat both controls in at least two paired seeds."
    else:
        exp2_statement = "Three-seed representation metrics are incomplete; no overall conclusion is asserted."

    lines = [
        "# Stage I Effectiveness Validation", "",
        f"Run status: `{exp1_complete}/24` Experiment 1 conditions and `{exp2_complete_seeds}/3` Experiment 2 seeds have complete results.",
        "", "## Protocol", "",
        "Jinan-1 labeled decision samples are split chronologically into 70/15/15 train/validation/test blocks. A 30-second embargo separates blocks, and labels in each block's initial 30-second sequence warmup are excluded. Training subsets are class-stratified, nested, and sampled at 10%, 25%, 50%, and 100%; each seed uses the same validation and test blocks. The test split is not used for checkpoint selection.",
        "", "Experiment 1 compares Stage I + Stage II against Stage II initialized randomly. Stage wall time includes data/model setup and all fitting/validation epochs but excludes final test inference; Full totals include Stage I. Validation top-1 cost regret selects the Stage II checkpoint; Macro-F1 is reported from that same checkpoint. Regret convergence is the first epoch within max(0.005, 5% of the run's best validation regret); Macro-F1 convergence is first within 0.01 of its run maximum. Convergence optimizer steps use the configured effective Stage II accumulation of eight examples per step. Predeclared tolerances are 0.01 held-out regret and 0.02 Macro-F1; the original low-data target compares Full at <=50% with w/o Stage I at 100%. An additional attribution check compares the minimum data fraction required by both arms. Wall-clock efficiency requires comparable test quality and lower total time in at least two-thirds of paired same-data comparisons.",
        "", "Experiment 2 uses the same splits and 100% training samples. Retrieval uses test-only paired image/text embeddings and treats identical captions as multiple valid positives. Linear probes freeze the Stage-I image representation and fit one affine head; all representation arms use the same probe-head initialization seed within each experiment seed. The congestion thresholds are training-only max-phase-queue tertiles, queue length is max phase queue, and waiting degree is max phase wait in seconds. Validation loss selects linear-head epoch; all reported final probe metrics use the held-out test split.",
        "", "## Experiment 1", "",
        "| Train data | Arm | n | Val regret | Test regret | Test Macro-F1 | Regret convergence steps | Stage I seconds | Stage II seconds | Total seconds | Samples seen to convergence |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_exp1:
        lines.append(
            f"| {row['percentage']}% | {row['arm']} | {row['n']} | {row['val_regret_mean_sd']} | {row['test_regret_mean_sd']} | {row['test_macro_f1_mean_sd']} | {row['regret_convergence_steps_mean_sd']} | {row['stage1_seconds_mean_sd']} | {row['stage2_seconds_mean_sd']} | {row['total_seconds_mean_sd']} | {row['samples_seen_to_convergence_mean_sd']} |"
        )
    lines += ["", f"At 100% data, paired held-out regret difference is {delta_text}.", exp1_statement,
              "", "The low-data target and wall-clock decision use the original thresholds; the same-data attribution check is a retrospective diagnostic, not a predeclared test. Validation metrics drive checkpoint selection and convergence summaries; held-out test metrics are only used for final comparisons.",
              "", "### Experiment 1 Figures", ""]
    lines += [f"- `{path}`" for path in figure_paths if "experiment1" in path]
    lines += ["", "## Experiment 2", "",
              "| Representation | n seeds | R@1 | R@5 | R@10 | Congestion Macro-F1 | Queue MAE | Queue R2 | Waiting MAE | Waiting R2 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate_exp2:
        lines.append(
            f"| {row['arm']} | {row['n']} | {row['retrieval_R@1']} | {row['retrieval_R@5']} | {row['retrieval_R@10']} | {row['test_congestion_macro_f1']} | {row['test_queue_mae']} | {row['test_queue_r2']} | {row['test_waiting_mae']} | {row['test_waiting_r2']} |"
        )
    lines += ["", exp2_statement, "", "The overall representation verdict requires Normal Stage I to beat both random initialization and shuffled pairs in R@1 and at least two of congestion Macro-F1, queue MAE, and waiting MAE, each in at least two paired seeds. Metric-level outcomes remain visible above and in the table.",
              "", "### Experiment 2 Figure", ""]
    lines += [f"- `{path}`" for path in figure_paths if "experiment2" in path]
    lines += ["", "## Interpretation Limits", "",
              "This is a three-seed, single-flow, temporally blocked Jinan-1 comparison, not a cross-flow generalization result. The data-efficiency and representation verdicts are descriptive decision rules, not significance tests.",
              "Convergence epochs are defined relative to each run's own best validation regret; fewer steps need not mean reaching the same absolute regret. Relative retrieval and probe wins do not imply strong absolute prediction: inspect R@1 and R2 alongside the paired comparisons.",
              "", "## Artifacts", "",
              "- `manifest.json`: experiment configuration, input/code hashes, GPU inventory, and protocol.",
              "- `splits/manifest.json`: timestamp ranges, embargo, counts, and split hashes.",
              "- `status/events.jsonl`: task/stage transitions and failures.",
              "- `experiment1_per_run.csv` and `experiment1_summary.csv`: per-seed and aggregated efficiency metrics.",
              "- `experiment2_per_run.csv` and `experiment2_summary.csv`: per-seed retrieval and probe metrics.",
              "- `figures/`: 300-DPI PNG and vector PDF plots.",
              "- Each completed Stage I/II attempt contains its command, stdout log, model checkpoints, and metrics.",
              "", "Incomplete failed attempts are retained. Rerunning `recordlocal/run_stage1_validation.sh` skips successful phases and resumes an interrupted phase from its latest epoch-level `last.pt` checkpoint.", ""]
    (reports / "ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "REPORT.md").write_text((reports / "ANALYSIS.md").read_text(encoding="utf-8"), encoding="utf-8")

    events = read_jsonl(root / "status" / "events.jsonl")
    status_path = root / "status" / "status.tsv"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    with status_path.open("w", encoding="utf-8") as handle:
        handle.write("time\tstage\tstatus\tgpu\ttask\tdetail\tpath\n")
        for row in events:
            values = [str(row.get(key, "") if row.get(key) is not None else "") for key in ("time", "stage", "status", "gpu", "task", "detail", "path")]
            handle.write("\t".join(value.replace("\t", " ").replace("\n", " ") for value in values) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage I effectiveness validation experiments")
    parser.add_argument("--mode", choices=("run", "prepare-only", "preflight-only", "representation-worker", "summarize"), default="run")
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=MODEL_DEFAULT)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = (args.root or (repo / "recordlocal" / "runs" / "stage1_validation")).resolve()
    if args.mode == "prepare-only":
        source = repo / DATA_RELATIVE
        if not source.is_file():
            raise FileNotFoundError(source)
        split_manifest = write_data_splits(repo, root, source)
        print(json.dumps({
            "source_rows": split_manifest["source_rows"],
            "timestamp_counts": split_manifest["timestamp_counts"],
            "decision_samples": split_manifest["decision_samples_after_warmup"],
            "seed_train_counts": {
                seed: {percentage: value["selected_decision_samples"]
                       for percentage, value in detail["train"].items()}
                for seed, detail in split_manifest["seeds"].items()
            },
            "split_manifest": str(root / "splits" / "manifest.json"),
        }, indent=2))
        return 0
    if args.mode == "preflight-only":
        return run_experiments(repo, root, args.model.resolve(), preflight_only=True)
    if args.mode == "representation-worker":
        if args.seed is None or args.gpu is None or args.output is None:
            parser.error("representation-worker requires --seed, --gpu, and --output")
        representation_worker(repo, root, args.model.resolve(), args.seed, args.output.resolve(), args.gpu)
        return 0
    if args.mode == "summarize":
        summarize(root)
        return 0
    return run_experiments(repo, root, args.model.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
