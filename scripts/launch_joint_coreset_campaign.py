#!/usr/bin/env python3
"""Launch the complete same-flow Stage-1/2 small-data experiment campaign.

The campaign starts three static pilots immediately while CPU workers build
H120/H300 labels.  It then builds strict reliable coresets and drains the full
matrix through one independent worker queue per GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/sysu/anaconda3/envs/hnr5090/bin/python"
FLOWS: Dict[str, Dict[str, str]] = {
    "jinan_real": {
        "data": "data/processed/jinan_real_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
        "flow": "anon_3_4_jinan_real.json",
    },
    "jinan_real_2000": {
        "data": "data/processed/jinan_real_2000_stage3_queue_wait_rule_gap2_p60_20260529.jsonl",
        "flow": "anon_3_4_jinan_real_2000.json",
    },
    "jinan_real_2500": {
        "data": "data/processed/jinan_real_2500_stage3_queue_wait_rule_gap3_p60_20260529.jsonl",
        "flow": "anon_3_4_jinan_real_2500.json",
    },
}
PRIMARY_SEEDS = (2024, 42, 123)
CONFIRMATION_SEEDS = (7, 11, 99, 314, 2718, 9001)


def write_status(root: Path, **updates: Any) -> None:
    path = root / "campaign_status.json"
    payload: Dict[str, Any] = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
    payload.update(updates)
    payload["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(command: List[str], log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n$ " + " ".join(command) + "\n")
        handle.flush()
        completed = subprocess.run(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"failed rc={completed.returncode}: {command[0]}")


def start(command: List[str], log_path: Path) -> subprocess.Popen[str]:
    handle = log_path.open("a", encoding="utf-8")
    handle.write("\n$ " + " ".join(command) + "\n")
    handle.flush()
    process = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, text=True)
    process._joint_log_handle = handle  # type: ignore[attr-defined]
    return process


def wait_all(processes: Iterable[subprocess.Popen[str]], purpose: str) -> None:
    failed = []
    for process in processes:
        code = process.wait()
        handle = getattr(process, "_joint_log_handle", None)
        if handle is not None:
            handle.close()
        if code != 0:
            failed.append(f"pid={process.pid} rc={code}")
    if failed:
        raise RuntimeError(f"{purpose} failed: {', '.join(failed)}")


def build_dataset(
    root: Path,
    flow_name: str,
    variant: str,
    mode: str,
    ratio: float,
    strict: bool,
) -> None:
    spec = FLOWS[flow_name]
    output_dir = root / "datasets" / ("strict" if strict else "static") / flow_name / variant
    command = [
        PYTHON, "scripts/build_joint_simexpert_coreset.py", "--input", spec["data"],
        "--output_dir", str(output_dir), "--mode", mode, "--ratio", str(ratio), "--seed", "2024",
    ]
    if strict:
        labels = root / "labels" / flow_name
        command.extend([
            "--h120", str(labels / "h120.jsonl"), "--h300", str(labels / "h300.jsonl"),
            "--require_horizon", "--min_margin", "0.05",
        ])
    run(command, root / "logs" / f"build_{'strict' if strict else 'static'}_{flow_name}_{variant}.log")


def run_horizon_chain(root: Path, flow_name: str) -> None:
    spec = FLOWS[flow_name]
    label_dir = root / "labels" / flow_name
    label_dir.mkdir(parents=True, exist_ok=True)
    for horizon in (120, 300):
        output = label_dir / f"h{horizon}.jsonl"
        if output.is_file():
            continue
        run(
            [
                PYTHON, "scripts/relabel_simexpert.py", "--input", spec["data"], "--output", str(output),
                "--roadnet", "data/Jinan/3_4/roadnet_3_4.json", "--traffic_file",
                f"data/Jinan/3_4/{spec['flow']}", "--dataset", "jinan", "--sim_horizon", str(horizon),
                "--work_dir", str(label_dir / f"work_h{horizon}"),
            ],
            root / "logs" / f"relabel_{flow_name}_h{horizon}.log",
        )


def dataset_dir(root: Path, flow_name: str, variant: str, strict: bool = True) -> Path:
    return root / "datasets" / ("strict" if strict else "static") / flow_name / variant


def add_tasks(
    tasks: List[Dict[str, Any]], root: Path, flow_name: str, name: str,
    stage1_variant: str, stage2_variant: str, seeds: Iterable[int], strict: bool = True,
) -> None:
    flow = FLOWS[flow_name]["flow"]
    stage1_data = dataset_dir(root, flow_name, stage1_variant, strict) / "stage1.jsonl"
    stage2_data = dataset_dir(root, flow_name, stage2_variant, strict) / "stage2.jsonl"
    for seed in seeds:
        task_name = f"{flow_name}_{name}_seed{seed}"
        tasks.append(
            {
                "name": task_name,
                "state": "pending",
                "flow": flow,
                "seed": int(seed),
                "epochs": 15,
                "eval_seeds": list(PRIMARY_SEEDS),
                "stage1_data": str(stage1_data),
                "stage2_data": str(stage2_data),
                "output_dir": str(root / "runs" / task_name),
            }
        )


def build_manifest(root: Path) -> Path:
    tasks: List[Dict[str, Any]] = []
    # Full Jinan-real ablation/budget matrix.
    add_tasks(tasks, root, "jinan_real", "full", "full", "full", PRIMARY_SEEDS)
    add_tasks(tasks, root, "jinan_real", "random35", "random35", "random35", PRIMARY_SEEDS)
    for ratio in ("selected10", "selected20", "selected35", "selected50"):
        add_tasks(tasks, root, "jinan_real", ratio, ratio, ratio, PRIMARY_SEEDS)
    add_tasks(tasks, root, "jinan_real", "stage1only35", "selected35", "full", PRIMARY_SEEDS)
    add_tasks(tasks, root, "jinan_real", "stage2only35", "full", "selected35", PRIMARY_SEEDS)
    # Repeat the plain quality selector to isolate horizon reliability.
    add_tasks(tasks, root, "jinan_real", "static35", "selected35", "selected35", (42, 123), strict=False)
    # Planned n=9 confirmation for the main 35% arm, including the three above.
    add_tasks(tasks, root, "jinan_real", "selected35_confirm", "selected35", "selected35", CONFIRMATION_SEEDS)
    # Same-flow replication of the frozen rule on the other two Jinan flows.
    for flow_name in ("jinan_real_2000", "jinan_real_2500"):
        add_tasks(tasks, root, flow_name, "full", "full", "full", PRIMARY_SEEDS)
        add_tasks(tasks, root, flow_name, "random35", "random35", "random35", PRIMARY_SEEDS)
        add_tasks(tasks, root, flow_name, "selected35", "selected35", "selected35", PRIMARY_SEEDS)

    manifest_path = root / "tasks.json"
    manifest_path.write_text(json.dumps({"tasks": tasks}, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--gpus", default="0,1,2")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    (root / "logs").mkdir(parents=True, exist_ok=True)
    write_status(root, state="starting", gpus=args.gpus, flows=list(FLOWS))

    # Static pilot datasets and the three immediate GPU jobs.
    for variant, mode, ratio in (("full", "full", 1.0), ("random35", "random", 0.35), ("selected35", "selected", 0.35)):
        build_dataset(root, "jinan_real", variant, mode, ratio, strict=False)
    pilot_specs = (("full", "0"), ("random35", "1"), ("selected35", "2"))
    pilots = []
    for variant, gpu in pilot_specs:
        task = f"jinan_real_static_{variant}_seed2024"
        pilots.append(
            start(
                [
                    PYTHON, "scripts/run_joint_coreset_job.py", "--task", task,
                    "--stage1_data", str(dataset_dir(root, "jinan_real", variant, False) / "stage1.jsonl"),
                    "--stage2_data", str(dataset_dir(root, "jinan_real", variant, False) / "stage2.jsonl"),
                    "--flow", FLOWS["jinan_real"]["flow"], "--gpu", gpu, "--seed", "2024",
                    "--output_dir", str(root / "pilots" / task), "--epochs", "15", "--eval_seeds", "2024", "42", "123",
                ],
                root / "logs" / f"pilot_{variant}.log",
            )
        )

    # Counterfactual relabeling is CPU-only and can run in parallel with GPU pilots.
    horizon_processes = []
    for flow_name in FLOWS:
        horizon_processes.append(
            start(
                [PYTHON, "scripts/launch_joint_coreset_campaign.py", "--internal_horizon", flow_name, "--output_root", str(root)],
                root / "logs" / f"horizon_{flow_name}.log",
            )
        )
    write_status(root, state="pilots_and_horizon_running", pilot_pids=[p.pid for p in pilots], horizon_pids=[p.pid for p in horizon_processes])
    wait_all(horizon_processes, "horizon labeling")
    write_status(root, state="building_strict_datasets")

    for flow_name in FLOWS:
        for variant, mode, ratio in (
            ("full", "full", 1.0), ("random35", "random", 0.35),
            ("selected10", "selected", 0.10), ("selected20", "selected", 0.20),
            ("selected35", "selected", 0.35), ("selected50", "selected", 0.50),
        ):
            build_dataset(root, flow_name, variant, mode, ratio, strict=True)

    # No GPU worker starts until the initial three independent pilots have released GPUs.
    wait_all(pilots, "static pilots")
    manifest = build_manifest(root)
    workers = []
    for gpu in [item.strip() for item in args.gpus.split(",") if item.strip()]:
        workers.append(
            start(
                [PYTHON, "scripts/run_joint_coreset_worker.py", "--manifest", str(manifest), "--gpu", gpu],
                root / "logs" / f"worker_gpu{gpu}.log",
            )
        )
    write_status(root, state="matrix_running", manifest=str(manifest), worker_pids=[p.pid for p in workers])
    wait_all(workers, "matrix workers")
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    failed = [task["name"] for task in manifest_data["tasks"] if task["state"] == "failed"]
    write_status(root, state="complete" if not failed else "complete_with_failures", failed_tasks=failed)


if __name__ == "__main__":
    # Hidden worker mode keeps the parent campaign implementation in one file.
    if "--internal_horizon" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--internal_horizon", required=True)
        parser.add_argument("--output_root", required=True)
        parsed = parser.parse_args()
        root_path = Path(parsed.output_root).resolve()
        (root_path / "logs").mkdir(parents=True, exist_ok=True)
        run_horizon_chain(root_path, parsed.internal_horizon)
    else:
        main()
