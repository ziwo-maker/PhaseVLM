#!/usr/bin/env python3
"""Claim and run isolated coreset tasks from a lock-protected JSON manifest."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PYTHON = "/home/sysu/anaconda3/envs/hnr5090/bin/python"


def claim(manifest_path: Path, gpu: str) -> dict[str, Any] | None:
    with manifest_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        manifest = json.load(handle)
        task = next((item for item in manifest["tasks"] if item["state"] == "pending"), None)
        if task is not None:
            task["state"] = "running"
            task["gpu"] = gpu
            task["worker_pid"] = os.getpid()
            task["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            handle.seek(0)
            json.dump(manifest, handle, indent=2)
            handle.truncate()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return task


def finish(manifest_path: Path, name: str, state: str, detail: str) -> None:
    with manifest_path.open("r+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        manifest = json.load(handle)
        task = next(item for item in manifest["tasks"] if item["name"] == name)
        task["state"] = state
        task["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        task["detail"] = detail[-500:]
        handle.seek(0)
        json.dump(manifest, handle, indent=2)
        handle.truncate()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--gpu", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    while True:
        task = claim(manifest_path, args.gpu)
        if task is None:
            return
        output_dir = str(Path(task["output_dir"]).resolve())
        command = [
            PYTHON, "scripts/run_joint_coreset_job.py", "--task", task["name"],
            "--stage1_data", task["stage1_data"], "--stage2_data", task["stage2_data"],
            "--flow", task["flow"], "--gpu", str(args.gpu), "--seed", str(task["seed"]),
            "--output_dir", output_dir, "--epochs", str(task.get("epochs", 15)), "--eval_seeds",
            *[str(seed) for seed in task.get("eval_seeds", [2024, 42, 123])],
        ]
        completed = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        finish(manifest_path, task["name"], "complete" if completed.returncode == 0 else "failed", completed.stderr)


if __name__ == "__main__":
    main()
