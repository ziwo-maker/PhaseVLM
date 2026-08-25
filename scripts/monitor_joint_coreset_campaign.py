#!/usr/bin/env python3
"""Detached 40-minute monitor for the Stage-1/Stage-2 coreset campaign.

The monitor only reads campaign artifacts.  It emits bounded snapshots to its
own log and sends best-effort Feishu updates without logging webhook secrets.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path("/home/code/LLMSTSC")
FEISHU_CONFIG = Path("/root/.claude/feishu.json")
ERROR_PATTERN = "Traceback|RuntimeError|ERROR|OOM|out of memory|FAILED"


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def run_snapshot(command: list[str], timeout_seconds: int = 12) -> str:
    try:
        result = subprocess.run(
            command,
            cwd=REPO,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
        return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def feishu_send(log_path: Path, text: str) -> None:
    try:
        config = read_json(FEISHU_CONFIG)
        if not isinstance(config, dict) or config.get("mode") != "push":
            append_log(log_path, "feishu=skipped")
            return
        webhook = config.get("webhook_url")
        if not isinstance(webhook, str) or not webhook:
            append_log(log_path, "feishu=skipped_missing_webhook")
            return
        payload = json.dumps(
            {"msg_type": "text", "content": {"text": text}}, ensure_ascii=False
        ).encode("utf-8")
        request = urllib.request.Request(
            webhook,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            append_log(log_path, f"feishu=sent_http_{response.status}")
    except Exception as exc:  # Best effort; never interrupt the monitor.
        append_log(log_path, f"feishu=failed_{type(exc).__name__}")


def manifest_counts(root: Path) -> Counter[str]:
    manifest = read_json(root / "tasks.json")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("tasks"), list):
        return Counter()
    return Counter(str(item.get("state", "unknown")) for item in manifest["tasks"] if isinstance(item, dict))


def latest_online_metrics(root: Path) -> str:
    summaries = sorted(root.rglob("eval_summary.json"), key=lambda item: item.stat().st_mtime)
    if not summaries:
        return "online_metrics=none"
    data = read_json(summaries[-1])
    if not isinstance(data, dict):
        return f"online_metrics=unreadable:{summaries[-1].parent.name}"
    values = data.get("mean") if isinstance(data.get("mean"), dict) else data
    if not isinstance(values, dict):
        return f"online_metrics=unreadable:{summaries[-1].parent.name}"
    metric_fields = []
    for key in ("ATT", "AQL", "AWT"):
        value = values.get(key)
        if isinstance(value, (int, float)):
            metric_fields.append(f"{key}={value:.3f}")
    return (
        f"online_metrics={','.join(metric_fields) if metric_fields else 'unavailable'}"
        f" latest={summaries[-1].parent.parent.name}"
    )


def error_lines(root: Path) -> list[str]:
    output = run_snapshot(
        [
            "bash",
            "-lc",
            f"rg -n '{ERROR_PATTERN}' '{root}' --glob '*.log' | tail -n 20",
        ],
    )
    return [line for line in output.splitlines() if line]


def current_summary(root: Path) -> tuple[str, dict[str, Any]]:
    campaign = read_json(root / "campaign_status.json")
    state = campaign.get("state", "unknown") if isinstance(campaign, dict) else "missing"
    counts = manifest_counts(root)
    gpu = run_snapshot(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    processes = run_snapshot(
        [
            "bash",
            "-lc",
            "ps -eo pid,ppid,stat,etime,cmd | "
            "rg 'launch_joint_coreset|run_joint_coreset_worker|run_joint_coreset_job|relabel_simexpert' "
            "| head -n 60",
        ]
    )
    errors = error_lines(root)
    lines = [
        "Joint SimExpert Stage1/2 coreset campaign",
        f"state={state}",
        "tasks=" + (", ".join(f"{name}={counts[name]}" for name in sorted(counts)) or "not_created"),
        latest_online_metrics(root),
        "gpus=" + (gpu.replace("\n", "; ") if gpu else "unavailable"),
        f"process_lines={len(processes.splitlines())}",
        f"errors={len(errors)}",
    ]
    if errors:
        lines.append("latest_error=" + errors[-1][:400])
    return "\n".join(lines), {"state": state, "counts": dict(counts), "errors": errors}


def done(snapshot: dict[str, Any]) -> bool:
    return snapshot["state"] in {"complete", "complete_with_failures"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--interval-seconds", type=int, default=2400)
    parser.add_argument("--max-hours", type=float, default=336.0)
    parser.add_argument("--notify-start", action="store_true")
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    log_path = root / "joint_coreset_monitor.log"
    state_path = root / "joint_coreset_monitor_state.json"
    deadline = time.time() + args.max_hours * 3600
    if args.notify_start:
        feishu_send(
            log_path,
            "Joint SimExpert Stage1/2 coreset campaign monitor started. "
            f"Polling every {args.interval_seconds // 60} minutes. Results: {root}",
        )

    iteration = 0
    while True:
        iteration += 1
        summary, snapshot = current_summary(root)
        append_log(log_path, summary.replace("\n", " | "))
        state_path.write_text(
            json.dumps({"time": now(), "iteration": iteration, **snapshot}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if done(snapshot):
            label = "completed with failures" if snapshot["state"] == "complete_with_failures" else "completed"
            feishu_send(log_path, f"Joint SimExpert Stage1/2 coreset campaign {label}.\n{summary}")
            append_log(log_path, "monitor=complete")
            return
        if snapshot["errors"]:
            feishu_send(log_path, "Joint SimExpert Stage1/2 coreset campaign alert.\n" + summary)
        else:
            feishu_send(log_path, "Joint SimExpert Stage1/2 coreset campaign progress.\n" + summary)
        if time.time() >= deadline:
            feishu_send(log_path, "Joint SimExpert Stage1/2 coreset monitor reached max runtime.\n" + summary)
            append_log(log_path, "monitor=max_runtime")
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
