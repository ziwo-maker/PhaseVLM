#!/usr/bin/env python3
"""Poll Phase-7 policy-responsive supplementary experiments.

The monitor is intentionally lightweight and bounded per poll:
it snapshots process state, GPU usage, result JSONs, and recent error lines,
then updates refine-logs/EXPERIMENT_FINAL_REPORT.md between explicit markers.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


REPO = Path("/home/code/LLMSTSC")
DEFAULT_SUPPLEMENT_DIR = REPO / "records/pipeline_runs/phase7_policy_responsive_supplement_20260814"
DEFAULT_REPORT = REPO / "refine-logs/EXPERIMENT_FINAL_REPORT.md"
FEISHU_CONFIG = Path("/root/.claude/feishu.json")

ARMS = {
    "h30_k32_more_a": {"labels": "responsive H30", "k": 32, "train_seeds": [7, 11, 99], "gpu": 0},
    "h30_k32_more_b": {"labels": "responsive H30", "k": 32, "train_seeds": [314, 2718, 9001], "gpu": 1},
    "h120_k32": {"labels": "responsive H120", "k": 32, "train_seeds": [2024, 42, 123], "gpu": 2},
}
EVAL_SEEDS = [2024, 42, 123]
ERROR_RE = re.compile(
    r"(FAILED|Traceback|RuntimeError|OutOfMemory|out of memory|CUDA out of memory|ERROR|SignalException|Killed)",
    re.IGNORECASE,
)
START = "<!-- PHASE7_SUPPLEMENT_MONITOR_START -->"
END = "<!-- PHASE7_SUPPLEMENT_MONITOR_END -->"


def run_cmd(cmd: list[str], timeout_s: int = 10) -> str:
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout_s)
    except subprocess.CalledProcessError as exc:
        out = exc.output or ""
    except subprocess.TimeoutExpired as exc:
        out = (exc.output or "") + "\n[TIMEOUT]"
    return out.strip()


def safe_load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def result_metrics(path: Path) -> dict[str, float] | None:
    payload = safe_load_json(path)
    if not payload:
        return None
    results = payload.get("results", {})
    try:
        return {
            "ATT": float(results["test_avg_travel_time_over"]),
            "AQL": float(results["test_avg_queue_len_over"]),
            "AWT": float(results["test_avg_waiting_time_over"]),
        }
    except Exception:
        return None


def fmt_float(x: float | None) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    return f"{x:.2f}"


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def read_tail(path: Path, n: int = 240) -> list[str]:
    try:
        lines = path.read_text(errors="replace").splitlines()
    except Exception:
        return []
    return lines[-n:]


def recent_errors(arm_dir: Path, top_n: int = 5) -> list[str]:
    hits: list[str] = []
    for path in sorted(arm_dir.rglob("*.log")):
        for line in read_tail(path):
            if ERROR_RE.search(line):
                hits.append(f"{path.relative_to(arm_dir)}: {line.strip()[:220]}")
    return hits[-top_n:]


def pids_for_arm(ps_output: str, arm_dir: Path) -> list[str]:
    marker = str(arm_dir)
    pids: list[str] = []
    for line in ps_output.splitlines():
        if marker in line and "monitor_phase7_supplement.py" not in line:
            parts = line.strip().split(None, 1)
            if parts and parts[0].isdigit():
                pids.append(parts[0])
    return pids


def collect_arm(arm_name: str, cfg: dict[str, Any], supplement_dir: Path, ps_output: str) -> dict[str, Any]:
    arm_dir = supplement_dir / arm_name
    result_files = sorted((arm_dir / "c30/k32").glob("tr*/evaluation/result_seed*.json"))
    expected_results = len(cfg["train_seeds"]) * len(EVAL_SEEDS)

    per_eval: list[dict[str, float]] = []
    per_train: dict[str, list[dict[str, float]]] = {}
    for path in result_files:
        metrics = result_metrics(path)
        if not metrics:
            continue
        per_eval.append(metrics)
        train_seed = path.parts[-3].removeprefix("tr")
        per_train.setdefault(train_seed, []).append(metrics)

    completed_train_seeds: list[str] = []
    train_means: list[dict[str, float]] = []
    for train_seed, rows in sorted(per_train.items()):
        if len(rows) >= len(EVAL_SEEDS):
            completed_train_seeds.append(train_seed)
            train_means.append({key: mean([row[key] for row in rows]) for key in ("ATT", "AQL", "AWT")})

    aggregate = safe_load_json(arm_dir / "aggregate.json")
    status_json = safe_load_json(arm_dir / "status.json")
    errors = recent_errors(arm_dir)
    pids = pids_for_arm(ps_output, arm_dir)

    metric_summary: dict[str, dict[str, float | None]] = {}
    metric_source = train_means if train_means else per_eval
    for key in ("ATT", "AQL", "AWT"):
        vals = [row[key] for row in metric_source if key in row]
        mu, sd = mean_std(vals)
        metric_summary[key] = {"mean": mu, "std": sd}

    complete = len(result_files) >= expected_results or bool(aggregate)
    if errors:
        status = "warning"
    elif complete:
        status = "done"
    elif pids:
        status = "running"
    else:
        status = "stalled"

    return {
        "name": arm_name,
        "labels": cfg["labels"],
        "k": cfg["k"],
        "gpu": cfg["gpu"],
        "expected_results": expected_results,
        "result_count": len(result_files),
        "completed_train_seeds": completed_train_seeds,
        "running_pids": pids,
        "status": status,
        "metrics": metric_summary,
        "errors": errors,
        "has_aggregate": bool(aggregate),
        "status_json": status_json,
    }


def collect_snapshot(supplement_dir: Path) -> dict[str, Any]:
    ps_output = run_cmd(["ps", "-eo", "pid,ppid,pgid,sid,stat,etime,cmd", "--forest"])
    gpu_output = run_cmd(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ]
    )
    arms = [collect_arm(name, cfg, supplement_dir, ps_output) for name, cfg in ARMS.items()]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "supplement_dir": str(supplement_dir),
        "gpu": gpu_output,
        "arms": arms,
    }


def render_metrics(metrics: dict[str, dict[str, float | None]]) -> str:
    parts = []
    for key in ("ATT", "AQL", "AWT"):
        mu = metrics[key]["mean"]
        sd = metrics[key]["std"]
        if mu is None:
            parts.append(f"{key}=—")
        else:
            parts.append(f"{key}={fmt_float(mu)}±{fmt_float(sd)}")
    return ", ".join(parts)


def render_markdown(snapshot: dict[str, Any], interval_seconds: int) -> str:
    lines = [
        START,
        "",
        "### Supplement Monitor (auto-updated)",
        "",
        f"- Last poll: `{snapshot['timestamp']}`",
        f"- Poll interval: {interval_seconds // 60} minutes",
        f"- Result directory: `{snapshot['supplement_dir']}`",
        "",
        "| Experiment | Labels | GPU | Status | Result JSONs | Completed training seeds | Current mean±std | Running PIDs | Notes |",
        "|---|---|---:|---|---:|---|---|---|---|",
    ]
    for arm in snapshot["arms"]:
        seeds = ", ".join(arm["completed_train_seeds"]) if arm["completed_train_seeds"] else "—"
        pids = ", ".join(arm["running_pids"]) if arm["running_pids"] else "—"
        notes = "; ".join(arm["errors"][:2]) if arm["errors"] else ("aggregate present" if arm["has_aggregate"] else "—")
        lines.append(
            "| {name} | {labels} | {gpu} | {status} | {result_count}/{expected_results} | {seeds} | {metrics} | {pids} | {notes} |".format(
                name=arm["name"],
                labels=arm["labels"],
                gpu=arm["gpu"],
                status=arm["status"],
                result_count=arm["result_count"],
                expected_results=arm["expected_results"],
                seeds=seeds,
                metrics=render_metrics(arm["metrics"]),
                pids=pids,
                notes=notes.replace("|", "/"),
            )
        )
    gpu = snapshot.get("gpu") or "unavailable"
    lines.extend(["", "GPU snapshot:", "", "```", gpu[:2000], "```", "", END])
    return "\n".join(lines) + "\n"


def update_report(report_path: Path, block: str) -> None:
    text = report_path.read_text()
    if START in text and END in text:
        before = text.split(START, 1)[0].rstrip()
        after = text.split(END, 1)[1].lstrip()
        report_path.write_text(before + "\n\n" + block + "\n" + after)
        return
    anchor = "### Final Decision (Phase 7, updated after policy-responsive H30)"
    if anchor in text:
        text = text.replace(anchor, block + "\n" + anchor, 1)
    else:
        text = text.rstrip() + "\n\n" + block
    report_path.write_text(text)


def notify_feishu(snapshot: dict[str, Any]) -> str:
    if not FEISHU_CONFIG.exists():
        return "feishu_config=missing"
    try:
        cfg = json.loads(FEISHU_CONFIG.read_text())
    except Exception:
        return "feishu_config=invalid"
    if cfg.get("mode") != "push" or not cfg.get("webhook_url"):
        return "feishu_mode=off"
    arm_lines = []
    for arm in snapshot["arms"]:
        arm_lines.append(
            f"{arm['name']}: {arm['status']}, results {arm['result_count']}/{arm['expected_results']}, "
            f"{render_metrics(arm['metrics'])}"
        )
    text = "[LLMSTSC] Phase7 supplement 40-min poll\n" + "\n".join(arm_lines)
    payload = {"msg_type": "text", "content": {"text": text[:3500]}}
    req = urllib.request.Request(
        cfg["webhook_url"],
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read(200).decode("utf-8", "replace")
        return f"feishu_status=sent {resp.status} {body[:120]}"
    except Exception as exc:
        return f"feishu_status=failed {type(exc).__name__}"


def all_terminal(snapshot: dict[str, Any]) -> bool:
    statuses = {arm["status"] for arm in snapshot["arms"]}
    return statuses <= {"done", "warning", "stalled"} and "running" not in statuses


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--supplement-dir", type=Path, default=DEFAULT_SUPPLEMENT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--interval-seconds", type=int, default=2400)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-notify", action="store_true")
    parser.add_argument("--max-polls", type=int, default=0, help="0 means unlimited until terminal")
    args = parser.parse_args()

    monitor_dir = args.supplement_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    snapshot_jsonl = monitor_dir / "monitor_snapshots.jsonl"

    polls = 0
    while True:
        snapshot = collect_snapshot(args.supplement_dir)
        block = render_markdown(snapshot, args.interval_seconds)
        update_report(args.report, block)
        with snapshot_jsonl.open("a") as fh:
            fh.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        print(f"[{snapshot['timestamp']}] updated {args.report}")
        for arm in snapshot["arms"]:
            print(
                f"  {arm['name']}: {arm['status']} {arm['result_count']}/{arm['expected_results']} "
                f"{render_metrics(arm['metrics'])}"
            )
        if not args.no_notify:
            print(notify_feishu(snapshot))
        sys.stdout.flush()

        polls += 1
        if args.once:
            break
        if all_terminal(snapshot):
            print("all experiments terminal; monitor exiting")
            break
        if args.max_polls and polls >= args.max_polls:
            print(f"reached max_polls={args.max_polls}; monitor exiting")
            break
        time.sleep(max(60, args.interval_seconds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
