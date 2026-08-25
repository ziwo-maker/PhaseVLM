#!/usr/bin/env python3
"""Background monitor for Phase-7 H120 follow-up experiments.

Polls result directories at a fixed interval, sends best-effort Feishu progress
updates, and sends a final completion notification with aggregate numbers.
Secrets are read from /root/.claude/feishu.json and never printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


ROOT = Path("/home/code/LLMSTSC")
FOLLOWUP = ROOT / "records/pipeline_runs/phase7_policy_responsive_h120_followup_20260814"
SUPPLEMENT = ROOT / "records/pipeline_runs/phase7_policy_responsive_supplement_20260814"
LOG_PATH = FOLLOWUP / "phase7_h120_followup_monitor.log"
STATE_PATH = FOLLOWUP / "phase7_h120_followup_monitor_state.json"
FEISHU_CFG = Path("/root/.claude/feishu.json")

EXPECTED = {
    "h120_k32_more_a": 3,
    "h120_k32_more_b": 3,
    "h120_k118": 3,
}
METRICS = ("ATT", "AQL", "AWT")
ZERO_SHOT = {"ATT": 335.63, "AQL": 199.32, "AWT": 67.75}
H30_K32_N9 = {"ATT": 327.45, "AQL": 188.40, "AWT": 66.23}
H30_K118 = {"ATT": 331.51, "AQL": 194.37, "AWT": 76.10}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"[{now()}] {msg}\n")


def feishu_send(text: str) -> bool:
    try:
        if not FEISHU_CFG.exists():
            log("feishu_config=missing")
            return False
        cfg = json.loads(FEISHU_CFG.read_text())
        if cfg.get("mode") != "push" or not cfg.get("webhook_url"):
            log("feishu_mode=off")
            return False
        payload = {"msg_type": "text", "content": {"text": text}}
        req = urllib.request.Request(
            cfg["webhook_url"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = resp.read(200).decode("utf-8", "replace")
        log(f"feishu_status=sent http={resp.status} body={body[:120]}")
        return True
    except Exception as exc:  # best effort
        log(f"feishu_status=failed {type(exc).__name__}")
        return False


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def status_for(exp: str) -> Dict[str, str]:
    data = read_json(FOLLOWUP / exp / "status.json")
    return data if isinstance(data, dict) else {}


def is_exp_done(exp: str) -> bool:
    status = status_for(exp)
    return len(status) >= EXPECTED[exp] and all(v == "ok" for v in status.values())


def exp_failed(exp: str) -> List[str]:
    return [f"{k}={v}" for k, v in status_for(exp).items() if v != "ok"]


def running_process_lines() -> List[str]:
    pattern = "phase7_policy_responsive_h120_followup|run_phase7_policy_responsive|run_phase7_crcd|run_projection_online|train_projection_bc"
    try:
        proc = subprocess.run(
            ["bash", "-lc", f"ps -eo pid,ppid,stat,etime,cmd | rg '{pattern}' | head -n 40"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            timeout=10,
        )
        return [line for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def count_result_jsons(exp: str) -> int:
    return len(list((FOLLOWUP / exp).rglob("result_seed*.json")))


def load_eval_rows(base: Path, k: str | None = None) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for path in sorted(base.rglob("eval_summary.json")):
        if k and f"/{k}/" not in str(path):
            continue
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        if all(m in data for m in METRICS):
            metrics = data
        elif isinstance(data.get("mean"), dict) and all(m in data["mean"] for m in METRICS):
            metrics = data["mean"]
        else:
            continue
        rows.append({m: float(metrics[m]) for m in METRICS})
    return rows


def stats(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if not rows:
        return out
    for m in METRICS:
        vals = [row[m] for row in rows]
        mean = sum(vals) / len(vals)
        pop_std = math.sqrt(sum((x - mean) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0.0
        out[m] = {"mean": mean, "std": pop_std, "n": float(len(vals))}
    return out


def fmt_stats(label: str, st: Dict[str, Dict[str, float]]) -> str:
    if not st:
        return f"{label}: no completed eval_summary yet"
    n = int(next(iter(st.values()))["n"])
    return (
        f"{label} n={n}: "
        f"ATT={st['ATT']['mean']:.2f}±{st['ATT']['std']:.2f}, "
        f"AQL={st['AQL']['mean']:.2f}±{st['AQL']['std']:.2f}, "
        f"AWT={st['AWT']['mean']:.2f}±{st['AWT']['std']:.2f}"
    )


def current_summary() -> Tuple[str, bool, bool]:
    lines = ["Phase7 H120 follow-up monitor"]
    all_done = True
    any_failed = False
    for exp in EXPECTED:
        done = is_exp_done(exp)
        all_done = all_done and done
        failures = exp_failed(exp)
        any_failed = any_failed or bool(failures)
        status = status_for(exp)
        lines.append(
            f"- {exp}: {'done' if done else 'running'}; "
            f"status_ok={sum(1 for v in status.values() if v == 'ok')}/{EXPECTED[exp]}; "
            f"result_jsons={count_result_jsons(exp)}/{EXPECTED[exp] * 3}"
        )
        if failures:
            lines.append(f"  failures: {'; '.join(failures[:3])}")
    rows_h120_k32 = (
        load_eval_rows(SUPPLEMENT / "h120_k32", "k32")
        + load_eval_rows(FOLLOWUP / "h120_k32_more_a", "k32")
        + load_eval_rows(FOLLOWUP / "h120_k32_more_b", "k32")
    )
    rows_h120_k118 = load_eval_rows(FOLLOWUP / "h120_k118", "k118")
    lines.append(fmt_stats("H120 K32 combined", stats(rows_h120_k32)))
    lines.append(fmt_stats("H120 K118", stats(rows_h120_k118)))
    proc_lines = running_process_lines()
    lines.append(f"active_process_lines={len(proc_lines)}")
    return "\n".join(lines), all_done, any_failed


def final_text() -> str:
    rows_h120_k32 = (
        load_eval_rows(SUPPLEMENT / "h120_k32", "k32")
        + load_eval_rows(FOLLOWUP / "h120_k32_more_a", "k32")
        + load_eval_rows(FOLLOWUP / "h120_k32_more_b", "k32")
    )
    rows_h120_k118 = load_eval_rows(FOLLOWUP / "h120_k118", "k118")
    st32 = stats(rows_h120_k32)
    st118 = stats(rows_h120_k118)
    verdicts: List[str] = []
    if st32:
        awt32 = st32["AWT"]["mean"]
        verdicts.append(
            f"H120 K32 vs H30 K32 n9: AWT {awt32:.2f} vs {H30_K32_N9['AWT']:.2f} "
            f"({'better' if awt32 < H30_K32_N9['AWT'] else 'not better'})"
        )
    if st118:
        awt118 = st118["AWT"]["mean"]
        verdicts.append(
            f"H120 K118 high-K gate: AWT {awt118:.2f} vs zero-shot {ZERO_SHOT['AWT']:.2f}, "
            f"H30 K118 {H30_K118['AWT']:.2f}"
        )
    return (
        "Phase7 H120 follow-up experiments completed.\n"
        + fmt_stats("H120 K32 combined", st32)
        + "\n"
        + fmt_stats("H120 K118", st118)
        + "\n"
        + "\n".join(verdicts)
        + f"\nResults: {FOLLOWUP}"
    )


def save_state(payload: Dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=2400)
    parser.add_argument("--max-hours", type=float, default=24.0)
    parser.add_argument("--notify-start", action="store_true")
    args = parser.parse_args()

    deadline = time.time() + args.max_hours * 3600
    if args.notify_start:
        feishu_send(
            "Phase7 H120 follow-up 40-minute monitor started. "
            f"Polling every {args.interval_seconds // 60} minutes. Results: {FOLLOWUP}"
        )
    iteration = 0
    while True:
        iteration += 1
        summary, all_done, any_failed = current_summary()
        log(summary.replace("\n", " | "))
        save_state({"time": now(), "iteration": iteration, "all_done": all_done, "any_failed": any_failed})
        if all_done:
            feishu_send(final_text())
            log("monitor_complete")
            return
        if any_failed:
            feishu_send("Phase7 H120 follow-up detected a failed status.\n" + summary)
        else:
            feishu_send("Phase7 H120 follow-up progress update.\n" + summary)
        if time.time() >= deadline:
            feishu_send("Phase7 H120 follow-up monitor reached max runtime before completion.\n" + summary)
            log("monitor_deadline_reached")
            return
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
