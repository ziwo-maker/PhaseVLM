#!/usr/bin/env bash
set -u

REPO_ROOT="${REPO_ROOT:-/home/code/LLMSTSC}"
PYTHON="${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
MAIN_PID="${MAIN_PID:?MAIN_PID is required}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-2400}"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/paper_additions/run_jinan_missing_ablation_feishu_monitor.log}"
FEISHU_PUSH="${FEISHU_PUSH:-$REPO_ROOT/scripts/feishu_push.py}"

progress_message() {
  local phase="$1"
  "$PYTHON" - "$RUN_ROOT" "$MAIN_PID" "$phase" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
pid = sys.argv[2]
phase = sys.argv[3]
train_seeds = ["2024", "42", "123"]
online_seeds = ["2024", "42", "123", "7"]
train_arms = [
    "control_shuffled_targets",
    "stage1_random_projector",
    "hard_bc",
    "hard_plus_soft",
    "hard_soft_ranking",
    "full_loss",
]
online_arms = train_arms + [
    "visual_shuffled_image",
    "visual_constant_image",
    "visual_zero_prefix",
]

train_done = 0
for arm in train_arms:
    for seed in train_seeds:
        metrics = root / "stage2" / arm / f"train_seed_{seed}" / "checkpoints" / "metrics.jsonl"
        best = root / "stage2" / arm / f"train_seed_{seed}" / "checkpoints" / "best.pt"
        max_epoch = 0
        if metrics.is_file() and best.is_file():
            for line in metrics.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    max_epoch = max(max_epoch, int(json.loads(line).get("epoch", 0)))
                except Exception:
                    pass
        train_done += int(max_epoch >= 15)

online_done = 0
for arm in online_arms:
    for train_seed in train_seeds:
        for online_seed in online_seeds:
            result = root / "online_eval" / arm / f"train_seed_{train_seed}" / f"online_seed_{online_seed}" / "result.json"
            if result.is_file() and result.stat().st_size > 0:
                try:
                    json.loads(result.read_text(encoding="utf-8"))
                    online_done += 1
                except Exception:
                    pass

current = "unknown"
status = root / "status.tsv"
if status.is_file():
    lines = [line for line in status.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    if len(lines) > 1:
        current = lines[-1]

print(
    "[Jinan ablation resume] "
    f"phase={phase} pid={pid} train={train_done}/{len(train_arms) * len(train_seeds)} "
    f"online={online_done}/{len(online_arms) * len(train_seeds) * len(online_seeds)} "
    f"run_root={root} latest_status={current[:900]}"
)
PY
}

send_update() {
  local phase="$1"
  local msg
  msg="$(progress_message "$phase" 2>>"$LOG_PATH")" || return 0
  "$PYTHON" "$FEISHU_PUSH" "$msg" >>"$LOG_PATH" 2>&1 || true
}

{
  printf '[%s] monitor start pid=%s run_root=%s interval=%s\n' "$(date --iso-8601=seconds)" "$MAIN_PID" "$RUN_ROOT" "$INTERVAL_SECONDS"
} >>"$LOG_PATH"

send_update launched
while kill -0 "$MAIN_PID" 2>/dev/null; do
  sleep "$INTERVAL_SECONDS"
  if kill -0 "$MAIN_PID" 2>/dev/null; then
    send_update progress
  fi
done
send_update final
printf '[%s] monitor done pid=%s\n' "$(date --iso-8601=seconds)" "$MAIN_PID" >>"$LOG_PATH"
