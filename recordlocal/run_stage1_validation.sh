#!/usr/bin/env bash
set -Eeuo pipefail

REPO=/home/code/LLMSTSC
ROOT="$REPO/recordlocal/runs/stage1_validation"
PYTHON=${PYTHON:-/home/sysu/anaconda3/envs/hnr5090/bin/python}
SESSION=stage1_validation

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is not executable: $PYTHON" >&2
  exit 2
fi

mkdir -p "$ROOT/logs"
if [[ -s "$ROOT/launcher.pid" ]]; then
  old_pid=$(<"$ROOT/launcher.pid")
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    old_cmd=$(timeout --foreground 5s ps -p "$old_pid" -o args= 2>/dev/null || true)
    if [[ "$old_cmd" == *"recordlocal/stage1_validation.py"* ]]; then
      echo "Stage I validation campaign is already running: pid=$old_pid log=$ROOT/logs/orchestrator.log"
      exit 0
    fi
  fi
fi

if [[ "${1:-}" == "--foreground" ]]; then
  exec "$PYTHON" "$REPO/recordlocal/stage1_validation.py" \
    --mode run --repo "$REPO" --root "$ROOT" --model /mnt/data/Qwen3-VL-8B-Instruct
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  pane_pid=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
  pane_cmd=$(timeout --foreground 5s ps -p "$pane_pid" -o args= 2>/dev/null || true)
  if [[ "$pane_cmd" == *"recordlocal/stage1_validation.py"* ]]; then
    printf '%s\n' "$pane_pid" > "$ROOT/launcher.pid"
    echo "Stage I validation campaign is already running: pid=$pane_pid session=$SESSION log=$ROOT/logs/orchestrator.log"
    exit 0
  fi
  echo "tmux session '$SESSION' exists but is not the Stage I validation campaign." >&2
  exit 3
fi

launch=(env PYTHONUNBUFFERED=1 "$PYTHON" "$REPO/recordlocal/stage1_validation.py"
  --mode run --repo "$REPO" --root "$ROOT" --model /mnt/data/Qwen3-VL-8B-Instruct)
printf -v launch_command '%q ' "${launch[@]}"
printf -v log_path '%q' "$ROOT/logs/orchestrator.log"
launch_command="exec $launch_command>> $log_path 2>&1"
tmux new-session -d -s "$SESSION" -c "$REPO" "$launch_command" \
  2> "$ROOT/logs/tmux-launch.log"
launcher_pid=$(tmux display-message -p -t "$SESSION" '#{pane_pid}')
printf '%s\n' "$launcher_pid" > "$ROOT/launcher.pid"
printf 'Started Stage I validation campaign: pid=%s session=%s\nLog: %s\nResults: %s\n' \
  "$launcher_pid" "$SESSION" "$ROOT/logs/orchestrator.log" "$ROOT"
