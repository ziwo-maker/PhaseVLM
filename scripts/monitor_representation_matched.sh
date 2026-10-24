#!/usr/bin/env bash
set -u
ROOT=/home/code/LLMSTSC
PY=/home/sysu/anaconda3/envs/hnr5090/bin/python
RUN_VISUAL="$1"
RUN_TEXT="$2"

summary() {
  "$PY" - "$RUN_VISUAL" "$RUN_TEXT" <<'PY'
import json, sys
from pathlib import Path
parts = []
for root_text in sys.argv[1:]:
    root = Path(root_text)
    done = sorted(root.glob("online/*/train_*/sim_*/result.json"))
    running = sorted(root.glob("status/train_*.status"))
    latest = "none"
    metrics = sorted(root.glob("train/*/seed_*/metrics.jsonl"), key=lambda p: p.stat().st_mtime)
    if metrics:
        try:
            last = json.loads(metrics[-1].read_text().splitlines()[-1])
            latest = f"epoch={last.get('epoch')} val_top1_cost_regret={last.get('val_top1_cost_regret')}"
        except (OSError, ValueError, IndexError):
            pass
    completion = (root / "completion.status").read_text().strip().replace("\n", ",") if (root / "completion.status").exists() else "running"
    parts.append(f"{root.name}: status={completion} train_status={len(running)} online={len(done)} latest={latest}")
print("PhaseVLM matched representation: " + "; ".join(parts))
PY
}

while true; do
  sleep 2400
  message="$(summary)"
  if rg -q '^state=(completed|failed)$' "$RUN_VISUAL/completion.status" 2>/dev/null &&
     rg -q '^state=(completed|failed)$' "$RUN_TEXT/completion.status" 2>/dev/null; then
    timeout --foreground 15s "$PY" "$ROOT/scripts/feishu_push.py" "Finished: $message" || true
    break
  fi
  timeout --foreground 15s "$PY" "$ROOT/scripts/feishu_push.py" "$message" || true
done
