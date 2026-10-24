#!/usr/bin/env python3
"""Best-effort Feishu progress for a long experiment; no secrets in output."""
import argparse
import json
import time
import urllib.request
from collections import deque
from pathlib import Path


def send(webhook, title, body):
    payload = {"msg_type": "interactive", "card": {"header": {"title": {"tag": "plain_text", "content": title}},
               "elements": [{"tag": "markdown", "content": body}]}}
    request = urllib.request.Request(webhook, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            print(f"notification_http_status={response.status}", flush=True)
    except Exception as exc:
        print(f"notification_error={type(exc).__name__}", flush=True)


def latest_line(path):
    if not path.exists():
        return "not started"
    with path.open(errors="replace") as handle:
        lines = deque((line.strip() for line in handle if "[train epoch" in line or "val_top1_cost_regret" in line), maxlen=1)
    return lines[0][-200:] if lines else "loading"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    root = Path(args.root)
    try:
        config = json.loads(Path("/root/.claude/feishu.json").read_text())
    except (OSError, ValueError):
        return
    mode = str(config.get("mode", config.get("notify_mode", "on"))).lower()
    webhook = config.get("webhook_url") or config.get("webhook")
    if mode != "push" or not webhook:
        return
    started = time.monotonic()
    next_update = started + 40 * 60
    while True:
        completion = root / "completion.status"
        if completion.exists():
            send(webhook, "Information-matched ablation complete", completion.read_text()[:300] + "\n" + str(root))
            return
        now = time.monotonic()
        if now >= next_update:
            lines = []
            for scenario in ("Jinan-1", "Hangzhou-1"):
                for arm in ("Text", "TS2I"):
                    status = root / "status" / f"{scenario}_{arm}.status"
                    progress = root / "train" / scenario / arm / "train_progress.log"
                    lines.append(f"{scenario} {arm}: {status.read_text().strip() if status.exists() else 'queued'}; latest {latest_line(progress)}")
            send(webhook, "Information-matched ablation progress", "\n".join(lines) + "\n" + str(root))
            next_update += 40 * 60
        time.sleep(60)


if __name__ == "__main__":
    main()
