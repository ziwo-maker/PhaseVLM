#!/usr/bin/env python3
"""Best-effort Feishu text push. Reads message from argv[1] or stdin.

No-op (exit 0) if ~/.claude/feishu.json is missing or not in push mode.
Used by detached experiment scripts; never prints secrets.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

CFG = os.path.expanduser("~/.claude/feishu.json")


def main() -> int:
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        text = sys.stdin.read().strip()
    if not text:
        return 0
    try:
        cfg = json.load(open(CFG))
        if cfg.get("mode") != "push" or not cfg.get("webhook_url"):
            return 0
        payload = {"msg_type": "text", "content": {"text": text[:4000]}}
        req = urllib.request.Request(
            cfg["webhook_url"],
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
        print("feishu sent", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"feishu skipped: {type(e).__name__}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
