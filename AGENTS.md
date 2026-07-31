# Codex Command Safety Rules

## General command policy

Codex must avoid long-running, blocking, interactive, or streaming commands unless explicitly requested by the user.

Before running any command, Codex should briefly explain:
- what the command does;
- why it is needed;
- whether it may take a long time;
- how output is limited.

All test, search, log, build, install, and server-start commands must limit runtime, output size, or both. Prefer `timeout --foreground 30s` for commands that may hang, and use shorter timeouts for quick probes.

For commands that need to keep running, start them detached with `nohup`, `tmux`, or `screen`, write logs to a file, and report the PID and log path. Inspect logs with bounded snapshots only.

Prefer bounded output commands such as:
- `tail -n 200 file`
- `sed -n '1,200p' file`
- `rg -n "ERROR|Exception|Traceback" file | tail -n 200`
- `docker logs --tail 200 --since 30m container`
- `journalctl -n 200 --no-pager`
- `kubectl logs --tail=200 --since=30m pod`

## Forbidden blocking commands

Do not run these commands directly:

```bash
tail -f
docker logs -f
watch
npm run dev
npm start
pnpm dev
yarn dev
python -m http.server
uvicorn
flask run
streamlit run
ping without count limit
pytest without timeout
grep -R without exclusions
find . without exclusions
less
more
journalctl -f
kubectl logs -f
```

## Required safe replacements

Use safe replacements with explicit time, output, or scope limits:

```bash
timeout --foreground 30s pytest
timeout --foreground 30s npm test
timeout --foreground 30s npm run build
timeout --foreground 30s pnpm test
timeout --foreground 30s pnpm build
timeout --foreground 10s ping -c 4 host
rg "pattern" --glob '!node_modules' --glob '!dist' --glob '!build'
find . -path './.git' -prune -o -path './node_modules' -prune -o -type f -print
tail -n 200 file.log
docker logs --tail 200 --since 30m container
```

When a development server is required, run it detached and bound the log inspection:

```bash
nohup npm run dev > /tmp/project-dev.log 2>&1 & echo $!
tail -n 200 /tmp/project-dev.log
```

## Completion Notification

After every task for this project is completed, send the user a concise Feishu
completion notification. Treat the notification as best effort: read
`~/.claude/feishu.json`, honor its configured mode, never expose its secrets,
and never delay or fail the task because Feishu is unavailable or unconfigured.
Include the task outcome and any material verification result in the message.

## Experiment Approval And Progress Notifications

Before launching any experiment, present the complete startup command to the
user and wait for explicit approval. Do not reserve GPUs, create a detached
session, or start the command until that approval is received.

For every running experiment, send a best-effort Feishu progress notification
every 40 minutes when Feishu is configured for push notifications. Each update
must include the current progress and the latest available result, such as the
most recent or best validation metric. Keep the final completion notification.

## Online Test Parallelism

When running multiple independent online tests, prefer multi-GPU parallel
execution whenever enough idle GPUs are available. Assign one test worker per
GPU with an explicit `CUDA_VISIBLE_DEVICES` value, and keep each worker's log,
result JSON, memo, status row, and working directory separate. Only fall back to
sequential execution when the tests share mutable state, GPU memory is
insufficient, the user explicitly requests sequential runs, or parallelism would
make the comparison invalid.

Before launching parallel online tests, still follow the experiment approval
rule above: show the complete startup commands, including GPU assignment and
output paths, and wait for explicit user approval.
