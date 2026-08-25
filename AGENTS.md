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

## Log Inspection Priority

When inspecting experiment logs, prioritize high-information-density artifacts
first. Start with result analyses, aggregate JSON/CSV files, completion audits,
status files, provenance manifests, and structured metric files such as
`metrics.jsonl` or `result.json`. If these do not answer the question, then read
training progress logs, pipeline logs, startup monitor snapshots, and targeted
error excerpts. Only inspect low-density raw outputs such as wrapper logs,
`nohup.out`, large replay logs, roadnet logs, or full simulator dumps after the
higher-density sources are insufficient.

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

## Experiment Progress Notifications

For every running experiment, send a best-effort Feishu progress notification
every 40 minutes when Feishu is configured for push notifications. Each update
must include the current progress and the latest available result, such as the
most recent or best validation metric. Keep the final completion notification.

## Startup Monitoring Window

For every experiment, training job, or training-dataset generation launched by
Codex, actively monitor the first 5 minutes before ending launch monitoring.
Use bounded snapshots only, not streaming commands. During this window, inspect
the process tree, log tail, and relevant resource signal (for example GPU memory
and utilization for GPU jobs) at launch, then poll exactly once every 60 seconds
for the full 5-minute window. Treat an unexpected process exit, traceback,
repeated retry, stalled progress, or absent resource allocation as a startup
failure; investigate and report it rather than claiming the launch succeeded.
End startup monitoring only after the job has remained healthy for the complete
5-minute window, or after it completes successfully.

## Online Test Parallelism

When running multiple independent online tests, prefer multi-GPU parallel
execution whenever enough idle GPUs are available. Assign one test worker per
GPU with an explicit `CUDA_VISIBLE_DEVICES` value, and keep each worker's log,
result JSON, memo, status row, and working directory separate. Only fall back to
sequential execution when the tests share mutable state, GPU memory is
insufficient, the user explicitly requests sequential runs, or parallelism would
make the comparison invalid.

## Default Traffic-Flow Evaluation Scope

Unless the user explicitly requests generalization, robustness, cross-flow, or
multi-flow evaluation, test every trained traffic-control policy only on the
same traffic flow used to collect or train its dataset. Do not add demand-shift
or other extra traffic flows by default. Record the training and test flow in
the experiment manifest so the same-flow protocol is auditable.
<!-- ARIS-CODEX:BEGIN -->
## ARIS Codex Skill Scope
ARIS Codex packages installed in this project: skills-codex
Managed entries: 83
Manifest: `.aris/installed-skills-codex.txt`
ARIS repo root: `/home/code/LLMSTSC/Auto-claude-code-research-in-sleep`
Project skill path: `.agents/skills/<skill-name>`
For ARIS Codex workflows, prefer the project-local skills under `.agents/skills/`.
When a skill needs ARIS helper scripts, resolve the repo root from the manifest or set it explicitly:
`ARIS_REPO=$(awk -F'\t' '$1=="repo_root"{print $2; exit}' "/home/code/LLMSTSC/.aris/installed-skills-codex.txt")`
Do not edit or delete symlinked skills in place; update upstream or rerun:
`bash /home/code/LLMSTSC/Auto-claude-code-research-in-sleep/tools/install_aris_codex.sh "/home/code/LLMSTSC" --reconcile`
For copied Codex installs, use:
`bash /home/code/LLMSTSC/Auto-claude-code-research-in-sleep/tools/smart_update_codex.sh --project "/home/code/LLMSTSC"`
<!-- ARIS-CODEX:END -->
