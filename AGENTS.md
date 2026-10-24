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

## Feishu Notification Policy

Only send Feishu notifications for experiments that are expected to run longer
than 1 hour, or that are discovered during monitoring to have already run longer
than 1 hour. Do not send Feishu notifications for ordinary tasks, code edits,
short commands, short experiments, or completion of non-long-running work.

For qualifying long-running experiments, send a best-effort Feishu progress
notification every 40 minutes when Feishu is configured for push notifications.
Each update must include the current progress and the latest available result,
such as the most recent or best validation metric. Send one final completion
notification for the qualifying long-running experiment. Treat all Feishu
notifications as best effort: read `~/.claude/feishu.json`, honor its configured
mode, never expose its secrets, and never delay or fail the task because Feishu
is unavailable or unconfigured.

## Startup Monitoring Window

For every experiment, training job, or training-dataset generation launched by
Codex, actively monitor the first 2 minutes before ending launch monitoring.
Use bounded snapshots only, not streaming commands. During this window, inspect
the process tree, log tail, and relevant resource signal (for example GPU memory
and utilization for GPU jobs) at launch, then poll exactly once every 60 seconds
for the full 2-minute window. Treat an unexpected process exit, traceback,
repeated retry, stalled progress, or absent resource allocation as a startup
failure; investigate and report it rather than claiming the launch succeeded.
End startup monitoring only after the job has remained healthy for the complete
2-minute window, or after it completes successfully.

## Online Test Parallelism

When running multiple independent online tests, prefer multi-GPU parallel
execution whenever enough idle GPUs are available. Assign one test worker per
GPU with an explicit `CUDA_VISIBLE_DEVICES` value, and keep each worker's log,
result JSON, memo, status row, and working directory separate. Only fall back to
sequential execution when the tests share mutable state, GPU memory is
insufficient, the user explicitly requests sequential runs, or parallelism would
make the comparison invalid.

## Online Test Default Protocol

Online tests must be plural by default: run multiple online tests unless the
user explicitly asks for a single run. Do not select or constrain a specific
epoch based on `online_test_best_weights/`; use the requested or experiment
protocol checkpoint selection instead.

## Explicit Five-Flow Cross-Generalization Protocol

For the requested Jinan-1 and Hangzhou-2 best-weight cross-generalization
comparison, the experiment count is five distinct source-to-target flow pairs,
not two model directions and not an automatically expanded seed matrix. Run
each pair once with seed 2024 unless the user explicitly asks for additional
seeds:

1. Jinan-1 best `paper_main_table_near_jinan1.pt` on Hangzhou-1
   `anon_4_4_hangzhou_real.json`.
2. Jinan-1 best `paper_main_table_near_jinan1.pt` on Hangzhou-2
   `anon_4_4_hangzhou_real_5816.json`.
3. Hangzhou-2 best `epoch_9.pt` on Jinan-1 `anon_3_4_jinan_real.json`.
4. Hangzhou-2 best `epoch_9.pt` on Jinan-2
   `anon_3_4_jinan_real_2000.json`.
5. Hangzhou-2 best `epoch_9.pt` on Jinan-3
   `anon_3_4_jinan_real_2500.json`.

The Jinan-1 checkpoint uses `/home/data/model/qwen-vl-8B`, hidden dimension
64, 8 prefix tokens, and max length 64. The Hangzhou-2 checkpoint uses
`/mnt/data/Qwen3-VL-8B-Instruct`, hidden dimension 256, 8 prefix tokens, and
max length 512. Preserve each checkpoint's own model family and architecture;
do not substitute the other checkpoint's settings. Use `run_counts=3600`, one
round, queue/wait/short prompt mode, 16-step images at 224 pixels, minimum
action time 30 seconds, traffic-series interval 2 seconds, bf16, and disabled
phase overrides. Record the five pairs explicitly in the manifest and keep
one result, status, and log per pair. Reuse already completed exact pair
results rather than rerunning them solely to reach a count of five.

## Experiment Launch Parallelism

When launching experiments, use multi-GPU execution by default whenever multiple
idle GPUs are available. Prefer assigning independent runs, seeds, folds, or
configurations across GPUs with explicit `CUDA_VISIBLE_DEVICES` values and
separate logs/results per worker. Use a single GPU or sequential execution only
when GPU memory is insufficient, the experiment requires shared mutable state,
the user explicitly requests sequential execution, or multi-GPU scheduling would
invalidate the comparison.

## Default Traffic-Flow Evaluation Scope

Unless the user explicitly requests generalization, robustness, cross-flow, or
multi-flow evaluation, test every trained traffic-control policy only on the
same traffic flow used to collect or train its dataset. Do not add demand-shift
or other extra traffic flows by default. Record the training and test flow in
the experiment manifest so the same-flow protocol is auditable.

## Online Test Best-Weight Archive

When an online test on any traffic flow exceeds the current best result for
that same flow, archive the winning weight files separately under
`online_test_best_weights/<flow_name>/`. Create the flow-specific subdirectory
if it does not already exist, copy the exact winning weights there, and keep
the archive independent from transient experiment output directories.

A completed single-seed online test is sufficient to select and archive a
checkpoint; multi-seed validation is not an archive prerequisite. Record the
exact seed and evaluation scope in the archive index. When the selection uses
a run-local composite metric rather than an all-time flow record, state that
scope explicitly and do not claim it supersedes historical records on metrics
where it does not improve them.

Maintain `online_test_best_weights/README.md` as the reproduction index. For
each archived online test, record the flow name, metric that beat the previous
best, previous best value, new best value, source checkpoint path, archived
weight path, exact online-test command, environment assumptions, relevant seed
or GPU settings, result artifact paths, and concise reproduction steps.

## Online Test Best-Weights Reproduction Scope

When asked to reproduce `online_test_best_weights/`, treat the archive as a
directional reproduction recipe unless the user explicitly requests bit-exact
checkpoint restoration. If an archived checkpoint is missing, do not block on
recovering that exact file by default. Instead, use the archive entry's
non-hardware parameters as the target protocol and rerun the pipeline from
Stage 1 with the current available bestweights/data recipe.

When reproducing `online_test_best_weights/`, prefer using all idle GPUs for
Stage 1, Stage 2, and the multi-seed online tests. Assign one worker per GPU
when the machine can support it, and only reduce GPU count when memory is
insufficient, the run needs shared mutable state, or the user explicitly asks
for fewer GPUs.

Preserve the important experiment semantics from the archive entry where
possible: dataset, traffic flow, prompt mode, sequence/image dimensions,
projector dimensions, prefix settings, max length, run counts, min action time,
override settings, model family/path preference, same-flow versus cross-flow
scope, and evaluation seeds. Hardware-dependent parameters such as GPU ids,
worker placement, DDP world size, batch size per GPU, gradient accumulation, and
timeouts may be adjusted for the available machine, as long as the run manifest
records the actual settings. This is a broad reproduction, not a strict
bit-for-bit reproduction, unless the user says otherwise.

## Newyork Checkpoint Interpretation

For Newyork 28x7 wait-optimization work, do not describe epoch 15 as the best
checkpoint by default. Treat epoch 15 only as a wait-oriented candidate until it
beats the current baseline under multi-seed online evaluation and the configured
selector constraints. Prefer statements such as "candidate", "reference", or
"needs multi-seed validation" unless aggregate online results prove otherwise.

## Default Model Path

Use `/mnt/data/Qwen3-VL-8B-Instruct/` as the default model path unless the user
explicitly requests a different model.
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
