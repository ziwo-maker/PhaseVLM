# Remote Hangzhou Real Stage 2 Structure Memory

Date: 2026-05-24

Remote root analyzed: `/gemini/code/LLMSTSC`

Purpose: local memory of the remote experiment/code directory structure for later Hangzhou real Stage 2 diagnosis. This file records structure and known artifacts only; no new experiment was run.

## High-Level Layout

The remote repository is a traffic-signal-control project combining CityFlow simulation, LLM/VLM policy inference, offline behavior cloning, and later PPO-style refinement.

Top-level areas:

- `projection_train/`: core Stage 2/Stage 3 model, dataset, policy, online evaluation, and PPO code.
- `scripts/`: orchestration wrappers for projection stages, checkpoint selection, and sim-expert data generation.
- `records/`: dominant experiment artifact root. It contains pipeline runs, checkpoints, logs, online test results, and sim-expert datasets.
- `data/`: CityFlow traffic data and online recorded rollout directories.
- `model/`: CityFlow model/output directories for online tests and historical runs.
- `README/`: project notes and experiment reports, including Stage 2 and sim-expert design notes.
- `run_sim_expert_action.py`: sim-expert data generation and post-processing entrypoint.
- `train_projection_ppo_traj.py`: Stage 3 trajectory PPO/refinement entrypoint at repo root on remote.
- `snapshot_seed_dataset.py`: snapshot-seed dataset helper for Stage 3.

Large remote disk usage observed:

- Whole repo: about `231G`.
- `records/`: about `222G`.
- `records/pipeline_runs/`: about `213G`.
- `records/sim_expert_validation_search/`: about `8.3G`.
- `data/`: about `7.8G`, mostly `data/recorded/`.

## Core Code Responsibilities

`projection_train/train_projection_bc.py`

- Stage 2 offline behavior cloning trainer.
- Uses sim-expert JSONL data.
- Loads Stage 1 alignment checkpoint through `--align_ckpt`.
- Key functions include dataset splitting, action distribution, class weights, soft targets, prompt formatting, macro-F1, checkpoint saving, and epoch loop.
- Important CLI knobs:
  - `--data_path`, `--train_data_path`, `--val_data_path`
  - `--llm_path`, `--align_ckpt`, `--output_dir`
  - `--seq_len`, `--image_size`, `--periodicity`, `--input_dim`, `--hidden_dim`
  - `--prefix_tokens`, `--prefix_mode`, `--max_length`, `--prompt_mode`
  - `--batch_size`, `--grad_accum_steps`, `--epochs`, `--lr`, `--warmup_ratio`
  - `--target_mode`, `--soft_target_alpha`, `--soft_label_temperature`
  - `--disable_class_balance`, `--action_balance_weight`, `--overconf_penalty_weight`, `--global_action_weights`

`projection_train/run_projection_online.py`

- Online CityFlow evaluation for Stage 2 checkpoints.
- Loads projection checkpoint and runs closed-loop control.
- Writes `test_result.json`, `test_summary.json`, projection logs, and CityFlow work/model dirs.
- Important pieces:
  - `_dataset_spec(dataset)` maps `hangzhou` to `run_counts=3600`, roadnet/traffic config, and template.
  - `ProjectionPolicyRunner` owns policy initialization, traffic-series loading, snapshot capture, action prediction, reward normalization, and `test()`.
  - CLI knobs mirror the Stage 2 runtime: `--projection_ckpt`, `--llm_path`, `--dataset`, `--traffic_file`, `--run_counts`, `--seq_len`, `--image_size`, `--prefix_tokens`, `--prompt_mode`, precision flags.

`projection_train/prefix_action_policy.py`

- Prefix-action policy module.
- Encodes traffic sequences and prompts, builds tagged action inputs, computes action logits/logprobs, entropy, value predictions, greedy actions, and checkpoint loading.
- Shared by BC and online/PPO paths.

`projection_train/policy.py`

- Older/general VLM policy wrapper.
- Includes action parsing, prompt formatting, input encoding, action generation, action logprob, and action logits.

`projection_train/grouped_dataset.py`

- Main grouped traffic-action JSONL dataset for Stage 2.
- `GroupedTrafficActionDataset` parses candidates, margins, histories, sequences, and action labels.
- `grouped_action_collate` builds BC batches.

`projection_train/jsonl_dataset.py`

- Prompt-series dataset helper.
- Builds traffic sequences from JSONL history records.

`scripts/run_projection_stage.py`

- Stage wrapper/orchestrator.
- Creates common stage directories, writes command scripts, runs train/test with logging, appends markdown reports, selects checkpoint/runtime defaults, and records timing.
- It is the source of `commands/train.sh`, `commands/test.sh`, `logs/`, and `artifacts/` structure under stage roots.

`scripts/select_projection_checkpoint.py`

- Selects checkpoints using online eval JSONs and guardrails.
- Can summarize action distribution from `state_action.json`.
- Supports constraints such as max/min ETWT, NTST, ELWL, NLSL.
- Can promote selected checkpoint to `best.pt`.

`run_sim_expert_action.py`

- Sim-expert data generator and post-processing pipeline.
- Important knobs:
  - `--dataset`, `--traffic_files`
  - `--sim_horizon_s`, `--sim_wait_speed`
  - `--expert_att_weight`, `--expert_aql_weight`, `--expert_awt_weight`, `--expert_spillback_weight`
  - clean/backfill controls for generated JSONL.

`train_projection_ppo_traj.py`

- Stage 3 PPO/trajectory refinement.
- Takes `--stage2_ckpt`, snapshot JSONL, dataset/traffic config, rollout/update parameters, reward shaping weights, guardrails, precision, and distributed settings.

## Hangzhou Real Data And Outputs

Traffic files for Hangzhou are under `data/Hangzhou/4_4/` and include:

- `anon_4_4_hangzhou_real.json`
- `anon_4_4_hangzhou_real_5816.json`
- `anon_4_4_hangzhou_synthetic_24000_60min.json`

Online recorded rollout directories observed under `data/recorded/` include many Hangzhou real and 5816 runs, for example:

- `data/recorded/anon_4_4_hangzhou_real.json_05_20_06_15_47`
- `data/recorded/anon_4_4_hangzhou_real.json_05_23_10_35_54`
- `data/recorded/anon_4_4_hangzhou_real.json_05_24_11_10_27`
- `data/recorded/anon_4_4_hangzhou_real_5816.json_05_21_17_53_28`
- many additional `anon_4_4_hangzhou_real_5816.json_05_22_*` and `05_23_*` runs.

Sim-expert training data for Hangzhou real:

- `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.cleaned.hangzhou_real.jsonl`
- `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl`
- `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl.summary.json`

## Hangzhou Real Stage 2 Experiment Roots

Main Hangzhou real experiment root:

- `records/pipeline_runs/exp_hangzhou_real/`

Important Stage 2 subtrees observed:

- `three_stage_20260520_035801/`
  - Earlier three-stage pipeline root.
  - Contains Stage 1 gated checkpoints and manual Stage 2 runs.
  - Relevant Stage 1 source checkpoint used later:
    - `stage1_gated_ep10_lr7e5_temp010_bs24/best.pt`
  - Manual Stage 2 roots:
    - `stage2_manual_no_balance_from_ep10_best/`
    - `stage2_manual_no_balance_lr7e5_alpha02_temp08_ep14_save1_20260520_065240/`
  - Reports:
    - `manual_stage2_report.md`
    - `manual_stage2_rerun_report.md`
    - `summary.tsv`

- `stage2_norebalance_lr7e5_alpha02_temp08_ep14_20260523_094530/`
  - Full Stage 2 BC train plus online test.
  - Structure:
    - `commands/train.sh`
    - `commands/test.sh`
    - `logs/train.log`
    - `logs/test.log`
    - `checkpoints/`
    - `artifacts/stage_summary.json`
    - `artifacts/train_summary.json`
    - `artifacts/test_summary.json`
    - `artifacts/test_result.json`
    - `test_runs/anon_4_4_hangzhou_real.json_05_23_10_35_54/`
    - `extra_tests/epoch_11/`
    - `extra_tests/last_pt/`
    - `stage2_report.md`

- `stage2_param_sweep_20260523_1715/`
  - Parameter sweep with configs:
    - `A_lr7e5_a02_t08_ep12`
    - `B_lr5e5_a02_t08_ep12`
    - `C_lr7e5_a035_t07_ep12`
  - Contains `driver.log`, per-config `checkpoints/` and `logs/`, and selected online eval:
    - `online_eval_selected_20260524_0415/B_best_epoch11`
    - `online_eval_selected_20260524_0415/C_best_epoch9`
    - `online_eval_selected_20260524_0415/artifacts`

- `stage2_param_sweep_20260524_0407_continue/`
  - Continuation sweep.
  - Contains `Cfull_lr7e5_a035_t07_ep12/` with `checkpoints/` and `logs/`.

- `stage2_wait_travel_sweep_20260524_053909/`
  - Wait/travel-focused Stage 2 sweep.
  - `README.txt` says no automatic online eval in the job itself.
  - Configs:
    - `W1_lr7e5_a045_t060_w095095110130`
    - `W2_lr7e5_a055_t055_w090095115145`
    - `W3_lr7e5_a040_t065_w100100105120`
    - `W4_lr5e5_a045_t060_w095095110130`
  - Later selected online eval:
    - `online_eval_selected_20260524_084149/W1_epoch12`
    - `online_eval_selected_20260524_084149/W2_epoch10`
    - `online_eval_selected_20260524_084149/W2_epoch11`
    - `online_eval_selected_20260524_084149/artifacts`

Stage 3 roots also exist under the same Hangzhou real experiment root, but they are not Stage 2:

- `stage3_from_hz5816_to_hzreal_target_att310_aql56_awt38_20260524_114440/`
- `stage3_from_hz5816_stage3best_epoch4_to_hzreal_eval5_full3600_20260524_130520/`
- `stage3_from_hz5816_stage3best_epoch4_to_hzreal_eval5_full3600_bs16_chunk2_20260524_131619/`
- `stage3_from_hz5816_stage3best_epoch4_to_hzreal_eval5_full3600_bs12_20260524_133910/`
- `stage3_from_hz5816_stage3best_epoch4_to_hzreal_eval5_full3600_bs8_ga2_mon_20260524_140745/`
- `stage3_hangzhou_real_from_hz5816_stage3best_report.md`

## Confirmed Main Stage 2 Run Parameters

Run root:

- `records/pipeline_runs/exp_hangzhou_real/stage2_norebalance_lr7e5_alpha02_temp08_ep14_20260523_094530`

Training command summary:

- Entrypoint: `projection_train/train_projection_bc.py`
- Data: `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl`
- Stage 1 alignment checkpoint: `records/pipeline_runs/exp_hangzhou_real/three_stage_20260520_035801/stage1_gated_ep10_lr7e5_temp010_bs24/best.pt`
- LLM path: `/gemini/pretrain2/qwen-vl-8B/`
- `seq_len=16`
- `image_size=224`
- `prefix_tokens=8`
- `prefix_mode=prefix`
- `max_length=64`
- `prompt_mode=image_only_short`
- `batch_size=8`
- `grad_accum_steps=8`
- `epochs=14`
- `val_ratio=0.1`
- `bf16=true`
- `target_mode=mixed`
- `soft_target_alpha=0.2`
- `soft_label_temperature=0.8`
- `lr=7e-5`
- `warmup_ratio=0.05`
- `disable_class_balance=true`
- `save_every=1`

Training summary for selected `best.pt`:

- Selected epoch: `9`
- Selection metric: `val_macro_f1`
- Best value: `0.5337873697280884`
- Previous best value: `0.47044745087623596`
- Train loss: `0.8218527676359925`
- Train accuracy: `0.6969964664310954`
- Train macro-F1: `0.5348392128944397`
- Validation loss: `0.9871673060626518`
- Validation accuracy: `0.6666666666666666`
- Validation macro-F1: `0.5337873697280884`

Online test command summary:

- Entrypoint: `projection_train/run_projection_online.py`
- Projection checkpoint: `.../checkpoints/best.pt`
- Dataset: `hangzhou`
- Traffic file: `anon_4_4_hangzhou_real.json`
- Run counts: `3600`
- Seed: `2024`
- `num_rounds=1`
- Runtime matches training: `seq_len=16`, `image_size=224`, `prefix_tokens=8`, `prefix_mode=prefix`, `max_length=64`, `prompt_mode=image_only_short`, `bf16=true`.

Online test result for the main Stage 2 run:

- `test_reward_over = -833.0309910076015`
- `test_avg_queue_len_over = 131.03333333333333`
- `test_queuing_vehicle_num_over = 15724.0`
- `test_avg_waiting_time_over = 325.2563762113847`
- `test_avg_travel_time_over = 388.48876969493796`

Timing:

- Stage started: `2026-05-23 09:45:35`
- Stage ended: `2026-05-23 10:58:53`
- Total duration: `01:13:18`
- Train duration: `00:49:30`
- Online test duration: `00:23:14`

## Sweep Notes

`stage2_param_sweep_20260523_1715/driver.log` shows:

- `A_lr7e5_a02_t08_ep12` started `2026-05-23 17:16:32`, completed `17:59:12`.
- `B_lr5e5_a02_t08_ep12` started `17:59:12`, completed `18:42:42`.
- `C_lr7e5_a035_t07_ep12` started `18:42:42`; full completion line was not captured in the bounded snapshot, but later eval directories exist.

`stage2_wait_travel_sweep_20260524_053909/README.txt` states:

- Started: `20260524_053909`.
- Base: `C_best_epoch9` direction, training from Stage 1 alignment checkpoint.
- No automatic online eval inside that job.
- Configs:
  - W1: `lr=7e-5`, `alpha=0.45`, `temp=0.60`, weights `0.95,0.95,1.10,1.30`.
  - W2: `lr=7e-5`, `alpha=0.55`, `temp=0.55`, weights `0.90,0.95,1.15,1.45`.
  - W3: `lr=7e-5`, `alpha=0.40`, `temp=0.65`, weights `1.00,1.00,1.05,1.20`.
  - W4: `lr=5e-5`, `alpha=0.45`, `temp=0.60`, weights `0.95,0.95,1.10,1.30`.

`stage2_wait_travel_sweep_20260524_053909/driver.log` shows all W1-W4 jobs completed successfully:

- W1 completed at `2026-05-24 06:21:50`.
- W2 completed at `2026-05-24 07:03:56`.
- W3 completed at `2026-05-24 07:47:42`.
- W4 completed at `2026-05-24 08:34:10`.

## Local Follow-Up Pointers

For deeper diagnosis later, read these remote artifacts without rerunning experiments:

- Main Stage 2 logs:
  - `records/pipeline_runs/exp_hangzhou_real/stage2_norebalance_lr7e5_alpha02_temp08_ep14_20260523_094530/logs/train.log`
  - `records/pipeline_runs/exp_hangzhou_real/stage2_norebalance_lr7e5_alpha02_temp08_ep14_20260523_094530/logs/test.log`
  - `records/pipeline_runs/exp_hangzhou_real/stage2_norebalance_lr7e5_alpha02_temp08_ep14_20260523_094530/test_runs/anon_4_4_hangzhou_real.json_05_23_10_35_54/projection_test.log`
- Main Stage 2 summaries:
  - `.../artifacts/train_summary.json`
  - `.../artifacts/test_summary.json`
  - `.../artifacts/test_result.json`
- Online rollout details:
  - `.../test_runs/anon_4_4_hangzhou_real.json_05_23_10_35_54/`
  - corresponding `data/recorded/anon_4_4_hangzhou_real.json_05_23_10_35_54/`
- Sweep online eval directories:
  - `records/pipeline_runs/exp_hangzhou_real/stage2_param_sweep_20260523_1715/online_eval_selected_20260524_0415/`
  - `records/pipeline_runs/exp_hangzhou_real/stage2_wait_travel_sweep_20260524_053909/online_eval_selected_20260524_084149/`
- Source training dataset summary:
  - `records/sim_expert_validation_search/anon_4_4_hangzhou_real/best/align_train.stage1_compact_ds.jsonl.summary.json`

