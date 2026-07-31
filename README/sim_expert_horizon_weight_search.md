# SimExpert horizon 消融与权重验证搜索

本文档说明如何对 `run_sim_expert_action.py` 做两类验证实验：

1. 对 `--sim_horizon_s` 做 `30/60/90` 秒消融
2. 在验证集上搜索 `expert_att/aql/awt/spillback_weight`

核心原则是：评估不只看候选动作内部的 `candidate_action_waits` 或 composite score，而是看每次完整仿真结束后输出的完整 `ATT / AQL / AWT`。

## 1. 脚本入口

新增脚本：

```bash
python scripts/run_sim_expert_horizon_weight_search.py --help
```

它不会重写 SimExpert 逻辑，而是循环调用：

```bash
python run_sim_expert_action.py ...
```

每个 `(traffic_file, sim_horizon_s, expert weights)` 都是一次独立运行。脚本负责：

- 生成实验矩阵
- 捕获每次运行日志
- 解析 `sim_eval_metrics_by_file`
- 写出 `run_results.csv`
- 按验证集均值写出 `summary.csv`
- 写出当前验证目标下的 `best_config.json`
- 将最佳候选的数据物化到 `best/`，并默认删除其他候选目录

默认情况下，该脚本会给 `run_sim_expert_action.py` 传入：

```text
--skip_llm_init --skip_clean --skip_llm_backfill
```

因此参数搜索阶段只运行 CityFlow + SimExpert 候选仿真，不启动大模型，也不做描述回填。JSONL 中的 `clip_caption` 会退回规则摘要，等最佳参数确定后再单独生成正式训练集和 LLM caption。

## 2. 推荐验证命令

先用一个验证 flow 做小规模搜索：

```bash
PY=/mnt/data/sysu/conda_envs/canav5090/bin/python3.10
DATASET_ID=anon_3_4_jinan_real_2000
ROOT=records/sim_expert_validation_search/${DATASET_ID}
mkdir -p "${ROOT}"

${PY} scripts/run_sim_expert_horizon_weight_search.py \
  --dataset jinan \
  --traffic_files ${DATASET_ID}.json \
  --horizons 30,60,90 \
  --llm_model gemma-3-27B \
  --llm_path /home/data/model/gemma-3-27B/ \
  --tensor_parallel_size 1 \
  --run_id ${DATASET_ID} \
  --output_dir "${ROOT}" \
  --python ${PY} \
  --cuda_visible_devices 0 \
  | tee "${ROOT}/search.log"
```

如果只传入一个 `traffic_file`，省略 `--run_id` 时默认会使用文件 stem 作为 `run_id`，例如
`anon_3_4_jinan_real_2000.json` 对应 `anon_3_4_jinan_real_2000`。推荐显式传入，保证命令可读。

默认权重集合是一个小网格：

```text
(1.0, 1.0, 1.0, 0.2)  # 当前默认
(2.0, 1.0, 1.0, 0.2)  # 更重 ATT
(1.0, 2.0, 1.0, 0.2)  # 更重 AQL
(1.0, 1.0, 2.0, 0.2)  # 更重 AWT
(1.0, 1.0, 1.0, 0.0)  # 不惩罚 spillback
(1.0, 1.0, 1.0, 0.5)  # 更重 spillback
```

因此默认会运行：

```text
3 个 horizon * 6 组权重 * traffic_file 数量
```

## 3. 自定义网格

如果要做完整笛卡尔网格，可以传每个权重的候选列表：

```bash
python scripts/run_sim_expert_horizon_weight_search.py \
  --dataset hangzhou \
  --traffic_files anon_4_4_hangzhou_real_5816.json \
  --horizons 30,60,90 \
  --att_weights 0.5,1.0,2.0 \
  --aql_weights 0.5,1.0,2.0 \
  --awt_weights 0.5,1.0,2.0 \
  --spillback_weights 0.0,0.2,0.5 \
  --llm_path /home/data/model/gemma-3-27B/
```

这会产生 `3 * 3 * 3 * 3 * 3 = 243` 次运行，成本较高。建议先用默认小网格筛方向，再扩大局部搜索。

也可以显式给定若干组权重：

```bash
python scripts/run_sim_expert_horizon_weight_search.py \
  --dataset hangzhou \
  --traffic_files anon_4_4_hangzhou_real_5816.json \
  --horizons 30,60,90 \
  --weight_sets "1,1,1,0.2;1,2,1,0.2;1,1,2,0.5" \
  --llm_path /home/data/model/gemma-3-27B/
```

## 4. 多验证文件

验证集可以包含多个 `traffic_file`：

```bash
python scripts/run_sim_expert_horizon_weight_search.py \
  --dataset hangzhou \
  --traffic_files \
    anon_4_4_hangzhou_real.json \
    anon_4_4_hangzhou_real_5816.json \
    anon_4_4_hangzhou_synthetic_24000_60min.json \
  --horizons 30,60,90 \
  --llm_path /home/data/model/gemma-3-27B/
```

`summary.csv` 会按同一组 `(horizon, weights)` 聚合多个验证文件的均值。

## 5. 输出目录

默认输出在：

```text
records/sim_expert_validation_search/<run_id>/
```

关键文件：

- `manifest.json`：实验矩阵和全局配置
- `run_results.csv`：每个 traffic file 的完整 ATT/AQL/AWT
- `summary.csv`：按配置聚合后的均值和排序
- `best_config.json`：当前验证目标下的最佳配置
- `best/traffic_prompt_series_2s.jsonl`：最佳参数对应的 raw JSONL
- `best/snapshots/`：最佳 raw JSONL 对应的 CityFlow 快照
- `best/best.log`：最佳候选的运行日志

中间候选会临时写到：

```text
records/sim_expert_validation_search/<run_id>/.candidates/
```

搜索完成后默认删除 `.candidates/` 和非最佳候选日志，只保留 `best/`。如果需要调试，可以加：

```bash
--keep_candidate_artifacts --keep_candidate_logs
```

## 6. 排序目标

`summary.csv` 的 `objective` 不是候选动作 score，而是完整运行指标的归一化加权和：

```text
objective = w_att * norm(mean_ATT)
          + w_aql * norm(mean_AQL)
          + w_awt * norm(mean_AWT)
```

默认三项权重都是 `1.0`，数值越小越好。

如果论文主指标更重视 `ATT`，可以这样改：

```bash
python scripts/run_sim_expert_horizon_weight_search.py \
  --dataset hangzhou \
  --traffic_files anon_4_4_hangzhou_real_5816.json \
  --selection_att_weight 2.0 \
  --selection_aql_weight 1.0 \
  --selection_awt_weight 1.0 \
  --llm_path /home/data/model/gemma-3-27B/
```

## 7. 只检查命令不运行

没有运行环境时，可以先 dry-run 检查矩阵和命令：

```bash
python scripts/run_sim_expert_horizon_weight_search.py \
  --dry_run \
  --dataset hangzhou \
  --traffic_files anon_4_4_hangzhou_real_5816.json \
  --horizons 30,60,90 \
  --weight_sets "1,1,1,0.2;1,2,1,0.2" \
  --llm_path /home/data/model/gemma-3-27B/
```

dry-run 仍会生成：

- `manifest.json`
- `logs/*.log`
- `run_results.csv`
- `summary.csv`
- `best_config.json`

但不会启动 CityFlow / vLLM。

如果确实要在搜索阶段也加载大模型，可以显式加：

```bash
--run_llm_init
```

一般不建议这样做，因为本脚本的目标是确定 `sim_horizon_s` 和 expert 权重，完整 `ATT/AQL/AWT` 与描述回填无关。
