# `run_sim_expert_action.py` 修改说明

本文档只记录对 `run_sim_expert_action.py` 的改动，不展开 `utils/sim_expert_collector.py` 内部打分细节。

## 1. 改动目标

这次对 `run_sim_expert_action.py` 的修改主要有三类：

1. 保留并兼容原先常用的命令行参数
2. 支持同一个 `dataset` 下批量处理多个 `traffic_file`
3. 把采集、清洗、LLM caption 回填串成一个命令

也就是说，脚本现在既能继续跑单个 flow 文件，也能一次性跑多个 flow 文件并汇总训练集输出，还能直接把后处理一步接完。

## 2. 当前支持的 `traffic_file` 用法

### 2.1 单文件

仍然支持单文件调用：

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json
```

### 2.2 多文件

现在支持一次传多个文件：

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file \
    anon_3_4_jinan_real.json \
    anon_3_4_jinan_real_2000.json \
    anon_3_4_jinan_real_2500.json
```

也支持逗号分隔写法：

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json,anon_3_4_jinan_real_2000.json
```

### 2.3 全量文件

如果希望直接跑该 `dataset` 下脚本内置的全部 flow 文件，可以使用：

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file all
```

## 3. 参数兼容性

为了兼容原有调用方式，下面这些入口参数保留可用：

- `--llm_model`
- `--llm_path`
- `--new_max_tokens`
- `--tensor_parallel_size`
- `--clip_caption_mode`
- `--clip_caption_language`
- `--clip_caption_max_tokens`
- `--clip_caption_temperature`
- `--clip_caption_top_p`

同时保留了当前 SimExpert 采集相关参数：

- `--sim_wait_speed`
- `--sim_horizon_s`
- `--expert_att_weight`
- `--expert_aql_weight`
- `--expert_awt_weight`
- `--expert_spillback_weight`

同时新增了数据集后处理参数，用来把“采集 -> 清洗 -> LLM caption 回填”串成一个命令：

- `--clean_seq_len`
- `--clean_output`
- `--backfill_output`
- `--skip_clean`
- `--skip_llm_backfill`
- `--backfill_llm_path`
- `--backfill_batch_size`

## 4. 脚本内部结构变化

原先入口基本是“解析参数后直接跑单个 `TRAFFIC_FILE`”。现在拆成了几步：

### 4.1 数据集信息解析

新增 `_get_dataset_spec()`，统一返回：

- `count`
- `road_net`
- `traffic_file_list`
- `template`

这样后面单文件和多文件都共用同一份 dataset 配置。

### 4.2 文件列表解析

新增：

- `_normalize_requested_traffic_files()`
- `_resolve_traffic_files()`

职责是把用户传入的：

- 单个文件
- 多个文件
- 逗号分隔字符串
- `all`

统一解析成一个最终要运行的 `traffic_file` 列表。

### 4.3 单文件执行逻辑抽离

新增 `_run_single_traffic_file()`，把“构建 config / 构建路径 / 初始化 `SimExpertCollector` / 调用 `train_test()`”收口到一个函数里。

这样 `main()` 在多文件模式下只需要循环调用这个函数即可。

### 4.4 输出路径拆分

新增：

- `_build_single_run_paths()`
- `_build_batch_run_paths()`

区别是：

- 单文件模式仍然按单次运行目录落盘
- 多文件模式会放到同一个 batch 根目录下，但每个 flow 文件仍然有独立子目录，避免互相覆盖

## 5. 多文件模式的输出

### 5.1 每个文件各自输出

批量模式下，每个 `traffic_file` 仍然会生成自己独立的：

- `PATH_TO_MODEL`
- `PATH_TO_WORK_DIRECTORY`
- 对应的训练集 jsonl

### 5.2 Batch Manifest

新增 `_write_batch_manifest()`，会在批次根目录下写：

```text
records/<memo>/batch_<timestamp>/batch_manifest.json
```

里面会记录每个 flow 对应的：

- `traffic_file`
- `run_counts`
- `path_to_model`
- `path_to_work_directory`
- `traffic_series_output_path`

### 5.3 合并后的总训练集

新增 `_merge_batch_training_sets()`，会自动把同一批次下各个 flow 产出的 jsonl 合并成一个总文件：

```text
data/recorded/batch_<timestamp>/traffic_prompt_series_2s.jsonl
```

这样多文件跑完之后，可以直接拿这个 merged jsonl 去做后续清洗或训练，而不需要手工再拼接。

## 6. 新增的一体化后处理

现在 `run_sim_expert_action.py` 在采集结束后，会默认继续执行两步：

1. 调用 `clean_align_jsonl` 逻辑生成 cleaned jsonl
2. 调用 `backfill_llm_captions` 逻辑生成带 LLM caption 的最终 jsonl

也就是说，之前这三条命令：

```bash
python run_sim_expert_action.py --dataset jinan --traffic_file anon_3_4_jinan_real.json

python utils/clean_align_jsonl.py --inputs ... --output data/recorded/align_train.cleaned.jsonl --seq_len 16

python utils/backfill_llm_captions.py \
  --input data/recorded/align_train.cleaned.jsonl \
  --output data/recorded/align_train.cleaned.llm_caption.jsonl \
  --llm_path /home/data/model/gemma-3-27B/
```

现在可以直接收敛成一条命令。

### 6.1 后处理输入来源

- 单文件模式：后处理输入就是该次运行生成的 `traffic_prompt_series_2s.jsonl`
- 多文件模式：后处理输入就是 batch merge 后的总 jsonl

### 6.2 默认输出命名

如果不显式传输出路径：

- 清洗输出默认为 `<raw>.cleaned.jsonl`
- 回填输出默认为 `<clean_or_raw>.llm_caption.jsonl`

例如单文件采集产物如果是：

```text
data/recorded/anon_3_4_jinan_real_03_25_12_00_00/traffic_prompt_series_2s.jsonl
```

那么默认会继续生成：

```text
data/recorded/anon_3_4_jinan_real_03_25_12_00_00/traffic_prompt_series_2s.cleaned.jsonl
data/recorded/anon_3_4_jinan_real_03_25_12_00_00/traffic_prompt_series_2s.cleaned.llm_caption.jsonl
```

### 6.3 可选跳过

如果你只想保留部分流程：

- `--skip_clean`：跳过清洗，直接对原始 jsonl 做 caption 回填
- `--skip_llm_backfill`：只采集并清洗，不做 caption 回填

## 7. 行为说明

### 7.1 单文件模式

如果 `--traffic_file` 最终只解析出一个文件：

- 只跑这个文件
- 不写 batch manifest
- 不做 merged jsonl

### 7.2 多文件模式

如果 `--traffic_file` 最终解析出多个文件：

- 依次顺序执行每个文件
- 每个文件有独立输出目录
- 自动生成 batch manifest
- 自动生成 merged jsonl

## 8. 推荐命令

### 8.1 单文件一体化命令

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --llm_model gemma-3-27B \
  --llm_path /home/data/model/gemma-3-27B/ \
  --clean_seq_len 16
```

### 8.2 指定和原来一致的输出文件名

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file anon_3_4_jinan_real.json \
  --llm_model gemma-3-27B \
  --llm_path /home/data/model/gemma-3-27B/ \
  --clean_seq_len 16 \
  --clean_output data/recorded/align_train.cleaned.jsonl \
  --backfill_output data/recorded/align_train.cleaned.llm_caption.jsonl
```

### 8.3 多文件

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file \
    anon_3_4_jinan_real.json \
    anon_3_4_jinan_real_2000.json \
    anon_3_4_jinan_real_2500.json \
  --llm_model gemma-3-27B \
  --llm_path /home/data/model/gemma-3-27B/ \
  --clean_seq_len 16
```

### 8.4 全量

```bash
python run_sim_expert_action.py \
  --dataset jinan \
  --traffic_file all \
  --llm_model gemma-3-27B \
  --llm_path /home/data/model/gemma-3-27B/ \
  --clean_seq_len 16
```

### 8.5 不生成快照

默认会生成 CityFlow snapshot 文件，并基于 snapshot 做候选动作短时仿真。如果只想生成基础序列、不写 snapshot 文件，可以加：

```bash
python run_sim_expert_action.py \
  --dataset newyork_28x7 \
  --traffic_file anon_28_7_newyork_real_double.json \
  --llm_model gemma-3-27B \
  --llm_path /home/data/model/gemma-3-27B/ \
  --tensor_parallel_size 2 \
  --backfill_batch_size 4 \
  --clean_seq_len 16 \
  --clean_output data/recorded/align_train.cleaned_newyork.jsonl \
  --backfill_output data/recorded/align_train.cleaned.llm_caption_newyork.jsonl \
  --skip_snapshot_generation
```

关闭快照后，输出 JSONL 中的 `snapshot_path`、`candidate_action_waits` 和 `candidate_action_metrics` 会为空，`expert_action` 会退回当前基础动作。

## 9. 当前结论

`run_sim_expert_action.py` 现在已经不再是“只能处理单个 flow 文件”的入口，而是一个支持：

- 单文件采集
- 多文件批量采集
- dataset 内全量采集
- 自动批次汇总
- 自动清洗
- 自动 LLM caption 回填

的训练集生成入口。
