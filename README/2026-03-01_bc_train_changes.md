# 2026-03-01 BC 训练接入 FSDP/DeepSpeed 的修改思路

本文记录我为 `projection_train/train_projection_bc.py` 与相关组件接入 **FSDP / DeepSpeed ZeRO‑3** 的设计思路与改动点，便于后续回溯与二次调整。

## 背景与目标

BC 训练的 OOM 主要来自 **8B 模型权重常驻 + 反向传播激活保留**。即使冻结 LLM，梯度仍需穿过 LLM 以更新 `ts_projector`，导致显存压力高。目标是：

- 在 2×4090 上把 LLM 权重 **分片**，降低单卡显存占用。
- 不破坏现有训练逻辑（BC 只训练 `ts_projector`，LLM 冻结）。
- 保持命令行易用，允许在 FSDP 与 DeepSpeed 之间切换。

## 总体设计

1. **FSDP（PyTorch 内置）**：用于快速分片 LLM 权重；与原有 DDP 共存。
2. **DeepSpeed ZeRO‑3**：用于更强的参数/优化器状态分片；在训练主循环中接管 backward/step。
3. **互斥约束**：DeepSpeed 与 FSDP 不同时启用（避免冲突）。

---

## FSDP 方案

### 修改点

- **`models.py`**
  - 给 `Qwen3VLBackbone` 增加 `fsdp_enabled` / `fsdp_sharding_strategy` 参数。
  - 新增 `_maybe_wrap_fsdp`：在 `torch.distributed` 初始化后，对 transformer block 自动 wrap；若未识别 block，则按参数量阈值包裹。
  - 仅在 **分布式 + CUDA** 条件满足时启用 FSDP。
  - 兼容 FSDP 包裹后的 `get_input_embeddings()` 与 `embedding_dim` 取值。

- **`projection_train/train_projection_bc.py`**
  - 新增 CLI 参数 `--llm_fsdp` / `--fsdp_sharding_strategy`。
  - 构造 `BCProjectionModel` 时传入 FSDP 配置。
  - 训练日志打印 FSDP 真实启用状态。

### 原因与取舍

- **自动 wrap 策略** 更稳健：对 Qwen 系模型结构变化更鲁棒。
- 仅在 LLM 模块内部 FSDP，不干预 `ts_projector`，避免影响训练参数同步。

---

## DeepSpeed ZeRO‑3 方案

### 修改点

- **`projection_train/train_projection_bc.py`**
  - 新增 CLI 参数 `--deepspeed_config`。
  - 引入 `deepspeed.initialize`，让 DeepSpeed 接管优化器和训练步骤。
  - 训练循环：使用 `engine.backward(loss)` + `engine.step()`。
  - 添加 `_load_deepspeed_config()`：动态写入 `train_micro_batch_size_per_gpu` / `train_batch_size`。
  - 添加 `_save_deepspeed_checkpoint()`：优先保存 DS shard checkpoint，同时单独保存 `ts_projector` 权重（便于推理侧加载）。
  - 禁止与 FSDP 同时启用。

- **`configs/deepspeed_zero3_bf16.json`**
  - 提供 ZeRO‑3 默认配置模板，bf16 + 常见 bucket 参数。

### 设计意图

- DeepSpeed **完全接管分片与优化器状态**，适合显存极限场景。
- BC 训练只更新 `ts_projector`，但通过 ZeRO‑3 可以显著降低 LLM 权重占用。

---

## 行为与兼容性说明

- **FSDP 与 DeepSpeed 互斥**：CLI 明确约束避免混用。
- **模型冻结**逻辑保持不变（LLM 冻结，`ts_projector` 可训练）。
- **保存逻辑**：DeepSpeed 会额外输出分片 checkpoint，且主进程写一个便于复用的 `bc_epoch_*.pt`。

---

## 启动建议（简要）

### FSDP

- 继续使用 `torchrun --nproc_per_node=2`
- 追加 `--llm_fsdp on` 即可

### DeepSpeed

- 使用 `--deepspeed_config configs/deepspeed_zero3_bf16.json`
- 同时 `--llm_fsdp off`

---

## 后续可增强点（未实现）

- 使用 `HfDeepSpeedConfig` 在 `from_pretrained` 阶段直接 ZeRO‑3 分片加载，进一步降低初始化峰值。
- 增加 `gradient_checkpointing` 选项来继续压缩激活占用。
- 为 `ts_projector` 增加梯度累积支持，保持更大等效 batch。

