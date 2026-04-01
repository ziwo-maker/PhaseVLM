# train_align.py 修改说明

## 本次改动

- 数据集加载阶段关闭了 split 前乱序
  - `TrafficClipJsonlDataset(..., shuffle_samples=False)`
- 验证集 loader 去掉了 `RandomSampler`
  - 验证阶段保持稳定顺序，不再做额外随机抽样。
- 运行上下文新增记录项
  - `contrastive_mode`
  - `shuffle_before_split`
  - `contrastive_global_batch_size`
- 默认 `warmup_ratio` 从 `0.5` 调整为 `0.1`
- backward 路径加入 `ddp_grad_scale`
  - 用于补偿 DDP 默认梯度平均。
- 梯度累积逻辑补了 epoch 尾部 flush
  - 当 `len(train_loader)` 不能整除 `grad_accum_steps` 时，最后一段梯度不再丢失。
- DDP epoch 边界补了双 barrier
  - 防止 `rank 0` 做验证或存 checkpoint 时，其他 rank 提前进入下一轮训练并卡死在 `ALLREDUCE`。

## 修改目的

- 让 Stage 1 的 train/val split 更接近真实 holdout，而不是先乱序再拆分。
- 让多卡对比学习和 DDP 梯度平均行为匹配。
- 缩短 projector-only 训练中不必要的长 warmup。
- 修掉 epoch 最后一段样本因累积步数未对齐而未参与更新的问题。

## 结果

- Stage 1 现在默认是更稳的训练设定。
- 多卡训练时，日志中会明确写出当前使用的是 `distributed_global_batch` 对比模式。
- 训练命令即使保留 `grad_accum_steps`，epoch 尾部数据也能被正确消费。

## DDP 注意事项

- 这次超时问题的根因不是对比学习里的 `all_gather`，而是 epoch 结束后只有 `rank 0` 执行了验证和 checkpoint，其他 rank 没有等待，直接进入了下一轮 `backward`。
- 在 DDP 下，只要某段逻辑不是所有 rank 一起执行，就必须显式审查它会不会打乱 collective 顺序。
- 特别是 epoch 边界的 `eval`、`checkpoint`、落盘、长日志汇总、可视化导出这类逻辑，如果只让 `rank 0` 执行，前后都要加 `dist.barrier()`。
- 如果只是 `rank 0` 做验证，优先走 `model.module`，不要直接拿 DDP wrapper 做只在单 rank 上运行的 forward。
