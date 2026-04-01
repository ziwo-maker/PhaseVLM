LLMLight: Large Language Models as Traffic Signal Control Agents

## 开发注意事项

- DDP/NCCL 训练中，不能只让 `rank 0` 在 epoch 边界执行 `eval`、`checkpoint`、长耗时 I/O 或其他独占逻辑，而不和其他 rank 做同步。
- 否则其他 rank 可能提前进入下一轮 `backward`，卡死在首个梯度 `all_reduce`，最终表现为 NCCL collective timeout。
- 规则：凡是只在部分 rank 执行的 epoch 级逻辑，前后都要显式做 `dist.barrier()`，或者改成所有 rank 都参与的实现。
- 如果只是 `rank 0` 做验证，优先使用 `model.module` 这类非 DDP wrapper 对象，避免验证阶段误触发额外 collective。
- 这次 `train_align.py` 的修复背景见 `README/train_align.md`。
