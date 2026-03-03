# Align ckpt 兼容性说明

## 简要思考过程
- 报错来自 `vision_projector` 的 `proj.weight` 维度不一致，导致 `load_state_dict` 失败。
- 该维度由 `prefix_tokens` 和 Qwen3-VL 的 `text_dim/vision_dim` 决定。
- 即便 LLM 本身不参与训练，只要 Qwen3-VL 的隐藏维度不同，投影矩阵也无法兼容。

## 注意事项
- 仅 `prefix_tokens` 不一致时，已支持自动截断/填充加载（复用已有权重）。
- 若 `text_dim` 或 `vision_dim` 不一致，当前策略是直接报错，必须重跑 `train_align.py`。
- `llm_path` 可以不同，但实际隐藏维度必须一致（同系列不同规模通常不一致）。
- 仅在 `prefix_mode=prefix` 时支持自动适配；`prefix_mode=single` 维度不一致会直接报错。

## 对齐训练范围说明
- BC/align 的训练主体确实是 `ts_image_adapter`（时序转图像投影）。
- 但 `vision_projector` 将视觉特征映射到文本空间，维度必须与 Qwen3-VL 的隐藏维度匹配，否则无法正确对齐。
