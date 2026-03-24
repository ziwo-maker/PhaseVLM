# losses.py 修改说明

## 本次改动

- 新增 `_resolve_logit_scale(...)`
  - 统一处理 `temperature` 和 `logit_scale` 的换算与 clamp 逻辑。
- 新增 `_gather_for_distributed_contrastive(...)`
  - 在 DDP 场景下对 `image_embeds` 和 `text_embeds` 做 `all_gather`。
  - 保留当前 rank 的本地切片参与 autograd，其他 rank 的特征使用 detach 后的副本。
- 更新 `clip_contrastive_loss(...)`
  - 单卡时保持原来的对称 InfoNCE。
  - 多卡时改为对全局 batch 做对比学习，不再只使用单卡负样本。

## 修改目的

- 解决 Stage 1 在多卡训练时负样本池仍然只来自单卡 batch 的问题。
- 让 `torchrun --nproc_per_node=N` 真正带来更大的对比学习负样本规模。

## 结果

- DDP 下的 Stage 1 对齐训练现在会使用全局 gather 后的 embedding 计算 loss。
- loss 数值本身仍然保持可读，梯度缩放补偿放在 `train_align.py` 的 backward 路径中处理。
