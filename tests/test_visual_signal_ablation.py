import pytest
import torch

from projection_train.prefix_action_policy import apply_visual_image_ablation


def test_full_and_zero_prefix_leave_images_unchanged():
    images = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)

    assert apply_visual_image_ablation(images, "full") is images
    assert apply_visual_image_ablation(images, "zero_prefix") is images


def test_constant_image_removes_all_image_state():
    images = torch.randn(3, 3, 4, 4)

    ablated = apply_visual_image_ablation(images, "constant_image")

    assert torch.equal(ablated, torch.zeros_like(images))
    assert ablated.shape == images.shape


def test_shuffled_image_uses_a_derangement_for_multi_sample_batches():
    images = torch.arange(36, dtype=torch.float32).reshape(3, 3, 2, 2)

    ablated = apply_visual_image_ablation(images, "shuffled_image")

    assert torch.equal(ablated, torch.roll(images, shifts=1, dims=0))
    assert all(not torch.equal(ablated[idx], images[idx]) for idx in range(images.shape[0]))


def test_shuffled_image_rejects_single_sample_batches():
    with pytest.raises(ValueError, match="at least two samples"):
        apply_visual_image_ablation(torch.zeros(1, 3, 2, 2), "shuffled_image")


def test_unknown_visual_ablation_is_rejected():
    with pytest.raises(ValueError, match="Unsupported visual ablation"):
        apply_visual_image_ablation(torch.zeros(2, 3, 2, 2), "unknown")
