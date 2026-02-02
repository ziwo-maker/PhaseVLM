from dataclasses import dataclass
from typing import Iterable, List, Tuple

import torch


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    epochs: int = 4
    mini_batch_size: int = 32


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # rewards, values, dones: [T, B]
    t_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    next_value = torch.zeros(rewards.shape[1], device=rewards.device)
    gae = torch.zeros(rewards.shape[1], device=rewards.device)

    for t in reversed(range(t_steps)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


def iter_minibatches(indices: List[int], mini_batch_size: int) -> Iterable[List[int]]:
    for start in range(0, len(indices), mini_batch_size):
        yield indices[start : start + mini_batch_size]
