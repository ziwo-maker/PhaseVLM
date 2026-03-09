from __future__ import annotations

import random
from typing import Dict, Generator, List, Sequence

import torch


class TrajectoryRolloutBuffer:
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clear()

    def clear(self) -> None:
        self.sequences: List[torch.Tensor] = []
        self.prompts: List[List[str]] = []
        self.actions: List[torch.Tensor] = []
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.advantages: torch.Tensor | None = None
        self.returns: torch.Tensor | None = None

    def add(
        self,
        sequences: torch.Tensor,
        prompts: Sequence[str],
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        values: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        self.sequences.append(sequences.detach().cpu())
        self.prompts.append([str(item) for item in prompts])
        self.actions.append(actions.detach().cpu())
        self.log_probs.append(log_probs.detach().cpu())
        self.values.append(values.detach().cpu())
        self.rewards.append(rewards.detach().cpu())
        self.dones.append(dones.detach().cpu())

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def num_steps(self) -> int:
        return len(self.sequences)

    @property
    def num_envs(self) -> int:
        if not self.actions:
            return 0
        return int(self.actions[0].shape[0])

    @property
    def num_transitions(self) -> int:
        return self.num_steps * self.num_envs

    def compute_returns_and_advantages(self, last_values: torch.Tensor) -> None:
        if not self.sequences:
            raise RuntimeError("Rollout buffer is empty.")
        last_values = last_values.detach().cpu().float()
        rewards = torch.stack(self.rewards, dim=0).float()
        values = torch.stack(self.values, dim=0).float()
        dones = torch.stack(self.dones, dim=0).float()

        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(last_values)
        next_values = last_values
        for step in reversed(range(rewards.shape[0])):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_values * next_non_terminal - values[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[step] = gae
            next_values = values[step]

        self.advantages = advantages
        self.returns = advantages + values

    def flattened(self) -> Dict[str, torch.Tensor | List[str]]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_returns_and_advantages() before flattening the buffer.")

        sequences = torch.cat(self.sequences, dim=0)
        actions = torch.cat(self.actions, dim=0).long()
        log_probs = torch.cat(self.log_probs, dim=0).float()
        values = torch.cat(self.values, dim=0).float()
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

        prompts: List[str] = []
        for items in self.prompts:
            prompts.extend(items)

        return {
            "sequences": sequences,
            "prompts": prompts,
            "actions": actions,
            "old_log_probs": log_probs,
            "old_values": values,
            "advantages": advantages,
            "returns": returns,
        }

    def mini_batches(
        self,
        mini_batch_size: int,
        shuffle: bool = True,
    ) -> Generator[Dict[str, torch.Tensor | List[str]], None, None]:
        flat = self.flattened()
        total = int(flat["actions"].shape[0])
        indices = list(range(total))
        if shuffle:
            random.shuffle(indices)
        batch_size = max(1, int(mini_batch_size))
        for start in range(0, total, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield {
                "sequences": flat["sequences"][batch_idx],
                "prompts": [flat["prompts"][idx] for idx in batch_idx],
                "actions": flat["actions"][batch_idx],
                "old_log_probs": flat["old_log_probs"][batch_idx],
                "old_values": flat["old_values"][batch_idx],
                "advantages": flat["advantages"][batch_idx],
                "returns": flat["returns"][batch_idx],
            }
