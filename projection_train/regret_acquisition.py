#!/usr/bin/env python3
"""Regret-Aware Active Counterfactual Query — Idea 1 from IDEA_REPORT.md.

Implements the acquisition functions for selecting which target-network states
to send to SimExpert for relabeling during cross-network adaptation.

Acquisition methods:
  1. random           — uniform random baseline
  2. entropy          — predictive entropy of policy distribution
  3. ood              — distance to source training distribution (feature-space)
  4. core_set         — k-center greedy subset selection in feature space
  5. full_ensemble    — ensemble disagreement (variance of action probabilities)
  6. regret_aware     — expected regret × epistemic uncertainty + MMR diversity

Reference: IDEA_REPORT.md § Idea 1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ── Base class ────────────────────────────────────────────────────────────


class AcquisitionFunction(ABC):
    """Base class for state acquisition scoring."""

    @abstractmethod
    def score_states(
        self,
        logits: np.ndarray,          # [N, num_actions] — policy logits
        features: Optional[np.ndarray] = None,  # [N, feature_dim] — for core-set/OOD
        costs: Optional[np.ndarray] = None,     # [N, num_actions] — SimExpert costs
    ) -> np.ndarray:
        """Return acquisition scores [N] (higher = more valuable to label)."""
        ...

    def select(
        self,
        logits: np.ndarray,
        K: int,
        features: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select top-K states by acquisition score. Returns (indices, scores)."""
        scores = self.score_states(logits, features=features, costs=costs)
        K = min(K, len(scores))
        top_indices = np.argpartition(-scores, K - 1)[:K]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return top_indices, scores[top_indices]


# ── Method 1: Random ──────────────────────────────────────────────────────


class RandomAcquisition(AcquisitionFunction):
    """Uniform random acquisition — baseline."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def score_states(self, logits, features=None, costs=None):
        return self.rng.uniform(0, 1, size=len(logits))


# ── Method 2: Entropy ─────────────────────────────────────────────────────


class EntropyAcquisition(AcquisitionFunction):
    """Predictive entropy of the policy distribution.

    H(p) = -Σ p(a) log p(a). Higher entropy = more uncertain → select.
    """

    def score_states(self, logits, features=None, costs=None):
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        return entropy


# ── Method 3: OOD ─────────────────────────────────────────────────────────


class OODAcquisition(AcquisitionFunction):
    """Distance to source training distribution in feature space.

    Score = negative of minimum distance to source centroid (or source points).
    States far from source distribution → higher OOD score → select.
    """

    def __init__(self, source_features: Optional[np.ndarray] = None):
        """
        Args:
            source_features: [M, D] feature vectors from source training set.
                             If provided, compute distance to nearest source point.
                             Otherwise, compute distance to global centroid.
        """
        self.source_features = source_features
        if source_features is not None and len(source_features) > 0:
            self.source_centroid = source_features.mean(axis=0, keepdims=True)
        else:
            self.source_centroid = None

    def score_states(self, logits, features=None, costs=None):
        if features is None:
            return np.zeros(len(logits))

        if self.source_features is not None and len(self.source_features) > 0:
            # Distance to nearest source point (approximate with centroid for speed)
            # Use batch computation for efficiency
            scores = []
            batch_size = 256
            for start in range(0, len(features), batch_size):
                batch = features[start:start + batch_size]
                dists = np.linalg.norm(
                    batch[:, None, :] - self.source_centroid[None, :, :],
                    axis=-1,
                )
                scores.append(dists.ravel())
            return np.concatenate(scores)
        elif self.source_centroid is not None:
            return np.linalg.norm(features - self.source_centroid, axis=-1)
        else:
            return np.zeros(len(logits))


# ── Method 4: Core-Set ────────────────────────────────────────────────────


class CoreSetAcquisition(AcquisitionFunction):
    """K-Center greedy selection in feature space.

    Iteratively selects the point farthest from the already-selected set.
    This is a diversity-based acquisition — ensures coverage of the state space.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def score_states(self, logits, features=None, costs=None):
        # Core-set doesn't use per-state scores — it's iterative
        # Return uniform for scoring, but override select()
        return self.rng.uniform(0, 1, size=len(logits))

    def select(
        self,
        logits: np.ndarray,
        K: int,
        features: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """K-center greedy selection."""
        if features is None:
            # Fall back to random
            return RandomAcquisition().select(logits, K)

        N = len(features)
        K = min(K, N)

        # Initialize with random point
        selected = [self.rng.randint(0, N)]
        min_dists = np.full(N, np.inf)

        for _ in range(K - 1):
            # Update distances to the closest selected point
            last_feat = features[selected[-1]]
            new_dists = np.linalg.norm(features - last_feat, axis=-1)
            min_dists = np.minimum(min_dists, new_dists)
            # Select farthest
            next_idx = int(np.argmax(min_dists))
            selected.append(next_idx)

        selected = np.array(selected, dtype=np.int64)
        return selected, np.zeros(K)


# ── Method 5: Full Ensemble ───────────────────────────────────────────────


class EnsembleDisagreementAcquisition(AcquisitionFunction):
    """Variance of action probabilities across an ensemble of models.

    Acquires states where ensemble members disagree most.
    """

    def __init__(self, ensemble_logits: Optional[List[np.ndarray]] = None):
        """
        Args:
            ensemble_logits: list of [N, num_actions] from each ensemble member.
                             If None, compute from a single deterministic model (variance=0).
        """
        self.ensemble_logits = ensemble_logits

    def score_states(self, logits, features=None, costs=None):
        if self.ensemble_logits is None or len(self.ensemble_logits) < 2:
            # No ensemble available — fall back to entropy
            return EntropyAcquisition().score_states(logits)

        # Stack ensemble probabilities
        all_probs = []
        for member_logits in self.ensemble_logits:
            probs = F.softmax(torch.from_numpy(member_logits), dim=-1).numpy()
            all_probs.append(probs)

        all_probs = np.stack(all_probs, axis=0)  # [M, N, num_actions]
        # Variance of predicted probability for the argmax action
        mean_probs = all_probs.mean(axis=0)
        argmax_actions = np.argmax(mean_probs, axis=-1)
        prob_variance = all_probs[:, np.arange(len(argmax_actions)), argmax_actions].var(axis=0)
        return prob_variance


# ── Method 6: Regret-Aware ────────────────────────────────────────────────


class RegretAwareAcquisition(AcquisitionFunction):
    """Regret-aware acquisition: expected regret × epistemic uncertainty + MMR diversity.

    Steps:
      1. Compute per-action expected costs from SimExpert (if available)
      2. Compute expected regret of policy argmax vs SimExpert-best
      3. Estimate epistemic uncertainty via MC Dropout ensemble
      4. Compute MMR (Maximal Marginal Relevance) diversity bonus
      5. Score = regret × uncertainty + λ × MMR

    This targets states where:
      - The policy's choice has high expected regret (wrong action has consequences)
      - The model is uncertain about which action is best (epistemic uncertainty)
      - The state is diverse from already-selected ones (MMR)
    """

    def __init__(
        self,
        ensemble_logits: Optional[List[np.ndarray]] = None,
        lambda_mmr: float = 0.3,
        seed: int = 42,
    ):
        self.ensemble_logits = ensemble_logits
        self.lambda_mmr = float(lambda_mmr)
        self.rng = np.random.RandomState(seed)

    def _expected_regret(self, logits: np.ndarray, costs: np.ndarray) -> np.ndarray:
        """Compute expected regret of policy action vs SimExpert-best.

        regret(s) = Σ_a p(a|s) · [cost(a) - cost(a*)]

        Where cost(a*) is the minimum SimExpert cost.
        """
        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        best_cost = costs.min(axis=-1, keepdims=True)  # [N, 1]
        regret_per_action = costs - best_cost  # [N, num_actions]
        expected_regret = (probs * regret_per_action).sum(axis=-1)  # [N]
        return expected_regret

    def _epistemic_uncertainty(self, logits: np.ndarray) -> np.ndarray:
        """Estimate epistemic uncertainty from ensemble variance.

        epistemic = variance of ensemble predictions / total variance
        """
        if self.ensemble_logits is None or len(self.ensemble_logits) < 2:
            # No ensemble: use predictive entropy as proxy
            probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
            return -np.sum(probs * np.log(probs + 1e-10), axis=-1)

        all_probs = []
        for member_logits in self.ensemble_logits:
            probs = F.softmax(torch.from_numpy(member_logits), dim=-1).numpy()
            all_probs.append(probs)

        all_probs = np.stack(all_probs, axis=0)  # [M, N, num_actions]
        mean_probs = all_probs.mean(axis=0)  # [N, num_actions]

        # Epistemic = variance of ensemble means
        epistemic = all_probs.var(axis=0).mean(axis=-1)  # [N]
        # Normalize to [0, 1]
        max_ep = epistemic.max()
        if max_ep > 0:
            epistemic = epistemic / max_ep

        return epistemic

    def score_states(self, logits, features=None, costs=None):
        # Expected regret
        if costs is not None:
            regret = self._expected_regret(logits, costs)
        else:
            regret = np.ones(len(logits))

        # Epistemic uncertainty
        uncertainty = self._epistemic_uncertainty(logits)

        # Base score = regret × uncertainty
        scores = regret * uncertainty

        # Normalize
        if scores.max() > 0:
            scores = scores / scores.max()

        return scores

    def select(
        self,
        logits: np.ndarray,
        K: int,
        features: Optional[np.ndarray] = None,
        costs: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MMR-augmented selection: regret × uncertainty + λ × diversity."""
        N = len(logits)
        K = min(K, N)

        # Compute base scores
        base_scores = self.score_states(logits, features=features, costs=costs)

        # MMR greedy selection
        selected: List[int] = []
        remaining = list(range(N))

        # First point: max base score
        first_idx = int(np.argmax(base_scores))
        selected.append(first_idx)
        remaining.remove(first_idx)

        for _ in range(K - 1):
            best_score = -float("inf")
            best_idx = -1

            for idx in remaining:
                # Base score
                score = base_scores[idx]

                # MMR diversity bonus: max similarity to already-selected
                if features is not None and selected:
                    feat = features[idx]
                    selected_feats = features[selected]
                    similarities = np.dot(selected_feats, feat) / (
                        np.linalg.norm(selected_feats, axis=-1) * np.linalg.norm(feat) + 1e-10
                    )
                    max_sim = float(np.max(similarities))
                    diversity_bonus = self.lambda_mmr * (1.0 - max_sim)
                    score += diversity_bonus

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                # Fallback: pick from remaining randomly
                r_idx = self.rng.choice(remaining)
                selected.append(r_idx)
                remaining.remove(r_idx)

        return np.array(selected, dtype=np.int64), base_scores[selected]


# ── Factory ────────────────────────────────────────────────────────────────


def create_acquisition(
    method: str,
    ensemble_logits: Optional[List[np.ndarray]] = None,
    source_features: Optional[np.ndarray] = None,
    seed: int = 42,
) -> AcquisitionFunction:
    """Factory for creating acquisition functions by name."""
    method = str(method).lower().strip()

    if method == "random":
        return RandomAcquisition(seed=seed)
    elif method == "entropy":
        return EntropyAcquisition()
    elif method == "ood":
        return OODAcquisition(source_features=source_features)
    elif method == "core_set":
        return CoreSetAcquisition(seed=seed)
    elif method == "full_ensemble":
        return EnsembleDisagreementAcquisition(ensemble_logits=ensemble_logits)
    elif method == "regret_aware":
        return RegretAwareAcquisition(ensemble_logits=ensemble_logits, seed=seed)
    else:
        raise ValueError(f"Unknown acquisition method: {method}. "
                         f"Choose from: random, entropy, ood, core_set, full_ensemble, regret_aware")


# ── MC Dropout Ensemble (utility) ─────────────────────────────────────────


def mc_dropout_logits(
    model,
    prepared_batch: Dict,
    n_samples: int = 10,
    dropout_enabled: bool = True,
) -> List[np.ndarray]:
    """Collect MC Dropout samples from a single model.

    The model must have dropout layers and be in train() mode for MC Dropout.
    Use model.train() before calling this (dropout active but no grad needed
    since we're doing forward passes under torch.no_grad()).

    Args:
        model: PrefixActionPolicy with dropout layers
        prepared_batch: from model.prepare_policy_batch()
        n_samples: number of MC samples
        dropout_enabled: if False, returns n_samples copies of deterministic output

    Returns:
        list of [B, num_actions] logit arrays
    """
    was_training = model.training
    if dropout_enabled:
        model.train()  # Enable dropout for MC sampling
    else:
        model.eval()

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model.get_action_logits_from_prepared(prepared_batch)
            samples.append(logits.float().cpu().numpy())

    if was_training:
        model.train()
    else:
        model.eval()

    return samples


def compute_regret_metrics(
    logits: np.ndarray,
    costs: np.ndarray,
) -> Dict[str, float]:
    """Compute regret-based evaluation metrics.

    Args:
        logits: [N, num_actions] policy logits
        costs: [N, num_actions] SimExpert costs

    Returns:
        dict with: top1_cost_regret, expected_cost_regret, top1_cost, best_cost
    """
    N = len(logits)
    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()

    # Best action by policy (argmax logits)
    policy_best_idx = np.argmax(logits, axis=-1)
    policy_best_cost = costs[np.arange(N), policy_best_idx]

    # Best possible cost (SimExpert oracle)
    best_cost = costs.min(axis=-1)

    # Regret of top-1 action
    top1_regret = policy_best_cost - best_cost

    # Expected regret
    expected_regret = (probs * (costs - best_cost[:, None])).sum(axis=-1)

    return {
        "top1_cost": float(policy_best_cost.mean()),
        "best_cost": float(best_cost.mean()),
        "top1_cost_regret": float(top1_regret.mean()),
        "expected_cost_regret": float(expected_regret.mean()),
        "top1_cost_regret_median": float(np.median(top1_regret)),
        "expected_cost_regret_median": float(np.median(expected_regret)),
    }
