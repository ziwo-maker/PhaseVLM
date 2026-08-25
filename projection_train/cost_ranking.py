from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn.functional as F

from projection_train.action_space import ACTION_LIST, action_to_index


DEFAULT_COST_WEIGHTS: Dict[str, float] = {
    "att": 1.0,
    "aql": 1.0,
    "awt": 1.0,
    "spillback": 0.2,
    "tail_wait": 0.0,
}


def _finite_float(value: object) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if torch.isfinite(torch.tensor(converted)) else None


def _weights_differ_from_defaults(weights: Dict[str, float]) -> bool:
    """Check whether any weight differs materially from its default."""
    for metric in DEFAULT_COST_WEIGHTS:
        if abs(float(weights.get(metric, DEFAULT_COST_WEIGHTS[metric])) - float(DEFAULT_COST_WEIGHTS[metric])) > 1e-8:
            return True
    return False


def _stored_weights_match(raw_metrics: Dict[str, object], weights: Dict[str, float]) -> bool:
    """Check whether stored weight_* metadata matches the requested weights."""
    for metric in DEFAULT_COST_WEIGHTS:
        stored_key = f"weight_{metric}"
        for entry in raw_metrics.values():
            if isinstance(entry, dict) and stored_key in entry:
                stored = _finite_float(entry[stored_key])
                if stored is not None:
                    requested = float(weights.get(metric, DEFAULT_COST_WEIGHTS[metric]))
                    if abs(stored - requested) > 1e-8:
                        return False
                break  # one action's metadata is enough
    return True


def _normalized_metric_costs(raw_metrics: Dict[str, object], weights: Dict[str, float]) -> Dict[str, float]:
    """Return per-action costs from raw candidate metrics on a common [0, 1] scale.

    Precomputed ``score`` values are reused only when (a) the caller's weights
    match the defaults AND (b) any stored ``weight_*`` metadata in the record
    also matches the requested weights.  Otherwise we recompute from raw
    per-metric fields so the intended weighting is actually applied.
    """
    weights_match_defaults = not _weights_differ_from_defaults(weights)
    use_precomputed = weights_match_defaults and _stored_weights_match(raw_metrics, weights)

    if use_precomputed:
        score_costs: Dict[str, float] = {}
        for action in ACTION_LIST:
            entry = raw_metrics.get(action)
            if isinstance(entry, dict):
                score = _finite_float(entry.get("score"))
                if score is not None:
                    score_costs[action] = score
        if len(score_costs) >= 2:
            return score_costs

    components: Dict[str, Dict[str, float]] = {}
    for metric in DEFAULT_COST_WEIGHTS:
        values: Dict[str, float] = {}
        for action in ACTION_LIST:
            entry = raw_metrics.get(action)
            if isinstance(entry, dict):
                value = _finite_float(entry.get(metric))
                if value is not None:
                    values[action] = value
        if len(values) >= 2:
            low, high = min(values.values()), max(values.values())
            spread = high - low
            components[metric] = {
                action: 0.0 if spread <= 1e-8 else (value - low) / spread
                for action, value in values.items()
            }

    costs: Dict[str, float] = {}
    for action in ACTION_LIST:
        weighted_values = [
            float(weights.get(metric, DEFAULT_COST_WEIGHTS[metric])) * component[action]
            for metric, component in components.items()
            if action in component
        ]
        if weighted_values:
            costs[action] = float(sum(weighted_values))
    return costs


def build_candidate_costs(
    raw_metrics_batch: Sequence[Dict[str, object]],
    fallback_waits_batch: Sequence[Dict[str, float]],
    device: torch.device,
    weights: Dict[str, float] | None = None,
    pareto_feasible_batch: Sequence[Dict[str, bool]] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a dense cost matrix and validity mask from SimExpert candidate metadata.

    Lower cost is better. `candidate_action_metrics.score` is used when the
    collector provided it; otherwise the raw metrics are min-max normalized per
    decision. Legacy datasets fall back to `candidate_action_waits`.

    When ``pareto_feasible_batch`` is provided, actions explicitly marked as
    infeasible (False) are excluded from the validity mask, so soft targets and
    ranking losses will not reward constraint-violating or dominated actions.
    """
    resolved_weights = dict(DEFAULT_COST_WEIGHTS)
    if weights:
        resolved_weights.update({key: float(value) for key, value in weights.items()})

    rows = []
    masks = []
    total = max(len(raw_metrics_batch), len(fallback_waits_batch))
    for index in range(total):
        raw_metrics = raw_metrics_batch[index] if index < len(raw_metrics_batch) else {}
        fallback_waits = fallback_waits_batch[index] if index < len(fallback_waits_batch) else {}
        costs = _normalized_metric_costs(raw_metrics, resolved_weights) if isinstance(raw_metrics, dict) else {}
        if len(costs) < 2 and isinstance(fallback_waits, dict):
            costs = {
                action: value
                for action in ACTION_LIST
                if (value := _finite_float(fallback_waits.get(action))) is not None
            }
        # Apply Pareto feasibility filter if available
        pareto_mask = None
        if pareto_feasible_batch is not None and index < len(pareto_feasible_batch):
            pareto_mask = pareto_feasible_batch[index]
        row = []
        mask_row = []
        for action in ACTION_LIST:
            has_cost = action in costs
            # An action is valid only if: has cost AND (no pareto mask OR pareto says feasible)
            is_feasible = True
            if pareto_mask and action in pareto_mask:
                is_feasible = bool(pareto_mask[action])
            is_valid = has_cost and is_feasible
            row.append(float(costs.get(action, 0.0)))
            mask_row.append(is_valid)
        rows.append(row)
        masks.append(mask_row)
    if not rows:
        return (
            torch.empty((0, len(ACTION_LIST)), dtype=torch.float32, device=device),
            torch.empty((0, len(ACTION_LIST)), dtype=torch.bool, device=device),
        )
    return (
        torch.tensor(rows, dtype=torch.float32, device=device),
        torch.tensor(masks, dtype=torch.bool, device=device),
    )


def cost_soft_targets(costs: torch.Tensor, valid_mask: torch.Tensor, temperature: float) -> torch.Tensor:
    safe_mask = valid_mask.clone()
    empty_rows = ~safe_mask.any(dim=-1)
    if bool(empty_rows.any().item()):
        safe_mask[empty_rows, 0] = True
    masked_costs = costs.masked_fill(~safe_mask, float("inf"))
    best_cost = masked_costs.amin(dim=-1, keepdim=True)
    scaled = -(costs - best_cost) / max(float(temperature), 1e-6)
    targets = torch.softmax(scaled.masked_fill(~safe_mask, float("-inf")), dim=-1)
    if bool(empty_rows.any().item()):
        targets[empty_rows] = 0.0
    return targets


def pairwise_cost_ranking_loss(
    logits: torch.Tensor,
    costs: torch.Tensor,
    valid_mask: torch.Tensor,
    min_gap: float = 0.0,
    margin: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encourage every lower-cost action to score above each worse action."""
    cost_gap = costs.unsqueeze(-1) - costs.unsqueeze(-2)
    valid_pair = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)
    # row action is better than column action when its cost is lower.
    better_pair = valid_pair & (cost_gap < -max(float(min_gap), 0.0))
    logit_gap = logits.unsqueeze(-1) - logits.unsqueeze(-2)
    pair_loss = F.softplus(float(margin) - logit_gap)
    pair_weight = (-cost_gap).clamp_min(0.0) * better_pair.to(costs.dtype)
    per_sample_denominator = pair_weight.sum(dim=(-2, -1))
    per_sample = (pair_loss * pair_weight).sum(dim=(-2, -1)) / per_sample_denominator.clamp_min(1e-8)
    return per_sample, per_sample_denominator > 0


def base_action_kl_loss(
    logits: torch.Tensor,
    base_actions: Sequence[str],
    prior_mass: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KL(pi || base-action prior), with a nonzero floor for every action."""
    action_count = len(ACTION_LIST)
    prior_mass = min(max(float(prior_mass), 1.0 / action_count), 1.0 - 1e-6)
    prior = torch.full_like(logits, (1.0 - prior_mass) / max(1, action_count - 1))
    valid = torch.zeros((logits.shape[0],), dtype=torch.bool, device=logits.device)
    for index, action in enumerate(base_actions):
        if str(action) in ACTION_LIST:
            prior[index, action_to_index(str(action))] = prior_mass
            valid[index] = True
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return (probs * (log_probs - prior.log())).sum(dim=-1), valid
