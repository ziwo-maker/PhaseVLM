from __future__ import annotations

import torch

from projection_train.train_projection_bc import base_policy_kl_mask
from projection_train.cost_ranking import (
    base_action_kl_loss,
    build_candidate_costs,
    pairwise_cost_ranking_loss,
)


def test_candidate_costs_prefer_collector_score_then_legacy_waits() -> None:
    costs, valid = build_candidate_costs(
        [{"ETWT": {"score": 0.8}, "NTST": {"score": 0.1}}],
        [{"ETWT": 1.0, "NTST": 9.0}],
        torch.device("cpu"),
    )
    assert valid[0, 0]
    assert valid[0, 1]
    assert costs[0, 1] < costs[0, 0]


def test_pairwise_cost_ranking_penalizes_wrong_ordering() -> None:
    costs = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    valid = torch.tensor([[True, True, False, False]])
    correct, _ = pairwise_cost_ranking_loss(torch.tensor([[3.0, 0.0, 0.0, 0.0]]), costs, valid)
    wrong, _ = pairwise_cost_ranking_loss(torch.tensor([[0.0, 3.0, 0.0, 0.0]]), costs, valid)
    assert correct.item() < wrong.item()


def test_base_action_kl_penalizes_leaving_prior_action() -> None:
    aligned, valid = base_action_kl_loss(
        torch.tensor([[4.0, 0.0, 0.0, 0.0]]), ["ETWT"], prior_mass=0.7
    )
    shifted, _ = base_action_kl_loss(
        torch.tensor([[0.0, 4.0, 0.0, 0.0]]), ["ETWT"], prior_mass=0.7
    )
    assert valid.tolist() == [True]
    assert aligned.item() < shifted.item()


def test_candidate_costs_apply_weight_overrides_instead_of_precomputed_score() -> None:
    """When weights differ from defaults, raw metrics are used — not the precomputed score."""
    raw = {
        "ETWT": {"score": 0.8, "aql": 0.0, "awt": 1.0},   # precomputed: ETWT bad
        "NTST": {"score": 0.1, "aql": 1.0, "awt": 0.0},   # precomputed: NTST good
    }
    # Default weights → precomputed score shortcut → NTST has lower cost (0.1 vs 0.8).
    costs_default, valid_default = build_candidate_costs(
        [raw], [{}], torch.device("cpu"),
    )
    assert valid_default[0, 0] and valid_default[0, 1]
    assert costs_default[0, 1].item() < costs_default[0, 0].item(), (
        "default weights should use precomputed score (NTST better)"
    )

    # Override awt → recompute from raw metrics. NTST has awt=0 while
    # ETWT has awt=1, so the AWT-heavy ranking must favor NTST.
    costs_awt, valid_awt = build_candidate_costs(
        [raw], [{}], torch.device("cpu"),
        weights={"awt": 10.0, "aql": 0.1},
    )
    assert valid_awt[0, 0] and valid_awt[0, 1]
    assert costs_awt[0, 1].item() < costs_awt[0, 0].item(), (
        "awt-heavy weights should recompute from raw metrics (NTST remains better)"
    )


def test_base_policy_kl_mask_skips_weighted_dagger_conflicts() -> None:
    mask = base_policy_kl_mask(
        base_actions=["ETWT", "NTST", "ELWL"],
        expert_actions=["NTST", "NTST", "NLSL"],
        sample_weights=torch.tensor([4.0, 4.0, 1.0]),
        mode="match_or_low_weight",
        low_weight_threshold=1.0,
        device=torch.device("cpu"),
    )
    assert mask.tolist() == [False, True, True]
