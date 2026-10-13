"""Exact policy math, caps/floors, protected gates and counterfactual controls."""

import json
from fractions import Fraction as F

import pytest

from openmed.training.federated.site_weighting import (
    CoarseSiteMetadata,
    SiteWeightingPolicy,
    compare_site_weighting,
    compute_site_weights,
    render_site_weighting_comparison,
)

GROUPS = (
    CoarseSiteMetadata("very_large"),
    CoarseSiteMetadata("tiny", ("low_resource",)),
)


def test_declared_strategies_are_exact_and_normalized():
    assert compute_site_weights(GROUPS, SiteWeightingPolicy("uniform")) == (
        F(1, 2),
        F(1, 2),
    )
    assert compute_site_weights(GROUPS, SiteWeightingPolicy("volume")) == (
        F(16, 17),
        F(1, 17),
    )
    assert compute_site_weights(GROUPS, SiteWeightingPolicy("inverse_volume")) == (
        F(1, 17),
        F(16, 17),
    )


def test_caps_redistribute_residual_without_violating_floors():
    policy = SiteWeightingPolicy(
        "inverse_volume",
        floors=(F(1, 5), F(1, 5)),
        caps=(F(7, 10), F(7, 10)),
        protected_minimums={"low_resource": F(7, 10)},
    )
    weights = compute_site_weights(GROUPS, policy)
    assert weights == (F(3, 10), F(7, 10))
    assert sum(weights) == 1


def test_selected_counterfactual_deltas_are_actual_math():
    policy = SiteWeightingPolicy(
        "inverse_volume",
        caps=(F(7, 10), F(7, 10)),
        protected_minimums={"low_resource": F(7, 10)},
    )
    report = compare_site_weighting(GROUPS, policy)
    assert report["outcome"] == "accept"
    assert report["selected_weights"] == [
        {"numerator": 3, "denominator": 10},
        {"numerator": 7, "denominator": 10},
    ]
    delta = report["selected_minus_baseline"]["volume"][1]
    assert F(delta["numerator"], delta["denominator"]) == F(7, 10) - F(1, 17)
    assert report["protected_coverage"]["low_resource"]["selected"] == {
        "numerator": 7,
        "denominator": 10,
    }


def test_protected_constraint_violation_blocks_weight_export_without_claiming_global_infeasibility():
    policy = SiteWeightingPolicy("volume", protected_minimums={"low_resource": F(1, 2)})
    with pytest.raises(ValueError, match="protected minimum"):
        compute_site_weights(GROUPS, policy)
    report = compare_site_weighting(GROUPS, policy)
    assert report["outcome"] == "hold"
    assert report["selected_weights"] is None
    assert all(value is None for value in report["selected_minus_baseline"].values())
    # Another explicitly selected policy can satisfy the same feasible constraint.
    assert compute_site_weights(
        GROUPS,
        SiteWeightingPolicy("uniform", protected_minimums={"low_resource": F(1, 2)}),
    ) == (F(1, 2), F(1, 2))


def test_overlapping_protected_slices_are_enforced_independently():
    groups = (
        CoarseSiteMetadata("tiny", ("language", "rare")),
        CoarseSiteMetadata("tiny", ("language",)),
        CoarseSiteMetadata("tiny"),
    )
    policy = SiteWeightingPolicy(
        "uniform", protected_minimums={"language": F(2, 3), "rare": F(1, 3)}
    )
    assert compute_site_weights(groups, policy) == (F(1, 3),) * 3
    with pytest.raises(ValueError):
        compute_site_weights(
            groups, SiteWeightingPolicy("uniform", protected_minimums={"rare": F(1, 2)})
        )


@pytest.mark.parametrize(
    "policy",
    [
        SiteWeightingPolicy("uniform", floors=(F(3, 5), F(3, 5))),
        SiteWeightingPolicy("uniform", caps=(F(2, 5), F(2, 5))),
        SiteWeightingPolicy("uniform", floors=(F(3, 5), F(0)), caps=(F(1, 2), F(1))),
        SiteWeightingPolicy("uniform", floors=(F(0),)),
    ],
)
def test_inconsistent_bounds_reject_before_export(policy):
    with pytest.raises(ValueError):
        compute_site_weights(GROUPS, policy)


def test_declared_floors_can_preserve_protected_slices_in_volume_strategy():
    policy = SiteWeightingPolicy(
        "volume", floors=(F(0), F(1, 2)), protected_minimums={"low_resource": F(1, 2)}
    )
    weights = compute_site_weights(GROUPS, policy)
    assert weights[1] >= F(1, 2)
    assert sum(weights) == 1


def test_reordering_positions_reorders_weights_without_changing_math():
    policy = SiteWeightingPolicy("inverse_volume")
    assert compute_site_weights(reversed(GROUPS), policy) == tuple(
        reversed(compute_site_weights(GROUPS, policy))
    )
    assert render_site_weighting_comparison(
        GROUPS, policy
    ) == render_site_weighting_comparison(GROUPS, policy)
    report = json.loads(render_site_weighting_comparison(GROUPS, policy))
    assert "volume_class" not in report
    assert "site_id" not in report
    assert "patient_count" not in report


def test_policy_snapshots_mutable_bounds_and_constraints():
    caps = [F(7, 10), F(7, 10)]
    minima = {"low_resource": F(1, 2)}
    policy = SiteWeightingPolicy("inverse_volume", caps=caps, protected_minimums=minima)
    caps[1] = 0
    minima["low_resource"] = 1
    assert policy.caps == (F(7, 10), F(7, 10))
    assert policy.protected_minimums["low_resource"] == F(1, 2)


@pytest.mark.parametrize(
    "build",
    [
        lambda: CoarseSiteMetadata("synthetic/private"),
        lambda: CoarseSiteMetadata("tiny", ("raw note",)),
        lambda: SiteWeightingPolicy("automatic"),
        lambda: SiteWeightingPolicy("uniform", caps=(True,)),
        lambda: SiteWeightingPolicy("uniform", caps=(float("nan"),)),
        lambda: SiteWeightingPolicy("uniform", caps=(F(1, 10**10),)),
        lambda: SiteWeightingPolicy(
            "uniform", protected_minimums={"synthetic/private": F(1, 2)}
        ),
    ],
)
def test_malformed_metadata_rejects_without_echo(build):
    with pytest.raises(ValueError) as exc:
        build()
    assert "synthetic/private" not in str(exc.value)
    assert "raw note" not in str(exc.value)


def test_128_group_stress_preserves_exact_normalization_and_caps():
    groups = tuple(
        CoarseSiteMetadata(("tiny", "small", "medium", "large", "very_large")[i % 5])
        for i in range(128)
    )
    policy = SiteWeightingPolicy("volume", caps=(F(2, 128),) * 128)
    weights = compute_site_weights(groups, policy)
    assert sum(weights) == 1
    assert all(0 <= weight <= F(2, 128) for weight in weights)


def test_unbounded_groups_are_consumed_only_to_limit():
    calls = []

    def infinite():
        while True:
            calls.append(1)
            yield GROUPS[0]

    with pytest.raises(ValueError, match="limit"):
        compute_site_weights(infinite(), SiteWeightingPolicy("uniform"))
    assert len(calls) == 129
