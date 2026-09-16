from __future__ import annotations

import socket
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.nli_batch_plan import (
    NliBatchPlanningError,
    NliPairCost,
    NliRuntimeProfile,
    plan_nli_batches,
)


def test_plans_deterministic_batches_from_both_ceilings() -> None:
    pairs = [
        NliPairCost("pair-a", 4),
        NliPairCost("pair-b", 5),
        NliPairCost("pair-c", 3),
        NliPairCost("pair-d", 2),
    ]
    profile = NliRuntimeProfile(max_tokens_per_batch=10, max_pairs_per_batch=2)

    first = plan_nli_batches(pairs, profile)
    second = plan_nli_batches(pairs, profile)

    assert first == second
    assert [batch.pair_ids for batch in first.batches] == [
        ("pair-a", "pair-b"),
        ("pair-c", "pair-d"),
    ]
    assert [batch.token_count for batch in first.batches] == [9, 5]
    assert first.deferred_pair_ids == ()


def test_token_ceiling_splits_before_pair_ceiling() -> None:
    plan = plan_nli_batches(
        [NliPairCost("pair-a", 6), NliPairCost("pair-b", 5)],
        NliRuntimeProfile(max_tokens_per_batch=10, max_pairs_per_batch=8),
    )

    assert [batch.pair_ids for batch in plan.batches] == [
        ("pair-a",),
        ("pair-b",),
    ]


def test_pair_ceiling_splits_before_token_ceiling() -> None:
    plan = plan_nli_batches(
        [NliPairCost("pair-a", 1), NliPairCost("pair-b", 1)],
        NliRuntimeProfile(max_tokens_per_batch=10, max_pairs_per_batch=1),
    )

    assert len(plan.batches) == 2


def test_pair_equal_to_token_ceiling_is_admitted() -> None:
    plan = plan_nli_batches(
        [NliPairCost("pair-a", 10)],
        NliRuntimeProfile(max_tokens_per_batch=10, max_pairs_per_batch=1),
    )

    assert plan.planned_pair_ids == ("pair-a",)
    assert plan.deferred_pair_ids == ()


def test_oversized_pair_is_deferred_and_later_pair_is_planned() -> None:
    plan = plan_nli_batches(
        [NliPairCost("pair-large", 11), NliPairCost("pair-small", 2)],
        NliRuntimeProfile(max_tokens_per_batch=10, max_pairs_per_batch=2),
    )

    assert plan.planned_pair_ids == ("pair-small",)
    assert plan.deferred_pair_ids == ("pair-large",)


def test_hard_total_token_budget_defers_pairs_that_do_not_fit() -> None:
    plan = plan_nli_batches(
        [
            NliPairCost("pair-a", 6),
            NliPairCost("pair-b", 5),
            NliPairCost("pair-c", 4),
        ],
        NliRuntimeProfile(
            max_tokens_per_batch=10,
            max_pairs_per_batch=3,
            max_total_tokens=10,
        ),
    )

    assert plan.planned_pair_ids == ("pair-a", "pair-c")
    assert plan.deferred_pair_ids == ("pair-b",)
    assert plan.token_count == 10


def test_hard_total_pair_budget_can_defer_every_pair() -> None:
    plan = plan_nli_batches(
        [NliPairCost("pair-a", 1), NliPairCost("pair-b", 1)],
        NliRuntimeProfile(
            max_tokens_per_batch=10,
            max_pairs_per_batch=2,
            max_total_pairs=0,
        ),
    )

    assert plan.batches == ()
    assert plan.deferred_pair_ids == ("pair-a", "pair-b")


def test_mapping_inputs_are_supported_without_text() -> None:
    plan = plan_nli_batches(
        [{"pair_id": "pair-a", "token_count": 2}],
        {"max_tokens_per_batch": 4, "max_pairs_per_batch": 1},
    )

    assert plan.planned_pair_ids == ("pair-a",)


def test_empty_input_returns_empty_plan() -> None:
    plan = plan_nli_batches(
        [],
        NliRuntimeProfile(max_tokens_per_batch=4, max_pairs_per_batch=1),
    )

    assert plan.batches == ()
    assert plan.deferred_pair_ids == ()


def test_planner_performs_no_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_socket(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", fail_socket)
    plan = plan_nli_batches(
        [NliPairCost("pair-a", 1)],
        NliRuntimeProfile(max_tokens_per_batch=4, max_pairs_per_batch=1),
    )

    assert plan.planned_pair_ids == ("pair-a",)


def test_representations_and_audit_report_hide_pair_identifiers() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    pair = NliPairCost(sentinel, 3)
    plan = plan_nli_batches(
        [pair],
        NliRuntimeProfile(max_tokens_per_batch=4, max_pairs_per_batch=1),
    )

    rendered = repr(pair) + repr(plan) + repr(plan.batches[0])
    audit = str(plan.to_audit_dict())
    assert sentinel not in rendered
    assert sentinel not in audit
    assert plan.planned_pair_ids == (sentinel,)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ({"max_tokens_per_batch": 0, "max_pairs_per_batch": 1}, "token ceiling"),
        ({"max_tokens_per_batch": 1, "max_pairs_per_batch": True}, "pair ceiling"),
        (
            {
                "max_tokens_per_batch": 1,
                "max_pairs_per_batch": 1,
                "max_total_tokens": -1,
            },
            "total token budget",
        ),
    ],
)
def test_invalid_profiles_fail_with_value_free_errors(
    profile: dict[str, object], message: str
) -> None:
    with pytest.raises(NliBatchPlanningError, match=message):
        plan_nli_batches([], profile)


def test_invalid_pair_does_not_echo_identifier() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    with pytest.raises(NliBatchPlanningError) as exc_info:
        plan_nli_batches(
            [{"pair_id": sentinel, "token_count": 0}],
            NliRuntimeProfile(max_tokens_per_batch=4, max_pairs_per_batch=1),
        )

    assert sentinel not in str(exc_info.value)


def test_duplicate_identifiers_fail_closed() -> None:
    with pytest.raises(NliBatchPlanningError, match="identifiers must be unique"):
        plan_nli_batches(
            [NliPairCost("pair-a", 1), NliPairCost("pair-a", 2)],
            NliRuntimeProfile(max_tokens_per_batch=4, max_pairs_per_batch=2),
        )


def test_inputs_and_results_are_immutable() -> None:
    pair = NliPairCost("pair-a", 1)
    plan = plan_nli_batches(
        [pair],
        NliRuntimeProfile(max_tokens_per_batch=4, max_pairs_per_batch=1),
    )

    with pytest.raises(FrozenInstanceError):
        pair.token_count = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.deferred_pair_ids = ()  # type: ignore[misc]
