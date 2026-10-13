"""Count-only federation feasibility and fixed-width arithmetic controls."""

import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.training.secure_aggregation_preflight import (
    MAX_COUNT,
    SecureAggregationPreflightReport,
    secure_aggregation_preflight,
)


def preflight(**changes):
    args = dict(
        participant_count=10,
        quorum=6,
        share_threshold=4,
        dropout_bound=2,
        minimum_aggregation_group=3,
    )
    return secure_aggregation_preflight(**(args | changes))


def test_positive_headroom_accepts_declared_counts_only():
    result = preflight()
    assert result.outcome == "accept"
    assert result.reasons == ()
    assert result.worst_case_survivors == 8


@pytest.mark.parametrize(
    "changes",
    [
        {"participant_count": 8},
        {"participant_count": 8, "quorum": 5, "share_threshold": 6},
    ],
)
def test_exact_or_inconsistent_threshold_boundaries(changes):
    result = preflight(**changes)
    if changes.get("share_threshold") == 6:
        assert result.outcome == "reject"
        assert "threshold_exceeds_quorum" in result.reasons
    else:
        assert result.outcome == "review"
        assert result.reasons == ("zero_headroom",)


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"quorum": 9}, "under_quorum"),
        ({"quorum": 9, "share_threshold": 9}, "unrecoverable_dropout"),
        ({"quorum": 9, "minimum_aggregation_group": 9}, "unsafe_small_group"),
        ({"minimum_aggregation_group": 1}, "unsafe_small_group"),
        ({"dropout_bound": 11}, "dropout_exceeds_participants"),
        ({"minimum_aggregation_group": 7}, "minimum_group_exceeds_quorum"),
        ({"share_threshold": 7}, "threshold_exceeds_quorum"),
    ],
)
def test_inconsistent_constraints_reject_independently(changes, reason):
    result = preflight(**changes)
    assert result.outcome == "reject"
    assert reason in result.reasons


@pytest.mark.parametrize(
    "field",
    [
        "participant_count",
        "quorum",
        "share_threshold",
        "dropout_bound",
        "minimum_aggregation_group",
    ],
)
@pytest.mark.parametrize("value", [True, "synthetic-private-value", -1, 1.5, None])
def test_malformed_values_produce_controlled_metadata(field, value):
    result = preflight(**{field: value})
    assert result.outcome == "reject"
    assert result.reasons == ("malformed_count",)
    assert "synthetic-private-value" not in result.to_json()
    assert result.worst_case_survivors is None


@pytest.mark.parametrize(
    "field",
    [
        "participant_count",
        "quorum",
        "share_threshold",
        "dropout_bound",
        "minimum_aggregation_group",
    ],
)
def test_input_overflow_rejects_before_arithmetic(field):
    assert preflight(**{field: MAX_COUNT + 1}).reasons == ("count_overflow",)


def test_checked_requirement_addition_detects_overflow():
    result = preflight(participant_count=MAX_COUNT, quorum=MAX_COUNT, dropout_bound=1)
    assert result.reasons == ("arithmetic_overflow",)
    assert result.worst_case_survivors is None


def test_maximum_representable_addition_is_exact_without_wraparound():
    result = preflight(
        participant_count=MAX_COUNT, quorum=MAX_COUNT - 1, dropout_bound=1
    )
    assert result.outcome == "review"
    assert result.worst_case_survivors == MAX_COUNT - 1


def test_zero_dropout_is_valid_and_all_dropout_rejects():
    assert preflight(dropout_bound=0).outcome == "accept"
    result = preflight(dropout_bound=10)
    assert result.outcome == "reject"
    assert set(result.reasons) == {
        "under_quorum",
        "unrecoverable_dropout",
        "unsafe_small_group",
    }


def test_report_is_immutable_and_contains_only_counts_and_categories():
    result = preflight()
    assert json.loads(result.to_json()) == {
        "schema_version": 1,
        "outcome": "accept",
        "reasons": [],
        "worst_case_survivors": 8,
    }
    with pytest.raises(FrozenInstanceError):
        result.outcome = "reject"


@pytest.mark.parametrize(
    "args",
    [
        ("approved", (), 8),
        ("accept", ("under_quorum",), 8),
        ("reject", (), 8),
        ("review", ("under_quorum",), 8),
        ("reject", ("synthetic-private-value",), None),
        ("accept", (), True),
        ("accept", (), MAX_COUNT + 1),
        ("accept", (), None),
        ("review", ("zero_headroom",), 1),
    ],
)
def test_direct_report_constructor_rejects_invalid_or_unsafe_metadata(args):
    with pytest.raises(ValueError) as exc:
        SecureAggregationPreflightReport(*args)
    assert "synthetic-private-value" not in str(exc.value)
