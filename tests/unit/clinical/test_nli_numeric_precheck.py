from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.nli_numeric_precheck import (
    NumericClaim,
    NumericPrecheckError,
    NumericPrecheckReason,
    NumericPrecheckStatus,
    check_numeric_contradiction,
    numeric_contradiction_precheck,
)


def test_equal_cross_unit_values_are_compatible() -> None:
    result = numeric_contradiction_precheck(
        {"measurement_key": "synthetic-a", "value": 10, "unit": "mg/dL"},
        {"measurement_key": "synthetic-a", "value": 0.1, "unit": "g/L"},
    )

    assert result.status is NumericPrecheckStatus.COMPATIBLE
    assert result.inference_allowed is True
    assert result.contradiction is False
    assert result.evidence == ()


def test_exact_value_mismatch_is_a_pre_model_contradiction() -> None:
    result = numeric_contradiction_precheck(
        {"value": 10, "unit": "mg/dL", "source_span": (4, 6)},
        {"value": 11, "unit": "mg/dL", "source_span": (20, 22)},
    )

    assert result.status is NumericPrecheckStatus.CONTRADICTION
    assert result.inference_allowed is False
    assert result.contradiction is True
    assert result.evidence[0].reason is NumericPrecheckReason.VALUE_MISMATCH
    assert result.evidence[0].field == "value"
    assert result.evidence[0].premise_span == (4, 6)


def test_incommensurable_units_are_a_structured_contradiction() -> None:
    result = check_numeric_contradiction(
        {"value": 5, "unit": "mg"},
        {"value": 5, "unit": "mL"},
    )

    assert result.status is NumericPrecheckStatus.CONTRADICTION
    assert result.evidence[0].reason is NumericPrecheckReason.UNIT_DIMENSION_MISMATCH
    assert result.evidence[0].field == "unit"


def test_overlapping_reference_intervals_are_compatible() -> None:
    result = numeric_contradiction_precheck(
        {"reference_interval": {"low": 8, "high": 12, "unit": "mg/dL"}},
        {"reference_interval": {"low": 0.1, "high": 0.15, "unit": "g/L"}},
    )

    assert result.status is NumericPrecheckStatus.COMPATIBLE


def test_disjoint_reference_intervals_are_reported_without_normality() -> None:
    result = numeric_contradiction_precheck(
        {"reference_low": 8, "reference_high": 9, "reference_unit": "mg/dL"},
        {
            "reference_low": 0.1,
            "reference_high": 0.12,
            "reference_unit": "g/L",
        },
    )

    assert result.status is NumericPrecheckStatus.CONTRADICTION
    assert (
        result.evidence[0].reason is NumericPrecheckReason.REFERENCE_INTERVAL_MISMATCH
    )
    serialized = result.to_dict()
    assert "normal" not in str(serialized).casefold()
    assert "abnormal" not in str(serialized).casefold()


def test_touching_exclusive_intervals_are_disjoint() -> None:
    result = numeric_contradiction_precheck(
        {
            "reference_interval": {"low": 1, "high": 2},
            "high_inclusive": False,
        },
        {"reference_interval": {"low": 2, "high": 3}},
    )

    assert result.status is NumericPrecheckStatus.CONTRADICTION


def test_one_sided_intervals_can_be_compared() -> None:
    result = numeric_contradiction_precheck(
        {"reference_high": 2, "high_inclusive": True},
        {"reference_low": 2, "low_inclusive": True},
    )

    assert result.status is NumericPrecheckStatus.COMPATIBLE


def test_reversed_interval_requires_review() -> None:
    result = numeric_contradiction_precheck(
        {"reference_low": 3, "reference_high": 1},
        {"reference_low": 1, "reference_high": 3},
    )

    assert result.status is NumericPrecheckStatus.REVIEW_REQUIRED
    assert (
        result.evidence[0].reason is NumericPrecheckReason.INCOMPARABLE_NUMERIC_SHAPES
    )


def test_missing_unit_context_requires_review() -> None:
    result = numeric_contradiction_precheck(
        {"value": 5, "unit": "mg"},
        {"value": 5},
    )

    assert result.status is NumericPrecheckStatus.REVIEW_REQUIRED
    assert result.requires_review is True
    assert result.inference_allowed is False
    assert result.evidence[0].reason is NumericPrecheckReason.MISSING_UNIT_CONTEXT


@pytest.mark.parametrize("unit", ["units", "not-a-unit"])
def test_ambiguous_or_unknown_units_require_review(unit: str) -> None:
    result = numeric_contradiction_precheck(
        {"value": 5, "unit": unit},
        {"value": 5, "unit": unit},
    )

    assert result.status is NumericPrecheckStatus.REVIEW_REQUIRED
    assert result.evidence[0].reason is NumericPrecheckReason.UNRESOLVED_UNIT


def test_unitless_numeric_values_are_compared_exactly() -> None:
    result = numeric_contradiction_precheck({"value": 2}, {"value": 3})

    assert result.status is NumericPrecheckStatus.CONTRADICTION
    assert result.evidence[0].reason is NumericPrecheckReason.VALUE_MISMATCH


def test_different_measurement_keys_are_not_compared() -> None:
    result = numeric_contradiction_precheck(
        {"measurement_key": "synthetic-a", "value": 2},
        {"measurement_key": "synthetic-b", "value": 3},
    )

    assert result.status is NumericPrecheckStatus.NOT_APPLICABLE
    assert result.inference_allowed is True


def test_value_and_interval_only_shapes_escalate() -> None:
    result = numeric_contradiction_precheck(
        {"value": 2, "unit": "mg"},
        {"reference_low": 1, "reference_high": 3, "reference_unit": "mg"},
    )

    assert result.status is NumericPrecheckStatus.REVIEW_REQUIRED
    assert (
        result.evidence[0].reason is NumericPrecheckReason.INCOMPARABLE_NUMERIC_SHAPES
    )


def test_results_are_deterministic() -> None:
    premise = {"value": 1, "unit": "mg", "source_span": (0, 1)}
    hypothesis = {"value": 2, "unit": "mg", "source_span": (5, 6)}

    first = numeric_contradiction_precheck(premise, hypothesis)
    second = numeric_contradiction_precheck(premise, hypothesis)

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_reports_repr_and_errors_do_not_echo_raw_values() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    claim = NumericClaim(value=17.25, unit="mg", measurement_key=sentinel)
    result = numeric_contradiction_precheck(
        claim,
        NumericClaim(value=18.5, unit="mg", measurement_key=sentinel),
    )

    rendered = repr(claim) + repr(result) + str(result.to_dict())
    assert sentinel not in rendered
    assert "17.25" not in rendered
    assert "18.5" not in rendered
    assert "mg" not in str(result.to_dict())

    with pytest.raises(NumericPrecheckError) as exc_info:
        numeric_contradiction_precheck(
            {"value": 1, "measurement_key": object()},
            {"value": 1},
        )
    assert sentinel not in str(exc_info.value)


def test_invalid_claims_fail_with_value_free_errors() -> None:
    with pytest.raises(
        NumericPrecheckError,
        match="numeric claim requires value or interval evidence",
    ):
        numeric_contradiction_precheck({}, {"value": 1})

    with pytest.raises(NumericPrecheckError, match="invalid numeric claim source span"):
        numeric_contradiction_precheck(
            {"value": 1, "source_span": (9, 2)},
            {"value": 1},
        )


def test_result_and_claim_are_immutable() -> None:
    claim = NumericClaim(value=1)
    result = numeric_contradiction_precheck(claim, NumericClaim(value=1))

    with pytest.raises(FrozenInstanceError):
        claim.value = 2  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status = NumericPrecheckStatus.CONTRADICTION  # type: ignore[misc]


@pytest.mark.parametrize(
    ("left", "right", "unit"),
    [(0, 5e-13, None), (1e-13, 2e-13, None), (1e-10, 2e-10, "g")],
)
def test_distinct_tiny_values_are_not_hidden_by_absolute_tolerance(left, right, unit):
    result = numeric_contradiction_precheck(
        {"value": left, "unit": unit}, {"value": right, "unit": unit}
    )
    assert result.status is NumericPrecheckStatus.CONTRADICTION
    assert result.evidence[0].reason is NumericPrecheckReason.VALUE_MISMATCH


@pytest.mark.parametrize("unit", [None, "g"])
def test_oversized_magnitudes_require_review(unit):
    result = numeric_contradiction_precheck(
        {"value": 10**400, "unit": unit}, {"value": 1, "unit": unit}
    )
    assert result.status is NumericPrecheckStatus.REVIEW_REQUIRED
    assert not result.inference_allowed


def test_invalid_value_callbacks_do_not_echo_content_in_tracebacks():
    import traceback

    secret = "synthetic-sensitive-555-0199"

    class BrokenValue:
        def __float__(self):
            raise RuntimeError(secret)

        def __str__(self):
            raise RuntimeError(secret)

    with pytest.raises(NumericPrecheckError) as error:
        numeric_contradiction_precheck({"value": BrokenValue()}, {"value": 1})
    assert secret not in "".join(traceback.format_exception(error.value))
