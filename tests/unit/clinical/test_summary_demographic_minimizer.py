"""Synthetic tests for purpose-bound demographic minimization."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.summary_demographic_minimizer import (
    NOT_ALLOWED_FOR_PURPOSE,
    DemographicEvidence,
    DemographicMinimizationError,
    DemographicPurposePolicy,
    minimize_demographic_evidence,
)


def _opaque(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


PURPOSE = _opaque("handoff-summary")
AGE = _opaque("age-band")
LANGUAGE = _opaque("preferred-language")
ETHNICITY = _opaque("ethnicity")


def _policy(*allowed: str) -> DemographicPurposePolicy:
    return DemographicPurposePolicy(PURPOSE, frozenset(allowed))


def test_only_explicitly_allowed_classes_reach_generation() -> None:
    allowed = DemographicEvidence(AGE, "synthetic-adult")
    removed_language = DemographicEvidence(LANGUAGE, "synthetic-language")
    removed_ethnicity = DemographicEvidence(ETHNICITY, "synthetic-ethnicity")

    result = minimize_demographic_evidence(
        [allowed, removed_language, removed_ethnicity],
        _policy(AGE),
    )

    assert result.allowed_evidence == (allowed,)
    assert result.report.input_count == 3
    assert result.report.allowed_count == 1
    assert result.report.removed_count == 2
    assert [
        item.attribute_class_id for item in result.report.removed_classes
    ] == sorted([LANGUAGE, ETHNICITY])
    assert all(
        item.reason_code == NOT_ALLOWED_FOR_PURPOSE
        for item in result.report.removed_classes
    )


def test_removed_values_are_not_retained_or_reported() -> None:
    sentinels = ("SYNTHETIC_LANGUAGE_VALUE", "SYNTHETIC_ETHNICITY_VALUE")
    result = minimize_demographic_evidence(
        [
            DemographicEvidence(LANGUAGE, sentinels[0]),
            DemographicEvidence(ETHNICITY, sentinels[1]),
        ],
        _policy(),
    )

    assert result.allowed_evidence == ()
    serialized = result.report.to_json()
    assert json.loads(serialized) == result.to_dict()
    for sentinel in sentinels:
        assert sentinel not in serialized
        assert sentinel not in repr(result)
    assert set(result.to_dict()) == {
        "allowed_count",
        "input_count",
        "purpose_id",
        "removed_classes",
        "removed_count",
        "schema_version",
    }


def test_allowed_values_are_available_but_excluded_from_safe_views() -> None:
    sentinel = "SYNTHETIC_ALLOWED_VALUE"
    result = minimize_demographic_evidence(
        [DemographicEvidence(AGE, sentinel)],
        _policy(AGE),
    )

    assert result.allowed_evidence[0].value == sentinel
    assert sentinel not in repr(result.allowed_evidence[0])
    assert sentinel not in repr(result)
    assert sentinel not in result.report.to_json()


def test_reports_are_deterministic_across_class_encounter_order() -> None:
    first = minimize_demographic_evidence(
        [
            DemographicEvidence(LANGUAGE, "value-one"),
            DemographicEvidence(ETHNICITY, "value-two"),
            DemographicEvidence(LANGUAGE, "value-three"),
        ],
        _policy(),
    )
    second = minimize_demographic_evidence(
        [
            DemographicEvidence(ETHNICITY, "different-value"),
            DemographicEvidence(LANGUAGE, "different-value"),
            DemographicEvidence(LANGUAGE, "different-value"),
        ],
        _policy(),
    )

    assert first.report == second.report
    assert first.report.to_json() == second.report.to_json()


def test_empty_input_and_empty_allowlist_are_supported() -> None:
    result = minimize_demographic_evidence([], _policy())

    assert result.allowed_evidence == ()
    assert result.report.input_count == 0
    assert result.report.removed_classes == ()


def test_invalid_inputs_fail_without_echoing_sensitive_values() -> None:
    sentinel = "SYNTHETIC_PATIENT_ATTRIBUTE"
    with pytest.raises(DemographicMinimizationError) as error:
        DemographicEvidence(sentinel, "value")
    assert sentinel not in str(error.value)

    with pytest.raises(DemographicMinimizationError, match="collection"):
        minimize_demographic_evidence([object()], _policy())  # type: ignore[list-item]
    with pytest.raises(DemographicMinimizationError, match="policy"):
        minimize_demographic_evidence([], object())  # type: ignore[arg-type]
    with pytest.raises(DemographicMinimizationError, match="set"):
        DemographicPurposePolicy(PURPOSE, (AGE,))  # type: ignore[arg-type]


def test_policy_and_result_are_immutable() -> None:
    policy = _policy(AGE)
    result = minimize_demographic_evidence([], policy)

    with pytest.raises(FrozenInstanceError):
        policy.purpose_id = PURPOSE  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.allowed_evidence = ()  # type: ignore[misc]
