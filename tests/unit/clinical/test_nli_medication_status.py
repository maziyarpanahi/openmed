from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.nli_medication_status import (
    EventTimeRelation,
    MedicationClaimStatus,
    MedicationStatusClaim,
    MedicationStatusPrecheckError,
    MedicationStatusPrecheckStatus,
    MedicationStatusReason,
    check_medication_status_contradiction,
    medication_status_contradiction_precheck,
    normalize_nli_medication_status,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("initiated", MedicationClaimStatus.STARTED),
        ("active", MedicationClaimStatus.CONTINUED),
        ("on hold", MedicationClaimStatus.HELD),
        ("discontinued", MedicationClaimStatus.STOPPED),
        ("past", MedicationClaimStatus.HISTORICAL),
        ("possible", MedicationClaimStatus.UNCERTAIN),
        (None, MedicationClaimStatus.UNCERTAIN),
    ],
)
def test_status_normalization_is_explicit_and_deterministic(
    raw: object,
    expected: MedicationClaimStatus,
) -> None:
    assert normalize_nli_medication_status(raw) is expected


def test_same_status_and_time_are_compatible() -> None:
    result = medication_status_contradiction_precheck(
        {
            "medication_key": "synthetic-a",
            "status": "started",
            "event_time": "2026-01-02",
        },
        {
            "medication_key": "synthetic-a",
            "status": "initiated",
            "event_time": "2026-01-02",
        },
    )

    assert result.status is MedicationStatusPrecheckStatus.COMPATIBLE
    assert result.inference_allowed is True
    assert result.evidence == ()


@pytest.mark.parametrize(
    ("premise_status", "hypothesis_status"),
    [
        ("started", "stopped"),
        ("continued", "held"),
        ("historical", "active"),
    ],
)
def test_same_time_active_inactive_states_are_contradictions(
    premise_status: str,
    hypothesis_status: str,
) -> None:
    result = medication_status_contradiction_precheck(
        {"status": premise_status, "event_time": "2026-01-02"},
        {"status": hypothesis_status, "event_time": "2026-01-02"},
    )

    assert result.status is MedicationStatusPrecheckStatus.CONTRADICTION
    assert result.contradiction is True
    assert result.inference_allowed is False
    assert result.evidence[0].reason is MedicationStatusReason.STATUS_CONTRADICTION
    assert result.evidence[0].time_relation is EventTimeRelation.SAME


def test_ordered_status_change_is_escalated_not_inferred() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "started", "event_time": "2026-01-02"},
        {"status": "stopped", "event_time": "2026-01-04"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert result.requires_review is True
    assert (
        result.evidence[0].reason is MedicationStatusReason.AMBIGUOUS_REGIMEN_TRANSITION
    )
    assert result.evidence[0].time_relation is EventTimeRelation.PREMISE_BEFORE


def test_changed_status_is_always_an_ambiguous_transition() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "changed", "event_time": "2026-01-02"},
        {"status": "continued", "event_time": "2026-01-02"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert (
        result.evidence[0].reason is MedicationStatusReason.AMBIGUOUS_REGIMEN_TRANSITION
    )


def test_uncertain_status_is_escalated() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "possible", "event_time": "2026-01-02"},
        {"status": "continued", "event_time": "2026-01-02"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert result.evidence[0].reason is MedicationStatusReason.UNCERTAIN_STATUS


def test_missing_status_is_not_defaulted_to_active() -> None:
    result = medication_status_contradiction_precheck(
        {"event_time": "2026-01-02"},
        {"status": "continued", "event_time": "2026-01-02"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert result.evidence[0].premise_status is MedicationClaimStatus.UNCERTAIN


def test_conflicting_status_without_both_times_requires_review() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "continued"},
        {"status": "stopped", "event_time": "2026-01-02"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert result.evidence[0].reason is MedicationStatusReason.MISSING_EVENT_TIME
    assert result.evidence[0].time_relation is EventTimeRelation.PARTIAL


def test_same_status_at_different_times_requires_review() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "started", "event_time": "2026-01-02"},
        {"status": "started", "event_time": "2026-02-02"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert result.evidence[0].reason is MedicationStatusReason.EVENT_TIME_MISMATCH


def test_different_time_granularity_is_not_treated_as_same_time() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "continued", "event_time": "2026"},
        {"status": "stopped", "event_time": "2026-01"},
    )

    assert result.status is MedicationStatusPrecheckStatus.REVIEW_REQUIRED
    assert result.evidence[0].reason is MedicationStatusReason.MISSING_EVENT_TIME
    assert result.evidence[0].time_relation is EventTimeRelation.PARTIAL


def test_same_status_without_times_is_compatible() -> None:
    result = medication_status_contradiction_precheck(
        {"status": "historical"},
        {"status": "past"},
    )

    assert result.status is MedicationStatusPrecheckStatus.COMPATIBLE


def test_different_medication_keys_are_not_compared() -> None:
    result = medication_status_contradiction_precheck(
        {"medication_key": "synthetic-a", "status": "started"},
        {"medication_key": "synthetic-b", "status": "stopped"},
    )

    assert result.status is MedicationStatusPrecheckStatus.NOT_APPLICABLE
    assert result.inference_allowed is True


def test_alias_matches_primary_function() -> None:
    premise = {"status": "continued", "event_time": "2026-01-02"}
    hypothesis = {"status": "held", "event_time": "2026-01-02"}

    assert check_medication_status_contradiction(
        premise, hypothesis
    ) == medication_status_contradiction_precheck(premise, hypothesis)


def test_results_are_deterministic() -> None:
    premise = {"status": "continued", "event_time": "2026-01-02"}
    hypothesis = {"status": "stopped", "event_time": "2026-01-02"}

    first = medication_status_contradiction_precheck(premise, hypothesis)
    second = medication_status_contradiction_precheck(premise, hypothesis)

    assert first == second
    assert first.to_dict() == second.to_dict()


def test_reports_repr_and_errors_never_echo_sensitive_values() -> None:
    sentinel = "SENSITIVE_SENTINEL"
    claim = MedicationStatusClaim(
        status="continued",
        event_time="2026-01-02",
        medication_key=sentinel,
        source_start=4,
        source_end=8,
    )
    result = medication_status_contradiction_precheck(
        claim,
        MedicationStatusClaim(
            status="stopped",
            event_time="2026-01-02",
            medication_key=sentinel,
        ),
    )

    rendered = repr(claim) + repr(result) + str(result.to_dict())
    assert sentinel not in rendered
    assert "2026-01-02" not in rendered
    assert "medication_key" not in str(result.to_dict())
    assert "recommended" not in str(result.to_dict()).casefold()

    with pytest.raises(MedicationStatusPrecheckError) as exc_info:
        medication_status_contradiction_precheck(
            {"status": sentinel},
            {"status": "continued"},
        )
    assert sentinel not in str(exc_info.value)


def test_invalid_event_time_fails_with_value_free_error() -> None:
    sentinel = "SENSITIVE_TIME_SENTINEL"
    with pytest.raises(
        MedicationStatusPrecheckError,
        match="invalid medication event time",
    ) as exc_info:
        medication_status_contradiction_precheck(
            {"status": "continued", "event_time": sentinel},
            {"status": "continued"},
        )
    assert sentinel not in str(exc_info.value)


def test_result_and_claim_are_immutable() -> None:
    claim = MedicationStatusClaim(status="continued")
    result = medication_status_contradiction_precheck(
        claim,
        MedicationStatusClaim(status="continued"),
    )

    with pytest.raises(FrozenInstanceError):
        claim.status = "stopped"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        result.status = MedicationStatusPrecheckStatus.CONTRADICTION  # type: ignore[misc]
