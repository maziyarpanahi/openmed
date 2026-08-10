"""Offline tests for deterministic structured access-review expiry gates."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from openmed.compliance.access_review_expiry import (
    ACCESS_REVIEW_BLOCK,
    ACCESS_REVIEW_PASS,
    REASON_EXPIRED,
    REASON_MISSING_DECISION_CATEGORIES,
    REASON_NOT_YET_VALID,
    REASON_POLICY_FINGERPRINT_MISMATCH,
    AccessReview,
    AccessReviewExpiryGate,
    AccessReviewValidationError,
    evaluate_access_review,
)

UTC = timezone.utc
ISSUED_AT = datetime(2026, 8, 10, 8, 0, tzinfo=UTC)
EXPIRES_AT = datetime(2026, 8, 10, 16, 0, tzinfo=UTC)
POLICY_FINGERPRINT = "sha256:" + ("a" * 64)


def _review(
    *,
    categories=("purpose", "scope", "retention"),
    policy_fingerprint=POLICY_FINGERPRINT,
) -> AccessReview:
    return AccessReview(
        issued_at=ISSUED_AT,
        expires_at=EXPIRES_AT,
        policy_fingerprint=policy_fingerprint,
        decision_categories=categories,
    )


def test_current_review_passes_at_issue_time_without_using_wall_clock() -> None:
    result = evaluate_access_review(
        _review(),
        expected_policy_fingerprint=POLICY_FINGERPRINT,
        required_decision_categories={"retention", "scope", "purpose"},
        as_of=ISSUED_AT,
    )

    assert result.decision == ACCESS_REVIEW_PASS
    assert result.passed is True
    assert result.reasons == ()
    assert result.policy_fingerprint_matches is True
    assert result.missing_decision_categories == ()


def test_expiry_is_exclusive_and_future_issue_time_blocks() -> None:
    at_expiry = evaluate_access_review(
        _review(),
        expected_policy_fingerprint=POLICY_FINGERPRINT,
        required_decision_categories=("purpose", "scope", "retention"),
        as_of=EXPIRES_AT,
    )
    before_issue = evaluate_access_review(
        _review(),
        expected_policy_fingerprint=POLICY_FINGERPRINT,
        required_decision_categories=("purpose", "scope", "retention"),
        as_of=ISSUED_AT - timedelta(microseconds=1),
    )

    assert at_expiry.decision == ACCESS_REVIEW_BLOCK
    assert at_expiry.reasons == (REASON_EXPIRED,)
    assert before_issue.decision == ACCESS_REVIEW_BLOCK
    assert before_issue.reasons == (REASON_NOT_YET_VALID,)


def test_policy_and_required_category_failures_are_stable_and_aggregate_only() -> None:
    result = evaluate_access_review(
        _review(categories=("purpose",)),
        expected_policy_fingerprint="sha256:" + ("b" * 64),
        required_decision_categories=("scope", "purpose", "retention"),
        as_of=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
    )

    assert result.decision == ACCESS_REVIEW_BLOCK
    assert result.reasons == (
        REASON_POLICY_FINGERPRINT_MISMATCH,
        REASON_MISSING_DECISION_CATEGORIES,
    )
    assert result.missing_decision_categories == ("retention", "scope")
    assert result.to_dict()["policy_fingerprint_matches"] is False
    assert "b" * 64 not in result.to_json()


def test_review_and_gate_serialization_is_deterministic_and_discards_mapping_values() -> (
    None
):
    review = AccessReview(
        issued_at=ISSUED_AT.isoformat().replace("+00:00", "Z"),
        expires_at=EXPIRES_AT,
        policy_fingerprint=POLICY_FINGERPRINT,
        decisions={
            "scope": "SYNTHETIC-DECISION-NOTE",
            "purpose": "SYNTHETIC-REQUEST-CONTENT",
        },
    )
    equivalent = AccessReview(
        issued_at=ISSUED_AT,
        expires_at=EXPIRES_AT,
        policy_fingerprint=POLICY_FINGERPRINT,
        decision_categories=("purpose", "scope"),
    )
    gate = AccessReviewExpiryGate(
        expected_policy_fingerprint=POLICY_FINGERPRINT,
        required_decision_categories=("scope", "purpose"),
    )

    assert review.to_json() == equivalent.to_json()
    result = gate.evaluate(review, now="2026-08-10T12:00:00Z")
    assert result.decision == ACCESS_REVIEW_PASS
    assert json.loads(result.to_json()) == result.to_dict()
    serialized = review.to_json() + result.to_json()
    assert "SYNTHETIC-DECISION-NOTE" not in serialized
    assert "SYNTHETIC-REQUEST-CONTENT" not in serialized


def test_structured_round_trip_rejects_unknown_fields() -> None:
    review = _review()
    restored = AccessReview.from_dict(json.loads(review.to_json()))

    assert restored == review
    with pytest.raises(AccessReviewValidationError, match="fields are invalid"):
        AccessReview.from_dict({**review.to_dict(), "request": "RAW-CONTENT"})


@pytest.mark.parametrize(
    ("factory", "value", "message"),
    [
        (
            lambda value: AccessReview(
                issued_at=value,
                expires_at=EXPIRES_AT,
                policy_fingerprint=POLICY_FINGERPRINT,
            ),
            datetime(2026, 8, 10, 8, 0),
            "timezone-aware",
        ),
        (
            lambda value: AccessReview(
                issued_at=ISSUED_AT,
                expires_at=value,
                policy_fingerprint=POLICY_FINGERPRINT,
            ),
            ISSUED_AT,
            "later than",
        ),
        (
            lambda value: AccessReview(
                issued_at=ISSUED_AT,
                expires_at=EXPIRES_AT,
                policy_fingerprint=value,
            ),
            "RAW POLICY CONTENT",
            "policy fingerprint",
        ),
    ],
)
def test_invalid_metadata_errors_do_not_echo_sensitive_values(
    factory, value, message
) -> None:
    with pytest.raises(AccessReviewValidationError, match=message) as error:
        factory(value)

    assert str(value) not in str(error.value)


def test_clock_is_required_and_no_implicit_wall_clock_is_used() -> None:
    with pytest.raises(AccessReviewValidationError, match="supplied clock"):
        evaluate_access_review(
            _review(),
            expected_policy_fingerprint=POLICY_FINGERPRINT,
        )
