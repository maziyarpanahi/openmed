"""Focused tests for the deterministic privacy-evidence freshness gate."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from openmed.compliance import (
    EXPIRED_EVIDENCE,
    FUTURE_TIMESTAMP,
    INVALID_SUPERSESSION_LINK,
    MISSING_EVIDENCE,
    MISSING_POLICY_VERSION,
    MISSING_TIMESTAMP,
    POLICY_MISMATCH,
    SUPERSEDED_EVIDENCE,
    EvidenceFreshnessError,
    EvidenceFreshnessPolicy,
    EvidenceRecord,
    assert_evidence_freshness,
    evaluate_evidence_freshness,
)

_AS_OF = datetime(2026, 8, 12, 12, 0, tzinfo=timezone.utc)


def _policy() -> EvidenceFreshnessPolicy:
    return EvidenceFreshnessPolicy(
        policy_version="privacy-v2",
        age_limits={
            "calibration": timedelta(days=7),
            "release": timedelta(days=30),
        },
    )


def _record(
    evidence_id: str,
    *,
    evidence_type: str = "release",
    generated_at: datetime = _AS_OF - timedelta(days=1),
    policy_version: str | None = "privacy-v2",
    supersedes: str | None = None,
    superseded_by: str | None = None,
) -> EvidenceRecord:
    return EvidenceRecord(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        generated_at=generated_at,
        policy_version=policy_version,
        supersedes=supersedes,
        superseded_by=superseded_by,
    )


def test_evaluation_is_replayable_with_an_injected_clock() -> None:
    record = {
        "id": "e-current",
        "kind": "release",
        "timestamp": "2026-08-11T12:00:00Z",
        "policy": "privacy-v2",
        "ignored_payload": "synthetic-payload",
    }

    first = evaluate_evidence_freshness(record, _policy(), as_of=_AS_OF)

    class FixedClock:
        calls = 0

        def now(self) -> datetime:
            self.calls += 1
            return _AS_OF

    clock = FixedClock()
    second = evaluate_evidence_freshness(record, _policy(), clock=clock)

    assert first.passed is True
    assert second.to_dict() == first.to_dict()
    assert clock.calls == 1
    assert "e-current" not in second.to_json()
    assert "synthetic-payload" not in second.to_json()


def test_missing_future_expired_and_policy_mismatched_records_fail_closed() -> None:
    records = [
        _record(
            "e-expired",
            generated_at=_AS_OF - timedelta(days=31),
        ),
        _record(
            "e-future",
            generated_at=_AS_OF + timedelta(minutes=1),
        ),
        EvidenceRecord(
            evidence_id="e-missing-time",
            evidence_type="release",
            policy_version="privacy-v2",
        ),
        _record("e-mismatched", policy_version="privacy-v1"),
        _record("e-missing-policy", policy_version=None),
        _record("e-current"),
    ]

    report = evaluate_evidence_freshness(records, _policy(), now=_AS_OF)

    assert report.passed is False
    assert report.total_count == 6
    assert report.current_count == 1
    assert report.rejected_count == 5
    assert report.reason_counts == {
        EXPIRED_EVIDENCE: 1,
        FUTURE_TIMESTAMP: 1,
        MISSING_POLICY_VERSION: 1,
        MISSING_TIMESTAMP: 1,
        POLICY_MISMATCH: 1,
    }
    assert "e-expired" not in report.to_json()
    assert "e-mismatched" not in report.failure_message()

    with pytest.raises(EvidenceFreshnessError) as raised:
        assert_evidence_freshness(records, _policy(), as_of=_AS_OF)
    assert raised.value.report == report
    assert "e-future" not in str(raised.value)


def test_empty_evidence_is_not_a_passing_gate() -> None:
    report = evaluate_evidence_freshness([], _policy(), as_of=_AS_OF)

    assert report.passed is False
    assert report.total_count == 0
    assert report.rejected_count == 0
    assert report.reason_counts == {MISSING_EVIDENCE: 1}


def test_age_limits_are_typed_and_naive_clocks_are_rejected() -> None:
    with pytest.raises(TypeError, match="datetime.timedelta"):
        EvidenceFreshnessPolicy(
            policy_version="privacy-v2",
            age_limits={"release": 30},  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="aware datetime"):
        evaluate_evidence_freshness(
            _record("e-current"),
            _policy(),
            as_of=datetime(2026, 8, 12, 12, 0),
        )

    with pytest.raises(ValueError, match="exactly one"):
        evaluate_evidence_freshness(_record("e-current"), _policy())


def test_evidence_at_the_age_limit_is_current_and_bad_links_fail_closed() -> None:
    boundary = _record(
        "e-boundary",
        generated_at=_AS_OF - timedelta(days=30),
    )
    invalid_link = EvidenceRecord(
        evidence_id="e-invalid-link",
        evidence_type="release",
        generated_at=_AS_OF - timedelta(days=1),
        policy_version="privacy-v2",
        supersession_link="not an opaque reference",
    )

    report = evaluate_evidence_freshness(
        [boundary, invalid_link],
        _policy(),
        as_of=_AS_OF,
    )

    assert report.current_count == 1
    assert report.reason_counts == {INVALID_SUPERSESSION_LINK: 1}


def test_supersession_rejects_old_evidence_without_rejecting_its_replacement() -> None:
    old = _record("e-old")
    replacement = _record("e-new", supersedes="e-old")

    report = evaluate_evidence_freshness(
        [old, replacement],
        _policy(),
        as_of=_AS_OF,
    )

    assert report.passed is False
    assert report.current_count == 1
    assert report.reason_counts == {SUPERSEDED_EVIDENCE: 1}

    active_only = evaluate_evidence_freshness(
        [replacement],
        _policy(),
        as_of=_AS_OF,
    )
    assert active_only.passed is True

    explicitly_superseded = evaluate_evidence_freshness(
        [_record("e-old", superseded_by="e-new")],
        _policy(),
        as_of=_AS_OF,
    )
    assert explicitly_superseded.reason_counts == {SUPERSEDED_EVIDENCE: 1}


def test_mapping_aliases_and_policy_mapping_are_supported_without_payload_copying() -> (
    None
):
    policy = {
        "version": "privacy-v2",
        "limits": {"release": timedelta(days=30)},
    }
    record = {
        "record_id": "e-mapped",
        "type": "release",
        "created_at": "2026-08-01T12:00:00+00:00",
        "policy_version": "privacy-v2",
        "raw_sensitive_fixture": "synthetic-only-value",
    }

    report = evaluate_evidence_freshness(record, policy, as_of=_AS_OF)

    assert report.passed is True
    assert "synthetic-only-value" not in report.to_json()
