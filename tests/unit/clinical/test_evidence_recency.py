"""Focused synthetic tests for guarded evidence recency labels."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from openmed.clinical.evidence_recency import (
    EVIDENCE_RECENCY_CURRENT_LABEL,
    EVIDENCE_RECENCY_FUTURE_DATED_LABEL,
    EVIDENCE_RECENCY_SCHEMA_VERSION,
    EVIDENCE_RECENCY_STALE_LABEL,
    EVIDENCE_RECENCY_UNKNOWN_LABEL,
    EvidenceRecencyError,
    EvidenceRecencyLabel,
    EvidenceRecencyPolicy,
    assess_evidence_recency,
    build_evidence_recency_report,
    classify_evidence_recency,
    render_evidence_recency_report,
)

_AS_OF = "2026-02-01T00:00:00Z"
_RAW_MARKER = "synthetic-patient-identifier-42"


def test_labels_cover_current_stale_future_and_unknown() -> None:
    policy = EvidenceRecencyPolicy(current_window=timedelta(days=30))

    assert classify_evidence_recency("2026-01-15T00:00:00Z", _AS_OF) is (
        EvidenceRecencyLabel.CURRENT
    )
    assert classify_evidence_recency("2026-01-02T00:00:00Z", _AS_OF) is (
        EvidenceRecencyLabel.CURRENT
    )
    assert (
        classify_evidence_recency("2026-01-01T00:00:00Z", _AS_OF, policy=policy)
        is EvidenceRecencyLabel.STALE
    )
    assert classify_evidence_recency("2026-02-02T00:00:00Z", _AS_OF) is (
        EvidenceRecencyLabel.FUTURE_DATED
    )
    assert classify_evidence_recency(None, _AS_OF) is EvidenceRecencyLabel.UNKNOWN
    assert classify_evidence_recency("not-a-trustworthy-timestamp", _AS_OF) is (
        EvidenceRecencyLabel.UNKNOWN
    )


def test_timezone_normalization_and_configurable_boundaries_are_deterministic() -> None:
    policy = EvidenceRecencyPolicy.from_value(
        {
            "stale_after_days": 7,
            "future_tolerance_seconds": 3600,
        }
    )

    # 2026-01-31T23:30:00+01:00 is 30 minutes before the reference time in
    # UTC, so the configured one-hour future tolerance is not used here.
    assert (
        classify_evidence_recency(
            "2026-02-01T00:30:00+01:00",
            _AS_OF,
            policy=policy,
        )
        is EvidenceRecencyLabel.CURRENT
    )
    assert (
        classify_evidence_recency(
            "2026-02-01T01:30:01Z",
            _AS_OF,
            policy=policy,
        )
        is EvidenceRecencyLabel.FUTURE_DATED
    )
    assert (
        classify_evidence_recency(
            "2026-01-24T00:00:00Z",
            _AS_OF,
            policy=policy,
        )
        is EvidenceRecencyLabel.STALE
    )
    assert policy.to_dict() == {
        "current_window_seconds": 604800,
        "future_tolerance_seconds": 3600,
        "schema_version": EVIDENCE_RECENCY_SCHEMA_VERSION,
    }


def test_missing_reference_time_is_explicitly_unknown_without_clock_fallback() -> None:
    result = assess_evidence_recency("2026-01-31T00:00:00Z")

    assert result.label is EvidenceRecencyLabel.UNKNOWN
    assert result.known is False
    assert result.review_required is True
    assert result.to_dict() == {
        "label": EVIDENCE_RECENCY_UNKNOWN_LABEL,
        "review_required": True,
        "schema_version": EVIDENCE_RECENCY_SCHEMA_VERSION,
    }


def test_report_labels_mapping_items_without_retaining_raw_values() -> None:
    evidence = [
        {
            "evidence_timestamp": "2026-01-15T00:00:00Z",
            "claim": _RAW_MARKER,
        },
        {
            "timestamp": "2025-12-31T00:00:00Z",
            "identifier": _RAW_MARKER,
        },
        {"claim": _RAW_MARKER},
        {"occurred_at": "2026-02-02T00:00:00Z"},
    ]

    report = build_evidence_recency_report(evidence, _AS_OF)
    rendered_json = report.to_json()
    rendered_markdown = report.to_markdown()

    assert [label.value for label in report.labels] == [
        EVIDENCE_RECENCY_CURRENT_LABEL,
        EVIDENCE_RECENCY_STALE_LABEL,
        EVIDENCE_RECENCY_UNKNOWN_LABEL,
        EVIDENCE_RECENCY_FUTURE_DATED_LABEL,
    ]
    assert report.label_counts == {
        EVIDENCE_RECENCY_CURRENT_LABEL: 1,
        EVIDENCE_RECENCY_STALE_LABEL: 1,
        EVIDENCE_RECENCY_FUTURE_DATED_LABEL: 1,
        EVIDENCE_RECENCY_UNKNOWN_LABEL: 1,
    }
    assert report.review_required_count == 3
    assert _RAW_MARKER not in repr(report)
    assert _RAW_MARKER not in rendered_json
    assert _RAW_MARKER not in rendered_markdown
    assert "2026-01-15" not in rendered_json
    assert "2026-01-15" not in rendered_markdown


def test_report_json_is_stable_and_contains_only_controlled_metadata() -> None:
    evidence = [
        datetime(2026, 1, 15, tzinfo=timezone.utc),
        datetime(2026, 2, 2, tzinfo=timezone.utc),
        None,
    ]
    first = build_evidence_recency_report(evidence, _AS_OF)
    second = build_evidence_recency_report(tuple(evidence), _AS_OF)

    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()
    assert set(first.to_dict()) == {
        "disclaimer",
        "label_counts",
        "policy",
        "record_count",
        "records",
        "review_required_count",
        "schema_version",
    }
    assert render_evidence_recency_report(evidence, _AS_OF, format="json") == (
        first.to_json()
    )


def test_invalid_policy_and_collection_errors_do_not_echo_sensitive_values() -> None:
    with pytest.raises(EvidenceRecencyError) as policy_error:
        EvidenceRecencyPolicy.from_value(
            {"unsupported": _RAW_MARKER},  # type: ignore[dict-item]
        )
    assert _RAW_MARKER not in str(policy_error.value)

    class BrokenCollection:
        def __iter__(self):
            raise RuntimeError(_RAW_MARKER)

    with pytest.raises(EvidenceRecencyError) as collection_error:
        build_evidence_recency_report(BrokenCollection(), _AS_OF)
    assert _RAW_MARKER not in str(collection_error.value)


@pytest.mark.parametrize(
    "value",
    [
        {"current_window_days": -1},
        {"current_window_seconds": float("inf")},
        {"future_tolerance": timedelta(days=-1)},
        {"schema_version": 2},
    ],
)
def test_invalid_policy_values_fail_closed(value: object) -> None:
    with pytest.raises(EvidenceRecencyError):
        EvidenceRecencyPolicy.from_value(value)
