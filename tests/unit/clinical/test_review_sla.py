"""Tests for deterministic, PHI-safe human-review queue SLA reports."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from openmed.clinical import (
    ReviewQueueCase,
    build_review_sla_report,
    compute_review_sla,
    opaque_case_key,
)

AS_OF = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_report_computes_age_priority_expiry_and_overdue_buckets() -> None:
    cases = [
        {
            "case_key": "synthetic-case-a",
            "queued_at": AS_OF - timedelta(minutes=30),
            "priority": "urgent",
        },
        {
            "case_key": "synthetic-case-b",
            "queued_at": AS_OF - timedelta(hours=2),
            "priority": "high",
        },
        {
            "case_key": "synthetic-case-c",
            "queued_at": AS_OF - timedelta(hours=26),
            "priority": "normal",
        },
        {
            "case_key": "synthetic-case-d",
            "queued_at": AS_OF - timedelta(hours=100),
            "priority": "low",
        },
    ]

    report = build_review_sla_report(cases, now=AS_OF)

    assert report.to_dict() == {
        "schema_version": "review-sla.v1",
        "as_of": "2026-08-11T12:00:00+00:00",
        "total_cases": 4,
        "priority_counts": {"urgent": 1, "high": 1, "normal": 1, "low": 1},
        "age_counts": {"0-1h": 1, "1-4h": 1, "4-24h": 0, "24h+": 2},
        "expiry_counts": {
            "expired": 2,
            "due-within-4h": 1,
            "due-after-4h": 1,
        },
        "overdue_counts": {
            "on-time": 2,
            "0-24h-overdue": 1,
            "24h+-overdue": 1,
        },
    }


def test_explicit_expiry_and_clock_callable_are_deterministic() -> None:
    calls: list[int] = []

    def clock() -> datetime:
        calls.append(1)
        return AS_OF

    report = build_review_sla_report(
        [
            ReviewQueueCase(
                case_key="synthetic-expiry-boundary",
                queued_at=AS_OF - timedelta(hours=1),
                priority="p1",
                expires_at=AS_OF,
            )
        ],
        clock=clock,
    )

    assert calls == [1]
    assert report.overdue_counts["on-time"] == 1
    assert report.expiry_counts["due-within-4h"] == 1


def test_report_and_records_are_stable_across_input_orderings_and_hide_keys() -> None:
    cases = [
        {
            "case_key": "synthetic-case-a",
            "queued_at": AS_OF - timedelta(hours=2),
            "priority": "high",
        },
        {
            "case_key": "synthetic-case-b",
            "queued_at": AS_OF - timedelta(minutes=5),
            "priority": "urgent",
        },
    ]

    first_report = build_review_sla_report(cases, now=AS_OF)
    second_report = build_review_sla_report(tuple(reversed(cases)), now=AS_OF)
    first_records = compute_review_sla(cases, now=AS_OF)
    second_records = compute_review_sla(tuple(reversed(cases)), now=AS_OF)

    assert first_report.to_json() == second_report.to_json()
    assert first_records == second_records
    assert all(record.case_key.startswith("sha256:") for record in first_records)

    serialized_report = first_report.to_json()
    serialized_records = json.dumps([record.to_dict() for record in first_records])
    for raw_key in ("synthetic-case-a", "synthetic-case-b"):
        assert raw_key not in serialized_report
        assert raw_key not in serialized_records
    assert "case_key" not in serialized_report


def test_opaque_keys_are_stable_without_containing_the_input() -> None:
    first = opaque_case_key("synthetic-case-a")
    second = opaque_case_key("synthetic-case-a")

    assert first == second
    assert first.startswith("sha256:")
    assert "synthetic-case-a" not in first


def test_custom_sla_duration_and_mapping_aliases_are_supported() -> None:
    record = compute_review_sla(
        [
            {
                "case_id": "synthetic-custom-sla",
                "enqueued_at": AS_OF - timedelta(hours=2),
                "priority": "standard",
                "sla": timedelta(hours=1),
            }
        ],
        now=AS_OF,
        priority_sla={"p0": timedelta(hours=2)},
    )[0]

    assert record.priority == "normal"
    assert record.age_bucket == "1-4h"
    assert record.expiry_bucket == "expired"
    assert record.overdue_bucket == "0-24h-overdue"
    assert record.overdue_seconds == 3600


def test_clock_is_required_and_invalid_inputs_do_not_echo_case_values() -> None:
    with pytest.raises(ValueError, match="injected clock"):
        compute_review_sla([])

    sensitive_fixture_key = "synthetic-private-fixture-key"
    with pytest.raises(ValueError) as error:
        compute_review_sla([{"case_key": sensitive_fixture_key}], now=AS_OF)

    assert sensitive_fixture_key not in str(error.value)


def test_invalid_queue_entries_and_duplicate_keys_are_rejected() -> None:
    with pytest.raises(ValueError, match="queued_at"):
        compute_review_sla([{"case_key": "synthetic-missing-time"}], now=AS_OF)

    case = {
        "case_key": "synthetic-duplicate",
        "queued_at": AS_OF,
    }
    with pytest.raises(ValueError, match="unique"):
        compute_review_sla([case, dict(case)], now=AS_OF)


def test_naive_clock_values_are_normalized_to_utc() -> None:
    report = build_review_sla_report(
        [{"case_key": "synthetic-naive", "queued_at": AS_OF.replace(tzinfo=None)}],
        now=AS_OF.replace(tzinfo=None),
    )

    assert report.as_of == "2026-08-11T12:00:00+00:00"
