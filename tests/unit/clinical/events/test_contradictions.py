"""Synthetic tests for privacy-safe clinical event contradiction reports."""

from __future__ import annotations

import json

from openmed.clinical import (
    EventInterval,
    EventStatusAssertion,
    report_event_contradictions,
)
from openmed.core.audit import hash_text


def test_overlap_report_is_deterministic_and_privacy_safe() -> None:
    events = [
        {
            "event_id": "event-b",
            "event_type": "medication_change",
            "entity_id": "synthetic-medication-1",
            "interval": {"start": "2026-06-05", "end": "2026-06-08"},
            "source_offsets": [42, 58],
            "text": "synthetic medication beta",
        },
        {
            "event_id": "event-a",
            "event_type": "medication_change",
            "entity_id": "synthetic-medication-1",
            "interval": {"start": "2026-06-01", "end": "2026-06-06"},
            "source_offsets": [4, 22],
            "text": "synthetic medication alpha",
        },
    ]

    baseline = report_event_contradictions(events)
    reordered = report_event_contradictions(reversed(events))

    assert baseline.to_dict() == reordered.to_dict()
    assert baseline.counts == {
        "conflicting_status": 0,
        "impossible_order": 0,
        "overlap": 1,
    }
    serialized = json.dumps(baseline.to_dict(), sort_keys=True)
    assert "synthetic medication alpha" not in serialized
    assert "synthetic medication beta" not in serialized
    assert "2026-06-01" not in serialized
    evidence = baseline.contradictions[0].evidence
    assert [(item.source_start, item.source_end) for item in evidence] == [
        (4, 22),
        (42, 58),
    ]
    assert all(item.fingerprint.startswith("sha256:") for item in evidence)


def test_invalid_interval_is_reported_as_impossible_order() -> None:
    report = report_event_contradictions(
        [
            EventInterval(
                event_id="event-invalid",
                event_type="lab_observation",
                interval_start="2026-06-10",
                interval_end="2026-06-09",
                source_start=10,
                source_end=25,
                fingerprint=hash_text("synthetic invalid interval"),
            )
        ]
    )

    assert report.counts["impossible_order"] == 1
    assert report.contradictions[0].right is None
    assert report.contradictions[0].reason == "typed event ordering is impossible"


def test_sequence_and_typed_start_end_order_are_review_signals() -> None:
    sequence_report = report_event_contradictions(
        [
            EventInterval(
                event_id="sequence-first",
                event_type="checkpoint",
                interval_start="2026-06-12",
                interval_end="2026-06-12",
                entity_id="synthetic-entity",
                sequence=1,
                source_start=0,
                source_end=5,
            ),
            EventInterval(
                event_id="sequence-second",
                event_type="checkpoint",
                interval_start="2026-06-01",
                interval_end="2026-06-01",
                entity_id="synthetic-entity",
                sequence=2,
                source_start=6,
                source_end=12,
            ),
        ]
    )
    typed_report = report_event_contradictions(
        [
            EventInterval(
                event_id="synthetic-admission",
                event_type="admission",
                interval_start="2026-06-20",
                interval_end="2026-06-20",
                source_start=0,
                source_end=9,
            ),
            EventInterval(
                event_id="synthetic-discharge",
                event_type="discharge",
                interval_start="2026-06-10",
                interval_end="2026-06-10",
                source_start=10,
                source_end=19,
            ),
        ]
    )

    assert sequence_report.counts["impossible_order"] == 1
    assert typed_report.counts["impossible_order"] == 1


def test_conflicting_status_is_scoped_to_same_overlapping_entity() -> None:
    report = report_event_contradictions(
        [],
        [
            EventStatusAssertion(
                entity_id="synthetic-condition",
                status="active",
                source_start=2,
                source_end=10,
                fingerprint=hash_text("synthetic status active"),
                interval_start="2026-06-01",
                interval_end="2026-06-05",
            ),
            EventStatusAssertion(
                entity_id="synthetic-condition",
                status="resolved",
                source_start=20,
                source_end=30,
                fingerprint=hash_text("synthetic status resolved"),
                interval_start="2026-06-04",
                interval_end="2026-06-08",
            ),
            EventStatusAssertion(
                entity_id="synthetic-condition",
                status="refuted",
                source_start=40,
                source_end=49,
                fingerprint=hash_text("synthetic status refuted"),
                interval_start="2026-07-01",
                interval_end="2026-07-02",
            ),
            EventStatusAssertion(
                entity_id="other-synthetic-condition",
                status="active",
                source_start=50,
                source_end=58,
                fingerprint=hash_text("synthetic other status"),
                interval_start="2026-06-04",
                interval_end="2026-06-08",
            ),
        ],
    )

    assert report.status_assertions_checked == 4
    assert report.counts == {
        "conflicting_status": 1,
        "impossible_order": 0,
        "overlap": 0,
    }
    assert {item.status for item in report.contradictions[0].evidence} == {
        "active",
        "inactive",
    }


def test_mapping_status_is_reported_without_retaining_raw_value() -> None:
    raw_value = "synthetic sensitive assertion"
    report = report_event_contradictions(
        [
            {
                "event_id": "status-event-a",
                "event_type": "problem",
                "entity_id": "synthetic-problem",
                "start": "2026-06-01",
                "end": "2026-06-03",
                "source_offsets": [1, 8],
                "status": "active",
                "value": raw_value,
            },
            {
                "event_id": "status-event-b",
                "event_type": "problem",
                "entity_id": "synthetic-problem",
                "start": "2026-06-02",
                "end": "2026-06-04",
                "source_offsets": [9, 16],
                "status": "refuted",
                "value": "synthetic conflicting assertion",
            },
        ]
    )

    serialized = json.dumps(report.to_dict(), sort_keys=True)
    assert report.counts["overlap"] == 1
    assert report.counts["conflicting_status"] == 1
    assert raw_value not in serialized
    assert "synthetic conflicting assertion" not in serialized
