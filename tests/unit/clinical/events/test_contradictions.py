"""Synthetic tests for privacy-safe clinical event contradiction reports."""

from __future__ import annotations

import json

import pytest

from openmed.clinical import (
    EventInterval,
    EventStatusAssertion,
    report_event_contradictions,
)
from openmed.clinical.events.contradictions import (
    ContradictionEvidence,
    EventContradiction,
    EventContradictionReport,
)
from openmed.clinical.temporal_intervals import normalize_temporal_interval
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
    raw_event_id = "patient-jane-doe"
    raw_event_type = "patient_jane_doe"
    report = report_event_contradictions(
        [
            {
                "event_id": raw_event_id,
                "event_type": raw_event_type,
                "entity_id": "synthetic-problem",
                "start": "2026-06-01",
                "end": "2026-06-03",
                "source_offsets": [1, 8],
                "status": "active",
                "value": raw_value,
            },
            {
                "event_id": "patient-john-doe",
                "event_type": raw_event_type,
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
    assert raw_event_id not in serialized
    assert raw_event_type not in serialized


def test_typed_record_serializers_omit_caller_identifiers() -> None:
    event = EventInterval(
        event_id="patient-jane-doe",
        event_type="patient_jane_doe",
        interval_start="2026-06-01",
        interval_end="2026-06-03",
        source_start=1,
        source_end=8,
    )
    assertion = EventStatusAssertion(
        entity_id="patient-jane-doe",
        status="active",
        source_start=9,
        source_end=16,
        assertion_id="assertion-jane-doe",
        event_id="patient-jane-doe",
    )

    serialized = json.dumps(
        {"event": event.to_dict(), "assertion": assertion.to_dict()},
        sort_keys=True,
    )

    assert "jane" not in serialized
    assert event.fingerprint in serialized
    assert assertion.fingerprint in serialized


def test_normalized_day_interval_retains_offsets_for_comparison() -> None:
    first_text = "2026-06-01/2026-06-06"
    second_text = "2026-06-05/2026-06-08"
    first = normalize_temporal_interval(first_text, (0, len(first_text)))
    second = normalize_temporal_interval(second_text, (0, len(second_text)))

    report = report_event_contradictions(
        [
            {"interval": first, "entity_id": "synthetic-test", "event_type": "lab"},
            {"interval": second, "entity_id": "synthetic-test", "event_type": "lab"},
        ]
    )

    assert report.counts["overlap"] == 1
    assert report.events_checked == 2
    assert report.unresolved_intervals == ()
    assert report.contradictions[0].evidence[0].source_offsets == (0, len(first_text))
    assert first_text not in json.dumps(report.to_dict())


def test_imprecise_and_conflicting_intervals_remain_unresolved() -> None:
    month = normalize_temporal_interval("2026-06", (0, 7))
    ambiguous = normalize_temporal_interval("03/04/2026", (0, 10))
    open_end = normalize_temporal_interval("since 2026-01-01", (0, 16))

    report = report_event_contradictions(
        [
            {"interval": month, "event_type": "lab"},
            {"interval": ambiguous, "event_type": "lab"},
            {"interval": open_end, "event_type": "lab"},
        ]
    )

    assert report.events_checked == 0
    assert report.counts == {
        "conflicting_status": 0,
        "impossible_order": 0,
        "overlap": 0,
    }
    assert report.to_dict()["unresolved_interval_count"] == 3
    assert report.to_dict()["schema_version"] == 2
    assert all(
        item.fingerprint.startswith("sha256:") for item in report.unresolved_intervals
    )


def test_event_dates_require_complete_iso_day_strings() -> None:
    with pytest.raises(ValueError, match="ISO dates"):
        EventInterval(
            event_id="synthetic-event",
            event_type="lab",
            interval_start="2026-06-01 extra text",
            interval_end="2026-06-02",
        )


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_precedes_reports_only_reversed_order(reverse):
    first, second = (
        ("2026-01-02", "2026-01-01") if reverse else ("2026-01-01", "2026-01-02")
    )
    events = [
        EventInterval("a", "event", first, first, precedes=("b",)),
        EventInterval("b", "event", second, second),
    ]
    report = report_event_contradictions(events)
    assert report.counts["impossible_order"] == int(reverse)


def test_implicit_sequences_do_not_compare_different_entities():
    report = report_event_contradictions(
        [
            EventInterval(
                "a", "start", "2026-01-02", "2026-01-02", entity_id="x", sequence=1
            ),
            EventInterval(
                "b", "stop", "2026-01-01", "2026-01-01", entity_id="y", sequence=2
            ),
        ]
    )
    assert report.counts["impossible_order"] == 0


@pytest.mark.parametrize("offset", [1.5, "synthetic_private_marker"])
def test_offsets_are_exact_integers_without_sensitive_exception_context(offset):
    with pytest.raises(TypeError) as caught:
        EventInterval("a", "event", "2026-01-01", "2026-01-02", source_start=offset)
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


def test_report_metadata_is_fixed_and_value_free():
    evidence = ContradictionEvidence(0, 1, "sha256:" + "a" * 64)
    with pytest.raises(ValueError):
        EventContradiction("overlap", (evidence,), "synthetic_private_marker")
    with pytest.raises(ValueError):
        EventContradictionReport((), 0, 0, disclaimer="synthetic_private_marker")
    with pytest.raises(ValueError):
        EventContradictionReport((), True, 0)
