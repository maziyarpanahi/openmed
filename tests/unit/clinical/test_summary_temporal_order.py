"""Tests for evidence-bound temporal ordering of summary claims."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.relations.temporal import (
    TemporalCueReference,
    TemporalRelationCandidate,
    TemporalSpanReference,
)
from openmed.clinical.summary_temporal_order import (
    SUMMARY_TEMPORAL_ORDER_ADVISORY,
    validate_summary_temporal_order,
)
from openmed.clinical.timeline import order_events
from openmed.clinical.timeline.resolver import Timeline
from openmed.core.audit import hash_text


def _timeline() -> Timeline:
    text = "Alpha before Bravo before Charlie."
    references = []
    spans = []
    for event_id, value in (
        ("private-alpha", "Alpha"),
        ("private-bravo", "Bravo"),
        ("private-charlie", "Charlie"),
    ):
        start = text.index(value)
        reference = TemporalSpanReference(
            span_id=event_id,
            label="EVENT",
            role="EVENT",
            start=start,
            end=start + len(value),
            score=1.0,
            text_hash=hash_text(value),
        )
        references.append(reference)
        spans.append(
            {
                "id": event_id,
                "label": "EVENT",
                "role": "EVENT",
                "start": reference.start,
                "end": reference.end,
            }
        )

    candidates = []
    for source, target in zip(references, references[1:]):
        cue_start = text.index("before", source.end)
        candidates.append(
            TemporalRelationCandidate(
                relation_type="BEFORE",
                source=source,
                target=target,
                confidence=1.0,
                cue=TemporalCueReference(
                    category="BEFORE",
                    start=cue_start,
                    end=cue_start + len("before"),
                    text_hash=hash_text("before"),
                ),
            )
        )
    return order_events(text, spans, tlink_candidates=candidates)


def test_validates_claim_order_using_transitive_timeline_evidence() -> None:
    result = validate_summary_temporal_order(
        ("private-alpha", "private-bravo", "private-charlie"),
        _timeline(),
    )

    assert result.status == "validated"
    assert result.review_required is False
    assert result.findings == ()
    assert result.reference_count == 3
    assert result.advisory == SUMMARY_TEMPORAL_ORDER_ADVISORY


def test_routes_proven_inversion_to_review_without_claim_text() -> None:
    result = validate_summary_temporal_order(
        ("private-charlie", "private-alpha"),
        _timeline(),
    )

    assert result.status == "review_required"
    assert result.findings[0].code == "order_inversion"
    assert result.findings[0].to_dict() == {
        "code": "order_inversion",
        "reference_indexes": [0, 1],
    }
    encoded = result.to_json()
    assert "private-charlie" not in encoded
    assert "private-alpha" not in encoded


def test_unlinked_timeline_events_are_unresolved_not_source_ordered() -> None:
    timeline = _timeline()
    unlinked = Timeline(events=timeline.events, edges=())

    result = validate_summary_temporal_order(
        ("private-alpha", "private-bravo"),
        unlinked,
    )

    assert [finding.code for finding in result.findings] == ["order_unresolved"]
    assert result.review_required is True


@pytest.mark.parametrize(
    ("references", "expected_code"),
    [
        (("private-alpha",), "insufficient_references"),
        (("private-alpha", "private-alpha"), "duplicate_reference"),
        (("private-alpha", "private-missing"), "unknown_event"),
    ],
)
def test_ambiguous_reference_sets_route_to_review(
    references: tuple[str, ...], expected_code: str
) -> None:
    result = validate_summary_temporal_order(references, _timeline())

    assert result.status == "review_required"
    assert result.findings[0].code == expected_code


def test_result_is_deterministic_and_value_free() -> None:
    references = ("private-charlie", "private-missing", "private-alpha")
    first = validate_summary_temporal_order(references, _timeline()).to_json()
    second = validate_summary_temporal_order(references, _timeline()).to_json()

    assert first == second
    assert json.loads(first)["findings"] == [
        {"code": "unknown_event", "reference_indexes": [0, 1]},
        {"code": "order_inversion", "reference_indexes": [0, 2]},
        {"code": "unknown_event", "reference_indexes": [1, 2]},
    ]
    assert all(reference not in first for reference in references)


def test_validation_needs_no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr("socket.socket", fail_network)

    assert (
        validate_summary_temporal_order(
            ("private-alpha", "private-bravo"),
            _timeline(),
        ).status
        == "validated"
    )


def test_invalid_identifier_error_does_not_echo_sensitive_input() -> None:
    sensitive = "private-patient-event"

    with pytest.raises(ValueError) as error:
        validate_summary_temporal_order((sensitive, ""), _timeline())

    assert sensitive not in str(error.value)
