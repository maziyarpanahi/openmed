"""Synthetic offline tests for SDOH review routing."""

from __future__ import annotations

import json
import socket

import pytest

from openmed.clinical.sdoh_deduplicate import SDOHSourceReference
from openmed.clinical.sdoh_review import (
    SDOHOutcome,
    SDOHReviewEvidence,
    SDOHReviewState,
    route_sdoh_review,
)


def _evidence(
    evidence_id: str,
    outcome: SDOHOutcome,
    confidence: float = 0.95,
    *,
    conflicting: bool = False,
) -> SDOHReviewEvidence:
    return SDOHReviewEvidence(
        evidence_id=evidence_id,
        category="transportation",
        outcome=outcome,
        confidence=confidence,
        source_references=(
            SDOHSourceReference(
                source_id="document-local-1",
                version_id="v1",
                start=2,
                end=9,
            ),
        ),
        conflicting=conflicting,
    )


def test_low_confidence_and_conflicting_evidence_routes_with_sources() -> None:
    queue = route_sdoh_review(
        (
            _evidence(
                "evidence-1",
                SDOHOutcome.AFFIRMED,
                confidence=0.4,
                conflicting=True,
            ),
        )
    )

    assert len(queue.items) == 1
    item = queue.items[0]
    assert item.states == (
        SDOHReviewState.CONFLICTING,
        SDOHReviewState.LOW_CONFIDENCE,
    )
    assert item.source_references[0].source_id == "document-local-1"
    assert item.to_dict()["automated_decision_allowed"] is False


@pytest.mark.parametrize(
    ("outcome", "state"),
    (
        (SDOHOutcome.UNKNOWN, SDOHReviewState.UNKNOWN),
        (SDOHOutcome.DECLINED, SDOHReviewState.DECLINED),
        (SDOHOutcome.REFUSED, SDOHReviewState.REFUSED),
    ),
)
def test_non_answer_outcomes_are_preserved_and_routed(
    outcome: SDOHOutcome,
    state: SDOHReviewState,
) -> None:
    item = route_sdoh_review((_evidence("evidence-1", outcome),)).items[0]

    assert item.outcome is outcome
    assert state in item.states


def test_high_confidence_uncontested_evidence_is_not_queued() -> None:
    queue = route_sdoh_review(
        (_evidence("evidence-1", SDOHOutcome.NEGATED, confidence=0.95),)
    )

    assert queue.items == ()
    assert queue.summary().total_count == 0


def test_queue_summary_contains_counts_only() -> None:
    queue = route_sdoh_review(
        (
            _evidence("evidence-secret-1", SDOHOutcome.UNKNOWN),
            _evidence("evidence-secret-2", SDOHOutcome.DECLINED),
        )
    )
    rendered = json.dumps(queue.summary().to_dict(), sort_keys=True)

    assert "evidence-secret" not in rendered
    assert "document-local" not in rendered
    assert queue.summary().to_dict() == {
        "total_count": 2,
        "state_counts": {"declined": 1, "unknown": 1},
        "category_counts": {"transportation": 2},
    }


def test_review_routing_is_deterministic() -> None:
    records = (
        _evidence("evidence-2", SDOHOutcome.DECLINED),
        _evidence("evidence-1", SDOHOutcome.UNKNOWN),
    )

    assert (
        route_sdoh_review(records).to_dict()
        == route_sdoh_review(reversed(records)).to_dict()
    )


def test_review_routing_performs_no_network_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", reject_network)

    assert route_sdoh_review((_evidence("evidence-1", SDOHOutcome.UNKNOWN),)).items
