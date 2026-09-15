"""Focused offline tests for claim-level citation support metrics."""

from __future__ import annotations

import socket

import pytest

from openmed.eval.citation_support_metrics import (
    ADJUDICATION_CONTRADICTS,
    ADJUDICATION_IRRELEVANT,
    ADJUDICATION_SUPPORTS,
    ADJUDICATION_UNCLEAR,
    AtomicClaim,
    CitationSupportError,
    compute_citation_support_metrics,
)


def test_reports_atomic_precision_recall_orphans_and_unused_evidence() -> None:
    claims = [
        {
            "claim_id": "claim-a",
            "claim_type": "finding",
            "span": {"start": 0, "end": 12},
            "source_id": "summary-a",
            "source_length": 40,
            "citations": ["evidence-a", "evidence-b"],
        },
        {
            "claim_id": "claim-b",
            "claim_type": "finding",
            "span": {"start": 13, "end": 25},
            "source_id": "summary-a",
            "source_length": 40,
            "citations": ["evidence-c"],
        },
        {
            "claim_id": "claim-c",
            "claim_type": "finding",
            "span": {"start": 26, "end": 38},
            "source_id": "summary-a",
            "source_length": 40,
            "citations": [],
        },
    ]
    evidence = [
        {
            "evidence_id": "evidence-a",
            "source_id": "source-a",
            "span": {"start": 2, "end": 9},
            "source_length": 20,
        },
        {
            "evidence_id": "evidence-b",
            "source_id": "source-a",
            "span": {"start": 11, "end": 18},
            "source_length": 20,
        },
        {
            "evidence_id": "evidence-c",
            "source_id": "source-b",
            "span": {"start": 4, "end": 10},
            "source_length": 20,
        },
        {
            "evidence_id": "evidence-unused",
            "source_id": "source-b",
            "span": {"start": 12, "end": 17},
            "source_length": 20,
        },
    ]
    adjudications = [
        {
            "claim_id": "claim-a",
            "evidence_id": "evidence-a",
            "label": ADJUDICATION_SUPPORTS,
        },
        {
            "claim_id": "claim-a",
            "evidence_id": "evidence-b",
            "label": ADJUDICATION_CONTRADICTS,
        },
        {
            "claim_id": "claim-b",
            "evidence_id": "evidence-c",
            "label": ADJUDICATION_IRRELEVANT,
        },
    ]

    report = compute_citation_support_metrics(
        claims, evidence, adjudications=adjudications
    )

    assert report.claim_count == 3
    assert report.evidence_count == 4
    assert report.citation_count == 3
    assert report.orphan_claim_count == 1
    assert report.orphan_claim_rate == pytest.approx(1 / 3)
    assert report.unused_evidence_count == 1
    assert report.unused_evidence_rate == pytest.approx(1 / 4)
    assert report.deterministic.valid_citation_count == 3
    assert report.deterministic.passed is True
    assert report.citation_precision == pytest.approx(1 / 3)
    assert report.support_recall == pytest.approx(1 / 3)
    assert report.adjudication.supported_claim_count == 1
    assert report.adjudication.adjudicated_claim_count == 2


def test_span_checks_are_deterministic_and_do_not_claim_semantic_support() -> None:
    claims = [
        {
            "claim_id": "claim-a",
            "span": {"start": 0, "end": 10},
            "source_id": "summary-a",
            "source_length": 10,
            "citations": ["evidence-a", "evidence-missing"],
        },
        {
            "claim_id": "claim-b",
            "span": {"start": 10, "end": 9},
            "source_id": "summary-a",
            "source_length": 10,
        },
    ]
    evidence = [
        {
            "evidence_id": "evidence-a",
            "source_id": "source-a",
            "span": {"start": 2, "end": 6},
            "source_length": 10,
        }
    ]

    report = compute_citation_support_metrics(claims, evidence)

    assert report.citation_precision is None
    assert report.support_recall is None
    assert report.deterministic.invalid_claim_span_count == 1
    assert report.deterministic.valid_citation_count == 1
    assert report.deterministic.invalid_citation_reasons == {"missing_evidence": 1}
    assert report.deterministic.passed is False


def test_explicit_citation_records_and_compact_mapping_inputs_are_supported() -> None:
    report = compute_citation_support_metrics(
        {"claim-a": (0, 5), "claim-b": (5, 9)},
        {"evidence-a": (2, 7)},
        {"claim-a": ["evidence-a"]},
        {
            "claim-a": {
                "evidence_id": "evidence-a",
                "label": ADJUDICATION_SUPPORTS,
            }
        },
    )

    assert report.claim_count == 2
    assert report.evidence_count == 1
    assert report.citation_count == 1
    assert report.orphan_claim_count == 1
    assert report.citation_precision == 1.0
    assert report.support_recall == 0.5


def test_report_serialization_is_order_independent_and_value_free() -> None:
    claims = [
        {
            "claim_id": "claim-b",
            "span": {"start": 5, "end": 9},
            "citations": ["evidence-b"],
            "text": "sensitive-claim-value",
        },
        {
            "claim_id": "claim-a",
            "span": {"start": 0, "end": 4},
            "citations": ["evidence-a"],
            "text": "another-sensitive-claim-value",
        },
    ]
    evidence = [
        {
            "evidence_id": "evidence-a",
            "span": {"start": 0, "end": 4},
            "source_text": "sensitive-source-value",
        },
        {
            "evidence_id": "evidence-b",
            "span": {"start": 5, "end": 9},
            "source_text": "another-sensitive-source-value",
        },
    ]
    adjudications = [
        {
            "claim_id": "claim-a",
            "evidence_id": "evidence-a",
            "label": ADJUDICATION_SUPPORTS,
        },
        {
            "claim_id": "claim-b",
            "evidence_id": "evidence-b",
            "label": ADJUDICATION_UNCLEAR,
        },
    ]

    first = compute_citation_support_metrics(
        claims, evidence, adjudications=adjudications
    )
    second = compute_citation_support_metrics(
        list(reversed(claims)),
        list(reversed(evidence)),
        adjudications=list(reversed(adjudications)),
    )

    assert first.to_json() == second.to_json()
    serialized = first.to_json() + first.to_markdown() + repr(first)
    for sensitive_value in (
        "sensitive-claim-value",
        "another-sensitive-claim-value",
        "sensitive-source-value",
        "claim-a",
        "evidence-a",
    ):
        assert sensitive_value not in serialized
    assert first.to_dict()["human_review_required"] is True
    assert first.to_dict()["provenance"]["input_digest"].startswith("sha256:")


def test_no_network_is_required_and_invalid_labels_are_value_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    with pytest.raises(CitationSupportError, match="adjudication label") as error:
        compute_citation_support_metrics(
            [AtomicClaim("claim-safe")],
            [],
            adjudications=[
                {
                    "claim_id": "claim-safe",
                    "evidence_id": "evidence-safe",
                    "label": "sensitive-review-value",
                }
            ],
        )
    assert "sensitive-review-value" not in str(error.value)


def test_duplicate_edges_are_counted_once_and_conflicting_review_fails_closed() -> None:
    report = compute_citation_support_metrics(
        [
            {
                "claim_id": "claim-a",
                "span": {"start": 0, "end": 5},
                "citations": ["evidence-a", "evidence-a"],
            }
        ],
        [{"evidence_id": "evidence-a", "span": {"start": 0, "end": 5}}],
        adjudications=[
            {
                "claim_id": "claim-a",
                "evidence_id": "evidence-a",
                "label": ADJUDICATION_SUPPORTS,
            },
            {
                "claim_id": "claim-a",
                "evidence_id": "evidence-a",
                "label": ADJUDICATION_CONTRADICTS,
            },
        ],
    )

    assert report.citation_count == 1
    assert report.deterministic.duplicate_citation_count == 1
    assert report.adjudication.conflicting_adjudication_count == 1
    assert report.adjudication.unclear_citation_count == 1
    assert report.citation_precision == 0.0
