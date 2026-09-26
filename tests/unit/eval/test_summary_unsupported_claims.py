"""Focused offline tests for summary unsupported-claim scoring."""

from __future__ import annotations

import socket

import pytest

from openmed.eval.summary_unsupported_claims import (
    APPROVED_EVIDENCE_DISCLAIMER,
    CONTRADICTED,
    SUPPORTED,
    UNCITED,
    UNRESOLVED,
    ClaimState,
    score_summary_claim,
    score_summary_claims,
)


def test_scores_four_states_and_rates_by_claim_class() -> None:
    claims = [
        {
            "claim_id": "claim-diagnosis-supported",
            "claim_class": "diagnosis",
            "claim_key": "synthetic-diagnosis-a",
            "evidence_ids": ["evidence-a"],
        },
        {
            "claim_id": "claim-diagnosis-contradicted",
            "claim_class": "diagnosis",
            "claim_key": "synthetic-diagnosis-b",
            "evidence_ids": ["evidence-b"],
        },
        {
            "claim_id": "claim-medication-unresolved",
            "claim_class": "medication",
            "claim_key": "synthetic-medication-a",
            "evidence_ids": ["evidence-c"],
        },
        {
            "claim_id": "claim-medication-uncited",
            "claim_class": "medication",
            "claim_key": "synthetic-medication-b",
        },
    ]
    evidence = [
        {
            "evidence_id": "evidence-a",
            "claim_key": "synthetic-diagnosis-a",
            "claim_class": "diagnosis",
            "approved": True,
            "relation": "supports",
        },
        {
            "evidence_id": "evidence-b",
            "claim_key": "synthetic-diagnosis-b",
            "claim_class": "diagnosis",
            "approved": True,
            "relation": "contradicts",
        },
        {
            "evidence_id": "evidence-c",
            "claim_key": "synthetic-medication-a",
            "claim_class": "medication",
            "approved": False,
            "relation": "supports",
        },
    ]

    report = score_summary_claims(claims, evidence, n_resamples=200, seed=7)

    assert report.state_counts == {
        SUPPORTED: 1,
        CONTRADICTED: 1,
        UNRESOLVED: 1,
        UNCITED: 1,
    }
    assert report.unsupported_count == 3
    assert report.unsupported_rate == pytest.approx(0.75)
    assert report.evidence_count == 3
    assert report.approved_evidence_count == 2
    assert report.evidence_digest.startswith("sha256:")
    assert report.review_required_count == 3
    assert report.by_claim_class["diagnosis"].unsupported_rate == pytest.approx(0.5)
    assert report.by_claim_class["medication"].unsupported_rate == 1.0
    assert report.overall.bootstrap_ci.n_resamples == 200
    assert report.overall.bootstrap_ci.lower <= report.unsupported_rate
    assert report.unsupported_rate <= report.overall.bootstrap_ci.upper


def test_scoring_and_serialization_are_order_independent_and_value_free() -> None:
    claims = [
        {
            "claim_id": "claim-two",
            "claim_class": "lab",
            "claim_key": "synthetic-lab-two",
            "evidence_ids": ["evidence-two"],
        },
        {
            "claim_id": "claim-one",
            "claim_class": "lab",
            "claim_key": "synthetic-lab-one",
            "evidence_ids": ["evidence-one"],
        },
    ]
    evidence = [
        {
            "evidence_id": "evidence-one",
            "claim_key": "synthetic-lab-one",
            "relation": "supported",
        },
        {
            "evidence_id": "evidence-two",
            "claim_key": "synthetic-lab-two",
            "relation": "contradicted",
        },
    ]

    first = score_summary_claims(claims, evidence, n_resamples=100, seed=11)
    second = score_summary_claims(
        list(reversed(claims)),
        list(reversed(evidence)),
        n_resamples=100,
        seed=11,
    )

    assert first.to_json() == second.to_json()
    payload = first.to_json()
    for sensitive_value in (
        "synthetic-lab-one",
        "synthetic-lab-two",
        "claim-one",
        "evidence-one",
    ):
        assert sensitive_value not in payload
    assert "synthetic-lab-one" not in first.to_markdown()
    assert first.to_benchmark_report().to_dict()["fixture_count"] == 2
    assert first.to_dict()["human_review_required"] is True


def test_invalid_or_conflicting_citations_fail_closed_to_unresolved() -> None:
    cases = [
        (
            "missing-evidence",
            ["does-not-exist"],
            [],
        ),
        (
            "mismatched-key",
            ["evidence-1"],
            [
                {
                    "evidence_id": "evidence-1",
                    "claim_key": "different-synthetic-key",
                    "relation": "supports",
                }
            ],
        ),
        (
            "unknown-relation",
            ["evidence-1"],
            [{"evidence_id": "evidence-1", "relation": "pending"}],
        ),
        (
            "conflicting-evidence",
            ["evidence-1", "evidence-2"],
            [
                {"evidence_id": "evidence-1", "relation": "supports"},
                {"evidence_id": "evidence-2", "relation": "contradicts"},
            ],
        ),
    ]

    for claim_id, evidence_ids, evidence in cases:
        claim = {
            "claim_id": claim_id,
            "claim_class": "finding",
            "claim_key": "synthetic-finding",
            "evidence_ids": evidence_ids,
        }
        assert score_summary_claim(claim, evidence).state == UNRESOLVED


def test_explicit_citation_link_can_score_without_repeating_claim_key() -> None:
    assessment = score_summary_claim(
        {
            "claim_id": "claim-linked",
            "claim_class": "procedure",
            "claim_key": "synthetic-procedure",
            "evidence_ids": ["evidence-linked"],
        },
        [
            {
                "evidence_id": "evidence-linked",
                "claim_id": "claim-linked",
                "relation": ClaimState.SUPPORTED,
            }
        ],
    )

    assert assessment.state == SUPPORTED
    assert assessment.review_required is False


def test_empty_and_single_claim_inputs_have_truthful_degenerate_intervals() -> None:
    empty = score_summary_claims([], [], n_resamples=25)
    single = score_summary_claims(
        [{"claim_id": "claim-empty", "claim_class": "other"}],
        [],
        n_resamples=25,
    )

    assert empty.unsupported_rate == 0.0
    assert empty.overall.bootstrap_ci.degenerate is True
    assert single.unsupported_rate == 1.0
    assert single.overall.bootstrap_ci.degenerate is True


def test_validation_errors_are_value_free_and_no_network_is_required(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_connect(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket.socket, "connect", fail_connect)

    with pytest.raises(ValueError, match="evidence relation") as error:
        score_summary_claims(
            [{"claim_id": "claim-safe", "claim_class": "finding"}],
            [{"evidence_id": "evidence-safe", "relation": "not-a-state"}],
        )
    assert "not-a-state" not in str(error.value)
    assert "synthetic" not in APPROVED_EVIDENCE_DISCLAIMER


def test_report_rejects_invalid_bootstrap_configuration() -> None:
    claim = [{"claim_id": "claim-config", "claim_class": "finding"}]
    with pytest.raises(ValueError, match="n_resamples"):
        score_summary_claims(claim, n_resamples=0)
    with pytest.raises(ValueError, match="alpha"):
        score_summary_claims(claim, alpha=1.0)
