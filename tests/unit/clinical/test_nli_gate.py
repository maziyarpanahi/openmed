"""Synthetic offline tests for the clinical NLI abstention gate."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.nli_gate import (
    HUMAN_REVIEW_QUEUE,
    NLI_GATE_ADVISORY,
    CalibratedNLIScores,
    ClinicalNLIGate,
    EvidenceLink,
    NLIThresholds,
    evaluate_evidence_links,
    evaluate_nli,
)


def _link() -> EvidenceLink:
    return EvidenceLink.from_text(
        source_id="synthetic-note-1",
        claim_id="synthetic-claim-1",
        source_text="Synthetic evidence supports the synthetic claim.",
        claim_text="The synthetic claim is supported.",
        start=0,
        end=48,
    )


def _thresholds() -> NLIThresholds:
    return NLIThresholds(
        entailment=0.90,
        contradiction=0.90,
        margin=0.10,
        calibration_id="synthetic-calibration-v1",
    )


def test_gate_returns_entailment_only_for_calibrated_separated_scores() -> None:
    result = evaluate_nli(
        {
            "entailment": 0.96,
            "contradiction": 0.02,
            "neutral": 0.02,
            "calibration_id": "synthetic-calibration-v1",
        },
        _link(),
        thresholds=_thresholds(),
    )

    assert result.outcome == "entailment"
    assert result.requires_human_review is False
    assert result.human_review is None
    assert result.autonomous_decision is False
    assert result.evidence.source_hash.startswith("sha256:")


def test_contradiction_is_typed_and_traceable_without_retaining_text() -> None:
    result = evaluate_nli(
        {"entailment": 0.01, "contradiction": 0.95, "neutral": 0.04},
        _link(),
        thresholds=_thresholds(),
    )

    assert result.status == "contradiction"
    payload = json.dumps(result.to_dict(), sort_keys=True)
    assert "Synthetic evidence" not in payload
    assert "The synthetic claim" not in payload
    assert result.to_audit_entry()["evidence"]["start"] == 0


def test_ambiguous_scores_abstain_and_create_a_human_review_handoff() -> None:
    result = evaluate_nli(
        {
            "entailment": 0.72,
            "contradiction": 0.19,
            "neutral": 0.09,
        },
        _link(),
        thresholds=_thresholds(),
    )

    assert result.outcome == "abstain"
    assert result.reason == "below_calibrated_threshold"
    assert result.requires_human_review is True
    assert result.human_review is not None
    assert result.human_review.queue == HUMAN_REVIEW_QUEUE
    assert result.human_review.status == "pending"
    assert "clinical decision" in NLI_GATE_ADVISORY


def test_close_high_scores_abstain_when_the_calibrated_margin_is_not_met() -> None:
    result = evaluate_nli(
        {"entailment": 0.94, "contradiction": 0.88, "neutral": 0.01},
        _link(),
        thresholds=_thresholds(),
    )

    assert result.outcome == "abstain"
    assert result.reason == "insufficient_margin"
    assert result.margin == pytest.approx(0.06)


def test_neutral_verifier_output_abstains_even_when_its_score_is_high() -> None:
    result = evaluate_nli(
        {"label": "neutral", "score": 0.94},
        {"source_id": "synthetic-source", "target_id": "synthetic-target"},
        thresholds=_thresholds(),
    )

    assert result.outcome == "abstain"
    assert result.reason == "neutral_or_unsupported"
    assert result.human_review_required is True


def test_typed_score_and_batch_apis_are_deterministic() -> None:
    scores = CalibratedNLIScores(
        entailment=0.94,
        contradiction=0.03,
        neutral=0.03,
        calibration_id="synthetic-calibration-v1",
    )
    gate = ClinicalNLIGate(_thresholds())
    records = ((scores, _link()), (scores, _link()))

    first = gate.evaluate_many(records)
    second = evaluate_evidence_links(records, thresholds=_thresholds())

    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
    assert all(item.outcome == "entailment" for item in first)


def test_calibration_mismatch_and_uncalibrated_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="calibration_id"):
        evaluate_nli(
            {
                "entailment": 0.95,
                "contradiction": 0.03,
                "neutral": 0.02,
                "calibration_id": "other-calibration",
            },
            _link(),
            thresholds=_thresholds(),
        )

    with pytest.raises(ValueError, match="must be calibrated"):
        NLIThresholds(calibrated=False)


def test_raw_text_is_hashed_and_not_retained_by_evidence_link() -> None:
    link = EvidenceLink.from_text(
        source_id="synthetic-source",
        claim_id="synthetic-claim",
        source_text="Synthetic sensitive-looking source value.",
        claim_text="Synthetic sensitive-looking claim value.",
    )

    payload = json.dumps(link.to_dict(), sort_keys=True)
    assert "sensitive-looking" not in payload
    assert link.source_hash != link.claim_hash
