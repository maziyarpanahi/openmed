from __future__ import annotations

import builtins
import json
import traceback
import urllib.request
from dataclasses import replace
from typing import Any

import pytest

from openmed.agent.workflows import (
    TRIAL_ELIGIBILITY_REVIEW_SCHEMA,
    CohortCriterion,
    CohortDefinition,
    CohortRecordEvidence,
    CriterionEvidence,
    CriterionKind,
    CriterionState,
    DisagreementCause,
    EligibilityCitation,
    EvidenceAssertion,
    ModelCriterionAssessment,
    TrialEligibilityReviewError,
    build_trial_eligibility_review_packet,
    explain_cohort_membership,
)

RECORD = "sha256:" + "1" * 64
EVIDENCE_A = "sha256:" + "a" * 64
EVIDENCE_B = "sha256:" + "b" * 64
EVIDENCE_C = "sha256:" + "c" * 64
EVIDENCE_D = "sha256:" + "d" * 64
SOURCE_A = "sha256:" + "e" * 64
SOURCE_B = "sha256:" + "f" * 64
MODEL = "sha256:" + "2" * 64


def _explanation(
    evidence: tuple[CriterionEvidence, ...],
    criteria: tuple[CohortCriterion, ...] | None = None,
):
    definition = CohortDefinition(
        "trial.synthetic_protocol",
        4,
        criteria
        or (
            CohortCriterion("trial.confirmed_condition", CriterionKind.INCLUSION),
            CohortCriterion("trial.excluded_therapy", CriterionKind.EXCLUSION),
        ),
    )
    record = CohortRecordEvidence(
        RECORD,
        definition.definition_id,
        definition.version,
        evidence,
    )
    return explain_cohort_membership(record, definition)


def _citation(
    source: str = SOURCE_A,
    start: int = 10,
    end: int = 24,
    evidence: str = EVIDENCE_C,
) -> EligibilityCitation:
    return EligibilityCitation(source, start, end, evidence)


def _assessment(
    criterion_id: str,
    state: CriterionState,
    *,
    citations: tuple[EligibilityCitation, ...] | None = None,
    uncertainty: float = 0.2,
) -> ModelCriterionAssessment:
    return ModelCriterionAssessment(
        criterion_id,
        state,
        (_citation(),) if citations is None else citations,
        uncertainty,
        MODEL,
    )


def test_routes_contradictory_outcome_with_citations_and_uncertainty() -> None:
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
            CriterionEvidence(
                "trial.excluded_therapy", EvidenceAssertion.NOT_MET, EVIDENCE_B
            ),
        )
    )
    later_citation = _citation(SOURCE_B, 40, 52, EVIDENCE_D)
    packet = build_trial_eligibility_review_packet(
        explanation,
        (
            _assessment(
                "trial.confirmed_condition",
                CriterionState.NOT_MET,
                citations=(later_citation, _citation()),
                uncertainty=0.35,
            ),
            _assessment("trial.excluded_therapy", CriterionState.NOT_MET),
        ),
    )

    comparison = packet.comparisons[0]
    assert comparison.criterion_id == "trial.confirmed_condition"
    assert comparison.causes == (DisagreementCause.OUTCOME_CONFLICT,)
    assert comparison.model_uncertainty == 0.35
    assert comparison.model_citations == tuple(sorted((_citation(), later_citation)))
    assert comparison.rule_evidence_digests == (EVIDENCE_A,)
    assert packet.requires_human_review
    assert packet.to_dict()["schema"] == TRIAL_ELIGIBILITY_REVIEW_SCHEMA
    assert packet.packet_digest.startswith("sha256:")


def test_classifies_indeterminate_and_missing_results_per_criterion() -> None:
    criteria = (
        CohortCriterion("trial.conflicting_rule", CriterionKind.INCLUSION),
        CohortCriterion("trial.missing_rule", CriterionKind.INCLUSION),
    )
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.conflicting_rule", EvidenceAssertion.MET, EVIDENCE_A
            ),
            CriterionEvidence(
                "trial.conflicting_rule", EvidenceAssertion.NOT_MET, EVIDENCE_B
            ),
        ),
        criteria,
    )
    packet = build_trial_eligibility_review_packet(
        explanation,
        (_assessment("trial.conflicting_rule", CriterionState.UNKNOWN),),
    )
    by_id = {value.criterion_id: value for value in packet.comparisons}

    assert by_id["trial.conflicting_rule"].causes == (
        DisagreementCause.MODEL_ASSESSMENT_UNKNOWN,
        DisagreementCause.RULE_EVIDENCE_CONFLICT,
    )
    assert by_id["trial.missing_rule"].causes == (
        DisagreementCause.MODEL_ASSESSMENT_MISSING,
        DisagreementCause.RULE_EVIDENCE_MISSING,
    )
    assert by_id["trial.missing_rule"].model_citations == ()
    assert by_id["trial.missing_rule"].model_uncertainty is None


def test_agreement_does_not_request_review_or_authorize_action() -> None:
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
            CriterionEvidence(
                "trial.excluded_therapy", EvidenceAssertion.NOT_MET, EVIDENCE_B
            ),
        )
    )
    packet = build_trial_eligibility_review_packet(
        explanation,
        (
            _assessment("trial.confirmed_condition", CriterionState.MET),
            _assessment("trial.excluded_therapy", CriterionState.NOT_MET),
        ),
    )

    assert not packet.requires_human_review
    assert not packet.authorizes_enrollment
    assert not packet.authorizes_contact
    assert packet.to_dict()["authorizes_enrollment"] is False
    assert packet.to_dict()["authorizes_contact"] is False
    assert all(not value.causes for value in packet.comparisons)


def test_packet_is_deterministic_across_input_order() -> None:
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
            CriterionEvidence(
                "trial.excluded_therapy", EvidenceAssertion.NOT_MET, EVIDENCE_B
            ),
        )
    )
    first_citation = _citation()
    second_citation = _citation(SOURCE_B, 40, 52, EVIDENCE_D)
    assessments = (
        _assessment(
            "trial.confirmed_condition",
            CriterionState.NOT_MET,
            citations=(second_citation, first_citation),
        ),
        _assessment("trial.excluded_therapy", CriterionState.NOT_MET),
    )

    first = build_trial_eligibility_review_packet(explanation, assessments)
    second = build_trial_eligibility_review_packet(
        explanation, tuple(reversed(assessments))
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json() == json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    )


def test_packet_digest_binds_review_content() -> None:
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
            CriterionEvidence(
                "trial.excluded_therapy", EvidenceAssertion.NOT_MET, EVIDENCE_B
            ),
        )
    )
    packet = build_trial_eligibility_review_packet(
        explanation,
        (_assessment("trial.confirmed_condition", CriterionState.NOT_MET),),
    )

    with pytest.raises(TrialEligibilityReviewError) as caught:
        replace(packet, packet_digest=EVIDENCE_D)

    assert caught.value.code == "packet_digest_mismatch"


def test_assessments_require_evidence_citations_and_bounded_uncertainty() -> None:
    with pytest.raises(TrialEligibilityReviewError, match="empty_collection"):
        _assessment("trial.rule", CriterionState.MET, citations=())
    with pytest.raises(TrialEligibilityReviewError, match="invalid_uncertainty"):
        _assessment("trial.rule", CriterionState.MET, uncertainty=float("nan"))
    with pytest.raises(TrialEligibilityReviewError, match="invalid_uncertainty"):
        _assessment("trial.rule", CriterionState.MET, uncertainty=1.01)


def test_unknown_and_duplicate_model_assessments_fail_closed() -> None:
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
        )
    )
    assessment = _assessment("trial.confirmed_condition", CriterionState.MET)

    with pytest.raises(TrialEligibilityReviewError) as caught:
        build_trial_eligibility_review_packet(
            explanation,
            (_assessment("trial.undeclared", CriterionState.MET),),
        )
    assert caught.value.code == "unknown_criterion"

    with pytest.raises(TrialEligibilityReviewError) as caught:
        build_trial_eligibility_review_packet(
            explanation,
            (assessment, assessment),
        )
    assert caught.value.code == "duplicate_model_assessment"


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EligibilityCitation(value, 0, 4, EVIDENCE_A),
        lambda value: ModelCriterionAssessment(
            value,
            CriterionState.MET,
            (_citation(),),
            0.1,
            MODEL,
        ),
        lambda value: ModelCriterionAssessment(
            "trial.rule",
            CriterionState.MET,
            (_citation(),),
            0.1,
            value,
        ),
    ],
)
def test_validation_errors_never_echo_sensitive_values(factory: Any) -> None:
    sentinel = "Synthetic Person / record-123 / protected-note"

    with pytest.raises(TrialEligibilityReviewError) as caught:
        factory(sentinel)

    assert sentinel not in str(caught.value)
    assert sentinel not in "".join(traceback.format_exception(caught.value))


def test_packet_repr_omits_sensitive_digest_metadata() -> None:
    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
        )
    )
    assessment = _assessment("trial.confirmed_condition", CriterionState.NOT_MET)
    packet = build_trial_eligibility_review_packet(explanation, (assessment,))

    rendered = repr(packet)
    assert RECORD not in rendered
    assert MODEL not in rendered
    assert SOURCE_A not in rendered
    assert EVIDENCE_C not in rendered


def test_comparison_is_local_and_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_io(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("eligibility comparison must remain local and in-memory")

    explanation = _explanation(
        (
            CriterionEvidence(
                "trial.confirmed_condition", EvidenceAssertion.MET, EVIDENCE_A
            ),
        )
    )
    monkeypatch.setattr(builtins, "open", fail_io)
    monkeypatch.setattr(urllib.request, "urlopen", fail_io)

    packet = build_trial_eligibility_review_packet(
        explanation,
        (_assessment("trial.confirmed_condition", CriterionState.NOT_MET),),
    )

    assert packet.requires_human_review
