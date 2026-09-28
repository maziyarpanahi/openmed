"""Reviewable public trial eligibility matching tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.journey import JourneySnapshot
from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id
from openmed.clinical.trials import (
    JourneySignal,
    TrialContractError,
    TrialCriterionKind,
    TrialCriterionOperator,
    TrialCriterionState,
    TrialEligibilityResult,
    TrialLocation,
    TrialMatchState,
    TrialStudyRecord,
    evaluate_trial_eligibility,
    load_trial_eligibility_schema,
    match_trial_candidates,
    parse_trial_criteria,
    retrieve_trial_candidates,
)
from openmed.eval.suites.trial_eligibility import (
    TrialEligibilityBenchmarkCase,
    run_trial_eligibility_benchmark,
)

SNAPSHOT = JourneySnapshot.at(derived_opaque_id("patient", "trialmatcher"), 7)
EVALUATED_AT = "2026-09-21T10:00:00Z"


def _study(*, unsupported: bool = False) -> TrialStudyRecord:
    if unsupported:
        criteria = """Inclusion Criteria:
- condition: Asthma
- Investigator judgment confirms suitability
"""
        return TrialStudyRecord(
            study_id="NCT00000002",
            overall_status="RECRUITING",
            brief_title="Synthetic asthma study",
            official_title=None,
            last_update_date="2026-09-20",
            conditions=("Asthma",),
            interventions=(),
            locations=(TrialLocation(country="France"),),
            eligibility_text=criteria,
            retrieved_at="2026-09-21T08:00:00Z",
            source_digest="sha256:" + "2" * 64,
        )
    criteria = """Inclusion Criteria:
- condition: Hypertension
- age >= 18 years
- hba1c <= 7.5 percent within 90 days
Exclusion Criteria:
- medication: Warfarin
"""
    return TrialStudyRecord(
        study_id="NCT00000001",
        overall_status="RECRUITING",
        brief_title="Synthetic hypertension study",
        official_title="A Synthetic Hypertension Eligibility Study",
        last_update_date="2026-09-20",
        conditions=("Hypertension",),
        interventions=(),
        locations=(TrialLocation(country="France"),),
        eligibility_text=criteria,
        retrieved_at="2026-09-21T08:00:00Z",
        source_digest="sha256:" + "1" * 64,
    )


def _signal(
    concept_kind: str,
    concept: str,
    suffix: str,
    *,
    numeric_value: float | None = None,
    unit: str | None = None,
    present: bool | None = True,
    observed_at: str | None = None,
    conflict: bool = False,
) -> JourneySignal:
    return JourneySignal(
        concept_kind=concept_kind,
        concept=concept,
        snapshot_id=SNAPSHOT.snapshot_id,
        snapshot_digest=canonical_digest(SNAPSHOT.to_dict()),
        numeric_value=numeric_value,
        unit=unit,
        present=present,
        observed_at=observed_at,
        fact_ids=(derived_opaque_id("fact", suffix),),
        evidence_ids=(derived_opaque_id("evidence", suffix),),
        conflict_ids=(derived_opaque_id("conflict", suffix),) if conflict else (),
    )


def _signals() -> tuple[JourneySignal, ...]:
    return (
        _signal("condition", "Hypertension", "hypertension"),
        _signal(
            "observation",
            "age",
            "age",
            numeric_value=42.125,
            unit="years",
        ),
        _signal(
            "observation",
            "hba1c",
            "hba1c",
            numeric_value=7.0,
            unit="percent",
            observed_at="2026-09-01T10:00:00Z",
        ),
        _signal("medication", "Warfarin", "warfarin", present=False),
    )


def test_parser_preserves_typed_criteria_windows_and_unsupported_fragments() -> None:
    parsed = parse_trial_criteria(_study())
    assert len(parsed.criteria) == 4
    assert parsed.criteria[0].kind is TrialCriterionKind.INCLUSION
    assert parsed.criteria[0].operator is TrialCriterionOperator.EXISTS
    assert parsed.criteria[1].value is not None
    assert parsed.criteria[1].value.normalized_value == "18"
    assert parsed.criteria[2].window is not None
    assert parsed.criteria[2].window.within_days == 90
    assert parsed == type(parsed).from_dict(parsed.to_dict())

    unsupported = parse_trial_criteria(_study(unsupported=True))
    assert unsupported.criteria[1].supported is False
    assert unsupported.criteria[1].unsupported_reason == "fragment_unsupported"


def test_complete_evidence_produces_eligible_value_free_result() -> None:
    study = _study()
    result = evaluate_trial_eligibility(
        study,
        SNAPSHOT,
        _signals(),
        evaluated_at=EVALUATED_AT,
    )
    assert result.state is TrialMatchState.ELIGIBLE
    assert result.eligible is True
    assert result.review_required is False
    assert [item.state for item in result.criterion_results] == [
        TrialCriterionState.MET,
        TrialCriterionState.MET,
        TrialCriterionState.MET,
        TrialCriterionState.NOT_MET,
    ]
    assert all(
        item.evidence.snapshot_id == SNAPSHOT.snapshot_id
        for item in result.criterion_results
    )
    assert all(
        item.study_version_id == study.version_id for item in result.criterion_results
    )

    packet = json.dumps(result.to_review_packet(), sort_keys=True)
    assert "42.125" not in packet
    assert "7.0" not in packet
    assert "numeric_value" not in packet
    assert TrialEligibilityResult.from_dict(result.to_dict()) == result

    schema = load_trial_eligibility_schema()
    validator_for(schema).check_schema(schema)
    assert not list(validator_for(schema)(schema).iter_errors(result.to_dict()))


def test_not_met_unknown_conflict_and_unsupported_states_are_explicit() -> None:
    study = _study()
    age_failed = tuple(
        _signal(
            item.concept_kind,
            item.concept,
            f"failed-{index}",
            numeric_value=16.0 if item.concept == "age" else item.numeric_value,
            unit=item.unit,
            present=item.present,
            observed_at=item.observed_at,
        )
        for index, item in enumerate(_signals())
    )
    failed = evaluate_trial_eligibility(
        study, SNAPSHOT, age_failed, evaluated_at=EVALUATED_AT
    )
    assert failed.state is TrialMatchState.NOT_ELIGIBLE
    assert TrialCriterionState.NOT_MET in {
        item.state for item in failed.criterion_results
    }

    missing = evaluate_trial_eligibility(
        study,
        SNAPSHOT,
        tuple(item for item in _signals() if item.concept != "hba1c"),
        evaluated_at=EVALUATED_AT,
    )
    assert missing.state is TrialMatchState.REVIEW_REQUIRED
    assert TrialCriterionState.UNKNOWN in {
        item.state for item in missing.criterion_results
    }
    assert missing.eligible is False

    conflicted_signals = (
        _signal("condition", "Hypertension", "conflicted", conflict=True),
        *_signals()[1:],
    )
    conflicted = evaluate_trial_eligibility(
        study, SNAPSHOT, conflicted_signals, evaluated_at=EVALUATED_AT
    )
    assert TrialCriterionState.CONFLICT in {
        item.state for item in conflicted.criterion_results
    }
    assert conflicted.state is TrialMatchState.REVIEW_REQUIRED

    unsupported = evaluate_trial_eligibility(
        _study(unsupported=True),
        SNAPSHOT,
        (_signal("condition", "Asthma", "asthma"),),
        evaluated_at=EVALUATED_AT,
    )
    assert TrialCriterionState.UNSUPPORTED in {
        item.state for item in unsupported.criterion_results
    }
    assert unsupported.eligible is False
    assert unsupported.review_required is True


def test_incomplete_numeric_evidence_cannot_be_overridden_by_one_match() -> None:
    signals = (
        *_signals(),
        _signal(
            "observation",
            "hba1c",
            "hba1c-value-missing",
            unit="percent",
            observed_at="2026-09-01T10:00:00Z",
        ),
    )
    result = evaluate_trial_eligibility(
        _study(), SNAPSHOT, signals, evaluated_at=EVALUATED_AT
    )
    assert result.state is TrialMatchState.REVIEW_REQUIRED
    assert result.eligible is False
    assert result.criterion_results[2].state is TrialCriterionState.UNKNOWN
    assert result.criterion_results[2].reason_code == "typed_value_missing"


def test_caller_supplied_criteria_must_match_the_public_study() -> None:
    study = _study()
    parsed = parse_trial_criteria(study)
    forged = replace(parsed, criteria=parsed.criteria[:3])
    with pytest.raises(TrialContractError, match="criteria"):
        evaluate_trial_eligibility(
            study,
            SNAPSHOT,
            _signals(),
            criteria=forged,
            evaluated_at=EVALUATED_AT,
        )

    candidate = retrieve_trial_candidates((study,), _signals())[0]
    with pytest.raises(TrialContractError, match="candidate"):
        evaluate_trial_eligibility(
            study,
            SNAPSHOT,
            _signals(),
            candidate=replace(candidate, study_id="NCT00000002"),
            evaluated_at=EVALUATED_AT,
        )


def test_candidate_retrieval_reranks_before_evaluation() -> None:
    studies = (_study(unsupported=True), _study())
    candidates = retrieve_trial_candidates(studies, _signals(), limit=2)
    assert [item.study_id for item in candidates] == ["NCT00000001", "NCT00000002"]
    assert candidates[0].retrieval_score > candidates[1].retrieval_score

    results = match_trial_candidates(
        studies,
        SNAPSHOT,
        _signals(),
        evaluated_at=EVALUATED_AT,
        limit=1,
    )
    assert len(results) == 1
    assert results[0].candidate.study_id == "NCT00000001"
    assert results[0].state is TrialMatchState.ELIGIBLE


def test_frozen_synthetic_benchmark_reports_metrics_with_provenance() -> None:
    study = _study()
    parsed = parse_trial_criteria(study)
    expected_states = {
        criterion.criterion_id: state
        for criterion, state in zip(
            parsed.criteria,
            (
                TrialCriterionState.MET,
                TrialCriterionState.MET,
                TrialCriterionState.MET,
                TrialCriterionState.NOT_MET,
            ),
            strict=True,
        )
    }
    case = TrialEligibilityBenchmarkCase(
        case_id="synthetic-complete-evidence",
        snapshot=SNAPSHOT,
        signals=_signals(),
        expected_study_ids=(study.study_id,),
        expected_criterion_states=expected_states,
        evaluated_at=EVALUATED_AT,
    )
    report = run_trial_eligibility_benchmark(
        (study, _study(unsupported=True)), (case,), limit=1
    )
    assert report.retrieval_recall_at_k == 1.0
    assert report.criterion_state_accuracy == 1.0
    assert report.case_count == 1
    assert report.fixture_digest.startswith("sha256:")
    assert report.policy_digest.startswith("sha256:")
    assert report.versions["matcher_version"]
    assert report.versions["parser_version"]
