"""Evidence-bound care-gap state, correction, and review tests."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
from jsonschema.validators import validator_for

from openmed.clinical.care_gaps import (
    CARE_GAP_ADVISORY,
    CareGapEvaluation,
    CareGapHistory,
    CareGapPolicy,
    CareGapReviewStatus,
    CareGapState,
    begin_care_gap_review,
    complete_care_gap_review,
    evaluate_care_gap,
    load_care_gap_schema,
)
from openmed.clinical.journey_contracts import canonical_digest, derived_opaque_id
from openmed.clinical.measures import (
    CalculationTraceStep,
    MeasureEngineIdentity,
    MeasureEvidence,
    MeasureSubjectResult,
    MeasureTimeWindow,
    PopulationKind,
    PopulationResult,
    PopulationState,
)
from openmed.structured.store import StoreState

SUBJECT = "patient_caregapsubject01"
DEFINITION_ID = "measureversion_caregapmeasure01"
DEFINITION_DIGEST = "sha256:" + "1" * 64
SNAPSHOT_ID = "snapshot_caregapsnapshot01"
SNAPSHOT_DIGEST = "sha256:" + "2" * 64
PERIOD = MeasureTimeWindow(start="2025-01-01T00:00:00Z", end="2025-12-31T23:59:59Z")
ENGINE = MeasureEngineIdentity(
    engine_id="openmed.native.measure",
    version="1.0.0",
    artifact_digest="sha256:" + "3" * 64,
    execution_mode="native",
)


def _policy() -> CareGapPolicy:
    return CareGapPolicy(
        policy_id="care_gap_default",
        version="1.0.0",
        initial_population_id="initial",
        denominator_population_id="denominator",
        numerator_population_id="numerator",
        exclusion_population_id="exclusion",
        exception_population_id="exception",
    )


def _population(
    population_id: str, kind: PopulationKind, state: PopulationState
) -> PopulationResult:
    suffix = {
        "initial": "a",
        "denominator": "b",
        "numerator": "c",
        "exclusion": "d",
        "exception": "e",
    }[population_id]
    evidence = MeasureEvidence(
        fact_ids=(f"fact_{suffix * 16}",),
        evidence_ids=(f"evidence_{suffix * 16}",),
        derivation_digests=(canonical_digest({"population": population_id}),),
    )
    return PopulationResult(
        population_id=population_id,
        kind=kind,
        state=state,
        evidence=evidence,
        reason_code=f"synthetic_{state.value}",
        error_code="synthetic_expression_error"
        if state is PopulationState.ERROR
        else None,
    )


def _measure_result(
    *,
    numerator: PopulationState,
    initial: PopulationState = PopulationState.MET,
    denominator: PopulationState = PopulationState.MET,
    exclusion: PopulationState = PopulationState.NOT_MET,
    exception: PopulationState = PopulationState.NOT_MET,
    evaluated_at: str = "2026-01-02T03:04:05Z",
    omit: str | None = None,
) -> MeasureSubjectResult:
    populations = (
        _population("initial", PopulationKind.INITIAL_POPULATION, initial),
        _population("denominator", PopulationKind.DENOMINATOR, denominator),
        _population("numerator", PopulationKind.NUMERATOR, numerator),
        _population("exclusion", PopulationKind.DENOMINATOR_EXCLUSION, exclusion),
        _population("exception", PopulationKind.DENOMINATOR_EXCEPTION, exception),
    )
    populations = tuple(item for item in populations if item.population_id != omit)
    trace = tuple(
        CalculationTraceStep(
            step_id=item.population_id,
            expression_ref=f"native:{item.population_id}",
            state=item.state,
            input_digest=canonical_digest(
                {"population": item.population_id, "state": item.state.value}
            ),
            evidence_digest=item.evidence.digest,
            reason_code=item.reason_code,
        )
        for item in populations
    )
    projection_digest = canonical_digest(
        {"populations": [item.to_dict() for item in populations]}
    )
    return MeasureSubjectResult(
        result_id=derived_opaque_id(
            "measureresult", numerator.value, initial.value, evaluated_at, omit
        ),
        subject_id=SUBJECT,
        definition_version_id=DEFINITION_ID,
        definition_digest=DEFINITION_DIGEST,
        source_snapshot_id=SNAPSHOT_ID,
        source_snapshot_digest=SNAPSHOT_DIGEST,
        measurement_period=PERIOD,
        engine=ENGINE,
        input_projection_digest=projection_digest,
        value_set_digests={},
        populations=populations,
        trace=trace,
        evaluated_at=evaluated_at,
    )


def _evaluate(**kwargs):
    result = evaluate_care_gap(_measure_result(**kwargs), _policy())
    assert result.value is not None
    return result.value


def test_frozen_measure_results_produce_all_four_gap_states() -> None:
    met = _evaluate(numerator=PopulationState.MET)
    opened = _evaluate(numerator=PopulationState.NOT_MET)
    not_applicable = _evaluate(
        numerator=PopulationState.NOT_MET, exclusion=PopulationState.MET
    )
    insufficient = _evaluate(numerator=PopulationState.UNKNOWN)

    assert met.state is CareGapState.MET
    assert opened.state is CareGapState.OPEN
    assert not_applicable.state is CareGapState.NOT_APPLICABLE
    assert insufficient.state is CareGapState.INSUFFICIENT_DATA
    assert met.review_status is CareGapReviewStatus.NOT_REQUIRED
    assert insufficient.review_status is CareGapReviewStatus.REQUIRED


def test_missing_or_error_data_never_becomes_an_open_gap() -> None:
    missing = _evaluate(numerator=PopulationState.NOT_MET, omit="denominator")
    errored = _evaluate(numerator=PopulationState.ERROR)

    assert missing.state is CareGapState.INSUFFICIENT_DATA
    assert missing.reason_code == "population_result_missing"
    assert errored.state is CareGapState.INSUFFICIENT_DATA
    assert errored.review_status is CareGapReviewStatus.REQUIRED


@pytest.mark.parametrize("population_id", ["initial", "denominator", "numerator"])
def test_population_role_mismatch_cannot_create_open_gap(population_id: str) -> None:
    result = _measure_result(numerator=PopulationState.NOT_MET)
    populations = tuple(
        replace(item, kind=PopulationKind.MEASURE_OBSERVATION)
        if item.population_id == population_id
        else item
        for item in result.populations
    )
    mismatched = replace(result, populations=populations)
    evaluated = evaluate_care_gap(mismatched, _policy())
    assert evaluated.value is not None
    assert evaluated.value.state is CareGapState.INSUFFICIENT_DATA
    assert evaluated.value.review_status is CareGapReviewStatus.REQUIRED
    assert evaluated.value.reason_code == "population_role_conflict"


def test_conflicting_inputs_require_review_and_review_cannot_be_skipped() -> None:
    measure_result = _measure_result(numerator=PopulationState.NOT_MET)
    evaluated = evaluate_care_gap(
        measure_result,
        _policy(),
        conflict_ids=("conflict_syntheticconflict1",),
    )
    assert evaluated.value is not None
    gap = evaluated.value
    assert gap.state is CareGapState.INSUFFICIENT_DATA
    assert gap.review_status is CareGapReviewStatus.REQUIRED

    skipped = complete_care_gap_review(
        gap,
        approved=True,
        authorization_digest="sha256:" + "4" * 64,
        decision_digest="sha256:" + "5" * 64,
        occurred_at="2026-01-02T03:05:00Z",
        reason_code="review_complete",
    )
    assert skipped.state is StoreState.CONFLICT
    assert skipped.code == "care_gap_review_not_started"

    started = begin_care_gap_review(
        gap,
        authorization_digest="sha256:" + "4" * 64,
        occurred_at="2026-01-02T03:05:00Z",
    )
    assert started.value is not None
    completed = complete_care_gap_review(
        started.value,
        approved=True,
        authorization_digest="sha256:" + "4" * 64,
        decision_digest="sha256:" + "5" * 64,
        occurred_at="2026-01-02T03:06:00Z",
        reason_code="review_complete",
    )
    assert completed.value is not None
    assert completed.value.review_status is CareGapReviewStatus.APPROVED
    assert completed.value.state is CareGapState.INSUFFICIENT_DATA
    history = CareGapHistory(
        gap_id=gap.gap_id, versions=(gap, started.value, completed.value)
    )
    assert history.current == completed.value


def test_correction_creates_a_new_version_and_preserves_history() -> None:
    opened = _evaluate(numerator=PopulationState.NOT_MET)
    corrected_measure = _measure_result(
        numerator=PopulationState.MET,
        evaluated_at="2026-01-03T03:04:05Z",
    )
    corrected = evaluate_care_gap(
        corrected_measure,
        _policy(),
        previous=opened,
    )
    assert corrected.value is not None
    assert corrected.value.gap_id == opened.gap_id
    assert corrected.value.parent_version_id == opened.version_id
    assert corrected.value.version_id != opened.version_id
    assert corrected.value.state is CareGapState.MET
    history = CareGapHistory(gap_id=opened.gap_id, versions=(opened, corrected.value))
    assert [item.state for item in history.versions] == [
        CareGapState.OPEN,
        CareGapState.MET,
    ]


def test_artifact_is_value_free_round_trips_and_validates_schema() -> None:
    result = _evaluate(numerator=PopulationState.UNKNOWN)
    restored = CareGapEvaluation.from_json(result.to_json())
    assert restored == result
    payload = result.to_dict()
    assert payload["advisory"] == CARE_GAP_ADVISORY
    serialized = json.dumps(payload, sort_keys=True)
    assert "protected" not in serialized
    assert "reviewer" not in serialized

    schema = load_care_gap_schema()
    validator = validator_for(schema)
    validator.check_schema(schema)
    assert not tuple(validator(schema).iter_errors(payload))


def test_tampered_version_digest_and_unchanged_correction_fail_closed() -> None:
    opened = _evaluate(numerator=PopulationState.NOT_MET)
    unchanged = evaluate_care_gap(
        _measure_result(numerator=PopulationState.NOT_MET),
        _policy(),
        previous=opened,
    )
    assert unchanged.state is StoreState.CONFLICT
    assert unchanged.code == "care_gap_correction_unchanged"

    payload = opened.to_dict()
    payload["version_digest"] = "sha256:" + "0" * 64
    try:
        CareGapEvaluation.from_dict(payload)
    except Exception as exc:
        assert "version digest differs" in str(exc)
    else:  # pragma: no cover - safety assertion
        raise AssertionError("tampered care-gap digest was accepted")
