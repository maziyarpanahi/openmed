from __future__ import annotations

import builtins
import json
import traceback
import urllib.request
from typing import Any

import pytest

from openmed.agent.workflows import (
    COHORT_EXPLANATION_SCHEMA,
    CohortCriterion,
    CohortDefinition,
    CohortExplanationError,
    CohortMembershipExplanation,
    CohortRecordEvidence,
    CriterionEvaluation,
    CriterionEvidence,
    CriterionKind,
    CriterionState,
    EvidenceAssertion,
    MembershipState,
    TimeWindowReference,
    explain_cohort_membership,
)

RECORD = "sha256:" + "1" * 64
EVIDENCE_A = "sha256:" + "a" * 64
EVIDENCE_B = "sha256:" + "b" * 64
EVIDENCE_C = "sha256:" + "c" * 64
EVIDENCE_D = "sha256:" + "d" * 64
WINDOW_A = TimeWindowReference("window.index_prior_year", "sha256:" + "e" * 64)
WINDOW_B = TimeWindowReference("window.current_episode", "sha256:" + "f" * 64)


def _definition(
    criteria: tuple[CohortCriterion, ...] | None = None,
) -> CohortDefinition:
    return CohortDefinition(
        "cohort.synthetic_registry",
        3,
        criteria
        or (
            CohortCriterion("clinical.age_range", CriterionKind.INCLUSION),
            CohortCriterion(
                "clinical.confirmed_condition",
                CriterionKind.INCLUSION,
                WINDOW_A,
            ),
            CohortCriterion(
                "clinical.excluded_medication",
                CriterionKind.EXCLUSION,
                WINDOW_B,
            ),
            CohortCriterion("clinical.prior_procedure", CriterionKind.EXCLUSION),
        ),
    )


def _record(
    evidence: tuple[CriterionEvidence, ...],
    *,
    definition_id: str = "cohort.synthetic_registry",
    definition_version: int = 3,
) -> CohortRecordEvidence:
    return CohortRecordEvidence(
        RECORD,
        definition_id,
        definition_version,
        evidence,
    )


def _evidence(
    criterion_id: str,
    assertion: EvidenceAssertion,
    digest: str = EVIDENCE_A,
    window: TimeWindowReference | None = None,
) -> CriterionEvidence:
    return CriterionEvidence(criterion_id, assertion, digest, window)


def test_explains_all_criterion_states_with_evidence_and_windows() -> None:
    definition = _definition()
    record = _record(
        (
            _evidence(
                "clinical.excluded_medication",
                EvidenceAssertion.NOT_MET,
                EVIDENCE_D,
                WINDOW_B,
            ),
            _evidence(
                "clinical.confirmed_condition",
                EvidenceAssertion.MET,
                EVIDENCE_B,
                WINDOW_A,
            ),
            _evidence(
                "clinical.excluded_medication",
                EvidenceAssertion.MET,
                EVIDENCE_C,
                WINDOW_B,
            ),
            _evidence(
                "clinical.age_range",
                EvidenceAssertion.NOT_MET,
                EVIDENCE_A,
            ),
        )
    )

    explanation = explain_cohort_membership(record, definition)
    by_id = {value.criterion_id: value for value in explanation.criteria}

    assert by_id["clinical.confirmed_condition"].state is CriterionState.MET
    assert by_id["clinical.age_range"].state is CriterionState.NOT_MET
    assert by_id["clinical.excluded_medication"].state is CriterionState.CONFLICT
    assert by_id["clinical.prior_procedure"].state is CriterionState.UNKNOWN
    assert by_id["clinical.confirmed_condition"].time_window == WINDOW_A
    assert by_id["clinical.excluded_medication"].evidence_digests == (
        EVIDENCE_C,
        EVIDENCE_D,
    )
    assert explanation.membership_state is MembershipState.REVIEW_REQUIRED
    assert explanation.requires_human_review is True


def test_definition_and_evidence_order_do_not_change_explanation() -> None:
    criteria = (
        CohortCriterion("clinical.first", CriterionKind.INCLUSION, WINDOW_A),
        CohortCriterion("clinical.second", CriterionKind.EXCLUSION),
    )
    evidence = (
        _evidence("clinical.first", EvidenceAssertion.MET, EVIDENCE_A, WINDOW_A),
        _evidence("clinical.second", EvidenceAssertion.NOT_MET, EVIDENCE_B),
    )

    first = explain_cohort_membership(_record(evidence), _definition(criteria))
    second = explain_cohort_membership(
        _record(tuple(reversed(evidence))),
        _definition(tuple(reversed(criteria))),
    )

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.to_json() == json.dumps(
        first.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert first.to_dict()["schema"] == COHORT_EXPLANATION_SCHEMA


def test_satisfied_criteria_are_eligible_but_never_authorize_action() -> None:
    definition = _definition(
        (
            CohortCriterion("clinical.required", CriterionKind.INCLUSION),
            CohortCriterion("clinical.excluded", CriterionKind.EXCLUSION),
        )
    )
    record = _record(
        (
            _evidence("clinical.required", EvidenceAssertion.MET, EVIDENCE_A),
            _evidence("clinical.excluded", EvidenceAssertion.NOT_MET, EVIDENCE_B),
        )
    )

    explanation = explain_cohort_membership(record, definition)

    assert explanation.membership_state is MembershipState.ELIGIBLE
    assert explanation.requires_human_review is False
    assert explanation.authorizes_enrollment is False
    assert explanation.authorizes_contact is False
    assert explanation.to_dict()["authorizes_enrollment"] is False
    assert explanation.to_dict()["authorizes_contact"] is False


@pytest.mark.parametrize(
    ("kind", "assertion"),
    [
        (CriterionKind.INCLUSION, EvidenceAssertion.NOT_MET),
        (CriterionKind.EXCLUSION, EvidenceAssertion.MET),
    ],
)
def test_failed_inclusion_or_met_exclusion_is_ineligible(
    kind: CriterionKind,
    assertion: EvidenceAssertion,
) -> None:
    definition = _definition((CohortCriterion("clinical.rule", kind),))
    explanation = explain_cohort_membership(
        _record((_evidence("clinical.rule", assertion),)),
        definition,
    )

    assert explanation.membership_state is MembershipState.INELIGIBLE
    assert explanation.requires_human_review is False
    assert explanation.authorizes_enrollment is False


@pytest.mark.parametrize(
    "evidence",
    [
        (),
        (_evidence("clinical.rule", EvidenceAssertion.UNKNOWN),),
        (
            _evidence("clinical.rule", EvidenceAssertion.MET, EVIDENCE_A),
            _evidence("clinical.rule", EvidenceAssertion.NOT_MET, EVIDENCE_B),
        ),
    ],
)
def test_unknown_or_conflicting_criterion_requires_human_review(
    evidence: tuple[CriterionEvidence, ...],
) -> None:
    definition = _definition(
        (CohortCriterion("clinical.rule", CriterionKind.INCLUSION),)
    )
    explanation = explain_cohort_membership(_record(evidence), definition)

    assert explanation.membership_state is MembershipState.REVIEW_REQUIRED
    assert explanation.requires_human_review is True
    assert explanation.authorizes_enrollment is False
    assert explanation.authorizes_contact is False


def test_decisive_evidence_wins_over_inconclusive_evidence() -> None:
    definition = _definition(
        (CohortCriterion("clinical.rule", CriterionKind.INCLUSION),)
    )
    record = _record(
        (
            _evidence("clinical.rule", EvidenceAssertion.UNKNOWN, EVIDENCE_A),
            _evidence("clinical.rule", EvidenceAssertion.MET, EVIDENCE_B),
        )
    )

    explanation = explain_cohort_membership(record, definition)

    assert explanation.criteria[0].state is CriterionState.MET
    assert explanation.membership_state is MembershipState.ELIGIBLE


@pytest.mark.parametrize(
    ("record", "code", "field_name"),
    [
        (
            _record(
                (_evidence("clinical.unknown", EvidenceAssertion.MET),),
            ),
            "unknown_criterion",
            "criterion_id",
        ),
        (
            _record(
                (
                    _evidence(
                        "clinical.confirmed_condition",
                        EvidenceAssertion.MET,
                        window=WINDOW_B,
                    ),
                )
            ),
            "time_window_mismatch",
            "time_window",
        ),
        (
            _record((), definition_id="cohort.other"),
            "definition_id_mismatch",
            "definition_id",
        ),
        (
            _record((), definition_version=4),
            "definition_version_mismatch",
            "definition_version",
        ),
    ],
)
def test_mismatched_contracts_fail_closed(
    record: CohortRecordEvidence,
    code: str,
    field_name: str,
) -> None:
    with pytest.raises(CohortExplanationError) as caught:
        explain_cohort_membership(record, _definition())

    assert caught.value.code == code
    assert caught.value.field_name == field_name


def test_duplicate_evidence_and_criteria_fail_closed() -> None:
    criterion = CohortCriterion("clinical.rule", CriterionKind.INCLUSION)
    evidence = _evidence("clinical.rule", EvidenceAssertion.MET)

    with pytest.raises(CohortExplanationError, match="duplicate_criterion"):
        _definition((criterion, criterion))
    with pytest.raises(CohortExplanationError, match="duplicate_evidence"):
        _record((evidence, evidence))
    with pytest.raises(CohortExplanationError, match="duplicate_evidence"):
        _record(
            (
                evidence,
                _evidence(
                    "clinical.rule",
                    EvidenceAssertion.NOT_MET,
                    evidence.evidence_digest,
                ),
            )
        )


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: TimeWindowReference(value, EVIDENCE_A),
        lambda value: CohortCriterion(value, CriterionKind.INCLUSION),
        lambda value: CohortDefinition(
            value, 1, (CohortCriterion("rule.a", CriterionKind.INCLUSION),)
        ),
        lambda value: CriterionEvidence(
            value,
            EvidenceAssertion.MET,
            EVIDENCE_A,
        ),
    ],
)
def test_invalid_identifiers_never_echo_sensitive_values(factory: Any) -> None:
    sentinel = "Jane Synthetic / MRN-12345"

    with pytest.raises(CohortExplanationError) as caught:
        factory(sentinel)

    rendered = "".join(traceback.format_exception(caught.value))
    assert caught.value.code == "invalid_identifier"
    assert sentinel not in rendered


def test_invalid_digest_never_echoes_sensitive_value() -> None:
    sentinel = "Jane Synthetic has condition Z99.999"

    with pytest.raises(CohortExplanationError) as caught:
        CohortRecordEvidence(
            sentinel,
            "cohort.synthetic_registry",
            3,
            (),
        )

    assert caught.value.code == "invalid_digest"
    assert sentinel not in "".join(traceback.format_exception(caught.value))


def test_direct_explanation_construction_rejects_inconsistent_state() -> None:
    criterion = CriterionEvaluation(
        "clinical.rule",
        CriterionKind.INCLUSION,
        CriterionState.UNKNOWN,
        (),
    )

    with pytest.raises(CohortExplanationError) as caught:
        CohortMembershipExplanation(
            record_digest=RECORD,
            definition_digest=EVIDENCE_A,
            evidence_metadata_digest=EVIDENCE_B,
            explanation_digest=EVIDENCE_C,
            membership_state=MembershipState.ELIGIBLE,
            criteria=(criterion,),
        )

    assert caught.value.code == "inconsistent_membership_state"


def test_evaluation_is_local_and_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_io(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("cohort evaluation must remain local and in-memory")

    monkeypatch.setattr(builtins, "open", fail_io)
    monkeypatch.setattr(urllib.request, "urlopen", fail_io)
    definition = _definition(
        (CohortCriterion("clinical.rule", CriterionKind.INCLUSION),)
    )
    record = _record((_evidence("clinical.rule", EvidenceAssertion.MET, EVIDENCE_A),))

    explanation = explain_cohort_membership(record, definition)

    assert explanation.membership_state is MembershipState.ELIGIBLE
