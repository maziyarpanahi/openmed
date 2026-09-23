"""Synthetic tests for severity-weighted summary omission budgets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pytest

from openmed.clinical.summary_omission_budget import (
    CLASS_BUDGET_REFUSAL,
    MANDATORY_OMISSION_REFUSAL,
    ImportanceClassPolicy,
    SummaryEvidenceCoverage,
    SummaryOmissionBudgetError,
    evaluate_summary_omission_budget,
)


def _opaque(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode()).hexdigest()


MANDATORY = ImportanceClassPolicy(
    class_id=_opaque("mandatory"),
    severity_weight=10,
    mandatory=True,
)
IMPORTANT = ImportanceClassPolicy(
    class_id=_opaque("important"),
    severity_weight=4,
    omission_limit=1,
)
ROUTINE = ImportanceClassPolicy(
    class_id=_opaque("routine"),
    severity_weight=1,
    omission_limit=2,
)


def _fact(label: str, policy: ImportanceClassPolicy, represented: bool = True):
    return SummaryEvidenceCoverage(
        evidence_id=_opaque(label),
        importance_class_id=policy.class_id,
        represented=represented,
    )


def test_weighted_budget_and_omissions_are_calculated_by_class() -> None:
    report = evaluate_summary_omission_budget(
        [
            _fact("mandatory-a", MANDATORY),
            _fact("important-a", IMPORTANT, represented=False),
            _fact("important-b", IMPORTANT),
            _fact("routine-a", ROUTINE, represented=False),
        ],
        [ROUTINE, MANDATORY, IMPORTANT],
    )

    assert report.passed is True
    assert report.refusal_code is None
    assert report.fact_count == 4
    assert report.omitted_count == 2
    assert report.weighted_omissions == 5
    assert report.weighted_budget == 6
    assert [item.class_id for item in report.classes] == sorted(
        [MANDATORY.class_id, IMPORTANT.class_id, ROUTINE.class_id]
    )


def test_mandatory_omission_fails_independent_of_aggregate_coverage() -> None:
    evidence = [_fact("mandatory-a", MANDATORY, represented=False)]
    evidence.extend(_fact(f"routine-{index}", ROUTINE) for index in range(100))

    report = evaluate_summary_omission_budget(
        evidence,
        [MANDATORY, ROUTINE],
    )

    assert report.passed is False
    assert report.refusal_code == MANDATORY_OMISSION_REFUSAL
    assert report.omitted_count == 1
    assert report.fact_count == 101


def test_class_budget_is_non_compensable() -> None:
    report = evaluate_summary_omission_budget(
        [
            _fact("important-a", IMPORTANT, represented=False),
            _fact("important-b", IMPORTANT, represented=False),
            *(_fact(f"routine-{index}", ROUTINE) for index in range(20)),
        ],
        [IMPORTANT, ROUTINE],
    )

    assert report.weighted_omissions == 8
    assert report.weighted_budget == 6
    assert report.passed is False
    assert report.refusal_code == CLASS_BUDGET_REFUSAL
    important_result = next(
        result for result in report.classes if result.class_id == IMPORTANT.class_id
    )
    assert important_result.passed is False


def test_empty_evidence_is_deterministic_and_preserves_configured_classes() -> None:
    first = evaluate_summary_omission_budget([], [ROUTINE, MANDATORY])
    second = evaluate_summary_omission_budget([], [MANDATORY, ROUTINE])

    assert first == second
    assert first.passed is True
    assert first.to_json() == second.to_json()
    assert json.loads(first.to_json()) == first.to_dict()


def test_reports_and_errors_never_contain_raw_sensitive_values() -> None:
    sentinel = "SYNTHETIC_PATIENT_DIAGNOSIS"
    report = evaluate_summary_omission_budget(
        [_fact(sentinel, MANDATORY, represented=False)],
        [MANDATORY],
    )

    assert sentinel not in report.to_json()
    assert "evidence_id" not in report.to_json()

    with pytest.raises(SummaryOmissionBudgetError) as error:
        SummaryEvidenceCoverage(
            evidence_id=sentinel,
            importance_class_id=MANDATORY.class_id,
            represented=False,
        )
    assert sentinel not in str(error.value)


def test_unknown_classes_duplicates_and_invalid_policy_fail_closed() -> None:
    with pytest.raises(SummaryOmissionBudgetError, match="unconfigured"):
        evaluate_summary_omission_budget(
            [_fact("evidence", ImportanceClassPolicy(_opaque("other"), 1))],
            [ROUTINE],
        )
    duplicate = _fact("duplicate", ROUTINE)
    with pytest.raises(SummaryOmissionBudgetError, match="duplicate evidence"):
        evaluate_summary_omission_budget([duplicate, duplicate], [ROUTINE])
    with pytest.raises(SummaryOmissionBudgetError, match="zero omission"):
        ImportanceClassPolicy(_opaque("invalid"), 1, omission_limit=1, mandatory=True)
    with pytest.raises(SummaryOmissionBudgetError, match="collection"):
        evaluate_summary_omission_budget([], [])


def test_public_records_are_immutable() -> None:
    record = _fact("immutable", ROUTINE)
    report = evaluate_summary_omission_budget([record], [ROUTINE])

    with pytest.raises(FrozenInstanceError):
        record.represented = False  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        report.passed = False  # type: ignore[misc]
