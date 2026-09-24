"""Synthetic, offline hard-negative gates for patient-level SDOH findings."""

from __future__ import annotations

from dataclasses import replace

import pytest

from openmed.clinical.sdoh import SDOHFinding, extract_sdoh
from openmed.clinical.sections import detect_sections
from openmed.eval.sdoh_false_positive_stress import (
    SDOHStressGateError,
    SDOHStressPrediction,
    assert_sdoh_stress_gate,
    default_sdoh_hard_negatives,
    run_sdoh_false_positive_stress,
)
from openmed.training.synthetic.social_history import SOCIAL_HISTORY_CATEGORIES


def _positive() -> SDOHFinding:
    return SDOHFinding(
        category="tobacco",
        value="tobacco",
        status="current",
        extent=None,
        temporality="recent",
        span=(0, 1),
        score=0.9,
    )


def test_builtin_hard_negatives_cover_categories_and_risk_patterns() -> None:
    cases = default_sdoh_hard_negatives()

    assert cases == default_sdoh_hard_negatives()
    assert len(cases) == 35
    assert {case.category for case in cases} == set(SOCIAL_HISTORY_CATEGORIES)
    for category in SOCIAL_HISTORY_CATEGORIES:
        assert {case.pattern for case in cases if case.category == category} == {
            "screening",
            "education",
            "boilerplate",
            "third_party",
            "outside_section",
            "negated",
            "historical",
        }
    assert all(case.synthetic and case.text for case in cases)


def test_default_patient_pipeline_passes_every_category_with_zero_actions() -> None:
    report = run_sdoh_false_positive_stress()

    assert report.case_count == 35
    assert all(result.case_count == 7 for result in report.categories)
    assert all(result.false_positive_count == 0 for result in report.categories)
    assert report.automated_eligibility_actions == 0
    assert report.passed
    assert_sdoh_stress_gate(report)


def test_each_category_has_its_own_false_positive_ceiling() -> None:
    tobacco = tuple(
        case for case in default_sdoh_hard_negatives() if case.category == "tobacco"
    )

    def unsafe_predictor(text: str) -> SDOHStressPrediction:
        return (
            SDOHStressPrediction((_positive(),))
            if "Ask about" in text
            else SDOHStressPrediction(())
        )

    strict = run_sdoh_false_positive_stress(tobacco, predictor=unsafe_predictor)
    tolerant = run_sdoh_false_positive_stress(
        tobacco,
        predictor=unsafe_predictor,
        max_false_positive_rates={"tobacco": 1 / 7},
    )

    assert strict.categories[2].category == "tobacco"
    assert strict.categories[2].false_positive_count == 1
    assert strict.categories[2].false_positive_rate == pytest.approx(1 / 7)
    assert not strict.passed
    assert tolerant.passed
    with pytest.raises(SDOHStressGateError, match="stress gate failed"):
        assert_sdoh_stress_gate(strict)


def test_automated_eligibility_action_gate_cannot_be_relaxed() -> None:
    [case] = default_sdoh_hard_negatives()[:1]

    report = run_sdoh_false_positive_stress(
        [case],
        predictor=lambda text: SDOHStressPrediction(
            (), automated_eligibility_actions=1
        ),
        max_false_positive_rates={"alcohol": 1.0},
    )

    assert report.automated_eligibility_actions == 1
    assert not report.passed
    with pytest.raises(SDOHStressGateError, match="stress gate failed"):
        assert_sdoh_stress_gate(report)


def test_report_and_errors_never_echo_case_or_finding_values() -> None:
    [case] = default_sdoh_hard_negatives()[:1]
    report = run_sdoh_false_positive_stress(
        [case], predictor=lambda text: SDOHStressPrediction((_positive(),))
    )

    rendered = repr(report.to_dict())
    assert case.text not in rendered
    assert "Social History" not in rendered
    assert "Mother" not in rendered
    assert "tobacco" in rendered  # Controlled category name only.

    with pytest.raises(ValueError, match="repository-authored synthetic"):
        run_sdoh_false_positive_stress([replace(case, synthetic=False)])
    with pytest.raises(ValueError, match="ceiling"):
        run_sdoh_false_positive_stress(
            max_false_positive_rates={"alcohol": float("nan")}
        )

    def leaking_predictor(text: str) -> SDOHStressPrediction:
        raise ValueError(text)

    with pytest.raises(RuntimeError, match="predictor failed") as error:
        run_sdoh_false_positive_stress([case], predictor=leaking_predictor)
    assert case.text not in str(error.value)


def test_noncurrent_findings_are_not_counted_as_positive() -> None:
    [case] = default_sdoh_hard_negatives()[:1]
    past = replace(_positive(), status="past")
    report = run_sdoh_false_positive_stress(
        [case], predictor=lambda text: SDOHStressPrediction((past,))
    )

    assert report.passed
    assert report.categories[0].false_positive_count == 0


def test_nonassertive_clause_does_not_hide_adjacent_patient_assertion() -> None:
    text = (
        "Social History:\nAsk about alcohol use at next visit. "
        "Patient drinks alcohol daily."
    )
    findings = extract_sdoh(text, (), sections=detect_sections(text))

    assert [(finding.category, finding.status) for finding in findings] == [
        ("alcohol", "current")
    ]


def test_negated_and_historical_substance_findings_remain_noncurrent() -> None:
    text = (
        "Social History:\nPatient denies tobacco use. "
        "Former alcohol use, stopped in 2019."
    )
    findings = extract_sdoh(text, (), sections=detect_sections(text))

    assert {(finding.category, finding.status) for finding in findings} == {
        ("tobacco", "none"),
        ("alcohol", "past"),
    }


def test_double_negated_unemployment_stays_unresolved() -> None:
    text = "Social History:\nPatient is not unemployed."
    findings = extract_sdoh(text, (), sections=detect_sections(text))

    assert [(finding.category, finding.status) for finding in findings] == [
        ("employment", "unknown")
    ]
