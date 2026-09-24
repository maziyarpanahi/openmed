"""Synthetic false-positive stress gate for patient-level SDOH findings.

The built-in cases are repository-authored and run through local section and
experiencer guards. Reports contain only controlled categories and counts;
source text and finding values never enter a report or exception.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from openmed.clinical.sdoh import SDOHFinding, extract_sdoh
from openmed.clinical.sdoh_experiencer import filter_sdoh_findings
from openmed.clinical.sections import detect_sections
from openmed.training.synthetic.social_history import SOCIAL_HISTORY_CATEGORIES

SDOH_STRESS_SOURCE = "openmed.synthetic.sdoh_false_positive_stress"
SDOH_STRESS_PATTERNS = (
    "screening",
    "education",
    "boilerplate",
    "third_party",
    "outside_section",
    "negated",
    "historical",
)

_CUES = {
    "alcohol": (
        "alcohol use",
        "drinks alcohol",
        "denies alcohol use",
        "former alcohol use, stopped in 2019",
    ),
    "drug": (
        "drug use",
        "uses recreational drugs",
        "denies drug use",
        "former drug use, stopped in 2019",
    ),
    "tobacco": (
        "tobacco use",
        "uses tobacco",
        "denies tobacco use",
        "former smoker, quit in 2019",
    ),
    "employment": (
        "employment status",
        "is unemployed",
        "is not unemployed",
        "previously unemployed, now employed",
    ),
    "living_status": (
        "housing status",
        "is homeless",
        "is not homeless",
        "formerly homeless, now housed",
    ),
}


@dataclass(frozen=True)
class SDOHStressCase:
    """One explicitly synthetic hard negative for a controlled category."""

    category: str
    pattern: str
    text: str = field(repr=False)
    synthetic: bool = True
    source: str = SDOH_STRESS_SOURCE


@dataclass(frozen=True)
class SDOHStressPrediction:
    """Patient-level findings and count of attempted eligibility actions."""

    findings: tuple[SDOHFinding, ...] = field(repr=False)
    automated_eligibility_actions: int = 0


@dataclass(frozen=True)
class SDOHCategoryStressResult:
    """Aggregate false-positive count and gate for one controlled category."""

    category: str
    case_count: int
    false_positive_count: int
    max_false_positive_rate: float

    @property
    def false_positive_rate(self) -> float:
        """Fraction of hard negatives yielding a positive patient finding."""

        return self.false_positive_count / self.case_count if self.case_count else 0.0

    @property
    def passed(self) -> bool:
        """Whether this category meets its configured ceiling."""

        return self.false_positive_rate <= self.max_false_positive_rate

    def to_dict(self) -> dict[str, Any]:
        """Return count-only category evidence."""

        return {
            "category": self.category,
            "case_count": self.case_count,
            "false_positive_count": self.false_positive_count,
            "false_positive_rate": self.false_positive_rate,
            "max_false_positive_rate": self.max_false_positive_rate,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class SDOHStressReport:
    """Value-free per-category stress evidence and hard action gate."""

    categories: tuple[SDOHCategoryStressResult, ...]
    automated_eligibility_actions: int

    @property
    def case_count(self) -> int:
        """Total number of synthetic hard negatives evaluated."""

        return sum(result.case_count for result in self.categories)

    @property
    def passed(self) -> bool:
        """Require every rate ceiling and zero automated eligibility actions."""

        return self.automated_eligibility_actions == 0 and all(
            result.passed for result in self.categories
        )

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate counts without text, spans, or finding values."""

        return {
            "case_count": self.case_count,
            "categories": [result.to_dict() for result in self.categories],
            "automated_eligibility_actions": self.automated_eligibility_actions,
            "passed": self.passed,
        }


class SDOHStressGateError(AssertionError):
    """The synthetic SDOH false-positive or automated-action gate failed."""


SDOHPredictor = Callable[[str], SDOHStressPrediction]


def default_sdoh_hard_negatives() -> tuple[SDOHStressCase, ...]:
    """Create deterministic hard negatives across five SDOH categories.

    Returns:
        Synthetic cases covering screening, education, boilerplate,
        third-party, section, negated, and historical language.
    """

    cases: list[SDOHStressCase] = []
    for category in SOCIAL_HISTORY_CATEGORIES:
        cue, third_party, negated, historical = _CUES[category]
        education = {
            "employment": "Education: review employment resources.",
            "living_status": "Education: review housing resources.",
        }.get(category, f"Education: avoid {cue}.")
        boilerplate = {
            "employment": "Employment status: [ ] employed [ ] unemployed.",
            "living_status": "Housing status: [ ] housed [ ] homeless.",
        }.get(category, f"{cue}: [ ] Yes [ ] No.")
        clauses = (
            ("screening", f"Social History:\nAsk about {cue} at next visit."),
            ("education", f"Social History:\n{education}"),
            ("boilerplate", f"Social History:\n{boilerplate}"),
            ("third_party", f"Social History:\nMother {third_party}."),
            ("outside_section", f"Assessment:\nPatient {third_party}."),
            ("negated", f"Social History:\nPatient {negated}."),
            ("historical", f"Social History:\nPatient {historical}."),
        )
        cases.extend(
            SDOHStressCase(category=category, pattern=pattern, text=text)
            for pattern, text in clauses
        )
    return tuple(cases)


def run_sdoh_false_positive_stress(
    cases: Sequence[SDOHStressCase] | None = None,
    *,
    predictor: SDOHPredictor | None = None,
    max_false_positive_rates: Mapping[str, float] | None = None,
) -> SDOHStressReport:
    """Measure patient-level false positives with per-category zero defaults.

    Args:
        cases: Synthetic hard negatives. Defaults to the built-in matrix.
        predictor: Local predictor returning findings and any attempted action
            count. Defaults to OpenMed extraction with its section and
            experiencer guards; it never takes an eligibility action.
        max_false_positive_rates: Optional per-category ceilings. Every
            unspecified category keeps the default ceiling of zero.

    Raises:
        TypeError: If a predictor emits an invalid prediction type.
        ValueError: If a fixture, rate, or action count is invalid.
    """

    challenge = default_sdoh_hard_negatives() if cases is None else tuple(cases)
    local_predictor = _patient_prediction if predictor is None else predictor
    if not callable(local_predictor):
        raise TypeError("predictor must be callable")
    ceilings = _ceilings(max_false_positive_rates)
    counts = dict.fromkeys(SOCIAL_HISTORY_CATEGORIES, 0)
    false_positives = dict.fromkeys(SOCIAL_HISTORY_CATEGORIES, 0)
    actions = 0
    for case in challenge:
        _validate_case(case)
        counts[case.category] += 1
        try:
            prediction = local_predictor(case.text)
        except Exception:
            raise RuntimeError("SDOH stress predictor failed") from None
        _validate_prediction(prediction)
        actions += prediction.automated_eligibility_actions
        if any(_is_positive(finding) for finding in prediction.findings):
            false_positives[case.category] += 1

    return SDOHStressReport(
        categories=tuple(
            SDOHCategoryStressResult(
                category=category,
                case_count=counts[category],
                false_positive_count=false_positives[category],
                max_false_positive_rate=ceilings[category],
            )
            for category in SOCIAL_HISTORY_CATEGORIES
        ),
        automated_eligibility_actions=actions,
    )


def assert_sdoh_stress_gate(report: SDOHStressReport) -> None:
    """Raise a value-free error if any category or the action gate fails."""

    if not isinstance(report, SDOHStressReport):
        raise TypeError("report must be an SDOHStressReport")
    if not report.passed:
        raise SDOHStressGateError("SDOH synthetic false-positive stress gate failed")


def _patient_prediction(text: str) -> SDOHStressPrediction:
    sections = detect_sections(text)
    findings = tuple(extract_sdoh(text, (), sections=sections))
    partition = filter_sdoh_findings(text, findings, sections=sections)
    return SDOHStressPrediction(
        findings=tuple(
            findings[record.input_index]
            for record in partition.patient_evidence
            if record.input_index is not None
        )
    )


def _validate_case(case: SDOHStressCase) -> None:
    if not isinstance(case, SDOHStressCase):
        raise TypeError("cases must contain SDOHStressCase values")
    if case.category not in SOCIAL_HISTORY_CATEGORIES:
        raise ValueError("stress case has unsupported category")
    if case.pattern not in SDOH_STRESS_PATTERNS:
        raise ValueError("stress case has unsupported pattern")
    if not case.synthetic or case.source != SDOH_STRESS_SOURCE:
        raise ValueError("stress cases must be repository-authored synthetic data")
    if not isinstance(case.text, str) or not case.text:
        raise ValueError("stress case text must be non-empty")


def _validate_prediction(prediction: SDOHStressPrediction) -> None:
    if not isinstance(prediction, SDOHStressPrediction):
        raise TypeError("predictor must return SDOHStressPrediction")
    if any(not isinstance(item, SDOHFinding) for item in prediction.findings):
        raise TypeError("predictor findings must contain SDOHFinding values")
    actions = prediction.automated_eligibility_actions
    if isinstance(actions, bool) or not isinstance(actions, int) or actions < 0:
        raise ValueError("automated eligibility action count must be non-negative")


def _ceilings(rates: Mapping[str, float] | None) -> dict[str, float]:
    ceilings = dict.fromkeys(SOCIAL_HISTORY_CATEGORIES, 0.0)
    if rates is None:
        return ceilings
    if not isinstance(rates, Mapping):
        raise TypeError("max_false_positive_rates must be a mapping")
    for category, value in rates.items():
        if category not in ceilings:
            raise ValueError("false-positive ceiling has unsupported category")
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError("false-positive ceiling must be between zero and one")
        if not 0.0 <= value <= 1.0:
            raise ValueError("false-positive ceiling must be between zero and one")
        ceilings[category] = float(value)
    return ceilings


def _is_positive(finding: SDOHFinding) -> bool:
    return finding.status in {"current", "unemployed", "homeless", "present"}


__all__ = [
    "SDOHCategoryStressResult",
    "SDOHStressCase",
    "SDOHStressGateError",
    "SDOHStressPrediction",
    "SDOHStressReport",
    "SDOH_STRESS_PATTERNS",
    "SDOH_STRESS_SOURCE",
    "assert_sdoh_stress_gate",
    "default_sdoh_hard_negatives",
    "run_sdoh_false_positive_stress",
]
