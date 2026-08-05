"""Synthetic offline Acc@5 evaluation for multilingual grounding."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.grounding import (
    MultilingualGroundingResult,
    ground_multilingual,
)

MULTILINGUAL_GROUNDING_TOP5_FLOOR = 0.80
SYNTHETIC_MULTILINGUAL_GROUNDING_PROVENANCE = "synthetic-cc0"


@dataclass(frozen=True)
class MultilingualGroundingCase:
    """One synthetic source-language mention and international-code gold."""

    case_id: str
    mention: str
    locale: str
    expected_system: str
    expected_code: str
    provenance: str = SYNTHETIC_MULTILINGUAL_GROUNDING_PROVENANCE


@dataclass(frozen=True)
class MultilingualGroundingReport:
    """Per-language top-5 accuracy with aggregate gate status."""

    per_language_acc_at_5: dict[str, float]
    per_language_hits: dict[str, int]
    per_language_total: dict[str, int]
    overall_acc_at_5: float
    floor: float
    synthetic_provenance: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a raw-text-free, JSON-serializable report."""

        return {
            "per_language_acc_at_5": dict(self.per_language_acc_at_5),
            "per_language_hits": dict(self.per_language_hits),
            "per_language_total": dict(self.per_language_total),
            "overall_acc_at_5": self.overall_acc_at_5,
            "floor": self.floor,
            "synthetic_provenance": self.synthetic_provenance,
            "passed": self.passed,
        }


SYNTHETIC_MULTILINGUAL_GROUNDING_GOLD: tuple[MultilingualGroundingCase, ...] = (
    MultilingualGroundingCase(
        "zh-icd-diabetes", "2型糖尿病", "zh-CN", "ICD10", "E11.9"
    ),
    MultilingualGroundingCase(
        "zh-icd-hypertension", "原发性高血压", "zh-CN", "ICD10", "I10"
    ),
    MultilingualGroundingCase("zh-icd-pneumonia", "肺炎", "zh-CN", "ICD10", "J18.9"),
    MultilingualGroundingCase("zh-hpo-fever", "发热", "zh-CN", "HPO", "HP:0001945"),
    MultilingualGroundingCase("zh-hpo-headache", "头痛", "zh-CN", "HPO", "HP:0002315"),
    MultilingualGroundingCase("hi-hpo-fever", "बुखार", "hi-IN", "HPO", "HP:0001945"),
    MultilingualGroundingCase(
        "hi-hpo-headache", "सिरदर्द", "hi-IN", "HPO", "HP:0002315"
    ),
    MultilingualGroundingCase(
        "hi-hpo-weakness", "मांसपेशियों में कमजोरी", "hi-IN", "HPO", "HP:0001324"
    ),
    MultilingualGroundingCase("bn-hpo-fever", "জ্বর", "bn-IN", "HPO", "HP:0001945"),
    MultilingualGroundingCase("bn-hpo-headache", "মাথাব্যথা", "bn-IN", "HPO", "HP:0002315"),
    MultilingualGroundingCase(
        "bn-hpo-weakness", "পেশী দুর্বলতা", "bn-IN", "HPO", "HP:0001324"
    ),
    MultilingualGroundingCase("ta-hpo-fever", "காய்ச்சல்", "ta-IN", "HPO", "HP:0001945"),
    MultilingualGroundingCase(
        "ta-hpo-headache", "தலைவலி", "ta-IN", "HPO", "HP:0002315"
    ),
    MultilingualGroundingCase(
        "ta-hpo-weakness", "தசை பலவீனம்", "ta-IN", "HPO", "HP:0001324"
    ),
    MultilingualGroundingCase("te-hpo-fever", "జ్వరం", "te-IN", "HPO", "HP:0001945"),
    MultilingualGroundingCase("te-hpo-headache", "తలనొప్పి", "te-IN", "HPO", "HP:0002315"),
    MultilingualGroundingCase(
        "te-hpo-weakness", "కండరాల బలహీనత", "te-IN", "HPO", "HP:0001324"
    ),
)

Grounder = Callable[[str, str], MultilingualGroundingResult]


def run_multilingual_grounding_eval(
    *,
    cases: Sequence[MultilingualGroundingCase] = SYNTHETIC_MULTILINGUAL_GROUNDING_GOLD,
    grounder: Grounder = ground_multilingual,
    floor: float = MULTILINGUAL_GROUNDING_TOP5_FLOOR,
) -> MultilingualGroundingReport:
    """Measure top-5 international-code accuracy for every source language."""

    if not cases:
        raise ValueError("multilingual grounding evaluation requires gold cases")
    if not 0.0 <= floor <= 1.0:
        raise ValueError("floor must be between 0.0 and 1.0")

    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for case in cases:
        result = grounder(case.mention, case.locale)
        language = result.source_language
        totals[language] += 1
        ranked = {
            (candidate.system, candidate.code) for candidate in result.candidates[:5]
        }
        if (case.expected_system, case.expected_code) in ranked:
            hits[language] += 1

    languages = sorted(totals)
    per_language = {
        language: hits[language] / totals[language] for language in languages
    }
    total = sum(totals.values())
    total_hits = sum(hits.values())
    synthetic = all(
        case.provenance == SYNTHETIC_MULTILINGUAL_GROUNDING_PROVENANCE for case in cases
    )
    passed = synthetic and all(score >= floor for score in per_language.values())
    return MultilingualGroundingReport(
        per_language_acc_at_5=per_language,
        per_language_hits={language: hits[language] for language in languages},
        per_language_total={language: totals[language] for language in languages},
        overall_acc_at_5=total_hits / total,
        floor=floor,
        synthetic_provenance=synthetic,
        passed=passed,
    )


__all__ = [
    "MULTILINGUAL_GROUNDING_TOP5_FLOOR",
    "MultilingualGroundingCase",
    "MultilingualGroundingReport",
    "SYNTHETIC_MULTILINGUAL_GROUNDING_GOLD",
    "SYNTHETIC_MULTILINGUAL_GROUNDING_PROVENANCE",
    "run_multilingual_grounding_eval",
]
