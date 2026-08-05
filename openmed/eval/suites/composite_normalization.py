"""Synthetic offline evaluation for composite normalization and re-linking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.grounding.ranker import CandidateRankingStage, RankingConfig
from openmed.clinical.grounding.vocab import VocabConcept, VocabularyIndex
from openmed.clinical.normalization.composite import (
    KNOWN_ATOMIC_COMPOSITES,
    normalize_composite,
)

COMPOSITE_NORMALIZATION = "composite_normalization"

_COMPOSITE_SURFACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nausea and vomiting", ("nausea", "vomiting")),
    ("fever and chills", ("fever", "chills")),
    ("cough and wheezing", ("cough", "wheezing")),
    ("headache and dizziness", ("headache", "dizziness")),
    ("fatigue and weakness", ("fatigue", "weakness")),
    ("constipation or diarrhea", ("constipation", "diarrhea")),
    ("rash or pruritus", ("rash", "pruritus")),
    ("edema and swelling", ("edema", "swelling")),
    ("tremor and rigidity", ("tremor", "rigidity")),
    ("dysuria and hematuria", ("dysuria", "hematuria")),
    ("nausée and vomiting", ("nausée", "vomiting")),
    ("fatigue, weakness and malaise", ("fatigue", "weakness", "malaise")),
    ("cough; dyspnea", ("cough", "dyspnea")),
    ("insomnia & anxiety", ("insomnia", "anxiety")),
    (
        "type 2 diabetes with diabetic nephropathy",
        ("type 2 diabetes", "diabetic nephropathy"),
    ),
    (
        "hypertension with chronic kidney disease",
        ("hypertension", "chronic kidney disease"),
    ),
    ("asthma with bronchospasm", ("asthma", "bronchospasm")),
    (
        "pneumonia complicated by pleural effusion",
        ("pneumonia", "pleural effusion"),
    ),
    ("cirrhosis complicated by ascites", ("cirrhosis", "ascites")),
    ("influenza associated with myalgia", ("influenza", "myalgia")),
    ("heart failure with pulmonary edema", ("heart failure", "pulmonary edema")),
    (
        "dermatitis with secondary infection",
        ("dermatitis", "secondary infection"),
    ),
    ("migraine with photophobia", ("migraine", "photophobia")),
    (
        "anemia associated with iron deficiency",
        ("anemia", "iron deficiency"),
    ),
    (
        "osteoarthritis with joint effusion",
        ("osteoarthritis", "joint effusion"),
    ),
    (
        "pancreatitis complicated by pseudocyst",
        ("pancreatitis", "pseudocyst"),
    ),
    (
        "chronic obstructive pulmonary disease exacerbation",
        ("chronic obstructive pulmonary disease", "exacerbation"),
    ),
    ("asthma acute exacerbation", ("asthma", "acute exacerbation")),
    ("eczema flare", ("eczema", "flare")),
    ("lupus flare", ("lupus", "flare")),
    ("heart failure exacerbation", ("heart failure", "exacerbation")),
    ("ulcer complication", ("ulcer", "complication")),
    ("diabetes complication", ("diabetes", "complication")),
    ("arthritis flare", ("arthritis", "flare")),
    ("bronchitis exacerbation", ("bronchitis", "exacerbation")),
    ("colitis acute exacerbation", ("colitis", "acute exacerbation")),
)


@dataclass(frozen=True)
class CompositeNormalizationCase:
    """One synthetic composite mention with child surfaces and gold codes."""

    mention: str
    children: tuple[str, ...]
    gold_codes: tuple[str, ...]


class _SyntheticVocabLoader:
    def __init__(self, index: VocabularyIndex) -> None:
        self._index = index

    def get_index(self, system: str) -> VocabularyIndex:
        if system.casefold() != self._index.system:
            raise ValueError(f"unsupported synthetic system {system!r}")
        return self._index


def build_composite_normalization_gold() -> tuple[CompositeNormalizationCase, ...]:
    """Return at least 30 deterministic synthetic composite gold cases."""

    terms = sorted({child for _, children in _COMPOSITE_SURFACES for child in children})
    codes = {term: f"SYN{index:04d}" for index, term in enumerate(terms, start=1)}
    return tuple(
        CompositeNormalizationCase(
            mention=mention,
            children=children,
            gold_codes=tuple(codes[child] for child in children),
        )
        for mention, children in _COMPOSITE_SURFACES
    )


def composite_normalization_metadata() -> dict[str, Any]:
    """Return synthetic-data and offline provenance for this suite."""

    return {
        "suite": COMPOSITE_NORMALIZATION,
        "source": "synthetic composite clinical surfaces",
        "redistribution": "safe; no DUA or production terminology",
        "offline": True,
    }


def evaluate_composite_normalization(
    cases: Sequence[CompositeNormalizationCase] | None = None,
) -> dict[str, Any]:
    """Measure split offsets, top-1 linking, and over/under-split rates.

    Child concepts are loaded into an in-memory synthetic ICD-10-CM-like index
    and linked through the production candidate generation plus ranking stage.
    The corpus and vocabulary are constructed locally and never perform network
    access.
    """

    gold = tuple(cases) if cases is not None else build_composite_normalization_gold()
    if len(gold) < 30:
        raise ValueError("composite normalization gold set must contain >=30 cases")
    term_codes = _term_code_map(gold)
    stage = _synthetic_ranking_stage(term_codes)

    concept_total = sum(len(case.children) for case in gold)
    top1_correct = 0
    over_split = 0
    under_split = 0
    offset_correct = 0
    offset_total = 0

    def ranked(surface: str):
        return stage.rank(surface, systems=("icd10cm",))

    for case in gold:
        normalized = normalize_composite(
            case.mention,
            is_linkable=lambda surface: bool(ranked(surface)),
        )
        predicted = normalized.children
        over_split += int(len(predicted) > len(case.children))
        under_split += int(len(predicted) < len(case.children))

        for index, expected in enumerate(case.children):
            if index >= len(predicted) or predicted[index].text != expected:
                continue
            candidates = ranked(predicted[index].text)
            if candidates and candidates[0].candidate.code == case.gold_codes[index]:
                top1_correct += 1

        encoded = case.mention.encode("utf-8")
        for child in predicted:
            offset_total += 1
            byte_surface = encoded[child.byte_start : child.byte_end].decode("utf-8")
            offset_correct += int(
                byte_surface == child.text
                and case.mention[child.start : child.end] == child.text
            )

    atomic_false_splits = sum(
        int(normalize_composite(term).was_split)
        for term in sorted(KNOWN_ATOMIC_COMPOSITES)
    )
    case_count = len(gold)
    return {
        "suite": COMPOSITE_NORMALIZATION,
        "case_count": case_count,
        "concept_count": concept_total,
        "top1_accuracy": top1_correct / concept_total,
        "over_split_rate": over_split / case_count,
        "under_split_rate": under_split / case_count,
        "offset_accuracy": offset_correct / offset_total if offset_total else 0.0,
        "atomic_term_count": len(KNOWN_ATOMIC_COMPOSITES),
        "atomic_false_splits": atomic_false_splits,
        "metadata": composite_normalization_metadata(),
    }


def run_composite_normalization(
    *,
    cases: Sequence[CompositeNormalizationCase] | None = None,
) -> dict[str, Any]:
    """Run the synthetic composite-normalization suite and return its report."""

    return evaluate_composite_normalization(cases)


def _term_code_map(
    cases: Sequence[CompositeNormalizationCase],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for case in cases:
        if len(case.children) != len(case.gold_codes):
            raise ValueError("composite child surfaces and gold codes must align")
        for term, code in zip(case.children, case.gold_codes):
            existing = mapping.setdefault(term, code)
            if existing != code:
                raise ValueError(f"conflicting synthetic codes for {term!r}")
    return mapping


def _synthetic_ranking_stage(term_codes: dict[str, str]) -> CandidateRankingStage:
    concepts = tuple(
        VocabConcept(
            system="icd10cm",
            code=code,
            preferred_term=term,
        )
        for term, code in sorted(term_codes.items())
    )
    index = VocabularyIndex("icd10cm", concepts)
    loader = _SyntheticVocabLoader(index)
    return CandidateRankingStage(
        loader,  # type: ignore[arg-type]
        config=RankingConfig(systems=("icd10cm",)),
    )
