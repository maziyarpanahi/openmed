"""Deterministic cross-lingual grounding agreement metrics.

This standalone suite scores how well non-English clinical mentions ground to
the same international codes as their English equivalents, closing the OM-751
multilingual-grounding chain. It reads the synthetic ``grounding_crosslingual``
gold (es/fr/de/zh aliases plus per-language ``gold_mentions`` and an
``expected_code``) and, for every language, links each mention through the
offline exact-alias linker stack.

The suite reports per-language top-1 accuracy (with an explicit English
baseline), macro/micro aggregation over the non-English languages, and a
cross-lingual consistency score measuring how often a non-English mention lands
on the same code as its English equivalent. A leakage-first scan rejects any
committed alias or gold text that carries restricted or DUA corpus markers.

All lookups are exact-alias and offline (no fuzzy stage, no network, no
randomness), so every reported number is byte-identical across runs.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.clinical.grounding import VocabLoader, VocabSource
from openmed.clinical.grounding.linkers.hpo import HpoLinker
from openmed.clinical.grounding.linkers.icd10cm import Icd10cmLinker
from openmed.clinical.grounding.linkers.rxnorm import RxNormLinker
from openmed.clinical.grounding.vocab import VocabularyNotFoundError

#: Registry key for the standalone cross-lingual grounding suite.
CROSS_LINGUAL_GROUNDING = "cross_lingual_grounding"

#: Language whose grounded code the non-English mentions are compared against.
ENGLISH_LANGUAGE = "en"

#: Reporting order for every language covered by the synthetic gold.
CROSS_LINGUAL_LANGUAGES: tuple[str, ...] = ("en", "es", "fr", "de", "zh")

#: Non-English languages aggregated by the macro/micro top-1 scores.
NON_ENGLISH_LANGUAGES: tuple[str, ...] = ("es", "fr", "de", "zh")

#: Minimum acceptable per-language top-1 accuracy for a non-English language.
NON_ENGLISH_TOP1_FLOOR = 0.80

#: Provenance every synthetic gold row must declare.
SYNTHETIC_PROVENANCE = "synthetic-permissive"

#: Restricted or DUA corpus markers that must never appear in committed gold.
RESTRICTED_CORPUS_MARKERS: tuple[str, ...] = (
    "mimic",
    "i2b2",
    "n2c2",
    "umls",
    "snomed",
    "cpt",
    "medical record number",
    "discharge summary",
)

#: A ranked-code grounder: ``(system, mention, language) -> ranked codes``.
Grounder = Callable[[str, str, str], Sequence[str]]

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "golden"
    / "fixtures"
    / "grounding_crosslingual.jsonl"
)

_LINKER_FACTORIES: dict[str, Callable[[Any], Any]] = {
    "icd10cm": Icd10cmLinker,
    "rxnorm": RxNormLinker,
    "hpo": HpoLinker,
}


@dataclass(frozen=True)
class CrossLingualGroundingCase:
    """One synthetic concept with per-language gold mentions and gold code."""

    system: str
    expected_code: str
    gold_mentions: Mapping[str, str]
    provenance: str


@dataclass(frozen=True)
class CrossLingualGroundingReport:
    """Per-language and aggregate cross-lingual grounding agreement scores."""

    languages: tuple[str, ...]
    concept_count: int
    per_language_top1: dict[str, float]
    per_language_hits: dict[str, int]
    per_language_total: dict[str, int]
    english_top1: float
    macro_top1_non_english: float
    micro_top1_non_english: float
    per_language_english_agreement: dict[str, float]
    cross_lingual_consistency: float
    fully_consistent_concepts: int
    non_english_top1_floor: float
    english_baseline_intact: bool
    synthetic_provenance: bool
    restricted_markers: tuple[str, ...]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable, offline-safe report with no raw text."""

        return {
            "languages": list(self.languages),
            "concept_count": self.concept_count,
            "per_language_top1": dict(self.per_language_top1),
            "per_language_hits": dict(self.per_language_hits),
            "per_language_total": dict(self.per_language_total),
            "english_top1": self.english_top1,
            "macro_top1_non_english": self.macro_top1_non_english,
            "micro_top1_non_english": self.micro_top1_non_english,
            "per_language_english_agreement": dict(self.per_language_english_agreement),
            "cross_lingual_consistency": self.cross_lingual_consistency,
            "fully_consistent_concepts": self.fully_consistent_concepts,
            "non_english_top1_floor": self.non_english_top1_floor,
            "english_baseline_intact": self.english_baseline_intact,
            "synthetic_provenance": self.synthetic_provenance,
            "restricted_markers": list(self.restricted_markers),
            "passed": self.passed,
        }


def load_cross_lingual_grounding_fixtures(
    path: str | Path | None = None,
) -> tuple[CrossLingualGroundingCase, ...]:
    """Load the synthetic cross-lingual grounding gold in file order."""

    fixture_path = Path(path) if path is not None else _FIXTURE
    cases: list[CrossLingualGroundingCase] = []
    for line in fixture_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        mentions = row["gold_mentions"]
        if not isinstance(mentions, Mapping):
            raise ValueError("cross-lingual gold row requires a gold_mentions mapping")
        cases.append(
            CrossLingualGroundingCase(
                system=str(row["system"]),
                expected_code=str(row["expected_code"]),
                gold_mentions={
                    str(language): str(mention)
                    for language, mention in mentions.items()
                },
                provenance=str(row.get("provenance", "")),
            )
        )
    return tuple(cases)


def _case_scan_text(case: CrossLingualGroundingCase) -> str:
    """All text a case carries, for the restricted-marker leakage scan."""

    return " ".join(
        [case.system, case.expected_code, case.provenance, *case.gold_mentions.values()]
    )


def scan_restricted_corpus_markers(text: str) -> tuple[str, ...]:
    """Return restricted/DUA corpus markers present in *text*, in scan order."""

    folded = text.casefold()
    return tuple(marker for marker in RESTRICTED_CORPUS_MARKERS if marker in folded)


def build_fixture_grounder(path: str | Path | None = None) -> Grounder:
    """Build a deterministic exact-alias grounder over the gold fixture.

    The returned callable maps ``(system, mention, language)`` to the ranked
    international codes produced by the offline linker for that system. The
    fixture doubles as the vocabulary, mirroring the language-threaded grounding
    exercised by the synthetic cross-lingual gold.
    """

    fixture_path = Path(path) if path is not None else _FIXTURE
    linkers: dict[str, Any] = {}
    for system, factory in _LINKER_FACTORIES.items():
        loader = VocabLoader(
            registry={system: VocabSource(system=system, path=fixture_path)}
        )
        try:
            index = loader.get_index(system)
        except VocabularyNotFoundError:
            # The gold carries no concepts for this system; skip it so
            # single-system fixtures still build a usable grounder.
            continue
        linkers[system] = factory(index)

    def _ground(system: str, mention: str, language: str) -> Sequence[str]:
        linker = linkers.get(system)
        if linker is None:
            return ()
        candidates = linker.link(mention, language=language, fuzzy=False)
        return [candidate.code for candidate in candidates]

    return _ground


def run_cross_lingual_grounding(
    *,
    cases: Sequence[CrossLingualGroundingCase] | None = None,
    grounder: Grounder | None = None,
    path: str | Path | None = None,
    floor: float = NON_ENGLISH_TOP1_FLOOR,
) -> CrossLingualGroundingReport:
    """Score cross-lingual grounding agreement over the synthetic gold.

    For each concept and language the mention is grounded to a ranked list of
    international codes; a top-1 hit means the leading code equals the concept's
    ``expected_code``. Non-English mentions are additionally checked for
    agreement with the code their English equivalent grounds to. The suite
    passes when the English baseline is intact, every non-English language
    clears ``floor``, the gold is synthetic, and no restricted corpus markers
    leak into the committed fixture.
    """

    fixture_path = Path(path) if path is not None else _FIXTURE
    gold = (
        tuple(cases)
        if cases is not None
        else load_cross_lingual_grounding_fixtures(fixture_path)
    )
    if not gold:
        raise ValueError("cross-lingual grounding evaluation requires gold concepts")
    ground = grounder if grounder is not None else build_fixture_grounder(fixture_path)

    hits = {language: 0 for language in CROSS_LINGUAL_LANGUAGES}
    totals = {language: 0 for language in CROSS_LINGUAL_LANGUAGES}
    agreement_hits = {language: 0 for language in NON_ENGLISH_LANGUAGES}
    agreement_totals = {language: 0 for language in NON_ENGLISH_LANGUAGES}
    fully_consistent = 0

    for case in gold:
        english_mention = case.gold_mentions.get(ENGLISH_LANGUAGE)
        english_codes = (
            ground(case.system, english_mention, ENGLISH_LANGUAGE)
            if english_mention is not None
            else ()
        )
        english_top = english_codes[0] if english_codes else None

        concept_consistent = True
        for language in CROSS_LINGUAL_LANGUAGES:
            mention = case.gold_mentions.get(language)
            if mention is None:
                continue
            codes = ground(case.system, mention, language)
            top = codes[0] if codes else None
            totals[language] += 1
            if top == case.expected_code:
                hits[language] += 1
            if language == ENGLISH_LANGUAGE:
                continue
            agreement_totals[language] += 1
            if top is not None and top == english_top:
                agreement_hits[language] += 1
            else:
                concept_consistent = False
        if concept_consistent:
            fully_consistent += 1

    per_language_top1 = {
        language: _rate(hits[language], totals[language])
        for language in CROSS_LINGUAL_LANGUAGES
    }
    per_language_english_agreement = {
        language: _rate(agreement_hits[language], agreement_totals[language])
        for language in NON_ENGLISH_LANGUAGES
    }
    non_english_hits = sum(hits[language] for language in NON_ENGLISH_LANGUAGES)
    non_english_total = sum(totals[language] for language in NON_ENGLISH_LANGUAGES)
    macro_non_english = sum(
        per_language_top1[language] for language in NON_ENGLISH_LANGUAGES
    ) / len(NON_ENGLISH_LANGUAGES)
    micro_non_english = _rate(non_english_hits, non_english_total)
    agreement_pairs = sum(agreement_hits.values())
    agreement_pair_total = sum(agreement_totals.values())
    consistency = _rate(agreement_pairs, agreement_pair_total)

    english_baseline_intact = per_language_top1[ENGLISH_LANGUAGE] >= 1.0
    synthetic_provenance = all(case.provenance == SYNTHETIC_PROVENANCE for case in gold)
    # Scan the actual scored data (works for in-memory cases too); when loading
    # from a file, also scan the raw fixture so markers in fields the parser
    # drops are still caught.
    scanned_text = "\n".join(_case_scan_text(case) for case in gold)
    if cases is None:
        scanned_text = f"{scanned_text}\n{fixture_path.read_text(encoding='utf-8')}"
    restricted_markers = scan_restricted_corpus_markers(scanned_text)
    non_english_floor_met = all(
        per_language_top1[language] >= floor for language in NON_ENGLISH_LANGUAGES
    )
    passed = (
        english_baseline_intact
        and non_english_floor_met
        and synthetic_provenance
        and not restricted_markers
    )

    return CrossLingualGroundingReport(
        languages=CROSS_LINGUAL_LANGUAGES,
        concept_count=len(gold),
        per_language_top1=per_language_top1,
        per_language_hits={
            language: hits[language] for language in CROSS_LINGUAL_LANGUAGES
        },
        per_language_total={
            language: totals[language] for language in CROSS_LINGUAL_LANGUAGES
        },
        english_top1=per_language_top1[ENGLISH_LANGUAGE],
        macro_top1_non_english=macro_non_english,
        micro_top1_non_english=micro_non_english,
        per_language_english_agreement=per_language_english_agreement,
        cross_lingual_consistency=consistency,
        fully_consistent_concepts=fully_consistent,
        non_english_top1_floor=floor,
        english_baseline_intact=english_baseline_intact,
        synthetic_provenance=synthetic_provenance,
        restricted_markers=restricted_markers,
        passed=passed,
    )


def cross_lingual_grounding_metadata(
    *, path: str | Path | None = None
) -> dict[str, Any]:
    """Return static report metadata for the cross-lingual grounding suite."""

    fixture_path = Path(path) if path is not None else _FIXTURE
    return {
        "suite": CROSS_LINGUAL_GROUNDING,
        "languages": list(CROSS_LINGUAL_LANGUAGES),
        "non_english_languages": list(NON_ENGLISH_LANGUAGES),
        "non_english_top1_floor": NON_ENGLISH_TOP1_FLOOR,
        "fixture": fixture_path.name,
        "provenance": SYNTHETIC_PROVENANCE,
    }


def _rate(numerator: int, denominator: int) -> float:
    """Return ``numerator / denominator`` as a float, guarding empty slices."""

    if denominator <= 0:
        return 0.0
    return numerator / denominator


__all__ = [
    "CROSS_LINGUAL_GROUNDING",
    "CROSS_LINGUAL_LANGUAGES",
    "ENGLISH_LANGUAGE",
    "NON_ENGLISH_LANGUAGES",
    "NON_ENGLISH_TOP1_FLOOR",
    "RESTRICTED_CORPUS_MARKERS",
    "SYNTHETIC_PROVENANCE",
    "CrossLingualGroundingCase",
    "CrossLingualGroundingReport",
    "Grounder",
    "build_fixture_grounder",
    "cross_lingual_grounding_metadata",
    "load_cross_lingual_grounding_fixtures",
    "run_cross_lingual_grounding",
    "scan_restricted_corpus_markers",
]
