"""Grounding accuracy evaluation for synthetic RxNorm/LOINC/ICD-10-CM gold.

This suite measures whether the sparse candidate generator maps a clinical
mention to the correct coded concept. For each permissive vocabulary system it
computes top-1 and top-5 accuracy per language over the ranked candidate output,
plus a not-groundable abstention rate, and emits the standard benchmark report
schema so a wrong crosswalk or ranking regression is quantifiable and gateable.

All gold data is synthetic and algorithmically generated: invented concept
codes, preferred terms, and alias surfaces with no UMLS/SNOMED/real terminology
content, so the corpus lives in the repository under a permissive licence. The
suite is fully offline and deterministic; it builds an in-memory alias index
from the gold vocabulary and scores the real
:class:`~openmed.clinical.grounding.candidate_generator.SparseCandidateGenerator`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from openmed.clinical.grounding.candidate_generator import SparseCandidateGenerator
from openmed.clinical.grounding.types import Candidate
from openmed.clinical.grounding.vocab import VocabConcept, VocabularyIndex
from openmed.eval.report import BenchmarkReport

#: Permissive vocabulary systems scored on English gold with strict floors.
PERMISSIVE_GROUNDING_SYSTEMS: tuple[str, ...] = ("rxnorm", "loinc", "icd10cm")
#: Languages evaluated; ``en`` is strict, ``zh``/``hi`` are provisional.
GROUNDING_ACCURACY_LANGUAGES: tuple[str, ...] = ("en", "zh", "hi")
#: Default ranked-candidate depth scored for top-5 accuracy.
DEFAULT_GROUNDING_TOP_K: int = 5
#: Committed synthetic gold directory (one JSONL per permissive system).
DEFAULT_GROUNDING_GOLD_DIR: Path = (
    Path(__file__).resolve().parent / "golden" / "grounding"
)
#: Substrings that would indicate restricted-vocabulary derived content.
RESTRICTED_VOCAB_MARKERS: tuple[str, ...] = (
    "umls",
    "snomed",
    "mrconso",
    "mrrel",
    "mrsty",
    "sct2",
)

#: A provider maps ``(mention, system, language, k)`` to ranked candidates.
GroundingCandidateProvider = Callable[[str, str, str, int], Sequence[Candidate]]


@dataclass(frozen=True)
class GroundingMention:
    """One evaluation surface form for a gold concept.

    ``groundable`` mentions carry an ``expected_code`` and contribute to the
    accuracy denominator; not-groundable mentions have no expected code and are
    scored for correct abstention (the generator should return no code).
    """

    text: str
    language: str
    expected_code: str | None
    groundable: bool
    match_kind: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingMention":
        """Build and validate a mention from a JSON-ready mapping."""

        text = str(payload.get("text", ""))
        if not text:
            raise ValueError("grounding mention text is required")
        groundable = bool(payload.get("groundable", True))
        expected = payload.get("expected_code")
        expected_code = None if expected is None else str(expected)
        if groundable and not expected_code:
            raise ValueError("groundable mention requires an expected_code")
        if not groundable and expected_code:
            raise ValueError("not-groundable mention must omit expected_code")
        return cls(
            text=text,
            language=str(payload.get("language") or "en"),
            expected_code=expected_code,
            groundable=groundable,
            match_kind=str(payload.get("match_kind") or ""),
        )


@dataclass(frozen=True)
class GroundingConcept:
    """One synthetic coded concept with alias surfaces and eval mentions."""

    system: str
    code: str
    preferred_term: str
    synonyms: tuple[str, ...]
    language_aliases: Mapping[str, tuple[str, ...]]
    mentions: tuple[GroundingMention, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "GroundingConcept":
        """Build and validate a concept, enforcing the synthetic marker."""

        metadata = dict(payload.get("metadata") or {})
        if metadata.get("synthetic") is not True:
            raise ValueError("grounding gold concepts must be marked synthetic")
        code = str(payload.get("code") or "")
        preferred = str(payload.get("preferred_term") or "")
        if not code or not preferred:
            raise ValueError("grounding concept requires code and preferred_term")
        language_aliases = {
            str(language): tuple(str(alias) for alias in aliases)
            for language, aliases in (payload.get("language_aliases") or {}).items()
        }
        mentions = tuple(
            GroundingMention.from_mapping(mention)
            for mention in payload.get("mentions") or ()
        )
        if not mentions:
            raise ValueError("grounding concept requires at least one mention")
        return cls(
            system=str(payload.get("system") or ""),
            code=code,
            preferred_term=preferred,
            synonyms=tuple(str(item) for item in payload.get("synonyms") or ()),
            language_aliases=language_aliases,
            mentions=mentions,
            metadata=metadata,
        )

    def vocab_concept(self, language: str) -> VocabConcept | None:
        """Return a language-specific :class:`VocabConcept`, or ``None``.

        For ``en`` the preferred term and synonyms are used; for other languages
        the concept's localized aliases become the index surfaces so the
        language-agnostic generator can match them.
        """

        if language == "en":
            return VocabConcept(
                system=self.system,
                code=self.code,
                preferred_term=self.preferred_term,
                synonyms=self.synonyms,
            )
        aliases = self.language_aliases.get(language, ())
        if not aliases:
            return None
        return VocabConcept(
            system=self.system,
            code=self.code,
            preferred_term=aliases[0],
            synonyms=tuple(aliases[1:]),
        )


@dataclass(frozen=True)
class GroundingGold:
    """The synthetic gold set for one vocabulary system."""

    system: str
    concepts: tuple[GroundingConcept, ...]

    @property
    def mentions(self) -> tuple[GroundingMention, ...]:
        """Return every mention across concepts in source order."""

        return tuple(
            mention for concept in self.concepts for mention in concept.mentions
        )

    @property
    def groundable_pair_count(self) -> int:
        """Return the number of groundable mention -> code pairs."""

        return sum(1 for mention in self.mentions if mention.groundable)

    def vocabulary_index(self, language: str) -> VocabularyIndex:
        """Build the in-memory alias index for one language."""

        concepts = [
            vocab
            for concept in self.concepts
            if (vocab := concept.vocab_concept(language)) is not None
        ]
        return VocabularyIndex(self.system, concepts)


@dataclass(frozen=True)
class LanguageGroundingAccuracy:
    """Grounding accuracy for one system and language.

    ``abstention_rate`` is the *wrongful*-abstention rate over groundable
    mentions (nothing returned for a mention that has a gold code).
    ``correct_abstention_rate`` is the *correct*-abstention rate over
    not-groundable mentions (no code returned for an unmappable surface). The
    abstention metrics are reported, not gated.
    """

    language: str
    support: int
    top1_accuracy: float
    top5_accuracy: float
    abstention_rate: float
    not_groundable_support: int = 0
    correct_abstention_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation."""

        return {
            "support": self.support,
            "top1_accuracy": self.top1_accuracy,
            "top5_accuracy": self.top5_accuracy,
            "abstention_rate": self.abstention_rate,
            "not_groundable_support": self.not_groundable_support,
            "correct_abstention_rate": self.correct_abstention_rate,
        }


@dataclass(frozen=True)
class SystemGroundingAccuracy:
    """Per-language grounding accuracy for one vocabulary system."""

    system: str
    languages: Mapping[str, LanguageGroundingAccuracy]

    def language(self, language: str) -> LanguageGroundingAccuracy | None:
        """Return the accuracy for one language, if evaluated."""

        return self.languages.get(language)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation sorted by language."""

        return {
            language: self.languages[language].to_dict()
            for language in sorted(self.languages)
        }


@dataclass(frozen=True)
class GroundingAccuracyReport:
    """Grounding accuracy across every scored system and language."""

    systems: Mapping[str, SystemGroundingAccuracy]
    top_k: int = DEFAULT_GROUNDING_TOP_K

    def system(self, system: str) -> SystemGroundingAccuracy | None:
        """Return the accuracy for one system, if evaluated."""

        return self.systems.get(system)

    @property
    def total_support(self) -> int:
        """Return the total number of scored groundable mentions."""

        return sum(
            language.support
            for system in self.systems.values()
            for language in system.languages.values()
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-friendly representation."""

        return {
            "top_k": self.top_k,
            "permissive_systems": list(PERMISSIVE_GROUNDING_SYSTEMS),
            "languages": list(GROUNDING_ACCURACY_LANGUAGES),
            "systems": {
                system: self.systems[system].to_dict()
                for system in sorted(self.systems)
            },
        }

    def to_benchmark_report(
        self,
        *,
        model_name: str = "sparse-candidate-generator",
        device: str = "cpu",
        generated_at: str | None = None,
    ) -> BenchmarkReport:
        """Emit the standard benchmark report schema for this result."""

        return BenchmarkReport(
            suite="grounding_accuracy",
            model_name=model_name,
            device=device,
            fixture_count=self.total_support,
            generated_at=generated_at,
            metrics=self.to_dict(),
            metadata={
                "synthetic": True,
                "artifact_type": "openmed.eval.grounding_accuracy",
            },
        )


def load_grounding_gold(
    path: str | Path = DEFAULT_GROUNDING_GOLD_DIR,
) -> dict[str, GroundingGold]:
    """Load the synthetic grounding gold, one :class:`GroundingGold` per system."""

    directory = Path(path)
    gold: dict[str, GroundingGold] = {}
    for system in PERMISSIVE_GROUNDING_SYSTEMS:
        system_path = directory / f"{system}.jsonl"
        concepts: list[GroundingConcept] = []
        with system_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    concepts.append(GroundingConcept.from_mapping(json.loads(line)))
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"invalid grounding gold {system_path}:{line_number}: {exc}"
                    ) from exc
        gold[system] = GroundingGold(system=system, concepts=tuple(concepts))
    return gold


def default_grounding_provider(
    gold: Mapping[str, GroundingGold],
) -> GroundingCandidateProvider:
    """Return a provider backed by per-language sparse candidate generators."""

    generators: dict[tuple[str, str], SparseCandidateGenerator] = {}

    def _generator(system: str, language: str) -> SparseCandidateGenerator:
        cached = generators.get((system, language))
        if cached is None:
            index = gold[system].vocabulary_index(language)
            cached = SparseCandidateGenerator(_SingleIndexLoader(index))
            generators[(system, language)] = cached
        return cached

    def provider(
        mention: str, system: str, language: str, k: int
    ) -> Sequence[Candidate]:
        return _generator(system, language).generate(mention, [system], k)

    return provider


class _SingleIndexLoader:
    """Minimal loader exposing one prebuilt alias index to the generator."""

    def __init__(self, index: VocabularyIndex) -> None:
        self._index = index

    def get_index(self, system: str) -> VocabularyIndex:  # noqa: ARG002
        """Return the single prebuilt index regardless of the system key."""

        return self._index


def evaluate_grounding_accuracy(
    gold: Mapping[str, GroundingGold] | None = None,
    *,
    gold_dir: str | Path = DEFAULT_GROUNDING_GOLD_DIR,
    provider: GroundingCandidateProvider | None = None,
    top_k: int = DEFAULT_GROUNDING_TOP_K,
) -> GroundingAccuracyReport:
    """Score grounding accuracy per system and language over the gold set."""

    resolved_gold = dict(gold) if gold is not None else load_grounding_gold(gold_dir)
    resolved_provider = (
        provider if provider is not None else default_grounding_provider(resolved_gold)
    )

    systems: dict[str, SystemGroundingAccuracy] = {}
    for system, gold_set in resolved_gold.items():
        languages: dict[str, LanguageGroundingAccuracy] = {}
        for language in GROUNDING_ACCURACY_LANGUAGES:
            accuracy = _score_language(gold_set, language, resolved_provider, top_k)
            if accuracy is not None:
                languages[language] = accuracy
        systems[system] = SystemGroundingAccuracy(system=system, languages=languages)
    return GroundingAccuracyReport(systems=systems, top_k=top_k)


def _score_language(
    gold: GroundingGold,
    language: str,
    provider: GroundingCandidateProvider,
    top_k: int,
) -> LanguageGroundingAccuracy | None:
    mentions = [
        mention
        for mention in gold.mentions
        if mention.language == language and mention.groundable
    ]
    if not mentions:
        return None

    top1 = top5 = abstained = 0
    for mention in mentions:
        candidates = provider(mention.text, gold.system, language, top_k)
        codes = [candidate.code for candidate in candidates]
        if not codes:
            abstained += 1
        if codes[:1] == [mention.expected_code]:
            top1 += 1
        if mention.expected_code in codes[:top_k]:
            top5 += 1

    # Score not-groundable mentions for CORRECT abstention: an unmappable surface
    # should yield no code. Reported, not gated.
    not_groundable = [
        mention
        for mention in gold.mentions
        if mention.language == language and not mention.groundable
    ]
    correct_abstentions = sum(
        1
        for mention in not_groundable
        if not provider(mention.text, gold.system, language, top_k)
    )
    not_groundable_support = len(not_groundable)

    support = len(mentions)
    return LanguageGroundingAccuracy(
        language=language,
        support=support,
        top1_accuracy=top1 / support,
        top5_accuracy=top5 / support,
        abstention_rate=abstained / support,
        not_groundable_support=not_groundable_support,
        correct_abstention_rate=(
            correct_abstentions / not_groundable_support
            if not_groundable_support
            else 0.0
        ),
    )


def restricted_vocab_markers_in(gold: Mapping[str, GroundingGold]) -> list[str]:
    """Return restricted-vocabulary markers found anywhere in the gold text."""

    found: set[str] = set()
    for gold_set in gold.values():
        for concept in gold_set.concepts:
            surfaces: Iterable[str] = (
                concept.code,
                concept.preferred_term,
                *concept.synonyms,
                *(
                    alias
                    for aliases in concept.language_aliases.values()
                    for alias in aliases
                ),
                *(mention.text for mention in concept.mentions),
            )
            haystack = " ".join(surfaces).lower()
            found.update(
                marker for marker in RESTRICTED_VOCAB_MARKERS if marker in haystack
            )
    return sorted(found)


def format_grounding_accuracy_table(report: GroundingAccuracyReport) -> str:
    """Render a deterministic per-system/per-language accuracy table."""

    header = (
        f"{'system':<10}  {'lang':<4}  {'support':>7}  "
        f"{'top1':>7}  {'top5':>7}  {'abstain':>7}"
    )
    lines = ["Grounding accuracy (synthetic gold)", "", header]
    for system in sorted(report.systems):
        accuracy = report.systems[system]
        for language in sorted(accuracy.languages):
            metrics = accuracy.languages[language]
            lines.append(
                f"{system:<10}  {language:<4}  {metrics.support:>7}  "
                f"{metrics.top1_accuracy:>7.4f}  {metrics.top5_accuracy:>7.4f}  "
                f"{metrics.abstention_rate:>7.4f}"
            )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_GROUNDING_GOLD_DIR",
    "DEFAULT_GROUNDING_TOP_K",
    "GROUNDING_ACCURACY_LANGUAGES",
    "PERMISSIVE_GROUNDING_SYSTEMS",
    "RESTRICTED_VOCAB_MARKERS",
    "GroundingAccuracyReport",
    "GroundingCandidateProvider",
    "GroundingConcept",
    "GroundingGold",
    "GroundingMention",
    "LanguageGroundingAccuracy",
    "SystemGroundingAccuracy",
    "default_grounding_provider",
    "evaluate_grounding_accuracy",
    "format_grounding_accuracy_table",
    "load_grounding_gold",
    "restricted_vocab_markers_in",
]
