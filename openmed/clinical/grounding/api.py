"""Canonical facade for deterministic clinical concept grounding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from openmed.core.labels import normalize_label

from ..context import ClinicalAssertion, RerankContext
from .embeddings import AliasEncoder
from .matcher import LexicalMatcher
from .ranker import CandidateRankingStage, RankingConfig
from .restricted import UserKeyVocabularyLoader
from .types import Candidate, GroundedSpan
from .vocab import (
    FREE_VOCAB_SYSTEMS,
    RestrictedVocabularyError,
    VocabLoader,
    normalize_language,
)

__all__ = ["DEFAULT_GROUNDING_SYSTEMS", "ground"]

DEFAULT_GROUNDING_SYSTEMS: tuple[str, ...] = (
    "rxnorm",
    "icd10cm",
    "loinc",
    "hpo",
)

_FREE_ALIASES = {
    "rxnorm": "rxnorm",
    "rx-norm": "rxnorm",
    "icd10": "icd10cm",
    "icd10cm": "icd10cm",
    "icd-10-cm": "icd10cm",
    "loinc": "loinc",
    "hpo": "hpo",
    "hp": "hpo",
    "mesh": "mesh",
}
_RESTRICTED_ALIASES = {
    "umls": "umls",
    "snomed": "snomed",
    "snomed-ct": "snomed",
    "snomedct": "snomed",
    "sct": "snomed",
}
_TEXT_FIELDS = ("text", "span_text", "entity_text", "surface", "word")
_START_FIELDS = ("start", "start_char", "offset")
_END_FIELDS = ("end", "end_char", "offset_end")
_LABEL_FIELDS = ("canonical_label", "label", "entity_label", "entity_type")


def ground(
    spans: Iterable[Any],
    systems: Sequence[str] = DEFAULT_GROUNDING_SYSTEMS,
    *,
    loader: VocabLoader | None = None,
    encoder: AliasEncoder | None = None,
    config: RankingConfig | None = None,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None = None,
    source_language: str | None = None,
) -> list[GroundedSpan]:
    """Ground clinical spans to one selected concept per requested system.

    The function composes the existing sparse/dense retrieval and reranking
    stage for freely redistributable vocabularies. UMLS and SNOMED CT are never
    bundled or downloaded: requesting either requires a matching explicit
    :class:`UserKeyVocabularyLoader` over a caller-owned local alias table.

    Span inputs may be :class:`GroundedSpan` objects, strings, mappings, or
    objects exposing ``text``/``start``/``end``. ``canonical_label``,
    ``assertion``, ``section``, and ``source_language`` fields are consumed when
    present. Results retain the source offsets and assertion context for
    deterministic FHIR and OMOP export.

    Args:
        spans: Clinical span records to ground.
        systems: Ordered vocabulary systems. The default is RxNorm, ICD-10-CM,
            LOINC, and HPO.
        loader: Optional free-vocabulary loader, typically configured with
            local/cache sources for offline execution.
        encoder: Optional local dense encoder. No encoder download occurs.
        config: Optional ranking configuration.
        restricted_loaders: Explicit user-key-gated local UMLS/SNOMED loaders.
        source_language: Default source language when a span omits one.

    Returns:
        One :class:`GroundedSpan` per input, including abstentions.

    Raises:
        ValueError: If no systems are requested or a span is malformed.
        RestrictedVocabularyError: If a restricted/proprietary system is
            requested without an explicit gated loader.
    """

    ordered_systems = _normalize_systems(systems)
    free_systems = tuple(
        system for system in ordered_systems if system in _FREE_ALIASES
    )
    restricted_systems = tuple(
        system for system in ordered_systems if system in _RESTRICTED_ALIASES
    )
    gated = _prepare_restricted_matchers(restricted_systems, restricted_loaders)
    stage = (
        CandidateRankingStage(loader, encoder=encoder, config=config)
        if free_systems
        else None
    )

    results: list[GroundedSpan] = []
    for index, raw_span in enumerate(spans):
        span = _coerce_span(raw_span, index=index, default_language=source_language)
        candidates: list[Candidate] = []
        if stage is not None:
            ranked = stage.rank(
                span.text,
                free_systems,
                context=_rerank_context(raw_span, span.assertion),
                source_language=span.source_language,
            )
            candidates.extend(_select_one_per_system(item.candidate for item in ranked))
        for system in restricted_systems:
            matcher, gated_loader = gated[system]
            matches = matcher.lookup(span.text, limit=1)
            if not matches:
                continue
            match = matches[0]
            candidates.append(
                Candidate(
                    system=system.upper(),
                    code=match.code,
                    display=match.display,
                    score=match.score,
                    source_language=span.source_language,
                    source="sparse",
                    matched_alias=match.matched_term,
                    match_kind=match.match_type,
                    vocab_version=gated_loader.content_hash,
                )
            )

        candidates = _ordered_candidates(candidates, ordered_systems)
        results.append(
            GroundedSpan(
                text=span.text,
                start=span.start,
                end=span.end,
                candidates=tuple(candidates),
                canonical_label=span.canonical_label,
                assertion=span.assertion,
                source_language=span.source_language,
                metadata=span.metadata,
            )
        )
    return results


def _normalize_systems(systems: Sequence[str]) -> tuple[str, ...]:
    if isinstance(systems, (str, bytes)):
        raise TypeError("systems must be a sequence of vocabulary names")
    normalized: list[str] = []
    for raw_system in systems:
        if not isinstance(raw_system, str):
            raise TypeError("grounding system names must be strings")
        key = raw_system.strip().casefold().replace("_", "-")
        if key in {"cpt", "cpt4", "cpt-4"}:
            raise RestrictedVocabularyError(
                "CPT is proprietary and remains caller-supplied and out of process."
            )
        system = _FREE_ALIASES.get(key) or _RESTRICTED_ALIASES.get(key)
        if system is None:
            allowed = sorted({*FREE_VOCAB_SYSTEMS, "umls", "snomed"})
            raise ValueError(
                f"unsupported grounding system {raw_system!r}; expected {allowed}"
            )
        if system not in normalized:
            normalized.append(system)
    if not normalized:
        raise ValueError("systems must contain at least one vocabulary")
    return tuple(normalized)


def _prepare_restricted_matchers(
    systems: Sequence[str],
    loaders: Mapping[str, UserKeyVocabularyLoader] | None,
) -> dict[str, tuple[LexicalMatcher, UserKeyVocabularyLoader]]:
    if not systems:
        return {}
    normalized_loaders = {
        _RESTRICTED_ALIASES.get(key.strip().casefold().replace("_", "-"), key): value
        for key, value in (loaders or {}).items()
    }
    result: dict[str, tuple[LexicalMatcher, UserKeyVocabularyLoader]] = {}
    for system in systems:
        gated_loader = normalized_loaders.get(system)
        if gated_loader is None or gated_loader.system != system:
            raise RestrictedVocabularyError(
                f"{system.upper()} grounding requires an explicit matching "
                "UserKeyVocabularyLoader; restricted content is never bundled "
                "or downloaded."
            )
        result[system] = (
            LexicalMatcher(
                gated_loader.load(),
                system_uri=gated_loader.system_uri,
            ),
            gated_loader,
        )
    return result


def _coerce_span(
    raw_span: Any,
    *,
    index: int,
    default_language: str | None,
) -> GroundedSpan:
    if isinstance(raw_span, GroundedSpan):
        return raw_span
    if isinstance(raw_span, str):
        return GroundedSpan(
            text=raw_span,
            start=0,
            end=len(raw_span),
            source_language=normalize_language(default_language),
        )

    text = _first_value(raw_span, _TEXT_FIELDS)
    if not isinstance(text, str):
        raise ValueError(f"span at index {index} is missing text")
    start = _first_value(raw_span, _START_FIELDS)
    end = _first_value(raw_span, _END_FIELDS)
    if start is None:
        start = 0
    if end is None:
        end = start + len(text)
    label = _first_value(raw_span, _LABEL_FIELDS)
    assertion = _coerce_assertion(_first_value(raw_span, ("assertion", "context")))
    language = _first_value(raw_span, ("source_language", "language", "lang"))
    metadata = _first_value(raw_span, ("metadata", "meta")) or {}
    canonical_label = None
    if label is not None and str(label).strip():
        canonical_label = normalize_label(
            str(label),
            normalize_language(language or default_language),
        )
    return GroundedSpan(
        text=text,
        start=start,
        end=end,
        canonical_label=canonical_label,
        assertion=assertion,
        source_language=normalize_language(language or default_language),
        metadata=metadata,
    )


def _coerce_assertion(value: Any) -> ClinicalAssertion | None:
    if value is None:
        return None
    if isinstance(value, ClinicalAssertion):
        return value
    if isinstance(value, Mapping):
        temporality = value.get("temporality")
        certainty = value.get("certainty")
        if temporality is None or certainty is None:
            return None
        return ClinicalAssertion(
            temporality=str(temporality),
            certainty=str(certainty),  # type: ignore[arg-type]
            negation=value.get("negation"),
            experiencer=value.get("experiencer"),
        )
    return None


def _rerank_context(
    raw_span: Any, assertion: ClinicalAssertion | None
) -> RerankContext:
    section = _first_value(raw_span, ("section", "section_label"))
    preferred = _first_value(raw_span, ("preferred_concepts",)) or ()
    return RerankContext(
        section=str(section) if section is not None else None,
        assertion=assertion,
        preferred_concepts=frozenset(tuple(item) for item in preferred),
    )


def _select_one_per_system(candidates: Iterable[Candidate]) -> list[Candidate]:
    selected: list[Candidate] = []
    seen: set[str] = set()
    for candidate in candidates:
        system = candidate.system.casefold()
        if system in seen:
            continue
        selected.append(candidate)
        seen.add(system)
    return selected


def _ordered_candidates(
    candidates: Sequence[Candidate], systems: Sequence[str]
) -> list[Candidate]:
    priority = {system: index for index, system in enumerate(systems)}
    return sorted(
        _select_one_per_system(candidates),
        key=lambda candidate: (
            -candidate.score,
            priority.get(candidate.system.casefold(), len(priority)),
            candidate.code,
        ),
    )


def _first_value(source: Any, fields: Sequence[str]) -> Any:
    for field in fields:
        if isinstance(source, Mapping) and field in source:
            return source[field]
        value = getattr(source, field, None)
        if value is not None:
            return value
    return None
