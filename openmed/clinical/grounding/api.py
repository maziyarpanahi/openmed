"""Canonical facade for deterministic clinical concept grounding."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from openmed.core.labels import normalize_label
from openmed.core.offline import network_blocked_if_offline

from ..context import ClinicalAssertion, ClinicalContextResult, RerankContext
from .decompose import decompose_and_relink
from .embeddings import AliasEncoder
from .matcher import ConceptMatch, LexicalMatcher
from .postcoordination import PostCoordinationStage
from .ranker import CandidateRankingStage, RankingConfig
from .restricted import UserKeyVocabularyLoader
from .systems import RESTRICTED_SYSTEMS, SYSTEM_URIS, canonical_system, system_uri
from .types import Candidate, GroundedSpan
from .vocab import (
    FREE_VOCAB_SYSTEMS,
    RestrictedVocabularyError,
    VocabLoader,
    normalize_language,
)

__all__ = ["DEFAULT_GROUNDING_SYSTEMS", "ground", "ground_payload"]

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
    spans: Iterable[Any] | Mapping[str, Any] | GroundedSpan | str,
    systems: Sequence[str] = DEFAULT_GROUNDING_SYSTEMS,
    *,
    loader: VocabLoader | None = None,
    encoder: AliasEncoder | None = None,
    config: RankingConfig | None = None,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None = None,
    restricted_endpoint: Any = None,
    source_language: str | None = None,
    offline: bool = False,
    local_only: bool | None = None,
    normalize_composites: bool = False,
    composite_atomic_terms: Iterable[str] | None = None,
    postcoordination: PostCoordinationStage | None = None,
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
        restricted_endpoint: Optional caller-configured out-of-process endpoint
            for restricted terminology lookups.
        source_language: Default source language when a span omits one.
        offline: Whether to block network access during grounding.
        local_only: Compatibility alias for ``offline`` when provided.
        normalize_composites: Opt in to rules-first composite decomposition and
            child re-linking before emission. Exact whole-span concepts remain
            single pre-coordinated results; uncodable proposals are retained as
            post-coordination abstentions.
        composite_atomic_terms: Additional atomic multi-word concepts that the
            opt-in normalizer must never split.
        postcoordination: Optional user-key-gated SNOMED expression stage. It is
            consulted only after lookup abstains or scores below the stage's
            pre-coordination threshold.

    Returns:
        Grounded spans, including abstentions. The default returns one per input;
        the opt-in composite stage may emit one span per linked child.

    Raises:
        ValueError: If no systems are requested or a span is malformed.
        RestrictedVocabularyError: If a restricted/proprietary system is
            requested without an explicit gated loader.
    """

    if local_only is not None:
        offline = bool(local_only)
    raw_spans = _coerce_span_inputs(spans)
    ordered_systems = _normalize_systems(systems)
    selected_loader = loader
    if selected_loader is None and any(
        canonical_system(system) not in RESTRICTED_SYSTEMS for system in ordered_systems
    ):
        selected_loader = VocabLoader(local_only=offline)
    if offline and isinstance(selected_loader, VocabLoader):
        selected_loader.local_only = True

    with network_blocked_if_offline(local_only=offline):
        return _ground_spans(
            raw_spans,
            ordered_systems,
            loader=selected_loader,
            encoder=encoder,
            config=config,
            restricted_loaders=restricted_loaders,
            restricted_endpoint=restricted_endpoint,
            source_language=source_language,
            offline=offline,
            normalize_composites=normalize_composites,
            composite_atomic_terms=composite_atomic_terms,
            postcoordination=postcoordination,
        )


def ground_payload(
    spans: Iterable[Any] | Mapping[str, Any] | GroundedSpan | str,
    systems: Sequence[str] = DEFAULT_GROUNDING_SYSTEMS,
    *,
    loader: VocabLoader | None = None,
    encoder: AliasEncoder | None = None,
    config: RankingConfig | None = None,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None = None,
    restricted_endpoint: Any = None,
    source_language: str | None = None,
    offline: bool = True,
    local_only: bool | None = None,
) -> dict[str, Any]:
    """Return the shared REST/CLI grounding response contract."""

    if local_only is not None:
        offline = bool(local_only)
    raw_spans = _coerce_span_inputs(spans)
    selected_loader = loader
    ordered_systems = _normalize_systems(systems)
    free_systems = tuple(
        system for system in ordered_systems if system in _FREE_ALIASES
    )
    if selected_loader is None and free_systems:
        selected_loader = VocabLoader(local_only=offline)
    results = ground(
        raw_spans,
        ordered_systems,
        loader=selected_loader,
        encoder=encoder,
        config=config,
        restricted_loaders=restricted_loaders,
        restricted_endpoint=restricted_endpoint,
        source_language=source_language,
        offline=offline,
    )
    snapshots = _snapshot_provenance(
        selected_loader,
        free_systems,
        restricted_loaders=restricted_loaders,
    )
    return {
        "schema_version": "openmed.grounding.v1",
        "offline": bool(offline),
        "systems": list(ordered_systems),
        "snapshots": snapshots,
        "results": [result.to_dict() for result in results],
    }


def _ground_spans(
    spans: Sequence[Any],
    systems: Sequence[str],
    *,
    loader: VocabLoader | None,
    encoder: AliasEncoder | None,
    config: RankingConfig | None,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None,
    restricted_endpoint: Any,
    source_language: str | None,
    offline: bool,
    normalize_composites: bool,
    composite_atomic_terms: Iterable[str] | None,
    postcoordination: PostCoordinationStage | None,
) -> list[GroundedSpan]:
    ordered_systems = _normalize_systems(systems)
    if not isinstance(normalize_composites, bool):
        raise TypeError("normalize_composites must be a boolean")
    if postcoordination is not None and not isinstance(
        postcoordination, PostCoordinationStage
    ):
        raise TypeError("postcoordination must be a PostCoordinationStage")
    if isinstance(composite_atomic_terms, (str, bytes)):
        raise TypeError("composite_atomic_terms must be an iterable of terms")
    atomic_terms = (
        None if composite_atomic_terms is None else tuple(composite_atomic_terms)
    )
    free_systems = tuple(
        system for system in ordered_systems if system in _FREE_ALIASES
    )
    restricted_systems = tuple(
        system for system in ordered_systems if system in _RESTRICTED_ALIASES
    )
    gated = _prepare_restricted_matchers(
        restricted_systems,
        restricted_loaders,
        restricted_endpoint=restricted_endpoint,
    )
    stage = (
        CandidateRankingStage(loader, encoder=encoder, config=config)
        if free_systems
        else None
    )
    snapshots = _snapshot_provenance(
        loader,
        free_systems,
        restricted_loaders=restricted_loaders,
    )

    results: list[GroundedSpan] = []
    for index, raw_span in enumerate(spans):
        span = _coerce_span(raw_span, index=index, default_language=source_language)
        rerank_context = _rerank_context(raw_span, span.assertion)

        def rank_surface(surface: str) -> tuple[list[Candidate], list[Candidate]]:
            candidates: list[Candidate] = []
            alternatives: list[Candidate] = []
            if stage is not None:
                ranked = stage.rank(
                    surface,
                    free_systems,
                    context=rerank_context,
                    source_language=span.source_language,
                )
                ranked_candidates = [item.candidate for item in ranked]
                candidates.extend(_select_one_per_system(ranked_candidates))
                selected_keys = {(item.system, item.code) for item in candidates}
                alternatives.extend(
                    item
                    for item in ranked_candidates
                    if (item.system, item.code) not in selected_keys
                )
            for system in restricted_systems:
                matcher, gated_loader = gated[system]
                matches = matcher.lookup(surface, limit=1)
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
                        source="endpoint" if restricted_endpoint else "sparse",
                        matched_alias=match.matched_term,
                        match_kind=match.match_type,
                        vocab_version=_restricted_version(gated_loader),
                    )
                )
            return _ordered_candidates(candidates, ordered_systems), alternatives

        def link_surface(surface: str) -> list[Candidate]:
            return rank_surface(surface)[0]

        if normalize_composites:
            byte_start = _first_value(raw_span, ("byte_start", "start_byte"))
            if byte_start is None:
                byte_start = span.metadata.get("byte_start", span.start)
            decomposition = decompose_and_relink(
                span.text,
                linker=link_surface,
                start=span.start,
                byte_start=byte_start,
                atomic_terms=atomic_terms,
                canonical_label=span.canonical_label,
                assertion=span.assertion,
                source_language=span.source_language,
                metadata=span.metadata,
            )
            emitted = tuple(
                replace(
                    item,
                    section=span.section,
                    provenance={
                        **item.provenance,
                        "offline": bool(offline),
                        "snapshot_provenance": snapshots,
                    },
                )
                for item in decomposition.spans
            )
            if postcoordination is not None:
                emitted = tuple(postcoordination.apply(item) for item in emitted)
            results.extend(emitted)
            continue

        candidates, alternatives = rank_surface(span.text)
        grounded_span = GroundedSpan(
            text=span.text,
            start=span.start,
            end=span.end,
            candidates=tuple(candidates),
            alternatives=tuple(alternatives),
            canonical_label=span.canonical_label,
            assertion=span.assertion,
            source_language=span.source_language,
            metadata=span.metadata,
            section=span.section,
            provenance={
                "offline": bool(offline),
                "snapshot_provenance": snapshots,
            },
        )
        if postcoordination is not None:
            grounded_span = postcoordination.apply(grounded_span)
        results.append(grounded_span)
    return results


def _normalize_systems(systems: Sequence[str]) -> tuple[str, ...]:
    if isinstance(systems, (str, bytes)):
        raise TypeError("systems must be a sequence of vocabulary names")
    normalized: list[str] = []
    for raw_system in systems:
        if not isinstance(raw_system, str):
            raise TypeError("grounding system names must be strings")
        system = canonical_system(raw_system)
        if system == "cpt":
            raise RestrictedVocabularyError(
                f"{system.upper()} is proprietary and remains caller-supplied and "
                "out of process."
            )
        if system not in FREE_VOCAB_SYSTEMS and system not in {
            "umls",
            "snomed",
        }:
            allowed = sorted({*FREE_VOCAB_SYSTEMS, "umls", "snomed"})
            raise ValueError(
                f"unsupported grounding system {raw_system!r}; expected {allowed}"
            )
        if system not in normalized:
            normalized.append(system)
    if not normalized:
        raise ValueError("systems must contain at least one vocabulary")
    return tuple(normalized)


def _coerce_span_inputs(
    spans: Iterable[Any] | Mapping[str, Any] | GroundedSpan | str,
) -> list[Any]:
    """Normalize text, one entity, or an iterable of entity records."""

    if isinstance(spans, (str, Mapping, GroundedSpan)):
        return [spans]
    try:
        return list(spans)
    except TypeError as exc:
        raise TypeError(
            "spans must be text, one entity mapping, or an iterable of spans"
        ) from exc


def _prepare_restricted_matchers(
    systems: Sequence[str],
    loaders: Mapping[str, UserKeyVocabularyLoader] | None,
    *,
    restricted_endpoint: Any = None,
) -> dict[str, tuple[Any, Any]]:
    if not systems:
        return {}
    if restricted_endpoint is not None:
        return {
            system: (
                _endpoint_matcher(restricted_endpoint, system),
                restricted_endpoint,
            )
            for system in systems
        }
    normalized_loaders = {
        _RESTRICTED_ALIASES.get(key.strip().casefold().replace("_", "-"), key): value
        for key, value in (loaders or {}).items()
    }
    result: dict[str, tuple[LexicalMatcher, UserKeyVocabularyLoader]] = {}
    for system in systems:
        gated_loader = normalized_loaders.get(system)
        if gated_loader is None or gated_loader.system != system:
            raise RestrictedVocabularyError(
                f"{system.upper()} grounding requires an explicit matching, "
                "configured, user-supplied "
                "out-of-process terminology endpoint; restricted content is never "
                "bundled or downloaded."
            )
        result[system] = (
            LexicalMatcher(
                gated_loader.load(),
                system_uri=gated_loader.system_uri,
            ),
            gated_loader,
        )
    return result


class _EndpointMatcher:
    """Adapter for an explicitly supplied out-of-process terminology service."""

    def __init__(self, endpoint: Any, system: str) -> None:
        self.endpoint = endpoint
        self.system = system
        if not callable(getattr(endpoint, "lookup", None)):
            raise TypeError("restricted_endpoint must expose lookup(system, text)")

    def lookup(self, query: str, *, limit: int = 1) -> tuple[ConceptMatch, ...]:
        raw_matches = self.endpoint.lookup(self.system, query, limit=limit)
        if isinstance(raw_matches, Mapping):
            raw_matches = raw_matches.get("matches", ())
        if isinstance(raw_matches, (str, bytes)) or raw_matches is None:
            return ()
        matches: list[ConceptMatch] = []
        for raw in raw_matches:
            if not isinstance(raw, Mapping):
                continue
            code = raw.get("code") or raw.get("concept_id")
            display = raw.get("display") or raw.get("preferred_term")
            if not code or not display:
                continue
            score = raw.get("confidence", raw.get("score", 0.0))
            matches.append(
                ConceptMatch(
                    system_uri=SYSTEM_URIS[self.system],
                    code=str(code),
                    display=str(display),
                    score=float(score),
                    match_type="exact",
                    matched_term=str(raw.get("matched_term") or display),
                    metadata={"source": "user-supplied-out-of-process"},
                )
            )
        return tuple(matches[:limit])


def _endpoint_matcher(endpoint: Any, system: str) -> _EndpointMatcher:
    """Return an adapter without touching the endpoint until a query arrives."""

    return _EndpointMatcher(endpoint, system)


def _coerce_span(
    raw_span: Any,
    *,
    index: int,
    default_language: str | None,
) -> GroundedSpan:
    if isinstance(raw_span, GroundedSpan):
        if isinstance(raw_span.assertion, ClinicalContextResult):
            return replace(raw_span, assertion=raw_span.assertion.to_assertion())
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
    language = _first_value(raw_span, ("source_language", "language", "lang"))
    metadata = _first_value(raw_span, ("metadata", "meta")) or {}
    section = _first_value(raw_span, ("section", "section_label"))
    assertion_value = _first_value(raw_span, ("assertion", "context"))
    if assertion_value is None and isinstance(metadata, Mapping):
        assertion_value = metadata.get("clinical_context")
    if assertion_value is None and isinstance(raw_span, Mapping):
        if any(
            key in raw_span
            for key in ("temporality", "certainty", "uncertainty", "negation")
        ):
            assertion_value = raw_span
    assertion = _coerce_assertion(assertion_value)
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
        section=str(section).strip() if section is not None else None,
    )


def _coerce_assertion(value: Any) -> ClinicalAssertion | None:
    if value is None:
        return None
    if isinstance(value, ClinicalAssertion):
        return value
    if isinstance(value, ClinicalContextResult):
        return value.to_assertion()
    if isinstance(value, Mapping):
        temporality = value.get("temporality")
        certainty = value.get("certainty", value.get("uncertainty"))
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


def _snapshot_provenance(
    loader: VocabLoader | None,
    systems: Sequence[str],
    *,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None,
) -> dict[str, dict[str, str]]:
    """Collect stable snapshot metadata without retaining source surfaces."""

    if loader is None and not restricted_loaders:
        return {}
    result: dict[str, dict[str, str]] = {}
    if loader is not None:
        snapshot_method = getattr(loader, "snapshot_provenance", None)
        if callable(snapshot_method) and systems:
            result.update(snapshot_method(systems))
    for raw_system, restricted_loader in (restricted_loaders or {}).items():
        system = canonical_system(raw_system)
        if system not in result:
            result[system] = {
                "system": system,
                "system_uri": system_uri(system) or "",
                "version": restricted_loader.content_hash,
                "sha256": restricted_loader.content_hash,
                "content_hash": restricted_loader.content_hash,
                "artifact": "user-supplied-local",
            }
    return result


def _restricted_version(loader: Any) -> str:
    """Return a stable endpoint/local-loader version without reading secrets."""

    value = getattr(loader, "content_hash", None)
    if isinstance(value, str) and value:
        return value
    value = getattr(loader, "version", None)
    if isinstance(value, str) and value:
        return value
    return "user-supplied-endpoint"
