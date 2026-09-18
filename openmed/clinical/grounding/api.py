"""Canonical facade for deterministic clinical concept grounding."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from openmed.core.labels import normalize_label
from openmed.core.offline import network_blocked_if_offline

from ..context import ClinicalAssertion, ClinicalContextResult, RerankContext
from .decompose import decompose_and_relink
from .embeddings import AliasEncoder
from .matcher import ConceptMatch, LexicalConcept, LexicalMatcher
from .postcoordination import PostCoordinationStage
from .ranker import CandidateRankingStage, RankingConfig
from .restricted import UserKeyVocabularyLoader
from .results import GroundingConfigError, GroundingResult
from .systems import RESTRICTED_SYSTEMS, SYSTEM_URIS, canonical_system, system_uri
from .types import Candidate, GroundedSpan
from .vocab import (
    FREE_VOCAB_SYSTEMS,
    RestrictedVocabularyError,
    VocabConcept,
    VocabLoader,
    VocabSource,
    VocabularyIndex,
    normalize_language,
)

__all__ = [
    "DEFAULT_GROUNDING_SYSTEMS",
    "GroundingConfigError",
    "GroundingResult",
    "ground",
    "ground_payload",
]

DEFAULT_GROUNDING_SYSTEMS: tuple[str, ...] = (
    "rxnorm",
    "loinc",
    "icd10cm",
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
_TOKEN_PATTERN = re.compile(r"\S+")
_TRIM_SURFACE = " \t\r\n.,;:!?()[]{}"


class _SnapshotLoader:
    """Adapt one or more validated local snapshots to the loader protocol."""

    local_only = True

    def __init__(self, snapshots: Mapping[str, Any]) -> None:
        self._snapshots = {
            _snapshot_system_key(key): value for key, value in snapshots.items()
        }
        self._indexes: dict[str, VocabularyIndex] = {}

    def _snapshot_for(self, system: str) -> Any:
        normalized = canonical_system(system)
        try:
            return self._snapshots[normalized]
        except KeyError:
            if "*" in self._snapshots:
                return self._snapshots["*"]
            raise ValueError(
                f"snapshot does not contain a local vocabulary for {normalized!r}"
            ) from None

    def get_index(self, system: str) -> VocabularyIndex:
        """Return the validated index for one requested system."""

        normalized = canonical_system(system)
        cached = self._indexes.get(normalized)
        if cached is not None:
            return cached
        snapshot = self._snapshot_for(system)
        if isinstance(snapshot, VocabularyIndex):
            return snapshot
        index = getattr(snapshot, "index", None)
        if isinstance(index, VocabularyIndex):
            return index
        getter = getattr(snapshot, "get_index", None)
        if callable(getter):
            try:
                candidate = getter(system)
            except TypeError:
                candidate = getter()
            if isinstance(candidate, VocabularyIndex):
                return candidate
        loader = getattr(snapshot, "load", None)
        if callable(loader):
            try:
                candidate = loader(system)
            except TypeError:
                candidate = loader()
            if isinstance(candidate, VocabularyIndex):
                self._indexes[normalized] = candidate
                return candidate
            if isinstance(candidate, Mapping):
                index = _index_from_terms(normalized, candidate)
                self._indexes[normalized] = index
                return index
        raise TypeError(
            "snapshot values must expose a VocabularyIndex, get_index(), or load()"
        )

    def snapshot_provenance(self, systems: Sequence[str]) -> dict[str, dict[str, str]]:
        """Return PHI-free release metadata for the adapted snapshots."""

        result: dict[str, dict[str, str]] = {}
        for raw_system in systems:
            system = canonical_system(raw_system)
            snapshot = self._snapshot_for(system)
            index = self.get_index(system)
            version = (
                getattr(snapshot, "release_version", None)
                or getattr(snapshot, "version", None)
                or getattr(snapshot, "content_hash", None)
                or index.content_hash
            )
            result[system] = {
                "system": system,
                "system_uri": str(
                    getattr(snapshot, "system_uri", None) or system_uri(system) or ""
                ),
                "version": str(version),
                "sha256": str(
                    getattr(snapshot, "artifact_sha256", None)
                    or getattr(snapshot, "content_hash", None)
                    or index.content_hash
                ),
                "content_hash": index.content_hash,
                "artifact": str(
                    getattr(snapshot, "artifact", None) or "local-snapshot"
                ),
            }
        return result


def _snapshot_system_key(value: object) -> str:
    """Normalize a snapshot mapping key or canonical system URI."""

    key = canonical_system(value)
    for system, uri in SYSTEM_URIS.items():
        if key.rstrip("/") == uri.rstrip("/"):
            return system
    return key


def _index_from_terms(system: str, terms: Mapping[str, Any]) -> VocabularyIndex:
    """Adapt a specialized loader's term mapping to the shared alias index."""

    concepts: list[VocabConcept] = []
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for raw_alias, raw_values in terms.items():
        alias = str(raw_alias).strip()
        if not alias:
            continue
        if isinstance(raw_values, (LexicalConcept, str, Mapping)):
            values = (raw_values,)
        else:
            try:
                values = tuple(raw_values)
            except TypeError as exc:
                raise TypeError(
                    "specialized vocabulary loader terms must contain concept "
                    "objects, mappings, codes, or sequences of those values"
                ) from exc
        for raw_value in values:
            if isinstance(raw_value, LexicalConcept):
                code = raw_value.code
                display = raw_value.display
                synonyms = (alias,)
                language_aliases: Mapping[str, tuple[str, ...]] = {}
            elif isinstance(raw_value, Mapping):
                code = raw_value.get("code") or raw_value.get("id")
                display = raw_value.get("display") or raw_value.get("preferred_term")
                synonyms_value = raw_value.get("synonyms", raw_value.get("aliases", ()))
                synonyms = _coerce_aliases(synonyms_value)
                language_aliases = _coerce_language_aliases(
                    raw_value.get("language_aliases", {})
                )
            else:
                code = raw_value
                display = alias
                synonyms = ()
                language_aliases = {}
            if code is None:
                raise TypeError("specialized vocabulary concepts must declare a code")
            preferred_term = str(display or alias).strip()
            if not preferred_term:
                raise TypeError(
                    "specialized vocabulary concepts must declare a display term"
                )
            all_synonyms = tuple(
                dict.fromkeys(
                    str(item).strip()
                    for item in (*synonyms, alias)
                    if str(item).strip() and str(item).strip() != preferred_term
                )
            )
            key = (str(code).strip(), preferred_term, all_synonyms)
            if key in seen:
                continue
            seen.add(key)
            concepts.append(
                VocabConcept(
                    system=system,
                    code=key[0],
                    preferred_term=preferred_term,
                    synonyms=all_synonyms,
                    language_aliases=language_aliases,
                    source="specialized-loader",
                )
            )
    if not concepts:
        raise ValueError(
            f"specialized vocabulary loader returned no terms for {system!r}"
        )
    return VocabularyIndex(system, concepts)


def _coerce_aliases(value: Any) -> tuple[str, ...]:
    """Normalize an optional alias sequence without retaining source objects."""

    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    try:
        return tuple(str(item) for item in value)
    except TypeError as exc:
        raise TypeError("specialized vocabulary aliases must be iterable") from exc


def _coerce_language_aliases(value: Any) -> Mapping[str, tuple[str, ...]]:
    """Normalize optional language-tagged aliases from a loader mapping."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("specialized vocabulary language_aliases must be a mapping")
    return {
        str(language): _coerce_aliases(aliases) for language, aliases in value.items()
    }


def _snapshot_loader(snapshot: Any, *, offline: bool) -> Any:
    """Normalize accepted caller-owned snapshot forms to a loader object."""

    if snapshot is None:
        return None
    if isinstance(snapshot, VocabLoader):
        return snapshot
    if isinstance(snapshot, VocabSource):
        return VocabLoader(
            local_only=offline,
            registry={snapshot.system: snapshot},
        )
    if isinstance(snapshot, (str, Path)):
        return VocabLoader(cache_dir=snapshot, local_only=offline)
    if isinstance(snapshot, VocabularyIndex):
        return _SnapshotLoader({snapshot.system: snapshot})
    if isinstance(snapshot, Mapping):
        if not snapshot:
            raise ValueError("snapshot mapping must contain at least one system")
        if all(
            isinstance(value, VocabSource) or isinstance(value, (str, Path))
            for value in snapshot.values()
        ):
            registry = {
                str(system): (
                    value
                    if isinstance(value, VocabSource)
                    else VocabSource(system=str(system), path=value)
                )
                for system, value in snapshot.items()
            }
            return VocabLoader(local_only=offline, registry=registry)
        return _SnapshotLoader(snapshot)
    if (
        callable(getattr(snapshot, "get_index", None))
        or hasattr(snapshot, "index")
        or callable(getattr(snapshot, "load", None))
    ):
        system = getattr(snapshot, "system_uri", None) or getattr(
            snapshot, "system", None
        )
        return _SnapshotLoader(
            {_snapshot_system_key(system) if system is not None else "*": snapshot}
        )
    raise TypeError(
        "snapshot must be a VocabLoader, local cache path, VocabularyIndex, "
        "mapping, specialized vocabulary loader, or validated local snapshot object"
    )


def _config_for_top_k(config: RankingConfig | None, top_k: int) -> RankingConfig | None:
    """Ensure the internal retrieval stage can retain the requested depth."""

    if config is None:
        return None
    if config.k >= top_k:
        return config
    return replace(config, k=top_k)


def _extract_text_entities(
    text: str,
    systems: Sequence[str],
    *,
    loader: Any,
    language: str,
) -> list[dict[str, Any]]:
    """Find local vocabulary aliases in raw text without a model or network.

    This is intentionally a conservative lexical NER pass: it enumerates token
    windows and confirms every emitted window against a caller-loaded snapshot.
    Longest non-overlapping matches win, preserving deterministic offsets while
    avoiding a model download in the public facade.
    """

    if not text:
        return [{"text": text, "start": 0, "end": 0}]
    free_systems = tuple(system for system in systems if system in _FREE_ALIASES)
    if loader is None or not free_systems:
        return [{"text": text, "start": 0, "end": len(text)}]

    tokens = [match.span() for match in _TOKEN_PATTERN.finditer(text)]
    if not tokens:
        return [{"text": text, "start": 0, "end": len(text)}]

    matches: list[tuple[int, int, int]] = []
    for system in free_systems:
        index = loader.get_index(system)
        aliases = index.aliases_for_language(language)
        max_tokens = min(
            16,
            max((len(alias.split()) for alias in aliases), default=1),
        )
        for start_index in range(len(tokens)):
            for end_index in range(
                start_index + 1,
                min(len(tokens), start_index + max_tokens) + 1,
            ):
                start, _ = tokens[start_index]
                _, end = tokens[end_index - 1]
                trimmed_start = start
                trimmed_end = end
                while (
                    trimmed_start < trimmed_end and text[trimmed_start] in _TRIM_SURFACE
                ):
                    trimmed_start += 1
                while (
                    trimmed_end > trimmed_start
                    and text[trimmed_end - 1] in _TRIM_SURFACE
                ):
                    trimmed_end -= 1
                if trimmed_start >= trimmed_end:
                    continue
                surface = text[trimmed_start:trimmed_end]
                if index.lookup_all(surface, language=language):
                    matches.append((trimmed_start, trimmed_end, len(surface)))

    selected: list[tuple[int, int]] = []
    for start, end, _ in sorted(matches, key=lambda item: (-item[2], item[0], item[1])):
        if any(
            start < existing_end and end > existing_start
            for existing_start, existing_end in selected
        ):
            continue
        selected.append((start, end))
    selected.sort()
    if not selected:
        return [{"text": text, "start": 0, "end": len(text)}]
    return [
        {
            "text": text[start:end],
            "start": start,
            "end": end,
            "metadata": {"extraction": "local-vocabulary-ner"},
        }
        for start, end in selected
    ]


def ground(
    text_or_entities: Iterable[Any] | Mapping[str, Any] | GroundedSpan | str,
    systems: Sequence[str] = DEFAULT_GROUNDING_SYSTEMS,
    lang: str = "en",
    top_k: int = 1,
    snapshot: Any = None,
    *,
    loader: VocabLoader | None = None,
    encoder: AliasEncoder | None = None,
    config: RankingConfig | None = None,
    restricted_loaders: Mapping[str, UserKeyVocabularyLoader] | None = None,
    restricted_endpoint: Any = None,
    source_language: str | None = None,
    offline: bool = True,
    local_only: bool | None = None,
    terminology_bridge: Any = None,
    bridge: Any = None,
    normalize_composites: bool = False,
    composite_atomic_terms: Iterable[str] | None = None,
    postcoordination: PostCoordinationStage | None = None,
) -> GroundingResult:
    """Ground text or extracted entities against local vocabulary snapshots.

    The function composes the existing sparse/dense retrieval and reranking
    stage for freely redistributable vocabularies. UMLS and SNOMED CT are never
    bundled or downloaded: requesting either requires a matching explicit
    :class:`UserKeyVocabularyLoader` over a caller-owned local alias table.

    A raw string is scanned with a deterministic, local vocabulary-backed NER
    pass before linking.  Pre-extracted entity records may be mappings or
    objects exposing ``text``/``start``/``end``. ``canonical_label``,
    ``assertion``, ``section``, and ``source_language`` fields are consumed when
    present. Results retain the source offsets and assertion context for
    deterministic FHIR and OMOP export.

    Text input::

        result = ground("metformin 500 mg", systems=["rxnorm"],
                        lang="en", top_k=1, snapshot=local_snapshot)
        concept = result.concepts[0]
        assert concept.start == 0 and concept.code

    Entity-list input::

        result = ground(
            [{"text": "metformin", "start": 0, "end": 9}],
            systems=["rxnorm"],
            snapshot=local_snapshot,
        )

    Multi-system input::

        result = ground(
            "metformin and type 2 diabetes",
            systems=["rxnorm", "loinc", "icd10cm"],
            snapshot=local_snapshot,
        )

    Args:
        text_or_entities: Raw text, one entity mapping, or extracted entity
            records.
        systems: Ordered vocabulary systems. The default is RxNorm, LOINC, and
            ICD-10-CM.
        lang: Source language used for local alias routing.
        top_k: Number of ranked candidates retained per system in the typed
            concept records.
        snapshot: A caller-owned local snapshot, ``VocabLoader``, cache path,
            or mapping of system names to local snapshot objects/sources.
        loader: Optional free-vocabulary loader, typically configured with
            local/cache sources for offline execution.
        encoder: Optional local dense encoder. No encoder download occurs.
        config: Optional ranking configuration.
        restricted_loaders: Explicit user-key-gated local UMLS/SNOMED loaders.
        restricted_endpoint: Optional caller-configured out-of-process endpoint
            for restricted terminology lookups.
        terminology_bridge: Alias for ``restricted_endpoint`` naming the
            caller-supplied restricted terminology bridge explicitly.
        bridge: Short alias for ``terminology_bridge``.
        source_language: Default source language when a span omits one.
        offline: Whether to block network access during grounding. It defaults to
            ``True`` for this local-first facade.
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
        A :class:`GroundingResult`. Iteration preserves the existing
        ``GroundedSpan`` sequence contract; ``result.concepts`` contains typed
        one-system-per-concept records.

    Raises:
        ValueError: If no systems are requested, ``top_k`` is invalid, or a span
            is malformed.
        GroundingConfigError: If restricted terminology is requested without a
            configured user-supplied bridge/resource.
    """

    if local_only is not None:
        offline = bool(local_only)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(lang, str) or not lang.strip():
        raise ValueError("lang must be a non-empty language string")
    effective_language = normalize_language(
        source_language if source_language is not None else lang
    )
    ordered_systems = _normalize_systems(systems)
    if loader is not None and snapshot is not None:
        raise ValueError("pass either loader or snapshot, not both")
    selected_loader = (
        loader if loader is not None else _snapshot_loader(snapshot, offline=offline)
    )
    if selected_loader is not None and not callable(
        getattr(selected_loader, "get_index", None)
    ):
        selected_loader = _snapshot_loader(selected_loader, offline=offline)
    if selected_loader is None and any(
        canonical_system(system) not in RESTRICTED_SYSTEMS for system in ordered_systems
    ):
        selected_loader = VocabLoader(local_only=offline)
    if offline and selected_loader is not None:
        try:
            selected_loader.local_only = True
        except (AttributeError, TypeError):
            pass

    bridge_values = [
        value
        for value in (restricted_endpoint, terminology_bridge, bridge)
        if value is not None
    ]
    if len({id(value) for value in bridge_values}) > 1:
        raise ValueError(
            "pass only one of restricted_endpoint, terminology_bridge, or bridge"
        )
    selected_endpoint = bridge_values[0] if bridge_values else None
    effective_config = _config_for_top_k(config, top_k)

    with network_blocked_if_offline(local_only=offline):
        if isinstance(text_or_entities, str):
            raw_spans = _extract_text_entities(
                text_or_entities,
                ordered_systems,
                loader=selected_loader,
                language=effective_language,
            )
        else:
            raw_spans = _coerce_span_inputs(text_or_entities)
        grounded_spans = _ground_spans(
            raw_spans,
            ordered_systems,
            loader=selected_loader,
            encoder=encoder,
            config=effective_config,
            restricted_loaders=restricted_loaders,
            restricted_endpoint=selected_endpoint,
            source_language=effective_language,
            offline=offline,
            normalize_composites=normalize_composites,
            composite_atomic_terms=composite_atomic_terms,
            postcoordination=postcoordination,
        )
    return GroundingResult.from_spans(
        grounded_spans,
        systems=ordered_systems,
        language=effective_language,
        top_k=top_k,
        offline=offline,
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
    if selected_loader is not None and not callable(
        getattr(selected_loader, "get_index", None)
    ):
        selected_loader = _snapshot_loader(selected_loader, offline=offline)
    results = ground(
        spans if isinstance(spans, str) else raw_spans,
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
                if isinstance(matcher, _EndpointMatcher):
                    matches = matcher.lookup(
                        surface,
                        limit=1,
                        language=span.source_language,
                    )
                else:
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
                f"user-supplied {system.upper()} out-of-process terminology "
                "bridge (for example, a caller-operated terminology endpoint); "
                "restricted content is never bundled or downloaded."
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
        methods = ("lookup", "match", "link", "resolve")
        if not callable(endpoint) and not any(
            callable(getattr(endpoint, name, None)) for name in methods
        ):
            raise TypeError(
                "restricted_endpoint must expose lookup/match/link/resolve(system, text)"
            )

    def lookup(
        self,
        query: str,
        *,
        limit: int = 1,
        language: str | None = None,
    ) -> tuple[ConceptMatch, ...]:
        raw_matches = _lookup_restricted_endpoint(
            self.endpoint,
            self.system,
            query,
            limit=limit,
            language=language,
        )
        if isinstance(raw_matches, Mapping):
            raw_matches = raw_matches.get("matches", ())
        if isinstance(raw_matches, (str, bytes)) or raw_matches is None:
            return ()
        matches: list[ConceptMatch] = []
        for raw in raw_matches:
            if isinstance(raw, ConceptMatch):
                matches.append(raw)
                continue
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


def _lookup_restricted_endpoint(
    endpoint: Any,
    system: str,
    query: str,
    *,
    limit: int,
    language: str | None,
) -> Any:
    """Call common bridge shapes without importing optional bridge packages."""

    methods = [
        getattr(endpoint, name, None) for name in ("lookup", "match", "link", "resolve")
    ]
    method = next((candidate for candidate in methods if callable(candidate)), None)
    if method is None and callable(endpoint):
        method = endpoint
    if method is None:
        raise TypeError(
            "restricted_endpoint must expose lookup/match/link/resolve(system, text)"
        )

    attempts = (
        lambda: method(system, query, limit=limit, language=language),
        lambda: method(system, query, limit=limit),
        lambda: method(query, limit=limit, language=language),
        lambda: method(query, limit=limit),
        lambda: method(query),
    )
    last_error: TypeError | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as error:
            last_error = error
    assert last_error is not None
    raise last_error


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
    loader: Any,
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
        elif callable(getattr(loader, "get_index", None)):
            for raw_system in systems:
                system = canonical_system(raw_system)
                index = loader.get_index(system)
                result[system] = {
                    "system": system,
                    "system_uri": system_uri(system) or "",
                    "version": index.content_hash,
                    "sha256": index.content_hash,
                    "content_hash": index.content_hash,
                    "artifact": "local-snapshot",
                }
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
