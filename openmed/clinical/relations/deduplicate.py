"""Deterministic, provenance-preserving collapse of clinical relations.

Relation extraction can emit the same normalized edge more than once when a
note repeats a mention, carries copied-forward text, or is processed by
several local extractors.  This module keeps those evidence locations while
aggregating confidence once per independent source.  It is intentionally
stdlib-only and does not resolve terminology or call a network service.

The input contract is deliberately small: callers provide normalized endpoint
identifiers, a relation type, a bounded score, and one or more source offsets.
Structural adapters also accept the existing relation objects and mappings so
the collapse boundary can be added after an extractor without changing that
extractor's public output.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

RELATION_DEDUPLICATION_SCHEMA_VERSION: Final[int] = 1
RELATION_DEDUPLICATION_ADVISORY: Final[str] = (
    "Clinical relation duplicate collapse is deterministic assistive output; "
    "independent-source confidence is not a clinical decision and requires "
    "human review."
)

SpanOffset = tuple[int, int]

_DEFAULT_SOURCE_ID: Final[str] = "document"
_HASH_RE = re.compile(r"^(?:hmac-)?sha256:[0-9a-f]{64}$")
_MISSING = object()
_CONTEXT_FIELDS: tuple[str, ...] = (
    "assertion_status",
    "polarity",
    "negation",
    "certainty",
    "temporality",
    "experiencer",
)
_RELATION_TYPE_FIELDS: tuple[str, ...] = (
    "relation_type",
    "normalized_relation_type",
    "predicate",
    "type",
    "label",
    "relation",
)
_SOURCE_FIELDS: tuple[str, ...] = (
    "source_id",
    "independent_source_id",
    "document_id",
    "document_hash",
    "doc_id",
)
_EVIDENCE_FIELDS: tuple[str, ...] = (
    "evidence_locations",
    "evidence",
    "locations",
)


class RelationDeduplicationError(ValueError):
    """Raised when relation candidates cannot be collapsed safely."""


@dataclass(frozen=True)
class RelationEvidence:
    """One source-local evidence location for a normalized relation.

    ``source_id`` is converted to an opaque SHA-256 identifier at construction
    time unless it is already a SHA-256 or HMAC-SHA-256 identifier.  The
    record therefore contains offsets and a source fingerprint, never source
    text or a caller's potentially identifying document name.
    """

    source_id: str
    start: int
    end: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_id", _source_identifier(self.source_id))
        start, end = _offset_pair((self.start, self.end))
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def offset(self) -> SpanOffset:
        """Return the half-open source offset."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free JSON-ready evidence location."""

        return {
            "source_id": self.source_id,
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class NormalizedRelationCandidate:
    """One normalized relation candidate before duplicate collapse.

    ``head`` and ``tail`` are normalized identifiers, not source surfaces.
    ``evidence`` must contain source offsets when the candidate is passed to
    :func:`collapse_duplicate_relations`.  A candidate may cite several source
    records; each cited source contributes at most once to the collapsed
    confidence score.
    """

    relation_type: str
    head: str = field(repr=False)
    tail: str = field(repr=False)
    score: float
    evidence: tuple[RelationEvidence, ...] = ()
    source_id: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_type",
            _normalized_text(self.relation_type, "relation_type"),
        )
        object.__setattr__(self, "head", _normalized_text(self.head, "head"))
        object.__setattr__(self, "tail", _normalized_text(self.tail, "tail"))
        object.__setattr__(self, "score", _score(self.score))

        evidence = tuple(self.evidence)
        if any(not isinstance(item, RelationEvidence) for item in evidence):
            raise RelationDeduplicationError(
                "relation candidate evidence must contain RelationEvidence"
            )
        object.__setattr__(self, "evidence", evidence)
        if self.source_id is not None:
            object.__setattr__(self, "source_id", _source_identifier(self.source_id))
        object.__setattr__(self, "context", _context_mapping(self.context))

    @property
    def normalized_head(self) -> str:
        """Return the normalized head identifier."""

        return self.head

    @property
    def normalized_tail(self) -> str:
        """Return the normalized tail identifier."""

        return self.tail

    def to_dict(self) -> dict[str, Any]:
        """Return the deterministic candidate representation."""

        payload: dict[str, Any] = {
            "relation_type": self.relation_type,
            "head": self.head,
            "tail": self.tail,
            "score": self.score,
            "evidence_locations": [item.to_dict() for item in self.evidence],
        }
        if self.source_id is not None:
            payload["source_id"] = self.source_id
        if self.context:
            payload["context"] = dict(self.context)
        return payload


@dataclass(frozen=True)
class CollapsedRelation:
    """One normalized relation with merged provenance and source-aware score.

    ``mention_count`` counts unique source/offset locations retained in
    ``evidence_locations``.  ``independent_source_count`` counts unique source
    fingerprints and is intentionally separate: ten mentions copied inside
    one source remain ten evidence locations but only one independent source.
    """

    relation_type: str
    head: str = field(repr=False)
    tail: str = field(repr=False)
    score: float
    evidence_locations: tuple[RelationEvidence, ...]
    mention_count: int
    independent_source_count: int
    candidate_count: int
    context: Mapping[str, str] = field(default_factory=dict, repr=False)
    schema_version: int = RELATION_DEDUPLICATION_SCHEMA_VERSION
    advisory: str = RELATION_DEDUPLICATION_ADVISORY

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_type",
            _normalized_text(self.relation_type, "relation_type"),
        )
        object.__setattr__(self, "head", _normalized_text(self.head, "head"))
        object.__setattr__(self, "tail", _normalized_text(self.tail, "tail"))
        object.__setattr__(self, "score", _score(self.score))
        if self.schema_version != RELATION_DEDUPLICATION_SCHEMA_VERSION:
            raise RelationDeduplicationError(
                "unsupported relation deduplication schema version"
            )
        if not isinstance(self.advisory, str) or not self.advisory:
            raise RelationDeduplicationError(
                "relation deduplication advisory is required"
            )

        evidence = tuple(sorted(set(self.evidence_locations), key=_evidence_key))
        if not evidence:
            raise RelationDeduplicationError(
                "collapsed relations require evidence locations"
            )
        if self.mention_count != len(evidence):
            raise RelationDeduplicationError("mention_count does not match evidence")
        source_count = len({item.source_id for item in evidence})
        if self.independent_source_count != source_count:
            raise RelationDeduplicationError(
                "independent_source_count does not match evidence"
            )
        if type(self.candidate_count) is not int or self.candidate_count < 1:
            raise RelationDeduplicationError("candidate_count must be positive")
        if type(self.mention_count) is not int or self.mention_count < 1:
            raise RelationDeduplicationError("mention_count must be positive")
        if type(self.independent_source_count) is not int or source_count < 1:
            raise RelationDeduplicationError(
                "independent_source_count must be positive"
            )
        object.__setattr__(self, "evidence_locations", evidence)
        object.__setattr__(self, "context", _context_mapping(self.context))

    @property
    def normalized_head(self) -> str:
        """Return the normalized head identifier."""

        return self.head

    @property
    def normalized_tail(self) -> str:
        """Return the normalized tail identifier."""

        return self.tail

    @property
    def evidence(self) -> tuple[RelationEvidence, ...]:
        """Return all unique supporting evidence locations."""

        return self.evidence_locations

    @property
    def source_count(self) -> int:
        """Return the independent-source count as a concise alias."""

        return self.independent_source_count

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free deterministic relation report."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "relation_type": self.relation_type,
            "head": self.head,
            "tail": self.tail,
            "score": self.score,
            "mention_count": self.mention_count,
            "independent_source_count": self.independent_source_count,
            "candidate_count": self.candidate_count,
            "evidence_locations": [item.to_dict() for item in self.evidence_locations],
            "aggregation": "noisy_or_by_independent_source",
            "advisory": self.advisory,
        }
        if self.context:
            payload["context"] = dict(self.context)
        return payload

    def to_json(self) -> str:
        """Return byte-stable JSON for local audit or review artifacts."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


@dataclass(frozen=True)
class _EndpointIdentity:
    key: str
    public: str


@dataclass(frozen=True)
class _CoercedCandidate:
    relation_type: str
    head: _EndpointIdentity
    tail: _EndpointIdentity
    score: float
    evidence: tuple[RelationEvidence, ...]
    context: Mapping[str, str]


def collapse_duplicate_relations(
    candidates: Iterable[Any] | Mapping[str, Any] | Any,
    *,
    source_id: str = _DEFAULT_SOURCE_ID,
    document_id: str | None = None,
    hash_secret: str | bytes | None = None,
) -> tuple[CollapsedRelation, ...]:
    """Collapse equivalent normalized relation candidates deterministically.

    Equivalence uses the normalized directed relation type, head, tail, and
    controlled assertion/context axes. Exact duplicate evidence coordinates
    are retained once; distinct offsets remain in the result. For each
    equivalent relation, the maximum candidate score from each independent
    source is selected and those source scores are combined with a bounded
    noisy-OR. Repeating a mention in one source therefore cannot inflate the
    score as if it were a second independent source.

    ``candidates`` may contain :class:`NormalizedRelationCandidate` records,
    mappings with ``relation_type``/``head``/``tail``/``evidence`` fields, or
    existing structural relation objects such as ``Relation``,
    ``MultilingualRelation``, and ``DocumentLevelRelation``. Mapping inputs
    should provide normalized endpoint fields (``normalized_head`` and
    ``normalized_tail``) when they differ from source surfaces. Existing
    relation objects fall back to hashed endpoint surfaces and preserve their
    source offsets without exporting those surfaces.

    Args:
        candidates: Candidate records or one candidate record.
        source_id: Default independent source for candidates whose evidence
            does not carry its own source identifier.
        document_id: Alias for ``source_id`` for document-batch callers.
        hash_secret: Optional caller-controlled HMAC key for source and
            fallback endpoint fingerprints. No key is persisted in a result.

    Returns:
        Collapsed relations in deterministic relation-key order.

    Raises:
        RelationDeduplicationError: If a candidate lacks normalized endpoints,
            a score/evidence location is malformed, or an input collection is
            not iterable. Error messages never include submitted values.
    """

    if document_id is not None:
        if source_id != _DEFAULT_SOURCE_ID:
            raise RelationDeduplicationError(
                "source_id and document_id cannot both override the default"
            )
        source_id = document_id
    _validate_hash_secret(hash_secret)
    default_source = _source_identifier(source_id, hash_secret=hash_secret)

    records = _candidate_collection(candidates)
    grouped: dict[tuple[Any, ...], list[_CoercedCandidate]] = {}
    for raw_candidate in records:
        candidate = _coerce_candidate(
            raw_candidate,
            default_source=default_source,
            hash_secret=hash_secret,
        )
        key = (
            candidate.relation_type,
            candidate.head.key,
            candidate.tail.key,
            tuple(candidate.context.items()),
        )
        grouped.setdefault(key, []).append(candidate)

    collapsed: list[CollapsedRelation] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        relation_type, _, _, context_items = key
        evidence = tuple(
            sorted(
                {item for candidate in group for item in candidate.evidence},
                key=_evidence_key,
            )
        )
        source_scores: dict[str, float] = {}
        for candidate in group:
            for item in candidate.evidence:
                source_scores[item.source_id] = max(
                    source_scores.get(item.source_id, 0.0),
                    candidate.score,
                )

        representative_head = min(
            (candidate.head for candidate in group),
            key=lambda endpoint: (endpoint.public, endpoint.key),
        )
        representative_tail = min(
            (candidate.tail for candidate in group),
            key=lambda endpoint: (endpoint.public, endpoint.key),
        )
        collapsed.append(
            CollapsedRelation(
                relation_type=relation_type,
                head=representative_head.public,
                tail=representative_tail.public,
                score=_independent_noisy_or(source_scores.values()),
                evidence_locations=evidence,
                mention_count=len(evidence),
                independent_source_count=len(source_scores),
                candidate_count=len(group),
                context=dict(context_items),
            )
        )
    return tuple(collapsed)


def _candidate_collection(candidates: Any) -> tuple[Any, ...]:
    if isinstance(candidates, (str, bytes, bytearray)):
        raise RelationDeduplicationError("relation candidates must be records")
    if isinstance(candidates, Mapping) or _is_relation_record(candidates):
        return (candidates,)
    try:
        return tuple(candidates)
    except (TypeError, ValueError):
        raise RelationDeduplicationError(
            "relation candidates must be an iterable of records"
        ) from None


def _is_relation_record(value: Any) -> bool:
    return any(
        _read(value, (field,), _MISSING) is not _MISSING
        for field in (
            "relation_type",
            "normalized_head",
            "head",
            "tail",
            "score",
        )
    )


def _coerce_candidate(
    raw: Any,
    *,
    default_source: str,
    hash_secret: str | bytes | None,
) -> _CoercedCandidate:
    if isinstance(raw, NormalizedRelationCandidate):
        relation_type = raw.relation_type
        head = _endpoint_identity(
            raw.head,
            explicit_normalized=True,
            hash_secret=hash_secret,
        )
        tail = _endpoint_identity(
            raw.tail,
            explicit_normalized=True,
            hash_secret=hash_secret,
        )
        evidence = _candidate_evidence(
            raw.evidence,
            raw_head=raw.head,
            raw_tail=raw.tail,
            candidate_source=raw.source_id or default_source,
            default_source=default_source,
            hash_secret=hash_secret,
        )
        return _CoercedCandidate(
            relation_type=relation_type,
            head=head,
            tail=tail,
            score=raw.score,
            evidence=evidence,
            context=_context_mapping(raw.context),
        )

    relation_value = _read(raw, _RELATION_TYPE_FIELDS, _MISSING)
    if relation_value is _MISSING:
        raise RelationDeduplicationError("relation candidate type is required")
    relation_type = _normalized_text(relation_value, "relation_type")

    raw_head, head_explicit = _endpoint_value(raw, "head")
    raw_tail, tail_explicit = _endpoint_value(raw, "tail")
    head = _endpoint_identity(
        raw_head,
        explicit_normalized=head_explicit,
        hash_secret=hash_secret,
    )
    tail = _endpoint_identity(
        raw_tail,
        explicit_normalized=tail_explicit,
        hash_secret=hash_secret,
    )

    raw_score = _read(raw, ("score", "confidence", "probability"), _MISSING)
    if raw_score is _MISSING:
        raise RelationDeduplicationError("relation candidate score is required")
    score = _score(raw_score)

    candidate_source = _read(raw, _SOURCE_FIELDS, _MISSING)
    if candidate_source is _MISSING:
        candidate_source = default_source
    evidence_value = _read(raw, _EVIDENCE_FIELDS, _MISSING)
    if evidence_value is _MISSING:
        evidence_value = _provenance_evidence(raw)
    evidence = _candidate_evidence(
        evidence_value,
        raw=raw,
        raw_head=raw_head,
        raw_tail=raw_tail,
        candidate_source=candidate_source,
        default_source=default_source,
        hash_secret=hash_secret,
    )
    return _CoercedCandidate(
        relation_type=relation_type,
        head=head,
        tail=tail,
        score=score,
        evidence=evidence,
        context=_candidate_context(raw),
    )


def _endpoint_value(raw: Any, endpoint: str) -> tuple[Any, bool]:
    if endpoint == "head":
        names = (
            "normalized_head",
            "head_normalized",
            "head_concept_id",
            "head_entity_id",
            "head_code",
            "head",
            "source",
            "subject",
        )
        explicit_names = names[:5]
    else:
        names = (
            "normalized_tail",
            "tail_normalized",
            "tail_concept_id",
            "tail_entity_id",
            "tail_code",
            "tail",
            "target",
            "object",
        )
        explicit_names = names[:5]
    value = _read(raw, names, _MISSING)
    if value is _MISSING:
        raise RelationDeduplicationError(
            f"relation candidate {endpoint} endpoint is required"
        )
    return value, any(
        _read(raw, (name,), _MISSING) is not _MISSING for name in explicit_names
    )


def _endpoint_identity(
    value: Any,
    *,
    explicit_normalized: bool,
    hash_secret: str | bytes | None,
) -> _EndpointIdentity:
    endpoint_value, endpoint_is_explicit = _endpoint_scalar(
        value,
        explicit_normalized=explicit_normalized,
    )
    key = _normalized_text(endpoint_value, "endpoint")
    public = (
        key
        if endpoint_is_explicit
        else _digest_identifier(
            key,
            namespace="relation-endpoint",
            hash_secret=hash_secret,
        )
    )
    return _EndpointIdentity(key=key, public=public)


def _endpoint_scalar(
    value: Any,
    *,
    explicit_normalized: bool,
) -> tuple[str, bool]:
    if isinstance(value, Mapping):
        normalized = _read(
            value,
            (
                "normalized",
                "normalised",
                "normalized_id",
                "concept_id",
                "code",
                "id",
            ),
            _MISSING,
        )
        if normalized is not _MISSING:
            system = _read(value, ("system", "code_system", "vocabulary"), _MISSING)
            if (
                system is not _MISSING
                and _read(value, ("code",), _MISSING) is not _MISSING
            ):
                normalized = f"{system}:{normalized}"
            return _scalar_text(normalized, "endpoint"), True
        surface = _read(
            value,
            ("text", "surface", "value", "display", "name"),
            _MISSING,
        )
        if surface is not _MISSING:
            return _scalar_text(surface, "endpoint"), explicit_normalized
        raise RelationDeduplicationError("relation endpoint identifier is required")

    normalized = _read(
        value,
        (
            "normalized",
            "normalised",
            "normalized_id",
            "concept_id",
            "code",
            "id",
        ),
        _MISSING,
    )
    if normalized is not _MISSING:
        return _scalar_text(normalized, "endpoint"), True
    surface = _read(value, ("text", "surface", "value", "display", "name"), _MISSING)
    if surface is not _MISSING:
        return _scalar_text(surface, "endpoint"), explicit_normalized
    return _scalar_text(value, "endpoint"), explicit_normalized


def _provenance_evidence(raw: Any) -> Any:
    mention_pairs = _read(raw, ("mention_pairs",), _MISSING)
    if mention_pairs is not _MISSING:
        return mention_pairs
    sentence_offsets = _read(
        raw,
        ("evidence_sentence_offsets", "source_offsets", "source_offset"),
        _MISSING,
    )
    if sentence_offsets is not _MISSING:
        return sentence_offsets
    provenance = _read(raw, ("provenance",), _MISSING)
    if isinstance(provenance, Mapping):
        return _read(provenance, _EVIDENCE_FIELDS, _MISSING)
    return _MISSING


def _candidate_evidence(
    value: Any,
    *,
    raw: Any = _MISSING,
    raw_head: Any = _MISSING,
    raw_tail: Any = _MISSING,
    candidate_source: Any,
    default_source: str,
    hash_secret: str | bytes | None,
) -> tuple[RelationEvidence, ...]:
    if value is _MISSING or value is None:
        items: list[Any] = []
    else:
        items = _flatten_evidence(value)

    if not items and raw is not _MISSING:
        raw_offset = _offset_from(raw)
        if raw_offset is not None:
            items.append(raw_offset)

    if not items and raw_head is not _MISSING and raw_tail is not _MISSING:
        head_offset = _offset_from(raw_head)
        tail_offset = _offset_from(raw_tail)
        if head_offset is not None and tail_offset is not None:
            items.append(
                (
                    min(head_offset[0], tail_offset[0]),
                    max(head_offset[1], tail_offset[1]),
                )
            )

    evidence: set[RelationEvidence] = set()
    for item in items:
        offset, item_source = _evidence_offset_and_source(item, candidate_source)
        evidence.add(
            _make_evidence(
                item_source or default_source,
                offset,
                hash_secret=hash_secret,
            )
        )
    if not evidence:
        raise RelationDeduplicationError("relation candidate evidence is required")
    return tuple(sorted(evidence, key=_evidence_key))


def _flatten_evidence(value: Any, *, inherited_source: Any = _MISSING) -> list[Any]:
    if isinstance(value, RelationEvidence):
        return [value]
    if isinstance(value, Mapping):
        source = _read(value, _SOURCE_FIELDS, inherited_source)
        if _offset_from(value) is not None:
            if (
                inherited_source is not _MISSING
                and _read(value, _SOURCE_FIELDS, _MISSING) is _MISSING
            ):
                return [{"source_id": inherited_source, **dict(value)}]
            return [value]
        nested = _read(
            value,
            (
                "evidence_locations",
                "evidence",
                "locations",
                "evidence_sentence_offsets",
                "offsets",
            ),
            _MISSING,
        )
        if nested is _MISSING:
            return []
        return _flatten_evidence(nested, inherited_source=source)
    nested_offsets = _read(
        value,
        ("evidence_locations", "evidence", "locations", "evidence_sentence_offsets"),
        _MISSING,
    )
    if nested_offsets is not _MISSING:
        source = _read(value, _SOURCE_FIELDS, inherited_source)
        return _flatten_evidence(nested_offsets, inherited_source=source)
    if _offset_from(value) is not None:
        if inherited_source is not _MISSING:
            return [{"source_id": inherited_source, "offset": value}]
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if _is_offset_pair(value):
            if inherited_source is not _MISSING:
                return [{"source_id": inherited_source, "offset": value}]
            return [value]
        items: list[Any] = []
        for child in value:
            items.extend(_flatten_evidence(child, inherited_source=inherited_source))
        return items
    return []


def _evidence_offset_and_source(
    value: Any,
    default_source: Any,
) -> tuple[SpanOffset, Any]:
    if isinstance(value, RelationEvidence):
        return value.offset, value.source_id
    if _is_offset_pair(value):
        return _offset_pair(value), default_source
    offset = _offset_from(value)
    if offset is None:
        raise RelationDeduplicationError("relation evidence offsets are required")
    source = _read(value, _SOURCE_FIELDS, default_source)
    return offset, source


def _make_evidence(
    source_id: Any,
    offset: SpanOffset,
    *,
    hash_secret: str | bytes | None,
) -> RelationEvidence:
    return RelationEvidence(
        source_id=_source_identifier(source_id, hash_secret=hash_secret),
        start=offset[0],
        end=offset[1],
    )


def _offset_from(value: Any) -> SpanOffset | None:
    if isinstance(value, RelationEvidence):
        return value.offset
    raw_offset = _read(
        value,
        ("source_offsets", "source_offset", "offset", "span", "offsets"),
        _MISSING,
    )
    if raw_offset is not _MISSING:
        try:
            return _offset_pair(raw_offset)
        except RelationDeduplicationError:
            return None
    start = _read(value, ("source_start", "start"), _MISSING)
    end = _read(value, ("source_end", "end"), _MISSING)
    if start is _MISSING or end is _MISSING:
        return None
    try:
        return _offset_pair((start, end))
    except RelationDeduplicationError:
        return None


def _is_offset_pair(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray))
        and len(value) == 2
        and type(value[0]) is int
        and type(value[1]) is int
    )


def _candidate_context(raw: Any) -> Mapping[str, str]:
    values: dict[str, Any] = {}
    direct_context = _read(raw, ("context", "qualifiers"), _MISSING)
    if isinstance(direct_context, Mapping):
        values.update(direct_context)
    assertion = _read(raw, ("assertion", "clinical_assertion"), _MISSING)
    if isinstance(assertion, Mapping):
        values.update(assertion)
    elif assertion is not _MISSING:
        values["assertion_status"] = assertion
    for field_name in _CONTEXT_FIELDS:
        value = _read(raw, (field_name,), _MISSING)
        if value is not _MISSING:
            values[field_name] = value
    return _context_mapping(values)


def _context_mapping(value: Mapping[str, Any] | Any) -> MappingProxyType:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise RelationDeduplicationError("relation context must be a mapping")
    normalized: dict[str, str] = {}
    for field_name in _CONTEXT_FIELDS:
        raw_value = value.get(field_name, _MISSING)
        if raw_value is _MISSING or raw_value is None:
            continue
        normalized[field_name] = _normalized_text(raw_value, field_name)
    return MappingProxyType(dict(sorted(normalized.items())))


def _read(value: Any, names: Sequence[str], default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        try:
            result = getattr(value, name)
        except (AttributeError, KeyError, TypeError):
            continue
        if result is not None:
            return result
    return default


def _normalized_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise RelationDeduplicationError(f"relation {field_name} must be text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized:
        raise RelationDeduplicationError(f"relation {field_name} must be non-empty")
    return normalized


def _scalar_text(value: Any, field_name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise RelationDeduplicationError(f"relation {field_name} must be scalar text")
    if isinstance(value, float) and not math.isfinite(value):
        raise RelationDeduplicationError(f"relation {field_name} must be finite")
    return str(value)


def _score(value: Any) -> float:
    if isinstance(value, bool):
        raise RelationDeduplicationError("relation score must be finite")
    try:
        score = float(value)
    except (TypeError, ValueError, OverflowError):
        raise RelationDeduplicationError("relation score must be finite") from None
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RelationDeduplicationError("relation score must be between zero and one")
    return score


def _offset_pair(value: Any) -> SpanOffset:
    if isinstance(value, Mapping):
        start = value.get("start", value.get("source_start", _MISSING))
        end = value.get("end", value.get("source_end", _MISSING))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise RelationDeduplicationError(
                "relation offsets must contain start and end"
            )
        start, end = value
    else:
        raise RelationDeduplicationError("relation offsets must contain start and end")
    if type(start) is not int or type(end) is not int or start < 0 or end <= start:
        raise RelationDeduplicationError(
            "relation offsets must satisfy 0 <= start < end"
        )
    return start, end


def _evidence_key(item: RelationEvidence) -> tuple[Any, ...]:
    return item.source_id, item.start, item.end


def _independent_noisy_or(scores: Iterable[float]) -> float:
    complement = 1.0
    for score in scores:
        complement *= 1.0 - _score(score)
    return round(1.0 - complement, 6)


def _validate_hash_secret(hash_secret: str | bytes | None) -> None:
    if hash_secret is not None and not isinstance(hash_secret, (str, bytes)):
        raise RelationDeduplicationError("hash_secret must be text or bytes")


def _source_identifier(
    value: Any,
    *,
    hash_secret: str | bytes | None = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RelationDeduplicationError("relation source_id must be non-empty text")
    normalized = value.strip()
    if _HASH_RE.fullmatch(normalized):
        return normalized
    return _digest_identifier(
        normalized,
        namespace="relation-source",
        hash_secret=hash_secret,
    )


def _digest_identifier(
    value: str,
    *,
    namespace: str,
    hash_secret: str | bytes | None,
) -> str:
    payload = f"{namespace}\0{value}".encode("utf-8")
    if hash_secret is None:
        digest = hashlib.sha256(payload).hexdigest()
        return f"sha256:{digest}"
    key = hash_secret.encode("utf-8") if isinstance(hash_secret, str) else hash_secret
    digest = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return f"hmac-sha256:{digest}"


collapse_relation_duplicates = collapse_duplicate_relations
collapse_duplicate_relation_candidates = collapse_duplicate_relations
deduplicate_relations = collapse_duplicate_relations


__all__ = [
    "RELATION_DEDUPLICATION_ADVISORY",
    "RELATION_DEDUPLICATION_SCHEMA_VERSION",
    "CollapsedRelation",
    "NormalizedRelationCandidate",
    "RelationDeduplicationError",
    "RelationEvidence",
    "collapse_duplicate_relation_candidates",
    "collapse_duplicate_relations",
    "collapse_relation_duplicates",
    "deduplicate_relations",
]
