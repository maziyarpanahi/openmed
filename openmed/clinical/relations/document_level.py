"""Privacy-safe document-level clinical relation aggregation."""

from __future__ import annotations

import hashlib
import hmac
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

from openmed.core.decoding import (
    EdgeCardinality,
    SpanEdge,
    SpanGraphConstraints,
    SpanNode,
    decode_span_graph,
)

from .candidate import (
    RelationCandidateBatch,
    RelationCandidateRule,
    SpanReference,
    build_relation_candidates,
    split_sentence_offsets,
)

DOCUMENT_RELATION_SCHEMA_VERSION = 1
DOCUMENT_RELATION_ADVISORY = (
    "Document-level relations are deterministic assistive extraction output, "
    "not a clinical decision and not a substitute for clinician review."
)

SpanOffset = tuple[int, int]


class _EntityLookupProvider(Protocol):
    def entity_ids_by_offset(self) -> Mapping[Any, str]: ...


EntityIdLookup = Mapping[Any, str] | _EntityLookupProvider

_HASH_RE = re.compile(r"(?:hmac-)?sha256:[0-9a-f]{64}")
_ENTITY_ID_KEYS = ("coref_entity_id", "entity_id", "cluster_id")


@dataclass(frozen=True)
class SafeRelationMention:
    """One relation endpoint represented without its source surface text."""

    start: int
    end: int
    label: str
    text_hash: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("mention offsets must satisfy 0 <= start < end")
        if not self.label:
            raise ValueError("mention label must be non-empty")
        if _HASH_RE.fullmatch(self.text_hash) is None:
            raise ValueError("mention text_hash must be SHA-256 or HMAC-SHA256")

    @property
    def offset(self) -> SpanOffset:
        """Return the half-open source offset."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record without source text."""

        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class MentionPairEvidence:
    """One supporting mention pair and its minimal sentence evidence."""

    head: SafeRelationMention
    tail: SafeRelationMention
    evidence_sentence_offsets: tuple[SpanOffset, ...]
    score: float
    sentence_distance: int

    def __post_init__(self) -> None:
        evidence = _normalized_offsets(self.evidence_sentence_offsets)
        if not evidence:
            raise ValueError("mention-pair evidence must include a sentence offset")
        if not _offset_is_covered(self.head.offset, evidence):
            raise ValueError("evidence must cover the head mention")
        if not _offset_is_covered(self.tail.offset, evidence):
            raise ValueError("evidence must cover the tail mention")
        if not math.isfinite(float(self.score)):
            raise ValueError("mention-pair score must be finite")
        if self.sentence_distance < 0:
            raise ValueError("sentence_distance must be non-negative")
        object.__setattr__(self, "evidence_sentence_offsets", evidence)
        object.__setattr__(self, "score", float(self.score))

    @property
    def is_cross_sentence(self) -> bool:
        """Return whether this evidence pair crosses a sentence boundary."""

        return self.sentence_distance > 0

    def to_dict(self) -> dict[str, Any]:
        """Return privacy-safe mention-pair evidence."""

        return {
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "evidence_sentence_offsets": [
                {"start": start, "end": end}
                for start, end in self.evidence_sentence_offsets
            ],
            "score": self.score,
            "sentence_distance": self.sentence_distance,
        }


@dataclass(frozen=True)
class DocumentLevelRelation:
    """One entity-level relation aggregated across document mention pairs."""

    document_hash: str
    relation_type: str
    head_entity_id: str
    tail_entity_id: str
    head: SafeRelationMention
    tail: SafeRelationMention
    score: float
    evidence_sentence_offsets: tuple[SpanOffset, ...]
    mention_pairs: tuple[MentionPairEvidence, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    advisory: str = DOCUMENT_RELATION_ADVISORY
    schema_version: int = DOCUMENT_RELATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in ("document_hash", "head_entity_id", "tail_entity_id"):
            if _HASH_RE.fullmatch(getattr(self, field_name)) is None:
                raise ValueError(f"{field_name} must be SHA-256 or HMAC-SHA256")
        if not self.relation_type:
            raise ValueError("relation_type must be non-empty")
        if not 0.0 <= float(self.score) <= 1.0:
            raise ValueError("document relation score must be between 0 and 1")
        mention_pairs = tuple(sorted(self.mention_pairs, key=_mention_pair_key))
        if not mention_pairs:
            raise ValueError("document relations require supporting mention pairs")
        evidence = _normalized_offsets(self.evidence_sentence_offsets)
        pair_evidence = {
            offset
            for mention_pair in mention_pairs
            for offset in mention_pair.evidence_sentence_offsets
        }
        if set(evidence) != pair_evidence:
            raise ValueError("relation evidence must equal its mention-pair evidence")
        object.__setattr__(self, "score", float(self.score))
        object.__setattr__(self, "mention_pairs", mention_pairs)
        object.__setattr__(self, "evidence_sentence_offsets", evidence)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def scope(self) -> str:
        """Return the relation evaluation scope."""

        return "document"

    @property
    def is_cross_sentence(self) -> bool:
        """Return whether any supporting mention pair crosses sentences."""

        return any(pair.is_cross_sentence for pair in self.mention_pairs)

    @property
    def head_entity_hash(self) -> str:
        """Return the privacy-safe head entity identity."""

        return self.head_entity_id

    @property
    def tail_entity_hash(self) -> str:
        """Return the privacy-safe tail entity identity."""

        return self.tail_entity_id

    def to_dict(self) -> dict[str, Any]:
        """Return offsets, hashes, scores, and provenance without source text."""

        max_sentence_distance = max(
            pair.sentence_distance for pair in self.mention_pairs
        )
        return {
            "schema_version": self.schema_version,
            "document_hash": self.document_hash,
            "type": self.relation_type,
            "scope": self.scope,
            "head_entity_id": self.head_entity_id,
            "tail_entity_id": self.tail_entity_id,
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "score": self.score,
            "evidence_sentence_offsets": [
                {"start": start, "end": end}
                for start, end in self.evidence_sentence_offsets
            ],
            "mention_pairs": [pair.to_dict() for pair in self.mention_pairs],
            "metadata": {
                "cross_sentence": self.is_cross_sentence,
                "sentence_distance": max_sentence_distance,
            },
            "provenance": _plain_value(self.provenance),
            "advisory": self.advisory,
        }


def aggregate_document_relations(
    text: str,
    candidates: RelationCandidateBatch,
    *,
    document_id: str = "document",
    entity_ids_by_offset: EntityIdLookup | None = None,
    cardinality: Mapping[str, EdgeCardinality] | None = None,
    min_relation_score: float = 0.5,
    hash_secret: str | bytes | None = None,
) -> tuple[DocumentLevelRelation, ...]:
    """Aggregate mention-pair candidates into constrained entity relations.

    Candidate edges are grouped by ``(head entity, tail entity, relation type)``
    and combined with noisy-or scoring. The shared span-graph decoder then
    applies the supplied cardinality rules once across the whole document.
    Output records intentionally exclude source text and candidate cue strings.

    Args:
        text: Original document used only to validate and hash source offsets.
        candidates: Mention-level nodes and scored relation candidates.
        document_id: Document-local provenance identity. Only its hash is emitted.
        entity_ids_by_offset: Optional coreference lookup. It may be a mapping
            keyed by ``(start, end)`` or ``(document_id, (start, end))``, or a
            coreference result exposing ``entity_ids_by_offset()``.
        cardinality: Per-relation document-wide cardinality constraints.
        min_relation_score: Minimum aggregated score passed to graph decoding.
        hash_secret: Optional HMAC secret for emitted document/entity/text hashes.

    Returns:
        Deterministically ordered, privacy-safe document-level relations.
    """

    if not document_id:
        raise ValueError("document_id must be non-empty")
    if not math.isfinite(float(min_relation_score)):
        raise ValueError("min_relation_score must be finite")
    lookup = _coerce_entity_lookup(entity_ids_by_offset)
    sentences = split_sentence_offsets(text)
    grouped_pairs: dict[tuple[str, str, str], list[MentionPairEvidence]] = defaultdict(
        list
    )
    grouped_languages: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    grouped_sources: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    entity_mentions: dict[str, dict[SpanOffset, SafeRelationMention]] = defaultdict(
        dict
    )

    for edge in candidates.candidates:
        head_reference = candidates.spans_by_node_id.get(edge.head)
        tail_reference = candidates.spans_by_node_id.get(edge.tail)
        if head_reference is None or tail_reference is None:
            continue
        head_identity = _entity_identity(
            lookup,
            document_id=document_id,
            reference=head_reference,
        )
        tail_identity = _entity_identity(
            lookup,
            document_id=document_id,
            reference=tail_reference,
        )
        head_entity_id = _hash_value(
            head_identity,
            namespace="entity",
            hash_secret=hash_secret,
        )
        tail_entity_id = _hash_value(
            tail_identity,
            namespace="entity",
            hash_secret=hash_secret,
        )
        if head_entity_id == tail_entity_id:
            continue

        head = _safe_mention(text, head_reference, hash_secret=hash_secret)
        tail = _safe_mention(text, tail_reference, hash_secret=hash_secret)
        evidence_offsets, sentence_distance = _candidate_evidence(
            edge,
            head_reference,
            tail_reference,
            sentences,
        )
        mention_pair = MentionPairEvidence(
            head=head,
            tail=tail,
            evidence_sentence_offsets=evidence_offsets,
            score=edge.score,
            sentence_distance=sentence_distance,
        )
        key = head_entity_id, tail_entity_id, edge.label
        grouped_pairs[key].append(mention_pair)
        entity_mentions[head_entity_id][head.offset] = head
        entity_mentions[tail_entity_id][tail.offset] = tail
        language = edge.metadata.get("language")
        if language:
            grouped_languages[key].add(str(language))
        source_relation = edge.metadata.get("source_relation")
        if source_relation:
            grouped_sources[key].add(str(source_relation))

    graph_nodes = tuple(
        _entity_node(entity_id, mentions)
        for entity_id, mentions in sorted(entity_mentions.items())
    )
    graph_edges: list[SpanEdge] = []
    for key, mention_pairs in sorted(grouped_pairs.items()):
        head_entity_id, tail_entity_id, relation_type = key
        graph_edges.append(
            SpanEdge(
                head=head_entity_id,
                tail=tail_entity_id,
                label=relation_type,
                score=_noisy_or(pair.score for pair in mention_pairs),
                metadata={"mention_pair_count": len(mention_pairs)},
            )
        )

    graph = decode_span_graph(
        graph_nodes,
        graph_edges,
        constraints=SpanGraphConstraints(cardinality=cardinality or {}),
        min_edge_score=float(min_relation_score),
    )
    document_hash = _hash_value(
        document_id,
        namespace="document",
        hash_secret=hash_secret,
    )
    relations: list[DocumentLevelRelation] = []
    for edge in graph.edges:
        key = edge.head, edge.tail, edge.label
        mention_pairs = tuple(grouped_pairs[key])
        evidence = tuple(
            sorted(
                {
                    offset
                    for mention_pair in mention_pairs
                    for offset in mention_pair.evidence_sentence_offsets
                }
            )
        )
        relations.append(
            DocumentLevelRelation(
                document_hash=document_hash,
                relation_type=edge.label,
                head_entity_id=edge.head,
                tail_entity_id=edge.tail,
                head=min((pair.head for pair in mention_pairs), key=_mention_key),
                tail=min((pair.tail for pair in mention_pairs), key=_mention_key),
                score=edge.score,
                evidence_sentence_offsets=evidence,
                mention_pairs=mention_pairs,
                provenance={
                    "aggregation": "noisy_or",
                    "candidate_count": len(mention_pairs),
                    "document_wide_cardinality": bool(cardinality),
                    "graph_decoder": "span_graph",
                    "languages": tuple(sorted(grouped_languages[key])),
                    "source_relations": tuple(sorted(grouped_sources[key])),
                },
            )
        )
    return tuple(sorted(relations, key=_relation_key))


def extract_document_relations(
    text: str,
    spans: Iterable[Any],
    rules: Iterable[RelationCandidateRule],
    *,
    document_id: str = "document",
    language: str = "en",
    max_sentence_distance: int = 2,
    entity_ids_by_offset: EntityIdLookup | None = None,
    cardinality: Mapping[str, EdgeCardinality] | None = None,
    min_relation_score: float = 0.5,
    hash_secret: str | bytes | None = None,
) -> tuple[DocumentLevelRelation, ...]:
    """Generate and aggregate bounded document-level relation candidates."""

    span_items = tuple(spans)
    lookup = entity_ids_by_offset
    if lookup is None:
        lookup = _entity_lookup_from_spans(
            span_items,
            document_id=document_id,
        )
    batch = build_relation_candidates(
        text,
        span_items,
        rules,
        language=language,
        max_sentence_distance=max_sentence_distance,
    )
    return aggregate_document_relations(
        text,
        batch,
        document_id=document_id,
        entity_ids_by_offset=lookup,
        cardinality=cardinality,
        min_relation_score=min_relation_score,
        hash_secret=hash_secret,
    )


def _coerce_entity_lookup(value: EntityIdLookup | None) -> Mapping[Any, str]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    resolver = getattr(value, "entity_ids_by_offset", None)
    if callable(resolver):
        resolved = resolver()
        if isinstance(resolved, Mapping):
            return resolved
    raise TypeError(
        "entity_ids_by_offset must be a mapping or coreference result lookup"
    )


def _entity_lookup_from_spans(
    spans: Iterable[Any],
    *,
    document_id: str,
) -> dict[Any, str]:
    lookup: dict[Any, str] = {}
    for span in spans:
        data = span if isinstance(span, Mapping) else vars(span)
        start = data.get("start", data.get("start_char"))
        end = data.get("end", data.get("end_char"))
        if start is None or end is None:
            continue
        for key in _ENTITY_ID_KEYS:
            entity_id = data.get(key)
            if entity_id:
                lookup[(document_id, (int(start), int(end)))] = str(entity_id)
                break
    return lookup


def _entity_identity(
    lookup: Mapping[Any, str],
    *,
    document_id: str,
    reference: SpanReference,
) -> str:
    offset = reference.offset_key()
    for key in ((document_id, offset), offset, f"{offset[0]}:{offset[1]}"):
        entity_id = lookup.get(key)
        if entity_id:
            return str(entity_id)
    return f"mention:{reference.label}:{reference.start}:{reference.end}"


def _safe_mention(
    text: str,
    reference: SpanReference,
    *,
    hash_secret: str | bytes | None,
) -> SafeRelationMention:
    if not 0 <= reference.start < reference.end <= len(text):
        raise ValueError("relation mention offsets fall outside document text")
    return SafeRelationMention(
        start=reference.start,
        end=reference.end,
        label=reference.label,
        text_hash=_hash_value(
            text[reference.start : reference.end],
            namespace="mention",
            hash_secret=hash_secret,
        ),
    )


def _candidate_evidence(
    edge: SpanEdge,
    head: SpanReference,
    tail: SpanReference,
    sentences: tuple[SpanOffset, ...],
) -> tuple[tuple[SpanOffset, ...], int]:
    head_sentence_index = _sentence_index(sentences, head.start, head.end)
    tail_sentence_index = _sentence_index(sentences, tail.start, tail.end)
    evidence = {
        sentences[head_sentence_index],
        sentences[tail_sentence_index],
    }
    cue_start = edge.metadata.get("cue_start")
    if isinstance(cue_start, int) and 0 <= cue_start:
        evidence.add(sentences[_sentence_index(sentences, cue_start, cue_start + 1)])
    return (
        tuple(sorted(evidence)),
        abs(head_sentence_index - tail_sentence_index),
    )


def _sentence_index(
    sentences: tuple[SpanOffset, ...],
    start: int,
    end: int,
) -> int:
    for index, (sentence_start, sentence_end) in enumerate(sentences):
        if sentence_start <= start and end <= sentence_end:
            return index
    return min(
        range(len(sentences)),
        key=lambda index: min(
            abs(start - sentences[index][0]),
            abs(start - sentences[index][1]),
        ),
    )


def _entity_node(
    entity_id: str,
    mentions: Mapping[SpanOffset, SafeRelationMention],
) -> SpanNode:
    representative = min(mentions.values(), key=_mention_key)
    return SpanNode(
        node_id=entity_id,
        start=representative.start,
        end=representative.end,
        label=representative.label,
        text_hash=representative.text_hash,
        metadata={"mention_offsets": tuple(sorted(mentions))},
    )


def _noisy_or(scores: Iterable[float]) -> float:
    complement = 1.0
    for score in scores:
        bounded = max(0.0, min(float(score), 1.0))
        complement *= 1.0 - bounded
    return round(1.0 - complement, 6)


def _hash_value(
    value: str,
    *,
    namespace: str,
    hash_secret: str | bytes | None,
) -> str:
    payload = f"{namespace}\0{value}".encode("utf-8")
    if hash_secret is None:
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"
    key = hash_secret.encode("utf-8") if isinstance(hash_secret, str) else hash_secret
    return f"hmac-sha256:{hmac.new(key, payload, hashlib.sha256).hexdigest()}"


def _normalized_offsets(offsets: Iterable[SpanOffset]) -> tuple[SpanOffset, ...]:
    normalized: set[SpanOffset] = set()
    for start, end in offsets:
        if start < 0 or end <= start:
            raise ValueError("evidence offsets must satisfy 0 <= start < end")
        normalized.add((int(start), int(end)))
    return tuple(sorted(normalized))


def _offset_is_covered(offset: SpanOffset, evidence: Iterable[SpanOffset]) -> bool:
    return any(start <= offset[0] and offset[1] <= end for start, end in evidence)


def _mention_key(mention: SafeRelationMention) -> tuple[int, int, str, str]:
    return mention.start, mention.end, mention.label, mention.text_hash


def _mention_pair_key(pair: MentionPairEvidence) -> tuple[Any, ...]:
    return (
        *_mention_key(pair.head),
        *_mention_key(pair.tail),
        pair.sentence_distance,
        -pair.score,
    )


def _relation_key(relation: DocumentLevelRelation) -> tuple[Any, ...]:
    return (
        relation.relation_type,
        relation.head.start,
        relation.head.end,
        relation.tail.start,
        relation.tail.end,
        relation.head_entity_id,
        relation.tail_entity_id,
    )


def _plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_value(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain_value(item) for item in value]
    return value


__all__ = [
    "DOCUMENT_RELATION_ADVISORY",
    "DOCUMENT_RELATION_SCHEMA_VERSION",
    "DocumentLevelRelation",
    "MentionPairEvidence",
    "SafeRelationMention",
    "aggregate_document_relations",
    "extract_document_relations",
]
