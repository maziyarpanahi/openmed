"""Relation candidate schemas and script-agnostic candidate construction."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

from openmed.core.decoding import SpanEdge, SpanNode
from openmed.core.labels import normalize_label
from openmed.processing.advanced_ner import EntitySpan

MedicationAttributeType = Literal[
    "dose",
    "route",
    "frequency",
    "duration",
    "form",
    "strength",
    "indication",
]
MedicationRelationType = Literal[
    "drug_to_dose",
    "drug_to_route",
    "drug_to_frequency",
    "drug_to_duration",
    "drug_to_form",
    "drug_to_strength",
    "drug_to_indication",
]
ProblemAttributeType = Literal["severity", "body_site", "status"]
ProblemRelationType = Literal[
    "problem_to_severity",
    "problem_to_body_site",
    "problem_to_status",
]
RelationAttributeType = MedicationAttributeType | ProblemAttributeType
RelationType = MedicationRelationType | ProblemRelationType

RELATION_SCHEMA_VERSION = 2
DRUG_TO_DOSE: MedicationRelationType = "drug_to_dose"
DRUG_TO_ROUTE: MedicationRelationType = "drug_to_route"
DRUG_TO_FREQUENCY: MedicationRelationType = "drug_to_frequency"
DRUG_TO_DURATION: MedicationRelationType = "drug_to_duration"
DRUG_TO_FORM: MedicationRelationType = "drug_to_form"
DRUG_TO_STRENGTH: MedicationRelationType = "drug_to_strength"
DRUG_TO_INDICATION: MedicationRelationType = "drug_to_indication"

RELATION_ORDER: tuple[MedicationRelationType, ...] = (
    DRUG_TO_DOSE,
    DRUG_TO_ROUTE,
    DRUG_TO_FREQUENCY,
    DRUG_TO_DURATION,
    DRUG_TO_FORM,
    DRUG_TO_STRENGTH,
    DRUG_TO_INDICATION,
)
RELATION_ATTRIBUTE_TYPES: dict[MedicationRelationType, MedicationAttributeType] = {
    DRUG_TO_DOSE: "dose",
    DRUG_TO_ROUTE: "route",
    DRUG_TO_FREQUENCY: "frequency",
    DRUG_TO_DURATION: "duration",
    DRUG_TO_FORM: "form",
    DRUG_TO_STRENGTH: "strength",
    DRUG_TO_INDICATION: "indication",
}
ATTRIBUTE_RELATION_TYPES: dict[MedicationAttributeType, MedicationRelationType] = {
    attribute_type: relation_type
    for relation_type, attribute_type in RELATION_ATTRIBUTE_TYPES.items()
}
PROBLEM_ATTRIBUTE_RELATION_TYPES: dict[ProblemAttributeType, ProblemRelationType] = {
    "severity": "problem_to_severity",
    "body_site": "problem_to_body_site",
    "status": "problem_to_status",
}
_ALL_ATTRIBUTE_RELATION_TYPES: dict[RelationAttributeType, RelationType] = {
    **ATTRIBUTE_RELATION_TYPES,
    **PROBLEM_ATTRIBUTE_RELATION_TYPES,
}


@dataclass(frozen=True)
class SpanReference:
    """Stable snapshot of an entity span and its source offsets."""

    text: str
    label: str
    start: int
    end: int
    score: float
    section: str | None = None
    derived: bool = False

    @classmethod
    def from_entity(
        cls,
        span: EntitySpan,
        *,
        document_text: str | None = None,
        section: str | None = None,
    ) -> "SpanReference":
        """Create a stable reference from an ``EntitySpan``."""

        span_text = span.text
        if document_text is not None and 0 <= span.start <= span.end <= len(
            document_text
        ):
            span_text = document_text[span.start : span.end]
        return cls(
            text=span_text,
            label=span.label,
            start=span.start,
            end=span.end,
            score=float(span.score),
            section=section,
        )

    def offset_key(self) -> tuple[int, int]:
        """Return the character-offset identity for this span."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        payload: dict[str, Any] = {
            "text": self.text,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "score": self.score,
        }
        if self.section is not None:
            payload["section"] = self.section
        if self.derived:
            payload["derived"] = True
        return payload


@dataclass(frozen=True)
class RelationCandidateRule:
    """Language-specific rule used to construct typed relation candidates.

    Candidate construction operates only on source character offsets and cue
    substrings. It intentionally does not tokenize on whitespace, which keeps
    the same path valid for Chinese and Indic scripts.
    """

    relation_type: str
    source_relation: str
    head_labels: frozenset[str]
    tail_labels: frozenset[str]
    cues: tuple[str, ...]
    max_character_distance: int = 96

    def __post_init__(self) -> None:
        if not self.relation_type:
            raise ValueError("relation_type must be non-empty")
        if not self.source_relation:
            raise ValueError("source_relation must be non-empty")
        if not self.head_labels or not self.tail_labels:
            raise ValueError("relation rules require head and tail labels")
        if not self.cues:
            raise ValueError("relation rules require at least one cue")
        if self.max_character_distance < 0:
            raise ValueError("max_character_distance must be non-negative")


@dataclass(frozen=True)
class RelationCandidateBatch:
    """Span-graph inputs produced from already-extracted NER spans."""

    nodes: tuple[SpanNode, ...]
    candidates: tuple[SpanEdge, ...]
    spans_by_node_id: Mapping[str, SpanReference]


@dataclass(frozen=True)
class JointSpanCandidate:
    """One contiguous token span considered by the joint decoder head.

    Token offsets are half-open indices into the encoder sequence. Character
    offsets are half-open Python code-point indices into the source text.

    Args:
        token_start: Inclusive encoder-token index.
        token_end: Exclusive encoder-token index.
        start: Inclusive source character offset.
        end: Exclusive source character offset.
    """

    token_start: int
    token_end: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.token_start < self.token_end:
            raise ValueError("token offsets must satisfy 0 <= start < end")
        if not 0 <= self.start < self.end:
            raise ValueError("character offsets must satisfy 0 <= start < end")

    @property
    def token_width(self) -> int:
        """Return the number of encoder tokens covered by the span."""

        return self.token_end - self.token_start

    def stable_key(self) -> tuple[int, int, int, int]:
        """Return the deterministic identity used for sampling and sorting."""

        return self.token_start, self.token_end, self.start, self.end


@dataclass(frozen=True)
class SpanPairCandidate:
    """One directed span pair and its optional training relation label.

    Args:
        head: Directed relation source span.
        tail: Directed relation target span.
        relation_label: Typed relation label, or ``None`` for a negative pair.
    """

    head: JointSpanCandidate
    tail: JointSpanCandidate
    relation_label: str | None = None

    def __post_init__(self) -> None:
        if self.head == self.tail:
            raise ValueError("span pair head and tail must differ")
        if _token_spans_overlap(self.head, self.tail):
            raise ValueError("span pair head and tail must not overlap")
        if self.relation_label == "":
            raise ValueError("relation_label must be non-empty when provided")

    @property
    def is_negative(self) -> bool:
        """Return whether this pair is a no-relation training example."""

        return self.relation_label is None

    def stable_key(self) -> tuple[int, ...]:
        """Return a deterministic directed pair identity."""

        return (*self.head.stable_key(), *self.tail.stable_key())


def enumerate_joint_span_candidates(
    token_offsets: Sequence[tuple[int, int]],
    *,
    max_span_width: int,
) -> tuple[JointSpanCandidate, ...]:
    """Enumerate contiguous SpERT-style spans over encoder token offsets.

    Args:
        token_offsets: Monotonic half-open source character offsets for every
            encoder token.
        max_span_width: Maximum number of encoder tokens in one candidate.

    Returns:
        Candidate spans in deterministic token-boundary order.
    """

    if max_span_width < 1:
        raise ValueError("max_span_width must be positive")
    normalized_offsets = _validate_token_offsets(token_offsets)
    candidates: list[JointSpanCandidate] = []
    for token_start in range(len(normalized_offsets)):
        limit = min(len(normalized_offsets), token_start + max_span_width)
        for token_end in range(token_start + 1, limit + 1):
            candidates.append(
                JointSpanCandidate(
                    token_start=token_start,
                    token_end=token_end,
                    start=normalized_offsets[token_start][0],
                    end=normalized_offsets[token_end - 1][1],
                )
            )
    return tuple(candidates)


def enumerate_span_pair_candidates(
    spans: Sequence[JointSpanCandidate],
    *,
    max_token_distance: int | None = None,
) -> tuple[SpanPairCandidate, ...]:
    """Enumerate directed, non-overlapping span pairs for relation scoring.

    Args:
        spans: Contiguous entity-span candidates.
        max_token_distance: Optional maximum count of tokens between endpoints.

    Returns:
        Span pairs in deterministic directed order.
    """

    if max_token_distance is not None and max_token_distance < 0:
        raise ValueError("max_token_distance must be non-negative when provided")
    pairs: list[SpanPairCandidate] = []
    for head in spans:
        for tail in spans:
            if head == tail or _token_spans_overlap(head, tail):
                continue
            if (
                max_token_distance is not None
                and _span_token_distance(head, tail) > max_token_distance
            ):
                continue
            pairs.append(SpanPairCandidate(head=head, tail=tail))
    return tuple(sorted(pairs, key=SpanPairCandidate.stable_key))


def sample_negative_span_pairs(
    spans: Sequence[JointSpanCandidate],
    positive_pairs: Iterable[SpanPairCandidate],
    *,
    max_negatives: int,
    seed: int = 0,
    max_token_distance: int | None = None,
) -> tuple[SpanPairCandidate, ...]:
    """Sample deterministic no-relation pairs for joint-head training.

    Positive direction is significant: the reverse of a typed relation may be
    sampled as a negative. Selection uses a stable SHA-256 rank rather than a
    process-global random generator, so repeated offline recipes are identical.

    Args:
        spans: Contiguous entity-span candidates.
        positive_pairs: Directed pairs carrying gold typed relations.
        max_negatives: Maximum no-relation examples to return.
        seed: Stable offline sampling seed.
        max_token_distance: Optional maximum count of tokens between endpoints.

    Returns:
        Deterministically selected pairs whose ``relation_label`` is ``None``.
    """

    if max_negatives < 0:
        raise ValueError("max_negatives must be non-negative")
    if max_negatives == 0:
        return ()
    positive_keys = {pair.stable_key() for pair in positive_pairs}
    negatives = [
        pair
        for pair in enumerate_span_pair_candidates(
            spans,
            max_token_distance=max_token_distance,
        )
        if pair.stable_key() not in positive_keys
    ]
    ranked = sorted(
        negatives,
        key=lambda pair: (
            hashlib.sha256(f"{seed}\0{pair.stable_key()}".encode("ascii")).hexdigest(),
            pair.stable_key(),
        ),
    )
    return tuple(sorted(ranked[:max_negatives], key=SpanPairCandidate.stable_key))


def _validate_token_offsets(
    token_offsets: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    previous_end = 0
    for offset in token_offsets:
        if len(offset) != 2:
            raise ValueError("each token offset must contain start and end")
        start, end = offset
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
        ):
            raise TypeError("token offsets must be integers")
        if start < previous_end or end <= start:
            raise ValueError("token offsets must be monotonic and non-overlapping")
        normalized.append((start, end))
        previous_end = end
    return tuple(normalized)


def _token_spans_overlap(
    left: JointSpanCandidate,
    right: JointSpanCandidate,
) -> bool:
    return left.token_start < right.token_end and right.token_start < left.token_end


def _span_token_distance(
    left: JointSpanCandidate,
    right: JointSpanCandidate,
) -> int:
    if left.token_end <= right.token_start:
        return right.token_start - left.token_end
    if right.token_end <= left.token_start:
        return left.token_start - right.token_end
    return 0


@dataclass(frozen=True)
class _MatchedCue:
    cue: str
    start: int
    end: int


def build_relation_candidates(
    text: str,
    spans: Iterable[Any],
    rules: Iterable[RelationCandidateRule],
    *,
    language: str,
    max_sentence_distance: int = 0,
) -> RelationCandidateBatch:
    """Build bounded graph candidates without word tokenization.

    Args:
        text: Original clinical text.
        spans: Existing NER spans with character offsets into ``text``.
        rules: Language-keyed relation rules.
        language: Language code recorded as safe graph provenance.
        max_sentence_distance: Maximum number of sentence boundaries between
            relation endpoints. The default preserves sentence-local candidate
            generation; positive values enable bounded document-level pairs.

    Returns:
        Nodes, candidate edges, and the stable node-to-span lookup used by the
        shared :func:`openmed.core.decoding.decode_span_graph` decoder.
    """

    if max_sentence_distance < 0:
        raise ValueError("max_sentence_distance must be non-negative")

    references = _coerce_relation_spans(text, spans)
    sentence_offsets = split_sentence_offsets(text)
    nodes: list[SpanNode] = []
    spans_by_node_id: dict[str, SpanReference] = {}
    for index, reference in enumerate(references):
        node_id = f"span-{index}"
        spans_by_node_id[node_id] = reference
        nodes.append(
            SpanNode(
                node_id=node_id,
                start=reference.start,
                end=reference.end,
                label=normalize_label(reference.label),
                score=reference.score,
                metadata={"language": language},
            )
        )

    candidates: list[SpanEdge] = []
    ordered_rules = sorted(
        rules,
        key=lambda rule: (rule.relation_type, rule.source_relation),
    )
    for head_node in nodes:
        for tail_node in nodes:
            if head_node.node_id == tail_node.node_id:
                continue
            head = spans_by_node_id[head_node.node_id]
            tail = spans_by_node_id[tail_node.node_id]
            head_sentence_index = _sentence_index_for_span(sentence_offsets, head)
            tail_sentence_index = _sentence_index_for_span(sentence_offsets, tail)
            sentence_distance = abs(head_sentence_index - tail_sentence_index)
            if sentence_distance > max_sentence_distance:
                continue
            distance = _character_distance(head, tail)
            window_start = min(head.start, tail.start)
            window = text[window_start : max(head.end, tail.end)]
            for rule in ordered_rules:
                if head_node.label not in rule.head_labels:
                    continue
                if tail_node.label not in rule.tail_labels:
                    continue
                if distance > rule.max_character_distance:
                    continue
                matched_cue = _matched_cue(window, rule.cues)
                if matched_cue is None:
                    continue
                cue_start = window_start + matched_cue.start
                cue_sentence_index = _sentence_index_for_offset(
                    sentence_offsets,
                    cue_start,
                )
                evidence_sentence_offsets = tuple(
                    sorted(
                        {
                            sentence_offsets[head_sentence_index],
                            sentence_offsets[tail_sentence_index],
                            sentence_offsets[cue_sentence_index],
                        }
                    )
                )
                candidates.append(
                    SpanEdge(
                        head=head_node.node_id,
                        tail=tail_node.node_id,
                        label=rule.relation_type,
                        score=_candidate_score(head, tail, distance),
                        metadata={
                            "character_distance": distance,
                            "cue_end": window_start + matched_cue.end,
                            "cue_start": cue_start,
                            "cross_sentence": sentence_distance > 0,
                            "evidence_sentence_offsets": evidence_sentence_offsets,
                            "head_sentence_offset": sentence_offsets[
                                head_sentence_index
                            ],
                            "language": language,
                            "matched_cue": matched_cue.cue,
                            "sentence_distance": sentence_distance,
                            "source_relation": rule.source_relation,
                            "tail_sentence_offset": sentence_offsets[
                                tail_sentence_index
                            ],
                        },
                    )
                )

    return RelationCandidateBatch(
        nodes=tuple(nodes),
        candidates=tuple(
            sorted(
                candidates,
                key=lambda edge: (edge.label, edge.head, edge.tail, -edge.score),
            )
        ),
        spans_by_node_id=MappingProxyType(spans_by_node_id),
    )


def split_sentence_offsets(text: str) -> tuple[tuple[int, int], ...]:
    """Return deterministic half-open offsets for document sentences."""

    offsets: list[tuple[int, int]] = []
    cursor = 0
    for boundary in re.finditer(r"[.!?。！？；;\n]+", text):
        start, end = _trim_sentence_offset(text, cursor, boundary.end())
        if start < end:
            offsets.append((start, end))
        cursor = boundary.end()
    start, end = _trim_sentence_offset(text, cursor, len(text))
    if start < end:
        offsets.append((start, end))
    if not offsets:
        return ((0, len(text)),)
    return tuple(offsets)


def _trim_sentence_offset(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_index_for_span(
    sentence_offsets: tuple[tuple[int, int], ...],
    span: SpanReference,
) -> int:
    for index, (start, end) in enumerate(sentence_offsets):
        if start <= span.start and span.end <= end:
            return index
    return _sentence_index_for_offset(sentence_offsets, span.start)


def _sentence_index_for_offset(
    sentence_offsets: tuple[tuple[int, int], ...],
    offset: int,
) -> int:
    for index, (start, end) in enumerate(sentence_offsets):
        if start <= offset < end:
            return index
    return min(
        range(len(sentence_offsets)),
        key=lambda index: min(
            abs(offset - sentence_offsets[index][0]),
            abs(offset - sentence_offsets[index][1]),
        ),
    )


def _coerce_relation_spans(
    text: str,
    spans: Iterable[Any],
) -> tuple[SpanReference, ...]:
    references: list[SpanReference] = []
    for item in spans:
        if isinstance(item, SpanReference):
            start = item.start
            end = item.end
            label = item.label
            score = item.score
            section = item.section
        elif isinstance(item, EntitySpan):
            start = item.start
            end = item.end
            label = item.label
            score = item.score
            section = None
        else:
            data = item if isinstance(item, Mapping) else vars(item)
            metadata = data.get("metadata") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            start = int(data.get("start", data.get("start_char", -1)))
            end = int(data.get("end", data.get("end_char", -1)))
            label = str(data.get("label", data.get("entity", "")))
            score = float(data.get("score", metadata.get("confidence", 1.0)))
            raw_section = data.get("section", metadata.get("section"))
            section = None if raw_section is None else str(raw_section)
        if not label or start < 0 or end <= start or end > len(text):
            continue
        references.append(
            SpanReference(
                text=text[start:end],
                label=str(label),
                start=start,
                end=end,
                score=float(score),
                section=section,
            )
        )

    unique = {
        (reference.start, reference.end, normalize_label(reference.label)): reference
        for reference in references
    }
    return tuple(
        sorted(
            unique.values(),
            key=lambda reference: (
                reference.start,
                reference.end,
                normalize_label(reference.label),
            ),
        )
    )


def _character_distance(left: SpanReference, right: SpanReference) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _matched_cue(window: str, cues: tuple[str, ...]) -> _MatchedCue | None:
    for cue in sorted(cues, key=lambda value: (-len(value), value)):
        match = re.search(re.escape(cue), window, flags=re.IGNORECASE)
        if match is not None:
            return _MatchedCue(cue=cue, start=match.start(), end=match.end())
    return None


def _candidate_score(
    head: SpanReference,
    tail: SpanReference,
    distance: int,
) -> float:
    entity_confidence = max(0.0, min((head.score + tail.score) / 2.0, 1.0))
    proximity = 1.0 / (1.0 + float(distance))
    return round(0.65 + 0.2 * entity_confidence + 0.15 * proximity, 6)


@dataclass(frozen=True)
class RelationCandidate:
    """Candidate ``drug -> attribute`` edge before constrained decoding."""

    relation_type: MedicationRelationType
    head: SpanReference
    attribute: SpanReference
    score: float
    confidence: float
    features: dict[str, float]
    explanation: tuple[str, ...]

    @property
    def attribute_type(self) -> MedicationAttributeType:
        """Return the schema attribute type for this candidate."""

        return RELATION_ATTRIBUTE_TYPES[self.relation_type]

    def stable_key(self) -> tuple[float, int, int, int, int, str]:
        """Sort key used by the deterministic constrained decoder."""

        relation_rank = RELATION_ORDER.index(self.relation_type)
        return (
            -self.score,
            relation_rank,
            self.head.start,
            self.attribute.start,
            self.attribute.end,
            self.attribute.text.casefold(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        return {
            "relation_type": self.relation_type,
            "attribute_type": self.attribute_type,
            "head": self.head.to_dict(),
            "attribute": self.attribute.to_dict(),
            "score": self.score,
            "confidence": self.confidence,
            "features": {key: self.features[key] for key in sorted(self.features)},
            "explanation": list(self.explanation),
        }


@dataclass(frozen=True)
class MedicationRelation:
    """Resolved medication relation with provenance and normalization."""

    relation_type: MedicationRelationType
    attribute_type: MedicationAttributeType
    head: SpanReference
    attribute: SpanReference
    score: float
    confidence: float
    features: dict[str, float]
    normalized: dict[str, Any] | None = None
    coreference: CoreferenceProvenance | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        payload: dict[str, Any] = {
            "relation_type": self.relation_type,
            "attribute_type": self.attribute_type,
            "head": self.head.to_dict(),
            "attribute": self.attribute.to_dict(),
            "head_offsets": {"start": self.head.start, "end": self.head.end},
            "attribute_offsets": {
                "start": self.attribute.start,
                "end": self.attribute.end,
            },
            "score": self.score,
            "confidence": self.confidence,
            "features": {key: self.features[key] for key in sorted(self.features)},
        }
        if self.normalized is not None:
            payload["normalized"] = dict(self.normalized)
        if self.coreference is not None:
            payload["coreference"] = self.coreference.to_dict()
        return payload


@dataclass(frozen=True)
class Relation:
    """Public clinical relation tuple with a bounded confidence score.

    Source-backed tails round-trip through ``start``/``end``. A derived tail,
    such as status consumed from clinical context metadata, carries
    ``derived=True`` and reuses its head offsets as provenance rather than
    claiming that the normalized status value occurs literally in the text.
    """

    head: SpanReference
    type: RelationAttributeType
    tail: SpanReference
    score: float

    def __post_init__(self) -> None:
        """Validate the generic relation confidence contract."""

        if self.type not in _ALL_ATTRIBUTE_RELATION_TYPES:
            raise ValueError(f"unsupported clinical relation type: {self.type!r}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("relation score must be between 0 and 1")

    @property
    def relation_type(self) -> RelationType:
        """Return the head-specific edge label."""

        return _ALL_ATTRIBUTE_RELATION_TYPES[self.type]

    def to_dict(self) -> dict[str, Any]:
        """Return the roadmap relation shape as a deterministic mapping."""

        return {
            "head": self.head.to_dict(),
            "type": self.type,
            "tail": self.tail.to_dict(),
            "score": self.score,
        }


@dataclass(frozen=True)
class CoreferenceSourceReference:
    """Privacy-safe source offsets and hash for one coreferent mention."""

    start: int
    end: int
    text_hash: str

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("coreference source offsets must satisfy 0 <= start < end")
        if re.fullmatch(r"hmac-sha256:[0-9a-f]{64}", self.text_hash) is None:
            raise ValueError("coreference source text_hash must be an HMAC-SHA256 hash")

    def to_dict(self) -> dict[str, int | str]:
        """Return a JSON-compatible reference without raw source text."""

        return {
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
        }


@dataclass(frozen=True)
class CoreferenceProvenance:
    """Document-local cluster identity and safe supporting mention evidence."""

    cluster_id: str
    representative: CoreferenceSourceReference
    supporting_mentions: tuple[CoreferenceSourceReference, ...]

    def __post_init__(self) -> None:
        if not self.cluster_id:
            raise ValueError("coreference cluster_id must be non-empty")
        supporting_mentions = tuple(
            sorted(
                self.supporting_mentions,
                key=lambda mention: (
                    mention.start,
                    mention.end,
                    mention.text_hash,
                ),
            )
        )
        offsets = [(mention.start, mention.end) for mention in supporting_mentions]
        if len(set(offsets)) != len(offsets):
            raise ValueError("coreference supporting mention offsets must be unique")
        if self.representative not in supporting_mentions:
            raise ValueError("coreference representative must be a supporting mention")
        object.__setattr__(self, "supporting_mentions", supporting_mentions)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic cluster provenance without mention surfaces."""

        return {
            "cluster_id": self.cluster_id,
            "representative": self.representative.to_dict(),
            "supporting_mentions": [
                mention.to_dict() for mention in self.supporting_mentions
            ],
        }


@dataclass(frozen=True)
class MedicationRelationGroup:
    """Medication head span plus its resolved typed attribute relations."""

    medication: SpanReference
    relations: tuple[MedicationRelation, ...]
    advisory: str
    coreference: CoreferenceProvenance | None = None

    @property
    def attributes(self) -> dict[MedicationAttributeType, MedicationRelation]:
        """Return relations keyed by attribute type."""

        return {relation.attribute_type: relation for relation in self.relations}

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic dictionary representation."""

        payload: dict[str, Any] = {
            "medication": self.medication.to_dict(),
            "relations": [relation.to_dict() for relation in self.relations],
            "advisory": self.advisory,
        }
        if self.coreference is not None:
            payload["coreference"] = self.coreference.to_dict()
        return payload


__all__ = [
    "ATTRIBUTE_RELATION_TYPES",
    "DRUG_TO_DOSE",
    "DRUG_TO_DURATION",
    "DRUG_TO_FREQUENCY",
    "DRUG_TO_FORM",
    "DRUG_TO_INDICATION",
    "DRUG_TO_ROUTE",
    "DRUG_TO_STRENGTH",
    "CoreferenceProvenance",
    "CoreferenceSourceReference",
    "MedicationAttributeType",
    "MedicationRelation",
    "MedicationRelationGroup",
    "MedicationRelationType",
    "ProblemAttributeType",
    "ProblemRelationType",
    "JointSpanCandidate",
    "RELATION_ATTRIBUTE_TYPES",
    "RELATION_ORDER",
    "RELATION_SCHEMA_VERSION",
    "RelationAttributeType",
    "RelationType",
    "PROBLEM_ATTRIBUTE_RELATION_TYPES",
    "RelationCandidateBatch",
    "RelationCandidateRule",
    "RelationCandidate",
    "Relation",
    "SpanPairCandidate",
    "SpanReference",
    "build_relation_candidates",
    "enumerate_joint_span_candidates",
    "enumerate_span_pair_candidates",
    "sample_negative_span_pairs",
    "split_sentence_offsets",
]
