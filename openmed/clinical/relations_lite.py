"""Offline, deterministic relation candidates for lightweight extraction.

This module deliberately sits beside the neural and feature-based relation
heads.  It consumes already extracted entity spans, applies a small typed
compatibility table, and returns offset-only candidates for downstream
grounding or export.  It does not copy entity surfaces into its public
records and never makes a clinical decision.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, Final, Literal

from openmed.clinical.sections import detect_sections
from openmed.core.labels import normalize_label
from openmed.processing.advanced_ner import EntitySpan

RelationType = Literal[
    "DRUG_DOSE",
    "DRUG_ROUTE",
    "PROBLEM_ANATOMY",
    "FINDING_SEVERITY",
]

RELATION_SCHEMA_VERSION: Final = 1
DRUG_DOSE: Final[RelationType] = "DRUG_DOSE"
DRUG_ROUTE: Final[RelationType] = "DRUG_ROUTE"
PROBLEM_ANATOMY: Final[RelationType] = "PROBLEM_ANATOMY"
FINDING_SEVERITY: Final[RelationType] = "FINDING_SEVERITY"
RELATION_TYPES: Final[tuple[RelationType, ...]] = (
    DRUG_DOSE,
    DRUG_ROUTE,
    PROBLEM_ANATOMY,
    FINDING_SEVERITY,
)

DEFAULT_MAX_CHARACTER_DISTANCE: Final = 64
DEFAULT_MAX_TOKEN_DISTANCE: Final = 8

_DRUG_LABELS: Final[frozenset[str]] = frozenset(
    {
        "ANTIBIOTIC",
        "ANESTHETIC_AGENT",
        "CHEMICAL",
        "DRUG",
        "MEDICATION",
        "VACCINE_NAME",
    }
)
_DOSE_LABELS: Final[frozenset[str]] = frozenset(
    {"DOSE", "DOSAGE", "DOSE_NUMBER", "STRENGTH"}
)
_ROUTE_LABELS: Final[frozenset[str]] = frozenset(
    {"ADMINISTRATION_ROUTE", "FEEDING_ROUTE", "ROUTE"}
)
_PROBLEM_LABELS: Final[frozenset[str]] = frozenset(
    {"CANCER", "CONDITION", "DISEASE", "PATHOLOGY", "PROBLEM", "SYMPTOM"}
)
_ANATOMY_LABELS: Final[frozenset[str]] = frozenset(
    {"ANATOMY", "BODY_SITE", "CELL", "ORGAN", "TISSUE"}
)
_FINDING_LABELS: Final[frozenset[str]] = frozenset(
    {"ABNORMALITY", "CONDITION", "FINDING", "PATHOLOGY", "RADIOLOGY_FINDING"}
)
_SEVERITY_LABELS: Final[frozenset[str]] = frozenset(
    {"FINDING_SEVERITY", "GRADE", "SEVERITY"}
)

RELATION_HEAD_LABELS: Final[Mapping[RelationType, frozenset[str]]] = {
    DRUG_DOSE: _DRUG_LABELS,
    DRUG_ROUTE: _DRUG_LABELS,
    PROBLEM_ANATOMY: _PROBLEM_LABELS,
    FINDING_SEVERITY: _FINDING_LABELS,
}
RELATION_TAIL_LABELS: Final[Mapping[RelationType, frozenset[str]]] = {
    DRUG_DOSE: _DOSE_LABELS,
    DRUG_ROUTE: _ROUTE_LABELS,
    PROBLEM_ANATOMY: _ANATOMY_LABELS,
    FINDING_SEVERITY: _SEVERITY_LABELS,
}

_RELATION_RANK: Final[dict[RelationType, int]] = {
    relation_type: index for index, relation_type in enumerate(RELATION_TYPES)
}

# ``normalize_label`` intentionally returns OTHER for a few common relation
# model labels that are not part of the shared taxonomy yet.  These aliases
# keep the scaffold compatible with those model outputs without broadening
# the global label taxonomy.
_LOCAL_LABEL_ALIASES: Final[dict[str, str]] = {
    "abnormality": "ABNORMALITY",
    "anatomicalsite": "ANATOMY",
    "bodylocation": "BODY_SITE",
    "dose": "DOSAGE",
    "doseamount": "DOSAGE",
    "findingseverity": "SEVERITY",
    "radiologyfinding": "FINDING",
    "routeofadministration": "ROUTE",
    "severityscore": "SEVERITY",
}

_FIELD_KEYS: Final[tuple[str, ...]] = (
    "start",
    "end",
    "start_char",
    "end_char",
    "label",
    "entity",
    "entity_type",
    "canonical_label",
    "type",
    "score",
    "confidence",
    "section",
    "section_id",
    "section_label",
    "section_name",
    "scope",
    "scope_id",
    "assertion_scope",
    "context_id",
    "sentence_id",
    "metadata",
    "assertion",
    "clinical_assertion",
    "clinical_context",
    "context",
    "negation",
    "certainty",
    "uncertainty",
    "temporality",
    "experiencer",
    "assertion_status",
    "status",
)

_SENTENCE_BOUNDARY_RE: Final = re.compile(r"[.!?。！？]+|\n+")
_CLAUSE_BOUNDARY_RE: Final = re.compile(r"[;；]+")
_TOKEN_RE: Final = re.compile(r"\b[\w]+(?:[-/]\w+)*\b", re.UNICODE)
_CONNECTIVE_AFTER_RE: Final = re.compile(
    r"^(?:and|also|as well as|because of|due to|for|in|of|that|to|which|with)\b",
    re.IGNORECASE,
)
_CONNECTIVE_BEFORE_RE: Final = re.compile(
    r"(?:and|because|due|for|of|to|which|with)\s*$",
    re.IGNORECASE,
)

_NEGATIVE_ASSERTION_VALUES: Final[frozenset[str]] = frozenset(
    {
        "absent",
        "denied",
        "false",
        "negated",
        "not_present",
        "refuted",
    }
)
_NON_ASSERTED_VALUES: Final[frozenset[str]] = frozenset({"conditional", "hypothetical"})
_ASSERTION_AXES: Final[tuple[str, ...]] = (
    "negation",
    "certainty",
    "temporality",
    "experiencer",
    "status",
)


@dataclass(frozen=True, slots=True)
class RelationSpan:
    """Privacy-safe entity span carried by a lightweight relation.

    Only source offsets, a normalized entity label, and the upstream score are
    retained.  In particular, the source surface text is intentionally absent
    so serializing a candidate cannot copy clinical text or PHI.
    """

    start: int
    end: int
    label: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.start < 0 or self.end <= self.start:
            raise ValueError("relation span offsets must satisfy 0 <= start < end")
        if not self.label:
            raise ValueError("relation span label must be non-empty")
        if not isfinite(self.score) or not 0.0 <= self.score <= 1.0:
            raise ValueError("relation span score must be between 0 and 1")

    @property
    def offset(self) -> tuple[int, int]:
        """Return the half-open source offset pair."""

        return self.start, self.end

    def to_dict(self) -> dict[str, int | float | str]:
        """Return an offset-only JSON-compatible representation."""

        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class RelationCandidate:
    """One typed, proximity-scored edge between two extracted entities."""

    head: RelationSpan
    tail: RelationSpan
    relation_type: RelationType
    confidence: float
    distance: int = 0
    token_distance: int = 0
    sentence_distance: int = 0
    explicit_connective: bool = False

    def __post_init__(self) -> None:
        if self.relation_type not in RELATION_TYPES:
            raise ValueError(
                f"unsupported lightweight relation: {self.relation_type!r}"
            )
        if self.head.offset == self.tail.offset:
            raise ValueError("relation candidate head and tail must differ")
        if self.distance < 0 or self.token_distance < 0 or self.sentence_distance < 0:
            raise ValueError("relation distances must be non-negative")
        if not isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("relation confidence must be between 0 and 1")

    @property
    def score(self) -> float:
        """Return the heuristic confidence under the common score alias."""

        return self.confidence

    @property
    def type(self) -> RelationType:
        """Return the typed relation under the common ``type`` alias."""

        return self.relation_type

    @property
    def head_offset(self) -> tuple[int, int]:
        """Return the head source offsets."""

        return self.head.offset

    @property
    def tail_offset(self) -> tuple[int, int]:
        """Return the tail source offsets."""

        return self.tail.offset

    def stable_key(self) -> tuple[int, int, int, int, int, int, str]:
        """Return the deterministic edge ordering key."""

        return (
            self.head.start,
            self.head.end,
            _RELATION_RANK[self.relation_type],
            self.tail.start,
            self.tail.end,
            -self.distance,
            self.tail.label,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic offset/type payload without source text."""

        return {
            "head": self.head.to_dict(),
            "tail": self.tail.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "distance": self.distance,
            "token_distance": self.token_distance,
            "sentence_distance": self.sentence_distance,
            "explicit_connective": self.explicit_connective,
        }


@dataclass(frozen=True, slots=True)
class _InputSpan:
    reference: RelationSpan
    section: str | None
    scope_id: str | None
    assertion: tuple[tuple[str, str], ...]


def extract_relation_candidates(
    text: str | Iterable[Any],
    spans: Iterable[Any] | str,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    max_distance: int = DEFAULT_MAX_CHARACTER_DISTANCE,
    max_token_distance: int | None = DEFAULT_MAX_TOKEN_DISTANCE,
    max_character_distance: int | None = None,
    max_tokens: int | None = None,
    allow_cross_sentence: bool = True,
    asserted_only: bool = True,
) -> tuple[RelationCandidate, ...]:
    """Extract deterministic typed candidates from already-located entities.

    ``text`` and ``spans`` follow the repository's existing clinical extractor
    convention.  For small integrations, the inverse ``(spans, text)`` order
    is also accepted.  A span may be an :class:`EntitySpan`, a mapping with
    ``start``/``end`` and a label, or a span-like object exposing those fields.

    Candidates are generated only for compatible head/tail labels, within both
    character and token windows.  Endpoints must share a detected or supplied
    section and compatible assertion scope.  Sentence-local candidates are
    preferred; a single sentence boundary is admitted only when the text at
    that boundary begins or ends with an explicit connective such as ``and``,
    ``with``, ``due to``, or ``in``.  The output is sorted independently of
    input order and contains no source surface text.

    Args:
        text: Source document text, or the span iterable for inverse argument
            order.
        spans: Entity spans, or the source text for inverse argument order.
        sections: Optional section mappings with ``start``, ``end``, and
            ``label``.  When omitted, local deterministic section detection is
            used.
        max_distance: Maximum character gap between non-overlapping spans.
        max_token_distance: Maximum number of intervening tokens. ``None``
            disables this second distance gate.
        max_character_distance: Explicit alias for ``max_distance``.
        max_tokens: Explicit alias for ``max_token_distance``.
        allow_cross_sentence: Whether the explicit-connective exception is
            enabled.  It never admits a bare cross-sentence pair.
        asserted_only: If true, skip endpoints explicitly marked negated,
            refuted, absent, hypothetical, or conditional.

    Returns:
        A stable tuple of :class:`RelationCandidate` records.
    """

    document_text, span_items = _coerce_call_order(text, spans)
    character_limit = (
        max_distance if max_character_distance is None else max_character_distance
    )
    token_limit = max_token_distance if max_tokens is None else max_tokens
    _validate_limits(character_limit, token_limit)

    section_items = _coerce_sections(document_text, sections)
    normalized_spans = _coerce_spans(
        document_text,
        span_items,
        sections=section_items,
    )
    if not normalized_spans:
        return ()

    sentence_offsets = _sentence_offsets(document_text)
    candidates: list[RelationCandidate] = []
    for relation_type in RELATION_TYPES:
        head_labels = RELATION_HEAD_LABELS[relation_type]
        tail_labels = RELATION_TAIL_LABELS[relation_type]
        heads = tuple(
            item for item in normalized_spans if item.reference.label in head_labels
        )
        tails = tuple(
            item for item in normalized_spans if item.reference.label in tail_labels
        )
        for head in heads:
            for tail in tails:
                candidate = _make_candidate(
                    document_text,
                    head,
                    tail,
                    relation_type,
                    sentence_offsets=sentence_offsets,
                    max_distance=character_limit,
                    max_token_distance=token_limit,
                    allow_cross_sentence=allow_cross_sentence,
                    asserted_only=asserted_only,
                )
                if candidate is not None:
                    candidates.append(candidate)

    return tuple(sorted(_deduplicate(candidates), key=RelationCandidate.stable_key))


def extract_relations_lite(
    text: str | Iterable[Any],
    spans: Iterable[Any] | str,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[RelationCandidate, ...]:
    """Alias for :func:`extract_relation_candidates`."""

    return extract_relation_candidates(text, spans, sections, **kwargs)


def extract_lite_relations(
    text: str | Iterable[Any],
    spans: Iterable[Any] | str,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[RelationCandidate, ...]:
    """Compatibility alias for :func:`extract_relation_candidates`."""

    return extract_relation_candidates(text, spans, sections, **kwargs)


def extract_relations(
    text: str | Iterable[Any],
    spans: Iterable[Any] | str,
    sections: Iterable[Any] | Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tuple[RelationCandidate, ...]:
    """Short alias for the lightweight relation candidate extractor."""

    return extract_relation_candidates(text, spans, sections, **kwargs)


def _coerce_call_order(
    text: str | Iterable[Any],
    spans: Iterable[Any] | str,
) -> tuple[str, tuple[Any, ...]]:
    if isinstance(text, str) and not isinstance(spans, str):
        return text, tuple(spans)
    if isinstance(spans, str) and not isinstance(text, str):
        return spans, tuple(text)
    raise TypeError("extract_relation_candidates requires text and iterable spans")


def _validate_limits(max_distance: int, max_token_distance: int | None) -> None:
    if isinstance(max_distance, bool) or max_distance < 0:
        raise ValueError("max_distance must be non-negative")
    if max_token_distance is not None and (
        isinstance(max_token_distance, bool) or max_token_distance < 0
    ):
        raise ValueError("max_token_distance must be non-negative or None")


def _coerce_sections(
    text: str,
    sections: Iterable[Any] | Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any], ...]:
    if sections is None:
        return tuple(detect_sections(text))
    if isinstance(sections, Mapping):
        if _offsets_from_mapping(sections) is not None:
            return (sections,)
        values = tuple(
            value for value in sections.values() if isinstance(value, Mapping)
        )
        return values
    return tuple(item for item in sections if isinstance(item, Mapping))


def _coerce_spans(
    text: str,
    spans: Sequence[Any],
    *,
    sections: Sequence[Mapping[str, Any]],
) -> tuple[_InputSpan, ...]:
    by_key: dict[tuple[int, int, str], _InputSpan] = {}
    for item in spans:
        data = _span_data(item)
        if data is None:
            continue
        offsets = _offsets_from_mapping(data)
        if offsets is None:
            continue
        start, end = offsets
        if start < 0 or end <= start or end > len(text):
            continue
        label = _canonical_label(
            _first_value(
                data, ("label", "entity", "entity_type", "canonical_label", "type")
            )
        )
        if not label:
            continue
        score = _span_score(data)
        section = _section_key(data) or _containing_section(start, end, sections)
        scope_id = _scope_id(data)
        assertion = _assertion_values(data)
        reference = RelationSpan(start=start, end=end, label=label, score=score)
        normalized = _InputSpan(
            reference=reference,
            section=section,
            scope_id=scope_id,
            assertion=assertion,
        )
        key = (start, end, label)
        current = by_key.get(key)
        if current is None or _input_span_key(normalized) < _input_span_key(current):
            by_key[key] = normalized
    return tuple(
        sorted(
            by_key.values(),
            key=lambda item: (
                item.reference.start,
                item.reference.end,
                item.reference.label,
            ),
        )
    )


def _span_data(item: Any) -> Mapping[str, Any] | None:
    if isinstance(item, Mapping):
        return item
    if isinstance(item, EntitySpan):
        return item.to_dict()
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    values = {key: getattr(item, key) for key in _FIELD_KEYS if hasattr(item, key)}
    return values or None


def _offsets_from_mapping(data: Mapping[str, Any]) -> tuple[int, int] | None:
    start = _integer_value(data.get("start", data.get("start_char")))
    end = _integer_value(data.get("end", data.get("end_char")))
    if start is None or end is None:
        raw_offset = data.get("offset", data.get("span"))
        if isinstance(raw_offset, Mapping):
            start = _integer_value(raw_offset.get("start"))
            end = _integer_value(raw_offset.get("end"))
        elif (
            isinstance(raw_offset, Sequence)
            and not isinstance(raw_offset, (str, bytes))
            and len(raw_offset) == 2
        ):
            start = _integer_value(raw_offset[0])
            end = _integer_value(raw_offset[1])
    if start is None or end is None:
        return None
    return start, end


def _integer_value(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _first_value(data: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if value is not None:
            return value
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        for key in keys:
            value = metadata.get(key)
            if value is not None:
                return value
    return None


def _span_score(data: Mapping[str, Any]) -> float:
    raw_score = _first_value(data, ("score", "confidence"))
    if raw_score is None:
        return 1.0
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return 1.0
    if not isfinite(score):
        return 1.0
    return max(0.0, min(1.0, score))


def _canonical_label(raw_label: Any) -> str:
    if not isinstance(raw_label, str) or not raw_label.strip():
        return ""
    normalized = normalize_label(raw_label)
    if normalized != "OTHER":
        return normalized
    key = re.sub(r"[^a-z0-9]", "", raw_label.casefold())
    return _LOCAL_LABEL_ALIASES.get(key, "")


def _section_key(data: Mapping[str, Any]) -> str | None:
    value = _first_value(
        data, ("section_id", "section", "section_label", "section_name")
    )
    return _safe_scope_value(value)


def _containing_section(
    start: int,
    end: int,
    sections: Sequence[Mapping[str, Any]],
) -> str | None:
    for section in sections:
        offsets = _offsets_from_mapping(section)
        label = section.get("label", section.get("name"))
        if offsets is None or label is None:
            continue
        section_start, section_end = offsets
        if section_start <= start and end <= section_end:
            return _safe_scope_value(label)
    return None


def _scope_id(data: Mapping[str, Any]) -> str | None:
    value = _first_value(
        data, ("scope_id", "assertion_scope", "context_id", "sentence_id")
    )
    if isinstance(value, Mapping):
        value = value.get("id", value.get("name", value.get("scope")))
    return _safe_scope_value(value)


def _assertion_values(data: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    containers: list[Mapping[str, Any]] = [data]
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
    for container in tuple(containers):
        for key in ("assertion", "clinical_assertion", "clinical_context", "context"):
            nested = container.get(key)
            nested_data = _span_data(nested)
            if nested_data is not None:
                containers.append(nested_data)

    values: dict[str, str] = {}
    for axis in _ASSERTION_AXES:
        for container in containers:
            raw_value = container.get(axis)
            if raw_value is None and axis == "certainty":
                raw_value = container.get("uncertainty")
            if raw_value is None:
                continue
            normalized = _normalized_value(raw_value)
            if normalized:
                values[axis] = normalized
                break
    for container in containers:
        raw_assertion = container.get("assertion")
        if isinstance(raw_assertion, str) and "status" not in values:
            normalized = _normalized_value(raw_assertion)
            if normalized:
                values["status"] = normalized
                break
    return tuple(sorted(values.items()))


def _normalized_value(value: Any) -> str:
    if isinstance(value, str):
        return re.sub(r"\s+", "_", value.strip().casefold())
    return ""


def _safe_scope_value(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip().casefold()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _input_span_key(
    item: _InputSpan,
) -> tuple[float, str | None, str | None, tuple[tuple[str, str], ...]]:
    return (-item.reference.score, item.section, item.scope_id, item.assertion)


def _make_candidate(
    text: str,
    head: _InputSpan,
    tail: _InputSpan,
    relation_type: RelationType,
    *,
    sentence_offsets: tuple[tuple[int, int], ...],
    max_distance: int,
    max_token_distance: int | None,
    allow_cross_sentence: bool,
    asserted_only: bool,
) -> RelationCandidate | None:
    if head.reference.offset == tail.reference.offset:
        return None
    if _overlap(head.reference, tail.reference):
        return None
    if not _same_section(head.section, tail.section):
        return None
    if head.scope_id is not None and tail.scope_id is not None:
        if head.scope_id != tail.scope_id:
            return None
    if not _same_assertion_scope(head.assertion, tail.assertion):
        return None
    if asserted_only and not _assertion_is_usable(head.assertion, tail.assertion):
        return None

    distance = _character_distance(head.reference, tail.reference)
    if distance > max_distance:
        return None
    token_distance = _token_distance(head.reference, tail.reference, text)
    if max_token_distance is not None and token_distance > max_token_distance:
        return None

    head_sentence = _sentence_index(sentence_offsets, head.reference)
    tail_sentence = _sentence_index(sentence_offsets, tail.reference)
    sentence_distance = abs(head_sentence - tail_sentence)
    explicit_connective = False
    if sentence_distance:
        if not allow_cross_sentence or sentence_distance != 1:
            return None
        explicit_connective = _has_explicit_connective(
            text,
            head.reference,
            tail.reference,
        )
        if not explicit_connective:
            return None
    elif _CLAUSE_BOUNDARY_RE.search(_between(head.reference, tail.reference, text)):
        return None

    return RelationCandidate(
        head=head.reference,
        tail=tail.reference,
        relation_type=relation_type,
        confidence=_confidence(
            head.reference,
            tail.reference,
            distance=distance,
            max_distance=max_distance,
            explicit_connective=explicit_connective,
        ),
        distance=distance,
        token_distance=token_distance,
        sentence_distance=sentence_distance,
        explicit_connective=explicit_connective,
    )


def _same_section(left: str | None, right: str | None) -> bool:
    return left is None or right is None or left == right


def _same_assertion_scope(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
) -> bool:
    left_values = dict(left)
    right_values = dict(right)
    for axis in _ASSERTION_AXES:
        left_value = left_values.get(axis)
        right_value = right_values.get(axis)
        if left_value and right_value and left_value != right_value:
            return False
    return True


def _assertion_is_usable(
    left: tuple[tuple[str, str], ...],
    right: tuple[tuple[str, str], ...],
) -> bool:
    values = dict(left)
    values.update(dict(right))
    return not (
        set(values.values()) & _NEGATIVE_ASSERTION_VALUES
        or set(values.values()) & _NON_ASSERTED_VALUES
    )


def _overlap(left: RelationSpan, right: RelationSpan) -> bool:
    return left.start < right.end and right.start < left.end


def _between(left: RelationSpan, right: RelationSpan, text: str) -> str:
    if left.end <= right.start:
        return text[left.end : right.start]
    if right.end <= left.start:
        return text[right.end : left.start]
    return ""


def _character_distance(left: RelationSpan, right: RelationSpan) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _token_distance(left: RelationSpan, right: RelationSpan, text: str) -> int:
    return len(_TOKEN_RE.findall(_between(left, right, text)))


def _sentence_offsets(text: str) -> tuple[tuple[int, int], ...]:
    if not text:
        return ()
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        start, end = _trim_offsets(text, cursor, boundary.end())
        if start < end:
            offsets.append((start, end))
        cursor = boundary.end()
    start, end = _trim_offsets(text, cursor, len(text))
    if start < end:
        offsets.append((start, end))
    return tuple(offsets) or ((0, len(text)),)


def _trim_offsets(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _sentence_index(
    sentence_offsets: tuple[tuple[int, int], ...],
    span: RelationSpan,
) -> int:
    for index, (start, end) in enumerate(sentence_offsets):
        if start <= span.start and span.end <= end:
            return index
    return min(
        range(len(sentence_offsets)),
        key=lambda index: min(
            abs(span.start - sentence_offsets[index][0]),
            abs(span.start - sentence_offsets[index][1]),
        ),
    )


def _has_explicit_connective(
    text: str,
    left: RelationSpan,
    right: RelationSpan,
) -> bool:
    between = _between(left, right, text)
    boundaries = tuple(_SENTENCE_BOUNDARY_RE.finditer(between))
    if len(boundaries) != 1:
        return False
    boundary = boundaries[0]
    before = between[: boundary.start()].rstrip()
    after = between[boundary.end() :].lstrip()
    return bool(
        _CONNECTIVE_AFTER_RE.match(after) or _CONNECTIVE_BEFORE_RE.search(before)
    )


def _confidence(
    head: RelationSpan,
    tail: RelationSpan,
    *,
    distance: int,
    max_distance: int,
    explicit_connective: bool,
) -> float:
    span_score = (head.score + tail.score) / 2.0
    proximity = (
        1.0 if distance == 0 else max(0.0, 1.0 - (distance / max(1, max_distance)))
    )
    value = 0.55 + (0.25 * span_score) + (0.2 * proximity)
    if explicit_connective:
        value -= 0.05
    return round(max(0.0, min(1.0, value)), 6)


def _deduplicate(
    candidates: Iterable[RelationCandidate],
) -> tuple[RelationCandidate, ...]:
    by_key: dict[
        tuple[RelationType, tuple[int, int], tuple[int, int]], RelationCandidate
    ] = {}
    for candidate in candidates:
        key = (candidate.relation_type, candidate.head.offset, candidate.tail.offset)
        current = by_key.get(key)
        if current is None or candidate.confidence > current.confidence:
            by_key[key] = candidate
    return tuple(by_key.values())


__all__ = [
    "DEFAULT_MAX_CHARACTER_DISTANCE",
    "DEFAULT_MAX_TOKEN_DISTANCE",
    "DRUG_DOSE",
    "DRUG_ROUTE",
    "FINDING_SEVERITY",
    "PROBLEM_ANATOMY",
    "RELATION_HEAD_LABELS",
    "RELATION_SCHEMA_VERSION",
    "RELATION_TAIL_LABELS",
    "RELATION_TYPES",
    "RelationCandidate",
    "RelationSpan",
    "RelationType",
    "extract_lite_relations",
    "extract_relation_candidates",
    "extract_relations",
    "extract_relations_lite",
]
