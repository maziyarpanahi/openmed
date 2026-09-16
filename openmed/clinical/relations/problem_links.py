"""Deterministic problem-to-attribute relation extraction."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.sections import detect_sections, validate_section_spans
from openmed.core.labels import BODY_SITE, PROBLEM, SEVERITY, normalize_label
from openmed.processing.advanced_ner import EntitySpan

from .candidate import ProblemAttributeType, Relation, SpanReference

PROBLEM_RELATION_ADVISORY = (
    "Problem attribute relations are deterministic assistive output for clinician "
    "review, not an automated diagnosis or clinical decision."
)

PROBLEM_STATUS_CUES = ("active", "resolved", "chronic")

_STATUS_LABELS = frozenset({"STATUS", "CLINICAL_STATUS", "PROBLEM_STATUS"})
_STATUS_CUE_RE = re.compile(
    r"\b(?:" + "|".join(PROBLEM_STATUS_CUES) + r")\b",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;\n]")
_TOKEN_RE = re.compile(r"\b\w+(?:[-/]\w+)*\b")
_MAX_CHARACTER_DISTANCE = 64
_MAX_TOKEN_DISTANCE = 6
_RELATION_ORDER: dict[ProblemAttributeType, int] = {
    "severity": 0,
    "body_site": 1,
    "status": 2,
}


@dataclass(frozen=True)
class _InputSpan:
    reference: SpanReference
    context_status: str | None = None


def extract_problem_relations(
    text: str,
    spans: Iterable[EntitySpan | Mapping[str, Any]],
    sections: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[Relation, ...]:
    """Bind problem spans to severity, body-site, and status attributes.

    Attributes are assigned to the nearest problem in the same clause and
    known section, within a conservative six-token window. Status is derived
    from the explicit ``active``, ``resolved``, and ``chronic`` cue vocabulary.
    When a problem span already carries OM-041 clinical context, negated or
    historical context overrides a lexical status cue. The function does not
    compute assertion or temporality axes itself.

    Args:
        text: Original clinical text indexed by the supplied span offsets.
        spans: Existing problem, severity, and body-site spans. Mapping inputs
            may also carry ``negation``/``temporality`` fields directly or
            under ``metadata["clinical_context"]``.
        sections: Optional precomputed contiguous section spans. When omitted,
            sections are detected locally and deterministically.

    Returns:
        Deterministically ordered ``Relation`` records. Context-derived status
        tails carry ``derived=True`` and use their problem offsets as source
        provenance. This assistive output is not an automated diagnosis or a
        substitute for clinician review.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    span_items = tuple(spans)
    section_items = tuple(detect_sections(text) if sections is None else sections)
    section_by_offset = _section_labels_by_offset(text, section_items)
    input_spans = _coerce_spans(
        text,
        span_items,
        sections=section_items,
    )
    problems = tuple(
        item for item in input_spans if _canonical_label(item.reference) == PROBLEM
    )
    if not problems:
        return ()

    relations: list[Relation] = []
    for item in input_spans:
        attribute_type = _problem_attribute_type(item.reference)
        if attribute_type not in {"severity", "body_site"}:
            continue
        head = _nearest_problem(item.reference, problems, text=text)
        if head is not None:
            relations.append(
                Relation(
                    head=head.reference,
                    type=attribute_type,
                    tail=item.reference,
                    score=_relation_score(head.reference, item.reference, text=text),
                )
            )

    status_tails = _status_tails(
        text,
        input_spans,
        section_by_offset=section_by_offset,
    )
    lexical_status_by_head: dict[tuple[int, int], SpanReference] = {}
    for tail in status_tails:
        head = _nearest_problem(tail, problems, text=text)
        if head is None:
            continue
        key = head.reference.offset_key()
        current = lexical_status_by_head.get(key)
        if current is None or _status_candidate_key(
            head.reference, tail, text=text
        ) < _status_candidate_key(head.reference, current, text=text):
            lexical_status_by_head[key] = tail

    for problem in problems:
        head = problem.reference
        if problem.context_status is not None:
            tail = _derived_status_tail(head, problem.context_status)
            score = _context_relation_score(head)
        else:
            tail = lexical_status_by_head.get(head.offset_key())
            if tail is None:
                continue
            score = _relation_score(head, tail, text=text)
        relations.append(
            Relation(
                head=head,
                type="status",
                tail=tail,
                score=score,
            )
        )

    return tuple(sorted(relations, key=_relation_sort_key))


def _coerce_spans(
    text: str,
    spans: Sequence[EntitySpan | Mapping[str, Any]],
    *,
    sections: Sequence[Mapping[str, Any]],
) -> tuple[_InputSpan, ...]:
    normalized: dict[tuple[int, int, str], _InputSpan] = {}
    for item in spans:
        data = _span_mapping(item)
        if data is None:
            continue
        try:
            start = int(data.get("start", data.get("start_char", -1)))
            end = int(data.get("end", data.get("end_char", -1)))
            raw_score = data.get("score", 1.0)
            score = 1.0 if raw_score is None else float(raw_score)
        except (TypeError, ValueError):
            continue
        label = _span_label(data)
        if not label or start < 0 or end <= start or end > len(text):
            continue
        raw_section = data.get("section")
        section = (
            str(raw_section)
            if raw_section is not None
            else _span_section(start, end, sections)
        )
        reference = SpanReference(
            text=text[start:end],
            label=label,
            start=start,
            end=end,
            score=max(0.0, min(score, 1.0)),
            section=section,
        )
        key = (start, end, _canonical_label(reference))
        normalized[key] = _InputSpan(
            reference=reference,
            context_status=_context_status(data),
        )
    return tuple(
        sorted(
            normalized.values(),
            key=lambda item: (
                item.reference.start,
                item.reference.end,
                _canonical_label(item.reference),
            ),
        )
    )


def _span_mapping(item: EntitySpan | Mapping[str, Any]) -> Mapping[str, Any] | None:
    if isinstance(item, Mapping):
        return item
    if isinstance(item, EntitySpan):
        return item.to_dict()
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    return None


def _span_label(data: Mapping[str, Any]) -> str:
    for key in ("label", "entity", "canonical_label", "entity_type"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _context_status(data: Mapping[str, Any]) -> str | None:
    containers: list[Any] = [data]
    context_keys = ("clinical_context", "clinical_assertion", "assertion", "context")
    for key in context_keys:
        nested = data.get(key)
        if _is_context_container(nested):
            containers.append(nested)
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
        for key in context_keys:
            nested = metadata.get(key)
            if _is_context_container(nested):
                containers.append(nested)

    negations = {
        _normalized_context_value(_context_value(container, "negation"))
        for container in containers
    }
    if negations & {"negated", "refuted"}:
        return "negated"

    temporalities = {
        _normalized_context_value(_context_value(container, "temporality"))
        for container in containers
    }
    if "historical" in temporalities:
        return "historical"
    return None


def _normalized_context_value(value: Any) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _is_context_container(value: Any) -> bool:
    return isinstance(value, Mapping) or any(
        hasattr(value, field) for field in ("negation", "temporality")
    )


def _context_value(container: Any, field: str) -> Any:
    if isinstance(container, Mapping):
        return container.get(field)
    return getattr(container, field, None)


def _status_tails(
    text: str,
    spans: Sequence[_InputSpan],
    *,
    section_by_offset: Mapping[tuple[int, int], str],
) -> tuple[SpanReference, ...]:
    status_by_offset: dict[tuple[int, int], SpanReference] = {}
    for item in spans:
        if _problem_attribute_type(item.reference) == "status":
            status_by_offset[item.reference.offset_key()] = item.reference
    for match in _STATUS_CUE_RE.finditer(text):
        offset = (match.start(), match.end())
        status_by_offset.setdefault(
            offset,
            SpanReference(
                text=text[match.start() : match.end()],
                label="STATUS",
                start=match.start(),
                end=match.end(),
                score=1.0,
                section=section_by_offset.get(offset),
            ),
        )
    return tuple(
        status_by_offset[offset]
        for offset in sorted(status_by_offset, key=lambda value: (value[0], value[1]))
    )


def _canonical_label(span: SpanReference) -> str:
    return normalize_label(span.label)


def _problem_attribute_type(span: SpanReference) -> ProblemAttributeType | None:
    canonical = _canonical_label(span)
    if canonical == SEVERITY:
        return "severity"
    if canonical == BODY_SITE:
        return "body_site"
    normalized_label = re.sub(r"[^A-Z0-9]+", "_", span.label.upper()).strip("_")
    if normalized_label in _STATUS_LABELS:
        return "status"
    return None


def _nearest_problem(
    tail: SpanReference,
    problems: Sequence[_InputSpan],
    *,
    text: str,
) -> _InputSpan | None:
    candidates = [
        problem
        for problem in problems
        if _candidate_is_in_scope(problem.reference, tail, text=text)
    ]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda problem: (
            _character_distance(problem.reference, tail),
            _token_distance(problem.reference, tail, text=text),
            problem.reference.start,
            problem.reference.end,
        ),
    )


def _candidate_is_in_scope(
    head: SpanReference,
    tail: SpanReference,
    *,
    text: str,
) -> bool:
    if (
        head.section is not None
        and tail.section is not None
        and head.section.casefold() != tail.section.casefold()
    ):
        return False
    between = _text_between(head, tail, text=text)
    return (
        _CLAUSE_BOUNDARY_RE.search(between) is None
        and _character_distance(head, tail) <= _MAX_CHARACTER_DISTANCE
        and _token_distance(head, tail, text=text) <= _MAX_TOKEN_DISTANCE
    )


def _text_between(
    left: SpanReference,
    right: SpanReference,
    *,
    text: str,
) -> str:
    if left.end <= right.start:
        return text[left.end : right.start]
    if right.end <= left.start:
        return text[right.end : left.start]
    return ""


def _character_distance(left: SpanReference, right: SpanReference) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _token_distance(
    left: SpanReference,
    right: SpanReference,
    *,
    text: str,
) -> int:
    return len(_TOKEN_RE.findall(_text_between(left, right, text=text)))


def _relation_score(
    head: SpanReference,
    tail: SpanReference,
    *,
    text: str,
) -> float:
    entity_confidence = (head.score + tail.score) / 2.0
    proximity = 1.0 / (1.0 + _token_distance(head, tail, text=text))
    return round(min(1.0, 0.7 + 0.15 * entity_confidence + 0.15 * proximity), 6)


def _context_relation_score(head: SpanReference) -> float:
    return round(min(1.0, 0.8 + 0.2 * head.score), 6)


def _derived_status_tail(head: SpanReference, status: str) -> SpanReference:
    return SpanReference(
        text=status,
        label="STATUS",
        start=head.start,
        end=head.end,
        score=head.score,
        section=head.section,
        derived=True,
    )


def _status_candidate_key(
    head: SpanReference,
    tail: SpanReference,
    *,
    text: str,
) -> tuple[int, int, int, int, str]:
    return (
        _character_distance(head, tail),
        _token_distance(head, tail, text=text),
        tail.start,
        tail.end,
        tail.text.casefold(),
    )


def _relation_sort_key(
    relation: Relation,
) -> tuple[int, int, int, int, int, str]:
    return (
        relation.head.start,
        relation.head.end,
        _RELATION_ORDER[relation.type],
        relation.tail.start,
        relation.tail.end,
        relation.tail.text.casefold(),
    )


def _section_labels_by_offset(
    text: str,
    sections: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], str]:
    if not sections:
        return {}
    validate_section_spans(text, sections)
    labels: dict[tuple[int, int], str] = {}
    for section in sections:
        start = int(section["start"])
        end = int(section["end"])
        label = str(section["label"])
        for match in _STATUS_CUE_RE.finditer(text, start, end):
            labels[(match.start(), match.end())] = label
    return labels


def _span_section(
    start: int,
    end: int,
    sections: Sequence[Mapping[str, Any]],
) -> str | None:
    containing = next(
        (
            section
            for section in sections
            if int(section["start"]) <= start and end <= int(section["end"])
        ),
        None,
    )
    return str(containing["label"]) if containing is not None else None


__all__ = [
    "PROBLEM_RELATION_ADVISORY",
    "PROBLEM_STATUS_CUES",
    "extract_problem_relations",
]
