"""Deterministic family-history condition-to-relative relation extraction."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.context import (
    CERTAIN,
    CERTAINTY_VALUES,
    FAMILY_EXPERIENCER,
    PATIENT_EXPERIENCER,
    UNCERTAIN,
    canonical_section_name,
    resolve_uncertainty,
)
from openmed.clinical.experiencer import OTHER_EXPERIENCER
from openmed.clinical.sections import detect_sections
from openmed.core.labels import CONDITION, DISEASE, FINDING, PROBLEM, normalize_label
from openmed.processing.advanced_ner import EntitySpan

from .candidate import SpanReference

FAMILY_HISTORY_RELATION_ADVISORY = (
    "Family-history relations are deterministic assistive output for clinician "
    "review, not a diagnosis or a clinical decision."
)

RELATION_TO_PATIENT = "RELATION_TO_PATIENT"
FAMILY_HISTORY_RELATION_TYPE = "condition_to_relative"

_CONDITION_LABELS = frozenset({CONDITION, DISEASE, FINDING, PROBLEM})
_RELATIVE_LABELS = frozenset(
    {
        "FAMILY_MEMBER",
        "FAMILY_HISTORY_RELATIVE",
        "RELATION_TO_PATIENT",
        "RELATIVE",
    }
)
_MAX_CHARACTER_DISTANCE = 96
_MAX_TOKEN_DISTANCE = 12
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.!?;,\n\r，；。！？]|(?<!\w)(?:but|however|whereas)(?!\w)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"\b\w+(?:[-/]\w+)*\b", re.UNICODE)

_RELATIVE_SURFACES: tuple[str, ...] = (
    "maternal grandmother",
    "paternal grandmother",
    "maternal grandfather",
    "paternal grandfather",
    "maternal grandparent",
    "paternal grandparent",
    "grandmother",
    "grandfather",
    "grandparent",
    "family medical history",
    "family history",
    "mother",
    "father",
    "sister",
    "brother",
    "sibling",
    "parent",
    "daughter",
    "son",
    "child",
    "aunt",
    "uncle",
    "cousin",
    "niece",
    "nephew",
    "wife",
    "husband",
    "spouse",
    "relative",
    "familial",
    "fhx",
    "mom",
    "mum",
    "dad",
)
_RELATIVE_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(surface)
        for surface in sorted(_RELATIVE_SURFACES, key=lambda item: (-len(item), item))
    )
    + r")(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FamilyHistoryRelation:
    """One condition-to-relative relation grounded in source spans.

    ``relative`` and ``condition`` retain source offsets for downstream
    interoperability. ``certainty`` is the shipped ConText uncertainty axis;
    the relation does not promote an uncertain family history to a patient
    problem-list condition.
    """

    relative: SpanReference
    condition: SpanReference
    certainty: str
    score: float
    advisory: str = FAMILY_HISTORY_RELATION_ADVISORY

    def __post_init__(self) -> None:
        if self.certainty not in CERTAINTY_VALUES:
            raise ValueError(
                "family-history relation certainty must be one of "
                f"{', '.join(CERTAINTY_VALUES)}"
            )
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("family-history relation score must be between 0 and 1")
        object.__setattr__(self, "score", score)

    @property
    def head(self) -> SpanReference:
        """Return the CONDITION endpoint of the directed relation."""

        return self.condition

    @property
    def tail(self) -> SpanReference:
        """Return the RELATION_TO_PATIENT endpoint of the relation."""

        return self.relative

    @property
    def type(self) -> str:
        """Return the concise relation type used by clinical relation APIs."""

        return "family_history"

    @property
    def relation_type(self) -> str:
        """Return the directed condition-to-relative relation type."""

        return FAMILY_HISTORY_RELATION_TYPE

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible relation mapping."""

        return {
            "relative": self.relative.to_dict(),
            "condition": self.condition.to_dict(),
            "certainty": self.certainty,
            "score": self.score,
            "advisory": self.advisory,
        }


@dataclass(frozen=True)
class _Section:
    start: int
    end: int
    label: str | None
    header_start: int | None = None
    header_end: int | None = None
    content_start: int | None = None


@dataclass(frozen=True)
class _InputSpan:
    reference: SpanReference
    data: Mapping[str, Any]
    section: str | None
    explicit_experiencer: str | None
    explicit_certainty: str | None


@dataclass(frozen=True)
class _RelativeCandidate:
    reference: SpanReference
    section: str | None
    explicit: bool = False


def extract_family_history_relations(
    text: str,
    spans: Iterable[EntitySpan | SpanReference | Mapping[str, Any] | Any],
    sections: Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | None = None,
) -> tuple[FamilyHistoryRelation, ...]:
    """Bind family-member mentions to their nearest condition spans.

    Existing NER spans are the only condition inputs. Family-member mentions
    are discovered from the local source text (or accepted from an explicit
    relation span), and candidates cannot cross a clause or section boundary.
    Explicit ``patient`` experiencer metadata suppresses a relation. The
    current OM-112 ``family`` value and its non-patient ``other`` value are
    accepted when a family surface cue supplies the relative endpoint; absent
    metadata, the surface cue is sufficient for graceful degradation.

    Args:
        text: Source clinical text indexed by every supplied span.
        spans: Condition spans and optional relation-to-patient spans. Span
            mappings may carry ConText axes at the top level or under the
            shipped clinical-context metadata containers.
        sections: Optional precomputed sections. When omitted, deterministic
            ``detect_sections`` output is used to keep family-history and
            patient sections separate.

    Returns:
        Deterministically ordered family-history relations. The output is
        assistive and is never added to the patient's own problem list.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    section_items = _coerce_sections(text, sections)
    input_spans = _coerce_spans(text, spans, section_items)
    conditions = tuple(
        item
        for item in input_spans
        if normalize_label(item.reference.label) in _CONDITION_LABELS
    )
    if not conditions:
        return ()

    relatives = _relative_candidates(text, input_spans, section_items)
    if not relatives:
        return ()

    relations: list[FamilyHistoryRelation] = []
    for condition in conditions:
        candidates = tuple(
            candidate
            for candidate in relatives
            if _candidate_is_in_scope(
                text,
                condition,
                candidate,
                sections=section_items,
            )
        )
        if not candidates or not _condition_is_eligible(condition):
            continue

        relative = min(
            candidates,
            key=lambda candidate: _candidate_sort_key(text, condition, candidate),
        )
        certainty = _condition_certainty(text, condition, section_items)
        relations.append(
            FamilyHistoryRelation(
                relative=relative.reference,
                condition=condition.reference,
                certainty=certainty,
                score=_relation_score(text, condition.reference, relative.reference),
            )
        )

    unique: dict[tuple[tuple[int, int], tuple[int, int]], FamilyHistoryRelation] = {}
    for relation in relations:
        unique.setdefault(
            (relation.relative.offset_key(), relation.condition.offset_key()), relation
        )
    return tuple(
        sorted(
            unique.values(),
            key=lambda relation: (
                relation.condition.start,
                relation.condition.end,
                relation.relative.start,
                relation.relative.end,
            ),
        )
    )


def _coerce_spans(
    text: str,
    spans: Iterable[EntitySpan | SpanReference | Mapping[str, Any] | Any],
    sections: Sequence[_Section],
) -> tuple[_InputSpan, ...]:
    items = _iter_items(spans)
    normalized: dict[tuple[int, int, str], _InputSpan] = {}
    for item in items:
        data = _span_mapping(item)
        if data is None:
            continue
        try:
            start = int(data.get("start", data.get("start_char", -1)))
            end = int(data.get("end", data.get("end_char", -1)))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or end > len(text):
            continue

        label = _raw_label(data)
        if not label:
            continue
        score = _bounded_score(data)
        explicit_section = _section_value(data)
        section = explicit_section or _section_for_span(start, end, sections)
        reference = SpanReference(
            text=text[start:end],
            label=label,
            start=start,
            end=end,
            score=score,
            section=section,
        )
        input_span = _InputSpan(
            reference=reference,
            data=data,
            section=section,
            explicit_experiencer=_context_experiencer(data),
            explicit_certainty=_context_certainty(data),
        )
        key = (start, end, normalize_label(label))
        previous = normalized.get(key)
        if previous is None or _context_richness(input_span) > _context_richness(
            previous
        ):
            normalized[key] = input_span

    return tuple(
        sorted(
            normalized.values(),
            key=lambda item: (
                item.reference.start,
                item.reference.end,
                normalize_label(item.reference.label),
            ),
        )
    )


def _iter_items(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, Mapping)) or _looks_like_span(value):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _looks_like_span(value: Any) -> bool:
    return all(hasattr(value, field) for field in ("start", "end")) or hasattr(
        value, "span"
    )


def _span_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        data = dict(value)
    else:
        nested = getattr(value, "span", None)
        if isinstance(nested, Mapping):
            data = dict(nested)
        else:
            to_dict = getattr(value, "to_dict", None)
            serialized = to_dict() if callable(to_dict) else None
            data = dict(serialized) if isinstance(serialized, Mapping) else {}

        for field in (
            "text",
            "label",
            "entity",
            "canonical_label",
            "entity_type",
            "start",
            "end",
            "section",
            "metadata",
            "experiencer",
            "certainty",
            "uncertainty",
            "assertion",
            "clinical_assertion",
            "clinical_context",
            "context",
            "assignment",
        ):
            if field not in data and hasattr(value, field):
                data[field] = getattr(value, field)

    nested_span = data.get("span")
    if ("start" not in data or "end" not in data) and isinstance(nested_span, Mapping):
        merged = dict(nested_span)
        merged.update({key: item for key, item in data.items() if key != "span"})
        data = merged
    return data or None


def _raw_label(data: Mapping[str, Any]) -> str:
    for key in ("label", "entity", "canonical_label", "entity_type", "type"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _bounded_score(data: Mapping[str, Any]) -> float:
    metadata = data.get("metadata")
    metadata_score = (
        metadata.get("confidence") if isinstance(metadata, Mapping) else None
    )
    raw = data.get("score", data.get("confidence", metadata_score))
    try:
        score = float(1.0 if raw is None else raw)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(score):
        return 0.0
    return min(1.0, max(0.0, score))


def _context_containers(data: Mapping[str, Any]) -> tuple[Any, ...]:
    containers: list[Any] = [data]
    index = 0
    while index < len(containers):
        container = containers[index]
        index += 1
        for key in (
            "metadata",
            "clinical_context",
            "clinical_assertion",
            "assertion",
            "context",
            "assignment",
        ):
            nested = _field(container, key)
            if nested is not None and not any(
                nested is existing for existing in containers
            ):
                containers.append(nested)
    return tuple(containers)


def _context_value(data: Mapping[str, Any], *keys: str) -> Any:
    for container in _context_containers(data):
        value = _field(container, *keys)
        if value is not None:
            return value
    return None


def _context_experiencer(data: Mapping[str, Any]) -> str | None:
    raw = _context_value(data, "experiencer")
    if raw is None:
        return None
    nested = _field(raw, "experiencer")
    if nested is not None and nested is not raw:
        raw = nested
    compact = _compact(str(raw))
    if compact in {"patient", "self", "subject", "currentpatient", "pt"}:
        return PATIENT_EXPERIENCER
    if compact in {
        "family",
        "familymember",
        "familyhistory",
        "familial",
        "relative",
        "mother",
        "father",
        "sibling",
        "grandparent",
    }:
        return FAMILY_EXPERIENCER
    if compact in {"other", "nonpatient"}:
        return OTHER_EXPERIENCER
    return compact or None


def _context_certainty(data: Mapping[str, Any]) -> str | None:
    raw = _context_value(data, "certainty", "uncertainty")
    if raw is None:
        return None
    if isinstance(raw, bool):
        return UNCERTAIN if raw else CERTAIN
    compact = _compact(str(raw))
    if compact in {"uncertain", "uncertainty", "possible", "probable", "hedged"}:
        return UNCERTAIN
    if compact in {"certain", "confirmed", "affirmed"}:
        return CERTAIN
    return None


def _context_richness(item: _InputSpan) -> int:
    return int(item.explicit_experiencer is not None) + int(
        item.explicit_certainty is not None
    )


def _field(value: Any, *names: str) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for name in names:
            if value.get(name) is not None:
                return value[name]
        return None
    for name in names:
        result = getattr(value, name, None)
        if result is not None:
            return result
    return None


def _coerce_sections(
    text: str,
    sections: Iterable[Mapping[str, Any]] | Mapping[str, Any] | str | None,
) -> tuple[_Section, ...]:
    if sections is None:
        raw_sections: tuple[Any, ...] = tuple(detect_sections(text))
    elif isinstance(sections, str):
        return (_Section(0, len(text), _canonical_section(sections)),)
    elif isinstance(sections, Mapping):
        raw_sections = (sections,)
    else:
        raw_sections = tuple(sections)

    result: list[_Section] = []
    for item in raw_sections:
        start = _integer_field(item, "start", "start_char")
        end = _integer_field(item, "end", "end_char")
        label = _canonical_section(_field(item, "label", "section", "name"))
        if start is None or end is None:
            if label is not None:
                result.append(_Section(0, len(text), label))
            continue
        if start < 0 or end <= start or end > len(text):
            continue
        result.append(
            _Section(
                start=start,
                end=end,
                label=label,
                header_start=_integer_field(item, "header_start"),
                header_end=_integer_field(item, "header_end"),
                content_start=_integer_field(item, "content_start"),
            )
        )
    return tuple(
        sorted(result, key=lambda item: (item.start, item.end, item.label or ""))
    )


def _integer_field(value: Any, *names: str) -> int | None:
    raw = _field(value, *names)
    if isinstance(raw, bool):
        return None
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _canonical_section(value: Any) -> str | None:
    if value is None:
        return None
    canonical = canonical_section_name(value)
    if canonical is not None:
        return canonical
    raw = str(value).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", raw).strip("_") or None


def _section_value(data: Mapping[str, Any]) -> str | None:
    value = _field(data, "section", "section_label", "section_name")
    if value is None:
        metadata = data.get("metadata")
        value = _field(metadata, "section", "section_label", "section_name")
    return _canonical_section(value)


def _section_for_span(start: int, end: int, sections: Sequence[_Section]) -> str | None:
    containing = [
        section for section in sections if section.start <= start and end <= section.end
    ]
    if not containing:
        return None
    return min(
        containing, key=lambda section: (section.end - section.start, section.start)
    ).label


def _relative_candidates(
    text: str,
    spans: Sequence[_InputSpan],
    sections: Sequence[_Section],
) -> tuple[_RelativeCandidate, ...]:
    candidates: dict[tuple[int, int], _RelativeCandidate] = {}
    for item in spans:
        label = _normalized_raw_label(item.reference.label)
        if label not in _RELATIVE_LABELS:
            continue
        candidate = _RelativeCandidate(
            reference=SpanReference(
                text=item.reference.text,
                label=RELATION_TO_PATIENT,
                start=item.reference.start,
                end=item.reference.end,
                score=item.reference.score,
                section=item.section,
            ),
            section=item.section,
            explicit=True,
        )
        candidates[item.reference.offset_key()] = candidate

    for match in _RELATIVE_RE.finditer(text):
        key = match.start(), match.end()
        if key in candidates:
            continue
        candidates[key] = _RelativeCandidate(
            reference=SpanReference(
                text=text[match.start() : match.end()],
                label=RELATION_TO_PATIENT,
                start=match.start(),
                end=match.end(),
                score=1.0,
                section=_section_for_span(match.start(), match.end(), sections),
            ),
            section=_section_for_span(match.start(), match.end(), sections),
        )
    return tuple(
        sorted(
            candidates.values(),
            key=lambda candidate: (
                candidate.reference.start,
                candidate.reference.end,
                candidate.reference.text.casefold(),
            ),
        )
    )


def _normalized_raw_label(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", value.upper()).strip("_")


def _candidate_is_in_scope(
    text: str,
    condition: _InputSpan,
    candidate: _RelativeCandidate,
    *,
    sections: Sequence[_Section],
) -> bool:
    relative = candidate.reference
    if (
        condition.section is not None
        and candidate.section is not None
        and condition.section.casefold() != candidate.section.casefold()
    ):
        return False
    distance = _character_distance(condition.reference, relative)
    if distance > _MAX_CHARACTER_DISTANCE:
        return False
    if _token_distance(condition.reference, relative, text) > _MAX_TOKEN_DISTANCE:
        return False
    between = _text_between(condition.reference, relative, text)
    if _CLAUSE_BOUNDARY_RE.search(
        between
    ) is not None and not _is_family_header_candidate(
        relative,
        condition,
        sections,
    ):
        return False
    return True


def _is_family_header_candidate(
    relative: SpanReference,
    condition: _InputSpan,
    sections: Sequence[_Section],
) -> bool:
    if condition.section != "family_history":
        return False
    if _compact(relative.text) not in {"familyhistory", "familymedicalhistory", "fhx"}:
        return False
    return any(
        section.label == "family_history"
        and section.header_start is not None
        and section.content_start is not None
        and section.header_start <= relative.start
        and relative.end <= section.content_start
        for section in sections
    )


def _condition_is_eligible(condition: _InputSpan) -> bool:
    experiencer = condition.explicit_experiencer
    if experiencer == PATIENT_EXPERIENCER:
        return False
    if experiencer is None:
        return True
    return experiencer in {FAMILY_EXPERIENCER, OTHER_EXPERIENCER}


def _condition_certainty(
    text: str,
    condition: _InputSpan,
    sections: Sequence[_Section],
) -> str:
    if condition.explicit_certainty is not None:
        return condition.explicit_certainty
    scoped_text = _mask_outside_condition_section(text, condition, sections)
    return resolve_uncertainty(
        {
            "text": text[condition.reference.start : condition.reference.end],
            "document_text": scoped_text,
            "start": condition.reference.start,
            "end": condition.reference.end,
        }
    )


def _mask_outside_condition_section(
    text: str,
    condition: _InputSpan,
    sections: Sequence[_Section],
) -> str:
    containing = [
        section
        for section in sections
        if section.start <= condition.reference.start
        and condition.reference.end <= section.end
    ]
    if not containing:
        return text
    section = min(containing, key=lambda item: item.end - item.start)
    return (
        " " * section.start
        + text[section.start : section.end]
        + " " * (len(text) - section.end)
    )


def _candidate_sort_key(
    text: str,
    condition: _InputSpan,
    candidate: _RelativeCandidate,
) -> tuple[int, int, int, int, int]:
    relative = candidate.reference
    return (
        _character_distance(condition.reference, relative),
        _token_distance(condition.reference, relative, text),
        0 if relative.end <= condition.reference.start else 1,
        relative.start,
        relative.end,
    )


def _relation_score(
    text: str,
    condition: SpanReference,
    relative: SpanReference,
) -> float:
    confidence = min(1.0, max(0.0, (condition.score + relative.score) / 2.0))
    distance = _character_distance(condition, relative)
    proximity = max(0.0, 1.0 - distance / (_MAX_CHARACTER_DISTANCE + 1))
    token_proximity = 1.0 / (1.0 + _token_distance(condition, relative, text))
    score = 0.5 * confidence + 0.3 * proximity + 0.2 * token_proximity
    return round(min(1.0, max(0.0, score)), 6)


def _character_distance(left: SpanReference, right: SpanReference) -> int:
    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def _token_distance(left: SpanReference, right: SpanReference, text: str) -> int:
    return len(_TOKEN_RE.findall(_text_between(left, right, text)))


def _text_between(left: SpanReference, right: SpanReference, text: str) -> str:
    if left.end <= right.start:
        return text[left.end : right.start]
    if right.end <= left.start:
        return text[right.end : left.start]
    return ""


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


__all__ = [
    "FAMILY_HISTORY_RELATION_ADVISORY",
    "FAMILY_HISTORY_RELATION_TYPE",
    "FamilyHistoryRelation",
    "RELATION_TO_PATIENT",
    "extract_family_history_relations",
]
