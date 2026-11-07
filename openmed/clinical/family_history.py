"""Deterministic family-history condition attribution.

Family-history extraction consumes condition spans and the already-resolved
experiencer axis.  It deliberately does not build a patient condition or
perform genetic-risk inference.  The output keeps only normalized relative
roles, optional age/status values, and source offsets; source text is never
copied into a record.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from openmed.clinical.context import (
    FAMILY_EXPERIENCER,
    PATIENT_EXPERIENCER,
    canonical_section_name,
)
from openmed.clinical.experiencer import resolve_experiencer
from openmed.clinical.sections import detect_sections
from openmed.core.labels import (
    AGE,
    CONDITION,
    DISEASE,
    FINDING,
    PROBLEM,
    normalize_label,
)

FAMILY_HISTORY_ADVISORY = (
    "Family-history attribution is deterministic assistive output for clinician "
    "review, not a diagnosis, pedigree, or genetic-risk decision."
)

_CONDITION_LABELS = frozenset({CONDITION, DISEASE, FINDING, PROBLEM})
_MAX_ATTRIBUTE_DISTANCE = 96
_MAX_STATUS_DISTANCE = 128
_CLAUSE_BOUNDARY_RE = re.compile(r"[.!?;\n]")
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")

_RELATIVE_ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("maternal grandmother", "maternal_grandmother"),
    ("paternal grandmother", "paternal_grandmother"),
    ("maternal grandfather", "maternal_grandfather"),
    ("paternal grandfather", "paternal_grandfather"),
    ("grandmother", "grandmother"),
    ("grandfather", "grandfather"),
    ("grandparent", "grandparent"),
    ("mother", "mother"),
    ("mom", "mother"),
    ("mum", "mother"),
    ("father", "father"),
    ("dad", "father"),
    ("parent", "parent"),
    ("sister", "sister"),
    ("brother", "brother"),
    ("sibling", "sibling"),
    ("daughter", "daughter"),
    ("son", "son"),
    ("child", "child"),
    ("aunt", "aunt"),
    ("uncle", "uncle"),
    ("cousin", "cousin"),
    ("niece", "niece"),
    ("nephew", "nephew"),
    ("wife", "spouse"),
    ("husband", "spouse"),
    ("spouse", "spouse"),
    ("maternal", "maternal"),
    ("paternal", "paternal"),
    ("family history", "family"),
    ("family medical history", "family"),
    ("fhx", "family"),
    ("familial", "family"),
    ("relative", "relative"),
)

_RELATIVE_ROLE_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(surface)
        for surface, _role in sorted(
            _RELATIVE_ROLE_PATTERNS,
            key=lambda item: (-len(item[0]), item[0]),
        )
    )
    + r")(?!\w)",
    re.IGNORECASE,
)
_CANONICAL_RELATIVE_ROLES = frozenset(
    role for _surface, role in _RELATIVE_ROLE_PATTERNS
)

_AGE_RE = re.compile(
    r"(?<!\w)(?:"
    r"at|aged|age(?:\s+of\s+onset)?|onset(?:\s+(?:at|age))?|"
    r"diagnosed\s+at|diagnosis\s+at"
    r")\s*(?:age\s*)?(?P<age>\d{1,3})(?!\w)",
    re.IGNORECASE,
)
_VITAL_STATUS_RE = re.compile(
    r"(?<!\w)(?:passed\s+away|deceased|dead|died|living|alive)(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class FamilyHistoryRecord:
    """One family-member condition with privacy-safe source provenance.

    ``condition_span`` and ``relative_span`` are half-open character offsets
    into the caller's source document.  The record intentionally stores no
    condition or relative surface text.
    """

    relative_role: str
    condition_span: tuple[int, int]
    onset_age: int | None = None
    vital_status: str | None = None
    relative_span: tuple[int, int] | None = None
    advisory: str = FAMILY_HISTORY_ADVISORY

    def __post_init__(self) -> None:
        role = _normalize_relative_role(self.relative_role)
        if role is None:
            raise ValueError("relative_role must be a non-empty family role")
        object.__setattr__(self, "relative_role", role)
        object.__setattr__(
            self,
            "condition_span",
            _validate_offset_pair(self.condition_span, "condition_span"),
        )
        if self.relative_span is not None:
            object.__setattr__(
                self,
                "relative_span",
                _validate_offset_pair(self.relative_span, "relative_span"),
            )
        if self.onset_age is not None:
            age = _parse_age(self.onset_age)
            if age is None:
                raise ValueError("onset_age must be an integer age")
            object.__setattr__(self, "onset_age", age)
        if self.vital_status is not None:
            status = _normalize_vital_status(self.vital_status)
            if status is None:
                raise ValueError("vital_status must be alive or deceased")
            object.__setattr__(self, "vital_status", status)

    @property
    def record_type(self) -> str:
        """Return the interoperability-facing record shape name."""

        return "FamilyHistory"

    @property
    def relative(self) -> str:
        """Return the normalized relative role."""

        return self.relative_role

    @property
    def condition(self) -> tuple[int, int]:
        """Return the condition's source offset pair."""

        return self.condition_span

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation without source text."""

        payload: dict[str, Any] = {
            "record_type": self.record_type,
            "relative_role": self.relative_role,
            "condition_span": list(self.condition_span),
            "onset_age": self.onset_age,
            "vital_status": self.vital_status,
            "advisory": self.advisory,
        }
        if self.relative_span is not None:
            payload["relative_span"] = list(self.relative_span)
        return payload


@dataclass(frozen=True)
class _InputSpan:
    start: int
    end: int
    label: str
    surface: str
    section: str | None
    data: Mapping[str, Any]

    @property
    def offset(self) -> tuple[int, int]:
        """Return the stable source offset identity."""

        return self.start, self.end


@dataclass(frozen=True)
class _Evidence:
    offset: tuple[int, int] | None = None
    experiencer: str | None = None
    relative_role: str | None = None
    cue: str | None = None
    cue_offset: tuple[int, int] | None = None
    relative_span: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Attribute:
    start: int
    end: int
    value: int | str
    kind: str
    section: str | None
    relative_role: str | None = None

    @property
    def offset(self) -> tuple[int, int]:
        """Return the source offset identity for an attribute."""

        return self.start, self.end


def extract_family_history(
    spans: Iterable[Any] | Mapping[str, Any],
    experiencer: Any,
    sections: Iterable[Any] | Mapping[str, Any] | str | None = None,
    *,
    text: str | None = None,
) -> tuple[FamilyHistoryRecord, ...]:
    """Extract family-history records from existing spans and assignments.

    Args:
        spans: Condition, age, vital-status, and optional relative spans. Each
            item needs ``start``/``end`` offsets and a model label. Span
            mappings may carry ``document_text`` when ``text`` is omitted.
        experiencer: Refined experiencer assertions, an offset-keyed mapping,
            an assertion/assignment, or values already attached to the spans.
            Only the ``family`` experiencer is emitted; patient and ``other``
            subjects are excluded.
        sections: Optional precomputed section spans. Attributes cannot cross
            a section boundary. When omitted, sections are detected if source
            text is available.
        text: Optional source document text. It is used only for local cue and
            proximity resolution and is never copied into returned records.

    Returns:
        Deterministically ordered, offset-only :class:`FamilyHistoryRecord`
        values. The output is assistive and requires clinician review.
    """

    span_items = _iter_items(spans)
    if text is not None and not isinstance(text, str):
        raise TypeError("text must be a string when provided")
    source_text = text or _document_text(span_items)
    input_spans = tuple(
        item
        for raw in span_items
        if (item := _coerce_span(raw, source_text)) is not None
    )
    if not input_spans:
        return ()

    section_items = _coerce_sections(sections, source_text)
    section_by_offset = {
        item.offset: _section_for_span(item, section_items) for item in input_spans
    }
    input_spans = tuple(
        replace(item, section=section_by_offset.get(item.offset))
        for item in input_spans
    )
    evidence_by_offset, positional_evidence, global_evidence = _index_evidence(
        experiencer
    )
    conditions = tuple(item for item in input_spans if _is_condition(item.label))
    if not conditions:
        return ()

    age_attributes = _age_attributes(input_spans, source_text, section_items)
    status_attributes = _status_attributes(input_spans, source_text, section_items)
    records: list[FamilyHistoryRecord] = []
    for index, condition in enumerate(conditions):
        section = section_by_offset.get(condition.offset)
        evidence = _evidence_for(
            condition,
            index=index,
            evidence_by_offset=evidence_by_offset,
            positional_evidence=positional_evidence,
            global_evidence=global_evidence,
        )
        resolved = _resolve_condition_evidence(
            condition,
            section=section,
            evidence=evidence,
            source_text=source_text,
            section_items=section_items,
        )
        if resolved is None:
            continue
        role, relative_span = resolved
        onset_age = _condition_value(condition, "onset_age", "age_of_onset")
        if onset_age is None:
            onset_age = _nearest_attribute(
                condition,
                age_attributes,
                section=section,
                role=role,
                source_text=source_text,
                max_distance=_MAX_ATTRIBUTE_DISTANCE,
                kind="age",
            )
        vital_status = _condition_value(condition, "vital_status", "family_status")
        if vital_status is None:
            vital_status = _nearest_attribute(
                condition,
                status_attributes,
                section=section,
                role=role,
                source_text=source_text,
                max_distance=_MAX_STATUS_DISTANCE,
                kind="status",
            )
        record = FamilyHistoryRecord(
            relative_role=role,
            condition_span=condition.offset,
            onset_age=_parse_age(onset_age),
            vital_status=_normalize_vital_status(vital_status),
            relative_span=relative_span,
        )
        records.append(record)

    unique: dict[tuple[str, tuple[int, int]], FamilyHistoryRecord] = {}
    for record in records:
        unique.setdefault((record.relative_role, record.condition_span), record)
    return tuple(
        sorted(
            unique.values(),
            key=lambda record: (
                record.condition_span[0],
                record.condition_span[1],
                record.relative_role,
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
    return all(hasattr(value, field) for field in ("start", "end"))


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    nested_span = getattr(value, "span", None)
    if isinstance(nested_span, Mapping):
        return nested_span
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return result
    values: dict[str, Any] = {}
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
    ):
        if hasattr(value, field):
            values[field] = getattr(value, field)
    return values or None


def _coerce_span(value: Any, text: str | None) -> _InputSpan | None:
    data = _as_mapping(value)
    if data is None:
        return None
    try:
        start = int(data.get("start", data.get("start_char", -1)))
        end = int(data.get("end", data.get("end_char", -1)))
    except (TypeError, ValueError):
        return None
    if start < 0 or end <= start or (text is not None and end > len(text)):
        return None
    label = _raw_label(data)
    if not label:
        return None
    surface = data.get("text", data.get("word", ""))
    if text is not None and end <= len(text):
        surface = text[start:end]
    elif not isinstance(surface, str):
        surface = str(surface)
    section = _section_value(data)
    return _InputSpan(
        start=start,
        end=end,
        label=str(label),
        surface=str(surface),
        section=section,
        data=data,
    )


def _raw_label(data: Mapping[str, Any]) -> str:
    for key in ("label", "entity", "canonical_label", "entity_type", "type"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _section_value(data: Mapping[str, Any]) -> str | None:
    for key in ("section", "section_label", "section_name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        for key in ("section", "section_label", "section_name"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _document_text(items: Sequence[Any]) -> str | None:
    keys = ("document_text", "context_text", "source_text", "full_text", "note_text")
    for item in items:
        data = _as_mapping(item)
        if data is not None:
            for key in keys:
                value = data.get(key)
                if isinstance(value, str):
                    return value
        for key in keys:
            value = getattr(item, key, None)
            if isinstance(value, str):
                return value
    return None


def _coerce_sections(
    sections: Iterable[Any] | Mapping[str, Any] | str | None,
    text: str | None,
) -> tuple[tuple[int, int, str | None], ...]:
    if sections is None:
        if text is None:
            return ()
        sections = detect_sections(text)
    if isinstance(sections, str):
        end = len(text) if text is not None else 0
        return ((0, end, _section_name(sections)),) if end else ()
    items = _iter_items(sections)
    result: list[tuple[int, int, str | None]] = []
    for item in items:
        data = _as_mapping(item)
        if data is None:
            continue
        try:
            start = int(data.get("start", data.get("start_char", -1)))
            end = int(data.get("end", data.get("end_char", -1)))
        except (TypeError, ValueError):
            continue
        if start < 0 or end <= start or (text is not None and end > len(text)):
            continue
        label = data.get("label", data.get("section", data.get("name")))
        result.append((start, end, _section_name(label)))
    return tuple(sorted(result, key=lambda item: (item[0], item[1], item[2] or "")))


def _section_name(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    return canonical_section_name(raw) or raw.casefold().replace(" ", "_")


def _section_for_span(
    span: _InputSpan,
    sections: Sequence[tuple[int, int, str | None]],
) -> str | None:
    if span.section is not None:
        return _section_name(span.section)
    containing = [
        section
        for section in sections
        if section[0] <= span.start and span.end <= section[1]
    ]
    if not containing:
        return None
    return min(containing, key=lambda section: (section[1] - section[0], section[0]))[2]


def _is_condition(label: str) -> bool:
    return normalize_label(label) in _CONDITION_LABELS


def _index_evidence(
    value: Any,
) -> tuple[dict[tuple[int, int], _Evidence], tuple[_Evidence, ...], _Evidence | None]:
    if value is None:
        return {}, (), None
    if isinstance(value, Mapping) and not _is_evidence_mapping(value):
        keyed: dict[tuple[int, int], _Evidence] = {}
        positional: list[_Evidence] = []
        for key, item in value.items():
            offset = _offset_key(key)
            evidence = _extract_evidence(item, offset_hint=offset)
            if evidence.offset is not None:
                keyed[evidence.offset] = evidence
            else:
                positional.append(evidence)
        return keyed, tuple(positional), None

    items = _iter_items(value)
    if len(items) == 1 and not _has_offset_value(items[0]):
        evidence = _extract_evidence(items[0])
        if evidence.offset is None:
            return {}, (), evidence
    keyed = {}
    positional = []
    for item in items:
        evidence = _extract_evidence(item)
        if evidence.offset is not None:
            keyed[evidence.offset] = evidence
        else:
            positional.append(evidence)
    return keyed, tuple(positional), None


def _is_evidence_mapping(value: Mapping[str, Any]) -> bool:
    return any(
        key in value
        for key in (
            "experiencer",
            "assignment",
            "assertion",
            "cue",
            "relative_role",
            "family_member",
        )
    ) or _has_offset_value(value)


def _has_offset_value(value: Any) -> bool:
    data = _as_mapping(value)
    if data is None:
        return False
    return "start" in data and "end" in data


def _offset_key(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return _validate_offset_pair((int(value[0]), int(value[1])), "offset")
        except (TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.fullmatch(r"\(?\s*(\d+)\s*[, :]\s*(\d+)\s*\)?", value)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def _extract_evidence(
    value: Any, *, offset_hint: tuple[int, int] | None = None
) -> _Evidence:
    data = _as_mapping(value) or {}
    nested_span = getattr(value, "span", None)
    if isinstance(nested_span, Mapping):
        span_data = nested_span
    else:
        span_data = data.get("span") if isinstance(data.get("span"), Mapping) else data
    offset = _offsets(span_data) or offset_hint

    assignment = getattr(value, "assignment", None)
    if assignment is None:
        assignment = data.get("assignment")
    assertion = getattr(value, "assertion", None)
    if assertion is None:
        assertion = data.get("assertion")
    if assertion is None:
        assertion = data.get("clinical_context") or data.get("context")
    if assertion is None:
        metadata = data.get("metadata")
        if isinstance(metadata, Mapping):
            assertion = metadata.get("clinical_context") or metadata.get("context")

    experiencer = _field(assignment, "experiencer") or _field(assertion, "experiencer")
    if experiencer is None:
        experiencer = _field(value, "experiencer")
    role = _field(assignment, "relative_role", "family_member", "relative") or _field(
        value, "relative_role", "family_member", "relative"
    )
    cue = _field(assignment, "cue") or _field(value, "cue")
    cue_offset = _offsets_from_field(
        _field(assignment, "cue_offset") or _field(value, "cue_offset")
    )
    relative_span = _offsets_from_field(
        _field(assignment, "relative_span")
        or _field(value, "relative_span")
        or _field(value, "subject_span")
    )
    if role is None and cue is not None:
        role = str(cue)
    if experiencer is None and role is not None:
        experiencer = FAMILY_EXPERIENCER
    return _Evidence(
        offset=offset,
        experiencer=_normalize_experiencer(experiencer),
        relative_role=None if role is None else str(role),
        cue=None if cue is None else str(cue),
        cue_offset=cue_offset,
        relative_span=relative_span,
    )


def _offsets(value: Any) -> tuple[int, int] | None:
    data = _as_mapping(value)
    if data is None:
        return None
    try:
        return _validate_offset_pair(
            (
                int(data.get("start", data.get("start_char", -1))),
                int(data.get("end", data.get("end_char", -1))),
            ),
            "offset",
        )
    except (TypeError, ValueError):
        return None


def _offsets_from_field(value: Any) -> tuple[int, int] | None:
    if value is None:
        return None
    return _offset_key(value) or _offsets(value)


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


def _evidence_for(
    condition: _InputSpan,
    *,
    index: int,
    evidence_by_offset: Mapping[tuple[int, int], _Evidence],
    positional_evidence: Sequence[_Evidence],
    global_evidence: _Evidence | None,
) -> _Evidence:
    direct = _extract_evidence(condition)
    keyed = evidence_by_offset.get(condition.offset)
    positional = (
        positional_evidence[index] if index < len(positional_evidence) else None
    )
    return _merge_evidence(direct, keyed, positional, global_evidence)


def _merge_evidence(*values: _Evidence | None) -> _Evidence:
    fields = (
        "offset",
        "experiencer",
        "relative_role",
        "cue",
        "cue_offset",
        "relative_span",
    )
    resolved: dict[str, Any] = {}
    for value in values:
        if value is None:
            continue
        for field in fields:
            current = getattr(value, field)
            if current is not None and resolved.get(field) is None:
                resolved[field] = current
    return _Evidence(**resolved)


def _resolve_condition_evidence(
    condition: _InputSpan,
    *,
    section: str | None,
    evidence: _Evidence,
    source_text: str | None,
    section_items: Sequence[tuple[int, int, str | None]],
) -> tuple[str, tuple[int, int] | None] | None:
    assignment = None
    if source_text is not None:
        scoped_text = _mask_outside_section(source_text, condition, section_items)
        assignment = resolve_experiencer(
            scoped_text,
            {"start": condition.start, "end": condition.end},
        )

    experiencer = evidence.experiencer
    if experiencer is None:
        if assignment is not None and assignment.source == "cue":
            experiencer = _normalize_experiencer(assignment.experiencer)
        elif section == "family_history":
            experiencer = FAMILY_EXPERIENCER
        else:
            experiencer = PATIENT_EXPERIENCER
    if experiencer != FAMILY_EXPERIENCER:
        return None

    role = _normalize_relative_role(evidence.relative_role)
    relative_span = evidence.relative_span or evidence.cue_offset
    if role is None and evidence.cue:
        role = _normalize_relative_role(evidence.cue)
    if assignment is not None and assignment.experiencer == FAMILY_EXPERIENCER:
        if role is None:
            role = _normalize_relative_role(assignment.cue)
        if relative_span is None:
            relative_span = assignment.cue_offset
    if role is None and source_text is not None:
        role, cue_offset = _nearest_relative_role(
            source_text,
            condition,
            sections=section_items,
        )
        relative_span = relative_span or cue_offset
    return role or "family", relative_span


def _mask_outside_section(
    text: str,
    span: _InputSpan,
    sections: Sequence[tuple[int, int, str | None]],
) -> str:
    containing = [
        section
        for section in sections
        if section[0] <= span.start and span.end <= section[1]
    ]
    if not containing:
        return text
    start, end, _label = min(containing, key=lambda section: section[1] - section[0])
    return " " * start + text[start:end] + " " * (len(text) - end)


def _nearest_relative_role(
    text: str,
    condition: _InputSpan,
    *,
    sections: Sequence[tuple[int, int, str | None]],
) -> tuple[str | None, tuple[int, int] | None]:
    bounded = _mask_outside_section(text, condition, sections)
    clause_start = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(bounded, 0, condition.start):
        clause_start = match.end()
    matches = [
        match
        for match in _RELATIVE_ROLE_RE.finditer(bounded, clause_start, condition.start)
    ]
    if not matches:
        return None, None
    match = min(matches, key=lambda item: (condition.start - item.end(), item.start()))
    return _normalize_relative_role(match.group(0)), match.span()


def _age_attributes(
    spans: Sequence[_InputSpan],
    text: str | None,
    sections: Sequence[tuple[int, int, str | None]],
) -> tuple[_Attribute, ...]:
    attributes: list[_Attribute] = []
    for span in spans:
        value = _condition_value(span, "onset_age", "age_of_onset")
        normalized = normalize_label(span.label)
        raw_label = _compact(span.label)
        if value is not None:
            parsed = _parse_age(value)
            if parsed is not None:
                attributes.append(
                    _Attribute(
                        span.start,
                        span.end,
                        parsed,
                        "age",
                        _section_name(span.section),
                        _explicit_role(span),
                    )
                )
        elif normalized == AGE or "onsetage" in raw_label or "ageofonset" in raw_label:
            parsed = _parse_age(span.surface)
            if parsed is not None:
                attributes.append(
                    _Attribute(
                        span.start,
                        span.end,
                        parsed,
                        "age",
                        _section_name(span.section),
                        _explicit_role(span),
                    )
                )
    if text is not None:
        for match in _AGE_RE.finditer(text):
            age = _parse_age(match.group("age"))
            if age is not None:
                attributes.append(
                    _Attribute(
                        match.start("age"),
                        match.end("age"),
                        age,
                        "age",
                        _section_for_offset(
                            match.start("age"), match.end("age"), sections
                        ),
                    )
                )
    return _dedupe_attributes(attributes)


def _status_attributes(
    spans: Sequence[_InputSpan],
    text: str | None,
    sections: Sequence[tuple[int, int, str | None]],
) -> tuple[_Attribute, ...]:
    attributes: list[_Attribute] = []
    for span in spans:
        value = _condition_value(span, "vital_status", "family_status")
        raw_label = _compact(span.label)
        if value is None and (
            "vitalstatus" in raw_label
            or "familystatus" in raw_label
            or _normalize_vital_status(span.surface) is not None
        ):
            value = span.surface
        status = _normalize_vital_status(value)
        if status is not None:
            attributes.append(
                _Attribute(
                    span.start,
                    span.end,
                    status,
                    "status",
                    _section_name(span.section),
                    _explicit_role(span),
                )
            )
    if text is not None:
        for match in _VITAL_STATUS_RE.finditer(text):
            status = _normalize_vital_status(match.group(0))
            if status is not None:
                attributes.append(
                    _Attribute(
                        match.start(),
                        match.end(),
                        status,
                        "status",
                        _section_for_offset(match.start(), match.end(), sections),
                    )
                )
    return _dedupe_attributes(attributes)


def _dedupe_attributes(attributes: Sequence[_Attribute]) -> tuple[_Attribute, ...]:
    unique: dict[tuple[int, int, str, str], _Attribute] = {}
    for attribute in attributes:
        unique.setdefault(
            (attribute.start, attribute.end, attribute.kind, str(attribute.value)),
            attribute,
        )
    return tuple(
        sorted(unique.values(), key=lambda item: (item.start, item.end, item.kind))
    )


def _section_for_offset(
    start: int,
    end: int,
    sections: Sequence[tuple[int, int, str | None]],
) -> str | None:
    containing = [
        section for section in sections if section[0] <= start and end <= section[1]
    ]
    if not containing:
        return None
    return min(containing, key=lambda section: (section[1] - section[0], section[0]))[2]


def _nearest_attribute(
    condition: _InputSpan,
    attributes: Sequence[_Attribute],
    *,
    section: str | None,
    role: str,
    source_text: str | None,
    max_distance: int,
    kind: str,
) -> int | str | None:
    candidates: list[tuple[int, int, int, int, _Attribute]] = []
    for attribute in attributes:
        if attribute.kind != kind or not _same_section(section, attribute.section):
            continue
        if attribute.relative_role is not None:
            attr_role = _normalize_relative_role(attribute.relative_role)
            if attr_role not in (None, "family", "relative", role):
                continue
        distance = _character_distance(condition.offset, attribute.offset)
        if distance > max_distance:
            continue
        if source_text is not None:
            between = _text_between(condition.offset, attribute.offset, source_text)
            boundary_re = (
                _CLAUSE_BOUNDARY_RE if kind == "age" else _SENTENCE_BOUNDARY_RE
            )
            if boundary_re.search(between):
                continue
            if kind == "age" and _death_context(between):
                continue
            if (
                kind == "age"
                and attribute.start < condition.end
                and re.search(r"\b(?:and|or|but)\b", between, re.IGNORECASE)
            ):
                continue
        direction = 0 if attribute.start >= condition.end else 1
        candidates.append(
            (direction, distance, attribute.start, attribute.end, attribute)
        )
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[:4])[4].value


def _same_section(left: str | None, right: str | None) -> bool:
    return left is None or right is None or left == right


def _character_distance(left: tuple[int, int], right: tuple[int, int]) -> int:
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


def _text_between(
    left: tuple[int, int],
    right: tuple[int, int],
    text: str,
) -> str:
    if left[1] <= right[0]:
        return text[left[1] : right[0]]
    if right[1] <= left[0]:
        return text[right[1] : left[0]]
    return ""


def _death_context(text: str) -> bool:
    return bool(re.search(r"\b(?:died|dead|deceased|passed\s+away)\b", text, re.I))


def _condition_value(span: _InputSpan, *names: str) -> Any:
    value = _field(span.data, *names)
    if value is not None:
        return value
    metadata = span.data.get("metadata")
    if isinstance(metadata, Mapping):
        value = _field(metadata, *names)
        if value is not None:
            return value
        for key in ("family_history", "clinical_context", "context"):
            nested = metadata.get(key)
            value = _field(nested, *names)
            if value is not None:
                return value
    for key in ("family_history", "clinical_context", "context"):
        nested = span.data.get(key)
        value = _field(nested, *names)
        if value is not None:
            return value
    return None


def _explicit_role(span: _InputSpan) -> str | None:
    value = _condition_value(
        span,
        "relative_role",
        "family_member",
        "relative",
        "subject_role",
    )
    return None if value is None else _normalize_relative_role(value)


def _normalize_experiencer(value: Any) -> str | None:
    if value is None:
        return None
    nested = _field(value, "experiencer")
    if nested is not None and nested is not value:
        return _normalize_experiencer(nested)
    if _normalize_relative_role(value) is not None:
        return FAMILY_EXPERIENCER
    normalized = _compact(str(value))
    if normalized in {
        "family",
        "familymember",
        "familyhistory",
        "familial",
        "relative",
        "parent",
        "mother",
        "father",
        "sibling",
        "grandparent",
    }:
        return FAMILY_EXPERIENCER
    if normalized in {"patient", "self", "subject", "currentpatient"}:
        return PATIENT_EXPERIENCER
    if normalized in {"other", "nonpatient", "donor", "roommate"}:
        return "other"
    role = _normalize_relative_role(normalized)
    return FAMILY_EXPERIENCER if role is not None else normalized


def _normalize_relative_role(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().casefold()
    if not raw:
        return None
    raw = raw.replace("’", "'")
    raw = re.sub(r"\s+", " ", raw)
    raw = re.sub(r"['’]s\b", "", raw)
    if raw in _CANONICAL_RELATIVE_ROLES:
        return raw
    underscored = raw.replace("_", " ")
    if underscored in _CANONICAL_RELATIVE_ROLES:
        return underscored
    for surface, role in _RELATIVE_ROLE_PATTERNS:
        if re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", raw):
            return role
    if raw in {"family", "family member", "relative"}:
        return "family" if raw != "relative" else "relative"
    return None


def _parse_age(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        age = value
    else:
        match = re.search(r"(?<!\w)(\d{1,3})(?!\w)", str(value))
        if match is None:
            return None
        age = int(match.group(1))
    return age if 0 <= age <= 130 else None


def _normalize_vital_status(value: Any) -> str | None:
    if value is None:
        return None
    normalized = _compact(str(value))
    if normalized in {"alive", "living"}:
        return "alive"
    if normalized in {"deceased", "dead", "died", "passedaway", "passed"}:
        return "deceased"
    return None


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _validate_offset_pair(value: Any, name: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must contain start and end offsets")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{name} offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{name} must be a non-empty half-open offset")
    return start, end


__all__ = [
    "FAMILY_HISTORY_ADVISORY",
    "FamilyHistoryRecord",
    "extract_family_history",
]
