"""Conservative, evidence-linked profiles for discharge-summary documents.

The profile is a deterministic view over :mod:`openmed.clinical.discharge_summary`.
It keeps the extracted surface text, section, and half-open source offsets while
attaching the existing ConText certainty, negation, and temporality axes.  It
does not infer a diagnosis, add a medication, or turn a follow-up instruction
into a recommendation.

Only local rules and caller-supplied spans are used.  No model, terminology
service, or network request is required by the default path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from typing import Any, Literal, cast

from .context import (
    AFFIRMED,
    CERTAIN,
    CERTAINTY_VALUES,
    NEGATION_VALUES,
    RECENT,
    TEMPORALITY_VALUES,
    Certainty,
    ClinicalAssertion,
    Negation,
    assert_context_axes,
)
from .discharge_summary import (
    DischargeMention,
    DischargeSummary,
    MedicationRegimen,
    extract_discharge_summary,
)
from .sections import detect_sections
from .sig_parser import parse_sig

SpanOffset = tuple[int, int]
DischargeProfileField = Literal[
    "diagnoses",
    "procedures",
    "medications",
    "follow_up",
    "instructions",
]

DISCHARGE_PROFILE_NAME = "discharge_summary"
DISCHARGE_PROFILE_SCHEMA_VERSION = "openmed.clinical.discharge_profile.v1"
DISCHARGE_PROFILE_FIELDS: tuple[DischargeProfileField, ...] = (
    "diagnoses",
    "procedures",
    "medications",
    "follow_up",
    "instructions",
)
DISCHARGE_PROFILE_ADVISORY = (
    "This discharge-summary profile is deterministic, evidence-linked, and "
    "assistive. It preserves stated uncertainty and does not generate clinical "
    "recommendations or new clinical facts; qualified clinical review is required."
)
DISCHARGE_SUMMARY_PROFILE_ADVISORY = DISCHARGE_PROFILE_ADVISORY

_PROFILE_SECTION_ALIASES: Mapping[DischargeProfileField, tuple[str, ...]] = {
    "diagnoses": (
        "discharge diagnoses",
        "discharge diagnosis",
        "diagnoses at discharge",
        "diagnosis at discharge",
        "diagnoses on discharge",
        "diagnosis on discharge",
        "discharge dx",
        "principal diagnosis",
        "principal dx",
        "secondary diagnoses",
        "secondary diagnosis",
        "final diagnoses",
        "final diagnosis",
        "diagnoses",
        "diagnosis",
        "problems at discharge",
        "problem list",
        "active problems",
        "problems",
    ),
    "procedures": (
        "procedures performed",
        "procedure performed",
        "procedures during admission",
        "procedures during hospitalization",
        "hospital procedures",
        "operative procedures",
        "operations performed",
        "operation performed",
        "operations",
        "procedures",
        "procedure",
        "interventions",
        "intervention",
    ),
    "medications": (
        "discharge medications",
        "discharge medication",
        "discharge meds",
        "discharge med",
        "d/c medications",
        "d/c medication",
        "d/c meds",
        "d/c med",
        "dc medications",
        "dc medication",
        "dc meds",
        "dc med",
        "medications at discharge",
        "medication at discharge",
        "medications on discharge",
        "medication on discharge",
        "meds at discharge",
        "meds on discharge",
        "prescriptions at discharge",
        "discharge medication list",
        "medications",
        "medication list",
        "meds",
    ),
    "follow_up": (
        "follow-up appointments",
        "follow up appointments",
        "follow-up plan",
        "follow up plan",
        "post-discharge follow-up",
        "post discharge follow up",
        "discharge follow-up",
        "discharge follow up",
        "follow-up",
        "follow up",
        "followup",
        "appointments",
        "disposition",
        "discharge plan",
        "assessment and plan",
        "assessment/plan",
        "plan",
    ),
    "instructions": (
        "follow-up instructions",
        "follow up instructions",
        "discharge instructions",
        "patient instructions",
        "home instructions",
        "return precautions",
        "activity restrictions",
        "wound care",
        "instructions",
        "activity",
        "diet",
    ),
}
_ALIAS_TO_FIELD = {
    re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip(): field_name
    for field_name, aliases in _PROFILE_SECTION_ALIASES.items()
    for alias in aliases
}
_HEADER_DELIMITER_RE = re.compile(r"(?::|：|﹕|꞉|\s+[—–-]\s+)")
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_NON_VALUE_RE = re.compile(
    r"^(?:none|no(?:ne)?|n/?a|not applicable|no instructions?|"
    r"no follow[- ]?up|no appointments?|see above)\.?$",
    re.IGNORECASE,
)
_MEDICATION_PREFIX_RE = re.compile(
    r"^(?:take|start|continue|resume|restart|begin|use|hold|stop)\s+",
    re.IGNORECASE,
)
_MEDICATION_BOUNDARY_RE = re.compile(
    r"(?<!\w)(?:\d+(?:\.\d+)?\s*"
    r"(?:mg|mcg|g|kg|ml|l|unit|units|meq|iu|%|tablet|tablets|tab|tabs|"
    r"capsule|capsules|cap|caps|puff|puffs|drop|drops|spray|sprays)\b|"
    r"po|iv|im|sc|sq|subq|oral|intravenous|intramuscular|subcutaneous|"
    r"topical|inhaled|by mouth|per os|qd|daily|once daily|bid|twice daily|"
    r"tid|three times daily|qid|four times daily|qhs|nightly|weekly|"
    r"as needed|when needed|if needed|prn)(?!\w)",
    re.IGNORECASE,
)


def _validate_span(value: object, name: str) -> SpanOffset:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item integer span")
    start, end = value
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{name} must be a two-item integer span")
    if start < 0 or end <= start:
        raise ValueError(f"{name} must satisfy 0 <= start < end")
    return start, end


def _validate_range(value: object, name: str) -> SpanOffset:
    """Validate a half-open range whose content may be empty."""

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item integer range")
    start, end = value
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{name} must be a two-item integer range")
    if start < 0 or end < start:
        raise ValueError(f"{name} must satisfy 0 <= start <= end")
    return start, end


def _normalize_axis(value: object, allowed: tuple[str, ...], default: str) -> str:
    if not isinstance(value, str):
        return default
    normalized = value.strip().casefold()
    return normalized if normalized in allowed else default


@dataclass(frozen=True)
class DischargeProfileSection:
    """A recognized discharge section with source-boundary provenance."""

    field: DischargeProfileField
    header: str
    span: SpanOffset
    content_span: SpanOffset

    def __post_init__(self) -> None:
        if self.field not in DISCHARGE_PROFILE_FIELDS:
            raise ValueError("section field is not supported by the profile")
        if not isinstance(self.header, str) or not self.header.strip():
            raise ValueError("section header must not be empty")
        start, end = _validate_span(self.span, "section span")
        content_start, content_end = _validate_range(
            self.content_span, "section content span"
        )
        if not start <= content_start <= content_end <= end:
            raise ValueError("section content span must be contained by section span")
        object.__setattr__(self, "span", (start, end))
        object.__setattr__(self, "content_span", (content_start, content_end))

    @property
    def start(self) -> int:
        """Return the inclusive section start."""

        return self.span[0]

    @property
    def end(self) -> int:
        """Return the exclusive section end."""

        return self.span[1]

    @property
    def content_start(self) -> int:
        """Return the first offset after the section heading."""

        return self.content_span[0]

    @property
    def content_end(self) -> int:
        """Return the exclusive content end."""

        return self.content_span[1]

    def to_dict(self) -> dict[str, Any]:
        """Return section metadata without copying section body text."""

        return {
            "field": self.field,
            "header": self.header,
            "start": self.start,
            "end": self.end,
            "content_start": self.content_start,
            "content_end": self.content_end,
            "provenance": {
                "start": self.content_start,
                "end": self.content_end,
            },
        }


@dataclass(frozen=True)
class DischargeProfileItem:
    """One extracted value linked to a source span and assertion axes.

    ``text`` is the exact source evidence for the item.  Medication items use
    ``value`` for the medication name and ``text`` for the complete written
    regimen; all other fields use the same value for both attributes.
    """

    text: str
    span: SpanOffset
    field: DischargeProfileField
    section: str
    value: str | None = None
    certainty: Certainty = CERTAIN
    negation: Negation = AFFIRMED
    temporality: str = RECENT
    status: str | None = None
    medication_span: SpanOffset | None = None
    sig: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("profile item text must not be empty")
        if self.field not in DISCHARGE_PROFILE_FIELDS:
            raise ValueError("profile item field is not supported by the profile")
        if not isinstance(self.section, str) or not self.section.strip():
            raise ValueError("profile item section must not be empty")
        start, end = _validate_span(self.span, "profile item span")
        value = self.text if self.value is None else self.value
        if not isinstance(value, str) or not value.strip():
            raise ValueError("profile item value must not be empty")
        certainty = _normalize_axis(self.certainty, CERTAINTY_VALUES, CERTAIN)
        negation = _normalize_axis(self.negation, NEGATION_VALUES, AFFIRMED)
        temporality = _normalize_axis(
            self.temporality,
            TEMPORALITY_VALUES,
            RECENT,
        )
        medication_span = self.medication_span
        if medication_span is not None:
            medication_span = _validate_span(medication_span, "medication span")
            if not start <= medication_span[0] <= medication_span[1] <= end:
                raise ValueError("medication span must be contained by item span")
        if self.field == "medications" and medication_span is None:
            raise ValueError("medication items require a medication span")
        if not isinstance(self.sig, Mapping):
            raise TypeError("profile item sig must be a mapping")
        object.__setattr__(self, "span", (start, end))
        object.__setattr__(self, "value", value.strip())
        object.__setattr__(self, "certainty", cast(Certainty, certainty))
        object.__setattr__(self, "negation", cast(Negation, negation))
        object.__setattr__(self, "temporality", temporality)
        object.__setattr__(self, "medication_span", medication_span)
        object.__setattr__(self, "sig", dict(self.sig))

    @property
    def start(self) -> int:
        """Return the inclusive evidence start."""

        return self.span[0]

    @property
    def end(self) -> int:
        """Return the exclusive evidence end."""

        return self.span[1]

    @property
    def source_span(self) -> SpanOffset:
        """Return the half-open source evidence span."""

        return self.span

    @property
    def evidence_span(self) -> dict[str, int]:
        """Return JSON-ready evidence offsets."""

        return {"start": self.start, "end": self.end}

    @property
    def uncertainty(self) -> Certainty:
        """Return the certainty value under the descriptive uncertainty alias."""

        return self.certainty

    @property
    def uncertain(self) -> bool:
        """Return whether the item is explicitly uncertain or hypothetical."""

        return self.certainty == "uncertain" or self.temporality == "hypothetical"

    @property
    def is_uncertain(self) -> bool:
        """Boolean alias for :attr:`uncertain`."""

        return self.uncertain

    @property
    def medication(self) -> str:
        """Return the medication name, or the item value for other fields."""

        return self.value or self.text

    @property
    def regimen(self) -> str:
        """Return the complete written medication regimen."""

        return self.text

    @property
    def assertion(self) -> ClinicalAssertion:
        """Return the advisory assertion axes attached to the evidence."""

        return ClinicalAssertion(
            temporality=self.temporality,
            certainty=self.certainty,
            negation=self.negation,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the evidence-linked item in a deterministic JSON shape."""

        payload: dict[str, Any] = {
            "field": self.field,
            "value": self.value,
            "text": self.text,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "span": self.evidence_span,
            "evidence": self.evidence_span,
            "certainty": self.certainty,
            "uncertainty": self.certainty,
            "uncertain": self.uncertain,
            "negation": self.negation,
            "temporality": self.temporality,
            "assertion": self.assertion.to_dict(),
        }
        if self.status is not None:
            payload["status"] = self.status
        if self.medication_span is not None:
            payload["medication_span"] = {
                "start": self.medication_span[0],
                "end": self.medication_span[1],
            }
            payload["sig"] = dict(self.sig)
        return payload


@dataclass(frozen=True)
class DischargeSummaryProfile:
    """Structured, review-first discharge-summary profile."""

    diagnoses: tuple[DischargeProfileItem, ...]
    procedures: tuple[DischargeProfileItem, ...]
    medications: tuple[DischargeProfileItem, ...]
    follow_up: tuple[DischargeProfileItem, ...]
    instructions: tuple[DischargeProfileItem, ...]
    document_type: str
    sections: tuple[DischargeProfileSection, ...] = ()
    advisory: str = DISCHARGE_PROFILE_ADVISORY

    def __post_init__(self) -> None:
        for name in DISCHARGE_PROFILE_FIELDS:
            values = tuple(getattr(self, name))
            if any(not isinstance(item, DischargeProfileItem) for item in values):
                raise TypeError(f"{name} must contain profile items")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "sections", tuple(self.sections))
        if not isinstance(self.document_type, str) or not self.document_type.strip():
            raise ValueError("document_type must not be empty")
        if not isinstance(self.advisory, str) or not self.advisory.strip():
            raise ValueError("profile advisory must not be empty")

    @property
    def record_type(self) -> str:
        """Return the stable serialized record name."""

        return "DischargeSummaryProfile"

    @property
    def profile_name(self) -> str:
        """Return the stable profile name."""

        return DISCHARGE_PROFILE_NAME

    @property
    def items(self) -> tuple[DischargeProfileItem, ...]:
        """Return all items in stable field and source order."""

        return tuple(
            item for name in DISCHARGE_PROFILE_FIELDS for item in getattr(self, name)
        )

    @property
    def field_items(self) -> dict[str, tuple[DischargeProfileItem, ...]]:
        """Return the extracted fields keyed by their public names."""

        return {name: getattr(self, name) for name in DISCHARGE_PROFILE_FIELDS}

    @property
    def discharge_diagnoses(self) -> tuple[DischargeProfileItem, ...]:
        """Compatibility alias for the canonical discharge-summary field."""

        return self.diagnoses

    @property
    def discharge_medications(self) -> tuple[DischargeProfileItem, ...]:
        """Compatibility alias for the canonical discharge-summary field."""

        return self.medications

    @property
    def follow_up_instructions(self) -> tuple[DischargeProfileItem, ...]:
        """Return follow-up items and separately stated instructions."""

        return self.follow_up + self.instructions

    @property
    def field_provenance(self) -> dict[str, list[dict[str, int]]]:
        """Return source offsets for every extracted field item."""

        return {
            name: [item.evidence_span for item in getattr(self, name)]
            for name in DISCHARGE_PROFILE_FIELDS
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access for JSON-oriented callers."""

        if key == "record_type":
            return self.record_type
        if key == "profile_name":
            return self.profile_name
        if key in DISCHARGE_PROFILE_FIELDS:
            return getattr(self, key)
        if key in {"discharge_diagnoses", "discharge_medications"}:
            return getattr(self, key)
        if key in {"field_provenance", "provenance"}:
            return self.field_provenance
        if key == "sections":
            return self.sections
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, JSON-compatible profile report."""

        return {
            "record_type": self.record_type,
            "schema_version": DISCHARGE_PROFILE_SCHEMA_VERSION,
            "profile_name": self.profile_name,
            "document_type": self.document_type,
            "diagnoses": [item.to_dict() for item in self.diagnoses],
            "procedures": [item.to_dict() for item in self.procedures],
            "medications": [item.to_dict() for item in self.medications],
            "follow_up": [item.to_dict() for item in self.follow_up],
            "instructions": [item.to_dict() for item in self.instructions],
            "sections": [section.to_dict() for section in self.sections],
            "field_provenance": self.field_provenance,
            "advisory": self.advisory,
        }

    def to_json(self) -> str:
        """Serialize the profile with fixed separators and sorted keys."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)


# Short aliases keep the module discoverable for callers that use the profile
# name without the full record vocabulary.
DischargeProfile = DischargeSummaryProfile
DischargeEvidence = DischargeProfileItem
DischargeFieldEvidence = DischargeProfileItem


@dataclass(frozen=True)
class _SectionHit:
    field: DischargeProfileField
    header: str
    start: int
    content_start: int


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _scan_profile_sections(
    text: str, language: str | None
) -> tuple[DischargeProfileSection, ...]:
    hits: list[_SectionHit] = []
    cursor = 0
    for raw_line in text.splitlines(keepends=True):
        line_start = cursor
        line_end = cursor + len(raw_line)
        content_end = line_end - len(raw_line) + len(raw_line.rstrip("\r\n"))
        line = text[line_start:content_end]
        leading = len(line) - len(line.lstrip())
        candidate = line[leading:]
        marker = _LIST_PREFIX_RE.match(candidate)
        if marker is not None:
            leading += marker.end()
            candidate = candidate[marker.end() :]
        candidate_start = line_start + leading
        delimiter = _HEADER_DELIMITER_RE.search(candidate)
        if delimiter is None:
            header = candidate.strip()
            content_start = line_end
        else:
            header = candidate[: delimiter.start()].strip()
            content_start = candidate_start + delimiter.end()
            while content_start < content_end and text[content_start] in " \t":
                content_start += 1
        field_name = _ALIAS_TO_FIELD.get(_normalize_header(header))
        if field_name is not None:
            hits.append(
                _SectionHit(
                    field=field_name,
                    header=header,
                    start=line_start,
                    content_start=min(content_start, line_end),
                )
            )
        cursor = line_end
    if cursor < len(text):
        line = text[cursor:]
        leading = len(line) - len(line.lstrip())
        candidate = line[leading:]
        delimiter = _HEADER_DELIMITER_RE.search(candidate)
        header = (
            candidate[: delimiter.start()].strip()
            if delimiter is not None
            else candidate.strip()
        )
        field_name = _ALIAS_TO_FIELD.get(_normalize_header(header))
        if field_name is not None:
            content_start = (
                cursor + leading + delimiter.end()
                if delimiter is not None
                else len(text)
            )
            hits.append(
                _SectionHit(
                    field=field_name,
                    header=header,
                    start=cursor,
                    content_start=min(content_start, len(text)),
                )
            )

    if not hits:
        return ()
    detected_boundaries = {
        int(section["start"])
        for section in detect_sections(text, language=language)
        if isinstance(section.get("start"), int)
    }
    boundaries = sorted(detected_boundaries | {hit.start for hit in hits})
    sections: list[DischargeProfileSection] = []
    seen: set[tuple[int, DischargeProfileField]] = set()
    for hit in sorted(hits, key=lambda item: item.start):
        key = (hit.start, hit.field)
        if key in seen:
            continue
        seen.add(key)
        end = next((value for value in boundaries if value > hit.start), len(text))
        if end <= hit.start:
            continue
        content_start = min(max(hit.content_start, hit.start), end)
        sections.append(
            DischargeProfileSection(
                field=hit.field,
                header=hit.header,
                span=(hit.start, end),
                content_span=(content_start, end),
            )
        )
    return tuple(sections)


def _trimmed_span(text: str, start: int, end: int) -> SpanOffset | None:
    start = max(0, start)
    end = min(len(text), end)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _remove_list_prefix(text: str, span: SpanOffset) -> SpanOffset:
    match = _LIST_PREFIX_RE.match(text[span[0] : span[1]])
    if match is None:
        return span
    return _trimmed_span(text, span[0] + match.end(), span[1]) or span


def _section_items(
    text: str, section: DischargeProfileSection
) -> tuple[SpanOffset, ...]:
    line_spans = [
        span
        for match in re.finditer(
            r"[^\r\n]+", text[section.content_start : section.content_end]
        )
        if (
            span := _trimmed_span(
                text,
                section.content_start + match.start(),
                section.content_start + match.end(),
            )
        )
    ]
    if not line_spans:
        return ()
    if len(line_spans) != 1 or ";" not in text[line_spans[0][0] : line_spans[0][1]]:
        result: list[SpanOffset] = []
        for span in line_spans:
            value_span = _remove_list_prefix(text, span)
            if not _is_non_value(text[value_span[0] : value_span[1]]):
                result.append(value_span)
        return tuple(result)
    only = line_spans[0]
    pieces: list[SpanOffset] = []
    for match in re.finditer(r"[^;]+", text[only[0] : only[1]]):
        span = _trimmed_span(text, only[0] + match.start(), only[0] + match.end())
        if span is None:
            continue
        span = _remove_list_prefix(text, span)
        if not _is_non_value(text[span[0] : span[1]]):
            pieces.append(span)
    return tuple(pieces)


def _is_non_value(value: str) -> bool:
    return _NON_VALUE_RE.fullmatch(value.strip()) is not None


def _section_for_span(
    sections: Iterable[DischargeProfileSection],
    start: int,
    end: int,
) -> DischargeProfileSection | None:
    containing = [
        section
        for section in sections
        if section.content_start <= start and end <= section.content_end
    ]
    return min(
        containing,
        key=lambda section: (section.end - section.start, section.start),
        default=None,
    )


def _medication_name_span(text: str, span: SpanOffset) -> SpanOffset:
    """Return the medication-name range within one written regimen."""

    source = text[span[0] : span[1]]
    leading = len(source) - len(source.lstrip())
    candidate = source[leading:]
    prefix = _MEDICATION_PREFIX_RE.match(candidate)
    if prefix is not None:
        leading += prefix.end()
        candidate = candidate[prefix.end() :]
    boundary = _MEDICATION_BOUNDARY_RE.search(candidate)
    end = boundary.start() if boundary is not None else len(candidate)
    name = candidate[:end].strip(" \t,;:-")
    if not name:
        name = candidate.strip()
    name_start = source.find(name, leading)
    if name_start < 0:
        name_start = leading
    return span[0] + name_start, span[0] + name_start + len(name)


def _mapping_for(value: object) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return mapped
    names = (
        "start",
        "end",
        "offset",
        "certainty",
        "uncertainty",
        "negation",
        "temporality",
        "assertion",
        "status",
    )
    mapped = {name: getattr(value, name) for name in names if hasattr(value, name)}
    return mapped or None


def _iter_input_spans(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        for key in ("entities", "clinical_entities", "spans"):
            if key in value:
                return _iter_input_spans(value[key])
        return (value,)
    if isinstance(value, (str, bytes)):
        return ()
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError:
        return (value,)


def _offset_from_mapping(data: Mapping[str, Any]) -> SpanOffset | None:
    raw = data.get("offset", data.get("offsets"))
    if isinstance(raw, (tuple, list)) and len(raw) == 2:
        start, end = raw
    else:
        start, end = data.get("start"), data.get("end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None
    return start, end


def _assertion_mapping(value: object) -> Mapping[str, Any]:
    data = _mapping_for(value) or {}
    nested = data.get("assertion")
    nested_mapping = _mapping_for(nested) if nested is not None else None
    return nested_mapping or data


def _input_assertions(value: object) -> dict[SpanOffset, Mapping[str, Any]]:
    result: dict[SpanOffset, Mapping[str, Any]] = {}
    for raw in _iter_input_spans(value):
        data = _mapping_for(raw)
        if data is None:
            continue
        offset = _offset_from_mapping(data)
        if offset is None or offset in result:
            continue
        assertion = _assertion_mapping(raw)
        result[offset] = dict(assertion)
    return result


def _explicit_axes(
    item: DischargeProfileItem,
    source_text: str,
    explicit: Mapping[str, Any] | None,
    section: str,
    language: str | None,
) -> DischargeProfileItem:
    target = {"text": source_text[item.start : item.end]}
    assertion = assert_context_axes(target, section=section, language=language)
    values: dict[str, Any] = {
        "certainty": assertion.certainty,
        "negation": assertion.negation,
        "temporality": assertion.temporality,
    }
    if item.status == "unconfirmed":
        values["certainty"] = "uncertain"
    elif item.status == "refuted":
        values["negation"] = "negated"
    if explicit:
        raw_certainty = explicit.get("certainty")
        if raw_certainty is None and isinstance(explicit.get("uncertainty"), bool):
            raw_certainty = "uncertain" if explicit["uncertainty"] else "certain"
        values["certainty"] = _normalize_axis(
            raw_certainty,
            CERTAINTY_VALUES,
            values["certainty"],
        )
        values["negation"] = _normalize_axis(
            explicit.get("negation"),
            NEGATION_VALUES,
            values["negation"],
        )
        values["temporality"] = _normalize_axis(
            explicit.get("temporality"),
            TEMPORALITY_VALUES,
            values["temporality"],
        )
    return replace(item, **values)


def _item_from_mention(
    mention: DischargeMention,
    field_name: Literal["diagnoses", "procedures", "follow_up", "instructions"],
    source_text: str,
    section: DischargeProfileSection | None,
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> DischargeProfileItem:
    section_name = section.header if section is not None else mention.section
    item = DischargeProfileItem(
        text=mention.text,
        value=mention.text,
        span=mention.span,
        field=field_name,
        section=section_name,
        status=mention.status,
    )
    return _explicit_axes(
        item,
        source_text,
        explicit_assertions.get(item.span),
        section_name,
        language,
    )


def _item_from_regimen(
    regimen: MedicationRegimen,
    source_text: str,
    section: DischargeProfileSection | None,
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> DischargeProfileItem:
    section_name = section.header if section is not None else regimen.section
    item = DischargeProfileItem(
        text=regimen.text,
        value=regimen.medication,
        span=regimen.span,
        field="medications",
        section=section_name,
        medication_span=regimen.medication_span,
        sig=regimen.sig,
    )
    explicit = explicit_assertions.get(item.span) or explicit_assertions.get(
        item.medication_span
    )
    return _explicit_axes(item, source_text, explicit, section_name, language)


def _item_from_instruction(
    source_text: str,
    span: SpanOffset,
    section: DischargeProfileSection,
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> DischargeProfileItem:
    item = DischargeProfileItem(
        text=source_text[span[0] : span[1]],
        span=span,
        field="instructions",
        section=section.header,
    )
    return _explicit_axes(
        item,
        source_text,
        explicit_assertions.get(span),
        section.header,
        language,
    )


def _item_from_raw_span(
    source_text: str,
    span: SpanOffset,
    field_name: DischargeProfileField,
    section: DischargeProfileSection,
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> DischargeProfileItem:
    """Build a profile item directly from a bounded section line."""

    text = source_text[span[0] : span[1]]
    if field_name == "medications":
        medication_span = _medication_name_span(source_text, span)
        item = DischargeProfileItem(
            text=text,
            value=source_text[medication_span[0] : medication_span[1]],
            span=span,
            field=field_name,
            section=section.header,
            medication_span=medication_span,
            sig=parse_sig(text),
        )
    else:
        item = DischargeProfileItem(
            text=text,
            span=span,
            field=field_name,
            section=section.header,
        )
    return _explicit_axes(
        item,
        source_text,
        explicit_assertions.get(span),
        section.header,
        language,
    )


def _merge_preferred_items(
    preferred: Iterable[DischargeProfileItem],
    fallback: Iterable[DischargeProfileItem],
) -> tuple[DischargeProfileItem, ...]:
    """Keep preferred structured records and fill only missing field values."""

    result = list(preferred)
    seen = {
        (item.field, re.sub(r"\s+", " ", item.value.casefold()).strip())
        for item in result
    }
    preferred_spans = tuple(item.span for item in result)
    for item in fallback:
        key = (item.field, re.sub(r"\s+", " ", item.value.casefold()).strip())
        overlaps_preferred = any(
            item.start < preferred_end and preferred_start < item.end
            for preferred_start, preferred_end in preferred_spans
        )
        if key not in seen and not overlaps_preferred:
            seen.add(key)
            result.append(item)
    return _deduplicate_items(result)


def _deduplicate_items(
    items: Iterable[DischargeProfileItem],
) -> tuple[DischargeProfileItem, ...]:
    result: list[DischargeProfileItem] = []
    seen: set[tuple[str, str]] = set()
    for item in sorted(items, key=lambda value: (value.start, value.end, value.text)):
        key = (item.field, re.sub(r"\s+", " ", item.value.casefold()).strip())
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return tuple(result)


def _local_field_items(
    source_text: str,
    field_name: DischargeProfileField,
    sections: Iterable[DischargeProfileSection],
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> tuple[DischargeProfileItem, ...]:
    return tuple(
        _item_from_raw_span(
            source_text,
            span,
            field_name,
            section,
            explicit_assertions,
            language,
        )
        for section in sections
        if section.field == field_name
        for span in _section_items(source_text, section)
    )


def _safe_mention_items(
    mentions: Iterable[DischargeMention],
    field_name: Literal["diagnoses", "procedures", "follow_up", "instructions"],
    source_text: str,
    sections: tuple[DischargeProfileSection, ...],
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> tuple[DischargeProfileItem, ...]:
    result: list[DischargeProfileItem] = []
    for mention in mentions:
        section = _section_for_span(sections, mention.start, mention.end)
        if section is None:
            continue
        if field_name == "instructions" and section.field != "instructions":
            continue
        if field_name != "instructions" and section.field == "instructions":
            continue
        result.append(
            _item_from_mention(
                mention,
                field_name,
                source_text,
                section,
                explicit_assertions,
                language,
            )
        )
    return tuple(result)


def _safe_regimen_items(
    regimens: Iterable[MedicationRegimen],
    source_text: str,
    sections: tuple[DischargeProfileSection, ...],
    explicit_assertions: Mapping[SpanOffset, Mapping[str, Any]],
    language: str | None,
) -> tuple[DischargeProfileItem, ...]:
    result: list[DischargeProfileItem] = []
    for regimen in regimens:
        section = _section_for_span(sections, regimen.start, regimen.end)
        if section is None or section.field != "medications":
            continue
        result.append(
            _item_from_regimen(
                regimen,
                source_text,
                section,
                explicit_assertions,
                language,
            )
        )
    return tuple(result)


def extract_discharge_profile(
    text: str,
    entities: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    spans: Iterable[Any] | Mapping[str, Any] | None = None,
    language: str | None = None,
) -> DischargeSummaryProfile:
    """Extract a deterministic, section-scoped discharge profile.

    ``entities``/``spans`` are optional caller-owned model spans.  Their
    assertion fields are preserved when supplied; otherwise the local ConText
    rules annotate the extracted evidence.  The source text is used only while
    extracting and is never written to logs or exceptions by this function.

    Args:
        text: One discharge-summary document.
        entities: Optional existing spans, optionally wrapped under
            ``entities``, ``clinical_entities``, or ``spans``.
        spans: Compatibility alias for ``entities``.
        language: Optional local context and section language.

    Returns:
        A :class:`DischargeSummaryProfile` containing diagnoses, procedures,
        medications, follow-up, and separately recognized instructions.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if entities is not None and spans is not None:
        raise TypeError("pass either entities or spans, not both")

    source_spans = spans if spans is not None else entities
    summary: DischargeSummary = extract_discharge_summary(
        text,
        entities=entities,
        spans=spans,
        language=language,
    )
    sections = _scan_profile_sections(text, language)
    explicit_assertions = _input_assertions(source_spans)

    diagnoses = _merge_preferred_items(
        _safe_mention_items(
            summary.discharge_diagnoses,
            "diagnoses",
            text,
            sections,
            explicit_assertions,
            language,
        ),
        _local_field_items(text, "diagnoses", sections, explicit_assertions, language),
    )
    procedures = _merge_preferred_items(
        _safe_mention_items(
            summary.procedures,
            "procedures",
            text,
            sections,
            explicit_assertions,
            language,
        ),
        _local_field_items(text, "procedures", sections, explicit_assertions, language),
    )
    medications = _merge_preferred_items(
        _safe_regimen_items(
            summary.discharge_medications,
            text,
            sections,
            explicit_assertions,
            language,
        ),
        _local_field_items(
            text, "medications", sections, explicit_assertions, language
        ),
    )
    follow_up = _merge_preferred_items(
        _safe_mention_items(
            summary.follow_up,
            "follow_up",
            text,
            sections,
            explicit_assertions,
            language,
        ),
        _local_field_items(text, "follow_up", sections, explicit_assertions, language),
    )
    instructions = _merge_preferred_items(
        _safe_mention_items(
            summary.follow_up,
            "instructions",
            text,
            sections,
            explicit_assertions,
            language,
        ),
        _local_field_items(
            text, "instructions", sections, explicit_assertions, language
        ),
    )

    return DischargeSummaryProfile(
        diagnoses=diagnoses,
        procedures=procedures,
        medications=medications,
        follow_up=follow_up,
        instructions=instructions,
        document_type=summary.document_type,
        sections=sections,
    )


extract_discharge_summary_profile = extract_discharge_profile
build_discharge_profile = extract_discharge_profile


__all__ = [
    "DISCHARGE_PROFILE_ADVISORY",
    "DISCHARGE_PROFILE_FIELDS",
    "DISCHARGE_PROFILE_NAME",
    "DISCHARGE_PROFILE_SCHEMA_VERSION",
    "DISCHARGE_SUMMARY_PROFILE_ADVISORY",
    "DischargeEvidence",
    "DischargeFieldEvidence",
    "DischargeProfile",
    "DischargeProfileField",
    "DischargeProfileItem",
    "DischargeProfileSection",
    "DischargeSummaryProfile",
    "build_discharge_profile",
    "extract_discharge_profile",
    "extract_discharge_summary_profile",
]
