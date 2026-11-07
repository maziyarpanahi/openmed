"""Deterministic, review-first discharge-summary extraction.

The extractor composes the local document-type router, section detector, list
parser, problem-list reconciliation, medication candidate filtering, and sig
normalization layers.  It deliberately has no model or network dependency: a
caller may provide model-produced spans and a local grounder, but the default
path remains useful for synthetic and rules-first discharge summaries.

Every emitted item retains half-open source offsets.  The result is a review
aid only; it is not automated medication reconciliation or a medical device.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from openmed.clinical.exporters.fhir import (
    MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
    to_bundle,
    to_fhir,
)
from openmed.clinical.grounding.types import Candidate, GroundedSpan
from openmed.clinical.medication_sig import filter_medication_candidates
from openmed.clinical.problem_list import (
    ProblemMention,
    ReconciledProblem,
    deduplicate_problem_list,
)
from openmed.clinical.sections import (
    UNSECTIONED_SECTION,
    classify_document,
    detect_sections,
)
from openmed.clinical.sig_parser import Sig, parse_sig
from openmed.processing.lists import parse_lists

SpanOffset = tuple[int, int]
DischargeFieldName = Literal[
    "discharge_diagnoses",
    "procedures",
    "discharge_medications",
    "follow_up",
]

DISCHARGE_SUMMARY_ADVISORY = (
    "Discharge-summary extraction is a review aid, not automated medication "
    "reconciliation or a medical device; qualified clinical review is required."
)
DISCHARGE_SUMMARY_REVIEW_ADVISORY = DISCHARGE_SUMMARY_ADVISORY
MEDICATION_DEVICE_REVIEW_AID_ADVISORY = DISCHARGE_SUMMARY_ADVISORY

DISCHARGE_FIELD_NAMES: tuple[DischargeFieldName, ...] = (
    "discharge_diagnoses",
    "procedures",
    "discharge_medications",
    "follow_up",
)

_SECTION_ALIASES: Mapping[DischargeFieldName, tuple[str, ...]] = {
    "discharge_diagnoses": (
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
        "admission diagnoses",
        "admission diagnosis",
        "admitting diagnoses",
        "admitting diagnosis",
        "diagnoses",
        "diagnosis",
        "problems at discharge",
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
    "discharge_medications": (
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
    ),
    "follow_up": (
        "follow-up instructions",
        "follow up instructions",
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
        "discharge instructions",
        "plan",
    ),
}

_ALIAS_TO_FIELD = {
    re.sub(r"[^a-z0-9]+", " ", alias.casefold()).strip(): field_name
    for field_name, aliases in _SECTION_ALIASES.items()
    for alias in aliases
}
_HEADER_DELIMITER_RE = re.compile(r"(?::|：|﹕|꞉|\s+[—–-]\s+)")
_LIST_PREFIX_RE = re.compile(r"^(?:[-*•]|\d+[.)]|[A-Za-z][.)])\s+")
_MEDICATION_DOSE_RE = re.compile(
    r"(?<!\w)(?:\d+(?:\.\d+)?|\.\d+)\s*"
    r"(?:mg|mcg|g|kg|ml|l|unit|units|meq|iu|%|tablet|tablets|tab|tabs|"
    r"capsule|capsules|cap|caps|puff|puffs|drop|drops|spray|sprays)\b",
    re.IGNORECASE,
)
_MEDICATION_ROUTE_RE = re.compile(
    r"(?<!\w)(?:po|iv|im|sc|sq|subq|oral|intravenous|intramuscular|"
    r"subcutaneous|topical|inhaled|by mouth|per os)(?!\w)",
    re.IGNORECASE,
)
_MEDICATION_FREQUENCY_RE = re.compile(
    r"(?<!\w)(?:qd|daily|once daily|once a day|bid|twice daily|"
    r"tid|three times daily|qid|four times daily|qhs|nightly|weekly|"
    r"q\s*\d+\s*(?:h|hr|hrs|hour|hours|d|day|days|wk|week|weeks)|"
    r"as needed|when needed|if needed|prn)(?!\w)",
    re.IGNORECASE,
)
_MEDICATION_PREFIX_RE = re.compile(
    r"^(?:take|start|continue|resume|restart|begin|use|hold|stop)\s+",
    re.IGNORECASE,
)
_NON_VALUE_RE = re.compile(
    r"^(?:none|no(?:ne)?|n/?a|not applicable|no medications?|"
    r"no procedures?|no diagnoses?|noncontributory|see above)\.?$",
    re.IGNORECASE,
)

_CONDITION_LABELS = frozenset(
    {"CONDITION", "DIAGNOSIS", "DISEASE", "FINDING", "PROBLEM"}
)
_PROCEDURE_LABELS = frozenset({"PROCEDURE", "OPERATION", "SURGERY", "INTERVENTION"})
_MEDICATION_LABELS = frozenset({"MEDICATION", "DRUG", "CHEMICAL", "RX"})
_FOLLOW_UP_LABELS = frozenset(
    {"FOLLOW_UP", "FOLLOWUP", "APPOINTMENT", "INSTRUCTION", "DISPOSITION"}
)


@dataclass(frozen=True)
class DischargeSectionSpan:
    """A discharge field section and its source offsets."""

    field: DischargeFieldName
    header: str
    start: int
    end: int
    content_start: int
    content_end: int

    def __post_init__(self) -> None:
        _validate_offset(self.start, self.end, "section span")
        if not self.start <= self.content_start <= self.content_end <= self.end:
            raise ValueError("section content offsets must stay within its span")
        if not self.header.strip():
            raise ValueError("section header must not be empty")

    @property
    def span(self) -> SpanOffset:
        """Return the complete section range."""

        return self.start, self.end

    @property
    def content_span(self) -> SpanOffset:
        """Return the section-body range."""

        return self.content_start, self.content_end

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready provenance without requiring source text."""

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
class DischargeMention:
    """One diagnosis, procedure, or follow-up mention with provenance."""

    text: str
    span: SpanOffset
    field: Literal["discharge_diagnoses", "procedures", "follow_up"]
    section: str
    status: str | None = None
    grounded: GroundedSpan | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("discharge mention text must not be empty")
        _validate_offset(*self.span, "mention span")
        if not self.section.strip():
            raise ValueError("discharge mention section must not be empty")

    @property
    def start(self) -> int:
        """Return the inclusive source start."""

        return self.span[0]

    @property
    def end(self) -> int:
        """Return the exclusive source end."""

        return self.span[1]

    @property
    def source_span(self) -> SpanOffset:
        """Return the source span alias used by review clients."""

        return self.span

    @property
    def provenance(self) -> dict[str, int]:
        """Return a JSON-ready span provenance mapping."""

        return {"start": self.start, "end": self.end}

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access for lightweight pipeline consumers."""

        if key in {"span", "source_span"}:
            return self.span
        if key == "provenance":
            return self.provenance
        if key == "kind":
            return self.field
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """Return the mention with explicit offset provenance."""

        payload: dict[str, Any] = {
            "text": self.text,
            "field": self.field,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "span": {"start": self.start, "end": self.end},
            "provenance": self.provenance,
        }
        if self.status is not None:
            payload["status"] = self.status
        return payload


@dataclass(frozen=True)
class MedicationRegimen:
    """One discharge medication reconciled with its structured sig."""

    medication: str
    text: str
    span: SpanOffset
    medication_span: SpanOffset
    sig: Sig
    section: str
    reconciled: bool = True
    grounded: GroundedSpan | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self.medication.strip() or not self.text.strip():
            raise ValueError("medication and regimen text must not be empty")
        _validate_offset(*self.span, "medication regimen span")
        _validate_offset(*self.medication_span, "medication span")
        if not (
            self.span[0] <= self.medication_span[0]
            and self.medication_span[1] <= self.span[1]
        ):
            raise ValueError("medication span must be contained by regimen span")
        if not self.section.strip():
            raise ValueError("medication section must not be empty")

    @property
    def start(self) -> int:
        """Return the full regimen start offset."""

        return self.span[0]

    @property
    def end(self) -> int:
        """Return the full regimen end offset."""

        return self.span[1]

    @property
    def source_span(self) -> SpanOffset:
        """Return the full regimen source span."""

        return self.span

    @property
    def provenance(self) -> dict[str, Any]:
        """Return full-regimen and medication-name source offsets."""

        return {
            "start": self.start,
            "end": self.end,
            "medication_start": self.medication_span[0],
            "medication_end": self.medication_span[1],
        }

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access for medication-list consumers."""

        if key in {"span", "source_span"}:
            return self.span
        if key == "provenance":
            return self.provenance
        if key in {"regimen", "regimen_text"}:
            return self.text
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """Return the structured regimen and both provenance ranges."""

        return {
            "medication": self.medication,
            "text": self.text,
            "section": self.section,
            "start": self.start,
            "end": self.end,
            "span": {"start": self.start, "end": self.end},
            "medication_span": {
                "start": self.medication_span[0],
                "end": self.medication_span[1],
            },
            "provenance": self.provenance,
            "sig": dict(self.sig),
            "reconciled": self.reconciled,
        }


@dataclass(frozen=True)
class DischargeSummary:
    """Typed discharge-summary record for human review and export."""

    discharge_diagnoses: tuple[DischargeMention, ...]
    procedures: tuple[DischargeMention, ...]
    discharge_medications: tuple[MedicationRegimen, ...]
    follow_up: tuple[DischargeMention, ...]
    document_type: str
    classification: Mapping[str, Any] = field(default_factory=dict)
    sections: tuple[DischargeSectionSpan, ...] = ()
    advisory: str = DISCHARGE_SUMMARY_ADVISORY

    def __post_init__(self) -> None:
        object.__setattr__(self, "discharge_diagnoses", tuple(self.discharge_diagnoses))
        object.__setattr__(self, "procedures", tuple(self.procedures))
        object.__setattr__(
            self,
            "discharge_medications",
            tuple(self.discharge_medications),
        )
        object.__setattr__(self, "follow_up", tuple(self.follow_up))
        object.__setattr__(self, "sections", tuple(self.sections))
        object.__setattr__(self, "classification", dict(self.classification))
        if not self.advisory.strip():
            raise ValueError("discharge-summary advisory must not be empty")

    @property
    def record_type(self) -> str:
        """Return the interoperability-facing record name."""

        return "DischargeSummary"

    @property
    def note_type(self) -> str:
        """Return the routed document type."""

        return self.document_type

    @property
    def section_spans(self) -> dict[str, list[dict[str, Any]]]:
        """Return field-indexed section provenance."""

        result: dict[str, list[dict[str, Any]]] = {
            field_name: [] for field_name in DISCHARGE_FIELD_NAMES
        }
        for section in self.sections:
            result[section.field].append(section.to_dict())
        return result

    @property
    def field_provenance(self) -> dict[str, list[dict[str, int]]]:
        """Return every extracted field item's source offsets."""

        return {
            "discharge_diagnoses": [
                item.provenance for item in self.discharge_diagnoses
            ],
            "procedures": [item.provenance for item in self.procedures],
            "discharge_medications": [
                item.provenance for item in self.discharge_medications
            ],
            "follow_up": [item.provenance for item in self.follow_up],
        }

    @property
    def follow_up_text(self) -> str:
        """Return follow-up text joined in source order."""

        return " ".join(item.text for item in self.follow_up)

    @property
    def summary_card(self) -> Any:
        """Return a PHI-free aggregate card for the extracted fields."""

        from .summary_card import ClinicalSummaryCard

        return ClinicalSummaryCard(
            problems=len(self.discharge_diagnoses),
            medications=len(self.discharge_medications),
            procedures=len(self.procedures),
            section_count=len(self.sections),
        )

    def __getitem__(self, key: str) -> Any:
        """Provide mapping-style access for JSON-oriented callers."""

        if key == "record_type":
            return self.record_type
        if key in {"section_spans", "sections"}:
            return self.section_spans if key == "section_spans" else self.sections
        if key in {"provenance", "field_provenance"}:
            return self.field_provenance
        return getattr(self, key)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete structured record with provenance."""

        return {
            "record_type": self.record_type,
            "document_type": self.document_type,
            "classification": dict(self.classification),
            "discharge_diagnoses": [
                item.to_dict() for item in self.discharge_diagnoses
            ],
            "procedures": [item.to_dict() for item in self.procedures],
            "discharge_medications": [
                item.to_dict() for item in self.discharge_medications
            ],
            "follow_up": [item.to_dict() for item in self.follow_up],
            "sections": self.section_spans,
            "field_provenance": self.field_provenance,
            "advisory": self.advisory,
        }

    def to_json(self) -> str:
        """Serialize the record deterministically."""

        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    def to_fhir(
        self,
        *,
        subject_reference: str = "Patient/openmed-subject",
        document_id: str = "openmed-discharge-summary",
        bundle: bool = True,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Export this record as FHIR resources or a transaction Bundle."""

        return export_discharge_summary_fhir(
            self,
            subject_reference=subject_reference,
            document_id=document_id,
            bundle=bundle,
        )


# Descriptive aliases keep the public record vocabulary discoverable without
# creating duplicate models.
DischargeSummaryRecord = DischargeSummary
DischargeDiagnosis = DischargeMention
DischargeProcedure = DischargeMention
FollowUpInstruction = DischargeMention


@dataclass(frozen=True)
class _SectionHit:
    field: DischargeFieldName
    header: str
    start: int
    header_start: int
    header_end: int
    content_start: int


@dataclass(frozen=True)
class _TextUnit:
    text: str
    start: int
    end: int

    @property
    def span(self) -> SpanOffset:
        """Return the unit's half-open source span."""

        return self.start, self.end


@dataclass(frozen=True)
class _InputEntity:
    start: int
    end: int
    label: str
    grounded: GroundedSpan | None = None
    score: float = 1.0


def extract_discharge_summary(
    text: str,
    entities: Iterable[Any] | Mapping[str, Any] | None = None,
    *,
    spans: Iterable[Any] | Mapping[str, Any] | None = None,
    language: str | None = None,
    use_learned: bool = False,
    learned_head: Any | None = None,
    model_path: str | None = None,
    grounder: Callable[[str], Sequence[Candidate]] | None = None,
) -> DischargeSummary:
    """Extract a typed discharge summary from one source document.

    Args:
        text: Source discharge-summary text.
        entities: Optional existing model spans.  Mappings may also wrap spans
            under ``entities``, ``clinical_entities``, or ``spans``.
        spans: Compatibility alias for ``entities``.
        language: Optional section language forwarded to local detection.
        use_learned: Opt-in local section-boundary refinement.
        learned_head: Optional caller-supplied local learned section head.
        model_path: Optional local learned section artifact path.
        grounder: Optional caller-supplied offline callable returning
            :class:`Candidate` values for FHIR export.

    Returns:
        A deterministic :class:`DischargeSummary`.  Missing sections produce
        empty tuples rather than guessed values.

    Raises:
        TypeError: If ``text`` is not a string or the span aliases conflict.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if entities is not None and spans is not None:
        raise TypeError("pass either entities or spans, not both")

    routed = classify_document(text)
    detected = detect_sections(
        text,
        language=language,
        use_learned=use_learned,
        learned_head=learned_head,
        model_path=model_path,
    )
    sections = _discharge_sections(
        text,
        detected,
        is_discharge=routed["type"] == "discharge_summary",
    )
    input_entities = _coerce_entities(
        spans if spans is not None else entities,
        text,
    )

    diagnoses = _extract_diagnoses(text, sections, input_entities, grounder)
    procedures = _extract_procedures(text, sections, input_entities, grounder)
    medications = _extract_medications(text, sections, input_entities, grounder)
    follow_up = _extract_follow_up(text, sections, input_entities)

    return DischargeSummary(
        discharge_diagnoses=tuple(diagnoses),
        procedures=tuple(procedures),
        discharge_medications=tuple(medications),
        follow_up=tuple(follow_up),
        document_type=str(routed["type"]),
        classification=routed,
        sections=sections,
    )


def _discharge_sections(
    text: str,
    detected: Sequence[Mapping[str, Any]],
    *,
    is_discharge: bool,
) -> tuple[DischargeSectionSpan, ...]:
    hits = list(_scan_discharge_headers(text))
    explicit_fields = {hit.field for hit in hits}

    # The generic detector has no reason to treat every "Medications" heading
    # as a discharge list.  It is accepted only after the note-type router has
    # identified a discharge summary, and only when no explicit discharge-med
    # heading is available.  The same bridge makes the existing section
    # detector useful for common problem-list and plan headings.
    if is_discharge:
        for section in detected:
            label = str(section.get("label", ""))
            field_name = _generic_discharge_field(label)
            if field_name is None or field_name in explicit_fields:
                continue
            header = str(section.get("header", label.replace("_", " ")))
            if field_name == "discharge_medications" and _normalize_alias(header) in {
                "current medications",
                "home medications",
                "medication list",
                "meds",
            }:
                continue
            start = _integer_field(section, "start")
            end = _integer_field(section, "end")
            if label == UNSECTIONED_SECTION or start is None or end is None:
                continue
            content_start = section.get("content_start", start)
            if not isinstance(content_start, int):
                content_start = start
            hits.append(
                _SectionHit(
                    field=field_name,
                    header=header,
                    start=start,
                    header_start=start,
                    header_end=min(start + len(header), end),
                    content_start=max(start, min(content_start, end)),
                )
            )

    if not hits:
        return ()

    # Use every detector boundary as a stop point so a discharge field cannot
    # absorb a following generic section (for example current medications).
    boundaries = sorted(
        {
            int(section["start"])
            for section in detected
            if isinstance(section.get("start"), int)
        }
        | {hit.start for hit in hits}
    )
    result: list[DischargeSectionSpan] = []
    seen_starts: set[tuple[int, DischargeFieldName]] = set()
    for hit in sorted(hits, key=lambda item: (item.start, item.header_end)):
        key = (hit.start, hit.field)
        if key in seen_starts:
            continue
        seen_starts.add(key)
        next_boundary = next(
            (boundary for boundary in boundaries if boundary > hit.start),
            len(text),
        )
        end = min(len(text), next_boundary)
        if end <= hit.start:
            continue
        content_start = min(max(hit.content_start, hit.start), end)
        result.append(
            DischargeSectionSpan(
                field=hit.field,
                header=hit.header,
                start=hit.start,
                end=end,
                content_start=content_start,
                content_end=end,
            )
        )
    return tuple(result)


def _scan_discharge_headers(text: str) -> tuple[_SectionHit, ...]:
    aliases = sorted(_ALIAS_TO_FIELD, key=lambda value: (-len(value), value))
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
        normalized = _normalize_alias(header)
        if normalized in aliases:
            hits.append(
                _SectionHit(
                    field=_ALIAS_TO_FIELD[normalized],
                    header=header,
                    start=line_start,
                    header_start=candidate_start,
                    header_end=candidate_start + len(header),
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
        normalized = _normalize_alias(header)
        if normalized in _ALIAS_TO_FIELD:
            content_start = (
                cursor + leading + delimiter.end()
                if delimiter is not None
                else len(text)
            )
            hits.append(
                _SectionHit(
                    field=_ALIAS_TO_FIELD[normalized],
                    header=header,
                    start=cursor,
                    header_start=cursor + leading,
                    header_end=cursor + leading + len(header),
                    content_start=min(content_start, len(text)),
                )
            )
    return tuple(hits)


def _generic_discharge_field(label: str) -> DischargeFieldName | None:
    normalized = _normalize_alias(label)
    if normalized in {"problem list", "active problems", "problems"}:
        return "discharge_diagnoses"
    if normalized in {"medications", "current medications", "home medications"}:
        return "discharge_medications"
    if normalized in {"plan", "assessment plan", "assessment and plan"}:
        return "follow_up"
    return None


def _extract_diagnoses(
    text: str,
    sections: Sequence[DischargeSectionSpan],
    entities: Sequence[_InputEntity],
    grounder: Callable[[str], Sequence[Candidate]] | None,
) -> list[DischargeMention]:
    field_sections = [
        section for section in sections if section.field == "discharge_diagnoses"
    ]
    entity_items = [
        entity
        for entity in entities
        if _label_family(entity.label) in _CONDITION_LABELS
        and _in_sections(entity, field_sections)
    ]
    if entity_items:
        mentions = _reconcile_entity_diagnoses(text, entity_items)
        return [
            _mention_with_grounding(mention, text, grounder) for mention in mentions
        ]

    raw_mentions: list[ProblemMention] = []
    source_by_offset: dict[SpanOffset, tuple[str, str]] = {}
    for section in field_sections:
        for unit in _section_units(text, section):
            if _is_non_value(unit.text):
                continue
            raw_mentions.append(ProblemMention(text=unit.text, offset=unit.span))
            source_by_offset[unit.span] = (unit.text, text[section.start : section.end])
    reconciled = deduplicate_problem_list(raw_mentions)
    return [
        _mention_from_problem(
            problem,
            text,
            source_by_offset,
            field="discharge_diagnoses",
            section=_section_label_for_offset(problem.source_offsets, field_sections),
            grounder=grounder,
        )
        for problem in reconciled
    ]


def _reconcile_entity_diagnoses(
    text: str,
    entities: Sequence[_InputEntity],
) -> list[DischargeMention]:
    mentions = [
        ProblemMention(
            text=text[entity.start : entity.end],
            offset=(entity.start, entity.end),
            system=_first_candidate_value(entity.grounded, "system"),
            code=_first_candidate_value(entity.grounded, "code"),
        )
        for entity in entities
    ]
    by_offset = {(entity.start, entity.end): entity for entity in entities}
    reconciled = deduplicate_problem_list(mentions)
    result: list[DischargeMention] = []
    for problem in reconciled:
        offset = problem.source_offsets[0] if problem.source_offsets else None
        if offset is None:
            continue
        entity = by_offset.get(offset)
        result.append(
            DischargeMention(
                text=text[offset[0] : offset[1]],
                span=offset,
                field="discharge_diagnoses",
                section="discharge diagnoses",
                status=problem.clinical_status,
                grounded=entity.grounded if entity else None,
            )
        )
    return result


def _mention_from_problem(
    problem: ReconciledProblem,
    text: str,
    source_by_offset: Mapping[SpanOffset, tuple[str, str]],
    *,
    field: Literal["discharge_diagnoses", "procedures", "follow_up"],
    section: str,
    grounder: Callable[[str], Sequence[Candidate]] | None,
) -> DischargeMention:
    offset = problem.source_offsets[0] if problem.source_offsets else (0, 0)
    surface = source_by_offset.get(offset, (text[offset[0] : offset[1]], ""))[0]
    grounded = _grounded_span(
        text,
        surface,
        offset,
        "CONDITION",
        grounder,
    )
    return DischargeMention(
        text=surface,
        span=offset,
        field=field,
        section=section or "discharge diagnoses",
        status=problem.clinical_status,
        grounded=grounded,
    )


def _extract_procedures(
    text: str,
    sections: Sequence[DischargeSectionSpan],
    entities: Sequence[_InputEntity],
    grounder: Callable[[str], Sequence[Candidate]] | None,
) -> list[DischargeMention]:
    field_sections = [section for section in sections if section.field == "procedures"]
    entity_items = [
        entity
        for entity in entities
        if _label_family(entity.label) in _PROCEDURE_LABELS
        and _in_sections(entity, field_sections)
    ]
    if entity_items:
        candidates = [
            DischargeMention(
                text=text[entity.start : entity.end],
                span=(entity.start, entity.end),
                field="procedures",
                section="procedures",
                grounded=_grounded_span(
                    text,
                    text[entity.start : entity.end],
                    (entity.start, entity.end),
                    "PROCEDURE",
                    grounder,
                    existing=entity.grounded,
                ),
            )
            for entity in entity_items
        ]
    else:
        candidates = [
            DischargeMention(
                text=unit.text,
                span=unit.span,
                field="procedures",
                section=section.header,
                grounded=_grounded_span(
                    text, unit.text, unit.span, "PROCEDURE", grounder
                ),
            )
            for section in field_sections
            for unit in _section_units(text, section)
            if not _is_non_value(unit.text)
        ]
    return _deduplicate_mentions(candidates)


def _extract_medications(
    text: str,
    sections: Sequence[DischargeSectionSpan],
    entities: Sequence[_InputEntity],
    grounder: Callable[[str], Sequence[Candidate]] | None,
) -> list[MedicationRegimen]:
    field_sections = [
        section for section in sections if section.field == "discharge_medications"
    ]
    entity_items = [
        entity
        for entity in entities
        if _label_family(entity.label) in _MEDICATION_LABELS
        and _in_sections(entity, field_sections)
    ]
    if entity_items:
        accepted = filter_medication_candidates(
            text,
            [
                {
                    "label": "MEDICATION",
                    "start": entity.start,
                    "end": entity.end,
                    "score": max(entity.score, 0.75),
                }
                for entity in entity_items
            ],
        )
        by_offset = {(entity.start, entity.end): entity for entity in entity_items}
        regimens: list[MedicationRegimen] = []
        for candidate in accepted:
            if candidate.start is None or candidate.end is None:
                continue
            entity = by_offset.get((candidate.start, candidate.end))
            section = _containing_section(
                candidate.start,
                candidate.end,
                field_sections,
            )
            if section is None:
                continue
            unit = _unit_for_offset(text, section, candidate.start, candidate.end)
            regimen_span = (
                unit.span if unit is not None else (candidate.start, candidate.end)
            )
            regimen_text = text[regimen_span[0] : regimen_span[1]]
            medication_span = (candidate.start, candidate.end)
            regimens.append(
                _build_regimen(
                    text,
                    regimen_text,
                    regimen_span,
                    medication_span,
                    section,
                    grounder,
                    existing=entity.grounded if entity else None,
                )
            )
    else:
        regimens = []
        for section in field_sections:
            for unit in _section_units(text, section):
                if _is_non_value(unit.text):
                    continue
                medication_span = _medication_name_span(unit.text, unit.start)
                regimens.append(
                    _build_regimen(
                        text,
                        unit.text,
                        unit.span,
                        medication_span,
                        section,
                        grounder,
                    )
                )
    return _deduplicate_regimens(regimens)


def _extract_follow_up(
    text: str,
    sections: Sequence[DischargeSectionSpan],
    entities: Sequence[_InputEntity],
) -> list[DischargeMention]:
    field_sections = [section for section in sections if section.field == "follow_up"]
    entity_items = [
        entity
        for entity in entities
        if _label_family(entity.label) in _FOLLOW_UP_LABELS
        and _in_sections(entity, field_sections)
    ]
    if entity_items:
        return [
            DischargeMention(
                text=text[entity.start : entity.end],
                span=(entity.start, entity.end),
                field="follow_up",
                section="follow-up",
            )
            for entity in entity_items
        ]
    return [
        DischargeMention(
            text=unit.text,
            span=unit.span,
            field="follow_up",
            section=text[section.start : section.end].splitlines()[0].strip()
            if text[section.start : section.end].splitlines()
            else "follow-up",
        )
        for section in field_sections
        for unit in _section_units(text, section)
        if not _is_non_value(unit.text)
    ]


def _section_units(text: str, section: DischargeSectionSpan) -> tuple[_TextUnit, ...]:
    source_start = section.content_start
    source = text[source_start : section.content_end]
    parsed = parse_lists(source)
    units: list[_TextUnit] = []
    for item in parsed:
        if item.nesting_level != 0:
            continue
        relative_start = (
            item.content_start if item.content_start is not None else item.start
        )
        relative_end = item.end
        unit = _trimmed_unit(
            text,
            source_start + relative_start,
            source_start + relative_end,
        )
        if unit is not None:
            units.append(unit)
    if units:
        return tuple(units)

    line_units = [
        _trimmed_unit(text, source_start + match.start(), source_start + match.end())
        for match in re.finditer(r"[^\r\n]+", source)
    ]
    line_units = [unit for unit in line_units if unit is not None]
    if len(line_units) > 1:
        return tuple(_remove_list_prefix(text, unit) for unit in line_units)
    if not line_units:
        return ()

    only = line_units[0]
    if ";" not in only.text:
        return (_remove_list_prefix(text, only),)
    # Build actual pieces from semicolon boundaries while retaining source
    # offsets for each item.
    parts: list[_TextUnit] = []
    pieces = re.split(r"\s*;\s*", only.text)
    parts = []
    cursor = only.start
    for piece in pieces:
        if not piece.strip():
            cursor += len(piece) + 1
            continue
        start = text.find(piece, cursor, only.end)
        if start < 0:
            start = cursor
        unit = _trimmed_unit(text, start, start + len(piece))
        if unit is not None:
            parts.append(unit)
        cursor = start + len(piece) + 1
    return tuple(_remove_list_prefix(text, unit) for unit in parts)


def _build_regimen(
    text: str,
    regimen_text: str,
    regimen_span: SpanOffset,
    medication_span: SpanOffset,
    section: DischargeSectionSpan,
    grounder: Callable[[str], Sequence[Candidate]] | None,
    *,
    existing: GroundedSpan | None = None,
) -> MedicationRegimen:
    medication = text[medication_span[0] : medication_span[1]].strip()
    sig = parse_sig(regimen_text)
    grounded = _grounded_span(
        text,
        medication,
        medication_span,
        "MEDICATION",
        grounder,
        existing=existing,
    )
    return MedicationRegimen(
        medication=medication,
        text=text[regimen_span[0] : regimen_span[1]].strip(),
        span=regimen_span,
        medication_span=medication_span,
        sig=sig,
        section=section.header,
        grounded=grounded,
    )


def _medication_name_span(item_text: str, absolute_start: int) -> SpanOffset:
    value = _MEDICATION_PREFIX_RE.sub("", item_text.strip(), count=1)
    local_prefix = len(item_text) - len(item_text.lstrip())
    local_prefix += len(item_text[local_prefix:]) - len(
        item_text[local_prefix:].lstrip()
    )
    prefix_match = _MEDICATION_PREFIX_RE.match(item_text[local_prefix:])
    if prefix_match is not None:
        local_prefix += prefix_match.end()
    candidates = [
        match
        for pattern in (
            _MEDICATION_DOSE_RE,
            _MEDICATION_ROUTE_RE,
            _MEDICATION_FREQUENCY_RE,
        )
        if (match := pattern.search(value)) is not None
    ]
    end = min((match.start() for match in candidates), default=len(value))
    name = value[:end].strip(" \t,;:-")
    if not name:
        name = value.strip()
        end = len(value)
    name_start_in_value = value.find(name)
    start = absolute_start + local_prefix + max(name_start_in_value, 0)
    return start, start + len(name)


def _grounded_span(
    text: str,
    surface: str,
    span: SpanOffset,
    label: str,
    grounder: Callable[[str], Sequence[Candidate]] | None,
    *,
    existing: GroundedSpan | None = None,
) -> GroundedSpan:
    if existing is not None:
        return existing
    candidates = tuple(grounder(surface)) if grounder is not None else ()
    if any(not isinstance(candidate, Candidate) for candidate in candidates):
        raise TypeError("grounder must return Candidate values")
    return GroundedSpan(
        text=surface,
        start=span[0],
        end=span[1],
        candidates=candidates,
        canonical_label=label,
    )


def _coerce_entities(
    value: Iterable[Any] | Mapping[str, Any] | None,
    text: str,
) -> tuple[_InputEntity, ...]:
    items = _iter_entity_values(value)
    result: list[_InputEntity] = []
    for item in items:
        if isinstance(item, GroundedSpan):
            start, end = item.start, item.end
            label = item.canonical_label or ""
            score = item.score
            grounded = item
        else:
            data = _entity_mapping(item)
            offset = _entity_offset(data)
            if offset is None:
                continue
            start, end = offset
            label = _entity_label(data)
            score = _float_value(data.get("score", data.get("confidence", 1.0)), 1.0)
            grounded = _grounded_from_mapping(data, text, start, end, label)
        if start < 0 or end <= start or end > len(text) or not label:
            continue
        result.append(
            _InputEntity(
                start=start,
                end=end,
                label=_label_family(label),
                grounded=grounded,
                score=score,
            )
        )
    return tuple(
        sorted(result, key=lambda entity: (entity.start, entity.end, entity.label))
    )


def _iter_entity_values(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        for key in ("entities", "clinical_entities", "spans"):
            if key in value:
                return _iter_entity_values(value[key])
        return (value,)
    if isinstance(value, (str, bytes)):
        return ()
    if isinstance(value, GroundedSpan) or all(
        hasattr(value, name) for name in ("start", "end")
    ):
        return (value,)
    try:
        return tuple(value)
    except TypeError:
        return (value,)


def _entity_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        mapped = to_dict()
        if isinstance(mapped, Mapping):
            return dict(mapped)
    return {
        name: getattr(value, name)
        for name in (
            "text",
            "label",
            "entity",
            "canonical_label",
            "entity_type",
            "start",
            "end",
            "score",
            "confidence",
            "candidates",
            "coding",
            "codings",
        )
        if hasattr(value, name)
    }


def _entity_offset(data: Mapping[str, Any]) -> SpanOffset | None:
    raw_offsets = data.get("offsets")
    if isinstance(raw_offsets, (tuple, list)) and len(raw_offsets) == 2:
        start, end = raw_offsets
    else:
        start, end = data.get("start"), data.get("end")
    if not isinstance(start, int) or isinstance(start, bool):
        return None
    if not isinstance(end, int) or isinstance(end, bool):
        return None
    return start, end


def _entity_label(data: Mapping[str, Any]) -> str:
    for key in ("canonical_label", "label", "entity", "entity_group", "type"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _grounded_from_mapping(
    data: Mapping[str, Any],
    text: str,
    start: int,
    end: int,
    label: str,
) -> GroundedSpan | None:
    raw_candidates = data.get("candidates", ())
    if isinstance(raw_candidates, Mapping):
        raw_candidates = (raw_candidates,)
    candidates: list[Candidate] = []
    if isinstance(raw_candidates, Iterable) and not isinstance(
        raw_candidates, (str, bytes)
    ):
        for raw in raw_candidates:
            if isinstance(raw, Candidate):
                candidates.append(raw)
            elif isinstance(raw, Mapping):
                candidate = _candidate_from_mapping(raw)
                if candidate is not None:
                    candidates.append(candidate)
    if not candidates:
        for key in ("coding", "codeable_concept"):
            raw = data.get(key)
            if isinstance(raw, Mapping):
                candidate = _candidate_from_mapping(raw)
                if candidate is not None:
                    candidates.append(candidate)
    if not candidates:
        return None
    return GroundedSpan(
        text=text[start:end],
        start=start,
        end=end,
        candidates=tuple(candidates),
        canonical_label=_label_family(label),
    )


def _candidate_from_mapping(data: Mapping[str, Any]) -> Candidate | None:
    system = data.get("system")
    code = data.get("code")
    if not isinstance(system, str) or not isinstance(code, str):
        return None
    return Candidate(
        system=system,
        code=code,
        display=str(data.get("display", code)),
        score=_float_value(data.get("score", 1.0), 1.0),
        source=str(data.get("source", "caller")),
        matched_alias=(
            str(data["matched_alias"]) if data.get("matched_alias") else None
        ),
        match_kind=(str(data["match_kind"]) if data.get("match_kind") else None),
        vocab_version=(
            str(data["vocab_version"]) if data.get("vocab_version") else None
        ),
    )


def _in_sections(
    entity: _InputEntity, sections: Sequence[DischargeSectionSpan]
) -> bool:
    return any(
        section.content_start <= entity.start and entity.end <= section.content_end
        for section in sections
    )


def _containing_section(
    start: int,
    end: int,
    sections: Sequence[DischargeSectionSpan],
) -> DischargeSectionSpan | None:
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


def _unit_for_offset(
    text: str,
    section: DischargeSectionSpan,
    start: int,
    end: int,
) -> _TextUnit | None:
    units = _section_units(text, section)
    return min(
        (unit for unit in units if unit.start <= start and end <= unit.end),
        key=lambda unit: (unit.end - unit.start, unit.start),
        default=None,
    )


def _section_label_for_offset(
    offsets: Sequence[SpanOffset],
    sections: Sequence[DischargeSectionSpan],
) -> str:
    if not offsets:
        return "discharge diagnoses"
    start = offsets[0][0]
    section = min(
        (section for section in sections if section.start <= start < section.end),
        key=lambda section: (section.end - section.start, section.start),
        default=None,
    )
    return section.header if section is not None else "discharge diagnoses"


def _trimmed_unit(text: str, start: int, end: int) -> _TextUnit | None:
    start = max(0, start)
    end = min(len(text), end)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    return _TextUnit(text=text[start:end], start=start, end=end)


def _remove_list_prefix(text: str, unit: _TextUnit) -> _TextUnit:
    prefix = _LIST_PREFIX_RE.match(unit.text)
    if prefix is None:
        return unit
    start = unit.start + prefix.end()
    return _trimmed_unit(text, start, unit.end) or unit


def _deduplicate_mentions(
    mentions: Iterable[DischargeMention],
) -> list[DischargeMention]:
    result: list[DischargeMention] = []
    seen: set[tuple[str, str]] = set()
    for mention in sorted(mentions, key=lambda item: (item.start, item.end, item.text)):
        key = (mention.field, _normalize_value(mention.text))
        if key in seen:
            continue
        seen.add(key)
        result.append(mention)
    return result


def _deduplicate_regimens(
    regimens: Iterable[MedicationRegimen],
) -> list[MedicationRegimen]:
    result: list[MedicationRegimen] = []
    seen: set[str] = set()
    for regimen in sorted(regimens, key=lambda item: (item.start, item.end)):
        key = _normalize_value(regimen.medication)
        if key in seen:
            continue
        seen.add(key)
        result.append(regimen)
    return result


def _mention_with_grounding(
    mention: DischargeMention,
    text: str,
    grounder: Callable[[str], Sequence[Candidate]] | None,
) -> DischargeMention:
    if mention.grounded is not None or grounder is None:
        return mention
    return replace(
        mention,
        grounded=_grounded_span(
            text,
            mention.text,
            mention.span,
            "CONDITION",
            grounder,
        ),
    )


def _first_candidate_value(
    grounded: GroundedSpan | None,
    key: Literal["system", "code"],
) -> str | None:
    if grounded is None or not grounded.candidates:
        return None
    value = getattr(grounded.candidates[0], key)
    return str(value)


def _label_family(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", str(label)).strip("_")
    normalized = re.sub(r"^(?:B|I|E|S)_", "", normalized, flags=re.IGNORECASE)
    return normalized.upper()


def _normalize_alias(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _normalize_value(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _is_non_value(value: str) -> bool:
    return _NON_VALUE_RE.fullmatch(value.strip()) is not None


def _validate_offset(start: int, end: int, label: str) -> None:
    if not isinstance(start, int) or isinstance(start, bool):
        raise TypeError(f"{label} start must be an integer")
    if not isinstance(end, int) or isinstance(end, bool):
        raise TypeError(f"{label} end must be an integer")
    if start < 0 or end <= start:
        raise ValueError(f"{label} must satisfy 0 <= start < end")


def _integer_field(mapping: Mapping[str, Any], key: str) -> int | None:
    value = mapping.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _float_value(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def export_discharge_summary_fhir(
    summary: DischargeSummary,
    *,
    subject_reference: str = "Patient/openmed-subject",
    document_id: str = "openmed-discharge-summary",
    bundle: bool = True,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Export diagnoses, procedures, regimens, and follow-up as FHIR R4."""

    if not isinstance(summary, DischargeSummary):
        raise TypeError("summary must be a DischargeSummary")
    resources: list[dict[str, Any]] = []
    for mention in summary.discharge_diagnoses:
        resources.append(
            _export_mention(
                mention,
                resource="Condition",
                subject_reference=subject_reference,
                document_id=document_id,
            )
        )
    for mention in summary.procedures:
        resources.append(
            _export_mention(
                mention,
                resource="Procedure",
                subject_reference=subject_reference,
                document_id=document_id,
            )
        )
    for regimen in summary.discharge_medications:
        medication_span = regimen.grounded or GroundedSpan(
            text=regimen.medication,
            start=regimen.medication_span[0],
            end=regimen.medication_span[1],
            canonical_label="MEDICATION",
        )
        resource = to_fhir(
            medication_span,
            resource="MedicationStatement",
            subject_reference=subject_reference,
            document_id=document_id,
        )
        if resource is None:
            continue
        resource["note"] = [{"text": summary.advisory}]
        dosage = _fhir_dosage(regimen.sig)
        if dosage:
            resource["dosage"] = [dosage]
        resources.append(resource)
    for index, follow_up in enumerate(summary.follow_up):
        resources.append(
            _follow_up_care_plan(
                follow_up,
                index=index,
                subject_reference=subject_reference,
                document_id=document_id,
                advisory=summary.advisory,
            )
        )
    if bundle:
        return to_bundle(resources, doc_id=document_id)
    return resources


def _export_mention(
    mention: DischargeMention,
    *,
    resource: Literal["Condition", "Procedure"],
    subject_reference: str,
    document_id: str,
) -> dict[str, Any]:
    grounded = mention.grounded or GroundedSpan(
        text=mention.text,
        start=mention.start,
        end=mention.end,
        canonical_label="CONDITION" if resource == "Condition" else "PROCEDURE",
    )
    exported = to_fhir(
        grounded,
        resource=resource,
        subject_reference=subject_reference,
        document_id=document_id,
    )
    if exported is None:
        raise ValueError("patient discharge mentions must export to a resource")
    return exported


def _fhir_dosage(sig: Sig) -> dict[str, Any]:
    dosage: dict[str, Any] = {}
    dose = sig.get("dose")
    unit = sig.get("unit") or sig.get("form")
    if dose is not None:
        dose_and_rate: dict[str, Any] = {"doseQuantity": {"value": dose}}
        if unit:
            dose_and_rate["doseQuantity"].update(
                {
                    "unit": str(unit),
                    "system": "http://unitsofmeasure.org",
                    "code": str(unit),
                }
            )
        dosage["doseAndRate"] = [dose_and_rate]
    route = sig.get("route")
    if route:
        dosage["route"] = {"text": str(route)}
    frequency = sig.get("frequency_per_day")
    period = sig.get("frequency_period")
    period_unit = sig.get("frequency_period_unit")
    as_needed = bool(sig.get("as_needed"))
    if frequency is not None or period is not None or as_needed:
        repeat: dict[str, Any] = {}
        if frequency is not None and float(frequency).is_integer():
            repeat["frequency"] = int(float(frequency))
        elif frequency is not None:
            repeat["frequency"] = float(frequency)
        if period is not None:
            repeat["period"] = period
        if period_unit is not None:
            repeat["periodUnit"] = period_unit
        if as_needed:
            repeat["asNeededBoolean"] = True
        dosage["timing"] = {"repeat": repeat}
    duration = sig.get("duration_days")
    if duration is not None:
        dosage.setdefault("timing", {}).setdefault("repeat", {})["boundsDuration"] = {
            "value": duration,
            "unit": "d",
            "system": "http://unitsofmeasure.org",
            "code": "d",
        }
    return dosage


def _follow_up_care_plan(
    mention: DischargeMention,
    *,
    index: int,
    subject_reference: str,
    document_id: str,
    advisory: str,
) -> dict[str, Any]:
    resource_id = f"openmed-follow-up-{index}-{mention.start}-{mention.end}"
    return {
        "resourceType": "CarePlan",
        "id": resource_id,
        "status": "active",
        "intent": "plan",
        "subject": {"reference": subject_reference},
        "description": mention.text,
        "activity": [
            {
                "detail": {
                    "status": "not-started",
                    "description": mention.text,
                }
            }
        ],
        "extension": [
            {
                "url": MEDICAL_DEVICE_ASSIST_EXTENSION_URL,
                "extension": [
                    {"url": "assist_only", "valueBoolean": True},
                    {"url": "autonomous_decision", "valueBoolean": False},
                    {"url": "evidence_start", "valueUnsignedInt": mention.start},
                    {"url": "evidence_end", "valueUnsignedInt": mention.end},
                    {"url": "disclaimer", "valueString": advisory},
                    {"url": "document_id", "valueString": document_id},
                ],
            }
        ],
    }


def discharge_summary_field_metrics(
    predicted: DischargeSummary | Mapping[str, Any],
    gold: DischargeSummary | Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Return precision, recall, and F1 for each discharge-summary field.

    Gold values may be strings, mappings with ``text``/``medication``, or
    record items.  Medication values are compared by medication name so a
    structured sig does not penalize the field-content metric.
    """

    metrics: dict[str, dict[str, float]] = {}
    for field_name in DISCHARGE_FIELD_NAMES:
        predicted_values = Counter(_field_values(predicted, field_name))
        gold_values = Counter(_field_values(gold, field_name))
        true_positive = sum((predicted_values & gold_values).values())
        predicted_count = sum(predicted_values.values())
        gold_count = sum(gold_values.values())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 1.0
            if not predicted_count and not gold_count
            else 0.0
        )
        metrics[field_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positive": float(true_positive),
            "predicted": float(predicted_count),
            "gold": float(gold_count),
        }
    return metrics


def discharge_summary_field_f1(
    predicted: DischargeSummary | Mapping[str, Any],
    gold: DischargeSummary | Mapping[str, Any],
) -> dict[str, float]:
    """Return field-level F1 values for the four canonical fields."""

    return {
        field_name: metrics["f1"]
        for field_name, metrics in discharge_summary_field_metrics(
            predicted, gold
        ).items()
    }


def _field_values(
    record: DischargeSummary | Mapping[str, Any],
    field_name: DischargeFieldName,
) -> list[str]:
    if isinstance(record, Mapping):
        values = record.get(field_name, ())
    else:
        values = getattr(record, field_name)
    if values is None:
        return []
    if isinstance(values, (str, bytes, Mapping)):
        values = (values,)
    result: list[str] = []
    for value in values:
        if isinstance(value, str):
            surface = value
        elif isinstance(value, MedicationRegimen):
            surface = value.medication
        elif isinstance(value, Mapping):
            surface = value.get("medication", value.get("text", ""))
        else:
            surface = getattr(value, "medication", getattr(value, "text", ""))
        if isinstance(surface, str) and surface.strip():
            result.append(_normalize_value(surface))
    return result


__all__ = [
    "DISCHARGE_FIELD_NAMES",
    "DISCHARGE_SUMMARY_ADVISORY",
    "DISCHARGE_SUMMARY_REVIEW_ADVISORY",
    "MEDICATION_DEVICE_REVIEW_AID_ADVISORY",
    "DischargeDiagnosis",
    "DischargeFieldName",
    "DischargeMention",
    "DischargeProcedure",
    "DischargeSectionSpan",
    "DischargeSummary",
    "DischargeSummaryRecord",
    "FollowUpInstruction",
    "MedicationRegimen",
    "discharge_summary_field_f1",
    "discharge_summary_field_metrics",
    "export_discharge_summary_fhir",
    "extract_discharge_summary",
]
