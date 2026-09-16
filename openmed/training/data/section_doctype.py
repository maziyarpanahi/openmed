"""Weak-labeled section and document-type training example assembly.

The builder is deliberately offline. It accepts caller-supplied public note
records and deterministic synthetic notes, delegates section boundaries and
document-type inference to the rules-first clinical runtime, and records how
each weak label was obtained for later adjudication.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeAlias

from openmed.clinical.sections import (
    UNKNOWN_DOCUMENT_TYPE,
    SectionSpan,
    classify_document,
    detect_sections,
    validate_section_spans,
)
from openmed.clinical.sections import (
    document_type_loinc_coverage as _document_type_loinc_coverage,
)
from openmed.training.synthetic import LocalePhiGenerator

RULE_LABEL_SOURCE = "rule"
SYNTHETIC_LABEL_SOURCE = "synthetic"
LABEL_SOURCES = (RULE_LABEL_SOURCE, SYNTHETIC_LABEL_SOURCE)
LabelSource = Literal["rule", "synthetic"]

SECTION_LABELS = (
    "history_of_present_illness",
    "past_medical_history",
    "medications",
    "allergies",
    "social_history",
    "assessment_and_plan",
    "findings",
    "impression",
)
TARGET_SECTION_LABELS = SECTION_LABELS

DOCUMENT_TYPES = (
    "discharge_summary",
    "progress_note",
    "radiology_report",
    "pathology_report",
    "operative_note",
    "history_and_physical",
    "consult_note",
)
SUPPORTED_DOCUMENT_TYPES = DOCUMENT_TYPES
DEFAULT_DOCUMENT_TYPE_MAX_TOKENS = 256

_TOKEN_RE = re.compile(
    r"\w+(?:['\N{RIGHT SINGLE QUOTATION MARK}-]\w+)*",
    re.UNICODE,
)
_DOCUMENT_TYPE_TITLES = {
    "discharge_summary": "DISCHARGE SUMMARY",
    "progress_note": "PROGRESS NOTE",
    "radiology_report": "RADIOLOGY REPORT",
    "pathology_report": "PATHOLOGY REPORT",
    "operative_note": "OPERATIVE NOTE",
    "history_and_physical": "HISTORY AND PHYSICAL",
    "consult_note": "CONSULTATION NOTE",
}
_SECTION_HEADERS = (
    ("HPI", "history_of_present_illness"),
    ("PMH", "past_medical_history"),
    ("MEDICATIONS", "medications"),
    ("ALLERGIES", "allergies"),
    ("SOCIAL HISTORY", "social_history"),
    ("ASSESSMENT/PLAN", "assessment_and_plan"),
    ("FINDINGS", "findings"),
    ("IMPRESSION", "impression"),
)
_SOCIAL_HISTORY_FRAGMENTS = (
    "Never smoked; no alcohol or recreational drug use.",
    "Lives with family; walks daily and reports no tobacco use.",
    "Former smoker; stopped years ago and drinks alcohol rarely.",
    "Works from home; uses no tobacco, alcohol, or recreational drugs.",
)


@dataclass(frozen=True)
class SectionDoctypeNote:
    """One source note consumed by the training-label builder.

    ``doc_type`` may be omitted for public notes. In that case the rules-first
    runtime classifier supplies the weak document-type label and abstains when
    no signature wins.
    """

    text: str
    record_id: str
    label_source: LabelSource = RULE_LABEL_SOURCE
    doc_type: str | None = None
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SectionTrainingExample:
    """A detector-aligned section text window and its weak label."""

    text: str
    section_label: str
    start: int
    end: int
    record_id: str
    label_source: LabelSource
    language: str = "en"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return the canonical classifier label."""

        return self.section_label

    @property
    def source(self) -> LabelSource:
        """Return the weak-label provenance alias."""

        return self.label_source

    def to_section_span(self) -> SectionSpan:
        """Recreate the runtime span used to build this example.

        Returns:
            A ``SectionSpan`` with the original detector metadata.
        """

        span_metadata = self.metadata.get("section_span", {})
        if not isinstance(span_metadata, Mapping):
            span_metadata = {}
        return SectionSpan(
            label=self.section_label,
            start=self.start,
            end=self.end,
            **dict(span_metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready training row.

        Returns:
            The section example as a plain dictionary.
        """

        return {
            "end": self.end,
            "label_source": self.label_source,
            "language": self.language,
            "metadata": dict(self.metadata),
            "record_id": self.record_id,
            "section_label": self.section_label,
            "start": self.start,
            "text": self.text,
        }


@dataclass(frozen=True)
class DocumentTypeTrainingExample:
    """A first-token-window note example and its document-type label."""

    text: str
    doc_type: str
    record_id: str
    label_source: LabelSource
    max_tokens: int
    token_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return the canonical document-type classifier label."""

        return self.doc_type

    @property
    def source(self) -> LabelSource:
        """Return the weak-label provenance alias."""

        return self.label_source

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-ready training row.

        Returns:
            The document-type example as a plain dictionary.
        """

        return {
            "doc_type": self.doc_type,
            "label_source": self.label_source,
            "max_tokens": self.max_tokens,
            "metadata": dict(self.metadata),
            "record_id": self.record_id,
            "text": self.text,
            "token_count": self.token_count,
        }


@dataclass(frozen=True)
class SectionDoctypeTrainingSet:
    """Section and document-type examples emitted in one builder pass."""

    section_examples: tuple[SectionTrainingExample, ...]
    document_type_examples: tuple[DocumentTypeTrainingExample, ...]

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return JSON-ready rows grouped by classifier task.

        Returns:
            Section and document-type rows under separate keys.
        """

        return {
            "document_type_examples": [
                example.to_dict() for example in self.document_type_examples
            ],
            "section_examples": [
                example.to_dict() for example in self.section_examples
            ],
        }


class _PublicNoteRecord(Protocol):
    """Structural contract implemented by public dataset adapter records."""

    text: str
    record_id: str
    language: str
    metadata: Mapping[str, Any]


NoteInput: TypeAlias = SectionDoctypeNote | Mapping[str, Any] | _PublicNoteRecord


@dataclass(frozen=True)
class SectionDoctypeLabelBuilder:
    """Build weak-labeled examples from public and synthetic clinical notes."""

    max_document_tokens: int = DEFAULT_DOCUMENT_TYPE_MAX_TOKENS
    section_labels: tuple[str, ...] = SECTION_LABELS

    def __post_init__(self) -> None:
        _validate_max_tokens(self.max_document_tokens)
        if not self.section_labels:
            raise ValueError("section_labels must not be empty")
        unknown = set(self.section_labels).difference(SECTION_LABELS)
        if unknown:
            raise ValueError(
                "unsupported section label(s): " + ", ".join(sorted(unknown))
            )

    def build(self, notes: Iterable[NoteInput]) -> SectionDoctypeTrainingSet:
        """Build both classifier datasets from one materialized note stream.

        Args:
            notes: Public adapter records, mappings, or synthetic note objects.

        Returns:
            The section and document-type examples.
        """

        normalized_notes = _normalize_notes(notes)
        return SectionDoctypeTrainingSet(
            section_examples=self.build_section_examples(normalized_notes),
            document_type_examples=self.build_document_type_examples(normalized_notes),
        )

    def build_section_examples(
        self,
        notes: Iterable[NoteInput],
    ) -> tuple[SectionTrainingExample, ...]:
        """Run the section rules and emit exact detector-aligned windows.

        Args:
            notes: Public adapter records, mappings, or synthetic note objects.

        Returns:
            Target section examples in note and offset order.
        """

        examples: list[SectionTrainingExample] = []
        allowed_labels = frozenset(self.section_labels)
        for note in _normalize_notes(notes):
            spans = detect_sections(
                note.text,
                language=note.language,
                include_unsectioned=True,
            )
            validate_section_spans(note.text, spans)
            for span in spans:
                if span.label not in allowed_labels:
                    continue
                span_metadata = {
                    key: value
                    for key, value in span.items()
                    if key not in {"label", "start", "end"}
                }
                examples.append(
                    SectionTrainingExample(
                        text=note.text[span.start : span.end],
                        section_label=span.label,
                        start=span.start,
                        end=span.end,
                        record_id=note.record_id,
                        label_source=note.label_source,
                        language=note.language,
                        metadata={
                            **dict(note.metadata),
                            "section_span": span_metadata,
                        },
                    )
                )
        return tuple(examples)

    def build_document_type_examples(
        self,
        notes: Iterable[NoteInput],
    ) -> tuple[DocumentTypeTrainingExample, ...]:
        """Emit first-N-token windows for known or rule-inferred note types.

        Args:
            notes: Public adapter records, mappings, or synthetic note objects.

        Returns:
            Recognized document-type examples in note order. Notes for which
            the rules abstain are omitted.
        """

        examples: list[DocumentTypeTrainingExample] = []
        for note in _normalize_notes(notes):
            doc_type, confidence = _document_type_label(note)
            if doc_type is None:
                continue
            window = first_token_window(
                note.text,
                max_tokens=self.max_document_tokens,
            )
            if not window:
                continue
            metadata = dict(note.metadata)
            if confidence is not None:
                metadata["rule_confidence"] = confidence
            examples.append(
                DocumentTypeTrainingExample(
                    text=window,
                    doc_type=doc_type,
                    record_id=note.record_id,
                    label_source=note.label_source,
                    max_tokens=self.max_document_tokens,
                    token_count=len(_TOKEN_RE.findall(window)),
                    metadata=metadata,
                )
            )
        return tuple(examples)


def first_token_window(text: str, *, max_tokens: int) -> str:
    """Return the original-text prefix ending at the Nth lexical token.

    This uses the same Unicode-aware token definition as the rules-first
    document classifier while preserving the source casing and punctuation.

    Args:
        text: Source note text.
        max_tokens: Maximum number of lexical tokens to retain.

    Returns:
        The original-text prefix through the last retained token.

    Raises:
        TypeError: If ``text`` is not a string.
        ValueError: If ``max_tokens`` is not a positive integer.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    _validate_max_tokens(max_tokens)
    final_match: re.Match[str] | None = None
    for index, match in enumerate(_TOKEN_RE.finditer(text), start=1):
        final_match = match
        if index == max_tokens:
            break
    if final_match is None:
        return ""
    return text[: final_match.end()].strip()


def build_section_examples(
    notes: Iterable[NoteInput],
) -> tuple[SectionTrainingExample, ...]:
    """Build weak-labeled section windows with the default label set.

    Args:
        notes: Public adapter records, mappings, or synthetic note objects.

    Returns:
        Detector-aligned examples for the eight target sections.
    """

    return SectionDoctypeLabelBuilder().build_section_examples(notes)


def build_document_type_examples(
    notes: Iterable[NoteInput],
    *,
    max_tokens: int = DEFAULT_DOCUMENT_TYPE_MAX_TOKENS,
) -> tuple[DocumentTypeTrainingExample, ...]:
    """Build weak-labeled document-type first-token windows.

    Args:
        notes: Public adapter records, mappings, or synthetic note objects.
        max_tokens: Maximum tokens retained from the start of each note.

    Returns:
        Recognized document-type examples in note order.
    """

    return SectionDoctypeLabelBuilder(
        max_document_tokens=max_tokens
    ).build_document_type_examples(notes)


def document_type_loinc_mapping_coverage(
    notes: Iterable[NoteInput],
) -> dict[str, Any]:
    """Report classifier-to-LOINC coverage for public or synthetic notes.

    The report contains aggregate counts and unmapped labels only. It is safe
    for the public harness because note text is consumed locally and never
    included in the returned evidence.
    """

    normalized_notes = _normalize_notes(notes)
    classifications = (classify_document(note.text) for note in normalized_notes)
    return _document_type_loinc_coverage(classifications)


def build_section_doctype_examples(
    notes: Iterable[NoteInput],
    *,
    max_tokens: int = DEFAULT_DOCUMENT_TYPE_MAX_TOKENS,
) -> SectionDoctypeTrainingSet:
    """Build both section and document-type examples from ``notes``.

    Args:
        notes: Public adapter records, mappings, or synthetic note objects.
        max_tokens: Maximum tokens retained for document-type examples.

    Returns:
        The section and document-type training rows.
    """

    return SectionDoctypeLabelBuilder(max_document_tokens=max_tokens).build(notes)


def generate_synthetic_social_history(*, seed: int = 0) -> str:
    """Generate deterministic, explicitly synthetic social-history text.

    Args:
        seed: Local random seed.

    Returns:
        A synthetic social-history fragment.
    """

    return random.Random(seed).choice(_SOCIAL_HISTORY_FRAGMENTS)


def generate_synthetic_notes(
    *,
    seed: int = 0,
) -> tuple[SectionDoctypeNote, ...]:
    """Generate messy offline notes covering all target labels and note types.

    Locale-PHI content exercises noisy content inside a section without using
    real identifiers. Social-history content is generated separately so the
    resulting training rows retain both synthetic-source provenance markers.

    Args:
        seed: Seed shared by the local synthetic generators.

    Returns:
        Six notes covering all document types and target sections.
    """

    locale_phi = LocalePhiGenerator(seed=seed).generate("en")
    social_history = generate_synthetic_social_history(seed=seed)
    contents = {
        "history_of_present_illness": locale_phi.text,
        "past_medical_history": "Synthetic history of seasonal wheeze.",
        "medications": "Example inhaler as directed; no real prescription.",
        "allergies": "No known allergies reported in this synthetic note.",
        "social_history": social_history,
        "assessment_and_plan": "Stable; continue synthetic follow-up plan.",
        "findings": "No acute synthetic finding.",
        "impression": "Synthetic example without clinical decision support.",
    }
    notes = tuple(
        SectionDoctypeNote(
            text=_render_synthetic_note(
                title=_DOCUMENT_TYPE_TITLES[doc_type],
                contents=contents,
                style=index,
            ),
            record_id=f"synthetic-{doc_type}-{seed}",
            label_source=SYNTHETIC_LABEL_SOURCE,
            doc_type=doc_type,
            metadata={
                "contains_real_phi": False,
                "synthetic": True,
                "synthetic_sources": ("locale_phi", "social_history"),
            },
        )
        for index, doc_type in enumerate(DOCUMENT_TYPES)
    )
    _validate_synthetic_coverage(notes)
    return notes


def _render_synthetic_note(
    *,
    title: str,
    contents: Mapping[str, str],
    style: int,
) -> str:
    lines = [title]
    for header, label in _SECTION_HEADERS:
        content = contents[label]
        variant = style % 4
        if variant == 0:
            lines.append(f"{header}: {content}")
        elif variant == 1:
            lines.extend((header, content))
        elif variant == 2:
            lines.append(f"- {header}: {content}")
        else:
            lines.extend((header, "---", content))
    return "\n".join(lines)


def _validate_synthetic_coverage(notes: Sequence[SectionDoctypeNote]) -> None:
    observed_sections = {
        span.label
        for note in notes
        for span in detect_sections(note.text, language=note.language)
    }
    missing_sections = set(SECTION_LABELS).difference(observed_sections)
    if missing_sections:
        raise RuntimeError(
            "synthetic notes are missing section labels: "
            + ", ".join(sorted(missing_sections))
        )
    observed_document_types = {note.doc_type for note in notes}
    missing_document_types = set(DOCUMENT_TYPES).difference(observed_document_types)
    if missing_document_types:
        raise RuntimeError(
            "synthetic notes are missing document types: "
            + ", ".join(sorted(missing_document_types))
        )


def _normalize_notes(
    notes: Iterable[NoteInput],
) -> tuple[SectionDoctypeNote, ...]:
    if isinstance(notes, (str, bytes)):
        raise TypeError("notes must be an iterable of note records")
    return tuple(_normalize_note(note, index) for index, note in enumerate(notes))


def _normalize_note(note: NoteInput, index: int) -> SectionDoctypeNote:
    if isinstance(note, SectionDoctypeNote):
        normalized = note
    elif isinstance(note, Mapping):
        metadata = note.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError(f"note {index} metadata must be a mapping")
        raw_source = note.get("label_source")
        if raw_source is None and note.get("source") in LABEL_SOURCES:
            raw_source = note["source"]
        if raw_source is None:
            raw_source = (
                SYNTHETIC_LABEL_SOURCE
                if note.get("synthetic") is True or metadata.get("synthetic") is True
                else RULE_LABEL_SOURCE
            )
        raw_doc_type = note.get("doc_type", metadata.get("doc_type"))
        normalized = SectionDoctypeNote(
            text=note.get("text", ""),
            record_id=str(note.get("record_id") or note.get("id") or f"note-{index}"),
            label_source=raw_source,
            doc_type=raw_doc_type,
            language=str(note.get("language") or note.get("lang") or "en"),
            metadata=dict(metadata),
        )
    elif hasattr(note, "text") and hasattr(note, "record_id"):
        metadata = getattr(note, "metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypeError(f"note {index} metadata must be a mapping")
        normalized_metadata = dict(metadata)
        for field_name in ("dataset", "split"):
            value = getattr(note, field_name, None)
            if value is not None:
                normalized_metadata.setdefault(field_name, value)
        raw_source = getattr(note, "label_source", RULE_LABEL_SOURCE)
        if raw_source not in LABEL_SOURCES:
            raw_source = RULE_LABEL_SOURCE
        normalized = SectionDoctypeNote(
            text=getattr(note, "text"),
            record_id=str(getattr(note, "record_id")),
            label_source=raw_source,
            doc_type=normalized_metadata.get("doc_type"),
            language=str(getattr(note, "language", "en")),
            metadata=normalized_metadata,
        )
    else:
        raise TypeError(
            f"note {index} must be SectionDoctypeNote, a mapping, "
            "or a public dataset record"
        )

    if not isinstance(normalized.text, str):
        raise TypeError(f"note {index} text must be a string")
    if not isinstance(normalized.metadata, Mapping):
        raise TypeError(f"note {index} metadata must be a mapping")
    if not normalized.record_id:
        raise ValueError(f"note {index} record_id must not be empty")
    if normalized.label_source not in LABEL_SOURCES:
        raise ValueError(f"note {index} label_source must be one of {LABEL_SOURCES!r}")
    if normalized.doc_type is not None:
        if not isinstance(normalized.doc_type, str):
            raise TypeError(f"note {index} doc_type must be a string")
        if normalized.doc_type not in DOCUMENT_TYPES:
            raise ValueError(
                f"note {index} has unsupported doc_type {normalized.doc_type!r}"
            )
    return normalized


def _document_type_label(note: SectionDoctypeNote) -> tuple[str | None, float | None]:
    if note.doc_type is not None:
        return note.doc_type, None
    classification = classify_document(note.text)
    if classification["type"] == UNKNOWN_DOCUMENT_TYPE:
        return None, None
    return classification["type"], classification["confidence"]


def _validate_max_tokens(max_tokens: int) -> None:
    if (
        not isinstance(max_tokens, int)
        or isinstance(max_tokens, bool)
        or max_tokens < 1
    ):
        raise ValueError("max_tokens must be a positive integer")


__all__ = [
    "DEFAULT_DOCUMENT_TYPE_MAX_TOKENS",
    "DOCUMENT_TYPES",
    "LABEL_SOURCES",
    "RULE_LABEL_SOURCE",
    "SECTION_LABELS",
    "SUPPORTED_DOCUMENT_TYPES",
    "SYNTHETIC_LABEL_SOURCE",
    "TARGET_SECTION_LABELS",
    "DocumentTypeTrainingExample",
    "LabelSource",
    "NoteInput",
    "SectionDoctypeLabelBuilder",
    "SectionDoctypeNote",
    "SectionDoctypeTrainingSet",
    "SectionTrainingExample",
    "build_document_type_examples",
    "build_section_doctype_examples",
    "build_section_examples",
    "document_type_loinc_mapping_coverage",
    "first_token_window",
    "generate_synthetic_notes",
    "generate_synthetic_social_history",
]
