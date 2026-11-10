"""Deterministic clinical form extraction with privacy-safe review output.

The extractor consumes :class:`openmed.multimodal.document_graph.DocumentGraph`
objects (or the same synthetic OCR block inputs accepted by the graph builder).
It understands explicit OCR key/value relationships, labelled form lines, and
table header/value rows.  Every field keeps graph offsets and page geometry;
serialization uses only a configured or deterministic local privacy transform.

This is extraction and review support, not an autonomous clinical decision
system.  Ambiguous or low-confidence matches remain visibly reviewable.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from openmed.multimodal.document_graph import (
    BBox,
    DocumentGraph,
    build_document_graph,
)

_KEY_VALUE_RE = re.compile(r"^\s*(.{1,100}?)\s*[:=：﹕꞉]\s*(.*?)\s*$")
_TAB_VALUE_RE = re.compile(r"^\s*([^\t|]{1,100}?)\s*[\t|]\s*(.*?)\s*$")
_SPACED_VALUE_RE = re.compile(r"^\s*(.{1,80}?)\s{2,}(.+?)\s*$")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d ().-]{6,}\d)(?!\w)")
_SSN_RE = re.compile(r"(?<!\w)\d{3}-\d{2}-\d{4}(?!\w)")
_DATE_RE = re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$")
_SYNTHETIC_ID_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:SYNTH|TEST|PATIENT|PERSON|MRN|ID)[-_][A-Za-z0-9][A-Za-z0-9_-]*(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_DIRECT_IDENTIFIER_TERMS = frozenset(
    {
        "address",
        "birth",
        "dob",
        "email",
        "identifier",
        "id",
        "mrn",
        "name",
        "patient",
        "phone",
        "record",
        "ssn",
        "telephone",
    }
)


def _value(source: Any, *names: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        for name in names:
            if name in source:
                return source[name]
        return default
    for name in names:
        if hasattr(source, name):
            return getattr(source, name)
    return default


def _normalise_key(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return text or "field"


def _coerce_confidence(value: Any, default: float) -> float:
    if value is None:
        return default
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, result))


def _coerce_bbox(value: Any) -> BBox | None:
    if value is None:
        return None
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 4:
            return None
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    return None


def _call_flexible(callable_: Callable[..., Any], value: str, field: Any) -> Any:
    try:
        return callable_(value, field=field)
    except TypeError:
        try:
            return callable_(value, field)
        except TypeError:
            return callable_(value)


def _iter_detected_spans(result: Any) -> tuple[tuple[int, int, str], ...]:
    if result is None or result is False:
        return ()
    if result is True:
        return ((0, -1, "PII"),)
    entities = _value(result, "entities", "pii_entities", "spans", default=None)
    if entities is None and isinstance(result, Mapping):
        entities = result.get("matches")
    if entities is None:
        entities = (
            result
            if isinstance(result, Iterable) and not isinstance(result, str)
            else ()
        )
    spans: list[tuple[int, int, str]] = []
    for entity in entities:
        start = _value(entity, "start", "begin", default=None)
        end = _value(entity, "end", "stop", default=None)
        if start is None or end is None:
            continue
        label = str(_value(entity, "label", "entity_type", "type", default="PII"))
        try:
            spans.append((int(start), int(end), label))
        except (TypeError, ValueError):
            continue
    return tuple(spans)


def _builtin_spans(value: str) -> tuple[tuple[int, int, str], ...]:
    matches: list[tuple[int, int, str]] = []
    for pattern, label in (
        (_EMAIL_RE, "EMAIL"),
        (_SSN_RE, "SSN"),
        (_SYNTHETIC_ID_RE, "IDENTIFIER"),
    ):
        matches.extend(
            (match.start(), match.end(), label) for match in pattern.finditer(value)
        )
    matches.extend(
        (match.start(), match.end(), "PHONE")
        for match in _PHONE_RE.finditer(value)
        if not _DATE_RE.fullmatch(match.group(0).strip())
    )
    return tuple(sorted(matches, key=lambda item: (item[0], item[1], item[2])))


def _is_direct_identifier_key(key: str) -> bool:
    tokens = set(_normalise_key(key).split("_"))
    return bool(tokens & _DIRECT_IDENTIFIER_TERMS)


def _transform_value(
    value: str,
    *,
    label: str,
    transformer: Callable[..., Any] | None,
) -> str:
    if transformer is None:
        return f"[{label.upper() or 'REDACTED'}]"
    result = _call_flexible(transformer, value, label)
    transformed = _value(
        result, "deidentified_text", "redacted_text", "value", default=None
    )
    if transformed is None:
        transformed = result
    if not isinstance(transformed, str):
        raise TypeError("privacy transformer must return text")
    return transformed


def _sanitize_value(
    value: str,
    *,
    key: str,
    field: Any,
    pii_detector: Callable[..., Any] | None,
    transformer: Callable[..., Any] | None,
) -> str:
    if not value:
        return value
    detected = list(_builtin_spans(value))
    if pii_detector is not None:
        detected.extend(
            _iter_detected_spans(_call_flexible(pii_detector, value, field))
        )
    if _is_direct_identifier_key(key):
        detected.append((0, len(value), key))
    if not detected:
        return value
    valid = sorted(
        {
            (max(0, start), min(len(value), end), label)
            for start, end, label in detected
            if end == -1 or start < end
        },
        key=lambda item: (item[0], -item[1], item[2]),
    )
    selected: list[tuple[int, int, str]] = []
    cursor = -1
    for start, end, label in valid:
        if end == -1:
            end = len(value)
        if start < cursor:
            continue
        selected.append((start, end, label))
        cursor = end
    if len(selected) == 1 and selected[0][0] == 0 and selected[0][1] == len(value):
        return _transform_value(value, label=selected[0][2], transformer=transformer)
    pieces: list[str] = []
    cursor = 0
    for start, end, label in selected:
        pieces.append(value[cursor:start])
        pieces.append(
            _transform_value(value[start:end], label=label, transformer=transformer)
        )
        cursor = end
    pieces.append(value[cursor:])
    return "".join(pieces)


@dataclass(frozen=True)
class FieldProvenance:
    """PHI-safe provenance for one extracted form value."""

    start: int
    end: int
    page: int | None
    bbox: BBox | None
    node_id: str | None = None
    text_sha256: str | None = None

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the half-open graph offsets for the value."""
        return (self.start, self.end)

    def to_dict(self) -> dict[str, Any]:
        """Return provenance without the original value."""
        return {
            "start": self.start,
            "end": self.end,
            "page": self.page,
            "bbox": self.bbox,
            "node_id": self.node_id,
            "text_sha256": self.text_sha256,
        }


@dataclass(frozen=True)
class FormField:
    """One deterministic form match and its review/privacy state."""

    link_id: str
    label: str
    value: str
    redacted_value: str
    confidence: float
    page: int | None
    bbox: BBox | None
    start: int
    end: int
    key_start: int | None = None
    key_end: int | None = None
    review_required: bool = False
    warnings: tuple[str, ...] = ()
    data_type: str = "string"
    repeats: bool = False
    provenance: FieldProvenance | None = None
    table_id: str | None = None
    row: int | None = None
    column: int | None = None

    @property
    def needs_review(self) -> bool:
        """Return whether a human must confirm this match."""
        return self.review_required

    @property
    def source_offsets(self) -> tuple[int, int]:
        """Return the value offsets in graph text."""
        return (self.start, self.end)

    def to_dict(self, *, include_original: bool = False) -> dict[str, Any]:
        """Return a review-safe field representation by default."""
        result: dict[str, Any] = {
            "link_id": self.link_id,
            "label": self.label,
            "value": self.redacted_value,
            "confidence": self.confidence,
            "page": self.page,
            "bbox": self.bbox,
            "start": self.start,
            "end": self.end,
            "source_offsets": {"start": self.start, "end": self.end},
            "key_offsets": (
                {"start": self.key_start, "end": self.key_end}
                if self.key_start is not None and self.key_end is not None
                else None
            ),
            "review_required": self.review_required,
            "warnings": list(self.warnings),
            "data_type": self.data_type,
            "repeats": self.repeats,
            "table_id": self.table_id,
            "row": self.row,
            "column": self.column,
        }
        if self.provenance is not None:
            result["provenance"] = self.provenance.to_dict()
        if include_original:
            result["original_value"] = self.value
        return result


@dataclass(frozen=True)
class FormExtractionResult:
    """Extracted fields, warnings, and the source graph used to derive them."""

    document: DocumentGraph
    fields: tuple[FormField, ...]
    warnings: tuple[str, ...] = ()
    privacy_applied: bool = True

    @property
    def review_required(self) -> bool:
        """Return whether any field or extraction warning needs review."""
        return bool(self.warnings) or any(
            field.review_required for field in self.fields
        )

    @property
    def values(self) -> Mapping[str, str]:
        """Return redacted values keyed by link ID."""
        return {field.link_id: field.redacted_value for field in self.fields}

    def to_dict(self, *, include_original: bool = False) -> dict[str, Any]:
        """Return a review artifact payload.

        ``include_original`` exists for local unit-test/debug callers only and
        is never used by the JSON or HTML review renderers.
        """
        return {
            "artifact": "openmed-clinical-form-review",
            "version": 1,
            "privacy_applied": self.privacy_applied,
            "review_required": self.review_required,
            "fields": [
                field.to_dict(include_original=include_original)
                for field in self.fields
            ],
            "warnings": list(self.warnings),
            "document": {
                "text_sha256": self.document.metadata.get("text_sha256"),
                "page_count": len(self.document.pages),
                "node_count": len(self.document.nodes),
            },
        }

    def to_review_json(self) -> str:
        """Serialize a deterministic PHI-free JSON review artifact."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_review_html(self) -> str:
        """Render a small PHI-free HTML review artifact."""
        rows = []
        for field in self.fields:
            warnings = "; ".join(field.warnings) or "None"
            rows.append(
                "<tr>"
                f"<td>{html.escape(field.label)}</td>"
                f"<td>{html.escape(field.redacted_value)}</td>"
                f"<td>{field.confidence:.3f}</td>"
                f"<td>{html.escape(str(field.page))}</td>"
                f"<td>{html.escape(str(field.bbox))}</td>"
                f"<td>{html.escape(warnings)}</td>"
                "</tr>"
            )
        warning_markup = "".join(
            f"<li>{html.escape(warning)}</li>" for warning in self.warnings
        )
        return (
            '<!doctype html><html><head><meta charset="utf-8">'
            "<title>Clinical form review</title></head><body>"
            "<h1>Clinical form review</h1>"
            f"<p>Review required: {str(self.review_required).lower()}</p>"
            f"<ul>{warning_markup}</ul>"
            "<table><thead><tr><th>Field</th><th>Value</th><th>Confidence</th>"
            "<th>Page</th><th>Region</th><th>Warnings</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></body></html>"
        )

    render_json = to_review_json
    render_html = to_review_html


class FormExtractor:
    """Reusable configured wrapper around :func:`extract_form_fields`."""

    def __init__(
        self,
        *,
        schema: Any = None,
        pii_detector: Callable[..., Any] | None = None,
        transformer: Callable[..., Any] | None = None,
        confidence_threshold: float = 0.75,
    ) -> None:
        self.schema = schema
        self.pii_detector = pii_detector
        self.transformer = transformer
        self.confidence_threshold = confidence_threshold

    def extract(self, document: Any, **overrides: Any) -> FormExtractionResult:
        """Extract fields using configured defaults and per-call overrides."""
        options = {
            "schema": self.schema,
            "pii_detector": self.pii_detector,
            "transformer": self.transformer,
            "confidence_threshold": self.confidence_threshold,
            **overrides,
        }
        return extract_form_fields(document, **options)

    __call__ = extract


def _schema_specs(schema: Any) -> list[dict[str, Any]]:
    if schema is None:
        return []
    if isinstance(schema, Mapping):
        items = _value(schema, "item", "items", "fields", default=None)
        if items is not None:
            if isinstance(items, Mapping):
                return [dict(items)]
            return [dict(item) for item in items]
        return [
            {
                "linkId": key,
                "text": key,
                **(dict(value) if isinstance(value, Mapping) else {}),
            }
            for key, value in schema.items()
        ]
    if isinstance(schema, Sequence) and not isinstance(schema, (str, bytes)):
        return [
            dict(item) if isinstance(item, Mapping) else {"text": str(item)}
            for item in schema
        ]
    return []


def _spec_match(
    label: str, specs: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], bool]:
    normalized = _normalise_key(label)
    candidates: list[dict[str, Any]] = []
    for raw in specs:
        spec = dict(raw)
        aliases = _value(spec, "aliases", "alias", default=[]) or []
        if isinstance(aliases, str):
            aliases = [aliases]
        names = [
            _value(spec, "linkId", "link_id", "id", "text", "label", default=""),
            *aliases,
        ]
        if any(_normalise_key(name) == normalized for name in names):
            candidates.append(spec)
    if candidates:
        return candidates[0], len(candidates) > 1
    return {"linkId": normalized, "text": label}, False


def _line_match(text: str) -> tuple[str, str, int, int, int, int, float] | None:
    for pattern, confidence in (
        (_KEY_VALUE_RE, 0.98),
        (_TAB_VALUE_RE, 0.93),
        (_SPACED_VALUE_RE, 0.74),
    ):
        match = pattern.match(text)
        if match is None:
            continue
        key = match.group(1).strip()
        value = match.group(2).strip()
        if not key or not value:
            return None
        key_start = match.start(1) + (
            len(match.group(1)) - len(match.group(1).lstrip())
        )
        key_end = key_start + len(key)
        value_start = match.start(2) + (
            len(match.group(2)) - len(match.group(2).lstrip())
        )
        value_end = value_start + len(value)
        return key, value, key_start, key_end, value_start, value_end, confidence
    return None


def _provenance(
    document: DocumentGraph,
    *,
    start: int,
    end: int,
    page: int | None,
    bbox: BBox | None,
    node_id: str | None,
) -> FieldProvenance:
    if page is None or bbox is None or node_id is None:
        region = document.project_span(start, end)
        if region:
            first = region[0]
            page = first.page if page is None else page
            bbox = first.bbox if bbox is None else bbox
            node_id = first.node_id if node_id is None else node_id
    return FieldProvenance(
        start=start,
        end=end,
        page=page,
        bbox=bbox,
        node_id=node_id,
        text_sha256=hashlib.sha256(
            document.text[start:end].encode("utf-8")
        ).hexdigest(),
    )


def _make_field(
    document: DocumentGraph,
    *,
    label: str,
    value: str,
    start: int,
    end: int,
    key_start: int | None,
    key_end: int | None,
    spec: Mapping[str, Any],
    confidence: float,
    warnings: Iterable[str] = (),
    page: int | None = None,
    bbox: BBox | None = None,
    node_id: str | None = None,
    table_id: str | None = None,
    row: int | None = None,
    column: int | None = None,
    pii_detector: Callable[..., Any] | None = None,
    transformer: Callable[..., Any] | None = None,
    confidence_threshold: float = 0.75,
    repeats: bool = False,
) -> FormField:
    warning_tuple = tuple(dict.fromkeys(str(item) for item in warnings))
    review_required = confidence < confidence_threshold or bool(warning_tuple)
    link_id = str(
        _value(spec, "linkId", "link_id", "id", default=_normalise_key(label))
    )
    data_type = str(_value(spec, "type", "data_type", "value_type", default="string"))
    redacted = _sanitize_value(
        value,
        key=link_id or label,
        field=spec,
        pii_detector=pii_detector,
        transformer=transformer,
    )
    return FormField(
        link_id=link_id,
        label=str(_value(spec, "text", "label", default=label)),
        value=value,
        redacted_value=redacted,
        confidence=confidence,
        page=page,
        bbox=bbox,
        start=start,
        end=end,
        key_start=key_start,
        key_end=key_end,
        review_required=review_required,
        warnings=warning_tuple,
        data_type=data_type,
        repeats=bool(_value(spec, "repeats", default=repeats)),
        provenance=_provenance(
            document,
            start=start,
            end=end,
            page=page,
            bbox=bbox,
            node_id=node_id,
        ),
        table_id=table_id,
        row=row,
        column=column,
    )


def _source_warnings(source: Any) -> tuple[str, ...]:
    metadata = _value(source, "metadata", default={}) or {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    raw_warnings = metadata.get("warnings", ())
    if isinstance(raw_warnings, str):
        raw_warnings = (raw_warnings,)
    warnings = [str(warning) for warning in raw_warnings or ()]
    if metadata.get("ambiguous"):
        warnings.append("ambiguous source match")
    if metadata.get("review_required"):
        warnings.append("source match requires review")
    return tuple(dict.fromkeys(warnings))


def _fields_from_explicit_graph(
    document: DocumentGraph,
    specs: Sequence[Mapping[str, Any]],
    *,
    pii_detector: Callable[..., Any] | None,
    transformer: Callable[..., Any] | None,
    confidence_threshold: float,
) -> list[FormField]:
    fields: list[FormField] = []
    for explicit in document.form_fields:
        spec, ambiguous = _spec_match(explicit.key, specs)
        warnings = (*_source_warnings(explicit),)
        if ambiguous:
            warnings = (*warnings, "ambiguous schema match")
        fields.append(
            _make_field(
                document,
                label=explicit.key,
                value=explicit.value,
                start=explicit.value_start,
                end=explicit.value_end,
                key_start=explicit.key_start,
                key_end=explicit.key_end,
                spec=spec,
                confidence=_coerce_confidence(explicit.confidence, 0.99),
                warnings=warnings,
                page=explicit.page,
                bbox=explicit.value_bbox or explicit.bbox,
                node_id=explicit.id,
                pii_detector=pii_detector,
                transformer=transformer,
                confidence_threshold=confidence_threshold,
            )
        )
    return fields


def _fields_from_lines(
    document: DocumentGraph,
    specs: Sequence[Mapping[str, Any]],
    existing_ranges: set[tuple[int, int]],
    *,
    pii_detector: Callable[..., Any] | None,
    transformer: Callable[..., Any] | None,
    confidence_threshold: float,
) -> list[FormField]:
    fields: list[FormField] = []
    for node in document.nodes:
        if node.kind in {"table_cell", "form_key", "form_value"}:
            continue
        match = _line_match(node.text)
        if match is None:
            continue
        label, value, key_start, key_end, value_start, value_end, confidence = match
        absolute_start = node.start + value_start
        absolute_end = node.start + value_end
        if (absolute_start, absolute_end) in existing_ranges:
            continue
        spec, ambiguous = _spec_match(label, specs)
        warnings = ("ambiguous schema match",) if ambiguous else _source_warnings(node)
        fields.append(
            _make_field(
                document,
                label=label,
                value=value,
                start=absolute_start,
                end=absolute_end,
                key_start=node.start + key_start,
                key_end=node.start + key_end,
                spec=spec,
                confidence=confidence,
                warnings=warnings,
                page=node.page,
                bbox=node.bbox,
                node_id=node.id,
                pii_detector=pii_detector,
                transformer=transformer,
                confidence_threshold=confidence_threshold,
            )
        )
    return fields


def _fields_from_tables(
    document: DocumentGraph,
    specs: Sequence[Mapping[str, Any]],
    existing_ranges: set[tuple[int, int]],
    *,
    pii_detector: Callable[..., Any] | None,
    transformer: Callable[..., Any] | None,
    confidence_threshold: float,
) -> list[FormField]:
    fields: list[FormField] = []
    for table in document.tables:
        if not table.cells:
            continue
        header_rows = {cell.row for cell in table.cells if cell.is_header}
        header_row = (
            min(header_rows) if header_rows else min(cell.row for cell in table.cells)
        )
        headers = {
            cell.column: cell
            for cell in table.cells
            if cell.row == header_row and cell.text.strip()
        }
        if not headers:
            continue
        for cell in table.cells:
            if cell.row == header_row or not cell.text.strip():
                continue
            header = headers.get(cell.column)
            if header is None or (cell.start, cell.end) in existing_ranges:
                continue
            spec, ambiguous = _spec_match(header.text, specs)
            warnings = (
                ("ambiguous schema match",) if ambiguous else _source_warnings(cell)
            )
            fields.append(
                _make_field(
                    document,
                    label=header.text,
                    value=cell.text,
                    start=cell.start,
                    end=cell.end,
                    key_start=header.start,
                    key_end=header.end,
                    spec=spec,
                    confidence=_coerce_confidence(cell.confidence, 0.96),
                    warnings=warnings,
                    page=cell.page,
                    bbox=cell.bbox,
                    node_id=cell.id,
                    table_id=table.id,
                    row=cell.row,
                    column=cell.column,
                    pii_detector=pii_detector,
                    transformer=transformer,
                    confidence_threshold=confidence_threshold,
                    repeats=True,
                )
            )
    return fields


def extract_form_fields(
    document: Any,
    *,
    schema: Any = None,
    pii_detector: Callable[..., Any] | None = None,
    transformer: Callable[..., Any] | None = None,
    transform: Callable[..., Any] | None = None,
    confidence_threshold: float = 0.75,
) -> FormExtractionResult:
    """Extract key/value and table fields with deterministic provenance.

    ``schema`` may be a FHIR Questionnaire-like mapping, a sequence of field
    mappings, or a mapping from link IDs to field options.  ``pii_detector``
    and ``transformer`` are local callables; when omitted, conservative local
    patterns and direct-identifier labels redact common synthetic identifiers.
    """
    if transform is not None:
        if transformer is not None:
            raise ValueError("provide only one of transformer and transform")
        transformer = transform
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0.0 and 1.0")
    graph = (
        document
        if isinstance(document, DocumentGraph)
        else build_document_graph(document)
    )
    specs = _schema_specs(schema)
    fields = _fields_from_explicit_graph(
        graph,
        specs,
        pii_detector=pii_detector,
        transformer=transformer,
        confidence_threshold=confidence_threshold,
    )
    ranges = {(field.start, field.end) for field in fields}
    fields.extend(
        _fields_from_lines(
            graph,
            specs,
            ranges,
            pii_detector=pii_detector,
            transformer=transformer,
            confidence_threshold=confidence_threshold,
        )
    )
    ranges.update((field.start, field.end) for field in fields)
    fields.extend(
        _fields_from_tables(
            graph,
            specs,
            ranges,
            pii_detector=pii_detector,
            transformer=transformer,
            confidence_threshold=confidence_threshold,
        )
    )
    fields.sort(key=lambda field: (field.start, field.link_id, field.label))
    counts = Counter(field.link_id for field in fields)
    normalized: list[FormField] = []
    for field in fields:
        duplicate = counts[field.link_id] > 1 and not field.repeats
        if duplicate:
            normalized.append(
                replace(
                    field,
                    review_required=True,
                    warnings=tuple(
                        dict.fromkeys(
                            (*field.warnings, "ambiguous repeated field label")
                        )
                    ),
                )
            )
        else:
            normalized.append(field)
    warnings = (
        ("one or more field matches require human review",)
        if any(field.review_required for field in normalized)
        else ()
    )
    return FormExtractionResult(
        document=graph,
        fields=tuple(normalized),
        warnings=warnings,
        privacy_applied=True,
    )


def redact_form_fields(
    result: FormExtractionResult,
    *,
    pii_detector: Callable[..., Any] | None = None,
    transformer: Callable[..., Any] | None = None,
) -> FormExtractionResult:
    """Apply a local privacy pass to an existing extraction result."""
    fields = tuple(
        replace(
            field,
            redacted_value=_sanitize_value(
                field.value,
                key=field.link_id,
                field=field,
                pii_detector=pii_detector,
                transformer=transformer,
            ),
        )
        for field in result.fields
    )
    return replace(result, fields=fields, privacy_applied=True)


def render_review_json(result: FormExtractionResult) -> str:
    """Return the PHI-free JSON review artifact for ``result``."""
    return result.to_review_json()


def render_review_html(result: FormExtractionResult) -> str:
    """Return the PHI-free HTML review artifact for ``result``."""
    return result.to_review_html()


def render_review_artifact(
    result: FormExtractionResult,
    *,
    format: str = "json",
) -> str:
    """Render JSON or HTML review output without original extracted values."""
    normalized = format.lower()
    if normalized == "json":
        return result.to_review_json()
    if normalized in {"html", "htm"}:
        return result.to_review_html()
    raise ValueError("review artifact format must be 'json' or 'html'")


extract_forms = extract_form_fields
extract_form = extract_form_fields
structure_form = extract_form_fields
to_review_json = render_review_json
to_review_html = render_review_html


__all__ = [
    "FieldProvenance",
    "FormField",
    "FormExtractionResult",
    "extract_form_fields",
    "extract_form",
    "extract_forms",
    "FormExtractor",
    "redact_form_fields",
    "render_review_artifact",
    "render_review_html",
    "render_review_json",
    "structure_form",
    "to_review_html",
    "to_review_json",
]
