"""PHI-safe provenance mapping for extracted DICOM-SR findings.

The DICOM-SR extractor exposes a deterministic ``node_path`` for every
content-tree item and a :class:`~openmed.multimodal.SourceSpan` for every
rendered line.  This module joins caller-owned, opaque finding identifiers to
those structural references without copying the item's concept, rendered
value, or any other report text into the resulting provenance.

All work is local and deterministic.  A path or offset that cannot be resolved
unambiguously is rejected rather than guessed, because an incorrect evidence
link is more dangerous than an absent one.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .base import ExtractedDocument, SourceSpan

__all__ = [
    "DICOM_SR_PROVENANCE_ADVISORY",
    "DICOM_SR_PROVENANCE_SCHEMA_VERSION",
    "AmbiguousItemPathError",
    "AmbiguousDicomSrItemPathError",
    "DicomSrProvenanceError",
    "DicomSrProvenanceRecord",
    "build_dicom_sr_provenance",
    "map_dicom_sr_provenance",
    "render_dicom_sr_provenance",
    "serialize_dicom_sr_provenance",
]


DICOM_SR_PROVENANCE_SCHEMA_VERSION = "openmed.dicom_sr_provenance.v1"
DICOM_SR_PROVENANCE_ADVISORY = (
    "DICOM-SR provenance contains structural references and character offsets "
    "only. It does not contain report values or a clinical interpretation."
)

_ITEM_PATH_RE = re.compile(r"^[1-9]\d*(?:\.[1-9]\d*)*$")
_PATH_FIELDS = ("item_path", "node_path", "source_item_path", "sr_item_path")
_IDENTIFIER_FIELDS = ("finding_id", "finding_identifier", "id", "identifier")
_TEMPLATE_FIELDS = ("template_id", "template_identifier")
_OFFSET_CONTAINER_FIELDS = (
    "source_offsets",
    "offsets",
    "span",
    "source_span",
)
_OFFSET_START_FIELDS = ("source_start", "start")
_OFFSET_END_FIELDS = ("source_end", "end")
_RECORD_FIELDS = (
    "finding_id",
    "item_path",
    "template_id",
    "source_start",
    "source_end",
)


class DicomSrProvenanceError(ValueError):
    """Base error for invalid or incomplete DICOM-SR provenance input."""


class AmbiguousDicomSrItemPathError(DicomSrProvenanceError):
    """Raised when more than one structural item could own a finding."""


# Short compatibility name for callers that do not need the DICOM qualifier.
AmbiguousItemPathError = AmbiguousDicomSrItemPathError


@dataclass(frozen=True)
class DicomSrProvenanceRecord(Mapping[str, Any]):
    """One value-free reference from a finding to a DICOM-SR content item.

    ``finding_id`` is an opaque identifier supplied by the caller.  The record
    intentionally has no concept name, rendered value, unit, or source text.
    ``source_start`` and ``source_end`` are half-open offsets into the
    extracted document text when a source span is available.
    """

    finding_id: str
    item_path: str
    template_id: str | None = None
    source_start: int | None = None
    source_end: int | None = None

    def __post_init__(self) -> None:
        _validate_identifier(self.finding_id, context="finding identifier")
        _validate_item_path(self.item_path, context="item path")
        _validate_template_id(self.template_id)
        _validate_offsets(self.source_start, self.source_end, context="source")

    @property
    def node_path(self) -> str:
        """Return the SR extractor's equivalent name for ``item_path``."""

        return self.item_path

    @property
    def source_offsets(self) -> tuple[int, int] | None:
        """Return the mapped half-open source range, when one is available."""

        if self.source_start is None or self.source_end is None:
            return None
        return self.source_start, self.source_end

    def to_dict(self) -> dict[str, Any]:
        """Return only the allowlisted, JSON-serializable record fields."""

        return {
            "finding_id": self.finding_id,
            "item_path": self.item_path,
            "template_id": self.template_id,
            "source_start": self.source_start,
            "source_end": self.source_end,
        }

    def __getitem__(self, key: str) -> Any:
        if key == "node_path":
            return self.item_path
        if key == "source_offsets":
            return self.source_offsets
        if key not in _RECORD_FIELDS:
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self):
        return iter(_RECORD_FIELDS)

    def __len__(self) -> int:
        return len(_RECORD_FIELDS)


@dataclass(frozen=True)
class _ItemInfo:
    template_id: str | None


@dataclass(frozen=True)
class _SpanInfo:
    start: int
    end: int


def build_dicom_sr_provenance(
    findings: Iterable[Mapping[str, Any] | Sequence[Any]] | Mapping[Any, Any],
    content_items: Sequence[Mapping[str, Any]] | ExtractedDocument | None = None,
    spans: Sequence[SourceSpan | Mapping[str, Any]] | None = None,
    *,
    source_spans: Sequence[SourceSpan | Mapping[str, Any]] | None = None,
    document: ExtractedDocument | None = None,
) -> tuple[DicomSrProvenanceRecord, ...]:
    """Build deterministic, value-free provenance for DICOM-SR findings.

    Args:
        findings: Finding records or a mapping from opaque finding identifiers
            to item paths. Records may use ``finding_id``/``id`` and
            ``item_path``/``node_path``. Source offsets may be supplied as
            ``source_start``/``source_end``, ``start``/``end``, or a
            ``source_offsets`` pair.
        content_items: The extractor's ``metadata["content_items"]`` list, or
            an :class:`~openmed.multimodal.ExtractedDocument`. The latter also
            supplies its spans. It may be omitted when every finding already
            carries an explicit path and no template lookup is needed.
        spans: Optional extracted-document source spans. Each span must carry
            its SR path in ``metadata["node_path"]`` (or ``item_path``).
        source_spans: Keyword alias for ``spans``.
        document: Explicit document convenience argument. It cannot be mixed
            with ``content_items`` or explicit spans.

    Returns:
        Records sorted by finding identifier and item path. Sorting is
        independent of input order, making JSON rendering reproducible.

    Raises:
        AmbiguousDicomSrItemPathError: If content items, source spans, path
            aliases, or offset matching identify more than one item.
        DicomSrProvenanceError: If an identifier, path, or offset is malformed
            or cannot be resolved.

    The function reads only structural fields from content items and spans;
    values such as ``value``, ``concept_name``, and arbitrary finding metadata
    are deliberately ignored.
    """
    resolved_items, resolved_spans = _resolve_sources(
        content_items,
        spans,
        source_spans=source_spans,
        document=document,
    )
    item_index = _index_content_items(resolved_items)
    span_index = _index_source_spans(resolved_spans)
    known_paths = set(item_index) | set(span_index)

    records: list[DicomSrProvenanceRecord] = []
    seen_finding_ids: set[str] = set()
    for index, raw_finding in enumerate(_normalise_findings(findings)):
        finding = _coerce_finding(raw_finding, index=index)
        finding_id = _finding_identifier(finding, index=index)
        if finding_id in seen_finding_ids:
            raise DicomSrProvenanceError(
                f"finding identifiers must be unique (duplicate at index {index})"
            )
        seen_finding_ids.add(finding_id)

        item_path = _finding_item_path(finding, index=index)
        explicit_start, explicit_end = _finding_offsets(finding, index=index)
        if item_path is None:
            item_path = _path_from_offsets(
                explicit_start,
                explicit_end,
                span_index,
                index=index,
            )
        elif known_paths and item_path not in known_paths:
            raise DicomSrProvenanceError(
                f"finding at index {index} references an unknown item path"
            )

        if item_path is None:  # Defensive; _path_from_offsets always resolves.
            raise DicomSrProvenanceError(
                f"finding at index {index} has no resolvable item path"
            )

        mapped_span = span_index.get(item_path)
        source_start = explicit_start
        source_end = explicit_end
        if source_start is None and mapped_span is not None:
            source_start = mapped_span.start
            source_end = mapped_span.end

        template_id = _nearest_template_id(item_path, item_index)
        if template_id is None:
            template_id = _finding_template_id(finding, index=index)

        records.append(
            DicomSrProvenanceRecord(
                finding_id=finding_id,
                item_path=item_path,
                template_id=template_id,
                source_start=source_start,
                source_end=source_end,
            )
        )

    return tuple(sorted(records, key=_record_sort_key))


def map_dicom_sr_provenance(
    findings: Iterable[Mapping[str, Any] | Sequence[Any]] | Mapping[Any, Any],
    content_items: Sequence[Mapping[str, Any]] | ExtractedDocument | None = None,
    spans: Sequence[SourceSpan | Mapping[str, Any]] | None = None,
    *,
    source_spans: Sequence[SourceSpan | Mapping[str, Any]] | None = None,
    document: ExtractedDocument | None = None,
) -> tuple[DicomSrProvenanceRecord, ...]:
    """Alias for :func:`build_dicom_sr_provenance` with mapping-oriented naming."""

    return build_dicom_sr_provenance(
        findings,
        content_items,
        spans,
        source_spans=source_spans,
        document=document,
    )


def render_dicom_sr_provenance(
    records: Iterable[DicomSrProvenanceRecord | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Render sorted, allowlisted provenance dictionaries.

    Mapping inputs are reconstructed through :class:`DicomSrProvenanceRecord`,
    so extra fields such as a raw ``value`` cannot leak into the rendered
    report.
    """

    normalized = [
        record
        if isinstance(record, DicomSrProvenanceRecord)
        else _record_from_mapping(record, index=index)
        for index, record in enumerate(records)
    ]
    return [record.to_dict() for record in sorted(normalized, key=_record_sort_key)]


def serialize_dicom_sr_provenance(
    records: Iterable[DicomSrProvenanceRecord | Mapping[str, Any]],
) -> str:
    """Serialize value-free records to canonical JSON without network access."""

    return json.dumps(
        render_dicom_sr_provenance(records),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def _resolve_sources(
    content_items: Sequence[Mapping[str, Any]] | ExtractedDocument | None,
    spans: Sequence[SourceSpan | Mapping[str, Any]] | None,
    *,
    source_spans: Sequence[SourceSpan | Mapping[str, Any]] | None,
    document: ExtractedDocument | None,
) -> tuple[Sequence[Mapping[str, Any]], Sequence[SourceSpan | Mapping[str, Any]]]:
    if spans is not None and source_spans is not None:
        raise TypeError("spans and source_spans are mutually exclusive")
    if content_items is not None and document is not None:
        raise TypeError("content_items and document are mutually exclusive")

    resolved_spans = source_spans if source_spans is not None else spans
    if document is not None:
        content_items = document.metadata.get("content_items")
        resolved_spans = document.spans
    elif isinstance(content_items, ExtractedDocument):
        document = content_items
        content_items = document.metadata.get("content_items")
        if resolved_spans is not None:
            raise TypeError("an ExtractedDocument already supplies its source spans")
        resolved_spans = document.spans

    if content_items is None:
        resolved_items: Sequence[Mapping[str, Any]] = ()
    elif isinstance(content_items, Mapping) or isinstance(content_items, (str, bytes)):
        raise TypeError("content_items must be a sequence of mappings")
    else:
        resolved_items = content_items

    return resolved_items, resolved_spans or ()


def _index_content_items(
    content_items: Sequence[Mapping[str, Any]],
) -> dict[str, _ItemInfo]:
    index: dict[str, _ItemInfo] = {}
    for item_index, item in enumerate(content_items):
        if not isinstance(item, Mapping):
            raise TypeError(f"content item at index {item_index} must be a mapping")
        path = _path_from_aliases(item, context=f"content item at index {item_index}")
        if path is None:
            raise DicomSrProvenanceError(
                f"content item at index {item_index} has no item path"
            )
        if path in index:
            raise AmbiguousDicomSrItemPathError(
                f"content items contain duplicate item path {path}"
            )
        index[path] = _ItemInfo(template_id=_template_from_mapping(item))
    return index


def _index_source_spans(
    spans: Sequence[SourceSpan | Mapping[str, Any]],
) -> dict[str, _SpanInfo]:
    index: dict[str, _SpanInfo] = {}
    for span_index, raw_span in enumerate(spans):
        start, end, metadata = _span_fields(raw_span, index=span_index)
        path = _path_from_aliases(
            metadata,
            context=f"source span at index {span_index}",
        )
        if path is None and isinstance(raw_span, Mapping):
            path = _path_from_aliases(
                raw_span,
                context=f"source span at index {span_index}",
            )
        if path is None:
            continue
        if path in index:
            raise AmbiguousDicomSrItemPathError(
                f"source spans contain duplicate item path {path}"
            )
        index[path] = _SpanInfo(start=start, end=end)
    return index


def _span_fields(
    raw_span: SourceSpan | Mapping[str, Any],
    *,
    index: int,
) -> tuple[int, int, Mapping[str, Any]]:
    if isinstance(raw_span, SourceSpan):
        metadata = raw_span.metadata
        start, end = raw_span.start, raw_span.end
    elif isinstance(raw_span, Mapping):
        metadata = raw_span.get("metadata", {})
        start = _first_present(raw_span, _OFFSET_START_FIELDS)
        end = _first_present(raw_span, _OFFSET_END_FIELDS)
    else:
        raise TypeError(f"source span at index {index} must be a SourceSpan or mapping")
    if not isinstance(metadata, Mapping):
        raise TypeError(f"source span metadata at index {index} must be a mapping")
    if start is None or end is None:
        raise DicomSrProvenanceError(
            f"source span at index {index} must contain start and end offsets"
        )
    _validate_offsets(start, end, context=f"source span at index {index}")
    # Validation above guarantees these are integers, but the explicit cast
    # keeps the internal descriptor narrow for type checkers.
    return int(start), int(end), metadata


def _normalise_findings(
    findings: Iterable[Mapping[str, Any] | Sequence[Any]] | Mapping[Any, Any],
) -> list[Any]:
    if isinstance(findings, Mapping):
        if _looks_like_single_finding(findings):
            return [findings]
        normalized: list[dict[str, Any]] = []
        for finding_id, specification in findings.items():
            if isinstance(specification, Mapping):
                row = dict(specification)
                row.setdefault("finding_id", finding_id)
            else:
                row = {"finding_id": finding_id, "item_path": specification}
            normalized.append(row)
        return normalized
    if isinstance(findings, (str, bytes)):
        raise TypeError("findings must be mappings or an iterable of finding records")
    return list(findings)


def _looks_like_single_finding(value: Mapping[Any, Any]) -> bool:
    return any(field in value for field in _IDENTIFIER_FIELDS + _PATH_FIELDS)


def _coerce_finding(raw: Any, *, index: int) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        if len(raw) != 2:
            raise TypeError(
                f"finding at index {index} must be a mapping or id/path pair"
            )
        return {"finding_id": raw[0], "item_path": raw[1]}
    raise TypeError(f"finding at index {index} must be a mapping or id/path pair")


def _finding_identifier(finding: Mapping[str, Any], *, index: int) -> str:
    value = _first_present(finding, _IDENTIFIER_FIELDS)
    if value is None:
        raise DicomSrProvenanceError(
            f"finding at index {index} is missing an opaque finding identifier"
        )
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(
            f"finding identifier at index {index} must be a string or integer"
        )
    identifier = str(value).strip()
    _validate_identifier(identifier, context=f"finding identifier at index {index}")
    return identifier


def _finding_item_path(
    finding: Mapping[str, Any],
    *,
    index: int,
) -> str | None:
    context = f"finding at index {index}"
    path = _path_from_aliases(finding, context=context)
    provenance = finding.get("provenance")
    if not isinstance(provenance, Mapping):
        return path
    nested_path = _path_from_aliases(provenance, context=context)
    if path is not None and nested_path is not None and path != nested_path:
        raise AmbiguousDicomSrItemPathError(
            f"{context} supplies conflicting item paths"
        )
    return path or nested_path


def _finding_template_id(
    finding: Mapping[str, Any],
    *,
    index: int,
) -> str | None:
    value = _first_present(finding, _TEMPLATE_FIELDS)
    if value is None:
        provenance = finding.get("provenance")
        if isinstance(provenance, Mapping):
            value = _first_present(provenance, _TEMPLATE_FIELDS)
    if value is None:
        return None
    template_id = _coerce_optional_string(value)
    _validate_template_id(template_id)
    return template_id


def _template_from_mapping(
    item: Mapping[str, Any],
) -> str | None:
    value = _first_present(item, _TEMPLATE_FIELDS)
    if value is None:
        return None
    template_id = _coerce_optional_string(value)
    _validate_template_id(template_id)
    return template_id


def _nearest_template_id(
    item_path: str,
    item_index: Mapping[str, _ItemInfo],
) -> str | None:
    parts = item_path.split(".")
    for end in range(len(parts), 0, -1):
        ancestor = ".".join(parts[:end])
        info = item_index.get(ancestor)
        if info is not None and info.template_id is not None:
            return info.template_id
    return None


def _finding_offsets(
    finding: Mapping[str, Any],
    *,
    index: int,
) -> tuple[int | None, int | None]:
    pairs: list[tuple[Any, Any]] = []
    for field in _OFFSET_CONTAINER_FIELDS:
        if field not in finding or finding[field] is None:
            continue
        pairs.append(_offset_pair(finding[field], index=index, field=field))

    provenance_spans = finding.get("provenance_spans")
    if isinstance(provenance_spans, Mapping) and "finding" in provenance_spans:
        pairs.append(
            _offset_pair(
                provenance_spans["finding"],
                index=index,
                field="provenance_spans",
            )
        )

    has_start = any(field in finding for field in _OFFSET_START_FIELDS)
    has_end = any(field in finding for field in _OFFSET_END_FIELDS)
    if has_start or has_end:
        start = _first_present(finding, _OFFSET_START_FIELDS)
        end = _first_present(finding, _OFFSET_END_FIELDS)
        pairs.append((start, end))

    if not pairs:
        provenance = finding.get("provenance")
        if isinstance(provenance, Mapping):
            nested_start = _first_present(provenance, _OFFSET_START_FIELDS)
            nested_end = _first_present(provenance, _OFFSET_END_FIELDS)
            if nested_start is not None or nested_end is not None:
                pairs.append((nested_start, nested_end))

    if not pairs:
        return None, None

    first = _validated_offset_pair(pairs[0], index=index)
    for candidate in pairs[1:]:
        if _validated_offset_pair(candidate, index=index) != first:
            raise AmbiguousDicomSrItemPathError(
                f"finding at index {index} supplies conflicting source offsets"
            )
    return first


def _offset_pair(value: Any, *, index: int, field: str) -> tuple[Any, Any]:
    if isinstance(value, Mapping):
        start = _first_present(value, _OFFSET_START_FIELDS)
        end = _first_present(value, _OFFSET_END_FIELDS)
        return start, end
    if isinstance(value, SourceSpan):
        return value.start, value.end
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise DicomSrProvenanceError(
                f"{field} at finding index {index} must contain two offsets"
            )
        return value[0], value[1]
    raise TypeError(
        f"{field} at finding index {index} must be a mapping, span, or pair"
    )


def _validated_offset_pair(
    pair: tuple[Any, Any],
    *,
    index: int,
) -> tuple[int, int]:
    start, end = pair
    _validate_offsets(start, end, context=f"finding offsets at index {index}")
    return int(start), int(end)


def _path_from_offsets(
    start: int | None,
    end: int | None,
    span_index: Mapping[str, _SpanInfo],
    *,
    index: int,
) -> str:
    if start is None or end is None:
        raise DicomSrProvenanceError(
            f"finding at index {index} must supply an item path or source offsets"
        )
    candidates = [
        path
        for path, span in span_index.items()
        if span.start <= start and end <= span.end
    ]
    if len(candidates) > 1:
        raise AmbiguousDicomSrItemPathError(
            f"source offsets at finding index {index} match multiple item paths"
        )
    if not candidates:
        raise DicomSrProvenanceError(
            f"source offsets at finding index {index} do not match an item path"
        )
    return candidates[0]


def _path_from_aliases(
    value: Mapping[str, Any],
    *,
    context: str,
) -> str | None:
    paths: list[str] = []
    for field in _PATH_FIELDS:
        if field not in value or value[field] is None:
            continue
        paths.append(_coerce_item_path(value[field], context=context))
    if not paths:
        return None
    if len(set(paths)) > 1:
        raise AmbiguousDicomSrItemPathError(
            f"{context} supplies conflicting item paths"
        )
    return paths[0]


def _coerce_item_path(value: Any, *, context: str) -> str:
    if isinstance(value, str):
        path = value.strip()
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if not value or any(
            isinstance(part, bool) or not isinstance(part, int) or part <= 0
            for part in value
        ):
            raise DicomSrProvenanceError(f"{context} has an invalid item path")
        path = ".".join(str(part) for part in value)
    else:
        raise DicomSrProvenanceError(f"{context} has an invalid item path")
    _validate_item_path(path, context=context)
    return path


def _validate_item_path(value: Any, *, context: str) -> None:
    if not isinstance(value, str) or _ITEM_PATH_RE.fullmatch(value) is None:
        raise DicomSrProvenanceError(f"{context} must be a 1-based dotted item path")


def _validate_identifier(value: Any, *, context: str) -> None:
    if not isinstance(value, str) or not value:
        raise DicomSrProvenanceError(f"{context} must be non-empty")


def _validate_template_id(value: Any) -> None:
    if value is not None and (not isinstance(value, str) or not value):
        raise DicomSrProvenanceError("template identifier must be a non-empty string")


def _validate_offsets(
    start: Any,
    end: Any,
    *,
    context: str,
) -> None:
    if start is None and end is None:
        return
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        raise DicomSrProvenanceError(
            f"{context} offsets must be non-negative half-open integers"
        )


def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise TypeError("template identifier must be a string or integer")
    result = str(value).strip()
    return result or None


def _first_present(value: Mapping[str, Any], fields: Sequence[str]) -> Any:
    for field in fields:
        if field in value and value[field] is not None:
            return value[field]
    return None


def _record_from_mapping(
    value: Mapping[str, Any],
    *,
    index: int,
) -> DicomSrProvenanceRecord:
    if not isinstance(value, Mapping):
        raise TypeError(f"provenance record at index {index} must be a mapping")
    finding_id = _first_present(value, _IDENTIFIER_FIELDS)
    item_path = _path_from_aliases(
        value,
        context=f"provenance record at index {index}",
    )
    if finding_id is None or item_path is None:
        raise DicomSrProvenanceError(
            f"provenance record at index {index} is missing required references"
        )
    normalized_id = _finding_identifier(
        {"finding_id": finding_id},
        index=index,
    )
    start = _first_present(value, _OFFSET_START_FIELDS)
    end = _first_present(value, _OFFSET_END_FIELDS)
    if start is None and end is None:
        offset_pair = _first_present(value, _OFFSET_CONTAINER_FIELDS)
        if offset_pair is not None:
            start, end = _offset_pair(offset_pair, index=index, field="source_offsets")
    validated_start, validated_end = (
        _validated_offset_pair((start, end), index=index)
        if start is not None or end is not None
        else (None, None)
    )
    template_id = _finding_template_id(value, index=index)
    return DicomSrProvenanceRecord(
        finding_id=normalized_id,
        item_path=item_path,
        template_id=template_id,
        source_start=validated_start,
        source_end=validated_end,
    )


def _record_sort_key(record: DicomSrProvenanceRecord) -> tuple[Any, ...]:
    return (
        record.finding_id,
        record.item_path,
        record.source_start is None,
        record.source_start if record.source_start is not None else -1,
        record.source_end if record.source_end is not None else -1,
    )
