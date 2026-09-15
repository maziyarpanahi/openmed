"""PHI-safe provenance checks for clinical section boundaries.

Section detection produces character ranges, but later normalization stages may
move, split, or nest those ranges.  This module checks the structural contract
between those stages without retaining section text.  Findings contain only
offsets, fixed categories, and SHA-256 hashes of the relevant source slices.

The validator is deliberately mechanical and local-first.  It does not resolve
clinical labels, fetch a source map, or make a network request.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openmed.core.audit import hash_text

SECTION_PROVENANCE_SCHEMA_VERSION = 1
SECTION_PROVENANCE_ADVISORY = (
    "Section-boundary provenance is structural traceability for human review; "
    "it is not a clinical decision or compliance certification."
)

_SOURCE_START_KEYS = (
    "source_start",
    "source_begin",
    "original_start",
    "source_offset_start",
)
_SOURCE_END_KEYS = (
    "source_end",
    "source_stop",
    "original_end",
    "source_offset_end",
)
_SOURCE_MAP_START_KEYS = (*_SOURCE_START_KEYS, "start", "begin")
_SOURCE_MAP_END_KEYS = (*_SOURCE_END_KEYS, "end", "stop")
_SOURCE_REFERENCE_KEYS = (
    "source_map_ref",
    "source_ref",
    "source_reference",
    "source_id",
    "source_map_id",
    "source_key",
    "reference",
    "ref",
)
_SOURCE_HASH_KEYS = ("source_hash", "text_hash", "content_hash", "hash")
_IDENTIFIER_KEYS = ("section_id", "id", "key")
_PARENT_KEYS = ("parent_id", "parent_ref", "parent_key", "parent")

_CONFLICT_CODES = frozenset(
    {
        "overlap",
        "out_of_order",
        "source_overlap",
        "source_out_of_order",
        "reference_conflict",
        "hash_mismatch",
        "outside_parent",
        "source_outside_parent",
        "parent_conflict",
    }
)
_GAP_CODES = frozenset({"gap", "source_gap"})


@dataclass(frozen=True)
class SectionRange:
    """A source-indexed or normalized clinical section range.

    The class is an optional typed input convenience.  The validator also
    accepts existing :class:`SectionSpan` mappings, mappings with equivalent
    keys, objects exposing ``start``/``end``, and two- or three-item tuples.
    ``label`` and identifiers are used for matching only and are never copied
    into a provenance report.
    """

    label: str | None = None
    start: int = 0
    end: int = 0
    section_id: str | int | None = None
    parent_id: str | int | None = None
    source_start: int | None = None
    source_end: int | None = None
    source_ref: str | int | None = None
    source_hash: str | None = None
    source_map: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the structural input shape without any section text."""

        result: dict[str, Any] = {
            "label": self.label,
            "start": self.start,
            "end": self.end,
        }
        if self.section_id is not None:
            result["section_id"] = self.section_id
        if self.parent_id is not None:
            result["parent_id"] = self.parent_id
        if self.source_start is not None:
            result["source_start"] = self.source_start
        if self.source_end is not None:
            result["source_end"] = self.source_end
        if self.source_ref is not None:
            result["source_ref"] = self.source_ref
        if self.source_hash is not None:
            result["source_hash"] = self.source_hash
        if self.source_map is not None:
            result["source_map"] = dict(self.source_map)
        return result


# These aliases make the typed input contract easy to discover for callers that
# use "boundary" terminology rather than "range" terminology.
SectionBoundary = SectionRange
SectionSpanRange = SectionRange


@dataclass(frozen=True)
class SectionSourceMap:
    """A source-map entry for one normalized section range.

    ``reference`` is caller-owned metadata.  It is hashed before appearing in
    any report, so a reference cannot accidentally become a PHI side channel.
    """

    source_start: int
    source_end: int
    reference: str | int
    source_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a source-map input mapping."""

        result: dict[str, Any] = {
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_ref": self.reference,
        }
        if self.source_hash is not None:
            result["source_hash"] = self.source_hash
        return result


@dataclass(frozen=True)
class SectionProvenanceIssue:
    """One PHI-safe structural finding emitted by the validator."""

    category: str
    code: str
    index: int | None = None
    related_index: int | None = None
    start: int | None = None
    end: int | None = None
    related_start: int | None = None
    related_end: int | None = None
    source_hash: str | None = None
    related_hash: str | None = None
    expected_hash: str | None = None
    source_ref_hash: str | None = None

    @property
    def kind(self) -> str:
        """Return the stable machine-readable finding code."""

        return self.code

    @property
    def text_hash(self) -> str | None:
        """Backward-friendly alias for the hash of the affected source slice."""

        return self.source_hash

    @property
    def range_hash(self) -> str | None:
        """Alias for the affected source hash."""

        return self.source_hash

    @property
    def offsets(self) -> dict[str, int | None]:
        """Return offsets without exposing source content."""

        return {
            "start": self.start,
            "end": self.end,
            "related_start": self.related_start,
            "related_end": self.related_end,
        }

    @property
    def message(self) -> str:
        """Return a sanitized diagnostic containing no caller-owned values."""

        if self.index is None:
            return f"section provenance {self.code}"
        return f"section provenance {self.code} at range index {self.index}"

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, PHI-free issue mapping."""

        result: dict[str, Any] = {
            "category": self.category,
            "code": self.code,
            "message": self.message,
            "index": self.index,
            "related_index": self.related_index,
            "start": self.start,
            "end": self.end,
            "related_start": self.related_start,
            "related_end": self.related_end,
        }
        optional = {
            "source_hash": self.source_hash,
            "related_hash": self.related_hash,
            "expected_hash": self.expected_hash,
            "source_ref_hash": self.source_ref_hash,
        }
        result.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        return result


@dataclass(frozen=True)
class SectionProvenanceRecord:
    """PHI-free structural summary for one input range."""

    index: int
    start: int | None
    end: int | None
    source_start: int | None
    source_end: int | None
    source_hash: str | None
    normalized_hash: str | None
    source_ref_hash: str | None
    parent_index: int | None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic structural range summary."""

        return {
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_hash": self.source_hash,
            "normalized_hash": self.normalized_hash,
            "source_ref_hash": self.source_ref_hash,
            "parent_index": self.parent_index,
        }


@dataclass(frozen=True)
class SectionProvenanceReport:
    """Deterministic, serializable result of section-boundary validation."""

    valid: bool
    document_hash: str
    ranges: tuple[SectionProvenanceRecord, ...]
    issues: tuple[SectionProvenanceIssue, ...] = ()
    schema_version: int = SECTION_PROVENANCE_SCHEMA_VERSION

    @property
    def ok(self) -> bool:
        """Return whether no structural findings were emitted."""

        return self.valid

    @property
    def is_valid(self) -> bool:
        """Return whether no structural findings were emitted."""

        return self.valid

    @property
    def has_errors(self) -> bool:
        """Return whether the report contains any findings."""

        return bool(self.issues)

    @property
    def errors(self) -> tuple[SectionProvenanceIssue, ...]:
        """Return all findings as an errors-compatible alias."""

        return self.issues

    @property
    def findings(self) -> tuple[SectionProvenanceIssue, ...]:
        """Return all findings as a review-friendly alias."""

        return self.issues

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the sorted unique finding categories."""

        return tuple(sorted({issue.category for issue in self.issues}))

    @property
    def gaps(self) -> tuple[SectionProvenanceIssue, ...]:
        """Return coverage findings."""

        return tuple(issue for issue in self.issues if issue.code in _GAP_CODES)

    @property
    def conflicts(self) -> tuple[SectionProvenanceIssue, ...]:
        """Return overlap, ordering, mapping, and containment conflicts."""

        return tuple(
            issue
            for issue in self.issues
            if issue.code in _CONFLICT_CODES
            or issue.category in {"parent_containment", "source_map"}
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready report with no raw source text."""

        issues = [issue.to_dict() for issue in self.issues]
        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "document_hash": self.document_hash,
            "range_count": len(self.ranges),
            "issue_count": len(issues),
            "categories": list(self.categories),
            "ranges": [item.to_dict() for item in self.ranges],
            "issues": issues,
            "advisory": SECTION_PROVENANCE_ADVISORY,
        }

    def as_dict(self) -> dict[str, Any]:
        """Alias for :meth:`to_dict`."""

        return self.to_dict()

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize the report with stable key ordering."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
            separators=None if indent is not None else (",", ":"),
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write the PHI-free report to a local JSON path."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def raise_for_errors(self) -> None:
        """Raise a sanitized error when the report is not valid."""

        if self.issues:
            raise SectionProvenanceError(
                f"section provenance validation found {len(self.issues)} issue(s)"
            )

    def __bool__(self) -> bool:
        return self.valid

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class SectionProvenanceError(ValueError):
    """Raised by explicit strict validation without including source text."""


@dataclass(frozen=True)
class _NormalizedRange:
    index: int
    identifier_key: tuple[str, str | int]
    start: int | None
    end: int | None
    source_start: int | None
    source_end: int | None
    source_ref: str
    explicit_source_ref: bool
    expected_source_hash: str | None
    parent_key: tuple[str, str | int] | None
    inline_parent: tuple[int, int] | None
    source_map_present: bool
    is_parent_definition: bool = False

    @property
    def valid_bounds(self) -> bool:
        return (
            self.start is not None
            and self.end is not None
            and self.source_start is not None
            and self.source_end is not None
        )


def validate_section_provenance(
    text: str | None = None,
    sections: Iterable[Any] | Any | None = None,
    source_map: Mapping[Any, Any] | Sequence[Any] | Any | None = None,
    *,
    source_text: str | None = None,
    section_ranges: Iterable[Any] | Any | None = None,
    parent_sections: Iterable[Any] | Any | None = None,
    parents: Iterable[Any] | Any | None = None,
    require_coverage: bool = True,
    require_source_map: bool = False,
    strict: bool = False,
) -> SectionProvenanceReport:
    """Validate section ranges and return a PHI-safe provenance report.

    Args:
        text: Exact source document used by the section offsets.
        sections: Normalized section ranges.  Mappings, ``SectionSpan`` values,
            :class:`SectionRange` values, span-like objects, and two-item
            ``(start, end)`` tuples are accepted.
        source_map: Optional local source-map entries keyed by section id or
            input index.  Entries may contain ``source_start``/``source_end``
            and ``source_ref``/``reference``.  Equivalent fields may be present
            directly on a section.
        parent_sections: Optional parent range definitions.  ``parents`` is a
            spelling alias.  Child ranges refer to a parent with ``parent_id``.
        require_coverage: Require top-level ranges to cover ``[0, len(text))``.
        require_source_map: Require every range to carry an explicit source-map
            reference.  Without this option, identity mappings are accepted for
            ordinary source-indexed section spans.
        strict: Raise :class:`SectionProvenanceError` after building the report
            when any finding is present.

    Returns:
        A deterministic report.  It contains offsets, hashes, categories, and
        structural counts only; no section label, identifier, or source text is
        copied into the report.

    The function is deterministic and never performs network or model I/O.
    """

    if source_text is not None:
        if text is not None:
            raise ValueError("text and source_text cannot both be provided")
        text = source_text
    if section_ranges is not None:
        if sections is not None:
            raise ValueError("sections and section_ranges cannot both be provided")
        sections = section_ranges
    if not isinstance(text, str):
        raise TypeError("section provenance text must be a string")
    if parents is not None:
        if parent_sections is not None:
            raise ValueError("parent_sections and parents cannot both be provided")
        parent_sections = parents

    raw_sections = _as_records(sections)
    raw_parents = _as_records(parent_sections) if parent_sections is not None else []
    issues: list[SectionProvenanceIssue] = []
    normalized: list[_NormalizedRange] = []

    for index, raw in enumerate(raw_sections):
        item, item_issues = _normalize_range(
            raw,
            index=index,
            text_length=len(text),
            source_map=source_map,
            require_source_map=require_source_map,
        )
        normalized.append(item)
        issues.extend(item_issues)

    parent_offset = len(normalized)
    for offset, raw in enumerate(raw_parents):
        index = parent_offset + offset
        item, item_issues = _normalize_range(
            raw,
            index=index,
            text_length=len(text),
            source_map=None,
            require_source_map=False,
            is_parent_definition=True,
        )
        normalized.append(item)
        issues.extend(item_issues)

    _add_duplicate_identifier_findings(raw_sections, issues)
    _resolve_parent_indices(normalized, issues)
    _check_ordering_and_coverage(
        text,
        normalized,
        issues,
        require_coverage=require_coverage,
    )
    _check_parent_containment(text, normalized, issues)
    _check_source_map_consistency(text, normalized, issues)

    issues = _sorted_unique_issues(issues)
    records = tuple(_to_public_record(text, item, normalized) for item in normalized)
    report = SectionProvenanceReport(
        valid=not issues,
        document_hash=hash_text(text),
        ranges=records,
        issues=tuple(issues),
    )
    if strict:
        report.raise_for_errors()
    return report


def validate_section_ranges(
    text: str,
    sections: Iterable[Any] | Any,
    source_map: Mapping[Any, Any] | Sequence[Any] | Any | None = None,
    **kwargs: Any,
) -> SectionProvenanceReport:
    """Alias for :func:`validate_section_provenance`."""

    return validate_section_provenance(text, sections, source_map, **kwargs)


def validate_section_boundaries(
    text: str,
    sections: Iterable[Any] | Any,
    source_map: Mapping[Any, Any] | Sequence[Any] | Any | None = None,
    **kwargs: Any,
) -> SectionProvenanceReport:
    """Alias for :func:`validate_section_provenance`."""

    return validate_section_provenance(text, sections, source_map, **kwargs)


check_section_provenance = validate_section_provenance


def _as_records(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, Mapping) or _has_span_attributes(value):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _has_span_attributes(value: Any) -> bool:
    return hasattr(value, "start") and hasattr(value, "end")


def _fields(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, SectionRange):
        return value.to_dict()
    if isinstance(value, SectionSourceMap):
        return value.to_dict()
    if _has_span_attributes(value):
        result: dict[str, Any] = {}
        for key in (
            "label",
            "name",
            "id",
            "section_id",
            "start",
            "end",
            "parent_id",
            "parent_ref",
            "parent_key",
            "parent",
            "source_start",
            "source_end",
            "source_ref",
            "source_map_ref",
            "source_hash",
            "source_map",
            "source_range",
            "source_span",
            "source",
        ):
            if hasattr(value, key):
                result[key] = getattr(value, key)
        return result
    if isinstance(value, (tuple, list)) and len(value) in {2, 3}:
        if len(value) == 2:
            return {"start": value[0], "end": value[1]}
        if isinstance(value[0], str):
            return {"label": value[0], "start": value[1], "end": value[2]}
        return {"start": value[0], "end": value[1], "label": value[2]}
    return None


def _first(fields: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in fields:
            return fields[key]
    return None


def _integer(value: Any) -> int | None:
    if type(value) is int:
        return value
    return None


def _identifier(value: Any, index: int) -> tuple[str, str | int]:
    if type(value) is int:
        return ("int", value)
    if isinstance(value, str) and value:
        return ("str", value)
    return ("index", index)


def _safe_reference(
    value: Any,
    index: int,
    *,
    start: int | None = None,
    end: int | None = None,
) -> tuple[str, bool]:
    if type(value) is int:
        return str(value), True
    if isinstance(value, str) and value:
        return value, True
    if start is not None and end is not None:
        return f"identity:{start}:{end}", False
    return f"identity:{index}", False


def _source_entry(
    raw: Any,
    fields: Mapping[str, Any],
    *,
    index: int,
    source_map: Any,
) -> tuple[dict[str, Any], bool]:
    entry: dict[str, Any] = {}
    present = False

    for key in ("source_map", "source_range", "source_span"):
        candidate = fields.get(key)
        if isinstance(candidate, Mapping):
            entry.update(candidate)
            present = True
        elif candidate is not None:
            entry["source_ref"] = candidate
            present = True
    if isinstance(fields.get("source"), Mapping):
        entry.update(fields["source"])
        present = True

    direct_keys = (*_SOURCE_START_KEYS, *_SOURCE_END_KEYS, *_SOURCE_REFERENCE_KEYS)
    if any(key in fields for key in direct_keys) or any(
        key in fields for key in _SOURCE_HASH_KEYS
    ):
        entry.update({key: fields[key] for key in direct_keys if key in fields})
        entry.update({key: fields[key] for key in _SOURCE_HASH_KEYS if key in fields})
        present = True

    if source_map is None:
        return entry, present

    mapped, mapped_present, mapped_key = _lookup_source_map(
        source_map,
        fields,
        index=index,
    )
    if mapped_present:
        if mapped_key is not None and not any(
            key in entry for key in _SOURCE_REFERENCE_KEYS
        ):
            entry["source_ref"] = mapped_key
        entry.update(mapped)
        present = True
    return entry, present


def _lookup_source_map(
    source_map: Any,
    fields: Mapping[str, Any],
    *,
    index: int,
) -> tuple[dict[str, Any], bool, Any | None]:
    if isinstance(source_map, SectionSourceMap):
        return source_map.to_dict(), True, None
    if isinstance(source_map, Mapping):
        if _looks_like_source_entry(source_map):
            return dict(source_map), True, None
        identifier = _first(fields, _IDENTIFIER_KEYS)
        candidates: list[Any] = [identifier, index, str(index)]
        explicit_ref = _first(fields, _SOURCE_REFERENCE_KEYS)
        if explicit_ref is not None:
            candidates.insert(0, explicit_ref)
        for candidate in candidates:
            try:
                value = source_map[candidate]
            except (KeyError, TypeError):
                continue
            return _source_value_fields(value), True, candidate
        return {}, False, None
    if isinstance(source_map, Sequence) and not isinstance(
        source_map, (str, bytes, bytearray)
    ):
        if index < len(source_map):
            value = source_map[index]
            return _source_value_fields(value), True, None
    return {}, False, None


def _source_value_fields(value: Any) -> dict[str, Any]:
    if isinstance(value, SectionSourceMap):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (tuple, list)) and len(value) in {2, 3}:
        if len(value) == 2:
            return {"source_start": value[0], "source_end": value[1]}
        if isinstance(value[0], str):
            return {
                "source_ref": value[0],
                "source_start": value[1],
                "source_end": value[2],
            }
        return {
            "source_start": value[0],
            "source_end": value[1],
            "source_ref": value[2],
        }
    return {"source_ref": value}


def _looks_like_source_entry(value: Mapping[Any, Any]) -> bool:
    keys = set(value)
    return bool(
        keys.intersection(_SOURCE_START_KEYS)
        or keys.intersection(_SOURCE_END_KEYS)
        or "start" in keys
        or "end" in keys
        or keys.intersection(_SOURCE_REFERENCE_KEYS)
        or keys.intersection(_SOURCE_HASH_KEYS)
    )


def _normalize_range(
    raw: Any,
    *,
    index: int,
    text_length: int,
    source_map: Any,
    require_source_map: bool,
    is_parent_definition: bool = False,
) -> tuple[_NormalizedRange, list[SectionProvenanceIssue]]:
    issues: list[SectionProvenanceIssue] = []
    fields = _fields(raw)
    if fields is None:
        return (
            _NormalizedRange(
                index=index,
                identifier_key=("index", index),
                start=None,
                end=None,
                source_start=None,
                source_end=None,
                source_ref=f"identity:{index}",
                explicit_source_ref=False,
                expected_source_hash=None,
                parent_key=None,
                inline_parent=None,
                source_map_present=False,
                is_parent_definition=is_parent_definition,
            ),
            [
                SectionProvenanceIssue(
                    category="invalid_range",
                    code="invalid_record",
                    index=index,
                )
            ],
        )

    start = _integer(fields.get("normalized_start", fields.get("start")))
    end = _integer(fields.get("normalized_end", fields.get("end")))
    if start is None or end is None:
        issues.append(
            SectionProvenanceIssue(
                category="invalid_range",
                code="missing_offsets",
                index=index,
            )
        )
    elif start < 0 or end > text_length or end <= start:
        issues.append(
            SectionProvenanceIssue(
                category="invalid_range",
                code="out_of_bounds",
                index=index,
                start=start,
                end=end,
            )
        )

    source_entry, source_present = _source_entry(
        raw,
        fields,
        index=index,
        source_map=source_map,
    )
    source_start = _integer(_first(source_entry, _SOURCE_MAP_START_KEYS))
    source_end = _integer(_first(source_entry, _SOURCE_MAP_END_KEYS))
    if source_start is None and source_end is None:
        source_start, source_end = start, end
    elif source_start is None or source_end is None:
        issues.append(
            SectionProvenanceIssue(
                category="source_map",
                code="incomplete_mapping",
                index=index,
                start=start,
                end=end,
            )
        )
    if (
        source_start is not None
        and source_end is not None
        and (source_start < 0 or source_end > text_length or source_end <= source_start)
    ):
        issues.append(
            SectionProvenanceIssue(
                category="source_map",
                code="invalid_mapping_offsets",
                index=index,
                start=source_start,
                end=source_end,
            )
        )

    raw_ref = _first(source_entry, _SOURCE_REFERENCE_KEYS)
    source_ref, explicit_source_ref = _safe_reference(
        raw_ref,
        index,
        start=start,
        end=end,
    )
    if require_source_map and not explicit_source_ref:
        issues.append(
            SectionProvenanceIssue(
                category="source_map",
                code="missing_reference",
                index=index,
                start=start,
                end=end,
            )
        )
    elif source_map is not None and not source_present:
        issues.append(
            SectionProvenanceIssue(
                category="source_map",
                code="missing_mapping",
                index=index,
                start=start,
                end=end,
            )
        )

    parent_value = _first(fields, _PARENT_KEYS)
    parent_key = None
    inline_parent = None
    if isinstance(parent_value, Mapping):
        parent_id = _first(parent_value, _IDENTIFIER_KEYS + ("parent_id",))
        if parent_id is not None:
            parent_key = _identifier(parent_id, -1)
        inline_start = _integer(
            parent_value.get("normalized_start", parent_value.get("start"))
        )
        inline_end = _integer(
            parent_value.get("normalized_end", parent_value.get("end"))
        )
        if inline_start is not None and inline_end is not None:
            inline_parent = (inline_start, inline_end)
    elif parent_value is not None:
        parent_key = _identifier(parent_value, -1)

    expected_hash = _first(source_entry, _SOURCE_HASH_KEYS)
    if expected_hash is not None and not isinstance(expected_hash, str):
        expected_hash = None

    return (
        _NormalizedRange(
            index=index,
            identifier_key=_identifier(_first(fields, _IDENTIFIER_KEYS), index),
            start=start,
            end=end,
            source_start=source_start,
            source_end=source_end,
            source_ref=source_ref,
            explicit_source_ref=explicit_source_ref,
            expected_source_hash=expected_hash,
            parent_key=parent_key,
            inline_parent=inline_parent,
            source_map_present=source_present,
            is_parent_definition=is_parent_definition,
        ),
        issues,
    )


def _add_duplicate_identifier_findings(
    raw_sections: Sequence[Any],
    issues: list[SectionProvenanceIssue],
) -> None:
    seen: dict[tuple[str, str | int], int] = {}
    for index, raw in enumerate(raw_sections):
        fields = _fields(raw)
        if fields is None:
            continue
        value = _first(fields, _IDENTIFIER_KEYS)
        if value is None:
            continue
        key = _identifier(value, index)
        previous = seen.get(key)
        if previous is not None:
            issues.append(
                SectionProvenanceIssue(
                    category="source_map",
                    code="duplicate_reference",
                    index=index,
                    related_index=previous,
                )
            )
        else:
            seen[key] = index


def _resolve_parent_indices(
    ranges: list[_NormalizedRange],
    issues: list[SectionProvenanceIssue],
) -> None:
    by_id: dict[tuple[str, str | int], int] = {}
    for item in ranges:
        if item.identifier_key in by_id:
            issues.append(
                SectionProvenanceIssue(
                    category="parent_containment",
                    code="parent_conflict",
                    index=item.index,
                    related_index=by_id[item.identifier_key],
                )
            )
        else:
            by_id[item.identifier_key] = item.index


def _parent_index_from_key(
    item: _NormalizedRange,
    ranges: Sequence[_NormalizedRange],
) -> int | None:
    if item.parent_key is None:
        return None
    key = item.parent_key
    if key[0] == "index":
        return int(key[1]) if 0 <= int(key[1]) < len(ranges) else None
    if key[0] == "int":
        candidate = int(key[1])
        if 0 <= candidate < len(ranges):
            return candidate
    for candidate in ranges:
        if candidate.identifier_key == key:
            return candidate.index
    return None


def _check_ordering_and_coverage(
    text: str,
    ranges: Sequence[_NormalizedRange],
    issues: list[SectionProvenanceIssue],
    *,
    require_coverage: bool,
) -> None:
    roots = [
        item
        for item in ranges
        if item.parent_key is None and item.inline_parent is None
    ]
    original_roots = [item for item in roots if not item.is_parent_definition]
    parent_roots = [item for item in roots if item.is_parent_definition]
    if original_roots:
        coverage_ranges = original_roots
    elif parent_roots:
        coverage_ranges = parent_roots
    else:
        coverage_ranges = [item for item in ranges if not item.is_parent_definition]

    _check_sibling_order(text, coverage_ranges, issues, source=False)
    _check_sibling_order(text, coverage_ranges, issues, source=True)
    if require_coverage:
        _check_coverage(
            text,
            coverage_ranges,
            issues,
            start=0,
            end=len(text),
            source=False,
        )
        _check_coverage(
            text,
            coverage_ranges,
            issues,
            start=0,
            end=len(text),
            source=True,
        )

    # Also validate sibling order for nested sections, even though nested
    # ranges must not participate in top-level document coverage.
    groups: dict[tuple[str, str | int] | None, list[_NormalizedRange]] = {}
    for item in ranges:
        groups.setdefault(item.parent_key, []).append(item)
    for parent_key, group in groups.items():
        if group is coverage_ranges or parent_key is None:
            continue
        _check_sibling_order(text, group, issues, source=False)
        _check_sibling_order(text, group, issues, source=True)


def _check_sibling_order(
    text: str,
    ranges: Sequence[_NormalizedRange],
    issues: list[SectionProvenanceIssue],
    *,
    source: bool,
) -> None:
    valid = [item for item in ranges if item.valid_bounds]
    previous: _NormalizedRange | None = None
    start_name = "source_start" if source else "start"
    end_name = "source_end" if source else "end"
    for item in valid:
        start = getattr(item, start_name)
        end = getattr(item, end_name)
        if previous is not None:
            previous_start = getattr(previous, start_name)
            previous_end = getattr(previous, end_name)
            if start < previous_start:
                issues.append(
                    SectionProvenanceIssue(
                        category="ordering" if not source else "source_map",
                        code="out_of_order" if not source else "source_out_of_order",
                        index=item.index,
                        related_index=previous.index,
                        start=start,
                        end=end,
                        related_start=previous_start,
                        related_end=previous_end,
                        source_hash=_slice_hash(text, start, end),
                        related_hash=_slice_hash(text, previous_start, previous_end),
                    )
                )
            if start < previous_end:
                issues.append(
                    SectionProvenanceIssue(
                        category="ordering" if not source else "source_map",
                        code="overlap" if not source else "source_overlap",
                        index=item.index,
                        related_index=previous.index,
                        start=start,
                        end=end,
                        related_start=previous_start,
                        related_end=previous_end,
                        source_hash=_slice_hash(text, start, end),
                        related_hash=_slice_hash(text, previous_start, previous_end),
                    )
                )
        previous = item


def _check_coverage(
    text: str,
    ranges: Sequence[_NormalizedRange],
    issues: list[SectionProvenanceIssue],
    *,
    start: int,
    end: int,
    source: bool,
) -> None:
    valid = [item for item in ranges if item.valid_bounds]
    if source:
        valid.sort(key=lambda item: (item.source_start or 0, item.index))
        start_name = "source_start"
        end_name = "source_end"
    else:
        valid.sort(key=lambda item: (item.start or 0, item.index))
        start_name = "start"
        end_name = "end"
    cursor = start
    for item in valid:
        item_start = getattr(item, start_name)
        item_end = getattr(item, end_name)
        if item_start > cursor:
            code = "source_gap" if source else "gap"
            issues.append(
                SectionProvenanceIssue(
                    category="source_map" if source else "coverage",
                    code=code,
                    index=item.index,
                    start=cursor,
                    end=item_start,
                    source_hash=_slice_hash(text, cursor, item_start),
                )
            )
        if item_end > cursor:
            cursor = item_end
    if cursor < end:
        code = "source_gap" if source else "gap"
        issues.append(
            SectionProvenanceIssue(
                category="source_map" if source else "coverage",
                code=code,
                start=cursor,
                end=end,
                source_hash=_slice_hash(text, cursor, end),
            )
        )


def _check_parent_containment(
    text: str,
    ranges: Sequence[_NormalizedRange],
    issues: list[SectionProvenanceIssue],
) -> None:
    for item in ranges:
        if item.parent_key is None and item.inline_parent is None:
            continue
        parent_index = _parent_index_from_key(item, ranges)
        parent = ranges[parent_index] if parent_index is not None else None
        if item.inline_parent is not None:
            parent_start, parent_end = item.inline_parent
        elif parent is not None:
            parent_start, parent_end = parent.start, parent.end
        else:
            issues.append(
                SectionProvenanceIssue(
                    category="parent_containment",
                    code="missing_parent",
                    index=item.index,
                    start=item.start,
                    end=item.end,
                )
            )
            continue
        if (
            item.start is None
            or item.end is None
            or parent_start is None
            or parent_end is None
            or item.start < parent_start
            or item.end > parent_end
            or item.end <= item.start
        ):
            issues.append(
                SectionProvenanceIssue(
                    category="parent_containment",
                    code="outside_parent",
                    index=item.index,
                    related_index=parent_index,
                    start=item.start,
                    end=item.end,
                    related_start=parent_start,
                    related_end=parent_end,
                    source_hash=_slice_hash(text, item.start, item.end),
                )
            )
        if parent_index == item.index:
            issues.append(
                SectionProvenanceIssue(
                    category="parent_containment",
                    code="parent_conflict",
                    index=item.index,
                    related_index=parent_index,
                )
            )
        if parent is not None and (
            item.source_start is not None
            and item.source_end is not None
            and parent.source_start is not None
            and parent.source_end is not None
            and (
                item.source_start < parent.source_start
                or item.source_end > parent.source_end
            )
        ):
            issues.append(
                SectionProvenanceIssue(
                    category="parent_containment",
                    code="source_outside_parent",
                    index=item.index,
                    related_index=parent_index,
                    start=item.source_start,
                    end=item.source_end,
                    related_start=parent.source_start,
                    related_end=parent.source_end,
                    source_hash=_slice_hash(text, item.source_start, item.source_end),
                )
            )


def _check_source_map_consistency(
    text: str,
    ranges: Sequence[_NormalizedRange],
    issues: list[SectionProvenanceIssue],
) -> None:
    valid = [item for item in ranges if item.valid_bounds]

    references: dict[str, _NormalizedRange] = {}
    for item in valid:
        existing = references.get(item.source_ref)
        if existing is not None and (
            existing.source_start != item.source_start
            or existing.source_end != item.source_end
        ):
            issues.append(
                SectionProvenanceIssue(
                    category="source_map",
                    code="reference_conflict",
                    index=item.index,
                    related_index=existing.index,
                    start=item.source_start,
                    end=item.source_end,
                    related_start=existing.source_start,
                    related_end=existing.source_end,
                    source_hash=_slice_hash(text, item.source_start, item.source_end),
                    related_hash=_slice_hash(
                        text,
                        existing.source_start,
                        existing.source_end,
                    ),
                    source_ref_hash=hash_text(item.source_ref),
                )
            )
        else:
            references[item.source_ref] = item

        actual_hash = _slice_hash(text, item.source_start, item.source_end)
        if (
            item.expected_source_hash is not None
            and actual_hash != item.expected_source_hash
        ):
            issues.append(
                SectionProvenanceIssue(
                    category="source_map",
                    code="hash_mismatch",
                    index=item.index,
                    start=item.source_start,
                    end=item.source_end,
                    source_hash=actual_hash,
                    expected_hash=item.expected_source_hash,
                    source_ref_hash=hash_text(item.source_ref),
                )
            )


def _slice_hash(text: str, start: int | None, end: int | None) -> str | None:
    if start is None or end is None or start < 0 or end < start or end > len(text):
        return None
    return hash_text(text[start:end])


def _to_public_record(
    text: str,
    item: _NormalizedRange,
    ranges: Sequence[_NormalizedRange],
) -> SectionProvenanceRecord:
    parent_index = _parent_index_from_key(item, ranges)
    return SectionProvenanceRecord(
        index=item.index,
        start=item.start,
        end=item.end,
        source_start=item.source_start,
        source_end=item.source_end,
        source_hash=_slice_hash(text, item.source_start, item.source_end),
        normalized_hash=_slice_hash(text, item.start, item.end),
        source_ref_hash=hash_text(item.source_ref),
        parent_index=parent_index,
    )


def _sorted_unique_issues(
    issues: Iterable[SectionProvenanceIssue],
) -> list[SectionProvenanceIssue]:
    unique = set(issues)
    return sorted(
        unique,
        key=lambda issue: (
            issue.index is None,
            issue.index if issue.index is not None else -1,
            issue.start is None,
            issue.start if issue.start is not None else -1,
            issue.category,
            issue.code,
            issue.related_index is None,
            issue.related_index if issue.related_index is not None else -1,
            issue.end is None,
            issue.end if issue.end is not None else -1,
        ),
    )


__all__ = [
    "SECTION_PROVENANCE_ADVISORY",
    "SECTION_PROVENANCE_SCHEMA_VERSION",
    "SectionBoundary",
    "SectionProvenanceError",
    "SectionProvenanceIssue",
    "SectionProvenanceRecord",
    "SectionProvenanceReport",
    "SectionRange",
    "SectionSourceMap",
    "SectionSpanRange",
    "check_section_provenance",
    "validate_section_boundaries",
    "validate_section_provenance",
    "validate_section_ranges",
]
