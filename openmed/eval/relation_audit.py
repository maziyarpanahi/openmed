"""Deterministic, raw-text-free audit summaries for relation candidates.

Candidate generation and filtering often happen before a relation reaches a
scorer or decoder.  This module keeps the useful aggregate evidence from that
stage without retaining candidate endpoints, source text, offsets, or other
record-level identifiers.  It is deliberately a pure standard-library
helper: callers provide already-created candidate records and the report is
computed locally.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

RELATION_CANDIDATE_AUDIT = "relation_candidate_audit"
RELATION_CANDIDATE_AUDIT_ARTIFACT = RELATION_CANDIDATE_AUDIT
RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION = 1
RELATION_AUDIT_ARTIFACT = RELATION_CANDIDATE_AUDIT_ARTIFACT
RELATION_AUDIT_SCHEMA_VERSION = RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION

UNSECTIONED_SECTION = "unsectioned"
UNKNOWN_RELATION_FAMILY = "unknown"
ACCEPTED_FILTERING_REASON = "accepted"
OTHER_FILTERING_REASON = "other"

_CATEGORY_MAX_LENGTH = 64
_CATEGORY_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SEPARATOR_RE = re.compile(r"[\s\-]+")
_NON_CATEGORY_RE = re.compile(r"[^a-z0-9_]+")
_REPEATED_SEPARATOR_RE = re.compile(r"_+")

_FAMILY_KEYS = ("relation_family", "family", "relation_group")
_RELATION_TYPE_KEYS = ("relation_type", "predicate", "type", "label")
_SECTION_KEYS = ("section", "section_name", "clinical_section")
_FILTER_REASON_KEYS = (
    "filtering_reason",
    "filter_reason",
    "rejection_reason",
    "drop_reason",
    "reason",
)
_METADATA_KEYS = ("metadata", "provenance")
_NESTED_RELATION_KEYS = ("relation", "edge")
_ENDPOINT_KEYS = ("head", "tail", "attribute", "source", "target")


def _normalise_category(value: Any, fallback: str) -> str:
    """Return a bounded category token without exposing arbitrary input text."""

    if not isinstance(value, str):
        return fallback
    normalized = value.strip().casefold()
    if not normalized:
        return fallback
    normalized = _SEPARATOR_RE.sub("_", normalized)
    normalized = _NON_CATEGORY_RE.sub("_", normalized)
    normalized = _REPEATED_SEPARATOR_RE.sub("_", normalized).strip("_")
    if len(normalized) > _CATEGORY_MAX_LENGTH or not _CATEGORY_RE.fullmatch(normalized):
        return fallback
    return normalized


def _normalise_count_mapping(
    value: Mapping[Any, Any] | None,
    *,
    fallback: str,
) -> MappingProxyType:
    """Normalize and sort one aggregate count mapping."""

    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise ValueError("audit count dimensions must be mappings")

    counts: Counter[str] = Counter()
    for key, raw_count in value.items():
        if isinstance(raw_count, bool) or not isinstance(raw_count, int):
            raise ValueError("audit counts must be non-negative integers")
        if raw_count < 0:
            raise ValueError("audit counts must be non-negative integers")
        counts[_normalise_category(key, fallback)] += raw_count
    return MappingProxyType(dict(sorted(counts.items())))


def _value(source: Any, keys: Sequence[str]) -> Any:
    """Read one of *keys* without stringifying or logging the source value."""

    if source is None:
        return None
    if isinstance(source, Mapping):
        for key in keys:
            try:
                value = source.get(key)
            except (AttributeError, KeyError, TypeError, ValueError):
                continue
            if value is not None:
                return value
        return None

    for key in keys:
        try:
            value = getattr(source, key)
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if value is not None and not callable(value):
            return value
    return None


def _nested_value(source: Any, keys: Sequence[str]) -> Any:
    """Read a category field from a candidate or its safe nested containers."""

    direct = _value(source, keys)
    if direct is not None:
        return direct

    for nested_key in _NESTED_RELATION_KEYS:
        nested = _value(source, (nested_key,))
        if nested is not None and nested is not source:
            nested_value = _value(nested, keys)
            if nested_value is not None:
                return nested_value

    metadata = _value(source, _METADATA_KEYS)
    if isinstance(metadata, Mapping):
        return _value(metadata, keys)
    return None


def _nested_section(source: Any) -> Any:
    """Read an explicit section, including a relation's endpoint metadata."""

    direct = _nested_value(source, _SECTION_KEYS)
    if direct is not None:
        return direct

    sections: list[Any] = []
    for endpoint_key in _ENDPOINT_KEYS:
        endpoint = _nested_value(source, (endpoint_key,))
        section = _nested_value(endpoint, _SECTION_KEYS)
        if section is not None:
            sections.append(section)
    normalized = {
        _normalise_category(section, UNSECTIONED_SECTION) for section in sections
    }
    if len(normalized) == 1:
        return next(iter(normalized))
    if len(normalized) > 1:
        return "mixed"
    return None


def _relation_family(value: Any, fallback: str) -> str:
    """Normalize a family label, deriving it from a typed relation if needed."""

    relation_type = _normalise_category(value, "")
    if not relation_type:
        return fallback
    for separator in ("_to_", "_", ":"):
        if separator in relation_type:
            family = relation_type.split(separator, 1)[0]
            if family:
                return family
    return relation_type


def _derive_relation_family(candidate: Any, fallback: str) -> str:
    """Extract an explicit family or derive one from relation type metadata."""

    family = _nested_value(candidate, _FAMILY_KEYS)
    normalized_family = _normalise_category(family, "")
    if normalized_family:
        return normalized_family

    relation_type = _nested_value(candidate, _RELATION_TYPE_KEYS)
    normalized_type = _normalise_category(relation_type, "")
    if not normalized_type:
        return fallback
    return _relation_family(normalized_type, fallback)


def _derive_filtering_reason(candidate: Any, fallback: str) -> str:
    """Extract a filter reason while reducing assertion statuses to categories."""

    explicit = _nested_value(candidate, _FILTER_REASON_KEYS)
    normalized_explicit = _normalise_category(explicit, "")
    if normalized_explicit:
        return normalized_explicit

    filtered = _nested_value(candidate, ("filtered", "rejected", "pruned"))
    if isinstance(filtered, bool):
        return "filtered" if filtered else fallback

    status = _nested_value(candidate, ("status", "decision", "disposition"))
    normalized_status = _normalise_category(status, "")
    if normalized_status in {"accepted", "asserted", "confirmed", "kept", "retained"}:
        return fallback
    if normalized_status in {"refuted", "conditional", "possible", "uncertain"}:
        return f"assertion_{normalized_status}"
    if normalized_status:
        return normalized_status
    return fallback


def _coerce_record(
    candidate: Any,
    *,
    default_relation_family: str,
    default_section: str,
    default_filtering_reason: str,
) -> "RelationCandidateAuditRecord":
    """Coerce one candidate without retaining its record-level payload."""

    if isinstance(candidate, RelationCandidateAuditRecord):
        return candidate

    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        if len(candidate) >= 3:
            return RelationCandidateAuditRecord(
                relation_family=candidate[0] or default_relation_family,
                section=candidate[1] or default_section,
                filtering_reason=candidate[2] or default_filtering_reason,
            )

    relation_family = _derive_relation_family(candidate, default_relation_family)
    section = _normalise_category(
        _nested_section(candidate),
        default_section,
    )
    filtering_reason = _derive_filtering_reason(
        candidate,
        default_filtering_reason,
    )
    return RelationCandidateAuditRecord(
        relation_family=relation_family,
        section=section,
        filtering_reason=filtering_reason,
    )


@dataclass(frozen=True)
class RelationCandidateAuditRecord:
    """One privacy-safe relation-candidate category tuple.

    The record intentionally has no endpoint, text, offset, score, or
    identifier fields.  It can be constructed directly by a caller or
    derived from a richer candidate object with :meth:`from_candidate`.
    """

    relation_family: str = UNKNOWN_RELATION_FAMILY
    section: str = UNSECTIONED_SECTION
    filtering_reason: str = ACCEPTED_FILTERING_REASON

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_family",
            _normalise_category(self.relation_family, UNKNOWN_RELATION_FAMILY),
        )
        object.__setattr__(
            self,
            "section",
            _normalise_category(self.section, UNSECTIONED_SECTION),
        )
        object.__setattr__(
            self,
            "filtering_reason",
            _normalise_category(self.filtering_reason, ACCEPTED_FILTERING_REASON),
        )

    @classmethod
    def from_candidate(
        cls,
        candidate: Any,
        *,
        default_relation_family: str = UNKNOWN_RELATION_FAMILY,
        default_section: str = UNSECTIONED_SECTION,
        default_filtering_reason: str = ACCEPTED_FILTERING_REASON,
    ) -> "RelationCandidateAuditRecord":
        """Extract only safe category labels from a candidate-like object."""

        return _coerce_record(
            candidate,
            default_relation_family=_normalise_category(
                default_relation_family,
                UNKNOWN_RELATION_FAMILY,
            ),
            default_section=_normalise_category(default_section, UNSECTIONED_SECTION),
            default_filtering_reason=_normalise_category(
                default_filtering_reason,
                ACCEPTED_FILTERING_REASON,
            ),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "RelationCandidateAuditRecord":
        """Build a record from a mapping while ignoring all non-category fields."""

        if not isinstance(data, Mapping):
            raise TypeError("candidate audit input must be a mapping")
        return cls.from_candidate(data)

    def to_dict(self) -> dict[str, str]:
        """Return the category-only representation of this record."""

        return {
            "filtering_reason": self.filtering_reason,
            "relation_family": self.relation_family,
            "section": self.section,
        }


# Short aliases make the category-only input type discoverable without
# exposing or importing the clinical relation candidate model.
CandidateAuditRecord = RelationCandidateAuditRecord
RelationCandidateAuditEntry = RelationCandidateAuditRecord


@dataclass(frozen=True)
class RelationCandidateAuditReport:
    """Deterministic aggregate counts for relation-candidate diagnostics."""

    candidate_count: int
    by_relation_family: Mapping[str, int] = field(default_factory=dict)
    by_section: Mapping[str, int] = field(default_factory=dict)
    by_filtering_reason: Mapping[str, int] = field(default_factory=dict)
    schema_version: int = RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.candidate_count, bool) or not isinstance(
            self.candidate_count, int
        ):
            raise ValueError("candidate_count must be a non-negative integer")
        if self.candidate_count < 0:
            raise ValueError("candidate_count must be a non-negative integer")
        if self.schema_version != RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported relation-candidate audit schema version")

        object.__setattr__(
            self,
            "by_relation_family",
            _normalise_count_mapping(
                self.by_relation_family,
                fallback=UNKNOWN_RELATION_FAMILY,
            ),
        )
        object.__setattr__(
            self,
            "by_section",
            _normalise_count_mapping(
                self.by_section,
                fallback=UNSECTIONED_SECTION,
            ),
        )
        object.__setattr__(
            self,
            "by_filtering_reason",
            _normalise_count_mapping(
                self.by_filtering_reason,
                fallback=OTHER_FILTERING_REASON,
            ),
        )

    @property
    def total_candidates(self) -> int:
        """Return the total number of candidate records summarized."""

        return self.candidate_count

    @property
    def relation_family_counts(self) -> Mapping[str, int]:
        """Return counts keyed by relation family."""

        return self.by_relation_family

    @property
    def section_counts(self) -> Mapping[str, int]:
        """Return counts keyed by section."""

        return self.by_section

    @property
    def filtering_reason_counts(self) -> Mapping[str, int]:
        """Return counts keyed by filtering reason."""

        return self.by_filtering_reason

    @property
    def counts(self) -> Mapping[str, Mapping[str, int]]:
        """Return the three aggregate dimensions as one read-only mapping."""

        return MappingProxyType(
            {
                "by_filtering_reason": self.by_filtering_reason,
                "by_relation_family": self.by_relation_family,
                "by_section": self.by_section,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report containing aggregate categories only."""

        return {
            "artifact": RELATION_CANDIDATE_AUDIT_ARTIFACT,
            "by_filtering_reason": dict(self.by_filtering_reason),
            "by_relation_family": dict(self.by_relation_family),
            "by_section": dict(self.by_section),
            "candidate_count": self.candidate_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RelationCandidateAuditReport":
        """Build a report from its aggregate JSON representation."""

        if not isinstance(data, Mapping):
            raise TypeError("relation-candidate audit report must be a mapping")
        schema_version = data.get(
            "schema_version", RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION
        )
        if schema_version != RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION:
            raise ValueError("unsupported relation-candidate audit schema version")
        candidate_count = data.get("candidate_count", data.get("total_candidates", 0))
        return cls(
            candidate_count=candidate_count,
            by_relation_family=data.get("by_relation_family", {}),
            by_section=data.get("by_section", {}),
            by_filtering_reason=data.get("by_filtering_reason", {}),
            schema_version=schema_version,
        )

    @classmethod
    def read_json(cls, path: str | Path) -> "RelationCandidateAuditReport":
        """Read a JSON report from *path*."""

        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)

    def to_json(self, *, indent: int = 2) -> str:
        """Serialize the report deterministically as JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            sort_keys=True,
        )

    def write_json(self, path: str | Path, *, indent: int = 2) -> Path:
        """Write a deterministic JSON report to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return output_path

    def to_markdown(self) -> str:
        """Render a deterministic Markdown summary with no candidate details."""

        lines = [
            "# Relation Candidate Audit",
            "",
            "| Field | Value |",
            "|---|---:|",
            f"| Candidates | {self.candidate_count} |",
            f"| Schema Version | {self.schema_version} |",
            "",
            "## By Relation Family",
            "",
            "| Relation Family | Count |",
            "|---|---:|",
        ]
        lines.extend(
            f"| `{key}` | {count} |" for key, count in self.by_relation_family.items()
        )
        lines.extend(
            [
                "",
                "## By Section",
                "",
                "| Section | Count |",
                "|---|---:|",
            ]
        )
        lines.extend(f"| `{key}` | {count} |" for key, count in self.by_section.items())
        lines.extend(
            [
                "",
                "## By Filtering Reason",
                "",
                "| Filtering Reason | Count |",
                "|---|---:|",
            ]
        )
        lines.extend(
            f"| `{key}` | {count} |" for key, count in self.by_filtering_reason.items()
        )
        return "\n".join(lines) + "\n"

    def write_markdown(self, path: str | Path) -> Path:
        """Write a deterministic Markdown summary to *path*."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_markdown(), encoding="utf-8")
        return output_path

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to the serialized report."""

        return self.to_dict()[key]


def audit_relation_candidates(
    candidates: Iterable[Any] | Mapping[str, Any] | None,
    *,
    default_relation_family: str = UNKNOWN_RELATION_FAMILY,
    default_section: str = UNSECTIONED_SECTION,
    default_filtering_reason: str = ACCEPTED_FILTERING_REASON,
) -> RelationCandidateAuditReport:
    """Aggregate relation candidates by family, section, and filter reason.

    ``candidates`` may contain mappings, objects, three-item sequences, or
    :class:`RelationCandidateAuditRecord` values.  The accepted category keys
    are ``relation_family``/``family``, ``section``/``section_name``, and
    ``filtering_reason``/``filter_reason``.  Typed relation objects may supply
    ``relation_type`` or ``label`` instead; the family is derived from the
    prefix before ``_to_``.  Candidate text, endpoint fields, IDs, offsets,
    scores, and unrecognized metadata are never copied to the report.

    Args:
        candidates: Candidate-like records to summarize. ``None`` is treated
            as an empty input and a mapping is treated as one record.
        default_relation_family: Category used when no family is available.
        default_section: Category used when no section is available.
        default_filtering_reason: Category used when no filtering decision is
            available; this is normally ``"accepted"``.

    Returns:
        A deterministic, aggregate-only report.
    """

    family_default = _normalise_category(
        default_relation_family,
        UNKNOWN_RELATION_FAMILY,
    )
    section_default = _normalise_category(default_section, UNSECTIONED_SECTION)
    reason_default = _normalise_category(
        default_filtering_reason,
        ACCEPTED_FILTERING_REASON,
    )

    if candidates is None:
        records: Iterable[Any] = ()
    elif isinstance(candidates, Mapping):
        records = (candidates,)
    else:
        records = candidates

    family_counts: Counter[str] = Counter()
    section_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    candidate_count = 0
    for candidate in records:
        record = _coerce_record(
            candidate,
            default_relation_family=family_default,
            default_section=section_default,
            default_filtering_reason=reason_default,
        )
        candidate_count += 1
        family_counts[record.relation_family] += 1
        section_counts[record.section] += 1
        reason_counts[record.filtering_reason] += 1

    return RelationCandidateAuditReport(
        candidate_count=candidate_count,
        by_relation_family=family_counts,
        by_section=section_counts,
        by_filtering_reason=reason_counts,
    )


build_relation_candidate_audit = audit_relation_candidates
build_relation_candidate_audit_report = audit_relation_candidates
relation_candidate_audit_report = audit_relation_candidates
summarize_relation_candidates = audit_relation_candidates


__all__ = [
    "ACCEPTED_FILTERING_REASON",
    "CandidateAuditRecord",
    "OTHER_FILTERING_REASON",
    "RELATION_AUDIT_ARTIFACT",
    "RELATION_AUDIT_SCHEMA_VERSION",
    "RELATION_CANDIDATE_AUDIT",
    "RELATION_CANDIDATE_AUDIT_ARTIFACT",
    "RELATION_CANDIDATE_AUDIT_SCHEMA_VERSION",
    "RelationCandidateAuditEntry",
    "RelationCandidateAuditRecord",
    "RelationCandidateAuditReport",
    "UNSECTIONED_SECTION",
    "UNKNOWN_RELATION_FAMILY",
    "audit_relation_candidates",
    "build_relation_candidate_audit",
    "build_relation_candidate_audit_report",
    "relation_candidate_audit_report",
    "summarize_relation_candidates",
]
