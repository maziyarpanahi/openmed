"""Deterministic, metadata-only ordering for guarded-output citations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Final, Iterable

CITATION_ORDERING_SCHEMA_VERSION: Final[int] = 1

_METADATA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


class CitationOrderingError(ValueError):
    """Raised when citation metadata cannot be ordered safely."""


@dataclass(frozen=True)
class Citation:
    """Value-free evidence coordinates for one guarded clinical claim."""

    document_id: str
    section: str
    source_start: int
    source_end: int
    evidence_id: str
    primary: bool = False

    def __post_init__(self) -> None:
        _validate_metadata_id(self.document_id)
        _validate_metadata_id(self.section)
        _validate_metadata_id(self.evidence_id)
        if (
            type(self.source_start) is not int
            or type(self.source_end) is not int
            or self.source_start < 0
            or self.source_end <= self.source_start
        ):
            raise CitationOrderingError("invalid citation source offset")
        if type(self.primary) is not bool:
            raise CitationOrderingError("invalid citation primary marker")

    @property
    def source_offset(self) -> tuple[int, int]:
        """Return the half-open ``[start, end)`` source offset."""

        return self.source_start, self.source_end

    @property
    def is_primary(self) -> bool:
        """Return whether this citation is explicitly primary evidence."""

        return self.primary

    def to_dict(self) -> dict[str, object]:
        """Return the closed metadata-only citation representation."""

        return {
            "document_id": self.document_id,
            "section": self.section,
            "source_offset": {
                "start": self.source_start,
                "end": self.source_end,
            },
            "evidence_id": self.evidence_id,
            "primary": self.primary,
        }


def order_citations(citations: Iterable[Citation]) -> tuple[Citation, ...]:
    """Validate and return citations in their canonical metadata order.

    The input represents citations for one guarded claim, so at most one row
    may carry the primary-evidence marker. Exact duplicate coordinates may be
    retained, but they cannot disagree about whether that evidence is primary.
    """

    if isinstance(citations, (str, bytes, bytearray)):
        raise CitationOrderingError("invalid citation collection")
    try:
        records = tuple(citations)
    except Exception:
        raise CitationOrderingError("invalid citation collection") from None
    if any(not isinstance(citation, Citation) for citation in records):
        raise CitationOrderingError("invalid citation collection")

    markers_by_key: dict[tuple[str, str, int, int, str], bool] = {}
    primary_keys: set[tuple[str, str, int, int, str]] = set()
    for citation in records:
        key = _citation_key(citation)
        existing_marker = markers_by_key.get(key)
        if existing_marker is not None and existing_marker is not citation.primary:
            raise CitationOrderingError("conflicting citation primary markers")
        markers_by_key[key] = citation.primary
        if citation.primary:
            primary_keys.add(key)

    if len(primary_keys) > 1:
        raise CitationOrderingError("conflicting citation primary markers")
    return tuple(sorted(records, key=_citation_key))


@dataclass(frozen=True)
class CitationOrdering:
    """Versioned deterministic artifact for one claim's ordered citations."""

    citations: tuple[Citation, ...]
    schema_version: int = CITATION_ORDERING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != CITATION_ORDERING_SCHEMA_VERSION
        ):
            raise CitationOrderingError("unsupported citation ordering schema")
        object.__setattr__(self, "citations", order_citations(self.citations))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready artifact with no source or evidence text."""

        return {
            "schema_version": self.schema_version,
            "citations": [citation.to_dict() for citation in self.citations],
        }

    def to_json(self) -> str:
        """Return byte-stable JSON for audit and regression artifacts."""

        return (
            json.dumps(
                self.to_dict(),
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def _citation_key(citation: Citation) -> tuple[str, str, int, int, str]:
    return (
        citation.document_id,
        citation.section,
        citation.source_start,
        citation.source_end,
        citation.evidence_id,
    )


def _validate_metadata_id(value: object) -> None:
    if type(value) is not str or _METADATA_ID_RE.fullmatch(value) is None:
        raise CitationOrderingError("invalid citation metadata identifier")


__all__ = [
    "CITATION_ORDERING_SCHEMA_VERSION",
    "Citation",
    "CitationOrdering",
    "CitationOrderingError",
    "order_citations",
]
