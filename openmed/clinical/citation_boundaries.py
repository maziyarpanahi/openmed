"""Deterministic, value-free validation for post-de-identification citations.

Citation offsets are meaningful only in the exact de-identified document that
was used to produce them.  This module binds citations to that document's
digest, resolves the source version against an available offset map, and
rejects spans whose endpoints would be ambiguous after a replacement.

The module deliberately stores lengths, offsets, digests, and fixed reason
codes only.  It does not retain source text, replacement text, mappings for
re-identification, or model/runtime state.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Final, NoReturn

CITATION_BOUNDARY_SCHEMA_VERSION: Final[int] = 1
CITATION_BOUNDARY_DISCLAIMER: Final[str] = (
    "Citation boundary validation is value-free assistive metadata for human "
    "review; it is not a clinical decision or compliance certification."
)

INVALID_CITATION_COLLECTION: Final[str] = "invalid_citation_collection"
INVALID_CITATION: Final[str] = "invalid_citation"
INVALID_CITATION_OFFSET: Final[str] = "invalid_citation_offset"
DOCUMENT_DIGEST_MISSING: Final[str] = "document_digest_missing"
DOCUMENT_DIGEST_MISMATCH: Final[str] = "document_digest_mismatch"
SOURCE_VERSION_UNAVAILABLE: Final[str] = "source_version_unavailable"
CITATION_CROSSES_REPLACEMENT: Final[str] = "citation_crosses_replacement_boundary"
INVALID_OFFSET_MAP: Final[str] = "invalid_offset_map"
OFFSET_MAP_CONTENT_MISMATCH: Final[str] = "offset_map_content_mismatch"

# Descriptive aliases keep the reason vocabulary easy to discover for callers
# that use a ``REASON_*`` naming convention.
REASON_CITATION_CROSSES_REPLACEMENT_BOUNDARY: Final[str] = CITATION_CROSSES_REPLACEMENT
REASON_DOCUMENT_DIGEST_MISMATCH: Final[str] = DOCUMENT_DIGEST_MISMATCH
REASON_DOCUMENT_DIGEST_MISSING: Final[str] = DOCUMENT_DIGEST_MISSING
REASON_SOURCE_VERSION_UNAVAILABLE: Final[str] = SOURCE_VERSION_UNAVAILABLE

_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()
_ERROR_MESSAGES = {
    INVALID_CITATION_COLLECTION: "citation boundary validation failed: invalid collection",
    INVALID_CITATION: "citation boundary validation failed: invalid citation",
    INVALID_CITATION_OFFSET: "citation boundary validation failed: invalid offset",
    DOCUMENT_DIGEST_MISSING: "citation boundary validation failed: missing document digest",
    DOCUMENT_DIGEST_MISMATCH: "citation boundary validation failed: document digest mismatch",
    SOURCE_VERSION_UNAVAILABLE: "citation boundary validation failed: source version unavailable",
    CITATION_CROSSES_REPLACEMENT: "citation boundary validation failed: replacement boundary crossed",
    INVALID_OFFSET_MAP: "citation boundary validation failed: invalid offset map",
    OFFSET_MAP_CONTENT_MISMATCH: "citation boundary validation failed: offset map content mismatch",
}


class CitationBoundaryError(ValueError):
    """Raised when citation coordinates cannot be validated safely.

    The exception contains only a fixed reason code and a fixed message.  It
    never interpolates a citation, digest, source version, or source value.
    """

    reason_code: str

    def __init__(self, reason_code: str) -> None:
        safe_reason = (
            reason_code if reason_code in _ERROR_MESSAGES else INVALID_CITATION
        )
        self.reason_code = safe_reason
        super().__init__(_ERROR_MESSAGES[safe_reason])


def _raise(reason_code: str) -> NoReturn:
    raise CitationBoundaryError(reason_code)


def _sha256(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _normalise_digest(value: Any, *, allow_none: bool = True) -> str | None:
    if value is None:
        if allow_none:
            return None
        _raise(INVALID_OFFSET_MAP)
    if isinstance(value, bytes):
        return _sha256(value)
    if not isinstance(value, str):
        _raise(INVALID_CITATION)
    candidate = value.strip().lower()
    if _DIGEST_RE.fullmatch(candidate):
        return candidate
    if _HEX_DIGEST_RE.fullmatch(candidate):
        return f"sha256:{candidate}"
    # Hash accidental non-opaque input rather than retaining it in an
    # artifact.  Callers should still provide a high-entropy opaque digest.
    return _sha256(value.encode("utf-8"))


def _normalise_version(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return _sha256(value)
    if not isinstance(value, str):
        _raise(INVALID_CITATION)
    return _normalise_digest(value)


def _normalise_identifier(value: Any) -> str | None:
    if value is None:
        return None
    return _normalise_digest(value)


def _field(value: Any, names: tuple[str, ...], default: Any = _MISSING) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        try:
            return getattr(value, name)
        except (AttributeError, TypeError):
            continue
        except Exception:
            return default
    return default


def _offset_pair(value: Any) -> tuple[Any, Any] | None:
    if isinstance(value, Mapping):
        start = _field(value, ("start", "source_start", "post_start"))
        end = _field(value, ("end", "source_end", "post_end"))
        if start is _MISSING or end is _MISSING:
            return None
        return start, end
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return value[0], value[1]
    return None


def _valid_span(start: Any, end: Any, *, length: int | None = None) -> bool:
    if type(start) is not int or type(end) is not int:
        return False
    if not 0 <= start < end:
        return False
    return length is None or end <= length


def _valid_length(value: Any) -> bool:
    return type(value) is int and value >= 0


@dataclass(frozen=True)
class ReplacementBoundary:
    """One source-to-post-de-identification replacement boundary.

    All offsets are half-open Python offsets.  ``post_start == post_end`` is
    valid for a removal, which leaves no citeable post-de-identification text.
    """

    source_start: int
    source_end: int
    post_start: int
    post_end: int
    replacement_digest: str | None = None

    def __post_init__(self) -> None:
        if not _valid_span(self.source_start, self.source_end):
            _raise(INVALID_OFFSET_MAP)
        if (
            type(self.post_start) is not int
            or type(self.post_end) is not int
            or self.post_start < 0
            or self.post_end < self.post_start
        ):
            _raise(INVALID_OFFSET_MAP)
        object.__setattr__(
            self,
            "replacement_digest",
            _normalise_digest(self.replacement_digest),
        )

    @property
    def original_start(self) -> int:
        """Return the original-source start offset."""

        return self.source_start

    @property
    def original_end(self) -> int:
        """Return the original-source end offset."""

        return self.source_end

    @property
    def target_start(self) -> int:
        """Return the post-de-identification start offset."""

        return self.post_start

    @property
    def target_end(self) -> int:
        """Return the post-de-identification end offset."""

        return self.post_end

    @property
    def deidentified_start(self) -> int:
        """Return the post-de-identification start offset."""

        return self.post_start

    @property
    def deidentified_end(self) -> int:
        """Return the post-de-identification end offset."""

        return self.post_end

    def to_dict(self) -> dict[str, Any]:
        """Return offsets and optional replacement digest without source text."""

        payload: dict[str, Any] = {
            "source_offset": {
                "start": self.source_start,
                "end": self.source_end,
            },
            "post_deidentification_offset": {
                "start": self.post_start,
                "end": self.post_end,
            },
        }
        if self.replacement_digest is not None:
            payload["replacement_digest"] = self.replacement_digest
        return payload


OffsetMapEntry = ReplacementBoundary


@dataclass(frozen=True)
class DeidentificationOffsetMap:
    """Immutable offset map for one exact post-de-identification document."""

    source_length: int
    post_length: int
    replacements: tuple[ReplacementBoundary, ...]
    document_digest: str
    source_version: str | None = None
    schema_version: int = CITATION_BOUNDARY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not _valid_length(self.source_length)
            or not _valid_length(self.post_length)
            or type(self.schema_version) is not int
            or self.schema_version != CITATION_BOUNDARY_SCHEMA_VERSION
        ):
            _raise(INVALID_OFFSET_MAP)
        digest = _normalise_digest(self.document_digest, allow_none=False)
        if digest is None:
            _raise(INVALID_OFFSET_MAP)
        object.__setattr__(self, "document_digest", digest)
        object.__setattr__(
            self, "source_version", _normalise_version(self.source_version)
        )

        if isinstance(self.replacements, (str, bytes, bytearray)):
            _raise(INVALID_OFFSET_MAP)
        try:
            replacements = tuple(self.replacements)
        except Exception:
            _raise(INVALID_OFFSET_MAP)
        if any(not isinstance(item, ReplacementBoundary) for item in replacements):
            _raise(INVALID_OFFSET_MAP)
        replacements = tuple(
            sorted(
                replacements,
                key=lambda item: (
                    item.source_start,
                    item.source_end,
                    item.post_start,
                    item.post_end,
                ),
            )
        )

        source_cursor = 0
        post_cursor = 0
        for item in replacements:
            if item.source_start < source_cursor:
                _raise(INVALID_OFFSET_MAP)
            if item.post_start < post_cursor:
                _raise(INVALID_OFFSET_MAP)
            if item.source_end > self.source_length or item.post_end > self.post_length:
                _raise(INVALID_OFFSET_MAP)
            expected_post_start = post_cursor + (item.source_start - source_cursor)
            if item.post_start != expected_post_start:
                _raise(INVALID_OFFSET_MAP)
            source_cursor = item.source_end
            post_cursor = item.post_end
        if post_cursor + (self.source_length - source_cursor) != self.post_length:
            _raise(INVALID_OFFSET_MAP)
        object.__setattr__(self, "replacements", replacements)

    @classmethod
    def from_replacements(
        cls,
        *,
        source_length: int,
        post_length: int,
        replacements: Iterable[ReplacementBoundary],
        document_digest: str,
        source_version: str | None = None,
    ) -> "DeidentificationOffsetMap":
        """Construct and validate a map from explicit replacement boundaries."""

        return cls(
            source_length=source_length,
            post_length=post_length,
            replacements=tuple(replacements),
            document_digest=document_digest,
            source_version=source_version,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, value-free map representation."""

        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "source_length": self.source_length,
            "post_deidentification_length": self.post_length,
            "document_digest": self.document_digest,
            "replacement_boundaries": [item.to_dict() for item in self.replacements],
        }
        if self.source_version is not None:
            payload["source_version"] = self.source_version
        return payload

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

    def map_post_span(self, start: int, end: int) -> tuple[int, int] | None:
        """Map a post-de-identification span to source offsets.

        A whole replacement interval is map-safe.  A citation that combines
        only part of a replacement with neighbouring text is ambiguous and
        returns ``None``.  Removal boundaries are also treated as hard edges.
        """

        if not _valid_span(start, end, length=self.post_length):
            _raise(INVALID_CITATION_OFFSET)
        for item in self.replacements:
            if item.post_start == item.post_end:
                if start < item.post_start < end:
                    return None
                continue
            if start < item.post_end and end > item.post_start:
                if start == item.post_start and end == item.post_end:
                    return item.source_start, item.source_end
                return None
        source_start = self._map_post_point(start, right=True)
        source_end = self._map_post_point(end, right=False)
        if source_start is None or source_end is None or source_start >= source_end:
            return None
        return source_start, source_end

    def map_source_span(self, start: int, end: int) -> tuple[int, int] | None:
        """Map a source span to post-de-identification offsets when unambiguous."""

        if not _valid_span(start, end, length=self.source_length):
            _raise(INVALID_CITATION_OFFSET)
        for item in self.replacements:
            if start < item.source_start or end > item.source_end:
                continue
            if start == item.source_start and end == item.source_end:
                return item.post_start, item.post_end
        for item in self.replacements:
            if start < item.source_end and end > item.source_start:
                return None
        source_start = self._map_source_point(start, right=True)
        source_end = self._map_source_point(end, right=False)
        if source_start is None or source_end is None or source_start >= source_end:
            return None
        return source_start, source_end

    def _map_post_point(self, point: int, *, right: bool) -> int | None:
        shift = 0
        for item in self.replacements:
            if point > item.post_end:
                shift += (item.source_end - item.source_start) - (
                    item.post_end - item.post_start
                )
                continue
            if point == item.post_end and item.post_end != item.post_start:
                shift += (item.source_end - item.source_start) - (
                    item.post_end - item.post_start
                )
                continue
            if item.post_start == item.post_end == point:
                if right:
                    shift += item.source_end - item.source_start
                return point + shift
            if item.post_start < point < item.post_end:
                return None
            if point <= item.post_start:
                break
        return point + shift

    def _map_source_point(self, point: int, *, right: bool) -> int | None:
        shift = 0
        for item in self.replacements:
            if point > item.source_end:
                shift += (item.post_end - item.post_start) - (
                    item.source_end - item.source_start
                )
                continue
            if point == item.source_end and item.source_end != item.source_start:
                shift += (item.post_end - item.post_start) - (
                    item.source_end - item.source_start
                )
                continue
            if item.source_start < point < item.source_end:
                return None
            if point == item.source_start:
                if right:
                    return item.post_start + shift
                return item.post_start + shift
            if point <= item.source_start:
                break
        return point + shift


OffsetMap = DeidentificationOffsetMap


@dataclass(frozen=True, init=False)
class CitationBoundary:
    """A citation coordinate in post-de-identification document space."""

    post_start: int
    post_end: int
    document_digest: str | None
    source_version: str | None
    citation_id: str | None

    def __init__(
        self,
        post_start: Any = None,
        post_end: Any = None,
        document_digest: Any = None,
        *,
        start: Any = None,
        end: Any = None,
        source_start: Any = None,
        source_end: Any = None,
        post_offset: Any = None,
        source_offset: Any = None,
        document_id: Any = None,
        source_version: Any = None,
        version: Any = None,
        citation_id: Any = None,
        id: Any = None,
    ) -> None:
        post_pair = _offset_pair(post_offset) or _offset_pair(source_offset)
        starts = [
            value for value in (post_start, start, source_start) if value is not None
        ]
        ends = [value for value in (post_end, end, source_end) if value is not None]
        if post_pair is not None:
            starts.append(post_pair[0])
            ends.append(post_pair[1])
        if not starts or not ends or any(value != starts[0] for value in starts):
            _raise(INVALID_CITATION_OFFSET)
        if any(value != ends[0] for value in ends):
            _raise(INVALID_CITATION_OFFSET)
        if not _valid_span(starts[0], ends[0]):
            _raise(INVALID_CITATION_OFFSET)
        digest_value = document_digest if document_digest is not None else document_id
        version_value = source_version if source_version is not None else version
        citation_identifier = citation_id if citation_id is not None else id
        object.__setattr__(self, "post_start", starts[0])
        object.__setattr__(self, "post_end", ends[0])
        object.__setattr__(self, "document_digest", _normalise_digest(digest_value))
        object.__setattr__(self, "source_version", _normalise_version(version_value))
        object.__setattr__(
            self,
            "citation_id",
            _normalise_identifier(citation_identifier),
        )

    @property
    def source_start(self) -> int:
        """Return the post-de-identification start offset."""

        return self.post_start

    @property
    def source_end(self) -> int:
        """Return the post-de-identification end offset."""

        return self.post_end

    @property
    def start(self) -> int:
        """Return the post-de-identification start offset."""

        return self.post_start

    @property
    def end(self) -> int:
        """Return the post-de-identification end offset."""

        return self.post_end

    @property
    def post_offset(self) -> tuple[int, int]:
        """Return the post-de-identification half-open offset pair."""

        return self.post_start, self.post_end

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free citation representation."""

        payload: dict[str, Any] = {
            "post_deidentification_offset": {
                "start": self.post_start,
                "end": self.post_end,
            },
        }
        if self.document_digest is not None:
            payload["document_digest"] = self.document_digest
        if self.source_version is not None:
            payload["source_version"] = self.source_version
        if self.citation_id is not None:
            payload["citation_id"] = self.citation_id
        return payload


Citation = CitationBoundary


@dataclass(frozen=True)
class ValidatedCitation:
    """A citation with its unambiguous original-source projection."""

    citation: CitationBoundary
    source_start: int
    source_end: int

    def to_dict(self) -> dict[str, Any]:
        """Return safe post- and original-source offsets."""

        payload = self.citation.to_dict()
        payload["source_offset"] = {
            "start": self.source_start,
            "end": self.source_end,
        }
        return payload


@dataclass(frozen=True)
class CitationBoundaryIssue:
    """One value-free rejection record identified only by its offsets."""

    post_start: int | None
    post_end: int | None
    reason_code: str

    def __post_init__(self) -> None:
        if self.post_start is not None and type(self.post_start) is not int:
            _raise(INVALID_CITATION)
        if self.post_end is not None and type(self.post_end) is not int:
            _raise(INVALID_CITATION)
        if (
            self.post_start is not None
            and self.post_end is not None
            and not _valid_span(self.post_start, self.post_end)
        ):
            _raise(INVALID_CITATION)
        if self.reason_code not in _ERROR_MESSAGES:
            _raise(INVALID_CITATION)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"reason_code": self.reason_code}
        if self.post_start is not None and self.post_end is not None:
            payload["post_deidentification_offset"] = {
                "start": self.post_start,
                "end": self.post_end,
            }
        return payload


@dataclass(frozen=True)
class CitationBoundaryReport:
    """Deterministic, PHI-free result for a citation validation batch."""

    validated_citations: tuple[ValidatedCitation, ...]
    issues: tuple[CitationBoundaryIssue, ...]
    document_digests: tuple[str, ...]
    source_versions: tuple[str, ...]
    replacement_boundary_count: int
    schema_version: int = CITATION_BOUNDARY_SCHEMA_VERSION
    disclaimer: str = CITATION_BOUNDARY_DISCLAIMER

    @property
    def valid(self) -> bool:
        """Return whether every submitted citation was accepted."""

        return not self.issues

    @property
    def is_valid(self) -> bool:
        """Alias for :attr:`valid`."""

        return self.valid

    @property
    def accepted_count(self) -> int:
        """Return the number of accepted citations."""

        return len(self.validated_citations)

    @property
    def rejected_count(self) -> int:
        """Return the number of rejected citations."""

        return len(self.issues)

    @property
    def reason_counts(self) -> dict[str, int]:
        """Return deterministic counts for fixed rejection reason codes."""

        counts: dict[str, int] = {}
        for issue in self.issues:
            counts[issue.reason_code] = counts.get(issue.reason_code, 0) + 1
        return dict(sorted(counts.items()))

    @property
    def accepted(self) -> tuple[ValidatedCitation, ...]:
        """Return accepted citations in canonical order."""

        return self.validated_citations

    @property
    def rejected(self) -> tuple[CitationBoundaryIssue, ...]:
        """Return value-free rejection records."""

        return self.issues

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready report without source or replacement values."""

        return {
            "schema_version": self.schema_version,
            "valid": self.valid,
            "checked_count": self.accepted_count + self.rejected_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "document_digests": list(self.document_digests),
            "source_versions": list(self.source_versions),
            "replacement_boundary_count": self.replacement_boundary_count,
            "reason_counts": self.reason_counts,
            "accepted_citations": [
                citation.to_dict() for citation in self.validated_citations
            ],
            "rejected_citations": [issue.to_dict() for issue in self.issues],
            "requires_human_review": True,
            "disclaimer": self.disclaimer,
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


CitationBoundaryValidation = CitationBoundaryReport
ValidationReport = CitationBoundaryReport
CitationBoundaryValidationError = CitationBoundaryError
PostDeidentificationOffsetMap = DeidentificationOffsetMap
ReplacementSpan = ReplacementBoundary


def _coerce_replacement_spec(
    value: Any,
) -> tuple[int, int, str | None, int | None, int | None]:
    if isinstance(value, ReplacementBoundary):
        return (
            value.source_start,
            value.source_end,
            None,
            value.post_start,
            value.post_end,
        )
    source_start = _field(value, ("source_start", "original_start", "start"))
    source_end = _field(value, ("source_end", "original_end", "end"))
    source_pair = _field(value, ("source_offset", "original_offset"))
    if source_pair is not _MISSING:
        pair = _offset_pair(source_pair)
        if pair is not None:
            source_start, source_end = pair
    post_pair = _field(
        value,
        (
            "post_deidentification_offset",
            "post_offset",
            "target_offset",
            "deidentified_offset",
        ),
    )
    explicit_post_start = explicit_post_end = None
    if post_pair is not _MISSING:
        pair = _offset_pair(post_pair)
        if pair is not None:
            explicit_post_start, explicit_post_end = pair
    replacement = _field(
        value,
        ("replacement", "redacted_text", "post_text", "surrogate"),
    )
    if replacement is _MISSING:
        replacement = None
    elif replacement is not None and not isinstance(replacement, str):
        _raise(INVALID_OFFSET_MAP)
    if (
        source_start is _MISSING
        or source_end is _MISSING
        or not _valid_span(source_start, source_end)
    ):
        _raise(INVALID_OFFSET_MAP)
    if explicit_post_start is not None and (
        type(explicit_post_start) is not int or explicit_post_start < 0
    ):
        _raise(INVALID_OFFSET_MAP)
    if explicit_post_end is not None and (
        type(explicit_post_end) is not int
        or explicit_post_end < (explicit_post_start or 0)
    ):
        _raise(INVALID_OFFSET_MAP)
    return (
        source_start,
        source_end,
        replacement,
        explicit_post_start,
        explicit_post_end,
    )


def _result_field(result: Any, names: tuple[str, ...], default: Any = _MISSING) -> Any:
    return _field(result, names, default)


def _build_map_from_texts(
    source_text: str,
    post_text: str,
    raw_replacements: Iterable[Any],
    *,
    document_digest: Any = None,
    source_version: Any = None,
) -> DeidentificationOffsetMap:
    if not isinstance(source_text, str) or not isinstance(post_text, str):
        _raise(INVALID_OFFSET_MAP)
    if isinstance(raw_replacements, (str, bytes, bytearray)):
        _raise(INVALID_OFFSET_MAP)
    try:
        specs = tuple(_coerce_replacement_spec(item) for item in raw_replacements)
    except CitationBoundaryError:
        raise
    except Exception:
        _raise(INVALID_OFFSET_MAP)
    specs = tuple(sorted(specs, key=lambda item: (item[0], item[1])))
    boundaries: list[ReplacementBoundary] = []
    rendered: list[str] = []
    source_cursor = 0
    post_cursor = 0
    for (
        source_start,
        source_end,
        replacement,
        explicit_post_start,
        explicit_post_end,
    ) in specs:
        if not _valid_span(source_start, source_end, length=len(source_text)):
            _raise(INVALID_OFFSET_MAP)
        if source_start < source_cursor:
            _raise(INVALID_OFFSET_MAP)
        rendered.append(source_text[source_cursor:source_start])
        post_start = post_cursor + (source_start - source_cursor)
        if explicit_post_start is not None and explicit_post_start != post_start:
            _raise(OFFSET_MAP_CONTENT_MISMATCH)
        if replacement is None:
            if explicit_post_end is None:
                _raise(INVALID_OFFSET_MAP)
            post_end = explicit_post_end
            replacement_for_render = post_text[post_start:post_end]
        else:
            post_end = post_start + len(replacement)
            replacement_for_render = replacement
            if explicit_post_end is not None and explicit_post_end != post_end:
                _raise(OFFSET_MAP_CONTENT_MISMATCH)
        if post_end < post_start or post_end > len(post_text):
            _raise(INVALID_OFFSET_MAP)
        rendered.append(replacement_for_render)
        boundaries.append(
            ReplacementBoundary(
                source_start=source_start,
                source_end=source_end,
                post_start=post_start,
                post_end=post_end,
                replacement_digest=(
                    _sha256(replacement.encode("utf-8"))
                    if replacement is not None
                    else None
                ),
            )
        )
        source_cursor = source_end
        post_cursor = post_end
    rendered.append(source_text[source_cursor:])
    if "".join(rendered) != post_text:
        _raise(OFFSET_MAP_CONTENT_MISMATCH)
    actual_digest = _sha256(post_text.encode("utf-8"))
    supplied_digest = _normalise_digest(document_digest)
    if supplied_digest is not None and supplied_digest != actual_digest:
        _raise(DOCUMENT_DIGEST_MISMATCH)
    return DeidentificationOffsetMap(
        source_length=len(source_text),
        post_length=len(post_text),
        replacements=tuple(boundaries),
        document_digest=actual_digest,
        source_version=_normalise_version(source_version),
    )


def build_deidentification_offset_map(
    source_or_result: Any,
    post_deidentified_text: str | None = None,
    replacements: Iterable[Any] | None = None,
    *,
    deidentified_text: str | None = None,
    offset_entries: Iterable[Any] | None = None,
    document_digest: Any = None,
    source_version: Any = None,
) -> DeidentificationOffsetMap:
    """Build a validated map from text/replacements or a de-identification result.

    ``source_or_result`` may be a pair's original source string, in which case
    ``post_deidentified_text`` and ``replacements`` are required, or a
    ``DeidentificationResult``-like object exposing ``original_text``,
    ``deidentified_text``, and ``pii_entities``.  Replacement records use
    original ``start``/``end`` offsets and ``redacted_text``.  The returned map
    stores only lengths, offsets, digests, and an optional opaque version.
    """

    if (
        post_deidentified_text is not None
        and deidentified_text is not None
        and post_deidentified_text != deidentified_text
    ):
        _raise(INVALID_OFFSET_MAP)
    if post_deidentified_text is None:
        post_deidentified_text = deidentified_text
    if replacements is not None and offset_entries is not None:
        _raise(INVALID_OFFSET_MAP)
    if replacements is None:
        replacements = offset_entries

    if isinstance(source_or_result, str):
        if post_deidentified_text is None or replacements is None:
            _raise(INVALID_OFFSET_MAP)
        return _build_map_from_texts(
            source_or_result,
            post_deidentified_text,
            replacements,
            document_digest=document_digest,
            source_version=source_version,
        )

    source_text = _result_field(source_or_result, ("original_text", "source_text"))
    post_text = _result_field(
        source_or_result,
        ("deidentified_text", "post_deidentified_text", "post_text"),
    )
    if source_text is _MISSING or post_text is _MISSING:
        _raise(INVALID_OFFSET_MAP)
    if replacements is None:
        replacements = _result_field(
            source_or_result,
            ("pii_entities", "entities", "replacements"),
        )
    if replacements is _MISSING or replacements is None:
        replacements = ()
    if document_digest is None:
        document_digest = _result_field(
            source_or_result,
            ("document_digest", "deidentified_text_hash"),
            None,
        )
    if source_version is None:
        source_version = _result_field(
            source_or_result,
            ("source_version", "version"),
            None,
        )
    return _build_map_from_texts(
        source_text,
        post_text,
        replacements,
        document_digest=document_digest,
        source_version=source_version,
    )


build_offset_map = build_deidentification_offset_map
offset_map_from_result = build_deidentification_offset_map


def _coerce_citation(value: Any) -> CitationBoundary:
    if isinstance(value, CitationBoundary):
        return value
    if isinstance(value, (str, bytes, bytearray)):
        _raise(INVALID_CITATION)
    try:
        post_pair = _field(
            value,
            (
                "post_deidentification_offset",
                "post_offset",
                "source_offset",
                "source_offsets",
                "offset",
            ),
        )
        document_digest = _field(
            value,
            (
                "document_digest",
                "post_document_digest",
                "document_hash",
                "digest",
                "document_id",
            ),
            None,
        )
        source_version = _field(
            value,
            ("source_version", "version", "document_version"),
            None,
        )
        citation_id = _field(
            value,
            ("citation_id", "id", "evidence_id"),
            None,
        )
        if post_pair is not _MISSING:
            return CitationBoundary(
                post_offset=post_pair,
                document_digest=document_digest,
                source_version=source_version,
                citation_id=citation_id,
            )
        return CitationBoundary(
            post_start=_field(value, ("post_start", "start", "source_start")),
            post_end=_field(value, ("post_end", "end", "source_end")),
            document_digest=document_digest,
            source_version=source_version,
            citation_id=citation_id,
        )
    except CitationBoundaryError:
        raise
    except Exception:
        _raise(INVALID_CITATION)


def _coerce_maps(value: Any) -> tuple[DeidentificationOffsetMap, ...]:
    if isinstance(value, DeidentificationOffsetMap):
        return (value,)
    result_source = _field(value, ("original_text", "source_text"))
    result_post = _field(
        value,
        ("deidentified_text", "post_deidentified_text", "post_text"),
    )
    if result_source is not _MISSING and result_post is not _MISSING:
        return (build_deidentification_offset_map(value),)
    if isinstance(value, Mapping):
        # A single serialized map is accepted as a convenience.  A mapping of
        # version -> map is used for multi-version validation.
        if "source_length" in value and "post_deidentification_length" in value:
            try:
                raw_boundaries = value.get("replacement_boundaries", ())
                boundaries = tuple(
                    ReplacementBoundary(
                        source_start=item["source_offset"]["start"],
                        source_end=item["source_offset"]["end"],
                        post_start=item["post_deidentification_offset"]["start"],
                        post_end=item["post_deidentification_offset"]["end"],
                        replacement_digest=item.get("replacement_digest"),
                    )
                    for item in raw_boundaries
                )
                return (
                    DeidentificationOffsetMap(
                        source_length=value["source_length"],
                        post_length=value["post_deidentification_length"],
                        replacements=boundaries,
                        document_digest=value["document_digest"],
                        source_version=value.get("source_version"),
                        schema_version=value.get("schema_version", 1),
                    ),
                )
            except CitationBoundaryError:
                raise
            except Exception:
                _raise(INVALID_OFFSET_MAP)
        maps: list[DeidentificationOffsetMap] = []
        for version, raw_map in value.items():
            if not isinstance(raw_map, DeidentificationOffsetMap):
                _raise(INVALID_OFFSET_MAP)
            normalized_version = _normalise_version(version)
            if raw_map.source_version is None:
                raw_map = replace(raw_map, source_version=normalized_version)
            elif raw_map.source_version != normalized_version:
                _raise(INVALID_OFFSET_MAP)
            maps.append(raw_map)
        return tuple(sorted(maps, key=lambda item: item.source_version or ""))
    if isinstance(value, (str, bytes, bytearray)):
        _raise(INVALID_OFFSET_MAP)
    try:
        raw_maps = tuple(value)
    except Exception:
        _raise(INVALID_OFFSET_MAP)
    if any(not isinstance(item, DeidentificationOffsetMap) for item in raw_maps):
        _raise(INVALID_OFFSET_MAP)
    return tuple(sorted(raw_maps, key=lambda item: item.source_version or ""))


def _available_versions(value: Any) -> set[str] | None:
    if value is None:
        return None
    values: Iterable[Any]
    if isinstance(value, Mapping):
        values = value.keys()
    elif isinstance(value, (str, bytes, bytearray)):
        values = (value,)
    else:
        try:
            values = tuple(value)
        except Exception:
            _raise(INVALID_OFFSET_MAP)
    return {
        normalized
        for item in values
        if (normalized := _normalise_version(item)) is not None
    }


def _select_map(
    maps: tuple[DeidentificationOffsetMap, ...],
    citation: CitationBoundary,
    available_versions: set[str] | None,
) -> DeidentificationOffsetMap:
    if not maps:
        _raise(SOURCE_VERSION_UNAVAILABLE)
    requested_version = citation.source_version
    if requested_version is not None:
        if (
            available_versions is not None
            and requested_version not in available_versions
        ):
            _raise(SOURCE_VERSION_UNAVAILABLE)
        for offset_map in maps:
            if offset_map.source_version == requested_version:
                return offset_map
        _raise(SOURCE_VERSION_UNAVAILABLE)
    if len(maps) == 1:
        offset_map = maps[0]
        if (
            available_versions is not None
            and offset_map.source_version is not None
            and offset_map.source_version not in available_versions
        ):
            _raise(SOURCE_VERSION_UNAVAILABLE)
        return offset_map
    _raise(SOURCE_VERSION_UNAVAILABLE)


def _citation_sort_key(citation: CitationBoundary) -> tuple[str, str, int, int, str]:
    return (
        citation.document_digest or "",
        citation.source_version or "",
        citation.post_start,
        citation.post_end,
        citation.citation_id or "",
    )


def _issue_sort_key(issue: CitationBoundaryIssue) -> tuple[int, int, str]:
    return (
        issue.post_start if issue.post_start is not None else -1,
        issue.post_end if issue.post_end is not None else -1,
        issue.reason_code,
    )


def validate_citation_boundaries(
    citations: Iterable[Any],
    offset_map: Any,
    *,
    document_digest: Any = None,
    available_source_versions: Any = None,
    raise_on_error: bool = True,
) -> CitationBoundaryReport:
    """Validate citations against post-de-identification offset maps.

    Citation offsets are interpreted in post-de-identification document space.
    A citation must carry the map's document digest, resolve to an available
    source version, and either remain in an unchanged region or cover one full
    replacement interval.  A citation that crosses a replacement boundary is
    rejected because its original-source projection is ambiguous.

    Args:
        citations: Citation mappings/objects or :class:`CitationBoundary`
            values.  ``start``/``end`` or ``source_offset`` are accepted as
            post-de-identification offsets for compatibility with existing
            clinical citation records.
        offset_map: One :class:`DeidentificationOffsetMap`, a serialized map,
            or a mapping/iterable of versioned maps.
        document_digest: Optional expected digest for every selected map.
        available_source_versions: Optional allow-list or mapping of opaque
            source versions.  When omitted, the supplied map set is the source
            version registry.
        raise_on_error: Raise :class:`CitationBoundaryError` on rejection by
            default.  Set to ``False`` to receive a PHI-free rejection report.

    Returns:
        A deterministic :class:`CitationBoundaryReport` containing only safe
        offsets, digests, counts, and fixed reason codes.
    """

    if isinstance(citations, (str, bytes, bytearray)):
        _raise(INVALID_CITATION_COLLECTION)
    if type(raise_on_error) is not bool:
        _raise(INVALID_CITATION)
    maps = _coerce_maps(offset_map)
    expected_digest = _normalise_digest(document_digest)
    available_versions = _available_versions(available_source_versions)
    if isinstance(citations, (CitationBoundary, Mapping)):
        raw_citations = (citations,)
    else:
        try:
            raw_citations = tuple(citations)
        except Exception:
            _raise(INVALID_CITATION_COLLECTION)

    safe_citations: list[CitationBoundary] = []
    coercion_issues: list[CitationBoundaryIssue] = []
    for raw in raw_citations:
        try:
            safe_citations.append(_coerce_citation(raw))
        except CitationBoundaryError as error:
            coercion_issues.append(CitationBoundaryIssue(None, None, error.reason_code))

    safe_citations.sort(key=_citation_sort_key)
    accepted: list[ValidatedCitation] = []
    issues = list(coercion_issues)
    for citation in safe_citations:
        try:
            selected_map = _select_map(maps, citation, available_versions)
            if citation.document_digest is None:
                _raise(DOCUMENT_DIGEST_MISSING)
            if (
                expected_digest is not None
                and selected_map.document_digest != expected_digest
            ):
                _raise(DOCUMENT_DIGEST_MISMATCH)
            if citation.document_digest != selected_map.document_digest:
                _raise(DOCUMENT_DIGEST_MISMATCH)
            source_span = selected_map.map_post_span(
                citation.post_start,
                citation.post_end,
            )
            if source_span is None:
                _raise(CITATION_CROSSES_REPLACEMENT)
            accepted.append(
                ValidatedCitation(
                    citation=citation,
                    source_start=source_span[0],
                    source_end=source_span[1],
                )
            )
        except CitationBoundaryError as error:
            issues.append(
                CitationBoundaryIssue(
                    citation.post_start,
                    citation.post_end,
                    error.reason_code,
                )
            )

    accepted.sort(key=lambda item: _citation_sort_key(item.citation))
    issues.sort(key=_issue_sort_key)
    report = CitationBoundaryReport(
        validated_citations=tuple(accepted),
        issues=tuple(issues),
        document_digests=tuple(sorted({item.document_digest for item in maps})),
        source_versions=tuple(
            sorted(
                {
                    item.source_version
                    for item in maps
                    if item.source_version is not None
                }
            )
        ),
        replacement_boundary_count=sum(len(item.replacements) for item in maps),
    )
    if report.issues and raise_on_error:
        _raise(report.issues[0].reason_code)
    return report


validate_citations = validate_citation_boundaries
check_citation_boundaries = validate_citation_boundaries
validate_deidentified_citations = validate_citation_boundaries


__all__ = [
    "CITATION_BOUNDARY_DISCLAIMER",
    "CITATION_BOUNDARY_SCHEMA_VERSION",
    "CITATION_CROSSES_REPLACEMENT",
    "Citation",
    "CitationBoundary",
    "CitationBoundaryError",
    "CitationBoundaryIssue",
    "CitationBoundaryReport",
    "CitationBoundaryValidation",
    "CitationBoundaryValidationError",
    "DeidentificationOffsetMap",
    "DOCUMENT_DIGEST_MISMATCH",
    "DOCUMENT_DIGEST_MISSING",
    "INVALID_CITATION",
    "INVALID_CITATION_COLLECTION",
    "INVALID_CITATION_OFFSET",
    "INVALID_OFFSET_MAP",
    "OFFSET_MAP_CONTENT_MISMATCH",
    "OffsetMap",
    "OffsetMapEntry",
    "PostDeidentificationOffsetMap",
    "ReplacementBoundary",
    "ReplacementSpan",
    "REASON_CITATION_CROSSES_REPLACEMENT_BOUNDARY",
    "REASON_DOCUMENT_DIGEST_MISMATCH",
    "REASON_DOCUMENT_DIGEST_MISSING",
    "REASON_SOURCE_VERSION_UNAVAILABLE",
    "SOURCE_VERSION_UNAVAILABLE",
    "ValidationReport",
    "ValidatedCitation",
    "build_deidentification_offset_map",
    "build_offset_map",
    "check_citation_boundaries",
    "offset_map_from_result",
    "validate_citation_boundaries",
    "validate_citations",
    "validate_deidentified_citations",
]
