"""Deterministic, value-free minimality checks for guarded claim citations.

The caller supplies the source span that an atomic claim needs and the span
that a citation actually covers.  This module compares those coordinates and
counts local lexical tokens while the source text is in memory.  The returned
records retain only opaque identifiers, offsets, token counts, and a controlled
review status; source text is never stored, serialized, or echoed by a
validation error.

Minimality is a review signal, not a semantic entailment test.  A human or an
upstream evidence annotator must identify the required span for each atomic
claim before this check can measure citation breadth.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

__all__ = [
    "CITATION_MINIMALITY_DISCLAIMER",
    "CITATION_MINIMALITY_SCHEMA_VERSION",
    "AtomicClaim",
    "ClaimCitation",
    "CitationMinimalityError",
    "CitationMinimalityRecord",
    "CitationMinimalityReport",
    "CitationMinimalityStatus",
    "CitationSpan",
    "build_citation_minimality_report",
    "check_citation_minimality",
    "export_citation_minimality",
]


CITATION_MINIMALITY_SCHEMA_VERSION: Final[int] = 1
CITATION_MINIMALITY_DISCLAIMER: Final[str] = (
    "Citation minimality is a deterministic, value-free review aid. It does "
    "not establish clinical truth, entailment, or compliance certification."
)

_OPAQUE_REFERENCE_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_MISSING = object()


class CitationMinimalityError(ValueError):
    """Raised when citation minimality inputs cannot be evaluated safely."""


class CitationMinimalityStatus(str, Enum):
    """Controlled outcomes for one citation-to-claim comparison."""

    MINIMAL = "minimal"
    EXCESS_CONTEXT = "excess_context"
    MISSING_REQUIRED_SPAN = "missing_required_span"


@dataclass(frozen=True, slots=True)
class CitationSpan:
    """A non-empty half-open source span represented only by offsets."""

    start: int
    end: int

    def __post_init__(self) -> None:
        start, end = _source_offsets(self.start, self.end)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)

    @property
    def source_offset(self) -> tuple[int, int]:
        """Return the inclusive-start, exclusive-end offset pair."""

        return self.start, self.end

    def to_dict(self) -> dict[str, int]:
        """Return the JSON-safe offset representation."""

        return {"start": self.start, "end": self.end}

    @classmethod
    def from_obj(cls, value: Any) -> "CitationSpan":
        """Build a span from a two-item sequence or span-like object.

        Mappings may use ``start``/``end`` or ``source_start``/
        ``source_end``.  Objects exposing either pair of attributes are also
        accepted so existing local span objects can be checked without
        copying their source values.
        """

        if isinstance(value, cls):
            return value

        start = _read(value, ("start", "source_start"))
        end = _read(value, ("end", "source_end"))
        if start is _MISSING or end is _MISSING:
            nested = _read(value, ("source_offset", "source_offsets", "offset"))
            if nested is not _MISSING and nested is not value:
                return cls.from_obj(nested)
            if isinstance(value, Mapping):
                raise CitationMinimalityError(
                    "citation spans require start and end offsets"
                )
            if isinstance(value, (str, bytes, bytearray)):
                raise CitationMinimalityError(
                    "citation spans require start and end offsets"
                )
            try:
                values = tuple(value)
            except Exception:
                raise CitationMinimalityError(
                    "citation spans require start and end offsets"
                ) from None
            if len(values) != 2:
                raise CitationMinimalityError(
                    "citation spans require start and end offsets"
                )
            start, end = values

        return cls(start=start, end=end)


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    """One guarded atomic claim and its caller-declared minimal source span.

    ``claim_id`` must be an opaque SHA-256 reference so review artifacts do
    not reproduce a claim label or other sensitive identifier.
    """

    claim_id: str
    required_span: CitationSpan

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _opaque_reference(self.claim_id))
        object.__setattr__(
            self,
            "required_span",
            CitationSpan.from_obj(self.required_span),
        )

    @property
    def minimal_span(self) -> CitationSpan:
        """Return the required minimal evidence span."""

        return self.required_span

    @property
    def minimal_offset(self) -> tuple[int, int]:
        """Return the required minimal evidence offset."""

        return self.required_span.source_offset

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free claim representation."""

        return {
            "claim_id": self.claim_id,
            "required_offset": self.required_span.to_dict(),
        }

    @classmethod
    def from_obj(cls, value: Any) -> "AtomicClaim":
        """Build a claim from a typed record or value-free mapping."""

        if isinstance(value, cls):
            return value
        claim_id = _read(value, ("claim_id", "id"))
        if claim_id is _MISSING:
            raise CitationMinimalityError("atomic claims require an opaque claim id")
        required = _span_value(
            value,
            (
                "required_span",
                "minimal_span",
                "required_source_span",
                "minimal_source_span",
                "required_offset",
                "minimal_offset",
                "required_source_offset",
                "minimal_source_offset",
                "source_span",
                "source_offset",
                "source_offsets",
            ),
        )
        if required is _MISSING:
            required = _span_from_fields(
                value,
                (
                    "required_start",
                    "minimal_start",
                    "required_source_start",
                    "minimal_source_start",
                    "source_start",
                    "start",
                ),
                (
                    "required_end",
                    "minimal_end",
                    "required_source_end",
                    "minimal_source_end",
                    "source_end",
                    "end",
                ),
            )
            if required is _MISSING:
                required = value
        return cls(claim_id=claim_id, required_span=CitationSpan.from_obj(required))


@dataclass(frozen=True, slots=True)
class ClaimCitation:
    """One citation span associated with an atomic claim.

    ``citation_id`` is optional when constructing a record.  In that case a
    deterministic opaque reference is derived from the claim reference and
    offsets only; no source value is involved.
    """

    claim_id: str
    source_span: CitationSpan
    citation_id: str | None = None

    def __post_init__(self) -> None:
        normalized_claim_id = _opaque_reference(self.claim_id)
        span = CitationSpan.from_obj(self.source_span)
        citation_id = self.citation_id
        if citation_id is None:
            citation_id = _derived_citation_reference(normalized_claim_id, span)
        else:
            citation_id = _opaque_reference(citation_id)
        object.__setattr__(self, "claim_id", normalized_claim_id)
        object.__setattr__(self, "source_span", span)
        object.__setattr__(self, "citation_id", citation_id)

    @property
    def citation_span(self) -> CitationSpan:
        """Return the source span covered by this citation."""

        return self.source_span

    @property
    def source_offset(self) -> tuple[int, int]:
        """Return the citation's half-open source offset."""

        return self.source_span.source_offset

    def to_dict(self) -> dict[str, Any]:
        """Return a value-free citation representation."""

        return {
            "claim_id": self.claim_id,
            "citation_id": self.citation_id,
            "citation_offset": self.source_span.to_dict(),
        }

    @classmethod
    def from_obj(cls, value: Any) -> "ClaimCitation":
        """Build a citation from a typed record or value-free mapping."""

        if isinstance(value, cls):
            return value
        claim_id = _read(value, ("claim_id", "id"))
        if claim_id is _MISSING:
            raise CitationMinimalityError("citations require an opaque claim id")
        citation_id = _read(value, ("citation_id", "evidence_id"))
        if citation_id is _MISSING:
            citation_id = None
        span = _span_value(
            value,
            (
                "source_span",
                "citation_span",
                "citation_source_span",
                "citation_offset",
                "citation_source_offset",
                "source_offset",
                "source_offsets",
                "offset",
            ),
        )
        if span is _MISSING:
            span = _span_from_fields(
                value,
                (
                    "citation_start",
                    "source_start",
                    "start",
                ),
                (
                    "citation_end",
                    "source_end",
                    "end",
                ),
            )
            if span is _MISSING:
                span = value
        return cls(
            claim_id=claim_id,
            source_span=CitationSpan.from_obj(span),
            citation_id=citation_id,
        )


@dataclass(frozen=True, slots=True)
class CitationMinimalityRecord:
    """Value-free result for one citation and one atomic claim."""

    claim_id: str
    citation_id: str
    citation_span: CitationSpan
    required_span: CitationSpan
    citation_token_count: int
    required_token_count: int
    excess_token_count: int
    excess_context_spans: tuple[CitationSpan, ...]
    status: CitationMinimalityStatus
    review_required: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "claim_id", _opaque_reference(self.claim_id))
        object.__setattr__(self, "citation_id", _opaque_reference(self.citation_id))
        object.__setattr__(
            self,
            "citation_span",
            CitationSpan.from_obj(self.citation_span),
        )
        object.__setattr__(
            self,
            "required_span",
            CitationSpan.from_obj(self.required_span),
        )
        for field_name in (
            "citation_token_count",
            "required_token_count",
            "excess_token_count",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise CitationMinimalityError(
                    "token counts must be non-negative integers"
                )
        try:
            status = CitationMinimalityStatus(self.status)
        except Exception:
            raise CitationMinimalityError(
                "unsupported citation minimality status"
            ) from None
        if type(self.review_required) is not bool:
            raise CitationMinimalityError("review_required must be a boolean")
        try:
            context_spans = tuple(self.excess_context_spans)
        except Exception:
            raise CitationMinimalityError(
                "excess context must contain CitationSpan records"
            ) from None
        if any(not isinstance(span, CitationSpan) for span in context_spans):
            raise CitationMinimalityError(
                "excess context must contain CitationSpan records"
            )
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "excess_context_spans", context_spans)

    @property
    def citation_offset(self) -> tuple[int, int]:
        """Return the citation's source offset."""

        return self.citation_span.source_offset

    @property
    def required_offset(self) -> tuple[int, int]:
        """Return the minimal required source offset."""

        return self.required_span.source_offset

    @property
    def minimal_offset(self) -> tuple[int, int]:
        """Alias for :attr:`required_offset`."""

        return self.required_offset

    @property
    def extra_token_count(self) -> int:
        """Alias for :attr:`excess_token_count`."""

        return self.excess_token_count

    @property
    def flagged(self) -> bool:
        """Return whether the citation requires human review."""

        return self.review_required

    @property
    def is_minimal(self) -> bool:
        """Return whether this citation passed the configured breadth budget."""

        return self.status is CitationMinimalityStatus.MINIMAL

    @property
    def excessive_context(self) -> bool:
        """Return whether extra context exceeded the configured budget."""

        return self.status is CitationMinimalityStatus.EXCESS_CONTEXT

    def to_dict(self) -> dict[str, Any]:
        """Return offsets, token counts, and controlled status only."""

        return {
            "claim_id": self.claim_id,
            "citation_id": self.citation_id,
            "citation_offset": self.citation_span.to_dict(),
            "required_offset": self.required_span.to_dict(),
            "citation_token_count": self.citation_token_count,
            "required_token_count": self.required_token_count,
            "excess_token_count": self.excess_token_count,
            "excess_context_offsets": [
                span.to_dict() for span in self.excess_context_spans
            ],
            "status": self.status.value,
            "review_required": self.review_required,
        }


@dataclass(frozen=True, slots=True)
class CitationMinimalityReport:
    """Deterministically ordered, value-free citation minimality results."""

    records: tuple[CitationMinimalityRecord, ...]
    claim_count: int
    max_excess_tokens: int = 0
    schema_version: int = CITATION_MINIMALITY_SCHEMA_VERSION
    disclaimer: str = CITATION_MINIMALITY_DISCLAIMER

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or (
            self.schema_version != CITATION_MINIMALITY_SCHEMA_VERSION
        ):
            raise CitationMinimalityError(
                "unsupported citation minimality schema version"
            )
        if type(self.claim_count) is not int or self.claim_count < 0:
            raise CitationMinimalityError("claim_count must be a non-negative integer")
        _nonnegative_int(self.max_excess_tokens, "max_excess_tokens")
        if (
            type(self.disclaimer) is not str
            or self.disclaimer != CITATION_MINIMALITY_DISCLAIMER
        ):
            raise CitationMinimalityError("citation minimality disclaimer is required")
        try:
            records = tuple(self.records)
        except Exception:
            raise CitationMinimalityError(
                "records must contain CitationMinimalityRecord values"
            ) from None
        if any(not isinstance(record, CitationMinimalityRecord) for record in records):
            raise CitationMinimalityError(
                "records must contain CitationMinimalityRecord values"
            )
        object.__setattr__(
            self,
            "records",
            tuple(sorted(records, key=_record_key)),
        )

    @property
    def citation_count(self) -> int:
        """Return the number of evaluated citations."""

        return len(self.records)

    @property
    def flagged_count(self) -> int:
        """Return the number of citations requiring human review."""

        return sum(record.review_required for record in self.records)

    @property
    def review_required_count(self) -> int:
        """Alias for :attr:`flagged_count`."""

        return self.flagged_count

    @property
    def minimal_count(self) -> int:
        """Return the number of citations within the breadth budget."""

        return self.citation_count - self.flagged_count

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic report without source or cited text."""

        return {
            "schema_version": self.schema_version,
            "claim_count": self.claim_count,
            "citation_count": self.citation_count,
            "flagged_count": self.flagged_count,
            "max_excess_tokens": self.max_excess_tokens,
            "records": [record.to_dict() for record in self.records],
            "disclaimer": self.disclaimer,
        }

    def to_json(self) -> str:
        """Serialize the report as compact, byte-stable JSON."""

        return (
            json.dumps(
                self.to_dict(),
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    def to_markdown(self) -> str:
        """Render offsets and token counts for a human review queue."""

        lines = [
            "# Citation Minimality Report",
            "",
            CITATION_MINIMALITY_DISCLAIMER,
            "",
            f"Claims: {self.claim_count}",
            f"Citations: {self.citation_count}",
            f"Flagged: {self.flagged_count}",
            "",
            "| # | Claim | Citation | Citation offset | Required offset | "
            "Citation tokens | Required tokens | Excess tokens | Status | Review |",
            "| ---: | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
        for index, record in enumerate(self.records, start=1):
            lines.append(
                "| "
                f"{index} | {record.claim_id} | {record.citation_id} | "
                f"[{record.citation_span.start}, {record.citation_span.end}) | "
                f"[{record.required_span.start}, {record.required_span.end}) | "
                f"{record.citation_token_count} | {record.required_token_count} | "
                f"{record.excess_token_count} | {record.status.value} | "
                f"{'yes' if record.review_required else 'no'} |"
            )
        return "\n".join(lines) + "\n"


def build_citation_minimality_report(
    source_text: str,
    claims: Iterable[AtomicClaim | Mapping[str, Any]],
    citations: Iterable[ClaimCitation | Mapping[str, Any]],
    *,
    max_excess_tokens: int = 0,
) -> CitationMinimalityReport:
    """Build a deterministic, value-free minimality report.

    Args:
        source_text: Source document used only in memory to validate offsets and
            count deterministic Unicode word/punctuation tokens. It is never
            retained in the returned report.
        claims: Atomic claims with opaque identifiers and caller-declared
            minimal ``required_span`` offsets.
        citations: Citation spans associated with those claim identifiers.
        max_excess_tokens: Inclusive per-citation allowance for tokens outside
            the required span. The default of zero flags every extra token.

    Returns:
        An immutable report containing only opaque identifiers, offsets, token
        counts, and review statuses.

    Raises:
        CitationMinimalityError: If inputs are malformed, out of bounds, or a
            citation refers to an unknown claim.
    """

    if type(source_text) is not str:
        raise CitationMinimalityError("source_text must be text")
    _nonnegative_int(max_excess_tokens, "max_excess_tokens")
    token_spans = tuple(
        (match.start(), match.end()) for match in _TOKEN_RE.finditer(source_text)
    )

    normalized_claims = _normalize_claims(claims)
    claims_by_id: dict[str, AtomicClaim] = {}
    required_token_indexes: dict[str, tuple[int, ...]] = {}
    for claim in normalized_claims:
        if claim.claim_id in claims_by_id:
            raise CitationMinimalityError("duplicate atomic claim id")
        _validate_in_bounds(claim.required_span, len(source_text))
        indexes = _token_indexes(token_spans, claim.required_span)
        if not indexes:
            raise CitationMinimalityError(
                "required claim spans must contain at least one token"
            )
        claims_by_id[claim.claim_id] = claim
        required_token_indexes[claim.claim_id] = indexes

    normalized_citations = _normalize_citations(citations)
    seen_citations: dict[tuple[str, str], CitationSpan] = {}
    results: list[CitationMinimalityRecord] = []
    for citation in normalized_citations:
        matched_claim: AtomicClaim | None = claims_by_id.get(citation.claim_id)
        if matched_claim is None:
            raise CitationMinimalityError("citation refers to an unknown atomic claim")
        citation_id = citation.citation_id
        if citation_id is None:
            raise CitationMinimalityError("citation id is required")
        citation_key = (citation.claim_id, citation_id)
        previous_span = seen_citations.get(citation_key)
        if previous_span is not None and previous_span != citation.source_span:
            raise CitationMinimalityError("citation id has conflicting offsets")
        seen_citations[citation_key] = citation.source_span

        _validate_in_bounds(citation.source_span, len(source_text))
        citation_indexes = _token_indexes(token_spans, citation.source_span)
        required_indexes = required_token_indexes[citation.claim_id]
        covers_required_span = (
            citation.source_span.start <= matched_claim.required_span.start
            and citation.source_span.end >= matched_claim.required_span.end
        )
        if covers_required_span:
            required_index_set = set(required_indexes)
            excess_token_count = sum(
                index not in required_index_set for index in citation_indexes
            )
            status = (
                CitationMinimalityStatus.EXCESS_CONTEXT
                if excess_token_count > max_excess_tokens
                else CitationMinimalityStatus.MINIMAL
            )
            excess_context_spans = _excess_context_spans(
                citation.source_span,
                matched_claim.required_span,
            )
        else:
            excess_token_count = 0
            status = CitationMinimalityStatus.MISSING_REQUIRED_SPAN
            excess_context_spans = ()

        results.append(
            CitationMinimalityRecord(
                claim_id=citation.claim_id,
                citation_id=citation_id,
                citation_span=citation.source_span,
                required_span=matched_claim.required_span,
                citation_token_count=len(citation_indexes),
                required_token_count=len(required_indexes),
                excess_token_count=excess_token_count,
                excess_context_spans=excess_context_spans,
                status=status,
                review_required=status is not CitationMinimalityStatus.MINIMAL,
            )
        )

    return CitationMinimalityReport(
        records=tuple(results),
        claim_count=len(normalized_claims),
        max_excess_tokens=max_excess_tokens,
    )


def check_citation_minimality(
    source_text: str,
    claims: Iterable[AtomicClaim | Mapping[str, Any]],
    citations: Iterable[ClaimCitation | Mapping[str, Any]],
    *,
    max_excess_tokens: int = 0,
) -> CitationMinimalityReport:
    """Check guarded claim citations and return a value-free report."""

    return build_citation_minimality_report(
        source_text,
        claims,
        citations,
        max_excess_tokens=max_excess_tokens,
    )


def export_citation_minimality(
    source_text: str,
    claims: Iterable[AtomicClaim | Mapping[str, Any]],
    citations: Iterable[ClaimCitation | Mapping[str, Any]],
    *,
    max_excess_tokens: int = 0,
) -> dict[str, Any]:
    """Return a JSON-ready citation minimality report."""

    return check_citation_minimality(
        source_text,
        claims,
        citations,
        max_excess_tokens=max_excess_tokens,
    ).to_dict()


def _normalize_claims(
    claims: Iterable[AtomicClaim | Mapping[str, Any]],
) -> tuple[AtomicClaim, ...]:
    values = _collection(claims, "claims")
    return tuple(AtomicClaim.from_obj(value) for value in values)


def _normalize_citations(
    citations: Iterable[ClaimCitation | Mapping[str, Any]],
) -> tuple[ClaimCitation, ...]:
    values = _collection(citations, "citations")
    return tuple(ClaimCitation.from_obj(value) for value in values)


def _collection(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (AtomicClaim, ClaimCitation)):
        return (value,)
    if isinstance(value, Mapping):
        nested = _read(value, (label,))
        value = (value,) if nested is _MISSING else nested
    if isinstance(value, (str, bytes, bytearray)):
        raise CitationMinimalityError(f"{label} must be an iterable of records")
    try:
        return tuple(value)
    except Exception:
        raise CitationMinimalityError(
            f"{label} must be an iterable of records"
        ) from None


def _span_value(value: Any, names: tuple[str, ...]) -> Any:
    for name in names:
        candidate = _read(value, (name,))
        if candidate is not _MISSING:
            return candidate
    return _MISSING


def _span_from_fields(
    value: Any,
    start_names: tuple[str, ...],
    end_names: tuple[str, ...],
) -> Any:
    start = _read(value, start_names)
    end = _read(value, end_names)
    if start is _MISSING and end is _MISSING:
        return _MISSING
    if start is _MISSING or end is _MISSING:
        raise CitationMinimalityError("citation spans require start and end offsets")
    return CitationSpan(start=start, end=end)


def _read(value: Any, names: tuple[str, ...]) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            try:
                if name in value:
                    return value[name]
            except Exception:
                raise CitationMinimalityError(
                    "citation record fields are invalid"
                ) from None
        return _MISSING
    for name in names:
        try:
            return getattr(value, name)
        except AttributeError:
            continue
        except Exception:
            raise CitationMinimalityError(
                "citation record fields are invalid"
            ) from None
    return _MISSING


def _opaque_reference(value: Any) -> str:
    if type(value) is not str or _OPAQUE_REFERENCE_RE.fullmatch(value) is None:
        raise CitationMinimalityError(
            "citation identifiers must be opaque sha256 references"
        )
    return value


def _derived_citation_reference(claim_id: str, span: CitationSpan) -> str:
    payload = f"citation\0{claim_id}\0{span.start}\0{span.end}".encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _source_offsets(start: Any, end: Any) -> tuple[int, int]:
    if type(start) is not int or type(end) is not int:
        raise CitationMinimalityError("citation offsets must be integers")
    if start < 0 or end <= start:
        raise CitationMinimalityError(
            "citation offsets must form a non-empty half-open span"
        )
    return start, end


def _nonnegative_int(value: Any, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise CitationMinimalityError(f"{field_name} must be a non-negative integer")
    return value


def _validate_in_bounds(span: CitationSpan, source_length: int) -> None:
    if span.end > source_length:
        raise CitationMinimalityError("citation offsets exceed source text")


def _token_indexes(
    token_spans: tuple[tuple[int, int], ...],
    span: CitationSpan,
) -> tuple[int, ...]:
    return tuple(
        index
        for index, (start, end) in enumerate(token_spans)
        if start < span.end and end > span.start
    )


def _excess_context_spans(
    citation_span: CitationSpan,
    required_span: CitationSpan,
) -> tuple[CitationSpan, ...]:
    context: list[CitationSpan] = []
    if citation_span.start < required_span.start:
        context.append(CitationSpan(citation_span.start, required_span.start))
    if citation_span.end > required_span.end:
        context.append(CitationSpan(required_span.end, citation_span.end))
    return tuple(context)


def _record_key(
    record: CitationMinimalityRecord,
) -> tuple[str, str, int, int, int, int]:
    return (
        record.claim_id,
        record.citation_id,
        record.citation_span.start,
        record.citation_span.end,
        record.required_span.start,
        record.required_span.end,
    )
