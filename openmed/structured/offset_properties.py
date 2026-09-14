"""Deterministic offset projection properties for document adapters.

Document adapters normalize different source formats into text plus half-open
character spans. This module provides the small, dependency-free contract used
to test that boundary consistently. It deliberately does not import a parser,
read a path, or call a network service; callers provide an adapter callable
that returns an object with ``text`` and ``spans`` attributes.

The public reports contain lengths, offsets, counts, and a text hash only.
Neither the source text nor a span's surface value is retained in a report.
Python string offsets are code-point offsets, matching the existing multimodal
``ExtractedDocument`` contract.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeAlias

Offset: TypeAlias = tuple[int, int]
FormatName: TypeAlias = Literal["pdf", "ocr", "rtf", "odt", "presentation"]
OffsetFailureCategory: TypeAlias = Literal[
    "invalid_input",
    "invalid_offsets",
    "span_order",
    "span_overlap",
    "text_mismatch",
    "unmapped_span",
    "unsupported_format",
    "missing_dependency",
    "adapter_error",
]

SUPPORTED_FORMATS: tuple[FormatName, ...] = (
    "pdf",
    "ocr",
    "rtf",
    "odt",
    "presentation",
)
DEFAULT_SYNTHETIC_SEED = 2519

_FORMAT_ALIASES: Mapping[str, FormatName] = {
    "pdf": "pdf",
    "ocr": "ocr",
    "rtf": "rtf",
    "odt": "odt",
    "pptx": "presentation",
    "powerpoint": "presentation",
    "presentation": "presentation",
}
_FAILURE_CATEGORIES = frozenset(
    {
        "invalid_input",
        "invalid_offsets",
        "span_order",
        "span_overlap",
        "text_mismatch",
        "unmapped_span",
        "unsupported_format",
        "missing_dependency",
        "adapter_error",
    }
)
_TOKEN_RE = re.compile(r"\S+", re.UNICODE)
_SYNTHETIC_MARKERS = ("alpha", "beta", "gamma", "delta")


class OffsetProjectionError(ValueError):
    """A safe, categorized failure in an adapter's offset contract.

    The exception message contains only the stable category. It never embeds
    source text, a path, an adapter payload, or an exception from a parser.
    """

    category: OffsetFailureCategory

    def __init__(self, category: str) -> None:
        safe_category = category if category in _FAILURE_CATEGORIES else "adapter_error"
        self.category = safe_category  # type: ignore[assignment]
        super().__init__(f"offset projection failed ({safe_category})")


@dataclass(frozen=True)
class OffsetSpan:
    """A half-open, code-point offset with optional non-text metadata.

    ``label`` is useful to a caller while constructing test inputs, but it is
    intentionally omitted from PHI-safe reports and projections.
    """

    start: int
    end: int
    label: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or not isinstance(self.end, int)
            or isinstance(self.end, bool)
        ):
            raise OffsetProjectionError("invalid_offsets")
        if self.start < 0 or self.end < self.start:
            raise OffsetProjectionError("invalid_offsets")
        if self.label is not None and not isinstance(self.label, str):
            raise OffsetProjectionError("invalid_input")

    @property
    def offsets(self) -> Offset:
        """Return the half-open ``(start, end)`` pair."""

        return self.start, self.end

    @property
    def is_empty(self) -> bool:
        """Return whether the span is a valid zero-width annotation."""

        return self.start == self.end

    def to_dict(self) -> dict[str, int]:
        """Return an offset-only representation without a surface value."""

        return {"start": self.start, "end": self.end}


@dataclass(frozen=True)
class OffsetProjection:
    """Projection of one normalized-text span to source span indexes."""

    start: int
    end: int
    source_span_indexes: tuple[int, ...]

    @property
    def is_empty(self) -> bool:
        """Return whether the projected annotation has zero width."""

        return self.start == self.end

    @property
    def is_mapped(self) -> bool:
        """Return whether at least one non-empty source span was covered."""

        return bool(self.source_span_indexes)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic, offset-only projection record."""

        return {
            "start": self.start,
            "end": self.end,
            "source_span_indexes": list(self.source_span_indexes),
        }


@dataclass(frozen=True)
class OffsetProjectionReport:
    """Aggregate evidence for one adapter result.

    The report is deliberately aggregate-safe. ``source_text_sha256`` lets a
    caller compare repeated runs without copying the source into an artifact.
    """

    format_name: str | None
    text_length: int
    source_span_count: int
    redaction_span_count: int
    empty_redaction_count: int
    mapped_redaction_count: int
    unmapped_redaction_count: int
    source_text_sha256: str
    projections: tuple[OffsetProjection, ...]

    @property
    def passed(self) -> bool:
        """Return whether every non-empty redaction span was mapped."""

        return self.unmapped_redaction_count == 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize offset-only evidence with stable key and item ordering."""

        return {
            "format": self.format_name,
            "text_length": self.text_length,
            "source_span_count": self.source_span_count,
            "redaction_span_count": self.redaction_span_count,
            "empty_redaction_count": self.empty_redaction_count,
            "mapped_redaction_count": self.mapped_redaction_count,
            "unmapped_redaction_count": self.unmapped_redaction_count,
            "source_text_sha256": self.source_text_sha256,
            "projections": [projection.to_dict() for projection in self.projections],
        }


@dataclass(frozen=True)
class SyntheticOffsetCase:
    """One deterministic, parser-free cross-format offset fixture."""

    format_name: str
    text: str
    source_spans: tuple[OffsetSpan, ...]
    redaction_spans: tuple[OffsetSpan, ...]
    column_ranges: tuple[OffsetSpan, ...]

    def __post_init__(self) -> None:
        canonical = canonical_format_name(self.format_name)
        if not isinstance(self.text, str):
            raise OffsetProjectionError("invalid_input")

        source_spans = _normalize_spans(
            self.source_spans,
            text_length=len(self.text),
            reject_overlap=True,
        )
        redaction_spans = _normalize_spans(
            self.redaction_spans,
            text_length=len(self.text),
            reject_overlap=False,
        )
        column_ranges = _normalize_spans(
            self.column_ranges,
            text_length=len(self.text),
            reject_overlap=True,
        )
        object.__setattr__(self, "format_name", canonical)
        object.__setattr__(self, "source_spans", source_spans)
        object.__setattr__(self, "redaction_spans", redaction_spans)
        object.__setattr__(self, "column_ranges", column_ranges)

    @property
    def format(self) -> str:
        """Return the canonical adapter format name."""

        return self.format_name

    @property
    def spans(self) -> tuple[OffsetSpan, ...]:
        """Alias for source spans used by adapter-shaped test fixtures."""

        return self.source_spans

    @property
    def redactions(self) -> tuple[OffsetSpan, ...]:
        """Alias for the detector spans projected by the test suite."""

        return self.redaction_spans

    @property
    def source_text_sha256(self) -> str:
        """Return a stable hash without retaining the source text."""

        return _text_hash(self.text)

    def to_dict(self) -> dict[str, Any]:
        """Return fixture metadata without raw text or surface values."""

        return {
            "format": self.format_name,
            "text_length": len(self.text),
            "source_text_sha256": self.source_text_sha256,
            "source_span_count": len(self.source_spans),
            "redaction_span_count": len(self.redaction_spans),
            "empty_redaction_count": sum(
                span.is_empty for span in self.redaction_spans
            ),
            "column_count": len(self.column_ranges),
            "line_count": self.text.count("\n") + 1 if self.text else 0,
            "source_spans": [span.to_dict() for span in self.source_spans],
            "redaction_spans": [span.to_dict() for span in self.redaction_spans],
            "column_ranges": [span.to_dict() for span in self.column_ranges],
        }


class OffsetAdapterResult(Protocol):
    """Minimal result shape required from a format adapter under test."""

    text: str
    spans: Iterable[Any]


def canonical_format_name(format_name: str) -> FormatName:
    """Return a supported canonical name or raise a safe category error."""

    if not isinstance(format_name, str):
        raise OffsetProjectionError("invalid_input")
    canonical = _FORMAT_ALIASES.get(format_name.strip().lower())
    if canonical is None:
        raise OffsetProjectionError("unsupported_format")
    return canonical


def safe_failure_category(error: BaseException) -> OffsetFailureCategory:
    """Classify an adapter failure without exposing its message or payload."""

    if isinstance(error, OffsetProjectionError):
        return error.category

    error_name = type(error).__name__.lower()
    if "missing" in error_name and "depend" in error_name:
        return "missing_dependency"
    if "unsupported" in error_name or "format" in error_name:
        return "unsupported_format"
    if isinstance(error, (TypeError, ValueError, KeyError, AttributeError)):
        return "invalid_input"
    return "adapter_error"


def project_offset_spans(
    text: str,
    source_spans: Iterable[Any],
    redaction_spans: Iterable[Any],
    *,
    require_coverage: bool = False,
) -> tuple[OffsetProjection, ...]:
    """Project normalized-text spans to overlapping source-span indexes.

    Empty redaction spans are valid and intentionally produce no source index.
    Non-empty spans without a source overlap are returned as unmapped unless
    ``require_coverage`` is true, in which case the safe ``unmapped_span``
    category is raised.
    """

    if not isinstance(text, str):
        raise OffsetProjectionError("invalid_input")
    normalized_source = _normalize_spans(
        source_spans,
        text_length=len(text),
        reject_overlap=True,
    )
    normalized_redactions = _normalize_spans(
        redaction_spans,
        text_length=len(text),
        reject_overlap=False,
    )
    return _project_normalized(
        normalized_source,
        normalized_redactions,
        require_coverage=require_coverage,
    )


def validate_offset_projection(
    text: str,
    source_spans: Iterable[Any],
    redaction_spans: Iterable[Any] = (),
    *,
    format_name: str | None = None,
    require_coverage: bool = True,
) -> OffsetProjectionReport:
    """Validate ordering, bounds, and redaction coverage for one adapter.

    Args:
        text: Adapter-normalized text indexed by all supplied spans.
        source_spans: Ordered, non-overlapping source spans.
        redaction_spans: Detector spans to project; zero-width spans are valid.
        format_name: Optional supported format name used in the report.
        require_coverage: Require every non-empty redaction to touch a source
            span. Structural separators may be checked with this set to false.

    Returns:
        PHI-safe aggregate evidence and offset-only projection records.
    """

    if not isinstance(text, str):
        raise OffsetProjectionError("invalid_input")
    canonical = None if format_name is None else canonical_format_name(format_name)
    normalized_source = _normalize_spans(
        source_spans,
        text_length=len(text),
        reject_overlap=True,
    )
    normalized_redactions = _normalize_spans(
        redaction_spans,
        text_length=len(text),
        reject_overlap=False,
    )
    projections = _project_normalized(
        normalized_source,
        normalized_redactions,
        require_coverage=require_coverage,
    )
    empty_count = sum(projection.is_empty for projection in projections)
    mapped_count = sum(
        projection.is_mapped and not projection.is_empty for projection in projections
    )
    non_empty_count = len(projections) - empty_count
    return OffsetProjectionReport(
        format_name=canonical,
        text_length=len(text),
        source_span_count=len(normalized_source),
        redaction_span_count=len(normalized_redactions),
        empty_redaction_count=empty_count,
        mapped_redaction_count=mapped_count,
        unmapped_redaction_count=non_empty_count - mapped_count,
        source_text_sha256=_text_hash(text),
        projections=projections,
    )


def build_synthetic_offset_cases(
    *, seed: int = DEFAULT_SYNTHETIC_SEED
) -> tuple[SyntheticOffsetCase, ...]:
    """Build deterministic line, column, Unicode, and empty-span fixtures.

    The seed selects a marker from a fixed vocabulary; it does not seed global
    state, read a file, or make a network call. Every returned case uses the
    same adapter result contract while varying the format identity and source
    layout label.
    """

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise OffsetProjectionError("invalid_input")
    marker = _SYNTHETIC_MARKERS[seed % len(_SYNTHETIC_MARKERS)]
    prefixes = {
        "pdf": "page",
        "ocr": "scan",
        "rtf": "legacy",
        "odt": "writer",
        "presentation": "slide",
    }
    cases: list[SyntheticOffsetCase] = []
    for format_name in SUPPORTED_FORMATS:
        columns = (
            (
                f"{prefixes[format_name]} line α synthetic-{marker}",
                "line β 東京",
            ),
            ("column γ café", "column δ 🧪"),
        )
        text, column_ranges = _assemble_columns(columns)
        source_spans = tuple(
            OffsetSpan(match.start(), match.end(), label="source")
            for match in _TOKEN_RE.finditer(text)
        )
        empty_positions = {
            0,
            len(text),
            *(index for index, value in enumerate(text) if value == "\n"),
        }
        redactions = [
            *(
                OffsetSpan(span.start, span.end, label="redaction")
                for index, span in enumerate(source_spans)
                if index % 2 == 0
            ),
            *(
                OffsetSpan(position, position, label="empty")
                for position in sorted(empty_positions)
            ),
        ]
        if len(source_spans) >= 3:
            redactions.append(
                OffsetSpan(
                    source_spans[0].start,
                    source_spans[2].end,
                    label="cross-line",
                )
            )
        cases.append(
            SyntheticOffsetCase(
                format_name=format_name,
                text=text,
                source_spans=source_spans,
                redaction_spans=tuple(
                    sorted(redactions, key=lambda span: (span.start, span.end))
                ),
                column_ranges=column_ranges,
            )
        )
    return tuple(cases)


def run_offset_property_suite(
    adapters: Mapping[str, Callable[[SyntheticOffsetCase], Any] | Any],
    *,
    cases: Iterable[SyntheticOffsetCase] | None = None,
    seed: int = DEFAULT_SYNTHETIC_SEED,
) -> tuple[OffsetProjectionReport, ...]:
    """Run the common offset contract against supplied adapter callables.

    An adapter may be a callable accepting :class:`SyntheticOffsetCase` or an
    object exposing ``extract(case)``. Its result must expose ``text`` and
    ``spans``. Adapter exceptions are converted to categorized failures without
    copying their messages into the resulting exception.
    """

    if not isinstance(adapters, Mapping):
        raise OffsetProjectionError("invalid_input")
    normalized_adapters: dict[FormatName, Any] = {}
    for name, adapter in adapters.items():
        canonical = canonical_format_name(name)
        if canonical in normalized_adapters:
            raise OffsetProjectionError("invalid_input")
        normalized_adapters[canonical] = adapter

    selected_cases = (
        build_synthetic_offset_cases(seed=seed) if cases is None else tuple(cases)
    )
    if not selected_cases:
        raise OffsetProjectionError("invalid_input")

    reports: list[OffsetProjectionReport] = []
    for case in selected_cases:
        if not isinstance(case, SyntheticOffsetCase):
            raise OffsetProjectionError("invalid_input")
        adapter = normalized_adapters.get(case.format_name)
        if adapter is None:
            raise OffsetProjectionError("unsupported_format")
        try:
            result = adapter(case) if callable(adapter) else adapter.extract(case)
            result_text = _result_field(result, "text")
            result_spans = _result_field(result, "spans")
        except Exception as error:
            raise OffsetProjectionError(safe_failure_category(error)) from None
        if result_text != case.text:
            raise OffsetProjectionError("text_mismatch")
        reports.append(
            validate_offset_projection(
                result_text,
                result_spans,
                case.redaction_spans,
                format_name=case.format_name,
            )
        )
    return tuple(reports)


def _result_field(result: Any, name: str) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise OffsetProjectionError("invalid_input")
        return result[name]
    value = getattr(result, name, None)
    if value is None:
        raise OffsetProjectionError("invalid_input")
    return value


def _normalize_spans(
    spans: Iterable[Any],
    *,
    text_length: int,
    reject_overlap: bool,
) -> tuple[OffsetSpan, ...]:
    try:
        normalized = tuple(_coerce_span(span) for span in spans)
    except OffsetProjectionError:
        raise
    except Exception:
        raise OffsetProjectionError("invalid_input") from None

    previous: OffsetSpan | None = None
    for span in normalized:
        if span.end > text_length:
            raise OffsetProjectionError("invalid_offsets")
        if previous is not None:
            if _span_key(span) < _span_key(previous):
                raise OffsetProjectionError("span_order")
            if reject_overlap and previous.end > span.start:
                raise OffsetProjectionError("span_overlap")
        previous = span
    return normalized


def _coerce_span(value: Any) -> OffsetSpan:
    if isinstance(value, OffsetSpan):
        return value

    start: Any = None
    end: Any = None
    label: Any = None
    if isinstance(value, Mapping):
        start = value.get("start")
        end = value.get("end")
        label = value.get("label")
        if start is None or end is None:
            offset = value.get("offset", value.get("span", value.get("char_span")))
            start, end = _pair_values(offset)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        start, end = _pair_values(value)
    else:
        start = getattr(value, "start", None)
        end = getattr(value, "end", None)
        label = getattr(value, "label", None)
        if start is None or end is None:
            start, end = _pair_values(getattr(value, "offset", None))

    try:
        return OffsetSpan(start=start, end=end, label=label)
    except OffsetProjectionError:
        raise
    except Exception:
        raise OffsetProjectionError("invalid_input") from None


def _pair_values(value: Any) -> tuple[Any, Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) >= 2:
            return value[0], value[1]
    raise OffsetProjectionError("invalid_input")


def _project_normalized(
    source_spans: Sequence[OffsetSpan],
    redaction_spans: Sequence[OffsetSpan],
    *,
    require_coverage: bool,
) -> tuple[OffsetProjection, ...]:
    projections: list[OffsetProjection] = []
    for redaction in redaction_spans:
        source_indexes = tuple(
            index
            for index, source in enumerate(source_spans)
            if source.start < redaction.end and source.end > redaction.start
        )
        if require_coverage and not redaction.is_empty and not source_indexes:
            raise OffsetProjectionError("unmapped_span")
        projections.append(
            OffsetProjection(
                start=redaction.start,
                end=redaction.end,
                source_span_indexes=source_indexes,
            )
        )
    return tuple(projections)


def _span_key(span: OffsetSpan) -> tuple[int, int]:
    return span.start, span.end


def _assemble_columns(
    columns: Sequence[Sequence[str]],
) -> tuple[str, tuple[OffsetSpan, ...]]:
    parts: list[str] = []
    ranges: list[OffsetSpan] = []
    cursor = 0
    for column_index, lines in enumerate(columns):
        column_text = "\n".join(lines)
        if column_index:
            parts.append("\n\n")
            cursor += 2
        start = cursor
        parts.append(column_text)
        cursor += len(column_text)
        ranges.append(OffsetSpan(start, cursor, label=f"column-{column_index}"))
    return "".join(parts), tuple(ranges)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_SYNTHETIC_SEED",
    "SUPPORTED_FORMATS",
    "Offset",
    "OffsetAdapterResult",
    "OffsetFailureCategory",
    "OffsetProjection",
    "OffsetProjectionError",
    "OffsetProjectionReport",
    "OffsetSpan",
    "SyntheticOffsetCase",
    "build_synthetic_offset_cases",
    "canonical_format_name",
    "project_offset_spans",
    "run_offset_property_suite",
    "safe_failure_category",
    "validate_offset_projection",
]
