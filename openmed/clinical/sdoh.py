"""SHAC-aligned SDOH finding schema and determinant dispatcher.

This module contains only the public SHAC trigger-and-argument shape and a
section-aware extension point. Determinant logic must use synthetic or public
data. The real Social History Annotated Corpus (SHAC) is DUA-gated, eval-only,
and must never be bundled with OpenMed or loaded by this runtime module.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

SOCIAL_HISTORY_SECTION = "social_history"
SHAC_DATA_POLICY = (
    "Real SHAC data is DUA-gated and eval-only; runtime extraction uses only "
    "synthetic or public data."
)

SpanOffset = tuple[int, int]


@dataclass(frozen=True)
class SDOHFinding:
    """One SHAC-style social-determinant trigger and its arguments.

    Args:
        category: Determinant trigger category, such as ``"tobacco"``.
        value: Trigger value or normalized determinant type.
        status: Optional SHAC Status argument.
        extent: Optional SHAC Extent argument.
        temporality: Optional SHAC Temporality argument.
        span: Half-open source character offsets for the trigger.
        score: Confidence between zero and one, inclusive.
    """

    category: str
    value: str
    status: str | None
    extent: str | None
    temporality: str | None
    span: SpanOffset
    score: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "category", _required_text(self.category, "category"))
        object.__setattr__(self, "value", _required_text(self.value, "value"))
        for field_name in ("status", "extent", "temporality"):
            object.__setattr__(
                self,
                field_name,
                _optional_text(getattr(self, field_name), field_name),
            )
        object.__setattr__(self, "span", _span_offset(self.span, "finding span"))

        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("score must be a number")
        score = float(self.score)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("score must be between 0.0 and 1.0")
        object.__setattr__(self, "score", score)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible finding mapping."""

        return {
            "category": self.category,
            "value": self.value,
            "status": self.status,
            "extent": self.extent,
            "temporality": self.temporality,
            "span": list(self.span),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SDOHFinding:
        """Build a finding from :meth:`to_dict` compatible data."""

        return cls(
            category=payload["category"],
            value=payload["value"],
            status=payload.get("status"),
            extent=payload.get("extent"),
            temporality=payload.get("temporality"),
            span=payload["span"],
            score=payload["score"],
        )


@runtime_checkable
class DeterminantExtractor(Protocol):
    """Callable contract implemented by one determinant extractor."""

    def __call__(
        self,
        text: str,
        spans: Sequence[Any],
    ) -> Iterable[SDOHFinding]:
        """Extract findings from source text and candidate spans."""


class DeterminantExtractorRegistry:
    """Deterministic registry keyed by a determinant name."""

    def __init__(self) -> None:
        self._extractors: dict[str, DeterminantExtractor] = {}

    def register(
        self,
        determinant: str,
        extractor: DeterminantExtractor,
        *,
        replace: bool = False,
    ) -> None:
        """Register a callable for one determinant.

        Args:
            determinant: Stable non-empty registry key.
            extractor: Callable satisfying :class:`DeterminantExtractor`.
            replace: Replace an existing extractor when true.

        Raises:
            TypeError: If ``extractor`` is not callable.
            ValueError: If the key is empty or already registered.
        """

        key = _required_text(determinant, "determinant")
        if not callable(extractor):
            raise TypeError("determinant extractor must be callable")
        if not replace and key in self._extractors:
            raise ValueError(f"determinant extractor already registered for {key!r}")
        self._extractors[key] = extractor

    def unregister(self, determinant: str) -> None:
        """Remove the extractor registered for ``determinant``."""

        del self._extractors[_required_text(determinant, "determinant")]

    def available(self) -> tuple[str, ...]:
        """Return registered determinant keys in deterministic order."""

        return tuple(sorted(self._extractors))

    def items(self) -> tuple[tuple[str, DeterminantExtractor], ...]:
        """Return a stable snapshot of registered extractors."""

        return tuple((key, self._extractors[key]) for key in sorted(self._extractors))

    def __len__(self) -> int:
        return len(self._extractors)


_DETERMINANT_EXTRACTORS = DeterminantExtractorRegistry()


def register_determinant_extractor(
    determinant: str,
    extractor: DeterminantExtractor,
    *,
    replace: bool = False,
) -> None:
    """Register a process-wide determinant extractor."""

    _DETERMINANT_EXTRACTORS.register(determinant, extractor, replace=replace)


def unregister_determinant_extractor(determinant: str) -> None:
    """Unregister a process-wide determinant extractor."""

    _DETERMINANT_EXTRACTORS.unregister(determinant)


def available_determinant_extractors() -> tuple[str, ...]:
    """Return process-wide determinant keys in deterministic order."""

    return _DETERMINANT_EXTRACTORS.available()


def extract_sdoh(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any] | object] | None = None,
) -> list[SDOHFinding]:
    """Dispatch registered determinant extractors over an optional section scope.

    Args:
        text: Original clinical document text.
        spans: Upstream candidate spans. Each scoped span must expose integer
            ``start`` and ``end`` mapping keys or attributes.
        sections: Optional section spans, normally returned by
            :func:`openmed.clinical.sections.detect_sections`. When supplied,
            only candidates and findings fully contained in canonical Social
            History sections are retained. When omitted, ``text`` and ``spans``
            are treated as an already selected caller-controlled window.

    Returns:
        Findings emitted by every registered determinant extractor.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    extractors = _DETERMINANT_EXTRACTORS.items()
    if not extractors:
        return []

    candidate_spans = tuple(spans)
    allowed_ranges: tuple[SpanOffset, ...] | None = None
    if sections is not None:
        allowed_ranges = _social_history_ranges(text, sections)
        if not allowed_ranges:
            return []
        candidate_spans = tuple(
            span
            for span in candidate_spans
            if _item_within_ranges(span, allowed_ranges)
        )

    findings: list[SDOHFinding] = []
    for _, extractor in extractors:
        for finding in extractor(text, candidate_spans):
            if not isinstance(finding, SDOHFinding):
                raise TypeError("determinant extractors must emit SDOHFinding values")
            if allowed_ranges is None or _offset_within_ranges(
                finding.span,
                allowed_ranges,
            ):
                findings.append(finding)
    return findings


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string when provided")
    normalized = value.strip()
    return normalized or None


def _span_offset(value: object, field_name: str) -> SpanOffset:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 2
    ):
        raise TypeError(f"{field_name} must be a two-item offset sequence")
    start, end = value
    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
    ):
        raise TypeError(f"{field_name} offsets must be integers")
    if start < 0 or end <= start:
        raise ValueError(f"{field_name} must satisfy 0 <= start < end")
    return start, end


def _social_history_ranges(
    text: str,
    sections: Iterable[Mapping[str, Any] | object],
) -> tuple[SpanOffset, ...]:
    ranges: list[SpanOffset] = []
    for section in sections:
        if _item_field(section, "label") != SOCIAL_HISTORY_SECTION:
            continue
        offset = _item_offset(section, "Social History section")
        if offset[1] > len(text):
            raise ValueError("Social History section is outside document bounds")
        ranges.append(offset)
    return tuple(sorted(ranges))


def _item_within_ranges(item: object, ranges: Sequence[SpanOffset]) -> bool:
    try:
        offset = _item_offset(item, "candidate span")
    except (TypeError, ValueError):
        return False
    return _offset_within_ranges(offset, ranges)


def _item_offset(item: object, field_name: str) -> SpanOffset:
    return _span_offset(
        (_item_field(item, "start"), _item_field(item, "end")),
        field_name,
    )


def _item_field(item: object, key: str) -> object:
    if isinstance(item, Mapping):
        return item.get(key)
    return getattr(item, key, None)


def _offset_within_ranges(
    offset: SpanOffset,
    ranges: Sequence[SpanOffset],
) -> bool:
    start, end = offset
    return any(
        range_start <= start and end <= range_end for range_start, range_end in ranges
    )


__all__ = [
    "SHAC_DATA_POLICY",
    "SOCIAL_HISTORY_SECTION",
    "DeterminantExtractor",
    "DeterminantExtractorRegistry",
    "SDOHFinding",
    "available_determinant_extractors",
    "extract_sdoh",
    "register_determinant_extractor",
    "unregister_determinant_extractor",
]
