"""SHAC-aligned SDOH finding schema and determinant dispatcher.

This module contains only the public SHAC trigger-and-argument shape and a
section-aware extension point. Determinant logic must use synthetic or public
data. The real Social History Annotated Corpus (SHAC) is DUA-gated, eval-only,
and must never be bundled with OpenMed or loaded by this runtime module.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from typing import Any, Protocol, runtime_checkable

import yaml

from .context import (
    HISTORICAL,
    HYPOTHETICAL,
    NEGATED,
    resolve_negation,
    resolve_temporality,
)
from .status_vocab import normalize_substance_status

SOCIAL_HISTORY_SECTION = "social_history"

SDOH_SUBSTANCE_CUES_RESOURCE = "data/sdoh_substance_cues.yaml"
_SUBSTANCE_CUES_PACKAGE = "openmed.clinical"
_SUBSTANCE_CATEGORIES = (
    "tobacco",
    "alcohol",
    "drug",
)
_SUBSTANCE_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?<!\w)(?:but|however|although|whereas)(?!\w)",
    re.IGNORECASE,
)
_SUBSTANCE_COORDINATOR_RE = re.compile(
    r",|(?<!\w)(?:and|or)(?!\w)",
    re.IGNORECASE,
)
_SUBSTANCE_LOCAL_STATUS_RE = re.compile(
    r"(?<!\w)(?:"
    r"active|current(?:ly)?|former|ex[-\s]?smoker|quit|stopped|"
    r"past|remote|history\s+of|hx\s+of|in\s+remission|status\s+post|s/p|"
    r"den(?:y|ies|ied)|never|none|no|not|without|does\s+not|"
    r"abstain(?:s|ed|ing)?|abstinent|non[-\s]?smoker|"
    r"smoker|smoking|uses|drinks?|vapes?|vaping|"
    r"occasional(?:ly)?|daily|weekly|monthly|rarely"
    r")(?!\w)",
    re.IGNORECASE,
)

_SDOH_SUBSTANCE_STATUS = {
    "current": "current",
    "former": "past",
    "never": "none",
    "unknown": "unknown",
}

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


def _extract_tobacco(
    text: str,
    spans: Sequence[Any],
) -> list[SDOHFinding]:
    del spans
    return _extract_substance_category(
        text,
        "tobacco",
    )


def _parse_tobacco_extent(text: str) -> str | None:
    match = re.search(
        r"\b(?P<amount>\d+(?:\.\d+)?)\s*pack(?:-|\s+)years?\b",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    amount = match.group("amount")
    return f"{amount} pack-years"


def _extract_alcohol(
    text: str,
    spans: Sequence[Any],
) -> list[SDOHFinding]:
    del spans
    return _extract_substance_category(
        text,
        "alcohol",
    )


def _parse_alcohol_extent(text: str) -> str | None:
    match = re.search(
        r"\b(?P<amount>\d+(?:\.\d+)?)\s+drinks?"
        r"\s*(?:/|per\s+|a\s+)week\b",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    amount = match.group("amount")
    return f"{amount} drinks/week"


def _extract_drug(
    text: str,
    spans: Sequence[Any],
) -> list[SDOHFinding]:
    del spans
    return _extract_substance_category(
        text,
        "drug",
    )


def _parse_drug_extent(text: str) -> str | None:
    match = re.search(
        r"\b(?:occasional(?:ly)?|daily|weekly|monthly|rarely)\b",
        text,
        re.IGNORECASE,
    )

    if match is None:
        return None

    value = match.group(0).lower()

    if value == "occasionally":
        return "occasional"

    return value


def _parse_substance_extent(
    category: str,
    text: str,
) -> str | None:
    if category == "tobacco":
        return _parse_tobacco_extent(text)

    if category == "alcohol":
        return _parse_alcohol_extent(text)

    if category == "drug":
        return _parse_drug_extent(text)

    return None


@lru_cache(maxsize=1)
def _load_substance_cues() -> dict[str, tuple[str, ...]]:
    resource = resources.files(_SUBSTANCE_CUES_PACKAGE).joinpath(
        SDOH_SUBSTANCE_CUES_RESOURCE
    )

    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))

    if not isinstance(payload, Mapping):
        raise ValueError("substance cue resource must be a mapping")

    if payload.get("schema_version") != 1:
        raise ValueError("substance cue resource requires schema_version 1")

    determinants = payload.get("determinants")

    if not isinstance(determinants, Mapping):
        raise ValueError("substance cue resource requires determinants")

    result: dict[str, tuple[str, ...]] = {}

    for category in _SUBSTANCE_CATEGORIES:
        entry = determinants.get(category)

        if not isinstance(entry, Mapping):
            raise ValueError(
                f"substance cue resource requires determinant {category!r}"
            )

        triggers = entry.get("triggers")

        if (
            not isinstance(triggers, Sequence)
            or isinstance(triggers, str | bytes)
            or not triggers
        ):
            raise ValueError(f"substance determinant {category!r} requires triggers")

        cleaned: list[str] = []

        for cue in triggers:
            if not isinstance(cue, str) or not cue.strip():
                raise ValueError(
                    f"substance determinant {category!r} contains an invalid trigger"
                )

            normalized = " ".join(cue.split())

            if normalized not in cleaned:
                cleaned.append(normalized)

        result[category] = tuple(cleaned)

    return result


def _substance_context_bounds(
    text: str,
    start: int,
    end: int,
) -> SpanOffset:
    boundaries = ".;\n!?"

    left = max(text.rfind(boundary, 0, start) for boundary in boundaries)

    right_positions = [text.find(boundary, end) for boundary in boundaries]

    right_positions = [position for position in right_positions if position != -1]

    right = min(right_positions) if right_positions else len(text)

    left += 1

    for boundary in _SUBSTANCE_CLAUSE_BOUNDARY_RE.finditer(text, left, right):
        if boundary.end() <= start:
            left = boundary.end()
        elif boundary.start() >= end:
            right = boundary.start()
            break

    return _coordinated_substance_context_bounds(
        text,
        start,
        end,
        left,
        right,
    )


def _coordinated_substance_context_bounds(
    text: str,
    start: int,
    end: int,
    left: int,
    right: int,
) -> SpanOffset:
    """Isolate explicit statuses while preserving shared coordinated cues."""

    coordinators = tuple(_SUBSTANCE_COORDINATOR_RE.finditer(text, left, right))
    if not coordinators:
        return left, right

    segments: list[SpanOffset] = []
    segment_start = left
    for coordinator in coordinators:
        segments.append((segment_start, coordinator.start()))
        segment_start = coordinator.end()
    segments.append((segment_start, right))

    target_index = next(
        (
            index
            for index, (segment_start, segment_end) in enumerate(segments)
            if segment_start <= start and end <= segment_end
        ),
        None,
    )
    if target_index is None:
        return left, right

    segment_categories = tuple(
        _substance_categories_in_text(text[segment_start:segment_end])
        for segment_start, segment_end in segments
    )
    categories = set().union(*segment_categories)
    if len(categories) < 2:
        return left, right

    local_status = tuple(
        bool(categories_in_segment)
        and _has_local_substance_status(
            text[segment_start:segment_end],
            categories_in_segment,
        )
        for (segment_start, segment_end), categories_in_segment in zip(
            segments,
            segment_categories,
            strict=True,
        )
    )
    if not any(local_status):
        return left, right

    if local_status[target_index]:
        left = segments[target_index][0]
    else:
        prior_local = [index for index in range(target_index) if local_status[index]]
        if prior_local:
            left = segments[prior_local[-1]][0]

    later_local = [
        index for index in range(target_index + 1, len(segments)) if local_status[index]
    ]
    if later_local:
        right = coordinators[later_local[0] - 1].start()

    return left, right


def _substance_categories_in_text(text: str) -> frozenset[str]:
    return frozenset(
        category
        for category in _SUBSTANCE_CATEGORIES
        if _substance_trigger_pattern(category).search(text) is not None
    )


def _has_local_substance_status(
    text: str,
    categories: Iterable[str],
) -> bool:
    if _SUBSTANCE_LOCAL_STATUS_RE.search(text) is not None:
        return True
    return any(
        _parse_substance_extent(category, text) is not None for category in categories
    )


def _extract_substance_category(
    text: str,
    category: str,
) -> list[SDOHFinding]:
    pattern = _substance_trigger_pattern(category)

    findings: list[SDOHFinding] = []
    seen_windows: set[SpanOffset] = set()

    for match in pattern.finditer(text):
        window = _substance_context_bounds(
            text,
            match.start(),
            match.end(),
        )
        if window in seen_windows:
            continue
        seen_windows.add(window)

        window_start, window_end = window
        context_text = text[window_start:window_end]
        target = {
            "text": match.group(0),
            "document_text": context_text,
            "start": match.start() - window_start,
            "end": match.end() - window_start,
        }

        negation = resolve_negation(target)
        temporality = resolve_temporality(target)

        status_text = context_text.strip()
        extent = _parse_substance_extent(
            category,
            status_text,
        )

        normalized_status = normalize_substance_status(
            status_text,
            negated=negation,
            temporality=temporality,
        )

        status = _SDOH_SUBSTANCE_STATUS[normalized_status]
        if status == "past":
            temporality = HISTORICAL

        if temporality == HYPOTHETICAL:
            status = "unknown"

        if (
            status == "unknown"
            and negation != NEGATED
            and temporality != HYPOTHETICAL
            and re.search(
                r"\b(?:occasional|occasionally)\b",
                status_text,
                re.IGNORECASE,
            )
        ):
            status = "current"
        if (
            status == "unknown"
            and extent is not None
            and negation != NEGATED
            and temporality != HYPOTHETICAL
        ):
            if temporality == HISTORICAL:
                status = "past"
            else:
                status = "current"

        if status == "none":
            extent = None

        findings.append(
            SDOHFinding(
                category=category,
                value=match.group(0),
                status=status,
                extent=extent,
                temporality=temporality,
                span=(match.start(), match.end()),
                score=1.0,
            )
        )
    return findings


@lru_cache(maxsize=None)
def _substance_trigger_pattern(category: str) -> re.Pattern[str]:
    cues = _load_substance_cues()[category]

    alternatives: list[str] = []

    for cue in sorted(cues, key=len, reverse=True):
        parts = cue.split()

        escaped = r"\s+".join(re.escape(part) for part in parts)

        prefix = r"(?<!\w)" if cue[0].isalnum() else ""
        suffix = r"(?!\w)" if cue[-1].isalnum() else ""

        alternatives.append(f"{prefix}(?:{escaped}){suffix}")

    return re.compile(
        "|".join(alternatives),
        re.IGNORECASE,
    )


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


register_determinant_extractor(
    "tobacco",
    _extract_tobacco,
)

register_determinant_extractor(
    "alcohol",
    _extract_alcohol,
)

register_determinant_extractor(
    "drug",
    _extract_drug,
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
