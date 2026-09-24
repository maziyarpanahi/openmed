"""Strict, value-free section scoping for SDOH candidate evidence.

SDOH cues in an assessment, plan, templated instruction, or third-party
history are not interchangeable with evidence in a configured clinical
section.  This module applies a deterministic section policy before a caller
dispatches candidates to an SDOH extractor.

Only candidates that are fully contained by an allowed section are retained
when an allowed section is present.  If no allowed section is available, the
caller must choose an explicit fallback.  The default fallback is fail-closed
(``"reject"``).  Scope reports retain counts, section labels, reasons, and
offset-free policy metadata only; candidate text and discarded candidate
objects are never copied into a report.

The implementation uses the local rules-first section detector when sections
are omitted.  Learned section detection and network services are not invoked.
This is a deterministic extraction guard, not a clinical decision or a
compliance certification.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final, Literal, cast

from .sdoh import SOCIAL_HISTORY_SECTION
from .sections import UNSECTIONED_SECTION, detect_sections

SectionScopeFallback = Literal["reject", "unsectioned", "document"]

SDOH_SECTION_SCOPE_SCHEMA_VERSION: Final[int] = 1
SDOH_SECTION_SCOPE_DISCLAIMER: Final[str] = (
    "SDOH section scope is a deterministic review guard, not a clinical "
    "decision or compliance certification."
)
DEFAULT_SDOH_SECTIONS: Final[tuple[str, ...]] = (SOCIAL_HISTORY_SECTION,)
SDOH_SCOPE_FALLBACK_REJECT: Final[SectionScopeFallback] = "reject"
SDOH_SCOPE_FALLBACK_UNSECTIONED: Final[SectionScopeFallback] = "unsectioned"
SDOH_SCOPE_FALLBACK_DOCUMENT: Final[SectionScopeFallback] = "document"
SDOH_SCOPE_FALLBACKS: Final[tuple[str, ...]] = (
    SDOH_SCOPE_FALLBACK_REJECT,
    SDOH_SCOPE_FALLBACK_UNSECTIONED,
    SDOH_SCOPE_FALLBACK_DOCUMENT,
)

SpanOffset = tuple[int, int]

_INVALID_SECTION = "invalid"
_CROSS_SECTION_BOUNDARY = "cross_section_boundary"
_NO_SECTION = UNSECTIONED_SECTION
_FALLBACK_ALIASES = {
    "all": SDOH_SCOPE_FALLBACK_DOCUMENT,
    "allow_all": SDOH_SCOPE_FALLBACK_DOCUMENT,
    "document_level": SDOH_SCOPE_FALLBACK_DOCUMENT,
    "full_document": SDOH_SCOPE_FALLBACK_DOCUMENT,
    "none": SDOH_SCOPE_FALLBACK_REJECT,
    "deny": SDOH_SCOPE_FALLBACK_REJECT,
    "exclude": SDOH_SCOPE_FALLBACK_REJECT,
    "strict": SDOH_SCOPE_FALLBACK_REJECT,
    "section": SDOH_SCOPE_FALLBACK_UNSECTIONED,
}


@dataclass(frozen=True, slots=True)
class _SectionRange:
    label: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SDOHSectionScopePolicy:
    """Configuration for one SDOH section-boundary evaluation.

    Args:
        allowed_sections: Canonical clinical section labels whose candidate
            spans may be used.  The default is ``("social_history",)``.
        fallback: Behavior when no configured section is present.  ``reject``
            fails closed; ``unsectioned`` permits only candidates contained in
            an explicitly unsectioned range; ``document`` permits all valid
            document-local candidates and should be used only by an explicit
            caller policy.
    """

    allowed_sections: tuple[str, ...] = DEFAULT_SDOH_SECTIONS
    fallback: SectionScopeFallback = SDOH_SCOPE_FALLBACK_REJECT

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "allowed_sections",
            _normalize_section_labels(self.allowed_sections, "allowed_sections"),
        )
        object.__setattr__(self, "fallback", _normalize_fallback(self.fallback))

    def to_dict(self) -> dict[str, Any]:
        """Return policy metadata without any candidate or source value."""

        return {
            "allowed_sections": list(self.allowed_sections),
            "fallback": self.fallback,
        }


@dataclass(frozen=True, slots=True)
class SDOHSectionScopeReport:
    """Aggregate, value-free evidence from one scope evaluation.

    ``excluded_candidate_counts`` is keyed by the canonical section that
    contained the candidate.  Candidates crossing a boundary are counted in
    the section containing their start offset and are also represented in
    ``excluded_reason_counts``.  A candidate outside every supplied section is
    counted under ``"unsectioned"``.
    """

    policy: SDOHSectionScopePolicy
    sections_available: bool
    configured_section_count: int
    fallback_used: bool
    input_candidate_count: int
    included_candidate_count: int
    excluded_candidate_counts: Mapping[str, int] = field(default_factory=dict)
    excluded_reason_counts: Mapping[str, int] = field(default_factory=dict)
    schema_version: int = SDOH_SECTION_SCOPE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SDOHSectionScopePolicy):
            raise TypeError("policy must be an SDOHSectionScopePolicy")
        if type(self.sections_available) is not bool:
            raise TypeError("sections_available must be a boolean")
        if type(self.fallback_used) is not bool:
            raise TypeError("fallback_used must be a boolean")
        for field_name in (
            "configured_section_count",
            "input_candidate_count",
            "included_candidate_count",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{field_name} must be an integer count")
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.included_candidate_count > self.input_candidate_count:
            raise ValueError("included candidate count exceeds input count")
        if self.schema_version != SDOH_SECTION_SCOPE_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH section scope schema version")
        object.__setattr__(
            self,
            "excluded_candidate_counts",
            _count_mapping(self.excluded_candidate_counts, "excluded_candidate_counts"),
        )
        object.__setattr__(
            self,
            "excluded_reason_counts",
            _count_mapping(self.excluded_reason_counts, "excluded_reason_counts"),
        )

    @property
    def excluded_candidate_count(self) -> int:
        """Return the number of candidates excluded by the policy."""

        return self.input_candidate_count - self.included_candidate_count

    @property
    def excluded_counts_by_section(self) -> Mapping[str, int]:
        """Alias for :attr:`excluded_candidate_counts`."""

        return self.excluded_candidate_counts

    @property
    def excluded_by_section(self) -> Mapping[str, int]:
        """Short alias for callers building aggregate review metrics."""

        return self.excluded_candidate_counts

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic report metadata containing no raw text."""

        return {
            "schema_version": self.schema_version,
            "policy": self.policy.to_dict(),
            "sections_available": self.sections_available,
            "configured_section_count": self.configured_section_count,
            "fallback_used": self.fallback_used,
            "input_candidate_count": self.input_candidate_count,
            "included_candidate_count": self.included_candidate_count,
            "excluded_candidate_count": self.excluded_candidate_count,
            "excluded_candidate_counts": dict(self.excluded_candidate_counts),
            "excluded_reason_counts": dict(self.excluded_reason_counts),
            "disclaimer": SDOH_SECTION_SCOPE_DISCLAIMER,
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize deterministic, value-free scope evidence as JSON."""

        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=indent,
            separators=(",", ":") if indent is None else None,
            sort_keys=True,
        )

    def to_markdown(self) -> str:
        """Render a compact counts-only review summary."""

        lines = [
            "# SDOH section-scope report",
            "",
            SDOH_SECTION_SCOPE_DISCLAIMER,
            "",
            f"- Fallback: `{self.policy.fallback}`",
            f"- Fallback used: `{str(self.fallback_used).lower()}`",
            f"- Candidates: {self.input_candidate_count}",
            f"- Included: {self.included_candidate_count}",
            f"- Excluded: {self.excluded_candidate_count}",
            "",
            "## Excluded candidates by section",
            "",
            "| Section | Count |",
            "|---|---:|",
        ]
        if self.excluded_candidate_counts:
            lines.extend(
                f"| `{section}` | {count} |"
                for section, count in self.excluded_candidate_counts.items()
            )
        else:
            lines.append("| none | 0 |")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True, slots=True)
class SDOHSectionScopeResult:
    """Scoped candidates plus a value-free section-scope report.

    ``candidates`` contains only candidates retained for downstream local
    extraction.  Excluded candidate objects are deliberately not retained.
    The field is hidden from the result representation and serialization so a
    result can be safely logged through its report methods.
    """

    candidates: tuple[Any, ...] = field(repr=False)
    report: SDOHSectionScopeReport

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if not isinstance(self.report, SDOHSectionScopeReport):
            raise TypeError("report must be an SDOHSectionScopeReport")
        if len(self.candidates) != self.report.included_candidate_count:
            raise ValueError("candidate count does not match scope report")

    @property
    def included_candidates(self) -> tuple[Any, ...]:
        """Return candidates retained for downstream extraction."""

        return self.candidates

    @property
    def scoped_candidates(self) -> tuple[Any, ...]:
        """Alias for :attr:`included_candidates`."""

        return self.candidates

    @property
    def excluded_candidate_counts(self) -> Mapping[str, int]:
        """Return excluded candidate counts grouped by section."""

        return self.report.excluded_candidate_counts

    @property
    def candidate_offsets(self) -> tuple[SpanOffset, ...]:
        """Return retained offsets without exposing candidate values."""

        offsets: list[SpanOffset] = []
        for candidate in self.candidates:
            offset, error = _candidate_offset(candidate)
            if error is not None or offset is None:
                raise ValueError("retained candidate has invalid offsets")
            offsets.append(offset)
        return tuple(offsets)

    def to_dict(self) -> dict[str, Any]:
        """Return only the value-free report representation."""

        return self.report.to_dict()

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize only value-free scope evidence."""

        return self.report.to_json(indent=indent)

    def __iter__(self):
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __repr__(self) -> str:
        return (
            "SDOHSectionScopeResult("
            f"included_candidate_count={len(self.candidates)}, "
            f"excluded_candidate_count={self.report.excluded_candidate_count})"
        )


def scope_sdoh_candidates(
    text: str,
    candidates: Iterable[Any],
    sections: Iterable[Mapping[str, Any] | object] | None = None,
    *,
    allowed_sections: Iterable[str] = DEFAULT_SDOH_SECTIONS,
    fallback: str = SDOH_SCOPE_FALLBACK_REJECT,
    policy: SDOHSectionScopePolicy | None = None,
) -> SDOHSectionScopeResult:
    """Restrict SDOH candidates to fully contained configured sections.

    Args:
        text: Original clinical document text. It is used only for local
            section detection and bounds validation; it is never copied into
            the returned report.
        candidates: Candidate mappings or objects exposing integer
            ``start``/``end`` offsets. A two-item ``span`` or ``offsets``
            attribute is also accepted. Candidate objects retained for
            extraction are returned by the result, while discarded objects are
            not retained.
        sections: Optional caller-supplied section spans. When omitted, the
            rules-first local detector runs with learned refinement disabled.
        allowed_sections: Canonical section labels eligible for extraction.
            Defaults to Social History.
        fallback: Explicit behavior when no allowed section is detected:
            ``"reject"`` (default), ``"unsectioned"``, or ``"document"``.
            The latter two are opt-in and are recorded in the report.
        policy: Optional typed policy. When provided, its section and fallback
            values are used; passing non-default policy keywords alongside it
            is rejected to avoid ambiguous configuration.

    Returns:
        An :class:`SDOHSectionScopeResult` with retained candidates and a
        counts-only report.

    Raises:
        TypeError: If ``text`` or policy inputs have invalid types.
        ValueError: If supplied section metadata has invalid or overlapping
            bounds, or if policy arguments conflict.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    if policy is not None:
        if not isinstance(policy, SDOHSectionScopePolicy):
            raise TypeError("policy must be an SDOHSectionScopePolicy")
        default_policy = SDOHSectionScopePolicy()
        requested_policy = _build_policy(
            allowed_sections,
            fallback,
        )
        if requested_policy != default_policy:
            raise ValueError("policy cannot be combined with scope keywords")
        active_policy = policy
    else:
        active_policy = _build_policy(allowed_sections, fallback)

    section_ranges = _resolve_sections(text, sections)
    configured_ranges = tuple(
        section
        for section in section_ranges
        if section.label in active_policy.allowed_sections
    )
    fallback_used = not configured_ranges

    materialized_candidates = _materialize_candidates(candidates)
    retained: list[Any] = []
    excluded_by_section: dict[str, int] = {}
    excluded_by_reason: dict[str, int] = {}

    for candidate in materialized_candidates:
        offset, error = _candidate_offset(candidate, text_length=len(text))
        if error is not None or offset is None:
            _record_exclusion(
                excluded_by_section,
                excluded_by_reason,
                _INVALID_SECTION,
                _INVALID_SECTION,
            )
            continue

        containing = _containing_section(offset, section_ranges)
        if configured_ranges:
            if (
                containing is not None
                and containing.label in active_policy.allowed_sections
            ):
                retained.append(candidate)
                continue
            reason = (
                _CROSS_SECTION_BOUNDARY
                if containing is None
                and _section_label_for_start(offset[0], section_ranges) != _NO_SECTION
                else "outside_allowed_section"
            )
            section_label = _section_label_for_start(offset[0], section_ranges)
            _record_exclusion(
                excluded_by_section,
                excluded_by_reason,
                section_label,
                reason,
            )
            continue

        if _fallback_allows_candidate(
            offset,
            containing,
            active_policy.fallback,
        ):
            retained.append(candidate)
            continue

        section_label = containing.label if containing is not None else _NO_SECTION
        _record_exclusion(
            excluded_by_section,
            excluded_by_reason,
            section_label,
            "fallback_rejected",
        )

    report = SDOHSectionScopeReport(
        policy=active_policy,
        sections_available=bool(section_ranges),
        configured_section_count=len(configured_ranges),
        fallback_used=fallback_used,
        input_candidate_count=len(materialized_candidates),
        included_candidate_count=len(retained),
        excluded_candidate_counts=excluded_by_section,
        excluded_reason_counts=excluded_by_reason,
    )
    return SDOHSectionScopeResult(candidates=tuple(retained), report=report)


def filter_sdoh_candidates(
    text: str,
    candidates: Iterable[Any],
    sections: Iterable[Mapping[str, Any] | object] | None = None,
    *,
    allowed_sections: Iterable[str] = DEFAULT_SDOH_SECTIONS,
    fallback: str = SDOH_SCOPE_FALLBACK_REJECT,
    policy: SDOHSectionScopePolicy | None = None,
) -> SDOHSectionScopeResult:
    """Alias for :func:`scope_sdoh_candidates`."""

    return scope_sdoh_candidates(
        text,
        candidates,
        sections,
        allowed_sections=allowed_sections,
        fallback=fallback,
        policy=policy,
    )


def _build_policy(
    allowed_sections: Iterable[str],
    fallback: str,
) -> SDOHSectionScopePolicy:
    return SDOHSectionScopePolicy(
        allowed_sections=_normalize_section_labels(
            allowed_sections,
            "allowed_sections",
        ),
        fallback=_normalize_fallback(fallback),
    )


def _resolve_sections(
    text: str,
    sections: Iterable[Mapping[str, Any] | object] | None,
) -> tuple[_SectionRange, ...]:
    if sections is None:
        sections = detect_sections(text, include_unsectioned=True)
    if isinstance(sections, (str, bytes)):
        raise TypeError("sections must be an iterable of section spans")
    try:
        materialized = tuple(sections)
    except TypeError:
        raise TypeError("sections must be an iterable of section spans") from None

    ranges: list[_SectionRange] = []
    for section in materialized:
        label_value = _field_value(section, "label")
        if label_value is None:
            label_value = _field_value(section, "name")
        label = _normalize_section_label(label_value, "section label")
        offset, error = _candidate_offset(section, text_length=len(text))
        if error is not None:
            raise ValueError("section spans require valid in-document offsets")
        assert offset is not None
        ranges.append(_SectionRange(label=label, start=offset[0], end=offset[1]))

    ordered = tuple(sorted(ranges, key=lambda item: (item.start, item.end, item.label)))
    for previous, current in zip(ordered, ordered[1:]):
        if current.start < previous.end:
            raise ValueError("section spans must not overlap")
    return ordered


def _materialize_candidates(candidates: Iterable[Any]) -> tuple[Any, ...]:
    if isinstance(candidates, (str, bytes)):
        raise TypeError("candidates must be an iterable of span records")
    try:
        return tuple(candidates)
    except TypeError:
        raise TypeError("candidates must be an iterable of span records") from None


def _candidate_offset(
    candidate: object,
    *,
    text_length: int | None = None,
) -> tuple[SpanOffset | None, str | None]:
    start = _field_value(candidate, "start")
    end = _field_value(candidate, "end")
    if start is None and end is None:
        for key in ("span", "offset", "offsets", "source_span"):
            pair = _field_value(candidate, key)
            if pair is None:
                continue
            if (
                isinstance(pair, Sequence)
                and not isinstance(pair, (str, bytes))
                and len(pair) == 2
            ):
                start, end = pair
                break

    if (
        isinstance(start, bool)
        or isinstance(end, bool)
        or not isinstance(start, int)
        or not isinstance(end, int)
        or start < 0
        or end <= start
    ):
        return None, _INVALID_SECTION
    if text_length is not None and end > text_length:
        return None, _INVALID_SECTION
    return (start, end), None


def _containing_section(
    offset: SpanOffset,
    sections: Sequence[_SectionRange],
) -> _SectionRange | None:
    start, end = offset
    return next(
        (
            section
            for section in sections
            if section.start <= start and end <= section.end
        ),
        None,
    )


def _section_label_for_start(
    start: int,
    sections: Sequence[_SectionRange],
) -> str:
    for section in sections:
        if section.start <= start < section.end:
            return section.label
    return _NO_SECTION


def _fallback_allows_candidate(
    offset: SpanOffset,
    containing: _SectionRange | None,
    fallback: SectionScopeFallback,
) -> bool:
    if fallback == SDOH_SCOPE_FALLBACK_DOCUMENT:
        return True
    return (
        fallback == SDOH_SCOPE_FALLBACK_UNSECTIONED
        and containing is not None
        and containing.label == UNSECTIONED_SECTION
        and _containing_section(offset, (containing,)) is not None
    )


def _record_exclusion(
    by_section: dict[str, int],
    by_reason: dict[str, int],
    section: str,
    reason: str,
) -> None:
    by_section[section] = by_section.get(section, 0) + 1
    by_reason[reason] = by_reason.get(reason, 0) + 1


def _field_value(item: object, field_name: str) -> object:
    if isinstance(item, Mapping):
        return item.get(field_name)
    return getattr(item, field_name, None)


def _normalize_section_labels(
    labels: Iterable[str],
    field_name: str,
) -> tuple[str, ...]:
    if isinstance(labels, str | bytes):
        labels = (labels,)
    try:
        values = tuple(labels)
    except TypeError:
        raise TypeError(f"{field_name} must be an iterable of section labels") from None
    normalized: list[str] = []
    for value in values:
        label = _normalize_section_label(value, field_name)
        if label not in normalized:
            normalized.append(label)
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one section label")
    return tuple(normalized)


def _normalize_section_label(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    normalized = "_".join(value.strip().lower().replace("-", " ").split())
    normalized = normalized.replace("/", "_")
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_fallback(value: object) -> SectionScopeFallback:
    if not isinstance(value, str):
        raise TypeError("fallback must be a string")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _FALLBACK_ALIASES.get(normalized, normalized)
    if normalized not in SDOH_SCOPE_FALLBACKS:
        raise ValueError("fallback must be reject, unsectioned, or document")
    return cast(SectionScopeFallback, normalized)


def _count_mapping(value: Mapping[str, int], field_name: str) -> Mapping[str, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    normalized: dict[str, int] = {}
    for key, count in value.items():
        label = _normalize_section_label(key, f"{field_name} key")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"{field_name} values must be non-negative integers")
        if count:
            normalized[label] = count
    return MappingProxyType(dict(sorted(normalized.items())))


SDOHSectionPolicy = SDOHSectionScopePolicy
SDOHSectionScope = SDOHSectionScopeResult
apply_sdoh_section_scope = scope_sdoh_candidates

__all__ = [
    "DEFAULT_SDOH_SECTIONS",
    "SDOH_SCOPE_FALLBACKS",
    "SDOH_SCOPE_FALLBACK_DOCUMENT",
    "SDOH_SCOPE_FALLBACK_REJECT",
    "SDOH_SCOPE_FALLBACK_UNSECTIONED",
    "SDOH_SECTION_SCOPE_DISCLAIMER",
    "SDOH_SECTION_SCOPE_SCHEMA_VERSION",
    "SDOHSectionPolicy",
    "SDOHSectionScope",
    "SDOHSectionScopePolicy",
    "SDOHSectionScopeReport",
    "SDOHSectionScopeResult",
    "apply_sdoh_section_scope",
    "filter_sdoh_candidates",
    "scope_sdoh_candidates",
]
