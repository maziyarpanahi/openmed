"""Deterministic temporal qualifiers for social-determinants evidence.

SDOH findings need a narrower temporal contract than the general ConText
axis.  A finding is not treated as current merely because it is written in
the present tense: this module requires an explicit temporal cue and returns
``unknown`` when no cue is available.  Conflicting cues are also returned as
``unknown`` and marked for human review.

The public result contains controlled labels, source offsets, cue offsets,
and review metadata only.  It never retains or serializes source text or the
value of an SDOH finding.  Cue matching is local and rule-based; it performs
no network access and does not consult the wall clock.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Final, Iterable, Mapping, Sequence

CURRENT: Final[str] = "current"
HISTORICAL: Final[str] = "historical"
FUTURE: Final[str] = "future"
UNKNOWN: Final[str] = "unknown"

TEMPORAL_CLASSES: Final[tuple[str, ...]] = (
    CURRENT,
    HISTORICAL,
    FUTURE,
    UNKNOWN,
)
SDOH_TEMPORAL_SCHEMA_VERSION: Final[int] = 1

SpanOffset = tuple[int, int]

_SENTENCE_BOUNDARIES = ".!?;\n"
_HARD_CLAUSE_BOUNDARY_RE = re.compile(
    r"(?<!\w)(?:but|however|although|yet)(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _CuePattern:
    temporal_class: str
    expression: str
    applies_after_target: bool = True


@dataclass(frozen=True, slots=True)
class _TemporalCue:
    temporal_class: str
    start: int
    end: int
    applies_after_target: bool


# Cues are deliberately compact and explicit.  Cues may qualify a finding on
# either side of its span; adversative clause boundaries prevent a cue for a
# separate assertion from silently changing the finding's class.
_CUE_PATTERNS: tuple[_CuePattern, ...] = (
    _CuePattern(
        CURRENT,
        r"(?<!\w)(?:at\s+present|as\s+of\s+today|as\s+of\s+now|currently|"
        r"current|presently|ongoing|active|today|now|still|continues?(?:\s+to)?)(?!\w)",
    ),
    _CuePattern(
        HISTORICAL,
        r"(?<!\w)(?:history\s+of|h/o|formerly|former|previously|prior|past|"
        r"used\s+to|no\s+longer|quit|stopped|was|were|had\s+been)(?!\w)",
    ),
    _CuePattern(
        HISTORICAL,
        r"(?<!\w)(?:in\s+(?:19|20)\d{2}|during\s+(?:19|20)\d{2}|"
        r"last\s+(?:week|month|year)|(?:\d+|one|two|three|several)\s+"
        r"years?\s+ago)(?!\w)",
        applies_after_target=True,
    ),
    _CuePattern(
        FUTURE,
        r"(?<!\w)(?:in\s+the\s+future|future|will|plans?\s+to|planned(?:\s+to)?|"
        r"expects?\s+to|expected\s+to|anticipated?|upcoming|next\s+"
        r"(?:week|month|year)|intends?\s+to|intended\s+to|scheduled\s+to|"
        r"starting\s+next)(?!\w)",
        applies_after_target=True,
    ),
    _CuePattern(
        UNKNOWN,
        r"(?<!\w)(?:unknown|unclear|uncertain|not\s+(?:documented|specified|"
        r"known)|unable\s+to\s+determine|no\s+(?:timeline|date)|"
        r"date\s+unknown)(?!\w)",
    ),
)
_COMPILED_CUE_PATTERNS: tuple[tuple[_CuePattern, re.Pattern[str]], ...] = tuple(
    (cue, re.compile(cue.expression, re.IGNORECASE)) for cue in _CUE_PATTERNS
)


@dataclass(frozen=True, slots=True)
class SDOHTemporalEvidence:
    """Value-free temporal metadata attached to one SDOH evidence span.

    Args:
        source_offsets: Half-open offsets of the SDOH evidence in the source
            document.
        temporal_class: One of ``current``, ``historical``, ``future`` or
            ``unknown``.
        cue_offsets: Half-open offsets of the temporal cues used for the
            classification.  The cue text is intentionally not retained.
        conflicting_classes: Distinct classes found in the same local scope
            when the result could not be resolved deterministically.
        review_required: Whether a human must review the temporal qualifier.
    """

    source_offsets: SpanOffset
    temporal_class: str
    cue_offsets: tuple[SpanOffset, ...] = ()
    conflicting_classes: tuple[str, ...] = ()
    review_required: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_offsets",
            _validate_offset(self.source_offsets, "source offsets"),
        )
        if self.temporal_class not in TEMPORAL_CLASSES:
            raise ValueError("unsupported SDOH temporal class")
        object.__setattr__(
            self,
            "cue_offsets",
            tuple(
                _validate_offset(offset, "cue offsets") for offset in self.cue_offsets
            ),
        )
        conflict_set = set(self.conflicting_classes)
        if any(value not in TEMPORAL_CLASSES for value in conflict_set):
            raise ValueError("unsupported conflicting SDOH temporal class")
        conflicts = tuple(value for value in TEMPORAL_CLASSES if value in conflict_set)
        object.__setattr__(self, "conflicting_classes", conflicts)
        if type(self.review_required) is not bool:
            raise TypeError("review_required must be a boolean")
        if conflicts and not self.review_required:
            raise ValueError("conflicting temporal cues require human review")

    @property
    def source_span(self) -> SpanOffset:
        """Return the original SDOH evidence offsets."""

        return self.source_offsets

    @property
    def source_offset(self) -> SpanOffset:
        """Return the original SDOH evidence offsets."""

        return self.source_offsets

    @property
    def qualifier(self) -> str:
        """Return the controlled temporal qualifier label."""

        return self.temporal_class

    @property
    def temporality(self) -> str:
        """Return the controlled temporal qualifier label."""

        return self.temporal_class

    @property
    def has_conflict(self) -> bool:
        """Return whether unresolved temporal classes were observed."""

        return bool(self.conflicting_classes)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic metadata without source or finding values."""

        return {
            "schema_version": SDOH_TEMPORAL_SCHEMA_VERSION,
            "source_offsets": {
                "start": self.source_offsets[0],
                "end": self.source_offsets[1],
            },
            "temporal_class": self.temporal_class,
            "cue_offsets": [list(offset) for offset in self.cue_offsets],
            "conflicting_classes": list(self.conflicting_classes),
            "review_required": self.review_required,
        }

    def to_json(self) -> str:
        """Return compact deterministic JSON containing metadata only."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SDOHTemporalEvidence":
        """Rebuild a value-free record from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise TypeError("temporal evidence payload must be a mapping")
        if payload.get("schema_version") != SDOH_TEMPORAL_SCHEMA_VERSION:
            raise ValueError("unsupported SDOH temporal schema version")
        source_payload = payload.get("source_offsets")
        if not isinstance(source_payload, Mapping):
            raise TypeError("temporal evidence source offsets are required")
        cue_offsets = payload.get("cue_offsets", ())
        conflicts = payload.get("conflicting_classes", ())
        if not isinstance(cue_offsets, Sequence) or isinstance(
            cue_offsets, str | bytes
        ):
            raise TypeError("temporal evidence cue offsets are invalid")
        if not isinstance(conflicts, Sequence) or isinstance(conflicts, str | bytes):
            raise TypeError("temporal evidence conflicts are invalid")
        temporal_class = payload.get("temporal_class")
        if not isinstance(temporal_class, str):
            raise TypeError("temporal evidence class is invalid")
        review_required = payload.get("review_required")
        if type(review_required) is not bool:
            raise TypeError("temporal evidence review flag is invalid")
        if any(not isinstance(value, str) for value in conflicts):
            raise TypeError("temporal evidence conflicts are invalid")
        return cls(
            source_offsets=_validate_offset(
                (source_payload.get("start"), source_payload.get("end")),
                "temporal evidence source offsets",
            ),
            temporal_class=temporal_class,
            cue_offsets=tuple(
                _validate_offset(offset, "temporal evidence cue offsets")
                for offset in cue_offsets
            ),
            conflicting_classes=tuple(conflicts),
            review_required=review_required,
        )


# Descriptive compatibility aliases keep the type discoverable under the two
# terms used by the issue: a qualifier is the value-free evidence record.
SDOHTemporalQualifier = SDOHTemporalEvidence
TemporalQualifier = SDOHTemporalEvidence


def qualify_sdoh_evidence(
    text: str,
    evidence: Iterable[Any] | Any,
) -> list[SDOHTemporalEvidence]:
    """Attach deterministic temporal classes to SDOH evidence spans.

    Args:
        text: Source document text. It is used transiently for local cue
            matching and is never copied into a returned record.
        evidence: An iterable of mappings/objects exposing ``start`` and
            ``end``, ``source_offsets``, or ``span``. A single such item or a
            single two-item offset sequence is also accepted. Existing
            ``SDOHFinding`` instances can be passed directly through their
            ``span`` attribute.

    Returns:
        Value-free records sorted by source offset. No explicit temporal cue
        produces ``unknown`` and requires review. Multiple distinct temporal
        classes in one local scope produce ``unknown`` with all conflicting
        classes and require review.

    Raises:
        TypeError: If the source text, evidence collection, or offsets are
            malformed.
        ValueError: If an offset is empty or outside the source text.
    """

    if not isinstance(text, str):
        raise TypeError("SDOH temporal source text must be a string")

    items = _evidence_items(evidence)
    records: list[tuple[int, SDOHTemporalEvidence]] = []
    for index, item in enumerate(items):
        source_offsets = _offset_from_item(item, len(text))
        records.append(
            (
                index,
                _qualify_one(text, source_offsets),
            )
        )

    records.sort(key=lambda item: (item[1].source_offsets, item[0]))
    return [record for _, record in records]


def qualify_sdoh_temporality(
    text: str,
    evidence: Iterable[Any] | Any,
) -> list[SDOHTemporalEvidence]:
    """Alias for :func:`qualify_sdoh_evidence` using the issue terminology."""

    return qualify_sdoh_evidence(text, evidence)


def qualify_sdoh_findings(
    text: str,
    findings: Iterable[Any] | Any,
) -> list[SDOHTemporalEvidence]:
    """Qualify existing SDOH findings without copying their values."""

    return qualify_sdoh_evidence(text, findings)


def attach_temporal_qualifiers(
    text: str,
    evidence: Iterable[Any] | Any,
) -> list[SDOHTemporalEvidence]:
    """Attach value-free temporal metadata to SDOH evidence spans."""

    return qualify_sdoh_evidence(text, evidence)


def _qualify_one(text: str, source_offsets: SpanOffset) -> SDOHTemporalEvidence:
    start, end = source_offsets
    cues = _cues_for_target(text, start, end)
    observed = {cue.temporal_class for cue in cues}
    ordered_observed = tuple(
        temporal_class
        for temporal_class in TEMPORAL_CLASSES
        if temporal_class in observed
    )

    if len(ordered_observed) == 1:
        temporal_class = ordered_observed[0]
        conflicts: tuple[str, ...] = ()
    elif not ordered_observed:
        temporal_class = UNKNOWN
        conflicts = ()
    else:
        temporal_class = UNKNOWN
        conflicts = ordered_observed

    return SDOHTemporalEvidence(
        source_offsets=source_offsets,
        temporal_class=temporal_class,
        cue_offsets=tuple((cue.start, cue.end) for cue in cues),
        conflicting_classes=conflicts,
        review_required=temporal_class == UNKNOWN or bool(conflicts),
    )


def _cues_for_target(
    text: str,
    target_start: int,
    target_end: int,
) -> tuple[_TemporalCue, ...]:
    scope_start, scope_end = _sentence_bounds(text, target_start, target_end)
    matched: dict[tuple[str, int, int], _TemporalCue] = {}

    for cue_pattern, pattern in _COMPILED_CUE_PATTERNS:
        for match in pattern.finditer(text, scope_start, scope_end):
            cue_start, cue_end = match.span()
            if not _cue_reaches_target(
                text,
                cue_pattern,
                cue_start,
                cue_end,
                target_start,
                target_end,
            ):
                continue
            key = (cue_pattern.temporal_class, cue_start, cue_end)
            matched[key] = _TemporalCue(
                temporal_class=cue_pattern.temporal_class,
                start=cue_start,
                end=cue_end,
                applies_after_target=cue_pattern.applies_after_target,
            )

    class_order = {value: index for index, value in enumerate(TEMPORAL_CLASSES)}
    return tuple(
        sorted(
            matched.values(),
            key=lambda cue: (cue.start, cue.end, class_order[cue.temporal_class]),
        )
    )


def _cue_reaches_target(
    text: str,
    cue_pattern: _CuePattern,
    cue_start: int,
    cue_end: int,
    target_start: int,
    target_end: int,
) -> bool:
    if target_start <= cue_start and cue_end <= target_end:
        return True

    if cue_end <= target_start:
        between = text[cue_end:target_start]
        return _HARD_CLAUSE_BOUNDARY_RE.search(between) is None

    if target_end <= cue_start and cue_pattern.applies_after_target:
        between = text[target_end:cue_start]
        return _HARD_CLAUSE_BOUNDARY_RE.search(between) is None

    return False


def _sentence_bounds(text: str, start: int, end: int) -> SpanOffset:
    left_candidates = [
        text.rfind(boundary, 0, start) for boundary in _SENTENCE_BOUNDARIES
    ]
    right_candidates = [text.find(boundary, end) for boundary in _SENTENCE_BOUNDARIES]
    right_candidates = [candidate for candidate in right_candidates if candidate >= 0]
    return (
        max(left_candidates, default=-1) + 1,
        min(right_candidates, default=len(text)),
    )


def _evidence_items(evidence: Iterable[Any] | Any) -> tuple[Any, ...]:
    if isinstance(evidence, Mapping) or _has_offset_field(evidence):
        return (evidence,)
    if _is_offset_sequence(evidence):
        return (evidence,)
    if isinstance(evidence, str | bytes | bytearray):
        raise TypeError("SDOH temporal evidence must contain source offsets")
    try:
        return tuple(evidence)
    except TypeError:
        raise TypeError("SDOH temporal evidence must be iterable") from None


def _has_offset_field(value: Any) -> bool:
    return any(
        hasattr(value, key)
        for key in (
            "span",
            "source_offsets",
            "source_offset",
            "source_start",
            "start",
        )
    )


def _is_offset_sequence(value: Any) -> bool:
    return (
        isinstance(value, Sequence)
        and not isinstance(value, str | bytes | bytearray)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _offset_from_item(item: Any, text_length: int) -> SpanOffset:
    candidate = item
    if isinstance(item, Mapping):
        for key in ("source_offsets", "source_offset", "span", "offsets"):
            if key in item:
                candidate = item[key]
                break
        else:
            candidate = item
    else:
        for key in ("source_offsets", "source_offset", "span", "offsets"):
            value = getattr(item, key, None)
            if value is not None:
                candidate = value
                break

    if isinstance(candidate, Mapping):
        start = candidate.get("start")
        end = candidate.get("end")
        if start is None or end is None:
            start = candidate.get("source_start")
            end = candidate.get("source_end")
    elif _is_offset_sequence(candidate):
        start, end = candidate
    else:
        start = _field(item, ("source_start", "start", "start_char", "begin"))
        end = _field(item, ("source_end", "end", "end_char", "stop"))

    return _validate_offset((start, end), "SDOH evidence offsets", text_length)


def _field(item: Any, keys: Sequence[str]) -> Any:
    if isinstance(item, Mapping):
        for key in keys:
            if key in item:
                return item[key]
        return None
    for key in keys:
        value = getattr(item, key, None)
        if value is not None:
            return value
    return None


def _validate_offset(
    value: Any,
    field_name: str,
    text_length: int | None = None,
) -> SpanOffset:
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
        raise TypeError(f"{field_name} must contain integer offsets")
    if start < 0 or end <= start:
        raise ValueError(f"{field_name} must form a non-empty half-open span")
    if text_length is not None and end > text_length:
        raise ValueError(f"{field_name} must be within the source text")
    return start, end


__all__ = [
    "CURRENT",
    "FUTURE",
    "HISTORICAL",
    "SDOH_TEMPORAL_SCHEMA_VERSION",
    "SDOHTemporalEvidence",
    "SDOHTemporalQualifier",
    "TEMPORAL_CLASSES",
    "TemporalQualifier",
    "UNKNOWN",
    "attach_temporal_qualifiers",
    "qualify_sdoh_evidence",
    "qualify_sdoh_findings",
    "qualify_sdoh_temporality",
]
