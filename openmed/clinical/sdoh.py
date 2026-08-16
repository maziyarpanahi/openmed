"""SHAC-aligned SDOH finding schema and determinant dispatcher.

The employment and living-status extractors use a compact OpenMed-maintained
cue table containing only synthetic/public phrases. Food insecurity is an
OpenMed extension beyond the five core SHAC determinant categories. The real
Social History Annotated Corpus (SHAC) is DUA-gated, eval-only, and must never
be bundled with OpenMed or loaded by this runtime module.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from .context import resolve_temporality
from .status_vocab import normalize_employment_status, normalize_living_status

SOCIAL_HISTORY_SECTION = "social_history"
SHAC_DATA_POLICY = (
    "Real SHAC data is DUA-gated and eval-only; runtime extraction uses only "
    "synthetic or public data."
)
SDOH_SOCIAL_CUES_RESOURCE = "data/sdoh_social_cues.yaml"
FOOD_INSECURITY_EXTENSION_NOTE = (
    "Food insecurity is an OpenMed extension beyond the five core SHAC "
    "determinant categories."
)

_SOCIAL_CUES_PACKAGE = "openmed.clinical"
_CLAUSE_RE = re.compile(r"[^.;!?\n]+")

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


@dataclass(frozen=True)
class _CueMatch:
    start: int
    end: int
    value: str


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


def load_sdoh_social_cues(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the unrestricted social-determinant cue table.

    Args:
        path: Optional replacement YAML path, primarily for downstream
            validation. The packaged OpenMed cue table is used when omitted.

    Returns:
        A detached copy of the validated cue-table payload.
    """

    if path is None:
        return copy.deepcopy(_load_default_sdoh_social_cues())
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return _validate_sdoh_social_cues(payload)


def extract_employment_findings(
    text: str,
    spans: Sequence[Any] = (),
) -> list[SDOHFinding]:
    """Extract deterministic employment status and occupation findings.

    Args:
        text: Caller-selected clinical text to scan.
        spans: Upstream candidates accepted for registry compatibility.

    Returns:
        Employment findings anchored to their source cue spans.

    ``spans`` is accepted for registry compatibility. These compact cue-based
    extractors scan caller-selected text directly; the dispatcher still applies
    candidate and Social History section boundaries to its inputs and outputs.
    """

    _validate_extractor_text(text)
    _ = spans
    config = _determinant_config("employment")
    findings: list[SDOHFinding] = []
    for clause_start, clause_end in _clause_offsets(text):
        clause = text[clause_start:clause_end]
        status_match = _status_match(clause, config)
        type_match = _typed_cue_match(clause, config["types"])
        if status_match is None and type_match is None:
            continue

        if status_match is None:
            status_match = _CueMatch(
                start=type_match.start,
                end=type_match.end,
                value="employed",
            )
        match_start, match_end = _combined_offset(status_match, type_match)
        absolute_start = clause_start + match_start
        absolute_end = clause_start + match_end
        temporality = _finding_temporality(text, absolute_start, absolute_end)
        status = normalize_employment_status(
            clause,
            temporality=temporality,
        )
        if status == "unknown":
            status = status_match.value
        findings.append(
            SDOHFinding(
                category=config["category"],
                value=type_match.value if type_match else status_match.value,
                status=status,
                extent=None,
                temporality=temporality,
                span=(absolute_start, absolute_end),
                score=config["score"],
            )
        )
    return findings


def extract_living_status_findings(
    text: str,
    spans: Sequence[Any] = (),
) -> list[SDOHFinding]:
    """Extract deterministic housing and living-situation findings.

    Args:
        text: Caller-selected clinical text to scan.
        spans: Upstream candidates accepted for registry compatibility.

    Returns:
        Living-status findings anchored to their source cue spans.
    """

    _validate_extractor_text(text)
    _ = spans
    config = _determinant_config("living_status")
    findings: list[SDOHFinding] = []
    for clause_start, clause_end in _clause_offsets(text):
        clause = text[clause_start:clause_end]
        status_match = _status_match(clause, config)
        if status_match is None:
            continue

        absolute_start = clause_start + status_match.start
        absolute_end = clause_start + status_match.end
        temporality = _finding_temporality(text, absolute_start, absolute_end)
        status = normalize_living_status(clause, temporality=temporality)
        if status == "unknown":
            status = status_match.value
        findings.append(
            SDOHFinding(
                category=config["category"],
                value=status_match.value,
                status=status,
                extent=None,
                temporality=temporality,
                span=(absolute_start, absolute_end),
                score=config["score"],
            )
        )
    return findings


def extract_food_insecurity_findings(
    text: str,
    spans: Sequence[Any] = (),
) -> list[SDOHFinding]:
    """Extract food-insecurity cues as an extension beyond core SHAC.

    Args:
        text: Caller-selected clinical text to scan.
        spans: Upstream candidates accepted for registry compatibility.

    Returns:
        Food-insecurity findings anchored to their source cue spans.
    """

    _validate_extractor_text(text)
    _ = spans
    config = _determinant_config("food_insecurity")
    findings: list[SDOHFinding] = []
    for clause_start, clause_end in _clause_offsets(text):
        cue_match = _cue_match(text[clause_start:clause_end], config["cues"])
        if cue_match is None:
            continue

        absolute_start = clause_start + cue_match.start
        absolute_end = clause_start + cue_match.end
        findings.append(
            SDOHFinding(
                category=config["category"],
                value=config["value"],
                status=config["status"],
                extent=None,
                temporality=_finding_temporality(
                    text,
                    absolute_start,
                    absolute_end,
                ),
                span=(absolute_start, absolute_end),
                score=config["score"],
            )
        )
    return findings


@lru_cache(maxsize=1)
def _load_default_sdoh_social_cues() -> dict[str, Any]:
    resource = resources.files(_SOCIAL_CUES_PACKAGE).joinpath(SDOH_SOCIAL_CUES_RESOURCE)
    payload = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return _validate_sdoh_social_cues(payload)


def _validate_sdoh_social_cues(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("SDOH social cue table requires schema_version 1")

    provenance = payload.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or not provenance.get("source")
        or provenance.get("restricted_data") is not False
    ):
        raise ValueError("SDOH social cues require unrestricted provenance")

    determinants = payload.get("determinants")
    if not isinstance(determinants, Mapping):
        raise ValueError("SDOH social cues require a determinants mapping")

    employment = _validate_status_determinant(determinants, "employment")
    types = employment.get("types")
    if not isinstance(types, Mapping) or not types:
        raise ValueError("employment social cues require occupation types")
    _validate_cue_mapping(types, "employment.types")

    _validate_status_determinant(determinants, "living_status")

    food = determinants.get("food_insecurity")
    if not isinstance(food, Mapping):
        raise ValueError("food_insecurity social cues must be a mapping")
    _validate_determinant_identity(food, "food_insecurity")
    _validate_cue_sequence(food.get("cues"), "food_insecurity.cues")
    if food.get("status") != "current" or food.get("value") != "food_insecure":
        raise ValueError("food_insecurity social cues require canonical values")
    if food.get("extension_beyond_core_shac") is not True:
        raise ValueError("food_insecurity must be marked as a SHAC extension")
    extension_note = food.get("extension_note")
    if not isinstance(extension_note, str) or "beyond the five core SHAC" not in (
        extension_note
    ):
        raise ValueError("food_insecurity requires its SHAC extension note")
    return payload


def _validate_status_determinant(
    determinants: Mapping[str, Any],
    determinant: str,
) -> Mapping[str, Any]:
    config = determinants.get(determinant)
    if not isinstance(config, Mapping):
        raise ValueError(f"{determinant} social cues must be a mapping")
    _validate_determinant_identity(config, determinant)

    priority = config.get("status_priority")
    status_cues = config.get("status_cues")
    _validate_cue_sequence(priority, f"{determinant}.status_priority")
    if not isinstance(status_cues, Mapping) or not status_cues:
        raise ValueError(f"{determinant}.status_cues must be a mapping")
    _validate_cue_mapping(status_cues, f"{determinant}.status_cues")
    if set(priority) != set(status_cues):
        raise ValueError(f"{determinant} status priority must cover every status")
    return config


def _validate_determinant_identity(
    config: Mapping[str, Any],
    determinant: str,
) -> None:
    if config.get("category") != determinant:
        raise ValueError(f"{determinant} requires a matching category")
    score = config.get("score")
    if (
        isinstance(score, bool)
        or not isinstance(score, int | float)
        or not math.isfinite(score)
        or not 0.0 <= score <= 1.0
    ):
        raise ValueError(f"{determinant} requires a score between 0.0 and 1.0")


def _validate_cue_mapping(value: Mapping[Any, Any], field_name: str) -> None:
    for key, cues in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{field_name} requires non-empty string keys")
        _validate_cue_sequence(cues, f"{field_name}.{key}")


def _validate_cue_sequence(value: Any, field_name: str) -> None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise ValueError(f"{field_name} requires non-empty string cues")


def _determinant_config(determinant: str) -> Mapping[str, Any]:
    return _load_default_sdoh_social_cues()["determinants"][determinant]


def _clause_offsets(text: str) -> Iterable[SpanOffset]:
    for match in _CLAUSE_RE.finditer(text):
        segment = match.group()
        leading_space = len(segment) - len(segment.lstrip())
        trailing_space = len(segment) - len(segment.rstrip())
        start = match.start() + leading_space
        end = match.end() - trailing_space
        if start < end:
            yield start, end


def _status_match(clause: str, config: Mapping[str, Any]) -> _CueMatch | None:
    status_cues = config["status_cues"]
    for status in config["status_priority"]:
        match = _cue_match(clause, status_cues[status])
        if match is not None:
            return _CueMatch(match.start, match.end, status)
    return None


def _typed_cue_match(
    clause: str,
    cue_mapping: Mapping[str, Sequence[str]],
) -> _CueMatch | None:
    matches: list[_CueMatch] = []
    for value, cues in cue_mapping.items():
        match = _cue_match(clause, cues)
        if match is not None:
            matches.append(_CueMatch(match.start, match.end, value))
    return (
        min(matches, key=lambda item: (item.start, -(item.end - item.start)))
        if matches
        else None
    )


def _cue_match(text: str, cues: Sequence[str]) -> _CueMatch | None:
    matches: list[_CueMatch] = []
    for cue in sorted(cues, key=len, reverse=True):
        match = _cue_pattern(cue).search(text)
        if match is not None:
            matches.append(_CueMatch(match.start(), match.end(), cue))
    return (
        min(matches, key=lambda item: (item.start, -(item.end - item.start)))
        if matches
        else None
    )


@lru_cache(maxsize=512)
def _cue_pattern(cue: str) -> re.Pattern[str]:
    escaped = re.escape(" ".join(cue.split())).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)


def _combined_offset(
    required: _CueMatch,
    optional: _CueMatch | None,
) -> SpanOffset:
    if optional is None:
        return required.start, required.end
    return min(required.start, optional.start), max(required.end, optional.end)


def _finding_temporality(text: str, start: int, end: int) -> str | None:
    try:
        return resolve_temporality(
            {
                "text": text[start:end],
                "context": text,
                "start": start,
                "end": end,
            }
        )
    except (TypeError, ValueError):
        return None


def _validate_extractor_text(text: object) -> None:
    if not isinstance(text, str):
        raise TypeError("text must be a string")


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


register_determinant_extractor("employment", extract_employment_findings)
register_determinant_extractor("food_insecurity", extract_food_insecurity_findings)
register_determinant_extractor("living_status", extract_living_status_findings)


__all__ = [
    "FOOD_INSECURITY_EXTENSION_NOTE",
    "SHAC_DATA_POLICY",
    "SDOH_SOCIAL_CUES_RESOURCE",
    "SOCIAL_HISTORY_SECTION",
    "DeterminantExtractor",
    "DeterminantExtractorRegistry",
    "SDOHFinding",
    "available_determinant_extractors",
    "extract_employment_findings",
    "extract_food_insecurity_findings",
    "extract_living_status_findings",
    "extract_sdoh",
    "load_sdoh_social_cues",
    "register_determinant_extractor",
    "unregister_determinant_extractor",
]
