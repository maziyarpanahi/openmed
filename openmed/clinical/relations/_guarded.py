"""Shared privacy-safe primitives for guarded clinical relation candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from openmed.clinical.context import assert_context_axes
from openmed.clinical.sections import detect_sections, validate_section_spans
from openmed.core.audit import hash_text, stable_hash
from openmed.core.labels import normalize_label
from openmed.processing.advanced_ner import EntitySpan

from .candidate import split_sentence_offsets


@dataclass(frozen=True)
class GuardedEvidenceSpan:
    """Offset-and-hash evidence that never serializes source surface text."""

    label: str
    start: int
    end: int
    text_hash: str
    section: str
    score: float = 1.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("evidence label must be non-empty")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("evidence offsets must satisfy 0 <= start < end")
        if not self.text_hash.startswith("sha256:"):
            raise ValueError("evidence text_hash must use SHA-256")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("evidence score must be between 0 and 1")

    @property
    def offsets(self) -> tuple[int, int]:
        """Return the half-open source offsets."""

        return self.start, self.end

    def to_dict(self) -> dict[str, Any]:
        """Return offset-only, deterministic evidence metadata."""

        return {
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "text_hash": self.text_hash,
            "section": self.section,
            "score": self.score,
        }


@dataclass(frozen=True)
class GuardedAssertion:
    """Controlled assertion axes attached to a review candidate."""

    negation: str
    certainty: str
    temporality: str
    experiencer: str

    @property
    def is_asserted(self) -> bool:
        """Return whether all axes describe an affirmed current patient mention."""

        return (
            self.negation == "affirmed"
            and self.certainty == "certain"
            and self.temporality == "recent"
            and self.experiencer == "patient"
        )

    def to_dict(self) -> dict[str, str]:
        """Return the controlled assertion mapping."""

        return {
            "negation": self.negation,
            "certainty": self.certainty,
            "temporality": self.temporality,
            "experiencer": self.experiencer,
        }


@dataclass(frozen=True)
class GuardedSpanInput:
    """Internal evidence plus caller metadata used while linking candidates."""

    evidence: GuardedEvidenceSpan
    data: Mapping[str, Any]


def coerce_guarded_spans(
    text: str,
    spans: Iterable[Any],
    sections: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[tuple[GuardedSpanInput, ...], tuple[Mapping[str, Any], ...]]:
    """Validate inputs and return deterministic privacy-safe span evidence."""

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    section_items = tuple(detect_sections(text) if sections is None else sections)
    validate_section_spans(text, section_items)
    normalized: dict[tuple[int, int, str], GuardedSpanInput] = {}
    for item in spans:
        data = _span_mapping(item)
        if data is None:
            continue
        try:
            start = int(data.get("start", data.get("start_char", -1)))
            end = int(data.get("end", data.get("end_char", -1)))
            score = float(data.get("score", 1.0))
        except (TypeError, ValueError):
            continue
        label = _span_label(data)
        if not label or start < 0 or end <= start or end > len(text):
            continue
        section = _section_label(start, end, data, section_items)
        normalized_label = normalize_label(label)
        if normalized_label == "OTHER" and str(label).strip().upper() != "OTHER":
            normalized_label = str(label).strip().upper()
        evidence = GuardedEvidenceSpan(
            label=normalized_label,
            start=start,
            end=end,
            text_hash=hash_text(text[start:end]),
            section=section,
            score=max(0.0, min(score, 1.0)),
        )
        normalized[(start, end, evidence.label)] = GuardedSpanInput(
            evidence=evidence,
            data=dict(data),
        )
    ordered = tuple(
        normalized[key]
        for key in sorted(normalized, key=lambda value: (value[0], value[1], value[2]))
    )
    return ordered, section_items


def assertion_for_span(text: str, item: GuardedSpanInput) -> GuardedAssertion:
    """Resolve caller-supplied or local deterministic assertion axes."""

    explicit = _assertion_mapping(item.data)
    target = {
        "document_text": text,
        "start": item.evidence.start,
        "end": item.evidence.end,
        "text": text[item.evidence.start : item.evidence.end],
    }
    inferred = assert_context_axes(target, section=item.evidence.section)
    negation = _controlled_axis(explicit, "negation", inferred.negation or "affirmed")
    certainty = _controlled_axis(
        explicit,
        "certainty",
        _certainty_from_uncertainty(explicit.get("uncertainty"), inferred.certainty),
    )
    temporality = _controlled_axis(
        explicit,
        "temporality",
        inferred.temporality,
    )
    experiencer = _controlled_axis(
        explicit,
        "experiencer",
        inferred.experiencer or "patient",
    )
    return GuardedAssertion(
        negation=negation,
        certainty=certainty,
        temporality=temporality,
        experiencer=experiencer,
    )


def evidence_from_offsets(
    text: str,
    *,
    label: str,
    start: int,
    end: int,
    section: str,
    score: float = 1.0,
) -> GuardedEvidenceSpan:
    """Build safe evidence for a lexically detected cue."""

    if start < 0 or end <= start or end > len(text):
        raise ValueError("detected evidence offsets are outside the source text")
    return GuardedEvidenceSpan(
        label=label,
        start=start,
        end=end,
        text_hash=hash_text(text[start:end]),
        section=section,
        score=max(0.0, min(float(score), 1.0)),
    )


def span_gap(left: GuardedEvidenceSpan, right: GuardedEvidenceSpan) -> int:
    """Return the character gap between two half-open spans."""

    if left.end <= right.start:
        return right.start - left.end
    if right.end <= left.start:
        return left.start - right.end
    return 0


def same_sentence(
    text: str, left: GuardedEvidenceSpan, right: GuardedEvidenceSpan
) -> bool:
    """Return whether both spans are contained by one deterministic sentence."""

    for start, end in split_sentence_offsets(text):
        if (
            start <= left.start
            and left.end <= end
            and start <= right.start
            and right.end <= end
        ):
            return True
    return False


def evidence_window(
    text: str, left: GuardedEvidenceSpan, right: GuardedEvidenceSpan
) -> tuple[int, str]:
    """Return the start offset and source slice spanning both endpoints."""

    start = min(left.start, right.start)
    end = max(left.end, right.end)
    return start, text[start:end]


def stable_candidate_id(kind: str, *parts: object) -> str:
    """Return an opaque deterministic candidate identifier."""

    return f"{kind}:{stable_hash((kind, *parts)).removeprefix('sha256:')[:24]}"


def nested_value(data: Mapping[str, Any], *keys: str) -> Any:
    """Return the first matching top-level or metadata value."""

    metadata = data.get("metadata")
    containers = (data, metadata) if isinstance(metadata, Mapping) else (data,)
    for container in containers:
        for key in keys:
            if key in container and container[key] is not None:
                return container[key]
    return None


def _span_mapping(item: Any) -> Mapping[str, Any] | None:
    if isinstance(item, Mapping):
        return item
    if isinstance(item, EntitySpan):
        return item.to_dict()
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        value = to_dict()
        if isinstance(value, Mapping):
            return value
    try:
        value = vars(item)
    except TypeError:
        return None
    return value if isinstance(value, Mapping) else None


def _span_label(data: Mapping[str, Any]) -> str:
    for key in ("label", "entity", "entity_type", "canonical_label", "role", "type"):
        value = data.get(key)
        if value:
            return str(value)
    return ""


def _section_label(
    start: int,
    end: int,
    data: Mapping[str, Any],
    sections: Sequence[Mapping[str, Any]],
) -> str:
    explicit = nested_value(data, "section", "section_label")
    if explicit is not None:
        return str(explicit)
    for section in sections:
        if int(section["start"]) <= start and end <= int(section["end"]):
            return str(section.get("label", "unsectioned"))
    return "unsectioned"


def _assertion_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    containers: list[Mapping[str, Any]] = [data]
    metadata = data.get("metadata")
    if isinstance(metadata, Mapping):
        containers.append(metadata)
    for container in tuple(containers):
        for key in ("assertion", "clinical_assertion", "clinical_context", "context"):
            nested = container.get(key)
            if isinstance(nested, Mapping):
                containers.append(nested)
    for container in containers:
        for key in (
            "negation",
            "certainty",
            "uncertainty",
            "temporality",
            "experiencer",
        ):
            if key in container and container[key] is not None:
                result[key] = container[key]
    return result


def _controlled_axis(data: Mapping[str, Any], key: str, default: str) -> str:
    value = data.get(key)
    if value is None:
        return str(default)
    normalized = str(value).strip().casefold().replace(" ", "_")
    aliases = {
        "negation": {
            "affirmed": "affirmed",
            "present": "affirmed",
            "negated": "negated",
            "refuted": "negated",
        },
        "certainty": {
            "certain": "certain",
            "confirmed": "certain",
            "possible": "uncertain",
            "probable": "uncertain",
            "uncertain": "uncertain",
        },
        "temporality": {
            "current": "recent",
            "historical": "historical",
            "hypothetical": "hypothetical",
            "recent": "recent",
        },
        "experiencer": {"family": "family", "other": "other"},
    }
    if key == "experiencer" and normalized == "patient":
        return "patient"
    return aliases[key].get(normalized, str(default))


def _certainty_from_uncertainty(value: Any, default: str) -> str:
    if value is None:
        return default
    normalized = str(value).strip().casefold()
    if normalized in {"uncertain", "possible", "probable"}:
        return "uncertain"
    if normalized in {"certain", "affirmed"}:
        return "certain"
    return default


__all__ = [
    "GuardedAssertion",
    "GuardedEvidenceSpan",
]
