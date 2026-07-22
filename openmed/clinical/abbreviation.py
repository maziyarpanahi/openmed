"""Deterministic clinical abbreviation sense disambiguation.

The bundled sense inventory is deliberately small, synthetic, and permissively
licensed.  It is not derived from UMLS LRABR or another restricted vocabulary.
Callers can extend or override it with a local JSON inventory while keeping the
resolver offline and deterministic.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

DEFAULT_SENSE_INVENTORY_RESOURCE = "data/abbreviation_senses_starter.json"
ABBREVIATION_DISAMBIGUATION_ADVISORY = (
    "Abbreviation expansion is deterministic assistive support based on a small "
    "sense inventory and local context. Ambiguous or low-evidence resolutions "
    "require review before clinical use."
)

_INVENTORY_PACKAGE = "openmed.clinical"
_ABBREVIATION_LABELS = {"abbreviation", "acronym", "short_form", "shortform"}
_LABEL_KEYS = ("label", "entity_type", "canonical_label", "semantic_type", "type")
_SECTION_KEYS = ("section", "section_label", "section_name")
_DIRECT_ENTITY_TYPE_KEYS = (
    "cooccurring_entity_types",
    "context_entity_types",
    "entity_types",
)


@dataclass(frozen=True)
class SenseDefinition:
    """One candidate meaning stored in a :class:`SenseInventory`."""

    long_form: str
    semantic_type: str
    source: str
    sections: tuple[str, ...] = ()
    entity_types: tuple[str, ...] = ()
    cue_words: tuple[str, ...] = ()
    prior: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this definition."""

        return {
            "long_form": self.long_form,
            "semantic_type": self.semantic_type,
            "source": self.source,
            "sections": list(self.sections),
            "entity_types": list(self.entity_types),
            "cue_words": list(self.cue_words),
            "prior": self.prior,
        }


@dataclass(frozen=True)
class SenseAlternative:
    """A non-winning candidate retained for review and downstream routing."""

    long_form: str
    semantic_type: str
    source: str
    score: float
    matched_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "long_form": self.long_form,
            "semantic_type": self.semantic_type,
            "source": self.source,
            "score": self.score,
            "matched_features": list(self.matched_features),
        }


@dataclass(frozen=True)
class AbbreviationSense:
    """Resolved meaning for one clinical abbreviation occurrence."""

    short_form: str
    long_form: str
    semantic_type: str
    source: str
    score: float
    alternatives: tuple[SenseAlternative, ...] = ()
    matched_features: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "short_form": self.short_form,
            "long_form": self.long_form,
            "semantic_type": self.semantic_type,
            "source": self.source,
            "score": self.score,
            "alternatives": [item.to_dict() for item in self.alternatives],
            "matched_features": list(self.matched_features),
        }


@dataclass(frozen=True)
class AbbreviationAnnotation:
    """A short-form span annotated with its resolved sense, if available."""

    start: int
    end: int
    short_form: str
    sense: AbbreviationSense | None
    section: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible representation."""

        return {
            "start": self.start,
            "end": self.end,
            "short_form": self.short_form,
            "section": self.section,
            "sense": self.sense.to_dict() if self.sense is not None else None,
        }


class SenseInventory(Mapping[str, tuple[SenseDefinition, ...]]):
    """Read-only mapping from normalized short form to candidate senses."""

    def __init__(self, senses: Mapping[str, Sequence[SenseDefinition]]) -> None:
        normalized: dict[str, tuple[SenseDefinition, ...]] = {}
        for short_form, definitions in senses.items():
            key = _normalize_short_form(short_form)
            if not key:
                raise ValueError("sense inventory short forms must be non-empty")
            candidates = tuple(definitions)
            if not candidates:
                raise ValueError(f"sense inventory entry {key!r} has no candidates")
            if not all(isinstance(item, SenseDefinition) for item in candidates):
                raise TypeError(
                    f"sense inventory entry {key!r} must contain SenseDefinition values"
                )
            normalized[key] = candidates
        self._senses = normalized

    def __getitem__(self, short_form: str) -> tuple[SenseDefinition, ...]:
        return self._senses[_normalize_short_form(short_form)]

    def __iter__(self) -> Iterator[str]:
        return iter(self._senses)

    def __len__(self) -> int:
        return len(self._senses)

    def candidates(self, short_form: object) -> tuple[SenseDefinition, ...]:
        """Return candidates for ``short_form`` or an empty tuple when unknown."""

        return self._senses.get(_normalize_short_form(short_form), ())

    def to_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Return the registry as a deterministic JSON-compatible mapping."""

        return {
            short_form: [candidate.to_dict() for candidate in candidates]
            for short_form, candidates in self._senses.items()
        }


class AbbreviationDisambiguator:
    """Score abbreviation candidates from section, entity, and cue evidence."""

    def __init__(self, inventory: SenseInventory | None = None) -> None:
        self.inventory = inventory or load_sense_inventory()

    def disambiguate(
        self,
        short_form: object,
        context: object = "",
        *,
        section: str | None = None,
        entity_types: Iterable[object] = (),
    ) -> AbbreviationSense | None:
        """Resolve one short form against local, non-network context.

        Args:
            short_form: Abbreviation surface form, matched case-insensitively.
            context: Local sentence or character window around the occurrence.
            section: Optional clinical section name for the occurrence.
            entity_types: Semantic types of nearby entity spans.

        Returns:
            The top-scoring :class:`AbbreviationSense` with ranked alternatives,
            or ``None`` when ``short_form`` is absent from the inventory.
        """

        normalized_short_form = _normalize_short_form(short_form)
        candidates = self.inventory.candidates(normalized_short_form)
        if not candidates:
            return None

        normalized_context = _normalize_text(context)
        normalized_section = _normalize_label(section)
        normalized_entity_types = {
            normalized
            for value in entity_types
            if (normalized := _normalize_label(value))
        }
        scored = [
            _score_candidate(
                candidate,
                index=index,
                context=normalized_context,
                section=normalized_section,
                entity_types=normalized_entity_types,
            )
            for index, candidate in enumerate(candidates)
        ]
        scored.sort(key=lambda item: (-item.raw_score, item.index))

        total = sum(item.raw_score for item in scored)
        if total <= 0.0:
            normalized_scores = [1.0 / len(scored)] * len(scored)
        else:
            normalized_scores = [item.raw_score / total for item in scored]

        winner = scored[0]
        return AbbreviationSense(
            short_form=normalized_short_form,
            long_form=winner.definition.long_form,
            semantic_type=winner.definition.semantic_type,
            source=winner.definition.source,
            score=round(normalized_scores[0], 6),
            alternatives=tuple(
                SenseAlternative(
                    long_form=item.definition.long_form,
                    semantic_type=item.definition.semantic_type,
                    source=item.definition.source,
                    score=round(score, 6),
                    matched_features=item.matched_features,
                )
                for item, score in zip(scored[1:], normalized_scores[1:])
            ),
            matched_features=winner.matched_features,
        )


@dataclass(frozen=True)
class _ScoredCandidate:
    definition: SenseDefinition
    index: int
    raw_score: float
    matched_features: tuple[str, ...]


@dataclass(frozen=True)
class _SpanRecord:
    original: object
    start: int
    end: int
    label: str | None
    section: str | None
    direct_entity_types: tuple[str, ...]


def load_sense_inventory(
    path: str | Path | None = None,
    *,
    include_starter: bool = True,
) -> SenseInventory:
    """Load the starter inventory and optionally merge a local JSON file.

    A user entry with the same short form and long form replaces the starter
    definition. New long forms and new short forms extend the registry. Set
    ``include_starter=False`` to load only the supplied inventory.

    Args:
        path: Optional path to a user-supplied JSON sense inventory.
        include_starter: Whether to start with the bundled permissive inventory.

    Returns:
        A validated, read-only :class:`SenseInventory`.
    """

    merged: dict[str, list[SenseDefinition]] = {}
    if include_starter:
        merged = {
            short_form: list(candidates)
            for short_form, candidates in _starter_inventory().items()
        }

    if path is not None:
        custom_path = Path(path)
        with custom_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        custom = _inventory_from_payload(payload, require_permissive=False)
        _merge_inventory(merged, custom)
    elif not include_starter:
        raise ValueError("path is required when include_starter is False")

    return SenseInventory(merged)


def load_abbreviation_inventory(
    path: str | Path | None = None,
    *,
    include_starter: bool = True,
) -> SenseInventory:
    """Alias for :func:`load_sense_inventory` with an explicit domain name."""

    return load_sense_inventory(path, include_starter=include_starter)


def disambiguate_abbreviation(
    short_form: object,
    context: object = "",
    *,
    section: str | None = None,
    entity_types: Iterable[object] = (),
    inventory: SenseInventory | None = None,
) -> AbbreviationSense | None:
    """Resolve one abbreviation with the default or supplied inventory."""

    return AbbreviationDisambiguator(inventory).disambiguate(
        short_form,
        context,
        section=section,
        entity_types=entity_types,
    )


def expand_abbreviations(
    text: str,
    spans: Iterable[object],
    *,
    inventory: SenseInventory | None = None,
    section: str | None = None,
    section_by_span: Mapping[tuple[int, int], str] | None = None,
    entity_types: Iterable[object] = (),
    context_window: int = 160,
) -> tuple[AbbreviationAnnotation, ...]:
    """Annotate detected short-form spans with resolved senses.

    Span mappings or objects must expose integer ``start`` and ``end`` values.
    A span is treated as a short-form target when its source slice exists in the
    inventory, its label is ``ABBREVIATION``/``ACRONYM``, or it has no label.
    Other labelled spans inside ``context_window`` supply co-occurring entity
    types. A target span may also provide ``section`` and an ``entity_types``
    sequence directly.

    Unknown labelled abbreviations are retained with ``sense=None``. Invalid
    offsets raise ``ValueError`` instead of being silently shifted.

    Args:
        text: Original document text.
        spans: Detected abbreviation spans and optional nearby entity spans.
        inventory: Optional loaded sense inventory.
        section: Fallback section for targets without span-level section data.
        section_by_span: Optional section lookup keyed by ``(start, end)``.
        entity_types: Additional document- or caller-level semantic types.
        context_window: Characters retained on each side of a target span.

    Returns:
        Offset-ordered abbreviation annotations.
    """

    if context_window < 0:
        raise ValueError("context_window must be non-negative")

    active_inventory = inventory or load_sense_inventory()
    disambiguator = AbbreviationDisambiguator(active_inventory)
    records = tuple(_coerce_span(text, span) for span in spans)
    section_lookup = section_by_span or {}
    shared_entity_types = tuple(entity_types)
    annotations: list[AbbreviationAnnotation] = []

    for record in records:
        short_form = text[record.start : record.end]
        known = bool(active_inventory.candidates(short_form))
        if not (known or record.label in _ABBREVIATION_LABELS or record.label is None):
            continue

        context_start = max(0, record.start - context_window)
        context_end = min(len(text), record.end + context_window)
        nearby_types = set(record.direct_entity_types)
        for other in records:
            if other is record or other.label is None:
                continue
            if other.label in _ABBREVIATION_LABELS:
                continue
            if other.end >= context_start and other.start <= context_end:
                nearby_types.add(other.label)
        nearby_types.update(
            normalized
            for item in shared_entity_types
            if (normalized := _normalize_label(item))
        )

        span_section = (
            record.section or section_lookup.get((record.start, record.end)) or section
        )
        sense = disambiguator.disambiguate(
            short_form,
            text[context_start:context_end],
            section=span_section,
            entity_types=nearby_types,
        )
        annotations.append(
            AbbreviationAnnotation(
                start=record.start,
                end=record.end,
                short_form=short_form,
                sense=sense,
                section=span_section,
            )
        )

    return tuple(sorted(annotations, key=lambda item: (item.start, item.end)))


@lru_cache(maxsize=1)
def _starter_inventory() -> SenseInventory:
    resource = resources.files(_INVENTORY_PACKAGE).joinpath(
        DEFAULT_SENSE_INVENTORY_RESOURCE
    )
    with resource.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _inventory_from_payload(payload, require_permissive=True)


def _inventory_from_payload(
    payload: object,
    *,
    require_permissive: bool,
) -> SenseInventory:
    if not isinstance(payload, Mapping):
        raise ValueError("sense inventory must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("sense inventory schema_version must be 1")

    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("sense inventory requires provenance metadata")
    if not _required_text(provenance.get("source")):
        raise ValueError("sense inventory provenance.source must be non-empty")
    if require_permissive:
        if provenance.get("restricted_data") is not False:
            raise ValueError(
                "starter sense inventory must declare restricted_data=false"
            )
        if not _required_text(provenance.get("license")):
            raise ValueError("starter sense inventory must declare a license")

    raw_senses = payload.get("senses")
    if not isinstance(raw_senses, Mapping) or not raw_senses:
        raise ValueError("sense inventory requires a non-empty senses mapping")

    senses: dict[str, tuple[SenseDefinition, ...]] = {}
    for raw_short_form, raw_definitions in raw_senses.items():
        short_form = _normalize_short_form(raw_short_form)
        if not short_form:
            raise ValueError("sense inventory short forms must be non-empty")
        if not _is_nonstring_sequence(raw_definitions) or not raw_definitions:
            raise ValueError(f"sense inventory entry {short_form!r} needs candidates")
        senses[short_form] = tuple(
            _definition_from_payload(short_form, item) for item in raw_definitions
        )
    return SenseInventory(senses)


def _definition_from_payload(short_form: str, payload: object) -> SenseDefinition:
    if not isinstance(payload, Mapping):
        raise ValueError(f"candidate for {short_form!r} must be an object")
    for key in ("long_form", "semantic_type", "source"):
        if not _required_text(payload.get(key)):
            raise ValueError(f"candidate for {short_form!r} requires {key}")

    prior = payload.get("prior", 0.1)
    if isinstance(prior, bool) or not isinstance(prior, (int, float)):
        raise ValueError(f"candidate prior for {short_form!r} must be numeric")
    if not 0.0 <= float(prior) <= 1.0:
        raise ValueError(f"candidate prior for {short_form!r} must be between 0 and 1")

    return SenseDefinition(
        long_form=str(payload["long_form"]).strip(),
        semantic_type=_normalize_label(payload["semantic_type"]),
        source=str(payload["source"]).strip(),
        sections=_string_tuple(short_form, "sections", payload.get("sections", ())),
        entity_types=_string_tuple(
            short_form,
            "entity_types",
            payload.get("entity_types", ()),
            normalize_labels=True,
        ),
        cue_words=_string_tuple(short_form, "cue_words", payload.get("cue_words", ())),
        prior=float(prior),
    )


def _merge_inventory(
    merged: dict[str, list[SenseDefinition]],
    custom: SenseInventory,
) -> None:
    for short_form, custom_candidates in custom.items():
        current = merged.setdefault(short_form, [])
        positions = {
            candidate.long_form.casefold(): index
            for index, candidate in enumerate(current)
        }
        for candidate in custom_candidates:
            key = candidate.long_form.casefold()
            if key in positions:
                current[positions[key]] = candidate
            else:
                positions[key] = len(current)
                current.append(candidate)


def _score_candidate(
    definition: SenseDefinition,
    *,
    index: int,
    context: str,
    section: str,
    entity_types: set[str],
) -> _ScoredCandidate:
    score = definition.prior
    features: list[str] = []

    normalized_sections = {_normalize_label(value) for value in definition.sections}
    if section and section in normalized_sections:
        score += 3.0
        features.append(f"section:{section}")

    matched_entity_types = sorted(entity_types & set(definition.entity_types))
    for entity_type in matched_entity_types:
        score += 2.0
        features.append(f"entity_type:{entity_type}")

    for cue in definition.cue_words:
        normalized_cue = _normalize_text(cue)
        if normalized_cue and _phrase_in_text(normalized_cue, context):
            score += 1.5
            features.append(f"cue:{normalized_cue}")

    return _ScoredCandidate(
        definition=definition,
        index=index,
        raw_score=score,
        matched_features=tuple(features),
    )


def _coerce_span(text: str, span: object) -> _SpanRecord:
    start = _field_value(span, "start")
    end = _field_value(span, "end")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
    ):
        raise ValueError("abbreviation spans require integer start and end")
    normalized_start = start
    normalized_end = end
    if (
        normalized_start < 0
        or normalized_end < normalized_start
        or normalized_end > len(text)
    ):
        raise ValueError("abbreviation span offsets must fall within text")

    raw_label = next(
        (
            value
            for key in _LABEL_KEYS
            if (value := _field_value(span, key)) is not None
        ),
        None,
    )
    raw_section = next(
        (
            value
            for key in _SECTION_KEYS
            if (value := _field_value(span, key)) is not None
        ),
        None,
    )
    direct_entity_types: list[str] = []
    for key in _DIRECT_ENTITY_TYPE_KEYS:
        value = _field_value(span, key)
        if value is None:
            continue
        if isinstance(value, str):
            values: Iterable[object] = (value,)
        elif _is_nonstring_sequence(value):
            values = value
        else:
            raise ValueError(f"span {key} must be a string sequence")
        direct_entity_types.extend(
            normalized for item in values if (normalized := _normalize_label(item))
        )

    return _SpanRecord(
        original=span,
        start=normalized_start,
        end=normalized_end,
        label=_normalize_label(raw_label) or None,
        section=str(raw_section).strip() if raw_section is not None else None,
        direct_entity_types=tuple(dict.fromkeys(direct_entity_types)),
    )


def _field_value(item: object, name: str) -> object | None:
    if isinstance(item, Mapping):
        return item.get(name)
    return getattr(item, name, None)


def _normalize_short_form(value: object) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFKC", str(value)).strip().upper()


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label(value: object) -> str:
    normalized = _normalize_text(value)
    return re.sub(r"[\s-]+", "_", normalized)


@lru_cache(maxsize=1024)
def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = re.escape(phrase).replace(r"\ ", r"\s+")
    return re.compile(r"(?<!\w)" + escaped + r"(?!\w)")


def _phrase_in_text(phrase: str, text: str) -> bool:
    return _phrase_pattern(phrase).search(text) is not None


def _required_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonstring_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _string_tuple(
    short_form: str,
    field_name: str,
    value: object,
    *,
    normalize_labels: bool = False,
) -> tuple[str, ...]:
    if not _is_nonstring_sequence(value):
        raise ValueError(f"candidate {field_name} for {short_form!r} must be a list")
    items: list[str] = []
    for item in value:
        if not _required_text(item):
            raise ValueError(
                f"candidate {field_name} for {short_form!r} must contain strings"
            )
        normalized = _normalize_label(item) if normalize_labels else str(item).strip()
        items.append(normalized)
    return tuple(dict.fromkeys(items))


__all__ = [
    "ABBREVIATION_DISAMBIGUATION_ADVISORY",
    "DEFAULT_SENSE_INVENTORY_RESOURCE",
    "AbbreviationAnnotation",
    "AbbreviationDisambiguator",
    "AbbreviationSense",
    "SenseAlternative",
    "SenseDefinition",
    "SenseInventory",
    "disambiguate_abbreviation",
    "expand_abbreviations",
    "load_abbreviation_inventory",
    "load_sense_inventory",
]
