"""Custom detector plugin registry for the privacy pipeline.

Third-party packages can register span-producing detectors without adding a
runtime dependency to OpenMed core. A plugin exposes the ``openmed.detectors``
entry-point group and returns one or more :class:`DetectorSpec` records:

```toml
[project.entry-points."openmed.detectors"]
acme_mrn = "acme_openmed_detectors:detector"
```

```python
from openmed.core.detector_plugins import DetectorSpec
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash


def detector() -> DetectorSpec:
    return DetectorSpec(
        name="acme_mrn",
        stage="deterministic",
        languages=("en",),
        detect=detect,
    )


def detect(text: str, *, lang: str, context=None):
    start = text.find("MRN:")
    if start < 0:
        return ()
    end = start + len("MRN: 12345")
    return (
        OpenMedSpan(
            doc_id="plugin-placeholder",
            start=start,
            end=end,
            text_hash=hmac_text_hash(text[start:end], "plugin-placeholder"),
            entity_type="ID_NUM",
            canonical_label="ID_NUM",
            score=0.95,
        ),
    )
```

Pipeline execution passes normalized text to plugin detectors and rewrites
``doc_id``, ``text_hash``, and detector provenance before arbitration. Plugins
must return offsets into the normalized text and must not place raw PHI/PII
surface text in metadata or evidence; the pipeline records only hashes.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .labels import (
    CANONICAL_LABELS,
    ID_SUBTYPE_ABHA,
    ID_SUBTYPE_GSTIN,
    ID_SUBTYPE_IFSC,
    ID_SUBTYPE_INDIAN_DRIVING_LICENCE,
    ID_SUBTYPE_INDIAN_PASSPORT,
    ID_SUBTYPE_PAN,
    ID_SUBTYPE_VOTER_ID_EPIC,
    id_subtype_for,
    normalize_label,
)
from .schemas.span import OpenMedSpan, hmac_text_hash

DETECTOR_ENTRY_POINT_GROUP = "openmed.detectors"
DETECTOR_STAGES = frozenset({"deterministic", "fast_pii", "clinical_phi"})
LANGUAGE_WILDCARDS = frozenset({"*", "all", "any"})
INDIAN_MULTI_ID_DETECTOR = "indian_multi_id"
INDIAN_MULTI_ID_ENTITY_TYPES = frozenset(
    {
        "abha",
        "gstin",
        "ifsc",
        "indian_driving_licence",
        "indian_passport",
        "indian_vehicle_registration",
        "pan",
        "voter_id_epic",
    }
)
INDIAN_MULTI_ID_SUBTYPES = {
    "abha": ID_SUBTYPE_ABHA,
    "gstin": ID_SUBTYPE_GSTIN,
    "ifsc": ID_SUBTYPE_IFSC,
    "indian_driving_licence": ID_SUBTYPE_INDIAN_DRIVING_LICENCE,
    "indian_passport": ID_SUBTYPE_INDIAN_PASSPORT,
    "pan": ID_SUBTYPE_PAN,
    "voter_id_epic": ID_SUBTYPE_VOTER_ID_EPIC,
}

DetectCallable = Callable[..., Sequence[OpenMedSpan]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorCapability:
    """Compile-time detector coverage used by policy proof checks."""

    name: str
    stage: str
    covered_labels: Sequence[str] | str
    languages: Sequence[str] | str = ("*",)
    provenance_prefix: str = "builtin"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("DetectorCapability.name must be non-empty")
        if ":" in name:
            raise ValueError("DetectorCapability.name must not contain ':'")

        stage = _normalize_stage(self.stage)
        languages = _normalize_languages(self.languages)
        covered_labels = _normalize_covered_labels(self.covered_labels)
        provenance_prefix = str(self.provenance_prefix or "builtin").strip().rstrip(":")
        if not provenance_prefix:
            raise ValueError("DetectorCapability.provenance_prefix must be non-empty")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "covered_labels", covered_labels)
        object.__setattr__(self, "provenance_prefix", provenance_prefix)

    @property
    def detector_id(self) -> str:
        """Return the stable detector identifier used in compiled plans."""

        return f"{self.provenance_prefix}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible capability record."""

        return {
            "name": self.name,
            "stage": self.stage,
            "detector_id": self.detector_id,
            "covered_labels": list(self.covered_labels),
            "languages": list(self.languages),
            "provenance_prefix": self.provenance_prefix,
        }


@dataclass(frozen=True)
class DetectorSpec:
    """Registration contract for an in-pipeline detector plugin."""

    name: str
    stage: str
    languages: Sequence[str] | str
    detect: DetectCallable
    provenance_prefix: str = "plugin"
    covered_labels: Sequence[str] | str = ()

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("DetectorSpec.name must be non-empty")
        if ":" in name:
            raise ValueError("DetectorSpec.name must not contain ':'")
        if not callable(self.detect):
            raise TypeError("DetectorSpec.detect must be callable")

        stage = _normalize_stage(self.stage)
        languages = _normalize_languages(self.languages)
        provenance_prefix = str(self.provenance_prefix or "plugin").strip().rstrip(":")
        covered_labels = _normalize_covered_labels(
            self.covered_labels,
            allow_empty=True,
        )
        if not provenance_prefix:
            raise ValueError("DetectorSpec.provenance_prefix must be non-empty")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "provenance_prefix", provenance_prefix)
        object.__setattr__(self, "covered_labels", covered_labels)

    def capability(self) -> DetectorCapability | None:
        """Return this plugin's policy-compiler capability, if declared."""

        if not self.covered_labels:
            return None
        return DetectorCapability(
            name=self.name,
            stage=self.stage,
            covered_labels=self.covered_labels,
            languages=self.languages,
            provenance_prefix=self.provenance_prefix,
        )


DETECTOR_REGISTRY: dict[str, dict[str, DetectorSpec]] = {
    stage: {} for stage in DETECTOR_STAGES
}

_DISCOVERY_LOCK = RLock()
_DISCOVERY_COMPLETE = False


def register_detector(spec: DetectorSpec) -> DetectorSpec:
    """Register a detector spec for later pipeline use."""

    if not isinstance(spec, DetectorSpec):
        raise TypeError("register_detector expects a DetectorSpec")

    with _DISCOVERY_LOCK:
        DETECTOR_REGISTRY.setdefault(spec.stage, {})[spec.name] = spec
    return spec


def detect_indian_identifiers(
    text: str,
    *,
    lang: str = "en",
    context: Any = None,
) -> tuple[OpenMedSpan, ...]:
    """Detect validator-backed Indian identifiers without registry lookups.

    Returned records contain offsets, HMAC hashes, structural validator names,
    and subtype metadata only. Raw identifier surfaces are never stored in
    evidence or metadata.
    """

    del lang, context
    from .pii_entity_merger import find_context_words
    from .pii_i18n import INDIAN_MULTI_ID_PII_PATTERNS

    candidates: list[tuple[int, int, int, OpenMedSpan]] = []
    for pattern in sorted(
        INDIAN_MULTI_ID_PII_PATTERNS,
        key=lambda item: item.priority,
        reverse=True,
    ):
        for match in re.finditer(pattern.pattern, text, pattern.flags):
            surface = match.group(0)
            if pattern.validator is not None and not pattern.validator(surface):
                continue
            has_context = find_context_words(
                text,
                match.start(),
                match.end(),
                pattern.context_words,
            )
            if pattern.safety_sweep_requires_context and not has_context:
                continue

            score = pattern.base_score
            if has_context:
                score = min(1.0, score + pattern.context_boost)
            canonical = normalize_label(pattern.entity_type)
            subtype = INDIAN_MULTI_ID_SUBTYPES.get(
                pattern.entity_type,
                id_subtype_for(pattern.entity_type),
            )
            validator_name = (
                pattern.validator.__name__ if pattern.validator is not None else "none"
            )
            metadata: dict[str, Any] = {
                "context_match": has_context,
                "identifier_type": subtype or canonical.casefold(),
                "validator": validator_name,
            }
            candidates.append(
                (
                    match.start(),
                    match.end(),
                    pattern.priority,
                    OpenMedSpan(
                        doc_id="builtin-indian-id-detector",
                        start=match.start(),
                        end=match.end(),
                        text_hash=hmac_text_hash(
                            surface,
                            "builtin-indian-id-detector",
                        ),
                        entity_type=pattern.entity_type,
                        canonical_label=canonical,
                        score=score,
                        detector="builtin",
                        evidence={
                            "structure_valid": True,
                            "validator": validator_name,
                        },
                        metadata=metadata,
                    ),
                )
            )

    accepted: list[OpenMedSpan] = []
    for start, end, _priority, span in sorted(
        candidates,
        key=lambda item: (item[0], -item[2], -(item[1] - item[0])),
    ):
        if any(start < existing.end and end > existing.start for existing in accepted):
            continue
        accepted.append(span)
    return tuple(sorted(accepted, key=lambda span: (span.start, span.end)))


def _register_builtin_detectors() -> None:
    register_detector(
        DetectorSpec(
            name=INDIAN_MULTI_ID_DETECTOR,
            stage="deterministic",
            languages=("en", "hi", "te"),
            detect=detect_indian_identifiers,
            provenance_prefix="builtin",
            covered_labels=("ID_NUM", "VEHICLE_REGISTRATION"),
        )
    )


def iter_detectors(stage: str, lang: str | None = None) -> tuple[DetectorSpec, ...]:
    """Return detectors registered for ``stage`` and ``lang``.

    Entry points are discovered lazily the first time this function is called.
    Discovery is idempotent, and faulty plugins are logged and skipped.
    """

    discover_detectors()
    normalized_stage = _normalize_stage(stage)
    normalized_lang = _normalize_language(lang or "*")
    with _DISCOVERY_LOCK:
        specs = tuple(DETECTOR_REGISTRY.get(normalized_stage, {}).values())
    return tuple(
        spec for spec in specs if _language_matches(spec.languages, normalized_lang)
    )


def default_detector_capabilities() -> tuple[DetectorCapability, ...]:
    """Return built-in detector coverage available to compiled policies."""

    return (
        DetectorCapability(
            name="privacy_label_model",
            stage="fast_pii",
            covered_labels=tuple(sorted(CANONICAL_LABELS)),
        ),
    )


def iter_detector_capabilities(
    stage: str | None = None,
    lang: str | None = None,
    *,
    include_builtin: bool = True,
) -> tuple[DetectorCapability, ...]:
    """Return declared detector capabilities for policy compilation."""

    normalized_stage = _normalize_stage(stage) if stage is not None else None
    normalized_lang = _normalize_language(lang or "*")
    capabilities: list[DetectorCapability] = []

    if include_builtin:
        capabilities.extend(default_detector_capabilities())

    discover_detectors()
    with _DISCOVERY_LOCK:
        specs = tuple(
            spec
            for registry_stage in sorted(DETECTOR_REGISTRY)
            for spec in DETECTOR_REGISTRY.get(registry_stage, {}).values()
        )

    for spec in specs:
        if normalized_stage is not None and spec.stage != normalized_stage:
            continue
        if not _language_matches(spec.languages, normalized_lang):
            continue
        capability = spec.capability()
        if capability is not None:
            capabilities.append(capability)

    return tuple(
        capability
        for capability in capabilities
        if normalized_stage is None or capability.stage == normalized_stage
    )


def discover_detectors() -> None:
    """Discover ``openmed.detectors`` entry points once."""

    global _DISCOVERY_COMPLETE

    with _DISCOVERY_LOCK:
        if _DISCOVERY_COMPLETE:
            return
        _DISCOVERY_COMPLETE = True

    try:
        entry_points = _entry_points_for_group(DETECTOR_ENTRY_POINT_GROUP)
    except Exception as exc:  # pragma: no cover - importlib defensive guard
        logger.warning(
            "Failed to enumerate OpenMed detector plugins: %s",
            exc.__class__.__name__,
        )
        return

    for entry_point in entry_points:
        entry_name = str(getattr(entry_point, "name", "<unknown>"))
        try:
            loaded = entry_point.load()
            for spec in _coerce_entry_point_specs(loaded):
                register_detector(spec)
        except Exception as exc:
            logger.warning(
                "Failed to load OpenMed detector plugin %s: %s",
                entry_name,
                exc.__class__.__name__,
            )


def detector_provenance(spec: DetectorSpec) -> str:
    """Return the detector provenance string recorded on plugin spans."""

    return f"{spec.provenance_prefix}:{spec.name}"


def _entry_points_for_group(group: str) -> Sequence[Any]:
    try:
        return tuple(importlib_metadata.entry_points(group=group))
    except TypeError:
        entry_points = importlib_metadata.entry_points()
        if hasattr(entry_points, "select"):
            return tuple(entry_points.select(group=group))
        return tuple(entry_points.get(group, ()))


def _coerce_entry_point_specs(value: Any) -> tuple[DetectorSpec, ...]:
    if isinstance(value, DetectorSpec):
        return (value,)
    if callable(value):
        return _coerce_entry_point_specs(value())
    if value is None:
        return ()
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        specs: list[DetectorSpec] = []
        for item in value:
            specs.extend(_coerce_entry_point_specs(item))
        return tuple(specs)
    raise TypeError("openmed.detectors entry points must return DetectorSpec records")


def _normalize_stage(stage: str) -> str:
    normalized = str(stage or "").strip().lower()
    if normalized not in DETECTOR_STAGES:
        raise ValueError(
            f"DetectorSpec.stage must be one of {', '.join(sorted(DETECTOR_STAGES))}"
        )
    return normalized


def _normalize_languages(languages: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(languages, str):
        values = (languages,)
    else:
        values = tuple(languages)
    normalized = tuple(_normalize_language(value) for value in values if str(value))
    return normalized or ("*",)


def _normalize_language(lang: str) -> str:
    return str(lang or "*").strip().lower().replace("_", "-")


def _normalize_covered_labels(
    labels: Sequence[str] | str,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(labels, str):
        values = (labels,)
    else:
        values = tuple(labels)
    if len(values) == 1 and str(values[0]).strip().lower() in {"*", "all", "any"}:
        return tuple(sorted(CANONICAL_LABELS))

    normalized: set[str] = set()
    for value in values:
        label = str(value).strip()
        if not label:
            continue
        canonical = normalize_label(label)
        if canonical != label:
            raise ValueError(
                "covered_labels must use canonical labels, "
                f"got {label!r} for {canonical!r}"
            )
        normalized.add(canonical)

    if not normalized and not allow_empty:
        raise ValueError("covered_labels must not be empty")
    return tuple(sorted(normalized))


def _language_matches(languages: Sequence[str], lang: str) -> bool:
    language_set = set(languages)
    return bool(language_set & LANGUAGE_WILDCARDS) or lang in language_set


def _reset_detector_registry_for_tests() -> None:
    global _DISCOVERY_COMPLETE

    with _DISCOVERY_LOCK:
        DETECTOR_REGISTRY.clear()
        DETECTOR_REGISTRY.update({stage: {} for stage in DETECTOR_STAGES})
        _DISCOVERY_COMPLETE = False
    _register_builtin_detectors()


_register_builtin_detectors()


__all__ = [
    "DETECTOR_ENTRY_POINT_GROUP",
    "DETECTOR_REGISTRY",
    "DETECTOR_STAGES",
    "INDIAN_MULTI_ID_DETECTOR",
    "INDIAN_MULTI_ID_ENTITY_TYPES",
    "INDIAN_MULTI_ID_SUBTYPES",
    "DetectorCapability",
    "DetectorSpec",
    "DetectCallable",
    "default_detector_capabilities",
    "detect_indian_identifiers",
    "detector_provenance",
    "discover_detectors",
    "iter_detector_capabilities",
    "iter_detectors",
    "register_detector",
]
