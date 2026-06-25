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
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from .schemas.span import OpenMedSpan

DETECTOR_ENTRY_POINT_GROUP = "openmed.detectors"
DETECTOR_STAGES = frozenset({"deterministic", "fast_pii", "clinical_phi"})
LANGUAGE_WILDCARDS = frozenset({"*", "all", "any"})

DetectCallable = Callable[..., Sequence[OpenMedSpan]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DetectorSpec:
    """Registration contract for an in-pipeline detector plugin."""

    name: str
    stage: str
    languages: Sequence[str] | str
    detect: DetectCallable
    provenance_prefix: str = "plugin"

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
        if not provenance_prefix:
            raise ValueError("DetectorSpec.provenance_prefix must be non-empty")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "languages", languages)
        object.__setattr__(self, "provenance_prefix", provenance_prefix)


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


def _language_matches(languages: Sequence[str], lang: str) -> bool:
    language_set = set(languages)
    return bool(language_set & LANGUAGE_WILDCARDS) or lang in language_set


def _reset_detector_registry_for_tests() -> None:
    global _DISCOVERY_COMPLETE

    with _DISCOVERY_LOCK:
        DETECTOR_REGISTRY.clear()
        DETECTOR_REGISTRY.update({stage: {} for stage in DETECTOR_STAGES})
        _DISCOVERY_COMPLETE = False


__all__ = [
    "DETECTOR_ENTRY_POINT_GROUP",
    "DETECTOR_REGISTRY",
    "DETECTOR_STAGES",
    "DetectorSpec",
    "DetectCallable",
    "detector_provenance",
    "discover_detectors",
    "iter_detectors",
    "register_detector",
]
