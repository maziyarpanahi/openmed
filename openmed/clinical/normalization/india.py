"""Indian-English clinical abbreviation normalization.

The map is deliberately small, deterministic, and terminology-free. It covers
common Indian prescription surfaces without bundling restricted clinical data
or attempting to infer a treatment recommendation.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

INDIAN_CLINICAL_NORMALIZATION_VERSION = "india-clinical-abbreviations-v1"

INDIAN_CLINICAL_ABBREVIATIONS: Mapping[str, str] = MappingProxyType(
    {
        "tab": "tablet",
        "cap": "capsule",
        "od": "once daily",
        "bd": "twice daily",
        "bid": "twice daily",
        "tds": "three times daily",
        "tid": "three times daily",
        "hs": "at bedtime",
        "sos": "as needed",
        "stat": "immediately",
        "ac": "before food",
        "pc": "after food",
        "1-0-1": "morning and evening",
    }
)

_ABBREVIATION_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        (
            re.escape(surface) + (r"\.?" if surface.isalpha() else "")
            for surface in sorted(
                INDIAN_CLINICAL_ABBREVIATIONS,
                key=len,
                reverse=True,
            )
        )
    )
    + r")(?!\w)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class IndianClinicalNormalization:
    """One normalized surface plus the abbreviation keys that changed it."""

    text: str
    expansions: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Return whether at least one registered abbreviation was expanded."""

        return bool(self.expansions)


def normalize_indian_clinical_abbreviation(surface: str) -> str:
    """Return the canonical form for one abbreviation-like surface.

    Unknown values are returned unchanged. Matching is case-insensitive and a
    trailing full stop is ignored for alphabetic abbreviations.
    """

    if not isinstance(surface, str):
        raise TypeError("surface must be a string")
    key = surface.strip().casefold()
    if key.endswith(".") and key[:-1].isalpha():
        key = key[:-1]
    return INDIAN_CLINICAL_ABBREVIATIONS.get(key, surface)


def normalize_indian_clinical_surface(surface: str) -> IndianClinicalNormalization:
    """Expand registered abbreviations inside an Indian-English entity surface."""

    if not isinstance(surface, str):
        raise TypeError("surface must be a string")

    expanded: list[str] = []

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        key = raw.casefold().removesuffix(".")
        canonical = INDIAN_CLINICAL_ABBREVIATIONS.get(key)
        if canonical is None:
            return raw
        expanded.append(key)
        return canonical

    return IndianClinicalNormalization(
        text=_ABBREVIATION_RE.sub(replace, surface),
        expansions=tuple(expanded),
    )


def normalize_indian_clinical_entities(
    entities: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy entities and normalize their surfaces before span merging.

    Offsets and original ``word`` values remain untouched. Changed entities
    gain a ``normalized_word`` field and PHI-free normalization provenance.
    """

    normalized_entities: list[dict[str, Any]] = []
    for entity in entities:
        copied = dict(entity)
        surface = str(copied.get("word", ""))
        normalization = normalize_indian_clinical_surface(surface)
        if normalization.changed:
            copied["normalized_word"] = normalization.text
            copied["clinical_normalization"] = {
                "locale": "en_IN",
                "version": INDIAN_CLINICAL_NORMALIZATION_VERSION,
                "expansions": list(normalization.expansions),
            }
        normalized_entities.append(copied)
    return normalized_entities


__all__ = [
    "INDIAN_CLINICAL_ABBREVIATIONS",
    "INDIAN_CLINICAL_NORMALIZATION_VERSION",
    "IndianClinicalNormalization",
    "normalize_indian_clinical_abbreviation",
    "normalize_indian_clinical_entities",
    "normalize_indian_clinical_surface",
]
