"""Stamp exported FHIR codings with caller-supplied terminology provenance.

FHIR R4 ``Coding`` supports a ``version`` element, but OpenMed cannot bundle
terminology release data or guess which release a caller used. This helper is
therefore intentionally mechanical: callers provide the vocabulary/version pin
map, and matching codings receive that version plus an optional source marker.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .codeable_concept_simple import system_uri

__all__ = [
    "CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL",
    "GROUNDING_CODE_PROVENANCE_EXTENSION_URL",
    "GROUNDED_CODE_PROVENANCE_EXTENSION_URL",
    "USER_SUPPLIED_TERMINOLOGY_PROVENANCE_EXTENSION_URL",
    "USER_SUPPLIED_TERMINOLOGY_ASSIST_ONLY_DISCLAIMER",
    "UserSuppliedTerminologyProvenance",
    "stamp_coding_provenance",
    "stamp_grounding_provenance",
    "stamp_user_supplied_terminology_provenance",
]

CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/code-system-version-source"
)
GROUNDING_CODE_PROVENANCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/grounded-code-provenance"
)
# Keep the adjective used by the existing FHIR exporter as a public alias.
GROUNDED_CODE_PROVENANCE_EXTENSION_URL = GROUNDING_CODE_PROVENANCE_EXTENSION_URL
USER_SUPPLIED_TERMINOLOGY_PROVENANCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/user-supplied-terminology-provenance"
)
USER_SUPPLIED_TERMINOLOGY_ASSIST_ONLY_DISCLAIMER = (
    "Assist-only terminology grounding for human review; not a clinical coding, "
    "treatment, or billing decision."
)


@dataclass(frozen=True)
class UserSuppliedTerminologyProvenance:
    """Provenance carried by one coding from a user-supplied dictionary."""

    source_name: str
    license_id: str
    restricted: bool

    def __post_init__(self) -> None:
        source_name = self.source_name.strip()
        license_id = self.license_id.strip()
        if not source_name:
            raise ValueError("terminology source_name must not be blank")
        if not license_id:
            raise ValueError("terminology license_id must not be blank")
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "license_id", license_id)


def stamp_coding_provenance(
    coding: Mapping[str, Any],
    version_pins: Mapping[str, str],
    *,
    source_label: str | None = None,
) -> dict[str, Any]:
    """Return a copy of ``coding`` stamped with code-system version provenance.

    ``version_pins`` may be keyed by a short vocabulary id accepted by
    :func:`openmed.clinical.exporters.codeable_concept_simple.system_uri`
    (for example ``"loinc"``) or by an already-canonical system URI (for
    example ``"http://loinc.org"``). Unpinned systems are returned unchanged:
    no version is guessed and no source marker is added.

    Args:
        coding: FHIR R4 ``Coding``-shaped mapping to stamp.
        version_pins: Caller-supplied system/version map.
        source_label: Optional label describing where the pin came from, such
            as a local catalog or release manifest. Emitted as a FHIR extension
            only when the coding receives a pinned version.

    Returns:
        A new ``dict`` with all input fields preserved, plus ``version`` and an
        optional source extension when ``coding.system`` is present in
        ``version_pins``.

    Raises:
        ValueError: If the coding system or a pin key is an unknown short
            vocabulary id rather than a canonical URI.
        TypeError: If ``coding.extension`` is present but is not a list.
    """

    result: dict[str, Any] = deepcopy(dict(coding))
    coding_system = result.get("system")
    if not isinstance(coding_system, str) or not coding_system:
        return result

    pins = _canonical_version_pins(version_pins)
    version = pins.get(system_uri(coding_system))
    if version is None:
        return result

    result["version"] = version
    if source_label is not None:
        _stamp_source_label(result, source_label)
    return result


def stamp_grounding_provenance(
    coding: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    evidence_start: int | None = None,
    evidence_end: int | None = None,
) -> dict[str, Any]:
    """Return a Coding copy carrying PHI-safe grounding provenance.

    The grounding adapter uses this helper to keep terminology provenance on
    the Coding that it explains. Only vocabulary-derived selection metadata and
    source offsets are emitted; arbitrary provenance keys (including raw text)
    are deliberately ignored. Existing extensions are preserved and a previous
    OpenMed grounding extension is replaced deterministically.

    Args:
        coding: FHIR R4 ``Coding``-shaped mapping to stamp.
        provenance: Mapping containing ``linker``, ``score`` or ``confidence``,
            optional ``matched_alias`` and ``vocab_version`` values. The
            ``linker_name``, ``source``, and
            ``vocabulary_snapshot_version`` aliases are accepted for results
            produced by newer grounding facades.
        evidence_start: Inclusive source offset. When omitted, ``start`` or
            ``evidence_start`` is read from ``provenance``.
        evidence_end: Exclusive source offset. When omitted, ``end`` or
            ``evidence_end`` is read from ``provenance``.

    Returns:
        A deep copy of ``coding`` with a nested grounding provenance extension.

    Raises:
        TypeError: If ``provenance`` or ``coding.extension`` has the wrong
            shape.
        ValueError: If the evidence offsets are missing or invalid, or if the
            score is not finite.
    """

    if not isinstance(provenance, Mapping):
        raise TypeError("grounding provenance must be a mapping")

    start = evidence_start
    if start is None:
        start = provenance.get("evidence_start", provenance.get("start"))
    end = evidence_end
    if end is None:
        end = provenance.get("evidence_end", provenance.get("end"))
    if type(start) is not int or start < 0:
        raise ValueError("grounding evidence_start must be a non-negative integer")
    if type(end) is not int or end <= start:
        raise ValueError(
            "grounding evidence_end must be an integer after evidence_start"
        )

    raw_score = provenance.get("score", provenance.get("confidence", 0.0))
    score = float(raw_score)
    if not math.isfinite(score):
        raise ValueError("grounding provenance score must be finite")

    linker = _provenance_value(
        provenance,
        "linker",
        "linker_name",
        "source",
        default="unavailable",
    )
    matched_alias = _provenance_value(
        provenance,
        "matched_alias",
        default="unavailable",
    )
    vocab_version = _provenance_value(
        provenance,
        "vocab_version",
        "vocabulary_snapshot_version",
        "snapshot_version",
        default="unavailable",
    )
    nested: list[dict[str, Any]] = [
        {"url": "linker", "valueString": linker},
        {"url": "score", "valueDecimal": score},
        {"url": "matched_alias", "valueString": matched_alias},
        {"url": "vocab_version", "valueString": vocab_version},
        {"url": "evidence_start", "valueUnsignedInt": start},
        {"url": "evidence_end", "valueUnsignedInt": end},
    ]
    text_hash = provenance.get("text_hash")
    if isinstance(text_hash, str) and text_hash:
        nested.append({"url": "text_hash", "valueString": text_hash})

    result: dict[str, Any] = deepcopy(dict(coding))
    extension = {
        "url": GROUNDING_CODE_PROVENANCE_EXTENSION_URL,
        "extension": nested,
    }
    extensions = result.get("extension")
    if extensions is None:
        result["extension"] = [extension]
        return result
    if not isinstance(extensions, list):
        raise TypeError("Coding.extension must be a list when present")
    result["extension"] = [
        item
        for item in extensions
        if not (
            isinstance(item, Mapping)
            and item.get("url") == GROUNDING_CODE_PROVENANCE_EXTENSION_URL
        )
    ]
    result["extension"].append(extension)
    return result


def stamp_user_supplied_terminology_provenance(
    coding: Mapping[str, Any],
    provenance: UserSuppliedTerminologyProvenance,
) -> dict[str, Any]:
    """Return a coding copy stamped with source, license, and assist-only scope.

    Local paths and dictionary surfaces are deliberately excluded. Repeated
    stamping replaces the OpenMed extension so output remains deterministic.
    """

    result: dict[str, Any] = deepcopy(dict(coding))
    extension = {
        "url": USER_SUPPLIED_TERMINOLOGY_PROVENANCE_EXTENSION_URL,
        "extension": [
            {"url": "sourceName", "valueString": provenance.source_name},
            {"url": "license", "valueString": provenance.license_id},
            {"url": "restricted", "valueBoolean": provenance.restricted},
            {"url": "assistOnly", "valueBoolean": True},
            {
                "url": "disclaimer",
                "valueString": USER_SUPPLIED_TERMINOLOGY_ASSIST_ONLY_DISCLAIMER,
            },
        ],
    }
    extensions = result.get("extension")
    if extensions is None:
        result["extension"] = [extension]
        return result
    if not isinstance(extensions, list):
        raise TypeError("Coding.extension must be a list when present")
    result["extension"] = [
        item
        for item in extensions
        if not (
            isinstance(item, Mapping)
            and item.get("url") == USER_SUPPLIED_TERMINOLOGY_PROVENANCE_EXTENSION_URL
        )
    ]
    result["extension"].append(extension)
    return result


def _provenance_value(
    provenance: Mapping[str, Any],
    *keys: str,
    default: str,
) -> str:
    """Return the first non-empty string provenance value."""

    for key in keys:
        value = provenance.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return default


def _canonical_version_pins(version_pins: Mapping[str, str]) -> dict[str, str]:
    """Return version pins keyed by canonical system URI."""

    return {
        system_uri(vocabulary): version for vocabulary, version in version_pins.items()
    }


def _stamp_source_label(coding: dict[str, Any], source_label: str) -> None:
    """Append or replace OpenMed's source-label extension in ``coding``."""

    provenance_extension = {
        "url": CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL,
        "valueString": source_label,
    }
    extensions = coding.get("extension")
    if extensions is None:
        coding["extension"] = [provenance_extension]
        return
    if not isinstance(extensions, list):
        raise TypeError("Coding.extension must be a list when present")

    coding["extension"] = [
        extension
        for extension in extensions
        if not (
            isinstance(extension, Mapping)
            and extension.get("url") == CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL
        )
    ]
    coding["extension"].append(provenance_extension)
