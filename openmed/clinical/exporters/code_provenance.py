"""Stamp exported FHIR codings with caller-supplied version provenance.

FHIR R4 ``Coding`` supports a ``version`` element, but OpenMed cannot bundle
terminology release data or guess which release a caller used. This helper is
therefore intentionally mechanical: callers provide the vocabulary/version pin
map, and matching codings receive that version plus an optional source marker.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .codeable_concept_simple import system_uri

__all__ = [
    "CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL",
    "stamp_coding_provenance",
]

CODE_SYSTEM_VERSION_SOURCE_EXTENSION_URL = (
    "https://openmed.ai/fhir/StructureDefinition/code-system-version-source"
)


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
