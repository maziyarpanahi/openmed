"""Standalone FHIR R4 CodeableConcept builder for a single coded entity.

Callers frequently have a single (system, code, display, text) tuple — e.g.
from a comparator adapter or a hand-coded mapping — and want a correctly shaped
FHIR ``CodeableConcept`` without going through grounding or the Athena index.

Two public entry points are exposed:

* :func:`coding` builds one ``{"system", "code", "display"}`` Coding dict from
  a vocabulary id (or already-canonical URI) plus a code string.
* :func:`codeable_concept` wraps one or more Codings into a ``CodeableConcept``
  dict, ordering them deterministically by a configurable system priority.

The canonical SYSTEM_URI map is the **single source of truth** for vocabulary
id → HL7 system URI mapping across OpenMed. Import :func:`system_uri` when you
need a canonical URI but not the full ``CodeableConcept`` machinery.

Out of scope: looking up codes against an Athena index or grounding. This module is the lower-level, purely mechanical
helper that higher-level builders can delegate to.
"""

from __future__ import annotations

from typing import Any

__all__ = ["system_uri", "coding", "codeable_concept"]

# Canonical HL7 FHIR R4 system URIs
# https://www.hl7.org/fhir/terminologies-systems.html

_SYSTEM_URI: dict[str, str] = {
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "icd-10-cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "loinc": "http://loinc.org",
    "snomed": "http://snomed.info/sct",
    "hpo": "http://purl.obolibrary.org/obo/hp.owl",
    "mesh": "https://meshb.nlm.nih.gov",
}

# Deterministic ordering for codings inside a CodeableConcept.
# Systems listed earlier sort first; systems absent from this list sort last
# (alphabetically among themselves so the output is still stable).
_DEFAULT_SYSTEM_PRIORITY: tuple[str, ...] = (
    "http://snomed.info/sct",
    "http://loinc.org",
    "http://www.nlm.nih.gov/research/umls/rxnorm",
    "http://hl7.org/fhir/sid/icd-10-cm",
    "http://purl.obolibrary.org/obo/hp.owl",
    "https://meshb.nlm.nih.gov",
)


def system_uri(vocabulary_id: str) -> str:
    """Return the canonical HL7 FHIR R4 system URI for *vocabulary_id*.

    Accepts either a short vocabulary id (case-insensitive) or an
    already-canonical URI.  Already-canonical URIs (those that start with
    ``http://`` or ``https://``) are returned unchanged so callers can
    safely pass either form without checking first.

    Args:
        vocabulary_id: A short vocabulary id such as ``"rxnorm"``,
            ``"loinc"``, ``"snomed"``, ``"icd-10-cm"``, ``"hpo"``, or
            ``"mesh"``; **or** an already-canonical system URI such as
            ``"http://loinc.org"``.

    Returns:
        The canonical HL7 system URI string.

    Raises:
        ValueError: If *vocabulary_id* is not a recognised short id and does
            not look like a canonical URI (i.e. does not start with
            ``http://`` or ``https://``).
    """
    # Already-canonical URI — pass through unchanged.
    if vocabulary_id.startswith(("http://", "https://")):
        return vocabulary_id

    key = vocabulary_id.lower().strip()
    if key not in _SYSTEM_URI:
        raise ValueError(
            f"Unknown vocabulary id: {vocabulary_id!r}. "
            f"Expected one of {sorted(_SYSTEM_URI)} or an already-canonical URI."
        )
    return _SYSTEM_URI[key]


def coding(
    system: str,
    code: str,
    display: str | None = None,
) -> dict[str, Any]:
    """Build one FHIR R4 Coding dict.

    Args:
        system: A short vocabulary id or
            an already-canonical system URI.  Resolved via :func:`system_uri`.
        code: The code string within the system (e.g. ``"1049502"``).
        display: Optional human-readable display label for the code.

    Returns:
        A ``{"system": ..., "code": ..., "display": ...}`` mapping.
        The ``"display"`` key is omitted when *display* is ``None``.

    Raises:
        ValueError: If *system* is not a recognised vocabulary id or
            canonical URI.
    """
    result: dict[str, Any] = {
        "system": system_uri(system),
        "code": code,
    }
    if display is not None:
        result["display"] = display
    return result


def codeable_concept(
    codings: list[dict[str, Any]],
    text: str | None = None,
    *,
    system_priority: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Build an R4-shaped CodeableConcept from one or more Codings.

    Codings are ordered deterministically by *system_priority* so that output
    is stable regardless of the order in which codings were passed.  Systems
    not present in the priority list are sorted alphabetically after those that
    are listed.

    Args:
        codings: One or more Coding dicts, typically produced by
            :func:`coding`.  Must not be empty.
        text: Optional free-text representation of the concept.  Emitted as
            ``CodeableConcept.text`` when provided.
        system_priority: Ordered tuple of canonical system URIs that controls
            the sort order of codings.  Defaults to
            :data:`_DEFAULT_SYSTEM_PRIORITY`.

    Returns:
        A ``{"coding": [...], "text": ...}`` mapping conforming to the FHIR R4
        ``CodeableConcept`` data type.  The ``"text"`` key is omitted when
        *text* is ``None``.

    Raises:
        ValueError: If *codings* is empty.
    """
    if not codings:
        raise ValueError("codeable_concept requires at least one coding")

    priority = (
        system_priority if system_priority is not None else _DEFAULT_SYSTEM_PRIORITY
    )
    priority_index = {uri: idx for idx, uri in enumerate(priority)}

    def _sort_key(c: dict[str, Any]) -> tuple[int, str]:
        uri = c.get("system", "")
        return (priority_index.get(uri, len(priority)), uri)

    ordered = sorted(codings, key=_sort_key)

    result: dict[str, Any] = {"coding": ordered}
    if text is not None:
        result["text"] = text
    return result
