"""Canonical terminology system identifiers used by grounding outputs."""

from __future__ import annotations

from collections.abc import Mapping

__all__ = [
    "RESTRICTED_SYSTEMS",
    "SYSTEM_ALIASES",
    "SYSTEM_URIS",
    "canonical_system",
    "system_uri",
]


SYSTEM_URIS: Mapping[str, str] = {
    "cvx": "http://hl7.org/fhir/sid/cvx",
    "rxnorm": "http://www.nlm.nih.gov/research/umls/rxnorm",
    "icd10cm": "http://hl7.org/fhir/sid/icd-10-cm",
    "loinc": "http://loinc.org",
    "hpo": "http://human-phenotype-ontology.org",
    "mesh": "https://www.nlm.nih.gov/mesh",
    "umls": "http://terminology.hl7.org/CodeSystem/umls",
    "snomed": "http://snomed.info/sct",
    "cpt": "urn:openmed:restricted:cpt",
}

SYSTEM_ALIASES: Mapping[str, str] = {
    "cvx": "cvx",
    "rxnorm": "rxnorm",
    "rx-norm": "rxnorm",
    "rx_norm": "rxnorm",
    "icd10": "icd10cm",
    "icd10cm": "icd10cm",
    "icd-10": "icd10cm",
    "icd-10-cm": "icd10cm",
    "icd_10_cm": "icd10cm",
    "loinc": "loinc",
    "hpo": "hpo",
    "hp": "hpo",
    "mesh": "mesh",
    "ms": "mesh",
    "umls": "umls",
    "snomed": "snomed",
    "snomed-ct": "snomed",
    "snomedct": "snomed",
    "snomed_ct": "snomed",
    "sct": "snomed",
    "cpt": "cpt",
    "cpt4": "cpt",
    "cpt-4": "cpt",
}

RESTRICTED_SYSTEMS = frozenset({"umls", "snomed", "cpt"})

# Input-only URIs that identify a known system but are never emitted. CPT output
# keeps ``SYSTEM_URIS["cpt"]``.
_INPUT_SYSTEM_URI_ALIASES: Mapping[str, str] = {
    "http://www.ama-assn.org/go/cpt": "cpt",
}

_SYSTEM_URI_KEYS: Mapping[str, str] = {
    **{uri.rstrip("/").casefold(): system for system, uri in SYSTEM_URIS.items()},
    **{
        uri.rstrip("/").casefold(): system
        for uri, system in _INPUT_SYSTEM_URI_ALIASES.items()
    },
}


def canonical_system(value: object) -> str:
    """Return a normalized terminology system key.

    Short names and aliases resolve through ``SYSTEM_ALIASES``. A known system
    URI, such as ``http://snomed.info/sct``, resolves to its key, ignoring case
    and a trailing slash. Unknown values are returned normalized but unmapped.

    Args:
        value: A terminology system short name, alias, or system URI.

    Returns:
        The canonical system key, or the normalized input when it is unknown.

    Raises:
        TypeError: If ``value`` is not a string.
    """

    key = _record_system_key(value)
    return _SYSTEM_URI_KEYS.get(key.rstrip("/"), key)


def _record_system_key(value: object) -> str:
    """Normalize the system stored on a grounding result record.

    Result records resolve short names and aliases only and keep any other value,
    including a URI, as supplied apart from case and spacing. Rebuilding a saved
    record therefore keeps its serialized ``system`` and ``system_uri``. Requests
    and snapshots use :func:`canonical_system`, which also maps known URIs.
    """

    if not isinstance(value, str):
        raise TypeError("terminology system must be a string")
    key = value.strip().casefold().replace(" ", "-")
    return SYSTEM_ALIASES.get(key, key)


def system_uri(value: object) -> str | None:
    """Return a canonical FHIR system URI, or ``None`` for local systems."""

    if not isinstance(value, str):
        return None
    if value.startswith(("http://", "https://", "urn:")):
        return value
    return SYSTEM_URIS.get(canonical_system(value))
