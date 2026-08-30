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


def canonical_system(value: object) -> str:
    """Return a normalized terminology system key."""

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
