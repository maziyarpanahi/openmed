"""Terminology system key and URI normalization tests."""

from __future__ import annotations

import pytest

from openmed.clinical.grounding import GroundingCandidate
from openmed.clinical.grounding.systems import (
    RESTRICTED_SYSTEMS,
    SYSTEM_ALIASES,
    SYSTEM_URIS,
    canonical_system,
    system_uri,
)

# Exact emitted identifiers. Changing any entry, including the legacy CPT URN,
# is an output change and must not happen as a side effect of input parsing.
EXPECTED_SYSTEM_URIS = {
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
HL7_CPT_URI = "http://www.ama-assn.org/go/cpt"


def test_emitted_system_uris_are_unchanged() -> None:
    assert dict(SYSTEM_URIS) == EXPECTED_SYSTEM_URIS
    assert RESTRICTED_SYSTEMS == frozenset({"umls", "snomed", "cpt"})
    for system, uri in EXPECTED_SYSTEM_URIS.items():
        assert system_uri(system) == uri
    assert system_uri("cpt") == "urn:openmed:restricted:cpt"


@pytest.mark.parametrize(("system", "uri"), sorted(EXPECTED_SYSTEM_URIS.items()))
def test_short_name_and_declared_uri_resolve_to_the_same_key(
    system: str, uri: str
) -> None:
    assert canonical_system(system) == system
    assert canonical_system(uri) == system
    assert canonical_system(f"{uri}/") == system
    assert canonical_system(f"  {uri}/  ") == system


def test_restricted_system_uri_resolves_to_its_key() -> None:
    assert canonical_system("http://snomed.info/sct") == "snomed"
    assert canonical_system("http://loinc.org/") == "loinc"


@pytest.mark.parametrize(
    "value",
    [
        "http://terminology.hl7.org/CodeSystem/umls",
        "http://terminology.hl7.org/codesystem/umls",
        "HTTP://TERMINOLOGY.HL7.ORG/CODESYSTEM/UMLS/",
    ],
)
def test_mixed_case_umls_uri_matches_normalized_input(value: str) -> None:
    assert canonical_system(value) == "umls"


@pytest.mark.parametrize("value", [HL7_CPT_URI, f"{HL7_CPT_URI}/"])
def test_hl7_cpt_uri_is_an_input_alias_for_cpt(value: str) -> None:
    assert canonical_system(value) == "cpt"
    assert canonical_system(value) in RESTRICTED_SYSTEMS
    assert system_uri(canonical_system(value)) == "urn:openmed:restricted:cpt"


@pytest.mark.parametrize(("alias", "system"), sorted(SYSTEM_ALIASES.items()))
def test_existing_short_name_aliases_are_preserved(alias: str, system: str) -> None:
    assert canonical_system(alias) == system


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" SCT ", "snomed"),
        ("Rx Norm", "rxnorm"),
        ("ICD-10", "icd10cm"),
        ("cpt-4", "cpt"),
    ],
)
def test_alias_normalization_controls(value: str, expected: str) -> None:
    assert canonical_system(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("not-a-system", "not-a-system"),
        ("Synthetic System", "synthetic-system"),
        (
            "http://example.invalid/fhir/CodeSystem/Synthetic",
            "http://example.invalid/fhir/codesystem/synthetic",
        ),
        ("https://loinc.org", "https://loinc.org"),
        ("http://snomed.info/sct/extension", "http://snomed.info/sct/extension"),
        ("snomed/", "snomed/"),
        ("/", "/"),
        ("", ""),
    ],
)
def test_unknown_inputs_stay_unmapped(value: str, expected: str) -> None:
    assert canonical_system(value) == expected
    assert canonical_system(value) not in SYSTEM_URIS


@pytest.mark.parametrize("value", [None, 1, b"http://snomed.info/sct"])
def test_non_string_inputs_keep_existing_type_checks(value: object) -> None:
    with pytest.raises(TypeError, match="terminology system must be a string"):
        canonical_system(value)
    assert system_uri(value) is None


@pytest.mark.parametrize("value", [*EXPECTED_SYSTEM_URIS.values(), HL7_CPT_URI])
def test_system_uri_passes_uri_input_through_unchanged(value: str) -> None:
    assert system_uri(value) == value


@pytest.mark.parametrize(
    ("value", "system"),
    [
        ("http://snomed.info/sct/", "snomed"),
        ("http://terminology.hl7.org/CodeSystem/umls", "umls"),
        ("urn:openmed:restricted:cpt", "cpt"),
        (HL7_CPT_URI, "cpt"),
        ("http://loinc.org/", "loinc"),
    ],
)
def test_uri_system_records_emit_the_canonical_system_identifier(
    value: str, system: str
) -> None:
    candidate = GroundingCandidate(
        system=value,
        code="SYN-0001",
        display="Synthetic concept",
        confidence=0.5,
    )

    assert candidate.system == system
    assert candidate.system_uri == EXPECTED_SYSTEM_URIS[system]
    assert candidate.to_dict()["system_uri"] == EXPECTED_SYSTEM_URIS[system]
