"""Terminology system key and URI normalization tests."""

from __future__ import annotations

import pytest

from openmed.clinical.grounding import (
    ConceptSpan,
    GroundedConcept,
    GroundingCandidate,
    GroundingResult,
)
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


# Serialized ``system`` and ``system_uri`` of result records at the PR base
# (174fd485c). Records keep a URI system as supplied apart from case, so saving
# and reloading a record must not move it to the short key or another URI.
SAVED_RECORD_SYSTEMS = [
    ("http://loinc.org/", "http://loinc.org/", "http://loinc.org/"),
    (
        "http://terminology.hl7.org/CodeSystem/umls",
        "http://terminology.hl7.org/codesystem/umls",
        "http://terminology.hl7.org/codesystem/umls",
    ),
    (HL7_CPT_URI, HL7_CPT_URI, HL7_CPT_URI),
    ("http://snomed.info/sct", "http://snomed.info/sct", "http://snomed.info/sct"),
    (
        "urn:openmed:restricted:cpt",
        "urn:openmed:restricted:cpt",
        "urn:openmed:restricted:cpt",
    ),
    ("cpt", "cpt", "urn:openmed:restricted:cpt"),
    ("sct", "snomed", "http://snomed.info/sct"),
    ("Rx Norm", "rxnorm", "http://www.nlm.nih.gov/research/umls/rxnorm"),
    (
        "http://example.invalid/fhir/CodeSystem/Synthetic",
        "http://example.invalid/fhir/codesystem/synthetic",
        "http://example.invalid/fhir/codesystem/synthetic",
    ),
]


def _saved_candidate(system: str) -> GroundingCandidate:
    return GroundingCandidate(
        system=system,
        code="SYN-0001",
        display="Synthetic concept",
        confidence=0.5,
    )


def _saved_concept(system: str) -> GroundedConcept:
    return GroundedConcept(
        span=ConceptSpan(0, 17),
        surface_text="synthetic finding",
        system=system,
        code="SYN-0001",
        display="Synthetic concept",
        confidence=0.5,
        candidates=(_saved_candidate(system),),
    )


@pytest.mark.parametrize(("value", "system", "uri"), SAVED_RECORD_SYSTEMS)
def test_saved_record_candidate_keeps_base_serialization(
    value: str, system: str, uri: str
) -> None:
    saved = _saved_candidate(value).to_dict()
    uri_only = GroundingCandidate.from_dict(
        {"system_uri": value, "code": "SYN-0001", "display": "Synthetic concept"}
    ).to_dict()

    assert (saved["system"], saved["system_uri"]) == (system, uri)
    assert GroundingCandidate.from_dict(saved).to_dict() == saved
    assert (uri_only["system"], uri_only["system_uri"]) == (system, uri)


@pytest.mark.parametrize(("value", "system", "uri"), SAVED_RECORD_SYSTEMS)
def test_saved_record_concept_keeps_base_serialization(
    value: str, system: str, uri: str
) -> None:
    saved = _saved_concept(value).to_dict()
    uri_only_candidate = GroundedConcept.from_dict(
        {
            "start": 0,
            "end": 17,
            "surface": "synthetic finding",
            "candidates": [
                {"system_uri": value, "code": "SYN-0001", "display": "Synthetic"}
            ],
        }
    ).to_dict()["candidates"][0]

    assert (saved["system"], saved["system_uri"]) == (system, uri)
    assert saved["candidates"][0]["system"] == system
    assert saved["candidates"][0]["system_uri"] == uri
    assert GroundedConcept.from_dict(saved).to_dict() == saved
    assert uri_only_candidate["system"] == system
    assert uri_only_candidate["system_uri"] == uri


@pytest.mark.parametrize(("value", "system", "uri"), SAVED_RECORD_SYSTEMS)
def test_saved_record_result_keeps_base_systems_and_codes(
    value: str, system: str, uri: str
) -> None:
    saved = GroundingResult(
        concepts=(_saved_concept(value),),
        systems=(value,),
    ).to_dict()
    reloaded = GroundingResult.from_dict(saved).to_dict()

    assert saved["systems"] == [system]
    assert reloaded["systems"] == [system]
    assert reloaded["concepts"] == saved["concepts"]
    assert reloaded["concepts"][0]["system_uri"] == uri
    assert reloaded["spans"][0]["codes"] == {system: "SYN-0001"}
    assert reloaded["spans"][0]["cui"] is None
    assert GroundingResult.from_dict(reloaded).to_dict() == reloaded
