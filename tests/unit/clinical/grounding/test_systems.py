"""Regression tests for FHIR system-URI recognition in grounding (#4574)."""

from __future__ import annotations

import socket

import pytest
from fastapi.testclient import TestClient

from openmed.clinical.grounding import (
    FREE_VOCAB_SYSTEMS,
    RESTRICTED_SYSTEMS,
    SYSTEM_URIS,
    canonical_system,
    ground,
    system_uri,
)
from openmed.clinical.grounding.vocab import RestrictedVocabularyError
from openmed.service.app import create_app

HL7_CPT = "http://www.ama-assn.org/go/cpt"


def test_canonical_system_maps_every_uri_to_its_key() -> None:
    # The acceptance-criterion systems: their short name, FHIR URI, and URI with
    # a trailing slash all resolve to the same canonical key. (FHIR system URIs
    # are case-sensitive; the UMLS URI in particular is not asserted here because
    # a case-mangled copy must stay unrecognized.)
    for system in ("loinc", "snomed", "cpt"):
        uri = SYSTEM_URIS[system]
        assert canonical_system(system) == system
        assert canonical_system(uri) == system
        assert canonical_system(uri + "/") == system


def test_canonical_system_recognizes_hl7_cpt_uri_for_input() -> None:
    # The HL7 CPT URI is accepted on input as ``cpt`` (recognized as restricted)
    # while OpenMed keeps emitting ``urn:openmed:restricted:cpt``.
    assert canonical_system(HL7_CPT) == "cpt"
    assert canonical_system(HL7_CPT + "/") == "cpt"
    assert canonical_system("urn:openmed:restricted:cpt") == "cpt"
    # Emits unchanged: the public URI for ``cpt`` is still the restricted urn.
    assert system_uri("cpt") == "urn:openmed:restricted:cpt"
    # A caller passing the HL7 CPT URI as the emitted system keeps it verbatim.
    assert system_uri(HL7_CPT) == HL7_CPT


def test_canonical_system_still_maps_free_system_uris() -> None:
    # Free vocabularies resolve through the same URI path and stay unrestricted.
    for system in ("loinc", "rxnorm", "icd10cm", "hpo", "mesh"):
        assert system in FREE_VOCAB_SYSTEMS
        assert canonical_system(SYSTEM_URIS[system]) == system
        assert system not in RESTRICTED_SYSTEMS


def test_canonical_system_unknown_string_is_returned_intact() -> None:
    # A system that is neither a known name nor a known URI passes through, so
    # callers (and downstream ``ValueError``s) keep the original text.
    assert canonical_system("urn:example:custom") == "urn:example:custom"


@pytest.mark.parametrize(
    "uri",
    [
        HL7_CPT,  # HL7 CPT (input-only URI, accepted on input)
        "urn:openmed:restricted:cpt",  # emitted CPT urn
        "http://snomed.info/sct",  # SNOMED CT
    ],
)
def test_ground_restricted_uri_without_loader_raises_and_makes_no_network_call(
    uri: str,
) -> None:
    # A restricted vocabulary URI with no user loader is rejected locally; the
    # guard fires before any vocabulary is touched, so no connection is opened.
    opened: list[object] = []

    def guard(*args: object, **kwargs: object) -> None:
        opened.append(" ".join(map(str, args)))

    real_create_connection = socket.create_connection
    try:
        socket.create_connection = guard  # type: ignore[method-assign]
        with pytest.raises(RestrictedVocabularyError):
            ground("metformin", systems=[uri], offline=True)
    finally:
        socket.create_connection = real_create_connection  # type: ignore[method-assign]

    assert opened == [], "no network connection may be opened for a restricted system"


def test_ground_route_maps_restricted_uri_to_reason_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The /ground endpoint maps a RestrictedVocabularyError to the
    # restricted_terminology_unconfigured reason code (not grounding_invalid_request).
    monkeypatch.setenv("OPENMED_PROFILE", "dev")
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        response = client.post(
            "/ground",
            json={"text": "metformin", "systems": [HL7_CPT]},
        )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "restricted_terminology_unconfigured"
