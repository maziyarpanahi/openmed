"""FHIR leakage safety for Indian health identifiers."""

from __future__ import annotations

import json

from openmed.clinical.exporters.fhir import (
    INDIA_HEALTH_ID_REDACTION,
    sanitize_india_health_identifiers,
    to_bundle,
)


def _patient_resource() -> dict:
    return {
        "resourceType": "Patient",
        "id": "synthetic-patient",
        "identifier": [
            {
                "system": "https://healthid.abdm.gov.in/abha-number",
                "value": "7604 8764 7593 81",
            },
            {
                "system": "https://healthid.abdm.gov.in/abha-address",
                "value": "patient.482901@abdm",
            },
            {
                "system": "https://synthetic.example/upi",
                "value": "refund.593104@okaxis",
            },
            {
                "system": "https://synthetic.example/ration-card",
                "value": "DL-4829013756",
            },
            {
                "system": "mailto",
                "value": "records@example.org",
            },
        ],
        "extension": [
            {
                "url": "https://synthetic.example/PatientID",
                "PatientID": "patient.482901@abdm",
            }
        ],
        "communication": [{"language": {"text": "Hindi"}}],
    }


def test_fhir_sanitizer_redacts_identifier_and_patient_id_fields_only() -> None:
    patient = _patient_resource()
    sanitized = sanitize_india_health_identifiers(patient)
    values = [item["value"] for item in sanitized["identifier"]]

    assert values[:4] == [INDIA_HEALTH_ID_REDACTION] * 4
    assert values[4] == "records@example.org"
    assert sanitized["extension"][0]["PatientID"] == INDIA_HEALTH_ID_REDACTION
    assert sanitized["communication"] == patient["communication"]


def test_fhir_bundle_contains_no_india_health_id_surface_and_is_pure() -> None:
    patient = _patient_resource()
    snapshot = json.dumps(patient, sort_keys=True)
    raw_values = [item["value"] for item in patient["identifier"][:4]]
    bundle = to_bundle([patient], doc_id="synthetic-india-health-record")
    serialized = json.dumps(bundle, sort_keys=True)

    assert INDIA_HEALTH_ID_REDACTION in serialized
    assert "records@example.org" in serialized
    for raw in raw_values:
        assert raw not in serialized
    assert json.dumps(patient, sort_keys=True) == snapshot


def test_fhir_sanitizer_does_not_treat_foreign_identifier_as_ration_card() -> None:
    resource = {
        "resourceType": "Patient",
        "identifier": [{"system": "urn:foreign:id", "value": "X1234567L"}],
    }

    sanitized = sanitize_india_health_identifiers(resource)

    assert sanitized["identifier"][0]["value"] == "X1234567L"
