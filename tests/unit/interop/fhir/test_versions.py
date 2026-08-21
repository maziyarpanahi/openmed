"""Focused tests for the explicit FHIR R4/R5 exchange boundary."""

from __future__ import annotations

import pytest

from openmed.interop.fhir import (
    FHIRVersion,
    FHIRVersionAdapter,
    UnsupportedFHIRFieldError,
    convert_resource,
    parse_fhir_version,
)


def _r4_medication() -> dict:
    return {
        "resourceType": "MedicationStatement",
        "id": "med-original",
        "status": "active",
        "medicationCodeableConcept": {
            "coding": [
                {
                    "system": "http://www.nlm.nih.gov/research/umls/rxnorm",
                    "code": "860975",
                }
            ]
        },
        "subject": {"reference": "Patient/patient-original"},
        "reasonCode": [{"text": "synthetic indication"}],
    }


def test_version_adapter_round_trips_openmed_medication_subset_without_loss():
    source = _r4_medication()

    r5 = convert_resource(source, "4.0.1", "5.0.0")
    restored = convert_resource(r5, FHIRVersion.R5, FHIRVersion.R4)

    assert r5["medication"] == {"concept": source["medicationCodeableConcept"]}
    assert r5["reason"] == [{"concept": source["reasonCode"][0]}]
    assert restored == source
    assert source["medicationCodeableConcept"] in r5["medication"].values()


def test_unsupported_cross_version_field_reports_resource_path():
    resource = {
        "resourceType": "Observation",
        "id": "observation-1",
        "status": "final",
        "code": {"text": "synthetic"},
        "unsupportedCrossVersionField": {"value": "not silently dropped"},
    }

    with pytest.raises(
        UnsupportedFHIRFieldError, match="Observation\\.unsupportedCrossVersionField"
    ) as exc:
        convert_resource(resource, FHIRVersion.R4, FHIRVersion.R5)

    assert exc.value.path == "Observation.unsupportedCrossVersionField"


def test_adapter_accepts_from_to_aliases_and_does_not_mutate_input():
    source = {
        "resourceType": "Patient",
        "id": "patient-1",
        "name": [{"text": "Synthetic"}],
    }
    adapter = FHIRVersionAdapter("R4", "R5")

    converted = adapter.convert(source, from_version="R4", to_version="R5")

    assert converted == source
    assert converted is not source
    assert parse_fhir_version("FHIR R5") is FHIRVersion.R5
