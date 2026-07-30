"""Tests for deterministic FHIR reference helpers."""

from openmed.clinical.exporters.fhir import (
    deterministic_fullurl as exported_deterministic_fullurl,
)
from openmed.clinical.exporters.fhir import to_bundle
from openmed.clinical.exporters.fhir.references import deterministic_fullurl


def test_helper_matches_bundle_fullurl():
    """Helper should reproduce the Bundle assembler fullUrl exactly."""

    bundle = to_bundle(
        [{"resourceType": "Patient", "id": "pat1"}],
        doc_id="doc-1",
    )

    assert bundle["entry"][0]["fullUrl"] == deterministic_fullurl(
        "doc-1",
        0,
    )


def test_helper_is_reexported_from_fhir_package():
    assert exported_deterministic_fullurl("doc-1", 0) == deterministic_fullurl(
        "doc-1",
        0,
    )


def test_helper_preserves_legacy_uuid_seed():
    assert (
        deterministic_fullurl("doc-1", 0)
        == "urn:uuid:44a61302-7fe7-537c-a5a5-965a5e3ef526"
    )


def test_precomputed_urn_reference_survives_assembly():
    """Exporter-provided deterministic URNs should remain unchanged."""

    obs_ref = deterministic_fullurl("doc-1", 0)

    resources = [
        {
            "resourceType": "Observation",
            "id": "obs1",
        },
        {
            "resourceType": "DiagnosticReport",
            "id": "dr1",
            "result": [{"reference": obs_ref}],
        },
    ]

    bundle = to_bundle(resources, doc_id="doc-1")

    report = bundle["entry"][1]["resource"]

    assert report["result"][0]["reference"] == obs_ref


def test_bundle_output_remains_byte_stable():
    resources = [
        {
            "resourceType": "Observation",
            "id": "obs1",
            "status": "final",
        },
        {
            "resourceType": "DiagnosticReport",
            "id": "dr1",
            "status": "final",
            "result": [{"reference": "Observation/obs1"}],
        },
    ]

    assert to_bundle(resources, doc_id="doc-1") == {
        "resourceType": "Bundle",
        "type": "transaction",
        "entry": [
            {
                "fullUrl": "urn:uuid:44a61302-7fe7-537c-a5a5-965a5e3ef526",
                "resource": {
                    "resourceType": "Observation",
                    "id": "obs1",
                    "status": "final",
                },
                "request": {"method": "POST", "url": "Observation"},
            },
            {
                "fullUrl": "urn:uuid:b410d1fe-83a1-5a60-955d-51919aab54ae",
                "resource": {
                    "resourceType": "DiagnosticReport",
                    "id": "dr1",
                    "status": "final",
                    "result": [
                        {"reference": ("urn:uuid:44a61302-7fe7-537c-a5a5-965a5e3ef526")}
                    ],
                },
                "request": {"method": "POST", "url": "DiagnosticReport"},
            },
        ],
    }
