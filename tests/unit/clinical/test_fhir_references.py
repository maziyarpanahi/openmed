"""Tests for deterministic FHIR reference helpers."""

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