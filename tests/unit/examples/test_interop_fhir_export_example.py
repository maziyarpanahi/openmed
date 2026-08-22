"""Smoke test for the synthetic FHIR interoperability example."""

from __future__ import annotations

import json

from examples import interop_fhir_export as example


def test_interop_fhir_export_runs_end_to_end(capsys) -> None:
    bundle = example.main()

    printed = json.loads(capsys.readouterr().out)
    assert printed == bundle
    example.validate_example_bundle_shape(bundle)

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    resources = [entry["resource"] for entry in bundle["entry"]]
    assert [resource["resourceType"] for resource in resources] == [
        "Condition",
        "MedicationStatement",
        "Observation",
    ]
    assert resources[2]["valueQuantity"] == {
        "value": 130,
        "unit": "mg/dL",
        "system": "http://unitsofmeasure.org",
        "code": "mg/dL",
    }
    assert all(
        resource["subject"]["reference"] == "Patient/synthetic-patient"
        for resource in resources
    )
