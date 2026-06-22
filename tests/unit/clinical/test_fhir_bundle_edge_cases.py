"""Regression tests for FHIR Bundle edge cases (openmed#547).

These tests lock in behaviors that are already correctly implemented but
were not explicitly verified: empty input handling and dangling reference
preservation.
"""

from openmed.clinical.exporters.fhir import to_bundle


class TestEmptyInput:
    """Verify to_bundle handles empty resource lists correctly."""

    def test_empty_resources_returns_well_formed_transaction_bundle(self):
        bundle = to_bundle([], doc_id="empty")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        assert bundle["entry"] == []

    def test_empty_resources_collection_bundle(self):
        bundle = to_bundle([], doc_id="empty", bundle_type="collection")
        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "collection"
        assert bundle["entry"] == []

    def test_empty_resources_batch_bundle(self):
        bundle = to_bundle([], doc_id="empty", bundle_type="batch")
        assert bundle["type"] == "batch"
        assert bundle["entry"] == []


class TestDanglingReferences:
    """Verify references to absent resources are preserved verbatim."""

    def test_dangling_reference_preserved_verbatim(self):
        """A reference to a resource not in the Bundle is left unchanged."""
        resources = [
            {
                "resourceType": "Condition",
                "id": "cond1",
                "subject": {"reference": "Patient/absent"},
            },
        ]
        bundle = to_bundle(resources, doc_id="doc-1")
        condition = bundle["entry"][0]["resource"]
        assert condition["subject"]["reference"] == "Patient/absent"

    def test_mixed_internal_and_dangling_references(self):
        """Internal references are rewritten; external ones are preserved."""
        resources = [
            {
                "resourceType": "Patient",
                "id": "pat1",
            },
            {
                "resourceType": "Condition",
                "id": "cond1",
                "subject": {"reference": "Patient/pat1"},  # internal
                "encounter": {"reference": "Encounter/missing"},  # dangling
                "evidence": [
                    {"detail": [{"reference": "Observation/absent"}]}  # dangling
                ],
            },
        ]
        bundle = to_bundle(resources, doc_id="mixed-refs")

        # Find the rewritten URLs
        patient_url = next(
            entry["fullUrl"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Patient"
        )

        condition = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Condition"
        )

        # Internal reference rewritten
        assert condition["subject"]["reference"] == patient_url

        # Dangling references preserved verbatim
        assert condition["encounter"]["reference"] == "Encounter/missing"
        assert (
            condition["evidence"][0]["detail"][0]["reference"] == "Observation/absent"
        )

    def test_multiple_dangling_references_same_resource(self):
        """Multiple dangling references on one resource are all preserved."""
        resources = [
            {
                "resourceType": "DiagnosticReport",
                "id": "report1",
                "subject": {"reference": "Patient/gone"},
                "performer": [{"reference": "Practitioner/absent"}],
                "result": [
                    {"reference": "Observation/missing1"},
                    {"reference": "Observation/missing2"},
                ],
            },
        ]
        bundle = to_bundle(resources, doc_id="multi-dangling")
        report = bundle["entry"][0]["resource"]

        assert report["subject"]["reference"] == "Patient/gone"
        assert report["performer"][0]["reference"] == "Practitioner/absent"
        assert report["result"][0]["reference"] == "Observation/missing1"
        assert report["result"][1]["reference"] == "Observation/missing2"
