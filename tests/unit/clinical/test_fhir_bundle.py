"""Tests for the FHIR R4 transaction Bundle assembler (OM-137)."""

import json

import pytest

from openmed.clinical.exporters.fhir import to_bundle
from openmed.clinical.exporters.fhir.bundle import _rewrite_references


def _collect_references(node):
    """Yield every ``reference`` string anywhere inside ``node``."""

    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                yield value
            else:
                yield from _collect_references(value)
    elif isinstance(node, list):
        for item in node:
            yield from _collect_references(item)


def _condition_observation_report():
    """A minimal Condition + Observation + DiagnosticReport set.

    ``DiagnosticReport.result`` points at the ``Observation`` and the
    ``Condition.evidence`` cites it too, so every reference is internal.
    """

    observation = {
        "resourceType": "Observation",
        "id": "obs1",
        "status": "final",
        "code": {"text": "Glucose"},
    }
    report = {
        "resourceType": "DiagnosticReport",
        "id": "dr1",
        "status": "final",
        "result": [{"reference": "Observation/obs1"}],
    }
    condition = {
        "resourceType": "Condition",
        "id": "cond1",
        "evidence": [{"detail": [{"reference": "Observation/obs1"}]}],
    }
    return [condition, observation, report]


class TestBundleStructure:
    def test_emits_transaction_bundle_with_one_entry_per_resource(self):
        resources = _condition_observation_report()
        bundle = to_bundle(resources, doc_id="doc-1")

        assert bundle["resourceType"] == "Bundle"
        assert bundle["type"] == "transaction"
        assert len(bundle["entry"]) == len(resources)

        for entry, resource in zip(bundle["entry"], resources):
            assert entry["fullUrl"].startswith("urn:uuid:")
            assert entry["resource"]["resourceType"] == resource["resourceType"]
            assert entry["request"] == {
                "method": "POST",
                "url": resource["resourceType"],
            }

    def test_full_urls_are_unique(self):
        bundle = to_bundle(_condition_observation_report(), doc_id="doc-1")
        full_urls = [entry["fullUrl"] for entry in bundle["entry"]]
        assert len(set(full_urls)) == len(full_urls)

    def test_collection_bundle_has_no_request_block(self):
        bundle = to_bundle(
            _condition_observation_report(),
            doc_id="doc-1",
            bundle_type="collection",
        )
        assert bundle["type"] == "collection"
        assert all("request" not in entry for entry in bundle["entry"])

    def test_batch_bundle_keeps_request_blocks(self):
        resources = _condition_observation_report()
        bundle = to_bundle(resources, doc_id="doc-1", bundle_type="batch")

        assert bundle["type"] == "batch"
        assert [entry["request"] for entry in bundle["entry"]] == [
            {"method": "POST", "url": resource["resourceType"]}
            for resource in resources
        ]

    def test_missing_resource_type_raises(self):
        with pytest.raises(ValueError):
            to_bundle([{"id": "x"}], doc_id="doc-1")

    def test_duplicate_resource_type_id_raises(self):
        resources = [
            {"resourceType": "Observation", "id": "obs1"},
            {"resourceType": "Observation", "id": "obs1"},
        ]

        with pytest.raises(
            ValueError, match="duplicate FHIR resource id: Observation/obs1"
        ):
            to_bundle(resources, doc_id="doc-1")


class TestReferenceResolution:
    def test_internal_references_rewritten_to_full_urls(self):
        resources = _condition_observation_report()
        bundle = to_bundle(resources, doc_id="doc-1")

        full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
        observation_url = next(
            entry["fullUrl"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "Observation"
        )

        report = next(
            entry["resource"]
            for entry in bundle["entry"]
            if entry["resource"]["resourceType"] == "DiagnosticReport"
        )
        assert report["result"][0]["reference"] == observation_url

        # No dangling references: every reference resolves to a fullUrl present
        # in the Bundle.
        references = list(_collect_references(bundle["entry"]))
        assert references  # sanity: there are references to resolve
        for reference in references:
            assert reference in full_urls

    def test_subject_result_encounter_all_resolved(self):
        # A richer graph exercising the three reference fields called out by
        # the issue (subject, result, encounter). The Patient/Encounter
        # fixtures are only here to anchor those references; the assembler
        # never synthesises them itself.
        resources = [
            {"resourceType": "Patient", "id": "pat1"},
            {
                "resourceType": "Encounter",
                "id": "enc1",
                "subject": {"reference": "Patient/pat1"},
            },
            {
                "resourceType": "Condition",
                "id": "cond1",
                "subject": {"reference": "Patient/pat1"},
                "encounter": {"reference": "Encounter/enc1"},
            },
            {
                "resourceType": "Observation",
                "id": "obs1",
                "subject": {"reference": "Patient/pat1"},
                "encounter": {"reference": "Encounter/enc1"},
            },
            {
                "resourceType": "DiagnosticReport",
                "id": "dr1",
                "subject": {"reference": "Patient/pat1"},
                "result": [{"reference": "Observation/obs1"}],
            },
        ]
        bundle = to_bundle(resources, doc_id="patient-graph")

        full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
        references = list(_collect_references(bundle["entry"]))
        assert len(references) == 7
        for reference in references:
            assert reference.startswith("urn:uuid:")
            assert reference in full_urls

    def test_references_to_absent_resources_are_left_untouched(self):
        # The targeted Patient is not part of the Bundle, so its reference is
        # preserved verbatim rather than rewritten.
        resources = [
            {
                "resourceType": "Condition",
                "id": "cond1",
                "subject": {"reference": "Patient/missing"},
            },
        ]
        bundle = to_bundle(resources, doc_id="doc-1")
        condition = bundle["entry"][0]["resource"]
        assert condition["subject"]["reference"] == "Patient/missing"

    def test_input_resources_not_mutated(self):
        resources = _condition_observation_report()
        snapshot = json.dumps(resources, sort_keys=True)
        to_bundle(resources, doc_id="doc-1")
        assert json.dumps(resources, sort_keys=True) == snapshot


class TestDeterminism:
    def test_output_is_byte_stable_across_runs(self):
        resources = _condition_observation_report()
        first = to_bundle(resources, doc_id="doc-1")
        second = to_bundle(resources, doc_id="doc-1")
        assert json.dumps(first) == json.dumps(second)

    def test_full_urls_seeded_by_doc_id_and_index(self):
        resources = _condition_observation_report()
        a = to_bundle(resources, doc_id="doc-A")
        b = to_bundle(resources, doc_id="doc-B")
        urls_a = [entry["fullUrl"] for entry in a["entry"]]
        urls_b = [entry["fullUrl"] for entry in b["entry"]]
        # Different document -> different urns; same document -> reproducible.
        assert urls_a != urls_b
        assert urls_a == [
            entry["fullUrl"] for entry in to_bundle(resources, doc_id="doc-A")["entry"]
        ]


def test_rewrite_references_is_pure_helper():
    ref_map = {"Observation/obs1": "urn:uuid:abc"}
    node = {"result": [{"reference": "Observation/obs1"}]}
    rewritten = _rewrite_references(node, ref_map)
    assert rewritten["result"][0]["reference"] == "urn:uuid:abc"
    # original untouched
    assert node["result"][0]["reference"] == "Observation/obs1"
