"""Focused tests for the synthetic FHIR Bundle round-trip fidelity suite."""

from __future__ import annotations

import json

import pytest

from openmed.eval.suites.fhir_roundtrip import (
    FHIR_ROUNDTRIP_FIXTURE_PATH,
    assert_fhir_roundtrip_fidelity,
    build_fhir_bundle,
    load_fhir_roundtrip_fixtures,
    run_fhir_roundtrip_suite,
)


def test_clean_fhir_roundtrip_preserves_resources_codes_and_references() -> None:
    report = run_fhir_roundtrip_suite()

    assert report.fixture_count == 2
    assert report.metrics["resource_match_rate"] == 1.0
    assert report.metrics["code_preservation_rate"] == 1.0
    assert report.metrics["dangling_reference_count"] == 0
    assert report.metrics["internal_reference_resolution_rate"] == 1.0
    assert report.metrics["passed"] is True
    assert report.metadata["synthetic"] is True
    assert report.metadata["contains_real_phi"] is False


def test_transaction_and_batch_fixtures_emit_request_blocks() -> None:
    fixtures = load_fhir_roundtrip_fixtures()

    assert {fixture.bundle_type for fixture in fixtures} == {"transaction", "batch"}
    for fixture in fixtures:
        bundle = build_fhir_bundle(fixture)
        assert bundle["type"] == fixture.bundle_type
        assert all(
            entry["request"]
            == {
                "method": "POST",
                "url": entry["resource"]["resourceType"],
            }
            for entry in bundle["entry"]
        )
        assert all(
            reference.startswith("urn:uuid:")
            for entry in bundle["entry"]
            for reference in _references(entry["resource"])
        )


def test_missing_internal_reference_is_counted_and_can_raise(tmp_path) -> None:
    source = tmp_path / "fhir_roundtrip.jsonl"
    fixture = json.loads(
        FHIR_ROUNDTRIP_FIXTURE_PATH.read_text(encoding="utf-8").splitlines()[0]
    )
    fixture["spans"][1]["references"][0]["target"] = "Patient/synthetic-missing"
    source.write_text(json.dumps(fixture) + "\n", encoding="utf-8")

    report = run_fhir_roundtrip_suite(fixture_path=source)

    assert report.metrics["dangling_reference_count"] == 1
    assert report.metrics["passed"] is False
    with pytest.raises(AssertionError, match="dangling_reference_count"):
        assert_fhir_roundtrip_fidelity(report)


def _references(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                yield value
            else:
                yield from _references(value)
    elif isinstance(node, list):
        for value in node:
            yield from _references(value)
