"""Smoke tests for the first-five-minutes redact/extract/FHIR example."""

from __future__ import annotations

from examples import first_five_minutes_redact_extract_fhir as example


def test_first_five_minutes_main_emits_non_empty_fhir_bundle(capsys):
    bundle = example.main()

    captured = capsys.readouterr().out
    assert "Redacted text" in captured
    assert "Extracted clinical entities" in captured
    assert "FHIR bundle" in captured
    assert "demo.patient@example.test" not in captured
    assert "212-555-0198" not in captured

    assert bundle["resourceType"] == "Bundle"
    assert bundle["type"] == "transaction"
    assert bundle["entry"]
    assert any(
        entry["resource"]["resourceType"] == "Observation" for entry in bundle["entry"]
    )
