"""Tests for de-identification span provenance and audit reports."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

from openmed.core.audit import AuditReport, verify_repro_hash
from openmed.core.pii import deidentify
from openmed.core.safety_sweep import SAFETY_SWEEP_SOURCE
from openmed.processing.outputs import EntityPrediction, PredictionResult


def _prediction(text: str) -> PredictionResult:
    return PredictionResult(
        text=text,
        entities=[
            EntityPrediction(
                text="John Doe",
                label="NAME",
                start=text.index("John Doe"),
                end=text.index("John Doe") + len("John Doe"),
                confidence=0.95,
                metadata={"detector": "unit"},
            )
        ],
        model_name="unit-test-model",
        timestamp=datetime.now().isoformat(),
    )


@patch("openmed.core.pii.extract_pii")
def test_deidentify_audit_report_records_ml_and_safety_sweep_provenance(mock_extract):
    text = "Patient John Doe emailed jane.patient@example.com."
    mock_extract.side_effect = lambda *args, **kwargs: _prediction(text)

    report = deidentify(text, method="mask", audit=True)

    assert isinstance(report, AuditReport)
    assert report.policy == "hipaa_safe_harbor"
    assert report.resolved_profile["method"] == "mask"
    assert report.resolved_profile["confidence_threshold"] == 0.7
    assert report.safety_sweep["enabled"] is True
    assert report.safety_sweep["spans_added"] == 1
    assert verify_repro_hash(report)

    by_label = {span.label: span for span in report.spans}
    assert by_label["NAME"].sources == ["ml"]
    assert by_label["NAME"].canonical_label == "PERSON"
    assert by_label["NAME"].threshold == 0.7
    assert by_label["NAME"].surrogate == "[NAME]"

    email_span = by_label["email"]
    assert email_span.sources == [SAFETY_SWEEP_SOURCE]
    assert email_span.canonical_label == "EMAIL"
    assert email_span.evidence["metadata"]["patterns_version"] == "safety-sweep-v1"
    assert "jane.patient@example.com" not in report.export_review_bundle_json()


@patch("openmed.core.pii.extract_pii")
def test_audit_repro_hash_is_stable_for_identical_inputs(mock_extract):
    text = "Patient John Doe emailed jane.patient@example.com."
    mock_extract.side_effect = lambda *args, **kwargs: _prediction(text)

    first = deidentify(text, method="mask", audit=True)
    second = deidentify(text, method="mask", audit=True)

    assert first.repro_hash == second.repro_hash
    assert first.to_json() == second.to_json()


@patch("openmed.core.pii.extract_pii")
def test_regular_deidentify_keeps_provenance_on_entities(mock_extract):
    text = "Patient John Doe."
    mock_extract.return_value = _prediction(text)

    result = deidentify(text, method="mask", use_safety_sweep=False)

    assert result.audit_report is None
    assert result.pii_entities[0].canonical_label == "PERSON"
    assert result.pii_entities[0].sources == ["ml"]
    assert result.pii_entities[0].evidence["model_id"] == "unit-test-model"
    assert result.pii_entities[0].action == "mask"
    assert result.pii_entities[0].surrogate == "[NAME]"
