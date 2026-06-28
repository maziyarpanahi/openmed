"""Tests for FHIR Provenance and AuditEvent export from audit reports."""

from __future__ import annotations

import json

import pytest

from openmed.clinical.exporters.fhir import (
    to_audit_event,
    to_bundle,
    to_provenance,
)
from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text


def _signed_report() -> AuditReport:
    original = "Patient John Doe called 555-1234."
    redacted = "Patient [NAME] called [PHONE]."
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={
            "method": "mask",
            "confidence_threshold": 0.7,
            "language": "en",
        },
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="unit-test-model",
                model_format="transformers",
                commit="abc123",
            )
        ],
        safety_sweep={
            "source": "safety_sweep",
            "patterns_version": "safety-sweep-v1",
            "spans_added": 0,
        },
        spans=[
            AuditSpan(
                start=8,
                end=16,
                label="NAME",
                canonical_label="PERSON",
                sources=["ml"],
                confidence=0.95,
                threshold=0.7,
                action="mask",
                surrogate="[NAME]",
                text_hash=hash_text("John Doe"),
                evidence={"raw_label": "NAME", "model_id": "unit-test-model"},
                context={"before": "Patient ", "after": " called 555-1234."},
            ),
            AuditSpan(
                start=24,
                end=32,
                label="PHONE",
                canonical_label="PHONE",
                sources=["regex"],
                confidence=0.99,
                threshold=0.7,
                action="mask",
                surrogate="[PHONE]",
                text_hash=hash_text("555-1234"),
                evidence={"raw_label": "PHONE"},
                context={"before": "called ", "after": "."},
            ),
        ],
        thresholds={"PERSON": 0.7, "PHONE": 0.7},
        residual_risk={
            "projected_leakage": 0.05,
            "risk_report_record_score": 0.0,
            "risk_report": {
                "leakage_rate": 0.0,
                "reid_rate": 0.0,
                "k_min": 0,
                "singleton_records": [],
                "quasi_identifiers": ["PERSON", "PHONE"],
            },
        },
        openmed_version="1.6.0",
        manifest_hash="sha256:manifest",
        document_length=len(original),
        input_hash=hash_text(original),
        deidentified_text_hash=hash_text(redacted),
    ).sign("release-key", key_id="unit-test")


def _detail_map(resource: dict) -> dict[str, str]:
    return {
        item["type"]: item["valueString"] for item in resource["entity"][0]["detail"]
    }


def test_to_provenance_references_targets_and_repro_hash():
    report = _signed_report()

    provenance = to_provenance(
        report,
        ["Condition/cond1", {"reference": "Observation/obs1", "display": "ignored"}],
    )

    assert provenance["resourceType"] == "Provenance"
    assert provenance["target"] == [
        {"reference": "Condition/cond1"},
        {"reference": "Observation/obs1"},
    ]
    assert provenance["recorded"].endswith("Z")
    assert provenance["activity"]["coding"][0]["code"] == "de-identify"

    agent = provenance["agent"][0]
    assert agent["who"]["identifier"]["value"] == "openmed"
    assert agent["who"]["display"] == "openmed 1.6.0"

    entity_identifier = provenance["entity"][0]["what"]["identifier"]
    assert entity_identifier["value"] == report.repro_hash


def test_to_audit_event_describes_deidentification_outcome_and_risk_details():
    report = _signed_report()

    audit_event = to_audit_event(report)
    details = _detail_map(audit_event)

    assert audit_event["resourceType"] == "AuditEvent"
    assert audit_event["type"]["code"] == "de-identification"
    assert {coding["code"] for coding in audit_event["subtype"]} == {
        "de-identify",
        "transform",
    }
    assert audit_event["action"] == "E"
    assert audit_event["outcome"] == "0"
    assert audit_event["agent"][0]["who"]["identifier"]["value"] == "openmed"
    assert audit_event["source"]["observer"]["display"] == "openmed 1.6.0"

    assert details["openmed.repro_hash"] == report.repro_hash
    assert details["openmed.span_labels"] == "PERSON,PHONE"
    assert details["openmed.span_count"] == "2"
    assert details["openmed.residual_risk.projected_leakage"] == "0.05"
    assert details["openmed.residual_risk.risk_report.reid_rate"] == "0.0"
    assert details["openmed.residual_risk.risk_report.quasi_identifiers_count"] == "2"


def test_resources_do_not_embed_raw_phi_or_span_text():
    report = _signed_report()

    payload = json.dumps(
        [
            to_provenance(report, ["Condition/cond1"]),
            to_audit_event(report),
        ],
        sort_keys=True,
    )

    assert "John Doe" not in payload
    assert "555-1234" not in payload
    assert "Patient " not in payload
    assert "called " not in payload
    assert "[NAME]" not in payload
    assert "[PHONE]" not in payload
    assert hash_text("John Doe") in payload
    assert "PERSON" in payload


def test_resources_assemble_cleanly_into_bundle():
    report = _signed_report()
    condition = {
        "resourceType": "Condition",
        "id": "cond1",
        "code": {"text": "redacted condition"},
    }
    provenance = to_provenance(report, ["Condition/cond1"])
    audit_event = to_audit_event(report)

    bundle = to_bundle([condition, provenance, audit_event], doc_id="doc-1")

    full_urls = {entry["fullUrl"] for entry in bundle["entry"]}
    condition_url = bundle["entry"][0]["fullUrl"]
    bundled_provenance = bundle["entry"][1]["resource"]

    assert bundled_provenance["target"] == [{"reference": condition_url}]
    assert len(bundle["entry"]) == 3
    assert all(entry["fullUrl"] in full_urls for entry in bundle["entry"])


def test_to_provenance_requires_target_references():
    with pytest.raises(ValueError, match="at least one target reference"):
        to_provenance(_signed_report(), [])
