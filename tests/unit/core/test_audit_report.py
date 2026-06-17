"""Tests for deterministic de-identification audit reports."""

from __future__ import annotations

import json

from openmed.core.audit import (
    AuditReport,
    AuditSpan,
    DetectorInfo,
    recompute_repro_hash,
    verify_repro_hash,
    hash_text,
)


def _report() -> AuditReport:
    text = "Patient John Doe called 555-1234."
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
            )
        ],
        thresholds={"PERSON": 0.7},
        residual_risk={
            "projected_leakage": 0.05,
            "risk_report_record_score": 0.0,
            "risk_report": {
                "leakage_rate": 0.0,
                "reid_rate": 0.0,
                "k_min": 0,
                "singleton_records": [],
                "quasi_identifiers": [],
            },
        },
        openmed_version="1.5.5",
        manifest_hash="sha256:manifest",
        document_length=len(text),
        input_hash=hash_text(text),
        deidentified_text_hash=hash_text("Patient [NAME] called [PHONE]."),
    )


def test_report_json_round_trip_is_byte_stable_and_hash_recomputes():
    report = _report()

    payload = report.to_json()
    restored = AuditReport.from_json(payload)

    assert restored == report
    assert restored.to_json() == payload
    assert recompute_repro_hash(restored) == report.repro_hash
    assert verify_repro_hash(json.loads(payload))


def test_report_sign_verify_and_tamper_detection():
    report = _report().sign("release-key", key_id="test-key")

    assert report.verify("release-key")
    assert not report.verify("wrong-key")

    tampered = AuditReport.from_json(report.to_json())
    tampered.spans[0].canonical_label = "EMAIL"

    assert not tampered.verify("release-key")


def test_review_bundle_excludes_full_document_and_span_text():
    report = _report()
    bundle_json = report.export_review_bundle_json()

    assert "Patient John Doe called 555-1234." not in bundle_json
    assert "John Doe" not in bundle_json
    assert "Patient " in bundle_json
    assert " called 555-1234." in bundle_json
