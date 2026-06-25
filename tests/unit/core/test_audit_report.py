"""Tests for deterministic de-identification audit reports."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from openmed.core.audit import (
    AuditReport,
    AuditSignature,
    AuditSpan,
    DetectorInfo,
    hash_text,
    recompute_repro_hash,
    stable_hash,
    verify_repro_hash,
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


@pytest.mark.parametrize("key", [None, "", b""])
def test_report_sign_rejects_missing_or_empty_hmac_key(key):
    with pytest.raises(ValueError, match="non-empty HMAC release key"):
        _report().sign(key)


def test_unsigned_report_verify_returns_false_with_valid_key():
    assert not _report().verify("release-key")


@pytest.mark.parametrize("key", [None, "", b""])
def test_report_verify_rejects_missing_or_empty_hmac_key(key):
    report = _report().sign("release-key")

    with pytest.raises(ValueError, match="non-empty HMAC release key"):
        report.verify(key)


def test_review_bundle_excludes_full_document_and_span_text():
    report = _report()
    bundle_json = report.export_review_bundle_json()

    assert "Patient John Doe called 555-1234." not in bundle_json
    assert "John Doe" not in bundle_json
    assert "Patient " in bundle_json
    assert " called 555-1234." in bundle_json


def _multi_span_report(spans: list[AuditSpan]) -> AuditReport:
    text = "Patient John Doe called 555-1234 from email a@b.co."
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
        spans=spans,
        thresholds={"PERSON": 0.7, "PHONE": 0.7, "EMAIL": 0.7},
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
        deidentified_text_hash=hash_text(
            "Patient [NAME] called [PHONE] from email [EMAIL]."
        ),
    )


def _example_spans() -> list[AuditSpan]:
    return [
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
            evidence={"raw_label": "NAME"},
            context={"before": "Patient ", "after": " called 555-1234"},
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
            context={"before": "called ", "after": " from email"},
        ),
        AuditSpan(
            start=44,
            end=50,
            label="EMAIL",
            canonical_label="EMAIL",
            sources=["regex"],
            confidence=0.99,
            threshold=0.7,
            action="mask",
            surrogate="[EMAIL]",
            text_hash=hash_text("a@b.co"),
            evidence={"raw_label": "EMAIL"},
            context={"before": "email ", "after": "."},
        ),
    ]


def test_repro_hash_is_invariant_to_input_span_order():
    spans = _example_spans()
    ordered = _multi_span_report(list(spans))
    shuffled = _multi_span_report([spans[2], spans[0], spans[1]])

    assert shuffled.repro_hash == ordered.repro_hash
    assert shuffled.to_json() == ordered.to_json()


def test_review_bundle_span_order_is_invariant_to_input_span_order():
    spans = _example_spans()
    ordered = _multi_span_report(list(spans))
    shuffled = _multi_span_report([spans[1], spans[2], spans[0]])

    assert shuffled.export_review_bundle_json() == ordered.export_review_bundle_json()
    ordered_starts = [s["start"] for s in ordered.export_review_bundle()["spans"]]
    assert ordered_starts == sorted(ordered_starts)


def test_signed_report_verifies_after_round_trip_regardless_of_input_order():
    spans = _example_spans()
    signed = _multi_span_report([spans[2], spans[1], spans[0]]).sign(
        "release-key", key_id="test-key"
    )

    assert signed.verify("release-key")

    restored = AuditReport.from_json(signed.to_json())
    assert restored.verify("release-key")


def test_repro_hash_is_invariant_for_spans_colliding_on_primary_sort_key():
    # Two spans share (start, end, canonical_label, action) but differ in
    # other fields, so the primary sort key alone leaves their order ambiguous.
    base = _example_spans()[0]

    def _variant(label: str, source: str, confidence: float) -> AuditSpan:
        return AuditSpan(
            start=base.start,
            end=base.end,
            label=label,
            canonical_label=base.canonical_label,
            sources=[source],
            confidence=confidence,
            threshold=base.threshold,
            action=base.action,
            surrogate=base.surrogate,
            text_hash=hash_text(label),
            evidence={"raw_label": label},
            context=dict(base.context),
        )

    a = _variant("NAME_A", "ml", 0.91)
    b = _variant("NAME_B", "regex", 0.92)

    ordered = _multi_span_report([a, b])
    shuffled = _multi_span_report([b, a])

    assert shuffled.repro_hash == ordered.repro_hash
    assert shuffled.to_json() == ordered.to_json()
    assert shuffled.export_review_bundle_json() == ordered.export_review_bundle_json()


def _legacy_signed_report_with_unsorted_spans(key: str):
    """Build a report whose stored hash + HMAC use the legacy stored span order.

    Reproduces what an earlier OpenMed version persisted: spans hashed in their
    array order rather than the deterministic sorted order.
    """
    spans = _example_spans()
    # Stored order deliberately differs from the deterministic (start-sorted) order.
    report = _multi_span_report([spans[2], spans[0], spans[1]])

    legacy_payload = report._payload(
        include_repro_hash=False,
        include_signature=False,
        spans=list(report.spans),
    )
    report.repro_hash = stable_hash(legacy_payload)

    message = json.dumps(
        report._payload(
            include_repro_hash=True,
            include_signature=False,
            spans=list(report.spans),
        ),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    report.signature = AuditSignature(
        key_id="release",
        algorithm="HMAC-SHA256",
        value=hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest(),
    )
    return report


def test_legacy_signed_report_with_unsorted_spans_still_verifies():
    key = "release-key"
    report = _legacy_signed_report_with_unsorted_spans(key)

    # Sanity: stored order is not the deterministic order, so the new hash differs.
    assert report.recompute_repro_hash() != report.repro_hash

    assert report.verify(key)
    assert verify_repro_hash(report)
    assert not report.verify("wrong-key")
