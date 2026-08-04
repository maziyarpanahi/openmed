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


def test_report_from_json_rejects_malformed_payload():
    with pytest.raises(ValueError, match="Invalid JSON for AuditReport"):
        AuditReport.from_json("{")


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
    bundle = report.export_review_bundle()
    bundle_json = report.export_review_bundle_json()

    assert "Patient John Doe called 555-1234." not in bundle_json
    assert "John Doe" not in bundle_json
    assert "Patient " not in bundle_json
    assert " called 555-1234." not in bundle_json
    assert bundle["spans"][0]["context"] == {
        "before": {
            "start": 0,
            "end": 8,
            "length": 8,
            "text_hash": hash_text("Patient "),
        },
        "after": {
            "start": 16,
            "end": 33,
            "length": 17,
            "text_hash": hash_text(" called 555-1234."),
        },
    }


def test_manual_raw_context_fails_closed_at_every_serialization_boundary():
    before = "123-45-6789 "
    after = " a@b.co"
    report = _multi_span_report(_example_spans())
    span = report.spans[1]
    span.context = {"before": before, "after": after}

    expected = {
        "before": {
            "start": span.start - len(before),
            "end": span.start,
            "length": len(before),
            "text_hash": hash_text(before),
        },
        "after": {
            "start": span.end,
            "end": span.end + len(after),
            "length": len(after),
            "text_hash": hash_text(after),
        },
    }
    report_dict = report.to_dict()
    report_json = report.to_json()
    review_bundle = report.export_review_bundle()
    combined = json.dumps(
        [report_dict, json.loads(report_json), review_bundle], sort_keys=True
    )

    assert before not in combined
    assert after not in combined
    assert report_dict["spans"][1]["context"] == expected
    assert review_bundle["spans"][1]["context"] == expected


def test_nested_raw_or_malformed_context_fields_are_dropped():
    report = _report()
    span = report.spans[0]
    safe_before = dict(span.context["before"])
    span.context = {
        "before": {**safe_before, "raw": "123-45-6789"},
        "after": {
            "start": span.end,
            "end": span.end + 8,
            "length": 8,
            "text_hash": "a@b.co",
            "raw": "a@b.co",
        },
        "unexpected": "4111 1111 1111 1111",
    }

    context = span.to_dict()["context"]
    serialized = report.to_json() + report.export_review_bundle_json()

    assert context == {"before": safe_before}
    assert "123-45-6789" not in serialized
    assert "a@b.co" not in serialized
    assert "4111 1111 1111 1111" not in serialized


def test_detailed_risk_shapes_are_reduced_to_aggregate_audit_evidence():
    report = _report()
    report.residual_risk = {
        "source_path": "/private/raw/records-canary.csv",
        "risk_report": {
            "leakage_rate": 0.25,
            "reid_rate": 0.5,
            "k_min": 1,
            "singleton_records": [
                {
                    "record_index": 7,
                    "record_id": "record-id-canary",
                    "effective_k": 1,
                    "quasi_identifier_key": [
                        {"category": "geography", "values": ["raw-qi-canary"]}
                    ],
                }
            ],
            "quasi_identifiers": [
                {
                    "record_index": 7,
                    "record_id": "record-id-canary",
                    "category": "geography",
                    "value": "raw-qi-canary",
                    "normalized_value": "normalized-qi-canary",
                    "source": "field",
                    "start": 1,
                    "end": 2,
                }
            ],
        },
        "kanon": {
            "record_count": 2,
            "quasi_identifiers": ["city"],
            "sensitive_attributes": ["condition"],
            "k": 1,
            "class_count": 2,
            "class_size_distribution": [[1, 2]],
            "equivalence_classes": [
                {
                    "key": ["class-key-canary"],
                    "size": 1,
                    "members": [7],
                    "l_diversity": {"condition": {"distinct": 1, "entropy": 0.0}},
                    "t_closeness": {"condition": 0.5},
                }
            ],
            "l": {"condition": 1},
            "l_diversity": {"condition": {"min_distinct": 1, "min_entropy": 0.0}},
            "t_closeness": {"condition": 0.5},
            "l_metric": "distinct",
            "t_distance": "variational",
        },
        "enforcement": {
            "schema_version": 1,
            "record_count": 2,
            "released_count": 1,
            "suppressed_count": 1,
            "target_k": 2,
            "target_l": 1,
            "target_t": 1.0,
            "quasi_identifiers": ["city"],
            "sensitive_attributes": ["condition"],
            "records": [{"city": "released-record-canary"}],
            "kanon": {
                "record_count": 1,
                "quasi_identifiers": ["city"],
                "sensitive_attributes": ["condition"],
                "k": 1,
                "class_count": 1,
                "class_size_distribution": [[1, 1]],
                "equivalence_classes": [
                    {
                        "key": ["enforced-key-canary"],
                        "size": 1,
                        "members": [0],
                        "l_diversity": {},
                        "t_closeness": {},
                    }
                ],
                "l": {"condition": 0},
                "l_diversity": {},
                "t_closeness": {},
                "l_metric": "distinct",
                "t_distance": "variational",
            },
            "suppressed_records": [
                {
                    "offset": 1,
                    "record_hash": "suppressed-hash-canary",
                    "reason": "privacy_class_violation",
                }
            ],
            "generalization": {
                "levels": {"city": {"level": 1, "name": "region", "loss": 0.5}},
                "information_loss": 0.5,
                "nodes_evaluated": 2,
            },
            "bounds": {
                "max_reidentification_upper_bound": 1.0,
                "target_reidentification_upper_bound": 0.5,
                "numeric_self_check": {
                    "passed": False,
                    "identity_bound_violations": [{"record_index": 0, "bound": 1.0}],
                    "l_diversity_satisfied": True,
                    "t_closeness_satisfied": True,
                },
                "per_record": [
                    {
                        "record_index": 0,
                        "record_hash": "per-record-hash-canary",
                    }
                ],
            },
        },
    }

    payload = report.to_dict()["residual_risk"]
    serialized = json.dumps(payload, sort_keys=True)

    for canary in (
        "/private/raw/records-canary.csv",
        "record-id-canary",
        "raw-qi-canary",
        "normalized-qi-canary",
        "class-key-canary",
        "released-record-canary",
        "enforced-key-canary",
        "suppressed-hash-canary",
        "per-record-hash-canary",
    ):
        assert canary not in serialized
    for forbidden_key in (
        '"normalized_value"',
        '"record_id"',
        '"equivalence_classes"',
        '"members"',
        '"records"',
        '"suppressed_records"',
        '"per_record"',
        '"source_path_hash"',
        '"record_id_hash"',
        '"record_hash"',
    ):
        assert forbidden_key not in serialized

    assert payload["risk_report"]["artifact"] == "deidentification_risk_summary"
    assert payload["risk_report"]["singleton_record_count"] == 1
    assert payload["risk_report"]["quasi_identifier_categories"] == {"geography": 1}
    assert payload["kanon"]["equivalence_class_detail_count"] == 1
    assert payload["kanon"]["l"] == {"condition": 1}
    assert (
        payload["enforcement"]["bounds"]["numeric_self_check"][
            "identity_bound_violation_count"
        ]
        == 1
    )


def test_anonymization_result_is_serialized_through_its_safe_summary():
    from openmed.risk import AnonymityPolicy, anonymize_release

    rows = [
        {
            "patient_id": "audit-patient-canary-a",
            "age": 30,
            "zip": "audit-raw-qi-canary",
            "condition": "audit-sensitive-canary-a",
            "site_code": "site-a",
            "source_batch": "batch-a",
        },
        {
            "patient_id": "audit-patient-canary-b",
            "age": 30,
            "zip": "audit-raw-qi-canary",
            "condition": "audit-sensitive-canary-b",
            "site_code": "site-a",
            "source_batch": "batch-a",
        },
    ]
    policy = AnonymityPolicy(
        quasi_identifiers=("age", "zip"),
        sensitive_attributes=("condition",),
        non_sensitive_attributes=("site_code",),
        excluded_attributes=("source_batch",),
        privacy_unit="patient_id",
        target_k=2,
    )
    result = anonymize_release(rows, policy)
    report = _report()
    report.residual_risk = {"release": result}

    payload = report.to_dict()["residual_risk"]["release"]
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["artifact"] == "deidentification_anonymization_summary"
    assert payload["after"]["meets_policy"] is True
    assert payload["policy"]["non_sensitive_attributes"] == ["site_code"]
    assert payload["policy"]["excluded_attributes"] == ["source_batch"]
    assert payload["released_schema_digest"] == result.released_schema_digest
    assert payload["generalization"]["search"]["complete"] is False
    assert payload["generalization"]["search"]["optimum_proven"] is True
    assert (
        payload["generalization"]["search"]["suppression_subsets_evaluated"]
        == result.generalization.suppression_subsets_evaluated
    )
    assert (
        payload["generalization"]["search"]["max_suppression_subsets"]
        == result.generalization.max_suppression_subsets
    )
    for stage in ("before", "after"):
        safe_k = payload[stage]["k_anonymity"]
        source_k = getattr(result, stage).to_dict()["k_anonymity"]
        assert safe_k["k_violating_class_count"] == source_k["k_violating_class_count"]
        assert safe_k["l_violating_class_count"] == source_k["l_violating_class_count"]
        assert safe_k["t_violating_class_count"] == source_k["t_violating_class_count"]
    assert '"records"' not in serialized
    for canary in (
        "audit-patient-canary-a",
        "audit-patient-canary-b",
        "audit-raw-qi-canary",
        "audit-sensitive-canary-a",
        "audit-sensitive-canary-b",
    ):
        assert canary not in serialized


def test_detailed_risk_category_is_allow_listed_without_echoing_forged_values():
    report = _report()
    report.residual_risk = {
        "risk_report": {
            "record_count": 1,
            "leakage_rate": 1.0,
            "reid_rate": 1.0,
            "k_min": 1,
            "singleton_records": [{"record_index": 0}],
            "quasi_identifiers": [
                {
                    "category": "forged-category-sensitive-canary",
                    "value": "raw-value-sensitive-canary",
                }
            ],
        }
    }

    payload = report.to_dict()["residual_risk"]["risk_report"]
    serialized = json.dumps(payload, sort_keys=True)

    assert payload["quasi_identifier_categories"] == {"unknown": 1}
    assert "forged-category-sensitive-canary" not in serialized
    assert "raw-value-sensitive-canary" not in serialized


def test_serialized_release_summary_drops_untrusted_metadata_strings():
    from openmed.risk import AnonymityPolicy, anonymize_release

    result = anonymize_release(
        [{"patient_id": "a", "age": 30}, {"patient_id": "b", "age": 30}],
        AnonymityPolicy(
            quasi_identifiers=("age",),
            privacy_unit="patient_id",
            target_k=2,
        ),
    )
    payload = result.to_safe_dict()
    payload["before"]["quasi_identifiers"].append("Alice Canary")
    payload["generalization"]["levels"][0]["name"] = "Hierarchy Canary Name"
    report = _report()
    report.residual_risk = {"release": payload}

    serialized = json.dumps(report.to_dict()["residual_risk"], sort_keys=True)

    assert "Alice Canary" not in serialized
    assert "Hierarchy Canary Name" not in serialized


def test_release_audit_sanitizer_rejects_forged_closed_vocabulary_strings():
    from openmed.risk import AnonymityPolicy, anonymize_release

    result = anonymize_release(
        [{"patient_id": "a", "age": 30}, {"patient_id": "b", "age": 30}],
        AnonymityPolicy(
            quasi_identifiers=("age",),
            privacy_unit="patient_id",
            target_k=2,
        ),
    )
    payload = result.to_safe_dict()
    canary = "closed-vocabulary-canary"
    payload["policy"]["l_metric"] = canary
    payload["policy"]["t_distance"] = canary
    payload["before"]["attribute_disclosure"] = [
        {
            "attribute": "age",
            "l_diversity": {"metric": canary},
            "t_closeness": {"distance": canary},
        }
    ]
    payload["generalization"]["search"]["strategy"] = canary
    payload["generalization"]["search"]["suppression_strategy"] = canary
    report = _report()
    report.residual_risk = {"release": payload}

    serialized = json.dumps(report.to_dict()["residual_risk"], sort_keys=True)

    assert canary not in serialized


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("start", "0"),
        ("start", 0.0),
        ("start", False),
        ("start", -1),
        ("end", "8"),
        ("end", 8.0),
        ("end", False),
        ("length", "8"),
        ("length", 8.0),
        ("length", False),
        ("length", 7),
        ("text_hash", "sha256:not-a-digest"),
    ],
)
def test_context_descriptor_validation_rejects_malformed_fields(field, value):
    report = _report()
    span = report.spans[0]
    descriptor = dict(span.context["before"])
    descriptor[field] = value
    span.context = {"before": descriptor}

    assert span.to_dict()["context"] == {}
    assert report.to_dict()["spans"][0]["context"] == {}
    assert report.export_review_bundle()["spans"][0]["context"] == {}


def test_context_descriptor_outside_document_is_dropped_by_report_sinks():
    report = _report()
    span = report.spans[0]
    after = "x" * (report.document_length - span.end + 1)
    span.context = {
        "after": {
            "start": span.end,
            "end": report.document_length + 1,
            "length": len(after),
            "text_hash": hash_text(after),
        }
    }

    # A standalone span cannot know the enclosing document bound.
    assert span.to_dict()["context"]["after"]["end"] == report.document_length + 1
    assert report.to_dict()["spans"][0]["context"] == {}
    assert report.export_review_bundle()["spans"][0]["context"] == {}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("start", "8", "start"),
        ("start", 8.0, "start"),
        ("start", True, "start"),
        ("start", -1, "start"),
        ("end", "16", "end"),
        ("end", 16.0, "end"),
        ("end", True, "end"),
        ("end", -1, "end"),
    ],
)
def test_deserialized_span_offsets_must_be_non_negative_integers(
    field,
    value,
    error,
):
    payload = _report().to_dict()
    payload["spans"][0][field] = value

    with pytest.raises(ValueError, match=f"audit span {error}"):
        AuditReport.from_dict(payload)


@pytest.mark.parametrize("value", ["33", 33.0, True, -1])
def test_deserialized_document_length_must_be_a_non_negative_integer(value):
    payload = _report().to_dict()
    payload["document_length"] = value

    with pytest.raises(ValueError, match="document_length"):
        AuditReport.from_dict(payload)


@pytest.mark.parametrize(("start", "end"), [(16, 8), (8, 34)])
def test_deserialized_span_bounds_must_be_ordered_and_within_document(start, end):
    payload = _report().to_dict()
    payload["spans"][0]["start"] = start
    payload["spans"][0]["end"] = end

    with pytest.raises(ValueError, match="audit span (end|offsets)"):
        AuditReport.from_dict(payload)


def test_mutated_span_offsets_fail_closed_at_serialization_boundaries():
    report = _report()
    report.spans[0].start = "8"
    report.spans[0].context = {"before": "123-45-6789"}

    with pytest.raises(ValueError, match="audit span start"):
        report.spans[0].to_dict()
    with pytest.raises(ValueError, match="audit span start"):
        report.to_dict()
    with pytest.raises(ValueError, match="audit span start"):
        report.export_review_bundle()


def test_report_from_json_rejects_non_object_payload():
    with pytest.raises(ValueError, match="must contain an object"):
        AuditReport.from_json("[]")


def test_deserialized_legacy_raw_context_is_safe_but_invalidates_integrity():
    key = "release-key"
    report = _report()
    legacy_payload = report.to_dict()
    raw_context = {
        "before": "Patient ",
        "after": " called 555-1234.",
    }
    legacy_payload["spans"][0]["context"] = raw_context
    legacy_hash_payload = dict(legacy_payload)
    legacy_hash_payload.pop("repro_hash")
    legacy_hash_payload.pop("signature")
    legacy_payload["repro_hash"] = stable_hash(legacy_hash_payload)
    legacy_signing_payload = dict(legacy_payload)
    legacy_signing_payload.pop("signature")
    message = json.dumps(
        legacy_signing_payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    legacy_payload["signature"] = {
        "key_id": "legacy",
        "algorithm": "HMAC-SHA256",
        "value": hmac.new(key.encode("utf-8"), message, hashlib.sha256).hexdigest(),
    }

    restored = AuditReport.from_dict(legacy_payload)
    serialized = json.dumps(
        [restored.to_dict(), restored.export_review_bundle()], sort_keys=True
    )

    assert raw_context["before"] not in serialized
    assert raw_context["after"] not in serialized
    assert not restored.repro_hash_matches()
    assert not restored.verify(key)


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

    payload = report.to_json()
    assert [span["start"] for span in json.loads(payload)["spans"]] == [44, 8, 24]
    assert verify_repro_hash(json.loads(payload))
    assert AuditReport.from_json(payload).verify(key)
