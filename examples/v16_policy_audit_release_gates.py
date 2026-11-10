#!/usr/bin/env python3
"""v1.6 policy, audit, risk, and release-evidence walkthrough.

This example is intentionally synthetic and offline-friendly. It constructs a
small de-identification result directly so reviewers can inspect the v1.6
contracts without downloading a model on first run.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logging.getLogger("openmed.core.models").setLevel(logging.ERROR)

from openmed.core.audit import (
    AuditReport,
    AuditSpan,
    DetectorInfo,
    hash_text,
    manifest_hash,
    verify_repro_hash,
)
from openmed.core.labels import PERSON, PHONE
from openmed.core.pii import DeidentificationResult, PIIEntity
from openmed.core.policy import list_policies, load_policy
from openmed.core.redaction_preview import redaction_preview
from openmed.core.redaction_strength import select_minimum_necessary
from openmed.core.schemas.span import OpenMedSpan, hmac_text_hash
from openmed.eval.leakage_heatmap import compute_leakage_heatmap
from openmed.risk.kanon import kanon_report

SYNTHETIC_TEXT = "Patient Casey Example called 212-555-0198 about metformin follow-up."
DEIDENTIFIED_TEXT = "Patient [PERSON] called [PHONE] about metformin follow-up."
AUDIT_KEY = "example-release-key-not-for-production"
SPAN_HASH_KEY = "example-span-hmac-key-not-for-production"


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)


def build_result() -> DeidentificationResult:
    """Build a representative v1.6 de-identification result."""
    person_start, person_end = _span(SYNTHETIC_TEXT, "Casey Example")
    phone_start, phone_end = _span(SYNTHETIC_TEXT, "212-555-0198")
    return DeidentificationResult(
        original_text=SYNTHETIC_TEXT,
        deidentified_text=DEIDENTIFIED_TEXT,
        pii_entities=[
            PIIEntity(
                text="Casey Example",
                label=PERSON,
                start=person_start,
                end=person_end,
                confidence=0.99,
                redacted_text="[PERSON]",
                canonical_label=PERSON,
                sources=["model", "safety_sweep"],
                threshold=0.7,
                action="mask",
            ),
            PIIEntity(
                text="212-555-0198",
                label=PHONE,
                start=phone_start,
                end=phone_end,
                confidence=1.0,
                redacted_text="[PHONE]",
                canonical_label=PHONE,
                sources=["safety_sweep"],
                threshold=0.7,
                action="mask",
            ),
        ],
        method="mask",
        timestamp=datetime(2026, 7, 1, tzinfo=timezone.utc),
        mapping={"[PERSON]": "Casey Example", "[PHONE]": "212-555-0198"},
        metadata={"policy": "hipaa_safe_harbor", "example": "v1.6"},
    )


def build_audit_report(result: DeidentificationResult) -> AuditReport:
    """Build and sign a deterministic audit report for the synthetic result."""
    profile = load_policy("hipaa_safe_harbor")
    spans = [
        AuditSpan(
            start=entity.start or 0,
            end=entity.end or 0,
            label=entity.label,
            canonical_label=entity.canonical_label or entity.label,
            sources=list(entity.sources),
            confidence=entity.confidence,
            threshold=entity.threshold or 0.0,
            action=entity.action or result.method,
            surrogate=entity.redacted_text,
            text_hash=hash_text(entity.text),
            evidence=entity.evidence,
            context={},
        )
        for entity in result.pii_entities
    ]
    report = AuditReport(
        policy=profile.name,
        resolved_profile=profile.to_dict(),
        detectors=[
            DetectorInfo(
                source="example",
                model_id="synthetic-offline-detector",
                model_format="fixture",
            )
        ],
        safety_sweep={"enabled": True, "matches": 1},
        spans=spans,
        thresholds={PERSON: 0.7, PHONE: 0.7},
        residual_risk={"critical_leakage": 0, "review_required": False},
        openmed_version="1.6.0+example",
        manifest_hash=manifest_hash(),
        document_length=len(result.original_text),
        input_hash=hash_text(result.original_text),
        deidentified_text_hash=hash_text(result.deidentified_text),
    )
    return report.sign(AUDIT_KEY, key_id="local-demo")


def build_span_contract(result: DeidentificationResult) -> OpenMedSpan:
    """Show the canonical span schema used by the policy pipeline."""
    entity = result.pii_entities[0]
    return OpenMedSpan(
        doc_id="synthetic-note-001",
        start=entity.start or 0,
        end=entity.end or 0,
        text_hash=hmac_text_hash(entity.text, SPAN_HASH_KEY),
        entity_type=entity.label,
        canonical_label=entity.canonical_label or entity.label,
        score=entity.confidence,
        detector="synthetic-offline-detector",
        action=entity.action or "mask",
        replacement=entity.redacted_text,
        section="assessment",
    )


def build_release_evidence_summary() -> dict[str, Any]:
    """Compute small release-evidence metrics from synthetic spans/records."""
    name_start, name_end = _span(SYNTHETIC_TEXT, "Casey Example")
    phone_start, phone_end = _span(SYNTHETIC_TEXT, "212-555-0198")
    gold = [
        {"start": name_start, "end": name_end, "label": PERSON, "lang": "en"},
        {"start": phone_start, "end": phone_end, "label": PHONE, "lang": "en"},
    ]
    predicted = [
        {"start": name_start, "end": name_start + 6, "label": PERSON, "lang": "en"},
        {"start": phone_start, "end": phone_end, "label": PHONE, "lang": "en"},
    ]
    heatmap = compute_leakage_heatmap(gold, predicted, source_text=SYNTHETIC_TEXT)
    kanon = kanon_report(
        [
            {"age_band": "40-49", "zip3": "100", "condition": "diabetes"},
            {"age_band": "40-49", "zip3": "100", "condition": "diabetes"},
            {"age_band": "50-59", "zip3": "941", "condition": "asthma"},
        ],
        quasi_identifiers=["age_band", "zip3"],
        sensitive_attributes=["condition"],
    )
    return {
        "leakage_total": heatmap.total.to_dict(),
        "worst_leakage_cells": [cell.to_dict() for cell in heatmap.worst_cells],
        "kanon": {
            "k": kanon["k"],
            "class_count": kanon["class_count"],
            "quasi_identifiers": kanon["quasi_identifiers"],
        },
    }


def main() -> None:
    result = build_result()
    report = build_audit_report(result)
    minimum_profile = select_minimum_necessary(
        "clinical_minimal_redaction",
        target_posture="hipaa_safe_harbor",
        risk_level_by_label={PERSON: "high", PHONE: "high"},
        name="example_minimum_necessary",
    )
    payload = {
        "policies_available": list(list_policies()),
        "preview": redaction_preview(SYNTHETIC_TEXT, result),
        "audit": {
            "repro_hash": report.repro_hash,
            "signature_key_id": report.signature.key_id if report.signature else None,
            "signature_verified": report.verify(
                AUDIT_KEY,
                original_text=result.original_text,
                deidentified_text=result.deidentified_text,
            ),
            "repro_hash_verified": verify_repro_hash(report),
            "review_bundle": report.export_review_bundle(),
        },
        "span_contract": build_span_contract(result).to_dict(),
        "minimum_necessary": {
            "name": minimum_profile.name,
            "comparison_to_base": minimum_profile.metadata["minimum_necessary"][
                "comparison_to_base"
            ],
            "person_action": minimum_profile.action_for(PERSON),
            "phone_action": minimum_profile.action_for(PHONE),
        },
        "release_evidence": build_release_evidence_summary(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
