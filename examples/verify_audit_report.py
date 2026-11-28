"""Sign a synthetic de-identification audit report, verify it, then detect tampering.

Run from the repository root: python -m examples.verify_audit_report

The HMAC key comes from ``OPENMED_AUDIT_KEY`` when it is set; otherwise a random
throwaway key is generated for this run. The key is never printed. The report is
built from fabricated spans, so no model is downloaded and nothing leaves the
machine.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from typing import Any

from openmed.core.audit import AuditReport, AuditSpan, DetectorInfo, hash_text

KEY_ENV = "OPENMED_AUDIT_KEY"
KEY_ID = "example-key"
SYNTHETIC_NOTE = "Patient Alex Example called 555-0100."
SYNTHETIC_REDACTED = "Patient [NAME] called [PHONE]."


def load_key(environ: Mapping[str, str] | None = None) -> tuple[bytes, str]:
    """Return the HMAC key and where it came from, without exposing it."""
    value = (os.environ if environ is None else environ).get(KEY_ENV, "")
    if value:
        return value.encode("utf-8"), "environment"
    return secrets.token_bytes(32), "generated"


def build_report() -> AuditReport:
    """Build an audit report from synthetic spans, as the unit tests do."""
    spans = [
        AuditSpan(
            start=8,
            end=20,
            label="NAME",
            canonical_label="PERSON",
            sources=["ml"],
            confidence=0.95,
            threshold=0.7,
            action="mask",
            surrogate="[NAME]",
            text_hash=hash_text(SYNTHETIC_NOTE[8:20]),
        ),
        AuditSpan(
            start=28,
            end=36,
            label="PHONE",
            canonical_label="PHONE",
            sources=["regex"],
            confidence=0.99,
            threshold=0.7,
            action="mask",
            surrogate="[PHONE]",
            text_hash=hash_text(SYNTHETIC_NOTE[28:36]),
        ),
    ]
    return AuditReport(
        policy="hipaa_safe_harbor",
        resolved_profile={"method": "mask", "confidence_threshold": 0.7},
        detectors=[
            DetectorInfo(
                source="ml",
                model_id="synthetic-example-model",
                model_format="transformers",
            )
        ],
        safety_sweep={"source": "safety_sweep", "spans_added": 0},
        spans=spans,
        thresholds={"PERSON": 0.7, "PHONE": 0.7},
        residual_risk={"projected_leakage": 0.0},
        openmed_version="example",
        manifest_hash="sha256:synthetic-manifest",
        document_length=len(SYNTHETIC_NOTE),
        input_hash=hash_text(SYNTHETIC_NOTE),
        deidentified_text_hash=hash_text(SYNTHETIC_REDACTED),
    )


def run_example(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Sign, verify a received copy, change one offset and verify again."""
    key, key_source = load_key(environ)
    signed_json = build_report().sign(key, key_id=KEY_ID).to_json()

    # A reviewer receives the JSON and checks it with the shared key.
    received = AuditReport.from_json(signed_json)
    original_verified = received.verify(key)

    # Someone shifts one span offset after signing.
    tampered = AuditReport.from_json(signed_json)
    tampered.spans[0].start += 1
    tampered_verified = tampered.verify(key)

    return {
        "key_source": key_source,
        "key_id": received.signature.key_id if received.signature else None,
        "algorithm": received.signature.algorithm if received.signature else None,
        "repro_hash": received.repro_hash,
        "span_count": len(received.spans),
        "original_verified": original_verified,
        "tampered_change": "spans[0].start +1",
        "tampered_verified": tampered_verified,
    }


def main() -> int:
    """Print the verification summary; exit non-zero if the checks misbehave."""
    summary = run_example()
    print(json.dumps(summary, indent=2, sort_keys=True))
    ok = summary["original_verified"] and not summary["tampered_verified"]
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
