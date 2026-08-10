"""Focused tests for strict, redacted audit-envelope parsing."""

from __future__ import annotations

import json

import pytest

from openmed.compliance import (
    AuditEnvelope,
    AuditEnvelopeBoundsError,
    AuditEnvelopeError,
    AuditEnvelopeParser,
    AuditEnvelopeSignatureError,
    AuditEnvelopeUnsignedError,
    create_audit_envelope,
    fingerprint_payload,
    parse_audit_envelope,
    redacted_audit_envelope_report,
)

_SYNTHETIC_PAYLOAD_MARKER = "synthetic-payload-marker"


def _payload() -> dict[str, object]:
    return {
        "kind": "synthetic-audit-fixture",
        "marker": _SYNTHETIC_PAYLOAD_MARKER,
        "count": 2,
    }


def _envelope() -> dict[str, object]:
    return create_audit_envelope(
        _payload(),
        signature={
            "algorithm": "HMAC-SHA256",
            "key_id": "synthetic-key",
            "value": "synthetic-signature",
        },
        envelope_id="synthetic-envelope",
        producer="synthetic-fixture",
        created_at="2026-08-11T00:00:00Z",
    )


def test_parse_discards_payload_and_emits_redacted_metadata() -> None:
    parsed = parse_audit_envelope(_envelope())

    assert isinstance(parsed, AuditEnvelope)
    assert parsed.schema_version == 1
    assert parsed.envelope_id == "synthetic-envelope"
    assert parsed.payload_present is True
    assert parsed.payload_type == "object"
    assert parsed.payload_fingerprint == fingerprint_payload(_payload())
    assert not hasattr(parsed, "payload")

    report = parsed.to_dict()
    serialized = parsed.to_json(indent=None)
    assert report["signed"] is True
    assert report["signature"]["key_id"] == "synthetic-key"
    assert "value" not in report["signature"]
    assert _SYNTHETIC_PAYLOAD_MARKER not in serialized
    assert _SYNTHETIC_PAYLOAD_MARKER not in repr(parsed)


def test_payload_fingerprint_is_canonical_and_input_order_independent() -> None:
    first = {"b": 2, "a": [True, "synthetic"]}
    second = {"a": [True, "synthetic"], "b": 2}

    assert fingerprint_payload(first) == fingerprint_payload(second)

    reparsed = AuditEnvelope.from_json(json.dumps(_envelope()))
    assert reparsed == parse_audit_envelope(_envelope())


def test_fingerprint_only_envelope_never_requires_payload_text() -> None:
    parsed = parse_audit_envelope(
        {
            "header": {
                "schema_version": 1,
                "envelope_id": "metadata-only",
                "producer": "synthetic-fixture",
            },
            "signature": "opaque-signature",
            "payload_fingerprint": fingerprint_payload({"redacted": True}),
            "payload_size": 17,
            "payload_type": "object",
        }
    )

    assert parsed.payload_present is False
    assert parsed.payload_size == 17
    assert parsed.payload_type == "object"
    assert parsed.signature.algorithm == "opaque"


def test_parser_configuration_and_report_helper_are_deterministic() -> None:
    parser = AuditEnvelopeParser()
    first = parser.parse(_envelope()).to_json()
    second = redacted_audit_envelope_report(_envelope())

    assert first == AuditEnvelope.from_json(json.dumps(_envelope())).to_json()
    assert second == parse_audit_envelope(_envelope()).to_dict()


def test_unsigned_or_malformed_signatures_fail_closed_without_echoing_values() -> None:
    unsigned = _envelope()
    unsigned["signature"] = {"algorithm": "HMAC-SHA256", "key_id": "synthetic-key"}
    with pytest.raises(AuditEnvelopeUnsignedError, match="signature is required"):
        parse_audit_envelope(unsigned)

    malformed = _envelope()
    malformed["signature"] = {
        "algorithm": "HMAC-SHA256",
        "key_id": "synthetic-key",
        "value": "",
    }
    with pytest.raises(AuditEnvelopeUnsignedError):
        parse_audit_envelope(malformed)

    malformed_signature = _envelope()
    malformed_signature["signature"] = {
        "algorithm": "HMAC-SHA256",
        "key_id": "synthetic-key",
        "value": "not allowed\n",
    }
    with pytest.raises(AuditEnvelopeSignatureError) as error:
        parse_audit_envelope(malformed_signature)
    assert _SYNTHETIC_PAYLOAD_MARKER not in str(error.value)
    assert error.value.to_dict()["redacted"] is True


def test_payload_tampering_and_unknown_fields_do_not_leak_values() -> None:
    tampered = _envelope()
    tampered["payload"] = {"marker": _SYNTHETIC_PAYLOAD_MARKER}
    with pytest.raises(AuditEnvelopeError) as error:
        parse_audit_envelope(tampered)
    assert _SYNTHETIC_PAYLOAD_MARKER not in str(error.value)
    assert _SYNTHETIC_PAYLOAD_MARKER not in json.dumps(error.value.to_dict())

    unknown = _envelope()
    unknown["sensitive-payload-field"] = _SYNTHETIC_PAYLOAD_MARKER
    with pytest.raises(AuditEnvelopeError) as error:
        parse_audit_envelope(unknown)
    assert _SYNTHETIC_PAYLOAD_MARKER not in str(error.value)
    assert "sensitive-payload-field" not in json.dumps(error.value.to_dict())


def test_duplicate_json_keys_and_unsupported_versions_are_rejected() -> None:
    duplicate = json.dumps(_envelope())[:-1] + ', "schema_version": 1}'
    with pytest.raises(AuditEnvelopeError):
        parse_audit_envelope(duplicate)

    unsupported = _envelope()
    unsupported["schema_version"] = 2
    with pytest.raises(AuditEnvelopeError):
        parse_audit_envelope(unsupported)


def test_envelope_and_payload_bounds_are_enforced() -> None:
    with pytest.raises(AuditEnvelopeBoundsError):
        parse_audit_envelope(_envelope(), max_bytes=32)

    with pytest.raises(AuditEnvelopeBoundsError):
        parse_audit_envelope(_envelope(), max_payload_bytes=4)
