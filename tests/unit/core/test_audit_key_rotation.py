"""Tests for caller-owned audit-report signing-key rotation."""

from __future__ import annotations

import json

import pytest

from openmed.core.audit import AuditReport, hash_text
from openmed.core.audit_key_rotation import (
    AuditKeyRotationError,
    AuditKeyRotationSigner,
    AuditKeyRotationVerifier,
    sign_audit_report,
    verify_audit_report,
)

_KEYS = {
    "audit-2025": b"synthetic-old-signing-key",
    "audit-2026": b"synthetic-new-signing-key",
}


def _report() -> AuditReport:
    empty_hash = hash_text("")
    return AuditReport(
        policy="synthetic_policy",
        resolved_profile={"method": "mask"},
        detectors=[],
        safety_sweep={},
        spans=[],
        thresholds={},
        residual_risk={},
        openmed_version="synthetic",
        manifest_hash=empty_hash,
        document_length=0,
        input_hash=empty_hash,
        deidentified_text_hash=empty_hash,
    )


def test_signer_resolves_only_the_selected_key_and_keeps_material_out_of_report():
    requested: list[str] = []

    def provider(key_id: str) -> bytes:
        requested.append(key_id)
        return _KEYS[key_id]

    signer = AuditKeyRotationSigner(key_id="audit-2026", key_provider=provider)
    signed = signer.sign(_report())
    serialized = signed.to_json()

    assert requested == ["audit-2026"]
    assert signed.signature is not None
    assert signed.signature.key_id == "audit-2026"
    assert signed.signature.algorithm == "HMAC-SHA256"
    assert "synthetic-old-signing-key" not in serialized
    assert "synthetic-new-signing-key" not in serialized
    assert "synthetic-new-signing-key" not in repr(signer)


def test_verifier_accepts_current_and_retained_previous_key():
    previous = AuditKeyRotationSigner("audit-2025", _KEYS).sign(_report())
    current = AuditKeyRotationSigner("audit-2026", _KEYS).sign(_report())
    verifier = AuditKeyRotationVerifier(_KEYS)

    assert verifier.verify(previous)
    assert verifier.verify(current)
    assert verify_audit_report(previous, key_provider=_KEYS)
    assert verify_audit_report(current.to_dict(), key_provider=_KEYS)


def test_verifier_fails_closed_when_retired_key_is_not_available():
    previous = AuditKeyRotationSigner("audit-2025", _KEYS).sign(_report())

    assert not AuditKeyRotationVerifier({"audit-2026": _KEYS["audit-2026"]}).verify(
        previous
    )


def test_verifier_fails_closed_for_an_unparseable_report_mapping():
    assert not AuditKeyRotationVerifier(_KEYS).verify({"signature": "invalid"})


def test_signing_is_deterministic_and_key_id_can_be_overridden_for_rotation():
    signer = AuditKeyRotationSigner("audit-2025", _KEYS)

    first = signer.sign(_report(), key_id="audit-2026")
    second = signer.sign(_report(), key_id="audit-2026")

    assert first.to_json() == second.to_json()
    assert first.signature is not None
    assert first.signature.key_id == "audit-2026"


def test_provider_failures_never_echo_key_material():
    def provider(_key_id: str) -> bytes:
        raise RuntimeError("synthetic-new-signing-key")

    with pytest.raises(AuditKeyRotationError) as error:
        AuditKeyRotationSigner("audit-2026", provider).sign(_report())

    assert "synthetic-new-signing-key" not in str(error.value)


@pytest.mark.parametrize("key_id", ["", "contains/slash", "<raw-key-material>"])
def test_signer_rejects_unsafe_key_ids(key_id: str):
    with pytest.raises(AuditKeyRotationError, match="non-secret"):
        AuditKeyRotationSigner(key_id, _KEYS)


def test_helper_signer_uses_caller_owned_provider():
    signed = sign_audit_report(
        _report(),
        key_id="audit-2026",
        key_provider=_KEYS,
    )

    assert AuditKeyRotationVerifier(_KEYS).verify(signed)
    assert json.loads(signed.to_json())["signature"]["key_id"] == "audit-2026"
