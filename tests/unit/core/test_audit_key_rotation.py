"""Tests for caller-owned audit-report signing-key rotation."""

from __future__ import annotations

import json

import pytest

from openmed.core.audit import AuditReport, hash_text
from openmed.core.audit_key_rotation import (
    MAX_AUDIT_HMAC_KEY_BYTES,
    MIN_AUDIT_HMAC_KEY_BYTES,
    AuditKeyRotationError,
    AuditKeyRotationSigner,
    AuditKeyRotationVerifier,
    AuditKeyUnavailableError,
    sign_audit_report,
    verify_audit_report,
)

_KEYS = {
    "audit-2025": b"synthetic-old-signing-key-material",
    "audit-2026": b"synthetic-new-signing-key-material",
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


def test_new_signatures_bind_key_id_even_when_provider_keys_match():
    shared_key = b"same-synthetic-key-material-value!" * 2
    provider = {"audit-primary": shared_key, "audit-alias": shared_key}
    signed = AuditKeyRotationSigner("audit-primary", provider).sign(_report())

    assert signed.signature is not None
    signed.signature.key_id = "audit-alias"

    assert not AuditKeyRotationVerifier(provider).verify(signed)


def test_verifier_accepts_legacy_direct_audit_report_signature():
    legacy = _report().sign(_KEYS["audit-2025"], key_id="audit-2025")

    assert AuditKeyRotationVerifier(_KEYS).verify(legacy)


def test_provider_failures_never_echo_key_material():
    def provider(_key_id: str) -> bytes:
        raise RuntimeError("synthetic-new-signing-key")

    with pytest.raises(AuditKeyRotationError) as error:
        AuditKeyRotationSigner("audit-2026", provider).sign(_report())

    assert "synthetic-new-signing-key" not in str(error.value)


def test_signer_rejects_unsafe_key_ids(subtests):
    for key_id in ["", "contains/slash", "<raw-key-material>"]:
        with subtests.test(key_id=key_id):
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


def test_signer_rejects_weak_oversized_or_mutable_key_material(subtests):
    for key in [
        b"x" * (MIN_AUDIT_HMAC_KEY_BYTES - 1),
        b"x" * (MAX_AUDIT_HMAC_KEY_BYTES + 1),
        "x" * (MIN_AUDIT_HMAC_KEY_BYTES - 1),
        bytearray(b"x" * MIN_AUDIT_HMAC_KEY_BYTES),
    ]:
        with subtests.test(key=key):
            with pytest.raises(AuditKeyRotationError, match="key material") as error:
                AuditKeyRotationSigner(
                    "audit-2026",
                    lambda _key_id: key,  # type: ignore[return-value]
                ).sign(_report())

            assert repr(key) not in str(error.value)


def test_verifier_rejects_unbound_mapping_fields_before_provider_lookup():
    signed = AuditKeyRotationSigner("audit-2026", _KEYS).sign(_report())
    requested: list[str] = []

    def provider(key_id: str) -> bytes:
        requested.append(key_id)
        return _KEYS[key_id]

    payload = signed.to_dict()
    payload["unbound_detail"] = "synthetic-private-value"
    assert not AuditKeyRotationVerifier(provider).verify(payload)

    payload = signed.to_dict()
    signature = payload["signature"]
    assert isinstance(signature, dict)
    signature["key_material"] = "synthetic-private-value"
    assert not AuditKeyRotationVerifier(provider).verify(payload)
    assert requested == []


def test_verifier_returns_false_for_invalid_runtime_input():
    verifier = AuditKeyRotationVerifier(_KEYS)

    assert not verifier.verify(None)  # type: ignore[arg-type]
    assert not verifier.verify([])  # type: ignore[arg-type]


_DIGEST = "sha256:" + "ab" * 32


def test_digest_signatures_bind_key_id_purpose_and_role():
    signer = AuditKeyRotationSigner(key_id="audit-2026", key_provider=_KEYS)
    verifier = AuditKeyRotationVerifier(key_provider=_KEYS)
    value = signer.sign_digest(_DIGEST, purpose="audit-continuity")

    assert len(value) == 64
    assert signer.sign_digest(_DIGEST, purpose="audit-continuity") == value
    assert verifier.verify_digest(
        _DIGEST, value, key_id="audit-2026", purpose="audit-continuity"
    )
    assert not verifier.verify_digest(
        _DIGEST, value, key_id="audit-2025", purpose="audit-continuity"
    )
    assert not verifier.verify_digest(
        _DIGEST,
        value,
        key_id="audit-2026",
        purpose="audit-continuity",
        role="incoming",
    )
    assert not verifier.verify_digest(
        "sha256:" + "cd" * 32, value, key_id="audit-2026", purpose="audit-continuity"
    )
    rotated = signer.sign_digest(
        _DIGEST, purpose="audit-continuity", role="outgoing", key_id="audit-2025"
    )
    assert verifier.verify_digest(
        _DIGEST,
        rotated,
        key_id="audit-2025",
        purpose="audit-continuity",
        role="outgoing",
    )


def test_digest_signatures_are_domain_separated_from_report_signatures():
    signer = AuditKeyRotationSigner(key_id="audit-2026", key_provider=_KEYS)
    report = signer.sign(_report())
    assert report.signature is not None

    digest_value = signer.sign_digest(report.repro_hash, purpose="audit-continuity")

    assert digest_value != report.signature.value
    assert not AuditKeyRotationVerifier(key_provider=_KEYS).verify_digest(
        report.repro_hash,
        report.signature.value,
        key_id="audit-2026",
        purpose="audit-continuity",
    )


def test_digest_signing_rejects_unknown_purposes_roles_and_digests(subtests):
    signer = AuditKeyRotationSigner(key_id="audit-2026", key_provider=_KEYS)
    cases = {
        "purpose": (_DIGEST, {"purpose": "release"}),
        "role": (_DIGEST, {"purpose": "audit-continuity", "role": "admin"}),
        "digest": ("not-a-digest", {"purpose": "audit-continuity"}),
    }
    for name, (digest, kwargs) in cases.items():
        with subtests.test(case=name):
            with pytest.raises(AuditKeyRotationError):
                signer.sign_digest(digest, **kwargs)


def test_unavailable_digest_keys_fail_closed_without_echoing_material():
    def provider(key_id: str) -> bytes:
        raise RuntimeError("vault refused " + _KEYS["audit-2026"].decode("ascii"))

    signer = AuditKeyRotationSigner(key_id="audit-2026", key_provider=provider)
    with pytest.raises(AuditKeyUnavailableError) as error:
        signer.sign_digest(_DIGEST, purpose="audit-continuity")

    assert "synthetic-new-signing-key" not in str(error.value)
    assert isinstance(error.value, AuditKeyRotationError)
    assert not AuditKeyRotationVerifier(key_provider=provider).verify_digest(
        _DIGEST, "0" * 64, key_id="audit-2026", purpose="audit-continuity"
    )
