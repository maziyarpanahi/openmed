"""Tests for local versioned key lifecycle management."""

from __future__ import annotations

import json

import pytest

from openmed.core import KeyLifecycle
from openmed.core.audit import AuditReport, hash_text
from openmed.core.surrogate_vault import SurrogateVault


def _report() -> AuditReport:
    return AuditReport(
        policy="synthetic",
        resolved_profile={},
        detectors=[],
        safety_sweep={},
        spans=[],
        thresholds={},
        residual_risk={},
        openmed_version="test",
        manifest_hash="sha256:synthetic",
        document_length=0,
        input_hash=hash_text(""),
        deidentified_text_hash=hash_text(""),
    )


def test_generated_rotation_keeps_retired_audit_key_verifiable(monkeypatch):
    generated = iter((b"a" * 32, b"b" * 32))
    monkeypatch.setattr(
        "openmed.core.key_lifecycle.secrets.token_bytes",
        lambda size: next(generated) if size == 32 else b"x" * size,
    )
    lifecycle = KeyLifecycle.generate(prefix="audit")
    first_id = lifecycle.active_key_id
    first_report = lifecycle.sign_audit(_report())

    rotated = lifecycle.rotate()
    second_report = lifecycle.sign_audit(_report())

    assert first_id == "audit-v0001"
    assert rotated.key_id == "audit-v0002"
    assert lifecycle.active_key_id == "audit-v0002"
    assert lifecycle.verify_audit(AuditReport.from_json(first_report.to_json()))
    assert lifecycle.verify_audit(AuditReport.from_json(second_report.to_json()))
    assert [record.to_dict() for record in lifecycle.metadata()] == [
        {"key_id": "audit-v0001", "version": 1, "state": "retired"},
        {"key_id": "audit-v0002", "version": 2, "state": "active"},
    ]


def test_from_keys_selects_active_key_and_restricts_retired_active_use():
    lifecycle = KeyLifecycle.from_keys(
        {
            "audit-v0001": b"a" * 32,
            "audit-v0002": b"b" * 32,
        },
        active_key_id="audit-v0002",
        prefix="audit",
    )

    assert lifecycle.active_key == b"b" * 32
    assert lifecycle.resolve("audit-v0001") == b"a" * 32
    with pytest.raises(KeyError, match="retired key"):
        lifecycle.resolve("audit-v0001", allow_retired=False)
    with pytest.raises(ValueError, match="rotate before retiring"):
        lifecycle.retire("audit-v0002")


def test_unknown_or_tampered_audit_key_id_fails_closed():
    lifecycle = KeyLifecycle(b"a" * 32, prefix="audit")
    report = _report().sign(b"a" * 32, key_id="audit-v9999")

    assert not lifecycle.verify_audit(report)


def test_root_key_rotation_can_preserve_a_synthetic_surrogate_mapping():
    lifecycle = KeyLifecycle(b"a" * 32, prefix="surrogate")
    old_vault = SurrogateVault.in_memory(lifecycle.active_key)
    surrogate = old_vault.get_or_create(
        "synthetic-person",
        label="NAME",
        create_surrogate=lambda _attempt: "Casey Example",
    )

    lifecycle.rotate(b"b" * 32)
    new_vault = SurrogateVault.in_memory(lifecycle.active_key)
    migrated = new_vault.get_or_create(
        "synthetic-person",
        label="NAME",
        create_surrogate=lambda _attempt: surrogate,
    )

    assert migrated == surrogate
    assert new_vault.get("synthetic-person", label="NAME") == surrogate
    assert old_vault.current_key_id != new_vault.current_key_id


def test_metadata_repr_and_errors_do_not_expose_key_material():
    secret = b"private-unit-key-material-000000"
    lifecycle = KeyLifecycle(secret, prefix="audit")

    visible = repr(lifecycle) + json.dumps(
        [record.to_dict() for record in lifecycle.metadata()], sort_keys=True
    )
    assert secret.decode("ascii") not in visible

    with pytest.raises(KeyError) as exc_info:
        lifecycle.resolve("audit-v9999")
    assert secret.decode("ascii") not in str(exc_info.value)


@pytest.mark.parametrize(
    ("key", "message"),
    [
        (b"short", "at least 32 bytes"),
        ("short", "at least 32 bytes"),
        (None, "str or bytes"),
    ],
)
def test_rejects_weak_or_invalid_key_material(key, message):
    with pytest.raises((TypeError, ValueError), match=message):
        KeyLifecycle(key, prefix="audit")


def test_restore_requires_versioned_ids_and_known_active_key():
    with pytest.raises(ValueError, match="versioned identifier"):
        KeyLifecycle.from_keys(
            {"current": b"a" * 32},
            active_key_id="current",
            prefix="audit",
        )
    with pytest.raises(ValueError, match="supplied key"):
        KeyLifecycle.from_keys(
            {"audit-v0001": b"a" * 32},
            active_key_id="audit-v0002",
            prefix="audit",
        )
    with pytest.raises(ValueError, match="distinct key material"):
        KeyLifecycle.from_keys(
            {
                "audit-v0001": b"a" * 32,
                "audit-v0002": b"a" * 32,
            },
            active_key_id="audit-v0002",
            prefix="audit",
        )
