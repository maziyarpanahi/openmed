"""Tests for deterministic, PHI-safe FHIR Bulk Data checkpoints."""

from __future__ import annotations

import json

import pytest

from openmed.interop.fhir.bulk_checkpoint import (
    CHECKPOINT_MANIFEST_VERSION,
    BulkCheckpointCompatibilityError,
    BulkCheckpointManifest,
    create_checkpoint,
    fingerprint_endpoint_scope,
    fingerprint_policy,
    is_resume_compatible,
    load_checkpoint,
    validate_resume,
)

_PAGE_TOKEN = "synthetic-page-token-001"
_POLICY = {
    "name": "synthetic-safe-policy",
    "date_shift_days": 7,
}
_ENDPOINT_SCOPE = {
    "base": "https://synthetic.example/fhir",
    "export": "group-synthetic",
}


def _checkpoint() -> BulkCheckpointManifest:
    return create_checkpoint(
        "Patient",
        _PAGE_TOKEN,
        _POLICY,
        _ENDPOINT_SCOPE,
        pages_processed=3,
        resources_processed=42,
    )


def test_checkpoint_is_deterministic_and_contains_only_digests_and_counts():
    first = _checkpoint()
    second = create_checkpoint(
        "Patient",
        _PAGE_TOKEN,
        {"date_shift_days": 7, "name": "synthetic-safe-policy"},
        {"export": "group-synthetic", "base": "https://synthetic.example/fhir"},
        pages_processed=3,
        resources_processed=42,
    )

    assert first == second
    assert first.manifest_version == CHECKPOINT_MANIFEST_VERSION
    assert first.progress == {"pages_processed": 3, "resources_processed": 42}
    assert first.to_dict() == json.loads(first.to_json())
    serialized = first.to_json()
    for raw_value in (_PAGE_TOKEN, "synthetic-safe-policy", "group-synthetic"):
        assert raw_value not in serialized
    assert first.page_token_digest.startswith("sha256:")
    assert first.policy_fingerprint == fingerprint_policy(_POLICY)
    assert first.endpoint_scope == fingerprint_endpoint_scope(_ENDPOINT_SCOPE)


def test_checkpoint_round_trips_through_atomic_local_manifest(tmp_path):
    checkpoint = _checkpoint()
    path = checkpoint.write(tmp_path / "nested" / "checkpoint.json")

    assert path.is_file()
    assert load_checkpoint(path) == checkpoint
    assert BulkCheckpointManifest.from_json(path.read_bytes()) == checkpoint


def test_matching_resume_context_is_accepted():
    checkpoint = _checkpoint()

    validate_resume(
        checkpoint,
        resource_type="Patient",
        page_token=_PAGE_TOKEN,
        policy=_POLICY,
        endpoint_scope=_ENDPOINT_SCOPE,
    )
    assert checkpoint.is_compatible(
        resource_type="Patient",
        page_token=_PAGE_TOKEN,
        policy=_POLICY,
        endpoint_scope=_ENDPOINT_SCOPE,
    )
    assert is_resume_compatible(
        checkpoint,
        resource_type="Patient",
        page_token=_PAGE_TOKEN,
        policy=_POLICY,
        endpoint_scope=_ENDPOINT_SCOPE,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_type", "Observation"),
        ("page_token", "synthetic-page-token-002"),
        ("policy", {"name": "different-synthetic-policy"}),
        (
            "endpoint_scope",
            {"base": "https://synthetic.example/other-fhir"},
        ),
        ("manifest_version", CHECKPOINT_MANIFEST_VERSION + 1),
    ],
)
def test_resume_fails_closed_when_identity_changes(field, value):
    checkpoint = _checkpoint()
    context = {
        "resource_type": "Patient",
        "page_token": _PAGE_TOKEN,
        "policy": _POLICY,
        "endpoint_scope": _ENDPOINT_SCOPE,
    }
    if field == "manifest_version":
        changed = BulkCheckpointManifest.from_dict(
            {**checkpoint.to_dict(), "manifest_version": value}
        )
        context_checkpoint = changed
    else:
        context[field] = value
        context_checkpoint = checkpoint

    with pytest.raises(BulkCheckpointCompatibilityError) as exc_info:
        validate_resume(context_checkpoint, **context)

    message = str(exc_info.value)
    assert _PAGE_TOKEN not in message
    assert "synthetic" not in message


def test_invalid_checkpoint_payload_is_rejected_without_echoing_values():
    payload = _checkpoint().to_dict()
    payload["page_token_digest"] = "synthetic-raw-token"

    with pytest.raises(ValueError) as exc_info:
        BulkCheckpointManifest.from_dict(payload)

    assert "synthetic-raw-token" not in str(exc_info.value)
