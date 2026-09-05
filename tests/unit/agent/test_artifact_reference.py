"""Tests for content-free agent artifact references."""

from __future__ import annotations

import builtins
import json
import traceback
import urllib.request
from typing import Any

import pytest

from openmed.agent.artifact_reference import (
    ARTIFACT_REFERENCE_VERSION,
    MAX_ARTIFACT_BYTE_SIZE,
    ArtifactKind,
    ArtifactReference,
    ArtifactReferenceError,
    validate_artifact_references,
)

ARTIFACT_ID = "art_" + "1" * 32
OTHER_ARTIFACT_ID = "art_" + "2" * 32
SHA256 = "a" * 64


def _payload(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "artifact_id": ARTIFACT_ID,
        "kind": "evidence",
        "schema_id": "openmed.agent.evidence.v1",
        "sha256": SHA256,
        "byte_size": 128,
    }
    payload.update(updates)
    return payload


@pytest.mark.parametrize("kind", [kind.value for kind in ArtifactKind])
def test_supported_kinds_round_trip_with_deterministic_json(kind: str) -> None:
    reference = ArtifactReference.from_dict(_payload(kind=kind))

    assert ArtifactReference.from_json(reference.to_json()) == reference
    assert json.loads(reference.to_json()) == reference.to_dict()
    assert reference.to_json() == json.dumps(
        reference.to_dict(), sort_keys=True, separators=(",", ":")
    )
    assert list(reference.to_dict()) == [
        "version",
        "artifact_id",
        "kind",
        "schema_id",
        "sha256",
        "byte_size",
    ]


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("version", 2, "invalid_version"),
        ("version", True, "invalid_version"),
        ("artifact_id", "evidence-001", "invalid_artifact_id"),
        ("artifact_id", "art_" + "A" * 32, "invalid_artifact_id"),
        ("kind", "report", "unknown_kind"),
        ("schema_id", "evidence", "invalid_schema_id"),
        ("schema_id", "openmed.agent.evidence.v0", "invalid_schema_id"),
        ("schema_id", "openmed.agent.Evidence.v1", "invalid_schema_id"),
        ("schema_id", "openmed." + "a" * 128 + ".v1", "invalid_schema_id"),
        ("sha256", "A" * 64, "invalid_sha256"),
        ("sha256", "a" * 63, "invalid_sha256"),
        ("byte_size", 0, "invalid_byte_size"),
        ("byte_size", -1, "invalid_byte_size"),
        ("byte_size", True, "invalid_byte_size"),
        ("byte_size", 1.5, "invalid_byte_size"),
        ("byte_size", MAX_ARTIFACT_BYTE_SIZE + 1, "invalid_byte_size"),
    ],
)
def test_invalid_fields_fail_closed(field: str, value: Any, code: str) -> None:
    with pytest.raises(ArtifactReferenceError) as caught:
        ArtifactReference.from_dict(_payload(**{field: value}))

    assert caught.value.code == code
    assert repr(value) not in str(caught.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_id", "/tmp/report.json"),
        ("artifact_id", "C:\\patients\\report.json"),
        ("artifact_id", "https://example.test/report.json"),
        ("schema_id", "file:///tmp/schema.json"),
        ("schema_id", "https://example.test/schema.json"),
    ],
)
def test_paths_and_urls_fail_without_echoing_values(field: str, value: str) -> None:
    with pytest.raises(ArtifactReferenceError) as caught:
        ArtifactReference.from_dict(_payload(**{field: value}))

    assert value not in str(caught.value)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"artifact_id": ARTIFACT_ID},
        "not-a-mapping",
        [ARTIFACT_ID],
    ],
)
def test_missing_and_non_mapping_payloads_fail_closed(payload: Any) -> None:
    with pytest.raises(ArtifactReferenceError):
        ArtifactReference.from_dict(payload)


def test_unknown_inline_payload_fails_with_phi_safe_error() -> None:
    sentinel = "Jane Synthetic has diagnosis Z99.999; bearer secret"
    payload = _payload(payload={"report": sentinel})

    with pytest.raises(ArtifactReferenceError) as caught:
        ArtifactReference.from_dict(payload)

    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.code == "unknown_field"
    assert sentinel not in rendered


@pytest.mark.parametrize("payload", ["{", b"\xff", 42])
def test_malformed_json_fails_closed(payload: Any) -> None:
    with pytest.raises(ArtifactReferenceError) as caught:
        ArtifactReference.from_json(payload)

    assert caught.value.code == "malformed_json"


def test_duplicate_json_fields_fail_closed() -> None:
    payload = json.dumps(_payload()).replace(
        f'"artifact_id": "{ARTIFACT_ID}"',
        f'"artifact_id": "{ARTIFACT_ID}", "artifact_id": "{OTHER_ARTIFACT_ID}"',
    )

    with pytest.raises(ArtifactReferenceError) as caught:
        ArtifactReference.from_json(payload)

    assert caught.value.code == "malformed_json"
    assert OTHER_ARTIFACT_ID not in str(caught.value)


def test_reference_collection_rejects_duplicate_ids() -> None:
    first = ArtifactReference.from_dict(_payload())
    duplicate = ArtifactReference.from_dict(_payload(kind="preview"))

    with pytest.raises(ArtifactReferenceError) as caught:
        validate_artifact_references([first, duplicate])

    assert caught.value.code == "duplicate_artifact_id"
    assert ARTIFACT_ID not in str(caught.value)


def test_reference_collection_is_immutable_and_preserves_order() -> None:
    first = ArtifactReference.from_dict(_payload())
    second = ArtifactReference.from_dict(
        _payload(artifact_id=OTHER_ARTIFACT_ID, kind="evaluation")
    )

    assert validate_artifact_references([first, second]) == (first, second)


def test_validation_never_opens_or_fetches_the_artifact(monkeypatch) -> None:
    def unexpected_io(*_args, **_kwargs):
        raise AssertionError("artifact I/O is outside the reference boundary")

    monkeypatch.setattr(builtins, "open", unexpected_io)
    monkeypatch.setattr(urllib.request, "urlopen", unexpected_io)

    reference = ArtifactReference.from_dict(_payload())
    assert ArtifactReference.from_json(reference.to_json()) == reference


def test_direct_construction_cannot_bypass_strict_types() -> None:
    with pytest.raises(ArtifactReferenceError, match="unknown_kind"):
        ArtifactReference(
            artifact_id=ARTIFACT_ID,
            kind="evidence",  # type: ignore[arg-type]
            schema_id="openmed.agent.evidence.v1",
            sha256=SHA256,
            byte_size=1,
        )


def test_contract_is_exported_from_public_agent_api() -> None:
    import openmed.agent as agent

    assert agent.ArtifactKind is ArtifactKind
    assert agent.ArtifactReference is ArtifactReference
    assert agent.ArtifactReferenceError is ArtifactReferenceError
    assert agent.ARTIFACT_REFERENCE_VERSION == ARTIFACT_REFERENCE_VERSION
    assert agent.MAX_ARTIFACT_BYTE_SIZE == MAX_ARTIFACT_BYTE_SIZE
    assert agent.validate_artifact_references is validate_artifact_references
