"""Synthetic tests for separated projection storage and policy."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from jsonschema.validators import validator_for

from openmed.clinical.journey_contracts import canonical_digest
from openmed.compliance.projections import (
    PROJECTION_SCHEMA_NAMES,
    PROJECTION_SCHEMA_VERSION,
    InMemoryTransformVault,
    ProjectionAuditEvent,
    ProjectionBoundary,
    ProjectionContractError,
    ProjectionNamespace,
    ProjectionOperation,
    ProjectionPolicy,
    ProjectionPolicyOutcome,
    ProjectionPolicyRequest,
    ProjectionRecord,
    TransformRecord,
    load_all_projection_schemas,
)
from openmed.structured.store import StoreState

T0 = "2026-01-02T03:04:05Z"
T1 = "2026-01-02T04:04:05Z"
SCOPE = "care-summary"
REVISION = "consent-v1"
IDENTIFIED_CONTENT = b"synthetic identified projection"
DEIDENTIFIED_CONTENT = b"synthetic deidentified projection"


def _request(
    namespace: ProjectionNamespace,
    operation: ProjectionOperation,
    *,
    purpose: str = "care",
    role: str = "clinician",
    attributes: tuple[str, ...] | None = None,
    tags: tuple[str, ...] = (),
    consent_state: str = "active",
) -> ProjectionPolicyRequest:
    if attributes is None:
        attributes = (
            ("identified_access",)
            if namespace is ProjectionNamespace.IDENTIFIED
            else ()
        )
    return ProjectionPolicyRequest(
        operation=operation,
        namespace=namespace,
        purpose=purpose,
        role=role,
        attributes=attributes,
        data_use_tags=tags,
        consent_state=consent_state,
    )


def _write(
    api,
    projection_id: str,
    content: bytes,
    namespace: ProjectionNamespace,
    *,
    occurred_at: str = T0,
):
    return api.write(
        projection_id,
        content,
        _request(namespace, ProjectionOperation.WRITE),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=occurred_at,
    )


def test_projection_contracts_round_trip_without_content() -> None:
    record = ProjectionRecord(
        projection_id="projection_aaaaaaaaaaaaaaaa",
        namespace=ProjectionNamespace.DEIDENTIFIED,
        version=1,
        content_digest=canonical_digest({"content": "synthetic"}),
        byte_size=12,
        consent_scope_fingerprint=canonical_digest({"scope": "synthetic"}),
        consent_revision_fingerprint=canonical_digest({"revision": "synthetic"}),
        created_at=T0,
    )
    restored = ProjectionRecord.from_json(record.to_json())

    assert restored == record
    assert restored.schema_version == PROJECTION_SCHEMA_VERSION
    assert "synthetic identified" not in record.to_json()

    payload = record.to_dict() | {"unknown": True}
    with pytest.raises(ProjectionContractError, match="missing or unknown"):
        ProjectionRecord.from_dict(payload)


def test_contract_json_rejects_duplicates_and_nonfinite_values() -> None:
    with pytest.raises(ProjectionContractError, match="invalid"):
        ProjectionRecord.from_json('{"projection_id":"a","projection_id":"b"}')
    with pytest.raises(ProjectionContractError, match="invalid"):
        ProjectionRecord.from_json('{"byte_size":NaN}')


def test_public_projection_records_validate_against_bundled_schemas() -> None:
    record = ProjectionRecord(
        projection_id="projection_aaaaaaaaaaaaaaaa",
        namespace=ProjectionNamespace.DEIDENTIFIED,
        version=1,
        content_digest=canonical_digest({"content": "synthetic"}),
        byte_size=12,
        consent_scope_fingerprint=canonical_digest({"scope": "synthetic"}),
        consent_revision_fingerprint=canonical_digest({"revision": "synthetic"}),
        created_at=T0,
    )
    request = _request(
        ProjectionNamespace.DEIDENTIFIED,
        ProjectionOperation.READ,
    )
    decision = ProjectionPolicy().evaluate(request, decided_at=T0)
    event = ProjectionAuditEvent(
        event_id="event_aaaaaaaaaaaaaaaa",
        occurred_at=T0,
        operation=request.operation,
        namespace=request.namespace,
        outcome="allow",
        reason_code="operation_allowed",
        request_digest=request.request_digest,
        resource_digest=canonical_digest({"resource": "synthetic"}),
    )
    transform = TransformRecord(
        transform_id="transform_aaaaaaaaaaaaaaaa",
        transform_type="pseudonymization",
        source_digest=canonical_digest({"source": "synthetic"}),
        output_digest=canonical_digest({"output": "synthetic"}),
        created_at=T0,
    )
    records = {
        "record": record,
        "policy_decision": decision,
        "audit_event": event,
        "transform_record": transform,
    }
    schemas = load_all_projection_schemas()

    assert set(schemas) == set(PROJECTION_SCHEMA_NAMES)
    assert ProjectionRecord.from_json(record.to_json()) == record
    assert type(decision).from_json(decision.to_json()) == decision
    assert ProjectionAuditEvent.from_json(event.to_json()) == event
    assert TransformRecord.from_json(transform.to_json()) == transform
    for name, item in records.items():
        schema = schemas[name]
        validator = validator_for(schema)
        validator.check_schema(schema)
        assert schema["schema_version"] == 1
        assert not tuple(validator(schema).iter_errors(item.to_dict()))


@given(st.sets(st.sampled_from(("export_approved", "identified_access"))))
def test_policy_attribute_canonicalization_is_deterministic(
    attributes: set[str],
) -> None:
    request = _request(
        ProjectionNamespace.DEIDENTIFIED,
        ProjectionOperation.READ,
        attributes=tuple(attributes),
    )
    restored = ProjectionPolicyRequest(
        operation=request.operation,
        namespace=request.namespace,
        purpose=request.purpose,
        role=request.role,
        attributes=tuple(reversed(request.attributes)),
    )

    assert restored.attributes == request.attributes
    assert restored.request_digest == request.request_digest


def test_namespaces_are_physically_separate_and_restart_safe(tmp_path: Path) -> None:
    root = tmp_path / "projections"
    boundary = ProjectionBoundary(root)
    identified = boundary.identified(InMemoryTransformVault())
    deidentified = boundary.deidentified()

    first = _write(
        identified,
        "projection_aaaaaaaaaaaaaaaa",
        IDENTIFIED_CONTENT,
        ProjectionNamespace.IDENTIFIED,
    )
    second = _write(
        deidentified,
        "projection_bbbbbbbbbbbbbbbb",
        DEIDENTIFIED_CONTENT,
        ProjectionNamespace.DEIDENTIFIED,
    )

    assert first.ok and second.ok
    assert (root / "identified" / "metadata.sqlite3").is_file()
    assert (root / "deidentified" / "metadata.sqlite3").is_file()
    assert os.stat(root / "identified").st_mode & 0o077 == 0
    assert os.stat(root / "deidentified").st_mode & 0o077 == 0
    assert os.stat(root / "identified" / "metadata.sqlite3").st_mode & 0o077 == 0
    assert not hasattr(deidentified, "resolve_transform")
    assert not hasattr(deidentified, "identified_root")
    assert not hasattr(deidentified, "vault")

    cross = deidentified.read(
        "projection_aaaaaaaaaaaaaaaa",
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.READ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )
    assert cross.state is StoreState.UNKNOWN
    assert cross.code == "projection_not_found"
    assert boundary.integrity_check().ok
    boundary.close()

    reopened = ProjectionBoundary(root)
    restored = reopened.deidentified().read(
        "projection_bbbbbbbbbbbbbbbb",
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.READ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )
    assert restored.ok and restored.value is not None
    assert restored.value.content == DEIDENTIFIED_CONTENT
    reopened.close()


def test_idempotency_conflict_correction_and_review_are_typed(tmp_path: Path) -> None:
    boundary = ProjectionBoundary(tmp_path / "projections")
    api = boundary.deidentified()
    projection_id = "projection_aaaaaaaaaaaaaaaa"

    first = _write(
        api,
        projection_id,
        DEIDENTIFIED_CONTENT,
        ProjectionNamespace.DEIDENTIFIED,
    )
    replay = _write(
        api,
        projection_id,
        DEIDENTIFIED_CONTENT,
        ProjectionNamespace.DEIDENTIFIED,
    )
    conflict = _write(
        api,
        projection_id,
        b"changed without correction",
        ProjectionNamespace.DEIDENTIFIED,
    )
    corrected = api.correct(
        projection_id,
        b"synthetic corrected projection",
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.CORRECT),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )
    review = api.review(
        projection_id,
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.REVIEW),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )

    assert first.ok and first.created
    assert replay.ok and not replay.created
    assert conflict.state is StoreState.CONFLICT
    assert corrected.ok and corrected.value is not None
    assert corrected.value.version == 2
    assert review.ok
    events = boundary.audit_events().value
    assert events is not None
    assert {event.operation for event in events} >= {
        ProjectionOperation.WRITE,
        ProjectionOperation.CORRECT,
        ProjectionOperation.REVIEW,
    }
    assert boundary.integrity_check().ok
    boundary.close()


def test_consent_withdrawal_invalidates_cache_and_denies_future_access(
    tmp_path: Path,
) -> None:
    boundary = ProjectionBoundary(tmp_path / "projections")
    api = boundary.deidentified()
    projection_id = "projection_aaaaaaaaaaaaaaaa"
    assert _write(
        api,
        projection_id,
        DEIDENTIFIED_CONTENT,
        ProjectionNamespace.DEIDENTIFIED,
    ).ok

    event = api.withdraw_consent(SCOPE, REVISION)
    denied = api.read(
        projection_id,
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.READ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )

    assert event.invalidated_count == 1
    assert denied.state is StoreState.DENIED
    assert denied.code == "consent_withdrawn"
    boundary.close()


def test_export_policy_distinguishes_denied_review_and_allowed(tmp_path: Path) -> None:
    boundary = ProjectionBoundary(tmp_path / "projections")
    api = boundary.deidentified()
    projection_id = "projection_aaaaaaaaaaaaaaaa"
    assert _write(
        api,
        projection_id,
        DEIDENTIFIED_CONTENT,
        ProjectionNamespace.DEIDENTIFIED,
    ).ok

    denied = api.export(
        projection_id,
        _request(
            ProjectionNamespace.DEIDENTIFIED,
            ProjectionOperation.EXPORT,
            attributes=("export_approved",),
            tags=("no-export",),
        ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )
    review = api.export(
        projection_id,
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.EXPORT),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )
    allowed = api.export(
        projection_id,
        _request(
            ProjectionNamespace.DEIDENTIFIED,
            ProjectionOperation.EXPORT,
            attributes=("export_approved",),
        ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )

    assert denied.state is StoreState.DENIED and denied.code == "data_use_denied"
    assert review.state is StoreState.PARTIAL
    assert review.code == "export_review_required"
    assert allowed.ok and allowed.value is not None
    assert allowed.value.content == DEIDENTIFIED_CONTENT
    events = boundary.audit_events().value
    assert events is not None
    export_events = [
        event for event in events if event.operation is ProjectionOperation.EXPORT
    ]
    assert {event.outcome for event in export_events} == {"allow", "deny", "review"}
    boundary.close()


def test_transform_material_is_only_resolvable_through_identified_api(
    tmp_path: Path,
) -> None:
    boundary = ProjectionBoundary(tmp_path / "projections")
    vault = InMemoryTransformVault()
    identified = boundary.identified(vault)
    deidentified = boundary.deidentified()
    record = TransformRecord(
        transform_id="transform_aaaaaaaaaaaaaaaa",
        transform_type="date_shift",
        source_digest=canonical_digest({"source": "synthetic"}),
        output_digest=canonical_digest({"output": "synthetic"}),
        created_at=T0,
    )
    material = b"synthetic protected transform material"

    stored = identified.record_transform(
        record,
        material,
        _request(ProjectionNamespace.IDENTIFIED, ProjectionOperation.WRITE),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T0,
    )
    resolved = identified.resolve_transform(
        record.transform_id,
        _request(ProjectionNamespace.IDENTIFIED, ProjectionOperation.READ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )

    assert stored.ok and resolved.ok and resolved.value == material
    assert not hasattr(deidentified, "resolve_transform")
    assert material.decode() not in record.to_json()
    assert material.decode() not in repr(vault)
    boundary.close()


def test_audit_events_are_value_free_and_survive_restart(tmp_path: Path) -> None:
    root = tmp_path / "projections"
    canary = b"synthetic raw patient value canary"
    boundary = ProjectionBoundary(root)
    api = boundary.deidentified()
    assert _write(
        api,
        "projection_aaaaaaaaaaaaaaaa",
        canary,
        ProjectionNamespace.DEIDENTIFIED,
    ).ok
    denied = api.read(
        "projection_aaaaaaaaaaaaaaaa",
        _request(
            ProjectionNamespace.IDENTIFIED,
            ProjectionOperation.READ,
            attributes=("identified_access",),
        ),
        consent_scope=SCOPE,
        consent_revision=REVISION,
        occurred_at=T1,
    )
    assert denied.state is StoreState.DENIED
    events = boundary.audit_events()
    assert events.ok and events.value is not None
    serialized = json.dumps([event.to_dict() for event in events.value])
    assert canary.decode() not in serialized
    assert "reviewer" not in serialized
    assert all(isinstance(event, ProjectionAuditEvent) for event in events.value)
    boundary.close()

    reopened = ProjectionBoundary(root)
    assert len(reopened.audit_events().value) == len(events.value)
    assert reopened.integrity_check().ok
    reopened.close()


def test_consent_hook_failure_is_value_free_and_denied(tmp_path: Path) -> None:
    canary = "synthetic hook failure patient canary"

    def broken_hook(request: ProjectionPolicyRequest) -> str:
        raise RuntimeError(canary)

    boundary = ProjectionBoundary(
        tmp_path / "projections",
        policy=ProjectionPolicy(consent_hook=broken_hook),
    )
    result = _write(
        boundary.deidentified(),
        "projection_aaaaaaaaaaaaaaaa",
        DEIDENTIFIED_CONTENT,
        ProjectionNamespace.DEIDENTIFIED,
    )

    assert result.state is StoreState.DENIED
    assert result.code == "consent_hook_failed"
    audit = boundary.audit_events().value
    assert audit is not None
    assert canary not in json.dumps([event.to_dict() for event in audit])
    boundary.close()


def test_newer_schema_version_returns_typed_unsupported(tmp_path: Path) -> None:
    root = tmp_path / "projections"
    boundary = ProjectionBoundary(root)
    boundary.close()
    database = root / "identified" / "metadata.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("UPDATE schema_migrations SET version = 99")
    connection.commit()
    connection.close()

    result = ProjectionBoundary.open(root)

    assert result.state is StoreState.UNSUPPORTED
    assert result.code == "schema_unsupported"
    assert result.value is None


def test_policy_outcomes_are_explicit() -> None:
    policy = ProjectionPolicy()
    allowed = policy.evaluate(
        _request(ProjectionNamespace.DEIDENTIFIED, ProjectionOperation.READ),
        decided_at=T0,
    )
    review = policy.evaluate(
        _request(
            ProjectionNamespace.DEIDENTIFIED,
            ProjectionOperation.READ,
            consent_state="unknown",
        ),
        decided_at=T0,
    )
    denied = policy.evaluate(
        _request(
            ProjectionNamespace.IDENTIFIED,
            ProjectionOperation.READ,
            purpose="research",
        ),
        decided_at=T0,
    )

    assert allowed.outcome is ProjectionPolicyOutcome.ALLOW
    assert review.outcome is ProjectionPolicyOutcome.REVIEW
    assert denied.outcome is ProjectionPolicyOutcome.DENY
