"""Tests for the exported exchange record JSON Schemas."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Any, NoReturn

import pytest
from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry

from openmed.agent.approvals import tokens
from openmed.agent.artifact_reference import (
    ARTIFACT_REFERENCE_VERSION,
    MAX_ARTIFACT_BYTE_SIZE,
    ArtifactKind,
    ArtifactReference,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.exchange_record_schemas import (
    EXCHANGE_RECORD_SCHEMA_DIALECT,
    ExchangeRecordSchemaError,
    build_exchange_record_schema,
    build_exchange_record_schema_catalog,
    list_exchange_record_schema_names,
    render_exchange_record_schema,
    render_exchange_record_schema_catalog,
)
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.permissions import grants
from openmed.agent.reviewer_handoff import (
    MAX_HANDOFF_EVIDENCE_REFERENCES,
    REVIEWER_HANDOFF_SCHEMA_VERSION,
    RequestedDecision,
    ReviewerHandoffPacket,
    allowed_handoff_reason_codes,
)
from openmed.agent.workflows import recovery

SCHEMA_NAMES = (
    "approval_receipt",
    "approval_token",
    "capability_grant",
    "recovery_checkpoint",
    "recovery_decision",
    "reviewer_handoff",
)

SCHEMA_FINGERPRINTS: dict[str, str] = {
    "approval_receipt": (
        "sha256:51ced4e5cc84a9096d3ded9e1c33814d64c7625b8d2a9fc8afcd8475585560bf"
    ),
    "approval_token": (
        "sha256:2695264b78ca4c984583156e16fae34648eb08d1b308bd1261367bb7ca721ffc"
    ),
    "capability_grant": (
        "sha256:ea1cab83e31c9b9596764821cfe846260473efcfcb61866b51659096fac5fa6f"
    ),
    "recovery_checkpoint": (
        "sha256:19c5085b7ce870e8311130f8336f7f586a7d97def2822d65e00b34755ab1f4d8"
    ),
    "recovery_decision": (
        "sha256:da85bfce0ea599bad873b443c2eb611ac34ced8e5a55849dd2e46623a7d2b138"
    ),
    "reviewer_handoff": (
        "sha256:ae5130f20f359cf429c0f45293f045439b16829afa66c6bfd9e2bde279a5e5ca"
    ),
}

_DIGEST = "sha256:" + "a" * 64
_TOKEN_DIGEST = "sha256:" + "b" * 64
_PLAN_DIGEST = "sha256:" + "c" * 64
_SOURCE_DIGEST = "sha256:" + "d" * 64
_SIGNATURE = "hmac-sha256:" + "e" * 64
_ROLE = "role:openmed.clinical/reviewer"
_NONCE = "nonce_" + "f" * 32
_IDEMPOTENCY_KEY = "idem_" + "1" * 64
_RUN_ID = RunId("run_" + "2" * 32)
_ACTION_ID = ActionId("act_" + "3" * 32)
_WORKFLOW_ID = WorkflowId("workflow:openmed.clinical/brief")
_TOOL_ID = ToolId("tool:openmed.clinical/record-writer")
_TIMESTAMP = 1_800_000_000
_MAX_TIMESTAMP = (1 << 63) - 1
_ISSUED_AT = datetime(2026, 10, 4, 12, 0, 0, tzinfo=timezone.utc)
_EMBEDDED_SIGNATURE = re.compile(r"hmac-sha256:[0-9a-f]{8}")

_ARTIFACT_REFERENCE = ArtifactReference(
    artifact_id="art_" + "4" * 32,
    kind=ArtifactKind.EVIDENCE,
    schema_id="openmed.clinical.brief.v1",
    sha256="5" * 64,
    byte_size=256,
)


def _payloads() -> dict[str, Callable[[], dict[str, Any]]]:
    return {
        "approval_token": lambda: tokens.ApprovalToken(
            action_digest=_DIGEST,
            reviewer_role=_ROLE,
            expires_at=_TIMESTAMP,
            nonce=_NONCE,
            signature=_SIGNATURE,
        ).to_dict(),
        "approval_receipt": lambda: tokens.ApprovalReceipt(
            action_digest=_DIGEST,
            reviewer_role=_ROLE,
            token_digest=_TOKEN_DIGEST,
            consumed_at=_TIMESTAMP - 1,
            expires_at=_TIMESTAMP,
        ).to_dict(),
        "reviewer_handoff": lambda: ReviewerHandoffPacket(
            run_id=_RUN_ID,
            workflow_id=_WORKFLOW_ID,
            reason_code="insufficient_evidence",
            requested_decision=RequestedDecision.REVIEW_EVIDENCE,
            evidence_references=(_ARTIFACT_REFERENCE,),
            issued_at=_ISSUED_AT,
            expires_at=_ISSUED_AT + timedelta(hours=1),
            validation_time=_ISSUED_AT,
        ).to_dict(),
        "capability_grant": lambda: grants.CapabilityGrantManifest(
            constraints=(
                grants.CapabilityGrantConstraint(
                    tool="tool:openmed.clinical/record-writer",
                    resource="resource:openmed.clinical/patient-record",
                    action="action:openmed.clinical/read",
                    policy_profile="policy:openmed.governance/default",
                ),
            ),
            expires_at=_TIMESTAMP,
            key_id="grant-key-1",
            signature=_SIGNATURE,
        ).to_dict(),
        "recovery_checkpoint": lambda: recovery.RecoveryCheckpoint.create(
            workflow_id=_WORKFLOW_ID,
            run_id=_RUN_ID,
            sequence=0,
            phase=recovery.RecoveryPhase.PLANNED,
            plan_digest=_PLAN_DIGEST,
            effects=(
                recovery.EffectRecord.create(
                    ordinal=0,
                    run_id=_RUN_ID,
                    action_id=_ACTION_ID,
                    tool_id=_TOOL_ID,
                    kind=recovery.EffectKind.LOCAL_TOOL,
                    operation_digest=_DIGEST,
                    approval_required=False,
                    compensation_limit=recovery.CompensationLimit.PROPOSE_ONLY,
                ),
            ),
        ).to_dict(),
        "recovery_decision": lambda: recovery.RecoveryDecision.create(
            disposition=recovery.RecoveryDisposition.RESUME,
            reason=recovery.RecoveryReason.SAFE_TO_RESUME,
            source_checkpoint_digest=_SOURCE_DIGEST,
            retry_effect_ids=(_ACTION_ID.serialize(),),
            retry_idempotency_keys=(_IDEMPOTENCY_KEY,),
        ).to_dict(),
    }


def _payload(name: str) -> dict[str, Any]:
    return _payloads()[name]()


def _validator(name: str) -> Draft202012Validator:
    def reject_remote_resolution(uri: str) -> NoReturn:
        raise AssertionError(f"unexpected remote schema resolution: {uri}")

    return Draft202012Validator(
        build_exchange_record_schema(name),
        registry=Registry(retrieve=reject_remote_resolution),
    )


def _references(schema: Any) -> list[str]:
    found: list[str] = []
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "$ref":
                found.append(value)
            else:
                found.extend(_references(value))
    elif isinstance(schema, list):
        for item in schema:
            found.extend(_references(item))
    return found


def _object_nodes(node: Any) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            collected.append(node)
        for value in node.values():
            collected.extend(_object_nodes(value))
    elif isinstance(node, list):
        for item in node:
            collected.extend(_object_nodes(item))
    return collected


def test_catalog_names_dialect_and_local_references() -> None:
    assert list_exchange_record_schema_names() == SCHEMA_NAMES
    assert set(build_exchange_record_schema_catalog()) == set(SCHEMA_NAMES)
    assert set(_payloads()) == set(SCHEMA_NAMES)

    for name in SCHEMA_NAMES:
        schema = build_exchange_record_schema(name)
        Draft202012Validator.check_schema(schema)
        assert schema["$schema"] == EXCHANGE_RECORD_SCHEMA_DIALECT
        assert "$id" not in schema
        for reference in _references(schema):
            assert reference.startswith("#/$defs/")
            assert reference.removeprefix("#/$defs/") in schema["$defs"]


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_canonical_payloads_validate_without_remote_resolution(name: str) -> None:
    _validator(name).validate(_payload(name))


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_schema_fields_and_required_match_serialized_payload(name: str) -> None:
    schema = build_exchange_record_schema(name)
    payload = _payload(name)

    assert schema["additionalProperties"] is False
    assert set(schema["properties"]) == set(payload)
    assert set(schema["required"]) == set(payload)


def test_version_constants_track_python_sources() -> None:
    token = build_exchange_record_schema("approval_token")
    receipt = build_exchange_record_schema("approval_receipt")
    handoff = build_exchange_record_schema("reviewer_handoff")
    grant = build_exchange_record_schema("capability_grant")
    checkpoint = build_exchange_record_schema("recovery_checkpoint")
    decision = build_exchange_record_schema("recovery_decision")

    assert (
        token["properties"]["schema_version"]["const"]
        == tokens.APPROVAL_TOKEN_SCHEMA_VERSION
    )
    assert (
        receipt["properties"]["schema_version"]["const"]
        == tokens.APPROVAL_RECEIPT_SCHEMA_VERSION
    )
    assert (
        handoff["properties"]["schema_version"]["const"]
        == REVIEWER_HANDOFF_SCHEMA_VERSION
    )
    assert (
        grant["properties"]["schema_version"]["const"]
        == grants.CAPABILITY_GRANT_SCHEMA_VERSION
    )
    assert (
        checkpoint["properties"]["schema_version"]["const"]
        == recovery.RECOVERY_CHECKPOINT_SCHEMA_VERSION
    )
    assert (
        decision["properties"]["schema_version"]["const"]
        == recovery.RECOVERY_EVIDENCE_SCHEMA_VERSION
    )


def test_enums_track_python_sources() -> None:
    handoff = build_exchange_record_schema("reviewer_handoff")
    assert set(handoff["properties"]["requested_decision"]["enum"]) == {
        item.value for item in RequestedDecision
    }
    assert set(handoff["properties"]["reason_code"]["enum"]) == set(
        allowed_handoff_reason_codes()
    )
    assert handoff["$defs"]["artifact_reference"]["properties"]["kind"]["enum"] == [
        item.value for item in ArtifactKind
    ]

    checkpoint = build_exchange_record_schema("recovery_checkpoint")
    effect = checkpoint["$defs"]["effect_record"]
    assert checkpoint["properties"]["phase"]["enum"] == [
        item.value for item in recovery.RecoveryPhase
    ]
    assert effect["properties"]["kind"]["enum"] == [
        item.value for item in recovery.EffectKind
    ]
    assert effect["properties"]["state"]["enum"] == [
        item.value for item in recovery.EffectState
    ]
    assert effect["properties"]["compensation_limit"]["enum"] == [
        item.value for item in recovery.CompensationLimit
    ]

    decision = build_exchange_record_schema("recovery_decision")
    assert decision["properties"]["disposition"]["enum"] == [
        item.value for item in recovery.RecoveryDisposition
    ]
    assert decision["properties"]["reason"]["enum"] == [
        item.value for item in recovery.RecoveryReason
    ]


def test_bounds_track_python_sources() -> None:
    handoff = build_exchange_record_schema("reviewer_handoff")
    assert (
        handoff["properties"]["evidence_references"]["maxItems"]
        == MAX_HANDOFF_EVIDENCE_REFERENCES
    )
    reference = handoff["$defs"]["artifact_reference"]
    assert reference["properties"]["version"]["const"] == ARTIFACT_REFERENCE_VERSION
    assert reference["properties"]["byte_size"]["minimum"] == 1
    assert reference["properties"]["byte_size"]["maximum"] == MAX_ARTIFACT_BYTE_SIZE
    assert reference["properties"]["sha256"]["maxLength"] == 64

    token = build_exchange_record_schema("approval_token")
    assert token["properties"]["action_digest"]["maxLength"] == 71
    assert token["properties"]["nonce"]["minLength"] == 38
    assert token["properties"]["nonce"]["maxLength"] == 38
    assert token["properties"]["expires_at"]["minimum"] == 0
    assert token["properties"]["expires_at"]["maximum"] == _MAX_TIMESTAMP
    assert token["properties"]["reviewer_role"]["maxLength"] == 512

    checkpoint = build_exchange_record_schema("recovery_checkpoint")
    assert checkpoint["properties"]["effects"]["minItems"] == 1
    assert checkpoint["properties"]["workflow_id"]["maxLength"] == 512
    ordinal = checkpoint["$defs"]["effect_record"]["properties"]["ordinal"]
    assert ordinal["maximum"] == _MAX_TIMESTAMP

    grant = build_exchange_record_schema("capability_grant")
    assert grant["properties"]["constraints"]["minItems"] == 1
    assert grant["properties"]["constraints"]["uniqueItems"] is True
    assert grant["properties"]["key_id"]["maxLength"] == 128
    constraint = grant["$defs"]["capability_grant_constraint"]
    assert set(constraint["required"]) == {
        "tool",
        "resource",
        "action",
        "policy_profile",
    }


def test_object_nodes_are_closed() -> None:
    for name in SCHEMA_NAMES:
        nodes = _object_nodes(build_exchange_record_schema(name))
        assert nodes
        for node in nodes:
            assert node["additionalProperties"] is False
            optional = set(node["properties"]) - set(node["required"])
            assert optional in (set(), {"version"})


def test_schemas_never_embed_signatures_or_secrets() -> None:
    for name in SCHEMA_NAMES:
        rendered = render_exchange_record_schema(name)
        assert _EMBEDDED_SIGNATURE.search(rendered) is None
        assert "secret" not in rendered
        assert "password" not in rendered
        assert "private_key" not in rendered
        assert '"examples"' not in rendered
        assert '"default"' not in rendered


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        pytest.param(
            "approval_token",
            lambda payload: payload.update(
                {"schema_version": "openmed.agent.approval_token.v2"}
            ),
            id="token-unsupported-version",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"action_digest": "sha256:" + "A" * 64}),
            id="token-uppercase-digest",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"action_digest": "a" * 64}),
            id="token-prefixless-digest",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"signature": "hmac-sha512:" + "e" * 64}),
            id="token-unknown-signature-algorithm",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"nonce": "nonce_" + "f" * 31}),
            id="token-short-nonce",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update(
                {"reviewer_role": payload["reviewer_role"] + "\n"}
            ),
            id="token-trailing-newline",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"signed_by": "operator-7"}),
            id="token-extra-field",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"expires_at": 1.5}),
            id="token-float-timestamp",
        ),
        pytest.param(
            "approval_token",
            lambda payload: payload.update({"expires_at": -1}),
            id="token-negative-timestamp",
        ),
        pytest.param(
            "approval_receipt",
            lambda payload: payload.update(
                {"schema_version": "openmed.agent.approval_receipt.v2"}
            ),
            id="receipt-unsupported-version",
        ),
        pytest.param(
            "approval_receipt",
            lambda payload: payload.pop("token_digest"),
            id="receipt-missing-token-digest",
        ),
        pytest.param(
            "approval_receipt",
            lambda payload: payload.update({"consumed_at": -1}),
            id="receipt-negative-consumed-at",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload.update({"reason_code": "unbounded_reason"}),
            id="handoff-unknown-reason",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload.update({"requested_decision": "approve_all"}),
            id="handoff-unknown-decision",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload["evidence_references"][0].update(
                {"sha256": _DIGEST}
            ),
            id="handoff-prefixed-artifact-digest",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload["evidence_references"][0].update({"byte_size": 0}),
            id="handoff-zero-byte-size",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload["evidence_references"][0].update({"version": 2}),
            id="handoff-unsupported-artifact-version",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload["evidence_references"][0].update(
                {"endpoint": "https://private.example.test"}
            ),
            id="handoff-content-bearing-artifact-field",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload.update({"issued_at": "2026-10-04T12:00:00.500Z"}),
            id="handoff-subsecond-timestamp",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload.update({"expires_at": "2026-10-04T13:00:00+00:00"}),
            id="handoff-offset-timestamp",
        ),
        pytest.param(
            "reviewer_handoff",
            lambda payload: payload.update({"authorizes_clinical_action": True}),
            id="handoff-extra-claim-field",
        ),
        pytest.param(
            "capability_grant",
            lambda payload: payload.update({"constraints": []}),
            id="grant-empty-constraints",
        ),
        pytest.param(
            "capability_grant",
            lambda payload: payload.update(
                {"constraints": list(payload["constraints"]) * 2}
            ),
            id="grant-duplicate-constraints",
        ),
        pytest.param(
            "capability_grant",
            lambda payload: payload.update({"key_id": "Grant-Key"}),
            id="grant-uppercase-key-id",
        ),
        pytest.param(
            "capability_grant",
            lambda payload: payload["constraints"][0].pop("policy_profile"),
            id="grant-constraint-missing-profile",
        ),
        pytest.param(
            "capability_grant",
            lambda payload: payload.update({"signature": "hmac-sha512:" + "e" * 64}),
            id="grant-unknown-signature-algorithm",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload["effects"][0].update({"ordinal": -1}),
            id="checkpoint-negative-ordinal",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload["effects"][0].update(
                {"idempotency_key": "idem_" + "1" * 63}
            ),
            id="checkpoint-short-idempotency-key",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload.update({"approval_action_digest": _DIGEST}),
            id="checkpoint-partial-approval-triple",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload["effects"][0].update(
                {"commit_evidence_digest": _DIGEST}
            ),
            id="checkpoint-pending-effect-with-digest",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload.update(
                {"phase": "approval_recorded", "approval_receipt_digest": None}
            ),
            id="checkpoint-approval-phase-without-receipt",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload.update(
                {"previous_checkpoint_digest": _DIGEST, "checkpoint_digest": _DIGEST}
            ),
            id="checkpoint-sequence-zero-with-previous",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload.update(
                {"sequence": 1, "previous_checkpoint_digest": None}
            ),
            id="checkpoint-sequence-without-previous",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload.update({"phase": "completed"}),
            id="checkpoint-completed-with-pending-effect",
        ),
        pytest.param(
            "recovery_checkpoint",
            lambda payload: payload.update({"recovered_at": _TIMESTAMP}),
            id="checkpoint-extra-field",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update({"retry_effect_ids": []}),
            id="decision-resume-without-retry",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update({"reason": "workflow_aborted"}),
            id="decision-resume-with-review-reason",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update(
                {"compensation_effect_ids": [_ACTION_ID.serialize()]}
            ),
            id="decision-resume-with-compensation",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update({"retry_effect_ids": ["run_" + "2" * 32]}),
            id="decision-retry-id-wrong-kind",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update(
                {
                    "committed_effects": [
                        {
                            "action_id": _ACTION_ID.serialize(),
                            "commit_evidence_digest": "not-a-digest",
                        }
                    ]
                }
            ),
            id="decision-committed-effect-bad-digest",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update({"evidence": "reviewer saw the note"}),
            id="decision-extra-field",
        ),
    ],
)
def test_malformed_payloads_fail(
    name: str, mutate: Callable[[dict[str, Any]], object]
) -> None:
    payload = copy.deepcopy(_payload(name))
    mutate(payload)

    with pytest.raises(ValidationError):
        _validator(name).validate(payload)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update(
                {
                    "disposition": "complete",
                    "reason": "effects_reconciled",
                    "retry_effect_ids": [_ACTION_ID.serialize()],
                }
            ),
            id="decision-complete-with-retry",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update(
                {"disposition": "review_required", "reason": "ambiguous_effect"}
            ),
            id="decision-review-required-with-retry",
        ),
        pytest.param(
            "recovery_decision",
            lambda payload: payload.update(
                {"disposition": "review_required", "reason": "safe_to_resume"}
            ),
            id="decision-review-required-with-resume-reason",
        ),
    ],
)
def test_decision_disposition_reason_pairs_are_enforced(
    name: str, mutate: Callable[[dict[str, Any]], object]
) -> None:
    payload = copy.deepcopy(_payload(name))
    mutate(payload)

    with pytest.raises(ValidationError):
        _validator(name).validate(payload)


def test_handoff_evidence_references_are_bounded() -> None:
    payload = _payload("reviewer_handoff")
    payload["evidence_references"] = [
        _ARTIFACT_REFERENCE.to_dict() | {"artifact_id": f"art_{index:032x}"}
        for index in range(MAX_HANDOFF_EVIDENCE_REFERENCES + 1)
    ]

    with pytest.raises(ValidationError):
        _validator("reviewer_handoff").validate(payload)


def test_checkpoint_requires_a_receipt_for_approval_gated_dispatch() -> None:
    payload = _payload("recovery_checkpoint")
    payload["phase"] = recovery.RecoveryPhase.DISPATCHING.value
    payload["effects"][0]["approval_required"] = True

    with pytest.raises(ValidationError):
        _validator("recovery_checkpoint").validate(payload)

    payload["approval_action_digest"] = _PLAN_DIGEST
    payload["approval_receipt_digest"] = _TOKEN_DIGEST
    payload["approval_expires_at"] = _TIMESTAMP
    _validator("recovery_checkpoint").validate(payload)


def test_checkpoint_completed_phase_requires_committed_effects() -> None:
    payload = _payload("recovery_checkpoint")
    payload["phase"] = recovery.RecoveryPhase.COMPLETED.value

    with pytest.raises(ValidationError):
        _validator("recovery_checkpoint").validate(payload)

    payload["effects"][0]["state"] = recovery.EffectState.COMMITTED.value
    payload["effects"][0]["commit_evidence_digest"] = _TOKEN_DIGEST
    _validator("recovery_checkpoint").validate(payload)


def test_runtime_only_invariants_are_not_claimed_by_the_schema() -> None:
    receipt = _payload("approval_receipt")
    receipt["consumed_at"] = receipt["expires_at"]
    _validator("approval_receipt").validate(receipt)

    handoff = _payload("reviewer_handoff")
    handoff["expires_at"] = handoff["issued_at"]
    _validator("reviewer_handoff").validate(handoff)

    checkpoint = _payload("recovery_checkpoint")
    checkpoint["effects"][0]["ordinal"] = 7
    checkpoint["approval_action_digest"] = _TOKEN_DIGEST
    checkpoint["approval_receipt_digest"] = _TOKEN_DIGEST
    checkpoint["approval_expires_at"] = _TIMESTAMP
    _validator("recovery_checkpoint").validate(checkpoint)

    decision = _payload("recovery_decision")
    decision["retry_idempotency_keys"] = []
    _validator("recovery_decision").validate(decision)


def test_rendering_is_byte_stable_and_returns_fresh_mappings() -> None:
    first = build_exchange_record_schema("reviewer_handoff")
    first["properties"].clear()

    rendered = render_exchange_record_schema("reviewer_handoff")

    assert build_exchange_record_schema("reviewer_handoff")["properties"]
    assert rendered == render_exchange_record_schema("reviewer_handoff")
    assert json.loads(rendered) == build_exchange_record_schema("reviewer_handoff")

    for name in SCHEMA_NAMES:
        encoded = render_exchange_record_schema(name)
        fingerprint = "sha256:" + hashlib.sha256(encoded.encode("ascii")).hexdigest()
        assert fingerprint == SCHEMA_FINGERPRINTS[name]

    catalog = json.loads(render_exchange_record_schema_catalog())
    assert catalog == {
        name: json.loads(render_exchange_record_schema(name)) for name in SCHEMA_NAMES
    }


def test_unknown_schema_names_fail_without_echo() -> None:
    for name in ("", "other", "approval_token\nsecret", None, [], b"approval_token"):
        with pytest.raises(ExchangeRecordSchemaError) as error:
            build_exchange_record_schema(name)  # type: ignore[arg-type]

        message = str(error.value)
        assert message == "unknown exchange record schema"
        assert "approval_token" not in message


def test_catalog_builds_independent_mappings() -> None:
    first = build_exchange_record_schema_catalog()
    first["approval_token"]["properties"].clear()

    assert build_exchange_record_schema_catalog()["approval_token"]["properties"]
