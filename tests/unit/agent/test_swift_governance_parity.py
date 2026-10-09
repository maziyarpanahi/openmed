"""Synthetic Python/Swift governance contract vectors and offline controls."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from openmed.agent.approval_evidence import (
    ApprovalEvidenceError,
    ApprovalEvidenceReason,
    ApprovalEvidenceResult,
    LocalApprovalEvidenceVerifier,
    ReceiptAuthority,
)
from openmed.agent.approvals.tokens import (
    ApprovalReceipt,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.agent.artifact_reference import ArtifactReference
from openmed.agent.review_previews import parse_omop_review_preview
from openmed.agent.reviewer_handoff import ReviewerHandoffPacket
from openmed.agent.run_summary import RunSummary
from openmed.interop.omop.mutation_batch import OmopMutation, OmopMutationBatch

FIXTURE = Path(__file__).parents[2] / "fixtures/agent/governance_parity/v1.json"
NOW = datetime(2026, 9, 20, 12, tzinfo=timezone.utc)
UNIX = int(NOW.timestamp())
ACTION = "sha256:" + "1" * 64
OTHER_ACTION = "sha256:" + "2" * 64
ROLE = "role:org.openmed/reviewer@1.0.0"
OTHER_ROLE = "role:org.openmed/other-reviewer"


def _canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _receipt(action_digest: str = ACTION) -> ApprovalReceipt:
    # Synthetic test issuance only; no token, key or issuance API in Swift/examples.
    token = ApprovalTokenSigner(b"synthetic-unit-test-only-key-000000").issue(
        action_digest=action_digest,
        reviewer_role=ROLE,
        expires_at=UNIX + 60,
        nonce="nonce_" + "a" * 32,
    )
    return ApprovalTokenVerifier(
        b"synthetic-unit-test-only-key-000000",
        InMemoryApprovalNonceStore(),
        clock=lambda: UNIX,
    ).consume(token, action_digest=action_digest, reviewer_role=ROLE)


def _parse(kind: str, raw: str) -> Any:
    if kind == "artifact":
        return ArtifactReference.from_json(raw)
    if kind == "handoff":
        return ReviewerHandoffPacket.from_json(raw, now=NOW)
    if kind == "receipt":
        return ApprovalReceipt.from_json(raw)
    if kind == "run":
        return RunSummary.from_json(raw)
    if kind == "preview":
        return parse_omop_review_preview(raw)
    if kind == "verification":
        return ApprovalEvidenceResult.from_json(raw)
    raise AssertionError("unknown_fixture_kind")


def make_fixture() -> dict[str, Any]:
    """Regenerate synthetic vectors from the actual Python producer contracts."""
    receipt = _receipt()
    reference = ArtifactReference.from_dict(
        {
            "artifact_id": "art_" + "b" * 32,
            "kind": "preview",
            "schema_id": "openmed.interop.omop.mutation_batch.v1",
            "sha256": "c" * 64,
            "byte_size": (1 << 63) - 1,
        }
    )
    handoff = ReviewerHandoffPacket.from_dict(
        {
            "run_id": "run_" + "d" * 32,
            "workflow_id": "workflow:org.openmed/review@1.0.0",
            "reason_code": "human_gate",
            "requested_decision": "review_evidence",
            "evidence_references": [reference.to_dict()],
            "issued_at": "2026-09-20T11:00:00Z",
            "expires_at": "2026-09-20T13:00:00Z",
        },
        now=NOW,
    )
    batch = OmopMutationBatch(
        [OmopMutation.insert("person", {"person_id": 1, "gender_concept_id": 0})]
    )
    refused_batch = OmopMutationBatch(
        [OmopMutation.update("person", {"person_id": 2}, {"gender_concept_id": 0})]
    )
    run = RunSummary.from_dict(
        {
            "schema_version": "openmed.agent.run_summary.v1",
            "workflow_ids": ["workflow-a", "workflow-b"],
            "outcome_counts": {
                "success": 1,
                "abstained": 0,
                "review_required": 1,
                "policy_denied": 0,
                "failed": 0,
            },
            "tool_call_count": 2,
            "duration_seconds": 1.25,
            "artifact_digests": [ACTION],
        }
    )
    verified = LocalApprovalEvidenceVerifier(
        lambda _: ReceiptAuthority.RECOGNIZED, clock=lambda: UNIX
    ).verify(receipt, action_digest=ACTION, reviewer_role=ROLE)
    objects = [
        ("artifact", reference),
        ("handoff", handoff),
        ("receipt", receipt),
        ("run", run),
        ("preview", batch.preview()),
        ("preview", refused_batch.preview()),
        ("verification", verified),
    ]
    valid = []
    by_kind = {}
    for kind, obj in objects:
        canonical = obj.to_json()
        fields = json.loads(canonical)
        valid.append(
            {
                "kind": kind,
                "json": json.dumps(fields, indent=2),
                "canonical_json": canonical,
                "digest": "sha256:" + hashlib.sha256(canonical.encode()).hexdigest(),
            }
        )
        by_kind.setdefault(kind, fields)
    # Normalize native default versions and integer duration exactly as Python does.
    for kind, field in [("artifact", "version"), ("handoff", "schema_version")]:
        fields = copy.deepcopy(by_kind[kind])
        del fields[field]
        obj = _parse(kind, _canonical(fields))
        valid.append(
            {
                "kind": kind,
                "json": _canonical(fields),
                "canonical_json": obj.to_json(),
                "digest": "sha256:"
                + hashlib.sha256(obj.to_json().encode()).hexdigest(),
            }
        )
    for duration in [0, -0.0, 1e-5, 1e-6, 1e-7, 5e-324, 31_536_000]:
        fields = copy.deepcopy(by_kind["run"])
        fields["duration_seconds"] = duration
        obj = _parse("run", _canonical(fields))
        valid.append(
            {
                "kind": "run",
                "json": _canonical(fields),
                "canonical_json": obj.to_json(),
                "digest": "sha256:"
                + hashlib.sha256(obj.to_json().encode()).hexdigest(),
            }
        )
    negative = []

    def reject(
        kind: str, name: str, fields: Any, swift: str = "invalid_metadata"
    ) -> None:
        negative.append(
            {
                "kind": kind,
                "name": name,
                "json": _canonical(fields),
                "swift_error": swift,
            }
        )

    for kind, original in by_kind.items():
        fields = copy.deepcopy(original)
        fields["private_payload"] = "SYNTHETIC_SECRET_SENTINEL"
        reject(kind, "unknown-content-field", fields, "invalid_fields")
        version_key = (
            "version"
            if kind == "artifact"
            else "schema"
            if kind == "preview"
            else "schema_version"
        )
        fields = copy.deepcopy(original)
        fields[version_key] = 2 if kind == "artifact" else "openmed.unknown.v999"
        reject(kind, "unknown-version", fields, "unsupported_version")
        fields = copy.deepcopy(original)
        del fields[next(key for key in fields if key != version_key)]
        reject(kind, "missing-field", fields, "invalid_fields")
        # Duplicate key after JSON escape decoding, also at nested boundaries.
        raw = _canonical(original)
        key = next(iter(original))
        escaped = "\\u%04x" % ord(key[0]) + key[1:]
        duplicate = raw[:-1] + ',"' + escaped + '":' + _canonical(original[key]) + "}"
        negative.append(
            {
                "kind": kind,
                "name": "duplicate-escaped-key",
                "json": duplicate,
                "swift_error": "duplicate_field",
            }
        )
    for key, value in [
        ("byte_size", True),
        ("byte_size", 1.0),
        ("byte_size", 1 << 63),
        ("byte_size", 0),
        ("sha256", "sha256:" + "c" * 64),
        ("kind", "execute"),
    ]:
        fields = copy.deepcopy(by_kind["artifact"])
        fields[key] = value
        reject("artifact", key + "-invalid", fields)
    for key, value in [
        ("expires_at", "2026-09-20T12:00:00Z"),
        ("issued_at", "2026-02-30T11:00:00Z"),
        ("issued_at", "2026-09-20T11:00:00+00:00"),
        ("reason_code", "approved"),
        ("requested_decision", "execute"),
        ("workflow_id", "workflow:org.openmed/review@01.0.0"),
        ("evidence_references", [reference.to_dict()] * 2),
    ]:
        fields = copy.deepcopy(by_kind["handoff"])
        fields[key] = value
        reject(
            "handoff",
            key + "-invalid",
            fields,
            "expired" if key == "expires_at" else "invalid_metadata",
        )
    for key, value in [
        ("consumed_at", True),
        ("consumed_at", 1.0),
        ("consumed_at", UNIX + 60),
        ("expires_at", 1 << 63),
        ("reviewer_role", "role:org.openmed/reviewer@01.0.0"),
        ("action_digest", OTHER_ACTION.upper()),
    ]:
        fields = copy.deepcopy(by_kind["receipt"])
        fields[key] = value
        reject("receipt", key + "-invalid", fields)
    for key, value in [
        ("workflow_ids", ["z", "a"]),
        ("workflow_ids", ["a", "a"]),
        ("artifact_digests", [ACTION, ACTION]),
        ("duration_seconds", True),
        ("tool_call_count", 1.0),
        ("tool_call_count", 10_000_001),
    ]:
        fields = copy.deepcopy(by_kind["run"])
        fields[key] = value
        reject("run", key + "-invalid", fields)
    for mutation in [
        "count",
        "valid",
        "digest",
        "ordinal",
        "operation",
        "reference_count",
        "row-digest",
    ]:
        fields = copy.deepcopy(by_kind["preview"])
        if mutation == "count":
            fields["mutation_count"] = True
        if mutation == "valid":
            fields["is_valid"] = 1
        if mutation == "digest":
            fields["batch_digest"] = OTHER_ACTION
        if mutation == "ordinal":
            fields["mutations"][0]["ordinal"] = 1
        if mutation == "operation":
            fields["mutations"][0]["operation"] = "execute"
        if mutation == "reference_count":
            fields["mutations"][0]["reference_count"] = True
        if mutation == "row-digest":
            fields["mutations"][0]["row_digest"] = "SYNTHETIC_SECRET_SENTINEL"
        reject(
            "preview",
            mutation + "-invalid",
            fields,
            "digest_mismatch" if mutation == "digest" else "invalid_metadata",
        )
    for key, value in [
        ("status", "refused"),
        ("reason_code", "execute"),
        ("action_digest", "SYNTHETIC_SECRET_SENTINEL"),
    ]:
        fields = copy.deepcopy(by_kind["verification"])
        fields[key] = value
        reject("verification", key + "-invalid", fields)
    for raw in [
        "NaN",
        '{"x":Infinity}',
        "[",
        '{"x":1e999}',
        '{"x":01}',
        '{"x":"\\ud800"}',
    ]:
        # Native metadata validators reject these; no protected input survives.
        negative.append(
            {
                "kind": "run",
                "name": "invalid-json",
                "json": raw,
                "swift_error": "invalid_json",
            }
        )
    scenarios = []
    for authority in ["recognized", "unrecognized", "unsupported", "throw"]:
        scenarios.append(
            {
                "name": authority,
                "authority": authority,
                "now": UNIX,
                "actions": [ACTION],
                "roles": [ROLE],
            }
        )
    scenarios.extend(
        [
            {
                "name": "replay",
                "authority": "recognized",
                "now": UNIX,
                "actions": [ACTION, ACTION],
                "roles": [ROLE, ROLE],
            },
            {
                "name": "changed-action-burns",
                "authority": "recognized",
                "now": UNIX,
                "actions": [OTHER_ACTION, ACTION],
                "roles": [ROLE, ROLE],
            },
            {
                "name": "changed-role-burns",
                "authority": "recognized",
                "now": UNIX,
                "actions": [ACTION, ACTION],
                "roles": [OTHER_ROLE, ROLE],
            },
            {
                "name": "exclusive-expiry",
                "authority": "recognized",
                "now": UNIX + 60,
                "actions": [ACTION],
                "roles": [ROLE],
            },
            {
                "name": "future-receipt",
                "authority": "recognized",
                "now": UNIX - 1,
                "actions": [ACTION],
                "roles": [ROLE],
            },
        ]
    )
    for scenario in scenarios:
        authority = scenario["authority"]

        def custody(_: str) -> ReceiptAuthority:
            if authority == "throw":
                raise RuntimeError("SYNTHETIC_SECRET_SENTINEL")
            return ReceiptAuthority(authority)

        verifier = LocalApprovalEvidenceVerifier(custody, clock=lambda: scenario["now"])
        scenario["expected_json"] = [
            verifier.verify(receipt, action_digest=action, reviewer_role=role).to_json()
            for action, role in zip(scenario["actions"], scenario["roles"])
        ]
    return {
        "schema_version": "openmed.tests.agent_governance_parity.v1",
        "synthetic": True,
        "now": UNIX,
        "valid": valid,
        "negative": negative,
        "receipt_json": receipt.to_json(),
        "bound_preview_receipt_json": _receipt(
            batch.preview().preview_digest
        ).to_json(),
        "changed_preview_json": OmopMutationBatch(
            [OmopMutation.insert("person", {"person_id": 3, "gender_concept_id": 0})]
        )
        .preview()
        .to_json(),
        "verification_scenarios": scenarios,
    }


def test_shared_vectors_regenerate_from_native_python_contracts() -> None:
    assert json.loads(FIXTURE.read_text()) == make_fixture()


@pytest.mark.parametrize("row", json.loads(FIXTURE.read_text())["valid"])
def test_python_consumes_shared_valid_vectors(row: dict[str, Any]) -> None:
    parsed = _parse(row["kind"], row["json"])
    assert parsed.to_json() == row["canonical_json"]
    if hasattr(parsed, "authorizes_clinical_action"):
        assert parsed.authorizes_clinical_action is False


@pytest.mark.parametrize("row", json.loads(FIXTURE.read_text())["negative"])
def test_python_rejects_shared_malformed_vectors(row: dict[str, Any]) -> None:
    with pytest.raises(ValueError) as caught:
        _parse(row["kind"], row["json"])
    assert "SYNTHETIC_SECRET_SENTINEL" not in str(caught.value)


def test_local_default_does_not_trust_serialized_receipts() -> None:
    verifier = LocalApprovalEvidenceVerifier(clock=lambda: UNIX)
    result = verifier.verify(_receipt(), action_digest=ACTION, reviewer_role=ROLE)
    assert result.reason_code is ApprovalEvidenceReason.UNSUPPORTED_AUTHORITY
    assert result.authorizes_clinical_action is False


def test_native_preview_and_receipt_binding_rejects_a_changed_batch() -> None:
    fixture = json.loads(FIXTURE.read_text())
    preview = parse_omop_review_preview(
        next(
            row["canonical_json"]
            for row in fixture["valid"]
            if row["kind"] == "preview"
        )
    )
    changed = parse_omop_review_preview(fixture["changed_preview_json"])
    receipt = ApprovalReceipt.from_json(fixture["bound_preview_receipt_json"])
    digest = "sha256:" + hashlib.sha256(receipt.to_json().encode()).hexdigest()
    assert receipt.action_digest == preview.preview_digest
    verifier = LocalApprovalEvidenceVerifier(
        lambda value: (
            ReceiptAuthority.RECOGNIZED
            if value == digest
            else ReceiptAuthority.UNRECOGNIZED
        ),
        clock=lambda: UNIX,
    )
    assert (
        verifier.verify(
            receipt, action_digest=changed.preview_digest, reviewer_role=ROLE
        ).reason_code
        is ApprovalEvidenceReason.ACTION_MISMATCH
    )
    assert (
        verifier.verify(
            receipt, action_digest=preview.preview_digest, reviewer_role=ROLE
        ).reason_code
        is ApprovalEvidenceReason.REPLAYED
    )


def test_local_custody_whitelist_checks_exact_canonical_receipt_digest() -> None:
    receipt = _receipt()
    digest = "sha256:" + hashlib.sha256(receipt.to_json().encode()).hexdigest()
    verifier = LocalApprovalEvidenceVerifier(
        lambda value: (
            ReceiptAuthority.RECOGNIZED
            if value == digest
            else ReceiptAuthority.UNRECOGNIZED
        ),
        clock=lambda: UNIX,
    )
    changed = ApprovalReceipt(
        receipt.action_digest,
        receipt.reviewer_role,
        receipt.token_digest,
        receipt.consumed_at - 1,
        receipt.expires_at,
    )
    assert (
        verifier.verify(changed, action_digest=ACTION, reviewer_role=ROLE).reason_code
        is ApprovalEvidenceReason.UNRECOGNIZED_RECEIPT
    )
    assert (
        verifier.verify(receipt, action_digest=ACTION, reviewer_role=ROLE).reason_code
        is ApprovalEvidenceReason.VERIFIED
    )


def test_atomic_concurrent_presentation_has_one_observation() -> None:
    receipt = _receipt()
    verifier = LocalApprovalEvidenceVerifier(
        lambda _: ReceiptAuthority.RECOGNIZED, clock=lambda: UNIX
    )
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda _: (
                    verifier.verify(
                        receipt, action_digest=ACTION, reviewer_role=ROLE
                    ).reason_code
                ),
                range(64),
            )
        )
    assert results.count(ApprovalEvidenceReason.VERIFIED) == 1
    assert results.count(ApprovalEvidenceReason.REPLAYED) == 63


def test_expiry_is_rechecked_after_local_authority_lookup() -> None:
    values = iter([UNIX, UNIX + 60])
    verifier = LocalApprovalEvidenceVerifier(
        lambda _: ReceiptAuthority.RECOGNIZED, clock=lambda: next(values)
    )
    assert (
        verifier.verify(
            _receipt(), action_digest=ACTION, reviewer_role=ROLE
        ).reason_code
        is ApprovalEvidenceReason.EXPIRED
    )


@pytest.mark.parametrize(
    "clock",
    [
        lambda: True,
        lambda: -1,
        lambda: 1 << 63,
        lambda: (_ for _ in ()).throw(RuntimeError("SYNTHETIC_SECRET_SENTINEL")),
    ],
)
def test_unavailable_clock_is_typed_and_value_free(clock: Any) -> None:
    result = LocalApprovalEvidenceVerifier(clock=clock).verify(
        _receipt(), action_digest=ACTION, reviewer_role=ROLE
    )
    assert result.reason_code is ApprovalEvidenceReason.CLOCK_UNAVAILABLE
    assert "SYNTHETIC_SECRET_SENTINEL" not in repr(result)


@pytest.mark.parametrize("answer", [True, "recognized", None])
def test_untyped_authority_result_fails_closed(answer: Any) -> None:
    result = LocalApprovalEvidenceVerifier(lambda _: answer, clock=lambda: UNIX).verify(
        _receipt(), action_digest=ACTION, reviewer_role=ROLE
    )
    assert result.reason_code is ApprovalEvidenceReason.AUTHORITY_UNAVAILABLE


@pytest.mark.parametrize("answer", [True, "yes", None])
def test_replay_store_failure_remains_explicit(answer: Any) -> None:
    class Store:
        def claim(self, *args: Any, **kwargs: Any) -> Any:
            if answer is True:
                raise RuntimeError("SYNTHETIC_SECRET_SENTINEL")
            return answer

    result = LocalApprovalEvidenceVerifier(
        lambda _: ReceiptAuthority.RECOGNIZED, replay_store=Store(), clock=lambda: UNIX
    ).verify(_receipt(), action_digest=ACTION, reviewer_role=ROLE)
    assert result.reason_code is ApprovalEvidenceReason.NONCE_STORE_UNAVAILABLE


def test_every_result_reason_round_trips_without_authority() -> None:
    for reason in ApprovalEvidenceReason:
        report = ApprovalEvidenceResult(reason, ACTION, OTHER_ACTION)
        assert ApprovalEvidenceResult.from_json(report.to_json()) == report
        assert report.authorizes_clinical_action is False


def test_malformed_expected_metadata_has_value_free_diagnostics() -> None:
    with pytest.raises(
        ApprovalEvidenceError, match="invalid_expected_metadata"
    ) as caught:
        LocalApprovalEvidenceVerifier().verify(
            _receipt(), action_digest="SYNTHETIC_SECRET_SENTINEL", reviewer_role=ROLE
        )
    assert "SYNTHETIC_SECRET_SENTINEL" not in str(caught.value)
