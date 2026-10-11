"""Offline composition of assembly, structural validation and exact approval."""

import hashlib
import socket
from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from openmed.agent.approvals.tokens import (
    ApprovalActionMismatchError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.interop.fhir.transactions import (
    TransactionApproval,
    TransactionAssemblyError,
    TransactionLimits,
    TransactionReviewerRole,
    assemble_transaction,
)
from openmed.interop.fhir.validation import validation_result


@dataclass
class SyntheticWrite:
    resource: dict
    kind: str
    conditional_predicate: str | None = None
    expected_version: str | None = None


def _prepare(monkeypatch):
    def refuse_network(*args, **kwargs):
        pytest.fail("offline assembly must not open a socket")

    monkeypatch.setattr(socket, "socket", refuse_network)
    instant = datetime(2026, 1, 2, tzinfo=timezone.utc)
    role = "role:synthetic.example/clinical-reviewer"
    # First authenticate the caller-owned action review. Its receipt can be
    # embedded without creating a self-referential final Bundle digest.
    key = b"synthetic-test-key-for-offline-assembly-only"
    signer = ApprovalTokenSigner(key, clock=lambda: 10)
    verifier = ApprovalTokenVerifier(
        key, InMemoryApprovalNonceStore(), clock=lambda: 10
    )
    token = signer.issue(
        action_digest="sha256:" + "a" * 64,
        reviewer_role=role,
        expires_at=100,
        nonce="nonce_" + "1" * 32,
    )
    receipt = verifier.consume(
        token, action_digest=token.action_digest, reviewer_role=role
    )
    approval = TransactionApproval(
        receipt.action_digest,
        "sha256:" + hashlib.sha256(receipt.to_json().encode()).hexdigest(),
        TransactionReviewerRole.CLINICAL_REVIEWER,
        ("urn:sha256:" + "c" * 64,),
    )
    entries = [
        SyntheticWrite(
            {"resourceType": "Patient", "id": "synthetic-patient"},
            "create",
            "identifier=urn%3Asynthetic%7Cpatient",
        ),
        SyntheticWrite(
            {
                "resourceType": "Observation",
                "id": "synthetic-observation",
                "status": "preliminary",
                "code": {"text": "synthetic measurement"},
                "subject": {"reference": "Patient/synthetic-patient"},
            },
            "update",
            expected_version="4",
        ),
    ]
    result = assemble_transaction(
        entries,
        approval=approval,
        limits=TransactionLimits(3, 50_000),
        clock=lambda: instant,
    )
    return entries, approval, result, signer, verifier, role, instant


@pytest.mark.integration
def test_exact_bundle_can_be_approved_locally_and_validated(monkeypatch):
    _, _, result, signer, verifier, role, _ = _prepare(monkeypatch)
    assert validation_result(result.bundle, "R4").valid
    final_token = signer.issue(
        action_digest=result.bundle_digest,
        reviewer_role=role,
        expires_at=100,
        nonce="nonce_" + "2" * 32,
    )
    final_receipt = verifier.consume(
        final_token, action_digest=result.bundle_digest, reviewer_role=role
    )
    assert (
        final_receipt.action_digest
        == "sha256:" + hashlib.sha256(result.serialized).hexdigest()
    )
    assert result.bundle["entry"][1]["request"]["ifMatch"] == 'W/"4"'
    assert (
        result.bundle["entry"][0]["request"]["ifNoneExist"]
        == "identifier=urn%3Asynthetic%7Cpatient"
    )
    assert result.entry_count == 3


@pytest.mark.integration
def test_changed_transaction_cannot_reuse_exact_bundle_approval(monkeypatch):
    entries, approval, result, signer, verifier, role, instant = _prepare(monkeypatch)
    token = signer.issue(
        action_digest=result.bundle_digest,
        reviewer_role=role,
        expires_at=100,
        nonce="nonce_" + "2" * 32,
    )
    entries[1].expected_version = "5"
    changed = assemble_transaction(
        entries,
        approval=approval,
        limits=TransactionLimits(3, 50_000),
        clock=lambda: instant,
    )
    with pytest.raises(ApprovalActionMismatchError):
        verifier.consume(token, action_digest=changed.bundle_digest, reviewer_role=role)


@pytest.mark.integration
def test_oversized_bundle_never_reaches_preview(monkeypatch):
    entries, approval, result, _, _, _, instant = _prepare(monkeypatch)
    previews = []
    with pytest.raises(TransactionAssemblyError, match="^size_limit_exceeded$"):
        bounded = assemble_transaction(
            entries,
            approval=approval,
            limits=TransactionLimits(3, len(result.serialized) - 1),
            clock=lambda: instant,
        )
        previews.append(bounded)
    assert previews == []
