"""Offline, synthetic proof of the v3.1 governed clinical workflow."""

from __future__ import annotations

import json

import pytest

from examples.agent.governed_clinical_workflow import run_synthetic_workflow
from openmed.agent.approvals.tokens import (
    ApprovalExpiredError,
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    dispatch_with_approval_token,
)
from openmed.agent.permissions.grants import (
    CapabilityGrantExpiredError,
    CapabilityGrantRequest,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
    dispatch_with_capability_grant,
)
from openmed.agent.tools import DataProjectionDeniedError, plan_data_projection
from openmed.interop.fhir.compensation_report import build_compensation_report
from openmed.interop.fhir.concurrency_guard import (
    FHIRWriteConflict,
    VersionEvidence,
    guard_update,
)
from tests.fixtures.agent.governed_workflow import (
    DATA_CLASS,
    EXPIRES_AT,
    GRANT_CONSTRAINT,
    KEY,
    NOW,
    PRIVATE_MARKER,
    PURPOSE,
    REVIEWER_ROLE,
    SyntheticEffectSink,
    reviewed_tool_schema,
)


@pytest.mark.integration
@pytest.mark.parametrize("interrupt_after", [0, 1, 2, 3])
def test_interrupted_and_uninterrupted_runs_have_identical_evidence(
    tmp_path, interrupt_after: int
) -> None:
    baseline = run_synthetic_workflow(tmp_path / "baseline", interrupt_after=0)
    recovered = run_synthetic_workflow(
        tmp_path / "recovered", interrupt_after=interrupt_after
    )
    assert recovered == baseline
    assert recovered["effect_count"] == 3
    assert recovered["replay_matched"]
    serialized = json.dumps(recovered, sort_keys=True)
    assert PRIVATE_MARKER not in serialized
    for path in (tmp_path / "recovered").rglob("*.json"):
        assert PRIVATE_MARKER not in path.read_text()


@pytest.mark.integration
def test_expiry_duplicate_and_broadened_projection_fail_before_dispatch() -> None:
    calls: list[str] = []
    digest = "sha256:" + "a" * 64
    signer = ApprovalTokenSigner(KEY)
    verifier = ApprovalTokenVerifier(KEY, InMemoryApprovalNonceStore())
    expired = signer.issue(
        action_digest=digest,
        reviewer_role=REVIEWER_ROLE,
        expires_at=NOW,
        nonce="nonce_" + "2" * 32,
    )
    with pytest.raises(ApprovalExpiredError):
        dispatch_with_approval_token(
            expired,
            action_digest=digest,
            reviewer_role=REVIEWER_ROLE,
            verifier=verifier,
            dispatch=lambda: calls.append("approval"),
            now=NOW,
        )
    token = signer.issue(
        action_digest=digest,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
        nonce="nonce_" + "3" * 32,
    )
    for duplicate in (False, True):
        if duplicate:
            with pytest.raises(ApprovalReplayError):
                dispatch_with_approval_token(
                    token,
                    action_digest=digest,
                    reviewer_role=REVIEWER_ROLE,
                    verifier=verifier,
                    dispatch=lambda: calls.append("approval"),
                    now=NOW,
                )
        else:
            dispatch_with_approval_token(
                token,
                action_digest=digest,
                reviewer_role=REVIEWER_ROLE,
                verifier=verifier,
                dispatch=lambda: calls.append("approval"),
                now=NOW,
            )
    assert calls == ["approval"]

    grant = CapabilityGrantSigner(KEY).issue([GRANT_CONSTRAINT], expires_at=EXPIRES_AT)
    request = CapabilityGrantRequest(**GRANT_CONSTRAINT.to_dict())
    with pytest.raises(CapabilityGrantExpiredError):
        dispatch_with_capability_grant(
            grant,
            request,
            CapabilityGrantVerifier(KEY),
            lambda: calls.append("grant"),
            now=EXPIRES_AT,
        )
    with pytest.raises(DataProjectionDeniedError) as denied:
        plan_data_projection(
            reviewed_tool_schema(),
            workflow_purpose=PURPOSE,
            granted_data_classes=(),
        )
    assert not denied.value.rationale.approved
    assert PRIVATE_MARKER not in str(denied.value.rationale.to_dict())
    assert calls == ["approval"]


@pytest.mark.integration
def test_stale_fhir_version_and_partial_failure_require_review(tmp_path) -> None:
    sink = SyntheticEffectSink()
    expected = VersionEvidence("4", "2026-01-01T00:00:00Z")
    observed = VersionEvidence("5", "2026-01-01T00:00:00Z")
    with pytest.raises(FHIRWriteConflict) as stale:
        guard_update(expected, observed)
    assert stale.value.reason_code == "stale_evidence"
    assert not sink.commits

    intended = {
        "resourceType": "Bundle",
        "type": "batch",
        "entry": [
            {
                "request": {"method": "POST", "url": "Observation"},
                "resource": {
                    "resourceType": "Observation",
                    "valueString": PRIVATE_MARKER,
                },
            },
            {"request": {"method": "PUT", "url": "Observation/synthetic"}},
        ],
    }
    received = {
        "resourceType": "Bundle",
        "type": "batch-response",
        "entry": [
            {"response": {"status": "201 Created"}},
            {"response": {"status": "412 Precondition Failed"}},
        ],
    }
    packet = build_compensation_report(intended, received)
    assert packet.has_partial_failure and packet.approval_required
    assert PRIVATE_MARKER not in repr(packet)
    assert not sink.commits

    result = run_synthetic_workflow(tmp_path / "reviewed", interrupt_after=1)
    assert PRIVATE_MARKER not in json.dumps(result["review_packet"])
