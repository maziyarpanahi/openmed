"""Offline end-to-end approval dispatch during a local key rotation."""

import pytest

from openmed.agent.approvals import (
    ApprovalKeyError,
    ApprovalNotYetValidError,
    ApprovalReplayError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    MappingApprovalKeyProvider,
    dispatch_with_approval_token,
)

pytestmark = pytest.mark.integration


def test_rotation_and_clock_skew_protect_effect_dispatch() -> None:
    now = 2_000_000_000
    digest = "sha256:" + "a" * 64
    role = "role:org.example/reviewer"
    keys = {
        "retiring": b"synthetic-retiring-key-material-32",
        "current": b"synthetic-current-key-material-32",
    }
    provider = MappingApprovalKeyProvider(keys)
    signer = ApprovalTokenSigner(provider, key_id="retiring", clock=lambda: now + 10)
    token = signer.issue(action_digest=digest, reviewer_role=role, expires_at=now + 60)
    verifier = ApprovalTokenVerifier(
        provider, InMemoryApprovalNonceStore(), clock=lambda: now, clock_skew_seconds=10
    )
    effects = []

    def dispatch(candidate):
        return dispatch_with_approval_token(
            candidate,
            action_digest=digest,
            reviewer_role=role,
            verifier=verifier,
            dispatch=lambda: effects.append("committed"),
        )

    with pytest.raises(ApprovalNotYetValidError):
        now -= 1
        dispatch(token.to_json())
    assert effects == []
    now += 1
    _, receipt = dispatch(token.to_json())
    assert receipt.code == "approved"
    assert set(receipt.to_dict()) == {
        "schema_version",
        "action_digest",
        "token_digest",
        "code",
    }
    assert effects == ["committed"]
    now += 65
    with pytest.raises(ApprovalReplayError):
        dispatch(token)
    assert len(effects) == 1
    del keys["retiring"]
    with pytest.raises(ApprovalKeyError, match="unknown_key"):
        dispatch(token)
    current = ApprovalTokenSigner(provider, key_id="current", clock=lambda: now).issue(
        action_digest=digest, reviewer_role=role, expires_at=now + 60
    )
    dispatch(current.to_dict())
    assert len(effects) == 2
