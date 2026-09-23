"""Offline tests for single-use human approval tokens."""

from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from openmed.agent.approvals.tokens import (
    APPROVAL_RECEIPT_SCHEMA_VERSION,
    APPROVAL_TOKEN_SCHEMA_VERSION,
    ApprovalActionMismatchError,
    ApprovalExpiredError,
    ApprovalNonceStoreError,
    ApprovalReceipt,
    ApprovalReplayError,
    ApprovalReviewerRoleMismatchError,
    ApprovalSignatureError,
    ApprovalToken,
    ApprovalTokenSigner,
    ApprovalTokenValidationError,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
    dispatch_with_approval_token,
)

KEY = b"synthetic-local-approval-key-32-bytes"
ACTION_DIGEST = "sha256:" + "a" * 64
CHANGED_ACTION_DIGEST = "sha256:" + "b" * 64
REVIEWER_ROLE = "role:org.openmed/clinical-reviewer@1.0.0"
OTHER_ROLE = "role:org.openmed/workflow-operator@1.0.0"
NONCE = "nonce_" + "1" * 32
EXPIRES_AT = 2_000_000_000
NOW = EXPIRES_AT - 100


def _token(**overrides: Any) -> ApprovalToken:
    values = {
        "action_digest": ACTION_DIGEST,
        "reviewer_role": REVIEWER_ROLE,
        "expires_at": EXPIRES_AT,
        "nonce": NONCE,
    }
    values.update(overrides)
    return ApprovalTokenSigner(KEY).issue(**values)


def _verifier() -> ApprovalTokenVerifier:
    return ApprovalTokenVerifier(KEY, InMemoryApprovalNonceStore())


def test_token_is_canonical_signed_and_deterministic_for_exact_claims() -> None:
    first = _token()
    second = _token()

    assert first == second
    assert first.to_json() == second.to_json()
    assert first.signature.startswith("hmac-sha256:")
    assert ApprovalToken.from_dict(first.to_dict()) == first
    assert ApprovalToken.from_json(first.to_json()) == first
    assert json.loads(first.to_json()) == first.to_dict()
    assert first.to_dict()["schema_version"] == APPROVAL_TOKEN_SCHEMA_VERSION


def test_default_nonce_is_random_and_injectable_for_deterministic_tests() -> None:
    signer = ApprovalTokenSigner(KEY)
    first = signer.issue(
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
    )
    second = signer.issue(
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
    )
    fixed = signer.issue(
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
        nonce_source=lambda size: bytes(range(size)),
    )

    assert first.nonce != second.nonce
    assert fixed.nonce == "nonce_000102030405060708090a0b0c0d0e0f"


def test_valid_token_is_consumed_before_dispatch_and_returns_safe_receipt() -> None:
    events: list[str] = []
    verifier = _verifier()
    original_consume = verifier.consume

    def recording_consume(*args: Any, **kwargs: Any) -> ApprovalReceipt:
        events.append("consumed")
        return original_consume(*args, **kwargs)

    verifier.consume = recording_consume  # type: ignore[method-assign]
    result, receipt = dispatch_with_approval_token(
        _token(),
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        verifier=verifier,
        dispatch=lambda: events.append("dispatched") or "ok",
        now=NOW,
    )

    assert result == "ok"
    assert events == ["consumed", "dispatched"]
    assert receipt.action_digest == ACTION_DIGEST
    assert receipt.reviewer_role == REVIEWER_ROLE
    assert receipt.consumed_at == NOW
    assert receipt.expires_at == EXPIRES_AT
    assert receipt.token_digest.startswith("sha256:")
    assert receipt.to_dict()["schema_version"] == APPROVAL_RECEIPT_SCHEMA_VERSION
    assert ApprovalReceipt.from_json(receipt.to_json()) == receipt
    assert "nonce" not in receipt.to_dict()
    assert "signature" not in receipt.to_dict()


def test_replay_is_rejected_before_a_second_dispatch() -> None:
    verifier = _verifier()
    token = _token()
    calls = 0

    def dispatch() -> None:
        nonlocal calls
        calls += 1

    dispatch_with_approval_token(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        verifier=verifier,
        dispatch=dispatch,
        now=NOW,
    )
    with pytest.raises(ApprovalReplayError, match="token: replayed"):
        dispatch_with_approval_token(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            verifier=verifier,
            dispatch=dispatch,
            now=NOW,
        )

    assert calls == 1


def test_material_action_change_invalidates_token() -> None:
    verifier = _verifier()
    token = _token()

    with pytest.raises(ApprovalActionMismatchError, match="action_mismatch"):
        verifier.consume(
            token,
            action_digest=CHANGED_ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )
    with pytest.raises(ApprovalReplayError, match="replayed"):
        verifier.consume(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )


def test_wrong_reviewer_role_fails_closed_and_consumes_token() -> None:
    verifier = _verifier()
    token = _token()

    with pytest.raises(
        ApprovalReviewerRoleMismatchError, match="reviewer_role_mismatch"
    ):
        verifier.consume(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=OTHER_ROLE,
            now=NOW,
        )
    with pytest.raises(ApprovalReplayError):
        verifier.consume(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )


@pytest.mark.parametrize("now", [EXPIRES_AT, EXPIRES_AT + 1])
def test_expiry_is_exclusive_and_does_not_dispatch(now: int) -> None:
    calls = 0

    def dispatch() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(ApprovalExpiredError, match="token: expired"):
        dispatch_with_approval_token(
            _token(),
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            verifier=_verifier(),
            dispatch=dispatch,
            now=now,
        )

    assert calls == 0


@pytest.mark.parametrize(
    "field",
    ["action_digest", "reviewer_role", "expires_at", "nonce"],
)
def test_changed_signed_claims_are_rejected_before_nonce_claim(field: str) -> None:
    token = _token()
    payload = token.to_dict()
    replacements = {
        "action_digest": CHANGED_ACTION_DIGEST,
        "reviewer_role": OTHER_ROLE,
        "expires_at": EXPIRES_AT + 1,
        "nonce": "nonce_" + "2" * 32,
    }
    payload[field] = replacements[field]
    verifier = _verifier()

    with pytest.raises(ApprovalSignatureError, match="invalid_signature"):
        verifier.consume(
            payload,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )
    assert verifier.consume(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        now=NOW,
    )


def test_nonce_claim_is_atomic_under_concurrent_verification() -> None:
    verifier = _verifier()
    token = _token()

    def attempt(_: int) -> str:
        try:
            verifier.consume(
                token,
                action_digest=ACTION_DIGEST,
                reviewer_role=REVIEWER_ROLE,
                now=NOW,
            )
        except ApprovalReplayError:
            return "replayed"
        return "accepted"

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(attempt, range(32)))

    assert outcomes.count("accepted") == 1
    assert outcomes.count("replayed") == 31


def test_dispatch_failure_still_consumes_token() -> None:
    verifier = _verifier()
    token = _token()

    with pytest.raises(RuntimeError, match="synthetic dispatch failure"):
        dispatch_with_approval_token(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            verifier=verifier,
            dispatch=lambda: (_ for _ in ()).throw(
                RuntimeError("synthetic dispatch failure")
            ),
            now=NOW,
        )
    with pytest.raises(ApprovalReplayError):
        verifier.consume(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )


def test_unknown_duplicate_and_missing_fields_fail_closed() -> None:
    payload = _token().to_dict()
    payload["free_text"] = "synthetic sensitive value"
    with pytest.raises(ApprovalTokenValidationError, match="unknown_field"):
        ApprovalToken.from_dict(payload)

    missing = _token().to_dict()
    del missing["nonce"]
    with pytest.raises(ApprovalTokenValidationError, match="missing_field"):
        ApprovalToken.from_dict(missing)

    duplicate = (
        _token()
        .to_json()
        .replace(
            f'"expires_at":{EXPIRES_AT}',
            f'"expires_at":{EXPIRES_AT},"expires_at":{EXPIRES_AT + 1}',
        )
    )
    with pytest.raises(ApprovalTokenValidationError, match="duplicate_field"):
        ApprovalToken.from_json(duplicate)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("action_digest", "sha256:not-a-digest", "invalid_digest"),
        ("reviewer_role", "synthetic-reviewer-name", "invalid_reviewer_role"),
        ("expires_at", True, "invalid_timestamp"),
        ("nonce", "nonce_too-short", "invalid_nonce"),
        ("schema_version", "openmed.agent.approval_token.v2", "unsupported"),
    ],
)
def test_invalid_claims_use_value_free_errors(
    field: str, value: Any, code: str
) -> None:
    payload = _token().to_dict()
    payload[field] = value

    with pytest.raises(ApprovalTokenValidationError) as caught:
        ApprovalToken.from_dict(payload)

    assert code in caught.value.code
    assert str(value) not in str(caught.value)


def test_diagnostics_reprs_and_receipts_do_not_expose_bearer_values() -> None:
    token = _token()
    receipt = _verifier().consume(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        now=NOW,
    )

    assert token.nonce not in repr(token)
    assert token.signature not in repr(token)
    assert token.nonce not in receipt.to_json()
    assert token.signature not in receipt.to_json()
    assert KEY.decode() not in repr(ApprovalTokenSigner(KEY))
    assert KEY.decode() not in repr(_verifier())

    sentinel = "synthetic-sensitive-review-value"

    class FailingStore:
        def claim(self, *_: Any, **__: Any) -> bool:
            raise RuntimeError(sentinel)

    with pytest.raises(ApprovalNonceStoreError) as caught:
        ApprovalTokenVerifier(KEY, FailingStore()).consume(
            _token(nonce="nonce_" + "3" * 32),
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert sentinel not in rendered


def test_invalid_key_nonce_source_store_and_dispatch_are_rejected() -> None:
    with pytest.raises(ApprovalTokenValidationError, match="invalid_key"):
        ApprovalTokenSigner(b"short")
    with pytest.raises(ApprovalTokenValidationError, match="invalid_nonce_source"):
        ApprovalTokenSigner(KEY).issue(
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            expires_at=EXPIRES_AT,
            nonce_source=lambda _: b"short",
        )
    with pytest.raises(ApprovalTokenValidationError, match="invalid_nonce_store"):
        ApprovalTokenVerifier(KEY, object())  # type: ignore[arg-type]
    with pytest.raises(ApprovalTokenValidationError, match="invalid_dispatch"):
        dispatch_with_approval_token(
            _token(),
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            verifier=_verifier(),
            dispatch=None,  # type: ignore[arg-type]
            now=NOW,
        )


def test_contract_is_exported_from_approvals_package() -> None:
    import openmed.agent.approvals as approvals

    assert approvals.ApprovalToken is ApprovalToken
    assert approvals.ApprovalReceipt is ApprovalReceipt
    assert approvals.ApprovalTokenSigner is ApprovalTokenSigner
    assert approvals.ApprovalTokenVerifier is ApprovalTokenVerifier
    assert approvals.InMemoryApprovalNonceStore is InMemoryApprovalNonceStore
