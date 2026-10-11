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
    ApprovalAuthorization,
    ApprovalExpiredError,
    ApprovalLifetimeError,
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
    return ApprovalTokenSigner(KEY, clock=lambda: NOW).issue(**values)


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
    signer = ApprovalTokenSigner(KEY, clock=lambda: NOW)
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
    assert receipt.code == "approved"
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
        ("schema_version", "openmed.agent.approval_token.v999", "unsupported"),
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
    assert KEY.decode() not in repr(ApprovalTokenSigner(KEY, clock=lambda: NOW))
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
        ApprovalTokenSigner(KEY, clock=lambda: NOW).issue(
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
    assert approvals.ApprovalAuthorization is ApprovalAuthorization
    assert approvals.ApprovalTokenSigner is ApprovalTokenSigner
    assert approvals.ApprovalTokenVerifier is ApprovalTokenVerifier
    assert approvals.InMemoryApprovalNonceStore is InMemoryApprovalNonceStore


def test_protected_authorization_preserves_verified_role_and_skew_bounds() -> None:
    from dataclasses import FrozenInstanceError

    verifier = ApprovalTokenVerifier(
        KEY, InMemoryApprovalNonceStore(), clock_skew_seconds=30
    )
    token = _token()
    authorization = verifier.consume_authorization(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        now=EXPIRES_AT - 1,
    )
    assert type(authorization) is ApprovalAuthorization
    assert authorization.reviewer_role == REVIEWER_ROLE
    assert authorization.consumed_at == EXPIRES_AT - 1
    assert authorization.expires_at == EXPIRES_AT + 30
    assert set(authorization.receipt.to_dict()) == {
        "schema_version",
        "code",
        "action_digest",
        "token_digest",
    }
    assert authorization.receipt.action_digest == ACTION_DIGEST
    assert (
        authorization.receipt.token_digest
        == _verifier()
        .consume(
            token, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
        )
        .token_digest
    )
    for value in (REVIEWER_ROLE, token.nonce, token.signature, KEY.decode()):
        assert value not in repr(authorization)
        assert value not in authorization.receipt.to_json()
    with pytest.raises(FrozenInstanceError):
        authorization.expires_at = EXPIRES_AT + 300
    with pytest.raises(ApprovalReplayError):
        verifier.consume_authorization(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=EXPIRES_AT + 29,
        )


def test_authorization_cannot_be_constructed_from_unverified_metadata() -> None:
    with pytest.raises(
        ApprovalTokenValidationError, match="authorization_requires_verification"
    ):
        ApprovalAuthorization()


class RecordingStore(InMemoryApprovalNonceStore):
    """Record claims to prove invalid tokens fail before replay mutation."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[str, int, int]] = []

    def claim(self, nonce_digest: str, *, expires_at: int, now: int) -> bool:
        self.calls.append((nonce_digest, expires_at, now))
        return super().claim(nonce_digest, expires_at=expires_at, now=now)


def _signed_payload(**changes: Any) -> dict[str, Any]:
    """Simulate a remote signer with independently configured time policy."""
    import hashlib
    import hmac

    payload = _token().to_dict()
    payload.update(changes)
    payload.pop("signature")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = (
        "hmac-sha256:" + hmac.new(KEY, encoded, hashlib.sha256).hexdigest()
    )
    return payload


def test_rotation_overlap_and_removal_fail_before_claim() -> None:
    from openmed.agent.approvals import ApprovalKeyError, MappingApprovalKeyProvider

    keys = {"retiring": KEY, "current": b"new-synthetic-approval-key-32-bytes"}
    provider = MappingApprovalKeyProvider(keys)
    token = ApprovalTokenSigner(provider, key_id="retiring", clock=lambda: NOW).issue(
        action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, expires_at=EXPIRES_AT
    )
    store = RecordingStore()
    verifier = ApprovalTokenVerifier(provider, store, clock=lambda: NOW)
    assert verifier.consume(
        token, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE
    )
    del keys["retiring"]
    with pytest.raises(ApprovalKeyError, match="unknown_key"):
        verifier.consume(
            token, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE
        )
    assert len(store.calls) == 1
    assert KEY.decode() not in token.key_id
    assert KEY.decode() not in repr(provider)
    current = ApprovalTokenSigner(provider, key_id="current", clock=lambda: NOW).issue(
        action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, expires_at=EXPIRES_AT
    )
    assert verifier.consume(
        current, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE
    )


@pytest.mark.parametrize("field,value", [("key_id", "other"), ("issued_at", NOW + 1)])
def test_new_claim_tampering_invalidates_signature(field: str, value: Any) -> None:
    from openmed.agent.approvals import MappingApprovalKeyProvider

    store = RecordingStore()
    verifier = ApprovalTokenVerifier(
        MappingApprovalKeyProvider({"default": KEY, "other": KEY}), store
    )
    payload = _token().to_dict()
    payload[field] = value
    with pytest.raises(ApprovalSignatureError, match="invalid_signature"):
        verifier.consume(
            payload, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
        )
    assert store.calls == []


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"expires_at": NOW + 901}, "lifetime_exceeded"),
        ({"expires_at": 4_000_000_000}, "lifetime_exceeded"),
        ({"issued_at": NOW + 31}, "not_yet_valid"),
        ({"issued_at": EXPIRES_AT}, "invalid_lifetime"),
    ],
)
def test_remote_signed_policy_failures_precede_nonce_claim(
    changes: dict, code: str
) -> None:
    from openmed.agent.approvals import ApprovalTokenError

    store = RecordingStore()
    verifier = ApprovalTokenVerifier(
        KEY, store, clock=lambda: NOW, clock_skew_seconds=30
    )
    with pytest.raises(ApprovalTokenError) as caught:
        verifier.consume(
            _signed_payload(**changes),
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
        )
    assert caught.value.code == code
    assert store.calls == []


@pytest.mark.parametrize(
    "issued,expires,code",
    [
        (NOW, NOW + 901, "lifetime_exceeded"),
        (NOW + 31, EXPIRES_AT, "not_yet_valid"),
        (NOW, NOW, "invalid_lifetime"),
        (NOW - 100, NOW - 30, "expired"),
    ],
)
def test_issuance_enforces_window_before_nonce_generation(
    issued: int, expires: int, code: str
) -> None:
    from openmed.agent.approvals import ApprovalTokenError

    generated = []
    signer = ApprovalTokenSigner(KEY, clock=lambda: NOW, clock_skew_seconds=30)
    with pytest.raises(ApprovalTokenError) as caught:
        signer.issue(
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            issued_at=issued,
            expires_at=expires,
            nonce_source=lambda size: generated.append(size) or bytes(size),
        )
    assert caught.value.code == code
    assert generated == []


@pytest.mark.parametrize(
    "now,accepted",
    [
        (NOW - 31, False),
        (NOW - 30, True),
        (EXPIRES_AT + 29, True),
        (EXPIRES_AT + 30, False),
    ],
)
def test_skew_edges_are_bounded_and_expiry_is_exclusive(
    now: int, accepted: bool
) -> None:
    from openmed.agent.approvals import ApprovalTokenError

    store = RecordingStore()
    verifier = ApprovalTokenVerifier(KEY, store, clock_skew_seconds=30)
    if accepted:
        receipt = verifier.consume(
            _token(), action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=now
        )
        assert receipt.code == "approved"
        assert store.calls[0][1] == EXPIRES_AT + 30
    else:
        with pytest.raises(ApprovalTokenError) as caught:
            verifier.consume(
                _token(),
                action_digest=ACTION_DIGEST,
                reviewer_role=REVIEWER_ROLE,
                now=now,
            )
        assert caught.value.code == ("not_yet_valid" if now < NOW else "expired")
        assert store.calls == []


def test_nonce_is_retained_through_expiry_skew_window() -> None:
    verifier = ApprovalTokenVerifier(
        KEY, InMemoryApprovalNonceStore(), clock_skew_seconds=30
    )
    token = _token()
    verifier.consume(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        now=EXPIRES_AT - 1,
    )
    for now in (EXPIRES_AT, EXPIRES_AT + 29):
        with pytest.raises(ApprovalReplayError):
            verifier.consume(
                token, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=now
            )


@pytest.mark.parametrize("representation", ["dict", "json", "object"])
def test_v1_requires_explicit_compatibility_and_exact_legacy_signature(
    representation: str,
) -> None:
    from openmed.agent.approvals import LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION

    payload = _signed_payload(schema_version=LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION)
    payload.pop("key_id")
    payload.pop("issued_at")
    # Sign the actual historical five-claim representation, not a v2 projection.
    import hashlib
    import hmac

    payload.pop("signature")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = (
        "hmac-sha256:" + hmac.new(KEY, encoded, hashlib.sha256).hexdigest()
    )
    candidate = payload
    if representation == "json":
        candidate = json.dumps(payload)
    elif representation == "object":
        candidate = ApprovalToken.from_dict(payload, allow_v1=True)
    store = RecordingStore()
    with pytest.raises(ApprovalTokenValidationError, match="legacy_token_disabled"):
        ApprovalTokenVerifier(KEY, store).consume(
            candidate, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
        )
    assert store.calls == []
    verifier = ApprovalTokenVerifier(KEY, store, allow_v1=True)
    receipt = verifier.consume(
        candidate, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
    )
    assert set(receipt.to_dict()) == {
        "schema_version",
        "code",
        "action_digest",
        "token_digest",
    }
    with pytest.raises(ApprovalReplayError):
        verifier.consume(
            candidate, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
        )
    with pytest.raises(ApprovalTokenValidationError, match="legacy_token_disabled"):
        ApprovalToken.from_json(json.dumps(payload))
    assert (
        ApprovalToken.from_json(json.dumps(payload), allow_v1=True).to_dict() == payload
    )
    store = RecordingStore()
    with pytest.raises(ApprovalLifetimeError, match="lifetime_exceeded"):
        ApprovalTokenVerifier(KEY, store, allow_v1=True).consume(
            candidate,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW - 901,
        )
    assert store.calls == []


@pytest.mark.parametrize(
    "setting,value",
    [
        ("clock_skew_seconds", -1),
        ("clock_skew_seconds", 301),
        ("clock_skew_seconds", True),
        ("max_lifetime_seconds", 0),
        ("max_lifetime_seconds", 86401),
        ("max_lifetime_seconds", 1.5),
    ],
)
def test_policy_settings_fail_closed(setting: str, value: Any) -> None:
    for constructor, args in [
        (ApprovalTokenSigner, (KEY,)),
        (ApprovalTokenVerifier, (KEY, InMemoryApprovalNonceStore())),
    ]:
        with pytest.raises(ApprovalTokenValidationError):
            constructor(*args, **{setting: value})


def test_provider_and_clock_failures_never_expose_source_values() -> None:
    from openmed.agent.approvals import ApprovalTokenError

    sentinel = "synthetic-private-key-or-clinical-text"

    class BrokenProvider:
        def get_key(self, key_id: str) -> bytes:
            raise RuntimeError(sentinel)

    def broken_clock() -> int:
        raise RuntimeError(sentinel)

    for provider, clock, code in [
        (BrokenProvider(), lambda: NOW, "key_provider_unavailable"),
        (KEY, broken_clock, "clock_unavailable"),
    ]:
        store = RecordingStore()
        with pytest.raises(ApprovalTokenError) as caught:
            ApprovalTokenVerifier(provider, store, clock=clock).consume(
                _token(), action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE
            )
        rendered = "".join(
            traceback.format_exception(caught.type, caught.value, caught.tb)
        )
        assert sentinel not in rendered
        assert caught.value.code == code
        assert store.calls == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("issued_at", True),
        ("issued_at", -1),
        ("issued_at", 1.5),
        ("issued_at", "synthetic-private-text"),
        ("key_id", "synthetic/private/path"),
        ("key_id", 12),
    ],
)
def test_v2_malformed_metadata_is_value_free_and_unclaimed(
    field: str, value: Any
) -> None:
    store = RecordingStore()
    with pytest.raises(ApprovalTokenValidationError) as caught:
        ApprovalTokenVerifier(KEY, store).consume(
            _signed_payload(**{field: value}),
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        )
    assert str(value) not in str(caught.value)
    assert store.calls == []


def test_configurable_lifetime_ceiling_is_inclusive() -> None:
    signer = ApprovalTokenSigner(KEY, clock=lambda: NOW, max_lifetime_seconds=1000)
    token = signer.issue(
        action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, expires_at=NOW + 1000
    )
    assert token.issued_at == NOW
    assert ApprovalTokenVerifier(
        KEY, InMemoryApprovalNonceStore(), max_lifetime_seconds=1000
    ).consume(token, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW)
    with pytest.raises(ApprovalLifetimeError, match="lifetime_exceeded"):
        _verifier().consume(
            token, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
        )


def test_skew_expiry_overflow_cannot_allow_nonce_replays() -> None:
    store = RecordingStore()
    now = 2**63 - 10
    payload = _signed_payload(issued_at=now, expires_at=2**63 - 1)
    with pytest.raises(ApprovalTokenValidationError, match="invalid_timestamp"):
        ApprovalTokenVerifier(KEY, store, clock_skew_seconds=30).consume(
            payload, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=now
        )
    assert store.calls == []


def test_legacy_provider_uses_only_explicit_selected_key() -> None:
    import hashlib
    import hmac

    from openmed.agent.approvals import (
        LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION,
        MappingApprovalKeyProvider,
    )

    payload = {
        k: v
        for k, v in _token().signing_payload().items()
        if k not in {"key_id", "issued_at"}
    }
    payload["schema_version"] = LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["signature"] = (
        "hmac-sha256:" + hmac.new(KEY, encoded, hashlib.sha256).hexdigest()
    )
    provider = MappingApprovalKeyProvider({"retiring": KEY})
    verifier = ApprovalTokenVerifier(
        provider, InMemoryApprovalNonceStore(), allow_v1=True, legacy_key_id="retiring"
    )
    assert verifier.consume(
        payload, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
    )
    payload["issued_at"] = NOW
    with pytest.raises(ApprovalTokenValidationError, match="unknown_field"):
        verifier.consume(
            payload, action_digest=ACTION_DIGEST, reviewer_role=REVIEWER_ROLE, now=NOW
        )
