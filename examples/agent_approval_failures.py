"""Demonstrate fail-closed human-approval verification failures offline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any

APPROVAL_MODULE = "openmed.agent.approvals"
KEY = b"synthetic-offline-approval-key-v1"
ACTION_DIGEST = "sha256:" + "a" * 64
CHANGED_ACTION_DIGEST = "sha256:" + "b" * 64
REVIEWER_ROLE = "role:org.openmed/clinical-reviewer@1.0.0"
OTHER_REVIEWER_ROLE = "role:org.openmed/workflow-operator@1.0.0"
EXPIRES_AT = 2_000_000_000
NOW = EXPIRES_AT - 100

EXPECTED_FAILURES = (
    ("not_yet_valid", "not_yet_valid"),
    ("lifetime_exceeded", "lifetime_exceeded"),
    ("unknown_key", "unknown_key"),
    ("legacy_disabled", "legacy_token_disabled"),
    ("expiry", "expired"),
    ("replay", "replayed"),
    ("wrong_action_digest", "action_mismatch"),
    ("wrong_reviewer_role", "reviewer_role_mismatch"),
    ("unsupported_schema_version", "unsupported_schema_version"),
)


@dataclass(frozen=True, slots=True)
class FailureResult:
    """One synthetic verification scenario and its stable failure reason."""

    scenario: str
    reason: str


def _approval_api() -> ModuleType:
    try:
        return import_module(APPROVAL_MODULE)
    except ModuleNotFoundError as error:
        if error.name == APPROVAL_MODULE:
            raise RuntimeError(
                "human approval examples require the approval-token contract from #2768"
            ) from None
        raise


def _token(api: ModuleType, nonce_digit: str) -> Any:
    return api.ApprovalTokenSigner(KEY, clock=lambda: NOW).issue(
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
        nonce="nonce_" + nonce_digit * 32,
    )


def _expect_failure(
    *,
    scenario: str,
    reason: str,
    error_type: type[Exception],
    operation: Callable[[], object],
) -> FailureResult:
    try:
        operation()
    except error_type as error:
        if getattr(error, "code", None) != reason:
            raise AssertionError(
                f"{scenario} returned an unexpected failure reason"
            ) from error
        return FailureResult(scenario=scenario, reason=reason)
    raise AssertionError(f"{scenario} did not fail closed")


def _expiry_failure(api: ModuleType) -> FailureResult:
    verifier = api.ApprovalTokenVerifier(KEY, api.InMemoryApprovalNonceStore())
    return _expect_failure(
        scenario="expiry",
        reason="expired",
        error_type=api.ApprovalExpiredError,
        operation=lambda: verifier.consume(
            _token(api, "0"),
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=EXPIRES_AT,
        ),
    )


def _replay_failure(api: ModuleType) -> FailureResult:
    verifier = api.ApprovalTokenVerifier(KEY, api.InMemoryApprovalNonceStore())
    token = _token(api, "1")
    verifier.consume(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        now=NOW,
    )
    return _expect_failure(
        scenario="replay",
        reason="replayed",
        error_type=api.ApprovalReplayError,
        operation=lambda: verifier.consume(
            token,
            action_digest=ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        ),
    )


def _wrong_action_digest_failure(api: ModuleType) -> FailureResult:
    verifier = api.ApprovalTokenVerifier(KEY, api.InMemoryApprovalNonceStore())
    return _expect_failure(
        scenario="wrong_action_digest",
        reason="action_mismatch",
        error_type=api.ApprovalActionMismatchError,
        operation=lambda: verifier.consume(
            _token(api, "2"),
            action_digest=CHANGED_ACTION_DIGEST,
            reviewer_role=REVIEWER_ROLE,
            now=NOW,
        ),
    )


def _wrong_reviewer_role_failure(api: ModuleType) -> FailureResult:
    verifier = api.ApprovalTokenVerifier(KEY, api.InMemoryApprovalNonceStore())
    return _expect_failure(
        scenario="wrong_reviewer_role",
        reason="reviewer_role_mismatch",
        error_type=api.ApprovalReviewerRoleMismatchError,
        operation=lambda: verifier.consume(
            _token(api, "3"),
            action_digest=ACTION_DIGEST,
            reviewer_role=OTHER_REVIEWER_ROLE,
            now=NOW,
        ),
    )


def _unsupported_schema_version_failure(api: ModuleType) -> FailureResult:
    payload = _token(api, "4").to_dict()
    payload["schema_version"] = "openmed.agent.approval_token.v999"
    return _expect_failure(
        scenario="unsupported_schema_version",
        reason="unsupported_schema_version",
        error_type=api.ApprovalTokenValidationError,
        operation=lambda: api.ApprovalToken.from_dict(payload),
    )


def _v2_policy_failures(api: ModuleType) -> tuple[FailureResult, ...]:
    future = api.ApprovalTokenSigner(KEY, clock=lambda: NOW + 31).issue(
        action_digest=ACTION_DIGEST,
        reviewer_role=REVIEWER_ROLE,
        expires_at=EXPIRES_AT,
        nonce="nonce_" + "5" * 32,
    )
    token = _token(api, "6")
    legacy = token.to_dict()
    legacy.pop("key_id")
    legacy.pop("issued_at")
    legacy["schema_version"] = api.LEGACY_APPROVAL_TOKEN_SCHEMA_VERSION
    # Use the original five-claim HMAC format for the migration example.
    import hashlib
    import hmac
    import json

    legacy.pop("signature")
    encoded = json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
    legacy["signature"] = (
        "hmac-sha256:" + hmac.new(KEY, encoded, hashlib.sha256).hexdigest()
    )
    scenarios = (
        (
            "not_yet_valid",
            "not_yet_valid",
            future,
            api.ApprovalTokenVerifier(
                KEY, api.InMemoryApprovalNonceStore(), clock_skew_seconds=30
            ),
        ),
        (
            "lifetime_exceeded",
            "lifetime_exceeded",
            token,
            api.ApprovalTokenVerifier(
                KEY, api.InMemoryApprovalNonceStore(), max_lifetime_seconds=99
            ),
        ),
        (
            "unknown_key",
            "unknown_key",
            token,
            api.ApprovalTokenVerifier(
                api.MappingApprovalKeyProvider({}), api.InMemoryApprovalNonceStore()
            ),
        ),
        (
            "legacy_disabled",
            "legacy_token_disabled",
            legacy,
            api.ApprovalTokenVerifier(KEY, api.InMemoryApprovalNonceStore()),
        ),
    )
    return tuple(
        _expect_failure(
            scenario=scenario,
            reason=reason,
            error_type=api.ApprovalTokenError,
            operation=lambda candidate=candidate, verifier=verifier: verifier.consume(
                candidate,
                action_digest=ACTION_DIGEST,
                reviewer_role=REVIEWER_ROLE,
                now=NOW,
            ),
        )
        for scenario, reason, candidate, verifier in scenarios
    )


def run_examples() -> tuple[FailureResult, ...]:
    """Run nine deterministic failures without dispatching an action."""

    api = _approval_api()
    results = (
        *_v2_policy_failures(api),
        _expiry_failure(api),
        _replay_failure(api),
        _wrong_action_digest_failure(api),
        _wrong_reviewer_role_failure(api),
        _unsupported_schema_version_failure(api),
    )
    observed = tuple((result.scenario, result.reason) for result in results)
    if observed != EXPECTED_FAILURES:
        raise AssertionError("approval failure examples changed unexpectedly")
    return results


def main() -> tuple[FailureResult, ...]:
    """Run the examples and print only synthetic scenario names and reasons."""

    results = run_examples()
    for result in results:
        print(f"{result.scenario}: {result.reason}")
    return results


if __name__ == "__main__":
    main()
