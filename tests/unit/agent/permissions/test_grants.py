"""Offline tests for signed local-agent capability grant manifests."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from openmed.agent.permissions.grants import (
    CAPABILITY_GRANT_SCHEMA_VERSION,
    CapabilityGrantConstraint,
    CapabilityGrantExpiredError,
    CapabilityGrantKeyError,
    CapabilityGrantManifest,
    CapabilityGrantRequest,
    CapabilityGrantRequiredError,
    CapabilityGrantScopeError,
    CapabilityGrantSignatureError,
    CapabilityGrantSigner,
    CapabilityGrantValidationError,
    CapabilityGrantVerifier,
    MappingCapabilityGrantKeyProvider,
    StaticCapabilityGrantKeyProvider,
    dispatch_with_capability_grant,
)

KEY = b"local-test-key-material-32-bytes!!"
OTHER_KEY = b"different-local-key-material-32b!"
EXPIRES_AT = 2_000_000_000


def _constraint(
    *,
    tool: str = "tool:org.example/redact@1.0.0",
    resource: str = "resource:org.example/clinical-document@1.0.0",
    action: str = "action:org.example/read@1.0.0",
    policy_profile: str = "policy:org.example/minimum-necessary@1.0.0",
) -> CapabilityGrantConstraint:
    return CapabilityGrantConstraint(
        tool=tool,
        resource=resource,
        action=action,
        policy_profile=policy_profile,
    )


def _manifest(
    constraints: list[CapabilityGrantConstraint] | None = None,
) -> CapabilityGrantManifest:
    return CapabilityGrantSigner(KEY).issue(
        constraints or [_constraint()], expires_at=EXPIRES_AT
    )


def _request(**overrides: str) -> CapabilityGrantRequest:
    values = _constraint().to_dict()
    values.update(overrides)
    return CapabilityGrantRequest(**values)


def _assert_not_dispatched(
    manifest: CapabilityGrantManifest | dict[str, Any] | str | bytes | None,
    request: CapabilityGrantRequest,
    error_type: type[Exception],
    *,
    verifier: CapabilityGrantVerifier | None = None,
    now: int = EXPIRES_AT - 1,
) -> None:
    calls = 0

    def dispatch() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(error_type):
        dispatch_with_capability_grant(
            manifest,
            request,
            verifier or CapabilityGrantVerifier(KEY),
            dispatch,
            now=now,
        )

    assert calls == 0


def test_manifest_is_canonical_and_deterministic() -> None:
    first = _constraint()
    second = _constraint(
        tool="tool:org.example/summarize@1.0.0",
        action="action:org.example/derive@1.0.0",
    )

    forward = _manifest([first, second])
    reverse = _manifest([second, first])

    assert forward == reverse
    assert forward.to_json() == reverse.to_json()
    assert forward.signature.startswith("hmac-sha256:")
    assert len(forward.signature) == len("hmac-sha256:") + 64
    assert forward.constraints == (first, second)
    assert json.loads(forward.to_json()) == forward.to_dict()
    assert CapabilityGrantManifest.from_json(forward.to_json()) == forward


def test_valid_exact_grant_is_verified_before_dispatch() -> None:
    events: list[str] = []
    verifier = CapabilityGrantVerifier(KEY)
    original_verify = verifier.verify

    def recording_verify(*args: Any, **kwargs: Any) -> CapabilityGrantManifest:
        events.append("verified")
        return original_verify(*args, **kwargs)

    verifier.verify = recording_verify  # type: ignore[method-assign]

    result = dispatch_with_capability_grant(
        _manifest(),
        _request(),
        verifier,
        lambda: events.append("dispatched") or "ok",
        now=EXPIRES_AT - 1,
    )

    assert result == "ok"
    assert events == ["verified", "dispatched"]


def test_missing_grant_fails_closed_before_dispatch() -> None:
    _assert_not_dispatched(None, _request(), CapabilityGrantRequiredError)


@pytest.mark.parametrize("now", [EXPIRES_AT, EXPIRES_AT + 1])
def test_expired_grant_fails_closed_before_dispatch(now: int) -> None:
    _assert_not_dispatched(
        _manifest(), _request(), CapabilityGrantExpiredError, now=now
    )


@pytest.mark.parametrize(
    ("field", "broadened_value"),
    [
        ("tool", "tool:org.example/export@1.0.0"),
        ("resource", "resource:org.example/all-records@1.0.0"),
        ("action", "action:org.example/write@1.0.0"),
        ("policy_profile", "policy:org.example/unrestricted@1.0.0"),
    ],
)
def test_signed_constraint_cannot_be_broadened(
    field: str, broadened_value: str
) -> None:
    payload = _manifest().to_dict()
    payload["constraints"][0][field] = broadened_value

    _assert_not_dispatched(payload, _request(), CapabilityGrantSignatureError)


@pytest.mark.parametrize(
    ("field", "requested_value"),
    [
        ("tool", "tool:org.example/export@1.0.0"),
        ("resource", "resource:org.example/other-document@1.0.0"),
        ("action", "action:org.example/write@1.0.0"),
        ("policy_profile", "policy:org.example/other-profile@1.0.0"),
    ],
)
def test_request_outside_exact_signed_scope_is_denied(
    field: str, requested_value: str
) -> None:
    _assert_not_dispatched(
        _manifest(),
        _request(**{field: requested_value}),
        CapabilityGrantScopeError,
    )


def test_invalid_signature_and_unknown_key_fail_closed() -> None:
    _assert_not_dispatched(
        _manifest(),
        _request(),
        CapabilityGrantSignatureError,
        verifier=CapabilityGrantVerifier(OTHER_KEY),
    )
    _assert_not_dispatched(
        _manifest(),
        _request(),
        CapabilityGrantKeyError,
        verifier=CapabilityGrantVerifier({"rotated": KEY}),
    )


def test_expiry_and_scope_are_not_checked_from_unsigned_metadata() -> None:
    payload = _manifest().to_dict()
    payload["expires_at"] = EXPIRES_AT + 10_000
    payload["constraints"].append(
        _constraint(action="action:org.example/write@1.0.0").to_dict()
    )

    _assert_not_dispatched(
        payload,
        _request(action="action:org.example/write@1.0.0"),
        CapabilityGrantSignatureError,
        now=EXPIRES_AT + 1,
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.pop("expires_at"),
        lambda payload: payload.update(extra="unsigned"),
        lambda payload: payload.update(schema_version="unknown"),
        lambda payload: payload.update(expires_at=True),
        lambda payload: payload.update(constraints=[]),
        lambda payload: payload["constraints"][0].pop("action"),
        lambda payload: payload["constraints"][0].update(extra="unsigned"),
    ],
)
def test_malformed_manifests_fail_closed(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    payload = _manifest().to_dict()
    mutate(payload)

    _assert_not_dispatched(payload, _request(), CapabilityGrantValidationError)


def test_duplicate_json_fields_fail_closed() -> None:
    serialized = _manifest().to_json()
    duplicated = serialized.replace(
        '"expires_at":2000000000',
        '"expires_at":2000000000,"expires_at":2000000001',
    )

    _assert_not_dispatched(duplicated, _request(), CapabilityGrantValidationError)


def test_duplicate_constraints_and_invalid_dispatch_are_rejected() -> None:
    with pytest.raises(CapabilityGrantValidationError) as duplicate:
        _manifest([_constraint(), _constraint()])
    assert duplicate.value.code == "duplicate_constraint"

    with pytest.raises(CapabilityGrantValidationError) as invalid_dispatch:
        dispatch_with_capability_grant(
            _manifest(),
            _request(),
            CapabilityGrantVerifier(KEY),
            None,  # type: ignore[arg-type]
            now=EXPIRES_AT - 1,
        )
    assert invalid_dispatch.value.code == "invalid_dispatch"


def test_key_providers_support_rotation_without_exposing_keys() -> None:
    signer = CapabilityGrantSigner(
        MappingCapabilityGrantKeyProvider({"rotated": KEY}), key_id="rotated"
    )
    manifest = signer.issue([_constraint()], expires_at=EXPIRES_AT)
    verifier = CapabilityGrantVerifier(
        StaticCapabilityGrantKeyProvider(KEY, key_id="rotated")
    )

    assert verifier.verify(manifest, _request(), now=EXPIRES_AT - 1) == manifest
    assert KEY.decode() not in repr(signer.key_provider)
    assert KEY.decode() not in repr(verifier.key_provider)


def test_diagnostics_and_reprs_never_echo_rejected_values() -> None:
    rejected = "sensitive-record-value-should-not-appear"

    with pytest.raises(CapabilityGrantValidationError) as caught:
        _constraint(resource=rejected)

    assert rejected not in str(caught.value)
    assert rejected not in repr(caught.value)
    manifest = _manifest()
    request = _request()
    assert manifest.constraints[0].resource not in repr(manifest)
    assert request.resource not in repr(request)
    assert request.resource not in repr(request.as_constraint())

    class FailingProvider:
        def get_key(self, key_id: str) -> bytes:
            raise RuntimeError(rejected)

    with pytest.raises(CapabilityGrantKeyError) as provider_error:
        CapabilityGrantVerifier(FailingProvider()).verify(
            manifest, request, now=EXPIRES_AT - 1
        )
    assert rejected not in str(provider_error.value)
    assert rejected not in repr(provider_error.value)


def test_schema_version_and_serialized_fields_are_explicit() -> None:
    payload = _manifest().to_dict()

    assert payload["schema_version"] == CAPABILITY_GRANT_SCHEMA_VERSION
    assert set(payload) == {
        "schema_version",
        "constraints",
        "expires_at",
        "key_id",
        "signature",
    }
    assert set(payload["constraints"][0]) == {
        "tool",
        "resource",
        "action",
        "policy_profile",
    }
