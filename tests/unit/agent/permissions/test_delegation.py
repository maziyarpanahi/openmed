"""Offline tests for non-amplifying clinical-agent delegation."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest

from openmed.agent.permissions import CapabilityGrantConstraint
from openmed.agent.permissions.delegation import (
    DELEGATION_AUDIT_SCHEMA_VERSION,
    DelegationChainError,
    DelegationCycleError,
    DelegationDepthError,
    DelegationExpiredError,
    DelegationGrant,
    DelegationGrantSigner,
    DelegationGrantVerifier,
    DelegationRequest,
    DelegationScope,
    DelegationScopeError,
    DelegationSignatureError,
    DelegationValidationError,
)

KEY = b"local-delegation-test-key-32-bytes"
OTHER_KEY = b"other-delegation-test-key-32bytes"
EXPIRES_AT = 2_000_000_000
PARENT = "agent:org.example/coordinator@1.0.0"
CHILD = "agent:org.example/redactor@1.0.0"
GRANDCHILD = "agent:org.example/reviewer@1.0.0"
PURPOSE = "purpose:org.example/care-summary@1.0.0"
OTHER_PURPOSE = "purpose:org.example/safety-review@1.0.0"
DATA_CLASS = "data:org.example/medications@1.0.0"
OTHER_DATA_CLASS = "data:org.example/demographics@1.0.0"


def _capability(
    *,
    tool: str = "tool:org.example/redact@1.0.0",
    resource: str = "resource:org.example/clinical-document@1.0.0",
    action: str = "action:org.example/read@1.0.0",
    policy_profile: str = "policy:org.example/minimum-necessary@1.0.0",
) -> CapabilityGrantConstraint:
    return CapabilityGrantConstraint(tool, resource, action, policy_profile)


def _other_capability() -> CapabilityGrantConstraint:
    return _capability(
        tool="tool:org.example/summarize@1.0.0",
        action="action:org.example/derive@1.0.0",
    )


def _scope(
    *,
    capabilities: tuple[CapabilityGrantConstraint, ...] | None = None,
    data_classes: tuple[str, ...] | None = None,
    purposes: tuple[str, ...] | None = None,
) -> DelegationScope:
    return DelegationScope(
        capabilities=capabilities or (_capability(),),
        data_classes=data_classes or (DATA_CLASS,),
        purposes=purposes or (PURPOSE,),
    )


def _root(
    *,
    scope: DelegationScope | None = None,
    expires_at: int = EXPIRES_AT,
    remaining_depth: int = 3,
) -> DelegationGrant:
    return DelegationGrantSigner(KEY).issue_root(
        principal=PARENT,
        scope=scope or _scope(),
        expires_at=expires_at,
        remaining_depth=remaining_depth,
    )


def _request(
    *,
    principal: str = CHILD,
    scope: DelegationScope | None = None,
    expires_at: int = EXPIRES_AT - 100,
    remaining_depth: int = 1,
) -> DelegationRequest:
    return DelegationRequest(
        principal=principal,
        scope=scope or _scope(),
        expires_at=expires_at,
        remaining_depth=remaining_depth,
    )


def _derive(
    parent: DelegationGrant,
    request: DelegationRequest | None = None,
) -> DelegationGrant:
    return (
        DelegationGrantSigner(KEY)
        .derive_child(
            parent,
            request or _request(),
            DelegationGrantVerifier(KEY),
            now=EXPIRES_AT - 1_000,
        )
        .grant
    )


def _resign(payload: dict[str, Any], key: bytes = KEY) -> DelegationGrant:
    payload["signature"] = ""
    unsigned = DelegationGrant.from_dict(payload)
    signing_payload = unsigned.signing_payload()
    encoded = json.dumps(
        signing_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    signed_payload = unsigned.to_dict()
    signed_payload["signature"] = (
        "hmac-sha256:" + hmac.new(key, encoded, hashlib.sha256).hexdigest()
    )
    return DelegationGrant.from_dict(signed_payload)


def test_root_grants_are_canonical_signed_and_deterministic() -> None:
    broad_scope = _scope(
        capabilities=(_other_capability(), _capability()),
        data_classes=(OTHER_DATA_CLASS, DATA_CLASS),
        purposes=(OTHER_PURPOSE, PURPOSE),
    )
    reverse_scope = _scope(
        capabilities=(_capability(), _other_capability()),
        data_classes=(DATA_CLASS, OTHER_DATA_CLASS),
        purposes=(PURPOSE, OTHER_PURPOSE),
    )

    first = _root(scope=broad_scope)
    second = _root(scope=reverse_scope)

    assert first == second
    assert first.to_json() == second.to_json()
    assert DelegationGrant.from_json(first.to_json()) == first
    assert first.digest().startswith("sha256:")
    assert DelegationGrantVerifier(KEY).verify(first, now=EXPIRES_AT - 1) == first


def test_child_is_strict_intersection_of_scope_validity_and_depth() -> None:
    parent_scope = _scope(
        capabilities=(_capability(), _other_capability()),
        data_classes=(DATA_CLASS, OTHER_DATA_CLASS),
        purposes=(PURPOSE, OTHER_PURPOSE),
    )
    requested_scope = _scope(
        capabilities=(
            _other_capability(),
            _capability(tool="tool:org.example/export@1.0.0"),
        ),
        data_classes=(
            OTHER_DATA_CLASS,
            "data:org.example/laboratory@1.0.0",
        ),
        purposes=(
            OTHER_PURPOSE,
            "purpose:org.example/research-export@1.0.0",
        ),
    )

    decision = DelegationGrantSigner(KEY).derive_child(
        _root(scope=parent_scope, remaining_depth=4),
        _request(
            scope=requested_scope,
            expires_at=EXPIRES_AT + 500,
            remaining_depth=9,
        ),
        DelegationGrantVerifier(KEY),
        now=EXPIRES_AT - 1_000,
    )

    assert decision.grant.scope == _scope(
        capabilities=(_other_capability(),),
        data_classes=(OTHER_DATA_CLASS,),
        purposes=(OTHER_PURPOSE,),
    )
    assert decision.grant.expires_at == EXPIRES_AT
    assert decision.grant.remaining_depth == 3
    assert decision.grant.principals == (PARENT, CHILD)
    assert (
        decision.grant.parent_digest
        == _root(scope=parent_scope, remaining_depth=4).digest()
    )
    assert decision.audit_record.to_dict() == {
        "schema_version": DELEGATION_AUDIT_SCHEMA_VERSION,
        "reason_code": "delegation_derived",
        "parent_grant_digest": decision.grant.parent_digest,
        "child_grant_digest": decision.grant.digest(),
    }


def test_complete_chain_verifies_locally() -> None:
    root = _root()
    child = _derive(root, _request(remaining_depth=1))
    grandchild = _derive(
        child,
        _request(principal=GRANDCHILD, expires_at=EXPIRES_AT - 200),
    )

    assert (
        DelegationGrantVerifier(KEY).verify_chain(
            (root, child, grandchild), now=EXPIRES_AT - 1_000
        )
        == grandchild
    )


@pytest.mark.parametrize(
    ("field", "value", "error_type", "code"),
    [
        (
            "expires_at",
            EXPIRES_AT + 1,
            DelegationExpiredError,
            "validity_amplified",
        ),
        ("remaining_depth", 3, DelegationDepthError, "depth_amplified"),
        (
            "parent_digest",
            "sha256:" + "0" * 64,
            DelegationChainError,
            "parent_mismatch",
        ),
    ],
)
def test_signed_child_chain_rejects_amplified_or_unbound_metadata(
    field: str,
    value: object,
    error_type: type[Exception],
    code: str,
) -> None:
    root = _root(remaining_depth=3)
    payload = _derive(root).to_dict()
    payload[field] = value
    child = _resign(payload)

    with pytest.raises(error_type) as caught:
        DelegationGrantVerifier(KEY).verify_link(root, child, now=EXPIRES_AT - 1_000)

    assert getattr(caught.value, "code") == code


def test_signed_child_chain_rejects_scope_amplification() -> None:
    root = _root()
    payload = _derive(root).to_dict()
    payload["scope"]["data_classes"].append(OTHER_DATA_CLASS)
    child = _resign(payload)

    with pytest.raises(DelegationScopeError) as caught:
        DelegationGrantVerifier(KEY).verify_link(root, child, now=EXPIRES_AT - 1_000)

    assert caught.value.code == "scope_amplified"


def test_tampered_child_fails_signature_before_chain_fields_are_trusted() -> None:
    child = _derive(_root())
    payload = child.to_dict()
    payload["remaining_depth"] = 0

    with pytest.raises(DelegationSignatureError):
        DelegationGrantVerifier(KEY).verify(payload, now=EXPIRES_AT - 1_000)


def test_cycle_depth_and_empty_intersection_fail_closed() -> None:
    with pytest.raises(DelegationCycleError) as cycle:
        DelegationGrantSigner(KEY).derive_child(
            _root(),
            _request(principal=PARENT),
            DelegationGrantVerifier(KEY),
            now=EXPIRES_AT - 1_000,
        )
    assert cycle.value.code == "cyclic_delegation"

    with pytest.raises(DelegationDepthError) as depth:
        _derive(_root(remaining_depth=0))
    assert depth.value.code == "depth_exhausted"

    disjoint = _scope(
        capabilities=(_other_capability(),),
        data_classes=(OTHER_DATA_CLASS,),
        purposes=(OTHER_PURPOSE,),
    )
    with pytest.raises(DelegationScopeError) as scope:
        _derive(_root(), _request(scope=disjoint))
    assert scope.value.code == "empty_scope_intersection"


def test_expired_intersection_is_rejected() -> None:
    with pytest.raises(DelegationExpiredError) as caught:
        DelegationGrantSigner(KEY).derive_child(
            _root(),
            _request(expires_at=EXPIRES_AT - 1_000),
            DelegationGrantVerifier(KEY),
            now=EXPIRES_AT - 1_000,
        )

    assert caught.value.code == "expired"


def test_audit_and_errors_contain_only_digests_and_reason_codes() -> None:
    rejected = "purpose:org.example/research-export@1.0.0"
    request = _request(
        scope=_scope(
            capabilities=(_other_capability(),),
            data_classes=(OTHER_DATA_CLASS,),
            purposes=(rejected,),
        )
    )

    with pytest.raises(DelegationScopeError) as caught:
        _derive(_root(), request)

    serialized = caught.value.audit_record.to_json()
    assert caught.value.audit_record.parent_grant_digest is not None
    assert caught.value.audit_record.child_grant_digest is None
    assert set(json.loads(serialized)) == {
        "schema_version",
        "reason_code",
        "parent_grant_digest",
        "child_grant_digest",
    }
    for value in (
        rejected,
        request.principal,
        request.scope.data_classes[0],
        request.scope.capabilities[0].tool,
    ):
        assert value not in str(caught.value)
        assert value not in repr(caught.value)
        assert value not in serialized
        assert value not in repr(request)


def test_wrong_key_and_clock_failure_do_not_echo_provider_details() -> None:
    root = _root()
    with pytest.raises(DelegationSignatureError):
        DelegationGrantVerifier(OTHER_KEY).verify(root, now=EXPIRES_AT - 1)

    rejected = "sensitive-clock-provider-detail"

    def failing_clock() -> int:
        raise RuntimeError(rejected)

    with pytest.raises(DelegationValidationError) as caught:
        DelegationGrantVerifier(KEY, clock=failing_clock).verify(root)

    assert caught.value.code == "clock_unavailable"
    assert rejected not in str(caught.value)
    assert rejected not in caught.value.audit_record.to_json()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DelegationScope((_capability(),), (), (PURPOSE,)),
        lambda: _scope(purposes=("free-text-purpose",)),
        lambda: _scope(capabilities=(_capability(), _capability())),
        lambda: _request(principal="agent-invalid"),
        lambda: _request(remaining_depth=True),
        lambda: _root(expires_at=True),
        lambda: DelegationGrant.from_dict({}),
    ],
)
def test_malformed_delegation_metadata_fails_closed(factory: Any) -> None:
    with pytest.raises(DelegationValidationError):
        factory()
