"""Synthetic adapter tests; a callback count is the effect/read evidence."""

import json
from dataclasses import replace

import pytest

from openmed.agent.approvals.tokens import (
    ApprovalReplayError,
    ApprovalSignatureError,
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.agent.permissions.access_tickets import AccessTicketRunMismatchError
from openmed.agent.permissions.delegation import DelegationSignatureError
from openmed.agent.permissions.grants import (
    CapabilityGrantScopeError,
    CapabilityGrantSignatureError,
)
from openmed.agent.permissions.revocation import AuthorityBinding, AuthorityDeniedError
from openmed.agent.permissions.runtime_authority import (
    ApprovalAuthorityBinding,
    delegation_authorities,
    dispatch_with_revocable_approval,
    dispatch_with_revocable_delegation,
    dispatch_with_revocable_grant,
    grant_authority,
    read_with_revocable_ticket,
    ticket_authority,
)
from tests.fixtures.agent.authority_status import ACTION_DIGEST, KEY, SyntheticAuthority


def _invoke(fixture, adapter, binding, callback):
    shared = dict(runtime=fixture.runtime, binding=binding)
    if adapter == "grant":
        return dispatch_with_revocable_grant(
            fixture.grant,
            fixture.grant_request,
            fixture.grant_verifier,
            callback,
            **shared,
        )
    if adapter == "ticket":
        return read_with_revocable_ticket(
            fixture.ticket,
            fixture.ticket_request,
            fixture.ticket_verifier,
            callback,
            **shared,
        )
    return dispatch_with_revocable_delegation(
        fixture.chain, fixture.delegation_verifier, callback, **shared
    )


@pytest.mark.parametrize(
    "adapter,index", [("grant", 0), ("ticket", 1), ("delegation", 2)]
)
def test_revocation_between_preview_and_execution_prevents_callback(adapter, index):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    calls = []
    assert _invoke(fixture, adapter, binding, lambda: calls.append("control") or 7) == 7
    fixture.provider.revoke(fixture.contracts[index])
    # Signature, expiry and static scope are still valid after revocation.
    fixture.grant_verifier.verify(fixture.grant, fixture.grant_request)
    fixture.ticket_verifier.verify(fixture.ticket, fixture.ticket_request)
    fixture.delegation_verifier.verify_chain(fixture.chain)
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        _invoke(fixture, adapter, binding, lambda: calls.append("forbidden"))
    assert calls == ["control"]


@pytest.mark.parametrize("ancestor", [0, 1, 2])
def test_revoked_parent_child_or_leaf_invalidates_descendant(ancestor):
    fixture = SyntheticAuthority()
    contracts = delegation_authorities(fixture.chain)
    binding = fixture.runtime.bind(contracts)
    fixture.provider.revoke(contracts[ancestor])
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        dispatch_with_revocable_delegation(
            fixture.chain,
            fixture.delegation_verifier,
            lambda: pytest.fail("revoked descendant dispatched"),
            runtime=fixture.runtime,
            binding=binding,
        )


def test_leaf_only_chain_cannot_hide_revoked_ancestry():
    fixture = SyntheticAuthority()
    with pytest.raises(ValueError, match="incomplete_authority_chain"):
        delegation_authorities((fixture.chain[-1],))


@pytest.mark.parametrize("adapter", ["grant", "ticket", "delegation"])
def test_adapter_rejects_binding_from_other_artifact_or_omitted_parent(adapter):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind((fixture.contracts[-1],))
    with pytest.raises(AuthorityDeniedError, match="authority_contract_mismatch"):
        _invoke(fixture, adapter, binding, lambda: pytest.fail("unbound callback"))


@pytest.mark.parametrize("adapter", ["grant", "ticket", "delegation"])
def test_static_authority_is_not_replaced_by_active_status(adapter):
    fixture = SyntheticAuthority()
    if adapter == "grant":
        fixture.grant_request = replace(
            fixture.grant_request, action="action:org.example/write@1.0.0"
        )
        error = CapabilityGrantScopeError
    elif adapter == "ticket":
        fixture.ticket_request = replace(
            fixture.ticket_request,
            run_id=type(fixture.ticket.run_id)("run_" + "2" * 32),
        )
        error = AccessTicketRunMismatchError
    else:
        forged = replace(fixture.chain[-1], signature="hmac-sha256:" + "0" * 64)
        fixture.chain = (*fixture.chain[:-1], forged)
        fixture.provider.register(delegation_authorities(fixture.chain))
        fixture.contracts = delegation_authorities(fixture.chain)
        error = DelegationSignatureError
    binding = fixture.runtime.bind(fixture.contracts)
    with pytest.raises(error):
        _invoke(
            fixture,
            adapter,
            binding,
            lambda: pytest.fail("invalid scope/signature dispatched"),
        )


def test_forged_grant_with_registered_status_still_fails_signature():
    fixture = SyntheticAuthority()
    fixture.grant = replace(fixture.grant, signature="hmac-sha256:" + "0" * 64)
    contract = grant_authority(fixture.grant)
    fixture.provider.register((contract,))
    binding = fixture.runtime.bind((contract,))
    with pytest.raises(CapabilityGrantSignatureError):
        _invoke(
            fixture, "grant", binding, lambda: pytest.fail("forged grant dispatched")
        )


@pytest.mark.parametrize("adapter", ["grant", "ticket", "delegation"])
def test_static_and_live_expiry_emit_content_free_expiry_receipt(adapter):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    fixture.clock.now = 100
    with pytest.raises(AuthorityDeniedError, match="authority_expired") as error:
        _invoke(fixture, adapter, binding, lambda: pytest.fail("expired callback"))
    assert error.value.receipt.to_dict()["reason_code"] == "authority_expired"


@pytest.mark.parametrize("adapter", ["grant", "ticket", "delegation"])
def test_status_outage_prevents_every_adapter_callback(adapter):
    fixture = SyntheticAuthority()
    binding = fixture.runtime.bind(fixture.contracts)
    fixture.provider.unavailable = True
    with pytest.raises(AuthorityDeniedError, match="authority_status_unavailable"):
        _invoke(fixture, adapter, binding, lambda: pytest.fail("outage callback"))


def _approval(fixture, *, store=None):
    token = ApprovalTokenSigner(KEY, clock=fixture.clock).issue(
        action_digest=ACTION_DIGEST,
        reviewer_role="role:org.example/reviewer@1.0.0",
        expires_at=100,
    )
    verifier = ApprovalTokenVerifier(
        KEY, store or InMemoryApprovalNonceStore(), clock=fixture.clock
    )
    binding = ApprovalAuthorityBinding.create(
        token, fixture.runtime.bind(fixture.contracts)
    )
    return token, verifier, binding


def _approve(fixture, token, verifier, binding, dispatch):
    return dispatch_with_revocable_approval(
        token,
        action_digest=ACTION_DIGEST,
        reviewer_role=token.reviewer_role,
        verifier=verifier,
        dispatch=dispatch,
        runtime=fixture.runtime,
        binding=binding,
    )


def test_approval_is_bound_to_preview_and_permanently_consumed_on_revocation():
    fixture = SyntheticAuthority()
    token, verifier, binding = _approval(fixture)
    fixture.provider.revoke(fixture.contracts[0])
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        _approve(
            fixture,
            token,
            verifier,
            binding,
            lambda: pytest.fail("revoked approval dispatched"),
        )
    with pytest.raises(ApprovalReplayError):
        _approve(
            fixture, token, verifier, binding, lambda: pytest.fail("approval replayed")
        )


def test_revocation_during_approval_consumption_is_rechecked_before_effect():
    fixture = SyntheticAuthority()

    class RevokingStore(InMemoryApprovalNonceStore):
        def claim(self, *args, **kwargs):
            result = super().claim(*args, **kwargs)
            fixture.provider.revoke(fixture.contracts[1])
            return result

    token, verifier, binding = _approval(fixture, store=RevokingStore())
    with pytest.raises(AuthorityDeniedError, match="authority_revoked"):
        _approve(
            fixture,
            token,
            verifier,
            binding,
            lambda: pytest.fail("claim race dispatched"),
        )


def test_changed_active_generation_invalidates_outstanding_approval():
    fixture = SyntheticAuthority()
    token, verifier, binding = _approval(fixture)
    contract = fixture.contracts[1]
    fixture.provider.records[(contract.kind, contract.digest)] = (1, False)
    restored = ApprovalAuthorityBinding.from_dict(
        json.loads(json.dumps(binding.to_dict()))
    )
    with pytest.raises(AuthorityDeniedError, match="authority_generation_changed"):
        _approve(
            fixture,
            token,
            verifier,
            restored,
            lambda: pytest.fail("stale approval dispatched"),
        )
    # A newly reviewed token, issued against a newly captured generation, succeeds.
    fresh_token, fresh_verifier, fresh_binding = _approval(fixture)
    value, receipt = _approve(
        fixture, fresh_token, fresh_verifier, fresh_binding, lambda: 7
    )
    assert value == 7 and receipt.action_digest == ACTION_DIGEST


def test_different_or_tampered_token_cannot_use_approval_binding():
    fixture = SyntheticAuthority()
    token, verifier, binding = _approval(fixture)
    changed = replace(token, nonce="nonce_" + "b" * 32)
    with pytest.raises(AuthorityDeniedError, match="authority_approval_mismatch"):
        _approve(
            fixture,
            changed,
            verifier,
            binding,
            lambda: pytest.fail("substituted token"),
        )
    forged_binding = ApprovalAuthorityBinding.create(changed, binding.authority)
    with pytest.raises(ApprovalSignatureError):
        _approve(
            fixture,
            changed,
            verifier,
            forged_binding,
            lambda: pytest.fail("forged approval"),
        )


def test_expired_approval_and_success_control_with_nested_authority_adapters():
    fixture = SyntheticAuthority()
    token, verifier, binding = _approval(fixture)
    calls = []
    result, _ = _approve(
        fixture,
        token,
        verifier,
        binding,
        lambda: _invoke(
            fixture,
            "grant",
            binding.authority,
            lambda: _invoke(
                fixture,
                "ticket",
                binding.authority,
                lambda: calls.append("effect") or 7,
            ),
        ),
    )
    assert result == 7 and calls == ["effect"]
    token, verifier, binding = _approval(fixture)
    fixture.clock.now = 100
    with pytest.raises(AuthorityDeniedError, match="authority_expired"):
        _approve(
            fixture, token, verifier, binding, lambda: pytest.fail("expired approval")
        )


def test_ticket_reference_commits_to_every_existing_authority_dimension():
    fixture = SyntheticAuthority()
    original = ticket_authority(fixture.ticket)
    changes = [
        dict(run_id=type(fixture.ticket.run_id)("run_" + "2" * 32)),
        dict(purpose="purpose:org.example/other@1.0.0"),
        dict(permitted_data_classes=("data:org.example/other@1.0.0",)),
        dict(
            record_selectors=(
                replace(
                    fixture.ticket.record_selectors[0], digest="hmac-sha256:" + "f" * 64
                ),
            )
        ),
        dict(
            permitted_tool_actions=(
                replace(
                    fixture.ticket.permitted_tool_actions[0],
                    action="action:org.example/write@1.0.0",
                ),
            )
        ),
        dict(expires_at=99),
    ]
    assert all(
        ticket_authority(replace(fixture.ticket, **change)) != original
        for change in changes
    )


def test_approval_binding_malformed_recovery_data_is_rejected():
    fixture = SyntheticAuthority()
    _, _, binding = _approval(fixture)
    for field in binding.to_dict():
        payload = binding.to_dict()
        del payload[field]
        with pytest.raises(ValueError, match="invalid_approval_authority_binding"):
            ApprovalAuthorityBinding.from_dict(payload)
    payload = binding.to_dict()
    payload["payload"] = "synthetic-private-payload"
    with pytest.raises(ValueError, match="invalid_approval_authority_binding"):
        ApprovalAuthorityBinding.from_dict(payload)


def test_approval_expiry_during_fresh_status_lookup_prevents_effect():
    fixture = SyntheticAuthority()
    token = ApprovalTokenSigner(KEY, clock=fixture.clock).issue(
        action_digest=ACTION_DIGEST,
        reviewer_role="role:org.example/reviewer@1.0.0",
        expires_at=20,
    )
    verifier = ApprovalTokenVerifier(
        KEY, InMemoryApprovalNonceStore(), clock=fixture.clock
    )
    binding = ApprovalAuthorityBinding.create(
        token, fixture.runtime.bind(fixture.contracts)
    )
    lookup = fixture.provider.get_status

    def slow_status(kind, digest):
        fixture.clock.now = 20
        return lookup(kind, digest)

    fixture.provider.get_status = slow_status
    calls = []
    with pytest.raises(AuthorityDeniedError, match="authority_expired"):
        _approve(fixture, token, verifier, binding, lambda: calls.append("effect"))
    assert calls == []
    fixture.clock.now = 10  # Verify the burned nonce within its original valid window.
    with pytest.raises(ApprovalReplayError):
        _approve(fixture, token, verifier, binding, lambda: calls.append("replay"))


def test_serializable_receipt_cannot_supply_execution_validity():
    fixture = SyntheticAuthority()
    token, verifier, binding = _approval(fixture)
    consume = verifier.consume_authorization

    def public_receipt(*args, **kwargs):
        return consume(*args, **kwargs).receipt

    verifier.consume_authorization = public_receipt
    with pytest.raises(AuthorityDeniedError, match="authority_approval_mismatch"):
        _approve(
            fixture, token, verifier, binding, lambda: pytest.fail("receipt dispatched")
        )
