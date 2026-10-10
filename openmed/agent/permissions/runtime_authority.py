"""Revocation adapters to existing grant, ticket, approval and recovery APIs.

Legacy static verifiers remain unchanged. Governed hosts must use these runtime
boundaries for every sensitive callback and retain preview bindings faithfully.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

from openmed.agent.approvals.tokens import (
    ApprovalAuthorization,
    ApprovalExpiredError,
    ApprovalReceipt,
    ApprovalToken,
    ApprovalTokenVerifier,
)
from openmed.agent.workflows.recovery import (
    EffectObservation,
    RecoveryCheckpoint,
    RecoveryDecision,
    recover_workflow,
)

from .access_tickets import (
    AccessTicket,
    AccessTicketExpiredError,
    AccessTicketRequest,
    AccessTicketVerifier,
)
from .delegation import DelegationExpiredError, DelegationGrant, DelegationGrantVerifier
from .grants import (
    CapabilityGrantExpiredError,
    CapabilityGrantManifest,
    CapabilityGrantRequest,
    CapabilityGrantVerifier,
)
from .revocation import (
    AuthorityBinding,
    AuthorityBoundary,
    AuthorityContract,
    AuthorityKind,
    AuthorityReason,
    AuthorityRuntime,
)

_T = TypeVar("_T")


def grant_authority(manifest: CapabilityGrantManifest) -> AuthorityContract:
    """Identify the complete signed grant; this does not verify its signature."""
    if type(manifest) is not CapabilityGrantManifest:
        raise ValueError("invalid_authority_grant")
    return AuthorityContract(
        AuthorityKind.GRANT, _sha256(manifest.to_json()), manifest.expires_at
    )


def ticket_authority(ticket: AccessTicket) -> AuthorityContract:
    """Commit to every ticket field without exposing record-selector content."""
    if type(ticket) is not AccessTicket:
        raise ValueError("invalid_authority_ticket")
    payload = {
        "schema_version": "openmed.agent.purpose_ticket_reference.v1",
        "run_id": ticket.run_id.serialize(),
        "purpose": ticket.purpose,
        "data_classes": list(ticket.permitted_data_classes),
        "selectors": [
            {"kind": s.kind, "digest": s.digest} for s in ticket.record_selectors
        ],
        "tool_actions": [
            {"tool": a.tool, "action": a.action} for a in ticket.permitted_tool_actions
        ],
        "expires_at": ticket.expires_at,
    }
    return AuthorityContract(
        AuthorityKind.PURPOSE_TICKET,
        _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))),
        ticket.expires_at,
    )


def delegation_authorities(
    chain: Sequence[DelegationGrant],
) -> tuple[AuthorityContract, ...]:
    """Reference every ancestor; static chain verification remains mandatory."""
    if not chain or any(type(grant) is not DelegationGrant for grant in chain):
        raise ValueError("invalid_authority_chain")
    if chain[0].parent_digest is not None:
        raise ValueError("incomplete_authority_chain")
    return tuple(
        AuthorityContract(AuthorityKind.DELEGATION, grant.digest(), grant.expires_at)
        for grant in chain
    )


@dataclass(frozen=True, slots=True)
class ApprovalAuthorityBinding:
    """Pin one outstanding token to its preview's authority generations."""

    token_digest: str
    authority: AuthorityBinding

    def __post_init__(self) -> None:
        # Reuse the contract's strict digest validation without storing a token.
        AuthorityContract(AuthorityKind.GRANT, self.token_digest, 0)
        if type(self.authority) is not AuthorityBinding:
            raise ValueError("invalid_approval_authority_binding")

    @classmethod
    def create(
        cls, token: ApprovalToken, authority: AuthorityBinding
    ) -> ApprovalAuthorityBinding:
        """Bind at issuance/preview; never replace this state at execution time."""
        if type(token) is not ApprovalToken:
            raise ValueError("invalid_authority_approval")
        return cls(_sha256(token.to_json()), authority)

    def to_dict(self) -> dict[str, Any]:
        """Retain digest-only approval association alongside protected run state."""
        return {
            "schema_version": "openmed.agent.approval_authority.v1",
            "token_digest": self.token_digest,
            "authority": self.authority.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ApprovalAuthorityBinding:
        """Restore exact pinned state, rejecting missing or extra fields."""
        try:
            if set(payload) != {"schema_version", "token_digest", "authority"} or (
                payload["schema_version"] != "openmed.agent.approval_authority.v1"
            ):
                raise ValueError
            return cls(
                payload["token_digest"],
                AuthorityBinding.from_dict(payload["authority"]),
            )
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:
            raise ValueError("invalid_approval_authority_binding") from None


def dispatch_with_revocable_grant(
    manifest: CapabilityGrantManifest,
    request: CapabilityGrantRequest,
    verifier: CapabilityGrantVerifier,
    dispatch: Callable[[], _T],
    *,
    runtime: AuthorityRuntime,
    binding: AuthorityBinding,
    now: int | None = None,
) -> _T:
    """Verify exact static scope and fresh authority immediately before an effect."""
    _callback(dispatch)
    boundary = AuthorityBoundary.EFFECT_DISPATCH
    runtime.require_contracts(binding, (grant_authority(manifest),), boundary)
    try:
        verifier.verify(manifest, request, now=now)
    except CapabilityGrantExpiredError:
        runtime.deny(binding, boundary, AuthorityReason.EXPIRED)
    runtime.check(binding, boundary)
    return dispatch()


def read_with_revocable_ticket(
    ticket: AccessTicket,
    request: AccessTicketRequest,
    verifier: AccessTicketVerifier,
    read: Callable[[], _T],
    *,
    runtime: AuthorityRuntime,
    binding: AuthorityBinding,
    now: int | None = None,
) -> _T:
    """Verify purpose, run, projection and current ticket status before a read."""
    _callback(read)
    boundary = AuthorityBoundary.SENSITIVE_READ
    runtime.require_contracts(binding, (ticket_authority(ticket),), boundary)
    try:
        verifier.verify(ticket, request, now=now)
    except AccessTicketExpiredError:
        runtime.deny(binding, boundary, AuthorityReason.EXPIRED)
    runtime.check(binding, boundary)
    return read()


def dispatch_with_revocable_delegation(
    chain: Sequence[DelegationGrant],
    verifier: DelegationGrantVerifier,
    dispatch: Callable[[], _T],
    *,
    runtime: AuthorityRuntime,
    binding: AuthorityBinding,
    now: int | None = None,
) -> _T:
    """Verify complete ancestry and reject any revoked parent before dispatch."""
    _callback(dispatch)
    chain = tuple(chain)
    boundary = AuthorityBoundary.EFFECT_DISPATCH
    runtime.require_contracts(binding, delegation_authorities(chain), boundary)
    try:
        verifier.verify_chain(chain, now=now)
    except DelegationExpiredError:
        runtime.deny(binding, boundary, AuthorityReason.EXPIRED)
    runtime.check(binding, boundary)
    return dispatch()


def dispatch_with_revocable_approval(
    token: ApprovalToken,
    *,
    action_digest: str,
    reviewer_role: str,
    verifier: ApprovalTokenVerifier,
    dispatch: Callable[[], _T],
    runtime: AuthorityRuntime,
    binding: ApprovalAuthorityBinding,
    now: int | None = None,
) -> tuple[_T, ApprovalReceipt]:
    """Consume exact approval then recheck its original authority generations.

    A revocation during nonce consumption prevents dispatch and burns the
    approval. Generation changes require a fresh preview and human approval.
    Compose with grant/ticket/delegation adapters for static scope checks.
    """
    _callback(dispatch)
    boundary = AuthorityBoundary.EFFECT_DISPATCH
    if (
        type(binding) is not ApprovalAuthorityBinding
        or type(token) is not ApprovalToken
    ):
        raise ValueError("invalid_approval_authority_binding")
    if _sha256(token.to_json()) != binding.token_digest:
        runtime.deny(binding.authority, boundary, AuthorityReason.APPROVAL_MISMATCH)
    try:
        authorization = verifier.consume_authorization(
            token, action_digest=action_digest, reviewer_role=reviewer_role, now=now
        )
    except ApprovalExpiredError:
        runtime.deny(binding.authority, boundary, AuthorityReason.EXPIRED)
    if type(authorization) is not ApprovalAuthorization:
        runtime.deny(binding.authority, boundary, AuthorityReason.APPROVAL_MISMATCH)
    runtime.check(
        binding.authority, boundary, approval_expires_at=authorization.expires_at
    )
    return dispatch(), authorization.receipt


def recover_with_revocable_authority(
    checkpoints: Iterable[RecoveryCheckpoint],
    observations: Iterable[EffectObservation],
    *,
    runtime: AuthorityRuntime,
    binding: AuthorityBinding,
    now: int,
) -> RecoveryDecision:
    """Check restored generations before consuming recovery state/observations.

    This is read-only reconciliation, not execution. Every subsequent read or
    effect must pass its own adapter check. Never re-bind restored authority to
    a newer active generation or restore the generation store from run data.
    """
    runtime.check(binding, AuthorityBoundary.RESUME)
    return recover_workflow(checkpoints, observations, now=now)


def _sha256(serialized: str) -> str:
    return "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _callback(callback: Callable[[], object]) -> None:
    if not callable(callback):
        raise ValueError("invalid_authority_callback")
