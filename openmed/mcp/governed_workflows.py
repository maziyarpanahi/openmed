"""Value-free governance observations and human-channel review requests.

Services are trusted local code. They own grant, purpose and projection policy,
durable receipt verification, run isolation and atomic handoff creation. No
method on this protocol executes an action or writes a clinical target.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Mapping
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol

from openmed.agent.action_phases import ActionPhase
from openmed.agent.artifact_reference import ArtifactKind
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.reviewer_handoff import ReviewerHandoffPacket
from openmed.mcp.consent_receipts import (
    ConsentReceipt,
    ConsentReceiptPolicy,
    ConsentReceiptVerifier,
    MappingConsentKeyProvider,
)

GOVERNED_MCP_REQUEST_VERSION = "openmed.mcp.governance_request.v1"
GOVERNED_MCP_RESULT_VERSION = "openmed.mcp.governance_result.v1"
GOVERNED_MCP_OPERATIONS: Mapping[str, str] = MappingProxyType(
    {
        "openmed_workflow_preflight": "preflight",
        "openmed_workflow_preview": "preview",
        "openmed_workflow_status": "status",
        "openmed_workflow_request_review": "request_review",
    }
)
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "workflow_id",
        "action_digest",
        "expected_state_digest",
    }
)
_ERROR_CODES = frozenset(
    {
        "invalid_arguments",
        "governance_unavailable",
        "governance_failed",
        "governance_conflict",
        "governance_denied",
        "governance_invalid_result",
    }
)


class GovernedMCPError(ValueError):
    """Fixed diagnostic with a closed code and no submitted values."""

    def __init__(self, code: str) -> None:
        if code not in _ERROR_CODES:
            raise ValueError("Invalid governance diagnostic code.")
        self.code = code
        super().__init__("Governed workflow request refused.")


class GovernanceDecision(str, Enum):
    """Closed observation of a service-owned policy decision."""

    ALLOWED = "allowed"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"


class GovernanceStatus(str, Enum):
    """Closed run outcome; no outcome itself grants execution authority."""

    READY = "ready"
    DENIED = "denied"
    REVIEW_REQUIRED = "review_required"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


class ReceiptObservation(str, Enum):
    """Service-owned, action-bound receipt observation without a token."""

    ABSENT = "absent"
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    EXPIRED = "expired"
    CONFLICT = "conflict"


def _digest(value: Any, *, optional: bool = False) -> None:
    if optional and value is None:
        return
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise GovernedMCPError("invalid_arguments")


@dataclass(frozen=True, repr=False)
class GovernedMCPRequest:
    """Opaque reference to a service-custodied action and expected run state.

    Args:
        run_id: Opaque run identifier, never a patient-derived identifier.
        workflow_id: Canonical developer-owned workflow identifier.
        action_digest: Digest of the exact proposed action held by the service.
        expected_state_digest: Required compare-and-set state for review requests.
    """

    run_id: RunId
    workflow_id: WorkflowId
    action_digest: str
    expected_state_digest: str | None = None

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId or type(self.workflow_id) is not WorkflowId:
            raise GovernedMCPError("invalid_arguments")
        RunId.parse(self.run_id.serialize())
        WorkflowId.parse(self.workflow_id.serialize())
        _digest(self.action_digest)
        _digest(self.expected_state_digest, optional=True)

    @classmethod
    def from_dict(cls, value: Any) -> GovernedMCPRequest:
        """Reject inline authority, unknown fields and malformed references."""
        if type(value) is not dict or value.keys() != _REQUEST_FIELDS:
            raise GovernedMCPError("invalid_arguments")
        if value["schema_version"] != GOVERNED_MCP_REQUEST_VERSION:
            raise GovernedMCPError("invalid_arguments")
        try:
            return cls(
                RunId.parse(value["run_id"]),
                WorkflowId.parse(value["workflow_id"]),
                value["action_digest"],
                value["expected_state_digest"],
            )
        except (TypeError, ValueError):
            pass
        raise GovernedMCPError("invalid_arguments")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete metadata-only request contract."""
        return {
            "schema_version": GOVERNED_MCP_REQUEST_VERSION,
            "run_id": self.run_id.serialize(),
            "workflow_id": self.workflow_id.serialize(),
            "action_digest": self.action_digest,
            "expected_state_digest": self.expected_state_digest,
        }


@dataclass(frozen=True, repr=False)
class GovernedMCPView:
    """Immutable policy, preview and run observation bound to one action.

    Decision and receipt fields report trusted service observations. They never
    convey a grant, a purpose ticket, reviewer identity or execution approval.
    """

    run_id: RunId
    workflow_id: WorkflowId
    action_digest: str
    state_digest: str
    phase: ActionPhase
    status: GovernanceStatus
    grant: GovernanceDecision
    purpose_ticket: GovernanceDecision
    projection: GovernanceDecision
    proposed_effect_count: int = 0
    committed_effect_count: int = 0
    receipt_verification: ReceiptObservation = ReceiptObservation.ABSENT
    review_request_digest: str | None = None

    def __post_init__(self) -> None:
        GovernedMCPRequest(self.run_id, self.workflow_id, self.action_digest)
        _digest(self.state_digest)
        _digest(self.review_request_digest, optional=True)
        if (
            type(self.phase) is not ActionPhase
            or type(self.status) is not GovernanceStatus
            or type(self.receipt_verification) is not ReceiptObservation
            or any(
                type(v) is not GovernanceDecision
                for v in (self.grant, self.purpose_ticket, self.projection)
            )
            or any(
                type(v) is not int or not 0 <= v <= 1024
                for v in (self.proposed_effect_count, self.committed_effect_count)
            )
            or self.committed_effect_count > self.proposed_effect_count
        ):
            raise GovernedMCPError("governance_invalid_result")
        if self.status is GovernanceStatus.READY and any(
            v is not GovernanceDecision.ALLOWED
            for v in (self.grant, self.purpose_ticket, self.projection)
        ):
            raise GovernedMCPError("governance_invalid_result")
        required_phase = {
            GovernanceStatus.COMPLETED: ActionPhase.COMPLETED,
            GovernanceStatus.CANCELLED: ActionPhase.ABORTED,
            GovernanceStatus.REVIEW_REQUIRED: ActionPhase.WAITING_REVIEW,
        }.get(self.status)
        if required_phase is not None and self.phase is not required_phase:
            raise GovernedMCPError("governance_invalid_result")

    def to_dict(self) -> dict[str, Any]:
        """Return bounded codes, counts and digests without private content."""
        return {
            "schema_version": GOVERNED_MCP_RESULT_VERSION,
            "run_id": self.run_id.serialize(),
            "workflow_id": self.workflow_id.serialize(),
            "action_digest": self.action_digest,
            "state_digest": self.state_digest,
            "phase": self.phase.value,
            "status": self.status.value,
            "authority": {
                "grant": self.grant.value,
                "purpose_ticket": self.purpose_ticket.value,
                "projection": self.projection.value,
            },
            "proposed_effect_count": self.proposed_effect_count,
            "committed_effect_count": self.committed_effect_count,
            "receipt_verification": self.receipt_verification.value,
            "review_request_digest": self.review_request_digest,
        }


@dataclass(frozen=True, repr=False)
class GovernedMCPReview:
    """Service-created human handoff; only its digest crosses the MCP boundary."""

    view: GovernedMCPView
    packet: ReviewerHandoffPacket


class GovernedMCPService(Protocol):
    """Trusted governance custody with no action-execution capability.

    Read methods must perform no writes. ``request_review`` must atomically
    compare the exact run/action/state, create only a human handoff, and never
    approve, resume, cancel or execute an action. Receipt observations must come
    from durable verification for that action; MCP caller claims are not input.
    """

    def preflight(self, request: GovernedMCPRequest) -> GovernedMCPView:
        """Observe grant, purpose ticket and minimum-data projection decisions."""
        ...

    def preview(self, request: GovernedMCPRequest) -> GovernedMCPView:
        """Inspect an existing proposal without its values or clinical targets."""
        ...

    def status(self, request: GovernedMCPRequest) -> GovernedMCPView:
        """Observe run state and value-free, action-bound receipt verification."""
        ...

    def request_review(self, request: GovernedMCPRequest) -> GovernedMCPReview:
        """Create one human handoff using an atomic state comparison."""
        ...


GovernedConsentProvider = Callable[[str, Mapping[str, Any]], ConsentReceipt | None]


def default_governed_consent_policy() -> ConsentReceiptPolicy:
    """Return a receipt-required policy with no keys and no implicit consent."""
    return ConsentReceiptPolicy(
        verifier=ConsentReceiptVerifier(MappingConsentKeyProvider({})),
        client="openmed-governance",
        resource="urn:openmed:governed-workflows",
        scope="mcp.governance.review",
        require_receipt=True,
    )


def _call(service: Any, method: str, request: GovernedMCPRequest) -> Any:
    if service is None or not callable(getattr(service, method, None)):
        raise GovernedMCPError("governance_unavailable")
    code = "governance_failed"
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            try:
                return getattr(service, method)(request)
            except NotImplementedError:
                code = "governance_unavailable"
            except GovernedMCPError as error:
                if error.code in _ERROR_CODES:
                    code = error.code
            except BaseException:
                pass
    raise GovernedMCPError(code)


def _view(value: Any, request: GovernedMCPRequest) -> GovernedMCPView:
    if type(value) is not GovernedMCPView:
        raise GovernedMCPError("governance_invalid_result")
    try:
        checked = replace(value)
    except (TypeError, ValueError):
        raise GovernedMCPError("governance_invalid_result") from None
    if (
        checked.run_id != request.run_id
        or checked.workflow_id != request.workflow_id
        or checked.action_digest != request.action_digest
    ):
        raise GovernedMCPError("governance_conflict")
    return checked


def _now(clock: Callable[[], datetime] | None) -> datetime:
    value = None
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            try:
                value = clock() if clock is not None else datetime.now(timezone.utc)
            except BaseException:
                pass
    if (
        type(value) is not datetime
        or value.tzinfo is None
        or value.utcoffset() != timedelta(0)
    ):
        raise GovernedMCPError("governance_failed")
    return value


def build_governed_mcp_handlers(
    service: GovernedMCPService | None,
    *,
    consent_policy: ConsentReceiptPolicy,
    receipt_provider: GovernedConsentProvider | None = None,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Callable[..., dict[str, Any]]]:
    """Bind the four tools to custody and server-side human-channel consent.

    Consent receipts never enter or leave the tool schema. The provider is
    trusted local code; it must retrieve a human-issued, single-use receipt
    bound to the exact tool and arguments, without granting implicit approval.
    A failed handoff acknowledgement may have unknown commit state: reconcile
    through status rather than automatically retrying a mutation.
    """

    def dispatch(operation: str, request: Any) -> dict[str, Any]:
        parsed = GovernedMCPRequest.from_dict(request)
        if operation != "request_review":
            return _view(_call(service, operation, parsed), parsed).to_dict()
        if parsed.expected_state_digest is None:
            raise GovernedMCPError("invalid_arguments")
        _now(clock)
        current = _view(_call(service, "status", parsed), parsed)
        if current.state_digest != parsed.expected_state_digest or current.phase in (
            ActionPhase.COMPLETED,
            ActionPhase.ABORTED,
        ):
            raise GovernedMCPError("governance_conflict")
        if current.status in (
            GovernanceStatus.DENIED,
            GovernanceStatus.CONFLICT,
            GovernanceStatus.UNAVAILABLE,
            GovernanceStatus.FAILED,
        ):
            raise GovernedMCPError("governance_denied")
        if (
            type(consent_policy) is not ConsentReceiptPolicy
            or not consent_policy.require_receipt
        ):
            raise GovernedMCPError("governance_denied")
        arguments = {"request": parsed.to_dict()}
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sink), redirect_stderr(sink):
                try:
                    receipt = (
                        receipt_provider(
                            "openmed_workflow_request_review", deepcopy(arguments)
                        )
                        if receipt_provider is not None
                        else None
                    )
                except BaseException:
                    raise GovernedMCPError("governance_failed") from None
                consent_policy.authorize(
                    tool="openmed_workflow_request_review",
                    arguments=arguments,
                    receipt=receipt,
                )
        result = _call(service, "request_review", parsed)
        if (
            type(result) is not GovernedMCPReview
            or type(result.packet) is not ReviewerHandoffPacket
        ):
            raise GovernedMCPError("governance_invalid_result")
        checked = _view(result.view, parsed)
        try:
            now = _now(clock)
            packet = ReviewerHandoffPacket.from_dict(result.packet.to_dict(), now=now)
        except (TypeError, ValueError):
            raise GovernedMCPError("governance_invalid_result") from None
        if (
            checked.status is not GovernanceStatus.REVIEW_REQUIRED
            or checked.committed_effect_count != current.committed_effect_count
            or checked.proposed_effect_count != current.proposed_effect_count
            or checked.receipt_verification is not ReceiptObservation.ABSENT
            or packet.run_id != parsed.run_id
            or packet.workflow_id != parsed.workflow_id
            or packet.issued_at > now
            or not any(
                reference.kind is ArtifactKind.EVIDENCE
                and reference.schema_id == "openmed.agent.action.v1"
                and reference.sha256 == parsed.action_digest.removeprefix("sha256:")
                for reference in packet.evidence_references
            )
        ):
            raise GovernedMCPError("governance_invalid_result")
        digest = (
            "sha256:" + hashlib.sha256(packet.to_json().encode("utf-8")).hexdigest()
        )
        if checked.review_request_digest not in (None, digest):
            raise GovernedMCPError("governance_invalid_result")
        return replace(checked, review_request_digest=digest).to_dict()

    return {
        name: (lambda _operation=operation, **kwargs: dispatch(_operation, **kwargs))
        for name, operation in GOVERNED_MCP_OPERATIONS.items()
    }


_DIGEST_SCHEMA = {"type": "string", "pattern": r"^sha256:[0-9a-f]{64}$"}
_OPTIONAL_DIGEST_SCHEMA = {"anyOf": [_DIGEST_SCHEMA, {"type": "null"}]}
GOVERNED_MCP_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "const": GOVERNED_MCP_REQUEST_VERSION},
        "run_id": {"type": "string", "pattern": r"^run_[0-9a-f]{32}$"},
        "workflow_id": {"type": "string", "maxLength": 256},
        "action_digest": _DIGEST_SCHEMA,
        "expected_state_digest": _OPTIONAL_DIGEST_SCHEMA,
    },
    "required": sorted(_REQUEST_FIELDS),
}
GOVERNED_MCP_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "schema_version": {"type": "string", "const": GOVERNED_MCP_RESULT_VERSION},
        "run_id": GOVERNED_MCP_REQUEST_SCHEMA["properties"]["run_id"],
        "workflow_id": GOVERNED_MCP_REQUEST_SCHEMA["properties"]["workflow_id"],
        "action_digest": _DIGEST_SCHEMA,
        "state_digest": _DIGEST_SCHEMA,
        "phase": {"type": "string", "enum": [v.value for v in ActionPhase]},
        "status": {"type": "string", "enum": [v.value for v in GovernanceStatus]},
        "authority": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                name: {"type": "string", "enum": [v.value for v in GovernanceDecision]}
                for name in ("grant", "purpose_ticket", "projection")
            },
            "required": ["grant", "purpose_ticket", "projection"],
        },
        "proposed_effect_count": {"type": "integer", "minimum": 0, "maximum": 1024},
        "committed_effect_count": {"type": "integer", "minimum": 0, "maximum": 1024},
        "receipt_verification": {
            "type": "string",
            "enum": [v.value for v in ReceiptObservation],
        },
        "review_request_digest": _OPTIONAL_DIGEST_SCHEMA,
    },
    "required": [
        "schema_version",
        "run_id",
        "workflow_id",
        "action_digest",
        "state_digest",
        "phase",
        "status",
        "authority",
        "proposed_effect_count",
        "committed_effect_count",
        "receipt_verification",
        "review_request_digest",
    ],
}
