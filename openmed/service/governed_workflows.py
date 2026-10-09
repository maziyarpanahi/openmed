"""Bounded HTTP contracts over trusted, caller-injected governance services.

The adapter has no effect executor. Services own run isolation, purpose and
projection policy, durable receipt custody, atomic state comparisons and
idempotency. Receipt parsing alone never establishes human authority.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.outcomes import WorkflowOutcome
from openmed.agent.workflows.recovery import (
    EffectRecord,
    EffectState,
    derive_idempotency_key,
)

if TYPE_CHECKING:
    from .auth import AuthPrincipal

WORKFLOW_REQUEST_VERSION = "openmed.service.workflow_request.v1"
WORKFLOW_RESPONSE_VERSION = "openmed.service.workflow_response.v1"
WORKFLOW_PREVIEW_VERSION = "openmed.service.workflow_preview.v1"
MAX_WORKFLOW_REQUEST_BYTES = 65_536
MAX_WORKFLOW_RESPONSE_BYTES = 262_144
MAX_WORKFLOW_EFFECTS = 128
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_REQUEST_ID = re.compile(r"req_[0-9a-f]{32}\Z")
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "run_id",
        "workflow_id",
        "action_digest",
        "expected_state_digest",
        "request_id",
    }
)
WORKFLOW_ERROR_STATUSES = MappingProxyType(
    {
        "workflow_authentication_required": 401,
        "workflow_forbidden": 403,
        "workflow_disabled": 503,
        "workflow_invalid_input": 422,
        "workflow_request_too_large": 413,
        "workflow_conflict": 409,
        "workflow_receipt_unverified": 403,
        "workflow_receipt_expired": 409,
        "workflow_receipt_future": 409,
        "workflow_terminal": 409,
        "workflow_unavailable": 503,
        "workflow_service_failed": 503,
        "workflow_invalid_result": 502,
        "workflow_mutation_unknown": 503,
    }
)


class WorkflowServiceError(ValueError):
    """Closed HTTP diagnostic that never retains submitted values.

    Args:
        code: One of the declared workflow error codes.
    """

    def __init__(self, code: str) -> None:
        self.code = (
            code
            if type(code) is str and code in WORKFLOW_ERROR_STATUSES
            else "workflow_service_failed"
        )
        self.status_code = WORKFLOW_ERROR_STATUSES[self.code]
        super().__init__("Governed workflow request refused.")


def _digest(value: Any) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise WorkflowServiceError("workflow_invalid_input")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class WorkflowReference:
    """Opaque reference to a service-custodied action and expected state.

    Args:
        run_id: Existing random run identifier.
        workflow_id: Canonical developer-authored workflow identifier.
        action_digest: Digest of the exact action held by the service.
        expected_state_digest: Required compare-and-set state for mutations.
        request_id: Random mutation idempotency identifier, never a patient ID.
    """

    run_id: RunId
    workflow_id: WorkflowId
    action_digest: str
    expected_state_digest: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.run_id) is not RunId or type(self.workflow_id) is not WorkflowId:
            raise WorkflowServiceError("workflow_invalid_input")
        RunId.parse(self.run_id.serialize())
        WorkflowId.parse(self.workflow_id.serialize())
        _digest(self.action_digest)
        if self.expected_state_digest is not None:
            _digest(self.expected_state_digest)
        if self.request_id is not None and (
            type(self.request_id) is not str
            or _REQUEST_ID.fullmatch(self.request_id) is None
        ):
            raise WorkflowServiceError("workflow_invalid_input")

    def require_mutation(self) -> None:
        """Require both a state comparison and a mutation idempotency key."""
        if self.expected_state_digest is None or self.request_id is None:
            raise WorkflowServiceError("workflow_invalid_input")

    def to_dict(self) -> dict[str, Any]:
        """Return only the exact transport metadata fields."""
        return {
            "schema_version": WORKFLOW_REQUEST_VERSION,
            "run_id": self.run_id.serialize(),
            "workflow_id": self.workflow_id.serialize(),
            "action_digest": self.action_digest,
            "expected_state_digest": self.expected_state_digest,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> WorkflowReference:
        """Reject unknown fields and versions without echoing rejected inputs."""
        if type(value) is not dict or value.keys() != _REQUEST_FIELDS:
            raise WorkflowServiceError("workflow_invalid_input")
        if value["schema_version"] != WORKFLOW_REQUEST_VERSION:
            raise WorkflowServiceError("workflow_invalid_input")
        try:
            return cls(
                RunId.parse(value["run_id"]),
                WorkflowId.parse(value["workflow_id"]),
                value["action_digest"],
                value["expected_state_digest"],
                value["request_id"],
            )
        except (TypeError, ValueError):
            raise WorkflowServiceError("workflow_invalid_input") from None


@dataclass(frozen=True, slots=True, repr=False)
class WorkflowView:
    """Validated snapshot with existing outcome and effect evidence types.

    Args:
        reference: Exact run, workflow and action being inspected.
        state_digest: Current trusted state digest.
        phase: Existing action lifecycle phase.
        effects: Ordered existing content-free effect records.
        outcome: Existing workflow outcome, when the service has one.
        receipt_digest: Commitment to a receipt held in trusted custody.
        cancellation_requested: Whether durable cancellation intent was recorded.
    """

    reference: WorkflowReference
    state_digest: str
    phase: ActionPhase
    effects: tuple[EffectRecord, ...] = ()
    outcome: WorkflowOutcome | None = None
    receipt_digest: str | None = None
    cancellation_requested: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.reference) is not WorkflowReference
            or type(self.phase) is not ActionPhase
            or type(self.cancellation_requested) is not bool
        ):
            raise WorkflowServiceError("workflow_invalid_result")
        WorkflowReference.from_dict(self.reference.to_dict())
        _digest(self.state_digest)
        if self.receipt_digest is not None:
            _digest(self.receipt_digest)
        if type(self.effects) is not tuple or len(self.effects) > MAX_WORKFLOW_EFFECTS:
            raise WorkflowServiceError("workflow_invalid_result")
        restored = []
        for ordinal, effect in enumerate(self.effects):
            if type(effect) is not EffectRecord or effect.ordinal != ordinal:
                raise WorkflowServiceError("workflow_invalid_result")
            effect = EffectRecord.from_dict(effect.to_dict())
            expected = derive_idempotency_key(
                run_id=self.reference.run_id,
                action_id=effect.action_id,
                tool_id=effect.tool_id,
                kind=effect.kind,
                operation_digest=effect.operation_digest,
            )
            if not hmac.compare_digest(expected, effect.idempotency_key):
                raise WorkflowServiceError("workflow_invalid_result")
            restored.append(effect)
        object.__setattr__(self, "effects", tuple(restored))
        if len({effect.action_id for effect in restored}) != len(restored) or len(
            {effect.idempotency_key for effect in restored}
        ) != len(restored):
            raise WorkflowServiceError("workflow_invalid_result")
        if self.outcome is not None:
            if type(self.outcome) is not WorkflowOutcome:
                raise WorkflowServiceError("workflow_invalid_result")
            object.__setattr__(
                self, "outcome", WorkflowOutcome.from_dict(self.outcome.to_dict())
            )
        if self.phase is ActionPhase.COMPLETED and any(
            e.state is not EffectState.COMMITTED for e in self.effects
        ):
            raise WorkflowServiceError("workflow_invalid_result")

    def to_dict(self) -> dict[str, Any]:
        """Return bounded inspection evidence without clinical action payloads."""
        effects = [e.to_dict() for e in self.effects]
        encoded = json.dumps(
            effects, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        preview_digest = (
            "sha256:"
            + hashlib.sha256(
                (WORKFLOW_PREVIEW_VERSION + "\0" + encoded).encode("ascii")
            ).hexdigest()
        )
        result = {
            "schema_version": WORKFLOW_RESPONSE_VERSION,
            "run_id": self.reference.run_id.serialize(),
            "workflow_id": self.reference.workflow_id.serialize(),
            "action_digest": self.reference.action_digest,
            "state_digest": self.state_digest,
            "phase": self.phase.value,
            "effects": effects,
            "preview_digest": preview_digest,
            "proposed_effect_count": len(effects),
            "committed_effect_count": sum(
                e.state is EffectState.COMMITTED for e in self.effects
            ),
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "receipt_digest": self.receipt_digest,
            "cancellation_requested": self.cancellation_requested,
        }
        if (
            len(
                json.dumps(result, separators=(",", ":"), ensure_ascii=True).encode(
                    "ascii"
                )
            )
            > MAX_WORKFLOW_RESPONSE_BYTES
        ):
            raise WorkflowServiceError("workflow_invalid_result")
        return result


@dataclass(frozen=True, slots=True, repr=False)
class WorkflowReceiptVerification:
    """Trusted, read-only custody verification bound to a run and state.

    Args:
        reference: Exact action and expected state checked by the service.
        receipt_digest: Commitment to the existing receipt checked in custody.
        verified: Whether trusted durable custody verified the receipt.
    """

    reference: WorkflowReference
    receipt_digest: str
    verified: bool

    def __post_init__(self) -> None:
        if (
            type(self.reference) is not WorkflowReference
            or type(self.verified) is not bool
        ):
            raise WorkflowServiceError("workflow_invalid_result")
        self.reference.require_mutation()
        WorkflowReference.from_dict(self.reference.to_dict())
        _digest(self.receipt_digest)


class WorkflowGovernanceService(Protocol):
    """Trusted local HTTP adapter boundary with no effect execution method.

    Reads and receipt verification must never dispatch or compensate an effect.
    Each mutation must atomically enforce caller isolation, exact action/state
    digests and durable idempotency, including receipt verification again at
    commit. Cancellation records intent and must never compensate effects.
    """

    def preflight(
        self, principal: AuthPrincipal, reference: WorkflowReference
    ) -> WorkflowView:
        """Read permission and policy findings without dispatching an action."""

    def preview(
        self, principal: AuthPrincipal, reference: WorkflowReference
    ) -> WorkflowView:
        """Inspect the service-custodied content-free effect plan."""

    def status(
        self, principal: AuthPrincipal, reference: WorkflowReference
    ) -> WorkflowView:
        """Read current lifecycle and effect evidence without execution."""

    def verify_receipt(
        self,
        principal: AuthPrincipal,
        reference: WorkflowReference,
        receipt: ApprovalReceipt,
        *,
        now: int,
    ) -> WorkflowReceiptVerification:
        """Verify existing durable receipt custody; caller role claims are advisory."""

    def submit_receipt(
        self,
        principal: AuthPrincipal,
        reference: WorkflowReference,
        receipt: ApprovalReceipt,
        *,
        now: int,
    ) -> WorkflowView:
        """Atomically record a verified receipt without dispatching an effect."""

    def cancel(
        self, principal: AuthPrincipal, reference: WorkflowReference
    ) -> WorkflowView:
        """Atomically request cancellation without replay or compensation."""


@dataclass(frozen=True, slots=True)
class WorkflowHTTPPolicy:
    """Server-owned opt-ins for inspection and the two non-executing mutations.

    Args:
        enabled: Allow the injected service's inspection routes.
        allow_review_submissions: Allow receipt submission with workflow:review.
        allow_cancellation: Allow cancellation intent with workflow:cancel.
        timeout_seconds: Bound one service call, including custody verification.
    """

    enabled: bool = False
    allow_review_submissions: bool = False
    allow_cancellation: bool = False
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.enabled,
                self.allow_review_submissions,
                self.allow_cancellation,
            )
        ) or (
            type(self.timeout_seconds) not in (int, float)
            or not 0 < self.timeout_seconds <= 30
        ):
            raise WorkflowServiceError("workflow_invalid_input")


def workflow_receipt_digest(receipt: ApprovalReceipt) -> str:
    """Commit to an existing receipt; a hash is not approval authority."""
    try:
        if type(receipt) is not ApprovalReceipt:
            raise ValueError()
        receipt = ApprovalReceipt.from_dict(receipt.to_dict())
        return "sha256:" + hashlib.sha256(receipt.to_json().encode("utf-8")).hexdigest()
    except (TypeError, ValueError):
        raise WorkflowServiceError("workflow_invalid_input") from None


def validate_workflow_view(value: Any, reference: WorkflowReference) -> WorkflowView:
    """Reject mismatched and malformed service results before serialization."""
    try:
        if type(value) is not WorkflowView:
            raise ValueError()
        value = replace(value)
        if (
            value.reference.run_id != reference.run_id
            or value.reference.workflow_id != reference.workflow_id
            or not hmac.compare_digest(
                value.reference.action_digest, reference.action_digest
            )
        ):
            raise WorkflowServiceError("workflow_conflict")
        value.to_dict()
        return value
    except WorkflowServiceError as error:
        if error.code == "workflow_conflict":
            raise
        raise WorkflowServiceError("workflow_invalid_result") from None
    except Exception:
        raise WorkflowServiceError("workflow_invalid_result") from None


def validate_workflow_receipt(
    reference: WorkflowReference, receipt: ApprovalReceipt, *, now: int
) -> str:
    """Check structural binding and time; trusted custody must still verify it."""
    reference.require_mutation()
    digest = workflow_receipt_digest(receipt)
    if type(now) is not int or not 0 <= now < 2**63:
        raise WorkflowServiceError("workflow_service_failed")
    if not hmac.compare_digest(reference.action_digest, receipt.action_digest):
        raise WorkflowServiceError("workflow_conflict")
    if now < receipt.consumed_at:
        raise WorkflowServiceError("workflow_receipt_future")
    if now >= receipt.expires_at:
        raise WorkflowServiceError("workflow_receipt_expired")
    return digest


def parse_workflow_json(raw: bytes) -> dict[str, Any]:
    """Parse bounded JSON with duplicate-key, nonfinite and nesting rejection."""
    if type(raw) is not bytes or len(raw) > MAX_WORKFLOW_REQUEST_BYTES:
        raise WorkflowServiceError("workflow_request_too_large")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise WorkflowServiceError("workflow_invalid_input")
            result[key] = value
        return result

    def constant(_value):
        raise WorkflowServiceError("workflow_invalid_input")

    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
        pending = [(value, 0)]
        count = 0
        while pending:
            item, depth = pending.pop()
            count += 1
            if count > 256 or depth > 8:
                raise ValueError()
            if type(item) is dict:
                pending.extend((v, depth + 1) for v in item.values())
            elif type(item) is list:
                pending.extend((v, depth + 1) for v in item)
        if type(value) is not dict:
            raise ValueError()
        return value
    except Exception:
        raise WorkflowServiceError("workflow_invalid_input") from None
