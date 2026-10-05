"""Guard authority on the existing workflow step executor boundary.

Providers are trusted local host integrations. Source payloads stay process-local;
only typed outcomes, opaque identifiers and digests cross the diagnostic boundary.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt, ApprovalToken
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.outcomes import OutcomeClass, WorkflowOutcome
from openmed.agent.permissions.access_tickets import (
    AccessTicket,
    AccessTicketRequest,
    AccessTicketVerifier,
)
from openmed.agent.permissions.grants import (
    CapabilityGrantManifest,
    CapabilityGrantRequest,
    CapabilityGrantVerifier,
)
from openmed.agent.tools.data_projection import DataProjectionPlan, plan_data_projection
from openmed.agent.workflows.recovery import (
    CompensationLimit,
    EffectKind,
    EffectObservation,
    EffectRecord,
    ObservationState,
    RecoveryCheckpoint,
    RecoveryDecision,
    RecoveryDisposition,
    RecoveryPhase,
    advance_checkpoint,
    recover_workflow,
)
from openmed.mcp.tool_registry import ToolSpec, validate_tool_input


class DispatchToolProvider(Protocol):
    """Resolve a registered contract and invoke its pinned local implementation."""

    def get(self, name: str) -> ToolSpec:
        """Return the registered tool specification without invoking it."""
        ...

    def invoke(
        self, spec: ToolSpec, arguments: Mapping[str, Any], *, effect: EffectRecord
    ) -> None:
        """Invoke the pinned spec once with its effect key; keep output private."""
        ...


class DispatchApprovalProvider(Protocol):
    """Consume exact, single-use approval (implemented by ApprovalTokenVerifier)."""

    def consume(
        self,
        token: ApprovalToken,
        *,
        action_digest: str,
        reviewer_role: str,
        now: int | None = None,
    ) -> ApprovalReceipt:
        """Return a verified receipt or deny without invoking a tool."""
        ...


class DispatchEffectStore(Protocol):
    """Atomic action reservation and content-free recovery storage boundary."""

    def claim(self, checkpoint: RecoveryCheckpoint) -> bool:
        """Persist intent atomically, once per run/action, including changed digests.

        A failed or interrupted claim must never permit another claim for that
        run/action. Existing claims return False. No submitted payload is stored.
        """
        ...

    def append(self, checkpoint: RecoveryCheckpoint) -> None:
        """Durably append a validated successor before returning."""
        ...

    def observe(self, effect: EffectRecord) -> EffectObservation:
        """Query actual sink state; return AMBIGUOUS when commit is uncertain."""
        ...


@dataclass(frozen=True, slots=True, repr=False)
class DispatchAuthority:
    """Host-supplied claims, kept outside declarative workflow inputs."""

    grant: CapabilityGrantManifest | None
    grant_request: CapabilityGrantRequest
    ticket: AccessTicket | None
    ticket_request: AccessTicketRequest
    projection: DataProjectionPlan
    approval: ApprovalToken | None = None


@dataclass(frozen=True, slots=True, repr=False)
class DispatchBinding:
    """Trusted host policy for one registered tool and one opaque action.

    ``ticket_request`` binds the run, purpose, selectors and tool action to the
    host's selected record source. Callers must not derive these bindings from
    untrusted workflow declarations. Changing arguments requires fresh approval.
    """

    spec: ToolSpec
    tool_id: ToolId
    workflow_id: WorkflowId
    run_id: RunId
    action_id: ActionId
    grant_request: CapabilityGrantRequest
    ticket_request: AccessTicketRequest
    reviewer_role: str = "role:org.openmed/clinician@1.0.0"
    approval_required: bool = True

    def __post_init__(self) -> None:
        if (
            type(self.spec) is not ToolSpec
            or type(self.tool_id) is not ToolId
            or type(self.workflow_id) is not WorkflowId
            or type(self.run_id) is not RunId
            or type(self.action_id) is not ActionId
            or type(self.grant_request) is not CapabilityGrantRequest
            or type(self.ticket_request) is not AccessTicketRequest
            or type(self.approval_required) is not bool
            or self.ticket_request.run_id != self.run_id
            or self.grant_request.tool != self.tool_id.serialize()
            or self.ticket_request.tool_action.tool != self.grant_request.tool
            or self.ticket_request.tool_action.action != self.grant_request.action
            or self.tool_id.version != self.spec.version
        ):
            raise ValueError("invalid_dispatch_binding")
        ApprovalReceipt(
            action_digest="sha256:" + "0" * 64,
            reviewer_role=self.reviewer_role,
            token_digest="sha256:" + "0" * 64,
            consumed_at=0,
            expires_at=1,
        )
        object.__setattr__(self, "spec", copy.deepcopy(self.spec))


@dataclass(frozen=True, slots=True)
class DispatchResult:
    """Content-free action result using existing outcome/recovery contracts."""

    phase: ActionPhase
    outcome: WorkflowOutcome
    checkpoint: RecoveryCheckpoint | None = field(default=None, repr=False)
    recovery: RecoveryDecision | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return safe evidence, never arguments, outputs or exception messages."""
        return {
            "phase": self.phase.value,
            "outcome": self.outcome.to_dict(),
            "checkpoint": None
            if self.checkpoint is None
            else self.checkpoint.to_dict(),
            "recovery": None if self.recovery is None else self.recovery.to_dict(),
        }


class GuardedDispatchError(ValueError):
    """Non-retryable workflow failure with a content-free typed result."""

    def __init__(self, result: DispatchResult) -> None:
        super().__init__("guarded_dispatch_stopped")
        self.result = result


class GuardedDispatchAdapter:
    """Callable step adapter for ``WorkflowRunner.executors``.

    Args:
        binding: Trusted policy and pinned tool contract.
        authority: Out-of-band claims for this action.
        tools: Registered tool provider; never consulted for invocation on denial.
        grants: Existing capability-grant verifier with local keys.
        approvals: Existing approval verifier or equivalent trusted provider.
        effects: Atomic reservation, journal and sink observation provider.
        clock: Injected epoch-second clock, shared by every authority check.
        cancelled: Cooperative cancellation probe, checked before invocation.
    """

    def __init__(
        self,
        *,
        binding: DispatchBinding,
        authority: DispatchAuthority,
        tools: DispatchToolProvider,
        grants: CapabilityGrantVerifier,
        approvals: DispatchApprovalProvider,
        effects: DispatchEffectStore,
        clock: Callable[[], int],
        cancelled: Callable[[], bool] = lambda: False,
    ) -> None:
        self._binding = copy.deepcopy(binding)
        self._authority = copy.deepcopy(authority)
        self._tools = tools
        self._grants = grants
        self._approvals = approvals
        self._effects = effects
        self._clock = clock
        self._cancelled = cancelled

    def action_digest(self, arguments: Mapping[str, Any]) -> str:
        """Bind exact JSON arguments and trusted policy for approval preview.

        Args:
            arguments: Resolved arguments to be approved and later dispatched.

        Returns:
            SHA-256 digest, with no source values in the returned evidence.
        """
        binding = self._binding
        return _digest(
            {
                "schema": "openmed.agent.guarded_dispatch.v1",
                "tool_contract": binding.spec.document(),
                "tool_id": binding.tool_id.serialize(),
                "workflow_id": binding.workflow_id.serialize(),
                "run_id": binding.run_id.serialize(),
                "action_id": binding.action_id.serialize(),
                "grant_request": binding.grant_request.as_constraint().to_dict(),
                "purpose": binding.ticket_request.purpose,
                "projection": list(binding.ticket_request.projection),
                "selectors": [
                    {"kind": item.kind, "digest": item.digest}
                    for item in binding.ticket_request.record_selectors
                ],
                "approval_required": self._approval_required,
                "reviewer_role": binding.reviewer_role,
                "arguments": arguments,
            }
        )

    @property
    def _approval_required(self) -> bool:
        return self._binding.approval_required or self._binding.spec.state_changing

    def __call__(self, **arguments: Any) -> dict[str, Any]:
        """Execute as a workflow step, stopping the runner on any unsafe outcome."""
        result = self.dispatch(arguments)
        if result.outcome.outcome_class is not OutcomeClass.SUCCESS:
            raise GuardedDispatchError(result)
        return result.to_dict()

    def dispatch(self, arguments: Mapping[str, Any]) -> DispatchResult:
        """Check every claim, reserve the action and invoke at most once.

        Args:
            arguments: Already resolved, process-local JSON tool arguments.

        Returns:
            A typed, content-free result. No uncertain effect is retried here.
        """
        binding = self._binding
        authority = self._authority
        # Sanitize provider failures at this boundary; never retain exceptions.
        try:
            now = self._clock()
            if type(now) is not int or now < 0:
                raise ValueError("invalid_clock")
            if (
                self._tools.get(binding.spec.name) != binding.spec
                or authority.grant_request != binding.grant_request
                or authority.ticket_request != binding.ticket_request
            ):
                raise ValueError("authority_mismatch")
            self._grants.verify(authority.grant, binding.grant_request, now=now)
            ticket = AccessTicketVerifier().verify(
                authority.ticket, binding.ticket_request, now=now
            )
            projection = plan_data_projection(
                binding.spec.input_schema,
                workflow_purpose=binding.ticket_request.purpose,
                granted_data_classes=ticket.permitted_data_classes,
            )
            if (
                authority.projection != projection
                or projection.data_classes != binding.ticket_request.projection
            ):
                raise ValueError("projection_mismatch")
            # Freeze the exact JSON payload before digesting or consulting approval.
            frozen_arguments = json.loads(_json(arguments))
            validate_tool_input(binding.spec, frozen_arguments)
            _check_paths(frozen_arguments, projection.field_paths)
            digest = self.action_digest(frozen_arguments)
            effect = EffectRecord.create(
                ordinal=0,
                run_id=binding.run_id,
                action_id=binding.action_id,
                tool_id=binding.tool_id,
                kind=EffectKind.LOCAL_TOOL,
                operation_digest=digest,
                approval_required=self._approval_required,
                compensation_limit=CompensationLimit.NONE,
            )
            checkpoint = RecoveryCheckpoint.create(
                workflow_id=binding.workflow_id,
                run_id=binding.run_id,
                sequence=0,
                phase=RecoveryPhase.PLANNED,
                plan_digest=digest,
                effects=(effect,),
            )
        except (Exception, asyncio.CancelledError):
            return DispatchResult(
                ActionPhase.ABORTED,
                WorkflowOutcome(OutcomeClass.POLICY_DENIED, "phi_policy"),
            )

        lineage = [checkpoint]
        try:
            if self._cancelled():
                return self._cancel(lineage, now, persist=False)
            # The store reserves run/action, not only the argument-derived key.
            if self._effects.claim(checkpoint) is not True:
                return self._review(lineage, now, persist=False)
        except (Exception, asyncio.CancelledError):
            return self._review(lineage, now, persist=False)
        try:
            if self._approval_required:
                if type(authority.approval) is not ApprovalToken:
                    raise ValueError("approval_required")
                receipt = self._approvals.consume(
                    authority.approval,
                    action_digest=digest,
                    reviewer_role=binding.reviewer_role,
                    now=now,
                )
                if (
                    type(receipt) is not ApprovalReceipt
                    or receipt.action_digest != digest
                    or receipt.reviewer_role != binding.reviewer_role
                    or receipt.expires_at <= now
                ):
                    raise ValueError("invalid_approval_receipt")
                checkpoint = _next(
                    checkpoint,
                    RecoveryPhase.APPROVAL_RECORDED,
                    approval_action_digest=digest,
                    approval_receipt_digest=_digest(receipt.to_dict()),
                    approval_expires_at=receipt.expires_at,
                )
                lineage.append(checkpoint)
                self._effects.append(checkpoint)
        except (Exception, asyncio.CancelledError):
            return self._cancel(lineage, now, denied=True)

        try:
            if self._cancelled():
                return self._cancel(lineage, now)
            # Check expiry again after potentially slow approval/storage callbacks.
            dispatch_time = self._clock()
            if type(dispatch_time) is not int or dispatch_time < now:
                raise ValueError("invalid_clock")
            self._grants.verify(
                authority.grant, binding.grant_request, now=dispatch_time
            )
            AccessTicketVerifier().verify(
                authority.ticket, binding.ticket_request, now=dispatch_time
            )
            if (
                checkpoint.approval_expires_at is not None
                and checkpoint.approval_expires_at <= dispatch_time
            ):
                raise ValueError("approval_expired")
            if self._tools.get(binding.spec.name) != binding.spec:
                raise ValueError("tool_changed")
            checkpoint = _next(checkpoint, RecoveryPhase.DISPATCHING)
            lineage.append(checkpoint)
            self._effects.append(checkpoint)
            if self._cancelled():
                return self._cancel(lineage, dispatch_time)
            # A durable append may itself take long enough to invalidate authority.
            final_time = self._clock()
            if type(final_time) is not int or final_time < dispatch_time:
                raise ValueError("invalid_clock")
            self._grants.verify(authority.grant, binding.grant_request, now=final_time)
            AccessTicketVerifier().verify(
                authority.ticket, binding.ticket_request, now=final_time
            )
            if (
                checkpoint.approval_expires_at is not None
                and checkpoint.approval_expires_at <= final_time
            ):
                raise ValueError("approval_expired")
            if self._tools.get(binding.spec.name) != binding.spec:
                raise ValueError("tool_changed")
            dispatch_time = final_time
        except (Exception, asyncio.CancelledError):
            return self._cancel(lineage, now, denied=True)

        try:
            self._tools.invoke(
                copy.deepcopy(binding.spec), frozen_arguments, effect=effect
            )
        except asyncio.CancelledError:
            return self._cancel(lineage, dispatch_time)
        except (Exception, asyncio.CancelledError):
            return self._review(lineage, dispatch_time)
        try:
            if self._cancelled():
                return self._cancel(lineage, dispatch_time)
            observation = self._effects.observe(effect)
            decision = recover_workflow(lineage, (observation,), now=dispatch_time)
            if decision.disposition is not RecoveryDisposition.COMPLETE:
                return self._review(lineage, dispatch_time)
            completed = advance_checkpoint(checkpoint, decision)
            self._effects.append(completed)
        except (Exception, asyncio.CancelledError):
            return self._review(lineage, dispatch_time)
        return DispatchResult(
            ActionPhase.COMPLETED,
            WorkflowOutcome(OutcomeClass.SUCCESS, "completed"),
            completed,
            decision,
        )

    def _cancel(
        self,
        lineage: list[RecoveryCheckpoint],
        now: int,
        *,
        denied: bool = False,
        persist: bool = True,
    ) -> DispatchResult:
        aborted = _next(lineage[-1], RecoveryPhase.ABORTED)
        try:
            if persist:
                self._effects.append(aborted)
        except (Exception, asyncio.CancelledError):
            pass  # Intent remains reserved even if the diagnostic append fails.
        decision = recover_workflow((*lineage, aborted), (), now=now)
        return DispatchResult(
            ActionPhase.ABORTED,
            WorkflowOutcome(OutcomeClass.POLICY_DENIED, "consent_required")
            if denied
            else WorkflowOutcome(OutcomeClass.REVIEW_REQUIRED, "safety_review"),
            aborted,
            decision,
        )

    def _review(
        self, lineage: list[RecoveryCheckpoint], now: int, *, persist: bool = True
    ) -> DispatchResult:
        checkpoint = lineage[-1]
        effect = checkpoint.effects[0]
        uncertain = EffectObservation(
            effect.action_id,
            effect.operation_digest,
            effect.idempotency_key,
            ObservationState.AMBIGUOUS,
        )
        decision = recover_workflow(lineage, (uncertain,), now=now)
        reconciling = advance_checkpoint(checkpoint, decision)
        try:
            if persist:
                self._effects.append(reconciling)
        except (Exception, asyncio.CancelledError):
            pass
        return DispatchResult(
            ActionPhase.WAITING_REVIEW,
            WorkflowOutcome(OutcomeClass.REVIEW_REQUIRED, "safety_review"),
            reconciling,
            decision,
        )


def _next(
    checkpoint: RecoveryCheckpoint, phase: RecoveryPhase, **changes: Any
) -> RecoveryCheckpoint:
    return RecoveryCheckpoint.create(
        workflow_id=checkpoint.workflow_id,
        run_id=checkpoint.run_id,
        sequence=checkpoint.sequence + 1,
        phase=phase,
        plan_digest=checkpoint.plan_digest,
        effects=checkpoint.effects,
        previous_checkpoint_digest=checkpoint.checkpoint_digest,
        approval_action_digest=changes.get(
            "approval_action_digest", checkpoint.approval_action_digest
        ),
        approval_receipt_digest=changes.get(
            "approval_receipt_digest", checkpoint.approval_receipt_digest
        ),
        approval_expires_at=changes.get(
            "approval_expires_at", checkpoint.approval_expires_at
        ),
    )


def _json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _check_paths(value: Any, paths: tuple[str, ...], path: str = "") -> None:
    # Every actual field must have a planner declaration. An included parent
    # never implicitly authorizes arbitrary nested keys or array-item fields.
    if type(value) is dict:
        for key, child in value.items():
            child_path = f"{path}/{key}"
            if child_path not in paths:
                raise ValueError("unprojected_field")
            _check_paths(child, paths, child_path)
    elif type(value) is list:
        for child in value:
            _check_paths(child, paths, f"{path}/*")
