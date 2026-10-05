"""Synthetic offline registered-tool, approval and effect-store providers."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace

from openmed.agent.approvals.tokens import (
    ApprovalTokenSigner,
    ApprovalTokenVerifier,
    InMemoryApprovalNonceStore,
)
from openmed.agent.correlation import ActionId, RunId
from openmed.agent.identifiers import ToolId, WorkflowId
from openmed.agent.permissions.access_tickets import (
    AccessTicket,
    AccessTicketRequest,
    RecordSelector,
    ToolAction,
)
from openmed.agent.permissions.grants import (
    CapabilityGrantRequest,
    CapabilityGrantSigner,
    CapabilityGrantVerifier,
)
from openmed.agent.tools.data_projection import plan_data_projection
from openmed.agent.workflows.dispatch import (
    DispatchAuthority,
    DispatchBinding,
    GuardedDispatchAdapter,
)
from openmed.agent.workflows.recovery import (
    EffectObservation,
    ObservationState,
    validate_checkpoint_lineage,
)
from openmed.mcp.tool_registry import ToolRegistry, ToolSpec

KEY = b"synthetic-dispatch-key-material!!"
PURPOSE = "purpose:org.example/summary@1.0.0"
DATA = "data:org.example/clinical-text@1.0.0"
ROLE = "role:org.example/clinician@1.0.0"
TOOL = "tool:org.example/summarize@1.0.0"
ACTION = "action:org.example/execute@1.0.0"
PRIVATE = "SYNTHETIC Alice Example 01/02/1970 /private/record API_SECRET"


class MemoryEffects:
    """Atomic test journal plus independently observed synthetic sink state."""

    def __init__(self):
        self.lock = threading.Lock()
        self.lineages = {}
        self.observations = {}
        self.fail_phase = None
        self.fail_claim = False

    def claim(self, checkpoint):
        with self.lock:
            key = (checkpoint.run_id, checkpoint.effects[0].action_id)
            if key in self.lineages:
                return False
            self.lineages[key] = [checkpoint]
            if self.fail_claim:
                raise RuntimeError(PRIVATE)
            return True

    def append(self, checkpoint):
        if checkpoint.phase is self.fail_phase:
            raise RuntimeError(PRIVATE)
        key = (checkpoint.run_id, checkpoint.effects[0].action_id)
        with self.lock:
            candidate = (*self.lineages[key], checkpoint)
            validate_checkpoint_lineage(candidate)
            self.lineages[key].append(checkpoint)

    def observe(self, effect):
        return self.observations.get(
            effect.idempotency_key,
            EffectObservation(
                effect.action_id,
                effect.operation_digest,
                effect.idempotency_key,
                ObservationState.AMBIGUOUS,
            ),
        )


class RegisteredTools:
    """Invoke a real ToolRegistry handler and publish a synthetic sink receipt."""

    def __init__(self, spec, effects):
        self.effects = effects
        self.calls = 0
        self.arguments = []
        self.error = None
        self.uncertain = False
        self.after_invoke = lambda: None
        self.registry = ToolRegistry()
        self.registry.register(spec, handler=self._handler)

    def get(self, name):
        return self.registry.get(name)

    def _handler(self, **arguments):
        self.calls += 1
        self.arguments.append(arguments)
        if self.error:
            raise self.error
        return {"private_output": PRIVATE}

    def invoke(self, spec, arguments, *, effect):
        assert self.registry.get(spec.name, spec.version) == spec
        output = self.registry.handler(spec.name, spec.version)(**arguments)
        if not self.uncertain:
            # Digest binds an actual synthetic commit receipt, never a release claim.
            digest = (
                "sha256:"
                + hashlib.sha256(output["private_output"].encode()).hexdigest()
            )
            self.effects.observations[effect.idempotency_key] = EffectObservation(
                effect.action_id,
                effect.operation_digest,
                effect.idempotency_key,
                ObservationState.COMMITTED,
                digest,
            )
        self.after_invoke()


class DispatchHarness:
    """Complete synthetic authority set, shared clocks and local providers."""

    def __init__(self, *, read_only=False, approval_required=True, schema=None):
        self.now = 10
        self.cancelled = False
        self.arguments = {"text": PRIVATE}
        schema = schema or {
            "type": "object",
            "x-openmed-purpose": PURPOSE,
            "properties": {
                "text": {
                    "type": "string",
                    "x-openmed-purpose": PURPOSE,
                    "x-openmed-minimum-data": "required",
                    "x-openmed-data-class": DATA,
                },
            },
            "required": ["text"],
            "additionalProperties": False,
        }
        self.spec = ToolSpec(
            "synthetic_summary",
            "Synthetic summary",
            schema,
            {"type": "object"},
            read_only_hint=read_only,
        )
        grant_request = CapabilityGrantRequest(
            TOOL,
            "resource:org.example/document@1.0.0",
            ACTION,
            "policy:org.example/minimum@1.0.0",
        )
        ticket_request = AccessTicketRequest(
            RunId("run_" + "a" * 32),
            PURPOSE,
            (DATA,),
            (
                RecordSelector(
                    "selector:org.example/document@1.0.0", "hmac-sha256:" + "1" * 64
                ),
            ),
            ToolAction(TOOL, ACTION),
        )
        self.binding = DispatchBinding(
            self.spec,
            ToolId(TOOL),
            WorkflowId("workflow:org.example/summary@1.0.0"),
            ticket_request.run_id,
            ActionId("act_" + "b" * 32),
            grant_request,
            ticket_request,
            ROLE,
            approval_required,
        )
        self.authority = DispatchAuthority(
            CapabilityGrantSigner(KEY).issue(
                [grant_request.as_constraint()], expires_at=100
            ),
            grant_request,
            AccessTicket(
                ticket_request.run_id,
                PURPOSE,
                (DATA,),
                ticket_request.record_selectors,
                (ticket_request.tool_action,),
                100,
            ),
            ticket_request,
            plan_data_projection(
                schema, workflow_purpose=PURPOSE, granted_data_classes=(DATA,)
            ),
        )
        self.effects = MemoryEffects()
        self.tools = RegisteredTools(self.spec, self.effects)
        self.grants = CapabilityGrantVerifier(KEY)
        self.approvals = ApprovalTokenVerifier(KEY, InMemoryApprovalNonceStore())
        digest = self.adapter().action_digest(self.arguments)
        self.token = ApprovalTokenSigner(KEY).issue(
            action_digest=digest,
            reviewer_role=ROLE,
            expires_at=100,
            nonce_source=lambda n: b"x" * n,
        )
        self.authority = replace(self.authority, approval=self.token)

    def adapter(self, **overrides):
        values = dict(
            binding=self.binding,
            authority=self.authority,
            tools=self.tools,
            grants=self.grants,
            approvals=self.approvals,
            effects=self.effects,
            clock=lambda: self.now,
            cancelled=lambda: self.cancelled,
        )
        values.update(overrides)
        return GuardedDispatchAdapter(**values)
