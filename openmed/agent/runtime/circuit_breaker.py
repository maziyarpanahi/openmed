"""Deterministic, metadata-only circuit breaker for agent escalation failures.

The caller must stop scheduling work when a handoff is returned. This module
does not execute actions, grant permission, or inspect clinical content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Final

from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId
from openmed.agent.reviewer_handoff import (
    RequestedDecision,
    ReviewerHandoffPacket,
)

_DIGEST_RE: Final = re.compile(r"[0-9a-f]{64}")
_MAX_THRESHOLD: Final = 32
_HANDOFF_LIFETIME: Final = timedelta(minutes=30)


class CircuitBreakerError(ValueError):
    """A value-free circuit-breaker validation or state failure.

    Args:
        code: Stable failure code with no submitted content.
        field_name: Optional public field name.
    """

    def __init__(self, code: str, field_name: str | None = None) -> None:
        self.code = code
        self.field_name = field_name
        super().__init__(code if field_name is None else f"{field_name}: {code}")


class FailureClass(str, Enum):
    """Closed failure classes that can trip an escalation circuit."""

    UNCERTAINTY = "uncertainty"
    PERMISSION_DENIAL = "permission_denial"
    TOOL_FAILURE = "tool_failure"


class RecoveryAction(str, Enum):
    """Human-directed recovery choices; none grants execution authority."""

    REVIEW_EVIDENCE = "review_evidence"
    RECHECK_PERMISSIONS = "recheck_permissions"
    REVIEW_TOOL_FAILURE = "review_tool_failure"
    ABORT_RUN = "abort_run"


_HANDOFF_METADATA: Final = {
    FailureClass.UNCERTAINTY: (
        "low_confidence",
        RequestedDecision.REVIEW_EVIDENCE,
        (RecoveryAction.REVIEW_EVIDENCE, RecoveryAction.ABORT_RUN),
    ),
    FailureClass.PERMISSION_DENIAL: (
        "human_gate",
        RequestedDecision.DECIDE_NEXT_STEP,
        (RecoveryAction.RECHECK_PERMISSIONS, RecoveryAction.ABORT_RUN),
    ),
    FailureClass.TOOL_FAILURE: (
        "safety_review",
        RequestedDecision.ASSESS_SAFETY,
        (RecoveryAction.REVIEW_TOOL_FAILURE, RecoveryAction.ABORT_RUN),
    ),
}


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    """Maximum failures per class and across one run, including the trip attempt.

    Args:
        uncertainty: Maximum uncertainty failures.
        permission_denial: Maximum permission denials.
        tool_failure: Maximum tool failures.
        total: Maximum failures of all classes combined.
    """

    uncertainty: int = 3
    permission_denial: int = 2
    tool_failure: int = 3
    total: int = 6

    def __post_init__(self) -> None:
        for field_name in ("uncertainty", "permission_denial", "tool_failure", "total"):
            value = getattr(self, field_name)
            if type(value) is not int or not 1 <= value <= _MAX_THRESHOLD:
                raise CircuitBreakerError("invalid_threshold", field_name)


@dataclass(frozen=True, slots=True)
class CircuitBreakerHandoff:
    """A reviewer packet and bounded, content-free failed-step history.

    Args:
        packet: Standard reviewer request; it never authorizes action.
        failure_class: Class of the attempt that tripped the circuit.
        attempted_step_digests: Ordered SHA-256 digests of attempted step
            metadata, supplied by the caller without raw step content.
        allowed_recovery_actions: Closed human-directed recovery choices.
    """

    packet: ReviewerHandoffPacket
    failure_class: FailureClass
    attempted_step_digests: tuple[str, ...]
    allowed_recovery_actions: tuple[RecoveryAction, ...]

    def __post_init__(self) -> None:
        if type(self.packet) is not ReviewerHandoffPacket:
            raise CircuitBreakerError("invalid_packet", "packet")
        if type(self.failure_class) is not FailureClass:
            raise CircuitBreakerError("unknown_failure_class", "failure_class")
        if (
            type(self.attempted_step_digests) is not tuple
            or not 1 <= len(self.attempted_step_digests) <= _MAX_THRESHOLD
            or any(
                type(digest) is not str or _DIGEST_RE.fullmatch(digest) is None
                for digest in self.attempted_step_digests
            )
        ):
            raise CircuitBreakerError("invalid_digests", "attempted_step_digests")
        reason_code, decision, actions = _HANDOFF_METADATA[self.failure_class]
        if (
            self.packet.reason_code != reason_code
            or self.packet.requested_decision is not decision
            or self.packet.evidence_references
        ):
            raise CircuitBreakerError("invalid_packet", "packet")
        if (
            type(self.allowed_recovery_actions) is not tuple
            or any(
                type(action) is not RecoveryAction
                for action in self.allowed_recovery_actions
            )
            or self.allowed_recovery_actions != actions
        ):
            raise CircuitBreakerError(
                "invalid_recovery_actions", "allowed_recovery_actions"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a deterministic, metadata-only handoff representation."""

        return {
            "reviewer_packet": self.packet.to_dict(),
            "failure_class": self.failure_class.value,
            "attempted_step_digests": list(self.attempted_step_digests),
            "allowed_recovery_actions": [
                action.value for action in self.allowed_recovery_actions
            ],
        }

    def to_json(self) -> str:
        """Serialize the handoff deterministically without free-text fields."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class EscalationCircuitBreaker:
    """Track one run's failed escalations and permanently halt at a limit.

    Args:
        run_id: Opaque identifier of the monitored run.
        workflow_id: Canonical workflow identifier.
        policy: Bounded class and total thresholds.
    """

    def __init__(
        self,
        run_id: RunId,
        workflow_id: WorkflowId,
        policy: CircuitBreakerPolicy | None = None,
    ) -> None:
        if type(run_id) is not RunId:
            raise CircuitBreakerError("wrong_identifier_kind", "run_id")
        if type(workflow_id) is not WorkflowId:
            raise CircuitBreakerError("wrong_identifier_kind", "workflow_id")
        if policy is not None and type(policy) is not CircuitBreakerPolicy:
            raise CircuitBreakerError("invalid_policy", "policy")
        self._run_id = run_id
        self._workflow_id = workflow_id
        self._policy = policy if policy is not None else CircuitBreakerPolicy()
        self._counts = {failure_class: 0 for failure_class in FailureClass}
        self._digests: list[str] = []
        self._handoff: CircuitBreakerHandoff | None = None

    @property
    def halted(self) -> bool:
        """Return whether this run has reached a failure threshold."""

        return self._handoff is not None

    @property
    def handoff(self) -> CircuitBreakerHandoff | None:
        """Return the immutable handoff once halted, or ``None`` before then."""

        return self._handoff

    def record_failure(
        self,
        failure_class: FailureClass,
        step_digest: str,
        *,
        now: datetime,
    ) -> CircuitBreakerHandoff | None:
        """Record one failed attempt and return a handoff at the first limit.

        ``now`` is explicit to keep the decision deterministic. The digest must
        be SHA-256 of non-sensitive step metadata; raw content is never accepted.
        Further records after a trip fail closed.
        """

        if self.halted:
            raise CircuitBreakerError("circuit_open")
        if type(failure_class) is not FailureClass:
            raise CircuitBreakerError("unknown_failure_class", "failure_class")
        if type(step_digest) is not str or _DIGEST_RE.fullmatch(step_digest) is None:
            raise CircuitBreakerError("invalid_digest", "step_digest")
        if (
            type(now) is not datetime
            or now.utcoffset() != timedelta(0)
            or now.tzinfo is None
            or now.microsecond != 0
        ):
            raise CircuitBreakerError("invalid_timestamp", "now")

        count = self._counts[failure_class] + 1
        total = len(self._digests) + 1
        limit = getattr(self._policy, failure_class.value)
        if count >= limit or total >= self._policy.total:
            reason_code, decision, actions = _HANDOFF_METADATA[failure_class]
            try:
                expires_at = now + _HANDOFF_LIFETIME
            except OverflowError:
                raise CircuitBreakerError("invalid_timestamp", "now") from None
            packet = ReviewerHandoffPacket(
                run_id=self._run_id,
                workflow_id=self._workflow_id,
                reason_code=reason_code,
                requested_decision=decision,
                evidence_references=(),
                issued_at=now,
                expires_at=expires_at,
                validation_time=now,
            )
            self._handoff = CircuitBreakerHandoff(
                packet=packet,
                failure_class=failure_class,
                attempted_step_digests=(*self._digests, step_digest),
                allowed_recovery_actions=actions,
            )
        self._counts[failure_class] = count
        self._digests.append(step_digest)
        return self._handoff


__all__ = [
    "CircuitBreakerError",
    "CircuitBreakerHandoff",
    "CircuitBreakerPolicy",
    "EscalationCircuitBreaker",
    "FailureClass",
    "RecoveryAction",
]
