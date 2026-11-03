"""Local operator CLI for authorized, value-free service recovery actions.

The commands in this module are adapters only. Backup storage, model
activation, migration engines and job replay remain owned by caller-injected
component services. ``inspect``, ``plan`` and ``verify`` call read-only
component methods and never consume approvals. ``apply`` runs one explicit
recovery action only after the exact plan, the active state generation, the
state digest, the recovery point and a signed single-use approval all match.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import IO, Any, NoReturn, Protocol

from openmed.agent.approvals.tokens import (
    ApprovalActionMismatchError,
    ApprovalExpiredError,
    ApprovalNonceStoreError,
    ApprovalNotYetValidError,
    ApprovalReceipt,
    ApprovalReplayError,
    ApprovalToken,
    ApprovalTokenError,
    ApprovalTokenVerifier,
)

from .governed_workflows import (
    WorkflowCLIError,
    _private_json,
    workflow_cli_receipt_digest,
)

SERVICE_RECOVERY_CLI_SCHEMA_VERSION = "openmed.cli.service_recovery.v1"
SERVICE_RECOVERY_PLAN_SCHEMA_VERSION = "openmed.cli.service_recovery_plan.v1"
SERVICE_RECOVERY_APPROVAL_DOMAIN = "openmed.cli.service_recovery.approval.v1"
MAX_RECOVERY_POINTS = 64
MAX_RECOVERY_STEPS = 4096
RECOVERY_HOLD_CODES = frozenset({"writes_active", "component_busy", "operator_hold"})

_MESSAGE = "Service recovery command refused."
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_POINT_REF = re.compile(r"rp_[0-9a-f]{32}\Z")
_MAX_GENERATION = 2**63
_COMMANDS = ("inspect", "plan", "verify", "apply")
_SINGLE_FLAGS = (
    "--action",
    "--recovery-point",
    "--output",
    "--plan",
    "--approval",
    "--json",
)
_INPUT_ERRORS = frozenset(
    {"input_invalid", "input_unprotected", "input_unreadable", "input_limit"}
)
_ERRORS = _INPUT_ERRORS | frozenset(
    {
        "output_exists",
        "output_unwritable",
        "adapter_unavailable",
        "service_failed",
        "service_result_invalid",
        "state_changed",
        "plan_stale",
        "plan_action_mismatch",
        "recovery_point_unknown",
        "recovery_point_incompatible",
        "recovery_held",
        "approval_invalid",
        "approval_stale",
        "approval_expired",
        "approval_denied",
        "approval_replayed",
    }
)
_ERROR_STATUS = MappingProxyType(
    {
        1: "failed",
        2: "invalid",
        3: "denied",
        4: "held",
        6: "conflict",
        7: "unavailable",
        8: "partial",
    }
)


class RecoveryAction(str, Enum):
    """Supported recovery actions, each owned by one injected component."""

    RESTORE_BACKUP = "restore-backup"
    ROLLBACK_MODEL = "rollback-model"
    REPLAY_JOBS = "replay-jobs"


class RecoveryCLIStatus(str, Enum):
    """Closed command outcomes with distinct stable process exit statuses."""

    READY = "ready"
    COMPLETED = "completed"
    FAILED = "failed"
    DENIED = "denied"
    HELD = "held"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    PARTIAL = "partial"


SERVICE_RECOVERY_EXIT_CODES = MappingProxyType(
    {
        RecoveryCLIStatus.READY: 0,
        RecoveryCLIStatus.COMPLETED: 0,
        RecoveryCLIStatus.FAILED: 1,
        RecoveryCLIStatus.DENIED: 3,
        RecoveryCLIStatus.HELD: 4,
        RecoveryCLIStatus.CONFLICT: 6,
        RecoveryCLIStatus.UNAVAILABLE: 7,
        RecoveryCLIStatus.PARTIAL: 8,
    }
)
_OUTCOME_STATUSES = frozenset(
    {
        RecoveryCLIStatus.COMPLETED,
        RecoveryCLIStatus.FAILED,
        RecoveryCLIStatus.HELD,
        RecoveryCLIStatus.CONFLICT,
        RecoveryCLIStatus.UNAVAILABLE,
        RecoveryCLIStatus.PARTIAL,
    }
)


class ServiceRecoveryError(ValueError):
    """Fixed diagnostic carrying a closed code and process status only.

    Args:
        code: Closed reason code; unknown values become ``input_invalid``.
        exit_code: Process status; unknown values become ``2`` (usage).
    """

    def __init__(self, code: str, exit_code: int = 2) -> None:
        self.code = code if type(code) is str and code in _ERRORS else "input_invalid"
        self.exit_code = (
            exit_code if type(exit_code) is int and exit_code in _ERROR_STATUS else 2
        )
        if self.code == "input_invalid":
            self.exit_code = 2
        super().__init__(_MESSAGE)


def _digest(value: Any, code: str = "input_invalid", exit_code: int = 2) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ServiceRecoveryError(code, exit_code)
    return value


def _generation(value: Any, code: str = "input_invalid", exit_code: int = 2) -> int:
    if type(value) is not int or not 0 <= value < _MAX_GENERATION:
        raise ServiceRecoveryError(code, exit_code)
    return value


def _point_ref(value: Any, code: str = "input_invalid", exit_code: int = 2) -> str:
    if type(value) is not str or _POINT_REF.fullmatch(value) is None:
        raise ServiceRecoveryError(code, exit_code)
    return value


def _sha256(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return "sha256:" + hashlib.sha256(canonical.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryStateSnapshot:
    """Value-free view of the active durable service state.

    Args:
        active_generation: Monotonic generation of the active service state.
        state_digest: Digest of the active state owned by the state service.
    """

    active_generation: int
    state_digest: str

    def __post_init__(self) -> None:
        _generation(self.active_generation, "service_result_invalid", 1)
        _digest(self.state_digest, "service_result_invalid", 1)

    def to_dict(self) -> dict[str, Any]:
        """Return the generation and state digest only."""
        return {
            "active_generation": self.active_generation,
            "state_digest": self.state_digest,
        }

    def __repr__(self) -> str:
        """Keep digests out of incidental diagnostics."""
        return "RecoveryStateSnapshot(<digest-only>)"


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryPoint:
    """Opaque recovery point offered by a component service.

    Args:
        reference: Opaque ``rp_`` reference, never a path or storage URL.
        digest: Integrity digest of the recovery point manifest.
        compatible: Whether the component can apply it to the active state.
    """

    reference: str
    digest: str
    compatible: bool

    def __post_init__(self) -> None:
        _point_ref(self.reference, "service_result_invalid", 1)
        _digest(self.digest, "service_result_invalid", 1)
        if type(self.compatible) is not bool:
            raise ServiceRecoveryError("service_result_invalid", 1)

    def to_dict(self) -> dict[str, Any]:
        """Return the opaque reference, digest and compatibility flag."""
        return {
            "reference": self.reference,
            "digest": self.digest,
            "compatible": self.compatible,
        }

    def __repr__(self) -> str:
        """Keep references out of incidental diagnostics."""
        return "RecoveryPoint(<opaque>)"


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryProposal:
    """Read-only component proposal for one recovery point.

    Args:
        recovery_point: The exact recovery point the proposal would apply.
        step_count: Number of bounded component steps, from 1 to 4,096.
        hold_code: Optional closed reason the component cannot proceed now.
    """

    recovery_point: RecoveryPoint
    step_count: int
    hold_code: str | None = None

    def __post_init__(self) -> None:
        if type(self.recovery_point) is not RecoveryPoint:
            raise ServiceRecoveryError("service_result_invalid", 1)
        object.__setattr__(self, "recovery_point", replace(self.recovery_point))
        if (
            type(self.step_count) is not int
            or not 1 <= self.step_count <= MAX_RECOVERY_STEPS
        ):
            raise ServiceRecoveryError("service_result_invalid", 1)
        if self.hold_code is not None and (
            type(self.hold_code) is not str or self.hold_code not in RECOVERY_HOLD_CODES
        ):
            raise ServiceRecoveryError("service_result_invalid", 1)

    def __repr__(self) -> str:
        """Keep references out of incidental diagnostics."""
        return "RecoveryProposal(<opaque>)"


_PLAN_FIELDS = (
    "schema_version",
    "action",
    "active_generation",
    "state_digest",
    "recovery_point_ref",
    "recovery_point_digest",
    "step_count",
)


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryPlan:
    """Exact recovery plan presented to the operator and bound by approval.

    The plan digest commits to every field. The approval digest commits to the
    plan digest, action, active generation and state digest under a dedicated
    domain label, so an approval for one plan or state cannot authorize another.

    Args:
        action: Recovery action the plan applies.
        active_generation: Active state generation observed while planning.
        state_digest: Active state digest observed while planning.
        recovery_point_ref: Opaque reference of the planned recovery point.
        recovery_point_digest: Digest of the planned recovery point.
        step_count: Number of bounded component steps.
        schema_version: Exact plan schema version.
    """

    action: RecoveryAction
    active_generation: int
    state_digest: str
    recovery_point_ref: str
    recovery_point_digest: str
    step_count: int
    schema_version: str = SERVICE_RECOVERY_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != SERVICE_RECOVERY_PLAN_SCHEMA_VERSION
            or type(self.action) is not RecoveryAction
        ):
            raise ServiceRecoveryError("input_invalid")
        _generation(self.active_generation)
        _digest(self.state_digest)
        _point_ref(self.recovery_point_ref)
        _digest(self.recovery_point_digest)
        if (
            type(self.step_count) is not int
            or not 1 <= self.step_count <= MAX_RECOVERY_STEPS
        ):
            raise ServiceRecoveryError("input_invalid")

    def _fields(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "action": self.action.value,
            "active_generation": self.active_generation,
            "state_digest": self.state_digest,
            "recovery_point_ref": self.recovery_point_ref,
            "recovery_point_digest": self.recovery_point_digest,
            "step_count": self.step_count,
        }

    @property
    def plan_digest(self) -> str:
        """Return the digest committing to every plan field."""
        return _sha256(self._fields())

    @property
    def approval_digest(self) -> str:
        """Return the action digest an approval token must be issued for."""
        return _sha256(
            {
                "domain": SERVICE_RECOVERY_APPROVAL_DOMAIN,
                "plan_digest": self.plan_digest,
                "action": self.action.value,
                "active_generation": self.active_generation,
                "state_digest": self.state_digest,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        """Return every plan field plus its plan and approval digests."""
        payload = self._fields()
        payload["plan_digest"] = self.plan_digest
        payload["approval_digest"] = self.approval_digest
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RecoveryPlan:
        """Parse exact plan fields and reject any edited or recomputed digest.

        Args:
            value: Plan document previously produced by :meth:`to_dict`.

        Returns:
            The validated plan.

        Raises:
            ServiceRecoveryError: ``input_invalid`` for unknown, missing or
                edited fields.
        """
        expected = set(_PLAN_FIELDS) | {"plan_digest", "approval_digest"}
        if type(value) is not dict or set(value) != expected:
            raise ServiceRecoveryError("input_invalid")
        try:
            plan = cls(
                action=RecoveryAction(value["action"]),
                active_generation=value["active_generation"],
                state_digest=value["state_digest"],
                recovery_point_ref=value["recovery_point_ref"],
                recovery_point_digest=value["recovery_point_digest"],
                step_count=value["step_count"],
                schema_version=value["schema_version"],
            )
        except Exception:
            raise ServiceRecoveryError("input_invalid") from None
        for name, actual in (
            ("plan_digest", plan.plan_digest),
            ("approval_digest", plan.approval_digest),
        ):
            claimed = value[name]
            if type(claimed) is not str or not hmac.compare_digest(claimed, actual):
                raise ServiceRecoveryError("input_invalid")
        return plan

    def __repr__(self) -> str:
        """Keep references and digests out of incidental diagnostics."""
        return f"RecoveryPlan(action={self.action.value!r}, <digest-bound>)"


def recovery_approval_digest(plan: RecoveryPlan) -> str:
    """Return the digest an operator approval must be issued for.

    Args:
        plan: Exact recovery plan to approve.

    Returns:
        The domain-separated digest of the plan, generation and state digest.
        Pass it as ``action_digest`` to
        :class:`~openmed.agent.approvals.tokens.ApprovalTokenSigner`.

    Raises:
        ServiceRecoveryError: ``input_invalid`` when ``plan`` is not a plan.
    """
    if type(plan) is not RecoveryPlan:
        raise ServiceRecoveryError("input_invalid")
    return replace(plan).approval_digest


@dataclass(frozen=True, slots=True, repr=False)
class RecoveryOutcome:
    """Component-reported result of one explicit recovery application.

    Args:
        plan_digest: Digest of the plan the component applied.
        status: Closed outcome; ``ready`` is not an application outcome.
        active_generation: Active generation after the call.
        state_digest: Active state digest after the call.
        applied_step_count: Number of steps durably applied.
    """

    plan_digest: str
    status: RecoveryCLIStatus
    active_generation: int
    state_digest: str
    applied_step_count: int

    def __post_init__(self) -> None:
        _digest(self.plan_digest, "service_result_invalid", 1)
        _generation(self.active_generation, "service_result_invalid", 1)
        _digest(self.state_digest, "service_result_invalid", 1)
        applied = self.applied_step_count
        if (
            type(self.status) is not RecoveryCLIStatus
            or self.status not in _OUTCOME_STATUSES
            or type(applied) is not int
            or not 0 <= applied <= MAX_RECOVERY_STEPS
            or (
                self.status in (RecoveryCLIStatus.COMPLETED, RecoveryCLIStatus.PARTIAL)
                and applied == 0
            )
            or (
                self.status
                not in (RecoveryCLIStatus.COMPLETED, RecoveryCLIStatus.PARTIAL)
                and applied != 0
            )
        ):
            raise ServiceRecoveryError("service_result_invalid", 1)

    def __repr__(self) -> str:
        """Keep digests out of incidental diagnostics."""
        return f"RecoveryOutcome(status={self.status.value!r}, <digest-only>)"


class RecoveryStateService(Protocol):
    """Caller-injected, read-only view of the active durable service state."""

    def inspect(self) -> RecoveryStateSnapshot:
        """Return the active generation and state digest without writes."""


class RecoveryComponentService(Protocol):
    """Caller-injected component that owns one recovery action.

    ``recovery_points`` and ``propose`` must be read-only. ``apply`` must
    atomically compare the plan's active generation and state digest with the
    live state before changing anything, report ``conflict`` without writes
    when they differ, never retry or compensate on its own, and report
    ``partial`` with the durable step count when it stops mid-plan.
    """

    def recovery_points(
        self, snapshot: RecoveryStateSnapshot
    ) -> Sequence[RecoveryPoint]:
        """List bounded recovery points for the observed state without writes."""

    def propose(
        self, snapshot: RecoveryStateSnapshot, recovery_point_ref: str
    ) -> RecoveryProposal:
        """Describe applying one recovery point to the observed state."""

    def apply(self, plan: RecoveryPlan, receipt: ApprovalReceipt) -> RecoveryOutcome:
        """Apply the exact approved plan under compare-and-set custody."""


@dataclass(frozen=True, slots=True, repr=False)
class ServiceRecoveryServices:
    """Explicitly injected recovery services; nothing is enabled by default.

    Args:
        state: Read-only active state service.
        components: Component service per supported recovery action.
        approval_verifier: Verifier that consumes single-use approval tokens.
            Required for ``apply``; read-only commands never call it.
        reviewer_role: Policy role the approval token must carry.
    """

    state: RecoveryStateService
    components: Mapping[RecoveryAction, RecoveryComponentService]
    approval_verifier: ApprovalTokenVerifier | None = None
    reviewer_role: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.components, Mapping) or any(
            type(action) is not RecoveryAction for action in self.components
        ):
            raise ServiceRecoveryError("input_invalid")
        object.__setattr__(self, "components", MappingProxyType(dict(self.components)))
        if self.approval_verifier is None:
            if self.reviewer_role is not None:
                raise ServiceRecoveryError("input_invalid")
        elif (
            not isinstance(self.approval_verifier, ApprovalTokenVerifier)
            or type(self.reviewer_role) is not str
        ):
            raise ServiceRecoveryError("input_invalid")

    def __repr__(self) -> str:
        """Hide adapters, roles and verifier internals."""
        return "ServiceRecoveryServices(<injected>)"


def _call(target: Any, operation: str, *args: Any) -> Any:
    try:
        with (
            open(os.devnull, "w") as silence,
            redirect_stdout(silence),
            redirect_stderr(silence),
        ):
            method = getattr(target, operation, None)
            if not callable(method):
                raise NotImplementedError()
            return method(*args)
    except NotImplementedError:
        error = ServiceRecoveryError("adapter_unavailable", 7)
    except ServiceRecoveryError as exc:
        error = ServiceRecoveryError(exc.code, exc.exit_code)
    except Exception:
        error = ServiceRecoveryError("service_failed", 1)
    raise error


def _checked(value: Any, kind: type) -> Any:
    try:
        if type(value) is not kind:
            raise ValueError()
        return replace(value)
    except Exception:
        raise ServiceRecoveryError("service_result_invalid", 1) from None


def _snapshot(services: ServiceRecoveryServices) -> RecoveryStateSnapshot:
    return _checked(_call(services.state, "inspect"), RecoveryStateSnapshot)


def _services(
    services: ServiceRecoveryServices | None,
) -> ServiceRecoveryServices:
    if services is None:
        raise ServiceRecoveryError("adapter_unavailable", 7)
    return services


def _component(
    services: ServiceRecoveryServices, action: RecoveryAction
) -> RecoveryComponentService:
    component = services.components.get(action)
    if component is None:
        raise ServiceRecoveryError("adapter_unavailable", 7)
    return component


def _same_state(left: RecoveryStateSnapshot, right: RecoveryStateSnapshot) -> bool:
    return left.active_generation == right.active_generation and hmac.compare_digest(
        left.state_digest, right.state_digest
    )


def _propose(
    component: RecoveryComponentService,
    snapshot: RecoveryStateSnapshot,
    reference: str,
) -> RecoveryProposal:
    proposal = _checked(
        _call(component, "propose", snapshot, reference), RecoveryProposal
    )
    if proposal.recovery_point.reference != reference:
        raise ServiceRecoveryError("service_result_invalid", 1)
    if not proposal.recovery_point.compatible:
        raise ServiceRecoveryError("recovery_point_incompatible", 6)
    return proposal


def _build_plan(
    action: RecoveryAction,
    snapshot: RecoveryStateSnapshot,
    proposal: RecoveryProposal,
) -> RecoveryPlan:
    return RecoveryPlan(
        action=action,
        active_generation=snapshot.active_generation,
        state_digest=snapshot.state_digest,
        recovery_point_ref=proposal.recovery_point.reference,
        recovery_point_digest=proposal.recovery_point.digest,
        step_count=proposal.step_count,
    )


def _plan_view(plan: RecoveryPlan) -> dict[str, Any]:
    return {
        "plan": plan.to_dict(),
        "plan_digest": plan.plan_digest,
        "approval_digest": plan.approval_digest,
    }


def _now(clock: Callable[[], int]) -> int:
    try:
        now = clock()
        if type(now) is not int or not 0 <= now < _MAX_GENERATION:
            raise ValueError()
        return now
    except Exception:
        raise ServiceRecoveryError("service_failed", 1) from None


def _read_protected(path: str) -> dict[str, Any]:
    try:
        return _private_json(path)
    except WorkflowCLIError as exc:
        code = exc.code if exc.code in _INPUT_ERRORS else "input_invalid"
        raise ServiceRecoveryError(code) from None


def _read_plan(path: str) -> RecoveryPlan:
    return RecoveryPlan.from_dict(_read_protected(path))


def _read_token(path: str) -> ApprovalToken:
    document = _read_protected(path)
    try:
        return ApprovalToken.from_dict(document)
    except Exception:
        raise ServiceRecoveryError("approval_invalid") from None


def _write_plan(path: str, plan: RecoveryPlan) -> None:
    payload = (
        json.dumps(plan.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        raise ServiceRecoveryError("output_exists") from None
    except Exception:
        raise ServiceRecoveryError("output_unwritable", 1) from None
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    except Exception:
        os.close(descriptor)
        try:
            os.unlink(path)
        except OSError:
            pass
        raise ServiceRecoveryError("output_unwritable", 1) from None
    os.close(descriptor)


def _preflight(
    plan: RecoveryPlan,
    services: ServiceRecoveryServices | None,
    *,
    token: ApprovalToken | None,
    now: int,
) -> tuple[RecoveryComponentService, ServiceRecoveryServices]:
    services = _services(services)
    component = _component(services, plan.action)
    snapshot = _snapshot(services)
    if snapshot.active_generation != plan.active_generation or not (
        hmac.compare_digest(snapshot.state_digest, plan.state_digest)
    ):
        raise ServiceRecoveryError("state_changed", 6)
    proposal = _propose(component, snapshot, plan.recovery_point_ref)
    current = _build_plan(plan.action, snapshot, proposal)
    if not hmac.compare_digest(current.plan_digest, plan.plan_digest):
        raise ServiceRecoveryError("plan_stale", 6)
    if proposal.hold_code is not None:
        raise ServiceRecoveryError("recovery_held", 4)
    if token is not None:
        if not hmac.compare_digest(token.action_digest, plan.approval_digest):
            raise ServiceRecoveryError("approval_stale", 6)
        if token.expires_at <= now:
            raise ServiceRecoveryError("approval_expired", 4)
    return component, services


def _consume(
    services: ServiceRecoveryServices,
    token: ApprovalToken,
    plan: RecoveryPlan,
    now: int,
) -> ApprovalReceipt:
    verifier = services.approval_verifier
    if verifier is None:
        raise ServiceRecoveryError("adapter_unavailable", 7)
    try:
        receipt = verifier.consume(
            token,
            action_digest=plan.approval_digest,
            reviewer_role=services.reviewer_role,
            now=now,
        )
    except ApprovalReplayError:
        error = ServiceRecoveryError("approval_replayed", 3)
    except (ApprovalExpiredError, ApprovalNotYetValidError):
        error = ServiceRecoveryError("approval_expired", 4)
    except ApprovalActionMismatchError:
        error = ServiceRecoveryError("approval_stale", 6)
    except ApprovalNonceStoreError:
        error = ServiceRecoveryError("service_failed", 1)
    except ApprovalTokenError:
        error = ServiceRecoveryError("approval_denied", 3)
    except Exception:
        error = ServiceRecoveryError("service_failed", 1)
    else:
        return _checked(receipt, ApprovalReceipt)
    raise error


def _check_outcome(outcome: RecoveryOutcome, plan: RecoveryPlan) -> None:
    status = outcome.status
    unchanged = outcome.active_generation == plan.active_generation and (
        hmac.compare_digest(outcome.state_digest, plan.state_digest)
    )
    if not hmac.compare_digest(outcome.plan_digest, plan.plan_digest):
        raise ServiceRecoveryError("service_result_invalid", 1)
    if status is RecoveryCLIStatus.COMPLETED:
        valid = (
            outcome.applied_step_count == plan.step_count
            and outcome.active_generation == plan.active_generation + 1
            and not hmac.compare_digest(outcome.state_digest, plan.state_digest)
        )
    elif status is RecoveryCLIStatus.PARTIAL:
        valid = (
            outcome.applied_step_count < plan.step_count
            and outcome.active_generation > plan.active_generation
        )
    elif status is RecoveryCLIStatus.CONFLICT:
        valid = True
    else:
        valid = unchanged
    if not valid:
        raise ServiceRecoveryError("service_result_invalid", 1)


def _inspect(
    services: ServiceRecoveryServices | None,
) -> tuple[RecoveryCLIStatus, dict[str, Any]]:
    services = _services(services)
    snapshot = _snapshot(services)
    components = []
    for action in RecoveryAction:
        component = services.components.get(action)
        points: list[dict[str, Any]] = []
        available = component is not None
        if component is not None:
            try:
                listed = _call(component, "recovery_points", snapshot)
            except ServiceRecoveryError as exc:
                if exc.code != "adapter_unavailable":
                    raise
                available = False
            else:
                if (
                    not isinstance(listed, (list, tuple))
                    or len(listed) > MAX_RECOVERY_POINTS
                ):
                    raise ServiceRecoveryError("service_result_invalid", 1)
                checked = [_checked(point, RecoveryPoint) for point in listed]
                if len({point.reference for point in checked}) != len(checked):
                    raise ServiceRecoveryError("service_result_invalid", 1)
                points = [point.to_dict() for point in checked]
        components.append(
            {
                "action": action.value,
                "available": available,
                "recovery_point_count": len(points),
                "recovery_points": points,
            }
        )
    data = snapshot.to_dict()
    data["components"] = components
    return RecoveryCLIStatus.READY, data


def _plan(
    args: argparse.Namespace, services: ServiceRecoveryServices | None
) -> tuple[RecoveryCLIStatus, dict[str, Any]]:
    action = RecoveryAction(args.action)
    reference = _point_ref(args.recovery_point)
    services = _services(services)
    component = _component(services, action)
    snapshot = _snapshot(services)
    proposal = _propose(component, snapshot, reference)
    if not _same_state(snapshot, _snapshot(services)):
        raise ServiceRecoveryError("state_changed", 6)
    plan = _build_plan(action, snapshot, proposal)
    data = _plan_view(plan)
    data["hold_code"] = proposal.hold_code
    if proposal.hold_code is not None:
        data["output_written"] = False
        return RecoveryCLIStatus.HELD, data
    if args.output is not None:
        _write_plan(args.output, plan)
    data["output_written"] = args.output is not None
    return RecoveryCLIStatus.READY, data


def _verify(
    args: argparse.Namespace,
    services: ServiceRecoveryServices | None,
    clock: Callable[[], int],
) -> tuple[RecoveryCLIStatus, dict[str, Any]]:
    plan = _read_plan(args.plan)
    token = _read_token(args.approval) if args.approval is not None else None
    now = _now(clock)
    _preflight(plan, services, token=token, now=now)
    data = _plan_view(plan)
    data["approval_bound"] = token is not None
    data["checks"] = [
        "plan_integrity",
        "component_available",
        "state_current",
        "recovery_point_compatible",
        "plan_current",
        "not_held",
    ] + (["approval_binding", "approval_unexpired"] if token is not None else [])
    return RecoveryCLIStatus.READY, data


def _apply(
    args: argparse.Namespace,
    services: ServiceRecoveryServices | None,
    clock: Callable[[], int],
) -> tuple[RecoveryCLIStatus, dict[str, Any]]:
    plan = _read_plan(args.plan)
    token = _read_token(args.approval)
    if plan.action is not RecoveryAction(args.apply_action):
        raise ServiceRecoveryError("plan_action_mismatch", 6)
    if services is None or services.approval_verifier is None:
        raise ServiceRecoveryError("adapter_unavailable", 7)
    component, services = _preflight(plan, services, token=token, now=_now(clock))
    receipt = _consume(services, token, plan, _now(clock))
    current = _snapshot(services)
    if current.active_generation != plan.active_generation or not (
        hmac.compare_digest(current.state_digest, plan.state_digest)
    ):
        raise ServiceRecoveryError("state_changed", 6)
    outcome = _checked(_call(component, "apply", plan, receipt), RecoveryOutcome)
    _check_outcome(outcome, plan)
    data = {
        "action": plan.action.value,
        "plan_digest": plan.plan_digest,
        "previous_generation": plan.active_generation,
        "active_generation": outcome.active_generation,
        "state_digest": outcome.state_digest,
        "step_count": plan.step_count,
        "applied_step_count": outcome.applied_step_count,
        "receipt_digest": workflow_cli_receipt_digest(receipt),
    }
    return outcome.status, data


def _dispatch(
    args: argparse.Namespace,
    services: ServiceRecoveryServices | None,
    clock: Callable[[], int],
) -> tuple[RecoveryCLIStatus, dict[str, Any]]:
    if services is not None and type(services) is not ServiceRecoveryServices:
        raise ServiceRecoveryError("adapter_unavailable", 7)
    operation = args.recovery_operation
    if operation == "inspect":
        return _inspect(services)
    if operation == "plan":
        return _plan(args, services)
    if operation == "verify":
        return _verify(args, services, clock)
    return _apply(args, services, clock)


def _command(argv: Sequence[str]) -> str:
    command = "service-recovery"
    if argv and argv[0] in _COMMANDS:
        command += " " + argv[0]
        if argv[0] == "apply" and len(argv) > 1:
            if argv[1] in {action.value for action in RecoveryAction}:
                command += " " + argv[1]
    return command


def _write(envelope: Mapping[str, Any], stream: IO[str] | None) -> None:
    out = stream if stream is not None else sys.stdout
    out.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")


def _error_envelope(command: str, error: ServiceRecoveryError) -> dict[str, Any]:
    return {
        "schema_version": SERVICE_RECOVERY_CLI_SCHEMA_VERSION,
        "ok": False,
        "command": command,
        "status": _ERROR_STATUS[error.exit_code],
        "error": {"code": error.code, "message": _MESSAGE},
    }


def _run_parsed(
    args: argparse.Namespace,
    command: str,
    services: ServiceRecoveryServices | None,
    clock: Callable[[], int],
    stream: IO[str] | None,
) -> int:
    try:
        status, data = _dispatch(args, services, clock)
        exit_code = SERVICE_RECOVERY_EXIT_CODES[status]
        envelope = {
            "schema_version": SERVICE_RECOVERY_CLI_SCHEMA_VERSION,
            "ok": status in (RecoveryCLIStatus.READY, RecoveryCLIStatus.COMPLETED),
            "command": command,
            "status": status.value,
            "data": data,
        }
    except Exception as exc:
        error = (
            ServiceRecoveryError(exc.code, exc.exit_code)
            if type(exc) is ServiceRecoveryError
            else ServiceRecoveryError("service_failed", 1)
        )
        exit_code = error.exit_code
        envelope = _error_envelope(command, error)
    _write(envelope, stream)
    return exit_code


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, message: str) -> NoReturn:
        raise ServiceRecoveryError("input_invalid")


class _ConsoleParser(_Parser):
    def error(self, message: str) -> NoReturn:
        _console_usage_error()


def _console_usage_error(_message: str = "") -> NoReturn:
    _write(
        _error_envelope("service-recovery", ServiceRecoveryError("input_invalid")),
        None,
    )
    raise SystemExit(2)


def _add_json(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Versioned value-free JSON (also the default).",
    )


def _add_commands(
    root: argparse.ArgumentParser,
    parser_class: type[argparse.ArgumentParser],
    handler: Callable[[argparse.Namespace], int],
) -> None:
    actions = [action.value for action in RecoveryAction]
    commands = root.add_subparsers(dest="recovery_operation", parser_class=parser_class)
    inspect = commands.add_parser(
        "inspect",
        help="Read active state and recovery points without writes.",
        description="Read active state and recovery points without writes.",
    )
    _add_json(inspect)
    plan = commands.add_parser(
        "plan",
        help="Build an exact recovery plan without writes or approval.",
        description="Build an exact recovery plan without writes or approval.",
    )
    plan.add_argument(
        "--action", required=True, choices=actions, help="Recovery action to plan."
    )
    plan.add_argument(
        "--recovery-point",
        required=True,
        help="Opaque rp_ recovery point reference from inspect.",
    )
    plan.add_argument(
        "--output",
        default=None,
        help="New private plan file (0600); an existing file is never replaced.",
    )
    _add_json(plan)
    verify = commands.add_parser(
        "verify",
        help="Check a plan and optional approval against live state, read-only.",
        description=(
            "Check a plan and optional approval against live state, read-only."
        ),
    )
    verify.add_argument(
        "--plan", required=True, help="Protected local plan JSON (0600)."
    )
    verify.add_argument(
        "--approval",
        default=None,
        help="Protected approval token JSON; checked but never consumed.",
    )
    _add_json(verify)
    apply = commands.add_parser(
        "apply",
        help="Explicitly apply one approved recovery plan.",
        description="Explicitly apply one approved recovery plan.",
    )
    apply_actions = apply.add_subparsers(dest="apply_action", parser_class=parser_class)
    for action in RecoveryAction:
        text = f"Apply an approved {action.value} plan under exact state custody."
        command = apply_actions.add_parser(action.value, help=text, description=text)
        command.add_argument(
            "--plan", required=True, help="Protected local plan JSON (0600)."
        )
        command.add_argument(
            "--approval",
            required=True,
            help="Protected single-use approval token JSON (0600).",
        )
        _add_json(command)
        command.set_defaults(handler=handler)
    for command in (inspect, plan, verify):
        command.set_defaults(handler=handler)


def _console_handler(args: argparse.Namespace) -> int:
    operation = getattr(args, "recovery_operation", None)
    command = "service-recovery"
    if operation is not None:
        command += " " + operation
        if operation == "apply":
            command += " " + args.apply_action
    return _run_parsed(args, command, None, lambda: int(time.time()), None)


def add_service_recovery_command(subparsers: Any) -> None:
    """Register the ``openmed service-recovery`` help and completion surface.

    The console entry point has no injected services, so every command reports
    ``adapter_unavailable`` (exit 7). Deployments call
    :func:`run_service_recovery_cli` with their own component services.

    Args:
        subparsers: Top-level argparse subparsers action.
    """
    recovery = subparsers.add_parser(
        "service-recovery",
        help="Inspect, plan, verify and apply authorized service recovery.",
        description="Inspect, plan, verify and apply authorized service recovery.",
    )
    recovery.error = _console_usage_error  # type: ignore[method-assign]
    _add_commands(recovery, _ConsoleParser, _console_handler)


def run_service_recovery_cli(
    argv: Sequence[str],
    *,
    services: ServiceRecoveryServices | None = None,
    clock: Callable[[], int] | None = None,
    stdout: IO[str] | None = None,
) -> int:
    """Run ``openmed service-recovery`` with explicitly injected services.

    Args:
        argv: Arguments after ``service-recovery``.
        services: Trusted local recovery services; omitted means unavailable.
        clock: Trusted Unix-second clock, never a command-line override.
        stdout: Optional caller-owned output stream.

    Returns:
        0 ready or completed, 1 failed, 2 invalid input, 3 approval denied,
        4 held, 6 conflict, 7 unavailable or 8 partial. Output is always one
        value-free JSON document, except successful help output.
    """
    parser = _Parser(
        prog="openmed service-recovery",
        description="Inspect, plan, verify and apply authorized service recovery.",
    )
    _add_commands(parser, _Parser, _console_handler)
    command = "service-recovery"
    try:
        if (
            isinstance(argv, (str, bytes))
            or len(argv) > 16
            or any(type(value) is not str or len(value) > 4096 for value in argv)
        ):
            raise ServiceRecoveryError("input_invalid")
        argv = list(argv)
        command = _command(argv)
        for flag in _SINGLE_FLAGS:
            if sum(v == flag or v.startswith(flag + "=") for v in argv) > 1:
                raise ServiceRecoveryError("input_invalid")
        args = parser.parse_args(argv)
        if getattr(args, "recovery_operation", None) is None or (
            args.recovery_operation == "apply"
            and getattr(args, "apply_action", None) is None
        ):
            parser.print_help(stdout)
            return 0
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    except Exception as exc:
        error = (
            ServiceRecoveryError(exc.code, exc.exit_code)
            if type(exc) is ServiceRecoveryError
            else ServiceRecoveryError("input_invalid")
        )
        _write(_error_envelope(command, error), stdout)
        return error.exit_code
    return _run_parsed(
        args, command, services, clock or (lambda: int(time.time())), stdout
    )


__all__ = [
    "SERVICE_RECOVERY_CLI_SCHEMA_VERSION",
    "SERVICE_RECOVERY_PLAN_SCHEMA_VERSION",
    "SERVICE_RECOVERY_APPROVAL_DOMAIN",
    "SERVICE_RECOVERY_EXIT_CODES",
    "MAX_RECOVERY_POINTS",
    "MAX_RECOVERY_STEPS",
    "RECOVERY_HOLD_CODES",
    "RecoveryAction",
    "RecoveryCLIStatus",
    "ServiceRecoveryError",
    "RecoveryStateSnapshot",
    "RecoveryPoint",
    "RecoveryProposal",
    "RecoveryPlan",
    "RecoveryOutcome",
    "RecoveryStateService",
    "RecoveryComponentService",
    "ServiceRecoveryServices",
    "recovery_approval_digest",
    "add_service_recovery_command",
    "run_service_recovery_cli",
]
