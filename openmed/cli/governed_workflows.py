"""Bounded, value-free CLI routing over caller-injected governance services."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, NoReturn, Protocol

from openmed.agent.action_phases import ActionPhase
from openmed.agent.approvals.tokens import ApprovalReceipt
from openmed.agent.correlation import RunId
from openmed.agent.identifiers import WorkflowId

WORKFLOW_CLI_SCHEMA_VERSION = "openmed.cli.governed_workflow.v1"
WORKFLOW_CLI_REQUEST_SCHEMA_VERSION = "openmed.cli.workflow_request.v1"
MAX_WORKFLOW_CLI_INPUT_BYTES = 65_536
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_COMMANDS = ("plan", "preview", "inspect", "submit-review", "cancel", "resume")
_ERRORS = frozenset(
    {
        "input_invalid",
        "input_unprotected",
        "input_unreadable",
        "input_limit",
        "adapter_unavailable",
        "service_failed",
        "service_result_invalid",
        "action_conflict",
        "state_conflict",
        "receipt_invalid",
        "receipt_expired",
        "receipt_future",
        "receipt_unverified",
        "receipt_conflict",
        "terminal_state",
    }
)


class WorkflowCLIStatus(str, Enum):
    """Closed outcomes with distinct stable process exit statuses."""

    READY = "ready"
    COMPLETED = "completed"
    DENIED = "denied"
    REVIEW_REQUIRED = "review_required"
    CANCELLED = "cancelled"
    CONFLICT = "conflict"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


WORKFLOW_CLI_EXIT_CODES = MappingProxyType(
    {
        WorkflowCLIStatus.READY: 0,
        WorkflowCLIStatus.COMPLETED: 0,
        WorkflowCLIStatus.DENIED: 3,
        WorkflowCLIStatus.REVIEW_REQUIRED: 4,
        WorkflowCLIStatus.CANCELLED: 5,
        WorkflowCLIStatus.CONFLICT: 6,
        WorkflowCLIStatus.UNAVAILABLE: 7,
        WorkflowCLIStatus.FAILED: 1,
    }
)


class WorkflowCLIError(ValueError):
    """Fixed diagnostic carrying a closed code and process status only."""

    def __init__(self, code: str, exit_code: int = 2):
        self.code = code if type(code) is str and code in _ERRORS else "input_invalid"
        self.exit_code = (
            exit_code if type(exit_code) is int and exit_code in range(1, 8) else 2
        )
        super().__init__("Governed workflow command refused.")


def _digest(value: Any) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise WorkflowCLIError("input_invalid")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class WorkflowCLIRequest:
    """Metadata-only request; clinical inputs remain in the caller's local service.

    Args:
        run_id: Existing opaque run identifier.
        workflow_id: Developer-authored workflow name, never a patient identifier.
        action_digest: Digest of the exact proposed action or preview.
        expected_state_digest: Current service state digest; required for mutations.
        schema_version: Exact public request schema version.
    """

    run_id: RunId
    workflow_id: WorkflowId
    action_digest: str
    expected_state_digest: str | None = None
    schema_version: str = WORKFLOW_CLI_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            self.schema_version != WORKFLOW_CLI_REQUEST_SCHEMA_VERSION
            or type(self.run_id) is not RunId
            or type(self.workflow_id) is not WorkflowId
        ):
            raise WorkflowCLIError("input_invalid")
        object.__setattr__(self, "run_id", RunId.parse(self.run_id.value))
        object.__setattr__(
            self, "workflow_id", WorkflowId.parse(self.workflow_id.value)
        )
        _digest(self.action_digest)
        if self.expected_state_digest is not None:
            _digest(self.expected_state_digest)

    def to_dict(self) -> dict[str, Any]:
        """Return the exact request metadata without paths or bearer material."""
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id.value,
            "workflow_id": self.workflow_id.value,
            "action_digest": self.action_digest,
            "expected_state_digest": self.expected_state_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkflowCLIRequest:
        """Parse exact request fields without normalizing names or unknown inputs."""
        if type(value) is not dict or set(value) != {
            "schema_version",
            "run_id",
            "workflow_id",
            "action_digest",
            "expected_state_digest",
        }:
            raise WorkflowCLIError("input_invalid")
        try:
            return cls(
                RunId.parse(value["run_id"]),
                WorkflowId.parse(value["workflow_id"]),
                value["action_digest"],
                value["expected_state_digest"],
                value["schema_version"],
            )
        except Exception:
            pass
        raise WorkflowCLIError("input_invalid")


@dataclass(frozen=True, slots=True, repr=False)
class WorkflowCLIView:
    """Strict value-free service snapshot or acknowledged command result.

    Service adapters own state digests and enforce compare-and-set mutations.
    A successful parse does not prove authority or clinical correctness.
    """

    run_id: RunId
    workflow_id: WorkflowId
    action_digest: str
    state_digest: str
    phase: ActionPhase
    status: WorkflowCLIStatus
    proposed_effect_count: int = 0
    committed_effect_count: int = 0
    receipt_digest: str | None = None
    review_request_digest: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.run_id) is not RunId
            or type(self.workflow_id) is not WorkflowId
            or type(self.phase) is not ActionPhase
            or type(self.status) is not WorkflowCLIStatus
        ):
            raise WorkflowCLIError("service_result_invalid", 1)
        object.__setattr__(self, "run_id", RunId.parse(self.run_id.value))
        object.__setattr__(
            self, "workflow_id", WorkflowId.parse(self.workflow_id.value)
        )
        for value in (self.action_digest, self.state_digest):
            _digest(value)
        for optional_digest in (self.receipt_digest, self.review_request_digest):
            if optional_digest is not None:
                _digest(optional_digest)
        for count in (self.proposed_effect_count, self.committed_effect_count):
            if type(count) is not int or not 0 <= count <= 1024:
                raise WorkflowCLIError("service_result_invalid", 1)
        if (
            self.committed_effect_count > self.proposed_effect_count
            or (
                self.status is WorkflowCLIStatus.CANCELLED
                and self.phase is not ActionPhase.ABORTED
            )
            or (
                self.status is WorkflowCLIStatus.COMPLETED
                and self.phase is not ActionPhase.COMPLETED
            )
        ):
            raise WorkflowCLIError("service_result_invalid", 1)

    def to_dict(self) -> dict[str, Any]:
        """Return only controlled metadata, counts and digests."""
        return {
            "run_id": self.run_id.value,
            "workflow_id": self.workflow_id.value,
            "action_digest": self.action_digest,
            "state_digest": self.state_digest,
            "phase": self.phase.value,
            "status": self.status.value,
            "proposed_effect_count": self.proposed_effect_count,
            "committed_effect_count": self.committed_effect_count,
            "receipt_digest": self.receipt_digest,
            "review_request_digest": self.review_request_digest,
        }


@dataclass(frozen=True, slots=True)
class WorkflowCLIReceiptVerification:
    """Trusted-service verification bound to the action, receipt and current state.

    Only the injected service may establish verification from durable custody.
    This record is not a bearer credential and never comes from a CLI input file.
    """

    action_digest: str
    receipt_digest: str
    state_digest: str
    verified: bool

    def __post_init__(self) -> None:
        for value in (self.action_digest, self.receipt_digest, self.state_digest):
            _digest(value)
        if type(self.verified) is not bool:
            raise WorkflowCLIError("service_result_invalid", 1)


class WorkflowCLIGovernanceService(Protocol):
    """Caller-injected governance operations; no adapter is enabled by default.

    Preview/inspect and receipt verification must be read-only. Mutations must
    atomically enforce the request's exact action and expected-state digests,
    and resume must verify durable receipt custody and expiry again at dispatch.
    Adapters own clinical effect execution, reconciliation and idempotency.
    """

    def preview(self, request: WorkflowCLIRequest) -> WorkflowCLIView:
        """Inspect the bounded proposed plan without writes or approval."""

    def inspect(self, request: WorkflowCLIRequest) -> WorkflowCLIView:
        """Read the current run snapshot without changing it."""

    def submit_review(self, request: WorkflowCLIRequest) -> WorkflowCLIView:
        """Create a review request, never an approval decision."""

    def cancel(self, request: WorkflowCLIRequest) -> WorkflowCLIView:
        """Request cancellation under exact current-state custody."""

    def verify_receipt(
        self, request: WorkflowCLIRequest, receipt: ApprovalReceipt, *, now: int
    ) -> WorkflowCLIReceiptVerification:
        """Verify an existing receipt in trusted durable custody without consuming a token."""

    def resume(
        self, request: WorkflowCLIRequest, receipt: ApprovalReceipt, *, now: int
    ) -> WorkflowCLIView:
        """Explicitly resume under fresh custody checks; never auto-retry or compensate."""


def _private_json(path: Any) -> dict[str, Any]:
    descriptor = None
    try:
        if not hasattr(os, "O_NOFOLLOW"):
            raise WorkflowCLIError("input_unprotected")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or (
            os.name == "posix"
            and (before.st_uid != os.geteuid() or stat.S_IMODE(before.st_mode) & 0o077)
        ):
            raise WorkflowCLIError("input_unprotected")
        if before.st_size > MAX_WORKFLOW_CLI_INPUT_BYTES:
            raise WorkflowCLIError("input_limit")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            payload = stream.read(MAX_WORKFLOW_CLI_INPUT_BYTES + 1)
            after = os.fstat(stream.fileno())
        if len(payload) > MAX_WORKFLOW_CLI_INPUT_BYTES:
            raise WorkflowCLIError("input_limit")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ) or len(payload) != after.st_size:
            raise WorkflowCLIError("input_unreadable")

        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise WorkflowCLIError("input_invalid")
                result[key] = value
            return result

        def constant(_value):
            raise WorkflowCLIError("input_invalid")

        document = json.loads(
            payload.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant
        )
        pending = [(document, 0)]
        count = 0
        while pending:
            value, depth = pending.pop()
            count += 1
            if count > 256 or depth > 8:
                raise WorkflowCLIError("input_limit")
            if type(value) is dict:
                pending.extend((v, depth + 1) for v in value.values())
            elif type(value) is list:
                pending.extend((v, depth + 1) for v in value)
        if type(document) is not dict:
            raise WorkflowCLIError("input_invalid")
        return document
    except WorkflowCLIError:
        raise
    except Exception:
        raise WorkflowCLIError("input_unreadable") from None
    finally:
        if descriptor is not None:
            os.close(descriptor)


def workflow_cli_receipt_digest(receipt: ApprovalReceipt) -> str:
    """Commit to the exact existing metadata-only receipt; do not create approval."""
    try:
        if type(receipt) is not ApprovalReceipt:
            raise WorkflowCLIError("receipt_invalid")
        restored = ApprovalReceipt.from_dict(receipt.to_dict())
        return (
            "sha256:" + hashlib.sha256(restored.to_json().encode("utf-8")).hexdigest()
        )
    except Exception:
        pass
    raise WorkflowCLIError("receipt_invalid")


def _call(service: WorkflowCLIGovernanceService, operation: str, *args, **kwargs):
    try:
        with (
            open(os.devnull, "w") as silence,
            redirect_stdout(silence),
            redirect_stderr(silence),
        ):
            method = getattr(service, operation, None)
            if not callable(method):
                raise NotImplementedError()
            return method(*args, **kwargs)
    except NotImplementedError:
        error = WorkflowCLIError("adapter_unavailable", 7)
    except BaseException as exc:
        error = (
            WorkflowCLIError(exc.code, exc.exit_code)
            if type(exc) is WorkflowCLIError
            else WorkflowCLIError("service_failed", 1)
        )
    raise error


def _view(value: Any, request: WorkflowCLIRequest) -> WorkflowCLIView:
    try:
        if type(value) is not WorkflowCLIView:
            raise WorkflowCLIError("service_result_invalid", 1)
        value = replace(value)
    except Exception:
        raise WorkflowCLIError("service_result_invalid", 1) from None
    if (
        value.run_id != request.run_id
        or value.workflow_id != request.workflow_id
        or not hmac.compare_digest(value.action_digest, request.action_digest)
    ):
        raise WorkflowCLIError("action_conflict", 6)
    return value


def _now(clock: Callable[[], int]) -> int:
    try:
        now = clock()
        if type(now) is not int or not 0 <= now < 2**63:
            raise ValueError()
        return now
    except Exception:
        raise WorkflowCLIError("service_failed", 1) from None


def _execute(
    args: argparse.Namespace,
    service: WorkflowCLIGovernanceService | None,
    clock: Callable[[], int],
) -> WorkflowCLIView:
    request = WorkflowCLIRequest.from_dict(_private_json(args.request))
    receipt = None
    if args.operation == "resume":
        try:
            receipt = ApprovalReceipt.from_dict(_private_json(args.receipt))
        except WorkflowCLIError:
            raise
        except Exception:
            raise WorkflowCLIError("receipt_invalid") from None
        if not hmac.compare_digest(receipt.action_digest, request.action_digest):
            raise WorkflowCLIError("receipt_conflict", 6)
        _now(clock)
    if service is None:
        raise WorkflowCLIError("adapter_unavailable", 7)
    if args.operation in ("plan", "preview", "inspect"):
        return _view(
            _call(
                service,
                "inspect" if args.operation == "inspect" else "preview",
                request,
            ),
            request,
        )
    if request.expected_state_digest is None:
        raise WorkflowCLIError("state_conflict", 6)
    current = _view(_call(service, "inspect", request), request)
    if not hmac.compare_digest(current.state_digest, request.expected_state_digest):
        raise WorkflowCLIError("state_conflict", 6)
    if current.phase in (ActionPhase.COMPLETED, ActionPhase.ABORTED):
        if args.operation == "cancel" and current.status is WorkflowCLIStatus.CANCELLED:
            return current
        raise WorkflowCLIError("terminal_state", 6)
    if current.status in (
        WorkflowCLIStatus.DENIED,
        WorkflowCLIStatus.CONFLICT,
        WorkflowCLIStatus.UNAVAILABLE,
        WorkflowCLIStatus.FAILED,
    ):
        return current
    if receipt is not None:
        now = _now(clock)
        verified = _call(service, "verify_receipt", request, receipt, now=now)
        try:
            if type(verified) is not WorkflowCLIReceiptVerification:
                raise ValueError()
            verified = replace(verified)
        except Exception:
            raise WorkflowCLIError("service_result_invalid", 1) from None
        if not verified.verified:
            raise WorkflowCLIError("receipt_unverified", 3)
        if (
            verified.action_digest != request.action_digest
            or verified.receipt_digest != workflow_cli_receipt_digest(receipt)
            or verified.state_digest != current.state_digest
        ):
            raise WorkflowCLIError("receipt_conflict", 6)
        now = _now(clock)
        result = _view(_call(service, "resume", request, receipt, now=now), request)
    else:
        result = _view(
            _call(
                service,
                "submit_review" if args.operation == "submit-review" else "cancel",
                request,
            ),
            request,
        )
    if (
        result.proposed_effect_count != current.proposed_effect_count
        or result.committed_effect_count < current.committed_effect_count
    ):
        raise WorkflowCLIError("service_result_invalid", 1)
    if result.status not in (
        WorkflowCLIStatus.DENIED,
        WorkflowCLIStatus.CONFLICT,
        WorkflowCLIStatus.UNAVAILABLE,
        WorkflowCLIStatus.FAILED,
    ):
        if (
            args.operation == "cancel"
            and result.status is not WorkflowCLIStatus.CANCELLED
        ):
            raise WorkflowCLIError("service_result_invalid", 1)
        if args.operation == "submit-review" and (
            result.status is not WorkflowCLIStatus.REVIEW_REQUIRED
            or result.phase is not ActionPhase.WAITING_REVIEW
            or result.review_request_digest is None
        ):
            raise WorkflowCLIError("service_result_invalid", 1)
        if receipt is not None and result.receipt_digest != workflow_cli_receipt_digest(
            receipt
        ):
            raise WorkflowCLIError("service_result_invalid", 1)
    return result


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        kwargs["allow_abbrev"] = False
        super().__init__(*args, **kwargs)

    def error(self, _message: str) -> NoReturn:
        raise WorkflowCLIError("input_invalid")


def _add_commands(agents: argparse.ArgumentParser) -> None:
    from .agent_admission import _add_admission_subcommands

    groups = agents.add_subparsers(dest="agent_group", parser_class=_Parser)
    _add_admission_subcommands(groups)
    workflow = groups.add_parser(
        "workflow", help="Governed workflow previews and explicit recovery."
    )
    commands = workflow.add_subparsers(dest="operation", parser_class=_Parser)
    help_text = {
        "plan": "Read a bounded plan without executing writes.",
        "preview": "Read the proposed effect counts and custody digests.",
        "inspect": "Read the current workflow snapshot.",
        "submit-review": "Request human review without approving the action.",
        "cancel": "Explicitly request cancellation of the bound run.",
        "resume": "Explicitly resume with a verified existing local receipt.",
    }
    for name in _COMMANDS:
        command = commands.add_parser(
            name, help=help_text[name], description=help_text[name]
        )
        command.add_argument(
            "--request",
            required=True,
            help="Protected local request JSON; no inline payloads or secrets.",
        )
        command.add_argument(
            "--json",
            action="store_true",
            help="Versioned value-free JSON (also the default).",
        )
        if name == "resume":
            command.add_argument(
                "--receipt",
                required=True,
                help="Protected existing approval receipt JSON, never a bearer token.",
            )
        command.set_defaults(
            command_path=f"agents workflow {name}", handler=_unconfigured_handler
        )


def _unconfigured_handler(_args: argparse.Namespace) -> int:
    raise WorkflowCLIError("adapter_unavailable", 7)


def add_governed_workflow_commands(subparsers: Any) -> None:
    """Add the complete governed help/completion surface to the console parser."""
    agents = subparsers.add_parser(
        "agents", help="Local agent governance and recovery."
    )
    _add_commands(agents)


def run_governed_workflow_cli(
    argv: Sequence[str],
    *,
    service: WorkflowCLIGovernanceService | None = None,
    clock: Callable[[], int] | None = None,
) -> int:
    """Run ``openmed agents`` routing with fixed errors and explicit service injection.

    Args:
        argv: Arguments after ``agents``, starting with ``workflow``.
        service: Trusted local governance adapter; omitted means unavailable.
        clock: Trusted Unix-second clock, never a command-line override.

    Returns:
        0 success, 1 service failure, 2 invalid input, 3 denied, 4 review required,
        5 cancelled, 6 conflict or 7 adapter unavailable. Output is always one
        value-free JSON document, except successful help output.
    """
    parser = _Parser(
        prog="openmed agents",
        allow_abbrev=False,
        description="Local agent governance and explicit recovery.",
    )
    _add_commands(parser)
    command = "agents workflow"
    try:
        if (
            isinstance(argv, (str, bytes))
            or len(argv) > 16
            or any(type(v) is not str or len(v) > 4096 for v in argv)
        ):
            raise WorkflowCLIError("input_invalid")
        if len(argv) >= 2 and argv[0] == "workflow" and argv[1] in _COMMANDS:
            command += " " + argv[1]
        for flag in ("--request", "--receipt", "--json"):
            if sum(v == flag or v.startswith(flag + "=") for v in argv) > 1:
                raise WorkflowCLIError("input_invalid")
        args = parser.parse_args(list(argv))
        if getattr(args, "operation", None) is None:
            parser.print_help()
            return 0
        view = _execute(args, service, clock or (lambda: int(time.time())))
        exit_code = WORKFLOW_CLI_EXIT_CODES[view.status]
        envelope = {
            "schema_version": WORKFLOW_CLI_SCHEMA_VERSION,
            "ok": view.status
            in (
                WorkflowCLIStatus.READY,
                WorkflowCLIStatus.COMPLETED,
                WorkflowCLIStatus.CANCELLED,
            ),
            "command": command,
            "data": view.to_dict(),
        }
    except SystemExit as exc:
        return 0 if exc.code == 0 else 2
    except Exception as exc:
        error = (
            WorkflowCLIError(exc.code, exc.exit_code)
            if type(exc) is WorkflowCLIError
            else WorkflowCLIError("input_invalid")
        )
        exit_code = error.exit_code
        envelope = {
            "schema_version": WORKFLOW_CLI_SCHEMA_VERSION,
            "ok": False,
            "command": command,
            "error": {
                "code": error.code,
                "message": "Governed workflow command refused.",
            },
        }
    sys.stdout.write(json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n")
    return exit_code


def add_governed_workflow_typer_command(
    agents_app: Any,
    typer_module: Any,
    *,
    service: WorkflowCLIGovernanceService | None = None,
) -> None:
    """Forward Typer workflow arguments through the same guarded console boundary."""

    def workflow(context):
        code = run_governed_workflow_cli(["workflow", *context.args], service=service)
        if code:
            raise typer_module.Exit(code=code)

    workflow.__annotations__ = {"context": typer_module.Context}
    agents_app.command(
        "workflow",
        help="Governed previews, review requests and explicit recovery.",
        add_help_option=False,
        context_settings={"ignore_unknown_options": True, "allow_extra_args": True},
    )(workflow)


__all__ = [
    "WORKFLOW_CLI_SCHEMA_VERSION",
    "WORKFLOW_CLI_REQUEST_SCHEMA_VERSION",
    "MAX_WORKFLOW_CLI_INPUT_BYTES",
    "WORKFLOW_CLI_EXIT_CODES",
    "WorkflowCLIStatus",
    "WorkflowCLIError",
    "WorkflowCLIRequest",
    "WorkflowCLIView",
    "WorkflowCLIReceiptVerification",
    "WorkflowCLIGovernanceService",
    "workflow_cli_receipt_digest",
    "add_governed_workflow_commands",
    "run_governed_workflow_cli",
    "add_governed_workflow_typer_command",
]
