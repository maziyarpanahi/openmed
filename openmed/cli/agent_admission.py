"""Content-free operator commands for durable effect admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openmed.agent.admission import (
    AdmissionError,
    AdmissionRole,
    AdmissionStatus,
    EffectAdmissionController,
    SQLiteAdmissionStore,
)
from openmed.agent.identifiers import GovernanceIdError, WorkflowId


def run_admission_command(
    operation: str,
    *,
    state: str | None,
    anchor: str | None,
    key_file: str | None,
    scope: str = "global",
    role: str = "operator",
    initialize: bool = False,
) -> AdmissionStatus:
    """Execute one local operator command using shared CLI validation.

    Args:
        operation: Status, stop or resume command code.
        state: Explicit local ledger path.
        anchor: Independently protected high-water anchor path.
        key_file: Protected operator signing-key file path.
        scope: Global or canonical workflow identifier.
        role: Authorized control-plane role code.
        initialize: Explicit provisioning request for a new store.

    Returns:
        Content-free status after the requested operation.

    Raises:
        AdmissionError: Controlled code for invalid metadata or untrusted state.
    """
    try:
        if operation not in {"status", "stop", "resume"}:
            raise AdmissionError("invalid_control_metadata")
        workflow = None if scope == "global" else WorkflowId.parse(scope)
        operator_role = AdmissionRole(role)
        if (
            state is None
            and anchor is None
            and key_file is None
            and operation == "status"
        ):
            controller = EffectAdmissionController()
        else:
            if state is None or anchor is None or key_file is None:
                raise AdmissionError("not_configured")
            try:
                with Path(key_file).open("rb") as stream:
                    key = stream.read(4097)
            except OSError:
                raise AdmissionError("unreadable_key") from None
            if len(key) > 4096:
                raise AdmissionError("invalid_key")
            store = SQLiteAdmissionStore(Path(state), Path(anchor), key)
            if initialize:
                store.initialize(role=operator_role)
            controller = EffectAdmissionController(store)
        if operation == "stop":
            controller.stop(workflow_id=workflow, role=operator_role)
        elif operation == "resume":
            controller.enable(workflow_id=workflow, role=operator_role)
        return controller.status(workflow)
    except AdmissionError:
        raise
    except (GovernanceIdError, ValueError, TypeError, OSError, RuntimeError):
        raise AdmissionError("invalid_control_metadata") from None


def add_argparse_admission_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register admission commands on the production console-script parser."""
    agents = subparsers.add_parser("agents", help="Local-agent governance commands.")
    commands = agents.add_subparsers(dest="admission_operation")
    _add_admission_subcommands(commands)


def _add_admission_subcommands(commands: argparse._SubParsersAction) -> None:
    """Register admission handlers on a shared agents command group."""
    for operation, help_text in (
        ("status", "Show content-free effect admission status."),
        ("stop", "Stop effects at their next boundary."),
        ("resume", "Record a fresh explicit effect enable."),
    ):
        command = commands.add_parser(operation, help=help_text)
        command.add_argument("--state", help="Local admission ledger.")
        command.add_argument("--anchor", help="Independent high-water anchor.")
        command.add_argument("--key-file", help="Protected signing key file.")
        command.add_argument(
            "--scope", default="global", help="Global or canonical workflow identifier."
        )
        command.set_defaults(
            role="operator",
            initialize=False,
            admission_operation=operation,
            handler=_handle_admission,
        )
        if operation != "status":
            command.add_argument(
                "--role",
                default="incident_commander" if operation == "stop" else "operator",
                help="Authorized operator role code.",
            )
        if operation == "resume":
            command.add_argument(
                "--initialize",
                action="store_true",
                help="Explicitly provision a new disabled store before enabling; refuse existing files.",
            )


def _handle_admission(args: argparse.Namespace) -> int:
    from openmed.cli._output import CliError, emit

    try:
        status = run_admission_command(
            args.admission_operation,
            state=args.state,
            anchor=args.anchor,
            key_file=args.key_file,
            scope=args.scope,
            role=args.role,
            initialize=args.initialize,
        )
        if status.reason_code == "untrusted_state":
            raise AdmissionError("untrusted_state")
    except AdmissionError as exc:
        raise CliError(str(exc), code=str(exc)) from None
    return emit(
        args, status.to_dict(), human=json.dumps(status.to_dict(), sort_keys=True)
    )


def add_admission_commands(agents_app: Any, typer_module: Any) -> None:
    """Register stop, explicit resume/enable, and content-free status commands."""

    def run(
        operation: str,
        state: str | None,
        anchor: str | None,
        key_file: str | None,
        scope: str,
        role: str,
        initialize: bool = False,
    ) -> None:
        try:
            status = run_admission_command(
                operation,
                state=state,
                anchor=anchor,
                key_file=key_file,
                scope=scope,
                role=role,
                initialize=initialize,
            )
            typer_module.echo(json.dumps(status.to_dict(), sort_keys=True))
            if status.reason_code == "untrusted_state":
                raise typer_module.Exit(code=1)
        except AdmissionError as exc:
            typer_module.echo(json.dumps({"reason_code": str(exc)}), err=True)
            raise typer_module.Exit(code=1) from None

    @agents_app.command("status")
    def status(
        state: str | None = typer_module.Option(None, help="Local admission ledger."),
        anchor: str | None = typer_module.Option(
            None, help="Independent high-water anchor."
        ),
        key_file: str | None = typer_module.Option(
            None, help="Protected signing key file."
        ),
        scope: str = typer_module.Option(
            "global", help="Global or canonical workflow identifier."
        ),
    ) -> None:
        """Show codes and digests only; no configuration means disabled."""
        run("status", state, anchor, key_file, scope, "operator")

    @agents_app.command("stop")
    def stop(
        state: str | None = typer_module.Option(None, help="Local admission ledger."),
        anchor: str | None = typer_module.Option(
            None, help="Independent high-water anchor."
        ),
        key_file: str | None = typer_module.Option(
            None, help="Protected signing key file."
        ),
        scope: str = typer_module.Option(
            "global", help="Global or canonical workflow identifier."
        ),
        role: str = typer_module.Option(
            "incident_commander", help="Authorized operator role code."
        ),
    ) -> None:
        """Stop effects at their next boundary; read-only work stays available."""
        run("stop", state, anchor, key_file, scope, role)

    @agents_app.command("resume")
    def resume(
        state: str | None = typer_module.Option(None, help="Local admission ledger."),
        anchor: str | None = typer_module.Option(
            None, help="Independent high-water anchor."
        ),
        key_file: str | None = typer_module.Option(
            None, help="Protected signing key file."
        ),
        scope: str = typer_module.Option(
            "global", help="Global or canonical workflow identifier."
        ),
        role: str = typer_module.Option(
            "operator", help="Authorized operator role code."
        ),
        initialize: bool = typer_module.Option(
            False,
            help="Explicitly provision a new disabled ledger before enabling. Existing files are never reset.",
        ),
    ) -> None:
        """Record a fresh enable; previous runs still need fresh authority checks."""
        run("resume", state, anchor, key_file, scope, role, initialize)
