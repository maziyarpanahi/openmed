"""Offline ``openmed migrations`` commands for durable service schemas.

* ``registry`` prints the durable-schema registry: every component, its
  versions, compatibility window, trigger, posture, recovery point, and
  rollback class.
* ``preflight`` observes live state read-only and prints the dry-run
  migration plan with its preflight checks.  Nothing is migrated or written.
* ``rehearse`` runs the offline rehearsal against synthetic fixtures in a
  temporary workspace and prints the compatibility matrix and release-gate
  verdict, optionally signed.

Output is value-free: versions, categories, counts, digests, states, and
controlled reason codes.  Paths, file names, and key material never appear.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ._output import EXIT_USAGE, CliError, emit
from .recovery import _add_source_arguments, _key_provider, _read_key_file, _sources

if TYPE_CHECKING:
    from openmed.service.migration_rehearsal import RehearsalReport
    from openmed.service.schema_registry import MigrationPlan


def add_migrations_command(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``openmed migrations`` command group."""

    parser = subparsers.add_parser(
        "migrations",
        help="Inspect, preflight, and rehearse durable service schema migrations.",
    )
    commands = parser.add_subparsers(dest="migrations_command")

    registry = commands.add_parser(
        "registry",
        help="Print the durable-schema registry and compatibility matrix.",
    )
    registry.set_defaults(handler=_handle_registry)

    preflight = commands.add_parser(
        "preflight",
        help="Dry-run the migration plan and preflight checks without writing.",
    )
    _add_source_arguments(preflight)
    preflight.add_argument(
        "--recovery-point",
        type=Path,
        default=None,
        help="Recovery point proposed as the rollback for restore-required steps.",
    )
    preflight.add_argument(
        "--key",
        action="append",
        default=[],
        metavar="KEY_ID=FILE",
        help="Backup key that verifies --recovery-point; repeat for older keys.",
    )
    preflight.add_argument(
        "--journey-key",
        action="append",
        default=[],
        metavar="KEY_ID=FILE",
        help="Journey store key whose availability the plan reports.",
    )
    preflight.set_defaults(handler=_handle_preflight)

    rehearse = commands.add_parser(
        "rehearse",
        help="Rehearse every migration path offline against synthetic fixtures.",
    )
    rehearse.add_argument(
        "--component",
        action="append",
        default=[],
        help="Rehearse only this component; repeatable (the gate is partial).",
    )
    rehearse.add_argument(
        "--signing-key",
        default=None,
        metavar="KEY_ID=FILE",
        help="Sign the report with HMAC-SHA256 (32 raw bytes or 64 hex chars).",
    )
    rehearse.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Also write the report JSON to this new file.",
    )
    rehearse.set_defaults(handler=_handle_rehearse)


def _handle_registry(args: argparse.Namespace) -> int:
    from openmed.service.schema_registry import registry_document

    document = registry_document()
    return emit(args, document, human=_format_registry(document))


def _handle_preflight(args: argparse.Namespace) -> int:
    from openmed.service.recovery import BACKUP_KEY_PURPOSE
    from openmed.service.schema_registry import plan_service_migrations
    from openmed.structured.store import JOURNEY_KEY_PURPOSE

    if args.key and args.recovery_point is None:
        raise CliError(
            "--key verifies a recovery point; pass --recovery-point too.",
            code="recovery_point_required",
            exit_code=EXIT_USAGE,
        )
    plan = plan_service_migrations(
        _sources(args),
        state_root=args.state_root,
        recovery_point=args.recovery_point,
        backup_key_provider=(
            _key_provider(list(args.key), purpose=BACKUP_KEY_PURPOSE)
            if args.key
            else None
        ),
        journey_key_provider=(
            _key_provider(list(args.journey_key), purpose=JOURNEY_KEY_PURPOSE)
            if args.journey_key
            else None
        ),
    )
    return emit(
        args,
        plan.to_dict(),
        human=_format_plan(plan),
        outcome="refused" if plan.admission == "blocked" else "completed",
    )


def _handle_rehearse(args: argparse.Namespace) -> int:
    from openmed.service.migration_rehearsal import rehearse_service_migrations
    from openmed.service.schema_registry import durable_schemas

    known = {schema.component for schema in durable_schemas()}
    unknown = set(args.component) - known
    if unknown:
        raise CliError(
            "An unknown component was requested.",
            code="component_unknown",
            exit_code=EXIT_USAGE,
        )
    signing: tuple[str, bytes] | None = None
    if args.signing_key is not None:
        key_id, separator, location = str(args.signing_key).partition("=")
        if not separator or not key_id or not location:
            raise CliError(
                "The signing key must be given as KEY_ID=FILE.",
                code="key_spec_invalid",
                exit_code=EXIT_USAGE,
            )
        signing = (key_id, _read_key_file(Path(location)))
    if args.output is not None and os.path.lexists(args.output):
        raise CliError(
            "The report output file already exists.",
            code="output_exists",
            exit_code=EXIT_USAGE,
        )
    report = rehearse_service_migrations(components=args.component or None)
    if signing is not None:
        try:
            report = report.sign(signing[1], key_id=signing[0])
        except ValueError:
            raise CliError(
                "The signing key specification is invalid.",
                code="key_spec_invalid",
                exit_code=EXIT_USAGE,
            ) from None
    if args.output is not None:
        _write_report(args.output, report)
    return emit(
        args,
        report.to_dict(),
        human=report.summary(),
        outcome="refused" if report.gate_state == "blocked" else "completed",
    )


def _write_report(path: Path, report: RehearsalReport) -> None:
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(report.to_json() + "\n")
    except OSError:
        raise CliError(
            "The report output file could not be written.",
            code="output_unavailable",
        ) from None


def _format_registry(document: dict[str, Any]) -> str:
    lines = [
        f"durable schema registry: release {document['release']}, "
        f"previous release {document['previous_release']}"
    ]
    for item in document["components"]:
        if item["presence"] != "present":
            lines.append(
                f"  {item['component']}: not_present (issue #{item['tracking_issue']})"
            )
            continue
        window = item["compatibility_window"]
        previous = window["previous_binary"]
        if window["previous_binary_code"]:
            previous += f" ({window['previous_binary_code']})"
        lines.append(
            f"  {item['component']}: v{item['current_version']}, "
            f"{item['rollback_class']}, {item['trigger']}, {item['posture']}, "
            f"recovery {item['recovery_point']}, previous binary {previous}"
        )
    findings = document["findings"]
    lines.append("findings: " + (", ".join(findings) if findings else "none"))
    return "\n".join(lines)


def _format_plan(plan: MigrationPlan) -> str:
    lines = [
        f"migration preflight: {plan.admission} (release {plan.release}, "
        "dry run, nothing written)"
    ]
    for action in plan.actions:
        if action.action == "none":
            continue
        line = f"  {action.order + 1}. {action.component}: {action.action}"
        if action.source_version is not None:
            line += f" {action.source_version}"
            if action.target_version is not None:
                line += f" -> {action.target_version}"
        for value in (action.rollback_class, action.recovery_point, action.posture):
            if value:
                line += f", {value}"
        if action.code:
            line += f" ({action.code})"
        lines.append(line)
    for check in plan.checks:
        state = "passed" if check.passed else "failed"
        lines.append(f"  check {check.name}: {state} ({check.code})")
    return "\n".join(lines)
