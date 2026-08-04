"""Offline ``openmed registry`` command surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from openmed.core.model_registry import MANIFEST_PATH
from openmed.core.registry_service import (
    REGISTRY_STATE_PATH,
    RegistryError,
    RegistryService,
)

from ._output import EXIT_ERROR, CliError, emit


def add_registry_command(subparsers: argparse._SubParsersAction) -> None:
    """Register model-registry pointer, lineage, and rollback commands."""

    parser = subparsers.add_parser(
        "registry",
        help="Inspect and mutate the offline model-registry pointer state.",
    )
    commands = parser.add_subparsers(dest="registry_command")

    list_parser = commands.add_parser("list", help="List named registry pointers.")
    _add_local_state_arguments(list_parser)
    list_parser.add_argument(
        "--family", default=None, help="Limit output to one family."
    )
    list_parser.set_defaults(handler=_handle_list)

    lineage_parser = commands.add_parser(
        "lineage",
        help="Show supersession and rollback lineage for one family.",
    )
    _add_local_state_arguments(lineage_parser)
    lineage_parser.add_argument("family", help="Manifest model family.")
    lineage_parser.set_defaults(handler=_handle_lineage)

    rollback_parser = commands.add_parser(
        "rollback",
        help="Repoint latest to last_green using a releasable gate report.",
    )
    _add_local_state_arguments(rollback_parser)
    rollback_parser.add_argument("family", help="Manifest model family.")
    rollback_parser.add_argument(
        "--gate-report",
        type=Path,
        required=True,
        help="Serialized RELEASABLE gate report for the last-green checkpoint.",
    )
    rollback_parser.set_defaults(handler=_handle_rollback)


def _add_local_state_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Local canonical model manifest.",
    )
    parser.add_argument(
        "--state",
        type=Path,
        default=REGISTRY_STATE_PATH,
        help="Committed registry-state JSON document.",
    )


def _service(args: argparse.Namespace) -> RegistryService:
    try:
        return RegistryService(manifest_path=args.manifest, state_path=args.state)
    except RegistryError as exc:
        raise CliError(
            f"Failed to load registry state: {exc}",
            code="registry_load_failed",
            exit_code=EXIT_ERROR,
        ) from exc


def _handle_list(args: argparse.Namespace) -> int:
    service = _service(args)
    pointers = service.pointers(args.family)
    if args.family is not None:
        payload = {"family": args.family, "pointers": pointers}
        human = _format_pointer_set(args.family, pointers)
    else:
        payload = {"families": pointers}
        human = "\n".join(
            _format_pointer_set(family, family_pointers)
            for family, family_pointers in pointers.items()
        )
        if not human:
            human = "No registry pointers configured."
    return emit(args, payload, human=human)


def _handle_lineage(args: argparse.Namespace) -> int:
    service = _service(args)
    lineage = service.lineage(args.family)
    human = (
        "\n".join(
            f"{edge['recorded_at']} {edge['relation']}: "
            f"{edge['from']} -> {edge['to']} ({edge['reason']})"
            for edge in lineage
        )
        or f"No lineage recorded for {args.family}."
    )
    return emit(
        args,
        {"family": args.family, "lineage": lineage},
        human=human,
    )


def _handle_rollback(args: argparse.Namespace) -> int:
    from openmed.eval.release_gates import GateReport

    service = _service(args)
    try:
        gate_report = GateReport.from_dict(
            json.loads(args.gate_report.read_text(encoding="utf-8"))
        )
        service.rollback(args.family, gate_report=gate_report)
    except (OSError, ValueError, RegistryError) as exc:
        raise CliError(
            f"Registry rollback failed: {exc}",
            code="registry_rollback_failed",
            exit_code=EXIT_ERROR,
        ) from exc
    pointers = service.pointers(args.family)
    lineage = service.lineage(args.family)
    return emit(
        args,
        {
            "family": args.family,
            "pointers": pointers,
            "lineage_edge": lineage[-1] if lineage else None,
        },
        human=(f"Rolled back {args.family} latest to {pointers['latest']}."),
    )


def _format_pointer_set(family: str, pointers: dict[str, object]) -> str:
    values = ", ".join(f"{name}={target or '-'}" for name, target in pointers.items())
    return f"{family}: {values}"
