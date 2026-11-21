"""Inspect the packaged intended-use registry from the command line."""

import argparse

from openmed.cli._output import EXIT_USAGE, CliError, add_json_flag, emit
from openmed.core.intended_use import (
    IntendedUse,
    RegulatoryTier,
    load_intended_use_registry,
)

_OBLIGATION_TEXT = {
    "notice_required": "Display the notice.",
    "acknowledgement_required": "Record acknowledgement.",
    "clinician_mode_required": "Enable clinician mode.",
    "crisis_routing_required": "Provide crisis routing.",
}


def add_intended_use_command(subparsers) -> None:
    """Register read-only capability list and detail commands."""
    group = subparsers.add_parser(
        "intended-use", help="Inspect capability intended uses."
    )
    commands = group.add_subparsers(dest="intended_use_command", required=True)
    listing = commands.add_parser(
        "list", help="List capabilities and required safeguards."
    )
    listing.add_argument("--tier", choices=[tier.value for tier in RegulatoryTier])
    add_json_flag(listing)
    listing.set_defaults(handler=_list, command_path="intended-use list")
    detail = commands.add_parser("show", help="Show a capability's full declaration.")
    detail.add_argument("capability")
    add_json_flag(detail)
    detail.set_defaults(handler=_show, command_path="intended-use show")


def _row(code: str, declaration: IntendedUse) -> dict:
    return {
        "code": code,
        "tier": declaration.regulatory_tier.value,
        "obligations": sorted(declaration.regulatory_tier.obligations),
    }


def _headline(row: dict) -> str:
    safeguards = " ".join(_OBLIGATION_TEXT[code] for code in row["obligations"])
    return f"{row['code']} ({row['tier']})  {safeguards or 'No tier obligations.'}"


def _list(args: argparse.Namespace) -> int:
    rows = [
        _row(code, declaration)
        for code, declaration in sorted(
            load_intended_use_registry().declarations.items()
        )
        if args.tier is None or args.tier == declaration.regulatory_tier.value
    ]
    return emit(
        args, {"capabilities": rows}, human="\n".join(_headline(row) for row in rows)
    )


def _show(args: argparse.Namespace) -> int:
    registry = load_intended_use_registry()
    code = registry.entry_points.get(args.capability, args.capability)
    if code not in registry.declarations:
        raise CliError(
            "Unknown capability.", code="unknown_capability", exit_code=EXIT_USAGE
        )
    declaration = registry.declarations[code]
    row = _row(code, declaration)
    human = "\n".join(
        (
            _headline(row),
            f"Purpose: {declaration.purpose}",
            f"Intended users: {', '.join(declaration.intended_users)}",
            f"Excluded uses: {', '.join(declaration.excluded_uses)}",
            f"Input modalities: {', '.join(declaration.input_modalities)}",
            f"Languages: {', '.join(declaration.languages)}",
            f"Oversight: {declaration.human_oversight}",
            "Autonomous clinical action: prohibited",
        )
    )
    return emit(args, {**row, "declaration": declaration.to_dict()}, human=human)
