#!/usr/bin/env python3
"""Inspect, mutate, and regenerate the committed offline model registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from openmed.core.manifest_diff import (
    DEFAULT_README_PATH,
    DEFAULT_REGISTRY_CARD_DIR,
    regenerate_registry_surfaces,
    registry_surface_errors,
)
from openmed.core.model_registry import MANIFEST_PATH
from openmed.core.registry_service import (
    REGISTRY_STATE_PATH,
    RegistryError,
    RegistryService,
    migrate_registry_state_file,
)
from openmed.eval.release_gates import GateReport


def build_parser() -> argparse.ArgumentParser:
    """Build the offline registry-control argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--state", type=Path, default=REGISTRY_STATE_PATH)
    parser.add_argument("--readme", type=Path, default=DEFAULT_README_PATH)
    parser.add_argument("--card-dir", type=Path, default=DEFAULT_REGISTRY_CARD_DIR)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List named pointers.")
    list_parser.add_argument("--slot", default=None)

    lineage_parser = commands.add_parser("lineage", help="Show slot lineage.")
    lineage_parser.add_argument("slot")

    rollback_parser = commands.add_parser(
        "rollback", help="Repoint latest to last_green."
    )
    rollback_parser.add_argument("slot")
    rollback_parser.add_argument("--gate-report", type=Path, required=True)

    commands.add_parser(
        "regenerate", help="Regenerate README counts and registry model cards."
    )
    commands.add_parser("check", help="Check committed registry-derived surfaces.")
    commands.add_parser(
        "migrate",
        help="Migrate a committed v1 registry state file to the v2 slot schema.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run an offline registry-control command."""

    args = build_parser().parse_args(argv)
    try:
        if args.command == "migrate":
            migrated = migrate_registry_state_file(
                state_path=args.state,
                manifest_path=args.manifest,
            )
            print(
                json.dumps(
                    {"slots": sorted(migrated["slots"])},
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "regenerate":
            snapshot = regenerate_registry_surfaces(
                manifest_path=args.manifest,
                state_path=args.state,
                readme_path=args.readme,
                card_dir=args.card_dir,
            )
            print(
                json.dumps(
                    {
                        "cards": len(snapshot.cards),
                        "registry_keys": len(snapshot.registry_keys),
                        "supported_languages": len(snapshot.supported_languages),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "check":
            errors = registry_surface_errors(
                manifest_path=args.manifest,
                state_path=args.state,
                readme_path=args.readme,
                card_dir=args.card_dir,
            )
            if errors:
                for error in errors:
                    print(error, file=sys.stderr)
                return 1
            print("Registry surfaces are coherent.")
            return 0

        service = RegistryService(
            manifest_path=args.manifest,
            state_path=args.state,
        )
        if args.command == "list":
            print(json.dumps(service.pointers(args.slot), indent=2, sort_keys=True))
            return 0
        if args.command == "lineage":
            print(json.dumps(service.lineage(args.slot), indent=2, sort_keys=True))
            return 0
        if args.command == "rollback":
            report = GateReport.from_dict(
                json.loads(args.gate_report.read_text(encoding="utf-8"))
            )
            service.rollback(args.slot, gate_report=report)
            print(json.dumps(service.pointers(args.slot), indent=2, sort_keys=True))
            return 0
    except (OSError, ValueError, RegistryError) as exc:
        print(f"Registry command failed: {exc}", file=sys.stderr)
        return 1
    raise AssertionError(f"unhandled registry command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
